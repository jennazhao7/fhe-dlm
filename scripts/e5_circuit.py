"""E5 deep circuit: one MDLM denoising pass where the SERVER also computes the
norms, the attention and the GELU under CKKS (HANDOFF §12). The client only
encrypts tokens, optionally refreshes the residual stream in place of a
bootstrap, and decrypts logits.

Each op can be given back to the client (decrypt, compute in the clear,
re-encrypt), which is how the circuit is brought up stage by stage against the
E1 student (scripts/e1_kd.py):

  --norm  fixed   c[mask_t] * g * (x - mean)       (E1 stage A)   | client: exact LayerNorm
  --attn  pow     (q.k/8 + b_h)^2 / z[h, mask_t]   (E1 stage B)   | client: rotary + softmax
  --gelu  poly    Chebyshev degree 31 on [-8, 8]   (E1 stage C)   | client: exact GELU
          rsqrt   exact LayerNorm on the server: var = mean(x^2) - mean(x)^2 by rotations, 1/sigma by a
                  Chebyshev start + Goldschmidt steps, x/sigma bootstrapped; needs --refresh boot|boot2 and
                  --var-file (public variance bounds per norm, from `calib`). No client inside the pass.
  --refresh client  the client re-encrypts x at every half-layer (no bootstrap; depth 13)
            boot    FIDESlib bootstrap (8.5 bits; depth 43, the 4-GPU build)
            boot2   two-pass bootstrap (16.4 bits, one more level)

Layout (single ciphertext, L = T = 64 tokens): slot = token*D + dim, D = 1024,
real dims 0..767. The residual stream is held as x/S; everything else is in
true units. Per layer:
  u    = x (.) c1[mask_t] S s                         1 full-slot ptm (token-wise constant)
  qkv  = (Wqkv diag(g (1+scale)) C / s) u + Wqkv shift     3 blocks; C = I - 11^T/768 centres, so the
                                                      norm costs no rotation
  q',k'= rotary (2 rot + 3 ptm each; 1/8 folded into q')
  for r in 0..63:  K_r = rot(k', -rD), V_r = rot(v, -rD)      token t sees key t - r
       s = headsum(q' K_r) (6 rot), w = replicate(((s + b)^2) (.) 1/z) (6 rot), acc += w V_r
  x   += (gate Wo / S) acc
  u    = x (.) c2[mask_t] S s;  p = (W1 ... / s) u + b;  g = cheb(p);  x += (gate W2 / S) g + gate b2 / S
Backends: NumpyBE executes the same op list exactly (the reference, also for
garbage slots), VmBE drives fides_vm, PairBE runs both and compares registers
at the stage boundaries (--debug).

  check  numpy circuit vs the torch model on random masked inputs (CPU, no FHE):
         python scripts/e5_circuit.py check --ckpt ckpt/e1_pilot3/stageC.pt --stage C
  run    encrypted generation / single passes on the GPU:
         python scripts/e5_circuit.py run --bin gpu/fides_e0/build/fides_vm --ckpt ... --stage C \
             --refresh client --steps 16 --out results/e5/deep_C.json
"""
import argparse, collections, json, math, os, sys, tempfile, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e5_pass import VM, start_state, sample_from, sample_report   # noqa: E402

D, DM, HD, NH, VOC = 1024, 768, 64, 12, 50258
# 7 rotation keys: 1, 32, -D for the matvec; window sums and replication are two-level sums over
# (1, 8) or (1, 32) and their negatives. Job 1747455 loaded 21 keys (+-2^k) next to the bootstrap
# keys and ran out of memory on the third bootstrap (22.6 of 24 GB per card).
def rot_keys(n1):
    return sorted({1, 8, n1, HD // 2, -1, -8, -n1, -(HD // 2), -D})      # +-32: rotary
UN = 64.0        # x/sigma is bootstrapped as u / UN (uncentred: |x|/sigma reaches 35 at layer 0)


def gelu_np(z):
    return 0.5 * z * (1 + np.tanh(math.sqrt(2 / math.pi) * (z + 0.044715 * z ** 3)))


# ---------------------------------------------------------------- backends
class NumpyBE:
    def __init__(self, slots):
        self.slots, self.T, self.r = slots, slots // D, {}

    def _full(self, v):
        v = np.asarray(v, np.float64)
        return np.tile(v, self.slots // len(v))

    def enc(self, r, v, level=0): self.r[r] = np.array(v, np.float64)
    def dec(self, r): return self.r[r].copy()
    def add(self, r, a, b): self.r[r] = self.r[a] + self.r[b]
    def sub(self, r, a, b): self.r[r] = self.r[a] - self.r[b]
    def mul(self, r, a, b): self.r[r] = self.r[a] * self.r[b]
    def sq(self, r, a): self.r[r] = self.r[a] ** 2
    def rot(self, r, a, k): self.r[r] = np.roll(self.r[a], -k)
    def mulpt(self, r, a, v): self.r[r] = self.r[a] * self._full(v)
    def tolevel(self, r, a, level): self.r[r] = self.r[a].copy()
    def mulc(self, r, a, c): self.r[r] = self.r[a] * c
    def mov(self, r, a): self.r[r] = self.r[a].copy()
    def boot(self, r, a): self.r[r] = self.r[a].copy()
    def info(self, r): return {"level": 0, "deg": 2}
    def check(self, tag, r): pass

    def free(self, *rs):
        for x in rs:
            self.r.pop(x, None)

    def matvec(self, name, Wb, bias, ins, outs):
        X = [self.r[i].reshape(self.T, D) for i in ins]
        for o, r in enumerate(outs):
            y = sum(X[i] @ Wb[o][i].astype(np.float64).T for i in range(len(ins)))
            if bias is not None:
                y = y + bias[o][None, :]
            self.r[r] = y.reshape(-1)

    def cheb(self, r, a, lo, hi, c):
        t = (2 * self.r[a] - (lo + hi)) / (hi - lo)
        self.r[r] = np.polynomial.chebyshev.chebval(t, np.asarray(c, np.float64))


class VmBE:
    def __init__(self, vm):
        self.vm, self.slots, self.T = vm, vm.slots, vm.T
        self.files, self.k = {}, 0

    def _vec(self, v):
        self.k = (self.k + 1) % 8
        path = os.path.join(self.vm.wd, f"pt{self.k}_{len(v)}.f64")
        np.ascontiguousarray(v, "<f8").tofile(path)
        return path

    def enc(self, r, v, level=0): self.vm.enc(r, v, level)
    def dec(self, r): return self.vm.dec(r)
    def add(self, r, a, b): self.vm.cmd(f"add {r} {a} {b}")
    def sub(self, r, a, b): self.vm.cmd(f"sub {r} {a} {b}")
    def mul(self, r, a, b): self.vm.cmd(f"mul {r} {a} {b}")
    def sq(self, r, a): self.vm.cmd(f"sq {r} {a}")
    def rot(self, r, a, k): self.vm.cmd(f"rot {r} {a} {k}")
    def mulpt(self, r, a, v): self.vm.cmd(f"mulpt {r} {a} {self._vec(v)} {len(v)}")
    def tolevel(self, r, a, level): self.vm.cmd(f"tolevel {r} {a} {level}")
    def mulc(self, r, a, c): self.vm.cmd(f"mulc {r} {a} {c!r}")
    def mov(self, r, a): self.vm.cmd(f"mov {r} {a}")
    def boot(self, r, a): self.vm.cmd(f"boot {r} {a}")
    def info(self, r): return self.vm.cmd(f"info {r}")
    def free(self, *rs): self.vm.cmd("free " + " ".join(rs))
    def check(self, tag, r): pass

    def matvec(self, name, Wb, bias, ins, outs):
        if name not in self.files:
            wf = os.path.join(self.vm.wd, f"w_{name}.f32")
            np.ascontiguousarray(Wb, "<f4").tofile(wf)
            bf, kb = "-", 1.0
            if bias is not None:
                kb = float(max(1.0, np.abs(bias).max()))
                bf = os.path.join(self.vm.wd, f"b_{name}.f64")
                np.ascontiguousarray(np.asarray(bias, np.float64) / kb, "<f8").tofile(bf)
            self.files[name] = (wf, bf, kb)
        wf, bf, kb = self.files[name]
        self.vm.cmd(f"matvec {len(outs)} {len(ins)} {wf} {bf} {' '.join(ins)} {' '.join(outs)} kb={kb!r}")

    def cheb(self, r, a, lo, hi, c):
        c = np.array(c, np.float64)
        c[0] *= 2                      # OpenFHE's series is c0/2 + sum c_k T_k
        self.vm.cmd(f"cheb {r} {a} {lo} {hi} {self._vec(c)} {len(c)}")


class PairBE:
    """Both backends on every op; check() compares a register. enc() after a
    client step writes the same value to both, so errors do not accumulate in
    the reference across client rounds."""
    def __init__(self, vmbe, npbe, log):
        self.v, self.n, self.log = vmbe, npbe, log
        self.slots, self.T = vmbe.slots, vmbe.T

    def __getattr__(self, name):
        fv, fn = getattr(self.v, name), getattr(self.n, name)

        def both(*a, **k):
            fn(*a, **k)
            return fv(*a, **k)
        return both

    def dec(self, r): return self.v.dec(r)
    def info(self, r): return self.v.info(r)

    def check(self, tag, r):
        got, ref = self.v.dec(r), self.n.dec(r)
        i = self.v.info(r)
        rec = {"tag": tag, "max_abs_err": float(np.abs(got - ref).max()), "max_abs_ref": float(np.abs(ref).max()),
               "level": i["level"], "deg": i["deg"], "gpu_gb": self.v.vm.cmd("mem")["gpu_gb"]}
        self.log.append(rec)
        print("[chk]", json.dumps(rec), flush=True)


# ---------------------------------------------------------------- parameters
def extract(m, L, fixed_norm, pow_attn, poly_gelu):
    """MDLM / E1-student weights as float64 numpy, adaLN constants evaluated."""
    import torch
    import torch.nn.functional as F
    bb = m.backbone
    n = lambda t: t.detach().double().cpu().numpy()
    with torch.no_grad():
        c = F.silu(bb.sigma_map(torch.zeros(1, device=bb.vocab_embed.embedding.device)))
        cos, sin = bb.rotary_emb(torch.zeros(1, L, DM, device=c.device))
        P = {"E": n(bb.vocab_embed.embedding), "cos": n(cos[0, :, 0, 0, :HD // 2]), "sin": n(sin[0, :, 0, 0, :HD // 2]), "layers": []}
        for b in bb.blocks:
            mod = [n(t) for t in b.adaLN_modulation(c)[0].chunk(6)]
            lay = dict(zip(("shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"), mod))
            lay.update(g1=n(b.norm1.weight), g2=n(b.norm2.weight), Wqkv=n(b.attn_qkv.weight), Wo=n(b.attn_out.weight),
                       W1=n(b.mlp[0].weight), b1=n(b.mlp[0].bias), W2=n(b.mlp[2].weight), b2=n(b.mlp[2].bias))
            # Without a student (exact MDLM) the server-side ops get placeholder constants: the
            # model output is then meaningless, but encrypted-vs-numpy checks of the mechanics hold.
            stu = hasattr(b, "pow_attn")
            if fixed_norm:
                lay.update(c1=n(b.norm1.log_c.exp()) if stu else np.full(2, 0.05), c2=n(b.norm2.log_c.exp()) if stu else np.full(2, 0.05))
            if pow_attn:
                lay.update(ab=n(b.pow_attn.b) if stu else np.zeros(NH), az=n(b.pow_attn.log_z.exp()) if stu else np.full((NH, 2), 1e4 * L))
            if poly_gelu:
                lay.update(gc=n(b.mlp[1].c) if stu else np.polynomial.chebyshev.chebinterpolate(lambda t: gelu_np(8.0 * t), 31),
                           R=float(b.mlp[1].R) if stu else 8.0)
            P["layers"].append(lay)
        ol = bb.output_layer
        sh, sc = (n(t) for t in ol.adaLN_modulation(c)[0].chunk(2))
        P["final"] = dict(g=n(ol.norm_final.weight), shift=sh, scale=sc, W=n(ol.linear.weight), b=n(ol.linear.bias))
        if fixed_norm:
            P["final"]["c"] = n(ol.norm_final.log_c.exp()) if hasattr(ol.norm_final, "log_c") else np.full(2, 0.05)
    return P


def blocks_of(W, n_out, n_in, rows, cols):
    """(out, in) matrix -> [n_out][n_in][D][D] float32; block o holds `rows` output dims, block i `cols` inputs."""
    Wb = np.zeros((n_out, n_in, D, D), np.float32)
    for o in range(n_out):
        for i in range(n_in):
            blk = W[o * rows:(o + 1) * rows, i * cols:(i + 1) * cols]
            Wb[o, i, :blk.shape[0], :blk.shape[1]] = blk
    return Wb


def bias_of(b, n_out, rows):
    out = np.zeros((n_out, D))
    for o in range(n_out):
        seg = b[o * rows:(o + 1) * rows]
        out[o, :len(seg)] = seg
    return out


# ---------------------------------------------------------------- circuit
class Circuit:
    def __init__(self, be, P, cfg):
        self.be, self.P, self.cfg = be, P, cfg
        self.T, self.S = be.T, cfg.S      # self.S: current scale of the x register (x is held as x/S)
        self.cache = {}
        self.stat = collections.defaultdict(float)
        self.real = np.zeros(D); self.real[:DM] = 1.0
        self.head1 = np.zeros(D); self.head1[np.arange(NH) * HD] = 1.0     # head-first dims

    # --- helpers
    def pack(self, X):
        out = np.zeros((self.T, D)); out[:, :X.shape[1]] = X
        return out.reshape(-1)

    def unpack(self, v, n=DM):
        return np.asarray(v).reshape(self.T, D)[:, :n]

    def tokdim(self, per_tok, dims):
        return (np.asarray(per_tok, np.float64)[:, None] * dims[None, :]).reshape(-1)

    def winsum(self, r, n1, s1, n2, s2):
        """r <- sum_{i < n1, j < n2} rot(r, i s1 + j s2), n1 + n2 - 2 rotations with keys s1, s2."""
        be = self.be
        for n, st in ((n1, s1), (n2, s2)):
            be.mov("ws", r)
            for _ in range(n - 1):
                be.rot("ws", "ws", st); be.add(r, r, "ws")
        be.free("ws")

    def weights(self, key, fn):
        if key not in self.cache:
            self.cache[key] = fn()
        return self.cache[key]

    def boot_reg(self, r):
        """Bootstrap register r (|values| < 1). boot2 = two-pass (HANDOFF §11): b1 = Boot(r),
        b2 = Boot((r - b1) 2^p), r = b1 + b2 2^-p."""
        be, cfg = self.be, self.cfg
        if cfg.refresh not in ("boot", "boot2"):
            return
        t0 = time.perf_counter()
        be.tolevel(r, r, cfg.depth - 1)
        if cfg.refresh == "boot":
            be.boot(r, r)
        else:
            be.boot("b1", r)
            be.sub("bd", r, "b1")
            for _ in range(cfg.boot_bits):
                be.add("bd", "bd", "bd")
            be.boot("bd", "bd")
            be.mulc("bd", "bd", 2.0 ** -cfg.boot_bits)
            be.add(r, "b1", "bd")
            be.free("b1", "bd")
        self.stat["bootstraps"] += 1 if cfg.refresh == "boot" else 2
        self.stat["refresh_s"] += time.perf_counter() - t0

    def var_range(self, tag):
        lo, hi = self.cfg.var[tag]
        return lo / self.cfg.var_margin, hi * self.cfg.var_margin

    def scale_for(self, tag):
        """Public scale of x at this norm: the residual stream grows ~400x from layer 0 to the head
        (var 0.06 -> 1.2e4), and bootstrap error is absolute in x/S, so one global S would drown
        the early layers. From calib: a power of two >= s_margin * max|x| there."""
        xm = (self.cfg.xmax or {}).get(tag)
        return self.cfg.S if xm is None else 2.0 ** math.ceil(math.log2(self.cfg.s_margin * xm))

    def rescale_x(self, tag):
        St = self.scale_for(tag)
        if St != self.S:
            self.be.mulc("x", "x", self.S / St)
            self.S = St

    def refresh(self, why):
        be, cfg = self.be, self.cfg
        t0 = time.perf_counter()
        if cfg.refresh in ("boot", "boot2"):
            self.boot_reg("x")
            be.check(f"refresh {why}", "x")
            return
        if cfg.refresh == "client":
            v = be.dec("x")
            self.stat["x_max"] = max(self.stat["x_max"], float(np.abs(v).max()) * self.S)
            be.enc("x", v, 0)
            self.stat["client_refresh"] += 1
        self.stat["refresh_s"] += time.perf_counter() - t0
        be.check(f"refresh {why}", "x")

    def levels_left(self):
        i = self.be.info("x")
        return self.cfg.depth - i["level"] - (i["deg"] - 1)

    def before_half(self, need, why):
        cfg = self.cfg
        if cfg.norm == "rsqrt":
            return                                   # the norm bootstraps x itself
        if cfg.refresh == "client":
            self.refresh(why)
        elif cfg.refresh in ("boot", "boot2") and self.levels_left() < need + 2:
            self.refresh(why)

    def normed_input(self, tag, c, g_mod, W, m, level=0):
        """Register `u` and the folded matrix so that matvec(u) = W (norm(x) (.) g_mod).
        Returns W' (rows as W)."""
        be, cfg = self.be, self.cfg
        if cfg.norm == "rsqrt":
            def fold():
                A = W * g_mod[None, :]
                A = A - A.mean(1, keepdims=True)
                s = 2.0 ** math.ceil(math.log2(max(1.0, float(np.abs(A).max()))))      # power of two: undone by self-adds
                return A / s, s
            Wp, s = self.weights(tag, fold)
            vmin, vmax = self.var_range(tag)
            self.rescale_x(tag)
            S, wmin = self.S, vmin / vmax
            self.refresh("norm " + tag)                  # x fresh: the norm uses ~17 levels below x
            e0 = np.zeros(D); e0[0] = 1.0
            be.sq("n2", "x"); be.mov("n1", "x")
            n1 = cfg.n1
            self.winsum("n2", n1, 1, D // n1, n1)        # sums over the token's 1024 dims, valid at dim 0
            self.winsum("n1", n1, 1, D // n1, n1)
            be.mulpt("n2", "n2", e0 * (S * S / (DM * vmax)))
            be.mulpt("n1", "n1", e0 * (S / (DM * math.sqrt(vmax))))
            be.sq("n1", "n1")
            be.sub("w", "n2", "n1")                      # w = var / vmax at dim 0, 0 elsewhere
            be.check(tag + " var/vmax", "w")

            ch = self.weights(("rsqrt", round(math.log2(wmin), 4)), lambda: np.polynomial.chebyshev.chebinterpolate(
                lambda t: 0.5 / np.sqrt(wmin + (t + 1) * (1 - wmin) / 2), cfg.rsqrt_deg))
            be.cheb("h", "w", wmin, 1.0, ch)             # h ~ 1/(2 sqrt(w))
            be.mul("g", "w", "h"); be.add("g", "g", "g")  # g = 2 w h ~ sqrt(w): the invariant Goldschmidt needs
            for _ in range(cfg.gold_iters):              # Goldschmidt: r = 1/2 - g h; g += g r; h += h r
                be.mul("r", "g", "h")
                i = be.info("r")
                be.enc("one", np.ones(be.slots), i["level"])
                be.mulpt("one", "one", np.full(D, 0.5))
                be.sub("r", "one", "r")
                be.mul("t", "g", "r"); be.add("g", "g", "t")
                be.mul("t", "h", "r"); be.add("h", "h", "t")
            be.mulpt("h", "h", e0 * (2.0 * S / (UN * math.sqrt(vmax))))     # 1/(UN sigma) in x/S units, dim 0 only
            self.winsum("h", n1, -1, D // n1, -n1)
            be.check(tag + " S/(UN sigma)", "h")
            be.mul("u", "x", "h")                        # (x - mean)/sigma / UN once the matrix centres it
            be.free("n1", "n2", "w", "h", "g", "r", "t", "one")
            self.boot_reg("u")
            for _ in range(int(round(math.log2(UN * s)))):
                be.add("u", "u", "u")
            # Fewer towers for what follows (matvec memory and time scale with them): the attention
            # half needs 7 levels below u, the FFN half 9; keep 3 spare for the next rescale of x.
            lv = level or (cfg.depth - (14 if tag.endswith(".up") else 12))
            if cfg.refresh in ("boot", "boot2") and lv > 0:
                be.tolevel("u", "u", lv)
            be.free("n1", "n2", "w", "h", "g", "r", "t", "one")
            self.stat["server_norm"] += 1
            return Wp
        if cfg.norm == "fixed":
            def fold():
                A = W * g_mod[None, :]
                A = A - A.mean(1, keepdims=True)          # . C: centring folded into the matrix
                s = max(1.0, float(np.abs(A).max()))
                return A / s, s
            Wp, s = self.weights(tag, fold)
            be.mulpt("u", "x", self.tokdim(c[m] * self.S * s, self.real))
        else:
            def fold():
                A = W * g_mod[None, :]
                s = max(1.0, float(np.abs(A).max()))
                return A / s, s
            Wp, s = self.weights(tag, fold)
            x = self.unpack(be.dec("x")) * self.S
            self.stat["x_max"] = max(self.stat["x_max"], float(np.abs(x).max()))
            xc = x - x.mean(1, keepdims=True)
            nrm = xc / np.sqrt((xc ** 2).mean(1, keepdims=True) + 1e-5)
            be.enc("u", self.pack(nrm * s), level)
            self.stat["client_norm"] += 1
        return Wp

    # --- attention
    def rotary_pts(self):
        def build():
            cos, sin = self.P["cos"], self.P["sin"]                    # (T, 32)
            P0 = np.zeros((self.T, D)); P1 = np.zeros((self.T, D)); P2 = np.zeros((self.T, D))
            for h in range(NH):
                lo = slice(h * HD, h * HD + HD // 2); hi = slice(h * HD + HD // 2, (h + 1) * HD)
                P0[:, lo] = cos; P0[:, hi] = cos
                P1[:, lo] = -sin                                       # x1 c - x2 s, x2 = rot(+32)
                P2[:, hi] = sin                                        # x1 s + x2 c, x1 = rot(-32)
            return [p.reshape(-1) for p in (P0, P1, P2)]
        return self.weights("rotary", build)

    def rotary(self, dst, src, scale):
        be = self.be
        P0, P1, P2 = self.rotary_pts()
        be.rot("ra", src, HD // 2); be.rot("rb", src, -(HD // 2))
        be.mulpt(dst, src, P0 * scale)
        be.mulpt("ra", "ra", P1 * scale); be.add(dst, dst, "ra")
        be.mulpt("rb", "rb", P2 * scale); be.add(dst, dst, "rb")
        be.free("ra", "rb")

    def attention_server(self, lay, m):
        be, T = self.be, self.T
        self.rotary("qr", "q", 1.0 / math.sqrt(HD))
        self.rotary("kr", "k", 1.0)
        be.check("rotary q", "qr")
        ptb = np.zeros(D); ptb[np.arange(NH) * HD] = lay["ab"]
        M = np.zeros((T, D)); M[:, np.arange(NH) * HD] = 1.0 / lay["az"][:, m].T       # (T, NH)
        M = M.reshape(-1)
        for r in range(T):
            if r:
                be.rot("kr", "kr", -D); be.rot("v", "v", -D)
            be.mul("s", "qr", "kr")
            self.winsum("s", 8, 1, 8, 8)                 # head sums at head-first dims
            if r == 0:
                i = be.info("s")
                be.enc("one", np.ones(be.slots), i["level"])
                be.mulpt("B", "one", ptb)
                if i["deg"] != 2:
                    raise RuntimeError(f"scores at deg {i['deg']}, expected 2")
                be.check("scores r=0", "s")
            be.add("s", "s", "B")
            be.sq("s", "s")
            be.mulpt("s", "s", M)
            self.winsum("s", 8, -1, 8, -8)               # replicate over the head's dims
            if r == 0:
                be.check("weights r=0", "s")
            if r == 0:
                be.mul("o", "s", "v")
            else:
                be.mul("s", "s", "v"); be.add("o", "o", "s")
        be.free("s", "B", "one", "qr", "kr", "q", "k", "v")

    def attention_client(self, lay, m):
        be = self.be
        Q, K, V = (self.unpack(be.dec(r)) for r in ("q", "k", "v"))
        cos, sin = self.P["cos"], self.P["sin"]
        def rot(X):
            X = X.reshape(self.T, NH, HD)
            x1, x2 = X[..., :HD // 2], X[..., HD // 2:]
            return np.concatenate([x1 * cos[:, None] - x2 * sin[:, None], x1 * sin[:, None] + x2 * cos[:, None]], -1)
        q, k, v = rot(Q), rot(K), V.reshape(self.T, NH, HD)
        s = np.einsum("thd,uhd->htu", q, k) / math.sqrt(HD)
        if self.cfg.attn == "client-pow":                           # the student's attention, in the clear
            p = (s + lay["ab"][:, None, None]) ** 2 / lay["az"][:, m][:, :, None]
        else:
            p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
        o = np.einsum("htu,uhd->thd", p, v).reshape(self.T, DM)
        be.enc("o", self.pack(o), max(0, self.cfg.depth - 6))      # one matvec left to do: few towers
        be.free("q", "k", "v")
        self.stat["client_attn"] += 1

    def attn_half(self, i, lay, m):
        be = self.be
        Wp = self.normed_input(f"L{i}.qkv", lay.get("c1"), lay["g1"] * (1 + lay["scale_msa"]), lay["Wqkv"], m)
        Wb = self.weights(f"L{i}.qkv.b", lambda: (blocks_of(Wp, 3, 1, DM, DM), bias_of(lay["Wqkv"] @ lay["shift_msa"], 3, DM)))
        be.matvec(f"L{i}.qkv", Wb[0], Wb[1] if np.abs(Wb[1]).max() > 0 else None, ["u"], ["q", "k", "v"])
        be.free("u")
        be.check(f"L{i} q", "q")
        if self.cfg.attn == "pow":
            self.attention_server(lay, m)
        else:
            self.attention_client(lay, m)
        be.check(f"L{i} attn acc", "o")
        Wo = self.weights((f"L{i}.out", self.S), lambda: blocks_of(lay["gate_msa"][:, None] * lay["Wo"] / self.S, 1, 1, DM, DM))
        be.matvec(f"L{i}.out.S{self.S:g}", Wo, None, ["o"], ["y"])
        be.add("x", "x", "y")
        be.free("o", "y")
        be.check(f"L{i} x after attn", "x")

    def ffn_half(self, i, lay, m):
        be = self.be
        Wp = self.normed_input(f"L{i}.up", lay.get("c2"), lay["g2"] * (1 + lay["scale_mlp"]), lay["W1"], m)
        Wb = self.weights(f"L{i}.up.b", lambda: (blocks_of(Wp, 3, 1, D, DM), bias_of(lay["W1"] @ lay["shift_mlp"] + lay["b1"], 3, D)))
        ps = ["p0", "p1", "p2"]
        be.matvec(f"L{i}.up", Wb[0], Wb[1], ["u"], ps)
        be.free("u")
        be.check(f"L{i} gelu in", "p0")
        if self.cfg.gelu == "poly":
            for p in ps:
                be.cheb(p, p, -lay["R"], lay["R"], lay["gc"])
        else:
            for p in ps:
                u = be.dec(p)
                self.stat["gelu_in_max"] = max(self.stat["gelu_in_max"], float(np.abs(u).max()))
                be.enc(p, gelu_np(u), max(0, self.cfg.depth - 6))
            self.stat["client_gelu"] += 1
        be.check(f"L{i} gelu out", "p0")
        Wd = self.weights((f"L{i}.down", self.S), lambda: (blocks_of(lay["gate_mlp"][:, None] * lay["W2"] / self.S, 1, 3, DM, D),
                                                             bias_of(lay["gate_mlp"] * lay["b2"] / self.S, 1, DM)))
        be.matvec(f"L{i}.down.S{self.S:g}", Wd[0], Wd[1], ps, ["y"])
        be.add("x", "x", "y")
        be.free("y", *ps)
        be.check(f"L{i} x after ffn", "x")

    # --- pass
    def forward(self, ids, mask_id=50257):
        be, P, cfg, T = self.be, self.P, self.cfg, self.T
        ids = np.asarray(ids)
        assert len(ids) == T, f"one ciphertext: L must be {T}"
        m = (ids == mask_id).astype(int)
        n_layers = len(P["layers"]) if cfg.layers is None else cfg.layers
        self.S = self.scale_for("L0.qkv") if cfg.norm == "rsqrt" else cfg.S
        if cfg.embed == "server":
            nb = math.ceil(VOC / D)
            oh = np.zeros((T, nb * D)); oh[np.arange(T), ids] = 1.0
            lvl = max(0, cfg.depth - 3)                     # few towers: the 50 blocks are cheap, x is refreshed next
            ins = [f"e{k}" for k in range(nb)]
            for k, r in enumerate(ins):
                be.enc(r, oh[:, k * D:(k + 1) * D].reshape(-1), lvl)
            Eb = self.weights(("embed", self.S), lambda: blocks_of(P["E"].T / self.S, 1, nb, DM, D))
            be.matvec(f"embed.S{self.S:g}", Eb, None, ins, ["x"])
            be.free(*ins)
        else:
            be.enc("x", self.pack(P["E"][ids] / self.S), 0)
        be.check("embed", "x")
        for i in range(n_layers):
            lay = P["layers"][i]
            self.before_half(9, f"L{i} attn")
            self.attn_half(i, lay, m)
            self.before_half(11, f"L{i} ffn")
            self.ffn_half(i, lay, m)
        f = P["final"]
        self.before_half(3, "head")
        nb = math.ceil(VOC / D)
        if cfg.refresh != "none" and cfg.norm == "fixed":
            if cfg.refresh != "client" and self.levels_left() < 5:
                self.refresh("head")
            be.tolevel("x", "x", max(0, cfg.depth - 4))      # the 50 head blocks on few towers
        Wp = self.normed_input("head", f.get("c"), f["g"] * (1 + f["scale"]), f["W"], m, level=max(0, cfg.depth - 3))
        Wb = self.weights("head.b", lambda: (blocks_of(Wp, nb, 1, D, DM), bias_of(f["W"] @ f["shift"] + f["b"], nb, D)))
        outs = [f"l{k}" for k in range(nb)]
        be.matvec("head", Wb[0], Wb[1], ["u"], outs)
        be.check("logits block 0", "l0")
        lg = np.concatenate([np.asarray(be.dec(r)).reshape(T, D) for r in outs], 1)[:, :VOC]
        be.free("u", "x", *outs)
        return lg


# ---------------------------------------------------------------- torch side
def build_model(ckpt, stage, dev, exact_norm=False):
    import torch
    import plain_probe as pp
    m = pp.load_model(dev)
    if ckpt in (None, "none"):
        return m, None
    import e1_kd as E
    H, nb = pp.blocks(m)[0].n_heads, len(pp.blocks(m))
    E.patch_attention(m)
    E.convert(m, {"norm_c": torch.ones(2 * nb + 1, 2), "attn_b": torch.zeros(nb, H), "attn_z": torch.ones(nb, H, 2)})
    m.load_state_dict(torch.load(ckpt, map_location=dev))
    m.to(dev).eval()
    E.EXACT_NORM = exact_norm
    E.set_stage(m, stage, 1.0, train=False)
    return m, E


def default_ops(a):
    """Server-side ops implied by the E1 stage unless given explicitly."""
    k = -1 if a.ckpt in (None, "none") else "ABCD".index(a.stage)
    if a.exact_norm and not a.norm:
        a.norm = "client"
    a.norm = a.norm or ("fixed" if k >= 0 else "client")
    a.attn = a.attn or ("pow" if k >= 1 else "client")
    a.gelu = a.gelu or ("poly" if k >= 2 else "client")
    return k


def cmd_check(a):
    import torch
    import plain_probe as pp
    torch.set_grad_enabled(False)
    k = default_ops(a)
    m, E = build_model(a.ckpt, a.stage, pp.DEV, a.exact_norm)
    P = extract(m, a.L, a.norm == "fixed", a.attn != "client", a.gelu == "poly")
    a.refresh, a.depth, a.embed = "none", 0, "client"
    load_var(a)
    g = torch.Generator().manual_seed(a.seed)
    data = pp.load_chunks(a.L, a.n, seed=a.seed)
    res = {"args": vars(a), "cases": []}
    for j in range(a.n):
        x = data[j].clone()
        msk = torch.rand(a.L, generator=g) < (j + 1) / (a.n + 1)
        msk[0] = False
        x[msk] = pp.MASK
        ref = pp.forward_logits(m, x[None].to(pp.DEV))[0, :, :pp.MASK].double().cpu().numpy()
        c = Circuit(NumpyBE(a.L * D), P, a)
        lg = c.forward(x.numpy())[:, :pp.MASK]
        rec = {"n_masked": int(msk.sum()), "logit_max_abs_err": float(np.abs(lg - ref).max()), "logit_max_abs": float(np.abs(ref).max()),
               "argmax_agree": float((lg.argmax(-1) == ref.argmax(-1)).mean()), "stat": dict(c.stat)}
        print("[e5c]", json.dumps(rec), flush=True)
        res["cases"].append(rec)
    res["ok"] = all(r["logit_max_abs_err"] < 1e-2 * max(1.0, r["logit_max_abs"]) for r in res["cases"])
    print("[e5c] numpy circuit vs torch:", "OK" if res["ok"] else "MISMATCH", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0 if res["ok"] else 1


def load_var(a):
    a.var = a.xmax = None
    if a.norm == "rsqrt":
        cal = json.load(open(a.var_file))
        a.var, a.xmax = cal["var"], cal.get("xmax")
        k = max(a.var.values(), key=lambda v: v[1] / v[0])
        print(f"[e5c] variance bounds from {a.var_file}: widest max/min {k[1] / k[0]:.1f} (x margin^2 {a.var_margin ** 2:g})", flush=True)


def cmd_calib(a):
    """Public variance bounds of every LayerNorm input (per token), over masked real text and
    the model's own sampling trajectories: what the rsqrt norm's polynomial has to cover."""
    import torch
    import plain_probe as pp
    torch.set_grad_enabled(False)
    default_ops(a)
    m, E = build_model(a.ckpt, a.stage, pp.DEV, a.exact_norm)
    rng, xmax = {}, {}

    def hook(tag):
        def f(mod, inp):
            v = inp[0].float().var(-1, unbiased=False)
            lo, hi = float(v.min()), float(v.max())
            rng[tag] = [min(rng.get(tag, [lo, hi])[0], lo), max(rng.get(tag, [lo, hi])[1], hi)]
            xmax[tag] = max(xmax.get(tag, 0.0), float(inp[0].abs().max()))
        return f
    for i, b in enumerate(pp.blocks(m)):
        b.norm1.register_forward_pre_hook(hook(f"L{i}.qkv"))
        b.norm2.register_forward_pre_hook(hook(f"L{i}.up"))
    m.backbone.output_layer.norm_final.register_forward_pre_hook(hook("head"))
    data = pp.load_chunks(a.L, a.n, seed=a.seed)
    g = torch.Generator().manual_seed(a.seed)
    for ratio in (0.15, 0.5, 0.9, 1.0):
        x = data.clone()
        msk = torch.rand(x.shape, generator=g) < ratio
        msk[:, 0] = False
        x[msk] = pp.MASK
        for st in range(0, len(x), 16):
            pp.forward_logits(m, x[st:st + 16].to(pp.DEV))
    if a.gen_n:
        pp.PIN0 = True
        pp.sample(m, a.gen_n, a.L, a.steps, a.seed)
    res = {"args": vars(a), "var": rng, "xmax": xmax, "widest_ratio": max(v[1] / v[0] for v in rng.values())}
    print("[e5c]", json.dumps({k: [round(v[0], 3), round(v[1], 1)] for k, v in rng.items()}), flush=True)
    print(f"[e5c] widest max/min variance ratio {res['widest_ratio']:.1f}; max|x| by norm "
          f"{[round(xmax[k], 1) for k in rng]}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    return 0


def cmd_run(a):
    import torch
    import plain_probe as pp
    torch.set_grad_enabled(False)
    k = default_ops(a)
    load_var(a)
    T = (1 << (a.logn - 1)) // D
    assert a.L == T, f"L must be {T} (one ciphertext)"
    t0 = time.perf_counter()
    m, E = build_model(a.ckpt, a.stage, pp.DEV, a.exact_norm)
    P = extract(m, a.L, a.norm == "fixed", a.attn != "client", a.gelu == "poly")
    wd = tempfile.mkdtemp(prefix="e5c_", dir=a.workdir)
    os.environ["FIDES_VM_N1"] = str(a.n1)
    vm = VM(a.bin, a.logn, a.depth, D, a.gpus, rot_keys(a.n1), a.refresh in ("boot", "boot2"), wd, cpu=a.cpu)
    print("[e5c]", json.dumps(vm.ready), flush=True)
    chk = []
    be = PairBE(VmBE(vm), NumpyBE(vm.slots), chk) if a.debug else VmBE(vm)
    circ = Circuit(be, P, a)
    res = {"args": vars(a), "ready": vm.ready, "setup_s": time.perf_counter() - t0, "passes": [], "checks": chk}
    real = pp.forward_logits

    def fl(mm, x):
        before = {kk: dict(v) for kk, v in vm.stats().items()}
        circ.stat.clear()
        t1 = time.perf_counter()
        lg = torch.from_numpy(circ.forward(x[0].cpu().numpy())).float()[None].to(x.device)
        lg[..., pp.MASK] = -float("inf")
        wall = time.perf_counter() - t1
        ref = real(mm, x)
        fin = torch.isfinite(ref)
        after = vm.stats()
        d = lambda key: {kk: after[key][kk] - before[key].get(kk, 0) for kk in after[key]}
        kl = float((ref.log_softmax(-1).exp() * (ref.log_softmax(-1) - lg.log_softmax(-1)))[fin].sum() / x.shape[1])
        rec = {"pass": len(res["passes"]), "n_masked": int((x == pp.MASK).sum()), "wall_s": wall, "vm_seconds": d("seconds"),
               "vm_calls": d("calls"), "matvec": d("matvec"), "logit_max_abs_err": float((lg - ref)[fin].abs().max()),
               "logit_max_abs": float(ref[fin].abs().max()), "argmax_agree": float((lg.argmax(-1) == ref.argmax(-1)).float().mean()),
               "kl_to_plain": kl, "stat": dict(circ.stat), "mem": vm.cmd("mem")}
        res["passes"].append(rec)
        print("[e5c]", json.dumps(rec), flush=True)
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
        if a.max_passes and len(res["passes"]) >= a.max_passes:
            raise StopIteration
        return lg

    pp.PIN0 = True
    m._e5_L = a.L
    start, orig = start_state(pp, a.L, a.infill, a.seed)
    pp.forward_logits = fl
    xs = None
    try:
        xs = sample_from(pp, m, start, a.steps, a.seed)
    except StopIteration:
        pass
    pp.forward_logits = real
    ps = res["passes"]
    res["summary"] = {"passes": len(ps), "s_per_pass": float(np.mean([p["wall_s"] for p in ps])),
                      "logit_max_abs_err": max(p["logit_max_abs_err"] for p in ps),
                      "argmax_agree_min": min(p["argmax_agree"] for p in ps), "kl_to_plain_max": max(p["kl_to_plain"] for p in ps),
                      "ops": {"norm": a.norm, "attn": a.attn, "gelu": a.gelu, "refresh": a.refresh, "embed": a.embed}}
    if xs is not None:
        res["sample"] = sample_report(pp, m, a, xs, start, orig)
        if start is not None:
            print("[e5c] start :", res["sample"]["start_text"].replace("\n", " ")[:400], flush=True)
        print("[e5c] sample:", res["sample"]["text"].replace("\n", " ")[:400], flush=True)
        print("[e5c] plain :", res["sample"]["plaintext_text"].replace("\n", " ")[:400], flush=True)
    print("[e5c]", json.dumps(res["summary"]), flush=True)
    vm.close()
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "run", "calib"):
        s = sub.add_parser(name)
        s.add_argument("--ckpt", default="none", help="E1 student state dict, or none for the exact MDLM")
        s.add_argument("--stage", default="A", choices=list("ABCD"))
        s.add_argument("--norm", choices=("fixed", "client", "rsqrt"), default=None)
        s.add_argument("--var-file", help="variance bounds from `calib` (for --norm rsqrt)")
        s.add_argument("--var-margin", type=float, default=3.0, help="bounds widened to [min / m, max * m]")
        s.add_argument("--s-margin", type=float, default=2.0, help="rsqrt norm: x scale per norm = 2^ceil(log2(m max|x|))")
        s.add_argument("--rsqrt-deg", type=int, default=15)
        s.add_argument("--gold-iters", type=int, default=2)
        s.add_argument("--n1", type=int, default=32, help="matvec baby steps (16 halves its memory; use with the bootstrap keys loaded)")
        s.add_argument("--boot-bits", type=int, default=8, help="two-pass bootstrap: floor of the single-pass precision")
        s.add_argument("--attn", choices=("pow", "client", "client-pow"), default=None)
        s.add_argument("--gelu", choices=("poly", "client"), default=None)
        s.add_argument("--exact-norm", action="store_true", help="student trained with e1_kd.py --exact-norm (norms on the client)")
        s.add_argument("--S", type=float, default=1024.0, help="residual-stream scale: x is held as x/S")
        s.add_argument("--L", type=int, default=64)
        s.add_argument("--layers", type=int, default=None)
        s.add_argument("--seed", type=int, default=11)
        s.add_argument("--out", default=None)
        if name == "check":
            s.add_argument("--n", type=int, default=3)
        elif name == "calib":
            s.add_argument("--n", type=int, default=64)
            s.add_argument("--gen-n", type=int, default=8)
            s.add_argument("--steps", type=int, default=16)
        else:
            s.add_argument("--bin", required=True)
            s.add_argument("--logn", type=int, default=17)
            s.add_argument("--depth", type=int, default=13)
            s.add_argument("--gpus", type=int, default=1)
            s.add_argument("--cpu", action="store_true")
            s.add_argument("--refresh", choices=("client", "boot", "boot2", "none"), default="client")
            s.add_argument("--embed", choices=("server", "client"), default="server")
            s.add_argument("--steps", type=int, default=16)
            s.add_argument("--infill", type=float, default=0.0, help="start from real text with this share masked (0: all masked)")
            s.add_argument("--max-passes", type=int, default=0)
            s.add_argument("--debug", action="store_true", help="numpy reference alongside; compare registers at stage boundaries")
            s.add_argument("--workdir", default=os.environ.get("TMPDIR", "/tmp"))
    a = ap.parse_args()
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sys.exit({"check": cmd_check, "run": cmd_run, "calib": cmd_calib}[a.cmd](a))


if __name__ == "__main__":
    main()
