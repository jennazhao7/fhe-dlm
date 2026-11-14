"""E5: end-to-end encrypted MDLM generation, client in the loop (HANDOFF §12).

The server is gpu/fides_e0/fides_vm (FIDESlib, CKKS on the GPU); this script is
the client and the protocol driver. Assist levels, from most to least client
work:

  linear   Every weight matrix of MDLM (token embedding, QKV, attention out,
           FFN up/down, LM head) is applied by the server to ciphertexts; the
           client decrypts, does the weight-free steps in the clear (LayerNorm,
           rotary + softmax attention, GELU, the adaLN constants, sampling) and
           re-encrypts. Exact MDLM, no E1 student, depth-2 CKKS, no bootstrap;
           ~50 rounds per pass, and the client sees every activation, so this
           protects the client's tokens from the server but not the model from
           the client. It is the skeleton the deeper levels replace op by op.

  selftest random matvec through the VM against numpy (toy ring on the CPU
           fallback, or the real ring on a GPU).

    python scripts/e5_pass.py selftest --bin gpu/fides_e0/build/fides_vm --logn 12 --D 64 --cpu
    PROBE_DEV=cpu python scripts/e5_pass.py linear --bin gpu/fides_e0/build/fides_vm \
        --L 64 --steps 16 --out results/e5/linear_L64.json

Layout: token-major, slot = token*D + dim, D = 1024, T = 2^16 / D = 64 tokens
per ciphertext; a (d_out x d_in) matrix is ceil(d_out/D) x ceil(d_in/D) blocks
of 1024 diagonals (2-half BSGS, GPU-encoded diagonals: job 1520180).
Scaling: the client sends x / s_in (public constant, default 2^15) and the
server holds W / s, b / (s s_in) with s >= max|W|, so every encoded value is
<= 1; the client multiplies the result by s s_in.
"""
import argparse, collections, json, math, os, subprocess, sys, tempfile, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- VM
class VM:
    def __init__(self, binary, logn, depth, D, gpus, rots, boot, workdir, cpu=False):
        env = dict(os.environ)
        if cpu:
            env["FIDES_E0_CPU"] = "1"
        self.D, self.slots, self.logn, self.depth = D, 1 << (logn - 1), logn, depth
        self.T = self.slots // D
        self.wd = workdir
        self.t = collections.defaultdict(float)     # seconds by op
        self.n = collections.defaultdict(int)
        self.mv = collections.defaultdict(float)    # matvec breakdown
        self.p = subprocess.Popen([binary, str(logn), str(depth), str(D), str(gpus), ",".join(map(str, rots)), str(int(boot))],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
        self.ready = self._reply()

    def _reply(self):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"fides_vm exited (code {self.p.poll()})")
            if line.startswith("OK "):
                return json.loads(line[3:])
            if line.startswith("ERR"):
                raise RuntimeError(line.strip())
            print("[vm]", line.rstrip(), file=sys.stderr, flush=True)

    def cmd(self, s):
        self.p.stdin.write(s + "\n"); self.p.stdin.flush()
        r = self._reply()
        self.t[r["op"]] += r["s"]; self.n[r["op"]] += 1
        if r["op"] == "matvec":
            for k in ("rot_s", "host_fft_s", "upload_s", "mult_s", "pt_mults", "rots", "skipped"):
                self.mv[k] += r[k]
        return r

    def enc(self, r, v, level=0):
        path = os.path.join(self.wd, f"io_{r}.f64")
        np.ascontiguousarray(v, "<f8").tofile(path)
        return self.cmd(f"enc {r} {path} {level}")

    def dec(self, r):
        path = os.path.join(self.wd, f"io_{r}.f64")
        self.cmd(f"dec {r} {path}")
        return np.fromfile(path, "<f8")

    def ct_bytes(self, level=0):
        return 2 * (self.depth + 1 - level) * (1 << self.logn) * 8

    def close(self):
        try:
            self.p.stdin.write("quit\n"); self.p.stdin.flush(); self.p.wait(timeout=60)
        except Exception:
            self.p.kill()

    def stats(self):
        return {"seconds": dict(self.t), "calls": dict(self.n), "matvec": dict(self.mv)}


class EncLinear:
    """y = W x + b per token, W applied by the server under encryption."""
    def __init__(self, vm, name, W, b=None, s_in=2.0 ** 15):
        self.vm, self.name, self.s_in = vm, name, float(s_in)
        self.out, self.inn = W.shape
        D = vm.D
        self.n_out, self.n_in = math.ceil(self.out / D), math.ceil(self.inn / D)
        self.s = float(max(1.0, np.abs(W).max(), 0.0 if b is None else np.abs(b).max() / self.s_in))
        Wp = np.zeros((self.n_out * D, self.n_in * D), np.float32)
        Wp[:self.out, :self.inn] = W / self.s
        self.wfile = os.path.join(vm.wd, f"w_{name}.f32")
        Wp.reshape(self.n_out, D, self.n_in, D).transpose(0, 2, 1, 3).astype("<f4").tofile(self.wfile)
        self.bfile = "-"
        if b is not None:
            bp = np.zeros(self.n_out * D)
            bp[:self.out] = b / (self.s * self.s_in)
            self.bfile = os.path.join(vm.wd, f"b_{name}.f64")
            bp.astype("<f8").tofile(self.bfile)
        self.calls = self.cts_up = self.cts_down = self.rescaled = 0

    def __call__(self, X):
        vm, D, T = self.vm, self.vm.D, self.vm.T
        X = np.asarray(X, np.float64)
        L = X.shape[0]
        s_in = self.s_in
        while np.abs(X).max() > s_in:        # the public bound was too small: note it, do not overflow
            s_in *= 2; self.rescaled += 1
        if s_in != self.s_in and self.bfile != "-":
            raise RuntimeError(f"{self.name}: |x| {np.abs(X).max():.3g} exceeds s_in {self.s_in:g} on a layer with a bias")
        Y = np.zeros((L, self.n_out * D))
        ins = [f"x{i}" for i in range(self.n_in)]
        outs = [f"y{o}" for o in range(self.n_out)]
        for c0 in range(0, L, T):
            n = min(T, L - c0)
            chunk = np.zeros((T, self.n_in * D))
            chunk[:n, :self.inn] = X[c0:c0 + n] / s_in
            for i, r in enumerate(ins):
                vm.enc(r, chunk[:, i * D:(i + 1) * D].reshape(-1))
            vm.cmd(f"matvec {self.n_out} {self.n_in} {self.wfile} {self.bfile} {' '.join(ins)} {' '.join(outs)}")
            for o, r in enumerate(outs):
                Y[c0:c0 + n, o * D:(o + 1) * D] = vm.dec(r).reshape(T, D)[:n]
            vm.cmd("free " + " ".join(ins + outs))
            self.cts_up += self.n_in; self.cts_down += self.n_out
        self.calls += 1
        return Y[:, :self.out] * (self.s * s_in)


# ---------------------------------------------------------------- selftest
def cmd_selftest(a):
    D = a.D
    N1 = 1 << (int(math.log2(D)) // 2)
    wd = tempfile.mkdtemp(prefix="e5_", dir=a.workdir)
    vm = VM(a.bin, a.logn, 2, D, 1, [1, N1, -D], False, wd, cpu=a.cpu)
    print("[e5]", json.dumps(vm.ready), flush=True)
    rng = np.random.default_rng(0)
    res = {"ready": vm.ready, "cases": []}
    for out, inn, bias, L in ((D, D, False, vm.T), (int(2.3 * D), int(0.75 * D), True, vm.T + 3), (int(0.75 * D), 3 * D, True, 5)):
        W = rng.normal(0, 1 / math.sqrt(inn), (out, inn))
        b = rng.normal(0, 1, out) if bias else None
        X = rng.uniform(-300, 300, (L, inn))
        lin = EncLinear(vm, f"t{out}x{inn}", W, b)
        t0 = time.perf_counter()
        Y = lin(X)
        ref = X @ W.astype(np.float32).astype(np.float64).T + (0 if b is None else b)
        c = {"out": out, "in": inn, "bias": bias, "L": L, "max_abs_err": float(np.abs(Y - ref).max()),
             "max_abs_ref": float(np.abs(ref).max()), "seconds": time.perf_counter() - t0}
        print("[e5]", json.dumps(c), flush=True)
        res["cases"].append(c)
    res["vm"] = vm.stats()
    res["ok"] = all(c["max_abs_err"] < 1e-3 for c in res["cases"])
    print("[e5]", json.dumps(res["vm"]), "OK" if res["ok"] else "FAILED", flush=True)
    vm.close()
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0 if res["ok"] else 1


# ---------------------------------------------------------------- linear assist
def cmd_linear(a):
    import torch
    import plain_probe as pp
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    D = 1024
    N1 = 32
    wd = tempfile.mkdtemp(prefix="e5_", dir=a.workdir)
    t0 = time.perf_counter()
    m = pp.load_model(pp.DEV)
    vm = VM(a.bin, a.logn, 2, D, a.gpus, [1, N1, -D], False, wd, cpu=a.cpu)
    print("[e5]", json.dumps(vm.ready), flush=True)
    ST = {"on": False, "errs": collections.defaultdict(float), "refmax": collections.defaultdict(float)}
    lins = {}

    class ServerLinear(torch.nn.Module):
        def __init__(self, name, lin, s_in):
            super().__init__()
            self.lin, self.name = lin, name
            W = lin.weight.detach().double().cpu().numpy()
            b = None if lin.bias is None else lin.bias.detach().double().cpu().numpy()
            self.enc = lins[name] = EncLinear(vm, name, W, b, s_in)

        def forward(self, x):
            ref = self.lin(x)
            if not ST["on"]:
                return ref
            assert x.shape[0] == 1, "one sequence per pass"
            y = torch.from_numpy(self.enc(x[0].double().cpu().numpy())).to(ref)[None]
            kind = self.name.split(".")[-1]
            ST["errs"][kind] = max(ST["errs"][kind], float((y - ref).abs().max()))
            ST["refmax"][kind] = max(ST["refmax"][kind], float(ref.abs().max()))
            return y

    class ServerEmbed(torch.nn.Module):
        def __init__(self, emb):
            super().__init__()
            self.emb = emb
            self.enc = lins["embed"] = EncLinear(vm, "embed", emb.embedding.detach().double().cpu().numpy().T, None, 1.0)

        def forward(self, idx):
            ref = self.emb(idx)
            if not ST["on"]:
                return ref
            oh = np.zeros((idx.shape[1], self.emb.embedding.shape[0]))
            oh[np.arange(idx.shape[1]), idx[0].cpu().numpy()] = 1.0
            y = torch.from_numpy(self.enc(oh)).to(ref)[None]
            ST["errs"]["embed"] = max(ST["errs"]["embed"], float((y - ref).abs().max()))
            ST["refmax"]["embed"] = max(ST["refmax"]["embed"], float(ref.abs().max()))
            return y

    bb = m.backbone
    for i, b in enumerate(pp.blocks(m)):
        if a.layers is not None and i >= a.layers:
            break
        b.attn_qkv = ServerLinear(f"L{i}.qkv", b.attn_qkv, a.s_in)
        b.attn_out = ServerLinear(f"L{i}.out", b.attn_out, a.s_in)
        b.mlp[0] = ServerLinear(f"L{i}.up", b.mlp[0], a.s_in)
        b.mlp[2] = ServerLinear(f"L{i}.down", b.mlp[2], a.s_in)
    if a.head == "server":
        bb.output_layer.linear = ServerLinear("head", bb.output_layer.linear, a.s_in)
    if a.embed == "server":
        bb.vocab_embed = ServerEmbed(bb.vocab_embed)
    setup_s = time.perf_counter() - t0
    blocks_per_pass = sum(l.n_in * l.n_out for l in lins.values())
    print(f"[e5] setup {setup_s:.1f}s; {len(lins)} server linears, {blocks_per_pass} 1024-diagonal blocks per 64-token ciphertext",
          flush=True)

    res = {"args": vars(a), "ready": vm.ready, "setup_s": setup_s, "rounds_per_pass": len(lins),
           "blocks_per_ct_pass": blocks_per_pass, "passes": []}
    real = pp.forward_logits

    def fl(mm, x):
        before = {k: dict(v) for k, v in vm.stats().items()}
        ST["errs"].clear(); ST["refmax"].clear()
        t1 = time.perf_counter()
        ST["on"] = True
        lg = real(mm, x)
        ST["on"] = False
        wall = time.perf_counter() - t1
        ref = real(mm, x)
        fin = torch.isfinite(ref)
        after = vm.stats()
        d = lambda k: {kk: after[k][kk] - before[k].get(kk, 0) for kk in after[k]}
        up = sum(l.cts_up for l in lins.values()); down = sum(l.cts_down for l in lins.values())
        rec = {"pass": len(res["passes"]), "n_masked": int((x == pp.MASK).sum()), "wall_s": wall,
               "vm_seconds": d("seconds"), "vm_calls": d("calls"), "matvec": d("matvec"),
               "logit_max_abs_err": float((lg - ref)[fin].abs().max()),
               "argmax_agree": float((lg.argmax(-1) == ref.argmax(-1)).float().mean()),
               "linear_max_abs_err": dict(ST["errs"]), "linear_max_abs_ref": dict(ST["refmax"]),
               "cts_up_total": up, "cts_down_total": down}
        res["passes"].append(rec)
        print("[e5]", json.dumps(rec), flush=True)
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
        if a.max_passes and len(res["passes"]) >= a.max_passes:
            raise StopIteration
        return lg

    pp.PIN0 = True
    pp.forward_logits = fl
    xs = None
    try:
        xs = pp.sample(m, 1, a.L, a.steps, a.seed)
    except StopIteration:
        pass
    pp.forward_logits = real
    n = len(res["passes"])
    res["summary"] = {
        "passes": n, "L": a.L, "cts_per_pass_up": sum(l.cts_up for l in lins.values()) / max(n, 1),
        "cts_per_pass_down": sum(l.cts_down for l in lins.values()) / max(n, 1), "ct_bytes": vm.ct_bytes(),
        "s_per_pass": float(np.mean([p["wall_s"] for p in res["passes"]])),
        "server_matvec_s_per_pass": float(np.mean([p["vm_seconds"].get("matvec", 0) for p in res["passes"]])),
        "client_encdec_s_per_pass": float(np.mean([p["vm_seconds"].get("enc", 0) + p["vm_seconds"].get("dec", 0) for p in res["passes"]])),
        "logit_max_abs_err": max(p["logit_max_abs_err"] for p in res["passes"]),
        "argmax_agree_min": min(p["argmax_agree"] for p in res["passes"]),
        "rescaled": {k: l.rescaled for k, l in lins.items() if l.rescaled},
        "mem": vm.cmd("mem")}
    if xs is not None:
        ref = pp.sample(m, 1, a.L, a.steps, a.seed)          # same seed, plaintext model
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(pp.snapshot("openai-community/gpt2"))
        res["sample"] = {"tokens": xs[0].tolist(), "text": tok.decode(xs[0].tolist()),
                         "plaintext_tokens": ref[0].tolist(), "plaintext_text": tok.decode(ref[0].tolist()),
                         "identical_to_plaintext": bool((xs == ref).all()),
                         "token_agreement": float((xs == ref).float().mean())}
        print("[e5] sample:", res["sample"]["text"].replace("\n", " ")[:400], flush=True)
        print("[e5] identical to the plaintext model's sample:", res["sample"]["identical_to_plaintext"], flush=True)
    print("[e5]", json.dumps(res["summary"]), flush=True)
    vm.close()
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("selftest", "linear"):
        s = sub.add_parser(name)
        s.add_argument("--bin", required=True)
        s.add_argument("--logn", type=int, default=17)
        s.add_argument("--cpu", action="store_true", help="FIDESlib's OpenFHE fallback (toy rings only)")
        s.add_argument("--workdir", default=os.environ.get("TMPDIR", "/tmp"))
        s.add_argument("--out", default=None)
        if name == "selftest":
            s.add_argument("--D", type=int, default=1024)
        else:
            s.add_argument("--gpus", type=int, default=1)
            s.add_argument("--L", type=int, default=64)
            s.add_argument("--steps", type=int, default=16)
            s.add_argument("--seed", type=int, default=11)
            s.add_argument("--s-in", type=float, default=2.0 ** 15)
            s.add_argument("--embed", choices=("server", "client"), default="server")
            s.add_argument("--head", choices=("server", "client"), default="server")
            s.add_argument("--layers", type=int, default=None, help="only the first n blocks go through the server (smoke tests)")
            s.add_argument("--max-passes", type=int, default=0)
    a = ap.parse_args()
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sys.exit(cmd_selftest(a) if a.cmd == "selftest" else cmd_linear(a))


if __name__ == "__main__":
    main()
