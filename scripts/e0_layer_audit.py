"""E0 / HANDOFF §6.3: per-layer depth audit. Builds the encrypted transformer
layer that e0_cost_model.py prices -- same interleaved layout, same kernels --
runs it under CKKS, checks it against numpy, and records how many levels
each stage really consumes on the critical path.

The cost model assumes 18 levels/layer (QKV 1, scores 2, power 1, /const 1,
replicate 1, xV 1, out-proj 1, 2 norms 1+1, FFN up 1, GELU 6, down 1).

Level accounting depends on the circuit, not on N or d, so the default run is
a toy model at N=2^12 (HEStd_NotSet) with the real parameters otherwise:
depth 43, 59/60, FLEXIBLEAUTO, UNIFORM_TERNARY, bootstrap budget [4,4], full
slots. d_model 48 padded to D=64 (vs 768 -> 1024), head_dim 16, FFN 4x
(3 blocks of D, like 3072 = 3 x 1024), T = slots/D = 32 tokens per ct,
B = 4 query tokens. The same script at --logn 17 --d 768 --dpad 1024
--hd 64 --B 8 is the full-size layer (needs tjws-class RAM; gives timings).

Layout: slot = dim*T + token. Kernels (all exact, see comments in code):
  matvec      BSGS over D diagonals, token masks folded into the diagonals
  Q replicate log2(T/B) rotations by -B*2^k (Q masked to tokens < B by QKV)
  K/V shift   token shift r with the wrap fixed by 2 masks (2 rot + 2 ptm)
  head sum    log2(hd) rotations by T*2^k, valid at each head's first dim
  power 2     ct x ct
  /Z + mask   one ptm (keeps head-first dims, scales by 1/Z)
  replicate   log2(hd) rotations by -T*2^k
  gather      log2(T/B) rotations by B*2^k, valid at tokens < B; the
              out-proj diagonals zero the other tokens
  norm        fixed-constant: c*(x - mean_real_dims(x)), padded dims re-zeroed;
              one ptm level (x*(c m) - sum(x)*(c/d m))
  GELU        EvalChebyshevFunction on [-8, 8]
The residual stream is carried as x/S (S folded into norm constants and the
out/down projections, so it costs nothing) to keep |x/S| < 1 for bootstrap.
Bootstraps are placed on the residual stream before a half-layer (attention
or FFN) whenever the levels left are fewer than that half needs.
"""
import argparse, json, math, sys, time
import numpy as np
from openfhe import (CCParamsCKKSRNS, GenCryptoContext, SecretKeyDist, SecurityLevel,
                     ScalingTechnique, PKESchemeFeature, FHECKKSRNS)

ap = argparse.ArgumentParser()
ap.add_argument("--logn", type=int, default=12)
ap.add_argument("--d", type=int, default=48)        # real d_model
ap.add_argument("--dpad", type=int, default=64)     # padded D (power of 2)
ap.add_argument("--hd", type=int, default=16)       # head dim
ap.add_argument("--B", type=int, default=4)         # query tokens per pass
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--gelu-degree", type=int, default=31)
ap.add_argument("--depth", type=int, default=43)
ap.add_argument("--budget", type=int, default=4)
ap.add_argument("--S", type=float, default=4.0)     # residual-stream scale
ap.add_argument("--packed", action="store_true", help="token-group packed matvecs")
ap.add_argument("--out", default=None)
A = ap.parse_args()

D, d, HD, B = A.dpad, A.d, A.hd, A.B
N1 = 1 << (int(math.log2(D)) // 2)
N2 = D // N1
GELU_RANGE = 8.0
gelu = lambda z: 0.5 * z * (1 + np.tanh(math.sqrt(2 / math.pi) * (z + 0.044715 * z ** 3)))

# ---------------- context ----------------
skd = SecretKeyDist.UNIFORM_TERNARY
p = CCParamsCKKSRNS()
p.SetSecretKeyDist(skd)
p.SetSecurityLevel(SecurityLevel.HEStd_128_classic if A.logn >= 16 else SecurityLevel.HEStd_NotSet)
p.SetRingDim(1 << A.logn); p.SetScalingModSize(59); p.SetFirstModSize(60)
p.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO); p.SetMultiplicativeDepth(A.depth)
cc = GenCryptoContext(p)
for f in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH, PKESchemeFeature.LEVELEDSHE,
          PKESchemeFeature.ADVANCEDSHE, PKESchemeFeature.FHE):
    cc.Enable(f)
SLOTS = cc.GetRingDimension() // 2
T = SLOTS // D
assert T % B == 0 and d % HD == 0 and D >= d
BOOT_DEPTH = int(FHECKKSRNS.GetBootstrapDepth([A.budget, A.budget], skd))
USABLE = A.depth - BOOT_DEPTH          # confirmed by e0_levels_offbyone.py

rots = set()
rots |= {b * T for b in range(1, N1)} | {g * N1 * T for g in range(1, N2)}     # matvec
rots |= {-B * 2 ** k for k in range(int(math.log2(T // B)))}                  # Q replicate
rots |= {B * 2 ** k for k in range(int(math.log2(T // B)))}                   # gather
rots |= {r for r in range(1, B)} | {r - T for r in range(1, B)}               # K/V shift
rots |= {T * 2 ** k for k in range(int(math.log2(D)))}                        # head sum, norm
rots |= {-T * 2 ** k for k in range(int(math.log2(HD)))}                      # replicate
rots = sorted(r % SLOTS if r % SLOTS else r for r in rots if r % SLOTS)
rots = sorted({r if r < SLOTS // 2 else r - SLOTS for r in rots})

t0 = time.perf_counter()
cc.EvalBootstrapSetup([A.budget, A.budget], [0, 0], SLOTS)
keys = cc.KeyGen(); cc.EvalMultKeyGen(keys.secretKey)
cc.EvalRotateKeyGen(keys.secretKey, rots)
cc.EvalBootstrapKeyGen(keys.secretKey, SLOTS)
SETUP_S = time.perf_counter() - t0


def lvl(ct):
    """Levels consumed so far (FLEXIBLEAUTO rescales lazily)."""
    return ct.GetLevel() + ct.GetNoiseScaleDeg() - 1


def enc_pt(v, level):
    return cc.MakeCKKSPackedPlaintext(list(map(float, v)), 1, level)


def pack(X):                       # X: (T, D) -> slot dim*T + token
    return np.asarray(X).T.reshape(-1)


def unpack(v):
    return np.asarray(v)[:D * T].reshape(D, T).T


def dec(ct):
    pt = cc.Decrypt(ct, keys.secretKey); pt.SetLength(SLOTS)
    return unpack(pt.GetRealPackedValue())


def rot(ct, r):
    r %= SLOTS
    return ct if r == 0 else cc.EvalRotate(ct, r if r < SLOTS // 2 else r - SLOTS)


def tok_mask(pred):                # slot vector, 1 where pred(token)
    return pack(np.tile(np.array([1.0 if pred(t) else 0.0 for t in range(T)])[:, None], (1, D)))


def dim_mask(pred, scale=1.0):
    return pack(np.tile(np.array([scale if pred(j) else 0.0 for j in range(D)])[None, :], (T, 1)))


# ---------------- kernels ----------------
def matvec(cts, W, out_tokens=None):
    """y_o = sum_i W[o][i] x_i for D x D blocks W[o][i] (out j <- in), BSGS;
    out_tokens(t) -> bool zeroes other tokens via the diagonals (free)."""
    tm = tok_mask(out_tokens) if out_tokens else None
    babies = [[ct] + [rot(ct, b * T) for b in range(1, N1)] for ct in cts]
    level = max(c.GetLevel() for c in cts)
    outs = []
    for Wo in W:
        acc_out = None
        for g in range(N2):
            acc = None
            for i, bab in enumerate(babies):
                for b in range(N1):
                    k = g * N1 + b
                    j = np.arange(D)
                    diag = np.roll(Wo[i][j, (j + k) % D], g * N1)
                    v = np.repeat(diag, T)
                    if tm is not None:
                        v = v * tm
                    if not v.any():
                        continue
                    term = cc.EvalMult(bab[b], enc_pt(v, level))
                    acc = term if acc is None else cc.EvalAdd(acc, term)
            if acc is None:
                continue
            acc = rot(acc, g * N1 * T)
            acc_out = acc if acc_out is None else cc.EvalAdd(acc_out, acc)
        outs.append(acc_out)
    return outs


def rotsum(ct, step, n, sign=1):
    """sum_{j<n} rot(ct, sign*j*step) in log2(n) rotations (n power of 2)."""
    for k in range(int(math.log2(n))):
        ct = cc.EvalAdd(ct, rot(ct, sign * step * 2 ** k))
    return ct


def tshift(ct, r):
    """token shift: slot (d,t) <- (d, (t+r) mod T). 2 rot + 2 ptm, 1 level."""
    if r == 0:
        return ct
    lo = cc.EvalMult(rot(ct, r), enc_pt(tok_mask(lambda t: t < T - r), ct.GetLevel()))
    hi = cc.EvalMult(rot(ct, r - T), enc_pt(tok_mask(lambda t: t >= T - r), ct.GetLevel()))
    return cc.EvalAdd(lo, hi)


def norm(ct, c):
    """c * (x - mean over the d real dims), padded dims kept at 0. 1 level."""
    s = rotsum(ct, T, D)                                   # every slot: sum over dims
    a = cc.EvalMult(ct, enc_pt(dim_mask(lambda j: j < d, c), ct.GetLevel()))
    b = cc.EvalMult(s, enc_pt(dim_mask(lambda j: j < d, c / d), s.GetLevel()))
    return cc.EvalSub(a, b)


# ---------------- reference (same function, float64) ----------------
def ref_norm(X, c):
    Y = np.zeros_like(X)
    Y[:, :d] = c * (X[:, :d] - X[:, :d].mean(1, keepdims=True))
    return Y


def calibrate(X, P):
    """Public constants calibrated once in plaintext (here on the audit
    input), as the cost model assumes: fixed-constant norms c = 1/std of the
    norm's input, attention normaliser Z. Sets P["c1"], P["c2"], P["Z"];
    returns max |GELU input|."""
    P["c1"] = 1.0 / X[:, :d].std()
    H = ref_norm(X, P["c1"])
    Q, K, V = H @ P["Wq"].T, H @ P["Wk"].T, H @ P["Wv"].T
    # public normaliser Z: calibrated so each query's weights sum to ~1
    P["Z"] = float(np.mean([((Q[:B, h*HD:(h+1)*HD] @ K[:, h*HD:(h+1)*HD].T) ** 2).sum(1).mean()
                            for h in range(d // HD)]))
    out = np.zeros_like(X)
    for h in range(d // HD):
        sl = slice(h * HD, (h + 1) * HD)
        out[:B, sl] = (Q[:B, sl] @ K[:, sl].T) ** 2 / P["Z"] @ V[:, sl]
    X2 = X + out @ P["Wo"].T * (np.arange(T) < B)[:, None]
    P["c2"] = 1.0 / X2[:, :d].std()
    return float(np.abs(ref_norm(X2, P["c2"]) @ P["W1"].T).max())


def ref_layer(X, P):
    """X: residual (T, D) in true units. Returns new residual, true units."""
    H = ref_norm(X, P["c1"])
    Q, K, V = H @ P["Wq"].T, H @ P["Wk"].T, H @ P["Wv"].T
    out = np.zeros_like(X)
    for h in range(d // HD):
        sl = slice(h * HD, (h + 1) * HD)
        Sc = Q[:B, sl] @ K[:, sl].T                       # (B, T)
        Pr = Sc ** 2 / P["Z"]
        out[:B, sl] = Pr @ V[:, sl]
    X = X + out @ P["Wo"].T * (np.arange(T) < B)[:, None]
    H = ref_norm(X, P["c2"])
    U = gelu(H @ P["W1"].T)
    return X + U @ P["W2"].T


def make_params(rng):
    def W(o, i, s):
        M = np.zeros((o, i)); M[:, :] = rng.normal(0, s, (o, i)); return M
    P = {"c1": 1.0, "c2": 1.0, "Z": float(T)}
    P["Wq"] = np.zeros((D, D)); P["Wq"][:d, :d] = W(d, d, 1 / math.sqrt(d)) / HD ** 0.25
    P["Wk"] = np.zeros((D, D)); P["Wk"][:d, :d] = W(d, d, 1 / math.sqrt(d)) / HD ** 0.25
    P["Wv"] = np.zeros((D, D)); P["Wv"][:d, :d] = W(d, d, 1 / math.sqrt(d))
    P["Wo"] = np.zeros((D, D)); P["Wo"][:d, :d] = W(d, d, 0.5 / math.sqrt(d))
    F = 4 * d                                              # FFN inner, <= 3*D blocks
    P["W1"] = np.zeros((3 * D, D)); P["W1"][:F, :d] = W(F, d, 1.0 / math.sqrt(d))
    P["W2"] = np.zeros((D, 3 * D)); P["W2"][:d, :F] = W(d, F, 0.3 / math.sqrt(F))
    return P


def blocks(M, o, i):
    return [[M[a * D:(a + 1) * D, b * D:(b + 1) * D] for b in range(i)] for a in range(o)]


# ---------------- encrypted layer ----------------
STAGES = []


def stage(name, ct, before):
    STAGES.append({"stage": name, "levels": lvl(ct) - before})
    return lvl(ct)


def enc_attention(x, P, S):
    """x: residual ct (units x/S). Returns x + attn/S."""
    L0 = lvl(x); mark = L0
    h = norm(x, P["c1"] * S)                               # true units
    mark = stage("norm1", h, mark)
    q = matvec([h], blocks(P["Wq"], 1, 1), out_tokens=lambda t: t < B)[0]
    k, v = matvec([h], blocks(np.concatenate([P["Wk"], P["Wv"]]), 2, 1))
    mark = stage("QKV matvec", k, mark)
    qrep = rotsum(q, B, T // B, sign=-1)
    hs_mask = dim_mask(lambda j: j < d and j % HD == 0, 1.0 / P["Z"])
    acc = None
    for r in range(B):
        kr, vr = tshift(k, r), tshift(v, r)
        if r == B - 1: m1 = stage("K/V token shift (wrap masks)", kr, mark)
        s = rotsum(cc.EvalMult(qrep, kr), T, HD)            # head sums at head-first dims
        if r == B - 1: m2 = stage("scores q*k (+head sum, 0 lvl)", s, m1)
        s = cc.EvalMult(s, s)
        if r == B - 1: m3 = stage("power 2", s, m2)
        s = cc.EvalMult(s, enc_pt(hs_mask, s.GetLevel()))
        s = rotsum(s, T, HD, sign=-1)
        if r == B - 1: m4 = stage("/Z + head mask + replicate", s, m3)
        pv = cc.EvalMult(s, vr)
        if r == B - 1: m5 = stage("x V", pv, m4)
        acc = pv if acc is None else cc.EvalAdd(acc, pv)
    o = rotsum(acc, B, T // B)                             # gather, valid at t < B
    o = matvec([o], blocks(P["Wo"] / S, 1, 1), out_tokens=lambda t: t < B)[0]
    stage("gather (0 lvl) + out-proj", o, m5)
    y = cc.EvalAdd(x, o)
    STAGES.append({"stage": "== attention half", "levels": lvl(y) - L0})
    return y


def enc_ffn(x, P, S):
    L0 = lvl(x); mark = L0
    h = norm(x, P["c2"] * S)
    mark = stage("norm2", h, mark)
    u = matvec([h], blocks(P["W1"], 3, 1))
    mark = stage("FFN up matvec 1->3", u[0], mark)
    g = [cc.EvalChebyshevFunction(gelu, ui, -GELU_RANGE, GELU_RANGE, A.gelu_degree) for ui in u]
    mark = stage(f"GELU Chebyshev deg {A.gelu_degree}", g[0], mark)
    y = matvec(g, blocks(P["W2"] / S, 1, 3))[0]
    stage("FFN down matvec 3->1", y, mark)
    y = cc.EvalAdd(x, y)
    STAGES.append({"stage": "== FFN half", "levels": lvl(y) - L0})
    return y


# ---------------- token-group packed layer (--packed) ----------------
# A DLM pass has only B live tokens in a ct of T token slots. Copy them into
# the G = T/B token groups (group g = tokens [gB, (g+1)B)) and give each group
# its own weight block: one 1024-diagonal pass then computes up to G block
# products (QKV, FFN up, FFN down, LM head), since the diagonal method never
# mixes token slots. Keys/values: the block's own B tokens (packed path) plus
# one encrypted KV-cache ct of T context tokens (original kernel).
G = T // B


def gmask(g, scale=1.0):
    return tok_mask(lambda t: t // B == g) * scale


def matvec_grouped(ct, Wg):
    """out(t) = Wg[t // B] x(t): one BSGS pass, group-dependent diagonals.
    Wg: list (len <= G) of D x D blocks or None (group zeroed)."""
    babies = [ct] + [rot(ct, b * T) for b in range(1, N1)]
    level = ct.GetLevel()
    j = np.arange(D)
    out = None
    for g_ in range(N2):
        acc = None
        for b in range(N1):
            k = g_ * N1 + b
            cols = np.zeros((D, T))
            for g, W in enumerate(Wg):
                if W is not None:
                    cols[:, g * B:(g + 1) * B] = np.roll(W[j, (j + k) % D], g_ * N1)[:, None]
            if not cols.any():
                continue
            term = cc.EvalMult(babies[b], enc_pt(cols.reshape(-1), level))
            acc = term if acc is None else cc.EvalAdd(acc, term)
        if acc is None:
            continue
        acc = rot(acc, g_ * N1 * T)
        out = acc if out is None else cc.EvalAdd(out, acc)
    return out


def norm_clean(ct, c):
    """norm() that also zeroes token groups >= 1 (garbage from FFN group sums
    lives there) -- the token mask rides in the same ptm, 1 level."""
    s = rotsum(ct, T, D)
    m = dim_mask(lambda j: j < d) * tok_mask(lambda t: t < B)
    a = cc.EvalMult(ct, enc_pt(m * c, ct.GetLevel()))
    b = cc.EvalMult(s, enc_pt(m * (c / d), s.GetLevel()))
    return cc.EvalSub(a, b)


def enc_attention_packed(x, P, S):
    L0 = lvl(x); mark = L0
    h = norm_clean(x, P["c1"] * S)
    mark = stage("norm1 (+token mask)", h, mark)
    hrep = rotsum(h, B, G, sign=-1)
    qkv = matvec_grouped(hrep, [P["Wq"], P["Wk"], P["Wv"]])
    mark = stage("QKV packed: 1 pass for 3 blocks", qkv, mark)
    m0 = gmask(0)
    q = cc.EvalMult(qkv, enc_pt(m0, qkv.GetLevel()))
    kb = cc.EvalMult(rot(qkv, B), enc_pt(m0, qkv.GetLevel()))
    vb = cc.EvalMult(rot(qkv, 2 * B), enc_pt(m0, qkv.GetLevel()))
    qrep, kbrep, vbrep = (rotsum(z, B, G, sign=-1) for z in (q, kb, vb))
    m1 = stage("split Q/K/V (group mask) + replicate", kbrep, mark)
    hs_mask = dim_mask(lambda j: j < d and j % HD == 0, 1.0 / P["Z"])
    Kc = cc.Encrypt(keys.publicKey, enc_pt(pack(P["Kc"]), 0))   # KV cache ct
    Vc = cc.Encrypt(keys.publicKey, enc_pt(pack(P["Vc"]), 0))

    def attend(kr, vr):
        s_ = rotsum(cc.EvalMult(qrep, kr), T, HD)
        s_ = cc.EvalMult(s_, s_)
        s_ = rotsum(cc.EvalMult(s_, enc_pt(hs_mask, s_.GetLevel())), T, HD, sign=-1)
        return cc.EvalMult(s_, vr)

    acc_b = acc_c = None
    for r in range(B):
        pv = attend(rot(kbrep, r), rot(vbrep, r))           # block keys, valid t < B
        acc_b = pv if acc_b is None else cc.EvalAdd(acc_b, pv)
        pc = attend(tshift(Kc, r), tshift(Vc, r))           # cache keys, all T slots
        acc_c = pc if acc_c is None else cc.EvalAdd(acc_c, pc)
    o = cc.EvalAdd(acc_b, rotsum(acc_c, B, G))              # valid at t < B
    m2 = stage("scores..xV (block + cache keys)", o, m1)
    o = matvec_grouped(o, [P["Wo"] / S])                    # zeroes groups >= 1
    stage("out-proj", o, m2)
    y = cc.EvalAdd(x, o)
    STAGES.append({"stage": "== attention half", "levels": lvl(y) - L0})
    return y


def enc_ffn_packed(x, P, S):
    L0 = lvl(x); mark = L0
    h = norm_clean(x, P["c2"] * S)
    mark = stage("norm2 (+token mask)", h, mark)
    hrep = rotsum(h, B, G, sign=-1)
    u = matvec_grouped(hrep, [W[0] for W in blocks(P["W1"], 3, 1)])
    mark = stage("FFN up packed: 1 pass for 3 blocks", u, mark)
    g = cc.EvalChebyshevFunction(gelu, u, -GELU_RANGE, GELU_RANGE, A.gelu_degree)
    mark = stage(f"GELU deg {A.gelu_degree} (1 call, not 3)", g, mark)
    y = matvec_grouped(g, blocks(P["W2"] / S, 1, 3)[0])
    y = cc.EvalAdd(cc.EvalAdd(y, rot(y, B)), rot(y, 2 * B))   # group sum -> t < B
    stage("FFN down packed + group sum", y, mark)
    y = cc.EvalAdd(x, y)            # groups >= 1 carry garbage; next norm drops it
    STAGES.append({"stage": "== FFN half", "levels": lvl(y) - L0})
    return y


def ref_attention_packed(X, P):
    H = ref_norm(X, P["c1"])
    Q, K, V = H @ P["Wq"].T, H @ P["Wk"].T, H @ P["Wv"].T
    out = np.zeros_like(X)
    for h in range(d // HD):
        sl = slice(h * HD, (h + 1) * HD)
        Pb = (Q[:, sl] @ K[:, sl].T) ** 2 / P["Z"]
        Pc = (Q[:, sl] @ P["Kc"][:, sl].T) ** 2 / P["Z"]
        out[:, sl] = Pb @ V[:, sl] + Pc @ P["Vc"][:, sl]
    return out


def ref_layer_packed(X, P):
    X = X + ref_attention_packed(X, P) @ P["Wo"].T
    U = gelu(ref_norm(X, P["c2"]) @ P["W1"].T)
    return X + U @ P["W2"].T


def calibrate_packed(X, P, rng):
    P["c1"] = 1.0 / X[:, :d].std()
    Cx = np.zeros((T, D)); Cx[:, :d] = rng.uniform(-1, 1, (T, d))   # context hidden
    Hc = ref_norm(Cx, 1.0 / Cx[:, :d].std())
    P["Kc"], P["Vc"] = Hc @ P["Wk"].T, Hc @ P["Wv"].T
    H = ref_norm(X, P["c1"]); Q, K = H @ P["Wq"].T, H @ P["Wk"].T
    P["Z"] = float(np.mean([((Q[:, h*HD:(h+1)*HD] @ np.concatenate([K, P["Kc"]])[:, h*HD:(h+1)*HD].T)
                             ** 2).sum(1).mean() for h in range(d // HD)]))
    X2 = X + ref_attention_packed(X, P) @ P["Wo"].T
    P["c2"] = 1.0 / X2[:, :d].std()
    return float(np.abs(ref_norm(X2, P["c2"]) @ P["W1"].T).max())


def main():
    rng = np.random.default_rng(0)
    X = np.zeros((T, D)); X[:, :d] = rng.uniform(-1, 1, (T, d))
    Ps = [make_params(rng) for _ in range(A.layers)]
    S = A.S
    if A.packed:
        X = X[:B]                      # only the B block tokens are live
        Xc = X.copy()
        for P in Ps:
            P["gelu_in_max"] = calibrate_packed(Xc, P, rng)
            Xc = ref_layer_packed(Xc, P)
        Xpad = np.zeros((T, D)); Xpad[:B] = X
        ct = cc.Encrypt(keys.publicKey, enc_pt(pack(Xpad / S), 0))
    else:
        ct = cc.Encrypt(keys.publicKey, enc_pt(pack(X / S), 0))
    need = {}
    boots, log, Xr = 0, [], X.copy()
    t_layer = time.perf_counter()
    if not A.packed:
        Xc = X.copy()
        for P in Ps:
            P["gelu_in_max"] = calibrate(Xc, P)
            Xc = ref_layer(Xc, P)
    halves = ((("attn", enc_attention_packed), ("ffn", enc_ffn_packed)) if A.packed
              else (("attn", enc_attention), ("ffn", enc_ffn)))
    for li, P in enumerate(Ps):
        for half, fn in halves:
            n0 = len(STAGES)
            if half in need and A.depth - lvl(ct) < need[half]:
                t0 = time.perf_counter()
                ct = cc.EvalBootstrap(ct); boots += 1
                log.append(f"bootstrap before layer {li} {half} ({time.perf_counter() - t0:.1f}s)")
            if A.depth - lvl(ct) < need.get(half, 0):
                sys.exit("not enough levels even after bootstrap")
            before = lvl(ct)
            ct = fn(ct, P, S)
            need[half] = lvl(ct) - before
            if li == 0:
                for s in STAGES[n0:]:
                    print(f"  {s['stage']:34s} {s['levels']:2d}", flush=True)
            if half == "ffn":
                Xr = (ref_layer_packed if A.packed else ref_layer)(Xr, P)
                full = dec(ct) * S
                got = full[:len(Xr)]
                err = float(np.abs(got - Xr)[:, :d].max())
                rng_ = float(np.abs(full / S).max())        # incl. garbage slots
                log.append(f"layer {li}: max|err| {err:.2e}  max|x/S| {rng_:.2f}  "
                           f"max|gelu in| {P['gelu_in_max']:.1f}  "
                           f"levels used {lvl(ct)}/{A.depth}")
                print(log[-1], flush=True)
    per_layer = need["attn"] + need["ffn"]
    res = {"packed": A.packed, "logn": A.logn, "slots": SLOTS, "D": D, "d": d, "hd": HD, "B": B, "T": T,
           "depth": A.depth, "bootstrap_depth": BOOT_DEPTH, "usable_after_boot": USABLE,
           "gelu_degree": A.gelu_degree, "levels_attention": need["attn"],
           "levels_ffn": need["ffn"], "levels_per_layer": per_layer,
           "cost_model_levels_per_layer": 18, "bootstraps": boots, "layers": A.layers,
           "stages_layer0": STAGES[:len(STAGES) // A.layers], "log": log,
           "setup_s": SETUP_S, "run_s": time.perf_counter() - t_layer}
    print(f"levels/layer: attention {need['attn']} + FFN {need['ffn']} = {per_layer} "
          f"(cost model: 18); usable per bootstrap window: {USABLE}")
    if A.out:
        json.dump(res, open(A.out, "w"), indent=2)


if __name__ == "__main__":
    main()
