"""E0: measure the real encrypted matvec, to calibrate e0_cost_model.py's
op-sum estimate (which charges every rotation / ptm / add at its standalone
microbenchmark cost).

Layout as in the cost model: d padded to D=1024, T = slots/D tokens per
ciphertext, interleaved slot = dim*T + token, so rotating by i*T maps dim j to
j+i (mod D) for every token at once. y = W x via BSGS over the D diagonals,
n1 = n2 = 32:

    y = sum_g rot_{g n1 T}( sum_b pt(rot_{-g n1 T} diag_{g n1 + b}) * rot_{b T}(x) )

Baby rotations of x are hoisted (one EvalFastRotationPrecompute, 31
EvalFastRotation). Diagonal plaintexts are encoded on the fly at the
ciphertext's level -- pre-encoding all 1024 at N=2^17 costs ~24 GB per matrix
-- and encode time is reported separately so both regimes can be read off.

Shapes: 1->1 (attention out-proj), 1->3 (QKV, FFN up), 3->1 (FFN down), each
at the post-bootstrap level and mid-window. The LM head (1->50 blocks) is
extrapolated from the per-output-block cost of the 1->3 run. Output is checked
against numpy on every token.
"""
import json, math, resource, sys, time
import numpy as np
from openfhe import (CCParamsCKKSRNS, GenCryptoContext, SecretKeyDist,
                     SecurityLevel, ScalingTechnique, PKESchemeFeature)

RING, DEPTH, USABLE = 1 << 17, 43, 21          # from openfhe_levels / e0_microbench
D, N1 = 1024, 32
N2 = D // N1
SHAPES = [(1, 1), (1, 3), (3, 1)]
LEVELS = [DEPTH - USABLE - 1, DEPTH - USABLE // 2 - 1]
VOCAB_BLOCKS = math.ceil(50257 / D)
MEM_CAP_GB = 60
resource.setrlimit(resource.RLIMIT_AS, (MEM_CAP_GB << 30, MEM_CAP_GB << 30))


def setup():
    p = CCParamsCKKSRNS()
    p.SetSecretKeyDist(SecretKeyDist.UNIFORM_TERNARY)
    p.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
    p.SetScalingModSize(59); p.SetFirstModSize(60)
    p.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
    p.SetMultiplicativeDepth(DEPTH); p.SetRingDim(RING)
    cc = GenCryptoContext(p)
    for f in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH, PKESchemeFeature.LEVELEDSHE):
        cc.Enable(f)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    T = cc.GetRingDimension() // 2 // D
    rots = [b * T for b in range(1, N1)] + [g * N1 * T for g in range(1, N2)]
    t0 = time.perf_counter()
    cc.EvalRotateKeyGen(keys.secretKey, rots)
    return cc, keys, T, time.perf_counter() - t0


def interleave(X, T):
    """X: (T, D) tokens x dims -> slot vector dim*T + token."""
    return X.T.reshape(-1)


def diag_slots(W, i, shift, T):
    """Slot vector of diagonal i of W (out j <- in j+i), pre-rotated by -shift dims."""
    j = np.arange(D)
    d = W[j, (j + i) % D]
    d = np.roll(d, shift)                       # rot_{-shift*T} on the slot vector
    return np.repeat(d, T)


def matvec(cc, cts, Ws, T, level, m):
    """cts: list of input cts (din blocks); Ws[o][i]: D x D block. Returns out cts + timing."""
    t = {"rotate_s": 0.0, "encode_s": 0.0, "mult_add_s": 0.0}
    t0 = time.perf_counter()
    babies = []
    for ct in cts:
        pre = cc.EvalFastRotationPrecompute(ct)
        babies.append([ct] + [cc.EvalFastRotation(ct, b * T, m, pre) for b in range(1, N1)])
    t["rotate_s"] += time.perf_counter() - t0
    outs = []
    for o in range(len(Ws)):
        acc_out = None
        for g in range(N2):
            acc = None
            for i, bab in enumerate(babies):
                for b in range(N1):
                    t1 = time.perf_counter()
                    pt = cc.MakeCKKSPackedPlaintext(
                        diag_slots(Ws[o][i], g * N1 + b, g * N1, T).tolist(), 1, level)
                    t2 = time.perf_counter()
                    term = cc.EvalMult(bab[b], pt)
                    acc = term if acc is None else cc.EvalAdd(acc, term)
                    t3 = time.perf_counter()
                    t["encode_s"] += t2 - t1; t["mult_add_s"] += t3 - t2
            t1 = time.perf_counter()
            if g:
                acc = cc.EvalRotate(acc, g * N1 * T)
            acc_out = acc if acc_out is None else cc.EvalAdd(acc_out, acc)
            t["rotate_s"] += time.perf_counter() - t1
        outs.append(acc_out)
    t["total_s"] = sum(t.values())
    t["compute_only_s"] = t["rotate_s"] + t["mult_add_s"]
    return outs, t


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "results/e0_matvec.json"
    cc, keys, T, keygen_s = setup()
    m = 2 * RING
    rng = np.random.default_rng(0)
    res = {"ring": RING, "depth": DEPTH, "D": D, "T": T, "n1": N1, "n2": N2,
           "rot_keygen_s": keygen_s, "runs": []}
    print(f"[mv] N=2^17 T={T} rot keys {keygen_s:.0f}s", flush=True)
    for level in LEVELS:
        for din, dout in SHAPES:
            X = [rng.uniform(-1, 1, (T, D)) for _ in range(din)]
            Ws = [[rng.uniform(-1, 1, (D, D)) / math.sqrt(D * din) for _ in range(din)]
                  for _ in range(dout)]
            cts = [cc.Encrypt(keys.publicKey,
                              cc.MakeCKKSPackedPlaintext(interleave(x, T).tolist(), 1, level))
                   for x in X]
            outs, t = matvec(cc, cts, Ws, T, level, m)
            err = 0.0
            for o, ct in enumerate(outs):
                dec = cc.Decrypt(ct, keys.secretKey); dec.SetLength(D * T)
                got = np.array(dec.GetRealPackedValue()).reshape(D, T).T
                want = sum(X[i] @ Ws[o][i].T for i in range(din))
                err = max(err, float(np.abs(got - want).max()))
            r = {"level": level, "din": din, "dout": dout, **t, "max_abs_error": err,
                 "levels_used": int(outs[0].GetLevel() - level)}
            res["runs"].append(r)
            print(f"[mv] level {level} {din}->{dout}: total {t['total_s']:.1f}s "
                  f"(rot {t['rotate_s']:.1f}, encode {t['encode_s']:.1f}, "
                  f"mult+add {t['mult_add_s']:.1f}) err {err:.1e}", flush=True)
    # LM head: per-output-block cost from the 1->3 run, plus its baby steps once.
    for level in LEVELS:
        r13 = next(r for r in res["runs"] if r["level"] == level and (r["din"], r["dout"]) == (1, 3))
        r11 = next(r for r in res["runs"] if r["level"] == level and (r["din"], r["dout"]) == (1, 1))
        per_block = (r13["total_s"] - r11["total_s"]) / 2
        per_block_c = (r13["compute_only_s"] - r11["compute_only_s"]) / 2
        res.setdefault("lm_head_extrapolated", []).append({
            "level": level, "blocks": VOCAB_BLOCKS,
            "total_s": r11["total_s"] + per_block * (VOCAB_BLOCKS - 1),
            "compute_only_s": r11["compute_only_s"] + per_block_c * (VOCAB_BLOCKS - 1)})
    res["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    json.dump(res, open(out_path, "w"), indent=2)
    print(f"[mv] LM head (extrapolated): {res['lm_head_extrapolated']}", flush=True)
    print(f"[mv] peak rss {res['peak_rss_gb']:.1f} GB; done", flush=True)


if __name__ == "__main__":
    main()
