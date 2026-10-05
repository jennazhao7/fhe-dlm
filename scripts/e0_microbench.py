"""E0 primitive microbenchmarks (CPU OpenFHE): the per-operation costs the E0
cost model multiplies by operation counts.

Packing-independent primitives only -- PCMM / attention / LM-head kernels are
built from these once the packing layout is fixed, and their counts come from
the analytical model, not from here. Measured per ring dimension N in
{2^16, 2^17} at 128-bit classical security with 59/60-bit moduli (the only
setting that bootstraps at >= 10 bits at 2^16, see openfhe_levels.py), at the
largest depth that ring admits:

  * add / ct-pt mult / ct-ct mult+relin / rotate / hoisted rotation, at several
    levels, since key-switching cost scales with the remaining RNS towers;
  * Chebyshev GELU on [-GELU_RANGE, GELU_RANGE] by degree: time, depth used,
    max error (the activation-domain question E1 has to answer);
  * bootstrap per level budget, sparse (BOOT_SLOTS) and full N/2 slots: setup,
    latency, levels left, precision, peak RSS.

Every ring and every bootstrap config runs in its own child process under
MEM_CAP_GB: OpenFHE's C++ bad_alloc aborts rather than raising, and keys stay
in a process-wide store, so isolation keeps one oversized config from taking
the sweep (or a shared box) down.
"""
import json, math, multiprocessing as mp, os, queue, random, resource, statistics, sys, time
from openfhe import (CCParamsCKKSRNS, GenCryptoContext, FHECKKSRNS,
                     SecretKeyDist, SecurityLevel, ScalingTechnique,
                     PKESchemeFeature)

SEC = SecurityLevel.HEStd_128_classic
SKD = SecretKeyDist.UNIFORM_TERNARY
RINGS = [1 << 16, 1 << 17]
SCALE, FIRST = 59, 60
D_MODEL = 768                  # GPT-2-small / MDLM-110M hidden size, for tokens-per-ct
GELU_RANGE = 8.0
GELU_DEGREES = [15, 31, 63, 119]
LEVEL_BUDGETS = [[2, 2], [3, 3], [4, 4]]
BOOT_SLOTS = 1 << 12
N_TEST = 64
REPS_FAST, REPS_SLOW = 20, 5
MEM_CAP_GB = 48


def params(depth, ring):
    p = CCParamsCKKSRNS()
    p.SetSecretKeyDist(SKD); p.SetSecurityLevel(SEC)
    p.SetScalingModSize(SCALE); p.SetFirstModSize(FIRST)
    p.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
    p.SetMultiplicativeDepth(depth); p.SetRingDim(ring)
    return p


def max_depth_at_ring(ring):
    best = None
    for depth in range(1, 120):
        p = params(depth, ring); p.SetRingDim(0)          # let OpenFHE pick N
        if GenCryptoContext(p).GetRingDimension() > ring:
            break
        best = depth
    return best


def context(depth, ring, fhe=False):
    cc = GenCryptoContext(params(depth, ring))
    feats = [PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH,
             PKESchemeFeature.LEVELEDSHE, PKESchemeFeature.ADVANCEDSHE]
    if fhe:
        feats.append(PKESchemeFeature.FHE)
    for f in feats:
        cc.Enable(f)
    return cc


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def gelu(x):
    return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))


def bench_ring(ring):
    depth = max_depth_at_ring(ring)
    cc = context(depth, ring)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalRotateKeyGen(keys.secretKey, [1])
    slots = ring // 2
    m = 2 * ring                                    # cyclotomic order, for hoisting
    rng = random.Random(0)
    x = [rng.uniform(-1, 1) for _ in range(slots)]
    r = {"ring_dim": ring, "depth": depth, "slots": slots,
         "tokens_per_ct_d768": slots // D_MODEL,
         "omp_threads": os.environ.get("OMP_NUM_THREADS", f"all ({os.cpu_count()})"),
         "by_level": [], "gelu": []}

    t0 = time.perf_counter(); pt = cc.MakeCKKSPackedPlaintext(x)
    r["encode_s"] = time.perf_counter() - t0
    r["encrypt_s"] = timed(lambda: cc.Encrypt(keys.publicKey, pt), REPS_SLOW)
    ct = cc.Encrypt(keys.publicKey, pt)
    r["decrypt_s"] = timed(lambda: cc.Decrypt(ct, keys.secretKey), REPS_SLOW)

    for lvl in sorted({0, depth // 2, depth - 2}):
        p_l = cc.MakeCKKSPackedPlaintext(x, 1, lvl)
        a = cc.Encrypt(keys.publicKey, p_l)
        b = cc.Encrypt(keys.publicKey, p_l)
        towers = len(a.GetElements()[0].GetAllElements()) if hasattr(
            a.GetElements()[0], "GetAllElements") else None
        row = {"level": lvl, "levels_left": depth - lvl}
        row["add_s"] = timed(lambda: cc.EvalAdd(a, b), REPS_FAST)
        row["mult_ct_pt_s"] = timed(lambda: cc.EvalMult(a, p_l), REPS_FAST)
        row["mult_ct_ct_relin_s"] = timed(lambda: cc.EvalMult(a, b), REPS_FAST)
        row["rotate_s"] = timed(lambda: cc.EvalRotate(a, 1), REPS_FAST)
        row["hoist_precompute_s"] = timed(lambda: cc.EvalFastRotationPrecompute(a), REPS_FAST)
        pre = cc.EvalFastRotationPrecompute(a)
        row["hoisted_rotate_s"] = timed(lambda: cc.EvalFastRotation(a, 1, m, pre), REPS_FAST)
        # Bytes on the wire: 2 polys x N coeffs x (remaining towers) x 8 B.
        # FLEXIBLEAUTO keeps one extra tower at the top of the chain.
        row["ct_bytes_est"] = 2 * ring * (depth + 2 - lvl) * 8
        if towers is not None:
            row["towers"] = towers
        r["by_level"].append(row)
        print(f"[e0] N=2^{ring.bit_length() - 1} level {lvl}: "
              + " ".join(f"{k}={v * 1e3:.1f}ms" for k, v in row.items() if k.endswith("_s")),
              flush=True)

    xs = [rng.uniform(-GELU_RANGE, GELU_RANGE) for _ in range(slots)]
    ct_g = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(xs))
    for deg in GELU_DEGREES:
        t0 = time.perf_counter()
        out = cc.EvalChebyshevFunction(gelu, ct_g, -GELU_RANGE, GELU_RANGE, deg)
        dt = time.perf_counter() - t0
        dec = cc.Decrypt(out, keys.secretKey); dec.SetLength(N_TEST * 16)
        err = max(abs(v - gelu(u)) for v, u in zip(dec.GetRealPackedValue(), xs))
        g = {"degree": deg, "seconds": dt, "depth_used": int(out.GetLevel() - ct_g.GetLevel()),
             "max_abs_error": err}
        r["gelu"].append(g)
        print(f"[e0] N=2^{ring.bit_length() - 1} GELU deg {deg}: {dt:.2f}s "
              f"depth {g['depth_used']} err {err:.2e}", flush=True)
    r["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    return r


def bench_boot(ring, budget, slots):
    depth = max_depth_at_ring(ring)
    cc = context(depth, ring, fhe=True)
    r = {"ring_dim": ring, "depth": depth, "level_budget": budget, "slots": slots,
         "bootstrap_depth": int(FHECKKSRNS.GetBootstrapDepth(budget, SKD))}
    t0 = time.perf_counter(); cc.EvalBootstrapSetup(budget, [0, 0], slots)
    keys = cc.KeyGen(); cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalBootstrapKeyGen(keys.secretKey, slots)
    r["setup_s"] = time.perf_counter() - t0
    x = [math.sin(0.1 * i) * 0.9 for i in range(N_TEST)]
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(x, 1, depth - 1, None, slots))
    t0 = time.perf_counter(); ct2 = cc.EvalBootstrap(ct); r["bootstrap_s"] = time.perf_counter() - t0
    r["levels_left_after"] = int(depth - ct2.GetLevel())
    dec = cc.Decrypt(ct2, keys.secretKey); dec.SetLength(N_TEST)
    err = max(abs(a - b) for a, b in zip(dec.GetRealPackedValue(), x))
    r["precision_bits"] = -math.log2(err) if err > 0 else None
    r["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    return r


def _child(q, fn, args):
    resource.setrlimit(resource.RLIMIT_AS, (MEM_CAP_GB << 30, MEM_CAP_GB << 30))
    try:
        q.put(("ok", fn(*args)))
    except Exception as e:
        q.put(("err", f"{type(e).__name__}: {e}"))


def run_isolated(fn, *args):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(q, fn, args))
    p.start()
    res = None
    # Drain before join: a child blocked writing a large result to the pipe
    # would otherwise never exit.
    while res is None and (p.is_alive() or not q.empty()):
        try:
            res = q.get(timeout=5)
        except queue.Empty:
            pass
    p.join()
    if res is None:
        return {"error": f"child died with exit code {p.exitcode} "
                         f"(likely bad_alloc at {MEM_CAP_GB} GB cap)"}
    return res[1] if res[0] == "ok" else {"error": res[1]}


def main():
    out = {"security": "HEStd_128_classic", "scaling_mod_size": SCALE,
           "first_mod_size": FIRST, "scaling_technique": "FLEXIBLEAUTO",
           "mem_cap_gb": MEM_CAP_GB, "rings": [], "bootstrap": []}
    for ring in RINGS:
        res = run_isolated(bench_ring, ring)
        out["rings"].append(res)
        if "error" in res:
            print(f"[e0] N=2^{ring.bit_length() - 1} primitives failed: {res['error']}", flush=True)
        depth = res.get("depth") or max_depth_at_ring(ring)
        for budget in LEVEL_BUDGETS:
            if depth - int(FHECKKSRNS.GetBootstrapDepth(budget, SKD)) <= 0:
                # No room above the bootstrap: OpenFHE does not refuse, it returns
                # garbage (a "40-bit" refresh) or corrupts the heap. Don't run it.
                out["bootstrap"].append({"ring_dim": ring, "level_budget": budget,
                                         "error": "skipped: bootstrap depth >= total depth"})
                print(f"[e0] N=2^{ring.bit_length() - 1} boot {budget}: skipped (no levels left)",
                      flush=True)
                continue
            for slots in (BOOT_SLOTS, ring // 2):
                b = run_isolated(bench_boot, ring, budget, slots)
                b.update({"ring_dim": ring, "level_budget": budget, "slots": slots})
                out["bootstrap"].append(b)
                msg = b.get("error") or (
                    f"setup={b['setup_s']:.1f}s boot={b['bootstrap_s']:.2f}s "
                    f"left={b['levels_left_after']} prec={b['precision_bits']:.1f}b "
                    f"rss={b['peak_rss_gb']:.1f}GB")
                print(f"[e0] N=2^{ring.bit_length() - 1} boot {budget} slots={slots}: {msg}",
                      flush=True)
        json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results/e0_microbench.json",
                            "w"), indent=2)
    print("[e0] done", flush=True)


if __name__ == "__main__":
    main()
