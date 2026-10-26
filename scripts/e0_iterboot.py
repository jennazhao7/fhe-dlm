"""E0: how many bits does iterative (multi-pass) CKKS bootstrapping buy at the
E5 parameter set, and what does it cost in time and levels?

Why this matters: the E5 bootstraps deliver ~8.5-9.2 bits at full slots
(N=2^17, [4,4], 59/60), and MDLM-OWT measured without its position-0 sink goes
from masked NLL 2.29 to 2.56 (gen-PPL 90 -> 929) at 2^-9 noise, but only to
2.33 at 2^-10. So a few extra bits per refresh may be worth more than any
amount of noise-aware training -- if they are cheap enough.

Iterative bootstrapping (Bae et al., "Meta-BTS", as implemented in OpenFHE):
bootstrap once, measure the residual against the input, scale the residual up
by 2^precision, bootstrap that, scale back down and add the correction.
OpenFHE exposes it as EvalBootstrap(ct, numIterations, precision), where
`precision` must be (the floor of) the bits a SINGLE pass actually achieves on
this parameter set -- passing more overflows the correction. Each extra
iteration consumes one more level and roughly one more bootstrap of time.

Measured per slot count, all in one key set:
  * 1 iteration: time, levels left, max-error bits and RMS-error bits over
    EVERY slot (not a 64-slot sample: the 9.2-bit full-slot figure is a max,
    and a max over 64 slots understates the tail);
  * k = 2, 3 iterations with precision = floor(1-pass max-error bits).
Usable levels follow the confirmed off-by-one (results/e0_levels_offbyone.json):
usable = depth - GetLevel() - 1.

Each slot count runs in its own spawned child under MEM_CAP_GB: OpenFHE's C++
bad_alloc aborts the process instead of raising, and a full-slot key set at
2^17 measured 33-39 GB RSS on tjws.

    python scripts/e0_iterboot.py results/e0_iterboot.json [--slots 4096 65536] [--manual]
"""
import argparse, json, math, multiprocessing as mp, queue, resource, sys, time

RING = 1 << 17
SCALE, FIRST = 59, 60          # E5 parameter set (HANDOFF §1)
BUDGET = [4, 4]
DEPTH = 43                     # max total depth at 2^17 for 59/60 (openfhe_levels.json)
ITERATIONS = [1, 2, 3]
MEM_CAP_GB = 60                # tjws has 125 GB and no per-user limit; leave headroom
SEED = 0


def params(depth):
    from openfhe import (CCParamsCKKSRNS, SecretKeyDist, SecurityLevel,
                         ScalingTechnique)
    p = CCParamsCKKSRNS()
    p.SetSecretKeyDist(SecretKeyDist.UNIFORM_TERNARY)
    p.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
    p.SetScalingModSize(SCALE); p.SetFirstModSize(FIRST)
    p.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
    p.SetMultiplicativeDepth(depth)
    p.SetRingDim(RING)
    return p


def bits(errs):
    m = max(errs)
    rms = math.sqrt(sum(e * e for e in errs) / len(errs))
    return (-math.log2(m) if m > 0 else None,
            -math.log2(rms) if rms > 0 else None, m, rms)


def run(slots, manual=False):
    import random
    from openfhe import GenCryptoContext, FHECKKSRNS, SecretKeyDist, PKESchemeFeature
    resource.setrlimit(resource.RLIMIT_AS, (MEM_CAP_GB << 30, MEM_CAP_GB << 30))

    boot_depth = int(FHECKKSRNS.GetBootstrapDepth(BUDGET, SecretKeyDist.UNIFORM_TERNARY))
    need = boot_depth + (max(ITERATIONS) - 1) + 1
    if DEPTH < need:
        raise RuntimeError(f"depth {DEPTH} < bootstrap {boot_depth} + extra iterations")

    cc = GenCryptoContext(params(DEPTH))
    for f in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH,
              PKESchemeFeature.LEVELEDSHE, PKESchemeFeature.ADVANCEDSHE,
              PKESchemeFeature.FHE):
        cc.Enable(f)
    r = {"ring_dim": int(cc.GetRingDimension()), "slots": slots,
         "depth": DEPTH, "bootstrap_depth": boot_depth, "budget": BUDGET,
         "scaling_mod": SCALE, "first_mod": FIRST}

    t0 = time.time()
    cc.EvalBootstrapSetup(BUDGET, [0, 0], slots)
    keys = cc.KeyGen(); cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalBootstrapKeyGen(keys.secretKey, slots)
    r["setup_seconds"] = time.time() - t0
    print(f"[iterboot] slots={slots}: keys ready in {r['setup_seconds']:.0f}s", flush=True)

    # Uniform on [-1, 1]: the residual stream is normalised by S so |x/S| < 1
    # (HANDOFF §9.2a), and uniform exercises the whole range rather than a
    # smooth pattern a bootstrap might happen to favour.
    rng = random.Random(SEED)
    x = [rng.uniform(-1.0, 1.0) for _ in range(slots)]
    # Encrypt at the last level, as openfhe_levels.py does, so the bootstrap
    # genuinely has to refresh it.
    pt = cc.MakeCKKSPackedPlaintext(x, 1, DEPTH - 1, None, slots)
    ct = cc.Encrypt(keys.publicKey, pt)

    def measure(ct_out):
        dec = cc.Decrypt(ct_out, keys.secretKey)
        dec.SetLength(slots)
        errs = [abs(a - b) for a, b in zip(dec.GetRealPackedValue(), x)]
        b_max, b_rms, m, rms = bits(errs)
        lvl = int(ct_out.GetLevel())
        return {"max_err_bits": b_max, "rms_err_bits": b_rms,
                "max_abs_err": m, "rms_abs_err": rms,
                "levels_left_raw": DEPTH - lvl, "usable_levels": DEPTH - lvl - 1}

    r["iterations"] = {}
    precision = None
    for k in ITERATIONS:
        try:
            t0 = time.time()
            if k == 1:
                out = cc.EvalBootstrap(ct)
            else:
                out = cc.EvalBootstrap(ct, k, precision)
            dt = time.time() - t0
            m = measure(out)
            m["seconds"] = dt
            if k == 1:
                # OpenFHE's example passes floor(bits of ONE pass). Using the
                # max-error bits (the worst slot) keeps the correction from
                # overflowing on that slot.
                precision = int(math.floor(m["max_err_bits"]))
                r["precision_arg"] = precision
            r["iterations"][str(k)] = m
            print(f"[iterboot] slots={slots} k={k}: {dt:6.1f}s  "
                  f"max {m['max_err_bits']:.2f} bits  rms {m['rms_err_bits']:.2f} bits  "
                  f"usable levels {m['usable_levels']}", flush=True)
        except Exception as e:
            r["iterations"][str(k)] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[iterboot] slots={slots} k={k}: FAILED {e}", flush=True)
    if manual and precision is not None:
        # Same two-pass algorithm, but built ONLY from ops FIDESlib's GPU path
        # also exposes (single EvalBootstrap, EvalSub, EvalAdd, EvalMult by a
        # double). FIDESlib accepts numIterations but drops it on the GPU, so
        # this is the sequence the GPU port will run; matching the built-in k=2
        # here is what makes that port trustworthy.
        #   b1   = Boot(ct)                 m + e1
        #   diff = ct - b1                  -e1   (EvalSub level-matches b1 down to ct)
        #   diff *= 2^p by p self-adds      -e1*2^p, |.| < 1 because p = floor(bits(e1))
        #   b2   = Boot(diff)               -e1*2^p + e2
        #   out  = b1 + b2 * 2^-p           m + e2*2^-p      (the extra level)
        try:
            t0 = time.time()
            b1 = cc.EvalBootstrap(ct)
            diff = cc.EvalSub(ct, b1)
            for _ in range(precision):
                diff = cc.EvalAdd(diff, diff)
            b2 = cc.EvalBootstrap(diff)
            out = cc.EvalAdd(b1, cc.EvalMult(b2, 2.0 ** -precision))
            dt = time.time() - t0
            m = measure(out)
            m["seconds"] = dt
            r["manual_2pass"] = m
            print(f"[iterboot] slots={slots} manual-2: {dt:6.1f}s  "
                  f"max {m['max_err_bits']:.2f} bits  rms {m['rms_err_bits']:.2f} bits  "
                  f"usable levels {m['usable_levels']}", flush=True)
        except Exception as e:
            r["manual_2pass"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[iterboot] slots={slots} manual-2: FAILED {e}", flush=True)

    r["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    return r


def _child(q, slots, manual):
    try:
        q.put(("ok", run(slots, manual)))
    except Exception as e:
        q.put(("err", f"{type(e).__name__}: {e}"))


def run_isolated(slots, manual=False):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(q, slots, manual))
    p.start()
    res = None
    # Drain before join: a child blocked on the pipe never exits.
    while res is None and (p.is_alive() or not q.empty()):
        try:
            res = q.get(timeout=5)
        except queue.Empty:
            pass
    p.join()
    if res is None:
        return {"slots": slots,
                "error": f"child died, exit {p.exitcode} (bad_alloc at {MEM_CAP_GB} GB?)"}
    status, val = res
    return val if status == "ok" else {"slots": slots, "error": val}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="results/e0_iterboot.json")
    # Sparse first: same depths and precision behaviour, ~half the time, so a
    # broken binding or parameter shows up before the 40 GB full-slot run.
    ap.add_argument("--slots", type=int, nargs="+", default=[4096, RING // 2])
    ap.add_argument("--manual", action="store_true",
                    help="also run the two-pass built from FIDESlib-available ops")
    a = ap.parse_args()
    out = {"ring_dim": RING, "scaling_mod": SCALE, "first_mod": FIRST,
           "budget": BUDGET, "depth": DEPTH, "iterations": ITERATIONS,
           "security": "HEStd_128_classic", "secret": "UNIFORM_TERNARY",
           "scaling_technique": "FLEXIBLEAUTO", "runs": []}
    for s in a.slots:
        out["runs"].append(run_isolated(s, a.manual))
        json.dump(out, open(a.out, "w"), indent=2)   # checkpoint per slot count
    print(f"[iterboot] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
