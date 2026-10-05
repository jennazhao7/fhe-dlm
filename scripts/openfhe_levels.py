"""Measure, not estimate: how many multiplicative levels are usable between
bootstraps for CKKS at 128-bit classical security, at ring dimensions 2^16
and 2^17.

Two numbers decide the FHE-DLM budget:
  * the total multiplicative depth a parameter set can carry while its ring
    dimension stays at the target under HEStd_128_classic, and
  * the depth bootstrapping itself consumes (GetBootstrapDepth).
Usable levels between bootstraps is the difference. Both depend on the scaling
modulus size and the bootstrap level budget, so we sweep them: at 59-bit
scaling only 20 levels fit at 2^16 while bootstrapping alone needs 22, i.e. the
"obvious" parameters leave nothing. For every configuration with a positive
budget we also run one real bootstrap end-to-end, time it, and measure its
precision, since a level count that cannot be refreshed accurately is not a
budget: "best" only considers configs at or above MIN_PRECISION_BITS.

The live bootstrap runs with BOOT_SLOTS sparse slots, not the full N/2: the
depth numbers do not depend on the slot count, but key material does, and a
full-slot [1, 1] budget needs tens of thousands of rotation keys (it OOM-killed
a 125 GB workstation). Timings are therefore for sparse packing. Each live
bootstrap runs in its own child process under MEM_CAP_GB: OpenFHE's C++
bad_alloc aborts the process rather than raising, so isolation is the only way
one oversized config cannot take the whole sweep down with it.
"""
import json, math, multiprocessing as mp, queue, resource, sys, time
from openfhe import (CCParamsCKKSRNS, GenCryptoContext, FHECKKSRNS,
                     SecretKeyDist, SecurityLevel, ScalingTechnique,
                     PKESchemeFeature)

SEC = SecurityLevel.HEStd_128_classic
SKD = SecretKeyDist.UNIFORM_TERNARY
TARGET_RINGS = [1 << 16, 1 << 17]
# (scaling_mod, first_mod). EvalBootstrap requires first - scaling to be at most
# its correction factor (7 here), so a 10-bit gap like (50, 60) cannot bootstrap.
SCALING = [(59, 60), (50, 55), (45, 50), (40, 45)]
LEVEL_BUDGETS = [[4, 4], [3, 3], [2, 2], [1, 1]]
MIN_PRECISION_BITS = 10        # below this a refreshed ciphertext is not usable
N_TEST = 64                    # slots checked for bootstrap precision
BOOT_SLOTS = 1 << 12           # sparse slots for the live bootstrap (see above)
MEM_CAP_GB = 48                # per child; fail that config instead of OOM-killing a shared box


def params(scale, first, depth, ring=None):
    p = CCParamsCKKSRNS()
    p.SetSecretKeyDist(SKD); p.SetSecurityLevel(SEC)
    p.SetScalingModSize(scale); p.SetFirstModSize(first)
    p.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
    p.SetMultiplicativeDepth(depth)
    if ring is not None:
        p.SetRingDim(ring)
    return p


def max_depth_at_ring(scale, first, ring):
    """Largest total depth whose auto-selected ring dimension is still <= ring.
    Above that OpenFHE must grow the ring to keep 128-bit security, which
    changes the cost model entirely -- so this is the real ceiling."""
    best = None
    for depth in range(1, 120):
        try:
            n = GenCryptoContext(params(scale, first, depth)).GetRingDimension()
        except Exception as e:
            print(f"[fhe] scale={scale} depth {depth}: {type(e).__name__}", flush=True)
            break
        if n > ring:
            break
        best = depth
    return best


def live_bootstrap(scale, first, depth, budget, ring):
    resource.setrlimit(resource.RLIMIT_AS, (MEM_CAP_GB << 30, MEM_CAP_GB << 30))
    cc = GenCryptoContext(params(scale, first, depth, ring))
    for f in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH,
              PKESchemeFeature.LEVELEDSHE, PKESchemeFeature.ADVANCEDSHE,
              PKESchemeFeature.FHE):
        cc.Enable(f)
    slots = BOOT_SLOTS
    r = {"ring_dim": int(cc.GetRingDimension()), "slots": int(slots)}
    t0 = time.time(); cc.EvalBootstrapSetup(budget, [0, 0], slots)
    keys = cc.KeyGen(); cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalBootstrapKeyGen(keys.secretKey, slots)
    r["setup_seconds"] = time.time() - t0
    x = [math.sin(0.1 * i) * 0.9 for i in range(N_TEST)]
    # Encrypt at the last level so the bootstrap genuinely has to refresh it.
    pt = cc.MakeCKKSPackedPlaintext(x, 1, depth - 1, None, slots)
    ct = cc.Encrypt(keys.publicKey, pt)
    t0 = time.time(); ct2 = cc.EvalBootstrap(ct); r["bootstrap_seconds"] = time.time() - t0
    r["levels_remaining_after_bootstrap"] = int(depth - ct2.GetLevel())
    dec = cc.Decrypt(ct2, keys.secretKey)
    dec.SetLength(N_TEST)
    err = max(abs(a - b) for a, b in zip(dec.GetRealPackedValue(), x))
    r["max_abs_error"] = err
    r["precision_bits"] = -math.log2(err) if err > 0 else None
    r["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    return r


def _child(q, args):
    try:
        q.put(("ok", live_bootstrap(*args)))
    except Exception as e:
        q.put(("err", f"{type(e).__name__}: {e}"))


def run_isolated(*args):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(q, args))
    p.start()
    res = None
    # Drain before join: a child blocked writing its result to the pipe would
    # otherwise never exit.
    while res is None and (p.is_alive() or not q.empty()):
        try:
            res = q.get(timeout=5)
        except queue.Empty:
            pass
    p.join()
    if res is not None:
        status, val = res
        if status == "ok":
            return val
        raise RuntimeError(val)
    raise RuntimeError(f"child died with exit code {p.exitcode} (likely bad_alloc at {MEM_CAP_GB} GB cap)")


def main():
    out = {"target_ring_dims": TARGET_RINGS, "security": "HEStd_128_classic",
           "scaling_technique": "FLEXIBLEAUTO", "boot_slots": BOOT_SLOTS,
           "min_precision_bits": MIN_PRECISION_BITS, "configs": []}
    for ring in TARGET_RINGS:
        for scale, first in SCALING:
            max_depth = max_depth_at_ring(scale, first, ring)
            print(f"[fhe] N=2^{ring.bit_length() - 1} scale={scale} first={first}: "
                  f"max total depth = {max_depth}", flush=True)
            for budget in LEVEL_BUDGETS:
                boot_depth = int(FHECKKSRNS.GetBootstrapDepth(budget, SKD))
                usable = (max_depth - boot_depth) if max_depth is not None else None
                c = {"target_ring_dim": ring, "scaling_mod_size": scale,
                     "first_mod_size": first, "level_budget": budget,
                     "max_total_depth_at_ring": max_depth, "bootstrap_depth": boot_depth,
                     "usable_levels_between_bootstraps": usable}
                print(f"[fhe]   budget={budget} bootstrap={boot_depth} -> USABLE {usable}",
                      flush=True)
                if usable is not None and usable > 0 and budget[0] == 1:
                    # One-level linear transforms need ~one rotation key per slot:
                    # even at BOOT_SLOTS this blows past MEM_CAP_GB. Record, don't run.
                    c["bootstrap_run_error"] = "skipped: [1, 1] key material exceeds memory cap"
                    print("[fhe]     live bootstrap skipped (key material too large)", flush=True)
                elif usable is not None and usable > 0:
                    try:
                        c.update(run_isolated(scale, first, max_depth, budget, ring))
                        print(f"[fhe]     live: setup={c['setup_seconds']:.1f}s "
                              f"bootstrap={c['bootstrap_seconds']:.2f}s "
                              f"levels_after={c['levels_remaining_after_bootstrap']} "
                              f"precision={c['precision_bits']:.1f} bits "
                              f"rss={c['peak_rss_gb']:.1f}GB", flush=True)
                    except Exception as e:
                        c["bootstrap_run_error"] = f"{type(e).__name__}: {e}"
                        print(f"[fhe]     live bootstrap failed: {c['bootstrap_run_error']}",
                              flush=True)
                out["configs"].append(c)

    out["best"] = {}
    for ring in TARGET_RINGS:
        ok = [c for c in out["configs"] if c["target_ring_dim"] == ring
              and (c.get("precision_bits") or 0) >= MIN_PRECISION_BITS]
        if not ok:
            print(f"[fhe] N=2^{ring.bit_length() - 1}: no config reaches "
                  f"{MIN_PRECISION_BITS} bits", flush=True)
            continue
        best = max(ok, key=lambda c: (c["levels_remaining_after_bootstrap"], c["precision_bits"]))
        out["best"][str(ring)] = {k: best[k] for k in (
            "scaling_mod_size", "first_mod_size", "level_budget",
            "levels_remaining_after_bootstrap", "precision_bits", "bootstrap_seconds")}
        print(f"[fhe] N=2^{ring.bit_length() - 1} most levels at >= {MIN_PRECISION_BITS} bits: "
              f"{out['best'][str(ring)]}", flush=True)

    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results/openfhe_levels.json", "w"),
              indent=2)
    print("[fhe] done", flush=True)


if __name__ == "__main__":
    main()
