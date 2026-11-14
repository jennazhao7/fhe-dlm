"""E0 / HANDOFF §6.6: is `depth - ct.GetLevel()` after EvalBootstrap (the
`levels_left_after` field of e0_microbench.py) really the number of usable
multiplicative levels, or one too many (the suspected FLEXIBLEAUTO extra
tower)?

Same scheme settings as the sweep (59/60, FLEXIBLEAUTO, UNIFORM_TERNARY) at a
toy ring (2^12, HEStd_NotSet) so it runs in seconds anywhere. Level
accounting does not depend on N. After a bootstrap, keep multiplying by an
encoded all-ones plaintext (one rescale = one level each, without the error
amplification of repeated squaring) and count how many products still
decrypt to within 1e-2; the chain ends with OpenFHE refusing to go further
or decryption blowing up.
"""
import json, math, sys
import numpy as np
from openfhe import (CCParamsCKKSRNS, GenCryptoContext, SecretKeyDist, SecurityLevel,
                     ScalingTechnique, PKESchemeFeature, FHECKKSRNS)

def run(budget, extra):
    skd = SecretKeyDist.UNIFORM_TERNARY
    depth = int(FHECKKSRNS.GetBootstrapDepth(budget, skd)) + extra
    p = CCParamsCKKSRNS()
    p.SetSecretKeyDist(skd); p.SetSecurityLevel(SecurityLevel.HEStd_NotSet)
    p.SetRingDim(1 << 12); p.SetScalingModSize(59); p.SetFirstModSize(60)
    p.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO); p.SetMultiplicativeDepth(depth)
    cc = GenCryptoContext(p)
    for f in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH, PKESchemeFeature.LEVELEDSHE,
              PKESchemeFeature.ADVANCEDSHE, PKESchemeFeature.FHE):
        cc.Enable(f)
    slots = cc.GetRingDimension() // 2
    cc.EvalBootstrapSetup(budget, [0, 0], slots)
    k = cc.KeyGen(); cc.EvalMultKeyGen(k.secretKey); cc.EvalBootstrapKeyGen(k.secretKey, slots)
    x = np.random.default_rng(0).uniform(-1.0, 1.0, slots)
    ct = cc.EvalBootstrap(cc.Encrypt(k.publicKey, cc.MakeCKKSPackedPlaintext(x.tolist(), 1, depth - 1)))
    reported = depth - ct.GetLevel()
    want, ok, errs = x.copy(), 0, []
    for i in range(reported + 2):
        try:
            ones = cc.MakeCKKSPackedPlaintext([1.0] * slots, 1, ct.GetLevel())
            ct = cc.EvalMult(ct, ones)
            d = cc.Decrypt(ct, k.secretKey); d.SetLength(slots)
            err = float(np.abs(np.array(d.GetRealPackedValue()) - want).max())
        except Exception as e:
            errs.append(f"mult {i + 1}: {str(e).splitlines()[0][-120:]}"); break
        errs.append(f"mult {i + 1}: err {err:.1e} level {ct.GetLevel()}")
        if err > 1e-2:
            break
        ok += 1
    return {"budget": budget, "depth": depth, "levels_left_after": reported,
            "usable_mults": ok, "trace": errs}

if __name__ == "__main__":
    out = [run(b, e) for b in ([2, 2], [4, 4]) for e in (3, 8)]
    for r in out:
        print(f"budget {r['budget']} depth {r['depth']}: reported {r['levels_left_after']}, "
              f"usable {r['usable_mults']}  | {r['trace'][-2:]}", flush=True)
    if len(sys.argv) > 1:
        json.dump(out, open(sys.argv[1], "w"), indent=2)
