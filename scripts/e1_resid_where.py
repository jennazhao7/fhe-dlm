"""Where does an e1_kd student's residual stream blow up during generation?

Job 1660384's stage-A eval reported max |block output| 1.6e8, but on the CPU
every eval chunk (<= 409) and 16 sampled sequences (<= 520) stayed bounded.
This replays evaluate()'s generation exactly (pin0 sampler, 64 samples, 128
steps, seed 11, on the GPU) and logs every forward: max |block output|, the
block, sample, position and token holding it, and how many entries exceed
1e3. The worst forwards are re-run (a) with that sample alone (batch
dependence) and (b) on the CPU (GPU vs CPU numerics; float64 is not possible, FixedNorm
casts to float32), and their contexts are
decoded.

    python scripts/e1_resid_where.py --ckpt ckpt/e1_pilot2/stageA.pt --stage A \
        --out results/plain/e1_resid_where.json
"""
import argparse, json, os, sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import e1_kd as E             # noqa: E402
import plain_probe as pp      # noqa: E402
import approx_probe as ap     # noqa: E402


def build(ck, stage, dev):
    m = pp.load_model(dev)
    H, nb = pp.blocks(m)[0].n_heads, len(pp.blocks(m))
    E.patch_attention(m)
    E.convert(m, {"norm_c": torch.ones(2 * nb + 1, 2), "attn_b": torch.zeros(nb, H),
                  "attn_z": torch.ones(nb, H, 2)})      # shapes only; the checkpoint overwrites them
    m.load_state_dict(torch.load(ck, map_location=dev))
    m.to(dev).eval()                                   # the placeholder constants were made on the CPU
    E.set_stage(m, stage, 1.0, train=False)
    return m


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--ckpt", required=True)
    a_.add_argument("--stage", default="A")
    a_.add_argument("--n", type=int, default=64)
    a_.add_argument("--steps", type=int, default=128)
    a_.add_argument("--seed", type=int, default=11)
    a_.add_argument("--bad", type=float, default=1e3)
    a_.add_argument("--out", required=True)
    a = a_.parse_args()

    m = build(a.ckpt, a.stage, pp.DEV)
    rec = []
    for i, b in enumerate(pp.blocks(m)):
        b.register_forward_hook(lambda mod, inp, o, i=i: rec.append(o.detach().abs().amax(-1)))   # (b, L)

    log, inputs = [], []
    real = pp.forward_logits
    ap._exact_fwd = real

    def fl(mm, x):
        rec.clear()
        out = ap.forward_logits_checked(mm, x)
        per = torch.stack(rec)                             # (blocks, b, L)
        mx = float(per.max())
        blk, bi, pos = (int(v) for v in torch.unravel_index(per.argmax(), per.shape))
        log.append({"fwd": len(log), "max": mx, "block": blk, "sample": bi, "pos": pos,
                    "masked": bool(x[bi, pos] == pp.MASK), "token": int(x[bi, pos]),
                    "n_masked": int((x[bi] == pp.MASK).sum()),
                    "n_tok_bad": int((per.amax(0) > a.bad).sum()),
                    "samples_bad": sorted({int(s) for s in (per.amax((0, 2)) > a.bad).nonzero()[:, 0]}),
                    "block_max": [float(v) for v in per.amax((1, 2))]})
        if mx > a.bad:
            inputs.append((len(log) - 1, x.detach().clone()))
        return out

    pp.PIN0 = True
    pp.forward_logits = fl
    err = None
    try:
        with torch.no_grad():
            xs = pp.sample(m, a.n, 256, a.steps, a.seed)
    except FloatingPointError as e:
        err, xs = str(e), None
    pp.forward_logits = real
    bad = [r for r in log if r["max"] > a.bad]
    res = {"args": vars(a), "error": err, "n_forwards": len(log), "max": max(r["max"] for r in log),
           "n_forwards_bad": len(bad), "samples_ever_bad": sorted({s for r in bad for s in r["samples_bad"]}),
           "log": log}
    print(json.dumps({k: v for k, v in res.items() if k != "log"}), flush=True)
    for r in bad[:20]:
        print(json.dumps({k: v for k, v in r.items() if k != "block_max"}), flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(pp.snapshot("openai-community/gpt2"))
    dec = lambda ids: tok.decode([t if t != pp.MASK else tok.convert_tokens_to_ids("_") for t in ids])
    worst = sorted(inputs, key=lambda t: -log[t[0]]["max"])[:3]
    m64 = None
    res["worst"] = []
    for k, x in worst:
        r = log[k]; bi = r["sample"]
        x1 = x[bi:bi + 1]
        with torch.no_grad():
            rec.clear(); E.fwd(m, x1); alone = float(torch.stack(rec).max())
            if m64 is None:
                m64 = build(a.ckpt, a.stage, "cpu")
                rec64 = []
                for b in pp.blocks(m64):
                    b.register_forward_hook(lambda mod, inp, o: rec64.append(float(o.detach().abs().max())))
            rec64.clear()
            try:
                E.fwd(m64, x1.cpu()); f64 = max(rec64)
            except Exception as e:                      # report, don't die
                f64 = f"error: {str(e)[:120]}"
        w = {"fwd": k, "max_batched": r["max"], "max_alone": alone, "max_cpu": f64,
             "pos": r["pos"], "token": r["token"], "token_str": tok.decode([r["token"]]) if r["token"] != pp.MASK else "[MASK]",
             "context": dec(x1[0].tolist())}
        res["worst"].append(w)
        print(json.dumps({k_: v for k_, v in w.items() if k_ != "context"}), flush=True)
        print("   context:", w["context"][:600].replace("\n", " "), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
