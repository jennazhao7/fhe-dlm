"""Where does MDLM-OWT's massive activation live once position 0 is unmasked?

Oct 17 (results/plain/outlier.json) saw spikes (|x| 4k-23k, layers 5-10) only
at position 0 *when position 0 was masked*, which suggested pinning position 0
unmasked as a deployable fix. A smoke run of `plain_probe.py noise --pin0`
then still showed S_max ~1.9e4. GPT-style models commonly put an attention
sink on a leading BOS token, and --pin0 pins position 0 to <|endoftext|> when
sampling, so the sink may just move. This records, per block, max|x| and the
position holding it, at L in {256, 64} (the smoke ran at 64), for:

  E  position 0 MASKED inside real context (15/50/90% of the rest masked) --
     the suspected true sink trigger; must reproduce Oct 17's spike
  A  position 0 = its real token, unmasked (NLL under --pin0)
  H  position 0 = <|endoftext|>, rest partly revealed (generation under --pin0)
  B  position 0 = <|endoftext|>, everything else masked (pin0 generation start)
  C  everything masked (no context at all)

A first pass with only A(50%)/B/C and "The" at 0 showed no spike anywhere
(peak <= 1.5e3), i.e. "position 0 masked" alone is not the trigger.

    PROBE_DEV=cpu python scripts/probe_sink_pos.py results/plain/sink_pos.json
"""
import json, os, sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import plain_probe as pp   # noqa: E402  (reuses its stubs, loader and forward)

N_SEQ = 4


@torch.no_grad()
def profile(m, x):
    """Per block: max|x| over the batch, the position holding it, and whether it
    sits at position 0; plus the largest value anywhere except position 0."""
    rows, hooks = [], []
    for i, b in enumerate(pp.blocks(m)):
        def h(mod, inp, out, i=i):
            a = out.detach().abs().amax(-1)              # (batch, L): per-token max over channels
            flat = int(a.argmax())
            pos = flat % a.shape[1]
            rows.append({"block": i, "max": float(a.max()), "argmax_pos": pos,
                         "max_excl_pos0": float(a[:, 1:].max()),
                         "pos0_max": float(a[:, 0].max())})
        hooks.append(b.register_forward_hook(h))
    pp.forward_logits(m, x)
    for h in hooks:
        h.remove()
    return rows


def cases(m, L):
    """Position-0 conditions x mask ratio, on real WikiText context."""
    data = pp.load_chunks(L, N_SEQ).to(pp.DEV)
    allmask = torch.full_like(data, pp.MASK)
    out = []
    # The suspected TRUE sink trigger: position 0 masked, real context around it.
    # Sanity check -- it must reproduce Oct 17's ~1e4-2e4 spike at position 0.
    for r in (0.15, 0.5, 0.9):
        x, _ = pp.mask_batch(data, r, torch.Generator().manual_seed(7))
        x[:, 0] = pp.MASK
        out.append((f"E_mask0_ctx{int(r*100)}", x))
    # NLL under --pin0: position 0 keeps its real token.
    for r in (0.15, 0.5, 0.9):
        x, _ = pp.mask_batch(data, r, torch.Generator().manual_seed(7))
        x[:, 0] = data[:, 0]
        out.append((f"A_real0_ctx{int(r*100)}", x))
    # Generation under --pin0, mid-trajectory: <|endoftext|> at 0, rest partly revealed.
    for r in (0.15, 0.5, 0.9):
        x, _ = pp.mask_batch(data, r, torch.Generator().manual_seed(7))
        x[:, 0] = pp.BOS
        out.append((f"H_bos0_ctx{int(r*100)}", x))
    x = allmask.clone(); x[:, 0] = pp.BOS
    out.append(("B_bos0_allmasked", x))
    out.append(("C_allmasked", allmask.clone()))
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "results/plain/sink_pos.json"
    torch.manual_seed(0)
    m = pp.load_model(pp.DEV)
    out = {"model": pp.MDLM, "n_seqs": N_SEQ, "runs": {}}
    for Lx in (256, 64):     # the --pin0 smoke that still showed S_max 1.9e4 ran at L=64
        for name, x in cases(m, Lx):
            rows = profile(m, x)
            key = f"L{Lx}_{name}"
            out["runs"][key] = rows
            peak = max(rows, key=lambda r: r["max"])
            p0 = max(rows, key=lambda r: r["pos0_max"])
            print(f"[sink] {key:22s} peak |x| {peak['max']:8.0f} @ blk {peak['block']:2d} pos {peak['argmax_pos']:3d}"
                  f" | pos0 max {p0['pos0_max']:8.0f} @ blk {p0['block']:2d}"
                  f" | excl pos0 {max(r['max_excl_pos0'] for r in rows):6.0f}", flush=True)
            json.dump(out, open(out_path, "w"), indent=1)
    print(f"[sink] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
