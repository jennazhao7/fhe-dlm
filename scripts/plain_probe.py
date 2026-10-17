"""Plaintext probes on the real MDLM-OWT checkpoint for the encrypted design.

  ranges  E1 input: per-layer ranges of the residual stream, LayerNorm inputs
          (and the spread of their 1/std across positions -- what a
          fixed-constant norm has to absorb), GELU inputs and qkv, by mask
          ratio and split into masked vs unmasked positions. The encrypted
          layer needs |x/S| < 1 at bootstraps and GELU inputs inside the
          Chebyshev interval [-8, 8]; so far that was only checked on random
          weights (results/e0_layer_audit_full_packed.json).
  noise   Does the model survive the bootstrap precision the GPU delivers?
          Full-slot [4,4] bootstraps at N=2^17 give 8.5 bits (FIDESlib) /
          9.2 bits (OpenFHE CPU). Bootstrap error is absolute w.r.t. the
          residual-stream scale S, so after every block (one bootstrap per
          layer in the E5 schedule; 12 here, conservative vs ~11) we add
          N(0, (2^-b * S)^2) with S = max|x| of that block output. Metrics:
          masked-token NLL at mask ratios {0.15, 0.5, 0.9}, and a small
          unconditional generation test (gen-PPL under GPT-2 Large + unigram
          entropy, the plan's guard against low-entropy samples).

Model: kuleshov-group/mdlm-no_flashattn-fp32-owt (the flash-attn build cannot
run on the sm_75 RTX 6000s). Its code still imports flash_attn for one rotary
helper and omegaconf for nothing, so both are stubbed before loading; the
rotary stub is the standard non-interleaved rotary that flash-attn's
apply_rotary_emb_qkv_ computes. Text: WikiText-103 validation (OpenWebText
itself is ~40 GB; numbers are for relative comparisons across noise levels).

Runs offline (CRC compute nodes have no network): everything is read from
$HF_HOME, prefetched on the front end.
"""
import argparse, json, math, os, sys, time, types

import torch
import torch.nn.functional as F

MASK = 50257
DEV = os.environ.get("PROBE_DEV", "cuda")   # PROBE_DEV=cpu for a front-end smoke test
MDLM = "kuleshov-group/mdlm-no_flashattn-fp32-owt"


# ---------------------------------------------------------------- stubs
def _rotary_qkv_(qkv, cos, sin, *a, **k):
    """flash_attn.layers.rotary.apply_rotary_emb_qkv_, non-interleaved:
    qkv (b, s, 3, h, d); cos/sin (s, r/2); rotates the first r dims of q, k in place."""
    r = 2 * cos.shape[-1]
    c = cos[None, :, None, :].to(qkv.dtype)
    s = sin[None, :, None, :].to(qkv.dtype)
    for i in (0, 1):
        x = qkv[:, :, i, :, :r]
        x1, x2 = x[..., : r // 2], x[..., r // 2:]
        qkv[:, :, i, :, :r] = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return qkv


def _stub(name):
    """Empty module with a __spec__ (torch._dynamo calls find_spec on imports)."""
    import importlib.machinery
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return mod


def install_stubs():
    fa = _stub("flash_attn")
    layers = _stub("flash_attn.layers")
    rotary = _stub("flash_attn.layers.rotary")
    rotary.apply_rotary_emb_qkv_ = _rotary_qkv_
    layers.rotary, fa.layers = rotary, layers
    sys.modules.update({"flash_attn": fa, "flash_attn.layers": layers,
                        "flash_attn.layers.rotary": rotary})
    sys.modules.setdefault("omegaconf", _stub("omegaconf"))


# ---------------------------------------------------------------- setup
def snapshot(repo):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, local_files_only=True)


def load_model(dev):
    install_stubs()
    from transformers import AutoModelForMaskedLM
    m = AutoModelForMaskedLM.from_pretrained(snapshot(MDLM), trust_remote_code=True)
    return m.to(dev).eval()


def load_chunks(L, n, seed=0):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot("openai-community/gpt2"))
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    text = "".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    chunks = ids[: (len(ids) // L) * L].view(-1, L)
    g = torch.Generator().manual_seed(seed)
    return chunks[torch.randperm(len(chunks), generator=g)[:n]]


def blocks(m):
    return m.backbone.blocks


CUR = {"mask": None}   # masked positions of the batch in flight (public in our protocol)


def forward_logits(m, x):
    CUR["mask"] = x == MASK
    t = torch.zeros(x.shape[0], device=x.device)
    logits = m(input_ids=x, timesteps=t)
    logits = logits[0] if isinstance(logits, tuple) else logits
    logits[..., MASK] = -float("inf")             # never predict the mask token
    return logits


def mask_batch(x, ratio, g):
    m = torch.rand(x.shape, generator=g, device="cpu").to(x.device) < ratio
    return torch.where(m, torch.full_like(x, MASK), x), m


# ---------------------------------------------------------------- ranges
class Stat:
    """Running stats for one tensor stream: max|x|, rms, a value subsample for
    quantiles, per-position max split by masked/unmasked, LayerNorm 1/std."""
    SAMPLE = 20_000

    def __init__(self, norm=False):
        self.max = 0.0; self.sq = 0.0; self.n = 0
        self.vals = []; self.mmax = 0.0; self.umax = 0.0
        self.norm = norm; self.inv = []; self.inv_m = []; self.inv_u = []; self.out8 = 0

    def add(self, x, mk):
        x = x.detach().float(); ab = x.abs()
        self.max = max(self.max, float(ab.max())); self.sq += float(x.pow(2).sum()); self.n += x.numel()
        flat = ab.flatten()
        self.vals.append(flat[torch.randint(0, flat.numel(), (self.SAMPLE,), device=flat.device)].cpu())
        self.out8 += int((ab > 8).sum())
        tok = ab.amax(-1)
        if mk.any(): self.mmax = max(self.mmax, float(tok[mk].max()))
        if (~mk).any(): self.umax = max(self.umax, float(tok[~mk].max()))
        if self.norm:
            inv = x.var(-1, unbiased=False).add(1e-5).rsqrt()
            self.inv.append(inv.flatten().cpu())
            if mk.any(): self.inv_m.append(inv[mk].cpu())
            if (~mk).any(): self.inv_u.append(inv[~mk].cpu())

    def summary(self):
        v = torch.cat(self.vals)
        r = {"max_abs": self.max, "rms": math.sqrt(self.sq / self.n),
             "p99_abs": float(torch.quantile(v, 0.99)), "p999_abs": float(torch.quantile(v, 0.999)),
             "max_abs_masked_pos": self.mmax, "max_abs_unmasked_pos": self.umax,
             "frac_abs_gt_8": self.out8 / self.n}
        if self.norm:
            inv = torch.cat(self.inv)
            r["inv_std"] = {"min": float(inv.min()), "max": float(inv.max()),
                            "cv": float(inv.std() / inv.mean()),
                            "masked_mean": float(torch.cat(self.inv_m).mean()) if self.inv_m else None,
                            "unmasked_mean": float(torch.cat(self.inv_u).mean()) if self.inv_u else None}
        return r


@torch.no_grad()
def cmd_ranges(a):
    dev = DEV
    m = load_model(dev)
    data = load_chunks(a.L, a.n)
    out = {"model": MDLM, "L": a.L, "n_seqs": int(len(data)), "ratios": {}}
    g = torch.Generator().manual_seed(1)
    for ratio in a.ratios:
        stats, cur = {}, {}

        def rec(name, use_input, norm=False):
            stats[name] = Stat(norm)
            def h(mod, inp, outp):
                stats[name].add(inp[0] if use_input else outp, cur["mask"])
            return h

        hooks = []
        for i, b in enumerate(blocks(m)):
            hooks += [b.register_forward_hook(rec(f"L{i}.resid_out", False)),
                      b.norm1.register_forward_hook(rec(f"L{i}.norm1_in", True, norm=True)),
                      b.norm2.register_forward_hook(rec(f"L{i}.norm2_in", True, norm=True)),
                      b.attn_qkv.register_forward_hook(rec(f"L{i}.qkv", False)),
                      b.mlp[0].register_forward_hook(rec(f"L{i}.gelu_in", False))]
        for s in range(0, len(data), a.bs):
            x = data[s:s + a.bs].to(dev)
            xm, mk = mask_batch(x, ratio, g)
            cur["mask"] = mk
            forward_logits(m, xm)
        for h in hooks:
            h.remove()
        layers = {k: st.summary() for k, st in stats.items()}
        out["ratios"][str(ratio)] = layers
        gl = max(v["max_abs"] for k, v in layers.items() if k.endswith("gelu_in"))
        rs = max(v["max_abs"] for k, v in layers.items() if k.endswith("resid_out"))
        print(f"[ranges] mask {ratio}: max|resid| {rs:.1f}  max|gelu_in| {gl:.1f}", flush=True)
        json.dump(out, open(a.out, "w"), indent=1)


# ---------------------------------------------------------------- noise
class Noise:
    """Adds N(0, (2^-bits * max|x|)^2) to every block output (bootstrap error)."""

    def __init__(self, m, bits):
        self.bits, self.hooks = bits, []
        if bits is None:
            return
        for b in blocks(m):
            self.hooks.append(b.register_forward_hook(self.hook))

    def hook(self, mod, inp, outp):
        S = outp.detach().abs().amax()
        return outp + torch.randn_like(outp) * (S * 2.0 ** -self.bits)

    def remove(self):
        for h in self.hooks:
            h.remove()


@torch.no_grad()
def masked_nll(m, data, ratio, bs, seed):
    g = torch.Generator().manual_seed(seed)
    tot, cnt = 0.0, 0
    for s in range(0, len(data), bs):
        x = data[s:s + bs].to(DEV)
        xm, mk = mask_batch(x, ratio, g)
        lp = forward_logits(m, xm).log_softmax(-1)
        nll = -lp.gather(-1, x[..., None])[..., 0]
        tot += float(nll[mk].sum()); cnt += int(mk.sum())
    return tot / max(cnt, 1)


@torch.no_grad()
def sample(m, n, L, steps, seed):
    """MDLM ancestral sampler, linear schedule: at t -> s each masked position
    is revealed with prob (t - s) / t, drawing from p(x0 | x_t)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.full((n, L), MASK, device=DEV)
    for k in range(steps, 0, -1):
        t, s_ = k / steps, (k - 1) / steps
        probs = forward_logits(m, x).float().softmax(-1)
        x0 = torch.multinomial(probs.view(-1, probs.shape[-1]), 1, generator=g).view(n, L)
        reveal = (torch.rand(n, L, generator=g, device=DEV) < (t - s_) / t) & (x == MASK)
        x = torch.where(reveal, x0, x)
    return x


@torch.no_grad()
def gen_ppl(judge, x, bs=8):
    tot, cnt = 0.0, 0
    for s in range(0, len(x), bs):
        xb = x[s:s + bs]
        lp = judge(xb).logits[:, :-1].float().log_softmax(-1)
        nll = -lp.gather(-1, xb[:, 1:, None])[..., 0]
        tot += float(nll.sum()); cnt += nll.numel()
    return math.exp(tot / cnt)


def unigram_entropy(x):
    ent = []
    for row in x:
        _, c = row.unique(return_counts=True)
        p = c.float() / c.sum()
        ent.append(float(-(p * p.log()).sum()))
    return sum(ent) / len(ent)


def cmd_noise(a):
    m = load_model(DEV)
    data = load_chunks(a.L, a.n)
    judge = None
    if a.gen:
        from transformers import AutoModelForCausalLM
        judge = AutoModelForCausalLM.from_pretrained(snapshot("openai-community/gpt2-large")).to(DEV).eval()
    out = {"model": MDLM, "L": a.L, "n_seqs": int(len(data)), "noise_at": "every block output",
           "sigma": "2^-bits * max|block output|", "runs": []}
    for bits in a.bits:
        b = None if bits <= 0 else bits
        nz = Noise(m, b)
        r = {"bits": b}
        t0 = time.time()
        for ratio in (0.15, 0.5, 0.9):
            r[f"nll_mask{ratio}"] = masked_nll(m, data, ratio, a.bs, seed=7)
        if a.gen and (b is None or b in a.gen_bits):
            xs = sample(m, a.gen_n, a.L, a.steps, seed=11)
            r["gen_ppl_gpt2_large"] = gen_ppl(judge, xs)
            r["gen_unigram_entropy"] = unigram_entropy(xs)
        nz.remove()
        r["seconds"] = time.time() - t0
        out["runs"].append(r)
        print("[noise] " + json.dumps(r), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)


# ---------------------------------------------------------------- outlier
@torch.no_grad()
def calibrate(m, data, ratio, bs, thresh, seed=3):
    """Per-layer mean block output at masked and at unmasked positions (the
    'public constants' a server could subtract), plus the channels whose
    |value| ever exceeds `thresh`, with their per-class statistics."""
    L = len(blocks(m)); d = m.config.hidden_dim
    acc = {k: torch.zeros(L, d, dtype=torch.float64, device=DEV) for k in ("m_sum", "u_sum", "m_sq", "u_sq")}
    big = torch.zeros(L, d, dtype=torch.float64, device=DEV)
    cnt = {"m": 0, "u": 0}
    mn = torch.full((L, d), float("inf"), device=DEV); mx = torch.full((L, d), -float("inf"), device=DEV)
    spikes = [{"pos": [], "masked": [], "mask_rank": [], "n_masked_in_seq": [], "val": [], "n_seqs": 0}
              for _ in range(L)]

    def hook(i):
        def h(mod, inp, outp):
            mk = CUR["mask"]; x = outp.detach().double()
            xm, xu = x[mk], x[~mk]
            acc["m_sum"][i] += xm.sum(0); acc["m_sq"][i] += (xm * xm).sum(0)
            acc["u_sum"][i] += xu.sum(0); acc["u_sq"][i] += (xu * xu).sum(0)
            big[i] = torch.maximum(big[i], x.abs().amax((0, 1)))
            if xm.numel():
                mn[i] = torch.minimum(mn[i], xm.amin(0).float()); mx[i] = torch.maximum(mx[i], xm.amax(0).float())
            if i == 0:
                cnt["m"] += int(mk.sum()); cnt["u"] += int((~mk).sum())
            # where do the spikes sit? sequence index, and rank among masked positions
            sp = x.abs().amax(-1) > thresh                        # (b, L)
            if sp.any():
                b_idx, pos = torch.nonzero(sp, as_tuple=True)
                rank = (mk.long().cumsum(-1) - 1)[b_idx, pos]     # 0 = first masked token
                spikes[i]["pos"] += pos.tolist(); spikes[i]["masked"] += mk[b_idx, pos].tolist()
                spikes[i]["mask_rank"] += rank.tolist()
                spikes[i]["n_masked_in_seq"] += mk.sum(-1)[b_idx].tolist()
                spikes[i]["val"] += x[b_idx, pos].abs().amax(-1).tolist()
            spikes[i]["n_seqs"] += x.shape[0]
        return h

    hs = [b.register_forward_hook(hook(i)) for i, b in enumerate(blocks(m))]
    g = torch.Generator().manual_seed(seed)
    for st in range(0, len(data), bs):
        x = data[st:st + bs].to(DEV)
        forward_logits(m, mask_batch(x, ratio, g)[0])
    for h in hs:
        h.remove()
    mean_m = acc["m_sum"] / max(cnt["m"], 1); mean_u = acc["u_sum"] / max(cnt["u"], 1)
    std_m = (acc["m_sq"] / max(cnt["m"], 1) - mean_m ** 2).clamp_min(0).sqrt()
    std_u = (acc["u_sq"] / max(cnt["u"], 1) - mean_u ** 2).clamp_min(0).sqrt()
    report = {}
    for i in range(L):
        chans = torch.nonzero(big[i] > thresh).flatten().tolist()
        report[f"L{i}"] = {"channels_over_thresh": [
            {"ch": c, "max_abs": float(big[i, c]),
             "masked_mean": float(mean_m[i, c]), "masked_std": float(std_m[i, c]),
             "masked_min": float(mn[i, c]), "masked_max": float(mx[i, c]),
             "unmasked_mean": float(mean_u[i, c]), "unmasked_std": float(std_u[i, c])} for c in chans]}
    for i in range(L):
        sp = spikes[i]
        if not sp["pos"]:
            continue
        from collections import Counter
        n = len(sp["pos"])
        report[f"L{i}"]["spikes"] = {
            "n_spike_tokens": n, "per_seq": n / sp["n_seqs"],
            "frac_masked": sum(sp["masked"]) / n,
            "top_positions": Counter(sp["pos"]).most_common(8),
            "top_mask_ranks": Counter(sp["mask_rank"]).most_common(8),
            "val_min": min(sp["val"]), "val_median": sorted(sp["val"])[n // 2], "val_max": max(sp["val"])}
    return {"mean_masked": mean_m.float(), "mean_unmasked": mean_u.float(),
            "outlier": big > thresh, "report": report}


class NoiseV:
    """Bootstrap noise with the outlier handled:
      offset  subtract the calibrated public constant (masked / unmasked mean
              per layer), take S = max|x - c|, add noise, add c back;
      oracle  outlier channels untouched, S and noise over the other channels."""

    def __init__(self, m, bits, mode, cal):
        self.bits, self.mode, self.cal, self.hooks, self.S = bits, mode, cal, [], []
        if bits is None:
            return
        for i, b in enumerate(blocks(m)):
            self.hooks.append(b.register_forward_hook(self.make(i)))

    def make(self, i):
        def h(mod, inp, outp):
            if self.mode == "offset":
                mk = CUR["mask"][..., None]
                c = torch.where(mk, self.cal["mean_masked"][i], self.cal["mean_unmasked"][i])
                r = outp - c
                S = r.detach().abs().amax()
                self.S.append(float(S))
                return c + r + torch.randn_like(r) * (S * 2.0 ** -self.bits)
            keep = ~self.cal["outlier"][i]                      # oracle
            S = outp.detach()[..., keep].abs().amax()
            self.S.append(float(S))
            return outp + torch.randn_like(outp) * (S * 2.0 ** -self.bits) * keep
        return h

    def remove(self):
        for h in self.hooks:
            h.remove()


def cmd_outlier(a):
    m = load_model(DEV)
    data = load_chunks(a.L, a.n)
    calib, evalset = data[: a.n // 2], data[a.n // 2:]          # calibrate and evaluate on disjoint text
    cal = calibrate(m, calib, 0.5, a.bs, a.thresh)
    out = {"model": MDLM, "L": a.L, "n_calib": int(len(calib)), "n_eval": int(len(evalset)),
           "thresh": a.thresh, "outliers": cal["report"], "runs": []}
    for k, v in cal["report"].items():
        if "spikes" in v:
            sp = v["spikes"]
            print(f"[spikes] {k}: {sp['n_spike_tokens']} tokens ({sp['per_seq']:.2f}/seq, "
                  f"{sp['frac_masked']*100:.0f}% masked) pos {sp['top_positions'][:5]} "
                  f"mask-rank {sp['top_mask_ranks'][:5]} |x| {sp['val_min']:.0f}..{sp['val_median']:.0f}..{sp['val_max']:.0f}",
                  flush=True)
        if v["channels_over_thresh"]:
            print(f"[outlier] {k}: " + "; ".join(
                f"ch{c['ch']} masked {c['masked_mean']:.0f}±{c['masked_std']:.1f} "
                f"[{c['masked_min']:.0f},{c['masked_max']:.0f}] unmasked {c['unmasked_mean']:.1f}±{c['unmasked_std']:.1f}"
                for c in v["channels_over_thresh"]), flush=True)
    judge = None
    if a.gen:
        from transformers import AutoModelForCausalLM
        judge = AutoModelForCausalLM.from_pretrained(snapshot("openai-community/gpt2-large")).to(DEV).eval()
    for mode in ("offset", "oracle"):
        for bits in a.bits:
            b = None if bits <= 0 else bits
            nz = NoiseV(m, b, mode, cal)
            r = {"mode": mode, "bits": b}
            for ratio in (0.15, 0.5, 0.9):
                r[f"nll_mask{ratio}"] = masked_nll(m, evalset, ratio, a.bs, seed=7)
            if nz.S:
                r["S_median"] = sorted(nz.S)[len(nz.S) // 2]; r["S_max"] = max(nz.S)
            if a.gen and b is not None and b in a.gen_bits:
                xs = sample(m, a.gen_n, a.L, a.steps, seed=11)
                r["gen_ppl_gpt2_large"] = gen_ppl(judge, xs)
                r["gen_unigram_entropy"] = unigram_entropy(xs)
            nz.remove()
            out["runs"].append(r)
            print("[outlier] " + json.dumps(r), flush=True)
            json.dump(out, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("ranges")
    r.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    n = sub.add_parser("noise")
    n.add_argument("--bits", type=float, nargs="+", default=[0, 16, 12, 10, 9, 8, 7, 6, 5])
    n.add_argument("--gen", action="store_true")
    n.add_argument("--gen-bits", type=float, nargs="+", default=[10, 9, 8, 7, 6])
    n.add_argument("--gen-n", type=int, default=32)
    n.add_argument("--steps", type=int, default=128)
    o = sub.add_parser("outlier")
    o.add_argument("--thresh", type=float, default=1000.0)
    o.add_argument("--bits", type=float, nargs="+", default=[0, 16, 12, 10, 9, 8, 7, 6])
    o.add_argument("--gen", action="store_true")
    o.add_argument("--gen-bits", type=float, nargs="+", default=[9, 8])
    o.add_argument("--gen-n", type=int, default=32)
    o.add_argument("--steps", type=int, default=128)
    for p in (r, n, o):
        p.add_argument("--L", type=int, default=256)
        p.add_argument("--n", type=int, default=256)
        p.add_argument("--bs", type=int, default=16)
        p.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    {"ranges": cmd_ranges, "noise": cmd_noise, "outlier": cmd_outlier}[a.cmd](a)


if __name__ == "__main__":
    main()
