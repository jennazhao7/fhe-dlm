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


def forward_logits(m, x):
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
    for p in (r, n):
        p.add_argument("--L", type=int, default=256)
        p.add_argument("--n", type=int, default=256)
        p.add_argument("--bs", type=int, default=16)
        p.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    {"ranges": cmd_ranges, "noise": cmd_noise}[a.cmd](a)


if __name__ == "__main__":
    main()
