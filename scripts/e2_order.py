"""E2 (plan P0, frozen teachers): which tokens to commit per encrypted pass.

Each pass decodes one block of B live tokens and the client commits k of
them, so a block costs ceil(B/k) passes. The open question for C4/E4 is how
much quality is lost as k grows, and how much a smarter choice of WHICH k
tokens to commit buys back. Spec-AR beats the DLM at k <= 4, ties at k = 8
(HANDOFF §9.3), so the DLM must keep quality at k >= 8.

Teachers: kuleshov-group/bd3lm-owt-block_size{4,8,16} (block-causal; run with
SDPA, its sampling path), plus MDLM-OWT (no-flash fp32) with the whole
sequence as one block (B = L) as a reference. Within the live block, order:
  random      k uniformly random masked positions (MDLM-style)
  confidence  the k whose sampled token has the highest probability
  margin      largest top-1 minus top-2 probability
  entropy     lowest predictive entropy
  l2r         leftmost k (semi-AR inside the block)
k = B commits the whole block in one pass, so order is moot there. Tokens are
sampled at T = 1 (top_p = 1, as in the noise probes) and with nucleus 0.9 (as
BD3-LM reports gen-PPL). BD3-LM sequences start with <|endoftext|>, as in
training. Metric: gen-PPL under GPT-2 Large and per-sample unigram entropy
(the guard against degenerate repetition), n samples of length L, one seed.

    python scripts/e2_order.py --out results/plain/e2_order.json

--sweep (entropy-matched follow-up): confidence ordering wins gen-PPL by
collapsing unigram entropy, so compare orders on the gen-PPL-vs-entropy
frontier instead of at one point. Two knobs: the sampling temperature T
(logits / T; T < 1 lowers random order's entropy, T > 1 raises confidence's),
and the choice temperature tau of confidence ordering, which ranks by
log p(x0) + tau * Gumbel (MaskGIT-style; tau = 0 is confidence, tau -> inf
is random). Also records the real-text reference (WikiText-103 validation
chunks of length L: the models are OWT-trained, so this is approximate).

    python scripts/e2_order.py --sweep --out results/plain/e2_entropy.json
"""
import argparse, json, math, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(__file__))
import plain_probe as pp   # noqa: E402

ORDERS = ("random", "confidence", "margin", "entropy", "l2r")


def load_bd3(B):
    pp.install_stubs()
    from transformers import AutoModelForMaskedLM
    m = AutoModelForMaskedLM.from_pretrained(pp.snapshot(f"kuleshov-group/bd3lm-owt-block_size{B}"),
                                             trust_remote_code=True, attn_backend="sdpa")
    return m.to(pp.DEV).eval()


def logits_bd3(m, x):
    t = torch.zeros(x.shape[0], device=x.device)
    out = m(input_ids=x, timesteps=t, sample_mode=True)
    out = out.logits if hasattr(out, "logits") else out[0] if isinstance(out, tuple) else out
    return out


def nucleus(probs, top_p, chunk=4):
    if top_p >= 1:
        return probs
    for i in range(0, len(probs), chunk):              # in place, a few rows at a time (B = L sorts OOM at 22 GB)
        sp, si = probs[i:i + chunk].sort(-1, descending=True)
        drop = sp.cumsum(-1) - sp > top_p              # keep the smallest prefix with mass >= top_p
        sp = sp.masked_fill_(drop, 0)
        probs[i:i + chunk] = torch.zeros_like(sp).scatter_(-1, si, sp) / sp.sum(-1, keepdim=True)
        del sp, si, drop
    return probs


def score(order, probs, x0, masked, g, ctemp=0.0):
    if order == "random":
        s = torch.rand(masked.shape, generator=g, device=masked.device)
    elif order == "confidence":
        s = probs.gather(-1, x0[..., None])[..., 0].clamp_min(1e-30).log()
        if ctemp > 0:
            u = torch.rand(masked.shape, generator=g, device=masked.device).clamp(1e-20, 1 - 1e-7)
            s = s - ctemp * (-u.log()).log()                     # + tau * Gumbel
    elif order == "margin":
        top = probs.topk(2, -1).values
        s = top[..., 0] - top[..., 1]
    elif order == "entropy":
        s = (probs * probs.clamp_min(1e-30).log()).sum(-1)       # = -entropy
    else:                                                        # l2r
        s = -torch.arange(masked.shape[1], device=masked.device, dtype=torch.float).expand(masked.shape)
    return s.masked_fill(~masked, -float("inf"))


@torch.no_grad()
def generate(fwd, n, L, B, k, order, top_p, seed, bos, temp=1.0, ctemp=0.0):
    """Block by block (B = L: one block); each pass commits k of the live
    block's masked positions. Returns the samples and the number of passes."""
    g = torch.Generator(device=pp.DEV).manual_seed(seed)
    x = torch.full((n, L), pp.MASK, device=pp.DEV)
    if bos:
        x[:, 0] = pp.BOS
    passes = 0
    for b0 in range(0, L, B):
        while True:
            masked = x[:, b0:b0 + B] == pp.MASK
            if not masked.any():
                break
            lg = fwd(x[:, :b0 + B])[:, b0:b0 + B].float()
            lg[..., pp.MASK] = -float("inf")
            probs = nucleus((lg / temp).softmax(-1), top_p)
            x0 = torch.multinomial(probs.view(-1, probs.shape[-1]), 1, generator=g).view(masked.shape)
            s = score(order, probs, x0, masked, g, ctemp)
            kk = min(k, int(masked.sum(-1).max()))
            sel = torch.zeros_like(masked).scatter_(-1, s.topk(kk, -1).indices, True) & masked
            blk = x[:, b0:b0 + B]
            x[:, b0:b0 + B] = torch.where(sel, x0, blk)
            passes += 1
    return x, passes


# (model, B, ks) x configs: about 30 min on one RTX 6000 at L = 256, n = 64
SWEEP_MODELS = (("bd3lm_B16", 16, (2, 4, 8)), ("bd3lm_B8", 8, (2, 4)), ("mdlm_global", None, (4, 8, 16)))
SWEEP_CONFIGS = ([("confidence", 1.0, c) for c in (0.25, 0.5, 1.0, 2.0, 4.0)]
                 + [("confidence", t, 0.0) for t in (1.25, 1.5)]
                 + [("random", t, 0.0) for t in (0.7, 0.8, 0.9)])


def sweep(a, judge):
    out = {"L": a.L, "gen_n": a.gen_n, "seed": 11, "judge": "gpt2-large", "top_p": 1.0, "runs": []}
    if os.path.exists(a.out):
        out = json.load(open(a.out))
    if "reference_text" not in out:
        xs = pp.load_chunks(a.L, a.gen_n).to(pp.DEV)
        out["reference_text"] = {"data": "wikitext-103 validation", "gen_ppl_gpt2_large": pp.gen_ppl(judge, xs),
                                 "gen_unigram_entropy": pp.unigram_entropy(xs)}
        print("[e2s] reference " + json.dumps(out["reference_text"]), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
    done = {(r["model"], r["k"], r["order"], r["temp"], r["ctemp"]) for r in out["runs"]}
    for name, B, ks in SWEEP_MODELS:
        B = B or a.L
        bos = name.startswith("bd3lm")
        if bos:
            m = load_bd3(B); fwd = lambda x, m=m: logits_bd3(m, x)
        else:
            m = pp.load_model(pp.DEV); fwd = lambda x, m=m: pp.forward_logits(m, x)
        for k in ks:
            for order, temp, ctemp in SWEEP_CONFIGS:
                if (name, k, order, temp, ctemp) in done:
                    continue
                t0 = time.time()
                xs, passes = generate(fwd, a.gen_n, a.L, B, k, order, 1.0, 11, bos, temp, ctemp)
                r = {"model": name, "B": B, "k": k, "order": order, "temp": temp, "ctemp": ctemp,
                     "passes": passes, "tokens_per_pass": (a.L - bos) / passes,
                     "gen_ppl_gpt2_large": pp.gen_ppl(judge, xs),
                     "gen_unigram_entropy": pp.unigram_entropy(xs),
                     "eos_per_seq": float((xs[:, 1:] == pp.BOS).sum(-1).float().mean()),
                     "seconds": time.time() - t0}
                out["runs"].append(r)
                print("[e2s] " + json.dumps(r), flush=True)
                json.dump(out, open(a.out, "w"), indent=1)
        del m
        torch.cuda.empty_cache() if pp.DEV == "cuda" else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--gen-n", type=int, default=64)
    ap.add_argument("--blocks", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--mdlm-ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--top-p", type=float, nargs="+", default=[1.0, 0.9])
    ap.add_argument("--orders", nargs="+", default=list(ORDERS))
    ap.add_argument("--no-mdlm", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="entropy-matched follow-up (see docstring)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    from transformers import AutoModelForCausalLM
    judge = AutoModelForCausalLM.from_pretrained(pp.snapshot("openai-community/gpt2-large")).to(pp.DEV).eval()
    if a.sweep:
        return sweep(a, judge)
    out = {"L": a.L, "gen_n": a.gen_n, "seed": 11, "judge": "gpt2-large", "runs": []}
    if os.path.exists(a.out):                                  # resume: skip finished configs
        out = json.load(open(a.out))
    done = {(r["model"], r["B"], r["k"], r["order"], r["top_p"]) for r in out["runs"]}

    plan = []
    for B in a.blocks:
        ks = [k for k in (1, 2, 4, 8, 16) if k <= B]
        plan.append((f"bd3lm_B{B}", B, ks, True))
    if not a.no_mdlm:
        plan.append(("mdlm_global", a.L, a.mdlm_ks, False))
    for name, B, ks, bos in plan:
        if name.startswith("bd3lm"):
            m = load_bd3(B); fwd = lambda x, m=m: logits_bd3(m, x)
        else:
            m = pp.load_model(pp.DEV); fwd = lambda x, m=m: pp.forward_logits(m, x)
        for top_p in a.top_p:
            for k in ks:
                for order in (a.orders if k < B else ["all"]):
                    key = (name, B, k, order, top_p)
                    if key in done:
                        continue
                    t0 = time.time()
                    xs, passes = generate(fwd, a.gen_n, a.L, B, k, "random" if order == "all" else order,
                                          top_p, 11, bos)
                    r = {"model": name, "B": B, "k": k, "order": order, "top_p": top_p,
                         "passes": passes, "tokens_per_pass": (a.L - bos) / passes,
                         "gen_ppl_gpt2_large": pp.gen_ppl(judge, xs),
                         "gen_unigram_entropy": pp.unigram_entropy(xs),
                         "eos_per_seq": float((xs[:, 1:] == pp.BOS).sum(-1).float().mean()),
                         "seconds": time.time() - t0}
                    out["runs"].append(r)
                    print("[e2] " + json.dumps(r), flush=True)
                    json.dump(out, open(a.out, "w"), indent=1)
        del m
        torch.cuda.empty_cache() if pp.DEV == "cuda" else None


if __name__ == "__main__":
    main()
