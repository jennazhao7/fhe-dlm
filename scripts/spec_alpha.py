"""Spec-AR acceptance alpha with 2- and 4-layer client drafts (HANDOFF §5.3).

The cost model prices client-drafted speculative AR at alpha in {0.6, 0.8}:
the client drafts k tokens in plaintext, one encrypted verify pass commits
(1 - alpha^(k+1)) / (1 - alpha) of them. At alpha = 0.8 spec-AR beats the DLM
at k <= 4 and ties at k = 8, so alpha sets the bar the DLM has to clear.

Target: GPT-2 small (124M), a stand-in for the AR control until that is
trained (kuleshov-group/owt-ar, the matched 110M OWT AR, is not downloaded).
Drafts: the target's embeddings, first n blocks, final norm and tied head,
distilled from the target (forward KL, T = 1) on WikiText-103 train; the
undistilled truncation is reported too (step 0).

alpha under speculative sampling at T = 1 is sum_x min(p(x), q(x)) = 1 - TV
at each prefix, so it is measured exactly (no rejection sampling noise) along
  gen   the target's own unconditional samples (the stream spec decoding
        actually emits), and
  text  WikiText-103 validation (teacher-forced).
Tokens per verify pass: the i.i.d. formula at mean alpha, and a walk over the
per-position alphas (accept draft j with its own alpha; stop at the first
rejection; +1 corrected/bonus token).

    python scripts/spec_alpha.py --out results/plain/spec_alpha.json
"""
import argparse, itertools, json, math, os, sys, time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import plain_probe as pp   # noqa: E402

GPT2 = "openai-community/gpt2"


def load_target():
    from transformers import GPT2LMHeadModel
    return GPT2LMHeadModel.from_pretrained(pp.snapshot(GPT2)).to(pp.DEV).eval()


def make_draft(target, n):
    from transformers import GPT2LMHeadModel
    cfg = type(target.config).from_dict(target.config.to_dict())
    cfg.n_layer = n
    d = GPT2LMHeadModel(cfg)
    sd = {k: v for k, v in target.state_dict().items()
          if not k.startswith("transformer.h.") or int(k.split(".")[2]) < n}
    missing, unexpected = d.load_state_dict(sd, strict=False)
    assert not unexpected and all("attn.bias" in k or "masked_bias" in k for k in missing), (missing, unexpected)
    d.tie_weights()
    return d.to(pp.DEV)


def train_tokens(n_rows):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(pp.snapshot(GPT2))
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    text = [t for t in ds[:n_rows]["text"] if t.strip()]
    ids = []
    for i in range(0, len(text), 10_000):
        ids += itertools.chain.from_iterable(tok(text[i:i + 10_000]).input_ids)
    return torch.tensor(ids)


def distill(target, draft, ids, steps, bs, L, lr, log_every=200):
    draft.train()
    opt = torch.optim.AdamW(draft.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    g = torch.Generator().manual_seed(0)
    hist = []
    for st in range(steps):
        idx = torch.randint(0, len(ids) - L, (bs,), generator=g)
        x = torch.stack([ids[i:i + L] for i in idx]).to(pp.DEV)
        with torch.no_grad():
            lp = target(x).logits.float().log_softmax(-1)
        lq = draft(x).logits.float().log_softmax(-1)
        loss = (lp.exp() * (lp - lq)).sum(-1).mean()            # KL(p || q)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
        opt.step(); sched.step()
        if st % log_every == 0 or st == steps - 1:
            hist.append((st, float(loss)))
            print(f"[alpha] distill step {st} KL {float(loss):.4f}", flush=True)
    draft.eval()
    return hist


@torch.no_grad()
def alphas(target, draft, xs, bs=16):
    """Per-position acceptance sum_x min(p, q) for predicting xs[:, 1:]."""
    out = []
    for i in range(0, len(xs), bs):
        x = xs[i:i + bs].to(pp.DEV)
        p = target(x).logits[:, :-1].float().softmax(-1)
        q = draft(x).logits[:, :-1].float().softmax(-1)
        out.append(torch.minimum(p, q).sum(-1).cpu())
    return torch.cat(out)                                       # (n, L-1)


def tokens_per_pass(al, k, reps=10, seed=0):
    """Walk each sequence: a pass at position i accepts drafts i, i+1, ...
    each with its own alpha until a rejection or k accepts, then commits one
    more (the corrected or bonus token)."""
    g = torch.Generator().manual_seed(seed)
    tot = passes = 0
    for _ in range(reps):
        u = torch.rand(al.shape, generator=g) < al
        for row in u.tolist():
            i, n = 0, len(row)
            while i < n:
                j = 0
                while j < k and i + j < n and row[i + j]:
                    j += 1
                tot += j + 1; passes += 1; i += j + 1
    return tot / passes


@torch.no_grad()
def sample_target(target, n, L, bs=16, seed=11):
    """Ancestral sampling at T = 1 from <|endoftext|>, L tokens. Not HF
    generate: that stops at <|endoftext|> and pads with it (4/16 samples, ~36
    pad tokens each, inflated gen-PPL and deflated alpha_gen in the first run).
    Here <|endoftext|> is an ordinary token, as in the DLM samplers."""
    g = torch.Generator(device=pp.DEV).manual_seed(seed)
    xs = []
    for i in range(0, n, bs):
        x = torch.full((min(bs, n - i), 1), pp.BOS, device=pp.DEV)
        past = None
        while x.shape[1] < L:
            out = target(x if past is None else x[:, -1:], past_key_values=past, use_cache=True)
            past = out.past_key_values
            p = out.logits[:, -1].float().softmax(-1)
            x = torch.cat([x, torch.multinomial(p, 1, generator=g)], 1)
        xs.append(x.cpu())
    return torch.cat(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train-rows", type=int, default=400_000)
    ap.add_argument("--gen-n", type=int, default=64)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--load-dir", default=None, help="re-measure drafts saved by --save-dir, no distillation")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    target = load_target()
    from transformers import AutoModelForCausalLM
    judge = AutoModelForCausalLM.from_pretrained(pp.snapshot("openai-community/gpt2-large")).to(pp.DEV).eval()
    t0 = time.time()
    xs_gen = sample_target(target, a.gen_n, a.L)
    xs_text = pp.load_chunks(a.L, a.gen_n)
    out = {"target": GPT2, "L": a.L, "gen_n": a.gen_n, "distill": {"steps": a.steps, "bs": a.bs, "lr": a.lr,
           "data": f"wikitext-103 train, first {a.train_rows} rows"},
           "target_gen_ppl_gpt2_large": pp.gen_ppl(judge, xs_gen.to(pp.DEV)),
           "target_gen_unigram_entropy": pp.unigram_entropy(xs_gen), "drafts": []}
    out["target_gen_eos_per_seq"] = float((xs_gen[:, 1:] == pp.BOS).sum(-1).float().mean())
    print(f"[alpha] target samples: gen-PPL {out['target_gen_ppl_gpt2_large']:.1f} ({time.time()-t0:.0f}s)", flush=True)
    del judge
    if a.load_dir:
        from transformers import GPT2LMHeadModel
        out["distill"]["loaded_from"] = a.load_dir
    else:
        ids = train_tokens(a.train_rows)
        out["distill"]["train_tokens"] = len(ids)
        print(f"[alpha] {len(ids)/1e6:.1f}M train tokens", flush=True)

    def measure(draft, n, step):
        r = {"layers": n, "distill_steps": step}
        for name, xs in (("gen", xs_gen), ("text", xs_text)):
            al = alphas(target, draft, xs)
            r[f"alpha_{name}"] = float(al.mean())
            r[f"alpha_{name}_quartiles"] = [float(v) for v in torch.quantile(al.flatten(), torch.tensor([.25, .5, .75]))]
            a_ = r[f"alpha_{name}"]
            r[f"tokens_per_pass_{name}"] = {k: {"iid": (1 - a_ ** (k + 1)) / (1 - a_), "walk": tokens_per_pass(al, k)}
                                            for k in a.ks}
        out["drafts"].append(r)
        print("[alpha] " + json.dumps({k: v for k, v in r.items() if "tokens" not in k}), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)

    for n in a.layers:
        if a.load_dir:
            draft = GPT2LMHeadModel.from_pretrained(os.path.join(a.load_dir, f"gpt2_draft_{n}L")).to(pp.DEV).eval()
            measure(draft, n, a.steps)
            del draft
            continue
        draft = make_draft(target, n)
        measure(draft, n, 0)
        t1 = time.time()
        hist = distill(target, draft, ids, a.steps, a.bs, a.L, a.lr)
        measure(draft, n, a.steps)
        out["drafts"][-1]["kl_history"] = hist
        out["drafts"][-1]["distill_seconds"] = time.time() - t1
        json.dump(out, open(a.out, "w"), indent=1)
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True)
            draft.save_pretrained(os.path.join(a.save_dir, f"gpt2_draft_{n}L"))
        del draft
        torch.cuda.empty_cache() if pp.DEV == "cuda" else None


if __name__ == "__main__":
    main()
