"""Quality metrics for the feasibility gate.

Generative perplexity under a frozen GPT-2 Large is ELF's own headline metric,
so it is here for comparability with their reported 24.1 at K=32. It is NOT
reported alone. Continuous diffusion LMs can drive gen-PPL down by collapsing
into repetition (arXiv 2607.00588), which would make a K-reduction or a
polynomial swap look like an *improvement* while destroying the generation. So
every gen-PPL number is reported beside unigram entropy (ELF's own companion
metric, reference 5.15) and an explicit repetition rate.

Token agreement against the unperturbed run is the metric Experiment 1 actually
turns on; perplexity is the sanity check around it.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import math
import torch


@torch.no_grad()
def generative_perplexity(token_ids: torch.Tensor, judge, judge_tok,
                          elf_tok, device="cpu", batch_size: int = 4) -> float:
    """PPL of ELF's generations under a frozen judge (GPT-2 Large).

    ELF decodes in T5 vocabulary and the judge scores in GPT-2 vocabulary, so
    the text is detokenised and re-tokenised rather than compared id-to-id.

    Padding is masked out with -100. Passing `labels=input_ids` unmasked would
    make the judge score the pad run too, which for a batch of uneven lengths
    drags the reported perplexity toward whatever the pad token is cheap to
    predict -- usually downward, i.e. it would flatter every result.
    """
    texts = elf_tok.batch_decode(token_ids, skip_special_tokens=True)
    texts = [t for t in texts if t.strip()]
    if not texts:
        return float("inf")
    if judge_tok.pad_token is None:
        judge_tok.pad_token = judge_tok.eos_token
    total_nll, total_tok = 0.0, 0
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = judge_tok(chunk, return_tensors="pt", padding=True,
                        truncation=True, max_length=1024).to(device)
        labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
        out = judge(input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"], labels=labels)
        # HF shifts internally: the loss is a mean over positions 1..n-1 that
        # carry a real label. Recover the sum so uneven batches pool correctly.
        n = int((labels[:, 1:] != -100).sum())
        if n == 0:
            continue
        total_nll += float(out.loss) * n
        total_tok += n
    return math.exp(total_nll / max(total_tok, 1))


def unigram_entropy_elf(texts, judge_tok, max_length: int = 1024) -> float:
    """ELF's own entropy metric, reproduced exactly (reference 5.15 for K=32).

    Their definition, from metrics_utils.py:221-234, is NOT the obvious one:
      * computed on the GPT-2 *retokenized* text, not on ELF's T5 ids;
      * per sample over that sample's valid (unpadded) tokens;
      * natural log; then averaged over samples (a MeanMetric).

    Pooling all tokens across samples instead -- the obvious reading -- gives a
    systematically different number, because pooling counts types across the
    whole corpus. Measured here: 6.06 pooled vs their 5.15 per-sample. Getting
    this wrong would have made every later config look more diverse than it is
    and broken comparability with the published reference.
    """
    import numpy as np
    if judge_tok.pad_token is None:
        judge_tok.pad_token = judge_tok.eos_token
    enc = judge_tok(list(texts), return_tensors="np", return_token_type_ids=False,
                    return_attention_mask=True, truncation=True, padding=True,
                    max_length=max_length)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    ents = []
    for i in range(ids.shape[0]):
        valid = ids[i, :int(mask[i].sum())]
        if valid.size == 0:
            continue
        _, counts = np.unique(valid, return_counts=True)
        probs = counts.astype(np.float32) / counts.sum()
        ents.append(float(-np.sum(probs * np.log(probs + 1e-10))))
    return sum(ents) / len(ents) if ents else 0.0


def unigram_entropy(token_ids: torch.Tensor, pad_id: int = 0) -> float:
    """Pooled entropy over ELF's own T5 ids. Reported as a secondary number --
    it is NOT comparable to the published 5.15; use unigram_entropy_elf for that.
    """
    ids = token_ids.flatten().tolist()
    ids = [i for i in ids if i != pad_id]
    if not ids:
        return 0.0
    counts = Counter(ids)
    n = len(ids)
    return -sum((c / n) * math.log(c / n) for c in counts.values())


def repetition_rate(token_ids: torch.Tensor, n: int = 4, pad_id: int = 0) -> float:
    """Fraction of n-grams that are duplicates, averaged over sequences.

    The guard against the repetition attractor: a model that collapses scores a
    low gen-PPL and a high value here at the same time.
    """
    rates = []
    for row in token_ids.tolist():
        row = [t for t in row if t != pad_id]
        if len(row) < n + 1:
            continue
        grams = [tuple(row[i:i + n]) for i in range(len(row) - n + 1)]
        rates.append(1.0 - len(set(grams)) / len(grams))
    return sum(rates) / len(rates) if rates else 0.0


def token_agreement(a: torch.Tensor, b: torch.Tensor, pad_id: int = 0) -> float:
    """Exact per-position token agreement -- Experiment 1's primary metric."""
    mask = (a != pad_id) | (b != pad_id)
    if not bool(mask.any()):
        return 1.0
    return float(((a == b) & mask).sum() / mask.sum())


def trajectory_divergence(ref_states: Sequence[torch.Tensor],
                          pert_states: Sequence[torch.Tensor]) -> dict:
    """Per-step cosine similarity and relative error.

    This is the plot that answers whether error compounds, stays flat, or is
    contracted away by the sampler.
    """
    cos, rel = [], []
    for r, p in zip(ref_states, pert_states):
        r = r.float().flatten(1)
        p = p.float().flatten(1)
        cos.append(float(torch.nn.functional.cosine_similarity(r, p, dim=1).mean()))
        rel.append(float(((p - r).norm(dim=1) / r.norm(dim=1).clamp_min(1e-12)).mean()))
    return {"cosine_by_step": cos, "rel_error_by_step": rel}
