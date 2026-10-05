"""Shared load/generate/score path for steps 3 and 4.

Every run here is fp32, EMA weights, fixed t-grid and replayed SDE noise, so any
two runs that share a seed differ only by what is deliberately changed
(common-random-number pairing). gen-PPL is padding-masked.
"""

from __future__ import annotations

import glob
import os
import time
from typing import Optional

import torch

from configs.config import load_config_from_yaml, SamplingConfig
from modules.model import ELF_models
from utils.generation_utils import _dlm_decode_batch, mask_after_eos

from fhedlm import metrics
from fhedlm.sampler import sample, draw_schedule
from fhedlm.perturb import Perturbation

# HF_HUB_OFFLINE resolves only canonical repo ids; the checkpoint config uses
# legacy aliases.
HF_ALIASES = {
    "t5-small": "google-t5/t5-small",
    "gpt2-large": "openai-community/gpt2-large",
}
ROOT = os.environ.get("FHEDLM_ROOT", "/groups/tjung/jzhao7/fhe-dlm")
CKPT_GLOB = f"{ROOT}/hf-cache/hub/models--embedded-language-flows--ELF-B-owt-torch/snapshots/*"


def hf_id(name: str) -> str:
    return HF_ALIASES.get(name, name)


def load_elf(device="cpu"):
    ckpt_path = glob.glob(f"{CKPT_GLOB}/checkpoint_95085")[0]
    conf_path = glob.glob(f"{CKPT_GLOB}/config.yml")[0]
    cfg = load_config_from_yaml(conf_path)
    cfg.use_bf16 = False

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    p = ckpt["params"]
    depth = 1 + max(int(k.split(".")[1]) for k in p if k.startswith("blocks."))
    hidden = p["blocks.0.norm1.weight"].shape[0]
    text_dim = p["final_layer.linear.weight"].shape[0]
    vocab = p["unembed_kernel"].shape[1]
    name = {12: "ELF-B", 24: "ELF-M", 32: "ELF-L"}[depth]

    model = ELF_models[name](
        text_encoder_dim=text_dim, max_length=cfg.max_length, vocab_size=vocab,
        bottleneck_dim=cfg.bottleneck_dim, num_time_tokens=cfg.num_time_tokens,
        num_self_cond_cfg_tokens=cfg.num_self_cond_cfg_tokens,
        num_model_mode_tokens=cfg.num_model_mode_tokens,
    ).to(device).float()
    # EMA weights: ELF's _build_eval_model evaluates ema_params1, and the
    # published 24.1 / 5.15 are EMA numbers.
    src = ckpt.get("ema_params1") or p
    model.load_state_dict({k: v.float() for k, v in src.items()}, strict=True)
    model.eval()
    return model, cfg, text_dim


def sampling_cfg(K: int) -> SamplingConfig:
    """ELF's own unconditional recipe, with gamma by step count."""
    return SamplingConfig(sampling_method="sde", num_sampling_steps=[K], cfgs=[1],
                          self_cond_cfg_scales=[3], time_schedule="logit_normal",
                          sde_gamma=1.5 if K <= 32 else 1.0)


@torch.no_grad()
def run_config(model, cfg, text_dim, K: int, n_samples: int, seed: int,
               tok, device="cpu", keep_states: bool = False,
               perturb: Optional[Perturbation] = None, step_hook=None):
    """One generation run under a fixed, replayable schedule."""
    sc = sampling_cfg(K)
    t_steps, z0, eps = draw_schedule(K, n_samples, cfg.max_length, text_dim,
                                     cfg, seed=seed, device=device)
    t0 = time.time()
    traj = sample(model=model, config=cfg, sampling_config=sc, t_steps=t_steps,
                  z0=z0, eps=eps, cfg_scale=1.0, self_cond_cfg_scale=3.0,
                  perturb=perturb or Perturbation(), keep_states=keep_states,
                  step_hook=step_hook, device=device)
    gen_s = time.time() - t0
    # The decode pass is a full forward too -- it must run under the same
    # approximation as the trajectory, or step 4 would silently evaluate an
    # exact decoder on approximate latents. It gets its OWN index (K) rather
    # than reusing the last sampling step's, so its polynomial is fitted on the
    # decode pass's own activations. Sharing step K-1 would pool two different
    # activation distributions into one interval and widen it for both.
    if step_hook is not None:
        step_hook(K)
    ids = _dlm_decode_batch(z=traj.z_final, model=model, t_final_val=float(t_steps[-1]),
                            config=cfg, self_cond_cfg_scale=3.0)
    ids = mask_after_eos(ids, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    texts = tok.batch_decode(ids, skip_special_tokens=True)
    return {"ids": ids, "texts": texts, "traj": traj, "gen_seconds": gen_s,
            "t_steps": t_steps.tolist()}


def score(ids, texts, tok, judge, judge_tok, device="cpu") -> dict:
    return {
        "gen_ppl": metrics.generative_perplexity(ids, judge, judge_tok, tok, device=device),
        "entropy_elf": metrics.unigram_entropy_elf(texts, judge_tok),
        "entropy_pooled_t5": metrics.unigram_entropy(ids, pad_id=tok.pad_token_id),
        "repetition_4gram": metrics.repetition_rate(ids, n=4, pad_id=tok.pad_token_id),
    }
