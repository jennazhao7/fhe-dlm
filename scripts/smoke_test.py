"""Step 1: does ELF-B run on CPU in fp32, and does it reproduce its reference?

Reference (PyTorch port README, ELF-B 105M, 32-step SDE):  gen-PPL 24.1, unigram entropy 5.15.

Three things, in one job, because CPU time is the scarce resource:
  A. wall-clock per denoiser forward (a required deliverable, and it tells us
     immediately whether steps 3 and 4 are affordable at all on CPU)
  B. the headline run: upstream's own sampler, K=32 SDE, 8 samples -> gen-PPL,
     unigram entropy, 4-gram repetition
  C. an equivalence check of our instrumented sampler against upstream's, run
     as a pure ODE so both are deterministic and must agree to fp32 precision.
     Steps 3 and 4 use our sampler, so this has to hold before they mean
     anything. The SDE path is excluded from this check on purpose: upstream
     draws its per-step noise inside the step, so the two cannot be made to
     draw the same numbers without editing upstream.
"""
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from configs.config import load_config_from_yaml, SamplingConfig
from modules.model import ELF_models
from utils.generation_utils import _generate_samples_single_batch, _dlm_decode_batch, mask_after_eos
from fhedlm import metrics
from fhedlm.sampler import sample, Trajectory
from fhedlm.perturb import Perturbation


# HF_HUB_OFFLINE resolves only the canonical repo id. The checkpoint config
# says "t5-small", which is a legacy alias and is NOT what sits in the cache, so
# offline lookup fails after the expensive part of the run has already happened.
HF_ALIASES = {
    "t5-small": "google-t5/t5-small",
    "t5-base": "google-t5/t5-base",
    "gpt2-large": "openai-community/gpt2-large",
    "gpt2": "openai-community/gpt2",
}


def hf_id(name: str) -> str:
    return HF_ALIASES.get(name, name)


def infer_dims(sd):
    """Read the architecture off the checkpoint instead of trusting a config."""
    depth = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    hidden = sd["blocks.0.norm1.weight"].shape[0]
    text_dim = sd["final_layer.linear.weight"].shape[0]
    vocab = sd["unembed_kernel"].shape[1]
    return depth, hidden, text_dim, vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="third_party/ELF/src/configs/training_configs/train_owt_ELF-B.yml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="results/step1_smoke.json")
    ap.add_argument("--skip-equiv", action="store_true")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    torch.set_grad_enabled(False)
    print(f"[smoke] torch {torch.__version__}  threads={torch.get_num_threads()}  device=cpu fp32", flush=True)

    cfg = load_config_from_yaml(args.config)
    cfg.use_bf16 = False                      # fp32 on purpose; also moot on CPU
    cfg.max_length = getattr(cfg, "max_length", 1024)

    t0 = time.time()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"[smoke] checkpoint keys: {list(ckpt.keys())}  ({time.time()-t0:.1f}s)", flush=True)
    params = ckpt["params"]
    depth, hidden, text_dim, vocab = infer_dims(params)
    print(f"[smoke] inferred: depth={depth} hidden={hidden} text_enc_dim={text_dim} vocab={vocab}", flush=True)

    heads = {768: 12, 1056: 16, 1280: 16}.get(hidden, 12)
    name = {12: "ELF-B", 24: "ELF-M", 32: "ELF-L"}.get(depth, "ELF-B")
    model = ELF_models[name](
        text_encoder_dim=text_dim, max_length=cfg.max_length, vocab_size=vocab,
        bottleneck_dim=cfg.bottleneck_dim, num_time_tokens=cfg.num_time_tokens,
        num_self_cond_cfg_tokens=cfg.num_self_cond_cfg_tokens,
        num_model_mode_tokens=cfg.num_model_mode_tokens,
    ).to(device).float()

    # Eval uses the EMA weights -- _build_eval_model() loads ema_params1, and the
    # reference 24.1 is an EMA number. Loading `params` here would quietly
    # measure a different model.
    src = ckpt.get("ema_params1") or params
    which = "ema_params1" if ckpt.get("ema_params1") else "params"
    missing, unexpected = model.load_state_dict(
        {k: v.float() for k, v in src.items()}, strict=False)
    model.eval()
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[smoke] loaded {which}: {n_param/1e6:.1f}M params  "
          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"[smoke]   missing[:5]={missing[:5]}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(hf_id(cfg.tokenizer_name or cfg.encoder_model_name))
    jt = AutoTokenizer.from_pretrained(hf_id("gpt2-large"))
    print(f"[smoke] tokenizers OK (elf vocab {tok.vocab_size}, judge {jt.name_or_path})", flush=True)

    sc = SamplingConfig(sampling_method="sde", num_sampling_steps=[args.steps],
                        cfgs=[1], self_cond_cfg_scales=[3],
                        time_schedule="logit_normal", sde_gamma=1.5)
    cfg_scale, sccfg = 1.0, 3.0
    res = {"n_param_M": n_param / 1e6, "weights": which, "depth": depth,
           "hidden": hidden, "text_enc_dim": text_dim, "vocab": vocab,
           "steps": args.steps, "n_samples": args.n_samples,
           "threads": torch.get_num_threads(), "dtype": "fp32", "device": "cpu"}

    # ---- A. per-forward wall clock -------------------------------------------
    for bs in (1, args.n_samples):
        z = torch.randn(bs, cfg.max_length, text_dim) * cfg.denoiser_noise_scale
        zin = torch.cat([z, torch.zeros_like(z)], dim=-1)
        tb = torch.full((bs,), 0.5)
        scb = torch.full((bs,), sccfg)
        model(zin, tb, self_cond_cfg_scale=scb)          # warm up
        t0 = time.time(); n_rep = 3
        for _ in range(n_rep):
            model(zin, tb, self_cond_cfg_scale=scb)
        per = (time.time() - t0) / n_rep
        res[f"sec_per_forward_bs{bs}"] = per
        print(f"[smoke] forward bs={bs}: {per:.2f}s  "
              f"=> K={args.steps} trajectory ~{per*(args.steps+1)/60:.1f} min", flush=True)

    # ---- B. headline: upstream sampler, SDE K=32 -----------------------------
    from utils.sampling_utils import get_sampling_steps
    g = torch.Generator().manual_seed(args.seed)
    t_steps = get_sampling_steps(args.steps, time_schedule="logit_normal",
                                 P_mean=cfg.denoiser_p_mean, P_std=cfg.denoiser_p_std,
                                 device="cpu", dtype=torch.float32)
    z0 = torch.randn((args.n_samples, cfg.max_length, text_dim),
                     generator=g, dtype=torch.float32) * cfg.denoiser_noise_scale
    print(f"[smoke] rolling out upstream sampler, K={args.steps}, n={args.n_samples}...", flush=True)
    t0 = time.time()
    latent = _generate_samples_single_batch(
        model=model, generator=g, z=z0, t_steps=t_steps,
        cond_seq=None, cond_seq_mask=None, config=cfg, sampling_config=sc,
        cfg_scale=cfg_scale, self_cond_cfg_scale=sccfg)
    res["gen_seconds"] = time.time() - t0
    print(f"[smoke] generation: {res['gen_seconds']:.1f}s", flush=True)

    t0 = time.time()
    ids = _dlm_decode_batch(z=latent, model=model, t_final_val=t_steps[-1].item(),
                            config=cfg, self_cond_cfg_scale=sccfg)
    res["decode_seconds"] = time.time() - t0

    ids = mask_after_eos(ids, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    texts = tok.batch_decode(ids, skip_special_tokens=True)
    res["entropy_elf"] = metrics.unigram_entropy_elf(texts, jt)   # comparable to 5.15
    res["entropy_pooled_t5"] = metrics.unigram_entropy(ids, pad_id=tok.pad_token_id)
    res["repetition_4gram"] = metrics.repetition_rate(ids, n=4, pad_id=tok.pad_token_id)
    res["sample_texts"] = [x[:300] for x in texts[:3]]
    res["all_texts"] = texts          # so metrics can be recomputed without regenerating

    print("[smoke] scoring gen-PPL under gpt2-large...", flush=True)
    jm = AutoModelForCausalLM.from_pretrained(hf_id("gpt2-large")).eval()
    res["gen_ppl"] = metrics.generative_perplexity(ids, jm, jt, tok, device="cpu")
    del jm

    print(f"\n[smoke] gen-PPL {res['gen_ppl']:.2f} (ref 24.1) | "
          f"entropy(ELF-def) {res['entropy_elf']:.3f} (ref 5.15) | "
          f"entropy(pooled) {res['entropy_pooled_t5']:.3f} | "
          f"rep-4gram {res['repetition_4gram']:.3f}", flush=True)

    # ---- C. our sampler == upstream, deterministic ODE ------------------------
    if not args.skip_equiv:
        print("[smoke] equivalence check (ODE, deterministic)...", flush=True)
        sc_ode = SamplingConfig(sampling_method="ode", num_sampling_steps=[8], cfgs=[1],
                                self_cond_cfg_scales=[3], time_schedule="uniform",
                                sde_gamma=0.0)
        t_ode = get_sampling_steps(8, time_schedule="uniform", device="cpu", dtype=torch.float32)
        z_small = torch.randn((2, cfg.max_length, text_dim),
                              generator=torch.Generator().manual_seed(7)) * cfg.denoiser_noise_scale
        up = _generate_samples_single_batch(
            model=model, generator=torch.Generator().manual_seed(7), z=z_small,
            t_steps=t_ode, cond_seq=None, cond_seq_mask=None, config=cfg,
            sampling_config=sc_ode, cfg_scale=cfg_scale, self_cond_cfg_scale=sccfg)
        mine = sample(model=model, config=cfg, sampling_config=sc_ode, t_steps=t_ode,
                      z0=z_small, eps=torch.zeros(7, 2, cfg.max_length, text_dim),
                      cfg_scale=cfg_scale, self_cond_cfg_scale=sccfg,
                      perturb=Perturbation(), device="cpu").z_final
        d = float((mine - up).abs().max())
        rel = d / float(up.abs().max())
        res["equiv_max_abs_diff"] = d
        res["equiv_max_rel_diff"] = rel
        print(f"[smoke] our sampler vs upstream: max|diff|={d:.3e} rel={rel:.3e}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[smoke] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
