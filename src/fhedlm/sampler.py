"""Deterministic, replayable, perturbation-instrumented ELF sampler.

This is a reimplementation of ELF's `_generate_samples_single_batch`
(third_party/ELF/src/utils/generation_utils.py) with three changes, all of them
required before Experiment 1 means anything:

1. **Fixed t-grid.** Upstream's default `time_schedule: logit_normal` draws a
   *random* sorted t-grid on every call, so two runs of the same model differ
   for reasons unrelated to any perturbation. The grid is drawn once here and
   replayed.

2. **Replayed SDE noise.** `_sde_step` draws its per-step noise with
   `torch.randn(..., device=...)` and ignores the `generator` it was handed
   (sampling_utils.py:241), so it is not reproducible even under a seed on CUDA.
   All per-step noise is pre-drawn and replayed identically across conditions.

   1 and 2 together make baseline and perturbed runs common-random-number
   paired: the *only* difference between two trajectories is the perturbation.
   Without this the measurement is swamped by sampler variance -- the SDE
   injects noise at ~47% of the latent RMS per step at gamma=1.5, K=32, which is
   orders of magnitude above the sigma_rel=1e-5 end of the sweep.

3. **Perturbation hooks** on both state channels carried across steps: `z` (the
   flow state) and `x_pred` (the self-conditioning input). An FHE evaluation
   corrupts both, and x_pred is the interesting one -- it is the feedback path.

Everything else -- the step maths, the final ODE step, the decode pass -- is
upstream's, and `verify_matches_upstream()` checks that claim numerically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from utils.sampling_utils import (  # ELF's own, unmodified
    restore_cond, _forward_sample, net_out_to_v_x,
)
from fhedlm.perturb import Perturbation


@dataclass
class Trajectory:
    """Per-step record. Full states are optional -- they are large."""
    z_final: torch.Tensor
    tokens: Optional[torch.Tensor] = None
    z_rms: list = field(default_factory=list)
    x_rms: list = field(default_factory=list)
    z_states: list = field(default_factory=list)   # kept only if keep_states
    x_states: list = field(default_factory=list)


def draw_schedule(n_steps: int, batch: int, seq_len: int, d_model: int,
                  config, seed: int, device, dtype=torch.float32):
    """Draw the t-grid and every per-step noise tensor ONCE, for replay.

    Returned as CPU tensors so a whole sweep of conditions can share one draw
    without holding several GB of GPU memory.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    # The t-grid MUST be drawn from `g`. ELF's get_sampling_steps() draws its
    # logit-normal grid with an unseeded torch.randn, so calling it here would
    # take the grid from the GLOBAL RNG -- meaning two runs in the same process
    # get DIFFERENT schedules, and the common-random-number pairing that step 4
    # depends on is silently broken. The logit-normal branch is reproduced
    # verbatim below (sampling_utils.py:61-72) with the generator threaded in.
    if config.time_schedule == "uniform":
        t_steps = torch.linspace(0.0, 1.0, n_steps + 1, dtype=dtype)
    elif config.time_schedule == "logit_normal":
        z = torch.randn((n_steps - 1,), generator=g, dtype=dtype) * config.denoiser_p_std \
            + config.denoiser_p_mean
        steps = torch.sort(torch.sigmoid(z)).values
        t_steps = torch.cat([torch.zeros(1, dtype=dtype), steps,
                             torch.ones(1, dtype=dtype)], dim=0)
    else:
        raise ValueError(f"unknown time_schedule: {config.time_schedule}")
    z0 = torch.randn((batch, seq_len, d_model), generator=g, dtype=dtype) \
        * config.denoiser_noise_scale
    # n_steps-1 SDE steps; the last step is always ODE and draws no noise.
    eps = torch.randn((max(n_steps - 1, 0), batch, seq_len, d_model),
                      generator=g, dtype=dtype) * config.denoiser_noise_scale
    return t_steps, z0, eps


@torch.no_grad()
def sample(model, config, sampling_config, t_steps, z0, eps,
           cfg_scale: float, self_cond_cfg_scale: float,
           perturb: Optional[Perturbation] = None,
           perturb_seed: int = 0, keep_states: bool = False,
           step_hook=None, device="cuda") -> Trajectory:
    """Roll out the sampler under a fixed schedule and a fixed perturbation."""
    perturb = perturb or Perturbation()
    pgen = torch.Generator(device=device).manual_seed(perturb_seed)

    z = z0.to(device)
    batch, seq_len, d_model = z.shape
    cond_seq = torch.zeros_like(z)
    cond_seq_mask = torch.zeros((batch, seq_len), dtype=z.dtype, device=device)
    x_pred = torch.zeros_like(z)

    step_kwargs = dict(model=model, config=config, cfg_scale=cfg_scale,
                       self_cond_cfg_scale=self_cond_cfg_scale,
                       cond_seq=cond_seq, cond_seq_mask=cond_seq_mask)
    gamma = getattr(sampling_config, "sde_gamma", 0.0)
    traj = Trajectory(z_final=None)
    n = t_steps.shape[0]

    def record(z_, x_):
        traj.z_rms.append(float(z_.float().pow(2).mean().sqrt()))
        traj.x_rms.append(float(x_.float().pow(2).mean().sqrt()))
        if keep_states:
            traj.z_states.append(z_.detach().to("cpu", torch.float16))
            traj.x_states.append(x_.detach().to("cpu", torch.float16))

    for i in range(n - 2):
        if step_hook is not None:
            step_hook(i)
        t, t_next = float(t_steps[i]), float(t_steps[i + 1])
        h = t_next - t
        alpha = max(0.0, min(1.0, 1.0 - gamma * h))
        t_back = alpha * t
        # Replayed noise, in place of _sde_step's unseeded torch.randn.
        e = eps[i].to(device)
        z_back = restore_cond(alpha * z + (1.0 - alpha) * e, cond_seq, cond_seq_mask)
        t_batch = torch.full((batch,), t_back, dtype=z.dtype, device=device)
        v_pred, x_pred = _forward_sample(z=z_back, t_batch=t_batch,
                                         x_pred_prev=x_pred, **step_kwargs)
        z = z_back + (t_next - t_back) * v_pred
        # Perturb AFTER the step: this is the state an FHE circuit would carry
        # into the next round, error and all.
        z = perturb.apply(z, "z", pgen)
        x_pred = perturb.apply(x_pred, "x_pred", pgen)
        record(z, x_pred)

    # Upstream always finishes on an ODE step.
    if step_hook is not None:
        step_hook(n - 2)
    t, t_next = float(t_steps[-2]), float(t_steps[-1])
    t_batch = torch.full((batch,), t, dtype=z.dtype, device=device)
    v_pred, x_pred = _forward_sample(z=z, t_batch=t_batch,
                                     x_pred_prev=x_pred, **step_kwargs)
    z = z + (t_next - t) * v_pred
    z = perturb.apply(z, "z", pgen)
    record(z, x_pred)

    traj.z_final = z
    return traj
