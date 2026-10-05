"""Perturbation models standing in for FHE/CKKS approximation error.

Each perturbation is expressed *relative to the tensor's own RMS*, so a single
sigma_rel sweep is comparable across the two state channels (z and x_pred),
which have different scales: z lives at denoiser_noise_scale=2.0 early in the
trajectory and decays toward latent_std=0.2, while x_pred sits near 0.2
throughout.  An absolute sigma would silently mean something different at every
timestep and make the sweep uninterpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


def _rms(x: torch.Tensor) -> torch.Tensor:
    """Per-example RMS, kept as a (B,1,1) tensor so scaling stays per-sequence."""
    return x.float().pow(2).mean(dim=tuple(range(1, x.dim())), keepdim=True).sqrt()


@dataclass
class Perturbation:
    """One perturbation condition.

    kind:
      "none"      -- reference trajectory
      "additive"  -- x + N(0, (sigma_rel*RMS)^2), the CKKS noise model
      "mult"      -- x * (1 + N(0, sigma_rel^2)), relative/rescaling error
      "quant"     -- round to `bits` uniformly over [-q*RMS, q*RMS], the
                     fixed-point/precision model
    """

    kind: str = "none"
    sigma_rel: float = 0.0
    bits: int = 16
    quant_range_rms: float = 4.0   # clip at +-4 RMS; covers >99.99% of a Gaussian
    channels: tuple = ("z", "x_pred")

    def label(self) -> str:
        if self.kind == "none":
            return "reference"
        if self.kind == "quant":
            return f"quant{self.bits}b"
        return f"{self.kind}{self.sigma_rel:g}"

    def apply(self, x: torch.Tensor, channel: str,
              gen: Optional[torch.Generator] = None) -> torch.Tensor:
        if self.kind == "none" or channel not in self.channels:
            return x
        if self.kind == "additive":
            noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=gen)
            return x + noise * (self.sigma_rel * _rms(x).to(x.dtype))
        if self.kind == "mult":
            noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=gen)
            return x * (1.0 + self.sigma_rel * noise)
        if self.kind == "quant":
            # Symmetric uniform quantiser at `bits` total bits (1 sign + rest),
            # range set from the tensor's own RMS rather than a fixed constant:
            # a global range would clip the early high-variance steps and waste
            # every level on the late low-variance ones.
            scale = (self.quant_range_rms * _rms(x)).to(x.dtype)
            n_levels = 2 ** (self.bits - 1) - 1
            step = scale / n_levels
            return torch.clamp(torch.round(x / step), -n_levels, n_levels) * step
        raise ValueError(f"unknown perturbation kind: {self.kind}")


# The Experiment 1 grid, straight from the brief.
ADDITIVE_SIGMAS = [1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
MULT_SIGMAS = [1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
QUANT_BITS = [16, 12, 10, 8, 6]


def experiment1_grid(channels=("z", "x_pred")):
    """Reference first -- every later condition is scored as a delta against it."""
    grid = [Perturbation(kind="none")]
    grid += [Perturbation("additive", sigma_rel=s, channels=channels) for s in ADDITIVE_SIGMAS]
    grid += [Perturbation("mult", sigma_rel=s, channels=channels) for s in MULT_SIGMAS]
    grid += [Perturbation("quant", bits=b, channels=channels) for b in QUANT_BITS]
    return grid
