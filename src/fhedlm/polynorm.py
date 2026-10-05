"""Step 4: replace the data-dependent rsqrt in every RMSNorm with a polynomial.

RMSNorm computes  y = w * x * rsqrt(mean(x^2) + eps).  Under CKKS the only hard
part is rsqrt of a ciphertext; everything else is a ct-ct square, a sum over the
feature axis (rotations), and ct-plaintext multiplies. So the operator we must
approximate is

    f(v) = v^(-1/2)   on the range of v = mean(x^2) + eps actually observed.

Two things make the range the whole problem:
  * f is unbounded as v -> 0, so a fit interval that reaches too low is hopeless
    at any low degree;
  * the observed range differs by orders of magnitude between the 768-wide
    norms (norm1/norm2/final) and the 64-wide qk-norms inside attention.

The sibling project found exactly this along a different axis: a single global
interval for Mamba-2's exp(A*dt) gives infinite perplexity at every scale, while
a per-head interval at degree 4 costs <0.014 PPL
(/groups/tjung/jzhao7/FHE-S4-norm/mamba2-poly-exp/FINDINGS.md). Here the
candidate conditioning axes are the layer and the diffusion timestep. t is
public, so per-(layer, t) coefficient selection is free under FHE and leaks
nothing -- the server knows which round it is evaluating.

Chebyshev fitting is reused read-only from
/groups/tjung/jzhao7/FHE-S4-norm/mamba2-poly-exp/baby_mamba/polynomial.py
(`fit_general`). Its `approximation_report` is hardcoded to exp, so the rsqrt
error report below is ours.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from modules.layers import RMSNorm

# Which diffusion step is being evaluated. The timestep is public, so keying
# coefficients on it is legitimate; this is set by the sampler's step_hook.
CURRENT_STEP: int = -1


def set_step(i: int) -> None:
    global CURRENT_STEP
    CURRENT_STEP = int(i)


def _poly_eval(coeffs: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Horner. Under CKKS this would be Paterson-Stockmeyer; the arithmetic
    result is the same and only the level count differs, which is accounted for
    separately rather than simulated here."""
    out = torch.full_like(v, float(coeffs[-1]))
    for c in reversed(coeffs[:-1].tolist()):
        out = out * v + c
    return out


class PolyRMSNorm(nn.Module):
    """Drop-in RMSNorm whose rsqrt can be exact, recorded, or polynomial.

    NOTHING IS CLIPPED. The polynomial is evaluated as-is, including outside its
    fitted interval, because CKKS cannot clamp a ciphertext: there is no
    comparison, so a real FHE evaluation has no way to notice that an input left
    the interval. Simulating a clamp here would hide exactly the failure mode we
    are trying to measure. Out-of-range inputs are counted and their error is
    reported separately instead.
    """

    RESERVOIR = 4096   # v samples kept per (layer, step) for offline analysis

    def __init__(self, base: RMSNorm, layer_id: str):
        super().__init__()
        self.hidden_size = base.hidden_size
        self.eps = base.eps
        self.weight = base.weight
        self.layer_id = layer_id
        self.mode = "exact"              # exact | collect | poly
        self.ranges: Dict[int, Tuple[float, float]] = {}
        self.coeffs: Dict[int, torch.Tensor] = {}
        self.fit_interval: Dict[int, Tuple[float, float]] = {}
        # step -> dict of counters; reported per (layer, t)
        self.stats: Dict[int, dict] = {}
        self.samples: Dict[int, list] = {}   # step -> reservoir of v values

    def _stat(self, step: int) -> dict:
        return self.stats.setdefault(step, {
            "n_seen": 0, "n_out": 0, "max_rel_all": 0.0, "max_rel_out": 0.0})

    def _stash(self, step: int, v: torch.Tensor) -> None:
        buf = self.samples.setdefault(step, [])
        if len(buf) < self.RESERVOIR:
            flat = v.detach().flatten()
            take = min(self.RESERVOIR - len(buf), flat.numel())
            idx = torch.randperm(flat.numel())[:take]
            buf.extend(flat[idx].tolist())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dt = hidden_states.dtype
        v = hidden_states.float().pow(2).mean(dim=-1, keepdim=True) + self.eps

        if self.mode == "collect":
            lo, hi = float(v.min()), float(v.max())
            prev = self.ranges.get(CURRENT_STEP)
            self.ranges[CURRENT_STEP] = ((min(lo, prev[0]), max(hi, prev[1]))
                                         if prev else (lo, hi))
            self._stash(CURRENT_STEP, v)
            inv = torch.rsqrt(v)

        elif self.mode == "poly":
            c = self.coeffs.get(CURRENT_STEP, self.coeffs.get(-1))
            if c is None:
                raise RuntimeError(f"{self.layer_id}: no coefficients for step {CURRENT_STEP}")
            inv = _poly_eval(c, v)          # no clamping, by design
            exact = torch.rsqrt(v)
            rel = (inv - exact).abs() / exact.abs().clamp_min(1e-30)   # metric only
            st = self._stat(CURRENT_STEP)
            st["n_seen"] += int(v.numel())
            st["max_rel_all"] = max(st["max_rel_all"], float(rel.max()))
            iv = self.fit_interval.get(CURRENT_STEP, self.fit_interval.get(-1))
            if iv is not None:
                oob = (v < iv[0]) | (v > iv[1])
                n_oob = int(oob.sum())
                st["n_out"] += n_oob
                if n_oob:
                    st["max_rel_out"] = max(st["max_rel_out"], float(rel[oob].max()))
            self._stash(CURRENT_STEP, v)
        else:
            inv = torch.rsqrt(v)

        return self.weight.to(dt) * (hidden_states * inv.to(dt))


def install(model: nn.Module) -> Dict[str, PolyRMSNorm]:
    """Swap every RMSNorm for a PolyRMSNorm. Returns them keyed by module path."""
    out: Dict[str, PolyRMSNorm] = {}
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, RMSNorm):
                full = f"{name}.{child_name}" if name else child_name
                p = PolyRMSNorm(child, full)
                setattr(mod, child_name, p)
                out[full] = p
    return out


def set_mode(norms: Dict[str, PolyRMSNorm], mode: str) -> None:
    for n in norms.values():
        n.mode = mode
        if mode == "poly":
            n.max_rel_err, n.n_out_of_range, n.n_seen = 0.0, 0, 0


def rsqrt_report(coeffs: np.ndarray, lo: float, hi: float, n_grid: int = 20001) -> dict:
    """Error of P against v^-1/2 on [lo, hi]. Relative error is the figure that
    matters: rsqrt output scales the whole activation, so a fixed absolute error
    means very different things at the two ends of a wide interval."""
    x = np.linspace(lo, hi, n_grid)
    y = 1.0 / np.sqrt(x)
    p = np.polyval(np.asarray(coeffs, dtype=np.float64)[::-1], x)
    rel = np.abs(p - y) / np.maximum(np.abs(y), 1e-300)
    return {"degree": int(len(coeffs) - 1), "interval": [float(lo), float(hi)],
            "range_decades": float(np.log10(hi / max(lo, 1e-30))),
            "max_abs_error": float(np.abs(p - y).max()),
            "max_rel_error": float(rel.max()), "mean_rel_error": float(rel.mean())}


def fit_all(norms: Dict[str, PolyRMSNorm], degree: int, per_step: bool,
            margin: float = 0.10) -> dict:
    """Fit coefficients from the collected ranges.

    per_step=False -> one polynomial per layer, over the union of every step's
    range. per_step=True -> one per (layer, step). The margin widens the fit
    interval so eval-time activations just outside the calibration range are
    still interpolated rather than extrapolated; extrapolating v^-1/2 below the
    fit interval diverges fast.
    """
    from baby_mamba.polynomial import fit_general   # read-only reuse, see module docstring

    f = lambda v: 1.0 / np.sqrt(v)
    report = {}
    for name, n in norms.items():
        if not n.ranges:
            continue
        if per_step:
            items = [(s, lohi) for s, lohi in n.ranges.items()]
        else:
            lo = min(l for l, _ in n.ranges.values())
            hi = max(h for _, h in n.ranges.values())
            items = [(-1, (lo, hi))]
        n.coeffs, n.fit_interval, n.stats = {}, {}, {}
        worst = None
        for step, (lo, hi) in items:
            span = hi - lo
            lo_m = max(lo - margin * span, lo * 0.5, 1e-12)
            hi_m = hi + margin * span
            c = fit_general(f, degree, lo_m, hi_m, method="chebyshev")
            n.coeffs[step] = torch.tensor(c, dtype=torch.float32)
            n.fit_interval[step] = (lo_m, hi_m)
            r = rsqrt_report(c, lo_m, hi_m)
            if worst is None or r["max_rel_error"] > worst["max_rel_error"]:
                worst = r
        report[name] = worst
    return report


# Depth model for the accounting. Under CKKS one Newton step
#   y <- y * (3 - v*y^2) / 2
# is y^2 (ct-ct), v*y^2 (ct-ct), then y*(...) (ct-ct): 3 sequential ct-ct
# multiplies, so 3 levels. The (3 - .)/2 is a plaintext affine map and is free.
NEWTON_LEVELS_PER_ITER = 3


def poly_levels(degree: int) -> int:
    """Levels for a degree-d polynomial under Paterson-Stockmeyer."""
    return int(np.ceil(np.log2(degree + 1))) + 1


def newton_analysis(norms: Dict[str, PolyRMSNorm], degree: int = 3,
                    iters: int = 3, target: float = 1e-3) -> dict:
    """How far does Newton refinement rescue a cheap initial polynomial?

    Runs offline on the activation reservoirs already collected -- no generation
    needed. The initial guess is the per-(layer, t) degree-`degree` fit, which is
    what makes this cheap: Newton converges quadratically, so a poor-but-bounded
    start plus 2-3 iterations can beat a much higher-degree direct fit at lower
    depth. It only works if the initial guess is in the basin -- the iteration
    diverges if v*y^2 > 3 -- so the per-(layer, t) start matters.
    """
    from baby_mamba.polynomial import fit_general
    f = lambda x: 1.0 / np.sqrt(x)
    per_layer = {}
    for name, n in norms.items():
        for step, buf in n.samples.items():
            if not buf:
                continue
            v = np.asarray(buf, dtype=np.float64)
            lo, hi = float(v.min()), float(v.max())
            span = hi - lo
            lo_m = max(lo - 0.10 * span, lo * 0.5, 1e-12)
            hi_m = hi + 0.10 * span
            c = fit_general(f, degree, lo_m, hi_m, method="chebyshev")
            y = np.polyval(np.asarray(c)[::-1], v)
            exact = f(v)
            errs = [float((np.abs(y - exact) / exact).max())]
            for _ in range(iters):
                y = y * (3.0 - v * y * y) / 2.0
                errs.append(float((np.abs(y - exact) / np.abs(exact)).max()))
            per_layer[f"{name}@t{step}"] = {
                "decades": float(np.log10(hi / max(lo, 1e-30))),
                "max_rel_err_by_iter": errs,
            }
    # Worst case across all (layer, t) is what the configuration must survive.
    worst = [max(d["max_rel_err_by_iter"][i] for d in per_layer.values())
             for i in range(iters + 1)] if per_layer else []
    cheapest = None
    for i, e in enumerate(worst):
        if e <= target:
            cheapest = {"newton_iters": i, "max_rel_err": e,
                        "levels_per_norm": poly_levels(degree) + i * NEWTON_LEVELS_PER_ITER,
                        "init_degree": degree}
            break
    return {"target": target, "init_degree": degree,
            "worst_max_rel_err_by_iter": worst,
            "levels_by_iter": [poly_levels(degree) + i * NEWTON_LEVELS_PER_ITER
                               for i in range(iters + 1)],
            "cheapest_meeting_target": cheapest,
            "per_layer_t": per_layer}
