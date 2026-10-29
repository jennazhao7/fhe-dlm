"""E1 zero-shot: how far is the frozen MDLM-OWT from the E5 circuit?

The encrypted layer (scripts/e0_layer_audit.py, 18 levels) computes none of
LayerNorm, softmax or exact GELU. It computes
  norm   c * (x - mean(x)) * w      c a PUBLIC constant, no data-dependent 1/std
  attn   (q.k)^2 / Z @ V            Z a public constant, no row normalisation
  GELU   Chebyshev deg 31 on [-8, 8], evaluated as the polynomial: outside the
         interval it diverges, exactly as it would under CKKS
E1 fine-tunes the model into that form. Before training anything, `approx`
swaps each op into the frozen model -- alone, then all together -- to measure
what E1 has to recover, op by op. Public constants are calibrated on a split
disjoint from the evaluation text, per norm / layer / head and, where it is
free, per public index: the token's mask status (public in our protocol) and
its position (a plaintext-vector multiply costs what a scalar one does). For
the norm, c minimises E||c*xc - xc/sigma||^2, i.e. c = sum(sigma) / sum(sigma^2).
GELU degree costs depth: deg 31 = 7 levels (audited), 63 = 8, 127 = 9; a layer
is 18 of the 20 levels a two-pass bootstrap leaves, so deg 63 is affordable.

`spikes`: where the remaining --pin0 massive activations (S_max 17-18k in
results/plain/noise2_pin0.json) come from -- step, block, position and the
token sitting there -- and whether excluding <|endoftext|> from sampling
removes them, and with them the 1-pass (b = 11) gen-PPL penalty.

All runs use plain_probe.py's --pin0 semantics (content token at position 0).

    python scripts/approx_probe.py approx --out results/plain/approx.json
    python scripts/approx_probe.py spikes --out results/plain/spikes.json
"""
import argparse, json, math, os, sys, time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import plain_probe as pp   # noqa: E402  (stubs, loader, NLL, sampler, judge)

# what the forward pass currently computes; the hooks below read it
STATE = {"layer": 0, "norm": None, "attn": None, "gelu": None, "calib": None, "final_exact": False, "resid_max": 0.0,
         "nobos": False, "x": None, "step": 0, "spikes": None, "oor": [0, 0]}


# ---------------------------------------------------------------- GELU
def gelu_np(z):
    return 0.5 * z * (1 + np.tanh(math.sqrt(2 / math.pi) * (z + 0.044715 * z ** 3)))


def cheb_fit(R, deg):
    """Chebyshev interpolant of GELU on [-R, R] (what EvalChebyshevFunction
    builds), plus its max error on the interval."""
    c = np.polynomial.chebyshev.chebinterpolate(lambda t: gelu_np(R * t), deg)
    t = np.linspace(-1, 1, 20001)
    err = float(np.abs(np.polynomial.chebyshev.chebval(t, c) - gelu_np(R * t)).max())
    return torch.tensor(c, dtype=torch.float32), err


def cheb_eval(c, t):
    """Clenshaw, float32: diverges outside [-1, 1] like the CKKS polynomial."""
    b1 = torch.zeros_like(t); b2 = torch.zeros_like(t)
    for ck in reversed(c[1:].tolist()):
        b1, b2 = 2 * t * b1 - b2 + ck, b1
    return t * b1 - b2 + float(c[0])


# ---------------------------------------------------------------- hooks
def install(m):
    """Norm, GELU and attention hooks; all are no-ops while STATE says exact."""
    blocks = pp.blocks(m)
    norms = []
    for i, b in enumerate(blocks):
        b.register_forward_pre_hook(lambda mod, inp, i=i: STATE.__setitem__("layer", i))
        norms += [b.norm1, b.norm2]
        b.mlp[1].register_forward_hook(gelu_hook)
    norms.append(m.backbone.output_layer.norm_final)
    for j, nm in enumerate(norms):
        nm.register_forward_hook(lambda mod, inp, out, j=j: norm_hook(j, mod, inp, out))
    mod = sys.modules[type(blocks[0]).__module__]
    mod.regular_attention_multi_headed = attention     # resolved at call time by DDiTBlock.forward
    return len(norms)


def norm_hook(j, mod, inp, out):
    x = inp[0].float()
    cal = STATE["calib"]
    if cal is not None and cal.get("collect"):
        sd = x.var(-1, unbiased=False).add(1e-5).sqrt()         # F.layer_norm's eps
        idx = (pp.CUR["mask"].long(), torch.arange(x.shape[1], device=x.device).expand(x.shape[:2]))
        cal["s1"][j].index_put_(idx, sd.double(), accumulate=True)
        cal["s2"][j].index_put_(idx, sd.double() ** 2, accumulate=True)
        return None
    if j == len(STATE["norm_c"]["layer"]) - 1:                    # final norm: residual-stream scale
        STATE["resid_max"] = max(STATE["resid_max"], float(x.abs().max()))
        if STATE["final_exact"]:
            return None
    if STATE["norm"] is None:
        return None
    c = STATE["norm_c"][STATE["norm"]][j]                         # (2, L) by (mask, position)
    mk = pp.CUR["mask"].long()
    cc = c[mk, torch.arange(x.shape[1], device=x.device)[None, :].expand_as(mk)][..., None]
    return (x - x.mean(-1, keepdim=True)) * cc * mod.weight[None, None, :]


def gelu_hook(mod, inp, out):
    cal = STATE["calib"]
    if cal is not None and cal.get("collect"):
        cal["gelu_max"] = max(cal["gelu_max"], float(inp[0].abs().max()))
        return None
    if STATE["gelu"] is None:
        return None
    c, R = STATE["gelu"]
    t = inp[0].float() / R
    STATE["oor"][0] += int((t.abs() > 1).sum()); STATE["oor"][1] += t.numel()
    return cheb_eval(c.to(t.device), t)


def attention(qkv):
    """Drop-in for modeling_mdlm's regular_attention_multi_headed(qkv),
    qkv (b, s, 3, h, d)."""
    q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
    s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])            # (b, h, s, s)
    mode, i, cal = STATE["attn"], STATE["layer"], STATE["calib"]
    if cal is not None and cal.get("collect"):
        mk = pp.CUR["mask"].long()                                  # query mask status
        r2, re = (s ** 2).sum(-1).double(), s.double().exp().sum(-1)
        for ms in (0, 1):
            w = (mk == ms)[:, None, :].double()
            cal["z2"][i, :, ms] += (r2 * w).sum((0, 2)); cal["ze"][i, :, ms] += (re * w).sum((0, 2))
            cal["zn"][i, :, ms] += w.sum((0, 2)).expand(s.shape[1])
        cal["smax"][i] = max(float(cal["smax"][i]), float(s.abs().max()))
        mode = None
    if mode is None:
        p = s.softmax(-1)
    elif mode == "pow2_row":                                        # diagnostic: shape only
        p = s ** 2; p = p / p.sum(-1, keepdim=True)
    else:
        kind, by = mode.split("_")                                  # pow2|exp, Z|Zmask
        num = s ** 2 if kind == "pow2" else s.double().exp()        # exp: float64 so it is Z, not overflow, that fails
        Z = STATE["attn_Z"][f"{kind}_{by}"][i].to(s.device)         # (h, 2)
        mk = pp.CUR["mask"].long()
        p = (num / Z[:, mk].permute(1, 0, 2)[..., None]).float()    # (b, h, s, 1)
    o = (p @ v).transpose(1, 2)
    return o.reshape(o.shape[0], o.shape[1], -1)


# ---------------------------------------------------------------- calibration
@torch.no_grad()
def calibrate(m, data, n_norms, bs, seed=5):
    """Exact forward on the calibration split, under the mask states a pin0
    run sees: per-row mask ratio ~ U(0, 1), position 0 never masked, plus one
    row in 8 fully masked (the first sampling step, position 0 included)."""
    L, nb, H = data.shape[1], len(pp.blocks(m)), m.config.n_heads
    z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=pp.DEV)
    cal = {"collect": True, "s1": z(n_norms, 2, L), "s2": z(n_norms, 2, L),
           "z2": z(nb, H, 2), "ze": z(nb, H, 2), "zn": z(nb, H, 2), "smax": [0.0] * nb, "gelu_max": 0.0}
    STATE["calib"] = cal
    g = torch.Generator().manual_seed(seed)
    for st in range(0, len(data), bs):
        x = data[st:st + bs]
        r = torch.rand(len(x), 1, generator=g)
        mk = torch.rand(x.shape, generator=g) < r
        mk[:, 0] = False
        mk[torch.rand(len(x), generator=g) < 1 / 8] = True
        pp.forward_logits(m, torch.where(mk, torch.full_like(x, pp.MASK), x).to(pp.DEV))
    STATE["calib"] = None
    # norm constants c = sum(sigma) / sum(sigma^2), pooled three ways
    s1, s2 = cal["s1"], cal["s2"]
    c_mp = s1 / s2.clamp_min(1e-30)
    c_m = (s1.sum(-1) / s2.sum(-1).clamp_min(1e-30))[..., None].expand_as(s1)
    c_l = (s1.sum((1, 2)) / s2.sum((1, 2)))[:, None, None].expand_as(s1)
    c_mp = torch.where(s2 > 0, c_mp, c_m)       # unseen (mask, position) cells -> per-mask value
    c_m = torch.where(s2.sum(-1, keepdim=True) > 0, c_m, c_l)
    STATE["norm_c"] = {"layer": c_l.float(), "mask": c_m.float(), "maskpos": c_mp.float()}
    zn = cal["zn"].clamp_min(1)
    STATE["attn_Z"] = {"pow2_Zmask": (cal["z2"] / zn).float(), "exp_Zmask": cal["ze"] / zn,
                       "pow2_Z": (cal["z2"].sum(-1, keepdim=True) / zn.sum(-1, keepdim=True)).expand(-1, -1, 2).float(),
                       "exp_Z": (cal["ze"].sum(-1, keepdim=True) / zn.sum(-1, keepdim=True)).expand(-1, -1, 2)}
    inv = 1 / (s1 / s2.clamp_min(1e-30))
    return {"gelu_in_max": cal["gelu_max"], "score_abs_max_per_layer": cal["smax"],
            "norm_c_cv_over_positions": [float(c_mp[j, 1].std() / c_mp[j, 1].mean()) for j in range(n_norms)],
            "norm_sigma_pos0_unmasked": [float(inv[j, 0, 0]) for j in range(n_norms)]}


# ---------------------------------------------------------------- eval
@torch.no_grad()
def nll_kl(m, data, ratio, bs, cfg, seed=7):
    """Masked-token NLL under `cfg`, and mean KL(exact || cfg) of the
    predictive at masked positions (same masks, exact pass first)."""
    g = torch.Generator().manual_seed(seed)
    tot = kl = 0.0; cnt = 0
    for st in range(0, len(data), bs):
        x = data[st:st + bs].to(pp.DEV)
        xm, mk = pp.mask_batch(x, ratio, g)
        set_cfg({})
        lp0 = pp.forward_logits(m, xm).float().log_softmax(-1)[mk]
        set_cfg(cfg)
        lp = pp.forward_logits(m, xm).float().log_softmax(-1)[mk]
        lp = torch.nan_to_num(lp, nan=-1e4, neginf=-1e4)
        tot += float(-lp.gather(-1, x[mk][:, None]).sum()); cnt += int(mk.sum())
        kl += float((lp0.exp() * (lp0.clamp_min(-1e4) - lp)).sum())
    set_cfg({})
    return tot / max(cnt, 1), kl / max(cnt, 1)


_exact_fwd = pp.forward_logits


def forward_logits_checked(m, x):
    """Raise on the host before NaN/inf reach torch.multinomial: its
    device-side assert kills the CUDA context (approx run 1640480 died at
    gelu_R8_d31). -inf alone is fine ([MASK] is -inf by construction)."""
    lg = _exact_fwd(m, x)
    if bool(torch.isnan(lg).any() | torch.isposinf(lg).any() | (lg.amax(-1) == -float("inf")).any()):
        raise FloatingPointError(f"non-finite logits at step with {int((x == pp.MASK).sum())} masked")
    return lg


def set_cfg(cfg):
    STATE["norm"], STATE["attn"] = cfg.get("norm"), cfg.get("attn")
    STATE["gelu"], STATE["final_exact"] = cfg.get("gelu"), cfg.get("final_exact", False)


def cmd_approx(a):
    pp.PIN0 = True
    m = pp.load_model(pp.DEV)
    n_norms = install(m)
    data = pp.load_chunks(a.L, a.n + a.n_cal)
    evalset, calset = data[: a.n], data[a.n:]     # eval = the first n, as in noise2_*.json
    cal = calibrate(m, calset, n_norms, a.bs)
    gmax = cal["gelu_in_max"]
    print(f"[approx] calib: max|gelu in| {gmax:.1f}, max|score| {max(cal['score_abs_max_per_layer']):.1f}", flush=True)
    judge = None
    if a.gen:
        from transformers import AutoModelForCausalLM
        judge = AutoModelForCausalLM.from_pretrained(pp.snapshot("openai-community/gpt2-large")).to(pp.DEV).eval()
    Rw = float(2 ** math.ceil(math.log2(gmax * 1.05)))           # smallest power-of-2 interval covering it
    gel = {}
    for R, d in [(8, 31), (16, 31), (Rw, 31), (Rw, 63), (Rw, 127)]:
        c, err = cheb_fit(R, d)
        gel[f"gelu_R{R:g}_d{d}"] = ((c, R), err)
    configs = [("exact", {})]
    configs += [(k, {"gelu": v[0]}) for k, v in gel.items()]
    configs += [(f"norm_{n}", {"norm": n}) for n in ("layer", "mask", "maskpos")]
    configs += [("norm_maskpos_final_exact", {"norm": "maskpos", "final_exact": True})]
    configs += [(f"attn_{t}", {"attn": t}) for t in ("pow2_row", "pow2_Z", "pow2_Zmask", "exp_Zmask")]
    configs += [(f"e5_circuit_d{d}", {"norm": "maskpos", "attn": "pow2_Zmask", "gelu": gel[f"gelu_R{Rw:g}_d{d}"][0]})
                for d in (63, 127)]
    out = {"model": pp.MDLM, "L": a.L, "n_eval": int(len(evalset)), "n_calib": int(len(calset)), "pin0": True,
           "gen_n": a.gen_n, "steps": a.steps, "calibration": cal,
           "gelu_fits": {k: {"R": v[0][1], "max_err_on_interval": v[1]} for k, v in gel.items()}, "runs": []}
    for name, cfg in configs:
        t0 = time.time(); STATE["oor"] = [0, 0]
        r = {"config": name}
        for ratio in (0.15, 0.5, 0.9):
            STATE["resid_max"] = 0.0
            r[f"nll_mask{ratio}"], r[f"kl_mask{ratio}"] = nll_kl(m, evalset, ratio, a.bs, cfg)
        r["final_resid_max_mask0.9"] = STATE["resid_max"]           # exact and approx passes both counted
        if a.gen:
            set_cfg(cfg)
            pp.forward_logits = forward_logits_checked
            try:
                xs = pp.sample(m, a.gen_n, a.L, a.steps, seed=11)
                r["gen_ppl_gpt2_large"] = pp.gen_ppl(judge, xs)
                r["gen_unigram_entropy"] = pp.unigram_entropy(xs)
            except FloatingPointError as e:                       # the circuit diverged
                r["gen_error"] = str(e)[:200]
            pp.forward_logits = _exact_fwd
            set_cfg({})
        if STATE["oor"][1]:
            r["gelu_frac_outside_interval"] = STATE["oor"][0] / STATE["oor"][1]
        r["seconds"] = time.time() - t0
        out["runs"].append(r)
        print("[approx] " + json.dumps(r), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)


# ---------------------------------------------------------------- spikes
_fwd = pp.forward_logits


def forward_logits_traced(m, x):
    """pp.forward_logits, plus: remember the ids in flight, count calls, and
    (when STATE['nobos']) never sample <|endoftext|>."""
    STATE["x"] = x; STATE["step"] += 1
    lg = _fwd(m, x)
    if STATE["nobos"]:
        lg[..., pp.BOS] = -float("inf")
    return lg


def spike_hook(i, thresh):
    def h(mod, inp, out):
        sp = STATE["spikes"]
        if sp is None:
            return
        a = out.detach().abs().amax(-1)                           # (b, L)
        sp["max"] = max(sp["max"], float(a.max()))
        idx = torch.nonzero(a > thresh)                           # one host sync per call, not per event
        if not len(idx):
            return
        x = STATE["x"]
        bb, pq = idx[:, 0], idx[:, 1]
        cols = torch.stack([bb, pq, a[bb, pq].round().long(), x[bb, pq],
                            (x == pp.BOS).sum(-1)[bb], (x == pp.MASK).sum(-1)[bb]], -1).tolist()
        for b, p, v, tok, ne, nm in cols:
            sp["rows"].append({"step": STATE["step"], "row": b, "block": i, "pos": p, "val": v,
                               "tok": "MASK" if tok == pp.MASK else "EOS" if tok == pp.BOS else "content",
                               "n_eos_in_seq": ne, "n_masked_in_seq": nm})
    return h


def summarize(sp, thresh):
    rows = sp["rows"]
    first = {}
    for r in rows:                                                # one event per (forward, sequence, position)
        first.setdefault((r["step"], r["row"], r["pos"]), r)
    ev = list(first.values())
    return {"thresh": thresh, "max_abs": sp["max"], "n_spike_token_events": len(ev),
            "by_token": dict(Counter(r["tok"] for r in ev)),
            "pos0_share": sum(r["pos"] == 0 for r in ev) / max(len(ev), 1),
            "top_positions": Counter(r["pos"] for r in ev).most_common(8),
            "blocks": dict(Counter(r["block"] for r in rows)),
            "with_eos_in_seq": sum(r["n_eos_in_seq"] > 0 for r in ev),
            "examples": ev[:20]}


def cmd_spikes(a):
    pp.PIN0 = True
    pp.forward_logits = forward_logits_traced                     # pp.sample / masked_nll look it up at call time
    m = pp.load_model(pp.DEV)
    for i, b in enumerate(pp.blocks(m)):
        b.register_forward_hook(spike_hook(i, a.thresh))
    data = pp.load_chunks(a.L, a.n)
    from transformers import AutoModelForCausalLM
    judge = AutoModelForCausalLM.from_pretrained(pp.snapshot("openai-community/gpt2-large")).to(pp.DEV).eval()
    out = {"model": pp.MDLM, "L": a.L, "pin0": True, "gen_n": a.gen_n, "steps": a.steps, "runs": []}
    # 1. attribution: NLL masks, then generation, clean, EOS allowed (= noise2_pin0 clean)
    for phase in ("nll", "gen"):
        STATE["spikes"] = {"rows": [], "max": 0.0}; STATE["nobos"] = False
        if phase == "nll":
            for ratio in (0.15, 0.5, 0.9):
                pp.masked_nll(m, data, ratio, a.bs, seed=7)
                print(f"[spikes] nll ratio {ratio}: {len(STATE['spikes']['rows'])} rows so far", flush=True)
        else:
            pp.sample(m, a.gen_n, a.L, a.steps, seed=11)
        s = summarize(STATE["spikes"], a.thresh)
        out[f"attribution_{phase}"] = s
        print(f"[spikes] {phase}: max {s['max_abs']:.0f}, events {s['n_spike_token_events']} by token {s['by_token']}, "
              f"pos0 {s['pos0_share']:.2f}, top pos {s['top_positions'][:4]}", flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
    STATE["spikes"] = None
    # 2. does never sampling <|endoftext|> make one bootstrap pass (b = 11) enough?
    for nobos in (False, True):
        for bits in a.bits:
            b = None if bits <= 0 else bits
            STATE["nobos"] = nobos
            nz = pp.Noise(m, b)
            t0 = time.time()
            xs = pp.sample(m, a.gen_n, a.L, a.steps, seed=11)
            r = {"nobos": nobos, "bits": b, "gen_ppl_gpt2_large": pp.gen_ppl(judge, xs),
                 "gen_unigram_entropy": pp.unigram_entropy(xs), "eos_per_seq": float((xs == pp.BOS).sum(-1).float().mean())}
            if nz.S:
                r["S_median"] = sorted(nz.S)[len(nz.S) // 2]; r["S_max"] = max(nz.S)
            nz.remove()
            r["seconds"] = time.time() - t0
            out["runs"].append(r)
            print("[spikes] " + json.dumps(r), flush=True)
            json.dump(out, open(a.out, "w"), indent=1)
    STATE["nobos"] = False


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("approx")
    p.add_argument("--n-cal", type=int, default=128)
    p.add_argument("--gen", action="store_true")
    s = sub.add_parser("spikes")
    s.add_argument("--thresh", type=float, default=2000.0)
    s.add_argument("--bits", type=float, nargs="+", default=[0, 11, 12, 13])
    for q in (p, s):
        q.add_argument("--L", type=int, default=256)
        q.add_argument("--n", type=int, default=256)
        q.add_argument("--bs", type=int, default=16)
        q.add_argument("--gen-n", type=int, default=64)
        q.add_argument("--steps", type=int, default=128)
        q.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    {"approx": cmd_approx, "spikes": cmd_spikes}[a.cmd](a)


if __name__ == "__main__":
    main()
