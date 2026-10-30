"""E1 pilot: KD fine-tune of MDLM-OWT into the E5 circuit, one op at a time.

approx.json showed that each E5 op alone breaks the frozen model (norms worst,
then attention shape, then GELU range). The pilot asks whether training
recovers it, how much, and at what cost, introducing the ops in that order
and evaluating after each stage:

  A  norms      LayerNorm -> c[norm, mask] * (x - mean x) * w, with c a public
                constant per norm and query mask status (the mean is linear,
                so it is free). Blended in: (x - mean) * w * ((1-lam)/sigma +
                lam * c), lam 0 -> 1.
  B  attention  softmax(s) -> (s + b)^2 / Z, s = q.k/sqrt(d), with b a public
                shift per (layer, head) (MPCFormer's "2Quad"; b = 0 is the
                pure square that approx.json scored at NLL 8) and Z public per
                (layer, head, query mask). Same depth as s^2: + b is free.
  C  GELU       deg-31 Chebyshev on [-8, 8] (the E5 polynomial). A squared
                hinge penalty on |u| > R0 pulls the rare inputs (0.04% > 8 in
                the teacher) inside; training evaluates the polynomial on
                clamp(u, -8, 8), evaluation does not clamp (CKKS cannot).
  D  noise      N(0, (2^-b * S)^2) added after every block, S = max|x| of that
                block output (the noise probe's model of a bootstrap), b = 11
                = one bootstrap pass RMS. If D holds, E5 needs only one pass.

Objective: KL(teacher || student) on masked positions, T = 1, teacher = the
frozen exact MDLM. Masks follow the pin0 protocol: per-row ratio ~ U(0, 1),
position 0 never masked, 1 row in 8 fully masked. Constants (c, b, Z) are
calibrated on the exact model, then trained (they stay public constants).
Data: WikiText-103 train (cached; MDLM was trained on OpenWebText, which is
not -- the full E1 run should use OWT). Eval matches approx.json: NLL / KL at
15/50/90% masking on 256 WikiText-103 validation chunks, gen-PPL + entropy
from the pin0 sampler, residual-stream max, fraction of GELU inputs outside
[-8, 8], with every introduced op at lam = 1 and GELU unclamped.

    python scripts/e1_kd.py --out results/plain/e1_pilot.json --ckpt-dir ckpt/e1_pilot
Resumes from the last finished stage in --ckpt-dir.
"""
import argparse, json, math, os, sys, time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import plain_probe as pp      # noqa: E402  (loader, data, sampler, judge)
import approx_probe as ap     # noqa: E402  (cheb_fit / cheb_eval, non-finite guard)
import spec_alpha             # noqa: E402  (train_tokens)

STAGES = ("A", "B", "C", "D")
# what the attention function / block hooks read; None = exact (the teacher never sets it)
S = {"attn": None, "collect": None, "noise_bits": None}


# ---------------------------------------------------------------- E5 ops
class FixedNorm(nn.Module):
    def __init__(self, ln, c0):
        super().__init__()
        self.weight, self.dim = ln.weight, ln.dim
        self.log_c = nn.Parameter(c0.log().float())            # (2,): query unmasked / masked
        self.lam = 0.0

    def forward(self, x):
        x = x.float()
        xc = x - x.mean(-1, keepdim=True)
        c = self.log_c.exp()[pp.CUR["mask"].long()][..., None]
        if self.lam < 1:
            c = (1 - self.lam) * (xc.pow(2).mean(-1, keepdim=True) + 1e-5).rsqrt() + self.lam * c
        return xc * c * self.weight[None, None, :]


class PowAttn(nn.Module):
    """Public constants of one layer's (s + b)^2 / Z attention."""
    def __init__(self, b0, z0):
        super().__init__()
        self.b = nn.Parameter(b0.float())                       # (h,)
        self.log_z = nn.Parameter(z0.log().float())             # (h, 2)
        self.lam = 0.0


def attention(qkv):
    """Drop-in for modeling_mdlm's regular_attention_multi_headed(qkv)."""
    q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
    s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])                  # (b, h, s, s)
    p = s.softmax(-1)
    if S["collect"] is not None:
        S["collect"](s.detach(), p.detach())
    pa = S["attn"]
    if pa is not None and pa.lam > 0:
        mk = pp.CUR["mask"].long()
        z = pa.log_z.exp()[:, mk].permute(1, 0, 2)[..., None]            # (b, h, s, 1)
        pw = (s + pa.b[None, :, None, None]) ** 2 / z
        p = pw if pa.lam >= 1 else (1 - pa.lam) * p + pa.lam * pw
    o = (p @ v).transpose(1, 2)
    return o.reshape(o.shape[0], o.shape[1], -1)


class PolyGELU(nn.Module):
    def __init__(self, R=8.0, deg=31, R0=6.0):
        super().__init__()
        c, self.fit_err = ap.cheb_fit(R, deg)
        self.register_buffer("c", c)
        self.R, self.R0 = R, R0
        self.lam, self.clamp, self.penalty = 0.0, True, False
        self.pen = None; self.oor = [0, 0]; self.umax = 0.0

    def forward(self, u):
        y = F.gelu(u, approximate="tanh")
        t = u / self.R
        self.oor[0] += int((t.abs() > 1).sum()); self.oor[1] += t.numel()
        self.umax = max(self.umax, float(u.detach().abs().max()))
        if self.penalty:
            self.pen = F.relu(u.abs() - self.R0).pow(2).sum() / (u.numel() / u.shape[-1])   # per token
        if self.lam > 0:
            yp = ap.cheb_eval(self.c, t.clamp(-1, 1) if self.clamp else t)
            y = yp if self.lam >= 1 else (1 - self.lam) * y + self.lam * yp
        return y


def noise_hook(mod, inp, out):
    b = S["noise_bits"]
    if b is None:
        return None
    sc = out.detach().abs().amax() * 2.0 ** -b
    return out + torch.randn_like(out) * sc


# ---------------------------------------------------------------- setup
def fwd(m, x):
    """Logits over the real vocabulary (no [MASK] column), autograd-safe."""
    pp.CUR["mask"] = x == pp.MASK
    lg = m(input_ids=x, timesteps=torch.zeros(x.shape[0], device=x.device))
    lg = lg[0] if isinstance(lg, tuple) else lg
    return lg[..., :pp.MASK].float()


def pin0_masks(x, g, full_frac=1 / 8):
    r = torch.rand(len(x), 1, generator=g)
    mk = torch.rand(x.shape, generator=g) < r
    mk[torch.rand(len(x), generator=g) < full_frac] = True
    mk[:, 0] = False
    mk = mk.to(x.device)
    return torch.where(mk, torch.full_like(x, pp.MASK), x), mk


@torch.no_grad()
def calibrate(m, cal, bs, grid=(0.0, 1.0, 2.0, 4.0, 8.0)):
    """On the exact model: norm c = sum(sigma) / sum(sigma^2) per (norm, mask);
    per (layer, head) the shift b in `grid` whose row-normalised (s + b)^2 is
    closest (KL) to softmax, and Z = mean row sum of (s + b)^2 per query mask."""
    blocks = pp.blocks(m)
    norms = [n for b in blocks for n in (b.norm1, b.norm2)] + [m.backbone.output_layer.norm_final]
    H, G = blocks[0].n_heads, len(grid)
    z = lambda *sh: torch.zeros(*sh, dtype=torch.float64, device=pp.DEV)
    s1, s2 = z(len(norms), 2), z(len(norms), 2)
    kl, zs, zn = z(len(blocks), H, G), z(len(blocks), H, G, 2), z(len(blocks), H, 2)
    cur = {"layer": 0}
    hooks = []
    for j, n in enumerate(norms):
        def h(mod, inp, j=j):
            x = inp[0].float()
            sd = x.var(-1, unbiased=False).add(1e-5).sqrt().double()
            mk = pp.CUR["mask"].long()
            for ms in (0, 1):
                w = (mk == ms).double()
                s1[j, ms] += (sd * w).sum(); s2[j, ms] += (sd ** 2 * w).sum()
        hooks.append(n.register_forward_pre_hook(h))
    for i, b in enumerate(blocks):
        hooks.append(b.register_forward_pre_hook(lambda mod, inp, i=i: cur.__setitem__("layer", i)))

    def collect(s, p):
        i, mk = cur["layer"], pp.CUR["mask"].long()
        lp = p.clamp_min(1e-30).log()
        for gi, b0 in enumerate(grid):
            pw = (s + b0) ** 2
            rs = pw.sum(-1, keepdim=True)
            kl[i, :, gi] += (p * (lp - (pw / rs).clamp_min(1e-30).log())).sum(-1).double().sum((0, 2))
            for ms in (0, 1):
                w = (mk == ms)[:, None, :].double()
                zs[i, :, gi, ms] += (rs[..., 0].double() * w).sum((0, 2))
        for ms in (0, 1):
            zn[i, :, ms] += (mk == ms).double().sum()
    S["collect"] = collect
    g = torch.Generator().manual_seed(5)
    for st in range(0, len(cal), bs):
        fwd(m, pin0_masks(cal[st:st + bs].to(pp.DEV), g)[0])
    S["collect"] = None
    for h in hooks:
        h.remove()
    c = s1 / s2.clamp_min(1e-30)
    c = torch.where(s2 > 0, c, (s1.sum(-1) / s2.sum(-1))[:, None].expand_as(c))
    best = kl.argmin(-1)                                                   # (layers, h)
    b = torch.tensor(grid, dtype=torch.float64, device=pp.DEV)[best]
    zsel = zs.gather(2, best[..., None, None].expand(-1, -1, 1, 2))[:, :, 0] / zn.clamp_min(1)
    return {"norm_c": c, "attn_b": b, "attn_z": zsel.clamp_min(1e-6),
            "attn_kl_by_shift": (kl / zn.sum(-1, keepdim=True).clamp_min(1)).mean(1).tolist()}


def convert(m, cal):
    """Install the E5 ops (all at lam = 0, i.e. still exact) into the student."""
    blocks = pp.blocks(m)
    j = 0
    for i, b in enumerate(blocks):
        b.dropout = 0.0
        b.norm1 = FixedNorm(b.norm1, cal["norm_c"][j]); j += 1
        b.norm2 = FixedNorm(b.norm2, cal["norm_c"][j]); j += 1
        b.pow_attn = PowAttn(cal["attn_b"][i], cal["attn_z"][i])
        b.mlp[1] = PolyGELU()
        b.register_forward_pre_hook(lambda mod, inp: S.__setitem__("attn", mod.pow_attn))
        b.register_forward_hook(lambda mod, inp, out: S.__setitem__("attn", None))
        b.register_forward_hook(noise_hook)
    ol = m.backbone.output_layer
    ol.norm_final = FixedNorm(ol.norm_final, cal["norm_c"][j])
    return m


def patch_attention(m):
    """Route every MDLM attention call through `attention` (teacher and student
    share the remote module; the teacher never sets S['attn'], so it stays exact)."""
    mod = sys.modules[type(pp.blocks(m)[0]).__module__]
    mod.regular_attention_multi_headed = attention


def ops(m, kind):
    blocks = pp.blocks(m)
    if kind == "norm":
        return [n for b in blocks for n in (b.norm1, b.norm2)] + [m.backbone.output_layer.norm_final]
    return [b.pow_attn for b in blocks] if kind == "attn" else [b.mlp[1] for b in blocks]


def set_stage(m, stage, lam, train):
    """Ops of stages before `stage` at lam = 1, this stage's at `lam`."""
    k = STAGES.index(stage)
    for kind, si in (("norm", 0), ("attn", 1), ("gelu", 2)):
        for o in ops(m, kind):
            o.lam = 1.0 if k > si else lam if k == si else 0.0
    for g in ops(m, "gelu"):
        g.clamp, g.penalty = train, train and k >= 2
    S["noise_bits"] = NOISE_BITS if (k == 3 and train) else None


# ---------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(student, teacher, judge, evalset, a, noise_bits=(None,), gen=True):
    out = {}
    for g_ in ops(student, "gelu"):
        g_.oor = [0, 0]; g_.umax = 0.0
    resid = {"max": 0.0}
    hk = pp.blocks(student)[-1].register_forward_hook(
        lambda mod, inp, o: resid.__setitem__("max", max(resid["max"], float(o.abs().max()))))
    for nb in noise_bits:
        S["noise_bits"] = nb
        tag = "" if nb is None else f"_b{nb:g}"
        for ratio in (0.15, 0.5, 0.9):
            g = torch.Generator().manual_seed(7)
            tot = kl = 0.0; cnt = 0
            for st in range(0, len(evalset), a.bs):
                x = evalset[st:st + a.bs].to(pp.DEV)
                m_ = (torch.rand(x.shape, generator=g) < ratio).to(pp.DEV); m_[:, 0] = False
                xm = torch.where(m_, torch.full_like(x, pp.MASK), x)
                lt = fwd(teacher, xm).log_softmax(-1)[m_]
                ls = torch.nan_to_num(fwd(student, xm).log_softmax(-1)[m_], nan=-1e4, neginf=-1e4)
                tot += float(-ls.gather(-1, x[m_][:, None]).sum()); cnt += int(m_.sum())
                kl += float((lt.exp() * (lt.clamp_min(-1e4) - ls)).sum())
            out[f"nll{tag}_mask{ratio}"] = tot / max(cnt, 1); out[f"kl{tag}_mask{ratio}"] = kl / max(cnt, 1)
        if gen and a.gen_n:
            pp.PIN0 = True
            exact_fwd, pp.forward_logits = pp.forward_logits, ap.forward_logits_checked
            ap._exact_fwd = exact_fwd
            try:
                xs = pp.sample(student, a.gen_n, a.L, a.gen_steps, seed=11)
                out[f"gen_ppl{tag}"] = pp.gen_ppl(judge, xs)
                out[f"gen_entropy{tag}"] = pp.unigram_entropy(xs)
            except FloatingPointError as e:
                out[f"gen_error{tag}"] = str(e)[:200]
            pp.forward_logits = exact_fwd
    S["noise_bits"] = None
    hk.remove()
    o, n = sum(g_.oor[0] for g_ in ops(student, "gelu")), sum(g_.oor[1] for g_ in ops(student, "gelu"))
    out["gelu_frac_outside"] = o / max(n, 1)
    out["gelu_in_max"] = max(g_.umax for g_ in ops(student, "gelu"))
    out["resid_max"] = resid["max"]
    return out


# ---------------------------------------------------------------- train
def train_stage(student, teacher, ids, stage, a, log):
    params = [p for n, p in student.named_parameters() if p.requires_grad]
    consts = [p for n, p in student.named_parameters() if n.endswith(("log_c", ".b", "log_z"))]
    ids_c = {id(p) for p in consts}
    opt = torch.optim.AdamW([{"params": [p for p in params if id(p) not in ids_c], "lr": a.lr},
                             {"params": consts, "lr": a.lr_const}], weight_decay=0.0)
    steps, ramp = a.steps[STAGES.index(stage)], a.ramp[STAGES.index(stage)]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup))
    g = torch.Generator().manual_seed(100 + STAGES.index(stage))
    t0 = time.time()
    for st in range(steps):
        set_stage(student, stage, min(1.0, (st + 1) / ramp), train=True)
        opt.zero_grad(set_to_none=True)
        stats = {"kl": 0.0, "pen": 0.0}
        for _ in range(a.accum):
            idx = torch.randint(0, len(ids) - a.L, (a.bs,), generator=g)
            x = torch.stack([ids[i:i + a.L] for i in idx]).to(pp.DEV)
            xm, mk = pin0_masks(x, g)
            S["noise_bits"], nb = None, S["noise_bits"]
            with torch.no_grad():
                lt = fwd(teacher, xm).log_softmax(-1)[mk]
            S["noise_bits"] = nb
            ls = fwd(student, xm).log_softmax(-1)[mk]
            loss = (lt.exp() * (lt - ls)).sum(-1).mean()
            stats["kl"] += float(loss) / a.accum
            if stage in ("C", "D"):
                pen = sum(g_.pen for g_ in ops(student, "gelu"))
                loss = loss + a.gelu_pen * pen
                stats["pen"] += float(pen) / a.accum
            (loss / a.accum).backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(gn):
            log({"stage": stage, "step": st, "skip": "non-finite grad"}); continue
        opt.step(); sched.step()
        if st % a.log_every == 0 or st == steps - 1:
            log({"stage": stage, "step": st, "lam": min(1.0, (st + 1) / ramp), "kl": stats["kl"],
                 "gelu_pen": stats["pen"], "grad_norm": float(gn), "s_per_step": (time.time() - t0) / (st + 1)})


NOISE_BITS = 11


def main():
    global NOISE_BITS
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--L", type=int, default=256)
    ap_.add_argument("--bs", type=int, default=16)
    ap_.add_argument("--accum", type=int, default=2)
    ap_.add_argument("--steps", type=int, nargs=4, default=[3000, 3000, 2000, 1000], help="per stage A B C D")
    ap_.add_argument("--ramp", type=int, nargs=4, default=[1000, 1000, 500, 1])
    ap_.add_argument("--warmup", type=int, default=100)
    ap_.add_argument("--lr", type=float, default=1e-4)
    ap_.add_argument("--lr-const", type=float, default=1e-3)
    ap_.add_argument("--gelu-pen", type=float, default=1e-3)
    ap_.add_argument("--noise-bits", type=float, default=11)
    ap_.add_argument("--train-rows", type=int, default=2_000_000)
    ap_.add_argument("--n-cal", type=int, default=128)
    ap_.add_argument("--n-eval", type=int, default=256)
    ap_.add_argument("--gen-n", type=int, default=64)
    ap_.add_argument("--gen-steps", type=int, default=128)
    ap_.add_argument("--log-every", type=int, default=50)
    ap_.add_argument("--stages", nargs="+", default=list(STAGES))
    ap_.add_argument("--abort-nll", type=float, default=20.0,
                     help="stop after a stage whose NLL@50%% exceeds this (exact: 3.84)")
    ap_.add_argument("--ckpt-dir", required=True)
    ap_.add_argument("--out", required=True)
    a = ap_.parse_args()
    NOISE_BITS = a.noise_bits
    torch.manual_seed(0)
    os.makedirs(a.ckpt_dir, exist_ok=True)

    teacher = pp.load_model(pp.DEV)
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = pp.load_model(pp.DEV)
    from transformers import AutoModelForCausalLM
    judge = AutoModelForCausalLM.from_pretrained(pp.snapshot("openai-community/gpt2-large")).to(pp.DEV).eval()
    for p in judge.parameters():
        p.requires_grad_(False)
    evalset = pp.load_chunks(a.L, a.n_eval)
    ids = spec_alpha.train_tokens(a.train_rows)
    cal = torch.stack([ids[i:i + a.L] for i in torch.randint(0, len(ids) - a.L, (a.n_cal,),
                                                              generator=torch.Generator().manual_seed(3))])
    out = {"args": vars(a), "train_tokens": len(ids), "stages": {}}
    if os.path.exists(a.out):
        out = json.load(open(a.out)) | {"args": vars(a)}

    def log(r):
        print("[e1] " + json.dumps(r), flush=True)
        out.setdefault("log", []).append(r)

    patch_attention(student)
    calib = calibrate(student, cal, a.bs)
    out["calibration"] = {"attn_b": calib["attn_b"].tolist(), "attn_kl_by_shift": calib["attn_kl_by_shift"],
                          "norm_c": calib["norm_c"].tolist()}
    convert(student, calib)
    if "exact" not in out["stages"]:
        set_stage(student, "A", 0.0, train=False)
        r = evaluate(student, teacher, judge, evalset, a)
        out["stages"]["exact"] = r; log({"eval": "exact", **r})
        json.dump(out, open(a.out, "w"), indent=1)

    done = [s for s in STAGES if os.path.exists(os.path.join(a.ckpt_dir, f"stage{s}.pt"))]
    if done:
        student.load_state_dict(torch.load(os.path.join(a.ckpt_dir, f"stage{done[-1]}.pt"), map_location=pp.DEV))
        log({"resumed_after": done[-1]})
    student.eval()                                   # dropout is 0 anyway; no train-mode paths
    for stage in a.stages:
        if stage in done:
            continue
        t0 = time.time()
        train_stage(student, teacher, ids, stage, a, log)
        set_stage(student, stage, 1.0, train=False)
        r = evaluate(student, teacher, judge, evalset, a, noise_bits=(None, 11, 16) if stage >= "C" else (None,))
        r["train_seconds"] = time.time() - t0
        out["stages"][stage] = r; log({"eval": stage, **r})
        torch.save(student.state_dict(), os.path.join(a.ckpt_dir, f"stage{stage}.pt"))
        json.dump(out, open(a.out, "w"), indent=1)
        if not r["nll_mask0.5"] < a.abort_nll:       # NaN counts as failed
            log({"abort": stage, "nll_mask0.5": r["nll_mask0.5"], "limit": a.abort_nll})
            json.dump(out, open(a.out, "w"), indent=1)
            break


if __name__ == "__main__":
    main()
