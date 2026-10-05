"""Step 4 -- zero-shot polynomialization of every RMSNorm rsqrt at K=8.

Configurations: {per-layer, per-(layer,t)} x {degree 3, 5}, plus an unmodified
reference. Calibration samples are a disjoint seed from evaluation samples, so
no polynomial is fitted on the activations it is then scored on.

All runs share one t-grid and one set of replayed SDE noise draws (CRN), so a
token that differs between the reference and a polynomial run differs *because
of the polynomial* and for no other reason.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from transformers import AutoTokenizer, AutoModelForCausalLM
from fhedlm.runner import load_elf, run_config, score, hf_id
from fhedlm import metrics, polynorm

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--n-samples", type=int, default=16)
ap.add_argument("--n-calib", type=int, default=16)
ap.add_argument("--eval-seed", type=int, default=0)
ap.add_argument("--calib-seed", type=int, default=1234)     # disjoint from eval
ap.add_argument("--degrees", type=int, nargs="+", default=[3, 5])
ap.add_argument("--out", default="results/step4_poly.json")
a = ap.parse_args()

torch.set_grad_enabled(False)
if os.environ.get("NSLOTS"):
    torch.set_num_threads(int(os.environ["NSLOTS"]))
print(f"[s4] threads={torch.get_num_threads()} K={a.k} n={a.n_samples}", flush=True)

model, cfg, text_dim = load_elf()
tok = AutoTokenizer.from_pretrained(hf_id(cfg.tokenizer_name or cfg.encoder_model_name))
jt = AutoTokenizer.from_pretrained(hf_id("gpt2-large"))
jm = AutoModelForCausalLM.from_pretrained(hf_id("gpt2-large")).eval()

norms = polynorm.install(model)
print(f"[s4] installed {len(norms)} PolyRMSNorm modules", flush=True)
out = {"k": a.k, "n_samples": a.n_samples, "n_calib": a.n_calib,
       "eval_seed": a.eval_seed, "calib_seed": a.calib_seed,
       "n_rmsnorm_per_forward": len(norms), "configs": {}}

# ---- calibration: observe the rsqrt input range, on samples we will NOT score
polynorm.set_mode(norms, "collect")
t0 = time.time()
run_config(model, cfg, text_dim, a.k, a.n_calib, a.calib_seed, tok,
           step_hook=polynorm.set_step)
print(f"[s4] calibration pass {time.time()-t0:.0f}s", flush=True)
out["observed_ranges"] = {
    n: {str(s): [float(lo), float(hi)] for s, (lo, hi) in m.ranges.items()}
    for n, m in norms.items()}
spans = [(n, min(l for l, _ in m.ranges.values()), max(h for _, h in m.ranges.values()))
         for n, m in norms.items() if m.ranges]
dec = [(n, np.log10(h / max(l, 1e-30))) for n, l, h in spans]
dec.sort(key=lambda x: -x[1])
out["widest_ranges_decades"] = [{"layer": n, "decades": float(d)} for n, d in dec[:8]]
print("[s4] widest observed rsqrt-input ranges (decades):", flush=True)
for n, d in dec[:5]:
    print(f"      {n}: {d:.2f}", flush=True)

# ---- reference: unmodified, exact rsqrt, same schedule as every poly run
polynorm.set_mode(norms, "exact")
ref = run_config(model, cfg, text_dim, a.k, a.n_samples, a.eval_seed, tok,
                 keep_states=True, step_hook=polynorm.set_step)
ref_m = score(ref["ids"], ref["texts"], tok, jm, jt)
ref_m["gen_seconds"] = ref["gen_seconds"]
ref_m["texts"] = ref["texts"]
out["configs"]["reference"] = ref_m
print(f"[s4] reference: gen-PPL {ref_m['gen_ppl']:.2f} | entropy {ref_m['entropy_elf']:.3f} "
      f"| rep4 {ref_m['repetition_4gram']:.3f}", flush=True)

# ---- polynomial configurations
for per_step in (False, True):
    for deg in a.degrees:
        tag = f"{'per-layer-t' if per_step else 'per-layer'}-deg{deg}"
        print(f"[s4] === {tag} ===", flush=True)
        fit_rep = polynorm.fit_all(norms, degree=deg, per_step=per_step)
        polynorm.set_mode(norms, "poly")
        try:
            r = run_config(model, cfg, text_dim, a.k, a.n_samples, a.eval_seed, tok,
                           keep_states=True, step_hook=polynorm.set_step)
            m = score(r["ids"], r["texts"], tok, jm, jt)
            m["token_agreement_vs_ref"] = metrics.token_agreement(
                r["ids"], ref["ids"], pad_id=tok.pad_token_id)
            div = metrics.trajectory_divergence(ref["traj"].z_states, r["traj"].z_states)
            m.update(div)
            c = div["rel_error_by_step"]
            # The compounding question: does deterministic polynomial bias grow
            # along the trajectory, stay flat, or get contracted away?
            m["rel_error_first"], m["rel_error_last"] = c[0], c[-1]
            m["rel_error_growth_ratio"] = c[-1] / max(c[0], 1e-30)
            # Per-(layer, t) out-of-range behaviour. Nothing is clipped, so an
            # input outside the fitted interval is extrapolated; these are the
            # points where a polynomial fails silently under CKKS.
            per_lt, tot_seen, tot_out = {}, 0, 0
            worst_all, worst_out = 0.0, 0.0
            for nm, nn in norms.items():
                for st, d in nn.stats.items():
                    if not d["n_seen"]:
                        continue
                    frac = d["n_out"] / d["n_seen"]
                    per_lt[f"{nm}@t{st}"] = {
                        "frac_out_of_range": frac,
                        "max_rel_err_all": d["max_rel_all"],
                        "max_rel_err_out_of_range": d["max_rel_out"],
                    }
                    tot_seen += d["n_seen"]; tot_out += d["n_out"]
                    worst_all = max(worst_all, d["max_rel_all"])
                    worst_out = max(worst_out, d["max_rel_out"])
            m["max_rel_rsqrt_err_heldout"] = worst_all
            m["max_rel_rsqrt_err_out_of_range"] = worst_out
            m["frac_activations_out_of_fit_range"] = tot_out / max(tot_seen, 1)
            m["per_layer_t"] = per_lt
            m["worst_out_of_range_layers"] = sorted(
                ({"key": k, **v} for k, v in per_lt.items() if v["frac_out_of_range"] > 0),
                key=lambda d: -d["frac_out_of_range"])[:10]
            m["fit_report_worst"] = max(fit_rep.values(), key=lambda d: d["max_rel_error"])
            m["texts"] = r["texts"]
            print(f"[s4] {tag}: agree {m['token_agreement_vs_ref']:.3f} | "
                  f"gen-PPL {m['gen_ppl']:.2f} | entropy {m['entropy_elf']:.3f} | "
                  f"rep4 {m['repetition_4gram']:.3f} | rsqrt max-rel-err "
                  f"{m['max_rel_rsqrt_err_heldout']:.3e} | rel-err step0->last "
                  f"{m['rel_error_first']:.3e}->{m['rel_error_last']:.3e}", flush=True)
        except Exception as e:
            m = {"error": f"{type(e).__name__}: {e}"}
            print(f"[s4] {tag}: FAILED {m['error']}", flush=True)
        out["configs"][tag] = m
        polynorm.set_mode(norms, "exact")
        json.dump(out, open(a.out, "w"), indent=2)   # checkpoint after each config

# ---- Newton refinement on the saved activations (offline, no generation)
# A wide range is not automatically fatal: Newton converges quadratically, so a
# cheap degree-3 start plus a few iterations may reach accuracy a direct
# low-degree fit cannot, at a known extra 3 levels per iteration.
print("[s4] Newton refinement analysis ...", flush=True)
try:
    out["newton"] = polynorm.newton_analysis(norms, degree=3, iters=3, target=1e-3)
    w = out["newton"]["worst_max_rel_err_by_iter"]
    lv = out["newton"]["levels_by_iter"]
    print("[s4] worst max-rel-err by Newton iteration (deg-3 per-(layer,t) start):", flush=True)
    for i, (e, l) in enumerate(zip(w, lv)):
        print(f"      iter {i}: {e:.3e}   ({l} levels/norm)", flush=True)
    ch = out["newton"]["cheapest_meeting_target"]
    print(f"[s4] cheapest reaching 1e-3: {ch}", flush=True)
except Exception as e:
    out["newton"] = {"error": f"{type(e).__name__}: {e}"}
    print(f"[s4] Newton analysis FAILED: {out['newton']['error']}", flush=True)

out["decode_pass"] = {
    "uses_timestep_index": a.k,
    "note": ("the decode forward gets its own bucket t=K, so its polynomial is "
             "calibrated on the decode pass's own activations, not pooled with "
             "the last sampling step"),
}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=2)
print(f"[s4] wrote {a.out}", flush=True)
