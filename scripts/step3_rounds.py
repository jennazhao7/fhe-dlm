"""Step 3 -- quality vs number of diffusion rounds. K in {8,4} (K=32 from step 1)."""
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from transformers import AutoTokenizer, AutoModelForCausalLM
from fhedlm.runner import load_elf, run_config, score, hf_id

ap = argparse.ArgumentParser()
ap.add_argument("--ks", type=int, nargs="+", default=[8, 4])
ap.add_argument("--n-samples", type=int, default=16)
ap.add_argument("--grid-seeds", type=int, nargs="+", default=[0, 1, 2, 3])
ap.add_argument("--out", default="results/step3_rounds.json")
a = ap.parse_args()

torch.set_grad_enabled(False)
if os.environ.get("NSLOTS"):
    torch.set_num_threads(int(os.environ["NSLOTS"]))
print(f"[s3] threads={torch.get_num_threads()} fp32 cpu", flush=True)

model, cfg, text_dim = load_elf()
tok = AutoTokenizer.from_pretrained(hf_id(cfg.tokenizer_name or cfg.encoder_model_name))
jt = AutoTokenizer.from_pretrained(hf_id("gpt2-large"))
print("[s3] model + tokenizers ready", flush=True)
jm = AutoModelForCausalLM.from_pretrained(hf_id("gpt2-large")).eval()

out = {"n_samples": a.n_samples, "grid_seeds": a.grid_seeds, "runs": {}}
# The logit-normal t-grid is redrawn per seed. At small K the grid dominates:
# two K=8 runs of the identical config differed by 69 vs 95 gen-PPL before the
# grid was seeded. A single draw per K would not distinguish "K=4 is better than
# K=8" from "this K=4 grid happened to be luckier".
import statistics
for K in a.ks:
    per_seed = []
    for gs in a.grid_seeds:
        r = run_config(model, cfg, text_dim, K, a.n_samples, gs, tok)
        m = score(r["ids"], r["texts"], tok, jm, jt)
        m["gen_seconds"] = r["gen_seconds"]
        m["grid_seed"] = gs
        m["texts"] = r["texts"]
        per_seed.append(m)
        print(f"[s3] K={K} seed={gs}: gen-PPL {m['gen_ppl']:.2f} | "
              f"entropy {m['entropy_elf']:.3f} | rep4 {m['repetition_4gram']:.3f} | "
              f"{r['gen_seconds']:.0f}s", flush=True)
    agg = {}
    for key in ("gen_ppl", "entropy_elf", "repetition_4gram"):
        vals = [m[key] for m in per_seed]
        agg[key + "_mean"] = statistics.mean(vals)
        agg[key + "_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        agg[key + "_values"] = vals
    out["runs"][str(K)] = {"per_seed": per_seed, "aggregate": agg}
    print(f"[s3] K={K} AGG: gen-PPL {agg['gen_ppl_mean']:.2f} +/- {agg['gen_ppl_std']:.2f} | "
          f"entropy {agg['entropy_elf_mean']:.3f} +/- {agg['entropy_elf_std']:.3f} | "
          f"rep4 {agg['repetition_4gram_mean']:.3f}", flush=True)
    json.dump(out, open(a.out, "w"), indent=2)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=2)
print(f"[s3] wrote {a.out}", flush=True)
