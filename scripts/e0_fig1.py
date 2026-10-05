"""Fig. 1 (E0): server seconds per committed token vs tokens committed per
round k, from results/e0_cost_model.json. DLM block sizes and speculative AR
are categorical series (palette slots 1-4, fixed order); the AR decode step is
a gray reference line. Every series is direct-labelled with markers because
slots 3-4 sit below 3:1 contrast on the light surface.
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# (name, color, vertical label nudge in points -- B=8 and spec. AR end together)
SERIES = [("DLM B=4", "#2a78d6", 0), ("DLM B=8", "#eb6834", 6), ("DLM B=16", "#1baf7a", 0),
          ("SpecAR a=0.8", "#eda100", -7)]

src = sys.argv[1] if len(sys.argv) > 1 else "results/e0_cost_model.json"
dst = sys.argv[2] if len(sys.argv) > 2 else "results/fig1_cost_per_token"
res = json.load(open(src))
rows = res["series"]
ar = next(r for r in rows if r["family"] == "ar")["s_per_token_server"]
C = res["unit_costs_s"]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(5.2, 3.4), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)

ax.axhline(ar / 60, color=INK2, linestyle=(0, (4, 3)), linewidth=1.5)
ax.annotate("AR decode step", (16, ar / 60), xytext=(0, 5), textcoords="offset points",
            color=INK2, fontsize=8, va="bottom", ha="center")
if res["assumptions"].get("packed"):     # B=8, spec. AR and B=16 meet at k=8
    SERIES = [("DLM B=4", "#2a78d6", 0), ("DLM B=8", "#eb6834", 9), ("DLM B=16", "#1baf7a", -2),
              ("SpecAR a=0.8", "#eda100", 0)]
for name, color, nudge in SERIES:
    pts = sorted((r["k"], r["s_per_token_server"] / 60) for r in rows if r["name"] == name)
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.2, solid_capstyle="round")
    ax.annotate(name.replace("SpecAR a=0.8", "spec. AR (α=0.8)"), (xs[-1], ys[-1]),
                xytext=(6, nudge), textcoords="offset points", color=INK, fontsize=8,
                va="center")

ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels(["1", "2", "4", "8", "16"])
ax.set_xlim(0.85, 30)
ax.set_ylim(0, None)
ax.set_xlabel("tokens committed per round, k")
ax.set_ylabel("server minutes per committed token")
PACKED = res["assumptions"].get("packed", False)
ax.set_title(f"Encrypted cost per committed token (CPU OpenFHE, N=2^{C['ring'].bit_length() - 1}"
             + (", packed)" if PACKED else ")"),
             fontsize=9, color=INK, loc="left")
fig.text(0.01, 0.01, f"Model from measured op costs; L={res['assumptions']['L']}, "
         f"{res['assumptions']['layers']} layers, d padded to {res['assumptions']['d_pad']}, "
         f"GELU deg {res['assumptions']['gelu_degree']};\nmatvec {res['assumptions']['matvec']}. "
         "Spec. AR: k = drafts verified per pass. Network adds <1% on CPU.",
         fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0.08, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(f"{dst}.{ext}", dpi=200, facecolor=SURFACE)

with open(f"{dst}.md", "w") as f:                    # table view of the same data
    f.write("| series | k | passes/token | server min/token | WAN min/token |\n|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['name']} | {r['k']} | {r['passes_per_token']:.2f} | "
                f"{r['s_per_token_server'] / 60:.1f} | {r['s_per_token_WAN'] / 60:.1f} |\n")
print("wrote", dst + ".{png,pdf,md}")
