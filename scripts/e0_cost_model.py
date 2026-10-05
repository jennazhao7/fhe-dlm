"""E0 analytical cost model: seconds per committed token for the encrypted
DLM block pass vs the AR decode step (and speculative AR verify-k), built from
operation counts x the per-op costs measured by e0_microbench.py.

Layout (pad d_model 768 -> D=1024): one ciphertext holds T = slots/D tokens,
interleaved, slot = dim*T + token. Rotating by a multiple of T rotates every
token's dimension vector cyclically, so the BSGS diagonal method runs natively
(no masks) and a <= T-token block costs the same as one token: that slot
amortization is exactly what C1 tests.

Per layer, per T-token query ciphertext (B queries, context l, c = ceil(l/T)
key/value ciphertexts in the encrypted KV cache):
  QKV        matvec 1->3 blocks           depth 1
  scores     per key ct, B token shifts: 2 rot + 2 ptm (wrap mask) + 1 ctm
             + 6 rot (sum the 64 dims of each head)          depth 2 (mask, mult)
  power p=2  1 ctm per score ct                              depth 1
  /constant  1 ptm per score ct (public normaliser, no inverse)  depth 1
  replicate  6 rot + 1 ptm per score ct (head slot -> 64 dims)   depth 1
  x V        2 rot + 2 ptm (V wrap mask) + 1 ctm per score ct    depth 1
  gather     2*log2(T/B) rot (replicas of the B queries) + same to replicate Q
  out proj   matvec 1->1                                     depth 1
  2 x norm   fixed-constant: mean over D (10 rot + 1 ptm) then 1 ptm  depth 1 each
  FFN        matvec 1->3, GELU on 3 cts (Chebyshev, measured), matvec 3->1
Plus once per pass: LM head matvec 1->ceil(V/D) on the final hidden ct (logits
for all T token slots; compacted to the committed positions before download:
1 ptm + 1 rot per output ct), confidence head (1 ptm + 10 rot).
matvec(i->o blocks): 32*i baby + 32*o giant rotations, 1024*i*o ptm and adds.

Bootstraps: ceil(layers*depth_per_layer / usable_levels) on the 1 residual ct
per T tokens (placed where the stream is a single ct). Op costs are the
measured cost interpolated at the middle of the post-bootstrap level window;
GELU uses its measured fresh-level time scaled by the ctm cost ratio.

Generation of L tokens is simulated pass by pass with the growing context:
  DLM  blocks of B, B/k rounds each (+1 clean pass per block to write the
       block's KV cache, BD3-LM style; KV_PASS)
  AR   one B=1 pass per token
  SpecAR  verify pass over k drafts, expected (1-a^(k+1))/(1-a) tokens committed
Wall-clock per round adds RTT + upload/download over the link + client
encrypt/decrypt (measured on the same 20-thread box; a phone is slower).

Calibration (MATVEC_JSON, from e0_matvec_bench.py): when present, every matvec
is charged its measured cost per diagonal -- compute plus on-the-fly diagonal
encoding -- at the level nearest the op level, instead of the op sum. The op
sum was within 1.0-1.6x on compute but omits encoding, which costs 1-3x the
compute on CPU (pre-encoding all ~174k diagonals at 2^17 would need TBs).

Every structural choice above is an assumption of this first-cut model, not a
measurement; the kernels it names are what E0's next step benchmarks directly.
"""
import json, math, os, sys

# --- model shape (MDLM / BD3-LM 110M, GPT-2 tokenizer) ---
LAYERS, D_PAD, HEAD_DIM, VOCAB, L_GEN = 12, 1024, 64, 50257, 256
GELU_DEGREE = 31                # E1 decides; 63 adds a level for ~1e-12 error
KV_PASS = True
NETS = {"LAN": (0.5e-3, 1e9), "WAN": (40e-3, 100e6), "mobile": (80e-3, 20e6)}
SPEC_ACCEPT = [0.6, 0.8]
MATVEC_JSON = "results/e0_matvec.json"   # set to None for the pure op-sum model
RING = 1 << 17                  # 2^16 is infeasible: 3 usable levels < GELU depth
BUDGET = [4, 4]


def interp(rows, key, level):
    rows = sorted(rows, key=lambda r: r["level"])
    for a, b in zip(rows, rows[1:]):
        if a["level"] <= level <= b["level"]:
            w = (level - a["level"]) / (b["level"] - a["level"])
            return a[key] * (1 - w) + b[key] * w
    return rows[0][key] if level < rows[0]["level"] else rows[-1][key]


def matvec_per_diag(level):
    """Measured seconds per (input block x output block x diagonal), compute +
    encode, at the benchmarked level nearest `level`; None if not measured."""
    if not MATVEC_JSON or not os.path.exists(MATVEC_JSON):
        return None
    runs = json.load(open(MATVEC_JSON))["runs"]
    lvl = min({r["level"] for r in runs}, key=lambda l: abs(l - level))
    per = [(r["compute_only_s"] + r["encode_s"]) / (1024 * r["din"] * r["dout"])
           for r in runs if r["level"] == lvl]
    return sum(per) / len(per)


def load_costs(bench_path):
    bench = json.load(open(bench_path))
    ring = next(r for r in bench["rings"] if r.get("ring_dim") == RING)
    boot = next(b for b in bench["bootstrap"] if b["ring_dim"] == RING
                and b["level_budget"] == BUDGET and b.get("slots") == RING // 2
                and "bootstrap_s" in b)
    depth, usable = ring["depth"], boot["levels_left_after"] - 1   # see openfhe_levels
    mid = depth - usable / 2
    lv = ring["by_level"]
    gelu = next(g for g in ring["gelu"] if g["degree"] == GELU_DEGREE)
    ctm_ratio = interp(lv, "mult_ct_ct_relin_s", mid) / interp(lv, "mult_ct_ct_relin_s", 0)
    return {
        "ring": RING, "slots": ring["slots"], "T": ring["slots"] // D_PAD,
        "depth": depth, "usable": usable, "op_level": mid,
        "rot": interp(lv, "rotate_s", mid), "ptm": interp(lv, "mult_ct_pt_s", mid),
        "ctm": interp(lv, "mult_ct_ct_relin_s", mid), "add": interp(lv, "add_s", mid),
        "gelu": gelu["seconds"] * ctm_ratio, "gelu_depth": gelu["depth_used"],
        "boot": boot["bootstrap_s"], "boot_precision_bits": boot["precision_bits"],
        "encrypt": ring["encrypt_s"], "decrypt": ring["decrypt_s"],
        "matvec_per_diag": matvec_per_diag(mid),
    }


MV = {"per_diag": None}       # set from load_costs; switches matvec to measured


def matvec(i, o):
    if MV["per_diag"] is not None:
        return {"mv_diag": 1024 * i * o}
    return {"rot": 32 * i + 32 * o, "ptm": 1024 * i * o, "add": 1024 * i * o}


def add(acc, ops, n=1):
    for k, v in ops.items():
        acc[k] = acc.get(k, 0) + v * n


def pass_ops(B, ctx, C):
    """Op counts for one server pass over B query tokens with context ctx."""
    T = C["T"]
    nq, c = math.ceil(B / T), math.ceil(ctx / T)
    Bq = min(B, T)                                   # queries per query ct
    rep = 2 * math.ceil(math.log2(T / Bq)) if Bq < T else 0
    hs = int(math.log2(HEAD_DIM))
    layer = {}
    add(layer, matvec(1, 3))                                       # QKV
    add(layer, {"rot": rep})                                       # replicate Q
    add(layer, {"rot": 2 + hs, "ptm": 2, "ctm": 1}, Bq * c)        # scores
    add(layer, {"ctm": 1, "ptm": 1}, Bq * c)                       # power, /const
    add(layer, {"rot": hs, "ptm": 1}, Bq * c)                      # replicate score
    add(layer, {"rot": 2, "ptm": 2, "ctm": 1, "add": 1}, Bq * c)   # x V
    add(layer, {"rot": rep})                                       # gather replicas
    add(layer, matvec(1, 1))                                       # out proj
    add(layer, {"rot": 10, "ptm": 2}, 2)                           # 2 norms
    add(layer, matvec(1, 3)); add(layer, {"gelu": 3}); add(layer, matvec(3, 1))
    depth_layer = 1 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + C["gelu_depth"] + 1 + 1
    ops = {}
    add(ops, layer, LAYERS * nq)
    vb = math.ceil(VOCAB / D_PAD)
    add(ops, matvec(1, vb), nq); add(ops, {"ptm": vb, "rot": vb}, nq)   # LM head + compact
    add(ops, {"ptm": 1, "rot": 10}, nq)                                  # confidence head
    add(ops, {"boot": math.ceil(LAYERS * depth_layer / C["usable"])}, nq)
    return ops, depth_layer


def pass_cost(B, ctx, C, committed):
    ops, depth_layer = pass_ops(B, ctx, C)
    by = {k: ops.get(k, 0) * C[k] for k in ("rot", "ptm", "ctm", "add", "gelu", "boot")}
    if ops.get("mv_diag"):
        by["matvec"] = ops["mv_diag"] * C["matvec_per_diag"]
    N, towers_up = C["ring"], depth_layer + 1
    nq = math.ceil(B / C["T"])
    vb = math.ceil(VOCAB / D_PAD)
    up = nq * N * towers_up * 8                       # seed-compressed fresh ct
    # logits for the committed positions (compacted) + confidence ct, both
    # mod-switched to one tower before download.
    down_cts = math.ceil(vb * max(committed, 1) / C["T"]) + 1
    down = down_cts * 2 * N * 8
    client = C["encrypt"] * nq + C["decrypt"] * down_cts
    return {"server_s": sum(by.values()), "by_op_s": by, "ops": ops,
            "depth_per_layer": depth_layer, "bytes_up": up, "bytes_down": down,
            "client_s": client}


def wall(pc, net):
    rtt, bw = NETS[net]
    return pc["server_s"] + pc["client_s"] + rtt + 8 * (pc["bytes_up"] + pc["bytes_down"]) / bw


def per_token(passes, net=None):
    t = sum((p["server_s"] if net is None else wall(p, net)) for p, _ in passes)
    toks = sum(n for _, n in passes)
    return t / toks, len(passes) / toks


def gen_dlm(B, k, C):
    passes = []
    for start in range(0, L_GEN, B):
        ctx = start + B
        for _ in range(B // k):
            passes.append((pass_cost(B, ctx, C, k), k))
        if KV_PASS:
            passes.append((pass_cost(B, ctx, C, 0), 0))
    return passes


def gen_ar(C):
    return [(pass_cost(1, p + 1, C, 1), 1) for p in range(L_GEN)]


def gen_spec(k, a, C):
    exp = (1 - a ** (k + 1)) / (1 - a)
    n = math.ceil(L_GEN / exp)
    return [(pass_cost(k + 1, int(i * exp) + k + 1, C, round(exp)), exp) for i in range(n)]


def main():
    bench = sys.argv[1] if len(sys.argv) > 1 else "results/e0_microbench.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "results/e0_cost_model.json"
    C = load_costs(bench)
    MV["per_diag"] = C["matvec_per_diag"]
    res = {"assumptions": {"layers": LAYERS, "d_pad": D_PAD, "vocab": VOCAB, "L": L_GEN,
                           "gelu_degree": GELU_DEGREE, "kv_pass": KV_PASS, "ring": RING,
                           "level_budget": BUDGET, "nets": NETS,
                           "matvec": "measured (compute+encode)" if MV["per_diag"] else "op sum"},
           "unit_costs_s": C, "series": []}
    nets = [None] + list(NETS)

    def record(name, family, B, k, passes, extra=None):
        row = {"name": name, "family": family, "B": B, "k": k,
               "passes_per_token": per_token(passes)[1]}
        for net in nets:
            row[f"s_per_token_{net or 'server'}"] = per_token(passes, net)[0]
        row.update(extra or {})
        res["series"].append(row)
        return row

    ar = record("AR", "ar", 1, 1, gen_ar(C))
    for B in (4, 8, 16):
        for k in (1, 2, 4, 8, 16):
            if k <= B:
                record(f"DLM B={B}", "dlm", B, k, gen_dlm(B, k, C))
    for a in SPEC_ACCEPT:
        for k in (2, 4, 8):
            record(f"SpecAR a={a}", "spec", k + 1, k, gen_spec(k, a, C), {"accept": a})

    one = pass_cost(8, 128, C, 4)
    res["example_pass_B8_ctx128"] = {k: v for k, v in one.items() if k != "ops"} | {
        "ops": one["ops"]}
    json.dump(res, open(out_path, "w"), indent=2)

    print(f"[cost] N=2^{RING.bit_length() - 1} T={C['T']} tokens/ct, usable={C['usable']} "
          f"levels, depth/layer={one['depth_per_layer']}, boot={C['boot']:.0f}s")
    print(f"[cost] one DLM pass (B=8, ctx=128): server {one['server_s']:.0f}s = "
          + ", ".join(f"{k} {v:.0f}s" for k, v in one["by_op_s"].items())
          + f"; up {one['bytes_up'] / 1e6:.0f} MB, down {one['bytes_down'] / 1e6:.0f} MB")
    print(f"{'series':<16}{'k':>3}{'pass/tok':>9}{'server s/tok':>14}"
          + "".join(f"{n + ' s/tok':>14}" for n in NETS))
    for r in res["series"]:
        print(f"{r['name']:<16}{r['k']:>3}{r['passes_per_token']:>9.2f}"
              f"{r['s_per_token_server']:>14.1f}"
              + "".join(f"{r['s_per_token_' + n]:>14.1f}" for n in NETS))


if __name__ == "__main__":
    main()
