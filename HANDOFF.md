# FHE-DLM — Agent Handoff (2026-10-05)

You are picking up the **E0 (cost model + microbenchmarks, kill gate)** stage of
the FHE-DLM project: encrypted (CKKS) inference for a ~110M masked/block
diffusion LM (MDLM / BD3-LM, GPT-2 tokenizer, d=768, 12 layers, L ≤ 256), where
the client is in the loop and each encrypted server pass commits k tokens.
Headline claim: fewer rounds and less wall-clock per committed token than an
HE-friendly AR model at matched quality. Target venue: ACL 2027 (Jan ARR cycle,
plan for ~Jan 5); ICML only if gates E0 and E2 are clearly positive by Dec 1.

The experiment plan (rev. 1, 2026-10-02) lives with the user, not in this repo.
Experiments: E0 cost model, E1 HE-friendly conversion, E2 ordering ablation,
E3 few-step student, E4 main Pareto, E5 encrypted end-to-end, E6 leakage
attack, E7 network + 2PC table. Claims C1–C8. **No quality experiment has run
yet**; everything below is E0 systems work.

---

## 1. TL;DR of what was found

1. **N = 2^16 is infeasible** under the plan's constraints (128-bit classic,
   UNIFORM_TERNARY secret). Best 2^16 setting (59/60-bit moduli, level budget
   [2,2]) leaves **3 levels** after bootstrap at ~9.5 bits (full slots); a
   degree-15 GELU alone needs 5 levels. → **Use N = 2^17, budget [4,4]**:
   max depth 43, **22 levels** after bootstrap, bootstrap 68 s on CPU
   (full slots), **9.2 bits** precision (full slots), 39 GB RSS.
2. **C1 kill condition NOT triggered** (calibrated model, CPU): AR decode step
   **226 min/token**; DLM B=8,k=4 **88**; B=16,k=8 **45**; B=16,k=16 **30**.
3. **Speculative AR is the real competitor**: client-drafted spec-AR at
   acceptance α=0.8 costs 69 (k=4) / 54 (k=8) min/token — it **beats DLM at
   k ≤ 4** and ties at k = 8. DLM's systems win depends on keeping quality at
   k ≥ 8. This makes C4/E4 decisive, not C1.
4. **Absolute cost is the E5 risk**: one encrypted pass ≈ **3.9 CPU-hours**
   (matvec ≈ 90%: compute + on-the-fly plaintext encoding). 256 tokens at k=4
   ≈ 375 CPU-hours → E5 requires GPU (FIDESlib), and nobody has checked that
   FIDESlib builds on the RTX 6000 (sm_75) or that 2^17 bootstrap keys fit in
   24 GB (they used 39 GB RSS on CPU).
5. Network (RTT + transfer) is < 1% of wall-clock while the server is on CPU.

## 2. Machines and access

All compute is on two Notre Dame machines; the local Mac checkout
`/Users/jiachenzhao/research/fhe-dlm` is the **source of truth** — edit there,
rsync out. Git: private repo https://github.com/jennazhao7/fhe-dlm
(`main`); tjws and CRC copies are plain rsync targets, not clones. `hf-cache/`
and `third_party/` exist only on CRC and are git-ignored.

### tjws-03 (CPU dev box) — `tjws-03.cse.nd.edu` (NOT `tjws.cse.nd.edu`)
- User `jzhao7`, password only. `~/.ssh/config` on the Mac has `Host tjws`
  with ControlMaster (`~/.ssh/cm/`, persist 8h). The user must log in once in
  a terminal tab (`ssh tjws`); then `ssh tjws '…'` / `rsync … tjws:` work
  without a password. Check with `ssh -O check tjws`. It expires overnight.
- Hardware: Xeon W-1290P 20 cores, 125 GB RAM, Quadro RTX 5000 16 GB.
- Project: `~/fhe-dlm`. Python env: conda `fhedlm`
  (`$HOME/miniconda3/envs/fhedlm`, Python 3.10, **openfhe-python 1.5.1**,
  numpy<2, scipy, matplotlib; no torch). `FHEDLM_CONDA_ENV` is exported at
  the top of `~/.bashrc` (backup `~/.bashrc.bak.fhedlm`).
  Always run with `PYTHONNOUSERSITE=1` (`~/.local` has stray packages).
- Other OpenFHEs on the box (don't use): system C++ 1.2.1 in `/usr/local`;
  `~/fhe-vit-project/venv` (openfhe-python 1.3.0 + torch 2.7, old project).
- Run long jobs in tmux, logs in `results/logs/`:
  `tmux new -d -s NAME "export PYTHONNOUSERSITE=1; \$FHEDLM_CONDA_ENV/bin/python scripts/X.py results/X.json > results/logs/X.log 2>&1; echo EXIT=\$? >> results/logs/X.log"`
- **No per-user memory limit.** A full-slot [1,1] bootstrap key-gen at 2^16
  once triggered the global OOM killer. Every script caps RLIMIT_AS
  (48–60 GB) and runs each risky config in a child process.

### CRC (training + GPU) — `crcfe01.crc.nd.edu`
- **Use the user's shared master socket; never ask for a password login:**
  ```bash
  ssh -tt -S "$HOME/.ssh/crc-dp-grpo.sock" -o BatchMode=yes jzhao7@crcfe01.crc.nd.edu
  ssh -S "$HOME/.ssh/crc-dp-grpo.sock" -o BatchMode=yes jzhao7@crcfe01.crc.nd.edu 'cmd'
  rsync -a -e "ssh -S $HOME/.ssh/crc-dp-grpo.sock -o BatchMode=yes" … jzhao7@crcfe01.crc.nd.edu:…
  ```
  If `ssh -S … -O check jzhao7@crcfe01.crc.nd.edu` fails, ask the user to
  re-authenticate the socket. Every remote command prints
  `Loading CRC_default/1.1` — filter it.
- Project: `/groups/tjung/jzhao7/fhe-dlm` (5 TB group fs; `~/fhe-dlm` on CRC
  symlinks to it; CRC home is only 100 GB, ~30 GB free). Has CRC-only
  `hf-cache/` (4.1 GB: ELF-B-owt-torch, gpt2-large, t5-small), `third_party/ELF`,
  and earlier `results/step1_smoke.json`, `step3_rounds*.json`,
  `step4_poly*.json` (from before this handoff period — **not reviewed here**).
  tjws results are mirrored to `results/tjws/`. Sync code with
  `--exclude results/`.
- Env: conda `fhedlm` at `/users/jzhao7/.conda/envs/fhedlm` — Python 3.10,
  torch 2.5.1+cu121, transformers 4.44.2, openfhe-python 1.5.1.
  `cluster/setup_env.sh` built it (front-end only; compute nodes have no
  outbound network — prefetch with `cluster/prefetch.sh`).
- GPUs: queue `gpu@@jung_gpu` = one node `qa-rtx6k-019`, 4× Quadro RTX 6000
  24 GB (Turing sm_75), shared with the user's other projects — don't grab
  cards speculatively. `free_gpus.sh @jung_gpu`. As of 2026-10-05 all 4 free,
  no jzhao7 jobs. GPU resource is `gpu_card` (shortcut `gpu`) → `-l gpu=1` works.

### Job launch procedure (user's required way)
1. `qstat -u jzhao7` **before** submitting or resubmitting anything.
2. Directives (GPU):
   ```bash
   #$ -q gpu@@jung_gpu
   #$ -l gpu=1
   #$ -pe smp <cores>
   #$ -M jzhao7@nd.edu
   #$ -m abe
   ```
   CPU jobs use `-q long` instead of the GPU queue/resource. `-M`/`-m` must be
   **file directives and always a pair** (CRC's qsub wrapper rejects one
   without the other; a command-line `-M` doesn't satisfy it). SGE reads `#$`
   only from the submitted file, so each job script repeats them.
3. `qsub path/to/job.job`, then `qstat -u jzhao7` again.

All four `cluster/job_*.sh` were updated on 2026-10-05 to carry
`-M jzhao7@nd.edu` / `-m abe` (local copy only — **not yet synced to CRC**).
No GPU job template exists yet; build one from the directives above +
`cluster/job_common.sh` + `cluster/env.sh`.

## 3. Repo layout (what matters for E0)

| Path | What |
|---|---|
| `scripts/openfhe_levels.py` | Sweep (N ∈ {2^16,2^17}) × (scaling/first ∈ {59/60, 50/55, 45/50, 40/45}) × budget {[4,4],[3,3],[2,2],[1,1]}: max depth at ring, bootstrap depth, live sparse-slot bootstrap (time, levels left, precision, RSS) in a child process; "best" per ring requires ≥ 10 bits. |
| `scripts/e0_microbench.py` | Per-op costs at 2^16/2^17 (59/60, max depth): add, ct×pt, ct×ct+relin, rotate, hoisted rotate at 3 levels; Chebyshev GELU on [-8,8] deg 15/31/63/119; bootstrap per budget, sparse and full slots. Skips budgets with no levels left. |
| `scripts/e0_matvec_bench.py` | Real BSGS matvec at 2^17 (d padded 768→1024, interleaved slot = dim·T + token, T = 64, n1=n2=32, hoisted baby steps, diagonals encoded on the fly), shapes 1→1, 1→3, 3→1 at levels 21 and 32, checked against numpy; LM head (50 blocks) extrapolated. |
| `scripts/e0_cost_model.py` | Analytical model: op counts per pass (layout, attention, norms, FFN, LM head, confidence head, bootstraps) × measured costs; matvec charged at measured per-diagonal cost (`MATVEC_JSON`, set None for pure op-sum). Simulates generating L=256 for DLM (B∈{4,8,16}, k≤B, +1 KV pass/block), AR, spec-AR (α∈{0.6,0.8}); LAN/WAN/mobile. All structural assumptions are in its docstring. |
| `scripts/e0_fig1.py` | Fig. 1 (min per committed token vs k) → `results/fig1_cost_per_token.{png,pdf,md}`. |
| `scripts/smoke_test.py, step3_rounds.py, step4_poly.py, check_rsqrt_fit.py`, `src/fhedlm/*` | Earlier plaintext work on ELF (perturbation/rounds/norm polynomialization); predates this period, not touched. |
| `cluster/` | CRC env + job scripts (see §2). `env.sh` never installs anything. |
| `results/` | All E0 outputs + `logs/`. `*_ofhe1.3.0*` = same sweep on openfhe 1.3.0; `*_2e16only*` = pre-2^17 sweep. |

Run order to reproduce on tjws (≈ 2–3 h total, run sequentially so timings
don't contend): `openfhe_levels.py` → `e0_microbench.py` → `e0_matvec_bench.py`
→ (locally or anywhere) `e0_cost_model.py` → `e0_fig1.py`.

## 4. Results in detail (openfhe-python 1.5.1, tjws CPU, 20 threads, FLEXIBLEAUTO, HEStd_128_classic, UNIFORM_TERNARY)

**Levels / bootstrap** (`results/openfhe_levels.json`, `e0_microbench.json`):
- Bootstrap depth: [4,4] 22, [3,3] 20, [2,2] 18, [1,1] 16.
- Bootstrapping requires first_mod − scaling_mod ≤ 7 bits (10-bit gaps fail).
- 2^16 max depth: 59/60 → 20, 50/55 → 24, 45/50 → 27, 40/45 → 30. Only
  59/60 [2,2] reaches ≥ 10 bits (13.7 sparse / 9.5 full slots, 3 levels left).
  Smaller moduli: ~5.7 bits (50/55), < 1 bit (45/50), decode fails (40/45).
- 2^17 max depth: 59/60 → 43, 50/55 → 50, 45/50 → 56, 40/45 → 64. Only [4,4]
  fits under a 48 GB cap; 59/60 [4,4]: 22 levels left, 38 s / 12.1 bits sparse
  (4096 slots), 68 s / 9.2 bits full (65536 slots), 33/39 GB RSS. [3,3]/[2,2]
  at 2^17 exceed 48 GB.
- [1,1] never runs live (≈ one rotation key per slot; OOM).
- openfhe 1.3.0 vs 1.5.1: identical depths, 1.5.1 key setup ~2× faster, ~1 bit
  less precision.

**Primitive costs** (fresh → near bottom of chain): 2^16 ct×ct+relin 107→15 ms,
rotate 119→15 ms, ct×pt 10→0.6 ms; 2^17 ct×ct 519→62 ms, rotate 507→51 ms,
ct×pt 41→1.8 ms. Hoisted rotation measured *slower* than plain `EvalRotate`
through the Python binding (unexplained; probably binding overhead or OpenFHE
already hoisting internally). GELU Chebyshev on [-8,8]: deg 15 depth 5 err
4.5e-2; deg 31 depth 6 err 9e-5; deg 63/119 depth 7 err ~1e-12; 2^17 time
6.7 / 9.8 / 15.5 / 24 s.

**Matvec** (`results/e0_matvec.json`, 2^17, error ≤ 3e-11 everywhere):
level 21: 1→1 131 s (rot 18, encode 67, mult+add 46); 1→3 377 s; 3→1 381 s.
Level 32: 1→1 73 s; 1→3 209 s; 3→1 240 s. Op-sum model is within 1.0–1.6× on
compute but omits encoding, which is 1–3× compute. Pre-encoding all ~174k
diagonals at 2^17 would need TBs → encoding is a real per-pass CPU cost.
LM head (1→50) extrapolated: 3,400–6,200 s per pass.

**Cost model** (`results/e0_cost_model.json`, Fig. 1): one pass (B=8,
ctx=128) ≈ 13,900 s; matvec 12,600 s, bootstraps 750 s (11 × 68 s), rotations
outside matvec 400 s, GELU 90 s, attention ct×ct 70 s. Upload ≈ 20 MB
(seed-compressed, 19 towers), download ≈ 10 MB (compacted logits). Per-token
table in `results/fig1_cost_per_token.md`. DLM at k=1 is *worse* than AR
(B/k rounds + the KV pass per block).

**Model assumptions to revisit** (all in the `e0_cost_model.py` docstring):
interleaved layout with padding to 1024; attention via B token-shift
alignments with wrap masks, power-2 attention with a public constant
normaliser (no inverse); fixed-constant norm = mean-subtract + constant scale
(no variance); 1 bootstrap per T tokens placed on the residual stream; GELU
degree 31; extra clean KV pass per block (BD3-LM style); spec-AR acceptance is
i.i.d. α; client cost measured on the 20-thread server (a phone is slower);
`levels_left_after` from OpenFHE is 1 higher than the levels sweep's usable
count (likely the FLEXIBLEAUTO extra tower — unconfirmed; the model uses −1).

## 5. Implications for the experiment plan (to raise with the user)

1. **E5 parameter set**: change N = 2^16 → **2^17, budget [4,4], 59/60**.
2. **Robustness noise sweep**: the plan sweeps 2^-10…2^-20, but full-slot
   bootstraps deliver ~2^-9. Extend down to ≤ 2^-8; if quality collapses there,
   evaluate OpenFHE iterative (two-pass) bootstrapping.
3. **C4/E4 is the decisive comparison**: DLM must hold quality at k ≥ 8 against
   spec-AR with realistic acceptance. Measure acceptance α of the 2- and
   4-layer drafts in plaintext early (it is cheap and sets the bar).
4. **Hardware risk for E5** must be resolved in October, not November (below).
5. Hardware line in the plan is right: "~24 GB GPU" = CRC RTX 6000s (the
   16 GB card is tjws's, not used for E5).

## 6. Recommended next steps (in priority order)

1. **GPU feasibility check on CRC (highest priority, E5 go/no-go).** One job on
   one RTX 6000: build FIDESlib (check sm_75 support first), run a 2^17,
   [4,4], full-slot bootstrap and one 1024×1024 BSGS matvec; record time and
   peak GPU memory. Fallback if keys don't fit in 24 GB: fewer bootstrap
   rotation keys (sparse slots / larger budget), key compression, or 2^17 with
   a sparse secret (needs a security argument). Write a GPU job template from
   §2's directives first; `qstat` before `qsub`.
2. **Reduce the matvec cost** (90% of a pass): cache encoded diagonals for the
   hot layers within RAM budget, encode at the exact level used, try
   double-hoisted BSGS, and re-benchmark; reconsider having the LM head
   (50 blocks, ~30% of matvec) restricted/compacted earlier.
3. **Per-layer depth audit**: the model assumes 18 levels/layer (≤ 22 usable).
   Build one full encrypted transformer layer at 2^17 (QKV → attention →
   out-proj → norm → FFN with GELU) and confirm depth, error, and time against
   the model. This is also the first piece of E5.
4. **Sync** the updated `cluster/job_*.sh` (new -M/-m directives) to CRC.
5. **Start E2 on the frozen original teacher** (plaintext, cheap, plan P0 for
   Oct 2–23) and the AR control + AR-FIM training (P0, Oct 9–30) — neither has
   started. Verify the checkpoint names flagged "verify" in the plan
   (`kuleshov-group/mdlm-owt`, `bd3lm-owt-block_size{4,8,16}`) and licenses.
6. Minor: confirm the `levels_left_after` off-by-one; explain the slow hoisted
   rotation (try the C++ API before relying on hoisting in the model).

## 7. Gotchas

- OpenFHE eval keys are **process-global** in openfhe-python. Between contexts
  call `openfhe.ClearEvalMultKeys()`, `cc.ClearEvalAutomorphismKeys()`,
  `openfhe.ReleaseAllContexts()` — or isolate in a child process (the
  scripts use `multiprocessing` spawn + a queue drained before `join`).
- C++ `bad_alloc` **aborts** the Python process (not catchable) — hence the
  child processes.
- A bootstrap budget whose depth ≥ total depth is **not refused** by OpenFHE:
  it returns garbage (a "40-bit" refresh) or corrupts the heap. Skip them.
- `EvalBootstrapSetup(budget, [0,0], slots)` + `EvalBootstrapKeyGen(sk, slots)`
  + `MakeCKKSPackedPlaintext(x, 1, level, None, slots)` for sparse slots.
- Both SSH masters expire (tjws: 8 h ControlPersist; CRC: whenever the user's
  socket dies). Check before long command chains.
