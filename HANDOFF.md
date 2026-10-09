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
   max depth 43, **21 usable levels** after bootstrap (OpenFHE reports 22;
   off-by-one confirmed, §8), bootstrap 68 s on CPU
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
  The socket has lived at `$HOME/.ssh/crc-dp-grpo.sock` and at
  `/tmp/crc-dp-grpo.sock` — `-O check` both and use the live one. If neither
  works, ask the user to re-authenticate it (they run
  `ssh -o ControlMaster=yes -o ControlPath=/tmp/crc-dp-grpo.sock -o ControlPersist=8h … jzhao7@crcfe01.crc.nd.edu`). Every remote command prints
  `Loading CRC_default/1.1` — filter it.
- Project: `/groups/tjung/jzhao7/fhe-dlm` (5 TB group fs; `~/fhe-dlm` on CRC
  symlinks to it; CRC home is only 100 GB, ~30 GB free). Has CRC-only
  `hf-cache/` (4.1 GB: ELF-B-owt-torch, gpt2-large, t5-small), `third_party/ELF`,
  and earlier `results/step1_smoke.json`, `step3_rounds*.json`,
  `step4_poly*.json` (from before this handoff period — **not reviewed here**).
  tjws results are mirrored to `results/tjws/`. Sync code with
  `--exclude results/`.
- Env: conda `fhedlm` at `/groups/tjung/jzhao7/conda-envs/fhedlm` (moved out of
  `$HOME` 2026-10-08; activate by path; package caches in
  `/groups/tjung/jzhao7/{conda-pkgs,pip-cache}`) — Python 3.10,
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
6. Minor: ~~confirm the `levels_left_after` off-by-one~~ (done, §8); explain
   the slow hoisted rotation (try the C++ API before relying on hoisting in
   the model).

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

## 8. Session 2026-10-05 (cloud agent, no access to tjws/CRC)

This session ran in a cloud container (4 cores, 15 GB, no SSH to ND, no GPU,
huggingface.co blocked), so nothing was run on tjws or CRC. Done instead:

1. **FIDESlib on sm_75 compiles** (§6.1, first half). FIDESlib 2.1.3
   (`593ad73`) + its patched OpenFHE (`fideslib-ref-v1.5.1.1`) build cleanly
   with `-DFIDESLIB_ARCH=75`, CUDA 12.4.131, gcc 12.4. The binary holds sm_75
   SASS only. Turing is not in FIDESlib's default arch list (starts at
   sm_80), but the code has no sm_80-only features. **CUDA 12.0 does not
   work** (nvcc 12.0 rejects its `std::source_location` defaults), so CRC
   needs a CUDA ≥ 12.4 module or the conda-forge toolkit (see
   `cluster/setup_fideslib.sh` header). Whether the kernels *run* correctly
   on sm_75 is still unchecked.
2. **GPU job ready, not submitted.** `gpu/fides_e0/` (C++ bench on the
   FIDESlib API, same parameters as the CPU sweep: boot at 4096/65536 slots
   [4,4]; 1024×1024 BSGS matvec at level 21, with encode / H2D / compute
   timed separately; GPU memory after each phase). `cluster/setup_fideslib.sh`
   (front end: fetch + build into `third_party/`), `cluster/job_gpu_fides.sh`
   (GPU job template with the §2 directives; one process per run;
   nvidia-smi sampling; writes `results/gpu_fides.json`). Both code paths
   were validated on FIDESlib's OpenFHE CPU fallback at N=2^12: matvec error
   3.6e-13; bootstrap 19.1 bits (256 slots) / 15.1 bits (2048 slots),
   levels_left_after 22. Dry run: `FIDES_E0_CPU=1 FIDES_E0_LOGN=12 fides_e0 matvec`.
   (The CPU fallback of `EvalMult(ct, pt)` has a FIDESlib bug — bad
   `any_cast` to `ConstPlaintext` — that was patched only in the scratch copy
   for the dry run. It does not touch the GPU path, so no patch is shipped.)
3. **The 24 GB question, estimated** (to be confirmed by the job): one
   key-switching key at N=2^17, depth 43, 59/60, dnum=3 is **354 MB**
   (serialized); a fresh ciphertext is 88 MB. Sparse [4,4] bootstrap
   (4096 slots) needs **63 rotation keys = 22 GB** before the C2S/S2C
   plaintexts and working ciphertexts. Full-slot key count not measured (OOM
   at 14 GB here); ≥ 63. The n1=n2=32 matvec needs 62 keys (= 22 GB) too.
   → **Expect E5 not to fit on one 24 GB RTX 6000.** Shrinking keys by lower
   dnum is not available: logQP is already ~3.5 kbit, at the 128-bit bound
   for 2^17. Fallbacks to evaluate:
   (a) FIDESlib multi-GPU (NCCL limb partitioning) over the node's 4 cards
       — needs NCCL in the build and all four cards;
   (b) keep keys in host memory and stream them over PCIe (~30 ms per key
       at ~12 GB/s, roughly doubling a rotation);
   (c) sparse-secret encapsulation (FIDESlib `SPARSE_ENCAPSULATED`):
       shallower bootstrap → fewer towers → smaller keys;
   (d) fewer BSGS keys in the matvec (n1·n2 split, reuse of giant steps).
4. **`levels_left_after` off-by-one confirmed** (§6.6):
   `scripts/e0_levels_offbyone.py` → `results/e0_levels_offbyone.json`.
   Usable levels = `depth − GetLevel()` − 1 = `depth − bootstrap_depth`,
   for budgets [2,2] and [4,4] and two depths. The cost model's −1 and
   `USABLE = 21` in `e0_matvec_bench.py` are right; the TL;DR now says 21.
5. **Checkpoint check** (§6.5) could not run here (huggingface.co blocked).
   `cluster/check_checkpoints.sh` (front end) prints existence, license and
   weight sizes for the four "verify" repos.

Still open for you: sync `cluster/` to CRC (§6.4: rsync from this repo,
`--exclude results/`), then `bash cluster/check_checkpoints.sh`,
`bash cluster/setup_fideslib.sh`, `qstat -u jzhao7`,
`qsub cluster/job_gpu_fides.sh`. §6.2, §6.3 and the hoisting question need
tjws-class RAM (≥ 40 GB) and were not started.

## 9. Session 2026-10-05, part 2: layer audit + token-group packing (cloud, no cluster)

1. **Per-layer depth audit (§6.3), done at toy size.** `scripts/e0_layer_audit.py`
   builds the cost model's layer as real CKKS (same interleaved layout and kernels),
   checks it against numpy, and records levels per stage. Toy shape, real crypto:
   N=2^12, depth 43, 59/60, [4,4], full slots, d=48→D=64, head 16, T=32, B=4.
   Level accounting does not depend on N or d. Full size:
   `--logn 17 --d 768 --dpad 1024 --hd 64 --B 8` on tjws (not run).
   Result (`results/e0_layer_audit.json`): **18 levels/layer (attention 8 + FFN 10)**,
   matching the model's total. The split differs from the model's count:
   - /Z, the head mask and the replicate step fold into one ptm (−1 vs the model);
   - **GELU deg 31 costs 7 levels, not 6.** `e0_microbench.py` under-reported it
     by ignoring FLEXIBLEAUTO's pending rescale (fixed for future runs).
   12 layers run end to end: **10 bootstraps** with a full-depth (43-level)
   upload, max error 1.2e-3. Bootstraps can only sit between half-layers on the
   residual stream, and 2 layers (36) > 21, so after the fresh window there is
   one bootstrap per layer. With the model's 19-tower upload the count is 11,
   which matches the model's `ceil(216/21)`, but only by coincidence. The model
   now simulates this schedule from the audited half-layer depths.
2. **Numerics the real model must respect** (all surfaced by the toy run):
   - (a) The residual stream must satisfy |x/S| < 1 at every bootstrap; S folds into
     the norm constants and the out/down projections at no level cost. But
     bootstrap error is relative to S: ~9 bits at full slots at 2^17.
   - (b) GELU inputs must stay within the Chebyshev interval ([-8, 8]). One input
     at 10.4 made decryption fail.
   - (c) The public normaliser Z and the fixed norm constants need calibration
     (uncalibrated power-2 attention diverged within 5 layers). These are E1 inputs.
3. **Token-group packing: ~2.5× cheaper matvec, validated.** A pass has only B
   live tokens in T=64 slots. Copy them into the G=T/B token groups and give
   each group its own weight block: one 1024-diagonal pass then computes up to G
   block products, because the diagonal method never mixes token slots. Per
   layer: QKV, FFN-up and FFN-down each take 1 pass (were 3+3+3), plus one GELU
   call instead of 3. The LM head takes ⌈50·B/64⌉ passes instead of 50. The
   group masks fold into existing ops, so the audit (`--packed`,
   `results/e0_layer_audit_packed.json`) shows the **same 18 levels/layer,
   10 bootstraps, max err 1.3e-3 over 12 layers**. It includes the block's own
   keys plus an encrypted KV-cache ct, and keeps garbage in idle slots that the
   next norm drops.
   Cost model with `FHEDLM_PACKED=1` (`results/e0_cost_model_packed.json`,
   `results/fig1_cost_per_token_packed.*`); the baseline files are unchanged:

   | min / committed token | baseline | packed |
   |---|---|---|
   | one pass B=8 ctx 128 (s) | 13,900 | 5,560 |
   | AR | 226 | 76 |
   | DLM B=8, k=4 | 88 | 35 |
   | DLM B=16, k=8 | 45 | 21 |
   | DLM B=16, k=16 | 30 | 14 |
   | spec-AR α=0.8, k=4 | 69 | 26 |
   | spec-AR α=0.8, k=8 | 54 | 23 |

   C1 still holds: DLM beats AR from k ≥ 2. But packing helps the small-B
   passes (AR, spec-AR verify) as much as DLM. **Spec-AR now beats DLM at
   k ≤ 4 by more (26 vs 35), ties at k=8, and DLM wins clearly only at
   B=16, k=16.** This strengthens §5.3: E4 decides the paper. The E5
   estimate drops from ~375 to ~150 CPU-hours (256 tokens, B=8, k=4).
   Bootstrap is now 13% of a pass and rotations 10%.
4. **Sparse-slot diagonal encoding: measured, no net gain.** In a dims-inner
   layout every diagonal is period-D, so OpenFHE can encode it with
   `slots=D`: correct (err ~1e-12) and 2.3–3.9× faster at 2^15–2^16. But that
   layout needs 1.5–2× more diagonals (token-boundary wrap), and OpenFHE still
   runs the full-size NTT. Break-even through the API; a real win needs a custom
   "small NTT + broadcast" encoder (C++/GPU), so this is parked.

Next, in order: run `e0_layer_audit.py` at full size on tjws (timings +
confirm 18 levels at d=768); have E1 report residual / GELU-input ranges of the
real model (items 2a–c); measure spec-AR acceptance α early (item 3 makes it
the decisive number).

## 10. Session 2026-10-09: first GPU run on CRC (job 1519057)

- FIDESlib 2.1.3 builds on CRC with `cuda/13.2.1` (nvcc 13.2 has compute_75;
  CRC has no 12.4) + system gcc 11.5 + pip cmake < 4; ~15 min.
  `results/crc/setup_fideslib.log`. The fhedlm env now lives in
  `/groups/tjung/jzhao7/conda-envs/fhedlm`.
- **All three runs OOM on one 24 GB RTX 6000** (nvidia-smi peak 22.7 GB;
  `results/crc/fhedlm_gpu_fides.o1519057`). FIDESlib kernels do run on sm_75
  (keygen + context load completed).
  - boot 4096 slots [4,4]: load_context put **34 rotation keys (12.0 GB) +
    120 C2S/S2C plaintexts (5.9 GB) = 18.8 GB** on the GPU, then OOM inside
    EvalBootstrap (working buffers).
  - boot 65536: OOM (more keys).
  - matvec: OOM loading its **62 BSGS rotation keys** (~354 MB each).
- **Bug:** every run logged `exit 0` and `results/gpu_fides.json` has no runs —
  FIDESlib's CUDA-failure path apparently exits 0. Judge runs by the `.o` log.
- Fixes to try next, cheapest first:
  1. matvec with **2 keys** (rotate by T and by 32T) generating baby/giant
     steps by successive rotation — same rotation count, no hoisting (which
     did not help on CPU anyway); ~0.7 GB of keys instead of ~22 GB.
  2. bootstrap: keep the 5.9 GB C2S/S2C plaintexts on the host and stream
     them, or a sparse-slot setup with fewer plaintexts; check FIDESlib
     options for host-resident precomputation.
  3. FIDESlib multi-GPU (NCCL) over the node's 4 cards.
  4. Sparse-secret encapsulation (shallower bootstrap, smaller keys).
- **Full-size layer audit on tjws (2026-10-09)**, `--logn 17 --d 768 --dpad
  1024 --hd 64 --B 8 --packed`, 2 layers, random weights:
  `results/e0_layer_audit_full_packed.{json,log}`. **18 levels/layer
  (attention 8 + FFN 10) = cost model**, GELU deg 31 = 7 levels; max err
  1.4e-4; max |x/S| 0.54; max GELU input 4.0. 62.5 GB RSS, setup 42 s,
  **run 1,868 s for 2 layers (934 s/layer)** — but the audit encrypts at the
  full 43-level chain with no bootstrap, so every op runs on ~44→8 towers,
  not the post-bootstrap window the model prices (~22→1). The ~2.6× gap to
  the model's ~360 s/layer is mostly that; to calibrate time, rerun with the
  input encrypted at the post-bootstrap level (21).

### Job 1519942 (2026-10-09): 2-key matvec runs on the GPU

- `fides_e0 matvec` now uses **2 rotation keys** (successive rotations by T for
  baby steps, Horner over giant steps by 32T). **Fits: 7.9 GB peak**, max err
  2.2e-11 (`results/crc/gpu_fides_1519942.json`).
- **GPU compute 1.37 s** (rotate 0.34, mult+add 1.03) vs 64 s on the 20-core
  CPU at level 21 → ~47×. **But total 134 s ≈ CPU**: host-side diagonal
  encoding 92 s + H2D 41 s. Pre-encoding all 1024 diagonals ≈ 23 GB/matrix,
  does not fit. **Next lever: encode diagonals on the GPU** (each diagonal is a
  period-D broadcast of 1024 values → small NTT + broadcast kernel), so the
  matvec cost becomes ~compute.
- Bootstraps still OOM (exit codes now reported correctly): sparse 4096 slots
  loads 34 keys (12.0 GB) + 120 plaintexts (5.9 GB) = 18.8 GB then OOMs in
  EvalBootstrap; full slots needs 248 plaintexts = 12.3 GB before keys.
  FIDESlib has no host-resident option for these; its multi-GPU path is
  compiled out (CMake found no NCCL; the env's NCCL is the cu12 torch wheel,
  FIDESlib is built with CUDA 13.2).

### Job 1520180 (2026-10-09): GPU-side diagonal encoding — matvec 134 s → 3.7 s

- `gpu/fides_e0/src/gpu_encode.{hpp,cu}` + `gpu_encode_host.cpp`, mode
  `fides_e0 matvec_enc` (`qsub -v FIDES_RUNS="matvec_enc" cluster/job_gpu_fides.sh`).
- Layout switched to **token-major** (`slot = token*D + dim`), so every diagonal
  plaintext is **period-D** = OpenFHE sparse encoding with slots = D: only 2D
  nonzero coefficients at stride N/(2D)=64. Host does OpenFHE's own size-D
  `FFTSpecialInv` + rounding (0.21 s for all 2048), uploads 2D int64 per
  plaintext (0.02 s total); a CUDA kernel scatters them mod each prime into a
  scratch RNSPoly, FIDESlib's NTT converts, and the limbs are copied D2D into
  the device plaintext behind a public handle (plaintext limbs are "constant",
  without the NTT aux buffer, hence the scratch). No FIDESlib patch.
- **Bit-exact vs OpenFHE's encode (0 residues differ).**
- Rotations cross token boundaries in this layout: diagonal i splits by the
  mask j < D−i into rot(x,i) and rot(x,i−D) = rot(rot(x,−D),i) halves → 2047
  plaintext mults, two baby sets, 3 keys (+1, +32, −1024).
- **Total 3.74 s** (GPU encode 1.23, mult+add 2.05, rotate 0.46) vs 134 s with
  host encoding and 131 s on the 20-core CPU; max err 1.6e-10; 13.9 GB peak.
- Not yet folded into `e0_cost_model.py`: per 1024-diagonal block the GPU
  cost is now ~3.7 s at level 21 (vs ~74 s CPU in the packed model), so
  matvec stops dominating and the **bootstrap (still OOM on one card)
  becomes the E5 bottleneck**.
