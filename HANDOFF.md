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
   (full slots), **9.2 bits** precision (full slots; that was a sample --
   over all 65536 slots the max-error is 8.4, RMS 11.0, §11), 39 GB RSS.
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
6. **Bootstrap precision: solved by two-pass bootstrapping (§11).** Full slots
   on 4 GPUs: 16.4 bits max / 18.9 RMS in 2.23 s (one pass: 8.5 / 11.0 in
   1.14 s), one extra level (20 usable), no extra memory. MDLM-OWT is clean at
   that noise (gen-PPL 112 vs 116 clean) even with its position-0 sink.
   FIDESlib's GPU `EvalBootstrap` silently ignores `numIterations`; the second
   pass is composed in `fides_e0 iterboot`. The sink forms when position 0
   holds [MASK] **or** `<|endoftext|>` amid content -- §10's "keep position 0
   unmasked (BOS)" is wrong; decode a content token at position 0 first.
7. **The bottleneck is now quality, not systems (§11 Next, 2026-10-27).**
   Zero-shot, the frozen MDLM does not survive any E5 op (fixed norms blow
   the residual stream to 1e10; (q·k)² attention NLL 8 vs 3.8; deg-31 GELU
   diverges on 0.04% of inputs beyond ±8), so E1 is a KD fine-tune.
   E2: at matched entropy, ordering does not matter; the window width does.
   MDLM whole-sequence passes hold up to k = 8-16 (gen-PPL 42-94), while BD3
   B = 16 is ~9× worse at k = 8. Spec-AR α with 2/4-layer drafts is
   0.39-0.45 on its own samples (≤ 1.8 tokens/pass), far below the
   cost model's 0.6-0.8.

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

### Job 1528196 (2026-10-11): bootstrap runs on 4 GPUs — full slots 1.1 s

- Multi-GPU FIDESlib (`cluster/setup_fideslib_mgpu.sh`: NCCL 2.32.3 from the
  `nvidia-nccl-cu13` wheel installed with `--target third_party/nccl-cu13`,
  FIDESlib → `third_party/fideslib-mgpu`), bench `gpu/fides_e0/build-mgpu`
  with `FIDES_E0_GPUS=4`, job `cluster/job_gpu_fides_mgpu.sh` (`-l gpu=4`,
  whole qa-rtx6k-019 node). `results/crc/gpu_fides_mgpu_1528196.json`.
- **Full slots (65536) [4,4]: 1.11 s / bootstrap** (1.106, 1.105, 1.123),
  22 levels left, **8.5 bits** (CPU 68 s, 9.2 bits) → ~60×. Loads 61 rotation
  keys (21.6 GB) + 248 plaintexts (12.3 GB) across 4 cards; **fullest card
  18.0 GB of 24** after warm-up. Setup: boot keygen 27.5 s, load 73.6 s,
  host RSS 42.5 GB.
- Sparse (4096): 0.77 s, 11.6 bits, 11.0 GB fullest card.
- Job wall time 3 min 18 s. CRC's front end hung twice around this job (login
  banner, then nothing); a stale ControlMaster after a network change looks
  the same — close it (`ssh -S … -O exit`) and log in again.
- Open: (1) the 3.7 s matvec was measured on 1 GPU; rerun `matvec_enc` in the
  4-GPU context (a real pass needs both in one context); (2) add a GPU column
  to `e0_cost_model.py` / Fig. 1 — rough per pass (packed, B=8): ~55
  1024-diagonal blocks × ~3.7 s + ~11 bootstraps × 1.1 s ≈ 4 min vs ~93 min
  on CPU; (3) full-slot GPU bootstrap precision is 8.5 bits — the noise sweep
  must cover ~2^-8.

### 2026-10-15: E5 fits on 3 GPUs in one context; MDLM fails ~9-bit noise (outliers)

**Job 1546459, `fides_e0 combo` on 3× RTX 6000** (`qsub -l gpu=3 -v
FIDES_E0_GPUS=3,FIDES_RUNS="combo 4" cluster/job_gpu_fides_mgpu.sh`): full-slot
bootstrap keys + matvec keys in one context (62 keys 21.9 GB + 248 plaintexts
12.3 GB; 17.9 GB/card after load). GPU-encoded matvec **2.81 s** (bit-exact,
err 3e-11; two-half variant so only 32 baby cts are live), bootstrap
**0.96 s**, 8.4 bits, 22 levels left. **Peak 22.1 GB on the fullest card** —
fits, ~2 GB headroom. `results/crc/gpu_combo_3gpu_1546459.json`.

**Job 1546416, `scripts/plain_probe.py` on MDLM-OWT** (no-flash fp32 build;
flash_attn rotary/omegaconf stubbed; WikiText-103 val, L=256, 256 seqs;
`results/plain/{ranges,noise}.json`):
- **Massive activation**: residual max jumps from 262 (layer 4) to **~21–23k
  from layer 5 on, at masked positions only** (unmasked max ≤ 1k; p99.9 only
  90–550). GELU inputs peak 131 (layers 5–6) but only 0.003–0.07% exceed 8.
  LayerNorm 1/std differs masked vs unmasked early (L0 2.50 vs 3.27) and has a
  10–28% CV across positions (supports H4 mask-aware constants).
- **Noise with σ = 2^-b · max|block out|** after every block: NLL@15% mask
  2.28 (clean) → 2.28 (b=16) → 6.8 (b=12) → 15.8 (b=10) → ~21 (b≈8.5);
  gen-PPL 90 → 1.5e5 at b=10. **The checkpoint as-is cannot take the 8.4–9.2
  bit bootstraps**, because the scale S is set by the one massive activation.
- Next: quantify the outlier (which channel(s), how constant at masked
  positions — masked positions are public in our protocol, so a near-constant
  massive value could be subtracted/re-added as a plaintext constant at no level
  cost) and rerun the noise probe with S excluding it; otherwise E1 must
  suppress it (clipping / activation regularisation during the HE-friendly
  fine-tune).

### 2026-10-17: the outlier is a position-0 attention sink; ~9 bits still costs 10× gen-PPL

Job 1574220 (`plain_probe.py outlier --gen`, `results/plain/outlier.json`):
- **All massive-activation tokens in layers 5–10 are sequence position 0, and
  only when position 0 is masked** (65/65 spike tokens; 0.51/seq = P(pos 0
  masked)); |x| 4k–23k, channels 682 (from L5), then 218/313/344/12/277/684.
  With position 0 unmasked there is no spike. → Keep position 0 unmasked (BOS /
  prompt token) in the encrypted protocol; it is public anyway.
- Subtracting a per-layer public constant (masked/unmasked mean) does nothing
  (spike is present half the time): NLL@15% 23.6 at 2^-9.
- **Oracle (outlier channels untouched, S over the rest ≈ 400–900):** 2^-10
  NLL@15% 2.33 (clean 2.29), 2^-9 2.56 / gen-PPL 929 (clean 90), 2^-8 3.88 /
  3,476, 2^-7 11.1. → Even without the sink, 8.4–9-bit bootstraps cost ~10×+
  gen-PPL; E5 needs ≳10 bits of effective precision or a model trained to
  tolerate the noise.
- Options for E1/E5: (a) noise-aware HE-friendly fine-tune (inject 2^-8..2^-9
  bootstrap-style noise during KD); (b) OpenFHE/FIDESlib iterative
  (two-pass) bootstrapping (~2× precision, extra levels/time); (c) smaller S at
  bootstrap points (per-channel scaling folded into weights, bootstrap after a
  norm); (d) all of the above. Next measurement: rerun the noise probe with
  position 0 forced unmasked (the realistic setting) to get the real S.

## 11. Session 2026-10-26: two-pass bootstrapping; noise probe at the bootstraps' real precision

### Two-pass (Meta-BTS) bootstrap at the E5 set (2^17, [4,4], 59/60, depth 43)

Error over **every** slot (max and RMS); usable = depth − level − 1 (§8).

| config | time | max bits | RMS bits | usable levels |
|---|---:|---:|---:|---:|
| GPU×4, full slots, 1 pass | 1.14 s | 8.53 | 10.96 | 21 |
| **GPU×4, full slots, 2 passes (composed)** | **2.23 s** | **16.40** | **18.94** | **20** |
| CPU OpenFHE, full, 1 pass | 62.7 s | 8.41 | 10.96 | 21 |
| CPU OpenFHE, full, 2 passes (built-in) | 124.3 s | 16.46 | 18.96 | 20 |
| CPU OpenFHE, full, 2 passes (composed) | 134.4 s | 16.40 | 18.96 | 20 |
| CPU OpenFHE, 4096 slots, 1 / 2 passes | 40.2 / 77.1 s | 11.77 / 22.39 | 13.48 / 24.48 | 21 / 20 |

- **FIDESlib's GPU `EvalBootstrap` silently ignores `numIterations` and
  `precision`**: `third_party/FIDESlib/api/CryptoContext.cpp` forwards them
  only on the CPU fallback; the GPU path calls `CKKS::Bootstrap(ct, slots,
  prescaled)`. Asking it for 2 passes runs 1.
- So the second pass is composed in `fides_e0 iterboot <slots> [budget]
  [precision]`: b1 = Boot(ct); diff = ct − b1; diff ×2^p by p self-adds;
  b2 = Boot(diff); out = b1 + b2·2^−p, with p = floor(1-pass max bits).
  `scripts/e0_iterboot.py --manual` is its CPU twin and matches OpenFHE's
  built-in run in the same key set (job 1629907: 16.40 vs 16.52 max, 18.96 vs
  18.96 RMS); the GPU matches the CPU built-in to 0.06 / 0.02 bits.
- OpenFHE supports only 1 or 2 iterations (`ckksrns-fhe.cpp:451`).
- Cost to E5: one level (21 → 20; a layer is 18, so still one bootstrap per
  layer) and ≈ +1.1 s per bootstrap (≈ +12 s on a ~4 min pass). Same 18.0 GB
  per card.
- Open: the GPU **4096-slot** `iterboot` run read 99 bits (error exactly 0 --
  probably a NaN decrypt). Unexplained; full slots are unaffected. The error
  loops in `fides_e0` (`boot_body` included) use `std::max` and are NaN-blind.
- Jobs: CPU 1629894 (built-in), 1629907 (+ composed); GPU 1629908
  (`cluster/job_iterboot.sh`, `cluster/job_gpu_iterboot.sh` -- its own script so
  it cannot overwrite `results/gpu_fides_mgpu.json`). Files:
  `results/crc/e0_iterboot{,_manual}.json`, `gpu_iterboot_1629908.json`, `.o` logs.

### The noise probe was calibrated ~2.5 bits too pessimistically

`plain_probe.py noise` adds N(0, (2^−b·S)²): `b` sets a Gaussian **σ** (an RMS),
but it was compared with the bootstraps' **max-error** bits. Bootstrap error is
near-Gaussian (max − RMS = 2.5 bits over 65,536 slots vs ≈2.1 expected), so
**one pass ≈ b 11 and two passes ≈ b 19**, not 8.5–9.

### The sink trigger (corrects §10, 2026-10-17)

`scripts/probe_sink_pos.py` → `results/plain/sink_pos.json` (CPU, 4 seqs,
L ∈ {256, 64}): the ~2e4 spike (block 9, position 0) forms when position 0
holds a **non-content token -- [MASK] or `<|endoftext|>`** -- while other
positions carry content (19.5k–24.0k). A content token at position 0 → no
spike (≤ 1.1k); no content anywhere (all masked) → none either. Pinning
`<|endoftext|>` at 0 is as bad as masking it.

`plain_probe.py noise --pin0` therefore keeps a **content** token at position
0: NLL never masks it; sampling decodes it **first**, from the all-masked state
(no sink yet), with `<|endoftext|>` excluded. In the protocol the client
commits position 0 in pass 1; with a prompt, don't prepend BOS.

### Noise probe at the real precision (job 1631162)

256 WikiText-103 seqs, L = 256, gen-PPL under GPT-2 Large at every level,
n = 64 samples, one seed. `results/plain/noise2_{sink,pin0}.json`.

| b | sink gen-PPL | sink NLL@15% | pin0 gen-PPL | pin0 NLL@15% |
|---:|---:|---:|---:|---:|
| clean | 115.8 | 2.276 | 119.6 | 2.152 |
| 9 | 409,077 | 20.51 | 2,122 | 2.276 |
| 10 | 157,870 | 14.86 | 269.2 | 2.182 |
| **11** (≈ 1 pass) | **32,597** | 11.50 | **190.7** | 2.158 |
| 12 | 5,162 | 6.70 | 186.0 | 2.152 |
| 13 | 2,209 | 3.29 | 118.6 | 2.153 |
| 16 | 116.0 | 2.286 | 117.0 | 2.152 |
| **19** (≈ 2 passes) | **112.3** | 2.276 | **117.0** | 2.152 |

(Clean NLL differs between conditions because pin0 never scores position 0.)

- **Two passes are clean in both conditions, sink included** → bootstrap noise
  does not need noise-aware training if E5 uses two-pass bootstrapping.
- **One pass**: the sink breaks it; pin0 keeps NLL clean but gen-PPL is 1.6×
  (190.7 vs 119.6), clean again by b = 13.
- pin0 dropped S_median from ~20,000 to ~185, but **S_max is still 17–18k**:
  something still spikes occasionally. Not attributed yet (hypothesis: an
  `<|endoftext|>` sampled mid-sequence during generation). Probably what leaves
  one-pass pin0 at 1.6×.
- Caveats: S is each batch's own max (a real S is a fixed public bound ≥ the
  max → slightly optimistic); noise after all 12 blocks vs ~11 bootstraps
  (slightly pessimistic); one generation seed, n = 64 (the clean-level runs
  spread ~3%).

### Next (re-prioritised 2026-10-27)

Bootstrap precision was the last E0 systems blocker, and it is solved. Nothing
left in E0 blocks the paper. The critical path is now quality, and **no quality
experiment has run yet**. Two unknowns decide the project:

- **C4/E4** -- does the DLM keep quality at k ≥ 8 against spec-AR (§9.3)? That
  needs spec-AR α and the E2 ordering curves.
- **E1** -- can MDLM be converted to the E5 circuit (fixed-constant norms,
  (q·k)²/Z attention, Chebyshev GELU)? The noise probe above keeps exact
  LayerNorm, softmax and GELU, so "clean at two-pass noise" does **not** mean
  "works encrypted".

In order (jobs submitted 2026-10-27, `cluster/job_plain_probe.sh`, one GPU each):

1. **E1 zero-shot op swap** (`scripts/approx_probe.py approx`, job 1640480 after
   e2ent -- 1636812/1640363 stalled on qa-rtx6k-019 GPU `2F:00.0`, which hangs at
   CUDA init; the job script now checks CUDA for 180 s and aborts,
   `results/plain/approx.json`). Each E5 op is swapped into the frozen MDLM
   (pin0), alone and together, with constants calibrated on a disjoint split:
   - norms with c per layer / per (layer, mask) / per (layer, mask, position);
   - attention (q·k)²/Z with Z per (layer, head [, query mask]), plus two
     diagnostics: row-normalised (q·k)² and exp/Z;
   - GELU deg 31 at R = 8 and 16, and deg 31/63/127 on the power-of-2
     interval that covers the calibrated input max.

   Each run reports NLL@{15,50,90}%, KL to exact, gen-PPL and the final
   residual max. A CPU smoke run (L = 64, 8 calibration seqs) already shows:
   - fixed-constant norms **blow up the residual stream** (max 1e6–3.5e9, NLL
     1e5+), and with the final norm left exact it is still NLL 11 vs 3.2;
   - deg-31 GELU diverges at R = 8 (inputs reach ~55–131); at R = 64 only
     deg 127 is near-exact;
   - (q·k)² attention alone gives NLL 8–13.

   So zero-shot conversion fails; E1 must *train* into the circuit. This run
   says which ops cost the most. Next is an E1 KD fine-tune pilot (MDLM →
   fixed norms + power attention + bounded GELU inputs). Needs a training
   budget: it shares the 4 jung_gpu cards with E5.
   - **Result** (job 1645944, `results/plain/approx.json`, full size, L =
     256, 128 calibration seqs). Exact NLL at 15 / 50 / 90% masking is
     2.15 / 3.84 / 7.06; gen-PPL 120. Final-norm input max is already 1.85e4
     in the exact model (the position-255 sink).

     | op swapped in (alone) | NLL@50% | KL@50% | gen-PPL |
     |---|---|---|---|
     | GELU deg 31, R = 8 | 9.4e3 | 9.4e3 | NaN |
     | GELU deg 31, R = 16 | 51 | 47 | 6e4 (entropy 0.03) |
     | GELU deg 31 / 63 / 127, R = 128 | 8.8 / 8.9 / 8.1 | 4.6-5.4 | 520-770 |
     | norm c per layer / mask / (mask, pos) | 2e6-1e7 | -- | 2e4-3e4 |
     | norm c per (mask, pos), final norm exact | 10.5 | 7.0 | 6 (entropy 0.72) |
     | attn (q·k)² row-normalised (ideal Z) | 8.0 | 4.6 | 4.6e3 |
     | attn (q·k)²/Z, Z per (layer, head [, mask]) | 12.5 / 10.4 | 9.0 / 6.9 | 1.5e4 |
     | full E5 circuit, GELU deg 63 / 127 | 1e4 | 1e4 | NaN |

     - **GELU: range, not degree.** Inputs reach 67, but only 0.04% exceed
       8 and 0.017% exceed 16. Deg 31 on [-8, 8] is accurate (max error
       1.3e-4) and those few outliers make it diverge. On [-128, 128] even
       deg 127 has error 0.38. The fix is bounded GELU inputs (train with a
       clamp or penalty to |x| ≤ 8), not a higher degree.
     - **Norms: per-token σ, not per position.** c varies only 2-8% over
       positions (CV), yet any fixed c blows the residual stream up to
       1e9-1e10. The massive-activation tokens have σ far from the
       population, and a fixed c cannot follow them. This is the worst op.
       Options: KD-train with fixed-scale norms; or keep an encrypted
       inverse-sqrt (Newton / polynomial, +depth) for the 25 norms; or
       remove the sink by training, so σ is tight.
     - **Attention: the shape costs most, not Z.** Even the ideal
       row-normalised (q·k)² gives NLL 8.0 vs 3.8. A public Z costs another
       2.5-4.5 nats. It has to be trained.
   - **Verdict:** every E5 op breaks the frozen MDLM by itself. E1 is a
     training project. Priority: norms > attention shape > GELU range. The
     KD pilot should introduce them one at a time in that order, measuring
     NLL after each.
2. **Remaining pin0 spikes** (`approx_probe.py spikes`, same job after approx;
   the first attempt, 1636812, stalled 6.5 h in the per-event hook and was
   cancelled -- the hook now syncs once per forward,
   `results/plain/spikes.json`). Records, per spike, the step, block, position
   and token there ([MASK] / `<|endoftext|>` / content). Then gen-PPL at
   b ∈ {clean, 11, 12, 13} with and without ever sampling `<|endoftext|>`:
   does one bootstrap pass become enough? This matters for E1 too, since the
   spikes set the norm and GELU input ranges.
   - **Result** (job 1640480). With pin0 the sink has **moved to the last
     position**: every spike above 2000 is a [MASK] at position 255, in
     blocks 5-11, up to 1.85e4 (NLL) / 1.56e4 (gen). It fires only when
     almost everything is masked: 43 events in generation, all at the
     first steps (255 of 256 masked), and 2 in the NLL pass at 90% masking.
     Never `<|endoftext|>`, never content.
   - Never sampling `<|endoftext|>` does not help. gen-PPL clean 120 / 123;
     b = 11: 156 / 176; b = 12: 164 / 141; b = 13: 140 / 117, with / without
     EOS (n = 64, one seed, ±15% noise). S_max stays at 1.5e4 because of the
     position-255 spike.
   - So one bootstrap pass is still not enough. The trigger is public (step
     index / mask count / position), so E1 can treat it: decode position 255
     early too, or give the near-all-masked first steps their own scale.
3. **E2 ordering ablation on the frozen teachers** (`scripts/e2_order.py`,
   job 1636813, `results/plain/e2_order.json`, resumable).
   - Models: BD3-LM B = 4/8/16 (SDPA sampling path), plus MDLM global as
     reference.
   - Grid: k ∈ {1, 2, 4, 8, 16} ≤ B; order ∈ {random, confidence, margin,
     entropy, l2r}; top-p 1.0 and 0.9.
   - Metrics: gen-PPL (GPT-2 Large) and unigram entropy; n = 64, L = 256.
   - **Result** (job 1636813; BD3-LM complete, MDLM top-p 0.9 hit OOM in the
     nucleus sort and resumes in job 1640364). gen-PPL, T = 1, one seed:

     | k (tokens/pass) | 1 | 2 | 4 | 8 | 16 |
     |---|---|---|---|---|---|
     | BD3 B16 random | 108 | 235 | 801 | 2523 | 5028 (all) |
     | BD3 B16 confidence | 3.8 | 7.6 | 12.5 | 145 | -- |
     | BD3 B8 random | 96 | 399 | 1740 | 3803 (all) | |
     | BD3 B4 random | 87 | 706 | 3040 (all) | | |
     | MDLM global random | 85 | 95 | 110 | 151 | 229 |
     | MDLM global confidence | 1.4 | 1.5 | 1.7 | 2.6 | 5.4 |

     With top-p 0.9 every number drops about 2.5-3× (B16 random 40 → 995 at
     k = 8), with the same shape.
   - **In-block parallel commits break BD3-LM fast.** With k = 2, gen-PPL is
     ×8 (B4), ×4 (B8) and ×2.2 (B16) of k = 1. MDLM global degrades only ×1.1
     at k = 2 and ×1.8 at k = 8: its k picks are spread over 256 positions and
     nearly independent, while BD3's k picks are neighbours.
   - **Confidence ordering wins gen-PPL by collapsing entropy.** Unigram
     entropy is 2.3-3.4 for BD3 confidence and 1.2-1.7 for MDLM confidence,
     against 4.5-5.0 for random. That is repetition, not quality.
     Margin/entropy/l2r are no better than random at k ≥ 2 (l2r is worst).
   - **So no frozen BD3-LM (B ≤ 16) configuration keeps quality at k ≥ 8 with
     healthy entropy.** C4/E4 needs either a wide live window (MDLM-style,
     whole-sequence pass cost) or an ordering that does not collapse entropy.
   - **Entropy-matched follow-up** (`e2_order.py --sweep`, job 1640480, held
     until 1640364 finishes; `results/plain/e2_entropy.json`, resumable).
     - Models and k: BD3 B16 k ∈ {2, 4, 8}, B8 k ∈ {2, 4}, MDLM global
       k ∈ {4, 8, 16}.
     - Sweeps: confidence with Gumbel choice temperature τ ∈ {0.25, 0.5, 1, 2,
       4} (log p + τ·Gumbel, MaskGIT-style); confidence at sampling T ∈ {1.25,
       1.5}; random at T ∈ {0.7, 0.8, 0.9}.
     - Plus a real-text reference: WikiText-103 validation chunks, gen-PPL
       and entropy. The models are OWT-trained, so the reference is
       approximate.
     - Read it as gen-PPL vs entropy: does any order beat random at the
       reference entropy for k ≥ 8?
   - **Result** (job 1640480). Reference text: gen-PPL 27, entropy 4.64.
     GPT-2 small samples at T = 1: 106.5 / 4.91.
     - **At matched entropy, confidence ordering ≈ random.**
       - BD3 B16, k = 4: confidence τ = 4 gives 136 @ 4.26, and random
         interpolated to 4.26 gives about 130.
       - BD3 B16, k = 8: confidence τ = 4 gives 599 @ 4.26; random T = 0.9
         gives 511 @ 4.26.
       - MDLM, k = 8 / 16: confidence τ = 4 gives 24.6 @ 4.12 / 37.4 @ 4.22;
         random T = 0.8 gives 26.7 @ 4.12 / 42.1 @ 4.19. That is a 5-10% edge.

       Confidence's low gen-PPL in the first table was entropy. Choosing
       *which* k tokens to commit is not the lever.
     - **The lever is the window width.** Near reference entropy (4.4-4.6):
       - MDLM global random T = 0.9 gives 42 / 59 / 94 at k = 4 / 8 / 16.
       - BD3 B16 gives 223 @ 4.54 at k = 4 and 511 @ 4.26 at k = 8.
       - At k = 8 that is 9× worse, at lower entropy.

       Committing k picks spread over 256 positions is nearly free; k picks
       out of 16 neighbours is not.
     - **Consequence for C4/E4.** A DLM at k ≥ 8 needs a live window much
       larger than k. Either use MDLM-style whole-sequence passes (cost ∝ L
       per pass, no KV reuse) or train BD3 with a large B (64-256) — no such
       checkpoint exists. The cost model must price tokens per pass *and*
       the cost per pass of the window size together.
     - Caveats: one seed, n = 64, L = 256; WikiText reference for
       OWT-trained models; entropy matching by interpolation between sweep
       points.
4. **Spec-AR α** (`scripts/spec_alpha.py`, job 1640364 -- 1636814 crashed on a
   CPU/GPU device mismatch in gen_ppl, fixed;
   `results/plain/spec_alpha.json`; drafts saved to `ckpt/drafts/`).
   - Target is GPT-2 small, a stand-in: the matched `kuleshov-group/owt-ar`
     (619 MB pickled `.bin`) was not downloaded -- needs the user's OK.
   - Drafts are the first 2 / 4 blocks plus the tied head, KD 3000 × 16 × 256
     on WikiText-103 train.
   - α = 1 − TV, exact, along the target's samples and along text; tokens per
     verify pass by i.i.d. formula and by per-position walk.
   - **First result** (job 1640364). T = 1, α = Σ min(p, q):

     | draft | KD KL end | α_gen | α_text | tok/pass k=4 (gen / text) | k=8 |
     |---|---|---|---|---|---|
     | 2L, step 0 | -- | 0.17 | 0.12 | | |
     | 2L, 3000 steps | 1.20 | 0.35* | 0.52 | 1.52* / 1.98 | 2.06 (text) |
     | 4L, 3000 steps | 0.89 | 0.42* | 0.59 | 1.68* / 2.24 | 2.39 (text) |

     \* α_gen is biased low: HF `generate` stopped at `<|endoftext|>` and
     padded with it (4/16 samples, ~36 pad tokens each); target gen-PPL 301
     is inflated for the same reason. `sample_target` is now a plain
     ancestral loop. The saved drafts are re-measured with `--load-dir`
     (probe `alpha2`, job 1640480 → `spec_alpha2.json`). A toy CPU run gave
     α_gen 0.44 for 2L. α_text is teacher-forced and unaffected.
   - Even α_text = 0.59 is well under the cost model's α ∈ {0.6, 0.8} (§9.3),
     so spec-AR commits ≈ 2.2-2.4 tokens per verify pass at k = 4-8, not
     3.4-4. Caveats: short KD (KL still falling), GPT-2 small rather than the
     matched OWT AR, T = 1 (α rises at lower temperature / top-p). This moves
     the C4 bar in the DLM's favour; re-price §9.3 once `alpha2` lands.
   - **Corrected (`spec_alpha2.json`, job 1640480).** Target samples are now
     gen-PPL 106.5, entropy 4.91, 0.28 EOS per sequence.
     - α_gen: 0.385 (2L), 0.445 (4L). α_text is unchanged: 0.52 / 0.59.
     - Tokens per verify pass along the target's own samples (walk):

       | draft | k = 4 | k = 8 |
       |---|---|---|
       | 2L | 1.60 | 1.62 |
       | 4L | 1.76 | 1.80 |

       Along text: 1.98-2.24 at k = 4 and 2.06-2.39 at k = 8.
     - Spec-AR at T = 1 therefore commits ≤ 2.4 tokens per encrypted pass
       with these drafts, against the 3.4-4 the cost model assumed at
       α = 0.8.
**E1 KD pilot -- design (2026-10-29, `scripts/e1_kd.py`, probe `e1pilot`;
not running: 1655431 and 1655468 (2026-10-30) were both
given qa-rtx6k-019 GPU 2F:00.0 and aborted at the CUDA check).**

The pilot asks one question: can training recover what the zero-shot swap
lost, and at what cost? It converts MDLM one op at a time, in the order the
damage ranked, and evaluates after each stage:

| stage | op (student) | steps (ramp) | public constants (init) |
|---|---|---|---|
| A | LayerNorm → c[norm, query mask]·(x − mean)·w | 3000 (1000) | c = Σσ/Σσ² per (norm, mask) |
| B | softmax → (s + b)²/Z | 3000 (1000) | b per (layer, head) from {0,1,2,4,8} by KL to softmax; Z = mean row sum per (head, query mask) |
| C | GELU → Chebyshev 31 on [-8, 8] + hinge² penalty on \|u\| > 6 | 2000 (500) | -- |
| D | + N(0, (2^-11·S)²) after every block | 1000 | S = block-output max |

- **Blending.** Each op is blended in, out = (1 − λ)·exact + λ·E5, with λ
  0 → 1 over the ramp. Earlier stages' ops stay at λ = 1.
- **Shifted square for attention.** It is (s + b)², not s²: MPCFormer's
  2Quad. The pure square up-weights very negative scores, which may be most
  of the "shape" cost in approx.json. Adding b is free in CKKS, so the depth
  is unchanged.
- **GELU training vs evaluation.** Training evaluates the polynomial on
  clamp(u, ±8). Evaluation does not clamp (CKKS cannot), so the out-of-range
  fraction it reports is the real risk.
- **Loss and masks.** The loss is KL(teacher ‖ student) on masked positions,
  with pin0 masks (ratio ~ U(0, 1), position 0 never masked, 1 row in 8
  fully masked). All weights are trained. Constants train at lr 1e-3, the
  rest at 1e-4 with AdamW, grad-clip 1, bs 16×2, L 256.
- **Data.** 9000 steps ≈ 74M tokens of WikiText-103 train. That is a domain
  shift from OWT; KD from the OWT teacher limits it, and the full E1 should
  use an OWT shard (download needs the user's OK).
- **Eval per stage.** The same eval as approx.json: NLL / KL @ 15/50/90%,
  pin0 gen-PPL + entropy, residual max, GELU input max and out-of-range
  fraction. From C on it also runs at noise b = 11 and 16.
- **Bookkeeping.** Checkpoints `ckpt/e1_pilot/stage{A..D}.pt` (680 MB each);
  the run resumes by stage. It aborts if NLL@50% after a stage exceeds 20.
  The CPU smoke test passed: all stages, resume, abort, and the NaN guard.
- **Budget.** Estimated ≈ 1.3 s/step on one RTX 6000 in fp32, so ≈ 3.5 h of
  training + ≈ 1 h of eval, i.e. one 8 h job on one card. Check s_per_step
  in the first log lines.
- **Read-out / go criteria for full E1.**
  - After C, NLL@50% within +0.3 nats of exact (3.84), and gen-PPL within
    1.5× of exact (120) at entropy ±0.2.
  - After D (b = 11), the same → one bootstrap pass suffices (≈ 1.1 s
    instead of 2.2 s per bootstrap).
  - If A alone stays > +1 nat, fixed-constant norms are not trainable at
    this budget. Fallbacks: per-layer sequential norm removal (one norm per
    ramp, as in LN-free GPT-2 fine-tuning) or an encrypted inverse sqrt for
    some norms (+depth; re-run the level audit).
- **Known risks.**
  - The smoke run's 2-step ramp diverged at λ = 1 (KL 7e4, grad-norm 3e8):
    the switch has to be gradual, and the abort rule is there for that.
  - The position-255 sink must be trained away for D to pass. Watch
    resid_max.
  - The final norm cannot stay exact unless the client applies the LM head,
    which would hand it the (tied) embedding.

5. Then: estimate GPU-hours for E1 / E3 / AR control against the Jan 5 ARR
   date. jung_gpu is the only GPU pool, and the user's other jobs use it.

Demoted (not blocking): fold two-pass into the cost model (bootstrap ≈ 2.23 s
on 4 GPUs, 20 usable; ≈ +12 s/pass); make the `fides_e0` error loops
NaN-aware and find the 4096-slot GPU anomaly (E5 uses full slots).
