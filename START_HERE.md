# FHE-DLM: start here (agent handoff, 2026-11-26)

Read this file first. `HANDOFF.md` holds the full history, with every job, number and dead end since
2026-10-02. This file covers the current state: where everything lives, how to reach it, the rules
the user has set, and what is left to do.

---

## 1. The project in one paragraph

Encrypted (CKKS) inference for a masked diffusion language model. The model is MDLM-OWT
(`kuleshov-group/mdlm-no_flashattn-fp32-owt`): about 110M parameters, d = 768, 12 layers, GPT-2
tokenizer. A server computes a denoising pass on an encrypted sequence, and the client decrypts the
logits and commits tokens. The client is allowed in the loop between passes. The paper claim is that
a DLM needs fewer rounds and less wall-clock per committed token than an HE-friendly autoregressive
model of matched quality. The target is **ACL 2027 (January ARR cycle, plan for about Jan 5, 2027)**.
The experiment plan (E0 to E7, claims C1 to C8) lives with the user, not in this repo.

**Status.** Full encrypted generation runs end to end at L = 64 tokens, with **no client step inside
a pass**. The client encrypts the tokens and decrypts the logits; the server, on 4 GPUs, runs the
embedding, every matrix product, exact LayerNorm, squared attention, GELU and 100 bootstraps per
pass. A 16-pass, 50%-infill generation (job 1770029) takes 703 s per pass plus a 155 s restart,
3 h 50 min in all. Its sample is **token-for-token the plaintext student's**: logit error ≤ 0.1, KL
≤ 3e-5.

---

## 2. Machines: what lives where

There are four places. Code is edited on the Mac, kept in GitHub, copied to CRC and run there.

| Place | Address / path | What lives there | What to do there |
|---|---|---|---|
| **Local Mac** (source of truth) | `/Users/jiachenzhao/research/fhe-dlm` | git checkout of all code, docs, result JSONs; `logs/pids.txt` (untracked) | edit code, read results, commit, push |
| **GitHub** | https://github.com/jennazhao7/fhe-dlm (private) | `main` and `claude/jolly-ride-7k66ct`, kept identical (fast-forward `main` to the branch after each push) | push only |
| **CRC front end** | `crcfe01.crc.nd.edu`, user `jzhao7`, project `/groups/tjung/jzhao7/fhe-dlm` (`~/fhe-dlm` on CRC symlinks to it) | a **plain copy** of the code (not a git clone), plus the CRC-only checkpoints, model cache, GPU libraries, builds, job logs and job output | build the GPU binaries, `qsub`/`qstat`, read logs; **no GPU, no `libcuda`** |
| **CRC GPU node** | `qa-rtx6k-019`, SGE queue `gpu@@jung_gpu`, 4× Quadro RTX 6000 24 GB (sm_75) | nothing persistent; jobs run in the project directory on the shared group filesystem | reached only through `qsub` |
| **tjws-03** (CPU workstation) | `tjws-03.cse.nd.edu` (not `tjws.cse.nd.edu`), `~/fhe-dlm`, ssh alias `tjws` | E0 CPU microbenchmarks from October (openfhe-python 1.5.1 env `fhedlm`, Python 3.10, no torch); Xeon 20 cores, 125 GB RAM, RTX 5000 16 GB | **not used since October**; only for CPU OpenFHE benchmarks. `/groups` is not mounted there |

Only on CRC, and not in git:
- `ckpt/`: the student checkpoints (§6).
- `hf-cache/`: Hugging Face models and data. Compute nodes have no network, so jobs run with
  `HF_HUB_OFFLINE=1`.
- `third_party/`: `fideslib` (1-GPU build), `fideslib-mgpu` (NCCL multi-GPU build), `openfhe-fides`,
  `nccl-cu13`, `FIDESlib` source, `ELF`.
- `gpu/fides_e0/build/` (1-GPU binary) and `gpu/fides_e0/build-mgpu/` (multi-GPU binary).
- `fhedlm_*.o<jobid>`: job logs, in the project root.
- `scratch/`: build logs, temporary files.

Not part of this project: `/Users/jiachenzhao/Desktop/overleaf-project2` is the user's DP-GRPO paper.
No FHE-DLM paper draft exists in any of these places.

---

## 3. Access: how to reach CRC (and tjws)

**The user owns the SSH master connection.** Never open, close, `-O exit` or otherwise touch it.
When it is down, ask the user to run this in their own terminal (it needs their password; off campus,
see below):

```bash
ssh -fN -o ControlMaster=yes -o ControlPath=~/.ssh/cm/%r@%h:%p -o ControlPersist=8h jzhao7@crcfe01.crc.nd.edu
```

Check it with `ssh -O check -S ~/.ssh/cm/jzhao7@crcfe01.crc.nd.edu:22 jzhao7@crcfe01.crc.nd.edu`.
Older sessions used the sockets `/tmp/crc-dp-grpo.sock` or `~/.ssh/crc-dp-grpo.sock`; check those too
if `~/.ssh/cm/...` is missing. Off campus without VPN, `crcfe01` does not resolve: the user adds
`-o ProxyCommand="ssh -W %h:%p tjws"`, which needs the tjws master up first.

**Command template.** macOS has no `timeout`, hence `perl alarm`. Every remote command prints
`Loading CRC_default/1.1`; filter it out.

```bash
S=<your scratch dir>
perl -e 'alarm shift; exec @ARGV' 60 ssh -S ~/.ssh/cm/jzhao7@crcfe01.crc.nd.edu:22 -o BatchMode=yes \
  jzhao7@crcfe01.crc.nd.edu 'cd /groups/tjung/jzhao7/fhe-dlm && qstat -u jzhao7' > $S/out.txt 2>&1 < /dev/null
grep -v CRC_default $S/out.txt
```

- **Always redirect the ssh output to a file, then read the file.** Never pipe ssh straight into
  `grep` or `tail`: the shared master keeps the pipe open after `alarm` fires, and the tool call hangs
  in the background.
- **To wait for a job, do it on CRC inside one foreground ssh:**
  `timeout 520 bash -c "until grep -q '=== e5 exit' fhedlm_e5.o<id>; do sleep 5; done"`, with the
  `perl alarm` set a little higher and the tool timeout under 600 s. Background `sleep` timers in the
  Claude harness proved unreliable on 2026-11-25: they reported done after seconds.
- **Copying code to CRC.** There is no git on that side, so stream the files and check checksums:
  ```bash
  F=(scripts/e5_circuit.py gpu/fides_e0/src/fides_vm.cpp)       # zsh: use an array, then ${F[*]}
  COPYFILE_DISABLE=1 tar cf $S/up.tar $F
  perl -e 'alarm shift; exec @ARGV' 60 ssh -S ... jzhao7@crcfe01.crc.nd.edu \
    "cd /groups/tjung/jzhao7/fhe-dlm && tar xf - && md5sum ${F[*]}" < $S/up.tar > $S/up.txt 2>&1
  ```
  Compare against `md5 -r $F` locally. For a single file:
  `ssh ... 'cat > f.new && mv f.new f && md5sum f' < f`.
- **Before editing a file on CRC, compare checksums.** On 2026-11-20 another session edited
  `gpu_encode.cu` and `HANDOFF.md` on both sides without committing.
- **tjws:** `ssh tjws '…'` works once the user has logged in (`ssh tjws`; master in `~/.ssh/cm/`, 8 h).

---

## 4. Standing rules from the user (verbatim where quoted; they still apply)

1. "The SSH control socket is the user's and is shared with their own sessions. **One ssh call at a
   time, foreground only, always wrapped in a timeout.** Never background ssh, scp, or rsync."
2. "**Never signal a process you did not start.** Append the PID of every process you launch to
   `logs/pids.txt`; only those PIDs may be killed. If anything hangs, stop and report. Do not clean up
   other processes."
3. "Never touch the master connection."
4. "No du/find/recursive listing anywhere under /groups except /groups/tjung/jzhao7/fhe-dlm."
5. "All work lives under /groups/tjung/jzhao7/fhe-dlm. Every HF download uses `allow_patterns` (torch
   weights, config, tokenizer only); state the expected size first."
6. "Other projects' directories and jobs are read-only, for code reuse only. Never write there; never
   submit, modify, or delete their jobs." The user runs other jobs on the same node (`fhe_norm_*`,
   `vla_influence`, `bfl_gate`, …): leave them alone.
7. Jobs: run `qstat -u jzhao7` **before** every `qsub`. Every job file carries
   `#$ -q gpu@@jung_gpu`, `#$ -l gpu=N`, `#$ -pe smp N`, and the pair `#$ -M jzhao7@nd.edu` +
   `#$ -m abe` (CRC's wrapper rejects one without the other). Use **only `gpu@@jung_gpu`**. Ask the
   user before using the general `gpu@@crc_rtx6k` queue (a 2026-11-20 session did, without a
   record). Don't hold cards speculatively.
8. Keep the user's email (in the job directives) out of anything sent outside CRC. Don't edit the
   user's `~/.bashrc` on CRC; they fixed it on 2026-11-17 so non-interactive shells skip conda.
9. Commit and push only when the user asks; they have asked to keep GitHub current. End commit
   messages with the attribution line the harness specifies.

---

## 5. CRC environment and builds

- **Python:** conda env by path, `/groups/tjung/jzhao7/conda-envs/fhedlm/bin/python` (Python 3.10,
  torch 2.5.1+cu121, transformers 4.44.2, openfhe-python 1.5.1). Non-interactive shells don't load
  conda: export `PATH=/groups/tjung/jzhao7/conda-envs/fhedlm/bin:$PATH`. For installs, set
  `CONDA_PKGS_DIRS=/groups/tjung/jzhao7/conda-pkgs PIP_CACHE_DIR=/groups/tjung/jzhao7/pip-cache`.
  Use `PYTHONNOUSERSITE=1`.
- **GPU library:** FIDESlib 2.1.3, built by `cluster/setup_fideslib.sh` (1 GPU) and
  `cluster/setup_fideslib_mgpu.sh` (NCCL, 4 GPUs) into `third_party/`. Already built; don't rebuild
  unless it's broken.
- **Building the server `fides_vm`** (about 1 min on the front end):
  ```bash
  cd /groups/tjung/jzhao7/fhe-dlm
  export PATH=/groups/tjung/jzhao7/conda-envs/fhedlm/bin:$PATH; module load cuda/13.2.1
  cmake --build gpu/fides_e0/build-mgpu --target fides_vm -j4 > scratch/build_build-mgpu.log 2>&1   # 4-GPU binary
  cmake --build gpu/fides_e0/build      --target fides_vm -j4 > scratch/build_build.log 2>&1        # 1-GPU binary
  ```
  **Rebuild both** after any change in `gpu/fides_e0/src/`. On 2026-11-18 the front end stalled for
  hours in CMake's depend step. If a build takes more than about 4 min, run it detached:
  `setsid nohup bash -c "…" > scratch/x.log 2>&1 < /dev/null &`. Put the `cd` inside the quoted
  command; then record the PID in `logs/pids.txt` on both sides.
- **The front end has no `libcuda.so.1`**, so GPU binaries don't start there. A stub in
  `scratch/stubcuda/` allows CPU-only toy runs (`FIDES_E0_CPU=1`).
- **CRC keeps no job history** (`qacct` has no accounting file): a job's only trace is its
  `fhedlm_*.o<id>` file. SGE copies the job script at `qsub`, but Python scripts are read when the
  job starts, so don't edit them while a job is queued.

### GPU node health (as of 2026-11-25)

| Card (PCI bus) | State |
|---|---|
| `2F:00.0` | works |
| `86:00.0` | works |
| `06:00.0` | **hangs at CUDA init** since at least 2026-11-20 |
| `D8:00.0` | **hangs at CUDA init** since at least 2026-11-20 |

With 2 cards, only 1-GPU runs work. The fully encrypted circuit (depth 43, bootstraps) needs about
22 GB per card on 4 cards; it has fit on 3, never on 2. The user has been asked to report this to
CRC: "GPUs 06:00.0 and D8:00.0 hang at CUDA init, reproducible in jobs 1819824 and 1819905." Recheck
any time with `qsub cluster/job_gpu_check.sh`: it requests all 4 cards and tests them in parallel
with a hard kill. `cluster/job_e5_pass.sh` skips the bad cards by bus ID (`E5_SKIP_BUS`, default
`"06:00.0 D8:00.0"`) and checks the rest in parallel, so **submit it with `-l gpu=4`**. Once the cards
are repaired, pass `E5_SKIP_BUS=none` (any value that matches no bus). Card hangs have come and gone
before (HANDOFF §11 and §12).

---

## 6. Code and models

| Path | What it is |
|---|---|
| `scripts/e5_circuit.py` | **Main entry point.** The deep encrypted circuit (one MDLM pass with norms, attention, GELU and bootstraps on the server), the client driver, and a numpy backend that runs the same op list exactly. Subcommands: `check` (numpy vs torch), `calib` (public variance and range bounds per layer → `results/e5/var_<tag>.json`), `run` (encrypted generation via `fides_vm`). The docstring explains the layout and every op. |
| `scripts/e5_pass.py` | Older client: `linear` (only the linear layers encrypted) and `selftest` (random matvecs vs numpy, run at the start of every 1-GPU job). |
| `gpu/fides_e0/src/fides_vm.cpp` | **The encrypted server:** an op interpreter over FIDESlib (stdin JSON commands, vectors as raw files). Ops: matvec (BSGS, GPU-encoded diagonals), add/sub/mul/sq/rot/mulpt/cheb/boot/tolevel/…, `mem`, profiling. Env: `FIDES_VM_BATCH=K`, `FIDES_VM_SPARSE=1`, `FIDES_VM_N1`, `FIDES_E0_GPUS`. |
| `gpu/fides_e0/src/gpu_encode.{cu,hpp}` | GPU encoding of the period-D weight-diagonal plaintexts (full path: scatter + NTT; batched; sparse: a 2D-point NTT + gather with a bit-exact self-check at startup). |
| `gpu/fides_e0/src/fides_e0.cpp` | E0 microbenchmarks (bootstrap, iterated two-pass bootstrap `iterboot`, matvec). |
| `cluster/job_e5_pass.sh` | **The job for all encrypted runs.** Env vars are documented in its header: `E5_SCRIPT`, `E5_ARGS`, `E5_TAG`, `E5_STUDENT`, `E5_GPUS`, `E5_CALIB`, `E5_SELFTEST`, `E5_SKIP_BUS`. |
| `cluster/job_plain_probe.sh` | Plaintext jobs, one GPU: E1 training (`PROBES=e1bc64` / `e1bc256` → `scripts/e1_kd.py`), noise probes, E2 ordering, spec-AR α. The probe list is in its header. |
| `cluster/job_gpu_check.sh` | Card health check (§5). |
| `scripts/e1_kd.py` | E1: distillation of MDLM into the HE-friendly student (stages A: fixed norm, abandoned; B: squared attention; C: polynomial GELU, failed; D: noise). |
| `scripts/plain_probe.py`, `approx_probe.py`, `e2_order.py`, `spec_alpha.py`, `probe_sink_pos.py` | Plaintext studies (noise tolerance, the position-0 attention sink, E2 ordering, spec-AR acceptance). |
| `scripts/e0_*.py`, `openfhe_levels.py` | E0 cost model and CPU microbenchmarks (tjws). |
| `src/fhedlm/`, `step3_rounds.py`, `step4_poly.py`, `smoke_test.py` | Early ELF work, unused. |

**Models (on CRC under `ckpt/`):**
- Teacher: `kuleshov-group/mdlm-no_flashattn-fp32-owt` in `hf-cache/`.
- **`ckpt/e1_bc64/stageB.pt`**, the student used for every encrypted run: exact LayerNorm, squared
  attention `(q·k/8 + b)²/z`, original GELU, trained for L = 64. Selected with
  `E5_STUDENT=$PWD/ckpt/e1_bc64`; output tags get the suffix `_stageB`.
- `ckpt/e1_bc256/stageB.pt`: the same for L = 256, trained and evaluated in plaintext (NLL close to
  the teacher's) but **never run encrypted**.
- Stage C (polynomial GELU) failed at both lengths: about 2% of GELU inputs fall outside the fitted
  range. The job script marks it aborted, and `--gelu server` replaces it without retraining.

**Results.** All JSONs are in `results/` on the Mac and in git; job logs stay on CRC.
`results/e5/<tag>_stageB.json` holds per-pass wall time, logit error, argmax agreement, KL, per-op
seconds (`vm_seconds`, `prof_s` by op@level) and memory. `var_<tag>_stageB.json` holds the
calibration bounds.

---

## 7. Canonical runs and how to reproduce them

| Run | Job | Result file | Per pass | Notes |
|---|---|---|---|---|
| Fully encrypted, 16 passes, 4 GPUs (**best**) | 1770029 | `results/e5/fast16_stageB.json` | 703 s + 155 s restart | sample = plaintext; 22.0 GB/card |
| Same, 1 pass, before speedups (profile) | 1765111 | `prof3_stageB.json` | 953 s | per-op breakdown |
| 1 GPU, client norm/GELU/refresh, sparse encode | 1819922 | `sparse1g_stageB.json` | 453 s | baseline `batchenc1_stageB.json` 535 s |
| 1 GPU, client norm/GELU/refresh, 16 passes | 1747841 | `deep_B64.json` | 520 s | |
| Linear layers only, 1 GPU, L = 64 / 256 | 1747094 / 1747331 | `linear_L64.json` / `linear_L256.json` | 195 s / 636 s | first encrypted generation |

Run all of these **on the CRC front end** from `/groups/tjung/jzhao7/fhe-dlm`, after `qstat -u jzhao7`.

Fully encrypted, 4 GPUs. Blocked until the two bad cards are fixed; add `E5_SKIP_BUS=none` then:
```bash
qsub -l gpu=4 -l h_rt=5:00:00 -v E5_GPUS=4,E5_CALIB=1,E5_SCRIPT=e5_circuit.py,E5_TAG=<tag>,E5_STUDENT=$PWD/ckpt/e1_bc64,E5_ARGS="run --norm rsqrt --refresh boot2 --depth 43 --n1 16 --gelu server --rotsum log --zfold --batch-enc 4 --infill 0.5 --steps 16 --restart-vm" cluster/job_e5_pass.sh
```
(`--max-passes 1` for a single pass. Adding `--sparse-enc` should be safe and saves about 2 GB per
card, but hasn't been tried on 4 GPUs. With sparse encode, `--batch-enc 16` is fine: the batch costs
no scratch memory.)

1 GPU, works now:
```bash
qsub -l gpu=4 -l h_rt=1:30:00 -v FIDES_VM_SPARSE=1,E5_SCRIPT=e5_circuit.py,E5_TAG=<tag>,E5_STUDENT=$PWD/ckpt/e1_bc64,E5_ARGS="run --n1 32 --depth 13 --refresh client --norm client --gelu client --batch-enc 16 --sparse-enc --infill 0.5 --steps 16 --max-passes 1" cluster/job_e5_pass.sh
```

The log is `fhedlm_e5.o<id>` in the project root. Look for `=== e5 exit 0`, a
`[e5c] {"passes": …}` summary line, and `gpu_encode: sparse encode level N: … 0 of … residues
differ`. Copy the result JSON to the Mac with `cat` over ssh into a scratch file, strip the
`Loading CRC_default` line, and save it under `results/e5/`.

Quick checks without a GPU, on the front end or the Mac with torch: `e5_circuit.py check` runs the
numpy circuit against torch. A dry run on the CPU toy ring needs the libcuda stub.

---

## 8. What stands between us and a working FHE-DLM model (priority order)

1. **Hardware.** Two of the four cards hang (§5). Nothing fully encrypted can run until CRC fixes
   them. The user handles this; recheck with `cluster/job_gpu_check.sh`.
2. **L = 256** (the real target; at L = 64 even the plaintext model's unconditional samples are
   degenerate, so every run is 50% infill). The circuit holds 64 tokens per ciphertext (slot =
   token·1024 + dim). L = 256 needs **attention across 4 ciphertexts**: each query ciphertext attends
   to keys in all 4, with rotations across ciphertext boundaries, 16× the attention loop and 4× the
   matvecs. The estimate is about 1 h per pass on 4 GPUs. Build it in the numpy backend of
   `e5_circuit.py` first, check it against torch with `check`, then on GPU. The L = 256 student
   `ckpt/e1_bc256/stageB.pt` exists, and `--gelu server` / `--norm rsqrt` should carry over after a
   new `calib`.
3. **Speed.** In the 703 s pass, about 370 s goes to matvecs, 113 s to bootstraps and about 95 s to
   rotations. The sparse
   encode showed that the remaining matvec time is the ct×pt products and the cross-card syncs, not
   the encoding. Options:
   - fewer diagonals: about 70k of 450k are already skipped, the head plus embedding are 100 of 220
     blocks, and `matvec@40` costs 147 s;
   - lower levels for the matvecs (13 to 15 towers where 9 to 11 would do);
   - fusing encode and multiply in one kernel;
   - **removing the 155 s per-pass restart.** FIDESlib's memory pool only grows, so each pass
     currently starts a fresh `fides_vm` with new keys.
4. **A real client/server split.** `fides_vm` holds the secret key: client and server are one
   process. The paper needs either a split (the client process holds the secret key; the server gets
   only evaluation keys) or an explicit argument for why the single process is equivalent.
5. **Quality at L = 256 under encryption:** gen-PPL and infill quality of the encrypted student vs
   the teacher. The L = 64 stage-B student has gen-PPL 45.7 vs the teacher's 17.8 (both degenerate at
   that length).
6. **Paper experiments built on this:** E4 (accuracy/cost Pareto against spec-AR; HANDOFF §9 and §11
   say E4 decides the paper, because spec-AR beats the DLM at k ≤ 4), E6 (leakage attack) and E7
   (network/2PC table) have not run on the encrypted pipeline. E2 (ordering) and the spec-AR
   α measurements exist in `results/plain/`.

---

## 9. Gotchas (each cost real time)

- **FIDESlib's GPU memory pool never shrinks** → pass 2 runs out of memory → `--restart-vm` (a fresh
  VM and new keys every pass, about 150 s). The 4-GPU run sits at 22.0 of 24 GB per card: keep 13
  rotation keys or fewer, `--n1 16`, and free registers early.
- **FIDESlib's GPU `EvalBootstrap` silently ignores `numIterations`.** The two-pass bootstrap (16.4
  bits instead of 8.5) is composed by hand (`boot2`).
- FIDESlib's CPU fallback of `EvalMult(ct, pt)` throws `bad any_cast`; `fides_vm` calls OpenFHE
  directly in that path.
- **The MDLM position-0 attention sink** (|x| up to about 2e4) forms when position 0 holds [MASK] or
  `<|endoftext|>` amid content. Fix: commit a content token at position 0 first. The residual stream
  grows about 400× across layers, so each norm uses a public scale from `calib`.
- The bootstrap error is absolute relative to the scale S, so one huge activation sets the precision
  for everything.
- `-M`/`-m` must be job-file directives and must come as a pair. SGE can't choose cards (hence the
  bus-ID skip).
- A job stuck in the driver ignores `timeout` (SIGTERM). Use `timeout -s KILL` and never `wait` on
  such a process.
- zsh doesn't word-split `$VAR`: use arrays.
- `HANDOFF.md` §1 ("TL;DR") is the October E0 summary and partly stale; §12 is the E5 log.
- The previous agent's private notes (the Claude memory directory on the Mac) are not available to
  you. Everything durable from them is in this file or in `HANDOFF.md`.

---

## 10. Where the history is (`HANDOFF.md`)

| Section | Topic |
|---|---|
| §1–§7 | E0, Oct 2–5: N = 2^17 / [4,4] choice, cost model, tjws benchmarks, gotchas |
| §8–§9 | Layer audit, token-group packing |
| §10 | First GPU runs on CRC, GPU diagonal encoding, 4-GPU bootstrap |
| §11 | Two-pass bootstrap, noise probes, the position-0 sink, E1/E2 plans and results |
| §12 | E5: the encrypted pipeline from 2026-11-13 on, all runs, speedups, sparse encode, card failures |
