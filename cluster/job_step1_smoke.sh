#!/bin/bash
#$ -N fhedlm_s1_smoke
#$ -j y
#$ -q long
#$ -pe smp 16
#$ -M jzhao7@nd.edu
#$ -m abe
#$ -l h_rt=8:00:00
#
# STEP 1 -- CPU smoke test. No GPU: the user's cards are booked by gsm-reward
# and the revised scope is CPU-only.
#
#   qsub cluster/job_step1_smoke.sh
#
# -M/-m are file directives, always as a pair: CRC's qsub wrapper rejects one
# without the other, and a command-line -M does not satisfy that check.
#
# 16 cores on the `long` queue (24-slot nodes, smp PE, 15k slots free). fp32.
# Everything is prefetched -- this job never reaches the network.

source ~/.bashrc
conda activate /groups/tjung/jzhao7/conda-envs/fhedlm
set -uo pipefail

export FHEDLM_ROOT=/groups/tjung/jzhao7/fhe-dlm
cd "$FHEDLM_ROOT"
source cluster/job_common.sh
export FHEDLM_CONDA_ENV="/groups/tjung/jzhao7/conda-envs/fhedlm"
source cluster/env.sh

CKPT=$(ls "$FHEDLM_ROOT"/hf-cache/hub/models--embedded-language-flows--ELF-B-owt-torch/snapshots/*/checkpoint_95085 2>/dev/null | head -1)
CONF=$(ls "$FHEDLM_ROOT"/hf-cache/hub/models--embedded-language-flows--ELF-B-owt-torch/snapshots/*/config.yml 2>/dev/null | head -1)
[ -f "$CKPT" ] || { echo "FATAL: no checkpoint at $CKPT"; exit 1; }
echo "[step1] ckpt: $CKPT"
echo "[step1] conf: $CONF"

# Keep BLAS threads and the SGE slot count in agreement; oversubscribing a
# shared node is how a CPU job gets 10x slower than its own benchmark.
export OMP_NUM_THREADS=${NSLOTS:-16}
export MKL_NUM_THREADS=${NSLOTS:-16}
echo "[step1] NSLOTS=${NSLOTS:-unset} OMP_NUM_THREADS=$OMP_NUM_THREADS"

# cwd must be the ELF repo: the checkpoint's config.yml carries a RELATIVE
# sampling_configs_path that only resolves from there.
cd "$FHEDLM_ROOT/third_party/ELF"

"$PY" "$FHEDLM_ROOT/scripts/smoke_test.py" \
    --config "$CONF" \
    --ckpt "$CKPT" \
    --n-samples 8 \
    --steps 32 \
    --threads "${NSLOTS:-16}" \
    --out "$FHEDLM_ROOT/results/step1_smoke.json"

echo "[step1] exit=$? at $(date)"
