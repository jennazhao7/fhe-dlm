#!/bin/bash
#$ -N fhedlm_s3_rounds
#$ -j y
#$ -q long
#$ -pe smp 16
#$ -M jzhao7@nd.edu
#$ -m abe
#$ -l h_rt=6:00:00
#
# STEP 3 -- quality vs diffusion rounds, K in {8,4}. CPU only.
# -M/-m are file directives, always as a pair (see cluster/README.md).
#   qsub cluster/job_step3_rounds.sh
source ~/.bashrc
conda activate /groups/tjung/jzhao7/conda-envs/fhedlm
set -uo pipefail
export FHEDLM_ROOT=/groups/tjung/jzhao7/fhe-dlm
cd "$FHEDLM_ROOT"
source cluster/job_common.sh
export FHEDLM_CONDA_ENV="/groups/tjung/jzhao7/conda-envs/fhedlm"
source cluster/env.sh
export OMP_NUM_THREADS=${NSLOTS:-16} MKL_NUM_THREADS=${NSLOTS:-16}
cd "$FHEDLM_ROOT/third_party/ELF"
"$PY" "$FHEDLM_ROOT/scripts/step3_rounds.py" --ks 8 4 --n-samples 64 --grid-seeds 0 1 2 3 \
      --out "$FHEDLM_ROOT/results/step3_rounds_n64.json"
echo "[step3] exit=$? at $(date)"
