#!/bin/bash
#$ -N fhedlm_s4_poly
#$ -j y
#$ -q long
#$ -pe smp 16
#$ -M jzhao7@nd.edu
#$ -m abe
#$ -l h_rt=6:00:00
#
# STEP 4 -- zero-shot norm polynomialization at K=8. CPU only.
# -M/-m are file directives, always as a pair (see cluster/README.md).
#   qsub cluster/job_step4_poly.sh
source ~/.bashrc
conda activate /groups/tjung/jzhao7/conda-envs/fhedlm
set -uo pipefail
export FHEDLM_ROOT=/groups/tjung/jzhao7/fhe-dlm
cd "$FHEDLM_ROOT"
source cluster/job_common.sh
export FHEDLM_CONDA_ENV="/groups/tjung/jzhao7/conda-envs/fhedlm"
source cluster/env.sh
export OMP_NUM_THREADS=${NSLOTS:-16} MKL_NUM_THREADS=${NSLOTS:-16}
# Chebyshev fitter reused read-only from the sibling FHE-S4-norm project.
export PYTHONPATH="/groups/tjung/jzhao7/FHE-S4-norm/mamba2-poly-exp:${PYTHONPATH:-}"
cd "$FHEDLM_ROOT/third_party/ELF"
"$PY" "$FHEDLM_ROOT/scripts/step4_poly.py" --k 8 --n-samples 64 --n-calib 64 \
            --out "$FHEDLM_ROOT/results/step4_poly_n64.json"
echo "[step4] exit=$? at $(date)"
