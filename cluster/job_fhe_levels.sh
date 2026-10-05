#!/bin/bash
#$ -N fhedlm_fhe_levels
#$ -j y
#$ -q long
#$ -pe smp 16
#$ -M jzhao7@nd.edu
#$ -m abe
#$ -l h_rt=4:00:00
#
# E0 -- usable CKKS levels between bootstraps at N=2^16/2^17. CPU only.
# ~1 h on 20 tjws cores; live 2^17 bootstraps need up to 48 GB per child.
# -M/-m are file directives, always as a pair (see cluster/README.md).
#   qsub cluster/job_fhe_levels.sh
source ~/.bashrc
conda activate fhedlm
set -uo pipefail
export FHEDLM_ROOT=/groups/tjung/jzhao7/fhe-dlm
cd "$FHEDLM_ROOT"
source cluster/job_common.sh
export FHEDLM_CONDA_ENV="$HOME/.conda/envs/fhedlm"
source cluster/env.sh
export OMP_NUM_THREADS=${NSLOTS:-16} MKL_NUM_THREADS=${NSLOTS:-16}
# Chebyshev fitter reused read-only from the sibling FHE-S4-norm project.
export PYTHONPATH="/groups/tjung/jzhao7/FHE-S4-norm/mamba2-poly-exp:${PYTHONPATH:-}"
cd "$FHEDLM_ROOT/third_party/ELF"
"$PY" "$FHEDLM_ROOT/scripts/openfhe_levels.py" "$FHEDLM_ROOT/results/openfhe_levels.json"
echo "[fhe] exit=$? at $(date)"
