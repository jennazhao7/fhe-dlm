#!/bin/bash
#$ -N fhedlm_iterboot
#$ -j y
#$ -q long
#$ -pe smp 24
#$ -M jzhao7@nd.edu
#$ -m abe
#$ -l h_rt=3:00:00
#
# E0 -- iterative (multi-pass) CKKS bootstrapping at the E5 parameter set
# (N=2^17, [4,4], 59/60): bits gained, levels and time per extra iteration.
# CPU only, on purpose: FIDESlib's GPU EvalBootstrap accepts numIterations but
# silently drops it (FIDESlib/api/CryptoContext.cpp, GPU path calls
# CKKS::Bootstrap(ct, slots, prescaled)). OpenFHE's CPU implementation is the
# reference the GPU version will be validated against.
#
# 24 slots = half a 48-core/125 GB long node. long does not enforce memory
# (h_vmem INFINITY); a full-slot 2^17 key set is ~39 GB, and the script caps
# each child at 60 GB, so half a node keeps the memory share proportionate.
# -M/-m are file directives, always as a pair (see cluster/README.md).
#   qsub cluster/job_iterboot.sh
source ~/.bashrc
conda activate /groups/tjung/jzhao7/conda-envs/fhedlm
set -uo pipefail
export FHEDLM_ROOT=/groups/tjung/jzhao7/fhe-dlm
cd "$FHEDLM_ROOT"
source cluster/job_common.sh
export FHEDLM_CONDA_ENV="/groups/tjung/jzhao7/conda-envs/fhedlm"
source cluster/env.sh
export OMP_NUM_THREADS=${NSLOTS:-24} MKL_NUM_THREADS=${NSLOTS:-24}
export PYTHONNOUSERSITE=1
# Overrides via qsub -v, e.g. the FIDESlib-op validation run:
#   qsub -v ITERBOOT_ARGS="--manual",ITERBOOT_OUT=results/e0_iterboot_manual.json cluster/job_iterboot.sh
OUT="$FHEDLM_ROOT/${ITERBOOT_OUT:-results/e0_iterboot.json}"
"$PY" "$FHEDLM_ROOT/scripts/e0_iterboot.py" "$OUT" ${ITERBOOT_ARGS:-}
echo "[iterboot] exit=$? at $(date)"
