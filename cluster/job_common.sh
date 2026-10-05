#!/bin/bash
# Shared CRC/SGE settings for fhe-dlm.
#
#   queue  gpu@@jung_gpu   the Jung group's hostgroup (qa-rtx6k-019, 4x Quadro
#                          RTX 6000, 24 GB, Turing sm_75). Chosen by the user
#                          over the idle general-queue A10s.
#   gpu=1  nothing in this gate needs more than one card. The node is shared
#          with the user's own bfl_gate and fhe_norm_pathA jobs -- do not grab
#          cards speculatively.
#   email  `#$ -M jzhao7@nd.edu` + `#$ -m abe` as file directives, always as a
#          pair (the wrapper rejects one without the other). See cluster/README.md.
#
# SGE only reads `#$` directives from the script it was handed, so they are
# duplicated at the top of each job script on purpose.

set -euo pipefail
export FHEDLM_QUEUE="gpu@@jung_gpu"

_JOB_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export FHEDLM_ROOT="${FHEDLM_ROOT:-$(dirname "$_JOB_COMMON_DIR")}"

echo "=============================================================="
echo "host      : $(hostname)"
echo "job       : ${JOB_NAME:-interactive} (${JOB_ID:-n/a})"
echo "queue     : ${QUEUE:-n/a}"
echo "started   : $(date)"
echo "cwd       : $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "=============================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed -- did this land on a GPU node?"
echo "=============================================================="
