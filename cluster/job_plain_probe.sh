#!/bin/bash
#$ -N fhedlm_plain_probe
#$ -j y
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 8
#$ -M jzhao7@nd.edu
#$ -m abe
#$ -l h_rt=4:00:00
#
# Plaintext probes on MDLM-OWT (scripts/plain_probe.py), one GPU:
#   ranges  E1 activation ranges by mask ratio  -> results/plain/ranges.json
#   noise   bootstrap-noise tolerance at 2^-b   -> results/plain/noise.json
# Checkpoints/data are prefetched into hf-cache on the front end; compute nodes
# have no network, so everything runs offline.
#
#   qstat -u jzhao7 ; qsub cluster/job_plain_probe.sh
#   qsub -v PROBES="noise" cluster/job_plain_probe.sh      # just one
source ~/.bashrc
set -uo pipefail
export FHEDLM_ROOT=/groups/tjung/jzhao7/fhe-dlm
cd "$FHEDLM_ROOT"
source cluster/job_common.sh
set +e
PY=/groups/tjung/jzhao7/conda-envs/fhedlm/bin/python
export HF_HOME="$FHEDLM_ROOT/hf-cache" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=${NSLOTS:-8}
mkdir -p results/plain results/logs
for p in ${PROBES:-ranges noise}; do
  echo "=== $p ($(date))"
  case $p in
    ranges) "$PY" scripts/plain_probe.py ranges --out results/plain/ranges.json ;;
    noise)  "$PY" scripts/plain_probe.py noise --gen --out results/plain/noise.json ;;
  esac
  echo "=== $p exit $? ($(date))"
done
