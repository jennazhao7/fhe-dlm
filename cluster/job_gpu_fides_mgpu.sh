#!/bin/bash
#$ -N fhedlm_gpu_fides_mgpu
#$ -j y
#$ -q gpu@@jung_gpu
#$ -l gpu=4
#$ -pe smp 16
#$ -M jzhao7@nd.edu
#$ -m abe
#$ -l h_rt=4:00:00
#
# E0 bootstrap on ALL FOUR RTX 6000s of qa-rtx6k-019 (HANDOFF §10): one card
# cannot hold the N=2^17, depth 43, [4,4] bootstrap, so FIDESlib (NCCL build,
# cluster/setup_fideslib_mgpu.sh) splits limbs and keys over 4 devices.
# This takes the whole jung_gpu node -- check `qstat -u jzhao7` and
# `free_gpus.sh @jung_gpu` first.
#
#   qsub cluster/job_gpu_fides_mgpu.sh
#   qsub -v FIDES_RUNS="boot 65536 4" cluster/job_gpu_fides_mgpu.sh
#
# Same harness as job_gpu_fides.sh (success = a RESULT line); runs default to
# the sparse then the full-slot bootstrap. nvidia-smi samples every card.
source ~/.bashrc
set -uo pipefail
export FHEDLM_ROOT=/groups/tjung/jzhao7/fhe-dlm
cd "$FHEDLM_ROOT"
source cluster/job_common.sh
set +e
[ "${CUDA_MODULE:-}" = none ] || module load "${CUDA_MODULE:-cuda/13.2.1}" 2>/dev/null
export LD_LIBRARY_PATH="$FHEDLM_ROOT/third_party/fideslib-mgpu/lib:$FHEDLM_ROOT/third_party/openfhe-fides/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=${NSLOTS:-8}
BIN="$FHEDLM_ROOT/gpu/fides_e0/build-mgpu/fides_e0"
export FIDES_E0_GPUS=${FIDES_E0_GPUS:-4}
NCCL_LIB=$(dirname "$(find "$FHEDLM_ROOT/third_party/nccl-cu13" -name libnccl.so.2 | head -1)")
[ -x "$BIN" ] || { echo "FATAL: $BIN missing -- run cluster/setup_fideslib_mgpu.sh"; exit 1; }
export LD_LIBRARY_PATH="$NCCL_LIB:$LD_LIBRARY_PATH"
echo "FIDES_E0_GPUS=$FIDES_E0_GPUS CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} NCCL=$NCCL_LIB"
mkdir -p results/logs
LOG=results/logs/gpu_fides_mgpu_${JOB_ID:-local}
OUT=results/gpu_fides_mgpu.json

# Peak device memory independent of the process's own view (1 s samples).
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader ${CUDA_VISIBLE_DEVICES:+-i "$CUDA_VISIBLE_DEVICES"} -l 1 > "$LOG.smi.csv" &
SMI=$!

: > "$LOG.results"
# FIDESlib's CUDA-failure path exits 0 (job 1519057 logged three OOMs as
# "exit 0"), so success means "printed a RESULT line", not the exit code.
run() {
  echo "=== fides_e0 $* ($(date))"
  local before after why
  before=$(grep -c '^RESULT ' "$LOG.results")
  "$BIN" "$@" 2> >(tee -a "$LOG.stderr" >&2) | tee -a "$LOG.results"
  local rc=${PIPESTATUS[0]}
  echo "=== exit $rc"
  after=$(grep -c '^RESULT ' "$LOG.results")
  if [ "$after" -eq "$before" ]; then
    why=$(grep -hoE "Cuda failure.*|terminate called.*|what\(\):.*" "$LOG.stderr" "$LOG.results" | tail -1 | tr '"' "'")
    echo "RESULT {\"mode\": \"$*\", \"exit\": $rc, \"failed\": true, \"error\": \"${why:-no RESULT line}\"}" >> "$LOG.results"
  fi
}
# Runs as "|"-separated fides_e0 argument lists; override with
#   qsub -v FIDES_RUNS="matvec_enc" cluster/job_gpu_fides.sh
IFS='|' read -ra RUNS <<< "${FIDES_RUNS:-boot 4096 4|boot 65536 4}"
for r in "${RUNS[@]}"; do
  read -ra args <<< "$r"
  run "${args[@]}"
done
kill $SMI

python3 - "$LOG.results" "$LOG.smi.csv" "$OUT" <<'PY'
import json, sys
rows = [json.loads(l[7:]) for l in open(sys.argv[1]) if l.startswith("RESULT ")]
mem = [float(l.split(",")[1].split()[0]) for l in open(sys.argv[2]) if l.count(",") >= 3]
out = {"runs": rows, "nvidia_smi_peak_mib": max(mem) if mem else None}
json.dump(out, open(sys.argv[3], "w"), indent=2)
print(json.dumps(out, indent=2))
PY
echo "[gpu] done at $(date)"
