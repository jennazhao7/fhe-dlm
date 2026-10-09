#!/bin/bash
#$ -N fhedlm_gpu_fides
#$ -j y
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 8
#$ -M jzhao7@nd.edu
#$ -m abe
#$ -l h_rt=4:00:00
#
# E0 GPU feasibility (HANDOFF §6.1, E5 go/no-go): FIDESlib on ONE RTX 6000
# (24 GB, sm_75) at the E5 parameter set N=2^17, depth 43, 59/60, budget [4,4].
# Build first on the front end: bash cluster/setup_fideslib.sh
#
#   qstat -u jzhao7 ; qsub cluster/job_gpu_fides.sh ; qstat -u jzhao7
#
# Runs, each in its own process so an OOM in one does not hide the rest:
#   1. boot  4096 slots   (sparse; the cheaper key set)
#   2. boot 65536 slots   (full slots; 39 GB host RSS on CPU -- may not fit)
#   3. matvec 1024x1024 at level 21 (rotation keys only, no bootstrap keys)
# Key generation is on the host (OpenFHE), so the node also needs ~40 GB RAM.
# Expectation (HANDOFF §8): one key-switching key at these parameters is
# 354 MB. Sparse [4,4] bootstrap needs 63 rotation keys (= 22 GB) and the
# n1=n2=32 matvec needs 62 (= 22 GB), so every run is at or over 24 GB at
# load_context. A failure there is a valid outcome -- it is the go/no-go.
# Output: results/gpu_fides.json (one RESULT line per run) + results/logs/.
# -M/-m are file directives, always as a pair (see cluster/job_common.sh).
source ~/.bashrc
set -uo pipefail
export FHEDLM_ROOT=/groups/tjung/jzhao7/fhe-dlm
cd "$FHEDLM_ROOT"
source cluster/job_common.sh
set +e
[ "${CUDA_MODULE:-}" = none ] || module load "${CUDA_MODULE:-cuda/13.2.1}" 2>/dev/null
export LD_LIBRARY_PATH="$FHEDLM_ROOT/third_party/fideslib/lib:$FHEDLM_ROOT/third_party/openfhe-fides/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=${NSLOTS:-8}
BIN="$FHEDLM_ROOT/gpu/fides_e0/build/fides_e0"
[ -x "$BIN" ] || { echo "FATAL: $BIN missing -- run cluster/setup_fideslib.sh"; exit 1; }
mkdir -p results/logs
LOG=results/logs/gpu_fides_${JOB_ID:-local}
OUT=results/gpu_fides.json

# Peak device memory independent of the process's own view (1 s samples).
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader ${CUDA_VISIBLE_DEVICES:+-i "${CUDA_VISIBLE_DEVICES%%,*}"} -l 1 > "$LOG.smi.csv" &
SMI=$!

: > "$LOG.results"
run() {
  echo "=== fides_e0 $* ($(date))"
  "$BIN" "$@" 2> >(tee -a "$LOG.stderr" >&2) | tee -a "$LOG.results"
  local rc=${PIPESTATUS[0]}
  echo "=== exit $rc"
  [ "$rc" -eq 0 ] || echo "RESULT {\"mode\": \"$*\", \"exit\": $rc}" >> "$LOG.results"
}
run boot 4096 4
run boot 65536 4
run matvec
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
