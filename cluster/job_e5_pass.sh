#!/bin/bash
#$ -N fhedlm_e5
#$ -j y
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 6
#$ -M jzhao7@nd.edu
#$ -m abe
#$ -l h_rt=12:00:00
#
# E5: end-to-end encrypted MDLM generation on the GPU (scripts/e5_pass.py,
# gpu/fides_e0/fides_vm; HANDOFF §12). One job = one fides_vm on one card.
#
#   qstat -u jzhao7
#   qsub -l gpu=2 -v E5_TAG=linear_L64,E5_ARGS="linear --L 64 --steps 16" cluster/job_e5_pass.sh
#
# E5_ARGS    script arguments after --bin/--out (default: the linear-assist pass, L=64, 16 steps)
# E5_TAG     names results/e5/<tag>.json
# E5_SCRIPT  e5_pass.py (default; linear assist) or e5_circuit.py (server-side norm/attention/GELU)
# E5_BIN     fides_vm binary (default gpu/fides_e0/build/fides_vm)
#   qsub -v E5_SCRIPT=e5_circuit.py,E5_TAG=deep_C64,E5_ARGS="run --ckpt ckpt/e1_bc64/stageC.pt --stage C --exact-norm --steps 16" cluster/job_e5_pass.sh
# E5_STUDENT=<ckpt dir>  (e5_circuit.py) picks the student from an e1_kd.py --exact-norm run: stage C if
#   its checkpoint exists and the run did not abort there, else stage B with the GELU on the client;
#   appends --ckpt/--stage/--exact-norm[/--gelu client] to E5_ARGS. Use with -hold_jid <the E1 job>.
# E5_SELFTEST=0 skips the real-size matvec self-test (results/e5/selftest_gpu_<job>.json)
# The node's cards hang at CUDA init now and then (HANDOFF §11), and SGE cannot
# choose one, so ask for a spare (`-l gpu=2`): every assigned card is checked
# and the first that passes is used.
source ~/.bashrc
set -uo pipefail
export FHEDLM_ROOT=/groups/tjung/jzhao7/fhe-dlm
cd "$FHEDLM_ROOT"
source cluster/job_common.sh
set +e
PY=/groups/tjung/jzhao7/conda-envs/fhedlm/bin/python
export HF_HOME="$FHEDLM_ROOT/hf-cache" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${NSLOTS:-6}
export PROBE_DEV=cpu      # the client's plaintext steps run on the CPU; the card is the server's
mkdir -p results/e5 results/logs

GOOD=""
for i in $(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 ))); do
  bus=$(nvidia-smi -i "$i" --query-gpu=pci.bus_id --format=csv,noheader)
  if CUDA_VISIBLE_DEVICES=$i timeout 180 "$PY" -c "import torch; x = torch.ones(1024, device='cuda'); torch.cuda.synchronize(); print('cuda ok', torch.cuda.get_device_name(), float(x.sum()))"; then
    echo "=== card $i ($bus) ok"; [ -z "$GOOD" ] && GOOD=$i
  else
    echo "=== CUDA sanity check failed or timed out on card $i ($bus)"
  fi
done
if [ -z "$GOOD" ]; then echo "=== no working card -- aborting"; exit 1; fi
echo "=== using card $GOOD"
export CUDA_VISIBLE_DEVICES=$GOOD

module load "${CUDA_MODULE:-cuda/13.2.1}" 2>/dev/null
export LD_LIBRARY_PATH="$FHEDLM_ROOT/third_party/fideslib/lib:$FHEDLM_ROOT/third_party/fideslib/lib64:$FHEDLM_ROOT/third_party/openfhe-fides/lib:${LD_LIBRARY_PATH:-}"
BIN="${E5_BIN:-$FHEDLM_ROOT/gpu/fides_e0/build/fides_vm}"
[ -x "$BIN" ] || { echo "FATAL: $BIN missing -- cmake --build gpu/fides_e0/build --target fides_vm"; exit 1; }
WORK="${TMPDIR:-$FHEDLM_ROOT/scratch/e5tmp}"; mkdir -p "$WORK"
TAG=${E5_TAG:-linear_L64}

nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -i "$GOOD" -l 5 > "results/logs/e5_${JOB_ID:-local}.smi.csv" &
SMI=$!
if [ "${E5_SELFTEST:-1}" = 1 ]; then
  echo "=== selftest ($(date))"
  "$PY" scripts/e5_pass.py selftest --bin "$BIN" --workdir "$WORK" --out "results/e5/selftest_gpu_${JOB_ID:-local}.json"
  rc=$?; echo "=== selftest exit $rc ($(date))"
  [ $rc -eq 0 ] || { kill $SMI; exit 1; }
fi
read -ra ARGS <<< "${E5_ARGS:-linear --L 64 --steps 16}"
if [ -n "${E5_STUDENT:-}" ]; then
  J="results/plain/$(basename "$E5_STUDENT").json"
  aborted=$("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(' '.join(r['abort'] for r in d.get('log', []) if 'abort' in r))" "$J" 2>/dev/null)
  echo "=== student $E5_STUDENT: checkpoints $(ls "$E5_STUDENT" 2>/dev/null | tr '\n' ' '); aborted stages: ${aborted:-none}"
  if [ -f "$E5_STUDENT/stageC.pt" ] && [[ " $aborted " != *" C "* ]]; then
    ARGS+=(--ckpt "$E5_STUDENT/stageC.pt" --stage C --exact-norm)
  elif [ -f "$E5_STUDENT/stageB.pt" ] && [[ " $aborted " != *" B "* ]]; then
    ARGS+=(--ckpt "$E5_STUDENT/stageB.pt" --stage B --exact-norm --gelu client); TAG="${TAG}_stageB"
  else
    echo "=== no usable student in $E5_STUDENT -- aborting"; kill $SMI; exit 1
  fi
fi
echo "=== ${E5_SCRIPT:-e5_pass.py} ${ARGS[*]} ($(date))"
"$PY" "scripts/${E5_SCRIPT:-e5_pass.py}" "${ARGS[0]}" --bin "$BIN" --workdir "$WORK" --out "results/e5/${TAG}.json" "${ARGS[@]:1}"
echo "=== e5 exit $? ($(date))"
kill $SMI
echo "peak GPU MiB: $(cut -d, -f2 "results/logs/e5_${JOB_ID:-local}.smi.csv" | tr -dc '0-9\n' | sort -n | tail -1)"
