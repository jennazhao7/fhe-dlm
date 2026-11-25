#!/bin/bash
#$ -N fhedlm_gpuchk
#$ -j y
#$ -q gpu@@jung_gpu
#$ -l gpu=4
#$ -pe smp 4
#$ -M jzhao7@nd.edu
#$ -m abe
#$ -l h_rt=0:20:00
#
# Which of the node's cards still work? Every assigned card is tested at the
# same time, each in its own background process with a hard limit, so one card
# that hangs in the driver cannot block the others (job 1819824 hung > 10 min
# inside `timeout 180` on card 06:00.0). The script reports after GPUCHK_WAIT
# seconds whatever has finished and exits; it never waits on a stuck process.
#
#   qsub cluster/job_gpu_check.sh          # all 4 cards
#   qsub -l gpu=2 cluster/job_gpu_check.sh
set -u
cd /groups/tjung/jzhao7/fhe-dlm
PY=/groups/tjung/jzhao7/conda-envs/fhedlm/bin/python
OUT=results/logs/gpucheck_${JOB_ID:-local}; mkdir -p "$OUT"
WAIT=${GPUCHK_WAIT:-240}

echo "host $(hostname)  job ${JOB_ID:-n/a}  $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "== uptime: $(uptime)"
echo "== processes in uninterruptible sleep (D) on the node:"
ps -eo pid,user,stat,etime,wchan:30,cmd | awk 'NR==1 || $3 ~ /^D/' | cut -c1-200
echo "== nvidia-smi"
timeout -s KILL 60 nvidia-smi --query-gpu=index,pci.bus_id,name,memory.used,memory.total,utilization.gpu,temperature.gpu,pstate,ecc.errors.uncorrected.volatile.total --format=csv \
  || echo "nvidia-smi query failed or timed out"
timeout -s KILL 60 nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv || true

N=$(timeout -s KILL 60 nvidia-smi -L | wc -l)
echo "== $N card(s) visible; testing all at once (limit ${WAIT}s)"
for i in $(seq 0 $((N - 1))); do
  (
    bus=$(timeout -s KILL 30 nvidia-smi -i "$i" --query-gpu=pci.bus_id --format=csv,noheader)
    echo "$bus" > "$OUT/card$i.bus"
    t0=$(date +%s.%N)
    CUDA_VISIBLE_DEVICES=$i timeout -s KILL $((WAIT - 20)) "$PY" - > "$OUT/card$i.log" 2>&1 <<'EOF'
import time, torch
t = time.time()
torch.cuda.init(); n = torch.cuda.device_count()
x = torch.randn(16384, 16384, device="cuda")          # 1 GiB
y = torch.empty(256 * 1024 * 1024, device="cuda")     # + 1 GiB
y.fill_(1.0)
z = x @ x; torch.cuda.synchronize()
print(f"ok {torch.cuda.get_device_name()} init+alloc+matmul {time.time() - t:.1f}s "
      f"sum {float(y.sum()):.0f} free {torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB")
EOF
    echo "rc=$? $(python3 -c "import time;print(f'{time.time()-$t0:.1f}')" 2>/dev/null)s" >> "$OUT/card$i.log"
  ) &
done

end=$(( $(date +%s) + WAIT ))
while [ "$(date +%s)" -lt "$end" ]; do
  done_n=$(grep -l "^rc=" "$OUT"/card*.log 2>/dev/null | wc -l)
  [ "$done_n" -ge "$N" ] && break
  sleep 5
done

echo "== results"
for i in $(seq 0 $((N - 1))); do
  bus=$(cat "$OUT/card$i.bus" 2>/dev/null || echo "?")
  if grep -q "^ok" "$OUT/card$i.log" 2>/dev/null; then
    echo "card $i ($bus): USABLE  $(grep '^ok' "$OUT/card$i.log")  $(grep '^rc=' "$OUT/card$i.log")"
  elif grep -q "^rc=" "$OUT/card$i.log" 2>/dev/null; then
    echo "card $i ($bus): FAILED  $(grep '^rc=' "$OUT/card$i.log")  $(grep -v '^rc=' "$OUT/card$i.log" | tail -2 | tr '\n' ' ' | cut -c1-200)"
  else
    echo "card $i ($bus): HUNG (no answer after ${WAIT}s)"
  fi
done
echo "== D-state processes after the test:"
ps -eo pid,user,stat,etime,wchan:30,cmd | awk 'NR==1 || $3 ~ /^D/' | cut -c1-200
echo "done $(date)"
exit 0
