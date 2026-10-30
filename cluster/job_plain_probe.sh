#!/bin/bash
#$ -N fhedlm_plain_probe
#$ -j y
#$ -q gpu@@jung_gpu
#$ -l gpu=1
#$ -pe smp 8
#$ -M jzhao7@nd.edu
#$ -m abe
#$ -l h_rt=8:00:00
#
# Plaintext probes on MDLM-OWT (scripts/plain_probe.py), one GPU:
#   ranges  E1 activation ranges by mask ratio  -> results/plain/ranges.json
#   noise   bootstrap-noise tolerance at 2^-b   -> results/plain/noise.json
#   outlier massive-activation stats + noise with the outlier handled (offset / oracle)
#           -> results/plain/outlier.json   (qsub -v PROBES=outlier ...)
#   noise2sink / noise2pin0  rerun at the bootstrap's RMS-matched precision
#           (1 pass ~ b=11, 2 passes ~ b=19) with gen-PPL at EVERY level, with
#           the position-0 sink as-is vs position 0 never masked (--pin0)
#           -> results/plain/noise2_{sink,pin0}.json
#           qsub -v PROBES="noise2sink noise2pin0" cluster/job_plain_probe.sh
#   approx  E1 zero-shot: each E5-circuit op (fixed-constant norm, (q.k)^2/Z
#           attention, Chebyshev GELU) swapped into MDLM, alone and together
#           -> results/plain/approx.json        (scripts/approx_probe.py)
#   spikes  attribute the remaining --pin0 spikes; 1-pass gen-PPL with
#           <|endoftext|> never sampled          -> results/plain/spikes.json
#   e2order E2 commit-ordering ablation, BD3-LM B=4/8/16 + MDLM global
#           -> results/plain/e2_order.json      (scripts/e2_order.py; resumable)
#   e2ent   E2 entropy-matched follow-up: confidence with Gumbel choice
#           temperature / sampling temperature, random at T < 1, real-text
#           reference -> results/plain/e2_entropy.json (e2_order.py --sweep)
#   alpha   spec-AR acceptance, 2/4-layer distilled GPT-2 drafts
#           -> results/plain/spec_alpha.json    (scripts/spec_alpha.py)
#           one probe per job, one GPU each:
#   alpha2  re-measure the saved drafts with the EOS-padding fix (no
#           distillation)               -> results/plain/spec_alpha2.json
#   e1pilot E1 KD pilot: MDLM -> E5 circuit, stages norms / attention / GELU /
#           bootstrap noise, eval after each -> results/plain/e1_pilot.json,
#           ckpt/e1_pilot/stage{A,B,C,D}.pt (scripts/e1_kd.py; resumes by stage)
#           qsub -l gpu=2 -v PROBES=e1pilot cluster/job_plain_probe.sh  (see the card check below)
#           for p in approx spikes e2order alpha; do qsub -v PROBES=$p cluster/job_plain_probe.sh; done
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
# 1636812 and 1640363 both landed on qa-rtx6k-019 GPU 00000000:2F:00.0 and sat
# for hours at ~500 MB RSS with no output (CUDA init never returned); the other
# three cards were fine. Fail fast instead of burning the h_rt. 1655431 and
# 1655468 were handed that card again (the only free one), so the e1pilot is
# submitted with `qsub -l gpu=2 ...`: SGE exposes the assigned cards as 0..n-1
# (CUDA_VISIBLE_DEVICES unset), and the first that passes is used.
GOOD=""
for i in $(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 ))); do
  bus=$(nvidia-smi -i "$i" --query-gpu=pci.bus_id --format=csv,noheader)
  if CUDA_VISIBLE_DEVICES=$i timeout 180 "$PY" -c "import torch; x = torch.ones(1024, device='cuda'); torch.cuda.synchronize(); print('cuda ok', torch.cuda.get_device_name(), float(x.sum()))"; then
    echo "=== using card $i ($bus)"; GOOD=$i; break
  fi
  echo "=== CUDA sanity check failed or timed out on card $i ($bus)"
done
if [ -z "$GOOD" ]; then echo "=== no working card -- aborting"; exit 1; fi
export CUDA_VISIBLE_DEVICES=$GOOD
for p in ${PROBES:-ranges noise}; do
  echo "=== $p ($(date))"
  case $p in
    ranges) "$PY" scripts/plain_probe.py ranges --out results/plain/ranges.json ;;
    noise)  "$PY" scripts/plain_probe.py noise --gen --out results/plain/noise.json ;;
    outlier) "$PY" scripts/plain_probe.py outlier --gen --out results/plain/outlier.json ;;
    noise2sink|noise2pin0)
      PIN=""; [ "$p" = noise2pin0 ] && PIN="--pin0"
      "$PY" scripts/plain_probe.py noise --gen $PIN --gen-n 64 \
        --bits 0 9 10 11 12 13 16 19 --gen-bits 9 10 11 12 13 16 19 \
        --out "results/plain/${p/noise2/noise2_}.json" ;;
    approx) "$PY" scripts/approx_probe.py approx --gen --out results/plain/approx.json ;;
    spikes) "$PY" scripts/approx_probe.py spikes --out results/plain/spikes.json ;;
    e2order) "$PY" scripts/e2_order.py --out results/plain/e2_order.json ;;
    e2ent)  "$PY" scripts/e2_order.py --sweep --out results/plain/e2_entropy.json ;;
    alpha)  "$PY" scripts/spec_alpha.py --save-dir "$FHEDLM_ROOT/ckpt/drafts" --out results/plain/spec_alpha.json ;;
    alpha2) "$PY" scripts/spec_alpha.py --load-dir "$FHEDLM_ROOT/ckpt/drafts" --out results/plain/spec_alpha2.json ;;
    e1pilot) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" scripts/e1_kd.py --ckpt-dir "$FHEDLM_ROOT/ckpt/e1_pilot" --out results/plain/e1_pilot.json ;;
  esac
  echo "=== $p exit $? ($(date))"
done
