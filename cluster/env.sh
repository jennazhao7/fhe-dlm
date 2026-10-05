#!/bin/bash
# Environment for every fhe-dlm job on CRC. Sourced by all job scripts here.
# Modelled on ~/benchmark-feedback-leakage/cluster/env.sh -- same house rules.
#
# THIS SCRIPT NEVER CREATES AN ENV OR INSTALLS ANYTHING.
# Installs happen deliberately from the front end via cluster/setup_env.sh.
# $HOME is 100 GB and shared with other running jobs; this project lives in
# /groups/tjung (5 TB) precisely so it cannot fill it.

set -uo pipefail

_ENV_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FHEDLM_ROOT="${FHEDLM_ROOT:-$(dirname "$_ENV_SH_DIR")}"
export FHEDLM_ROOT

FHEDLM_CONDA_ENV="${FHEDLM_CONDA_ENV:-${CONDA_PREFIX:-}}"
if [ -z "$FHEDLM_CONDA_ENV" ]; then
  echo "FATAL: FHEDLM_CONDA_ENV unset and no conda env active."
  echo "       qsub does not forward your environment unless you pass -V."
  return 1 2>/dev/null || exit 1
fi
PY="$FHEDLM_CONDA_ENV/bin/python"
[ -x "$PY" ] || { echo "FATAL: no python at $PY"; return 1 2>/dev/null || exit 1; }
export PY
export PATH="$FHEDLM_CONDA_ENV/bin:$PATH"
# ELF's own modules import as `modules.*` / `utils.*`, so its src/ must be on the
# path, not just the repo root.
export PYTHONPATH="${FHEDLM_ROOT}/src:${FHEDLM_ROOT}/third_party/ELF/src:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

export HF_HOME="${HF_HOME:-${FHEDLM_ROOT}/hf-cache}"
# Compute nodes have no outbound network; cluster/prefetch.sh runs on the front end.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-$([ "${FHEDLM_ALLOW_DOWNLOAD:-0}" = "1" ] && echo 0 || echo 1)}"
export TOKENIZERS_PARALLELISM=false

# fp32 everywhere, on purpose. Experiment 1 measures the model's response to
# injected relative error down to 1e-5; bf16's own ~3e-3 relative error would sit
# in the middle of that sweep and confound every point in it. The jung queue is
# Turing (sm_75) and has no bf16 anyway.
export FHEDLM_FORCE_FP32=1

"$PY" - <<'PYEOF' || { echo "FATAL: environment is incomplete (see above)"; return 1 2>/dev/null || exit 1; }
import importlib.util as u, sys
need = ("torch", "transformers", "numpy", "yaml", "einops")
missing = [m for m in need if u.find_spec(m) is None]
if missing:
    print("FATAL: missing packages:", ", ".join(missing)); sys.exit(1)
import torch
print(f"[env] python {sys.version.split()[0]}  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"[env] {p.name} sm_{p.major}{p.minor} {p.total_memory/1024**3:.1f} GB")
else:
    print("[env] WARNING: no CUDA device -- did this land on a GPU node?")
PYEOF

echo "[env] FHEDLM_ROOT: $FHEDLM_ROOT"
echo "[env] HF_HOME:     $HF_HOME (offline=$HF_HUB_OFFLINE)"
AVAIL_GB=$(df -BG --output=avail "$FHEDLM_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
echo "[env] free on $FHEDLM_ROOT: ${AVAIL_GB:-?} GB"
