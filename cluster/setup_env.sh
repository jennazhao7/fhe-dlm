#!/bin/bash
# One-time, run ON THE FRONT-END only (compute nodes have no outbound network).
# Builds the `fhedlm` conda env and prefetches every checkpoint the gate needs.
#
# fp32 ON PURPOSE: Experiment 1 injects controlled perturbations, so the model
# must have no other numerical error source. bf16 would add uncontrolled noise
# of the same order as the sigma_rel=1e-3 regime we are trying to measure.
# fp32 also means the Turing (sm_75) cards in gpu@@jung_gpu are usable.
source ~/.bashrc
set -uo pipefail
ROOT=~/fhe-dlm
export HF_HOME=$ROOT/hf-cache
mkdir -p "$HF_HOME"

# The env and every package cache live in group space, not the 100 GB $HOME.
ENV=/groups/tjung/jzhao7/conda-envs/fhedlm
export CONDA_PKGS_DIRS=/groups/tjung/jzhao7/conda-pkgs PIP_CACHE_DIR=/groups/tjung/jzhao7/pip-cache
if [ ! -x "$ENV/bin/python" ]; then
  conda create -p "$ENV" python=3.10 -y || exit 1
fi
conda activate "$ENV" || exit 1
python -m pip install -q --upgrade pip
# cu121 wheels: works on Turing sm_75 (jung queue) and Ampere A10 alike.
python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu121 || exit 1
python -m pip install -q "numpy<2.0.0" PyYAML tqdm einops "transformers>=4.41.2,<4.45.0" \
    datasets huggingface-hub sacrebleu rouge-score matplotlib pandas scipy || exit 1

echo "=== prefetching checkpoints into $HF_HOME ==="
python - <<PY
from huggingface_hub import snapshot_download
for r in ["embedded-language-flows/ELF-B-owt-torch",  # the model under test, 105M
          "openai-community/gpt2-large"]:             # frozen judge for gen. PPL
    p = snapshot_download(r)
    print("ok", r, p)
PY
echo "=== done ==="
python -c "import torch,transformers;print(torch,torch.__version__,tf,transformers.__version__)"
du -sh "$HF_HOME"
