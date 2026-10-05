#!/bin/bash
# Front-end only: compute nodes have no outbound network.
#
# allow_patterns is NOT optional here. A bare snapshot_download of gpt2-large
# pulls the TF, Flax, ONNX and Rust copies too -- 19 GB instead of 3 -- which is
# how this project briefly ate two thirds of the free space in a shared $HOME.
source ~/.bashrc
conda activate fhedlm
set -uo pipefail
export FHEDLM_ROOT="${FHEDLM_ROOT:-/groups/tjung/jzhao7/fhe-dlm}"
export HF_HOME="$FHEDLM_ROOT/hf-cache"
mkdir -p "$HF_HOME"

python - <<'PY'
from huggingface_hub import snapshot_download
TORCH_ONLY = ["*.json", "*.txt", "*.model", "*.safetensors", "merges.txt", "vocab.json"]
for repo, pats in [
    ("embedded-language-flows/ELF-B-owt-torch", None),   # small (802 MB), take all
    ("openai-community/gpt2-large",             TORCH_ONLY),
    ("google-t5/t5-small",                      TORCH_ONLY),  # tokenizer for decoding
]:
    p = snapshot_download(repo, allow_patterns=pats)
    print("ok", repo, "->", p, flush=True)
PY
du -sh "$HF_HOME"/hub/* 2>/dev/null
