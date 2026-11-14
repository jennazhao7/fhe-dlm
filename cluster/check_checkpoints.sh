#!/bin/bash
# FRONT-END ONLY (needs network). HANDOFF §6.5: the experiment plan marks the
# MDLM / BD3-LM checkpoint names "verify". Prints, per repo: exists?, license,
# weight files and total size -- nothing is downloaded.
#   bash cluster/check_checkpoints.sh
source ~/.bashrc
conda activate /groups/tjung/jzhao7/conda-envs/fhedlm
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
REPOS = ["kuleshov-group/mdlm-owt",
         "kuleshov-group/bd3lm-owt-block_size4",
         "kuleshov-group/bd3lm-owt-block_size8",
         "kuleshov-group/bd3lm-owt-block_size16"]
for r in REPOS:
    try:
        i = api.model_info(r, files_metadata=True)
    except Exception as e:
        print(f"{r:45s} MISSING ({type(e).__name__})"); continue
    lic = (i.card_data or {}).get("license") if i.card_data else None
    lic = lic or [t for t in i.tags if t.startswith("license:")]
    w = [s for s in i.siblings if s.rfilename.endswith((".safetensors", ".bin", ".ckpt", ".pt"))]
    gb = sum((s.size or 0) for s in i.siblings) / 2**30
    print(f"{r:45s} license={lic}  weights={[s.rfilename for s in w]}  total={gb:.2f} GB")
# Anything else the group publishes on OWT, in case a name changed.
print("\nall kuleshov-group *owt* models:")
for m in api.list_models(author="kuleshov-group", search="owt"):
    print("  ", m.id)
PY
