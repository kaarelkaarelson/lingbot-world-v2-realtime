#!/bin/bash
# Weights in the layout LightX2V's lingbot_world_fast runner resolves from --model_path (wan_runner.py):
#   $LX/model/Wan2.1_VAE.pth                      vae_name default            (14B repo)
#   $LX/model/models_t5_umt5-xxl-enc-bf16.pth     find_torch_model_path       (14B repo, bf16, 11.4 GB)
#   $LX/model/google/umt5-xxl/                    T5 tokenizer                (14B repo)
#   $LX/model/transformers/model-0000?-of-00006.safetensors   dit_original_ckpt in config_1p3b.json (1.3B repo, fp32 shards, 6.8 GB)
# The 1.3B repo has no config.json and none is written: the DiT geometry lives in config_1p3b.json
# (set_config.py would let a model_path/config.json override it). ~19 GB total.
set -euo pipefail
LX=${LX:-/workspace/lx}; M="$LX/model"; mkdir -p "$M"
export HF_HOME=${HF_HOME:-$LX/hf}
: "${HF_TOKEN:?HF_TOKEN must be set}"
df -h /workspace | sed 's/^/disk: /'
"$LX/.venv/bin/python" - "$M" <<'PY'
import sys
from huggingface_hub import HfApi, snapshot_download
m = sys.argv[1]
want = {
    "robbyant/lingbot-world-v2-1.3b-causal-fast": (["*.safetensors"], f"{m}/transformers"),
    "robbyant/lingbot-world-v2-14b-causal-fast": (["Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl/*"], m),
}
api = HfApi()
total = 0
for repo, (patterns, _) in want.items():
    import fnmatch
    for s in api.model_info(repo, files_metadata=True).siblings:
        if any(fnmatch.fnmatch(s.rfilename, p) for p in patterns):
            print(f"{s.size / 1e9:7.2f} GB  {repo}/{s.rfilename}")
            total += s.size or 0
print(f"{total / 1e9:7.2f} GB  total to download", flush=True)
for repo, (patterns, local_dir) in want.items():
    snapshot_download(repo, allow_patterns=patterns, local_dir=local_dir)
PY
ls -la "$M" "$M/transformers" "$M/google/umt5-xxl"
du -sh "$M"
df -h /workspace | sed 's/^/disk: /'
