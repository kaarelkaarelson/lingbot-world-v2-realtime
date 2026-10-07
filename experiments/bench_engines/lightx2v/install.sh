#!/bin/bash
# LightX2V (ModelTC) with LingBot-World 2.0 1.3B causal_fast on one RTX 5090 — install.
# Own venv at /workspace/lx/.venv. LightX2V pins no torch; its reference image is
# pytorch/pytorch:2.8.0-cuda12.8 (dockerfiles/Dockerfile), and the sm_120 community wheels
# (flash_attn 2.8.3.post1, sageattention 2.2.0) are built for torch 2.8.0+cu128/cp312, so
# torch 2.8.0+cu128 it is (driver 570 = CUDA 12.8; no cu130). Attention default: torch SDPA.
set -euo pipefail
LX=${LX:-/workspace/lx}; mkdir -p "$LX"; cd "$LX"
LIGHTX2V_SHA=69018c92b0a42d9b0cf962a248fadbfe0cbc03de   # 2026-09-17, "feat: support pipefusion for flux2 (#1268)"
WHL=https://github.com/kaarelkaarelson/lingbot-world-v2-realtime/releases/download/v0.1.0
df -h /workspace / | sed 's/^/disk: /'
command -v ~/.local/bin/uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
U=~/.local/bin/uv
[ -d LightX2V ] || git clone -q https://github.com/ModelTC/LightX2V.git
git -C LightX2V checkout -q "$LIGHTX2V_SHA"
$U python install 3.12
[ -x "$LX/.venv/bin/python" ] || $U venv --python 3.12 "$LX/.venv"
P="$LX/.venv/bin/python"
$U pip install --python "$P" torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
$U pip install --python "$P" -e "$LX/LightX2V"                   # pure python; pulls diffusers/transformers/imageio-ffmpeg etc.
# pyproject misses pyzmq (lightx2v.disagg imports it at module load); requirements.txt has it, minus the torch pins
$U pip install --python "$P" -r <(grep -vE '^(torch|torchvision|torchaudio|flash|sage)' "$LX/LightX2V/requirements.txt")
$U pip install --python "$P" "$WHL/flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl" "$WHL/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl"
"$P" - <<'PY'
import torch
assert torch.__version__.startswith("2.8.0"), torch.__version__     # a dep must not have replaced the cu128 torch
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0), "sm", torch.cuda.get_device_capability(0))
import lightx2v                                                     # runs set_ai_device (needs the GPU)
from lightx2v.models.runners.runner_factory import RUNNER_MODULES
assert "lingbot_world_fast" in RUNNER_MODULES, sorted(RUNNER_MODULES)
from lightx2v.models.runners.wan.wan_lingbot_fast_runner import LingbotFastRunner  # noqa: F401
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER
print("attn backends:", [k for k in ("torch_sdpa", "flash_attn2", "sage_attn2") if k in ATTN_WEIGHT_REGISTER])
import flash_attn, sageattention  # noqa: F401
from importlib.metadata import version
print("flash_attn", version("flash_attn"), "sageattention", version("sageattention"))
q = torch.randn(1, 64, 12, 128, device="cuda", dtype=torch.bfloat16)
print("flash_attn_func on sm_120 ok:", flash_attn.flash_attn_func(q, q, q).shape)
print("install ok")
PY
df -h /workspace / | sed 's/^/disk: /'
