#!/bin/bash
# FlashDreams (NVIDIA) with LingBot-World 2.0 1.3B causal_fast on one RTX 5090 — install.
# Own venv (FlashDreams needs torch >= 2.9; the cuda12 group resolves torch 2.11+cu128, fine on driver 570).
# Disk: the UMT5 text encoder it downloads is fp32, 22.7 GB — point HF_HOME at a disk with >= 25 GB free.
set -euo pipefail
FD=${FD:-/workspace/fd}; mkdir -p "$FD"; cd "$FD"
apt-get install -y -qq ffmpeg >/dev/null 2>&1 || true          # the MP4 sink shells out to ffmpeg
command -v ~/.local/bin/uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
U=~/.local/bin/uv; $U self update >/dev/null 2>&1 || true       # pyproject requires uv >= 0.11.8
[ -d flashdreams ] || git clone -q https://github.com/NVIDIA/flashdreams.git
cd flashdreams && git checkout -q c1889e0
$U python install 3.12
$U sync --python 3.12 --package flashdreams --no-group cuda13 --group cuda12 --extra runners --extra serving
$U pip install --no-deps -e apps/cam2v -e apps/v2v -e integrations_v2/flashvsr -e integrations_v2/lingbot
$U pip install ninja
cp -r "$(dirname "$0")/lingbot13" .                              # out-of-tree 1.3B pipeline config + slug
$U pip install --no-deps -e lingbot13
$U run --no-sync flashdreams-run-v2 --help | grep -q cam2v-lingbot-1p3b && echo "slug cam2v-lingbot-1p3b registered"
$U run --no-sync python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name())'
