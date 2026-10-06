#!/usr/bin/env bash
# Second venv for the stack-upgrade experiment: torch 2.14 + CUDA 13.0, next to setup.sh's torch 2.8 + cu128 .venv,
# so both stacks are measured on the same pod. Run after setup.sh (it reuses ./weights). Needs the CUDA 13.0 toolkit
# (Vast image vastai/pytorch:cuda-13.0.3-auto) because SageAttention and flash_attn are compiled here against it.
#
#   bash tools/setup_cu130.sh            # venv + patched SageAttention (~5 min), flash_attn in the background
#   source .venv-cu130/bin/activate      # then run benches as usual (the warm worker hashes sys.executable)
set -euo pipefail
cd "$(dirname "$0")/.."
TORCH=${TORCH:-2.14.0}; VISION=${VISION:-0.29.0}; IDX=${IDX:-https://download.pytorch.org/whl/cu130}
JOBS=${JOBS:-$(( $(nproc) / 2 ))}
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0
nvcc --version | tail -1

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; export PATH="$HOME/.local/bin:$PATH"; }
[ -d .venv-cu130 ] || uv venv -q --python 3.12 .venv-cu130
. .venv-cu130/bin/activate
uv pip install -q torch==$TORCH torchvision==$VISION --index-url $IDX
grep -v -E '^(torch|torchvision|torchao)==' requirements.txt > /tmp/req_cu130.txt
uv pip install -q -r /tmp/req_cu130.txt torchao setuptools wheel ninja packaging
uv pip install -q --no-deps -e .
python -c "import torch, torchao; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| torchao', torchao.__version__)"

echo "== SageAttention 2.2 + current-stream patch, built for sm_120 against CUDA $(nvcc --version | grep -o 'release [0-9.]*') =="
mkdir -p build && cd build
[ -d SageAttention ] || git clone -q https://github.com/thu-ml/SageAttention.git
cd SageAttention && git checkout -q . && git apply ../../patches/sageattention-current-stream.patch && sed -i "s/-std=c++17/-std=c++20/g" setup.py  # torch 2.14 headers need C++20
EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=$JOBS python setup.py -q bdist_wheel > ../sage_cu130_build.log 2>&1
uv pip install -q --force-reinstall --no-deps dist/sageattention-*.whl
cd ../..
python -c "import sageattention, torch; q=torch.randn(1,2,512,128,device='cuda',dtype=torch.bfloat16); print('sage ok', sageattention.sageattn(q,q,q,tensor_layout='HND').shape)"

echo "== flash_attn 2.8.3 for sm_120 (background; cross-attention falls back to SDPA until it lands) =="
(FLASH_ATTN_CUDA_ARCHS=120 MAX_JOBS=$JOBS uv pip install --no-build-isolation flash-attn==2.8.3 > build/flash_cu130_build.log 2>&1 \
  && echo "flash_attn cu130 installed" >> build/flash_cu130_build.log) &
echo "cu130 venv ready (flash_attn building: tail -f build/flash_cu130_build.log)"
