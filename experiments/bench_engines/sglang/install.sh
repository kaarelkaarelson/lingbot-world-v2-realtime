#!/usr/bin/env bash
# Fresh SGLang v0.5.17 (+ diffusion extras) venv for an RTX 5090 pod on driver 570 / CUDA 12.8.
#
# Why this exact combination (README.md, research/engines/sglang.md):
#   - v0.5.17 is the last SGLang release with LingBot-World 2.0 that pins torch 2.11 (v0.5.18+ pin 2.13, CUDA-13 only).
#   - PyPI sglang==0.5.17 pulls torch 2.11.0 built for CUDA 13; driver 570 only speaks CUDA 12.8, so torch is
#     force-reinstalled from the cu128 wheel index (all four wheels exist there: torch 2.11.0, torchvision 0.26.0,
#     torchaudio 2.11.0, torchcodec 0.11.1). This mirrors the CUDA-12 lane in v0.5.17's docs/get-started/install.mdx.
#   - sglang-kernel / sgl-deep-gemm have no cu128 build; the +cu129 wheels from docs.sglang.ai run on a 12.8 driver
#     under CUDA minor-version compatibility.
#   - flashinfer-python is one universal wheel; [cu12] vs [cu13] only changes the cutlass-dsl libs extra. It is only
#     used for the in-place RoPE kernel on the LingBot path (runtime/layers/rotary_embedding/utils.py). The Triton
#     fallback only triggers when `import flashinfer` fails; if the import succeeds but the kernel cannot be JIT-built
#     (no nvcc in the image) the first chunk errors out. flashinfer-jit-cache (cu128 build, arch list includes 12.0a)
#     ships that kernel prebuilt, so nvcc is never needed.
#
# Disk: ~10 GB after cleanup (jit-cache is 1.3 GB). Override SGL_VENV to put the venv on the container disk (/) if
# /workspace is tight; the model dir (18.7 GB) stays on /workspace either way.
set -euo pipefail

UV="$HOME/.local/bin/uv"
SGL_ROOT=/workspace/sgl
VENV="${SGL_VENV:-$SGL_ROOT/.venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SGL_ROOT/.uv-cache}"   # same filesystem as the venv -> hardlinks, no double copy
export CUDA_HOME=/usr/local/cuda

SGLANG_VERSION=0.5.17
SGLANG_KERNEL_VERSION=0.4.5
SGL_DEEP_GEMM_VERSION=0.1.5.post1
FLASHINFER_VERSION=0.6.15.post1

echo "== disk before"
df -h /workspace / | sed 's/^/   /'

mkdir -p "$SGL_ROOT"
"$UV" --version
"$UV" venv --python 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "== 1/6 sglang[diffusion]==$SGLANG_VERSION (brings the CUDA-13 torch; replaced in step 2)"
"$UV" pip install --prerelease=allow "sglang[diffusion]==$SGLANG_VERSION"

echo "== 2/6 torch 2.11.0 cu128 stack (download.pytorch.org hosts every torch dependency, so --index-url is safe here)"
"$UV" pip install --force-reinstall \
    torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 torchcodec==0.11.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

echo "== 3/6 sglang-kernel + sgl-deep-gemm cu129 wheels (--no-deps: never let these re-resolve torch)"
"$UV" pip install --force-reinstall --no-deps "sglang-kernel==$SGLANG_KERNEL_VERSION" \
    --index-url https://docs.sglang.ai/whl/cu129/
"$UV" pip install --force-reinstall --no-deps "sgl-deep-gemm==$SGL_DEEP_GEMM_VERSION" \
    --index-url https://docs.sglang.ai/whl/cu129/

echo "== 4/6 flashinfer cu12 extras (no --force-reinstall: that would re-resolve torch from PyPI)"
"$UV" pip install "flashinfer-python[cu12]==$FLASHINFER_VERSION"

echo "== 5/6 flashinfer-jit-cache cu128 (prebuilt kernels incl. sm_120; 1.3 GB; removes the nvcc dependency)"
"$UV" pip install --no-deps "flashinfer-jit-cache==$FLASHINFER_VERSION" \
    --index-url https://flashinfer.ai/whl/cu128/

echo "== 6/6 drop the orphaned CUDA-13 runtime libs that the PyPI torch pulled in (~3.5 GB)"
# The cu13 and cu12 nvidia-* packages share the nvidia/ namespace directory: uninstalling the cu13 ones also
# deletes files the cu12 ones own (metadata says installed, libcusparseLt.so.0 / libnvshmem_host.so.3 are gone).
# So the torch cu128 stack is reinstalled after the removal (measured on pod 11: this is what made the import pass).
ORPHANS=$("$UV" pip list --format=freeze \
    | grep -E '^nvidia-(cuda-nvrtc|cuda-runtime|cuda-cupti|cublas|cudnn|cufft|curand|cusolver|cusparse|cusparselt|nccl|nvtx|nvjitlink|cufile|nvshmem)-cu13==' \
    | cut -d= -f1 || true)
if [[ -n "$ORPHANS" ]]; then
    echo "$ORPHANS" | sed 's/^/   removing /'
    # shellcheck disable=SC2086
    "$UV" pip uninstall $ORPHANS
fi
"$UV" pip install --reinstall torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 torchcodec==0.11.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
"$UV" cache clean

echo "== pins actually installed"
"$UV" pip list --format=freeze | grep -E '^(torch|torchvision|torchaudio|torchcodec|sglang|sglang-kernel|sgl-deep-gemm|flashinfer-python|flashinfer-jit-cache|nvidia-cutlass-dsl|cuda-python|cuda-bindings|diffusers|transformers|triton|huggingface-hub|hf-xet|websockets|msgspec|imageio|imageio-ffmpeg)=='
echo "nvcc: $(command -v nvcc || echo 'not found (fine: flashinfer-jit-cache is installed)')"
echo "driver: $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null || echo 'nvidia-smi unavailable')"

echo "== import smoke test"
python - <<'PY'
import importlib.metadata as md
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print("cuda available", torch.cuda.is_available(),
      "capability", torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
import sglang
print("sglang", sglang.__version__)
print("sglang-kernel", md.version("sglang-kernel"))
import sgl_kernel  # noqa: F401
print("sgl_kernel import ok (cu129 wheel on the cu128 driver)")
try:
    import flashinfer
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace  # noqa: F401
    print("flashinfer", flashinfer.__version__, "jit-cache", md.version("flashinfer-jit-cache"))
except Exception as e:  # noqa: BLE001
    # not fatal: sglang falls back to the Triton RoPE kernel with a one-time warning
    print(f"WARNING flashinfer import failed ({type(e).__name__}: {e}); LingBot RoPE will use the Triton fallback")
import msgspec, websockets, imageio, imageio_ffmpeg, huggingface_hub  # noqa: F401,E401
print("huggingface_hub", md.version("huggingface_hub"), "ffmpeg", imageio_ffmpeg.get_ffmpeg_exe())
from sglang.multimodal_gen.runtime.pipelines.lingbot_world_causal_dmd_pipeline import LingBotWorldCausalDMDPipeline  # noqa: F401,E501
from sglang.multimodal_gen.configs.pipeline_configs.lingbot_world import LingBotWorldV2CausalDMDConfig  # noqa: F401
from sglang.multimodal_gen.runtime.platforms import current_platform
print("sm120", current_platform.is_sm120(), "-> attention backend on this card:", "TORCH_SDPA" if current_platform.is_sm120() else "FA")
print("SGL install ok")
PY

echo "== disk after"
df -h /workspace / | sed 's/^/   /'
du -sh "$VENV"
