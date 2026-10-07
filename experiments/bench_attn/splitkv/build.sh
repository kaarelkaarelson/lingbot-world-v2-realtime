#!/usr/bin/env bash
# Build SageAttention (upstream main d1a57a5) + current-stream patch + splitkv.patch for sm_120.
#   bash experiments/bench_attn/splitkv/build.sh
# Result: $ROOT/SageAttention/dist/sageattention-*.whl (also installed into the active python env unless NO_INSTALL=1).
# Run it from the repo with the target venv activated (e.g. .venv-cu130) so torch is the one the wheel is built against.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
ROOT="${SAGE_SPLITKV_ROOT:-$REPO/build/splitkv}"
COMMIT=d1a57a5
PY="${PYTHON:-python}"
export PATH="/usr/local/cuda/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

mkdir -p "$ROOT"
cd "$ROOT"
[ -d SageAttention/.git ] || git clone -q https://github.com/thu-ml/SageAttention.git
cd SageAttention
git checkout -q "$COMMIT"
git reset -q --hard "$COMMIT"
git clean -qfdx
git apply --check "$REPO/patches/sageattention-current-stream.patch"
git apply "$REPO/patches/sageattention-current-stream.patch"
git apply --check "$HERE/splitkv.patch"
git apply "$HERE/splitkv.patch"
echo "== applied current-stream + splitkv on $(git rev-parse --short HEAD)"

TORCH_VER="$("$PY" -c 'import torch; print(torch.__version__.split("+")[0])')"
if "$PY" -c "import sys; from packaging.version import Version; sys.exit(0 if Version('$TORCH_VER') >= Version('2.10') else 1)"; then
  sed -i "s/-std=c++17/-std=c++20/g" setup.py  # torch >= 2.10 headers need C++20
  echo "== torch $TORCH_VER: building with C++20"
else
  echo "== torch $TORCH_VER: building with C++17"
fi

rm -rf build dist
JOBS="${MAX_JOBS:-$(( $(nproc) / 2 ))}"
EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS="$JOBS" "$PY" setup.py bdist_wheel 2>&1 | tail -n 20
WHL="$(ls -1 "$PWD"/dist/sageattention-*.whl | head -n1)"
[ "${NO_INSTALL:-0}" = 1 ] || "$PY" -m pip install -q --force-reinstall --no-deps "$WHL"

echo "== resource usage of the sm89 FP8 kernels (registers / spills; the split_kv one is the 2nd instantiation with 'true, true>' tail):"
SO="$(find build -name '_qattn_sm89*.so' | head -n1)"
cuobjdump --dump-resource-usage "$SO" 2>/dev/null | grep -A1 "qk_int_sv_f8_attn_kernel\|splitkv_combine_kernel" | grep -E "Function|REG|STACK|SHARED" || echo "(cuobjdump not found or no output)"
echo "== wheel: $WHL"
