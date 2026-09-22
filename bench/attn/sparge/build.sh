#!/usr/bin/env bash
# Build thu-ml/SpargeAttn @ ae5b629 for sm_120 on the pod.
#   bash bench/attn/sparge/build.sh
# Result: /workspace/sparge/dist/spas_sage_attn-*.whl installed into venv /workspace/sparge/venv_sparge
# (created with --system-site-packages CLONED from /workspace/sage_h2/venv_cfg_base so the pod's torch and
# the working sageattention are reused; the shipped installs are untouched).
# Prints the wheel path and the per-kernel register/smem usage of the sm89-family fp8 block-sparse kernel.
# Then: /workspace/sparge/venv_sparge/bin/python bench/attn/sparge/bench.py
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$HERE/sm120.patch"
[ -f "$PATCH" ] || { echo "no such patch: $PATCH" >&2; exit 2; }
ROOT="${SPARGE_ROOT:-/workspace/sparge}"
SRC="$ROOT/src"
VENV="$ROOT/venv_sparge"
BASE_VENV="${SAGE_BASE_VENV:-/workspace/sage_h2/venv_cfg_base}"
COMMIT=ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a
PY="${PYTHON:-python3}"
export PATH="/usr/local/cuda/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-16}"

mkdir -p "$ROOT"
if [ ! -d "$SRC/.git" ]; then
  git clone -q https://github.com/thu-ml/SpargeAttn.git "$SRC"
fi
cd "$SRC"
git fetch -q origin
git checkout -q "$COMMIT"
git reset -q --hard "$COMMIT"
git clean -qfdx
git apply --check "$PATCH"
git apply "$PATCH"
echo "== applied $PATCH on $(git rev-parse --short HEAD)"
grep -n "SUPPORTED_ARCHS" setup.py
grep -n "120a" setup.py

# The venv must see the SAME torch + the working sageattention as cfg_base, so we build the venv
# with --system-site-packages and additionally point it at cfg_base's site-packages.
if [ ! -x "$VENV/bin/python" ]; then
  "$PY" -m venv --system-site-packages "$VENV"
fi
if [ -d "$BASE_VENV" ]; then
  BASE_SP="$("$BASE_VENV/bin/python" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
  VENV_SP="$("$VENV/bin/python" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
  echo "$BASE_SP" > "$VENV_SP/_sage_cfg_base.pth"
  echo "== reusing base venv site-packages: $BASE_SP"
else
  echo "== WARNING: base venv $BASE_VENV not found; falling back to system site-packages only"
fi
"$VENV/bin/python" -m pip install -q --upgrade pip setuptools wheel packaging ninja
"$VENV/bin/python" -m pip install -q einops
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
"$VENV/bin/python" -c "import sageattention, os; print('sageattention', os.path.dirname(sageattention.__file__))"
nvcc --version | tail -2

rm -rf build dist
time "$VENV/bin/python" setup.py bdist_wheel 2>&1 | tail -n 25
WHL="$(ls -1 "$SRC"/dist/spas_sage_attn-*.whl | head -n1)"
"$VENV/bin/python" -m pip install -q --force-reinstall --no-deps "$WHL"

echo "== SASS targets present in the built extension:"
SO="$("$VENV/bin/python" -c "import spas_sage_attn._qattn as m; print(m.__file__)")"
cuobjdump --list-elf "$SO" 2>/dev/null | head -n 5 || true

echo "== resource usage of the fp8 block-sparse kernel (registers / smem / spills):"
cuobjdump --dump-resource-usage "$SO" 2>/dev/null \
  | grep -A4 -i "block_sparse_attn_kernel" \
  | grep -E "Function|REG|SHARED|STACK|SPILL" | head -n 40 \
  || echo "(cuobjdump not found or no matching kernel)"

echo "== wheel: $WHL"
echo "== venv:  $VENV"
echo "== next:  $VENV/bin/python $HERE/bench.py"
