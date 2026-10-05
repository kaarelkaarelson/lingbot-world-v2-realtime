#!/usr/bin/env bash
# Build one H2 tile configuration of SageAttention (thu-ml/SageAttention @ d1a57a5) for sm_120 on the pod.
#   bash bench/attn/h2_tiles/build.sh cfg_a          # cfg_base | cfg_a | cfg_b | cfg_c | cfg_d
# Result: /workspace/sage_h2/<cfg>/dist/sageattention-*.whl installed into venv /workspace/sage_h2/venv_<cfg>
# (created with --system-site-packages so the pod's torch is reused; the shipped sageattention in the
# system site-packages is untouched). Prints the wheel path and the per-kernel register/smem usage.
# Then: /workspace/sage_h2/venv_<cfg>/bin/python bench/attn/h2_tiles/bench.py --cfg <cfg>
set -euo pipefail
CFG="${1:?usage: build.sh <cfg>  (cfg_base|cfg_a|cfg_b|cfg_c|cfg_d)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$HERE/$CFG.patch"
[ -f "$PATCH" ] || { echo "no such patch: $PATCH" >&2; exit 2; }
ROOT="${SAGE_H2_ROOT:-/workspace/sage_h2}"
SRC="$ROOT/$CFG"
VENV="$ROOT/venv_$CFG"
COMMIT=d1a57a5
PY="${PYTHON:-python3}"
export PATH="/usr/local/cuda/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-16}"

mkdir -p "$ROOT"
if [ ! -d "$SRC/.git" ]; then
  git clone -q https://github.com/thu-ml/SageAttention.git "$SRC"
fi
cd "$SRC"
git checkout -q "$COMMIT"
git reset -q --hard "$COMMIT"
git clean -qfdx
git apply --check "$PATCH"
git apply "$PATCH"
echo "== $CFG: applied $PATCH on $(git rev-parse --short HEAD)"
grep -E "^#define SAGE_H2_" csrc/qattn/h2_tiles.h

if [ ! -x "$VENV/bin/python" ]; then
  "$PY" -m venv --system-site-packages "$VENV"
fi
"$VENV/bin/python" -m pip install -q --upgrade pip setuptools wheel packaging ninja
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

rm -rf build dist
time "$VENV/bin/python" setup.py bdist_wheel 2>&1 | tail -n 20
WHL="$(ls -1 "$SRC"/dist/sageattention-*.whl | head -n1)"
"$VENV/bin/python" -m pip install -q --force-reinstall --no-deps "$WHL"

echo "== resource usage of the sm89 FP8 kernels (registers / smem / spills):"
SO="$("$VENV/bin/python" -c "import sageattention._qattn_sm89 as m; print(m.__file__)")"
cuobjdump --dump-resource-usage "$SO" 2>/dev/null | grep -B1 -A1 "qk_int_sv_f8_attn_kernel" | grep -E "Function|REG|SHARED|STACK" | head -n 40 || echo "(cuobjdump not found or no output)"

"$VENV/bin/python" - <<'PYEOF'
import sageattention._qattn_sm89 as m
print("== compiled h2_tile_config (CTA_Q, WARP_Q, CTA_K):", m.h2_tile_config())
PYEOF
echo "== wheel: $WHL"
echo "== venv:  $VENV"
