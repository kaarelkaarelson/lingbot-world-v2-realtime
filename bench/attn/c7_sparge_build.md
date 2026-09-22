# C7 SpargeAttn: sm_120 Build Debug Recipe

**Status**: Ready to execute on a pod with RTX 5090, CUDA 12.8, gcc 13.3.
**Author**: Claude Code (2026-09-22), corrected after review
**Note**: SpargeAttn patch is OURS (commit bf13a9a), unverified. Upstream PR #123 was closed unmerged.

## Context

SpargeAttn's `setup.py` refuses sm_120 (RTX 5090 / Blackwell). After forcing the arch with the patch at
`bench/attn/sparge/sm120.patch`, the build dies at the host-pass C++ compilation with:

```
redefinition of std::__terminate in libstdc++'s c++config.h
```

**Key observation**: SageAttention (same vendor, same toolchain) builds clean on gcc 13.3 + CUDA 12.8.
The critical difference: **SageAttention has NO `-include,cassert` flag. SpargeAttn does.**

Our patch USED TO contain (setup.py, lines 106-109):
```python
if nvcc_cuda_version < Version("13"):
    NVCC_FLAGS += ["-Xcompiler", "-include,cassert"]
```

We are on CUDA 12.8 (< 13), so **the flag was STILL ADDED on our box**: the gate targets CUDA 13+ and did
nothing for us. The patch was written against the wrong axis, most likely gcc version rather than CUDA
version.

**Fixed 2026-09-22 in `bench/attn/sparge/sm120.patch`: the flag is now dropped unconditionally.** Verified
off-pod by cloning upstream at `ae5b629`, applying the patch, and confirming `grep cassert setup.py` returns
only the explanatory comment and that `setup.py` still parses. So step 1 below needs no edit, just a build.

**Live suspects, in order to test** (cheapest first):
1. **FIRST (5 min)**: Remove the `-include,cassert` flag unconditionally. SageAttention does not have it and builds.
2. **SECOND (15 min)**: The venv was created with `--system-site-packages`, layering two torch include trees.
   This may independently trigger double-parsing of headers.
3. **GUARDRAIL (15 min)**: An unpatched arch 8.9 baseline build was never attempted to prove the build works before changing arch.

## Build Recipe: Four-Phase Debug

**Rationale for ordering**: Phase 1 (cassert removal) is the cheapest and most probable cause.
Phase 2 (clean venv) is cheap but requires rebuilding the venv. Phase 3 (baseline) validates the toolchain.
All can run in parallel by fetching two worktrees if pod has time.

### Phase 0: Environment Verification (CPU only, ~1 min)

```bash
# On pod with RTX 5090
export HOME=$(pwd)  # or use real home if in /tmp
cd /tmp/sparge-build
mkdir -p /tmp/sparge-build
cd /tmp/sparge-build

# Verify toolchain
nvcc --version          # should show CUDA 12.8
gcc --version           # should show gcc 13.3.x
python --version        # should show 3.10 or 3.11
python -c "import torch; print(f'torch {torch.__version__}, cuda {torch.version.cuda}')"

# Check pod GPU
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# Expected: NVIDIA RTX 5090, 12.0
```

### Phase 1: Remove `-include,cassert` Unconditionally (Tests Suspect: cassert flag toxicity)

**Goal**: Test whether removing the `-include,cassert` flag (which SageAttention lacks) allows sm_120 to build.
This is the fastest test; SageAttention is proof that absence of this flag works on the same machine.

```bash
cd /tmp/sparge-build
git clone --depth=1 https://github.com/thu-ml/SpargeAttn.git sparge_nocassert
cd sparge_nocassert

# Apply our patch first
patch -p1 < /path/to/bench/attn/sparge/sm120.patch

# NOW: Remove the `-include,cassert` lines from NVCC_FLAGS entirely (not conditionally).
# Edit setup.py around lines 106-109:
# REPLACE THIS:
#   if nvcc_cuda_version < Version("13"):
#       NVCC_FLAGS += ["-Xcompiler", "-include,cassert"]
# WITH THIS (delete all 3 lines):
#   # cassert flag removed: SageAttention builds without it

# Or use sed to remove it:
sed -i.bak '106,109d' setup.py  # Delete lines 106-109
# Verify the deletion
grep -n "include,cassert" setup.py && echo "✗ FAILED TO REMOVE CASSERT" || echo "✓ Cassert removed"

# Build at arch 12.0
export TORCH_CUDA_ARCH_LIST="12.0"
python setup.py build 2>&1 | tee build_sm120_nocassert.log

# Outcome
if grep -q "Successfully installed" build_sm120_nocassert.log; then
    echo "✓ SM_120 BUILD WITHOUT CASSERT FLAG PASSED"
    echo "→ The -include,cassert flag is the culprit; it triggers redefinition with gcc 13.3"
else
    tail -50 build_sm120_nocassert.log | grep -A 5 "redefinition\|error:" | head -20
    if grep -q "__terminate" build_sm120_nocassert.log; then
        echo "✗ Still __terminate redefinition WITHOUT cassert flag"
        echo "→ Cassert is NOT the cause; move to Phase 2 (venv isolation)"
    else
        echo "✗ Different error; inspect log above"
    fi
fi
```

**Validation Checklist for Phase 1**:
- [ ] cassert flag successfully removed via sed
- [ ] sm_120 build succeeds → **flag is the culprit; fix is to remove it**
- [ ] sm_120 build still fails with __terminate → move to Phase 2 (venv isolation)

### Phase 2: Clean Venv WITHOUT `--system-site-packages` (Tests Suspect: double torch includes)

**Goal**: If Phase 1 still fails, isolate whether the venv is layering two torch include trees,
which may independently trigger double-parsing of headers.

```bash
cd /tmp/sparge-build
rm -rf venv_clean
python -m venv venv_clean    # NO --system-site-packages

source venv_clean/bin/activate
pip install --upgrade pip setuptools wheel ninja
pip install 'torch @ file:///path/to/torch-2.11-nightly-cu12.8.whl'  # or torch==2.11-nightly
pip install peft einops packaging  # SpargeAttn dependencies

cd /tmp/sparge-build
rm -rf sparge_clean_venv
git clone --depth=1 https://github.com/thu-ml/SpargeAttn.git sparge_clean_venv
cd sparge_clean_venv

# Apply the patch
patch -p1 < /path/to/bench/attn/sparge/sm120.patch

# Build at arch 12.0
export TORCH_CUDA_ARCH_LIST="12.0"
python setup.py build 2>&1 | tee build_sm120_clean_venv.log

# Outcome
if grep -q "Successfully installed" build_sm120_clean_venv.log; then
    echo "✓ SM_120 BUILD IN CLEAN VENV PASSED"
    echo "→ The original venv's --system-site-packages layered torch includes"
else
    echo "✗ SM_120 still fails in clean venv"
    grep -A 5 "redefinition\|error:" build_sm120_clean_venv.log | head -20
fi
```

**Validation Checklist for Phase 2**:
- [ ] Clean venv has only ONE torch include tree → `python -c "import torch; print(torch.utils.cpp_extension.include_paths())"`
- [ ] sm_120 build succeeds → original venv was misconfigured
- [ ] sm_120 build fails with same error → move to Phase 3 (baseline validation)

### Phase 3: Baseline Unpatched Build at Arch 8.9 (Guardrail: Tests Suspect B)

**Goal**: Confirm the toolchain works on upstream SpargeAttn at arch 8.9 before debugging sm_120.
This guarantees that gcc 13.3 + CUDA 12.8 + SpargeAttn can build *something*.

```bash
cd /tmp/sparge-build
rm -rf sparge_baseline
git clone --depth=1 https://github.com/thu-ml/SpargeAttn.git sparge_baseline
cd sparge_baseline

# NO PATCH — use upstream unmodified
# Force arch 8.9, do NOT apply sm_120 patch

export TORCH_CUDA_ARCH_LIST="8.9"
python setup.py build 2>&1 | tee build_baseline_8.9.log

# Outcome
if grep -q "Successfully installed" build_baseline_8.9.log; then
    echo "✓ BASELINE ARCH 8.9 BUILD PASSED"
    echo "→ Toolchain is sound; any failure is in our patch or its interaction with sm_120"
else
    echo "✗ Baseline failed"
    tail -50 build_baseline_8.9.log
    echo "→ Toolchain itself is broken on this pod; cannot proceed"
    exit 1
fi
```

**Validation Checklist for Phase 3**:
- [ ] Baseline 8.9 build succeeds → toolchain is not globally broken
- [ ] Baseline fails → GPU pod or compiler setup is misconfigured; escalate

## Expected Outcomes

| Phase | Outcome | Interpretation | Next Step |
|-------|---------|-----------------|-----------|
| 0 | ✓ GPU is sm_120, CUDA 12.8 | Environment is correct | Proceed to Phase 1 |
| 1 | ✓ sm_120 builds WITHOUT cassert | **Flag is toxic.** Remove it; candidate buildable. | Go to integration: `bench/attn/c7_sparge.py` |
| 1 | ✗ Still `__terminate` without cassert | Cassert is NOT the cause; venv isolation may be. | Go to Phase 2 |
| 2 | ✓ sm_120 builds in clean venv | **Original venv layered includes.** Use clean venv; candidate buildable. | Go to integration: `bench/attn/c7_sparge.py` |
| 2 | ✗ Still `__terminate` in clean venv | Both suspected causes ruled out; toolchain issue. | Go to Phase 3 |
| 3 | ✓ Baseline 8.9 unpatched builds | Toolchain works; problem is in our patch or sm_120 target. Investigate further. | Debug patch in detail or escalate |
| 3 | ✗ Baseline 8.9 fails | Toolchain itself broken; pod or compiler misconfigured. | Escalate; cannot proceed on this pod. |

## Copy-Paste Commands for Quick Iteration

```bash
# Phase 1: Remove cassert and test
cd /tmp/sparge-build/sparge_nocassert && \
patch -p1 < /path/to/bench/attn/sparge/sm120.patch && \
sed -i.bak '106,109d' setup.py && \
export TORCH_CUDA_ARCH_LIST="12.0" && \
python setup.py build 2>&1 | tee build_nocassert.log && \
tail -5 build_nocassert.log

# Phase 2: Clean venv + patched build
cd /tmp/sparge-build && \
rm -rf venv_test && \
python -m venv venv_test && \
source venv_test/bin/activate && \
pip install -q pip setuptools wheel ninja torch peft einops packaging && \
git clone --depth=1 https://github.com/thu-ml/SpargeAttn.git sparge_clean && \
cd sparge_clean && \
patch -p1 < /path/to/bench/attn/sparge/sm120.patch && \
export TORCH_CUDA_ARCH_LIST="12.0" && \
python setup.py build 2>&1 | tee build_clean.log && \
tail -5 build_clean.log

# Phase 3: Baseline 8.9 unpatched
cd /tmp/sparge-build && \
git clone --depth=1 https://github.com/thu-ml/SpargeAttn.git sparge_base && \
cd sparge_base && \
export TORCH_CUDA_ARCH_LIST="8.9" && \
python setup.py build 2>&1 | tee build_base.log && \
tail -5 build_base.log

# Extract error signature (if failure)
tail -100 build_nocassert.log | grep -A 10 "error:\|redefinition" | head -20
```

## Known Workarounds (If All Phases Fail)

1. **Manual header sanitization**: If double-include persists, edit `csrc/*.cuh` to add include guards:
   ```cpp
   #ifndef __INCLUDE_GUARD_MYCUH_
   #define __INCLUDE_GUARD_MYCUH_
   // ... content
   #endif
   ```
   (Though SpargeAttn headers should already have `#pragma once`.)

2. **Fallback: Use pre-built wheel** from [mobcat40/sageattention-blackwell](https://github.com/mobcat40/sageattention-blackwell)
   which claims to handle the build issues, though it is SageAttention, not SpargeAttn.

3. **Minimal reproduction**: If the error persists, isolate with:
   ```bash
   nvcc -x cu -std=c++17 -I/path/to/torch/include -I/path/to/cuda/include \
        csrc/mma.cuh -c -o /dev/null 2>&1
   ```

## Stopping Condition

- **Do not spend more than 90 minutes on this phase.**
- If baseline 8.9 + clean venv + suspect B/C investigation do not resolve it, the candidate is blocked
  and the effort is better spent on other optimizations.

---

## Summary Table: Verdict Path

| Build Outcome | Suspect A | Suspect B | Suspect C | Verdict |
|---|---|---|---|---|
| ✓ Phase 1 sm_120 clean | ruled out | ruled out | ruled out | **BUILDABLE** — move to bench |
| ✗ Phase 1; ✓ baseline 8.9; gencode error in Phase 2 | ruled out | ruled out | **CONFIRMED** | CUDA 12.9+ required; candidate viable if upgraded |
| ✗ Phase 1; ✓ baseline 8.9; __terminate in Phase 2/3 | ruled out | ruled out | ruled out | Header conflict; may need upstream fix |
| ✗ Phase 1 baseline 8.9 | — | **NOT ruled out** | — | Toolchain broken; cannot test |
