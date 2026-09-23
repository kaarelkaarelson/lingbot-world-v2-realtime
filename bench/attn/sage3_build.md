# Sage3: FP4 Blackwell attention build recipe (`sageattn3_blackwell`)

**Status**: Ready to execute on a pod with RTX 5090 (sm_120), CUDA 12.8, torch 2.8.0.
**Author**: Claude Code (2026-09-22). Verified off-pod against `thu-ml/SageAttention` `main`
(`setup.py`, `sageattention3_blackwell/setup.py`, `sageattention3_blackwell/sageattn3/api.py`
fetched and read directly, 2026-09-22). Nothing here was run on a GPU.
**Integration**: `wan/modules/attention.py`, `LINGBOT_ATTN=sage3`, env-gated, default off. See the
patch in that file (search `_SAGE3`) — produced by editing the real tree and `git diff`, not
hand-written.

## Context

`sageattention3_blackwell` is a second, independent package inside the same upstream repo we
already build Sage 2.2 from. It ships FP4 (NVFP4) attention for sm_120/sm_121/sm_100 via
`mma.sync` — no `tcgen05` needed, so it is not gated on a datacenter-class Blackwell part. The
paper (arXiv 2505.11594) claims 1038 TOPS on a 5090, 62 % of the 1676 TFLOP/s dense FP4 peak —
the same utilisation our INT8 kernel already gets, so attention's measured 0.288 s/chunk should
fall to about 0.16 s, chunk 0.98 → 0.85 s, **16.3 → 18.8 FPS**.

## Package collision: verified NO conflict, both installs coexist

Checked directly against the two `setup.py` files:

| | top-level (already installed) | `sageattention3_blackwell/` |
|---|---|---|
| pip distribution name | `sageattention` (2.2.0) | `sageattn3` (1.0.0) — `PACKAGE_NAME = "sageattn3"` in its own `setup.py` |
| python import root | `sageattention` | `sageattn3` (`from sageattn3 import sageattn3_blackwell`) |
| compiled extensions | `sageattention._qattn_sm80/89/90`, `sageattention._fused` (namespaced under the `sageattention` package dir) | `fp4attn_cuda`, `fp4quant_cuda` (top-level extension names — installed as their own top-level `.so` files in site-packages, **not** nested under `sageattn3/`) |
| `install_requires` | none pinned beyond build-time torch | `torch`, `einops`, `packaging`, `ninja` — no dependency on `sageattention` itself |

No shared distribution name, no shared python package name, no shared `.so` name. `pip install`ing
one does not uninstall, upgrade, or shadow the other, and nothing in either `setup.py` scans for
or refuses to coexist with the other. **Our working Sage 2 / `sage_kvq` path keeps running
unmodified** — confirmed by the CPU-only dispatch test in `attention.py` (both `_SAGE` and `_SAGE3`
guarded by disjoint `LINGBOT_ATTN` values, only one import fires per process).

Two friction points found, neither a blocker, both worth checking on the pod before trusting the
build silently succeeded:

1. **`sageattn3/api.py` does `import triton`** but `triton` is absent from `install_requires`. It
   ships bundled with the Linux CUDA wheel of torch 2.8.0 (our pinned version, `requirements.txt`),
   so it should already be present — verify with `python -c "import triton"` before building, don't
   assume.
2. **The README says `python>=3.13`**, but the package's own `setup.py` only enforces
   `python_requires=">=3.8"` — the README requirement is not mechanically enforced. Our venv is
   cp312 (same as the cached Sage 2.2 wheel, `sageattention-2.2.0-cp312-cp312-linux_x86_64.whl`).
   Nothing in `api.py`/`setup.py` uses a 3.13-only language feature as far as static reading shows,
   but this is unverified without an actual build — if `python setup.py install` refuses cp312 for
   a reason unrelated to CUDA, that is the first place to look.

## Build recipe

### Phase 0: Environment verification (CPU only, ~1 min)

```bash
nvcc --version                 # need CUDA >= 12.8 (sageattention3_blackwell/setup.py hard-asserts this)
python --version                # cp312 expected; README says 3.13, setup.py only requires >=3.8
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # want 2.8.0, 12.8
python -c "import triton; print(triton.__version__)"   # must succeed — see friction point 1 above
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader   # want RTX 5090, 12.0
pip show sageattention          # confirm Sage 2 is present and untouched before we start
```

### Phase 1: Build (same venv as Sage 2 — no isolation needed, see collision check above)

```bash
cd /tmp   # or wherever you keep build checkouts; NOT inside this repo
git clone --depth=1 https://github.com/thu-ml/SageAttention.git sage3_build
cd sage3_build/sageattention3_blackwell

# The build vendors NVIDIA cutlass itself via `git clone` inside setup.py — needs network
# access and a few hundred MB of disk; nothing to do manually here.

export TORCH_CUDA_ARCH_LIST="12.0"   # setup.py maps this to `-gencode arch=compute_120a,code=sm_120a`
python setup.py install 2>&1 | tee build_sage3.log

# What success looks like: log ends with a normal setuptools "Finished processing dependencies
# for sageattn3==1.0.0" / "Successfully installed sageattn3-1.0.0", and both extension modules
# import cleanly:
python -c "import fp4attn_cuda, fp4quant_cuda, sageattn3; from sageattn3 import sageattn3_blackwell; print('sage3 OK')"

# Sage 2 must still work — this is the whole point of the collision check above:
python -c "from sageattention import sageattn; print('sage2 OK')"
```

Budget 15–30 minutes: two small `CUDAExtension`s (`fp4attn_cuda`, `fp4quant_cuda`) versus Sage
2.2's much larger `_qattn_sm80/89/90` + `_fused` build (26 min per `OPTIMIZATIONS.md` §opt5), plus
the one-time `git clone` of cutlass.

### If it fails

- **`CUDA 12.0 or higher is required`** / arch error: confirm `nvcc --version` actually reports
  12.8 in *this* shell (a stray `CUDA_HOME` pointing at an older toolkit is the usual cause on a
  pod with multiple CUDA installs — same class of bug as the sm_120 patch notes in
  `c7_sparge_build.md`).
- **cutlass clone fails** (no network from the build sandbox, or git not configured): clone
  `https://github.com/NVIDIA/cutlass.git` manually into
  `sage3_build/sageattention3_blackwell/csrc/cutlass` first (depth 1 is enough — `setup.py` only
  checks the directory exists, it does not pin a commit) and rerun `python setup.py install`.
- **`import triton` fails**: `pip install triton` — do **not** let pip pull a version that
  disagrees with the one torch 2.8.0 bundles; check `pip show torch | grep -i requires` first.
- **Fallback**: stay on `LINGBOT_ATTN=sage` (Sage 2 INT8). Nothing about this build touches that
  path; a failed Sage 3 build is a no-op for the deployed stack.

## Quality gate

### 1. Kernel level (do this first, cheapest, no rollout needed)

Cosine similarity against an fp32 `torch.nn.functional.scaled_dot_product_attention` reference, at
our **exact production shape**: `q [1, 6032, 12, 128]`, `k/v [1, 27144, 12, 128]`, bf16, produced
in NHD `[B, L, H, D]` the way `attention()` receives them, then transposed to HND before the Sage 3
call (as the patch does). `is_causal=False`. This is the same protocol used for every other
attention candidate in this repo (`OPTIMIZATIONS.md` §20): **our INT8 baseline measures 0.999246
on this exact shape.** Sage 3 must be measured fresh here — do not reuse or compare to:

- **the paper's reported 0.9952** — that is the *attention map* (not the output tensor) against a
  *naive* FP4 quantisation baseline, measured on CogVideoX, a different model, different shapes,
  different reference, different metric. It answers "how much does SageAttention3's smoothing/
  online-scaling help over naive FP4", not "how close is this kernel's output to fp32 SDPA at our
  shape". **Not comparable to 0.999246.**

```python
import math, torch
from sageattn3 import sageattn3_blackwell

torch.manual_seed(0)
B, LQ, LKV, H, D = 1, 6032, 27144, 12, 128
q = torch.randn(B, LQ, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, LKV, H, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, LKV, H, D, device="cuda", dtype=torch.bfloat16)

ref = torch.nn.functional.scaled_dot_product_attention(
    q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float(),
    is_causal=False).transpose(1, 2)

out = sageattn3_blackwell(q.transpose(1, 2), k.transpose(1, 2).clone(), v.transpose(1, 2),
                           is_causal=False).transpose(1, 2)

cos = torch.nn.functional.cosine_similarity(ref.flatten().float(), out.flatten().float(), dim=0)
print(f"cosine vs fp32 SDPA: {cos.item():.6f}  (INT8 baseline: 0.999246)")
```

Gate: report the number, do not pre-commit to a pass/fail threshold before seeing it — cosine
naturally degrades with fewer mantissa bits (INT8 → NVFP4 is a much larger step than any INT8
kernel variant in §20's table, all of which moved the 4th decimal). If it lands materially below
0.999, that alone does not kill the candidate (FP4 is expected to be lossier than INT8) but it
does mean the rollout gate below must actually run, not be assumed.

Also verify **synthetic, not-near-uniform inputs** the way `c7_sparge.py` does for C7 — pure
`randn` gives near-uniform softmax and is not representative of real attention mass; if time
allows, reuse that harness's clustered-key generator instead of raw `randn` for a second data
point, but the `randn` number above is the one directly comparable to the 0.999246 baseline
(which was also measured on `randn`, `OPTIMIZATIONS.md` §20).

### 2. Rollout gate (`bench-world-model-quality` skill)

Kernel-level cosine cannot certify the model: this is a **numerical** lever (same keys attended,
lower precision), so it is Class B (metric-equivalent) territory per the skill's classification —
not Class D like the KV-window candidate that was just killed. Concretely:

- Run the noise band first (3× the unoptimised production config, same seed) if it is not already
  fresh for this pod/torch/cuDNN combination.
- One `LINGBOT_ATTN=sage3` run against the deterministic reference protocol (`LINGBOT_DIT_FUSION=1
  LINGBOT_DIT_FUSION_EXACT_T=1 LINGBOT_SYNCFREE=1 LINGBOT_VAE_FUSED=eager LINGBOT_VAE_STREAM=1
  LINGBOT_TORCH_COMPILE= LINGBOT_FP8=0`, same seed/poses/image) is **not** applicable as-is since
  that reference pins `LINGBOT_ATTN=` (empty, i.e. FlashAttention) — use it only for a same-seed,
  divergent-trajectory comparison the way §20's INT8 kernel swaps were judged, not for bit-identity.
- Score with `stream/tools/score_run.py`: first-chunk PSNR/LPIPS, MUSIQ, colour, flicker,
  `lossless_class`. Threshold per the skill's table: first-chunk PSNR inside the measured band,
  MUSIQ ±1.5, colour ±3, flicker ±10%. **Prefer LPIPS as the deciding metric** (skill note,
  2026-09-23): three identical baseline runs already differ by 18–33% on Laplacian sharpness but
  only 0.0070–0.0074 on LPIPS, so LPIPS is the only one of these with enough headroom below the
  noise floor to actually detect an FP4-sized effect.
- n = 1 pod run is enough for go/no-go; a published "lossless" needs 3 scenes × 3 seeds.

## Copy-paste commands for a pod session

```bash
# --- 1. build (Phase 1 above, ~15-30 min) ---
cd /tmp && git clone --depth=1 https://github.com/thu-ml/SageAttention.git sage3_build
cd sage3_build/sageattention3_blackwell
export TORCH_CUDA_ARCH_LIST="12.0"
python setup.py install 2>&1 | tee build_sage3.log
python -c "import fp4attn_cuda, fp4quant_cuda; from sageattn3 import sageattn3_blackwell; from sageattention import sageattn; print('both installs OK')"

# --- 2. kernel-only timing, our exact shape (mirrors bench/attn/h2_tiles/bench_kernel_only.py) ---
cd $REPO   # this repo, e.g. ~/lingbot-world-v2-realtime
python - <<'PY'
import math, time, torch
from sageattn3 import sageattn3_blackwell
dev = torch.device("cuda")
torch.manual_seed(0)
q = torch.randn(1, 12, 6032, 128, device=dev, dtype=torch.bfloat16)   # HND directly for the kernel-only timing
k = torch.randn(1, 12, 27144, 128, device=dev, dtype=torch.bfloat16)
v = torch.randn(1, 12, 27144, 128, device=dev, dtype=torch.bfloat16)
for _ in range(10):
    sageattn3_blackwell(q, k.clone(), v, is_causal=False)
torch.cuda.synchronize()
t0 = time.perf_counter()
N = 30
for _ in range(N):
    sageattn3_blackwell(q, k.clone(), v, is_causal=False)
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / N * 1e3
flops = 4.0 * 6032 * 27144 * 12 * 128
print(f"sage3 wall time: {ms:.4f} ms/call  ({flops/(ms/1e3)/1e12:.1f} TOPS)")
PY

# --- 3. quality gate: kernel-level cosine (paste the Python block from "Quality gate" §1 above) ---

# --- 4. integration bench: our env flag, on the real model ---
LINGBOT_ATTN=sage3 LINGBOT_DIT_FUSION=1 LINGBOT_SYNCFREE=1 LINGBOT_VAE_FUSED=1 \
LINGBOT_FP8=1 LINGBOT_TORCH_COMPILE=1 python bench_perf.py --label sage3_fp4 ...  # match the flags of the current 16.3 FPS baseline row in OPTIMIZATIONS.md

# --- 5. rollout quality (bench-world-model-quality skill; see §2 above for exact protocol) ---
```

## What could not be verified without a GPU

- Whether the build actually succeeds on this exact pod (cutlass clone, `-gencode
  arch=compute_120a`, cp312 vs the README's stated `>=3.13`, triton availability) — Phase 0/1 above
  are designed to surface each of these fast if they are real problems.
- The kernel-level cosine number itself — no number is claimed anywhere in this document; the
  script in "Quality gate §1" must actually be run.
- Wall-clock speedup — the 0.288 s → ~0.16 s arithmetic is the paper's claimed utilisation carried
  over from our own measured INT8 utilisation; it is not confirmed against a measured Sage 3
  kernel time on our shape until step 2 above is run.
- Whether the transpose/clone overhead added by the two landmine fixes (layout transpose in/out,
  K clone) is small relative to the ~0.13 s of headroom being chased — it must be counted, not
  assumed negligible, per the instruction that motivated this build.
- Any interaction with `torch.compile` (graph breaks on the new op, the way Sage 2's fused pybind
  functions are non-custom-ops and would graph-break) — untested here.
