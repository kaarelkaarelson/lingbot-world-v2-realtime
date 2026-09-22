# C7 — SpargeAttn (thu-ml): fewer scores instead of faster scores

SageAttention 2.2 runs at 62–70 % of the INT8 tensor peak and the remainder is the
per-tile softmax dependency chain, which we cannot remove. C7 attacks the other
axis: compute **fewer** scores. SpargeAttn (arXiv 2502.18137, ICML 2025,
<https://github.com/thu-ml/SpargeAttn>) is a training-free block-sparse attention
built directly on top of the SageAttention2 kernels — a selective block mask from
compressed Q/K mean-similarity, plus an online softmax-aware PV filter that skips
blocks whose contribution to the output is negligible.

Harness: `bench/attn/c7_sparge.py` → `bench/attn/results/c7_sparge.json`.
CPU plan check: `python bench/attn/c7_sparge.py --dry` (exits 0, touches no GPU).

## The API (verified against source)

`spas_sage_attn/core.py` — <https://github.com/thu-ml/SpargeAttn/blob/main/spas_sage_attn/core.py>

```python
from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda   # recommended by the repo
from spas_sage_attn import spas_sage2_attn_meansim_cuda        # threshold form
from spas_sage_attn import block_sparse_sage2_attn_cuda        # bring-your-own mask

spas_sage2_attn_meansim_topk_cuda(
    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
    smooth_k=True, simthreshd1=-0.1, cdfthreshd=None, topk=0.5, pvthreshd=50,
    attention_sink=False, tensor_layout="HND", output_dtype=torch.float16,
    return_sparsity=False)

spas_sage2_attn_meansim_cuda(
    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
    smooth_k=True, simthreshd1=0.6, cdfthreshd=0.98, pvthreshd=50,
    attention_sink=False, tensor_layout="HND", output_dtype=torch.float16,
    return_sparsity=False)

block_sparse_sage2_attn_cuda(q, k, v, mask_id=None, ...)   # mask_id: (B, H, ceil(L/128), ceil(L/64)), 0/1
```

Hyperparameters:

| param | meaning | direction |
| --- | --- | --- |
| `topk` | fraction of QK blocks kept by the stage-1 mask | lower = sparser, less accurate |
| `simthreshd1` | mean-similarity threshold for the compressed-Q/K block filter | lower = sparser |
| `cdfthreshd` | CDF mass a query block must cover before blocks are dropped | lower = sparser |
| `pvthreshd` | stage-2 online-softmax PV filter (int, per-head allowed) | higher = sparser |
| `attention_sink` | keeps the first block always dense | relevant to our 6 sink latents |
| `return_sparsity` | returns `(out, qk_sparsity)` — fraction of **QK** blocks skipped; the PV-stage skips are **not** counted | — |

`tensor_layout="NHD"` is supported (it rearranges to HND internally), so our
`[B, L, H, D]` tensors go in unchanged. `is_causal=False` is what we want: the
model is causal at the latent level but attention is dense inside the rolling
27 k-token window. Head dim must be 64 or 128 (ours is 128) and `q.size(-2) >= 128`.

## Is a tuning pass needed? No.

Both `*_topk_cuda` and `*_meansim_cuda` are plug-and-play with global thresholds —
the repo's own README calls this "a very simple usage without tuning or
calibration". The tuning machinery
(`spas_sage_attn.autotune.SparseAttentionMeansim`, driven by `TUNE_MODE=1` /
`PARALLEL_TUNE=1`, with `extract_sparse_attention_state_dict` /
`load_sparse_attention_state_dict` and the per-model `.pt` files in
`evaluate/models_dict/`) only exists to fit **per-head** `simthreshd1`,
`cdfthreshd`, `pvthreshd`, and a per-head `is_sparse` on-off flag, using the model's
own activations as the calibration set. It is a quality optimization on top,
not a precondition for a meaningful kernel number — so the harness reports the
un-tuned grid honestly and does not pretend to be tuned.

If C7 survives the kernel benchmark, the tuning pass is the natural follow-up:
it needs a real forward pass of our transformer with the attention module swapped
for `SparseAttentionMeansim`, which is real integration work, not a bench flag.

## sm_120: NOT supported upstream. Needs a patch.

`setup.py` (<https://github.com/thu-ml/SpargeAttn/blob/main/setup.py>) line 52:

```python
SUPPORTED_ARCHS = {"8.0", "8.6", "8.7", "8.9", "9.0"}
```

There is no `12.0`. On a 5090 the build auto-detects `12.0`, finds no valid arch,
and dies with `RuntimeError: None of the CUDA architectures in TORCH_CUDA_ARCH_LIST
... is supported` — this is open issue
[#109 "5090 black is not support？"](https://github.com/thu-ml/SpargeAttn/issues/109),
alongside [#76 "Support Blackwell family"](https://github.com/thu-ml/SpargeAttn/issues/76).
The only Blackwell mention in the README is a CUDA-version line (`>=12.8 for
Blackwell`), which is aspirational: there are `csrc/qattn/instantiations_sm80`,
`_sm89` and `_sm90` directories and no sm120 one. A community PR,
[#123 "SM120 (Blackwell) support"](https://github.com/thu-ml/SpargeAttn/pull/123),
did the work but **was closed unmerged**.

To build it on the pod, three things must change (this is what #123 did):

1. `setup.py`: add `"12.0"` to `SUPPORTED_ARCHS` and emit arch-specific gencode
   (`arch=compute_120a,code=sm_120a`, not plain `compute_120` — the INT8/FP8 MMA
   the kernels use is an `a`-suffixed feature).
2. `csrc/wgmma.cuh` (and the other SM90 guards): the guard is
   `#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900))`, which is **true**
   for sm_120 (`__CUDA_ARCH__ == 1200`), so Hopper-only `wgmma` gets compiled into
   the Blackwell build and fails. Clamp those guards to `>= 900 && < 1000`.
3. `csrc/mma.cuh`, `csrc/numeric_conversion.cuh`, `csrc/cp_async.cuh`: add
   `#include <cstdint>` (newer CUDA headers no longer pull it in transitively).

Dispatch then works without further changes: `core.py` reads
`torch.cuda.get_device_capability()` → `"sm120"`, which falls through the
`sm80/86/87` and `sm90` branches to the SageAttention2++ fp8 path
(`qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold`),
gated on `SAGE2PP_ENABLED`, which requires CUDA ≥ 12.8 — we have 12.8.

```bash
git clone https://github.com/thu-ml/SpargeAttn && cd SpargeAttn
# apply the three patches above
pip install ninja
TORCH_CUDA_ARCH_LIST="12.0" python setup.py install     # or: pip install -e .
```

Budget 20–40 minutes of compile. If the import fails the harness says so and
reports the dense baseline alone rather than a fabricated number.

## What the harness measures

- **Dense baseline**: `sageattn()` on our exact shape — `q [1, 6032, 12, 128]`,
  `k/v [1, 27144, 12, 128]`, bf16, NHD, `is_causal=False`, `sm_scale=1/sqrt(128)`.
  20 warm / 50 timed, CUDA events. `1.006 TFLOP` per call by the dense formula.
- **SpargeAttn grid**: `topk` ∈ {0.9, 0.7, 0.5, 0.3} and three `(simthreshd1,
  cdfthreshd, pvthreshd)` triples from conservative to aggressive. Each row reports
  ms, **effective TOPS computed with the DENSE flop count** (so a sparse kernel that
  skips half the blocks shows up as ~2× the dense TOPS — the number stays directly
  comparable to the dense baseline and to the INT8 peak), speedup, and
  `qk_block_sparsity` from `return_sparsity=True`.
- **Quality, two references on identical inputs**:
  - vs **fp32 torch SDPA** — total error, quantization + sparsity together;
  - vs **dense `sageattn()`** — the number that actually decides C7, because it
    isolates *what sparsity costs on top of the quantization we already accept*.
  Both as cosine similarity, max abs error, mean abs error and relative L1.

## Synthetic inputs — read this before believing any quality number

The harness generates q/k/v synthetically and **says so in its own output and in
the JSON**. Pure `randn` would be the wrong choice in a way that biases the test
against C7: random q/k give a near-uniform softmax, every block carries the same
mass, there is nothing for a sparse filter to skip, so the speedup collapses and
the error looks terrible. Neither number would transfer.

So the generator builds structure instead: 24 contiguous key clusters per head
(contiguous runs, so the structure lands on block boundaries the way a real video
latent grid does), each query block steered at 3 of them, logits sharpened, plus
`SINK_TOKENS = 6` sink keys parked on a direction every query has a component
along — those take a few percent of every query's mass and are exactly the block a
sparse filter must never drop. Spot-checked on CPU: ~90 % of the softmax mass sits
in the top 10 % of keys and the sinks hold ~1–9 %.

This is a **plausible guess at our attention distribution, not a measurement of it.**
The speed numbers from this harness are meaningful (block sparsity is a real
kernel-level effect, and the dense baseline is our true shape). The quality
numbers are **directional only**. A real go/no-go on C7 requires the exp15 lossless
band measured on generated video through the actual model — synthetic tensor cosine
similarity has no calibrated relationship to perceived video quality, and the
failure mode of sparse attention in a rolling-window video model (drift accumulating
over frames as dropped blocks compound) cannot appear in a single-call tensor test
at all.

## Known risks

- **`q_len != kv_len`.** Our 6032/27144 split is unusual; SpargeAttn's published
  use is self-attention with equal lengths. The block-map builder takes q and k
  independently so it should hold, but neither length is a multiple of the 128×64
  block size (6032 = 47.125 × 128, 27144 = 424.125 × 64). The harness catches per-
  setting exceptions and records them rather than aborting the run.
- **sm_120 is an unmerged community patch**, not upstream support. Numerical
  correctness on Blackwell is unvalidated by the authors; the `vs fp32 SDPA` column
  is the first place a miscompiled kernel would show up.
- **The reported sparsity undercounts.** `return_sparsity` covers the QK-stage
  block mask only; the PV-stage online filter skips more, unreported. Trust the
  wall-clock, not the sparsity percentage.
