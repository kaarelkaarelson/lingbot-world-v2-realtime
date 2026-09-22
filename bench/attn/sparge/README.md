# SpargeAttn on sm_120 — build, benchmark, and the honest prior

**Status: not yet run on hardware.** Everything here is CPU-prepared and verified by
`git apply --check` only. Nothing in this directory has touched a GPU.

## What SpargeAttn is

thu-ml/SpargeAttn (arXiv [2502.18137](https://arxiv.org/abs/2502.18137)) is a *training-free
sparse* attention that sits on top of SageAttention2's quantised kernels. Per call it:

1. builds a **block map** over 128×64 (Q×K) tiles by a mean-similarity / CDF test on the
   per-block mean of Q and K (`get_block_map_meansim_fuse_quant` in `spas_sage_attn/utils.py`);
2. compacts the surviving blocks into a **LUT** + `valid_block_num`;
3. runs the SageAttention2 int8-QK / fp8-PV block-sparse kernel over only those blocks,
   with a second online **`pvthreshd`** gate that skips PV accumulation for blocks whose
   softmax mass turns out negligible.

So the win is entirely "how many 128×64 blocks can be skipped". There is no algorithmic
speedup per retained block over dense SageAttention2 — the retained blocks run the *same*
kernel family.

## API

Upstream default branch, commit `ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a`.
Exports (`spas_sage_attn/__init__.py`):

| function | recommended by upstream | selection knob |
|---|---|---|
| `spas_sage2_attn_meansim_topk_cuda` | **yes** | `topk` (explicit density budget) |
| `spas_sage2_attn_meansim_cuda` | no | `simthreshd1`, `cdfthreshd` |
| `spas_sage_attn_meansim_topk_cuda` | yes (SageAttention**1** backend) | `topk` |
| `spas_sage_attn_meansim_cuda` | no | thresholds |
| `block_sparse_sage2_attn_cuda` | — | caller-supplied `mask_id` |

Signature (`core.py:106`):

```python
spas_sage2_attn_meansim_topk_cuda(
    q, k, v,
    attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
    smooth_k=True,              # subtract per-head K mean before the similarity test
    simthreshd1=-0.1,           # mean-cosine threshold for block pruning (per head when tuned)
    cdfthreshd=None,            # softmax-CDF coverage target (per head when tuned)
    topk=0.5,                   # fraction of 128x64 blocks to keep
    pvthreshd=50,               # online PV-skip gate (per head when tuned)
    attention_sink=False,       # always keep the first block(s)
    tensor_layout="HND",        # we pass "NHD"
    output_dtype=torch.float16,
    return_sparsity=False,      # -> (o, qk_sparsity) ; qk_sparsity = fraction of blocks skipped
)
```

Constraints worth knowing: `assert q.size(-2) >= 128` (`core.py:42`); inputs are cast to
bf16/fp16 internally (V always to fp16); `@torch.compiler.disable` on every entry point, so
it will graph-break inside a compiled model.

## Is tuning required?

**Two different answers, and the distinction is the whole story.**

- **`topk` needs no tuning.** It is a direct density budget: `topk=0.5` keeps ~50% of blocks.
  A speed number at a given `topk` is meaningful immediately. What you still have to check
  is the *error* at that budget — the budget is free to set, not free to pay for.
- **`simthreshd1` / `cdfthreshd` / `pvthreshd` do need tuning**, and they are *per attention
  head*. Upstream's offline tuner is `spas_sage_attn/autotune.py`
  (`SparseAttentionMeansim.tune_cdfthreshd` at line 175, `.tune_pvthreshd` at line 146):
  it binary-searches each threshold per head against a **real model forward pass** under an
  L1-error target and saves a `.pt` checkpoint. The README's tuning flow
  (`--tune` / `--parallel_tune` on `evaluate/cogvideo_example.py`) and the model zoo
  (`Xiang-cd/sparge-attention-model-zoo`) are exactly this artefact.
  Calling `spas_sage2_attn_meansim_cuda` with the module defaults
  (`simthreshd1=0.6, cdfthreshd=0.98, pvthreshd=50`) is **not** a tuned configuration and its
  speed/quality point tells you nothing about a real deployment.

`bench.py` therefore reports the `topk` sweep as the primary result, labels the untuned
`meansim` point explicitly, and refuses to present it as a deployable number. It does not run
the tuner, because the tuner has nothing to tune against on synthetic tensors.

## sm_120 status

Upstream does **not** support sm_120. `setup.py:51`:

```python
SUPPORTED_ARCHS = {"8.0", "8.6", "8.7", "8.9", "9.0"}
```

This is a stale list, not a hardware capability gate — `get_torch_arch_list()` intersects
`TORCH_CUDA_ARCH_LIST` with it and raises
`"None of the CUDA architectures in TORCH_CUDA_ARCH_LIST env variable (12.0) is supported"`
when nothing survives.

Why the kernels are actually fine on sm_120 (verified at `ae5b629`):

- the sm80/sm89 kernels use `mma.sync` `m16n8k32` s8 / e4m3 (`csrc/mma.cuh:45` and `:51`,
  guarded `__CUDA_ARCH__ >= 890`) — valid on sm_120;
- `csrc/wgmma.cuh` (Hopper `wgmma`, genuinely sm_90-only) is reached only via
  `csrc/qattn/qk_int_sv_f8_cuda_sm90.cu`, which `setup.py:182` adds **only** `if HAS_SM90`.
  Both files' `SM90_ENABLED` guards were open-ended `__CUDA_ARCH__ >= 900`
  (`csrc/wgmma.cuh:22`, `csrc/qattn/qk_int_sv_f8_cuda_sm90.cuh:28`), which would admit
  sm_120 if that TU were ever pulled in — the patch clamps them to `< 1000`.

**Prior art: [PR #123 "SM120 Blackwell support"](https://github.com/thu-ml/SpargeAttn/pull/123)**
(closed unmerged). It reports a full build, 26 passing tests, and **1.44× sparse-vs-dense at
seq ≥ 8192** on sm_120. `sm120.patch` reuses its source changes verbatim; the PR's bench
scripts, result JSONs and planning docs are not carried over.

### `sm120.patch`

Generated with `git diff` against `ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a`;
`git apply --check` passes on a clean checkout of that commit.

| file | change |
|---|---|
| `setup.py` | `SUPPORTED_ARCHS` += `"12.0", "12.1"` |
| `setup.py` | map `120` → gencode `compute_120a/sm_120a`, `121` → `121a` (arch-accelerated target) |
| `setup.py` | hard error if CC 12.x is requested on nvcc < 12.8 |
| `setup.py` | `-Xcompiler -include,cassert` made conditional on nvcc < 13 (no-op on our CUDA 12.8; prevents a `c++config.h` double-include on CUDA 13) |
| `setup.py` | `os.system("python …")` → `sys.executable` — **load-bearing for `build.sh`**, otherwise the instantiation generators run under the wrong interpreter inside a venv |
| `csrc/mma.cuh`, `csrc/cp_async.cuh`, `csrc/numeric_conversion.cuh` | `#include <cstdint>` (+`<cstddef>`) — newer toolkits dropped the transitive include |
| `csrc/wgmma.cuh`, `csrc/qattn/qk_int_sv_f8_cuda_sm90.cuh` | clamp `SM90_ENABLED` to `900 ≤ arch < 1000` |
| `spas_sage_attn/core.py` | `is_causal` passthrough — see below |

The `core.py` hunk is **not** needed to build. It is included because leaving it out means
`is_causal=True` is silently wrong; see the landmine.

## Landmine: `is_causal` is hardcoded `False`

At `ae5b629`, **every** arch branch of the sage2 entry points passes a literal `False` as the
kernel's causal flag:

- `spas_sage2_attn_meansim_cuda`: `core.py:74` (sm80/86/87), `:88` (sm90), `:90` (Sage2++), `:92` (sm89-family)
- `spas_sage2_attn_meansim_topk_cuda`: `core.py:140`, `:154`, `:156`, `:158`
- `block_sparse_sage2_attn_cuda`: `core.py:207`, `:221`, `:223`, `:225` (this function has no
  `is_causal` parameter at all — masking comes from `mask_id`, so `False` is correct there)

Correction to an earlier review note: the sm80 path does **not** pass `is_causal` through
either — `core.py:74` is `1, False, 1, scale, 0`. All sage2 paths are affected, not just the
fp8 ones. The Python-level `is_causal` reaches only the block-map builder
(`get_block_map_meansim*`), never the kernel, so upstream computes a **non-causal** result
while the caller believes it asked for causal.

Our own use is genuinely non-causal *at the kernel* (the causal structure is expressed by the
rolling KV window, not by a triangular mask inside the call), so this does not bite us.
`sm120.patch` fixes the 8 sage2 sites anyway, and `bench.py --is-causal` prints a loud warning.

## Files

- `sm120.patch` — the diff (7 files, +29/−13), base `ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a`
- `build.sh` — clone at that SHA into `/workspace/sparge`, apply, build a wheel into
  `/workspace/sparge/venv_sparge` (`--system-site-packages`, pointed at
  `/workspace/sage_h2/venv_cfg_base`'s site-packages so torch and the working
  `sageattention` are reused and the shipped install is untouched). Prints the wheel path,
  `cuobjdump --list-elf`, and `cuobjdump --dump-resource-usage` for the block-sparse kernel.
- `bench.py` — the benchmark; `--dry` runs on CPU and exits 0.

```bash
bash bench/attn/sparge/build.sh
/workspace/sparge/venv_sparge/bin/python bench/attn/sparge/bench.py
```

`bench.py` measures, on q `[1,6032,12,128]` / k,v `[1,27144,12,128]` bf16 NHD,
`sm_scale = 1/√128`, 20 warmup + 50 timed with CUDA events:

- dense `sageattn()` baseline;
- `spas_sage2_attn_meansim_topk_cuda` at `topk ∈ {0.9, 0.75, 0.5, 0.25}`;
- `spas_sage2_attn_meansim_cuda` with untuned defaults, labelled as such.

For each: ms, effective TOPS against the **dense** flop count
`4·6032·27144·12·128 = 1.006 Tflop`, `qk_sparsity` from `return_sparsity=True`, and cosine +
max-abs error against **both** an fp32 torch SDPA reference (computed head-by-head — a full
fp32 score matrix is ~7.9 GB) and the dense `sageattn` output.

## Honest expectation

**The prior is that there is little left for SpargeAttn to skip, and I expect no useful win.**

Our attention is a 27k-token **rolling KV window with a 6-latent sink** in a **causal video
world model**. That window is *already a pruned set*: the model only ever sees the recent past
plus the sink, and everything outside has been dropped by construction. SpargeAttn's job is to
find redundancy *within* the set it is given. We hand it a set from which the obvious
redundancy has already been removed, so it is looking for second-order redundancy in an
already-selected window.

Against that, **every published SpargeAttn evaluation is bidirectional diffusion** — CogVideoX,
Wan2.1, Flux, full-sequence DiT attention where each query attends to the entire clip and large
contiguous regions of the score matrix are genuinely near-zero. That is the regime where a
mean-similarity block test finds 50–80% skippable blocks. Our regime is the opposite end.

Two further costs push the same way:

- the block-map build (`get_block_map_meansim_fuse_quant` + LUT construction) is *per call*
  overhead that dense SageAttention does not pay, and it does not shrink with sparsity;
- at Q=6032 the block grid is 48 × 425 tiles — not large enough for the LUT-compaction
  overhead to disappear into the kernel time the way it does at the ≥8192 lengths PR #123 used
  for its 1.44× claim.

### What would change this conclusion

A result is worth acting on only if **all** of these hold, measured with `--qkv` on a **real**
activation dump (synthetic Gaussian q/k/v have no structure to find, so a null result on random
inputs proves nothing — it is a lower bound on sparsity, not evidence):

1. `qk_sparsity ≥ 0.30` at a `topk` budget the error can afford — i.e. the method finds real
   redundancy in the rolling window, not just whatever the budget forced it to drop;
2. wall-clock **≥ 1.25×** faster than dense `sageattn` end-to-end *including* the block-map
   build, at our exact shapes — not extrapolated from a longer sequence;
3. cosine vs. the dense `sageattn` output **≥ 0.999** and max-abs error within the noise band
   that dense SageAttention itself already introduces against fp32 SDPA. If SpargeAttn's error
   against dense is comparable to dense's error against fp32, the extra approximation is free;
   if it is an order of magnitude worse, it is not.

Failing (1) is the expected outcome and means the window is already tight — which is itself a
useful confirmation that the rolling-window design is doing its job, and closes this line of
work. Failing (2) while passing (1) means the redundancy is real but the block-map overhead
eats it, which would point at fusing the map build rather than at abandoning the approach.

A `topk` sweep that shows speed rising smoothly with sparsity while cosine-vs-dense stays
≥0.999 down to `topk=0.5` would be the surprise, and would justify capturing a real activation
dump and running the per-head tuner against the actual model.
