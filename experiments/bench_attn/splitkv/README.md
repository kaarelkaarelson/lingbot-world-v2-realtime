# Split-KV for the SageAttention sm89 kernel on sm_120

Nothing here was compiled or run: the machine this was written on has no nvcc and no GPU. `build.sh` and `check.py` are the first test.

## Problem

`sageattn(q, k, v, "HND")` on card 0 (q `[1,10,6032,128]`, k/v `[1,10,27144,128]`) launches `qk_int_sv_f8_attn_kernel<128,64,32,64,128,...>`: 48 q-tiles x 10 heads = 480 CTAs. At 255 registers and 4 warps, 2 CTAs fit per SM, so 340 slots on 170 SMs. That is 1.41 waves: the second round runs 140 CTAs (41 % of the slots; 1.41 of 2 rounds is the 71 % utilisation) while the first round is full. Each CTA walks all 425 K/V tiles. Capping registers was measured 32-55 % slower (spills).

Split-KV gives each CTA a contiguous range of K/V tiles. With S=2 the grid is 960 CTAs of half the work: 2.82 waves, 3 rounds of T/2 instead of 2 rounds of T.

## Source version

thu-ml/SageAttention `main` at `d1a57a5` ("Update README", 2026-01-17), with `patches/sageattention-current-stream.patch` applied first. `splitkv.patch` is a diff against that state (`git apply --check` passed on a fresh clone with the stream patch applied). It touches two files:

- `csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh`: new template parameter and kernel arguments, partial epilogue, new `splitkv_combine_kernel`.
- `csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu`: the launcher for the model's path (`pv_accum_dtype="fp32+fp16"`), plus a `try_splitkv` helper.

No new files, so `setup.py` is unchanged. The other six sm89 launchers are untouched.

## Interface

`SAGE_SPLITKV=S` in the environment, read on every call (so one process can compare S values). Unset or `1`: the original path. There is no Python or pybind signature change, so `sageattn()` and any direct `sm89_compile.*` caller (such as `wan/modules/sage_kvq.py`) pick it up unchanged.

Rules:

- `S` is clamped to the number of K/V tiles; `tiles_per_split = ceil(tiles / S)` and the effective S is recomputed so no split is empty (S=3 on 425 tiles gives 142/142/141).
- If the effective S is 1 (one K/V tile), the original kernel runs.
- `is_causal=1` or `return_lse=1` ignore the variable (with a one-time warning). Only this launcher has the split path.

## Default behaviour is unchanged

- `split_kv` is a new trailing kernel template parameter, default `false`. Every existing instantiation keeps its code: the new code sits behind `if constexpr (split_kv)` and `split_kv ? ... : ...` on compile-time constants.
- The four new kernel arguments (`O_part`, `ML_part`, `num_splits`, `tiles_per_split`) have default values, so the six other launchers need no edit. This relies on default arguments on a `__global__` function being accepted with `<<<>>>` (it is standard nvcc behaviour, but see risks).
- The split kernel is a separate instantiation, created by the helper `try_splitkv<ENABLE, ...>`. `ENABLE = !IS_CAUSAL && !RETURN_LSE` is a template parameter so the causal and lse instantiations never see the split kernel (and its `static_assert`s). Build time grows by one extra instantiation per (granularity, dtype) pair for this launcher: 4 more kernels (per_warp/per_thread x half/bf16).

## Kernel design

Grid `(q_tiles, heads, batch * S)`; `blockIdx.z = batch_id * S + split_id`. With batch 1, split 0 for all heads is scheduled first, then split 1 (each phase reads half the K/V of each head).

For a split starting at tile `t0` and holding `n` tiles, the original loop runs unchanged with `num_iterations = n`, after these offsets:

| state | offset |
|---|---|
| `K_lane_base_ptr` | `+ t0 * CTA_K * stride_seq_k` |
| `V_lane_base_ptr` | `+ t0 * CTA_K` bytes (V is `[d, kv]`, kv contiguous) |
| `k_scale_idx` | `+ t0 * k_scale_advance_offset` (per-warp K scales are indexed per tile) |
| `K_idx_lane_base`, `K_load_idx_lane_base` | `+ t0 * CTA_K` (absolute indices) |

Absolute indices keep the existing out-of-bound mask and predicated K loads correct: they only ever trigger in the split that owns the last tile (the original already applies them only on the last two iterations of the loop; in other splits they are a no-op). The prologue, the one-buffer K/V pipeline, `wait_group` counts and `__syncthreads` are untouched.

The epilogue differs. The original does, in order: `normalize_d` (quad-reduce `d`, multiply `RO` by `1/d`), `fuse_v_scale` (multiply by per-channel V scale), convert to bf16/fp16 through smem, store. The partial path stops before all of it:

1. Quad-reduce `d` with the same two `__shfl_xor_sync` the original uses.
2. Store `RO` as fp32, un-normalized and un-scaled, straight from registers (`float2` stores; each row segment is a full 32 B sector) to `O_part[split][batch][head][row][128]`.
3. Lane `lane%4 == 0` stores `(m, d)` for the row to `ML_part[split][batch][head][row]`.
4. Rows `>= qo_len` are skipped.

`m` is in the kernel's log2 domain and includes `-S_FP8_OFFSET` (8.807) and the Q/K dequant scale; the offset and scales are identical across splits, so `m_i - m` is directly comparable.

### The deferred steps

- **Normalization.** Deferred to the combine: `O = sum_i 2^(m_i - m) O_i / sum_i 2^(m_i - m) d_i`, which is the flash-decoding merge. `d_i` is the full row sum for split `i` (after the quad reduction), accumulated from the fp32 `RS_f32` before the fp8 conversion, as in the original.
- **Fused V scale.** The original multiplies the normalized `RO` by `V_scale[b, h/groups, channel]` once at the end. That scale is per channel and constant over keys, so it commutes with the sum: the partials hold the fp8-domain `P·V` (V as stored, without the scale), and the combine applies the scale after dividing by the merged denominator, the same order as the original. `fuse_v_mean` is not used by this launcher and is rejected by a `static_assert`.
- **fp16-accumulate PV.** `RO` is fp32 (`DTypeSVAccum = float`). The `fp32+fp16` mode sums only one 64-key tile in fp16 (`RO_int32`, the "inst buf") and adds it into the fp32 `RO` at the end of each tile. No fp16 state crosses a tile boundary, so a split boundary falls between flushes and the partial `RO` is fp32. The fp16 overflow margin (P <= 448, V <= 2.25, 64 keys) is per tile and unchanged.
- **P quantization.** P is quantized to e4m3 per tile against the running row max, as before. A split that starts from a fresh max quantizes against its own local max (see numerics).

### Combine kernel

`splitkv_combine_kernel<head_dim, DTypeOut, fuse_v_scale>`: one warp per (batch, head, row), grid `(ceil(q/4), heads, batch)`, block `(32, 4)`. Each lane owns 4 channels (`float4` loads). Per row: `m = max m_i`; `L = sum 2^(m_i-m) d_i`; `o = sum 2^(m_i-m) o_i`; `o * rcp(L) * v_scale`; convert to bf16/half; one 8-byte store into the original output layout (the kernel's own `stride_bz_o/stride_seq_o/stride_h_o`, so HND and NHD both work). It uses the same `ex2.approx` and `rcp.approx` helpers as the main kernel. No atomics, so the result is deterministic for a given S and shape.

## Register and shared-memory impact

- Shared memory: unchanged (32 KB/CTA); the split kernel never uses the smem O staging.
- Registers: the main loop is the same code. The partial epilogue only reads `RO`, `m`, `d`, so it should not raise the peak, and the original was at the 255 cap already. The compiler may allocate the second instantiation differently, so check `build.sh`'s `cuobjdump` output for the split kernel: 255 registers with zero spill is the expectation, any spill stores mean the split kernel is a loss. Occupancy stays 2 CTAs/SM.
- Workspace (allocated per call through the torch caching allocator): `S * B * H * q * (128*4 + 8)` bytes. Card 0, S=2: 62.7 MB. Card 1, S=2: 12.5 MB. The 5090's L2 is large enough that part of this should be read back from L2.

## Expected time

Idealised (full-rate CTAs, no overhead): S=1 is 2 rounds of T; S=2 is 3 rounds of T/2 = 1.5T. That is 25 % off the kernel; a perfectly balanced 1.41T would be the 29 % ceiling.

Costs:

- Combine: writes 62.7 MB of partials plus reads them back plus 15.4 MB of bf16 output, at roughly 1.5 TB/s of usable bandwidth: on the order of 60-100 us, if the partials mostly miss L2. Against a call of about 1.5 ms (10 heads) that is 4-6 %.
- Each CTA repeats the Q load and the pipeline fill: 1-2 %.
- The last round of S=1 runs 140 CTAs and each of those runs faster than a CTA in a full round (less contention), so the real tail is less than 25 % of the call.

My estimate for card 0: a net gain of 12-20 %, and it could be less. This is a prediction, not a measurement.

Card 1 (2 heads): 96 CTAs on 170 SMs leave 74 SMs idle, which is a bigger loss than the tail. S=2 gives 192 CTAs, S=3 gives 288, S=4 gives 384 (just over the 340 slots), so this card may gain more than card 0 and may prefer S=3 or 4. Run `check.py --splits 1 2 3 4`.

If the combine is too expensive, the follow-up is an in-kernel combine: the last CTA to finish for a given q-tile (atomic counter per tile) merges in-register and the workspace never leaves L2. That is a bigger change and not done here.

## Numerical difference vs S=1

Not bit-identical. What is identical: the int8 Q/K quantization, every QK^T int32 value and its dequantization (K scales are per tile and are indexed by absolute tile), the fp32 row sums per tile, and the per-tile fp16 PV sum for a given P tile. What differs:

1. **Running max.** Split 1 starts with a fresh `m` instead of the running max of tiles 0..t0-1. P = `2^(s - m_running)` is rounded to e4m3 (3 mantissa bits), so a different `m_running` gives different rounding on those P entries. This is rounding noise of the same size the kernel already has, not a bias: e4m3 relative precision is constant over its normal range. A split whose local max is below the global max sees its P values scaled up by `2^(m_global - m_local)` before quantization, which if anything puts fewer of them in the subnormal range; the merge then scales them back down.
2. **Summation order.** The fp32 sum over tiles is split into two partial sums merged at the end; `RO` rescales (`RO *= 2^(m_prev - m)`) are replaced by one merge factor per split.
3. **Merge arithmetic.** `ex2.approx.ftz` (~2 ulp) on `m_i - m`, and `rcp.approx.ftz` on the merged denominator; the original also uses `rcp.approx` once.
4. **Output rounding.** One fp32 to bf16 rounding, as in the original, on slightly different fp32 inputs: expect a fraction of elements to differ by one bf16 ulp.

I have no measured numbers. The expectation is S=2 vs S=1 at or below the S=1 vs SDPA error, and S=2 vs SDPA within a few percent of the S=1 vs SDPA error; `check.py` enforces `rel_l2(S vs ref) <= 1.1 * rel_l2(S=1 vs ref)` and `rel_l2(S vs S=1) <= rel_l2(S=1 vs ref)`. A fixed S gives the same output on every run.

## Test plan

1. `bash build.sh` (CUDA toolkit, torch of the target venv). Read the printed `cuobjdump` registers and spill stores for both `qk_int_sv_f8_attn_kernel` instantiations; stop if the split one spills.
2. `python check.py --edge`: odd shapes (kv 65, 64, 1000, 4097, q not a multiple of 128). Covers: last tile with a single valid key, S larger than the tile count, one tile (fallback), ragged q rows (partial store guards). Use `--splits 1 2 3 4 8`.
3. `python check.py`: the card 0 and card 1 shapes, accuracy then timing. Look at the S=1 line first: it must be bit-identical to the unset-env call, and its time must match the stream-patch wheel.
4. Repeat with the model's real tensors (dump q/k/v from one denoise step) because randn inputs have flat softmax rows, which understate the effect of the running-max change on peaky rows.
5. `compute-sanitizer --tool memcheck` on `--edge` (partial stores, K/V pointer offsets).
6. Time under the real pipeline (both cards at once, the L2 and clocks differ from a single-card microbenchmark), and with the full-model output check from `bench-world-model-quality`. Check `wan/modules/sage_kvq.py`: if it calls this launcher with `is_causal=0, return_lse=0` the variable applies; its KV cache path was not read here.

## Risks

- Not compiled. Most likely failure points: default arguments on the `__global__` kernel at launch sites that omit them (every other launcher), the macro-argument commas in the `try_splitkv<...>` call inside `DISPATCH_*` (variadic, same as the existing `kernel_func` line), and `int` template arguments converted to `uint32_t` kernel parameters.
- The split instantiation could allocate registers differently from the original (spills would erase the gain).
- 62.7 MB of workspace traffic could cost more than estimated if the partials miss L2.
- `SAGE_SPLITKV` is read at launch time: inside a CUDA graph capture the value at capture time is frozen.
- `batch * S` goes into `gridDim.z` (limit 65535); not a concern at these batch sizes.
