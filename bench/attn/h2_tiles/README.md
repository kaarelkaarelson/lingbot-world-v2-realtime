# H2: SageAttention sm89 FP8 kernel tile sizes on sm_120

Hypothesis: the tile constants of the kernel LingBot-World runs on the RTX 5090
(`sageattention_sm89::qk_int_sv_f8_attn_kernel<128u, 64u, 32u, 64u, 128u, ...>`) were chosen for Ada
(sm_89) and leave performance on the table on consumer Blackwell (sm_120).

Everything below refers to thu-ml/SageAttention at commit `d1a57a5`
(`https://github.com/thu-ml/SageAttention/blob/d1a57a5/<path>#L<n>`). Nothing here was run on a GPU;
the register and occupancy numbers are estimates that `build.sh` verifies with `cuobjdump`.

## What runs on sm_120

- `sageattention/core.py#L152-L153`: on `sm120` `sageattn()` calls
  `sageattn_qk_int8_pv_fp8_cuda(..., qk_quant_gran="per_warp", pv_accum_dtype="fp32+fp16")`.
- `core.py#L791`: Q/K are quantised with `per_warp_int8_cuda(..., BLKQ=128, WARPQ=32, BLKK=64)` -- the
  scale-tensor layout is tied to the kernel tiles (checked by the launcher, see below).
- `core.py#L809`: V goes through `per_channel_fp8`, which pads the sequence to a multiple of 64
  (`quant.py#L273`, `#L278`; hard-coded `CTA_SIZE = 64` in `csrc/fused/fused.cu` `transpose_pad_permute_cuda`).
- `core.py#L818-L819`: `pv_accum_dtype == "fp32+fp16"` -> `sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf`
  -> launcher `csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu`.

## The tile constants

Launcher `csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu#L124-L127` (identical in all
seven `sm89_*.cu` launchers):

```
constexpr int CTA_Q = 128;   // query rows per thread block
constexpr int CTA_K = 64;    // key/value rows per pipeline step
constexpr int WARP_Q = 32;   // query rows per warp
constexpr int WARP_K = 64;   // key rows per warp (== CTA_K)
```

- `#L160`: `block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K))` -> 4 warps = 128 threads per CTA.
- `#L159`: `grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch)` -> at q=6032, 12 heads: 48 x 12 = 576 CTAs.
- `#L152`: dynamic smem = `max(CTA_Q*D + CTA_K*D + CTA_K*D (int8/fp8 Q,K,V), CTA_Q*D*2 (fp16 O))`
  = 128*128 + 64*128 + 64*128 = **32 KB** per CTA (D = 128). Set via `cudaFuncSetAttribute` at `#L157`.
- `#L136-L137`: per-warp scale shapes are checked against `div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q)` and
  `div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K)`, so the quantiser's `BLKQ/WARPQ/BLKK` must equal `CTA_Q/WARP_Q/CTA_K`.
- `#L130`: `assert(value.size(3) >= div_ceil(kv_len, CTA_K) * CTA_K)` -- V must be padded to whole `CTA_K` tiles
  (the kernel loads V without a predicate, `qk_int_sv_f8_cuda_sm89.cuh#L260`).

Kernel `csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh` -- how the constants turn into MMA tiles:

- `#L35-L42`: `MMA_QK_M/N/K = 16/16/32` (int8 `mma.sync.m16n8k32.s8.s8.s32`, two n8 halves, `csrc/mma.cuh#L329`),
  `MMA_SV_M/N/K = 16/16/32` (fp8 `mma.sync.m16n8k32.e4m3.e4m3.f16`, `csrc/mma.cuh#L577`, used by the
  `fp16_accu` PV path `attn_utils.cuh#L924`).
- `#L65-L71`: `num_warps_q = CTA_Q/WARP_Q`, `num_warps_k = CTA_K/WARP_K`, `num_tiles_q = WARP_Q/16`,
  `num_tiles_k = WARP_K/16`, `num_tiles_qk_inner = head_dim/32 = 4`, `num_tiles_v = head_dim/16 = 8`.
  Baseline: 4 warps, each warp owns a 32 x 64 S tile (2 x 4 MMA tiles) and a 32 x 128 O tile (2 x 8).
- `#L93-L96`: per-thread register arrays: `RS[nq][nk][8]` int32, `RO[nq][8][8]` fp32, `m/d[nq][2]`;
  `#L287` `RS_f32[nq][nk][8]`, `#L319` `RS_f8[nq][nk/2][4]`; the fp16-accumulate PV adds `RO_int32[nq][8][4]`
  (`attn_utils.cuh#L905`). With nq=2, nk=4: 64 + 128 + 64 + 16 + 64 = 336 32-bit values if all were live at once;
  the compiler overlaps lifetimes but this kernel is expected to sit at or near the 255-register cap (verify with
  `cuobjdump --dump-resource-usage`, `build.sh` prints it).
- `#L170-L178`: smem layout `[Q: CTA_Q x 128 int8][K: CTA_K x 128 int8][V: 128 x CTA_K fp8]`, O written over Q at the
  end (`#L180`). V is stored transposed with row stride `V_SMEM_STRIDE = CTA_K` bytes (`#L76`) and swizzle
  `k64B` or `k128B` (`#L177`).
- `#L222-L226`: `num_iterations = div_ceil(kv_len, CTA_K)` = 425 K/V steps per CTA at kv=27144.
- Pipeline: there is **no `stages` parameter**. K and V each have exactly one smem buffer; the next K tile is issued
  after the QK MMA of the current step (`#L327-#L332`) and the next V after the PV MMA (`#L360-#L366`), with
  `cp_async::wait_group<1>` (`#L273`, `#L338`). Deeper prefetch needs extra smem buffers and a rewrite of the
  loop -- it cannot be switched on by a constant. That is why there is no "more stages" patch here.

### Constraints a configuration must satisfy

| constraint | where |
|---|---|
| `CTA_K % 64 == 0` | kernel `static_assert`, `qk_int_sv_f8_cuda_sm89.cuh#L62` |
| `CTA_Q / CTA_K <= 2` (causal path) | `#L63` |
| `CTA_K` in {64, 128}: V row stride is `CTA_K` bytes and only `k64B`/`k128B` swizzles exist | `#L76`, `#L177`, `csrc/permuted_smem.cuh` |
| `WARP_K == CTA_K` (one warp spans the whole K tile): the `inst_buf` PV routines compute their V offset from the lane only (`smem_V_col_base = (lane/8)%2`, warp_idx_k commented out) and the epilogue never reduces across `num_warps_k` (`#L570` TODO) | `attn_utils.cuh#L902-L903`, `#L570` |
| `WARP_Q % 16 == 0`, `WARP_K % 32 == 0` (`num_tiles_k/2` fp8 fragments) | `#L68-L69`, `attn_utils.cuh#L479` |
| `CTA_Q % WARP_Q == 0`, `CTA_K % num_warps == 0`, `128 % (4*num_warps) == 0` (global->smem copy split: 4 rows of 128 B per warp per iteration) | `#L182-L203` |
| quantiser only instantiates `BLKQ,BLKK` in {64, 128} and `WARPQ` in {16, 32} | `csrc/dispatch_utils.h#L88-L112`, `fused.cu` `quant_per_warp_int8_cuda` |
| V padded to a multiple of `CTA_K` | launcher `#L130`; `quant.py#L273` pads to 64 only |
| fp16 PV accumulator: the `fp16_accu` PV path sums one whole K tile in fp16 before the fp32 flush (`attn_utils.cuh#L936-L978`); P <= 448 and V <= `scale_max` (2.25, `core.py#L805-L807`) give 64 * 448 * 2.25 = 64512 < 65504, so `CTA_K = 128` needs `scale_max = 1.125` or it overflows | `core.py#L805-L807` |
| `return_lse` only correct for `num_tiles_q == 2` (WARP_Q = 32) | `#L693` |
| dynamic smem <= 99 KB per block (sm_120 opt-in limit) | CUDA guide table below |

So the design space reachable without touching the kernel body is `CTA_Q` in {64,128} x `CTA_K` in {64,128} x
`WARP_Q` in {16,32}, with `WARP_K = CTA_K`. The MMA shapes themselves (m16n8k32) are fixed by the instruction
and the register fragment layouts; nothing about them is Ada-specific (sm_120 executes the same `mma.sync`
PTX; it has no `wgmma`/TMA).

## sm_89 vs sm_120 hardware budget

CUDA C++ Programming Guide 12.8, "Technical Specifications per Compute Capability"
(`https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#features-and-technical-specifications`):

| | 8.9 (Ada) | 9.0 (Hopper) | 12.0 (consumer Blackwell) |
|---|---|---|---|
| max resident blocks / SM | 24 | 32 | 32 |
| max resident warps / SM | 48 | 64 | 48 |
| max resident threads / SM | 1536 | 2048 | 1536 |
| 32-bit registers / SM | 64 K | 64 K | 64 K |
| max registers / thread | 255 | 255 | 255 |
| max shared memory / SM | 100 KB | 228 KB | 100 KB |
| max shared memory / block (opt-in) | 99 KB | 227 KB | 99 KB |

The per-SM budget of sm_120 is identical to sm_89 (48 warps, 64 K registers, 100 KB smem). What changed is the
SM count and clocks (RTX 5090: 170 SMs vs RTX 4090: 128) and the FP8/INT8 tensor-core throughput per SM. So the
"Ada tiles are too small/large for Blackwell's bigger smem" version of H2 is false: there is no bigger smem.
What is left of H2 is (a) occupancy -- the 4-warp / ~255-register CTA gives at most 2 CTAs = 8 warps per SM out
of 48, so latency hiding relies on ILP inside 4 warps; (b) wave quantisation -- 576 CTAs over 170 SMs x 2 slots =
1.69 waves (see `bench/attn/h4_wave_quantization.py`); (c) L2->SM traffic -- every CTA streams the whole K and V
of its head: 48 x 12 x 27144 x 256 B = 4.0 GB per call.

## Configurations

`WARP_K = CTA_K` in all of them. smem from launcher `#L152`; registers are estimates of simultaneously-live
32-bit values per thread (RS + RS_f32 during QK, RO + RO_int32 + RS_f8 during PV, plus ~30 for addressing);
CTAs/SM = min(regs limit, smem limit, 32).

| cfg | CTA_Q | CTA_K | WARP_Q | warps/CTA | smem/CTA | est. regs/thr | CTAs/SM (regs) | CTAs/SM (smem) | warps/SM | CTAs in grid | K steps/CTA | L2->SM GB/call |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 128 | 64 | 32 | 4 | 32 KB | ~255 (2 x 4 S tiles, 2 x 8 O tiles) | 2 | 3 | 8 | 576 | 425 | 4.0 |
| cfg_a | 64 | 64 | 16 | 4 | 24 KB | ~140-168 | 3 | 4 | 12 | 1140 | 425 | 7.9 |
| cfg_b | 64 | 128 | 16 | 4 | 40 KB | ~200-255 (1 x 8 S tiles) | 2 | 2 | 8 | 1140 | 213 | 7.9 |
| cfg_c | 128 | 64 | 16 | 8 | 32 KB | ~140-168 | 1 (2 if <= 128 regs) | 3 | 8-16 | 576 | 425 | 4.0 |
| cfg_d | 128 | 128 | 16 | 8 | 48 KB | ~200-255 | 1 | 2 | 8 | 576 | 213 | 4.0 |

Why each is plausible on sm_120:

- **cfg_a (64/64/16)**: halves the per-warp S and O tiles, which is the only lever that reduces register pressure
  in this kernel; 3 CTAs/SM (12 warps) instead of 2 (8), and 1140 CTAs give 2.2 waves over 510 slots instead of
  1.7 over 340. Cost: every K/V byte is pulled from L2 twice as often (7.9 GB/call, ~4.4 TB/s at the current
  1.8 ms) and each warp does half the MMAs per `ldmatrix` of K/V (smem bandwidth per MMA doubles). Whether
  the 5090's L2 sustains that is exactly what the run measures. This is the sm80 fp16 head_dim-128 choice
  upstream already makes (`core.py#L602`, WARPQ=16).
- **cfg_b (64/128/16)**: the SM90 tiling upstream uses for its own FP8 kernel (`core.py#L967`: BLKQ 64, WARPQ 16,
  BLKK 128). Halves the number of K steps and `__syncthreads` per CTA (213 vs 425) and doubles the V reuse per
  softmax pass; registers go back up (8 S tiles per warp) so occupancy is like the baseline. Needs V padded to
  128 rows and the V scale ceiling halved to 1.125 so the fp16 accumulator cannot overflow (the patch does both in
  `core.py`; the pad costs one extra V copy, the smaller e4m3 range costs nothing in relative precision).
- **cfg_c (128/64/16)**: identical grid, smem and L2 traffic to the baseline; only splits each warp's 32 query rows
  across two warps (8 warps/CTA). Isolates the register/ILP question from the traffic question: if cfg_c beats
  the baseline, the 255-register 4-warp CTA was the limiter; if it loses, the halved MMA-per-ldmatrix ratio
  dominates. Lowest-risk experiment.
- **cfg_d (128/128/16)**: cfg_b's K-step halving with cfg_c's grid; 48 KB smem, one CTA per SM, 8 warps.
  Included because it is cheap once the mechanism exists, not because it is expected to win.
- **cfg_base**: the mechanism patch with upstream defaults -- builds the reference wheel from the same source and
  toolchain so the comparison is apples to apples (the shipped wheel may have been built differently).

Honest fit estimate: every configuration fits sm_120's smem (max 48 KB of 99 KB) and the static asserts in
`h2_tiles.h` pass for all five (checked locally with `clang++ -fsyntax-only`). The unknown is registers: nq=1
configurations should drop well under 255; nk=8 configurations (cfg_b, cfg_d) may spill. `build.sh` prints
`cuobjdump --dump-resource-usage` for the built kernels -- discard any configuration that reports spill stores
before benchmarking it. `return_lse=True` is broken for WARP_Q=16 (kernel `#L693`); LingBot's attention call does
not use it.

## Files

- `cfg_base.patch`, `cfg_a.patch` ... `cfg_d.patch` -- git patches against `d1a57a5`, each self-contained. They
  add `csrc/qattn/h2_tiles.h` (the three macros with defaults for that cfg and the `static_assert`s above),
  make all seven `sm89_*.cu` launchers read `CTA_Q/CTA_K/WARP_Q` from the macros (`WARP_K = CTA_K`), export
  `_qattn_sm89.h2_tile_config()` from `pybind_sm89.cpp`, make `core.py` derive the quantiser's `BLKQ/WARPQ/BLKK`
  and the V padding from that, and let `setup.py` override the macros from `SAGE_H2_CTA_Q/CTA_K/WARP_Q` env vars
  (so one patch can build any cfg: `SAGE_H2_CTA_Q=64 SAGE_H2_WARP_Q=16 ... python setup.py bdist_wheel`).
  There is no runtime switch: the tiles are template arguments and each instantiation costs the full ~26 min
  build, so each cfg is a separate wheel.
- `cfg_nosoftmax.patch` -- **timing-only** control, not a candidate: cfg_base (same 128/64/32 tiles, same
  `h2_tiles.h`/pybind/launcher/`core.py`/`setup.py` edits, so `build.sh` and `bench.py` run unchanged) plus
  `#define SAGE_H2_NO_SOFTMAX 1`, which compiles the online softmax out of `qk_int_sv_f8_cuda_sm89.cuh`:
  in all three K-step bodies the `update_mdo` call (row max + shuffles, `exp2(S*scale - m)`, `exp2(m_old - m_new)`,
  `d *= o_scale`, the 64-multiply `RO *= o_scale` rescale; kernel `#L305-#L312`, `#L413-#L420`, `#L520-#L527`)
  and `accumulate_d` (row sum; `#L314-#L317`, `#L422-#L425`, `#L529-#L532`) are replaced by `RS_f32 *= sm_scale`,
  and the unused tensor-core `accumulate_d_f8` branch (`#L322-#L325`, `#L430-#L433`, `#L537-#L540`) is guarded too. QK int8 MMA, `__int2float_rz`,
  `RS_32_to_8` (S->fp8 P), PV fp8 MMA, loads/ldmatrix/cp.async, `__syncthreads` and the epilogue are untouched;
  `d` stays at its init value 1 so `normalize_d` (`#L572`, once per CTA) still divides O but by a constant.
  Output is garbage (`bench.py` will report cos ~0 or NaN; ignore it). Read: `ms(cfg_base) - ms(cfg_nosoftmax)`
  = softmax time that is *exposed* (not hidden behind the MMAs/loads), i.e. an upper bound on what a cheaper
  softmax could recover. It cannot separate the softmax arithmetic from the int32->fp32->fp8 P conversion, which
  is kept in both builds; and the fp16 PV accumulator will overflow with unnormalised P, which does not change
  the instruction stream but means the output must not be used for anything.
- `build.sh <cfg>` -- clone at `d1a57a5` into `/workspace/sage_h2/<cfg>`, apply the patch, build a wheel with
  `TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=16`, install it into `/workspace/sage_h2/venv_<cfg>` (a
  `--system-site-packages` venv, so the pod's torch and the shipped sageattention stay untouched), print the
  wheel path and the kernels' register/smem/spill usage.
- `bench.py --cfg <cfg>` -- 20 warm + 50 timed `sageattn()` calls at the model shapes with CUDA events,
  correctness vs fp32 SDPA (max abs, cosine), writes `bench/attn/results/h2_<cfg>.json`. `--dry` runs on CPU with a
  fake module.

Pod sequence (each build ~26 min):

```
cd ~/lingbot-world-v2-realtime
for c in cfg_base cfg_c cfg_a cfg_b; do
  bash bench/attn/h2_tiles/build.sh $c && /workspace/sage_h2/venv_$c/bin/python bench/attn/h2_tiles/bench.py --cfg $c
done
```

Production caveat: `wan/modules/sage_kvq.py` (the KV-cache path) calls `sm89_compile.*` directly with its own
quantiser and hard-codes `BLK_K, BLK_Q, WARP_Q = 64, 128, 32` (`sage_kvq.py#L29`), and its fp16-overflow margin
(`#L30`) assumes 64 keys per PV step. A winning cfg must be carried into `sage_kvq.py` (read
`_qattn_sm89.h2_tile_config()`, pad the cache to `CTA_K`, rescale the V margin for `CTA_K = 128`) before it
helps the model; `bench.py` only measures the `sageattn()` path.

Read `cfg_c` vs `cfg_base` first (same traffic, different occupancy), then `cfg_a` (traffic vs occupancy), then
`cfg_b` (fewer, bigger K steps). A win must reproduce with cosine >= the baseline's and no spill stores.
