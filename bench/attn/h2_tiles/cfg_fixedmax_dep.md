# cfg_fixedmax_dep -- dependency-vs-ALU control for cfg_fixedmax

**Timing control, not a candidate kernel.** Output is bit-identical to `cfg_fixedmax`.

## The confound it resolves

Measured on the 5090 at the model shapes (q=6032, kv=27144, 12 heads, D=128):

| build | ms | vs base |
|---|---|---|
| `cfg_base` (upstream online softmax) | 1.728 | -- |
| `cfg_fixedmax` (constant phi) | 1.574 | -8.9 % |

`cfg_fixedmax` changes two things at once, and the 8.9 % cannot be attributed to either:

- **(a) the serial DEPENDENCY** -- in `cfg_base` every tile's `exp2` waits on a warp-wide max:
  `mma.sync -> __int2float_rz -> 15-deep fmaxf tree -> __shfl_xor_sync(0x1) -> fmaxf ->
  __shfl_xor_sync(0x2) -> fmaxf -> m update -> negative_m -> exp2`. With 4 warps/CTA and 1-2 CTAs/SM
  there is nothing else to run while that chain drains.
- **(b) real ALU WORK** -- the max tree itself, the 2 SHFL, the `o_scale = exp2(m_prev - m_new)`
  and the `num_tiles_v * 4` multiply `RO` rescale (128 FMULs per K tile per thread at the baseline
  tiling).

`cfg_fixedmax` deletes both. This build deletes (b) exactly as `cfg_fixedmax` does and puts (a) back.

## What the patch does

`SAGE_H2_DEP_CONTROL` (in `csrc/qattn/h2_tiles.h`, default 1, `#error`s without `SAGE_H2_FIXED_MAX`)
switches `apply_fixed_max()` in `csrc/qattn/attn_utils.cuh` to a variant that, per `(fq, k)` row group
(`num_tiles_q * 2 = 4` groups per K tile at 128/64/32/64), does before the group's `exp2`:

```cpp
float dummy = RS[fq][0][k * 2 + 0];                              // this tile's own mma result
dummy = fmaxf(dummy, __shfl_xor_sync(0xffffffff, dummy, 0x1));   // same mask/width/op as update_mdo
dummy = fmaxf(dummy, __shfl_xor_sync(0xffffffff, dummy, 0x2));
asm volatile("" : "+f"(dummy));                                  // optimisation barrier
const float bias_dep = bias + __fmul_rn(dummy, 0.0f);            // == bias, exactly
```

and then uses `bias_dep` instead of `bias` in that group's `fmaf(RS, sm_scale, ...)`. The element
indexing is `update_mdo`'s (`{k*2+0, k*2+1, k*2+4, k*2+5}`), so all 8 elements of every fragment are
still processed exactly once -- verified by a host stub: 0 untouched elements.

**Why the compiler cannot eliminate the chain.** `__shfl_xor_sync` is convergent with a sync side
effect, so NVVM may not sink, hoist, duplicate or drop it while its result is live.
`asm volatile("" : "+f"(dummy))` both reads and writes `dummy`'s register and is a volatile asm
statement, so it is never deleted and keeps the producing chain live; because its *output* is the
operand of the multiply, ptxas carries a true register dependency
`SHFL -> FMNMX -> FMUL -> FADD -> FFMA -> MUFU.EX2`. After the barrier the compiler knows nothing
about `dummy`'s value, so it cannot fold `__fmul_rn(dummy, 0.0f)`; folding `x * 0 -> 0` is not an
IEEE-754-valid rewrite in any case (wrong for inf, NaN, -0) and nvcc/ptxas do not do it without
fast-math. `__fmul_rn` additionally pins the op to `mul.rn.f32` in PTX.

**Why the value is provably zero.** `dummy` is a max of four `RS_f32` values, and `RS_f32` is
`__int2float_rz` of an int32 QK accumulator (optionally times a finite dequant scale), so `dummy` is
**always finite** -- never inf, never NaN. For finite `x`, `x * 0.0f` is exactly `+0.0` or `-0.0`,
never NaN; this is exactly why the chain multiplies by zero rather than using `dummy - dummy`, which
would be NaN for a non-finite seed. Adding `±0.0` to the finite float `bias` is exact in
round-to-nearest for every `bias`, including `bias == 0` (`0 + -0 == +0`). So `bias_dep` is
bit-identical to `bias`, and the kernel output is bit-identical to `cfg_fixedmax`'s.

## Expected per-K-tile instruction sequence (per thread, tiling 128/64/32/64: `num_tiles_q=2`, `num_tiles_k=4`)

Four times per K tile (once per `(fq, k)` row group), in the softmax region only:

```
SHFL.BFLY PT, Ra, Rdummy, 0x1, 0x1f
FMNMX     Rdummy, Rdummy, Ra, !PT          ; fmaxf
SHFL.BFLY PT, Rb, Rdummy, 0x2, 0x1f
FMNMX     Rdummy, Rdummy, Rb, !PT
                                           ; asm volatile("") emits nothing
FMUL      Rz, Rdummy, RZ                   ; __fmul_rn(dummy, 0.0f)
FADD      Rbias, Rz, <bias>                ; bias_dep  (may appear as FFMA/FADD with a const operand)
```

followed by the 16 elements of that row group:

```
FFMA      Rx, RS_f32, <sm_scale>, Rbias    ; x16
MUFU.EX2  Rp, Rx                           ; x16
```

Per K tile, totals per thread:

| | SHFL.BFLY | FMNMX | FMUL | FADD | FFMA | MUFU.EX2 | RO rescale FMUL |
|---|---|---|---|---|---|---|---|
| `cfg_base` | 8 | ~60 (15-deep tree x4) | 4 (`d *= o_scale`) | 4 | 32 | 32 + 4 (`o_scale`) | 128 |
| `cfg_fixedmax` | 0 | 0 | 0 | 0 | 32 | 32 | 0 |
| `cfg_fixedmax_dep` | 8 | 8 | 4 | 4 | 32 | 32 | 0 |

**Honest caveat:** this control is not a *pure* dependency restoration. It re-adds 24 instructions per
K tile per thread (8 SHFL + 8 FMNMX + 4 FMUL + 4 FADD) against `cfg_fixedmax`'s 64 in the same region.
The 8 SHFL and 8 FMNMX are inseparable from the dependency -- a cross-lane dependency *is* two
butterfly exchanges -- but the 8 FMUL/FADD are genuinely extra ALU that neither other build executes.
So `ms(dep) - ms(fixedmax)` is a slight **over**-estimate of the dependency's cost, by at most the
issue cost of 8 cheap ALU ops per K tile (well under 1 % of the tile's instruction count).

## How to read the result

Run at the **same `SAGE_H2_PHI`** as the `cfg_fixedmax` run, and first confirm the correctness numbers
(`max_abs`, `cosine`) match `cfg_fixedmax`'s **exactly** -- the patch claims bit-identical output, and
if they differ the "no-op" is not one and the timing is uninterpretable.

| `ms(cfg_fixedmax_dep)` | reading |
|---|---|
| ~1.574 (= `cfg_fixedmax`) | the serial dependency costs **nothing**. The whole 8.9 % was the removed ALU work: the 128-FMUL `RO` rescale, the max tree and the `o_scale` exp2. Optimising for a cheaper rescale (e.g. conditional rescale, fp16 `o_scale`) is the productive direction; restructuring to break the dependency is not. |
| ~1.728 (= `cfg_base`) | the dependency was **the whole cost** and the ALU work is free (it hides behind the stall). The fixed-max quality risk buys nothing that a dependency-breaking restructure of the *exact* softmax could not also buy -- e.g. two-pass / decoupled max, or software-pipelining the max of tile *i+1* against the PV mma of tile *i*. Prefer that over changing the math. |
| strictly between | linear split. Attribute `(1.728 - ms_dep) / (1.728 - 1.574)` of the 8.9 % to ALU work and the remainder to the dependency, and decide whether the fixed-max quality risk is worth only its share. |
| **faster than 1.574** | the control is broken (the chain was eliminated) or the measurement is noise. Do not interpret; go to the SASS check below. |

Noise band first: the difference of interest is ~0.15 ms on 1.6 ms (~9 %); run enough repeats that the
run-to-run spread is well under 0.02 ms before reading anything from a sub-1 % gap.

## Risk: ptxas may schedule the chain off the critical path

The dummy chain has no consumer other than the barrier and the zero-multiply, and the four row groups'
chains are mutually independent. ptxas can therefore issue all four SHFL pairs early and overlap their
latency with the QK `mma.sync` drain and with each other -- in which case the control reports "the
dependency is free" when what it actually measured is "a **2-step** cross-lane dependency is free".
`cfg_base`'s real chain is longer (the 15-deep `fmaxf` tree sits *between* the mma and the shuffles) and
therefore less hideable, so a null result here does **not** by itself prove `cfg_base`'s full chain was
free. Treat "~1.574" as evidence that the *shuffle* part of the dependency is free, and if that is the
outcome, follow up with a second control that also restores the max-tree depth (accepting that it
re-adds ALU work) to bound the rest.

### SASS evidence that the chain survived

From `cuobjdump -sass` (or `nvdisasm -c`) on the built `_qattn_sm89*.so`, inside
`qk_int_sv_f8_attn_kernel`:

1. **4 `SHFL.BFLY` pairs per K-tile body** (8 `SHFL.BFLY` total per unrolled tile), each pair with
   immediate lane masks `0x1` then `0x2`, each followed by an `FMNMX`. Count them against the loop
   unroll factor -- `cfg_fixedmax`'s SASS has **zero** `SHFL` in the K loop, so any `SHFL` at all is
   proof the chain is present.
2. **An `FMUL Rz, Rdummy, RZ` (or `FMUL Rz, Rdummy, 0`) reading the register the second `FMNMX` wrote**,
   and an `FADD`/`FFMA` producing `bias_dep` from it. If the `FMUL` is absent, ptxas folded the zero and
   the dependency is gone even though the shuffles remain -- the run is invalid.
3. **The `FFMA` feeding each `MUFU.EX2` must take its addend from that `bias_dep` register, not from a
   constant bank slot `c[0x0][...]` or an immediate.** This is the decisive check: in `cfg_fixedmax` the
   addend is a uniform value; in this build it must be a register written by the shuffle chain.
4. With `nvdisasm -c`, the control info on that `FFMA` should show a wait/stall on the barrier the
   `SHFL` set (`SHFL` is variable-latency, so it writes a scoreboard barrier and the first consumer
   waits on it). A zero stall count there means ptxas hid the latency -- useful to know, and exactly the
   scheduling risk above.

## Build / run

```bash
bash bench/attn/h2_tiles/build.sh cfg_fixedmax_dep
SAGE_H2_PHI=<same phi as the cfg_fixedmax run> \
  /workspace/sage_h2/venv_cfg_fixedmax_dep/bin/python bench/attn/h2_tiles/bench.py --cfg cfg_fixedmax_dep
```

`_qattn_sm89.h2_dep_control()` returns 1 in this wheel (and `h2_fixed_max_config()` the same
`(fixed_max_on, sat_check_on, phi_log2)` triple as `cfg_fixedmax`), so the results JSON can record
which build produced it. `SAGE_H2_DEP_CONTROL=0 bash build.sh cfg_fixedmax_dep` reproduces
`cfg_fixedmax` byte-for-byte in the softmax path, which is the cheapest way to re-measure the pair on
one clone.
