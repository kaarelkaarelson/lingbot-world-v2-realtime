# cfg_depnull -- decisive follow-up to the retracted cfg_fixedmax_dep split

**Timing control, not a candidate kernel.** Output is bit-identical to `cfg_fixedmax`.

## Why this exists

`cfg_fixedmax` (C5) is 8.7% faster than `cfg_base` (1.577 vs 1.726-1.741 ms kernel-only, pod 16,
`OPTIMIZATIONS.md` #20). `cfg_fixedmax_dep` tried to split that into "serial dependency" vs
"removed ALU work" by putting a *surrogate* dependency back in front of C5's exp2: a 2-shuffle
chain seeded from one S fragment (`RS[fq][0][k*2+0]`), ~6 instructions deep, seeded and biased into
the exp2 argument via `bias + dummy * 0.0f`. It measured 1.627 ms (33/67 split), and its red-team
**rejected the split**: the surrogate chain roots in a different S fragment than `update_mdo`'s
real max (which reduces over every `num_tiles_k` fragment, not just `fk=0`) and is a different
number of dependent instructions deep (~6 against ~20); the control also restructures the loop
nest relative to `cfg_base`. `SAGE_H2_DEP_CONTROL=0` therefore does not isolate the dependency
from the restructuring. `OPTIMIZATIONS.md` records the split as retracted and names the follow-up
this patch builds: "the same restructuring with the dummy chain present but not feeding the exp."

## What the patch does

`cfg_depnull` starts from `cfg_fixedmax` unchanged (same `apply_fixed_max`: a flat `fq/fk/e` loop,
`P = exp2(S * sm_scale - phi_log2 + S_FP8_OFFSET)`, one independent `fmaf`+`exp2` per element, no
`m`/`d` state, no rescale). Before each of the three `apply_fixed_max` call sites, when
`SAGE_H2_DEP_NULL` is set, `sink_real_max_dep_null()` (`csrc/qattn/attn_utils.cuh`) runs first,
reading the same fresh `RS_f32` tile `apply_fixed_max` is about to overwrite in place:

```cpp
float m_temp = -5000000.0f;
for (uint32_t fk = 0; fk < num_tiles_k; fk++)                    // every S fragment, not just fk=0
{
  float m_local = max(max(RS[fq][fk][k*2+0], RS[fq][fk][k*2+1]),
                       max(RS[fq][fk][k*2+4], RS[fq][fk][k*2+5]));
  m_temp = max(m_temp, m_local);
}
m_temp = fmaf(m_temp, sm_scale, -S_FP8_OFFSET);                  // exp_offset=true, fuse_scale=false
m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x1));
m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x2));
asm volatile("" : "+f"(m_temp));                                 // sink: keeps the chain live
```

This is `update_mdo`'s row-max computation verbatim: the same `fk` loop over every
`num_tiles_k` fragment (not a single fragment), the same indexing (`{k*2+0, k*2+1, k*2+4, k*2+5}`),
the same `exp_offset=true, fuse_scale=false` pre-shuffle transform (the instantiation
`qk_int_sv_f8_cuda_sm89.cuh` actually calls at all three `update_mdo` sites), and the same two
`__shfl_xor_sync` with the same mask and widths. **The difference from `cfg_fixedmax_dep`: there is
no arithmetic edge back into `RS`/the exp2 argument at all.** `m_temp` dead-ends at the `asm
volatile` barrier. `apply_fixed_max`'s exp2 loop is untouched and reads/writes `RS` exactly as in
`cfg_fixedmax` -- nothing here feeds it, blocks it, or reorders around it except whatever ptxas
chooses to do with two data-independent instruction streams.

**Why the compiler cannot delete it.** Same reasoning as `cfg_fixedmax_dep`: `__shfl_xor_sync` is a
convergent intrinsic with a sync side effect (NVVM must not sink/hoist/duplicate/drop it while its
result is live), and `asm volatile("" : "+f"(m_temp))` both reads and writes `m_temp`'s register and
is never deleted or reordered against other volatile asm. The whole producing chain -- the `fk` max
tree, the `exp_offset` fmaf, both shuffles -- must retire before the barrier.

**Why it's safe to call before `apply_fixed_max`.** `sink_real_max_dep_null` only reads `RS`; it
writes nothing back into `RS`, `m`, or `d`. `apply_fixed_max` is called immediately after and
overwrites `RS` in place with `P`, exactly as in `cfg_fixedmax`. Output is therefore bit-identical
to `cfg_fixedmax`'s -- same phi, same P, same cosine -- and that must be verified before the timing
is read (see "How to read the result" below).

## Prediction table (write this down before reading `bench.py`'s output)

Let `ms(base)` = 1.726-1.741 (pod 16 range), `ms(fixedmax)` = 1.577 (both kernel-only,
`bench_kernel_only.py`).

| `ms(cfg_depnull)` | reading |
|---|---|
| **≈ 1.577** (matches `cfg_fixedmax`, i.e. adding the disconnected real max-tree costs nothing measurable) | The dependency **EDGE** is the whole story: computing the real max is free as long as it does not gate the exp2 (ptxas hides its latency behind other issue slots / other warps). C5's win is attributable to breaking the `S -> max -> exp2` chain, not to instruction count. Productive direction: dependency-breaking restructures that still compute a real max (two-pass / decoupled max, software-pipelining tile `i+1`'s max against tile `i`'s PV mma) rather than deleting the math and taking on phi's calibration risk. |
| **partway between, or ≈ 1.726-1.741** (close to `cfg_base`) | The real ALU work costs cycles even when it is not gating anything -- register pressure, occupancy, or issue-port contention from carrying ~20 extra dependent instructions per row group across 4 warps. Some or most of C5's win is from **deleting work**, not from breaking a dependency, and no dependency-preserving restructure will recover it -- only deleting the max tree (and accepting phi's calibration risk) buys the saving. |
| **strictly between** | Split: `(ms(base) - ms(depnull)) / (ms(base) - ms(fixedmax))` is the edge's share; `(ms(depnull) - ms(fixedmax)) / (ms(base) - ms(fixedmax))` is the disconnected-ALU-work's share. Unlike the retracted split, this one uses a same-fragment, same-depth chain, so the arithmetic is defensible -- state it as a range, not a point estimate, and quote the noise band alongside it. |
| **faster than 1.577** | Broken: check SASS -- ptxas DCE'd the chain despite the barrier (a genuinely live 8-bit-wide `__shfl_xor_sync` chain executing has a real cost; "faster than the build that doesn't compute it at all" is not physically plausible) -- or it is noise; add repeats until the spread is well under the gap of interest. |
| **slower than 1.741** (worse than `cfg_base`) | Also suspicious: `cfg_depnull` does strictly more total work than `cfg_base` (base at least *uses* its max; depnull computes an equivalent one and throws it away) -- check register/spill counts. If `cuobjdump --dump-resource-usage` shows spills where `cfg_fixedmax` had none, carrying two independent live chains (the real max tree and the constant-phi P path) through the same 255-register budget is itself informative and worth reporting regardless of the ms number. |

Noise band first, same as `cfg_fixedmax_dep`: the gaps of interest are all under ~0.15 ms on a
~1.6 ms kernel (under ~9%); run enough repeats that run-to-run spread is well under 0.02 ms before
reading anything from a sub-1% difference.

## Honest assessment: is this a clean single-variable control?

**Closer than `cfg_fixedmax_dep`, but not perfectly clean -- one residual confound, named here
rather than hidden.**

What it fixes relative to the rejected control:
- **Same S fragment.** The max reduces over every `num_tiles_k` fragment via the real `fk` loop,
  not a single fragment (`fk=0`) read twice through two shuffles.
- **Same depth.** The chain is `update_mdo`'s actual `~2-3 fmaxf per fk` times `num_tiles_k`,
  chained through the running `m_temp` accumulator, then the real `exp_offset` fmaf, then the real
  two shuffles -- the same instruction count and dependency depth as `cfg_base`'s real chain
  (`OPTIMIZATIONS.md`'s red-team put this at ~20; `cfg_fixedmax_dep`'s surrogate was ~6).
- **Does not restructure `apply_fixed_max`'s loop nest.** The exp2/P computation is untouched from
  `cfg_fixedmax`; the real-max computation is a wholly separate, data-independent addition, not a
  rewrite of the exp2 path the way `cfg_fixedmax_dep` rewrote it into two passes with the dependency
  threaded through the bias.

**What it does not fix, because "not feeding the exp" structurally requires it:** in `cfg_base`,
`update_mdo` is a *single* fused pass -- the max computation and the exp2 computation share one loop
nest because the exp2 is a true data consumer of the max (`negative_m = -m[fq][k]`). In
`cfg_depnull`, by design, the max computation and the exp2/P computation are two *disjoint* loop
nests with no data edge between them. That gives ptxas scheduling freedom `cfg_base` never has: it
can fully overlap or defer the disconnected max-tree behind the next tile's `mma.sync` or `cp.async`
wait, in a way it cannot do when the exp2 genuinely blocks on the max. So:

- A **"no slowdown" (≈ 1.577) result is an upper bound on what a dependency-preserving restructure
  could recover, not proof that a real one would reach it.** Any real restructure that keeps a
  softmax mathematically correct must eventually *use* the computed max (to correct `O`/`d`, as in
  two-pass / decoupled softmax), which reintroduces a real data edge this sink never pays for.
  `cfg_depnull` bounds the best case; it does not build the restructure.
- This is the same risk `cfg_fixedmax_dep.md` flagged for its own (shallower, wrong-fragment)
  chain -- carried forward here because it is inherent to "compute something and prove it dead,"
  not specific to either patch's implementation.

**Recommended follow-up if `cfg_depnull` reads "no slowdown":** confirm with SASS (below) that the
full real chain survived rather than being partially folded, then treat the result as license to
prototype an actual two-pass restructure (not a further `dep_*` control) -- that is the only way to
learn whether the scheduling freedom this control exploits is available to a version of the kernel
that must use its max.

## SASS evidence to collect (same method as `cfg_fixedmax_dep`)

From `cuobjdump -sass` (or `nvdisasm -c`) on the built `_qattn_sm89*.so`, inside
`qk_int_sv_f8_attn_kernel`:

1. **`SHFL.BFLY` pairs at the loop-unroll factor for `num_tiles_k`-fragment max trees**, same as
   `cfg_base`'s (8 `SHFL.BFLY` per K tile at the baseline 128/64/32/64 tiling) -- `cfg_fixedmax`'s
   SASS has zero `SHFL` in the K loop, so any `SHFL` at all is evidence the chain is present, and
   the *count* should match `cfg_base`'s (not `cfg_fixedmax_dep`'s halved/surrogate count).
2. **No new operand of `apply_fixed_max`'s `FFMA -> MUFU.EX2` sequence traces back to the
   `sink_real_max_dep_null` chain.** This is the decisive check for "not feeding the exp": in
   `cfg_fixedmax_dep` the whole point was an `FFMA` reading a register the shuffle chain wrote; here
   the opposite must hold -- `apply_fixed_max`'s `FFMA` addend should be the same uniform
   constant-bank `bias` as plain `cfg_fixedmax`, with zero data dependency on the sink chain's
   registers.
3. **The sink chain's last `FMNMX` should feed nothing but a register that is otherwise dead** (no
   consumer besides whatever no-op the volatile asm barrier materializes as, typically nothing in
   the visible SASS beyond the barrier's own no-side-effect marker).
4. With `nvdisasm -c`, check whether the sink chain's `SHFL`/`FMNMX` are scheduled interleaved with
   `apply_fixed_max`'s independent `FFMA`/`MUFU.EX2` instructions (evidence of the scheduling
   freedom named above) or pushed to a separate block. Either is consistent with the control being
   valid; it is useful context for interpreting a "no slowdown" result.

## Build / run

```bash
bash bench/attn/h2_tiles/build.sh cfg_depnull
/workspace/sage_h2/venv_cfg_depnull/bin/python bench/attn/h2_tiles/bench.py --cfg cfg_depnull
/workspace/sage_h2/venv_cfg_depnull/bin/python bench/attn/h2_tiles/bench_kernel_only.py --cfg cfg_depnull
```

Run at the **same `SAGE_H2_PHI`** as the `cfg_fixedmax` measurement being compared against, and
first confirm correctness (`max_abs`, `cosine`) matches `cfg_fixedmax`'s exactly -- the patch claims
bit-identical output; if they differ, the sink leaked into the real computation and the timing is
uninterpretable.

`_qattn_sm89.h2_dep_null()` returns 1 in this wheel (and `h2_fixed_max_config()` the same
`(fixed_max_on, sat_check_on, phi_log2)` triple as `cfg_fixedmax`), so results JSON can record which
build produced it. `SAGE_H2_DEP_NULL=0 bash build.sh cfg_depnull` reproduces `cfg_fixedmax`
byte-for-byte in the softmax path, the cheapest way to re-measure the pair on one clone.
