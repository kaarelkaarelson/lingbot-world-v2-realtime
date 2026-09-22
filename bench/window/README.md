# KV-window sweep — "shorten the attention window" candidate

## What this actually changes

`local_attn_size` and `sink_size` are already plain, untouched constructor kwargs / CLI flags
on the baseline — `generate.py --local_attn_size --sink_size` (`generate.py:203-210`),
`WanI2VCausal.__init__` (`wan/image2video.py:245-246, 301-302`), `lingbot/play/live.py:260-273`
(`build_pipe`, defaults `local_attn_size=18, sink_size=6`). **No baseline file was modified.**
`bench/window/window_sweep.py` is a new, standalone script that only prints/drives
`generate.py` (via `lingbot clip`'s own example defaults) with different values of those two
existing flags, plus `LINGBOT_DUMP_LATENTS` (already env-gated in `wan/image2video.py:1353-1355`)
for the quality gate.

**Read this before picking a window size:** `kv_size = frame_seqlen * local_attn_size`
(`wan/image2video.py:568-573`, `:1081-1085`) is the *whole* attended KV buffer, not
sink + window. `sink_size` tokens are pinned at the front of that same buffer
(`wan/modules/model_fast.py:145`, `kv_ring_plan` at `:28-46`); the rolling part is
`local_attn_size - sink_size` latents. At the baseline (18, 6) that is a 12-latent rolling
window inside an 18-latent (27 144-token) total, **not** an 18-latent window on top of a
6-latent sink (24 latents / 27 144 would be a different number and does not match the code).
"Half the kv tokens" is `local_attn_size = 9`, which leaves a 3-latent rolling window against
a 4-latent chunk — degenerate (see script output). Keeping `sink_size = 6` fixed, the sweep
below stops at `local_attn_size = 10` (rolling = chunk size) for the "safe" points and calls
out 9 separately as an explicit stress probe.

## Commands

Predict + print copy-pasteable commands (works anywhere, no GPU/torch needed):

```
python3 bench/window/window_sweep.py --windows 18,14,12,10,9 --dump_dir /path/to/dumps
```

One setting via the env convention asked for (`LINGBOT_KV_WINDOW` -> `--local_attn_size`,
`LINGBOT_KV_SINK` -> `--sink_size`; unset = baseline):

```
LINGBOT_KV_WINDOW=12 LINGBOT_KV_SINK=6 python3 bench/window/window_sweep.py --dump_dir /path/to/dumps
```

On the pod, with a GPU, `--run` actually executes the commands (refuses if `torch.cuda.is_available()`
is false instead of failing silently):

```
python3 bench/window/window_sweep.py --windows 18,14,12,10,9 --dump_dir /path/to/dumps --run
```

Each printed line is independently runnable, e.g.:

```
LINGBOT_DUMP_LATENTS=/path/to/dumps/w12_s6.pt python3 generate.py --image examples/03/image.jpg \
  --action_path examples/03 --prompt "..." --save_dir outputs --frame_num 193 --bench \
  --local_attn_size 12 --sink_size 6 --preset fast --base_seed 42
```

`--bench` (`generate.py:270-297`) prints steady-state s/chunk, denoise-loop FPS, and as-played
FPS (with VAE decode) directly — this reproduces exactly the numbers OPTIMIZATIONS.md §18 uses
("Ours" table: 0.98 s/chunk, 16.1 FPS), so the sweep's measured rows are directly comparable to
that table without any extra parsing step.

## Quality gate (bench-world-model-quality skill)

This lever changes the rollout (shorter memory -> different attention output -> different next
latent), the same class as FP8 (§4) and SageAttention (§5) in OPTIMIZATIONS.md: **class C,
measured quality trade**, trajectory diverges from the first differing chunk, so bit/latent
identity is the wrong bar. Reuse §4/§5's method, not trajectory identity:

1. **Noise band first.** Run the *unmodified* baseline (`local_attn_size=18, sink_size=6`) three
   times, same seed, before judging any candidate — `window_sweep.py`'s Step 0 commands do this.
   That is what "inside noise" means for every column below (per the skill: two runs of the
   FP8+Sage+compile stack already differ by mean |Δ| ≈ 9.6/255 run-to-run).
2. **Record and score in the same run**, same seed as the noise-band runs, `LINGBOT_DUMP_LATENTS`
   set per candidate (Step 1 commands) — this repo's own hook, no separate capture pass needed.
   `generate.py` also writes the decoded clip via `save_video` in the same run.
3. **Metrics, not identity.** First-chunk PSNR/LPIPS against the reference decode (should sit
   inside the noise band — a window change should not move the *first* chunk much, since the
   cache isn't even full yet); MUSIQ / CLIP-IQA / sharpness / colour on the full clip; and
   critically for *this* lever, **drift**: bin the clip into ~160-frame windows and compare each
   bin's metrics against the reference's worst bin (the skill's "Schedule / chunk change over
   60 s" row) — a shrunk rolling window is specifically expected to hurt long-range coherence
   (things the sink didn't happen to capture "falling out of memory" faster), which is exactly
   what per-bin drift measures and first-chunk PSNR does not.
4. **This repo has no metrics libtools installed** (no lpips/pyiqa/skimage in `pyproject.toml`,
   no `quality_metrics.py`/`score_run.py` — those live in the `lingbot-world-v2-stream` repo per
   OPTIMIZATIONS.md's own tooling references). Loading them (or bringing over
   `stream/tools/score_run.py`) is a prerequisite the GPU-side runner needs to do before scoring;
   this script only gets the same latents/video on disk for both arms.
5. **Sample size.** Go/no-go: n=1, one scene (`ex03`/"lake"), the `--frame_num 193` clip §18
   already used. A "safe to ship" verdict needs the skill's full bar: 3 scenes × 3 seeds × 60 s.

## Predicted numbers (to falsify against the measured `--bench` output)

Baseline measured (OPTIMIZATIONS.md §18, pod 14): q 6032 × kv 27144, attention 0.288 s of a
0.98 s chunk, 16.1 FPS as played. Linear-in-kv scaling of the attention term is not an
assumption of convenience: §19 hypothesis 3 (`bench/attn/h3_l2_working_set.py`) independently
measured attention's per-key-per-head cost as flat (6.5 ± 0.2 ns) from 4k to 54k keys, i.e. the
kernel really is close to linear in kv length at this shape family. Everything else in the
chunk (DiT matmuls, decoder, elementwise) is held at its measured baseline cost.

| local_attn_size | sink_size | rolling (latents) | kv (tokens) | pred. attn | pred. chunk | pred. FPS | note |
|---:|---:|---:|---:|---:|---:|---:|---|
| 18 (baseline) | 6 | 12 | 27144 | 0.288 s | 0.980 s | 16.10 | — |
| 14 | 6 | 8 | 21112 | 0.224 s | 0.916 s | 17.22 | safe |
| 12 | 6 | 6 | 18096 | 0.192 s | 0.884 s | 17.85 | safe |
| 10 | 6 | 4 | 15080 | 0.160 s | 0.852 s | 18.52 | rolling == chunk size, minimum non-degenerate |
| 9 | 6 | 3 | 13572 | 0.144 s | 0.836 s | 18.87 | **degenerate**: rolling < chunk size |

Reproduce/extend this table: `python3 bench/window/window_sweep.py --windows <csv>`.

Note the “+17 %” figure quoted for “halving the window” corresponds to `local_attn_size=9`
(13 572 kv ≈ half of 27 144), which is exactly the degenerate row — every chunk would then evict
more than a full chunk's worth of the 3-latent rolling region on every step, i.e. almost no
cross-chunk memory beyond the sink. The nearest *non-degenerate* point, `local_attn_size=10`,
predicts +15.0% FPS (16.10 -> 18.52), not +17%, and even that is the most aggressive setting
that doesn't obviously break the memory mechanism — it should be treated as a stress point to
validate quality on, not the recommended default.

## What could not be verified without a GPU

- Whether the *measured* attention time actually scales linearly with kv length as predicted —
  §19's flat per-key-cost result is from a standalone attention microbench at one kv length
  range on one GPU (pod 15); it was not re-measured at kv < 27144 by this candidate.
- Whether `torch.compile(dynamic=True)` (`wan/image2video.py:430-439`) recompiles cleanly for
  each new `kv_size` within the raised `recompile_limit=64` (`wan/image2video.py:348`) without
  hitting the limit or a shape-guard failure — dynamic-shape compile behavior can only be
  observed on hardware; the mechanism read from the source is `dynamic=True` on the regional
  `torch.compile` calls, so each `local_attn_size` should trigger at most one first-call
  compile/recompile, not a crash, but that is inferred from source, not measured.
- Any SageAttention (`LINGBOT_ATTN=sage`) tile-alignment interaction with a smaller kv length —
  the shipped kernel already runs at kv=27144 (not a multiple of the 128-key tile), so
  misalignment is not new at smaller kv, but no run was done to confirm the padding path handles
  every candidate kv size (15080, 18096, 21112, 13572) without a new failure mode.
- Actual quality numbers — no metrics tooling is installed in this repo (see gate step 4 above)
  and no run was executed.
