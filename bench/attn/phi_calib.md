# phi_calib — is candidate C5 ("fixed global max") usable on the real model?

C5 (`cfg_fixedmax`, FlashDecoding++'s "unified max value phi") replaces the online per-row max
in the SageAttention kernel with a constant `phi`. It is worth ~9 % on the 5090 at our shapes,
but only if one constant can serve every row. This script measures the real distribution of
attention-logit row maxima in the 1.3B causal DiT and decides that.

## Units — read this first

The kernel's exp2 argument (`csrc/qattn/attn_utils.cuh`, `apply_fixed_max`) is

```
exp2( S * sm_scale * log2(e) - phi + S_FP8_OFFSET )      S_FP8_OFFSET = 8.807
```

so `phi` is in **base-2 units**. Everything the script writes and prints is in the same units:

```
m_log2(row) = max over keys of (q . k) * sm_scale * log2(e)
```

`sm_scale` is `1/sqrt(128) = 0.08838835` (the model never passes `softmax_scale`, so Sage's
default applies; the script records the value it used in the `sm_scale` column).

**`phi` must be an upper bound on `m_log2` for every row it covers.** The usable window, from
the synthetic-input sweep, is

```
m_log2  <=  phi  <=  m_log2 + 3.5
```

- `phi < m_log2` → the exp2 argument exceeds `S_FP8_OFFSET`, `P > 448`, `halfx4_to_e4m3x4`
  clamps. Silent.
- `phi > m_log2 + 3.5` → the row's whole P block sits below the bottom of the e4m3 grid and
  flushes. Also silent.

Both failure modes are invisible in the output, which is why this has to be measured rather
than assumed. (`cfg_fixedmax` has a `SAGE_H2_SAT_CHECK` debug printf for the first case only.)

## Running it

On the pod, from the repo root, with the venv active:

```bash
cd ~/lingbot-world-v2-realtime && . .venv/bin/activate
PHI_CALIB_TSV=bench/attn/results/phi_calib.tsv \
  python bench/attn/phi_calib.py --run --frame_num 193 --bench
```

`--run` installs the hook and then runs `generate.py` **in-process** with exactly
`lingbot clip`'s defaults and environment — it reads `CLIP_DEFAULTS` and `_env()` straight out
of `lingbot/cli.py` and applies `lingbot/presets.py`, so it cannot drift from

```bash
lingbot clip --frame_num 193 --bench        # 12 chunks; what --run reproduces
```

In-process is required: `lingbot clip` launches `generate.py` as a subprocess, and installing
the hook into a subprocess without editing a repo file would mean dropping a `usercustomize.py`
somewhere on its path. Everything after `--run` is forwarded to `generate.py` verbatim, so
`--frame_num 361 --bench` gives you the `lingbot bench` 22 s clip instead.

The launcher forces `LINGBOT_TORCH_COMPILE=0` / `LINGBOT_INDUCTOR_TUNE=0`. The hook keeps
Python state per call, which dynamo would constant-fold, and it would try to trace the score
tiling into the graph. It also refuses `LINGBOT_ATTN=sage_kvq`, which bypasses `attention()`
entirely (the fused pre-quantised KV path); the `fast` preset's `LINGBOT_ATTN=sage` is what you
want. Neither affects the measured numbers — `q` and `k` are the same tensors either way.

Nothing in the repo is modified. `phi_calib.install()` rebinds
`wan.modules.attention.attention` (plus every `from .attention import attention` rebinding in
`wan.*`) and wraps the `forward` of each class whose signature takes `current_start`, i.e. the
`CausalWanSelfAttention` in `model_fast_fusion.py` / `model_fast.py` / `model_causal.py`.
The model's output is untouched: the hook measures, then returns `_orig_attention(...)`.

### Sampling controls

| flag | default | meaning |
|---|---|---|
| `--chunks` | `6-8` | chunk indices. 6-8 is steady state: the 18-latent KV window is full, `Lk = 27144`. |
| `--forwards` | `0` | forward within the chunk: `0-3` are the denoising steps, `4` is the cache-write forward. One of five, because the extra score pass roughly doubles that forward's cost. |
| `--layers` | `all` | all 30 DiT layers. |
| `--keys-tile` | `4096` | keys per score tile. |
| `--budget-mb` | `96` | memory cap for the live score block; the query tile is derived from it (`96 MB / (12 heads * 4096 keys * 4 B) = 512 rows`). |
| `--min-kv` | `4096` | skip `attention()` calls with fewer keys — excludes cross-attention if `LINGBOT_XATTN=sage` is ever set. |

Default cost: 30 layers x 3 chunks = 90 extra score passes, ~0.5 GFLOP-heavy each in fp32
(TF32 is disabled so the measured max is the true fp32 value, not a 10-bit-mantissa
approximation of a number we are budgeting to 3.5 log2 units). Expect the three sampled
forwards to take roughly twice as long as usual; the other 57 forwards of the run are
untouched.

Once the cheap run says something, widen it:

```bash
# all five forwards of one chunk — is the cache-write forward different?
python bench/attn/phi_calib.py --run --forwards all --chunks 7 --frame_num 193 --bench
```

The TSV is appended to, so successive runs accumulate; delete it to start fresh.

## Output

One row per `(chunk, forward, layer, head)`:

```
chunk  forward  layer  head  lq  lk  sm_scale  max_log2  p9999_log2  p999_log2  mean_log2  min_log2
```

The last five are per-head statistics **over the `Lq` row maxima** of that call, all in log2
units. `max_log2` is the number that matters — `phi` has to clear it. `p9999_log2` is there to
show whether the max is a lone outlier row (a big `max − p99.99` gap means a single-constant
phi is being set by one row and wastes headroom for everyone else); `mean`/`min` show how much
of the window the bulk of the rows are actually using.

## Analysing

```bash
python bench/attn/phi_calib.py --analyse bench/attn/results/phi_calib.tsv
```

prints the global max (the phi a single-constant kernel would need), the per-layer max and
p99.99, the per-layer spread, the number of `(layer, head)` pairs more than 3.5 log2 below a
single global phi, and a verdict.

## Decision rule

Let `G` = global max over everything, `Lmax(l)` = max over that layer, `H(l,h)` = max over that
(layer, head).

- **single global phi viable** — `G - min over all (l,h) of H(l,h) <= 3.5`.
  One constant covers everything. Ship `cfg_fixedmax` with `phi = ceil(G)` plus a small margin.
- **needs per-layer phi** — the above fails, but within every layer
  `max_h H(l,h) - min_h H(l,h) <= 3.5`. One constant per layer: 30 values, passed as a kernel
  argument from the Python side. Still deletes the whole online-max chain.
- **needs per-layer-per-head phi** — some single layer's heads span more than 3.5 log2. 360
  values, which is still a constant table but a much bigger claim about stability across
  prompts, scenes and chunk positions.

Margin: whatever phi you pick has to hold on *unseen* input, not just the sampled chunks. The
budget is 3.5 log2 total, so a phi set at `G + delta` leaves `3.5 - delta - (G - H(l,h))` for
the quietest head. Pick `delta` from the chunk-to-chunk spread the TSV shows, and if the
verdict is anything other than "single global phi viable", re-run over more chunks
(`--chunks all`) and a second scene before committing — the numbers here are one prompt.

## Self-test

```bash
python bench/attn/phi_calib.py --dry
```

fabricates a per-layer-drifting distribution, runs the whole analysis path against it and exits
0. CPU only, no torch, no model. Use it to check the analysis after editing it.
