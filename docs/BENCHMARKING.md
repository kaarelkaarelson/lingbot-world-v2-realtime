# Benchmarking

How every speed number in this repository is measured, on one GPU or several. Report throughput and
latency together: a model is real time only if it produces frames faster than they play and the
first frame arrives quickly ([Self Forcing](https://self-forcing.github.io/static/self_forcing.pdf), §4).

## The two methods

**End-to-end (`--bench_e2e`), the headline.** Each chunk is decoded as soon as it is generated, on a
side stream, as `lingbot play` does. CUDA events mark the start of generation and the moment each
chunk's frames are decoded; nothing synchronizes inside the loop.

- **Throughput:** 16 frames ÷ the median interval between consecutive chunks' frames, over chunks 6+
  (index 5 on: the KV window is full and every graph is compiled). Also p95 and p99 of that interval.
- **First-frame latency:** start of generation → first chunk's frames decoded.
- **MPPS:** throughput × width × height, for comparison with models at other resolutions.
- One warm-up rollout is discarded; the headline is the median over `--trials` rollouts (default 3).

It only looks at when frames come out, so it applies unchanged to any multi-GPU layout, including
ones where the DiT and the decoder run at the same time on different cards.

**Per-chunk (`--bench`), for continuity.** A sync after every chunk times the DiT; the whole clip is
decoded once at the end and divided by the chunk count. FPS = 16 ÷ (median DiT chunk, chunks 6+,
+ decode per chunk). Every number published before October 2026 uses this method. It assumes the
decoder runs after the DiT, so it does not apply to layouts that overlap them. It is the only method
for `--preset stock`, because the paper's code decodes the whole clip at the end; there it times
every fifth DiT call, a chunk boundary, from outside without editing `reference/`.

The two must agree within noise on one GPU; check that before comparing anything else.

## Input-to-frame latency

For interactive use, the delay from a key press to the first frame that reflects it:
`lingbot play --headless-seconds 120` taps `W` on a schedule and prints key-to-present (`k2p`)
percentiles. Run it once per configuration that can play.

## Commands

    lingbot clip --frame_num 157 --bench_e2e --trials 1                  # ours, one GPU
    lingbot clip --frame_num 157 --bench                                  # same, the per-chunk method
    lingbot clip --frame_num 157 --bench --preset stock                   # the paper's code
    torchrun --nproc_per_node=2 -m lingbot.generate --preset stock --bench \
        --ulysses_size 2 --dit_fsdp --t5_fsdp --frame_num 157 ...         # the paper's code, two GPUs

All runs use example 03, seed 42, 832×464, `--frame_num 157` (10 chunks: 5 warm-up, 5 steady). On one
RTX 5090 chunk times are flat from the 6th chunk (0.623-0.627 s), so 5 steady chunks pin the median
to about 0.3%.

## Rules for comparisons

- Compare configurations only on the same machine: one-GPU baselines run on one card of the
  two-GPU pod, never on a different pod.
- Run `tools/podcheck.sh` before each session (clock, throttle reasons, PCIe link) and record it.
- Same driver, CUDA, PyTorch and kernel versions across every configuration.
- Report the median of trials with p95; one run is not a result.
- State what is included: our throughput includes the VAE decode; T5 is cached and the start image
  is encoded once per rollout.
