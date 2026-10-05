"""End-to-end throughput and latency from per-chunk frame-ready times (`--bench_e2e`).

The pipeline records a CUDA event when generation starts and one on the decoder stream when each
chunk's frames are decoded; nothing synchronizes inside the loop. From those times:

  throughput   16 frames / interval between consecutive chunks' frames being ready, steady state
               (chunk index 5 on: the KV window is full and every graph is compiled); p50/p95/p99
  first frame  generation start -> first chunk's frames ready, as Self-Forcing reports latency
  MPPS         megapixels per second, for comparison across resolutions

By default one rollout is measured (`--trials 1`, no warm-up); more trials and a warm-up rollout are
optional. With several trials the headline is the median of the per-trial values. Without a warm-up
the first-frame latency includes compilation and is reported as cold. The same numbers apply to one GPU and to any multi-GPU layout, since they only
look at when frames come out.
"""
import statistics

# chunk index; matches the --bench rule. Measured on one RTX 5090 (2026-10-05): compiles at chunks 0, 1, 4, the
# KV window full by chunk 4, flat from chunk 5 on (0.623-0.627 s)
STEADY_FROM = 5


def _pct(xs, q):
    xs = sorted(xs)
    i = (len(xs) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def trial_stats(ready_ms, frames_per_chunk, steady_from=STEADY_FROM):
    """ready_ms: per chunk, ms from generation start to that chunk's frames being decoded."""
    if len(ready_ms) <= steady_from:
        raise ValueError(f"need more than {steady_from} chunks for a steady state, got {len(ready_ms)}")
    intervals = [b - a for a, b in zip(ready_ms[steady_from - 1:], ready_ms[steady_from:])]
    p50 = _pct(intervals, 0.5)
    return dict(fps=frames_per_chunk / (p50 / 1000), interval_p50_ms=p50, interval_p95_ms=_pct(intervals, 0.95),
                interval_p99_ms=_pct(intervals, 0.99), first_frame_ms=ready_ms[0], steady_chunks=len(intervals))


def summarize(trials, frames_per_chunk, height, width):
    """trials: list of ready_ms lists. Returns per-trial stats and the median of the per-trial values."""
    per = [trial_stats(t, frames_per_chunk) for t in trials]
    med = {k: statistics.median(p[k] for p in per) for k in per[0] if k != "steady_chunks"}
    med["mpps"] = med["fps"] * height * width / 1e6
    return per, med


def format_lines(per, med, preset, gpus, height, width, cold=False):
    lines = [f"BENCH_E2E trial {i + 1}: {p['fps']:.2f} FPS (interval p50 {p['interval_p50_ms']:.1f} ms, "
             f"p95 {p['interval_p95_ms']:.1f}, p99 {p['interval_p99_ms']:.1f}; {p['steady_chunks']} steady chunks), "
             f"first frame {p['first_frame_ms']:.0f} ms" for i, p in enumerate(per)]
    what = "1 rollout" if len(per) == 1 else f"median of {len(per)} trials"
    lines.append(f"BENCH_E2E preset={preset} gpus={gpus} {width}x{height}, {what}: "
                 f"{med['fps']:.2f} FPS, {med['mpps']:.2f} MPPS, interval p95 {med['interval_p95_ms']:.1f} ms, "
                 f"first frame {med['first_frame_ms']:.0f} ms" + (" (cold: includes compile)" if cold else ""))
    return lines
