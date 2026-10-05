"""lingbot/benchmark.py: throughput, percentiles and first-frame latency from frame-ready times."""
import pytest

from lingbot.benchmark import format_lines, summarize, trial_stats


def ready(first=1500.0, interval=980.0, n=22, spikes=()):
    t, out = first, [first]
    for i in range(1, n):
        t += interval + (300.0 if i in spikes else 0.0)
        out.append(t)
    return out


def test_steady_state_skips_warmup_chunks():
    # chunks 1-7 are slow (window filling); only intervals ending at chunk index >= 7 count
    r = ready(interval=980.0)
    r = [x + (2000.0 if i < 7 else 2000.0) for i, x in enumerate(r)]  # constant shift: intervals unchanged
    s = trial_stats(r, 16)
    assert s["steady_chunks"] == 15 and s["interval_p50_ms"] == pytest.approx(980.0)
    assert s["fps"] == pytest.approx(16 / 0.98)
    assert s["first_frame_ms"] == pytest.approx(3500.0)


def test_slow_chunks_show_in_p95_not_p50():
    s = trial_stats(ready(spikes={10}), 16)
    assert s["interval_p50_ms"] == pytest.approx(980.0)
    assert s["interval_p99_ms"] > 1200.0


def test_median_of_trials_and_mpps():
    per, med = summarize([ready(interval=1000.0), ready(interval=980.0), ready(interval=960.0)], 16, 464, 832)
    assert len(per) == 3 and med["fps"] == pytest.approx(16 / 0.98)
    assert med["mpps"] == pytest.approx(16 / 0.98 * 464 * 832 / 1e6)
    lines = format_lines(per, med, "fast", 2, 464, 832)
    assert lines[-1].startswith("BENCH_E2E preset=fast gpus=2 832x464, median of 3 trials: 16.33 FPS")


def test_too_few_chunks_rejected():
    with pytest.raises(ValueError):
        trial_stats(ready(n=7), 16)
