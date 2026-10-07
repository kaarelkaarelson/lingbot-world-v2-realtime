#!/usr/bin/env python3
"""Classify each GPU as GPU-bound, CPU (launch)-bound or dependency-bound
from a torch.profiler Chrome trace.

Usage: python cpu_gpu_bound.py trace.json [--start-ms A --end-ms B] [--json out.json]
       python cpu_gpu_bound.py --selftest
"""
import argparse
import json
import statistics
from collections import defaultdict

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
LEAD_THRESH_US = 20.0
GAP_THRESH_US = 50.0
BUSY_GPU_BOUND = 0.90  # idle below 10% of the window means GPU-bound


def load_events(trace):
    """Return (gpu_events, runtime_calls_by_correlation, all_runtime_calls)."""
    gpu, runtime = [], {}
    for e in trace["traceEvents"]:
        cat, args = e.get("cat"), e.get("args", {})
        if cat in GPU_CATS and "device" in args and "dur" in e:
            gpu.append({"dev": args["device"], "ts": e["ts"], "end": e["ts"] + e["dur"],
                        "name": e.get("name", "?"), "corr": args.get("correlation")})
        elif cat == "cuda_runtime" and "dur" in e:
            runtime[args.get("correlation")] = e
    return gpu, runtime


def pick_window(gpu, start_ms, end_ms):
    t0 = min(e["ts"] for e in gpu)
    t1 = max(e["end"] for e in gpu)
    if start_ms is not None or end_ms is not None:
        lo = t0 + (start_ms or 0.0) * 1e3
        hi = t0 + end_ms * 1e3 if end_ms is not None else t1
        return lo, hi
    span = t1 - t0  # default: middle 50% skips warm-up / compile
    return t0 + 0.25 * span, t0 + 0.75 * span


def merge_intervals(events):
    """Union of intervals as [start, end, first_event, last_event] (last = latest-ending)."""
    merged = []
    for e in sorted(events, key=lambda e: e["ts"]):
        if merged and e["ts"] <= merged[-1][1]:
            if e["end"] > merged[-1][1]:
                merged[-1][1], merged[-1][3] = e["end"], e
        else:
            merged.append([e["ts"], e["end"], e, e])
    return merged


def analyze_device(events, runtime, lo, hi):
    window = hi - lo
    in_win = [e for e in events if e["end"] > lo and e["ts"] < hi]
    merged = merge_intervals(in_win)
    busy = sum(max(0.0, min(m[1], hi) - max(m[0], lo)) for m in merged)

    # Step 1: launch lead = GPU start - launching runtime call start.
    leads = []
    for e in in_win:
        call = runtime.get(e["corr"])
        if call is not None and lo <= e["ts"] < hi:
            leads.append(e["ts"] - call["ts"])

    # Step 2: idle gaps between consecutive busy intervals.
    gaps, cpu_wait, dep_wait = [], 0.0, 0.0
    for prev, nxt in zip(merged, merged[1:]):
        gap_start, gap_end = prev[1], nxt[0]
        gap = gap_end - gap_start
        if gap <= GAP_THRESH_US or gap_start < lo or gap_end > hi:
            continue
        call = runtime.get(nxt[2]["corr"])
        # Not even submitted when the GPU went idle -> waiting for the CPU.
        kind = "cpu" if call is not None and call["ts"] >= gap_start else "dependency"
        if kind == "cpu":
            cpu_wait += gap
        else:
            dep_wait += gap
        gaps.append({"dur_us": gap, "kind": kind, "before": prev[3]["name"], "after": nxt[2]["name"]})
    gaps.sort(key=lambda g: -g["dur_us"])

    idle_frac = 1.0 - busy / window
    if busy / window >= BUSY_GPU_BOUND:
        verdict = "GPU-bound"
    elif cpu_wait >= dep_wait:
        verdict = "CPU-bound"
    else:
        verdict = "dependency-bound"
    return {
        "busy_pct": 100 * busy / window,
        "idle_pct": 100 * idle_frac,
        "n_events": len(in_win),
        "median_lead_us": statistics.median(leads) if leads else None,
        "pct_lead_lt_20us": 100 * sum(x < LEAD_THRESH_US for x in leads) / len(leads) if leads else None,
        "idle_waiting_cpu_us": cpu_wait,
        "idle_waiting_dependency_us": dep_wait,
        "top_gaps": gaps[:5],
        "verdict": verdict,
    }


def analyze(trace, start_ms=None, end_ms=None):
    gpu, runtime = load_events(trace)
    if not gpu:
        raise SystemExit("no GPU events found in trace")
    lo, hi = pick_window(gpu, start_ms, end_ms)
    t0 = min(e["ts"] for e in gpu)
    by_dev = defaultdict(list)
    for e in gpu:
        by_dev[e["dev"]].append(e)

    launch_us = defaultdict(float)  # per CPU thread: time inside cuda_runtime calls
    for call in runtime.values():
        if lo <= call["ts"] < hi:
            launch_us[call.get("tid")] += call["dur"]

    return {
        "window_ms": [(lo - t0) / 1e3, (hi - t0) / 1e3],
        "devices": {d: analyze_device(ev, runtime, lo, hi) for d, ev in sorted(by_dev.items())},
        "cuda_runtime_us_by_tid": dict(launch_us),
    }


def fmt(x, spec=".1f"):
    return "n/a" if x is None else format(x, spec)


def print_report(res):
    lo, hi = res["window_ms"]
    win_us = (hi - lo) * 1e3
    print(f"window: {lo:.2f} .. {hi:.2f} ms (relative to first GPU event), {hi - lo:.2f} ms long")
    for dev, r in res["devices"].items():
        print(f"\n== GPU {dev} ==  ({r['n_events']} GPU events in window)")
        print(f"busy: {r['busy_pct']:.1f}%   median launch lead: {fmt(r['median_lead_us'])} us   "
              f"kernels with lead < {LEAD_THRESH_US:.0f} us: {fmt(r['pct_lead_lt_20us'])}%")
        print(f"idle (gaps > {GAP_THRESH_US:.0f} us): waiting for CPU {r['idle_waiting_cpu_us'] / 1e3:.2f} ms, "
              f"waiting for dependency {r['idle_waiting_dependency_us'] / 1e3:.2f} ms")
        for g in r["top_gaps"]:
            print(f"  gap {g['dur_us']:9.1f} us [{g['kind']:>10}]  {g['before'][:45]}  ->  {g['after'][:45]}")
        print(f"verdict: {r['verdict']}")
    print("\nCPU time inside cuda_runtime calls in window (launch overhead):")
    for tid, us in sorted(res["cuda_runtime_us_by_tid"].items(), key=lambda kv: -kv[1]):
        print(f"  tid {tid}: {us / 1e3:.2f} ms ({100 * us / win_us:.1f}% of window)")


def _fake_trace():
    """dev 0: launch-bound (lead 5 us, launch after gap start). dev 1: GPU-bound (lead 2 ms)."""
    base, corr, ev = 1e6, 0, []

    def add(dev, start, dur, launch_ts, tid=1):
        nonlocal corr
        corr += 1
        ev.append({"cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": launch_ts, "dur": 4,
                   "tid": tid, "args": {"correlation": corr}})
        ev.append({"cat": "kernel", "name": f"k{dev}_{corr}", "ts": start, "dur": dur,
                   "args": {"device": dev, "stream": 7, "correlation": corr}})

    for i in range(400):
        start = base + 255 * i  # 100 us kernel, 155 us gap, launch 5 us before start
        add(0, start, 100, start - 5)
    for i in range(1020):  # same ~100 ms span as dev 0
        start = base + 100 * i  # back-to-back kernels, enqueued 2 ms ahead
        add(1, start, 100, start - 2000)
    return {"traceEvents": ev}


def selftest():
    res = analyze(_fake_trace())
    d0, d1 = res["devices"][0], res["devices"][1]
    assert d0["verdict"] == "CPU-bound", d0
    assert d0["idle_waiting_cpu_us"] > 10 * max(d0["idle_waiting_dependency_us"], 1.0), d0
    assert d0["median_lead_us"] == 5 and d0["pct_lead_lt_20us"] == 100, d0
    assert d1["verdict"] == "GPU-bound", d1
    assert d1["median_lead_us"] == 2000 and d1["busy_pct"] > 99, d1
    print_report(res)
    print("\nselftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="?")
    ap.add_argument("--start-ms", type=float, help="window start, relative to first GPU event")
    ap.add_argument("--end-ms", type=float, help="window end, relative to first GPU event")
    ap.add_argument("--json", help="write the numbers to this file")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.trace:
        ap.error("trace path required (or --selftest)")
    with open(args.trace) as f:
        res = analyze(json.load(f), args.start_ms, args.end_ms)
    print_report(res)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
