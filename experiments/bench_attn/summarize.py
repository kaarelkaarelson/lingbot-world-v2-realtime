#!/usr/bin/env python3
"""Collect bench/attn/results/*.json into one markdown table, plus the h7 log summary lines."""
import argparse
import glob
import json
import os
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

FAKE = {
    "h1_softmax_exposed.json": {"hypothesis": "H1 softmax exposed",
                                "measured": "attn kernel time with/without softmax",
                                "key_number": "softmax share 18 %"},
    "h3_l2_working_set.json": {"hypothesis": "H3 L2 working set",
                               "measured": "L2 hit rate vs KV length",
                               "key_number": "hit 61 % at 22 s"},
    "h4_wave_quantization.json": {"hypothesis": "H4 wave quantization",
                                  "measured": "tail-wave utilization",
                                  "key_number": "last wave 37 % full"},
    "h5_pv_accum.json": {"hypothesis": "H5 PV accumulate", "measured": "fp32 vs bf16 accum time",
                         "key_number": "1.12x"},
}


def describe(data):
    hyp = data.get("hypothesis", "?")
    measured = data.get("measured") or data.get("what") or "?"
    key = data.get("key_number")
    if key is None:
        # fall back to the first numeric top-level field
        nums = [(k, v) for k, v in data.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        key = f"{nums[0][0]}={nums[0][1]}" if nums else "?"
    return hyp, measured, key


def h7_lines(results_dir):
    out = []
    for name in ("h7_clock_sampler.log", "h7_bench.log"):
        path = os.path.join(results_dir, name)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            lines = [l.rstrip() for l in f if l.startswith("h7:")]
        if lines:
            out.append(f"**{name}**")
            out.extend(f"- {l}" for l in lines)
    return out


def summarize(results_dir):
    paths = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    print("| hypothesis | what was measured | key number | verdict |")
    print("|---|---|---|---|")
    for p in paths:
        try:
            with open(p) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"| {os.path.basename(p)} | unreadable: {e} | ? | — |")
            continue
        hyp, measured, key = describe(data)
        print(f"| {hyp} | {measured} | {key} | — |")
    if not paths:
        print("| (no result JSONs found) | | | — |")
    extra = h7_lines(results_dir)
    if extra:
        print()
        print("\n".join(extra))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    if args.dry:
        with tempfile.TemporaryDirectory() as d:
            for name, data in FAKE.items():
                with open(os.path.join(d, name), "w") as f:
                    json.dump(data, f)
            with open(os.path.join(d, "h7_clock_sampler.log"), "w") as f:
                f.write("h7: sm_clock_mhz median=2265 min=2250 max=2310\n"
                        "h7: effective_int8_peak_tops=788 (= 838 * 2265/2410)\n")
            summarize(d)
        return
    summarize(args.results)


if __name__ == "__main__":
    main()
