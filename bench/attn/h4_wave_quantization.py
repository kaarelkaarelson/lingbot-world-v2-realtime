#!/usr/bin/env python3
"""H4: wave quantisation of the SageAttention self-attention kernel on sm_120.

The kernel tiles queries in BLOCK_M=128 rows, one CTA per (query tile, head):
ceil(6032/128)=48 tiles x 12 heads = 576 CTAs on the RTX 5090's 170 SMs = 3.39
waves, so the last wave is only ~40 % full (assuming one resident CTA per SM).
If that tail is the loss, ms per query token should drop at Lq values that fill
the last wave and rise just past a tile boundary.

Sweep Lq at Lk=27144 for H=12, then H=1 and H=24 (wave counts change), reporting
ms, us per query token, achieved TOPS, CTAs = ceil(Lq/128)*H and waves = CTAs/SMs.
The SM count is read from torch.cuda.get_device_properties(0) (170 on the 5090).

Pod:  python3 bench/attn/h4_wave_quantization.py
Dry:  python3 bench/attn/h4_wave_quantization.py --dry   (CPU, fake kernel, tiny shapes)
"""
import argparse
import json
import math
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
D = 128
BLOCK_M = 128
LK = 27144
LQ_SWEEP = [4096, 5120, 5888, 6016, 6032, 6144, 6272, 7168, 8192, 12288]
HEADS = [12, 1, 24]
INT8_PEAK_TOPS = 838.0  # RTX 5090 dense INT8/FP8 tensor peak
SM_COUNT_5090 = 170


def install_fake_sageattention():
    import torch

    m = types.ModuleType("sageattention")
    m.sageattn = lambda q, k, v, tensor_layout="NHD", is_causal=False, sm_scale=None: torch.zeros_like(q)
    sys.modules["sageattention"] = m


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry", action="store_true", help="CPU control-flow check with a fake kernel and tiny shapes")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warm", type=int, default=10)
    p.add_argument("--out", default=os.path.join(HERE, "results", "h4.json"))
    return p.parse_args()


def bench_ms(fn, warm, iters, dev):
    import torch

    for _ in range(warm):
        fn()
    if dev == "cuda":
        torch.cuda.synchronize()
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            fn()
        t1.record()
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) / iters
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1e3 / iters


def run_case(Lq, Lk, H, sm_count, warm, iters, dev):
    import torch
    from sageattention import sageattn

    q = torch.randn(1, Lq, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, Lk, H, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, Lk, H, D, device=dev, dtype=torch.bfloat16)
    s = D ** -0.5
    ms = bench_ms(lambda: sageattn(q, k, v, tensor_layout="NHD", is_causal=False, sm_scale=s), warm, iters, dev)
    flops = 4 * Lq * Lk * H * D
    ctas = math.ceil(Lq / BLOCK_M) * H
    waves = ctas / sm_count
    last_wave_fill = waves - math.floor(waves) if waves != math.floor(waves) else 1.0
    return dict(Lq=Lq, Lk=Lk, H=H, ms=ms, us_per_qtok=ms * 1e3 / Lq, tops=flops / ms / 1e9,
                pct_peak=100 * flops / ms / 1e9 / INT8_PEAK_TOPS, ctas=ctas, waves=waves,
                last_wave_fill=last_wave_fill)


def print_table(title, rows):
    print(f"\n{title}")
    print(f"{'H':>3} {'Lq':>6} {'Lk':>6} {'ms':>9} {'us/qtok':>9} {'TOPS':>8} {'%peak':>6} {'CTAs':>6} {'waves':>7} {'last%':>6}")
    for r in rows:
        print(f"{r['H']:>3} {r['Lq']:>6} {r['Lk']:>6} {r['ms']:>9.4f} {r['us_per_qtok']:>9.4f} {r['tops']:>8.1f} "
              f"{r['pct_peak']:>6.1f} {r['ctas']:>6} {r['waves']:>7.2f} {100 * r['last_wave_fill']:>6.0f}")


def main():
    a = parse_args()
    if a.dry:
        install_fake_sageattention()
    import torch

    dev = "cpu" if a.dry else "cuda"
    torch.manual_seed(0)
    lqs, lk = LQ_SWEEP, LK
    if a.dry:
        sm_count = SM_COUNT_5090
        lqs = [128, 256, 384, 6032]
        lk = 256
        print(f"DRY RUN: cpu, fake sageattention, Lk={lk}, SM count assumed {sm_count}")
    else:
        props = torch.cuda.get_device_properties(0)
        sm_count = props.multi_processor_count
        print(f"{props.name}: {sm_count} SMs (RTX 5090 = {SM_COUNT_5090}), torch {torch.__version__}, "
              f"INT8 peak assumed {INT8_PEAK_TOPS:.0f} TOPS")
    print(f"warm {a.warm}, timed {a.iters}; FLOPs = 4*Lq*Lk*H*D; CTAs = ceil(Lq/{BLOCK_M})*H; "
          f"waves = CTAs/{sm_count} (1 CTA per SM assumed); last% = fill of the final wave")

    sweeps = {}
    with torch.no_grad():
        for H in HEADS:
            rows = [run_case(Lq, lk, H, sm_count, a.warm, a.iters, dev) for Lq in lqs]
            print_table(f"Lq sweep, H={H}, Lk={lk}", rows)
            sweeps[f"h{H}"] = rows

    out = dict(hypothesis="H4 wave quantization", dry=a.dry, device=dev, sm_count=sm_count, block_m=BLOCK_M,
               int8_peak_tops=INT8_PEAK_TOPS, warm=a.warm, iters=a.iters, sweeps=sweeps)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
