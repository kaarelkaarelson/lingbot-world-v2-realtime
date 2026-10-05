#!/usr/bin/env python3
"""H3: the SageAttention self-attention kernel on sm_120 is compute-bound only while
K/V is served from L2.

Each CTA (128 query rows) streams all of K/V. Per 64-key tile that is 16 KB
(int8 K + fp8 V) for 4.2 MFLOP, ~256 FLOP/B, below the ~468 FLOP/B DRAM ridge of
the RTX 5090, so the kernel can only reach its INT8 peak if K/V (~7 MB per head at
Lk=27144) stays resident in the 96 MB L2. At H=12 that is ~84 MB, at the edge.

Sweeps (10 warm + 30 timed, CUDA events):
  (a) heads H at fixed Lq=6032, Lk=27144   -> ms per head; a step up past the H
      where the working set exceeds L2 is the signature
  (b) Lk at H=12                            -> per-token-per-head cost
  (c) Lk at H=1 (control: always fits L2)

Pod:  python3 bench/attn/h3_l2_working_set.py
Dry:  python3 bench/attn/h3_l2_working_set.py --dry   (CPU, fake kernel, tiny shapes)
"""
import argparse
import json
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
D = 128
LQ, LK = 6032, 27144
HEADS_SWEEP = [1, 2, 3, 4, 6, 8, 12, 16, 24]
LK_SWEEP = [4096, 8192, 13568, 20352, 27144, 40704, 54288]
INT8_PEAK_TOPS = 838.0  # RTX 5090 dense INT8/FP8 tensor peak
L2_BYTES_DEFAULT = 96 << 20


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
    p.add_argument("--out", default=os.path.join(HERE, "results", "h3.json"))
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


def run_case(Lq, Lk, H, warm, iters, dev, l2_bytes):
    import torch
    from sageattention import sageattn

    q = torch.randn(1, Lq, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, Lk, H, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, Lk, H, D, device=dev, dtype=torch.bfloat16)
    s = D ** -0.5
    ms = bench_ms(lambda: sageattn(q, k, v, tensor_layout="NHD", is_causal=False, sm_scale=s), warm, iters, dev)
    flops = 4 * Lq * Lk * H * D
    ws = H * Lk * D * 2  # quantised working set: int8 K + fp8 V
    return dict(Lq=Lq, Lk=Lk, H=H, ms=ms, ms_per_head=ms / H, ns_per_tok_head=ms * 1e6 / (Lk * H),
                tops=flops / ms / 1e9, pct_peak=100 * flops / ms / 1e9 / INT8_PEAK_TOPS,
                ws_mb=ws / 2**20, fits_l2=ws <= l2_bytes)


def print_table(title, rows):
    print(f"\n{title}")
    print(f"{'H':>3} {'Lq':>6} {'Lk':>6} {'ms':>9} {'ms/head':>9} {'ns/tok/head':>12} {'TOPS':>8} {'%peak':>6} {'K/V ws MB':>10} {'L2?':>4}")
    for r in rows:
        print(f"{r['H']:>3} {r['Lq']:>6} {r['Lk']:>6} {r['ms']:>9.4f} {r['ms_per_head']:>9.4f} {r['ns_per_tok_head']:>12.3f} "
              f"{r['tops']:>8.1f} {r['pct_peak']:>6.1f} {r['ws_mb']:>10.1f} {'yes' if r['fits_l2'] else 'NO':>4}")


def main():
    a = parse_args()
    if a.dry:
        install_fake_sageattention()
    import torch

    dev = "cpu" if a.dry else "cuda"
    torch.manual_seed(0)
    heads, lks, lq, lk = HEADS_SWEEP, LK_SWEEP, LQ, LK
    l2_bytes = L2_BYTES_DEFAULT
    if a.dry:
        lq, lk = 128, 256
        lks = [64, 128, 256, 512]
        l2_bytes = 24 * 256 * D * 2 // 2  # H=12 fits, H=16 does not: exercises the NO branch
        print(f"DRY RUN: cpu, fake sageattention, Lq={lq} Lk={lk}")
    else:
        props = torch.cuda.get_device_properties(0)
        l2_bytes = getattr(props, "L2_cache_size", L2_BYTES_DEFAULT)
        print(f"{props.name}: {props.multi_processor_count} SMs, L2 {l2_bytes / 2**20:.0f} MB, "
              f"torch {torch.__version__}, INT8 peak assumed {INT8_PEAK_TOPS:.0f} TOPS")
    print(f"warm {a.warm}, timed {a.iters}; FLOPs = 4*Lq*Lk*H*D; K/V working set = H*Lk*{D}*2 B (int8 K + fp8 V)")

    with torch.no_grad():
        sweep_h = [run_case(lq, lk, H, a.warm, a.iters, dev, l2_bytes) for H in heads]
        print_table(f"(a) heads sweep, Lq={lq} Lk={lk}", sweep_h)
        sweep_lk_h12 = [run_case(lq, L, 12, a.warm, a.iters, dev, l2_bytes) for L in lks]
        print_table(f"(b) Lk sweep, H=12, Lq={lq}", sweep_lk_h12)
        sweep_lk_h1 = [run_case(lq, L, 1, a.warm, a.iters, dev, l2_bytes) for L in lks]
        print_table(f"(c) Lk sweep, H=1 (control), Lq={lq}", sweep_lk_h1)

    out = dict(hypothesis="H3 L2 working set", dry=a.dry, device=dev, l2_bytes=l2_bytes,
               int8_peak_tops=INT8_PEAK_TOPS, warm=a.warm, iters=a.iters,
               heads_sweep=sweep_h, lk_sweep_h12=sweep_lk_h12, lk_sweep_h1=sweep_lk_h1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
