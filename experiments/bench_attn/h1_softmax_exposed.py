#!/usr/bin/env python3
"""H1: is the non-matmul (softmax) work of SageAttention's sm89 kernel exposed on sm_120?

Times, at the model's self-attention shapes (q [1,6032,12,128], k/v [1,27144,12,128] bf16, NHD):
  (a) sageattn()                              production call (INT8 QK + FP8 PV, fp32+fp16 accum)
  (b) pure INT8 QK^T  via torch._int_mm       per head, 12 heads
  (c) pure FP8  P@V   via torch._scaled_mm    per head, 12 heads (fp32 accumulate in cuBLASLt)
  (d) flash_attn_func bf16                    reference, skipped if flash_attn is not installed
and reports achieved TOPS vs the RTX 5090 peaks and the "exposed softmax" estimate
  sageattn ms - (QK ms + PV ms).

Pod:  python3 bench/attn/h1_softmax_exposed.py
Dry:  python3 bench/attn/h1_softmax_exposed.py --dry     (CPU, fake sageattention, tiny shapes)
"""
import argparse
import json
import math
import os
import sys
import time
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PEAK = {"int8": 838.0, "fp8_fp32acc": 419.0, "fp8_fp16acc": 838.0, "bf16": 209.5}  # TOPS / TFLOP/s, dense


def install_fake_sageattention():
    """CPU stand-in so --dry exercises the control flow without a GPU."""
    def sageattn(q, k, v, tensor_layout="NHD", is_causal=False, sm_scale=None, **kw):
        qt, kt, vt = (t.transpose(1, 2).float() for t in (q, k, v))
        o = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, scale=sm_scale)
        return o.transpose(1, 2).to(q.dtype)
    m = types.ModuleType("sageattention")
    m.sageattn = sageattn
    sys.modules["sageattention"] = m


def make_timer(dev):
    if dev.type == "cuda":
        def bench(fn, warm, iters):
            for _ in range(warm):
                fn()
            torch.cuda.synchronize()
            t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(iters):
                fn()
            t1.record()
            torch.cuda.synchronize()
            return t0.elapsed_time(t1) / iters
    else:
        def bench(fn, warm, iters):
            for _ in range(warm):
                fn()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            return (time.perf_counter() - t0) * 1e3 / iters
    return bench


def quant_int8(x):
    """Per-row symmetric int8 (rows = tokens); x [N, D]."""
    s = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
    return (x.float() / s).round().clamp(-127, 127).to(torch.int8), s


def parse_shapes(s):
    b, sq, sk, h, d = (int(x) for x in s.split(","))
    return b, sq, sk, h, d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true", help="CPU, fake sageattention, tiny shapes; exit 0")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warm", type=int, default=20)
    p.add_argument("--shapes", default=None, help="B,SQ,SK,H,D (default 1,6032,27144,12,128; dry 1,64,96,2,32)")
    p.add_argument("--out", default=os.path.join(HERE, "results", "h1.json"))
    a = p.parse_args()

    if a.dry:
        install_fake_sageattention()
        dev = torch.device("cpu")
        shapes = a.shapes or "1,64,96,2,32"
    else:
        dev = torch.device("cuda")
        shapes = a.shapes or "1,6032,27144,12,128"
    from sageattention import sageattn
    try:
        from flash_attn import flash_attn_func
    except Exception as e:  # noqa: BLE001
        flash_attn_func = None
        flash_err = repr(e)

    B, SQ, SK, H, D = parse_shapes(shapes)
    assert B == 1, "per-head loops below assume batch 1"
    sm_scale = 1.0 / math.sqrt(D)
    bench = make_timer(dev)
    torch.manual_seed(0)
    q = torch.randn(B, SQ, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, SK, H, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, SK, H, D, device=dev, dtype=torch.bfloat16)

    flops_qk = 2.0 * SQ * SK * D * H
    flops_pv = 2.0 * SQ * SK * D * H
    flops_attn = flops_qk + flops_pv
    print(f"shapes q {tuple(q.shape)} k/v {tuple(k.shape)} bf16 NHD  attn FLOPs/call {flops_attn:.4e}  "
          f"device {dev}  warm {a.warm} iters {a.iters}")

    rows = []  # (name, ms, flops, peak_key, note)

    # (a) sageattn
    ms = bench(lambda: sageattn(q, k, v, tensor_layout="NHD", is_causal=False, sm_scale=sm_scale), a.warm, a.iters)
    rows.append(("sageattn (INT8 QK + FP8 PV)", ms, flops_attn, "int8", "quant + kernel"))

    # (b) INT8 QK^T per head: q_int8[SQ,D] @ k_int8[SK,D].T -> mat2 is column-major, as cuBLASLt wants
    q8 = [quant_int8(q[0, :, h])[0].contiguous() for h in range(H)]
    k8t = [quant_int8(k[0, :, h])[0].contiguous().t() for h in range(H)]

    def qk_int8():
        for h in range(H):
            torch._int_mm(q8[h], k8t[h])
    try:
        ms = bench(qk_int8, a.warm, a.iters)
        rows.append(("INT8 QK^T  torch._int_mm x H", ms, flops_qk, "int8", "s32 out, no scaling"))
    except Exception as e:  # noqa: BLE001
        if not a.dry:
            raise
        print(f"[dry] _int_mm unavailable on CPU ({e!r}); using float matmul stand-in")
        ms = bench(lambda: [q8[h].float() @ k8t[h].float() for h in range(H)], a.warm, a.iters)
        rows.append(("INT8 QK^T  (dry float stand-in)", ms, flops_qk, "int8", "dry"))

    # (c) FP8 P@V per head. cuBLASLt fp8 needs K % 16 == 0: pad SK up (27144 -> 27152).
    SKp = (SK + 15) // 16 * 16
    p_fp8 = [torch.rand(SQ, SKp, device=dev).to(torch.float8_e4m3fn) for _ in range(H)]
    # mat2 must be column-major: build [D, SKp] row-major and transpose
    v_fp8 = [torch.randn(D, SKp, device=dev).to(torch.float8_e4m3fn).t() for _ in range(H)]
    one = torch.ones((), device=dev, dtype=torch.float32)
    flops_pv_pad = 2.0 * SQ * SKp * D * H

    def pv_fp8():
        for h in range(H):
            torch._scaled_mm(p_fp8[h], v_fp8[h], scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    try:
        ms = bench(pv_fp8, a.warm, a.iters)
        rows.append(("FP8 P@V   torch._scaled_mm x H", ms, flops_pv_pad, "fp8_fp32acc", f"K padded {SK}->{SKp}, fp32 acc"))
    except Exception as e:  # noqa: BLE001
        if not a.dry:
            raise
        print(f"[dry] _scaled_mm unavailable on CPU ({e!r}); using float matmul stand-in")
        ms = bench(lambda: [p_fp8[h].float() @ v_fp8[h].float() for h in range(H)], a.warm, a.iters)
        rows.append(("FP8 P@V   (dry float stand-in)", ms, flops_pv_pad, "fp8_fp32acc", "dry"))

    # (d) flash_attn bf16
    if flash_attn_func is not None:
        try:
            ms = bench(lambda: flash_attn_func(q, k, v, softmax_scale=sm_scale, causal=False), a.warm, a.iters)
            rows.append(("flash_attn_func bf16", ms, flops_attn, "bf16", ""))
        except Exception as e:  # noqa: BLE001
            print(f"flash_attn_func raised, skipped: {e!r}")
    else:
        print(f"flash_attn not importable, (d) skipped: {flash_err}")

    print(f"\n{'variant':36s} {'ms/call':>9s} {'TOPS':>9s} {'peak':>7s} {'%peak':>6s}  note")
    results = {"shapes": dict(B=B, SQ=SQ, SK=SK, H=H, D=D), "warm": a.warm, "iters": a.iters, "dry": a.dry,
               "peaks_tops": PEAK, "flops_attn_per_call": flops_attn, "rows": []}
    by = {}
    for name, ms, flops, peak_key, note in rows:
        tops = flops / (ms * 1e-3) / 1e12
        peak = PEAK[peak_key]
        print(f"{name:36s} {ms:9.4f} {tops:9.1f} {peak:7.1f} {100 * tops / peak:5.1f}%  {note}")
        by[name] = ms
        results["rows"].append(dict(name=name, ms=ms, flops=flops, tops=tops, peak_key=peak_key,
                                    peak=peak, pct_peak=100 * tops / peak, note=note))
        if peak_key == "fp8_fp32acc":
            print(f"{'':36s} {'':9s} {'':9s} {PEAK['fp8_fp16acc']:7.1f} {100 * tops / PEAK['fp8_fp16acc']:5.1f}%  vs fp16-acc peak")

    qk_ms = next(ms for n, ms in by.items() if n.startswith("INT8 QK"))
    pv_ms = next(ms for n, ms in by.items() if n.startswith("FP8 P@V"))
    sage_ms = by["sageattn (INT8 QK + FP8 PV)"]
    exposed = sage_ms - (qk_ms + pv_ms)
    ideal_ms = flops_attn / (PEAK["int8"] * 1e12) * 1e3
    print(f"\nexposed softmax estimate = sageattn - (QK + PV) = {sage_ms:.4f} - ({qk_ms:.4f} + {pv_ms:.4f}) "
          f"= {exposed:.4f} ms ({100 * exposed / sage_ms:.1f}% of sageattn)")
    print(f"ideal at INT8 peak: {ideal_ms:.4f} ms; matmul-only sum {qk_ms + pv_ms:.4f} ms")
    results.update(qk_ms=qk_ms, pv_ms=pv_ms, sage_ms=sage_ms, exposed_softmax_ms=exposed,
                   exposed_pct=100 * exposed / sage_ms, ideal_int8_peak_ms=ideal_ms)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
