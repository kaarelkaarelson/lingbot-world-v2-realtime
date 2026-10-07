#!/usr/bin/env python3
"""H5: does the fp32+fp16 PV accumulate (periodic fp16->fp32 flush) of the sm89 FP8 kernel cost time on sm_120?

Times every SageAttention CUDA kernel variant reachable through the public API at the model's shapes
(q [1,6032,12,128], k/v [1,27144,12,128] bf16, NHD, non-causal):
  sageattn_qk_int8_pv_fp8_cuda   pv_accum_dtype in {fp32, fp32+fp16, fp32+fp32} x qk_quant_gran {per_warp, per_thread}
  sageattn_qk_int8_pv_fp16_cuda  pv_accum_dtype in {fp16, fp32, fp16+fp32}     x qk_quant_gran {per_warp, per_thread}
plus sageattn() as the production baseline. Combinations that raise are recorded, not fatal.
Each variant's output is compared with an fp32 torch SDPA reference (max abs, cosine).

Upstream signatures (thu-ml/SageAttention @ d1a57a5, sageattention/core.py):
  sageattn_qk_int8_pv_fp8_cuda (q, k, v, tensor_layout="HND", is_causal=False, qk_quant_gran="per_thread",
                                sm_scale=None, pv_accum_dtype="fp32+fp16", smooth_k=True, smooth_v=False, return_lse=False)
  sageattn_qk_int8_pv_fp16_cuda(q, k, v, tensor_layout="HND", is_causal=False, qk_quant_gran="per_thread",
                                sm_scale=None, pv_accum_dtype="fp32",      smooth_k=True, smooth_v=False, return_lse=False)
  sageattn() on sm_120 -> sageattn_qk_int8_pv_fp8_cuda(..., qk_quant_gran="per_warp", pv_accum_dtype="fp32+fp16")

Pod:  python3 bench/attn/h5_pv_accum.py
Dry:  python3 bench/attn/h5_pv_accum.py --dry     (CPU, fake sageattention, tiny shapes)
"""
import argparse
import json
import math
import os
import sys
import time
import types

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PEAK_INT8 = 838.0

FP8_ACCUM = ["fp32", "fp32+fp16", "fp32+fp32"]
FP16_ACCUM = ["fp16", "fp32", "fp16+fp32"]
GRANS = ["per_warp", "per_thread"]


def install_fake_sageattention():
    """CPU stand-in: validates the accepted strings like upstream, then runs SDPA. One combo raises on purpose."""
    def _attn(q, k, v, sm_scale, accepted, **kw):
        assert kw.get("qk_quant_gran", "per_thread") in GRANS, "qk_quant_gran must be either 'per_warp' or 'per_thread'."
        if kw.get("pv_accum_dtype") not in accepted:
            raise ValueError(f"Unsupported pv_accum_dtype: {kw.get('pv_accum_dtype')}")
        if kw.get("qk_quant_gran") == "per_thread" and kw.get("pv_accum_dtype") == "fp16":
            raise RuntimeError("dry: simulated triton failure")
        qt, kt, vt = (t.transpose(1, 2).float() for t in (q, k, v))
        o = F.scaled_dot_product_attention(qt, kt, vt, scale=sm_scale)
        return o.transpose(1, 2).to(q.dtype)
    m = types.ModuleType("sageattention")
    m.sageattn = lambda q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None, **kw: \
        _attn(q, k, v, sm_scale, FP8_ACCUM, pv_accum_dtype="fp32+fp16", qk_quant_gran="per_warp")
    m.sageattn_qk_int8_pv_fp8_cuda = lambda q, k, v, sm_scale=None, **kw: _attn(q, k, v, sm_scale, FP8_ACCUM, **kw)
    m.sageattn_qk_int8_pv_fp16_cuda = lambda q, k, v, sm_scale=None, **kw: _attn(q, k, v, sm_scale, FP16_ACCUM, **kw)
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


def sdpa_fp32_ref(q, k, v, sm_scale):
    """Per head so the fp32 [SQ, SK] score matrix stays under ~1 GB at the model shapes."""
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for h in range(q.size(2)):
        out[:, :, h] = F.scaled_dot_product_attention(
            q[:, :, h].float().unsqueeze(1), k[:, :, h].float().unsqueeze(1), v[:, :, h].float().unsqueeze(1),
            scale=sm_scale).squeeze(1)
    return out


def err_stats(out, ref):
    o, r = out.float().flatten(), ref.flatten()
    return (o - r).abs().max().item(), F.cosine_similarity(o, r, dim=0).item()


def parse_shapes(s):
    b, sq, sk, h, d = (int(x) for x in s.split(","))
    return b, sq, sk, h, d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true", help="CPU, fake sageattention, tiny shapes; exit 0")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warm", type=int, default=20)
    p.add_argument("--shapes", default=None, help="B,SQ,SK,H,D (default 1,6032,27144,12,128; dry 1,64,96,2,32)")
    p.add_argument("--out", default=os.path.join(HERE, "results", "h5.json"))
    a = p.parse_args()

    if a.dry:
        install_fake_sageattention()
        dev = torch.device("cpu")
        shapes = a.shapes or "1,64,96,2,32"
    else:
        dev = torch.device("cuda")
        shapes = a.shapes or "1,6032,27144,12,128"
    import sageattention as sa

    B, SQ, SK, H, D = parse_shapes(shapes)
    sm_scale = 1.0 / math.sqrt(D)
    flops = 4.0 * B * SQ * SK * H * D
    bench = make_timer(dev)
    torch.manual_seed(0)
    q = torch.randn(B, SQ, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, SK, H, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, SK, H, D, device=dev, dtype=torch.bfloat16)
    print(f"shapes q {tuple(q.shape)} k/v {tuple(k.shape)} bf16 NHD  FLOPs/call {flops:.4e}  device {dev}  "
          f"warm {a.warm} iters {a.iters}")
    ref = sdpa_fp32_ref(q, k, v, sm_scale)

    common = dict(tensor_layout="NHD", is_causal=False, sm_scale=sm_scale, smooth_k=True, smooth_v=False, return_lse=False)
    variants = [("sageattn() [prod: fp8, per_warp, fp32+fp16]",
                 lambda: sa.sageattn(q, k, v, tensor_layout="NHD", is_causal=False, sm_scale=sm_scale))]
    for gran in GRANS:
        for acc in FP8_ACCUM:
            variants.append((f"pv_fp8_cuda  {gran:10s} pv_accum={acc}",
                             lambda acc=acc, gran=gran: sa.sageattn_qk_int8_pv_fp8_cuda(
                                 q, k, v, qk_quant_gran=gran, pv_accum_dtype=acc, **common)))
    for gran in GRANS:
        for acc in FP16_ACCUM:
            variants.append((f"pv_fp16_cuda {gran:10s} pv_accum={acc}",
                             lambda acc=acc, gran=gran: sa.sageattn_qk_int8_pv_fp16_cuda(
                                 q, k, v, qk_quant_gran=gran, pv_accum_dtype=acc, **common)))

    rows = []
    for name, fn in variants:
        try:
            out = fn()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            max_abs, cos = err_stats(out, ref)
            del out
            ms = bench(fn, a.warm, a.iters)
            rows.append(dict(name=name, ok=True, ms=ms, tops=flops / (ms * 1e-3) / 1e12, max_abs=max_abs, cos=cos))
        except Exception as e:  # noqa: BLE001
            rows.append(dict(name=name, ok=False, error=f"{type(e).__name__}: {e}"))
            if dev.type == "cuda":
                torch.cuda.synchronize()

    print(f"\n{'variant':48s} {'ms/call':>9s} {'TOPS':>8s} {'%int8pk':>7s} {'max|err|':>10s} {'cos':>9s}")
    for r in rows:
        if r["ok"]:
            print(f"{r['name']:48s} {r['ms']:9.4f} {r['tops']:8.1f} {100 * r['tops'] / PEAK_INT8:6.1f}% "
                  f"{r['max_abs']:10.3e} {r['cos']:9.6f}")
        else:
            print(f"{r['name']:48s}   SKIPPED  {r['error'][:100]}")

    base = next((r for r in rows if r["ok"] and r["name"].startswith("sageattn()")), None)
    if base:
        print(f"\ndelta vs sageattn() ({base['ms']:.4f} ms):")
        for r in rows:
            if r["ok"] and r is not base:
                print(f"  {r['name']:48s} {r['ms'] - base['ms']:+8.4f} ms ({100 * (r['ms'] / base['ms'] - 1):+6.1f}%)")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(shapes=dict(B=B, SQ=SQ, SK=SK, H=H, D=D), warm=a.warm, iters=a.iters, dry=a.dry,
                       flops_per_call=flops, peak_int8_tops=PEAK_INT8, rows=rows), f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
