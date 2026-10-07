#!/usr/bin/env python3
"""H2: time one SageAttention tile configuration (built by build.sh) on the model shapes.

Times sageattn() (sm_120 -> sageattn_qk_int8_pv_fp8_cuda, per_warp, fp32+fp16) on
q [1,6032,12,128], k/v [1,27144,12,128] bf16 NHD, sm_scale 1/sqrt(128), non-causal:
20 warm + 50 timed iterations with CUDA events, output checked against fp32 torch SDPA (cosine, max abs).
Records the compiled (CTA_Q, WARP_Q, CTA_K) reported by _qattn_sm89.h2_tile_config() when present.

Pod:  /workspace/sage_h2/venv_<cfg>/bin/python bench/attn/h2_tiles/bench.py --cfg <cfg>
Dry:  python3 bench/attn/h2_tiles/bench.py --cfg cfg_a --dry     (CPU, fake sageattention, tiny shapes)
Writes bench/attn/results/h2_<cfg>.json
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
RESULTS = os.path.join(HERE, "..", "results")
PEAK_INT8 = 838.0


def install_fake_sageattention(cfg):
    """CPU stand-in with the same surface bench.py touches: sageattn() and _qattn_sm89.h2_tile_config()."""
    def sageattn(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None, **kw):
        qt, kt, vt = (t.transpose(1, 2).float() for t in (q, k, v))
        o = F.scaled_dot_product_attention(qt, kt, vt, scale=sm_scale)
        return o.transpose(1, 2).to(q.dtype)
    tiles = {"cfg_base": (128, 32, 64), "cfg_a": (64, 16, 64), "cfg_b": (64, 16, 128),
             "cfg_c": (128, 16, 64), "cfg_d": (128, 16, 128)}
    m = types.ModuleType("sageattention")
    m.sageattn = sageattn
    q = types.ModuleType("sageattention._qattn_sm89")
    q.h2_tile_config = lambda: tiles.get(cfg, (128, 32, 64))
    m._qattn_sm89 = q
    sys.modules["sageattention"] = m
    sys.modules["sageattention._qattn_sm89"] = q


def compiled_tile_config():
    try:
        from sageattention import _qattn_sm89
        if hasattr(_qattn_sm89, "h2_tile_config"):
            return tuple(int(x) for x in _qattn_sm89.h2_tile_config())
    except ImportError:
        pass
    return None


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
    p.add_argument("--cfg", required=True, help="configuration name, used for the results file name")
    p.add_argument("--dry", action="store_true", help="CPU, fake sageattention, tiny shapes; exit 0")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warm", type=int, default=20)
    p.add_argument("--shapes", default=None, help="B,SQ,SK,H,D (default 1,6032,27144,12,128; dry 1,64,96,2,32)")
    p.add_argument("--out", default=None, help="default bench/attn/results/h2_<cfg>.json")
    a = p.parse_args()
    out_path = a.out or os.path.join(RESULTS, f"h2_{a.cfg}.json")

    if a.dry:
        install_fake_sageattention(a.cfg)
        dev = torch.device("cpu")
        shapes = a.shapes or "1,64,96,2,32"
    else:
        dev = torch.device("cuda")
        shapes = a.shapes or "1,6032,27144,12,128"
    import sageattention as sa

    tiles = compiled_tile_config()
    B, SQ, SK, H, D = parse_shapes(shapes)
    sm_scale = 1.0 / math.sqrt(D)
    flops = 4.0 * B * SQ * SK * H * D
    bench = make_timer(dev)
    torch.manual_seed(0)
    q = torch.randn(B, SQ, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, SK, H, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, SK, H, D, device=dev, dtype=torch.bfloat16)
    print(f"cfg {a.cfg}  compiled tiles (CTA_Q, WARP_Q, CTA_K) {tiles}  sageattention {getattr(sa, '__file__', '?')}")
    print(f"shapes q {tuple(q.shape)} k/v {tuple(k.shape)} bf16 NHD  FLOPs/call {flops:.4e}  device {dev}  "
          f"warm {a.warm} iters {a.iters}")
    ref = sdpa_fp32_ref(q, k, v, sm_scale)

    fn = lambda: sa.sageattn(q, k, v, tensor_layout="NHD", is_causal=False, sm_scale=sm_scale)  # noqa: E731
    out = fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    assert out.shape == q.shape, (out.shape, q.shape)
    max_abs, cos = err_stats(out, ref)
    del out
    ms = bench(fn, a.warm, a.iters)
    tops = flops / (ms * 1e-3) / 1e12
    if tiles:
        cta_q, warp_q, cta_k = tiles
        ctas = math.ceil(SQ / cta_q) * H * B
        warps_per_cta = cta_q // warp_q
    else:
        ctas = warps_per_cta = None
    print(f"{'ms/call':>9s} {'TOPS':>8s} {'%int8pk':>7s} {'max|err|':>10s} {'cos':>9s}   ctas warps/cta")
    print(f"{ms:9.4f} {tops:8.1f} {100 * tops / PEAK_INT8:6.1f}% {max_abs:10.3e} {cos:9.6f}   {ctas} {warps_per_cta}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dict(hypothesis="H2 tile sizes", cfg=a.cfg, tiles=tiles, dry=a.dry, device=str(dev),
                       shapes=dict(B=B, SQ=SQ, SK=SK, H=H, D=D), warm=a.warm, iters=a.iters,
                       flops_per_call=flops, peak_int8_tops=PEAK_INT8, ms=ms, tops=tops,
                       pct_peak=100 * tops / PEAK_INT8, max_abs=max_abs, cos=cos, ctas=ctas,
                       warps_per_cta=warps_per_cta, torch=torch.__version__), f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
