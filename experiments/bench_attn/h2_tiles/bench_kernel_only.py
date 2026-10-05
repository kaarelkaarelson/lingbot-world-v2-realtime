#!/usr/bin/env python3
"""Kernel-only time of one sageattn() call: torch.profiler over N calls, self-CUDA time of the attention kernel
(`qk_int_sv_f8_attn_kernel`) and of everything else (quant, transpose, pad), per call. Complements bench.py, which
times the whole sageattn() and therefore charges a config for extra host-side work such as V padding.

  <venv>/bin/python bench/attn/h2_tiles/bench_kernel_only.py --cfg cfg_b
"""
import argparse, json, math, os, time
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", required=True)
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--warm", type=int, default=10)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
a = ap.parse_args()

from sageattention import sageattn
dev = torch.device("cuda")
torch.manual_seed(0)
q = torch.randn(1, 6032, 12, 128, device=dev, dtype=torch.bfloat16)
k = torch.randn(1, 27144, 12, 128, device=dev, dtype=torch.bfloat16)
v = torch.randn(1, 27144, 12, 128, device=dev, dtype=torch.bfloat16)
scale = 1.0 / math.sqrt(128)
for _ in range(a.warm):
    sageattn(q, k, v, tensor_layout="NHD", is_causal=False, sm_scale=scale)
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(a.iters):
        sageattn(q, k, v, tensor_layout="NHD", is_causal=False, sm_scale=scale)
    torch.cuda.synchronize()
attn = other = 0.0
names = {}
for e in prof.key_averages():
    if e.device_time_total <= 0 or e.key.startswith("cuda") or e.key.startswith("aten::"):
        continue
    t = e.device_time_total / 1e3 / a.iters
    if "attn_kernel" in e.key:
        attn += t
    else:
        other += t
    names[e.key[:80]] = round(t, 4)
flops = 4.0 * 6032 * 27144 * 12 * 128
res = dict(cfg=a.cfg, attn_kernel_ms=round(attn, 4), other_kernels_ms=round(other, 4), total_ms=round(attn + other, 4),
           attn_tops=round(flops / (attn / 1e3) / 1e12, 1), attn_pct_of_838=round(100 * flops / (attn / 1e3) / 838e12, 1), kernels=names)
print(json.dumps({k: v for k, v in res.items() if k != "kernels"}))
for n, t in sorted(names.items(), key=lambda kv: -kv[1])[:8]:
    print(f"  {t:8.4f} ms  {n}")
os.makedirs(a.out, exist_ok=True)
json.dump(res, open(os.path.join(a.out, f"h2k_{a.cfg}.json"), "w"), indent=1)
