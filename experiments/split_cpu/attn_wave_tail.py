"""Is card 0's SageAttention wave tail real? Times the attention kernel alone (qk_int_sv_f8_attn_kernel, without
Sage's quantization kernels) for several head counts on one full GPU.

Sage uses 255 registers per thread -> 8 warps per SM -> 1,360 warp slots on 170 SMs. Each head of 6,032 queries is
48 blocks x 4 warps = 192 warps, so 7 heads (1,344 warps) fit one wave and 10 heads (1,920) need 1.41 waves.
  tail real : t(10) / t(7) ~ 2.0 (the second wave costs a full wave)   -> attention work on card 0 is worth it
  no tail   : t(10) / t(7) ~ 10/7 = 1.43 (time scales with work)        -> the 60 % vs 77 % gap has another cause

  python experiments/split_cpu/attn_wave_tail.py            # GPU 0, heads 6..12
"""
import argparse

import torch
from sageattention import sageattn

ap = argparse.ArgumentParser()
ap.add_argument("--heads", type=int, nargs="+", default=[6, 7, 8, 9, 10, 11, 12])
ap.add_argument("--device", type=int, default=0)
a = ap.parse_args()
torch.cuda.set_device(a.device)
dev = f"cuda:{a.device}"
Lq, Lk, d = 6032, 27144, 128
sms = torch.cuda.get_device_properties(dev).multi_processor_count
slots = sms * 8  # warps per SM at 255 registers/thread


def kernel_ms(h):
    q = torch.randn(1, h, Lq, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, h, Lk, d, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, h, Lk, d, device=dev, dtype=torch.bfloat16)
    for _ in range(3):
        sageattn(q, k, v, tensor_layout="HND")
    torch.cuda.synchronize()
    n = 20
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            sageattn(q, k, v, tensor_layout="HND")
        torch.cuda.synchronize()
    us = sum(e.device_time_total for e in prof.key_averages() if "qk_int_sv_f8_attn_kernel" in e.key)
    return us / n / 1e3


t = {h: kernel_ms(h) for h in a.heads}
print(f"GPU {a.device}: {sms} SMs, {slots} warp slots at 8 warps/SM")
for h in a.heads:
    warps = h * 48 * 4
    waves = warps / slots
    print(f"  {h:2d} heads: {t[h]:6.3f} ms  warps {warps:5d}  waves {waves:4.2f} (runs as {-(-warps // slots)})  "
          f"ms per head {t[h] / h:.3f}")
if 7 in t and 10 in t:
    r = t[10] / t[7]
    print(f"t(10)/t(7) = {r:.2f}  (no tail ~1.43, full tail ~2.0) -> "
          + ("TAIL REAL" if r > 1.7 else "NO MEANINGFUL TAIL" if r < 1.55 else "PARTIAL TAIL"))
