"""SageAttention register cap vs wave tail, kernel only: time sageattn on the split's shapes and compare outputs to the
uncapped reference saved by the first run (bit-identity: a register cap changes no math).

  python experiments/split_cpu/sage_regcap_bench.py --tag base   # with the uncapped wheel installed: saves reference
  python experiments/split_cpu/sage_regcap_bench.py --tag cap168 # after installing a capped wheel: times + compares
"""
import argparse
import os

import torch
from sageattention import sageattn

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--out", default="/workspace/runs/sagecap")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(0)
Lq, Lk, d = 6032, 27144, 128
shapes = {"card0_10h_170sm": (10, 0), "card1_2h": (2, 1)}
for name, (h, dev) in shapes.items():
    torch.cuda.set_device(dev)
    g = torch.Generator(device=f"cuda:{dev}").manual_seed(0)
    q = torch.randn(1, h, Lq, d, device=f"cuda:{dev}", dtype=torch.bfloat16, generator=g)
    k = torch.randn(1, h, Lk, d, device=f"cuda:{dev}", dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, h, Lk, d, device=f"cuda:{dev}", dtype=torch.bfloat16, generator=g)
    for _ in range(5):
        o = sageattn(q, k, v, tensor_layout="HND")
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(50):
        o = sageattn(q, k, v, tensor_layout="HND")
    e1.record(); torch.cuda.synchronize()
    ms = e0.elapsed_time(e1) / 50
    ref = os.path.join(a.out, f"ref_{name}.pt")
    if a.tag == "base":
        torch.save(o.cpu(), ref); same = "reference saved"
    else:
        r = torch.load(ref)
        same = "bit-identical" if torch.equal(o.cpu(), r) else f"DIFFERS max|d| {(o.cpu().float()-r.float()).abs().max().item():.3g}"
    flops = 4 * Lq * Lk * d * h
    print(f"{a.tag:8s} {name:16s} {ms:7.3f} ms  {flops / ms / 1e9:6.0f} TOPS  {same}")
