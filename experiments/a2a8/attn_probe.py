"""Deterministic check of the 8-bit exchange: one real forward's attention inputs, no sampling noise.

Dump (eager, no compile, fast numerics otherwise):
  LINGBOT_TORCH_COMPILE= LINGBOT_DUMP_QKV=/workspace/runs/qkv_dump python -m lingbot.generate --preset fast ...
Then: python experiments/a2a8/attn_probe.py /workspace/runs/qkv_dump

Per layer, relative L2 error of the attention output against exact attention (fp32 math):
  sage          SageAttention on the original inputs (what `fast` does today)
  sage+qk8      q and every key rounded as the 8-bit exchange would (keys per 6,032-token chunk)
  sage+qkv8     also v in FP8
and the change sage -> sage+qk8 itself.
"""
import glob
import os
import sys

import torch
from sageattention import sageattn

from lingbot.parallel import a2a_quant as a2aq

CHUNK = 6032


def per_chunk(x, fn):
    # keys in the cache arrived one chunk at a time, each split across the two cards
    t = x.shape[1]
    starts = list(range(t - CHUNK, -1, -CHUNK))[::-1]
    head = x[:, :starts[0]] if starts and starts[0] > 0 else x[:, :0]
    parts = ([fn(head)] if head.shape[1] else []) + [fn(x[:, s:s + CHUNK]) for s in starts]
    return torch.cat(parts, dim=1)


def exact(q, k, v, block=512):
    # fp32 math attention, in query blocks to bound memory; NHD -> HND
    qh, kh, vh = (t.float().transpose(1, 2) for t in (q, k, v))
    out = [torch.nn.functional.scaled_dot_product_attention(qh[:, :, i:i + block], kh, vh)
           for i in range(0, qh.shape[2], block)]
    return torch.cat(out, dim=2).transpose(1, 2)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


print("layer\tq tokens\tkv tokens\tsage vs exact\tsage+qk8 vs exact\tsage+qkv8 vs exact\tqk8 change vs sage")
for f in sorted(glob.glob(os.path.join(sys.argv[1], "layer*.pt"))):
    d = torch.load(f)
    q, k, v = (d[n].cuda() for n in ("q", "k", "v"))
    ref = exact(q, k, v)
    sg = lambda q_, k_, v_: sageattn(q_, k_, v_, tensor_layout="NHD")
    base = sg(q, k, v)
    q8 = a2aq.roundtrip_int8(q)
    k8 = per_chunk(k, lambda x: a2aq.roundtrip_int8(x, smooth=True))
    v8 = per_chunk(v, a2aq.roundtrip_fp8)
    qk = sg(q8, k8, v)
    qkv = sg(q8, k8, v8)
    print(f"{os.path.basename(f)[5:7]}\t{q.shape[1]}\t{k.shape[1]}\t{rel(base, ref):.5f}\t{rel(qk, ref):.5f}\t{rel(qkv, ref):.5f}\t{rel(qk, base):.5f}", flush=True)
