"""Bit-identity of lingbot/layers/sage_preq.py against stock sageattn on one GPU (run on the pod, not in CI).

Drives a KV cache through chunk writes, same-chunk rewrites (denoise steps) and eviction shifts, and after every
write compares `sage_preq.attend` with `sageattn(q, k_win, v_win, "NHD")`.
Usage: LINGBOT_ATTN=sage python experiments/bench_attn/sage_preq_check.py
"""
import torch
from sageattention import sageattn

from lingbot.layers import kv_cache as kvc
from lingbot.layers import sage_preq

torch.manual_seed(0)
dev, H, D = "cuda", 10, 128
T, n_tok, sink, local = 1000, 300, 200, 10   # window 1000 tokens, 300 per chunk, 200 in the sink; odd sizes on purpose
kc = kvc.allocate(1, [1, T, H, D], torch.bfloat16, dev)[0]
bad = 0


def rnd(n):
    x = torch.randn(1, n, H, D, device=dev, dtype=torch.bfloat16)
    return x * (1 + 20 * (torch.rand(1, 1, H, D, device=dev) > 0.98))      # outlier channels


for step, (chunk, rewrite) in enumerate([(c, r) for c in range(6) for r in range(3)]):
    n = n_tok - (7 if chunk == 2 else 0)
    k, v, q = rnd(n), rnd(n), rnd(n)
    end, cur_end = kvc.write(kc, k, v, chunk * n_tok, sink, local)
    start = end - n
    out = sage_preq.attend(q, kc, start, end)
    ref = sageattn(q, kc["k"][:, :end], kc["v"][:, :end], tensor_layout="NHD", is_causal=False)
    ok = torch.equal(out, ref)
    kvc.commit(kc, cur_end, end)
    bad += not ok
    print(f"chunk {chunk} rewrite {rewrite} end {end}: attention {'identical' if ok else 'DIFFERENT'}")
print("ALL BIT-IDENTICAL" if not bad else f"{bad} MISMATCHES")
