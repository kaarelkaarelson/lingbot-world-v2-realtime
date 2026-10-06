"""Can CUDA graphs run inside a green-context SM partition? (prerequisite for graphing the split DiT's card-1 work)

1. capture a bf16 matmul on a 40-SM green-context stream; replay it there and on an ordinary stream; time vs the
   full GPU (inside the partition it should take ~170/40 = 4.25x as long)
2. capture SageAttention (patched: current stream), FP8 torch._scaled_mm and FlashAttention on the partition
   stream; replay must equal eager bit for bit

  python experiments/overlap_layer/gc_graph_test.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from greenctx import split_streams  # noqa: E402

torch.set_grad_enabled(False)
d = torch.device("cuda:1")
torch.cuda.set_device(d)
(g40, rest), sms = split_streams(1, 40)
full = torch.cuda.Stream(device=d)
print(f"partition: {sms[0]} SMs / rest {sms[1]}")


def timed(fn, stream, reps=10):
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
        torch.cuda.synchronize(d)
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize(d)
    return (time.perf_counter() - t) / reps * 1e3


a = torch.randn(8192, 8192, device=d, dtype=torch.bfloat16)
b = torch.randn(8192, 8192, device=d, dtype=torch.bfloat16)
c = torch.empty(8192, 8192, device=d, dtype=torch.bfloat16)
mm = lambda: torch.matmul(a, b, out=c)
t_full = timed(mm, full)
t_part = timed(mm, g40)
print(f"eager matmul: full GPU {t_full:.2f} ms, 40-SM partition {t_part:.2f} ms ({t_part / t_full:.2f}x, expect ~4.25x)")

ok = True
try:
    with torch.cuda.stream(g40):
        mm()
        torch.cuda.synchronize(d)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=g40, capture_error_mode="thread_local"):
            mm()
    print("capture on the green-context stream: OK")
    t_gp = timed(g.replay, g40)
    t_gf = timed(g.replay, full)
    print(f"graph replay: on partition stream {t_gp:.2f} ms ({t_gp / t_full:.2f}x full), "
          f"on an ordinary stream {t_gf:.2f} ms ({t_gf / t_full:.2f}x full)")
except Exception as e:  # noqa: BLE001
    ok = False
    print(f"capture on the green-context stream FAILED: {type(e).__name__}: {str(e)[:300]}")

if ok:
    from flash_attn import flash_attn_func
    from sageattention import sageattn
    q = torch.randn(1, 2, 6032, 128, device=d, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 27144, 128, device=d, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 27144, 128, device=d, dtype=torch.bfloat16)
    x8 = torch.randn(1005, 1536, device=d).to(torch.float8_e4m3fn)
    w8 = torch.randn(1536, 1536, device=d).to(torch.float8_e4m3fn)
    sa, sb = torch.ones(1005, 1, device=d), torch.ones(1, 1536, device=d)
    cq = torch.randn(1, 1005, 12, 128, device=d, dtype=torch.bfloat16)
    ck = torch.randn(1, 512, 12, 128, device=d, dtype=torch.bfloat16)
    out = {}

    def work():
        out["sage"] = sageattn(q, k, v, tensor_layout="HND")
        out["fp8"] = torch._scaled_mm(x8, w8.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        out["flash"] = flash_attn_func(cq, ck, ck)

    for name in ("eager",):
        with torch.cuda.stream(g40):
            work(); work()
            torch.cuda.synchronize(d)
            ref = {kk: vv.clone() for kk, vv in out.items()}
    try:
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.stream(g40):
            with torch.cuda.graph(g2, stream=g40, capture_error_mode="thread_local"):
                work()
            g2.replay()
            torch.cuda.synchronize(d)
        for kk in ref:
            print(f"graph vs eager, {kk}: max |diff| {(out[kk].float() - ref[kk].float()).abs().max().item():.3g}")
    except Exception as e:  # noqa: BLE001
        print(f"capturing Sage / FP8 GEMM / FlashAttention FAILED: {type(e).__name__}: {str(e)[:300]}")
print("GC GRAPH TEST DONE")
