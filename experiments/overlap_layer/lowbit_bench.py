"""Does the RTX 5090 give more FLOP/s below 8 bits? Matmul throughput per number format, through the paths
PyTorch 2.8 can reach: BF16, FP8 (tensorwise and rowwise, what LINGBOT_FP8 uses), INT8, MXFP8, MXFP4, NVFP4.

Shapes: a large square (peak) and the DiT's three GEMMs for one chunk (6,032 tokens). Spec dense peaks
(RTX Blackwell whitepaper, FP32 accumulate): BF16 209.5, FP8 419, INT8 838 (repo's attention denominator),
FP4 838 TFLOP/s at 2.41 GHz.

  python experiments/overlap_layer/lowbit_bench.py
"""
import time

import torch

dev = torch.device("cuda:0")
torch.manual_seed(0)
SHAPES = [("square 8192", 8192, 8192, 8192), ("q|k|v proj", 6032, 1536, 4608), ("FFN up", 6032, 1536, 8960),
          ("FFN down", 6032, 8960, 1536)]


def timed(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(dev)
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize(dev)
    return (time.perf_counter() - t) / reps


def to_blocked(x):
    """Swizzle a [rows, cols] scale matrix into cuBLAS's 128x4 blocked layout (as torchao does)."""
    try:
        from torchao.prototype.mx_formats.utils import to_blocked as tb
        return tb(x)
    except Exception:
        rows, cols = x.shape
        n_rb, n_cb = -(-rows // 128), -(-cols // 4)
        pad = torch.zeros(n_rb * 128, n_cb * 4, dtype=x.dtype, device=x.device)
        pad[:rows, :cols] = x
        blocks = pad.view(n_rb, 128, n_cb, 4).permute(0, 2, 1, 3)
        return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def formats(M, K, N):
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    b = torch.randn(N, K, device=dev, dtype=torch.bfloat16)  # weight [N, K], used as b.t()
    out = {}
    out["BF16"] = lambda: a @ b.t()
    a8, b8 = a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn)
    one = torch.ones((), device=dev)
    out["FP8 tensorwise"] = lambda: torch._scaled_mm(a8, b8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    sa, sb = torch.ones(M, 1, device=dev), torch.ones(1, N, device=dev)
    out["FP8 rowwise (ours)"] = lambda: torch._scaled_mm(a8, b8.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    ai, bi = torch.randint(-64, 64, (M, K), device=dev, dtype=torch.int8), torch.randint(-64, 64, (N, K), device=dev, dtype=torch.int8)
    if M > 16:
        out["INT8 (_int_mm)"] = lambda: torch._int_mm(ai, bi.t())
    # MX formats: one e8m0 scale per 32 elements along K, cuBLAS blocked layout
    e8 = getattr(torch, "float8_e8m0fnu", None)
    if e8 is not None:
        sa32 = to_blocked(torch.full((M, K // 32), 1.0, device=dev).to(e8))
        sb32 = to_blocked(torch.full((N, K // 32), 1.0, device=dev).to(e8))
        out["MXFP8"] = lambda: torch._scaled_mm(a8, b8.t(), scale_a=sa32, scale_b=sb32, out_dtype=torch.bfloat16)
        f4 = getattr(torch, "float4_e2m1fn_x2", None)
        if f4 is not None:
            a4 = torch.randint(0, 255, (M, K // 2), device=dev, dtype=torch.uint8).view(f4)
            b4 = torch.randint(0, 255, (N, K // 2), device=dev, dtype=torch.uint8).view(f4)
            out["MXFP4"] = lambda: torch._scaled_mm(a4, b4.t(), scale_a=sa32, scale_b=sb32, out_dtype=torch.bfloat16)
            sa16 = to_blocked(torch.full((M, K // 16), 1.0, device=dev).to(torch.float8_e4m3fn))
            sb16 = to_blocked(torch.full((N, K // 16), 1.0, device=dev).to(torch.float8_e4m3fn))
            out["NVFP4"] = lambda: torch._scaled_mm(a4, b4.t(), scale_a=sa16, scale_b=sb16, out_dtype=torch.bfloat16)
    return out


print(f"torch {torch.__version__}, {torch.cuda.get_device_name(dev)}, sm_{''.join(map(str, torch.cuda.get_device_capability(dev)))}")
print("shape\tformat\tms\tTFLOP/s")
for name, M, K, N in SHAPES:
    for fmt, fn in formats(M, K, N).items():
        try:
            t = timed(fn)
            print(f"{name}\t{fmt}\t{t * 1e3:.3f}\t{2 * M * K * N / t / 1e12:.0f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{name}\t{fmt}\tunsupported: {type(e).__name__}: {str(e).splitlines()[0][:110]}", flush=True)
