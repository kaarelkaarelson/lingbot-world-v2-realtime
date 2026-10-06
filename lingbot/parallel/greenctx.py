"""CUDA green contexts (SM partitions) as PyTorch streams, for torch < 2.11 (no torch.cuda.green_contexts).

Work launched on a returned stream may only use that partition's SMs; memory is shared as usual.
SM groups come in multiples of 8 on the RTX 5090 (170 SMs).
"""
import torch
from cuda.bindings import driver as cu


def _ok(r):
    err = r[0] if isinstance(r, tuple) else r
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {err}")
    return r[1:] if isinstance(r, tuple) and len(r) > 2 else (r[1] if isinstance(r, tuple) and len(r) == 2 else None)


def split_streams(device_index, first_sms, priorities=(-5, 0)):
    """Two streams on one GPU: the first limited to `first_sms` SMs (rounded up to 8), the second to the rest."""
    torch.cuda.init()
    _ok(cu.cuInit(0))
    dev = _ok(cu.cuDeviceGet(device_index))
    res = _ok(cu.cuDeviceGetDevResource(dev, cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
    groups, _, rest = _ok(cu.cuDevSmResourceSplitByCount(1, res, 0, first_sms))
    streams, sms = [], []
    for r, prio in ((groups[0], priorities[0]), (rest, priorities[1])):
        desc = _ok(cu.cuDevResourceGenerateDesc([r], 1))
        gctx = _ok(cu.cuGreenCtxCreate(desc, dev, cu.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM))
        s = _ok(cu.cuGreenCtxStreamCreate(gctx, cu.CUstream_flags.CU_STREAM_NON_BLOCKING, prio))
        streams.append(torch.cuda.ExternalStream(int(s), device=torch.device(f"cuda:{device_index}")))
        sms.append(r.sm.smCount)
    return streams, sms


if __name__ == "__main__":
    # does a PyTorch matmul on a partition stream really run on fewer SMs (time scales ~ 170 / n), and stay correct?
    import time
    d = torch.device("cuda:1")
    a = torch.randn(8192, 8192, device=d, dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device=d, dtype=torch.bfloat16)
    ref = a @ b

    def bench(stream):
        with torch.cuda.stream(stream):
            for _ in range(3):
                c = a @ b
            torch.cuda.synchronize(d)
            t = time.perf_counter()
            for _ in range(10):
                c = a @ b
            torch.cuda.synchronize(d)
        return (time.perf_counter() - t) / 10 * 1e3, c

    full, _ = bench(torch.cuda.Stream(device=d))
    print(f"full GPU (170 SMs): {full:.2f} ms per 8192^3 bf16 matmul")
    for n in (40, 64, 128):
        (s0, s1), (n0, n1) = split_streams(1, n)
        t0, c0 = bench(s0)
        t1, c1 = bench(s1)
        print(f"partition {n0} SMs: {t0:.2f} ms ({full / t0 * 170 / n0 * 100:.0f}% of linear scaling, max diff {(c0 - ref).abs().max().item()}); "
              f"rest {n1} SMs: {t1:.2f} ms")
