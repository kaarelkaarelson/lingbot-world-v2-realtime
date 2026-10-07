"""Candidate #2: send q, k, v across the two cards in 8 bits instead of bf16 (SageAttention quantizes them
anyway, so quantizing before the exchange instead of after costs no extra accuracy in principle).
One process, both GPUs, plain cross-device copy_ (copy engines), both directions at once.
Per layer of 2-way Ulysses: 6 exchanges of a 9.3 MB bf16 activation (3,016 tokens x 1,536)."""
import time

import torch

dev = [torch.device("cuda:0"), torch.device("cuda:1")]
TOK, D, H = 3016, 1536, 12
BLK = 128  # tokens per quantization block, per head (SageAttention-style per-block scales)


def sync():
    for d in dev:
        torch.cuda.synchronize(d)


streams = [torch.cuda.Stream(device=d) for d in dev]


def exchange(bufs_out, bufs_in):
    """bufs_out[g]: what card g sends; bufs_in[g]: where card g receives. Both directions concurrently."""
    for g in (0, 1):
        streams[g].wait_stream(torch.cuda.current_stream(dev[g]))
    with torch.cuda.stream(streams[0]):
        bufs_in[1].copy_(bufs_out[0], non_blocking=True)
    with torch.cuda.stream(streams[1]):
        bufs_in[0].copy_(bufs_out[1], non_blocking=True)
    for g in (0, 1):
        torch.cuda.current_stream(dev[g]).wait_stream(streams[g])


def timed(fn, reps=50):
    for _ in range(5):
        fn()
    sync()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return (time.perf_counter() - t) / reps * 1e6


def bufs(shape, dtype):
    out = [torch.randn(shape, device=d).to(dtype) if dtype.is_floating_point else
           torch.randint(-127, 127, shape, dtype=dtype, device=d) for d in dev]
    return out, [torch.empty(shape, dtype=dtype, device=d) for d in dev]


def quant_int8(x):
    """x [TOK, H, 128] bf16 -> int8 values + fp32 scale per (128-token block, head)."""
    t = x.shape[0]
    pad = (-t) % BLK
    xb = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad)).view(-1, BLK, H, 128)
    s = xb.abs().amax(dim=(1, 3), keepdim=True).float().clamp_min(1e-8) / 127
    q = (xb.float() / s).round().clamp(-127, 127).to(torch.int8)
    return q.view(-1, H, 128)[:t], s.flatten()


def dequant_int8(q, s):
    t = q.shape[0]
    pad = (-t) % BLK
    qb = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad)).view(-1, BLK, H, 128)
    return (qb.float() * s.view(-1, 1, H, 1)).view(-1, H, 128)[:t]


half_shape = (TOK // 2, H, 128)  # each direction moves half of the local shard (2-way all-to-all)
res = {}
b16 = bufs(half_shape, torch.bfloat16)
res["bf16 activation (today)"] = timed(lambda: exchange(*b16))
i8 = bufs(half_shape, torch.int8)
res["int8 values"] = timed(lambda: exchange(*i8))
f8 = bufs(half_shape, torch.float8_e4m3fn)
res["fp8 values"] = timed(lambda: exchange(*f8))
nscale = -(-TOK // 2 // BLK) * H
sc = bufs((nscale,), torch.float32)
res["scales only"] = timed(lambda: exchange(*sc))
both = (i8[0] + sc[0], i8[1] + sc[1])
res["int8 values + scales"] = timed(lambda: (exchange(i8[0], i8[1]), exchange(sc[0], sc[1])))
x = torch.randn(TOK, H, 128, device=dev[0]).to(torch.bfloat16)
res["quantize int8 (torch eager, 1 tensor)"] = timed(lambda: quant_int8(x))
qc = torch.compile(quant_int8)
res["quantize int8 (torch.compile, 1 tensor)"] = timed(lambda: qc(x))
res["cast to fp8 (1 tensor)"] = timed(lambda: x.to(torch.float8_e4m3fn))

print("what\tus")
for k, v in res.items():
    print(f"{k}\t{v:.1f}")

# per layer: 6 exchanges today; candidate: q, k, v in 8 bits (+ scales), the other 3 stay bf16
base = 6 * res["bf16 activation (today)"]
cand = 3 * res["int8 values + scales"] + 3 * res["bf16 activation (today)"]
all8 = 6 * res["int8 values + scales"]
print(f"per layer today (6 x bf16)\t{base:.0f}")
print(f"per layer q,k,v in 8 bits (3 x int8+scales, 3 x bf16)\t{cand:.0f}")
print(f"per layer all 6 in 8 bits (upper bound)\t{all8:.0f}")
print(f"per chunk (150 layer passes): today {base * 150 / 1e3:.0f} ms, qkv-8bit {cand * 150 / 1e3:.0f} ms, all-8bit {all8 * 150 / 1e3:.0f} ms")

# data and accuracy checks
exchange(*i8)
sync()
assert torch.equal(i8[1][1].cpu(), i8[0][0].cpu()) and torch.equal(i8[1][0].cpu(), i8[0][1].cpu()), "wrong data"
q, s = quant_int8(x)
err = (dequant_int8(q, s) - x.float()).abs().mean() / x.float().abs().mean()
print(f"data check: ok; int8 per-block relative error {err:.4f} (the same rounding Sage applies after the exchange)")

# Each message has a fixed cost (scales-only above). Packing q, k, v (and their scales) into one message per
# direction cuts the count; per layer the minimum is 4 (self q|k|v, self out, cross q, cross out).
print("\npacked\tus")
pk16 = bufs((3,) + half_shape, torch.bfloat16)
t_qkv16 = timed(lambda: exchange(*pk16))
n8 = 3 * half_shape[0] * H * 128 + 3 * nscale * 4  # int8 values + fp32 scales, as raw bytes
pk8 = bufs((n8,), torch.int8)  # raw bytes
t_qkv8 = timed(lambda: exchange(*pk8))
print(f"q|k|v packed, bf16 (one message)\t{t_qkv16:.1f}")
print(f"q|k|v packed, int8 + scales (one message)\t{t_qkv8:.1f}")
one = res["bf16 activation (today)"]
p16 = t_qkv16 + 3 * one
p8 = t_qkv8 + 3 * one
print(f"per layer packed bf16 (1 x qkv + 3 x bf16)\t{p16:.0f}")
print(f"per layer packed 8-bit qkv (1 x qkv8 + 3 x bf16)\t{p8:.0f}")
print(f"per chunk: today {base * 150 / 1e3:.0f} ms, packed bf16 {p16 * 150 / 1e3:.0f} ms, packed 8-bit {p8 * 150 / 1e3:.0f} ms")
