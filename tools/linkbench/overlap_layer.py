"""How much of a sequence-parallel layer's exchange can hide behind compute that depends on it?
One process, both GPUs. Per layer: 6 exchanges of a 9.3 MB activation (half each way, copy engines),
and ~2.1 ms of bf16 matmul per card that consumes the exchanged data. Variants:
  compute only / exchange then compute (serial) / exchange in k pieces, each piece's compute starting
  when that piece lands (pipelined; the next piece transfers while the current one computes)."""
import time

import torch

dev = [torch.device("cuda:0"), torch.device("cuda:1")]
N_EX, MB, LAYERS = 6, 9.3, 30
n = int(MB * 2**20 / 2) // 64 * 64
h = n // 2
x = [torch.randn(n, dtype=torch.bfloat16, device=d) for d in dev]
y = [torch.empty(n, dtype=torch.bfloat16, device=d) for d in dev]
# compute sized to ~2.1 ms per layer per card at ~240 TFLOP/s, split evenly over the exchanged pieces
M = 1024
w = [torch.randn(M, M, dtype=torch.bfloat16, device=d) for d in dev]
a = [torch.randn(M, M, dtype=torch.bfloat16, device=d) for d in dev]
flop_per_mm = 2 * M**3
per_exchange = max(8, round(2.1e-3 * 240e12 / flop_per_mm / N_EX) // 8 * 8)  # divisible by 2, 4 and 8 pieces
mm_per_layer = per_exchange * N_EX
cs = [torch.cuda.Stream(device=d) for d in dev]  # copy streams
ks = [torch.cuda.Stream(device=d) for d in dev]  # compute streams


def sync():
    for d in dev:
        torch.cuda.synchronize(d)


def exchange_piece(lo, hi, ev, after_compute=False):
    if after_compute:  # the data to send is the previous computation's output
        for g in (0, 1):
            cs[g].wait_stream(ks[g])
    with torch.cuda.stream(cs[0]):
        y[1][lo:hi].copy_(x[0][h + lo:h + hi], non_blocking=True)
        ev[1].record(cs[0])
    with torch.cuda.stream(cs[1]):
        y[0][h + lo:h + hi].copy_(x[1][lo:hi], non_blocking=True)
        ev[0].record(cs[1])


def compute(g, count):
    with torch.cuda.stream(ks[g]):
        for _ in range(count):
            torch.mm(a[g], w[g])


def layer(mode, pieces):
    for _ in range(N_EX):
        if mode == "compute_only":
            for g in (0, 1):
                compute(g, per_exchange)
            continue
        k = 1 if mode == "serial" else pieces
        step = -(-h // k)
        per = per_exchange // k
        for i in range(k):
            ev = [torch.cuda.Event(), torch.cuda.Event()]
            exchange_piece(i * step, min(h, (i + 1) * step), ev, after_compute=(i == 0))
            for g in (0, 1):
                ks[g].wait_event(ev[g])  # this piece's compute needs this piece's data
                compute(g, per)


def run(mode, pieces=1, reps=3):
    for _ in range(2):
        for _ in range(LAYERS):
            layer(mode, pieces)
    sync()
    t = time.perf_counter()
    for _ in range(reps):
        for _ in range(LAYERS):
            layer(mode, pieces)
    sync()
    return (time.perf_counter() - t) / reps / LAYERS


base = run("compute_only")
ser = run("serial")
print(f"per layer: {mm_per_layer} matmuls of {M}^3 per card (~{mm_per_layer * flop_per_mm / 240e12 * 1e3:.1f} ms), "
      f"{N_EX} exchanges of {MB} MB")
print(f"compute only        {base * 1e3:.2f} ms")
print(f"exchange then compute (serial) {ser * 1e3:.2f} ms  -> exchange cost {(ser - base) * 1e3:.2f} ms")
for k in (2, 4, 8):
    p = run("pipelined", k)
    hidden = 1 - (p - base) / (ser - base)
    print(f"pipelined, {k} pieces  {p * 1e3:.2f} ms  -> {hidden * 100:.0f}% of the exchange hidden")
