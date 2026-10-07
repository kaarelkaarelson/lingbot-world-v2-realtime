"""One process driving both GPUs: host links, driver-staged GPU->GPU copies, manual pinned staging,
and whether transfers slow a concurrent matmul. Prints TSV rows: test, size_MB, us, GB/s."""
import time

import torch

SIZES = [0.004, 0.064, 1, 4.6, 9.3, 18.5, 64, 256]
dev = [torch.device("cuda:0"), torch.device("cuda:1")]
print(f"# torch {torch.__version__} | peer access 0->1 {torch.cuda.can_device_access_peer(0, 1)}, 1->0 {torch.cuda.can_device_access_peer(1, 0)}")


def sync():
    for d in dev:
        torch.cuda.synchronize(d)


def timeit(step, nbytes, reps=None):
    reps = reps or (200 if nbytes < 2**20 else 50 if nbytes < 2**27 else 10)
    for _ in range(5):
        step()
    sync()
    t = time.perf_counter()
    for _ in range(reps):
        step()
    sync()
    return (time.perf_counter() - t) / reps


def row(name, mb, dt, moved):
    print(f"{name}\t{mb}\t{dt * 1e6:.1f}\t{moved / dt / 1e9:.1f}", flush=True)


def numel(mb):
    return max(2, int(mb * 2**20 / 2)) // 2 * 2


print("test\tsize_MB\tus\tGB/s")
for mb in SIZES:
    n = numel(mb)
    nb = n * 2
    host = [torch.empty(n, dtype=torch.bfloat16, pin_memory=True) for _ in range(2)]
    g = [torch.randn(n, dtype=torch.bfloat16, device=d) for d in dev]
    g2 = [torch.empty(n, dtype=torch.bfloat16, device=d) for d in dev]
    s = [torch.cuda.Stream(device=d) for d in dev]
    s2 = [torch.cuda.Stream(device=d) for d in dev]

    # host links (pinned)
    row("H2D gpu0", mb, timeit(lambda: g[0].copy_(host[0], non_blocking=True), nb), nb)
    row("D2H gpu0", mb, timeit(lambda: host[0].copy_(g[0], non_blocking=True), nb), nb)
    row("H2D gpu1", mb, timeit(lambda: g[1].copy_(host[1], non_blocking=True), nb), nb)

    def both_h2d():
        with torch.cuda.stream(s[0]):
            g[0].copy_(host[0], non_blocking=True)
        with torch.cuda.stream(s[1]):
            g[1].copy_(host[1], non_blocking=True)
    row("H2D both GPUs at once (total)", mb, timeit(both_h2d, nb), 2 * nb)

    def d2h_h2d_same_gpu():
        with torch.cuda.stream(s[0]):
            host[0].copy_(g[0], non_blocking=True)
        with torch.cuda.stream(s2[0]):
            g2[0].copy_(host[1], non_blocking=True)
    row("D2H+H2D gpu0 at once (total, duplex)", mb, timeit(d2h_h2d_same_gpu, nb), 2 * nb)

    # GPU->GPU through the driver (no P2P on GeForce: staged through host memory)
    row("copy_ gpu0->gpu1", mb, timeit(lambda: g2[1].copy_(g[0]), nb), nb)

    def bidir():
        with torch.cuda.stream(s[0]):
            g2[1].copy_(g[0], non_blocking=True)
        with torch.cuda.stream(s[1]):
            g2[0].copy_(g[1], non_blocking=True)
    row("copy_ both directions at once (total)", mb, timeit(bidir, nb), 2 * nb)

    h = n // 2

    def a2a():  # what a 2-GPU all_to_all moves: each card sends half its buffer to the other
        with torch.cuda.stream(s[0]):
            g2[1][:h].copy_(g[0][h:], non_blocking=True)
        with torch.cuda.stream(s[1]):
            g2[0][h:].copy_(g[1][:h], non_blocking=True)
    row("all_to_all by copy_ (busbw, 2-GPU)", mb, timeit(a2a, nb), nb * 0.5)

    # manual staging: D2H on gpu0 then H2D on gpu1 through a pinned buffer, pipelined in 4 MB pieces
    piece = min(n, numel(4))
    pieces = [(i, min(i + piece, n)) for i in range(0, n, piece)]
    ev = [torch.cuda.Event() for _ in pieces]

    def staged():
        for k, (a, b) in enumerate(pieces):
            with torch.cuda.stream(s[0]):
                host[0][a:b].copy_(g[0][a:b], non_blocking=True)
                ev[k].record(s[0])
            with torch.cuda.stream(s[1]):
                s[1].wait_event(ev[k])
                g2[1][a:b].copy_(host[0][a:b], non_blocking=True)
    row("manual pinned staging gpu0->gpu1, 4 MB pieces", mb, timeit(staged, nb), nb)

# does traffic slow compute? bf16 8192^3 matmul on gpu0, alone and while both directions copy 64 MB continuously
a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=dev[0])
flops = 2 * 8192**3
dt = timeit(lambda: a @ a, 0, reps=30)
print(f"matmul gpu0 alone\t-\t{dt * 1e6:.0f}\t{flops / dt / 1e12:.0f} TFLOP/s")
n = numel(64)
x = [torch.randn(n, dtype=torch.bfloat16, device=d) for d in dev]
y = [torch.empty(n, dtype=torch.bfloat16, device=d) for d in dev]
cs = [torch.cuda.Stream(device=d) for d in dev]
ms = torch.cuda.Stream(device=dev[0])


def both():
    with torch.cuda.stream(cs[0]):
        for _ in range(24):
            y[1].copy_(x[0], non_blocking=True)
    with torch.cuda.stream(cs[1]):
        for _ in range(24):
            y[0].copy_(x[1], non_blocking=True)
    with torch.cuda.stream(ms):
        for _ in range(10):
            a @ a


for _ in range(2):
    both()
sync()
t = time.perf_counter()
ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
for _ in range(5):
    with torch.cuda.stream(cs[0]):
        for _ in range(24):
            y[1].copy_(x[0], non_blocking=True)
    with torch.cuda.stream(cs[1]):
        for _ in range(24):
            y[0].copy_(x[1], non_blocking=True)
    with torch.cuda.stream(ms):
        ev0.record(ms)
        for _ in range(10):
            a @ a
        ev1.record(ms)
    sync()
mm = ev0.elapsed_time(ev1) / 10 / 1000
print(f"matmul gpu0 during copies\t-\t{mm * 1e6:.0f}\t{flops / mm / 1e12:.0f} TFLOP/s")
