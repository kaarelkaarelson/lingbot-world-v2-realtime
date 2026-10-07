"""2-GPU all_to_all through pinned host memory, both directions pipelined in pieces, copy engines only."""
import time, torch
dev = [torch.device("cuda:0"), torch.device("cuda:1")]
def sync():
    for d in dev: torch.cuda.synchronize(d)
print("test\tsize_MB\tpiece_MB\tus\tbusbw GB/s")
for mb in (4.6, 9.3, 18.5, 64):
    n = int(mb * 2**20 / 2) // 4 * 4; h = n // 2
    x = [torch.randn(n, dtype=torch.bfloat16, device=d) for d in dev]
    y = [torch.empty(n, dtype=torch.bfloat16, device=d) for d in dev]
    host = [torch.empty(h, dtype=torch.bfloat16, pin_memory=True) for _ in range(2)]  # one per direction
    d2h = [torch.cuda.Stream(device=d) for d in dev]; h2d = [torch.cuda.Stream(device=d) for d in dev]
    for piece_mb in (1, 2, 4):
        p = max(1, int(piece_mb * 2**20 / 2))
        pieces = [(a, min(a + p, h)) for a in range(0, h, p)]
        ev = [[torch.cuda.Event() for _ in pieces] for _ in range(2)]
        def step():
            # direction 0: gpu0's second half -> gpu1's first half; direction 1: gpu1's first half -> gpu0's second half
            for k, (a, b) in enumerate(pieces):
                with torch.cuda.stream(d2h[0]): host[0][a:b].copy_(x[0][h + a:h + b], non_blocking=True); ev[0][k].record(d2h[0])
                with torch.cuda.stream(d2h[1]): host[1][a:b].copy_(x[1][a:b], non_blocking=True); ev[1][k].record(d2h[1])
                with torch.cuda.stream(h2d[1]): h2d[1].wait_event(ev[0][k]); y[1][a:b].copy_(host[0][a:b], non_blocking=True)
                with torch.cuda.stream(h2d[0]): h2d[0].wait_event(ev[1][k]); y[0][h + a:h + b].copy_(host[1][a:b], non_blocking=True)
        for _ in range(5): step()
        sync(); t = time.perf_counter(); reps = 40
        for _ in range(reps): step()
        sync(); dt = (time.perf_counter() - t) / reps
        print(f"staged a2a\t{mb}\t{piece_mb}\t{dt*1e6:.1f}\t{n*2*0.5/dt/1e9:.1f}", flush=True)
# correctness of the exchange at the last size
assert torch.equal(y[1][:h], x[0][h:].to(dev[1])) and torch.equal(y[0][h:], x[1][:h].to(dev[0])), "wrong data"
print("data check: ok")
