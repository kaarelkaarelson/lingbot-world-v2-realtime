"""NCCL collectives between the two GPUs (torchrun --nproc_per_node=2). TSV: variant, op, size_MB, us, busbw GB/s.
busbw follows nccl-tests: all_reduce x2(n-1)/n, all_gather/reduce_scatter/all_to_all x(n-1)/n of the full buffer."""
import os
import torch
import torch.distributed as dist

backend = os.environ.get("BENCH_BACKEND", "nccl")
dist.init_process_group(backend)
r = dist.get_rank()
torch.cuda.set_device(r)
variant = os.environ.get("VARIANT", "default")
SIZES = [0.004, 0.064, 1, 4.6, 9.3, 18.5, 64, 256]


def bench(fn, nbytes):
    reps = 100 if nbytes < 2**20 else 30 if nbytes < 2**27 else 8
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    dist.barrier()
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps / 1000


for mb in SIZES:
    n = max(4, int(mb * 2**20 / 2)) // 4 * 4
    x = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    y = torch.empty_like(x)
    half_in, half_out = x[: n // 2].clone(), torch.empty(n // 2, dtype=torch.bfloat16, device="cuda")
    nb = n * 2
    peer = 1 - r

    def sendrecv():  # both directions at once, n bytes each way; batched, or the two sends deadlock
        for q in dist.batch_isend_irecv([dist.P2POp(dist.isend, x, peer), dist.P2POp(dist.irecv, y, peer)]):
            q.wait()
    ops = [("all_reduce", lambda: dist.all_reduce(x), 1.0),
           ("all_gather", lambda: dist.all_gather_into_tensor(y, half_in), 0.5),
           ("reduce_scatter", lambda: dist.reduce_scatter_tensor(half_out, x), 0.5),
           ("broadcast", lambda: dist.broadcast(x, 0), 1.0),
           ("sendrecv_bidir", sendrecv, 1.0)]
    if backend == "nccl":
        ops.insert(1, ("all_to_all", lambda: dist.all_to_all_single(y, x), 0.5))
    for name, fn, f in ops:
        try:
            dt = bench(fn, nb)
            if r == 0:
                print(f"{variant}\t{name}\t{mb}\t{dt * 1e6:.1f}\t{nb * f / dt / 1e9:.1f}", flush=True)
        except Exception as ex:  # noqa: BLE001
            if r == 0:
                print(f"{variant}\t{name}\t{mb}\tFAILED\t{type(ex).__name__}", flush=True)
dist.destroy_process_group()
