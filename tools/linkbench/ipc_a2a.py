"""Two processes, one per GPU (as torchrun deploys): can a process copy straight out of the other card's
memory via CUDA IPC handles when there is no P2P, and how fast is the 2-GPU all_to_all exchange that way?
Each process receives the peer's buffer as an IPC tensor (on the peer's device) and copies its half in."""
import time

import torch
import torch.multiprocessing as mp

SIZES = [4.6, 9.3, 18.5, 64]


def worker(rank, q_out, q_in, barrier, results):
    torch.cuda.set_device(rank)
    peer = 1 - rank
    out = {}
    for mb in SIZES:
        n = int(mb * 2**20 / 2) // 4 * 4
        h = n // 2
        x = torch.full((n,), float(rank + 1), dtype=torch.bfloat16, device=f"cuda:{rank}")
        y = torch.empty(n, dtype=torch.bfloat16, device=f"cuda:{rank}")
        q_out.put(x)                      # shared with the peer through a CUDA IPC handle
        xp = q_in.get()                   # the peer's buffer, still on the peer's device
        src = xp[:h] if rank == 1 else xp[h:]
        dst = y[h:] if rank == 1 else y[:h]
        try:
            for _ in range(5):
                dst.copy_(src)
            torch.cuda.synchronize(rank)
            ok = bool((dst == float(peer + 1)).all())
            barrier.wait()
            t = time.perf_counter()
            reps = 40
            for _ in range(reps):
                dst.copy_(src)            # both processes pull their half at the same time
            torch.cuda.synchronize(rank)
            dt = (time.perf_counter() - t) / reps
            barrier.wait()
            out[mb] = (dt, ok, None)
        except Exception as e:  # noqa: BLE001
            barrier.wait()
            barrier.wait()
            out[mb] = (None, False, f"{type(e).__name__}: {str(e)[:120]}")
        del xp, src
        barrier.wait()
    results[rank] = out


if __name__ == "__main__":
    mp.set_start_method("spawn")
    with mp.Manager() as m:
        q01, q10, results = mp.Queue(), mp.Queue(), m.dict()
        barrier = mp.Barrier(2)
        ps = [mp.Process(target=worker, args=(0, q01, q10, barrier, results)),
              mp.Process(target=worker, args=(1, q10, q01, barrier, results))]
        for p in ps:
            p.start()
        for p in ps:
            p.join()
        print("test\tsize_MB\tus\tbusbw GB/s\tdata ok")
        for mb in SIZES:
            r0, r1 = results[0][mb], results[1][mb]
            if r0[2] or r1[2]:
                print(f"ipc a2a (2 processes)\t{mb}\tFAILED\t{r0[2] or r1[2]}")
                continue
            dt = max(r0[0], r1[0])
            n = int(mb * 2**20 / 2) // 4 * 4
            print(f"ipc a2a (2 processes)\t{mb}\t{dt * 1e6:.1f}\t{n * 2 * 0.5 / dt / 1e9:.1f}\t{r0[1] and r1[1]}")
