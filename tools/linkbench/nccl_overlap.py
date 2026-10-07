"""Does NCCL (SHM transport, runs as GPU kernels) slow a concurrent matmul? torchrun --nproc_per_node=2."""
import torch, torch.distributed as dist
dist.init_process_group("nccl"); r = dist.get_rank(); torch.cuda.set_device(r)
a = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda"); flops = 2 * 8192**3
x = torch.randn(int(18.5 * 2**20 / 2) // 4 * 4, dtype=torch.bfloat16, device="cuda"); y = torch.empty_like(x)
ms, cs = torch.cuda.Stream(), torch.cuda.Stream()
def mm_time(with_comm):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    dist.barrier(); torch.cuda.synchronize()
    if with_comm:
        with torch.cuda.stream(cs):
            for _ in range(120): dist.all_to_all_single(y, x)
    with torch.cuda.stream(ms):
        s.record(ms)
        for _ in range(10): a @ a
        e.record(ms)
    torch.cuda.synchronize(); return s.elapsed_time(e) / 10 / 1000
for _ in range(2): mm_time(False); mm_time(True)
alone, during = mm_time(False), mm_time(True)
if r == 0:
    print(f"matmul alone {flops/alone/1e12:.0f} TFLOP/s, during NCCL all_to_all {flops/during/1e12:.0f} TFLOP/s ({during/alone:.2f}x time)")
dist.destroy_process_group()
