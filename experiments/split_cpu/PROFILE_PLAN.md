# Profiling plan: where the 25 FPS split is bound

Question: for the best split (10:2 + green contexts, decoder on card 1, `LINGBOT_SPLIT_CPPWRAP=1
LINGBOT_SPLIT_SKIPGUARD=50 LINGBOT_SPLIT_PIECES=2`), which resource sets the chunk time (the CPU thread,
card 0, card 1, or the transfers between them), and how far is each from its speed of light?

Pod: Vast 54497707 (EPYC 9454, 3.8 µs per eager op, 575 W, PCIe 5.0 x16 at 56 GB/s both cards, `NODE`).
Run everything on the torch 2.8 stack first, then repeat steps 1, 2 and 6 on torch 2.14 + cu130.

## 0. Ground truth, unprofiled

3 warm runs each of the split and of B1 through the worker. Record FPS and the p50 chunk interval. Every
profiled number is checked against these; a profiler that moves the interval by more than ~5 % is distorting.

## 1. One timeline: CPU thread, both cards, all copies (Nsight Systems)

- Start the warm worker under `nsys` (it has to be the parent, like py-spy). Capture only steady chunks of
  a warm run: `--capture-range=cudaProfilerApi`, with the pipeline calling `cudaProfilerStart/Stop` around
  chunks 5-8 when `LINGBOT_NSYS_CHUNKS=5:8` is set.
- NVTX ranges on the CPU side: chunk, forward, layer, and inside each layer `pre`, `send qkv`, `attention`,
  `send back`, `post`. Cost: a few µs per range.
- Gives: per-card and per-stream busy time, kernel durations by class, every D2H/H2D piece with bytes and
  duration, every CUDA API call with its timestamp, and the CPU thread's state (running vs blocked).

## 2. CPU breakdown (py-spy through the worker)

`LINGBOT_WORKER_PYSPY=1`, analyse `_warm_run` only, with `--idle` so blocked time is counted separately
from busy time. Gives CPU seconds per chunk by function: compiled-stage calls, exchange, Sage wrapper, KV
bookkeeping, pipeline, decoder launch.

## 3. Numbers to extract per steady chunk

| Resource | Measured | Speed of light | Gap |
|---|---|---|---|
| CPU launch thread | busy time per chunk (py-spy); launch lead = kernel start minus its API call (nsys) | 0 when hidden behind the GPU | |
| Card 0 DiT | sum of kernel time; busy % of the interval | FLOPs / peak per kernel class (FP8 GEMM, Sage INT8/FP8, elementwise bytes / 1.8 TB/s) | |
| Card 1 DiT (40 SMs) | same, plus slowdown while the decoder runs | same, scaled to 40/170 SMs | |
| Card 1 decoder (130 SMs) | kernel time per chunk | conv FLOPs / peak scaled to 130 SMs | |
| Transfers | bytes per layer; per-piece GB/s; message latency (send ready → receive done); exposed time (compute stream waiting on a transfer event) | bytes / 56 GB/s per direction | |

Launch lead decides the CPU question directly: a median lead of tens of µs means each kernel is issued just
before it runs (CPU-bound); a lead of milliseconds means the queue is full (GPU-bound).

## 4. Critical path per layer pass

For each of the 150 layer passes in a chunk, find which event finishes last before the next layer can start:
card 0 compute, card 1 compute, a transfer, or the CPU issuing the next launch. Sum the attribution over the
chunk. This is the answer to "where is the bound": the resource that is on the critical path most often, and
for how many ms. A script (`critical_path.py`) reads the nsys SQLite export.

## 5. Causal checks (one knob each, 3 warm runs)

Profiles show correlation; these show cause. The resource that bounds the chunk is the one whose knob moves FPS.

| Knob | Tests |
|---|---|
| decoder off (`--bench` instead of `--bench_e2e`) | card 1 contention from the decoder |
| card 1 SMs 32 / 40 / 48 | card 1 DiT vs decoder balance |
| head split 9:3 / 10:2 / 11:1 | card 0 vs card 1 balance |
| transfer pieces 1 / 2 / 4 | transfer latency vs overlap |
| `cpp_wrapper` off | CPU sensitivity on the fast CPU |

## 6. Deliverables

- A per-chunk budget table: CPU, card 0, card 1, transfers, each with measured time, speed of light and gap.
- The critical-path breakdown with the bound named, and the next lever it implies.
- The CPU + GPU timeline in the results page; a learnings entry; the raw files in `experiments/split_cpu/results/`.

## Instrumentation to build first (code only, before pod time)

1. `LINGBOT_NSYS_CHUNKS=a:b`: `cudaProfilerStart/Stop` around those chunks in the pipeline.
2. NVTX ranges in `split_dit.py` (`LINGBOT_NVTX=1`, off by default so normal runs pay nothing).
3. `LINGBOT_WORKER_NSYS=1`: start the worker under `nsys profile` with the capture range, like the py-spy mode.
4. `critical_path.py` and an extension of `cpu_gpu_bound.py` for the nsys SQLite export.

Estimated pod time: ~1 hour per stack.
