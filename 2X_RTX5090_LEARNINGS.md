# 2× RTX 5090: how the two cards talk, measured

Two RTX 5090s in one RunPod pod (`6i6dru89fazcbo`, Community Cloud, $1.98/hr, 2026-10-05) talk only
through host memory: GeForce cards have no direct GPU-to-GPU path. Every way of moving data between
them was benchmarked at the message sizes this model sends. The fastest is plain copy-engine copies
(about 26 GB/s for the exchange sequence parallelism needs, 45 GB/s one way); NCCL's all-to-all is
2.6-2.7× slower than that at exactly this model's sizes. With the measured link, splitting the DiT across
the two cards gains 1.3× without overlapping communication and compute; splitting DiT and decoder
across the cards gains 1.55×.

Scripts: `tools/linkbench/` (`bash tools/linkbench/run.sh`). Raw data: `bench/2x_rtx5090_link/`.

## Glossary

- **PCIe**: the bus that connects each GPU to the motherboard. Here gen 5, 16 lanes, about 57 GB/s
  each way per card.
- **Peer-to-peer (P2P)**: one GPU reading or writing another GPU's memory directly over PCIe,
  without going through host memory. NVIDIA turns it off on GeForce cards; workstation and data-center
  cards have it.
- **NCCL**: NVIDIA's library for moving data between GPUs (pronounced "nickel"). PyTorch's
  `torch.distributed` uses it for collectives such as all-reduce and all-to-all. It picks a transport
  per GPU pair: NVLink, P2P over PCIe, or **SHM** (shared host memory) when neither is available, as
  here. NCCL runs its transfers as small GPU kernels.
- **Copy engines**: dedicated DMA units on the GPU that move memory without using the compute cores.
  A plain `tensor.copy_()` between devices uses them; the driver stages the data through host memory
  when there is no P2P.
- **all-to-all**: each card sends a different part of its data to each other card. Sequence
  parallelism (Ulysses) needs four per layer for self-attention and two for cross-attention.
- **busbw**: bandwidth normalised the way nccl-tests does it, so different collectives compare.

## What the hardware is

| Check | Result |
|---|---|
| Both cards in one machine | yes (`nvidia-smi -L` in one pod) |
| Topology | `NODE`: same NUMA node (1), different PCIe host bridges (`c8:00.0`, `d8:00.0`), no switch, no NVLink |
| PCIe link | gen 5 ×16 on both (gen 1 when idle, gen 5 under load) |
| Health | 237 and 238 TFLOP/s bf16 matmul under load, no throttling |
| Peer-to-peer | **no**: `torch.cuda.can_device_access_peer` is False both ways |
| NCCL | 2.27.3 (PyTorch 2.8.0, CUDA 12.8); transport `via SHM/direct/direct` |
| Nsight Compute | unavailable (unprivileged container) |
| NUMA memory binding | not permitted in the container; CPU binding works and changes nothing |

## Every way to move data, at this model's sizes

Sequence parallelism over two cards exchanges a 9.3 MB activation (3,016 tokens × 1,536 × bf16) six
times per layer; tensor parallelism all-reduces an 18.5 MB activation three times per layer.

| Method | 9.3 MB exchange | 18.5 MB | 64 MB | Small message |
|---|---|---|---|---|
| Card → host, one way (pinned) | 57 GB/s | 58 | 58 | 6 µs |
| Both cards → host at once | 111 GB/s total | 112 | 113 | |
| GPU → GPU `copy_`, one way (driver-staged) | 42 GB/s | 44 | 45 | 25 µs |
| GPU → GPU `copy_`, both ways at once | 59 GB/s total | 63 | 65 | |
| Manual pinned staging, 4 MB pieces, one way | 49 GB/s | 55 | 56 | |
| **all-to-all by `copy_` on two streams** | **177-185 µs, 26-27.5 GB/s** | 330 µs, 29 | 1,045 µs, 32 | 99 µs |
| all-to-all by manual pinned staging, both ways | 255 µs, 19 | 384 µs, 25 | 881 µs, 38 | |
| NCCL send/recv pair (both ways, half each) | 200 µs, 24 | 386 µs, 25 | not measured | 41 µs |
| **NCCL `all_to_all`** | **484 µs, 10 GB/s** | 386 µs, 25 | 1,305 µs, 26 | 22 µs |
| NCCL `all_reduce` | 317 µs, 31 | 618 µs, 31 | 2,081 µs, 32 | 22 µs |
| NCCL `broadcast` | 196 µs, 50 | 386 µs, 50 | 1,307 µs, 51 | 20 µs |
| NCCL with SHM disabled (sockets) | 2,656 µs, 1.8 | 5,238 µs | 14,025 µs | 34 µs |
| gloo | | | | 113-268 µs |

NCCL settings tried, none faster than the default: `NCCL_ALGO=Ring|Tree`, `NCCL_PROTO=LL|LL128|Simple`,
`NCCL_SHM_USE_CUDA_MEMCPY=1` (slower: 19.7 vs 25.0 GB/s all-to-all at 18.5 MB), 16 channels,
`NCCL_BUFFSIZE=16 MB`, binding to either NUMA node, and `NCCL_P2P_DISABLE=1` (no change: P2P was never
in use). `NCCL_PROTO=LL` and `LL128` are 1.7-2.3× slower on all-reduce; `Tree` fails on all-gather,
reduce-scatter and broadcast. `NCCL_SHM_DISABLE=1` falls back to sockets and is 10× slower.

## Learnings

1. **No peer-to-peer on GeForce, and nothing in the pod changes that.** Every transfer goes card →
   host memory → card. NVIDIA limits P2P to workstation and data-center cards; the community driver
   patch that enables it has to be installed on the host, which RunPod does not allow.
2. **The ceiling is host memory staging, not PCIe.** Each card reaches 57 GB/s to the host and both
   together 112 GB/s, but GPU-to-GPU tops out at 45 GB/s one way and 65 GB/s both ways: every byte
   crosses PCIe twice and passes through host memory.
3. **NCCL's all-to-all has a hole at 4-10 MB**: 10 GB/s at 4.6 and 9.3 MB, 25 GB/s from 18.5 MB up,
   in every NCCL configuration. That is exactly the size sequence parallelism sends. The same data
   moved by copy-engine copies on two streams takes 177-185 µs instead of 484 µs (2.6-2.7× faster), and NCCL
   send/recv pairs take about 200 µs.
4. **Communication does not slow compute.** A bf16 matmul runs at 239-241 TFLOP/s alone, during
   copy-engine traffic in both directions, and during NCCL all-to-alls. Overlapping communication with
   compute is free on these cards; how much overlap the model's data dependencies allow is the open
   question.
5. **NCCL is still best for tiny messages** (about 20 µs against 25-100 µs for copies), and for
   broadcast (50 GB/s).
6. **Placement knobs do not matter here.** NUMA binding, channel count and buffer size change nothing
   measurable; the host-memory path is the bottleneck.
7. **Software was not the problem.** Forum reports of 5090 multi-GPU failures were old NCCL versions;
   PyTorch 2.8 ships NCCL 2.27.3, which worked without changes.

## What it means for splitting LingBot-World 2.0 across the two cards

One card today: DiT 0.64 s + decoder 0.34 s per 16-frame chunk, 16.3 FPS. Each DiT chunk is five
passes through 30 layers. Predictions with the measured times above (compute halves on two cards,
except tensor parallelism's elementwise work, which both cards repeat):

| Layout | Link time per chunk | DiT, no overlap | DiT, full overlap | FPS, no overlap | FPS, full overlap |
|---|---|---|---|---|---|
| One card | — | 0.64 s | | **16.3** | |
| Sequence parallel, NCCL all-to-all | 436 ms | 0.756 s (0.85×) | 0.436 s | 14.6 | 20.6 |
| Sequence parallel, copy-engine exchange | 159 ms | 0.479 s (1.34×) | 0.320 s | **19.5** | 24.2 |
| Tensor parallel, NCCL all-reduce | 278 ms | 0.645 s (0.99×) | 0.367 s | 16.3 | 22.6 |
| DiT on card 0, decoder on card 1 | 0.77 MB, ~40 µs | 0.64 s | | **25.0** | |

- **Splitting DiT and decoder is the clear first step**: 1.55× with almost no communication. It
  raises frame rate, not the delay to each frame.
- **Sequence parallelism pays only with the copy-engine exchange**: with NCCL's all-to-all it is
  slower than one card. Combined with the decoder on its own card, the DiT is the bottleneck, so
  sequence parallelism plus the split is the ceiling to aim for.
- **Tensor parallelism gains nothing without overlap** on this link.

These are predictions from link benchmarks and single-card kernel times; the model runs are next.

## Caveats

- One pod, one session. Another 2× 5090 host can wire its cards differently (`PIX`, `PHB` or `SYS`
  topology); rerun `tools/linkbench/run.sh` on each new pod.
- Copy-engine all-to-all times varied between runs (104 vs 143 µs at 4.6 MB, 177 vs 185 µs at
  9.3 MB); the predictions use 177 µs. The one-card FPS here (16.3) is 16 ÷ (0.64 + 0.34); the
  README's measured 16.1 includes run-to-run spread.
- The single-process measurements drive both cards from one process; a two-process setup (one per
  card, as `torchrun` launches) would need CUDA IPC or NCCL to reach the other card's memory.
