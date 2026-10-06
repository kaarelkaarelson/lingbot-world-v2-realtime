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
8. **Each exchange is mostly fixed cost, so message count matters more than bytes.** A nearly empty
   message takes ~100-120 µs; a 9.3 MB bf16 exchange takes 160-185 µs. Sending q, k, v in 8 bits as
   separate messages is *slower* than bf16 (1,144 vs 968 µs per layer), because the scales add a
   message. Packing q|k|v and their scales into one 8-bit message per layer cuts it to 713 µs: 145 → 107
   ms per chunk (-26%). Packing in bf16 alone saves little (922 µs). `tools/linkbench/quant_a2a.py`,
   data in `bench/2x_rtx5090_link/quant_a2a.tsv`. Untested: numerics inside SageAttention (its K
   smoothing needs one mean shared by both halves, e.g. the previous step's).
9. **8-bit q/k/v that is dequantized on the receiver is not lossless.** The receiver turns the int8
   values back into bf16 and SageAttention rounds them again with its own, finer blocks, so the errors
   stack. On one real forward (chunk 7, window full), attention-output error against exact fp32
   attention, layers 0/10/20/29:

   | | Sage today | + q, k int8 | + q, k int8, v fp8 |
   |---|---|---|---|
   | error | 0.90-2.41 % | 1.02-2.74 % | 1.39-3.53 % |
   | vs today | | +10-14 % | +36-47 % |

   Rollouts agree: first-chunk PSNR against one-card `fast` is 27.0/31.1/27.8 dB for q, k and
   27.5/29.0/25.6 for q, k, v, against a run-to-run band of 29.9-31.9. `experiments/a2a8/`
   (`attn_probe.py`, `score_pairs.py`, `results/`), measured on a Vast 2× 5090 (500 W cap).
10. **Hosts differ in ways the listing does not show.** Vast m:150233 (Taiwan) has the same link
   bandwidth as the RunPod pod (PCIe 5.0 ×16, 56 GB/s host to card) but its cards sit on different CPU
   sockets (`SYS`): a 9.3 MB exchange takes 207 µs vs 161-184, the empty-message floor 140 µs vs
   97-121. Its cards are also capped at 500 W (default 575), which makes the model 7 % slower
   (0.667 vs 0.624 s per chunk) while a short matmul still reaches 227 TFLOP/s. Read the power limit
   (`nvidia-smi -q -d POWER`) and topology on every new pod.
11. **Splitting the DiT 9:3 pays (-21 % per layer); overlapping in head groups does not.** One DiT layer
   at real shapes (FP8 projections, SageAttention, local cross-attention, compiled token-local parts),
   steady-state ms per layer (median from layer 4 of 12): one card 3.86; split 6:6 3.79, or 3.14 with each
   message sent in 4 pipelined pieces; split 9:3 3.35, or **3.05** with 4 pieces; 9:3 plus head-group
   overlap 4.12 / 3.89 (fewer heads per SageAttention call leave SMs idle and add messages). Profiled
   9:3 layer: card 0 spends 1.90 ms at roofline, 0.72 ms of SageAttention running below peak (56 %),
   0.36 ms waiting; each 10.4 MB q|k|v message takes 0.57 ms, half the link rate one card reaches alone.
   Card 1 is busy 1.01 ms per layer, leaving ~2 ms for the decoder (needs 2.23). An earlier reading of
   "no gain" averaged 8 layers from a cold start and was wrong. `experiments/overlap_layer/`
   (`steady.py`, `breakdown.py`, `results/`).
13. **With the decoder on card 1, the 9:3 split loses to B1 (0.753 vs 0.579 s per chunk).** Measured with
   the real compiled decoder running on card 1 next to the split layers: the layer goes from 3.08 to
   10.41 ms. Card 1's DiT kernels slow 2.2× (1.01 -> 2.25 ms per layer) because the decoder's long
   convolutions fill its SMs, and card 0 waits on card 1 twice per layer, so the delay lands on card 0's
   critical path; the decoder itself only slows 16 % (355 -> 413 ms per chunk). Stream priorities cannot
   fix it today: SageAttention's CUDA kernels launch on the legacy default stream (wrong results on any
   other stream, with or without a device guard), so the DiT must stay on stream 0, which has the lowest
   priority. Even perfectly scheduled, card 1 needs 150 × 1.01 ms + 0.355 s = 0.505 s per chunk: at most
   +15 % over B1. `experiments/overlap_layer/with_decoder.py`.
14. **The exact bottleneck is head-of-line blocking on card 1, and the SageAttention stream patch only
   partly fixes it.** Before the patch, card 1's 34 DiT kernels in one layer waited 31.35 ms to start (ran
   2.30 ms, 1.08 alone) behind a cuDNN 3D convolution of the decoder (`sm80_xmma_fprop_implicit_gemm`,
   36,192 blocks, ~4.7 ms): at equal priority the block scheduler drains the earlier kernel's queued
   blocks first. SageAttention launched all 21 of its CUDA kernels without a stream (legacy default
   stream), so the DiT could not leave stream 0. Patched to launch on the current stream
   (`patches/sageattention-current-stream.patch`, same results, max diff 0.0), the DiT runs at priority -5:
   waits drop to 2.31 ms per layer, the layer with the decoder from 10.35 to 5.07 ms, the chunk from
   0.741 to 0.634 s. Resident decoder blocks still share card 1's SMs, so its DiT kernels run ~2× slower.
   With the decoder on card 1 every split loses to B1 (0.579 s): 9:3 0.634, 10:2 0.622, 11:1 0.659.
   Confining the decoder to its own SMs (MPS, CUDA green contexts) is the remaining lever; the container
   does not allow it. `experiments/overlap_layer/sched_analysis.py`, `with_decoder.py` (`HIPRIO=1`, `H0`).
15. **Green contexts make the split beat B1, with nothing approximated.** CUDA green contexts (driver API via
   `cuda-bindings`, PyTorch 2.8 has no wrapper; SM groups in multiples of 8 of the 5090's 170) confine card
   1's DiT share and the decoder to separate SMs. Sweep (chunk time vs B1's 0.579 s): 10:2 with 40 SMs for
   the DiT share and 130 for the decoder **0.497 s (-14 %)**, the layer 3.32 ms with the decoder running vs
   3.28 without (interference gone; unpartitioned +61 %); 10:2 at 32 / 48 / 64 SMs 0.526 / 0.532 / 0.521;
   9:3 at 32-64 SMs 0.743-0.569 (its larger card-1 share needs more SMs than the decoder can spare).
   Partitioning does not change results (matmul max diff 0.0). Projected ~30 FPS from B1's 25.69, still
   from 12 real layers, not the pipeline. `experiments/overlap_layer/greenctx.py`, `sweep_gc.sh`.
16. **Below 8 bits the 5090 is really faster; MXFP8 beats our FP8 kernel.** Matmul TFLOP/s at the DiT's
   shapes: BF16 176-229, our FP8 rowwise 375-430, INT8 537-648, **MXFP8 554-599**, **NVFP4 1,064-1,188**
   (8192^3: 227 / 462 / 671 / 640 / 1,186). NVFP4 is ~2.7x our FP8 but lossy. MXFP8 is the same precision
   class with per-32 block scales and runs 1.35-1.5x faster than the rowwise FP8 we use (cuBLAS path),
   a near-lossless lever on the ~36 % of DiT time in linear layers (needs a quality check). Bare GEMMs with
   constant scales; real use adds quantization kernels. `experiments/overlap_layer/lowbit_bench.py`.
17. **LightVAE (`lightvaew2_1`) is 9.2x faster and clearly lossy.** It is the Wan VAE at a quarter of the
   channels (`dim=24`), so it runs through our own fused compiled decoder: 39.8 vs 366 ms per 16-frame
   chunk. On identical latents vs the full VAE: PSNR 31.3 dB, LPIPS 0.10 (vs source 30.5 dB / 0.122;
   full VAE 34.9 / 0.017). Not usable under a lossless rule. `experiments/overlap_layer/lightvae_test.py`.
18. **MXFP8 as implemented is a loss; drop it.** The speed gap in learning 16 is real hardware (plain FP8 with
   FP32 accumulate runs at half rate on consumer Blackwell, block-scaled MXFP8 does not: FlashInfer RFC #3628
   measured ~102 vs ~202 TFLOP/s on sm_120), but in the model the per-32 activation quantization and scale
   swizzle (plain PyTorch, `LINGBOT_FP8=mx`) cost more than the GEMMs save, and power-of-two scales are less
   accurate than our exact per-row scales. One card: 0.895 vs 0.666 s per chunk; B1: 20.04 vs 24.19 FPS;
   first chunk LPIPS 0.040 vs a 0.030 noise pair, whole-clip LPIPS 0.19 vs 0.10. FFN only (`mx_ffn`): -1.5 %
   DiT time, first-chunk LPIPS 0.033, whole-clip 0.18: not lossless for 1.5 %. Revisit only with
   quantization fused into the producing kernel. `experiments/mxfp8/results_vast_2026-10-06.tsv`.
19. **Baselines on the Vast pod (500 W cap):** one card 0.666 s per chunk DiT, 15.7 FPS as played; B1 24.19
   FPS (RunPod: 25.69). Compare split results here against these.
20. **The warm bench worker removes the per-run cold start.** `python -m lingbot.worker run -- <generate
   args>` keeps weights loaded and the model compiled per code hash: a repeat run took 14 s instead of
   ~150 s, switching to B1 `--bench_e2e` in the same worker 38 s; results match direct runs (0.675 vs 0.666
   s per chunk, 24.04 vs 24.19 FPS). `experiments/mxfp8/worker_test_vast_2026-10-06.tsv`, `CLAUDE.md`.
12. **Engineering traps found on the way.** A cross-device `copy_` without P2P holds the CPU (~200 µs per
   9 MB, 38 µs for a tiny one), so it cannot sit in a loop that also launches compute; explicit pinned
   staging enqueues in ~6 µs. SageAttention does not order all its work after a custom current stream:
   run it on the default stream or its output is read before it is written. Driving two cards from one
   Python thread is CPU-bound on this host (engineering-sample Xeon, 15 µs per kernel launch).

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

## What others have found

- **A patched driver enables P2P on 5090s, but only on hardware you own.** tinygrad's
  open-gpu-kernel-modules patch and its forks (aikitoria, the CachyOS DKMS package) map each card's
  memory over PCIe BAR1. They need Resizable BAR, IOMMU in passthrough mode and ACS off, all host
  settings. Some 5090 owners still hit `cudaErrorMapBufferObjectFailed` with it.
- **Patched P2P is not faster than this pod's host-memory path.** Reported with the patch: about
  26 GB/s one way and 51 GB/s both ways, against 45 and 65 GB/s measured here through host memory.
  On a dual-5090 vLLM box, turning patched P2P on changed throughput by +2.7% on average (range
  −1.9% to +4.7%); an earlier multi-run test on the same box measured −5 to −7%.
- **Without P2P, inference engines fall back to NCCL and lose their fast paths.** vLLM disables its
  custom all-reduce (which reads the other card's memory directly) and warns; old NCCL versions
  (before 2.26.5) made 5090 tensor parallelism fail outright. pie-project saw two RTX PRO 6000s with
  P2P off run tensor parallelism at 5,380 tokens/s against 15,450 on one card, because every layer's
  activations went through host memory.
- **Engines recommend pipeline parallelism on PCIe without NVLink.** It splits the model by layers
  and sends one small activation between cards, instead of exchanging every layer's output. That is
  the DiT-on-one-card, decoder-on-the-other layout above.
- **NCCL's copy-engine collectives are NVLink-only.** NCCL 2.28 added collectives that use copy
  engines instead of GPU cores (`NCCL_CTA_POLICY_ZERO` with symmetric memory), documented for NVLink
  domains only. The hand-written copy-engine exchange above is the PCIe equivalent.
- **A newer NCCL does not fix the all-to-all hole.** NCCL 2.32.3 (the latest, loaded with
  `LD_PRELOAD`) measures 476 µs at 9.3 MB, the same as 2.27.3's 484 µs.

## Open questions

1. **Why does NCCL's all-to-all drop to 10 GB/s at 4-10 MB on the SHM transport?** Not in NCCL's
   issue tracker; present from 2.27.3 to 2.32.3 and in every protocol, algorithm and channel setting.
   Worth reporting upstream with `tools/linkbench/nccl.py`.
2. **How much communication can the model actually overlap?** Transfers cost no compute, but
   sequence parallelism needs each all-to-all's result before attention starts. Splitting each
   exchange into pieces and pipelining them against compute is the known technique (xDiT's
   PipeFusion, overlapped Ulysses); how much of the 159 ms per chunk it hides is untested.
   A first micro-benchmark (`tools/linkbench/overlap_layer.py`) was inconclusive: with small matmuls
   it is bound by Python launching kernels (its compute-only time varied 4.8-6.1 ms between runs),
   so it measures launch overhead, not overlap. It needs CUDA graphs or the real model.
3. **Do other 2× 5090 hosts wire the cards differently?** This one is `NODE`. A `SYS` host (cards on
   different CPU sockets) may be slower through host memory; RunPod listings do not say.
4. **Does a busy neighbour slow the host-memory path?** Host memory bandwidth is shared with other
   tenants on the machine; these numbers come from one quiet session.
5. **Do RTX PRO cards get P2P on RunPod?** NVIDIA allows P2P on professional cards, but cloud
   hypervisors can still block it (an ESXi report on RTX PRO 6000: missing PCIe ATS blocks P2P), and
   pie-project's RTX PRO 6000 pair ran with P2P off. Only `can_device_access_peer` on the pod answers it.
6. **Two processes or one? Answered: no difference.** With one process per card (as `torchrun`
   deploys), each process copies its half out of the other card's buffer through a CUDA IPC handle.
   It works without P2P, the data checks out, and it is as fast as one process
   (`tools/linkbench/ipc_a2a.py`):

   | Exchange | Two processes, CUDA IPC | One process | NCCL all-to-all |
   |---|---|---|---|
   | 4.6 MB | 114 µs | 104-143 µs | 246 µs |
   | 9.3 MB | 181 µs | 177-185 µs | 484 µs |
   | 18.5 MB | 321 µs | 330 µs | 388 µs |
   | 64 MB | 1,022 µs | 1,045 µs | 1,304 µs |

## Caveats

- One pod, one session. Another 2× 5090 host can wire its cards differently (`PIX`, `PHB` or `SYS`
  topology); rerun `tools/linkbench/run.sh` on each new pod.
- Copy-engine all-to-all times varied between runs (104 vs 143 µs at 4.6 MB, 177 vs 185 µs at
  9.3 MB); the predictions use 177 µs. The one-card FPS here (16.3) is 16 ÷ (0.64 + 0.34); the
  README's measured 16.1 includes run-to-run spread.
- The single-process measurements drive both cards from one process; a two-process setup (one per
  card, as `torchrun` launches) would need CUDA IPC or NCCL to reach the other card's memory.

## Sources

- [tinygrad/open-gpu-kernel-modules #44: P2P on 2× RTX 5090 fails](https://github.com/tinygrad/open-gpu-kernel-modules/issues/44)
- [aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules), [CachyOS P2P DKMS](https://github.com/A1RM4X/CachyOS-P2P-Nvidia)
- [vLLM forum: dual RTX 5090 TP=2, SHM vs patched BAR1 P2P](https://discuss.vllm.ai/t/dual-rtx-5090-tp-2-shm-vs-patched-bar1-p2p-cumem-single-pass-2-7-mean-point-estimate/2870)
- [vLLM forum: vLLM does not work with 2× 5090 in TP 2](https://discuss.vllm.ai/t/vllm-does-not-work-with-2x-5090-in-tp-2/1630)
- [pie-project #713: TP2 slower than TP1 over host memory](https://github.com/pie-project/pie/issues/713), [#741: peer all-reduce vs NCCL by message size](https://github.com/pie-project/pie/issues/741)
- [NVIDIA: copy engine collectives in NCCL 2.28](https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/)
- [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [RTX PRO 6000 Blackwell: no PCIe ATS blocks ESXi P2P](https://forums.developer.nvidia.com/t/rtx-pro-6000-blackwell-does-not-advertise-pcie-ats-blocking-esxis-p2p-path/362222)
- [xDiT](https://github.com/xdit-project/xDiT)
