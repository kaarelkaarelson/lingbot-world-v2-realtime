# LingBot-World 2.0 realtime

<p align="center">
  <a href="https://kaarelkaarelson.com/lingbot/"><b>Blog</b></a> &nbsp;·&nbsp;
  <a href="https://arxiv.org/abs/2607.07534">Original paper</a>
</p>

A 1.3B world model running at **<!-- n:fps_ours -->16.1<!-- /n --> FPS on one RTX 5090**. <!-- n:speedup_paper -->2.7×<!-- /n --> faster than the original paper's code with lossless performance.

![lingbot play dragon at 16 fps](docs/dragon_16fps.gif)

## 1× RTX 5090: performance vs other engines

<!-- table:engines -->
| Engine | s / chunk | FPS | Ours vs it |
|---|---|---|---|
| Original paper's code | 2.68 | 6.0 | **2.7×** |
| SGLang Diffusion | 2.48 | 6.45 | **2.5×** |
| LightX2V | 2.07 | 7.73 | **2.1×** |
| NVIDIA FlashDreams | 1.85 | 8.65 | **1.9×** |
| **Ours** | 0.98 | **16.1** | — |
<!-- /table:engines -->

![the same clip at each engine's measured cadence](docs/engines_same_clip_cadence.gif)

Measured with `lingbot bench` on a stock RunPod RTX 5090 (2026-09-17).

## 2× RTX 5090: pipeline and sequence parallelism

Frames per second with the decoder streaming, one Vast pod (2× RTX 5090, 500 W cap, PCIe, no P2P),
median of 3 warm runs. Not tensor parallelism: the cards split the pipeline (DiT on one, decoder on the
other) or the sequence (each card takes a share of the tokens and attention heads). Full log:
[`experiments/split_cpu/results/fps_log_2026-10-05.tsv`](experiments/split_cpu/results/fps_log_2026-10-05.tsv).

| Layout | FPS |
|---|---|
| One card | 15.7 |
| Classic split 6:6, decoder after the DiT on card 0 | 14.0 |
| **Pipeline (B1): DiT on card 0, decoder on card 1** | **24.0** |
| Sequence split 10:2 + green contexts, decoder on card 1 | 17.6 |
| … + reused transfer buffers | 19.0 |
| … + Inductor `cpp_wrapper` | 22.4 |
| **… + guard skipping + 2 transfer pieces** | **25.0** |

On a pod with a faster CPU (EPYC 9454, 575 W, PCIe 5.0 x16), same method:

| Layout | FPS |
|---|---|
| Pipeline (B1) | 25.3 |
| Sequence split 10:2, all of the above | 27.7 |
| … + no host sync per forward + 48 SMs for card 1's DiT share | 29.2 |
| **… + copy-free q\|k\|v assembly (`LINGBOT_SPLIT_ZEROCOPY=1`)** | **29.8** |

## How the world model works

The model generates video by predicting one chunk of 16 frames at a time. For each chunk:

1. **Take the inputs:** 4 camera poses (one per 4 frames), the text prompt and, in the first chunk only, the start image.
2. **Start from noise:** random numbers the size of 4 compressed *latent* frames (16 × 4 × 58 × 104), one per 4 video frames.
3. **Denoise 4 times:** each step is one pass of the 1.3B transformer that looks at the noisy chunk, the inputs and memory, and guesses the clean chunk, starting from less noise each time (100% → 94% → 83% → 63%).
4. **Save to memory:** one more pass over the clean chunk at 0% noise, so later chunks can look back at it.
5. **Decode:** the VAE decoder turns the 4 latent frames into 16 RGB frames.

## Speed of light

$$\text{floor} = \max\left(\frac{\text{FLOPs}}{\text{peak FLOP/s}},\ \frac{\text{bytes}}{\text{1,792 GB/s}}\right) \qquad \text{of speed of light} = \frac{\text{floor}}{\text{measured}}$$

<!-- table:roofline -->
| Operation | Precision and peak | Work / chunk | Bound | Floor | Measured | Of speed of light |
|---|---|---|---|---|---|---|
| Attention | INT8, 838 TOPS | 151 TFLOP | compute | 0.180 s | 0.288 s | **63 %** |
| DiT matmuls | FP8, 419 TFLOP/s | 79 TFLOP | compute | 0.188 s | 0.201 s | **94 %** |
| Decoder convolutions | FP16, 209.5 TFLOP/s | 52 TFLOP | compute | 0.249 s | 0.296 s | **84 %** |
| DiT norm, RoPE, modulation, residual | FP16, 1,792 GB/s | ~117 GB | memory | 0.065 s | 0.093 s | **70 %** |
| Decoder norm, SiLU, pad, upsample | FP16, 1,792 GB/s | ~40 GB | memory | 0.022 s | 0.032 s | **70 %** |
| Attention K/V re-quant | INT8, 1,792 GB/s | 38 GB | memory | 0.021 s | 0.039 s | **54 %** |
| **Chunk** | | | | **0.72 s, 22 FPS** | **0.98 s, 16.1 FPS** | **74 %** |
<!-- /table:roofline -->

Measured with the roofline method from Google's [How to Scale Your Model](https://jax-ml.github.io/scaling-book/).

## Quick start

### Setup

~15 min on Linux, needs a Hugging Face token to download the weights.

```bash
git clone https://github.com/kaarelkaarelson/lingbot-world-v2-realtime
cd lingbot-world-v2-realtime
HF_TOKEN=hf_... ./setup.sh && . .venv/bin/activate
```

### Play

```bash
lingbot play dragon
```

The first start compiles for about 2.5 min, later starts take 35 s.

### Commands

| Key | Action |
|---|---|
| `W` `A` `S` `D` | move (hold `Shift` to run) |
| `Q` `E` | down / up |
| `←` `→` `↑` `↓` | look (45°/s); mouse drag also looks |
| `R` | restart the world from the image |
| `Esc` | quit |

### Scripts

| Script | |
|---|---|
| `lingbot play [scene]` | a window on the world; scenes: `lake` (default), `wall`, `stonehenge`, `alley`, `castle`, `dragon` |
| `lingbot play --image me.jpg --prompt "..."` | your own world from any image |
| `SDL_VIDEODRIVER=dummy lingbot play --headless-seconds 120` | no display (a cloud pod): same model, no window, taps `W` and prints the HUD summary |
| `lingbot bench` | the minimal speed run: 10 chunks (5 warm-up, 5 steady), prints s/chunk and FPS; `--bench_e2e` adds first-frame latency |
| `lingbot clip --image me.jpg --action_path my_poses/ --prompt "..."` | offline generation from a camera path, `poses.npy` and `intrinsics.npy` as in `examples/` |

## Requirements

| | GPU | |
|---|---|---|
| Recommended | RTX 5090, 32 GB | everything here was measured on it; `setup.sh` ships prebuilt kernels for it (sm_120) |
| Minimum | RTX 4090, 24 GB | untested: every patch supports sm_89, expect ~12 FPS; needs `sageattention` and `flash_attn` built from source and T5 on the CPU to fit |

The default preset peaks at 30.7 GB of VRAM when it decodes a whole clip at once, which leaves
little headroom on a 32 GB card, and a long enough clip will run out. Decoding chunk by chunk
brings the peak to 26.4 GB.

One card serves one stream. Two streams cost 1.89 times one and four cost 3.82 times, because a
single stream already saturates the tensor cores, so the card puts out about 17 frames per second
in total however many streams share it.

## Optimizations

<!-- table:ladder -->
| Step | Change | Stored in memory | Computed in | s/chunk |
|---|---|---|---|---|
| Decoder | [Wan 2.1 VAE](https://arxiv.org/abs/2503.20314) in fp32 → **fp16 with [sub-pixel](https://arxiv.org/abs/1609.05158) upsampling** | FP32 → **FP16** | TF32 → **FP16** | 2.68&nbsp;→&nbsp;≈2.06 |
| Attention | FlashAttention-2 → **[SageAttention 2.2](https://arxiv.org/abs/2505.21136)** | BF16 | BF16 → **INT8 QKᵀ, FP8 PV** | ≈2.06&nbsp;→&nbsp;≈1.63 |
| Matmuls | bf16 linears → **FP8 rowwise via torch._scaled_mm** | BF16 → **FP8 weights, BF16 activations** | BF16 → **FP8** | ≈1.63&nbsp;→&nbsp;≈1.42 |
| Compiler | PyTorch eager, 13 graphs → **one compiled graph** | – | – | ≈1.42&nbsp;→&nbsp;≈1.15 |
| Host&nbsp;syncs | CPU↔GPU sync on every layer → **bookkeeping on the GPU** | – | – | ≈1.15&nbsp;→&nbsp;≈1.04 |
| Kernel&nbsp;fusion | one kernel per operation → **fused norm, RoPE, residual and FP8 conversion** | – | – | ≈1.04&nbsp;→&nbsp;0.98 |
| **Total** | 6.0&nbsp;FPS&nbsp;→&nbsp;**16.1&nbsp;FPS** | | | **2.68&nbsp;→&nbsp;0.98** |
<!-- /table:ladder -->

Per chunk, against the paper's code (profiler traces in `OPTIMIZATIONS.md` §13, §17; host syncs counted over three chunks):

<!-- table:baseline -->
| | Original paper's code | Ours |
|---|---|---|
| DiT | 1.62 s | **0.64 s** |
| Decoder | 1.06 s | **0.34 s** |
| GPU busy | 90% | **98%** |
| Kernel launches | ~20,000 | **~4,800** |
| Host syncs | 110 | **2** |
<!-- /table:baseline -->

What's left runs in four library kernels, three of them near the RTX 5090's peak (NVIDIA spec). Attention has the most headroom, but a custom kernel at 90% of peak would add only ~1 FPS (§17).

<!-- table:peaks -->
| Kernel | Reached | Peak on RTX 5090 | of peak |
|---|---|---|---|
| FP8 matmuls | 393 TFLOP/s | 419 TFLOP/s FP8 | **94%** |
| Decoder convolutions | 176 TFLOP/s | 210 TFLOP/s FP16 | **84%** |
| Norm, activation, residual | ~1.3 TB/s | 1.8 TB/s memory | **~70%** |
| SageAttention\* | 524 TOPS | 838 TOPS INT8 | **63%** |
<!-- /table:peaks -->

\* Limited by its softmax bookkeeping, not tensor throughput (§19–20).

### Quality

<!-- table:quality -->
| | What it measures | Paper (fp32 decoder) | Ours (fp16 decoder) |
|---|---|---|---|
| PSNR | Average pixel difference from the reference, in decibels; higher is closer | reference | **43.6 dB** |
| SSIM | How closely local structure and contrast match the reference; 1 is identical | reference | **0.981** |
| LPIPS | How different the frames look to a network trained on human judgements; 0 is identical | reference | **0.004** |
| MUSIQ | Image quality score from a model trained on human ratings; no reference | 68.98 | **68.99** |
| CLIP-IQA | How much a frame looks like "a good photo" to CLIP; no reference | 0.592 | **0.590** |
| Sharpness (Laplacian), first / last s | Edge detail; higher is sharper | 1022 / 298 | **1023 / 298** |
| Colourfulness, first / last s | How saturated and varied the colours are | 41.9 / 50.2 | **41.9 / 50.2** |
| Brightness, first / last s | Mean luminance, 0 to 1 | 0.692 / 0.384 | **0.692 / 0.384** |
| Flicker | Average change between consecutive frames; lower is steadier | 0.0381 | **0.0381** |
| DiT latents | Whether our rewritten DiT code matches the paper's bit for bit when both run the same BF16 math (FP8 and SageAttention off) | reference | **bit-identical** |
<!-- /table:quality -->

`OPTIMIZATIONS.md` is the full log. It has every experiment with its measurement, the profiles, and the levers that were tried and rejected.

## Two GPUs (2× RTX 5090, PCIe, no P2P)

What worked. Details and the negative results: [`2X_RTX5090_LEARNINGS.md`](2X_RTX5090_LEARNINGS.md).

| Result | Change | Measured |
|---|---|---|
| **Decoder on card 1** (pipeline parallelism, B1) | DiT on card 0, decoder on card 1, 0.77 MB of latents per chunk between them | 16.8 → **25.69 FPS** (1.52×), byte-identical to one card |
| DiT split 9:3 (sequence parallelism) | card 0: 75 % of tokens and 9 heads; card 1 the rest; q/k/v and outputs exchanged per layer through host memory in 4 pipelined pieces | 3.86 → **3.05 ms per layer** (−21 %), one real layer, DiT only |
| Green contexts on card 1 | card 1's SMs split 40 / 130 between its DiT share (10:2) and the decoder, so the decoder can't block the DiT's kernels | 0.579 → **0.497 s per chunk** (−14 %), one real layer, 12 deep |
| SageAttention stream fix | all 21 kernel launches use the current stream ([`patches/`](patches/sageattention-current-stream.patch)), so the DiT can run at high priority | kernel wait-to-start 31.35 → **2.31 ms** per layer; same output |
| **Split 10:2 + green contexts, full pipeline** | the split above with the decoder on card 1's 130-SM partition; Inductor `cpp_wrapper`, guard skipping after warm-up, 2 transfer pieces, reused transfer buffers | 17.64 → **25.03 FPS** (B1 on the same pod: 24.04); quality inside the noise band |
| Warm bench worker | `python -m lingbot.worker run -- …` keeps weights and compiled model per code hash | repeat run **14 s** instead of ~150 s, same numbers |

## Presets

| `--preset` | What runs | s / chunk | FPS |
|---|---|---|---|
| `stock` | the original paper's code | <!-- n:s_paper -->2.68<!-- /n --> | <!-- n:fps_paper -->6.0<!-- /n --> |
| `exact` | `fast` with the time-embedding MLP computed as in the paper; still FP8 and SageAttention, so not bit identical | <!-- n:s_exact -->1.07<!-- /n --> | <!-- n:fps_exact -->14.8<!-- /n --> |
| **`fast`** (default) | ours, FP8 linears, SageAttention, compiled and fused DiT, fp16 decoder | **<!-- n:s_ours -->0.98<!-- /n -->** | **<!-- n:fps_ours -->16.1<!-- /n -->** |

## Tests

`pytest tests/` runs on the CPU, no GPU needed. It checks the fused DiT and decoder against the paper's modules and against golden outputs recorded before the restructure (bit for bit), runs the whole generation loop with a tiny DiT and mock VAE to check the wiring, and runs `lingbot play --dry` on a stand-in model.

## Repository layout

| Path | What it is |
|---|---|
| `lingbot/models/lingbot_world/` | the model: fused DiT (`transformer.py`), fused decoder (`vae.py`), text encoder |
| `lingbot/layers/` | building blocks shared by any model: attention backends, FP8 linear, KV cache |
| `lingbot/pipelines/` | the generation loop, chunk by chunk |
| `lingbot/parallel/`, `lingbot/configs/hardware.py` | multi-GPU layout, and the default layout per GPU model and count |
| `lingbot/registry.py`, `lingbot/presets.py` | which model to build; `fast` / `exact` / `stock` runtime presets |
| `lingbot/generate.py`, `lingbot/cli.py`, `lingbot/play/` | offline generation, the `lingbot` command, the live player |
| `reference/` | the paper's code, unmodified: the `stock` baseline and the shared primitives |
| `experiments/` | code from experiments that did not ship (see its README and `OPTIMIZATIONS.md`) |
| `tests/`, `tools/` | CPU tests and golden outputs; README tables, roofline, pod checks |

## License and credit

This repository is derived from [LingBot-World 2.0](https://github.com/Robbyant/lingbot-world-v2) by the Robbyant team, whose [paper](https://arxiv.org/abs/2607.07534) is by Zelin Gao and others. The model, the sampler and the examples are theirs. The [weights](https://huggingface.co/robbyant/lingbot-world-v2-1.3b-causal-fast) are theirs too and are not redistributed here. Upstream is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), and so is this repository, see `LICENSE.txt`. That means non commercial use, attribution, and the same license for anything built on it. It is provided as is, without warranty. My changes are the inference patches listed under Optimizations and the `lingbot` CLI, applied on upstream commit `1895d30`. The `reference/wan/` directory is upstream's copy of [Wan2.2](https://github.com/Wan-Video/Wan2.2), which is Apache 2.0. The kernels used are [SageAttention](https://github.com/thu-ml/SageAttention), [torchao](https://github.com/pytorch/ao) and [FlashAttention](https://github.com/Dao-AILab/flash-attention).

```bibtex
@article{lingbot-world-v2,
  title   = {Infinite Worlds with Versatile Interactions},
  author  = {Zelin Gao and Qiuyu Wang and Jiapeng Zhu and Jingye Chen and Zichen Liu and Qingyan Bai and Jiahao Wang and Yufeng Yuan and Hanlin Wang and Yichong Lu and Ka Leong Cheng and Haojie Zhang and Jian Gao and Tianrui Feng and Yuzheng Liu and Yao Yao and Yinghao Xu and Xing Zhu and Yujun Shen and Hao Ouyang},
  journal = {arXiv preprint arXiv:2607.07534},
  year    = {2026}
}
```
