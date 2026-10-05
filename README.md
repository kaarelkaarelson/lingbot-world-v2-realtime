# LingBot-World 2.0 realtime

<p align="center">
  <a href="https://kaarelkaarelson.com/lingbot/"><b>Blog</b></a> &nbsp;·&nbsp;
  <a href="https://arxiv.org/abs/2607.07534">Original paper</a>
</p>

A 1.3B world model running at **<!-- n:fps_ours -->16.1<!-- /n --> FPS on one RTX 5090**. <!-- n:speedup_paper -->2.7×<!-- /n --> faster than the original paper's code with lossless performance.

![lingbot play dragon at 16 fps](docs/dragon_16fps.gif)

## Performance vs other engines

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

## How the world model works

The model generates video by predicting one chunk of 16 frames at a time. For each chunk:

1. **Take the inputs:** 4 camera poses for this chunk, one per 4 frames. The start image and text prompt are given once per rollout: the prompt is used in every chunk, the start image (encoded into a latent) only in the first; after that it lives on in memory.
2. **Start from noise:** 16 × 4 × 58 × 104 random numbers, the size of 4 compressed *latent* frames (one per 4 video frames).
3. **Denoise 4 times.** Each step is one pass of the 1.3B transformer. Its inputs are the noisy chunk, the noise level (100%, 94%, 83%, 63%), the camera poses as a 3D ray per pixel, memory (by default the first 6 and the latest 8 latent frames made so far), the prompt and, in the first chunk, the start image. It returns a guess of the clean chunk; after steps 1 to 3 noise is mixed back in at the next, lower level, and the guess from step 4 is the result.
4. **Save to memory:** one more pass over the clean chunk at 0% noise, so later chunks can look back at it.
5. **Decode:** the VAE decoder turns the 4 latent frames into 16 RGB frames.

## Speed of light

Every operation has a floor, either its arithmetic divided by the peak of the precision it runs at, or its memory traffic divided by the bandwidth, whichever is larger. Those floors add up to 0.72 s per chunk and the chunk takes 0.98 s, so the stack reaches 74 % of what the card physically allows. Every large operation is compute bound, which means the gap that remains is inside the kernels rather than in how data moves. The original paper's code reaches 80 % of its own floor, 2.16 s.

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
| `lingbot bench` | the 22 s clip to `outputs/`, prints s/chunk and FPS |
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

The table decodes the same latents two ways: the reference is the paper's fp32 decoder, "Ours" is our fp16 decoder. For example, PSNR 43.6 dB means a pixel differs from the reference by about 1.7 out of 255 on average, about the noise an mp4 encode adds. PSNR, SSIM and LPIPS compare the two frame by frame; the other metrics score each clip on its own. The last row is the DiT: its code changes give bit-identical latents when run in BF16 with FlashAttention-2. FP8 and SageAttention change the numbers slightly and are not in this table: SageAttention's first chunk matches FlashAttention-2 at cosine 0.999969, and FP8 with SageAttention moves first-chunk LPIPS by 0.005.

<!-- table:quality -->
| | What it measures | Original paper's code | Ours |
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
| DiT latents | Whether the DiT's output matches the paper's bit for bit, with our code changes run in BF16 and FlashAttention-2 | reference | **bit-identical** |
<!-- /table:quality -->

`OPTIMIZATIONS.md` is the full log. It has every experiment with its measurement, the profiles, and the levers that were tried and rejected.

## Presets

| `--preset` | What runs | s / chunk | FPS |
|---|---|---|---|
| `stock` | the original paper's code | <!-- n:s_paper -->2.68<!-- /n --> | <!-- n:fps_paper -->6.0<!-- /n --> |
| `exact` | `fast` with the time-embedding MLP computed as in the paper; still FP8 and SageAttention, so not bit identical | <!-- n:s_exact -->1.07<!-- /n --> | <!-- n:fps_exact -->14.8<!-- /n --> |
| **`fast`** (default) | ours, FP8 linears, SageAttention, compiled and fused DiT, fp16 decoder | **<!-- n:s_ours -->0.98<!-- /n -->** | **<!-- n:fps_ours -->16.1<!-- /n -->** |

The same seed does not give the same video twice on `fast`. Two identical runs differ by about 9.6
levels out of 255 on average, because FP8 and the attention kernel are not bit reproducible and the
difference compounds as the clip goes on. `exact` runs the same kernels, so it is not bit reproducible either. Latents bit identical to the paper's need FP8, SageAttention and the compiler off: `LINGBOT_FP8=0 LINGBOT_ATTN= LINGBOT_TORCH_COMPILE= LINGBOT_INDUCTOR_TUNE= lingbot clip --preset exact`.

## Tests

`pytest tests/` runs on the CPU, no GPU needed. It checks the fused decoder and DiT against the stock modules and runs `lingbot play --dry` on a stand in model.

## License and credit

This repository is derived from [LingBot-World 2.0](https://github.com/Robbyant/lingbot-world-v2) by the Robbyant team, whose [paper](https://arxiv.org/abs/2607.07534) is by Zelin Gao and others. The model, the sampler and the examples are theirs. The [weights](https://huggingface.co/robbyant/lingbot-world-v2-1.3b-causal-fast) are theirs too and are not redistributed here. Upstream is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), and so is this repository, see `LICENSE.txt`. That means non commercial use, attribution, and the same license for anything built on it. It is provided as is, without warranty. My changes are the inference patches listed under Optimizations and the `lingbot` CLI, applied on upstream commit `1895d30`. The `wan/` directory is upstream's copy of [Wan2.2](https://github.com/Wan-Video/Wan2.2), which is Apache 2.0. The kernels used are [SageAttention](https://github.com/thu-ml/SageAttention), [torchao](https://github.com/pytorch/ao) and [FlashAttention](https://github.com/Dao-AILab/flash-attention).

```bibtex
@article{lingbot-world-v2,
  title   = {Infinite Worlds with Versatile Interactions},
  author  = {Zelin Gao and Qiuyu Wang and Jiapeng Zhu and Jingye Chen and Zichen Liu and Qingyan Bai and Jiahao Wang and Yufeng Yuan and Hanlin Wang and Yichong Lu and Ka Leong Cheng and Haojie Zhang and Jian Gao and Tianrui Feng and Yuzheng Liu and Yao Yao and Yinghao Xu and Xing Zhu and Yujun Shen and Hao Ouyang},
  journal = {arXiv preprint arXiv:2607.07534},
  year    = {2026}
}
```
