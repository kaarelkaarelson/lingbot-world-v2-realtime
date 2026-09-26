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

## Speed of light

Every operation has a floor, either its arithmetic divided by the peak of the precision it runs at, or its memory traffic divided by the bandwidth, whichever is larger. Those floors add up to 0.72 s per chunk and the chunk takes 0.98 s, so the stack reaches 74 % of what the card physically allows. Every large operation is compute bound, which means the gap that remains is inside the kernels rather than in how data moves. The original paper's code reaches 80 % of its own floor, 2.16 s.

<!-- table:roofline -->
| Operation | Precision | Work / chunk | Bound | Floor | Measured | Of speed of light |
|---|---|---|---|---|---|---|
| Attention | INT8 | 151 TFLOP | compute | 0.180 s | 0.288 s | **63 %** |
| DiT matmuls | FP8 | 79 TFLOP | compute | 0.188 s | 0.201 s | **94 %** |
| Decoder convolutions | FP16 | 52 TFLOP | compute | 0.249 s | 0.296 s | **84 %** |
| DiT norm, RoPE, modulation, residual | FP16 | ~117 GB | memory | 0.065 s | 0.093 s | **70 %** |
| Decoder norm, SiLU, pad, upsample | FP16 | ~40 GB | memory | 0.022 s | 0.032 s | **70 %** |
| Attention K/V re-quant | INT8 | 38 GB | memory | 0.021 s | 0.039 s | **54 %** |
| **Chunk** | | | | **0.72 s, 22 FPS** | **0.98 s, 16.1 FPS** | **74 %** |
<!-- /table:roofline -->

Measured with the roofline method from Google's [How to Scale Your Model](https://jax-ml.github.io/scaling-book/). The two memory bound rows carry estimated bytes, because compiled kernels bypass the tracer.

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

Nothing about the model changed. The checkpoint, the sampler and the decoder are upstream's, with the same 4 steps, chunks of 4 latents and a KV window of 18 frames. I worked through the stack from the top down, cheapest and most general layer first, measured each step, and stopped at the kernel boundary. The table shows seconds per chunk after each step in the order they were applied. A chunk is 16 frames, one second of video.

<!-- table:ladder -->
| Step | Before | After | s/chunk |
|---|---|---|---|
| Host&nbsp;syncs | CPU↔GPU sync on every layer | bookkeeping on the GPU | 2.68&nbsp;→&nbsp;2.57 |
| Decoder | [Wan 2.1 VAE](https://arxiv.org/abs/2503.20314) in fp32 | fp16 with [sub-pixel](https://arxiv.org/abs/1609.05158) upsampling | 2.57&nbsp;→&nbsp;1.95 |
| Compiler | PyTorch eager | one compiled graph | 1.95&nbsp;→&nbsp;1.68 |
| Matmuls | bf16 linears | FP8 rowwise via [torchao](https://github.com/pytorch/ao/tree/main/torchao/float8) | 1.68&nbsp;→&nbsp;1.47 |
| Attention | FlashAttention-2 | [SageAttention 2.2](https://arxiv.org/abs/2505.21136) | 1.47&nbsp;→&nbsp;1.04 |
| Kernel&nbsp;fusion | one kernel per operation | fused kernels for norm, RoPE, residual and FP8 quant | 1.04&nbsp;→&nbsp;0.98 |
| **Total** | 6.0&nbsp;FPS | **16.1&nbsp;FPS** | **2.68&nbsp;→&nbsp;0.98** |
<!-- /table:ladder -->

\* `torch.compile` traces the model into a graph and fuses the small ops around it (norm, RoPE, residual adds) into fewer, larger kernels; it doesn't touch FlashAttention or the GEMMs, which stay calls into hand-tuned CUDA libraries. Tracing breaks wherever control flow reads a tensor's value, and the DiT forward started at 13 graphs and 12 breaks per call. Making the grid size and step count plain Python ints, precomputing the RoPE table, and caching the per-chunk camera-MLP output cut that to one graph, zero breaks (`OPTIMIZATIONS.md` §2, §10).

The table compares the original paper's code with ours, per chunk. GPU busy and kernel launches come from profiler traces of both, described in sections 13 and 17 of `OPTIMIZATIONS.md`. Host syncs are counted over three chunks.

<!-- table:baseline -->
| | Original paper's code | Ours |
|---|---|---|
| FPS | 6.0 | **16.1** |
| s / chunk | 2.68 | **0.98** |
| DiT | 1.62 s | **0.64 s** |
| Decoder | 1.06 s | **0.34 s** |
| GPU busy | 90% | **98%** |
| Kernel launches | ~20,000 | **~4,800** |
| Host syncs | 110 | **2** |
<!-- /table:baseline -->

What is left runs in four kernels written by others, and three of them are near the card's peak. Attention has the most room. A hand written kernel at 90 % of peak would gain about one frame per second, so there is none. The details are in section 17 of `OPTIMIZATIONS.md`. The peaks are from NVIDIA's RTX 5090 specification.

<!-- table:peaks -->
| Kernel | Reached | Peak on RTX 5090 | of peak |
|---|---|---|---|
| FP8 matmuls | 393 TFLOP/s | 419 TFLOP/s FP8 | **94 %** |
| Decoder convolutions | 176 TFLOP/s | 210 TFLOP/s FP16 | **84 %** |
| Fused norm, activation, residual | ~1.3 TB/s | 1.8 TB/s memory | **~70 %** |
| SageAttention | 524 TOPS | 838 TOPS INT8 | **63 %** |
<!-- /table:peaks -->

The result is lossless. Four of the six steps are bit identical to the paper's code, and FP8 and the attention kernel were checked on identical inputs. The attention kernel was measured again on a loop where nothing else varies, and its first chunk of latents matches FlashAttention-2 to a cosine of 0.999969, so the kernel itself is lossless and everything after that is the rollout drifting. PSNR, SSIM and LPIPS compare the same latents decoded by the paper's fp32 decoder and by ours. The rest are no reference metrics on the generated clips, measured on the first and last second. The numbers are in `quality_summary.tsv` from experiment 15.

<!-- table:quality -->
| | Original paper's code | Ours |
|---|---|---|
| PSNR | reference | **43.6 dB** |
| SSIM | reference | **0.981** |
| LPIPS | reference | **0.004** |
| MUSIQ | 68.98 | **68.99** |
| CLIP-IQA | 0.592 | **0.590** |
| Sharpness (Laplacian), first / last s | 1022 / 298 | **1023 / 298** |
| Colourfulness, first / last s | 41.9 / 50.2 | **41.9 / 50.2** |
| Brightness, first / last s | 0.692 / 0.384 | **0.692 / 0.384** |
| Flicker | 0.0381 | **0.0381** |
| DiT latents, exact preset | reference | **bit-identical** |
<!-- /table:quality -->

What those metrics mean. The first three need a reference frame to compare against, so they only
say something when both clips show the same thing. The rest score a clip on its own.

| Metric | What it measures | Reference | Better |
|---|---|---|---|
| PSNR | Average pixel error, in decibels. Strict, and it punishes a one pixel shift as hard as real damage. | needed | higher |
| [SSIM](https://doi.org/10.1109/TIP.2003.819861) | Local structure, contrast and luminance rather than raw pixel values. | needed | higher |
| [LPIPS](https://arxiv.org/abs/1801.03924) | Distance between two images as a pretrained vision network sees them, fitted to human judgements of which distortion looks closer. The most reliable of the three here. | needed | lower |
| [MUSIQ](https://arxiv.org/abs/2108.05997) | A quality score from a transformer trained on human ratings. | none | higher |
| [CLIP-IQA](https://arxiv.org/abs/2207.12396) | How close the frame sits to "a good photo" rather than "a bad photo" in CLIP's embedding space. | none | higher |
| Sharpness | Edge energy, the variance of a Laplacian filter. It rises for fine detail and for artifacts alike, so read it next to the others and never on its own. | none | context |
| Colourfulness | Spread and saturation of colour. | none | context |
| Brightness | Mean luminance. | none | context |
| Flicker | Mean change between consecutive frames, a stand in for temporal stability. | none | lower |

The split matters more than it looks, because the model is autoregressive and every chunk is
conditioned on the ones before it. Change a kernel or a precision and you do not get a worse version
of the same video, you get a different video that is just as coherent, so a reference metric late in
a clip is comparing two different scenes and reporting the difference as damage. That is why the
attention kernel above is checked on the first chunk, before the two runs drift apart.


`OPTIMIZATIONS.md` is the full log. It has every experiment with its measurement, the profiles, and the levers that were tried and rejected.

## Presets

| `--preset` | What runs | s / chunk | FPS |
|---|---|---|---|
| `stock` | the original paper's code | <!-- n:s_paper -->2.68<!-- /n --> | <!-- n:fps_paper -->6.0<!-- /n --> |
| `exact` | ours, with the DiT latents bit identical to the paper's bf16 model | <!-- n:s_exact -->1.07<!-- /n --> | <!-- n:fps_exact -->14.8<!-- /n --> |
| **`fast`** (default) | ours, FP8 linears, SageAttention, compiled and fused DiT, fused fp16 decoder | **<!-- n:s_ours -->0.98<!-- /n -->** | **<!-- n:fps_ours -->16.1<!-- /n -->** |

The same seed does not give the same video twice on `fast`. Two identical runs differ by about 9.6
levels out of 255 on average, because FP8 and the attention kernel are not bit reproducible and the
difference compounds as the clip goes on. `exact` is bit identical run to run and across machines.

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
