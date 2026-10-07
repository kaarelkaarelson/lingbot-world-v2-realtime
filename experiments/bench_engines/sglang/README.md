# SGLang v0.5.17 realtime pipeline, LingBot-World 2.0 1.3B `causal_fast`, one RTX 5090

Scripts to measure SGLang's LingBot-World realtime pipeline with our 1.3B checkpoint at our streaming settings. Based on `research/engines/sglang.md`; every flag, config key and message field was checked against the `sglang-src/` clone at tag `v0.5.17` (`29481685`). Nothing here has been run on a GPU.

## Run order (on the pod, from this directory)

| Step | Command | Expected | Output |
|---|---|---|---|
| 1 | `bash install.sh` | 15-25 min. Ends with `SGL install ok` and torch/sglang versions. ~10 GB at `/workspace/sgl/.venv`. | venv |
| 2 | `python repackage.py` | Prints disk and download sizes first (18.7 GB total), then downloads. 5-20 min depending on Hub speed. Ends with `SGL repackage ok`. Add `--dry-run` to only fetch the two small files and print the derived config. | `/workspace/sgl/lingbot-world-v2-1.3b-causal-fast-diffusers/` |
| 3 | `bash serve.sh` | Preflight prints the resolved config (must be `LingBotWorldV2CausalDMDConfig`, shift 5, `[1000, 750, 500, 250]`, 30 layers, block 4). Model load 2-5 min. Ends with `READY`. | `/workspace/sgl/serve.log`, `/workspace/sgl/serve.pid` |
| 4 | `python bench.py` | 23 chunks, 365 frames. Chunk 0 is slow (T5 encode on CPU, kernel autotune); budget up to a few minutes for it, then ~23 × per-chunk time. Prints `SGL ...` lines. | `/workspace/sgl/out.mp4` |

Step 2 needs `HF_TOKEN` in the environment (both repos are gated). Steps 3-4 need the venv active: `source /workspace/sgl/.venv/bin/activate`. Stop the server with `kill "$(cat /workspace/sgl/serve.pid)"`.

Disk: /workspace holds the venv (~10 GB) and the model (18.7 GB). `repackage.py` writes files straight into the model directory (no second copy) and caps the hf_xet chunk cache at 1 GB under `HF_HOME`, which it puts on whichever of `/workspace` and `/tmp` has more free space unless `HF_HOME` is already set. If /workspace is short, `SGL_VENV=/root/sgl-venv bash install.sh` puts the venv on the container disk (then `SGL_VENV` must be exported for `serve.sh` too).

## What the numbers mean

`bench.py` prints one line per chunk and three summary lines, all prefixed `SGL `.

Per chunk (all server-side, milliseconds, from the `chunk_stats` message; the same values are in `serve.log` as `realtime chunk timing:` lines):

- `prepare_ms`: sampling the camera script and building the request.
- `forward_ms` (`scheduler_forward_ms`): the whole chunk through the pipeline: cached T5, 4 DiT steps, streaming Wan-VAE decode of every latent frame. This is the number to compare with our per-chunk end-to-end time. There is no denoise-only figure.
- `total_ms`: `forward_ms` plus raw payload build and WebSocket write.
- `arrival_ms`: measured by `bench.py`, wall-clock between the last frame bytes of consecutive chunks. Output pacing is off, so this tracks generation, not the 16 fps playback clock.

Summary:

- `SGL steady chunks>=6 ... median_forward_s=X fps_forward=16/X`: the headline. Median over chunks 6-22 (17 chunks). SGLang's own harness (`test/server/realtime_consistency.py`) drops the first 2 chunks and reports p95; `p95_forward_s` is printed too.
- `fps_total`, `fps_arrival`: the same with `total_ms` and `arrival_ms`.
- `SGL wall ...`: frames / wall time for the whole session including chunk 0 and the T5 encode. Not comparable to anything of ours.
- `SGL ttff_s`: init sent to first frame bytes received.

Chunk 0 decodes to 13 frames (1 + 3 × 4), every later chunk to 16. `chunk0_frames` in the last line confirms it.

## Settings, and where they differ from our metric

Our metric (`BENCHMARK.md` / `OPTIMIZATIONS.md`): 832×464, chunk 4 latents = 16 frames, 4 steps, sink 6, window 18, Wan 2.1 VAE, 16 fps, flash_attn 2.8.3, torch 2.8 cu128.

| Setting | This setup | Same as ours? | How it is set |
|---|---|---|---|
| Resolution | 832×464 (latent 104×58, 1508 tokens/frame) | yes | `bench.py` `size` |
| Chunk | 4 latents = 16 frames | yes | `transformer/config.json` `num_frames_per_block: 4` (14B ships 3); `repackage.py` |
| Steps | 4 at `[1000, 750, 500, 250]` warped through shift 5 | same nominal schedule | `LingBotWorldV2CausalDMDConfig.dmd_denoising_steps`; the request's `num_inference_steps` does not change the count |
| Sink / window | 6 / 18 latent frames | yes | request `realtime_causal_sink_size` / `realtime_causal_kv_cache_num_frames` (config.json also set to 6 / 18; 14B ships sink 9); `local_attn_size -1` means the window is the KV cache size |
| VAE | Wan 2.1 VAE, one latent frame at a time with a persistent feature cache, fp32 | same model, different call pattern, fp32 not bf16 | server default `--vae-precision fp32`; add `--vae-precision bf16` to `serve.sh` for a bf16 decoder |
| DiT precision | bf16 (fp32 shards cast on load) | close | `--dit-precision` default |
| Attention | Torch SDPA (SGLang refuses FlashAttention on SM12.x) | no: we use flash_attn 2.8.3 | automatic, `runtime/platforms/cuda.py` |
| RoPE | FlashInfer in-place kernel from `flashinfer-jit-cache` (Triton fallback if the import fails) | SGLang caches RoPE/time/cam modulation across chunks; we recompute | `install.sh` |
| Text encoder | UMT5-XXL bf16, CPU-offloaded, encoded once per session | same cost class | `--text-encoder-cpu-offload true` |
| CFG | 1 | yes | request `guidance_scale` |
| Camera | key script, one action list per latent frame, mapped to poses by SGLang (0.05 m/step, 4° pitch, 6° yaw, K = [500, 500, w/2, h/2]) | same embedding only if those constants equal ours | `bench.py` `camera_script` |
| Seed | 42 | — | request `seed` |
| Output | raw RGB24 over WebSocket, x264 crf 12 for `out.mp4` | transport differs | request `realtime_output_format: raw` |
| KV-cache quant, torch.compile, warmup | off | — | `serve.sh` |

Label for the comparison row: "SGLang v0.5.17, sm_120 SDPA, chunk 4 / steps 4 / window 18 / sink 6, fp32 streaming VAE, raw output, per-chunk end-to-end (`scheduler_forward_ms`)".

## Where the research report and the v0.5.17 source disagree

- `runtime/loader/utils.py` in v0.5.17 has no `_select_safetensors_index_file`; only `diffusion_pytorch_model.safetensors.index.json` is read, and only to check shards exist. Our `model.safetensors.index.json` is ignored, which is fine: shards are globbed.
- The "Resolved model via explicit --model-id" line is logged at DEBUG, so `serve.sh` verifies config resolution in a preflight instead of grepping the log.
- v0.5.17 has no `quality: "high"` request switch for a bf16 VAE; decode precision is the server flag `--vae-precision`.
- The text encoder is bf16, 11.36 GB (HF API, `text_encoder/config.json` `torch_dtype: bfloat16`), not fp32 22.7 GB.
- The sglang-kernel 0.4.5 cu129 wheel has no sm_120 directory; on a 5090 it loads `sm100/common_ops.abi3.so`, which does contain sm_120 SASS. `sgl_kernel.rmsnorm` is used by the DiT's RMSNorm, so this wheel must load and run on the 12.8 driver.

## Not verified without a GPU

- That the cu128 torch + cu129 sglang-kernel + cuda-python 13 combination imports and runs on driver 570. `install.sh` ends with an import test; if `sgl_kernel` fails there, the answer is a driver ≥ 580 pod with stock `sglang[diffusion]`.
- That the FlashInfer RoPE kernel from `flashinfer-jit-cache` loads on sm_120; if `import flashinfer` fails the Triton fallback is used, if the import works but the kernel does not, chunk 0 errors.
- Peak VRAM at window 18 (estimate ~10 GB with T5 on CPU).
- The 13 / 16 frame split for chunk 0 / later chunks (inferred from the decode path).
- Whether SGLang's key-to-pose constants equal ours, so whether the camera trajectory is the same.
