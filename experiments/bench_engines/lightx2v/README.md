# LightX2V — LingBot-World 2.0 1.3B `causal_fast` on one RTX 5090

Scripts that benchmark ModelTC/LightX2V's `lingbot_world_fast` runner with the 1.3B checkpoint,
in the same shape as `bench/engines/flashdreams/` in the kernels repo. Everything here was
verified by reading LightX2V at commit `69018c92b0a42d9b0cf962a248fadbfe0cbc03de`
(2026-09-17, "feat: support pipefusion for flux2 (#1268)"); nothing was run on a GPU.

## Run order (pod: Ubuntu 24.04, driver 570, CUDA 12.8, Python 3.12, uv 0.12)

| Step | Command | Expected | Disk |
|---|---|---|---|
| 1 | `bash install.sh` | 5–10 min: torch 2.8.0+cu128, `pip install -e LightX2V` (pure Python), the two sm_120 wheels, then an import + `flash_attn_func` smoke test | ~8 GB in `/workspace/lx/.venv` |
| 2 | `bash download.sh` | 18.7 GB printed file by file before the download starts: 1.3B DiT shards 6.8 GB (fp32), UMT5 11.4 GB (bf16), Wan2.1 VAE 0.5 GB, tokenizer | `/workspace/lx/model` |
| 3 | `bash bench.sh 03` | 4–8 min: load (fp32 shards cast to bf16, T5 to CPU RAM), T5 encode + VAE encode of the 361-frame padded clip (once, ~1 min), 22 chunks of DiT + decode, mp4 write. Prints `LX ...` lines at the end | `/workspace/lx/out/out.mp4`, `lx_03_torch_sdpa.log` |

`ATTN=flash_attn2 bash bench.sh 03` or `ATTN=sage_attn2 bash bench.sh 03` re-run with the
other backends (both wheels come from the kernels repo release v0.1.0; sage is int8-QK, so it is
the "our stack" comparison, not the lossless one). `LX=/other/dir` and `EXAMPLES=...` override the
paths. `HF_TOKEN` must be in the environment for step 2.

## What the numbers mean

Every `[Profile]` line is wall time between two `torch.cuda.synchronize()` calls
(`lightx2v/utils/profiler.py`), so the per-chunk numbers are exact GPU time plus the CPU work in
between, not launch time.

- `LX chunk k dit=` — `chunk end2end k/22` from `wan_lingbot_fast_runner.py:129`: the 4 denoising
  forwards at timesteps `[999, 937, 833, 624]` plus the fifth t=0 forward that rewrites the KV
  cache with the clean chunk (`is_rerun=True`, `context_noise` 0). Same five forwards as upstream.
- `LX chunk k dec=` — `[AsyncVAEChunkDecoder] sync VAE chunk` (`utils/async_vae.py:122`): the
  Wan 2.1 causal decode of that chunk's 4 latents, 13 frames for chunk 1 then 16, synchronized
  before and after. Runs after the chunk's DiT, on the same stream (`async_vae_decode: false`).
- `LX steady chunks>=6 ... median` — medians over chunks 6–22 (1-based, as printed). The 18-frame
  window fills at chunk 5 (4 chunks × 4 latents = 16, chunk 5 is the first eviction), so chunk 6
  is the first fully steady one.
- `LX fps_as_played = 16 / median(dit + dec)` — the metric of `bench_perf.py` (`as_played_fps`),
  16 output frames per chunk. `fps_dit_only = 16 / median(dit)` is the denoise-loop number.
- `LX loop_total` — `AR chunk total 22 chunks`: all DiT + all decodes, sequential.
  `encoders` = T5 + VAE encode (once per clip, outside the loop). `total_cost` = process total
  including model load.
- `peak_vram_allocated_gb` / `reserved_gb` — `torch.cuda.max_memory_allocated/reserved` at process
  end. Reserved is what `nvidia-smi` shows minus the ~0.5 GB context.

Output: 349 frames at 832×464, 16 fps, 22 chunks. `num_frames: 361` → 91 latents → 88 after
`num_output_frames = lat_f - lat_f % 4` (`common/kvcache/manager.py:252`) → 13 + 21×16 = 349.
Upstream `generate.py --frame_num 361` does the same arithmetic when it has ≥361 poses.

## Settings that differ from our metric

Same as ours: 4 steps + 1 cache-write forward, chunk 4 latents, window 18 with sink 6 (identical
eviction arithmetic to upstream `model_fast.py:190-207`, checked line by line against
`self_forcing/transformer_infer.py:258-268` and `kvcache/rolling.py:378`), shift 5.0, bf16 DiT,
no quantization, no CFG, per-chunk causal Wan 2.1 VAE decode (13 then 16 frames), seed 42,
`examples/03`, 832×464 (`size: [480, 832]` is a max-area like upstream `--size 480*832`; for the
16:9 input it yields 464×832; `[464, 832]` would give 816×464).

Different:

1. **Attention.** `torch_sdpa` by default (torch's own flash kernel on sm_120); ours is
   SageAttention 2.2 sm_120 for self-attention and FA2 for cross-attention. `ATTN=sage_attn2`
   applies sage to all three attention keys (LightX2V's own lingbot config does the same).
2. **VAE precision.** LightX2V casts the whole VAE to the DiT dtype (bf16 weights, no autocast;
   `vae.py:887-905`). Ours is fp16 weights + channels_last + `torch.compile` (0.44 s per chunk);
   the stock reference is fp32. `vae_dtype: "fp32"` in `config_1p3b.json` gives the stock decoder.
   `vae_dtype: "fp16"` would crash: bf16 latents divided by fp16 scale promote to fp32 before the
   fp16 `conv2` (`vae.py:779-783`), so fp16 is only reachable with `DTYPE=FP16` for the whole
   pipeline.
3. **No compile, no CUDA graphs, no fusion** on LightX2V's AR path; `torch.cuda.empty_cache()`
   after every chunk (`wan_lingbot_fast_runner.py:148`); with profiling on, a synchronize
   around every `step_pre` / `infer_main` / `step_post`, so no CPU launch-ahead across steps.
   RMSNorm is the torch fallback (`sgl-kernel` not installed; the op has an ImportError fallback
   in `rms_norm_weight.py:328`). Causal RoPE is LightX2V's Triton kernel.
4. **Camera trajectory.** `examples/03` has 269 poses. Upstream truncates the clip to 269 frames
   (`image2video.py:915-917`; our `bench_perf.py --frame_num 361` runs actually produce 269 frames,
   17 chunks). LightX2V instead resamples the 269 poses onto the 88 latents
   (`wan_runner.py:_interp_c2ws_to_latf`), so the same path is played 1.3× slower over 349 frames.
   Per-chunk cost is unaffected; pixels are not comparable to upstream's. `num_frames: 269`
   reproduces upstream's exact frame count and pose sampling (17 chunks, 12 steady samples).
5. **Timesteps.** `timesteps_index [0, 250, 500, 750]` is upstream `generate()`'s default and
   with shift 5 evaluates to `[999, 937, 833, 624]` (same `FlowUniPC` arithmetic in
   `schedulers/wan/self_forcing/scheduler.py:40-51`). The `[999, 957, 899, 702]` in the brief is
   `[0, 179, 358, 679]` at shift 5, i.e. v1-fast's index list; if that is what our stack runs,
   change `timesteps_index`.
6. **T5** streams its 24 blocks from CPU RAM through one GPU buffer (`t5_cpu_offload: true`,
   `t5/model.py:487`), ~11 GB of host RAM, once per clip. Keeps ~11 GB of VRAM free for the
   5.0 GB KV cache (18 × 1508 tokens × 30 layers) and the 361-frame VAE encode.
7. **Intrinsics** are taken per frame and resampled (`Ks_np[ks_idx]`); upstream uses `Ks[0]` for
   all frames. Identical for `examples/03` (all 269 rows equal), and rescaled from 480×832 with the
   same `get_Ks_transformed` call.

## What the runner cannot express

- Truncating the clip to the pose count (item 4). Only `num_frames` in the JSON controls length;
  `--num_frames` on the CLI is rejected for this runner (`supported_request_fields` drops it).
- Frame count is otherwise free (`num_frames`), as are chunk (`num_frame_per_chunk`), window
  (`local_attn_size`), sink (`sink_size`), steps (`timesteps_index`), shift, resolution. Async /
  decode-first ordering exists (`async_vae_decode: true`, side stream) but is off here so `dec` is
  measurable.
- Interactive input: none. Camera comes from `poses.npy` + `intrinsics.npy` only; `action.npy` is
  read only with `control_type: "act"`.

## Unverifiable without a GPU

- That the 1.3B loads and matches upstream pixels. The weight-key set of the 1.3B index
  (`model.safetensors.index.json`, 1071 tensors) is exactly what `weights/lingbot/*.py` +
  `weights/transformer_weights.py` register (cam layers, `patch_embedding_wancamctrl` 1536×1536
  = 6 Plücker channels × 8×8 pixels × 1×2×2 patch, no `k_img`/`v_img`), and the Plücker packing
  order matches upstream's `rearrange`, but nobody has run this checkpoint through LightX2V.
  Compare `out.mp4` against a `generate.py` run with `--frame_num 269` before trusting quality.
- That `pip install -e LightX2V` on top of torch 2.8.0 keeps torch 2.8.0 (LightX2V pins nothing;
  `install.sh` asserts the version after install).
- VRAM: estimate 3.4 GB DiT + 5.0 GB KV + 0.25 GB VAE + activations; the 361-frame VAE encode
  and the chunk decodes set the peak.
