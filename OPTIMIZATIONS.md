# Kernel optimisation log — what worked and what did not

This is the experiment record behind the numbers in the README: LingBot-World 2.0 (1.3B
`causal_fast`) on one RTX 5090, from 5.5 FPS stock to 16.2 FPS as played with the original Wan 2.1
decoder. It is copied from the lab repository
([lingbot-world-v2-stream](https://github.com/kaarelkaarelson/lingbot-world-v2-stream), `OPTIMIZATIONS.md`),
where the raw timing tables, profiles, quality clips and red-team audits live; the pod/setup
details in it refer to that repository's scripts. Read it as a map of the design space: every
lever below is either in `--preset fast`, in `--preset exact`, or listed as a dead end with the
reason, so nobody has to re-try it.

## Scoreboard

| Lever | Effect on a 1 s chunk (DiT + decode) | Lossless? | Status |
|---|---|---|---|
| KV-cache bookkeeping without per-layer syncs (exp. 1) | first win over stock | yes | in both presets |
| `torch.compile` on the DiT (exp. 2) | large | yes | in both presets |
| DiT graph fusion: 13 graphs / 12 breaks → 1 / 0; RoPE cos/sin table; cross-attn K/V once per generation; camera MLP cached per chunk (exp. 10) | DiT 0.66 s | bit-exact with `EXACT_T=1` (one row of the time MLP) | in both presets |
| FP8 rowwise linears (exp. 4) | DiT forward 0.129 s at 6 032 tokens | same-latent 43.6 dB / SSIM 0.981 | `fast` |
| SageAttention 2.2 for sm_120 (exp. 5) | attention at 65 % of the 8-bit peak | same-latent lossless by metric | `fast` |
| Sync-free denoising loop, fp32c RoPE, Inductor coordinate-descent tuning (exp. 15) | 1.01 → 0.976 s/chunk | yes (fp32c RoPE only wins stacked with the tuning) | `fast` |
| Fused, compiled fp16 Wan VAE decoder + sub-pixel conv (exp. 9, 12, 15) | decode 1.05 → 0.34 s/chunk | 43.6 dB vs fp32 decode | both presets |
| **Final: 0.64 + 0.34 = 0.976 s/chunk, 16.2 FPS as played** | | | |

Dead ends, with the reason: TAEHV / LightTAE / Flash-VAED tiny decoders (fast but fail the eye
test — soft; exp. 3, 6, 8); CAS sharpening (adds contrast, not detail; exp. 7); KV-cache ring
buffer (two revisions, −15/+35 ms, net zero; exp. 11); Sage KV quantisation (negative in torch);
TensorRT / second-stream VAE (bounded below the budget by the roofline; exp. 10); batching two
players on one 5090 (compute-bound, 1.89× per chunk; exp. 16).

Determinism: the `exact` preset (eager bf16 / FlashAttention-2, `LINGBOT_DIT_FUSION_EXACT_T=1`) is
bit-identical run to run and across pods; the `fast` preset (FP8 + SageAttention + compile) is not
run-to-run reproducible at the pixel level (mean |Δ| ≈ 9.6/255 between identical runs), which is
why lossless claims for it are metric-based on the same latents, never byte comparisons.

---

# Optimization experiments — LingBot-World 2.0 (1.3B) on one RTX 5090

Each experiment is one modular patch under `patch/`, applied on top of the previous ones, and measured with one run of the same clip. The baseline is never edited; it lives in [BENCHMARK.md](BENCHMARK.md).

## Baseline (from BENCHMARK.md, run B)

`ex03`, 269 frames = 16.8 s at 832×464, 4 steps/chunk, 17 chunks, `--offload_model False`, stock inference code (loader patch only — it does not touch generation).

| Metric | Baseline |
|---|---|
| Denoise loop (17 chunks) | 28 s |
| Per chunk, steady state (chunks 8–17) | 1.75 s |
| Per chunk, first chunk | 1.27 s |
| Denoise-loop FPS | 9.6 |
| Generation end-to-end (T5 + DiT + VAE) | 75.4 s → 3.6 FPS |
| As played (denoise + decode per 1 s chunk) | ~2.9 s → ~5.5 FPS, 0.35× real-time |
| Peak VRAM | 31.8 GB |
| Host load average at start | 568 |

Reproduce: `python3 bench_perf.py --example 03 --frame_num 361 --no_offload --label <label>` on the pod.

## Measurement note

Experiments 0–1 were timed from tqdm, which reports whole seconds for the loop and an EMA rate per chunk. From experiment 2 on, `patch/instrument_chunk_timing.diff` (env-gated, `LINGBOT_BENCH_TIMING=1`, measurement only) logs an exact `cuda.synchronize`d time per chunk and for the VAE decode; `bench_perf.py` reports the steady-state mean over chunks 8–17 (the KV window is full from chunk ~6). The experiment-1 state was re-measured with it to give an exact reference:

| Exact reference (loader patch + exp. 1) | |
|---|---|
| Per chunk, steady state (chunks 8–17) | **1.645 s** (range 1.641–1.648) |
| First chunk | 1.61 s |
| Loop, 17 chunks | 27.39 s → 9.8 FPS |
| VAE decode, 269 frames | 17.9 s = 1.05 s per chunk |
| As played (1.645 + 1.05 per 1 s chunk) | **2.70 s → 5.9 FPS** |
| DiT forwards per chunk | 5 (4 denoising steps + 1 KV-cache write at t=0) |

## Experiment log

| # | Patch | What it changes | Steady s/chunk | Loop FPS | VAE decode | As played | Verdict |
|---|---|---|---|---|---|---|---|
| 0 | — | baseline | 1.75 (tqdm) | 9.6 | ~20 s | ~5.5 FPS | — |
| 1 | `patch/opt1_pr3_kv_sync.diff` | remove per-layer `.item()` GPU→CPU syncs in KV-cache eviction | 1.63 (tqdm) / **1.645** (exact) | 9.8 | 17.9 s | 5.9 FPS | keep: −6 %, bit-exact, free |
| 2 | `patch/opt2_torch_compile.diff` | `torch.compile(dit, dynamic=True)`, default inductor mode, env-gated | **1.555** | 10.3 (steady) | 17.9 s | 6.1 FPS | keep, off by default: −5.5 %, costs 92 s first launch and +0.3 GB VRAM at the 32 GB edge |
| 3 | `patch/opt3_taehv_decoder.diff` | TAEHV tiny decoder (`taew2_1`) instead of the Wan VAE, env-gated; compile off | 1.654 (unchanged) | 9.6 | **1.14 s** (15.7×) | **9.2 FPS** | keep for live play: +56 % as played, −4.5 GB VRAM; visibly softer foliage |
| 1+2+3 | all three | combined measurement | **1.554** | 10.2 | 0.67 s | **9.9 FPS** | gains stack additively; 52 s first chunk from Dynamo even with a warm kernel cache |
| 4 | `patch/opt4_fp8_linears.diff` | rowwise FP8 W8A8 on all 420 block Linears (`FP8Linear`, `_scaled_mm`), on top of 1+2+3 | **1.349** | 11.7 | 0.67 s | **11.4 FPS** | keep for play: −13 %; per-frame quality unchanged, rollout trajectory diverges from bf16 (not bit-comparable) |
| 5 | `patch/opt5_sageattention.diff` | SageAttention 2.2 (INT8 QK, FP8 PV) in place of FlashAttention-2, on top of 1–4 | **0.917** | 17.4 | 0.69 s | **16.5 FPS — real-time** | keep for play: −32 %; no measurable frame-quality loss at 17 s (QUALITY.md); trajectory not bit-comparable |
| 6 | `patch/opt6_lighttae_scale.diff` | LightTAE (`lighttaew2_1`, LightX2V) in place of TAEHV, fed the Wan-VAE latent space | unchanged | unchanged | 0.55 s (same) | unchanged | **keep as the live decoder**: MUSIQ back to the full VAE's level (69.0 vs 67.4), LPIPS 0.059 vs 0.072, less flicker; zero speed cost |

## 1. Remove per-layer GPU syncs in KV-cache bookkeeping

**Source:** upstream [PR #3](https://github.com/Robbyant/lingbot-world-v2/pull/3) (unmerged, closed without comment the day it was opened; applies cleanly to commit `1895d30`). 153 lines across `model_fast.py`, `image2video.py`, `sequence_parallel.py`.

**Why it should matter here.** Once the local attention window fills, `CausalWanSelfAttention.forward` reads `global_end_index` / `local_end_index` with `.item()` about seven times per layer per forward — a blocking GPU→CPU round-trip each. With 30 layers × 4 steps that is ~840 forced syncs per chunk, and each stalls the GPU until the host CPU answers. The PR author measured 905 → 1813 ms/step on an H200 when an unrelated CPU job was running. Our pod's host load average sat at 400–700 throughout, and our per-chunk time rises from 1.37 s (chunk 1) to 1.75 s (chunk 8+) — the point at which the window fills and the sync path starts. The PR keeps the two indices as Python ints (seeded in `_initialize_self_kv_cache`) and writes the tensor copies back with an async `fill_`; the author reports bit-exact latents (max abs diff 0.0) at 961 frames.

**Command:** `python3 bench_perf.py --example 03 --frame_num 361 --no_offload --label opt1_pr3_kvsync`

**Result** (2026-09-15 00:28 UTC, host load average 425):

| | Baseline | Exp. 1 | Δ |
|---|---|---|---|
| Per chunk, steady state | 1.75 s | 1.63 s | −7 % |
| Per chunk, chunks 1–4 (window not yet full, sync path inactive) | 1.27–1.50 s | 1.41–1.52 s | noise |
| Denoise loop (tqdm, whole seconds) | 28 s | 27 s | |
| Loop FPS | 9.6 | 10.0 | |
| As played (1.63 s + ~1.2 s decode per 1 s chunk) | ~5.5 FPS | ~5.6 FPS | |
| Mean GPU utilization in loop | ~90 % | 91 % | |
| Peak VRAM | 31.8 GB | 31.8 GB | |

**Verdict: keep.** The gain is real and confined to the window-filled regime where the syncs lived, but it is 7 %, not the 2× the PR author saw on an H200 under CPU load. Our loop was already ~90 % GPU-bound on this host, so there was little idle time to recover. Bit-exactness was not re-verified here (the author's claim); frames were visually identical.

**Not attributable to this patch:** generation end-to-end fell from 75.4 s to 58.5 s (3.6 → 4.6 FPS) in this run. That is the loader patch's on-disk prompt-embedding cache skipping the T5 forward pass entirely (~16 s), which was still executed in the baseline run. It is a legitimate, separate win for any fixed-prompt session and is credited to the loader patch, not to experiment 1.

## 2. `torch.compile` on the DiT

**Patch:** `patch/opt2_torch_compile.diff` (19 lines). After the DiT is loaded, `torch.compile(self.model, dynamic=True, mode=<env>)` when `LINGBOT_TORCH_COMPILE` is set (`1` = default inductor mode; or a mode name). Inert otherwise, so the same code path serves both arms.

**Why `dynamic=True`.** The forward takes `current_start` as a Python int that changes every chunk, and the KV-cache write uses Python-int slice bounds (experiment 1 made them ints). Dynamo specialises on ints by default, which would mean a recompile per chunk and, after 8, a silent fall back to eager. `dynamic=True` makes them symbolic. The script sets `TORCH_LOGS=recompiles` and counts recompile events in the log.

**What to expect.** The field reports 1.3–1.6× on 4-step Wan-1.3B loops (Self-Forcing "recommended", TeleFuser's 4×H100 gate uses it). The first chunk absorbs compilation time and is reported separately as first-chunk latency; the verdict is on the steady-state mean.

**Command:** `python3 bench_perf.py --example 03 --frame_num 361 --no_offload --compile 1 --label opt2_torch_compile`

**Result** (2026-09-15 00:36 UTC, host load average 409):

| | Exp. 1 (exact) | Exp. 2 | Δ |
|---|---|---|---|
| Per chunk, steady state (chunks 8–17) | 1.645 s | **1.555 s** (1.550–1.559) | −5.5 % |
| Steady-state loop FPS (16 frames / chunk) | 9.7 | 10.3 | |
| As played (steady + 1.05 s decode) | 2.70 s → 5.9 FPS | 2.61 s → 6.1 FPS | |
| First chunk | 1.61 s | 91.6 s (compilation; chunks 2 and 5 recompiled at 5.4 s and 6.6 s) | one-time per process; Inductor cache on disk makes later launches fast |
| Recompiles | — | 11, converged by chunk 6. Triggers: input dtype fp32 → bf16 between first and later steps; an optional kwarg that is `None` on the first call | |
| Graph breaks | — | two, at the KV-cache bookkeeping (`model_fast.py:118`) and cross-attention (`:303`), so each block compiles as several graphs | |
| Peak VRAM | 31.8 GB | 32.07 GB | 0.5 GB below the card's 32.6 GB |
| VAE decode | 17.9 s | 17.9 s | untouched (not compiled) |

**Verdict: keep, off by default.** 5.5 % for 92 s of first-launch compilation and 300 MB of VRAM we don't have to spare. The small gain is itself informative: at ~10 ms per transformer block and 91 % GPU utilization, the loop is bound by flash-attention and cuBLAS GEMMs — neither of which Inductor changes — not by Python or launch overhead. The 1.3–1.6× other stacks report comes from workloads with more overhead to remove. A CUDA-graphs variant (`reduce-overhead`) was not run: with dynamic shapes and two graph breaks it would re-record often, and the launch overhead it targets is ~5 ms of a 1550 ms chunk.

**Implication for what's next.** Further loop gains need fewer or cheaper FLOPs — FP8 GEMMs, fewer tokens (resolution), a smaller KV window — not better scheduling. And the decode (1.05 s per chunk, 40 % of as-played) is now the largest single lever.

## 3. TAEHV tiny decoder instead of the Wan VAE

**Patch:** `patch/opt3_taehv_decoder.diff` (37 lines) plus the vendored `wan/modules/taehv.py` from [madebyollin/taehv](https://github.com/madebyollin/taehv) at commit `011dfc2` and the `taew2_1.pth` weights (SHA-256 `d26151e7…c797e`, 22.7 MB) in the assets dir. Env-gated by `LINGBOT_TAEHV=<path>`; compile left off so the effect is isolated.

**Latent contract** (from the Strix Halo repo's validation on this exact model, confirmed here): TAE takes the model-space `x0` latents as-is — no Wan VAE mean/std transform (applying it oversaturates) — with layout `[C,T,H,W] → [1,T,C,H,W]`; output is `[0,1]`, rescaled to `[-1,1]` for the existing `save_video`. `parallel=True` OOMs at this length with T5 resident (12.5 GiB request); `parallel=False` (sequential, what Self-Forcing's demo uses and the analogue of per-chunk streaming) is used.

**Command:** `python3 bench_perf.py --example 03 --frame_num 361 --no_offload --taehv /workspace/lingbot-world-v2-14b-causal-fast/taew2_1.pth --label opt3_taehv`

**Result** (2026-09-15 00:44 UTC, host load average 468):

| | Exp. 1 ref (Wan VAE) | Exp. 3 (TAEHV) | Δ |
|---|---|---|---|
| Per chunk, steady state | 1.645 s | 1.654 s | unchanged |
| Decode, 269 frames | 17.9 s | **1.14 s** | 15.7× |
| Decode per 1 s chunk | 1.05 s | 0.07 s | |
| As played (loop + decode per chunk) | 2.70 s → 5.9 FPS | **1.72 s → 9.2 FPS** | +56 % |
| Generation end-to-end | 4.5 FPS | 6.1 FPS | |
| Peak VRAM | 31.8 GB | **27.3 GB** | the Wan VAE decode was the memory peak |

**Quality:** `videos/decoder_compare_wanvae_left_taehv_right.png` (2× crops, frames 60 and 200). Sky, mountains and water are near-identical; fine foliage is visibly softer and blobbier with some green-white fringing. High-frequency texture only. (The Strix Halo repo measured mean absolute frame difference 0.032 in [0,1] on the same model.)

**Verdict: keep for live play, not for reference output.** The decode is now 4 % of the per-chunk budget instead of 40 %, and the VRAM ceiling that made `--no_offload` marginal is gone. Same conclusion the Self-Forcing authors reached for their demo.

## Combined run (1 + 2 + 3)

`python3 bench_perf.py --example 03 --frame_num 361 --no_offload --compile 1 --taehv … --label opt123_combined` (2026-09-15 00:47 UTC, load average 404):

| | Baseline | Exp. 1 | + 3 | + 2 (all) |
|---|---|---|---|---|
| Per chunk, steady state | 1.75 s | 1.645 s | 1.654 s | **1.554 s** |
| Decode per 1 s chunk | ~1.05 s | 1.05 s | 0.07 s | 0.04 s |
| As played | ~5.5 FPS | 5.9 FPS | 9.2 FPS | **9.9 FPS, 0.62× real-time** |
| Peak VRAM | 31.8 GB | 31.8 GB | 27.3 GB | 27.3 GB |
| First chunk | 1.3 s | 1.6 s | 1.8 s | 52 s |

The three gains are additive (predicted 9.9 from the individual runs; measured 9.9). The 52 s first chunk with the Inductor kernel cache already warm shows that most of compile's start-up cost is Dynamo tracing and guard construction on this contended host, not kernel compilation — a per-process cost whichever way it's cached.

## Where this leaves the stack

Loader patch + exp. 1 + exp. 3 (compile off): **1.72 s per 1 s chunk → 9.2 FPS as played, 0.58× real-time**, 27.3 GB. All three: **9.9 FPS, 0.62×**, at the price of ~50 s per process start. 96 % of the remaining per-chunk time is the DiT's 5 forwards, which are bound by flash-attention and GEMMs. Next levers are FLOP reduction: FP8 GEMMs, fewer tokens (resolution), a smaller KV window.

## Profile of the per-chunk time (eager, exp. 1 + 3)

`python3 bench_perf.py … --taehv … --profile 8-10 --label profile_eager_opt13`, then `python3 analyze_profile.py profiles --chunk_s 1.654`. `torch.profiler` over steady-state chunks 8–10 (15 DiT forwards); device time from the Chrome trace's kernel events (`profiles/trace.json.gz`, open in Perfetto); GEMM FLOPs from the profiler, attention FLOPs analytic (q = 6 032 tokens per forward, kv = 27 144, 30 layers, 5 forwards per chunk). In-loop SM clock 2.75–2.89 GHz vs the 2.41 GHz the 209.5 TFLOP/s spec assumes, so utilizations are given against both. Instrumentation: `patch/instrument_profile.diff` (env-gated).

GPU kernel time 1.612 s per chunk against 1.654 s wall: **97 % busy**.

| Bucket | Per chunk | Share | Inside-kernel rate | vs spec peak | vs clock-corrected peak |
|---|---|---|---|---|---|
| Self-attention, `flash_fwd_kernel` (FA2, 900 calls) | 0.823 s | 51 % | 187 TFLOP/s | 89 % | 74 % |
| GEMMs, cuBLAS/CUTLASS bf16 (5 490 calls; `cutlass_80` sm80 kernels on Blackwell) | 0.472 s | 29 % | 199 TFLOP/s | 95 % | 79 % |
| Elementwise / norm / copy (unfused, mostly fp32; ~15 000 launches) | 0.263 s | 16 % | memory-bound | | |
| Memcpy DtoD (KV-cache window shift on eviction) | 0.053 s | 3 % | | | |

**Achieved 150 TFLOP/s per chunk: 71 % MFU vs spec, 60 % vs the observed clock.** Both large kernels are near their bf16 ceilings; the loop is compute-bound, not overhead-bound — consistent with compile's 5 %.

Levers, by the time they can touch:

1. Attention (51 %): only fewer FLOPs help — a smaller KV window (`local_attn_size` 18 → 12 ≈ −⅓ of attention FLOPs) or lower resolution. Both change model behaviour/spec; one run each to learn the slope.
2. GEMMs (29 %): FP8 (`torchao` float8 dynamic) — the only route past the bf16 peak; up to ~−0.24 s if the FP8 rate holds with fp32 accumulate. Quality cost to measure.
3. Elementwise (16 %): fix the two graph breaks so compile fuses the fp32 modulation/residual path, or run it in bf16.
4. Memcpy (3 %): ring-buffer KV cache instead of shifting.

Planned experiment 4: FP8 GEMMs — largest lever that leaves the model's inputs unchanged.

## Code review of the hot paths (2026-09-15)

From `wan/modules/attention.py` and `wan/modules/model_fast.py` on the pod, checked against the trace.

**Self-attention (51 %).** FlashAttention-2 2.8.3, generic sm80 kernel on sm_120 (FA3 is probed for but is Hopper-only). Full attention over cache + current chunk, no mask. Called through `flash_attn_varlen_func` for batch 1 with no padding: each call builds `q_lens`/`k_lens` from pageable CPU memory (`torch.tensor(...).to(device, non_blocking=True)` — synchronous in practice) plus `cat` + `cumsum` → ~600 pageable H2D copies per chunk (1 890 `Memcpy HtoD (Pageable→Device)` events over 3 profiled chunks). RoPE (`causal_rope_apply`) runs in fp64 complex and rebuilds the frequency table with `expand`/`cat` every call, 300× per chunk for identical positions; its `grid_sizes.tolist()` is a Dynamo graph break. KV-cache eviction shifts the whole 27 k-token window with `.clone()` once per chunk per layer instead of a ring buffer.

**GEMMs (29 %).** `nn.Linear` bf16 under autocast → cuBLAS `cutlass_80_tensorop_bf16` at ~79 % of peak; nothing to fix there. But the camera-injection MLP (`cam_injector_layer1/2`, `cam_scale_layer`, `cam_shift_layer`: four 1536×1536 Linears per block) is recomputed on all 5 forwards of a chunk although its input `c2ws_plucker_emb` depends only on the camera path — 18 % of GEMM FLOPs for one distinct result per chunk. One fp32 SIMT GEMM per forward (`cutlass_80_simt_sgemm`, ~18 ms/chunk) runs outside autocast somewhere small (time embedding or head).

**Elementwise (16 %), sub-bucketed from the trace, per chunk:** fp32 add/mul 0.134 s (the residual stream is kept in fp32 by design — `assert e.dtype == float32`, explicit fp32 autocast blocks — so each block runs ~8 separate memory-bound passes over a 37 MB tensor: modulation, two scaled residual adds, cam scale/shift); memcpy 0.053 s; bf16↔fp32 casts at every Linear boundary 0.048 s; RoPE fp64 0.026 s; GELU 0.018 s; LayerNorm/RMSNorm 0.024 s. This is what `torch.compile` fuses; the graph breaks are why it recovered only a third.

Exact, quality-free wins available: cam-MLP cache ~0.07 s; dense `flash_attn_func` 0.02–0.04 s; RoPE precompute 0.02–0.03 s; ring buffer ~0.02 s; remaining fusion ~0.05 s. Together ~0.15–0.2 s of the 1.55 s chunk.

## Optimization candidates from the literature (2026-09-15)

Anchor: **Light Forcing** (arXiv 2602.04789, github.com/chengtao-lv/LightForcing) — same backbone family (Self-Forcing / Wan2.1-1.3B causal, 4-step, KV window), measured on an RTX 5090, training-free, code released. Cumulative on a 5 s clip: FA2 baseline 9.09 s → +hierarchical sparse attention (88 % sparsity, Triton) 6.83 s (1.33×) → +FP8 linears 5.90 s (1.54×) → +fused RoPE/RMSNorm 5.37 s (1.69×) → +tiny VAE 2.96 s (3.07×, 27.4 FPS). Our TAEHV step is their tiny-VAE step; the other three map onto our attention / GEMM / elementwise buckets.

| # | Candidate | Expected on our 1.55 s chunk | Certainty | Effort | Quality cost | Source |
|---|---|---|---|---|---|---|
| 4 | Cache the camera-injection MLP per chunk | −0.07 s | high (exact) | low | none | code review |
| 5 | FP8 linears, torchao rowwise; keep the small projections bf16 (LongLive keeps 6 of ~300 in bf16) | −0.15 to −0.24 s | medium: `_scaled_mm` on sm_120 / CUDA 12.8 not benchmarked in any source found — microbench first. 5090 FP8 with fp32 accumulate ≈ 419 TFLOPS dense, 2× bf16 | low | "near-lossless" (Light Forcing, LongLive) | LightForcing README; NVlabs/LongLive |
| 6 | SageAttention 2.2 (INT8 QK, FP8 PV), drop-in `sageattn(q,k,v)` | −0.2 to −0.4 s (community: ~30 % faster sampling on 50-series) | medium: upstream lists Ampere/Ada/Hopper; sm_120 wheels are community builds | low | INT8 QK untested on camera-conditioned world models | github.com/thu-ml/SageAttention |
| 7 | Dense `flash_attn_func`, RoPE cos/sin precomputed per chunk, ring-buffer KV cache | −0.05 to −0.08 s | high (exact) | low–medium | none | code review |
| 8 | Fused RoPE/RMSNorm/adaLN Triton kernels (SGLang's reusable) or fixing the two graph breaks so Inductor fuses the fp32 path | −0.10 to −0.15 s (Light Forcing −9 % e2e; LongLive 2.0 +18.6 %) | medium | medium | none | LightForcing; NVlabs/LongLive |
| 9 | Light Forcing hierarchical sparse attention, ported to our window+sink cache | −0.2 s (their 1.33× e2e) | medium | high | small (VBench 84.5) | arXiv 2602.04789 |

Ruled out: FlashAttention-3/4 (FA3 is Hopper-only; FA4 needs TMEM absent on GB202 — sm_120 runs FA2; flash-attention issue #2307); NVFP4 (torchao requires CC ≥ 10.0; LightX2V's Wan2.2-NVFP4 is quantization-aware step distillation, not applicable to these weights; Nunchaku needs SVDQuant calibration with unverified Wan support); step/feature caching (TeaCache/MagCache — nothing to skip in a 4-step model; Sparse Forcing reports 1.1–1.17×); FlexAttention/cuDNN SDPA (window is dense, no block-skipping to exploit, no 5090 numbers).

Arithmetic: 4 + 5 + 7 + 8 ≈ −0.4 to −0.55 s → ~1.0–1.15 s per chunk (14–16 FPS) without touching attention precision; 6 or 9 on top is what reached 27 FPS in Light Forcing.

## Queue (decided 2026-09-15): biggest levers first, each atomic

Order: **FP8 linears** (exp. 4) → **SageAttention 2.2** (exp. 5) → the exact cleanups (cam-MLP cache, dense flash + RoPE precompute + ring buffer, fusion) → Light Forcing sparse attention. Reference for every FP8/attention experiment is the combined run (1.554 s per chunk, 9.9 FPS), since both need compile on.

### FP8 pre-check on the RTX 5090 (2026-09-15)

`torch._scaled_mm` rowwise (e4m3, fp32 accumulate, bf16 out) at the model's GEMM shapes, M = 6032:

| Shape | bf16 | FP8 kernel | Speedup | FP8 incl. unfused activation quant |
|---|---|---|---|---|
| K=1536 N=1536 (q/k/v/o, cam) | 0.149 ms, 191 TFLOP/s | 0.081 ms, 351 TFLOP/s | 1.84× | 0.156 ms (0.96×) |
| K=1536 N=8960 (FFN up) | 0.723 ms, 230 | 0.378 ms, 440 | 1.92× | 0.455 ms (1.59×) |
| K=8960 N=1536 (FFN down) | 0.808 ms, 205 | 0.415 ms, 400 | 1.95× | 1.231 ms (0.66×) |

The FP8 kernel doubles throughput as the 419 TFLOP/s spec promises — the previously unverified sm_120 / CUDA 12.8 path works. Unfused activation quantisation (amax → divide → cast in PyTorch) erases the gain, so FP8 only pays with the quantisation fused. torchao 0.13.0 (the release matching torch 2.8; 0.14+ requires torch ≥ 2.11) on a bf16 FFN block: eager 2.45 ms (slower than bf16's 1.65), **compiled 0.911 ms (1.80×)** — within 15 % of the GEMM-only ideal. Per-GEMM rel. error 3.7e-2 (rowwise), FFN block 5.3e-2.

## 4. FP8 linears

**Patch:** `patch/opt4_fp8_linears.diff` (58 lines). `FP8Linear`: rowwise e4m3 weights and dynamic rowwise e4m3 activations as plain buffers/ops, `torch._scaled_mm` with fp32 accumulate, bf16 out. Replaces all 420 `nn.Linear`s inside `model.blocks` (q/k/v/o, cross-attn q/k/v/o, FFN, the four camera-injection layers); the time-embedding MLP, text embedding and head stay as they are (the time-embedding path asserts fp32 — the first attempt that quantised it failed on that assert). Env-gated `LINGBOT_FP8=1`; requires compile.

**Why not torchao.** torchao 0.13.0's `Float8DynamicActivationFloat8WeightConfig(PerRow())` quantised 427 layers fine but compile failed at guard construction (`Cannot access data pointer of Tensor (FakeTensor)`): its `Float8Tensor` subclass and our `dynamic=True` don't mix. LongLive validates exactly torchao 0.13 + torch 2.8.0+cu128 + compile for the same rowwise W8A8 recipe — on H100 with static shapes, recompiling per KV-cache shape. `FP8Linear` is the same math without the subclass. Eager FP8 is slower than bf16 (unfused quantisation); compiled it reaches 1.8× on the FFN block.

**Command:** `python3 bench_perf.py --example 03 --frame_num 361 --no_offload --compile 1 --taehv … --fp8 --label opt4_fp8`

**Result** (2026-09-15 01:45 UTC, load average 453):

| | Reference (1+2+3) | Exp. 4 | Δ |
|---|---|---|---|
| Per chunk, steady state | 1.554 s | **1.349 s** (1.346–1.352) | −13.2 % (predicted −0.2 s from the microbench; measured −0.205) |
| As played | 1.593 s → 9.9 FPS | **1.388 s → 11.4 FPS** | +15 % |
| Peak VRAM | 27.3 GB | 26.5 GB | |
| First chunk | 52 s | 76 s | more graphs to compile; chunks 2 and 5 recompile (8.4 s, 7.1 s) |

**Quality** (`videos/fp8_compare_bf16_left_fp8_right.png`, same seed and decoder): per-frame fidelity unchanged — frame 60 is near-identical, no artifacts anywhere. The rollout diverges: mean abs frame difference vs bf16 grows from 0.017 in the first second to 0.070 in the last (one frame of motion ≈ 0.030), and by frame 200 the scene layout differs. Autoregressive amplification of ~4 % per-GEMM rounding, the same effect as a different seed. "Near-lossless" in the literature means per-frame quality, not identical trajectories.

**Verdict: keep for live play; not for bit-comparable replay.**

## Where this leaves the stack (after exp. 4)

1 + 2 + 3 + 4: **1.349 s per chunk, 11.4 FPS as played, 0.71× real-time**, 26.5 GB. Per-chunk budget now ≈ 0.82 s attention (61 %), ~0.27 s GEMM, ~0.26 s elementwise/memcpy. Attention is next (exp. 5, SageAttention 2.2).

## 5. SageAttention 2.2 in place of FlashAttention-2

**Patch:** `patch/opt5_sageattention.diff` (33 lines in `wan/modules/attention.py`): with `LINGBOT_ATTN=sage`, the batched no-padding call goes to `sageattn(q, k, v, tensor_layout="NHD", is_causal=False)`; everything else unchanged. Built from upstream `thu-ml/SageAttention` at `d1a57a5` with `TORCH_CUDA_ARCH_LIST=12.0` (upstream lists 12.0 as supported; needs CUDA ≥ 12.8; do not add 9.0 — its `wgmma` kernels don't compile for sm_120, upstream issue #291). Build took 26 min on the loaded pod; the wheel is cached at `/workspace/wheels/` and `wheels/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl` — reinstall is `pip install <wheel>`. The community "PyTorch header patch" is an MSVC/torch-2.11-nightly problem and does not apply on Linux with torch 2.8. On sm_120 the dispatch is the Ada-class `sageattn_qk_int8_pv_fp16_cuda` path.

**Standalone check at the model's shapes** (q 6 032 × kv 27 144, 12 heads × 128, bf16): FA2 dense 5.365 ms, FA2 varlen (as the repo calls it) 5.413 ms, **SageAttention 2.115 ms — 2.54×** (476 vs 187 effective TFLOP/s). Rel. error vs fp32 SDPA: FA2 2.2e-3, Sage 3.9e-2.

**Command:** `python3 bench_perf.py --example 03 --frame_num 361 --no_offload --compile 1 --taehv … --fp8 --attn sage --label opt5_sageattn`

**Result** (2026-09-15 02:14 UTC, load average 431):

| | Reference (1–4) | Exp. 5 | Δ |
|---|---|---|---|
| Per chunk, steady state | 1.349 s | **0.917 s** (0.901–0.931) | −32 % |
| As played | 1.388 s → 11.4 FPS | **0.957 s → 16.5 FPS** | +45 %; **1.03× real-time** |
| Per-chunk time vs KV fill | rises ~0.3 s once the window fills | flat from chunk 3 | SageAttention scales more gently with key length |
| Peak VRAM | 26.5 GB | 26.5 GB | |
| First chunk | 76 s | 70 s | Dynamo traced `sageattn` without graph breaks |

**Quality** (`videos/sage_compare_fp8_left_sage_right.png`; numeric vs bf16 and FP8 runs, same seed and decoder): first second indistinguishable (mean diff vs FP8 0.012, under one frame of motion); trajectory diverges like FP8's; end-of-clip sharpness differs (bf16 288, FP8 251, FP8+Sage 216) but see QUALITY.md: every configuration including bf16 loses ~70 % sharpness over the clip because the camera pans to open water and the scene darkens — the spread between configs at the end is inside the "different world" effect and is **not** attributable to INT8 attention. First-chunk LPIPS vs bf16 is 0.088, the same as FP8-only; MUSIQ/CLIP-IQA are not lower. Mean brightness unchanged: no colour drift.

**Verdict: keep for live play — this is the run that reaches real-time.** Quality attribution in QUALITY.md: the visible softening in the stack comes from TAEHV, not from INT8 attention. Re-check on a 60 s rollout before relying on it for long sessions.

## Where this leaves the stack (after exp. 5)

1 + 2 + 3 + 4 + 5: **0.917 s per chunk, 16.5 FPS as played, 1.03× real-time**, 26.5 GB. From the stock 5.5 FPS: 3.0×, five patches, no retraining. Remaining per-chunk budget ≈ 0.39 s attention, 0.27 s GEMM, 0.26 s elementwise/memcpy — the exact cleanups (6–8) now target ~30 % of what's left.

## 6. LightTAE decoder in place of TAEHV (quality, not speed)

**Patch:** `patch/opt6_lighttae_scale.diff` (17 lines): with `LINGBOT_TAEHV_SCALE=vae` the tiny decoder is fed `x0 · std + mean` — the Wan VAE's own latent space — instead of the model-space `x0`. `bench_perf.py` sets it automatically when the decoder file name contains `lighttae`. Weights: `lighttaew2_1.pth` from huggingface.co/lightx2v/Autoencoders (45 MB fp32; **same architecture and key set as `taew2_1`**, re-distilled by LightX2V against the official VAE — TAEHV's loader takes it as-is).

**Why the convention matters.** Fed the model-space `x0` like TAEHV, LightTAE produces washed-out, brown-tinted output (PSNR 15.4 dB, LPIPS 0.43 vs the reference, `videos/decoder_triple_wanvae_taehv_lighttae.png` before the fix). TAEHV has the opposite convention (the Strix Halo repo found the VAE transform oversaturates it). Neither decoder documents this.

**Measurement** (same-latent triple on pod 3, `ex03` seed 42, bf16 loop without compile/FP8/Sage so the latents are identical; `quality_metrics.py --comparable`):

| | Wan VAE (ref) | TAEHV | LightTAE |
|---|---|---|---|
| Decode, 269 frames | 17.9 s | 0.55 s | **0.55 s** |
| MUSIQ mean | 68.98 | 67.43 | **69.00** |
| CLIP-IQA mean | 0.592 | 0.621 | 0.611 |
| LPIPS vs ref, first chunk / whole clip | — | 0.084 / 0.072 | **0.062 / 0.059** |
| PSNR / SSIM vs ref, whole clip | — | 28.9 / 0.865 | 28.6 / 0.841 |
| Laplacian sharpness, first / last second | 1022 / 298 | 686 / 240 | 607 / 279 |
| Flicker | 0.038 | 0.039 | **0.036** |

MUSIQ and LPIPS — the two metrics the validation literature treats as decisive — both move to the reference's level. PSNR/SSIM are marginally lower and Laplacian variance is lower in the first second: LightTAE is smoother-but-cleaner where TAEHV's blobby fringing registered as spurious high-frequency energy (the crops agree). Reproducibility note: the TAEHV row reproduced pod 1's numbers to four decimals — the bf16 loop is bit-identical across pods.

**Verdict: keep. LightTAE replaces TAEHV as the live decoder at zero speed cost.** The as-played numbers (16.5 FPS on pod 1, 17.7 FPS on pod 3) are unchanged.

## Pod 3 reproduction (2026-09-15)

Full stack (1–5) on the replacement pod (`y2yv1ex472v00o`, idle host, load average 6): **0.859 s per chunk, 17.7 FPS as played** vs 0.917 s / 16.5 FPS on pod 1 (load average ~430). The 6 % difference is host CPU contention reaching into a GPU-bound loop through kernel-launch latency; earlier numbers in this file were measured on the loaded host. Model load 7 s, first chunk 44 s with a fresh compile cache.

## 7. CAS sharpening on the tiny-decoder output — negative result

**What:** AMD FidelityFX Contrast-Adaptive Sharpening (`cas_sharpen.py`, ported from `ffx_cas.h`: 3×3 cross soft-min/max, `amp = √(saturate(min(mn, 1−mx)/mx))`, `peak = −1/lerp(8,5,s)`, green-channel weight, run in linear light via the gamma-2.0 approximation the source prescribes — a first version applied it on gamma-encoded frames, which the source warns "will yield over-sharpening", and was discarded). Applied to the same-latent LightTAE clip; measured vs the full Wan VAE decode of identical latents.

| | Full VAE | LightTAE | + CAS 0.3 | + CAS 0.6 | + CAS 1.0 |
|---|---|---|---|---|---|
| Laplacian sharpness (first s) | 1022 | 607 | 1264 | 1531 | 2499 |
| SSIM vs VAE | — | 0.841 | 0.800 | 0.788 | 0.746 |
| LPIPS vs VAE | — | 0.059 | 0.095 | 0.110 | 0.165 |
| MUSIQ | 69.0 | 69.0 | 70.7 | 71.1 | 71.8 |
| Flicker | 0.038 | 0.036 | 0.040 | 0.041 | 0.045 |

**Reading:** sharpening raises the Laplacian past the reference and pleases the no-reference metrics, while every fidelity metric to the real decoder output worsens monotonically from the lowest strength, and temporal flicker rises. It exaggerates edges the tiny decoder drew; it cannot recover texture it never drew. The Laplacian-variance metric is satisfiable without any gain in true detail — a reason to keep it secondary to same-latent SSIM/LPIPS and human A/B.

**Verdict: not a quality fix.** Kept as an optional display-side knob (~0.15 lands near the reference's Laplacian) for taste; off by default.

## 8. Flash-VAED decoder (pruned Wan 2.1 decoder, arXiv 2602.19161)

**Patch:** `patch/opt8_flashvaed.diff` (`_flashvaed_decode` in `patch/image2video.stage8.py`). `LINGBOT_FLASHVAED=<Flash_VAED_Wan.pth>` loads the released `model_hybrid_aggressive` student (5.86 M params, the paper's variant; bf16 autocast, same latent contract as `Wan2_1_VAE.decode`). `bench_perf.py --flashvaed <path>`.

**Upstream bug found (first report; repo is six weeks old).** `WanVAE_.decode(i, z, scale)` clears its temporal feature cache when `i == 0` and again when `i == 20` — an "end of clip" cleanup hard-coded to the paper's 81-frame (21-latent) training clips; the paper and README never mention it, and stock Wan 2.1 `vae.py` clears only after the loop. On our 68-latent clip the run as shipped produced 266 frames instead of 269 (latent 21 takes the first-frame `'Rep'` path: 1 output frame instead of 4, so everything after frame 81 is 3 frames early) and index-aligned PSNR vs the Wan VAE decode of the same latents fell from 27.5 dB to 20 dB after 12 s. Three red-team passes (code, paper, literature) agreed the collapse was that misalignment plus one corrupted latent, not cache drift: the decoder is a finite-receptive-field causal-conv stack (CACHE_T=2 conv caches, 2-D attention, per-position RMSNorm, no positional state), so a continuous cache is exact. Measured: one uninterrupted cache over all 68 latents gives PSNR 29.923 dB; 21-latent windows with a 4-latent re-warm give 29.924 — identical in every 2-second bucket. The patch now bypasses `decode()` and drives `decoder` with one persistent cache (`LINGBOT_FLASHVAED_WINDOW>0` keeps the windowed scheme as a knob; asserts on `warm ≥ 1` and on missing decoder weights, since upstream loads with `strict=False`). Streaming use needs no reset schedule.

**Measurement** (same latents as exp. 6: `ex03` seed 42, bf16 loop; `quality_metrics.py --comparable`, `quality_results_opt8b.tsv` / `quality_results_opt8c.tsv`):

| | Wan VAE (ref) | TAEHV | LightTAE | Flash-VAED as shipped | Flash-VAED fixed |
|---|---|---|---|---|---|
| Decode, 269 frames | 17.9 s | 0.55 s | 0.55 s | 1.83 s | 1.84 s (0.11 s/chunk) |
| PSNR / SSIM vs ref, whole clip | — | 28.96 / 0.865 | 28.63 / 0.842 | 25.11 / 0.759 | **29.92 / 0.879** |
| LPIPS vs ref, first chunk / whole clip | — | 0.084 / 0.071 | 0.062 / 0.058 | 0.067 / 0.129 | 0.067 / **0.056** |
| PSNR per 2 s, 0→17 s | — | 26.8 … 27.2 | 26.5 … 26.5 | 27.5 27.8 25.8 28.9 24.7 26.6 21.3 19.8 20.5 | 27.5 27.8 30.1 31.9 31.0 31.0 31.0 29.7 29.1 |
| MUSIQ mean | 69.1 | 67.6 | 69.1 | 64.4 | 64.3 |
| CLIP-IQA mean | 0.593 | 0.622 | 0.611 | 0.542 | 0.541 |
| Laplacian sharpness, first / last chunk | 1022 / 283 | 686 / 234 | 607 / 256 | 442 / 126 | 442 / 120 |
| Flicker | 0.038 | 0.039 | 0.036 | 0.037 | 0.037 |

Per frame the fixed decoder beats TAEHV by 0.5–2 dB almost everywhere, yet its Laplacian energy is ~42 % of the reference's throughout (TAEs 60–100 %) and MUSIQ/CLIP-IQA sit 4.8 points / 0.05 below. The softness is by design, not a bug: the student is distilled with 10·L1 + 5·SSIM + 2·LPIPS + distillation and no adversarial loss (the Wan teacher was GAN-finished), and the two high-resolution stages that carry fine detail (up2/up3) are pruned to 24/12 channels and made 2-D; the paper's limitations section admits "motion blur … loss of fine details". Higher PSNR with lower sharpness than TAEHV is the L1-vs-GAN signature. Our 29.9 dB is not comparable to the paper's 37.6 dB (theirs is vs ground-truth real video at 480p/81 frames; ours is vs the teacher's decode of generated latents) — a real-video encode→decode control at 832×464 would separate content from latent-distribution shift.

**Verdict: fits the real-time budget (0.11 s/chunk vs 0.07 s TAEs, 0.86 s DiT) and is the fidelity leader; not adopted.** The user's complaint is softness, and this is the softest of the three (A/B: `videos/decoder_abc_wanvae_taehv_flashvaed_labeled.mp4`). No decode-time trick changes it; the literature's only remedies are training-side (see `vae_decoder_dropoff_literature.md`, `tiny_ae_error_correction_report.md`). Worth an upstream issue: delete the `i == 20` reset.

## 9. Full Wan VAE decoder — lossless speed levers (2026-09-15)

**Goal:** keep the original decoder (the tiny decoders are the visible quality loss, exp. 3/6/8) and measure how close lossless levers get it to the ~0.14 s per 16-frame chunk that real time leaves next to the 0.86 s DiT. The pipeline runs `Wan2_1_VAE` at its default `dtype=torch.float` — **fp32**, the slowest configuration.

**Method:** `vae_decode_bench.py` on the pod. Latents dumped from the reference run (`LINGBOT_DUMP_LATENTS`, `latents_ex03_s42.pt`; that run reproduced `q_ref_wanvae.mp4` bit-for-bit), then each lever applied alone against the fp32 whole-clip decode; fidelity = PSNR / max-abs / share of values off by more than 4 8-bit levels, on the decoded frames; timing = mean of 2 timed passes after 2 warm-ups (`torch._dynamo.reset()` per variant, `recompile_limit=64`). 68 latents → 269 frames, 17 chunks. Raw: `bench_results/vae_bench_exp9c.tsv` / `exp9d.tsv` (+ `.out` with kernel profiles). Provenance note: `exp9.tsv`/`exp9b.tsv` came from earlier script versions (one warm-up, default recompile limit, no dynamo reset) and are superseded by 9c/9d, which re-measure every lever under one protocol; the exp-9 red-team (`exp9-redteam`) found the earlier "compile +0 %" was Dynamo's recompile limit (8) being exhausted by the shared `CausalConv3d`/`RMS_norm`/`Resample` code objects → silent eager fallback, and that "fp16 autocast" left the norm/SiLU/residual path in fp32.

| Lever | s per chunk | Speedup vs fp32 1.049 | PSNR vs fp32 | values off > 4/255 |
|---|---|---|---|---|
| fp32, whole clip (pipeline as shipped) | 1.049 | 1.00× | — | — |
| bf16 autocast | 0.776 | 1.35× | 54.4 dB (max abs 0.385) | — |
| fp16 autocast (norm/SiLU/residual stay fp32, casts around every conv) | 0.776 | 1.35× | 72.2 dB | 0.0006 % |
| **true fp16 model** (`decoder.half()`, no autocast, scale constants fp32) | **0.601** | **1.74×** | 71.7 dB | 0.0008 % |
| per-chunk streaming decode (4 latents per call, persistent cache) | 1.058 | 1.00× | exact | 0 |
| `cudnn.benchmark` | 1.063 | 1.00× | exact | 0 |
| fp16 autocast + `channels_last_3d` on Conv3d | 0.724 | 1.45× | 72.2 dB | 0.0006 % (414 NCHW↔NHWC transpose kernels remain) |
| fp16 autocast + channels_last on Conv3d **and Conv2d** | 0.722 | 1.45× | 72.2 dB | 0.0006 % (0 transpose kernels) |
| fp16 autocast + `torch.compile(dynamic=True)`, recompile limit 64 | 0.629 | 1.67× | 72.2 dB | 0.0006 % |
| fp16 model + channels_last | 0.544 | 1.93× | 71.7 dB | 0.0008 % |
| fp16 model + compile | 0.593 | 1.77× | 72.0 dB | 0.0007 % |
| **fp16 model + channels_last + compile** | **0.443** | **2.37×** | 72.0 dB | 0.0007 % |
| + streaming + cudnn.benchmark (full stack) | 0.445 | 2.36× | 72.0 dB | 0.0007 % |

**Kernel profile, one fp16-model + channels_last + compiled chunk:** convolutions 0.26 s (cuDNN fp16 implicit GEMM NHWC; 52 TFLOP per chunk at ~80–85 % of the 209.5 TFLOP/s peak — the exp-9 review's FLOP check agrees), the rest fused Triton kernels (`fused_cat_constant_pad_nd_convolution`, `fused_div_linalg_vector_norm_mul`, `fused__to_copy__unsafe_index_clone`) ≈ 0.1 s, layout transposes 0. Before fusion the same chunk spent 0.41 s in ~1 900 unfused elementwise launches at 79 % of DRAM bandwidth (`bench_results/vae_roofline_exp9.txt`). 45 Dynamo recompiles in the first decode (27 `CausalConv3d`, 8 `RMS_norm`, 7 `Resample` — one per distinct layer shape), all one-time and cached by Inductor on disk.

**Reading.** The decoder is now at the roofline floor computed in §10 (conv at ~85 % of peak + fused elementwise traffic). fp16 weights are lossless for any purpose (72 dB; no NaN/overflow on this clip — the fp16 RMS-norm sum of squares over 384 channels is the one thing to watch on other content, and the pipeline's NaN check would show it). Half precision is worth 1.74× once the whole model is fp16, not 1.35×; my earlier "not GEMM-bound" reading was an artefact of autocast. Gap to the real-time budget: 0.44 s vs 0.14 s per chunk; the remaining 0.26 s of convolution is irreducible in fp16.

**Verdict: shipped as "quality mode v2" (`bench_perf.py --vae_fast`; env `LINGBOT_VAE_HALF=1 LINGBOT_VAE_CL=1 LINGBOT_VAE_COMPILE=1 LINGBOT_VAE_WARM=1`).** In the pipeline with the full DiT stack (1/2/4/5): DiT 0.861 s + VAE 0.440 s = **1.301 s per chunk, 12.16 FPS as played** (`quality2_fullstack_vaefast`), peak VRAM 26.4 GB (v1 fp16-autocast: 31.8 GB). Same-latent check through the pipeline and mp4 encode vs the fp32 reference video: PSNR 43.6 dB, SSIM 0.981, LPIPS 0.004, Laplacian 1022.8 vs 1022.3, MUSIQ 68.98 vs 68.98 (`quality_results_exp9c.tsv`; the 43 dB is the mp4 noise floor on both sides). v1 (`--vae_fp16`, fp16 autocast + channels_last, 10.02 FPS, 72 dB) stays as the fallback. Compile is a one-time ~20 s cost at first decode (warmed untimed at the first call; Inductor caches it). Videos: `videos/quality2_fullstack_vaefast.mp4`, as-played side-by-side `videos/as_played_baseline_vs_quality2.mp4`. **Eye test: pass — v1 at 10 FPS and v2 at 12.2 FPS both rated indistinguishable from the original by the user.**

## Decoder summary with the eye test (2026-09-15)

Same latents (`ex03` seed 42), Wan VAE fp32 as reference. "Eye test" is the user's verdict on the native-16-fps A/B clips, recorded so the automated metrics can be checked against a human: the Laplacian-variance column tracks the verdict (≥ ~1000 passes, ≤ ~700 fails), the no-reference scores do not (MUSIQ rates LightTAE equal to the reference; CLIP-IQA rates TAEHV above it), and full-reference PSNR/LPIPS rank Flash-VAED best while it is the softest. Laplacian + same-latent LPIPS together are the pair to trust; neither alone.

| Decoder | s / chunk | As played (full DiT stack) | PSNR / LPIPS vs fp32 VAE | Laplacian first / last | MUSIQ | Eye test |
|---|---|---|---|---|---|---|
| Wan VAE fp32 (stock) | 1.05 | 8.4 FPS | exact | 1022 / 283 | 69.1 | reference |
| Wan VAE fp16 autocast + channels_last (exp. 9, quality mode v1) | 0.72 | 10.0 FPS | 72.2 dB / — | 1022 / 283 (identical) | 69.1 | **pass** |
| Wan VAE fp16 model + channels_last + compile (exp. 9c, quality mode v2) | 0.44 | **12.2 FPS** | 72.0 dB raw; 43.6 dB / 0.004 after mp4 | 1023 / 298 (identical) | 69.0 | **pass** |
| Flash-VAED fixed (exp. 8) | 0.11 | ~17 FPS | 29.9 dB / 0.056 | 442 / 120 | 64.3 | not rated yet |
| LightTAE (exp. 6) | 0.07 | 17.7 FPS | 28.6 dB / 0.058 | 607 / 256 | 69.1 | fail — softer, judged blurrier than TAEHV |
| TAEHV (exp. 3) | 0.07 | 17.7 FPS | 29.0 dB / 0.071 | 686 / 234 | 67.6 | fail — visibly softer than the original |

## 10. Roofline of the 10 FPS quality-mode configuration (2026-09-15)

Per chunk (16 frames, 832×464): 1.578 s = DiT 0.859 s (exp. 1/2/4/5 stack, `torch.profiler` over chunks 8–10, `bench_results/dit_profile_quality_fullstack.txt`) + Wan VAE fp16 0.719 s (`vae_roofline.py` on one steady-state chunk: FLOPs from `FlopCounterMode`, DRAM bytes from a dispatch-level tracer excluding view ops — an upper bound — kernel time from the profiler; `bench_results/vae_roofline_exp9.txt`). Peaks: RTX 5090 dense fp16/bf16 with fp32 accumulate 209.5 TFLOP/s, FP8 419, INT8 838 TOPS, 1 792 GB/s.

| Bucket | s / chunk | Share of 1.578 s | Work | Achieved vs roofline | Headroom without changing the model |
|---|---|---|---|---|---|
| **VAE elementwise** (SiLU, RMSNorm, residual adds, cache `cat`/`pad`/`clone`, autocast `_to_copy`: ~1 900 launches) | 0.407 | 26 % | ≤ 580 GB DRAM traffic | ≥ 1 417 GB/s = **79 % of bandwidth** — memory-bound at roofline | only fewer bytes: fusion. Each activation is read/written ~10× per layer; fused (one read, one write per block) ≈ 0.1 s. **≈ −0.3 s** — needs a compile-friendly rewrite of the decoder (the stock one breaks graphs on its Python-list cache) or a hand-fused decoder |
| **VAE convolutions** (cuDNN fp16 implicit GEMM, NHWC) | 0.314 | 20 % | 52.2 TFLOP (3.3 TFLOP per frame) | 166 TFLOP/s = **79 % of peak**, intensity 1 331 FLOP/B (ridge 117) — compute-bound at roofline | ~0.06 s at 100 %; otherwise only fewer FLOPs (pruning = Flash-VAED, exp. 8) |
| **DiT attention** (SageAttention int8/fp8 kernels + its quant/transpose kernels) | 0.320 | 20 % | 154 TFLOP analytic (q 6 032 × kv 27 144, 30 layers × 5 forwards) | ~480 TFLOP/s effective in the attention kernel ≈ 57 % of the INT8 peak | only fewer FLOPs: smaller KV window (`local_attn_size` 18 → 12 ≈ −⅓), sparse attention (Light Forcing) — both change model behaviour |
| **DiT GEMMs** (FP8 `_scaled_mm` + remaining bf16) | 0.256 | 16 % | ~72 TFLOP analytic (the profiler's flop counter ignores `_scaled_mm`) | ~280 TFLOP/s ≈ 67 % of FP8 peak | ~0.05 s from better FP8 kernels / fusing the activation quantisation |
| **DiT elementwise / norm / copy** (fp32 modulation + residual path, unfused) | 0.196 | 12 % | memory-bound | — | ≈ −0.1 s by fixing the two graph breaks so compile fuses it (queue item 8) |
| DiT KV-cache memcpy (window shift on eviction) | 0.048 | 3 % | | | ring buffer ≈ −0.04 s (queue item 8) |
| Idle / launch gaps | ~0.04 | 2 % | | GPU busy 98 % (DiT), ~100 % (VAE) | — |

**Whole-picture reading.** Every bucket is at or near its roofline; nothing is "slow", there is simply too much work: 52 TFLOP of decoder convolution + ~226 TFLOP of DiT per second of video. The only lossless headroom is traffic, not FLOPs: VAE elementwise fusion (−0.3 s) and DiT elementwise fusion + ring buffer (−0.14 s) → ≈ 1.14 s per chunk ≈ **14 FPS**, the ceiling for the original decoder on one 5090. Real time (≤ 1.0 s) needs fewer FLOPs — a pruned decoder (Flash-VAED, 0.11 s, fails the sharpness bar) or a second GPU for the decoder, which is what LingBot, LongLive 2.0 and Matrix-Game 3.0 do.

Ranked by size × headroom: (1) VAE elementwise fusion, (2) DiT elementwise fusion + KV ring buffer, (3) FP8 quant fusion; then model-changing: KV window / sparse attention, decoder pruning.

## 11. KV-cache ring buffer — negative result as delivered (2026-09-15)

**Patch:** `patch/opt11_kv_ring.diff` / `patch/model_fast_kvring.py` (agent `kv-ring`, worktree `wt/kv-ring`; research in `RESEARCH_kv_ring.md`: the whole Self-Forcing → LongLive → Matrix-Game → LingBot lineage shifts the window with `.clone()`; the transferable pattern is Mistral's `RotatingBufferCache`). `LINGBOT_KV_RING=1` stores frame *t* at `t − window·max((t − sink) // window, 0)` — identity while filling, modulo ring afterwards — and never shifts; the read slice is unchanged. Correction from the agent: the 18-frame cache *includes* the 6-frame sink, so the rolling window is 12 frames = 3 chunks. CPU test (`tests/test_kv_ring_cpu.py`, 12 chunks × 5 forwards, fp32): outputs match to 1.2e-7, attended token sets identical every forward, Dynamo compiles 7 → 5.

**Pod:** deterministic bf16/FA2 loop, latents vs the reference dump: bit-identical for chunks 0–3, diverging from chunk 4 (first eviction) — mean 4e-3 growing to 0.2 by chunk 16. That is floating-point summation order over a permuted kv set, amplified by the autoregressive rollout: the same class as the FP8/SageAttention run-to-run divergence, not an error. Full stack timing: **steady 0.900 s vs 0.861 s — 0.039 s/chunk slower** (`opt11_kvring_fullstack`, 11.8 FPS vs 12.2). The slowdown is present before any eviction (chunks 2–3: 0.786/0.861 vs 0.733/0.809): the ring writes each chunk frame by frame, 4 copies per K/V per layer per forward (~900 extra launches per chunk), which costs more than the ~0.02 s the eviction roll was worth (the 0.048 s "memcpy" bucket in the profile includes the normal cache writes). It did remove the chunk-4 recompile (1.755 → 0.898 s once).

**Verdict: not shipped; sent back for a single-contiguous-write revision.**

**Rev 2** (`1bdfae6`: one slice write per tensor, offsets planned once per chunk on the host via `kv_ring_plan`, read slice unchanged; CPU test still exact, attended sets identical). Full stack (`opt11b_kvring_fullstack`): non-straddling chunks 0.846 s (−0.015 s vs 0.861), but every third chunk from the first eviction straddles the ring end and needs a second write: 0.896 s (+0.035) — the 3-chunk period the agent predicted (the 6-frame sink is 1.5 chunks, so chunk slots land at 3016·(2c−3) mod 18096 and one of the three positions runs past the end). Mean 0.863 s vs 0.861 s baseline; chunk-4 recompile back (2.6 s once). Ceiling if the straddling were solved (index_copy_ with a precomputed position tensor): ~15 ms/chunk, 1.7 %, against a bf16 rollout that is no longer bit-reproducible after the first eviction. **Final verdict: negative; the KV roll is not a lever worth its cost on this model.** The gated code stays in `patch/model_fast_kvring.py` (env off = the stock shift path, bit-identical; deployed on the pod because `image2video.stage8.py` imports `kv_ring_plan`).

## 12. Fused / compiled Wan VAE decoder driver — quality mode v3 (2026-09-15)

**Patch:** `patch/vae2_1_fused.py` (agent `vae-fusion`, worktree `wt/vae-fusion`; `RESEARCH_vae_fusion.md`: no one ships a compiled per-step Wan 2.1 decoder; diffusers #14771 (stateless cache → 1.45× under compile), sglang #33546/#34125 (fused RMSNorm+SiLU, one-pass cat/pad/cache) and vllm-omni #7056 (deferred bias, cuDNN padding, gather upsample, 1.84× on GB200) validate each piece, none reusable as code). `FusedDecoder` drives the stock `Decoder3d` modules functionally: the temporal cache is a list of 2-frame fp16 tensors returned per latent (first-latent 1-frame caches stored as `[0, x]` so every slot is static-shaped), `CausalConv3d` = one `cat([cache, x])` + `F.conv3d` with spatial padding on cuDNN, fp16 conv weights + channels_last without autocast, RMS_norm→SiLU as an fp32 island (free under Inductor: fp32 registers, fp16 stores), gather-only upsample; `torch.compile(dynamic=False, fullgraph=True)`, mode `max-autotune-no-cudagraphs`; ~77 launches per latent vs ~475. Streaming API `decode_step(z, state)`. Env `LINGBOT_VAE_FUSED=1|eager|<mode>`; `bench_perf.py --vae_fused`. CPU test `tests/test_vae_fused_cpu.py`: bit-exact vs the stock loop in eager and streaming, 3e-6 compiled, 0 graph breaks.

**Decoder micro-bench** (`bench_results/vae_bench_exp12.tsv`, same protocol as §9):

| Variant | s per chunk | vs fp32 | PSNR vs fp32 | first call |
|---|---|---|---|---|
| fp32 | 1.050 | 1.00× | — | |
| fp16 model + channels_last + compile (§9c best) | 0.444 | 2.36× | 72.0 dB | 21 s |
| fused, eager | 0.623 | 1.69× | 72.2 dB | 11 s |
| **fused, compiled (max-autotune)** | **0.351** | **2.99×** | **72.6 dB** | 117 s cold (Inductor disk cache after) |

**Two integration bugs found by the same-latent check, both fixed** (`quality_results_exp12.tsv`): (1) `--vae_fused` as delivered built `Wan2_1_VAE(dtype=fp16)`, which also ran the VAE *encoder* (start-image conditioning) in fp16 — different conditioning latents, DiT rollout diverging from chunk 0, 28 dB whole clip while the decoder alone was 72.6 dB. `FusedDecoder` now takes its own `dtype`; the VAE object and encoder stay fp32. The same applies to quality mode v1 (`--vae_fp16`, `LINGBOT_VAE_DTYPE=fp16`): its encoder was fp16; v2/v3 keep it fp32. (2) `decode_step` ran inside the pipeline's outer bf16 autocast; now guarded. Lesson recorded in the routine: a decoder swap must be checked on **dumped latents** (are they identical?) before frames.

**Pipeline:** DiT 0.862 s + VAE 0.350 s = **1.212 s per chunk, 13.06 FPS as played** (`quality3_fullstack_vaefused`); same-latent through the pipeline vs the fp32 reference video: PSNR 43.56 dB, SSIM 0.981, LPIPS 0.0037, Laplacian 1022.4 / 297.7 vs 1022.3 / 298.3, MUSIQ 68.99 vs 68.98 — identical to v2's numbers. Peak VRAM 30.7 GB with the whole-clip decode (v2: 26.4 GB) — the streaming `decode_step` is the fix if a longer clip pushes past 32 GB. Videos: `videos/quality3_fullstack_vaefused.mp4`, `videos/as_played_baseline_vs_quality3.mp4`.

**Verdict: shipped as quality mode v3.** The decoder is at 0.35 s/chunk against a 0.26 s pure-convolution floor; what is left is CUDA-graphing the steady step (est. ≤ 0.02 s) and the convolutions themselves.

## 10b. DiT graph-break removal + exact per-chunk caches (exp. 10, 2026-09-15)

**Patch:** `patch/model_fast.py` (deployed as `wan/modules/model_fast_fusion.py`), `patch/opt10_dit_fusion.diff`, gate `LINGBOT_DIT_FUSION=1` in `patch/image2video.stage8.py` (+ `LINGBOT_TORCH_COMPILE=regional` fallback). Agent `dit-fusion`, worktree `wt/dit-fusion`, `RESEARCH_dit_fusion.md`, `REPORT_dit_fusion.md`. Changes: grid sizes / seq lens as Python ints (no `.tolist()`/`.item()`/tensor-to-bool breaks); RoPE as a real `(cos, sin)` float64 table built once per forward; time embedding computed on one row `[B,1,6,C]` and broadcast; cross-attn K/V filled once per generation (`init_crossattn_cache`); camera-MLP scale/shift cached per chunk (`cam_cache`, +1.1 GB in the loop, freed before decode); KV eviction code untouched. CPU test (`tests/test_dit_fusion_cpu.py`): 20 forwards `allclose(1e-6)` vs upstream (max 1.8e-7; bitwise except the 1-row time MLP), `dynamo.explain` 13 graphs / 12 breaks per forward → 1 / 0; compiled loop = 3 unique graphs (append / overwrite / evict), no later recompiles.

**Pod.** Eager bf16/FA2 loop: 1.617 → 1.543 s/chunk. **Full stack (compile + FP8 + Sage) + fused VAE (exp. 12): DiT 0.862 → 0.662 s/chunk (−0.20 s, −23 %); 0.662 + 0.350 = 1.012 s per chunk, 15.64 FPS as played with the original decoder** (`opt10_ditfusion_fullstack`; 9 recompile lines vs 43, steady chunks flat at 0.662, peak VRAM 30.7 GB, first chunk 37 s compile, chunk 4 6.1 s = the eviction-variant compile).

**Exactness — resolved.** Dumped latents on the deterministic eager loop with `LINGBOT_DIT_FUSION_EXACT_T=1` (time-embedding MLP on the L pre-expanded rows, as upstream): **bit-identical to the reference over the whole 68-latent rollout** — which verifies every other change (RoPE real-pair table on CUDA, cam-MLP cache, cross-attn cache, int `t`, graph-break removal) bitwise. The default 1-row time MLP is the only deviation: same dot products through a different cuBLAS kernel (~1 fp32 ulp in `e`), which in bf16 seeds a rollout divergence like a different seed (chunk 0 max |Δ| 0.14). Pixel space, first chunk, same fp32 decoder: PSNR 41.8 dB, SSIM 0.981, LPIPS 0.003 vs the reference video (mp4 floor ≈ 43.5 dB); MUSIQ 69.10 vs 68.98, Laplacian 1022 vs 1022.

| DiT fusion variant | DiT s/chunk | + fused VAE 0.350 | As played | Numerics |
|---|---|---|---|---|
| `EXACT_T=1` | 0.740 | 1.090 | **14.51 FPS** | bit-identical latents |
| default (1-row time MLP, broadcast `e`) | 0.662 | 1.012 | **15.64 FPS** | ~1 fp32 ulp in the modulation vector; fp-order class |

The 1-row form is worth 0.078 s/chunk because Inductor then fuses the modulation path against a broadcast `e` instead of streaming the per-token `[B,L,6,C]` `e` (111 MB per forward) through all 30 layers. Both variants: 9 recompile lines (3 unique graphs: append / overwrite / evict), steady chunks flat, peak VRAM 30.7 GB, first chunk ~32–37 s compile (cached afterwards), chunk 4 ~3–6 s (eviction-variant compile).

**Verdict: shipped.** Default = 1-row (15.64 FPS; FP8 and SageAttention already in the stack are far larger deviations); `LINGBOT_DIT_FUSION_EXACT_T=1` is the bit-exact mode (14.51 FPS) for verification or a strict-lossless deployment. Videos: `videos/opt10_ditfusion_fullstack.mp4`, `videos/as_played_baseline_vs_quality4.mp4`; exact variant `videos/opt10_ditfusion_exactT_fullstack.mp4`.

## Final stack (2026-09-15) — original decoder, lossless

| Configuration | DiT + VAE s/chunk | As played | Numerics vs reference | Eye test |
|---|---|---|---|---|
| Baseline: stock repo, fp32 Wan VAE (BENCHMARK.md run B) | 2.87 + 1.05 = 3.9 | 5.5 FPS | reference | reference |
| Exp. 1/2/4/5 DiT stack + stock fp32 VAE | 0.86 + 1.05 = 1.91 | 8.4 FPS | FP8/Sage: near-lossless (§4, §5) | — |
| + quality mode v1 (fp16 autocast + channels_last, exp. 9) | 0.86 + 0.72 = 1.58 | 10.0 FPS | 72 dB decoder (encoder also fp16 — see exp. 12 note) | pass |
| + quality mode v2 (fp16 weights + channels_last + compile, exp. 9c) | 0.86 + 0.44 = 1.30 | 12.2 FPS | 72 dB; latents identical | pass |
| + quality mode v3 (fused compiled VAE driver, exp. 12) | 0.86 + 0.35 = 1.21 | 13.1 FPS | 72.6 dB; latents identical | clip rendered |
| + DiT fusion, bit-exact mode (exp. 10, `LINGBOT_DIT_FUSION_EXACT_T=1`) | 0.74 + 0.35 = 1.09 | **14.5 FPS** | **latents bit-identical, whole rollout** | — |
| **+ DiT fusion, default (1-row time MLP)** | **0.66 + 0.35 = 1.01** | **15.6 FPS** | ~1 fp32 ulp in `e`; first chunk 41.8 dB / LPIPS 0.003, MUSIQ & Laplacian identical | clip rendered |
| (play mode: same DiT + LightTAE tiny decoder, exp. 6) | 0.66 + 0.07 ≈ 0.73 | est. ~22 FPS, not re-measured | tiny decoder: fails the eye test | fail |

Command for the final lossless configuration on the pod (`/workspace/lingbot-world-v2`, `model_fast_fusion.py` and `vae2_1_fused.py` installed under `wan/modules/`):

```
LINGBOT_DIT_FUSION=1 TORCHINDUCTOR_CACHE_DIR=/workspace/torchinductor_cache \
python3 bench_perf.py --example 03 --frame_num 361 --no_offload --compile 1 --fp8 --attn sage --vae_fused --label final
# add LINGBOT_DIT_FUSION_EXACT_T=1 for the bit-exact variant
```

Negative or neutral results, kept for the record: exp. 7 CAS sharpening (adds contrast, not detail), exp. 8 Flash-VAED (fidelity leader among fast decoders after fixing its cache reset, but the softest — by design, no GAN loss), exp. 11 KV ring buffer (two revisions, −15/+35 ms per chunk, net zero), TensorRT / second-stream VAE (bounded below the budget by the roofline, not run).

What remains, all model-changing or hardware: KV window 18 → 12 or sparse attention (Light Forcing) on the DiT; a second GPU for the decoder (LingBot v2, LongLive 2.0, Matrix-Game 3.0); a GAN-finished fast decoder trained on our own latents (data plan in QUALITY.md / the literature reports).

## 13. Global + local profile of the final lossless stack (pod 4, 2026-09-15)

**Setup.** Pod 4 (`psd0peal0hkuo7`, RTX 5090, provisioned from scratch with `provision_pod.sh` — the final stack reproduces: steady DiT chunks 0.666–0.674 s vs 0.662 on pod 3). `torch.profiler` (CPU + CUDA) over chunks 8–10 of the DiT loop with `record_function` phase markers (`prep`, `denoise_step0-3`, `cache_write`, `bench_sync`) plus a second window over the whole-clip VAE decode (`LINGBOT_PROFILE_VAE=1`); `analyze_trace.py` attributes every kernel to its phase (via launch correlation) and category, and measures GPU idle gaps and host syncs. Raw: `bench_results/profile_final_stack_pod4.txt`, `profile_final_exactT_pod4.txt`; traces in `profiles/pod4_*` (not committed, 7 MB). Peaks used: RTX 5090 fp16/bf16 dense 209.5 TFLOP/s (fp32 acc), FP8 419 (fp32 acc) / 838 (fp16 acc), INT8 838 TOPS, DRAM 1 792 GB/s.

### Global: one 16-frame chunk = 1.028 s (DiT 0.672 + VAE 0.356)

| Bucket | s / chunk | share of 1.028 | launches / chunk | work → achieved | vs roofline |
|---|---|---|---|---|---|
| **VAE convolutions** (cuDNN fp16 NHWC xmma 0.129 + Inductor Triton conv templates 0.172 — max-autotune picked Triton for the big layers; corrected after the audit from 0.315) | **0.301** | 29 % | ~150 | 52.2 TFLOP → 173 TFLOP/s | **83 % of fp16 peak**, compute-bound |
| **DiT SageAttention** kernel (`qk_int_sv_f8_attn_kernel`, int8 QK · fp8 PV) | **0.275** | 27 % | 150 | 151 TFLOP analytic → 549 TFLOP/s | above the 419 fp32-accumulate FP8 peak, 65 % of the 838 8-bit peak — **at the kernel's ceiling**; only fewer FLOPs help |
| **DiT FP8 GEMMs** (`_scaled_mm`, CUTLASS sm100 kernels) | **0.202** | 20 % | 1 320 | 75 TFLOP → 373 TFLOP/s | **89 % of the 419 FP8 peak** |
| **DiT fused elementwise** (Inductor Triton: RMSNorm/modulation/residual/RoPE/FP8 activation quant) | 0.108 | 10.5 % | 2 860 (38 µs avg) | memory- and launch-bound | effective bandwidth ≈ 20–30 % — the least efficient bucket; ~0.03 of it is the FP8 activation-quant reductions (`fused__scaled_mm_…abs_amax_clamp_div`), ~0.03 the q/k RMSNorm reductions (`fused_…mean_pow_stack`) |
| **DiT Sage per-call re-quantisation** (`TransposePadPermute` 0.020, `MeanScale` 0.010, `QuantInt8` 0.010) — K/V of the whole 27 144-token cache re-quantised on every forward | 0.040 | 4 % | 600 | ~37 GB/chunk → ~50 % of DRAM BW | **measured ceiling of the "quantize-once KV" hypothesis** |
| **VAE fused elementwise** (Triton: cat/pad/cache, norm+SiLU, clone) | 0.049 (corrected after the audit from 0.035: Triton norm/SiLU kernels named `…convolution…` had been counted as conv) | 4.8 % | ~400 | | fused; ≤ 0.02 left |
| DiT GPU idle (launch gaps; 193 gaps ≥ 20 µs per chunk = 0.015 s, largest 6 ms once) | 0.026 | 2.5 % | | | **measured ceiling of the CUDA-graph / launch hypothesis** |
| DiT cross-attention (still FA2 bf16 on 512 text tokens) | 0.017 | 1.7 % | 150 | | small; Sage or caching of the text K/V would give ≤ 0.01 |
| VAE attention (mid-block, mem-efficient SDPA) + VAE idle | 0.005 + 0.001 | 0.6 % | | | — |
| memcpy/memset, bf16 GEMMs, unfused elementwise, other | 0.004 | 0.4 % | ~1 500 | | — |

GPU busy: DiT window 96.1 %, VAE window 99.7 %. Compute buckets (VAE conv + attention + FP8 GEMM) = 0.79 s = 77 % of the chunk, all at 79–89 % of their peaks (attention at its kernel ceiling). Memory/launch buckets = 0.18 s = 18 %. Idle 3 %.

### Local: inside the DiT chunk (0.672 s wall, 0.646 s GPU)

| Phase | GPU s / chunk | what it is |
|---|---|---|
| `prep` (poses, kwargs) | ~0 | host-side only |
| `denoise_step0` | 0.115 | one forward + cam-MLP (first call, then cached): +0.010 over the other steps |
| `denoise_step1`, `2`, `3` | 0.106 each | one forward each: attention 0.055, FP8 GEMM 0.039, re-quant 0.008, FA2 cross-attn 0.003 |
| `cache_write` | 0.106 | the 5th forward at t = 0 whose only output is the KV cache — **same cost as a denoise step, 20 % of the DiT**. Its last layer's attention output/FFN/head are discarded: skipping them is worth ~1/30 ≈ 0.004 s (measured share), not more |
| Inductor fused kernels (no phase — launched from the compiled graph) | 0.108 | spread over the five forwards, ≈ 0.022 each |
| `bench_sync` | 0 GPU | one `cudaDeviceSynchronize` per chunk from the timing instrumentation |

Per forward ≈ 0.129 s all-in; five forwards per chunk. Five forwards exist because the 4-step distilled sampler needs four and the KV cache write needs one more at t = 0 — a model-design cost, not an implementation one.

### Round-trip / host side

- **37 host syncs per chunk**: 25 `cudaStreamSynchronize` from `aten::to` (the scheduler's *CPU* timestep tensors are copied to the GPU every forward — a pageable H2D copy synchronises the stream; 5 per forward), 7 from `.item()` on the timestep (`_convert_flow_pred_to_x0` / `add_noise`), 3 from `nonzero`, 1–2 from the bench timing / profiler. CPU blocked in those waits: **~105 ms per chunk**. They do not cost GPU time directly (the queue is drained, not emptied, and the GPU is 96 % busy) but they stop the CPU from running ahead of the GPU, which is what caps the idle at 0.026 s and what CUDA graphs / stream pipelining would need. Fix (lossless, ~20 lines): keep the timesteps and sigmas on the GPU, index them by step, no `.item()`. Measured ceiling = the 0.026 s idle.
- The DiT loop launches ~6 700 kernels per chunk; the VAE ~600 per chunk. Average kernel 96 µs (DiT), the fused-elementwise ones 38 µs: launch overhead (~5 µs each) is ≈ 0.03 s per chunk of the GPU timeline — consistent with the measured idle.
- Non-GPU work per chunk (pose interpolation, Plücker embedding on the cached path, scheduler arithmetic, Python) is inside the same wall time and hidden behind the GPU queue except at the syncs above; video encoding happens once after the loop and is not part of the chunk.

### Bit-exact variant (`LINGBOT_DIT_FUSION_EXACT_T=1`), same profile

DiT 0.750 s wall / 0.732 s GPU. The *only* categories that move: Triton fused 0.108 → 0.176 s (+0.068: the per-token `[B, L, 6, C]` modulation tensor `e` streamed through every layer's fused kernels) and bf16 GEMM 0.001 → 0.020 s (+0.019: the time-embedding MLP on L rows). Attention 0.276, FP8 GEMM 0.201, re-quant 0.040: identical. So the 1-row time MLP is 0.078 s/chunk of pure traffic, nothing else.

### Hypotheses, now measured (per chunk, lossless)

| # | Lever | Measured ceiling | Note |
|---|---|---|---|
| 1 | Sage quantize-once KV cache (store int8 K / fp8 V + scales per chunk instead of re-quantising 27 144 tokens × 30 layers × 5 forwards) | **0.040 s** (0.030 s realistic: Q must still be quantised per call) | 4 % of the chunk; the largest lever left |
| 2 | Remove the host syncs (GPU-resident timesteps) + CUDA-graph the steady forward | **0.026 s** (the whole idle) | the sync fix is a prerequisite for graphs and for any CPU run-ahead |
| 3 | FP8 activation-quant fused into the GEMM (cuBLASLt amax epilogue / torchao) | **≤ 0.030 s** | the `fused__scaled_mm_…amax` reductions |
| 4 | q/k RMSNorm reduction kernels (`fused_…mean_pow_stack`, 0.030 s) → fold into the QKV GEMM epilogue or a single fused norm | ≤ 0.02 s | |
| 5 | Cross-attention on Sage / cached text K/V | ≤ 0.010 s | |
| 6 | Cache-write forward tail skip (last layer's attention output, FFN, head) | ~0.004 s | not worth the code |
| 7 | VAE: remaining fused elementwise + CUDA graphs | ≤ 0.02 s | conv is at 79 % of peak: Inductor's Triton conv templates vs cuDNN is worth one autotune comparison (`max-autotune` vs cuDNN-only), ± 0.02 |

Sum of the measurable lossless headroom: **≈ 0.10–0.13 s per chunk → 0.90–0.93 s ≈ 17–18 FPS** with the original decoder. Everything larger is model-side: the fifth forward (0.13 s), the KV window (attention 0.275 s scales with it), and the VAE's 52 TFLOP of convolution.

## 14. Research pass on the measured bottlenecks + pod-4 checks (2026-09-15)

**14a. cuDNN 9.26 drop-in for the VAE (from `RESEARCH_vae_conv_blackwell.md`).** Consumer Blackwell (sm_120) has no tcgen05/wgmma: every fp16 conv path (cuDNN "sm80" xmma, Triton MMA v2) uses the same `mma.sync` instruction, so the 79 %-of-peak convolution has ≤ 0.06 s of kernel-tuning headroom and none from a new tensor-core generation. Tested `nvidia-cudnn-cu12==9.26.0.51` on torch 2.8 (loads fine, `cudnn.version()` 92600; pip warns about the pin): fused compiled decoder 0.351 (pod 3, 9.10) → **0.342 s/chunk on pod 4 with 9.26 (−0.009; cross-pod, within the ~2 % host variance — needs a same-pod control)**, fp16-model+compile 0.444 → 0.416, eager fp16 0.544 → 0.518, 72.6 dB unchanged (`bench_results/vae_bench_cudnn926.tsv`, fresh Inductor cache). **Side effect found: the cuDNN version changes the DiT rollout** — the VAE *encoder* (conditioning latents) and the DiT patch-embedding conv run on cuDNN, so the bf16 loop's latents differ from pod 3 under 9.26 (frame diff mean 5/255, growing) while under 9.10 they match (frame diff flat at ~1/255 = x264 encoder noise across machines). Reverted to 9.10.2.21 for the rest of the series; 9.26 is a −0.01 s option to take at the end with a fresh reference. Lesson: a library upgrade is a kernel change for the rollout — run it through the dumped-latents check.

**14b. fp16-accumulate convolutions: not reachable from PyTorch — no kernel to trade (2026-09-22, CPU-only research pass, no GPU).** The dead-ends line's "fp16-accumulate convs (−0.13 s)" estimate came from §19 H5 (fp32 PV accumulate cost the SageAttention kernel +14 % on this GeForce part) and was never checked against the decoder's own cuDNN fp16 convolutions (0.30 s of the 0.35 s decoder, §12). Question: is there a knob analogous to `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction`, but for `aten::conv2d`/`conv3d`?

**Finding: no such knob exists.** PyTorch's cuDNN convolution path hardcodes the accumulator to fp32 for half/bfloat16 inputs, unconditionally. `aten/src/ATen/native/cudnn/Conv_v8.cpp::getConvDescriptor()`:
```cpp
if (scalar_type == kBFloat16 || scalar_type == kHalf) {
  dataType = CUDNN_DATA_FLOAT;
}
```
and the same file's `filterEngineConfigs()` forces every fused pointwise op (bias-add, activation) to `CUDNN_DATA_FLOAT` too, commented "need computation to be done in FLOAT type regardless of reduced precision input." `torch.backends.cudnn` exposes exactly four settings (`benchmark`, `benchmark_limit`, `allow_tf32`, `fp32_precision`; `torch/backends/cudnn/__init__.py`) and none reach this: `allow_tf32`/`fp32_precision` govern TF32 mantissa truncation for **fp32** inputs (our conv inputs are already fp16), and `benchmark` only picks the fastest cuDNN engine config among those on offer — all of which share the forced-fp32 compute type baked into the descriptor before engine search runs. `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction` and the newer `allow_fp16_accumulation` (`torch/backends/cuda/__init__.py`) are real flags, but both bind to `torch._C._get/set_cublas_*` — cuBLAS GEMM only, never in the dispatch path for convolution. **PyTorch exposes accumulate-precision control for matmul, not for convolution — exactly the caveat this doc's dead-ends line implied but never confirmed.**

**Alternatives checked, each a dead end for this candidate:** Inductor's Triton conv template (`torch/_inductor/kernel/conv.py`) also hardcodes `acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)` — it's opt-in (needs `max-autotune` with the Triton conv backend enabled, off by default) and offers no fp16-accumulate option even when on. The one real route is the standalone `nvidia-cudnn-frontend` PyPI package (the public cuDNN Graph API, separate from PyTorch's internal use of the same library): `Conv_fprop_attributes.set_compute_data_type(HALF)` is a genuine, exposed knob there. Reaching it means bypassing `aten::conv3d` entirely — hand-building the decoder's causal/channels-last conv graph in the frontend's graph API, owning engine-config selection and workspace allocation, and handing it PyTorch's CUDA tensors via `__cuda_array_interface__`/DLPack. That is a GPU-side kernel-integration project, not a flag: it cannot be written correctly blind on a CPU-only box, and §9's fp16 RMS-norm overflow hazard (sum of squares over 384 channels) only gets worse under fp16 accumulate, so it would still need the full dumped-latent gate before shipping. **Not attempted — scoped out as its own follow-up, not a same-day candidate.**

**Verdict: dead end at the PyTorch level; no code shipped.** No source file touched outside this note — an env-gated `LINGBOT_VAE_FP16_ACCUM=1` flag would either be a no-op (nothing in the fp16 conv path reads an accumulate-precision setting) or require the unverified cuDNN-frontend rewrite above; neither belongs in a one-line env-gated patch. The −0.13 s dead-ends estimate stands as unreachable, not merely undone.

## 15. Implementation pass on the profiled bottlenecks — measured on pod 5 (2026-09-15/16)

Pod 5 (`dwuu0l5nbseicm`) provisioned from scratch; its reference latents/video are bit-identical to pod 3's. Baseline on this host: DiT 0.666 s + VAE 0.348 s = **1.014 s/chunk, 15.6 FPS** (`base_pod5`). Each lever: dumped-latent exactness on the deterministic loop against the bit-exact (`EXACT_T=1`) baseline, then full-stack timing (`--compile 1 --fp8 --attn sage --vae_fused`, `LINGBOT_DIT_FUSION=1`), then same-latent quality. Agents implemented in worktrees with CPU exactness tests; all GPU numbers below are mine.

| # | Lever (env) | Agent / branch | Latents | DiT s/chunk (Δ vs 0.666) | Predicted | Verdict |
|---|---|---|---|---|---|---|
| 15a | Sync-free sampling loop (`LINGBOT_SYNCFREE=1`): GPU-resident timesteps/sigmas, no `.to()`/`.item()`/`nonzero` per forward | `dit-elementwise` / `wt/dit-elementwise` | **bit-identical** | 0.662 (−0.004) | −0.010–0.015 | keep: host syncs 110 → 2 per 3 chunks, GPU idle 0.026 → 0.013 s/chunk (profile `bench_results/profile_all3_pod5.txt`); the saving is small because the GPU was already 96 % busy |
| 15b | RoPE in compensated fp32 instead of fp64 (`LINGBOT_DIT_FUSION_ROPE=fp32c`) | same | **bit-identical** (0 bf16 flips) | **0.690 (+0.024)** alone | −0.018–0.020 | **regression alone**: the Veltkamp/Dekker chain broke Inductor's q/k-RMSNorm+RoPE fusion — one `red_fused…mean_pow_stack` kernel (72+17 ms/3 chunks) became three (`poi_fused_add_mul_neg_sub` 67 ms + two copies 28+25 ms). The fp64 hypothesis was wrong: that kernel is memory-bound, not fp64-throughput-bound (`profile_rope32c_pod5.txt`). Red-teamed: the profile settled it (fusion break; the tuner's 22.5 ms config restores the win) — the separate red-team agent was lost to network drops and produced no report |
| 15c | Inductor knobs (`LINGBOT_INDUCTOR_TUNE=1`: `triton.multi_kernel=1`, `coordinate_descent_tuning`, `realize_reads_threshold=8`) | same | bit-identical | 0.664 (−0.002) | −0.005–0.010 | keep only with 15b (below) |
| **15a+b+c** | all three | | **bit-identical** (eager bf16/FA2 loop; the compiled Triton kernel for 15b uses FMA contraction — compiled-loop exactness check queued, audit M1) | **0.645 (−0.021), 15.95 FPS** | −0.035–0.045 | **keep**: the tuner finds a 22.5 ms config for the fp32c apply kernel (vs 67 ms untuned) and turns the amax reductions persistent; fused elementwise 0.108 → 0.092 s/chunk |
| 15d | Pre-quantised int8-K/fp8-V cache for SageAttention (`LINGBOT_ATTN=sage_kvq`) | `sage-kvq` / `wt/sage-kvq` | numerics OK (cos 0.99924 vs fp32 SDPA, same as Sage; frozen per-chunk K-mean and V-scale) | **+0.015 s/chunk (probe, compiled)**; +0.133 eager | −0.02–0.03 | **negative**: Sage's own CUDA quant of the full 27 144-token K/V costs only 0.29 ms per call (2.146 vs 1.852 ms kernel-only = 0.044 s/chunk), while the torch/Inductor quantisation of just the new chunk costs 0.21 ms per forward plus `requant_all` 0.88 ms per layer per chunk (~8× the bandwidth floor). `bench_results/probe_sage_kvq_pod5.out`. Red-teamed (§15e) |
| 15e-VAE | Exact sub-pixel rewrite of the 3 upsample convs (`LINGBOT_VAE_SUBPIXEL=1`; 2×2 conv with 4·Cout + pixel shuffle; −2.54 of 52 TFLOP) | `vae-subpixel` / `wt/vae-subpixel` | decoder 72.55 dB vs 72.64 (one fp16 rounding of summed taps) | VAE 0.349 → **0.343 (−0.006)** | −0.015–0.020 | **keep, below prediction**: the upsample-stage convs did get cheaper (chunk-0 profile: 34.6 → 29.1 ms, 14.6 → 5.4 ms) but the `[4·Cout, Cin, 2, 2]` shape runs at lower tensor-core efficiency than the 3×3 and a `stack` kernel (4.3 ms) assembles the phases. `bench_results/vae_bench_subpixel.tsv` |
| 14a | cuDNN 9.26 (VAE −0.009 s) | — | perturbs the DiT rollout via the encoder | not stacked (kept 9.10 for comparability) | | optional last step with a fresh reference |

**All four together (`opt15_all4`): DiT 0.640 + VAE 0.336 = 0.976 s per chunk, 16.21 FPS as played — real time with the original decoder.** Same-latent through the pipeline (`q_all4_samelatent`, `quality_results_exp15.tsv`): PSNR 43.56 dB, SSIM 0.981, LPIPS 0.0037, Laplacian 1023.0 / 298.0 vs 1022.3 / 298.3, MUSIQ 68.99 vs 68.98, flicker identical — indistinguishable from quality mode v3. Peak VRAM 29.6 GB.

Environment for the final configuration:
```
LINGBOT_DIT_FUSION=1 LINGBOT_SYNCFREE=1 LINGBOT_DIT_FUSION_ROPE=fp32c LINGBOT_INDUCTOR_TUNE=1 LINGBOT_VAE_SUBPIXEL=1 \
TORCHINDUCTOR_CACHE_DIR=/workspace/torchinductor_cache \
python3 bench_perf.py --example 03 --frame_num 361 --no_offload --compile 1 --fp8 --attn sage --vae_fused --label final
# + LINGBOT_DIT_FUSION_EXACT_T=1 for the bit-identical variant (~0.72 s DiT)
```

## Final scoreboard (2026-09-16) — original decoder, measured as played on the pod

| Configuration | DiT + VAE s/chunk | FPS as played | Numerics vs stock | Eye test |
|---|---|---|---|---|
| Baseline: stock repo, fp32 Wan VAE (BENCHMARK.md run B, measured) | 2.87 + 1.05 = 3.9 | 5.7 | reference | reference |
| Quality mode v3: exp 1/2/4/5 DiT + fused compiled VAE (exp 12) | 0.86 + 0.35 = 1.21 | 13.1 | latents identical; decoder 72.6 dB | — |
| + DiT fusion (exp 10) | 0.66 + 0.35 = 1.01 | 15.6 | 1-row time MLP: ~1 fp32 ulp | pass (user, 15.6 clip) |
| + sync-free loop, fp32c RoPE, Inductor tune, sub-pixel VAE (exp 15), `EXACT_T=1` | 0.73 + 0.34 = 1.07 | **14.8** (`opt15_all4_exactT`: run log only — the TSV row was not retrieved before pod 5's host was lost) | DiT patches bit-identical to stock *at bf16/FA2*; the FP8 + Sage stack is the near-lossless kernel swap of exp 4/5 (first-chunk LPIPS +0.005); decoder 72.55 dB | — |
| **+ same, default DiT** (`opt15_all4`) | **0.64 + 0.34 = 0.98** | **16.2** | as above + the 1-row time MLP (~1 fp32 ulp); decoder same-latent 43.6 dB / LPIPS 0.004 / Laplacian & MUSIQ identical | clip `as_played_baseline_vs_quality5.mp4` |

From 5.7 to 16.2 FPS (2.84×) with no model change: real time on one RTX 5090 with the full-quality decoder. What "lossless" means here, precisely: the decoder chain is measured against the fp32 decoder on identical latents (72.55 dB raw; 43.6 dB / LPIPS 0.004 after mp4, the encoder floor); the DiT changes of exp 10/15 are bit-identical to the stock model on the deterministic bf16/FA2 loop (`EXACT_T=1`); the FP8 GEMMs (exp 4) and SageAttention (exp 5) are kernel swaps validated as near-lossless (first-chunk LPIPS +0.005 vs bf16, MUSIQ/Laplacian unchanged), not bit-identical. Every step measured on the pod with the same protocol (dumped-latent exactness → timing → same-latent quality → eye test). Audit of the claims and raw data: `REDTEAM_final.md`.

**Dead ends and parked items, all measured:** KV ring buffer (exp 11, neutral), Sage pre-quantised KV cache (exp 15d, +0.015 s as torch; Triton redesign ceiling −0.026 s, parked), CAS sharpening (exp 7), Flash-VAED (exp 8, softness by design), tiny decoders (fail the eye test), CUDA graphs (research: ≤ 0.011 s, torch 2.8 re-records on the int cache indices), TensorRT / second-stream VAE (bounded below the budget by the roofline), cuDNN 9.26 (−0.009 s, perturbs the rollout through the encoder; optional last step). Not lossless, not done: SageAttention3 FP4 (attention −0.12 s), fp16-accumulate FP8 GEMMs (−0.09 s), fp16-accumulate convs (−0.13 s estimate; §14b: no PyTorch/cuDNN knob reaches this, confirmed unreachable not merely untried), smaller KV window / sparse attention, second GPU for the decoder.

**What is left, lossless:** ≈ 0.03–0.05 s/chunk in total — Sage quant in Triton (−0.026), remaining launch gaps (0.013 s idle), cuDNN 9.26 (−0.009). The chunk is now 0.64 s of DiT (attention 0.27 at the kernel's ceiling, FP8 GEMMs 0.20 at 89 % of peak, fused elementwise 0.09, Sage re-quant 0.04, idle 0.01) + 0.34 s of VAE (convolution 0.30 at ~80 % of peak). Beyond that the hardware roofline for this op graph (~0.70 s) needs fewer FLOPs, i.e. model-side changes.

## Audit (2026-09-16) — `REDTEAM_final.md`

Independent review of the shipped patches and of every quoted number against the raw TSVs/profiles. No code defect that changes numerics or moves work out of a timed window; timed windows are identical across configurations. Corrections applied above: the scoreboard's "bit-identical / identical quality" wording now states what was measured at which precision (H1); the 14.8 FPS `opt15_all4_exactT` row is marked run-log-only (H2); baseline 5.7 not 5.5 (M2); VAE buckets re-summed with the fixed analyzer (M5: conv 0.301, fused 0.049 s/chunk); cuDNN 9.26 marked cross-pod (M6); pod-4 rows restored as `bench_results/results_pod4.tsv` (M3); `bench_perf.py` now records the `LINGBOT_*` flags in each TSV row (M4); the latent dump moved outside the timed decode. Open item: exactness of 15b/15c under `torch.compile` (FMA contraction in the compiled RoPE kernel) — one compiled bf16/FA2 dumped-latent run, queued for the next pod.

## 16. Batched serving: DiT cost per chunk at batch 1 / 2 / 4 — measured on pod 6 (2026-09-16)

Question: how many concurrent streams can one RTX 5090 serve, and does batching buy throughput? The §10/§13 rooflines said no (the DiT is compute-bound at batch 1: 6 032 query tokens per forward, attention at the SageAttention kernel ceiling, FP8 GEMMs at 89 % of peak). This section measures it.

**Method.** `LINGBOT_BATCH=B` (env-gated, `patch/image2video.stage8.py`, `_generate_causal_fast` only) replicates the single stream B times along the batch dimension: `x`, `y`, `context` lists of length B, KV / cross-attn / cam caches allocated with `batch_size = B`; the camera embedding stays `[1, L, C]` and broadcasts. Identical inputs, so the result of stream 0 is unchanged; the cost is exactly that of B independent streams sharing every kernel launch. Final lossless stack (`LINGBOT_DIT_FUSION=1 LINGBOT_SYNCFREE=1 LINGBOT_DIT_FUSION_ROPE=fp32c LINGBOT_INDUCTOR_TUNE=1 LINGBOT_VAE_SUBPIXEL=1`, `--compile 1 --fp8 --attn sage --vae_fused`), `ex03`, `--frame_num 193` = 12 chunks, steady state = chunks 8–12 (per-chunk time is flat from chunk 5 on; spread within 0.006 s). Pod 6 `wa28d41rjz0dho` (Community Cloud RTX 5090, US; pod 4's host and its replacement on the same host both had a dead driver, `cuInit` 999). Logs: `bench_results/batch_pod6/batch{1,2,4}.log`.

| B (streams) | DiT s/chunk, steady median | per stream | vs B× linear | aggregate DiT frames/s per GPU |
|---|---|---|---|---|
| 1 | 0.633 | 0.633 | — | 25.3 |
| 2 | 1.198 | 0.599 | 0.946× | 26.7 |
| 4 | 2.416 | 0.604 | 0.954× | 26.5 |

**Reading.** Time per chunk is linear in B to within 5 %: batching two streams costs 1.89× one stream, four cost 3.82×. The 5 % is the launch/elementwise overhead that a second stream amortises (the ~13 ms of idle gaps and the fixed per-launch cost from §13), and it is fully collected at B=2 — B=4 gains nothing over B=2. There is no memory-bound regime to exploit: one stream already saturates the tensor cores, exactly as the roofline said. The aggregate output of the GPU is the constant that matters: ≈ 26 DiT frames/s, ≈ 17 frames/s once the decoder is included (16 frames per 0.976 s at B=1; 32 per 1.88 s at B=2).

**Memory.** Each stream adds 5.0 GB of KV cache (27 144 tokens × 12 heads × 128 × bf16 × K,V × 30 layers). B=4 ran the whole 12-chunk DiT loop (27.3 GB allocated) and OOM'd only in the whole-clip VAE decode afterwards, with the four KV caches still resident. In a streaming server the decoder runs per chunk on the same GPU, so B=4 does not fit on 32 GB with the full decoder; B=2 does. B=1 peak was 29.6 GB in §15 only because of T5 (11 GB, kept resident by `--no_offload`); a server would drop T5 after the prompt encode.

**Per-user FPS.** Every user in the batch waits for the whole step, so per-user frame rate is 16 frames per (B × per-stream cost): with the decoder taken as B× linear like the DiT (its convolutions are at 79 % of fp16 peak at B=1, §13; `vae_batch_bench.py` is written but was not run), a step costs 0.976 s at B=1, 1.88 s at B=2, 3.79 s at B=4 — **16.4, 8.5 and 4.2 FPS per user**. Real time (16 FPS) holds only at B=1. (An earlier draft of this section divided the step cost by B and concluded two users would each see 17 FPS; that was wrong — the per-stream figure is a cost, not a frame rate.)

**Answer.** One RTX 5090 serves **one real-time user** of this model. Batching does not change that: the GPU's output is ≈ 17 as-played frames/s no matter how the batch is arranged, and one user consumes 16 of them. Two users on one card each get 8.5 FPS; the 5 % launch saving is real but cannot be turned into a second real-time stream. Scaling is therefore linear in GPUs — N real-time users need N 5090s (plus a scheduler that pins one session per GPU, which is what TurboServe does for LongLive-class models and what LingBot/LongLive 2.0/Matrix-Game 3.0 do by giving the decoder its own GPU). Batching only makes sense for non-interactive use where 8 FPS per user is acceptable, and even then buys 5 %, not 2×. This is the opposite of LLM decode, where a batch of 1 leaves the GPU memory-bound and batching to 64–128 is nearly free (see the DSpark / speculative-decoding notes in `~/.superset/projects/deepseek-dspark`).

**Housekeeping found on the way.** `bench_perf.py` referenced `env` outside `run_generate` (the audit's M4 change was never run end-to-end) — fixed via `args.run_env`; the B=1 row was re-parsed into `results.tsv`. `bench_perf.py`'s GPU-CSV parser also fails on a two-digit-year timestamp that this pod's `nvidia-smi` emitted once mid-run (`26/09/16 07:14:23.856`): not fixed, the B=2 and B=4 rows are log-only. On this host `pkill -f <script>` from an ssh one-liner kills the ssh session itself when the pattern appears in its own command line; a stray `generate.py` from that survived and caused the first B=2 OOM (25.6 GB held by the dead run) — use `pgrep -f "[g]enerate.py"` from a separate session.

## 17. Re-profile of the shipped v0.2.0 stack against the "levers above the kernel" — pod 10 (2026-09-17)

Question: with the release cut, is any layer above the kernel still worth work (batching, compiler fusion, caching, memory management, launch overhead — the standard list; quantization and step reduction excluded as already classified)? Three `lingbot clip --frame_num 193 --bench` runs on a stock RunPod RTX 5090 (driver 570.195, pod `yo9qf5st6fbdpw`): batch 1 with `LINGBOT_PROFILE` over chunks 8–10, batch 2 (`LINGBOT_BATCH=2`), batch 1 unprofiled as the control. Report: `lingbot-world-v2-stream/bench_results/profile_shipped_v020_pod10.txt`.

Control: DiT 0.649 s + VAE 0.339 s per chunk → 16.2 FPS as played (§15: 0.640 + 0.336). Profiled window: 3 chunks = 1.933 s wall, GPU busy 1.894 s (**98.0 %**), 21 546 kernels, 85 gaps ≥ 20 µs totalling 8 ms, host syncs 2.

| Lever | Measured, per chunk | Headroom | Verdict |
|---|---|---|---|
| Launch overhead / host syncs | idle 0.013 s (1.3 %); largest gap 0.7 ms | 0.013 s — the whole prize of CUDA graphs | saturated |
| Compiler fusion coverage | unfused elementwise/norm/copy **0.001 s**; Inductor-fused 0.092 s in 3 441 launches | the 0.092 s is bandwidth-bound work, not missed fusion | saturated |
| Memory management (KV window shift = the paged-attention analogue) | memcpy/memset 0.002 s | 0 | saturated |
| Prefix / conditioning caching (text K/V once, RoPE table, camera MLP per chunk, T5 on disk) | cross-attn FA2 0.017 s, bf16 GEMMs 0.001 s, T5 absent from the loop | ≤ 0.01 s | saturated |
| Prefill analogue (`cache_write`, the 5th forward) | 0.122 s, same cost as a denoise step | model semantics (the KV write), not scheduling | n/a |
| Batching | batch 2: 1.249 s/chunk = 1.93× batch 1 → 0.625 s per stream (+3.7 % throughput, 2× latency) | none for one player | saturated (§16 confirmed) |

Kernel-level roofline of the same chunk (peaks: RTX 5090 INT8 838 TOPS, FP8 dense 419 TFLOP/s, FP16 dense 209.5 TFLOP/s, 1.79 TB/s):

| Kernel (author) | s/chunk | Achieved | Utilisation | Floor at 100 % |
|---|---|---|---|---|
| SageAttention 2.2 INT8-QK/FP8-PV (Thu et al.), 151 TFLOP/chunk | 0.278 | 543 TOPS | 65 % | 0.180 |
| FP8 GEMMs `_scaled_mm` (cuBLASLt/CUTLASS), ~78 TFLOP/chunk | 0.201 | ~390 TFLOP/s | ~90 % | 0.186 |
| Sage per-call K/V re-quant (Sage) | 0.039 | — | quant-once would leave 0.013 | 0.013 |
| Inductor fused elementwise (torch.compile) | 0.092 | bandwidth-bound | ~70 % est. | ~0.06 |
| VAE decoder convolution (cuDNN fp16), 52 TFLOP/chunk | 0.30 | 173 TFLOP/s | 83 % | 0.248 |
| **Chunk** | **0.98** | | **~65 % of absolute peak** | **~0.70 s → 23 FPS** |

Reading: every lever above the kernel is at its floor (each ≤ 1.3 % of the chunk). The remaining time is inside four kernels written by others, three of them at 83–90 % of peak. The only kernel with real headroom is attention (65 % of the INT8 peak): a hand-written sm_120 kernel reaching 90 % would take 0.278 → ~0.20 s, **+8 % on the chunk, 16.2 → ~17.5 FPS**. On consumer Blackwell (`mma.sync` only, no tcgen05/wgmma) that is a multi-week CUDA effort for about one frame per second, which is why this log ends at the kernel boundary without a custom kernel. What would move the number more is model-side: the fifth forward (0.12 s), the KV window (attention scales with it), and the 4-step schedule.


## 18. Roofline of one chunk per operation, both stacks — pod 14 (2026-09-22)

**Question.** The rooflines so far were per-kernel utilisations (§13, §17) with analytic FLOPs and, for the decoder only, traced bytes (§10). The scaling-book form (Austin et al. 2025, *How to Scale Your Model* §1) asks for every operation: bytes moved, FLOPs, arithmetic intensity against the hardware's ridge point, `T_math = FLOPs / peak`, `T_comms = bytes / bandwidth`, floor = max of the two, and measured time over the floor. This section fills that table for the shipped stack and for the original paper's code, from one pod, so the speed of light in the write-up is measured rather than derived.

**Method.** `tools/roofline.py` + an env-gated hook in `wan/image2video.py` (`LINGBOT_ROOFLINE=<dir>`): the existing `torch.profiler` window over chunks 8–10 (KV window full from chunk 6) with `record_shapes`, plus `key_averages(group_by_input_shape=True)` so every `aten::_scaled_mm` / `aten::mm` / `aten::linear` call has its M, K, N. `LINGBOT_ROOFLINE_TRACE=1` adds a `TorchDispatchMode` byte tracer over the same chunks and over the whole-clip decode (bytes of every CUDA tensor read or written by every aten op, bucketed into attention / matmul / decoder conv / elementwise / memcpy; views excluded) and a `FlopCounterMode` over the decode. The tracer only works on the eager stack: a dispatch mode under `torch.compile` changes what runs, and Inductor's own bandwidth profiler (`TORCHINDUCTOR_PROFILE=1`) hit an illegal memory access re-launching the Sage/FP8 graph, so the two fused elementwise rows of the fast stack carry estimated bytes (measured time at ~70 % of bandwidth, the §17 figure). Attention FLOPs and bytes are analytic from the shapes (q 6 032 × kv 27 144, 12 heads × 128, 30 layers × 5 forwards; the tracer's own count for the stock stack was 36.7 GB against 30.6 analytic, the varlen buffers). Tracer bytes are an upper bound on DRAM traffic — an intermediate that stays in the 96 MB L2 is counted as if it went to memory — so intensities are lower bounds. Runs: `lingbot clip --frame_num 193 --bench` on the dragon example, `--preset fast` profiled, `--preset stock` profiled with the tracer, `--preset stock` plain as the timing control (DiT 1.638 + decoder 1.045 = 2.68 s per chunk, the README's number; the profiled stock run's decoder kernels sum to 1.047 s). Fast stack in the profiled run: DiT 0.639 s per chunk steady, kernels 0.955 s per chunk against 0.98 s wall. Raw: `lingbot-world-v2-stream/bench_results/roofline/pod14/` (kernel tables, per-shape records, byte and FLOP counts, `roofline.json`, logs; traces not committed, 20 MB). Peaks: RTX 5090 dense with FP32 accumulate, INT8 838 TOPS, FP8 419, FP16/BF16 209.5, TF32 104.8 TFLOP/s, 1 792 GB/s (whitepaper App. A). Ridge = peak / bandwidth: 468, 234, 117, 58 FLOP/B.

**Pod.** `0ofgp5gaso0620`, Community Cloud RTX 5090, driver 570, checked with `tools/podcheck.sh` before setup: bf16 8192³ matmul 235 TFLOP/s, 2 482 MHz under load, no throttle reason active, PCIe gen 5. A first attempt on pod 13 (`pf3ox9qd8q0gvu`) was discarded: that host power-capped the card (1 875 MHz, 314 W, `sw_power_cap`, PCIe gen 4, 132 TFLOP/s on the same matmul) and every kernel ran 1.6–2.0× slower; its FLOP and byte counts agreed with pod 14's to the last digit, as they must. `podcheck.sh` exists because of it.

**Nsight Compute cannot run on RunPod.** `ncu` is on the image, but the counters need `NVreg_RestrictProfilingToAdminUsers=0` on the host driver or `--cap-add=SYS_ADMIN` on the container (`ERR_NVGPUCTRPERM`); RunPod grants neither on any tier. Per-kernel SM and DRAM throughput from hardware counters needs a full VM or bare metal with root. Everything below is therefore `torch.profiler` time plus counted FLOPs and bytes, which is the scaling-book method, not Nsight's.

### Ours (`--preset fast`), per 16-frame chunk

| Operation | Precision | FLOP / chunk | Bytes / chunk | FLOP / B | Ridge | Bound | Floor | Measured | Of speed of light |
|---|---|---|---|---|---|---|---|---|---|
| Attention | INT8 | 151 T | 17 GB | 9,048 | 468 | compute | 0.180 s | 0.288 s | **63 %** |
| DiT matmuls | FP8 | 79 T | 64 GB | 1,243 | 234 | compute | 0.188 s | 0.201 s | **94 %** |
| Decoder convolutions | FP16 | 52 T | 39 GB | 1,332 | 117 | compute | 0.249 s | 0.296 s | **84 %** |
| DiT elementwise, fused | FP16 | — | ~117 GB | — | 117 | memory | 0.065 s | 0.093 s | **70 %** |
| Decoder elementwise, fused | FP16 | — | ~40 GB | — | 117 | memory | 0.022 s | 0.032 s | **70 %** |
| Attention K/V re-quant | INT8 | — | 38 GB | — | 468 | memory | 0.021 s | 0.039 s | **54 %** |
| **Chunk** | | | | | | | **0.72 s, 22 FPS** | **0.98 s, 16.1 FPS** | **74 %** |

### Original paper's code (`--preset stock`), per chunk

| Operation | Precision | FLOP / chunk | Bytes / chunk | FLOP / B | Ridge | Bound | Floor | Measured | Of speed of light |
|---|---|---|---|---|---|---|---|---|---|
| Attention | BF16 | 151 T | 31 GB | 4,935 | 117 | compute | 0.720 s | 0.820 s | **88 %** |
| DiT matmuls | BF16 | 79 T | 112 GB | 704 | 117 | compute | 0.377 s | 0.471 s | **80 %** |
| Decoder convolutions | TF32 | 51 T | 78 GB | 663 | 58 | compute | 0.490 s | 0.621 s | **79 %** |
| DiT elementwise, one kernel per op | FP32 | — | 826 GB | — | 58 | memory | 0.264 s | 0.264 s | — |
| Decoder elementwise, one kernel per op | FP32 | — | 516 GB | — | 58 | memory | 0.288 s | 0.393 s | **73 %** |
| KV-cache shift, memcpy | FP32 | — | 28 GB | — | 58 | memory | 0.016 s | 0.053 s | **30 %** |
| **Chunk** | | | | | | | **2.16 s, 7.4 FPS** | **2.68 s, 6.0 FPS** | **80 %** |

**Reading.**

1. *Nothing is on the wrong side of the ridge.* Every large operation is compute bound by a wide margin: attention at 9 048 FLOP/B against a ridge of 468, matmuls at 1 243 against 234, the decoder's convolutions at 1 332 against 117. Moving data is not what limits this model; the arithmetic is. The memory-bound rows (fused elementwise, K/V re-quant, memcpy) are 0.17 s of the 0.98 s chunk and sit at 54–70 % of bandwidth.
2. *The chunk is at 74 % of speed of light*, 0.72 s floor against 0.98 s measured, a 22 FPS ceiling against 16.1. The gap is: attention 0.11 s (63 % of the INT8 peak), decoder convolutions 0.05 s (84 %), fused elementwise 0.04 s, K/V re-quant 0.02 s, matmuls 0.01 s (94 %), the rest launch gaps and cross-attention. §17's 0.69 s floor counted the re-quant as "quant once" and the decoder's elementwise inside one number; this table counts the work as the kernels do it.
3. *The paper's code is at 80 % of its own speed of light*, 2.16 s floor against 2.68 s, a 7.4 FPS ceiling. Its floor is high not because its kernels are bad — FlashAttention-2 at 88 % and cuBLAS at 80 % of the BF16 peak are respectable — but because the peaks it runs against are 2–4× lower (BF16 209.5 against FP8 419 and INT8 838; TF32 104.8 against FP16 209.5) and because unfused fp32 elementwise work touches 1.3 TB per chunk (826 GB in the DiT, 516 GB in the decoder) where the fused stack touches ~160 GB. With perfect fusion at the paper's precisions the floor would still be 1.67 s, 9.6 FPS: real time was unreachable at those precisions no matter the kernels. That is the whole argument for the precision moves in §4, §5 and §9.
4. *Two rows of the §17 peaks table move*: FP8 matmuls 390 → 393 TFLOP/s (78.9 TFLOP measured from the shapes over 0.201 s, 94 %), SageAttention 543 → 524 TOPS (0.288 s on this pod against 0.278 on pod 10, 63 %), decoder convolutions 173 → 176 TFLOP/s (84 %). The README and the write-up render from `bench/summary.json` and carry the new values.
5. *What the table says to do next*, in order of gap: attention (0.11 s; a kernel exists to try, comfy-kitchen's INT8-PV SDPA, TODO), FP8 convolutions for the decoder (moves its floor 0.249 → 0.125 s; no kernel exists for sm_120), K/V quantised once per chunk instead of per call (0.02 s; a change inside SageAttention's API). Everything else is under 0.05 s.

**Not measured.** Per-kernel hardware counters (Nsight, see above); the fast stack's fused-kernel bytes (estimated). Reference: Austin J. et al. (2025), *How to Scale Your Model*, Part 1: All About Rooflines, jax-ml.github.io/scaling-book.


## 19. Attention at 63 % of the INT8 peak: seven hypotheses tested one at a time, three candidate patches — pod 15 (2026-09-22)

**Question.** §18 left one kernel with headroom: SageAttention 2.2's Ada path (`sageattention_sm89::qk_int_sv_f8_attn_kernel<128,64,32,64,…>`, run on sm_120 because consumer Blackwell has no wgmma/tcgen05), 0.288 s of the 0.98 s chunk at 63 % of the 838 TOPS INT8 peak, while FlashAttention-2 reaches 88 % of its BF16 peak on the same shapes. Seven hypotheses, each with its own bench under `bench/attn/` and each measured alone, then red-teamed one by one (one review agent per hypothesis, one per candidate patch); where a review found a hole the measurement was repeated. Shapes throughout: q [1, 6032, 12, 128], k/v [1, 27144, 12, 128], bf16, NHD, 1.006 TFLOP per call. Pod 15 (`5p629028bpcomb`), checked with `tools/podcheck.sh` before use: bf16 matmul 237 TFLOP/s, 2.66 GHz, no throttle. "Whole call" = `sageattn()` including its quant kernels (CUDA events, 20 warm + 50 timed); "kernel only" = the attention kernel's self time under `torch.profiler` over 30 calls (`bench/attn/h2_tiles/bench_kernel_only.py`). Raw numbers: `bench/attn/results/`. The `sageattention` wheels were rebuilt from upstream `d1a57a5` with `TORCH_CUDA_ARCH_LIST=12.0` per configuration (`bench/attn/h2_tiles/build.sh`); the rebuilt base is within 0.03 % of the shipped wheel.

| # | Hypothesis | Bench | Measured | Red-team | Verdict |
|---|---|---|---|---|---|
| 1 | The online softmax (fp32 max, exp2, sum, O rescale) is exposed: the same work at INT8 as at BF16 while the mma is 4× faster | `cfg_nosoftmax` kernel variant with the softmax arithmetic removed, S→P conversion and both mma phases kept (the first design, a standalone QK GEMM, was invalid: it has to write the 6032×27144 score matrix, 7.9 GB per call, that the fused kernel never materialises, and ran memory-bound at 74 TOPS) | kernel only: base 1.742 ms, no softmax **1.518 ms** → the softmax is **0.22 ms, 13 %** of the kernel; without it the kernel reaches 663 TOPS, 79 % of 838 | patch removes exactly the softmax; P still depends on every S element so nothing else was dead-code-eliminated; the remaining 21 % includes CUDA-core conversion and flush work, not only the mma/load pipeline | **partly confirmed: 13 %**, the ceiling for any softmax-hiding trick |
| 2 | Tile constants (CTA_Q 128 / CTA_K 64 / WARP_Q 32) were chosen for Ada and are wrong for sm_120 | four rebuilt tilings (`cfg_a` 64/64/16, `cfg_b` 64/128/16, `cfg_c` 128/64/16, `cfg_d` 128/128/16) plus `cfg_a` with `-maxrregcount=168` for 3 CTAs/SM | whole call: every alternative slower (2.15–2.44 vs 2.09 ms). Kernel only: **cfg_b 1.628 ms, 6 % faster than base 1.738**; cfg_d 1.769, cfg_a 1.881, cfg_c 2.096; cfg_a at 168 registers, no spills, 3 CTAs/SM: 1.743, identical to base | review found that sm_120's per-SM budget (48 warps, 64 K registers, 100 KB smem) is identical to sm_89, that the whole-call numbers charged cfg_b for a V pad copy inside `sageattn()` (fixable outside the hot path, as `sage_kvq.py` already pads its cache), and that all first-round configs ran at the same 8 warps/SM, hence the capped rebuild | **one config wins 6 % kernel-only** (cfg_b: 213 K steps of 128 keys instead of 425 of 64); ~~**occupancy refuted** (3 CTAs/SM = 2 CTAs/SM to the µs)~~ **retracted 2026-09-22, see the note below** |
| 3 | The K/V working set (12 heads ≈ 84 MB quantised) sits at the edge of the 96 MB L2; each CTA re-reads all of K/V at ~256 FLOP/B, below the 468 FLOP/B INT8 ridge | heads 1–24 and Lk 4k–54k sweeps (`h3_l2_working_set.py`) | per-head, per-key cost flat: 6.5 ± 0.2 ns per key per head from 4k to 54k keys and from 6 to 24 heads, including working sets of 106 and 159 MB | stands; the 1→3-head plateau is SM occupancy (48 CTAs per head), not L2 | **refuted** |
| 4 | Wave quantisation: 48 query tiles × 12 heads = 576 CTAs on 170 SMs | Lq sweep 4096–12288 at 1, 12, 24 heads (`h4_wave_quantization.py`) | whole-call utilisation follows the last wave's fill: Lq 6032 (39 % full) 56.8 %, 5120 (82 %) 59.9 %, 7168 (95 %) 64.5 %, 12288 66.1 %; at H = 1 the time is a flat 0.55 ms for 32–96 CTAs | the data fit one resident CTA per SM (~0.53 ms per wave), not the two the register count allows; effect ~9–11 points at our shape; a lone CTA already saturates its SM at ~65 % of the SM's peak | **confirmed, ~10 %**; in the model the shape is fixed (576 CTAs, 85 % wave efficiency), a KV-split kernel (1152 CTAs, 97 %) would be the lever |
| 5 | The fp32+fp16 PV accumulate (fp16 mma accumulation flushed to fp32 once per 64-key tile, fused V scale) costs time | all 12 (kernel, granularity, accumulate) variants of `sageattn_qk_int8_pv_fp8_cuda` / `_pv_fp16_cuda` (`h5_pv_accum.py`) | the shipped mode is the fastest: pure fp32 accumulate +14 %, fp32+fp32 +15 %, the fp16-PV kernels +21–75 % (fp32 accumulate is half-rate FP8 on GeForce parts) | stands; note `pv_fp16 per_warp fp16` gives cos 0.99990 vs 0.99925 at +21 % time, an accuracy option, not a speed one | **refuted** |
| 6 | K/V re-quantised on every call (`TransposePad` / `MeanScale` / `QuantInt8`, 0.039 s per chunk) | exp. 15d, `LINGBOT_ATTN=sage_kvq` | +0.015 s per chunk | — | **refuted** (§15d) |
| 7 | The clock drops under sustained INT8 load below the 2 407 MHz the 838 TOPS figure assumes | `nvidia-smi` at 100 ms during 20 s of `sageattn()` (`h7_clock_sampler.py`, `results/h7_sage.json`) | at the 600 W power cap (588 W mean, `sw_power_cap` active in 194 of 197 busy samples) the card holds a median **2 677 MHz**, above spec; peak at that clock ≈ 932 TOPS | the first run (a `torch._int_mm` load at 28 % of peak) was not representative; the Sage-load sample is | **refuted as a cause; changes the denominator**: 578 TOPS is 62 % of what this card delivers, not 69 % |

**Reading.** Nothing outside the kernel explains the gap: not L2, not the accumulate mode, not the clock (occupancy is unresolved, see the retraction below). The gap is inside one CTA's instruction stream, and it splits into the softmax (13 %, measured by removal) and the rest of the non-mma work plus the last wave (~10 %). This agrees with the literature: nobody reports beating ~70 % of the INT8/FP8 peak with `mma.sync` — SageAttention's own best is ~71 % on a 4090, SageAttention 3 reports 62 % of the FP4 peak on the 5090, FlashAttention-4 reaches 71 % on a B200 even with wgmma/tcgen05 and exp emulation — while BF16 attention on the same 5090 reaches 94–97 % (gau-nernst's write-up, cuDNN, FA2). At BF16 the mma is slow enough to hide the fixed-cost softmax; at 8-bit it is not. Our 62 % is the state of the art for this hardware class, not a defect (`bench/attn/` literature review, 2026-09-22; sources in TODO.md).

**Retraction, 2026-09-22: the occupancy row above does not support its verdict.** The register cap was
applied to `cfg_a`, which also changes the tiling (CTA_Q 128 to 64, WARP_Q 32 to 16), and the result was
then compared against the *baseline* tiling. That comparison mixes two variables. Compared against the
right control, the same tiling at its natural 182 registers, the cap is not neutral at all:

| build | registers | CTAs/SM | kernel-only |
|---|---|---|---|
| base tiling, 128/64/32 | 255 | 2 | 1.738 ms |
| cfg_a tiling, 64/64/16 | 182 | 2 | 1.881 ms |
| cfg_a tiling, 64/64/16 | 168 (`-maxrregcount`) | 3 | 1.743 ms |

Raising occupancy from 2 to 3 CTAs/SM is worth **7.3 %** within one tiling. It only looks like a wash
against the baseline because `cfg_a`'s tiling doubles L2 to SM traffic (4.0 GB to 7.9 GB) and gives back
almost exactly what the occupancy buys. So occupancy helps; what is untested is whether the *baseline*
tiling has anything left to gain, since it needs 255 registers and a cap there may spill.

The experiment that would settle it, one build and one bench:

```
SAGE_H2_CTA_Q=128 SAGE_H2_WARP_Q=32 SAGE_H2_CTA_K=64 \
NVCC_APPEND_FLAGS="-maxrregcount=130" python setup.py bdist_wheel
```

then `bench_kernel_only.py` against the untouched baseline. Watch the build log for spill stores: at
128 threads per CTA, 130 registers is the threshold for 3 CTAs/SM, and the baseline kernel may not fit
without spilling, in which case the answer is that the baseline is register-bound and occupancy is
genuinely unreachable for it.


**Candidate patches** (`bench/attn/h2_tiles/`, one file each, each against the same base, built and measured alone; **written and reviewed on 2026-09-22, not yet built or run**: the pod was stopped before they were ready):

| Patch | Idea (source) | Red-team | Expected |
|---|---|---|---|
| `cfg_b` | CTA_K 128: half the K steps and `__syncthreads` per CTA; needs the V cache padded to 128 rows and `scale_max` 1.125 for the fp16 PV budget | V pad must move out of the hot path; `sage_kvq.py` hard-codes 64-key tiles and must follow | **measured: −6 % kernel-only** (1.628 vs 1.738 ms) |
| `cfg_expemu` (C1) | 1 in 4 exp2 on the FMA pipe via a degree-3 minimax polynomial (FlashAttention-4) | math verified (clamp at −125, rel. error 7.5e-5, 830× under the e4m3 half-ulp, no divergence); spill risk at 255 registers; fix a signed-shift UB with `__float_as_uint` | ≤ a share of the 13 %; reject if spill stores appear |
| `cfg_condrescale` (C2) | keep a stale row max unless the tile max exceeds it by τ, skip the 128-multiply O rescale (FlashAttention-4); τ = 0 skips only exact no-ops, τ = 2 lowers the fp8 offset by 2 bits | exact for O/d, `__any_sync` placed safely; **τ = 2 raises P's flush-to-zero floor ~4×, a downward bias on flat attention rows** — must pass the exp15 quality band, not be assumed lossless | τ = 0: a few % of the rescale time; τ = 2: more, with a quality check |
| `cfg_fusedsoftmax` (C3) | fuse the max reduction with the S→P conversion, 5 register passes → 2 (SageAttention 3) | correct, but a no-op after unrolling (ptxas already interleaves), and the I2F→magic swap targets an instruction that is already full-rate (`I2FP`) on sm_86+; SageAttention 3's 10 % comes from a per-16-element block max that only FP4 micro-scaling needs | **≈ 0, dead end; not worth a build** |
| C4 delayed fp32 conversion (SageAttention 2++) | — | already in 2.2: the fp32+fp16 mode flushes once per 64-key tile (H5 review) | done |

**What is worth doing next, in order.** (1) `cfg_b` into the model: pad the K/V cache to 128-key tiles once per chunk, carry the tile constants into `sage_kvq.py`, halve `scale_max`; −0.02 s per chunk if the 6 % holds in the loop. (2) Build and bench C1 and C2 (τ = 0), 5 min each on the pod; keep whatever shows no spills and no quality change. (3) A KV-split (flash-decoding style) variant of the kernel for the wave tail, ~10 %: a kernel change, not a flag. (4) Nsight Compute on a machine that allows counters, to see where the last 21 % goes. The whole attention headroom, all of it taken, is about 0.06 s per chunk, one frame per second — consistent with §17's estimate, now measured.

**Method notes.** Two `run_all.sh`-style waiters never fired because of a zsh quoting bug (`$S` with embedded spaces); every bench above was run by hand. The first candidate bench for H1 (`h1_softmax_exposed.py`) is kept in `bench/attn/` as a record of the invalid design. Pod 15's `/workspace` is a network filesystem shared with pod 14; the venv there corrupted twice under concurrent pip installs and now lives on local disk (`/root/venv15`, symlinked); `setup.sh` should do that by default when `/workspace` is a network mount.


## 20. Attention: five candidate kernels built and measured, one win — pod 16 (2026-09-22)

**Question.** §19 left attention at 63 % of the INT8 peak with a measured 13 % of the kernel in the online softmax
(`cfg_nosoftmax`) and a literature review saying nobody beats ~70 % of the 8-bit peak with `mma.sync`. This section builds
the candidates that review named, measures each alone, and red-teams each result with its own reviewer (plus one reviewer
whose only job was contradictions between results).

**Method.** Each candidate is a patch against upstream `d1a57a5` under `bench/attn/h2_tiles/`, built into its own wheel and
venv (`build.sh`), and timed two ways: `bench.py` (whole `sageattn()` call, CUDA events, 20 warm + 50 timed) and
`bench_kernel_only.py` (the attention kernel's self time under `torch.profiler`, 30 calls) with the base re-timed
immediately before and after every candidate. Quality gate: cosine against an fp32 SDPA reference on the same inputs,
baseline 0.999246. Shapes: q [1, 6032, 12, 128], k/v [1, 27144, 12, 128], bf16, NHD; 1.006 TFLOP per call. Pod 16
(`xeq9a67r1mi0n0`), checked with `tools/podcheck.sh`: 240 TFLOP/s bf16, 2.66 GHz, no throttle, ncu blocked as always.
SASS via `cuobjdump -sass` on the built wheels. Raw: `bench/attn/results/`.

| Candidate | What it changes | Kernel only | vs base | Cosine | Verdict |
|---|---|---|---|---|---|
| base (7 runs, 2 pods) | — | 1.726–1.741 ms | — | 0.999246 | reference, 0.9 % spread |
| **C5 `cfg_fixedmax`, φ = 8** | online row max → a constant φ (FlashDecoding++), so the max tree, its 2 shuffles, the `o_scale` exps and the 128-multiply rescale loop all go | **1.577 ms** | **−8.7 %** | 0.999230 | **the win**, with a calibration risk |
| C5 control `cfg_fixedmax_dep` | same ALU removed, serial dependency artificially restored (real `__shfl_xor_sync` pair + volatile-asm barrier; SASS SHFL back to 32) | 1.627 ms | −5.8 % | — | **attribution retracted, see below** |
| **C6 `cfg_packedexp`** | 4 scalar `ptx_exp2` per fragment → 2 `ex2.approx.f16x2` (the overload upstream ships and never calls) | **1.677 ms** | **−3.1 %** | 0.999245 | **real gain, mechanism unexplained** |
| cfg_b (from §19) | CTA_K 64 → 128: 213 K steps instead of 425 | 1.628 ms | −6 % (reviewer: ~5.6 %/MAC) | 0.999245 | stands; blocked on `sage_kvq.py` |
| C1 `cfg_expemu` | 1 in 4 exps on an FMA degree-3 polynomial (FlashAttention-4) | 1.738 ms | +0.7 % | 0.999246 | null, confirmed by SASS |
| C2 `cfg_condrescale`, τ = 2 | skip the O rescale unless the row max moves by τ (FlashAttention-4) | 1.720 ms | −0.4 % | 0.999234 | null, confirmed by SASS |
| C3 `cfg_fusedsoftmax` | fuse the max reduction with the S→P conversion (SageAttention 3) | not built | — | — | dead end by review: a no-op after unrolling |
| C7 SpargeAttn | block sparsity on top of Sage | not built | — | — | blocked: its `setup.py` refuses sm_120 |
| `cfg_nosoftmax` (bound) | the whole softmax deleted | 1.518 ms | −13 % | n/a | the ceiling |

**What the SASS says.** Every null was checked in the binary rather than assumed:

- C1 ran: `MUFU.EX2` 204 → 156 (−23.5 %, the intended 1-in-4) with FFMA 1063 → 1351, and it was still slower. Trading one
  MUFU op for ~6 FMA ops loses when the FMA pipe is also busy: **transcendental throughput is not the queue.**
- C2 ran: `VOTE` 0 → 12 and `BRA` 10 → 22 with no predicated FMULs, so the rescale really was skipped ~94 % of the time,
  and nothing moved. **The rescale is not on the critical path when it is skipped conditionally** — but deleting it
  statically (C5) does pay, because the branch and the vote cost what the skip saves.
- C5 removed only 12 `MUFU.EX2` (204 → 192, the `o_scale` ones) yet is 8.7 % faster. The control puts the shuffles back
  (SHFL 8 → 32) and gives half the win back.
- C6 emits 192 `MUFU.EX2.F16` + 12 scalar against base's 204 scalar — **the same 204 issue slots**, because sm_120 has no
  packed MUFU unit and ptxas expands each `ex2.approx.f16x2` into two (plus a `PRMT` to pack the halves). So the −3.1 % is
  real but is *not* the "half the transcendental slots" the patch claims. A literature pass found no published
  microbenchmark that measures the `.F16` MUFU variants separately (the Blackwell microbenchmarking paper does not cover
  the SFU at all), and NVIDIA's arithmetic-throughput table for CC 12.0 could not be retrieved; the most probable cause
  on the evidence is downstream register and instruction savings around the packed form rather than MUFU issue cost. The comments in `h2_tiles.h` and the cfg README assert hardware behaviour that does not exist on
  sm_120 and must be corrected.

**Decomposition of the softmax's 13 %: attempted, and retracted.** The control was built to split C5's 0.151 ms saving
into "serial dependency" and "removed ALU work", and on its face gave 0.050 / 0.100 ms (33 / 67 %). Its reviewer rejected
that split: the dummy chain roots in a different S fragment than the real one and is a different number of dependent
instructions deep (~6 against ~20), and the control also changes the loop nest, so `SAGE_H2_DEP_CONTROL=0` does not
isolate the dependency from the restructuring. **What is safe to say: C5 saves 0.151 ms reproducibly; the cause is not yet
attributed.** A `dep_null` variant (the same restructuring with the dummy chain present but not feeding the exp) is the
cheapest decisive follow-up. What the other experiments do establish is narrower and still useful: the cost is not
transcendental throughput (C1 cut MUFU ops by a quarter and got slower) and not the rescale multiply when it is skipped
conditionally (C2 skipped ~94 % of them, confirmed by SASS, and gained nothing).

**The φ calibration risk, quantified.** φ must upper-bound `max(S · sm_scale)` for every row, layer and head. The usable
window on the bench inputs (true max ≈ 6.5 log₂ units) is narrow and both failure modes are silent:

| φ | 3 | 4 | 5 | 6 | **8** | 10 | 12 | 14 | 16 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| cosine | 0.982 | 0.997 | 0.9991 | 0.99923 | **0.99923** | 0.99921 | 0.9965 | 0.895 | 0.409 | **0.000** |

Below the true max, P saturates e4m3 (clipping); above it, P falls off the bottom of the grid (resolution loss). No NaN,
no error, no warning — a wrong φ silently returns a plausible but wrong video. Production use needs per-layer offline
calibration, a saturation counter (the patch's `SAGE_H2_SAT_CHECK`, which the reviewer judged not yet adequate) and a
fallback to the online max. The reviewer also notes that because `d` accumulates from fp32 P while the numerator comes
from e4m3 P, the 2^-φ factor does not cancel exactly, which is why C5 never quite reaches the baseline cosine.

**What is left open** (each named by its reviewer, none blocking the speed numbers): a spill-count and `ncu` stall-reason
pass to explain C6; a bit-exact `cfg_nosoftmax` dead-code check so it can be quoted as a bound; re-stating every
utilisation figure against one denominator (838 TOPS at the 2 407 MHz spec clock, not the 932 the card actually delivers);
and, for cfg_b, the `sage_kvq.py` work (128-key cache alignment, halved V-scale headroom) before it can reach the model.

**If all of this were taken into the model**, C5 and C6 do not simply add (both touch the same inner loop), but the
attention block is 0.288 s of a 0.98 s chunk, so an 8–11 % kernel gain is 0.02–0.03 s per chunk, 16.1 → about 16.5 FPS.
The honest summary remains §19's: at 8-bit, attention is instruction-bound in its softmax, the hardware has not raised
transcendental throughput in eight generations (Volta 32 → 16 per SM per clock, flat since; tensor throughput 4–8× up),
and 62–70 % of the 8-bit peak is where everyone in this hardware class sits.

## 21. The KV window: a measured +18 % from one flag — pod 20 (2026-09-23)

**Question.** §18 put attention at 0.288 s of a 0.98 s chunk, and §19/§20 established that every
kernel-efficiency lever left is capped at about +12 % end to end, because attention's own roofline floor is
0.18 s. The only way past that is to do less work. Attention cost is linear in the key count, and the key
count is a runtime flag nobody had ever swept.

**What the flag actually means, corrected.** `--local_attn_size` (default 18) is the WHOLE attended buffer,
sink included: `kv_size = frame_seqlen * local_attn_size` (`wan/image2video.py:568-573`, and again at
`:1081-1085`), and 1508 x 18 = 27144 matches the measured shape exactly, where an additive reading
(6 + 18 = 24) would give 36192. So the rolling window is `local_attn_size - sink_size` = 12 latents, not 18.
This agrees with the independent correction recorded in §11 during the KV-ring work.

**Method.** `bench/window/window_sweep.py`, `generate.py --bench`, example 03 lakeside, seed 42, 193 frames,
12 chunks, steady median over chunks 7+. Baseline re-measured first on the same pod in the same sweep, so the
deltas are same-pod. Pod 20 (`b57clgx8sa8gnc`) checked with `tools/podcheck.sh`: 231 TFLOP/s bf16, no throttle,
ncu blocked as always. Raw logs `/workspace/sweep_logs/w{18,12,10}.log`.

| `local_attn_size` | rolling | kv tokens | DiT s/chunk | decode s/chunk | chunk | as-played FPS | vs base | predicted |
|---|---|---|---|---|---|---|---|---|
| **18 (baseline)** | 12 | 27 144 | 0.643 | 0.337 | 0.980 | **16.3** | — | 16.10 |
| **12** | 6 | 18 096 | 0.532 | 0.337 | 0.869 | **18.4** | **+12.9 %** | 17.85 |
| **10** | 4 | 15 080 | 0.493 | 0.338 | 0.831 | **19.3** | **+18.4 %** | 18.52 |

**The baseline reproduces §18 exactly** (0.980 s, 16.3 FPS), which is what makes the deltas trustworthy.

**Both candidates beat their predictions**, and consistently. The prediction scaled only attention's 0.288 s
linearly in kv; at window 12 that is a 0.096 s saving, but the DiT actually dropped 0.111 s. The extra ~0.015 s
is the K/V re-quant term (0.039 s per chunk, §18) shrinking with the window too, which the prediction omitted.
The decoder is unchanged to the millisecond in all three rows, exactly as it should be.

**A configuration below this range is silently broken, not merely degenerate.** In
`wan/modules/model_fast.py:196-206`, `num_rolled_tokens = local_end - num_evicted_tokens - sink_tokens` goes
NEGATIVE once the rolling window is smaller than `chunk_size`. The slice `[sink : sink + negative]` is empty, so
the cache shift becomes a no-op, stale keys survive and attention runs on a corrupted cache with no exception
raised. At `local_attn_size=9` with `sink_size=6` (rolling 3 < chunk 4) this triggers once the buffer fills.
**Do not run 9.** At 10, `num_rolled_tokens` is exactly 0: legal, but no cross-chunk memory beyond the sink
survives, so 10 is expected to fail on quality even though its timing is valid.

**Status: CLOSED, not shipped. See 21c for the full record and the reason.** Speed is real; losslessness is not.
What follows was the plan at the time of writing, kept for provenance:

**Originally: speed measured, quality not yet measured.** This is a class-C change under the
`bench-world-model-quality` skill: the attended context changes, so the rollout diverges from the baseline and
per-frame identity is not the test. The gate is a 30 s clip with per-160-frame drift bins against a
three-run noise band on the same stack, which had never been recorded for this stack before. Nothing here
ships until that is in.

### 21b. The KV window is not redundant: a causal ablation says so (pod 20, 2026-09-23)

The rollout metrics could not settle whether window 12 was lossless: three identical baseline runs
differ by 18 % on whole-clip Laplacian sharpness and 33 % in the last 160-frame bin, and the
candidate landed above that band on one scene and below it on another. So the question was answered
a different way, on one frozen forward pass where there is no seed, no divergence and no band.

**Method.** Run at `--local_attn_size 24`, dump q/k/v for all 30 layers of the first forward whose
KV buffer exceeds 27 000 tokens (`LINGBOT_DUMP_QKV_DIR`, `wan/modules/attention.py`), which caught
the buffer at 20 latents. Then, offline and deterministically, recompute the attention output while
keeping only the sink plus the newest `N - sink` latents, and measure the relative change against
the full 20-latent output: `||out_trunc - out_full|| / ||out_full||`. This is the published CAOTE
idea (arXiv 2504.14051). Attention WEIGHTS were deliberately not used: they are not a faithful
proxy (Jain & Wallace, arXiv 1902.10186), and softmax dilutes mass as O(1/n) so a long window makes
every token look unimportant regardless of whether it is. Script: `bench/window/attn_ablate.py`,
raw: `bench/window/attn_ablate.json`.

| window N | mean rel. output change | worst layer |
|---|---|---|
| 10 | 0.346 | 0.744 |
| **12** (the +12.9 % FPS candidate) | **0.133** | **0.301** |
| 14 | 0.079 | 0.175 |
| 16 | 0.047 | 0.101 |
| **18** (current default) | **0.025** | **0.050** |
| 19 | 0.014 | 0.030 |
| 20 | 0 | 0 |

**Reading: the model uses its whole window.** The curve decays smoothly with no knee and no
plateau, which is what "this context is being used" looks like; a redundant window would show a
flat region that truncation does not disturb. Window 12 perturbs the attention output by 13 % mean
and 30 % at the worst layer, so **the +12.9 % FPS is not lossless**. For under 5 % change the window
would have to stay at 16, and under 2 % needs 19, i.e. above today's default.

**Why the pixel metrics missed it.** A truncated window does not degrade the video, it produces a
coherent DIFFERENT one: the rollout diverges into another plausible trajectory of the same scene.
Per-frame sharpness and MUSIQ cannot separate "different" from "worse", and the run-to-run band is
wider than the effect. This is a general lesson for any lever that changes what the model attends
to, and it is now recorded in the `bench-world-model-quality` skill.

**Caveats.** One forward pass at one point in one rollout, 30 layers, 512 sampled query rows; it
measures immediate output sensitivity, not whether a long-horizon failure (revisit inconsistency)
appears later. The worst-layer column is much worse than the mean at every N, so some layers are
far more context-hungry than others: a per-layer window would be the real optimisation here, and
nothing in production video diffusion does that yet (PyramidKV/Ada-KV do it for LLMs).

### 21c. The KV window, closed: everything tried and why we stopped (2026-09-23)

One place to look before anyone reopens this. The lever was real and the speedup reproduced, but it
fails the losslessness bar by a wide margin and every attempt to rescue it also failed.

**Configurations measured** (pod 20, `generate.py --bench`, seed 42, steady median; lakeside
`examples/03` 193 frames and dragon `examples/00` 481 frames, two scenes agreeing to 0.1 FPS):

| `local_attn_size` | `sink_size` | rolling | kv tokens | lakeside | dragon | vs base |
|---|---|---|---|---|---|---|
| **18** (default) | 6 | 12 | 27 144 | 16.3 | 16.3 | — |
| 12 | 6 | 6 | 18 096 | 18.4 | 18.4 | +12.9 % |
| 10 | 6 | 4 | 15 080 | 19.3 | 19.2 | +18.4 % |
| 12 | 3 | 9 | 18 096 | — | 18.4 | +12.9 % |
| 10 | 3 | 7 | 15 080 | — | 19.3 | +18.4 % |
| 9 | 6 | 3 | — | **never run** | — | structurally broken, see below |

**What was tried, in order, and what happened**

1. **Straight window shrink.** Reproduced on two scenes. Both candidates BEAT their predictions
   (+12.9 % vs 17.85 predicted, +18.4 % vs 18.52), because the K/V re-quant term (0.039 s per chunk)
   shrinks with the window too and the prediction only scaled attention. Diminishing returns are
   sharp: 18 to 12 is +12.9 %, but 12 to 10 adds only +4.9 %.
2. **Quality on a 16.8 s clip (lakeside).** Built the first noise band this stack has ever had:
   three identical runs. w12 landed BELOW the band on sharpness in both later bins. Suggestive of
   late-clip degradation.
3. **Quality on a 30 s clip (dragon).** The opposite result. The band on a fast camera flight is
   enormous (18 % whole-clip, 33 % in the last 160-frame bin) and w12 scored ABOVE it on sharpness,
   MUSIQ and flicker. Two scenes, two contradictory verdicts: the method had no power.
4. **The LongLive sink split.** LongLive (arXiv 2509.22622) ablates 3-27 latents and reports
   "9-local + 3-sink achieves consistency close to a 21-frame window" (verified verbatim). Same kv
   budget as w12, so free. It did NOT transfer: `w12_sink3` had the best MUSIQ and the lowest
   flicker of anything measured but the WORST sharpness, below the band in both later bins.
   LongLive gets its result because it TRAINS for the short window with streaming long tuning; we
   cannot.
5. **Per-layer windows.** The obvious rescue, and PyramidKV/Ada-KV report 30-70 % memory savings
   doing it for LLMs. Killed by our own data: every one of the 30 layers needs 15-20 latents for
   under 2 % output change (mean 18.5). The layers are uniform, so per-layer allocation saves 8 %.
6. **The causal ablation (21b), which settled it.** On a frozen forward pass, truncating to 12
   latents changes the attention output by 13.3 % mean and 30.1 % at the worst layer, on a smooth
   curve with no knee and no plateau. The model uses its whole window.

**Why we stopped.** The trade is not in our favour at any point on the curve:

| shippable at | FPS gain | output perturbation |
|---|---|---|
| window 12 | +12.9 % | 13.3 % mean, 30.1 % worst layer |
| window 16 (most aggressive under 5 %) | **+3.7 %** | 4.7 % |

To stay near lossless the window can only reach 16, and 16 is worth +3.7 % (0.288 to 0.256 s of a
0.98 s chunk, 16.3 to 16.9 FPS). The headline +12.9 % is only available by accepting a 13 %
perturbation. For comparison, FP4 attention alone is about +15 % with a cosine-testable gate, and
the decoder's fp16-accumulate is -0.13 s with a deterministic one. The same effort is worth roughly
4x more in the numerical lane, and certifiable there in a way the window can never be.

**Why the pixel metrics never resolved it.** A shorter window does not produce a worse video, it
produces a coherent DIFFERENT one. Per-frame sharpness and MUSIQ cannot separate "different" from
"worse", and with a coefficient of variation of 15-20 % you need n = 25-40 to detect a 5 % effect;
we had n = 1 per candidate. This is the general lesson: **for any lever that changes what the model
attends to, measure the mechanism, not the pixels.**

**What the literature said** (all URLs in TODO.md): the default 18 has no published justification
and is inherited from upstream; FlashDreams ships 14/6 for this exact checkpoint; attention sinks
provably carry NO semantic memory (StreamingLLM's garbage-linebreak-token experiment), so the
6-latent sink protects no scene content; Wan 2.1 and Self-Forcing fix the window across BOTH
training and inference, so shrinking at inference is off-distribution; and nobody has published a
window-vs-quality ablation for this model family at all.

**What survives and is worth keeping:**

- `bench/window/attn_ablate.py` — deterministic, per-layer, noise-free certification for any
  context-changing lever. This is the tool that will certify block sparsity next.
- `bench/window/score_window.py` + the first noise band for this stack (`bench/window/README.md`).
- A corrected metric policy, now in the `bench-world-model-quality` skill: prefer LPIPS, never gate
  on Laplacian sharpness, and run the band before the candidate.
- One positive finding: **the default 18 is validated**, sitting at 2.5 % error against 20. It is a
  sensible inherited value, not an arbitrary one, which is the reason to stop looking here.

**Do not reopen** unless the model is retrained with a short window (the LongLive route), or unless
someone wants the +12.9 % knowingly as a quality trade rather than as a lossless win.

### 21d. Block sparsity, certified and rejected (pod 20, 2026-09-23)

The counterpart to 21b. Truncating the window fails because it removes information the model uses;
sparsity keeps the whole window and skips only key blocks that contribute nothing, so in principle
it can be lossless where truncation cannot. Measured the same way: one frozen forward pass, the real
dumped activations, deterministic, no rollout noise.

**Method.** `bench/window/block_sparsity.py` on the same 30-layer dump at 20 latents. For each layer
and each of 6 query-block positions: compute the real dense softmax, read each 64-key block's actual
mass off it, drop the lowest-mass fraction, **renormalise the softmax over the survivors** (what a
block-sparse kernel really does) and measure the relative output change. Raw:
`bench/window/block_sparsity_64.json`.

| blocks dropped | mean output change | worst layer | theoretical FPS |
|---|---|---|---|
| 10 % | 0.0069 | 0.039 | +3.0 % |
| 20 % | 0.0152 | 0.073 | +6.2 % |
| 40 % | 0.0377 | 0.134 | +13.3 % |
| 50 % | 0.0536 | 0.172 | +17.2 % |

Largest sparsity inside an error budget: **10 % at 1 %, 20 % at 2 %, 40 % at 5 %.**

**Verdict: not worth building.** At a 1 % budget sparsity buys +3.0 % *theoretically*, and a real
kernel pays per-block mask, gather/scatter and reduced tile occupancy on top, so the realised number
is lower still. This confirms the prior reasoning that our rolling window has already spent the
structural sparsity: there is little left to skip inside an already-local 12-latent window. **C7
SpargeAttn is therefore not worth its build cost either** — its 1.3 % expected-value line assumed
sparsity our workload does not have.

**Two structural findings worth keeping.** Within a layer, the kept-block sets of different query
blocks overlap with a Jaccard of only **0.43**, so a cheap static mask would miss most of the
available sparsity and it would have to be recomputed per step, as SpargeAttn does — more overhead,
on top of an already thin win. Across layers the overlap is 0.76, i.e. layers agree with each other
more than queries within a layer do.

**And an unexpected one: the sink blocks are dropped FIRST at every fraction.** The 6 pinned sink
latents (141 of ~471 blocks, 30 % of the buffer) hold low attention mass in this model. That is
consistent with 21b's finding that recency dominates, and with StreamingLLM's own position that
sinks are a numerical stabiliser rather than a memory. Whether a 6-latent sink earns 30 % of the KV
buffer here is an open question, but it is not pursued: the window work is closed (21c).
