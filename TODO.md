# To-do

## Benchmark other engines that run this model (apples-to-apples FPS on one RTX 5090)

Goal: put our 16.2 FPS "as played" (4 steps, 832×464, original Wan decoder) next to every other
public engine's number for the *same* checkpoint on the *same* card, same metric (seconds per
1 s chunk = 4 denoising steps + decode of 16 frames; the denoise loop alone is a different number).
None of these has been run by us yet; the notes are what the research pass found.

| Engine | What it runs | Known settings / claims | To do |
|---|---|---|---|
| [SGLang](https://github.com/sgl-project/sglang) (diffusion / "realtime" LingBot recipe) | LingBot-World 2 via its diffusion serving path; config seen: shift 5, timesteps `[1000,750,500,250]` warped, `local_attention_frames 45`, sink 3, chunk 3 | recipe targets 8 GPUs / server-class cards; streams JPEG over WebSocket | install on a 5090 pod, load the 1.3B `causal_fast`, run its realtime demo at 832×464, read its per-chunk timing; note chunk 3 vs our chunk 4 (chunk 3 is faster per chunk but was rejected here on sharpness) |
| [vLLM-Omni](https://github.com/vllm-project/vllm-omni) | one session per (multi-GPU) server; `denoise_step()` per step | no single-GPU 5090 number published | same: single-GPU run if the model loads; record s/chunk and VRAM |
| [NVIDIA FlashDreams](https://github.com/NVIDIA/flashdreams) (`integrations_v2/lingbot`) | LingBot cam2v at 16 fps, chunk 3 (`len_t=3`), window 63/0 or 15/3, NVENC WebRTC serving | shipped models need ~120 GB VRAM; the LingBot integration path may fit a 5090 | run its LingBot integration on the 5090; read `chunk_done{control_latency_ms}`; compare s/chunk at its chunk 3 |
| [Reactor cookbook](https://github.com/reactor-team) (`models/lingbot-world-v2`) | production serving of this model: chunk 4, window 18 / sink 6, shift 10, timesteps `[0,179,358,679]` (v1's 14B schedule) | "sub-1 s latency at 16 fps", hosted ≈ $12/stream-h; multi-GPU | run the cookbook's adapter single-GPU if it exposes a local runner; otherwise record their published fps and note the schedule difference |
| Upstream [`Robbyant/lingbot-world-v2`](https://github.com/Robbyant/lingbot-world-v2) `run_fast.sh` | the reference: `torchrun --nproc_per_node=2`, Ulysses + FSDP | 2-GPU recipe; our `--preset stock` is its single-GPU equivalent (5.7 FPS) | already in the README table |
| Community forks | search GitHub for `lingbot-world-v2` forks / "causal_fast" mentions (TeleFuser, MoVerse students, Waypoint-style ports) | unknown | one search pass; list any repo with a runnable single-GPU path and its fps |

Protocol for each: same pod image and driver as `setup.sh`, 361-frame clip from `examples/03`
where the engine allows canned poses, steady-state s/chunk over chunks ≥ 6, peak VRAM, and the
quality check from `lingbot-world-v2-stream/quality_metrics.py` on the output (MUSIQ / sharpness
/ colour) so a faster number that comes from fewer steps or a tiny decoder is labelled as such.
Publish the comparison as a table in the README with links to each engine's config.

## Engine repo

- Bring `wan/image2video.py` up to the streaming pipeline (per-latent emission, `LINGBOT_TIMESTEPS`,
  late input sample, warm reset, record hook); move `live.py` + `control.py` from
  `lingbot-world-v2-stream/stream/` in as `lingbot/play/`.
- `lingbot play`: local window + keyboard (no codec, no network) — the gamer entry point.
- `run.sh play`, README requirements (5090, Linux/WSL2, CUDA 12.8 driver, Python 3.12, ~20 GB disk,
  first start ~3 min), then tag `v0.2.0`.
- Certification on a fresh pod: `exact` preset identity (3 runs, one md5), 3-seed metric band for
  `fast`; write the numbers into README from the run, not from the lab.
- Fresh-user test of `setup.sh` on a stock RunPod image (Python 3.11 / CUDA 12.4 image is common:
  document or handle `python3.12` install).

## Launch target: local RTX 5090 owners (mostly Windows)

Priority order for the launch: (1) Windows 11 via WSL2 — PowerShell bootstrap that enables WSL2 +
Ubuntu and runs `setup.sh` + `lingbot play` inside (WSLg gives the window; driver ≥ 570 exposes the
card); needs ONE verification run on a real Windows 5090 box before launch — none available in
this project, so recruit a tester or launch labelled "WSL2: please report"; (2) Linux:
`setup.sh` + `lingbot play` (verified on the pod); (3) native Windows later (embedded Python +
community sm_120 Windows wheels). Local play = no codec, no network: key→pixel ≈ 1.0–1.2 s at 16 fps.

## Distribution: one command for gamers

No .dmg — no Mac has an RTX 5090. In order of effort:

1. `uvx --from git+https://github.com/kaarelkaarelson/lingbot-world-v2-realtime@vX lingbot play`
   (Linux / WSL2): declare the sm_120 wheels as direct-URL deps in `pyproject.toml` (GitHub release
   assets), weights from Hugging Face on first run. Check whether the `robbyant/…` repos are
   public — if so, drop the `HF_TOKEN` requirement, the biggest friction in `setup.sh`.
2. Docker image on GHCR with Python, CUDA runtime, both wheels and the **pre-compiled Inductor cache**
   (no 2-min warm-up); weights baked in or on a mounted volume. Same image = RunPod one-click
   template. UI = the browser client on `localhost:8765` (no geography; at localhost bitrates the
   codec loss vanishes).
3. Windows: PowerShell bootstrap (enable WSL2 + Ubuntu, run route 1 inside), later a portable zip
   with embedded Python + community sm_120 Windows wheels for SageAttention / flash_attn (Inno
   Setup or MSIX).
4. Launchers: a Pinokio install script; Stability Matrix package.

Every route needs: public weights, the shipped compile cache, and an early hard check that prints
"needs an RTX 5090 (sm_120), driver ≥ 570" instead of failing ten minutes in.

## Findings from the fresh-user test of `setup.sh` (2026-09-17, stock RunPod image)

- Stock image = Python 3.11 / CUDA 12.4 toolkit: `python3.12 not found` → fixed, `setup.sh` fetches
  3.12 via `uv`; `add-apt-repository` is broken on that image, so no deadsnakes.
- RunPod's direct SSH port never published on this pod; the relay works only as an interactive PTY.

## Optimization candidates (2026-09-18, from launch feedback)

- **comfy-kitchen attention** ([Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen), Apache-2.0): pure INT8 SDPA,
  INT8 QK with block-Hadamard like Sage, but INT8 P·V where SageAttention 2.2 uses FP8. Tuned for sm_120. Attention is the one
  kernel with headroom (§17: 65 % of INT8 peak, 0.278 s of the 0.98 s chunk), so a few percent is plausible; no first-party
  benchmark against Sage 2.2 exists, and the 1.4–1.65× figures around are for its block-sparse "Sol" attention (approximate, and
  slower than dense below ~12k tokens; we are at ~27k). Its wheels need CUDA 13 / driver r580, so on the cu128 stack it is a source
  build. Experiment: drop-in behind `LINGBOT_ATTN=kitchen`, one `lingbot bench`, then the exp15 quality set (INT8 PV is coarser
  than FP8 PV). Suggested by a commenter; unverified.
- **comfy-kitchen PR #167** (kijai, H3 VAE): fused GroupNorm+SiLU+pad and fp16-accumulate CUTLASS conv3d in the encoder, CUTLASS
  GEMM decoder with fused residual epilogues. The quoted 2.3× is the encode path; decode 34.6 → 31.1 s. Wan 2.1's decoder is
  conv-based and already fused fp16 at 83 % of peak, so the ceiling here is ~0.06 s per chunk. Not planned.
- **Run-to-run determinism.** Two identical `--preset fast` 22 s runs on one pod differ from frame 0 (36 dB) and diverge to
  ~13 dB by 20 s; same seed does not give the same video. Candidates: `cudnn.benchmark=False`, `use_deterministic_algorithms`,
  pinned autotune choices via `torch.compiler.save_cache_artifacts()` (also removes most of the 2.5 min warm-up and makes results
  reproducible across 5090s). Measure the FPS cost; state the current behaviour under Limits.
- **Length-independent conditioning** (`LINGBOT_REF_FRAMES`, uncommitted patch in `wan/image2video.py`): the sampler draws the
  noise for the whole clip up front and normalises the camera path by its whole-trajectory maximum, so a 961-frame run cannot
  reproduce a 361-frame run's opening even with the same seed. The patch draws the first N latents' noise and the path
  normalisation as the N-frame run does. Only useful once determinism above holds.
- **Per-kernel hardware counters (Nsight Compute).** Not possible on RunPod (any tier: `ERR_NVGPUCTRPERM`, host driver
  setting, see OPTIMIZATIONS.md §18). Needs a full VM or bare metal with root (EC2, Crusoe; Vast.ai VM offers unverified).
  Would settle whether SageAttention's 65 % is tensor-pipe or memory stalls before anyone writes a kernel.
- **Fused-kernel bytes of the fast stack.** The dispatch tracer cannot see compiled kernels and Inductor's bandwidth profiler
  crashes on the Sage/FP8 graph; the two fused elementwise rows of the roofline are estimates. Route: read the Triton kernel
  argument sizes from Inductor's generated wrapper (`TORCH_COMPILE_DEBUG=1`) or profile the fused decoder in eager mode.

## Attention: why 63 % of the INT8 peak — hypotheses and the benches that test them (2026-09-22)

The profiled kernel is SageAttention's Ada path, `sageattention_sm89::qk_int_sv_f8_attn_kernel<128,64,32,64,…>`, run on
sm_120 because consumer Blackwell has no wgmma/tcgen05 and Sage's Hopper kernels need them. 150 calls per chunk, 1.80 ms
each, 1.0 TFLOP per call → ~560 TOPS inside the kernel; FA2 reaches 88 % of *its* BF16 peak on the same shapes. Each
hypothesis has its own script under `bench/attn/` (written 2026-09-22, dry-run on CPU, **not yet run on a GPU** — pod 14's
host had no free 5090 when they were ready); `bench/attn/run_all.sh` runs them one at a time, independently, and
`summarize.py` tables the JSONs. Measure each alone; do not stack them.

| # | Hypothesis | Why it is plausible | Test (`bench/attn/`) | Prior |
|---|---|---|---|---|
| 1 | **Softmax exposed.** The fp32 max / exp / sum / rescale per (q, k) pair is the same work at INT8 as at BF16, but the mma is 4× faster, so the tensor cores wait on it. FlashAttention-3 fixes exactly this on Hopper with warp specialisation and softmax/mma ping-pong; the sm89 kernel has neither. | Predicts 60–70 % on its own; matches FA2's 88 % at BF16 (slow mma hides the same softmax). | `h1_softmax_exposed.py`: sageattn vs a pure INT8 QK GEMM (`torch._int_mm`) plus a pure FP8 PV GEMM (`_scaled_mm`) of the same shapes; the difference is the exposed non-mma time. | strongest |
| 2 | **Tiles tuned for Ada.** CTA_Q 128 / CTA_K 64 / WARP_Q 32 were chosen for sm_89's smem and register budget; never autotuned for sm_120 (170 SMs, different smem, 96 MB L2). | Compile-time template constants; the kernel exists for 8 configs of head_dim, not for the card. | `h2_tiles/`: patches against upstream `d1a57a5` with alternative tile constants, `build.sh` (26-min wheel build per config), `bench.py`. | medium; may not fit registers |
| 3 | **L2 working set.** Each CTA streams the whole K/V: a 64-key tile is 16 KB for 4.2 MFLOP, ~256 FLOP/B, *below* the 468 FLOP/B INT8 ridge, so the kernel is compute-bound only if K/V comes from L2. 12 heads of quantised K/V ≈ 84 MB against a 96 MB L2. | Interleaved head order would thrash; a step in per-head time vs head count is the signature. | `h3_l2_working_set.py`: heads 1–24 at fixed shapes, KV length sweep at 12 and 1 heads, working set printed per row. | medium |
| 4 | **Wave quantisation.** 48 query tiles × 12 heads = 576 CTAs on 170 SMs = 3.4 waves, the last 40 % full. | ~10 % of the call is the tail. | `h4_wave_quantization.py`: Lq sweep 4 096–12 288 (per-token cost; CTAs and waves computed from the device's SM count), at 1, 12 and 24 heads. | small but free |
| 5 | **fp32+fp16 PV accumulate.** PV accumulates in fp16 and flushes to fp32 periodically, plus a fused per-block V-scale; extra non-mma instructions in the inner loop (and the reason V is clamped to ±2.25 in `sage_kvq.py`). | Sage exposes the accumulate mode as an argument. | `h5_pv_accum.py`: `sageattn_qk_int8_pv_fp8_cuda` with pv_accum_dtype fp32 / fp32+fp16 / fp32+fp32 and the fp16-PV kernel's modes, per_warp and per_thread; time and error vs fp32 SDPA. | small |
| 6 | **K/V re-quantised on every call** (0.039 s per chunk of `TransposePad` / `MeanScale` / `QuantInt8`, not overlapped). | — | Already measured, exp. 15d (`LINGBOT_ATTN=sage_kvq`): **negative**, +0.015 s per chunk, because torch/Inductor quantisation of the new chunk plus the per-chunk `requant_all` cost more than Sage's own 0.29 ms per call. Not repeated. | closed |
| 7 | **Clock under sustained INT8 load.** The 838 TOPS figure assumes 2.41 GHz; dense tensor work is the highest-power state. | Pod 10 ran 2.75–2.89 GHz in the loop, so probably not, but it changes the denominator. | `h7_clock_sampler.py`: 100 ms `nvidia-smi` samples during a `lingbot bench` and during a standalone INT8 matmul load; effective peak = 838 × clock / 2 410. | weak |

What would move the number: 1 needs a kernel with softmax/mma overlap on `mma.sync` hardware (the comfy-kitchen INT8-PV
SDPA is the one to try first, above); 2 and 5 are a rebuild of Sage; 3 and 4 are shape/scheduling changes inside the kernel.
A hand-written sm_120 kernel at 90 % of peak would gain ~0.08 s per chunk, about one frame per second (§17).

## Next round: attention at 63 % of the INT8 peak — seven hypotheses, one bench each (2026-09-22)

The profiled kernel is SageAttention's Ada path, `sageattention_sm89::qk_int_sv_f8_attn_kernel<128,64,32,64,…>`, run on
sm_120 because consumer Blackwell has no wgmma/tcgen05 and Sage's Hopper kernels need them. 150 calls per chunk, 1.80 ms
each, 1.0 TFLOP per call → ~560 TOPS inside the kernel; FA2 reaches 88 % of *its* BF16 peak on the same shapes
(`lingbot-world-v2-stream/bench_results/roofline/pod14/rl_fast/kernels_top.txt`, OPTIMIZATIONS.md §18). Each hypothesis
has its own script under `bench/attn/`; **all measured on pod 15 on 2026-09-22, results and verdicts in OPTIMIZATIONS.md §19**. `bench/attn/run_all.sh` runs them one at a time, each in its own process, and
`bench/attn/summarize.py` tables the JSONs in `bench/attn/results/`. Measure each alone; do not stack them.

| # | Hypothesis | Why plausible | Bench | Prior |
|---|---|---|---|---|
| 1 | **Softmax exposed.** The fp32 max / exp / sum / rescale per (q, k) pair costs the same at INT8 as at BF16, but the mma is 4× faster, so the tensor cores wait on it. FlashAttention-3 fixes this on Hopper with warp specialisation and softmax/mma ping-pong; the sm89 kernel has neither. | Predicts 60–70 % by itself and explains FA2's 88 % at BF16 (slow mma hides the same softmax). | `h1_softmax_exposed.py`: sageattn vs a pure INT8 QK GEMM (`torch._int_mm`) plus a pure FP8 PV GEMM (`torch._scaled_mm`) of the same shapes; the difference is the exposed non-mma time. Caveat: the K = 128 GEMM may itself sit below peak, which makes the estimate conservative. | strongest |
| 2 | **Tiles tuned for Ada.** CTA_Q 128 / CTA_K 64 / WARP_Q 32 were chosen for sm_89's smem and register budget, never autotuned for sm_120 (170 SMs, 96 MB L2). | Compile-time template constants. | `h2_tiles/` (README with the constants and constraints from upstream `d1a57a5`, patches with alternative tiles, `build.sh` for a 26-min wheel build per config, `bench.py`). | medium; may not fit registers |
| 3 | **L2 working set.** Each CTA streams the whole K/V: a 64-key tile is 16 KB for 4.2 MFLOP, ~256 FLOP/B, *below* the 468 FLOP/B INT8 ridge, so the kernel is compute-bound only if K/V comes from L2; 12 heads of quantised K/V ≈ 84 MB against a 96 MB L2. | Interleaved head order would thrash; a step in per-head time vs head count is the signature. | `h3_l2_working_set.py`: heads 1–24 at the model shapes, KV-length sweep at 12 and at 1 head, working set printed per row. | medium |
| 4 | **Wave quantisation.** 48 query tiles × 12 heads = 576 CTAs on 170 SMs = 3.4 waves, the last 40 % full. | ~10 % of the call is the tail. | `h4_wave_quantization.py`: Lq sweep 4 096–12 288 (per-token cost; CTAs and waves from the device's SM count) at 1, 12 and 24 heads. | small, free |
| 5 | **fp32+fp16 PV accumulate.** PV accumulates in fp16 and flushes to fp32 periodically, plus a fused per-block V scale: extra non-mma instructions in the inner loop (and why V is clamped to ±2.25 in `wan/modules/sage_kvq.py`). | Sage exposes the accumulate mode as an argument. | `h5_pv_accum.py`: `sageattn_qk_int8_pv_fp8_cuda` with pv_accum_dtype fp32 / fp32+fp16 / fp32+fp32 and the fp16-PV kernel's modes, per_warp and per_thread; time and error vs fp32 SDPA. | small |
| 6 | **K/V re-quantised on every call** (0.039 s per chunk of `TransposePad` / `MeanScale` / `QuantInt8`, not overlapped). | — | Already measured, exp. 15d (`LINGBOT_ATTN=sage_kvq`, `wan/modules/sage_kvq.py`, `lingbot-world-v2-stream/probe_sage_kvq.py`, `bench_results/probe_sage_kvq_pod5.out`): **negative**, +0.015 s per chunk. Not repeated. | closed |
| 7 | **Clock under sustained INT8 load.** The 838 TOPS figure assumes 2.41 GHz; dense tensor work is the highest-power state. | Pod 10 ran 2.75–2.89 GHz in the loop, so probably not, but it changes the denominator. | `h7_clock_sampler.py`: 100 ms `nvidia-smi` samples during a `lingbot bench` and during a standalone INT8 matmul load; effective peak = 838 × clock / 2 410. | weak |

If 1 holds, the fix is a kernel with softmax/mma overlap on `mma.sync` hardware; comfy-kitchen's INT8-PV SDPA (above) is the
existing candidate, a hand-written sm_120 kernel the expensive one (~0.08 s per chunk, one frame per second, §17). 2 and 5
are a Sage rebuild; 3 and 4 are scheduling changes inside the kernel.

### After §19 (2026-09-22): what is left on attention

- `cfg_b` tiling (CTA_K 128) into the model: −6 % kernel-only measured; needs the K/V cache padded to 128-key tiles and `sage_kvq.py` updated. ~0.02 s per chunk.
- Build and bench `cfg_expemu` (C1) and `cfg_condrescale` (C2, τ = 0 first): patches reviewed, unbuilt. Ceiling for both together is the measured 0.22 ms softmax (13 % of the kernel). C2 at τ = 2 needs the exp15 quality band.
- KV-split kernel for the wave tail (576 → 1152 CTAs, 85 → 97 % wave efficiency, ~10 %): a kernel change.
- Dead ends, do not repeat: alternative tilings other than cfg_b, L2 working set, PV accumulate modes, clock (the card runs above spec), K/V re-quant (§15d), fused max+quant (C3, no-op after unrolling).
- Literature (2026-09-22 review): nobody beats ~70 % of the 8-bit peak with `mma.sync`; SageAttention 3 (arxiv 2505.11594) 62 % of FP4 peak on the 5090; FlashAttention-4 (arxiv 2603.05451) 71 % on B200; gau-nernst BF16 FA on the 5090 94 % (gau-nernst.github.io/fa-5090). Our 62 % is the state of the art for this hardware class.

### After §20 (2026-09-22): attention candidates, what is settled and what is open

Settled by measurement (pod 16, kernel-only, red-teamed one reviewer per result): **C5 fixed max -8.7 %** (the 2/3 ALU, 1/3 dependency
split is retracted, see OPTIMIZATIONS.md; the saving is real, its decomposition is not) and **C6 packed exp -3.1 %** at parity quality are the only
wins; C1 (FMA-polynomial exp) and C2 (conditional rescale) are nulls confirmed in SASS; C3 is a no-op after unrolling.

Open, in the order worth doing:

**Run order for the next pod session** (nothing below has touched hardware; every figure is a prediction):
1. `bench/window/window_sweep.py` - no build, largest predicted win, one variable.
2. `build.sh cfg_depnull` then `bench_kernel_only.py` - one build, settles whether C5's 8.7 % is safely reachable.
3. FP4 (SageAttention 3) - biggest ceiling, but a build plus two integration fixes plus its own quality run.
4. `bench/attn/c7_sparge_build.md` phases 1-3 - build debug, modest payoff.

- **~~Shorten the KV window~~ CLOSED 2026-09-23, do not reopen. See OPTIMIZATIONS.md 21c.** Real
  speedup (+12.9 % at window 12, +18.4 % at 10, reproduced on two scenes), but NOT lossless: a
  causal ablation on a frozen forward pass shows window 12 changes attention outputs by 13.3 % mean
  and 30.1 % worst-layer, on a smooth curve with no plateau, i.e. the model uses its whole window.
  Staying under 5 % output change caps the window at 16, which is worth only +3.7 %. Rescue attempts
  that also failed: LongLive's 9-local+3-sink split (did not transfer, they train for it), per-layer
  windows (all 30 layers want 15-20 latents, saves 8 %), and two rounds of rollout quality metrics
  (contradicted each other across scenes; n=1 against a band spanning 18-33 %). Reopen only if the
  model is retrained with a short window, or if the +12.9 % is wanted knowingly as a quality trade.
  Tooling that survives and is reusable: `bench/window/attn_ablate.py` (deterministic per-layer
  certification for any context-changing lever), `bench/window/score_window.py`, and the stack's
  first noise band in `bench/window/README.md`.
- **`cfg_depnull`: does C5's 8.7 % survive without phi?** C5 is dead (see below), but its saving is real and
  unattributed. If the win comes from the loop restructuring rather than from deleting the online max, it is available
  with no phi and no calibration risk. `bench/attn/h2_tiles/cfg_depnull.patch` computes the REAL row max (same
  fragments, same ~20-deep chain, same two `__shfl_xor_sync`) and sinks it through `asm volatile("" : "+f"(m_temp))`
  with no arithmetic edge back into exp2, fixing both flaws that got `cfg_fixedmax_dep` retracted. Verified off-pod:
  applies clean at `d1a57a5`. Read: ~1.577 ms means the dependency edge was the whole story; ~1.73 ms means the ALU
  work costs cycles even disconnected and no restructure recovers it; between the two is a defensible split.
  **Known residual confound, stated by its author:** "not feeding the exp" forces two disjoint loop nests where the
  baseline fuses them, giving ptxas scheduling freedom the real chain never has, so a "no slowdown" result is an UPPER
  BOUND on what a dependency-preserving restructure could recover, not proof one would reach it. Design and SASS
  checklist: `bench/attn/h2_tiles/cfg_depnull.md`.

- **FP4 attention via SageAttention 3 (`sageattention3_blackwell`), the largest attention lever available.** Public,
  Apache-2.0, targets sm_120 via `mma.sync` (no tcgen05 needed), builds at `sm_120a` on CUDA 12.8 which is what we run.
  Claimed 1038 TOPS on a 5090 = 62 % of the 1676 TFLOP/s dense FP4 peak, the same utilisation our INT8 kernel already
  gets, so the arithmetic is coherent: attention 0.288 s to ~0.16 s, chunk 0.98 to 0.85, about 16.3 to 18.8 FPS. This
  matches the "-0.12 s" we had already scoped for it from a different direction.
  - **The C5 calibration trap does NOT apply.** NVFP4 scales are computed online per 1x16 block from the data; there is
    no offline constant, no per-layer and no per-head calibration. K/Q smoothing is a data-dependent per-layer mean.
  - **Two integration landmines, both verified against the released `sageattn3/api.py` on 2026-09-22, neither obvious:**
    1. **Layout.** `sageattn3_blackwell(q, k, v, attn_mask=None, is_causal=False, per_block_mean=True)` has NO
       `tensor_layout` argument and hardcodes `QL = q.size(2)`, `pad_128` on dim 2, `k.mean(dim=-2)`. That is HND
       `[B,H,L,D]`. We call Sage 2 with `tensor_layout="NHD"` `[B,L,H,D]` (`wan/modules/attention.py:165`). Passing our
       tensors straight through would treat 12 heads as the sequence and pad it to 128, silently, with no error.
       Transpose(1,2) in and back out, and count the permute in the timing.
    2. **It mutates K in place.** `preprocess_qkv` begins `k -= k.mean(dim=-2, keepdim=True)`. Our call site passes
       `k.to(dtype)`, and `.to()` returns the SAME tensor when the dtype already matches, so this would subtract the
       mean from the live KV cache and corrupt every later chunk. Presents as gradual quality drift, not a crash.
       Clone K before the call, or confirm a copy actually happens.
  - Shape fit is otherwise fine: head_dim 128 is accepted (only >= 256 falls back to SDPA), `is_causal=False` is a real
    parameter, `QL` and `KL` are read independently so our 6032-vs-27144 asymmetry is supported, and `pad_128` handles
    kv 27144 not being a multiple of 128.
  - Quality is unmeasured for us. The paper's 0.9952 is the attention map against NAIVE FP4 quantisation on CogVideoX,
    not against fp32 SDPA at our shapes, so it is not comparable to our 0.999246 and must be measured here.

- **Occupancy on the baseline tiling, unresolved and previously mis-reported.** The old "occupancy refuted" verdict
  compared a register-capped `cfg_a` against the *baseline*, which changes tiling and registers together. Against the
  right control, `cfg_a` at its own 182 registers, the cap is a 7.3 % win (1.881 to 1.743 ms), so occupancy does help.
  Untested: whether the baseline tiling gains anything, since it wants 255 registers. One build, one bench:
  `SAGE_H2_CTA_Q=128 SAGE_H2_WARP_Q=32 SAGE_H2_CTA_K=64 NVCC_APPEND_FLAGS="-maxrregcount=130" python setup.py bdist_wheel`,
  then `bench_kernel_only.py` against the untouched baseline. Check the build log for spill stores first: if the
  baseline cannot fit in 130 registers without spilling, the answer is that it is register-bound and this is closed.
- **C5 is dead on the real model, confirmed twice.** A single global phi cannot work: per-head row maxima inside one
  layer span 2.6 to 32.2 log2 (29.6) against a 3.5 log2 fp8 window (`bench/attn/results/phi_calib_steady.json`). The
  synthetic-only window measurement was sound, the generalisation was not. Any revival needs per-(layer, head) phi plus
  saturation guards, which is a different and much larger change.
- `bench/attn/phi_calib.py` is deprecated and must not be used: torch.compile binds the original `attention()` object,
  so an external monkey-patch never sees the real calls. Use `LINGBOT_DUMP_QKV=<path>` (in `wan/modules/attention.py`)
  and analyse offline, which is how the phi numbers above were produced.
1. **Why is C6 faster?** Its stated mechanism is false: sm_120 has no packed MUFU unit, so `ex2.approx.f16x2` expands to
   two `MUFU.EX2.F16` and the issue count is unchanged (204 either way). Candidates: `.F16` throughput or register
   pressure at the 255 cap. Fix the wrong comments in `h2_tiles.h` and the cfg README regardless.
2. **Do C5 and C6 compose?** Both rewrite the same inner loop; never measured together, and that combination is what a
   model patch would ship.
3. **phi in the real model.** Measured only on randn. Needs per-layer/head calibration over a rollout, a saturation
   counter and a fallback; the usable window is about +-2 log2 units and both failures are silent (cosine 0.000 at phi=20).
   Transformer Engine's amax-history/delayed-scaling is the shape to copy.
4. **Quality gates.** Every candidate was gated on cosine against fp32 SDPA on synthetic tensors. None has been through
   the exp15 lossless band (PSNR/SSIM/LPIPS/MUSIQ on generated video), which is the bar that matters.
5. **`cfg_nosoftmax` as a bound** needs a dead-code check before 1.518 ms can be quoted.
6. **One denominator**: state utilisation against 838 TOPS at the 2 407 MHz spec clock everywhere (the card delivers
   ~932 at its measured 2 677 MHz; mixing the two made two summaries inconsistent).
7. **C7 SpargeAttn**: build unresolved, one verified fix ready. Upstream PR #123 is **closed, unmerged and abandoned
   by its author**, so it is NOT evidence the approach works; an earlier note here citing it as a "3-line setup.py
   change" was wrong. What is established: SageAttention carries no `-include,cassert` and builds clean on this
   toolchain, SpargeAttn carries it at `setup.py:65` and fails with the `std::__terminate` redefinition, so the flag is
   the prime suspect and `sm120.patch` now drops it unconditionally (verified off-pod against upstream `ae5b629`).
   Recipe with three suspects in cost order: `bench/attn/c7_sparge_build.md`. Harness: `bench/attn/c7_sparge.py`.
   Expected gain revised down to 1.2-1.4x, not the paper's 1.83x: our rolling window has already spent the structural
   sparsity, leaving only content-dependent gains.
8. **cfg_b** needs the `sage_kvq.py` work (128-key cache alignment, halved V-scale headroom) before its 6 % reaches the model.

### C7 SpargeAttn: unresolved, resumable (2026-09-22)

The sm_120 arch gate is fixed (`bench/attn/sparge/sm120.patch`, applies on `ae5b629`), but the build dies in the host pass
of the 73 generated `instantiations_sm80/*.cu` with `redefinition of std::__terminate` in gcc 13's `c++config.h`.
Ruled out: our extra `<cstddef>`, build parallelism, and a globally broken toolchain (SageAttention builds clean on the
same gcc 13.3 + CUDA 12.8). NOT ruled out, in the order to test: (1) a baseline build of UNPATCHED upstream at arch 8.9
in a clean venv, which is what actually exonerates or implicates our patch; (2) `build.sh`'s `--system-site-packages`
venv layering two torch include trees, which matches the double-parse signature and is the red-team's strongest suspect;
(3) `120a` gencode possibly needing CUDA 12.9+ (we are on 12.8; upstream PR #123 uses `12.0a/12.1a`); (4) `-ccbin g++-12`
(g++-12 12.4.0 was installed on the lost pod; gcc 13.3 exceeds CUDA 12.8's documented max of 13.2).
Excluding `instantiations_sm80` is NOT an option: the bf16 kernels live there.
Expected value remains low - our rolling window is already the pruned set - so this is worth at most one more session.

### phi calibration: first data, rerun needed (2026-09-22, pod 18)

`bench/attn/phi_calib.py` runs and produces data (`bench/attn/results/phi_calib_pod18.tsv`, 120 rows), but the first
result is NOT usable: every row has `lk=6032`, i.e. chunk 0 with an unfilled KV cache rather than the steady state
(`lk=27144`), only 2 of 4 requested layers appear, and only chunk 0 appears despite `--chunks 0-11`. As measured it says
the (layer, head) row maxima span 63.8 log2 units against a 3.5 log2 fp8 window - "needs per-layer-per-head phi" - which
would make C5 much less attractive, but the data does not yet support that conclusion.

Before trusting it: fix the layer index (must be `enumerate(self.blocks)` order, and assert every requested layer appears)
and the chunk gate, then rerun `--chunks 6-8 --forwards 0 --frame_num 193` over all 30 layers and filter the analysis to
`lk == 27144`. Three harness bugs were already fixed today and pushed: `LINGBOT_TORCH_COMPILE` is a MODE STRING so "0" is
truthy (unset it, or set it empty in the shell - `apply_preset` uses `setdefault`, so a shell value wins); the hook's
wrapper must tolerate a list first argument (`WanModel.forward`); and `_flush()` silently writes nothing when no rows
were recorded, so an empty TSV means the sampling gate never passed, not that the model has no logits.

### phi calibration RESULT (2026-09-22, pod 20): C5's simple form is dead

Measured on real steady-state tensors (q [1,6032,12,128], k [1,27144,12,128], captured with the repo's own
`LINGBOT_DUMP_QKV` hook during a 16.3 FPS run): the per-head row maxima inside ONE layer are
5.6, 22.5, 3.2, 2.6, 32.2, 29.7, 20.1, 17.5, 2.6, 5.1, 13.1, 31.3 log2 - a spread of **29.6 log2 against an fp8 usable
window of ~3.5**. A single phi set for the hottest head flushes the coldest heads' probabilities to zero, so neither a
global phi nor a per-layer phi is viable; C5 would need a per-(layer,head) phi tensor, offline calibration and a
saturation guard. That is a different, much larger change than the constant the -8.7 % was measured with, and the
failure mode stays silent. **Recommendation: do not pursue C5 in its current form.** Its measurement still stands as the
evidence that the softmax's cost is instruction count rather than SFU throughput.

Instrumentation lesson: `bench/attn/phi_calib.py`'s hook never fired, because the DiT is torch.compiled and the compiled
graph binds the original `attention()` function object - patching module-level references afterwards cannot affect it
(the repo's own dump hook works precisely because it lives inside that function). Future in-model instrumentation must
live inside `attention()` or run with compilation genuinely disabled; and `@torch._dynamo.disable` is required on any
sampling code that does run inside the graph (`torch.quantile` fails on fake tensors and the recording is skipped
silently).
