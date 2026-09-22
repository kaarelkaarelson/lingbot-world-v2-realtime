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
