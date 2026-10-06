"""Offline generation: image + prompt + camera path -> mp4.

    python -m lingbot.generate --preset fast  ...    # ours (lingbot/pipelines)
    python -m lingbot.generate --preset stock ...    # the paper's code (reference/), unmodified
    torchrun --nproc_per_node=2 -m lingbot.generate --preset stock --ulysses_size 2 --dit_fsdp --t5_fsdp ...

`lingbot clip` and `lingbot bench` call this with the example's defaults.
"""
import argparse
import logging
import os
import random
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import torch
import torch.distributed as dist
from PIL import Image

from lingbot.presets import PRESETS, apply_preset
from lingbot.registry import DEFAULT_MODEL, MODELS


def _early_setup(argv):
    """Before anything imports reference/ or the attention backend: export the preset's env and
    pin this process's GPU (reference/wan/modules/t5.py touches the current device at import)."""
    preset = "fast"
    for i, a in enumerate(argv):
        if a == "--preset" and i + 1 < len(argv):
            preset = argv[i + 1]
        elif a.startswith("--preset="):
            preset = a.split("=", 1)[1]
    apply_preset(preset)
    if "--bench" in argv:
        os.environ.setdefault("LINGBOT_BENCH_TIMING", "1")
    if "--bench_e2e" in argv:  # per-chunk decode on a side stream, timed with CUDA events (lingbot/benchmark.py)
        for k in ("LINGBOT_BENCH_E2E", "LINGBOT_VAE_STREAM", "LINGBOT_DECODE_FIRST"):
            os.environ.setdefault(k, "1")
    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return preset


PRESET = _early_setup(sys.argv[1:])

from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS  # noqa: E402
from wan.utils.utils import save_video, str2bool  # noqa: E402

MODEL = MODELS[DEFAULT_MODEL]
_EXAMPLE = {
    "prompt": "A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a "
              "brilliant blue sky — stone pathways stretch into the distance, watchtowers stand sentinel, and vibrant "
              "foliage blankets the mountainsides as the camera glides smoothly forward, capturing the grandeur and "
              "timeless majesty of this ancient wonder.",
    "image": "examples/04/image.jpg",
}


def _parse_args():
    p = argparse.ArgumentParser(description="Generate a video with LingBot-World 2.0")
    p.add_argument("--preset", default="fast", choices=sorted(PRESETS),
                   help="fast: our stack (default); exact: fast with the paper's time-embedding MLP; "
                        "stock: the paper's code in reference/")
    p.add_argument("--bench", action="store_true", help="Print per-chunk time and the as-played FPS at the end.")
    p.add_argument("--bench_e2e", action="store_true",
                   help="End-to-end throughput and first-frame latency from frame-ready times (lingbot/benchmark.py).")
    p.add_argument("--trials", type=int, default=1, help="Timed rollouts for --bench_e2e (default 1; more is optional).")
    p.add_argument("--warmup", type=int, default=0,
                   help="Untimed rollouts before the timed ones (default 0). Throughput is unaffected either way (steady "
                        "chunks come after the compiles); with 0, first-frame latency includes compilation (reported as cold).")
    p.add_argument("--task", default=MODEL["task"], choices=list(WAN_CONFIGS.keys()))
    p.add_argument("--size", default="480*832", choices=list(SIZE_CONFIGS.keys()),
                   help="Area (width*height); the aspect ratio follows the input image.")
    p.add_argument("--frame_num", type=int, default=None, help="Frames to generate, 4n+1.")
    p.add_argument("--chunk_size", type=int, default=4, help="Latent frames per chunk (4 = 16 video frames).")
    p.add_argument("--ckpt_dir", default=MODEL["ckpt_dir"])
    p.add_argument("--assets_dir", default=MODEL["assets_dir"],
                   help="Directory with T5 / VAE / tokenizer when the DiT checkpoint does not include them.")
    p.add_argument("--offload_model", type=str2bool, default=False,
                   help="Move the DiT to the CPU after generation (and empty the cache between forwards).")
    p.add_argument("--t5_cpu", action="store_true", help="Run T5 on the CPU.")
    p.add_argument("--decoder_gpu", type=int, default=None,
                   help="Run the decoder on this GPU, overlapped with the DiT on GPU 0 (two-GPU layout B1).")
    p.add_argument("--ulysses_size", type=int, default=1, help="Sequence parallel degree (stock preset only, for now).")
    p.add_argument("--dit_fsdp", action="store_true", help="FSDP for the DiT (stock preset only).")
    p.add_argument("--t5_fsdp", action="store_true", help="FSDP for T5 (stock preset only).")
    p.add_argument("--save_file", default=None)
    p.add_argument("--save_dir", default="output")
    p.add_argument("--prompt", default=None)
    p.add_argument("--image", default=None)
    p.add_argument("--action_path", default=None, help="Directory with poses.npy and intrinsics.npy.")
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--sample_shift", type=float, default=None)
    p.add_argument("--local_attn_size", type=int, default=18, help="KV window in latent frames.")
    p.add_argument("--sink_size", type=int, default=6, help="Latent frames kept from the start of the rollout.")
    p.add_argument("--max_attention_size", type=int, default=None)
    args = p.parse_args()

    assert args.task in WAN_CONFIGS, f"unsupported task: {args.task}"
    assert not (args.bench and args.bench_e2e), "--bench syncs every chunk, which would distort --bench_e2e; pick one"
    assert args.size in SUPPORTED_SIZES[args.task], \
        f"unsupported size {args.size} for {args.task}; supported: {', '.join(SUPPORTED_SIZES[args.task])}"
    args.prompt = args.prompt or _EXAMPLE["prompt"]
    args.image = args.image or _EXAMPLE["image"]
    cfg = WAN_CONFIGS[args.task]
    args.sample_shift = cfg.sample_shift if args.sample_shift is None else args.sample_shift
    args.frame_num = cfg.frame_num if args.frame_num is None else args.frame_num
    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    return args


def _time_reference(pipe, passes_per_chunk):
    """Per-chunk timing for the paper's loop without editing reference/: sync after every
    `passes_per_chunk`-th DiT call (a chunk boundary: 4 denoising passes + 1 cache write) and around
    the decode, the same method LINGBOT_BENCH_TIMING uses in our loop."""
    calls, state = [0], {"t_prev": None}
    pipe.bench_chunk_s = []
    forward, decode = pipe.model.forward, pipe.vae.decode

    def timed_forward(*a, **kw):
        if state["t_prev"] is None:
            torch.cuda.synchronize()
            state["t_prev"] = time.perf_counter()
        out = forward(*a, **kw)
        calls[0] += 1
        if calls[0] % passes_per_chunk == 0:
            torch.cuda.synchronize()
            now = time.perf_counter()
            pipe.bench_chunk_s.append(now - state["t_prev"])
            logging.info(f"BENCH chunk={len(pipe.bench_chunk_s) - 1} chunk_s={now - state['t_prev']:.3f}")
            state["t_prev"] = now
        return out

    def timed_decode(*a, **kw):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = decode(*a, **kw)
        torch.cuda.synchronize()
        pipe.bench_vae_decode_s = time.perf_counter() - t0
        logging.info(f"BENCH vae_decode_s={pipe.bench_vae_decode_s:.3f}")
        return out

    pipe.model.forward, pipe.vae.decode = timed_forward, timed_decode


def build_pipeline(args, cfg, rank, device):
    if PRESET == "stock" and args.decoder_gpu is not None:
        raise SystemExit("--decoder_gpu runs our pipeline only; the paper's code decodes on the DiT's GPU")
    if PRESET == "stock" and args.bench_e2e:
        raise SystemExit("--bench_e2e needs per-chunk decoding; the paper's code decodes the whole clip at the end. "
                         "Use --bench for the stock preset.")
    if PRESET == "stock":
        import wan as paper
        pipe = paper.WanI2VCausal(
            config=cfg, checkpoint_dir=args.ckpt_dir, device_id=device, rank=rank, t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp, use_sp=args.ulysses_size > 1, t5_cpu=args.t5_cpu,
            local_attn_size=args.local_attn_size, sink_size=args.sink_size, infer_mode="causal_fast",
            assets_dir=args.assets_dir)
        if args.bench:
            _time_reference(pipe, passes_per_chunk=5)
        return pipe
    if args.ulysses_size > 1 or args.dit_fsdp or args.t5_fsdp:
        raise SystemExit("--ulysses_size / --dit_fsdp / --t5_fsdp: only the stock preset runs on several GPUs so far")
    from lingbot.registry import pipeline_class
    return pipeline_class(DEFAULT_MODEL)(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=device, rank=rank, t5_cpu=args.t5_cpu,
        local_attn_size=args.local_attn_size, sink_size=args.sink_size, assets_dir=args.assets_dir,
        decoder_device_id=args.decoder_gpu)


def _print_bench_summary(pipe, args, cfg):
    """Steady-state chunk time (chunk index 5 on, as lingbot.benchmark) and FPS as played."""
    chunk_s = getattr(pipe, "bench_chunk_s", [])
    if not chunk_s:
        print("BENCH: no chunk timings recorded (pass --bench)")
        return
    from lingbot.benchmark import STEADY_FROM
    steady = chunk_s[STEADY_FROM:] if len(chunk_s) > STEADY_FROM else chunk_s[len(chunk_s) // 2:]
    steady_med = sorted(steady)[len(steady) // 2]
    frames_per_chunk = args.chunk_size * cfg.vae_stride[0]
    print(f"BENCH preset={PRESET} gpus={int(os.getenv('WORLD_SIZE', 1))} chunks={len(chunk_s)} "
          f"dit_s_per_chunk(first)={chunk_s[0]:.3f} steady_median={steady_med:.3f} "
          f"-> denoise-loop FPS {frames_per_chunk / steady_med:.1f}")
    vae_s = getattr(pipe, "bench_vae_decode_s", None)
    if vae_s is not None:
        dec_per_chunk = vae_s / len(chunk_s)
        print(f"BENCH vae_decode_s_per_chunk={dec_per_chunk:.3f} "
              f"-> as-played FPS {frames_per_chunk / (steady_med + dec_per_chunk):.1f} (real time = {cfg.sample_fps})")
    print("BENCH chunk_s=" + " ".join(f"{s:.3f}" for s in chunk_s))


def main():
    args = _parse_args()
    rank, world_size = int(os.getenv("RANK", 0)), int(os.getenv("WORLD_SIZE", 1))
    device = int(os.getenv("LOCAL_RANK", 0))
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.ERROR,
                        format="[%(asctime)s] %(levelname)s: %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    cfg = WAN_CONFIGS[args.task]

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp or args.ulysses_size > 1), \
            "FSDP and sequence parallel need a distributed launch (torchrun)"
    if args.ulysses_size > 1:
        from wan.distributed.util import init_distributed_group
        assert args.ulysses_size == world_size, "ulysses_size must equal the world size"
        assert cfg.num_heads % args.ulysses_size == 0, f"{cfg.num_heads=} is not divisible by {args.ulysses_size=}"
        init_distributed_group()
    if dist.is_initialized():
        seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(seed, src=0)
        args.base_seed = seed[0]

    logging.info(f"Generation job args: {args}")
    pipe = build_pipeline(args, cfg, rank, device)
    run_generation(args, pipe, cfg, rank, device, world_size)

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    logging.info("Finished.")


def run_generation(args, pipe, cfg, rank=0, device=0, world_size=1):
    """Generate, save and print bench results with an already-built pipeline (also used by lingbot.worker)."""
    img = Image.open(args.image).convert("RGB")
    logging.info("Generating video ...")
    ready = []
    for rollout in range(args.warmup + args.trials if args.bench_e2e else 1):  # the first --warmup rollouts are untimed
        video = pipe.generate(args.prompt, img, action_path=args.action_path, chunk_size=args.chunk_size,
                              max_area=MAX_AREA_CONFIGS[args.size], frame_num=args.frame_num, shift=args.sample_shift,
                              seed=args.base_seed, offload_model=args.offload_model,
                              max_attention_size=args.max_attention_size)
        if args.bench_e2e and rollout >= args.warmup:
            ready.append(pipe.bench_ready_ms)

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        if args.save_file is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
            args.save_file = os.path.join(args.save_dir, f"lingbot-world-v2_{PRESET}_{args.size}_{world_size}gpu_{prompt}_{stamp}.mp4")
        logging.info(f"Saving generated video to {args.save_file}")
        save_video(tensor=video[None], save_file=args.save_file, fps=cfg.sample_fps, nrow=1, normalize=True,
                   value_range=(-1, 1))
        if args.bench:
            _print_bench_summary(pipe, args, cfg)
        if args.bench_e2e:
            from lingbot.benchmark import format_lines, summarize
            h, w = video.shape[-2:]
            per, med = summarize(ready, args.chunk_size * cfg.vae_stride[0], h, w)
            gpus = world_size + (args.decoder_gpu is not None and args.decoder_gpu != device)
            print("\n".join(format_lines(per, med, PRESET, gpus, h, w, cold=args.warmup == 0)))


if __name__ == "__main__":
    main()
