"""`lingbot` console script: play (local window), bench, clip.

`bench` and `clip` run generate.py in a subprocess from the repository root with run.sh's
environment, so their output is exactly `./run.sh`'s. `play` builds the pipeline in-process.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# One word per upstream example scene (image.jpg + prompt.txt + intrinsics.npy in examples/NN).
SCENES = {"lake": "examples/03", "wall": "examples/04", "stonehenge": "examples/01", "alley": "examples/02",
          "castle": "examples/00", "dragon": "examples/05"}
EXAMPLE = SCENES["lake"]   # relative to REPO (bench/clip run there; play resolves the defaults below)
PROMPT = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), EXAMPLE, "prompt.txt")).read().strip()
CLIP_DEFAULTS = ["--image", f"{EXAMPLE}/image.jpg", "--action_path", EXAMPLE, "--prompt", PROMPT, "--save_dir", "outputs"]
CKPT_DIR = "weights/lingbot-world-v2-1.3b-causal-fast"
ASSETS_DIR = "weights/lingbot-world-v2-14b-causal-fast"
_R = lambda p: os.path.join(REPO, p)


def _env():
    env = dict(os.environ)
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(REPO, ".inductor_cache"))
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
    return env


def cmd_clip(argv: list[str]) -> int:
    """lingbot.generate with the example's defaults; any of its flags may follow (later flags win)."""
    if argv[:1] in (["-h"], ["--help"]):
        print("usage: lingbot clip [lingbot.generate flags]\n\nDefaults: " + " ".join(CLIP_DEFAULTS[:4] + ["--prompt", "<lakeside prompt>", "--save_dir", "outputs"])
              + "\nExamples:\n  lingbot clip --frame_num 361 --bench\n  lingbot clip --image me.jpg --action_path my_poses/ --prompt \"...\" --preset exact\n")
        return subprocess.call([sys.executable, "-m", "lingbot.generate", "--help"], cwd=REPO, env=_env())
    return subprocess.call([sys.executable, "-m", "lingbot.generate", *CLIP_DEFAULTS, *argv], cwd=REPO, env=_env())


def cmd_bench(argv: list[str]) -> int:
    """`./run.sh --frame_num 361 --bench`: a 22 s clip from examples/03, printing s/chunk and the FPS as played."""
    return cmd_clip(["--frame_num", "361", "--bench", *argv])


def _play_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lingbot play", description="Open a window on the world model and drive it with the keyboard.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("scene", nargs="?", default="lake", choices=sorted(SCENES), help="example scene: its image, prompt and intrinsics")
    p.add_argument("--image", default=None, help="first frame (default: the scene's image.jpg)")
    p.add_argument("--action_path", default=None, help="directory with intrinsics.npy (default: the scene's; poses.npy is replaced by the keys)")
    p.add_argument("--prompt", default=None, help="default: the scene's prompt.txt")
    p.add_argument("--preset", default="fast", choices=["fast", "exact"], help="runtime preset (env defaults); the paper's code has no live hooks")
    p.add_argument("--frame_num", type=int, default=361, help="frames per rollout; the world restarts from the image after that (or on R)")
    p.add_argument("--chunk_size", type=int, default=4, help="latents per chunk (4 = 16 frames = 1 s of input per chunk)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--decoder_gpu", type=int, default=None, help="decode on this GPU, overlapped with the DiT on GPU 0")
    p.add_argument("--ckpt_dir", default=_R(CKPT_DIR))
    p.add_argument("--assets_dir", default=_R(ASSETS_DIR))
    p.add_argument("--input-mode", choices=["history", "hold"], default="history",
                   help="history: the previous chunk period's key edges replay into the 4 slots (taps land in the latent they were "
                        "made in, one period of lag); hold: every slot gets the keys held at the sample (held keys respond from "
                        "frame 0, release overshoots up to one chunk), taps released before the sample are carried into slot 0")
    p.add_argument("--surplus", choices=["rate", "wait"], default="rate",
                   help="source faster than real time: rate = play at the measured arrival rate (up to +25%%, lowest latency); "
                        "wait = nominal playback, the next chunk starts just in time")
    p.add_argument("--trough", type=float, default=2.0, help="playout-queue trough the pacer holds (frames)")
    p.add_argument("--prefill", type=int, default=6, help="frames queued before playout starts")
    p.add_argument("--timing_tsv", default=None, help="write per-chunk timing rows")
    p.add_argument("--headless-seconds", type=float, default=None, metavar="N",
                   help="no display (SDL_VIDEODRIVER=dummy), scripted W taps every 2.5 s, run N s after the warm-up, "
                        "print the HUD stats: the pod check")
    p.add_argument("--dry", action="store_true", help="CPU: DryPipe stand-in for the model, headless, 3 chunks (the CI check)")
    return p


def cmd_play(argv: list[str]) -> int:
    from .play import window
    from .play.control import InputState
    args = _play_parser().parse_args(argv)
    scene = _R(SCENES[args.scene])
    args.image = args.image or os.path.join(scene, "image.jpg")
    args.action_path = args.action_path or scene
    args.prompt = args.prompt or open(os.path.join(scene, "prompt.txt")).read().strip()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    headless = args.dry or args.headless_seconds is not None
    control = InputState(mode=args.input_mode)
    if args.dry:
        from .play.live import DryPipe, LiveSource
        for k, v in {"LINGBOT_VAE_FUSED": "1", "LINGBOT_VAE_STREAM": "1", "LINGBOT_DECODE_FIRST": "1", "LINGBOT_WARM_CHUNKS": "0"}.items():
            os.environ.setdefault(k, v)
        width, height = 832, 464
        src = LiveSource(DryPipe(n_chunks=3, chunk_seconds=0.3, h=height, w=width, decode_first=True), None, args.action_path,
                         args.prompt, frame_num=args.frame_num, chunk_size=args.chunk_size, seed=args.seed,
                         timing_tsv=args.timing_tsv, width=width, height=height, loop=False, control=control)
    else:
        from PIL import Image
        from .play.live import LiveSource, build_pipe, output_size
        os.environ.update(_env())   # run.sh's TORCHINDUCTOR_CACHE_DIR and cuda PATH
        img = Image.open(args.image).convert("RGB")
        width, height = output_size(*img.size)
        t0 = time.monotonic()
        pipe = build_pipe(args.ckpt_dir, args.assets_dir, preset=args.preset, decoder_device_id=args.decoder_gpu)
        logging.info("pipeline built in %.1f s", time.monotonic() - t0)
        src = LiveSource(pipe, img, args.action_path, args.prompt, frame_num=args.frame_num, chunk_size=args.chunk_size,
                         seed=args.seed, timing_tsv=args.timing_tsv, width=width, height=height, loop=True, control=control)
    display = window.open_display(width, height, "lingbot play", headless=headless)
    logging.info("%dx%d @ 16 fps, chunk %d latents, input mode %s, surplus %s%s", width, height, args.chunk_size, args.input_mode,
                 args.surplus, "  [WASD move, QE up/down, arrows look, Shift run, drag mouse to look, R reset, Esc quit]" if not headless else "  [headless: scripted W taps]")
    stats = window.play_loop(src, control, display, prefill=args.prefill, trough=args.trough, surplus=args.surplus,
                             seconds=args.headless_seconds, scripted=headless)
    for line in window.summary_lines(stats):
        print(line)
    if src.error is not None:
        return 1
    if args.dry and not (stats["presented"] > 0 and stats["k2p_onset_ms"]):
        print("PLAY dry check FAILED: no frames presented or no key edge consumed")
        return 1
    return 0


COMMANDS = {"play": cmd_play, "bench": cmd_bench, "clip": cmd_clip}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: lingbot <command> [flags]\n\n"
              "  play   open a window on the world model; WASD / arrows drive it (lingbot play --help)\n"
              "         lingbot play [" + "|".join(sorted(SCENES)) + "]  (default lake)\n"
              "  bench  ./run.sh --frame_num 361 --bench: a 22 s clip from examples/03, s/chunk and FPS as played\n"
              "  clip   offline generation with run.sh's flags (lingbot clip --help)")
        return 0
    cmd = COMMANDS.get(argv[0])
    if cmd is None:
        print(f"lingbot: unknown command {argv[0]!r} (play, bench, clip)", file=sys.stderr)
        return 2
    return cmd(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
