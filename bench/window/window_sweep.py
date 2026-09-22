#!/usr/bin/env python3
"""KV-window sweep harness for the "shorten the attention window" candidate.

Baseline (unchanged): local_attn_size=18, sink_size=6 (generate.py / lingbot/play/live.py
defaults). Both are already plain constructor kwargs / CLI flags on the untouched baseline
(generate.py --local_attn_size / --sink_size) -- this script does not patch any model code,
it only drives generate.py (via `lingbot clip --bench`, run.sh's own path) with different
values and prints the arithmetic prediction to check the measurement against.

IMPORTANT (read before running): kv_size = frame_seqlen * local_attn_size is the WHOLE
attended buffer, not sink + window. sink_size tokens are pinned at the front of that same
buffer; the rolling part is (local_attn_size - sink_size) latents. So "half the window" is
NOT local_attn_size=9 -- with sink_size=6 that leaves only 3 rolling latents against a
4-latent chunk, which is a degenerate configuration (every chunk evicts more than a full
chunk's worth of the rolling region every step; see window_sweep.py's WARNING output and
bench/window/README.md). The default sweep below stays at or above local_attn_size=10
(rolling >= chunk_size) and calls out 9 as a separate, explicitly degenerate probe.

Env overrides (one-variable convention requested for this candidate):
  LINGBOT_KV_WINDOW=<n_latents>   -> --local_attn_size (default: 18, i.e. baseline)
  LINGBOT_KV_SINK=<n_latents>     -> --sink_size        (default: 6,  i.e. baseline)
When either is set, --run acts on that single configuration instead of the sweep list.

No GPU is available in this environment. This script only predicts and prints commands
unless --run is passed, and --run refuses to proceed if CUDA is not visible (it hands back
the commands to copy onto the pod instead of failing silently).
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)  # so `from lingbot.cli import CLIP_DEFAULTS` works regardless of cwd

# ---------------------------------------------------------------------------
# Config constants, verified against the running code (not assumed):
#   frame_seqlen  = lat_h * lat_w // (patch_h * patch_w)   [wan/image2video.py:565, :1081]
#   chunk latents = 4 (lingbot/cli.py chunk_size default; frames_per_chunk = 4 * chunk_size)
#   at max_area=480*832 (lingbot/play/live.py output_size default) this gives frame_seqlen=1508,
#   so q = chunk_size * frame_seqlen = 4 * 1508 = 6032 and, at the baseline local_attn_size=18,
#   kv = frame_seqlen * local_attn_size = 18 * 1508 = 27144 -- both match the measured shapes in
#   OPTIMIZATIONS.md §18/§19 ("q 6032 x kv 27144, 12 heads x 128") exactly, so frame_seqlen=1508
#   is confirmed, not guessed.
FRAME_SEQLEN = 1508
CHUNK_LATENTS = 4
FRAMES_PER_CHUNK = CHUNK_LATENTS * 4  # vae_stride[0] == 4 -> 16 frames/chunk

BASE_WINDOW = 18
BASE_SINK = 6
BASE_KV = FRAME_SEQLEN * BASE_WINDOW  # 27144

# Measured, not assumed: OPTIMIZATIONS.md §18, "Ours (--preset fast)" table, pod 14, chunks 8-10
# (KV window already full). Attention measured 0.288 s of a 0.98 s measured chunk (16.1 FPS
# as-played). h3_l2_working_set.py (§19, hypothesis 3) independently measured attention's
# per-key-per-head cost as flat (6.5 +/- 0.2 ns) from 4k to 54k keys -- i.e. close to linear in
# kv length at this shape family -- which is the basis for scaling BASE_ATTN_S below linearly.
BASE_ATTN_S = 0.288
BASE_CHUNK_S = 0.98
BASE_FPS = 16.1


def predict(window: int, sink: int) -> dict:
    kv = FRAME_SEQLEN * window
    rolling = window - sink
    attn_s = BASE_ATTN_S * (kv / BASE_KV)
    chunk_s = (BASE_CHUNK_S - BASE_ATTN_S) + attn_s
    fps = BASE_FPS * (BASE_CHUNK_S / chunk_s)
    warn = []
    if rolling <= 0:
        warn.append("BROKEN: rolling window <= 0 (sink_size >= local_attn_size); "
                     "kv_ring_plan divides by (kv_cache_size - sink_tokens) -> ZeroDivisionError "
                     "or negative slice under the non-ring eviction path.")
    elif rolling < CHUNK_LATENTS:
        warn.append(f"DEGENERATE: rolling window ({rolling} latents) < chunk size "
                     f"({CHUNK_LATENTS} latents) -- every chunk evicts more than a full chunk's "
                     "worth of the rolling region each step; effectively no cross-chunk memory "
                     "beyond the sink. Expect a real quality hit, not just a smaller window.")
    return dict(window=window, sink=sink, rolling=rolling, kv=kv, attn_s=attn_s,
                chunk_s=chunk_s, fps=fps, warn=warn)


def fmt_row(p: dict) -> str:
    w = f"  [{'; '.join(p['warn'])}]" if p["warn"] else ""
    return (f"local_attn_size={p['window']:>3} sink_size={p['sink']:>2} "
            f"rolling={p['rolling']:>3}lat kv={p['kv']:>6}tok  "
            f"pred_attn={p['attn_s']:.3f}s pred_chunk={p['chunk_s']:.3f}s "
            f"pred_fps={p['fps']:.2f}{w}")


def clip_cmd(window: int, sink: int, frame_num: int, preset: str, seed: int) -> list[str]:
    # Mirrors `lingbot clip` (lingbot/cli.py:cmd_clip), which is itself a thin subprocess
    # wrapper around generate.py with the "lake" example's image/action_path/prompt filled
    # in. Reused directly (no torch import needed for lingbot.cli) so this script does not
    # duplicate example-selection logic, and so the command below is copy-pasteable as-is.
    from lingbot.cli import CLIP_DEFAULTS  # noqa: PLC0415
    return [sys.executable, "generate.py", *CLIP_DEFAULTS,
            "--frame_num", str(frame_num), "--bench",
            "--local_attn_size", str(window), "--sink_size", str(sink),
            "--preset", preset, "--base_seed", str(seed)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", default="18,14,12,10,9",
                     help="comma-separated local_attn_size values to sweep (default includes the "
                          "18-baseline and the degenerate 9-probe; see module docstring)")
    ap.add_argument("--sink", type=int, default=BASE_SINK, help="sink_size held fixed across the sweep")
    ap.add_argument("--frame_num", type=int, default=193, help="rollout length (frames); matches §18's pod-14 clip")
    ap.add_argument("--preset", default="fast")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump_dir", default=None, help="dir for LINGBOT_DUMP_LATENTS per run (quality gate)")
    ap.add_argument("--run", action="store_true", help="actually invoke `lingbot clip --bench` (needs a GPU)")
    args = ap.parse_args()

    env_window = os.environ.get("LINGBOT_KV_WINDOW")
    env_sink = os.environ.get("LINGBOT_KV_SINK")
    if env_window is not None or env_sink is not None:
        windows = [int(env_window) if env_window is not None else BASE_WINDOW]
        sink = int(env_sink) if env_sink is not None else args.sink
    else:
        windows = [int(w) for w in args.windows.split(",")]
        sink = args.sink

    print(f"# frame_seqlen={FRAME_SEQLEN} chunk_latents={CHUNK_LATENTS} "
          f"baseline: local_attn_size={BASE_WINDOW} sink_size={BASE_SINK} kv={BASE_KV} "
          f"attn={BASE_ATTN_S}s chunk={BASE_CHUNK_S}s fps={BASE_FPS} (OPTIMIZATIONS.md §18)\n")

    preds = [predict(w, sink) for w in windows]
    for p in preds:
        print(fmt_row(p))

    print("\n# Commands (run on the pod from the repo root; each is `lingbot clip --bench` with an "
          "override -- no code changes needed, generate.py already exposes --local_attn_size/--sink_size):")
    print("# Step 0 -- noise band, once per stack (bench-world-model-quality skill): run the BASELINE "
          "3x with the same seed before judging any candidate.")
    for i in range(3):
        dump = os.path.join(args.dump_dir, f"baseline_run{i}.pt") if args.dump_dir else None
        cmd = clip_cmd(BASE_WINDOW, BASE_SINK, args.frame_num, args.preset, args.seed)
        env = f"LINGBOT_DUMP_LATENTS={dump} " if dump else ""
        print(f"{env}{shlex.join(cmd)}")

    print("\n# Step 1 -- each candidate, same seed, latents dumped for the quality gate:")
    for p in preds:
        dump = os.path.join(args.dump_dir, f"w{p['window']}_s{p['sink']}.pt") if args.dump_dir else None
        cmd = clip_cmd(p["window"], p["sink"], args.frame_num, args.preset, args.seed)
        env = f"LINGBOT_DUMP_LATENTS={dump} " if dump else ""
        print(f"{env}{shlex.join(cmd)}")

    if not args.run:
        print("\n(--run not passed: commands only, nothing executed. This box has no GPU.)")
        return 0

    try:
        import torch  # noqa: PLC0415
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if not has_cuda:
        print("\nREFUSING to --run: no CUDA device visible in this environment. "
              "Copy the commands above onto the GPU pod instead.", file=sys.stderr)
        return 1

    for p in preds:
        dump = os.path.join(args.dump_dir, f"w{p['window']}_s{p['sink']}.pt") if args.dump_dir else None
        cmd = clip_cmd(p["window"], p["sink"], args.frame_num, args.preset, args.seed)
        env = dict(os.environ)
        if dump:
            env["LINGBOT_DUMP_LATENTS"] = dump
        print(f"\n$ {shlex.join(cmd)}")
        subprocess.run(cmd, cwd=REPO, env=env, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
