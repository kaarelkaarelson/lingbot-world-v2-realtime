#!/usr/bin/env python3
"""Drive SGLang's realtime LingBot-World endpoint for N chunks and report per-chunk + steady-state timing.

Protocol (sglang v0.5.17, runtime/entrypoints/openai/realtime/realtime_video_api.py + realtime_output_adapter.py):
  client -> server : one msgpack map {"type": "init", ...RealtimeVideoGenerationsRequest fields}
  server -> client : per chunk, for raw output: a msgpack "frame_batch_header" map followed by one binary message
                     holding num_frames * bytes_per_frame RGB24 bytes (payloads >= 64 KB are sent this way; smaller ones
                     arrive inline as a "frame_batch" map with a "payload" key), then a msgpack "chunk_stats" map.
                     {"type": "error", "content": ...} on failure. The server closes the socket (code 1000) after
                     max_chunks.
  Camera control  : condition_inputs.camera_actions = list[list[str]], one entry per LATENT frame (chunk = 4 latents),
                    keys in {w,a,s,d,i,k,j,l}; the script is consumed in order and padded with [] when it runs out.

Timing fields (all measured server-side, ms):
  request_prepare_ms   sampling of the camera script + Req build
  scheduler_forward_ms the whole chunk through the pipeline: cached T5 + 4 DiT steps + streaming Wan-VAE decode
                       (this is the number comparable to our "per-chunk end-to-end")
  chunk_total_ms       scheduler_forward + raw payload build + WebSocket write
Client-side:
  arrival_ms           wall-clock between the final frame batch of consecutive chunks, as seen by this client
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time
import urllib.request

import imageio.v2 as imageio
import msgspec.msgpack
import numpy as np
import websockets

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_RGB_CONTENT_TYPE = "application/x-raw-rgb"
PROMPT = (
    "A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped mountains "
    "under a bright blue sky with drifting white clouds."
)
FIRST_CHUNK_TIMEOUT_S = 1800.0  # chunk 0 pays T5 encode (CPU-offloaded) + flashinfer/Triton JIT + SDPA autotune
CHUNK_TIMEOUT_S = 300.0


def camera_script(num_chunks: int, chunk_size: int) -> list[list[str]]:
    """Fixed per-latent-frame key script: forward, a right turn, forward, a left turn, forward."""
    segments = [(6, ["w"]), (4, ["w", "l"]), (5, ["w"]), (4, ["w", "j"])]
    script: list[list[str]] = []
    for chunks, keys in segments:
        script += [list(keys) for _ in range(chunks * chunk_size)]
    script += [["w"] for _ in range(num_chunks * chunk_size - len(script))]
    return script[: num_chunks * chunk_size]


def http_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as r:
        return msgspec.json.decode(r.read())


def decode_raw_frames(header: dict, payload: bytes) -> list[np.ndarray]:
    if header.get("content_type") != RAW_RGB_CONTENT_TYPE:
        raise RuntimeError(f"unexpected content_type {header.get('content_type')!r}; init with realtime_output_format=raw")
    w, h, c = int(header["width"]), int(header["height"]), int(header["channels"])
    n, bpf = int(header["num_frames"]), int(header["bytes_per_frame"])
    if len(payload) != n * bpf:
        raise RuntimeError(f"payload size {len(payload)} != {n} * {bpf}")
    buf = np.frombuffer(payload, dtype=np.uint8).reshape(n, h, w, c)
    return [buf[i, :, :, :3].copy() for i in range(n)]


async def run(args: argparse.Namespace) -> None:
    base = f"http://{args.host}:{args.port}"
    ws_url = f"ws://{args.host}:{args.port}/v1/realtime_video/generate"
    server_info = http_json(f"{base}/server_info")
    models = http_json(f"{base}/models")
    print(f"SGL server version={server_info.get('version')} pipeline_class={models.get('pipeline_class')} "
          f"dit_precision={models.get('dit_precision')} vae_precision={models.get('vae_precision')} "
          f"vae_decode_precision={models.get('vae_decode_precision')} model_path={models.get('model_path')}")

    chunk_size = args.chunk_size
    actions = camera_script(args.chunks, chunk_size)
    with open(args.image, "rb") as f:
        first_frame = f.read()
    init = {
        "type": "init",
        "prompt": args.prompt,
        "first_frame": first_frame,
        "size": f"{args.width}x{args.height}",
        "fps": args.fps,
        # condition length only; the adapter floors it to (2 * chunk_size - 1) * 4 + 1 = 29 for chunk_size 4
        "num_frames": (2 * chunk_size - 1) * 4 + 1,
        "num_inference_steps": args.steps,
        "guidance_scale": 1.0,
        "seed": args.seed,
        "max_chunks": args.chunks,
        "realtime_causal_sink_size": args.sink,
        "realtime_causal_kv_cache_num_frames": args.window,
        "realtime_output_format": "raw",
        "realtime_output_pacing": False,  # send chunks as soon as they are generated; arrival_ms then tracks generation
        "condition_inputs": {"camera_actions": actions},
    }
    print(f"SGL config size={args.width}x{args.height} steps={args.steps} chunk={chunk_size} sink={args.sink} "
          f"window={args.window} seed={args.seed} chunks={args.chunks} fps={args.fps} output=raw pacing=off "
          f"actions={len(actions)} image={os.path.basename(args.image)}")

    all_frames: list[np.ndarray] = []  # kept in RAM (365 x 832x464x3 = ~0.4 GB) and encoded after the session,
    stats: dict[int, dict] = {}         # so x264 never sits in the receive loop and skews arrival_ms
    final_arrival: dict[int, float] = {}
    frames_per_chunk: dict[int, int] = {}
    total_frames = 0
    t_init = time.perf_counter()
    t_first_frame = None

    async with websockets.connect(ws_url, max_size=None, ping_interval=None) as ws:
        await ws.send(msgspec.msgpack.encode(init))
        while len(stats) < args.chunks or len(final_arrival) < args.chunks:
            timeout = FIRST_CHUNK_TIMEOUT_S if not final_arrival else CHUNK_TIMEOUT_S
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except websockets.exceptions.ConnectionClosedOK:
                break
            header = msgspec.msgpack.decode(msg)
            kind = header.get("type")
            if kind == "error":
                raise SystemExit(f"SGL error from server: {header.get('content')}")
            if kind == "chunk_stats":
                header["client_recv_s"] = time.perf_counter() - t_init
                stats[int(header["chunk_index"])] = header
                continue
            if kind == "frame_batch":
                payload = header.pop("payload")
            elif kind == "frame_batch_header":
                payload = await asyncio.wait_for(ws.recv(), timeout=timeout)
            else:
                raise RuntimeError(f"unexpected message: {header}")
            if not isinstance(payload, (bytes, bytearray)):
                raise RuntimeError("frame payload is not binary")
            frames = decode_raw_frames(header, bytes(payload))
            now = time.perf_counter()
            if t_first_frame is None:
                t_first_frame = now
            all_frames.extend(frames)
            idx = int(header["chunk_index"])
            frames_per_chunk[idx] = frames_per_chunk.get(idx, 0) + len(frames)
            total_frames += len(frames)
            if header.get("is_final_frame_batch", True):
                final_arrival[idx] = now
    t_end = time.perf_counter()

    with imageio.get_writer(args.out, fps=args.fps, codec="libx264", pixelformat="yuv420p", quality=None,
                            ffmpeg_params=["-crf", "12", "-preset", "medium"]) as writer:
        for fr in all_frames:
            writer.append_data(fr)

    if len(stats) < args.chunks or len(final_arrival) < args.chunks:
        raise SystemExit(f"SGL incomplete: chunk_stats={len(stats)} frame_chunks={len(final_arrival)} of {args.chunks}")

    print(f"SGL ttff_s={t_first_frame - t_init:.2f}  (init sent -> first frame bytes received)")
    prev = None
    for idx in sorted(stats):
        s = stats[idx]
        arrival = (final_arrival[idx] - prev) * 1000 if prev is not None else float("nan")
        prev = final_arrival[idx]
        print(f"SGL chunk idx={idx:2d} frames={frames_per_chunk.get(idx, 0):2d} prepare_ms={s['request_prepare_ms']:4d} "
              f"forward_ms={s['scheduler_forward_ms']:5d} total_ms={s['chunk_total_ms']:5d} "
              f"ws_write_ms={s['ws_write_ms']:3d} payload_mb={s['ws_payload_bytes'] / 2**20:5.1f} arrival_ms={arrival:7.0f}")

    steady = [idx for idx in sorted(stats) if idx >= args.steady_from]
    fwd = [stats[i]["scheduler_forward_ms"] / 1000 for i in steady]
    tot = [stats[i]["chunk_total_ms"] / 1000 for i in steady]
    arr = [(final_arrival[i] - final_arrival[i - 1]) for i in steady if i - 1 in final_arrival]
    frames_steady = statistics.median(frames_per_chunk[i] for i in steady)
    med_fwd, med_tot, med_arr = statistics.median(fwd), statistics.median(tot), statistics.median(arr)
    p95_fwd = sorted(fwd)[min(len(fwd) - 1, int(len(fwd) * 0.95))]
    print(f"SGL steady chunks>={args.steady_from} n={len(steady)} frames/chunk={frames_steady:.0f} "
          f"median_forward_s={med_fwd:.3f} fps_forward={frames_steady / med_fwd:.2f} p95_forward_s={p95_fwd:.3f} "
          f"median_total_s={med_tot:.3f} fps_total={frames_steady / med_tot:.2f} "
          f"median_arrival_s={med_arr:.3f} fps_arrival={frames_steady / med_arr:.2f}")
    wall = t_end - t_init
    print(f"SGL wall total_s={wall:.1f} frames={total_frames} fps_wall={total_frames / wall:.2f} "
          f"chunk0_frames={frames_per_chunk.get(0)} out={args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--image", default=os.path.join(HERE, "first_frame.jpg"),
                    help="first frame; SGLang resizes it to --width x --height without cropping")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=464)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sink", type=int, default=6, help="realtime_causal_sink_size (latent frames)")
    ap.add_argument("--window", type=int, default=18, help="realtime_causal_kv_cache_num_frames (latent frames)")
    ap.add_argument("--chunk-size", type=int, default=4,
                    help="latent frames per chunk; must equal num_frames_per_block in transformer/config.json")
    ap.add_argument("--chunks", type=int, default=23, help="23 chunks = 13 + 22 * 16 = 365 frames")
    ap.add_argument("--steady-from", type=int, default=6, help="first chunk index counted as steady state")
    ap.add_argument("--out", default="/workspace/sgl/out.mp4")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
