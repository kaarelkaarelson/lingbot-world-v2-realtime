"""LiveSource: the real model as a frame source for the window.

Runs `LingBotWorldPipeline.generate` in a background thread with the `fast` preset and receives each
decoded latent through the `frame_sink` hook in lingbot/pipelines/lingbot_world.py: the hook enqueues, on the VAE
side stream, the float->uint8 HWC conversion and a non-blocking copy into a pinned host buffer,
records one CUDA event per call and returns. A host thread waits on the event (blocking only
itself), stamps `t_ready`, hands the frames to the ChunkQueue and returns the pinned buffer to the
pool. Nothing on the consumer side touches torch.

In play mode the camera comes from `lingbot.play.control.InputState` through `pipe.pose_provider`
(sampled at each chunk start) instead of the example's poses.npy; `frame_num` then sets the rollout
length and `loop=True` starts the next rollout in the same warm process.
"""
from __future__ import annotations

import logging
import math
import os
import queue
import threading
import time
from typing import Iterator

import numpy as np
import torch

from .control import RolloutReset
from .source import ChunkQueue, Frame

log = logging.getLogger("lingbot.play.live")

REQUIRED_ENV = {"LINGBOT_VAE_FUSED": None, "LINGBOT_VAE_STREAM": "1"}
WARM_CHUNKS = 6  # chunk index at which the source counts as warm (compiles happen at chunks 0, 1, 2, 5)
# on top of the `fast` preset: per-chunk decode on a side stream (needed by frame_sink) and per-latent
# emission right after x0, so the first frame of a chunk leaves ~0.5 s earlier (stream OPTIMIZATIONS.md)
PLAY_ENV = {"LINGBOT_VAE_STREAM": "1", "LINGBOT_DECODE_FIRST": "1"}


def to_uint8_hwc(fr):
    """[C,F,H,W] float in [-1,1] -> [F,H,W,C] uint8 (same rounding as save_video: (x+1)*127.5)."""
    return fr.permute(1, 2, 3, 0).add(1).mul(127.5).round_().clamp_(0, 255).to(torch.uint8)


def output_size(img_w: int, img_h: int, max_area: int = 480 * 832, vae_stride: int = 8, patch: int = 2) -> tuple[int, int]:
    """(width, height) of the generated frames for an input image, as LingBotWorldPipeline computes it."""
    ar = img_h / img_w
    lat_h = round(math.sqrt(max_area * ar) // vae_stride // patch * patch)
    lat_w = round(math.sqrt(max_area / ar) // vae_stride // patch * patch)
    return lat_w * vae_stride, lat_h * vae_stride


class _PinnedPool:
    def __init__(self, n, shape, torch):
        self.torch = torch
        self.free: queue.Queue = queue.Queue()
        for _ in range(n):
            self.free.put(torch.empty(shape, dtype=torch.uint8, pin_memory=torch.cuda.is_available()))

    def get(self):
        try:
            return self.free.get_nowait()
        except queue.Empty:
            return None

    def put(self, buf):
        self.free.put(buf)


class LiveSource:
    fps = 16
    width = 832
    height = 464
    frames_per_chunk = 16   # set per instance: 4 * chunk_size

    def __init__(self, pipe, img, action_path: str, prompt: str, frame_num: int = 361,
                 chunk_size: int = 4, max_area: int = 480 * 832, shift: float = 5.0, seed: int = 42,
                 pool_chunks: int = 4, max_chunks_queued: int = 2, timing_tsv: str | None = None,
                 width: int = 832, height: int = 464, loop: bool = False, control=None):
        """pipe: a LingBotWorldPipeline (see build_pipe) with the `frame_sink` hook.
        img: PIL image. action_path: dir with poses.npy/intrinsics.npy."""
        self.torch = torch
        self.width, self.height = width, height
        for k, v in REQUIRED_ENV.items():
            if os.environ.get(k) in (None, "") or (v is not None and os.environ.get(k) != v):
                raise RuntimeError(f"LiveSource needs {k}={v or '<set>'} in the environment before the pipeline is built")
        self.pipe, self.img, self.action_path, self.prompt = pipe, img, action_path, prompt
        self.chunk_size = chunk_size
        self.frames_per_chunk = 4 * chunk_size           # chunk 0 has 4 * (chunk_size - 1) + 1
        self.gen_kwargs = dict(chunk_size=chunk_size, max_area=max_area, frame_num=frame_num,
                               shift=shift, seed=seed, offload_model=False)
        # queue unit = one sink call: one latent (4 frames) under LINGBOT_DECODE_FIRST, else a chunk,
        # so size it in latents to keep `max_chunks_queued` chunks either way
        decode_first = os.environ.get("LINGBOT_DECODE_FIRST") == "1"
        self.queue = ChunkQueue(max_chunks_queued * (chunk_size if decode_first else 1))
        # pool: one buffer per sink call in flight; per-latent emission makes chunk_size calls per chunk,
        # so allocate that many whole-chunk buffers (the env may differ from the pipe in tests)
        self.pool = _PinnedPool(pool_chunks * chunk_size, (self.frames_per_chunk, self.height, self.width, 3), torch)
        self._pending: queue.Queue = queue.Queue()
        self.dropped_at_sink = 0
        self.timing_rows: list[dict] = []
        self._timing_f = open(timing_tsv, "w") if timing_tsv else None
        if self._timing_f:
            self._timing_f.write("chunk\tgen_start\tdit_end\tdecode_end\ts_per_chunk\tframes\n")
        self._last_t0 = None
        self.error: BaseException | None = None
        # warm-up: the compiled graph variants (append / overwrite / evict) exist once the KV window has
        # filled; the last variant (eviction) first runs at chunk 18 // chunk_size (window of 18 latents):
        # 4 -> chunk 4, 3 -> 6, 2 -> 9, 1 -> 18; ready two chunks after it so its stall stays hidden
        self.warm_chunks = int(os.environ.get("LINGBOT_WARM_CHUNKS", str(max(WARM_CHUNKS, 18 // chunk_size + 2))))
        self.ready = threading.Event()
        self.warm_progress = 0      # chunks seen during warm-up (for the HUD's progress text)
        # loop=True: start the next rollout when one ends (same warm process, seed+1 per loop);
        # chunk ids keep counting across rollouts so frame_id stays monotonic for the presenter.
        self.loop, self.rollouts = loop, 0
        self._chunk_base, self._rollout_chunks = 0, 0
        self._closed = threading.Event()
        pipe.frame_sink = self._sink
        # control: lingbot.play.control.InputState -> the camera comes from the keys instead of poses.npy
        self.control = control
        pipe.pose_provider = None
        if control is not None:
            from .control import make_pose_provider
            control.slots = chunk_size   # one action slot per latent
            pipe.pose_provider = make_pose_provider(control)
        # throttle: set by the presenter (--surplus wait); called on the generation thread before each
        # chunk and may block, so a source faster than real time generates just in time instead of
        # ahead of the player (no frame drops, no fast playback)
        self.throttle = None
        pipe.chunk_gate = self._chunk_gate
        self.frames_launched = 0    # handed to the sink (queued on the GPU)
        self.frames_delivered = 0   # copied to the host and queued for the presenter
        self._host = threading.Thread(target=self._host_loop, name="live-host", daemon=True)
        self._gen = threading.Thread(target=self._gen_loop, name="live-gen", daemon=True)
        self._host.start()
        self._gen.start()

    # --- generation thread -------------------------------------------------
    def _gen_loop(self):
        try:
            while True:
                kw = dict(self.gen_kwargs, seed=self.gen_kwargs["seed"] + self.rollouts)
                try:
                    self.pipe.generate(self.prompt, self.img, action_path=self.action_path, **kw)
                except RolloutReset:
                    if self._closed.is_set():  # close(): unwound at the chunk gate
                        break
                    log.info("rollout %d reset by the player", self.rollouts)
                self.rollouts += 1
                self._chunk_base += self._rollout_chunks
                self._rollout_chunks = 0
                if self.control is not None:
                    self.control.chunk_offset = self._chunk_base
                if not self.loop or self._closed.is_set():
                    break
                log.info("rollout %d done; looping (seed %d)", self.rollouts, kw["seed"] + 1)
        except BaseException as e:  # surfaced to the consumer via .error; the queue is closed either way
            log.exception("generation thread died")
            self.error = e
        finally:
            self._pending.put(None)

    @property
    def gpu_inflight(self) -> int:
        """Frames launched on the GPU but not yet on the host (the previous chunk's later latents)."""
        return self.frames_launched - self.frames_delivered

    def _chunk_gate(self, chunk_id):
        if self.throttle is not None and self.ready.is_set() and not self._closed.is_set():
            self.throttle()
        if self._closed.is_set():
            # close() is waiting to join this thread: leave generate() before the next chunk starts, so the
            # thread ends while the interpreter is still whole (a generation thread alive at exit dies inside
            # torch's static teardown: "terminate called without an active exception")
            raise RolloutReset()

    def _sink(self, chunk_id, fr, stream, chunk_gen_start, frame_offset=0):
        """Called on the generation thread with `fr` [C,F,H,W] queued on `stream`. Must not block.
        frame_offset: index of fr's first frame within the chunk (per-latent emission hands 4 at a time)."""
        torch = self.torch
        t_call = time.monotonic()
        n = fr.shape[1]
        self._rollout_chunks = max(self._rollout_chunks, chunk_id + 1)
        chunk_id += self._chunk_base
        self.frames_launched += n
        buf = self.pool.get()
        if buf is None:
            self.dropped_at_sink += 1
            log.warning("chunk %d dropped: no free pinned buffer (consumer behind)", chunk_id)
            return
        if fr.is_cuda:
            with torch.cuda.stream(stream):
                u8 = to_uint8_hwc(fr)
                buf[:n].copy_(u8, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(stream)
        else:  # dry mode on CPU
            buf[:n].copy_(to_uint8_hwc(fr))
            ev = None
        self._pending.put((chunk_id, n, buf, ev, chunk_gen_start, t_call, frame_offset))

    # --- host thread -------------------------------------------------------
    def _host_loop(self):
        while True:
            item = self._pending.get()
            if item is None:
                break
            chunk_id, n, buf, ev, t0, t_call, off = item
            if ev is not None:
                ev.synchronize()
            t_ready = time.monotonic()
            arr = buf[:n].numpy()
            frames = [Frame(chunk=chunk_id, idx=off + i, t_ready=t_ready, chunk_gen_start=t0, rgb=arr[i].copy(),
                            stride=self.frames_per_chunk)
                      for i in range(n)]
            self.pool.put(buf)
            # ready is set before the chunk is queued: the presenter's first admitted frame must be this
            # chunk's first (the log line below releases the GIL, so the order matters)
            if off == 0 and not self.ready.is_set():
                self.warm_progress = chunk_id + 1
                if chunk_id >= self.warm_chunks:
                    self.ready.set()
                    log.info("warm-up done at chunk %d (%.1f s since first chunk)", chunk_id,
                             t_ready - self.timing_rows[0]["decode_end"] if self.timing_rows else 0.0)
            self.queue.put_chunk(frames)
            self.frames_delivered += n
            if off != 0:  # per-latent emission: one timing row per chunk, stamped at its first latent
                continue
            row = dict(chunk=chunk_id, gen_start=t0, dit_end=t_call, decode_end=t_ready,
                       s_per_chunk=(t0 - self._last_t0) if self._last_t0 is not None else float("nan"), frames=n)
            self._last_t0 = t0
            self.timing_rows.append(row)
            if self._timing_f:
                self._timing_f.write("\t".join(str(row[k]) for k in ("chunk", "gen_start", "dit_end", "decode_end", "s_per_chunk", "frames")) + "\n")
                self._timing_f.flush()
        self.queue.close()

    # --- frame source ------------------------------------------------------
    def frames(self) -> Iterator[Frame]:
        return self.queue.frames()

    @property
    def dropped_chunks(self) -> int:
        return self.queue.dropped_chunks + self.dropped_at_sink

    def close(self, timeout: float = 10.0) -> None:
        """Stop generation at the next chunk gate and join the threads (at most `timeout` s each; a compile
        in flight can hold the generation thread longer, in which case it is left running and reported)."""
        self._closed.set()
        self._gen.join(timeout)
        if self._gen.is_alive():
            log.warning("generation thread still busy %.0f s after close (compile in flight?); not joined", timeout)
        self._host.join(timeout)
        self.queue.close()
        # the host thread writes timing rows until it has drained the sink; the file is closed only once it is done
        if self._timing_f and not self._host.is_alive():
            self._timing_f.close()


def build_pipe(ckpt_dir: str, assets_dir: str | None, preset: str = "fast", local_attn_size: int = 18,
               sink_size: int = 6, device_id: int = 0, decoder_device_id: int | None = None):
    """lingbot.generate's pipeline (single process, rank 0) with the preset's env plus PLAY_ENV, applied
    before the pipeline is imported (the attention backend is chosen at import time)."""
    from ..presets import apply_preset
    from ..registry import DEFAULT_MODEL, MODELS, pipeline_class
    apply_preset(preset)
    for k, v in PLAY_ENV.items():
        os.environ.setdefault(k, v)
    from wan.configs import WAN_CONFIGS
    cfg = WAN_CONFIGS[MODELS[DEFAULT_MODEL]["task"]]
    return pipeline_class(DEFAULT_MODEL)(config=cfg, checkpoint_dir=ckpt_dir, device_id=device_id, rank=0,
                                         local_attn_size=local_attn_size, sink_size=sink_size, assets_dir=assets_dir,
                                         decoder_device_id=decoder_device_id)


class DryPipe:
    """Stand-in for LingBotWorldPipeline on CPU: drives frame_sink with the real call
    pattern (chunk N's frames handed over at the start of chunk N+1, the last
    one after the loop) so LiveSource's plumbing runs end-to-end without a GPU."""

    def __init__(self, n_chunks=3, chunk_seconds=0.05, h=464, w=832, decode_first=False,
                 boundary_seconds=0.0, ops_per_chunk=0):
        self.frame_sink = None
        self.pose_provider = None
        self.chunk_gate = None
        self.poses = []  # what the provider returned per chunk (tests)
        self.decode_first = decode_first  # LINGBOT_DECODE_FIRST: per-latent emission at the end of the chunk
        self.n_chunks, self.chunk_seconds, self.h, self.w = n_chunks, chunk_seconds, h, w
        # boundary_seconds: host-side setup time at the start of each rollout (the real pipeline: 0.5-0.9 s).
        # ops_per_chunk: small torch ops per chunk, each releasing and reacquiring the GIL like the pipeline's
        # kernel launches do, so a presenter that busy-spins shows up here as it does on the GPU.
        self.boundary_seconds, self.ops_per_chunk = boundary_seconds, ops_per_chunk

    def generate(self, prompt, img, action_path, chunk_size=4, **kw):
        t0s, pending = [], None
        time.sleep(self.boundary_seconds)
        for c in range(self.n_chunks):
            if getattr(self, "chunk_gate", None) is not None:
                self.chunk_gate(c)
            t0s.append(time.monotonic())
            x = torch.zeros(8)
            for _ in range(self.ops_per_chunk):
                x = x + 1
            if pending is not None:
                self.frame_sink(c - 1, pending, None, t0s[c - 1])
            if self.pose_provider is not None:  # same point as the real loop: before the chunk's denoise
                self.poses.append(self.pose_provider(c, chunk_size))
            time.sleep(self.chunk_seconds)
            n = 4 * (chunk_size - 1) + 1 if c == 0 else chunk_size * 4
            frames = torch.rand(3, n, self.h, self.w) * 2 - 1
            if self.decode_first:
                off = 0
                for li in range(chunk_size):
                    k = 1 if (c == 0 and li == 0) else 4
                    self.frame_sink(c, frames[:, off:off + k], None, t0s[c], off)
                    off += k
            else:
                pending = frames
        if pending is not None:
            self.frame_sink(self.n_chunks - 1, pending, None, t0s[-1])
        return None
