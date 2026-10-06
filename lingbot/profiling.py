"""torch.profiler windows over the generation loop, for timing splits and the roofline tables.

LINGBOT_PROFILE=<dir>          profile steady-state chunks (LINGBOT_PROFILE_CHUNKS=8-10, 1-based)
LINGBOT_PROFILE_VAE=1          also profile the whole-clip decode
LINGBOT_ROOFLINE=<dir>         both of the above, into <dir>
LINGBOT_ROOFLINE_TRACE=1       add tools/roofline.py's byte tracer and a decoder FLOP count (eager only)
LINGBOT_PROFILE_CHUNKS=a:b     cudaProfilerStart before chunk a, Stop after chunk b-1 (0-based, b exclusive) on
                               every run; for `nsys --capture-range=cudaProfilerApi` (LINGBOT_WORKER_NSYS=1).
                               With LINGBOT_PROFILE the "8-10" form of the torch.profiler window applies instead
LINGBOT_NVTX=1                 NVTX ranges: chunk, decode, fwd, per layer and its stages (no-op when unset)
"""
import contextlib
import json
import logging
import os
import sys

import torch

NVTX = os.environ.get("LINGBOT_NVTX") == "1"
_NULL = contextlib.nullcontext()
nvtx = torch.cuda.nvtx.range if NVTX else (lambda name: _NULL)
nvtx_push = torch.cuda.nvtx.range_push if NVTX else (lambda name: None)
nvtx_pop = torch.cuda.nvtx.range_pop if NVTX else (lambda: None)

_TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")


def _byte_tracer():
    sys.path.insert(0, _TOOLS)
    from roofline import ByteTracer
    return ByteTracer()


class ChunkProfiler:
    """Opens a profiler before chunk `lo` and writes kernels/trace after chunk `hi`."""

    def __init__(self):
        if os.environ.get("LINGBOT_ROOFLINE"):
            os.environ.setdefault("LINGBOT_PROFILE", os.environ["LINGBOT_ROOFLINE"])
            os.environ.setdefault("LINGBOT_PROFILE_VAE", "1")
        self.dir = os.environ.get("LINGBOT_PROFILE")
        self.prof = self.tracer = None
        self.lo = self.hi = -1
        if self.dir:
            lo, hi = os.environ.get("LINGBOT_PROFILE_CHUNKS", "8-10").split("-")
            self.lo, self.hi = int(lo) - 1, int(hi) - 1

    def before_chunk(self, chunk_id):
        if not self.dir or chunk_id != self.lo:
            return
        torch.cuda.synchronize()
        self.prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, with_flops=True)
        self.prof.__enter__()
        if os.environ.get("LINGBOT_ROOFLINE_TRACE") == "1":
            self.tracer = _byte_tracer()
            self.tracer.__enter__()

    def after_chunk(self, chunk_id):
        if self.prof is None or chunk_id != self.hi:
            return
        torch.cuda.synchronize()
        os.makedirs(self.dir, exist_ok=True)
        if self.tracer is not None:
            self.tracer.__exit__(None, None, None)
            self.tracer.dump(os.path.join(self.dir, "bytes.json"))
            self.tracer = None
        self.prof.__exit__(None, None, None)
        self.prof.export_chrome_trace(os.path.join(self.dir, "trace.json.gz"))
        ka = self.prof.key_averages()
        with open(os.path.join(self.dir, "kernels_top.txt"), "w") as f:
            f.write(ka.table(sort_by="self_device_time_total", row_limit=80))
        rows = [{"name": e.key, "self_cuda_us": e.self_device_time_total, "cuda_us": e.device_time_total,
                 "cpu_us": e.self_cpu_time_total, "count": e.count, "flops": getattr(e, "flops", 0) or 0,
                 "shapes": str(getattr(e, "input_shapes", ""))[:200]} for e in ka]
        with open(os.path.join(self.dir, "kernels.json"), "w") as f:
            json.dump({"chunks": [self.lo + 1, self.hi + 1], "events": rows}, f)
        kas = self.prof.key_averages(group_by_input_shape=True)
        with open(os.path.join(self.dir, "kernels_shapes.json"), "w") as f:
            json.dump([{"name": e.key, "count": e.count, "cuda_us": e.device_time_total, "shapes": str(e.input_shapes)[:400]}
                       for e in kas if e.input_shapes], f)
        logging.info(f"BENCH profile written to {self.dir} (chunks {self.lo + 1}-{self.hi + 1})")
        self.prof = None


class DecodeProfiler:
    """Profiles the whole-clip decode when LINGBOT_PROFILE and LINGBOT_PROFILE_VAE=1 are set."""

    def __init__(self, enabled):
        self.dir = os.environ.get("LINGBOT_PROFILE")
        self.on = enabled and bool(self.dir) and os.environ.get("LINGBOT_PROFILE_VAE") == "1"
        self.prof = self.tracer = self.flops = None

    def __enter__(self):
        if not self.on:
            return self
        self.prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
        self.prof.__enter__()
        if os.environ.get("LINGBOT_ROOFLINE_TRACE") == "1":
            from torch.utils.flop_counter import FlopCounterMode
            self.flops = FlopCounterMode(display=False)
            self.flops.__enter__()
            self.tracer = _byte_tracer()
            self.tracer.__enter__()
        return self

    def finish(self, num_chunks):
        if not self.on:
            return
        torch.cuda.synchronize()
        os.makedirs(self.dir, exist_ok=True)
        if self.tracer is not None:
            self.tracer.__exit__(None, None, None)
            self.tracer.dump(os.path.join(self.dir, "bytes_vae.json"))
            self.flops.__exit__(None, None, None)
            conv = sum(v for k, v in self.flops.get_flop_counts()["Global"].items() if "conv" in str(k))
            with open(os.path.join(self.dir, "flops_vae.json"), "w") as f:
                json.dump({"conv_flops": conv, "total_flops": self.flops.get_total_flops()}, f)
        with open(os.path.join(self.dir, "vae_meta.json"), "w") as f:
            json.dump({"chunks": num_chunks}, f)
        self.prof.__exit__(None, None, None)
        self.prof.export_chrome_trace(os.path.join(self.dir, "trace_vae.json.gz"))

    def __exit__(self, *exc):
        return False


class CudaProfilerRange:
    """cudaProfilerStart before chunk a, cudaProfilerStop once chunk b-1 has finished on every GPU."""

    def __init__(self, n_chunks):
        spec = os.environ.get("LINGBOT_PROFILE_CHUNKS", "")
        self.lo = self.hi = -1
        if ":" in spec:
            lo, hi = spec.split(":")
            self.lo, self.hi = int(lo), min(int(hi), n_chunks) - 1

    def _sync(self):
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)

    def before_chunk(self, chunk_id):
        if chunk_id == self.lo:
            self._sync()
            torch.cuda.cudart().cudaProfilerStart()

    def after_chunk(self, chunk_id):
        if chunk_id == self.hi:
            self._sync()
            torch.cuda.cudart().cudaProfilerStop()
