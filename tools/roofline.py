#!/usr/bin/env python3
"""Roofline of one steady-state chunk, per operation class, the way "How to Scale Your Model" §1 does it:
bytes moved, FLOPs, arithmetic intensity, T_math = FLOPs / peak, T_comms = bytes / bandwidth,
floor = max(T_math, T_comms), measured kernel time, and measured / floor as the fraction of speed of light.

Two passes over the same clip, chunks 8-10 (the KV window is full from chunk 6):
  1. torch.profiler (LINGBOT_PROFILE)          -> kernel time per class, GEMM/conv FLOPs from the profiler
  2. a dispatch-level byte tracer (this file)  -> bytes of every tensor read or written per aten op, per class
Attention FLOPs are analytic (4 * q * kv * heads * head_dim per forward, 30 layers, 5 forwards per chunk):
the profiler does not count custom kernels. Bytes from the tracer are an upper bound on DRAM traffic
(a fused kernel that keeps a tile in shared memory reads less), so the intensities are lower bounds.

  LINGBOT_ROOFLINE=/workspace/roofline_fast python generate.py --preset fast --bench --frame_num 193 ...
  python tools/roofline.py /workspace/roofline_fast   # prints the table, writes roofline.json next to it
"""
import argparse, collections, gzip, json, os, re, sys

PEAKS = {  # RTX 5090, NVIDIA RTX Blackwell whitepaper App. A, dense, FP32 accumulate; GDDR7 512-bit @ 28 Gbps
    "INT8": 838e12, "FP8": 419e12, "FP16": 209.5e12, "BF16": 209.5e12, "TF32": 104.8e12, "FP32": 104.8e12}
BW = 1792e9

# kernel-name buckets (torch.profiler) and aten-op buckets (tracer) map onto the same classes
KCLASS = [
    ("attention", re.compile(r"qk_int|sv_f8|sageattn|attn_kernel|flash_fwd|fmha", re.I)),
    ("attention quant", re.compile(r"QuantInt8|MeanScale|TransposePad|quant_per_block|scale_fuse", re.I)),
    ("decoder conv", re.compile(r"triton_tem_.*convolution|implicit_gemm|cudnn::.*fprop|nchwToNhwc|nhwcToNchw|fprop|convolution", re.I)),
    ("matmul", re.compile(r"e4m3|f8f8|cutlass3x.*f8|sm10_or_later.*gemm|enable_3x_kernel_for_sm10|gemm|cutlass|nvjet|Cijk|sgemm|hgemm|bgemm", re.I)),
    ("memcpy", re.compile(r"^Memcpy|^Memset", re.I)),
    ("elementwise", re.compile(r"^triton_|elementwise|vectorized|reduce|norm|silu|gelu|softmax|copy|cat|fill|index|scatter|gather|unrolled|where|clamp|rope|cub::|upsample|repeat|_to_copy|mul|add|div|sub|pow|rsqrt|exp|sigmoid|to_copy", re.I)),
]
OCLASS = [
    ("attention", re.compile(r"scaled_dot_product|flash|sage|attention", re.I)),
    ("decoder conv", re.compile(r"convolution|conv", re.I)),
    ("matmul", re.compile(r"^(mm|addmm|bmm|baddbmm|matmul|linear|_scaled_mm|scaled_mm)$", re.I)),
    ("memcpy", re.compile(r"^(copy_|clone|contiguous)$", re.I)),
]
VIEWS = {"slice", "expand", "view", "permute", "unsqueeze", "squeeze", "transpose", "alias", "detach", "select", "t",
         "as_strided", "narrow", "reshape", "_unsafe_view", "split", "chunk", "unbind"}


def kclass(name):
    for c, rx in KCLASS:
        if rx.search(name):
            return c
    return "other"


def oclass(name):
    for c, rx in OCLASS:
        if rx.search(name):
            return c
    return "elementwise" if name not in VIEWS else None


class ByteTracer:
    """TorchDispatchMode: bytes of all CUDA tensor inputs + outputs per aten op, bucketed by class."""

    def __init__(self):
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode

        tracer = self

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                c = oclass(str(func.overloadpacket.__name__))
                if c is not None:
                    b = 0
                    for t in torch.utils._pytree.tree_leaves((args, kwargs, out)):
                        if isinstance(t, torch.Tensor) and t.is_cuda:
                            b += t.numel() * t.element_size()
                    tracer.bytes[c] += b
                    tracer.calls[c] += 1
                return out

        self.mode = Mode()
        self.bytes = collections.Counter()
        self.calls = collections.Counter()

    def __enter__(self):
        self.mode.__enter__(); return self

    def __exit__(self, *a):
        self.mode.__exit__(*a)

    def dump(self, path):
        json.dump({"bytes": dict(self.bytes), "calls": dict(self.calls)}, open(path, "w"), indent=1)


def attention_work(q, kv, heads, hd, layers, forwards, q_bytes, kv_bytes, o_bytes):
    """FLOPs and bytes per chunk for dense attention over the KV window: 4·q·kv·h·d per forward; q, k, v read and o written once."""
    f = 4.0 * q * kv * heads * hd * layers * forwards
    b = (q * heads * hd * q_bytes + 2 * kv * heads * hd * kv_bytes + q * heads * hd * o_bytes) * layers * forwards
    return f, b


def matmul_work(shape_rows, chunks):
    """FLOPs and bytes per chunk for the DiT linears from the profiler's per-shape aten records (mm / addmm / _scaled_mm / linear)."""
    f = b = 0.0
    rx = re.compile(r"\[\[(\d+), (\d+)\], \[(\d+), (\d+)\]")
    for e in shape_rows:
        if not re.match(r"aten::(mm|addmm|_scaled_mm|linear|bmm)$", e["name"]):
            continue
        m = rx.search(e["shapes"])
        if not m:
            continue
        a0, a1, b0, b1 = map(int, m.groups())
        if a1 == b0:
            M, K, N = a0, a1, b1
        elif a1 == b1:
            M, K, N = a0, a1, b0
        else:
            continue
        el_in = 1 if e["name"] == "aten::_scaled_mm" else 2
        f += 2.0 * M * K * N * e["count"]
        b += ((M * K + K * N) * el_in + M * N * 2) * e["count"]
    return f / chunks, b / chunks


def inductor_bytes(path, trace_kernels, chunks):
    """Bytes per chunk of Inductor's Triton kernels: bytes per launch from TORCHINDUCTOR_PROFILE (each kernel is
    benchmarked once, bytes = its argument tensors), times launches per chunk from the trace."""
    per_launch = {}
    for line in open(path):
        m = re.search(r"([\d.]+)\s*GB\s.*?([\d.]+)\s*GB/s\s*(\S+)", line)
        if m:
            per_launch[m.group(3).strip()] = float(m.group(1)) * 1e9
    launches = collections.Counter(k["name"] for k in trace_kernels if k["name"].startswith("triton_"))
    total = sum(per_launch.get(n, 0.0) * c for n, c in launches.items())
    missing = sum(c for n, c in launches.items() if n not in per_launch)
    return total / chunks, missing / chunks


DECODER_CONV_FP16 = dict(flops=52.2e12, bytes=39.2e9)   # §10 tracer, fp16 channels-last decoder, per 16-frame chunk


def report(d, chunks, preset):
    def load(p):
        return json.load(gzip.open(p) if p.endswith(".gz") else open(p))

    fast = preset == "fast"
    prec = ({"attention": "INT8", "attention quant": "INT8", "matmul": "FP8", "decoder conv": "FP16", "elementwise": "FP16", "memcpy": "FP16"}
            if fast else {"attention": "BF16", "matmul": "BF16", "decoder conv": "TF32", "elementwise": "FP32", "memcpy": "FP32",
                          "decoder elementwise": "FP32", "decoder memcpy": "FP32"})
    ev = load(os.path.join(d, "trace.json.gz"))["traceEvents"]
    kern = [k for k in ev if k.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    time = collections.Counter(); calls = collections.Counter()
    for k in kern:
        c = kclass(k["name"]); time[c] += k["dur"] / 1e6 / chunks; calls[c] += 1 / chunks
    flops = collections.Counter(); byt = collections.Counter(); src = {}
    q = dict(q=6032, kv=27144, heads=12, hd=128, layers=30, forwards=5)
    flops["attention"], byt["attention"] = attention_work(**q, q_bytes=1 if fast else 2, kv_bytes=1 if fast else 2, o_bytes=2)
    src["attention"] = "FLOPs and bytes analytic from the shapes"
    shape_rows = load(os.path.join(d, "kernels_shapes.json"))
    flops["matmul"], byt["matmul"] = matmul_work(shape_rows, chunks)
    src["matmul"] = "FLOPs and bytes from the profiler's per-shape records"
    vp = os.path.join(d, "trace_vae.json.gz")
    if fast:
        flops["decoder conv"], byt["decoder conv"] = DECODER_CONV_FP16["flops"], DECODER_CONV_FP16["bytes"]
        src["decoder conv"] = "FLOPs and bytes from the §10 tracer of the fp16 decoder"
        if os.path.exists(vp):  # the decoder ran after the loop: its kernels are in the VAE window, not the DiT one
            vc = load(os.path.join(d, "vae_meta.json"))["chunks"]
            time.pop("decoder conv", None)
            for k in load(vp)["traceEvents"]:
                if k.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
                    c = kclass(k["name"]); c = c if c == "decoder conv" else "decoder " + c
                    time[c] += k["dur"] / 1e6 / vc; calls[c] += 1 / vc
    else:
        b = load(os.path.join(d, "bytes.json"))["bytes"]
        for k, v in b.items():
            if k not in ("attention", "matmul"):
                byt[k] += v / chunks
        src["elementwise"] = src["memcpy"] = "bytes from the dispatch tracer"
        if os.path.exists(vp):
            vc = load(os.path.join(d, "vae_meta.json"))["chunks"]
            for k in load(vp)["traceEvents"]:
                if k.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
                    c = kclass(k["name"]); c = c if c == "decoder conv" else "decoder " + c
                    time[c] += k["dur"] / 1e6 / vc; calls[c] += 1 / vc
            bv = load(os.path.join(d, "bytes_vae.json"))["bytes"]
            for k, v in bv.items():
                byt[k if k == "decoder conv" else "decoder " + k] += v / vc
            flops["decoder conv"] = load(os.path.join(d, "flops_vae.json"))["conv_flops"] / vc
            src["decoder conv"] = "FLOPs from the FLOP counter, bytes from the dispatch tracer"
    rt = os.path.join(d, "ref_times.json")
    if os.path.exists(rt):  # kernel times from a reference profile (this pod's card may be clocked differently)
        ref = load(rt); time = collections.Counter(ref["times"])
        for c in time:
            src[c] = src.get(c, "") + f"; time from {ref['source']}"
    rows = []
    for c in sorted(set(time) | set(byt), key=lambda c: -time.get(c, 0)):
        if c == "other" or c.startswith("decoder other"):
            continue
        f, by, t = flops.get(c, 0.0), byt.get(c, 0.0), time.get(c, 0.0)
        pk = PEAKS[prec.get(c, "FP16")]
        tm, tc = (f / pk if f else 0.0), (by / BW if by else 0.0)
        floor = max(tm, tc)
        rows.append(dict(op=c, precision=prec.get(c, "FP16"), flops=f, bytes=by, intensity=(f / by if by else None), ridge=pk / BW,
                         bound=("compute" if tm >= tc else "memory") if floor else "-", t_math=tm, t_comms=tc, floor=floor,
                         measured=t, sol=(floor / t if t and floor else None), calls=calls.get(c, 0.0), source=src.get(c, "time only")))
    return rows


def print_rows(rows):
    print(f"{'op':22s} {'prec':5s} {'FLOP/chunk':>11s} {'GB/chunk':>9s} {'FLOP/B':>7s} {'ridge':>6s} {'bound':7s} {'T_math':>7s} {'T_comms':>8s} {'floor':>6s} {'meas':>6s} {'%SOL':>5s} {'launch':>7s}")
    tot_floor = tot_meas = 0.0
    for r in rows:
        tot_floor += r["floor"]; tot_meas += r["measured"]
        print(f"{r['op']:22s} {r['precision']:5s} {r['flops']/1e12:10.1f}T {r['bytes']/1e9:9.1f} {(r['intensity'] or 0):7.0f} {r['ridge']:6.0f} {r['bound']:7s} "
              f"{r['t_math']:7.3f} {r['t_comms']:8.3f} {r['floor']:6.3f} {r['measured']:6.3f} {100*(r['sol'] or 0):5.0f} {r['calls']:7.0f}")
    print(f"{'chunk':22s} {'':5s} {'':11s} {'':9s} {'':7s} {'':6s} {'':7s} {'':7s} {'':8s} {tot_floor:6.3f} {tot_meas:6.3f} {100*tot_floor/tot_meas if tot_meas else 0:5.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--preset", default="fast", choices=["fast", "stock"])
    a = ap.parse_args()
    rows = report(a.dir, a.chunks, a.preset)
    print_rows(rows)
    for r in rows:
        print(f"  {r['op']}: {r['source']}")
    json.dump({"preset": a.preset, "chunks": a.chunks, "rows": rows}, open(os.path.join(a.dir, "roofline.json"), "w"), indent=1)
