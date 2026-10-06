"""Does the decoder fit on card 1 next to its 25 % share of the DiT? Measures the two interference rates.

  1. decoder alone on card 1: kernels and GPU time per chunk (4 latents -> 16 frames)
  2. 9:3 split alone: steady ms per layer
  3. both at once (decoder queued first on its own stream, so it runs through the whole window): profiled;
     layer period with the decoder running, and the fraction of a decoder chunk completed in the window
     (decoder kernels finished / kernels per chunk)
Then the chunk time: 150 layer passes and one decoded chunk, with the decoder running alongside until done.

  PIECES=4 python experiments/overlap_layer/with_decoder.py
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import real_layer as R  # noqa: E402
from lingbot.models.lingbot_world.vae import FusedDecoder  # noqa: E402
from wan.modules.vae2_1 import Wan2_1_VAE  # noqa: E402

torch.set_grad_enabled(False)
T, D, H, HD, WIN, NL = R.T, R.D, R.H, R.HD, R.WIN, 12
PASSES = 150
VAE = "weights/lingbot-world-v2-14b-causal-fast/Wan2.1_VAE.pth"

# decoder on card 1, compiled as in the pipeline
dec = FusedDecoder(Wan2_1_VAE(vae_pth=VAE, dtype=torch.float, device=R.dev[1]), compile_mode="max-autotune-no-cudagraphs")
# HIPRIO=1: the DiT on the highest-priority streams, the decoder on the lowest (needs a SageAttention build that
# launches on the current stream; the stock wheel launches on the legacy default stream)
HIPRIO = os.environ.get("HIPRIO") == "1"
dstream = torch.cuda.Stream(device=R.dev[1], priority=0)
z = torch.randn(16, 4, 58, 104, device=R.dev[1])
with torch.cuda.stream(dstream):
    _, state = dec.decode_step(z[:, :1])
    for _ in range(3):
        _, state = dec.decode_step(z, state)
torch.cuda.synchronize(R.dev[1])


def decode_chunk():
    global state
    with torch.cuda.stream(dstream):
        _, state = dec.decode_step(z, state)


# 1. decoder alone: GPU time and kernel count per chunk
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    with torch.cuda.stream(dstream):
        torch.cuda._sleep(200_000_000)
    decode_chunk()
    torch.cuda.synchronize(R.dev[1])
prof.export_chrome_trace("/tmp/dec_alone.json")
k_alone = sorted((e for e in json.load(open("/tmp/dec_alone.json"))["traceEvents"]
                  if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "spin" not in e["name"]), key=lambda e: e["ts"])
dec_kernels = len(k_alone)
alone_durs = [e["dur"] for e in k_alone]
dec_alone_ms = sum(alone_durs) / 1e3
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
with torch.cuda.stream(dstream):
    torch.cuda._sleep(200_000_000)
    ev[0].record(dstream)
decode_chunk()
with torch.cuda.stream(dstream):
    ev[1].record(dstream)
torch.cuda.synchronize(R.dev[1])
dec_wall_ms = ev[0].elapsed_time(ev[1])
print(f"decoder alone: {dec_wall_ms:.1f} ms per chunk wall, {dec_alone_ms:.1f} ms of kernels, {dec_kernels} kernels")

# 2. 9:3 split alone
H0 = int(os.environ.get("H0", "9"))  # heads on card 0; tokens split in the same ratio
T0 = T * H0 // H
sp = R.Split(T0, H0, 1)
if HIPRIO:
    lo, hi = torch.cuda.Stream.priority_range() if hasattr(torch.cuda.Stream, "priority_range") else (0, -5)
    sp.comp = [torch.cuda.Stream(device=d, priority=-5) for d in R.dev]
    print("DiT on priority -5 streams, decoder on priority 0")
ws = [[R.Weights(d) for d in R.dev] for _ in range(NL)]
kv = [[(torch.randn(1, len(sp.heads[c]), WIN, HD, device=R.dev[c], dtype=torch.bfloat16),
        torch.randn(1, len(sp.heads[c]), WIN, HD, device=R.dev[c], dtype=torch.bfloat16)) for c in (0, 1)]
      for _ in range(NL)]
x0 = torch.randn(T, D, device=R.dev[0], dtype=torch.bfloat16)
x_in = [x0[:T0].clone(), x0[T0:].to(R.dev[1])]


def layer_fn(st, l):
    if l < 0:
        return [x_in[0].clone(), x_in[1].clone()]
    return sp.layer(st, ws[l], [kv[l][0][0], kv[l][1][0]], [kv[l][0][1], kv[l][1][1]])


layer_fn(layer_fn(None, -1), 0)
alone = sorted(R.steady_period(layer_fn, sp.comp, NL) for _ in range(3))[1]
print(f"{H0}:{H - H0} split alone: {alone:.2f} ms per layer")


def split_run_profiled(path):
    for d in R.dev:
        torch.cuda.synchronize(d)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as pr:
        for s in sp.comp:
            with torch.cuda.stream(s):
                torch.cuda._sleep(2_000_000_000)
        st_ = layer_fn(None, -1)
        for l in range(NL):
            st_ = layer_fn(st_, l)
        for d in R.dev:
            torch.cuda.synchronize(d)
    pr.export_chrome_trace(path)


split_run_profiled("/tmp/split_alone.json")  # for sched_analysis.py


# 3. both at once, profiled: decoder chunk queued first on its own stream, then NL layers
def corun():
    for d in R.dev:
        torch.cuda.synchronize(d)
    ends = [[], []]
    for s in (*sp.comp, dstream):
        with torch.cuda.stream(s):
            torch.cuda._sleep(2_000_000_000)
    decode_chunk()
    st_ = layer_fn(None, -1)
    for l in range(NL):
        st_ = layer_fn(st_, l)
        for i, s in enumerate(sp.comp):
            e = torch.cuda.Event(enable_timing=True)
            e.record(s)
            ends[i].append(e)
    for d in R.dev:
        torch.cuda.synchronize(d)
    per = []
    for evs in ends:
        iv = sorted(a.elapsed_time(b) for a, b in zip(evs[4:], evs[5:]))
        per.append(iv[len(iv) // 2])
    return max(per), ends


corun()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    with_dec, _ = corun()
prof.export_chrome_trace("/tmp/corun.json")
tr = [e for e in json.load(open("/tmp/corun.json"))["traceEvents"] if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
# the profiler names CUDA streams by handle, not torch's stream_id: the decoder's stream is the one running convolutions
dev1 = [e for e in tr if e["args"].get("device") == 1 and "spin" not in e["name"]]
dec_sid = next(e["args"]["stream"] for e in dev1 if "convolution" in e["name"] or "fprop" in e["name"])
dec_ev = [e for e in dev1 if e["args"].get("stream") == dec_sid]
comp_sid = next(e["args"]["stream"] for e in dev1 if "qk_int_sv_f8" in e["name"])
dit_ev = [e for e in dev1 if e["args"].get("stream") == comp_sid]
w0 = min(e["ts"] for e in dit_ev)
w1 = max(e["ts"] + e["dur"] for e in dit_ev)
dec_ev.sort(key=lambda e: e["ts"])
before = [e for e in dec_ev if e["ts"] + e["dur"] <= w0]
done = [e for e in dec_ev if e["ts"] + e["dur"] <= w1]
# progress = standalone kernel time of the decoder kernels finished inside the window, over a whole chunk's
frac = (sum(alone_durs[len(before):len(done)])) / sum(alone_durs)
window_ms = (w1 - w0) / 1e3
card1_dit_ms = sum(e["dur"] for e in dit_ev if e.get("cat") == "kernel") / 1e3 / NL
print(f"{H0}:{H - H0} split with the decoder running on card 1: {with_dec:.2f} ms per layer ({(with_dec / alone - 1) * 100:+.0f}%)")
print(f"  card 1 DiT kernel time per layer while decoding: {card1_dit_ms:.2f} ms (alone ~1.01)")
print(f"  decoder progress in the {window_ms:.1f} ms window: kernels {len(before)}..{len(done)} of {dec_kernels} = {frac * 100:.1f}% of a chunk's work"
      f" -> {window_ms / frac:.0f} ms per chunk while sharing (alone {dec_wall_ms:.0f})")

# chunk time: the decoder runs alongside until its chunk is done, then the DiT runs alone
dec_shared_ms = window_ms / frac
layers_during = dec_shared_ms / with_dec
if layers_during >= PASSES:
    chunk = PASSES * with_dec
    note = "decoder never finishes inside the DiT time: decode-bound"
    chunk = max(chunk, dec_shared_ms)
else:
    chunk = dec_shared_ms + (PASSES - layers_during) * alone
    note = f"decoder finishes after {layers_during:.0f} of {PASSES} passes"
print(f"chunk: {chunk / 1e3:.3f} s ({note}); B1 DiT-bound chunk on this pod: {PASSES * 3.86 / 1e3:.3f} s")
json.dump(dict(hiprio=HIPRIO, dec_alone_wall_ms=dec_wall_ms, dec_kernels=dec_kernels, split_alone_ms=alone, split_with_dec_ms=with_dec,
               card1_dit_ms_per_layer_with_dec=card1_dit_ms, dec_frac_in_window=frac, window_ms=window_ms,
               dec_shared_ms_per_chunk=dec_shared_ms, chunk_s=chunk / 1e3), open("/workspace/runs/with_decoder_%d_%s.json" % (H0, "hiprio" if HIPRIO else "stream0"), "w"), indent=1)
