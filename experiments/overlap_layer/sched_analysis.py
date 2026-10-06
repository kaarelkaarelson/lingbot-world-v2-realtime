"""Which kernel waits for what on card 1 when the decoder shares it with the 9:3 split.

Reads two profiler traces of the same 12 split layers: alone (/tmp/split_alone.json) and with the decoder
running on card 1 (/tmp/corun.json). For each card-1 DiT kernel in a middle layer: when its inputs were
ready (previous kernel on its stream ended, or the awaited copy finished), when it started, how long it ran
vs the same kernel alone, and which decoder kernels held the SMs meanwhile (with their grid sizes).
Card 0's waits are reported too. Output: sched.json for the page, and a printed table.
"""
import json
import sys

AL, CO = "/tmp/split_alone.json", "/tmp/corun.json"


def load(p):
    tr = json.load(open(p))["traceEvents"]
    return [e for e in tr if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "spin" not in e["name"]]


def short(n):
    for k, v in (("qk_int_sv_f8", "Sage attention"), ("QuantInt8", "Sage quantize"), ("TransposePad", "Sage transpose V"),
                 ("MeanScale", "Sage V scale"), ("reduce_kernel", "Sage K mean"), ("cutlass", "FP8 GEMM"),
                 ("flash_fwd", "cross-attention"), ("elementwise", "copy/place"), ("triton_poi", "fused elementwise"),
                 ("triton_red", "fused norm/quant"), ("triton_per", "fused norm/quant"), ("convolution", "decoder conv"),
                 ("Memcpy HtoD", "PCIe in"), ("Memcpy DtoH", "PCIe out"), ("Memcpy DtoD", "local copy"), ("Memset", "memset")):
        if k in n:
            return v
    return n[:28]


co = load(CO)
dev1 = [e for e in co if e["args"].get("device") == 1]
dec_sid = next(e["args"]["stream"] for e in dev1 if "convolution" in e["name"])
dec = sorted((e for e in dev1 if e["args"]["stream"] == dec_sid), key=lambda e: e["ts"])
dit1 = sorted((e for e in dev1 if e["args"]["stream"] != dec_sid and e["cat"] == "kernel"), key=lambda e: e["ts"])
copies1 = sorted((e for e in dev1 if e["cat"] == "gpu_memcpy" and e["args"]["stream"] != dec_sid and "DtoD" not in e["name"]), key=lambda e: e["ts"])
dit0 = sorted((e for e in co if e["args"].get("device") == 0 and e["cat"] == "kernel"), key=lambda e: e["ts"])

# the layer structure on card 1: kernels per layer from the alone trace
al = load(AL)
al1 = sorted((e for e in al if e["args"].get("device") == 1 and e["cat"] == "kernel"), key=lambda e: e["ts"])
per_layer = len(al1) // 12
alone_dur = [e["dur"] for e in al1[per_layer * 6: per_layer * 7]]  # a middle layer, alone
L = 6
win = dit1[per_layer * L: per_layer * (L + 1)]
t0 = win[0]["ts"]
rows = []
prev_end = None
for i, k in enumerate(win):
    # ready: previous kernel on this stream finished, or (for the first kernel after a receive) the copy landed
    ready = prev_end if prev_end is not None else k["ts"]
    landed = [c for c in copies1 if "HtoD" in c["name"] and c["ts"] + c["dur"] <= k["ts"] and c["ts"] + c["dur"] > ready]
    if landed:
        ready = max(c["ts"] + c["dur"] for c in landed)
    wait = k["ts"] - ready
    overl = [d for d in dec if d["ts"] < k["ts"] + k["dur"] and d["ts"] + d["dur"] > ready]
    rows.append(dict(i=i, kernel=short(k["name"]), start=(k["ts"] - t0) / 1e3, ready=(ready - t0) / 1e3,
                     wait_ms=wait / 1e3, dur_ms=k["dur"] / 1e3, alone_ms=alone_dur[i] / 1e3 if i < len(alone_dur) else None,
                     grid=k["args"].get("grid"), block=k["args"].get("block"),
                     blocking=[dict(name=short(d["name"]), grid=d["args"].get("grid"), dur_ms=d["dur"] / 1e3,
                                    start=(d["ts"] - t0) / 1e3) for d in overl][:4]))
    prev_end = k["ts"] + k["dur"]

lay_end = win[-1]["ts"] + win[-1]["dur"]
c0 = [e for e in dit0 if t0 - 4000 <= e["ts"] <= lay_end]
gaps0 = []
for a, b in zip(c0, c0[1:]):
    g = b["ts"] - (a["ts"] + a["dur"])
    if g > 50:
        gaps0.append(dict(after=short(a["name"]), before=short(b["name"]), at=(a["ts"] + a["dur"] - t0) / 1e3, gap_ms=g / 1e3))
dec_lane = [dict(name=short(d["name"]), start=(d["ts"] - t0) / 1e3, dur=d["dur"] / 1e3, grid=d["args"].get("grid"))
            for d in dec if d["ts"] + d["dur"] > t0 - 4000 and d["ts"] < lay_end]
dit_lane = [dict(name=short(k["name"]), start=(k["ts"] - t0) / 1e3, dur=k["dur"] / 1e3) for k in win]
c0_lane = [dict(name=short(k["name"]), start=(k["ts"] - t0) / 1e3, dur=k["dur"] / 1e3) for k in c0]
json.dump(dict(rows=rows, card0_gaps=gaps0, dec_lane=dec_lane, dit1_lane=dit_lane, card0_lane=c0_lane,
               layer_ms=(lay_end - t0) / 1e3), open(sys.argv[1], "w"), indent=1)

print(f"card 1, layer {L}: {per_layer} DiT kernels, {(lay_end - t0) / 1e3:.2f} ms")
print("kernel\tready->start wait ms\tran ms\talone ms\tgrid\tdecoder kernels holding the GPU")
for r in rows:
    b = "; ".join(f"{x['name']} grid {x['grid']} {x['dur_ms']:.2f}ms" for x in r["blocking"][:2])
    print(f"{r['kernel']}\t{r['wait_ms']:.3f}\t{r['dur_ms']:.3f}\t{'' if r['alone_ms'] is None else f'{r['alone_ms']:.3f}'}\t{r['grid']}\t{b}")
print("card 0 gaps > 50 us:")
for g in gaps0:
    print(f"  at {g['at']:.2f} ms: {g['gap_ms']:.3f} ms idle between {g['after']} and {g['before']}")
dsum = {}
for d in dec_lane:
    dsum[d["name"]] = dsum.get(d["name"], 0) + d["dur"]
print("decoder kernel mix in window:", {k: round(v, 2) for k, v in dsum.items()})
