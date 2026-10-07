"""Chrome trace (torch.profiler) -> compact per-lane JSON for the page's trace viewer.

Every GPU kernel, copy and memset on both cards, by (card, stream), with start/duration in ms from the
first DiT kernel, launch grid, layer and stage (from record_function labels via the launch correlation id).
  python compact_trace.py trace.json out.json [layers_from layers_to]
"""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
lo, hi = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (4, 8)
tr = json.load(open(src))["traceEvents"]
ann = [e for e in tr if e.get("cat") == "user_annotation"]
by_tid = {}
for a in ann:
    by_tid.setdefault(a["tid"], []).append(a)
rt = {e["args"]["correlation"]: e for e in tr if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {})}


def labels(r):
    inside = [a for a in by_tid.get(r["tid"], []) if a["ts"] <= r["ts"] <= a["ts"] + a["dur"]]
    layer = next((int(a["name"].split()[1]) for a in inside if a["name"].startswith("layer ")), None)
    dec = any(a["name"] == "decoder chunk" for a in inside)
    st = [a for a in inside if not a["name"].startswith("layer ") and a["name"] != "decoder chunk"]
    return layer, dec, (min(st, key=lambda a: a["dur"])["name"] if st else "")


def short(n):
    for k, v in (("qk_int_sv_f8", "Sage attention"), ("QuantInt8", "Sage quantize"), ("TransposePad", "Sage transpose V"),
                 ("MeanScale", "Sage V scale"), ("reduce_kernel", "reduce"), ("cutlass", "FP8 GEMM"),
                 ("flash_fwd", "FlashAttention"), ("fprop_implicit_gemm", "cuDNN conv3d"), ("convolution", "Triton conv3d"),
                 ("elementwise_kernel", "copy/elementwise"), ("triton_poi", "fused pointwise"), ("triton_red", "fused reduction"),
                 ("triton_per", "fused reduction"), ("Memcpy HtoD", "copy host → card"), ("Memcpy DtoH", "copy card → host"),
                 ("Memcpy DtoD", "copy on card"), ("Memset", "memset"), ("spin_kernel", "sleep")):
        if k in n:
            return v
    return n[:40]


ev = []
for e in tr:
    if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset") or "spin_kernel" in e["name"]:
        continue
    r = rt.get(e.get("args", {}).get("correlation"))
    layer, dec, st = labels(r) if r else (None, False, "")
    ev.append(dict(dev=e["args"].get("device"), stream=e["args"].get("stream"), ts=e["ts"], dur=e["dur"], name=short(e["name"]),
                   full=e["name"][:120], grid=e["args"].get("grid"), layer=layer, dec=dec, stage=st))
dit = [e for e in ev if e["layer"] is not None and lo <= e["layer"] < hi]
t0 = min(e["ts"] for e in dit)
t1 = max(e["ts"] + e["dur"] for e in dit)
win = [e for e in ev if e["ts"] + e["dur"] >= t0 and e["ts"] <= t1]
lanes = {}
for e in win:
    lanes.setdefault((e["dev"], e["stream"]), []).append(e)


def lane_label(dev, evs):
    names = {e["name"] for e in evs}
    if any(e["dec"] for e in evs) or names & {"cuDNN conv3d", "Triton conv3d"}:
        return f"card {dev} · decoder", 2
    if "Sage attention" in names:
        return f"card {dev} · DiT", 0
    if names <= {"copy card → host"}:
        return f"card {dev} → host", 3
    if names <= {"copy host → card"}:
        return f"host → card {dev}", 4
    return f"card {dev} · other", 5


out = []
for (dev, stream), evs in lanes.items():
    lab, rank = lane_label(dev, evs)
    out.append(dict(label=lab, dev=dev, rank=rank, events=[
        [round((e["ts"] - t0) / 1e3, 4), round(e["dur"] / 1e3, 4), e["name"], e["stage"], e["layer"],
         e["grid"][0] * e["grid"][1] * e["grid"][2] if e.get("grid") else None, e["full"]] for e in sorted(evs, key=lambda e: e["ts"])]))
out.sort(key=lambda l: (l["dev"], l["rank"]))
json.dump(dict(span_ms=round((t1 - t0) / 1e3, 3), layers=[lo, hi], lanes=out), open(dst, "w"))
print(dst, f"{(t1 - t0) / 1e3:.2f} ms,", [(l["label"], len(l["events"])) for l in out])
