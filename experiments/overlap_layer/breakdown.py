"""Where the time goes in the split layer: every kernel and copy of both cards on one clock (torch.profiler),
labelled by stage, next to each stage's roofline. Writes breakdown.json for the timeline page.

  SPLIT=9:3 python experiments/overlap_layer/breakdown.py out.json
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import real_layer as R  # noqa: E402

torch.set_grad_enabled(False)
T, D, H, HD, WIN, FFN, CTX = R.T, R.D, R.H, R.HD, R.WIN, R.FFN, R.CTX
PEAK = {"INT8": 838e12, "FP8": 419e12, "BF16": 209.5e12}
HBM = 1792e9
LINK = {"D2H": 42.3e9, "H2D": 56.4e9}  # measured on this pod, one card alone
LAYERS = 6
h0 = {"9:3": 9, "6:6": 6}[os.environ.get("SPLIT", "9:3")]
t0 = T * h0 // H

split = R.Split(t0, h0, 1)
ws = [[R.Weights(d) for d in R.dev] for _ in range(LAYERS)]
kv = [[(torch.randn(1, len(split.heads[c]), WIN, HD, device=R.dev[c], dtype=torch.bfloat16),
        torch.randn(1, len(split.heads[c]), WIN, HD, device=R.dev[c], dtype=torch.bfloat16)) for c in (0, 1)]
      for _ in range(LAYERS)]
x0 = torch.randn(T, D, device=R.dev[0], dtype=torch.bfloat16)
x_in = [x0[:t0].clone(), x0[t0:].to(R.dev[1])]


def run(layers=LAYERS):
    xs = [x_in[0].clone(), x_in[1].clone()]
    for l in range(layers):
        with R.stage(f"layer {l}"):
            xs = split.layer(xs, ws[l], [kv[l][0][0], kv[l][1][0]], [kv[l][0][1], kv[l][1][1]])


run()
run()
for d in R.dev:
    torch.cuda.synchronize(d)
gpu_ms = R.gpu_only(run, [split.comp[0], split.comp[1]], layers=LAYERS)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
    for s in split.comp:  # CPU queues everything while both GPUs sleep, so GPU timestamps are GPU-bound
        with torch.cuda.stream(s):
            torch.cuda._sleep(1_000_000_000)
    run()
    for d in R.dev:
        torch.cuda.synchronize(d)
prof.export_chrome_trace("/tmp/split_trace.json")
tr = json.load(open("/tmp/split_trace.json"))["traceEvents"]

# CPU side: stage ranges (user annotations) and the runtime calls inside them, keyed by correlation id
ann = [e for e in tr if e.get("cat") == "user_annotation"]
rt = {e["args"]["correlation"]: e for e in tr if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {})}


def labels_at(e):
    inside = [a for a in ann if a["tid"] == e["tid"] and a["ts"] <= e["ts"] <= a["ts"] + a["dur"]]
    layer = next((int(a["name"].split()[1]) for a in inside if a["name"].startswith("layer ")), None)
    st = [a for a in inside if not a["name"].startswith("layer ")]
    return layer, (min(st, key=lambda a: a["dur"])["name"] if st else "other")


gpu = []
for e in tr:
    if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset") or "correlation" not in e.get("args", {}):
        continue
    r = rt.get(e["args"]["correlation"])
    if r is None:
        continue
    layer, st = labels_at(r)
    if layer is None:
        continue
    kind = "kernel" if e["cat"] == "kernel" else e["name"]
    gpu.append(dict(layer=layer, stage=st, device=e["args"].get("device"), stream=e["args"].get("stream"),
                    ts=e["ts"], dur=e["dur"], name=e["name"][:80], kind=kind))

L = LAYERS // 2  # a middle layer
lay = [g for g in gpu if g["layer"] == L]
t_start = min(g["ts"] for g in lay)
t_end = max(g["ts"] + g["dur"] for g in lay)
# lanes: per device, the compute stream is the one with the most kernel time; copies are their own lanes
lanes = {}
for g in lay:
    key = (g["device"], g["stream"])
    lanes.setdefault(key, []).append(g)


def lane_name(dev, stream, evs):
    if any(e["kind"] == "kernel" for e in evs):
        return f"card {dev} compute"
    if any("DtoH" in e["kind"] for e in evs):
        return f"card {dev} -> host"
    return f"host -> card {dev}"


out_lanes = []
for (dv, st), evs in sorted(lanes.items()):
    out_lanes.append(dict(lane=lane_name(dv, st, evs), device=dv, stream=st,
                          events=[dict(stage=e["stage"], name=e["name"], start=round((e["ts"] - t_start) / 1e3, 4),
                                       dur=round(e["dur"] / 1e3, 4), kind=e["kind"]) for e in sorted(evs, key=lambda e: e["ts"])]))

# roofline per stage and card
n = [t0, T - t0]
hc = [h0, H - h0]


def mm(m, k, nn, prec="FP8"):
    f = 2 * m * k * nn
    b = k * nn * (1 if prec == "FP8" else 2) + m * k * 2 + m * nn * 2
    return f, b, prec


roof = []
for c in (0, 1):
    p = 1 - c
    stages = {
        "q|k|v proj (peer heads)": [mm(n[c], D, 3 * hc[p] * HD)],
        "q|k|v proj (own heads) + place": [mm(n[c], D, 3 * hc[c] * HD), (0, 3 * n[c] * hc[c] * HD * 2 * 2, "BF16")],
        "place received q|k|v": [(0, 3 * n[p] * hc[c] * HD * 2 * 2, "BF16")],
        "self-attention (Sage)": [(4 * hc[c] * T * WIN * HD, hc[c] * (T + 2 * WIN) * HD + hc[c] * T * HD * 2, "INT8")],
        "pack output for peer": [(0, n[p] * hc[c] * HD * 2 * 2, "BF16")],
        "assemble heads": [(0, n[c] * H * HD * 2 * 2, "BF16")],
        "out proj + cross q": [mm(n[c], D, D), mm(n[c], D, D)],
        "cross-attention": [(4 * n[c] * CTX * H * HD, n[c] * D * 2 * 2 + 2 * CTX * D * 2, "BF16")],
        "cross out + FFN": [mm(n[c], D, D), mm(n[c], D, FFN), mm(n[c], FFN, D)],
    }
    for name, parts in stages.items():
        t_math = sum(f / PEAK[pr] for f, b, pr in parts)
        t_mem = sum(b / HBM for f, b, pr in parts)
        meas = sum(e["dur"] for e in lay if e["device"] == c and e["stage"] == name and e["kind"] == "kernel") / 1e3
        roof.append(dict(card=c, stage=name, flops=sum(f for f, b, pr in parts), bytes=sum(b for f, b, pr in parts),
                         t_math_ms=t_math * 1e3, t_mem_ms=t_mem * 1e3, roofline_ms=max(t_math, t_mem) * 1e3,
                         measured_ms=meas))
    for msg, nbytes in (("send q|k|v", 3 * n[c] * hc[p] * HD * 2), ("send output", n[p] * hc[c] * HD * 2)):
        d2h = [e for e in lay if e["stage"] == msg and e["device"] == c and "DtoH" in e["kind"]]
        h2d = [e for e in lay if e["stage"] == msg and e["device"] == p and "HtoD" in e["kind"]]
        meas = ((max(e["ts"] + e["dur"] for e in h2d) - min(e["ts"] for e in d2h)) / 1e3) if d2h and h2d else None
        # pipelined pieces: the slower hop sets the rate, plus one piece of the other hop
        roof_t = max(nbytes / LINK["D2H"], nbytes / LINK["H2D"]) + min(nbytes / LINK["D2H"], nbytes / LINK["H2D"]) / R.PIECES
        roof.append(dict(card=c, stage=msg + " (to card %d)" % p, bytes=nbytes, flops=0, t_math_ms=0,
                         t_mem_ms=roof_t * 1e3, roofline_ms=roof_t * 1e3, measured_ms=meas))

busy = {c: sum(e["dur"] for e in lay if e["device"] == c and e["kind"] != "kernel" and "DtoD" not in e["kind"]
               and "Memset" not in e["kind"]) for c in (0, 1)}
json.dump(dict(split=os.environ.get("SPLIT", "9:3"), pieces=R.PIECES, tokens=n, heads=hc, layer=L, layer_span_ms=(t_end - t_start) / 1e3,
               gpu_only_ms_per_layer=gpu_ms, lanes=out_lanes, roofline=roof), open(sys.argv[1], "w"), indent=1)
print(f"layer span {(t_end - t_start) / 1e3:.2f} ms; GPU-only {gpu_ms:.2f} ms/layer; lanes:",
      [(l["lane"], len(l["events"])) for l in out_lanes])
for r in roof:
    m = r["measured_ms"]
    print(f"card {r['card']}\t{r['stage']}\troofline {r['roofline_ms']:.3f} ms\tmeasured {m if m is None else round(m, 3)}")
