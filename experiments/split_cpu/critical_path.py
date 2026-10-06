#!/usr/bin/env python3
"""Critical path of the split DiT from an Nsight Systems report exported to SQLite.

    nsys export --type sqlite --force-overwrite true -o r.sqlite r.nsys-rep
    python critical_path.py r.sqlite [--skip 1] [--json out.json] [--card0 ID --card1 ID]
    python critical_path.py --selftest

Needs a run with LINGBOT_NVTX=1 (ranges "chunk N", "fwd", "L{li}", pre/send_qkv/attn/send_back/post, decode) and
the cuda trace. Every kernel/copy is attributed to the NVTX ranges that were open on the launching CPU thread when
its runtime call was made (correlationId), so a "chunk" is what the CPU issued inside the chunk range, not what the
GPU ran in that wall-clock window. Streams are labelled by their kernels (decoder = conv-heavy, dit = the rest).
--skip drops the first captured chunks (the GPU queue starts empty there).
"""
import argparse
import bisect
import json
import os
import re
import sqlite3
import statistics
import sys
import tempfile
from collections import defaultdict

DIRS = {1: "HtoD", 2: "DtoH", 8: "DtoD", 9: "HtoH", 10: "PtoP"}
STAGES = ("pre", "send_qkv", "attn", "send_back", "post")
_RULES = [
    ("sage_attn", r"qk_int8|sv_f8|quant_per|sub_mean|sage|attn_inst|transpose_pad_permute|per_channel_fp8"),
    ("flash_attn", r"flash|fmha"),
    ("conv", r"conv|fprop|dgrad|implicit_gemm|cudnn|winograd|im2col"),
    ("fp8_gemm", r"(e4m3|fp8|f8).*(gemm|cutlass|nvjet|scaled_mm)|(gemm|cutlass|nvjet|scaled_mm).*(e4m3|fp8|f8)"),
    ("triton", r"^triton_"),
    ("other_gemm", r"gemm|cutlass|nvjet|cublas|gemv|splitk"),
]
_RULES = [(c, re.compile(r)) for c, r in _RULES]
DIT_CLS = ("fp8_gemm", "sage_attn", "flash_attn", "triton", "other_gemm")
EPS = 3000  # ns: an event ending this much after a gap's end still counts as what ended it


def classify(name):
    n = name.lower()
    return next((c for c, rx in _RULES if rx.search(n)), "other")


def _need(db, table, cols):
    have = [r[1] for r in db.execute(f"PRAGMA table_info({table})")]
    if not have:
        tabs = sorted(r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'"))
        raise SystemExit(f"table {table} not found (have: {', '.join(tabs)}); need an nsys report recorded with "
                         f"-t cuda,nvtx and exported with `nsys export --type sqlite`")
    miss = [c for c in cols if c not in have]
    if miss:
        raise SystemExit(f"table {table}: missing columns {miss} (have {have})")
    return have


def _kind(text):
    m = re.fullmatch(r"chunk (\d+)", text)
    if m:
        return "chunk", int(m[1])
    m = re.fullmatch(r"L(\d+)", text)
    if m:
        return "L", int(m[1])
    if text in ("fwd", "decode") or text in STAGES:
        return ("stage" if text in STAGES else text), text
    return None, None


def load(db):
    """-> (ops, ranges, calls). ops: kernels and copies (ns); calls: correlationId -> (start, end, tid)."""
    _need(db, "StringIds", ["id", "value"])
    strs = dict(db.execute("SELECT id, value FROM StringIds"))
    kc = _need(db, "CUPTI_ACTIVITY_KIND_KERNEL", ["start", "end", "deviceId", "streamId", "correlationId"])
    ncol = next((c for c in ("demangledName", "shortName", "name") if c in kc), None)
    if ncol is None:
        raise SystemExit(f"CUPTI_ACTIVITY_KIND_KERNEL: no name column in {kc}")
    ops = []
    for t0, t1, dev, st, corr, nm in db.execute(
            f"SELECT start, end, deviceId, streamId, correlationId, {ncol} FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        name = strs.get(nm, str(nm)) if isinstance(nm, int) else str(nm)
        ops.append(dict(k="k", t0=t0, t1=t1, dev=dev, st=st, corr=corr, name=name, cls=classify(name)))
    _need(db, "CUPTI_ACTIVITY_KIND_MEMCPY", ["start", "end", "deviceId", "streamId", "correlationId", "bytes", "copyKind"])
    for t0, t1, dev, st, corr, nb, kind in db.execute(
            "SELECT start, end, deviceId, streamId, correlationId, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
        ops.append(dict(k="m", t0=t0, t1=t1, dev=dev, st=st, corr=corr, bytes=nb or 0, dir=DIRS.get(kind, f"kind{kind}"),
                        cls="copy"))
    _need(db, "CUPTI_ACTIVITY_KIND_RUNTIME", ["start", "end", "globalTid", "correlationId"])
    calls = {c: (a, b, tid) for a, b, tid, c in db.execute(
        "SELECT start, end, globalTid, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME")}
    nc = _need(db, "NVTX_EVENTS", ["start", "end", "globalTid"])
    if "text" not in nc and "textId" not in nc:
        raise SystemExit(f"NVTX_EVENTS: neither text nor textId column (have {nc})")
    sel = f"{'text' if 'text' in nc else 'NULL'}, {'textId' if 'textId' in nc else 'NULL'}"
    flt = " AND eventType IN (59, 60)" if "eventType" in nc else ""
    ranges = []
    for t0, t1, tid, text, tid_s in db.execute(f"SELECT start, end, globalTid, {sel} FROM NVTX_EVENTS WHERE end IS NOT NULL{flt}"):
        text = text if text is not None else strs.get(tid_s)
        kind, n = _kind(text or "")
        if kind:
            ranges.append(dict(kind=kind, n=n, t0=t0, t1=t1, tid=tid))
    if not any(r["kind"] == "chunk" for r in ranges):
        raise SystemExit('no NVTX "chunk N" ranges: run with LINGBOT_NVTX=1 and the capture range covering chunks')
    return ops, ranges, calls


def tag(ops, ranges, calls):
    by = defaultdict(list)
    for i, r in enumerate(sorted(ranges, key=lambda r: r["t0"])):
        r["id"] = i
        by[(r["tid"], r["kind"])].append(r)
    starts = {k: [r["t0"] for r in v] for k, v in by.items()}

    def find(tid, kind, t):
        v = by.get((tid, kind))
        if not v:
            return None
        i = bisect.bisect_right(starts[(tid, kind)], t) - 1
        return v[i] if i >= 0 and v[i]["t1"] >= t else None

    for o in ops:
        o["call"] = calls.get(o["corr"])
        o["chunk"] = o["Lid"] = o["Ln"] = o["sid"] = o["stage"] = None
        if o["call"] is None:
            continue
        t, tid = o["call"][0], o["call"][2]
        r = find(tid, "chunk", t)
        o["chunk"] = r["n"] if r else None
        r = find(tid, "L", t)
        if r:
            o["Lid"], o["Ln"] = r["id"], r["n"]
        r = find(tid, "stage", t)
        if r:
            o["sid"], o["stage"] = r["id"], r["n"]


def union(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def busy(ops):
    return sum(b - a for a, b in union((o["t0"], o["t1"]) for o in ops))


def gaps(ops):
    """Idle intervals between merged busy intervals: (start, end, first op of the next interval)."""
    merged = []
    for o in sorted(ops, key=lambda o: o["t0"]):
        if merged and o["t0"] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], o["t1"])
        else:
            merged.append([o["t0"], o["t1"], o])
    return [(p[1], n[0], n[2]) for p, n in zip(merged, merged[1:])]


def pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))] if s else None


def stream_labels(ops):
    t = defaultdict(lambda: defaultdict(int))
    for o in ops:
        if o["k"] == "k":
            t[(o["dev"], o["st"])][o["cls"]] += o["t1"] - o["t0"]
    lab = {}
    for key, c in t.items():
        lab[key] = "decoder" if c["conv"] >= 0.3 * sum(c.values()) else "dit"
    return lab


def walk(a, b, cands, ends):
    """Attribute the idle interval [a, b) on card 0 to what finished last, walking back through the chain
    (receive copy <- send copy <- card-1 kernel). Returns {cause: ns}."""
    out = defaultdict(int)
    while b > a:
        i = bisect.bisect_right(ends, b + EPS) - 1
        while i >= 0 and cands[i]["t0"] >= b:
            i -= 1
        if i < 0 or cands[i]["t1"] <= a:
            break
        o = cands[i]
        out["transfer" if o["k"] == "m" else "card1"] += b - max(a, o["t0"])
        b = o["t0"]
    out["other"] += max(0, b - a)
    return out


def analyze(path, skip=1, card0=None, card1=None, gap_us=50.0):
    db = sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True)
    ops, ranges, calls = load(db)
    tag(ops, ranges, calls)
    lab = stream_labels(ops)
    for o in ops:
        o["lab"] = lab.get((o["dev"], o["st"]), "copy")
    dit_devs = sorted({o["dev"] for o in ops if o["k"] == "k" and o["lab"] == "dit"})
    dec_devs = sorted({o["dev"] for o in ops if o["k"] == "k" and o["lab"] == "decoder"})
    if not dit_devs:
        raise SystemExit("no DiT kernels found (no stream with gemm/attention kernels)")
    card1 = card1 if card1 is not None else (dec_devs[0] if dec_devs else dit_devs[-1])
    card0 = card0 if card0 is not None else next((d for d in dit_devs if d != card1), dit_devs[0])
    dit0 = lambda os_: sorted((o for o in os_ if o["k"] == "k" and o["dev"] == card0 and o["lab"] == "dit"),
                              key=lambda o: o["t0"])
    chunk_r = {r["n"]: r for r in ranges if r["kind"] == "chunk"}
    by_chunk = defaultdict(list)
    for o in ops:
        if o["chunk"] is not None:
            by_chunk[o["chunk"]].append(o)
    end0 = {c: max(o["t1"] for o in d) for c in by_chunk if (d := dit0(by_chunk[c]))}
    steady = [c for c in sorted(chunk_r)[skip:] if c in end0]
    if not steady:
        raise SystemExit(f"no steady chunk with DiT kernels on card {card0} (chunks {sorted(chunk_r)}, skip {skip})")
    cands = sorted((o for o in ops if o["k"] == "m" or (o["dev"] == card1 and o["lab"] == "dit")), key=lambda o: o["t1"])
    ends = [o["t1"] for o in cands]
    res = dict(card0=card0, card1=card1, chunks=[], skipped=sorted(chunk_r)[:skip])
    leads, msgs, piece = defaultdict(list), defaultdict(list), defaultdict(list)
    idle = defaultdict(lambda: defaultdict(float))
    for c in steady:
        co = by_chunk[c]
        d0 = dit0(co)
        lo, hi = d0[0]["t0"], end0[c]
        interval = end0[c] - end0[c - 1] if c - 1 in end0 else None
        ref = interval or (hi - lo)
        r = dict(chunk=c, cpu_ms=(chunk_r[c]["t1"] - chunk_r[c]["t0"]) / 1e6, interval_ms=interval and interval / 1e6,
                 card0_span_ms=(hi - lo) / 1e6)
        sb = defaultdict(list)
        for o in co:
            sb[(o["dev"], o["st"])].append(o)
        r["streams"] = [dict(dev=d, stream=s, label=lab.get((d, s), "copy"), busy_ms=busy(v) / 1e6, busy_pct=100 * busy(v) / ref)
                        for (d, s), v in sorted(sb.items())]
        cls = defaultdict(lambda: defaultdict(float))
        for o in co:
            cls[o["dev"]][o["cls"]] += (o["t1"] - o["t0"]) / 1e6
        r["classes_ms"] = {d: dict(v) for d, v in sorted(cls.items())}
        cp = defaultdict(lambda: dict(n=0, bytes=0, ms=0.0))
        for o in co:
            if o["k"] == "m":
                e = cp[o["dir"]]
                e["n"] += 1
                e["bytes"] += o["bytes"]
                e["ms"] += (o["t1"] - o["t0"]) / 1e6
                if o["t1"] > o["t0"]:
                    piece[o["dir"]].append(o["bytes"] / (o["t1"] - o["t0"]))
        r["copies"] = {k: dict(v, gbps=v["bytes"] / (v["ms"] * 1e6) if v["ms"] else None) for k, v in cp.items()}
        grp = defaultdict(list)
        for o in co:
            if o["k"] == "m" and o["sid"] is not None and o["stage"] in ("send_qkv", "send_back"):
                grp[(o["sid"], o["stage"])].append(o)
        for (_, stage), v in grp.items():
            d2h = [o for o in v if o["dir"] == "DtoH"]
            h2d = [o for o in v if o["dir"] == "HtoD"]
            if d2h and h2d:
                lat = max(o["t1"] for o in h2d) - min(o["t0"] for o in d2h)
                nb = sum(o["bytes"] for o in d2h)
                msgs[(stage, d2h[0]["dev"])].append((lat / 1e3, nb, nb / lat))
        for d in sorted({o["dev"] for o in co if o["k"] == "k"}):
            for o in co:
                if o["k"] == "k" and o["dev"] == d and o["call"]:
                    leads[d].append((o["t0"] - o["call"][1]) / 1e3)
            for name, sel in ((f"dev{d} all", lambda o: True), (f"dev{d} dit", lambda o: o["lab"] == "dit")):
                g = [o for o in co if o["k"] == "k" and o["dev"] == d and sel(o)]
                if not g:
                    continue
                a = idle[name]
                a["window"] += max(o["t1"] for o in g) - min(o["t0"] for o in g)
                a["busy"] += busy(g)
                for ga, gb, nx in gaps(g):
                    if gb - ga > gap_us * 1e3 and nx["call"]:
                        a["cpu" if nx["call"][0] >= ga else "dependency"] += gb - ga
        # critical path on card 0: busy time, then every idle interval attributed
        bound = defaultdict(float)
        bound["card0"] = busy(d0)
        for ga, gb, nx in gaps(d0):
            if nx["call"] and nx["call"][0] >= ga:
                cut = min(max(nx["call"][1], ga), gb)
                bound["cpu"] += cut - ga
                ga = cut
            for k, v in walk(ga, gb, cands, ends).items():
                bound[k] += v
        r["bound_ms"] = {k: v / 1e6 for k, v in bound.items()}
        r["bound_ms"]["window"] = (hi - lo) / 1e6
        lay = defaultdict(list)
        for o in co:
            if o["Lid"] is not None:
                lay[o["Lid"]].append(o)
        last = defaultdict(int)
        for v in lay.values():
            e = {"card0": [o["t1"] for o in v if o["k"] == "k" and o["dev"] == card0 and o["lab"] == "dit"],
                 "card1": [o["t1"] for o in v if o["k"] == "k" and o["dev"] == card1 and o["lab"] == "dit"],
                 "transfer": [o["t1"] for o in v if o["k"] == "m"]}
            e = {k: max(x) for k, x in e.items() if x}
            if e:
                last[max(e, key=e.get)] += 1
        r["layer_last_finisher"] = dict(last)
        res["chunks"].append(r)
    res["messages"] = [dict(stage=s, src_dev=d, n=len(v), lat_us_p50=pct([x[0] for x in v], .5),
                            lat_us_p90=pct([x[0] for x in v], .9), bytes_p50=pct([x[1] for x in v], .5),
                            gbps_p50=pct([x[2] for x in v], .5)) for (s, d), v in sorted(msgs.items())]
    res["copy_piece_gbps"] = {k: dict(n=len(v), p10=pct(v, .1), p50=pct(v, .5), p90=pct(v, .9)) for k, v in piece.items()}
    res["launch_lead_us"] = {d: dict(n=len(v), p10=pct(v, .1), p50=pct(v, .5), p90=pct(v, .9)) for d, v in sorted(leads.items())}
    n = len(steady)
    res["idle"] = {k: dict(busy_pct=100 * v["busy"] / v["window"], cpu_ms=v["cpu"] / 1e6 / n,
                           dependency_ms=v["dependency"] / 1e6 / n) for k, v in sorted(idle.items())}
    return res


def f(x, spec=".1f"):
    return "n/a" if x is None else format(x, spec)


def report(res):
    n = len(res["chunks"])
    print(f"card0 = device {res['card0']}, card1 = device {res['card1']}; steady chunks "
          f"{[c['chunk'] for c in res['chunks']]} (skipped {res['skipped']})")
    print("\n(a) per chunk: cpu = NVTX range on the launch thread; interval = card-0 DiT end to previous chunk's end")
    for c in res["chunks"]:
        print(f"chunk {c['chunk']}: cpu {c['cpu_ms']:.1f} ms, card0 DiT span {c['card0_span_ms']:.1f} ms, "
              f"interval {f(c['interval_ms'])} ms")
        for s in c["streams"]:
            print(f"    dev{s['dev']} {s['label']:<8} stream {s['stream']:<4} busy {s['busy_ms']:8.2f} ms ({s['busy_pct']:5.1f}%)")
    print("\n(b) kernel/copy time by class, ms per chunk (mean)")
    for d in sorted({d for c in res["chunks"] for d in c["classes_ms"]}):
        tot = defaultdict(float)
        for c in res["chunks"]:
            for k, v in c["classes_ms"].get(d, {}).items():
                tot[k] += v / n
        print(f"  dev{d}: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(tot.items(), key=lambda kv: -kv[1])))
    print("\n(c) transfers per chunk (mean): direction, pieces, MB, ms, avg GB/s")
    for k in sorted({k for c in res["chunks"] for k in c["copies"]}):
        v = [c["copies"].get(k, dict(n=0, bytes=0, ms=0.0)) for c in res["chunks"]]
        nb, ms = sum(x["bytes"] for x in v), sum(x["ms"] for x in v)
        print(f"  {k}: {sum(x['n'] for x in v) / n:.0f} pieces, {nb / n / 1e6:.2f} MB, {ms / n:.2f} ms, "
              f"{f(nb / (ms * 1e6) if ms else None, '.1f')} GB/s")
    for k, v in res["copy_piece_gbps"].items():
        print(f"  piece GB/s {k}: p10 {v['p10']:.1f} p50 {v['p50']:.1f} p90 {v['p90']:.1f} (n={v['n']})")
    print("  per-layer messages (first D2H start to last H2D end), by sending device:")
    for m in res["messages"]:
        print(f"    {m['stage']:<9} from dev{m['src_dev']}: n={m['n']} latency p50 {m['lat_us_p50']:.0f} us p90 {m['lat_us_p90']:.0f} us, "
              f"{m['bytes_p50'] / 1e6:.2f} MB, {m['gbps_p50']:.1f} GB/s effective")
    print("\n(d) launch lead (kernel start - end of its launch call), us")
    for d, v in res["launch_lead_us"].items():
        print(f"  dev{d}: p10 {v['p10']:.0f}  p50 {v['p50']:.0f}  p90 {v['p90']:.0f}  (n={v['n']})")
    print("\n(e) idle (gaps > 50 us; waiting for CPU = next launch call began after the gap did), ms per chunk")
    for k, v in res["idle"].items():
        print(f"  {k}: busy {v['busy_pct']:.1f}% of its span, waiting for CPU {v['cpu_ms']:.2f}, for dependency {v['dependency_ms']:.2f}")
    print("\n(f) critical path on card 0 DiT (busy + each idle interval attributed to what finished last)")
    for c in res["chunks"]:
        b = c["bound_ms"]
        print(f"chunk {c['chunk']}: bound by: cpu {b.get('cpu', 0):.2f} ms, card0 {b.get('card0', 0):.2f} ms, "
              f"card1 {b.get('card1', 0):.2f} ms, transfer {b.get('transfer', 0):.2f} ms, other {b.get('other', 0):.2f} ms "
              f"(of {b['window']:.2f}); layers where last to finish was {c['layer_last_finisher']}")


def _synth(path):
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INT, end INT, deviceId INT, contextId INT, streamId INT,
            correlationId INT, globalPid INT, demangledName INT, shortName INT, gridX INT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INT, end INT, deviceId INT, contextId INT, streamId INT,
            correlationId INT, globalPid INT, bytes INT, copyKind INT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INT, end INT, eventClass INT, globalTid INT, correlationId INT, nameId INT);
        CREATE TABLE NVTX_EVENTS(start INT, end INT, eventType INT, text TEXT, textId INT, globalTid INT, rangeId INT, domainId INT);
    """)
    ids, corr = {}, [0]
    us = 1000

    def sid(s):
        if s not in ids:
            ids[s] = len(ids) + 1
            db.execute("INSERT INTO StringIds VALUES (?, ?)", (ids[s], s))
        return ids[s]

    def call(t):
        corr[0] += 1
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 0, 1, ?, 0)", (t * us, (t + 10) * us, corr[0]))
        return corr[0]

    def k(dev, st, t0, t1, name, ct):
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, 1, ?, ?, 1, ?, 0, 1)",
                   (t0 * us, t1 * us, dev, st, call(ct), sid(name)))

    def m(dev, st, t0, t1, nb, kind, ct):
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, 1, ?, ?, 1, ?, ?)",
                   (t0 * us, t1 * us, dev, st, call(ct), nb, kind))

    def nv(text, t0, t1, by_id):
        db.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, 59, ?, ?, 1, 0, 0)",
                   (t0 * us, t1 * us, None if by_id else text, sid(text) if by_id else None))

    TRI, SAGE = "triton_poi_fused_add_mul_0", "qk_int8_sv_f8_accum_f32_attn_inst_buf_fwd_kernel"
    GEMM, CONV = "cutlass3x_sm120_tensorop_gemm_e4m3_e4m3_f32", "sm80_xmma_fprop_implicit_gemm_f16f16"
    for c in (5, 6, 7):
        b = 1_000_000 + c * 100_000
        nv(f"chunk {c}", b - 50, b + 3500, True)
        nv("fwd", b, b + 3400, False)
        nv("L0", b, b + 500, True)
        nv("L1", b + 500, b + 3300, False)
        nv("decode", b + 3450, b + 3480, True)
        k(1, 11, b + 3500, b + 3800, CONV, b + 3455)
        for l, (off, send, ca, cp) in enumerate([(0, 15, (10, 20, 30, 12), (16, 19)),
                                                (500, 530, (510, 3150, 3200, 520), (532, 536))]):
            s = b + 2000 + l * 1000
            nv("send_qkv", b + send, b + send + 15, l == 0)
            k(0, 7, s, s + 100, TRI, b + ca[0])
            k(0, 7, s + 400, s + 700, SAGE, b + ca[1])
            k(0, 7, s + 700, s + 1000, GEMM, b + ca[2])
            k(1, 9, s + 50, s + 200, GEMM, b + ca[3])
            m(1, 20, s + 210, s + 250, 1_000_000, 2, b + cp[0])
            m(0, 21, s + 260, s + 400, 1_000_000, 1, b + cp[1])
    db.commit()
    db.close()


def selftest():
    close = lambda a, b: abs(a - b) < 1e-6
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "synth.sqlite")
        _synth(p)
        res = analyze(p)
    assert (res["card0"], res["card1"]) == (0, 1) and res["skipped"] == [5], res
    c6, c7 = res["chunks"]
    assert (c6["chunk"], c7["chunk"]) == (6, 7)
    assert close(c6["interval_ms"], 100.0) and close(c6["card0_span_ms"], 2.0) and close(c6["cpu_ms"], 3.55), c6
    lab = {(s["dev"], s["stream"]): s["label"] for s in c6["streams"]}
    assert lab == {(0, 7): "dit", (1, 9): "dit", (1, 11): "decoder", (1, 20): "copy", (0, 21): "copy"}, lab
    assert close(next(s for s in c6["streams"] if s["stream"] == 7)["busy_ms"], 1.4)
    c0, c1 = c6["classes_ms"][0], c6["classes_ms"][1]
    assert close(c0["triton"], 0.2) and close(c0["sage_attn"], 0.6) and close(c0["fp8_gemm"], 0.6), c0
    assert close(c1["fp8_gemm"], 0.3) and close(c1["conv"], 0.3), c1
    assert c6["copies"]["DtoH"]["n"] == 2 and close(c6["copies"]["HtoD"]["ms"], 0.28), c6["copies"]
    (m,) = [m for m in res["messages"] if m["stage"] == "send_qkv"]
    assert m["src_dev"] == 1 and m["n"] == 4 and close(m["lat_us_p50"], 190.0), m
    b = c6["bound_ms"]
    exp = dict(card0=1.4, cpu=0.06, card1=0.16, transfer=0.38)
    assert all(close(b[k], v) for k, v in exp.items()) and close(b.get("other", 0), 0) and close(b["window"], 2.0), b
    assert c6["layer_last_finisher"] == {"card0": 2}, c6
    assert res["idle"]["dev0 dit"]["cpu_ms"] == 0.3 and res["idle"]["dev0 dit"]["dependency_ms"] == 0.3, res["idle"]
    assert res["launch_lead_us"][0]["n"] == 12 and res["launch_lead_us"][1]["p50"] > 0, res["launch_lead_us"]
    report(res)
    print("\nselftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite", nargs="?")
    ap.add_argument("--skip", type=int, default=1, help="captured chunks to drop from the start (default 1)")
    ap.add_argument("--card0", type=int, help="deviceId of card 0 (default: the DiT device without the decoder)")
    ap.add_argument("--card1", type=int, help="deviceId of card 1 (default: the device running the decoder)")
    ap.add_argument("--json", help="write the numbers to this file")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.sqlite:
        ap.error("sqlite path required (or --selftest)")
    res = analyze(a.sqlite, a.skip, a.card0, a.card1)
    report(res)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    sys.exit(main())
