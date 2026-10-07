"""Nsight Systems capture -> Perfetto (Chrome trace JSON), optionally opened in ui.perfetto.dev.

  python tools/nsys_to_perfetto.py capture.sqlite [-o trace.json.gz] [--open]
  python tools/nsys_to_perfetto.py capture.nsys-rep --open      # exports the SQLite first (needs nsys)

Tracks: the busiest CPU thread (its CUDA API calls and NVTX ranges, nested chunk / forward / layer / stage), and per
GPU one track per stream, named by what runs on it (DiT, decoder, copies). Each kernel and memcpy has an arrow from
the CUDA call that launched it. `--open` serves the file once on 127.0.0.1:9001 and opens ui.perfetto.dev on it with
every track group expanded (as Perfetto's `open_trace_in_ui` does, plus a startup command); the trace stays local.
Standard library only.
"""
import argparse
import collections
import gzip
import json
import os
import sqlite3
import subprocess
import sys

COPY = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP"}
# expand every track group (per-GPU streams, CPU NVTX levels) once the trace loads
EXPAND_ALL = [{"id": "dev.perfetto.ExpandTracksByRegex", "args": [".*"]}]


def _sqlite(path):
    if path.endswith(".sqlite"):
        return path
    out = path.rsplit(".", 1)[0] + ".sqlite"
    if not os.path.exists(out):
        subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "-o", out, path], check=True,
                       stdout=subprocess.DEVNULL)
    return out


def _stream_name(kernels):
    n = " ".join(kernels).lower()
    if any(k in n for k in ("fprop", "convolution", "conv3d", "cudnn")) and not any(k in n for k in ("sage", "qk_int")):
        return "decoder"
    if any(k in n for k in ("qk_int", "sage", "gemm", "device_kernel", "triton")):
        return "DiT"
    return "kernels"


def _tables(c):
    return {r[0] for r in c.execute("select name from sqlite_master where type='table'")}


def _occupancy(g, sms, block, regs, smem, grid):
    """Blocks resident per SM (limited by registers, shared memory, warps, block slots) and how many waves the grid
    needs on `sms` SMs. The last wave's fill is the classic tail loss: 1.41 waves runs as 2 at 71 % fill."""
    if not g or not block:
        return {}
    _, sm_count, regs_sm, smem_sm, warps_sm, blocks_sm = g
    sms = sms or sm_count
    warps = -(-block // 32)
    regs_warp = -(-max(regs, 1) * 32 // 256) * 256
    lim = dict(regs=regs_sm // (regs_warp * warps), warps=warps_sm // warps, slots=blocks_sm,
               smem=(smem_sm // (smem + 1024)) if smem else blocks_sm)
    per_sm = max(min(lim.values()), 1)
    waves = grid / (per_sm * sms)
    full = -(-grid // (per_sm * sms))
    return dict(blocks_per_SM=per_sm, limited_by=min(lim, key=lim.get), occupancy_warps_pct=round(100 * per_sm * warps / warps_sm),
                SMs=sms, waves=round(waves, 2), waves_run=full, last_wave_fill_pct=round(100 * waves / full))


def convert(db, out, sms=None, flops=None):
    sms = sms or {}
    c = sqlite3.connect(db)
    S = dict(c.execute("select id, value from StringIds"))
    K = list(c.execute("select deviceId, streamId, start, end, shortName, demangledName, correlationId, "
                       "gridX*gridY*gridZ, blockX*blockY*blockZ, registersPerThread, staticSharedMemory+dynamicSharedMemory "
                       "from CUPTI_ACTIVITY_KIND_KERNEL"))
    gpu = {r[0]: r for r in c.execute("select cuDevice, smCount, maxRegistersPerSm, maxShmemPerSm, maxWarpsPerSm, "
                                     "maxBlocksPerSm from TARGET_INFO_GPU")} if "TARGET_INFO_GPU" in _tables(c) else {}
    if not K:
        sys.exit(f"{db}: no kernels (was the capture range empty?)")
    t0 = min(k[2] for k in K)
    us = lambda t: (t - t0) / 1000
    ev, by_stream = [], collections.defaultdict(list)
    for d, s, st, en, sn, dn, cid, grid, block, regs, smem in K:
        name = S.get(sn) or S.get(dn) or "kernel"
        by_stream[(d, s)].append(name)
        args = dict(full=(S.get(dn) or "")[:300], grid_blocks=grid, threads_per_block=block, regs_per_thread=regs,
                    smem_per_block_KB=round(smem / 1024, 1))
        args.update(_occupancy(gpu.get(d), sms.get((d, s)), block, regs, smem, grid))
        ev.append(dict(ph="X", pid=100 + d, tid=s, ts=us(st), dur=(en - st) / 1000, name=name[:80], cat="kernel", args=args))
        ev.append(dict(ph="f", bp="e", pid=100 + d, tid=s, ts=us(st), id=cid, name="launch", cat="flow"))
    copy_streams = set()
    for d, s, st, en, k, b, cid in c.execute("select deviceId, streamId, start, end, copyKind, bytes, correlationId "
                                             "from CUPTI_ACTIVITY_KIND_MEMCPY"):
        copy_streams.add((d, s))
        ev.append(dict(ph="X", pid=100 + d, tid=s, ts=us(st), dur=(en - st) / 1000, cat="memcpy",
                       name=f"memcpy {COPY.get(k, k)} {b / 1e6:.2f} MB", args=dict(GBps=round(b / max(en - st, 1), 1))))
        ev.append(dict(ph="f", bp="e", pid=100 + d, tid=s, ts=us(st), id=cid, name="launch", cat="flow"))
    R = list(c.execute("select globalTid, start, end, nameId, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME"))
    main = collections.Counter(r[0] for r in R).most_common(1)[0][0]
    for g, st, en, n, cid in R:
        if g == main:
            ev.append(dict(ph="X", pid=1, tid=2, ts=us(st), dur=(en - st) / 1000, name=S.get(n, "cuda")[:60], cat="cuda_api"))
            ev.append(dict(ph="s", pid=1, tid=2, ts=us(st), id=cid, name="launch", cat="flow"))
    if "NVTX_EVENTS" in _tables(c):
        cols = {r[1] for r in c.execute("pragma table_info(NVTX_EVENTS)")}
        tx = "coalesce(text, (select value from StringIds where id = textId))" if "textId" in cols else "text"
        for st, en, t, g in c.execute(f"select start, end, {tx}, globalTid from NVTX_EVENTS "
                                      "where end is not null and eventType in (59, 60)"):
            if g != main or not t:
                continue
            lvl = 0 if t.startswith("chunk") else 1 if t in ("fwd", "decode") else 2 if t[:1] == "L" and t[1:].isdigit() else 3
            ev.append(dict(ph="X", pid=1, tid=10 + lvl, ts=us(st), dur=(en - st) / 1000, name=t, cat="nvtx"))
    ev += _counters(K, c, us, by_stream)
    ev += _interference(K, S, us, by_stream, ev, _slots(K, R, c, main))
    if flops:
        ev += _layer_mfu(K, R, c, S, main, us, flops,
                         lambda d, s_: sms.get((d, s_), gpu[d][1]) / gpu[d][1] if d in gpu else 1.0)
    ev += _pin_percent_axes(ev)
    meta = [dict(ph="M", pid=1, name="process_name", args=dict(name="CPU: launch thread")),
            dict(ph="M", pid=1, tid=2, name="thread_name", args=dict(name="CUDA API calls"))]
    meta += [dict(ph="M", pid=1, tid=10 + i, name="thread_name", args=dict(name=n))
             for i, n in enumerate(["chunk", "forward / decode", "layer", "stage"])]
    for d in sorted({d for d, _ in by_stream} | {d for d, _ in copy_streams}):
        meta.append(dict(ph="M", pid=100 + d, name="process_name", args=dict(name=f"GPU {d}")))
    for (d, s) in set(by_stream) | copy_streams:
        nm = _stream_name(by_stream[(d, s)]) if by_stream.get((d, s)) else "copies"
        meta.append(dict(ph="M", pid=100 + d, tid=s, name="thread_name", args=dict(name=f"{nm} (stream {s})")))
    with gzip.open(out, "wt") as f:
        json.dump(dict(traceEvents=meta + ev, displayTimeUnit="ms"), f)
    return len(ev)


def _pin_percent_axes(ev):
    """Perfetto scales each counter track's y-axis to that track's own maximum, so 64 % of peak fills the track like
    100 %. A 1 us point at 100 just before each %-track's first sample pins the axis to 0-100 (invisible at any zoom)."""
    first = {}
    for e in ev:
        if e.get("ph") == "C" and "%" in e["name"]:
            k = (e["pid"], e["name"])
            if k not in first or e["ts"] < first[k]["ts"]:
                first[k] = e
    return [dict(ph="C", pid=p, name=n, ts=e["ts"] - 0.001, args={list(e["args"])[0]: 100}) for (p, n), e in first.items()]


def _slots(K, R, c, main):
    """Position of each kernel in the program: (innermost NVTX range name open on the launching thread, n-th kernel of
    the same name inside that range instance). One kernel name (e.g. a GEMM) runs many sizes with the same launch
    shape; its position tells them apart. Empty when the capture has no NVTX ranges."""
    import bisect
    if "NVTX_EVENTS" not in _tables(c):
        return {}
    cols = {r[1] for r in c.execute("pragma table_info(NVTX_EVENTS)")}
    tx = "coalesce(text, (select value from StringIds where id = textId))" if "textId" in cols else "text"
    N = sorted((st, en, t) for st, en, t, g in c.execute(f"select start, end, {tx}, globalTid from NVTX_EVENTS "
               "where end is not null and eventType in (59, 60)") if g == main and t)
    leaf = [x for i, x in enumerate(N) if not (i + 1 < len(N) and N[i + 1][0] < x[1])]  # ranges with nothing nested
    starts = [x[0] for x in leaf]
    call = {cid: st for g, st, en, n, cid in R if g == main}
    count, out = collections.Counter(), {}
    for k in sorted(K, key=lambda k: call.get(k[6], 0)):
        t = call.get(k[6])
        if t is None:
            continue
        i = bisect.bisect_right(starts, t) - 1
        if i < 0 or t > leaf[i][1]:
            continue
        key = (k[0], k[1], i, k[4])
        out[(k[0], k[1], k[2])] = (leaf[i][2], count[key])
        count[key] += 1
    return out


def _interference(K, S, us, by_stream, ev, slots, bin_ns=1_000_000, min_ns=20_000):
    """Co-running kernels on other compute streams of the same GPU (e.g. a decoder next to a DiT share on separate
    green-context SMs) still share L2 and DRAM bandwidth. For every kernel >= 20 us: the co-runner kernel class that
    overlapped it most, and its slowdown vs the median duration of the same kernel (same name and launch shape: one
    GEMM kernel name runs many sizes with one launch shape, so also its position: NVTX stage + n-th of that name) on
    the same stream when nothing overlapped. Added to the kernel's args, plus a per-stream counter track (time-weighted slowdown % per 1 ms)."""
    import bisect
    import re
    cls = lambda n: re.sub(r"[_<(].*$", "", (n or "kernel").split("::")[-1])[:24]
    by = collections.defaultdict(list)
    for k in K:
        by[(k[0], k[1])].append(k)
    compute = {key for key, names in by_stream.items() if names}
    out, slow_args = [], {}
    for (d, s), ks in by.items():
        others = sorted(k for (dd, ss), kk in by.items() if dd == d and ss != s and (dd, ss) in compute for k in kk)
        if not others:
            continue
        starts = [o[2] for o in others]
        ov = []
        for k in ks:
            st, en = k[2], k[3]
            i, best = bisect.bisect_right(starts, en) - 1, collections.Counter()
            while i >= 0 and i > bisect.bisect_right(starts, st) - 200:
                o = others[i]
                if o[3] > st:
                    best[cls(S.get(o[4]))] += min(o[3], en) - max(o[2], st)
                i -= 1
            ov.append(best.most_common(1)[0][0] if best else None)
        shape = lambda k: (k[4], k[7], k[8], slots.get((k[0], k[1], k[2])))  # name, grid, block, position
        base = collections.defaultdict(list)
        for k, o in zip(ks, ov):
            if o is None:
                base[shape(k)].append(k[3] - k[2])
        med = {n: sorted(v)[len(v) // 2] for n, v in base.items() if len(v) >= 5}
        t0, nb = min(k[2] for k in K), (max(k[3] for k in K) - min(k[2] for k in K)) // bin_ns + 1
        act, ref = [0] * nb, [0] * nb
        for k, o in zip(ks, ov):
            dur = k[3] - k[2]
            if dur < min_ns or shape(k) not in med:
                continue
            m = med[shape(k)]
            slow_args[(d, s, k[2])] = dict(co_running=o or "nothing", slowdown_vs_alone_pct=round(100 * (dur / m - 1), 1))
            b = (k[2] - t0) // bin_ns
            act[b] += dur
            ref[b] += m
        nm = _stream_name(by_stream[(d, s)])
        name = f"interference: slowdown % {nm} (stream {s})"
        for i in range(nb):
            if ref[i]:
                out.append(dict(ph="C", pid=100 + d, ts=us(t0 + i * bin_ns), name=name,
                                args={"slowdown %": round(100 * (act[i] / ref[i] - 1), 1)}))
        out.append(dict(ph="C", pid=100 + d, ts=us(max(k[3] for k in ks)), name=name, args={"slowdown %": 0}))  # end at the stream's last kernel
    t_first = min(k[2] for k in K)
    for e in ev:
        if e.get("cat") == "kernel":
            a = slow_args.get((e["pid"] - 100, e["tid"], round(e["ts"] * 1000) + t_first))
            if a:
                e["args"].update(a)
    return out


def _counters(K, c, us, by_stream, bin_ns=1_000_000):
    """Per 1 ms: busy % of each GPU stream (union of its kernels), PCIe GB/s per direction per GPU, kernels started."""
    ev, t0 = [], min(k[2] for k in K)
    nb = (max(k[3] for k in K) - t0) // bin_ns + 1
    busy, starts = collections.defaultdict(lambda: [0] * nb), collections.defaultdict(lambda: [0] * nb)
    for d, s, st, en, *_ in K:
        starts[d][(st - t0) // bin_ns] += 1
        a = st
        while a < en:
            b_ = (a - t0) // bin_ns
            e = min(en, t0 + (b_ + 1) * bin_ns)
            busy[(d, s)][b_] += e - a
            a = e
    for (d, s), v in busy.items():
        nm = _stream_name(by_stream[(d, s)])
        for i, x in enumerate(v):
            ev.append(dict(ph="C", pid=100 + d, ts=us(t0 + i * bin_ns), name=f"busy % {nm} (stream {s})",
                           args={"busy %": round(min(100 * x / bin_ns, 100), 1)}))
    for d, v in starts.items():
        for i, x in enumerate(v):
            ev.append(dict(ph="C", pid=100 + d, ts=us(t0 + i * bin_ns), name="kernels started per ms", args={"kernels": x}))
    gbs = collections.defaultdict(lambda: [0] * nb)
    for d, st, en, k, b in c.execute("select deviceId, start, end, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY "
                                     "where copyKind in (1, 2)"):
        i = (st - t0) // bin_ns
        if 0 <= i < nb:
            gbs[(d, COPY[k])][i] += b
    for (d, k), v in gbs.items():
        for i, x in enumerate(v):
            ev.append(dict(ph="C", pid=100 + d, ts=us(t0 + i * bin_ns), name=f"PCIe {k} GB/s", args={"GB/s": round(x / bin_ns, 1)}))
    return ev


def _layer_mfu(K, R, c, S, main, us, f, share):
    """Achieved TFLOP/s and % of peak per (GPU, layer pass) for GEMMs and attention, from a FLOP model of the layer
    (`f`: per-device GEMM and attention FLOPs per layer pass, peaks). Kernels map to a layer through their launch call
    and the NVTX `L<n>` range open on the CPU at that moment. Peaks scale with the stream's SM share (green contexts)."""
    import bisect
    call = {cid: st for g, st, en, n, cid in R if g == main}
    cols = {r[1] for r in c.execute("pragma table_info(NVTX_EVENTS)")}
    tx = "coalesce(text, (select value from StringIds where id = textId))" if "textId" in cols else "text"
    L = sorted((st, en, t) for st, en, t, g in c.execute(f"select start, end, {tx}, globalTid from NVTX_EVENTS "
               "where end is not null and eventType in (59, 60)") if g == main and t and t[:1] == "L" and t[1:].isdigit())
    starts = [x[0] for x in L]
    acc = collections.defaultdict(lambda: dict(gemm=0, attn=0, t0=None, t1=0, share=1.0, ngemm=0))
    for d, s, st, en, sn, dn, cid, *_ in K:
        t = call.get(cid)
        if t is None:
            continue
        i = bisect.bisect_right(starts, t) - 1
        if i < 0 or t > L[i][1]:
            continue
        name = (S.get(sn) or "") + " " + (S.get(dn) or "")
        a = acc[(d, i)]
        cls = "attn" if "qk_int_sv" in name else "gemm" if ("device_kernel" in name or "gemm" in name.lower()) and "fprop" not in name else None
        if cls:
            a["share"] = share(d, s)
            a[cls] += en - st
            a["ngemm"] += cls == "gemm"
            a["t0"] = st if a["t0"] is None else min(a["t0"], st)
            a["t1"] = max(a["t1"], en)
    # layer passes with more GEMM kernels than usual run extra GEMMs (e.g. camera conditioning once per chunk):
    # add the model's "gemm_extra" FLOPs there, so they read as work, not as a drop in MFU
    usual = {d: collections.Counter(a["ngemm"] for (dd, _), a in acc.items() if dd == d).most_common(1)[0][0]
             for d in {d for d, _ in acc}}
    ev, tot = [], collections.defaultdict(lambda: dict(gemm=0, attn=0, n=0, share=1.0, flops_gemm=0))
    for (d, i), a in sorted(acc.items(), key=lambda x: x[1]["t0"] or 0):
        fd = f["devices"].get(str(d))
        if not fd or not a["t0"]:
            continue
        args = {}
        fl = dict(gemm=fd["gemm"] + (fd.get("gemm_extra", 0) if a["ngemm"] > usual[d] else 0), attn=fd["attn"])
        tot[d]["flops_gemm"] += fl["gemm"] if a["gemm"] else 0
        for cls, peak in (("gemm", f["peak_gemm"]), ("attn", f["peak_attn"])):
            if a[cls]:
                tf = fl[cls] / (a[cls] / 1e9) / 1e12
                args[f"{cls} TFLOP/s"] = round(tf)
                args[f"{cls} % of peak"] = round(100 * tf / (peak * a["share"]), 1)
                tot[d][cls] += a[cls]
        tot[d]["n"] += 1
        tot[d]["share"] = a["share"]
        for k, v in args.items():
            ev.append(dict(ph="C", pid=100 + d, ts=us(a["t0"]), name=f"MFU per layer: {k}", args={k: v}))
    summ = {}
    for d, t in tot.items():
        fd = f["devices"][str(d)]
        for cls, peak in (("gemm", f["peak_gemm"]), ("attn", f["peak_attn"])):
            if t[cls]:
                tf = (t["flops_gemm"] if cls == "gemm" else fd[cls] * t["n"]) / (t[cls] / 1e9) / 1e12
                summ[f"GPU {d} {cls}: TFLOP/s"] = round(tf)
                summ[f"GPU {d} {cls}: % of peak"] = round(100 * tf / (peak * t["share"]), 1)
                summ[f"GPU {d} {cls}: SMs share of peak"] = round(t["share"], 3)
                summ[f"GPU {d} {cls}: ms per layer pass"] = round(t[cls] / t["n"] / 1e6, 3)
    summ["peaks (TFLOP/s)"] = f"gemm {f['peak_gemm']}, attention {f['peak_attn']}; {f.get('note', '')}"
    t_all = [k[2] for k in K] + [k[3] for k in K]
    ev.append(dict(ph="X", pid=0, tid=0, ts=us(min(t_all)), dur=(max(t_all) - min(t_all)) / 1000,
                   name="summary: MFU over the capture (click)", cat="summary", args=summ))
    ev.append(dict(ph="M", pid=0, name="process_name", args=dict(name="Summary")))
    return ev


def open_in_ui(path, commands=EXPAND_ALL):
    """Serve `path` once on 127.0.0.1:9001 (the only local origin ui.perfetto.dev's CSP allows; same as Perfetto's
    open_trace_in_ui) and open the UI on it, running `commands` (UI automation) once the trace has loaded."""
    import http.server
    import socketserver
    import urllib.parse
    import webbrowser
    path = os.path.abspath(path)
    fname = os.path.basename(path)

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=os.path.dirname(path), **k)

        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "https://ui.perfetto.dev")
            self.send_header("Cache-Control", "no-cache")
            super().end_headers()

        def do_GET(self):
            if self.path != "/" + fname:
                return self.send_error(404)
            self.server.done = True
            super().do_GET()

        def do_POST(self):
            self.send_error(404)

        def log_message(self, *a):
            pass

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", 9001), H) as httpd:
        q = f"url=http://127.0.0.1:9001/{fname}"
        if commands:
            q += "&startupCommands=" + urllib.parse.quote(json.dumps(commands))
        webbrowser.open_new_tab(f"https://ui.perfetto.dev/#!/?{q}")
        httpd.done = False
        while not httpd.done:
            httpd.handle_request()
    print("opened in ui.perfetto.dev")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", help=".sqlite (nsys export) or .nsys-rep, or an existing .json/.json.gz to just open")
    ap.add_argument("-o", "--out")
    ap.add_argument("--open", action="store_true", help="open in ui.perfetto.dev (local browser)")
    ap.add_argument("--sms", default="", help="SMs per green-context stream for occupancy, e.g. 1:24=40,1:27=130")
    ap.add_argument("--flops", help="JSON FLOP model for per-layer MFU: {devices: {dev: {gemm, attn}}, peak_gemm, "
                                    "peak_attn, note} (FLOPs per layer pass, peaks in TFLOP/s)")
    a = ap.parse_args()
    if a.capture.endswith((".json", ".json.gz")):
        out = a.capture
    else:
        out = a.out or a.capture.rsplit(".", 1)[0] + "_perfetto.json.gz"
        sms = {(int(k.split(":")[0]), int(k.split(":")[1])): int(v) for k, v in
               (x.split("=") for x in a.sms.split(",") if x)}
        flops = json.load(open(a.flops)) if a.flops else None
        n = convert(_sqlite(a.capture), out, sms, flops)
        print(f"{out}: {n} events, {os.path.getsize(out) / 1e6:.1f} MB")
    if a.open:
        open_in_ui(out)


if __name__ == "__main__":
    main()
