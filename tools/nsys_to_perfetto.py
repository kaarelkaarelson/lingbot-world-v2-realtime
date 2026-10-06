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


def convert(db, out):
    c = sqlite3.connect(db)
    S = dict(c.execute("select id, value from StringIds"))
    K = list(c.execute("select deviceId, streamId, start, end, shortName, demangledName, correlationId "
                       "from CUPTI_ACTIVITY_KIND_KERNEL"))
    if not K:
        sys.exit(f"{db}: no kernels (was the capture range empty?)")
    t0 = min(k[2] for k in K)
    us = lambda t: (t - t0) / 1000
    ev, by_stream = [], collections.defaultdict(list)
    for d, s, st, en, sn, dn, cid in K:
        name = S.get(sn) or S.get(dn) or "kernel"
        by_stream[(d, s)].append(name)
        ev.append(dict(ph="X", pid=100 + d, tid=s, ts=us(st), dur=(en - st) / 1000, name=name[:80], cat="kernel",
                       args=dict(full=(S.get(dn) or "")[:300])))
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
    tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    if "NVTX_EVENTS" in tables:
        cols = {r[1] for r in c.execute("pragma table_info(NVTX_EVENTS)")}
        tx = "coalesce(text, (select value from StringIds where id = textId))" if "textId" in cols else "text"
        for st, en, t, g in c.execute(f"select start, end, {tx}, globalTid from NVTX_EVENTS "
                                      "where end is not null and eventType in (59, 60)"):
            if g != main or not t:
                continue
            lvl = 0 if t.startswith("chunk") else 1 if t in ("fwd", "decode") else 2 if t[:1] == "L" and t[1:].isdigit() else 3
            ev.append(dict(ph="X", pid=1, tid=10 + lvl, ts=us(st), dur=(en - st) / 1000, name=t, cat="nvtx"))
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
    a = ap.parse_args()
    if a.capture.endswith((".json", ".json.gz")):
        out = a.capture
    else:
        out = a.out or a.capture.rsplit(".", 1)[0] + "_perfetto.json.gz"
        n = convert(_sqlite(a.capture), out)
        print(f"{out}: {n} events, {os.path.getsize(out) / 1e6:.1f} MB")
    if a.open:
        open_in_ui(out)


if __name__ == "__main__":
    main()
