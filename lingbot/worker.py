"""Warm bench worker: keep the weights loaded and the model compiled between runs.

    python -m lingbot.worker run -- --preset fast --bench --frame_num 157 [any lingbot.generate args]
    python -m lingbot.worker status | stop

`run` sends the arguments to a long-lived worker for the current *code hash* (git tree + uncommitted diff +
the preset's LINGBOT_* settings + visible GPUs) and streams its output. The first run for a hash pays the cold
start (weights, Dynamo tracing, Inductor compile); later runs with the same hash only generate. Anything that
changes the hash, or how the pipeline is built (checkpoint, window, sink, T5 on CPU), starts a fresh worker and
stops the old one first so two never hold the GPUs. Per-run settings that leave the compiled model alone run
warm: --bench / --bench_e2e, --frame_num, --decoder_gpu, prompt, image, seed, output file.

Not handled here (run lingbot.generate directly): --preset stock and torchrun launches.
"""
import glob
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

SOCK_DIR = os.environ.get("LINGBOT_WORKER_DIR", "/tmp")
BENCH_ENV = ("LINGBOT_BENCH_TIMING", "LINGBOT_BENCH_E2E", "LINGBOT_VAE_STREAM", "LINGBOT_DECODE_FIRST")
BUILD_ARGS = ("task", "ckpt_dir", "assets_dir", "t5_cpu", "local_attn_size", "sink_size")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _preset(argv):
    for i, a in enumerate(argv):
        if a == "--preset" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--preset="):
            return a.split("=", 1)[1]
    return "fast"


def code_hash(preset):
    """Everything that changes what gets compiled: source, settings, GPUs, interpreter."""
    from lingbot.presets import PRESETS
    git = lambda *a: subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout
    env = dict(PRESETS.get(preset, {}))
    env.update({k: v for k, v in os.environ.items() if k.startswith("LINGBOT_") and k not in BENCH_ENV})
    parts = [git("rev-parse", "HEAD^{tree}"), git("diff", "HEAD", "--", "lingbot", "reference"),
             git("ls-files", "--others", "--exclude-standard", "lingbot"), json.dumps(sorted(env.items())),
             preset, os.environ.get("CUDA_VISIBLE_DEVICES", ""), sys.executable]
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:12]


def _sock(key):
    return os.path.join(SOCK_DIR, f"lingbot-worker-{key}.sock")


def _connect(path):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(path)
    return s


def _send(path, msg):
    s = _connect(path)
    s.sendall((json.dumps(msg) + "\n").encode())
    return s


def _live_workers():
    out = []
    for p in glob.glob(os.path.join(SOCK_DIR, "lingbot-worker-*.sock")):
        try:
            _connect(p).close()
            out.append(p)
        except OSError:
            os.remove(p)  # stale socket from a dead worker
    return out


def _stop(path):
    try:
        s = _send(path, {"cmd": "stop"})
        s.recv(16)
        s.close()
    except OSError:
        pass
    for _ in range(120):  # wait until the process has let go of the GPUs
        if not os.path.exists(path):
            return
        time.sleep(0.5)


def _ensure_worker(key, preset):
    path = _sock(key)
    for other in _live_workers():
        if other != path:
            print(f"[worker] stopping {os.path.basename(other)} (different code hash)", flush=True)
            _stop(other)
    if path in _live_workers():
        return path
    log = path.replace(".sock", ".log")
    print(f"[worker] starting worker {key} (cold start: load + compile on first run); log {log}", flush=True)
    subprocess.Popen([sys.executable, "-m", "lingbot.worker", "serve", key, preset], cwd=os.getcwd(),
                     stdout=open(log, "a"), stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(600):
        if os.path.exists(path):
            try:
                _connect(path).close()
                return path
            except OSError:
                pass
        time.sleep(0.5)
    raise SystemExit(f"[worker] did not come up; see {log}")


def run(argv):
    preset = _preset(argv)
    if preset == "stock":
        raise SystemExit("[worker] --preset stock: run python -m lingbot.generate directly")
    key = code_hash(preset)
    for attempt in range(2):
        path = _ensure_worker(key, preset)
        s = _send(path, {"cmd": "run", "argv": argv, "cwd": os.getcwd()})
        buf, code = b"", None
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for ln in lines:
                if ln.startswith(b"__EXIT__ "):
                    code = ln.split()[1].decode()
                else:
                    sys.stdout.buffer.write(ln + b"\n")
                    sys.stdout.flush()
        s.close()
        if code == "rebuild":  # build arguments changed: restart the worker once
            _stop(path)
            continue
        return int(code) if code not in (None, "rebuild") else 1
    return 1


# ---------------------------------------------------------------- server side


class _Stream:
    """File-like object that forwards writes to the current client connection."""

    def __init__(self):
        self.conn = None

    def write(self, s):
        if self.conn is not None and s:
            try:
                self.conn.sendall(s.encode() if isinstance(s, str) else s)
            except OSError:
                self.conn = None
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return False


def serve(key, preset):
    path = _sock(key)
    stream = _Stream()
    real_out, real_err = sys.stdout, sys.stderr
    sys.argv = [sys.argv[0], "--preset", preset]  # lingbot.generate applies the preset at import
    import logging

    import torch

    import lingbot.generate as G
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(stream)], force=True)
    pipe, built_with = None, None
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if os.path.exists(path):
        os.remove(path)
    srv.bind(path)
    srv.listen(1)
    print(f"worker {key} ({preset}) listening on {path}", file=real_out, flush=True)
    try:
        while True:
            conn, _ = srv.accept()
            line = conn.makefile().readline()
            if not line.strip():  # a liveness probe: connect and close
                conn.close()
                continue
            msg = json.loads(line)
            if msg["cmd"] == "stop":
                conn.sendall(b"ok")
                conn.close()
                break
            stream.conn = conn
            sys.stdout = sys.stderr = stream
            code = "0"
            try:
                os.chdir(msg["cwd"])
                sys.argv = [sys.argv[0], *msg["argv"]]
                args = G._parse_args()
                build = {k: getattr(args, k) for k in BUILD_ARGS}
                if pipe is not None and build != built_with:
                    code = "rebuild"
                else:
                    for k in BENCH_ENV:  # per-run switches, read inside generate()
                        os.environ.pop(k, None)
                    if args.bench:
                        os.environ["LINGBOT_BENCH_TIMING"] = "1"
                    if args.bench_e2e:
                        for k in ("LINGBOT_BENCH_E2E", "LINGBOT_VAE_STREAM", "LINGBOT_DECODE_FIRST"):
                            os.environ[k] = "1"
                    cfg = G.WAN_CONFIGS[args.task]
                    if pipe is None:
                        t = time.perf_counter()
                        pipe = G.build_pipeline(args, cfg, 0, 0)
                        built_with = build
                        print(f"[worker] pipeline built in {time.perf_counter() - t:.0f} s (kept for later runs)")
                    else:
                        print("[worker] warm: reusing loaded weights and compiled model")
                        pipe.set_decoder_device(args.decoder_gpu)
                    for attr in ("bench_chunk_s", "bench_vae_decode_s", "bench_ready_ms"):
                        if hasattr(pipe, attr):
                            delattr(pipe, attr)
                    t = time.perf_counter()
                    G.run_generation(args, pipe, cfg)
                    torch.cuda.synchronize()
                    print(f"[worker] run took {time.perf_counter() - t:.1f} s")
            except SystemExit as e:
                code = str(e.code if isinstance(e.code, int) else 1)
                if not isinstance(e.code, int):
                    print(e)
            except Exception:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                code = "1"
            finally:
                sys.stdout, sys.stderr = real_out, real_err
                try:
                    conn.sendall(f"\n__EXIT__ {code}\n".encode())
                    conn.close()
                except OSError:
                    pass
                stream.conn = None
    finally:
        srv.close()
        if os.path.exists(path):
            os.remove(path)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "serve":
        serve(sys.argv[2], sys.argv[3])
    elif cmd == "run":
        argv = sys.argv[2:]
        if argv and argv[0] == "--":
            argv = argv[1:]
        sys.exit(run(argv))
    elif cmd == "stop":
        for p in _live_workers():
            _stop(p)
            print(f"stopped {os.path.basename(p)}")
    else:
        live = _live_workers()
        print("\n".join(os.path.basename(p) for p in live) or "no worker running")


if __name__ == "__main__":
    main()
