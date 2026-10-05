#!/usr/bin/env python3
"""H7: does the SM clock (or power cap) drop under sustained INT8 load so the
effective peak is below the 838 TOPS @ 2.41 GHz spec? Samples nvidia-smi at
100 ms, writes a TSV, prints a summary."""
import argparse
import io
import os
import statistics
import subprocess
import sys
import time

SPEC_TOPS = 838.0
SPEC_CLOCK_MHZ = 2410.0
UTIL_THRESHOLD = 90.0
FIELDS = [
    "timestamp", "clocks.sm", "clocks.mem", "power.draw", "temperature.gpu",
    "clocks_throttle_reasons.active", "clocks_throttle_reasons.sw_power_cap",
    "clocks_throttle_reasons.hw_slowdown", "clocks_throttle_reasons.sw_thermal_slowdown",
    "utilization.gpu",
]
INTERVAL_S = 0.1

FAKE_CSV = """\
2026/09/22 10:00:00.000, 2400, 14001, 120.5, 45, 0x0000000000000000, Not Active, Not Active, Not Active, 5
2026/09/22 10:00:00.100, 2385, 14001, 420.3, 55, 0x0000000000000000, Not Active, Not Active, Not Active, 98
2026/09/22 10:00:00.200, 2310, 14001, 575.0, 62, 0x0000000000000004, Active, Not Active, Not Active, 100
2026/09/22 10:00:00.300, 2250, 14001, 574.8, 66, 0x0000000000000004, Active, Not Active, Not Active, 100
2026/09/22 10:00:00.400, 2265, 14001, 573.9, 68, 0x0000000000000004, Active, Not Active, Not Active, 99
2026/09/22 10:00:00.500, 2100, 14001, 200.0, 60, 0x0000000000000000, Not Active, Not Active, Not Active, 40
"""


def parse_line(line):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != len(FIELDS):
        return None
    row = dict(zip(FIELDS, parts))
    try:
        row["clocks.sm"] = float(row["clocks.sm"])
        row["clocks.mem"] = float(row["clocks.mem"])
        row["power.draw"] = float(row["power.draw"])
        row["temperature.gpu"] = float(row["temperature.gpu"])
        row["utilization.gpu"] = float(row["utilization.gpu"])
    except ValueError:
        return None
    return row


def throttled(row):
    # active is a hex bitmask; 0 means no reason is asserted
    try:
        active = int(row["clocks_throttle_reasons.active"], 16)
    except ValueError:
        active = 0
    flags = (
        row["clocks_throttle_reasons.sw_power_cap"],
        row["clocks_throttle_reasons.hw_slowdown"],
        row["clocks_throttle_reasons.sw_thermal_slowdown"],
    )
    return active != 0 or any(f == "Active" for f in flags)


def write_tsv(rows, path):
    with open(path, "w") as f:
        f.write("\t".join(FIELDS) + "\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in FIELDS) + "\n")


def summarize(rows):
    if not rows:
        print("h7: no samples")
        return None
    busy = [r["clocks.sm"] for r in rows if r["utilization.gpu"] > UTIL_THRESHOLD]
    # samples are equally spaced, so sample fraction == time-weighted fraction
    thr_frac = sum(throttled(r) for r in rows) / len(rows)
    mean_power = statistics.fmean(r["power.draw"] for r in rows)
    print(f"h7: samples={len(rows)} busy_samples(util>{UTIL_THRESHOLD:.0f}%)={len(busy)}")
    if not busy:
        print("h7: no samples with utilization > 90 %; clock stats unavailable")
        print(f"h7: throttle_fraction={thr_frac:.3f} mean_power_w={mean_power:.1f}")
        return {"samples": len(rows), "busy_samples": 0, "throttle_fraction": thr_frac,
                "mean_power_w": mean_power}
    med = statistics.median(busy)
    eff = SPEC_TOPS * med / SPEC_CLOCK_MHZ
    print(f"h7: sm_clock_mhz median={med:.0f} min={min(busy):.0f} max={max(busy):.0f}")
    print(f"h7: throttle_fraction={thr_frac:.3f} mean_power_w={mean_power:.1f}")
    print(f"h7: effective_int8_peak_tops={eff:.0f} (= {SPEC_TOPS:.0f} * {med:.0f}/{SPEC_CLOCK_MHZ:.0f})")
    return {"samples": len(rows), "busy_samples": len(busy), "sm_clock_median_mhz": med,
            "sm_clock_min_mhz": min(busy), "sm_clock_max_mhz": max(busy),
            "throttle_fraction": thr_frac, "mean_power_w": mean_power,
            "effective_int8_peak_tops": eff}


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_pidfile(path):
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with open(path) as f:
                return int(f.read().strip())
        except (FileNotFoundError, ValueError):
            time.sleep(0.2)
    sys.exit(f"h7: pidfile {path} not readable after 30 s")


def sample(seconds, pidfile, extra_stop=None):
    cmd = ["nvidia-smi", "--query-gpu=" + ",".join(FIELDS),
           "--format=csv,noheader,nounits", "-lms", str(int(INTERVAL_S * 1000))]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pid = read_pidfile(pidfile) if pidfile else None
    rows = []
    start = time.time()
    try:
        for line in proc.stdout:
            row = parse_line(line)
            if row:
                rows.append(row)
            if seconds is not None and time.time() - start >= seconds:
                break
            if pid is not None and not pid_alive(pid):
                break
            if extra_stop and extra_stop():
                break
    finally:
        proc.terminate()
        proc.wait()
    if proc.returncode not in (0, -15) and not rows:
        sys.exit(f"h7: nvidia-smi failed: {proc.stderr.read().strip()}")
    return rows


def run_standalone_load(seconds):
    import torch
    n = 8192
    a = torch.randint(-127, 127, (n, n), dtype=torch.int8, device="cuda")
    b = torch.randint(-127, 127, (n, n), dtype=torch.int8, device="cuda")
    torch._int_mm(a, b)
    torch.cuda.synchronize()
    iters = 0
    start = time.time()
    while time.time() - start < seconds:
        for _ in range(10):
            torch._int_mm(a, b)
        torch.cuda.synchronize()
        iters += 10
    elapsed = time.time() - start
    tops = 2 * n ** 3 * iters / elapsed / 1e12
    print(f"h7: standalone int_mm {iters} iters in {elapsed:.1f} s -> measured {tops:.0f} TOPS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--pidfile")
    ap.add_argument("--standalone", action="store_true",
                    help="run an int8 matmul load in this process while sampling")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "results", "h7_clock.tsv"))
    args = ap.parse_args()

    if args.dry:
        rows = [r for r in (parse_line(l) for l in io.StringIO(FAKE_CSV)) if r]
        summarize(rows)
        return

    if args.standalone and args.seconds is None:
        ap.error("--standalone needs --seconds")
    if args.seconds is None and args.pidfile is None:
        ap.error("need --seconds or --pidfile")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.standalone:
        import threading
        done = threading.Event()

        def load():
            try:
                run_standalone_load(args.seconds)
            finally:
                done.set()

        t = threading.Thread(target=load)
        t.start()
        rows = sample(None, None, extra_stop=done.is_set)
        t.join()
    else:
        rows = sample(args.seconds, args.pidfile)

    write_tsv(rows, args.out)
    print(f"h7: wrote {len(rows)} samples to {args.out}")
    summarize(rows)


if __name__ == "__main__":
    main()
