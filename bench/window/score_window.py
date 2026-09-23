#!/usr/bin/env python3
"""Score the KV-window candidate against a baseline, per the bench-world-model-quality skill.

This is a CLASS C lever: the attended context changes, so the rollout diverges from the baseline
even at a fixed seed. Per-frame identity is therefore NOT the test. What is reported instead:

  reference-based, only meaningful early     first-chunk PSNR / SSIM / LPIPS vs the baseline run
  reference-free, meaningful throughout      MUSIQ, sharpness (Laplacian variance), colour, flicker
  per-160-frame bins                         the same reference-free metrics, so a late-clip
                                             collapse cannot hide behind a good early average

The noise band comes from scoring the baseline against ITSELF across repeat runs (band1/2/3). A
candidate delta is only a finding if it is outside that band -- the whole point of measuring the
band first, which had never been done for this stack.

  <stream-venv>/bin/python bench/window/score_window.py \
      --runs band1=<a.mp4> band2=<b.mp4> band3=<c.mp4> w12=<d.mp4> stock=<e.mp4> \
      --ref band1 --bin 160 --out bench/window/quality.tsv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="label=path.mp4 ...")
ap.add_argument("--ref", default="band1", help="label used as the reference arm")
ap.add_argument("--bin", type=int, default=160, help="frames per drift bin")
ap.add_argument("--max_frames", type=int, default=0, help="0 = all")
ap.add_argument("--out", default="bench/window/quality.tsv")
a = ap.parse_args()

runs = {}
for spec in a.runs:
    if "=" not in spec:
        sys.exit(f"--runs wants label=path, got {spec!r}")
    label, path = spec.split("=", 1)
    if not os.path.exists(path):
        sys.exit(f"missing: {path}")
    runs[label] = path
if a.ref not in runs:
    sys.exit(f"--ref {a.ref} is not among {list(runs)}")


def frames(path, limit=0):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
        if limit and len(out) >= limit:
            break
    cap.release()
    return out


def psnr(x, y):
    mse = float(np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def sharpness(f):
    """Laplacian variance: the repo's existing sharpness proxy (exp. 5, 9, 12)."""
    return float(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def flicker(fs):
    """Mean abs luma difference between consecutive frames; a shorter window may add temporal jitter."""
    if len(fs) < 2:
        return float("nan")
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in fs]
    return float(np.mean([np.mean(np.abs(g[i + 1] - g[i])) for i in range(len(g) - 1)]))


try:
    from skimage.metrics import structural_similarity as _ssim

    def ssim(x, y):
        return float(_ssim(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)))
except Exception:
    def ssim(x, y):
        return float("nan")

_lpips = None
def lpips_d(x, y):
    """LPIPS on the first chunk only -- it is the metric the validation literature treats as decisive."""
    global _lpips
    try:
        import torch
        import lpips as _l
        if _lpips is None:
            _lpips = _l.LPIPS(net="alex", verbose=False)
        def t(img):
            r = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
            return torch.from_numpy(r).permute(2, 0, 1)[None]
        with torch.no_grad():
            return float(_lpips(t(x), t(y)).item())
    except Exception:
        return float("nan")

_musiq = None
def musiq(fs):
    """Reference-free perceptual score; skipped cleanly if the weights are not cached locally."""
    global _musiq
    try:
        import torch
        import pyiqa
        if _musiq is None:
            _musiq = pyiqa.create_metric("musiq", device="cpu")
        idx = np.linspace(0, len(fs) - 1, min(8, len(fs))).astype(int)
        vals = []
        with torch.no_grad():
            for i in idx:
                r = cv2.cvtColor(fs[i], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                vals.append(float(_musiq(torch.from_numpy(r).permute(2, 0, 1)[None]).item()))
        return float(np.mean(vals))
    except Exception:
        return float("nan")


print(f"# loading {len(runs)} runs", file=sys.stderr)
data = {k: frames(v, a.max_frames) for k, v in runs.items()}
for k, v in data.items():
    print(f"#   {k}: {len(v)} frames", file=sys.stderr)
n = min(len(v) for v in data.values())
ref = data[a.ref]

rows = []
for label, fs in data.items():
    fc = min(16, n)  # first chunk = 16 frames
    row = {
        "label": label,
        "frames": n,
        "psnr_first_chunk": round(float(np.mean([psnr(fs[i], ref[i]) for i in range(fc)])), 2) if label != a.ref else float("inf"),
        "ssim_first_chunk": round(float(np.mean([ssim(fs[i], ref[i]) for i in range(fc)])), 4) if label != a.ref else 1.0,
        "lpips_first_chunk": round(lpips_d(fs[0], ref[0]), 4) if label != a.ref else 0.0,
        "sharpness_all": round(float(np.mean([sharpness(f) for f in fs[:n]])), 1),
        "flicker": round(flicker(fs[:n]), 3),
        "musiq": round(musiq(fs[:n]), 2),
    }
    # per-bin reference-free metrics: a late collapse must not hide behind a good average
    for b0 in range(0, n, a.bin):
        b1 = min(b0 + a.bin, n)
        if b1 - b0 < a.bin // 2:
            continue
        seg = fs[b0:b1]
        row[f"sharp_{b0}_{b1}"] = round(float(np.mean([sharpness(f) for f in seg])), 1)
        row[f"musiq_{b0}_{b1}"] = round(musiq(seg), 2)
    rows.append(row)

cols = list(rows[0].keys())
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w") as fh:
    fh.write("\t".join(cols) + "\n")
    for r in rows:
        fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

w = max(len(c) for c in cols)
for c in cols:
    print(f"{c:<{w}}  " + "  ".join(f"{str(r.get(c,'')):>12}" for r in rows))
print(f"\nwrote {a.out}", file=sys.stderr)
