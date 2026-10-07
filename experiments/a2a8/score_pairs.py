"""Score clips against a reference clip, per chunk and over the whole clip (8-bit exchange quality test).

  python experiments/a2a8/score_pairs.py --pairs fast2:fast1 qk1:fast1 ... --dir /workspace/runs/a2a8

Each pair is cand:ref (mp4 basenames). Chunks: the first is 13 frames, then 16 each. Metrics: PSNR, mean
|diff| (/255) on every frame, LPIPS (alex) on every 4th frame. Pairs of two identical-config runs give the
noise band; a candidate is "inside noise" if its numbers fall within that band. Also reference-free, per clip:
Laplacian sharpness (first chunk and whole clip) and flicker (mean |frame-to-frame diff|), supporting only
(Laplacian varies ~18 % between identical runs).
"""
import argparse
import os

import imageio.v2 as imageio
import cv2
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", nargs="+", required=True)
ap.add_argument("--dir", required=True)
ap.add_argument("--out", default=None)
a = ap.parse_args()

import lpips  # noqa: E402

net = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
cache = {}


def load(name):
    if name not in cache:
        cache[name] = np.stack(imageio.mimread(os.path.join(a.dir, name + ".mp4"), memtest=False)).astype(np.float32)
    return cache[name]


def lp(x, y):
    t = lambda f: torch.from_numpy(f).permute(2, 0, 1)[None].cuda() / 127.5 - 1
    with torch.no_grad():
        return float(net(t(x), t(y)))


rows = []
for pair in a.pairs:
    c, r = pair.split(":")
    x, y = load(c), load(r)
    n = min(len(x), len(y))
    bounds = [0, 13] + list(range(29, n + 1, 16))
    per_chunk = []
    for lo, hi in zip(bounds, bounds[1:]):
        d = np.abs(x[lo:hi] - y[lo:hi])
        mse = (d ** 2).mean(axis=(1, 2, 3))
        psnr = float(np.mean(10 * np.log10(255 ** 2 / np.maximum(mse, 1e-10))))
        per_chunk.append((psnr, float(d.mean()), float(np.mean([lp(x[i], y[i]) for i in range(lo, hi, 4)]))))
    pc = np.array(per_chunk)
    sharp = lambda fs: float(np.mean([cv2.Laplacian(cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for f in fs]))
    flick = lambda fs: float(np.abs(np.diff(fs, axis=0)).mean())
    extra = dict(sharp_c1=sharp(x[:13]), sharp_c1_ref=sharp(y[:13]), sharp_clip=sharp(x[:n:4]), sharp_clip_ref=sharp(y[:n:4]),
                 flicker=flick(x[:n]), flicker_ref=flick(y[:n]))
    rows.append((pair, n, pc[0, 0], pc[0, 2], pc[:, 0].mean(), pc[:, 1].mean(), pc[:, 2].mean(), pc[-1, 0], pc[-1, 2],
                 extra["sharp_c1"], extra["sharp_c1_ref"], extra["sharp_clip"], extra["sharp_clip_ref"], extra["flicker"], extra["flicker_ref"]))
    print(f"{pair}\tframes {n}\tper-chunk PSNR " + " ".join(f"{p:.1f}" for p in pc[:, 0]), flush=True)

hdr = ("pair\tframes\tpsnr_chunk1\tlpips_chunk1\tpsnr_clip\tmeandiff_clip_255\tlpips_clip\tpsnr_last_chunk\tlpips_last_chunk"
       "\tsharp_chunk1\tsharp_chunk1_ref\tsharp_clip\tsharp_clip_ref\tflicker\tflicker_ref")
lines = [hdr] + ["\t".join([r[0], str(r[1])] + [f"{v:.4f}" if i in (1, 4, 6) else (f"{v:.0f}" if 7 <= i <= 10 else f"{v:.2f}") for i, v in enumerate(r[2:])]) for r in rows]
print("\n".join(lines))
if a.out:
    open(a.out, "w").write("\n".join(lines) + "\n")
