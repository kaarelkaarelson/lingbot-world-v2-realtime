#!/usr/bin/env python3
"""How short can the KV window get before the attention OUTPUT changes? A causal ablation.

Replaces the attention-mass version of this script (bench/window/attn_decay.py). The literature is
clear that attention weights are not a faithful proxy for what a model uses: Jain & Wallace,
"Attention is not Explanation" (arXiv 1902.10186), show attention scores are uncorrelated with
gradient-based importance and that very different attention distributions give identical
predictions; softmax also dilutes mass as O(1/n) with context length, so a long window makes every
individual token look unimportant whether or not it is. Mass is a heuristic; ablation is causal.

The trick that makes this cheap AND precise: ablate on ONE FROZEN forward pass. Our rollout-level
sweep cannot resolve the effect because different windows produce different rollouts, and three
identical runs already differ by 18-33 % on late-clip sharpness. But with q/k/v dumped from a real
steady-state call, truncating the keys is deterministic. There is no seed, no divergence, no noise
band: the only thing that changes is how many keys the softmax sees.

  out_full  = softmax(q @ k_all^T  * s) @ v_all         # window 24, everything the model was given
  out_trunc = softmax(q @ k_kept^T * s) @ v_kept        # sink + the newest (N - sink) latents
  error(N)  = ||out_trunc - out_full|| / ||out_full||

Run the model at a window LONGER than the default (24) so the ablation can ask whether the default
18 was even being used, not just whether 12 is survivable.

  <venv>/bin/python bench/window/attn_ablate.py --dump_dir <dir> [--sink 6] [--frame_seqlen 1508]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--dump_dir", required=True, help="dir written by LINGBOT_DUMP_QKV_DIR")
ap.add_argument("--frame_seqlen", type=int, default=1508, help="tokens per latent at 832x464")
ap.add_argument("--sink", type=int, default=6, help="pinned sink latents at the front of the buffer")
ap.add_argument("--q_sample", type=int, default=512, help="query rows sampled per layer")
ap.add_argument("--device", default="cpu", help="cuda makes this seconds instead of many minutes")
ap.add_argument("--out", default="bench/window/attn_ablate.json")
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.dump_dir, "call_*.pt")))
if not files:
    raise SystemExit(f"no call_*.pt in {a.dump_dir}")


def attend(q, k, v, scale):
    """[H,n,D] x [H,L,D] -> [H,n,D] in fp32; chunked over queries to bound memory."""
    outs = []
    for i in range(0, q.shape[1], 128):
        qs = q[:, i:i + 128]
        p = torch.softmax(torch.bmm(qs, k.transpose(1, 2)) * scale, dim=-1)
        outs.append(torch.bmm(p, v))
    return torch.cat(outs, dim=1)


print(f"# {len(files)} layer dumps, sink={a.sink}")
per_layer = []
for path in files:
    d = torch.load(path, map_location="cpu")
    dev = a.device
    q, k, v = (d["q"].float().to(dev), d["k"].float().to(dev), d["v"].float().to(dev))  # [B,L,H,D] NHD
    B, Lq, H, D = q.shape
    Lk = k.shape[1]
    n_lat = Lk // a.frame_seqlen
    scale = 1.0 / np.sqrt(D)

    idx = np.linspace(0, Lq - 1, min(a.q_sample, Lq)).astype(int)
    qs = q[0, idx].permute(1, 0, 2).contiguous()      # [H, n, D]
    ks = k[0].permute(1, 0, 2).contiguous()           # [H, Lk, D]
    vs = v[0].permute(1, 0, 2).contiguous()

    ref = attend(qs, ks, vs, scale)
    ref_norm = ref.norm()

    fs, S = a.frame_seqlen, a.sink
    row = {"call": int(d.get("call", -1)), "n_latents": n_lat, "errors": {}}
    for N in range(S + 1, n_lat + 1):
        roll = N - S
        # sink stays pinned at the front; the rest is the newest `roll` latents, which is exactly
        # what a shorter local_attn_size would have retained at this point in the rollout
        keep = torch.cat([torch.arange(0, S * fs), torch.arange(Lk - roll * fs, Lk)]).to(dev)
        out = attend(qs, ks[:, keep], vs[:, keep], scale)
        row["errors"][str(N)] = float(((out - ref).norm() / ref_norm).item())
    per_layer.append(row)
    e = row["errors"]
    print(f"#   call {row['call']:3d}: latents={n_lat}  err@12={e.get('12', float('nan')):.4f}"
          f"  err@18={e.get('18', float('nan')):.4f}")

n_lat = per_layer[0]["n_latents"]
windows = [str(N) for N in range(a.sink + 1, n_lat + 1)]
M = np.array([[r["errors"][w] for w in windows] for r in per_layer if r["n_latents"] == n_lat])
mean_err, max_err = M.mean(axis=0), M.max(axis=0)

print(f"\nrelative change in attention output when the window is truncated to N latents")
print(f"(reference = the full {n_lat}-latent window the model was actually given)\n")
print(f"  {'N':>3}  {'mean err':>9}  {'worst layer':>11}")
for w, me, xe in zip(windows, mean_err, max_err):
    mark = "  <- current default" if w == "18" else ""
    print(f"  {w:>3}  {me:>9.4f}  {xe:>11.4f}{mark}")


def smallest_under(th):
    for w, me in zip(windows, mean_err):
        if me <= th:
            return int(w)
    return None


out = {"n_latents": n_lat, "sink": a.sink, "layers": int(M.shape[0]), "windows": windows,
       "mean_rel_err": [round(float(x), 6) for x in mean_err],
       "max_rel_err": [round(float(x), 6) for x in max_err],
       "window_under_1pct": smallest_under(0.01), "window_under_2pct": smallest_under(0.02),
       "window_under_5pct": smallest_under(0.05), "per_layer": per_layer}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print(f"\nsmallest window with <1% mean output change: {out['window_under_1pct']}")
print(f"                              <2%:            {out['window_under_2pct']}")
print(f"                              <5%:            {out['window_under_5pct']}")
print(f"wrote {a.out}")
