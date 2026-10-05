#!/usr/bin/env python3
"""Is the KV window's content redundant, block by block? A causal, measured certificate.

Sibling of `attn_ablate.py`. §21b of OPTIMIZATIONS.md ablated the KV window by TRUNCATING it
(dropping the oldest latents) and found the model uses the whole thing: 13 % mean / 30 % worst-
layer output change at window 12, smooth decay, no plateau. That kills truncation. But block
sparsity is a different kind of change: it keeps the WHOLE window and skips only the key BLOCKS
whose contribution to the output is negligible right now, decided per query rather than by age.
Truncation removes information; sparsity (if the model's attention is genuinely peaky) removes
nothing the query was using. This script asks, with the same frozen-forward-pass method as
attn_ablate.py, whether that premise holds.

  out_full    = softmax(q_blk @ k_all^T  * s) @ v_all           # every key block kept
  out_sparse  = softmax(q_blk @ k_kept^T * s) @ v_kept          # lowest-mass X% of key BLOCKS dropped,
                                                                 # renormalized over what's kept
  error(f)    = ||out_sparse - out_full|| / ||out_full||

Block importance is read off the FULL softmax this call actually computed (mass a key block holds
under the query it was scored against), not a cheap proxy — same "measure the real thing" discipline
as attn_ablate.py's causal ablation vs. attention-weight heuristics (Jain & Wallace, arXiv 1902.10186).
Blocks are then dropped lowest-mass-first and the output is recomputed with softmax renormalized over
the surviving keys, which is what a real block-sparse kernel (SpargeAttn, SageAttention's block-sparse
path) actually does — this is not a leave-one-block-out ablation, it is the real skip-and-renormalize
operation, evaluated at a sweep of sparsity fractions.

Key blocks use `--block_size` (64 or 128 are the interesting ones: SageAttention's CTA_K tiling,
`OPTIMIZATIONS.md` section 19). Query rows are grouped into `--q_block`-sized blocks (128, matching
CTA_Q) at `--n_qblocks` positions spread across the query range, so the script can also answer
whether the surviving block SET is stable across query blocks and layers (a static mask would be
cheap; a scattered, query-dependent set is what SpargeAttn computes per step, at a cost).

  <venv>/bin/python bench/window/block_sparsity.py --dump_dir <dir> --block_size 64 [--device cuda]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--dump_dir", required=True, help="dir written by LINGBOT_DUMP_QKV_DIR (call_*.pt)")
ap.add_argument("--frame_seqlen", type=int, default=1508, help="tokens per latent at 832x464")
ap.add_argument("--sink", type=int, default=6, help="pinned sink latents at the front of the buffer")
ap.add_argument("--block_size", type=int, default=64, help="key block size; 64 or 128 = SageAttention CTA_K")
ap.add_argument("--q_block", type=int, default=128, help="query rows per block; 128 = SageAttention CTA_Q")
ap.add_argument("--n_qblocks", type=int, default=6, help="query blocks sampled per layer, spread over Lq")
ap.add_argument("--sparsity_fracs", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                 help="comma list of key-BLOCK drop fractions to evaluate")
ap.add_argument("--error_budgets", default="0.01,0.02,0.05", help="comma list of error budgets to report")
ap.add_argument("--stability_frac", type=float, default=0.5, help="drop fraction used for the stability check")
ap.add_argument("--attn_s_per_chunk", type=float, default=0.288, help="measured attention s/chunk, OPTIMIZATIONS.md sec 18-21")
ap.add_argument("--chunk_s", type=float, default=0.980, help="measured total s/chunk, baseline window 18")
ap.add_argument("--frames_per_chunk", type=int, default=16)
ap.add_argument("--device", default="cpu", help="cuda makes this seconds instead of many minutes")
ap.add_argument("--out", default="bench/window/block_sparsity.json")
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.dump_dir, "call_*.pt")))
if not files:
    raise SystemExit(f"no call_*.pt in {a.dump_dir}")

fracs = [float(x) for x in a.sparsity_fracs.split(",")]
budgets = [float(x) for x in a.error_budgets.split(",")]


def block_bounds(Lk: int, bs: int) -> list[tuple[int, int]]:
    return [(i, min(i + bs, Lk)) for i in range(0, Lk, bs)]


def sparse_error(scores, v, blocks, order, frac, ref, ref_norm):
    """Drop the lowest-mass `frac` of blocks (by `order`, ascending mass), renormalize, recompute."""
    n_blocks = len(blocks)
    n_drop = int(round(frac * n_blocks))
    if n_drop == 0:
        return 0.0, n_blocks
    kept_blocks = sorted(order[n_drop:].tolist())
    if not kept_blocks:
        return float("nan"), 0
    kept_idx = torch.cat([torch.arange(blocks[b][0], blocks[b][1]) for b in kept_blocks]).to(scores.device)
    probs_kept = torch.softmax(scores.index_select(-1, kept_idx), dim=-1)
    out = torch.matmul(probs_kept, v.index_select(1, kept_idx))
    err = ((out - ref).norm() / ref_norm).item()
    return err, len(kept_blocks)


print(f"# {len(files)} layer dumps, block_size={a.block_size}, q_block={a.q_block}, "
      f"n_qblocks={a.n_qblocks}, sink={a.sink}")

per_layer = []          # one row per (layer, qblock): {"call", "qblock_start", "errors": {frac: err}}
kept_sets_by_layer = {}  # call -> list of kept-set-at-stability_frac, one per qblock (for cross-qblock stability)
sink_survival_by_layer = {}  # call -> list of (frac, fraction of qblocks where sink block(s) fully kept)

for path in files:
    d = torch.load(path, map_location="cpu")
    dev = a.device
    q, k, v = (d["q"].float().to(dev), d["k"].float().to(dev), d["v"].float().to(dev))  # [B,L,H,D] NHD
    B, Lq, H, D = q.shape
    Lk = k.shape[1]
    n_lat = Lk // a.frame_seqlen
    scale = 1.0 / np.sqrt(D)
    call = int(d.get("call", -1))

    ks = k[0].permute(1, 0, 2).contiguous()   # [H, Lk, D]
    vs = v[0].permute(1, 0, 2).contiguous()
    blocks = block_bounds(Lk, a.block_size)
    n_blocks = len(blocks)
    sink_end_tok = a.sink * a.frame_seqlen
    sink_blocks = {i for i, (s, e) in enumerate(blocks) if s < sink_end_tok}

    starts = np.linspace(0, max(Lq - a.q_block, 0), a.n_qblocks).astype(int)
    kept_sets, sink_rows = [], []
    for qi, qs0 in enumerate(starts):
        qe0 = min(qs0 + a.q_block, Lq)
        qs = q[0, qs0:qe0].permute(1, 0, 2).contiguous()   # [H, qblk, D]

        scores = torch.matmul(qs, ks.transpose(1, 2)) * scale     # [H, qblk, Lk]
        probs = torch.softmax(scores, dim=-1)
        out_full = torch.matmul(probs, vs)
        ref_norm = out_full.norm()

        mass = torch.stack([probs[:, :, s:e].sum(-1) for s, e in blocks], dim=-1)  # [H, qblk, n_blocks]
        block_mass = mass.mean(dim=(0, 1))                                        # [n_blocks]
        order = torch.argsort(block_mass)                                         # ascending: drop these first

        row = {"call": call, "n_latents": n_lat, "qblock": int(qi), "q_start": int(qs0), "errors": {}}
        for f in fracs:
            err, n_kept = sparse_error(scores, vs, blocks, order, f, out_full, ref_norm)
            row["errors"][str(f)] = float(err)
        per_layer.append(row)

        n_drop_stab = int(round(a.stability_frac * n_blocks))
        kept_sets.append(frozenset(order[n_drop_stab:].tolist()))
        for f in fracs:
            n_drop = int(round(f * n_blocks))
            dropped = set(order[:n_drop].tolist())
            sink_rows.append((f, len(sink_blocks & dropped) == 0))

    kept_sets_by_layer[call] = kept_sets
    sink_survival_by_layer[call] = sink_rows
    last_sink_blocks = sink_blocks

    e = per_layer[-1]["errors"] if per_layer else {}
    print(f"#   call {call:3d}: latents={n_lat} n_blocks={n_blocks}  "
          f"err@50%drop(last qblk)={e.get(str(a.stability_frac), float('nan')):.4f}")

# --- aggregate curve: mean/max over qblocks within a layer, then mean/max over layers ---
calls = sorted({r["call"] for r in per_layer})
per_call_mean = {c: [] for c in calls}
for c in calls:
    rows = [r for r in per_layer if r["call"] == c]
    for f in fracs:
        vals = [r["errors"][str(f)] for r in rows]
        per_call_mean[c].append(float(np.mean(vals)))
M = np.array([per_call_mean[c] for c in calls])   # [n_layers, n_fracs]
mean_err, max_err = M.mean(axis=0), M.max(axis=0)

print(f"\nrelative output change when the lowest-mass X% of key BLOCKS (size {a.block_size}) are "
      f"dropped and softmax is renormalized over what remains")
print(f"(reference = the same call's full dense output; {len(calls)} layers, {a.n_qblocks} query "
      f"blocks each, q_block={a.q_block})\n")
print(f"  {'drop%':>6}  {'mean err':>9}  {'worst layer':>11}")
for f, me, xe in zip(fracs, mean_err, max_err):
    print(f"  {f*100:5.0f}%  {me:>9.4f}  {xe:>11.4f}")


def largest_sparsity_under(th):
    best = None
    for f, me in zip(fracs, mean_err):
        if me <= th:
            best = f
    return best


print()
budget_rows = []
for b in budgets:
    frac = largest_sparsity_under(b)
    budget_rows.append({"budget": b, "sparsity_frac": frac})
    print(f"largest sparsity fraction with mean output change <= {b*100:.0f}%: "
          f"{'none (even 10% exceeds it)' if frac is None else f'{frac*100:.0f}%'}")

# --- FPS arithmetic: attention linear in retained keys (OPTIMIZATIONS.md sec 19, h3) ---
print(f"\nFPS arithmetic (theoretical, attn={a.attn_s_per_chunk:.3f}s of {a.chunk_s:.3f}s/chunk, "
      f"{a.frames_per_chunk} frames/chunk, cost linear in retained keys):")
print(f"  {'drop%':>6}  {'theor. attn s':>13}  {'theor. chunk s':>14}  {'theor. FPS':>10}  {'vs baseline':>11}")
base_fps = a.frames_per_chunk / a.chunk_s
fps_rows = []
for f in [0.0] + fracs:
    attn_s = a.attn_s_per_chunk * (1 - f)
    chunk_s = a.chunk_s - a.attn_s_per_chunk * f
    fps = a.frames_per_chunk / chunk_s
    fps_rows.append({"drop_frac": f, "theoretical_attn_s": attn_s, "theoretical_chunk_s": chunk_s,
                      "theoretical_fps": fps})
    print(f"  {f*100:5.0f}%  {attn_s:13.3f}  {chunk_s:14.3f}  {fps:10.2f}  {(fps/base_fps-1)*100:+10.1f}%")
print("  NB theoretical only: a real block-sparse kernel still pays per-block mask/index overhead "
      "(the QK-stage similarity pass, gather/scatter, and reduced tile occupancy at small block "
      "counts), so realised FPS sits BELOW this line; SpargeAttn's own reported speedups are well "
      "below the dense-FLOP-implied number for exactly this reason.")

# --- structural stability: is the kept-block SET stable across query blocks / layers? ---
def jaccard(a_set, b_set):
    if not a_set and not b_set:
        return 1.0
    return len(a_set & b_set) / len(a_set | b_set)


within_layer_jaccard = []
for c in calls:
    sets = kept_sets_by_layer[c]
    pairs = [jaccard(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    if pairs:
        within_layer_jaccard.append(float(np.mean(pairs)))

first_qblock_sets = [kept_sets_by_layer[c][0] for c in calls if kept_sets_by_layer[c]]
across_layer_pairs = [jaccard(first_qblock_sets[i], first_qblock_sets[j])
                       for i in range(len(first_qblock_sets)) for j in range(i + 1, len(first_qblock_sets))]

mean_within = float(np.mean(within_layer_jaccard)) if within_layer_jaccard else float("nan")
mean_across = float(np.mean(across_layer_pairs)) if across_layer_pairs else float("nan")

print(f"\nstructural stability of the kept-block set at drop={a.stability_frac*100:.0f}% "
      f"(Jaccard overlap; 1.0 = identical set, matches a static mask; 0.0 = disjoint)")
print(f"  within a layer, across the {a.n_qblocks} query blocks: mean Jaccard = {mean_within:.3f}")
print(f"  across layers (same query block, layer to layer):     mean Jaccard = {mean_across:.3f}")
if mean_within == mean_within and mean_within > 0.8:
    print("  -> HIGH within-layer overlap: different queries in this call keep nearly the same key "
          "blocks. A STATIC per-layer mask could plausibly capture most of this sparsity.")
elif mean_within == mean_within:
    print("  -> LOW/MODERATE within-layer overlap: different queries keep different key blocks. "
          "A static mask would miss content-dependent sparsity; the mask must be computed per step, "
          "as SpargeAttn does.")

# --- sink survival: does the certified-safe sparsity level ever drop the sink? ---
print(f"\nsink block survival (sink={a.sink} latents, blocks {sorted(last_sink_blocks)} "
      f"of the last layer read):")
for f in fracs:
    kept = [ok for rows in sink_survival_by_layer.values() for (ff, ok) in rows if ff == f]
    if kept:
        print(f"  drop={f*100:3.0f}%: sink fully retained in {sum(kept)}/{len(kept)} (layer, qblock) cases")

out = {
    "n_layers": len(calls), "block_size": a.block_size, "q_block": a.q_block, "n_qblocks": a.n_qblocks,
    "sink_latents": a.sink, "sparsity_fracs": fracs,
    "mean_rel_err": [round(float(x), 6) for x in mean_err],
    "max_rel_err": [round(float(x), 6) for x in max_err],
    "budgets": budget_rows, "fps_arithmetic": fps_rows,
    "stability_frac": a.stability_frac,
    "within_layer_jaccard_mean": mean_within, "across_layer_jaccard_mean": mean_across,
    "per_layer_qblock": per_layer,
}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print(f"\nwrote {a.out}")
