#!/usr/bin/env python3
"""C7: SpargeAttn (thu-ml) vs dense SageAttention 2.2 on our world-model attention shape.

Dense reference : sageattn()            -- what we run today
Candidate       : spas_sage2_attn_*     -- SpargeAttn = block-sparse SageAttention2
Quality ref     : torch SDPA in fp32 on the same inputs

Run on the 5090 pod. CPU-only smoke test: --dry (exits 0, touches no GPU).

SpargeAttn API (verified against
https://github.com/thu-ml/SpargeAttn/blob/main/spas_sage_attn/core.py):

  spas_sage2_attn_meansim_topk_cuda(q, k, v, attn_mask=None, dropout_p=0.0,
      is_causal=False, scale=None, smooth_k=True, simthreshd1=-0.1,
      cdfthreshd=None, topk=0.5, pvthreshd=50, attention_sink=False,
      tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False)

  spas_sage2_attn_meansim_cuda(q, k, v, ..., simthreshd1=0.6, cdfthreshd=0.98,
      pvthreshd=50, attention_sink=False, tensor_layout="HND", ...)

Both are training-free and need NO tuning pass. The per-model tuning pass
(spas_sage_attn.autotune.SparseAttentionMeansim, TUNE_MODE=1) only exists to
fit PER-HEAD simthreshd1/cdfthreshd/pvthreshd; the two functions above take
global thresholds and work plug-and-play.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- shapes ----
B, HQ, D = 1, 12, 128
LQ, LKV = 6032, 27144
SINK_TOKENS = 6 * 1  # the 6 "sink" latents at the start of the KV window
SM_SCALE = 1.0 / math.sqrt(D)
DENSE_FLOPS = 4.0 * B * HQ * LQ * LKV * D  # 2 GEMMs x 2 flop/MAC ~= 1.006 TFLOP

RESULTS = Path(__file__).resolve().parent / "results" / "c7_sparge.json"

# --------------------------------------------------------------- settings ---
# Conservative -> aggressive. topk variants use the recommended *_topk_cuda API;
# meansim variants use the threshold API (same kernel, different mask builder).
SETTINGS = [
    dict(name="topk_0.9", api="topk", topk=0.9),
    dict(name="topk_0.7", api="topk", topk=0.7),
    dict(name="topk_0.5", api="topk", topk=0.5),  # repo default / recommended
    dict(name="topk_0.3", api="topk", topk=0.3),
    dict(name="meansim_conservative", api="meansim", simthreshd1=0.7, cdfthreshd=0.995, pvthreshd=20),
    dict(name="meansim_default", api="meansim", simthreshd1=0.6, cdfthreshd=0.98, pvthreshd=50),
    dict(name="meansim_aggressive", api="meansim", simthreshd1=0.4, cdfthreshd=0.95, pvthreshd=100),
]


# ------------------------------------------------------------- synthetic -----
def make_inputs(device, dtype, seed=0):
    """SYNTHETIC inputs with a plausible structure.

    Pure randn gives near-uniform attention: every block matters equally, so a
    sparse kernel has nothing to skip and its speedup is understated (while its
    quality loss is overstated). We instead build:
      * NCLUST key clusters; each query block is steered at 2-3 of them,
      * `SINK_TOKENS` sink keys at position 0 that every query block attends to,
      * a mild positional recency bias over the rolling window.
    This is a guess at the real distribution, NOT a measurement of it.
    """
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    NCLUST = 24
    QBLK = 128
    nqb = (LQ + QBLK - 1) // QBLK

    # cluster centres in key space, per head
    centres = torch.randn(HQ, NCLUST, D, generator=g) * 0.9

    # `u`: a per-head direction shared by every query -- the sink axis
    u = torch.randn(HQ, D, generator=g)
    u = u / u.norm(dim=-1, keepdim=True)

    # keys: each key belongs to one cluster (contiguous runs -> block structure)
    k = torch.empty(B, LKV, HQ, D)
    run = max(1, LKV // (NCLUST * 8))
    cid = (torch.arange(LKV) // run) % NCLUST
    for h in range(HQ):
        k[0, :, h, :] = centres[h, cid] + 0.35 * torch.randn(LKV, D, generator=g)

    # queries: each q block aims at a few clusters (+ recency drift)
    q = torch.empty(B, LQ, HQ, D)
    for h in range(HQ):
        for b in range(nqb):
            lo, hi = b * QBLK, min((b + 1) * QBLK, LQ)
            picks = torch.randint(0, NCLUST, (3,), generator=g)
            tgt = centres[h, picks].mean(0)
            q[0, lo:hi, h, :] = tgt + 0.30 * torch.randn(hi - lo, D, generator=g)
    q *= 1.6  # sharpen logits; flat logits => no sparsity to exploit

    # sink tokens: keys parked on the shared axis `u`, which every query has a
    # component along -> they take a few percent of the mass from every query,
    # exactly the block a sparse filter must never drop.
    for h in range(HQ):
        q[0, :, h, :] += 3.0 * u[h]
        k[0, :SINK_TOKENS, h, :] = 18.0 * u[h] + 0.35 * torch.randn(SINK_TOKENS, D, generator=g)

    v = torch.randn(B, LKV, HQ, D, generator=g)
    to = lambda t: t.to(device=device, dtype=dtype).contiguous()
    return to(q), to(k), to(v)


# ----------------------------------------------------------------- timing ---
def cuda_time(fn, warm, iters):
    import torch

    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def quality(out, ref):
    import torch

    a = out.float().flatten()
    b = ref.float().flatten()
    return dict(
        cos_sim=float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
        max_abs_err=float((a - b).abs().max()),
        mean_abs_err=float((a - b).abs().mean()),
        rel_l1=float((a - b).abs().sum() / b.abs().sum()),
    )


# ------------------------------------------------------------------- main ---
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry", action="store_true", help="CPU-only plan check, exits 0")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attention-sink", action="store_true",
                   help="pass attention_sink=True (SpargeAttn keeps the first block always dense)")
    p.add_argument("--out", type=str, default=str(RESULTS))
    p.add_argument("--only", type=str, default=None, help="comma-separated setting names")
    args = p.parse_args()

    settings = SETTINGS
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        settings = [s for s in SETTINGS if s["name"] in want]

    rec = {
        "candidate": "C7 SpargeAttn (thu-ml, arXiv 2502.18137)",
        "shape": dict(B=B, Hq=HQ, Lq=LQ, Lkv=LKV, D=D, dtype="bf16",
                      tensor_layout="NHD", is_causal=False, sm_scale=SM_SCALE),
        "dense_flops_per_call": DENSE_FLOPS,
        "inputs": "SYNTHETIC (clustered keys + 6 sink tokens) - NOT real model activations",
        "warmup": args.warmup, "iters": args.iters, "seed": args.seed,
        "attention_sink": args.attention_sink,
        "host": platform.node(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "settings": [s["name"] for s in settings],
    }

    if args.dry:
        print("=== C7 SpargeAttn harness -- DRY RUN (CPU only, no GPU touched) ===")
        print(f"shape        q[{B},{LQ},{HQ},{D}]  k/v[{B},{LKV},{HQ},{D}]  bf16 NHD is_causal=False")
        print(f"dense flops  {DENSE_FLOPS/1e12:.3f} TFLOP/call  (sm_scale={SM_SCALE:.6f})")
        print(f"timing       {args.warmup} warm / {args.iters} timed, CUDA events")
        print("inputs       SYNTHETIC: 24 clustered key runs/head + 6 sink tokens, sharpened logits")
        print("             (randn would be near-uniform -> understates sparsity benefit)")
        print("reference    fp32 torch SDPA  AND  dense sageattn() (the on-top-of-quant delta)")
        print("baseline     sageattn(q,k,v, tensor_layout='NHD', is_causal=False, sm_scale=...)")
        print("candidates:")
        for s in settings:
            if s["api"] == "topk":
                print(f"  - {s['name']:<22} spas_sage2_attn_meansim_topk_cuda(topk={s['topk']}, pvthreshd=50)")
            else:
                print(f"  - {s['name']:<22} spas_sage2_attn_meansim_cuda("
                      f"simthreshd1={s['simthreshd1']}, cdfthreshd={s['cdfthreshd']}, pvthreshd={s['pvthreshd']})")
        print("tuning pass  NOT required: both APIs are training-free/plug-and-play.")
        print("             (autotune.SparseAttentionMeansim + TUNE_MODE=1 only fits PER-HEAD")
        print("              thresholds; not needed for a kernel-level number.)")
        print("sm_120       NOT supported by SpargeAttn main -- setup.py SUPPORTED_ARCHS =")
        print("             {8.0, 8.6, 8.7, 8.9, 9.0}; building on a 5090 raises RuntimeError.")
        print("             See c7_sparge.md for the 3-line patch; harness aborts cleanly if unbuilt.")
        print(f"results      {args.out}")
        print("DRY OK")
        return 0

    import torch

    if not torch.cuda.is_available():
        print("FAIL: no CUDA device. Run on the 5090 pod (or use --dry).", file=sys.stderr)
        return 2

    cap = torch.cuda.get_device_capability(0)
    rec["gpu"] = torch.cuda.get_device_name(0)
    rec["sm"] = f"sm_{cap[0]}{cap[1]}"
    rec["torch"] = torch.__version__
    print(f"GPU {rec['gpu']}  {rec['sm']}  torch {rec['torch']}")

    try:
        from sageattention import sageattn
    except Exception as exc:  # pragma: no cover
        print(f"FAIL: sageattention not importable: {exc}", file=sys.stderr)
        return 2

    sparge_err = None
    try:
        import spas_sage_attn
        from spas_sage_attn import (spas_sage2_attn_meansim_cuda,
                                    spas_sage2_attn_meansim_topk_cuda)
        rec["spas_sage_attn"] = getattr(spas_sage_attn, "__version__", "unknown")
    except Exception as exc:
        sparge_err = f"{type(exc).__name__}: {exc}"
        spas_sage2_attn_meansim_cuda = spas_sage2_attn_meansim_topk_cuda = None
        print(f"WARN: spas_sage_attn not importable ({sparge_err}).")
        print("      On sm_120 this is expected unless SpargeAttn was patched+rebuilt "
              "(see c7_sparge.md). Reporting the dense baseline only.")
    rec["sparge_import_error"] = sparge_err

    dev = torch.device("cuda")
    q, k, v = make_inputs(dev, torch.bfloat16, seed=args.seed)
    print(f"inputs: SYNTHETIC clustered q/k/v, {q.shape} / {k.shape}, bf16")

    # ---- fp32 SDPA quality reference (chunked over queries to bound memory) --
    with torch.no_grad():
        qh = q.permute(0, 2, 1, 3).float()
        kh = k.permute(0, 2, 1, 3).float()
        vh = v.permute(0, 2, 1, 3).float()
        chunks = []
        step = 1024
        for i in range(0, LQ, step):
            chunks.append(torch.nn.functional.scaled_dot_product_attention(
                qh[:, :, i:i + step], kh, vh, is_causal=False, scale=SM_SCALE))
        ref32 = torch.cat(chunks, dim=2).permute(0, 2, 1, 3).contiguous()  # NHD
        del qh, kh, vh, chunks
        torch.cuda.empty_cache()

    # ---- dense sageattn baseline -------------------------------------------
    dense_fn = lambda: sageattn(q, k, v, tensor_layout="NHD", is_causal=False, sm_scale=SM_SCALE)
    with torch.no_grad():
        dense_out = dense_fn()
        ms = cuda_time(dense_fn, args.warmup, args.iters)
    rec["dense_sageattn"] = dict(
        ms=ms, tops=DENSE_FLOPS / (ms * 1e-3) / 1e12,
        vs_fp32_sdpa=quality(dense_out, ref32),
    )
    print(f"\ndense sageattn      {ms:8.3f} ms   {rec['dense_sageattn']['tops']:7.1f} TOPS(dense-flops)"
          f"   cos vs fp32 {rec['dense_sageattn']['vs_fp32_sdpa']['cos_sim']:.6f}")

    # ---- SpargeAttn grid ----------------------------------------------------
    rec["sparge"] = []
    for s in settings:
        row = dict(s)
        if spas_sage2_attn_meansim_cuda is None:
            row["status"] = "skipped: spas_sage_attn unavailable"
            rec["sparge"].append(row)
            continue
        common = dict(is_causal=False, scale=SM_SCALE, tensor_layout="NHD",
                      attention_sink=args.attention_sink)
        if s["api"] == "topk":
            call = lambda rs=False: spas_sage2_attn_meansim_topk_cuda(
                q, k, v, topk=s["topk"], pvthreshd=50, return_sparsity=rs, **common)
        else:
            call = lambda rs=False: spas_sage2_attn_meansim_cuda(
                q, k, v, simthreshd1=s["simthreshd1"], cdfthreshd=s["cdfthreshd"],
                pvthreshd=s["pvthreshd"], return_sparsity=rs, **common)
        try:
            with torch.no_grad():
                out, qk_sparsity = call(rs=True)
                ms_s = cuda_time(lambda: call(rs=False), args.warmup, args.iters)
        except Exception as exc:
            row["status"] = f"error: {type(exc).__name__}: {exc}"
            print(f"  {s['name']:<22} ERROR {type(exc).__name__}: {exc}")
            rec["sparge"].append(row)
            continue
        row.update(
            status="ok", ms=ms_s,
            tops_dense_flops=DENSE_FLOPS / (ms_s * 1e-3) / 1e12,
            qk_block_sparsity=qk_sparsity,      # fraction of QK blocks skipped (PV filter not counted)
            speedup_vs_dense_sageattn=ms / ms_s,
            vs_fp32_sdpa=quality(out, ref32),
            vs_dense_sageattn=quality(out, dense_out),  # <- cost of sparsity alone
        )
        print(f"  {s['name']:<22} {ms_s:8.3f} ms  {row['tops_dense_flops']:7.1f} TOPS"
              f"  x{row['speedup_vs_dense_sageattn']:5.2f}  qk_sparsity {qk_sparsity*100:5.1f}%"
              f"  cos/fp32 {row['vs_fp32_sdpa']['cos_sim']:.6f}"
              f"  cos/dense {row['vs_dense_sageattn']['cos_sim']:.6f}"
              f"  maxabs/dense {row['vs_dense_sageattn']['max_abs_err']:.4f}")
        rec["sparge"].append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=2))
    print(f"\nwrote {out_path}")
    print("NOTE: inputs are SYNTHETIC. Any go/no-go decision needs the exp15 lossless")
    print("      band measured on generated video, not these tensor metrics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
