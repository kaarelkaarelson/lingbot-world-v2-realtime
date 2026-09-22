#!/usr/bin/env python3
"""SpargeAttn (thu-ml, arXiv 2502.18137) vs dense SageAttention on our rolling-KV shapes.

  /workspace/sparge/venv_sparge/bin/python bench/attn/sparge/bench.py
  python bench/attn/sparge/bench.py --dry      # CPU, no GPU, exits 0

Shapes are ours: q [1, 6032, 12, 128], k/v [1, 27144, 12, 128], bf16, NHD, sm_scale = 1/sqrt(128).
Effective TOPS is always reported against the DENSE flop count so a sparse kernel that
skips work shows up as super-dense throughput rather than being rewarded twice.
"""
import argparse
import math
import sys

Q_LEN, KV_LEN, HEADS, DIM = 6032, 27144, 12, 128
BATCH = 1
SM_SCALE = 1.0 / math.sqrt(DIM)
DENSE_FLOPS = 4 * Q_LEN * KV_LEN * HEADS * DIM  # QK^T + PV, 2 flop/MAC each
WARMUP, ITERS = 20, 50


def banner(msg):
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


def config_summary():
    return (
        f"q [{BATCH},{Q_LEN},{HEADS},{DIM}]  k/v [{BATCH},{KV_LEN},{HEADS},{DIM}]  bf16 NHD\n"
        f"sm_scale = 1/sqrt({DIM}) = {SM_SCALE:.8f}   is_causal = False\n"
        f"dense flops = 4*{Q_LEN}*{KV_LEN}*{HEADS}*{DIM} = {DENSE_FLOPS:,} "
        f"({DENSE_FLOPS/1e12:.4f} Tflop / call)\n"
        f"timing: {WARMUP} warmup + {ITERS} timed, CUDA events"
    )


TUNING_NOTICE = """\
TUNING REQUIREMENT -- read before trusting any number below.

SpargeAttn has two kinds of knob (spas_sage_attn/core.py):

  1. `topk` (spas_sage2_attn_meansim_topk_cuda, the API upstream recommends)
     A DIRECT per-call density budget: topk=0.5 keeps ~50% of the 128x64 blocks.
     No tuning pass is needed -- you set the budget, the kernel honours it. The
     speed number for a given topk is meaningful on its own; the ERROR number is
     what tells you whether that budget is affordable.

  2. `simthreshd1` / `cdfthreshd` / `pvthreshd`
     (spas_sage2_attn_meansim_cuda and the topk variant's fallback path)
     These are PER-HEAD parameters. Upstream's offline tuner
     (spas_sage_attn/autotune.py: SparseAttentionMeansim.tune_cdfthreshd /
     .tune_pvthreshd) binary-searches them per attention head against a REAL model's
     activations under an L1-error target, then saves them to a .pt checkpoint
     (README "Tuning", model zoo Xiang-cd/sparge-attention-model-zoo).
     Running spas_sage2_attn_meansim_cuda with the module DEFAULTS
     (simthreshd1=0.6, cdfthreshd=0.98, pvthreshd=50) is NOT a tuned configuration.
     Its speed/quality point says nothing about what a tuned deployment would give.

This script therefore reports the topk sweep as the primary result and labels the
untuned meansim point explicitly. It does not run the offline tuner, because the
tuner needs a real model forward pass -- there is nothing to tune against here.

SECOND, LARGER CAVEAT when running on synthetic inputs (the default):
random Gaussian q/k/v have no low-rank / locally-redundant structure, so a
selectivity method has nothing to find. Measured sparsity on random inputs is a
LOWER BOUND on what real activations would allow, and measured error at a forced
topk budget is an UPPER BOUND on real error. Use --qkv <file.pt> with a captured
{'q','k','v'} dump from the real model for a number you can act on.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="CPU only, print config, exit 0")
    ap.add_argument("--qkv", default=None,
                    help="path to a torch.save({'q','k','v'}) dump of REAL activations "
                         "(NHD). Strongly preferred over the synthetic default.")
    ap.add_argument("--topk", default="0.9,0.75,0.5,0.25",
                    help="comma-separated topk density budgets to sweep")
    ap.add_argument("--is-causal", action="store_true",
                    help="request causal masking (see the landmine warning)")
    args = ap.parse_args()

    banner("SpargeAttn vs dense SageAttention -- configuration")
    print(config_summary())
    print()
    print(TUNING_NOTICE)

    if args.is_causal:
        banner("LANDMINE WARNING: is_causal=True")
        print(
            "Upstream thu-ml/SpargeAttn @ ae5b629 HARDCODES `False` for the is_causal\n"
            "kernel argument in EVERY arch branch of spas_sage2_attn_meansim_cuda\n"
            "(core.py:74, 88, 90, 92) and spas_sage2_attn_meansim_topk_cuda\n"
            "(core.py:140, 154, 156, 158) -- the sm80 path included. The Python-level\n"
            "is_causal only reaches the block-map builder, never the kernel, so upstream\n"
            "returns a NON-CAUSAL result while claiming causal.\n"
            "bench/attn/sparge/sm120.patch fixes this (passes int(is_causal) through).\n"
            "If you are running an unpatched build, is_causal=True is SILENTLY WRONG."
        )

    if args.dry:
        print("\n--dry: no GPU work performed. OK")
        return 0

    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available; use --dry on a CPU box", file=sys.stderr)
        return 2

    from spas_sage_attn import (spas_sage2_attn_meansim_topk_cuda,
                                spas_sage2_attn_meansim_cuda)
    from sageattention import sageattn

    dev = torch.device("cuda")
    major, minor = torch.cuda.get_device_capability(0)
    banner(f"device: {torch.cuda.get_device_name(0)}  sm_{major}{minor}  "
           f"torch {torch.__version__} / cuda {torch.version.cuda}")

    if args.qkv:
        d = torch.load(args.qkv, map_location=dev)
        q, k, v = (d["q"].to(dev, torch.bfloat16).contiguous(),
                   d["k"].to(dev, torch.bfloat16).contiguous(),
                   d["v"].to(dev, torch.bfloat16).contiguous())
        print(f"loaded REAL activations from {args.qkv}: q{tuple(q.shape)} k{tuple(k.shape)}")
    else:
        g = torch.Generator(device=dev).manual_seed(0)
        mk = lambda L: torch.randn(BATCH, L, HEADS, DIM, generator=g,
                                   device=dev, dtype=torch.bfloat16)
        q, k, v = mk(Q_LEN), mk(KV_LEN), mk(KV_LEN)
        print("SYNTHETIC random inputs -- sparsity here is a LOWER BOUND (see notice above)")

    # ---- references -------------------------------------------------------
    # fp32 torch SDPA, BHLD layout
    # Done one head at a time: a full fp32 [1,12,6032,27144] score matrix is ~7.9 GB
    # and the math backend would materialise it.
    ref32 = torch.empty(q.shape, device=dev, dtype=torch.float32)
    with torch.no_grad():
        for h in range(q.shape[2]):
            qh = q[:, :, h:h + 1].transpose(1, 2).float()
            kh = k[:, :, h:h + 1].transpose(1, 2).float()
            vh = v[:, :, h:h + 1].transpose(1, 2).float()
            oh = torch.nn.functional.scaled_dot_product_attention(
                qh, kh, vh, is_causal=args.is_causal, scale=SM_SCALE)
            ref32[:, :, h:h + 1] = oh.transpose(1, 2)
            del qh, kh, vh, oh
    torch.cuda.empty_cache()

    def err(o, ref):
        a, b = o.float().flatten(), ref.float().flatten()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        return cos, (a - b).abs().max().item()

    def timeit(fn):
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(ITERS):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / ITERS

    rows = []

    def record(name, fn, sparsity=None):
        out = fn()
        if isinstance(out, tuple):
            out, sparsity = out
        ms = timeit(lambda: fn())
        tops = DENSE_FLOPS / (ms * 1e-3) / 1e12
        c32, m32 = err(out, ref32)
        cds, mds = err(out, dense_out if dense_out is not None else out)
        rows.append((name, ms, tops, sparsity, c32, m32, cds, mds))
        print(f"  {name:<46} {ms:8.3f} ms  {tops:7.1f} TOPS(dense-flops)", flush=True)
        return out

    # ---- dense SageAttention baseline ------------------------------------
    dense_out = None
    banner("dense SageAttention baseline")
    dense_out = record("sageattn (dense)",
                       lambda: sageattn(q, k, v, tensor_layout="NHD",
                                        is_causal=args.is_causal, sm_scale=SM_SCALE))

    # ---- SpargeAttn topk sweep -------------------------------------------
    banner("SpargeAttn spas_sage2_attn_meansim_topk_cuda (topk = explicit density budget)")
    for tk in [float(x) for x in args.topk.split(",")]:
        record(f"sparge topk={tk:g}",
               lambda tk=tk: spas_sage2_attn_meansim_topk_cuda(
                   q, k, v, topk=tk, is_causal=args.is_causal, scale=SM_SCALE,
                   tensor_layout="NHD", output_dtype=torch.bfloat16,
                   return_sparsity=True))

    # ---- SpargeAttn untuned meansim --------------------------------------
    banner("SpargeAttn spas_sage2_attn_meansim_cuda -- UNTUNED DEFAULTS, not a deployable point")
    record("sparge meansim (simthreshd1=0.6, cdfthreshd=0.98) UNTUNED",
           lambda: spas_sage2_attn_meansim_cuda(
               q, k, v, is_causal=args.is_causal, scale=SM_SCALE,
               tensor_layout="NHD", output_dtype=torch.bfloat16,
               return_sparsity=True))

    # ---- table ------------------------------------------------------------
    banner("results")
    hdr = (f"{'variant':<46} {'ms':>8} {'TOPS*':>8} {'qk_sparsity':>12} "
           f"{'cos/fp32':>10} {'maxabs/fp32':>12} {'cos/dense':>10} {'maxabs/dense':>13} {'speedup':>8}")
    print(hdr)
    print("-" * len(hdr))
    base_ms = rows[0][1]
    for name, ms, tops, sp, c32, m32, cds, mds in rows:
        sps = "n/a" if sp is None else f"{sp*100:.1f}%"
        print(f"{name:<46} {ms:8.3f} {tops:8.1f} {sps:>12} "
              f"{c32:10.5f} {m32:12.4g} {cds:10.5f} {mds:13.4g} {base_ms/ms:7.2f}x")
    print("\n* TOPS is against the DENSE flop count; a sparse variant exceeding the dense")
    print("  kernel's TOPS is skipping work, not computing faster per flop.")
    print("  qk_sparsity = fraction of 128x64 QK blocks skipped (core.py return_sparsity).")
    if not args.qkv:
        print("\nREMINDER: synthetic inputs. Rerun with --qkv <real dump> before drawing")
        print("          any conclusion about whether SpargeAttn helps our workload.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
