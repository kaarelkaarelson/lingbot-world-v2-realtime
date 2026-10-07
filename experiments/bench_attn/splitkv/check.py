#!/usr/bin/env python3
"""Split-KV SageAttention check: accuracy of S>1 vs S=1 vs a bf16 SDPA reference, and timing, on the LingBot shapes.

Needs a GPU and a wheel built by build.sh (nothing here was run when it was written):
  python check.py                    # card-0 + card-1 shapes, S in {1,2}
  python check.py --splits 1 2 3 4   # more splits
  python check.py --edge             # odd lengths (kv not a multiple of 64, q not a multiple of 128, S > tiles)

The kernel reads SAGE_SPLITKV on every call, so one process can flip between S values.
"""
import argparse
import os

import torch
import torch.nn.functional as F
from sageattention import sageattn

SHAPES = {  # name: (heads, q_len, kv_len)
    "card0_10h": (10, 6032, 27144),
    "card1_2h": (2, 6032, 27144),
}
EDGE_SHAPES = {
    "edge_kv65": (4, 300, 65),          # 2 tiles, last tile has 1 valid key
    "edge_kv64": (4, 300, 64),          # 1 tile -> S>1 falls back to the original path
    "edge_kv1000": (4, 1000, 1000),     # ragged q and kv
    "edge_kv4097": (2, 513, 4097),
}


def run(q, k, v, splits):
    if splits is None:
        os.environ.pop("SAGE_SPLITKV", None)
    else:
        os.environ["SAGE_SPLITKV"] = str(splits)
    return sageattn(q, k, v, tensor_layout="HND")


def timeit(fn, warm=20, iters=50):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


def errs(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    return dict(
        max_abs=d.max().item(),
        max_rel=(d.max() / b.abs().max()).item(),
        rel_l2=((a - b).norm() / b.norm()).item(),
        cos=F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
    )


def fmt(e):
    return f"max_abs {e['max_abs']:.3e}  max_rel {e['max_rel']:.3e}  rel_l2 {e['rel_l2']:.3e}  cos {e['cos']:.7f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--edge", action="store_true", help="odd-length shapes instead of the model shapes")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert 1 in args.splits, "S=1 is the baseline"

    shapes = EDGE_SHAPES if args.edge else SHAPES
    dev = torch.device("cuda")
    ok = True
    for name, (h, nq, nk) in shapes.items():
        g = torch.Generator(device=dev).manual_seed(args.seed)
        q = torch.randn(1, h, nq, 128, device=dev, dtype=torch.bfloat16, generator=g)
        k = torch.randn(1, h, nk, 128, device=dev, dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, h, nk, 128, device=dev, dtype=torch.bfloat16, generator=g)
        ref = F.scaled_dot_product_attention(q, k, v)  # bf16 reference
        print(f"\n== {name}: q [1,{h},{nq},128]  k/v [1,{h},{nk},128]  bf16 HND")

        o_default = run(q, k, v, None)
        o = {s: run(q, k, v, s) for s in args.splits}
        same = torch.equal(o_default, o[1])
        print(f"S=1 (SAGE_SPLITKV=1) bit-identical to unset env: {same}")
        ok &= same
        base_ref = errs(o[1], ref)
        print(f"S=1 vs bf16 SDPA : {fmt(base_ref)}")
        for s in args.splits:
            if s == 1:
                continue
            vs1 = errs(o[s], o[1])
            vref = errs(o[s], ref)
            print(f"S={s} vs S=1       : {fmt(vs1)}")
            print(f"S={s} vs bf16 SDPA : {fmt(vref)}")
            # S>1 should add no more error vs the reference than the kernel already has
            good = vref["rel_l2"] <= 1.1 * base_ref["rel_l2"] and vs1["rel_l2"] <= base_ref["rel_l2"]
            print(f"S={s} accuracy check (rel_l2 vs ref <= 1.1x S=1 and S-vs-S=1 <= S=1 vs ref): {'PASS' if good else 'FAIL'}")
            ok &= good

        if not args.no_time:
            t = {s: timeit(lambda s=s: run(q, k, v, s)) for s in args.splits}
            for s in args.splits:
                print(f"S={s}: {t[s]:.3f} ms/call" + ("" if s == 1 else f"   speedup vs S=1 {t[1] / t[s]:.3f}x"))
    print("\nOVERALL", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
