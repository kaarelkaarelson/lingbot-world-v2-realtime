"""Bit-exact regression test for the fused DiT: its outputs must not change across refactors.

Runs the tiny-config 5-chunk causal loop from test_dit_fusion_cpu.py (seeded weights, CPU SDPA in
place of FlashAttention/SageAttention) and compares every denoising forward with torch.equal against
tests/golden/dit_fused.pt, for the default 1-row time MLP and for LINGBOT_DIT_FUSION_EXACT_T=1.

Regenerate only when a numerics change is intended:  python tests/test_golden_cpu.py --regen
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_dit_fusion_cpu as T  # noqa: E402

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "dit_fused.pt")


def outputs():
    T._stub_packages()
    up = T.load_model_module(T.UPSTREAM_MODEL, "wan.modules.model_fast")
    state = T.build(up).state_dict()
    out = {"default": T.run_loop(T.build(T.load_model_module(T.PATCHED_MODEL, "wan.modules.model_fast_fusion"), state), fusion=True)}
    os.environ["LINGBOT_DIT_FUSION_EXACT_T"] = "1"
    try:
        m = T.build(T.load_model_module(T.PATCHED_MODEL, "wan.modules.model_fast_fusion_exact"), state)
        out["exact_t"] = T.run_loop(m, fusion=True)
    finally:
        del os.environ["LINGBOT_DIT_FUSION_EXACT_T"]
    return out


def test_golden_cpu():
    ref = torch.load(GOLDEN)
    got = outputs()
    for mode in ref:
        assert len(ref[mode]) == len(got[mode]) == 20
        for i, (a, b) in enumerate(zip(ref[mode], got[mode])):
            assert torch.equal(a, b), f"{mode} forward {i}: max|diff|={(a - b).abs().max().item():.3e}"


if __name__ == "__main__":
    if "--regen" in sys.argv:
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        torch.save(outputs(), GOLDEN)
        print("wrote", GOLDEN)
    else:
        test_golden_cpu()
        print("OK: bit-identical to", GOLDEN)
