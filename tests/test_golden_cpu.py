"""Bit-exact regression test for the fused DiT: its outputs must not change across refactors.

Runs the tiny-config 5-chunk causal loop from test_dit_fusion_cpu.py (seeded weights, CPU SDPA in
place of FlashAttention/SageAttention) and compares every denoising forward with torch.equal against
tests/golden/dit_fused.pt, in four modes: default and fp32c RoPE, each with and without LINGBOT_DIT_FUSION_EXACT_T=1
(fp32c is the fast preset, fp32c + EXACT_T the exact preset).

Regenerate only when a numerics change is intended:  python tests/test_golden_cpu.py --regen
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_dit_fusion_cpu as T  # noqa: E402

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "dit_fused.pt")
GOLDEN_VAE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "vae_fused.pt")


def outputs():
    T._stub_packages()
    up = T.load_model_module(T.UPSTREAM_MODEL, "wan.modules.model_fast")
    state = T.build(up).state_dict()
    out = {}
    # fast preset = fp32c rope; exact preset = fp32c rope + EXACT_T
    for mode, env in (("default", {}), ("exact_t", {"LINGBOT_DIT_FUSION_EXACT_T": "1"}),
                      ("fp32c", {"LINGBOT_DIT_FUSION_ROPE": "fp32c"}),
                      ("fp32c_exact_t", {"LINGBOT_DIT_FUSION_ROPE": "fp32c", "LINGBOT_DIT_FUSION_EXACT_T": "1"})):
        os.environ.update(env)
        try:
            m = T.build(T.load_model_module(T.PATCHED_MODEL, f"wan.modules.model_fast_fusion_{mode}"), state)
            out[mode] = T.run_loop(m, fusion=True)
        finally:
            for k in env:
                del os.environ[k]
    return out


def vae_outputs():
    """Fused decoder, eager fp32 (CPU), the small seeded VAE from test_vae_fused_cpu.py."""
    import copy
    import test_vae_fused_cpu as V
    fused = V.fused  # lingbot/models/lingbot_world/vae.py
    vae = V.build()
    z = torch.randn(16, 9, 8, 16, generator=torch.Generator().manual_seed(7))
    out = {}
    # fp32 only: fp16 conv3d on CPU takes tens of minutes; the shipped fp16 path is the same code
    for dtype, sub in ((torch.float32, False), (torch.float32, True)):
        dec = fused.FusedDecoder(copy.deepcopy(vae), compile_mode=None, channels_last=False,
                                 dtype=dtype, subpixel=sub)
        out[f"{dtype}, subpixel={sub}"] = dec.decode(z)
    return out


def test_golden_vae_cpu():
    ref, got = torch.load(GOLDEN_VAE), vae_outputs()
    for k in ref:
        assert torch.equal(ref[k], got[k]), f"vae {k}: max|diff|={(ref[k] - got[k]).abs().max().item():.3e}"


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
        torch.save(vae_outputs(), GOLDEN_VAE)
        print("wrote", GOLDEN, GOLDEN_VAE)
    else:
        test_golden_cpu()
        test_golden_vae_cpu()
        print("OK: bit-identical to", GOLDEN)
