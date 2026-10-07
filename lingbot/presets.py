"""LINGBOT_* environment presets shared by `python -m lingbot.generate` and `lingbot play`.

Presets are applied as environment defaults before `wan` is imported, because the fused DiT
and the attention backend are chosen at import time. Explicitly exported LINGBOT_* variables
win over the preset.
"""
import os

PRESETS = {
    # 16.2 FPS on one RTX 5090 (OPTIMIZATIONS.md in lingbot-world-bench, exp. 15):
    # torch.compile + coordinate-descent tuning, fused DiT, sync-free loop,
    # compensated-fp32 RoPE, FP8 rowwise linears, SageAttention, fused fp16 VAE.
    "fast": {
        "LINGBOT_TORCH_COMPILE": "1", "LINGBOT_INDUCTOR_TUNE": "1",
        "LINGBOT_DIT_FUSION": "1", "LINGBOT_SYNCFREE": "1",
        "LINGBOT_DIT_FUSION_ROPE": "fp32c", "LINGBOT_FP8": "1",
        "LINGBOT_ATTN": "sage", "LINGBOT_VAE_FUSED": "1",
        "LINGBOT_VAE_SUBPIXEL": "1", "LINGBOT_VAE_WARM": "1",
    },
    # Same, with the time-embedding MLP on L rows as in the paper (14.8 FPS). Still FP8 +
    # SageAttention, so not bit-identical; that needs LINGBOT_FP8=0 LINGBOT_ATTN= LINGBOT_TORCH_COMPILE=.
    "exact": {"LINGBOT_DIT_FUSION_EXACT_T": "1"},
    # The paper's code in reference/, unmodified: for A/B comparisons (lingbot.generate only).
    "stock": {},
}
PRESETS["exact"] = {**PRESETS["fast"], **PRESETS["exact"]}


def apply_preset(preset: str) -> None:
    """Export the preset's LINGBOT_* variables as defaults (existing values win)."""
    if preset not in PRESETS:
        raise SystemExit(f"--preset must be one of {sorted(PRESETS)}, got {preset!r}")
    for k, v in PRESETS[preset].items():
        os.environ.setdefault(k, v)
