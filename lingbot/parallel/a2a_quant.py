"""8-bit encoding for the sequence-parallel q/k/v exchange (2X_RTX5090_LEARNINGS.md, learning 8).

The sender quantizes its half of the tokens; the receiver dequantizes to bf16, so the KV cache and
SageAttention see bf16 as today. `roundtrip_*` apply both ends on one GPU, which reproduces the
exchange's numerics exactly without a second card (`LINGBOT_SIM_A2A8`).
"""
import torch

BLOCK = 128  # tokens per int8 scale, per head
PARTS = 2    # sequence shards: one per card


def _int8_part(x, smooth):
    # x [B, t, H, d]. Subtracting the shard's own mean (sent along with the scales) keeps channel
    # offsets in K from eating the int8 range; the receiver adds it back, so it need not be shared.
    m = x.float().mean(dim=1, keepdim=True) if smooth else None
    xf = x.float() - m if smooth else x.float()
    b, t, h, d = xf.shape
    pad = (-t) % BLOCK
    xb = torch.nn.functional.pad(xf, (0, 0, 0, 0, 0, pad)).view(b, -1, BLOCK, h, d)
    s = xb.abs().amax(dim=(2, 4), keepdim=True).clamp_min(1e-8) / 127
    y = ((xb / s).round().clamp(-127, 127) * s).view(b, -1, h, d)[:, :t]
    return (y + m if smooth else y).to(x.dtype)


def _fp8_part(x):
    # per (head, channel) scale over the shard's tokens, e4m3
    xf = x.float()
    s = xf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 448
    return ((xf / s).to(torch.float8_e4m3fn).float() * s).to(x.dtype)


def roundtrip_int8(x, smooth=False):
    return torch.cat([_int8_part(p, smooth) for p in x.chunk(PARTS, dim=1)], dim=1)


def roundtrip_fp8(x):
    return torch.cat([_fp8_part(p) for p in x.chunk(PARTS, dim=1)], dim=1)
