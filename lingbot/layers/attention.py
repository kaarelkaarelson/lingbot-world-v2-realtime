"""Attention backends: FlashAttention (the paper's) or SageAttention 2.2.

`LINGBOT_ATTN=sage` routes the plain batched self-attention call (no q_lens/k_lens, the only way
the causal DiT calls it) through SageAttention: INT8 QK^T, FP8 PV. Everything else, including
cross-attention, uses the paper's FlashAttention wrapper unchanged.
"""
import os

from wan.modules.attention import FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE
from wan.modules.attention import attention as _paper_attention
from wan.modules.attention import flash_attention as _paper_flash

__all__ = ["attention", "flash_attention"]


def flash_attention(q, k, v, k_lens=None, **kw):
    """The paper's FlashAttention call; PyTorch SDPA when flash_attn is not installed (e.g. no cu130 wheel)."""
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return _paper_flash(q, k, v, k_lens=k_lens, **kw)
    return _paper_attention(q, k, v, k_lens=k_lens, **kw)

_SAGE = None
if os.environ.get("LINGBOT_ATTN") == "sage":
    from sageattention import sageattn as _SAGE


def attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None, q_scale=None,
              causal=False, window_size=(-1, -1), deterministic=False, dtype=None, fa_version=None):
    import torch
    dtype = torch.bfloat16 if dtype is None else dtype
    if _SAGE is not None and q_lens is None and k_lens is None and dropout_p == 0.:
        # q/k/v are [B, L, H, D], SageAttention's "NHD" layout; Lq != Lk is fine without a mask
        out_dtype = q.dtype
        x = _SAGE(q.to(dtype), k.to(dtype), v.to(dtype), tensor_layout="NHD",
                  is_causal=causal, sm_scale=softmax_scale)
        return x.to(out_dtype)
    return _paper_attention(q, k, v, q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p,
                            softmax_scale=softmax_scale, q_scale=q_scale, causal=causal,
                            window_size=window_size, deterministic=deterministic, dtype=dtype,
                            fa_version=fa_version)
