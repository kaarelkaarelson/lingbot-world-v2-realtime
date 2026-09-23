import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import os
import warnings

# LINGBOT_ATTN=sage routes self-attention through SageAttention (INT8 QK,
# FP8/FP16 PV) instead of FlashAttention-2. Only for the plain batched call
# (no q_lens/k_lens), which is how the causal DiT calls it. sage_kvq keeps
# this fallback; the fused causal DiT then bypasses attention() with its
# pre-quantised KV cache (model_fast_fusion.py + sage_kvq.py).
_SAGE = None
if os.environ.get("LINGBOT_ATTN") in ("sage", "sage_kvq"):
    from sageattention import sageattn as _SAGE

# LINGBOT_ATTN=sage3 routes the same batched call through SageAttention 3's
# FP4 Blackwell kernel (`sageattn3_blackwell`) instead of Sage 2's INT8 path.
# Separate package (`sageattn3`, installed from thu-ml/SageAttention's
# sageattention3_blackwell/ subdir) and separate flag so sage/sage_kvq keep
# working unmodified; see bench/attn/sage3_build.md.
_SAGE3 = None
if os.environ.get("LINGBOT_ATTN") == "sage3":
    from sageattn3 import sageattn3_blackwell as _SAGE3

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if (_SAGE is not None or _SAGE3 is not None) and q_lens is None and k_lens is None and dropout_p == 0.:
        _dump = os.environ.get("LINGBOT_DUMP_QKV")
        if _dump and k.shape[1] >= 27000 and not os.path.exists(_dump):
            # one full-window self-attention call (q/k/v as passed to Sage) for the KV-quant probe
            torch.save({"q": q.detach().cpu(), "k": k.detach().cpu(), "v": v.detach().cpu()}, _dump)
        # LINGBOT_DUMP_QKV_DIR=<dir>: same idea but keeps the first N full-window calls of one
        # forward, numbered in call order, so attention decay can be read per layer rather than
        # from whichever layer happened to fire first. Receptive field varies with depth, so a
        # single layer cannot answer "how much window does this model actually use".
        _ddir = os.environ.get("LINGBOT_DUMP_QKV_DIR")
        if _ddir and k.shape[1] >= 27000:
            global _DUMP_N
            try:
                _DUMP_N
            except NameError:
                _DUMP_N = 0
            if _DUMP_N < int(os.environ.get("LINGBOT_DUMP_QKV_MAX", "30")):
                os.makedirs(_ddir, exist_ok=True)
                torch.save({"q": q.detach().cpu(), "k": k.detach().cpu(), "v": v.detach().cpu(),
                            "call": _DUMP_N, "lk": int(k.shape[1])},
                           os.path.join(_ddir, f"call_{_DUMP_N:03d}.pt"))
                _DUMP_N += 1
        # q/k/v: [B, L, H, D] == SageAttention's "NHD" layout. Cross-length
        # (Lq != Lk) is fine without a causal mask, which is our case.
        out_dtype = q.dtype
        if _SAGE3 is not None:
            # landmine: sageattn3_blackwell(q, k, v, attn_mask=None, is_causal=False,
            # per_block_mean=True, **kwargs) has no sm_scale parameter, so a non-default
            # softmax_scale would be silently swallowed by **kwargs. It hardcodes
            # softmax_scale = (packed_q.shape[-1] * 2) ** -0.5 == head_dim ** -0.5 (the packed
            # FP4 tensor is head_dim // 2 wide, so * 2 recovers head_dim) -- today that equals
            # our default (softmax_scale=None), so this is benign but must never go silently
            # wrong if a caller later passes an explicit scale.
            assert softmax_scale is None or softmax_scale == q.size(-1) ** -0.5, (
                "sageattn3_blackwell has no sm_scale param and hardcodes head_dim ** -0.5; "
                "a different softmax_scale would be silently ignored")
            # landmine: sageattn3_blackwell has NO tensor_layout argument and hardcodes HND
            # [B, H, L, D] (QL = q.size(2), pad_128 on dim 2, k.mean(dim=-2)). We call Sage 2
            # with NHD [B, L, H, D] above, so transpose in and back out; the permute cost is
            # counted in the timing.
            q3 = q.to(dtype).transpose(1, 2)
            k3 = k.to(dtype).transpose(1, 2)
            v3 = v.to(dtype).transpose(1, 2)
            # landmine: preprocess_qkv begins `k -= k.mean(dim=-2, keepdim=True)`, mutating
            # its `k` argument in place. `.to(dtype)` above returns the SAME tensor (a view
            # after transpose) when k is already bf16/fp16, so without this clone the live KV
            # cache would be corrupted in place -- gradual quality drift, not a crash.
            k3 = k3.clone()
            x = _SAGE3(q3, k3, v3, is_causal=causal).transpose(1, 2).contiguous()
        else:
            x = _SAGE(q.to(dtype), k.to(dtype), v.to(dtype), tensor_layout="NHD",
                      is_causal=causal, sm_scale=softmax_scale)
        return x.to(out_dtype)
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
