"""Pre-quantised KV cache for SageAttention 2.2 on sm_120 (LINGBOT_ATTN=sage_kvq).

sageattn() re-quantises the whole K/V slice on every call: K to int8 per
64-token block with the sequence mean `km` subtracted, V to fp8 e4m3 per
channel, transposed to [B, D, H, L_pad] with every 16 tokens permuted for the
fp8 mma. Here the quantised K/V live in the layer's cache next to the bf16
ones: a full re-quant on a chunk's first forward (fresh km / V scale, after
the eviction shift regrouped the blocks), then only Q and the current chunk's
block-aligned superset on the other forwards. The attention kernel is the one
sageattn() dispatches to on sm_120 (per-warp Q, pv_accum_dtype="fp32+fp16").

The quantisation is plain torch so Inductor fuses it into the DiT graph
(Sage's _fused.* pybind functions are not custom ops and would graph-break).
Rounding matches the CUDA kernels: int8 = cvt.rni.sat (round-half-even,
saturate), fp8 = cvt.rn (round-nearest), scale = amax / 127 with a 1e-7 floor
resp. amax / 2.25, applied as x * (127 / amax) in fp32. The one addition: V is
clamped to +-2.25 so a token past the chunk's frozen amax * V_MARGIN saturates
instead of overflowing the kernel's fp16 PV accumulator (64 keys * 448 * 2.25
is its budget).
"""
import torch
import torch.nn.functional as F

try:
    from sageattention import sm89_compile as _sm89
except ImportError:  # CPU tests substitute a torch emulation of the kernel
    _sm89 = None

BLK_K, BLK_Q, WARP_Q = 64, 128, 32
# fp32+fp16 PV: P <= 2^8.807 = 448 and 64 keys * 448 * 2.25 < fp16 max
V_SCALE_MAX = 2.25
# headroom of the per-chunk frozen V scale for the chunk's later forwards
V_MARGIN = 1.5
# transpose_pad_permute_cuda: output token j of each 16 = input token V_PERM[j]
V_PERM = [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]


def alloc_cache(shape, dtype, device):
    """Quantised buffers for one layer's cache dict; `shape` = [B, kv_size, H, D]."""
    b, kv_size, h, d = shape
    l_pad = (kv_size + BLK_K - 1) // BLK_K * BLK_K
    return {
        "k_int8": torch.zeros(b, kv_size, h, d, dtype=torch.int8, device=device),
        "k_scale": torch.zeros(b, h, l_pad // BLK_K, dtype=torch.float32, device=device),
        "v_fp8": torch.zeros(b, d, h, l_pad, dtype=torch.float8_e4m3fn, device=device),
        "v_amax": torch.ones(b, h, d, dtype=torch.float32, device=device),
        "km": torch.zeros(b, 1, h, d, dtype=dtype, device=device),
    }


def release(kv_cache):
    for c in kv_cache:
        for key in ("k_int8", "k_scale", "v_fp8", "v_amax", "km"):
            c.pop(key, None)


def _quant_int8(x, blk, pad_to, mean=None):
    """QuantInt8Kernel: x [B, n, H, D] -> int8 [B, n, H, D] and fp32 scale
    [B, H, ceil(n / pad_to) * pad_to / blk]; tokens past n count as zero."""
    b, n, h, d = x.shape
    xf = x.float() if mean is None else x.float() - mean.float()
    lp = (n + pad_to - 1) // pad_to * pad_to
    xf = F.pad(xf, (0, 0, 0, 0, 0, lp - n)).view(b, lp // blk, blk, h, d)
    amax = xf.abs().amax(dim=(2, 4)).clamp_min(1e-7)  # [B, nb, H]
    xq = (xf * (127.0 / amax)[:, :, None, :, None]).round().clamp(-128, 127).to(torch.int8)
    return xq.view(b, lp, h, d)[:, :n], (amax / 127.0).transpose(1, 2).contiguous()


def _quant_v_fp8(v, v_amax):
    """transpose_pad_permute_cuda + scale_fuse_quant_cuda: v [B, n, H, D] with
    n % 16 == 0 -> fp8 [B, D, H, n], tokens permuted inside every 16."""
    b, n, h, d = v.shape
    vp = v.view(b, n // 16, 16, h, d)[:, :, V_PERM].view(b, n, h, d)
    x = vp.float() * (V_SCALE_MAX / v_amax)[:, None]
    return x.clamp(-V_SCALE_MAX, V_SCALE_MAX).to(torch.float8_e4m3fn).permute(0, 3, 2, 1)


def write_chunk(cache, start, end):
    """Quantise cache["k"/"v"][:, start:end] with the frozen km / v_amax. The
    attended slice always starts at token 0, so the 64-token K blocks and
    16-token V groups are aligned to the cache: the rewrite covers
    [floor64(start), end); tokens past `end` count as zero (as in sageattn)
    and are re-quantised when the next chunk lands in their block."""
    a = start // BLK_K * BLK_K
    k_int8, k_scale = _quant_int8(cache["k"][:, a:end], BLK_K, BLK_K, cache["km"])
    cache["k_int8"][:, a:end] = k_int8
    cache["k_scale"][:, :, a // BLK_K:(end + BLK_K - 1) // BLK_K] = k_scale
    e16 = (end + 15) // 16 * 16
    v = F.pad(cache["v"][:, a:end], (0, 0, 0, 0, 0, e16 - end))
    cache["v_fp8"][:, :, :, a:e16] = _quant_v_fp8(v, cache["v_amax"])


def requant_all(cache, kv_len, margin=V_MARGIN):
    """Fresh km / V scale over the attended slice, then quantise all of it."""
    k, v = cache["k"][:, :kv_len], cache["v"][:, :kv_len]
    cache["km"].copy_(k.mean(dim=1, keepdim=True))
    cache["v_amax"].copy_(v.abs().amax(dim=1).float() * margin)
    write_chunk(cache, 0, kv_len)


def attend(q, cache, kv_len, sm_scale=None):
    """sageattn(q, k, v, "NHD") on sm_120 with K/V taken from the cache;
    q [B, Lq, H, D] -> [B, Lq, H, D]. Only Q is quantised here."""
    q_int8, q_scale = _quant_int8(q, WARP_Q, BLK_Q)
    nb = (kv_len + BLK_K - 1) // BLK_K
    o = torch.empty_like(q)
    _sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
        q_int8, cache["k_int8"][:, :kv_len], cache["v_fp8"], o, q_scale,
        cache["k_scale"][:, :, :nb].contiguous(), cache["v_amax"] / V_SCALE_MAX,
        0, 0, 2, q.size(-1) ** -0.5 if sm_scale is None else sm_scale, 0)
    return o
