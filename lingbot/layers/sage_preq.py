"""SageAttention 2.2 (sm_120 path) with V quantised straight from the KV cache (`LINGBOT_SAGE_PREQ=1`).

`sageattn()` quantises V in three launches per call: TransposePadPermuteKernel (bf16 copy of the whole window,
transposed to [B, D, H, L_pad] with every 16 tokens permuted), a per-channel min/max pass over that copy, and the
fp8 quantisation pass. Here one Triton kernel reads the cache and writes the fp8 tensor in its final layout, and the
per-channel min/max comes from a per-64-token-block table kept next to the cache (only the blocks this forward
wrote are recomputed; the whole window is re-read only after an eviction shift). Q and K go through the very same
`per_warp_int8` calls as stock, and the same sm89 kernel runs on the result.

Bit-identical to `sageattn(q, k_win, v_win, "NHD")` because every step reproduces Sage's arithmetic:
- max / min are exact and order-free, so block-wise reduction equals the kernel's reduction. The kernel also reduces
  over the zero padding up to a multiple of 16 tokens, so 0 joins the min/max when kv_len % 16 != 0.
- the divisions are `div.full.f32`, what nvcc emits under Sage's `--use_fast_math` (inline PTX, not Triton's `/`).
- x * (2.25 / amax) in fp32, then round-to-nearest e4m3 (values are within +-2.25, so saturation never applies).
- token position q of each 16-group holds input token V_PERM[q], V_PERM = [0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15].
K cannot be cached the same way: its smoothing mean is over the whole window and changes whenever any token does.
"""
import torch
import triton
import triton.language as tl
from sageattention import sm89_compile
from sageattention.quant import per_warp_int8 as per_warp_int8_cuda

BLK = 64         # tokens per min/max block (= Sage's CTA_K)
BC = 128         # (head, channel) columns per program
V_SCALE_MAX = 2.25  # pv_accum_dtype "fp32+fp16"


@triton.jit
def _fdiv(a, b):
    return tl.inline_asm_elementwise("div.full.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32,
                                     is_pure=True, pack=1)


@triton.jit
def _blk_minmax(v_ptr, bmax_ptr, bmin_ptr, first_blk, end, stride_tok, HD, BLK: tl.constexpr, BC: tl.constexpr):
    b = first_blk + tl.program_id(0)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    tok = b * BLK + tl.arange(0, BLK)
    m = (tok < end)[:, None] & (cols < HD)[None, :]
    x = tl.load(v_ptr + tok[:, None] * stride_tok + cols[None, :], mask=m, other=0.0).to(tl.float32)
    tl.store(bmax_ptr + b * HD + cols, tl.max(tl.where(m, x, float("-inf")), axis=0), mask=cols < HD)
    tl.store(bmin_ptr + b * HD + cols, tl.min(tl.where(m, x, float("inf")), axis=0), mask=cols < HD)


@triton.jit
def _v_scale(bmax_ptr, bmin_ptr, nb, has_pad, scale_ptr, recp_ptr, HD, NBLK: tl.constexpr, BC: tl.constexpr):
    cols = tl.program_id(0) * BC + tl.arange(0, BC)
    cm = cols < HD
    mx = tl.full([BC], float("-inf"), tl.float32)
    mn = tl.full([BC], float("inf"), tl.float32)
    for b0 in range(0, nb, NBLK):
        rows = b0 + tl.arange(0, NBLK)
        m = (rows < nb)[:, None] & cm[None, :]
        off = rows[:, None] * HD + cols[None, :]
        mx = tl.maximum(mx, tl.max(tl.load(bmax_ptr + off, mask=m, other=float("-inf")), axis=0))
        mn = tl.minimum(mn, tl.min(tl.load(bmin_ptr + off, mask=m, other=float("inf")), axis=0))
    mx = tl.where(has_pad != 0, tl.maximum(mx, 0.0), mx)
    mn = tl.where(has_pad != 0, tl.minimum(mn, 0.0), mn)
    amax = tl.maximum(tl.abs(mx), tl.abs(mn))
    smax = tl.full([BC], 2.25, tl.float32)
    tl.store(scale_ptr + cols, _fdiv(amax, smax), mask=cm)
    tl.store(recp_ptr + cols, _fdiv(smax, amax), mask=cm)


@triton.jit
def _quant_v(v_ptr, recp_ptr, out_ptr, kv_len, stride_tok, stride_d, stride_h, D: tl.constexpr):
    tb = tl.program_id(0)
    h = tl.program_id(1)
    q = tl.arange(0, 64)
    r = (q & 1) + ((q >> 2) & 3) * 2 + ((q >> 1) & 1) * 8   # V_PERM[q % 16]
    src = tb * 64 + (q & 48) + r
    d = tl.arange(0, D)
    x = tl.load(v_ptr + src[:, None] * stride_tok + h * D + d[None, :], mask=(src < kv_len)[:, None],
                other=0.0).to(tl.float32)
    y = (x * tl.load(recp_ptr + h * D + d)[None, :]).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(out_ptr + d[:, None] * stride_d + h * stride_h + (tb * 64 + q)[None, :], tl.trans(y))


def usable(q, kc, end, max_attention_size):
    """The cases this path reproduces exactly; anything else takes the stock call."""
    v = kc["v"]
    return (q.shape[0] == 1 and q.dtype == kc["k"].dtype == v.dtype == torch.bfloat16 and v.shape[-1] == 128
            and v.is_contiguous() and end - max_attention_size <= 0)


def _update(kc, first_blk, end):
    """Recompute the min/max blocks covering tokens [first_blk * BLK, end)."""
    v = kc["v"]
    h, d = v.shape[2], v.shape[3]
    if "pq_vmax" not in kc:
        nb = (v.shape[1] + BLK - 1) // BLK
        kc["pq_vmax"] = torch.empty(nb, h * d, dtype=torch.float32, device=v.device)
        kc["pq_vmin"] = torch.empty_like(kc["pq_vmax"])
        first_blk = 0
    n = (end + BLK - 1) // BLK - first_blk
    _blk_minmax[(n, triton.cdiv(h * d, BC))](v, kc["pq_vmax"], kc["pq_vmin"], first_blk, end, v.stride(1), h * d,
                                              BLK=BLK, BC=BC)


def refresh(kc, sink_tokens):
    """The window just shifted left past the sink (kv_cache.evict): re-read everything after it."""
    torch.cuda.set_device(kc["v"].device)
    _update(kc, sink_tokens // BLK, kc["v"].shape[1])


def attend(q, kc, start, end):
    """`sageattn(q, k_win, v_win, tensor_layout="NHD")` with the window [0, end) of cache `kc`, whose tokens
    [start, end) were written by this forward. q [1, Lq, H, D] -> [1, Lq, H, D]."""
    torch.cuda.set_device(q.device)
    v = kc["v"]
    h, d = v.shape[2], v.shape[3]
    _update(kc, start // BLK, end)
    k = kc["k"][:, :end]
    km = k.mean(dim=1, keepdim=True)
    q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km, tensor_layout="NHD", BLKQ=128, WARPQ=32, BLKK=64)
    l_pad = (end + BLK - 1) // BLK * BLK
    v_scale = torch.empty(1, h, d, dtype=torch.float32, device=q.device)
    recp = torch.empty(h * d, dtype=torch.float32, device=q.device)
    nb = l_pad // BLK
    _v_scale[(triton.cdiv(h * d, BC),)](kc["pq_vmax"], kc["pq_vmin"], nb, int(end % 16 != 0), v_scale, recp, h * d,
                                         NBLK=32, BC=BC)
    v_fp8 = torch.empty(1, d, h, l_pad, dtype=torch.uint8, device=q.device)
    _quant_v[(nb, h)](v, recp, v_fp8, end, v.stride(1), h * l_pad, l_pad, D=d)
    o = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
        q_int8, k_int8, v_fp8.view(torch.float8_e4m3fn), o, q_scale, k_scale, v_scale, 0, 0, 2, d ** -0.5, 0)
    return o
