"""FP8 linear layers: rowwise (`LINGBOT_FP8=1`) or MXFP8 (`LINGBOT_FP8=mx`).

Weights are converted to FP8 e4m3 once at load, with one FP32 scale per output row. Activations
stay BF16 in memory and are converted to FP8 per token on every call; under torch.compile that
conversion fuses into the neighbouring kernels. The GEMM accumulates in FP32 and writes BF16.
"""
import torch


class FP8Linear(torch.nn.Module):
    FMAX = torch.finfo(torch.float8_e4m3fn).max

    def __init__(self, lin):
        super().__init__()
        w = lin.weight.detach()
        w_scale = (w.abs().amax(dim=1, keepdim=True).float() / self.FMAX).clamp(min=1e-12)  # [N,1]
        self.register_buffer("w8", (w.float() / w_scale).to(torch.float8_e4m3fn))            # [N,K]
        self.register_buffer("w_scale_t", w_scale.t().contiguous())                           # [1,N]
        self.bias = lin.bias

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        x_scale = (x2.abs().amax(dim=1, keepdim=True).float() / self.FMAX).clamp(min=1e-12)   # [M,1]
        x8 = (x2.float() / x_scale).to(torch.float8_e4m3fn)
        y = torch._scaled_mm(x8, self.w8.t(), scale_a=x_scale, scale_b=self.w_scale_t,
                             bias=None if self.bias is None else self.bias.to(torch.bfloat16),
                             out_dtype=torch.bfloat16)
        return y.reshape(*shape[:-1], y.shape[-1])


def _to_blocked(scales):
    """[rows, cols] block scales -> cuBLAS's swizzled layout (128-row x 4-col tiles), as torchao lays them out."""
    rows, cols = scales.shape
    rb, cb = -(-rows // 128), -(-cols // 4)
    if (rows, cols) != (rb * 128, cb * 4):
        padded = torch.zeros(rb * 128, cb * 4, dtype=scales.dtype, device=scales.device)
        padded[:rows, :cols] = scales
        scales = padded
    blocks = scales.view(rb, 128, cb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _mx_quant(x):
    """[M, K] -> FP8 e4m3 values and one power-of-two (e8m0) scale per 32 consecutive values along K,
    rounded up so no value overflows FP8's 448."""
    m, k = x.shape
    xb = x.float().view(m, k // 32, 32)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=2.0 ** -126)
    scale = torch.exp2(torch.ceil(torch.log2(amax / FP8Linear.FMAX)))
    q = (xb / scale).to(torch.float8_e4m3fn).view(m, k)
    return q, scale.view(m, k // 32).to(torch.float8_e8m0fnu)


class MXFP8Linear(torch.nn.Module):
    """MXFP8: FP8 e4m3 with a power-of-two scale per 32 values along K, for weights and activations.
    Same 8-bit class as FP8Linear, finer scaling; cuBLAS runs it ~1.35-1.5x faster on the RTX 5090
    (2X_RTX5090_LEARNINGS.md, learning 16)."""

    def __init__(self, lin):
        super().__init__()
        w8, ws = _mx_quant(lin.weight.detach())
        self.register_buffer("w8", w8)                        # [N, K]
        self.register_buffer("w_scale", _to_blocked(ws))
        self.bias = lin.bias

    def forward(self, x):
        shape = x.shape
        x8, xs = _mx_quant(x.reshape(-1, shape[-1]))
        y = torch._scaled_mm(x8, self.w8.t(), scale_a=_to_blocked(xs), scale_b=self.w_scale, out_dtype=torch.bfloat16)
        if self.bias is not None:
            y = y + self.bias.to(torch.bfloat16)
        return y.reshape(*shape[:-1], y.shape[-1])


def convert_to_fp8(blocks, mx=False):
    """Swap the large nn.Linear layers inside `blocks` for FP8Linear (or MXFP8Linear). Only the blocks: the
    time-embedding MLP and the head stay FP32/BF16. Returns (fp8 count, linear count)."""
    n_all, n_fp8 = 0, 0
    for parent in list(blocks.modules()):
        for name, m in list(parent.named_children()):
            if not isinstance(m, torch.nn.Linear):
                continue
            n_all += 1
            if (m.in_features >= 1024 and m.out_features >= 1024
                    and m.in_features % (32 if mx else 16) == 0 and m.out_features % 16 == 0):
                setattr(parent, name, (MXFP8Linear if mx else FP8Linear)(m))
                n_fp8 += 1
    return n_fp8, n_all
