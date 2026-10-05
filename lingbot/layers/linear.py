"""FP8 rowwise linear layer (`LINGBOT_FP8=1`).

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


def convert_to_fp8(blocks):
    """Swap the large nn.Linear layers inside `blocks` for FP8Linear. Only the blocks: the
    time-embedding MLP and the head stay FP32/BF16. Returns (fp8 count, linear count)."""
    n_all, n_fp8 = 0, 0
    for parent in list(blocks.modules()):
        for name, m in list(parent.named_children()):
            if not isinstance(m, torch.nn.Linear):
                continue
            n_all += 1
            if (m.in_features >= 1024 and m.out_features >= 1024
                    and m.in_features % 16 == 0 and m.out_features % 16 == 0):
                setattr(parent, name, FP8Linear(m))
                n_fp8 += 1
    return n_fp8, n_all
