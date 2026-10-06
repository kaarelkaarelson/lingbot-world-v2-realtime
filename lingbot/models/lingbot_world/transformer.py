"""The causal DiT of LingBot-World 2.0, fused for speed (one compiled graph, no host syncs).

Same maths as the paper's `reference/wan/modules/model_fast.py`; see OPTIMIZATIONS.md exp. 10 and 15.
Some of the functions are borrowed from SelfForcing (https://github.com/guandeh17/Self-Forcing).
"""
import math
import os
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as torch_F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from wan.modules.model import (
    WanRMSNorm,
    WanLayerNorm,
    WanSelfAttention,
    rope_params,
    sinusoidal_embedding_1d
)

from lingbot.layers import kv_cache as kvc
from lingbot.layers.attention import attention, flash_attention
from lingbot.parallel import a2a_quant as a2aq

# Verification switches (A/B against upstream latents):
#   LINGBOT_DIT_FUSION_EXACT_T=1    time-embedding MLP on the L pre-expanded
#                                   rows and e0 [B, L, 6, C] exactly as upstream
#                                   (the 1-row MLP is the same maths through a
#                                   different GEMM tiling, ~1 fp32 ulp)
#   LINGBOT_DIT_FUSION_ROPE=fp32c   rope apply in compensated fp32 (double-
#                                   float products, no fp64 in the kernel;
#                                   fp64 is 1/64 rate on the RTX 5090) — the
#                                   fp64 (cos, sin) table is split once per
#                                   forward into fp32 hi + lo parts
_EXACT_T = os.environ.get("LINGBOT_DIT_FUSION_EXACT_T") == "1"
_SIM_A2A8 = os.environ.get("LINGBOT_SIM_A2A8", "")  # "qk" or "qkv": see lingbot/parallel/a2a_quant.py
_DUMP_QKV = os.environ.get("LINGBOT_DUMP_QKV", "")  # dir: save one forward's attention inputs (experiments/a2a8)
_dump_calls = [0]


def _dump_qkv(q, k, v, forward=30, layers=(0, 10, 20, 29), num_layers=30):
    # forward 30 = chunk 6, first denoise step (5 forwards per chunk): the KV window is full by then
    i = _dump_calls[0]
    _dump_calls[0] += 1
    if i // num_layers == forward and i % num_layers in layers:
        os.makedirs(_DUMP_QKV, exist_ok=True)
        torch.save({"q": q.cpu(), "k": k.cpu(), "v": v.cpu()}, os.path.join(_DUMP_QKV, f"layer{i % num_layers:02d}.pt"))
_ROPE_FP32C = os.environ.get("LINGBOT_DIT_FUSION_ROPE") == "fp32c"


def causal_rope_freqs(grid_sizes, freqs, start_frame=0):
    r"""
    Rotary multipliers for one forward, as real (cos, sin) tables. Computed
    once per forward in the model and shared by every layer's q and k (the
    original recomputed the complex table per layer per call).

    Args:
        grid_sizes: list of (F, H, W) Python ints, one per sample
        freqs: [1024, C / num_heads / 2, 2] float64, view_as_real of the
            polar rope table
    Returns:
        (cos, sin), each [B, L, 1, C / num_heads / 2] float64
    """
    c = freqs.size(1)
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for f, h, w in grid_sizes:
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1, 2).expand(f, h, w, -1, -1),
            freqs[1][:h].view(1, h, 1, -1, 2).expand(f, h, w, -1, -1),
            freqs[2][:w].view(1, 1, w, -1, 2).expand(f, h, w, -1, -1)
        ],
            dim=-2).reshape(f * h * w, 1, -1, 2)
        output.append(freqs_i)
    return torch.stack(output).unbind(-1)


def rope_fp32c_tables(rope):
    r"""
    (cos, sin) float64 -> ((cos_hi, cos_lo), (sin_hi, sin_lo)) float32 with
    hi + lo == the float64 value to ~2^-48, computed once per forward.
    """
    out = []
    for t in rope:
        hi = t.float()
        out.append((hi, (t - hi.double()).float()))
    return out


def _split(a):
    # Veltkamp: a == hi + lo with hi on 12 significant bits (fp32)
    t = a * 4097.0
    hi = t - (t - a)
    return hi, a - hi


def _two_prod(a, b):
    # Dekker: a * b == p + e exactly (torch has no fma)
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _rope_apply_fp32c(x, rope):
    r"""
    xr*cos - xi*sin and xr*sin + xi*cos as double-float sums: exact products
    and sums in (value, error) pairs, the error terms plus the lo parts of the
    table folded in last. ~48 bits of the fp64 result survive, so the bf16
    (or fp32) output rounds the same way as the fp64 path except within
    ~2^-45 of a rounding boundary (measured: 0 bf16 flips in 94 M elements).
    """
    (ch, cl), (sh, sl) = rope
    xr, xi = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    pr1, er1 = _two_prod(xr, ch)
    pr2, er2 = _two_prod(xi, sh)
    pi1, ei1 = _two_prod(xr, sh)
    pi2, ei2 = _two_prod(xi, ch)
    sr, esr = _two_sum(pr1, -pr2)
    si, esi = _two_sum(pi1, pi2)
    re = sr + (esr + (er1 - er2) + (xr * cl - xi * sl))
    im = si + (esi + (ei1 + ei2) + (xr * sl + xi * cl))
    return torch.stack([re, im], dim=-1).flatten(-2).type_as(x)


def causal_rope_apply(x, rope):
    r"""
    Same math as the complex form (x_i * freqs_i in float64, then cast back)
    written on real pairs so Inductor fuses it with the surrounding casts.
    `rope` is the (cos, sin) pair from `causal_rope_freqs`; x is [B, L, N, D].
    """
    if _ROPE_FP32C:
        return _rope_apply_fp32c(x, rope)
    cos, sin = rope
    pairs = x.to(torch.float64).unflatten(-1, (-1, 2))
    xr, xi = pairs.unbind(-1)
    out = torch.stack([xr * cos - xi * sin, xr * sin + xi * cos], dim=-1)
    return out.flatten(-2).type_as(x)


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        kv_cache=None,
        current_start=0,
        max_attention_size=1_000_000,
        frame_seqlen=None,
        seq_lens_int=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            grid_sizes: list of (F, H, W) Python ints, one per sample
            freqs: (cos, sin) rope tables for this chunk from `causal_rope_freqs`
            frame_seqlen(int, optional): Pre-computed H*W/(patch_h*patch_w).
            seq_lens_int(int, optional): Accepted for signature parity with
                the SP path (sp_attn_forward_causal). Unused here.
        """
        del seq_lens_int
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if frame_seqlen is None:
            frame_seqlen = grid_sizes[0][1] * grid_sizes[0][2]
        roped_query = causal_rope_apply(q, freqs).type_as(v)
        roped_key = causal_rope_apply(k, freqs).type_as(v)
        if _SIM_A2A8:
            # numerics of the 8-bit sequence-parallel exchange, on one GPU (lingbot/parallel/a2a_quant.py)
            roped_query = a2aq.roundtrip_int8(roped_query)
            roped_key = a2aq.roundtrip_int8(roped_key, smooth=True)
            if _SIM_A2A8 == "qkv":
                v = a2aq.roundtrip_fp8(v)
        sink_tokens = self.sink_size * frame_seqlen
        local_end_index, current_end = kvc.write(kv_cache, roped_key, v, current_start, sink_tokens,
                                                 self.local_attn_size)
        k_cache, v_cache = kvc.window(kv_cache, local_end_index, max_attention_size)
        if _DUMP_QKV:
            _dump_qkv(roped_query, k_cache, v_cache)
        x = attention(roped_query, k_cache, v_cache)
        kvc.commit(kv_cache, current_end, local_end_index)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, crossattn_cache=None,
                cross_attn_first_call=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            cross_attn_first_call(bool, optional): If provided, used as the
                "first call this generation" gate instead of reading
                crossattn_cache["is_init"].item() (which forces a CPU↔GPU
                sync). Caller (pipeline) tracks this as a Python bool.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if cross_attn_first_call is None:
                is_first = crossattn_cache["is_init"].item() == 0
            else:
                is_first = cross_attn_first_call
            if is_first:
                crossattn_cache["is_init"].fill_(1)
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"].copy_(k)
                crossattn_cache["v"].copy_(v)
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim=dim,
                                                num_heads=num_heads,
                                                local_attn_size=local_attn_size,
                                                sink_size=sink_size,
                                                qk_norm=qk_norm,
                                                eps=eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        self.cam_injector_layer1 = nn.Linear(dim, dim)
        self.cam_injector_layer2 = nn.Linear(dim, dim)
        self.cam_scale_layer = nn.Linear(dim, dim)
        self.cam_shift_layer = nn.Linear(dim, dim)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        dit_cond_dict=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        max_attention_size=1_000_000,
        frame_seqlen=None,
        cross_attn_first_call=None,
        seq_lens_int=None,
        cam_cache=None,
        cam_first_call=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 1, 6, C] (or [B, L, 6, C]); broadcast over L
            grid_sizes: list of (F, H, W) Python ints, one per sample
            freqs: (cos, sin) rope tables for this chunk from `causal_rope_freqs`
            cam_cache(dict, optional): per-layer ``scale``/``shift`` buffers
                [B, L, C]. The camera MLP depends only on the chunk's poses, so
                it is computed on the chunk's first forward (``cam_first_call``)
                and read back on the other forwards of the chunk.
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32
        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs, kv_cache, current_start, max_attention_size,
            frame_seqlen=frame_seqlen, seq_lens_int=seq_lens_int)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        # cam injection (only if dit_cond_dict is provided and contains c2ws_plucker_emb)
        if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
            if cam_cache is not None and not cam_first_call:
                cam_scale, cam_shift = cam_cache["scale"], cam_cache["shift"]
            else:
                c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]
                c2ws_hidden_states = self.cam_injector_layer2(torch_F.silu(self.cam_injector_layer1(c2ws_plucker_emb)))
                c2ws_hidden_states = c2ws_hidden_states + c2ws_plucker_emb
                cam_scale = self.cam_scale_layer(c2ws_hidden_states)
                cam_shift = self.cam_shift_layer(c2ws_hidden_states)
                if cam_cache is not None:
                    cam_cache["scale"].copy_(cam_scale)
                    cam_cache["shift"].copy_(cam_shift)
            x = (1.0 + cam_scale) * x + cam_shift

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None,
                           cross_attn_first_call=None):
            x = x + self.cross_attn(self.norm3(x), context, context_lens,
                                    crossattn_cache=crossattn_cache,
                                    cross_attn_first_call=cross_attn_first_call)
            y = self.ffn(
                self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache,
                           cross_attn_first_call=cross_attn_first_call)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, 1, C] (or [B, L1, C]); broadcast over L1
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class WanModelFast(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        control_dim = 6

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.patch_embedding_wancamctrl = nn.Linear(
            control_dim * 64 * patch_size[0] * patch_size[1] * patch_size[2], dim)
        self.c2ws_hidden_states_layer1 = nn.Linear(dim, dim)
        self.c2ws_hidden_states_layer2 = nn.Linear(dim, dim)

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        # Kept as real (cos, sin) pairs [1024, d/2, 2]: Inductor has no
        # complex kernels, so the rope maths runs on the real parts.
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.view_as_real(torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1))

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        dit_cond_dict=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        max_attention_size=1_000_000,
        frame_seqlen=None,
        cross_attn_first_call=None,
        cam_cache=None,
        cam_first_call=None,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            dit_cond_dict (`dict`, *optional*, defaults to None):
                Dictionary of conditioning signals. May contain key ``c2ws_plucker_emb``
                with camera Plucker embeddings of shape [B, C, F, H, W] for camera control.
            kv_cache (`list[dict]`, *optional*, defaults to None):
                Per-layer self-attention KV cache. Each dict contains keys ``k``, ``v``
                (Tensor of shape [B, kv_size, num_heads, head_dim]), ``global_end_index``,
                and ``local_end_index`` (scalar Tensors tracking cache position).
            crossattn_cache (`list[dict]`, *optional*, defaults to None):
                Per-layer cross-attention KV cache. Each dict contains keys ``k``, ``v``
                (Tensor of shape [B, text_len, num_heads, head_dim]) and ``is_init`` (bool).
            current_start (`int`, *optional*, defaults to 0):
                Token offset of the current chunk in the full sequence. Used to index
                into the KV cache and compute positional embeddings correctly.
            max_attention_size (`int`, *optional*, defaults to 1_000_000):
                Maximum number of KV tokens each query can attend to. Limits the
                effective context window of self-attention to control memory usage.
            cam_cache (`list[dict]`, *optional*, defaults to None):
                Per-layer camera-modulation cache (``scale``/``shift`` [B, L, C]),
                filled when ``cam_first_call`` is True and read otherwise.

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert y is not None

        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings. Grid sizes and sequence lengths stay Python ints (no
        # tensor round-trips): Dynamo traces them symbolically instead of
        # breaking the graph on .tolist()/.item().
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = [tuple(u.shape[2:]) for u in x]
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = [u.size(1) for u in x]
        assert max(seq_lens) <= seq_len
        x = torch.cat(x)

        # time embeddings. One timestep per sample: embed it once ([B, 1, C])
        # and let the blocks broadcast over the tokens instead of running the
        # MLP on L identical rows.
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_lens[0]) if _EXACT_T else t.unsqueeze(1)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt, lt = t.shape
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, lt)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self._embed_context(context)

        # rope tables for this chunk, shared by all layers
        if frame_seqlen is None:
            frame_seqlen = grid_sizes[0][1] * grid_sizes[0][2]
        rope = causal_rope_freqs(grid_sizes, self.freqs,
                                 start_frame=current_start // frame_seqlen)
        if _ROPE_FP32C:
            rope = rope_fp32c_tables(rope)

        # cam (skipped when the blocks will read their cam_cache)
        if (dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict
                and (cam_cache is None or cam_first_call)):
            c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]
            c2ws_plucker_emb = [
                rearrange(
                    i,
                    '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)',
                    c1=self.patch_size[0],
                    c2=self.patch_size[1],
                    c3=self.patch_size[2],
                ) for i in c2ws_plucker_emb
            ]
            c2ws_plucker_emb = torch.cat(
                c2ws_plucker_emb, dim=1)  # [1, (L1+...+Ln), C]
            c2ws_plucker_emb = self.patch_embedding_wancamctrl(c2ws_plucker_emb)
            c2ws_hidden_states = self.c2ws_hidden_states_layer2(
                torch_F.silu(self.c2ws_hidden_states_layer1(c2ws_plucker_emb)))
            dit_cond_dict = dict(dit_cond_dict)
            dit_cond_dict["c2ws_plucker_emb"] = (
                c2ws_plucker_emb + c2ws_hidden_states)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=rope,
            context=context,
            context_lens=context_lens,
            dit_cond_dict=dit_cond_dict,
            max_attention_size=max_attention_size,
            frame_seqlen=frame_seqlen,
            cross_attn_first_call=cross_attn_first_call,
            cam_first_call=cam_first_call)

        for block_index, block in enumerate(self.blocks):
            kwargs.update(
                {
                    "kv_cache": kv_cache[block_index],
                    "crossattn_cache": crossattn_cache[block_index],
                    "current_start": current_start,
                    "cam_cache": None if cam_cache is None else cam_cache[block_index],
                }
            )
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)

        return [u.float() for u in x]

    def _embed_context(self, context):
        return self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

    @torch.no_grad()
    def init_crossattn_cache(self, context, crossattn_cache):
        r"""
        Fill every layer's cross-attention K/V cache from the text context once
        per generation, so no forward needs the first-call branch (a Python
        bool the compiled graph would otherwise specialise on).
        """
        context = self._embed_context(context)
        for block, cache in zip(self.blocks, crossattn_cache):
            attn = block.cross_attn
            b, n, d = context.size(0), attn.num_heads, attn.head_dim
            cache["k"].copy_(attn.norm_k(attn.k(context)).view(b, -1, n, d))
            cache["v"].copy_(attn.v(context).view(b, -1, n, d))
            cache["is_init"].fill_(1)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (list[tuple[int, int, int]]):
                Original spatial-temporal grid dimensions before patching,
                    (F_patches, H_patches, W_patches) per sample

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

