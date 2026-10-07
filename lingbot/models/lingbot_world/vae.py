"""Compile-friendly driver for the stock Wan 2.1 VAE decoder (exp. 11).

Same modules and weights as `vae2_1.Decoder3d`; only the way the graph is driven
changes, so the numerics are those of the fp16 + channels_last_3d path (exp. 9):

* the temporal cache is a list of 2-frame tensors returned functionally instead
  of a mutated Python list, so one latent step is one static graph that Dynamo
  captures whole (the stock loop breaks the graph at every layer);
* `CausalConv3d`'s clone + cat + F.pad becomes one cat (spatial zero padding is
  handed to cuDNN); the next cache is a 2-frame clone of the same buffer;
* the decoder holds fp16 conv weights (exp. 9c's true-fp16 model, no autocast);
  RMS_norm -> SiLU is written as an fp32 island, which under torch.compile is
  free (Inductor computes fused kernels in fp32 and only stores fp16) and in
  eager costs what autocast cost;
* the nearest-exact upsample is a pure gather, so its fp32 round trip is skipped;
* `LINGBOT_VAE_SUBPIXEL=1` (or `subpixel=True`): each upsample `Resample` (nearest-exact 2x
  then 3x3 conv) runs as one 2x2 conv with 4*Cout channels on the un-upsampled input plus a
  pixel shuffle - the same linear map with pre-summed taps, 2.25x fewer MACs.

Under torch.compile every norm / SiLU / cast / residual add / cache copy chain
fuses into one read and one write per block.  Streaming API: `decode_step`.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from wan.modules.vae2_1 import CACHE_T, AttentionBlock, Resample, ResidualBlock

CL3D = torch.channels_last_3d


def _causal_conv(conv, x, cache, new):
    """`cache`: previous step's slots (None on the first latent = zero pad); slot = visit order."""
    if conv._padding[4] == 0:  # 1x1x1: no temporal context, no cache slot
        return F.conv3d(x, conv.weight, conv.bias)
    xin = torch.cat([cache[len(new)], x], 2) if cache is not None else F.pad(x, (0, 0, 0, 0, conv._padding[4], 0))
    new.append(xin[:, :, -CACHE_T:].clone(memory_format=CL3D))
    return F.conv3d(xin, conv.weight, conv.bias, conv.stride, (0, conv._padding[2], conv._padding[0]))


def _norm(norm, x):
    return F.normalize(x.float(), dim=1) * norm.scale * norm.gamma + norm.bias


def _norm_silu(norm, x):
    return F.silu(_norm(norm, x)).to(x.dtype)


def _res_block(blk, x, cache, new):
    r = blk.residual  # RMS_norm, SiLU, CausalConv3d, RMS_norm, SiLU, Dropout, CausalConv3d
    h = x if isinstance(blk.shortcut, nn.Identity) else _causal_conv(blk.shortcut, x, cache, new)
    y = _causal_conv(r[2], _norm_silu(r[0], x), cache, new)
    y = _causal_conv(r[6], _norm_silu(r[3], y), cache, new)
    return y + h


def _attn(blk, x):
    b, c, t, h, w = x.shape
    identity = x
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x = _norm(blk.norm, x).to(identity.dtype)
    q, k, v = blk.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
    x = F.scaled_dot_product_attention(q, k, v)
    x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
    x = blk.proj(x).reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    return x + identity


# 3-tap -> 2-tap maps per output phase: nearest-exact 2x upsampled row 2i reads source rows
# i-1 (w0) and i (w1 + w2); row 2i+1 reads i (w0 + w1) and i+1 (w2). Same for columns.
_PHASE = torch.tensor([[[1., 0., 0.], [0., 1., 1.]], [[1., 1., 0.], [0., 0., 1.]]])


def _subpixel_weight(conv):
    """[Cout, Cin, 3, 3] -> [4*Cout, Cin, 2, 2] (fp32), phase (ph, pw) at channel 4*o + 2*ph + pw."""
    w = conv.weight.detach().float()
    w = torch.einsum('pak,qbl,ockl->opqcab', _PHASE.to(w.device), _PHASE.to(w.device), w)
    return w.reshape(-1, w.shape[3], 2, 2), conv.bias.detach().float().repeat_interleave(4)


def _subpixel_conv(x, weight, bias):
    """Padding 1 on the source gives (h+1, w+1) taps; phase (ph, pw) is the window [ph:ph+h, pw:pw+w]."""
    n, _, h, w = x.shape
    y = F.conv2d(x, weight, bias, padding=1).reshape(n, -1, 2, 2, h + 1, w + 1)
    y = torch.stack([y[:, :, 0, 0, :h, :w], y[:, :, 0, 1, :h, 1:], y[:, :, 1, 0, 1:, :w], y[:, :, 1, 1, 1:, 1:]], 2)
    y = F.pixel_shuffle(y.reshape(n, -1, h, w), 2)
    # a conv's output takes its weight's layout; pixel_shuffle's does not, and Inductor would hand the next conv NCHW
    cl = weight.is_contiguous(memory_format=torch.channels_last)
    return y.contiguous(memory_format=torch.channels_last if cl else torch.contiguous_format)


def _resample(rs, x, cache, new):
    b, c, t, h, w = x.shape
    if rs.mode == 'upsample3d':
        if cache is None:
            # stock 'Rep': the first latent skips time_conv; from then on the slot is two zero frames
            new.append(x.new_zeros((b, c, CACHE_T, h, w)).contiguous(memory_format=CL3D))
        else:
            x = _causal_conv(rs.time_conv, x, cache, new)
            x = x.reshape(b, 2, c, t, h, w)
            x = torch.stack((x[:, 0], x[:, 1]), 3).reshape(b, c, t * 2, h, w)
    t = x.shape[2]
    up, conv = rs.resample
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    if hasattr(rs, 'sub_weight'):
        x = _subpixel_conv(x, rs.sub_weight, rs.sub_bias)
    else:
        x = conv(F.interpolate(x, scale_factor=up.scale_factor, mode=up.mode))
    return x.reshape(b, t, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)


def _decoder_step(dec, x, cache):
    """One latent through `Decoder3d`. cache: None (first latent) or the list from the previous step."""
    new = []
    x = _causal_conv(dec.conv1, x, cache, new)
    for layer in list(dec.middle) + list(dec.upsamples):
        if isinstance(layer, ResidualBlock):
            x = _res_block(layer, x, cache, new)
        elif isinstance(layer, Resample):
            x = _resample(layer, x, cache, new)
        elif isinstance(layer, AttentionBlock):
            x = _attn(layer, x)
    x = _causal_conv(dec.head[2], _norm_silu(dec.head[0], x), cache, new)
    return x, new


class FusedDecoder:
    """Drop-in for `Wan2_1_VAE.decode` on the stock decoder (same object, same weights).

    compile_mode: None = eager, else a torch.compile mode (e.g. max-autotune-no-cudagraphs,
    reduce-overhead). Converts the decoder's conv weights to `dtype` (+ channels_last:
    cuDNN NHWC kernels, exp. 9) in place, so the stock loop on the same `vae` is fp32-only afterwards.
    subpixel: None = env LINGBOT_VAE_SUBPIXEL=1; the fused 2x2 weights are summed in fp32 from the
    fp32 taps, then cast once (one fp16 rounding of the pre-summed taps).
    """

    def __init__(self, vae, compile_mode=None, channels_last=True, dtype=torch.float16, subpixel=None):
        # `dtype` is the decoder's own; the vae object (and its encoder, which conditions
        # the DiT) stays at its dtype — building Wan2_1_VAE in fp16 changed the encoder
        # output and diverged the rollout from chunk 0.
        self.vae, self.dtype = vae, dtype
        m = vae.model
        self.dec, self.conv2 = m.decoder, m.conv2
        if subpixel is None:
            subpixel = os.environ.get("LINGBOT_VAE_SUBPIXEL") == "1"
        for mod in list(self.dec.modules()) + [self.conv2]:
            if subpixel and isinstance(mod, Resample) and mod.mode in ('upsample2d', 'upsample3d'):
                w, b = _subpixel_weight(mod.resample[1])
                mod.sub_weight = w.to(dtype).contiguous(memory_format=torch.channels_last if channels_last else torch.contiguous_format)
                mod.sub_bias = b.to(dtype)
            if isinstance(mod, (nn.Conv3d, nn.Conv2d)):
                mod.to(dtype)
                if channels_last:
                    mod.weight.data = mod.weight.data.contiguous(
                        memory_format=CL3D if isinstance(mod, nn.Conv3d) else torch.channels_last)
        self.cudagraphs = compile_mode == "reduce-overhead"
        self.step = _decoder_step if compile_mode is None else torch.compile(
            _decoder_step, mode=compile_mode, dynamic=False, fullgraph=True)

    @torch.no_grad()
    @torch.amp.autocast("cuda", enabled=False)  # the pipeline calls this inside its bf16 autocast
    def decode_step(self, z, state=None):
        """z: [C,T,H,W] model-space latents (T >= 1); state: None for the first latent of a clip.
        Returns ([C,F,H,W] frames in [-1,1], state) — F = 1 for the first latent, 4 per latent after."""
        mean, inv_std = self.vae.scale
        zz = z.unsqueeze(0) / inv_std.float().view(1, -1, 1, 1, 1) + mean.float().view(1, -1, 1, 1, 1)
        x = F.conv3d(zz.to(self.dtype), self.conv2.weight, self.conv2.bias)
        outs = []
        for i in range(x.shape[2]):
            if self.cudagraphs:
                torch.compiler.cudagraph_mark_step_begin()
            # clone, not contiguous(): a T=1 slice already counts as channels_last, and its
            # source-dependent strides would be a new Dynamo guard (a recompile) per call site
            y, state = self.step(self.dec, x[:, :, i:i + 1].clone(memory_format=CL3D), state)
            if self.cudagraphs:  # the graph pool reuses these buffers on the next replay
                y, state = y.clone(), [c.clone() for c in state]
            outs.append(y)
        return torch.cat(outs, 2).float().clamp_(-1, 1).squeeze(0), state

    def decode(self, z):
        return self.decode_step(z)[0]
