"""The DiT split across two GPUs (`LINGBOT_SPLIT=10:2`): Ulysses-style sequence parallelism with uneven shares.

Card 0 holds the first h0/12 of the tokens and heads 0..h0-1, card 1 the rest. Each card runs the token-local
work (norms, projections, camera conditioning, cross-attention, FFN) on its own tokens and self-attention for
its own heads over all tokens, with its own KV cache for those heads. Per layer: one packed q|k|v message each
way (the peer's heads of my tokens) and one message back (my heads of the peer's tokens), through pinned host
memory in pipelined pieces (no P2P on GeForce). Card 1's DiT share runs on a green-context partition of its SMs
(`LINGBOT_SPLIT_SMS`, default 40) at high priority; the decoder gets the rest (`decoder_stream`).

Same computation as one card up to floating-point order (2X_RTX5090_LEARNINGS.md, learnings 11-15). Needs the
SageAttention build that launches on the current stream (patches/sageattention-current-stream.patch).
"""
import copy
import os

import torch
import torch.nn.functional as F

from lingbot.layers import kv_cache as kvc
from lingbot.layers.attention import attention
from lingbot.models.lingbot_world import transformer as T
from wan.modules.model import sinusoidal_embedding_1d

PIECES = int(os.environ.get("LINGBOT_SPLIT_PIECES", "4"))


def _slice_rope(rope, a, b):
    if isinstance(rope, torch.Tensor):
        return rope[:, a:b]
    return type(rope)(_slice_rope(r, a, b) for r in rope)


class _Exchange:
    """Card c -> card p through pinned host memory, copy streams only, in pipelined pieces."""

    def __init__(self, devs, comp):
        self.devs, self.comp = devs, comp
        self.send = [torch.cuda.Stream(device=d) for d in devs]
        self.recv = [torch.cuda.Stream(device=d) for d in devs]

    def __call__(self, c, src):
        p = 1 - c
        flat = src.reshape(-1)
        host = torch.empty(flat.shape, dtype=src.dtype, pin_memory=True)
        ready = torch.cuda.Event()
        ready.record(self.comp[c])
        self.send[c].wait_event(ready)
        src.record_stream(self.send[c])
        with torch.cuda.stream(self.recv[p]):
            dst = torch.empty(flat.shape, dtype=src.dtype, device=self.devs[p])
        step = -(-flat.numel() // PIECES)
        for i in range(0, flat.numel(), step):
            with torch.cuda.stream(self.send[c]):
                host[i:i + step].copy_(flat[i:i + step], non_blocking=True)
                d2h = torch.cuda.Event()
                d2h.record(self.send[c])
            self.recv[p].wait_event(d2h)
            with torch.cuda.stream(self.recv[p]):
                dst[i:i + step].copy_(host[i:i + step], non_blocking=True)
        done = torch.cuda.Event()
        done.record(self.recv[p])
        dst.record_stream(self.comp[p])
        return dst.view(src.shape), done


# token-local stages, compiled once per card and shape (the block is a module argument, so all 30 layers share
# a graph); SageAttention and the exchange run between them
@torch.compile(dynamic=False)
def _pre(blk, x, e0, rope):
    with torch.amp.autocast('cuda', dtype=torch.float32):
        e = (blk.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
    sa = blk.self_attn
    xin = blk.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
    b, s, n, d = x.size(0), x.size(1), sa.num_heads, sa.head_dim
    q = sa.norm_q(sa.q(xin)).view(b, s, n, d)
    k = sa.norm_k(sa.k(xin)).view(b, s, n, d)
    v = sa.v(xin).view(b, s, n, d)
    q = T.causal_rope_apply(q, rope).type_as(v)
    k = T.causal_rope_apply(k, rope).type_as(v)
    return torch.stack([q[0], k[0], v[0]])            # [3, tokens, heads, d]


@torch.compile(dynamic=False)
def _post(blk, x, attn, e0, cam_scale, cam_shift, ca_k, ca_v):
    with torch.amp.autocast('cuda', dtype=torch.float32):
        e = (blk.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
    y = blk.self_attn.o(attn.flatten(-2).unsqueeze(0))
    with torch.amp.autocast('cuda', dtype=torch.float32):
        x = x + y * e[2].squeeze(2)
    x = (1.0 + cam_scale) * x + cam_shift
    ca = blk.cross_attn
    b, n, d = x.size(0), ca.num_heads, ca.head_dim
    q = ca.norm_q(ca.q(blk.norm3(x))).view(b, -1, n, d)
    c = T.flash_attention(q, ca_k, ca_v, k_lens=None)
    x = x + ca.o(c.flatten(2))
    yf = blk.ffn(blk.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
    with torch.amp.autocast('cuda', dtype=torch.float32):
        x = x + yf * e[5].squeeze(2)
    return x


@torch.compile(dynamic=False)
def _cam(blk, plucker):
    h = blk.cam_injector_layer2(F.silu(blk.cam_injector_layer1(plucker))) + plucker
    return blk.cam_scale_layer(h), blk.cam_shift_layer(h)


class SplitDiT:
    """Drop-in for WanModelFast in the pipeline: same forward signature, returns card 0 tensors."""

    def __init__(self, model, h0=10, dit_sms=40):
        from lingbot.parallel.greenctx import split_streams
        self.devs = [torch.device("cuda:0"), torch.device("cuda:1")]
        self.m = [model, copy.deepcopy(model).to(self.devs[1])]
        self.config = model.config
        H = model.num_heads
        self.heads = [range(0, h0), range(h0, H)]
        self.h0 = h0
        (dit1, self.decoder_stream), self.sms = split_streams(1, dit_sms)
        self.comp = [torch.cuda.current_stream(self.devs[0]), dit1]
        self.xch = _Exchange(self.devs, self.comp)
        self._kv, self._kv_key, self._ca = None, None, [None, None]

    # --- caches ---------------------------------------------------------------------------------------------
    def init_crossattn_cache(self, context, crossattn_cache):
        self.m[0].init_crossattn_cache(context, crossattn_cache)
        self._ca[0] = crossattn_cache
        with torch.cuda.stream(self.comp[1]):
            ctx1 = [c.to(self.devs[1]) for c in context]
            self._ca[1] = [{k: (v.to(self.devs[1]) if torch.is_tensor(v) else v) for k, v in c.items()} for c in crossattn_cache]
            self.m[1].init_crossattn_cache(ctx1, self._ca[1])

    def _caches(self, kv_cache):
        if self._kv_key == id(kv_cache):
            return
        shape = kv_cache[0]["k"].shape                      # [B, window, heads, d], allocated by the pipeline
        L = len(kv_cache)
        self._kv = [kvc.allocate(L, [shape[0], shape[1], len(self.heads[c]), shape[3]], kv_cache[0]["k"].dtype,
                                 self.devs[c]) for c in (0, 1)]
        self._cam = [[None] * L, [None] * L]
        self._kv_key = id(kv_cache)

    # --- forward --------------------------------------------------------------------------------------------
    def __call__(self, x, t, context, seq_len, y=None, dit_cond_dict=None, kv_cache=None, crossattn_cache=None,
                 current_start=0, max_attention_size=1_000_000, frame_seqlen=None, cross_attn_first_call=None,
                 cam_cache=None, cam_first_call=None):
        self._caches(kv_cache)
        devs, comp = self.devs, self.comp
        comp[1].wait_stream(torch.cuda.current_stream(devs[1]))
        pl = dit_cond_dict["c2ws_plucker_emb"] if dit_cond_dict else None
        st = [None, None]
        for c in (0, 1):
            with torch.cuda.stream(comp[c]):
                mv = (lambda a: a.to(devs[c], non_blocking=True)) if c else (lambda a: a)
                st[c] = self._prelude(self.m[c], [mv(u) for u in x], mv(t), [mv(u) for u in y],
                                      [mv(u) for u in pl] if pl is not None else None, current_start, frame_seqlen,
                                      cam_first_call)
        L_tok = st[0]["L"]
        n0 = L_tok * self.h0 // self.m[0].num_heads          # card 0's share of the tokens, same ratio as heads
        tok = [range(0, n0), range(n0, L_tok)]
        xs = []
        for c in (0, 1):
            with torch.cuda.stream(comp[c]):
                s = st[c]
                xs.append(s["x"][:, tok[c].start:tok[c].stop])
                s["rope"] = _slice_rope(s["rope"], tok[c].start, tok[c].stop)
                s["pl"] = None if s["pl"] is None else s["pl"][:, tok[c].start:tok[c].stop]
        frame_seqlen = st[0]["frame_seqlen"]
        for li in range(len(self.m[0].blocks)):
            xs = self._layer(li, xs, st, tok, current_start, max_attention_size, frame_seqlen, cam_first_call)
        outs = []
        for c in (0, 1):
            with torch.cuda.stream(comp[c]):
                outs.append(self.m[c].head(xs[c], st[c]["e"]))
        buf, ev = self.xch(1, outs[1].contiguous())
        with torch.cuda.stream(comp[0]):
            comp[0].wait_event(ev)
            full = torch.cat([outs[0], buf], dim=1)
            return [u.float() for u in self.m[0].unpatchify(full, st[0]["grid"])]

    def _prelude(self, m, x, t, y, pl, current_start, frame_seqlen, cam_first_call):
        if m.freqs.device != m.patch_embedding.weight.device:
            m.freqs = m.freqs.to(m.patch_embedding.weight.device)
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        x = [m.patch_embedding(u.unsqueeze(0)) for u in x]
        grid = [tuple(u.shape[2:]) for u in x]
        x = torch.cat([u.flatten(2).transpose(1, 2) for u in x])
        tt = t.unsqueeze(1) if t.dim() == 1 else t
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt, lt = tt.shape
            e = m.time_embedding(sinusoidal_embedding_1d(m.freq_dim, tt.flatten()).unflatten(0, (bt, lt)).float())
            e0 = m.time_projection(e).unflatten(2, (6, m.dim))
        if frame_seqlen is None:
            frame_seqlen = grid[0][1] * grid[0][2]
        rope = T.causal_rope_freqs(grid, m.freqs, start_frame=current_start // frame_seqlen)
        if T._ROPE_FP32C:
            rope = T.rope_fp32c_tables(rope)
        plk = None
        if pl is not None and cam_first_call:
            ps = m.patch_size
            plk = torch.cat([T.rearrange(i, '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)', c1=ps[0], c2=ps[1],
                                         c3=ps[2]) for i in pl], dim=1)
            plk = m.patch_embedding_wancamctrl(plk)
            plk = plk + m.c2ws_hidden_states_layer2(F.silu(m.c2ws_hidden_states_layer1(plk)))
        return dict(x=x, e=e, e0=e0, rope=rope, grid=grid, L=x.size(1), pl=plk, frame_seqlen=frame_seqlen)

    def _layer(self, li, xs, st, tok, current_start, max_attention_size, frame_seqlen, cam_first_call):
        devs, comp, H = self.devs, self.comp, self.m[0].num_heads
        qkv, inbox = [None, None], [None, None]
        for c in (0, 1):                                   # projections; send the peer's heads of my tokens
            p, ph = 1 - c, self.heads[1 - c]
            with torch.cuda.stream(comp[c]):
                qkv[c] = _pre(self.m[c].blocks[li], xs[c], st[c]["e0"], st[c]["rope"])
                pk = qkv[c][:, :, ph.start:ph.stop].contiguous()
            inbox[p] = self.xch(c, pk)
        attn_own, back = [None, None], [None, None]
        for c in (0, 1):                                   # my heads over all tokens: KV cache, SageAttention
            oh, blk = self.heads[c], self.m[c].blocks[li]
            sa = blk.self_attn
            with torch.cuda.stream(comp[c]):
                own = qkv[c][:, :, oh.start:oh.stop]
                buf, ev = inbox[c]
                comp[c].wait_event(ev)
                full = torch.cat([own, buf], dim=1) if c == 0 else torch.cat([buf, own], dim=1)  # global token order
                q, k, v = (full[i].unsqueeze(0) for i in range(3))
                kc = self._kv[c][li]
                end, cur_end = kvc.write(kc, k, v, current_start, sa.sink_size * frame_seqlen, sa.local_attn_size)
                k_win, v_win = kvc.window(kc, end, max_attention_size)
                a = attention(q, k_win, v_win)[0]          # [L, my heads, d]
                kvc.commit(kc, cur_end, end)
                mine = a[tok[c].start:tok[c].stop]
                back[c] = a[tok[1 - c].start:tok[1 - c].stop].contiguous()
                attn_own[c] = mine
        recv = [None, None]
        for c in (0, 1):
            recv[1 - c] = self.xch(c, back[c])
        out = [None, None]
        for c in (0, 1):                                   # assemble all heads of my tokens; o proj, cam, cross, FFN
            blk = self.m[c].blocks[li]
            with torch.cuda.stream(comp[c]):
                buf, ev = recv[c]
                comp[c].wait_event(ev)
                parts = [attn_own[c], buf] if c == 0 else [buf, attn_own[c]]   # heads 0..h0-1, then h0..H-1
                attn = torch.cat(parts, dim=1)              # [my tokens, H, d]
                if cam_first_call and st[c]["pl"] is not None:
                    self._cam[c][li] = _cam(blk, st[c]["pl"])
                cs, csh = self._cam[c][li]
                ca = self._ca[c][li]
                out[c] = _post(blk, xs[c], attn, st[c]["e0"], cs, csh, ca["k"], ca["v"])
        return out
