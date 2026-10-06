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
import logging
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from lingbot.layers import kv_cache as kvc
from lingbot.layers.attention import attention
from lingbot.models.lingbot_world import transformer as T
from lingbot.profiling import nvtx


def sinusoidal_embedding_1d(dim, position):
    # wan's version builds arange on the CPU and copies it over: a pageable H2D copy that blocks the CPU once per forward
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half, device=position.device, dtype=torch.float64).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


PIECES = int(os.environ.get("LINGBOT_SPLIT_PIECES", "4"))
# CPU-side knobs (the split is launch-bound): the compiled stages launch their kernels from generated C++ instead
# of the Python wrapper; after SKIPGUARD forwards (shapes settled), Dynamo checks only the guards that pick a graph
_OPTS = {"cpp_wrapper": True} if os.environ.get("LINGBOT_SPLIT_CPPWRAP") == "1" else None
SKIPGUARD = int(os.environ.get("LINGBOT_SPLIT_SKIPGUARD", "0"))
TOK = float(os.environ.get("LINGBOT_SPLIT_TOK", "0"))
# 1: q|k|v of both cards lands in place in one global-order buffer per card (no cat / .contiguous() around the
# exchange); the returned heads are concatenated inside the compiled _post. Eager layers only (not GRAPH).
ZEROCOPY = os.environ.get("LINGBOT_SPLIT_ZEROCOPY") == "1"
# CUDA graphs for the steady-state forwards (KV window full): "pp" the compiled stages (+ the cat feeding them),
# "1"/"all" also the attention section (KV write, SageAttention); the card-to-card exchange stays eager
GRAPH = {"1": "all", "all": "all", "pp": "pp"}.get(os.environ.get("LINGBOT_SPLIT_GRAPH", "0"))
_LN = [f"L{i}" for i in range(256)]                      # NVTX layer names (LINGBOT_NVTX=1)


def _slice_rope(rope, a, b):
    if isinstance(rope, torch.Tensor):
        return rope[:, a:b]
    return type(rope)(_slice_rope(r, a, b) for r in rope)


def _like(rope):
    if isinstance(rope, torch.Tensor):
        return torch.empty_like(rope)
    return type(rope)(_like(r) for r in rope)


def _copy_tree(dst, src):
    if isinstance(dst, torch.Tensor):
        dst.copy_(src)
    else:
        for d, s in zip(dst, src):
            _copy_tree(d, s)


class _Slot:
    __slots__ = ("p", "host", "dst", "out", "hp", "cp", "done", "free", "used")


class _Exchange:
    """Card c -> card p through pinned host memory, copy streams only, in pipelined pieces.

    Staging buffers, destinations and events are cached per (c, shape, dtype). Within an epoch (calls between two
    `release()`s, made once every received buffer has been consumed on comp[p]) each call takes a fresh slot, so
    slots are never shared by live buffers; the cursor restarts after `release()`. A reused slot is ordered on
    both sides: H2D waits `free` (recorded on comp[p] at release: consumers are done reading dst) and D2H waits
    `done` (the previous H2D finished reading the host buffer). With `dst` (a view [n, ...] of a caller buffer whose
    leading-index slices are contiguous) the H2D pieces land in it directly; the caller's consumer must run on
    comp[p] before the next `release()`, which is what orders the next layer's writes after its reads.
    """

    def __init__(self, devs, comp):
        self.devs, self.comp = devs, comp
        self.send = [torch.cuda.Stream(device=d) for d in devs]
        self.recv = [torch.cuda.Stream(device=d) for d in devs]
        self.ready = [torch.cuda.Event() for _ in devs]
        self.d2h = [torch.cuda.Event() for _ in devs]
        self.slots, self.cur, self.live, self.bounds = {}, {}, [], {}

    def _slot(self, c, p, key, src, dst=None):
        n = self.cur.get(key, 0)
        self.cur[key] = n + 1
        lst = self.slots.setdefault(key, [])
        if n < len(lst):
            return lst[n]
        s = _Slot()
        numel = src.numel()
        bd = self.bounds.get(numel)
        if bd is None:
            step = -(-numel // PIECES)
            bd = self.bounds[numel] = [(i, i + step) for i in range(0, numel, step)]
        s.p = p
        s.host = torch.empty(numel, dtype=src.dtype, pin_memory=True)
        s.hp = [s.host[i:j] for i, j in bd]
        if dst is None:
            with torch.cuda.stream(self.recv[p]):
                s.dst = torch.empty(numel, dtype=src.dtype, device=self.devs[p])
            s.out = s.dst.view(src.shape)
            s.cp = [[(h, s.dst[i:j])] for h, (i, j) in zip(s.hp, bd)]
        else:
            s.out = dst
            m = numel // dst.shape[0]
            seg = [dst[k].reshape(-1) for k in range(dst.shape[0])]
            s.cp = [[(s.host[max(i, k * m):min(j, (k + 1) * m)], seg[k][max(i, k * m) - k * m:min(j, (k + 1) * m) - k * m])
                     for k in range(i // m, (j - 1) // m + 1)] for i, j in bd]
        s.done, s.free, s.used = torch.cuda.Event(), torch.cuda.Event(), False
        lst.append(s)
        return s

    def release(self):
        for s in self.live:
            s.free.record(self.comp[s.p])
        self.live.clear()
        self.cur.clear()

    def __call__(self, c, src, dst=None):
        p = 1 - c
        flat = src.reshape(-1)
        key = (c, src.shape, src.dtype) + (() if dst is None else (dst.data_ptr(),))
        s = self._slot(c, p, key, src, dst)
        self.live.append(s)
        sc, rp = self.send[c], self.recv[p]
        self.ready[c].record(self.comp[c])
        sc.wait_event(self.ready[c])
        if s.used:
            sc.wait_event(s.done)
            rp.wait_event(s.free)
        s.used = True
        src.record_stream(sc)
        ev = self.d2h[c]
        for k, (i, j) in enumerate(self.bounds[flat.numel()]):
            with torch.cuda.stream(sc):
                s.hp[k].copy_(flat[i:j], non_blocking=True)
                ev.record(sc)
            rp.wait_event(ev)
            with torch.cuda.stream(rp):
                for h, d in s.cp[k]:
                    d.copy_(h, non_blocking=True)
        s.done.record(rp)
        return s.out, s.done


# token-local stages, compiled once per card and shape (the block is a module argument, so all 30 layers share
# a graph); SageAttention and the exchange run between them
def _pre_impl(blk, x, e0, rope):
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


_pre = torch.compile(_pre_impl, dynamic=False, options=_OPTS)


@torch.compile(dynamic=False, options=_OPTS)
def _pre_split(blk, x, e0, rope, oh, ph):
    # my heads (a view) and the peer's heads (contiguous: the send needs no copy kernel of its own)
    qkv = _pre_impl(blk, x, e0, rope)
    return qkv[:, :, oh[0]:oh[1]], qkv[:, :, ph[0]:ph[1]].contiguous()


def _post_impl(blk, x, attn, e0, cam_scale, cam_shift, ca_k, ca_v):
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


_post = torch.compile(_post_impl, dynamic=False, options=_OPTS)


@torch.compile(dynamic=False, options=_OPTS)
def _post_cat(blk, x, a0, a1, e0, cam_scale, cam_shift, ca_k, ca_v):
    return _post_impl(blk, x, torch.cat([a0, a1], dim=1), e0, cam_scale, cam_shift, ca_k, ca_v)


@torch.compile(dynamic=False, options=_OPTS)
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
        if dit_sms > 0:   # card 1 shared with the decoder: separate SM partitions
            (dit1, self.decoder_stream), self.sms = split_streams(1, dit_sms)
        else:             # decoder elsewhere (classic split, decoder after the DiT on card 0): card 1's DiT gets every SM
            dit1, self.decoder_stream = torch.cuda.Stream(device=torch.device("cuda:1"), priority=-5), None
            self.sms = (torch.cuda.get_device_properties(1).multi_processor_count, 0)
        self.comp = [torch.cuda.current_stream(self.devs[0]), dit1]
        self.xch = _Exchange(self.devs, self.comp)
        self._kv, self._kv_src, self._ca, self._F = None, None, [None, None], None
        self._gv, self._gv_warned = None, set()
        if GRAPH:
            self._pool = []
            for d in self.devs:
                with torch.cuda.device(d):
                    self._pool.append(torch.cuda.graph_pool_handle())
            # the legacy default stream (card 0's comp) cannot capture: capture on a side stream, replay on comp
            self._cap = [s if s.cuda_stream else torch.cuda.Stream(device=s.device) for s in self.comp]

    # --- caches ---------------------------------------------------------------------------------------------
    def init_crossattn_cache(self, context, crossattn_cache):
        # called once per generate(): start the per-card self-attention and camera caches fresh too
        self._kv_src = self._gv = None                      # graphs bake in the cache addresses
        self.m[0].init_crossattn_cache(context, crossattn_cache)
        self._ca[0] = crossattn_cache
        with torch.cuda.stream(self.comp[1]):
            ctx1 = [c.to(self.devs[1]) for c in context]
            self._ca[1] = [{k: (v.to(self.devs[1]) if torch.is_tensor(v) else v) for k, v in c.items()} for c in crossattn_cache]
            self.m[1].init_crossattn_cache(ctx1, self._ca[1])

    def _caches(self, kv_cache):
        # hold the pipeline's list itself (not its id: ids of freed lists are reused across generate() calls)
        if self._kv_src is kv_cache:
            return
        shape = kv_cache[0]["k"].shape                      # [B, window, heads, d], allocated by the pipeline
        L = len(kv_cache)
        kv = []
        for c in (0, 1):
            with torch.cuda.stream(self.comp[c]):           # allocated where it is used
                kv.append(kvc.allocate(L, [shape[0], shape[1], len(self.heads[c]), shape[3]], kv_cache[0]["k"].dtype,
                                       self.devs[c]))
        self._kv, self._gv = kv, None
        self._cam = [[None] * L, [None] * L]
        self._kv_src = kv_cache

    # --- forward --------------------------------------------------------------------------------------------
    def __call__(self, *a, **kw):
        self._calls = getattr(self, "_calls", 0) + 1
        with nvtx("fwd"):
            if SKIPGUARD and self._calls > SKIPGUARD:
                with torch.compiler.set_stance("default", skip_guard_eval_unsafe=True):
                    return self._forward(*a, **kw)
            return self._forward(*a, **kw)

    def _forward(self, x, t, context, seq_len, y=None, dit_cond_dict=None, kv_cache=None, crossattn_cache=None,
                 current_start=0, max_attention_size=1_000_000, frame_seqlen=None, cross_attn_first_call=None,
                 cam_cache=None, cam_first_call=None):
        self._caches(kv_cache)
        devs, comp = self.devs, self.comp
        comp[1].wait_stream(torch.cuda.current_stream(devs[1]))
        pl = dit_cond_dict["c2ws_plucker_emb"] if dit_cond_dict else None
        # inputs to card 1 through the async pinned exchange: a cross-device .to() without P2P holds the CPU
        # (~25 ms per forward measured), starving both cards of launches
        ins = [x[0], y[0], t] + (list(pl) if pl is not None else [])
        with torch.cuda.stream(comp[0]):
            sent = [self.xch(0, u.contiguous()) for u in ins]
        with torch.cuda.stream(comp[1]):
            for _, ev in sent:
                comp[1].wait_event(ev)
        r1 = [b for b, _ in sent]
        args = [([x[0]], t, [y[0]], pl), ([r1[0]], r1[2], [r1[1]], r1[3:] if pl is not None else None)]
        st = [None, None]
        for c in (0, 1):
            with torch.cuda.stream(comp[c]):
                xx, tt, yy, pp = args[c]
                st[c] = self._prelude(self.m[c], xx, tt, yy, pp, current_start, frame_seqlen, cam_first_call)
        self.xch.release()                                  # inputs consumed
        L_tok = st[0]["L"]
        # card 0's share of the tokens for the token-local work (projections, FFN): the head ratio by default;
        # LINGBOT_SPLIT_TOK=f moves token-local work between cards independently of the head split
        n0 = int(L_tok * TOK) if TOK else L_tok * self.h0 // self.m[0].num_heads
        tok = [range(0, n0), range(n0, L_tok)]
        xs = []
        for c in (0, 1):
            with torch.cuda.stream(comp[c]):
                s = st[c]
                xs.append(s["x"][:, tok[c].start:tok[c].stop])
                s["rope"] = _slice_rope(s["rope"], tok[c].start, tok[c].stop)
                s["pl"] = None if s["pl"] is None else s["pl"][:, tok[c].start:tok[c].stop]
        frame_seqlen = st[0]["frame_seqlen"]
        V = self._graph_variant(xs, st, current_start, max_attention_size, frame_seqlen) if GRAPH else None
        if V is not None:
            xs = self._layers_graph(V, xs, st, tok, current_start, max_attention_size, frame_seqlen, cam_first_call)
        else:
            for li in range(len(self.m[0].blocks)):
                with nvtx(_LN[li]):
                    xs = self._layer(li, xs, st, tok, current_start, max_attention_size, frame_seqlen, cam_first_call)
        outs = []
        for c in (0, 1):
            with torch.cuda.stream(comp[c]):
                outs.append(self.m[c].head(xs[c], st[c]["e"]))
        buf, ev = self.xch(1, outs[1].contiguous())
        with torch.cuda.stream(comp[0]):
            comp[0].wait_event(ev)
            full = torch.cat([outs[0], buf], dim=1)
            self.xch.release()
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
        qkv, inbox, dst = [None, None], [None, None], None
        for c in (0, 1):                                   # projections; send the peer's heads of my tokens
            p, ph, oh, tc = 1 - c, self.heads[1 - c], self.heads[c], tok[c]
            with torch.cuda.stream(comp[c]), nvtx("pre"):
                if ZEROCOPY:
                    qkv[c], pk = _pre_split(self.m[c].blocks[li], xs[c], st[c]["e0"], st[c]["rope"],
                                            (oh.start, oh.stop), (ph.start, ph.stop))
                    F = self._full(qkv[c], tok[1].stop)
                    F[c][:, tc.start:tc.stop].copy_(qkv[c])
                    dst = F[p][:, tc.start:tc.stop]
                else:
                    qkv[c] = _pre(self.m[c].blocks[li], xs[c], st[c]["e0"], st[c]["rope"])
                    pk = qkv[c][:, :, ph.start:ph.stop].contiguous()
            with nvtx("send_qkv"):
                inbox[p] = self.xch(c, pk, dst)
        attn_own, back = [None, None], [None, None]
        for c in (0, 1):                                   # my heads over all tokens: KV cache, SageAttention
            oh = self.heads[c]
            with torch.cuda.stream(comp[c]), nvtx("attn"):
                buf, ev = inbox[c]
                comp[c].wait_event(ev)
                attn_own[c], back[c] = self._attend(c, li, qkv[c] if ZEROCOPY else qkv[c][:, :, oh.start:oh.stop], buf,
                                                    tok, current_start, max_attention_size, frame_seqlen,
                                                    full=self._F[c] if ZEROCOPY else None)
        self.xch.release()                                 # inbox consumed
        recv = [None, None]
        for c in (0, 1):
            with nvtx("send_back"):
                recv[1 - c] = self.xch(c, back[c])
        out = [None, None]
        for c in (0, 1):                                   # assemble all heads of my tokens; o proj, cam, cross, FFN
            blk = self.m[c].blocks[li]
            with torch.cuda.stream(comp[c]), nvtx("post"):
                buf, ev = recv[c]
                comp[c].wait_event(ev)
                parts = [attn_own[c], buf] if c == 0 else [buf, attn_own[c]]   # heads 0..h0-1, then h0..H-1
                if cam_first_call and st[c]["pl"] is not None:
                    self._cam[c][li] = _cam(blk, st[c]["pl"])
                cs, csh = self._cam[c][li]
                ca = self._ca[c][li]
                if ZEROCOPY:                                # [my tokens, H, d] assembled inside the compiled stage
                    out[c] = _post_cat(blk, xs[c], *parts, st[c]["e0"], cs, csh, ca["k"], ca["v"])
                else:
                    out[c] = _post(blk, xs[c], torch.cat(parts, dim=1), st[c]["e0"], cs, csh, ca["k"], ca["v"])
        self.xch.release()                                 # recv consumed
        return out

    def _full(self, own, n):
        # per card [3, all tokens, my heads, d]: q|k|v in global token order, filled in place (mine and the peer's)
        sh = [(3, n, len(h), own.shape[-1]) for h in self.heads]
        if self._F is None or [tuple(f.shape) for f in self._F] != sh or self._F[0].dtype != own.dtype:
            self._F = []
            for c in (0, 1):
                with torch.cuda.stream(self.comp[c]):
                    self._F.append(torch.empty(sh[c], dtype=own.dtype, device=self.devs[c]))
        return self._F

    def _attend(self, c, li, own, buf, tok, current_start, max_attention_size, frame_seqlen, fixed=None, full=None):
        # `fixed` = (start, end) of the K/V write in the cache (steady state, graph path: eviction already done)
        sa = self.m[c].blocks[li].self_attn
        if full is None:
            full = torch.cat([own, buf], dim=1) if c == 0 else torch.cat([buf, own], dim=1)  # global token order
        q, k, v = (full[i].unsqueeze(0) for i in range(3))
        kc = self._kv[c][li]
        if fixed is None:
            end, cur_end = kvc.write(kc, k, v, current_start, sa.sink_size * frame_seqlen, sa.local_attn_size)
        else:
            start, end = fixed
            kc["k"][:, start:end] = k
            kc["v"][:, start:end] = v
        k_win, v_win = kvc.window(kc, end, max_attention_size)
        a = attention(q, k_win, v_win)[0]                  # [L, my heads, d]
        if fixed is None:
            kvc.commit(kc, cur_end, end)
        return a[tok[c].start:tok[c].stop], a[tok[1 - c].start:tok[1 - c].stop].contiguous()

    # --- CUDA graphs (LINGBOT_SPLIT_GRAPH) --------------------------------------------------------------------
    def _graph_variant(self, xs, st, current_start, max_attention_size, frame_seqlen):
        """The static buffers + graph cache when this forward is steady state (window full, so every KV offset is
        fixed), else None (growing window, other shapes): eager."""
        sa = self.m[0].blocks[0].self_attn
        kc = self._kv[0][0]
        if sa.local_attn_size == -1:
            return None
        if kvc.plan(kc, st[0]["L"], current_start, sa.sink_size * frame_seqlen, sa.local_attn_size)[3] \
                != kc["k"].shape[1]:
            return None
        key = (xs[0].shape, xs[1].shape, st[0]["e0"].shape, st[0]["e0"].dtype, max_attention_size, frame_seqlen)
        if self._gv is None:
            n = len(self.m[0].blocks)
            self._gv = SimpleNamespace(key=key, g=[{}, {}], x=[None, None], e0=[None, None], rope=[None, None],
                                       pl=[None, None], attn=[None, None], cam=[[None] * n, [None] * n])
        if self._gv.key != key:
            if key not in self._gv_warned:
                self._gv_warned.add(key)
                logging.warning(f"SplitDiT graphs: steady forward with new shapes {key}, running eager")
            return None
        return self._gv

    def _graph(self, V, c, key, fn):
        e = V.g[c].get(key)
        if e is None:
            fn()                                           # warm up (compile, autotune, workspaces) before capture
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._pool[c], stream=self._cap[c], capture_error_mode="thread_local"):
                out = fn()
            e = V.g[c][key] = (g, out)
        e[0].replay()
        return e[1]

    def _layers_graph(self, V, xs, st, tok, current_start, max_attention_size, frame_seqlen, cam_first_call):
        # Per card and layer li two graphs, exchanges and event waits eager between them:
        #   P(li): cat(attn heads) + _post(li-1) + _pre(li) + the peer's q|k|v slice (P(0): _pre only; P(L): _post only)
        #   A(li): cat(own, peer heads) + KV write at fixed offsets + window + SageAttention + token slices ("all")
        # Inputs that change per forward live in static buffers refreshed here; graph outputs stay in the pool.
        comp, alls = self.comp, GRAPH == "all"
        L, L_tok = len(self.m[0].blocks), st[0]["L"]
        for c in (0, 1):
            with torch.cuda.stream(comp[c]):
                s = st[c]
                if V.x[c] is None:
                    V.x[c], V.e0[c], V.rope[c] = torch.empty_like(xs[c]), torch.empty_like(s["e0"]), _like(s["rope"])
                V.x[c].copy_(xs[c])
                V.e0[c].copy_(s["e0"])
                _copy_tree(V.rope[c], s["rope"])
                for li in range(L):                        # camera modulation: eager results (other forwards) -> static
                    cam, vc = self._cam[c][li], V.cam[c][li]
                    if cam is not vc:
                        if vc is None:
                            vc = V.cam[c][li] = tuple(t.clone() for t in cam)
                        else:
                            vc[0].copy_(cam[0])
                            vc[1].copy_(cam[1])
                        self._cam[c][li] = vc
                if cam_first_call and s["pl"] is not None:
                    if V.pl[c] is None:
                        V.pl[c] = torch.empty_like(s["pl"])
                    V.pl[c].copy_(s["pl"])
                    for li in range(L):
                        blk, vc = self.m[c].blocks[li], V.cam[c][li]

                        def cam_fn(blk=blk, vc=vc, pl=V.pl[c]):
                            cs, csh = _cam(blk, pl)
                            vc[0].copy_(cs)
                            vc[1].copy_(csh)
                        self._graph(V, c, ("C", li), cam_fn)
        xin, pend = list(V.x), [None, None]                # pend[c]: (attn_own, recv buf) of the layer before
        for li in range(L + 1):
            outs = [None, None]
            for c in (0, 1):                               # [cat +] _post(li-1) + _pre(li)
                blks, ca_, vc = self.m[c].blocks, self._ca[c], V.cam[c]
                with torch.cuda.stream(comp[c]):
                    attn, parts, ptr = None, None, 0
                    if li:
                        attn_own, (buf, ev) = pend[c]
                        comp[c].wait_event(ev)
                        parts = [attn_own, buf] if c == 0 else [buf, attn_own]   # heads 0..h0-1, then h0..H-1
                        if alls:
                            ptr = buf.data_ptr()
                        elif V.attn[c] is None:
                            V.attn[c] = attn = torch.cat(parts, dim=1)
                        else:
                            attn = torch.cat(parts, dim=1, out=V.attn[c])

                    def pp_fn(c=c, li=li, x=xin[c], attn=attn, parts=parts):
                        if li:
                            a = attn if attn is not None else torch.cat(parts, dim=1)
                            ca = ca_[li - 1]
                            x = _post(blks[li - 1], x, a, V.e0[c], vc[li - 1][0], vc[li - 1][1], ca["k"], ca["v"])
                        if li == L:
                            return (x,)
                        qkv = _pre(blks[li], x, V.e0[c], V.rope[c])
                        ph = self.heads[1 - c]
                        return x, qkv, qkv[:, :, ph.start:ph.stop].contiguous()
                    outs[c] = self._graph(V, c, ("P", li, ptr), pp_fn)
            if li:
                self.xch.release()                         # recv consumed
            xin = [o[0] for o in outs]
            if li == L:
                return xin
            inbox = [None, None]
            for c in (0, 1):                               # send the peer's heads of my tokens
                inbox[1 - c] = self.xch(c, outs[c][2])
            attn_own, back = [None, None], [None, None]
            for c in (0, 1):                               # my heads over all tokens: KV cache, SageAttention
                oh, kc = self.heads[c], self._kv[c][li]
                sa = self.m[c].blocks[li].self_attn
                with torch.cuda.stream(comp[c]):
                    buf, ev = inbox[c]
                    comp[c].wait_event(ev)
                    own = outs[c][1][:, :, oh.start:oh.stop]
                    if not alls:
                        attn_own[c], back[c] = self._attend(c, li, own, buf, tok, current_start,
                                                            max_attention_size, frame_seqlen)
                        continue
                    sink = sa.sink_size * frame_seqlen
                    evicted, rolled, start, end, cur_end = kvc.plan(kc, L_tok, current_start, sink, sa.local_attn_size)
                    if evicted:                            # 1 forward in 5: the window shifts, eagerly
                        kvc.evict(kc, sink, evicted, rolled)

                    def attn_fn(c=c, li=li, own=own, buf=buf, fixed=(start, end)):
                        return self._attend(c, li, own, buf, tok, current_start, max_attention_size, frame_seqlen,
                                            fixed)
                    attn_own[c], back[c] = self._graph(V, c, ("A", li, start, end, buf.data_ptr()), attn_fn)
                    kc["global_end_int"], kc["local_end_int"] = cur_end, end   # the tensor indices go stale; unread
            self.xch.release()                             # inbox consumed
            recv = [None, None]
            for c in (0, 1):
                recv[1 - c] = self.xch(c, back[c])
            pend = [(attn_own[c], recv[c]) for c in (0, 1)]
