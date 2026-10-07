"""`SplitDiT` with one process per card (`LINGBOT_SPLIT=10:2 LINGBOT_SPLIT_MP=1`): the same layer schedule and math,
but each card's kernels are launched by its own Python thread.

The main process owns the pipeline, the decoder and card 0's share; a helper process (spawned once with a copy of the
weights) owns cuda:1's share. Per forward the main process sends a small tuple over a pipe and its inputs through
IPC-shared device buffers; per layer the q|k|v head exchange and the attention-output return go through
preallocated IPC-shared device buffers (`_Chan`). The consumer pulls the producer's buffer into its own card with a
device-to-device copy (no P2P: tools/linkbench/ipc_a2a.py). Ordering between the processes is CUDA events
(`interprocess=True`), waited on by streams, never by the CPU; the one CPU-side coupling is a shared counter each
side bumps after enqueueing a record, because a stream wait captures the event's latest record at call time.

SM partitions (green contexts) exist inside one process only. Two processes on one card time-slice (unless MPS), so
the helper's DiT runs on a plain high-priority stream and does not share card 1 concurrently with a decoder in the
main process (`LINGBOT_SPLIT_MP_PCT` caps the helper's SMs when MPS is running).
"""
import atexit
import logging
import math
import os
import queue
import time
import traceback

import torch
import torch.multiprocessing as mp

from lingbot.layers import kv_cache as kvc
from lingbot.layers.attention import attention
from lingbot.parallel.split_dit import SplitDiT, _cam, _post, _pre, _slice_rope

TIMEOUT = 900.0   # first forwards compile in both processes


def _ac():
    return torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else None


class _Chan:
    """One direction, a ring of R slots. Producer: box[s] (own card, shared) <- src, `ev[s]` = ready. Consumer:
    pulls the peer's box[s] into its own box[s], `ev[s]` = consumed, so the producer may reuse the slot.
    sync[i] counts messages sent, sync[i+1] messages consumed."""

    def __init__(self, link, i, R, prod, box, ev, peer_ev, peer_box):
        self.link, self.i, self.R, self.prod = link, i, R, prod
        self.box, self.ev, self.peer_ev, self.peer_box = box, ev, peer_ev, peer_box
        self.n, self.last = 0, 0
        self.local = torch.cuda.Event()
        self.arrived = [torch.cuda.Event() for _ in range(R)]
        self.freed = [torch.cuda.Event() for _ in range(R)]

    def send(self, src, comp):
        L, n = self.link, self.n
        self.n += 1
        s = n % self.R
        nb = src.numel() * src.element_size()
        assert src.is_contiguous() and nb <= self.box[s].numel(), "message does not fit its channel"
        ss = L.send_s
        if n >= self.R:
            L.wait(self.i + 1, n - self.R + 1)             # the peer's CPU has recorded "consumed" for message n-R
            ss.wait_event(self.peer_ev[s])
        self.local.record(comp)
        ss.wait_event(self.local)
        src.record_stream(ss)
        with torch.cuda.stream(ss):
            self.box[s][:nb].view(src.dtype).view(src.shape).copy_(src)
            self.ev[s].record(ss)
        L.sync[self.i] = n + 1

    def recv(self, shape, dtype, comp):
        L, n = self.link, self.n
        self.n += 1
        s = self.last = n % self.R
        nb = math.prod(shape) * torch.empty(0, dtype=dtype).element_size()
        L.wait(self.i, n + 1)                              # the peer's CPU has recorded "ready" for message n
        rs = L.recv_s
        if n >= self.R:
            rs.wait_event(self.freed[s])                   # my consumers are done with the slot's previous message
        rs.wait_event(self.peer_ev[s])
        dst = self.box[s][:nb].view(dtype).view(shape)
        src = self.peer_box[s][:nb].view(dtype).view(shape)
        with torch.cuda.stream(rs), torch.cuda.stream(L.peer_s):
            dst.copy_(src)                                 # runs on the peer card's stream, ordered with rs
            self.ev[s].record(rs)
            self.arrived[s].record(rs)
        L.sync[self.i + 1] = n + 1
        comp.wait_event(self.arrived[s])
        return dst

    def free(self, comp):
        self.freed[self.last].record(comp)


class _Link:
    def __init__(self, rank, q_out, q_in, sync, check):
        self.rank = rank
        self.dev, self.pdev = torch.device(f"cuda:{rank}"), torch.device(f"cuda:{1 - rank}")
        self.q_out, self.q_in, self.sync, self.check = q_out, q_in, sync.numpy(), check
        self.send_s, self.recv_s = torch.cuda.Stream(device=self.dev), torch.cuda.Stream(device=self.dev)
        self.peer_s = torch.cuda.Stream(device=self.pdev)
        self.ch, self.next = {}, 0

    def wait(self, i, v):
        t0 = time.monotonic()
        while self.sync[i] < v:
            time.sleep(0)
            if time.monotonic() - t0 > 1.0:
                self.check()
                if time.monotonic() - t0 > TIMEOUT:
                    raise TimeoutError("peer process did not reach the sync point")

    def _get(self):
        t0 = time.monotonic()
        while True:
            try:
                return self.q_in.get(timeout=1.0)
            except queue.Empty:
                self.check()
                if time.monotonic() - t0 > TIMEOUT:
                    raise TimeoutError("peer process did not answer the channel setup")

    def open(self, specs):
        """specs: [(name, producer rank, bytes, ring slots)]; both processes call it with the same list. Handles
        (tensors, events) cross once through the queues; counters come from the shared array in spec order."""
        mine, bundle = [], {}
        for name, prod, nbytes, R in specs:
            i, self.next = self.next, self.next + 2
            with torch.cuda.device(self.dev):
                box = [torch.empty(nbytes, dtype=torch.uint8, device=self.dev) for _ in range(R)]
                ev = [torch.cuda.Event(interprocess=True) for _ in range(R)]   # ready (producer) / consumed (consumer)
                bundle[name] = (box if prod == self.rank else None, [e.ipc_handle() for e in ev])
            mine.append((name, i, R, prod, box, ev))
        self.q_out.put(bundle)
        theirs = self._get()
        for name, i, R, prod, box, ev in mine:
            peer_box, hs = theirs[name]
            peer_ev = [torch.cuda.Event.from_ipc_handle(self.dev, h) for h in hs]
            self.ch[name] = _Chan(self, i, R, prod, box, ev, peer_ev, peer_box)


class _Card:
    """One card's share of every layer: tokens `tok[c]`, heads `heads[c]`, its own KV / camera / cross-attn caches."""

    def __init__(self, c, m, link, h0):
        self.c, self.m, self.link, self.h0 = c, m, link, h0
        H = m.num_heads
        self.heads = [range(0, h0), range(h0, H)]
        self.kv = self.cam = self.ca = None

    def set_kv(self, shape, dtype, L):
        self.kv = kvc.allocate(L, [shape[0], shape[1], len(self.heads[self.c]), shape[3]], dtype, self.link.dev)
        self.cam = [None] * L

    def run(self, x, t, y, pl, current_start, max_attention_size, frame_seqlen, cam_first_call, comp):
        c, p, m, ch = self.c, 1 - self.c, self.m, self.link.ch
        st = SplitDiT._prelude(None, m, [x], t, [y], pl, current_start, frame_seqlen, cam_first_call)
        L_tok = st["L"]
        n0 = L_tok * self.h0 // m.num_heads                  # card 0's share of the tokens, same ratio as heads
        tok = [range(0, n0), range(n0, L_tok)]
        d, oh, ph = m.dim // m.num_heads, self.heads[c], self.heads[p]
        xs = st["x"][:, tok[c].start:tok[c].stop]
        rope = _slice_rope(st["rope"], tok[c].start, tok[c].stop)
        plk = None if st["pl"] is None else st["pl"][:, tok[c].start:tok[c].stop]
        for li, blk in enumerate(m.blocks):
            qkv = _pre(blk, xs, st["e0"], rope)
            pk = qkv[:, :, ph.start:ph.stop].contiguous()
            ch[f"qkv{c}"].send(pk, comp)                    # the peer's heads of my tokens
            buf = ch[f"qkv{p}"].recv((3, len(tok[p]), len(oh), d), pk.dtype, comp)
            attn_own, back = self._attend(li, qkv[:, :, oh.start:oh.stop], buf, tok, current_start,
                                          max_attention_size, frame_seqlen)
            ch[f"qkv{p}"].free(comp)
            ch[f"back{c}"].send(back, comp)                 # my heads of the peer's tokens
            buf = ch[f"back{p}"].recv((len(tok[c]), len(ph), d), back.dtype, comp)
            parts = [attn_own, buf] if c == 0 else [buf, attn_own]   # heads 0..h0-1, then h0..H-1
            attn = torch.cat(parts, dim=1)                  # [my tokens, H, d]
            ch[f"back{p}"].free(comp)
            if cam_first_call and plk is not None:
                self.cam[li] = _cam(blk, plk)
            cs, csh = self.cam[li]
            ca = self.ca[li]
            xs = _post(blk, xs, attn, st["e0"], cs, csh, ca["k"], ca["v"])
        return m.head(xs, st["e"]), st

    def _attend(self, li, own, buf, tok, current_start, max_attention_size, frame_seqlen):
        c = self.c
        sa = self.m.blocks[li].self_attn
        full = torch.cat([own, buf], dim=1) if c == 0 else torch.cat([buf, own], dim=1)  # global token order
        q, k, v = (full[i].unsqueeze(0) for i in range(3))
        kc = self.kv[li]
        end, cur_end = kvc.write(kc, k, v, current_start, sa.sink_size * frame_seqlen, sa.local_attn_size)
        k_win, v_win = kvc.window(kc, end, max_attention_size)
        a = attention(q, k_win, v_win)[0]                   # [L, my heads, d]
        kvc.commit(kc, cur_end, end)
        return a[tok[c].start:tok[c].stop], a[tok[1 - c].start:tok[1 - c].stop].contiguous()


def _amp(ac):
    return torch.amp.autocast("cuda", dtype=ac, enabled=ac is not None)


def _helper(model, h0, conn, q_out, q_in, sync, ppid):
    try:
        torch.cuda.set_device(1)
        dev = torch.device("cuda:1")
        m = model.to(dev).eval().requires_grad_(False)
        del model
        comp = torch.cuda.Stream(device=dev, priority=-5)

        def check():
            if os.getppid() != ppid:
                raise SystemExit("main process is gone")

        link = _Link(1, q_out, q_in, sync, check)
        card = _Card(1, m, link, h0)
        conn.send(("ready",))
        while True:
            try:
                cmd, msg = conn.recv()
            except EOFError:
                return
            if cmd == "stop":
                return
            if cmd == "open":
                link.open(msg)
                continue
            with torch.cuda.stream(comp), torch.no_grad(), _amp(msg["ac"]):
                if cmd == "ca":
                    ctx = [link.ch[f"ctx{k}"].recv(s, dt, comp) for k, (s, dt) in enumerate(msg["ctx"])]
                    (shape, dt), L = msg["ca"], msg["L"]
                    card.ca = [{"k": torch.zeros(shape, dtype=dt, device=dev), "v": torch.zeros(shape, dtype=dt, device=dev),
                                "is_init": torch.tensor(0, dtype=torch.int32, device=dev)} for _ in range(L)]
                    m.init_crossattn_cache(ctx, card.ca)
                    for k in range(len(ctx)):
                        link.ch[f"ctx{k}"].free(comp)
                elif cmd == "fwd":
                    if msg["kv"]:
                        card.set_kv(*msg["kv"])
                    ins = [link.ch[f"in{k}"].recv(s, dt, comp) for k, (s, dt) in enumerate(msg["ins"])]
                    pl = ins[3:] or None
                    out, _ = card.run(ins[0], ins[2], ins[1], pl, msg["current_start"], msg["max_attention_size"],
                                      msg["frame_seqlen"], msg["cam_first_call"], comp)
                    for k in range(len(ins)):
                        link.ch[f"in{k}"].free(comp)
                    link.ch["out"].send(out.contiguous(), comp)
    except BaseException:  # noqa: BLE001
        try:
            conn.send(("error", traceback.format_exc()))
        except OSError:
            pass
        raise


class SplitDiTMP:
    """Drop-in for SplitDiT (same forward signature, returns card 0 tensors), the card 1 half in a helper process."""

    def __init__(self, model, h0=10, dit_sms=0):
        del dit_sms                                         # no SM partition across processes (module docstring)
        self.devs = [torch.device("cuda:0"), torch.device("cuda:1")]
        self.config, self.m, self.h0 = model.config, model, h0
        self.decoder_stream = None                          # the pipeline falls back to a plain stream on the decoder's card
        self.sms = (torch.cuda.get_device_properties(1).multi_processor_count,) * 2
        ctx = mp.get_context("spawn")
        q01, q10 = ctx.Queue(), ctx.Queue()
        sync = torch.zeros(1 << 14, dtype=torch.int64).share_memory_()
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_helper, args=(model, h0, child, q10, q01, sync, os.getpid()), daemon=True)
        pct = os.environ.get("LINGBOT_SPLIT_MP_PCT")        # MPS only: the helper's share of card 1's SMs
        old = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
        if pct:
            os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = pct
        self.proc.start()                                   # pickles the model: CUDA tensors cross as IPC handles
        if pct and old is None:
            del os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"]
        elif pct:
            os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = old
        child.close()
        while not self.conn.poll(1.0):
            if not self.proc.is_alive():
                raise RuntimeError("split_mp helper process died during start-up")
        msg = self.conn.recv()
        if msg[0] != "ready":
            raise RuntimeError(f"split_mp helper failed to start:\n{msg[1]}")
        self.link = _Link(0, q01, q10, sync, self._check)
        self.card = _Card(0, model, self.link, h0)
        self._kv_src = self._key = self._ckey = None
        atexit.register(self.close)
        logging.info(f"SplitDiTMP: helper pid {self.proc.pid} drives cuda:1")

    def _check(self):
        if self.conn.poll():
            msg = self.conn.recv()
            raise RuntimeError(f"split_mp helper failed:\n{msg[1]}")
        if not self.proc.is_alive():
            raise RuntimeError("split_mp helper process died")

    def close(self):
        try:
            self.conn.send(("stop", None))
        except (OSError, ValueError):
            pass

    def _open(self, specs, key_attr, key):
        if getattr(self, key_attr) != key:
            self.conn.send(("open", specs))
            self.link.open(specs)
            setattr(self, key_attr, key)

    def init_crossattn_cache(self, context, crossattn_cache):
        self._kv_src = None                                 # per generate(): fresh self-attention and camera caches
        self.m.init_crossattn_cache(context, crossattn_cache)
        self.card.ca = crossattn_cache
        comp = torch.cuda.current_stream(self.devs[0])
        ctx = [u.contiguous() for u in context]
        meta = [(u.shape, u.dtype) for u in ctx]
        self._open([(f"ctx{k}", 0, u.numel() * u.element_size(), 1) for k, u in enumerate(ctx)], "_ckey", tuple(meta))
        for k, u in enumerate(ctx):
            self.link.ch[f"ctx{k}"].send(u, comp)
        k0 = crossattn_cache[0]["k"]
        self.conn.send(("ca", dict(ctx=meta, ca=(k0.shape, k0.dtype), L=len(crossattn_cache), ac=_ac())))

    def __call__(self, x, t, context, seq_len, y=None, dit_cond_dict=None, kv_cache=None, crossattn_cache=None,
                 current_start=0, max_attention_size=1_000_000, frame_seqlen=None, cross_attn_first_call=None,
                 cam_cache=None, cam_first_call=None):
        m, ch, comp = self.m, self.link.ch, torch.cuda.current_stream(self.devs[0])
        kv = None
        if self._kv_src is not kv_cache:                    # hold the pipeline's list itself (ids are reused)
            k0 = kv_cache[0]["k"]
            kv = (tuple(k0.shape), k0.dtype, len(kv_cache))
            with torch.cuda.stream(comp):
                self.card.set_kv(*kv)
            self._kv_src = kv_cache
        pl = dit_cond_dict["c2ws_plucker_emb"] if dit_cond_dict else None
        ins = [u.contiguous() for u in [x[0], y[0], t] + (list(pl) if pl is not None else [])]
        ps = m.patch_size
        if frame_seqlen is None:
            frame_seqlen = (x[0].shape[-2] // ps[1]) * (x[0].shape[-1] // ps[2])
        L_tok = (x[0].shape[1] // ps[0]) * frame_seqlen
        n = [L_tok * self.h0 // m.num_heads]
        n.append(L_tok - n[0])
        H, h, d, out_f = m.num_heads, [self.h0, m.num_heads - self.h0], m.dim // m.num_heads, m.head.head.out_features
        specs = [(f"in{k}", 0, u.numel() * u.element_size(), 2) for k, u in enumerate(ins)]
        for c in (0, 1):                                    # capacity at 4 bytes per element covers any dtype
            specs += [(f"qkv{c}", c, 3 * n[c] * h[1 - c] * d * 4, 4), (f"back{c}", c, n[1 - c] * h[c] * d * 4, 4)]
        specs.append(("out", 1, n[1] * out_f * 4, 2))
        meta = [(u.shape, u.dtype) for u in ins]
        self._open(specs, "_key", (tuple(meta), L_tok))
        for k, u in enumerate(ins):
            ch[f"in{k}"].send(u, comp)
        self.conn.send(("fwd", dict(ins=meta, kv=kv, current_start=current_start, max_attention_size=max_attention_size,
                                    frame_seqlen=frame_seqlen, cam_first_call=cam_first_call, ac=_ac())))
        out, st = self.card.run(x[0], t, y[0], pl, current_start, max_attention_size, frame_seqlen, cam_first_call,
                                comp)
        assert st["L"] == L_tok, (st["L"], L_tok)
        buf = ch["out"].recv((1, n[1], out_f), out.dtype, comp)
        full = torch.cat([out, buf], dim=1)
        ch["out"].free(comp)
        return [u.float() for u in m.unpatchify(full, st["grid"])]
