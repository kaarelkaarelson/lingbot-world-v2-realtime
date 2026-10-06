"""Can the sequence-parallel exchange hide behind attention? One DiT layer at real shapes, 30 of them in a
row (own weights and KV window each, so nothing stays in L2), on both cards from one process.

Per layer and card (Ulysses): FP8 q/k/v projection on the card's tokens -> send the peer's heads (q|k|v
packed, one message) -> SageAttention for the card's heads over all tokens and the 27,144-token KV window
-> send the peer's tokens back (one message) -> FP8 output projection, cross-attention (local: the text
K/V are on both cards), FP8 FFN. Rope, RMSNorm on q/k and modulation are left out (elementwise, small).

Overlap: heads are processed in G groups; group g's exchange runs while group g-1 attends.
Splits: even (6 heads, 3,016 tokens each) and 9:3 (B3-balanced: card 1 also decodes).

  python experiments/overlap_layer/real_layer.py
"""
import os
import time

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from sageattention import sageattn

from lingbot.layers.linear import FP8Linear

stage = torch.profiler.record_function  # labels for experiments/overlap_layer/breakdown.py

D, H, HD, FFN, T, WIN, CTX, LAYERS = 1536, 12, 128, 8960, 6032, 27144, 512, 30
FWD_PER_CHUNK = 5
dev = [torch.device("cuda:0"), torch.device("cuda:1")]
torch.manual_seed(0)


def fp8(i, o, device, w=None):
    lin = torch.nn.Linear(i, o, bias=True, device=device, dtype=torch.bfloat16)
    if w is not None:
        lin.weight.data.copy_(w)
    return FP8Linear(lin)


class Weights:
    """One layer's weights on one card (replicated across cards, as Ulysses keeps them)."""

    def __init__(self, device, src=None):
        g = src or {}
        self.wqkv = g.get("wqkv", torch.randn(3 * D, D, device=device, dtype=torch.bfloat16) / D ** 0.5).to(device)
        self.o = fp8(D, D, device)
        self.cq, self.co = fp8(D, D, device), fp8(D, D, device)
        self.f1, self.f2 = fp8(D, FFN, device), fp8(FFN, D, device)
        self.ctx_k = torch.randn(1, CTX, H, HD, device=device, dtype=torch.bfloat16)
        self.ctx_v = torch.randn(1, CTX, H, HD, device=device, dtype=torch.bfloat16)
        self.proj = {}

    def qkv_for(self, heads):
        """FP8 projection producing q|k|v for a subset of heads."""
        key = tuple(heads)
        if key not in self.proj:
            rows = torch.cat([torch.arange(h * HD, (h + 1) * HD) + p * D for p in range(3) for h in heads])
            self.proj[key] = fp8(D, 3 * len(heads) * HD, self.wqkv.device, self.wqkv[rows.to(self.wqkv.device)])
        return self.proj[key]


# Token-local pieces, compiled like the production model (weights are module arguments, so the 30 layers share
# one graph per card and shape). Sage and FlashAttention run eagerly between them.
torch._dynamo.config.cache_size_limit = 64
PIECES = int(os.environ.get("PIECES", "1"))
MODE = os.environ.get("COMPILE_MODE") or None  # "reduce-overhead" adds CUDA graphs


@torch.compile(dynamic=False, mode=MODE)
def norm(x):
    return F.layer_norm(x, (D,))


@torch.compile(dynamic=False, mode=MODE)
def proj_pack(h, m, k):
    """q|k|v for k heads of these tokens, packed head-major: [3, k, tokens, d]."""
    return m(h).view(-1, 3, k, HD).permute(1, 2, 0, 3).contiguous()


@torch.compile(dynamic=False, mode=MODE)
def post_attn(x, cols, o, cq):
    x = x + o(cols)
    return x, cq(F.layer_norm(x, (D,))).view(1, -1, H, HD)


@torch.compile(dynamic=False, mode=MODE)
def ffn(x, c, co, f1, f2):
    x = x + co(c)
    return x + f2(F.gelu(f1(F.layer_norm(x, (D,))), approximate="tanh"))


def tail(x, cols, w):
    """After self-attention: output projection, cross-attention (all heads, local text K/V), FFN."""
    with stage("out proj + cross q"):
        x, q = post_attn(x, cols, w.o, w.cq)
    with stage("cross-attention"):
        c = flash_attn_func(q, w.ctx_k, w.ctx_v).reshape(-1, D)
    with stage("cross out + FFN"):
        return ffn(x, c, w.co, w.f1, w.f2)


def one_card_layer(x, w, kwin, vwin):
    qkv = proj_pack(norm(x), w.qkv_for(range(H)), H)  # [3, H, T, d]
    kwin[0, :, WIN - T:] = qkv[1]
    vwin[0, :, WIN - T:] = qkv[2]
    a = sageattn(qkv[0][None], kwin, vwin, tensor_layout="HND")  # [1, H, T, d]
    return tail(x, a[0].permute(1, 0, 2).reshape(T, D), w)


class Split:
    """Two cards: card c owns tokens tok[c] and heads heads[c]; heads processed in `groups` contiguous groups.

    Exchange: explicit pinned staging (card -> host on the sender's copy stream, host -> card on the
    receiver's copy stream). A cross-device `copy_` without P2P holds the CPU ~200 us per 9 MB; these two
    async copies cost ~6 us each to enqueue.
    """

    def __init__(self, tokens0, heads0, groups):
        self.tok = [range(0, tokens0), range(tokens0, T)]
        self.heads = [range(0, heads0), range(heads0, H)]
        self.G = groups
        self.gh = [[range(r.start + len(r) * g // groups, r.start + len(r) * (g + 1) // groups) for g in range(groups)]
                   for r in self.heads]
        # compute on each card's default stream: SageAttention does not order all of its work after a custom
        # current stream (its output was read before it was written), so it must run where it expects
        self.comp = [torch.cuda.default_stream(d) for d in dev]
        self.send = [torch.cuda.Stream(device=d) for d in dev]
        self.recv = [torch.cuda.Stream(device=d) for d in dev]

    def xfer(self, c, src, pieces=None):
        """Send `src` (on card c, produced on comp[c]) to the peer; return (buffer on peer, event to wait on).
        In `pieces` parts so the card->host and host->card hops overlap."""
        pieces = pieces or PIECES
        p = 1 - c
        flat = src.reshape(-1)
        host = torch.empty(flat.shape, dtype=src.dtype, pin_memory=True)
        ready = torch.cuda.Event()
        ready.record(self.comp[c])
        self.send[c].wait_event(ready)
        src.record_stream(self.send[c])
        with torch.cuda.stream(self.recv[p]):
            dst = torch.empty(flat.shape, dtype=src.dtype, device=dev[p])
        step = -(-flat.numel() // pieces)
        for i in range(0, flat.numel(), step):
            with torch.cuda.stream(self.send[c]):
                host[i:i + step].copy_(flat[i:i + step], non_blocking=True)
                d2h = torch.cuda.Event()
                d2h.record(self.send[c])
            self.recv[p].wait_event(d2h)
            with torch.cuda.stream(self.recv[p]):
                dst[i:i + step].copy_(host[i:i + step], non_blocking=True)
        with torch.cuda.stream(self.recv[p]):
            done = torch.cuda.Event()
            done.record(self.recv[p])
        dst.record_stream(self.comp[p])
        return dst.view(src.shape), done

    def layer(self, xs, ws, kwins, vwins):
        n = [len(r) for r in self.tok]
        q_all = [torch.empty(1, len(self.heads[c]), T, HD, device=dev[c], dtype=torch.bfloat16) for c in (0, 1)]
        out_own = [[None] * self.G for _ in (0, 1)]
        inbox_qkv = [[None] * self.G for _ in (0, 1)]
        inbox_out = [[None] * self.G for _ in (0, 1)]
        hs = [None, None]
        for c in (0, 1):
            self.comp[c].wait_stream(torch.cuda.current_stream(dev[c]))  # inputs come from the caller's stream
            with torch.cuda.stream(self.comp[c]), stage("norm"):
                hs[c] = norm(xs[c])
        for g in range(self.G):
            # 1. both cards project the peer's head group and send it (one packed q|k|v message each way)
            for c in (0, 1):
                ph = self.gh[1 - c][g]
                with torch.cuda.stream(self.comp[c]), stage("q|k|v proj (peer heads)"):
                    pk = proj_pack(hs[c], ws[c].qkv_for(ph), len(ph))
                with stage("send q|k|v"):
                    inbox_qkv[1 - c][g] = self.xfer(c, pk)
            # 2. own head group: project locally while the peer's tokens travel, place them, attend
            for c in (0, 1):
                oh, b = self.gh[c][g], self.heads[c].start
                i0, i1 = oh.start - b, oh.stop - b
                r0, r1 = self.tok[c], self.tok[1 - c]
                with torch.cuda.stream(self.comp[c]):
                    with stage("q|k|v proj (own heads) + place"):
                        own = proj_pack(hs[c], ws[c].qkv_for(oh), len(oh))
                        q_all[c][0, i0:i1, r0.start:r0.stop] = own[0]
                        kwins[c][0, i0:i1, WIN - T + r0.start:WIN - T + r0.stop] = own[1]
                        vwins[c][0, i0:i1, WIN - T + r0.start:WIN - T + r0.stop] = own[2]
                    buf, ev = inbox_qkv[c][g]
                    self.comp[c].wait_event(ev)
                    with stage("place received q|k|v"):
                        q_all[c][0, i0:i1, r1.start:r1.stop] = buf[0]
                        kwins[c][0, i0:i1, WIN - T + r1.start:WIN - T + r1.stop] = buf[1]
                        vwins[c][0, i0:i1, WIN - T + r1.start:WIN - T + r1.stop] = buf[2]
                    with stage("self-attention (Sage)"):
                        a = sageattn(q_all[c][:, i0:i1], kwins[c][:, i0:i1], vwins[c][:, i0:i1], tensor_layout="HND")
                    with stage("pack output for peer"):
                        back = a[0, :, r1.start:r1.stop].contiguous()        # the peer's tokens, my heads
                    out_own[c][g] = a[0, :, r0.start:r0.stop]
                with stage("send output"):
                    inbox_out[1 - c][g] = self.xfer(c, back)
        for c in (0, 1):
            with torch.cuda.stream(self.comp[c]):
                cols = torch.empty(n[c], H, HD, device=dev[c], dtype=torch.bfloat16)
                for g in range(self.G):
                    oh, ph = self.gh[c][g], self.gh[1 - c][g]
                    with stage("assemble heads"):
                        cols[:, oh.start:oh.stop] = out_own[c][g].permute(1, 0, 2)
                    buf, ev = inbox_out[c][g]
                    self.comp[c].wait_event(ev)
                    with stage("assemble heads"):
                        cols[:, ph.start:ph.stop] = buf.permute(1, 0, 2)
                xs[c] = tail(xs[c], cols.reshape(n[c], D), ws[c])
            torch.cuda.current_stream(dev[c]).wait_stream(self.comp[c])
        return xs


def timed_split(split, reps=3):
    ws = [[Weights(d) for d in dev] for _ in range(LAYERS)]
    kv = [[(torch.randn(1, len(split.heads[c]), WIN, HD, device=dev[c], dtype=torch.bfloat16),
            torch.randn(1, len(split.heads[c]), WIN, HD, device=dev[c], dtype=torch.bfloat16)) for c in (0, 1)]
          for _ in range(LAYERS)]
    x0 = torch.randn(T, D, device=dev[0], dtype=torch.bfloat16)
    # inputs placed once: a cross-device copy_ without P2P blocks the CPU, which would spoil gpu_only()
    x_in = [x0[split.tok[0].start:split.tok[0].stop].clone(), x0[split.tok[1].start:split.tok[1].stop].to(dev[1])]

    def run(layers=LAYERS):
        xs = [x_in[0].clone(), x_in[1].clone()]
        for l in range(layers):
            xs = split.layer(xs, ws[l], [kv[l][0][0], kv[l][1][0]], [kv[l][0][1], kv[l][1][1]])
        return xs

    run()
    for d in dev:
        torch.cuda.synchronize(d)
    t = time.perf_counter()
    run()
    cpu = (time.perf_counter() - t) / LAYERS * 1e3  # time to enqueue: if close to wall, launch-bound
    for d in dev:
        torch.cuda.synchronize(d)
    ev = [[torch.cuda.Event(enable_timing=True) for _ in range(2)] for _ in dev]
    t = time.perf_counter()
    for c in (0, 1):
        ev[c][0].record(split.comp[c])
    for _ in range(reps):
        run()
    for c in (0, 1):
        ev[c][1].record(split.comp[c])
    for d in dev:
        torch.cuda.synchronize(d)
    wall = (time.perf_counter() - t) / reps / LAYERS * 1e3
    return wall, [ev[c][0].elapsed_time(ev[c][1]) / reps / LAYERS for c in (0, 1)], cpu, gpu_only(run, [split.comp[0], split.comp[1]])


def timed_one_card(reps=3):
    ws = [Weights(dev[0]) for _ in range(LAYERS)]
    kv = [(torch.randn(1, H, WIN, HD, device=dev[0], dtype=torch.bfloat16),
           torch.randn(1, H, WIN, HD, device=dev[0], dtype=torch.bfloat16)) for _ in range(LAYERS)]
    x0 = torch.randn(T, D, device=dev[0], dtype=torch.bfloat16)

    def run(layers=LAYERS):
        x = x0
        for l in range(layers):
            x = one_card_layer(x, ws[l], *kv[l])
        return x

    run()
    torch.cuda.synchronize(dev[0])
    t = time.perf_counter()
    for _ in range(reps):
        run()
    torch.cuda.synchronize(dev[0])
    return (time.perf_counter() - t) / reps / LAYERS * 1e3, gpu_only(run, [torch.cuda.default_stream(dev[0])])


def steady_period(layer_fn, streams, layers=12, skip=4):
    """Steady-state ms per layer: CPU queued ahead behind a sleep, an event on each card after every layer,
    median interval between consecutive layer ends from layer `skip` on (no start-up ramp)."""
    for d in dev:
        torch.cuda.synchronize(d)
    ends = [[] for _ in streams]
    for s in streams:
        with torch.cuda.stream(s):
            torch.cuda._sleep(2_000_000_000)
    state = layer_fn(None, -1)
    for l in range(layers):
        state = layer_fn(state, l)
        for i, s in enumerate(streams):
            e = torch.cuda.Event(enable_timing=True)
            e.record(s)
            ends[i].append(e)
    for d in dev:
        torch.cuda.synchronize(d)
    per = []
    for evs in ends:
        iv = sorted(a.elapsed_time(b) for a, b in zip(evs[skip:], evs[skip + 1:]))
        per.append(iv[len(iv) // 2])
    return max(per)


def gpu_only(run, streams, layers=8):
    """GPU time per layer with the CPU out of the way: park every GPU on a sleep kernel, queue `layers` layers
    behind it, then time from the end of the sleep to the end of the work."""
    for d in dev:
        torch.cuda.synchronize(d)
    ev = []
    for s in streams:
        with torch.cuda.stream(s):
            torch.cuda._sleep(2_000_000_000)  # ~1 s at 2 GHz: longer than queuing the work takes
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(s)
        ev.append((a, b))
    run(layers)
    for s, (a, b) in zip(streams, ev):
        b.record(s)
    for d in dev:
        torch.cuda.synchronize(d)
    return max(a.elapsed_time(b) for a, b in ev) / layers


def check_split_matches_one_card():
    """Same weights and inputs: the split layer must equal the one-card layer (up to Sage's own noise)."""
    w0 = Weights(dev[0])
    w1 = Weights(dev[1], {"wqkv": w0.wqkv})
    for a, b in ((w1.o, w0.o), (w1.cq, w0.cq), (w1.co, w0.co), (w1.f1, w0.f1), (w1.f2, w0.f2)):
        for name in ("w8", "w_scale_t"):
            getattr(a, name).copy_(getattr(b, name))
        a.bias.data.copy_(b.bias.data)
    w1.ctx_k.copy_(w0.ctx_k)
    w1.ctx_v.copy_(w0.ctx_v)
    kw = torch.randn(1, H, WIN, HD, device=dev[0], dtype=torch.bfloat16)
    vw = torch.randn(1, H, WIN, HD, device=dev[0], dtype=torch.bfloat16)
    x = torch.randn(T, D, device=dev[0], dtype=torch.bfloat16)
    ref = one_card_layer(x, w0, kw.clone(), vw.clone())
    s = Split(T // 2, 6, 2)
    kws = [kw[:, :6].clone(), kw[:, 6:].to(dev[1])]
    vws = [vw[:, :6].clone(), vw[:, 6:].to(dev[1])]
    xs = s.layer([x[:T // 2].clone(), x[T // 2:].to(dev[1])], [w0, w1], kws, vws)
    for d in dev:
        torch.cuda.synchronize(d)
    out = torch.cat([xs[0], xs[1].to(dev[0])])
    return ((out.float() - ref.float()).norm() / ref.float().norm()).item()


if __name__ == "__main__":
    torch.set_grad_enabled(False)  # the random weights are Parameters: without this autograd keeps every activation
    print(f"wiring check: split layer vs one-card layer, relative error {check_split_matches_one_card():.2e}")
    one, one_gpu = timed_one_card()
    print(f"one card: {one:.2f} ms per layer wall, {one_gpu:.2f} ms GPU-only -> DiT {one_gpu * LAYERS * FWD_PER_CHUNK / 1e3:.3f} s per chunk")
    print("split\tgroups\twall ms/layer\tCPU enqueue ms\tGPU-only ms/layer\tDiT s/chunk (GPU-only)")
    for name, t0, h0 in (("even 6:6", T // 2, 6), ("9:3", T * 3 // 4, 9)):
        for g in (1, 2, 3):
            wall, per, cpu, gpu = timed_split(Split(t0, h0, g))
            print(f"{name}\t{g}\t{wall:.2f}\t{cpu:.2f}\t{gpu:.2f}\t{gpu * LAYERS * FWD_PER_CHUNK / 1e3:.3f}",
                  flush=True)
