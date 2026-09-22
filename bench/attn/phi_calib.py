#!/usr/bin/env python3
"""phi calibration for candidate C5 ("fixed global max") in the SageAttention kernel.

Measures the real distribution of attention-logit row maxima in the 1.3B causal DiT,
so we can tell whether one constant phi can serve every row / layer / head / step.

UNITS. The kernel computes  exp2( S * sm_scale * log2(e) - phi + S_FP8_OFFSET ).
Everything this script records and prints is therefore in LOG2 units:

    m_log2(row) = max_over_keys( q . k ) * sm_scale * log2(e)

phi must be an upper bound on m_log2 for every row. Usable window (from the synthetic
sweep): phi in [m_log2, m_log2 + WINDOW]. Below it P saturates e4m3 (clipped, silent);
above it P falls off the bottom of the fp8 grid (flushed, silent).

Run:   python bench/attn/phi_calib.py --run --frame_num 193 --bench
Analyse: python bench/attn/phi_calib.py --analyse bench/attn/results/phi_calib.tsv
Self-test: python bench/attn/phi_calib.py --dry
"""
from __future__ import annotations

import argparse
import atexit
import math
import os
import sys

LOG2E = math.log2(math.e)
WINDOW = 3.5            # log2 units of headroom before fp8 resolution is lost
S_FP8_OFFSET = 8.807    # kernel constant, quoted here only for the md's arithmetic

COLUMNS = ["chunk", "forward", "layer", "head", "lq", "lk", "sm_scale",
           "max_log2", "p9999_log2", "p999_log2", "mean_log2", "min_log2"]


# --------------------------------------------------------------------------- selection

def parse_sel(spec: str):
    """'all' | '6-8' | '0,2,5' -> None (= all) or a set of ints."""
    if spec is None or spec.strip().lower() == "all":
        return None
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


class Config:
    def __init__(self, out, layers, forwards, chunks, keys_tile, min_kv, budget_mb):
        self.out = out
        self.layers = parse_sel(layers)
        self.forwards = parse_sel(forwards)
        self.chunks = parse_sel(chunks)
        self.keys_tile = keys_tile
        self.min_kv = min_kv
        self.budget = budget_mb << 20

    def wanted(self, chunk, forward, layer):
        return ((self.chunks is None or chunk in self.chunks)
                and (self.forwards is None or forward in self.forwards)
                and (self.layers is None or layer in self.layers))


# --------------------------------------------------------------------------- hook state

class _State:
    def __init__(self):
        self.current_start = None
        self.lq = None
        self.chunk = -1
        self.forward = 0
        self.order = []          # id(self_attn module) -> layer index, in first-seen order
        self.seen = set()        # module ids already visited in the current forward
        self.layer = None
        self.rows = []
        self.calls = 0


_state = _State()
_cfg: Config = None
_orig_attention = None


def _flush():
    if not _state.rows or _cfg is None:
        return
    new = not os.path.exists(_cfg.out)
    os.makedirs(os.path.dirname(os.path.abspath(_cfg.out)) or ".", exist_ok=True)
    with open(_cfg.out, "a") as f:
        if new:
            f.write("\t".join(COLUMNS) + "\n")
        for r in _state.rows:
            f.write("\t".join(str(c) for c in r) + "\n")
    _state.rows = []


# --------------------------------------------------------------------------- measurement

def _row_max_log2(q, k, sm_scale, keys_tile, budget):
    """Per-(head,row) max of q.k * sm_scale * log2(e), without materialising [Lq, Lk].

    q: [1, Lq, H, D]  k: [1, Lk, H, D], both bf16 on cuda. Returns [H, Lq] fp32.
    Tiles over keys AND queries so the live score block stays inside `budget` bytes.
    """
    import torch

    lq, h, d = q.shape[1], q.shape[2], q.shape[3]
    lk = k.shape[1]
    # [H, L, D]; fp32 with TF32 off so the measured max is the true fp32 value, not a
    # 10-bit-mantissa approximation of it (we are deciding a 3.5-log2 budget).
    qf = q[0].transpose(0, 1).float().contiguous()
    kf = k[0].transpose(0, 1).float().contiguous()
    q_tile = max(64, budget // max(1, h * keys_tile * 4))
    out = torch.full((h, lq), -float("inf"), device=q.device, dtype=torch.float32)
    for qs in range(0, lq, q_tile):
        qb = qf[:, qs:qs + q_tile]
        acc = out[:, qs:qs + q_tile]
        for ks in range(0, lk, keys_tile):
            scores = torch.bmm(qb, kf[:, ks:ks + keys_tile].transpose(1, 2))
            torch.maximum(acc, scores.amax(dim=2), out=acc)
            del scores
    del qf, kf
    return out.mul_(sm_scale * LOG2E)


def _record(q, k, sm_scale):
    import torch

    m = _row_max_log2(q, k, sm_scale, _cfg.keys_tile, _cfg.budget)          # [H, Lq]
    qs = torch.tensor([0.9999, 0.999], device=m.device, dtype=torch.float32)
    quant = torch.quantile(m, qs, dim=1)                                    # [2, H]
    stats = torch.stack([m.amax(1), quant[0], quant[1], m.mean(1), m.amin(1)]).cpu()
    del m
    for head in range(stats.shape[1]):
        mx, p9999, p999, mean, mn = (float(v) for v in stats[:, head])
        _state.rows.append([_state.chunk, _state.forward, _state.layer, head,
                            q.shape[1], k.shape[1], f"{sm_scale:.8g}",
                            f"{mx:.4f}", f"{p9999:.4f}", f"{p999:.4f}",
                            f"{mean:.4f}", f"{mn:.4f}"])
    _state.calls += 1
    if _state.calls % 30 == 0:
        _flush()


def _attention_hook(q, k, v, *args, **kwargs):
    """Wraps wan.modules.attention.attention: measures, then returns the real output."""
    import torch

    measured = False
    if (_state.layer is not None and k.shape[1] >= _cfg.min_kv
            and _cfg.wanted(_state.chunk, _state.forward, _state.layer)):
        scale = kwargs.get("softmax_scale")
        if scale is None and len(args) >= 4:
            scale = args[3]
        if scale is None:
            scale = q.shape[-1] ** -0.5          # SageAttention / FA default
        with torch.no_grad():
            _record(q, k, float(scale))
        measured = True
    out = _orig_attention(q, k, v, *args, **kwargs)
    if measured:
        torch.cuda.empty_cache()
    return out


def _selfattn_hook(orig, pos):
    """Wraps CausalWanSelfAttention.forward to supply chunk / forward / layer indices.

    chunk  : current_start // num_new_tokens (exact, and immune to warm-up forwards).
    forward: incremented when a block we have already visited comes round again, so the
             layer count auto-detects instead of being hard-coded to 30.
    layer  : first-seen order of the self-attention modules inside one forward, which is
             exactly enumerate(self.blocks).
    """
    def wrapper(self, x, *args, **kwargs):
        # the block passes current_start positionally; `pos` is its index in *args
        cs = args[pos] if len(args) > pos >= 0 else kwargs.get("current_start", 0)
        lq = x.shape[1]
        if cs != _state.current_start:
            _state.current_start = cs
            _state.chunk = int(cs // lq) if lq else 0
            _state.forward = 0
            _state.seen = set()
        mid = id(self)
        if mid in _state.seen:
            _state.forward += 1
            _state.seen = set()
        _state.seen.add(mid)
        if mid not in _state.order:
            _state.order.append(mid)
        _state.layer = _state.order.index(mid)
        try:
            return orig(self, x, *args, **kwargs)
        finally:
            _state.layer = None
    wrapper.__phi_calib__ = True
    return wrapper


def install(cfg: Config):
    """Patch attention() and the self-attention forward in place, without editing them."""
    global _cfg, _orig_attention
    import torch
    import wan.modules.attention as attn_mod

    _cfg = cfg
    torch.backends.cuda.matmul.allow_tf32 = False
    _orig_attention = attn_mod.attention
    attn_mod.attention = _attention_hook
    # Every `from .attention import attention` holds its own reference; rebind those too.
    for name, mod in list(sys.modules.items()):
        if name.startswith("wan.") and getattr(mod, "attention", None) is _orig_attention:
            setattr(mod, "attention", _attention_hook)

    patched = set()
    for name, mod in list(sys.modules.items()):
        if not name.startswith("wan.modules."):
            continue
        for obj in vars(mod).values():
            if not isinstance(obj, type) or obj in patched:
                continue
            fwd = obj.__dict__.get("forward")          # defined here, not inherited
            code = getattr(fwd, "__code__", None)
            if code is None or getattr(fwd, "__phi_calib__", False):
                continue
            names = code.co_varnames[:code.co_argcount + code.co_kwonlyargcount]
            if "current_start" in names:
                obj.forward = _selfattn_hook(fwd, names.index("current_start") - 2)
                patched.add(obj)
    if not patched:
        raise SystemExit("phi_calib: found no self-attention forward taking current_start")
    atexit.register(_flush)
    print(f"[phi_calib] hook installed ({len(patched)} self-attn classes) -> {cfg.out}", flush=True)


# --------------------------------------------------------------------------- launcher

def run(rest, cfg: Config):
    import runpy

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(repo)
    sys.path.insert(0, repo)
    from lingbot.cli import CLIP_DEFAULTS, _env
    from lingbot.presets import apply_preset

    preset = "fast"
    for i, a in enumerate(rest):
        if a == "--preset" and i + 1 < len(rest):
            preset = rest[i + 1]
        elif a.startswith("--preset="):
            preset = a.split("=", 1)[1]
    apply_preset(preset)                       # before `import wan`: backend is import-time
    if "--bench" in rest:
        os.environ.setdefault("LINGBOT_BENCH_TIMING", "1")
    os.environ.update(_env())
    # The hook runs Python per attention call and mutates Python state; dynamo would
    # constant-fold the indices and try to trace the score tiling.
    if os.environ.get("LINGBOT_TORCH_COMPILE") not in (None, "0"):
        print("[phi_calib] forcing LINGBOT_TORCH_COMPILE=0 for the calibration run", flush=True)
    os.environ["LINGBOT_TORCH_COMPILE"] = "0"
    os.environ["LINGBOT_INDUCTOR_TUNE"] = "0"
    if os.environ.get("LINGBOT_ATTN") == "sage_kvq":
        raise SystemExit("phi_calib: LINGBOT_ATTN=sage_kvq bypasses attention(); use sage")

    import wan.modules.attention  # noqa: F401  (import wan with the preset already applied)
    import wan.modules.model_fast_fusion  # noqa: F401
    install(cfg)

    sys.argv = ["generate.py", *CLIP_DEFAULTS, *rest]
    try:
        runpy.run_path(os.path.join(repo, "generate.py"), run_name="__main__")
    finally:
        _flush()


# --------------------------------------------------------------------------- analysis

def _read_tsv(path):
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        rows = []
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            rec = dict(zip(head, line.split("\t")))
            rows.append({
                "chunk": int(rec["chunk"]), "forward": int(rec["forward"]),
                "layer": int(rec["layer"]), "head": int(rec["head"]),
                "max": float(rec["max_log2"]), "p9999": float(rec["p9999_log2"]),
                "p999": float(rec["p999_log2"]), "mean": float(rec["mean_log2"]),
                "min": float(rec["min_log2"]),
            })
    if not rows:
        raise SystemExit(f"phi_calib: no rows in {path}")
    return rows


def _pct(vals, q):
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def analyse(rows, window=WINDOW):
    by_layer, by_lh = {}, {}
    for r in rows:
        by_layer.setdefault(r["layer"], []).append(r)
        by_lh.setdefault((r["layer"], r["head"]), []).append(r)

    g_max = max(r["max"] for r in rows)
    layers = sorted(by_layer)
    lmax = {l: max(r["max"] for r in by_layer[l]) for l in layers}
    lp9999 = {l: _pct([r["p9999"] for r in by_layer[l]], 1.0) for l in layers}
    lhmax = {k: max(r["max"] for r in v) for k, v in by_lh.items()}

    print(f"rows={len(rows)}  chunks={sorted({r['chunk'] for r in rows})}  "
          f"forwards={sorted({r['forward'] for r in rows})}  layers={len(layers)}  "
          f"heads={len({r['head'] for r in rows})}")
    print(f"window for fp8 resolution: phi - m_log2 must stay in [0, {window}]")
    print()
    print(f"GLOBAL MAX m_log2 = {g_max:.4f}   <- phi for a single-constant kernel "
          f"(use phi = {g_max:.4f} rounded up, e.g. {math.ceil(g_max * 100) / 100:.2f})")
    print(f"  kernel exp2 arg at the hottest row = S_FP8_OFFSET = {S_FP8_OFFSET}")
    print()
    print("per layer:")
    print(f"  {'layer':>5}  {'max':>9}  {'p99.99':>9}  {'gap to global':>13}  {'heads >window below':>19}")
    for l in layers:
        heads_lo = sum(1 for (ll, _), v in lhmax.items() if ll == l and g_max - v > window)
        print(f"  {l:>5}  {lmax[l]:>9.4f}  {lp9999[l]:>9.4f}  {g_max - lmax[l]:>13.4f}  {heads_lo:>19}")
    print()
    spread = max(lmax.values()) - min(lmax.values())
    lh_spread = g_max - min(lhmax.values())
    n_lo = sum(1 for v in lhmax.values() if g_max - v > window)
    print(f"per-layer max spread (largest - smallest) = {spread:.4f} log2")
    print(f"global max - smallest (layer,head) max    = {lh_spread:.4f} log2")
    print(f"(layer,head) pairs more than {window} log2 below a single global phi: "
          f"{n_lo} / {len(lhmax)}")

    worst_layer = max(
        (max(v for (ll, _), v in lhmax.items() if ll == l)
         - min(v for (ll, _), v in lhmax.items() if ll == l)) for l in layers)
    print(f"worst within-layer head spread            = {worst_layer:.4f} log2")
    print()
    if lh_spread <= window:
        verdict = "single global phi viable"
    elif worst_layer <= window:
        verdict = "needs per-layer phi"
    else:
        verdict = "needs per-layer-per-head phi"
    print(f"VERDICT: {verdict}")
    return verdict


# --------------------------------------------------------------------------- dry run

def dry(cfg: Config):
    import random

    random.seed(0)
    path = cfg.out
    if os.path.exists(path):
        os.remove(path)
    rows = []
    for chunk in (6, 7, 8):
        for layer in range(30):
            base = 12.0 + 0.35 * layer                       # deliberate per-layer drift
            for head in range(12):
                mx = base + random.uniform(-1.5, 1.5)
                rows.append([chunk, 0, layer, head, 6032, 27144, "0.08838835",
                             f"{mx:.4f}", f"{mx - 0.4:.4f}", f"{mx - 1.1:.4f}",
                             f"{mx - 6.0:.4f}", f"{mx - 14.0:.4f}"])
    _state.rows = rows
    global _cfg
    _cfg = cfg
    _flush()
    print(f"[phi_calib --dry] fabricated {len(rows)} rows -> {path}\n")
    analyse(_read_tsv(path))
    return 0


# --------------------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", action="store_true",
                   help="install the hook and run generate.py with `lingbot clip` defaults; "
                        "all remaining args are passed to generate.py")
    p.add_argument("--analyse", metavar="TSV", help="analyse an existing TSV and exit")
    p.add_argument("--dry", action="store_true", help="fabricate a distribution, analyse it, exit 0")
    p.add_argument("--out", default=os.environ.get("PHI_CALIB_TSV", "bench/attn/results/phi_calib.tsv"))
    p.add_argument("--layers", default="all", help="layer indices to sample (default: all 30)")
    p.add_argument("--forwards", default="0",
                   help="forward index within the chunk, 0-3 denoise + 4 cache-write (default: 0)")
    p.add_argument("--chunks", default="6-8", help="chunk indices to sample (default: 6-8, steady state)")
    p.add_argument("--keys-tile", type=int, default=4096, help="keys per score tile")
    p.add_argument("--min-kv", type=int, default=4096,
                   help="skip calls with fewer keys than this (excludes cross-attention)")
    p.add_argument("--budget-mb", type=int, default=96, help="max live score block per tile")
    p.add_argument("--window", type=float, default=WINDOW, help="usable phi window in log2 units")
    args, rest = p.parse_known_args(argv)

    cfg = Config(args.out, args.layers, args.forwards, args.chunks,
                 args.keys_tile, args.min_kv, args.budget_mb)
    if args.dry:
        return dry(Config(args.out if args.out != p.get_default("out") else
                          os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "results", "phi_calib_dry.tsv"),
                          args.layers, args.forwards, args.chunks,
                          args.keys_tile, args.min_kv, args.budget_mb))
    if args.analyse:
        analyse(_read_tsv(args.analyse), window=args.window)
        return 0
    if args.run:
        run(rest, cfg)
        return 0
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
