"""CPU equivalence + Dynamo graph-break check for wan/modules/model_fast_fusion.py (opt10, DiT fusion).

Builds WanModelFast at a tiny config with seeded random weights from the unmodified
upstream module and from wan/modules/model_fast_fusion.py, runs the same 5-chunk causal loop
(4 denoising forwards + 1 KV-cache write per chunk, KV window + sink, cross-attn
cache — the shape of image2video._generate_causal_fast) through both, and asserts
torch.allclose(atol=1e-6) on every forward's output. Then runs torch._dynamo.explain
on both and a torch.compile(dynamic=True) loop on the patched module to count graphs
and recompiles.

Run:  ~/lingbot-world-bench/.venv/bin/python tests/test_dit_fusion_cpu.py   (or pytest)
diffusers / flash_attn are not needed: they are stubbed below.
"""
import copy
import importlib.util
import os
import sys
import types

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = ROOT
UPSTREAM_MODEL = os.path.join(UPSTREAM, "reference/wan/modules/model_fast.py")
PATCHED_MODEL = os.path.join(ROOT, "lingbot/models/lingbot_world/transformer.py")

CFG = dict(model_type='i2v', patch_size=(1, 2, 2), text_len=8, in_dim=36, dim=64, ffn_dim=128,
           freq_dim=64, text_dim=32, out_dim=16, num_heads=2, num_layers=2,
           local_attn_size=9, sink_size=2, qk_norm=True, cross_attn_norm=True, eps=1e-6)
LAT_F, LAT_H, LAT_W, CHUNK = 20, 4, 6, 4  # 5 chunks of 4 latent frames; window fills at chunk 2
TIMESTEPS = torch.tensor([1000, 750, 500, 250], dtype=torch.long)  # int64 like the scheduler's
FRAME_SEQLEN = LAT_H * LAT_W // 4
SEQ_LEN = CHUNK * FRAME_SEQLEN


def _stub_packages():
    # diffusers mixins are only used for from_pretrained/config; wan/__init__ pulls in cv2 etc.
    d, cu = types.ModuleType("diffusers"), types.ModuleType("diffusers.configuration_utils")
    mu, mm = types.ModuleType("diffusers.models"), types.ModuleType("diffusers.models.modeling_utils")
    cu.ConfigMixin = type("ConfigMixin", (), {})
    cu.register_to_config = lambda fn: fn
    mm.ModelMixin = type("ModelMixin", (torch.nn.Module,), {})
    sys.modules.update({"diffusers": d, "diffusers.configuration_utils": cu,
                        "diffusers.models": mu, "diffusers.models.modeling_utils": mm})
    # package stubs with __path__ set, so submodules import without running the heavy __init__s
    sys.path.insert(0, ROOT)
    for name, path in (("wan", "reference/wan"), ("wan.modules", "reference/wan/modules")):
        pkg = types.ModuleType(name)
        pkg.__path__ = [os.path.join(ROOT, path)]
        sys.modules[name] = pkg


def _sdpa(q, k, v, *args, **kwargs):
    # [B, L, H, D] in/out, replaces flash_attn / sageattn on CPU
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return o.transpose(1, 2).contiguous()


def load_model_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    m.flash_attention = m.attention = _sdpa
    return m


def build(mod, state_dict=None):
    torch.manual_seed(0)
    m = mod.WanModelFast(**CFG)
    torch.nn.init.normal_(m.head.head.weight, std=0.02)  # init_weights zeroes it
    if state_dict is not None:
        m.load_state_dict(state_dict)
    return m.eval()


def make_caches(fusion):
    heads, head_dim = CFG['num_heads'], CFG['dim'] // CFG['num_heads']
    kv_size = FRAME_SEQLEN * CFG['local_attn_size']
    kv = [dict(k=torch.zeros(1, kv_size, heads, head_dim), v=torch.zeros(1, kv_size, heads, head_dim),
               global_end_index=torch.tensor([0]), local_end_index=torch.tensor([0]),
               global_end_int=0, local_end_int=0) for _ in range(CFG['num_layers'])]
    cross = [dict(k=torch.zeros(1, CFG['text_len'], heads, head_dim),
                  v=torch.zeros(1, CFG['text_len'], heads, head_dim),
                  is_init=torch.tensor(0, dtype=torch.int32)) for _ in range(CFG['num_layers'])]
    cam = [dict(scale=torch.zeros(1, SEQ_LEN, CFG['dim']), shift=torch.zeros(1, SEQ_LEN, CFG['dim']))
           for _ in range(CFG['num_layers'])] if fusion else None
    return kv, cross, cam, kv_size


def run_loop(model, fusion, call=None, record=None):
    """Mirror of _generate_causal_fast. `record` collects deep-copied kwargs per forward."""
    call = model if call is None else call
    g = torch.Generator().manual_seed(1234)
    noise = torch.randn(16, LAT_F, LAT_H, LAT_W, generator=g)
    cond = torch.randn(20, LAT_F, LAT_H, LAT_W, generator=g)
    cam = torch.randn(1, 384, LAT_F, LAT_H, LAT_W, generator=g)
    ctx = torch.randn(CFG['text_len'], CFG['text_dim'], generator=g)
    kv, cross, cam_cache, kv_size = make_caches(fusion)
    initialized = False
    outs = []

    def fwd(**kw):
        if record is not None:
            record.append(copy.deepcopy(kw))  # caches are mutated in place
        return call(**kw)

    with torch.no_grad():
        if fusion:
            model.init_crossattn_cache([ctx], cross)
            initialized = True
        for c in range(LAT_F // CHUNK):
            latent = noise[:, c * CHUNK:(c + 1) * CHUNK]
            kwargs = dict(context=[ctx], seq_len=SEQ_LEN, y=[cond[:, c * CHUNK:(c + 1) * CHUNK]],
                          dit_cond_dict={"c2ws_plucker_emb": cam[:, :, c * CHUNK:(c + 1) * CHUNK].chunk(1, dim=0)},
                          kv_cache=kv, crossattn_cache=cross, current_start=c * SEQ_LEN,
                          max_attention_size=kv_size, frame_seqlen=FRAME_SEQLEN)
            if fusion:
                kwargs['cam_cache'] = cam_cache
            for s in range(len(TIMESTEPS)):
                if fusion:
                    kwargs['cam_first_call'] = s == 0
                pred = fwd(x=[latent], t=TIMESTEPS[s:s + 1].clone(),
                           cross_attn_first_call=not initialized, **kwargs)[0]
                initialized = True
                outs.append(pred)
                x0 = latent - float(TIMESTEPS[s]) / 1000 * pred
                if s < len(TIMESTEPS) - 1:
                    nxt = float(TIMESTEPS[s + 1]) / 1000
                    latent = (1 - nxt) * x0 + nxt * torch.randn(x0.shape, generator=g)
            if fusion:
                kwargs['cam_first_call'] = False
            t0 = TIMESTEPS[-1:] * 0 if fusion else TIMESTEPS[-1:] * 0.0  # pipeline: int64 vs float
            fwd(x=[x0], t=t0, cross_attn_first_call=False, **kwargs)
    return outs


def explain(model, kwargs):
    import torch._dynamo as dynamo
    dynamo.reset()
    with torch._dynamo.config.patch(assume_static_by_default=False):  # == torch.compile(dynamic=True)
        try:
            return dynamo.explain(model)(**kwargs), None
        except Exception as e:  # upstream fails to trace under dynamic ints on torch >= 2.14
            return None, e


def main():
    _stub_packages()
    up = load_model_module(UPSTREAM_MODEL, "wan.modules.model_fast")
    fu = load_model_module(PATCHED_MODEL, "wan.modules.model_fast_fusion")
    m_up = build(up)
    m_fu = build(fu, m_up.state_dict())

    calls_up, calls_fu = [], []
    ref = run_loop(m_up, fusion=False, record=calls_up)
    out = run_loop(m_fu, fusion=True, record=calls_fu)
    assert len(ref) == len(out) == 20
    print(f"torch {torch.__version__}; {len(ref)} denoising forwards over {LAT_F // CHUNK} chunks")
    worst = 0.0
    for i, (a, b) in enumerate(zip(ref, out)):
        d = (a - b).abs().max().item()
        worst = max(worst, d)
        assert torch.allclose(a, b, atol=1e-6, rtol=0), f"forward {i}: max|diff|={d}"
    print(f"numerics: all 20 forwards allclose(atol=1e-6): True; max |diff| = {worst:.2e}")

    # the only non-bitwise change is the time-embedding MLP on 1 row instead of L identical
    # rows (BLAS blocking); with t pre-expanded to [B, L] the patched module is bit-identical
    def call_t_expanded(x, t, **kw):
        return m_fu(x=x, t=t.expand(t.size(0), SEQ_LEN), **kw)
    out2 = run_loop(m_fu, fusion=True, call=call_t_expanded)
    print("bitwise identical with t pre-expanded to [B, L]:", all(torch.equal(a, b) for a, b in zip(ref, out2)))

    # the verification switch: EXACT_T (time MLP on the L pre-expanded rows, as upstream)
    for env in ({"LINGBOT_DIT_FUSION_EXACT_T": "1"},):
        os.environ.update(env)
        m_ex = build(load_model_module(PATCHED_MODEL, "wan.modules.model_fast_fusion_exact"), m_up.state_dict())
        out3 = run_loop(m_ex, fusion=True)
        print(f"bitwise identical with {'+'.join(env)}:", all(torch.equal(a, b) for a, b in zip(ref, out3)))
        ex, err = explain(m_ex, calls_fu[10])
        print(f"   explain chunk2 step0: graphs={ex.graph_count} breaks={ex.graph_break_count}" if ex else f"   explain FAILED: {err}")
        for k in env:
            del os.environ[k]

    # graph breaks: first forward, an overwrite forward, first eviction forward
    print("\ntorch._dynamo.explain (dynamic shapes), per forward:")
    for label, model, calls in (("upstream", m_up, calls_up), ("patched ", m_fu, calls_fu)):
        for i, what in ((0, "chunk0 step0 (append)"), (6, "chunk1 step1 (overwrite)"), (10, "chunk2 step0 (evict)")):
            ex, err = explain(model, calls[i])
            if ex is None:
                print(f"  {label} {what}: FAILED to trace: {str(err).splitlines()[0][:110]}")
            else:
                print(f"  {label} {what}: graphs={ex.graph_count} breaks={ex.graph_break_count} ops={ex.op_count}")
                for br in ex.break_reasons[:6]:
                    lines = [f.line for f in br.user_stack if f.filename and 'model_fast' in f.filename]
                    print("      break:", str(br.reason).splitlines()[0][:70], "| at:", (lines or ['?'])[-1][:80])
    with torch._dynamo.config.patch(assume_static_by_default=True):  # static shapes: upstream traces
        import torch._dynamo as dynamo
        dynamo.reset()
        ex = dynamo.explain(m_up)(**calls_up[10])
        print(f"  upstream chunk2 step0 with static shapes: graphs={ex.graph_count} breaks={ex.graph_break_count} "
              f"(x{CFG['num_layers']} layers; scales with the layer count)")

    # compiled loop: unique graphs == recompile count; output must equal eager exactly (aot_eager)
    from torch._dynamo.utils import counters
    import torch._dynamo as dynamo
    dynamo.reset(); counters.clear()
    compiled = torch.compile(m_fu, dynamic=True, backend="aot_eager", fullgraph=True)
    outc = run_loop(m_fu, fusion=True, call=compiled)
    print(f"\ntorch.compile(dynamic=True, fullgraph=True) over the 5-chunk loop: unique graphs = "
          f"{counters['stats']['unique_graphs']} (expected 3: cam-first+append, overwrite, cam-first+evict), "
          f"graph breaks = {sum(counters['graph_break'].values())}")
    assert counters['stats']['unique_graphs'] == 3
    assert all(torch.equal(a, b) for a, b in zip(out, outc)), "compiled != eager"
    print("compiled output == eager patched output (bitwise): True")
    print("\nOK")


def test_dit_fusion_cpu():
    main()


if __name__ == "__main__":
    main()
