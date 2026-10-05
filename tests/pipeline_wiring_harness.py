"""Run the generation loop end to end on CPU with a tiny DiT and mock VAE/T5, and record what flows
through it. Used by test_pipeline_wiring_cpu.py; `--old <checkout>` runs the pre-restructure
pipeline (wan/image2video.py at the `pre-cleanup` tag) to produce the golden file.

    python tests/pipeline_wiring_harness.py --new  <out.pt>
    python tests/pipeline_wiring_harness.py --old <pre-cleanup checkout> <out.pt>
"""
import hashlib
import os
import sys
import types

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.cuda.current_device = lambda: 0  # the paper's t5.py evaluates this at import; nothing here touches CUDA

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CFG = dict(model_type='i2v', patch_size=(1, 2, 2), text_len=8, in_dim=36, dim=64, ffn_dim=128, freq_dim=64,
           text_dim=32, out_dim=16, num_heads=2, num_layers=2, local_attn_size=9, sink_size=2, qk_norm=True,
           cross_attn_norm=True, eps=1e-6)
PROMPT = "a tiny test world"
CHUNK, FRAMES, H, W = 4, 61, 64, 96  # 16 latent frames = 4 chunks; window 9 < 16 so eviction runs


def _sdpa(q, k, v, *args, **kwargs):  # [B, L, H, D]; stands in for FlashAttention/SageAttention on CPU
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).contiguous()


class MockVAE:
    def __init__(self, log):
        self.log = log

    def encode(self, videos):
        c, f, h, w = videos[0].shape
        self.log["vae_encode_in"] = tuple(videos[0].shape)
        g = torch.Generator().manual_seed(11)
        return [torch.randn(16, (f - 1) // 4 + 1, h // 8, w // 8, generator=g)]


class MockDecoder:
    def __init__(self, log):
        self.log = log

    def decode(self, z):
        self.log["decoded_latents"] = z.detach().clone()
        t, h, w = z.shape[1:]
        return torch.zeros(3, 1 + 4 * (t - 1), 8 * h, 8 * w) + z.mean()


def run(old_root=None):
    log, calls = {}, []
    os.environ.update({"LINGBOT_DIT_FUSION": "1", "LINGBOT_SYNCFREE": "1"})
    if old_root:
        sys.path.insert(0, old_root)
        import wan.image2video as I
        import wan.modules.model_fast_fusion as M
        pipe = object.__new__(I.WanI2VCausal)
        extra = dict(sp_size=1, _vae_fused=MockDecoder(log), _vae_cl=False, _vae_half=False, _taehv=None,
                     _flashvaed=None, _cross_attn_initialized=False, infer_mode="causal_fast")
        cfg_mod = "wan.configs"
    else:
        sys.path.insert(0, ROOT)
        import lingbot.models.lingbot_world.transformer as M
        import lingbot.pipelines.lingbot_world as I
        pipe = object.__new__(I.LingBotWorldPipeline)
        extra = dict(decoder=MockDecoder(log), decoder_device=torch.device("cpu"), split_decoder=False)
        cfg_mod = "wan.configs"
    M.attention = M.flash_attention = _sdpa
    from importlib import import_module
    sched_cls = import_module("wan.utils.fm_solvers_unipc").FlowUniPCMultistepScheduler
    cfg = import_module(cfg_mod).WAN_CONFIGS["i2v-1.3B"]

    torch.manual_seed(0)
    model = M.WanModelFast(**CFG)
    torch.nn.init.normal_(model.head.head.weight, std=0.02)  # init_weights zeroes it
    model.eval().requires_grad_(False)
    forward = model.forward

    def counting_forward(*a, **kw):
        kv = kw["kv_cache"][0]
        calls.append((float(kw["t"].float().reshape(-1)[0]), kw["current_start"], kw.get("cam_first_call"),
                      kv["global_end_int"], kv["local_end_int"]))
        return forward(*a, **kw)
    model.forward = counting_forward

    key = hashlib.sha256(PROMPT.encode('utf-8')).hexdigest()
    ctx = torch.randn(CFG['text_len'], CFG['text_dim'], generator=torch.Generator().manual_seed(3))
    pipe.__dict__.update(
        device=torch.device("cpu"), config=cfg, rank=0, t5_cpu=True, num_train_timesteps=cfg.num_train_timesteps,
        param_dtype=torch.float32, pipe_dtype=torch.float32, local_attn_size=CFG['local_attn_size'],
        sink_size=CFG['sink_size'], vae_stride=cfg.vae_stride, patch_size=cfg.patch_size, model=model,
        scheduler=sched_cls(num_train_timesteps=cfg.num_train_timesteps, shift=1, use_dynamic_shifting=False),
        vae=MockVAE(log), frame_sink=None, pose_provider=None, chunk_gate=None, _y_cache=None,
        _t5_cache={key: [ctx]}, _t5_disk_cache_dir="/nonexistent", text_encoder=None, **extra)

    from PIL import Image
    img = Image.fromarray(np.random.default_rng(5).integers(0, 255, (H, W, 3), dtype=np.uint8))
    with torch.no_grad():
        video = pipe.generate(PROMPT, img, os.path.join(ROOT, "examples", "03"), chunk_size=CHUNK,
                              max_area=H * W, frame_num=FRAMES, timesteps_index=[0, 250, 500, 750], shift=5.0,
                              seed=42, offload_model=False, max_sequence_length=CFG['text_len'])
    return dict(calls=calls, video_shape=tuple(video.shape), **log)


if __name__ == "__main__":
    if sys.argv[1] == "--old":
        out = run(old_root=sys.argv[2]); path = sys.argv[3]
    else:
        out = run(); path = sys.argv[2]
    torch.save(out, path)
    print(f"{len(out['calls'])} DiT calls, decoder got {tuple(out['decoded_latents'].shape)}, video {out['video_shape']}")
