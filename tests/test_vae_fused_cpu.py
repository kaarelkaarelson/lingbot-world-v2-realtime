#!/usr/bin/env python3
"""CPU fp32 check of lingbot/models/lingbot_world/vae.py against the paper's Wan2.1 decoder loop
(random seeded weights, small clip). Run: ~/lingbot-world-bench/.venv/bin/python tests/test_vae_fused_cpu.py
"""
import copy, importlib.util, os, sys, types
os.environ.setdefault("OMP_NUM_THREADS", "1")  # this venv links two libomp copies on macOS: deadlocks when threaded
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name, path):  # `reference/wan/__init__.py` pulls in the whole pipeline; load just the two VAE files
    # package stubs with their real __path__, so later imports in the same process still resolve
    for pkg, pkg_dir in (("wan", "reference/wan"), ("wan.modules", "reference/wan/modules"), ("lingbot", "lingbot"),
                      ("lingbot.models", "lingbot/models"), ("lingbot.models.lingbot_world", "lingbot/models/lingbot_world")):
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)
            sys.modules[pkg].__path__ = [os.path.join(ROOT, pkg_dir)]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


WanVAE_ = load("wan.modules.vae2_1", os.path.join(ROOT, "reference", "wan", "modules", "vae2_1.py")).WanVAE_
fused = load("lingbot.models.lingbot_world.vae", os.path.join(ROOT, "lingbot", "models", "lingbot_world", "vae.py"))


def build(seed=0):
    torch.manual_seed(seed)
    model = WanVAE_(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
                    temperal_downsample=[False, True, True]).eval().requires_grad_(False)
    for n, p in model.named_parameters():
        p.copy_(1 + 0.1 * torch.randn_like(p) if n.endswith("gamma") else 0.05 * torch.randn_like(p))
    for p in model.decoder.head[2].parameters():
        p.mul_(0.1)  # keep the output inside [-1, 1] so the clamp hides nothing
    mean, std = torch.randn(16), 0.5 + torch.rand(16)
    return types.SimpleNamespace(model=model, dtype=torch.float32, scale=[mean, 1.0 / std])


def stock_decode(vae, z):
    return vae.model.decode(z.unsqueeze(0), vae.scale).float().clamp_(-1, 1).squeeze(0)


def check(name, out, ref):
    ok = out.shape == ref.shape and torch.allclose(out, ref, atol=1e-5)
    print(f"{name}: shape {tuple(out.shape)} vs {tuple(ref.shape)}, max|diff| {(out - ref).abs().max().item():.2e}, "
          f"|ref| max {ref.abs().max().item():.3f}, nan {torch.isnan(out).any().item()} -> {'OK' if ok else 'FAIL'}")
    return ok


def main():
    vae = build()
    z = torch.randn(16, 9, 8, 16)  # 9 latents -> 1 + 4*8 = 33 frames
    with torch.no_grad():
        ref = stock_decode(vae, z)
        print(f"reference: {tuple(ref.shape)}, saturated {(ref.abs() >= 1).float().mean().item():.3f}")
        # channels_last is a cuDNN (NHWC) layout choice; CPU conv kernels sum in a different order per layout (~2e-5)
        check("eager fused, channels_last weights", fused.FusedDecoder(copy.deepcopy(vae)).decode(z), ref)
        dec = fused.FusedDecoder(copy.deepcopy(vae), channels_last=False)
        oks = [check("eager fused vs stock", dec.decode(z), ref)]
        outs, state = [], None
        for s in range(0, z.shape[1], 2):
            y, state = dec.decode_step(z[:, s:s + 2], state)
            outs.append(y)
        oks.append(check("streaming 2 latents/call", torch.cat(outs, 1), ref))
        assert ref.shape[1] == 33 and len(state) == 32, (ref.shape, len(state))
        try:
            cdec = fused.FusedDecoder(copy.deepcopy(vae), compile_mode="default", channels_last=False)
            oks.append(check("torch.compile(default, fullgraph) vs stock", cdec.decode(z), ref))
        except Exception as e:  # noqa: BLE001
            print(f"torch.compile on CPU: not testable here ({type(e).__name__}: {str(e)[:200]})")
    print("ALL OK" if all(oks) else "FAILED"); sys.exit(0 if all(oks) else 1)


if __name__ == "__main__":
    main()
