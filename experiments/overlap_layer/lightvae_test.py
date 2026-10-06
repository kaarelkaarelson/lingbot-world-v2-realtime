"""LightVAE (LightX2V's lightvaew2_1: the Wan 2.1 VAE with every channel count cut to a quarter) vs the full
Wan VAE, both through the same fused fp16 compiled decoder (FusedDecoder), on identical latents.

Latents: the full Wan VAE's encoding of a real rollout (a2a8/fast1.mp4, 157 frames). Decoded chunk by chunk as
the pipeline does (first latent alone, then 4 at a time). Reports steady decode time per chunk and, per frame,
PSNR / LPIPS of LightVAE against the full VAE's decode (same latents) and of both against the source video.

  python experiments/overlap_layer/lightvae_test.py
"""
import sys
import time

import imageio.v2 as imageio
import lpips
import numpy as np
import torch

import lingbot  # noqa: F401  (puts reference/ on sys.path)
from lingbot.models.lingbot_world.vae import FusedDecoder
from wan.modules.vae2_1 import Wan2_1_VAE, _video_vae

torch.set_grad_enabled(False)
dev = torch.device("cuda:0")
FULL = "weights/lingbot-world-v2-14b-causal-fast/Wan2.1_VAE.pth"
LIGHT = "/workspace/weights_extra/lightvaew2_1.pth"
SRC = sys.argv[1] if len(sys.argv) > 1 else "/workspace/runs/a2a8/fast1.mp4"

frames = np.stack(imageio.mimread(SRC, memtest=False))                        # [T, H, W, 3] uint8
video = torch.from_numpy(frames).to(dev).permute(3, 0, 1, 2).float() / 127.5 - 1  # [3, T, H, W]
full = Wan2_1_VAE(vae_pth=FULL, dtype=torch.float, device=dev)
z = full.encode([video])[0]
print(f"source {SRC}: {frames.shape[0]} frames {frames.shape[1]}x{frames.shape[2]} -> latents {tuple(z.shape)}")

light = Wan2_1_VAE(vae_pth=FULL, dtype=torch.float, device=dev)
light.model = _video_vae(LIGHT, z_dim=16, dim=24).eval().requires_grad_(False).to(dev)
decs = {"full Wan VAE": FusedDecoder(full, compile_mode="max-autotune-no-cudagraphs"),
        "LightVAE (dim 24)": FusedDecoder(light, compile_mode="max-autotune-no-cudagraphs")}


def decode_all(dec, timed=False):
    out, state, times = [], None, []
    pieces = [z[:, :1]] + [z[:, i:i + 4] for i in range(1, z.shape[1], 4)]
    for p in pieces:
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fr, state = dec.decode_step(p, state)
        e1.record()
        out.append(fr)
        times.append((e0, e1))
    torch.cuda.synchronize(dev)
    ms = [a.elapsed_time(b) for a, b in times]
    return torch.cat(out, 1), ms


res = {}
for name, dec in decs.items():
    decode_all(dec)          # compile + autotune
    decode_all(dec)
    x, ms = decode_all(dec)
    steady = sorted(ms[2:])[len(ms[2:]) // 2]
    res[name] = x
    print(f"{name}: {steady:.1f} ms per 4-latent chunk (16 frames), steady median")

net = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()


def psnr(a, b):
    mse = ((a - b) ** 2).mean(dim=(0, 2, 3))
    return (10 * torch.log10(4 / mse)).mean().item()   # frames in [-1, 1]: peak-to-peak 2


def lp(a, b, step=4):
    return float(np.mean([net(a[:, i][None], b[:, i][None]).item() for i in range(0, a.shape[1], step)]))


src = video[:, :res["full Wan VAE"].shape[1]]
L, F_ = res["LightVAE (dim 24)"], res["full Wan VAE"]
print(f"LightVAE vs full VAE (same latents): PSNR {psnr(L, F_):.2f} dB, LPIPS {lp(L, F_):.4f}")
print(f"full VAE vs source video:            PSNR {psnr(F_, src):.2f} dB, LPIPS {lp(F_, src):.4f}")
print(f"LightVAE vs source video:            PSNR {psnr(L, src):.2f} dB, LPIPS {lp(L, src):.4f}")
side = torch.cat([F_[:, 60], L[:, 60]], dim=2)        # frame 60, full | light
imageio.imwrite("/workspace/runs/lightvae_frame60_full_left_light_right.png",
                ((side.permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy())
