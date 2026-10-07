#!/usr/bin/env python3
"""CPU fp32 check of the sub-pixel upsample rewrite (LINGBOT_VAE_SUBPIXEL=1 in lingbot/models/lingbot_world/vae.py):
each upsample `Resample` alone vs the stock nearest-exact + 3x3 conv, the whole fused decoder with and
without the flag on the harness of test_vae_fused_cpu.py, and the conv FLOPs at the real shapes.
Run: ~/lingbot-world-bench/.venv/bin/python tests/test_vae_subpixel_cpu.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_vae_fused_cpu import build, check, fused, stock_decode  # noqa: E402  (sets the OMP env, loads the VAE modules)
import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

Resample = sys.modules["wan.modules.vae2_1"].Resample
STAGES = [(384, 58, 104), (384, 116, 208), (192, 232, 416)]  # (Cin, H, W) of the three upsample convs, Cout = Cin // 2


def flops(fn, *a):
    with FlopCounterMode(display=False) as fc:
        fn(*a)
    return fc.get_total_flops()


def main():
    oks = []
    torch.manual_seed(0)
    for mode, dim, h, w in [('upsample2d', 8, 5, 7), ('upsample3d', 8, 1, 1), ('upsample2d', 6, 2, 3), ('upsample3d', 16, 9, 13)]:
        rs = Resample(dim, mode).eval()
        x = torch.randn(3, dim, h, w)
        with torch.no_grad():
            ref = rs.resample(x)
            out = fused._subpixel_conv(x, *fused._subpixel_weight(rs.resample[1]))
        oks.append(check(f"{mode} dim {dim} {h}x{w} sub-pixel vs stock", out, ref))

    vae = build()
    z = torch.randn(16, 5, 8, 16)  # 5 latents -> 17 frames
    with torch.no_grad():
        ref = stock_decode(vae, z)
        # channels_last=False: CPU conv kernels sum in a different order per layout (~2e-5), see test_vae_fused_cpu
        plain = fused.FusedDecoder(build(), channels_last=False, dtype=torch.float32, subpixel=False).decode(z)
        os.environ["LINGBOT_VAE_SUBPIXEL"] = "1"
        dec = fused.FusedDecoder(build(), channels_last=False, dtype=torch.float32)  # flag from the env
        assert all(hasattr(m, "sub_weight") for m in dec.dec.modules() if isinstance(m, Resample)), "env flag not honoured"
        sub = dec.decode(z)
    oks.append(check("fused decoder, subpixel vs stock", sub, ref))
    oks.append(check("fused decoder, subpixel vs fused without flag", sub, plain))

    print("conv FLOPs per frame (stock = nearest-exact 2x + 3x3 conv, sub-pixel = 2x2 conv on the source + pixel shuffle):")
    tot = [0, 0]
    for cin, h, w in STAGES:
        rs = Resample(cin, 'upsample2d').eval()
        x = torch.randn(1, cin, h, w)
        wt, b = fused._subpixel_weight(rs.resample[1])
        with torch.no_grad():
            a, s = flops(rs.resample, x), flops(fused._subpixel_conv, x, wt, b)
        tot[0] += a; tot[1] += s
        print(f"  {cin}->{cin // 2} {h}x{w} -> {2 * h}x{2 * w}: {a / 1e9:.2f} -> {s / 1e9:.2f} GFLOP ({a / s:.2f}x fewer)")
    print(f"  three stages: {tot[0] / 1e9:.2f} -> {tot[1] / 1e9:.2f} GFLOP per frame, x16 frames per chunk: "
          f"{tot[0] * 16 / 1e12:.2f} -> {tot[1] * 16 / 1e12:.2f} TFLOP (-{(tot[0] - tot[1]) * 16 / 1e12:.2f})")
    print("ALL OK" if all(oks) else "FAILED"); sys.exit(0 if all(oks) else 1)


if __name__ == "__main__":
    main()
