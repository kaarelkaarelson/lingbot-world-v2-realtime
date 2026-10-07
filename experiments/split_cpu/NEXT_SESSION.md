# Next session: pick up here

State (2026-10-06): best split 29.8-29.9 FPS vs B1 25.3 on Vast 54497707 (EPYC 9454, 2x RTX 5090, PCIe 5.0 x16).
Config: `LINGBOT_SPLIT=10:2 LINGBOT_SPLIT_SMS=48 LINGBOT_SPLIT_CPPWRAP=1 LINGBOT_SPLIT_SKIPGUARD=50
LINGBOT_SPLIT_PIECES=2 LINGBOT_SPLIT_ZEROCOPY=1`, torch 2.8 + cu128 (`.venv`), `--decoder_gpu 1`. Card 0's kernels
set the chunk time (519 of 541 ms); card 1 is ~26 % idle. What was tried and why it failed: learnings 21-28 in
`2X_RTX5090_LEARNINGS.md`.

## 1. Bring a pod up (instance 54497707 was destroyed on 2026-10-06: start from a new one, ~20-30 min)

Rent a new pod per the `vast-pod-guide` skill (EPYC 9454-class CPU, PCIe 5.0 x16, one socket, template image
`vastai/pytorch:cuda-12.8.1-auto`, >= 45 GB disk; the Norway host m:53317 was the best seen). Then on the pod:
`git clone -b refactor/structure https://github.com/kaarelkaarelson/lingbot-world-v2-realtime.git /workspace/repo`,
`bash tools/podcheck.sh`, `HF_TOKEN=... bash setup.sh`, then install the patched Sage wheel from the Mac (`~/lingbot-world-v2-realtime-wt/wheels/sageattention-2.2.0-1currentstream-*.whl`),
`pip install "cuda-bindings==12.8.*" lpips`, and regenerate the B1 reference video (`runs/s0/b1_1.mp4`) with one B1 run.

## 2. Run the three measurements (~20 min, one command)

```bash
bash experiments/split_cpu/next_measurements.sh
```

| Measurement | Question | Verdict printed | Decides |
|---|---|---|---|
| 1. `attn_wave_tail.py` + kernel registers | Is card 0's attention wave tail real? Attention kernel alone at 6-12 heads on one full GPU | `t(10)/t(7)`: ~2.0 = tail real, ~1.43 = no tail | whether more attention work (split-KV on cu128, head splits) is worth it |
| 2. token share 0.7454 vs head ratio | Did the token-share sweep lose to card 1's GEMM wave quantization? At 0.7454 card 1's GEMMs launch 144 blocks = exactly 3 waves on 48 SMs | `+x %` and what to do next | whether card 1's idle ~26 % is reachable by moving token-local work |
| 3. classic split 6:6 with all the 10:2 fixes | The README's 14.0 FPS used the first split code on the slow pod: what does the classic layout reach now? Decoder after the DiT on card 0 | median FPS (expected ~21, still below B1) | replaces the 14.0 in the README layout table |

Results land in `/workspace/runs/next/`; copy them into `experiments/split_cpu/results/` and log FPS in
`results/fps_log_2026-10-05.tsv`.

## 3. Depending on the verdicts

- Tail real and 0.7454 wins: sweep token shares at whole-wave points only, then combine with attention work.
- Tail real, 0.7454 loses: capture the 0.80 run (`LINGBOT_WORKER_NSYS=1 LINGBOT_NVTX=1 LINGBOT_NSYS_CHUNKS=5:9`),
  open it with `python tools/nsys_to_perfetto.py <sqlite> --sms 1:24=48,1:27=122 --flops experiments/split_cpu/flops_split_10_2.json --open`
  and read card 1's GEMM grids and waves (click a GEMM).
- No tail: the card 0 gap is elsewhere; next is the per-kernel breakdown of the ~200 ms of small kernels on card 0
  (existing capture `~/lingbot-world-v2-realtime-wt/traces/split_zc_warm.sqlite`, no pod needed).

## Already done without a pod: card 0's small kernels (2026-10-06)

From the copy-free capture (`traces/split_zc_warm.sqlite`, 4 warm chunks): everything on card 0's DiT stream except
the FP8 GEMMs, SageAttention's attention kernel and cross-attention is **124 ms per chunk in 3,877 launches**. Top items:

| ms / chunk | launches | Kernel | What it is |
|---|---|---|---|
| 16.8 | 150 | TransposePadPermuteKernel | Sage: V transpose + pad |
| 15.8 | 150 | triton_poi_fused_add_mul_neg_sub_2 | RoPE (compensated fp32) |
| 13.7 | 150 | triton_poi_fused_stack_3 | stacking q\|k\|v for the exchange |
| 13.3 | 150 | triton_red_fused__scaled_mm_..._amax_clamp_div_7 | FP8 activation quantization (a GEMM input) |
| 9.8 | 300 | QuantInt8Kernel | Sage: Q/K int8 quantization |
| 9.0 | 150 | triton_red_fused__scaled_mm_..._native_layer_norm | norm + modulation + FP8 quantization |
| 7.8 | 150 | reduce_kernel | Sage: K mean (smoothing) |
| 7.6 | 248 | elementwise_kernel | copies |
| 6.6 | 150 | MeanScaleKernel | Sage: V scale |
| 5.1 | 150 | triton_poi_fused_add_mul_8 | residual / modulation |

Sage's own prep (V transpose 16.8 + Q/K quant 9.8 + K mean 7.8 + V scale 6.6) is **~41 ms per chunk**, the largest
fusion target; RoPE + q|k|v stacking (~29 ms) is the second. Byte-level rooflines for these are not done yet.

## Still open, not scheduled

- Late-chunk divergence between runs in the same worker (B1): determinism test (`torch.use_deterministic_algorithms`,
  decoder on the same card, synchronous) to tell a race from a nondeterministic kernel.
- Quality beyond the first chunk, n >= 3.
- Same configs on a third host.
- Stop the pod when done (`vastai stop instance 54497707`).
