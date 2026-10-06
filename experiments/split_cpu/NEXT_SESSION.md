# Next session: pick up here

State (2026-10-06): best split 29.8-29.9 FPS vs B1 25.3 on Vast 54497707 (EPYC 9454, 2x RTX 5090, PCIe 5.0 x16).
Config: `LINGBOT_SPLIT=10:2 LINGBOT_SPLIT_SMS=48 LINGBOT_SPLIT_CPPWRAP=1 LINGBOT_SPLIT_SKIPGUARD=50
LINGBOT_SPLIT_PIECES=2 LINGBOT_SPLIT_ZEROCOPY=1`, torch 2.8 + cu128 (`.venv`), `--decoder_gpu 1`. Card 0's kernels
set the chunk time (519 of 541 ms); card 1 is ~26 % idle. What was tried and why it failed: learnings 21-28 in
`2X_RTX5090_LEARNINGS.md`.

## 1. Bring a pod up (instance 54497707 was destroyed on 2026-10-06: start from a new one, ~20-30 min)

```bash
vastai start instance 54497707          # see the vast-pod-guide skill; IP/port: vastai show instance 54497707 --raw
ssh -p <port> root@<ip>
cd /workspace/repo && git pull
```

If the instance is gone: new pod per the `vast-pod-guide` skill (EPYC 9454-class CPU, PCIe 5.0 x16, `cuda-12.8.1-auto`),
`setup.sh`, then install the patched Sage wheel from the Mac (`~/lingbot-world-v2-realtime-wt/wheels/sageattention-2.2.0-1currentstream-*.whl`),
`pip install "cuda-bindings==12.8.*" lpips`, and regenerate the B1 reference video (`runs/s0/b1_1.mp4`) with one B1 run.

## 2. Run the two measurements (~15 min, one command)

```bash
bash experiments/split_cpu/next_measurements.sh
```

| Measurement | Question | Verdict printed | Decides |
|---|---|---|---|
| 1. `attn_wave_tail.py` + kernel registers | Is card 0's attention wave tail real? Attention kernel alone at 6-12 heads on one full GPU | `t(10)/t(7)`: ~2.0 = tail real, ~1.43 = no tail | whether more attention work (split-KV on cu128, head splits) is worth it |
| 2. token share 0.7454 vs head ratio | Did the token-share sweep lose to card 1's GEMM wave quantization? At 0.7454 card 1's GEMMs launch 144 blocks = exactly 3 waves on 48 SMs | `+x %` and what to do next | whether card 1's idle ~26 % is reachable by moving token-local work |

Results land in `/workspace/runs/next/`; copy them into `experiments/split_cpu/results/` and log FPS in
`results/fps_log_2026-10-05.tsv`.

## 3. Depending on the verdicts

- Tail real and 0.7454 wins: sweep token shares at whole-wave points only, then combine with attention work.
- Tail real, 0.7454 loses: capture the 0.80 run (`LINGBOT_WORKER_NSYS=1 LINGBOT_NVTX=1 LINGBOT_NSYS_CHUNKS=5:9`),
  open it with `python tools/nsys_to_perfetto.py <sqlite> --sms 1:24=48,1:27=122 --flops experiments/split_cpu/flops_split_10_2.json --open`
  and read card 1's GEMM grids and waves (click a GEMM).
- No tail: the card 0 gap is elsewhere; next is the per-kernel breakdown of the ~200 ms of small kernels on card 0
  (existing capture `~/lingbot-world-v2-realtime-wt/traces/split_zc_warm.sqlite`, no pod needed).

## Still open, not scheduled

- Late-chunk divergence between runs in the same worker (B1): determinism test (`torch.use_deterministic_algorithms`,
  decoder on the same card, synchronous) to tell a race from a nondeterministic kernel.
- Quality beyond the first chunk, n >= 3.
- Same configs on a third host.
- Stop the pod when done (`vastai stop instance 54497707`).
