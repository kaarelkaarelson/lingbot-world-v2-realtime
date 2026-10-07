#!/bin/bash
# bench.sh [scene 00-05] — headless LightX2V lingbot_world_fast run on examples/<scene>, then the LX summary.
# Camera: --action_path <dir> reads poses.npy + intrinsics.npy (wan_runner.py _build_lingbot_dit_cond_dict);
# the 269 poses of examples/03 are resampled onto the 88 output latents (upstream truncates to 269 frames instead).
# Timing: PROFILING_DEBUG_LEVEL=2 makes every ProfilingContext cuda.synchronize() and print (level 2 only adds
#   Load models / Run Encoders / Run DiT, all outside the chunk loop)
#   [Profile] ... chunk end2end k/N cost X seconds      DiT per chunk: 4 denoise forwards + 1 t=0 cache-write forward
#   [AsyncVAEChunkDecoder] sync VAE chunk k/k decode=X  Wan 2.1 VAE causal decode of that chunk (13 frames, then 16)
# ATTN=torch_sdpa|flash_attn2|sage_attn2 rewrites the three *_attn_*_type keys (all three; sage_attn2 is int8-QK, not lossless).
set -euo pipefail
SCENE=${1:-03}; ATTN=${ATTN:-torch_sdpa}
LX=${LX:-/workspace/lx}; E=${EXAMPLES:-/workspace/lingbot-world-v2-realtime/examples}/$SCENE
HERE=$(cd "$(dirname "$0")" && pwd); P="$LX/.venv/bin/python"; OUT="$LX/out"; mkdir -p "$OUT"
CFG="$OUT/config_${SCENE}_${ATTN}.json"; LOG="$OUT/lx_${SCENE}_${ATTN}.log"; MP4="$OUT/out.mp4"
export PROFILING_DEBUG_LEVEL=2 DTYPE=BF16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$P" - "$HERE/config_1p3b.json" "$CFG" "$ATTN" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
for k in ("self_attn_1_type", "cross_attn_1_type", "cross_attn_2_type"):
    cfg[k] = sys.argv[3]
json.dump(cfg, open(sys.argv[2], "w"), indent=4)
PY
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/gpu: /'
"$P" - "$LX/model" "$CFG" "$E" "$MP4" <<'PY' 2>&1 | tee "$LOG"
import sys, torch
model, cfg, ex, mp4 = sys.argv[1:5]
sys.argv = ["lightx2v.infer", "--model_cls", "lingbot_world_fast", "--task", "i2v", "--model_path", model, "--config_json", cfg,
            "--image_path", f"{ex}/image.jpg", "--action_path", ex, "--prompt", open(f"{ex}/prompt.txt").read().strip(),
            "--seed", "42", "--save_result_path", mp4]
from lightx2v.infer import main
main()
print(f"LXRAW peak_vram_allocated_gb={torch.cuda.max_memory_allocated() / 2**30:.2f} peak_vram_reserved_gb={torch.cuda.max_memory_reserved() / 2**30:.2f}", flush=True)
PY
"$P" - "$LOG" "$MP4" "$ATTN" "$SCENE" <<'PY'
import re, statistics, sys
import av
log, mp4, attn, scene = sys.argv[1:5]
text = open(log).read()
dit = {int(k): float(s) for k, _, s in re.findall(r"chunk end2end (\d+)/(\d+) cost ([\d.]+) seconds", text)}
dec = {int(k): float(s) for k, s in re.findall(r"sync VAE chunk (\d+)/\d+ decode=([\d.]+) seconds", text)}
loop = re.search(r"AR chunk total (\d+) chunks cost ([\d.]+) seconds", text)
enc = re.search(r"Run Encoders cost ([\d.]+) seconds", text)
total = re.search(r"Total Cost cost ([\d.]+) seconds", text)
vram = re.search(r"LXRAW (peak_vram_allocated_gb=[\d.]+ peak_vram_reserved_gb=[\d.]+)", text)
assert dit and dec and loop, "no [Profile] chunk lines in the log — did the run finish? PROFILING_DEBUG_LEVEL must be >= 1"
with av.open(mp4) as c:
    v = c.streams.video[0]; w, h = v.width, v.height
    frames = v.frames or sum(1 for _ in c.decode(v))
chunks = sorted(dit)
per = {k: dit[k] + dec.get(k, float("nan")) for k in chunks}
steady = [k for k in chunks if k >= 6]
med = lambda xs: statistics.median(xs)
m_dit, m_dec, m_chunk = med([dit[k] for k in steady]), med([dec[k] for k in steady]), med([per[k] for k in steady])
print(f"LX engine=LightX2V@69018c9 attn={attn} scene={scene} res={w}x{h} frames_out={frames} chunks={len(chunks)} dit_forwards_per_chunk=5 vae=wan2.1_bf16_per_chunk")
for k in chunks:
    print(f"LX chunk {k:2d} dit={dit[k]:.3f}s dec={dec.get(k, float('nan')):.3f}s chunk={per[k]:.3f}s")
print(f"LX first_chunk dit={dit[1]:.3f}s dec={dec[1]:.3f}s chunk={per[1]:.3f}s (13 frames)")
print(f"LX steady chunks>={steady[0]} n={len(steady)} median dit={m_dit:.3f}s dec={m_dec:.3f}s chunk={m_chunk:.3f}s")
print(f"LX fps_as_played={16 / m_chunk:.2f} (16/median chunk)  fps_dit_only={16 / m_dit:.2f} (16/median dit)")
print(f"LX loop_total={float(loop.group(2)):.1f}s for {loop.group(1)} chunks (DiT+decode, sync)  encoders={enc.group(1) if enc else 'n/a'}s  total_cost={total.group(1) if total else 'n/a'}s")
print(f"LX {vram.group(1) if vram else 'peak_vram=n/a'}  video={mp4}  log={log}")
PY
