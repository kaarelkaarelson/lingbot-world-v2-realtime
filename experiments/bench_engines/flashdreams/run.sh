#!/bin/bash
# run.sh <scene 00-05> [total_blocks]  — headless benchmark, NVIDIA's own --stats-path path (CUDA-event-synced per-block
# encode/diffuse/decode/finalize ms). Static camera: the Cam2V app does not replay poses.npy (perf-equivalent to a moving
# one, pixels differ). 23 blocks at len_t=4 = 13 + 22*16 = 365 frames.
set -euo pipefail
SCENE=${1:-03}; BLOCKS=${2:-23}
FD=${FD:-/workspace/fd}; E=${EXAMPLES:-/workspace/lingbot-world-v2-realtime/examples}/$SCENE
export HF_HOME=${HF_HOME:-$FD/hf} FLASHDREAMS_MIN_CACHE_FREE_GB=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # xet needs ~2x the 22.7 GB text encoder transiently
cd "$FD/flashdreams"; mkdir -p out
~/.local/bin/uv run --no-sync flashdreams-run-v2 cam2v-lingbot-1p3b \
  --output-path out/1p3b_$SCENE.mp4 --stats-path out/1p3b_${SCENE}_stats.json -- \
  --image-path "$E/image.jpg" --pose-path "$E/poses.npy" --intrinsic-path "$E/intrinsics.npy" \
  --prompt "$(cat "$E/prompt.txt")" --no-ui --total-blocks "$BLOCKS" --warmup-blocks 5 --seed 42 2>&1 | tee out/1p3b_$SCENE.log
grep -E "Cam2V AR|encode .* diffuse" out/1p3b_$SCENE.log | sed 's/.*Cam2V AR/AR/; s/.*encode/encode/' | tail -6
