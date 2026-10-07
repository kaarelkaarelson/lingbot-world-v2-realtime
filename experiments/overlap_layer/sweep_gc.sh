#!/usr/bin/env bash
# Green-context sweep: split ratio x SMs for card 1's DiT share (rest to the decoder). One process per config.
cd /workspace/repo && source .venv/bin/activate
export PATH=/usr/local/cuda/bin:$PATH TORCHINDUCTOR_CACHE_DIR=/workspace/repo/.inductor_cache PIECES=4 HIPRIO=1
for h in 10 9; do
  for gc in 0 32 40 48 64; do
    echo "== H0=$h GC=$gc"
    H0=$h GC=$gc python experiments/with_decoder.py 2>&1 | grep -E "partitioned|split alone|split with|^  card|^  decoder|^chunk"
  done
done
echo "SWEEP DONE"
