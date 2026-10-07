#!/usr/bin/env bash
# GPU-to-GPU communication benchmark for a 2-GPU pod. Results land in ./linkbench_out/.
# Usage: bash tools/linkbench/run.sh      (needs torch with CUDA, 2 visible GPUs)
set -u
D=$(cd "$(dirname "$0")" && pwd); OUT=${OUT:-linkbench_out}; mkdir -p "$OUT"
nvidia-smi -L > "$OUT/gpus.txt"; nvidia-smi topo -m >> "$OUT/gpus.txt"
python3 "$D/single.py" > "$OUT/single.tsv"
: > "$OUT/nccl.tsv"
run() { v=$1; shift; env VARIANT="$v" "$@" torchrun --nproc_per_node=2 "$D/nccl.py" 2>/dev/null >> "$OUT/nccl.tsv"; }
run default
run algo_ring NCCL_ALGO=Ring
run algo_tree NCCL_ALGO=Tree
run proto_LL NCCL_PROTO=LL
run proto_LL128 NCCL_PROTO=LL128
run proto_Simple NCCL_PROTO=Simple
run shm_cuda_memcpy NCCL_SHM_USE_CUDA_MEMCPY=1
run channels_16 NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16
run shm_disabled_socket NCCL_SHM_DISABLE=1
VARIANT=gloo BENCH_BACKEND=gloo torchrun --nproc_per_node=2 "$D/nccl.py" 2>/dev/null >> "$OUT/nccl.tsv"
python3 "$D/staged_a2a.py" > "$OUT/staged.tsv"
torchrun --nproc_per_node=2 "$D/nccl_overlap.py" 2>/dev/null > "$OUT/overlap.txt"
NCCL_DEBUG=INFO torchrun --nproc_per_node=2 "$D/nccl.py" 2>&1 | grep -m3 "via \|NCCL version" > "$OUT/nccl_transport.txt"
echo "done: $OUT"
