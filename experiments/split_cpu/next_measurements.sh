#!/usr/bin/env bash
# The two minimal measurements from TODO.md (top item), on the pod, ~15 min. Prints a verdict per measurement.
#   bash experiments/split_cpu/next_measurements.sh            # from /workspace/repo, old stack (.venv)
# 1. Is card 0's SageAttention wave tail real?   (kernel only, ~2 min)
# 2. Token share 0.7454: card 1's GEMMs land on whole waves (1,536 tokens = 12 tiles = 144 blocks = 3 waves on 48 SMs).
#    Compared against the same config at the head ratio (0.833). 3 warm runs each (~10 min).
set -u
cd "$(dirname "$0")/../.."
. .venv/bin/activate
export PATH=/usr/local/cuda/bin:$PATH TORCHINDUCTOR_CACHE_DIR=$PWD/.inductor_cache
O=/workspace/runs/next; mkdir -p $O

echo "== 1. attention wave tail (GPU 0, kernel only) $(date +%T)"
python experiments/split_cpu/attn_wave_tail.py | tee $O/wave_tail.txt
SO=$(python -c "import sageattention, glob, os; print(glob.glob(os.path.join(os.path.dirname(sageattention.__file__), '_qattn_sm89*.so'))[0])")
echo "registers of the attention kernel(s) in $(basename $SO):"
cuobjdump --dump-resource-usage "$SO" 2>/dev/null | grep -A1 "qk_int_sv_f8_attn_kernel" | grep -o "REG:[0-9]*" | sort | uniq -c | tee -a $O/wave_tail.txt

echo "== 2. token share at whole GEMM waves on card 1 $(date +%T)"
P="$(cat examples/03/prompt.txt)"; A=(--preset fast --frame_num 157 --image examples/03/image.jpg --action_path examples/03 --prompt "$P" --decoder_gpu 1 --bench_e2e)
f() { python -m lingbot.worker run -- "${A[@]}" --save_file $O/$1.mp4 > $O/$1.log 2>&1; grep -oE "[0-9.]+ FPS \(interval p50 [0-9.]+ ms" $O/$1.log | head -1; }
med() { sort -n | sed -n 2p; }
run() { tag=$1; shift; python -m lingbot.worker stop >/dev/null; ( export "$@"; f ${tag}_cold >/dev/null; for r in 1 2 3; do f ${tag}_$r; done; python -m lingbot.worker stop >/dev/null ) | tee $O/$tag.txt; }
export LINGBOT_SPLIT=10:2 LINGBOT_SPLIT_CPPWRAP=1 LINGBOT_SPLIT_SKIPGUARD=50 LINGBOT_SPLIT_PIECES=2 LINGBOT_SPLIT_SMS=48 LINGBOT_SPLIT_ZEROCOPY=1
run head_ratio LINGBOT_SPLIT_TOK=0
run tok7454 LINGBOT_SPLIT_TOK=0.7454
a=$(grep -oE "^[0-9.]+" $O/head_ratio.txt | med); b=$(grep -oE "^[0-9.]+" $O/tok7454.txt | med)
python experiments/a2a8/score_pairs.py --dir $O --pairs tok7454_1:head_ratio_1 > $O/q.txt 2>&1; grep -v -i warn $O/q.txt | tail -2
echo "median FPS: head ratio $a, share 0.7454 $b"
python - "$a" "$b" <<'EOF'
import sys
a, b = map(float, sys.argv[1:])
d = 100 * (b / a - 1)
print(f"share 0.7454 vs head ratio: {d:+.1f} %  ->  " + (
    "card 1 can take token-local work: sweep around 0.7454 (whole-wave points only)" if d > 0.5 else
    "still a loss: capture tok=0.80 (LINGBOT_WORKER_NSYS=1 LINGBOT_NVTX=1 LINGBOT_NSYS_CHUNKS=5:9) and read card 1's GEMM grids"))
EOF
echo "results in $O"
