#!/usr/bin/env bash
# podcheck.sh — 30-second health check of a fresh GPU box before installing anything on it.
# Prints one line per check and a verdict. Needs only nvidia-smi and a torch with CUDA (the RunPod base image has both).
#
#   bash tools/podcheck.sh            # RTX 5090 defaults
#   MIN_TFLOPS=150 bash tools/podcheck.sh
#
# Checks, and why:
#   clock/throttle  the card can be power-capped by the host (seen 2026-09-22: 1.9 GHz, 314 W, sw_power_cap, PCIe 4 —
#                   every kernel 1.7x slower); a bf16 matmul under load shows it in 3 s. Not fixable from inside a pod.
#   counters        Nsight Compute needs NVreg_RestrictProfilingToAdminUsers=0 on the host; RunPod containers never have it.
#                   Profiling is impossible there, everything else works.
#   disk            /workspace needs ~30 GB (18 GB weights + venv + compile cache).
#   cuInit          a broken driver returns cuInit 999 (seen twice on 2026-09-15/16).
#   power cap       power.limit below the card's default (Vast 2x 5090, 2026-10-05: 500 of 575 W) lowers sustained clocks.
#   cpu             PyTorch pipelines are often launch-bound: one Python thread issues every kernel. An engineering-sample
#                   Xeon took ~15 us per eager op (2026-10-05) and capped the two-GPU split at 17.6 FPS; a container CPU
#                   quota (cgroup cpu.max) throttles compile and data loading. Measured as us per tiny eager op.
#   multi-gpu       per-GPU matmul (one slow card sets the pace), P2P access, topology (SYS = across CPU sockets) and
#                   pinned host<->card bandwidth (with P2P off all card-to-card traffic stages through host memory).
set -u
MIN_TFLOPS="${MIN_TFLOPS:-170}"   # RTX 5090 bf16 dense measures ~195-200 TFLOP/s at its ~2.8 GHz boost; 170 leaves margin
PY="${PY:-python3}"
ok=1
REPO_DIR=$(ls -d /workspace/lingbot-world-v2-realtime /workspace/repo 2>/dev/null | head -1)  # RunPod / Vast layouts

echo "== gpu"
nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.current,pcie.link.gen.max,power.limit,clocks.max.sm --format=csv,noheader || { echo "nvidia-smi failed"; exit 1; }

echo "== cuInit"
$PY - <<'EOF' || ok=0
import torch, sys
try:
    torch.cuda.init(); x = torch.ones(1, device="cuda"); torch.cuda.synchronize(); print("cuda ok,", torch.cuda.get_device_name(0))
except Exception as e:
    print("CUDA BROKEN:", e); sys.exit(1)
EOF

echo "== throttle (bf16 matmul under load, every GPU)"
$PY - "$MIN_TFLOPS" <<'EOF' || ok=0
import torch, time, subprocess, sys
minf, bad = float(sys.argv[1]), False
for i in range(torch.cuda.device_count()):
    d = torch.device(f"cuda:{i}")
    a = torch.randn(8192, 8192, device=d, dtype=torch.bfloat16); b = torch.randn_like(a)
    for _ in range(5): a @ b
    torch.cuda.synchronize(d); t = time.time(); n = 60
    for _ in range(n): a @ b
    q = subprocess.run(["nvidia-smi", "-i", str(i), "--query-gpu=clocks.sm,power.draw,clocks_throttle_reasons.active,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.hw_thermal_slowdown",
                        "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    torch.cuda.synchronize(d); dt = time.time() - t
    tf = 2 * 8192 ** 3 * n / dt / 1e12
    print(f"gpu {i}: bf16 matmul {tf:.0f} TFLOP/s (need >= {minf:.0f}); under load: sm clock, power, throttle reasons = {q}")
    bad |= tf < minf
    del a, b
if bad:
    print("THROTTLED: do not benchmark on this box")
    sys.exit(1)
EOF

echo "== power cap"
nvidia-smi --query-gpu=index,power.limit,power.default_limit --format=csv,noheader,nounits | while IFS=', ' read -r i lim def; do
  if awk "BEGIN{exit !($lim < $def - 1)}"; then
    echo "gpu $i: CAPPED at ${lim} W (default ${def} W): lower sustained clocks; record it next to every result"
  else
    echo "gpu $i: power limit ${lim} W (default ${def} W)"
  fi
done

echo "== cpu (launch speed: PyTorch inference is often launch-bound)"
echo "model: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //'); $(nproc) usable cores; $(grep 'physical id' /proc/cpuinfo | sort -u | wc -l) socket(s)"
quota=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo max)
case "$quota" in
  max*) echo "cgroup cpu quota: none" ;;
  *) echo "cgroup cpu quota: $(echo "$quota" | awk '{printf "%.1f", $1/$2}') cores (cpu.max = $quota): parallel compile and data loading are throttled" ;;
esac
$PY - <<'EOF'
import torch, time
x = torch.ones(16, device="cuda")
for _ in range(2000): x.add_(1)
torch.cuda.synchronize(); n = 20000; t = time.perf_counter()
for _ in range(n): x.add_(1)
us = (time.perf_counter() - t) / n * 1e6
torch.cuda.synchronize()
print(f"{us:.1f} us per tiny eager op (fast desktop/server CPUs: ~4-7 us)")
if us > 10:
    print("SLOW CPU LAUNCH: launch-bound runs (multi-GPU splits, eager loops) are capped by the CPU on this box and are not "
          "comparable to a faster host; CUDA graphs and compiled regions matter more here")
EOF

echo "== multi-gpu"
$PY - <<'EOF'
import torch, time
n = torch.cuda.device_count()
if n < 2:
    print("single GPU: skipped"); raise SystemExit
for i in range(n):
    for j in range(i + 1, n):
        print(f"P2P {i}<->{j}: " + ("yes" if torch.cuda.can_device_access_peer(i, j) else "NO: card-to-card traffic stages through host memory"))
h = torch.empty(256 << 20, dtype=torch.uint8, pin_memory=True)
for i in range(n):
    d = torch.empty(256 << 20, dtype=torch.uint8, device=f"cuda:{i}")
    gbs = []
    for src, dst in ((h, d), (d, h)):
        dst.copy_(src, non_blocking=True); torch.cuda.synchronize(i); t = time.perf_counter()
        for _ in range(5): dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize(i); gbs.append(5 * (256 << 20) / (time.perf_counter() - t) / 1e9)
    print(f"gpu {i}: pinned host->card {gbs[0]:.1f} GB/s, card->host {gbs[1]:.1f} GB/s (PCIe 5 x16 ~50, gen 4 ~25)")
EOF
nvidia-smi topo -m 2>/dev/null | head -n "$(( $(nvidia-smi -L | wc -l) + 1 ))"
echo "(SYS = GPUs on different CPU sockets; NODE/PHB/PIX = same socket)"

echo "== counters (Nsight Compute)"
if command -v ncu >/dev/null 2>&1; then
  cat > /tmp/podcheck_k.cu <<'EOF'
__global__ void k(float* a){ a[threadIdx.x] *= 2.f; }
int main(){ float* d; cudaMalloc(&d, 1024); k<<<1,256>>>(d); cudaDeviceSynchronize(); return 0; }
EOF
  NVCC=$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)
  if $NVCC -o /tmp/podcheck_k /tmp/podcheck_k.cu >/dev/null 2>&1 && ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed /tmp/podcheck_k 2>&1 | grep -q "sm__throughput"; then
    echo "ncu ok: hardware counters readable"
  else
    echo "ncu unavailable (ERR_NVGPUCTRPERM or no nvcc): profiling not possible here, benchmarking is"
  fi
else
  echo "ncu not installed"
fi

echo "== disk"
df -h /workspace 2>/dev/null | tail -1 || df -h / | tail -1
free_gb=$(df -BG /workspace 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4}')
# The ~30 GB rule is for a FRESH setup (18 GB weights + venv + compile cache). On an already-provisioned
# pod the weights are on disk already and a smaller margin is fine, so warn instead of failing: on
# 2026-09-22 a healthy 5090 (231 TFLOP/s, no throttle) was reported "red" purely on 13 GB free.
if [ -n "${free_gb:-}" ] && [ "$free_gb" -lt 30 ]; then
  if [ -n "$REPO_DIR" ]; then
    echo "NOTE: ${free_gb} GB free, under the 30 GB fresh-setup rule, but the repo is already provisioned - not a blocker"
  else
    echo "LOW DISK: ${free_gb} GB free, need ~30 for a fresh setup"; ok=0
  fi
fi

echo "== network (10 s from Hugging Face)"
# 2026-10-05: a 2x 5090 host downloaded at ~12 MB/s, so the 15 GB of weights took ~20 min.
# Informational only: a slow network does not affect measurements, only setup time.
mbps=$(curl -sL -o /dev/null --max-time 10 -w "%{speed_download}" \
  "https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-fast/resolve/main/models_t5_umt5-xxl-enc-bf16.pth" \
  2>/dev/null | awk '{printf "%.0f", $1/1e6}')
if [ -n "$mbps" ] && [ "$mbps" -gt 0 ]; then
  echo "download ${mbps} MB/s (one connection); full weights (~15 GB) take ~$(( 15000 / (mbps * 3 + 1) / 60 + 1 )) min with parallel downloads"
  [ "$mbps" -lt 20 ] && echo "SLOW NETWORK: setup.sh will spend a long time on the weights; reuse a volume that has them if you can"
else
  echo "download test failed (no network or gated file)"
fi

echo "== existing state"
[ -n "$REPO_DIR" ] && echo "repo present at $REPO_DIR: skip setup.sh" || echo "empty: full setup (~15 min)"

echo
if [ "$ok" = 1 ]; then echo "VERDICT: green, proceed"; else echo "VERDICT: red, do not use this pod for measurements"; exit 1; fi
