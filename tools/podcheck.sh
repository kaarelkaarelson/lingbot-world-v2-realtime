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
set -u
MIN_TFLOPS="${MIN_TFLOPS:-170}"   # RTX 5090 bf16 dense measures ~195-200 TFLOP/s at its ~2.8 GHz boost; 170 leaves margin
PY="${PY:-python3}"
ok=1

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

echo "== throttle (bf16 matmul under load)"
$PY - "$MIN_TFLOPS" <<'EOF' || ok=0
import torch, time, subprocess, sys
minf = float(sys.argv[1])
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); b = torch.randn_like(a)
for _ in range(5): a @ b
torch.cuda.synchronize(); t = time.time(); n = 60
for _ in range(n): a @ b
torch.cuda.synchronize(); dt = time.time() - t
tf = 2 * 8192 ** 3 * n / dt / 1e12
q = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,clocks_throttle_reasons.active,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.hw_thermal_slowdown",
                    "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
print(f"bf16 matmul {tf:.0f} TFLOP/s (need >= {minf:.0f}); under load: sm clock, power, throttle reasons = {q}")
if tf < minf:
    print("THROTTLED: do not benchmark on this box")
    sys.exit(1)
EOF

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
  if [ -d /workspace/lingbot-world-v2-realtime ]; then
    echo "NOTE: ${free_gb} GB free, under the 30 GB fresh-setup rule, but the repo is already provisioned - not a blocker"
  else
    echo "LOW DISK: ${free_gb} GB free, need ~30 for a fresh setup"; ok=0
  fi
fi

echo "== existing state"
ls -d /workspace/lingbot-world-v2-realtime 2>/dev/null && echo "repo present: skip setup.sh" || echo "empty: full setup (~15 min)"

echo
if [ "$ok" = 1 ]; then echo "VERDICT: green, proceed"; else echo "VERDICT: red, do not use this pod for measurements"; exit 1; fi
