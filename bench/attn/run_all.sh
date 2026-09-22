#!/usr/bin/env bash
# Runs each attention hypothesis bench in its own process; a failure does not stop the rest.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
RESULTS="$HERE/results"
mkdir -p "$RESULTS"
cd "$REPO"
if [ -f "$REPO/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$REPO/.venv/bin/activate"
fi

declare -a NAMES=() CODES=()

run_step() {
  local name="$1"; shift
  local log="$RESULTS/$name.log"
  echo "== $name"
  "$@" >"$log" 2>&1
  local rc=$?
  NAMES+=("$name"); CODES+=("$rc")
  tail -n 3 "$log"
}

run_py() {
  local script="$HERE/$1.py"; shift
  if [ ! -f "$script" ]; then
    echo "== $1: missing $script, skipped" | tee "$RESULTS/$1.log"
    NAMES+=("$1"); CODES+=("127")
    return
  fi
  run_step "$1" python "$script" "$@"
}

run_py h1_softmax_exposed
run_py h3_l2_working_set
run_py h4_wave_quantization
run_py h5_pv_accum
run_py h7_clock_sampler --standalone --seconds 20 --out "$RESULTS/h7_standalone.tsv"

if [ -f "$HERE/h7_clock_sampler.py" ]; then
  lingbot bench >"$RESULTS/lingbot_bench.log" 2>&1 &
  BENCH_PID=$!
  echo "$BENCH_PID" >"$RESULTS/lingbot_bench.pid"
  run_step h7_bench python "$HERE/h7_clock_sampler.py" --pidfile "$RESULTS/lingbot_bench.pid" \
    --out "$RESULTS/h7_bench.tsv"
  wait "$BENCH_PID"; rc=$?
  NAMES+=("lingbot_bench"); CODES+=("$rc")
fi

echo
for i in "${!NAMES[@]}"; do
  if [ "${CODES[$i]}" -eq 0 ]; then s=PASS; else s=FAIL; fi
  echo "$s ${NAMES[$i]} (exit ${CODES[$i]})"
done
