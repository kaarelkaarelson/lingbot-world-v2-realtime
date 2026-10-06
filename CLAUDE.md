# lingbot-world-v2-realtime

## Commands

WHEN running more than one benchmark or generation on a GPU pod, you MUST go through the warm worker instead of `python -m lingbot.generate` -- a fresh process re-loads weights and re-traces the model (~1.5-2.5 min of cold start per run; ~10 s is generation).

```bash
python -m lingbot.worker run -- --preset fast --bench --frame_num 157 --image examples/03/image.jpg --action_path examples/03 --prompt "$(cat examples/03/prompt.txt)"
python -m lingbot.worker status   # which worker (code hash) is alive
python -m lingbot.worker stop     # free the GPUs when done
```

- `run` takes the same arguments as `lingbot.generate`. Runs warm: `--bench`, `--bench_e2e`, `--frame_num`, `--decoder_gpu`, prompt, image, seed, output file.
- A new code hash (any change under `lingbot/` or `reference/`, the preset, `LINGBOT_*` env, visible GPUs) starts a fresh worker and stops the old one. Expect one cold start per code version.
- Keep both GPUs visible (don't set `CUDA_VISIBLE_DEVICES` per run): it is part of the hash, so changing it restarts the worker. Use `--decoder_gpu 1` for two-card runs.
- Not supported: `--preset stock` and `torchrun`; run `python -m lingbot.generate` directly for those.
- Worker logs: `/tmp/lingbot-worker-<hash>.log`.
- CPU profile of warm runs: `LINGBOT_WORKER_PYSPY=1 python -m lingbot.worker run -- ...` (twice or more), then `python -m lingbot.worker stop`; the py-spy file lands next to the log. Analyse only `_warm_run` (the first run includes compile): `python ~/.claude/skills/gpu-profiling/scripts/pyspy_breakdown.py <file> --under "_warm_run" --depth 3`. `py-spy --pid` attach is blocked in containers.
- GPU+CPU timeline of warm runs: `LINGBOT_WORKER_NSYS=1 LINGBOT_NVTX=1 LINGBOT_NSYS_CHUNKS=5:9 python -m lingbot.worker run -- ...` (twice or more), then `python -m lingbot.worker stop`; the worker runs under `nsys` (capture range = those chunks, one numbered `.nsys-rep` per run next to the log, finalized after stop). Analyse the last report (warm run): `nsys export --type sqlite <rep>`, then `python experiments/split_cpu/critical_path.py <rep>.sqlite`.
- Perfetto view of a capture: `python tools/nsys_to_perfetto.py <capture.sqlite|.nsys-rep> --open` (converts, then opens ui.perfetto.dev on this machine via Perfetto's `open_trace_in_ui`; the trace is not uploaded). Run it on the Mac after copying the `.sqlite` off the pod.
