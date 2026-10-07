#!/usr/bin/env bash
# Start `sglang serve` on one GPU with the repackaged 1.3B model, in the background, logging to /workspace/sgl/serve.log.
# Blocks until /health answers (the HTTP server is only started after every worker reports the model loaded,
# runtime/launch_server.py), then prints sanity lines and READY. Stop with:
#   kill "$(cat /workspace/sgl/serve.pid)"      (or: killall_sglang)
#
# Flags (all verified against sglang v0.5.17 runtime/server_args/server_args.py and cli/serve.py):
#   --model-type diffusion           skip auto-detection (the dir has model_index.json, so auto would also work)
#   --model-id <14B repo short name> config-class resolution is by registered HF repo name; the local dir name does not
#                                    contain it, so this picks LingBotWorldV2CausalDMDConfig (flow_shift 5, [1000,750,500,250])
#   --pipeline-class-name            same as SGLang's own 1-GPU LingBot CI case (gpu_cases.py)
#   --text-encoder-cpu-offload true  UMT5-XXL (11 GB bf16) stays on CPU except when a prompt is encoded (once per session)
#   --vae-cpu-offload false          every *_cpu_offload flag is set explicitly: performance_mode=auto otherwise picks
#   --image-encoder-cpu-offload false layerwise CPU offload for components left unset (auto_tune.py), which on a 32 GB
#                                    card can put the Wan VAE decoder on the streaming-offload path and slow decode
#   --warmup-mode off                no synthetic warmup request; bench.py discards chunks < 6 anyway
#   --enable-torch-compile false     torch.compile "will likely cause precision drifts" per the flag's own help text
# Attention on sm_120 falls back to Torch SDPA automatically (runtime/platforms/cuda.py); nothing to pass.
# VAE decode precision is a server flag in v0.5.17 (default fp32): add `--vae-precision bf16` to match a bf16 decoder.
set -euo pipefail

SGL_ROOT=/workspace/sgl
VENV="${SGL_VENV:-$SGL_ROOT/.venv}"
MODEL="$SGL_ROOT/lingbot-world-v2-1.3b-causal-fast-diffusers"
LOG="$SGL_ROOT/serve.log"
PIDFILE="$SGL_ROOT/serve.pid"
PORT=30000

export CUDA_HOME=/usr/local/cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # as in SGLang's CI cases

# shellcheck disable=SC1091
source "$VENV/bin/activate"

test -f "$MODEL/model_index.json"
test -f "$MODEL/transformer/config.json"
test -f "$MODEL/text_encoder/model.safetensors.index.json" || { echo "text_encoder/ missing (repackage.py --skip-text-encoder?)"; exit 1; }
ls "$MODEL"/transformer/*.safetensors >/dev/null

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "server already running, pid $(cat "$PIDFILE"); kill it first"
    exit 1
fi

# Preflight (CPU only, same resolution code the server runs): --model-id must select LingBotWorldV2CausalDMDConfig
# (shift 5, [1000,750,500,250]); the 1.x config would be shift 10 / [1000,821,642,321]. The server only logs this at
# DEBUG, so check it here instead of grepping the log.
# FlashInfer probes the GPU at import time in the entrypoint process, before CUDA is initialised there, so its
# TARGET_CUDA_ARCHS is empty and every JIT call fails with "FlashInfer requires GPUs with sm75 or higher"; the
# documented override fixes it (12.0a = RTX 5090).
export FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST:-12.0a}
# The realtime entrypoint process never loads transformer/config.json (only the GPU worker does), so the chunk size it
# requests is the dataclass default num_frames_per_block=3 regardless of the config. For the chunk-4 measurement the
# default in the installed package is set to 4 (sglang v0.5.17 bug; the block-3 run was recorded too).
sed -i 's/num_frames_per_block: int = 3/num_frames_per_block: int = 4/' \
    "$(python -c 'import sglang.multimodal_gen.configs.models.dits.lingbot_world as m; print(m.__file__)')"
echo "== preflight: config resolution"
MODEL="$MODEL" python - <<'PY'
import json, os
from sglang.multimodal_gen.registry import get_model_info
model = os.environ["MODEL"]
info = get_model_info(model, model_id="lingbot-world-v2-14b-causal-fast-diffusers")
assert info is not None, "get_model_info returned None"
cfg = info.pipeline_config_cls()
print(f"   pipeline={info.pipeline_cls.__name__} config={type(cfg).__name__} sampling={info.sampling_param_cls.__name__}")
print(f"   flow_shift={cfg.flow_shift} dmd_denoising_steps={cfg.dmd_denoising_steps} warp={cfg.warp_denoising_step} "
      f"dit_precision={cfg.dit_precision} vae_precision={cfg.vae_precision}")
assert type(cfg).__name__ == "LingBotWorldV2CausalDMDConfig", "wrong pipeline config class (1.x picked?)"
assert cfg.flow_shift == 5.0 and list(cfg.dmd_denoising_steps) == [1000, 750, 500, 250]
cfg.dit_config.update_model_arch(json.load(open(os.path.join(model, "transformer", "config.json"))))
a = cfg.dit_config.arch_config
print(f"   arch: layers={a.num_layers} heads={a.num_attention_heads} hidden={a.hidden_size} ffn={a.ffn_dim} "
      f"block={a.num_frames_per_block} sink={a.sink_size} window={a.sliding_window_num_frames} local_attn={a.local_attn_size}")
assert (a.num_layers, a.hidden_size, a.ffn_dim, a.num_frames_per_block) == (30, 1536, 8960, 4)
print("   preflight ok")
PY

: > "$LOG"
nohup sglang serve \
    --model-type diffusion \
    --model-path "$MODEL" \
    --model-id lingbot-world-v2-14b-causal-fast-diffusers \
    --pipeline-class-name LingBotWorldCausalDMDPipeline \
    --num-gpus 1 \
    --dit-cpu-offload false \
    --text-encoder-cpu-offload true \
    --vae-cpu-offload false \
    --image-encoder-cpu-offload false \
    --enable-torch-compile false \
    --warmup-mode off \
    --host 127.0.0.1 \
    --port "$PORT" \
    --log-level info \
    >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "sglang serve pid $(cat "$PIDFILE"), log $LOG"

# model load: DiT 6.8 GB fp32 -> bf16, VAE, T5 11 GB to CPU; a few minutes from a warm disk
for _ in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "server died; last 60 log lines:"
        tail -n 60 "$LOG"
        exit 1
    fi
    sleep 5
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null || { echo "no /health after 20 min"; tail -n 60 "$LOG"; exit 1; }

echo "== READY (model loaded; /health answered). sanity lines from $LOG:"
grep -i "SM12\|SDPA\|attention backend" "$LOG" | head -5 || true
grep -i "flashinfer\|Loading CausalLingBotWorldTransformer3DModel" "$LOG" | head -3 || true
curl -fsS "http://127.0.0.1:$PORT/models"; echo
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "READY"
