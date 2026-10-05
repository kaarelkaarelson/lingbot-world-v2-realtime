#!/usr/bin/env python3
"""Assemble LingBot-World 2.0 1.3B causal_fast in the diffusers layout SGLang v0.5.17 loads.

Output (default /workspace/sgl/lingbot-world-v2-1.3b-causal-fast-diffusers/):
  model_index.json, scheduler/, tokenizer/, vae/, text_encoder/   <- robbyant/lingbot-world-v2-14b-causal-fast-diffusers
  transformer/model-0000N-of-00006.safetensors + model.safetensors.index.json
                                                                   <- robbyant/lingbot-world-v2-1.3b-causal-fast (flat repo)
  transformer/config.json                                          <- 14B config.json with the 1.3B geometry written in
  transformer/config.14b.json                                      <- the untouched 14B original, for reference

Sizes on the Hub (HF API, 2026-09-17; text_encoder/config.json says torch_dtype bfloat16, so 11.4 GB, not fp32):
  text_encoder/  3 shards, bf16 UMT5-XXL      11.36 GB   (4.94 + 4.98 + 1.44)  -> --skip-text-encoder leaves it out
  transformer/   6 shards, F32 1.3B DiT        6.84 GB   (1.10 1.23 1.14 1.14 1.15 1.09)
  vae/                                          0.51 GB
  tokenizer/ + scheduler/ + model_index.json    0.02 GB
  total                                        18.73 GB   (the 14B repo's own transformer/ shards, 74 GB, are never touched)

Disk: files are written straight into --out (huggingface_hub local_dir mode: <out>/.cache/huggingface/download/*.incomplete
then rename on the same filesystem, no second copy). The only other write is the hf_xet chunk cache under $HF_HOME/xet
(10 GB by default); this script caps it at 1 GB and, unless HF_HOME is already set, puts HF_HOME on whichever of
/workspace and / has more free space.

Why SGLang accepts this layout (all read from sglang v0.5.17 source):
  - runtime/loader/utils.py:_list_safetensors_files globs transformer/*.safetensors; the index file is optional
    (only diffusion_pytorch_model.safetensors.index.json is consulted, and only for a completeness check).
  - configs/models/base.py:update_model_arch sets every key of transformer/config.json onto LingBotWorldArchConfig, and
    hidden_size = num_attention_heads * attention_head_dim, so the 1.3B geometry comes entirely from config.json.
  - configs/models/dits/lingbot_world.py:param_names_mapping renames the raw Wan names (blocks.N.self_attn.q.weight ...)
    on load; the 1.3B flat checkpoint uses exactly the same raw names as the 14B-diffusers transformer/ shards.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys


def _pick_hf_home() -> str:
    if os.environ.get("HF_HOME"):
        return os.environ["HF_HOME"]
    candidates = [d for d in ("/workspace", "/tmp") if os.path.isdir(d)]
    best = max(candidates, key=lambda d: shutil.disk_usage(d).free)
    return os.path.join(best, "hf-home")


# huggingface_hub reads these at import time
os.environ["HF_HOME"] = _pick_hf_home()
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", str(1 * 2**30))

from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: E402

DIFFUSERS_REPO = "robbyant/lingbot-world-v2-14b-causal-fast-diffusers"
DIT_REPO = "robbyant/lingbot-world-v2-1.3b-causal-fast"
DEFAULT_OUT = "/workspace/sgl/lingbot-world-v2-1.3b-causal-fast-diffusers"

BASE_PATTERNS = ["model_index.json", "scheduler/*", "tokenizer/*", "vae/*"]
TEXT_ENCODER_PATTERNS = ["text_encoder/*"]
DIT_PATTERNS = ["model-*-of-*.safetensors", "model.safetensors.index.json"]

# 1.3B geometry from lingbot-world-v2 wan/configs/wan_i2v_1_3B.py, plus our streaming settings
# (chunk 4 latents = 16 frames, --sink_size 6, --local_attn_size 18). Every key must already exist in the 14B
# config.json (checked below) so nothing is invented; SGLang's arch dataclass declares all of them too.
OVERRIDES = {
    "num_attention_heads": 12,
    "num_layers": 30,
    "ffn_dim": 8960,
    "num_frames_per_block": 4,
    "sink_size": 6,
    "sliding_window_num_frames": 18,
    "local_attn_size": -1,
}
ATTENTION_HEAD_DIM = 128  # unchanged from the 14B; 12 * 128 = 1536 = the 1.3B hidden size

# Copied verbatim from sglang v0.5.17 python/sglang/multimodal_gen/configs/models/dits/lingbot_world.py
# (LingBotWorldArchConfig.param_names_mapping keys). Every checkpoint tensor must match one of these.
PARAM_NAME_PATTERNS = [
    r"^patch_embedding\.(.*)$",
    r"^patch_embedding_wancamctrl\.(.*)$",
    r"^c2ws_hidden_states_layer1\.(.*)$",
    r"^c2ws_hidden_states_layer2\.(.*)$",
    r"^text_embedding\.0\.(.*)$",
    r"^text_embedding\.2\.(.*)$",
    r"^time_embedding\.0\.(.*)$",
    r"^time_embedding\.2\.(.*)$",
    r"^time_projection\.1\.(.*)$",
    r"^blocks\.(\d+)\.modulation$",
    r"^blocks\.(\d+)\.self_attn\.q\.(.*)$",
    r"^blocks\.(\d+)\.self_attn\.k\.(.*)$",
    r"^blocks\.(\d+)\.self_attn\.v\.(.*)$",
    r"^blocks\.(\d+)\.self_attn\.o\.(.*)$",
    r"^blocks\.(\d+)\.self_attn\.norm_q\.(.*)$",
    r"^blocks\.(\d+)\.self_attn\.norm_k\.(.*)$",
    r"^blocks\.(\d+)\.norm3\.(.*)$",
    r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$",
    r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$",
    r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$",
    r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$",
    r"^blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$",
    r"^blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$",
    r"^blocks\.(\d+)\.ffn\.0\.(.*)$",
    r"^blocks\.(\d+)\.ffn\.2\.(.*)$",
    r"^blocks\.(\d+)\.cam_injector_layer1\.(.*)$",
    r"^blocks\.(\d+)\.cam_injector_layer2\.(.*)$",
    r"^blocks\.(\d+)\.cam_scale_layer\.(.*)$",
    r"^blocks\.(\d+)\.cam_shift_layer\.(.*)$",
    r"^head\.modulation$",
    r"^head\.head\.(.*)$",
]

# Shapes the 1.3B geometry implies; checked against the downloaded safetensors headers.
EXPECTED_SHAPES = {
    "patch_embedding.weight": [1536, 36, 1, 2, 2],
    "patch_embedding_wancamctrl.weight": [1536, 1536],
    "blocks.0.self_attn.q.weight": [1536, 1536],
    "blocks.0.ffn.0.weight": [8960, 1536],
    "time_projection.1.weight": [9216, 1536],
    "head.head.weight": [64, 1536],
}

GB = 1e9


def log(msg: str) -> None:
    print(msg, flush=True)


def _matches(rel: str, patterns: list[str]) -> bool:
    import fnmatch

    return any(fnmatch.fnmatch(rel, p) for p in patterns)


def plan_downloads(api: HfApi, out: str, skip_text_encoder: bool) -> list[tuple[str, str, str, int]]:
    """Return (repo, hub path, local path, size) for every file to fetch, skipping files already complete on disk."""
    plan = []
    diffusers_files = api.model_info(DIFFUSERS_REPO, files_metadata=True).siblings
    dit_files = api.model_info(DIT_REPO, files_metadata=True).siblings

    groups = [
        (DIFFUSERS_REPO, diffusers_files, BASE_PATTERNS, ""),
        (DIT_REPO, dit_files, DIT_PATTERNS, "transformer"),
    ]
    if not skip_text_encoder:
        groups.append((DIFFUSERS_REPO, diffusers_files, TEXT_ENCODER_PATTERNS, ""))

    for repo, siblings, patterns, subdir in groups:
        for s in siblings:
            if not _matches(s.rfilename, patterns):
                continue
            local = os.path.join(out, subdir, s.rfilename)
            size = s.size or 0
            if os.path.exists(local) and os.path.getsize(local) == size:
                continue
            plan.append((repo, s.rfilename, local, size))
    return plan


def check_index_names(index_path: str) -> dict[str, str]:
    weight_map = json.load(open(index_path))["weight_map"]
    compiled = [re.compile(p) for p in PARAM_NAME_PATTERNS]
    unmatched = [n for n in weight_map if not any(c.match(n) for c in compiled)]
    if unmatched:
        raise SystemExit(f"{len(unmatched)} tensor names do not match SGLang's LingBot param_names_mapping, "
                         f"e.g. {unmatched[:5]}")
    blocks = {int(m.group(1)) for n in weight_map for m in [re.match(r"^blocks\.(\d+)\.", n)] if m}
    if blocks != set(range(OVERRIDES["num_layers"])):
        raise SystemExit(f"index has blocks {sorted(blocks)[:3]}..{max(blocks)}, expected 0..{OVERRIDES['num_layers'] - 1}")
    log(f"   index ok: {len(weight_map)} tensors, {len(blocks)} blocks, all names match the loader mapping")
    return weight_map


def safetensors_header(path: str) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        if n > 100 * 2**20:
            raise SystemExit(f"{path}: implausible safetensors header size {n}")
        return json.loads(f.read(n))


def check_shapes(transformer_dir: str, weight_map: dict[str, str]) -> None:
    headers: dict[str, dict] = {}
    for name, expected in EXPECTED_SHAPES.items():
        shard = weight_map[name]
        if shard not in headers:
            headers[shard] = safetensors_header(os.path.join(transformer_dir, shard))
        entry = headers[shard][name]
        if entry["shape"] != expected:
            raise SystemExit(f"{name}: shape {entry['shape']} != expected {expected} (config overrides would not match the weights)")
        log(f"   {name}: {entry['dtype']} {entry['shape']} ok")


def derive_transformer_config(config_14b: dict) -> dict:
    missing = [k for k in OVERRIDES if k not in config_14b]
    if missing:
        raise SystemExit(f"keys {missing} are not in the 14B transformer/config.json; refusing to invent them")
    if config_14b.get("attention_head_dim") != ATTENTION_HEAD_DIM:
        raise SystemExit(f"14B attention_head_dim={config_14b.get('attention_head_dim')}, expected {ATTENTION_HEAD_DIM}")
    if config_14b.get("_class_name") != "CausalLingBotWorldTransformer3DModel":
        raise SystemExit(f"unexpected _class_name {config_14b.get('_class_name')!r}")
    derived = dict(config_14b)
    derived.update(OVERRIDES)
    return derived


def check_against_sglang(derived: dict) -> None:
    """When sglang is importable (on the pod), confirm every config key is a field of the arch dataclass."""
    try:
        from dataclasses import fields

        from sglang.multimodal_gen.configs.models.dits.lingbot_world import LingBotWorldArchConfig
    except Exception as e:  # noqa: BLE001
        log(f"   sglang not importable here ({type(e).__name__}); skipping dataclass field check")
        return
    arch = LingBotWorldArchConfig()
    names = {f.name for f in fields(arch)}
    unknown = [k for k in derived if k not in names and not k.startswith("_")]
    if unknown:
        raise SystemExit(f"config keys {unknown} are not LingBotWorldArchConfig fields")
    live = list(arch.param_names_mapping.keys())
    if live != PARAM_NAME_PATTERNS:
        raise SystemExit("installed sglang's param_names_mapping differs from the v0.5.17 copy in this script")
    log("   sglang LingBotWorldArchConfig: all config keys are dataclass fields; param_names_mapping matches")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--skip-text-encoder", action="store_true", help="do not download text_encoder/ (11.4 GB)")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch only the 14B transformer/config.json and the 1.3B index; print the plan and derived config")
    args = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is not set")
    api = HfApi()
    out = args.out
    transformer_dir = os.path.join(out, "transformer")
    os.makedirs(transformer_dir, exist_ok=True)
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    log("== disk")
    for d in sorted({"/workspace", "/", out, os.environ["HF_HOME"]}):
        if os.path.isdir(d):
            u = shutil.disk_usage(d)
            log(f"   {u.free / GB:6.1f} GB free of {u.total / GB:6.1f} GB  {d}")
    log(f"   HF_HOME={os.environ['HF_HOME']}  HF_XET_CHUNK_CACHE_SIZE_BYTES={os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES']}")

    log("== download plan")
    plan = plan_downloads(api, out, args.skip_text_encoder)
    by_group: dict[str, int] = {}
    for repo, rel, _local, size in plan:
        key = f"{repo.split('/')[1]}:{rel.split('/')[0] if '/' in rel else rel}"
        by_group[key] = by_group.get(key, 0) + size
    for key, size in sorted(by_group.items()):
        log(f"   {size / GB:6.2f} GB  {key}")
    total = sum(p[3] for p in plan)
    free = shutil.disk_usage(out).free
    log(f"   {total / GB:6.2f} GB  total still to download; {free / GB:.1f} GB free at {out}")
    if args.skip_text_encoder:
        log("   text_encoder/ skipped (--skip-text-encoder); the server cannot start without it")

    log("== 14B transformer/config.json + 1.3B index (small)")
    cfg_path = hf_hub_download(DIFFUSERS_REPO, "transformer/config.json", local_dir=out)
    config_14b = json.load(open(cfg_path))
    shutil.copyfile(cfg_path, os.path.join(transformer_dir, "config.14b.json"))
    index_path = hf_hub_download(DIT_REPO, "model.safetensors.index.json", local_dir=transformer_dir)
    weight_map = check_index_names(index_path)

    log("== transformer/config.json (14B file with the 1.3B geometry)")
    derived = derive_transformer_config(config_14b)
    for k, v in OVERRIDES.items():
        log(f"   {k}: {config_14b[k]} -> {v}")
    check_against_sglang(derived)
    with open(os.path.join(transformer_dir, "config.json"), "w") as f:
        json.dump(derived, f, indent=2)
        f.write("\n")

    if args.dry_run:
        log("== dry run: derived config")
        log(json.dumps(derived, indent=2))
        return

    if total + 1 * GB > free:
        raise SystemExit(f"need {total / GB:.1f} GB + margin, only {free / GB:.1f} GB free; use --skip-text-encoder or free space")

    log("== 1.3B DiT shards -> transformer/")
    snapshot_download(DIT_REPO, local_dir=transformer_dir, allow_patterns=DIT_PATTERNS)
    check_shapes(transformer_dir, weight_map)

    log("== model_index.json, scheduler/, tokenizer/, vae/")
    snapshot_download(DIFFUSERS_REPO, local_dir=out, allow_patterns=BASE_PATTERNS)

    if args.skip_text_encoder:
        log("== text_encoder/ skipped")
    else:
        log("== text_encoder/ (11.4 GB, last)")
        snapshot_download(DIFFUSERS_REPO, local_dir=out, allow_patterns=TEXT_ENCODER_PATTERNS)

    # snapshot_download re-fetches transformer/config.json only if asked for; it is not in BASE_PATTERNS, so the
    # derived file written above is what stays on disk.
    log("== layout")
    for root, _dirs, files in os.walk(out):
        if "/.cache" in root:
            continue
        for name in sorted(files):
            p = os.path.join(root, name)
            log(f"   {os.path.getsize(p) / GB:6.2f} GB  {os.path.relpath(p, out)}")
    log(f"SGL repackage ok: {out}")


if __name__ == "__main__":
    sys.exit(main())
