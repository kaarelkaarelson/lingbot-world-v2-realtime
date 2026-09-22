import gc
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
# import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward_causal, sp_dit_forward_causal
from .distributed.util import get_world_size
from .modules.model_fast import kv_ring_plan
_SAGE_KVQ = os.environ.get("LINGBOT_ATTN") == "sage_kvq"
if _SAGE_KVQ:
    from .modules import sage_kvq
if os.environ.get("LINGBOT_DIT_FUSION") == "1":
    # Graph-break-free DiT (one Dynamo graph per forward, rope table per
    # forward, cam-MLP and cross-attn K/V cached per chunk / generation).
    from .modules.model_fast_fusion import WanModelFast
else:
    from .modules.model_fast import WanModelFast
from .modules.model_causal import WanModelCausal
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE

from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.cam_utils import (
    compute_relative_poses,
    interpolate_camera_poses,
    get_plucker_embeddings,
    get_Ks_transformed,
)
from einops import rearrange


def _resolve_asset_path(filename, checkpoint_dir, assets_dir=None):
    """Resolve T5 / VAE / tokenizer paths, falling back to ``assets_dir``.

    The 1.3B Hugging Face upload currently ships only DiT weights; T5, VAE,
    and the tokenizer can be reused from the 14B release via ``--assets_dir``.
    """
    candidates = [os.path.join(checkpoint_dir, filename)]
    if assets_dir:
        candidates.append(os.path.join(assets_dir, filename))
    for path in candidates:
        if os.path.exists(path):
            return path
    searched = ", ".join(candidates)
    raise FileNotFoundError(
        f"Required asset {filename!r} not found. Looked in: {searched}. "
        "Pass --assets_dir pointing at a 14B (or Wan) checkpoint that contains "
        "models_t5_umt5-xxl-enc-bf16.pth, Wan2.1_VAE.pth, and google/umt5-xxl."
    )


def _resolve_dit_dir(checkpoint_dir, subfolder):
    """Prefer ``checkpoint_dir/subfolder`` when it exists, else the root.

    14B causal-fast stores DiT weights under ``transformers/``. The 1.3B
    causal-fast upload currently places shards at the repository root.
    """
    if subfolder:
        candidate = os.path.join(checkpoint_dir, subfolder)
        if os.path.isdir(candidate):
            return candidate
    return checkpoint_dir


def _load_safetensors_state_dict(dit_dir, device="cpu"):
    """Load a (possibly sharded) safetensors dump from ``dit_dir``."""
    from safetensors.torch import load_file

    device = str(device)
    for index_name in (
            "model.safetensors.index.json",
            "diffusion_pytorch_model.safetensors.index.json",
    ):
        index_path = os.path.join(dit_dir, index_name)
        if not os.path.isfile(index_path):
            continue
        with open(index_path) as f:
            index = json.load(f)
        state = {}
        for shard in sorted(set(index["weight_map"].values())):
            state.update(load_file(os.path.join(dit_dir, shard), device=device))
        return state

    for single_name in (
            "model.safetensors",
            "diffusion_pytorch_model.safetensors",
    ):
        single_path = os.path.join(dit_dir, single_name)
        if os.path.isfile(single_path):
            return load_file(single_path, device=device)

    raise FileNotFoundError(
        f"No safetensors weights found in {dit_dir}. Expected a sharded "
        "index (model.safetensors.index.json) or a single model.safetensors."
    )


class FP8Linear(torch.nn.Module):
    """nn.Linear replacement: rowwise e4m3 weight, dynamic rowwise e4m3
    activations, fp32-accumulate GEMM via torch._scaled_mm, bf16 output."""

    FMAX = torch.finfo(torch.float8_e4m3fn).max

    def __init__(self, lin):
        super().__init__()
        w = lin.weight.detach()
        w_scale = (w.abs().amax(dim=1, keepdim=True).float() / self.FMAX).clamp(min=1e-12)  # [N,1]
        self.register_buffer("w8", (w.float() / w_scale).to(torch.float8_e4m3fn))            # [N,K]
        self.register_buffer("w_scale_t", w_scale.t().contiguous())                           # [1,N]
        self.bias = lin.bias

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        x_scale = (x2.abs().amax(dim=1, keepdim=True).float() / self.FMAX).clamp(min=1e-12)   # [M,1]
        x8 = (x2.float() / x_scale).to(torch.float8_e4m3fn)
        y = torch._scaled_mm(x8, self.w8.t(), scale_a=x_scale, scale_b=self.w_scale_t,
                             bias=None if self.bias is None else self.bias.to(torch.bfloat16),
                             out_dtype=torch.bfloat16)
        return y.reshape(*shape[:-1], y.shape[-1])


def _inductor_tune():
    """LINGBOT_INDUCTOR_TUNE=1: Inductor knobs for the memory-bound Triton
    kernels (LN/modulation/quant reductions over 1536- and 8960-wide rows).
    multi_kernel benchmarks persistent vs looped reductions and
    coordinate_descent_tuning tunes block sizes at first compile; the
    reduction order may change -> fp32-ulp differences, not bitwise.
    realize_reads_threshold: the norm2 LN+modulation output (6 reads) is
    realised and its FP8 quant lands in a second kernel, unlike norm1's;
    keeping it inlined lets the amax fuse (same values, recomputed instead
    of stored)."""
    import torch._inductor.config as inductor_config
    inductor_config.triton.multi_kernel = 1
    inductor_config.coordinate_descent_tuning = True
    inductor_config.realize_reads_threshold = 8
    logging.info("Inductor: multi_kernel=1, coordinate_descent_tuning, realize_reads_threshold=8")


def _dit_kwargs_from_config(config, extra=None):
    kwargs = dict(
        model_type="i2v",
        patch_size=tuple(config.patch_size),
        text_len=config.text_len,
        in_dim=getattr(config, "in_dim", 36),
        dim=config.dim,
        ffn_dim=config.ffn_dim,
        freq_dim=config.freq_dim,
        text_dim=getattr(config, "text_dim", 4096),
        out_dim=getattr(config, "out_dim", 16),
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        qk_norm=config.qk_norm,
        cross_attn_norm=config.cross_attn_norm,
        eps=config.eps,
    )
    if extra:
        kwargs.update(extra)
    return kwargs


def load_dit_model(model_cls, checkpoint_dir, subfolder, config, torch_dtype,
                   extra=None, device="cpu"):
    """Load a DiT from ``transformers/`` or the checkpoint root.

    Uses ``from_pretrained`` when ``config.json`` is present. Otherwise builds
    the module from the task EasyDict and loads sharded safetensors — the
    layout of the current 1.3B Hugging Face upload.
    """
    extra = extra or {}
    dit_dir = _resolve_dit_dir(checkpoint_dir, subfolder)
    logging.info(f"Loading {model_cls.__name__} from {dit_dir}")
    if os.path.isfile(os.path.join(dit_dir, "config.json")):
        return model_cls.from_pretrained(
            dit_dir, torch_dtype=torch_dtype, **extra)

    logging.info(
        f"config.json not found in {dit_dir}; building {model_cls.__name__} "
        "from the task config and loading safetensors weights."
    )
    # Build directly on the target device: random init of 1.3B params is
    # ~1 s on the GPU vs minutes on a contended CPU, and the weights then
    # load straight there instead of CPU -> cast -> copy.
    with torch.device(device):
        model = model_cls(**_dit_kwargs_from_config(config, extra))
    state = _load_safetensors_state_dict(dit_dir, device=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logging.warning(f"Missing keys when loading DiT: {missing}")
    if unexpected:
        logging.warning(f"Unexpected keys when loading DiT: {unexpected}")
    return model.to(dtype=torch_dtype)


def parse_timesteps(env_value, default, num_train_timesteps=1000):
    """LINGBOT_TIMESTEPS: comma-separated scheduler indices (e.g. "0,179,358") overriding
    `timesteps_index`; fewer entries = fewer denoising forwards (the t=0 cache-write forward
    stays). Must be strictly increasing ints in [0, num_train_timesteps)."""
    if env_value is None or env_value.strip() == "":
        return list(default)
    try:
        idx = [int(x) for x in env_value.split(",")]
    except ValueError as e:
        raise ValueError(f"LINGBOT_TIMESTEPS={env_value!r}: expected comma-separated ints") from e
    if not idx:
        raise ValueError(f"LINGBOT_TIMESTEPS={env_value!r}: empty")
    if any(i < 0 or i >= num_train_timesteps for i in idx):
        raise ValueError(f"LINGBOT_TIMESTEPS={env_value!r}: indices must be in [0, {num_train_timesteps})")
    if any(b <= a for a, b in zip(idx, idx[1:])):
        raise ValueError(f"LINGBOT_TIMESTEPS={env_value!r}: indices must be strictly increasing")
    return idx


class WanI2VCausal:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        pipe_dtype=torch.bfloat16,
        local_attn_size=-1,
        sink_size=0,
        infer_mode="causal_fast",
        assets_dir=None,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            infer_mode (`str`, *optional*, defaults to "causal_fast"):
                Inference mode. "causal_fast" uses the distilled few-step
                model (config.fast_checkpoint) with KV-cache windowing
                (local_attn_size / sink_size). "causal_pretrain" uses the
                pretrained causal model (config.causal_checkpoint) with
                40-step CFG sampling.
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
            assets_dir (`str`, *optional*):
                Directory that holds shared T5 / VAE / tokenizer files when
                they are not packaged with ``checkpoint_dir`` (the 1.3B DiT
                upload). Typically the 14B causal-fast checkpoint directory.
        """
        assert infer_mode in ("causal_fast", "causal_pretrain"), \
            f"Unsupported infer_mode: {infer_mode}"
        self.infer_mode = infer_mode

        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype
        self.pipe_dtype = pipe_dtype
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        # T5 is built lazily: prompt embeddings are cached on disk, so the
        # 11 GB encoder is only constructed for a prompt nobody has encoded yet.
        self._t5_kwargs = dict(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=_resolve_asset_path(
                config.t5_checkpoint, checkpoint_dir, assets_dir),
            tokenizer_path=_resolve_asset_path(
                config.t5_tokenizer, checkpoint_dir, assets_dir),
            shard_fn=shard_fn if t5_fsdp else None,
        )
        self.text_encoder = None
        self._t5_disk_cache_dir = os.path.join(assets_dir or checkpoint_dir, "t5_cache")

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        # Exp. 9 "quality mode": the stock VAE runs fp32; fp16 is 72 dB vs fp32 and
        # 1.35x faster, channels_last_3d removes cuDNN's NCHW<->NHWC transposes (1.45x).
        vae_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
            os.environ.get("LINGBOT_VAE_DTYPE", ""), torch.float)
        self.vae = Wan2_1_VAE(
            vae_pth=_resolve_asset_path(
                config.vae_checkpoint, checkpoint_dir, assets_dir),
            dtype=vae_dtype,
            device=self.device)
        self._vae_cl = os.environ.get("LINGBOT_VAE_CL") == "1"
        # exp. 9c: true fp16 weights without autocast (autocast keeps norm/SiLU/residual in
        # fp32: 0.78 -> 0.60 s/chunk), Conv2d channels_last too (0 transpose kernels), and
        # torch.compile with the recompile limit raised (default 8 is exhausted by the
        # shared CausalConv3d/RMS_norm/Resample code objects -> silent eager fallback).
        self._vae_half = os.environ.get("LINGBOT_VAE_HALF") == "1"
        if self._vae_half:
            self.vae.model.decoder.half(); self.vae.model.conv2.half()
        if self._vae_cl:
            for mod in self.vae.model.decoder.modules():
                if isinstance(mod, torch.nn.Conv3d):
                    mod.weight.data = mod.weight.data.contiguous(memory_format=torch.channels_last_3d)
                elif isinstance(mod, torch.nn.Conv2d):
                    mod.weight.data = mod.weight.data.contiguous(memory_format=torch.channels_last)
        if os.environ.get("LINGBOT_VAE_COMPILE") == "1":
            torch._dynamo.config.recompile_limit = 64
            self.vae.model.decoder = torch.compile(self.vae.model.decoder, dynamic=True)
        if vae_dtype != torch.float or self._vae_cl or self._vae_half:
            logging.info(f"Wan VAE decoder: dtype={vae_dtype}, half={self._vae_half}, channels_last={self._vae_cl}, "
                         f"compile={os.environ.get('LINGBOT_VAE_COMPILE') == '1'}")
        # Exp. 11: functional-cache driver of the same decoder, one static graph per latent
        # step, so torch.compile fuses the elementwise work that the list cache blocks.
        # LINGBOT_VAE_FUSED: '1' = max-autotune-no-cudagraphs, 'eager', or a torch.compile mode.
        self._vae_fused = None
        fused = os.environ.get("LINGBOT_VAE_FUSED")
        if fused:
            from .modules.vae2_1_fused import FusedDecoder
            mode = {"1": "max-autotune-no-cudagraphs", "eager": None}.get(fused, fused)
            self._vae_fused = FusedDecoder(self.vae, compile_mode=mode)
            logging.info(f"Wan VAE fused decoder enabled (compile={mode})")
        # stream/live.py: with LINGBOT_VAE_STREAM=1, each decoded chunk is handed to
        # frame_sink(chunk_id, frames[C,F,H,W] in [-1,1], vae_stream, chunk_gen_start)
        # on the VAE stream instead of being kept for the whole-clip video.
        self.frame_sink = None
        # stream/control.py: pose_provider(chunk_id, n_latents) -> [n,4,4] framewise relative
        # OpenCV poses for the chunk; replaces the precomputed poses.npy trajectory (frame_num
        # then sets the rollout length) and may raise to end the rollout early.
        self.pose_provider = None
        # stream/live.py: chunk_gate(chunk_id) is called before each chunk's pose sample and denoise; it may block
        self.chunk_gate = None
        self._y_cache = None  # (key, y): the I2V conditioning latent is a pure function of (image, F, h, w)

        dit_device = torch.device('cpu') if (dit_fsdp or use_sp) else self.device
        if self.infer_mode == "causal_fast":
            self.model = load_dit_model(
                WanModelFast,
                checkpoint_dir,
                config.fast_checkpoint,
                config,
                torch.bfloat16,
                extra=dict(
                    local_attn_size=self.local_attn_size,
                    sink_size=self.sink_size),
                device=dit_device,
            )
        else:
            self.model = load_dit_model(
                WanModelCausal,
                checkpoint_dir,
                config.causal_checkpoint,
                config,
                torch.bfloat16,
                device=dit_device,
            )

        self.model = self._configure_model(
            model=self.model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype).to(self.device)

        # Optional rowwise FP8 for the large Linears. Plain buffers +
        # torch._scaled_mm (no tensor subclass: torchao's Float8Tensor breaks
        # Dynamo guards under dynamic=True). Needs torch.compile to fuse the
        # activation quantisation; eager FP8 is slower than bf16 on this card.
        if os.environ.get("LINGBOT_FP8") == "1":
            # Blocks only: the time-embedding MLP and head must stay fp32/bf16.
            n_all, n_fp8 = 0, 0
            for parent in list(self.model.blocks.modules()):
                for name, m in list(parent.named_children()):
                    if not isinstance(m, torch.nn.Linear):
                        continue
                    n_all += 1
                    if (m.in_features >= 1024 and m.out_features >= 1024
                            and m.in_features % 16 == 0 and m.out_features % 16 == 0):
                        setattr(parent, name, FP8Linear(m))
                        n_fp8 += 1
            logging.info(f"FP8 rowwise enabled on {n_fp8} of {n_all} Linear layers")

        compile_mode = os.environ.get("LINGBOT_TORCH_COMPILE")
        if compile_mode and os.environ.get("LINGBOT_INDUCTOR_TUNE") == "1":
            _inductor_tune()
        if compile_mode == "regional":
            # One graph per block, reused by all 30: same in-block fusion as
            # the whole-model graph at a fraction of the compile time.
            for i, block in enumerate(self.model.blocks):
                self.model.blocks[i] = torch.compile(block, dynamic=True)
            logging.info("torch.compile enabled (regional, per block)")
        elif compile_mode:
            # dynamic=True: current_start and the KV-cache slice bounds are
            # Python ints that change every chunk; specialising on them would
            # recompile per chunk and trip Dynamo's recompile limit.
            self.model = torch.compile(
                self.model, dynamic=True,
                mode=None if compile_mode == "1" else compile_mode)
            logging.info(f"torch.compile enabled (mode={compile_mode})")

        # Optional tiny decoder (madebyollin/taehv, taew2_1 weights) in place
        # of the Wan VAE decoder. It takes the model-space x0 latents directly:
        # applying the Wan VAE mean/std first is wrong (oversaturated output).
        taehv_path = os.environ.get("LINGBOT_TAEHV")
        self._taehv = None
        if taehv_path:
            from .modules.taehv import TAEHV
            self._taehv = TAEHV(checkpoint_path=taehv_path).to(self.device, torch.float16).eval()
            logging.info(f"TAEHV decoder enabled from {taehv_path}")

        # Optional Flash-VAED (arXiv 2602.19161) pruned/distilled Wan 2.1 decoder:
        # same latent contract as Wan2_1_VAE.decode (model-space latents in,
        # [C,T,H,W] in [-1,1] out), ~6x faster than the full decoder.
        self._flashvaed = None
        fv_path = os.environ.get("LINGBOT_FLASHVAED")
        if fv_path:
            import importlib.util
            repo = os.environ.get("LINGBOT_FLASHVAED_REPO", "/workspace/Flash-VAED")
            spec = importlib.util.spec_from_file_location(
                "flash_vaed_wan_student", os.path.join(repo, "models", "wan", "model_hybrid_aggressive.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._flashvaed = mod.WanVAE(vae_pth=fv_path, device=self.device, dtype=torch.bfloat16)
            res = self._flashvaed.model.load_state_dict(
                torch.load(fv_path, map_location="cpu", weights_only=True), strict=False)
            missing = [k for k in res.missing_keys if not k.startswith("encoder")]
            assert not missing, f"Flash-VAED checkpoint is missing decoder weights: {missing[:5]}"
            # Its decode() hard-codes a cache reset at latent 20 (the paper's 21-latent
            # clip), which drops 3 frames and misaligns everything after. The decoder is
            # a pure causal-conv stack, so an uninterrupted cache is exact (measured
            # identical to 21-latent windows with re-warm); window>0 keeps that scheme
            # as an option, re-warming with the previous window's last `warm` latents.
            self._flashvaed_window = int(os.environ.get("LINGBOT_FLASHVAED_WINDOW", "0"))
            self._flashvaed_warm = int(os.environ.get("LINGBOT_FLASHVAED_WARM", "4"))
            assert self._flashvaed_warm >= 1
            logging.info(f"Flash-VAED decoder enabled from {fv_path} "
                         f"(window={self._flashvaed_window or 'continuous'}, warm={self._flashvaed_warm})")

        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

        # T5 prompt-embedding cache. Same-prompt re-encodes hit this dict
        # instead of re-running the umt5-xxl encoder (~360 ms/call).
        # Keyed by sha256(prompt.utf8); value is the list returned by
        # T5EncoderModel.__call__ (already device-resident). Unbounded;
        # callers can clear via `pipe.clear_text_cache()` if needed.
        self._t5_cache: dict[str, list] = {}

        # Reset per generate() and flipped True after the first DiT forward.
        # Passed into model.forward as `cross_attn_first_call` to skip the
        # crossattn_cache["is_init"].item() sync inside WanCrossAttention.
        self._cross_attn_initialized: bool = False

    def clear_text_cache(self):
        """Drop all cached T5 prompt embeddings. Frees ~4 MB per entry."""
        self._t5_cache.clear()

    def prewarm(
        self,
        img,
        max_area: int = 480 * 832,
        frame_num: int = 81,
        chunk_size: int = 3,
        text_seq_len: int = 512,
    ):
        """Opt-in pre-warm. Run one dummy DiT forward at the same shape a
        subsequent generate() call will use, so CUDA kernels are autotuned,
        FSDP all-gathers happen, and Ulysses all-to-alls handshake — all
        outside the timed generate() window.

        Without this call, the first generate() pays a ~7s warmup tax in
        chunk 0 (CUDA lazy init, kernel autotuning, NCCL handshake). On
        8xH100 at 480*832/81 frames, calling prewarm() before the first
        generate() reduces generate()'s wall-clock by ~6.5s (~30%) with
        bit-identical output.

        Idempotent: subsequent calls on the same pipe are no-ops.
        Shape-keyed: if generate() is later invoked with a different shape,
        the autotuner will warm those kernels on demand in chunk 0 (no
        incorrect output, just the tax re-paid once).

        Args:
            img: PIL image or torch tensor — used only for its h/w to match
                generate()'s lat_h/lat_w derivation.
            max_area, frame_num, chunk_size: shape parameters; must match
                the subsequent generate() call to be effective.
            text_seq_len: T5 sequence length (defaults to config.text_len).

        Caller pattern:
            pipe = WanI2VCausal(...)
            pipe.prewarm(img, max_area=..., frame_num=...)
            # start your timer here
            video = pipe.generate(prompt, img, ...)
        """
        if self.infer_mode != "causal_fast":
            logging.info("prewarm is only supported for infer_mode='causal_fast'; skipping.")
            return
        if getattr(self, "_warmed", False):
            return

        cfg = self.config

        # Match generate()'s shape derivation exactly.
        F = frame_num
        h, w = (img.shape[1], img.shape[2]) if hasattr(img, 'shape') else (img.size[1], img.size[0])
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // cfg.vae_stride[1] //
            cfg.patch_size[1] * cfg.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // cfg.vae_stride[2] //
            cfg.patch_size[2] * cfg.patch_size[2])
        lat_f = (F - 1) // cfg.vae_stride[0] + 1
        lat_f = int(lat_f - (lat_f % chunk_size))

        frame_seqlen = (lat_h * lat_w) // (cfg.patch_size[1] * cfg.patch_size[2])
        max_seq_len = chunk_size * frame_seqlen
        head_dim = cfg.dim // cfg.num_heads
        local_num_heads = cfg.num_heads // self.sp_size

        if self.local_attn_size > -1:
            kv_size = frame_seqlen * self.local_attn_size
        else:
            kv_size = frame_seqlen * lat_f

        transformer_dtype = self.pipe_dtype
        # generate() folds the VAE spatial stride into the Plücker channel
        # dim via rearrange 'f (h s1) (w s2) c -> (f h w) (c s1 s2)' with
        # s1=s2=vae_stride[1]=8, so control_dim=6 → 6 * 8 * 8 = 384.
        plucker_channels = 6 * cfg.vae_stride[1] * cfg.vae_stride[2]
        # T5 (umt5-xxl) hidden size; cross-attn projects t5_hidden → cfg.dim.
        t5_hidden = 4096

        warmup_self_kv = self._initialize_self_kv_cache(
            num_layers=cfg.num_layers,
            shape=[1, kv_size, local_num_heads, head_dim],
            dtype=transformer_dtype,
            device=self.device)
        warmup_cross_kv = self._initialize_crossattn_cache(
            num_layers=cfg.num_layers,
            shape=[1, text_seq_len, cfg.num_heads, head_dim],
            dtype=transformer_dtype,
            device=self.device)
        if os.environ.get("LINGBOT_KV_RING") == "1":
            kv_ring_plan(warmup_self_kv, 0, max_seq_len, self.sink_size * frame_seqlen)

        # `y` is concat([msk_4ch, vae_latent_16ch]) → 20 channels; combined
        # with latent's 16 ch at patch-embed concat, the DiT sees 36 ch in.
        dummy_latent = torch.zeros(
            16, chunk_size, lat_h, lat_w,
            device=self.device, dtype=torch.float32)
        dummy_y = torch.zeros(
            20, chunk_size, lat_h, lat_w,
            device=self.device, dtype=transformer_dtype)
        dummy_c2ws = torch.zeros(
            1, plucker_channels, chunk_size, lat_h, lat_w,
            device=self.device, dtype=self.param_dtype)
        dummy_context = torch.zeros(
            text_seq_len, t5_hidden,
            device=self.device, dtype=self.param_dtype)
        dummy_t = torch.tensor(
            [500.0], device=self.device, dtype=torch.float32)

        @contextmanager
        def _noop_no_sync():
            yield
        no_sync_model = getattr(self.model, 'no_sync', _noop_no_sync)

        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
        t0 = time.perf_counter()

        with torch.amp.autocast('cuda', dtype=self.param_dtype), \
             torch.no_grad(), \
             no_sync_model():
            _ = self.model(
                x=[dummy_latent],
                t=dummy_t,
                context=[dummy_context],
                seq_len=max_seq_len,
                y=[dummy_y],
                dit_cond_dict={"c2ws_plucker_emb": (dummy_c2ws,)},
                kv_cache=warmup_self_kv,
                crossattn_cache=warmup_cross_kv,
                current_start=0,
                max_attention_size=kv_size,
                frame_seqlen=frame_seqlen,
            )

        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            logging.info(f"WanI2VCausal.prewarm: {dt_ms:.0f} ms")

        del (warmup_self_kv, warmup_cross_kv, dummy_latent, dummy_y,
             dummy_c2ws, dummy_context, dummy_t)
        torch.cuda.empty_cache()
        self._warmed = True

    def _encode_prompts(self, prompts, offload_model):
        """T5-encode ``prompts`` via the in-memory and on-disk caches.

        Returns one context list per prompt, on ``self.device``. The encoder
        is only instantiated when a prompt misses both caches.
        """
        out, missing = {}, []
        for p in prompts:
            key = hashlib.sha256(p.encode('utf-8')).hexdigest()
            path = os.path.join(self._t5_disk_cache_dir, key + '.pt')
            if key in self._t5_cache:
                out[p] = self._t5_cache[key]
            elif os.path.isfile(path):
                out[p] = [t.to(self.device) for t in torch.load(path, map_location='cpu')]
                self._t5_cache[key] = out[p]
            else:
                missing.append((p, key, path))
        if missing:
            if self.text_encoder is None:
                self.text_encoder = T5EncoderModel(**self._t5_kwargs)
            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
            enc_device = torch.device('cpu') if self.t5_cpu else self.device
            os.makedirs(self._t5_disk_cache_dir, exist_ok=True)
            for p, key, path in missing:
                context = [t.to(self.device) for t in self.text_encoder([p], enc_device)]
                torch.save([t.cpu() for t in context], path)
                self._t5_cache[key] = context
                out[p] = context
            if not self.t5_cpu:
                # T5-XXL is 11 GB; keep it off the GPU once the prompt is cached
                # so the DiT loop has the card to itself.
                self.text_encoder.model.cpu()
                torch.cuda.empty_cache()
        return [out[p] for p in prompts]

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward_causal, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward_causal, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor, scheduler) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, F, H, W]
        xt: the input noisy data with shape [B, C, F, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt, scheduler.sigmas, scheduler.timesteps]
        )
        timestep_id = torch.argmin((timesteps - timestep).abs())
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred

        return x0_pred.to(original_dtype)

    def _syncfree_schedule(self, timesteps, dit_fusion):
        """
        LINGBOT_SYNCFREE: everything the denoising loop reads from the
        scheduler, resolved once per generate() and kept on the device, so no
        forward pays a pageable H2D copy (`.to(device)` of a CPU scalar syncs
        the stream), `nonzero` or `.item()`. The lookups reproduce the eager
        path exactly: `_convert_flow_pred_to_x0` takes the *first* schedule
        entry equal to the int64 timestep (argmin), `add_noise` the *second*
        (`index_for_timestep`) when the truncated timestep repeats — they are
        different sigmas near sigma = 1.
        """
        sched, n = self.scheduler, len(timesteps)
        x0_idx = [int(torch.argmin((sched.timesteps.double() - t).abs())) for t in timesteps]
        noise_idx = [sched.index_for_timestep(t) for t in timesteps[1:]]
        t_dev = timesteps.to(self.device)
        return dict(
            # fresh contiguous [1] tensors, as torch.stack(...).to(device) gave
            t=[t_dev[i:i + 1].clone() for i in range(n)],
            t_zero=t_dev[-1:] * (0 if dit_fusion else 0.0),
            sigma_x0=sched.sigmas.double()[x0_idx].reshape(n, 1, 1, 1, 1).to(self.device),
            sigma_noise=sched.sigmas[noise_idx].reshape(n - 1, 1, 1, 1, 1).to(self.device),
        )


    def _wanvae_decode(self, z):
        """Stock Wan2_1_VAE.decode loop with channels_last_3d inputs (same numerics; the
        stock decode() slices NCHW views, which makes cuDNN transpose every conv)."""
        vae, m = self.vae, self.vae.model
        with torch.amp.autocast("cuda", dtype=vae.dtype, enabled=vae.dtype != torch.float32 and not self._vae_half):
            m.clear_cache()
            zz = z.unsqueeze(0) / vae.scale[1].float().view(1, -1, 1, 1, 1) + vae.scale[0].float().view(1, -1, 1, 1, 1)
            x = m.conv2(zz.to(m.conv2.weight.dtype))
            outs = []
            for i in range(x.shape[2]):
                m._conv_idx = [0]
                xi = x[:, :, i:i + 1].contiguous(memory_format=torch.channels_last_3d)
                outs.append(m.decoder(xi, feat_cache=m._feat_map, feat_idx=m._conv_idx))
            m.clear_cache()
            return torch.cat(outs, 2).float().clamp_(-1, 1).squeeze(0)

    def _flashvaed_decode(self, z):
        """z: [C,T,H,W] model-space latent -> [C,F,H,W] in [-1,1]. window<=0: one uninterrupted cache."""
        fv, win, warm = self._flashvaed, self._flashvaed_window, self._flashvaed_warm
        m = fv.model
        with torch.amp.autocast("cuda", dtype=fv.dtype):
            zz = z.unsqueeze(0) / fv.scale[1].view(1, -1, 1, 1, 1) + fv.scale[0].view(1, -1, 1, 1, 1)
            x = m.conv2(zz)
            T = x.shape[2]
            win = win if win > 0 else T
            frames = []
            for s in range(0, T, win):
                m.clear_cache()
                lo = max(0, s - warm)
                for i in range(lo, min(T, s + win)):
                    m._conv_idx = [0]
                    out = m.decoder(x[:, :, i:i + 1], feat_cache=m._feat_map, feat_idx=m._conv_idx)
                    if i >= s:
                        frames.append(out)
            m.clear_cache()
            return torch.cat(frames, dim=2).squeeze(0).float().clamp_(-1, 1)

    def generate(self,
                 input_prompt,
                 img,
                 action_path,
                 chunk_size=3,
                 max_area=480 * 832,
                 frame_num=81,
                 timesteps_index=[0, 250, 500, 750],
                 shift=5.0,
                 seed=-1,
                 offload_model=True,
                 max_sequence_length=512,
                 max_attention_size=None,):
        r"""
        Generates video frames from input image and text prompt.

        Dispatches to the mode-specific implementation according to
        `self.infer_mode`:
            - "causal_fast": distilled few-step sampling (`_generate_causal_fast`)
            - "causal_pretrain": 40-step CFG sampling (`_generate_causal_pretrain`)
        """
        gen_fn = (self._generate_causal_fast
                  if self.infer_mode == "causal_fast"
                  else self._generate_causal_pretrain)
        return gen_fn(
            input_prompt,
            img,
            action_path,
            chunk_size=chunk_size,
            max_area=max_area,
            frame_num=frame_num,
            timesteps_index=timesteps_index,
            shift=shift,
            seed=seed,
            offload_model=offload_model,
            max_sequence_length=max_sequence_length,
            max_attention_size=max_attention_size)

    def _generate_causal_fast(self,
                              input_prompt,
                              img,
                              action_path,
                              chunk_size=3,
                              max_area=480 * 832,
                              frame_num=81,
                              timesteps_index=[0, 179, 358, 679],
                              shift=5.0,
                              seed=-1,
                              offload_model=True,
                              max_sequence_length=512,
                              max_attention_size=None,):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """

        if input_prompt is not None and isinstance(input_prompt, str):
            batch_size = 1
        elif input_prompt is not None and isinstance(input_prompt, list):
            batch_size = len(input_prompt)
        else:
            batch_size = 1
        # LINGBOT_BATCH=B: replicate the single stream B times along the batch
        # dim (identical inputs) to measure batched-serving cost per chunk.
        batch_size = int(os.environ.get("LINGBOT_BATCH", batch_size))
        
        assert action_path is not None, "action_path is required"
        c2ws = np.load(os.path.join(action_path, "poses.npy")) # opencv coordinate
        frame_num = ((frame_num - 1) // 4) * 4 + 1
        if self.pose_provider is not None:
            c2ws = np.tile(np.eye(4, dtype=c2ws.dtype), (frame_num, 1, 1))  # placeholder; replaced per chunk
        len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
        frame_num = min(frame_num, len_c2ws)
        c2ws = c2ws[:frame_num]

        # preprocess
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        lat_f = (F - 1) // self.vae_stride[0] + 1
        lat_f = int(lat_f - (lat_f % chunk_size))
        F = (lat_f - 1) * 4 + 1
        max_seq_len = chunk_size * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
        # Reset per-generate state: cross-attn K/V cache will be freshly
        # initialized below; the first DiT forward must compute and store.
        self._cross_attn_initialized = False

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            lat_f,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        # 2. Prepare timesteps
        self.scheduler.set_timesteps(self.num_train_timesteps, shift=shift)
        timesteps_index = parse_timesteps(os.environ.get("LINGBOT_TIMESTEPS"), timesteps_index, self.num_train_timesteps)
        if os.environ.get("LINGBOT_TIMESTEPS") and not getattr(self, "_timesteps_logged", False):
            logging.info(f"LINGBOT_TIMESTEPS={timesteps_index}: {len(timesteps_index)} denoising forwards + 1 cache write per chunk")
            self._timesteps_logged = True
        timesteps = self.scheduler.timesteps[timesteps_index]
        syncfree = os.environ.get("LINGBOT_SYNCFREE") == "1"
        # "1": sync before the pose sample in play mode; "force": also without a pose provider (bit-identity check)
        late_sample = os.environ.get("LINGBOT_LATE_SAMPLE", "1") if self.device.type == "cuda" else ""
        late_sample = late_sample if late_sample in ("1", "force") else ""
        if syncfree:
            sched_dev = self._syncfree_schedule(timesteps, os.environ.get("LINGBOT_DIT_FUSION") == "1")

        # preprocess
        context = self._encode_prompts([input_prompt], offload_model)[0]

        Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()

        # The provided intrinsics are for original image size (480p). We need to transform them according to the new image size (h, w).
        Ks = get_Ks_transformed(Ks,
                                height_org=480,
                                width_org=832,
                                height_resize=h,
                                width_resize=w,
                                height_final=h,
                                width_final=w)
        Ks = Ks[0]

        len_c2ws = len(c2ws)
        len_c2ws_ = int((len_c2ws - 1) // 4) + 1
        len_c2ws_ = int(len_c2ws_ - (len_c2ws_ % chunk_size))
        c2ws_infer = interpolate_camera_poses(
            src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
            src_rot_mat=c2ws[:, :3, :3],
            src_trans_vec=c2ws[:, :3, 3],
            tgt_indices=np.linspace(0, len_c2ws - 1, len_c2ws_),
        )
        c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
        Ks = Ks.repeat(len(c2ws_infer), 1)

        c2ws_infer = c2ws_infer.to(self.device)
        Ks = Ks.to(self.device)
        wasd_action = None
        c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w)
        c2ws_plucker_emb = rearrange(
            c2ws_plucker_emb,
            'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
            c1=int(h // lat_h),
            c2=int(w // lat_w),
        )
        c2ws_plucker_emb = c2ws_plucker_emb[None, ...] # [b, f*h*w, c]
        c2ws_plucker_emb = rearrange(c2ws_plucker_emb, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
        if wasd_action is not None:
            wasd_action_tensor = wasd_action[:, None, None, :].repeat(1, h, w, 1) # [f, h, w, 3]
            wasd_action_tensor = rearrange(
                wasd_action_tensor,
                'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                c1=int(h // lat_h),
                c2=int(w // lat_w),
            )
            wasd_action_tensor = wasd_action_tensor[None, ...] # [b, f*h*w, c]
            wasd_action_tensor = rearrange(wasd_action_tensor, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
            c2ws_plucker_emb = torch.cat([c2ws_plucker_emb, wasd_action_tensor], dim=1)

        y_key = (hash(img.cpu().numpy().tobytes()), F, h, w)
        if self._y_cache is not None and self._y_cache[0] == y_key:
            y = self._y_cache[1]
        else:
            y = self.vae.encode([
                torch.concat([
                    torch.nn.functional.interpolate(
                        img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                            0, 1),
                    torch.zeros(3, F - 1, h, w)
                ],
                             dim=1).to(self.device)
            ])[0]
            self._y_cache = (y_key, y)
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_model = getattr(self.model, 'no_sync', noop_no_sync)

        # Initialize KV cache to all zeros
        model_args = self.model.config
        transformer_dtype = self.pipe_dtype
        frame_seqlen = int(noise.shape[-2] * noise.shape[-1]// 4)
        if self.local_attn_size > -1:
            kv_size = frame_seqlen * self.local_attn_size
        else:
            kv_size = frame_seqlen * lat_f
        head_dim = model_args.dim // model_args.num_heads
        local_num_heads = model_args.num_heads // self.sp_size
        self_kv_shape = [batch_size, kv_size, local_num_heads, head_dim]
        self_kv_cache = self._initialize_self_kv_cache(num_layers=model_args.num_layers,
                                                       shape=self_kv_shape,
                                                       dtype=transformer_dtype,
                                                       device=self.device)
        cross_kv_shape = [batch_size, max_sequence_length, model_args.num_heads, head_dim]
        cross_kv_cache = self._initialize_crossattn_cache(num_layers=model_args.num_layers,
                                                          shape=cross_kv_shape,
                                                          dtype=transformer_dtype,
                                                          device=self.device)
        dit_fusion = os.environ.get("LINGBOT_DIT_FUSION") == "1"
        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_model(),
        ):
            if dit_fusion:
                # Fill the cross-attn K/V once here (same autocast as the
                # loop) so no forward takes the first-call branch, and
                # allocate the per-chunk camera-modulation cache.
                self.model.init_crossattn_cache([context[0]] * batch_size, cross_kv_cache)
                self._cross_attn_initialized = True
                cam_cache = self._initialize_cam_cache(
                    num_layers=model_args.num_layers,
                    shape=[batch_size, max_seq_len, model_args.dim],
                    dtype=transformer_dtype, device=self.device)
            # sample videos
            latent = noise
            latents_chunk = latent.split(chunk_size, dim=1) # [c, f, h, w]
            condition_chunk = y.split(chunk_size, dim=1)
            c2ws_plucker_emb_chunk = c2ws_plucker_emb.split(chunk_size, dim=2)
            num_inference_chunk = len(latents_chunk)
            pred_latent_chunks = []
            # LINGBOT_VAE_STREAM=1: decode chunk N on a side stream while chunk N+1 denoises
            # (fused driver's streaming decode_step); the final whole-clip decode is skipped.
            vae_stream_on = os.environ.get("LINGBOT_VAE_STREAM") == "1" and self._vae_fused is not None
            if vae_stream_on:
                vae_stream, dec_state, dec_pending, dec_frames = torch.cuda.Stream(), None, None, []
            # LINGBOT_DECODE_FIRST=1: decode right after x0 on the side stream, overlapped with this
            # chunk's cache-write forward only, then the main stream waits before the next chunk, so
            # the chunk's frames are complete before the next forward and gen_start stays honest. (An
            # async variant without the wait was tried; its only upside was ~40 ms of first-frame time.)
            decode_first = os.environ.get("LINGBOT_DECODE_FIRST") == "1"
            if self.frame_sink is not None:
                assert vae_stream_on, "frame_sink needs LINGBOT_VAE_STREAM=1 and LINGBOT_VAE_FUSED"
                chunk_t0 = []
            bench_timing = os.environ.get("LINGBOT_BENCH_TIMING") == "1"
            if bench_timing:
                torch.cuda.synchronize()
                t_loop0 = t_prev = time.perf_counter()
            # Optional torch.profiler window over steady-state chunks
            # (LINGBOT_PROFILE=<out_dir>, LINGBOT_PROFILE_CHUNKS=8-10).
            if os.environ.get("LINGBOT_ROOFLINE"):
                os.environ.setdefault("LINGBOT_PROFILE", os.environ["LINGBOT_ROOFLINE"])
                os.environ.setdefault("LINGBOT_PROFILE_VAE", "1")
            prof_dir = os.environ.get("LINGBOT_PROFILE")
            prof, prof_lo, prof_hi = None, -1, -1
            tracer = None
            if prof_dir:
                lo, hi = os.environ.get("LINGBOT_PROFILE_CHUNKS", "8-10").split("-")
                prof_lo, prof_hi = int(lo) - 1, int(hi) - 1  # 1-based in env, 0-based here
            for chunk_id in tqdm(range(num_inference_chunk)):
                if prof_dir and chunk_id == prof_lo:
                    torch.cuda.synchronize()
                    prof = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=True, with_flops=True)
                    prof.__enter__()
                    if os.environ.get("LINGBOT_ROOFLINE_TRACE") == "1":
                        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
                        from roofline import ByteTracer
                        tracer = ByteTracer(); tracer.__enter__()
                _rf_chunk = torch.profiler.record_function(f"chunk{chunk_id}"); _rf_chunk.__enter__()
                if (self.pose_provider is not None or late_sample == "force") and late_sample and chunk_id > 0:
                    # sample the input when the GPU can actually start this chunk: the host runs ~0.35 s ahead
                    # (the previous chunk's latent 2-4 decodes + KV write are still queued) and the pageable
                    # .to(device) below blocks on them anyway, so this moves the sample later at no cost
                    torch.cuda.current_stream().synchronize()
                if self.chunk_gate is not None:
                    self.chunk_gate(chunk_id)   # stream/live.py: may block (just-in-time generation); before the pose sample
                if self.frame_sink is not None:
                    chunk_t0.append(time.monotonic())
                _rf = torch.profiler.record_function("prep"); _rf.__enter__()
                current_latent = latents_chunk[chunk_id]
                current_condition = condition_chunk[chunk_id]
                current_c2ws_plucker_emb = c2ws_plucker_emb_chunk[chunk_id]
                if self.pose_provider is not None:
                    rel = torch.from_numpy(np.asarray(self.pose_provider(chunk_id, chunk_size), dtype=np.float32)).to(self.device)
                    emb = get_plucker_embeddings(rel, Ks[:chunk_size], h, w)
                    emb = rearrange(emb, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)', c1=int(h // lat_h), c2=int(w // lat_w))
                    current_c2ws_plucker_emb = rearrange(emb[None], 'b (f h w) c -> b c f h w', f=chunk_size, h=lat_h, w=lat_w).to(self.param_dtype)

                dit_cond_dict = {
                    "c2ws_plucker_emb": current_c2ws_plucker_emb.chunk(1, dim=0),
                }

                kwargs = {
                    'context': [context[0]] * batch_size,
                    'seq_len': max_seq_len,
                    'y': [current_condition] * batch_size,
                    'dit_cond_dict': dit_cond_dict,
                    'kv_cache': self_kv_cache,
                    'crossattn_cache': cross_kv_cache,
                    'current_start': chunk_id * chunk_size * frame_seqlen,
                    'max_attention_size': kv_size if max_attention_size is None else max_attention_size,
                    'frame_seqlen': frame_seqlen,
                }
                if dit_fusion:
                    kwargs['cam_cache'] = cam_cache

                if offload_model:
                    torch.cuda.empty_cache()
                if os.environ.get("LINGBOT_KV_RING") == "1":
                    kv_ring_plan(self_kv_cache, kwargs['current_start'],
                                 chunk_size * frame_seqlen, self.sink_size * frame_seqlen)

                _rf.__exit__(None, None, None)
                if vae_stream_on and dec_pending is not None:
                    with torch.profiler.record_function("vae_stream_launch"), torch.no_grad():
                        vae_stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(vae_stream):
                            dec_pending.record_stream(vae_stream)
                            fr, dec_state = self._vae_fused.decode_step(dec_pending, dec_state)
                            if self.frame_sink is not None:
                                self.frame_sink(chunk_id - 1, fr, vae_stream, chunk_t0[chunk_id - 1])
                            else:
                                dec_frames.append(fr)
                    dec_pending = None
                for timestep_idx in range(len(timesteps)):
                    _rf = torch.profiler.record_function(f"denoise_step{timestep_idx}"); _rf.__enter__()
                    latent_model_input = [current_latent.to(self.device)] * batch_size
                    current_timestep = [timesteps[timestep_idx]]

                    timestep = sched_dev['t'][timestep_idx] if syncfree else torch.stack(current_timestep).to(self.device)

                    if dit_fusion:
                        # cam MLP computed on the chunk's first forward, reused after
                        kwargs['cam_first_call'] = timestep_idx == 0
                    noise_pred = self.model(
                        x=latent_model_input, t=timestep,
                        cross_attn_first_call=not self._cross_attn_initialized,
                        **kwargs)[0]
                    self._cross_attn_initialized = True

                    if offload_model:
                        torch.cuda.empty_cache()

                    if syncfree:
                        # same ops and dtypes as _convert_flow_pred_to_x0 / add_noise
                        x0 = (current_latent.double() - sched_dev['sigma_x0'][timestep_idx] * noise_pred.double()).to(noise_pred.dtype)
                    else:
                        x0 = self._convert_flow_pred_to_x0(
                            flow_pred=noise_pred,
                            xt=current_latent,
                            timestep=current_timestep[0],
                            scheduler=self.scheduler,
                        )

                    if timestep_idx < len(timesteps) - 1:
                        next_timestep = timesteps[timestep_idx + 1]
                        next_noise = torch.randn(x0.shape, generator=seed_g, device=x0.device, dtype=x0.dtype)
                        if syncfree:
                            assert x0.dtype == sched_dev['sigma_noise'].dtype
                            sigma_t = sched_dev['sigma_noise'][timestep_idx]
                            current_latent = (1 - sigma_t) * x0 + sigma_t * next_noise
                        else:
                            current_latent = self.scheduler.add_noise(x0, next_noise, next_timestep)
                        _rf.__exit__(None, None, None)
                    else:
                        # note return x0
                        _rf.__exit__(None, None, None)
                        break

                pred_latent_chunks.append(x0)
                if vae_stream_on and decode_first:
                    # LINGBOT_DECODE_FIRST=1: decode this chunk now, latent by latent, and hand each
                    # latent's frames to the sink as they finish; the main stream then waits, so the
                    # frames leave before the next forward instead of competing with it on a
                    # saturated GPU (same throughput, first frame ~0.5 s earlier).
                    with torch.profiler.record_function("vae_decode_first"), torch.no_grad():
                        vae_stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(vae_stream):
                            x0.record_stream(vae_stream)
                            off = 0
                            for li in range(x0.shape[1]):
                                fr, dec_state = self._vae_fused.decode_step(x0[:, li:li + 1], dec_state)
                                if self.frame_sink is not None:
                                    self.frame_sink(chunk_id, fr, vae_stream, chunk_t0[chunk_id], off)
                                else:
                                    dec_frames.append(fr)
                                off += fr.shape[1]
                elif vae_stream_on:
                    dec_pending = x0
                _rf = torch.profiler.record_function("cache_write"); _rf.__enter__()

                # Update kv cache
                if dit_fusion:
                    # int64 like the denoising steps: a float t here would be
                    # a second dtype variant of the compiled graph. The
                    # embedding casts t to float64 either way (same value).
                    context_timestep = [timesteps[-1] * 0]
                    kwargs['cam_first_call'] = False
                else:
                    context_timestep = [timesteps[-1] * 0.0]
                timestep = sched_dev['t_zero'] if syncfree else torch.stack(context_timestep).to(self.device)
                self.model(x=[x0] * batch_size, t=timestep,
                           cross_attn_first_call=False,
                           **kwargs)
                _rf.__exit__(None, None, None)
                if vae_stream_on and decode_first:
                    torch.cuda.current_stream().wait_stream(vae_stream)  # decode done before the next chunk
                if bench_timing:
                    with torch.profiler.record_function("bench_sync"):
                        torch.cuda.synchronize()
                    now = time.perf_counter()
                    logging.info(f"BENCH chunk={chunk_id} chunk_s={now - t_prev:.3f} loop_s={now - t_loop0:.3f}")
                    self.bench_chunk_s = getattr(self, "bench_chunk_s", []) + [now - t_prev]
                    t_prev = now
                _rf_chunk.__exit__(None, None, None)
                if prof is not None and chunk_id == prof_hi:
                    torch.cuda.synchronize()
                    if tracer is not None:
                        tracer.__exit__(None, None, None); os.makedirs(prof_dir, exist_ok=True)
                        tracer.dump(os.path.join(prof_dir, "bytes.json")); tracer = None
                    prof.__exit__(None, None, None)
                    os.makedirs(prof_dir, exist_ok=True)
                    prof.export_chrome_trace(os.path.join(prof_dir, "trace.json.gz"))
                    ka = prof.key_averages()
                    with open(os.path.join(prof_dir, "kernels_top.txt"), "w") as f:
                        f.write(ka.table(sort_by="self_device_time_total", row_limit=80))
                    rows = [{"name": e.key, "self_cuda_us": e.self_device_time_total, "cuda_us": e.device_time_total,
                             "cpu_us": e.self_cpu_time_total, "count": e.count, "flops": getattr(e, "flops", 0) or 0,
                             "shapes": str(getattr(e, "input_shapes", ""))[:200]} for e in ka]
                    with open(os.path.join(prof_dir, "kernels.json"), "w") as f:
                        json.dump({"chunks": [prof_lo + 1, prof_hi + 1], "events": rows}, f)
                    kas = prof.key_averages(group_by_input_shape=True)
                    with open(os.path.join(prof_dir, "kernels_shapes.json"), "w") as f:
                        json.dump([{"name": e.key, "count": e.count, "cuda_us": e.device_time_total, "shapes": str(e.input_shapes)[:400]}
                                   for e in kas if e.input_shapes], f)
                    logging.info(f"BENCH profile written to {prof_dir} (chunks {prof_lo + 1}-{prof_hi + 1})")
                    prof = None

            pred_latent_chunks = torch.cat(pred_latent_chunks, dim=1)
            if dit_fusion:
                kwargs['cam_cache'] = cam_cache = None  # ~1.1 GB; free before the decode
            if _SAGE_KVQ:
                sage_kvq.release(self_kv_cache)  # ~2.5 GB; same reason

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                # not under vae_stream_on: the whole-clip decode never runs there and decode_step is warm since
                # chunk 0, so the block would only add ~0.4 s to the first rollout boundary of a live session
                if os.environ.get("LINGBOT_VAE_WARM") == "1" and (self._vae_cl or self._vae_half or self._vae_fused) \
                        and not vae_stream_on and not getattr(self, "_vae_warmed", False):
                    self._vae_warmed = True
                    # one-time compile/autotune of the decoder happens here, not in the timed decode
                    with torch.no_grad():
                        if self._vae_fused is not None:
                            self._vae_fused.decode(pred_latent_chunks[:, :5])
                        else:
                            self._wanvae_decode(pred_latent_chunks[:, :5])
                dump = os.environ.get("LINGBOT_DUMP_LATENTS")
                if dump:
                    torch.save(pred_latent_chunks.detach().cpu(), dump)  # outside the timed decode
                if bench_timing:
                    torch.cuda.synchronize()
                    t_dec0 = time.perf_counter()
                vprof = vtracer = vflops = None
                if os.environ.get("LINGBOT_PROFILE") and os.environ.get("LINGBOT_PROFILE_VAE") == "1" and not vae_stream_on:
                    vprof = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                    vprof.__enter__()
                    if os.environ.get("LINGBOT_ROOFLINE_TRACE") == "1":
                        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
                        from roofline import ByteTracer
                        from torch.utils.flop_counter import FlopCounterMode
                        vflops = FlopCounterMode(display=False); vflops.__enter__()
                        vtracer = ByteTracer(); vtracer.__enter__()
                if vae_stream_on:
                    with torch.no_grad():
                        vae_stream.wait_stream(torch.cuda.current_stream())
                        if dec_pending is not None:
                            with torch.cuda.stream(vae_stream):
                                fr, dec_state = self._vae_fused.decode_step(dec_pending, dec_state)
                                if self.frame_sink is not None:
                                    self.frame_sink(num_inference_chunk - 1, fr, vae_stream, chunk_t0[-1])
                                else:
                                    dec_frames.append(fr)
                        torch.cuda.current_stream().wait_stream(vae_stream)
                        videos = [torch.cat(dec_frames, 1)] if dec_frames else [None]
                elif self._flashvaed is not None:
                    with torch.no_grad():
                        videos = [self._flashvaed_decode(pred_latent_chunks)]
                elif self._taehv is not None:
                    with torch.no_grad():
                        z = pred_latent_chunks
                        if os.environ.get("LINGBOT_TAEHV_SCALE") == "vae":
                            # decoders trained on the Wan VAE's own latent space (LightTAE)
                            # take x0 * std + mean, i.e. the canonical decoder's input.
                            mean, inv_std = self.vae.scale
                            z = z / inv_std.view(-1, 1, 1, 1).to(z) + mean.view(-1, 1, 1, 1).to(z)
                        z = z.permute(1, 0, 2, 3)[None].to(torch.float16)  # [C,T,H,W] -> [1,T,C,H,W]
                        rgb = self._taehv.decode_video(z, parallel=False, show_progress_bar=False)  # [1,T,3,H,W] in [0,1]; sequential = low memory
                    videos = [rgb[0].permute(1, 0, 2, 3).float().mul_(2).sub_(1).clamp_(-1, 1)]  # [3,T,H,W] in [-1,1]
                elif self._vae_fused is not None:
                    videos = [self._vae_fused.decode(pred_latent_chunks)]
                elif self._vae_cl or self._vae_half:
                    with torch.no_grad():
                        videos = [self._wanvae_decode(pred_latent_chunks)]
                else:
                    videos = self.vae.decode([pred_latent_chunks])
                if vprof is not None:
                    torch.cuda.synchronize()
                    os.makedirs(os.environ["LINGBOT_PROFILE"], exist_ok=True)
                    if vtracer is not None:
                        vtracer.__exit__(None, None, None); vtracer.dump(os.path.join(os.environ["LINGBOT_PROFILE"], "bytes_vae.json"))
                        vflops.__exit__(None, None, None)
                        conv = sum(v for k, v in vflops.get_flop_counts()["Global"].items() if "conv" in str(k))
                        json.dump({"conv_flops": conv, "total_flops": vflops.get_total_flops()}, open(os.path.join(os.environ["LINGBOT_PROFILE"], "flops_vae.json"), "w"))
                    json.dump({"chunks": num_inference_chunk}, open(os.path.join(os.environ["LINGBOT_PROFILE"], "vae_meta.json"), "w"))
                    vprof.__exit__(None, None, None)
                    vprof.export_chrome_trace(os.path.join(os.environ["LINGBOT_PROFILE"], "trace_vae.json.gz"))
                if bench_timing:
                    torch.cuda.synchronize()
                    logging.info(f"BENCH vae_decode_s={time.perf_counter() - t_dec0:.3f}")
                    self.bench_vae_decode_s = time.perf_counter() - t_dec0

        # del noise, latent, x0
        # del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def _generate_causal_pretrain(self,
                                  input_prompt,
                                  img,
                                  action_path,
                                  chunk_size=3,
                                  max_area=480 * 832,
                                  frame_num=81,
                                  timesteps_index=None,
                                  shift=5.0,
                                  seed=-1,
                                  offload_model=True,
                                  max_sequence_length=512,
                                  max_attention_size=None,):
        r"""
        Generates video frames with the pretrained causal model using
        40-step CFG sampling per chunk. `timesteps_index` is unused in this
        mode (kept for signature compatibility with `generate`).
        """
        guide_scale = 5.0
        n_prompt = "画面突变，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

        if input_prompt is not None and isinstance(input_prompt, list):
            batch_size = len(input_prompt)
        else:
            batch_size = 1

        if action_path is not None:
            c2ws = np.load(os.path.join(action_path, "poses.npy"))  # opencv coordinate
            len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
            frame_num = ((frame_num - 1) // 4) * 4 + 1
            frame_num = min(frame_num, len_c2ws)
            c2ws = c2ws[:frame_num]

        # preprocess
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        lat_f = (F - 1) // self.vae_stride[0] + 1
        lat_f = int(lat_f - (lat_f % chunk_size))
        F = (lat_f - 1) * 4 + 1
        max_seq_len = chunk_size * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16, lat_f, lat_h, lat_w,
            dtype=torch.float32, generator=seed_g, device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        # 2. Prepare timesteps (scheduler object created once, state reset per chunk)
        sample_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False)

        # preprocess text: cond + uncond
        context, context_null = self._encode_prompts([input_prompt, n_prompt], offload_model)

        # cam preparation (only if action_path is provided)
        c2ws_plucker_emb = None
        if action_path is not None:
            Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()
            Ks = get_Ks_transformed(Ks,
                                    height_org=480, width_org=832,
                                    height_resize=h, width_resize=w,
                                    height_final=h, width_final=w)
            Ks = Ks[0]

            len_c2ws = len(c2ws)
            len_c2ws_ = int((len_c2ws - 1) // 4) + 1
            len_c2ws_ = int(len_c2ws_ - (len_c2ws_ % chunk_size))
            c2ws_infer = interpolate_camera_poses(
                src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
                src_rot_mat=c2ws[:, :3, :3],
                src_trans_vec=c2ws[:, :3, 3],
                tgt_indices=np.linspace(0, len_c2ws - 1, len_c2ws_),
            )
            c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
            Ks = Ks.repeat(len(c2ws_infer), 1)

            c2ws_infer = c2ws_infer.to(self.device)
            Ks = Ks.to(self.device)
            c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w)
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb,
                'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                c1=int(h // lat_h), c2=int(w // lat_w),
            )
            c2ws_plucker_emb = c2ws_plucker_emb[None, ...]  # [b, f*h*w, c]
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb, 'b (f h w) c -> b c f h w',
                f=lat_f, h=lat_h, w=lat_w,
            ).to(self.param_dtype)

        y = self.vae.encode([
            torch.concat(
                [
                    torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
                    torch.zeros(3, F - 1, h, w)
                ], 
                dim=1,
            ).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_model = getattr(self.model, 'no_sync', noop_no_sync)

        model_args = self.model.config
        transformer_dtype = self.pipe_dtype
        frame_seqlen = int(noise.shape[-2] * noise.shape[-1] // 4)
        kv_size = frame_seqlen * lat_f
        head_dim = model_args.dim // model_args.num_heads
        local_num_heads = model_args.num_heads // self.sp_size
        self_kv_shape = [batch_size, kv_size, local_num_heads, head_dim]
        cross_kv_shape = [batch_size, max_sequence_length, model_args.num_heads, head_dim]

        # CFG requires separate caches for the cond / uncond streams.
        self_kv_cache_cond = self._initialize_self_kv_cache(
            num_layers=model_args.num_layers,
            shape=self_kv_shape,
            dtype=transformer_dtype,
            device=self.device)
        self_kv_cache_uncond = self._initialize_self_kv_cache(
            num_layers=model_args.num_layers,
            shape=self_kv_shape,
            dtype=transformer_dtype,
            device=self.device)
        cross_kv_cache_cond = self._initialize_crossattn_cache_pretrain(
            num_layers=model_args.num_layers,
            shape=cross_kv_shape,
            dtype=transformer_dtype,
            device=self.device)
        cross_kv_cache_uncond = self._initialize_crossattn_cache_pretrain(
            num_layers=model_args.num_layers,
            shape=cross_kv_shape,
            dtype=transformer_dtype,
            device=self.device)

        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_model(),
        ):
            latent = noise
            latents_chunk = latent.split(chunk_size, dim=1)          # [c, f, h, w]
            condition_chunk = y.split(chunk_size, dim=1)
            c2ws_plucker_emb_chunk = c2ws_plucker_emb.split(chunk_size, dim=2)
            num_inference_chunk = len(latents_chunk)
            pred_latent_chunks = []

            for chunk_id in tqdm(range(num_inference_chunk)):
                # Reset the multi-step scheduler state for each chunk
                sample_scheduler.set_timesteps(40, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps

                current_latent = latents_chunk[chunk_id]
                current_condition = condition_chunk[chunk_id]
                current_c2ws_plucker_emb = c2ws_plucker_emb_chunk[chunk_id]
                dit_cond_dict = {
                    "c2ws_plucker_emb": current_c2ws_plucker_emb.chunk(1, dim=0),
                }

                common = {
                    'seq_len': max_seq_len,
                    'y': [current_condition],
                    'dit_cond_dict': dit_cond_dict,          # camera condition is kept for uncond as well
                    'current_start': chunk_id * chunk_size * frame_seqlen,
                    'max_attention_size': kv_size if max_attention_size is None else max_attention_size,
                }
                kwargs_cond = {
                    **common,
                    'context': [context[0]],
                    'kv_cache': self_kv_cache_cond,
                    'crossattn_cache': cross_kv_cache_cond,
                }
                kwargs_uncond = {
                    **common,
                    'context': context_null,
                    'kv_cache': self_kv_cache_uncond,
                    'crossattn_cache': cross_kv_cache_uncond,
                }

                if offload_model:
                    torch.cuda.empty_cache()

                for timestep_idx in tqdm(range(len(timesteps)), desc=f"infer chunk {chunk_id}"):
                    latent_model_input = [current_latent.to(self.device)]
                    t = timesteps[timestep_idx]
                    timestep = torch.stack([t]).to(self.device)

                    noise_pred_cond = self.model(
                        x=latent_model_input, t=timestep, **kwargs_cond)[0]
                    noise_pred_uncond = self.model(
                        x=latent_model_input, t=timestep, **kwargs_uncond)[0]
                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond)

                    if offload_model:
                        torch.cuda.empty_cache()

                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0), t,
                        current_latent.unsqueeze(0),
                        return_dict=False, generator=seed_g)[0]
                    current_latent = temp_x0.squeeze(0)

                    del latent_model_input, timestep

                pred_latent_chunks.append(current_latent)

                # Update both self KV caches with the clean latent (once for cond, once for uncond)
                timestep0 = torch.stack([timesteps[-1] * 0.0]).to(self.device)
                self.model(x=[current_latent], t=timestep0, **kwargs_cond)
                self.model(x=[current_latent], t=timestep0, **kwargs_uncond)

            pred_latent_chunks = torch.cat(pred_latent_chunks, dim=1)

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode([pred_latent_chunks])

        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def _initialize_self_kv_cache(self, num_layers, shape, dtype, device):
        """
        Initialize a Per-GPU KV cache for the SelfAttn.
        """
        self_kv_cache = []
        for _ in range(num_layers):
            self_kv_cache.append({
                'k': torch.zeros(shape, dtype=dtype, device=device),
                'v': torch.zeros(shape, dtype=dtype, device=device),
                'global_end_index': torch.tensor([0], dtype=torch.long, device=device),
                'local_end_index': torch.tensor([0], dtype=torch.long, device=device),
                # Python-int mirrors of the two indices above. The attention
                # layers advance these host-side so the eviction schedule
                # needs no .item() GPU syncs; the tensors are kept in sync
                # for external readers.
                'global_end_int': 0,
                'local_end_int': 0,
            })
            if _SAGE_KVQ:
                self_kv_cache[-1].update(sage_kvq.alloc_cache(shape, dtype, device))

        return self_kv_cache


    def _initialize_crossattn_cache(self, num_layers, shape, dtype, device):
        """
        Initialize a per-GPU cross-attention cache.
        """
        crossattn_cache = []
        for _ in range(num_layers):
            crossattn_cache.append({
                'k': torch.zeros(shape, dtype=dtype, device=device),
                'v': torch.zeros(shape, dtype=dtype, device=device),
                'is_init': torch.tensor(0, dtype=torch.int32, device=device),
            })

        return crossattn_cache

    def _initialize_cam_cache(self, num_layers, shape, dtype, device):
        """
        Per-layer camera-modulation cache (LINGBOT_DIT_FUSION): the cam MLP
        output is fixed for a chunk, so it is computed once per chunk.
        """
        return [{
            'scale': torch.zeros(shape, dtype=dtype, device=device),
            'shift': torch.zeros(shape, dtype=dtype, device=device),
        } for _ in range(num_layers)]

    def _initialize_crossattn_cache_pretrain(self, num_layers, shape, dtype, device):
        """
        Initialize a per-GPU cross-attention cache for the pretrained causal
        model, which expects `is_init` to be a plain Python bool.
        """
        crossattn_cache = []
        for _ in range(num_layers):
            crossattn_cache.append({
                'k': torch.zeros(shape, dtype=dtype, device=device),
                'v': torch.zeros(shape, dtype=dtype, device=device),
                'is_init': False,
            })

        return crossattn_cache