"""The LingBot-World 2.0 generation loop: one chunk of 4 latent frames at a time.

Per chunk: 4 denoising passes of the fused DiT, one pass at t=0 that writes the clean chunk into
the KV cache, then the fused decoder. Moved from the paper's `wan/image2video.py` (`WanI2VCausal`,
causal_fast mode) with the optimizations of OPTIMIZATIONS.md; the paper's own loop is in
`reference/wan/image2video.py`.
"""
import gc
import hashlib
import json
import logging
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from einops import rearrange
from tqdm import tqdm

from lingbot.layers import kv_cache as kvc
from lingbot.layers.linear import convert_to_fp8
from lingbot.models.lingbot_world.text_encoder import TextEncoder
from lingbot.models.lingbot_world.transformer import WanModelFast
from lingbot.models.lingbot_world.vae import FusedDecoder
from lingbot.profiling import ChunkProfiler, DecodeProfiler
from wan.modules.vae2_1 import Wan2_1_VAE
from wan.utils.cam_utils import (
    compute_relative_poses,
    get_Ks_transformed,
    get_plucker_embeddings,
    interpolate_camera_poses,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def _resolve_asset_path(filename, checkpoint_dir, assets_dir=None):
    """Resolve T5 / VAE / tokenizer paths, falling back to ``assets_dir``.

    The 1.3B Hugging Face upload ships only DiT weights; T5, VAE and the tokenizer can be reused
    from the 14B release via ``--assets_dir``.
    """
    candidates = [os.path.join(checkpoint_dir, filename)]
    if assets_dir:
        candidates.append(os.path.join(assets_dir, filename))
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Required asset {filename!r} not found. Looked in: {', '.join(candidates)}. "
        "Pass --assets_dir pointing at a 14B (or Wan) checkpoint that contains "
        "models_t5_umt5-xxl-enc-bf16.pth, Wan2.1_VAE.pth, and google/umt5-xxl."
    )


def _resolve_dit_dir(checkpoint_dir, subfolder):
    """Prefer ``checkpoint_dir/subfolder`` when it exists (14B layout), else the root (1.3B)."""
    if subfolder:
        candidate = os.path.join(checkpoint_dir, subfolder)
        if os.path.isdir(candidate):
            return candidate
    return checkpoint_dir


def _load_safetensors_state_dict(dit_dir, device="cpu"):
    """Load a (possibly sharded) safetensors dump from ``dit_dir``."""
    from safetensors.torch import load_file

    device = str(device)
    for index_name in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"):
        index_path = os.path.join(dit_dir, index_name)
        if not os.path.isfile(index_path):
            continue
        with open(index_path) as f:
            index = json.load(f)
        state = {}
        for shard in sorted(set(index["weight_map"].values())):
            state.update(load_file(os.path.join(dit_dir, shard), device=device))
        return state
    for single_name in ("model.safetensors", "diffusion_pytorch_model.safetensors"):
        single_path = os.path.join(dit_dir, single_name)
        if os.path.isfile(single_path):
            return load_file(single_path, device=device)
    raise FileNotFoundError(
        f"No safetensors weights found in {dit_dir}. Expected a sharded "
        "index (model.safetensors.index.json) or a single model.safetensors.")


def _inductor_tune():
    """LINGBOT_INDUCTOR_TUNE=1: Inductor knobs for the memory-bound Triton kernels (LN/modulation/
    quant reductions over 1536- and 8960-wide rows). multi_kernel benchmarks persistent vs looped
    reductions and coordinate_descent_tuning tunes block sizes at first compile; the reduction order
    may change -> fp32-ulp differences, not bitwise. realize_reads_threshold keeps the norm2
    LN+modulation output inlined so its FP8 amax fuses."""
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


def load_dit_model(model_cls, checkpoint_dir, subfolder, config, torch_dtype, extra=None, device="cpu"):
    """Load the DiT with ``from_pretrained`` when ``config.json`` exists, else build it from the
    task config and load the safetensors shards (the 1.3B Hugging Face layout)."""
    extra = extra or {}
    dit_dir = _resolve_dit_dir(checkpoint_dir, subfolder)
    logging.info(f"Loading {model_cls.__name__} from {dit_dir}")
    if os.path.isfile(os.path.join(dit_dir, "config.json")):
        return model_cls.from_pretrained(dit_dir, torch_dtype=torch_dtype, **extra)
    logging.info(f"config.json not found in {dit_dir}; building {model_cls.__name__} "
                 "from the task config and loading safetensors weights.")
    # Build directly on the target device: random init of 1.3B params is ~1 s on the GPU vs
    # minutes on a contended CPU, and the weights then load straight there.
    with torch.device(device):
        model = model_cls(**_dit_kwargs_from_config(config, extra))
    state = _load_safetensors_state_dict(dit_dir, device=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logging.warning(f"Missing keys when loading DiT: {missing}")
    if unexpected:
        logging.warning(f"Unexpected keys when loading DiT: {unexpected}")
    return model.to(dtype=torch_dtype)


class LingBotWorldPipeline:
    """Image + text + camera path -> video, chunk by chunk, on one GPU.

    Hooks used by `lingbot play` (lingbot/play/live.py):
      frame_sink(chunk_id, frames[C,F,H,W] in [-1,1], vae_stream, chunk_gen_start[, offset])
          receives each decoded chunk on the decoder stream (needs LINGBOT_VAE_STREAM=1)
      pose_provider(chunk_id, n_latents) -> [n,4,4] framewise relative OpenCV poses for the chunk;
          replaces poses.npy (frame_num then sets the rollout length) and may raise to end it
      chunk_gate(chunk_id) is called before each chunk's pose sample and may block
    """

    def __init__(self, config, checkpoint_dir, device_id=0, rank=0, t5_cpu=False, pipe_dtype=torch.bfloat16,
                 local_attn_size=-1, sink_size=0, assets_dir=None, decoder_device_id=None):
        self.device = torch.device(f"cuda:{device_id}")

        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        self.pipe_dtype = pipe_dtype
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size

        # T5 is built lazily: prompt embeddings are cached on disk, so the 11 GB encoder is only
        # constructed for a prompt nobody has encoded yet.
        self._t5_kwargs = dict(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=_resolve_asset_path(config.t5_checkpoint, checkpoint_dir, assets_dir),
            tokenizer_path=_resolve_asset_path(config.t5_tokenizer, checkpoint_dir, assets_dir),
        )
        self.text_encoder = None
        self._t5_disk_cache_dir = os.path.join(assets_dir or checkpoint_dir, "t5_cache")
        # Same-prompt re-encodes hit this dict (sha256(prompt) -> device-resident context).
        self._t5_cache: dict[str, list] = {}

        # The encoder (start-image conditioning) stays fp32 as in the paper; only the decoder is
        # rebuilt as the fused fp16 driver. LINGBOT_VAE_FUSED: '1' = max-autotune-no-cudagraphs,
        # 'eager', or a torch.compile mode.
        self.vae = Wan2_1_VAE(vae_pth=_resolve_asset_path(config.vae_checkpoint, checkpoint_dir, assets_dir),
                              dtype=torch.float, device=self.device)
        fused = os.environ.get("LINGBOT_VAE_FUSED", "1")
        self._decoder_mode = {"1": "max-autotune-no-cudagraphs", "eager": None}.get(fused, fused)
        self._vae_path = _resolve_asset_path(config.vae_checkpoint, checkpoint_dir, assets_dir)
        self._decoders = {}
        self.set_decoder_device(decoder_device_id)

        self.frame_sink = None
        self.pose_provider = None
        self.chunk_gate = None
        self._y_cache = None  # (key, y): the I2V conditioning latent is a pure function of (image, F, h, w)

        self.model = load_dit_model(WanModelFast, checkpoint_dir, config.fast_checkpoint, config, torch.bfloat16,
                                    extra=dict(local_attn_size=self.local_attn_size, sink_size=self.sink_size),
                                    device=self.device)
        self.model.eval().requires_grad_(False)

        # Rowwise FP8 for the large Linears (plain buffers + torch._scaled_mm). Needs torch.compile
        # to fuse the activation quantisation; eager FP8 is slower than bf16 on this card.
        fp8 = os.environ.get("LINGBOT_FP8")
        if fp8 == "mx_ffn":  # MXFP8 for the FFN only, rowwise FP8 for the projections
            n_mx = sum(convert_to_fp8(b.ffn, mx=True)[0] for b in self.model.blocks)
            n_fp8, n_all = convert_to_fp8(self.model.blocks)
            logging.info(f"FP8 MX on {n_mx} FFN Linear layers, rowwise on the other {n_fp8}")
        elif fp8 in ("1", "mx"):
            n_fp8, n_all = convert_to_fp8(self.model.blocks, mx=fp8 == "mx")
            logging.info(f"FP8 {'MX (per-32 block scales)' if fp8 == 'mx' else 'rowwise'} enabled on {n_fp8} of {n_all} Linear layers")

        # LINGBOT_SPLIT=10:2: the DiT across both GPUs, card 1's share and the decoder on separate SM partitions
        # (lingbot/parallel/split_dit.py); it compiles its own token-local stages
        split = os.environ.get("LINGBOT_SPLIT")
        if split:
            from lingbot.parallel.split_dit import SplitDiT
            assert decoder_device_id == 1, "LINGBOT_SPLIT needs --decoder_gpu 1 (the decoder shares card 1)"
            sms = int(os.environ.get("LINGBOT_SPLIT_SMS", "40"))
            self.model = SplitDiT(self.model, h0=int(split.split(":")[0]), dit_sms=sms)
            logging.info(f"DiT split {split} across cuda:0/1; card 1: {self.model.sms[0]} SMs DiT, {self.model.sms[1]} SMs decoder")
        compile_mode = None if split else os.environ.get("LINGBOT_TORCH_COMPILE")
        if compile_mode:
            if os.environ.get("LINGBOT_INDUCTOR_TUNE") == "1":
                _inductor_tune()
            # dynamic=True: current_start and the KV-cache slice bounds are Python ints that change
            # every chunk; specialising on them would recompile per chunk.
            if compile_mode == "regional":
                # one graph per transformer block, shared by all 30 (same code): a code change re-traces one
                # block instead of the whole model, which shortens cold starts after edits
                for i, block in enumerate(self.model.blocks):
                    self.model.blocks[i] = torch.compile(block, dynamic=True)
            else:
                self.model = torch.compile(self.model, dynamic=True, mode=None if compile_mode == "1" else compile_mode)
            logging.info(f"torch.compile enabled (mode={compile_mode})")

        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False)

    def set_decoder_device(self, decoder_device_id=None):
        """Decode on this GPU (None: the DiT's). Another GPU overlaps decoding with the DiT
        (2X_RTX5090_LEARNINGS.md); each chunk's 4 clean latents (0.77 MB) cross over, nothing else does.
        Each GPU gets its own fused decoder on first use (it converts its VAE copy to fp16 in place) and
        keeps it, so a benchmark sweep can switch placements inside one process."""
        dev = self.device if decoder_device_id is None else torch.device(f"cuda:{decoder_device_id}")
        if dev not in self._decoders:
            vae = self.vae if dev == self.device else Wan2_1_VAE(vae_pth=self._vae_path, dtype=torch.float, device=dev)
            self._decoders[dev] = FusedDecoder(vae, compile_mode=self._decoder_mode)
            logging.info(f"Fused decoder on {dev} (compile={self._decoder_mode})")
        self.decoder_device, self.split_decoder, self.decoder = dev, dev != self.device, self._decoders[dev]

    def _to_decoder(self, x):
        """Latents onto the decoder's GPU (a no-op on one GPU). Across GPUs PyTorch runs the copy on the
        source GPU's current stream, after the work that produced x, with barriers on both sides."""
        return x if not self.split_decoder else x.to(self.decoder_device, non_blocking=True)

    def _encode_prompts(self, prompts):
        """T5-encode ``prompts`` via the in-memory and on-disk caches; one context list per prompt."""
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
                self.text_encoder = TextEncoder(**self._t5_kwargs)
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
                # T5-XXL is 11 GB; keep it off the GPU once the prompt is cached.
                self.text_encoder.model.cpu()
                torch.cuda.empty_cache()
        return [out[p] for p in prompts]

    def _schedule(self, timesteps):
        """Everything the loop reads from the scheduler, resolved once and kept on the device, so no
        forward pays a pageable H2D copy, `nonzero` or `.item()`. The lookups reproduce the paper's
        loop exactly: its x0 conversion takes the *first* schedule entry equal to the int64 timestep
        (argmin), `add_noise` the *second* (`index_for_timestep`) when the truncated timestep repeats."""
        sched, n = self.scheduler, len(timesteps)
        x0_idx = [int(torch.argmin((sched.timesteps.double() - t).abs())) for t in timesteps]
        noise_idx = [sched.index_for_timestep(t) for t in timesteps[1:]]
        t_dev = timesteps.to(self.device)
        return dict(
            t=[t_dev[i:i + 1].clone() for i in range(n)],
            # int64 like the denoising steps: a float t would be a second dtype variant of the graph
            t_zero=t_dev[-1:] * 0,
            sigma_x0=sched.sigmas.double()[x0_idx].reshape(n, 1, 1, 1, 1).to(self.device),
            sigma_noise=sched.sigmas[noise_idx].reshape(n - 1, 1, 1, 1, 1).to(self.device),
        )

    def generate(self, input_prompt, img, action_path, chunk_size=3, max_area=480 * 832, frame_num=81,
                 timesteps_index=(0, 250, 500, 750), shift=5.0, seed=-1, offload_model=True,
                 max_sequence_length=512, max_attention_size=None):
        """Returns the video as [C, F, H, W] in [-1, 1] on rank 0 (None elsewhere)."""
        batch_size = 1
        # LINGBOT_BENCH_E2E=1 (lingbot.benchmark): CUDA events at the start and when each chunk's frames
        # are decoded, read once at the end; nothing synchronizes inside the loop.
        e2e = os.environ.get("LINGBOT_BENCH_E2E") == "1"
        if e2e:
            e2e_start = torch.cuda.Event(enable_timing=True)
            e2e_start.record(torch.cuda.current_stream(self.decoder_device))  # same GPU as the ready events
            e2e_ready = []
        assert action_path is not None, "action_path is required"
        c2ws = np.load(os.path.join(action_path, "poses.npy"))  # opencv coordinate
        frame_num = ((frame_num - 1) // 4) * 4 + 1
        if self.pose_provider is not None:
            c2ws = np.tile(np.eye(4, dtype=c2ws.dtype), (frame_num, 1, 1))  # placeholder; replaced per chunk
        len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
        frame_num = min(frame_num, len_c2ws)
        c2ws = c2ws[:frame_num]

        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1])
        lat_w = round(np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        lat_f = (F - 1) // self.vae_stride[0] + 1
        lat_f = int(lat_f - (lat_f % chunk_size))
        F = (lat_f - 1) * 4 + 1
        max_seq_len = int(math.ceil(chunk_size * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])))

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(16, lat_f, lat_h, lat_w, dtype=torch.float32, generator=seed_g, device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        self.scheduler.set_timesteps(self.num_train_timesteps, shift=shift)
        timesteps = self.scheduler.timesteps[list(timesteps_index)]
        sched = self._schedule(timesteps)
        # "1": sync before the pose sample in play mode; "force": also without a pose provider
        late_sample = os.environ.get("LINGBOT_LATE_SAMPLE", "1") if self.device.type == "cuda" else ""
        late_sample = late_sample if late_sample in ("1", "force") else ""

        context = self._encode_prompts([input_prompt])[0]

        # The provided intrinsics are for the original 480p image; transform them to (h, w).
        Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()
        Ks = get_Ks_transformed(Ks, height_org=480, width_org=832, height_resize=h, width_resize=w,
                                height_final=h, width_final=w)[0]

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
        c2ws_plucker_emb = rearrange(c2ws_plucker_emb, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                                     c1=int(h // lat_h), c2=int(w // lat_w))
        c2ws_plucker_emb = c2ws_plucker_emb[None, ...]  # [b, f*h*w, c]
        c2ws_plucker_emb = rearrange(c2ws_plucker_emb, 'b (f h w) c -> b c f h w',
                                     f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)

        y_key = (hash(img.cpu().numpy().tobytes()), F, h, w)
        if self._y_cache is not None and self._y_cache[0] == y_key:
            y = self._y_cache[1]
        else:
            y = self.vae.encode([
                torch.concat([
                    torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
                    torch.zeros(3, F - 1, h, w)
                ], dim=1).to(self.device)
            ])[0]
            self._y_cache = (y_key, y)
        y = torch.concat([msk, y])

        model_args = self.model.config
        frame_seqlen = int(noise.shape[-2] * noise.shape[-1] // 4)
        kv_size = frame_seqlen * (self.local_attn_size if self.local_attn_size > -1 else lat_f)
        head_dim = model_args.dim // model_args.num_heads
        self_kv_cache = kvc.allocate(model_args.num_layers, [batch_size, kv_size, model_args.num_heads, head_dim],
                                     self.pipe_dtype, self.device)
        cross_kv_cache = [{
            'k': torch.zeros([batch_size, max_sequence_length, model_args.num_heads, head_dim],
                             dtype=self.pipe_dtype, device=self.device),
            'v': torch.zeros([batch_size, max_sequence_length, model_args.num_heads, head_dim],
                             dtype=self.pipe_dtype, device=self.device),
            'is_init': torch.tensor(0, dtype=torch.int32, device=self.device),
        } for _ in range(model_args.num_layers)]
        with torch.amp.autocast('cuda', dtype=self.param_dtype), torch.no_grad():
            # Fill the cross-attn K/V once (same autocast as the loop) so no forward takes the
            # first-call branch, and allocate the per-chunk camera-modulation cache.
            self.model.init_crossattn_cache([context[0]] * batch_size, cross_kv_cache)
            cam_cache = [{
                'scale': torch.zeros([batch_size, max_seq_len, model_args.dim], dtype=self.pipe_dtype, device=self.device),
                'shift': torch.zeros([batch_size, max_seq_len, model_args.dim], dtype=self.pipe_dtype, device=self.device),
            } for _ in range(model_args.num_layers)]
            latents_chunk = noise.split(chunk_size, dim=1)  # [c, f, h, w]
            condition_chunk = y.split(chunk_size, dim=1)
            c2ws_plucker_emb_chunk = c2ws_plucker_emb.split(chunk_size, dim=2)
            num_inference_chunk = len(latents_chunk)
            pred_latent_chunks = []
            # LINGBOT_VAE_STREAM=1: decode chunk N on a side stream while chunk N+1 denoises; the
            # final whole-clip decode is skipped.
            vae_stream_on = os.environ.get("LINGBOT_VAE_STREAM") == "1"
            if vae_stream_on:
                vae_stream = getattr(self.model, "decoder_stream", None) or torch.cuda.Stream(device=self.decoder_device)
                dec_state, dec_pending, dec_frames = None, None, []
            # LINGBOT_DECODE_FIRST=1: decode right after x0 on the side stream, overlapped with this
            # chunk's cache-write forward only, then the main stream waits before the next chunk.
            decode_first = os.environ.get("LINGBOT_DECODE_FIRST") == "1"
            if self.frame_sink is not None:
                assert vae_stream_on, "frame_sink needs LINGBOT_VAE_STREAM=1"
                chunk_t0 = []
            bench_timing = os.environ.get("LINGBOT_BENCH_TIMING") == "1"
            assert not e2e or (vae_stream_on and decode_first and not bench_timing), \
                "LINGBOT_BENCH_E2E needs per-chunk decoding (LINGBOT_VAE_STREAM=1, LINGBOT_DECODE_FIRST=1) and no per-chunk syncs"
            if bench_timing:
                torch.cuda.synchronize(self.device)
                t_loop0 = t_prev = time.perf_counter()
            profiler = ChunkProfiler()
            for chunk_id in tqdm(range(num_inference_chunk)):
                profiler.before_chunk(chunk_id)
                _rf_chunk = torch.profiler.record_function(f"chunk{chunk_id}"); _rf_chunk.__enter__()
                if (self.pose_provider is not None or late_sample == "force") and late_sample and chunk_id > 0:
                    # Sample the input when the GPU can actually start this chunk: the host runs ~0.35 s
                    # ahead and the pageable .to(device) below blocks on the queue anyway.
                    torch.cuda.current_stream(self.device).synchronize()
                if self.chunk_gate is not None:
                    self.chunk_gate(chunk_id)
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
                    current_c2ws_plucker_emb = rearrange(emb[None], 'b (f h w) c -> b c f h w',
                                                         f=chunk_size, h=lat_h, w=lat_w).to(self.param_dtype)
                kwargs = {
                    'context': [context[0]] * batch_size,
                    'seq_len': max_seq_len,
                    'y': [current_condition] * batch_size,
                    'dit_cond_dict': {"c2ws_plucker_emb": current_c2ws_plucker_emb.chunk(1, dim=0)},
                    'kv_cache': self_kv_cache,
                    'crossattn_cache': cross_kv_cache,
                    'current_start': chunk_id * chunk_size * frame_seqlen,
                    'max_attention_size': kv_size if max_attention_size is None else max_attention_size,
                    'frame_seqlen': frame_seqlen,
                    'cam_cache': cam_cache,
                }
                if offload_model:
                    torch.cuda.empty_cache()
                _rf.__exit__(None, None, None)
                if vae_stream_on and dec_pending is not None:
                    with torch.profiler.record_function("vae_stream_launch"), torch.no_grad():
                        vae_stream.wait_stream(torch.cuda.current_stream(self.device))
                        with torch.cuda.stream(vae_stream):
                            if not self.split_decoder:
                                dec_pending.record_stream(vae_stream)
                            fr, dec_state = self.decoder.decode_step(self._to_decoder(dec_pending), dec_state)
                            if self.frame_sink is not None:
                                self.frame_sink(chunk_id - 1, fr, vae_stream, chunk_t0[chunk_id - 1])
                            else:
                                dec_frames.append(fr)
                    dec_pending = None
                for timestep_idx in range(len(timesteps)):
                    _rf = torch.profiler.record_function(f"denoise_step{timestep_idx}"); _rf.__enter__()
                    # cam MLP computed on the chunk's first forward, reused after
                    kwargs['cam_first_call'] = timestep_idx == 0
                    noise_pred = self.model(x=[current_latent.to(self.device)] * batch_size, t=sched['t'][timestep_idx],
                                            cross_attn_first_call=False, **kwargs)[0]
                    if offload_model:
                        torch.cuda.empty_cache()
                    # same ops and dtypes as the paper's flow-to-x0 conversion and add_noise
                    x0 = (current_latent.double() - sched['sigma_x0'][timestep_idx] * noise_pred.double()).to(noise_pred.dtype)
                    if timestep_idx < len(timesteps) - 1:
                        next_noise = torch.randn(x0.shape, generator=seed_g, device=x0.device, dtype=x0.dtype)
                        assert x0.dtype == sched['sigma_noise'].dtype
                        sigma_t = sched['sigma_noise'][timestep_idx]
                        current_latent = (1 - sigma_t) * x0 + sigma_t * next_noise
                    _rf.__exit__(None, None, None)

                pred_latent_chunks.append(x0)
                if vae_stream_on and decode_first:
                    # Decode this chunk now, latent by latent, handing each latent's frames to the sink
                    # as they finish; the main stream then waits, so the frames leave before the next
                    # forward instead of competing with it (same throughput, first frame ~0.5 s earlier).
                    with torch.profiler.record_function("vae_decode_first"), torch.no_grad():
                        vae_stream.wait_stream(torch.cuda.current_stream(self.device))
                        with torch.cuda.stream(vae_stream):
                            if not self.split_decoder:
                                x0.record_stream(vae_stream)
                            z = self._to_decoder(x0)
                            off = 0
                            for li in range(z.shape[1]):
                                fr, dec_state = self.decoder.decode_step(z[:, li:li + 1], dec_state)
                                if self.frame_sink is not None:
                                    self.frame_sink(chunk_id, fr, vae_stream, chunk_t0[chunk_id], off)
                                else:
                                    dec_frames.append(fr)
                                off += fr.shape[1]
                            if e2e:
                                e2e_ready.append(torch.cuda.Event(enable_timing=True))
                                e2e_ready[-1].record(vae_stream)
                elif vae_stream_on:
                    dec_pending = x0
                _rf = torch.profiler.record_function("cache_write"); _rf.__enter__()
                kwargs['cam_first_call'] = False
                self.model(x=[x0] * batch_size, t=sched['t_zero'], cross_attn_first_call=False, **kwargs)
                _rf.__exit__(None, None, None)
                if vae_stream_on and decode_first and not self.split_decoder:
                    # one GPU: finish the decode before the next chunk instead of competing with it;
                    # with the decoder on its own GPU there is nothing to compete with
                    torch.cuda.current_stream(self.device).wait_stream(vae_stream)
                if bench_timing:
                    with torch.profiler.record_function("bench_sync"):
                        torch.cuda.synchronize(self.device)
                    now = time.perf_counter()
                    logging.info(f"BENCH chunk={chunk_id} chunk_s={now - t_prev:.3f} loop_s={now - t_loop0:.3f}")
                    self.bench_chunk_s = getattr(self, "bench_chunk_s", []) + [now - t_prev]
                    t_prev = now
                _rf_chunk.__exit__(None, None, None)
                profiler.after_chunk(chunk_id)

            pred_latent_chunks = torch.cat(pred_latent_chunks, dim=1)
            kwargs['cam_cache'] = cam_cache = None  # ~1.1 GB; free before the decode

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            videos = [None]
            if self.rank == 0:
                # One-time compile/autotune of the decoder, outside the timed decode. Not under
                # vae_stream_on: decode_step is warm from chunk 0 there.
                if os.environ.get("LINGBOT_VAE_WARM") == "1" and not vae_stream_on and not getattr(self.decoder, "warmed", False):
                    self.decoder.warmed = True
                    with torch.no_grad():
                        self.decoder.decode(self._to_decoder(pred_latent_chunks[:, :5]))
                if bench_timing:
                    torch.cuda.synchronize(self.device)
                    torch.cuda.synchronize(self.decoder_device)
                    t_dec0 = time.perf_counter()
                with DecodeProfiler(enabled=not vae_stream_on) as dprof:
                    if vae_stream_on:
                        with torch.no_grad():
                            vae_stream.wait_stream(torch.cuda.current_stream(self.device))
                            if dec_pending is not None:
                                with torch.cuda.stream(vae_stream):
                                    fr, dec_state = self.decoder.decode_step(dec_pending, dec_state)
                                    if self.frame_sink is not None:
                                        self.frame_sink(num_inference_chunk - 1, fr, vae_stream, chunk_t0[-1])
                                    else:
                                        dec_frames.append(fr)
                            torch.cuda.current_stream(self.device).wait_stream(vae_stream)
                            videos = [torch.cat(dec_frames, 1)] if dec_frames else [None]
                    else:
                        videos = [self.decoder.decode(self._to_decoder(pred_latent_chunks))]
                    dprof.finish(num_inference_chunk)
                if bench_timing:
                    torch.cuda.synchronize(self.decoder_device)
                    logging.info(f"BENCH vae_decode_s={time.perf_counter() - t_dec0:.3f}")
                    self.bench_vae_decode_s = time.perf_counter() - t_dec0

        if e2e:
            torch.cuda.synchronize(self.decoder_device)
            self.bench_ready_ms = [e2e_start.elapsed_time(e) for e in e2e_ready]
        if offload_model:
            gc.collect()
            torch.cuda.synchronize(self.device)
        if dist.is_initialized():
            dist.barrier()
        return videos[0] if self.rank == 0 else None
