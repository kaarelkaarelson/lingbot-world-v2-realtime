# experiments/

Code from experiments that did not ship, kept for reference. Nothing here is imported by `lingbot/`;
it targets the code as it was before the restructure, so run it from the `pre-cleanup` tag
(`git checkout pre-cleanup`). Every experiment, its measurement and the reason it was rejected are
in [docs/OPTIMIZATIONS.md](../docs/OPTIMIZATIONS.md).

| Path | What it is | docs/OPTIMIZATIONS.md |
|---|---|---|
| `bench_attn/` | SageAttention studies: hypotheses H1-H7, tile configs, SpargeAttention, FP4 (SageAttention 3), INT8 calibration | §19, §20, §22, §23 |
| `bench_window/` | KV-window sweeps, attention ablation, block sparsity | §21 |
| `bench_engines/` | the other engines we compared against (SGLang, LightX2V, FlashDreams) | README "Performance vs other engines" |
| `modules/sage_kvq.py` | KV cache kept pre-quantized for SageAttention (`LINGBOT_ATTN=sage_kvq`) | §20 |
| `modules/model_fast_kv_ring.py` | the paper's `model_fast.py` with the host-sync fix and the KV ring buffer (`LINGBOT_KV_RING`) | §1, §11 |
| `modules/attention_sage3_qkv_dumps.py` | attention with SageAttention 3 FP4 (`LINGBOT_ATTN=sage3`) and the Q/K/V dump hooks | §23 |
| `modules/sequence_parallel_int_mirrors.py` | the paper's sequence-parallel forward with Python-int cache indices | §1 |

Removed from the generation loop at the same time, all reachable at `pre-cleanup`: the tiny decoders
(TAEHV, LightTAE, Flash-VAED; §3, §6, §8), decoder quality modes v1/v2 (§9), `LINGBOT_TIMESTEPS`,
`LINGBOT_REF_FRAMES`, `LINGBOT_BATCH` (§16), `LINGBOT_DUMP_LATENTS`, `LINGBOT_XATTN`,
`LINGBOT_DIT_FUSION_EXACT_ROPE`, regional compile, `prewarm()`, and the `causal_pretrain` mode
(the paper's 40-step model; still runnable from `reference/`).
