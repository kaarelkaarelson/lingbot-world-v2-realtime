# reference/: the paper's code, unmodified

An exact copy of [robbyant/lingbot-world](https://github.com/robbyant/lingbot-world) at commit
`1895d30` ("Fix README for causal_fast 1.3B command details"): `wan/`, `generate.py`, `run_fast.sh`,
`requirements.txt` and `README.md`. Only `__init__.py` (so `reference.wan` is importable) and this
file were added.

It serves two purposes:

- **Baseline.** `--preset stock` runs `reference/generate.py`, so "original paper's code" numbers
  are measured on the paper's code itself. Its own multi-GPU path is
  `torchrun --nproc_per_node=N reference/generate.py ... --dit_fsdp --t5_fsdp --ulysses_size N`.
- **Shared primitives.** The optimized code in `lingbot/` imports unchanged building blocks from
  here (configs, VAE and T5 modules, RMSNorm, RoPE helpers, camera utilities, scheduler).

Do not edit files in this directory. Verify it is still byte-identical (needs the `upstream` remote):

    cd reference && for f in $(find wan generate.py run_fast.sh requirements.txt README.md -type f); do
      git show upstream/main:$f | cmp -s - $f || echo "DIFFERS: $f"; done
