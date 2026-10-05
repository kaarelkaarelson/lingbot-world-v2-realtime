"""The generation loop is wired as before the restructure: same DiT call sequence (timestep,
current_start, cam cache flag, KV-cache positions through eviction) and bit-identical latents into the
decoder. CPU only: tiny random DiT, mock VAE/T5 (tests/pipeline_wiring_harness.py).

tests/golden/pipeline_wiring.pt was recorded from the pre-restructure pipeline:
    git archive pre-cleanup wan lingbot | tar -x -C /tmp/old
    python tests/pipeline_wiring_harness.py --old /tmp/old tests/golden/pipeline_wiring.pt
"""
import os
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def test_pipeline_wiring_cpu():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "new.pt")
        # own process: the harness puts the repo on sys.path and imports the pipeline from scratch
        subprocess.run([sys.executable, os.path.join(HERE, "pipeline_wiring_harness.py"), "--new", out],
                       check=True, cwd=d, capture_output=True)
        got = torch.load(out)
    ref = torch.load(os.path.join(HERE, "golden", "pipeline_wiring.pt"))
    assert got["calls"] == ref["calls"], "DiT call sequence changed"
    assert got["vae_encode_in"] == ref["vae_encode_in"] and got["video_shape"] == ref["video_shape"]
    assert torch.equal(got["decoded_latents"], ref["decoded_latents"]), "latents into the decoder changed"
