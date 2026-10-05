"""The paper's UMT5-XXL text encoder with a faster loader.

Same model and weights as `reference/wan/modules/t5.py`; only loading differs: the module is built
on the meta device (random-initialising 5.7B parameters on CPU takes minutes and every value is
overwritten anyway) and the checkpoint is memory-mapped.

Importing `wan.modules.t5` evaluates `torch.cuda.current_device()` as a default argument,
which opens a CUDA context on the current device; multi-GPU launchers must call
`torch.cuda.set_device(local_rank)` before importing this module.
"""
import logging

import torch

from wan.modules.t5 import HuggingfaceTokenizer, T5EncoderModel, umt5_xxl


class TextEncoder(T5EncoderModel):

    def __init__(self, text_len, dtype=torch.bfloat16, device=None, checkpoint_path=None,
                 tokenizer_path=None, shard_fn=None):
        self.text_len = text_len
        self.dtype = dtype
        self.device = torch.cuda.current_device() if device is None else device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        model = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=dtype,
                         device='meta').eval().requires_grad_(False)
        logging.info(f'loading {checkpoint_path}')
        try:
            state = torch.load(checkpoint_path, map_location='cpu', mmap=True)
        except RuntimeError:  # legacy (non-zip) .pth cannot be mmapped
            state = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state, assign=True)
        self.model = model
        if shard_fn is not None:
            self.model = shard_fn(self.model, sync_module_states=False)
        else:
            self.model.to(self.device)
        self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=text_len, clean='whitespace')
