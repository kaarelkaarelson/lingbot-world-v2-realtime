"""Self-attention KV cache for the causal DiT: a sink of the first frames plus a sliding window.

Each layer's cache is a dict of K/V buffers and end indices. Positions are tracked as Python ints
("*_int" keys) so the eviction schedule is pure host arithmetic; the historical `.item()` reads
forced a GPU->CPU sync per layer per forward. The tensor indices are kept in sync for any external
reader.
"""
import torch


def allocate(num_layers, shape, dtype, device):
    """One cache dict per layer; `shape` is [batch, window tokens, heads, head_dim]."""
    return [{
        'k': torch.zeros(shape, dtype=dtype, device=device),
        'v': torch.zeros(shape, dtype=dtype, device=device),
        'global_end_index': torch.tensor([0], dtype=torch.long, device=device),
        'local_end_index': torch.tensor([0], dtype=torch.long, device=device),
        'global_end_int': 0,
        'local_end_int': 0,
    } for _ in range(num_layers)]


def plan(kv_cache, num_new_tokens, current_start, sink_tokens, local_attn_size):
    """Host-side bookkeeping of `write`: (evicted, rolled, local_start_index, local_end_index, current_end).

    `evicted` > 0 means the window is full: `rolled` non-sink tokens must shift left by `evicted` first.
    """
    current_end = current_start + num_new_tokens
    kv_cache_size = kv_cache["k"].shape[1]
    global_end = kv_cache.get("global_end_int")
    if global_end is None:
        global_end = kv_cache["global_end_index"].item()
        local_end = kv_cache["local_end_index"].item()
    else:
        local_end = kv_cache["local_end_int"]
    evicted = rolled = 0
    if local_attn_size == -1:
        # No eviction possible (global cache): both indices advance with the stream.
        local_end_index = current_end
        local_start_index = current_start
    elif (current_end > global_end) and (num_new_tokens + local_end > kv_cache_size):
        evicted = num_new_tokens + local_end - kv_cache_size
        rolled = local_end - evicted - sink_tokens
        local_end_index = local_end + current_end - evicted - global_end
        local_start_index = local_end_index - num_new_tokens
    else:
        local_end_index = local_end + current_end - global_end
        local_start_index = local_end_index - num_new_tokens
    return evicted, rolled, local_start_index, local_end_index, current_end


def evict(kv_cache, sink_tokens, evicted, rolled):
    """Shift the window left past the sink to drop the oldest tokens. Clone the source slice
    to avoid overlapping-memory errors."""
    kv_cache["k"][:, sink_tokens:sink_tokens + rolled] = \
        kv_cache["k"][:, sink_tokens + evicted:sink_tokens + evicted + rolled].clone()
    kv_cache["v"][:, sink_tokens:sink_tokens + rolled] = \
        kv_cache["v"][:, sink_tokens + evicted:sink_tokens + evicted + rolled].clone()


def write(kv_cache, k, v, current_start, sink_tokens, local_attn_size):
    """Store this forward's K/V, evicting the oldest non-sink tokens when the window is full.

    Returns (local_end_index, current_end): where the chunk ends in the buffer, and in the stream.
    """
    evicted, rolled, local_start_index, local_end_index, current_end = plan(
        kv_cache, k.shape[1], current_start, sink_tokens, local_attn_size)
    if evicted:
        evict(kv_cache, sink_tokens, evicted, rolled)
    kv_cache["k"][:, local_start_index:local_end_index] = k
    kv_cache["v"][:, local_start_index:local_end_index] = v
    return local_end_index, current_end


def window(kv_cache, local_end_index, max_attention_size):
    """The K/V slice this forward attends to."""
    start = max(0, local_end_index - max_attention_size)
    return kv_cache["k"][:, start:local_end_index], kv_cache["v"][:, start:local_end_index]


def commit(kv_cache, current_end, local_end_index):
    kv_cache["global_end_int"] = current_end
    kv_cache["local_end_int"] = local_end_index
    kv_cache["global_end_index"].fill_(current_end)
    kv_cache["local_end_index"].fill_(local_end_index)
