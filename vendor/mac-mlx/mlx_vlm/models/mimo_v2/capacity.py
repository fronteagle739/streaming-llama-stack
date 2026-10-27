"""Real KV allocation/decode capacity check, distinct from prompt-quality tests."""
import json
import mlx.core as mx
from mlx_lm.models.cache import QuantizedKVCache


def probe_capacity(model, tokens, forward, *, workspace_bytes=16 * 2**30, limit_bytes=0):
    caches = model.make_cache()
    required = 0
    for layer, cache in zip(model.pipeline_layers, caches):
        a = layer.self_attn
        n = min(tokens, cache.max_size) if layer.is_sliding_window else tokens
        per_value = 2 if layer.is_sliding_window else (cache.bits / 8 + 4 / cache.group_size)
        required += int(n * a.n_kv_heads * (a.head_dim + a.v_head_dim) * per_value)
    base = mx.get_active_memory()
    ceiling = limit_bytes or mx.device_info()['max_recommended_working_set_size']
    if base + required + workspace_bytes + 2 * 2**30 > ceiling:
        raise MemoryError(f'MiMo {tokens}-token KV plus runtime reserve exceeds rank memory ceiling')
    # MLX shape dimensions are signed 32-bit; reserve large workspaces as a
    # collection of real allocations rather than an overflowing 16-GiB axis.
    workspace = [mx.zeros((min(2**30, workspace_bytes - offset),), dtype=mx.uint8)
                 for offset in range(0, workspace_bytes, 2**30)]
    try:
        mx.eval(workspace)
        for layer, cache in zip(model.pipeline_layers, caches):
            a = layer.self_attn
            if isinstance(cache, QuantizedKVCache):
                def packed(dim):
                    shape = (1, a.n_kv_heads, tokens)
                    return (mx.zeros((*shape, dim * cache.bits // 32), dtype=mx.uint32),
                            mx.ones((*shape, dim // cache.group_size), dtype=mx.bfloat16),
                            mx.zeros((*shape, dim // cache.group_size), dtype=mx.bfloat16))
                cache.keys, cache.values = packed(a.head_dim), packed(a.v_head_dim)
                cache.offset = tokens - 1
                mx.eval(cache.keys, cache.values)
            else:
                retained = min(tokens - 1, cache.max_size)
                k = mx.zeros((1, a.n_kv_heads, retained, a.head_dim), dtype=mx.bfloat16)
                v = mx.zeros((1, a.n_kv_heads, retained, a.v_head_dim), dtype=mx.bfloat16)
                mx.eval(cache.update_and_fetch(k, v))
                cache.offset = tokens - 1
            mx.clear_cache()
        output = forward(caches)
        mx.eval(output)
        if not bool(mx.all(mx.isfinite(output)).item()):
            raise ValueError('MiMo synthetic long-context decode produced nonfinite output')
        receipt = dict(scope='synthetic_KV_allocation_and_decode_not_prompt_quality',
            rank=model.pipeline_rank, tokens=tokens, kv_bytes=required,
            workspace_bytes=workspace_bytes, active_gib=mx.get_active_memory()/2**30,
            peak_gib=mx.get_peak_memory()/2**30, finite=True)
        print('MIMO_CONTEXT_CAPACITY ' + json.dumps(receipt), flush=True)
        return receipt
    finally:
        del caches, workspace
        mx.clear_cache()
