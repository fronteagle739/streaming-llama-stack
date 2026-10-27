"""Bounded-memory prefill for MiMo's quantized global KV cache."""
import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention as _attention


def scaled_dot_product_attention(queries, keys, values, *, cache, scale, mask, sinks=None):
    if isinstance(keys, tuple) and queries.shape[-2] > 1:
        # Quantized matmul prefill materializes [heads, chunk, context] scores.
        # Dequantize only this layer's KV and use Metal's fused SDPA instead.
        # Persistent cache stays Q8; the temporary KV dies after this layer.
        keys = mx.dequantize(*keys, group_size=cache.group_size, bits=cache.bits)
        values = mx.dequantize(*values, group_size=cache.group_size, bits=cache.bits)
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask, sinks=sinks)
    if isinstance(keys, tuple):
        # Upstream's quantized GQA reshape assumes key_dim == value_dim.
        # MiMo uses 192/128, so preserve the actual value dimension here.
        batch, heads, length, width = queries.shape
        kv_heads = keys[0].shape[-3]
        repeats = heads // kv_heads
        q = (queries * scale).reshape(batch, kv_heads, repeats, length, width)
        k = tuple(mx.expand_dims(a, -3) for a in keys)
        v = tuple(mx.expand_dims(a, -3) for a in values)
        scores = mx.quantized_matmul(q, *k, transpose=True,
            group_size=cache.group_size, bits=cache.bits)
        if mask is not None and not isinstance(mask, str):
            if mask.ndim > 3:
                mask = mx.expand_dims(mask, -3)
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min) if mask.dtype == mx.bool_ else scores + mask
        if sinks is not None:
            raise ValueError("quantized MiMo global attention does not support sinks")
        scores = mx.softmax(scores, axis=-1, precise=True)
        out = mx.quantized_matmul(scores, *v, transpose=False,
            group_size=cache.group_size, bits=cache.bits)
        return out.reshape(batch, heads, length, out.shape[-1])
    return _attention(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks)
