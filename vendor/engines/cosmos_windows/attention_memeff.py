# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/attention_memeff.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
_orig_sdpa = None

def enable_memeff_attention(allow_math_fallback: bool=False) -> bool:
    global _orig_sdpa
    import torch.nn.functional as F
    if _orig_sdpa is not None:
        return True
    from torch.nn.attention import sdpa_kernel, SDPBackend
    import inspect
    try:
        _has_priority = 'set_priority' in inspect.signature(sdpa_kernel).parameters
    except (ValueError, TypeError):
        _has_priority = False
    _orig_sdpa = F.scaled_dot_product_attention
    backends = []
    _cudnn = getattr(SDPBackend, 'CUDNN_ATTENTION', None)
    if _cudnn is not None:
        backends.append(_cudnn)
    backends += [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]
    if allow_math_fallback:
        backends.append(SDPBackend.MATH)

    def patched(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        if enable_gqa and key.shape[1] != query.shape[1] and (query.shape[1] % key.shape[1] == 0):
            rep = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(rep, dim=1)
            value = value.repeat_interleave(rep, dim=1)
            enable_gqa = False
        cm = sdpa_kernel(backends, set_priority=True) if _has_priority else sdpa_kernel(backends)
        with cm:
            return _orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    F.scaled_dot_product_attention = patched
    return True

def disable_memeff_attention() -> None:
    global _orig_sdpa
    if _orig_sdpa is not None:
        import torch.nn.functional as F
        F.scaled_dot_product_attention = _orig_sdpa
        _orig_sdpa = None
