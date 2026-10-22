# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/_cosmos_compat.py
# Machine fallback paths cleared; no model data included.
import torch

def _compute_default_rope_parameters(config=None, device=None, seq_len=None, **rope_kwargs):
    base = config.rope_theta
    partial_rotary_factor = getattr(config, 'partial_rotary_factor', 1.0)
    head_dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0
    inv_freq = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
    return (inv_freq, attention_factor)

def apply():
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        ROPE_INIT_FUNCTIONS.setdefault('default', _compute_default_rope_parameters)
    except Exception as e:
        print(f'[cosmos_compat] rope shim skipped: {e}')
apply()
