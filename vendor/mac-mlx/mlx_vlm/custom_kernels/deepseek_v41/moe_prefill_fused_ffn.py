"""Attach-latched fused prefill adapter; routing weights remain after down."""
import math

import mlx.core as mx

from .expert_tiles_qmm import expert_tiles_metadata, expert_tiles_qmm
from .fused_ffn_qmm import expert_gate_up_swiglu, fused_ffn_metadata, prewarm_fused_ffn
from .moe_prefill_expert_tiles import validate_mlp

_ENGAGED = False


def prepare_fused_ffn(layers):
    """Validate every backbone MLP and compile before publishing the latch."""
    from ...models.deepseek_v41.language import ClampedSwiGLU

    try:
        for layer in layers:
            mlp = layer.ffn.switch_mlp
            validate_mlp(mlp)
            if type(mlp.activation) is not ClampedSwiGLU or not math.isfinite(mlp.activation.limit):
                raise ValueError("fused FFN requires ClampedSwiGLU with a finite limit")
        prewarm_fused_ffn()
    except Exception as exc:
        raise RuntimeError("DSV41_FFN_FUSED=1 attach failed; refusing to enable the custom kernel") from exc


def moe_prefill_fused_ffn(mlp, x, indices):
    """SwitchGLU-compatible [tokens,topk,dim] FP32 output, no route weighting."""
    from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

    if x.dtype != mx.bfloat16 or x.shape[0] < 64:
        raise ValueError("fused FFN requires BF16 prefill input (n >= 64)")
    if indices.ndim != 2 or indices.shape[0] != x.shape[0]:
        raise ValueError("fused FFN route indices must be [tokens, topk]")
    global _ENGAGED
    if not _ENGAGED:
        print(f"[DSV41] FFN_FUSED engaged n={x.shape[0]}", flush=True)
        _ENGAGED = True
    x, ids, inverse = _gather_sort(mx.expand_dims(x, (-2, -3)), indices)
    if mlp.training:
        ids = mx.stop_gradient(ids)
    ids = ids.astype(mx.uint32)
    fused_plan = fused_ffn_metadata(ids, mlp.up_proj.weight.shape[0])
    # Two small GPU metadata launches keep each projection's proven tile mapping.
    down_plan = expert_tiles_metadata(ids, mlp.down_proj.weight.shape[0])
    gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
    hidden = expert_gate_up_swiglu(x, gate.weight, gate.scales, up.weight, up.scales,
                                   fused_plan, mlp.activation.limit)
    y = expert_tiles_qmm(hidden, down.weight, down.scales, down_plan)
    return _scatter_unsort(y, inverse, indices.shape).squeeze(-2)
