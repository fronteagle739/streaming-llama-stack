"""Opt-in, attach-latched expert-aligned routed FFN. No GPU-to-host reads."""
import mlx.core as mx

from .expert_tiles_qmm import (
    expert_tiles_metadata, expert_tiles_qmm, prewarm_expert_tiles, validate_weight,
)

_ENGAGED = False


def validate_mlp(mlp):
    projections = (mlp.gate_proj, mlp.up_proj, mlp.down_proj)
    shapes = []
    for layer in projections:
        if (getattr(layer, "mode", None) != "mxfp4"
                or getattr(layer, "group_size", None) != 32
                or getattr(layer, "bits", None) != 4
                or layer.get("bias") is not None or layer.get("biases") is not None):
            raise ValueError("expert tiles requires bias-free MXFP4/group32/bits4 projections")
        shapes.append(validate_weight(layer.weight, layer.scales))
    if shapes[0] != shapes[1] or shapes[2] != shapes[0][::-1]:
        raise ValueError("expert tiles gate/up/down projection dimensions do not match")


def prepare_expert_tiles(layers):
    """Validate all backbone projections, then compile both dtypes at attach."""
    try:
        for layer in layers:
            validate_mlp(layer.ffn.switch_mlp)
        prewarm_expert_tiles()
    except Exception as exc:
        raise RuntimeError(
            "DSV41_FFN_EXPERT_TILES=1 attach failed; refusing to enable the custom kernel"
        ) from exc


def moe_prefill_expert_tiles(mlp, x, indices):
    """SwitchGLU-compatible [tokens,topk,dim] output; caller combines FP32."""
    from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

    if x.dtype != mx.bfloat16:
        raise ValueError("expert tiles routed gate/up input must be BF16")
    if x.shape[0] < 64:
        raise ValueError("expert tiles is a prefill-only path (n >= 64)")
    if indices.ndim != 2 or indices.shape[0] != x.shape[0]:
        raise ValueError("expert tiles route indices must be [tokens, topk]")
    global _ENGAGED
    if not _ENGAGED:
        print(f"[DSV41] FFN_EXPERT_TILES engaged n={x.shape[0]}", flush=True)
        _ENGAGED = True
    x, ids, inverse = _gather_sort(mx.expand_dims(x, (-2, -3)), indices)
    if mlp.training:
        ids = mx.stop_gradient(ids)
    routing = expert_tiles_metadata(ids.astype(mx.uint32), mlp.up_proj.weight.shape[0])

    def project(value, layer):
        return expert_tiles_qmm(value, layer.weight, layer.scales, routing)

    up = project(x, mlp.up_proj)
    gate = project(x, mlp.gate_proj)
    hidden = mlp.activation(up, gate)  # Existing ClampedSwiGLU and rounding boundaries.
    if hidden.dtype != mx.float32:
        raise ValueError("expert tiles requires the unchanged FP32 ClampedSwiGLU output")
    down = project(hidden, mlp.down_proj)
    return _scatter_unsort(down, inverse, indices.shape).squeeze(-2)
