"""Opt-in grouped routed experts; preserves SwitchGLU's rounding boundaries."""
import os

import mlx.core as mx

from ...custom_kernels.deepseek_v41.grouped_gemm import (
    _grouped_mxfp4,
    expert_offsets,
    prepare_worklist,
)


def moe_prefill_grouped(moe, xf, indices, weights, *, gemm_config=None):
    """Return routed FP32 sum; gemm_config is an explicit benchmark override."""
    mlp = moe.switch_mlp
    projections = (mlp.gate_proj, mlp.up_proj, mlp.down_proj)
    for layer in projections:
        if (getattr(layer, "mode", None) != "mxfp4"
                or getattr(layer, "group_size", None) != 32
                or getattr(layer, "bits", None) != 4
                or layer.get("bias") is not None or layer.get("biases") is not None):
            raise ValueError("grouped prefill requires bias-free MXFP4/group32 experts")
    if xf.dtype != mx.bfloat16:
        raise ValueError("grouped prefill requires BF16 routed input")
    if indices.ndim != 2 or indices.shape != weights.shape or indices.shape[0] != xf.shape[0]:
        raise ValueError("route indices/weights must be [tokens, topk]")
    # Same flatten/argsort/order//topk/inverse permutation as SwitchGLU.
    flat_ids = indices.flatten()
    order = mx.argsort(flat_ids)
    inverse = mx.argsort(order)
    x_sorted = xf[order // indices.shape[-1]]
    offsets = expert_offsets(flat_ids[order], projections[0].weight.shape[0])
    work = prepare_worklist(flat_ids.size, offsets,
                            bm=32 if gemm_config is None else gemm_config.tiles[0])

    def project(x, layer, dtype):
        return _grouped_mxfp4(x, layer.weight, layer.scales, offsets,
                              out_dtype=dtype, worklist=work, config=gemm_config)

    gate = project(x_sorted, mlp.gate_proj, mx.bfloat16)
    up = project(x_sorted, mlp.up_proj, mx.bfloat16)
    # Calls the existing ClampedSwiGLU: FP32 casts, exact clip/min/silu*up.
    hidden = mlp.activation(up, gate)
    down = project(hidden, mlp.down_proj, mx.float32)
    y = down[inverse].reshape(*indices.shape, xf.shape[-1])
    if os.environ.get("DSV41_COMPILE_PRIMITIVES", "0") == "1":
        from .language import _v41_moe_combine

        return _v41_moe_combine(y, weights)
    return mx.sum(y.astype(mx.float32) * weights[..., None], axis=-2)
