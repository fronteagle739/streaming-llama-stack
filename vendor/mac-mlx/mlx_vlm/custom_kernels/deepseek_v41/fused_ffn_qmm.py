"""Experimental M32 gate/up + exact-boundary FP32 ClampedSwiGLU; default off.

Uses the pinned, unmodified MLX Steel/decoder header from expert_tiles_qmm.
No host reads or synchronization in metadata/projection calls.
"""
from functools import lru_cache
import math
from pathlib import Path

import mlx.core as mx

from .expert_tiles_qmm import (
    EXPERTS, _kernels as _base_kernels, _sources as _base_sources,
    expert_tiles_metadata, prewarm_expert_tiles, validate_weight,
)


@lru_cache(maxsize=1)
def _kernel():
    _base_kernels()  # Enforce exactly MLX 0.32.2 and the Metal backend.
    source = Path(__file__).with_suffix(".metal").read_text(encoding="utf-8")
    header, body = source.split("// --- body ---", 1)
    return mx.fast.metal_kernel(
        name="expert_gate_up_swiglu",
        input_names=["X", "GateW", "GateS", "UpW", "UpS", "Offsets", "Prefix", "Limit"],
        output_names=["Hidden"], header=_base_sources()[0] + header, source=body,
        ensure_row_contiguous=True, compile_options={"math_mode": "safe"})


def fused_ffn_metadata(sorted_ids, experts=EXPERTS):
    return expert_tiles_metadata(sorted_ids, experts, tile_rows=32)


def expert_gate_up_swiglu(x, gate_w, gate_s, up_w, up_s, routing, limit):
    """Sorted [R,1,K] BF16 -> [R,1,I] FP32, with BF16 gate/up rounding."""
    k, n = validate_weight(gate_w, gate_s)
    if validate_weight(up_w, up_s) != (k, n):
        raise ValueError("fused gate/up dimensions must match")
    if x.dtype != mx.bfloat16 or x.shape != (routing.rows, 1, k):
        raise ValueError("fused X must be BF16 [routing.rows, 1, K]")
    if (routing.tile_rows != 32 or routing.experts != EXPERTS
            or routing.offsets.shape != (EXPERTS + 1,)
            or routing.prefix.shape != (EXPERTS + 1,)
            or routing.offsets.dtype != mx.uint32 or routing.prefix.dtype != mx.uint32):
        raise ValueError("fused requires a 384-expert uint32 M32 RoutingPlan")
    if not math.isfinite(limit):
        raise ValueError("fused swiglu limit must be finite")
    if routing.rows == 0:
        return mx.zeros((0, 1, n), mx.float32)
    return _kernel()(
        inputs=[x, gate_w, gate_s, up_w, up_s, routing.offsets, routing.prefix,
                mx.array([limit], mx.float32)],
        template=[("T", mx.bfloat16)],
        grid=(128 * (n // 32), routing.candidates, 1), threadgroup=(128, 1, 1),
        output_shapes=[(routing.rows, 1, n)], output_dtypes=[mx.float32])[0]


@lru_cache(maxsize=1)
def prewarm_fused_ffn():
    """Attach-only compile-or-fail, covering device and tiny constant ID inputs."""
    prewarm_expert_tiles()  # Down remains the proven FP32 M16 implementation.
    ids = mx.concatenate((mx.zeros((33,), mx.uint32), mx.full((16,), 383, mx.uint32)))
    plan = fused_ffn_metadata(ids)
    w = mx.zeros((EXPERTS, 32, 4), mx.uint32)
    s = mx.full((EXPERTS, 32, 1), 127, mx.uint8)
    hidden = expert_gate_up_swiglu(mx.zeros((49, 1, 32), mx.bfloat16),
                                  w, s, w, s, plan, 10.0)
    tiny = fused_ffn_metadata(mx.array([383], mx.uint32))
    mx.async_eval(hidden, tiny.offsets, tiny.prefix)
    mx.synchronize()
