"""Opt-in dense MXFP8 prefill via a temporary BF16 weight and Steel GEMM."""

import os

import mlx.core as mx
import mlx.nn as nn


_PREFILL_MXFP8_STEEL = os.environ.get("DSV41_PREFILL_MXFP8_STEEL", "0") == "1"
# Ignore the tuning variable entirely on the default path.
_PREFILL_STEEL_MIN_M = (
    int(os.environ.get("DSV41_PREFILL_STEEL_MIN_M", "256"))
    if _PREFILL_MXFP8_STEEL else 256
)
STEEL_CALLS = 0


def prefill_linear(layer, x):
    """Use Steel only for eligible BF16 MXFP8 matrices; otherwise call layer.

    Switches are read at import time. STEEL_CALLS counts constructed projections,
    not GPU completions (MLX is lazy). Never retain dequantized weights on layer.
    """
    if (
        not _PREFILL_MXFP8_STEEL
        or x.ndim < 2
        or x.shape[-2] < _PREFILL_STEEL_MIN_M
        or x.dtype != mx.bfloat16
        or not isinstance(layer, nn.QuantizedLinear)
        or layer.mode != "mxfp8"
        or layer.group_size != 32
        or layer.bits != 8
    ):
        return layer(x)

    weight = mx.dequantize(
        layer.weight, layer.scales, getattr(layer, "biases", None),
        group_size=32, bits=8, mode="mxfp8", dtype=mx.bfloat16,
    )
    out = x @ weight.T
    bias = getattr(layer, "bias", None)
    if bias is not None:
        out = out + bias
    global STEEL_CALLS
    STEEL_CALLS += 1
    return out
