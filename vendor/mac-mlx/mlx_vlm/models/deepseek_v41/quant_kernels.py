"""Opt-in Metal ports of the checkpoint's inplace activation quantisers.

These simulate quantisation and reconstruct the original activation dtype;
they do not pack the KV cache. Each SIMD group owns one quantisation block.
No model state or cache buffers are mutated. The caller retains the eager
implementation and DSV41_QAT bypass for numerical comparisons.
"""

from functools import lru_cache

import mlx.core as mx


_HEADER = r"""
inline float dsv41_pow2(int e) {
    return as_type<float>(uint(e + 127) << 23);
}
inline float dsv41_round_scale(float a, float reciprocal) {
    uint bits = as_type<uint>(a * reciprocal);
    int e = int((bits >> 23) & 255u) - 127;
    e += (bits & 0x7fffffu) != 0;
    return dsv41_pow2(e);
}
inline float dsv41_e4m3(float x) {
    float a = metal::abs(x);
    // E4M3FN has subnormal spacing 2^-9 and three fraction bits.
    int e = int((as_type<uint>(a) >> 23) & 255u) - 127;
    float step = dsv41_pow2(metal::max(e - 3, -9));
    float rounded = metal::rint(a / step) * step;
    // 464 is the tie between the last finite even code and the NaN code.
    if (a > 464.0f) return as_type<float>(0x7fc00000u);
    return metal::copysign(metal::min(rounded, 448.0f), x);
}
inline float dsv41_e2m1(float x) {
    float a = metal::abs(x);
    uint upper = a >= 1.75f;
    uint middle = a >= (upper ? 3.5f : 0.75f);
    float last = upper ? (middle ? 5.0f : 2.5f)
                       : (middle ? 1.25f : 0.25f);
    uint code = 4 * upper + 2 * middle + uint(a > last);
    constexpr float levels[8] = {0, .5, 1, 1.5, 2, 3, 4, 6};
    return metal::copysign(levels[code], x);
}
"""

_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint group = thread_position_in_grid.x / 32;
    if (group >= N / BLOCK) return;
    float amax = 0.0f;
    for (uint j = lane; j < BLOCK; j += 32)
        amax = metal::max(amax, metal::abs(float(inp[group * BLOCK + j])));
    amax = metal::simd_max(amax);
    float scale;
    if (MODE == 0) {
        scale = dsv41_round_scale(metal::max(amax, 1e-4f), 1.0f / 448.0f);
    } else if (MODE == 1) {
        scale = dsv41_round_scale(metal::max(amax, 6.0f * 0x1p-126f), 1.0f / 6.0f);
    } else {
        scale = dsv41_e4m3(metal::precise::divide(metal::max(amax, 6.0f * 0x1p-9f), 6.0f));
    }
    float limit = MODE == 0 ? 448.0f : 6.0f;
    for (uint j = lane; j < BLOCK; j += 32) {
        uint i = group * BLOCK + j;
        float v = metal::clamp(metal::precise::divide(float(inp[i]), scale), -limit, limit);
        float q = MODE == 0 ? dsv41_e4m3(v) : dsv41_e2m1(v);
        out[i] = T(q * scale);
    }
"""


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="dsv41_fake_quant", input_names=["inp"], output_names=["out"],
        header=_HEADER, source=_SOURCE,
    )


def fake_quant(x: mx.array, block: int, mode: str) -> mx.array:
    """One dispatch for contiguous finite FP32/FP16/BF16 activation inputs.

    MLX materialises noncontiguous inputs if needed (an additional dispatch).
    Scale modes match act_quant UE8M0, fp4_act_quant E8M0, and its E4M3
    compressed-KV variant. Unsupported contracts fail instead of falling back.
    """
    modes = {"fp8_ue8m0": 0, "fp4_ue8m0": 1, "fp4_e4m3": 2}
    if mode not in modes:
        raise ValueError(f"unknown V4.1 quantisation mode: {mode}")
    if block <= 0 or not x.ndim or x.shape[-1] % block:
        raise ValueError(f"last dimension must be divisible by positive block {block}")
    if x.dtype not in (mx.float32, mx.float16, mx.bfloat16):
        raise ValueError(f"unsupported V4.1 activation dtype: {x.dtype}")
    if x.size == 0:
        return x
    return _kernel()(
        inputs=[x], template=[("T", x.dtype), ("BLOCK", block),
                              ("MODE", modes[mode]), ("N", x.size)],
        grid=(x.size // block * 32, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[x.shape], output_dtypes=[x.dtype],
    )[0]
