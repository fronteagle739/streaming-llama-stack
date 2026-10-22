#pragma once

// Native arithmetic: stack-backend c8aa547e03ff14839b32763045a2c86f2e8362b5,
// metal/dsv41.metal Engram add and stable ratio-2 pooling. Kept separate from
// old V4 operators; these are forward-only V4.1 graph boundaries.
#include "ggml-dsv41-quant.h"
#include <math.h>

#ifdef __CUDACC__
#define GGML_DSV41_OP_HD __host__ __device__
#else
#define GGML_DSV41_OP_HD
#endif

// Native V4.1 keeps the sign of zero here. ggml_sgn(dot) * sqrt(...)
// gives zero at +0 and changes the gate from ~0.50025 to 0.5.
GGML_DSV41_OP_HD static inline float ggml_dsv41_engram_gate(float dot) {
    const float signed_root = copysignf(sqrtf(fmaxf(fabsf(dot), 1.0e-6f)), dot);
    return 1.0f / (1.0f + expf(-signed_root));
}

GGML_DSV41_OP_HD static inline float ggml_dsv41_pool2_value(float ka, float kb, float sa, float sb) {
    const float peak = fmaxf(sa, sb);
    const float ea = expf(sa - peak);
    const float eb = expf(sb - peak);
    return ggml_dsv41_bf16(ggml_dsv41_divide(ka * ea + kb * eb, ea + eb));
}

GGML_DSV41_OP_HD static inline float ggml_dsv41_rope_product(float a, float b) {
#ifdef __CUDA_ARCH__
    // Explicit non-FTZ arithmetic also preserves valid BF16 subnormal values
    // under the rest of the backend's production fast-math recipe.
    float result; asm("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b)); return result;
#else
    return a * b;
#endif
}

GGML_DSV41_OP_HD static inline float ggml_dsv41_rope_fma(float a, float b, float c) {
#ifdef __CUDA_ARCH__
    float result; asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(a), "f"(b), "f"(c)); return result;
#else
    return fmaf(a, b, c);
#endif
}

GGML_DSV41_OP_HD static inline float ggml_dsv41_add_rn(float a,float b) {
#ifdef __CUDA_ARCH__
    float result;asm("add.rn.f32 %0, %1, %2;":"=f"(result):"f"(a),"f"(b));return result;
#else
    return a+b;
#endif
}
GGML_DSV41_OP_HD static inline float ggml_dsv41_relu(float value) {
    const uint32_t bits=ggml_dsv41_float_bits(value),magnitude=bits&0x7fffffffu;
    // Same finite/NaN behavior as fmax(value,0), without fast-math max.ftz
    // erasing valid positive subnormal index scores.
    return (bits&0x80000000u) || magnitude>0x7f800000u ? 0.f : value;
}

struct ggml_dsv41_rope_result { float re, im; };
GGML_DSV41_OP_HD static inline ggml_dsv41_rope_result ggml_dsv41_rope_pair(
        float re, float im, float theta, bool inverse) {
    // Fast single-precision trig intrinsics lose long-context phase accuracy.
    // Double libdevice trig remains precise under -use_fast_math; the angle
    // itself is still the native rounded F32 position*frequency product.
    const float c = (float)cos((double)theta);
    const float s = (inverse ? -1.f : 1.f) * (float)sin((double)theta);
    // nvcc -use_fast_math lowers floating negation to neg.ftz.f32. Even a
    // sign-bit xor can be folded back to that instruction; explicit non-FTZ
    // negation avoids silently erasing a valid BF16 subnormal operand.
#ifdef __CUDA_ARCH__
    float minus_im; asm("neg.f32 %0, %1;" : "=f"(minus_im) : "f"(im));
#else
    const float minus_im = ggml_dsv41_bits_float(ggml_dsv41_float_bits(im) ^ 0x80000000u);
#endif
    return {ggml_dsv41_bf16(ggml_dsv41_rope_fma(minus_im, s, ggml_dsv41_rope_product(re, c))),
            ggml_dsv41_bf16(ggml_dsv41_rope_fma(im, c, ggml_dsv41_rope_product(re, s)))};
}

#undef GGML_DSV41_OP_HD
