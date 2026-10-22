#pragma once

// Source arithmetic: stack-backend c8aa547e, metal/dsv41.metal and
// metal/dsv4_kv.metal. Shared scalar functions keep CPU and CUDA selection-
// critical rounding/scale boundaries identical. V4's old modes are unchanged.
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef __CUDACC__
#define GGML_DSV41_HD __host__ __device__
#else
#define GGML_DSV41_HD
#endif

GGML_DSV41_HD static inline uint32_t ggml_dsv41_float_bits(float x) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(x);
#else
    uint32_t bits; memcpy(&bits, &x, sizeof(bits)); return bits;
#endif
}

GGML_DSV41_HD static inline float ggml_dsv41_bits_float(uint32_t bits) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(bits);
#else
    float x; memcpy(&x, &bits, sizeof(x)); return x;
#endif
}

GGML_DSV41_HD static inline float ggml_dsv41_bf16(float x) {
    uint32_t bits = ggml_dsv41_float_bits(x);
    if ((bits & 0x7f800000u) != 0x7f800000u) {
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    }
    return ggml_dsv41_bits_float(bits & 0xffff0000u);
}

GGML_DSV41_HD static inline float ggml_dsv41_pow2_ceil(float x) {
    const uint32_t bits = ggml_dsv41_float_bits(x);
    return ggml_dsv41_bits_float((bits & 0x7f800000u) + ((bits & 0x7fffffu) ? 0x800000u : 0u));
}

GGML_DSV41_HD static inline float ggml_dsv41_e4m3_value(int code) {
    const int exponent = code >> 3, mantissa = code & 7;
    return exponent ? ldexpf(float(8 + mantissa), exponent - 10) : ldexpf(float(mantissa), -9);
}

GGML_DSV41_HD static inline float ggml_dsv41_e4m3(float x) {
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (ggml_dsv41_e4m3_value(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    if (lo < 126) {
        const float d0 = fabsf(ax - ggml_dsv41_e4m3_value(lo));
        const float d1 = fabsf(ax - ggml_dsv41_e4m3_value(lo + 1));
        if (d1 < d0 || (d1 == d0 && (lo & 1))) ++lo;
    }
    return copysignf(ggml_dsv41_e4m3_value(lo), x);
}

GGML_DSV41_HD static inline float ggml_dsv41_e2m1_value(int code) {
    // Avoid a device constant array / an extra GPU allocation.
    switch (code) {
        case 0: return 0.f; case 1: return .5f; case 2: return 1.f;
        case 3: return 1.5f; case 4: return 2.f; case 5: return 3.f;
        case 6: return 4.f; default: return 6.f;
    }
}

GGML_DSV41_HD static inline float ggml_dsv41_e2m1(float x) {
    const float ax = fminf(fabsf(x), 6.f);
    int best = 0;
    float diff = ax;
    for (int i = 1; i < 8; ++i) {
        const float d = fabsf(ax - ggml_dsv41_e2m1_value(i));
        if (d < diff || (d == diff && !(i & 1) && (best & 1))) { best = i; diff = d; }
    }
    return copysignf(ggml_dsv41_e2m1_value(best), x);
}

// Production CUDA uses fast-math. Its reciprocal approximation can move an
// exact E2M1 midpoint across a tie (75 / 30 -> 2.5000002, selecting 3 instead
// of ties-to-even 2). Selection-critical divisions require IEEE round-to-
// nearest even independently of the surrounding kernel's math flags.
GGML_DSV41_HD static inline float ggml_dsv41_divide(float value, float scale) {
#ifdef __CUDA_ARCH__
    return __fdiv_rn(value, scale);
#else
    return value / scale;
#endif
}

// Native formats: 0 BF16; 1 FP8/E8M0; 2 FP4/E8M0; 3 FP4/E4M3.
// Inputs to the block amax have ALREADY been BF16-rounded. FP4/E4M3 uses
// 16-element blocks; FP4/E8M0 and FP8/E8M0 use 32, including the RoPE tail.
GGML_DSV41_HD static inline float ggml_dsv41_scale(float amax, int format) {
    if (format == 3) return ggml_dsv41_e4m3(ggml_dsv41_divide(fmaxf(amax, .01171875f), 6.f));
    if (format == 2) return ggml_dsv41_pow2_ceil(fmaxf(amax, 7.052966104933725e-38f) * (1.f / 6.f));
    return ggml_dsv41_pow2_ceil(fmaxf(amax, 1.e-4f) * (1.f / 448.f));
}

GGML_DSV41_HD static inline float ggml_dsv41_quantized(float value, float scale, int format) {
    const float normalized = ggml_dsv41_divide(fabsf(value), scale);
    const float magnitude = format == 1 ? ggml_dsv41_e4m3(normalized)
                                        : ggml_dsv41_e2m1(normalized);
    return ggml_dsv41_bf16(copysignf(magnitude, value) * scale);
}

// CPU row implementation, also used by model-free boundary fixtures. The
// caller validates width divisibility and format before entering the op.
static inline void ggml_dsv41_quantize_row(float * dst, const float * src, int64_t width, int format) {
    if (format == 0) {
        for (int64_t i = 0; i < width; ++i) dst[i] = ggml_dsv41_bf16(src[i]);
        return;
    }
    const int64_t block = format == 3 ? 16 : 32;
    for (int64_t off = 0; off < width; off += block) {
        float values[32], amax = 0.f;
        for (int64_t i = 0; i < block; ++i) {
            values[i] = ggml_dsv41_bf16(src[off + i]);
            amax = fmaxf(amax, fabsf(values[i]));
        }
        const float scale = ggml_dsv41_scale(amax, format);
        for (int64_t i = 0; i < block; ++i) {
            dst[off + i] = ggml_dsv41_quantized(values[i], scale, format);
        }
    }
}

#undef GGML_DSV41_HD
