#include <metal_simdgroup_matrix>

template <typename T> inline T dsv41_scale(uint s) {
    // MLX fp8_e8m0: 0 -> 2^-127, 255 -> +inf (not NaN).
    return T(as_type<float>(s == 0 ? 0x00400000u : s << 23));
}

template <typename T> inline T dsv41_decode(uint q, T scale) {
    // MLX fp4_e2m1: [0,.5,1,1.5,2,3,4,6], with signed zero.
    float v = float(as_type<half>(ushort((q & 7) << 9))) * 16384.0f;
    v = (q & 8) ? -v : v;
    return T(scale * T(v));
}

// --- kernel body ---
// Expert-aligned MXFP4: four SIMD groups, 2 x 2 arrangement of warp tiles.
// T is BF16 for gate/up and FP32 for down (including weight staging).
// FP32 fragments use the lane mapping in MLX Steel BaseMMAFrag<float,8,8>.
static_assert(BM % 16 == 0 && BN % 16 == 0 && BK % 32 == 0,
              "tiles must cover complete SIMD matrices and scale groups");
const uint tid = thread_index_in_threadgroup;
const uint lane = tid % 32;
const uint sg = tid / 32;
const uint task = threadgroup_position_in_grid.y;
const int expert = Work[2 * task];
if (expert < 0) return; // Uniform TG exit for unused worklist capacity.
const int row_start = Offsets[expert] + Work[2 * task + 1];
const int row_end = Offsets[expert + 1];
const uint col_start = threadgroup_position_in_grid.z * BN;
constexpr int LD = BK + 8;
constexpr int TM = BM / 16;
constexpr int TN = BN / 16;
threadgroup T xs[BM * LD];
threadgroup T ws[BN * LD];
simdgroup_float8x8 acc[TM][TN];
for (int mi = 0; mi < TM; ++mi)
    for (int ni = 0; ni < TN; ++ni)
        acc[mi][ni] = simdgroup_float8x8(0.0f);

const uint fm = ((lane / 4) & 4) + ((lane / 2) % 4);
const uint fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
for (uint kb = 0; kb < K; kb += BK) {
    for (uint i = tid; i < BM * BK; i += 128) {
        const uint m = i / BK, k = i % BK;
        xs[m * LD + k] = row_start + m < row_end && kb + k < K
            ? X[size_t(row_start + m) * K + kb + k] : T(0);
    }
    // Each packed word is loaded once and its eight nibbles decoded once.
    for (uint i = tid; i < BN * (BK / 8); i += 128) {
        const uint n = i / (BK / 8), k = (i % (BK / 8)) * 8;
        if (col_start + n < N && kb + k < K) {
            const size_t wr = size_t(expert) * N + col_start + n;
            const uint packed = W[wr * (K / 8) + (kb + k) / 8];
            const uint s = Scales[wr * (K / 32) + (kb + k) / 32];
            const T scale = dsv41_scale<T>(s);
            for (uint j = 0; j < 8; ++j) {
                const uint q = (packed >> (4 * j)) & 15;
                ws[n * LD + k + j] = dsv41_decode<T>(q, scale);
            }
        } else {
            for (uint j = 0; j < 8; ++j) ws[n * LD + k + j] = T(0);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < BK; kk += 8) {
        simdgroup_float8x8 a[TM], b[TN];
        for (uint mi = 0; mi < TM; ++mi) {
            const uint m = (sg / 2) * 8 + mi * 16 + fm;
            for (uint j = 0; j < 2; ++j)
                a[mi].thread_elements()[j] = float(xs[m * LD + kk + fn + j]);
        }
        for (uint ni = 0; ni < TN; ++ni) {
            const uint n = (sg % 2) * 8 + ni * 16 + fn;
            for (uint j = 0; j < 2; ++j)
                b[ni].thread_elements()[j] = float(ws[(n + j) * LD + kk + fm]);
        }
        for (uint mi = 0; mi < TM; ++mi)
            for (uint ni = 0; ni < TN; ++ni)
                simdgroup_multiply_accumulate(acc[mi][ni], a[mi], b[ni], acc[mi][ni]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint mi = 0; mi < TM; ++mi) {
    const uint row = row_start + (sg / 2) * 8 + mi * 16 + fm;
    for (uint ni = 0; ni < TN; ++ni) {
        const uint col = col_start + (sg % 2) * 8 + ni * 16 + fn;
        for (uint j = 0; j < 2; ++j)
            if (row < row_end && col + j < N)
                Output[size_t(row + GUARD) * N + col + j] = O(acc[mi][ni].thread_elements()[j]);
    }
}
