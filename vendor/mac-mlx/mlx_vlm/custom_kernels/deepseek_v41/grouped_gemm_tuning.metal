// The production header is prepended by Python. MODE: 0 full, 1 decode, 2 MMA.
// Keep scale multiplication in T: BF16 and FP32 have different rounding limits.
constant float dsv41_fp4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Metal vectors have at most four components: two T4 stores form a BF16x8.
template <typename T> inline void dsv41_decode8(
    uint packed, T scale, threadgroup T* dst) {
    vec<T, 4> lo, hi;
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 4; ++j) {
        lo[j] = T(scale * T(dsv41_fp4_lut[(packed >> (4 * j)) & 15]));
        hi[j] = T(scale * T(dsv41_fp4_lut[(packed >> (16 + 4 * j)) & 15]));
    }
    *reinterpret_cast<threadgroup vec<T, 4>*>(dst) = lo;
    *reinterpret_cast<threadgroup vec<T, 4>*>(dst + 4) = hi;
}

// --- kernel body ---
static_assert(BM % 16 == 0 && BN % 16 == 0 && BK % 32 == 0,
              "tiles must cover complete SIMD matrices and scale groups");
const uint tid = thread_index_in_threadgroup;
const uint lane = tid % 32;
const uint sg = tid / 32;
const uint task = threadgroup_position_in_grid.y;
const uint band = threadgroup_position_in_grid.z;
const size_t checksum_index = (size_t(task) * ((N + BN - 1) / BN) + band) * 128 + tid;
const int expert = Work[2 * task];
// task is a THREADGROUP coordinate: every lane reads the same sentinel.
// Never change this to a per-thread task index or a per-row early return.
if (expert < 0) {
    if constexpr (MODE == 1) Output[checksum_index] = 0;
    return;
}
const int row_start = Offsets[expert] + Work[2 * task + 1];
const int row_end = Offsets[expert + 1];
const uint col_start = band * BN;
constexpr int LD = BK + PAD;
constexpr int TM = BM / 16;
constexpr int TN = BN / 16;
alignas(16) threadgroup T xs[BM * LD];
alignas(16) threadgroup T ws[BN * LD];
simdgroup_float8x8 acc[TM][TN];
if constexpr (MODE != 1) {
    for (int mi = 0; mi < TM; ++mi)
        for (int ni = 0; ni < TN; ++ni)
            acc[mi][ni] = simdgroup_float8x8(0.0f);
}
uint checksum = 0;
// Runtime dummy data prevents constant-zero MMA elimination. Fixed, bounded
// rounds initialize the entire allocation, INCLUDING padding, before one
// unconditional TG barrier. Keep this diagnostic loop out of compiler unrolling.
if constexpr (MODE == 2) {
    #pragma clang loop unroll(disable)
    for (uint base = 0; base < BM * LD; base += 128) {
        const uint i = base + tid;
        if (i < BM * LD) xs[i] = i % LD < BK ? X[(i % LD) % K] : T(0);
    }
    #pragma clang loop unroll(disable)
    for (uint base = 0; base < BN * LD; base += 128) {
        const uint i = base + tid;
        if (i < BN * LD) ws[i] = i % LD < BK ? X[(i % LD) % K] : T(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
const uint fm = ((lane / 4) & 4) + ((lane / 2) % 4);
const uint fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
for (uint kb = 0; kb < K; kb += BK) {
    if constexpr (MODE != 2) {
        for (uint i = tid; i < BM * BK; i += 128) {
            const uint m = i / BK, k = i % BK;
            xs[m * LD + k] = row_start + m < row_end && kb + k < K
                ? X[size_t(row_start + m) * K + kb + k] : T(0);
        }
        if constexpr (VECTOR_DECODE) {
            // A thread owns a contiguous 32-value scale group (uint4), reusing
            // one E8M0 conversion across four packed words / eight T4 stores.
            for (uint i = tid; i < BN * (BK / 32); i += 128) {
                const uint n = i / (BK / 32), k = (i % (BK / 32)) * 32;
                if (col_start + n < N && kb + k < K) {
                    const size_t wr = size_t(expert) * N + col_start + n;
                    const uint4 words = *reinterpret_cast<const device uint4*>(
                        W + wr * (K / 8) + (kb + k) / 8);
                    const T scale = dsv41_scale<T>(Scales[wr * (K / 32) + (kb + k) / 32]);
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 4; ++j)
                        dsv41_decode8<T>(words[j], scale, ws + n * LD + k + j * 8);
                } else {
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 8; ++j)
                        *reinterpret_cast<threadgroup vec<T, 4>*>(ws + n * LD + k + j * 4) = vec<T, 4>(0);
                }
            }
        } else {
            // Original decode schedule, retained for attribution of round 1.
            for (uint i = tid; i < BN * (BK / 8); i += 128) {
                const uint n = i / (BK / 8), k = (i % (BK / 8)) * 8;
                if (col_start + n < N && kb + k < K) {
                    const size_t wr = size_t(expert) * N + col_start + n;
                    const uint packed = W[wr * (K / 8) + (kb + k) / 8];
                    const uint s = Scales[wr * (K / 32) + (kb + k) / 32];
                    const T scale = dsv41_scale<T>(s);
                    for (uint j = 0; j < 8; ++j)
                        ws[n * LD + k + j] = dsv41_decode<T>((packed >> (4 * j)) & 15, scale);
                } else {
                    for (uint j = 0; j < 8; ++j) ws[n * LD + k + j] = T(0);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if constexpr (MODE == 1) {
        // Consume EVERY staged element at EVERY K step, with a different lane
        // reading it. This preserves shared-memory stores and both barriers.
        // Hash/readback overhead makes this a diagnostic, not additive timing.
        // This is a per-lane hash, NOT a cross-lane reduction. No barriers or
        // data-dependent loop exits belong here. Bound the dependent hash chain
        // explicitly and prohibit unrolling (suspected probe/compiler trigger).
        static_assert((BM * BK) % 128 == 0 && (BN * BK) % 128 == 0,
                      "checksum rounds must cover complete lane groups");
        const uint reader = (tid + 32) % 128;
        #pragma clang loop unroll(disable)
        for (uint step = 0; step < (BM * BK) / 128; ++step) {
            const uint i = step * 128 + reader;
            checksum = rotate(checksum, 5u) ^ as_type<uint>(float(xs[(i / BK) * LD + i % BK]));
        }
        #pragma clang loop unroll(disable)
        for (uint step = 0; step < (BN * BK) / 128; ++step) {
            const uint i = step * 128 + reader;
            checksum = rotate(checksum, 5u) ^ as_type<uint>(float(ws[(i / BK) * LD + i % BK]));
        }
    } else {
        // Preserve ascending 8-wide reduction order, including BK64 and BM128.
        #pragma clang loop unroll_count(UNROLL)
        for (uint kk = 0; kk < BK; kk += 8) {
            if constexpr (NATIVE_LOADS) {
                simdgroup_matrix<T, 8, 8> a[TM], b[TN];
                #pragma clang loop unroll(full)
                for (uint mi = 0; mi < TM; ++mi) {
                    const uint m = CONTIGUOUS ? (sg / 2) * (BM / 2) + mi * 8 : (sg / 2) * 8 + mi * 16;
                    simdgroup_load(a[mi], xs + m * LD + kk, LD);
                }
                #pragma clang loop unroll(full)
                for (uint ni = 0; ni < TN; ++ni) {
                    const uint n = CONTIGUOUS ? (sg % 2) * (BN / 2) + ni * 8 : (sg % 2) * 8 + ni * 16;
                    simdgroup_load(b[ni], ws + n * LD + kk, LD, ulong2(0), true);
                }
                #pragma clang loop unroll(full)
                for (uint mi = 0; mi < TM; ++mi)
                    #pragma clang loop unroll(full)
                    for (uint ni = 0; ni < TN; ++ni)
                        simdgroup_multiply_accumulate(acc[mi][ni], a[mi], b[ni], acc[mi][ni]);
            } else {
                simdgroup_float8x8 a[TM], b[TN];
                for (uint mi = 0; mi < TM; ++mi) {
                    const uint m = (CONTIGUOUS ? (sg / 2) * (BM / 2) + mi * 8 : (sg / 2) * 8 + mi * 16) + fm;
                    for (uint j = 0; j < 2; ++j)
                        a[mi].thread_elements()[j] = float(xs[m * LD + kk + fn + j]);
                }
                for (uint ni = 0; ni < TN; ++ni) {
                    const uint n = (CONTIGUOUS ? (sg % 2) * (BN / 2) + ni * 8 : (sg % 2) * 8 + ni * 16) + fn;
                    for (uint j = 0; j < 2; ++j)
                        b[ni].thread_elements()[j] = float(ws[(n + j) * LD + kk + fm]);
                }
                for (uint mi = 0; mi < TM; ++mi)
                    for (uint ni = 0; ni < TN; ++ni)
                        simdgroup_multiply_accumulate(acc[mi][ni], a[mi], b[ni], acc[mi][ni]);
            }
        }
    }
    // All 128 lanes reach both K-step barriers in EVERY MODE.
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if constexpr (MODE == 1) {
    Output[checksum_index] = checksum;
} else {
    for (uint mi = 0; mi < TM; ++mi) {
        const uint row = row_start + (CONTIGUOUS ? (sg / 2) * (BM / 2) + mi * 8 : (sg / 2) * 8 + mi * 16) + fm;
        for (uint ni = 0; ni < TN; ++ni) {
            const uint col = col_start + (CONTIGUOUS ? (sg % 2) * (BN / 2) + ni * 8 : (sg % 2) * 8 + ni * 16) + fn;
            for (uint j = 0; j < 2; ++j)
                if (row < row_end && col + j < N)
                    Output[size_t(row + GUARD) * N + col + j] = O(acc[mi][ni].thread_elements()[j]);
        }
    }
}
