#include "common.cuh"
#include "cp-async.cuh"
#include "mma.cuh"
#include "fattn-common.cuh"

// TEST: Phase 2.2 angle 47 — f32 VKQ accumulator for SM120 NCOLS=64 MLA path.
// Enables a parallel tile<16,8,float> accumulator alongside the existing
// tile<16,8,half2> path. Auto-activated only for (DKQ=576, DV=512, ncols=64).
// Other shapes constexpr-fold to identical-to-HEAD code paths.
// To disable: comment this #define out and rebuild.
// [2026-08-27] DISABLED. This accumulator is not a fix for the MLA shape -- it IS the defect.
//
// Measured with test-backend-ops on hsk=576/hsv=512 (the DeepSeek/GLM MLA shape), after
// extending the kv sweep past 1024 and relaxing the nr2 filter that pinned that shape to
// kv=512 (both gaps closed in commit ca80c5b0):
//     enabled  (ncols=64 only, as shipped)   86-89/144 passed, ~55 failures
//     enabled  everywhere (ncols=32 too)     69/144    passed,  worse still
//     DISABLED                               **144/144 passed, ZERO failures**
//
// The failing set and the fp32-enabled set coincide exactly: the dispatcher sends nb>=32 to
// the ncols=64 path (Q->ne[1] <= 32/ncols2 is false), which is the only path where this was
// active, and nb=1/3 -- which never failed -- take the smaller-ncols path where it was off.
//
// End-to-end on GLM-5.2, the configuration that produced token soup with it enabled:
//     dense+fa=on, enabled    0/3, degenerate ("5-7-1-0-0-0-0-...")
//     dense+fa=on, DISABLED   6/9 recall, all 3 misses REFUSALS not garbage; 6/6 when answered
//     and at 53,076 tokens multi-turn the needle comes back EXACT, with turn-1 prefill
//     418.5s versus 659.6s for the unfused fa=off path -- correct AND 1.58x faster.
//
// The original rationale was "cumulative half2 accumulation saturates over 50+ MLA layers".
// The 53k run exercises all 78 layers and recalls exactly, which is evidence against that
// concern -- but it is ONE configuration, so re-enable via this define if a case is found
// where half2 genuinely drifts. The code below is left intact for that purpose.
// #define LLAMA_FA_VKQ_F32_ACC 1

using namespace ggml_cuda_mma;

// Config options for the MMA kernel.
// Should not affect results, only speed/register pressure/shared memory use.
struct fattn_mma_config {
    int  nthreads;       // Number of threads per CUDA block.
    int  occupancy;      // Targeted occupancy for the MMA kernel.
    int  nbatch_fa;      // Number of KV rows per softmax rescaling of KQ rowsums and VKQ accumulators.
    int  nbatch_K2;      // Number of K half2 values in direction of DKQ to load in parallel.
    int  nbatch_V2;      // Number of V half2 values in direction of DV to load in parallel.
    int  nbatch_combine; // Number of VKQ half2 values in direction of DV to combine in parallel.
    int  nstages_target; // Number of pipeline stages to use ideally, 1 == always load data synchronously, 2 == preload data if there is hardware support.
    bool Q_in_reg;       // Whether the Q values should be kept permanently in registers.

    constexpr __host__ __device__ fattn_mma_config(
            int nthreads, int occupancy, int nbatch_fa, int nbatch_K2, int nbatch_V2, int nbatch_combine, int nstages_target, bool Q_in_reg) :
        nthreads(nthreads), occupancy(occupancy), nbatch_fa(nbatch_fa), nbatch_K2(nbatch_K2), nbatch_V2(nbatch_V2), nbatch_combine(nbatch_combine),
        nstages_target(nstages_target), Q_in_reg(Q_in_reg) {}
};

#define GGML_CUDA_FATTN_MMA_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads_, occupancy_, nbatch_fa_, nbatch_K2_, nbatch_V2_, nbatch_combine_, nstages_target_, Q_in_reg_) \
    if (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                                                                                       \
        static_assert((nthreads_)       % 32 == 0 && (nthreads_)       <= 512, "bad nthreads");                                                                    \
        static_assert(                               (occupancy_)      <=   8, "bad occupancy");                                                                   \
        static_assert((nbatch_fa_)      % 32 == 0 && (nbatch_fa_)      <= 256, "bad nbatch_fa");                                                                   \
        static_assert((nbatch_K2_)      %  4 == 0 && (nbatch_K2_)      <= 512, "bad nbatch_K2");                                                                   \
        static_assert((nbatch_V2_)      %  4 == 0 && (nbatch_V2_)      <= 256, "bad nbatch_V2");                                                                   \
        static_assert((nbatch_combine_) %  4 == 0 && (nbatch_combine_) <= 128, "bad nbatch_combine");                                                              \
        static_assert((nstages_target_)      >= 1 && (nstages_target_) <=   2, "bad nstages_target");                                                              \
        return fattn_mma_config{(nthreads_), (occupancy_), (nbatch_fa_), (nbatch_K2_), (nbatch_V2_), (nbatch_combine_), (nstages_target_), (Q_in_reg_)};           \
    }                                                                                                                                                              \

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_ampere(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64,  8, 128, 2, 128,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 16, 128, 2,  64,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 32, 128, 2,  64,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 64, 128, 2,  64,  32,  32,  32, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80,  8, 128, 2, 128,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 16, 128, 2,  64,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 32, 128, 2,  64,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 64, 128, 2,  64,  40,  40,  40, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96,  8, 128, 2, 128,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 16, 128, 2,  64,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 32, 128, 2,  64,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 64, 128, 2,  64,  48,  48,  48, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112,  8, 128, 2, 128,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 16, 128, 2,  64,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 32, 128, 2,  64,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 64, 128, 2,  64,  56,  56,  56, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  8, 128, 2, 128,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16, 128, 2,  64,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 128, 2,  64,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64, 128, 2,  64,  64,  64,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8,  64, 4,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16,  64, 4,  32, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  32, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  32, 128, 128, 128, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8,  64, 4,  32, 256, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16,  64, 4,  32, 256, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32, 288, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32, 288, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128, 128, 1, false);

    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_turing(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8, 128, 2,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 128, 2,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  64, 128, 128,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  64, 128, 128,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128, 128, 1, false);

    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_volta(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8,  64, 4,  32, 256, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16,  64, 4,  32, 256, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  32, 128, 128,  64, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32, 288, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32, 288, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128,  64, 1, false);

    // TODO tune specifically for Volta
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_rdna(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 128, 2,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  64, 128, 128,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  64, 128, 128,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128, 128, 1, false);

    // TODO tune specifically for RDNA
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
}

static __host__ fattn_mma_config ggml_cuda_fattn_mma_get_config(const int DKQ, const int DV, const int ncols, const int cc) {
    if (ampere_mma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
    }
    if (turing_mma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_turing(DKQ, DV, ncols);
    }
    if (amd_wmma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_rdna(DKQ, DV, ncols);
    }
    GGML_ASSERT(volta_mma_available(cc));
    return ggml_cuda_fattn_mma_get_config_volta(DKQ, DV, ncols);
}

static constexpr __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config(const int DKQ, const int DV, const int ncols) {
#if defined(AMPERE_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
#elif defined(TURING_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_turing(DKQ, DV, ncols);
#elif defined(VOLTA_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_volta(DKQ, DV, ncols);
#elif defined(AMD_WMMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_rdna(DKQ, DV, ncols);
#else
    GGML_UNUSED_VARS(DKQ, DV, ncols);
    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
#endif // defined(AMPERE_MMA_AVAILABLE)
}

static __host__ int ggml_cuda_fattn_mma_get_nthreads(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nthreads;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nthreads(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nthreads;
}

static __host__ int ggml_cuda_fattn_mma_get_occupancy(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).occupancy;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_occupancy(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).occupancy;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_fa(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_fa;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_fa(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_fa;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_K2(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_K2;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_K2(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_K2;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_V2(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_V2;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_V2(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_V2;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_combine(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_combine;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_combine(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_combine;
}

static __host__ int ggml_cuda_fattn_mma_get_nstages_target(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nstages_target;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nstages_target(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nstages_target;
}

static __host__ bool ggml_cuda_fattn_mma_get_Q_in_reg(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).Q_in_reg;
}

static constexpr __device__ bool ggml_cuda_fattn_mma_get_Q_in_reg(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).Q_in_reg;
}

static constexpr __device__ int get_cols_per_thread() {
#if defined(AMD_WMMA_AVAILABLE)
    return 1; // RDNA has a single column.
#else
    return 2; // This is specifically KQ columns, Volta only has a single VKQ column.
#endif // defined(AMD_WMMA_AVAILABLE)
}

static __host__ int get_cols_per_warp(const int cc) {
    if (turing_mma_available(cc) || amd_wmma_available(cc)) {
        return 16;
    } else {
        // Volta
        return 32;
    }
}

// ------------------------------------------------------------------------------------------------------------------

static __host__ int ggml_cuda_fattn_mma_get_nstages(const int DKQ, const int DV, const int ncols1, const int ncols2, const int cc) {
    return cp_async_available(cc) && ncols2 >= 2 ? ggml_cuda_fattn_mma_get_nstages_target(DKQ, DV, ncols1*ncols2, cc) : 0;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nstages(const int DKQ, const int DV, const int ncols1, const int ncols2) {
#ifdef CP_ASYNC_AVAILABLE
    return ncols2 >= 2 ? ggml_cuda_fattn_mma_get_nstages_target(DKQ, DV, ncols1*ncols2) : 0;
#else
    GGML_UNUSED_VARS(DKQ, DV, ncols1, ncols2);
    return 0;
#endif // CP_ASYNC_AVAILABLE
}

// ------------------------------------------------------------------------------------------------------------------

// Sparse gather variant: load nbatch_fa rows from KV_base using per-row indices[base_idx + i].
// Each row starts at KV_base + indices[base_idx+i]*stride_KV and contributes D2 half2 values.
// Loads are done synchronously (no cp_async) row-by-row.
template<int stride_tile, int nwarps, int nbatch_fa, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_f16_load_tile_sparse(
        const half2 * const __restrict__ KV_base,
        const int    * const __restrict__ indices,
        const int base_idx,
        half2 * const __restrict__ tile_KV,
        const int D2, const int stride_KV, const int i_sup) {
    // Each thread cooperates to fill tile_KV[i * stride_tile + k] for i in [0, nbatch_fa), k in [0, D2).
    // Use the same coopearative pattern as the dense loader (decreasing granularity).
    auto load = [&] __device__ (const int n) {
        const int stride_k = WARP_SIZE >> n;
        const int k0_start = stride_k == WARP_SIZE ? 0 : D2 - D2 % (2*stride_k);
        const int k0_stop  =                             D2 - D2 % (1*stride_k);
        const int stride_i = WARP_SIZE / stride_k;

        if (k0_start == k0_stop) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
            const int i = i0 + threadIdx.y*stride_i + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

            if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                break;
            }

            if (oob_check && i >= i_sup) {
                // zero-fill out-of-bounds rows
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);
                    tile_KV[i*stride_tile + k] = make_half2(0.0f, 0.0f);
                }
                continue;
            }

            // Gather: map row i to physical row via indices
            const int phys_row = __ldg(indices + base_idx + i);

#pragma unroll
            for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);
                tile_KV[i*stride_tile + k] = phys_row >= 0 ?
                    KV_base[int64_t(phys_row) * stride_KV + k] :
                    make_half2(0.0f, 0.0f);
            }
        }
    };
    ggml_cuda_unroll<4>{}(load);
}

template<int stride_tile, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_f16_load_tile(
        const half2 * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int D2, const int stride_KV, const int i_sup) {
    // K/V data is loaded with decreasing granularity for D for better memory bandwidth.
    // The minimum granularity with cp.async is 16 bytes, with synchronous data loading it's 4 bytes.
    if constexpr (use_cp_async) {
        static_assert(!oob_check, "OOB check not compatible with cp_async");
        constexpr int preload = 64;
        constexpr int h2_per_chunk = 16/sizeof(half2);
        const int chunks_per_row = D2 / h2_per_chunk;

        const unsigned int tile_KV_32 = ggml_cuda_cvta_generic_to_shared(tile_KV);

        auto load = [&] __device__ (auto n) {
            const int stride_k = WARP_SIZE >> n;
            const int k0_start = stride_k == WARP_SIZE ? 0 : chunks_per_row - chunks_per_row % (2*stride_k);
            const int k0_stop  =                             chunks_per_row - chunks_per_row % (1*stride_k);
            const int stride_i = WARP_SIZE / stride_k;

            if (k0_start == k0_stop) {
                return;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
                const int i = i0 + threadIdx.y*stride_i + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

                if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                    break;
                }

#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    cp_async_cg_16<preload>(tile_KV_32 + i*(stride_tile*sizeof(half2)) + k*16, KV + i*stride_KV + k*h2_per_chunk);
                }
            }
        };
        // 1: max 32*16=512 bytes, 256 half
        // 2: max 16*16=256 bytes, 128 half
        // 3: max  8*16=128 bytes,  64 half
        // 4: max  4*16= 64 bytes,  32 half
        // 5: max  2*16= 32 bytes,  16 half
        // 6: max  1*16= 16 bytes,   8 half
        ggml_cuda_unroll<6>{}(load);
    } else {
        // TODO use ggml_cuda_memcpy_1
        auto load = [&] __device__ (const int n) {
            const int stride_k = WARP_SIZE >> n;
            const int k0_start = stride_k == WARP_SIZE ? 0 : D2 - D2 % (2*stride_k);
            const int k0_stop  =                             D2 - D2 % (1*stride_k);
            const int stride_i = WARP_SIZE / stride_k;

            if (k0_start == k0_stop) {
                return;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
                const int i = i0 + threadIdx.y*stride_i + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

                if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                    break;
                }

#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    tile_KV[i*stride_tile + k] = !oob_check || i < i_sup ? KV[i*stride_KV + k] : make_half2(0.0f, 0.0f);
                }
            }
        };
        // 1: max 32* 4=128 bytes,  64 half
        // 2: max 16* 4= 64 bytes,  32 half
        // 3: max  8* 4= 32 bytes,  16 half
        // 4: max  4* 4= 16 bytes,   8 half
        ggml_cuda_unroll<4>{}(load);
    }
}

template<int ncols1, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_f16_load_mask(
        const half * const __restrict__ mask_h, half * const __restrict__ tile_mask,
        const int stride_mask, const int i_sup, const int j0, const uint3 ne01) {
    if constexpr (use_cp_async) {
        static_assert(nbatch_fa <= 8*WARP_SIZE && nbatch_fa % 8 == 0, "bad nbatch_fa");
        static_assert(!oob_check, "OOB check incompatible with cp_async");
        constexpr int preload = nbatch_fa >= 32 ? nbatch_fa * sizeof(half) : 64;
        constexpr int cols_per_warp = 8*WARP_SIZE/nbatch_fa;
        constexpr int stride_j = nwarps * cols_per_warp;

        const unsigned int tile_mask_32 = ggml_cuda_cvta_generic_to_shared(tile_mask);

#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += stride_j) {
            const int j_sram = j1 + threadIdx.y*cols_per_warp + threadIdx.x / (WARP_SIZE/cols_per_warp);
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + stride_j > ncols1 && j_sram >= ncols1) {
                break;
            }

            const int i = 8 * (threadIdx.x % (nbatch_fa/8));

            cp_async_cg_16<preload>(tile_mask_32 + j_sram*(nbatch_fa*sizeof(half) + 16) + i*sizeof(half), mask_h + j_vram*stride_mask + i);
        }
    } else if constexpr (oob_check) {
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
            const int j_sram = j1 + threadIdx.y;
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + nwarps > ncols1 && j_sram >= ncols1) {
                break;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                tile_mask[j_sram*(nbatch_fa + 8) + i] = i < i_sup ? mask_h[j_vram*stride_mask + i] : half(0.0f);
            }
        }
    } else if constexpr (nbatch_fa < 2*WARP_SIZE) {
        constexpr int cols_per_warp = 2*WARP_SIZE/nbatch_fa;
        constexpr int stride_j = nwarps * cols_per_warp;
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += stride_j) {
            const int j_sram = j1 + threadIdx.y*cols_per_warp + threadIdx.x / (WARP_SIZE/cols_per_warp);
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + stride_j > ncols1 && j_sram >= ncols1) {
                break;
            }

            const int i = threadIdx.x % (WARP_SIZE/cols_per_warp);

            ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + 2*i, mask_h + j_vram*stride_mask + 2*i);
        }
    } else {
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
            const int j_sram = j1 + threadIdx.y;
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + nwarps > ncols1 && j_sram >= ncols1) {
                break;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += 2*WARP_SIZE) {
                const int i = i0 + 2*threadIdx.x;

                ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + i, mask_h + j_vram*stride_mask + i);
            }
        }
    }
}

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps,
    bool use_logit_softcap, bool V_is_K_view, bool needs_fixup, bool is_fixup, bool last_iter, bool oob_check,
    bool sparse,
    typename T_A_KQ, typename T_B_KQ, typename T_C_KQ, typename T_A_VKQ, typename T_B_VKQ, typename T_C_VKQ>
static __device__ __forceinline__ void flash_attn_ext_f16_iter(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half   * const __restrict__ mask_h,
        const int    * const __restrict__ indices,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int ne11,
        const bool direct_causal,
        const int n_past_causal,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        half2        * const __restrict__ tile_Q,
        half2        * const __restrict__ tile_K,
        half2        * const __restrict__ tile_V,
        half         * const __restrict__ tile_mask,
        T_B_KQ       * const __restrict__ Q_B,
        T_C_VKQ      * const __restrict__ VKQ_C,
        tile<16, 8, float> * const __restrict__ VKQ_C_f32,
        float        * const __restrict__ KQ_max,
        float        * const __restrict__ KQ_rowsum,
        const int jt,
        const int kb0,
        const int k_VKQ_sup) {
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA4))
    constexpr int  ncols           = ncols1 * ncols2;
    constexpr int  cols_per_warp   = T_B_KQ::I;
    constexpr int  cols_per_thread = get_cols_per_thread();
    constexpr int  np              = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols; // Number of parallel CUDA warps per Q column.
    constexpr int  nbatch_fa       = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
    constexpr int  nbatch_K2       = ggml_cuda_fattn_mma_get_nbatch_K2(DKQ, DV, ncols);
    constexpr int  nbatch_V2       = ggml_cuda_fattn_mma_get_nbatch_V2(DKQ, DV, ncols);
    constexpr bool Q_in_reg        = ggml_cuda_fattn_mma_get_Q_in_reg (DKQ, DV, ncols);
    constexpr int  nstages         = ggml_cuda_fattn_mma_get_nstages  (DKQ, DV, ncols1, ncols2);
    // LLAMA_FA_VKQ_F32_ACC: keep VKQ accumulator in fp32 for the SM120 NCOLS=64 buggy
    // MLA shape (DKQ=576, DV=512, ncols=64) where cumulative half2 accumulation
    // saturates over 50+ MLA layers. All other shapes keep the existing half2 path.
#if defined(LLAMA_FA_VKQ_F32_ACC) && defined(TURING_MMA_AVAILABLE)
    constexpr bool use_f32_acc = (DKQ == 576 && DV == 512 && ncols == 64 && cols_per_warp != 8);
#else
    constexpr bool use_f32_acc = false;
#endif

    constexpr int stride_tile_Q = DKQ/2     + 4;
    constexpr int stride_tile_K = nbatch_K2 + 4;

    constexpr int stride_tile_V = V_is_K_view ? stride_tile_K : nbatch_V2 + 4;

    const int k_VKQ_0 = kb0 * nbatch_fa;
#if defined(TURING_MMA_AVAILABLE)
    T_C_KQ KQ_C[nbatch_fa/(np*(cols_per_warp == 8 ? T_C_KQ::I : T_C_KQ::J))];
#elif defined(AMD_WMMA_AVAILABLE)
    T_C_KQ KQ_C[nbatch_fa/(np*T_C_KQ::J)];
#else // Volta
    T_C_KQ KQ_C[nbatch_fa/(np*T_C_KQ::J)];
#endif // defined(TURING_MMA_AVAILABLE)

    if constexpr (nstages > 1) {
        static_assert(!oob_check, "OOB check incompatible with multi-stage pipeline");
        static_assert(!V_is_K_view, "K data reuse not implemented multi-stage loading");
        static_assert(nbatch_K2 == DKQ/2, "batching not implemented for multi stage loading");
        static_assert(!sparse, "sparse mode incompatible with multi-stage pipeline");
        constexpr bool use_cp_async = true;
        cp_async_wait_all();
        __syncthreads();
        flash_attn_ext_f16_load_tile<stride_tile_V, nwarps, nbatch_fa, use_cp_async, oob_check>
            (V_h2 + int64_t(k_VKQ_0)*stride_V, tile_V, nbatch_V2, stride_V, k_VKQ_sup);
    } else {
        constexpr bool use_cp_async = nstages == 1;
        if (mask_h) { // NOMASK(2026-05-30): null mask = direct_causal inline-m path; skip load (avoid null deref + free O(ub*n_kv) mask). Behavior-identical when mask present.
            flash_attn_ext_f16_load_mask<ncols1, nwarps, nbatch_fa, use_cp_async, oob_check>
                (mask_h + k_VKQ_0, tile_mask, stride_mask, k_VKQ_sup, jt*ncols1, ne01);
        }
    }

    // For MLA K and V have the same data.
    // Therefore, iterate over K in reverse and later re-use the data if possible.
#pragma unroll
    for (int k0_start = (DKQ/2-1) - (DKQ/2-1) % nbatch_K2; k0_start >= 0; k0_start -= nbatch_K2) {
        const int k0_stop = k0_start + nbatch_K2 < DKQ/2 ? k0_start + nbatch_K2 : DKQ/2;
        const int k0_diff = k0_stop - k0_start;

        if constexpr (nstages <= 1) {
            if constexpr (sparse) {
                // Sparse gather: load K tile rows via indices[k_VKQ_0 + i], column slice [k0_start, k0_start+k0_diff).
                // All nwarps*WARP_SIZE threads cooperate over nbatch_fa * k0_diff half2 elements.
                {
                    const int tid           = threadIdx.y * WARP_SIZE + threadIdx.x;
                    const int total_threads = nwarps * WARP_SIZE;
                    const int total_elems   = nbatch_fa * k0_diff;
                    for (int elem = tid; elem < total_elems; elem += total_threads) {
                        const int i  = elem / k0_diff;
                        const int kk = elem % k0_diff; // column within this batch
                        // GLM_TOPK_GATHER_MASK_V1 preserves invalid slots as -1.
                        // Zero-fill them without forming or dereferencing a row-0 pointer.
                        if (!oob_check || i < k_VKQ_sup) {
                            const int phys = __ldg(indices + k_VKQ_0 + i);
                            tile_K[i * stride_tile_K + kk] = phys >= 0 ?
                                K_h2[int64_t(phys) * stride_K + k0_start + kk] :
                                make_half2(0.0f, 0.0f);
                        } else {
                            tile_K[i * stride_tile_K + kk] = make_half2(0.0f, 0.0f);
                        }
                    }
                }
                __syncthreads();
            } else {
                constexpr bool use_cp_async = nstages == 1;
                flash_attn_ext_f16_load_tile<stride_tile_K, nwarps, nbatch_fa, use_cp_async, oob_check>
                    (K_h2 + int64_t(k_VKQ_0)*stride_K + k0_start, tile_K, k0_diff, stride_K, k_VKQ_sup);
                if (use_cp_async) {
                    cp_async_wait_all();
                }
                __syncthreads();
            }
        }

        // Calculate tile of KQ:
        if constexpr (Q_in_reg) {
#pragma unroll
            for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
                const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*T_A_KQ::I;
#pragma unroll
                for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += T_A_KQ::J) {
                    T_A_KQ K_A;
                    load_ldmatrix(K_A, tile_K + i_KQ_0*stride_tile_K + (k_KQ_0 - k0_start), stride_tile_K);
                    if constexpr (cols_per_warp == 8) {
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[k_KQ_0/T_A_KQ::J]);
                    } else {
                        // Wide version of KQ_C is column-major
#if defined(AMD_WMMA_AVAILABLE)
                        // RDNA matrix C is column-major.
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[k_KQ_0/T_A_KQ::J]);
#else
                        // swap A and B for CUDA.
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], Q_B[k_KQ_0/T_A_KQ::J], K_A);
#endif // defined(AMD_WMMA_AVAILABLE)
                    }
                }
            }
        } else {
#pragma unroll
            for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += T_A_KQ::J) {
                load_ldmatrix(Q_B[0], tile_Q + (threadIdx.y / np)*(T_B_KQ::I*stride_tile_Q) + k_KQ_0, stride_tile_Q);

#pragma unroll
                for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
                    const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*T_A_KQ::I;

                    T_A_KQ K_A;
                    load_ldmatrix(K_A, tile_K + i_KQ_0*stride_tile_K + (k_KQ_0 - k0_start), stride_tile_K);

                    if constexpr (cols_per_warp == 8) {
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[0]);
                    } else {
                        // Wide version of KQ_C is column-major
#if defined(AMD_WMMA_AVAILABLE)
                        // RDNA matrix C is column-major.
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[0]);
#else
                        // swap A and B for CUDA.
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], Q_B[0], K_A);
#endif // defined(AMD_WMMA_AVAILABLE)
                    }
                }
            }
        }

        if constexpr (nstages <= 1) {
            __syncthreads(); // Only needed if tile_K == tile_V.
        }
    }

    if (use_logit_softcap) {
        constexpr int stride = cols_per_warp == 8 ? np*T_C_KQ::I : np*T_C_KQ::J;
        static_assert(nbatch_fa % stride == 0, "bad loop size");
#pragma unroll
        for (int i = 0; i < nbatch_fa/stride; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                KQ_C[i].x[l] = logit_softcap*tanhf(KQ_C[i].x[l]);
            }
        }
    }

    float KQ_max_new[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max_new[col] = KQ_max[col];
    }
    float KQ_rowsum_add[cols_per_thread] = {0.0f};

    if constexpr (cols_per_warp == 8) {
        if (ncols2 > 1 || mask_h) {
#ifdef KIMI_KVMAX_DEBUG
            // DECISIVE: this IS the live MLA NCOLS=32 mask-apply (cols_per_warp==8 branch; the
            // else-722 branch did not link => not instantiated). Print query row 0's mask for KV
            // cells 0,1,2 on the FIRST chunk (k_VKQ_0==0). q0_k1: -inf => mask correct in-kernel
            // (bug is KV-iter/softmax); 0 => mask-load bug (kernel sees cell1 unmasked for pos0).
            if (k_VKQ_0 == 0 && blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
                static __device__ unsigned int kimi_mp8_calls = 0;
                unsigned int kc = atomicAdd(&kimi_mp8_calls, 1u);
                if (kc < 16u) {
                    const half * mr0 = tile_mask + 0*(nbatch_fa + 8);
                    printf("KIMI_MP8 call=%u kVKQ0=%d q0_k0=%g q0_k1=%g q0_k2=%g q0_k3=%g nbatch_fa=%d ncols2=%d np=%d\n",
                        kc, (int) k_VKQ_0,
                        (double) __half2float(mr0[0]), (double) __half2float(mr0[1]),
                        (double) __half2float(mr0[2]), (double) __half2float(mr0[3]),
                        (int) nbatch_fa, (int) ncols2, (int) np);
                }
            }
#endif
#pragma unroll
            for (int i00 = 0; i00 < nbatch_fa; i00 += np*T_C_KQ::I) {
                const int i0 = i00 + (threadIdx.y % np)*T_C_KQ::I;
#pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    const int i = i0 + T_C_KQ::get_i(l);
                    const int j = ((threadIdx.y / np)*T_C_KQ::J + T_C_KQ::get_j(l)) / ncols2;

                    if (direct_causal) {
                        // LLAMA_FA_MLA_CAUSAL_DIRECT: compute the causal mask INLINE from positions
                        // instead of reading the materialized f16 tile_mask. Must match
                        // ggml-cuda.cu kernel_causal_mask bit-for-bit:
                        //   mask = (j_key < n_kv_visible && j_key <= pos_base + i_query) ? 0 : -INF
                        // abs_key = k_VKQ_0 + i (k_VKQ_0 = this K-tile base abs index, i = key in tile).
                        // i_query = jt*ncols1 + j (j = query row in tile). pos_base = n_past_causal.
                        // n_kv_visible = ne11 (per-ubatch visible KV count in the kernel's frame).
                        const int abs_key = k_VKQ_0 + i;
                        const int q_row   = jt*ncols1 + j;
                        const float m = (abs_key < ne11 && abs_key <= n_past_causal + q_row) ? 0.0f : -INFINITY;
                        KQ_C[i00/(np*T_C_KQ::I)].x[l] += slope * m;
                    } else {
                        KQ_C[i00/(np*T_C_KQ::I)].x[l] += slope * __half2float(tile_mask[j*(nbatch_fa + 8) + i]);
                    }
                }
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(nbatch_fa % (np*T_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::I) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = l % 2;
#endif // defined(AMD_WMMA_AVAILABLE)
                    KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[k0/(np*T_C_KQ::I)].x[l] + FATTN_KQ_MAX_OFFSET);
                }
            }
        }

        // Values per KQ column are spread across 8 threads:
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = 16; offset >= 4; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, WARP_SIZE));
            }
        }

        static_assert(nbatch_fa % (np*T_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::I) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = l % 2;
#endif // defined(AMD_WMMA_AVAILABLE)
                    KQ_C[k0/(np*T_C_KQ::I)].x[l] = expf(KQ_C[k0/(np*T_C_KQ::I)].x[l] - KQ_max_new[KQ_idx]);
                    KQ_rowsum_add[KQ_idx] += KQ_C[k0/(np*T_C_KQ::I)].x[l];
                } else {
                    KQ_C[k0/(np*T_C_KQ::I)].x[l] = 0.0f;
                }
            }
        }
    } else { // not Turing mma or T_B_KQ::I > 8
        if (ncols2 > 1 || mask_h) {
#ifdef KIMI_KVMAX_DEBUG
            // DECISIVE causal-mask probe for global query position 0 (MLA NCOLS=32 path:
            // ncols=ncols1*ncols2=2*16=32 -> mma_tile_sizes<32> -> T_B_KQ=tile<16,8,half2>
            // -> cols_per_warp=T_B_KQ::I=16 != 8, so execution lands in THIS else-branch,
            // which is exactly where tile_mask is ADDED into the KQ scores (lines below)
            // BEFORE the exp/softmax. This is the KQ+mask-apply phase inside
            // flash_attn_ext_f16_iter (called per KV chunk), NOT the combine/writeback
            // phase where tile_mask is later overwritten by tile_Q reuse.
            //
            // tile_mask shared-memory layout (from flash_attn_ext_f16_load_mask, non-cp_async
            // store at line ~424/457): tile_mask[j_sram*(nbatch_fa + 8) + i], where j_sram is
            // the query-token-in-tile [0..ncols1-1] and i is the KV index [0..nbatch_fa-1]
            // within this FA chunk. jt==0 => token tile 0 => j_sram==0 is GLOBAL query pos 0;
            // kb0==0 => k_VKQ_0==0 => i==1 is KV cell 1. One thread reads the first row directly.
            // CORRECT causal: q0_mask_k0=0, q0_mask_k1=-inf (or large negative); pos0 attends
            // ONLY to K[0]. BUG (over-attention): q0_mask_k1=0 => pos0 sums unmasked future K[1].
            if (threadIdx.x == 0 && threadIdx.y == 0 && jt == 0 && kb0 == 0) {
                static __device__ unsigned int kimi_mask_calls = 0;
                unsigned int c = atomicAdd(&kimi_mask_calls, 1u);
                if (c < 16u) {
                    const half * mrow0 = tile_mask + 0*(nbatch_fa + 8); // query row j_sram==0
                    printf("KIMI_MASKAPPLY call=%u q0_mask_k0=%g q0_mask_k1=%g q0_mask_k2=%g q0_mask_k3=%g q0_mask_k4=%g slope=%g ncols1=%d ncols2=%d cols_per_warp=%d nbatch_fa=%d np=%d k_VKQ_sup=%d\n",
                           c,
                           (double) __half2float(mrow0[0]),
                           (double) __half2float(mrow0[1]),
                           (double) __half2float(mrow0[2]),
                           (double) __half2float(mrow0[3]),
                           (double) __half2float(mrow0[4]),
                           (double) slope,
                           (int) ncols1, (int) ncols2, (int) cols_per_warp,
                           (int) nbatch_fa, (int) np, (int) k_VKQ_sup);
                }
            }
#endif
#pragma unroll
            for (int i00 = 0; i00 < nbatch_fa; i00 += np*T_C_KQ::J) {
                const int i0 = i00 + (threadIdx.y % np)*T_C_KQ::J;
#pragma unroll
                for (int l0 = 0; l0 < T_C_KQ::ne; l0 += 2) {
                    const int i = (i0 + T_C_KQ::get_j(l0)) / 2;
                    const int j = ((threadIdx.y / np)*cols_per_warp + T_C_KQ::get_i(l0)) / ncols2;

                    if (direct_causal) {
                        // LLAMA_FA_MLA_CAUSAL_DIRECT: inline causal mask (see site-1 comment).
                        // This half2 tile_mask access packs two adjacent keys: element l0+0 is
                        // KV cell 2*i, element l0+1 is KV cell 2*i+1 (both within this K-tile).
                        // abs_key = k_VKQ_0 + 2*i (+1 for the odd half). q_row = jt*ncols1 + j.
                        // n_kv_visible = ne11; pos_base = n_past_causal. Matches kernel_causal_mask.
                        const int q_row    = jt*ncols1 + j;
                        const int abs_key0 = k_VKQ_0 + 2*i;
                        const int abs_key1 = abs_key0 + 1;
                        const float m0 = (abs_key0 < ne11 && abs_key0 <= n_past_causal + q_row) ? 0.0f : -INFINITY;
                        const float m1 = (abs_key1 < ne11 && abs_key1 <= n_past_causal + q_row) ? 0.0f : -INFINITY;
                        KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 0] += slope*m0;
                        KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 1] += slope*m1;
                    } else {
                        const float2 tmp = __half22float2(((const half2 *)tile_mask)[j*(nbatch_fa/2 + 4) + i]);
                        KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 0] += slope*tmp.x;
                        KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 1] += slope*tmp.y;
                    }
                }
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = (l/2) % 2;
#endif // defined(AMD_WMMA_AVAILABLE)
                    KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[(k0/(np*T_C_KQ::J))].x[l] + FATTN_KQ_MAX_OFFSET);
                }
            }
        }

#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#if defined(TURING_MMA_AVAILABLE)
            // Values per KQ column are spread across 4 threads:
            constexpr int offset_first = 2;
            constexpr int offset_last  = 1;
#elif defined(AMD_WMMA_AVAILABLE)
            // Values per KQ column are spread across 2 threads:
            constexpr int offset_first = 16;
            constexpr int offset_last  = 16;
#else // Volta
            // Values per KQ column are spread across 2 threads:
            constexpr int offset_first = 2;
            constexpr int offset_last  = 2;
#endif // defined(TURING_MMA_AVAILABLE)
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, WARP_SIZE));
            }
        }

        static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = (l/2) % 2;
#endif // defined(AMD_WMMA_AVAILABLE)
                    KQ_C[(k0/(np*T_C_KQ::J))].x[l] = expf(KQ_C[(k0/(np*T_C_KQ::J))].x[l] - KQ_max_new[KQ_idx]);
                    KQ_rowsum_add[KQ_idx] += KQ_C[(k0/(np*T_C_KQ::J))].x[l];
                } else {
                    KQ_C[(k0/(np*T_C_KQ::J))].x[l] = 0.0f;
                }
            }
        }
    }

    {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const float KQ_max_diff = KQ_max[col] - KQ_max_new[col];
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new[col];

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_rowsum_add[col];
        }

#if defined(TURING_MMA_AVAILABLE)
        if constexpr (cols_per_warp == 8) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[cols_per_thread - 1]);
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::I; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                        VKQ_C[i].x[l0 + col] *= KQ_max_scale_h2;
                    }
                }
            }
            if constexpr (use_f32_acc) {
                // FIX 2026-05-13: rescale per-col-parity within each N-slice tile.
                // For tile<16,8,float>: thread holds 4 floats at (row 0/8, col 0/1).
                // Even col positions (l%2==0) use KQ_max_scale[0]; odd use KQ_max_scale[1].
                const int n_acc_pairs = (DV/2)/T_C_VKQ::J;
                const float s_even = KQ_max_scale[0];
                const float s_odd  = KQ_max_scale[1];
#pragma unroll
                for (int slice = 0; slice < 2; ++slice) {  // 0 = N=0..7, 1 = N=8..15
#pragma unroll
                    for (int i = 0; i < n_acc_pairs; ++i) {
#pragma unroll
                        for (int l = 0; l < tile<16, 8, float>::ne; ++l) {
                            const float s = (l % 2 == 0) ? s_even : s_odd;
                            VKQ_C_f32[2*i + slice].x[l] *= s;
                        }
                    }
                }
            }
        }
#elif defined(AMD_WMMA_AVAILABLE)
        const half2 KQ_max_scale_h2 = make_half2(
            KQ_max_scale[0], KQ_max_scale[0]);
#pragma unroll
        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }
#else // Volta
        const half2 KQ_max_scale_h2 = make_half2(
            KQ_max_scale[(threadIdx.x / 2) % 2], KQ_max_scale[(threadIdx.x / 2) % 2]);
#pragma unroll
        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }
#endif // defined(TURING_MMA_AVAILABLE)
    }

    // Convert KQ C tiles into B tiles for VKQ calculation:
    T_B_VKQ B[nbatch_fa/(np*2*T_B_VKQ::J)];
    static_assert(nbatch_fa % (np*2*T_B_VKQ::J) == 0, "bad loop size");
    if constexpr (cols_per_warp == 8) {
#pragma unroll
        for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
            B[k] = get_transposed(get_half2(KQ_C[k]));
        }
    } else {
        for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
            B[k] = get_half2(KQ_C[k]);
        }
    }

    if constexpr (nstages > 1) {
        static_assert(!V_is_K_view, "K data reuse not implemented multi-stage loading");
        static_assert(!sparse, "sparse mode incompatible with multi-stage pipeline (site 3)");
        // Preload K tile for next iteration:
        constexpr bool use_cp_async = true;
        cp_async_wait_all();
        __syncthreads();
        if (!last_iter) {
            if (mask_h) { // NOMASK(2026-05-30): null mask = direct_causal inline-m path; skip load. Behavior-identical when mask present.
                flash_attn_ext_f16_load_mask<ncols1, nwarps, nbatch_fa, use_cp_async, oob_check>
                    (mask_h + k_VKQ_0 + nbatch_fa, tile_mask, stride_mask, k_VKQ_sup, jt*ncols1, ne01);
            }
            flash_attn_ext_f16_load_tile<stride_tile_K, nwarps, nbatch_fa, use_cp_async, oob_check>
                (K_h2 + int64_t(k_VKQ_0 + nbatch_fa)*stride_K, tile_K, nbatch_K2, stride_K, k_VKQ_sup);
        }
    }


#if defined(AMD_WMMA_AVAILABLE) && !defined(LDMATRIX_TRANS_AVAILABLE)
    T_A_VKQ A_identity;
    make_identity_mat(A_identity);
#endif // defined(AMD_WMMA_AVAILABLE) && !defined(LDMATRIX_TRANS_AVAILABLE)

    // Calculate VKQ tile, need to use logical rather than physical elements for i0 due to transposition of V:
#pragma unroll
    for (int i0_start = 0; i0_start < DV; i0_start += 2*nbatch_V2) {
        static_assert(DV % (2*nbatch_V2) == 0, "bad loop size");
        const int i0_stop = i0_start + 2*nbatch_V2;
        const int i0_diff = i0_stop - i0_start;

        if constexpr (nstages <= 1) {
            if (!V_is_K_view || i0_stop > 2*nbatch_K2) {
                if constexpr (sparse) {
                    // Sparse gather: load V tile rows via indices[k_VKQ_0 + i], column slice [i0_start/2, i0_start/2 + i0_diff/2).
                    {
                        const int col_start     = i0_start / 2;
                        const int col_count     = i0_diff  / 2;
                        const int tid           = threadIdx.y * WARP_SIZE + threadIdx.x;
                        const int total_threads = nwarps * WARP_SIZE;
                        const int total_elems   = nbatch_fa * col_count;
                        for (int elem = tid; elem < total_elems; elem += total_threads) {
                            const int i  = elem / col_count;
                            const int kk = elem % col_count;
                            // GLM_TOPK_GATHER_MASK_V1: -1 slots do not gather V row 0.
                            if (!oob_check || i < k_VKQ_sup) {
                                const int phys = __ldg(indices + k_VKQ_0 + i);
                                tile_V[i * stride_tile_V + kk] = phys >= 0 ?
                                    V_h2[int64_t(phys) * stride_V + col_start + kk] :
                                    make_half2(0.0f, 0.0f);
                            } else {
                                tile_V[i * stride_tile_V + kk] = make_half2(0.0f, 0.0f);
                            }
                        }
                    }
                    __syncthreads();
                } else {
                    constexpr bool use_cp_async = nstages == 1;
                    flash_attn_ext_f16_load_tile<stride_tile_V, nwarps, nbatch_fa, use_cp_async, oob_check>
                        (V_h2 + int64_t(k_VKQ_0)*stride_V + i0_start/2, tile_V, i0_diff/2, stride_V, k_VKQ_sup);
                    if (use_cp_async) {
                        cp_async_wait_all();
                    }
                    __syncthreads();
                }
            }
        }
        const half2 * tile_V_i = !V_is_K_view || i0_stop > 2*nbatch_K2 ? tile_V : tile_V + i0_start/2;

#if defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        constexpr int i0_stride = cols_per_warp == 8 ? T_C_VKQ::I : 2*T_C_VKQ::J;
#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += i0_stride) {
            static_assert((nbatch_fa/2) % (np*T_A_VKQ::J) == 0, "bad loop size");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa/2; k00 += np*T_A_VKQ::J) {
                const int k0 = k00 + (threadIdx.y % np)*T_A_VKQ::J;

                T_A_VKQ A; // Transposed in SRAM but not in registers, gets transposed on load.
#if defined(LDMATRIX_TRANS_AVAILABLE)
                load_ldmatrix_trans(A, tile_V_i + 2*k0*stride_tile_V + (i_VKQ_0 - i0_start)/2, stride_tile_V);
#else
                // TODO: Try to transpose tile_V when loading gmem to smem.
                // Use mma to transpose T_A_VKQ for RDNA.
                T_A_VKQ A_trans;
                load_ldmatrix(A_trans, tile_V_i + 2*k0*stride_tile_V + (i_VKQ_0 - i0_start)/2, stride_tile_V);
                mma(A, A_trans, A_identity);
#endif // defined(TURING_MMA_AVAILABLE)
                if constexpr (T_B_KQ::I == 8) {
                    mma(VKQ_C[i_VKQ_0/i0_stride], A, B[k00/(np*T_A_VKQ::J)]);
                } else {
                    // Wide version of VKQ_C is column-major.
#if defined(AMD_WMMA_AVAILABLE)
                    // RDNA matrix C is column-major.
                    mma(VKQ_C[i_VKQ_0/i0_stride], A, B[k00/(np*T_A_VKQ::J)]);
#else
                    // swap A and B for CUDA.
                    mma(VKQ_C[i_VKQ_0/i0_stride], B[k00/(np*T_A_VKQ::J)], A);
                    if constexpr (use_f32_acc) {
                        // FIX 2026-05-13: drop the broken tile<8,8,half2> B-packing.
                        // Issue raw m16n8k16.f32 PTX twice using the SAME register split
                        // the half2 wrapper at mma.cuh:935-940 uses for its two MMAs:
                        //   half2 MMA 1: D.x[0..1] from PTX-B = (V.x[0], V.x[2])  → covers N=0..7
                        //   half2 MMA 2: D.x[2..3] from PTX-B = (V.x[1], V.x[3])  → covers N=8..15
                        // For our scope: code calls mma(D, B_S, A_V) (swap), so PTX-A=B and PTX-B=A.
                        const int * Axi = (const int *) B[k00/(np*T_A_VKQ::J)].x;  // PTX-A = S
                        const int * Bxi = (const int *) A.x;                       // PTX-B = V
                        const int idx = i_VKQ_0/i0_stride;
                        int * D0xi = (int *) VKQ_C_f32[2*idx + 0].x;  // N=0..7
                        int * D1xi = (int *) VKQ_C_f32[2*idx + 1].x;  // N=8..15
                        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+r"(D0xi[0]), "+r"(D0xi[1]), "+r"(D0xi[2]), "+r"(D0xi[3])
                            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]),
                              "r"(Bxi[0]), "r"(Bxi[2]));
                        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+r"(D1xi[0]), "+r"(D1xi[1]), "+r"(D1xi[2]), "+r"(D1xi[3])
                            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]),
                              "r"(Bxi[1]), "r"(Bxi[3]));
                    }
#endif // defined(AMD_WMMA_AVAILABLE)
                }
            }
        }
#else // Volta
        constexpr int i0_stride = 2*T_C_VKQ::J;
#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += i0_stride) {
            static_assert(nbatch_fa % (np*T_A_VKQ::I) == 0, "bad loop size");
            static_assert(2*T_B_VKQ::J == T_A_VKQ::I, "bad tile sizes");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa; k00 += np*T_A_VKQ::I) {
                const int k0 = k00 + (threadIdx.y % np)*T_A_VKQ::I;

                T_A_VKQ A; // Transposed in both SRAM and registers, load normally.
                load_ldmatrix(A, tile_V_i + k0*stride_tile_V + (i_VKQ_0 - i0_start)/2, stride_tile_V);
                mma(VKQ_C[i_VKQ_0/i0_stride], B[k00/(np*T_A_VKQ::I)], A);
            }
        }
#endif // defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

        if constexpr (nstages <= 1) {
            __syncthreads(); // Only needed if tile_K == tile_V.
        }
    }
#else
    GGML_UNUSED_VARS(Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup,
        scale, slope, logit_softcap, ne01, ne02,
        stride_K, stride_V, stride_mask,
        tile_Q, tile_K, tile_V, tile_mask,
        Q_B, VKQ_C, KQ_max, KQ_rowsum, kb0);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA4))
}

#if defined(TURING_MMA_AVAILABLE)
template<int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  8, half2>; // column-major
};
template<> struct mma_tile_sizes<8> {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile< 8,  8, half2>; // column-major
    using T_C_KQ  = tile<16,  8, float>; // row-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile< 8,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  4, half2>; // row-major
};
#elif defined(AMD_WMMA_AVAILABLE)
template<int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  8, half2>; // column-major
};
#else // Volta
template<int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile< 8,  4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<32,  4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_C_KQ  = tile<32,  8, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile< 8,  4, half2, DATA_LAYOUT_J_MAJOR_MIRRORED>; // column-major
    using T_B_VKQ = tile<32,  4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_C_VKQ = tile<32,  4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
};
#endif // defined(TURING_MMA_AVAILABLE)

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps, bool use_logit_softcap, bool V_is_K_view, bool needs_fixup, bool is_fixup, bool sparse>
static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half   * const __restrict__ mask_h,
        const int    * const __restrict__ indices,
        const float  * const __restrict__ sinks_f,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int gqa_ratio,
        const int ne11,
        const int stride_Q1,
        const int stride_Q2,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        const bool direct_causal,
        const int jt,
        const int zt_gqa,
        const int kb0_start,
        const int kb0_stop,
        const int n_past_causal) {
    // LLAMA_FA_MLA_CAUSAL_DIRECT: direct_causal + n_past_causal are now forwarded into
    // flash_attn_ext_f16_iter, which computes the causal mask INLINE from positions at the
    // two KQ mask-add sites (replacing the materialized tile_mask read). Gated entirely by
    // direct_causal so the default production path (direct_causal==false) is byte-untouched.
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA4))
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int ncols = ncols1 * ncols2;
    using     T_A_KQ    = typename mma_tile_sizes<ncols>::T_A_KQ;
    using     T_B_KQ    = typename mma_tile_sizes<ncols>::T_B_KQ;
    using     T_C_KQ    = typename mma_tile_sizes<ncols>::T_C_KQ;
    using     T_A_VKQ   = typename mma_tile_sizes<ncols>::T_A_VKQ;
    using     T_B_VKQ   = typename mma_tile_sizes<ncols>::T_B_VKQ;
    using     T_C_VKQ   = typename mma_tile_sizes<ncols>::T_C_VKQ;

    constexpr int  cols_per_warp   = T_B_KQ::I;
    constexpr int  cols_per_thread = get_cols_per_thread();
    constexpr int  np              = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols; // Number of parallel CUDA warps per Q column.
    constexpr int  nbatch_fa       = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols);
    constexpr int  nbatch_K2       = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols);
    constexpr int  nbatch_V2       = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols);
    constexpr int  nbatch_combine  = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols);
    constexpr bool Q_in_reg        = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols);
    constexpr int  nstages         = ggml_cuda_fattn_mma_get_nstages       (DKQ, DV, ncols1, ncols2);

    if (cols_per_warp > ncols) {
        NO_DEVICE_CODE;
        return;
    }

    static_assert(nwarps * (cols_per_warp/ncols2) % ncols1 == 0, "bad nwarps");

    constexpr int stride_tile_Q = DKQ/2     + 4;
    constexpr int stride_tile_K = nbatch_K2 + 4;

    constexpr int stride_tile_V = V_is_K_view ? stride_tile_K : nbatch_V2 + 4;
    constexpr int stride_tile_KV_max = stride_tile_K > stride_tile_V ? stride_tile_K : stride_tile_V;

    extern __shared__ half2 tile_Q[];
    half2 * tile_K    = Q_in_reg              ? tile_Q                             : tile_Q + ncols     * stride_tile_Q;
    half2 * tile_V    =           nstages > 1 ? tile_K + nbatch_fa * stride_tile_K : tile_K;
    half  * tile_mask = (half *) (nstages > 1 ? tile_V + nbatch_fa * stride_tile_V : tile_V + nbatch_fa * stride_tile_KV_max);

    T_B_KQ    Q_B[(Q_in_reg ? DKQ/(2*T_B_KQ::J) : 1)];
#if defined(TURING_MMA_AVAILABLE)
    T_C_VKQ VKQ_C[cols_per_warp == 8 ? DV/T_C_VKQ::I : DV/(2*T_C_VKQ::J)];
#elif defined(AMD_WMMA_AVAILABLE)
    T_C_VKQ VKQ_C[                                     DV/(2*T_C_VKQ::J)];
#else // Volta
    T_C_VKQ VKQ_C[                                     DV/(2*T_C_VKQ::J)];
#endif // defined(TURING_MMA_AVAILABLE)
    // LLAMA_FA_VKQ_F32_ACC parallel fp32 accumulator (active only when use_f32_acc==true).
    // Two fp32 tiles per half2 tile (one per packed column 0/1). For other shapes this
    // array has size 1 and is unused.
#if defined(LLAMA_FA_VKQ_F32_ACC) && defined(TURING_MMA_AVAILABLE)
    constexpr bool use_f32_acc = (DKQ == 576 && DV == 512 && ncols == 64 && cols_per_warp != 8);
#else
    constexpr bool use_f32_acc = false;
#endif
#if defined(LLAMA_FA_VKQ_F32_ACC) && defined(TURING_MMA_AVAILABLE)
    constexpr int VKQ_C_count_f32 = use_f32_acc ? 2 * (DV/(2*T_C_VKQ::J)) : 1;
    tile<16, 8, float> VKQ_C_f32[VKQ_C_count_f32];
#pragma unroll
    for (int i = 0; i < VKQ_C_count_f32; ++i) {
#pragma unroll
        for (int l = 0; l < tile<16, 8, float>::ne; ++l) {
            VKQ_C_f32[i].x[l] = 0.0f;
        }
    }
#endif

    float KQ_rowsum[cols_per_thread] = {0.0f};
    float KQ_max[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -FLT_MAX/2.0f;
    }

    // Load Q data into tile_Q, either temporarily or permanently.
    // Q in registers is faster, but register pressure is the biggest bottleneck.
    // The loading is done with decreasing granularity for D for better memory bandwidth.
    const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
    for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
        const int k0_start  = stride_k == WARP_SIZE ? 0 : DKQ/2 - (DKQ/2) % (2*stride_k);
        const int k0_stop   =                             DKQ/2 - (DKQ/2) % (1*stride_k);
        const int stride_jc = WARP_SIZE / stride_k;

        if (k0_start == k0_stop) {
            continue;
        }

#pragma unroll
        for (int jc0 = 0; jc0 < ncols; jc0 += nwarps*stride_jc) {
            const int jc = jc0 + threadIdx.y*stride_jc + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

            if (jc0 + nwarps*stride_jc > ncols && jc >= ncols) {
                break;
            }

            const int j = jc / ncols2;
            const int c = jc % ncols2;

            if ((ncols1 == 1 || jt*ncols1 + j < int(ne01.z)) && (ncols2 == 1 || zt_gqa*ncols2 + c < gqa_ratio)) {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    const float2 tmp = Q_f2[(jt*ncols1 + j)*stride_Q1 + c*stride_Q2 + k];
                    tile_Q[jc*stride_tile_Q + k] = scale_h2 * make_half2(tmp.x, tmp.y);
                }
            } else {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    tile_Q[jc*stride_tile_Q + k] = make_half2(0.0f, 0.0f);
                }
            }
        }
    }

    __syncthreads();

    if (Q_in_reg) {
        const int j0 = (threadIdx.y / np) * cols_per_warp;

#pragma unroll
        for (int k0 = 0; k0 < DKQ/2; k0 += T_B_KQ::J) {
            load_ldmatrix(Q_B[k0/T_B_KQ::J], tile_Q + j0*stride_tile_Q + k0, stride_tile_Q);
        }
    }

    __syncthreads();

    int kb0 = kb0_start;

    // CAUSAL BLOCK-SKIP (2026-05-30, direct_causal / LLAMA_FA_MLA_CAUSAL_DIRECT): cap the K-loop to
    // the tile's causal boundary so the iterated block COUNT is independent of ne11 (the padded
    // cache width). PROVEN root cause of multi-ub incoherence: identical K but different ne11 =>
    // iterating a different number of fully-masked far blocks perturbs the online-softmax by ~1 ULP
    // => amplified by the deep MoE net => garbage. Capping at the causal boundary makes multi-ub and
    // single-ub iterate IDENTICAL blocks per query => bit-identical => coherent. The causal mask is
    // monotonic (0..0,-inf..-inf) so binary-search the tile's MAX query row for the 0->-inf boundary.
    // NOTE: safe only when one block owns the whole tile's K-range (kb0_start==0, i.e. stream-K off);
    // gated by direct_causal so the default production path is untouched.
    // CAUSAL BLOCK-SKIP via POSITION ARITHMETIC (race-free — no mask read; the mask-derived version
    // raced the async f16-mask-refresh D2D and read NaN). causal_stop = ceil((n_past + r_max + 1)/nbatch_fa).
    // VALIDATION STAGE: n_past hardcoded 0 (CORRECT for ubatch u0, whose queries start at global 0;
    // WRONG for u1/u2 which need their global offset plumbed). Use this to validate u0's FA row ==
    // single-ub (proves the concept + arithmetic), then plumb n_past=pos_base for the full fix.
    int kb0_stop_eff = kb0_stop;
    // Causal block-skip (WIP, default-off). Gated on direct_causal (LLAMA_FA_MLA_CAUSAL_DIRECT) so
    // production is UNTOUCHED. NOTE: direct_causal selects a stub variant that NaNs; the working fix
    // needs a new kernel-arg n_past in the direct_causal_env=false variant (see plan doc). The
    // validation (temporary ne11>ne01.z gate) confirmed: working-variant cap is non-NaN; short-causal
    // rows match single-ub (~1e-6); long-causal rows differ (cap removes masked-block perturbation =>
    // inconclusive vs uncapped ref; needs e2e + per-ubatch n_past to validate coherence).
    // n_past_causal >= 0 => causal-skip enabled (host set op_params[3]=pos_base+1 under
    // LLAMA_FA_CAUSAL_SKIP). n_past_causal = the ubatch's global query offset. Race-free (no mask read).
    // Only when this block owns the whole tile K-range (kb0_start==0, stream-K off).
    if (n_past_causal >= 0 && kb0_start == 0) {
        const int n_past = n_past_causal;
        const int r_max = min((int)(jt*ncols1 + ncols1 - 1), (int)ne01.z - 1);
        if (r_max >= 0) {
            const int causal_stop = (n_past + r_max + nbatch_fa) / nbatch_fa; // ceil((n_past+r_max+1)/nbatch_fa)
            if (causal_stop < kb0_stop_eff) kb0_stop_eff = causal_stop;
            if (kb0_stop_eff < kb0_start + 1) kb0_stop_eff = kb0_start + 1;
            if (threadIdx.x == 0 && threadIdx.y == 0 && jt < 3) {
                static __device__ unsigned int csk_calls = 0;
                unsigned int c = atomicAdd(&csk_calls, 1u);
                if (c < 12u) printf("CAUSAL_SKIP_KERN jt=%d n_past=%d r_max=%d ne11=%d nbatch_fa=%d kb0_start=%d kb0_stop=%d->eff=%d\n",
                    (int)jt, n_past_causal, r_max, ne11, (int)nbatch_fa, kb0_start, kb0_stop, kb0_stop_eff);
            }
        }
    }

    // Preload mask and K data for first iteration when using cp_async with multiple stages:
    if constexpr (nstages > 1) {
        static_assert(nbatch_K2 == DKQ/2, "batching not implemented for multi-stage pipeline");
        constexpr bool use_cp_async = true;
        constexpr bool oob_check    = false;
        constexpr int  k_VKQ_sup    = nbatch_fa;
        if (mask_h) { // NOMASK(2026-05-30): null mask = direct_causal inline-m path; skip load. Behavior-identical when mask present.
            flash_attn_ext_f16_load_mask<ncols1, nwarps, nbatch_fa, use_cp_async, oob_check>
                (mask_h + kb0*nbatch_fa, tile_mask, stride_mask, k_VKQ_sup, jt*ncols1, ne01);
        }
        flash_attn_ext_f16_load_tile<stride_tile_K, nwarps, nbatch_fa, use_cp_async, oob_check>
            (K_h2 + int64_t(kb0)*nbatch_fa*stride_K, tile_K, nbatch_K2, stride_K, k_VKQ_sup);
    }

    // kb0_start is always < kb0_stop so the last iter can be executed unconditionally.
#if defined(LLAMA_FA_VKQ_F32_ACC) && defined(TURING_MMA_AVAILABLE)
    tile<16, 8, float> * const VKQ_C_f32_ptr = VKQ_C_f32;
#else
    tile<16, 8, float> * const VKQ_C_f32_ptr = nullptr;
#endif
    if constexpr (ncols2 == 1) {
        constexpr bool oob_check = true;
        for (; kb0 < kb0_stop_eff-1; ++kb0) {
            constexpr bool last_iter = false;
            constexpr int  k_VKQ_sup = nbatch_fa;
            flash_attn_ext_f16_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, needs_fixup, is_fixup, last_iter, oob_check, sparse,
                 T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
                (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, ne11, direct_causal, n_past_causal, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
                 VKQ_C_f32_ptr, KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup);
        }
        constexpr bool last_iter = true;
        const     int  k_VKQ_sup = ne11 - kb0*nbatch_fa;
        flash_attn_ext_f16_iter
            <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, needs_fixup, is_fixup, last_iter, oob_check, sparse,
              T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
            (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, ne11, direct_causal, n_past_causal, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
             VKQ_C_f32_ptr, KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup);
    } else {
        constexpr bool oob_check = false;
        for (; kb0 < kb0_stop_eff-1; ++kb0) {
            constexpr bool last_iter = false;
            constexpr int  k_VKQ_sup = nbatch_fa;
            flash_attn_ext_f16_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, needs_fixup, is_fixup, last_iter, oob_check, sparse,
                 T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
                (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, ne11, direct_causal, n_past_causal, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
                 VKQ_C_f32_ptr, KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup);
#ifdef KIMI_KVMAX_DEBUG
            // DECISIVE causal-mask probe (in process_tile, where KIMI_DENOM is proven to
            // link+fire). This is the LIVE ncols2>1 else-branch (selected because
            // ncols2==16 != 1 for the MLA NCOLS=32 instantiation DKQ=576/DV=512/ncols1=2/
            // ncols2=16). For this shape nstages==1, so flash_attn_ext_f16_iter loads
            // tile_mask for its OWN chunk at the START of the call (line ~541, use_cp_async)
            // and does NOT overwrite it at the end (the nstages>1 next-chunk preload at
            // ~927-942 is constexpr-dead). The K-load inside iter performs
            // cp_async_wait_all()+__syncthreads(), so after this iter call returns the
            // first chunk's mask is committed to shared memory and visible to thread(0,0).
            // Gated kb0==kb0_start so only the FIRST KV chunk's mask is read (kb0 is not yet
            // incremented at this point in the first loop pass). mask layout (cp_async store
            // at ~408): tile_mask[j_sram*(nbatch_fa+8) + i]; jt==0 => token tile 0 => row
            // j_sram==0 is GLOBAL query pos 0; i is KV cell index in this chunk (kb0==0 =>
            // cell i). CORRECT causal: q0_k0=0, q0_k1=-inf/large-neg (pos0 attends only to
            // K[0]). BUG (over-attention): q0_k1==0 => pos0 sums unmasked future K[1].
            if (threadIdx.x == 0 && threadIdx.y == 0 && jt == 0 && kb0 == kb0_start) {
                static __device__ unsigned int kimi_maskp_calls = 0;
                unsigned int c = atomicAdd(&kimi_maskp_calls, 1u);
                if (c < 16u) {
                    const half * mrow0 = tile_mask + 0*(nbatch_fa + 8);
                    printf("KIMI_MASKP call=%u jt=%d q0_k0=%g q0_k1=%g q0_k2=%g q0_k3=%g nbatch_fa=%d ncols1=%d ncols2=%d kb0_start=%d kb0_stop=%d\n",
                           c, (int)jt, (double)__half2float(mrow0[0]), (double)__half2float(mrow0[1]),
                           (double)__half2float(mrow0[2]), (double)__half2float(mrow0[3]),
                           (int)nbatch_fa, (int)ncols1, (int)ncols2, (int)kb0_start, (int)kb0_stop);
                }
            }
#endif
        }
        constexpr bool last_iter = true;
        constexpr int  k_VKQ_sup = nbatch_fa;
        flash_attn_ext_f16_iter
            <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, needs_fixup, is_fixup, last_iter, oob_check, sparse,
             T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
            (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, ne11, direct_causal, n_past_causal, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
             VKQ_C_f32_ptr, KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup);
#ifdef KIMI_KVMAX_DEBUG
        // Single-chunk case (kb0_stop - kb0_start == 1): the loop above did not run, so
        // this last-iter call IS the first iter and kb0 still == kb0_start. Same gate +
        // SAME atomic counter (kimi_maskp_calls) as the loop probe so the first executed
        // chunk prints at most once. If the loop ran (multi-chunk), kb0 > kb0_start here
        // and tile_mask holds a LATER chunk, so this is correctly skipped.
        if (threadIdx.x == 0 && threadIdx.y == 0 && jt == 0 && kb0 == kb0_start) {
            static __device__ unsigned int kimi_maskp_calls = 0;
            unsigned int c = atomicAdd(&kimi_maskp_calls, 1u);
            if (c < 16u) {
                const half * mrow0 = tile_mask + 0*(nbatch_fa + 8);
                printf("KIMI_MASKP call=%u jt=%d q0_k0=%g q0_k1=%g q0_k2=%g q0_k3=%g nbatch_fa=%d ncols1=%d ncols2=%d kb0_start=%d kb0_stop=%d\n",
                       c, (int)jt, (double)__half2float(mrow0[0]), (double)__half2float(mrow0[1]),
                       (double)__half2float(mrow0[2]), (double)__half2float(mrow0[3]),
                       (int)nbatch_fa, (int)ncols1, (int)ncols2, (int)kb0_start, (int)kb0_stop);
            }
        }
#endif
    }

    // With multi-stage loading there is no __syncthreads at the end of the iter,
    //     there can be a race condition on shared memory access for combining/writing back results.
    if constexpr (nstages > 1 && nwarps*cols_per_warp > nbatch_fa) {
        __syncthreads();
    }

    // Finally, sum up partial KQ rowsums.
    {
#if defined(TURING_MMA_AVAILABLE)
        // The partial sums are spread across 8/4 threads.
        constexpr int offset_first = cols_per_warp == 8 ? 16 : 2;
        constexpr int offset_last  = cols_per_warp == 8 ?  4 : 1;
#elif defined(AMD_WMMA_AVAILABLE)
        // The partial sums are spread across 2 threads.
        constexpr int offset_first = 16;
        constexpr int offset_last  = 16;
#else // Volta
        // The partial sums are spread across 2 threads.
        constexpr int offset_first = 2;
        constexpr int offset_last  = 2;
#endif // defined(TURING_MMA_AVAILABLE)
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[col], offset, WARP_SIZE);
            }
        }
    }

    // If attention sinks are used, potentially re-scale if KQ_max is small.
    // Also add the sink as a value to KQ_rowsum, this is done after synchonization of KQ_rowsum
    //     so it's being done unconditionally for every thread.
    if (!is_fixup && (np == 1 || threadIdx.y % np == 0) && sinks_f) {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            // Phase 2.2 angle 36 (2026-05-13): port ggml-org/llama.cpp upstream
            // fix — add (threadIdx.y/np)*cols_per_warp offset to jc index for
            // multi-warp configs (nwarps > 1). Without this, multi-warp
            // sinks_f indexing reads the wrong sink slot. Kimi K2.6 has no
            // sinks (sinks_f == nullptr in this branch is the common case)
            // so the bug doesn't fire for Kimi production. Ship it for
            // models that do have sinks (e.g. some Gemma variants).
            const int jc = (threadIdx.y/np)*cols_per_warp + (cols_per_warp == 8 ? T_C_KQ::get_j(col) : T_C_KQ::get_i(2*col));
            const float sink = sinks_f[jc % ncols2];

            const float KQ_max_new = fmaxf(KQ_max[col], sink);
            const float KQ_max_diff = KQ_max[col] - KQ_max_new;
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new;

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            const float KQ_max_add = expf(sink - KQ_max_new);
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_max_add;
        }

#if defined(TURING_MMA_AVAILABLE)
        if constexpr (cols_per_warp == 8) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[cols_per_thread - 1]);
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::I; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                        VKQ_C[i].x[l0 + col] *= KQ_max_scale_h2;
                    }
                }
            }
            if constexpr (use_f32_acc) {
                // FIX 2026-05-13: same per-col-parity rescale as the earlier site.
                const float s_even = KQ_max_scale[0];
                const float s_odd  = KQ_max_scale[1];
#pragma unroll
                for (int slice = 0; slice < 2; ++slice) {
#pragma unroll
                    for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                        for (int l = 0; l < tile<16, 8, float>::ne; ++l) {
                            const float s = (l % 2 == 0) ? s_even : s_odd;
                            VKQ_C_f32[2*i + slice].x[l] *= s;
                        }
                    }
                }
            }
        }
#elif defined(AMD_WMMA_AVAILABLE)
        const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[0]);
#pragma unroll
        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }
#else // Volta
        const int col = (threadIdx.x / 2) % 2;
        const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }
#endif // defined(TURING_MMA_AVAILABLE)
    }

    // Combine VKQ accumulator values if np > 1.
    // It's also faster to do small writes to shared memory, then large write to VRAM than to do small writes to VRAM.
    // So also write VKQ accumulators to shared memory in column-major format if np == 1.

    constexpr int tile_stride = nbatch_combine + 4;
    static_assert((DV/2) % nbatch_combine == 0, "bad nbatch_combine");

    if constexpr (cols_per_warp == 8) {
        const int jc_cwmo = (threadIdx.x % (2*T_C_VKQ::J)) / T_C_VKQ::J; // jc combine write meta offset
        const int jc_cwm = threadIdx.y*(2*T_C_VKQ::J) + 2*T_C_VKQ::get_j(-1) + jc_cwmo; // jc combine write meta
        const float2 KQ_cmr = make_float2(KQ_max[jc_cwmo], KQ_rowsum[jc_cwmo]); // KQ combine max rowsum

        if (((!needs_fixup && !is_fixup) || np > 1) && threadIdx.x < 2*T_C_VKQ::J) {
            // Use the 16 bytes of padding in each row to store the meta data: KQ max, KQ rowsum, KQ max scale.
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
            if (needs_fixup && threadIdx.x < T_B_KQ::I) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && threadIdx.x < T_B_KQ::I) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
        }
    } else {
        // jc_cwm = jc combine write meta
        // KQ_cmr = KQ combine max rowsum
        // Use the 16 bytes of padding in each Q column to store the meta data: KQ max, KQ rowsum, KQ max scale.
#if defined(TURING_MMA_AVAILABLE)
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(threadIdx.x % 4);
        const float2 KQ_cmr = make_float2(KQ_max[threadIdx.x % cols_per_thread], KQ_rowsum[threadIdx.x % cols_per_thread]);
        const bool thread_should_write = threadIdx.x % 4 < cols_per_thread;
#elif defined(AMD_WMMA_AVAILABLE)
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(0);
        const float2 KQ_cmr = make_float2(KQ_max[0], KQ_rowsum[0]);
        const bool thread_should_write = threadIdx.x / 16 < cols_per_thread;
#else // Volta
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_KQ::get_i(threadIdx.x & 2);
        const float2 KQ_cmr = make_float2(KQ_max[(threadIdx.x & 2) / 2], KQ_rowsum[(threadIdx.x & 2) / 2]);
        const bool thread_should_write = T_C_KQ::J == 8 || T_C_KQ::get_j(threadIdx.x & 2) < 8;
#endif // defined(TURING_MMA_AVAILABLE)

        if (((!needs_fixup && !is_fixup) || np > 1) && thread_should_write) {
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
            if (needs_fixup && thread_should_write) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && thread_should_write) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
        }
    }

    if (np > 1 && threadIdx.y % np == 0) {
        // Combine the meta data for parallel warps via shared memory.
        // Warps with threadIdx.y % np != 0 must NOT return early.
        // All threads must return simultaneously to avoid race conditions with work on the next tile.

        constexpr int nmeta = np*cols_per_warp >= WARP_SIZE ? np*cols_per_warp/WARP_SIZE : 1;

        const int jc_meta = threadIdx.y*cols_per_warp + (np*cols_per_warp < WARP_SIZE ? threadIdx.x % (np*cols_per_warp) : threadIdx.x);
        float2 * const meta_ptr = ((float2 *) tile_Q) + jc_meta*(tile_stride/2) + nbatch_combine/2;
        float2 meta[nmeta];
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            meta[imeta] = meta_ptr[imeta * WARP_SIZE * tile_stride/2];
        }

        float KQ_cmn = meta[0].x; // KQ combine max new, max between all parallel warps.
#pragma unroll
        for (int imeta = 1; imeta < nmeta; ++imeta) {
            KQ_cmn = fmaxf(KQ_cmn, meta[imeta].x);
        }
#pragma unroll
        for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
            if (offset < WARP_SIZE) {
                KQ_cmn = fmaxf(KQ_cmn, __shfl_xor_sync(0xFFFFFFFF, KQ_cmn, offset, WARP_SIZE));
            }
        }

        float KQ_cms[nmeta]; // KQ combine max scale per warp.
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            KQ_cms[imeta] = expf(meta[imeta].x - KQ_cmn);
        }

        float KQ_crs = KQ_cms[0]*meta[0].y; // KQ combine rowsum, scaled sum of all parallel warps.
#pragma unroll
        for (int imeta = 1; imeta < nmeta; ++imeta) {
            KQ_crs += KQ_cms[imeta]*meta[imeta].y;
        }
#pragma unroll
        for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
            if (offset < WARP_SIZE) {
                KQ_crs += __shfl_xor_sync(0xFFFFFFFF, KQ_crs, offset, WARP_SIZE);
            }
        }

        __syncthreads();

        // Write back combined meta data:
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            if (np*cols_per_warp >= WARP_SIZE || threadIdx.x < np*cols_per_warp) {
                // Combined KQ max scale + rowsum.
                meta_ptr[imeta * WARP_SIZE * tile_stride/2] = make_float2(KQ_cms[imeta], KQ_crs);
            }
        }

        // Combined KQ max + rowsum.
        static_assert(cols_per_warp <= WARP_SIZE);
        if (needs_fixup && (cols_per_warp == WARP_SIZE || threadIdx.x < cols_per_warp)) {
            float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
            dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
        }
        if (is_fixup && (cols_per_warp == WARP_SIZE || threadIdx.x < cols_per_warp)) {
            float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
            dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
        }
    } else if (np > 1) {
        // Warps with threadIdx.y % np == 0 execute a __syncthreads() in the if branch.
        // Therefore, all other warps also need to execute a __syncthreads().
        // Otherwise the points at which warps synchronize with each other would become misaligned.
        __syncthreads();
    }

#pragma unroll
    for (int k00 = 0; k00 < DV/2; k00 += nbatch_combine) {
        if constexpr (cols_per_warp == 8) {
            const int jc_cwd = threadIdx.y*T_B_KQ::I + T_B_KQ::get_i(-1); // jc combine write data
#pragma unroll
            for (int k1 = 0; k1 < nbatch_combine; k1 += T_B_KQ::J) {
                const T_B_KQ B = get_transposed(VKQ_C[(k00 + k1)/T_B_KQ::J]); // Conversion of C to B matrix puts it in column-major format.

#pragma unroll
                for (int l = 0; l < T_B_KQ::ne; ++l) {
                    const int k = k1 + T_B_KQ::get_j(l);

                    tile_Q[jc_cwd*tile_stride + k] = B.x[l];
                }
            }
        } else {
            const int j0 = threadIdx.y*cols_per_warp;
#pragma unroll
            for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J) {
                const int i_acc = (k00 + k1)/T_C_VKQ::J;
                if constexpr (use_f32_acc) {
                    // FIX 2026-05-13: f32 N-slice writeback.
                    // VKQ_C is half2 representing 16x16 D. Per thread, x[l] holds half-col
                    // pair at (row get_i(l), cols get_j(l)*2 + 0,1). Layouts:
                    //   l=0: (row 0, cols 0..1) → from T0  (N=0..7)
                    //   l=1: (row 8, cols 0..1) → from T0  (T0 thread 0: x[2..3] at row 8)
                    //   l=2: (row 0, cols 8..9) → from T1  (N=8..15, T1 thread 0: x[0..1] at row 0)
                    //   l=3: (row 8, cols 8..9) → from T1  (T1 thread 0: x[2..3] at row 8)
                    // For tile<16,8,float> thread 0: x[0]=(r0,c0), x[1]=(r0,c1), x[2]=(r8,c0), x[3]=(r8,c1).
                    const tile<16, 8, float> & T0 = VKQ_C_f32[2*i_acc + 0];
                    const tile<16, 8, float> & T1 = VKQ_C_f32[2*i_acc + 1];
#pragma unroll
                    for (int l = 0; l < T_C_VKQ::ne; ++l) {
                        const int j = j0 + T_C_VKQ::get_i(l);
                        const int k = k1 + T_C_VKQ::get_j(l);
                        const tile<16, 8, float> & T = (l < 2) ? T0 : T1;
                        const int sub = (l & 1) ? 2 : 0;   // sub=0 → x[0,1]; sub=2 → x[2,3]
                        tile_Q[j*tile_stride + k] = make_half2(
                            __float2half(T.x[sub + 0]),
                            __float2half(T.x[sub + 1]));
                    }
                } else {
#pragma unroll
                    for (int l = 0; l < T_C_VKQ::ne; ++l) {
                        const int j = j0 + T_C_VKQ::get_i(l);
                        const int k = k1 + T_C_VKQ::get_j(l);

                        tile_Q[j*tile_stride + k] = VKQ_C[i_acc].x[l];
                    }
                }
            }
        }

        __syncthreads();

        if (np == 1 || threadIdx.y % np == 0) {
            // The first 2*2*gridDim.x*ncols floats in dstk_fixup are for storing max. values and row sums.
            // The values after that are for the partial results of the individual blocks.
            float2 * dstk_fixup_data = dstk_fixup + gridDim.x*(2*ncols) + blockIdx.x*(ncols*(DV/2));

#pragma unroll
            for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
                const int k0_start  = stride_k == WARP_SIZE ? 0 : nbatch_combine - nbatch_combine % (2*stride_k);
                const int k0_stop   =                             nbatch_combine - nbatch_combine % (1*stride_k);
                const int stride_jc = WARP_SIZE / stride_k;

                if (k0_start == k0_stop) {
                    continue;
                }

#pragma unroll
                for (int jc0_dst = 0; jc0_dst < ncols; jc0_dst += (nwarps/np)*stride_jc) {
                    const int jc_dst = jc0_dst + (threadIdx.y/np)*stride_jc + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

                    if (jc0_dst + (nwarps/np)*stride_jc > ncols && jc_dst >= ncols) {
                        break;
                    }

                    const int jc_tile_K = (jc_dst/cols_per_warp)*(np*cols_per_warp) + jc_dst % cols_per_warp;

                    const int j_dst = jc_dst / ncols2;
                    const int c_dst = jc_dst % ncols2;

                    if (!is_fixup && ((ncols1 > 1 && jt*ncols1 + j_dst >= int(ne01.z)) || (ncols2 > 1 && zt_gqa*ncols2 + c_dst >= gqa_ratio))) {
                        continue;
                    }

                    const float * meta_j = (const float *) tile_Q + jc_tile_K*tile_stride + nbatch_combine;
#ifdef KIMI_KVMAX_DEBUG
                    // Decisive softmax-denominator probe for query position 0.
                    // jc_dst==0 => j_dst==0 (token tile pos 0) && c_dst==0 (GQA head 0);
                    // jt==0 (first token tile) && zt_gqa==0 (first GQA head group) pins
                    // this to global query position 0 regardless of which stream-K block
                    // computes/combines it. meta_j[1] is KQ_crs, the FULLY combined (across
                    // the np=2 cooperating warps) softmax row-sum that normalizes query 0's
                    // output at line ~1668 (dstk_val /= KQ_rowsum_j). meta_j[0] is the per-warp
                    // combine scale KQ_cms (secondary). CORRECT (pos0 attends only to K[0]):
                    // denom ~= 1. BUGGY (pos0 summed unmasked future cells): denom > 1.
                    if (jc_dst == 0 && jt == 0 && zt_gqa == 0 && k0_start == 0) {
                        static __device__ unsigned int kimi_den_calls = 0;
                        unsigned int c = atomicAdd(&kimi_den_calls, 1u);
                        if (c < 12u) {
                            printf("KIMI_DENOM call=%u q0_KQ_sum=%g q0_combine_scale=%g needs_fixup=%d is_fixup=%d ncols1=%d ncols2=%d np_warps=%d\n",
                                   c, (double) meta_j[1], (double) meta_j[0],
                                   (int) needs_fixup, (int) is_fixup,
                                   (int) ncols1, (int) ncols2, (int) np);
                        }
                    }
#endif
#pragma unroll
                    for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                        const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                        float2 dstk_val = make_float2(0.0f, 0.0f);
#pragma unroll
                        for (int ip = 0; ip < np; ++ip) {
                            const float KQ_crs = np == 1 ? 1.0f : meta_j[ip*cols_per_warp * tile_stride + 0];
                            const float2 dstk_val_add = __half22float2(tile_Q[(jc_tile_K + ip*cols_per_warp) * tile_stride + k]);
                            dstk_val.x += dstk_val_add.x*KQ_crs;
                            dstk_val.y += dstk_val_add.y*KQ_crs;
                        }

                        if (!needs_fixup && !is_fixup) {
                            const float KQ_rowsum_j = meta_j[1];
                            dstk_val.x /= KQ_rowsum_j;
                            dstk_val.y /= KQ_rowsum_j;
                        }

                        if (is_fixup) {
                            dstk_fixup_data[jc_dst*(DV/2) + k00 + k] = dstk_val;
                        } else {
                            dstk[((jt*ncols1 + j_dst)*ne02 + c_dst)*(DV/2) + k00 + k] = dstk_val;
                        }
                    }
                }
            }
        }
        if (np > 1) {
            __syncthreads();
        }
    }
#else
    GGML_UNUSED_VARS(Q_f2, K_h2, V_h2, mask_h, indices, sinks_f, dstk, dstk_fixup,
        scale, slope, logit_softcap, ne01, ne02, gqa_ratio,
        stride_Q1, stride_Q2, stride_K, stride_V, stride_mask,
        jt, kb0_start, kb0_stop);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA4))
}

template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap, bool V_is_K_view, bool direct_causal_env = false>
__launch_bounds__(ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols1*ncols2), ggml_cuda_fattn_mma_get_occupancy(DKQ, DV, ncols1*ncols2))
static __global__ void flash_attn_ext_f16(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33,
                            const int32_t n_past_causal) {
#if defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)))

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(DKQ == 128 || DKQ == 256)) {
        NO_DEVICE_CODE;
        return;
    }
#ifdef VOLTA_MMA_AVAILABLE
    if (ncols1*ncols2 < 32) {
        NO_DEVICE_CODE;
        return;
    }
#endif // VOLTA_MMA_AVAILABLE

#if __CUDA_ARCH__ == GGML_CUDA_CC_TURING
    if (ncols1*ncols2 > 32) {
        NO_DEVICE_CODE;
        return;
    }
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_TURING

#if defined(AMD_WMMA_AVAILABLE)
    if (ncols1*ncols2 > 32 || ncols1*ncols2 < 16 || DKQ > 128 || ncols2 == 1) {
        NO_DEVICE_CODE;
        return;
    }
#endif // defined(AMD_WMMA_AVAILABLE)

    constexpr int ncols     = ncols1 * ncols2;
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
    constexpr int nthreads  = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols);
    constexpr int nwarps    = nthreads / WARP_SIZE;

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    const int stride_Q1   = nb01 / sizeof(float2);
    const int stride_Q2   = nb02 / sizeof(float2);
    const int stride_K    = nb11 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half);

    const int stride_V = V_is_K_view ? stride_K : nb21 / sizeof(half2);
    // 2026-05-30: relaxed for the causal-block-skip fix. The skip only needs a causal mask +
    // max_bias==0; the old ne11%ne01.z layout conditions were from the unimplemented design and
    // failed for the real multi-ub layout (ne11 = padded full cache, not a multiple of ubatch).
    const bool direct_causal =
        direct_causal_env && DKQ == 576 && (mask != nullptr || n_past_causal >= 0) && max_bias == 0.0f && ne01.z > 0;

    const int iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    const int iter_j     = (ne01.z    + (ncols1    - 1)) / ncols1;
    const int iter_z_gqa = (gqa_ratio + (ncols2    - 1)) / ncols2;

    // kbc == k block continuous, current index in continuous ijk space.
    int       kbc      = int64_t(blockIdx.x + 0)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / gridDim.x;
    const int kbc_stop = int64_t(blockIdx.x + 1)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / gridDim.x;

    // If the seams of 2 CUDA blocks fall within an output tile their results need to be combined.
    // For this we need to track both the block that starts the tile (needs_fixup) and the block that finishes the tile (is_fixup).
    // In the most general case >2 seams can fall into the same tile.

    // kb0 == k start index when in the output tile.
    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);

    while (kbc < kbc_stop && kb0_stop == iter_k) {
        // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
        const int sequence =  kbc /(iter_k*iter_j*iter_z_gqa*ne12);
        const int z_KV     = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
        const int zt_gqa   = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
        const int jt       = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

        const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

        const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
        const half2  * K_h2   = (const half2  *) (K + nb13*sequence + nb12*z_KV);
        const half   * mask_h = ncols2 == 1 && !mask ? nullptr :
            (const half *) (mask + nb33*(sequence % ne33));
        float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + zt_Q) * (DV/2);

        const half2 * V_h2 = V_is_K_view ? K_h2 : (const half2 *) (V + nb23*sequence + nb22*z_KV);
        const float * sinks_f = sinks ? (const float *) sinks + zt_Q : nullptr;

        const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;

        if (KV_max) {
            kb0_stop = min(kb0_stop, KV_max[sequence*iter_j + jt] / nbatch_fa);
        }
#ifdef KIMI_KVMAX_DEBUG
        {
            static __device__ unsigned int kimi_kb_calls = 0;
            if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
                unsigned int c = atomicAdd(&kimi_kb_calls, 1u);
                if (c < 8u) {
                    printf("KIMI_KB0 call=%u jt=%d kb0_start=%d kb0_stop=%d KV_rows=%d nbatch_fa=%d\n",
                           c, (int) jt, (int) kb0_start, (int) kb0_stop, (int) (kb0_stop * nbatch_fa), (int) nbatch_fa);
                }
            }
        }
#endif
        constexpr bool is_fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        if (kb0_start == 0) {
            constexpr bool needs_fixup = false; // CUDA block is working on an entire tile.
            flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, needs_fixup, is_fixup, false>
                (Q_f2, K_h2, V_h2, mask_h, nullptr, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, direct_causal, jt, zt_gqa, kb0_start, kb0_stop, n_past_causal);
        } else {
            constexpr bool needs_fixup = true; // CUDA block is missing the beginning of a tile.
            flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, needs_fixup, is_fixup, false>
                (Q_f2, K_h2, V_h2, mask_h, nullptr, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, direct_causal, jt, zt_gqa, kb0_start, kb0_stop, n_past_causal);
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index.
    const int sequence =  kbc /(iter_k*iter_j*iter_z_gqa*ne12);
    const int z_KV     = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    const int zt_gqa   = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    const int jt       = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
    const half2  * K_h2   = (const half2  *) (K + nb13*sequence + nb12*z_KV);
    const half   * mask_h = ncols2 == 1 && !mask ? nullptr :
        (const half *) (mask + nb33*(sequence % ne33));
    float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + zt_Q) * (DV/2);

    const half2 * V_h2 = V_is_K_view ? K_h2 : (const half2 *) (V + nb23*sequence + nb22*z_KV);
    const float * sinks_f = sinks ? (const float *) sinks + zt_Q : nullptr;

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;

    if (KV_max) {
        kb0_stop = min(kb0_stop, KV_max[sequence*iter_j + jt] / nbatch_fa);
    }

    constexpr bool is_fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    constexpr bool needs_fixup = false;
    flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, needs_fixup, is_fixup, false>
        (Q_f2, K_h2, V_h2, mask_h, nullptr, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
         ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, direct_causal, jt, zt_gqa, kb0_start, kb0_stop, n_past_causal);
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)))
}

template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    constexpr int ncols = ncols1 * ncols2;

    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc);
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);
    const int  nstages        = ggml_cuda_fattn_mma_get_nstages       (DKQ, DV, ncols1, ncols2, cc);

    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const int nwarps        = nthreads / WARP_SIZE;

    const size_t nbytes_shared_KV_1stage = nbatch_fa            * std::max(nbatch_K2 + 4,  nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_KV_2stage = nbatch_fa            *         (nbatch_K2 + 4 + nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_Q         = ncols                * (DKQ/2 + 4)                             * sizeof(half2);
    const size_t nbytes_shared_mask      = ncols1               * (nbatch_fa/2 + 4)                       * sizeof(half2);
    const size_t nbytes_shared_combine   = nwarps*cols_per_warp * (nbatch_combine + 4)                    * sizeof(half2);

    const size_t nbytes_shared_KV = nstages <= 1 ? nbytes_shared_KV_1stage : nbytes_shared_KV_2stage;

    const size_t nbytes_shared_total = std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q,  nbytes_shared_KV + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

#if defined(GGML_USE_HIP)
    using fattn_kernel_ptr_t = const void*;
#else
    using fattn_kernel_ptr_t = fattn_kernel_t;
#endif // defined(GGML_USE_HIP)
    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        if constexpr (DKQ == 576) {
            const bool use_v_as_k_view = std::getenv("LLAMA_FA_MLA_NO_K_REUSE") == nullptr;
            const bool use_direct_causal = std::getenv("LLAMA_FA_MLA_CAUSAL_DIRECT") != nullptr;
            if (use_v_as_k_view) {
                fattn_kernel = use_direct_causal
                    ? flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, true, true>
                    : flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, true, false>;
            } else {
                fattn_kernel = use_direct_causal
                    ? flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, false, true>
                    : flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, false, false>;
            }
        } else {
            fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, false>;
        }

#if !defined(GGML_USE_MUSA)
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
        // Phase 2.2 angle 31 (2026-05-13): diagnostic for H2 (shmem carveout
        // silently truncates opt-in request). Log actual vs requested per
        // first launch when LLAMA_FA_SHMEM_AUDIT=1.
        if (const char * env = std::getenv("LLAMA_FA_SHMEM_AUDIT")) {
            if (env[0] != '\0' && env[0] != '0') {
                static bool logged = false;
                if (!logged) {
                    logged = true;
                    cudaFuncAttributes attr;
                    cudaError_t e = cudaFuncGetAttributes(&attr, reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel));
                    fprintf(stderr,
                        "*** FA_SHMEM_AUDIT: DKQ=%d DV=%d ncols1=%d ncols2=%d nwarps=%d "
                        "requested_shared=%d maxDynShared=%d staticShared=%zu "
                        "regs/thread=%d numRegs=%d ptxVersion=%d binaryVersion=%d err=%s\n",
                        DKQ, DV, ncols1, ncols2, nwarps,
                        nbytes_shared_total,
                        attr.maxDynamicSharedSizeBytes, attr.sharedSizeBytes,
                        attr.numRegs, attr.numRegs, attr.ptxVersion, attr.binaryVersion,
                        cudaGetErrorString(e));
                    fflush(stderr);
                }
            }
        }
#endif // !defined(GGML_USE_MUSA)
    } else {
        constexpr bool use_logit_softcap = true;
        if constexpr (DKQ == 576) {
            const bool use_v_as_k_view = std::getenv("LLAMA_FA_MLA_NO_K_REUSE") == nullptr;
            const bool use_direct_causal = std::getenv("LLAMA_FA_MLA_CAUSAL_DIRECT") != nullptr;
            if (use_v_as_k_view) {
                fattn_kernel = use_direct_causal
                    ? flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, true, true>
                    : flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, true, false>;
            } else {
                fattn_kernel = use_direct_causal
                    ? flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, false, true>
                    : flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, false, false>;
            }
        } else {
            fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, false>;
        }

#if !defined(GGML_USE_MUSA)
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
        // Phase 2.2 angle 31 (2026-05-13): audit second branch too (logit_softcap=true path).
        if (const char * env = std::getenv("LLAMA_FA_SHMEM_AUDIT")) {
            if (env[0] != '\0' && env[0] != '0') {
                static bool logged_b = false;
                if (!logged_b) {
                    logged_b = true;
                    cudaFuncAttributes attr;
                    cudaError_t e = cudaFuncGetAttributes(&attr, reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel));
                    fprintf(stderr,
                        "*** FA_SHMEM_AUDIT_B: DKQ=%d DV=%d ncols1=%d ncols2=%d nwarps=%d "
                        "requested_shared=%zu maxDynShared=%d staticShared=%zu numRegs=%d err=%s\n",
                        DKQ, DV, ncols1, ncols2, nwarps,
                        nbytes_shared_total,
                        attr.maxDynamicSharedSizeBytes, attr.sharedSizeBytes,
                        attr.numRegs, cudaGetErrorString(e));
                    fflush(stderr);
                }
            }
        }
#endif // !defined(GGML_USE_MUSA)
    }

    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, true, true, true);
}

// ============================================================
// V6 sparse-MMA path: the dense MMA kernel with indices gather
// ============================================================

// Sparse variant of the outer kernel. Reads the indices from src[3] (sparse op layout:
//   src[0]=Q, src[1]=K_full, src[2]=V_full, src[3]=indices, src[4]=mask).
// The kernel body is identical to flash_attn_ext_f16 except process_tile is called with sparse=true.
template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap, bool V_is_K_view>
__launch_bounds__(ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols1*ncols2), ggml_cuda_fattn_mma_get_occupancy(DKQ, DV, ncols1*ncols2))
static __global__ void flash_attn_ext_f16_sparse(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,      // sparse mask [top_k, n_q]
        const int  * __restrict__ indices,   // gathered row indices [top_k, n_q]
        const int32_t indices_stride,         // stride between query columns in int32 elements
        const char * __restrict__ sinks,     // unused for sparse, pass nullptr
        const int  * __restrict__ KV_max,    // unused for sparse, pass nullptr
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#if defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)))

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(DKQ == 128 || DKQ == 256)) {
        NO_DEVICE_CODE;
        return;
    }
#ifdef VOLTA_MMA_AVAILABLE
    if (ncols1*ncols2 < 32) {
        NO_DEVICE_CODE;
        return;
    }
#endif // VOLTA_MMA_AVAILABLE

#if __CUDA_ARCH__ == GGML_CUDA_CC_TURING
    if (ncols1*ncols2 > 32) {
        NO_DEVICE_CODE;
        return;
    }
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_TURING

#if defined(AMD_WMMA_AVAILABLE)
    if (ncols1*ncols2 > 32 || ncols1*ncols2 < 16 || DKQ > 128 || ncols2 == 1) {
        NO_DEVICE_CODE;
        return;
    }
#endif // defined(AMD_WMMA_AVAILABLE)

    constexpr int ncols     = ncols1 * ncols2;
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
    constexpr int nthreads  = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols);
    constexpr int nwarps    = nthreads / WARP_SIZE;

    const int gqa_ratio = ne02 / ne12;

    const int stride_Q1   = nb01 / sizeof(float2);
    const int stride_Q2   = nb02 / sizeof(float2);
    const int stride_K    = nb11 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half);

    const int stride_V = V_is_K_view ? stride_K : nb21 / sizeof(half2);

    const int iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    const int iter_j     = (ne01.z    + (ncols1    - 1)) / ncols1;
    const int iter_z_gqa = (gqa_ratio + (ncols2    - 1)) / ncols2;

    int       kbc      = int64_t(blockIdx.x + 0)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / gridDim.x;
    const int kbc_stop = int64_t(blockIdx.x + 1)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / gridDim.x;

    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);

    while (kbc < kbc_stop && kb0_stop == iter_k) {
        const int sequence =  kbc /(iter_k*iter_j*iter_z_gqa*ne12);
        const int z_KV     = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
        const int zt_gqa   = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
        const int jt       = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

        const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2;

        const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
        const half2  * K_h2   = (const half2  *) (K + nb13*sequence + nb12*z_KV);
        // Sparse mask is [top_k, n_q] (or equivalent flattened query axis).
        // The per-query offset is applied by flash_attn_ext_f16_load_mask via stride_mask and jt.
        const half   * mask_h = mask ? (const half *) mask : nullptr;
        float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + zt_Q) * (DV/2);

        const half2 * V_h2 = V_is_K_view ? K_h2 : (const half2 *) (V + nb23*sequence + nb22*z_KV);

        // For sparse, process_tile uses kb0*nbatch_fa as the row offset inside this query's index column.
        const int * indices_jt = indices + jt * indices_stride;

        const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;

        constexpr bool is_fixup = false;
        if (kb0_start == 0) {
            constexpr bool needs_fixup = false;
            flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, needs_fixup, is_fixup, true>
                (Q_f2, K_h2, V_h2, mask_h, indices_jt, nullptr, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, false, jt, zt_gqa, kb0_start, kb0_stop, -1);
        } else {
            constexpr bool needs_fixup = true;
            flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, needs_fixup, is_fixup, true>
                (Q_f2, K_h2, V_h2, mask_h, indices_jt, nullptr, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, false, jt, zt_gqa, kb0_start, kb0_stop, -1);
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    const int sequence =  kbc /(iter_k*iter_j*iter_z_gqa*ne12);
    const int z_KV     = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    const int zt_gqa   = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    const int jt       = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2;

    const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
    const half2  * K_h2   = (const half2  *) (K + nb13*sequence + nb12*z_KV);
    const half   * mask_h = mask ? (const half *) mask : nullptr;
    float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + zt_Q) * (DV/2);

    const half2 * V_h2 = V_is_K_view ? K_h2 : (const half2 *) (V + nb23*sequence + nb22*z_KV);
    const int * indices_jt = indices + jt * indices_stride;

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;

    constexpr bool is_fixup = true;
    constexpr bool needs_fixup = false;
    flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, needs_fixup, is_fixup, true>
        (Q_f2, K_h2, V_h2, mask_h, indices_jt, nullptr, dstk, dst_meta, scale, slope, logit_softcap,
         ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, false, jt, zt_gqa, kb0_start, kb0_stop, -1);
#else
    GGML_UNUSED_VARS(Q, K, V, mask, indices, indices_stride, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && ...
}

#define DECL_FATTN_MMA_F16_CASE(DKQ, DV, ncols1, ncols2)                          \
    template void ggml_cuda_flash_attn_ext_mma_f16_case                           \
    <DKQ, DV, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(DKQ, DV, ncols)   \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 1,  1); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 2,  2); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 4,  4); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 8,  8); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/16, 16); \

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,   8)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,  16)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,  32)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,  64)

// DeepSeek V4 raw SWA attention. GQA dispatch fixes ncols2=8, leaving four
// ncols1 variants for decode and prefill query counts.
extern DECL_FATTN_MMA_F16_CASE(512, 512, 1, 8);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 2, 8);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 4, 8);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 8, 8);

// The number of viable configurations for Deepseek is very limited:
extern DECL_FATTN_MMA_F16_CASE(576, 512, 1, 16);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 2, 16);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 4, 16);

// For GLM 4.7 Flash
extern DECL_FATTN_MMA_F16_CASE(576, 512,  4,  4);
extern DECL_FATTN_MMA_F16_CASE(576, 512,  8,  4);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 16,  4);
extern DECL_FATTN_MMA_F16_CASE(576, 512,  1, 32);
extern DECL_FATTN_MMA_F16_CASE(576, 512,  2, 32);
