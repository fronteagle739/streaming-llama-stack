#include "dispatch-counter.cuh"
#include "argsort.cuh"
#include "top-k.cuh"
#include "ggml-dsv41-stream-topk.h"
#include <atomic>
static ggml_cuda_dispatch_counter dsv41_stream_topk_queued{0};
extern "C" uint64_t ggml_backend_cuda_dsv41_stream_topk_queued(void){return dsv41_stream_topk_queued.load(std::memory_order_acquire);}

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef GGML_CUDA_USE_CUB

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif  // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static constexpr int TOP_K_TILE_SIZE = 8192;

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env);

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env);
}

#endif  // CUB_TOP_K_AVAILABLE

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
    if(ggml_dsv41_stream_topk_requested(dst)){
        GGML_ASSERT(ggml_dsv41_stream_topk_shape(dst));
        top_k_f32_i32_cuda_bitonic_tiles_guarded(src0_d,dst_d,int(ncols),int(nrows),8192,int(k),stream,true);
        CUDA_CHECK(cudaGetLastError());dsv41_stream_topk_queued.fetch_add(1,std::memory_order_release);return;
    }
#ifdef CUB_TOP_K_AVAILABLE
    static const bool force_bitonic_single_row = []() {
        const char * env = std::getenv("LLAMA_DSV4_TOPK_FORCE_BITONIC");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    // LLAMADECODE: a single row takes the SAME deterministic bitonic / bitonic-tiles
    // network as multi-row (prefill), so a 1-row decode step selects the same ids
    // in the same order on every run. CUB DeviceTopK (determinism::not_guaranteed,
    // unsorted output) remains only as the explicit A/B control
    // LLAMA_DSV4_TOPK_CUB_1ROW=1, and for k > TOP_K_TILE_SIZE.
    static const bool cub_single_row = []() {
        const char * env = std::getenv("LLAMA_DSV4_TOPK_CUB_1ROW");
        return env != nullptr && env[0] == '1' && env[1] == '\0';
    }();
    const bool deterministic_rows = nrows > 1 || force_bitonic_single_row || !cub_single_row;

    // DeviceTopK has no segmented API yet. Use the one-kernel argsort path for multi-row calls when the
    // padded row fits in the default dynamic shared-memory limit. It writes only the first k descending
    // indices, so no full-size output allocation or follow-up copy is needed.
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = std::min<size_t>(
        ggml_cuda_info().devices[ggml_cuda_get_device()].smpb, 48 * 1024);

    if (deterministic_rows && shared_mem <= max_shared_mem) {
        if (nrows == 1) {
            GGML_LOG_DEBUG("DSV4_TOPK_BITONIC_1ROW: ncols=%lld k=%lld shared_mem=%zu\n",
                    (long long) ncols, (long long) k, shared_mem);
        }
        top_k_f32_i32_cuda_bitonic(src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }

    if (deterministic_rows && k <= TOP_K_TILE_SIZE) {
        const int first_width = ((ncols + TOP_K_TILE_SIZE - 1) / TOP_K_TILE_SIZE) * k;
        ggml_cuda_pool_alloc<int> candidates_a_alloc(pool, (size_t) nrows * first_width);
        int * candidates_a = candidates_a_alloc.get();

        top_k_f32_i32_cuda_bitonic_tiles(
                src0_d, candidates_a, ncols, nrows, TOP_K_TILE_SIZE, k, stream);

        int width = first_width;
        int * candidates = candidates_a;
        if (width > TOP_K_TILE_SIZE) {
            const int second_width = ((width + TOP_K_TILE_SIZE - 1) / TOP_K_TILE_SIZE) * k;
            ggml_cuda_pool_alloc<int> candidates_b_alloc(pool, (size_t) nrows * second_width);
            int * candidates_b = candidates_b_alloc.get();

            while (width > TOP_K_TILE_SIZE) {
                const int next_width = ((width + TOP_K_TILE_SIZE - 1) / TOP_K_TILE_SIZE) * k;
                int * next_candidates = candidates == candidates_a ? candidates_b : candidates_a;
                top_k_f32_i32_cuda_bitonic_candidates(src0_d, candidates, next_candidates,
                        ncols, nrows, width, TOP_K_TILE_SIZE, k, stream);
                candidates = next_candidates;
                width = next_width;
            }

            top_k_f32_i32_cuda_bitonic_candidates(
                    src0_d, candidates, dst_d, ncols, nrows, width, TOP_K_TILE_SIZE, k, stream);
            return;
        }

        top_k_f32_i32_cuda_bitonic_candidates(
                src0_d, candidates, dst_d, ncols, nrows, width, TOP_K_TILE_SIZE, k, stream);
        return;
    }

    // DeviceTopK remains faster for single-row decode and handles rows too wide for the shared-memory
    // argsort kernel. The per-row loop is also the general fallback for k > TOP_K_TILE_SIZE.
    for (int64_t i = 0; i < nrows; ++i) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
    return;
#else  // CUB_TOP_K_AVAILABLE

#ifdef GGML_CUDA_USE_CUB
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    if (shared_mem > max_shared_mem || ncols > 1024) {
        argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    } else {
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    }
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#else  // GGML_CUDA_USE_CUB
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif  // GGML_CUDA_USE_CUB
#endif  // CUB_TOP_K_AVAILABLE
}
