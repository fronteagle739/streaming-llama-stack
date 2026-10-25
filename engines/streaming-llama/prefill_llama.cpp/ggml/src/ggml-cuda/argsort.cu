#include "argsort.cuh"
#include "ggml-dsv41-topk.h"
#include <atomic>

static std::atomic<uint64_t> dsv41_topk_radix_queued{0};
extern "C" uint64_t ggml_backend_cuda_dsv41_topk_radix_queued(void) {
    return dsv41_topk_radix_queued.load(std::memory_order_acquire);
}

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 1)
#        define STRIDED_ITERATOR_AVAILABLE
#    endif
using namespace cub;
#endif  // GGML_CUDA_USE_CUB

static __global__ void init_indices(int * indices, const int ncols, const int nrows) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;

    if (col < ncols && row < nrows) {
        indices[row * ncols + col] = col;
    }
}

#ifndef STRIDED_ITERATOR_AVAILABLE
static __global__ void init_offsets(int * offsets, const int ncols, const int nrows) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= nrows) {
        offsets[idx] = idx * ncols;
    }
}
#endif  // STRIDED_ITERATOR_AVAILABLE

#ifdef GGML_CUDA_USE_CUB
void argsort_f32_i32_cuda_cub(ggml_cuda_pool & pool,
                              const float *    x,
                              int *            dst,
                              const int        ncols,
                              const int        nrows,
                              ggml_sort_order  order,
                              cudaStream_t     stream) {
    ggml_cuda_pool_alloc<int>   temp_indices_alloc(pool, ncols * nrows);
    ggml_cuda_pool_alloc<float> temp_keys_alloc(pool, ncols * nrows);

    int *   temp_indices = temp_indices_alloc.get();
    float * temp_keys    = temp_keys_alloc.get();

    static const int block_size = 256;
    const dim3 grid_size((ncols + block_size - 1) / block_size, nrows);
    init_indices<<<grid_size, block_size, 0, stream>>>(temp_indices, ncols, nrows);

#ifdef STRIDED_ITERATOR_AVAILABLE
    auto offset_iterator = cuda::make_strided_iterator(cuda::make_counting_iterator(0), ncols);
#else
    ggml_cuda_pool_alloc<int> offsets_alloc(pool, nrows + 1);
    int *                     offset_iterator = offsets_alloc.get();
    const dim3                offset_grid((nrows + block_size - 1) / block_size);
    init_offsets<<<offset_grid, block_size, 0, stream>>>(offset_iterator, ncols, nrows);
#endif
    CUDA_CHECK(cudaMemcpyAsync(temp_keys, x, ncols * nrows * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    size_t temp_storage_bytes = 0;

    if (order == GGML_SORT_ORDER_ASC) {
        if (nrows == 1) {
            DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, temp_keys, temp_keys,
                                       temp_indices, dst, ncols, 0, sizeof(float) * 8, stream);
        } else {
            DeviceSegmentedSort::SortPairs(nullptr, temp_storage_bytes, temp_keys, temp_keys,
                                           temp_indices, dst, ncols * nrows, nrows,
                                           offset_iterator, offset_iterator + 1, stream);
        }
    } else {
        if (nrows == 1) {
            DeviceRadixSort::SortPairsDescending(nullptr, temp_storage_bytes, temp_keys, temp_keys,
                                                 temp_indices, dst, ncols, 0, sizeof(float) * 8, stream);
        } else {
            DeviceSegmentedSort::SortPairsDescending(nullptr, temp_storage_bytes, temp_keys, temp_keys,
                                                     temp_indices, dst, ncols * nrows, nrows,
                                                     offset_iterator, offset_iterator + 1, stream);
        }
    }

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    if (order == GGML_SORT_ORDER_ASC) {
        if (nrows == 1) {
            DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, temp_keys, temp_keys,
                                       temp_indices, dst, ncols, 0, sizeof(float) * 8, stream);
        } else {
            DeviceSegmentedSort::SortPairs(d_temp_storage, temp_storage_bytes, temp_keys, temp_keys,
                                           temp_indices, dst, ncols * nrows, nrows,
                                           offset_iterator, offset_iterator + 1, stream);
        }
    } else {
        if (nrows == 1) {
            DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, temp_keys, temp_keys,
                                                 temp_indices, dst, ncols, 0, sizeof(float) * 8, stream);
        } else {
            DeviceSegmentedSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, temp_keys, temp_keys,
                                                     temp_indices, dst, ncols * nrows, nrows,
                                                     offset_iterator, offset_iterator + 1, stream);
        }
    }
}
#endif  // GGML_CUDA_USE_CUB

template<typename T>
static inline __device__ void ggml_cuda_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

struct store_ser {
    constexpr static bool has_thresh = true;
    int   min_experts;
    float thresh_experts;
    store_ser(int min, float thresh) : min_experts(min), thresh_experts(thresh) {}
};

struct store {
    constexpr static bool has_thresh = false;
};

template<ggml_sort_order order>
static __device__ __forceinline__ void sort(int ncols_pad, int ncols, int col, const float * x_row, int * dst_row) {
    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ncols ||
                        (dst_row[ixj] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (dst_row[ixj] >= ncols ||
                        (dst_row[col] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            __syncthreads();
        }
    }
}

template<ggml_sort_order order>
static __device__ __forceinline__ void sort_strided(
        int ncols_pad, int ncols, const float * x_row, int * dst_row) {
    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int col = threadIdx.x; col < ncols_pad; col += blockDim.x) {
                const int ixj = col ^ j;
                if (ixj > col) {
                    if ((col & k) == 0) {
                        if (dst_row[col] >= ncols ||
                            (dst_row[ixj] < ncols && (order == GGML_SORT_ORDER_ASC ?
                                x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                                x_row[dst_row[col]] < x_row[dst_row[ixj]]))) {
                            ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                        }
                    } else {
                        if (dst_row[ixj] >= ncols ||
                            (dst_row[col] < ncols && (order == GGML_SORT_ORDER_ASC ?
                                x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                                x_row[dst_row[col]] > x_row[dst_row[ixj]]))) {
                            ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

template<ggml_sort_order order>
static __global__ void k_top_k_f32_i32(
        const float * x, int * dst, const int ncols, const int ncols_pad, const int ntop) {
    const int row = blockIdx.x;

    const float * x_row = x + row * ncols;
    extern __shared__ int dst_row[];

    for (int col = threadIdx.x; col < ncols_pad; col += blockDim.x) {
        dst_row[col] = col;
    }

    __syncthreads();

    sort_strided<order>(ncols_pad, ncols, x_row, dst_row);

    for (int col = threadIdx.x; col < ntop; col += blockDim.x) {
        dst[row * ntop + col] = dst_row[col];
    }
}

static __device__ __forceinline__ bool top_k_candidate_before(
        const float * x_row, const int lhs, const int rhs) {
    if (lhs < 0) {
        return false;
    }
    if (rhs < 0) {
        return true;
    }

    const float lhs_value = x_row[lhs];
    const float rhs_value = x_row[rhs];
    return lhs_value > rhs_value || (lhs_value == rhs_value && lhs < rhs);
}

static __device__ __forceinline__ void top_k_sort_candidates_strided(
        const int width_pad, const float * x_row, int * candidates) {
    for (int k = 2; k <= width_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int col = threadIdx.x; col < width_pad; col += blockDim.x) {
                const int peer = col ^ j;
                if (peer > col) {
                    const bool descending = (col & k) == 0;
                    const bool swap = descending
                        ? top_k_candidate_before(x_row, candidates[peer], candidates[col])
                        : top_k_candidate_before(x_row, candidates[col], candidates[peer]);
                    if (swap) {
                        ggml_cuda_swap(candidates[col], candidates[peer]);
                    }
                }
            }
            __syncthreads();
        }
    }
}

// Preserve the original network, including its NaN and index-tie semantics.
// Cached keys move with their candidate; no score arithmetic is introduced.
static __device__ __forceinline__ bool top_k_cached_before(
        const int lhs, const float lhs_value, const int rhs, const float rhs_value) {
    if (lhs < 0) return false;
    if (rhs < 0) return true;
    return lhs_value > rhs_value || (lhs_value == rhs_value && lhs < rhs);
}

template<bool implicit_iota, bool cached_keys = false>
static __global__ void k_top_k_f32_i32_tiles(
        const float * x, const int * src_candidates, int * dst,
        const int ncols, const int width, const int tile_size, const int ntop, const bool reject_nan,
        const unsigned * only_rows) {
    const int tile = blockIdx.x;
    const int row  = blockIdx.y;
    if (only_rows && !only_rows[row]) return;  // LLAMADECODE fast select: row already exact
    const int tile_begin = tile * tile_size;
    const int tile_width = min(tile_size, width - tile_begin);
    const int tile_width_pad = 1 << (32 - __clz(tile_width - 1));
    const int ntiles = (width + tile_size - 1) / tile_size;
    const float * x_row = x + (size_t) row * ncols;

    extern __shared__ int tile_candidates[];
    float * tile_keys = reinterpret_cast<float *>(tile_candidates + tile_size);
    for (int col = threadIdx.x; col < tile_width_pad; col += blockDim.x) {
        if (col < tile_width) {
            const int input_col = tile_begin + col;
            if constexpr (implicit_iota) {
                tile_candidates[col] = input_col;
            } else {
                tile_candidates[col] = src_candidates[(size_t) row * width + input_col];
            }
        } else {
            tile_candidates[col] = -1;
        }
        if(reject_nan&&tile_candidates[col]>=0&&
                (__float_as_uint(x_row[tile_candidates[col]])&0x7fffffffu)>0x7f800000u){
            printf("DSV41_STREAM_TOPK_NAN_REFUSED row=%d column=%d\n",row,tile_candidates[col]);__trap();
        }
        if constexpr (cached_keys) {
            const int id = tile_candidates[col];
            tile_keys[col] = id < 0 ? 0.0f : x_row[id];
        }
    }

    __syncthreads();
    if constexpr (cached_keys) {
        for (int k = 2; k <= tile_width_pad; k *= 2) {
            for (int j = k / 2; j > 0; j /= 2) {
                for (int col = threadIdx.x; col < tile_width_pad; col += blockDim.x) {
                    const int peer = col ^ j;
                    if (peer > col) {
                        const bool descending = (col & k) == 0;
                        const bool swap = descending
                            ? top_k_cached_before(tile_candidates[peer], tile_keys[peer], tile_candidates[col], tile_keys[col])
                            : top_k_cached_before(tile_candidates[col], tile_keys[col], tile_candidates[peer], tile_keys[peer]);
                        if (swap) {
                            ggml_cuda_swap(tile_candidates[col], tile_candidates[peer]);
                            ggml_cuda_swap(tile_keys[col], tile_keys[peer]);
                        }
                    }
                }
                __syncthreads();
            }
        }
    } else {
        top_k_sort_candidates_strided(tile_width_pad, x_row, tile_candidates);
    }

    const int output_width = ntiles * ntop;
    for (int col = threadIdx.x; col < ntop; col += blockDim.x) {
        dst[(size_t) row * output_width + tile * ntop + col] =
            col < tile_width_pad ? tile_candidates[col] : -1;
    }
}

#ifdef GGML_CUDA_USE_CUB
// Reused from ds4-win b12b8ab:ds4_cuda.cu:22280-22326. Adapted to
// tiled/global candidate IDs. Canonicalize signed zero to the original
// comparator's equal-value tie. NaNs retain the original comparison NETWORK.
using dsv41_TopKBlockSort = cub::BlockRadixSort<uint64_t, 512, 16>;
static __device__ __forceinline__ uint64_t dsv41_topk_pack_key(float v, const int id) {
    if (id < 0) return 0;
    const uint32_t u = __float_as_uint(v == 0.0f ? 0.0f : v);
    const uint32_t ordered = (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
    return (uint64_t(ordered) << 32u) | uint64_t(0xffffffffu - uint32_t(id));
}

template<bool implicit_iota>
static __global__ void __launch_bounds__(512, 1) k_dsv41_topk_radix_tiles(
        const float *x, const int *src_candidates, int *dst,
        const int ncols, const int width, const int ntop, const bool reject_nan,
        const unsigned *only_rows) {
    constexpr int tile_size = 8192, items = 16;
    const int tile = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
    if (only_rows && !only_rows[row]) return;  // LLAMADECODE fast select: row already exact
    const int begin = tile * tile_size, tile_width = min(tile_size, width - begin);
    const int padded = 1 << (32 - __clz(tile_width - 1));
    const int ntiles = (width + tile_size - 1) / tile_size;
    const float *x_row = x + size_t(row) * ncols;
    extern __shared__ __align__(16) unsigned char radix_smem[];
    __shared__ unsigned has_nan;
    if (tid == 0) has_nan = 0;
    __syncthreads();
    uint64_t keys[items];
#pragma unroll
    for (int item = 0; item < items; ++item) {
        const int i = tid * items + item;
        int id = -1;
        if (i < tile_width) {
            if constexpr (implicit_iota) id = begin + i;
            else id = src_candidates[size_t(row) * width + begin + i];
        }
        const float value = id < 0 ? 0.0f : x_row[id];
        const uint32_t magnitude = __float_as_uint(value) & 0x7fffffffu;
        if (id < -1 || (id >= 0 && magnitude > 0x7f800000u)) atomicOr(&has_nan, 1u);
        if (id >= 0 && magnitude && magnitude < 0x00800000u) atomicOr(&has_nan, 2u);
        keys[item] = dsv41_topk_pack_key(value, id);
    }
    __syncthreads();
    if(reject_nan&&(has_nan&1u)){
        if(tid==0)printf("DSV41_STREAM_TOPK_NAN_REFUSED row=%d tile=%d\n",row,tile);
        __syncthreads(); // enqueue the diagnostic before another warp traps
        __trap();
    }
    if (has_nan) {
        int *candidates = reinterpret_cast<int *>(radix_smem);
#pragma unroll
        for (int item = 0; item < items; ++item) {
            const int i = tid * items + item;
            int id = -1;
            if (i < tile_width) {
                if constexpr (implicit_iota) id = begin + i;
                else id = src_candidates[size_t(row) * width + begin + i];
            }
            candidates[i] = id;
        }
        __syncthreads();
        top_k_sort_candidates_strided(padded, x_row, candidates);
        for (int i = tid; i < ntop; i += blockDim.x)
            dst[size_t(row) * ntiles * ntop + tile * ntop + i] = i < padded ? candidates[i] : -1;
        return;
    }
    auto &storage = *reinterpret_cast<dsv41_TopKBlockSort::TempStorage *>(radix_smem);
    dsv41_TopKBlockSort(storage).SortDescending(keys);
#pragma unroll
    for (int item = 0; item < items; ++item) {
        const int i = tid * items + item;
        if (i < ntop) dst[size_t(row) * ntiles * ntop + tile * ntop + i] = int(0xffffffffu - uint32_t(keys[item]));
    }
}

static constexpr size_t dsv41_topk_radix_smem =
    sizeof(dsv41_TopKBlockSort::TempStorage) > 8192 * sizeof(int) ?
    sizeof(dsv41_TopKBlockSort::TempStorage) : 8192 * sizeof(int);
static bool dsv41_topk_radix_enabled(const int tile_size, const int k) {
    const char *env = std::getenv("LLAMA_DSV41_TOPK_RADIX");
    return env && env[0] == '1' && tile_size == 8192 && k == 512 &&
        dsv41_topk_radix_smem + sizeof(unsigned) <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo;
}
#endif

template<ggml_sort_order order, typename Store, typename dst_t>
static __global__ void k_argsort_f32_T(const float * x, dst_t * dst, const int ncols, int ncols_pad, int ntop, Store s) {
    int col = threadIdx.x;
    int row = blockIdx.x;

    if (col >= ncols_pad) {
        return;
    }

    const float * x_row = x + row * ncols;
    extern __shared__ int dst_row[];

    dst_row[col] = col;

    __syncthreads();

    sort<order>(ncols_pad, ncols, col, x_row, dst_row);

    if constexpr (Store::has_thresh) {
        __syncthreads();
        float max_val = x_row[dst_row[0]];
        if (col < ntop) {
            if constexpr (std::is_same_v<dst_t, int>) {
                dst[row * ntop + col] = col < s.min_experts || x_row[dst_row[col]] >= s.thresh_experts * max_val ? dst_row[col] : -1;
            } else {
                dst[row * ntop + col] = col < s.min_experts || x_row[dst_row[col]] >= s.thresh_experts * max_val ? x_row[dst_row[col]] : 0.0f;
            }
        }
    } else {
        if (col < ntop) {
            if constexpr (std::is_same_v<dst_t, int>) {
                dst[row * ntop + col] = dst_row[col];
            } else {
                dst[row * ntop + col] = x_row[dst_row[col]];
            }
        }
    }
}

template<ggml_sort_order order>
static __global__ void k_argsort_f32_u8(const float * x, uint8_t * dst, const int ncols, int ncols_pad, int ntop) {
    int col = threadIdx.x;
    int row = blockIdx.x;

    if (col >= ncols_pad) {
        return;
    }

    const float * x_row = x + row * ncols;
    extern __shared__ int dst_row[];

    dst_row[col] = col;

    __syncthreads();

    sort<order>(ncols_pad, ncols, col, x_row, dst_row);

    if (col < ncols) {
        dst[row * ncols + dst_row[col]] = col < ntop ? 1 : 0;
    }
}

template<ggml_sort_order order>
static __global__ void k_argsort_f32_f32_i32(const float * x_biased, const float * x, const uint8_t * group_mask,
        float * weights, int * ids, const int ncols, int ncols_pad, int ntop, size_t nb_ids, int n_per_group, int n_groups) {
    int col = threadIdx.x;
    int row = blockIdx.x;

    if (col >= ncols_pad) {
        return;
    }

    extern __shared__ int dst_row[];
    auto x_row = (float *)(dst_row + ncols_pad);

    dst_row[col] = col;
    int ig       = col / n_per_group;
    x_row[col]   = ig < n_groups && group_mask[row * n_groups + ig] ? x_biased[row * ncols + col] : -INFINITY;

    __syncthreads();

    sort<order>(ncols_pad, ncols, col, x_row, dst_row);

    if (col < ntop) {
        weights[row * ntop + col] = 1 / (1 + expf(-x[row * ncols + dst_row[col]]));
        auto row_ids = (int *)((char *) ids + row * nb_ids);
        row_ids[col] = dst_row[col];
    }
}

template<ggml_sort_order order>
static __global__ void k_argsort_biased_f32_f32_i32(const float * x, const float * bias, float * weights, int * ids, const int ncols, int ncols_pad, int ntop,
        size_t nb_ids) {
    int col = threadIdx.x;
    int row = blockIdx.x;

    if (col >= ncols_pad) {
        return;
    }

    extern __shared__ int dst_row[];
    auto x_row = (float *)(dst_row + ncols_pad);

    dst_row[col] = col;
    x_row[col]   = col < ncols ? 1 / (1 + expf(-x[row * ncols + col])) + bias[col] : -INFINITY;

    __syncthreads();

    sort<order>(ncols_pad, ncols, col, x_row, dst_row);

    if (col < ntop) {
        weights[row * ntop + col] = 1 / (1 + expf(-x[row * ncols + dst_row[col]]));
        auto row_ids = (int *)((char *) ids + row * nb_ids);
        row_ids[col] = dst_row[col];
    }
}

template<ggml_sort_order order>
static __global__ void k_openai_f32_f32_i32(const float * x, float * weights, int * ids, const int ncols, int ncols_pad, int ntop,
        size_t nb_ids) {
    int col = threadIdx.x;
    int row = blockIdx.x;

    if (col >= ncols_pad) {
        return;
    }

    extern __shared__ int dst_row[];
    auto x_row = x + row * ncols;

    dst_row[col] = col;

    __syncthreads();

    sort<order>(ncols_pad, ncols, col, x_row, dst_row);

    float max = x_row[dst_row[0]];
    float val = col < ntop ? expf(x_row[dst_row[col]] - max) : 0.0f;
    float sum = warp_reduce_sum(val);
    if (blockDim.x > WARP_SIZE) {
        __syncthreads();
        float * s_sum       = (float *)(dst_row + ncols_pad);
        const int warp_id   = threadIdx.x / WARP_SIZE;
        const int lane_id   = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = sum;
        }
        __syncthreads();
        sum = 0.0f;
        if (lane_id < (static_cast<int>(blockDim.x) / WARP_SIZE)) {
            sum = s_sum[lane_id];
        }
        sum = warp_reduce_sum(sum);
    }
    float norm = 1 / sum;
    if (col < ntop) {
        weights[row * ntop + col] = norm * val;
        auto row_ids = (int *)((char *) ids + row * nb_ids);
        row_ids[col] = dst_row[col];
    }
}

template<ggml_sort_order order>
static __global__ void k_topk_sum(const float * x, const float * bias, float * x_p, float * dst,
        const int ne00, const int ncols, int ncols_pad, int n_top_k) {
    int col = threadIdx.x;
    int row = blockIdx.x;

    if (col >= ncols_pad) {
        return;
    }

    const float * x_row = x + row * ncols;
    extern __shared__ int dst_row[];

    dst_row[col] = col;
    if (bias && x_p) {
        float * x_p_row = x_p + row * ncols;
        if (col < ncols) {
            x_p_row[col] = 1 / (1 + expf(-x_row[col])) + bias[(row * ncols + col) % ne00];
        }
        x_row = x_p_row;
    }

    __syncthreads();

    sort<order>(ncols_pad, ncols, col, x_row, dst_row);

    if (n_top_k == 2) {
        float val = x_row[dst_row[0]] + x_row[dst_row[1]];
        if (col == 0) {
            dst[row] = val;
        }
    } else {
        float val = col < n_top_k ? x_row[dst_row[col]] : 0.0f;
        val = warp_reduce_sum(val);
        if (blockDim.x > WARP_SIZE) {
            __syncthreads();
            float * s_sum       = (float *)(dst_row + ncols_pad);
            const int warp_id   = threadIdx.x / WARP_SIZE;
            const int lane_id   = threadIdx.x % WARP_SIZE;
            if (lane_id == 0) {
                s_sum[warp_id] = val;
            }
            __syncthreads();
            val = 0.0f;
            if (lane_id < (static_cast<int>(blockDim.x) / WARP_SIZE)) {
                val = s_sum[lane_id];
            }
            val = warp_reduce_sum(val);
        }

        if (col == 0) {
            dst[row] = val;
        }
    }
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

template <typename dst_t>
static void argsort_f32_T_cuda(const float * x, dst_t * dst, const int ncols, const int nrows, int ntop,
        ggml_sort_order order, int min_experts, float thresh_experts, cudaStream_t stream) {
    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    const size_t shared_mem = ncols_pad * sizeof(int);

    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    if (order == GGML_SORT_ORDER_ASC) {
        if (min_experts >= 0 && min_experts < ncols && thresh_experts > 0) {
            k_argsort_f32_T<GGML_SORT_ORDER_ASC, store_ser><<<block_nums, block_dims, shared_mem, stream>>>(
                    x, dst, ncols, ncols_pad, ntop, { min_experts, thresh_experts });
        } else {
            k_argsort_f32_T<GGML_SORT_ORDER_ASC, store><<<block_nums, block_dims, shared_mem, stream>>>(
                    x, dst, ncols, ncols_pad, ntop, {});
        }
    } else if (order == GGML_SORT_ORDER_DESC) {
        if (min_experts >= 0 && min_experts < ncols && thresh_experts > 0) {
            k_argsort_f32_T<GGML_SORT_ORDER_DESC, store_ser><<<block_nums, block_dims, shared_mem, stream>>>(
                    x, dst, ncols, ncols_pad, ntop, { min_experts, thresh_experts });
        } else {
            k_argsort_f32_T<GGML_SORT_ORDER_DESC, store><<<block_nums, block_dims, shared_mem, stream>>>(
                    x, dst, ncols, ncols_pad, ntop, {});
        }
    } else {
        GGML_ABORT("fatal error");
    }
}

void argsort_f32_i32_cuda_bitonic(const float *   x,
                                  int *           dst,
                                  const int       ncols,
                                  const int       nrows,
                                  ggml_sort_order order,
                                  cudaStream_t    stream) {
    argsort_f32_T_cuda(x, dst, ncols, nrows, ncols, order, -1, 0.0f, stream);
}

void top_k_f32_i32_cuda_bitonic(const float * x,
                                int *         dst,
                                const int     ncols,
                                const int     nrows,
                                const int     k,
                                cudaStream_t  stream) {
    const int ncols_pad = next_power_of_2(ncols);
    const int block_size = std::min(ncols_pad, 1024);
    const size_t shared_mem = ncols_pad * sizeof(int);

    k_top_k_f32_i32<GGML_SORT_ORDER_DESC><<<nrows, block_size, shared_mem, stream>>>(
            x, dst, ncols, ncols_pad, k);
}

static bool top_k_shared_keys_enabled(const int tile_size) {
    const char * env = std::getenv("LLAMA_DSV41_TOPK_SHARED_KEYS");
    return env && env[0] == '1' && tile_size * (sizeof(int) + sizeof(float)) <=
        ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo;
}

void top_k_f32_i32_cuda_bitonic_tiles_guarded(const float * x,
                                      int *         dst,
                                      const int     ncols,
                                      const int     nrows,
                                      const int     tile_size,
                                      const int     k,
                                      cudaStream_t  stream,const bool reject_nan) {
    top_k_f32_i32_cuda_bitonic_tiles_filtered(x,dst,ncols,nrows,tile_size,k,stream,reject_nan,nullptr);
}

void top_k_f32_i32_cuda_bitonic_tiles_filtered(const float * x,
                                      int *         dst,
                                      const int     ncols,
                                      const int     nrows,
                                      const int     tile_size,
                                      const int     k,
                                      cudaStream_t  stream,const bool reject_nan,const unsigned * only_rows) {
    const int ntiles = (ncols + tile_size - 1) / tile_size;
    const int block_size = std::min(tile_size, 1024);
    const size_t shared_mem = tile_size * sizeof(int);

#ifdef GGML_CUDA_USE_CUB
    if (dsv41_topk_radix_enabled(tile_size, k)) {
        CUDA_SET_SHARED_MEMORY_LIMIT(k_dsv41_topk_radix_tiles<true>, dsv41_topk_radix_smem);
        k_dsv41_topk_radix_tiles<true><<<dim3(ntiles, nrows), 512, dsv41_topk_radix_smem, stream>>>(x, nullptr, dst, ncols, ncols, k,reject_nan,only_rows);
        dsv41_topk_radix_queued.fetch_add(1, std::memory_order_release);return;
    }
#endif
    if (top_k_shared_keys_enabled(tile_size)) {
        const auto kernel = k_top_k_f32_i32_tiles<true, true>;
        CUDA_SET_SHARED_MEMORY_LIMIT(kernel, ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo);
        k_top_k_f32_i32_tiles<true, true><<<dim3(ntiles, nrows), block_size, 2 * shared_mem, stream>>>(
                x, nullptr, dst, ncols, ncols, tile_size, k,reject_nan,only_rows);
        return;
    }

    k_top_k_f32_i32_tiles<true><<<dim3(ntiles, nrows), block_size, shared_mem, stream>>>(
            x, nullptr, dst, ncols, ncols, tile_size, k,reject_nan,only_rows);
}

void top_k_f32_i32_cuda_bitonic_tiles(const float*x,int*dst,int ncols,int nrows,int tile_size,int k,cudaStream_t stream){
    top_k_f32_i32_cuda_bitonic_tiles_guarded(x,dst,ncols,nrows,tile_size,k,stream,false);
}

void top_k_f32_i32_cuda_bitonic_candidates(const float * x,
                                           const int *   candidates,
                                           int *         dst,
                                           const int     ncols,
                                           const int     nrows,
                                           const int     width,
                                           const int     tile_size,
                                           const int     k,
                                           cudaStream_t  stream) {
    top_k_f32_i32_cuda_bitonic_candidates_filtered(x,candidates,dst,ncols,nrows,width,tile_size,k,stream,nullptr);
}

void top_k_f32_i32_cuda_bitonic_candidates_filtered(const float * x,
                                           const int *   candidates,
                                           int *         dst,
                                           const int     ncols,
                                           const int     nrows,
                                           const int     width,
                                           const int     tile_size,
                                           const int     k,
                                           cudaStream_t  stream,const unsigned * only_rows) {
    const int ntiles = (width + tile_size - 1) / tile_size;
    const int block_size = std::min(tile_size, 1024);
    const size_t shared_mem = tile_size * sizeof(int);

#ifdef GGML_CUDA_USE_CUB
    if (dsv41_topk_radix_enabled(tile_size, k)) {
        CUDA_SET_SHARED_MEMORY_LIMIT(k_dsv41_topk_radix_tiles<false>, dsv41_topk_radix_smem);
        k_dsv41_topk_radix_tiles<false><<<dim3(ntiles, nrows), 512, dsv41_topk_radix_smem, stream>>>(x, candidates, dst, ncols, width, k,false,only_rows);
        dsv41_topk_radix_queued.fetch_add(1, std::memory_order_release);return;
    }
#endif
    if (top_k_shared_keys_enabled(tile_size)) {
        const auto kernel = k_top_k_f32_i32_tiles<false, true>;
        CUDA_SET_SHARED_MEMORY_LIMIT(kernel, ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo);
        k_top_k_f32_i32_tiles<false, true><<<dim3(ntiles, nrows), block_size, 2 * shared_mem, stream>>>(
                x, candidates, dst, ncols, width, tile_size, k,false,only_rows);
        return;
    }

    k_top_k_f32_i32_tiles<false><<<dim3(ntiles, nrows), block_size, shared_mem, stream>>>(
            x, candidates, dst, ncols, width, tile_size, k,false,only_rows);
}

static void argsort_f32_u8_cuda(const float * x, uint8_t * dst, const int ncols, const int nrows, int ntop,
        ggml_sort_order order, cudaStream_t stream) {
    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    const size_t shared_mem = ncols_pad * sizeof(int);

    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    if (order == GGML_SORT_ORDER_ASC) {
        k_argsort_f32_u8<GGML_SORT_ORDER_ASC><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad, ntop);
    } else if (order == GGML_SORT_ORDER_DESC) {
        k_argsort_f32_u8<GGML_SORT_ORDER_DESC><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad, ntop);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void argsort_f32_f32_i32_cuda(const float * x_biased, const float * x, const uint8_t * group_mask,
        float * weights, int * ids, const int ncols, const int nrows, int ntop,
        size_t nb_ids, int n_per_group, ggml_sort_order order, cudaStream_t stream) {
    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    const size_t shared_mem = ncols_pad * (sizeof(int) + sizeof(float));

    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    if (order == GGML_SORT_ORDER_ASC) {
        k_argsort_f32_f32_i32<GGML_SORT_ORDER_ASC><<<block_nums, block_dims, shared_mem, stream>>>(
                x_biased, x, group_mask, weights, ids, ncols, ncols_pad, ntop, nb_ids, n_per_group, ncols / n_per_group);
    } else if (order == GGML_SORT_ORDER_DESC) {
        k_argsort_f32_f32_i32<GGML_SORT_ORDER_DESC><<<block_nums, block_dims, shared_mem, stream>>>(
                x_biased, x, group_mask, weights, ids, ncols, ncols_pad, ntop, nb_ids, n_per_group, ncols / n_per_group);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void argsort_biased_f32_f32_i32_cuda(const float * x, const float * bias, float * weights, int * ids, const int ncols, const int nrows, int ntop,
        size_t nb_ids, ggml_sort_order order, cudaStream_t stream) {
    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    const size_t shared_mem = ncols_pad * (sizeof(int) + sizeof(float));

    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    if (order == GGML_SORT_ORDER_ASC) {
        k_argsort_biased_f32_f32_i32<GGML_SORT_ORDER_ASC><<<block_nums, block_dims, shared_mem, stream>>>(
                x, bias, weights, ids, ncols, ncols_pad, ntop, nb_ids);
    } else if (order == GGML_SORT_ORDER_DESC) {
        k_argsort_biased_f32_f32_i32<GGML_SORT_ORDER_DESC><<<block_nums, block_dims, shared_mem, stream>>>(
                x, bias, weights, ids, ncols, ncols_pad, ntop, nb_ids);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void argsort_openai_f32_f32_i32_cuda(const float * x, float * weights, int * ids, const int ncols, const int nrows, int ntop,
        size_t nb_ids, ggml_sort_order order, cudaStream_t stream) {
    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    const size_t shared_mem = (ncols_pad + ncols_pad > WARP_SIZE ? WARP_SIZE : 0) * sizeof(int);

    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    if (order == GGML_SORT_ORDER_ASC) {
        k_openai_f32_f32_i32<GGML_SORT_ORDER_ASC><<<block_nums, block_dims, shared_mem, stream>>>(
                x, weights, ids, ncols, ncols_pad, ntop, nb_ids);
    } else if (order == GGML_SORT_ORDER_DESC) {
        k_openai_f32_f32_i32<GGML_SORT_ORDER_DESC><<<block_nums, block_dims, shared_mem, stream>>>(
                x, weights, ids, ncols, ncols_pad, ntop, nb_ids);
    } else {
        GGML_ABORT("fatal error");
    }
}

void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    float *             dst_d  = (float *) dst->data;
    cudaStream_t        stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

#ifdef GGML_CUDA_USE_CUB
    const int    ncols_pad      = next_power_of_2((int) ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;

    if (shared_mem > max_shared_mem || ncols > 1024) {
        ggml_cuda_pool & pool = ctx.pool();
        argsort_f32_i32_cuda_cub(pool, src0_d, (int *) dst_d, (int) ncols, (int) nrows, order, stream);
    } else {
        argsort_f32_i32_cuda_bitonic(src0_d, (int *) dst_d, (int) ncols, (int) nrows, order, stream);
    }
#else
    argsort_f32_i32_cuda_bitonic(src0_d, (int *) dst_d, (int) ncols, (int) nrows, order, stream);
#endif
}

static void ggml_cuda_op_topk_sum(ggml_backend_cuda_context & ctx, const float * src, const float * bias, float * src_p, float * dst,
        int ne00, int ncols, int nrows, int n_top_k) {
    GGML_ASSERT(n_top_k <= ncols);

    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    const size_t shared_mem = (ncols_pad + WARP_SIZE) * sizeof(int);
    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    k_topk_sum<GGML_SORT_ORDER_DESC><<<block_nums, block_dims, shared_mem, ctx.stream()>>>(
            src, bias, src_p, dst, ne00, ncols, ncols_pad, n_top_k);
}

void cuda_bailingmoev2_experts(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * topk_view) {
    GGML_ASSERT(topk_view->op == GGML_OP_VIEW);

    auto topk = topk_view->src[0];
    GGML_ASSERT(topk && topk->op == GGML_OP_ARGSORT);

    auto masked_probs = topk->src[0];
    GGML_ASSERT(masked_probs && masked_probs->op == GGML_OP_RESHAPE);

    auto set_rows = masked_probs->src[0];
    GGML_ASSERT(set_rows && set_rows->op == GGML_OP_SET_ROWS);

    auto fill = set_rows->src[2];
    GGML_ASSERT(fill && fill->op == GGML_OP_FILL);

    auto selection_groups = fill->src[0];
    GGML_ASSERT(selection_groups && selection_groups->op == GGML_OP_RESHAPE);

    auto selection_input = selection_groups->src[0];
    GGML_ASSERT(selection_input && selection_input->op == GGML_OP_ADD);

    auto probs = selection_input->src[0];
    auto bias  = selection_input->src[1];
    GGML_ASSERT(probs && probs->type == GGML_TYPE_F32);
    GGML_ASSERT(bias && bias->type == GGML_TYPE_F32);

    auto expert_groups_view = set_rows->src[1];
    GGML_ASSERT(expert_groups_view && expert_groups_view->op == GGML_OP_VIEW);

    auto expert_groups = expert_groups_view->src[0];
    GGML_ASSERT(expert_groups && expert_groups->op == GGML_OP_ARGSORT);

    auto group_scores_2d = expert_groups->src[0];
    GGML_ASSERT(group_scores_2d && group_scores_2d->op == GGML_OP_RESHAPE);

    auto group_scores_sum = group_scores_2d->src[0];
    GGML_ASSERT(group_scores_sum && group_scores_sum->op == GGML_OP_SUM_ROWS);

    auto group_scores_3d = group_scores_sum->src[0];
    GGML_ASSERT(group_scores_3d && group_scores_3d->op == GGML_OP_RESHAPE);

    auto grouped_rows = group_scores_3d->src[0];
    GGML_ASSERT(grouped_rows && grouped_rows->op == GGML_OP_GET_ROWS);
    GGML_ASSERT(grouped_rows->src[0] && grouped_rows->src[0]->op == GGML_OP_RESHAPE);
    GGML_ASSERT(grouped_rows->src[0]->src[0] == selection_groups);

    auto grouped_topk = grouped_rows->src[1];
    GGML_ASSERT(grouped_topk && grouped_topk->op == GGML_OP_VIEW);

    const int nrows        = (int) ggml_nrows(probs);
    const int n_groups     = (int) selection_groups->ne[1];
    const int n_top_groups = (int) expert_groups_view->ne[0];
    const int nk           = (int) grouped_topk->ne[0];

    const int ne00 = (int) probs->ne[0];
    const int ne0  = (int) topk_view->ne[0];
    GGML_ASSERT(ggml_is_contiguous(probs));
    GGML_ASSERT(bias->ne[1] == 1);
    GGML_ASSERT(bias->ne[0] == probs->ne[0]);
    GGML_ASSERT(ne0 == dst->ne[1]);
    GGML_ASSERT(ne0 <= ne00);
    GGML_ASSERT(ne00 % n_groups == 0);
    const int n_per_group = ne00 / n_groups;
    GGML_ASSERT(nk <= n_per_group);
    GGML_ASSERT(n_top_groups <= n_groups);

    ggml_cuda_pool_alloc<float> biased_probs(ctx.pool(), (size_t) ne00 * nrows);
    ggml_cuda_pool_alloc<float> group_scores(ctx.pool(), (size_t) nrows * n_groups);
    ggml_cuda_op_topk_sum(ctx, (const float *) probs->data, (const float *) bias->data, biased_probs.get(), group_scores.get(),
            ne00, n_per_group, nrows * n_groups, nk);
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_pool_alloc<uint8_t> group_mask(ctx.pool(), (size_t) nrows * n_groups);
    argsort_f32_u8_cuda(group_scores.get(), group_mask.get(), n_groups, nrows, n_top_groups, GGML_SORT_ORDER_DESC, ctx.stream());
    CUDA_CHECK(cudaGetLastError());

    argsort_f32_f32_i32_cuda(biased_probs.get(), (const float *) probs->data, group_mask.get(),
            (float *) dst->data, (int *) topk->data,
            ne00, nrows, ne0, topk->nb[1], n_per_group, GGML_SORT_ORDER_DESC, ctx.stream());
}

void cuda_glm45moe_experts(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * topk_view) {
    GGML_ASSERT(topk_view->op == GGML_OP_VIEW);

    auto topk     = topk_view->src[0];
    auto topk_src = topk->src[0];
    auto probs    = topk_src->src[0]->src[0];
    auto bias     = topk_src->src[1];

    const int nrows = (int) ggml_nrows(probs);

    const int ne00 = (int) probs->ne[0];
    const int ne0  = (int) topk_view->ne[0];
    GGML_ASSERT(ggml_is_contiguous(probs));
    GGML_ASSERT(bias->ne[1] == 1);
    GGML_ASSERT(bias->ne[0] == probs->ne[0]);
    GGML_ASSERT(ne0 == dst->ne[1]);
    GGML_ASSERT(ne0 <= ne00);

    argsort_biased_f32_f32_i32_cuda((const float *) probs->data, (const float *) bias->data, (float *) dst->data, (int *) topk->data,
            ne00, nrows, ne0, topk->nb[1], GGML_SORT_ORDER_DESC, ctx.stream());
}

void cuda_openai_experts(ggml_backend_cuda_context & ctx, ggml_tensor * topk, ggml_tensor * softmax) {
    GGML_ASSERT(topk->op == GGML_OP_ARGSORT);

    auto probs = topk->src[0];
    const int ntop = (int) softmax->ne[0];

    const int nrows = (int) ggml_nrows(probs);
    const int ne00  = (int) probs->ne[0];
    const int ne0   = (int) softmax->ne[0];
    GGML_ASSERT(ggml_is_contiguous(probs));
    GGML_ASSERT(ggml_is_contiguous(softmax));
    GGML_ASSERT(ne0 <= ne00);
    GGML_ASSERT(ne0 == ntop);

    argsort_openai_f32_f32_i32_cuda((const float *) probs->data, (float *) softmax->data, (int *) topk->data,
            ne00, nrows, ne0, topk->nb[1], GGML_SORT_ORDER_DESC, ctx.stream());
}

// ---------------------------------------------------------------------------
// LLAMADECODE fast exact top-512 select (dsv41-topk-select.cuh). Default OFF.
#include "dsv41-topk-select.cuh"
static std::atomic<int> dsv41_topk_fast_on{0};
static std::atomic<uint64_t> dsv41_topk_fast_queued{0};
static __device__ unsigned long long dsv41_topk_fast_dirty_rows;  // rows sent to the current pipeline
extern "C" void ggml_backend_cuda_dsv41_topk_fast_set(int on) { dsv41_topk_fast_on.store(on ? 1 : 0, std::memory_order_release); }
extern "C" int ggml_backend_cuda_dsv41_topk_fast_enabled(void) { return dsv41_topk_fast_on.load(std::memory_order_acquire); }
extern "C" uint64_t ggml_backend_cuda_dsv41_topk_fast_queued(void) { return dsv41_topk_fast_queued.load(std::memory_order_acquire); }
extern "C" uint64_t ggml_backend_cuda_dsv41_topk_fast_dirty_rows(void) {
    // Caller has synchronized the compute stream (executor: after the tile graph).
    unsigned long long v = 0;
    CUDA_CHECK(cudaMemcpyFromSymbol(&v, dsv41_topk_fast_dirty_rows, sizeof(v)));
    return uint64_t(v);
}

bool top_k_f32_i32_cuda_dsv41_select_eligible(const int ncols, const int nrows, const int k) {
#ifdef GGML_CUDA_USE_CUB
    return dsv41_topk_fast_on.load(std::memory_order_acquire) && nrows > 1 &&
        k == dsv41_topk_select::k && ncols >= dsv41_topk_select::min_cols;
#else
    GGML_UNUSED(ncols); GGML_UNUSED(nrows); GGML_UNUSED(k);
    return false;
#endif
}

#ifdef GGML_CUDA_USE_CUB
using dsv41_SelectSort = cub::BlockRadixSort<uint64_t, 512, 1>;
// One 512-thread block per row. Pass 0 also classifies the row; a NaN/subnormal
// row sets dirty[row]=1 and writes nothing (the current pipeline then runs it).
static __global__ void __launch_bounds__(512) k_dsv41_topk_select512(
        const float * x, int * dst, unsigned * dirty, const int ncols) {
    namespace S = dsv41_topk_select;
    const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t * xr = reinterpret_cast<const uint32_t *>(x) + size_t(row) * ncols;
    __shared__ unsigned hist[256];
    __shared__ unsigned s_flag, s_bin, s_above, s_inbin;
    __shared__ unsigned warp_a[16], warp_r[16];
    __shared__ union {
        dsv41_SelectSort::TempStorage sort;
        uint64_t sel[512];
    } u;
    S::state st{0u, 0u, uint32_t(S::k)};
    for (int shift = 24;; shift -= 8) {
        if (tid < 256) hist[tid] = 0;
        if (tid == 0) s_flag = 0;
        __syncthreads();
        unsigned flag = 0;
        for (int base = 0; base < ncols; base += 512) {
            const int col = base + tid;
            const bool valid = col < ncols;
            const uint32_t bits = valid ? __ldg(xr + col) : 0u;
            if (shift == 24 && valid) flag |= S::classify(bits);
            const uint32_t o = S::ordered(bits);
            const bool in = valid && S::in_region(st, o);
            const unsigned digit = (o >> shift) & 0xffu;
            const unsigned active = __ballot_sync(0xffffffffu, in);
            if (in) {
                const unsigned peers = __match_any_sync(active, digit);
                if (lane == __ffs(peers) - 1) atomicAdd(&hist[digit], unsigned(__popc(peers)));
            }
        }
        if (shift == 24 && flag) atomicOr(&s_flag, flag);
        __syncthreads();
        if (shift == 24) {
            if (s_flag) {
                if (tid == 0) { dirty[row] = 1u; atomicAdd(&dsv41_topk_fast_dirty_rows, 1ull); }
                return;
            }
            if (tid == 0) dirty[row] = 0u;
        }
        // Bins in descending order: thread t holds bin 255-t; inclusive scan.
        unsigned h = 0, incl = 0;
        if (tid < 256) {
            h = hist[255 - tid];
            incl = h;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                const unsigned v = __shfl_up_sync(0xffffffffu, incl, d);
                if (lane >= d) incl += v;
            }
            if (lane == 31) warp_a[warp] = incl;
        }
        __syncthreads();
        if (tid < 256) {
            for (int w = 0; w < warp; ++w) incl += warp_a[w];
            const unsigned excl = incl - h;
            if (excl < st.need && st.need <= incl) { s_bin = 255u - unsigned(tid); s_above = excl; s_inbin = h; }
        }
        __syncthreads();
        const bool done = S::advance(st, shift, s_bin, s_above, s_inbin);
        __syncthreads();  // s_* and hist are rewritten by the next pass
        if (done) break;
    }
    // Index-ordered collection: every value above the region, then the region's
    // lowest-index elements until `need` are taken.
    const unsigned c_above = unsigned(S::k) - st.need;
    unsigned run_a = 0, run_r = 0;
    const unsigned below = (1u << lane) - 1u;
    for (int base = 0; base < ncols; base += 512) {
        const int col = base + tid;
        const bool valid = col < ncols;
        const uint32_t o = S::ordered(valid ? __ldg(xr + col) : 0u);
        const bool a = valid && S::above(st, o), r = valid && S::in_region(st, o);
        const unsigned ba = __ballot_sync(0xffffffffu, a), br = __ballot_sync(0xffffffffu, r);
        if (lane == 0) { warp_a[warp] = __popc(ba); warp_r[warp] = __popc(br); }
        __syncthreads();
        unsigned pa = run_a, pr = run_r, ta = 0, tr = 0;
#pragma unroll
        for (int w = 0; w < 16; ++w) {
            if (w < warp) { pa += warp_a[w]; pr += warp_r[w]; }
            ta += warp_a[w]; tr += warp_r[w];
        }
        pa += __popc(ba & below); pr += __popc(br & below);
        if (a) u.sel[pa] = S::pack(o, col);
        if (r && pr < st.need) u.sel[c_above + pr] = S::pack(o, col);
        run_a += ta; run_r += tr;
        __syncthreads();
    }
    if (run_a != c_above || run_r < st.need) {  // impossible unless the select is wrong: fail loudly
        if (tid == 0) printf("DSV41_TOPK_FAST_INVARIANT row=%d above=%u expected=%u region=%u need=%u\n", row, run_a, c_above, run_r, st.need);
        __syncthreads();
        __trap();
    }
    uint64_t key[1] = { u.sel[tid] };
    __syncthreads();  // sel aliases the sort storage
    dsv41_SelectSort(u.sort).SortDescending(key);
    dst[size_t(row) * S::k + tid] = S::unpack_id(key[0]);
}
#endif  // GGML_CUDA_USE_CUB

void top_k_f32_i32_cuda_dsv41_select512(const float * x, int * dst, unsigned * dirty,
        const int ncols, const int nrows, cudaStream_t stream) {
#ifdef GGML_CUDA_USE_CUB
    k_dsv41_topk_select512<<<nrows, 512, 0, stream>>>(x, dst, dirty, ncols);
    CUDA_CHECK(cudaGetLastError());
    dsv41_topk_fast_queued.fetch_add(1, std::memory_order_release);
#else
    GGML_UNUSED(x); GGML_UNUSED(dst); GGML_UNUSED(dirty); GGML_UNUSED(ncols); GGML_UNUSED(nrows); GGML_UNUSED(stream);
    GGML_ABORT("dsv41 fast top-k needs CUB");
#endif
}
