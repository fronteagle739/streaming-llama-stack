// Native-half Q4_K MMID, PANEL variant (streaming-llama ds4split-q4k-half-panel.cu, selected there by
// LLAMA_DSV41_NATIVE_RHS_PACK=1 and built with DS4PERF_PANEL_COLUMNS=32), ported verbatim for
// ds4-win: four-expert natural-F16 panels plus a once-per-product half RHS gather; same kernels
// and bounds; ds4 entry at the end. No weight requantization.
#include "ds4_q4k_half_common.cuh"
#include <algorithm>
#include <limits>
#include <atomic>
#include <cstdio>
#define DS4PERF_PANEL_COLUMNS 32
#ifndef DS4PERF_PANEL_COLUMNS
#define DS4PERF_PANEL_COLUMNS 16
#endif
// Explicit build variant only. Original builds retain16; no environment route.
static_assert(DS4PERF_PANEL_COLUMNS==32,
              "Unsupported native-half column schedule");
namespace ds4split_q4k_half_panel_impl {
static_assert(sizeof(block_q4_K)==144, "original Q4_K source storage");
static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// Same scale/min and floating expression as actual convert.cu199-226, followed
// by explicit natural-order F16 pair rounding. Must pass282 actual GPU goldens.
static __device__ __forceinline__ half2 q4k_natural_half2(const char*row,int p){
    const auto coordinate=ds4split_q4k_half::natural_pair(p);
    const block_q4_K&b=reinterpret_cast<const block_q4_K*>(row)[coordinate.block];
    const int group=coordinate.group,index=coordinate.payload;
    uint8_t scale,minimum;get_scale_min_k4(group,b.scales,scale,minimum);
    const float dall=__low2half(b.dm),dmin=__high2half(b.dm);
    const float d=dall*scale,m=dmin*minimum;
    const int q0=(group&1)?(b.qs[index]>>4):(b.qs[index]&15);
    const int q1=(group&1)?(b.qs[index+1]>>4):(b.qs[index+1]&15);
    const float v0=d*q0-m,v1=d*q1-m;
    return __floats2half2_rn(v0,v1);
}
static __global__ void build_tasks(const int32_t*ids,const int32_t*bounds,int tokens,int experts,int slots,int bound,int32_t*expert,int32_t*base,int32_t*count,int32_t*task_prefix=nullptr){
    __shared__ int prefix[512];const int t=int(threadIdx.x);
    for(int token=t;token<tokens;token+=512)for(int rank=0;rank<6;++rank){const int e=ids[token*6+rank];if(e<0||e>=experts)asm volatile("trap;");for(int prior=0;prior<rank;++prior)if(ids[token*6+prior]==e)asm volatile("trap;");}
    if(t==0&&(bounds[0]!=0||bounds[experts]!=slots))asm volatile("trap;");
    int n=0;
    if(t<experts){const int a=bounds[t],b=bounds[t+1];if(a<0||b<a||b>slots)asm volatile("trap;");n=(b-a+DS4PERF_PANEL_COLUMNS-1)/DS4PERF_PANEL_COLUMNS;}
    prefix[t]=n;__syncthreads();
    for(int offset=1;offset<512;offset*=2){const int add=t>=offset?prefix[t-offset]:0;__syncthreads();prefix[t]+=add;__syncthreads();}
    if(task_prefix){if(t<experts)task_prefix[t]=t?prefix[t-1]:0;if(t==experts)task_prefix[t]=prefix[experts-1];}
    if(t==0){if(prefix[experts-1]>bound)asm volatile("trap;");*count=prefix[experts-1];}
    if(t<experts){const int first=t?prefix[t-1]:0;for(int j=0;j<n;++j){if(first+j>=bound)asm volatile("trap;");expert[first+j]=t;base[first+j]=j*DS4PERF_PANEL_COLUMNS;}}
}
static __global__ void loader_golden(const char*source,half2*out,int k,int rows){
    const int64_t at=int64_t(blockIdx.x)*blockDim.x+threadIdx.x,total=int64_t(k/2)*rows;
    if(at<total){const int row=int(at/(k/2)),pair=int(at%(k/2));out[at]=q4k_natural_half2(source+int64_t(row)*(k/256)*sizeof(block_q4_K),2*pair);}
}
// The golden loader's natural Q4_K -> F16 conversion, applied only to a
// bounded panel of experts with nonzero routing counts. Unused sparse fixture
// pages are never read. This is dequantization, not weight requantization.
static __global__ void load_native_panel(const char*source,half2*out,const int32_t*bounds,
        int first,int count,int k,int rows,int64_t expert_stride){
    const int64_t at=int64_t(blockIdx.x)*blockDim.x+threadIdx.x,pairs=int64_t(k)*rows/2;
    const int local=int(at/pairs);if(local>=count)return;
    const int expert=first+local;if(bounds[expert]==bounds[expert+1])return;
    const int64_t within=at-int64_t(local)*pairs;
    const int row=int(within/(k/2)),pair=int(within%(k/2));
    out[at]=q4k_natural_half2(source+int64_t(expert)*expert_stride+int64_t(row)*(k/256)*sizeof(block_q4_K),2*pair);
}
template <typename T, int rows_per_block, int cols_per_block, int nwarps>
__launch_bounds__(ggml_cuda_get_physical_warp_size()*nwarps, 1)
static __global__ void mul_mat_q4k_half_ids(
        const char * __restrict__ x, const float * __restrict__ y,
        const int32_t * __restrict__ ids_src_compact, const int32_t * __restrict__ ids_dst_compact,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ task_expert,
        const int32_t * __restrict__ task_col_base, const int32_t * __restrict__ task_count, float * __restrict__ dst,
        const int ncols, const int ncols_dst_total, const int nchannels_dst, const int stride_row, const int stride_col_y, const int stride_col_dst,
        const int channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const uint3 sis1_fd, const uint3 nch_fd, const int32_t*task_prefix,int first,int count,const half* rhs_packed) {
// TODO: handle this in a consistent and simpler way after AMD MFMA support has been added
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
#if defined(AMD_WMMA_AVAILABLE)
    if constexpr (!(std::is_same_v<T, half2> || std::is_same_v<T, nv_bfloat162>) || rows_per_block != MMF_ROWS_PER_BLOCK) {NO_DEVICE_CODE;} else {
    typedef tile<16, 8,  T,     get_input_data_layout()> tile_A;
    typedef tile<16, 8,  T,     get_input_data_layout()> tile_B;
    typedef tile<16, 16, float, DATA_LAYOUT_J_MAJOR>     tile_C;
#elif defined(AMD_MFMA_AVAILABLE)
    if constexpr (rows_per_block != MMF_ROWS_PER_BLOCK_CDNA) {NO_DEVICE_CODE;} else {
    typedef tile<16, 8,  T,     DATA_LAYOUT_I_MAJOR> tile_A;
    typedef tile<16, 8,  T,     DATA_LAYOUT_I_MAJOR> tile_B;
    typedef tile<16, 16, float, DATA_LAYOUT_J_MAJOR> tile_C;
#else
#ifdef VOLTA_MMA_AVAILABLE
    if constexpr (!std::is_same_v<T, half2> || rows_per_block != MMF_ROWS_PER_BLOCK) {NO_DEVICE_CODE;} else {
    typedef tile<32, 4, T,     DATA_LAYOUT_I_MAJOR>          tile_A;
    typedef tile< 8, 4, T,     DATA_LAYOUT_I_MAJOR_MIRRORED> tile_B;
    typedef tile<32, 8, float, DATA_LAYOUT_I_MAJOR>          tile_C;
#else
    if constexpr (rows_per_block != MMF_ROWS_PER_BLOCK) {NO_DEVICE_CODE;} else {
    typedef tile<16, 8, T>     tile_A;
    typedef tile<8,  8, T>     tile_B;
    typedef tile<16, 8, float> tile_C;
#endif // VOLTA_MMA_AVAILABLE
#endif // defined(AMD_WMMA_AVAILABLE)
    if constexpr (!tile_A::supported() || !tile_B::supported() || !tile_C::supported()) {
        NO_DEVICE_CODE;
        return;
    }


    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int tile_k_padded = warp_size + mmf_get_padding();
    constexpr int ntA = rows_per_block / tile_A::I;
    constexpr int ntB = (cols_per_block + tile_B::I - 1) / tile_B::I;

    static_assert(std::is_same_v<T, half2>, "Q4_K family uses actual half2 MMA inputs");
    const int task_begin=task_prefix[first],task_end=task_prefix[first+count];
    for(int task=task_begin+int(blockIdx.y);task<task_end;task+=int(gridDim.y)){
    const char* const panel_x=x;float* const panel_dst=dst;
    const int row0        = blockIdx.x * rows_per_block;

    const int expert_idx = task_expert[task];
    const int expert_start = expert_bounds[expert_idx];
    const int expert_end   = expert_bounds[expert_idx + 1];
    const int ncols_expert = expert_end - expert_start;

    const int col_base = task_col_base[task];
    if (col_base < 0 || col_base >= ncols_expert) asm volatile("trap;");

    GGML_UNUSED(channel_ratio);if(stride_sample_y!=0)asm volatile("trap;");

    const int channel_x   = expert_idx-first;
    const int sample_dst  = 0;
    const int sample_x    = sample_dst / sample_ratio;
    const int sample_y    = sample_dst;

    x   += int64_t(sample_x) * stride_sample_x + int64_t(channel_x) * stride_channel_x + int64_t(row0) * stride_row;
    y   += int64_t(sample_y)  *stride_sample_y;
    dst += int64_t(sample_dst)*stride_sample_dst;

    const int32_t * ids_src_expert = ids_src_compact + expert_start;
    const int32_t * ids_dst_expert = ids_dst_compact + expert_start;

    extern __shared__ char data_mmv[];
    char * compute_base = data_mmv;

    //const float2 * y2 = (const float2 *) y;

    tile_C C[ntA][ntB];

    T * tile_xy = (T *) compute_base + threadIdx.y*(tile_A::I * tile_k_padded);

    for (int col = threadIdx.y*warp_size + threadIdx.x; col < ncols; col += nwarps*warp_size) {
        tile_A A[ntA][warp_size / tile_A::J];
#pragma unroll
        for (int itA = 0; itA < ntA; ++itA) {
#pragma unroll
            for (int i = 0; i < tile_A::I; ++i) {
                tile_xy[i*tile_k_padded + threadIdx.x] = reinterpret_cast<const half2*>(x + int64_t(itA*tile_A::I + i)*stride_row)[col];
            }
#pragma unroll
            for (int k0 = 0; k0 < warp_size; k0 += tile_A::J) {
                load_ldmatrix(A[itA][k0/tile_A::J], tile_xy + k0, tile_k_padded);
            }
        }

        if constexpr (std::is_same_v<T, float>) {
            float vals_buf[2][tile_B::I];
            auto gather_tile = [&](int tile_idx_local, float *vals) {
#pragma unroll
                for (int j0 = 0; j0 < tile_B::I; ++j0) {
                    const int j = j0 + tile_idx_local*tile_B::I;
                    const int global_j = col_base + j;
                    float val = 0.0f;
                    if (j < cols_per_block && global_j < ncols_expert) {
                        const int src_entry = ids_src_expert[global_j];
                        const uint2 qrm = fast_div_modulo((uint32_t) src_entry, sis1_fd);
                        const int token   = (int) qrm.x;
                        const int channel = (int) qrm.y;
                        if (token < ncols_dst_total) {
                            val = y[channel*stride_channel_y + token*stride_col_y + col];
                        }
                    }
                    vals[j0] = val;
                }
            };

            gather_tile(0, vals_buf[0]);

            int curr_buf = 0;
            int next_buf = 1;
#pragma unroll
            for (int itB = 0; itB < ntB; ++itB) {
#pragma unroll
                for (int j0 = 0; j0 < tile_B::I; ++j0) {
                    tile_xy[j0*tile_k_padded + threadIdx.x] = vals_buf[curr_buf][j0];
                }

                if (itB + 1 < ntB) {
                    gather_tile(itB + 1, vals_buf[next_buf]);
                }

#pragma unroll
                for (int k0 = 0; k0 < warp_size; k0 += tile_B::J) {
                    tile_B B;
                    load_ldmatrix(B, tile_xy + k0, tile_k_padded);
#pragma unroll
                    for (int itA = 0; itA < ntA; ++itA) {
                        mma(C[itA][itB], A[itA][k0/tile_B::J], B);
                    }
                }

                if (itB + 1 < ntB) {
                    curr_buf ^= 1;
                    next_buf ^= 1;
                }
            }
        } else if constexpr (std::is_same_v<T, half2> || std::is_same_v<T, nv_bfloat162>) {
            // Shared GET_ROWS has already applied the identical scalar RN half
            // conversion once. Keep the MMF MMA and warp reduction unchanged.
            T vals_buf[2][tile_B::I];
            auto gather_tile = [&](int tile_idx_local, T *vals) {
#pragma unroll
                for (int j0 = 0; j0 < tile_B::I; ++j0) {
                    const int j = j0 + tile_idx_local*tile_B::I;
                    const int global_j = col_base + j;
                    T value = __float2half2_rn(0.0f);
                    if (j < cols_per_block && global_j < ncols_expert) {
                        value = reinterpret_cast<const T*>(rhs_packed)[int64_t(expert_start+global_j)*ncols+col];
                    }
                    vals[j0] = value;
                }
            };

            if (ntB > 0) {
                gather_tile(0, vals_buf[0]);
            }

            int curr_buf = 0;
            int next_buf = 1;
#pragma unroll
            for (int itB = 0; itB < ntB; ++itB) {
#pragma unroll
                for (int j0 = 0; j0 < tile_B::I; ++j0) {
                    tile_xy[j0*tile_k_padded + threadIdx.x]=vals_buf[curr_buf][j0];
                }

                if (itB + 1 < ntB) {
                    gather_tile(itB + 1, vals_buf[next_buf]);
                }

#pragma unroll
                for (int k0 = 0; k0 < warp_size; k0 += tile_B::J) {
                    tile_B B;
                    load_ldmatrix(B, tile_xy + k0, tile_k_padded);
#pragma unroll
                    for (int itA = 0; itA < ntA; ++itA) {
                        mma(C[itA][itB], A[itA][k0/tile_B::J], B);
                    }
                }

                if (itB + 1 < ntB) {
                    curr_buf ^= 1;
                    next_buf ^= 1;
                }
            }
        } else {
            static_assert(std::is_same_v<T, void>, "unsupported type");
        }
    }

    float * buf_iw = (float *) compute_base;
    constexpr int kiw = nwarps*rows_per_block + mmf_get_padding();

    if (nwarps > 1) {
        __syncthreads();
    }
#pragma unroll
    for (int itB = 0; itB < ntB; ++itB) {
#pragma unroll
        for (int itA = 0; itA < ntA; ++itA) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = threadIdx.y*rows_per_block + itA*tile_C::I + tile_C::get_i(l);
                const int j = itB*tile_C::J + tile_C::get_j(l);
                buf_iw[j*kiw + i] = C[itA][itB].x[l];
            }
        }
    }

    if (nwarps > 1) {
        __syncthreads();
    }

#pragma unroll
    for (int j0 = 0; j0 < cols_per_block; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j0 + nwarps > cols_per_block && j >= cols_per_block) {
            continue;
        }

        float sum[rows_per_block/warp_size] = {0.0f};
        static_assert((rows_per_block % warp_size) == 0, "rows_per_block must be a multiple of warp_size.");
#pragma unroll
        for (int i0 = 0; i0 < nwarps*rows_per_block; i0 += rows_per_block) {
#pragma unroll
            for (int i1 = 0; i1 < sizeof(sum)/sizeof(sum[0]); ++i1) {
                const int i = i0 + i1*warp_size + threadIdx.x;

                sum[i1] += buf_iw[j * kiw + i];
            }
        }

        const int global_j = col_base + j;
        if (j < cols_per_block && global_j < ncols_expert && nchannels_dst > 0) {
            const int dst_entry = ids_dst_expert[global_j];
            const uint2 qrm = fast_div_modulo((uint32_t) dst_entry, nch_fd);
            const int token = (int) qrm.x;
            if (token < ncols_dst_total) {
                const int slot = (int) qrm.y;
#pragma unroll
                for (int i0 = 0; i0 < sizeof(sum)/sizeof(sum[0]); ++i0) {
                    dst[slot * stride_channel_dst + token * stride_col_dst + row0 + i0*warp_size + threadIdx.x] = sum[i0];
                }
            }
        }
    }
    __syncthreads();
    x=panel_x;dst=panel_dst;
    } // next independent native MMF task
    }
#else
    GGML_UNUSED_VARS(x, y, ids_src_compact, ids_dst_compact, expert_bounds, dst,
        ncols, ncols_dst_total, nchannels_dst, stride_row, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, sis1_fd, nch_fd);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
}

} // namespace ds4split_q4k_half_panel_impl

// ---- ds4-win entry (raw pointers; mirrors ggml_cuda_mul_mat_q4k_half_panel_id_workspace) ----
// W: expert-major Q4_K [experts][M][K/256 blocks]. X: F32 [tokens][channels][K], channels 1 or 6.
// ids [tokens][6]; out F32 [tokens][6][M]. Calls over 4096 tokens are split. Refuses (nonzero)
// anything outside llama's bounds; never falls back.
namespace {
/* llama's get_rows_cuda(F32 -> F16) gather: dst row i = half_rn(src row src[i]). */
__global__ void q4khp_pack_rhs(const float *x, const int32_t *src, half *dst, int k, int64_t rows) {
    const int64_t at = int64_t(blockIdx.x) * blockDim.x + threadIdx.x, total = rows * k;
    if (at >= total) return;
    const int64_t row = at / k, col = at % k;
    dst[at] = __float2half(x[int64_t(src[row]) * k + col]);
}
void *g_q4khp_ws = nullptr;
size_t g_q4khp_ws_bytes = 0;
}
static int q4khp_launch(const char *w, const float *x, const int32_t *ids, float *out, int M, int K,
                        int tokens, int experts, int channels, cudaStream_t stream) {
    using namespace ds4split_q4k_half_panel_impl;
    const int k = K, n = M;
    const auto p = ds4split_q4k_half::plan(tokens, experts);
    constexpr int panel_experts = 4;
    /* One device workspace: plan maps | task_prefix | F16 panel | packed half RHS. */
    const uint64_t prefix_off = (p.bytes + 255) & ~uint64_t(255);
    const uint64_t panel_off = (prefix_off + uint64_t(experts + 1) * 4 + 255) & ~uint64_t(255);
    const uint64_t rhs_off = panel_off + uint64_t(std::min(experts, panel_experts)) * k * n * 2;
    const uint64_t total = rhs_off + uint64_t(tokens) * 6 * k * 2;
    if (g_q4khp_ws_bytes < total) {
        if (g_q4khp_ws) cudaFree(g_q4khp_ws);
        g_q4khp_ws = nullptr; g_q4khp_ws_bytes = 0;
        if (cudaMalloc(&g_q4khp_ws, (size_t)total) != cudaSuccess) { (void)cudaGetLastError(); return -2; }
        g_q4khp_ws_bytes = (size_t)total;
        fprintf(stderr, "DS4_Q4K_HALF_PANEL workspace %.1f MiB (tokens=%d experts=%d k=%d)\n",
                (double)total / 1048576.0, tokens, experts, k);
    }
    char *b = static_cast<char *>(g_q4khp_ws);
    auto src = reinterpret_cast<int32_t *>(b + p.src_map), dest = reinterpret_cast<int32_t *>(b + p.dst_map),
         bounds = reinterpret_cast<int32_t *>(b + p.expert_bounds), task_expert = reinterpret_cast<int32_t *>(b + p.task_expert),
         task_base = reinterpret_cast<int32_t *>(b + p.task_col_base), task_count = reinterpret_cast<int32_t *>(b + p.task_count);
    int32_t *task_prefix = reinterpret_cast<int32_t *>(b + prefix_off);
    half *panel = reinterpret_cast<half *>(b + panel_off), *packed_rhs = reinterpret_cast<half *>(b + rhs_off);
    ggml_cuda_launch_mm_ids_helper(ids, src, dest, bounds, experts, tokens, 6, channels, 6, channels, stream);
    build_tasks<<<1, 512, 0, stream>>>(ids, bounds, tokens, experts, int(p.slots), int(p.maximum_tasks),
                                       task_expert, task_base, task_count, task_prefix);
    if (cudaGetLastError() != cudaSuccess) return -3;
    const int64_t rhs_rows = int64_t(tokens) * 6, rhs_elems = rhs_rows * k;
    q4khp_pack_rhs<<<unsigned((rhs_elems + 255) / 256), 256, 0, stream>>>(x, src, packed_rhs, k, rhs_rows);
    if (cudaGetLastError() != cudaSuccess) return -3;
    int device = 0; cudaGetDevice(&device);
    const int cc = ggml_cuda_info().devices[device].cc, warp = ggml_cuda_info().devices[device].warp_size;
    if (warp != 32) return -4;
    const int ncols_x = k / 2;
    int nwarps_best = 1, niter_best = (ncols_x + warp * 2 - 1) / (warp * 2);
    for (int nw = 2; nw <= mmf_get_max_block_size(cc) / warp; ++nw) {
        const int it = (ncols_x + nw * warp * 2 - 1) / (nw * warp * 2);
        if (it < niter_best) { niter_best = it; nwarps_best = nw; }
    }
    const int shared_iter = nwarps_best * 16 * (warp + mmf_get_padding(cc)) * 4;
    const int shared_combine = DS4PERF_PANEL_COLUMNS * (nwarps_best * 32 + mmf_get_padding(cc)) * 4;
    const int shared = std::max(shared_iter, shared_combine) + 16 * int(sizeof(int));
    const dim3 grid(n / 32, 16, 1), block(warp, nwarps_best, 1);
    const uint3 sis = init_fastdiv_values(unsigned(channels)), nch = init_fastdiv_values(6);
    const int64_t expert_stride = int64_t(n) * (k / 256) * int64_t(sizeof(block_q4_K));
    const int stride_col_dst = 6 * n, stride_channel_dst = n;
#define DS4_Q4KHP_LAUNCH(NW) case NW: { \
        if (shared > 48 * 1024) { \
            int maximum = 0; cudaDeviceGetAttribute(&maximum, cudaDevAttrMaxSharedMemoryPerBlockOptin, device); \
            if (shared > maximum) return -5; \
            cudaFuncSetAttribute(mul_mat_q4k_half_ids<half2, 32, DS4PERF_PANEL_COLUMNS, NW>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared); } \
        mul_mat_q4k_half_ids<half2, 32, DS4PERF_PANEL_COLUMNS, NW><<<grid, block, shared, stream>>>( \
            reinterpret_cast<const char *>(panel), x, src, dest, bounds, task_expert, task_base, task_count, out, \
            k / 2, tokens, 6, k * 2, 0, stride_col_dst, 0, k * n * 2, 0, stride_channel_dst, 1, 0, 0, 0, sis, nch, \
            task_prefix, first, count, packed_rhs); break; }
    for (int first = 0; first < experts; first += panel_experts) {
        const int count = std::min(panel_experts, experts - first);
        const int64_t pairs = int64_t(count) * k * n / 2;
        load_native_panel<<<unsigned((pairs + 255) / 256), 256, 0, stream>>>(w, reinterpret_cast<half2 *>(panel),
            bounds, first, count, k, n, expert_stride);
        if (cudaGetLastError() != cudaSuccess) return -7;
        switch (nwarps_best) {
            DS4_Q4KHP_LAUNCH(1) DS4_Q4KHP_LAUNCH(2) DS4_Q4KHP_LAUNCH(3) DS4_Q4KHP_LAUNCH(4)
            DS4_Q4KHP_LAUNCH(5) DS4_Q4KHP_LAUNCH(6) DS4_Q4KHP_LAUNCH(7) DS4_Q4KHP_LAUNCH(8)
            default: return -6;
        }
    }
#undef DS4_Q4KHP_LAUNCH
    static std::atomic<unsigned> logged{0};
    const unsigned bit = k == 5120 ? 1u : 2u;
    if (!(logged.fetch_or(bit, std::memory_order_relaxed) & bit))
        fprintf(stderr, "DS4PERF_EXPERT_SCHEDULE columns=%d rows=32 k=%d n=%d tokens=%d warps=%d shared_bytes=%d "
                        "task_bound=%llu rhs_prepacked_half=1 panel_experts=4 (ds4-win panel port)\n",
                DS4PERF_PANEL_COLUMNS, k, n, tokens, nwarps_best, shared, (unsigned long long)p.maximum_tasks);
    return cudaGetLastError() == cudaSuccess ? 0 : -7;
}

extern "C" int ds4_q4k_half_panel_moe(const void *W, const float *X, const int32_t *ids, float *out, int M, int K,
                                      int tokens, int experts, int channels, cudaStream_t stream) {
    if (!W || !X || !ids || !out || tokens < 1 || (channels != 1 && channels != 6)) return -1;
    if (!((K == 5120 && M == 2304) || (K == 2304 && M == 5120))) return -1;   /* llama's bounds */
    if (experts < 6 || experts > ds4split_q4k_half::maximum_experts) return -1;
    int device = 0; cudaGetDevice(&device);
    if (ggml_cuda_info().devices[device].cc != 1200) return -1;
    for (int first = 0; first < tokens; first += ds4split_q4k_half::maximum_tokens) {
        const int n = std::min(tokens - first, ds4split_q4k_half::maximum_tokens);
        const int rc = q4khp_launch(static_cast<const char *>(W), X + int64_t(first) * channels * K, ids + int64_t(first) * 6,
                                    out + int64_t(first) * 6 * M, M, K, n, experts, channels, stream);
        if (rc) return rc;
    }
    return 0;
}
