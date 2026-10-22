// SOURCE candidate: separate explicit Q4_K+half-RHS MMID family.
// No original operator or dispatch changes. No full-matrix dequant in this path.
#include "ds4split-q4k-half-mmid.cuh"
#include "mmf.cuh"
#include "mmid.cuh"
#include "dequantize.cuh"
#include <algorithm>
#include <limits>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifndef DS4PERF_NATIVE_COLUMNS
#define DS4PERF_NATIVE_COLUMNS 16
#endif
// Explicit build variant only. Original builds retain16; no environment route.
static_assert(DS4PERF_NATIVE_COLUMNS==16||DS4PERF_NATIVE_COLUMNS==32||DS4PERF_NATIVE_COLUMNS==64,
              "Unsupported native-half column schedule");
#ifndef DS4PERF_NATIVE_LOADER_UNROLL
#define DS4PERF_NATIVE_LOADER_UNROLL 0
#endif
static_assert(DS4PERF_NATIVE_LOADER_UNROLL==0||DS4PERF_NATIVE_LOADER_UNROLL==1,"Unsupported loader unroll schedule");
namespace ds4split_q4k_half_impl {
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
static __global__ void build_tasks(const int32_t*ids,const int32_t*bounds,int tokens,int experts,int slots,int bound,int32_t*expert,int32_t*base,int32_t*count){
    __shared__ int prefix[512];const int t=int(threadIdx.x);
    for(int token=t;token<tokens;token+=512)for(int rank=0;rank<6;++rank){const int e=ids[token*6+rank];if(e<0||e>=experts)asm volatile("trap;");for(int prior=0;prior<rank;++prior)if(ids[token*6+prior]==e)asm volatile("trap;");}
    if(t==0&&(bounds[0]!=0||bounds[experts]!=slots))asm volatile("trap;");
    int n=0;
    if(t<experts){const int a=bounds[t],b=bounds[t+1];if(a<0||b<a||b>slots)asm volatile("trap;");n=(b-a+DS4PERF_NATIVE_COLUMNS-1)/DS4PERF_NATIVE_COLUMNS;}
    prefix[t]=n;__syncthreads();
    for(int offset=1;offset<512;offset*=2){const int add=t>=offset?prefix[t-offset]:0;__syncthreads();prefix[t]+=add;__syncthreads();}
    if(t==0){if(prefix[experts-1]>bound)asm volatile("trap;");*count=prefix[experts-1];}
    if(t<experts){const int first=t?prefix[t-1]:0;for(int j=0;j<n;++j){if(first+j>=bound)asm volatile("trap;");expert[first+j]=t;base[first+j]=j*DS4PERF_NATIVE_COLUMNS;}}
}
static __global__ void loader_golden(const char*source,half2*out,int k,int rows){
    const int64_t at=int64_t(blockIdx.x)*blockDim.x+threadIdx.x,total=int64_t(k/2)*rows;
    if(at<total){const int row=int(at/(k/2)),pair=int(at%(k/2));out[at]=q4k_natural_half2(source+int64_t(row)*(k/256)*sizeof(block_q4_K),2*pair);}
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
        const uint3 sis1_fd, const uint3 nch_fd) {
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
    const int task = int(blockIdx.y);
    if (task >= *task_count) return;
    const int row0        = blockIdx.x * rows_per_block;

    const int expert_idx = task_expert[task];
    const int expert_start = expert_bounds[expert_idx];
    const int expert_end   = expert_bounds[expert_idx + 1];
    const int ncols_expert = expert_end - expert_start;

    const int col_base = task_col_base[task];
    if (col_base < 0 || col_base >= ncols_expert) asm volatile("trap;");

    GGML_UNUSED(channel_ratio);

    const int channel_x   = expert_idx;
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
#if DS4PERF_NATIVE_LOADER_UNROLL == 1
#pragma unroll 1
#else
#pragma unroll
#endif
            for (int i = 0; i < tile_A::I; ++i) {
                tile_xy[i*tile_k_padded + threadIdx.x] = q4k_natural_half2(x + int64_t(itA*tile_A::I + i)*stride_row, 2*col);
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
            // Wider schedules keep the SAME half conversion at the input
            // boundary, but store its result in the existing prefetch buffers.
            // This halves temporary register storage; no F32 MMA sum changes.
            using rhs_value=std::conditional_t<(cols_per_block>16),T,float2>;
            rhs_value vals_buf[2][tile_B::I];
            auto gather_tile = [&](int tile_idx_local, rhs_value *vals) {
#pragma unroll
                for (int j0 = 0; j0 < tile_B::I; ++j0) {
                    const int j = j0 + tile_idx_local*tile_B::I;
                    const int global_j = col_base + j;
                    float2 tmp = make_float2(0.0f, 0.0f);
                    if (j < cols_per_block && global_j < ncols_expert) {
                        const int src_entry = ids_src_expert[global_j];
                        const uint2 qrm = fast_div_modulo((uint32_t) src_entry, sis1_fd);
                        const int token   = (int) qrm.x;
                        const int channel = (int) qrm.y;
                        if (token < ncols_dst_total) {
                            tmp = *(const float2*) &y[channel*stride_channel_y + 2*(token*stride_col_y + col)];
                        }
                    }
                    if constexpr(cols_per_block>16)vals[j0]=ggml_cuda_cast<T>(tmp);
                    else vals[j0] = tmp;
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
                    if constexpr(cols_per_block>16)tile_xy[j0*tile_k_padded + threadIdx.x]=vals_buf[curr_buf][j0];
                    else tile_xy[j0*tile_k_padded + threadIdx.x]=ggml_cuda_cast<T>(vals_buf[curr_buf][j0]);
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
            return;
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
    }
#else
    GGML_UNUSED_VARS(x, y, ids_src_compact, ids_dst_compact, expert_bounds, dst,
        ncols, ncols_dst_total, nchannels_dst, stride_row, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, sis1_fd, nch_fd);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
}

} // namespace ds4split_q4k_half_impl
bool ggml_cuda_supports_q4k_half_mmid(const ggml_tensor*w,const ggml_tensor*x,const ggml_tensor*ids,const ggml_tensor*dst,int cc){
    if(!w||!x||!ids||!dst||cc!=1200||!w->data||!x->data||!ids->data||!dst->data)return false;
    if(w->type!=GGML_TYPE_Q4_K||x->type!=GGML_TYPE_F32||ids->type!=GGML_TYPE_I32||dst->type!=GGML_TYPE_F32)return false;
    const int64_t k=w->ne[0],n=w->ne[1],e=w->ne[2],t=ids->ne[1];
    if(!((k==5120&&n==2304)||(k==2304&&n==5120))||e<6||e>384||t<1||t>4096)return false;
    if(w->ne[3]!=1||x->ne[0]!=k||(x->ne[1]!=1&&x->ne[1]!=6)||x->ne[2]!=t||x->ne[3]!=1||ids->ne[0]!=6||ids->ne[2]!=1||ids->ne[3]!=1||dst->ne[0]!=n||dst->ne[1]!=6||dst->ne[2]!=t||dst->ne[3]!=1)return false;
    if(!ggml_is_contiguous(w)||!ggml_is_contiguous(x)||!ggml_is_contiguous(ids)||!ggml_is_contiguous(dst)||ggml_get_op_params_i32(dst,0)!=0)return false;
    return w->nb[0]==sizeof(block_q4_K)&&w->nb[1]==size_t(k/256)*sizeof(block_q4_K)&&w->nb[2]==6635520&&k%256==0&&n%32==0;
}
void ggml_cuda_q4k_half_prepare_tasks(const int32_t*ids,int tokens,int experts,int channels,void*workspace,size_t bytes,cudaStream_t stream){
    using namespace ds4split_q4k_half_impl;
    const auto p=ds4split_q4k_half::plan(tokens,experts);GGML_ASSERT(ids&&workspace&&bytes>=p.bytes&&(channels==1||channels==6)&&reinterpret_cast<uintptr_t>(workspace)%4==0);
    char*b=static_cast<char*>(workspace);auto src=reinterpret_cast<int32_t*>(b+p.src_map),dest=reinterpret_cast<int32_t*>(b+p.dst_map),bounds=reinterpret_cast<int32_t*>(b+p.expert_bounds),task_expert=reinterpret_cast<int32_t*>(b+p.task_expert),task_base=reinterpret_cast<int32_t*>(b+p.task_col_base),task_count=reinterpret_cast<int32_t*>(b+p.task_count);
    ggml_cuda_launch_mm_ids_helper(ids,src,dest,bounds,experts,tokens,6,channels,6,channels,stream);CUDA_CHECK(cudaGetLastError());
    build_tasks<<<1,512,0,stream>>>(ids,bounds,tokens,experts,int(p.slots),int(p.maximum_tasks),task_expert,task_base,task_count);CUDA_CHECK(cudaGetLastError());
}
void ggml_cuda_mul_mat_q4k_half_id_workspace(ggml_backend_cuda_context&ctx,const ggml_tensor*w,const ggml_tensor*x,const ggml_tensor*ids,ggml_tensor*dst,void*workspace,size_t bytes,ds4split_q4k_half_launch*witness){
    const char * panel=std::getenv("LLAMA_DSV41_NATIVE_RHS_PACK");
    GGML_ASSERT(!panel||!std::strcmp(panel,"0")||!std::strcmp(panel,"1"));
    if(panel&&!std::strcmp(panel,"1")){
        GGML_ASSERT(DS4PERF_NATIVE_COLUMNS==16&&DS4PERF_NATIVE_LOADER_UNROLL==0);
        ggml_cuda_mul_mat_q4k_half_panel_id_workspace(ctx,w,x,ids,dst,workspace,bytes,witness);
        return;
    }
    using namespace ds4split_q4k_half_impl;
    const int cc=ggml_cuda_info().devices[ctx.device].cc;GGML_ASSERT(ggml_cuda_supports_q4k_half_mmid(w,x,ids,dst,cc));
    const int k=int(w->ne[0]),n=int(w->ne[1]),experts=int(w->ne[2]),tokens=int(ids->ne[1]);const auto p=ds4split_q4k_half::plan(tokens,experts);
    GGML_ASSERT(workspace&&bytes>=p.bytes&&reinterpret_cast<uintptr_t>(workspace)%4==0);
    char*b=static_cast<char*>(workspace);auto src=reinterpret_cast<int32_t*>(b+p.src_map),dest=reinterpret_cast<int32_t*>(b+p.dst_map),bounds=reinterpret_cast<int32_t*>(b+p.expert_bounds),task_expert=reinterpret_cast<int32_t*>(b+p.task_expert),task_base=reinterpret_cast<int32_t*>(b+p.task_col_base),task_count=reinterpret_cast<int32_t*>(b+p.task_count);
    ggml_cuda_q4k_half_prepare_tasks(static_cast<const int32_t*>(ids->data),tokens,experts,int(x->ne[1]),workspace,bytes,ctx.stream());
    const int ncols_x=k/2,warp=ggml_cuda_info().devices[ctx.device].warp_size;GGML_ASSERT(warp==32);
    int nwarps_best=1,niter_best=(ncols_x+warp*2-1)/(warp*2);
    for(int nw=2;nw<=mmf_get_max_block_size(cc)/warp;++nw){const int it=(ncols_x+nw*warp*2-1)/(nw*warp*2);if(it<niter_best){niter_best=it;nwarps_best=nw;}}
    const int shared_iter=nwarps_best*16*(warp+mmf_get_padding(cc))*4;
    const int shared_combine=DS4PERF_NATIVE_COLUMNS*(nwarps_best*32+mmf_get_padding(cc))*4;
    const int shared=std::max(shared_iter,shared_combine)+16*int(sizeof(int));
    const dim3 grid(n/32,unsigned(p.maximum_tasks),1),block(warp,nwarps_best,1);
    const uint3 sis=init_fastdiv_values(unsigned(x->nb[2]/x->nb[1])),nch=init_fastdiv_values(6);
#define DS4SPLIT_Q4K_HALF_LAUNCH(NW) case NW:{ \
    if(shared>48*1024){int maximum=0;CUDA_CHECK(cudaDeviceGetAttribute(&maximum,cudaDevAttrMaxSharedMemoryPerBlockOptin,ctx.device));GGML_ASSERT(shared<=maximum); \
        CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q4k_half_ids<half2,32,DS4PERF_NATIVE_COLUMNS,NW>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));} \
    mul_mat_q4k_half_ids<half2,32,DS4PERF_NATIVE_COLUMNS,NW><<<grid,block,shared,ctx.stream()>>>(static_cast<const char*>(w->data),static_cast<const float*>(x->data),src,dest,bounds,task_expert,task_base,task_count,static_cast<float*>(dst->data),k/2,tokens,6,int(w->nb[1]),int(x->nb[2]/8),int(dst->nb[2]/4),0,int(w->nb[2]),x->ne[1]==1?0:int(x->nb[1]/4),int(dst->nb[1]/4),1,0,0,0,sis,nch);break;}
    switch(nwarps_best){DS4SPLIT_Q4K_HALF_LAUNCH(1);DS4SPLIT_Q4K_HALF_LAUNCH(2);DS4SPLIT_Q4K_HALF_LAUNCH(3);DS4SPLIT_Q4K_HALF_LAUNCH(4);DS4SPLIT_Q4K_HALF_LAUNCH(5);DS4SPLIT_Q4K_HALF_LAUNCH(6);DS4SPLIT_Q4K_HALF_LAUNCH(7);DS4SPLIT_Q4K_HALF_LAUNCH(8);default:GGML_ABORT("Q4_K half MMID unsupported warp count");}
#undef DS4SPLIT_Q4K_HALF_LAUNCH
    CUDA_CHECK(cudaGetLastError());
    if(witness){witness->kernel_family=1;witness->device=ctx.device;witness->cc=cc;witness->nwarps=nwarps_best;witness->rows_per_block=32;witness->columns_per_block=DS4PERF_NATIVE_COLUMNS;witness->tokens=tokens;witness->experts=experts;witness->maximum_tasks=int64_t(p.maximum_tasks);witness->workspace_requested_bytes=size_t(p.bytes);witness->shared_bytes=size_t(shared);}
    if constexpr(DS4PERF_NATIVE_COLUMNS!=16||DS4PERF_NATIVE_LOADER_UNROLL==1){static std::atomic<unsigned>logged{0};const unsigned bit=k==5120?1u:2u;
        if(!(logged.fetch_or(bit,std::memory_order_relaxed)&bit))std::fprintf(stderr,
            "DS4PERF_EXPERT_SCHEDULE columns=%d rows=32 k=%d n=%d tokens=%d warps=%d shared_bytes=%d task_bound=%llu workspace=%llu completion=queued_only\n",
            DS4PERF_NATIVE_COLUMNS,k,n,tokens,nwarps_best,shared,(unsigned long long)p.maximum_tasks,(unsigned long long)p.bytes);}
}
void ggml_cuda_mul_mat_q4k_half_id(ggml_backend_cuda_context&ctx,const ggml_tensor*w,const ggml_tensor*x,const ggml_tensor*ids,ggml_tensor*dst,ds4split_q4k_half_launch*witness){
    GGML_ASSERT(ggml_cuda_supports_q4k_half_mmid(w,x,ids,dst,ggml_cuda_info().devices[ctx.device].cc));const auto p=ds4split_q4k_half::plan(ids->ne[1],w->ne[2]);ggml_cuda_pool_alloc<char>scratch(ctx.pool(),size_t(p.bytes));ggml_cuda_mul_mat_q4k_half_id_workspace(ctx,w,x,ids,dst,scratch.get(),size_t(p.bytes),witness);
}
void ggml_cuda_q4k_half_loader_golden(const void*source,half*out,int64_t k,int64_t rows,cudaStream_t stream){
    GGML_ASSERT(source&&out&&((k==5120&&rows==2304)||(k==2304&&rows==5120)));const int64_t pairs=k*rows/2;ds4split_q4k_half_impl::loader_golden<<<unsigned((pairs+255)/256),256,0,stream>>>(static_cast<const char*>(source),reinterpret_cast<half2*>(out),int(k),int(rows));CUDA_CHECK(cudaGetLastError());
}
