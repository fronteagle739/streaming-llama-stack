#include "quantize.cuh"
#include "convert.cuh"
#include "ggml-dsv41-native-fp4.h"
#include <atomic>
#include <cstring>
#include <cstdint>

__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)
static __global__ void quantize_q8_1(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2) {
    const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i3 = fastdiv(blockIdx.z, ne2);
    const int64_t i2 = blockIdx.z - i3*ne2.z;
    const int64_t i1 = blockIdx.y;

    const int64_t & i00 = i0;
    const int64_t & i01 = i1;
    const int64_t & i02 = i2;
    const int64_t & i03 = i3;

    const int64_t i_cont = ((i3*ne2.z + i2) * ne1 + i1) * ne0 + i0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib  = i_cont / QK8_1; // block index
    const int64_t iqs = i_cont % QK8_1; // quant index

    const float xi = i0 < ne00 ? x[i03*s03 + i02*s02 + i01*s01 + i00] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    y[ib].ds = make_half2(d, sum);
}

__device__ __forceinline__ uint8_t compute_e8m0_scale(float amax) {
    if (!(amax > 0.0f)) {
        return 0;
    }

    // FP4 E2M1: max exponent (unbiased) is 2.
    constexpr int FP4_E2M1_EMAX = 2;

    const float e = log2f(amax);

    // "even" -> round-to-nearest integer, ties-to-even
    const int e_int = __float2int_rn(e);

    const int shared_exp = e_int - FP4_E2M1_EMAX;

    int biased = shared_exp + 127;

    biased = max(biased, 0);
    biased = min(biased, 254);

    return static_cast<uint8_t>(biased);
}

// quantize values in the format mxfp4 is stored which is interleaved nibbles
// i.e. a block a0-a31 is represented as a0a16,a1a17 ...a15a31
static inline __device__ void dsv41_q4k_scale_min(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// Scalar source adapter of convert.cu:dequantize_block_q4_K. The FP4
// reduction/scale/rounding/store body below is unchanged. Each warp reads one
// original32-value sub-block, using the same Q4 scale/min and F32 expression.
static __device__ __forceinline__ float dsv41_q4k_weight_load(const void *vx,int64_t index){
    const auto *x=static_cast<const block_q4_K*>(vx);const int64_t i=index/QK_K;
    const int within=int(index%QK_K),il=within/64,is=within/32;
    const float dall=__low2half(x[i].dm),dmin=__high2half(x[i].dm);
    uint8_t sc,m;dsv41_q4k_scale_min(is,x[i].scales,sc,m);
    const float d1=dall*sc,m1=dmin*m;
    const uint8_t q=x[i].qs[32*il+(within%32)];
    return d1*((within%64)<32?(q&0xF):(q>>4))-m1;
}

template<bool ordinary_weights,bool q4k_source=false>
static __global__ void quantize_mmq_mxfp4(const float * __restrict__ x,
                                          const int32_t * __restrict__ ids,
                                          void * __restrict__ vy,
                                          const int64_t ne00,
                                          const int64_t s01,
                                          const int64_t s02,
                                          const int64_t s03,
                                          const int64_t ne0,
                                          const int     ne1,
                                          const int     ne2) {
    constexpr int vals_per_scale = 32;
    constexpr int vals_per_warp  = 2 * vals_per_scale;  // Each warp processes 2 blocks of 32 = 64 values

    const int warp_id = threadIdx.y;
    const int lane_id_32 = threadIdx.x;

    const int nwarps = blockDim.y;

    const int64_t warp_start_offset = (blockIdx.y * nwarps + warp_id) * vals_per_warp;

    if (warp_start_offset >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    block_fp4_mmq * y = (block_fp4_mmq *) vy;

    const int64_t block_fp4_mmq_size = 8 * QK_MXFP4;  // 256 values
    const int64_t ib0                = blockIdx.z * ((int64_t) ne1 * (ne0 / block_fp4_mmq_size));
    const int64_t ib = ib0 + (warp_start_offset / block_fp4_mmq_size) * ne1 + blockIdx.x;
    const int64_t quad_idx_in_block  = (warp_start_offset % block_fp4_mmq_size) / vals_per_warp;

    const int group_id = lane_id_32 / 4;
    const int lane_in_group = lane_id_32 % 4;
    const int base = group_id * 2;
    char2 * yqs2 = (char2 *) y[ib].qs;

    const int64_t base_pos = i03 * s03 + i02 * s02 + i01 * s01;

    uint8_t scales[2];

#pragma unroll
    for (int b = 0; b < 2; ++b) {
        const int64_t i0 = warp_start_offset + b * vals_per_scale + lane_id_32;
        float xi=0.0f;
        if(i0<ne00){
            if constexpr(q4k_source)xi=dsv41_q4k_weight_load(x,base_pos+i0);
            else xi=x[base_pos+i0];
        }

        float amax = fabsf(xi);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, WARP_SIZE));
        }

        const uint8_t e = compute_e8m0_scale(amax);
        scales[b] = e;
        const int64_t ordinary_ib=i1*(ne0/QK_MXFP4)+warp_start_offset/QK_MXFP4+b;
        if constexpr(ordinary_weights){
            if(lane_id_32==0)static_cast<block_mxfp4*>(vy)[ordinary_ib].e=e;
        }
        const float inv_s = (amax == 0.0f) ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(e));

#if CUDART_VERSION >= 12080
        const float scaled_val = xi * inv_s;

        const float val0 = __shfl_sync(0xFFFFFFFF, scaled_val, base, WARP_SIZE);
        const float val1 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 16, WARP_SIZE);
        const float val2 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 1, WARP_SIZE);
        const float val3 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 17, WARP_SIZE);

        if (lane_in_group == 0) {
            __nv_fp4x4_e2m1 fp4_packed(make_float4(val0, val1, val2, val3));

            const char2 q=*(char2 *) &fp4_packed;
            if constexpr(ordinary_weights){
                // The17-byte ordinary block need not be two-byte aligned.
                auto *qs=static_cast<block_mxfp4*>(vy)[ordinary_ib].qs;
                qs[base]=uint8_t(q.x);qs[base+1]=uint8_t(q.y);
            }else yqs2[quad_idx_in_block * 16 + b * 8 + group_id] = q;
        }
#else
        // Fallback: manual FP4 conversion using LUT
        const uint8_t q_val = ggml_cuda_float_to_fp4_e2m1(xi, inv_s);

        const uint8_t q_lo_0 = __shfl_sync(0xFFFFFFFF, q_val, base,      WARP_SIZE);
        const uint8_t q_lo_1 = __shfl_sync(0xFFFFFFFF, q_val, base + 1,  WARP_SIZE);
        const uint8_t q_hi_0 = __shfl_sync(0xFFFFFFFF, q_val, base + 16, WARP_SIZE);
        const uint8_t q_hi_1 = __shfl_sync(0xFFFFFFFF, q_val, base + 17, WARP_SIZE);

        if (lane_in_group == 0) {
            char2 q;
            q.x = (q_hi_0 << 4) | q_lo_0;
            q.y = (q_hi_1 << 4) | q_lo_1;
            if constexpr(ordinary_weights){
                auto *qs=static_cast<block_mxfp4*>(vy)[ordinary_ib].qs;
                qs[base]=uint8_t(q.x);qs[base+1]=uint8_t(q.y);
            }else yqs2[quad_idx_in_block * 16 + b * 8 + group_id] = q;
        }
#endif // CUDART_VERSION >= 12080
    }

    if (!ordinary_weights && lane_id_32 == 0) {
        // Store 2 scales packed into 1 uint32
        y[ib].d4[quad_idx_in_block] = (scales[1] << 8) | scales[0];
    }
}

template <mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i00 = i0;
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    const float4 * x4 = (const float4 *) x;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
    const int64_t ib  = ib0 + (i0 / (4*QK8_1))*ne1 + blockIdx.x;                    // block index in channel
    const int64_t iqs = i0 % (4*QK8_1);                                             // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = i0 < ne00 ? x4[(i03*s03 + i02*s02 + i01*s01 + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

        // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);

    // Write back 4 int8 values as a single 32 bit value for better memroy bandwidth:
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) {
            return;
        }

        y[ib].d2s6[2 + iqs/16] = sum;

        if (iqs % 64 != 0) {
            return;
        }

        const float d = 1.0f / d_inv;

        y[ib].d2s6[iqs/64] = d;

        return;
    }

    if (iqs % 32 != 0) {
        return;
    }

    const float d = 1.0f / d_inv;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs/32] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs/32]  = d;
    }
}

void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(!ids);
    GGML_ASSERT(ne0 % QK8_1 == 0);

    const uint3 ne2_fastdiv = init_fastdiv_values(ne2);

    const int64_t block_num_x = (ne0 + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ne1, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, ne00, s01, s02, s03, ne0, ne1, ne2_fastdiv);
    GGML_UNUSED(type_src0);
}

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % (4*QK8_1) == 0);

    // ne1 tends to assume the highest values, therefore use it as the "x" dimension of the CUDA grid:
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void quantize_mmq_mxfp4_cuda(const float *                    x,
                             const int32_t *                  ids,
                             void *                           vy,
                             [[maybe_unused]] const ggml_type type_src0,
                             const int64_t                    ne00,
                             const int64_t                    s01,
                             const int64_t                    s02,
                             const int64_t                    s03,
                             const int64_t                    ne0,
                             const int64_t                    ne1,
                             const int64_t                    ne2,
                             const int64_t                    ne3,
                             cudaStream_t                     stream) {
    GGML_ASSERT(ne0 % (2 * QK_MXFP4) == 0);

    constexpr int nwarps = 8;
    constexpr int vals_per_warp  = 2 * QK_MXFP4;
    constexpr int vals_per_block = nwarps * vals_per_warp;

    const int64_t block_num_y = (ne0 + vals_per_block - 1) / vals_per_block;
    const dim3    num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3    block_size(WARP_SIZE, nwarps, 1);

    quantize_mmq_mxfp4<false><<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}

static std::atomic<uint64_t> dsv41_mxfp4_weight_packs{0};
extern "C" uint64_t ggml_backend_cuda_dsv41_mxfp4_weight_packs_queued(void){return dsv41_mxfp4_weight_packs.load(std::memory_order_acquire);}
extern "C" int ggml_backend_cuda_dsv41_pack_weights_mxfp4(
        const ggml_tensor *source,ggml_tensor *destination,void *scratch,size_t scratch_bytes,void *stream_ptr){
    if(!source||!destination||!source->data||!destination->data||!scratch||!stream_ptr||
       !source->buffer||!destination->buffer||destination->type!=GGML_TYPE_MXFP4||
       (source->type!=GGML_TYPE_Q8_0&&source->type!=GGML_TYPE_F16&&source->type!=GGML_TYPE_Q4_K)||
       !ggml_is_contiguous(source)||!ggml_is_contiguous(destination)||
       source->ne[0]<256||source->ne[0]>20480||source->ne[0]%256||
       std::memcmp(source->ne,destination->ne,sizeof(source->ne))||
       scratch_bytes>64ull*1024*1024||scratch_bytes<size_t(source->ne[0])*sizeof(float))return int(cudaErrorInvalidValue);
    const size_t source_bytes=ggml_nbytes(source),destination_bytes=ggml_nbytes(destination);
    auto range=[](const void*p,size_t n,uintptr_t&a,uintptr_t&b){a=reinterpret_cast<uintptr_t>(p);if(!n||n>UINTPTR_MAX-a)return false;b=a+n;return true;};
    uintptr_t a=0,b=0,c=0,d=0,e=0,f=0;
    if(!range(source->data,source_bytes,a,b)||!range(destination->data,destination_bytes,c,d)||!range(scratch,scratch_bytes,e,f)||
       !(b<=c||d<=a)||!(b<=e||f<=a)||!(d<=e||f<=c))return int(cudaErrorInvalidValue);
    auto owned=[&](const ggml_tensor*t,uintptr_t start,uintptr_t end){uintptr_t base=0,limit=0;return range(ggml_backend_buffer_get_base(t->buffer),ggml_backend_buffer_get_size(t->buffer),base,limit)&&start>=base&&end<=limit;};
    if(!owned(source,a,b)||!owned(destination,c,d))return int(cudaErrorInvalidValue);
    const auto decode=ggml_get_to_fp32_cuda(source->type);if(!decode)return int(cudaErrorInvalidValue);
    const int64_t k=source->ne[0],rows=ggml_nrows(source);
    const int64_t chunk_rows=std::min<int64_t>(65535,std::min<size_t>(scratch_bytes,48ull*1024*1024)/(size_t(k)*sizeof(float)));
    if(rows<=0||chunk_rows<=0)return int(cudaErrorInvalidValue);
    const auto stream=static_cast<cudaStream_t>(stream_ptr);
    for(int64_t first=0;first<rows;first+=chunk_rows){
        const int64_t count=std::min(chunk_rows,rows-first);
        decode(static_cast<const char*>(source->data)+size_t(first)*source->nb[1],static_cast<float*>(scratch),count*k,stream);
        auto error=cudaGetLastError();if(error!=cudaSuccess)return int(error);
        constexpr int nwarps=8;const dim3 blocks(unsigned(count),unsigned((k+2*QK_MXFP4*nwarps-1)/(2*QK_MXFP4*nwarps)),1),threads(WARP_SIZE,nwarps,1);
        quantize_mmq_mxfp4<true><<<blocks,threads,0,stream>>>(static_cast<const float*>(scratch),nullptr,
            static_cast<char*>(destination->data)+size_t(first)*destination->nb[1],k,k,count*k,count*k,k,int(count),1);
        error=cudaGetLastError();if(error!=cudaSuccess)return int(error);
    }
    dsv41_mxfp4_weight_packs.fetch_add(1,std::memory_order_release);return int(cudaSuccess);
}


// Existing weight quantizer with the original Q4_K decoder at its source
// boundary. Input/output remain disjoint; the CPU in-place adapter stages each
// raw chunk into its already-owned scratch before calling this entry point.
extern "C" int ggml_backend_cuda_dsv41_pack_q4k_mxfp4_fused(
        const ggml_tensor *source,ggml_tensor *destination,void *stream_ptr){
    if(!source||!destination||!source->buffer||!destination->buffer||!source->data||!destination->data||!stream_ptr||
        source->type!=GGML_TYPE_Q4_K||destination->type!=GGML_TYPE_MXFP4||!ggml_is_contiguous(source)||!ggml_is_contiguous(destination)||
        source->ne[0]<256||source->ne[0]>20480||source->ne[0]%256||std::memcmp(source->ne,destination->ne,sizeof(source->ne)))return int(cudaErrorInvalidValue);
    auto owned=[](const ggml_tensor *t,uintptr_t&a,uintptr_t&b){
        const auto base=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(t->buffer));const auto size=ggml_backend_buffer_get_size(t->buffer);a=reinterpret_cast<uintptr_t>(t->data);
        const auto bytes=ggml_nbytes(t);if(!bytes||a<base||a-base>size||bytes>size-(a-base)||bytes>UINTPTR_MAX-a)return false;b=a+bytes;return true;
    };
    uintptr_t a=0,b=0,c=0,d=0;if(!owned(source,a,b)||!owned(destination,c,d)||!(b<=c||d<=a))return int(cudaErrorInvalidValue);
    const int64_t k=source->ne[0],rows=ggml_nrows(source);if(rows<=0)return int(cudaErrorInvalidValue);
    const auto stream=static_cast<cudaStream_t>(stream_ptr);
    for(int64_t first=0;first<rows;first+=65535){const int count=int(std::min<int64_t>(65535,rows-first));
        constexpr int nwarps=8;const dim3 blocks(unsigned(count),unsigned((k+2*QK_MXFP4*nwarps-1)/(2*QK_MXFP4*nwarps)),1),threads(WARP_SIZE,nwarps,1);
        quantize_mmq_mxfp4<true,true><<<blocks,threads,0,stream>>>(reinterpret_cast<const float*>(static_cast<const char*>(source->data)+size_t(first)*source->nb[1]),nullptr,
            static_cast<char*>(destination->data)+size_t(first)*destination->nb[1],k,k,int64_t(count)*k,int64_t(count)*k,k,count,1);
        const auto error=cudaGetLastError();if(error!=cudaSuccess)return int(error);
    }
    dsv41_mxfp4_weight_packs.fetch_add(1,std::memory_order_release);return int(cudaSuccess);
}
