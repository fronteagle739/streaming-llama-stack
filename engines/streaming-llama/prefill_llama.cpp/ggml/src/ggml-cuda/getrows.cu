#include "dispatch-counter.cuh"
#include "getrows.cuh"
#include "dequantize.cuh"
#include "convert.cuh"
#include "ggml-dsv41-gather-half.h"
#include <atomic>

// ─── K-quant get_rows support (added 2026-04-19 for split-decode Phase 2b) ──
// The legacy k_get_rows template (below) works only on non-K-quants where a
// value can be dequantized in isolation via a dequantize_kernel_t(block, iqs).
// K-quants (Q4_K, Q5_K, Q6_K, ...) need block-at-a-time dequant because
// scales + mins are packed per super-block. So we have a separate kernel per
// K-quant type that mirrors the layout of dequantize_block_q{4,5,6}_K.

static inline __device__ void get_scale_min_k4_getrows(
        int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

template<typename dst_t>
static __global__ void k_get_rows_q4_K(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {
    // Grid: x = i10 (dst row idx), y = block-within-row, z = ne11*ne12
    // Threads: 32 (one per quarter-pair of the 256-element block)
    const int64_t i10   = blockIdx.x;
    const int64_t z_idx = blockIdx.z;
    const int64_t i11   = z_idx / ne12;
    const int64_t i12   = z_idx % ne12;

    const int32_t i01 = src1[i10*s10 + i11*s11 + i12*s12];

    const char * row_bytes = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;
    const block_q4_K * row_blocks = (const block_q4_K *) row_bytes;

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;

    const int64_t blk_per_row = ne00 / QK_K;  // QK_K = 256

    // Each CUDA block handles one Q4_K super-block (gridDim.y covers them all).
    for (int64_t blk = blockIdx.y; blk < blk_per_row; blk += gridDim.y) {
        const block_q4_K & b = row_blocks[blk];
        dst_t * y = dst_row + blk * QK_K;

        // Mirror dequantize_block_q4_K (32 threads, il=0..3, ir=0..7)
        const int64_t tid = threadIdx.x;
        const int64_t il  = tid / 8;   // 0..3
        const int64_t ir  = tid % 8;   // 0..7
        const int64_t is  = 2 * il;
        const int64_t n   = 4;

        const float dall = __low2half(b.dm);
        const float dmin = __high2half(b.dm);

        const uint8_t * q = b.qs + 32*il + n*ir;

        uint8_t sc, m;
        get_scale_min_k4_getrows(is + 0, b.scales, sc, m);
        const float d1 = dall * sc; const float m1 = dmin * m;
        get_scale_min_k4_getrows(is + 1, b.scales, sc, m);
        const float d2 = dall * sc; const float m2 = dmin * m;

        dst_t * y_pos = y + 64*il + n*ir;
        for (int l = 0; l < n; ++l) {
            y_pos[l + 0]  = ggml_cuda_cast<dst_t>(d1 * (float)(q[l] & 0xF) - m1);
            y_pos[l + 32] = ggml_cuda_cast<dst_t>(d2 * (float)(q[l] >>  4) - m2);
        }
    }
}

template<typename dst_t>
static void get_rows_cuda_q4_K(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_K == 0);
    const int64_t blk_per_row = ne00 / QK_K;
    const size_t s1  = nb1  / sizeof(dst_t);
    const size_t s2  = nb2  / sizeof(dst_t);
    const size_t s3  = nb3  / sizeof(dst_t);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    const dim3 block_dims(32, 1, 1);
    const dim3 grid_dims((unsigned)ne10, (unsigned)std::min<int64_t>(blk_per_row, 65535), (unsigned)(ne11*ne12));

    k_get_rows_q4_K<<<grid_dims, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d, ne00, ne11, ne12,
        s1, s2, s3, nb01, nb02, nb03, s10, s11, s12);
}

template<typename dst_t>
static __global__ void k_get_rows_q5_K(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {
    // Grid: x = i10 (dst row idx), y = block-within-row, z = ne11*ne12.
    // Threads: 64, matching dequantize_block_q5_K for one 256-value block.
    const int64_t i10   = blockIdx.x;
    const int64_t z_idx = blockIdx.z;
    const int64_t i11   = z_idx / ne12;
    const int64_t i12   = z_idx % ne12;

    const int32_t i01 = src1[i10*s10 + i11*s11 + i12*s12];

    const char * row_bytes = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;
    const block_q5_K * row_blocks = (const block_q5_K *) row_bytes;

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;

    const int64_t blk_per_row = ne00 / QK_K;

    for (int64_t blk = blockIdx.y; blk < blk_per_row; blk += gridDim.y) {
        const block_q5_K & b = row_blocks[blk];
        dst_t * y = dst_row + blk * QK_K;

        const int64_t tid = threadIdx.x;
        const int64_t il  = tid / 16;  // 0..3
        const int64_t ir  = tid % 16;  // 0..15
        const int64_t is  = 2 * il;

        const float dall = __low2half(b.dm);
        const float dmin = __high2half(b.dm);

        const uint8_t * ql = b.qs + 32*il + 2*ir;
        const uint8_t * qh = b.qh + 2*ir;

        uint8_t sc, m;
        get_scale_min_k4_getrows(is + 0, b.scales, sc, m);
        const float d1 = dall * sc; const float m1 = dmin * m;
        get_scale_min_k4_getrows(is + 1, b.scales, sc, m);
        const float d2 = dall * sc; const float m2 = dmin * m;

        dst_t * y_pos = y + 64*il + 2*ir;
        uint8_t hm = 1 << (2*il);
        y_pos[ 0] = ggml_cuda_cast<dst_t>(d1 * ((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0)) - m1);
        y_pos[ 1] = ggml_cuda_cast<dst_t>(d1 * ((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0)) - m1);
        hm <<= 1;
        y_pos[32] = ggml_cuda_cast<dst_t>(d2 * ((ql[0] >>  4) + (qh[0] & hm ? 16 : 0)) - m2);
        y_pos[33] = ggml_cuda_cast<dst_t>(d2 * ((ql[1] >>  4) + (qh[1] & hm ? 16 : 0)) - m2);
    }
}

template<typename dst_t>
static void get_rows_cuda_q5_K(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_K == 0);
    const int64_t blk_per_row = ne00 / QK_K;
    const size_t s1  = nb1  / sizeof(dst_t);
    const size_t s2  = nb2  / sizeof(dst_t);
    const size_t s3  = nb3  / sizeof(dst_t);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    const dim3 block_dims(64, 1, 1);
    const dim3 grid_dims((unsigned)ne10, (unsigned)std::min<int64_t>(blk_per_row, 65535), (unsigned)(ne11*ne12));

    k_get_rows_q5_K<<<grid_dims, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d, ne00, ne11, ne12,
        s1, s2, s3, nb01, nb02, nb03, s10, s11, s12);
}

template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void k_get_rows(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const int64_t ne12, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    for (int64_t z = blockIdx.z; z < ne11*ne12; z += gridDim.z) {
        for (int64_t i00 = 2*(blockIdx.y*blockDim.x + threadIdx.x); i00 < ne00; i00 += gridDim.y*blockDim.x) {
            // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
            const int i10 =  blockIdx.x;
            const int i11 =  z / ne12; // TODO fastdiv
            const int i12 =  z % ne12;

            const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

            dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
            const void * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;

            const int ib   =  i00/qk;      // block index
            const int iqs  = (i00%qk)/qr;  // quant index
            const int iybs = i00 - i00%qk; // dst block start index
            const int y_offset = qr == 1 ? 1 : qk/2;

            // dequantize
            float2 v;
            dequantize_kernel(src0_row, ib, iqs, v);

            dst_row[iybs + iqs + 0]        = ggml_cuda_cast<dst_t>(v.x);
            dst_row[iybs + iqs + y_offset] = ggml_cuda_cast<dst_t>(v.y);
        }
    }
}

template<typename src0_t, typename dst_t, bool masked_half = false, typename extra_t = src0_t>
static __global__ void k_get_rows_float(
        const src0_t * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const int64_t ne12, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/,
        const float * mask = nullptr, int64_t source_rows = 0,
        const extra_t * extra = nullptr, int64_t first_rows = 0) {

    for (int64_t z = blockIdx.z; z < ne11*ne12; z += gridDim.z) {
        for (int64_t i00 = blockIdx.y*blockDim.x + threadIdx.x; i00 < ne00; i00 += gridDim.y*blockDim.x) {
            // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
            const int i10 = blockIdx.x;
            const int i11 = z / ne12; // TODO fastdiv
            const int i12 = z % ne12;

            if (i00 >= ne00) {
                return;
            }

            const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

            dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
            if constexpr (masked_half) {
                if (i01 < 0 || i01 >= source_rows) asm volatile("trap;");
                if (__float_as_uint(mask[i10*s10 + i11*s11 + i12*s12]) == 0xff800000u) {
                    dst_row[i00] = 0.f;
                    continue;
                }
            }
            if constexpr (masked_half) {
                const float value=extra && i01>=first_rows ?
                    ggml_cuda_cast<float>(extra[int64_t(i01-first_rows)*ne00+i00]) :
                    ((const src0_t *)((const char *)src0+i01*nb01+i11*nb02+i12*nb03))[i00];
                dst_row[i00] = __half2float(__float2half_rn(value));
            } else {
                const src0_t * src0_row = (const src0_t *)((const char *)src0+i01*nb01+i11*nb02+i12*nb03);
                dst_row[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);
            }
        }
    }
}

template<typename grad_t, typename dst_t>
static __global__ void k_get_rows_back_float(
        const grad_t * __restrict__ grad, const int32_t * __restrict__ rows, dst_t * __restrict__ dst, const int64_t ncols, const int64_t nrows_grad) {
    const int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (col >= ncols) {
        return;
    }

    const int dst_row = blockIdx.y*blockDim.y + threadIdx.y;

    float sum = 0.0f;

    for (int64_t i = 0; i < nrows_grad; ++i) {
        if (rows[i] != dst_row) {
            continue;
        }
        sum += grad[i*ncols + col];
    }

    dst[dst_row*ncols + col] = sum;
}

template<int qk, int qr, dequantize_kernel_t dq, typename dst_t>
static void get_rows_cuda_q(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_y = (ne00 + 2*CUDA_GET_ROWS_BLOCK_SIZE - 1) / (2*CUDA_GET_ROWS_BLOCK_SIZE);
    const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ne00 % 2 == 0);

    k_get_rows<qk, qr, dq><<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);
}

template<typename src0_t, typename dst_t, bool masked_half = false, typename extra_t = src0_t>
static void get_rows_cuda_float(
        const src0_t * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream, const float * mask = nullptr, int64_t source_rows = 0,
        const extra_t * extra = nullptr, int64_t first_rows = 0) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_y = (ne00 + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
    const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    k_get_rows_float<src0_t,dst_t,masked_half,extra_t><<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/, mask, source_rows, extra, first_rows);
}

static ggml_cuda_dispatch_counter dsv41_gather_half_queued{0};
static ggml_cuda_dispatch_counter dsv41_gather_half_split_queued{0};
extern "C" uint64_t ggml_backend_cuda_dsv41_gather_half_split_queued(void) {
    return dsv41_gather_half_split_queued.load(std::memory_order_acquire);
}
extern "C" uint64_t ggml_backend_cuda_dsv41_gather_half_queued(void) {
    return dsv41_gather_half_queued.load(std::memory_order_acquire);
}
void ggml_cuda_op_dsv41_gather_half(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_dsv41_gather_half_valid(dst));
    const auto * table=dst->src[0], * mask=dst->src[1], * ids=dst->src[2];
    const auto * extra=dst->src[3];
    GGML_ASSERT(table->data && mask->data && ids->data && dst->data && dst->data!=table->data);
    GGML_ASSERT(!extra||(extra->data&&extra->data!=dst->data));
    const int64_t source_rows=table->ne[1]+(extra?extra->ne[1]:0);
    if(extra&&extra->type==GGML_TYPE_BF16){
        get_rows_cuda_float<float,float,true,nv_bfloat16>((const float *)table->data,(const int32_t *)ids->data,(float *)dst->data,
            512,table->nb[1],table->nb[2],table->nb[3],ggml_nelements(ids),1,1,sizeof(int32_t),0,0,
            512*sizeof(float),0,0,ctx.stream(),(const float *)mask->data,source_rows,(const nv_bfloat16*)extra->data,table->ne[1]);
    }else{
        get_rows_cuda_float<float,float,true>((const float *)table->data,(const int32_t *)ids->data,(float *)dst->data,
            512,table->nb[1],table->nb[2],table->nb[3],ggml_nelements(ids),1,1,sizeof(int32_t),0,0,
            512*sizeof(float),0,0,ctx.stream(),(const float *)mask->data,source_rows,extra?(const float*)extra->data:nullptr,table->ne[1]);
    }
    CUDA_CHECK(cudaGetLastError());
    dsv41_gather_half_queued.fetch_add(1,std::memory_order_release);
    if(extra)dsv41_gather_half_split_queued.fetch_add(1,std::memory_order_release);
}

template <typename dst_t>
static void ggml_cuda_get_rows_switch_src0_type(
        const void * src0_d, const ggml_type src0_type, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    switch (src0_type) {
        case GGML_TYPE_F16:
            get_rows_cuda_float((const half *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_F32:
            get_rows_cuda_float((const float *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_I32:
            get_rows_cuda_float((const int32_t *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_BF16:
            get_rows_cuda_float((const nv_bfloat16 *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_0:
            get_rows_cuda_q<QK4_0, QR4_0, dequantize_q4_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_1:
            get_rows_cuda_q<QK4_1, QR4_1, dequantize_q4_1>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_0:
            get_rows_cuda_q<QK5_0, QR5_0, dequantize_q5_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_1:
            get_rows_cuda_q<QK5_1, QR5_1, dequantize_q5_1>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q8_0:
            get_rows_cuda_q<QK8_0, QR8_0, dequantize_q8_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_K:
            get_rows_cuda_q4_K(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_K:
            get_rows_cuda_q5_K(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        default:
            // TODO: remaining k-quants (Q6_K, Q2_K, Q3_K) + IQ quants
            GGML_ABORT("%s: unsupported src0 type: %s\n", __func__, ggml_type_name(src0_type));
            break;
    }
}

void get_rows_cuda(
        const void * src0_d, ggml_type src0_type, const int32_t * src1_d, void * dst_d, ggml_type dst_type,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12, size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        cudaStream_t stream) {
    switch (dst_type) {
        case GGML_TYPE_F32:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (float *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_I32:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (int32_t *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (half *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (nv_bfloat16 *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        default:
            GGML_ABORT("%s: unsupported dst type: %s\n", __func__, ggml_type_name(dst_type));
            break;
    }
}

void ggml_cuda_op_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ne13 == 1);

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(dst->nb[0]  == ggml_type_size(dst->type));

    get_rows_cuda(src0->data, src0->type, (const int32_t *) src1->data, dst->data, dst->type,
        ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
}

void ggml_cuda_op_get_rows_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    const ggml_tensor * src1 = dst->src[1]; // src1 in forward pass

    GGML_TENSOR_BINARY_OP_LOCALS

    const float   * src0_d = (const float   *) src0->data;
    const int32_t * src1_d = (const int32_t *) src1->data;
    float         * dst_d  = (float         *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(ne02*ne03 == 1);
    GGML_ASSERT(ne12*ne13 == 1);
    GGML_ASSERT(ne2*ne3 == 1);

    const dim3 block_dims(CUDA_GET_ROWS_BACK_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ne00 + CUDA_GET_ROWS_BACK_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BACK_BLOCK_SIZE;
    const dim3 block_nums(block_num_x, ne1, 1);

    k_get_rows_back_float<<<block_nums, block_dims, 0, stream>>>(src0_d, src1_d, dst_d, ne00, ne10);
}
