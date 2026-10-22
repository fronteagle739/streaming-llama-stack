#include "reduce_rows.cuh"
#include "sumrows.cuh"

void sum_rows_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int  id  = ggml_cuda_get_device();
    const int  nsm = ggml_cuda_info().devices[id].nsm;
    const dim3 block_nums(nrows, 1, 1);
    if ((nrows / nsm) < 2) {
        const dim3 block_dims(512, 1, 1);
        reduce_rows_f32</*norm=*/false><<<block_nums, block_dims, 0, stream>>>(x, dst, ncols);
    } else {
        const dim3 block_dims(ncols < 1024 ? 32 : 128, 1, 1);
        reduce_rows_f32</*norm=*/false><<<block_nums, block_dims, 0, stream>>>(x, dst, ncols);
    }
}

void ggml_cuda_op_sum_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const dim3 block_nums(nrows, 1, 1);

    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;
    if ((nrows / nsm) < 2) {
        // Increase num threads to 512 for small nrows to better hide the latency
        const dim3 block_dims(512, 1, 1);
        reduce_rows_f32</*norm=*/false><<<block_nums, block_dims, 0, stream>>>(src0_d, dst_d, ncols);
    } else {
        // Enough active SMs to hide latency, use smaller blocks to allow better scheduling
        const dim3 block_dims(ncols < 1024 ? 32 : 128, 1, 1);
        reduce_rows_f32</*norm=*/false><<<block_nums, block_dims, 0, stream>>>(src0_d, dst_d, ncols);
    }
}

template <bool columnwise>
static __global__ void k_sum_rows_add_div_f32(
        const float * __restrict__ src,
        const float * __restrict__ eps,
        float       * __restrict__ dst,
        int                         ne0,
        int                         ne1) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    // In row mode, one block handles a contiguous [ne0] row.  In column mode,
    // one block handles a row of CONT(PERMUTE(src, 1, 0, 2, 3)) without ever
    // materializing that transpose: the reduction dimension is src dim 1.
    const int ncols = columnwise ? ne1 : ne0;
    const int outer = columnwise ? row / ne0 : row;
    const int i0    = columnwise ? row - outer * ne0 : 0;

    float     sum        = 0.0f;
    const int num_unroll = 8;
    float     temp[num_unroll];
    float     sum_temp[num_unroll] = { 0.0f };

    for (int i = col; i < ncols;) {
        for (int j = 0; j < num_unroll; ++j) {
            if (i < ncols) {
                const size_t src_idx = columnwise ?
                    size_t(outer) * ne0 * ne1 + size_t(i) * ne0 + i0 :
                    size_t(row) * ne0 + i;
                temp[j] = src[src_idx];
            } else {
                temp[j] = 0.0f;
            }
            i += blockDim.x;
        }
        for (int j = 0; j < num_unroll; ++j) {
            sum_temp[j] += temp[j];
        }
    }
    for (int j = 0; j < num_unroll; ++j) {
        sum += sum_temp[j];
    }

    __shared__ float shared_vals[32];
    __shared__ float denominator;
    sum = block_reduce<block_reduce_method::SUM>(sum, shared_vals);
    if (col == 0) {
        denominator = __fadd_rn(sum, eps[0]);
    }
    __syncthreads();

    for (int i = col; i < ncols; i += blockDim.x) {
        const size_t dst_idx = columnwise ?
            size_t(outer) * ne0 * ne1 + size_t(i) * ne0 + i0 :
            size_t(row) * ne0 + i;
        dst[dst_idx] = src[dst_idx] / denominator;
    }
}

void ggml_cuda_op_fused_sum_rows_div(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor         * src,
        const ggml_tensor         * eps,
        ggml_tensor               * dst,
        bool                        columnwise) {
    GGML_ASSERT(src->type == GGML_TYPE_F32 && eps->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src) && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_nelements(eps) == 1);

    const int64_t nrows = columnwise ? src->ne[0] * src->ne[2] * src->ne[3] : ggml_nrows(src);
    const int64_t ncols = columnwise ? src->ne[1] : src->ne[0];
    const int id        = ggml_cuda_get_device();
    const int nsm       = ggml_cuda_info().devices[id].nsm;
    const int nthreads  = (nrows / nsm) < 2 ? 512 : (ncols < 1024 ? 32 : 128);

    const dim3 block_nums(nrows, 1, 1);
    const dim3 block_dims(nthreads, 1, 1);
    if (columnwise) {
        k_sum_rows_add_div_f32<true><<<block_nums, block_dims, 0, ctx.stream()>>>(
            (const float *) src->data, (const float *) eps->data, (float *) dst->data,
            src->ne[0], src->ne[1]);
    } else {
        k_sum_rows_add_div_f32<false><<<block_nums, block_dims, 0, ctx.stream()>>>(
            (const float *) src->data, (const float *) eps->data, (float *) dst->data,
            src->ne[0], src->ne[1]);
    }
}

static __global__ void k_lid_score_reduce_f32(
        const float * __restrict__ kq,
        const float * __restrict__ weights,
        float       * __restrict__ dst,
        int                         n_lid,
        int                         n_token,
        int                         n_head,
        float                       scale) {
    const int i_lid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_token  = blockIdx.y;
    const int i_stream = blockIdx.z;

    if (i_lid >= n_lid) {
        return;
    }

    // kq is the contiguous, as-produced MUL_MAT result:
    //   [n_lid, n_token, n_head, n_stream].
    // Reading one head plane at a time keeps neighboring lanes coalesced and
    // avoids both materialized transposes from the portable decomposition.
    const size_t kq_stream_stride = (size_t) n_lid * n_token * n_head;
    const size_t kq_token_base = (size_t) i_stream * kq_stream_stride +
        (size_t) i_token * n_lid + i_lid;
    const size_t weight_base = ((size_t) i_stream * n_token + i_token) * n_head;

    float score = 0.0f;
    for (int i_head = 0; i_head < n_head; ++i_head) {
        const float activated = fmaxf(kq[kq_token_base + (size_t) i_head * n_lid * n_token], 0.0f);
        const float scaled_weight = __fmul_rn(weights[weight_base + i_head], scale);
        score = __fadd_rn(score, __fmul_rn(activated, scaled_weight));
    }

    dst[((size_t) i_stream * n_token + i_token) * n_lid + i_lid] = score;
}

void ggml_cuda_op_fused_lid_score_reduce(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor         * kq,
        const ggml_tensor         * weights,
        ggml_tensor               * dst,
        float                       scale) {
    GGML_ASSERT(kq->type == GGML_TYPE_F32 && weights->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(kq) && ggml_is_contiguous(weights) && ggml_is_contiguous(dst));
    GGML_ASSERT(kq->ne[0] == dst->ne[0] && kq->ne[1] == dst->ne[1]);
    GGML_ASSERT(kq->ne[2] == weights->ne[0] && kq->ne[3] == weights->ne[3]);

    constexpr int nthreads = 256;
    const dim3 block_dims(nthreads, 1, 1);
    const dim3 block_nums((kq->ne[0] + nthreads - 1) / nthreads, kq->ne[1], kq->ne[3]);
    k_lid_score_reduce_f32<<<block_nums, block_dims, 0, ctx.stream()>>>(
        (const float *) kq->data, (const float *) weights->data, (float *) dst->data,
        kq->ne[0], kq->ne[1], kq->ne[2], scale);
}
