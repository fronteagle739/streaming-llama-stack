#include "dsv41-ops.cuh"
#include "dsv41-ops-kernels.cuh"
#include "ggml-dsv41-rope-inplace.h"
#include "ggml-dsv41-index-window.h"
#include "ggml-dsv41-gather-half.h"
#include "getrows.cuh"
#include <atomic>

static std::atomic<uint64_t> dsv41_rope_inplace_queued{0};
static std::atomic<uint64_t> dsv41_candidate_window_queued{0};
extern "C" uint64_t ggml_backend_cuda_dsv41_candidate_window_queued(void){return dsv41_candidate_window_queued.load(std::memory_order_acquire);}
extern "C" uint64_t ggml_backend_cuda_dsv41_rope_inplace_queued(void) {
    return dsv41_rope_inplace_queued.load(std::memory_order_acquire);
}

void ggml_cuda_op_dsv41_engram_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int64_t rows = 4 * dst->ne[2];
    dsv41_engram_add_kernel<<<unsigned(rows), 32, 0, ctx.stream()>>>(
        (float *) dst->data, (const float *) dst->src[0]->data,
        (const float *) dst->src[1]->data, (const float *) dst->src[2]->data,
        (const float *) dst->src[3]->data,
        dst->src[4] ? (const int8_t *) dst->src[4]->data : nullptr,
        dst->ne[0], rows, ggml_get_op_params_f32(dst, 0));
}

void ggml_cuda_op_dsv41_pool2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int64_t count = ggml_nelements(dst);
    dsv41_pool2_kernel<<<unsigned((count + 255) / 256), 256, 0, ctx.stream()>>>(
        (float *) dst->data, (const float *) dst->src[0]->data,
        (const float *) dst->src[1]->data,
        dst->src[2] ? (const float *) dst->src[2]->data : nullptr,
        dst->src[3] ? (const float *) dst->src[3]->data : nullptr,
        dst->ne[0], count);
}

void ggml_cuda_op_dsv41_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const bool inplace = dst->data == dst->src[0]->data;
    GGML_ASSERT(!inplace || ggml_dsv41_rope_inplace_requested(dst));
    const int64_t rows = dst->ne[1] * dst->ne[2];
    dsv41_rope_kernel<<<unsigned(rows), 32, 0, ctx.stream()>>>(
        (float *) dst->data, (const float *) dst->src[0]->data,
        (const int32_t *) dst->src[1]->data, (const float *) dst->src[2]->data,
        dst->ne[0], dst->ne[1], rows, ggml_get_op_params_i32(dst, 0) != 0);
    CUDA_CHECK(cudaGetLastError());
    if (inplace) dsv41_rope_inplace_queued.fetch_add(1, std::memory_order_release);
}

void ggml_cuda_op_dsv41_raw_commit(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    dsv41_raw_commit_kernel<<<unsigned(dst->src[0]->ne[1]), 128, 0, ctx.stream()>>>(
        (float *) dst->data, (const float *) dst->src[0]->data,
        uint32_t(ggml_get_op_params_i32(dst, 0)), dst->src[0]->ne[1]);
}

void ggml_cuda_op_dsv41_index_score(ggml_backend_cuda_context & ctx,ggml_tensor * dst){
    dsv41_index_score_kernel<<<dim3(unsigned((dst->ne[0]+3)/4),unsigned(dst->ne[1])),128,0,ctx.stream()>>>(
        (float*)dst->data,(const float*)dst->src[0]->data,(const float*)dst->src[1]->data,
        (const float*)dst->src[2]->data,(const int32_t*)dst->src[3]->data,
        dst->src[4]?(const float*)dst->src[4]->data:nullptr,dst->ne[0]);
}
void ggml_cuda_op_dsv41_candidate_blocks(ggml_backend_cuda_context & ctx,ggml_tensor * dst){
    dsv41_candidate_blocks_kernel<<<unsigned((ggml_nelements(dst)+127)/128),128,0,ctx.stream()>>>(
        (float*)dst->data,(const float*)dst->src[0]->data,(const int32_t*)dst->src[1]->data,dst->src[0]->ne[0],dst->ne[1],ggml_dsv41_index_window_offset(dst));
    CUDA_CHECK(cudaGetLastError());if(ggml_dsv41_index_window_requested(dst))dsv41_candidate_window_queued.fetch_add(1,std::memory_order_release);
}
void ggml_cuda_op_dsv41_selected_mask(ggml_backend_cuda_context & ctx,ggml_tensor * dst){
    dsv41_selected_mask_kernel<<<unsigned((ggml_nelements(dst)+127)/128),128,0,ctx.stream()>>>(
        (float*)dst->data,(const int32_t*)dst->src[0]->data,(const int32_t*)dst->src[1]->data,
        dst->src[2]?(const float*)dst->src[2]->data:nullptr,dst->src[2]?dst->src[2]->ne[0]:0,dst->ne[0],ggml_nelements(dst));
}

void ggml_cuda_op_dsv41_masked_half(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if(ggml_dsv41_gather_half_requested(dst)) { ggml_cuda_op_dsv41_gather_half(ctx,dst); return; }
    GGML_ASSERT(!dst->src[2]);
    const int64_t count = ggml_nelements(dst);
    dsv41_masked_half_kernel<<<unsigned((count + 255) / 256), 256, 0, ctx.stream()>>>(
        (float *) dst->data, (const float *) dst->src[0]->data, (const float *) dst->src[1]->data, dst->ne[0], count);
}
