#include "dispatch-counter.cuh"
#include "dsv4-act-qat.cuh"
#include "dsv4-act-qat-kernels.cuh"
#include "ggml-dsv41-bf16-vector.h"
#include <atomic>
#include <cstdlib>

static ggml_cuda_dispatch_counter dsv41_bf16_vector_queued{0};
extern "C" uint64_t ggml_backend_cuda_dsv41_bf16_vector_queued(void) {
    return dsv41_bf16_vector_queued.load(std::memory_order_acquire);
}

void ggml_cuda_op_dsv4_act_qat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    int32_t mode  = 0;
    int32_t n_rot = 0;
    memcpy(&mode,  (const int32_t *) dst->op_params + 0, sizeof(int32_t));
    memcpy(&n_rot, (const int32_t *) dst->op_params + 1, sizeof(int32_t));

    const float * src_d = (const float *) src0->data;
    float       * dst_d = (float       *) dst->data;
    cudaStream_t  stream = ctx.stream();

    const int64_t ne0  = src0->ne[0];
    const int64_t nrow = ggml_nrows(src0);

    if (mode >= GGML_DSV41_QAT_BF16) {
        GGML_ASSERT(mode <= GGML_DSV41_QAT_FP8_E8M0 && n_rot == 0);
        if (mode == GGML_DSV41_QAT_BF16) {
            const char *vector_env=std::getenv("LLAMA_DSV41_BF16_VECTOR");
            if(vector_env&&vector_env[0]=='1'&&vector_env[1]=='\0'&&ggml_dsv41_bf16_vector_eligible(dst)){
                const int64_t vectors=ggml_nelements(dst)/4;
                dsv41_bf16_vector4_kernel<<<unsigned((vectors+255)/256),256,0,stream>>>(
                    (const float4*)src_d,(float4*)dst_d,vectors);
                CUDA_CHECK(cudaGetLastError());
                dsv41_bf16_vector_queued.fetch_add(1,std::memory_order_release);
                return;
            }
            // This format has no per-row/group reduction. Reuse the identical
            // existing FORMAT=0 body over the contiguous flat tensor, with a
            // normal 256-thread launch instead of one tiny CTA per 32 values.
            const int64_t total = ggml_nelements(src0);
            dsv41_act_qat_kernel<256,0><<<unsigned((total+255)/256),256,0,stream>>>(src_d,dst_d,total,1);
            return;
        }
        const int block = mode == GGML_DSV41_QAT_FP4_E4M3 ? 16 : 32;
        GGML_ASSERT(mode == GGML_DSV41_QAT_BF16 || ne0 % block == 0);
        const char *tiled=std::getenv("LLAMA_DSV41_QAT_TILED");
        if(!(tiled && tiled[0]=='0' && tiled[1]=='\0')) {
            const int64_t total=ggml_nelements(dst);
            const unsigned blocks=unsigned((total+255)/256);
            switch(mode) {
                case GGML_DSV41_QAT_FP4_E8M0:
                    dsv41_act_qat_tiled_kernel<32,2><<<blocks,256,0,stream>>>(src_d,dst_d,total);break;
                case GGML_DSV41_QAT_FP4_E4M3:
                    dsv41_act_qat_tiled_kernel<16,3><<<blocks,256,0,stream>>>(src_d,dst_d,total);break;
                case GGML_DSV41_QAT_FP8_E8M0:
                    dsv41_act_qat_tiled_kernel<32,1><<<blocks,256,0,stream>>>(src_d,dst_d,total);break;
                default:GGML_ABORT("invalid V4.1 QAT mode");
            }
            CUDA_CHECK(cudaGetLastError());return;
        }
        const int64_t max_grid_dim = 65535;
        const unsigned gy = unsigned(nrow < max_grid_dim ? nrow : max_grid_dim);
        const unsigned gz = unsigned((nrow + max_grid_dim - 1) / max_grid_dim);
        GGML_ASSERT(gy && gz <= max_grid_dim);
        const dim3 grid(unsigned((ne0 + block - 1) / block), gy, gz);
        switch (mode) {
            case GGML_DSV41_QAT_BF16:
                dsv41_act_qat_kernel<32, 0><<<grid, 32, 0, stream>>>(src_d, dst_d, ne0, nrow); break;
            case GGML_DSV41_QAT_FP4_E8M0:
                dsv41_act_qat_kernel<32, 2><<<grid, 32, 0, stream>>>(src_d, dst_d, ne0, nrow); break;
            case GGML_DSV41_QAT_FP4_E4M3:
                dsv41_act_qat_kernel<16, 3><<<grid, 16, 0, stream>>>(src_d, dst_d, ne0, nrow); break;
            case GGML_DSV41_QAT_FP8_E8M0:
                dsv41_act_qat_kernel<32, 1><<<grid, 32, 0, stream>>>(src_d, dst_d, ne0, nrow); break;
            default: GGML_ABORT("invalid V4.1 QAT mode");
        }
        return;
    }

    const bool    is_fp8 = (mode == GGML_DSV4_QAT_FP8_E4M3);
    const int64_t blk    = is_fp8 ? 64 : 32;
    const int64_t n_q    = is_fp8 ? (ne0 - n_rot) : ne0;

    GGML_ASSERT(n_q >= 0 && (n_q % blk) == 0);

    // Cover the whole row so the untouched tail is copied by the same launch.
    const int64_t nblocks = (ne0 + blk - 1) / blk;

    // gridDim.y is capped at 65535 on every CUDA arch. nrow here is indexer_head_count(64) *
    // n_tokens, so a 1024-token ubatch needs 65536 -- one over. Spill the overflow into
    // gridDim.z (also capped at 65535) and bounds-check inside the kernel; that lifts the
    // ceiling to ~4.3e9 rows, far past anything this model can produce.
    const int64_t MAX_GRID_DIM = 65535;
    const unsigned gy = (unsigned) (nrow < MAX_GRID_DIM ? nrow : MAX_GRID_DIM);
    const unsigned gz = (unsigned) ((nrow + MAX_GRID_DIM - 1) / MAX_GRID_DIM);
    GGML_ASSERT(gz <= MAX_GRID_DIM);
    const dim3 grid((unsigned) nblocks, gy, gz);

    if (is_fp8) {
        dsv4_act_qat_kernel<64, true><<<grid, 64, 0, stream>>>(src_d, dst_d, ne0, n_q, nrow);
    } else {
        dsv4_act_qat_kernel<32, false><<<grid, 32, 0, stream>>>(src_d, dst_d, ne0, n_q, nrow);
    }
}
