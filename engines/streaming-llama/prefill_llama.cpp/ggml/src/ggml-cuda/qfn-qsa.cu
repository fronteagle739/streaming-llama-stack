// QFN: QSA selection kernel. One 256-thread block per token.
// Keys are the rectified scores as bit patterns (non-negative floats order like their bits; -0 and negatives map
// to 0). An exact 4-pass radix descent finds the n_top-th largest key; pools above it are kept, and pools equal to
// it are kept in ascending index order until n_top are taken (ds4 idx_select semantics). The kept pools are then
// emitted in ascending pool order (each expanded to `ratio` cells), followed by the incomplete tail cells and -1.
// ggml_compute_forward_qfn_qsa_select (ggml-cpu/ops.cpp) is the bit-exact reference.
#include "qfn-qsa.cuh"

#include <climits>

#define QFN_QSA_THREADS 256

static __device__ __forceinline__ uint32_t qfn_qsa_key(const float x) {
    return x > 0.0f ? __float_as_uint(x) : 0u;
}

static __global__ void __launch_bounds__(QFN_QSA_THREADS) qfn_qsa_select_kernel(
        const float * __restrict__ score, const int64_t score_stride,
        const int32_t * __restrict__ pos,
        int32_t * __restrict__ out, const int64_t out_stride,
        const int n_top, const int ratio, const int n_sel) {
    const int t   = blockIdx.x;
    const int tid = threadIdx.x;
    const int p   = pos[t];
    const int nv  = (p + 1) / ratio;
    const float * row = score + (int64_t) t * score_stride;
    int32_t     * o   = out   + (int64_t) t * out_stride;

    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_thr, s_need;
    __shared__ uint32_t s_gt[QFN_QSA_THREADS], s_eq[QFN_QSA_THREADS], s_keep[QFN_QSA_THREADS];

    int w0;  // number of cells written by the pool part
    if (nv <= n_top) {
        for (int c = tid; c < nv * ratio; c += QFN_QSA_THREADS) {
            o[c] = c;
        }
        w0 = nv * ratio;
    } else {
        if (tid == 0) { s_thr = 0; s_need = (uint32_t) n_top; }
        __syncthreads();
        for (int pass = 0; pass < 4; ++pass) {
            const int      shift = 24 - 8*pass;
            const uint32_t mask  = pass ? (0xFFFFFFFFu << (shift + 8)) : 0u;
            hist[tid] = 0;
            __syncthreads();
            const uint32_t prefix = s_thr;
            for (int b = tid; b < nv; b += QFN_QSA_THREADS) {
                const uint32_t key = qfn_qsa_key(row[b]);
                if ((key & mask) == prefix) {
                    atomicAdd(&hist[(key >> shift) & 255u], 1u);
                }
            }
            __syncthreads();
            if (tid == 0) {
                uint32_t need = s_need;
                for (int d = 255; d >= 0; --d) {
                    if (hist[d] >= need) { s_thr = prefix | ((uint32_t) d << shift); break; }
                    need -= hist[d];
                }
                s_need = need;
            }
            __syncthreads();
        }
        const uint32_t thr  = s_thr;
        const uint32_t need = s_need;  // how many keys equal to thr must be kept
        const int chunk = (nv + QFN_QSA_THREADS - 1) / QFN_QSA_THREADS;
        const int beg   = min(nv, tid * chunk);
        const int end   = min(nv, beg + chunk);
        uint32_t gt = 0, eq = 0;
        for (int b = beg; b < end; ++b) {
            const uint32_t key = qfn_qsa_key(row[b]);
            gt += key > thr;
            eq += key == thr;
        }
        s_gt[tid] = gt;
        s_eq[tid] = eq;
        __syncthreads();
        if (tid == 0) {
            // exclusive prefix of equal counts, then the kept count per chunk and its exclusive prefix
            uint32_t eqp = 0, kp = 0;
            for (int i = 0; i < QFN_QSA_THREADS; ++i) {
                const uint32_t e = s_eq[i], g = s_gt[i];
                const uint32_t take = eqp >= need ? 0u : min(e, need - eqp);
                s_eq[i]   = eqp;
                s_keep[i] = kp;
                eqp += e;
                kp  += g + take;
            }
        }
        __syncthreads();
        uint32_t erank = s_eq[tid];
        uint32_t w     = s_keep[tid];
        for (int b = beg; b < end; ++b) {
            const uint32_t key = qfn_qsa_key(row[b]);
            bool keep = key > thr;
            if (key == thr) {
                keep = erank < need;
                ++erank;
            }
            if (keep) {
                for (int j = 0; j < ratio; ++j) {
                    o[w*ratio + j] = b*ratio + j;
                }
                ++w;
            }
        }
        w0 = n_top * ratio;
    }
    // tail cells, then padding
    const int n_tail = p + 1 - nv * ratio;
    for (int i = tid; i < n_sel - w0; i += QFN_QSA_THREADS) {
        o[w0 + i] = i < n_tail ? nv * ratio + i : -1;
    }
}

void ggml_cuda_op_qfn_qsa_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * score = dst->src[0];
    const ggml_tensor * pos   = dst->src[1];
    GGML_ASSERT(score->type == GGML_TYPE_F32 && pos->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32);
    GGML_ASSERT(score->nb[0] == sizeof(float) && ggml_is_contiguous(pos));
    const int n_top = ggml_get_op_params_i32(dst, 0);
    const int ratio = ggml_get_op_params_i32(dst, 1);
    const int64_t n_tok = score->ne[1];
    GGML_ASSERT(n_tok <= INT_MAX);
    if (n_tok == 0) {
        return;
    }
    qfn_qsa_select_kernel<<<(unsigned) n_tok, QFN_QSA_THREADS, 0, ctx.stream()>>>(
        (const float *) score->data, (int64_t) (score->nb[1] / sizeof(float)),
        (const int32_t *) pos->data,
        (int32_t *) dst->data, (int64_t) (dst->nb[1] / sizeof(int32_t)),
        n_top, ratio, (int) dst->ne[0]);
    CUDA_CHECK(cudaGetLastError());
}
