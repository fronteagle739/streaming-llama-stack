// QFNSTREAM: QSA sparse attention, ported from antirez/ds4 (MIT) ds4_qwen4_cuda.cuh attention_group and its
// tensor-core helpers (ds4_cuda.cu tt_*). One 256-thread block per (KV head, token): the heads of a KV group share
// the token's cell list, both products run on m16n8k16 tensor cores with the queries and probabilities split into
// half-precision high/low parts (FP32-level accuracy), K and V are the F16 caches. Changes from ds4: explicit cache
// strides, the per-token position bound comes from pos, the gate is optional, and the list stops at its first -1.
#include "qfn-qsa.cuh"
#include "qfn-attention.cuh"
#include "ggml-qfn-service.h"

#include <climits>

#define QFN_QSA_D 256

static __device__ __forceinline__ int qfn_mma_c_i(uint32_t lane, int l) { return ((l >> 1) << 3) + (int) (lane >> 2); }
static __device__ __forceinline__ int qfn_mma_c_j(uint32_t lane, int l) { return (int) ((lane & 3u) << 1) + (l & 1); }
static __device__ __forceinline__ unsigned qfn_smem(const void * p) { return (unsigned) __cvta_generic_to_shared(p); }

static __device__ __forceinline__ void qfn_ldmatrix_x4(uint32_t (&r)[4], const void * p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(qfn_smem(p)));
#else
    GGML_UNUSED(p); r[0] = r[1] = r[2] = r[3] = 0;
#endif
}
static __device__ __forceinline__ void qfn_ldmatrix_x2(uint32_t (&r)[2], const void * p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(qfn_smem(p)));
#else
    GGML_UNUSED(p); r[0] = r[1] = 0;
#endif
}
static __device__ __forceinline__ void qfn_ldmatrix_x2_trans(uint32_t (&r)[2], const void * p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(qfn_smem(p)));
#else
    GGML_UNUSED(p); r[0] = r[1] = 0;
#endif
}
static __device__ __forceinline__ void qfn_mma_f16_f32(float * d, const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t d0 = __float_as_uint(d[0]), d1 = __float_as_uint(d[1]), d2 = __float_as_uint(d[2]), d3 = __float_as_uint(d[3]);
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                 : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    d[0] = __uint_as_float(d0); d[1] = __uint_as_float(d1); d[2] = __uint_as_float(d2); d[3] = __uint_as_float(d3);
#else
    GGML_UNUSED(d); GGML_UNUSED(a); GGML_UNUSED(b);
#endif
}
static __device__ __forceinline__ void qfn_cp_async_16B(void * dst, const void * src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (pred) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(qfn_smem(dst)), "l"(src));
    } else {
        *reinterpret_cast<int4 *>(dst) = make_int4(0, 0, 0, 0);
    }
#else
    GGML_UNUSED(src); GGML_UNUSED(pred); *reinterpret_cast<int4 *>(dst) = make_int4(0, 0, 0, 0);
#endif
}
static __device__ __forceinline__ void qfn_cp_async_commit_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
#endif
}
static __device__ __forceinline__ float qfn_sigmoid(float x) {
    const float e = expf(-fabsf(x));
    return x >= 0 ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

static __global__ void __launch_bounds__(256) qfn_qsa_attn_kernel(
        float * __restrict__ out, const float * __restrict__ q, const float * __restrict__ gate,
        const __half * __restrict__ kc, const __half * __restrict__ vc, const int64_t cell_stride,
        const int32_t * __restrict__ sel, const int64_t sel_stride, const int n_sel,
        const int32_t * __restrict__ pos, const int H, const int Hkv, const float scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr unsigned D = QFN_QSA_D;
    const unsigned tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const unsigned t = blockIdx.y, kh = blockIdx.x, group = H / Hkv;
    const unsigned qr = tid / 16, col = tid % 16, h = kh * group + qr;
    const unsigned pmax = (unsigned) pos[t];
    const int32_t * cells = sel + (int64_t) t * sel_stride;
    __shared__ __align__(32) __half qh[16][264], ql[16][264], kv[32][264];
    union Scores { float part[2][16][32]; __half prob[2][16][40]; };
    __shared__ __align__(32) Scores scores;
    __shared__ float qs[16], max_score[16], denom[16], correction[16];
    __shared__ unsigned positions[32];
    float mx = 0;
    for (unsigned d = col; d < D; d += 16) {
        if (qr < group) mx = fmaxf(mx, fabsf(q[((uint64_t) t * H + h) * D + d] * scale));
    }
    for (unsigned off = 8; off; off /= 2) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off, 16));
    const int ex = mx > 0 ? max(-120, min(120, (int) ((__float_as_uint(mx) >> 23) & 255) - 127)) : 0;
    const float inv = ldexpf(1, -ex);
    if (!col) { qs[qr] = ldexpf(1, ex); max_score[qr] = -3e38f; denom[qr] = 0; }
    for (unsigned d = col; d < D; d += 16) {
        const float vq = qr < group ? q[((uint64_t) t * H + h) * D + d] * scale * inv : 0;
        qh[qr][d] = __float2half_rn(vq);
        ql[qr][d] = __float2half_rn((vq - __half2float(qh[qr][d])) * 4096);
    }
    float result[4][4] = {};
    __syncthreads();
    for (int j0 = 0; j0 < n_sel; j0 += 32) {
        if (tid < 32) {
            const int j = j0 + (int) tid;
            const int c = j < n_sel ? cells[j] : -1;
            positions[tid] = (c >= 0 && (unsigned) c <= pmax) ? (unsigned) c : UINT_MAX;
        }
        __syncthreads();
        if (positions[0] == UINT_MAX) {
            break;  // valid cells come first: an invalid first entry ends the list (uniform across the block)
        }
        for (unsigned i = tid * 8; i < 32 * D; i += 256 * 8) {
            const unsigned r = i / D, d = i % D, p = positions[r];
            qfn_cp_async_16B(&kv[r][d], kc + (int64_t) (p == UINT_MAX ? 0 : p) * cell_stride + (int64_t) kh * D + d, p != UINT_MAX);
        }
        qfn_cp_async_commit_wait();
        __syncthreads();
        float hi[4] = {}, lo[4] = {};
        const unsigned split = warp / 4, key0 = (warp % 4) * 8;
        for (unsigned k = split * 128; k < (split + 1) * 128; k += 16) {
            uint32_t ah[4], al[4], b[2];
            qfn_ldmatrix_x4(ah, &qh[lane % 16][k + (lane / 16) * 8]);
            qfn_ldmatrix_x4(al, &ql[lane % 16][k + (lane / 16) * 8]);
            qfn_ldmatrix_x2(b, &kv[key0 + lane % 8][k + ((lane % 16) / 8) * 8]);
            qfn_mma_f16_f32(hi, ah, b);
            qfn_mma_f16_f32(lo, al, b);
        }
#pragma unroll
        for (unsigned i = 0; i < 4; i++) {
            scores.part[split][qfn_mma_c_i(lane, i)][key0 + qfn_mma_c_j(lane, i)] = hi[i] + lo[i] * 0x1p-12f;
        }
        __syncthreads();
        float prob[2], peak = max_score[qr];
#pragma unroll
        for (unsigned i = 0; i < 2; i++) {
            const unsigned key = col + i * 16;
            prob[i] = positions[key] != UINT_MAX ? (scores.part[0][qr][key] + scores.part[1][qr][key]) * qs[qr] : -3e38f;
            peak = fmaxf(peak, prob[i]);
        }
        for (unsigned off = 8; off; off /= 2) peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, off, 16));
        const float old = expf(max_score[qr] - peak);
        float total = 0;
#pragma unroll
        for (unsigned i = 0; i < 2; i++) {
            prob[i] = positions[col + i * 16] != UINT_MAX ? expf(prob[i] - peak) : 0;
            total += prob[i];
        }
        for (unsigned off = 8; off; off /= 2) total += __shfl_xor_sync(0xffffffff, total, off, 16);
        __syncthreads();
        if (!col) {
            correction[qr] = old;
            max_score[qr]  = peak;
            denom[qr]      = denom[qr] * old + total;
        }
#pragma unroll
        for (unsigned i = 0; i < 2; i++) {
            const __half p = __float2half_rn(prob[i]);
            scores.prob[0][qr][col + i * 16] = p;
            scores.prob[1][qr][col + i * 16] = __float2half_rn((prob[i] - __half2float(p)) * 4096);
        }
        for (unsigned i = tid * 8; i < 32 * D; i += 256 * 8) {
            const unsigned r = i / D, d = i % D, p = positions[r];
            qfn_cp_async_16B(&kv[r][d], vc + (int64_t) (p == UINT_MAX ? 0 : p) * cell_stride + (int64_t) kh * D + d, p != UINT_MAX);
        }
        qfn_cp_async_commit_wait();
        __syncthreads();
#pragma unroll
        for (unsigned tile = 0; tile < 4; tile++) {
            float hi2[4] = {}, lo2[4] = {};
#pragma unroll
            for (unsigned k = 0; k < 32; k += 16) {
                uint32_t ah[4], al[4], b[2];
                qfn_ldmatrix_x4(ah, &scores.prob[0][lane % 16][k + (lane / 16) * 8]);
                qfn_ldmatrix_x4(al, &scores.prob[1][lane % 16][k + (lane / 16) * 8]);
                qfn_ldmatrix_x2_trans(b, &kv[k + lane % 16][warp * 32 + tile * 8]);
                qfn_mma_f16_f32(hi2, ah, b);
                qfn_mma_f16_f32(lo2, al, b);
            }
#pragma unroll
            for (unsigned i = 0; i < 4; i++) {
                result[tile][i] = result[tile][i] * correction[qfn_mma_c_i(lane, i)] + (hi2[i] + lo2[i] * 0x1p-12f);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned tile = 0; tile < 4; tile++) {
#pragma unroll
        for (unsigned i = 0; i < 4; i++) {
            const unsigned r = qfn_mma_c_i(lane, i), d = warp * 32 + tile * 8 + qfn_mma_c_j(lane, i);
            if (r < group) {
                const uint64_t o = ((uint64_t) t * H + kh * group + r) * D + d;
                const float val = denom[r] > 0 ? result[tile][i] / denom[r] : 0;
                out[o] = gate ? val * qfn_sigmoid(gate[o]) : val;
            }
        }
    }
#else
    GGML_UNUSED(out); GGML_UNUSED(q); GGML_UNUSED(gate); GGML_UNUSED(kc); GGML_UNUSED(vc); GGML_UNUSED(cell_stride);
    GGML_UNUSED(sel); GGML_UNUSED(sel_stride); GGML_UNUSED(n_sel); GGML_UNUSED(pos); GGML_UNUSED(H); GGML_UNUSED(Hkv); GGML_UNUSED(scale);
    NO_DEVICE_CODE;
#endif
}

void ggml_cuda_op_qfn_qsa_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_get_op_params_i32(dst,1)) {
        GGML_ASSERT(ggml_qfn_attention_requested(dst));ggml_cuda_op_qfn_basalt_attention(ctx,dst);return;
    }
    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * gate = dst->src[1];
    const ggml_tensor * k    = dst->src[2];
    const ggml_tensor * v    = dst->src[3];
    const ggml_tensor * sel  = dst->src[4];
    const ggml_tensor * pos  = dst->src[5];
    GGML_ASSERT(q->ne[0] == QFN_QSA_D && k->nb[2] == v->nb[2] && k->nb[2] % sizeof(half) == 0);
    const int H = (int) q->ne[1], Hkv = (int) k->ne[1];
    GGML_ASSERT(H % Hkv == 0 && H / Hkv <= 16);
    const int64_t n_tok = q->ne[2];
    GGML_ASSERT(n_tok <= 65535);
    if (n_tok == 0) {
        return;
    }
    const dim3 grid((unsigned) Hkv, (unsigned) n_tok, 1);
    qfn_qsa_attn_kernel<<<grid, 256, 0, ctx.stream()>>>(
        (float *) dst->data, (const float *) q->data, gate ? (const float *) gate->data : nullptr,
        (const __half *) k->data, (const __half *) v->data, (int64_t) (k->nb[2] / sizeof(half)),
        (const int32_t *) sel->data, (int64_t) (sel->nb[1] / sizeof(int32_t)), (int) sel->ne[0],
        (const int32_t *) pos->data, H, Hkv, ggml_get_op_params_f32(dst, 0));
    CUDA_CHECK(cudaGetLastError());
}

// ---- QFNSTREAM: GGML_OP_QFN_SLOT_MAP (see ggml_qfn_slot_map in ggml.h).

// One block. Negative timeout records only the first miss for host replay. Otherwise publish, ring (req), then wait
// for the host's acknowledgement. The table is re-read with volatile loads after the wait: the host rewrote it.
static __global__ void k_qfn_slot_map(const int32_t * sel, const volatile int32_t * table, const int32_t * seqp,
                                      ggml_qfn_bell * bell, int32_t * record_latch, int layer, int n, int ne, int32_t * out) {
    __shared__ int miss;
    if (threadIdx.x == 0) miss = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int e = sel[i];
        if (e < 0 || e >= ne || table[e] < 0) atomicAdd(&miss, 1);
    }
    __syncthreads();
    if (miss && bell != nullptr) {
        if (threadIdx.x == 0) {
            const int32_t req = seqp[0] * 64 + layer;
            if (bell->timeout_ms >= 0 && bell->error != 0) {
                // Preserve the FIRST timed-out request/selection while the host may still own it.
                // Later nodes complete with safe placeholder IDs; this whole failed graph is rejected.
                // Sequential slot-map nodes need no unsupported mapped-host atomic operation.
            } else if (bell->timeout_ms < 0) {
                // QFNV3OPT: record the first miss only; no device wait, host restores and replays.
                // WDDM mapped host allocations cannot receive native GPU atomics on this RTX.
                // A device latch preserves atomic first-winner ownership and the mapped host record.
                int32_t * latch = record_latch ? record_latch : (int32_t *) &bell->error;
                if (atomicCAS(latch, 0, req) == 0) {
                    bell->layer = layer;
                    bell->n = n;
                    for (int i = 0; i < n; ++i) bell->sel[i] = sel[i];
                    __threadfence_system();
                    bell->error = req;
                    __threadfence_system();
                }
            } else {
                bell->layer = layer;
                bell->n = n;
                for (int i = 0; i < n; ++i) bell->sel[i] = sel[i];
                __threadfence_system();
                bell->req = req;
                __threadfence_system();
                // bounded: a host that never answers must not hang the GPU into a driver reset (other work shares it)
                unsigned long long t0, t;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
                while (bell->ack != req) {
    #if __CUDA_ARCH__ >= 700
                    __nanosleep(500);
    #endif
                    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
                    if (t - t0 > (unsigned long long) (bell->timeout_ms > 0 ? bell->timeout_ms : 500) * 1000000ull) { bell->error = req; break; }
                }
                __threadfence_system();
            }
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int e = sel[i];
        const int s = (e >= 0 && e < ne) ? table[e] : -1;
        out[i] = s >= 0 ? s : 0;   // never an invalid slot for the expert kernels; a timed-out step is discarded by the host
    }
}

void ggml_cuda_op_qfn_slot_map(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * sel = dst->src[0], * table = dst->src[1], * seq = dst->src[2];
    const uint64_t lo = (uint32_t) ggml_get_op_params_i32(dst, 0), hi = (uint32_t) ggml_get_op_params_i32(dst, 1);
    ggml_qfn_bell * bell = (ggml_qfn_bell *) (uintptr_t) (lo | (hi << 32));
    const int layer = ggml_get_op_params_i32(dst, 2);
    const uint64_t latch_lo = (uint32_t) ggml_get_op_params_i32(dst, 3), latch_hi = (uint32_t) ggml_get_op_params_i32(dst, 4);
    int32_t * record_latch = (int32_t *) (uintptr_t) (latch_lo | (latch_hi << 32));
    const int n = (int) ggml_nelements(dst);
    k_qfn_slot_map<<<1, 64, 0, ctx.stream()>>>((const int32_t *) sel->data, (const volatile int32_t *) table->data,
                                               (const int32_t *) seq->data, bell, record_latch, layer, n, (int) table->ne[0], (int32_t *) dst->data);
    CUDA_CHECK(cudaGetLastError());
}

// ---- QFNSTREAM: fused hyper-connection mix / combine (same arithmetic and order as the unfused ggml graph)
static __global__ void k_qfn_hc_mix(const float * x, const float * g, float * o, int64_t E, int64_t HC, int64_t NT, float scale) {
    const int64_t r = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= E * NT) return;
    const int64_t i = r % E, t = r / E;
    const float * xt = x + t * E * HC, * gt = g + t * E * HC;
    // explicit round-to-nearest ops: no FMA contraction, so each product and sum rounds as in the unfused graph
    float acc = __fmul_rn(xt[i], 1.0f / (1.0f + expf(-gt[i])));
    for (int64_t h = 1; h < HC; ++h) acc = __fadd_rn(acc, __fmul_rn(xt[h * E + i], 1.0f / (1.0f + expf(-gt[h * E + i]))));
    o[t * E + i] = __fmul_rn(acc, scale);
}

static __global__ void k_qfn_hc_combine(const float * rs, const float * bk, const float * ij, float * o, int64_t E, int64_t HC, int64_t NT) {
    const int64_t r = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= E * HC * NT) return;
    const int64_t i = r % E, h = (r / E) % HC, t = r / (E * HC);
    const float w = __fmul_rn(1.0f / (1.0f + expf(-ij[t * HC + h])), 2.0f);
    o[r] = __fadd_rn(rs[r], __fmul_rn(bk[t * E + i], w));
}

void ggml_cuda_op_qfn_hc_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * xn = dst->src[0], * gate = dst->src[1];
    const int64_t E = xn->ne[0], HC = xn->ne[1], NT = xn->ne[2];
    const int64_t n = E * NT;
    k_qfn_hc_mix<<<(unsigned) ((n + 255) / 256), 256, 0, ctx.stream()>>>((const float *) xn->data, (const float *) gate->data,
                                                                        (float *) dst->data, E, HC, NT, ggml_get_op_params_f32(dst, 0));
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_qfn_hc_combine(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * rs = dst->src[0], * bk = dst->src[1], * ij = dst->src[2];
    const int64_t E = rs->ne[0], HC = rs->ne[1], NT = rs->ne[2];
    const int64_t n = E * HC * NT;
    k_qfn_hc_combine<<<(unsigned) ((n + 255) / 256), 256, 0, ctx.stream()>>>((const float *) rs->data, (const float *) bk->data,
                                                                            (const float *) ij->data, (float *) dst->data, E, HC, NT);
    CUDA_CHECK(cudaGetLastError());
}

// Diagnostic-only future-fence service copy: all bytes precede slot/ack publication.
static __global__ void k_qfn_doorbell_micro_copy(const uint8_t * source, uint8_t * destination, uint64_t bytes,
                                               int32_t * table, ggml_qfn_bell * bell,
                                               const volatile int32_t * aborted) {
    if (!bytes) return; // module warmup outside capture has no data access
    __shared__ int skip;
    if (threadIdx.x==0) skip=*aborted;
    __syncthreads();
    if (skip) return; // aborted descriptors never consume DS bytes or acknowledge a successful miss
    auto * output=reinterpret_cast<uint4 *>(destination);
    const auto * input=reinterpret_cast<const uint4 *>(source);
    for (uint64_t i=threadIdx.x;i<bytes/16;i+=blockDim.x) output[i]=input[i];
    __threadfence_system(); // each producer publishes its own stores before the block acknowledgement
    __syncthreads();
    if (threadIdx.x==0) {
        __threadfence_system();
        table[0]=7;
        __threadfence_system();
        bell->ack=bell->req;
        __threadfence_system();
    }
}
extern "C" int ggml_backend_cuda_qfn_doorbell_micro_prepare(void ** function) {
    if (!function) return int(cudaErrorInvalidValue);
    k_qfn_doorbell_micro_copy<<<1,256>>>(nullptr,nullptr,0,nullptr,nullptr,nullptr);
    const auto launch=cudaGetLastError();
    if (launch!=cudaSuccess) return int(launch);
    const auto complete=cudaDeviceSynchronize();
    if (complete==cudaSuccess) *function=reinterpret_cast<void *>(k_qfn_doorbell_micro_copy);
    return int(complete);
}

static __device__ bool qfn_service_valid(const volatile ggml_qfn_service_record * record) {
    if (!record) return false; // preparation warmup only
    const auto * control = reinterpret_cast<const volatile ggml_qfn_service_control *>(record->control);
    if (!control || control->aborted || !record->active) return false;
    const bool valid = record->magic == GGML_QFN_SERVICE_MAGIC && record->table && record->bell &&
        record->ticket >= control->first_ticket && record->ticket <= control->last_ticket &&
        record->demotions <= GGML_QFN_SERVICE_MAX && record->fills <= GGML_QFN_SERVICE_MAX &&
        record->updates <= 2 * GGML_QFN_SERVICE_MAX;
    if (!valid && blockIdx.x == 0 && threadIdx.x == 0)
        reinterpret_cast<volatile ggml_qfn_service_control *>(record->control)->error = 1;
    return valid;
}

static __device__ void qfn_service_copy_bytes(uint64_t source, uint64_t destination, uint64_t bytes,
                                             uint64_t thread, uint64_t stride) {
    const auto * src = reinterpret_cast<const uint8_t *>(source);
    auto * dst = reinterpret_cast<uint8_t *>(destination);
    if (((source | destination) & 15) == 0) {
        const auto * vector_src = reinterpret_cast<const uint4 *>(src);
        auto * vector_dst = reinterpret_cast<uint4 *>(dst);
        for (uint64_t i = thread; i < bytes / 16; i += stride) vector_dst[i] = vector_src[i];
        for (uint64_t i = (bytes / 16) * 16 + thread; i < bytes; i += stride) dst[i] = src[i];
    } else {
        for (uint64_t i = thread; i < bytes; i += stride) dst[i] = src[i];
    }
}

static __global__ void k_qfn_service_demote(const volatile ggml_qfn_service_record * record) {
    __shared__ int proceed;
    if (threadIdx.x == 0) proceed = qfn_service_valid(record);
    __syncthreads();
    if (!proceed) return;
    // ONE block retains original stream order even if a tiny RAM tier chooses
    // the same destination more than once. Fill cannot start until this node ends.
    for (uint32_t i = 0; i < record->demotions; ++i) {
        qfn_service_copy_bytes(record->demote[i].source, record->demote[i].destination,
                               record->demote[i].bytes, threadIdx.x, blockDim.x);
        __threadfence_system();
        __syncthreads();
    }
}

static __global__ void k_qfn_service_fill(const volatile ggml_qfn_service_record * record) {
    __shared__ int proceed;
    if (threadIdx.x == 0) proceed = qfn_service_valid(record);
    __syncthreads();
    if (!proceed) return;
    const uint64_t thread = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    for (uint32_t i = 0; i < record->fills; ++i)
        qfn_service_copy_bytes(record->fill[i].source, record->fill[i].destination,
                               record->fill[i].bytes, thread, stride);
    // Every producer makes its writes visible before the dependent publication.
    __threadfence_system();
}

static __global__ void k_qfn_service_publish(const volatile ggml_qfn_service_record * record) {
    if (!qfn_service_valid(record) || threadIdx.x || blockIdx.x) return;
    auto * control = reinterpret_cast<volatile ggml_qfn_service_control *>(record->control);
    auto * table = reinterpret_cast<volatile int32_t *>(record->table);
    auto * bell = reinterpret_cast<ggml_qfn_bell *>(record->bell);
    for (uint32_t i = 0; i < record->updates; ++i) {
        if (record->update[i].index >= record->table_entries) { control->error = 1; return; }
        table[record->update[i].index] = record->update[i].slot;
    }
    __threadfence_system();
    if (control->aborted || control->error || bell->error || bell->req != record->request) return;
    bell->ack = record->request;
    __threadfence_system();
}

extern "C" int ggml_backend_cuda_qfn_service_prepare(void ** demote, void ** fill, void ** publish) {
    if (!demote || !fill || !publish) return int(cudaErrorInvalidValue);
    k_qfn_service_demote<<<1,256>>>(nullptr);
    k_qfn_service_fill<<<16,256>>>(nullptr);
    k_qfn_service_publish<<<1,1>>>(nullptr);
    auto status = cudaGetLastError();
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    if (status == cudaSuccess) {
        *demote = reinterpret_cast<void *>(k_qfn_service_demote);
        *fill = reinterpret_cast<void *>(k_qfn_service_fill);
        *publish = reinterpret_cast<void *>(k_qfn_service_publish);
    }
    return int(status);
}
