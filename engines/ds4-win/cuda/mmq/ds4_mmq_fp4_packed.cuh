// DSV4.1 kernel-native packed FP4 routed experts (deepseek41.expert_layout =
// "flashinfer-sm120-mxfp8-mxfp4-v1") for the single-engine decode.
//
// One cache slot holds one expert projection of N rows by K columns:
//   values [N][K/2]      byte i of each 32-element block holds elements 2i
//                        (low nibble) and 2i+1 (high nibble), E2M1 codes;
//   scales [K/128][N][4] E8M0, one per 32-element block.
// This is the byte layout the FlashInfer SM120 W4A8 prefill kernel consumes,
// so decode reads the same bytes the prefill streams, with no repack.
//
// Arithmetic is the canonical MXFP4 vector path's: activations are quantized
// to Q8_1 by the same quantizer, and every 32-element block contributes
//   (e8m0(scale) * 0.5f * d8) * sum(int8(2*E2M1) * q8)
// with an exact integer block sum. Only the float summation order of the
// block terms differs from the canonical kernel.
//
// Included at the end of ds4_mmq.cu (it uses that unit's ggml helpers).

static __device__ __forceinline__ int2 ds4_fp4p_expand(const uint32_t q4) {
    // Eight sequential nibbles -> two ints of four int8 table values each,
    // in element order (unlike get_int_from_table_16, which splits low and
    // high nibbles for the canonical layout).
    const uint32_t *t32 = (const uint32_t *) kvalues_mxfp4;
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
    const uint32_t lo0 = __byte_perm(t32[0], t32[1], q4);
    const uint32_t hi0 = __byte_perm(t32[2], t32[3], q4);
    const uint32_t lo1 = __byte_perm(t32[0], t32[1], q4 >> 16);
    const uint32_t hi1 = __byte_perm(t32[2], t32[3], q4 >> 16);
    return make_int2((int) __byte_perm(lo0, hi0, sel), (int) __byte_perm(lo1, hi1, sel >> 16));
}

// The canonical MXFP4 vector kernels (ds4_mmq_moe_*_q8_1_qwarp32) give each
// row a half-warp: lane l takes 32-element blocks l/2, l/2+8, ... and, within
// a block, half h = l%2 = elements 8h..8h+7 and 16+8h..16+8h+7, adding
// e8m0*0.5*d8 * isum per block half; the half-warp then reduces with
// shfl_down 8,4,2,1. The packed kernels below keep that mapping, order and
// expression, so they reproduce the canonical kernels' results bit for bit.
// In the packed layout those two element groups are words h and h+2 of the
// block's 16 bytes.
// Slots live in the caller's first tier (the expert-cache pool, original
// expert stride) or, read in place, its second (victim) tier.
struct ds4_fp4p_tiers {
    const uint8_t *l1, *l2;
    uint64_t l1_bytes, l2_bytes;
    uint32_t l1_slots;
};
static __device__ __forceinline__ const uint8_t *ds4_fp4p_slot(const ds4_fp4p_tiers &t, const uint32_t id) {
    return id < t.l1_slots ? t.l1 + (uint64_t) id * t.l1_bytes : t.l2 + (uint64_t) (id - t.l1_slots) * t.l2_bytes;
}

static __device__ __forceinline__ float ds4_fp4p_half(
        const uint32_t wlo, const uint32_t whi, const int *q, const int h, const float d) {
    const int2 lo = ds4_fp4p_expand(wlo);   // elements 8h .. 8h+7
    const int2 hi = ds4_fp4p_expand(whi);   // elements 16+8h .. 16+8h+7
    int sumi = 0;
    sumi = ggml_cuda_dp4a(lo.x, q[2 * h + 0], sumi);
    sumi = ggml_cuda_dp4a(hi.x, q[2 * h + 4], sumi);
    sumi = ggml_cuda_dp4a(lo.y, q[2 * h + 1], sumi);
    sumi = ggml_cuda_dp4a(hi.y, q[2 * h + 5], sumi);
    return d * sumi;
}

static __device__ __forceinline__ uint32_t ds4_fp4p_ldw(const uint32_t *p) {
    // Streamed once per token: keep it out of L1.
    uint32_t r;
    asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(r) : "l"(p));
    return r;
}

static __device__ __forceinline__ float ds4_fp4p_half_warp_sum(float v) {
    const uint32_t mask = 0xffffu << (threadIdx.x & 16u);
    for (int offset = 8; offset > 0; offset >>= 1) v += __shfl_down_sync(mask, v, offset, 16);
    return v;
}

// Gate + up + SwiGLU(clamp) * router weight -> mid. One CTA = 16 rows of one
// (token, slot) assignment, one half-warp per row.
template <int K>
static __global__ void __launch_bounds__(256) ds4_fp4p_gate_up_kernel(
        const ds4_fp4p_tiers            Tg,
        const ds4_fp4p_tiers            Tu,
        const block_q8_1 * __restrict__ xq,
        const uint32_t                  xq_stride,
        const int32_t    * __restrict__ ids,
        const float      * __restrict__ weights,
        float            * __restrict__ mid,
        const uint32_t                  N,
        const uint32_t                  n_slots,
        const float                     clamp) {
    constexpr int NB = K / 32, NG = K / 128, R = 16;
    __shared__ int sx[NB * 8];
    __shared__ float sd[NB];
    __shared__ uint32_t ssg[R][NG];
    __shared__ uint32_t ssu[R][NG];
    const uint32_t pair = blockIdx.y, tok = pair / 6u;
    const int32_t id = ids[pair];
    const bool bad = id < 0 || (uint32_t) id >= n_slots;
    const uint32_t n0 = blockIdx.x * R;
    const block_q8_1 *xb = xq + (uint64_t) tok * xq_stride;
    for (int i = threadIdx.x; i < NB * 8; i += 256) sx[i] = ((const int *) xb[i >> 3].qs)[i & 7];
    for (int b = threadIdx.x; b < NB; b += 256) sd[b] = __low2float(xb[b].ds);
    const uint8_t *Gs = bad ? nullptr : ds4_fp4p_slot(Tg, (uint32_t) id);
    const uint8_t *Us = bad ? nullptr : ds4_fp4p_slot(Tu, (uint32_t) id);
    const uint64_t value_bytes = (uint64_t) N * (K / 2);
    if (!bad) {
        const uint32_t *sg = (const uint32_t *) (Gs + value_bytes);
        const uint32_t *su = (const uint32_t *) (Us + value_bytes);
        for (int i = threadIdx.x; i < NG * R; i += 256) {
            const int g = i / R, r = i - g * R;
            ssg[r][g] = sg[(uint64_t) g * N + n0 + r];
            ssu[r][g] = su[(uint64_t) g * N + n0 + r];
        }
    }
    __syncthreads();
    const int lane = threadIdx.x & 15, r = threadIdx.x >> 4, h = lane & 1;
    const uint32_t row = n0 + r;
    float gate = 0.0f, up = 0.0f;
    if (!bad) {
        const uint32_t *vg = (const uint32_t *) (Gs + (uint64_t) row * (K / 2));
        const uint32_t *vu = (const uint32_t *) (Us + (uint64_t) row * (K / 2));
#pragma unroll 4
        for (int b = lane >> 1; b < NB; b += 8) {
            const int *q = sx + b * 8;
            const uint32_t sh = 8u * (uint32_t) (b & 3);
            const float dg = ggml_cuda_e8m0_to_fp32((uint8_t) (ssg[r][b >> 2] >> sh)) * 0.5f * sd[b];
            const float du = ggml_cuda_e8m0_to_fp32((uint8_t) (ssu[r][b >> 2] >> sh)) * 0.5f * sd[b];
            gate += ds4_fp4p_half(ds4_fp4p_ldw(vg + 4 * b + h), ds4_fp4p_ldw(vg + 4 * b + 2 + h), q, h, dg);
            up   += ds4_fp4p_half(ds4_fp4p_ldw(vu + 4 * b + h), ds4_fp4p_ldw(vu + 4 * b + 2 + h), q, h, du);
        }
    }
    gate = ds4_fp4p_half_warp_sum(gate);
    up   = ds4_fp4p_half_warp_sum(up);
    if (lane == 0 && row < N) {
        if (!isfinite(gate)) gate = 0.0f;
        if (!isfinite(up)) up = 0.0f;
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const float silu = gate / (1.0f + expf(-gate));
        mid[(uint64_t) pair * N + row] = silu * up * weights[pair];
    }
}

// Down projection summed over the token's six slots -> out. One CTA = 16
// rows, one half-warp per row; slots summed in order after their reduction.
template <int K>
static __global__ void __launch_bounds__(256) ds4_fp4p_down_sum6_kernel(
        const ds4_fp4p_tiers            Td,
        const block_q8_1 * __restrict__ mq,
        const uint32_t                  mq_stride,
        const int32_t    * __restrict__ ids,
        float            * __restrict__ out,
        const uint32_t                  N,
        const uint32_t                  n_slots) {
    constexpr int NB = K / 32, NG = K / 128, TOT = 6 * NB, R = 16;
    __shared__ int sx[TOT * 8];
    __shared__ float sd[TOT];
    __shared__ uint32_t ss[R][6 * NG];
    __shared__ int32_t sid[6];
    const uint32_t tok = blockIdx.y, n0 = blockIdx.x * R;
    if (threadIdx.x < 6) {
        const int32_t id = ids[tok * 6u + threadIdx.x];
        sid[threadIdx.x] = id < 0 || (uint32_t) id >= n_slots ? -1 : id;
    }
    for (int i = threadIdx.x; i < TOT * 8; i += 256) {
        const int a = i / (NB * 8), rem = i - a * (NB * 8);
        sx[i] = ((const int *) mq[(uint64_t) (tok * 6u + a) * mq_stride + (rem >> 3)].qs)[rem & 7];
    }
    for (int i = threadIdx.x; i < TOT; i += 256) {
        const int a = i / NB, b = i - a * NB;
        sd[i] = __low2float(mq[(uint64_t) (tok * 6u + a) * mq_stride + b].ds);
    }
    __syncthreads();
    const uint64_t value_bytes = (uint64_t) N * (K / 2);
    for (int i = threadIdx.x; i < 6 * NG * R; i += 256) {
        const int a = i / (NG * R), rem = i - a * (NG * R), g = rem / R, r = rem - g * R;
        const int32_t id = sid[a];
        ss[r][a * NG + g] = id < 0 ? 0u :
            ((const uint32_t *) (ds4_fp4p_slot(Td, (uint32_t) id) + value_bytes))[(uint64_t) g * N + n0 + r];
    }
    __syncthreads();
    const int lane = threadIdx.x & 15, r = threadIdx.x >> 4, h = lane & 1;
    const uint32_t row = n0 + r;
    float total = 0.0f;
#pragma unroll
    for (int a = 0; a < 6; ++a) {
        const int32_t id = sid[a];
        float acc = 0.0f;
        if (id >= 0) {
            const uint32_t *v = (const uint32_t *) (ds4_fp4p_slot(Td, (uint32_t) id) + (uint64_t) row * (K / 2));
#pragma unroll 3
            for (int b = lane >> 1; b < NB; b += 8) {
                const int j = a * NB + b;
                const float d = ggml_cuda_e8m0_to_fp32((uint8_t) (ss[r][a * NG + (b >> 2)] >> (8u * (uint32_t) (b & 3)))) * 0.5f * sd[j];
                acc += ds4_fp4p_half(ds4_fp4p_ldw(v + 4 * b + h), ds4_fp4p_ldw(v + 4 * b + 2 + h), sx + j * 8, h, d);
            }
        }
        acc = ds4_fp4p_half_warp_sum(acc);
        if (lane == 0) {
            if (!isfinite(acc)) acc = 0.0f;
            total += acc;
        }
    }
    if (lane == 0 && row < N) out[(uint64_t) tok * N + row] = total;
}

// Lossless packed -> canonical block_mxfp4 (17 B per 32 elements) for one
// (assignment) projection; the reference mode feeds the canonical vector kernels.
static __global__ void ds4_fp4p_to_canonical_kernel(
        const ds4_fp4p_tiers T, const int32_t * __restrict__ ids,
        uint8_t * __restrict__ dst, const uint32_t N, const uint32_t K) {
    const uint32_t a = blockIdx.y;
    const uint64_t t = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t nb = K / 32u;
    if (t >= (uint64_t) N * nb) return;
    const uint32_t n = (uint32_t) (t / nb), b = (uint32_t) (t - (uint64_t) n * nb);
    const uint8_t *s = ds4_fp4p_slot(T, (uint32_t) ids[a]);
    const uint8_t *v = s + (uint64_t) n * (K / 2u) + (uint64_t) b * 16u;
    uint8_t *o = dst + (uint64_t) a * N * nb * 17u + t * 17u;
    o[0] = s[(uint64_t) N * (K / 2u) + ((uint64_t) (b >> 2) * N + n) * 4u + (b & 3u)];
    for (int i = 0; i < 16; ++i) {
        const uint8_t lo = (v[i >> 1] >> (4 * (i & 1))) & 15u;
        const uint8_t hi = (v[(i + 16) >> 1] >> (4 * (i & 1))) & 15u;
        o[1 + i] = (uint8_t) (lo | (hi << 4));
    }
}

static __global__ void ds4_fp4p_scale_kernel(float *x, uint32_t n, float s) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

static __global__ void ds4_fp4p_iota_kernel(int32_t *ids, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ids[i] = (int32_t) i;
}

extern "C" int ds4_mmq_mxfp4_packed_moe(
        const void * W_gate, const void * W_up, const void * W_down,
        uint64_t gate_slot_bytes, uint64_t down_slot_bytes,
        const void * W_gate2, const void * W_up2, const void * W_down2,
        uint64_t gate2_slot_bytes, uint64_t down2_slot_bytes, int l1_slots,
        const float * X_f32, const int32_t * ids, const float * weights,
        float * mid_f32, float * out_f32,
        int mid_dim, int in_dim, int out_dim, int n_tokens, int n_slots,
        float clamp, int reference, cudaStream_t stream) {
    const char *tag = "ds4_mmq_mxfp4_packed_moe";
    if (!W_gate || !W_up || !W_down || !X_f32 || !ids || !weights || !mid_f32 || !out_f32 ||
        mid_dim != 2304 || in_dim != 5120 || out_dim != 5120 || n_tokens < 1 || n_tokens > 65535 || n_slots <= 0 ||
        gate_slot_bytes != (uint64_t) mid_dim * in_dim / 32u * 17u ||
        down_slot_bytes != (uint64_t) out_dim * mid_dim / 32u * 17u || l1_slots <= 0 || l1_slots > n_slots ||
        (l1_slots < n_slots && (!W_gate2 || !W_up2 || !W_down2 || gate2_slot_bytes < gate_slot_bytes || down2_slot_bytes < down_slot_bytes))) {
        fprintf(stderr, "%s: unsupported packed FP4 geometry\n", tag);
        return -1;
    }
    static const int check = [] { const char *v = getenv("DS4_FP4_PACKED_CHECK"); return v && v[0] == '1' && !v[1]; }();
    static bool in_check = false;
    if (check && !reference && n_tokens == 1 && !in_check) {
        in_check = true;
        // Diagnostic: packed kernels and the canonical reference on the same
        // slots; the reference result is kept, the differences are printed.
        static int calls = 0;
        const int rc = ds4_mmq_mxfp4_packed_moe(W_gate, W_up, W_down, gate_slot_bytes, down_slot_bytes, W_gate2, W_up2, W_down2,
                                                gate2_slot_bytes, down2_slot_bytes, l1_slots, X_f32, ids, weights,
                                                mid_f32, out_f32, mid_dim, in_dim, out_dim, 1, n_slots, clamp, 0, stream);
        in_check = false;
        if (rc) return rc;
        std::vector<float> mp((size_t) 6 * mid_dim), op(out_dim), mr((size_t) 6 * mid_dim), orf(out_dim);
        cudaMemcpyAsync(mp.data(), mid_f32, mp.size() * 4, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(op.data(), out_f32, op.size() * 4, cudaMemcpyDeviceToHost, stream);
        const int rr = ds4_mmq_mxfp4_packed_moe(W_gate, W_up, W_down, gate_slot_bytes, down_slot_bytes, W_gate2, W_up2, W_down2,
                                                gate2_slot_bytes, down2_slot_bytes, l1_slots, X_f32, ids, weights,
                                                mid_f32, out_f32, mid_dim, in_dim, out_dim, 1, n_slots, clamp, 1, stream);
        if (rr) return rr;
        cudaMemcpyAsync(mr.data(), mid_f32, mr.size() * 4, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(orf.data(), out_f32, orf.size() * 4, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (calls++ < 120) {
            auto rel = [](const std::vector<float> &a, const std::vector<float> &b, int *worst) {
                double num = 0, den = 0, mx = 0; *worst = 0;
                for (size_t i = 0; i < a.size(); ++i) {
                    const double d = (double) a[i] - b[i]; num += d * d; den += (double) b[i] * b[i];
                    if (fabs(d) > mx) { mx = fabs(d); *worst = (int) i; }
                }
                return den > 0 ? sqrt(num / den) : sqrt(num);
            };
            int wm = 0, wo = 0;
            const double rm = rel(mp, mr, &wm), ro = rel(op, orf, &wo);
            fprintf(stderr, "FP4P_CHECK call=%d mid_relrms=%.3e out_relrms=%.3e mid_worst=%d (%g vs %g) out_worst=%d (%g vs %g)\n",
                    calls - 1, rm, ro, wm, mp[wm], mr[wm], wo, op[wo], orf[wo]);
        }
        return 0;
    }
    const ds4_fp4p_tiers Tg{(const uint8_t *) W_gate, (const uint8_t *) W_gate2, gate_slot_bytes, gate2_slot_bytes, (uint32_t) l1_slots};
    const ds4_fp4p_tiers Tu{(const uint8_t *) W_up, (const uint8_t *) W_up2, gate_slot_bytes, gate2_slot_bytes, (uint32_t) l1_slots};
    const ds4_fp4p_tiers Td{(const uint8_t *) W_down, (const uint8_t *) W_down2, down_slot_bytes, down2_slot_bytes, (uint32_t) l1_slots};
    if (reference) {
        // Exact canonical vector arithmetic on the same bytes: relayout the
        // selected slots, then run the vendored MXFP4 vector kernels.
        if (n_tokens != 1) { fprintf(stderr, "%s: reference mode is single-token only\n", tag); return -1; }
        static uint8_t *canon = nullptr; static int32_t *iota = nullptr;
        const uint64_t gb = gate_slot_bytes * 6u, db = down_slot_bytes * 6u;
        if (!canon) {
            if (cudaMalloc((void **) &canon, 2u * gb + db) != cudaSuccess ||
                cudaMalloc((void **) &iota, 6u * sizeof(int32_t)) != cudaSuccess) {
                fprintf(stderr, "%s: reference scratch allocation failed\n", tag); return -2;
            }
            ds4_fp4p_iota_kernel<<<1, 32, 0, stream>>>(iota, 6u);
        }
        const uint32_t gt = (uint32_t) mid_dim * (in_dim / 32), dt = (uint32_t) out_dim * (mid_dim / 32);
        ds4_fp4p_to_canonical_kernel<<<dim3((gt + 255) / 256, 6), 256, 0, stream>>>(Tg, ids, canon, mid_dim, in_dim);
        ds4_fp4p_to_canonical_kernel<<<dim3((gt + 255) / 256, 6), 256, 0, stream>>>(Tu, ids, canon + gb, mid_dim, in_dim);
        ds4_fp4p_to_canonical_kernel<<<dim3((dt + 255) / 256, 6), 256, 0, stream>>>(Td, ids, canon + 2u * gb, out_dim, mid_dim);
        if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "%s: relayout launch failed\n", tag); return -3; }
        int rc = ds4_mmq_mxfp4_moe_gate_up_mid_vec(canon, canon + gb, X_f32, iota, weights, mid_f32,
                                                    mid_dim, in_dim, 1, 6, 6, clamp, stream);
        if (rc == 0) rc = ds4_mmq_mxfp4_moe_down_sum6_vec(canon + 2u * gb, mid_f32, iota, out_f32,
                                                          out_dim, mid_dim, 1, 6, 6, stream);
        static const int perturb = [] { const char *v = getenv("DS4_FP4_PACKED_PERTURB"); return v && v[0] == '1' && !v[1]; }();
        if (rc == 0 && perturb) ds4_fp4p_scale_kernel<<<(out_dim + 255) / 256, 256, 0, stream>>>(out_f32, (uint32_t) out_dim, 1.0f + 1.0e-6f);
        return rc;
    }
    const int64_t x_pad = GGML_PAD((int64_t) in_dim, MATRIX_ROW_PADDING);
    const int64_t m_pad = GGML_PAD((int64_t) mid_dim, MATRIX_ROW_PADDING);
    const int n_assign = n_tokens * 6;
    // Persistent Q8_1 activation scratch. A per-call pool allocation costs a
    // host commit check plus cudaMallocAsync/cudaFreeAsync twice per layer,
    // which dominated the token time. It grows (never shrinks) with the rows.
    const size_t xq_bytes = ((size_t) n_tokens * x_pad * sizeof(block_q8_1) / QK8_1 + 255u) & ~(size_t) 255u;
    const size_t mq_bytes = (size_t) n_assign * m_pad * sizeof(block_q8_1) / QK8_1;
    static char *scratch = nullptr;
    static size_t scratch_bytes = 0;
    if (scratch_bytes < xq_bytes + mq_bytes) {
        if (scratch) {
            if (cudaStreamSynchronize(stream) != cudaSuccess) { fprintf(stderr, "%s: scratch sync failed\n", tag); return -2; }
            (void) cudaFree(scratch); scratch = nullptr; scratch_bytes = 0;
        }
        if (cudaMalloc((void **) &scratch, xq_bytes + mq_bytes) != cudaSuccess) {
            scratch = nullptr; fprintf(stderr, "%s: scratch allocation failed\n", tag); return -2;
        }
        scratch_bytes = xq_bytes + mq_bytes;
    }
    char *xq = ds4_mmq_folded_q81(X_f32, in_dim, n_tokens, x_pad);
    if (!xq) {
        xq = scratch;
        quantize_row_q8_1_cuda(X_f32, nullptr, (void *) xq, GGML_TYPE_MXFP4, in_dim,
                               in_dim, in_dim, (int64_t) in_dim * n_tokens, x_pad, 1, n_tokens, 1, stream);
    }
    char *mq = scratch + xq_bytes;
    ds4_fp4p_gate_up_kernel<5120><<<dim3((unsigned) mid_dim / 16u, (unsigned) n_assign), 256, 0, stream>>>(
        Tg, Tu, (const block_q8_1 *) xq,
        (uint32_t) (x_pad / QK8_1), ids, weights, mid_f32, (uint32_t) mid_dim, (uint32_t) n_slots, clamp);
    quantize_row_q8_1_cuda(mid_f32, nullptr, (void *) mq, GGML_TYPE_MXFP4, mid_dim,
                           mid_dim, mid_dim, (int64_t) mid_dim * n_assign, m_pad, 1, n_assign, 1, stream);
    ds4_fp4p_down_sum6_kernel<2304><<<dim3((unsigned) out_dim / 16u, (unsigned) n_tokens), 256, 0, stream>>>(
        Td, (const block_q8_1 *) mq, (uint32_t) (m_pad / QK8_1),
        ids, out_f32, (uint32_t) out_dim, (uint32_t) n_slots);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "%s: launch failed: %s\n", tag, cudaGetErrorString(err)); return -3; }
    return 0;
}
