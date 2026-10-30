/* CPU service for RAM-resident packed FP4 experts (single-row decode).
 *
 * Reproduces ds4_fp4p_gate_up_kernel / ds4_fp4p_down_sum6_kernel
 * (cuda/mmq/ds4_mmq_fp4_packed.cuh) on the CPU, operation for operation:
 *   - activations quantized to Q8_1 exactly like quantize_q8_1
 *     (d = amax/127, q = roundf(x/d), d stored as half);
 *   - per 32-element block and half h, an exact integer sum of int8(2*E2M1)*q8;
 *   - per row, 16 "lanes" (lane = 2*(block%8)+h) each accumulating
 *     fma(e8m0*0.5f*d8, float(isum), acc) in block order, then the
 *     half-warp shfl_down 8,4,2,1 tree;
 *   - SwiGLU with the clamp, CUDA's expf range reduction and weight product.
 * The only non-reproduced operation is the GPU's ex2.approx inside expf
 * (here an exactly rounded exp2f of the same reduced argument). The down
 * projection returns each expert's per-row sum; the GPU kernel adds those
 * in slot order, so the merged output follows the GPU's summation order.
 *
 * Persistent worker pool; submit() returns immediately, wait() joins.
 * Weights are read in place from the locked RAM tier: no copies. */
#include <immintrin.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#include <intrin.h>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace {
constexpr int DIM_IN = 5120, DIM_MID = 2304, DIM_OUT = 5120, MAXE = 6;
constexpr int NB_IN = DIM_IN / 32, NB_MID = DIM_MID / 32;

struct q81 {                       // one quantized activation vector, nibble-split
    alignas(64) int8_t xl[DIM_IN / 2]; // per block: q[2i], i = 0..15
    alignas(64) int8_t xh[DIM_IN / 2]; // per block: q[2i+1]
    alignas(64) float sdd[DIM_IN / 16];// per (block, half): half(d), duplicated
};

static inline float half_round(float v) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtps_ph(_mm_set_ss(v), _MM_FROUND_TO_NEAREST_INT)));
}

/* quantize_q8_1 for one 32-block, then split by nibble position. */
static void quantize_split(const float *x, int nb, int8_t *xl, int8_t *xh, float *sdd) {
    for (int b = 0; b < nb; ++b) {
        const float *v = x + 32 * b;
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
        const float d = amax / 127.0f;
        int8_t q[32];
        for (int i = 0; i < 32; ++i) q[i] = amax == 0.0f ? 0 : (int8_t) roundf(v[i] / d);
        for (int i = 0; i < 16; ++i) { xl[16 * b + i] = q[2 * i]; xh[16 * b + i] = q[2 * i + 1]; }
        sdd[2 * b] = sdd[2 * b + 1] = half_round(d);
    }
}

static const __m512i k_abs = _mm512_set4_epi32(0x0c080604, 0x03020100, 0x0c080604, 0x03020100);
static const __m512i k_idx = _mm512_setr_epi32(0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29);

/* 64 packed bytes (4 blocks) against the matching split activations:
 * dwords 0,1 of each 128-bit lane = (half0, half1) integer sums. */
static inline __m512i block4(const uint8_t *w, const int8_t *xl, const int8_t *xh) {
    const __m512i v = _mm512_loadu_si512(w);
    const __m512i m = _mm512_set1_epi8(0x0f);
    const __m512i lo = _mm512_and_si512(v, m);
    const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(v, 4), m);
    const __mmask64 nl = _mm512_test_epi8_mask(lo, _mm512_set1_epi8(8));
    const __mmask64 nh = _mm512_test_epi8_mask(hi, _mm512_set1_epi8(8));
    const __m512i al = _mm512_shuffle_epi8(k_abs, lo), ah = _mm512_shuffle_epi8(k_abs, hi);
    __m512i ql = _mm512_loadu_si512(xl), qh = _mm512_loadu_si512(xh);
    const __m512i z = _mm512_setzero_si512();
    ql = _mm512_mask_sub_epi8(ql, nl, z, ql);
    qh = _mm512_mask_sub_epi8(qh, nh, z, qh);
    __m512i s = _mm512_dpbusd_epi32(z, al, ql);
    s = _mm512_dpbusd_epi32(s, ah, qh);
    return _mm512_add_epi32(s, _mm512_shuffle_epi32(s, _MM_PERM_BADC));
}

/* e8m0 bytes of blocks 8j..8j+7 of a row -> 16 lanes of e8m0*0.5f. */
static inline __m512 scale16(uint32_t s0, uint32_t s1) {
    const __m128i b = _mm_cvtsi64_si128((long long) ((uint64_t) s0 | ((uint64_t) s1 << 32)));
    const __m512i e = _mm512_cvtepu8_epi32(_mm_unpacklo_epi8(b, b));
    __m512i bits = _mm512_slli_epi32(e, 23);
    bits = _mm512_mask_mov_epi32(bits, _mm512_cmpeq_epi32_mask(e, _mm512_setzero_si512()), _mm512_set1_epi32(0x00400000));
    return _mm512_mul_ps(_mm512_castsi512_ps(bits), _mm512_set1_ps(0.5f));
}

/* half-warp shfl_down 8,4,2,1 tree: lane 0's value. */
static inline float tree16(__m512 v) {
    const __m256 a = _mm256_add_ps(_mm512_castps512_ps256(v), _mm512_extractf32x8_ps(v, 1));
    const __m128 b = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    const __m128 c = _mm_add_ps(b, _mm_movehl_ps(b, b));
    return _mm_cvtss_f32(_mm_add_ss(c, _mm_shuffle_ps(c, c, 1)));
}

/* One row of a [N][K] packed projection: K/32 blocks, scales [K/128][N][4]. */
template <int K>
static inline float row_dot(const uint8_t *slot, uint32_t N, uint32_t n, const int8_t *xl, const int8_t *xh, const float *sdd) {
    constexpr int NJ = K / 256;                 // 8 blocks per lane vector
    const uint8_t *v = slot + (uint64_t) n * (K / 2);
    const uint32_t *sc = (const uint32_t *) (slot + (uint64_t) N * (K / 2));
    __m512 acc = _mm512_setzero_ps();
    for (int j = 0; j < NJ; ++j) {
        const __m512i t0 = block4(v + 128 * j, xl + 128 * j, xh + 128 * j);
        const __m512i t1 = block4(v + 128 * j + 64, xl + 128 * j + 64, xh + 128 * j + 64);
        const __m512 is = _mm512_cvtepi32_ps(_mm512_permutex2var_epi32(t0, k_idx, t1));
        const __m512 d = _mm512_mul_ps(scale16(sc[(uint64_t) (2 * j) * N + n], sc[(uint64_t) (2 * j + 1) * N + n]),
                                       _mm512_loadu_ps(sdd + 16 * j));
        acc = _mm512_fmadd_ps(d, is, acc);
    }
    static_assert(K % 256 == 0, "lane layout: 8 blocks per vector");
    return tree16(acc);
}

static inline float cuda_silu_den(float g) {
    // GPU: fma(ex2(r10), 2^j, 1.0f) with the exp of -g.
    const float a = -g;
    auto f = [](uint32_t u) { float x; memcpy(&x, &u, 4); return x; };
    float r2 = fmaf(a, f(0x3BBB989Du), 0.5f);
    float r3 = r2 < 0.f ? 0.f : r2 > 1.f ? 1.f : r2;
    // fma.rm: AVX-512 embedded rounding toward minus infinity.
    const float r6 = _mm_cvtss_f32(_mm_fmadd_round_ss(_mm_set_ss(r3), _mm_set_ss(252.0f), _mm_set_ss(f(0x4B400001u)),
                                                      _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
    float r7 = r6 + f(0xCB40007Fu);
    float r10 = fmaf(a, f(0x3FB8AA3Bu), -r7);
    r10 = fmaf(a, f(0x32A57060u), r10);
    uint32_t u6; memcpy(&u6, &r6, 4);
    const float scale = f(u6 << 23);
    return fmaf(exp2f(r10), scale, 1.0f);
}

struct job_t {
    int k = 0;
    float clamp = 0;
    const uint8_t *gate[MAXE], *up[MAXE], *down[MAXE];
    float w[MAXE];
    float *acc[MAXE] = {};         // per expert: DIM_OUT down-row sums
    const float *x = nullptr;
    /* Optional GPU epilogue: gu = [k][gate 2304, up 2304] raw rows -> mid [k][2304]. */
    int (*epilogue)(void *, const float *, const float *, int, float, float *) = nullptr;
    void *epilogue_ctx = nullptr;
    float *gu = nullptr;
};

struct pool_t {
    std::vector<std::thread> threads;
    std::mutex m;
    std::condition_variable cv, done_cv;
    uint64_t gen = 0;
    bool stop = false;
    int n = 0;
    job_t job;
    alignas(64) std::atomic<int> next1{0}, fin1{0}, next2{0}, fin2{0}, quant_ready{0}, idle{0}, epi_done{0};
    int epi_ok = 1;
    alignas(64) float gu[MAXE * 2 * DIM_MID];
    q81 xq;
    alignas(64) float mid[MAXE * DIM_MID];
    alignas(64) int8_t ml[MAXE][DIM_MID / 2], mh[MAXE][DIM_MID / 2];
    alignas(64) float msd[MAXE][DIM_MID / 16];
    std::atomic<bool> busy{false};
    double ms = 0;
    std::chrono::steady_clock::time_point t0;
};
static pool_t *g = nullptr;
static int (*g_epilogue)(void *, const float *, const float *, int, float, float *) = nullptr;
static void *g_epilogue_ctx = nullptr;
constexpr int CH1 = 32, CH2 = 64;

static void work(int tid) {
    auto &p = *g;
    const job_t &J = p.job;
    // Phase 0: thread 0 quantizes x; others wait (tiny).
    if (tid == 0) {
        quantize_split(J.x, NB_IN, p.xq.xl, p.xq.xh, p.xq.sdd);
        p.quant_ready.store(1, std::memory_order_release);
    } else {
        while (!p.quant_ready.load(std::memory_order_acquire)) _mm_pause();
    }
    // Phase 1: gate+up rows of every expert -> mid.
    const int rows1 = J.k * DIM_MID, chunks1 = rows1 / CH1;
    for (int c; (c = p.next1.fetch_add(1)) < chunks1;) {
        for (int r = c * CH1; r < c * CH1 + CH1; ++r) {
            const int e = r / DIM_MID, n = r % DIM_MID;
            float gt = row_dot<DIM_IN>(J.gate[e], DIM_MID, n, p.xq.xl, p.xq.xh, p.xq.sdd);
            float u = row_dot<DIM_IN>(J.up[e], DIM_MID, n, p.xq.xl, p.xq.xh, p.xq.sdd);
            if (J.epilogue) {        // raw projections; the GPU applies the SwiGLU epilogue
                J.gu[(size_t) e * 2 * DIM_MID + n] = gt;
                J.gu[(size_t) e * 2 * DIM_MID + DIM_MID + n] = u;
                continue;
            }
            if (!std::isfinite(gt)) gt = 0.f;
            if (!std::isfinite(u)) u = 0.f;
            if (J.clamp > 1.0e-6f) {
                if (gt > J.clamp) gt = J.clamp;
                if (u > J.clamp) u = J.clamp;
                if (u < -J.clamp) u = -J.clamp;
            }
            const float silu = gt / cuda_silu_den(gt);
            p.mid[r] = silu * u * J.w[e];
        }
        p.fin1.fetch_add(1, std::memory_order_acq_rel);
    }
    while (p.fin1.load(std::memory_order_acquire) < chunks1) _mm_pause();
    // Phase 1a: SwiGLU on the GPU (its expf is ex2.approx, which the CPU cannot
    // reproduce bit for bit); thread 0 runs the round trip, the others wait.
    if (J.epilogue) {
        if (tid == 0) {
            p.epi_ok = J.epilogue(J.epilogue_ctx, J.gu, J.w, J.k, J.clamp, p.mid);
            p.epi_done.store(1, std::memory_order_release);
        } else {
            while (!p.epi_done.load(std::memory_order_acquire)) _mm_pause();
        }
    }
    // Phase 1b: quantize mid per expert (one expert per thread).
    for (int e = tid; e < J.k; e += p.n) {
        quantize_split(p.mid + (size_t) e * DIM_MID, NB_MID, p.ml[e], p.mh[e], p.msd[e]);
        p.quant_ready.fetch_add(1, std::memory_order_acq_rel);
    }
    while (p.quant_ready.load(std::memory_order_acquire) < 1 + J.k) _mm_pause();
    // Phase 2: down rows, each expert's per-row sum.
    const int chunks2 = DIM_OUT / CH2;
    for (int c; (c = p.next2.fetch_add(1)) < chunks2;) {
        for (int n = c * CH2; n < c * CH2 + CH2; ++n)
            for (int e = 0; e < J.k; ++e)
                J.acc[e][n] = row_dot<DIM_MID>(J.down[e], DIM_OUT, n, p.ml[e], p.mh[e], p.msd[e]);
        p.fin2.fetch_add(1, std::memory_order_acq_rel);
    }
}

static void worker_main(int tid) {
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> l(g->m);
            g->cv.wait(l, [&] { return g->stop || g->gen != seen; });
            if (g->stop) return;
            seen = g->gen;
        }
        work(tid);
        if (g->idle.fetch_add(1, std::memory_order_acq_rel) + 1 == g->n) {
            g->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g->t0).count();
            std::lock_guard<std::mutex> l(g->m);
            g->busy.store(false, std::memory_order_release);
            g->done_cv.notify_all();
        }
    }
}

static bool cpu_ok() {
    int r[4];
    __cpuidex(r, 7, 0);
    const bool avx512f = r[1] & (1 << 16), bw = r[1] & (1 << 30), vl = r[1] & (1u << 31), dq = r[1] & (1 << 17);
    const bool vnni = r[2] & (1 << 11);
    __cpuid(r, 1);
    const bool f16c = r[2] & (1 << 29), fma = r[2] & (1 << 12);
    return avx512f && bw && vl && dq && vnni && f16c && fma;
}
} // namespace

extern "C" int ds4_cpu_fp4_init(int threads) {
    if (g) return g->n;
    if (!cpu_ok()) { fprintf(stderr, "ds4_cpu_fp4: AVX-512 BW/VL/DQ/VNNI + F16C/FMA required\n"); return 0; }
    if (threads < 1) threads = 1;
    if (threads > 32) threads = 32;
    g = new pool_t();
    g->n = threads;
    for (int i = 0; i < threads; ++i) g->threads.emplace_back(worker_main, i);
    return threads;
}

extern "C" void ds4_cpu_fp4_shutdown(void) {
    if (!g) return;
    { std::lock_guard<std::mutex> l(g->m); g->stop = true; g->cv.notify_all(); }
    for (auto &t : g->threads) t.join();
    delete g; g = nullptr;
}

/* Starts the job and returns. x: 5120 floats (stable until wait), payloads in
 * place, acc[e]: 5120 floats each (complete when wait returns). */
extern "C" int ds4_cpu_fp4_submit(const float *x, const uint8_t *const *gate, const uint8_t *const *up,
                                  const uint8_t *const *down, const float *w, int k, float clamp, float *const *acc) {
    if (!g || k < 1 || k > MAXE || !x || !acc || g->busy.load()) return 0;
    job_t &J = g->job;
    J.k = k; J.clamp = clamp; J.x = x;
    for (int e = 0; e < k; ++e) { J.gate[e] = gate[e]; J.up[e] = up[e]; J.down[e] = down[e]; J.w[e] = w[e]; J.acc[e] = acc[e]; }
    J.epilogue = g_epilogue; J.epilogue_ctx = g_epilogue_ctx; J.gu = g->gu;
    g->next1 = 0; g->fin1 = 0; g->next2 = 0; g->fin2 = 0; g->quant_ready = 0; g->idle = 0; g->epi_done = 0; g->epi_ok = 1;
    g->t0 = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> l(g->m);
    g->busy.store(true, std::memory_order_release);
    ++g->gen;
    g->cv.notify_all();
    return 1;
}

/* Blocks until the submitted job completes; returns its service time (ms). */
extern "C" double ds4_cpu_fp4_wait(void) {
    if (!g) return -1;
    if (g->busy.load(std::memory_order_acquire)) {
        // Short spin first: the join usually finds the job finished or nearly.
        for (int i = 0; i < 2000 && g->busy.load(std::memory_order_acquire); ++i) _mm_pause();
        std::unique_lock<std::mutex> l(g->m);
        g->done_cv.wait(l, [] { return !g->busy.load(std::memory_order_acquire); });
    }
    return g->epi_ok ? g->ms : -1;
}

/* Installs the GPU SwiGLU epilogue used by every later job (NULL = CPU SwiGLU,
 * which differs from the GPU only through ex2.approx inside expf). */
extern "C" void ds4_cpu_fp4_set_epilogue(int (*fn)(void *, const float *, const float *, int, float, float *), void *ctx) {
    g_epilogue = fn; g_epilogue_ctx = ctx;
}

extern "C" int ds4_cpu_fp4_busy(void) { return g && g->busy.load(std::memory_order_acquire); }

/* Diagnostic: the quantized mid of the last job (for verification). */
extern "C" const float *ds4_cpu_fp4_mid(void) { return g ? g->mid : nullptr; }
