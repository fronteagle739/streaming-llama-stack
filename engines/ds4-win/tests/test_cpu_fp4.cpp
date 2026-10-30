/* Standalone check of ds4_cpu_fp4.cpp (no GPU): the AVX-512 lane against a
 * literal scalar transcription of ds4_fp4p_gate_up_kernel /
 * ds4_fp4p_down_sum6_kernel and quantize_q8_1 (element order, per-lane fma,
 * shfl_down tree), on random packed experts; then service time per job.
 * Build: cl /O2 /std:c++17 /EHsc /arch:AVX512 tests\test_cpu_fp4.cpp ds4_cpu_fp4.cpp */
#include <immintrin.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" int ds4_cpu_fp4_init(int threads);
extern "C" void ds4_cpu_fp4_shutdown(void);
extern "C" int ds4_cpu_fp4_submit(const float *x, const uint8_t *const *gate, const uint8_t *const *up,
                                  const uint8_t *const *down, const float *w, int k, float clamp, float *const *acc);
extern "C" double ds4_cpu_fp4_wait(void);

static const int8_t kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
static float e8m0(uint8_t x) { uint32_t b = x ? (uint32_t) x << 23 : 0x00400000u; float f; memcpy(&f, &b, 4); return f; }
static float h16(float v) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtps_ph(_mm_set_ss(v), _MM_FROUND_TO_NEAREST_INT))); }

struct Q { std::vector<int8_t> q; std::vector<float> d; };
static Q quant(const float *x, int n) {           // quantize_q8_1: one warp per 32 block
    Q r; r.q.resize(n); r.d.resize(n / 32);
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0; for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[32 * b + i]));
        const float d = amax / 127.0f;
        for (int i = 0; i < 32; ++i) r.q[32 * b + i] = amax == 0.0f ? 0 : (int8_t) roundf(x[32 * b + i] / d);
        r.d[b] = h16(d);
    }
    return r;
}
static int elem(const uint8_t *row, int e) { const uint8_t byte = row[e >> 1]; return kv[(e & 1) ? byte >> 4 : byte & 15]; }
/* One row exactly as a 16-lane half-warp computes it. */
static float row_ref(const uint8_t *slot, int N, int K, int n, const Q &x) {
    const uint8_t *row = slot + (size_t) n * (K / 2);
    const uint8_t *sc = slot + (size_t) N * (K / 2);
    float lane[16];
    for (int l = 0; l < 16; ++l) {
        float acc = 0; const int h = l & 1;
        for (int b = l >> 1; b < K / 32; b += 8) {
            const float d = e8m0(sc[((size_t) (b >> 2) * N + n) * 4 + (b & 3)]) * 0.5f * x.d[b];
            int s = 0;
            for (int i = 0; i < 8; ++i) s += elem(row + 16 * b, 8 * h + i) * x.q[32 * b + 8 * h + i];
            for (int i = 0; i < 8; ++i) s += elem(row + 16 * b, 16 + 8 * h + i) * x.q[32 * b + 16 + 8 * h + i];
            acc = fmaf(d, (float) s, acc);
        }
        lane[l] = acc;
    }
    for (int off = 8; off > 0; off >>= 1) for (int l = 0; l < off; ++l) lane[l] += lane[l + off];
    return lane[0];
}
static float silu_den(float g) {                // CUDA sm_120 expf sequence, ex2 exact
    auto f = [](uint32_t u) { float x; memcpy(&x, &u, 4); return x; };
    const float a = -g;
    float r2 = fmaf(a, f(0x3BBB989Du), 0.5f); r2 = r2 < 0 ? 0 : r2 > 1 ? 1 : r2;
    const float r6 = _mm_cvtss_f32(_mm_fmadd_round_ss(_mm_set_ss(r2), _mm_set_ss(252.0f), _mm_set_ss(f(0x4B400001u)), _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
    const float r7 = r6 + f(0xCB40007Fu);
    float r10 = fmaf(a, f(0x3FB8AA3Bu), -r7); r10 = fmaf(a, f(0x32A57060u), r10);
    uint32_t u; memcpy(&u, &r6, 4);
    return fmaf(exp2f(r10), f(u << 23), 1.0f);
}

int main(int argc, char **argv) {
    const int IN = 5120, MID = 2304, OUT = 5120;
    const size_t GB = (size_t) MID * IN / 32 * 17, DB = (size_t) OUT * MID / 32 * 17;
    const int pool = argc > 1 ? atoi(argv[1]) : 24;          // distinct experts (defeat L3)
    std::mt19937 rng(1234);
    std::vector<uint8_t> G(GB * pool), U(GB * pool), D(DB * pool);
    auto fill = [&](std::vector<uint8_t> &v, size_t slot, int N, int K) {
        for (size_t s = 0; s < (size_t) pool; ++s) {
            uint8_t *p = v.data() + s * slot;
            for (size_t i = 0; i < (size_t) N * K / 2; ++i) p[i] = (uint8_t) rng();
            for (size_t i = 0; i < (size_t) N * K / 32; ++i) p[(size_t) N * K / 2 + i] = (uint8_t) (110 + rng() % 12);
        }
    };
    fill(G, GB, MID, IN); fill(U, GB, MID, IN); fill(D, DB, OUT, MID);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(IN); for (auto &v : x) v = nd(rng);
    const float clamp = 10.0f;
    const float w[6] = {0.31f, 0.22f, 0.17f, 0.12f, 0.1f, 0.08f};
    if (!ds4_cpu_fp4_init(argc > 2 ? atoi(argv[2]) : 12)) return 2;
    // Correctness: k = 6 experts 0..5 vs the scalar transcription.
    std::vector<std::vector<float>> acc(6, std::vector<float>(OUT));
    const uint8_t *pg[6], *pu[6], *pd[6]; float *pa[6];
    for (int e = 0; e < 6; ++e) { pg[e] = &G[e * GB]; pu[e] = &U[e * GB]; pd[e] = &D[e * DB]; pa[e] = acc[e].data(); }
    ds4_cpu_fp4_submit(x.data(), pg, pu, pd, w, 6, clamp, pa); ds4_cpu_fp4_wait();
    const Q xq = quant(x.data(), IN);
    size_t same = 0, total = 0; double maxrel = 0;
    for (int e = 0; e < 6; ++e) {
        std::vector<float> mid(MID);
        for (int n = 0; n < MID; ++n) {
            float g = row_ref(pg[e], MID, IN, n, xq), u = row_ref(pu[e], MID, IN, n, xq);
            if (!std::isfinite(g)) g = 0; if (!std::isfinite(u)) u = 0;
            if (g > clamp) g = clamp; if (u > clamp) u = clamp; if (u < -clamp) u = -clamp;
            mid[n] = g / silu_den(g) * u * w[e];
        }
        const Q mq = quant(mid.data(), MID);
        for (int n = 0; n < OUT; ++n) {
            const float r = row_ref(pd[e], OUT, MID, n, mq);
            same += memcmp(&r, &acc[e][n], 4) == 0; ++total;
            maxrel = std::max(maxrel, (double) fabsf(r - acc[e][n]) / std::max(1e-30, (double) fabsf(r)));
        }
    }
    printf("bit_identical=%zu/%zu max_rel=%.3e\n", same, total, maxrel);
    // Timing: rotate through the pool so weights come from DRAM.
    for (int k = 1; k <= 6; ++k) {
        double sum = 0; int reps = 40;
        for (int r = 0; r < reps; ++r) {
            for (int e = 0; e < k; ++e) { const int s = (r * 6 + e) % pool; pg[e] = &G[s * GB]; pu[e] = &U[s * GB]; pd[e] = &D[s * DB]; }
            const auto t = std::chrono::steady_clock::now();
            ds4_cpu_fp4_submit(x.data(), pg, pu, pd, w, k, clamp, pa); ds4_cpu_fp4_wait();
            sum += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
        }
        printf("k=%d wall_ms=%.3f per_expert_ms=%.3f GBps=%.1f\n", k, sum / reps, sum / reps / k, (GB * 2 + DB) * k / (sum / reps) / 1e6);
    }
    ds4_cpu_fp4_shutdown();
    return same == total ? 0 : 1;
}
