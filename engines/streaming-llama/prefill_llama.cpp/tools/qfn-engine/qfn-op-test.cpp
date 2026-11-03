// QFNSTREAM op checks: each new ggml type/op on CUDA against the CPU backend (and an F32 reference where the
// two backends quantize activations differently). Exit code 0 only if every case passes.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml-qfn-nvfp4.h"
#include "ggml-qfn-expert-reduce.h"
#include "../../../../qfn-basalt-nvfp4/qfn_basalt_nvfp4.h"
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

struct backend_pair {
    ggml_backend_t cpu = nullptr, cuda = nullptr;
};

int g_fail = 0;

// Builds the graph with `build` twice (once per backend), fills inputs with `fill`, returns every output tensor's
// data as float (I32 outputs are converted) for comparison.
struct run_result { std::vector<std::vector<float>> outs; };

run_result run(ggml_backend_t be, const std::function<std::vector<ggml_tensor *>(ggml_context *, std::vector<ggml_tensor *> &)> & build,
               const std::function<void(std::vector<ggml_tensor *> &)> & fill, bool required_static = false) {
    ggml_init_params ip = { 256 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<ggml_tensor *> inputs;
    std::vector<ggml_tensor *> outs = build(ctx, inputs);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    fill(inputs);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (auto * o : outs) ggml_build_forward_expand(gf, o);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(ga, gf);
    // inputs were allocated in buf before graph allocation; fill again in case gallocr placed views
    fill(inputs);
    if (required_static) {
        ggml_backend_cuda_graphs_disable_pop(be);
        if (ggml_backend_cuda_graph_compute_static(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "required capture failed\n"); g_fail++; }
        ggml_backend_synchronize(be);
        fill(inputs); // changed inputs at stable addresses, staged outside capture
        if (ggml_backend_cuda_graph_compute_static(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "required replay failed\n"); g_fail++; }
        ggml_backend_cuda_graphs_disable_push(be);
    } else if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); g_fail++; }
    run_result r;
    for (auto * o : outs) {
        std::vector<float> v(ggml_nelements(o));
        if (o->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(o, v.data(), 0, v.size() * 4);
        } else if (o->type == GGML_TYPE_I32) {
            std::vector<int32_t> iv(v.size());
            ggml_backend_tensor_get(o, iv.data(), 0, iv.size() * 4);
            for (size_t i = 0; i < iv.size(); ++i) v[i] = (float) iv[i];
        } else {
            fprintf(stderr, "unexpected output type\n"); g_fail++;
        }
        r.outs.push_back(std::move(v));
    }
    ggml_gallocr_free(ga);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { const double d = (double) a[i] - b[i]; num += d * d; den += (double) b[i] * b[i]; }
    return den > 0 ? num / den : num;
}
double maxabs(const std::vector<float> & a, const std::vector<float> & b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double) std::fabs((double) a[i] - b[i]));
    return m;
}
bool finite(const std::vector<float> & a) {
    for (float x : a) if (!std::isfinite(x)) return false;
    return true;
}
void report(const std::string & name, bool ok, const std::string & detail) {
    printf("%-58s %s  %s\n", name.c_str(), ok ? "PASS" : "FAIL", detail.c_str());
    if (!ok) g_fail++;
}

std::mt19937 rng(1234);
std::vector<float> randv(size_t n, float s = 1.0f) {
    std::normal_distribution<float> d(0.0f, s);
    std::vector<float> v(n);
    for (auto & x : v) x = d(rng);
    return v;
}

std::vector<uint8_t> quantize(ggml_type t, const std::vector<float> & w, int64_t n_per_row) {
    const int64_t nrows = (int64_t) w.size() / n_per_row;
    std::vector<uint8_t> q(ggml_row_size(t, n_per_row) * nrows);
    ggml_quantize_chunk(t, w.data(), q.data(), 0, nrows, n_per_row, nullptr);
    return q;
}

// ---------------------------------------------------------------- Q2_0
void test_q2_0_get_rows(const backend_pair & bp) {
    const int64_t K = 640, R = 37;
    auto w = randv(K * R);
    auto q = quantize(GGML_TYPE_Q2_0, w, K);
    std::vector<int32_t> ids = { 0, 5, 36, 17, 17, 2 };
    auto build = [&](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q2_0, K, R);
        ggml_tensor * i = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) ids.size());
        in = { a, i };
        return std::vector<ggml_tensor *>{ ggml_get_rows(ctx, a, i) };
    };
    auto fill = [&](std::vector<ggml_tensor *> & in) {
        ggml_backend_tensor_set(in[0], q.data(), 0, q.size());
        ggml_backend_tensor_set(in[1], ids.data(), 0, ids.size() * 4);
    };
    auto c = run(bp.cpu, build, fill), g = run(bp.cuda, build, fill);
    report("Q2_0 get_rows CUDA vs CPU (exact)", maxabs(c.outs[0], g.outs[0]) == 0.0, "maxabs=" + std::to_string(maxabs(c.outs[0], g.outs[0])));
}

void test_q2_0_mul_mat(const backend_pair & bp, int64_t K, int64_t N, int64_t M) {
    auto w = randv(K * N, 0.05f);
    auto q = quantize(GGML_TYPE_Q2_0, w, K);
    // F32 reference from the dequantized weights
    std::vector<float> wd(K * N);
    ggml_get_type_traits(GGML_TYPE_Q2_0)->to_float(q.data(), wd.data(), K * N);
    auto x = randv(K * M);
    auto build_q = [&](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q2_0, K, N);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        in = { a, b };
        return std::vector<ggml_tensor *>{ ggml_mul_mat(ctx, a, b) };
    };
    auto fill_q = [&](std::vector<ggml_tensor *> & in) {
        ggml_backend_tensor_set(in[0], q.data(), 0, q.size());
        ggml_backend_tensor_set(in[1], x.data(), 0, x.size() * 4);
    };
    auto build_f = [&](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        in = { a, b };
        return std::vector<ggml_tensor *>{ ggml_mul_mat(ctx, a, b) };
    };
    auto fill_f = [&](std::vector<ggml_tensor *> & in) {
        ggml_backend_tensor_set(in[0], wd.data(), 0, wd.size() * 4);
        ggml_backend_tensor_set(in[1], x.data(), 0, x.size() * 4);
    };
    auto ref = run(bp.cpu, build_f, fill_f);
    auto c = run(bp.cpu, build_q, fill_q), g = run(bp.cuda, build_q, fill_q);
    const double e_c = nmse(c.outs[0], ref.outs[0]), e_g = nmse(g.outs[0], ref.outs[0]);
    char d[160];
    snprintf(d, sizeof d, "nmse cpu=%.2e cuda=%.2e (vs f32 dequant ref)", e_c, e_g);
    report("Q2_0 mul_mat K=" + std::to_string(K) + " N=" + std::to_string(N) + " M=" + std::to_string(M),
           finite(g.outs[0]) && e_g < 2e-4 && e_g < 4 * e_c + 1e-6, d);
}

void test_q2_0_mul_mat_id(const backend_pair & bp, int64_t M) {
    const int64_t K = 2560, N = 640, E = 16, U = 4;
    auto w = randv(K * N * E, 0.05f);
    auto q = quantize(GGML_TYPE_Q2_0, w, K);
    std::vector<float> wd(K * N * E);
    ggml_get_type_traits(GGML_TYPE_Q2_0)->to_float(q.data(), wd.data(), K * N * E);
    auto x = randv(K * M);
    std::vector<int32_t> ids(U * M);
    std::uniform_int_distribution<int> de(0, (int) E - 1);
    for (int64_t t = 0; t < M; ++t) {
        std::vector<int> pick;
        while ((int64_t) pick.size() < U) { int e = de(rng); if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e); }
        for (int64_t u = 0; u < U; ++u) ids[t * U + u] = pick[u];
    }
    auto mk = [&](ggml_type t) {
        return [&, t](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            ggml_tensor * a = ggml_new_tensor_3d(ctx, t, K, N, E);
            ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, M);
            ggml_tensor * i = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, U, M);
            in = { a, b, i };
            return std::vector<ggml_tensor *>{ ggml_mul_mat_id(ctx, a, b, i) };
        };
    };
    auto fill_q = [&](std::vector<ggml_tensor *> & in) {
        ggml_backend_tensor_set(in[0], q.data(), 0, q.size());
        ggml_backend_tensor_set(in[1], x.data(), 0, x.size() * 4);
        ggml_backend_tensor_set(in[2], ids.data(), 0, ids.size() * 4);
    };
    auto fill_f = [&](std::vector<ggml_tensor *> & in) {
        ggml_backend_tensor_set(in[0], wd.data(), 0, wd.size() * 4);
        ggml_backend_tensor_set(in[1], x.data(), 0, x.size() * 4);
        ggml_backend_tensor_set(in[2], ids.data(), 0, ids.size() * 4);
    };
    auto ref = run(bp.cpu, mk(GGML_TYPE_F32), fill_f);
    auto g = run(bp.cuda, mk(GGML_TYPE_Q2_0), fill_q);
    auto c = run(bp.cpu, mk(GGML_TYPE_Q2_0), fill_q);
    const double e_c = nmse(c.outs[0], ref.outs[0]), e_g = nmse(g.outs[0], ref.outs[0]);
    char d[160];
    snprintf(d, sizeof d, "nmse cpu=%.2e cuda=%.2e", e_c, e_g);
    report("Q2_0 mul_mat_id K=2560 N=640 E=16 top4 M=" + std::to_string(M), finite(g.outs[0]) && e_g < 2e-4 && e_g < 4 * e_c + 1e-6, d);
}

// Required capture keeps four token views in one graph, with the supported indexed-GEMM layout.
void test_q2_capture_mmid(const backend_pair & bp, int rows) {
    const int E = 128, N = 128, NE = 16, K = 10, T = 4;
    std::mt19937 local_rng(731 + rows);
    std::normal_distribution<float> normal(0.0f, 0.05f);
    std::vector<float> weights(E * N * NE), a(E * rows * T), b(a.size());
    for (auto & v : weights) v = normal(local_rng);
    for (size_t i = 0; i < a.size(); ++i) { a[i] = normal(local_rng); b[i] = a[i] * -0.75f + 0.01f; }
    const auto quantized = quantize(GGML_TYPE_Q2_0, weights, E);
    std::vector<int32_t> selected(K * T);
    for (int t = 0; t < T; ++t) for (int k = 0; k < K; ++k)
        selected[t * K + k] = k == 0 ? 0 : k == 1 ? 15 : (t % 2 ? K - k : k - 1);
    auto build = [&](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        auto * w = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_0, E, N, NE);
        auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, E, rows, T);
        auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, K, T);
        in = { w, x, ids };
        ggml_tensor * output = nullptr;
        for (int t = 0; t < T; ++t) {
            auto * xt = ggml_cont(ctx, ggml_view_3d(ctx, x, E, rows, 1, x->nb[1], x->nb[2], t * x->nb[2]));
            auto * it = ggml_cont(ctx, ggml_view_2d(ctx, ids, K, 1, ids->nb[1], t * ids->nb[1]));
            auto * yt = ggml_mul_mat_id(ctx, w, xt, it);
            output = output ? ggml_concat(ctx, output, yt, 2) : yt;
        }
        return std::vector<ggml_tensor *>{output};
    };
    auto set = [&](std::vector<ggml_tensor *> & in, const std::vector<float> & x) {
        ggml_backend_tensor_set(in[0], quantized.data(), 0, quantized.size());
        ggml_backend_tensor_set(in[1], x.data(), 0, x.size() * sizeof(float));
        ggml_backend_tensor_set(in[2], selected.data(), 0, selected.size() * sizeof(int32_t));
    };
    auto eager = run(bp.cuda, build, [&](std::vector<ggml_tensor *> & in) { set(in, b); });
    int fills = 0;
    auto captured = run(bp.cuda, build, [&](std::vector<ggml_tensor *> & in) { set(in, ++fills >= 3 ? b : a); }, true);
    report("Q2_0 MMID n4 static packing rows=" + std::to_string(rows),
           finite(captured.outs[0]) && captured.outs[0] == eager.outs[0] && fills == 3,
           "one n4 graph, four supported ne2=1 MMIDs; changed-input required capture/replay bit-identical to eager");
}

// Adjacent gate/up rows with the full [gate | up | down] expert stride, as in banks, cache and MTP.
void test_q2_0_gate_up(const backend_pair & bp, int64_t M) {
    const int64_t K = 2560, N = 640, E = 16, U = 4;
    auto q = quantize(GGML_TYPE_Q2_0, randv(K * 3 * N * E, 0.05f), K);
    auto x = randv(K * M);
    std::vector<int32_t> ids(U * M);
    // Include the first/last experts, repeated selections across tokens, and uneven expert populations.
    for (int64_t t = 0; t < M; ++t) {
        ids[t * U + 0] = 0;
        ids[t * U + 1] = E - 1;
        ids[t * U + 2] = 1 + t % (E - 2);
        ids[t * U + 3] = 1 + (t + 5) % (E - 2);
    }
    auto build = [&](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_0, K, 3 * N, E);
        ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, M);
        ggml_tensor * i = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, U, M);
        in = { a, b, i };
        auto view = [&](int64_t rows, size_t offset) {
            return ggml_view_3d(ctx, a, K, rows, E, a->nb[1], a->nb[2], offset);
        };
        ggml_tensor * gate = ggml_mul_mat_id(ctx, view(N, 0), b, i);
        ggml_tensor * up = ggml_mul_mat_id(ctx, view(N, N * a->nb[1]), b, i);
        ggml_tensor * both = ggml_mul_mat_id(ctx, view(2 * N, 0), b, i);
        auto half = [&](size_t offset) {
            return ggml_cont(ctx, ggml_view_3d(ctx, both, N, U, M, both->nb[1], both->nb[2], offset));
        };
        std::vector<ggml_tensor *> outs = { gate, up, half(0), half(N * sizeof(float)),
                                          ggml_swiglu_split(ctx, gate, up), ggml_swiglu(ctx, both) };
        // Keep every comparison output alive even when also consumed by the activation.
        for (auto * o : outs) ggml_set_output(o);
        return outs;
    };
    auto fill = [&](std::vector<ggml_tensor *> & in) {
        ggml_backend_tensor_set(in[0], q.data(), 0, q.size());
        ggml_backend_tensor_set(in[1], x.data(), 0, x.size() * sizeof(float));
        ggml_backend_tensor_set(in[2], ids.data(), 0, ids.size() * sizeof(int32_t));
    };
    auto g = run(bp.cuda, build, fill);
    const char * grouped = std::getenv("QFN_MMQID_Q2_0");
    for (int p = 0; p < 3; ++p) {
        const auto & a = g.outs[p < 2 ? p : 4];
        const auto & b = g.outs[p < 2 ? p + 2 : 5];
        const bool identical = a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
        size_t different = 0;
        for (size_t j = 0; j < a.size(); ++j)
            different += std::memcmp(&a[j], &b[j], sizeof(float)) != 0;
        char detail[200];
        snprintf(detail, sizeof detail, "byte-identical=%d different=%zu/%zu maxabs=%.9g grouped=%s",
                 (int) identical, different, a.size(), maxabs(a, b), grouped ? grouped : "0");
        report("Q2_0 fused gate/up M=" + std::to_string(M) + " " + (p == 0 ? "gate" : p == 1 ? "up" : "swiglu"),
               identical && finite(a) && finite(b), detail);
    }
}

// The same input prefix and weights at two MMQ batch sizes. Fusion changes N;
// this separately checks M, including uneven expert populations in routed MMQ.
void test_q2_0_batch_shape(const backend_pair & bp, bool routed) {
    const int64_t K = 2560, N = 640, E = routed ? 16 : 1, U = routed ? 4 : 1;
    const int64_t small = 40, large = 512;
    auto q = quantize(GGML_TYPE_Q2_0, randv(K * N * E, 0.05f), K);
    auto x = randv(K * large);
    std::vector<int32_t> ids(U * large);
    for (int64_t t = 0; t < large; ++t) for (int64_t u = 0; u < U; ++u)
        ids[t * U + u] = (int32_t) ((t + u * 5) % E);
    auto build = [&](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_0, K, N, E);
        ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, large);
        ggml_tensor * i = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, U, large);
        in = { a, b, i };
        ggml_tensor * short_b = ggml_view_3d(ctx, b, K, 1, small, b->nb[1], b->nb[2], 0);
        ggml_tensor * short_i = ggml_view_2d(ctx, i, U, small, i->nb[1], 0);
        ggml_tensor * s = routed ? ggml_mul_mat_id(ctx, a, short_b, short_i) :
            ggml_mul_mat(ctx, a, ggml_reshape_2d(ctx, short_b, K, small));
        ggml_tensor * l = routed ? ggml_mul_mat_id(ctx, a, b, i) :
            ggml_mul_mat(ctx, a, ggml_reshape_2d(ctx, b, K, large));
        ggml_set_output(s); ggml_set_output(l);
        return std::vector<ggml_tensor *>{s, l};
    };
    auto fill = [&](std::vector<ggml_tensor *> & in) {
        ggml_backend_tensor_set(in[0], q.data(), 0, q.size());
        ggml_backend_tensor_set(in[1], x.data(), 0, x.size() * sizeof(float));
        ggml_backend_tensor_set(in[2], ids.data(), 0, ids.size() * sizeof(int32_t));
    };
    auto g = run(bp.cuda, build, fill);
    const auto & s = g.outs[0];
    std::vector<float> prefix(g.outs[1].begin(), g.outs[1].begin() + s.size());
    const bool identical = std::memcmp(s.data(), prefix.data(), s.size() * sizeof(float)) == 0;
    size_t different = 0;
    for (size_t j = 0; j < s.size(); ++j) different += std::memcmp(&s[j], &prefix[j], sizeof(float)) != 0;
    char detail[200];
    snprintf(detail, sizeof detail, "byte-identical=%d different=%zu/%zu maxabs=%.9g", (int) identical,
             different, s.size(), maxabs(s, prefix));
    report(std::string("Q2_0 ") + (routed ? "mul_mat_id" : "mul_mat") + " M=40 vs M=512 prefix", identical && finite(s) && finite(prefix), detail);
}

// ---------------------------------------------------------------- gated delta net
void test_gdn(const backend_pair & bp, int64_t T, int64_t K_snap) {
    const int64_t S = 128, Hk = 16, Hv = 48;
    auto qv = randv(S * Hk * T), kv = randv(S * Hk * T), vv = randv(S * Hv * T);
    // GDN inputs are L2-normalised q/k in the model; normalise here so the state stays bounded
    auto l2 = [&](std::vector<float> & a) {
        for (size_t r = 0; r < a.size() / S; ++r) {
            double s = 0; for (int64_t i = 0; i < S; ++i) s += a[r * S + i] * a[r * S + i];
            const float inv = (float) (1.0 / std::sqrt(s + 1e-6));
            for (int64_t i = 0; i < S; ++i) a[r * S + i] *= inv;
        }
    };
    l2(qv); l2(kv);
    std::vector<float> g(Hv * T), b(Hv * T);
    std::uniform_real_distribution<float> ug(-1.5f, -0.01f), ub(0.05f, 0.95f);
    for (auto & x : g) x = ug(rng);
    for (auto & x : b) x = ub(rng);
    auto s0 = randv(S * S * Hv, 0.1f);
    auto build = [&](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, T, 1);
        ggml_tensor * k  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, T, 1);
        ggml_tensor * v  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hv, T, 1);
        ggml_tensor * gt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, Hv, T, 1);
        ggml_tensor * bt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, Hv, T, 1);
        ggml_tensor * st = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, Hv, 1);
        in = { q, k, v, gt, bt, st };
        return std::vector<ggml_tensor *>{ ggml_gated_delta_net(ctx, q, k, v, gt, bt, st, K_snap) };
    };
    auto fill = [&](std::vector<ggml_tensor *> & in) {
        ggml_backend_tensor_set(in[0], qv.data(), 0, qv.size() * 4);
        ggml_backend_tensor_set(in[1], kv.data(), 0, kv.size() * 4);
        ggml_backend_tensor_set(in[2], vv.data(), 0, vv.size() * 4);
        ggml_backend_tensor_set(in[3], g.data(), 0, g.size() * 4);
        ggml_backend_tensor_set(in[4], b.data(), 0, b.size() * 4);
        ggml_backend_tensor_set(in[5], s0.data(), 0, s0.size() * 4);
    };
    auto c = run(bp.cpu, build, fill), gg = run(bp.cuda, build, fill);
    const size_t n_out = (size_t) S * Hv * T;
    std::vector<float> co(c.outs[0].begin(), c.outs[0].begin() + n_out), go(gg.outs[0].begin(), gg.outs[0].begin() + n_out);
    std::vector<float> cs(c.outs[0].begin() + n_out, c.outs[0].begin() + n_out + S * S * Hv);
    std::vector<float> gs(gg.outs[0].begin() + n_out, gg.outs[0].begin() + n_out + S * S * Hv);
    char d[200];
    snprintf(d, sizeof d, "out nmse=%.2e maxabs=%.2e | state nmse=%.2e maxabs=%.2e", nmse(go, co), maxabs(go, co), nmse(gs, cs), maxabs(gs, cs));
    report("GDN S=128 Hk=16 Hv=48 T=" + std::to_string(T) + " K=" + std::to_string(K_snap) + (T >= 64 ? " (chunked MMA)" : " (recurrent)"),
           finite(go) && finite(gs) && nmse(go, co) < 1e-8 && nmse(gs, cs) < 1e-8, d);
}

// ---------------------------------------------------------------- QSA select
void test_qsa_select(const backend_pair & bp) {
    const int n_top = 512, ratio = 4, P = 3000;   // pools in the score rows
    const int T = 200;
    std::vector<int32_t> pos(T);
    std::uniform_int_distribution<int> dp(0, P * ratio - 1);
    for (int t = 0; t < T; ++t) pos[t] = t < 20 ? t * 97 : dp(rng);  // small (all visible) and large positions
    std::vector<float> sc((size_t) P * T);
    std::uniform_int_distribution<int> lvl(-3, 40);
    for (auto & x : sc) { const int l = lvl(rng); x = l < 0 ? (l == -3 ? -0.0f : -1.0f * l) : 0.125f * (float) l; }  // ties, zeros, -0, negatives
    auto build = [&](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        ggml_tensor * s = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, P, T);
        ggml_tensor * p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
        in = { s, p };
        return std::vector<ggml_tensor *>{ ggml_qfn_qsa_select(ctx, s, p, n_top, ratio) };
    };
    auto fill = [&](std::vector<ggml_tensor *> & in) {
        ggml_backend_tensor_set(in[0], sc.data(), 0, sc.size() * 4);
        ggml_backend_tensor_set(in[1], pos.data(), 0, pos.size() * 4);
    };
    auto c = run(bp.cpu, build, fill), g = run(bp.cuda, build, fill);
    // independent check of one row's semantics on the CPU output: cells < pos, ascending pools, count
    bool sane = true;
    const int n_sel = n_top * ratio + ratio - 1;
    for (int t = 0; t < T && sane; ++t) {
        const int p = pos[t], nv = (p + 1) / ratio;
        const int expect = (nv <= n_top ? nv * ratio : n_top * ratio) + (p + 1 - nv * ratio);
        int cnt = 0, prev = -1;
        for (int i = 0; i < n_sel; ++i) {
            const int v = (int) c.outs[0][(size_t) t * n_sel + i];
            if (v < 0) continue;
            cnt++;
            if (v > p || v <= prev) sane = false;
            prev = v;
        }
        if (cnt != expect) sane = false;
    }
    report("QSA select CUDA vs CPU (exact, ties/zeros/-0/negatives)", maxabs(c.outs[0], g.outs[0]) == 0.0 && sane,
           std::string("maxabs=") + std::to_string(maxabs(c.outs[0], g.outs[0])) + (sane ? " semantics ok" : " SEMANTICS BROKEN"));
}

void test_iq4_nl_get_rows(const backend_pair & bp) {
    const int64_t K = 160, R = 53;  // the PLE n-gram row width
    auto w = randv(K * R);
    auto q = quantize(GGML_TYPE_IQ4_NL, w, K);
    std::vector<int32_t> ids = { 0, 52, 7, 7, 31, 1, 2, 3 };
    auto build = [&](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_NL, K, R);
        ggml_tensor * i = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) ids.size());
        in = { a, i };
        return std::vector<ggml_tensor *>{ ggml_get_rows(ctx, a, i) };
    };
    auto fill = [&](std::vector<ggml_tensor *> & in) {
        ggml_backend_tensor_set(in[0], q.data(), 0, q.size());
        ggml_backend_tensor_set(in[1], ids.data(), 0, ids.size() * 4);
    };
    auto c = run(bp.cpu, build, fill), g = run(bp.cuda, build, fill);
    report("IQ4_NL get_rows K=160 CUDA vs CPU (exact)", maxabs(c.outs[0], g.outs[0]) == 0.0, "maxabs=" + std::to_string(maxabs(c.outs[0], g.outs[0])));
}

// ---------------------------------------------------------------- QSA sparse attention
void test_qsa_attn(const backend_pair & bp, int n_ctx, int T, int p0) {
    const int D = 256, H = 24, Hkv = 2, n_top = 512, ratio = 4;
    const int n_pool = n_ctx / ratio;
    auto qv = randv((size_t) D * H * T), gv = randv((size_t) D * H * T);
    auto kf = randv((size_t) D * Hkv * n_ctx), vf = randv((size_t) D * Hkv * n_ctx);
    std::vector<ggml_fp16_t> kh(kf.size()), vh(vf.size());
    for (size_t i = 0; i < kf.size(); ++i) { kh[i] = ggml_fp32_to_fp16(kf[i]); vh[i] = ggml_fp32_to_fp16(vf[i]); }
    std::vector<int32_t> pos(T);
    for (int t = 0; t < T; ++t) pos[t] = p0 + t;
    std::vector<float> sc((size_t) n_pool * T);
    std::uniform_real_distribution<float> u(0.0f, 3.0f);
    for (auto & x : sc) x = u(rng);
    auto build = [&](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, T);
        ggml_tensor * g = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, T);
        ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, Hkv, n_ctx);
        ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, Hkv, n_ctx);
        ggml_tensor * s = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_pool, T);
        ggml_tensor * p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
        in = { q, g, k, v, s, p };
        ggml_tensor * sel = ggml_qfn_qsa_select(ctx, s, p, n_top, ratio);
        return std::vector<ggml_tensor *>{ ggml_qfn_qsa_attn(ctx, q, g, k, v, sel, p, 1.0f / 16.0f) };
    };
    auto fill = [&](std::vector<ggml_tensor *> & in) {
        ggml_backend_tensor_set(in[0], qv.data(), 0, qv.size() * 4);
        ggml_backend_tensor_set(in[1], gv.data(), 0, gv.size() * 4);
        ggml_backend_tensor_set(in[2], kh.data(), 0, kh.size() * 2);
        ggml_backend_tensor_set(in[3], vh.data(), 0, vh.size() * 2);
        ggml_backend_tensor_set(in[4], sc.data(), 0, sc.size() * 4);
        ggml_backend_tensor_set(in[5], pos.data(), 0, pos.size() * 4);
    };
    auto c = run(bp.cpu, build, fill), g = run(bp.cuda, build, fill);
    char d[160];
    snprintf(d, sizeof d, "nmse=%.2e maxabs=%.2e", nmse(g.outs[0], c.outs[0]), maxabs(g.outs[0], c.outs[0]));
    report("QSA attn n_ctx=" + std::to_string(n_ctx) + " T=" + std::to_string(T) + " pos0=" + std::to_string(p0),
           finite(g.outs[0]) && nmse(g.outs[0], c.outs[0]) < 1e-9, d);
}


// Fused weighted expert reduction versus the exact separate CUDA MUL/ADD
// graph. memcmp is intentional: float equality would miss signed-zero changes.
static void test_expert_reduce(const backend_pair & bp, int64_t T, bool strided, bool captured = false, bool permuted = false) {
    const int64_t E = 2560, K = 10, XE = E + (strided ? 3 : 0), WK = K + (strided ? 2 : 0);
    GGML_ASSERT(!captured || T > 1); // the replay marker must not overwrite first-token edge cases
    auto xs = randv(size_t(XE*K*T), 0.7f);
    std::vector<float> ws(size_t(WK*T));
    for (int64_t t = 0; t < T; ++t) for (int64_t k = 0; k < WK; ++k)
        ws[size_t(t*WK+k)] = float(1 + (t*7+k*3)%23)/29.0f;
    // All-negative zero detects an extra +0 initialization. The next rows
    // expose FMA contraction and reassociation/cancellation respectively.
    for (int64_t k = 0; k < K; ++k) xs[size_t(k*XE)] = -0.0f;
    for (int64_t k = 0; k < K; ++k) { xs[size_t(k*XE+1)] = 0; xs[size_t(k*XE+2)] = 0; }
    ws[0] = 1.0f; ws[1] = 1.0f - 0x1p-23f; ws[2] = 1.0f;
    xs[1] = -1.0f; xs[size_t(XE+1)] = 1.0f + 0x1p-23f;
    xs[2] = 16777216.0f; xs[size_t(XE+2)] = 1.0f/(1.0f-0x1p-23f); xs[size_t(2*XE+2)] = -16777216.0f;
    std::vector<float> permuted_storage(permuted ? size_t((K+1)*(E+1)*T) : 0, 42.0f);
    auto build = [&](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        auto * xb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, permuted ? K+1 : XE, permuted ? E+1 : K, T);
        auto * wb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, WK, T);
        in = {xb, wb};
        auto * x = permuted ? ggml_permute(ctx,
            ggml_view_3d(ctx, xb, K, E, T, xb->nb[1], xb->nb[2], size_t(K+2)*sizeof(float)), 1,0,2,3) :
            ggml_view_3d(ctx, xb, E, K, T, xb->nb[1], xb->nb[2], 0);
        if (permuted) GGML_ASSERT(x->nb[0] == size_t(K+1)*sizeof(float) && x->view_offs != 0);
        auto * w = ggml_view_3d(ctx, wb, 1, K, T, wb->nb[1], wb->nb[2], 0);
        // Materialize only the reference: the custom consumer sees the actual
        // nonunit feature stride and nonzero view offset.
        auto * products = ggml_mul(ctx, permuted ? ggml_cont(ctx, x) : x, w);
        auto * ref = ggml_view_2d(ctx, products, E, T, products->nb[2], 0);
        for (int64_t k = 1; k < K; ++k)
            ref = ggml_add(ctx, ref, ggml_view_2d(ctx, products, E, T, products->nb[2], k*products->nb[1]));
        auto * fused = ggml_qfn_expert_reduce(ctx, x, w);
        // A mismatched weights shape must not be admitted as a broadcast.
        const int64_t saved = w->ne[1]; w->ne[1] = K-1;
        const bool rejects = !ggml_qfn_expert_reduce_inputs_valid(x, w); w->ne[1] = saved;
        report("expert reduction shape validation", rejects, "mismatched top-k rejected");
        return std::vector<ggml_tensor *>{ref, fused};
    };
    int fills = 0;
    auto fill = [&](std::vector<ggml_tensor *> & in) {
        ++fills;
        if (captured) for (int64_t k = 0; k < K; ++k) {
            xs[size_t(((T-1)*K+k)*XE+E-1)] = fills >= 3 ? 2.0f : 1.0f;
            ws[size_t((T-1)*WK+k)] = 1.0f;
        }
        if (permuted) {
            for (int64_t t = 0; t < T; ++t) for (int64_t k = 0; k < K; ++k) for (int64_t d = 0; d < E; ++d)
                permuted_storage[size_t(K+2+(t*(E+1)+d)*(K+1)+k)] = xs[size_t((t*K+k)*XE+d)];
            ggml_backend_tensor_set(in[0], permuted_storage.data(), 0, permuted_storage.size()*sizeof(float));
        } else ggml_backend_tensor_set(in[0], xs.data(), 0, xs.size()*sizeof(float));
        ggml_backend_tensor_set(in[1], ws.data(), 0, ws.size()*sizeof(float));
    };
    const auto g = run(bp.cuda, build, fill, captured);
    const bool bits = g.outs.size() == 2 && g.outs[0].size() == g.outs[1].size() &&
        std::memcmp(g.outs[0].data(), g.outs[1].data(), g.outs[0].size()*sizeof(float)) == 0;
    const bool sensitive = std::signbit(g.outs[0][0]) && g.outs[0][1] == 0.0f && g.outs[0][2] == 0.0f;
    const bool replay_changed = !captured || (fills >= 3 && g.outs[0].back() == 20.0f && g.outs[1].back() == 20.0f);
    report("expert reduction T=" + std::to_string(T) + (permuted ? " permuted+offset" : strided ? " strided" : " contiguous") +
           (captured ? " changed-input replay" : " eager"), bits && sensitive && replay_changed,
           "all output bytes vs separate MUL/ascending ADD; signed zero, FMA, cancellation, nonuniform weights");
}

// fused hyper-connection mix/combine: bit-identical to the unfused ggml graph on CUDA, close to the CPU
void test_hc_fused(const backend_pair & bp, int64_t T) {
    const int64_t E = 2560, HC = 4;
    std::vector<float> xs((size_t) (E * HC * T)), gs(xs.size()), rs(xs.size()), bs((size_t) (E * T)), is((size_t) (HC * T));
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto * v : { &xs, &gs, &rs, &bs, &is }) for (auto & x : *v) x = nd(rng);
    auto fill = [&](std::vector<ggml_tensor *> & in) {
        ggml_backend_tensor_set(in[0], xs.data(), 0, xs.size() * 4);
        ggml_backend_tensor_set(in[1], gs.data(), 0, gs.size() * 4);
        ggml_backend_tensor_set(in[2], rs.data(), 0, rs.size() * 4);
        ggml_backend_tensor_set(in[3], bs.data(), 0, bs.size() * 4);
        ggml_backend_tensor_set(in[4], is.data(), 0, is.size() * 4);
    };
    auto build = [&](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, E, HC, T);
        ggml_tensor * g = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E * HC, T);
        ggml_tensor * r = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, E, HC, T);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, T);
        ggml_tensor * i = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HC, T);
        in = { x, g, r, b, i };
        // unfused (the engine's graph)
        ggml_tensor * gated = ggml_reshape_3d(ctx, ggml_mul(ctx, ggml_reshape_2d(ctx, x, E * HC, T), ggml_sigmoid(ctx, g)), E, HC, T);
        ggml_tensor * mixed = ggml_cont(ctx, ggml_view_2d(ctx, gated, E, T, ggml_row_size(gated->type, E) * HC, 0));
        for (int64_t c = 1; c < HC; ++c)
            mixed = ggml_add(ctx, mixed, ggml_view_2d(ctx, gated, E, T, ggml_row_size(gated->type, E) * HC, ggml_row_size(gated->type, E) * c));
        mixed = ggml_scale(ctx, mixed, 1.0f / (float) HC);
        ggml_tensor * w = ggml_reshape_3d(ctx, ggml_scale(ctx, ggml_sigmoid(ctx, i), 2.0f), 1, HC, T);
        ggml_tensor * comb = ggml_add(ctx, r, ggml_mul(ctx, ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, b, E, 1, T), E, HC, T, 1), w));
        return std::vector<ggml_tensor *>{ mixed, comb, ggml_qfn_hc_mix(ctx, x, g, 1.0f / (float) HC), ggml_qfn_hc_combine(ctx, r, b, i) };
    };
    auto g = run(bp.cuda, build, fill), c = run(bp.cpu, build, fill);
    const bool mix_bits = g.outs[0] == g.outs[2], comb_bits = g.outs[1] == g.outs[3];
    char d[200];
    snprintf(d, sizeof d, "CUDA fused==unfused mix %d comb %d; CPU-vs-CUDA nmse mix %.2e comb %.2e", (int) mix_bits, (int) comb_bits,
             nmse(c.outs[2], g.outs[2]), nmse(c.outs[3], g.outs[3]));
    report("fused hc mix/combine T=" + std::to_string(T), mix_bits && comb_bits && nmse(c.outs[2], g.outs[2]) < 1e-10, d);
}
}  // namespace

static void test_slot_map_record_only(const backend_pair & bp) {
    ggml_qfn_bell * bell = nullptr;
    void * device = nullptr;
    if (cudaHostAlloc((void **) &bell, sizeof(*bell), cudaHostAllocMapped) != cudaSuccess) {
        report("QFNV3OPT slot map record-only", false, "mapped allocation failed");
        return;
    }
    if (cudaHostGetDevicePointer(&device, bell, 0) != cudaSuccess) {
        report("QFNV3OPT slot map record-only", false, "mapped pointer failed");
        cudaFreeHost(bell);
        return;
    }
    std::memset(bell, 0, sizeof(*bell));
    bell->timeout_ms = -1;
    bell->req = bell->ack = -1;
    int32_t * latch = nullptr;
    if (cudaMalloc((void **) &latch, sizeof(int32_t)) != cudaSuccess ||
        cudaMemset(latch, 0, sizeof(int32_t)) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        report("QFNV3OPT slot map record-only", false, "device latch allocation/reset failed");
        if (latch) cudaFree(latch);
        cudaFreeHost(bell);
        return;
    }
    const int32_t ids[] = { 1, 2, 2, 3 }, table[] = { 7, 5, -1, 9 }, seq = 7;
    auto check = [&](int layer) {
        auto build = [&](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            auto * s = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 1);
            auto * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
            auto * q = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            in = { s, t, q };
            auto * mapped = ggml_qfn_slot_map(ctx, s, t, q, device, layer);
            ggml_qfn_slot_map_set_record_latch(mapped, latch);
            return std::vector<ggml_tensor *>{ mapped };
        };
        auto fill = [&](std::vector<ggml_tensor *> & in) {
            ggml_backend_tensor_set(in[0], ids, 0, sizeof(ids));
            ggml_backend_tensor_set(in[1], table, 0, sizeof(table));
            ggml_backend_tensor_set(in[2], &seq, 0, sizeof(seq));
        };
        const auto r = run(bp.cuda, build, fill);
        return r.outs[0] == std::vector<float>({ 5, 0, 0, 9 });
    };
    const bool first = check(3);
    const bool second = check(9);
    const bool kept = bell->error == 7 * 64 + 3 && bell->layer == 3 && bell->n == 4 &&
                      std::memcmp(bell->sel, ids, sizeof(ids)) == 0 && bell->req == -1 && bell->ack == -1;
    report("QFNV3OPT slot map record-only", first && second && kept,
           "device atomic latch + mapped record; first layer/selection/error kept; missing -> slot 0; no wait");
    cudaFree(latch);
    cudaFreeHost(bell);
}

// The real bounded-wait kernel must retain its FIRST failed record across later graph nodes/replays.
// A 1ms fixture timeout bounds the test; production's 500ms policy is unchanged.
static void test_slot_map_timeout_latch(const backend_pair & bp) {
    ggml_qfn_bell * bell = nullptr;
    void * device = nullptr;
    if (cudaHostAlloc((void **) &bell, sizeof(*bell), cudaHostAllocMapped) != cudaSuccess) {
        report("QFN doorbell first-timeout latch", false, "mapped allocation failed"); return;
    }
    if (cudaHostGetDevicePointer(&device, bell, 0) != cudaSuccess) {
        report("QFN doorbell first-timeout latch", false, "mapped pointer failed"); cudaFreeHost(bell); return;
    }
    std::memset(bell, 0, sizeof(*bell));
    bell->req = bell->ack = -1; bell->timeout_ms = 1;
    const int32_t first_ids[] = {1,2,2,3};
    const int32_t table[] = {7,5,-1,9}, sequence = 7;
    auto check = [&](int layer, const int32_t * selected, bool captured) {
        auto build = [&](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
            auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 1);
            auto * slots = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
            auto * seq = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            in = {ids, slots, seq};
            return std::vector<ggml_tensor *>{ggml_qfn_slot_map(ctx, ids, slots, seq, device, layer)};
        };
        auto fill = [&](std::vector<ggml_tensor *> & in) {
            ggml_backend_tensor_set(in[0], selected, 0, sizeof(first_ids));
            ggml_backend_tensor_set(in[1], table, 0, sizeof(table));
            ggml_backend_tensor_set(in[2], &sequence, 0, sizeof(sequence));
        };
        return run(bp.cuda, build, fill, captured);
    };
    const auto first = check(3, first_ids, true);
    // A distinct later request also misses; it must not overwrite the timed-out host-owned record.
    const int32_t later_missing[] = {2,3,1,0};
    const auto later = check(9, later_missing, false);
    const bool retained = bell->error == 7*64+3 && bell->req == 7*64+3 && bell->ack == -1 &&
        bell->layer == 3 && bell->n == 4 && std::memcmp(bell->sel, first_ids, sizeof(first_ids)) == 0;
    report("QFN doorbell first-timeout latch", retained && first.outs[0] == std::vector<float>({5,0,0,9}) &&
        later.outs[0] == std::vector<float>({0,9,5,7}), "required static/replay; first error/request/layer/selection immutable; safe failed-step IDs");
    cudaFreeHost(bell);
}

// Exercise the actual GGML CUSTOM dispatch, external router weighting and stable static graph.
// Direct pinned-DLL output is a wiring reference; independent kernel arithmetic is gated separately.
static void test_nvfp4_custom(const backend_pair & bp, int T, int mode,
                              bool expected_zero = false, float input_amplitude = 0.0001f) {
    const bool decode_calibrated=mode==QFN_NVFP4_CALIBRATED_DECODE;
    const uint32_t physical_slots=decode_calibrated ? 513 : 512;
    const std::string name = "NVFP4 CUSTOM T=" + std::to_string(T) + " mode=" + std::to_string(mode) +
        (expected_zero ? " underflow-zero" : " changing-output");
    char admission[512] = {};
    if (!ggml_backend_cuda_qfn_nvfp4_available(admission, sizeof admission) ||
        (mode == QFN_NVFP4_CALIBRATED_A4 && !ggml_backend_cuda_qfn_nvfp4_calibrated_available(admission, sizeof admission)) ||
        (decode_calibrated && !ggml_backend_cuda_qfn_nvfp4_calibrated_decode_available(admission,sizeof admission)) ||
        ggml_backend_cuda_qfn_nvfp4_prepare(nullptr) != cudaSuccess) {
        report(name, false, std::string("DLL/prepare admission failed: ") + admission); return;
    }
    wchar_t path[32768];
    const DWORD n = GetModuleFileNameW(nullptr, path, (DWORD) (sizeof(path) / sizeof(path[0])));
    if (!n || n >= sizeof(path) / sizeof(path[0])) { report(name, false, "executable path unavailable"); return; }
    wchar_t * slash = wcsrchr(path, L'\\');
    if (!slash) { report(name, false, "executable directory unavailable"); return; }
    wcscpy_s(slash + 1, sizeof(path) / sizeof(path[0]) - (slash + 1 - path), L"qfn-basalt-nvfp4.dll");
    HMODULE dll = LoadLibraryExW(path, nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    auto direct = dll ? reinterpret_cast<decltype(&qfn_basalt_nvfp4_run)>(GetProcAddress(dll, "qfn_basalt_nvfp4_run")) : nullptr;
    auto calibrated = dll ? reinterpret_cast<decltype(&qfn_basalt_nvfp4_run_calibrated)>(GetProcAddress(dll, "qfn_basalt_nvfp4_run_calibrated")) : nullptr;
    auto decoded=dll ? reinterpret_cast<decltype(&qfn_basalt_nvfp4_run_calibrated_decode)>(GetProcAddress(dll,"qfn_basalt_nvfp4_run_calibrated_decode")) : nullptr;
    const uint64_t bytes = ggml_backend_cuda_qfn_nvfp4_workspace_bytes(T, 10, physical_slots, mode);
    if (!direct || !bytes || (mode == QFN_NVFP4_CALIBRATED_A4 && !calibrated) || (decode_calibrated && !decoded)) {
        report(name, false, "direct reference export/workspace unavailable"); if (dll) FreeLibrary(dll); return;
    }
    ggml_init_params ip = { 96 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip), * scratch_ctx = ggml_init({ 2 * ggml_tensor_overhead(), nullptr, true });
    auto * scratch = ggml_new_tensor_1d(scratch_ctx, GGML_TYPE_I8, bytes);
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, T);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 10, T);
    auto * raw = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, 2764816, physical_slots);
    auto * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 10, T);
    auto * input_scale = mode == QFN_NVFP4_CALIBRATED_A4 || decode_calibrated ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 512) : nullptr;
    auto * logical_ids=decode_calibrated ? ggml_new_tensor_2d(ctx,GGML_TYPE_I32,10,T) : nullptr;
    auto * reference = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2560, 10, T);
    auto * out = ggml_qfn_nvfp4_experts(ctx, x, ids, raw, scratch, mode, input_scale, logical_ids);
    auto * weighted = ggml_mul(ctx, out, weights);
    auto * sum = ggml_cont(ctx, ggml_view_2d(ctx, weighted, 2560, T, 2560 * 10 * sizeof(float), 0));
    for (int k = 1; k < 10; ++k)
        sum = ggml_add(ctx, sum, ggml_view_2d(ctx, weighted, 2560, T, 2560 * 10 * sizeof(float), 2560 * k * sizeof(float)));
    auto * gf = ggml_new_graph(ctx); ggml_build_forward_expand(gf, sum);
    auto scratch_buf = ggml_backend_alloc_ctx_tensors(scratch_ctx, bp.cuda);
    auto buf = ggml_backend_alloc_ctx_tensors(ctx, bp.cuda);
    if (!buf || !scratch_buf || uintptr_t(scratch->data) % 256) {
        report(name, false, "device allocation/workspace alignment failed");
        if (buf) ggml_backend_buffer_free(buf); if (scratch_buf) ggml_backend_buffer_free(scratch_buf);
        ggml_free(ctx); ggml_free(scratch_ctx); FreeLibrary(dll); return;
    }
    std::vector<uint8_t> blob(2764816);
    for (uint32_t slot = 0; slot < physical_slots; ++slot) {
        for (size_t block = 0; block < 2764800; block += 36) {
            std::memset(blob.data() + block, 0x20, 4);
            std::memset(blob.data() + block + 4, (slot % 2) ? 0x19 : 0x21, 32);
        }
        const float scales[] = { 0.5f + slot * 0.0001f, 0.75f, 0.625f, 0.0f };
        std::memcpy(blob.data() + 2764800, scales, sizeof scales);
        ggml_backend_tensor_set(raw, blob.data(), size_t(slot) * blob.size(), blob.size());
    }
    std::vector<float> xv(size_t(2560) * T), wv(size_t(10) * T);
    std::vector<float> globals(3 * 512);
    for (int e = 0; e < 512; ++e) {
        globals[3*e] = 0.0000001f * float(1 + e%3);
        globals[3*e+1] = 0.0000003f * float(1 + e%5);
        globals[3*e+2] = 0.00000001f * float(1 + e%7);
    }
    std::vector<int32_t> iv(size_t(10) * T);
    std::vector<int32_t> lv(iv.size());
    for (size_t i = 0; i < xv.size(); ++i) xv[i] = float(int(i % 29) - 14) * input_amplitude;
    for (int t = 0; t < T; ++t) for (int k = 0; k < 10; ++k) {
        const int index = t * 10 + k;
        iv[index] = (t % 2) ? (k == 0 ? 511 : k - 1) : (k == 9 ? 511 : 8 - k);
        lv[index]=iv[index];
        if(decode_calibrated) iv[index]=iv[index]==511 ? 512 : iv[index]+17;
        wv[index] = float(k + 1) / 55.0f;
    }
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
    ggml_backend_tensor_set(ids, iv.data(), 0, iv.size() * sizeof(int32_t));
    if(logical_ids) ggml_backend_tensor_set(logical_ids,lv.data(),0,lv.size()*sizeof(int32_t));
    ggml_backend_tensor_set(weights, wv.data(), 0, wv.size() * sizeof(float));
    if (input_scale) ggml_backend_tensor_set(input_scale, globals.data(), 0, globals.size() * sizeof(float));
    ggml_backend_synchronize(bp.cuda);
    auto direct_reference = [&]() {
        if(decode_calibrated) return decoded((const float *)x->data,(const int32_t *)ids->data,
            (const int32_t *)logical_ids->data,(const uint8_t *)raw->data,(const float *)input_scale->data,
            2764816,physical_slots,T,10,(float *)reference->data,scratch->data,bytes,nullptr);
        return input_scale ? calibrated((const float *) x->data, (const int32_t *) ids->data,
            (const uint8_t *) raw->data, (const float *) input_scale->data, 2764816, 512, T, 10,
            (float *) reference->data, scratch->data, bytes, nullptr)
            : direct((const float *) x->data, (const int32_t *) ids->data, (const uint8_t *) raw->data,
                2764816, 512, T, 10, (float *) reference->data, scratch->data, bytes, nullptr, mode);
    };
    const int status = direct_reference();
    bool ok = status == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
    std::vector<float> ref(size_t(2560) * 10 * T), actual(ref.size()), reduced(size_t(2560) * T);
    ggml_backend_tensor_get(reference, ref.data(), 0, ref.size() * sizeof(float));
    auto reference_diagnostics = [&](const char * phase, int result, const std::vector<float> * previous) {
        size_t nonzero = 0, different = 0; double maximum = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            nonzero += ref[i] != 0; maximum = std::max(maximum, double(std::fabs(ref[i])));
            if (previous) different += ref[i] != (*previous)[i];
        }
        std::printf("NVFP4_REFERENCE mode=%d T=%d fixture=%s phase=%s status=%d finite=%d nonzero=%zu maxabs=%.9g different=%zu\n",
            mode,T,expected_zero?"underflow-zero":"changing-output",phase,result,int(finite(ref)),nonzero,maximum,different);
        return finite(ref) && (expected_zero ? nonzero == 0 : nonzero > 0) && (!previous || expected_zero || different > 0);
    };
    ok = reference_diagnostics("initial",status,nullptr) && ok;
    auto check = [&](const char * phase) {
        ggml_backend_synchronize(bp.cuda);
        ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
        ggml_backend_tensor_get(sum, reduced.data(), 0, reduced.size() * sizeof(float));
        bool pass = finite(actual) && actual == ref;
        size_t different = 0;
        for (size_t i = 0; i < actual.size(); ++i) different += actual[i] != ref[i];
        for (int t = 0; t < T; ++t) for (int d = 0; d < 2560; ++d) {
            float expected = ref[(size_t(t) * 10) * 2560 + d] * wv[t * 10];
            for (int k = 1; k < 10; ++k) expected += ref[(size_t(t) * 10 + k) * 2560 + d] * wv[t * 10 + k];
            const float value = reduced[size_t(t) * 2560 + d];
            pass = pass && std::isfinite(value) && std::fabs(value - expected) <= 1e-6f * (1.0f + std::fabs(expected));
        }
        std::printf("NVFP4_CUSTOM_COMPARE mode=%d T=%d fixture=%s phase=%s finite=%d different=%zu maxabs=%.9g weighted_sum_pass=%d\n",
            mode,T,expected_zero?"underflow-zero":"changing-output",phase,int(finite(actual)),different,maxabs(actual,ref),int(pass));
        return pass;
    };
    ok = ggml_backend_graph_compute(bp.cuda, gf) == GGML_STATUS_SUCCESS && ok;
    ok = check("eager") && ok;
    ggml_backend_cuda_graphs_disable_pop(bp.cuda);
    const void * stable_x = x->data, * stable_aux = input_scale ? input_scale->data : nullptr;
    const void * stable_logical=logical_ids ? logical_ids->data : nullptr;
    for (int replay = 0; replay < (decode_calibrated ? 4 : 2); ++replay) {
        if (replay > 0) {
            const auto previous = ref;
            if(decode_calibrated && replay==1) {
                for(auto & id:lv) id=(id+103)%512;
                ggml_backend_tensor_set(logical_ids,lv.data(),0,lv.size()*sizeof(int32_t));
            } else if(decode_calibrated && replay==3) {
                std::fill(iv.begin(),iv.end(),512); // safe repeated physical placeholders on discarded misses
                ggml_backend_tensor_set(ids,iv.data(),0,iv.size()*sizeof(int32_t));
            } else {
                for (size_t i = 0; i < xv.size(); ++i) xv[i] = xv[i] * 0.875f + 0.000013f;
                ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
                if (input_scale) {
                    for (int e = 0; e < 512; ++e) {
                        globals[3*e] *= 0.75f; globals[3*e+1] *= 1.25f; globals[3*e+2] *= 1.5f;
                    }
                    ggml_backend_tensor_set(input_scale, globals.data(), 0, globals.size() * sizeof(float));
                }
            }
            ggml_backend_synchronize(bp.cuda); // input mutation/reference execution are outside capture
            const int changed_status = direct_reference();
            ok = changed_status == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess && ok;
            ggml_backend_tensor_get(reference, ref.data(), 0, ref.size() * sizeof(float));
            ok = reference_diagnostics("changed-input",changed_status,&previous) && ok;
            ok = x->data == stable_x && (!input_scale || input_scale->data == stable_aux) && ok;
            ok = (!logical_ids || logical_ids->data==stable_logical) && ok;
        }
        ok = ggml_backend_cuda_graph_compute_static(bp.cuda, gf) == GGML_STATUS_SUCCESS && ok;
        ok = check(replay==0 ? "captured" : decode_calibrated && replay==1 ? "logical-only-replay" :
            decode_calibrated && replay==3 ? "duplicate-physical-placeholders" : "changed-replay") && ok;
    }
    ggml_backend_cuda_graphs_disable_push(bp.cuda);
    report(name, ok, "unweighted direct-DLL bit identity; original router mul/sequential sum; required static capture + changed-x/aux replay at stable pointers");
    ggml_backend_buffer_free(buf); ggml_backend_buffer_free(scratch_buf);
    ggml_free(ctx); ggml_free(scratch_ctx); FreeLibrary(dll);
}

#include "qfn-qsa-custom-test.h"
#include "qfn-index-fp4-test.h"
#include "qfn-index-refine-test.h"

#include "qfn-doorbell-interop-test.h"

int main(int argc, char ** argv) {
    const bool expert_reduce = argc==2 && std::strcmp(argv[1],"--expert-reduce-self-test")==0;
    const bool index_refine = argc==2 && std::strcmp(argv[1],"--indexer-refine-self-test")==0;
    const bool index_fp4 = argc == 2 && std::strcmp(argv[1], "--indexer-fp4-self-test") == 0;
    const bool interop = argc == 3 && std::strcmp(argv[1], "--doorbell-interop-self-test") == 0;
    const bool gateup = argc == 2 && std::strcmp(argv[1], "--gateup-self-test") == 0;
    if (argc != 1 && !interop && !gateup && !index_fp4 && !index_refine && !expert_reduce) {
        std::fprintf(stderr,"usage: qfn-op-test.exe [--expert-reduce-self-test | --gateup-self-test | --indexer-refine-self-test | --indexer-fp4-self-test | --doorbell-interop-self-test FILE_WITH_TWO_1MiB_PAYLOADS]\n");
        return 2;
    }
    ggml_backend_load_all();
    backend_pair bp;
    bp.cpu  = ggml_backend_cpu_init();
    bp.cuda = ggml_backend_cuda_init(0);
    if (!bp.cpu || !bp.cuda) { fprintf(stderr, "backend init failed\n"); return 2; }
    ggml_backend_cpu_set_n_threads(bp.cpu, 16);
    ggml_backend_cuda_graphs_disable_push(bp.cuda);  // dynamic test graphs: node-by-node path, as the engine prefill uses

    if (expert_reduce) {
        for (int64_t T : {1,2,3,4,17,2048}) test_expert_reduce(bp,T,false);
        test_expert_reduce(bp,17,true);
        test_expert_reduce(bp,4,true,true);
        test_expert_reduce(bp,17,false,false,true);
        ggml_backend_free(bp.cuda); ggml_backend_free(bp.cpu);
        return g_fail ? 1 : 0;
    }

    if(index_refine){for(int T:{1,17})for(auto kt:{GGML_TYPE_F16,GGML_TYPE_F32})test_qfn_index_refine(bp,T,kt);
        ggml_backend_free(bp.cuda);ggml_backend_free(bp.cpu);return g_fail?1:0;}
    if (index_fp4) {
        for (int T : {1,17}) for(auto kt : {GGML_TYPE_F16,GGML_TYPE_F32}) test_qfn_index_fp4(bp,T,kt);
        ggml_backend_free(bp.cuda);ggml_backend_free(bp.cpu);return g_fail?1:0;
    }

    if (interop) {
        int status=1;
        try { status=test_doorbell_interop(bp,argv[2]); }
        catch (const std::exception & error) { std::fprintf(stderr,"DOORBELL_INTEROP FAIL: %s\n",error.what()); }
        ggml_backend_free(bp.cuda);ggml_backend_free(bp.cpu);
        return status;
    }


    if (gateup) {
        for (int64_t M : { 1, 4, 40, 512 }) test_q2_0_gate_up(bp, M);
        ggml_backend_free(bp.cuda);
        ggml_backend_free(bp.cpu);
        printf("%s (%d failing)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
        return g_fail ? 1 : 0;
    }

    test_qsa_score_custom(bp,1,2,GGML_TYPE_F16);
    test_qsa_score_custom(bp,17,2,GGML_TYPE_F32);
    test_qsa_attention_custom(bp,1,false);
    test_qsa_attention_custom(bp,4,true);
    test_qsa_attention_custom(bp,17,true);

    test_nvfp4_custom(bp, 4, QFN_NVFP4_NATIVE_Q8_1, true); // original tiny-input underflow remains an explicit gate
    test_nvfp4_custom(bp, 4, QFN_NVFP4_NATIVE_Q8_1, false, 0.01f); // nondegenerate hidden F16/q8 scale, MUST change
    test_nvfp4_custom(bp, 1024, QFN_NVFP4_W4A4);
    test_nvfp4_custom(bp, 2048, QFN_NVFP4_MMQ_Q8);
    test_nvfp4_custom(bp, 17, QFN_NVFP4_CALIBRATED_A4);
    test_nvfp4_custom(bp, 4, QFN_NVFP4_CALIBRATED_DECODE);
    test_q2_capture_mmid(bp, 1);
    test_q2_capture_mmid(bp, 10);

    test_q2_0_get_rows(bp);
    test_iq4_nl_get_rows(bp);
    for (int64_t M : { 1, 3, 8, 16, 64, 300 }) test_q2_0_mul_mat(bp, 2560, 640, M);
    test_q2_0_mul_mat(bp, 640, 2560, 37);
    for (int64_t M : { 1, 3, 40, 512 }) test_q2_0_mul_mat_id(bp, M);
    for (int64_t M : { 1, 4, 40, 512 }) test_q2_0_gate_up(bp, M);
    test_q2_0_batch_shape(bp, false);
    test_q2_0_batch_shape(bp, true);
    for (int64_t T : { 1, 7, 63 }) test_gdn(bp, T, 1);
    for (int64_t T : { 64, 200, 1000 }) test_gdn(bp, T, 1);
    test_gdn(bp, 9, 4);
    test_qsa_select(bp);
    test_slot_map_record_only(bp);
    test_slot_map_timeout_latch(bp);
    for (int64_t T : { 1, 4, 300 }) test_hc_fused(bp, T);
    for (int64_t T : {1,4,2048}) test_expert_reduce(bp,T,false);
    test_qsa_attn(bp, 1024, 64, 0);        // every pool visible (dense causal + tail)
    test_qsa_attn(bp, 8192, 48, 7000);     // top-512 pools of ~1750 visible, plus the tail

    ggml_backend_free(bp.cuda);
    ggml_backend_free(bp.cpu);
    printf("%s (%d failing)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
