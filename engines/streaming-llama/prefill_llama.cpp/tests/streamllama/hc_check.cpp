// Independent scalar check for the legacy V4/Kimi HC operation and the new
// elementwise gated variant. No model, CUDA initialization or downloads.
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

static bool check(int hc) {
    constexpr int width = 17, tokens = 3;
    ggml_init_params params{4 * 1024 * 1024, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, hc, tokens);
    auto * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, tokens);
    auto * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, hc, tokens);
    auto * pre = ggml_dsv4_hc_pre(ctx, x, weights);
    auto * gated = ggml_dsv4_hc_pre_gated(ctx, x, gate, .25f);
    auto * post = ggml_dsv4_hc_post(ctx, pre, x, weights, nullptr);
    float * xv = static_cast<float *>(x->data);
    float * wv = static_cast<float *>(weights->data);
    float * gv = static_cast<float *>(gate->data);
    for (int i = 0; i < width * hc * tokens; ++i) {
        xv[i] = std::sin(float(i) * .13f);
        gv[i] = std::cos(float(i) * .17f);
    }
    for (int i = 0; i < hc * tokens; ++i) wv[i] = std::cos(float(i) * .27f);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, pre);
    ggml_build_forward_expand(graph, gated);
    ggml_build_forward_expand(graph, post);
    bool ok = ggml_graph_compute_with_ctx(ctx, graph, 2) == GGML_STATUS_SUCCESS;
    float error = 0;
    for (int t = 0; t < tokens; ++t) {
        for (int i = 0; i < width; ++i) {
            float expected = 0, expected_gated = 0;
            for (int h = 0; h < hc; ++h) {
                const int at = i + width * (h + hc * t);
                expected += xv[at] * wv[h + hc * t];
                expected_gated += xv[at] / (1.f + std::exp(-gv[at]));
            }
            const float a = static_cast<float *>(pre->data)[i + width * t];
            const float b = static_cast<float *>(gated->data)[i + width * t];
            ok = ok && std::isfinite(a) && std::isfinite(b);
            error = std::max(error, std::abs(a - expected));
            error = std::max(error, std::abs(b - .25f * expected_gated));
            for (int h = 0; h < hc; ++h) {
                const int at = i + width * (h + hc * t);
                const float c = static_cast<float *>(post->data)[at];
                ok = ok && std::isfinite(c);
                error = std::max(error, std::abs(c - (expected * wv[h + hc * t] + xv[at])));
            }
        }
    }
    ok = ok && error < 1e-5f;
    std::printf("HC streams=%d legacy_pre,gated_pre,identity_post max_abs_diff=%g %s\n",
                hc, double(error), ok ? "PASS" : "FAIL");
    ggml_free(ctx);
    return ok;
}

int main() {
    const bool v4 = check(4), kimi = check(12);
    return v4 && kimi ? 0 : 1;
}
