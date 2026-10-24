// LLAMADECODE: a single-row ggml top-k (decode) must select the same ids in the
// same order as a multi-row top-k (prefill) over the same values, including
// widths above the 12288-column shared-memory bitonic limit, and repeat exactly.
// Runs on the CUDA backend. Control: LLAMA_DSV4_TOPK_CUB_1ROW=1 restores CUB.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

static int failures = 0;
static void expect(bool ok, const char *what) { std::printf("%s %s\n", ok ? "PASS" : "FAIL", what); if (!ok) ++failures; }

static std::vector<int32_t> run_topk(ggml_backend_t backend, const std::vector<float> &row, int64_t nrows, int k) {
    const int64_t ncols = int64_t(row.size());
    ggml_init_params params{16 * 1024 * 1024, nullptr, true};
    ggml_context *ctx = ggml_init(params);
    ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
    ggml_tensor *y = ggml_top_k(ctx, x, k);
    ggml_cgraph *g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    std::vector<float> all(size_t(ncols * nrows));
    for (int64_t r = 0; r < nrows; ++r) std::copy(row.begin(), row.end(), all.begin() + r * ncols);
    ggml_backend_tensor_set(x, all.data(), 0, all.size() * 4);
    ggml_backend_graph_compute(backend, g);
    std::vector<int32_t> out(size_t(k * nrows));
    ggml_backend_tensor_get(y, out.data(), 0, out.size() * 4);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    out.resize(size_t(k));  // first row
    return out;
}

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::printf("FAIL no CUDA backend\n"); return 2; }
    const bool cub_control = std::getenv("LLAMA_DSV4_TOPK_CUB_1ROW") && !std::strcmp(std::getenv("LLAMA_DSV4_TOPK_CUB_1ROW"), "1");
    std::printf("mode=%s\n", cub_control ? "CUB_1ROW control" : "deterministic (default)");
    std::mt19937 rng(1017);
    std::uniform_real_distribution<float> uni(-4.f, 4.f);
    struct cfg { int ncols, k; };
    for (const cfg c : {cfg{272, 272}, cfg{384, 6}, cfg{8192, 512}, cfg{16384, 512}, cfg{65536, 512}, cfg{131072, 2048}}) {
        std::vector<float> row(size_t(c.ncols));
        for (auto &v : row) v = std::round(uni(rng) * 64.f) / 64.f;  // coarse values: many exact ties
        const auto one_a = run_topk(backend, row, 1, c.k);
        const auto one_b = run_topk(backend, row, 1, c.k);
        const auto multi = run_topk(backend, row, 3, c.k);
        // Reference: value descending, index ascending on ties.
        std::vector<int32_t> ref(size_t(c.ncols));
        std::iota(ref.begin(), ref.end(), 0);
        std::stable_sort(ref.begin(), ref.end(), [&](int a, int b) { return row[a] > row[b]; });
        ref.resize(size_t(c.k));
        auto same_set = [&](std::vector<int32_t> a, std::vector<int32_t> b) { std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end()); return a == b; };
        char what[200];
        std::snprintf(what, sizeof what, "ncols=%d k=%d: 1-row repeats exactly", c.ncols, c.k);
        expect(one_a == one_b, what);
        std::snprintf(what, sizeof what, "ncols=%d k=%d: 1-row order == multi-row order", c.ncols, c.k);
        expect(one_a == multi, what);
        std::snprintf(what, sizeof what, "ncols=%d k=%d: selected set == reference set", c.ncols, c.k);
        expect(same_set(one_a, ref), what);
    }
    ggml_backend_free(backend);
    std::printf(failures ? "FAILED %d\n" : "ALL PASS\n", failures);
    return failures ? 1 : 0;
}
