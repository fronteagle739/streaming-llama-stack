// LLAMADECODE B0: admission of the explicit F16 dense toggle by row count.
// Builds real ggml MUL_MAT nodes over a backend-allocated F16 weight with an
// allowlisted name, and checks that:
//  - prefill-sized steps (>8 rows) tag the projection and are admitted;
//  - a prefill-sized step whose graph tags nothing is still refused;
//  - 1-row and 8-row steps cannot be tagged and are admitted untagged;
//  - a tag at <=8 rows would be refused (the converse proof).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama-dsv41-dense-half-admission.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

static int failures = 0;
static void expect(bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

static unsigned tagged_products(int64_t rows, const char *weight_name) {
    ggml_init_params params{16 * 1024 * 1024, nullptr, true};
    ggml_context *weights = ggml_init(params);
    ggml_tensor *w = ggml_new_tensor_2d(weights, GGML_TYPE_F16, 64, 32);
    ggml_set_name(w, weight_name);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(weights, cpu);
    ggml_context *graph_ctx = ggml_init(params);
    ggml_tensor *x = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, 64, rows);
    ggml_tensor *y = ggml_mul_mat(graph_ctx, w, x);
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    ggml_cgraph *graph = ggml_new_graph(graph_ctx);
    ggml_build_forward_expand(graph, y);
    unsigned accepted = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
        if (ggml_dsv41_dense_half_tag(ggml_graph_node(graph, i))) ++accepted;
    ggml_free(graph_ctx);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(cpu);
    ggml_free(weights);
    return accepted;
}

int main() {
    const char *allowed = "blk.3.attn_q_a.weight", *excluded = "blk.3.ffn_gate_inp.weight";
    for (int64_t rows : std::initializer_list<int64_t>{9, 16, 4096}) {
        const unsigned n = tagged_products(rows, allowed);
        char what[160];
        std::snprintf(what, sizeof what, "rows=%lld allowlisted projection tagged and admitted", (long long)rows);
        expect(n == 1 && !llama_dsv41_dense_half_admission(uint32_t(rows), n), what);
    }
    {
        const unsigned n = tagged_products(16, excluded);
        const char *refused = llama_dsv41_dense_half_admission(16, n);
        expect(n == 0 && refused && std::strstr(refused, "matched no admitted projection"),
               "rows=16 with no taggable projection is still refused (prefill bar unchanged)");
    }
    for (int64_t rows : std::initializer_list<int64_t>{1, 8}) {
        const unsigned n = tagged_products(rows, allowed);
        char what[160];
        std::snprintf(what, sizeof what, "rows=%lld cannot be tagged and is admitted untagged", (long long)rows);
        expect(n == 0 && !llama_dsv41_dense_half_admission(uint32_t(rows), n), what);
    }
    expect(llama_dsv41_dense_half_admission(1, 1) != nullptr, "a tag at 1 row is refused (converse proof)");
    std::printf(failures ? "FAILED %d\n" : "ALL PASS\n", failures);
    return failures ? 1 : 0;
}
