// LLAMADECODE B1: router partition of a layer-shaped graph. Router-independent
// nodes (attention stand-in, router, shared expert, a late pool-tail output)
// come first; only routed consumers follow; topological order is kept.
#include "ggml.h"
#include "ggml-cpu.h"
#include "llama-dsv41-route-split.h"
#include <cstdio>
#include <unordered_set>

static int failures = 0;
static void expect(bool ok, const char *what) { std::printf("%s %s\n", ok ? "PASS" : "FAIL", what); if (!ok) ++failures; }

int main() {
    ggml_init_params params{64 * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(params);
    const int d = 64, e = 8, k = 2;
    ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, 1);
    ggml_tensor *w_attn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, d);
    ggml_tensor *w_router = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, e);
    ggml_tensor *w_shared = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, d);
    ggml_tensor *w_pool = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, d);
    ggml_tensor *experts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, d, e);
    ggml_tensor *h = ggml_mul_mat(ctx, w_attn, x);                       // attention stand-in
    ggml_tensor *logits = ggml_mul_mat(ctx, w_router, h);
    ggml_tensor *ids = ggml_cont(ctx, ggml_argsort_top_k(ctx, logits, k));
    ggml_set_name(ids, "dsv41_ffn_selected_experts");
    ggml_tensor *routed = ggml_mul_mat_id(ctx, experts, ggml_reshape_3d(ctx, h, d, 1, 1), ids);
    ggml_tensor *moe = ggml_view_2d(ctx, routed, d, 1, routed->nb[2], 0);
    ggml_tensor *shared = ggml_mul_mat(ctx, w_shared, h);
    ggml_tensor *out = ggml_cont(ctx, ggml_add(ctx, moe, shared));
    ggml_tensor *pool = ggml_cont(ctx, ggml_mul_mat(ctx, w_pool, x));    // late, router-independent output
    ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_build_forward_expand(graph, pool);
    const int n = ggml_graph_n_nodes(graph);
    std::string why;
    const int split = llama_dsv41_route_partition(graph, "dsv41_ffn_selected_experts", why);
    expect(split > 0 && split < n, "partition found a split");
    expect(ggml_graph_n_nodes(graph) == n, "node count unchanged");
    std::unordered_set<const ggml_tensor *> before, seen;
    bool topo = true, router_first = false, shared_first = false, pool_first = false, routed_after = true;
    for (int i = 0; i < n; ++i) {
        auto *node = ggml_graph_node(graph, i);
        for (const auto *src : node->src) if (src && src->op != GGML_OP_NONE && !seen.count(src)) {
            bool is_node = false; for (int j = 0; j < n; ++j) if (ggml_graph_node(graph, j) == src) is_node = true;
            if (is_node) topo = false;
        }
        seen.insert(node);
        if (i < split) before.insert(node);
    }
    router_first = before.count(ids) == 1; shared_first = before.count(shared) == 1; pool_first = before.count(pool) == 1;
    for (int i = 0; i < split; ++i) if (ggml_graph_node(graph, i) == routed) routed_after = false;
    expect(topo, "topological order holds after partition");
    expect(router_first, "router is before the split");
    expect(shared_first, "shared expert (router-independent) is before the split");
    expect(pool_first, "late router-independent output is before the split");
    expect(routed_after, "routed expert product is after the split");
    {   // a graph without a router is refused
        ggml_cgraph *g2 = ggml_new_graph(ctx); ggml_build_forward_expand(g2, pool);
        std::string w2; expect(llama_dsv41_route_partition(g2, "dsv41_ffn_selected_experts", w2) < 0 && !w2.empty(), "graph without router is refused");
    }
    // Execute both halves on CPU and compare with the unpartitioned result.
    ggml_set_f32(x, 0.5f); ggml_set_f32(w_attn, 0.01f); ggml_set_f32(w_shared, 0.02f); ggml_set_f32(w_pool, 0.03f); ggml_set_f32(experts, 0.04f);
    for (int i = 0; i < e; ++i) for (int j = 0; j < d; ++j) ggml_set_f32_nd(w_router, j, i, 0, 0, 0.001f * float(i));
    auto first = ggml_graph_view(graph, 0, split); auto second = ggml_graph_view(graph, split, n);
    ggml_graph_compute_with_ctx(ctx, &first, 1); ggml_graph_compute_with_ctx(ctx, &second, 1);
    const float split_value = ggml_get_f32_1d(out, 0);
    ggml_cgraph *ref = ggml_new_graph(ctx); ggml_build_forward_expand(ref, out); ggml_graph_compute_with_ctx(ctx, ref, 1);
    expect(split_value == ggml_get_f32_1d(out, 0), "split execution equals whole-graph execution");
    std::printf(failures ? "FAILED %d\n" : "ALL PASS\n", failures);
    ggml_free(ctx);
    return failures ? 1 : 0;
}
