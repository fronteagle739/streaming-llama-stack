#pragma once
// Experimental graph composition from existing casts, subtraction, scaling,
// addition and the qualified-shape F16/F16/F32 cuBLAS dispatch. No CUDA kernel.
// Native dense weights are already F16 at this point. Only the F32 RHS is split;
// HC, attention probabilities, norms, routing and expert products are excluded
// by the existing dense projection allowlist. This is NOT a bit-exact claim.
#include "ggml-dsv41-dense-half.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

inline ggml_cgraph * llama_dsv41_dense_residual_graph(ggml_context * ctx,
        ggml_cgraph * original, unsigned terms, uint64_t * products,
        uint64_t * backend_temporary) {
    if (!ctx || !original || !products || !backend_temporary || terms < 2 || terms > 3)
        throw std::runtime_error("Dense residual composition requires exactly two or three admitted terms");
    *products = 0;
    const int nodes = ggml_graph_n_nodes(original);
    for (int i = 0; i < nodes; ++i) {
        auto * node = ggml_graph_node(original, i);
        if (!ggml_dsv41_dense_half_tag(node)) continue;
        auto * weight = node->src[0];
        auto * residual = node->src[1];
        std::vector<ggml_tensor *> pieces;
        float output_scale = 1.f;
        for (unsigned part = 0; part < terms; ++part) {
            auto * product = ggml_mul_mat(ctx, weight, residual);
            ggml_mul_mat_set_prec(product, GGML_PREC_F32);
            if (!ggml_dsv41_dense_half_tag(product))
                throw std::runtime_error("Dense residual product lost the original weight/shape allowlist");
            ggml_format_name(product, "dsv41_dense_residual_%u", part);
            *backend_temporary = std::max(*backend_temporary,
                uint64_t(ggml_nelements(residual))*2 + 65536);
            ++*products;
            pieces.push_back(part ? ggml_scale(ctx, product, output_scale) : product);
            if (part + 1 < terms) {
                auto * rounded = ggml_cast(ctx, ggml_cast(ctx, residual, GGML_TYPE_F16), GGML_TYPE_F32);
                // Power-of-two scaling keeps the small correction away from
                // F16 underflow. Existing finite-output validation remains
                // mandatory; overflow and rounding are measured, not hidden.
                residual = ggml_scale(ctx, ggml_sub(ctx, residual, rounded), 4096.f);
                output_scale *= 1.f/4096.f;
            }
        }
        auto * joined = pieces.back();
        for (size_t part = pieces.size()-1; part > 0; --part)
            joined = ggml_add(ctx, pieces[part-1], joined);
        if (std::memcmp(node->ne, joined->ne, sizeof(node->ne)) ||
            std::memcmp(node->nb, joined->nb, sizeof(node->nb)))
            throw std::runtime_error("Dense residual composition changed output geometry");
        // Preserve the exact tensor identity retained by all downstream users.
        node->op = joined->op;
        std::memcpy(node->op_params, joined->op_params, sizeof(node->op_params));
        std::memcpy(node->src, joined->src, sizeof(node->src));
        node->view_src = joined->view_src;
        node->view_offs = joined->view_offs;
    }
    auto * graph = ggml_new_graph_custom(ctx, ggml_graph_size(original), false);
    for (int i = 0; i < nodes; ++i)
        ggml_build_forward_expand(graph, ggml_graph_node(original, i));
    return graph;
}
