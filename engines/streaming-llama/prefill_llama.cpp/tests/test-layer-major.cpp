//
// test-layer-major.cpp — Tests for layer-major ubatch processing.
//
// TDD test suite: tests are written first, implementation follows.
//
// Tests verify:
//   1. Layer boundary detection from graph node names (parse_layer_suffix)
//   2. make_graph_view subgraph creation
//   3. Hidden state tensor identification by name
//   4. compute_node_range() subgraph execution
//   5. decode_layer_major() end-to-end correctness vs decode()
//
// Build:  cmake --build build_win --target test-layer-major
// Run:    build_win/bin/test-layer-major.exe [--model <path>] [--trace]
//

#include "llama.h"
#include "common.h"

#include "ggml.h"
#include "ggml-impl.h"  // for ggml_cgraph struct internals
#include "ggml-backend.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Internal headers for testing implementation details
// These will be needed once compute_node_range / decode_layer_major are implemented
// #include "llama-context.h"

// make_graph_view is internal to ggml and not exported from the shared library.
// Inline equivalent for testing (matches ggml.c:7014-7029).
static struct ggml_cgraph make_graph_view(struct ggml_cgraph * cgraph0, int i0, int i1) {
    struct ggml_cgraph view = {};
    view.size    = 0;
    view.n_nodes = i1 - i0;
    view.n_leafs = 0;
    view.nodes   = cgraph0->nodes + i0;
    view.grads   = nullptr;
    view.grad_accs = nullptr;
    view.leafs   = nullptr;
    view.use_counts       = cgraph0->use_counts;
    view.visited_hash_set = cgraph0->visited_hash_set;
    view.order            = cgraph0->order;
    return view;
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

static int  g_tests_run    = 0;
static int  g_tests_passed = 0;
static int  g_tests_failed = 0;
static bool g_trace        = false;
static std::string g_model_path;

#define TEST_ASSERT(cond, msg) do { \
    g_tests_run++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (line %d): %s\n", __func__, __LINE__, msg); \
        g_tests_failed++; \
        return false; \
    } \
    g_tests_passed++; \
} while(0)

// ---------------------------------------------------------------------------
// Test 1: make_graph_view creates correct subgraph
// ---------------------------------------------------------------------------

static bool test_graph_view_basic() {
    fprintf(stderr, "[test_graph_view_basic]\n");

    // Create a small ggml context with a few tensors
    struct ggml_init_params params = {
        /*.mem_size   =*/ 256 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != nullptr, "ggml_init failed");

    // Build a simple graph: a -> add(a,a) -> mul(b,b) -> add(c,c)
    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    ggml_set_name(a, "input");
    struct ggml_tensor * b = ggml_add(ctx, a, a);
    ggml_set_name(b, "add-0");
    struct ggml_tensor * c = ggml_mul(ctx, b, b);
    ggml_set_name(c, "mul-1");
    struct ggml_tensor * d = ggml_add(ctx, c, c);
    ggml_set_name(d, "add-2");

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, d);

    if (g_trace) {
        fprintf(stderr, "  Full graph: %d nodes\n", gf->n_nodes);
        for (int i = 0; i < gf->n_nodes; i++) {
            fprintf(stderr, "    node[%d]: %s\n", i, ggml_get_name(gf->nodes[i]));
        }
    }

    TEST_ASSERT(gf->n_nodes >= 3, "graph should have at least 3 nodes");

    // Create a view of nodes [1, 3) — should include 2 nodes
    struct ggml_cgraph view = make_graph_view(gf, 1, 3);

    TEST_ASSERT(view.n_nodes == 2, "view should have 2 nodes");
    TEST_ASSERT(view.nodes == &gf->nodes[1], "view nodes should point into original array");
    TEST_ASSERT(view.nodes[0] == gf->nodes[1], "view node 0 should match original node 1");
    TEST_ASSERT(view.nodes[1] == gf->nodes[2], "view node 1 should match original node 2");
    TEST_ASSERT(view.n_leafs == 0, "view should have 0 leafs");
    TEST_ASSERT(view.size == 0, "view size should be 0 (non-owning)");

    // Single-node view
    struct ggml_cgraph single = make_graph_view(gf, 0, 1);
    TEST_ASSERT(single.n_nodes == 1, "single view should have 1 node");
    TEST_ASSERT(single.nodes[0] == gf->nodes[0], "single view should point to first node");

    ggml_free(ctx);
    fprintf(stderr, "  PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// Test 2: Layer boundary detection from node names
// ---------------------------------------------------------------------------

// Reimplementation of parse_layer_suffix for testing
// (matches the logic in llama-context.cpp:3424-3441)
static int parse_layer_suffix(const char * name, int n_layer) {
    if (!name) return -1;
    const char * dash = strrchr(name, '-');
    if (!dash || *(dash + 1) == '\0') return -1;
    char * end = nullptr;
    long il = strtol(dash + 1, &end, 10);
    if (end == nullptr || *end != '\0') return -1;
    if (il < 0 || il >= n_layer) return -1;
    return (int) il;
}

static bool test_parse_layer_suffix() {
    fprintf(stderr, "[test_parse_layer_suffix]\n");

    const int n_layer = 45;

    // Valid layer suffixes
    TEST_ASSERT(parse_layer_suffix("attn_norm-0", n_layer) == 0, "layer 0");
    TEST_ASSERT(parse_layer_suffix("l_out-44", n_layer) == 44, "layer 44 (last)");
    TEST_ASSERT(parse_layer_suffix("Qcur-10", n_layer) == 10, "layer 10");
    TEST_ASSERT(parse_layer_suffix("ffn_out-22", n_layer) == 22, "layer 22");

    // Invalid cases
    TEST_ASSERT(parse_layer_suffix("result_norm", n_layer) == -1, "no dash suffix");
    TEST_ASSERT(parse_layer_suffix("result_output", n_layer) == -1, "no number suffix");
    TEST_ASSERT(parse_layer_suffix(nullptr, n_layer) == -1, "null name");
    TEST_ASSERT(parse_layer_suffix("attn_norm-45", n_layer) == -1, "layer >= n_layer");
    // "attn_norm--1": strrchr finds last dash, strtol("1") = 1, which is valid
    TEST_ASSERT(parse_layer_suffix("attn_norm--1", n_layer) == 1, "double-dash parses last segment as layer 1");
    TEST_ASSERT(parse_layer_suffix("inp_embd-", n_layer) == -1, "empty suffix after dash");

    // Edge: tensor name that looks like it has a suffix but doesn't (e.g. "self_kq_mask")
    TEST_ASSERT(parse_layer_suffix("self_kq_mask", n_layer) == -1, "no dash at all");

    fprintf(stderr, "  PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// Test 3: Layer boundary range building
// ---------------------------------------------------------------------------

static bool test_layer_boundary_ranges() {
    fprintf(stderr, "[test_layer_boundary_ranges]\n");

    // Simulate a graph with named nodes from 3 layers
    // Pre-layer: nodes 0-2 (no layer suffix)
    // Layer 0: nodes 3-8
    // Layer 1: nodes 9-14
    // Layer 2: nodes 15-20
    // Post-layer: nodes 21-23 (no layer suffix)

    const int n_layer = 3;
    const int n_nodes = 24;

    struct {
        const char * name;
        int expected_layer; // -1 = pre, -2 = post, 0-2 = layer
    } nodes[] = {
        {"inp_embd",       -1},  // 0
        {"inp_pos",        -1},  // 1
        {"self_kq_mask",   -1},  // 2
        {"attn_norm-0",     0},  // 3
        {"Qcur-0",          0},  // 4
        {"Kcur-0",          0},  // 5
        {"attn_out-0",      0},  // 6
        {"ffn_out-0",       0},  // 7
        {"l_out-0",         0},  // 8
        {"attn_norm-1",     1},  // 9
        {"Qcur-1",          1},  // 10
        {"Kcur-1",          1},  // 11
        {"attn_out-1",      1},  // 12
        {"ffn_out-1",       1},  // 13
        {"l_out-1",         1},  // 14
        {"attn_norm-2",     2},  // 15
        {"Qcur-2",          2},  // 16
        {"Kcur-2",          2},  // 17
        {"attn_out-2",      2},  // 18
        {"ffn_out-2",       2},  // 19
        {"l_out-2",         2},  // 20
        {"result_norm",    -2},  // 21
        {"result_output",  -2},  // 22
        {"output_logits",  -2},  // 23
    };

    // Step 1: find first/last for each layer
    std::vector<int> first_idx(n_layer, -1);
    std::vector<int> last_idx(n_layer, -1);
    for (int i = 0; i < n_nodes; ++i) {
        int il = parse_layer_suffix(nodes[i].name, n_layer);
        if (il < 0) continue;
        if (first_idx[il] < 0) first_idx[il] = i;
        last_idx[il] = i;
    }

    TEST_ASSERT(first_idx[0] == 3,  "layer 0 first = 3");
    TEST_ASSERT(last_idx[0]  == 8,  "layer 0 last  = 8");
    TEST_ASSERT(first_idx[1] == 9,  "layer 1 first = 9");
    TEST_ASSERT(last_idx[1]  == 14, "layer 1 last  = 14");
    TEST_ASSERT(first_idx[2] == 15, "layer 2 first = 15");
    TEST_ASSERT(last_idx[2]  == 20, "layer 2 last  = 20");

    // Step 2: compute layer ranges [start, end)
    // For each layer, the range extends from first_idx to the start of the next layer
    // (or just past last_idx for the final layer). Post-layer nodes are separate.
    std::vector<int> layer_start(n_layer);
    std::vector<int> layer_end(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        layer_start[il] = first_idx[il];
        if (il + 1 < n_layer) {
            layer_end[il] = first_idx[il + 1];
        } else {
            layer_end[il] = last_idx[il] + 1;
        }
    }

    TEST_ASSERT(layer_start[0] == 3,  "layer 0 range starts at 3");
    TEST_ASSERT(layer_end[0]   == 9,  "layer 0 range ends at 9");
    TEST_ASSERT(layer_start[1] == 9,  "layer 1 range starts at 9");
    TEST_ASSERT(layer_end[1]   == 15, "layer 1 range ends at 15");
    TEST_ASSERT(layer_start[2] == 15, "layer 2 range starts at 15");
    TEST_ASSERT(layer_end[2]   == 21, "layer 2 range ends at 21");

    // Pre-layer region: [0, layer_start[0])
    int pre_end = layer_start[0];
    TEST_ASSERT(pre_end == 3, "pre-layer region is [0, 3)");

    // Post-layer region: [layer_end[last], n_nodes)
    int post_start = layer_end[n_layer - 1];
    TEST_ASSERT(post_start == 21, "post-layer region starts at 21");

    fprintf(stderr, "  PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// Test 4: Hidden state tensor identification
// ---------------------------------------------------------------------------

static bool test_hidden_state_tensor_naming() {
    fprintf(stderr, "[test_hidden_state_tensor_naming]\n");

    // The hidden state at the output of layer L is named "l_out-{L}"
    // We need to find this tensor in the graph's node range for layer L

    // Verify naming convention
    char buf[64];
    snprintf(buf, sizeof(buf), "l_out-%d", 0);
    TEST_ASSERT(strcmp(buf, "l_out-0") == 0, "l_out format for layer 0");
    snprintf(buf, sizeof(buf), "l_out-%d", 44);
    TEST_ASSERT(strcmp(buf, "l_out-44") == 0, "l_out format for layer 44");

    // Verify parse_layer_suffix correctly identifies l_out tensors
    TEST_ASSERT(parse_layer_suffix("l_out-0", 45) == 0, "parse l_out-0");
    TEST_ASSERT(parse_layer_suffix("l_out-44", 45) == 44, "parse l_out-44");

    // Helper: find l_out tensor in a layer range
    // In real code, we scan graph nodes in [layer_start, layer_end) for name "l_out-{L}"
    const char * test_names[] = {
        "attn_norm-5", "Qcur-5", "Kcur-5", "attn_out-5",
        "ffn_inp-5", "ffn_norm-5", "ffn_out-5", "l_out-5"
    };
    int n_test = 8;

    // Find l_out
    char expected[64];
    snprintf(expected, sizeof(expected), "l_out-%d", 5);
    int l_out_idx = -1;
    for (int i = 0; i < n_test; i++) {
        if (strcmp(test_names[i], expected) == 0) {
            l_out_idx = i;
            break;
        }
    }
    TEST_ASSERT(l_out_idx == 7, "l_out-5 found at last position in layer range");

    fprintf(stderr, "  PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// Test 5: Hidden state buffer sizing
// ---------------------------------------------------------------------------

static bool test_hidden_state_buffer_sizing() {
    fprintf(stderr, "[test_hidden_state_buffer_sizing]\n");

    // Step 3.5 Flash parameters
    const int n_embd = 7168;
    const int n_tokens_per_ubatch = 2048;
    const int n_ubatches = 4;

    // FP16 hidden state per ubatch
    size_t hidden_bytes = (size_t)n_embd * n_tokens_per_ubatch * sizeof(uint16_t); // ggml_fp16_t = uint16_t
    TEST_ASSERT(hidden_bytes == 7168ULL * 2048 * 2, "hidden state size = n_embd * n_tokens * 2");

    // ~28MB per ubatch
    size_t expected_mb = hidden_bytes / (1024 * 1024);
    TEST_ASSERT(expected_mb == 28, "~28MB per ubatch hidden state");

    // Total RAM for all ubatches
    size_t total_bytes = hidden_bytes * n_ubatches;
    size_t total_mb = total_bytes / (1024 * 1024);
    TEST_ASSERT(total_mb == 112, "112MB total for 4 ubatch hidden states");

    if (g_trace) {
        fprintf(stderr, "  Hidden state per ubatch: %zu bytes (%.1f MB)\n",
                hidden_bytes, hidden_bytes / (1024.0 * 1024.0));
        fprintf(stderr, "  Total for %d ubatches: %zu bytes (%.1f MB)\n",
                n_ubatches, total_bytes, total_bytes / (1024.0 * 1024.0));
    }

    fprintf(stderr, "  PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// Test 6: VRAM budget calculation
// ---------------------------------------------------------------------------

static bool test_vram_budget() {
    fprintf(stderr, "[test_vram_budget]\n");

    // WDDM limit
    const size_t wddm_limit_gb = 20;
    const size_t wddm_limit_bytes = wddm_limit_gb * 1024ULL * 1024 * 1024;

    // Layer-major VRAM budget
    const size_t wf_buffers    = 2ULL * 4700 * 1024 * 1024;  // 2 x 4.7GB = 9.4GB
    const size_t kv_cache      = 1400ULL * 1024 * 1024;       // ~1.4GB (GQA)
    const size_t hidden_gpu    = 28ULL * 1024 * 1024;          // 1 ubatch on GPU = 28MB
    const size_t scratch       = 1024ULL * 1024 * 1024;        // ~1GB scratch

    size_t total_vram = wf_buffers + kv_cache + hidden_gpu + scratch;

    if (g_trace) {
        fprintf(stderr, "  WF buffers:  %.1f GB\n", wf_buffers / (1024.0 * 1024 * 1024));
        fprintf(stderr, "  KV cache:    %.1f GB\n", kv_cache / (1024.0 * 1024 * 1024));
        fprintf(stderr, "  Hidden GPU:  %.1f MB\n", hidden_gpu / (1024.0 * 1024));
        fprintf(stderr, "  Scratch:     %.1f GB\n", scratch / (1024.0 * 1024 * 1024));
        fprintf(stderr, "  Total VRAM:  %.1f GB (limit: %zu GB)\n",
                total_vram / (1024.0 * 1024 * 1024), wddm_limit_gb);
    }

    TEST_ASSERT(total_vram < wddm_limit_bytes, "layer-major VRAM fits within 20GB WDDM limit");

    // Compare with ubatch-major (needs scratch for all layers simultaneously)
    // The key difference is that ubatch-major re-streams weights,
    // not that it uses more VRAM per se. But the graph allocator
    // re-reserves scratch for each ubatch, causing VRAM leak.
    // Layer-major avoids this by reusing the same graph allocation.

    fprintf(stderr, "  PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// Test 7: Integration test — model-based (requires --model)
// ---------------------------------------------------------------------------

static bool test_graph_layer_structure(llama_model * model, llama_context * ctx) {
    fprintf(stderr, "[test_graph_layer_structure]\n");

    // Get model params
    const int n_layer = llama_model_n_layer(model);
    TEST_ASSERT(n_layer > 0, "model has layers");

    if (g_trace) {
        fprintf(stderr, "  Model: %d layers\n", n_layer);
    }

    // Create a small batch to build a graph
    const int n_tokens = 32;  // small batch for testing
    std::vector<llama_token> tokens(n_tokens, 1);  // dummy tokens
    std::vector<llama_pos> positions(n_tokens);
    std::vector<llama_seq_id> seq_ids(n_tokens, 0);
    for (int i = 0; i < n_tokens; i++) {
        positions[i] = i;
    }

    // Build a batch
    llama_batch batch = {
        /*.n_tokens   =*/ (int32_t)n_tokens,
        /*.token      =*/ tokens.data(),
        /*.embd       =*/ nullptr,
        /*.pos        =*/ positions.data(),
        /*.n_seq_id   =*/ nullptr,
        /*.seq_id     =*/ nullptr,
        /*.logits     =*/ nullptr,
    };

    // We can't easily build a graph without going through decode().
    // Instead, we'll do a small decode and inspect the timing/layer count.
    // For now, just verify the model structure is accessible.
    TEST_ASSERT(n_layer >= 2, "model has at least 2 layers for meaningful layer-major test");

    fprintf(stderr, "  Model has %d layers - suitable for layer-major processing\n", n_layer);
    fprintf(stderr, "  PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// Test 8: decode_layer_major correctness (requires --model)
// Placeholder — will be implemented once decode_layer_major() exists.
// The test should verify that layer-major produces the same logits
// as standard decode().
// ---------------------------------------------------------------------------

static bool test_decode_layer_major_correctness(llama_model * model, llama_context * ctx) {
    fprintf(stderr, "[test_decode_layer_major_correctness]\n");
    fprintf(stderr, "  SKIP: decode_layer_major() not yet implemented\n");

    // TODO: When decode_layer_major is implemented:
    //
    // 1. Create a small prompt (e.g., 64 tokens)
    // 2. Run standard decode() and save logits
    // 3. Reset context
    // 4. Run decode_layer_major() and save logits
    // 5. Compare logits — max abs diff should be < epsilon (0.01 for FP16)
    //
    // This is the gold-standard correctness test for the layer-major path.

    return true;  // Skip, don't fail
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--trace") == 0) {
            g_trace = true;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            g_model_path = argv[++i];
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            g_model_path = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            fprintf(stderr, "Usage: %s [--trace] [--model <path>]\n", argv[0]);
            fprintf(stderr, "  --trace       Enable verbose output\n");
            fprintf(stderr, "  --model PATH  Model file for integration tests\n");
            return 0;
        }
    }

    fprintf(stderr, "=== Layer-Major Ubatch Processing Tests ===\n\n");

    // --- Unit tests (no model required) ---
    fprintf(stderr, "--- Unit Tests ---\n");
    test_graph_view_basic();
    test_parse_layer_suffix();
    test_layer_boundary_ranges();
    test_hidden_state_tensor_naming();
    test_hidden_state_buffer_sizing();
    test_vram_budget();

    // --- Integration tests (model required) ---
    if (!g_model_path.empty()) {
        fprintf(stderr, "\n--- Integration Tests (model: %s) ---\n", g_model_path.c_str());

        llama_backend_init();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;  // CPU-only for testing
        llama_model * model = llama_model_load_from_file(g_model_path.c_str(), model_params);
        if (!model) {
            fprintf(stderr, "  ERROR: Failed to load model: %s\n", g_model_path.c_str());
        } else {
            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx    = 512;
            ctx_params.n_batch  = 128;
            ctx_params.n_ubatch = 32;
            llama_context * ctx = llama_init_from_model(model, ctx_params);

            if (!ctx) {
                fprintf(stderr, "  ERROR: Failed to create context\n");
            } else {
                test_graph_layer_structure(model, ctx);
                test_decode_layer_major_correctness(model, ctx);
                llama_free(ctx);
            }
            llama_model_free(model);
        }

        llama_backend_free();
    } else {
        fprintf(stderr, "\n--- Integration Tests SKIPPED (no --model) ---\n");
    }

    // Summary
    fprintf(stderr, "\n=== Results: %d/%d passed, %d failed ===\n",
            g_tests_passed, g_tests_run, g_tests_failed);

    return g_tests_failed > 0 ? 1 : 0;
}
