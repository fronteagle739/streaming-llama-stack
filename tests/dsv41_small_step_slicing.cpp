// LLAMADECODE B0: a <=8-row step slices a large F16 PREC_F32 projection by output
// rows (bounded backend temporary), exactly as prefill steps do; a prefill step
// whose RHS has <=8 rows stays unsliced (unchanged behaviour).
#include "ggml.h"
#include <cstdint>
#include <cstdio>

ggml_cgraph *llama_dsv41_native_batch_graph(ggml_context *ctx, ggml_cgraph *original,
        uint32_t count, uint64_t operation_limit, uint64_t *backend_temporary,
        uint32_t *rounded_matrices, uint32_t *split_matrices);

static int failures = 0;
static void expect(bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

struct outcome { uint64_t temporary; uint32_t split; };
static outcome run(uint32_t count, int64_t rhs_rows) {
    ggml_init_params params{64 * 1024 * 1024, nullptr, true};
    ggml_context *ctx = ggml_init(params);
    // Engram KV geometry: F16 [6144, 25600] is 600 MiB when expanded to F32.
    ggml_tensor *w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 6144, 25600);
    ggml_set_name(w, "blk.1.engram_kv.weight");
    ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6144, rhs_rows);
    ggml_tensor *y = ggml_mul_mat(ctx, w, x);
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    ggml_cgraph *graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(graph, y);
    uint64_t temporary = 0; uint32_t rounded = 0, split = 0;
    llama_dsv41_native_batch_graph(ctx, graph, count, 96ull * 1024 * 1024, &temporary, &rounded, &split);
    ggml_free(ctx);
    return {temporary, split};
}

int main() {
    const uint64_t limit = 96ull * 1024 * 1024, whole = 6144ull * 25600 * 4;
    auto one = run(1, 1);
    expect(one.split == 1 && one.temporary <= limit, "1-row step: Engram KV sliced, temporary <= 96 MiB");
    auto eight = run(8, 8);
    expect(eight.split == 1 && eight.temporary <= limit, "8-row step: sliced, temporary <= 96 MiB");
    auto prefill = run(545, 545);
    expect(prefill.split == 1 && prefill.temporary <= limit, "545-row prefill step: sliced as before");
    auto narrow = run(545, 4);
    expect(narrow.split == 0 && narrow.temporary == whole, "prefill step with a 4-row RHS: unchanged (unsliced)");
    std::printf("temporaries: one=%llu eight=%llu prefill=%llu narrow=%llu\n", (unsigned long long)one.temporary,
        (unsigned long long)eight.temporary, (unsigned long long)prefill.temporary, (unsigned long long)narrow.temporary);
    std::printf(failures ? "FAILED %d\n" : "ALL PASS\n", failures);
    return failures ? 1 : 0;
}
