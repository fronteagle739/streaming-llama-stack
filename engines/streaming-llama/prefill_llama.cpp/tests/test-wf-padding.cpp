// test-wf-padding.cpp — Prove that CUDA mmq requires MATRIX_ROW_PADDING
// between quantized tensors when ne[0] % 512 != 0.
//
// Test strategy:
//   1. Create two Q8_0 weight tensors: A [192, 64] and B [192, 64]
//   2. Allocate in a PROPER ggml CUDA buffer (with get_alloc_size padding)
//   3. Fill A with known quantized data, do mul_mat(A, x), verify no NaN
//   4. Pack A and B TIGHTLY (no padding) in a raw cudaMalloc buffer
//   5. Do mul_mat(A_tight, x) — expect NaN from garbage padding
//
// If test 3 passes and test 5 produces NaN, the padding is confirmed
// as the root cause of the WF streaming NaN issue.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static constexpr int NE0 = 192;   // ne[0] — NOT a multiple of 512
static constexpr int NE1 = 64;    // ne[1]
static constexpr int N_TOKENS = 4;

// Quantize f32 data to Q8_0 in-place
static void quantize_to_q8_0(const float * src, void * dst, int64_t n_elements) {
    // Use ggml's quantization
    ggml_quantize_chunk(GGML_TYPE_Q8_0, src, dst, 0, 1, n_elements, nullptr);
}

static int count_nan(const float * data, int n) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (std::isnan(data[i])) count++;
    }
    return count;
}

int main() {
    printf("=== test-wf-padding: CUDA mmq MATRIX_ROW_PADDING test ===\n");
    printf("Testing Q8_0 tensor with ne[0]=%d (not multiple of 512)\n\n", NE0);

    // Check CUDA is available
    if (ggml_backend_cuda_get_device_count() == 0) {
        printf("SKIP: no CUDA device\n");
        return 0;
    }

    // Get CUDA backend and buffer type
    ggml_backend_t gpu = ggml_backend_init_by_name("CUDA0", nullptr);
    if (!gpu) {
        printf("FAIL: cannot init CUDA backend\n");
        return 1;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(gpu);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    // Check alloc_size vs nbytes
    {
        struct ggml_init_params ip = { 2 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, NE0, NE1);
        size_t nb = ggml_nbytes(t);
        size_t as = ggml_backend_buft_get_alloc_size(buft, t);
        printf("Q8_0 [%d, %d]: nbytes=%zu alloc_size=%zu padding=%zu\n",
               NE0, NE1, nb, as, as - nb);
        if (as == nb) {
            printf("SKIP: no padding needed (ne[0] %% 512 == 0?)\n");
            ggml_free(ctx);
            ggml_backend_free(gpu);
            return 0;
        }
        ggml_free(ctx);
    }

    // Create f32 source data for the weight tensor
    std::vector<float> weight_f32(NE0 * NE1);
    for (int i = 0; i < NE0 * NE1; i++) {
        weight_f32[i] = 0.01f * ((i % 100) - 50);  // values in [-0.5, 0.5]
    }

    // Quantize to Q8_0
    size_t q8_nbytes = ggml_row_size(GGML_TYPE_Q8_0, NE0) * NE1;
    std::vector<uint8_t> weight_q8(q8_nbytes);
    quantize_to_q8_0(weight_f32.data(), weight_q8.data(), NE0 * NE1);

    // Create input tensor (f32)
    std::vector<float> input_f32(NE0 * N_TOKENS);
    for (int i = 0; i < NE0 * N_TOKENS; i++) {
        input_f32[i] = 0.1f * ((i % 20) - 10);  // values in [-1.0, 1.0]
    }

    // --- TEST 1: Proper ggml buffer (with padding) ---
    printf("\n--- TEST 1: Proper ggml CUDA buffer (with alloc_size padding) ---\n");
    {
        struct ggml_init_params ip = { 1024 * 1024, nullptr, true };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, NE0, NE1);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NE0, N_TOKENS);
        ggml_tensor * y = ggml_mul_mat(ctx, w, x);  // [NE1, N_TOKENS]

        // Allocate in proper CUDA buffer
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft,
            ggml_backend_buft_get_alloc_size(buft, w) +
            ggml_nbytes(x) + ggml_nbytes(y) + 4096);

        // Set tensor data in the buffer
        size_t off = 0;
        w->buffer = buf; w->data = (char*)ggml_backend_buffer_get_base(buf) + off;
        ggml_backend_buffer_init_tensor(buf, w);  // zeros padding!
        off += ggml_backend_buft_get_alloc_size(buft, w);

        x->buffer = buf; x->data = (char*)ggml_backend_buffer_get_base(buf) + off;
        off += ggml_nbytes(x);

        y->buffer = buf; y->data = (char*)ggml_backend_buffer_get_base(buf) + off;

        // Upload data
        ggml_backend_tensor_set(w, weight_q8.data(), 0, q8_nbytes);
        ggml_backend_tensor_set(x, input_f32.data(), 0, NE0 * N_TOKENS * sizeof(float));

        // Build and compute graph
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_backend_graph_compute(gpu, gf);

        // Check result
        std::vector<float> result(NE1 * N_TOKENS);
        ggml_backend_tensor_get(y, result.data(), 0, result.size() * sizeof(float));
        int n_nan = count_nan(result.data(), result.size());
        printf("Result: %d/%zu NaN, first=[%.4g, %.4g, %.4g, %.4g]\n",
               n_nan, result.size(), result[0], result[1], result[2], result[3]);
        printf("TEST 1: %s\n", n_nan == 0 ? "PASS (no NaN with proper padding)" : "FAIL");

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    // --- TEST 2: Tightly packed buffer (no padding) ---
    printf("\n--- TEST 2: Tightly packed buffer (NO padding, like WF) ---\n");
    {
        struct ggml_init_params ip = { 1024 * 1024, nullptr, true };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, NE0, NE1);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NE0, N_TOKENS);
        ggml_tensor * y = ggml_mul_mat(ctx, w, x);

        // Allocate buffer WITHOUT padding between tensors
        // Pack: [weight_q8_data][input_f32_data][output] — no gap after weight
        size_t total = q8_nbytes + NE0 * N_TOKENS * sizeof(float) + NE1 * N_TOKENS * sizeof(float) + 4096;
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, total);
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        size_t off = 0;
        w->buffer = buf; w->data = (char*)ggml_backend_buffer_get_base(buf) + off;
        // Do NOT call init_tensor — no padding clear!
        off += q8_nbytes;  // tight packing, no alloc_size gap

        x->buffer = buf; x->data = (char*)ggml_backend_buffer_get_base(buf) + off;
        off += NE0 * N_TOKENS * sizeof(float);

        y->buffer = buf; y->data = (char*)ggml_backend_buffer_get_base(buf) + off;

        // Upload data — input data goes RIGHT AFTER weight with no gap
        // The mmq kernel will read x's f32 data as Q8_0 "padding"
        ggml_backend_tensor_set(w, weight_q8.data(), 0, q8_nbytes);
        ggml_backend_tensor_set(x, input_f32.data(), 0, NE0 * N_TOKENS * sizeof(float));

        // Build and compute graph
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_backend_graph_compute(gpu, gf);

        // Check result — expect NaN from garbage in padding area
        std::vector<float> result(NE1 * N_TOKENS);
        ggml_backend_tensor_get(y, result.data(), 0, result.size() * sizeof(float));
        int n_nan = count_nan(result.data(), result.size());
        printf("Result: %d/%zu NaN, first=[%.4g, %.4g, %.4g, %.4g]\n",
               n_nan, result.size(), result[0], result[1], result[2], result[3]);
        if (n_nan > 0) {
            printf("TEST 2: PASS (NaN confirms padding is needed — root cause proven)\n");
        } else {
            printf("TEST 2: INFO (no NaN — kernel may not read padding for this shape)\n");
        }

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    // --- TEST 3: Tight packing with zero-fill (simulates cudaMemset fix) ---
    printf("\n--- TEST 3: Tight packing with zero-fill (cudaMemset before load) ---\n");
    {
        struct ggml_init_params ip = { 1024 * 1024, nullptr, true };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, NE0, NE1);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NE0, N_TOKENS);
        ggml_tensor * y = ggml_mul_mat(ctx, w, x);

        // Allocate with alloc_size spacing but use WEIGHTS usage
        size_t w_alloc = ggml_backend_buft_get_alloc_size(buft, w);
        size_t total = w_alloc + NE0 * N_TOKENS * sizeof(float) + NE1 * N_TOKENS * sizeof(float) + 4096;
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, total);
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        // Zero entire buffer first (simulates cudaMemset in transfer_layer)
        ggml_backend_buffer_clear(buf, 0);

        size_t off = 0;
        w->buffer = buf; w->data = (char*)ggml_backend_buffer_get_base(buf) + off;
        off += w_alloc;  // alloc_size spacing — leaves zero-filled gap

        x->buffer = buf; x->data = (char*)ggml_backend_buffer_get_base(buf) + off;
        off += NE0 * N_TOKENS * sizeof(float);

        y->buffer = buf; y->data = (char*)ggml_backend_buffer_get_base(buf) + off;

        // Upload data
        ggml_backend_tensor_set(w, weight_q8.data(), 0, q8_nbytes);
        ggml_backend_tensor_set(x, input_f32.data(), 0, NE0 * N_TOKENS * sizeof(float));

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_backend_graph_compute(gpu, gf);

        std::vector<float> result(NE1 * N_TOKENS);
        ggml_backend_tensor_get(y, result.data(), 0, result.size() * sizeof(float));
        int n_nan = count_nan(result.data(), result.size());
        printf("Result: %d/%zu NaN, first=[%.4g, %.4g, %.4g, %.4g]\n",
               n_nan, result.size(), result[0], result[1], result[2], result[3]);
        printf("TEST 3: %s\n", n_nan == 0
               ? "PASS (alloc_size + zero-fill = correct padding)"
               : "FAIL (NaN even with alloc_size spacing and zero-fill)");

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    printf("\n=== Summary ===\n");
    printf("If TEST 1 passes and TEST 2 shows NaN: padding is root cause\n");
    printf("If TEST 3 passes: alloc_size spacing + zero-fill is sufficient fix\n");

    ggml_backend_free(gpu);
    return 0;
}
