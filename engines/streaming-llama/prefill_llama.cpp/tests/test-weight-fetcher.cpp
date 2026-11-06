//
// test-weight-fetcher.cpp — Unit tests for the weight fetcher pipeline.
//
// Tests the NVMe → RAM → GPU pipeline independently of the LLM model.
// Creates a temporary test file, registers fake "layers" against it,
// and verifies the fetcher reads correct data through the full pipeline.
//
// Build:  cmake --build build_win --target test-weight-fetcher
// Run:    build_win/bin/test-weight-fetcher.exe [--trace]
//

#include "llama-weight-fetcher.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  if defined(LLAMA_DIRECTSTORAGE)
#    include <dstorage.h>
#    include <objbase.h>
#    include <dxgi1_4.h>    // IDXGIFactory4 for GPU adapter enumeration
#    if defined(LLAMA_CUDA_INTEROP)
#      include <cuda_runtime.h>
#    endif
#  endif
#else
#  include <unistd.h>
#  include <fcntl.h>
#endif

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

static int  g_tests_run    = 0;
static int  g_tests_passed = 0;
static int  g_tests_failed = 0;
static bool  g_trace        = false;
static float g_compute_ms   = 0.0f;   // --compute-ms <ms>  simulated per-layer compute

// Tunable I/O parameters (CrystalDiskMark-style flags)
static int  g_block_size_kb = 256;   // --bs <KiB>     block size in KiB
static int  g_threads       = 8;     // --threads <N>  I/O thread count
static int  g_queue_depth   = 8;     // --qd <N>       queue depth per handle
static int  g_ram_bufs      = 4;     // --ram-bufs <N> pinned RAM buffers
static int  g_gpu_bufs      = 2;     // --gpu-bufs <N> GPU staging buffers
static int  g_nlayers       = 8;     // --nlayers <N>  layers per file (fewer = bigger reads)
static int  g_wf_cycles     = 1;     // --wf-cycles <N> recreate WF to exercise persistent DS reuse
static int  g_wf_layer_mb   = 0;     // --wf-layer-mb <MiB> cap bytes read per synthetic layer

#define TEST_ASSERT(cond, msg) do { \
    g_tests_run++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (line %d): %s\n", __func__, __LINE__, msg); \
        g_tests_failed++; \
        return false; \
    } \
    g_tests_passed++; \
} while(0)

// Create a temporary file with known data pattern.
// Each byte = (offset % 251) — a prime to detect misalignment.
static std::string create_test_file(size_t size) {
#ifdef _WIN32
    char tmp_path[MAX_PATH];
    char tmp_file[MAX_PATH];
    GetTempPathA(MAX_PATH, tmp_path);
    GetTempFileNameA(tmp_path, "wft", 0, tmp_file);

    HANDLE h = CreateFileA(tmp_file, GENERIC_WRITE, 0, NULL,
                            CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Failed to create test file: %s\n", tmp_file);
        return "";
    }

    // Write in 1MB chunks
    std::vector<uint8_t> buf(1024 * 1024);
    size_t written = 0;
    while (written < size) {
        size_t chunk = std::min(buf.size(), size - written);
        for (size_t i = 0; i < chunk; i++) {
            buf[i] = (uint8_t)((written + i) % 251);
        }
        DWORD n = 0;
        WriteFile(h, buf.data(), (DWORD)chunk, &n, NULL);
        written += n;
    }
    CloseHandle(h);
    return std::string(tmp_file);
#else
    char tmp_file[] = "/tmp/wf_test_XXXXXX";
    int fd = mkstemp(tmp_file);
    if (fd < 0) {
        fprintf(stderr, "Failed to create test file\n");
        return "";
    }

    std::vector<uint8_t> buf(1024 * 1024);
    size_t written = 0;
    while (written < size) {
        size_t chunk = std::min(buf.size(), size - written);
        for (size_t i = 0; i < chunk; i++) {
            buf[i] = (uint8_t)((written + i) % 251);
        }
        ssize_t n = write(fd, buf.data(), chunk);
        written += (n > 0) ? (size_t)n : 0;
    }
    close(fd);
    return std::string(tmp_file);
#endif
}

static void delete_test_file(const std::string & path) {
#ifdef _WIN32
    DeleteFileA(path.c_str());
#else
    unlink(path.c_str());
#endif
}

// Verify data matches the expected pattern
static bool verify_pattern(const void * data, size_t offset, size_t nbytes) {
    const uint8_t * p = (const uint8_t *)data;
    for (size_t i = 0; i < nbytes; i++) {
        uint8_t expected = (uint8_t)((offset + i) % 251);
        if (p[i] != expected) {
            fprintf(stderr, "  Data mismatch at offset %zu: got 0x%02x expected 0x%02x\n",
                    offset + i, p[i], expected);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Test 1: Create and destroy (lifecycle)
// ---------------------------------------------------------------------------

static bool test_create_destroy() {
    fprintf(stderr, "test_create_destroy...\n");

    std::string test_file = create_test_file(4 * 1024 * 1024);  // 4 MB
    TEST_ASSERT(!test_file.empty(), "create test file");

    wf_params p = wf_params_default();
    p.model_path      = test_file.c_str();
    p.n_layers        = 4;
    p.max_layer_bytes = 1024 * 1024;  // 1 MB per layer
    p.mode            = WF_MODE_MMAP; // use mmap mode for CPU-only test
    p.io_threads      = 2;
    p.trace           = g_trace;

    wf_context * wf = wf_create(&p);
    TEST_ASSERT(wf != nullptr, "wf_create succeeded");

    wf_transfer_mode mode = wf_get_effective_mode(wf);
    TEST_ASSERT(mode == WF_MODE_MMAP, "effective mode is MMAP");

    wf_destroy(wf);
    delete_test_file(test_file);

    fprintf(stderr, "  PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// Test 2: Register layers and verify span offsets
// ---------------------------------------------------------------------------

static bool test_register_layers() {
    fprintf(stderr, "test_register_layers...\n");

    std::string test_file = create_test_file(4 * 1024 * 1024);
    TEST_ASSERT(!test_file.empty(), "create test file");

    wf_params p = wf_params_default();
    p.model_path      = test_file.c_str();
    p.n_layers        = 4;
    p.max_layer_bytes = 1024 * 1024;
    p.mode            = WF_MODE_MMAP;
    p.io_threads      = 2;
    p.trace           = g_trace;

    wf_context * wf = wf_create(&p);
    TEST_ASSERT(wf != nullptr, "wf_create");

    // Register 4 layers, each with 2 spans
    for (int il = 0; il < 4; il++) {
        wf_weight_span spans[2];
        memset(spans, 0, sizeof(spans));

        spans[0].tensor_name = "weight_a";
        spans[0].file_offset = il * 1024 * 1024;
        spans[0].nbytes      = 512 * 1024;
        spans[0].file_idx    = 0;

        spans[1].tensor_name = "weight_b";
        spans[1].file_offset = il * 1024 * 1024 + 512 * 1024;
        spans[1].nbytes      = 512 * 1024;
        spans[1].file_idx    = 0;

        wf_register_layer(wf, il, spans, 2, 1024 * 1024);
    }

    // Verify span offsets
    TEST_ASSERT(wf_get_span_gpu_offset(wf, 0, 0) == 0, "span 0 offset is 0");
    TEST_ASSERT(wf_get_span_gpu_offset(wf, 0, 1) == 512 * 1024, "span 1 offset is 512K");

    wf_destroy(wf);
    delete_test_file(test_file);

    fprintf(stderr, "  PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// Test 3: mmap fallback path (no direct I/O, uses existing pointers)
// ---------------------------------------------------------------------------

static bool test_mmap_fallback() {
    fprintf(stderr, "test_mmap_fallback...\n");

    // Create test data in heap (simulating mmap'd model weights)
    const int n_layers = 8;
    const size_t layer_size = 256 * 1024;  // 256 KB per layer
    std::vector<uint8_t> fake_weights(n_layers * layer_size);
    for (size_t i = 0; i < fake_weights.size(); i++) {
        fake_weights[i] = (uint8_t)(i % 251);
    }

    std::string test_file = create_test_file(1024);  // minimal file (not actually read)
    TEST_ASSERT(!test_file.empty(), "create test file");

    wf_params p = wf_params_default();
    p.model_path      = test_file.c_str();
    p.n_layers        = n_layers;
    p.max_layer_bytes = layer_size;
    p.mode            = WF_MODE_MMAP;
    p.io_threads      = 2;
    p.n_gpu_buffers   = 2;
    p.n_ram_buffers   = 2;
    p.prefetch_depth  = 3;
    p.trace           = g_trace;

    wf_context * wf = wf_create(&p);
    TEST_ASSERT(wf != nullptr, "wf_create");

    // Register layers with mmap_addr pointers into our fake weights
    for (int il = 0; il < n_layers; il++) {
        wf_weight_span span;
        memset(&span, 0, sizeof(span));
        span.tensor_name = "weights";
        span.file_offset = 0;  // not used for mmap mode
        span.nbytes      = layer_size;
        span.mmap_addr   = fake_weights.data() + il * layer_size;
        span.file_idx    = 0;

        wf_register_layer(wf, il, &span, 1, layer_size);
    }

    // Start pipeline
    wf_start(wf);

    // Request and wait for each layer
    for (int il = 0; il < n_layers; il++) {
        void * gpu_ptr = wf_wait_layer_gpu(wf, il);
        // In mmap mode without CUDA, gpu_ptr points to a malloc'd buffer
        // that should contain the copied weight data
        if (gpu_ptr) {
            // Verify the data was copied correctly (mmap fallback = memcpy)
            // Note: without CUDA, the "GPU" buffers are malloc'd, so we can read them
            bool ok = (memcmp(gpu_ptr, fake_weights.data() + il * layer_size, layer_size) == 0);
            if (!ok) {
                fprintf(stderr, "  Layer %d: data mismatch in GPU buffer\n", il);
            }
            TEST_ASSERT(ok, "layer data matches after mmap fetch");
        }
        wf_release_layer(wf, il);
    }

    // Check stats
    wf_stats stats;
    wf_get_stats(wf, &stats);
    TEST_ASSERT(stats.layers_fetched == n_layers, "all layers fetched");
    TEST_ASSERT(stats.stall_count >= 0, "stall count is valid");  // may or may not stall

    if (g_trace) {
        wf_print_stats(wf);
    }

    wf_stop(wf);
    wf_destroy(wf);
    delete_test_file(test_file);

    fprintf(stderr, "  PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// Test 4: Direct I/O path (reads from file via IOCP/pread)
// ---------------------------------------------------------------------------

static bool test_direct_io() {
    fprintf(stderr, "test_direct_io...\n");

    const int n_layers = 4;
    const size_t layer_size = 512 * 1024;  // 512 KB per layer
    const size_t file_size = n_layers * layer_size;

    std::string test_file = create_test_file(file_size);
    TEST_ASSERT(!test_file.empty(), "create test file");

    wf_params p = wf_params_default();
    p.model_path      = test_file.c_str();
    p.n_layers        = n_layers;
    p.max_layer_bytes = layer_size;
    p.mode            = WF_MODE_DIRECT_IO;
    p.io_direct       = false;  // don't use O_DIRECT for test (alignment issues)
    p.io_block_size   = 64 * 1024;  // 64 KiB blocks for testing
    p.io_threads      = 4;
    p.n_gpu_buffers   = 2;
    p.n_ram_buffers   = 2;
    p.prefetch_depth  = 2;
    p.trace           = g_trace;

    wf_context * wf = wf_create(&p);
    TEST_ASSERT(wf != nullptr, "wf_create");

    // Register layers as contiguous file regions
    for (int il = 0; il < n_layers; il++) {
        wf_weight_span span;
        memset(&span, 0, sizeof(span));
        span.tensor_name = "weights";
        span.file_offset = il * layer_size;
        span.nbytes      = layer_size;
        span.mmap_addr   = nullptr;  // force file read path
        span.file_idx    = 0;

        wf_register_layer(wf, il, &span, 1, layer_size);
    }

    // Start pipeline
    wf_start(wf);

    // Fetch and verify each layer
    for (int il = 0; il < n_layers; il++) {
        void * gpu_ptr = wf_wait_layer_gpu(wf, il);
        TEST_ASSERT(gpu_ptr != nullptr, "got GPU pointer");

        // Without CUDA, "GPU" buffers are heap memory — we can verify contents
        bool ok = verify_pattern(gpu_ptr, il * layer_size, layer_size);
        TEST_ASSERT(ok, "file data matches expected pattern");

        wf_release_layer(wf, il);
    }

    wf_stats stats;
    wf_get_stats(wf, &stats);
    TEST_ASSERT(stats.layers_fetched == n_layers, "all layers fetched");
    TEST_ASSERT(stats.io_ops > 0, "I/O operations were performed");
    TEST_ASSERT(stats.bytes_read_nvme >= file_size, "all bytes read from file");

    if (g_trace) {
        wf_print_stats(wf);
    }

    fprintf(stderr, "  I/O stats: %d ops, %.2f MB read, %.2f GB/s\n",
            stats.io_ops,
            stats.bytes_read_nvme / (1024.0 * 1024.0),
            stats.nvme_read_bw_gbps);

    wf_stop(wf);
    wf_destroy(wf);
    delete_test_file(test_file);

    fprintf(stderr, "  PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// Test 5: MoE expert filtering
// ---------------------------------------------------------------------------

static bool test_moe_filtering() {
    fprintf(stderr, "test_moe_filtering...\n");

    const size_t expert_size = 64 * 1024;
    const int n_experts = 8;
    const size_t shared_size = 64 * 1024;
    const size_t layer_size = shared_size + n_experts * expert_size;

    // Create fake weights
    std::vector<uint8_t> fake_weights(layer_size);
    for (size_t i = 0; i < fake_weights.size(); i++) {
        fake_weights[i] = (uint8_t)(i % 251);
    }

    std::string test_file = create_test_file(1024);
    TEST_ASSERT(!test_file.empty(), "create test file");

    wf_params p = wf_params_default();
    p.model_path      = test_file.c_str();
    p.n_layers        = 1;
    p.max_layer_bytes = layer_size;
    p.mode            = WF_MODE_MMAP;
    p.moe_mode        = WF_MOE_FETCH_ACTIVE;
    p.io_threads      = 2;
    p.trace           = g_trace;

    wf_context * wf = wf_create(&p);
    TEST_ASSERT(wf != nullptr, "wf_create");

    // Register: 1 shared span + 8 expert spans
    std::vector<wf_weight_span> spans(1 + n_experts);
    memset(spans.data(), 0, sizeof(wf_weight_span) * spans.size());

    // Shared weights
    spans[0].tensor_name = "ffn_gate";
    spans[0].nbytes      = shared_size;
    spans[0].mmap_addr   = fake_weights.data();
    spans[0].is_expert   = false;

    // Expert weights
    for (int e = 0; e < n_experts; e++) {
        spans[1 + e].tensor_name = "expert";
        spans[1 + e].nbytes      = expert_size;
        spans[1 + e].mmap_addr   = fake_weights.data() + shared_size + e * expert_size;
        spans[1 + e].is_expert   = true;
        spans[1 + e].expert_idx  = e;
    }

    wf_register_layer(wf, 0, spans.data(), (int)spans.size(), layer_size);

    // Set only experts 2 and 5 as active
    int active[] = {2, 5};
    wf_set_active_experts(wf, 0, active, 2);

    // Start and fetch
    wf_start(wf);
    void * gpu_ptr = wf_wait_layer_gpu(wf, 0);
    TEST_ASSERT(gpu_ptr != nullptr, "got GPU pointer");

    // Verify: shared weights should be copied
    bool shared_ok = (memcmp(gpu_ptr, fake_weights.data(), shared_size) == 0);
    TEST_ASSERT(shared_ok, "shared weights copied correctly");

    // Expert 2 should be copied, expert 0 should be zeroed
    uint8_t * expert_0_gpu = (uint8_t *)gpu_ptr + shared_size + 0 * expert_size;
    uint8_t * expert_2_gpu = (uint8_t *)gpu_ptr + shared_size + 2 * expert_size;

    // Expert 0 was filtered — should be all zeros
    bool e0_zeroed = true;
    for (size_t i = 0; i < expert_size; i++) {
        if (expert_0_gpu[i] != 0) { e0_zeroed = false; break; }
    }
    TEST_ASSERT(e0_zeroed, "inactive expert 0 is zeroed");

    // Expert 2 was active — should have correct data
    bool e2_ok = (memcmp(expert_2_gpu, fake_weights.data() + shared_size + 2 * expert_size, expert_size) == 0);
    TEST_ASSERT(e2_ok, "active expert 2 has correct data");

    // Check stats
    wf_stats stats;
    wf_get_stats(wf, &stats);
    TEST_ASSERT(stats.experts_skipped == 6, "6 experts skipped");
    TEST_ASSERT(stats.expert_bytes_saved == 6 * expert_size, "correct bytes saved");

    wf_release_layer(wf, 0);
    wf_stop(wf);
    wf_destroy(wf);
    delete_test_file(test_file);

    fprintf(stderr, "  PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// Test 6: Pipeline throughput (measures sustained bandwidth)
// ---------------------------------------------------------------------------

static bool test_pipeline_throughput() {
    fprintf(stderr, "test_pipeline_throughput...\n");

    const int n_layers = 32;
    const size_t layer_size = 2 * 1024 * 1024;  // 2 MB per layer (small for testing)
    const size_t file_size = n_layers * layer_size;

    std::string test_file = create_test_file(file_size);
    TEST_ASSERT(!test_file.empty(), "create test file");

    wf_params p = wf_params_default();
    p.model_path      = test_file.c_str();
    p.n_layers        = n_layers;
    p.max_layer_bytes = layer_size;
    p.mode            = WF_MODE_DIRECT_IO;
    p.io_direct       = false;
    p.io_block_size   = 512 * 1024;
    p.io_threads      = 8;
    p.io_queue_depth  = 4;
    p.n_gpu_buffers   = 2;
    p.n_ram_buffers   = 4;  // 1 dispatch, 1 ready, 1 receiving, 1 spare
    p.prefetch_depth  = 4;
    p.trace           = g_trace;

    wf_context * wf = wf_create(&p);
    TEST_ASSERT(wf != nullptr, "wf_create");

    // Register layers
    for (int il = 0; il < n_layers; il++) {
        wf_weight_span span;
        memset(&span, 0, sizeof(span));
        span.tensor_name = "weights";
        span.file_offset = il * layer_size;
        span.nbytes      = layer_size;
        span.file_idx    = 0;

        wf_register_layer(wf, il, &span, 1, layer_size);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    wf_start(wf);

    // Simulate compute: fetch each layer, "compute" for 1ms, release
    for (int il = 0; il < n_layers; il++) {
        void * ptr = wf_wait_layer_gpu(wf, il);
        TEST_ASSERT(ptr != nullptr, "got layer pointer");

        // Simulate GPU compute time
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        wf_release_layer(wf, il);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    float total_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    float bw_gbps = (file_size / (total_ms / 1000.0f)) / (1024.0f * 1024.0f * 1024.0f);

    wf_stats stats;
    wf_get_stats(wf, &stats);

    fprintf(stderr, "  %d layers, %.2f MB total, %.2fms wall time\n",
            n_layers, file_size / (1024.0 * 1024.0), total_ms);
    fprintf(stderr, "  Effective pipeline: %.2f GB/s\n", bw_gbps);
    fprintf(stderr, "  NVMe read: %.2f GB/s, H2D: %.2f GB/s\n",
            stats.nvme_read_bw_gbps, stats.ram_to_gpu_bw_gbps);
    fprintf(stderr, "  Stalls: %d (%.2fms total)\n", stats.stall_count, stats.total_stall_ms);

    TEST_ASSERT(stats.layers_fetched == n_layers, "all layers fetched");

    if (g_trace) {
        wf_print_stats(wf);
    }

    wf_stop(wf);
    wf_destroy(wf);
    delete_test_file(test_file);

    fprintf(stderr, "  PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// Test 7: Pass reset (KEEP_FETCHER mode)
// ---------------------------------------------------------------------------

static bool test_pass_reset() {
    fprintf(stderr, "test_pass_reset...\n");

    const int n_layers = 4;
    const size_t layer_size = 128 * 1024;

    std::vector<uint8_t> fake_weights(n_layers * layer_size);
    for (size_t i = 0; i < fake_weights.size(); i++) {
        fake_weights[i] = (uint8_t)(i % 251);
    }

    std::string test_file = create_test_file(1024);
    TEST_ASSERT(!test_file.empty(), "create test file");

    wf_params p = wf_params_default();
    p.model_path      = test_file.c_str();
    p.n_layers        = n_layers;
    p.max_layer_bytes = layer_size;
    p.mode            = WF_MODE_MMAP;
    p.post_mode       = WF_POST_KEEP_FETCHER;
    p.io_threads      = 2;
    p.trace           = g_trace;

    wf_context * wf = wf_create(&p);
    TEST_ASSERT(wf != nullptr, "wf_create");

    for (int il = 0; il < n_layers; il++) {
        wf_weight_span span;
        memset(&span, 0, sizeof(span));
        span.tensor_name = "w";
        span.nbytes      = layer_size;
        span.mmap_addr   = fake_weights.data() + il * layer_size;

        wf_register_layer(wf, il, &span, 1, layer_size);
    }

    // Pass 1
    wf_start(wf);
    for (int il = 0; il < n_layers; il++) {
        void * ptr = wf_wait_layer_gpu(wf, il);
        TEST_ASSERT(ptr != nullptr, "pass 1: got layer");
        wf_release_layer(wf, il);
    }
    wf_end_pass(wf);  // should reset, not destroy

    // Pass 2 — should work identically
    wf_start(wf);
    for (int il = 0; il < n_layers; il++) {
        void * ptr = wf_wait_layer_gpu(wf, il);
        TEST_ASSERT(ptr != nullptr, "pass 2: got layer");
        wf_release_layer(wf, il);
    }

    wf_stats stats;
    wf_get_stats(wf, &stats);
    TEST_ASSERT(stats.layers_fetched == 2 * n_layers, "fetched layers across 2 passes");

    wf_stop(wf);
    wf_destroy(wf);
    delete_test_file(test_file);

    fprintf(stderr, "  PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// Test 8: Real GGUF weight fetch (reads actual model weights from file)
//
// Reads chunks from a real GGUF model file (--model path) through the
// fetcher pipeline and verifies the data matches a direct fread() read.
// This tests the full async I/O path with real file sizes and alignments.
// ---------------------------------------------------------------------------

static bool test_real_gguf(const char * model_path) {
    fprintf(stderr, "test_real_gguf (%s)...\n", model_path);

    // Open the file to measure size and read reference data
    FILE * f = fopen(model_path, "rb");
    TEST_ASSERT(f != nullptr, "open model file");

#ifdef _WIN32
    _fseeki64(f, 0, SEEK_END);
    size_t file_size = (size_t)_ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
#else
    fseek(f, 0, SEEK_END);
    size_t file_size = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
#endif

    fprintf(stderr, "  File size: %.2f GB\n", file_size / (1024.0 * 1024.0 * 1024.0));
    TEST_ASSERT(file_size > 1024 * 1024, "file is larger than 1MB");

    // We'll simulate 64 layers, reading equal-sized chunks from the file.
    // Skip the first 16MB (GGUF header/metadata) to start at weight data.
    const size_t data_offset = 16 * 1024 * 1024;
    const int n_layers = 64;
    const size_t usable_size = file_size - data_offset;
    const size_t layer_size = (usable_size / n_layers) & ~((size_t)4095);  // align to 4K
    const size_t total_read = n_layers * layer_size;

    fprintf(stderr, "  Simulating %d layers, %.2f MB each (%.2f GB total)\n",
            n_layers, layer_size / (1024.0 * 1024.0), total_read / (1024.0 * 1024.0 * 1024.0));

    // Read reference data for first and last layer using fread
    std::vector<uint8_t> ref_first(layer_size);
    std::vector<uint8_t> ref_last(layer_size);

#ifdef _WIN32
    _fseeki64(f, (__int64)data_offset, SEEK_SET);
#else
    fseek(f, (long)data_offset, SEEK_SET);
#endif
    size_t n1 = fread(ref_first.data(), 1, layer_size, f);
    TEST_ASSERT(n1 == layer_size, "read first layer reference");

#ifdef _WIN32
    _fseeki64(f, (__int64)(data_offset + (n_layers - 1) * layer_size), SEEK_SET);
#else
    fseek(f, (long)(data_offset + (n_layers - 1) * layer_size), SEEK_SET);
#endif
    size_t n2 = fread(ref_last.data(), 1, layer_size, f);
    TEST_ASSERT(n2 == layer_size, "read last layer reference");
    fclose(f);

    // Create weight fetcher
    wf_params p = wf_params_default();
    p.model_path      = model_path;
    p.n_layers        = n_layers;
    p.max_layer_bytes = layer_size;
    p.mode            = WF_MODE_DIRECT_IO;
    p.io_direct       = true;   // FILE_FLAG_NO_BUFFERING — required for full RAID bandwidth
    p.io_block_size   = 512 * 1024;  // 512 KiB blocks
    p.io_threads      = 8;
    p.io_queue_depth  = 4;
    p.n_gpu_buffers   = 2;
    p.n_ram_buffers   = 4;  // 1 dispatch, 1 ready, 1 receiving, 1 spare
    p.prefetch_depth  = 4;
    p.trace           = g_trace;

    wf_context * wf = wf_create(&p);
    TEST_ASSERT(wf != nullptr, "wf_create");

    // Register layers as contiguous file regions starting after the header
    for (int il = 0; il < n_layers; il++) {
        wf_weight_span span;
        memset(&span, 0, sizeof(span));
        span.tensor_name = "layer_weights";
        span.file_offset = data_offset + il * layer_size;
        span.nbytes      = layer_size;
        span.file_idx    = 0;

        wf_register_layer(wf, il, &span, 1, layer_size);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    wf_start(wf);

    // Fetch all layers, verify first and last
    for (int il = 0; il < n_layers; il++) {
        void * ptr = wf_wait_layer_gpu(wf, il);
        TEST_ASSERT(ptr != nullptr, "got layer pointer");

        if (il == 0) {
            bool ok = (memcmp(ptr, ref_first.data(), layer_size) == 0);
            if (!ok) {
                fprintf(stderr, "  Layer 0 data mismatch! First bytes: got %02x %02x %02x, expected %02x %02x %02x\n",
                        ((uint8_t *)ptr)[0], ((uint8_t *)ptr)[1], ((uint8_t *)ptr)[2],
                        ref_first[0], ref_first[1], ref_first[2]);
            }
            TEST_ASSERT(ok, "first layer data matches fread reference");
        }
        if (il == n_layers - 1) {
            bool ok = (memcmp(ptr, ref_last.data(), layer_size) == 0);
            TEST_ASSERT(ok, "last layer data matches fread reference");
        }

        wf_release_layer(wf, il);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    float total_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    float bw_gbps = (total_read / (total_ms / 1000.0f)) / (1024.0f * 1024.0f * 1024.0f);

    wf_stats stats;
    wf_get_stats(wf, &stats);

    fprintf(stderr, "  %d layers, %.2f GB, %.2fms wall time\n",
            n_layers, total_read / (1024.0 * 1024.0 * 1024.0), total_ms);
    fprintf(stderr, "  Pipeline throughput:  %.2f GB/s\n", bw_gbps);
    fprintf(stderr, "  NVMe read bandwidth:  %.2f GB/s (%d I/O ops)\n",
            stats.nvme_read_bw_gbps, stats.io_ops);
    fprintf(stderr, "  GPU stalls: %d (%.2fms total)\n", stats.stall_count, stats.total_stall_ms);

    TEST_ASSERT(stats.layers_fetched == n_layers, "all layers fetched");

    wf_stop(wf);
    wf_destroy(wf);

    fprintf(stderr, "  PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// Test 9: Split GGUF — reads all split files for full-model bandwidth test
//
// Discovers all split files from a -00001-of-NNNNN pattern, reads the
// entire dataset through the weight fetcher pipeline, measures throughput.
// ---------------------------------------------------------------------------

// Discover split files from a path like "model-00001-of-00022.gguf"
// Returns all discovered paths and fills n_total from the filename pattern.
static std::vector<std::string> discover_split_files(const char * first_path) {
    std::vector<std::string> paths;
    std::string base(first_path);

    // Find the "-00001-of-" pattern
    size_t pos = base.find("-00001-of-");
    if (pos == std::string::npos) {
        // Not a split file — return single file
        paths.push_back(base);
        return paths;
    }

    // Extract total count from "of-NNNNN"
    size_t of_pos = pos + strlen("-00001-of-");
    std::string count_str;
    size_t ci = of_pos;
    while (ci < base.size() && base[ci] >= '0' && base[ci] <= '9') {
        count_str += base[ci++];
    }
    int n_total = std::stoi(count_str);
    std::string suffix = base.substr(ci);  // e.g. ".gguf"
    std::string prefix = base.substr(0, pos + 1); // up to and including the "-"

    int width = (int)count_str.size(); // zero-padding width

    for (int i = 1; i <= n_total; i++) {
        char num_buf[32];
        snprintf(num_buf, sizeof(num_buf), "%0*d", width, i);
        char total_buf[32];
        snprintf(total_buf, sizeof(total_buf), "%0*d", width, n_total);
        std::string path = prefix + std::string(num_buf) + "-of-" + std::string(total_buf) + suffix;
        paths.push_back(path);
    }
    return paths;
}

static bool test_real_gguf_split(const char * first_path) {
    fprintf(stderr, "test_real_gguf_split (%s)...\n", first_path);

    // Discover all split files
    auto split_paths = discover_split_files(first_path);
    int n_splits = (int)split_paths.size();
    fprintf(stderr, "  Discovered %d split files\n", n_splits);

    if (n_splits <= 1) {
        fprintf(stderr, "  Not a split model, skipping (use test_real_gguf instead)\n");
        return true;
    }

    // Get file sizes
    struct split_info { std::string path; size_t size; };
    std::vector<split_info> splits;
    size_t total_file_size = 0;

    for (int i = 0; i < n_splits; i++) {
        FILE * f = fopen(split_paths[i].c_str(), "rb");
        if (!f) {
            fprintf(stderr, "  Cannot open split %d: %s\n", i + 1, split_paths[i].c_str());
            TEST_ASSERT(f != nullptr, "open split file");
        }
#ifdef _WIN32
        _fseeki64(f, 0, SEEK_END);
        size_t sz = (size_t)_ftelli64(f);
#else
        fseek(f, 0, SEEK_END);
        size_t sz = (size_t)ftell(f);
#endif
        fclose(f);
        splits.push_back({split_paths[i], sz});
        total_file_size += sz;
    }
    fprintf(stderr, "  Total: %.2f GB across %d files\n",
            total_file_size / (1024.0 * 1024.0 * 1024.0), n_splits);

    // Read each file sequentially through its own wf_context.
    // Uses the proven single-file DiskSpd path (8 handles, QD=4, contiguous ranges)
    // that already showed 11 GB/s. Create → register → stream → destroy for each file.
    const size_t data_offset = 16 * 1024 * 1024;  // skip GGUF header
    const int n_layers_per_file = g_nlayers;        // tunable via --nlayers

    auto t_global = std::chrono::high_resolution_clock::now();
    size_t total_bytes_read = 0;

    for (int fi = 0; fi < n_splits; fi++) {
        size_t skip = (fi == 0) ? data_offset : 0;
        size_t usable = splits[fi].size > skip ? splits[fi].size - skip : 0;
        if (usable < 1024 * 1024) continue;  // skip tiny files

        int n_layers = std::min(n_layers_per_file, (int)(usable / (1024 * 1024)));
        if (n_layers < 1) n_layers = 1;
        size_t layer_size = (usable / n_layers) & ~((size_t)4095);
        size_t file_read = (size_t)n_layers * layer_size;

        fprintf(stderr, "  [File %d/%d] %.2f GB, %d layers × %.2f MB — %s\n",
                fi + 1, n_splits,
                splits[fi].size / (1024.0 * 1024.0 * 1024.0),
                n_layers, layer_size / (1024.0 * 1024.0),
                splits[fi].path.c_str());

        // Create fetcher for this file
        wf_params p = wf_params_default();
        p.model_path      = splits[fi].path.c_str();
        p.n_layers        = n_layers;
        p.max_layer_bytes = layer_size;
        p.mode            = WF_MODE_DIRECT_IO;
        p.io_direct       = true;
        p.io_block_size   = g_block_size_kb * 1024;
        p.io_threads      = g_threads;
        p.io_queue_depth  = g_queue_depth;
        p.n_gpu_buffers   = g_gpu_bufs;
        p.n_ram_buffers   = g_ram_bufs;
        p.prefetch_depth  = g_ram_bufs;  // match RAM buffer count
        p.trace           = false;  // quiet per-file

        wf_context * wf = wf_create(&p);
        if (!wf) {
            fprintf(stderr, "  wf_create failed for file %d\n", fi);
            continue;
        }

        // Register layers
        for (int il = 0; il < n_layers; il++) {
            wf_weight_span span;
            memset(&span, 0, sizeof(span));
            span.tensor_name = "layer_weights";
            span.file_offset = skip + il * layer_size;
            span.nbytes      = layer_size;
            span.file_idx    = 0;  // always file 0 (this fetcher only has one file)

            wf_register_layer(wf, il, &span, 1, layer_size);
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        wf_start(wf);

        for (int il = 0; il < n_layers; il++) {
            void * ptr = wf_wait_layer_gpu(wf, il);
            TEST_ASSERT(ptr != nullptr, "got layer pointer");
            wf_release_layer(wf, il);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        float file_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        float file_bw = (file_read / (file_ms / 1000.0f)) / (1024.0f * 1024.0f * 1024.0f);

        total_bytes_read += file_read;
        float elapsed_s = std::chrono::duration<float>(t1 - t_global).count();
        float cumul_bw = (total_bytes_read / elapsed_s) / (1024.0f * 1024.0f * 1024.0f);

        fprintf(stderr, "         %.2f GB in %.2fs = %.2f GB/s (cumulative: %.2f GB in %.1fs = %.2f GB/s)\n",
                file_read / (1024.0 * 1024.0 * 1024.0), file_ms / 1000.0f, file_bw,
                total_bytes_read / (1024.0 * 1024.0 * 1024.0), elapsed_s, cumul_bw);

        wf_stop(wf);
        wf_print_stats(wf);
        wf_destroy(wf);

        // Brief pause between files — ensure NVMe driver fully drains
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    float total_s = std::chrono::duration<float>(t_end - t_global).count();
    float overall_bw = (total_bytes_read / total_s) / (1024.0f * 1024.0f * 1024.0f);

    fprintf(stderr, "\n");
    fprintf(stderr, "  ====== SPLIT MODEL RESULTS ======\n");
    fprintf(stderr, "  Files:              %d\n", n_splits);
    fprintf(stderr, "  Total data:         %.2f GB\n", total_bytes_read / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "  Wall time:          %.2f s\n", total_s);
    fprintf(stderr, "  Overall BW:         %.2f GB/s\n", overall_bw);
    fprintf(stderr, "  ================================\n");

    fprintf(stderr, "  PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// Raw I/O benchmark — DiskSpd/CrystalDiskMark-equivalent architecture.
// N independent threads, each with its own file handle AND its own IOCP.
// Each thread reads a contiguous region of the file, maintaining steady QD.
// Zero cross-thread contention — matches DiskSpd exactly.
// ---------------------------------------------------------------------------

#ifdef _WIN32

// Interlocked sequential I/O — shared state across all threads (DiskSpd -si)
struct raw_io_shared {
    std::atomic<uint64_t> next_block{0};  // shared counter — all threads grab from here
    uint64_t              total_blocks;
    size_t                bs;
    const char *          path;
    int                   qd;
    std::atomic<uint64_t> total_bytes{0};
    std::atomic<uint64_t> total_ops{0};
};

static void raw_io_thread_si(raw_io_shared * shared, int thread_id) {
    HANDLE hFile = CreateFileA(shared->path, GENERIC_READ, FILE_SHARE_READ, NULL,
                                OPEN_EXISTING,
                                FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
                                NULL);
    if (hFile == INVALID_HANDLE_VALUE) return;

    HANDLE iocp = CreateIoCompletionPort(hFile, NULL, 0, 1);
    if (!iocp) { CloseHandle(hFile); return; }

    int qd = shared->qd;
    size_t bs = shared->bs;

    void * buf = _aligned_malloc((size_t)qd * bs, 4096);
    if (!buf) { CloseHandle(hFile); CloseHandle(iocp); return; }

    std::vector<OVERLAPPED> ovs(qd);

    // Submit initial QD reads — grab blocks from shared counter
    int submitted = 0;
    for (int i = 0; i < qd; i++) {
        uint64_t blk = shared->next_block.fetch_add(1, std::memory_order_relaxed);
        if (blk >= shared->total_blocks) break;
        memset(&ovs[i], 0, sizeof(OVERLAPPED));
        uint64_t off = blk * bs;
        ovs[i].Offset     = (DWORD)(off & 0xFFFFFFFF);
        ovs[i].OffsetHigh = (DWORD)(off >> 32);
        ReadFile(hFile, (char *)buf + (size_t)i * bs, (DWORD)bs, NULL, &ovs[i]);
        submitted++;
    }

    uint64_t bytes = 0, ops = 0;
    int outstanding = submitted;

    while (outstanding > 0) {
        DWORD transferred = 0;
        ULONG_PTR key = 0;
        LPOVERLAPPED pov = NULL;
        BOOL ok = GetQueuedCompletionStatus(iocp, &transferred, &key, &pov, 2000);
        if (!ok && !pov) continue;
        if (!pov) break;

        bytes += transferred;
        ops++;
        outstanding--;

        // Re-submit: grab next block from shared counter
        uint64_t blk = shared->next_block.fetch_add(1, std::memory_order_relaxed);
        if (blk < shared->total_blocks) {
            memset(pov, 0, sizeof(OVERLAPPED));
            uint64_t off = blk * bs;
            pov->Offset     = (DWORD)(off & 0xFFFFFFFF);
            pov->OffsetHigh = (DWORD)(off >> 32);
            int si = (int)((char *)pov - (char *)&ovs[0]) / (int)sizeof(OVERLAPPED);
            ReadFile(hFile, (char *)buf + (size_t)si * bs, (DWORD)bs, NULL, pov);
            outstanding++;
        }
    }

    shared->total_bytes.fetch_add(bytes, std::memory_order_relaxed);
    shared->total_ops.fetch_add(ops, std::memory_order_relaxed);

    _aligned_free(buf);
    CloseHandle(hFile);
    CloseHandle(iocp);
}

static bool test_raw_io(const char * path) {
    fprintf(stderr, "test_raw_io: interlocked sequential benchmark (%s)\n", path);
    fprintf(stderr, "  bs=%dK  threads=%d  qd=%d (per thread)\n",
            g_block_size_kb, g_threads, g_queue_depth);

    HANDLE hSize = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hSize == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "  ERROR: cannot open file\n");
        return false;
    }
    LARGE_INTEGER li;
    GetFileSizeEx(hSize, &li);
    uint64_t file_size = (uint64_t)li.QuadPart;
    CloseHandle(hSize);

    const size_t bs = (size_t)g_block_size_kb * 1024;
    uint64_t aligned_size = file_size & ~(bs - 1);

    fprintf(stderr, "  File size: %.2f GB (%.2f GB aligned)\n",
            file_size / (1024.0 * 1024.0 * 1024.0),
            aligned_size / (1024.0 * 1024.0 * 1024.0));

    raw_io_shared shared;
    shared.total_blocks = aligned_size / bs;
    shared.bs   = bs;
    shared.path = path;
    shared.qd   = g_queue_depth;

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    for (int i = 0; i < g_threads; i++) {
        threads.emplace_back(raw_io_thread_si, &shared, i);
    }
    for (auto & t : threads) t.join();
    auto t1 = std::chrono::high_resolution_clock::now();

    uint64_t total_bytes = shared.total_bytes.load();
    uint64_t total_ops   = shared.total_ops.load();
    float elapsed = std::chrono::duration<float>(t1 - t0).count();
    float bw = (total_bytes / elapsed) / (1024.0 * 1024.0 * 1024.0);

    fprintf(stderr, "\n");
    fprintf(stderr, "  ====== RAW I/O RESULTS (interlocked sequential) ======\n");
    fprintf(stderr, "  Threads:      %d (each: own handle + own IOCP)\n", g_threads);
    fprintf(stderr, "  QD/thread:    %d  (total QD: %d)\n", g_queue_depth, g_threads * g_queue_depth);
    fprintf(stderr, "  Block size:   %d KiB\n", g_block_size_kb);
    fprintf(stderr, "  Data read:    %.2f GB\n", total_bytes / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "  Wall time:    %.2f s\n", elapsed);
    fprintf(stderr, "  Bandwidth:    %.2f GB/s\n", bw);
    fprintf(stderr, "  I/O ops:      %llu\n", (unsigned long long)total_ops);
    fprintf(stderr, "  IOPS:         %.0f\n", total_ops / elapsed);
    fprintf(stderr, "  =====================================================\n");

    return true;
}
#endif // _WIN32

// ---------------------------------------------------------------------------
// DirectStorage benchmark — Windows only, requires LLAMA_DIRECTSTORAGE
//
// Uses Microsoft DirectStorage 1.3 with BypassIO on Win11 to read a file
// into system memory.  Compares against the raw IOCP baseline above.
// Activated with: --dstorage <path>
// ---------------------------------------------------------------------------

#if defined(_WIN32) && defined(LLAMA_DIRECTSTORAGE)

#ifndef DSTORAGE_MAX_QUEUE_CAPACITY
#  define DSTORAGE_MAX_QUEUE_CAPACITY 8192
#endif

static bool test_directstorage_io(const char * path) {
    fprintf(stderr, "test_directstorage_io: DirectStorage memory benchmark (%s)\n", path);
    fprintf(stderr, "  bs=%dK\n", g_block_size_kb);

    // COM is required by DirectStorage
    HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    bool com_inited = SUCCEEDED(hr);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
        fprintf(stderr, "  ERROR: CoInitializeEx failed: 0x%08lx\n", (unsigned long)hr);
        return false;
    }

    // Get the DirectStorage factory
    IDStorageFactory * factory = nullptr;
    hr = DStorageGetFactory(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: DStorageGetFactory failed: 0x%08lx\n", (unsigned long)hr);
        fprintf(stderr, "  Ensure dstorage.dll + dstoragecore.dll are beside the test binary.\n");
        if (com_inited) CoUninitialize();
        return false;
    }

    // Convert path to wide string for DirectStorage API
    int wlen = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);
    std::vector<wchar_t> wpath(wlen);
    MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath.data(), wlen);

    // Open the file
    IDStorageFile * dsFile = nullptr;
    hr = factory->OpenFile(wpath.data(), IID_PPV_ARGS(&dsFile));
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: OpenFile failed: 0x%08lx\n", (unsigned long)hr);
        factory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }

    // Get file size
    BY_HANDLE_FILE_INFORMATION finfo;
    hr = dsFile->GetFileInformation(&finfo);
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: GetFileInformation failed: 0x%08lx\n", (unsigned long)hr);
        dsFile->Release();
        factory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }
    uint64_t file_size = ((uint64_t)finfo.nFileSizeHigh << 32) | finfo.nFileSizeLow;

    const size_t bs = (size_t)g_block_size_kb * 1024;
    uint64_t aligned_size = file_size & ~((uint64_t)bs - 1);
    uint64_t total_blocks = aligned_size / bs;

    fprintf(stderr, "  File size: %.2f GB (%.2f GB aligned to %dK blocks)\n",
            file_size   / (1024.0 * 1024.0 * 1024.0),
            aligned_size / (1024.0 * 1024.0 * 1024.0),
            g_block_size_kb);

    // Buffer pool — cap at 256 MB of memory
    const size_t max_buf_bytes = 256ULL * 1024 * 1024;
    int n_buf_slots = (int)(max_buf_bytes / bs);
    if (n_buf_slots > (int)DSTORAGE_MAX_QUEUE_CAPACITY) n_buf_slots = (int)DSTORAGE_MAX_QUEUE_CAPACITY;
    if (n_buf_slots > (int)total_blocks) n_buf_slots = (int)total_blocks;
    if (n_buf_slots < 1) n_buf_slots = 1;

    // Queue capacity must be power of 2
    int queue_cap = 1;
    while (queue_cap * 2 <= n_buf_slots) queue_cap *= 2;

    fprintf(stderr, "  Queue capacity: %d  Buffer pool: %.1f MB\n",
            queue_cap, ((double)queue_cap * bs) / (1024.0 * 1024.0));

    // Create queue — memory destination (no D3D12 device needed)
    DSTORAGE_QUEUE_DESC queueDesc = {};
    queueDesc.Capacity   = (UINT16)queue_cap;
    queueDesc.Priority   = DSTORAGE_PRIORITY_HIGH;
    queueDesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    queueDesc.Device     = nullptr;  // memory destination

    IDStorageQueue * queue = nullptr;
    hr = factory->CreateQueue(&queueDesc, IID_PPV_ARGS(&queue));
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: CreateQueue failed: 0x%08lx\n", (unsigned long)hr);
        dsFile->Release();
        factory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }

    // Try to get IDStorageQueue1 for EnqueueSetEvent (DS 1.1+)
    IDStorageQueue1 * queue1 = nullptr;
    queue->QueryInterface(IID_PPV_ARGS(&queue1));

    // If no Queue1, use IDStorageStatusArray for polling
    IDStorageStatusArray * statusArray = nullptr;
    HANDLE completion_event = NULL;
    if (queue1) {
        completion_event = CreateEvent(NULL, FALSE, FALSE, NULL);
        fprintf(stderr, "  Sync: IDStorageQueue1::EnqueueSetEvent (Win32 event)\n");
    } else {
        hr = factory->CreateStatusArray(1, "bench", IID_PPV_ARGS(&statusArray));
        if (FAILED(hr)) {
            fprintf(stderr, "  ERROR: CreateStatusArray failed: 0x%08lx\n", (unsigned long)hr);
            queue->Release();
            dsFile->Release();
            factory->Release();
            if (com_inited) CoUninitialize();
            return false;
        }
        fprintf(stderr, "  Sync: IDStorageStatusArray polling (IDStorageQueue1 not available)\n");
    }

    // Allocate 4K-aligned read buffer (required for BypassIO / direct I/O)
    void * buf = _aligned_malloc((size_t)queue_cap * bs, 4096);
    if (!buf) {
        fprintf(stderr, "  ERROR: buffer alloc failed (%.1f MB)\n",
                ((double)queue_cap * bs) / (1024.0 * 1024.0));
        if (statusArray) statusArray->Release();
        if (queue1) queue1->Release();
        if (completion_event) CloseHandle(completion_event);
        queue->Release();
        dsFile->Release();
        factory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }

    fprintf(stderr, "  Submitting %llu blocks in batches of %d...\n",
            (unsigned long long)total_blocks, queue_cap);

    auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t submitted = 0;

    while (submitted < total_blocks) {
        int this_batch = (int)std::min((uint64_t)queue_cap, total_blocks - submitted);

        for (int i = 0; i < this_batch; i++) {
            DSTORAGE_REQUEST req = {};
            req.Options.SourceType        = DSTORAGE_REQUEST_SOURCE_FILE;
            req.Options.DestinationType   = DSTORAGE_REQUEST_DESTINATION_MEMORY;
            req.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
            req.Source.File.Source         = dsFile;
            req.Source.File.Offset        = (submitted + i) * bs;
            req.Source.File.Size          = (UINT32)bs;
            req.Destination.Memory.Buffer = (char *)buf + (size_t)i * bs;
            req.Destination.Memory.Size   = (UINT32)bs;
            req.UncompressedSize          = (UINT32)bs;

            queue->EnqueueRequest(&req);
        }

        if (queue1) {
            queue1->EnqueueSetEvent(completion_event);
            queue->Submit();
            WaitForSingleObject(completion_event, INFINITE);
        } else {
            queue->EnqueueStatus(statusArray, 0);
            queue->Submit();
            while (!statusArray->IsComplete(0)) {
                YieldProcessor();
            }
        }

        submitted += this_batch;
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    // Check for errors
    DSTORAGE_ERROR_RECORD ds_err = {};
    queue->RetrieveErrorRecord(&ds_err);
    if (FAILED(ds_err.FirstFailure.HResult)) {
        fprintf(stderr, "  WARNING: DirectStorage error: 0x%08lx\n",
                (unsigned long)ds_err.FirstFailure.HResult);
    }

    uint64_t total_bytes = submitted * bs;
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double bw = ((double)total_bytes / elapsed) / (1024.0 * 1024.0 * 1024.0);

    fprintf(stderr, "\n");
    fprintf(stderr, "  ====== DIRECTSTORAGE RESULTS (memory destination, BypassIO) ======\n");
    fprintf(stderr, "  Block size:   %d KiB\n", g_block_size_kb);
    fprintf(stderr, "  Queue cap:    %d\n", queue_cap);
    fprintf(stderr, "  Buffer:       %.1f MB\n", ((double)queue_cap * bs) / (1024.0 * 1024.0));
    fprintf(stderr, "  Data read:    %.2f GB\n", total_bytes / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "  Wall time:    %.2f s\n", elapsed);
    fprintf(stderr, "  Bandwidth:    %.2f GB/s\n", bw);
    fprintf(stderr, "  Requests:     %llu\n", (unsigned long long)submitted);
    fprintf(stderr, "  IOPS:         %.0f\n", submitted / elapsed);
    fprintf(stderr, "  ================================================================\n");

    // Cleanup
    if (completion_event) CloseHandle(completion_event);
    if (statusArray) statusArray->Release();
    if (queue1) queue1->Release();
    _aligned_free(buf);
    queue->Release();
    dsFile->Release();
    factory->Release();
    if (com_inited) CoUninitialize();

    return true;
}

// ---------------------------------------------------------------------------
// DirectStorage GPU-direct benchmark — NVMe → GPU VRAM (zero RAM touches)
//
// Creates a D3D12 device, allocates a DEFAULT-heap buffer in GPU VRAM,
// and uses DirectStorage with DSTORAGE_REQUEST_DESTINATION_BUFFER to DMA
// file data directly from NVMe to GPU memory, bypassing system RAM entirely.
// Activated with: --dstorage-gpu <path>
// ---------------------------------------------------------------------------

static bool test_directstorage_gpu(const char * path) {
    fprintf(stderr, "test_directstorage_gpu: NVMe → GPU VRAM direct benchmark (%s)\n", path);
    fprintf(stderr, "  bs=%dK\n", g_block_size_kb);

    HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    bool com_inited = SUCCEEDED(hr);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
        fprintf(stderr, "  ERROR: CoInitializeEx failed: 0x%08lx\n", (unsigned long)hr);
        return false;
    }

    // --- Create DXGI factory and find hardware GPU ---
    IDXGIFactory4 * dxgiFactory = nullptr;
    hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory));
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: CreateDXGIFactory1 failed: 0x%08lx\n", (unsigned long)hr);
        if (com_inited) CoUninitialize();
        return false;
    }

    IDXGIAdapter1 * adapter = nullptr;
    for (UINT i = 0; dxgiFactory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            adapter->Release();
            adapter = nullptr;
            continue;
        }
        fprintf(stderr, "  GPU: %ls (VRAM: %.0f MB)\n",
                desc.Description,
                desc.DedicatedVideoMemory / (1024.0 * 1024.0));
        break;
    }
    if (!adapter) {
        fprintf(stderr, "  ERROR: No hardware GPU adapter found\n");
        dxgiFactory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }

    // --- Create D3D12 device ---
    ID3D12Device * device = nullptr;
    hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
    adapter->Release();
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: D3D12CreateDevice failed: 0x%08lx\n", (unsigned long)hr);
        dxgiFactory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }

    // --- DirectStorage factory + file ---
    IDStorageFactory * factory = nullptr;
    hr = DStorageGetFactory(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: DStorageGetFactory failed: 0x%08lx\n", (unsigned long)hr);
        device->Release();
        dxgiFactory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }

    int wlen = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);
    std::vector<wchar_t> wpath(wlen);
    MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath.data(), wlen);

    IDStorageFile * dsFile = nullptr;
    hr = factory->OpenFile(wpath.data(), IID_PPV_ARGS(&dsFile));
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: OpenFile failed: 0x%08lx\n", (unsigned long)hr);
        factory->Release();
        device->Release();
        dxgiFactory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }

    BY_HANDLE_FILE_INFORMATION finfo;
    hr = dsFile->GetFileInformation(&finfo);
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: GetFileInformation failed: 0x%08lx\n", (unsigned long)hr);
        dsFile->Release();
        factory->Release();
        device->Release();
        dxgiFactory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }
    uint64_t file_size = ((uint64_t)finfo.nFileSizeHigh << 32) | finfo.nFileSizeLow;

    const size_t bs = (size_t)g_block_size_kb * 1024;
    uint64_t aligned_size = file_size & ~((uint64_t)bs - 1);
    uint64_t total_blocks = aligned_size / bs;

    fprintf(stderr, "  File size: %.2f GB (%.2f GB aligned to %dK blocks)\n",
            file_size    / (1024.0 * 1024.0 * 1024.0),
            aligned_size / (1024.0 * 1024.0 * 1024.0),
            g_block_size_kb);

    // --- GPU buffer + queue sizing ---
    const size_t max_buf_bytes = 256ULL * 1024 * 1024;  // 256 MB GPU VRAM ring
    int n_buf_slots = (int)(max_buf_bytes / bs);
    if (n_buf_slots > (int)DSTORAGE_MAX_QUEUE_CAPACITY) n_buf_slots = (int)DSTORAGE_MAX_QUEUE_CAPACITY;
    if (n_buf_slots > (int)total_blocks) n_buf_slots = (int)total_blocks;
    if (n_buf_slots < 1) n_buf_slots = 1;

    int queue_cap = 1;
    while (queue_cap * 2 <= n_buf_slots) queue_cap *= 2;
    size_t gpu_buf_size = (size_t)queue_cap * bs;

    fprintf(stderr, "  Queue capacity: %d  GPU buffer: %.1f MB\n",
            queue_cap, gpu_buf_size / (1024.0 * 1024.0));

    // --- Create D3D12 buffer in GPU VRAM (DEFAULT heap) ---
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC resDesc = {};
    resDesc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
    resDesc.Width              = gpu_buf_size;
    resDesc.Height             = 1;
    resDesc.DepthOrArraySize   = 1;
    resDesc.MipLevels          = 1;
    resDesc.SampleDesc.Count   = 1;
    resDesc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ID3D12Resource * gpuBuffer = nullptr;
    hr = device->CreateCommittedResource(
        &heapProps, D3D12_HEAP_FLAG_NONE, &resDesc,
        D3D12_RESOURCE_STATE_COMMON, nullptr,
        IID_PPV_ARGS(&gpuBuffer));
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: CreateCommittedResource failed: 0x%08lx (%.1f MB)\n",
                (unsigned long)hr, gpu_buf_size / (1024.0 * 1024.0));
        dsFile->Release();
        factory->Release();
        device->Release();
        dxgiFactory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }

    // --- Create DirectStorage queue WITH D3D12 device (enables GPU destination) ---
    DSTORAGE_QUEUE_DESC queueDesc = {};
    queueDesc.Capacity   = (UINT16)queue_cap;
    queueDesc.Priority   = DSTORAGE_PRIORITY_HIGH;
    queueDesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    queueDesc.Device     = device;  // KEY: D3D12 device enables NVMe → GPU DMA

    IDStorageQueue * queue = nullptr;
    hr = factory->CreateQueue(&queueDesc, IID_PPV_ARGS(&queue));
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: CreateQueue (GPU) failed: 0x%08lx\n", (unsigned long)hr);
        gpuBuffer->Release();
        dsFile->Release();
        factory->Release();
        device->Release();
        dxgiFactory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }

    // Sync mechanism
    IDStorageQueue1 * queue1 = nullptr;
    queue->QueryInterface(IID_PPV_ARGS(&queue1));

    IDStorageStatusArray * statusArray = nullptr;
    HANDLE completion_event = NULL;
    if (queue1) {
        completion_event = CreateEvent(NULL, FALSE, FALSE, NULL);
        fprintf(stderr, "  Sync: IDStorageQueue1::EnqueueSetEvent\n");
    } else {
        hr = factory->CreateStatusArray(1, "bench-gpu", IID_PPV_ARGS(&statusArray));
        if (FAILED(hr)) {
            fprintf(stderr, "  ERROR: CreateStatusArray failed: 0x%08lx\n", (unsigned long)hr);
            queue->Release();
            gpuBuffer->Release();
            dsFile->Release();
            factory->Release();
            device->Release();
            dxgiFactory->Release();
            if (com_inited) CoUninitialize();
            return false;
        }
        fprintf(stderr, "  Sync: IDStorageStatusArray polling\n");
    }

    fprintf(stderr, "  Submitting %llu blocks in batches of %d (NVMe → GPU VRAM direct)...\n",
            (unsigned long long)total_blocks, queue_cap);

    // --- Benchmark loop ---
    auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t submitted = 0;

    while (submitted < total_blocks) {
        int this_batch = (int)std::min((uint64_t)queue_cap, total_blocks - submitted);

        for (int i = 0; i < this_batch; i++) {
            DSTORAGE_REQUEST req = {};
            req.Options.SourceType        = DSTORAGE_REQUEST_SOURCE_FILE;
            req.Options.DestinationType   = DSTORAGE_REQUEST_DESTINATION_BUFFER;
            req.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
            req.Source.File.Source         = dsFile;
            req.Source.File.Offset        = (submitted + i) * bs;
            req.Source.File.Size          = (UINT32)bs;
            req.Destination.Buffer.Resource = gpuBuffer;
            req.Destination.Buffer.Offset   = (UINT64)i * bs;  // circular within buffer
            req.Destination.Buffer.Size     = (UINT32)bs;
            req.UncompressedSize            = (UINT32)bs;

            queue->EnqueueRequest(&req);
        }

        if (queue1) {
            queue1->EnqueueSetEvent(completion_event);
            queue->Submit();
            WaitForSingleObject(completion_event, INFINITE);
        } else {
            queue->EnqueueStatus(statusArray, 0);
            queue->Submit();
            while (!statusArray->IsComplete(0)) {
                YieldProcessor();
            }
        }

        submitted += this_batch;
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    // Check for errors
    DSTORAGE_ERROR_RECORD ds_err = {};
    queue->RetrieveErrorRecord(&ds_err);
    if (FAILED(ds_err.FirstFailure.HResult)) {
        fprintf(stderr, "  WARNING: DirectStorage error: 0x%08lx\n",
                (unsigned long)ds_err.FirstFailure.HResult);
    }

    uint64_t total_bytes = submitted * bs;
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double bw = ((double)total_bytes / elapsed) / (1024.0 * 1024.0 * 1024.0);

    fprintf(stderr, "\n");
    fprintf(stderr, "  ====== DIRECTSTORAGE GPU-DIRECT RESULTS (NVMe → GPU VRAM) ======\n");
    fprintf(stderr, "  Block size:   %d KiB\n", g_block_size_kb);
    fprintf(stderr, "  Queue cap:    %d\n", queue_cap);
    fprintf(stderr, "  GPU buffer:   %.1f MB (D3D12_HEAP_TYPE_DEFAULT = VRAM)\n",
            gpu_buf_size / (1024.0 * 1024.0));
    fprintf(stderr, "  Data read:    %.2f GB\n", total_bytes / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "  Wall time:    %.2f s\n", elapsed);
    fprintf(stderr, "  Bandwidth:    %.2f GB/s\n", bw);
    fprintf(stderr, "  Requests:     %llu\n", (unsigned long long)submitted);
    fprintf(stderr, "  IOPS:         %.0f\n", submitted / elapsed);
    fprintf(stderr, "  ================================================================\n");

    // Cleanup
    if (completion_event) CloseHandle(completion_event);
    if (statusArray) statusArray->Release();
    if (queue1) queue1->Release();
    queue->Release();
    gpuBuffer->Release();
    dsFile->Release();
    factory->Release();
    device->Release();
    dxgiFactory->Release();
    if (com_inited) CoUninitialize();

    return true;
}

// ---------------------------------------------------------------------------
// DirectStorage 2-buffer prefill pipeline test — NVMe → GPU VRAM with overlap
//
// Simulates the streaming prefill pipeline with 2 ping-pong GPU buffers:
//   Buffer A: DMA layer N   while GPU "computes" layer N-1 on Buffer B
//   Buffer B: DMA layer N+1 while GPU "computes" layer N   on Buffer A
//
// The file is divided into N equal-sized "layers" and each layer is DMA'd
// into alternating GPU VRAM buffers via DirectStorage.  An optional
// --compute-ms flag lets you simulate GPU compute time to measure overlap.
//
// Activated with: --dstorage-prefill <path>
// ---------------------------------------------------------------------------

static bool test_dstorage_prefill(const char * path) {
    fprintf(stderr, "test_dstorage_prefill: 2-buffer NVMe → GPU prefill pipeline (%s)\n", path);

    const int    n_layers = g_nlayers;
    const size_t bs       = (size_t)g_block_size_kb * 1024;

    fprintf(stderr, "  layers=%d  bs=%dK  compute-ms=%.1f\n", n_layers, g_block_size_kb, g_compute_ms);

    // --- COM + DXGI + D3D12 setup ---
    HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    bool com_inited = SUCCEEDED(hr);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
        fprintf(stderr, "  ERROR: CoInitializeEx failed: 0x%08lx\n", (unsigned long)hr);
        return false;
    }

    IDXGIFactory4 * dxgiFactory = nullptr;
    hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory));
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: CreateDXGIFactory1 failed: 0x%08lx\n", (unsigned long)hr);
        if (com_inited) CoUninitialize();
        return false;
    }

    IDXGIAdapter1 * adapter = nullptr;
    for (UINT i = 0; dxgiFactory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) { adapter->Release(); adapter = nullptr; continue; }
        fprintf(stderr, "  GPU: %ls (VRAM: %.0f MB)\n",
                desc.Description, desc.DedicatedVideoMemory / (1024.0 * 1024.0));
        break;
    }
    if (!adapter) {
        fprintf(stderr, "  ERROR: No hardware GPU adapter found\n");
        dxgiFactory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }

    ID3D12Device * device = nullptr;
    hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
    adapter->Release();
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: D3D12CreateDevice failed: 0x%08lx\n", (unsigned long)hr);
        dxgiFactory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }

    // --- DirectStorage factory + file ---
    IDStorageFactory * factory = nullptr;
    hr = DStorageGetFactory(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: DStorageGetFactory failed: 0x%08lx\n", (unsigned long)hr);
        device->Release(); dxgiFactory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }

    int wlen = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);
    std::vector<wchar_t> wpath(wlen);
    MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath.data(), wlen);

    IDStorageFile * dsFile = nullptr;
    hr = factory->OpenFile(wpath.data(), IID_PPV_ARGS(&dsFile));
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: OpenFile failed: 0x%08lx\n", (unsigned long)hr);
        factory->Release(); device->Release(); dxgiFactory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }

    BY_HANDLE_FILE_INFORMATION finfo;
    hr = dsFile->GetFileInformation(&finfo);
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: GetFileInformation failed: 0x%08lx\n", (unsigned long)hr);
        dsFile->Release(); factory->Release(); device->Release(); dxgiFactory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }
    uint64_t file_size = ((uint64_t)finfo.nFileSizeHigh << 32) | finfo.nFileSizeLow;

    // --- Compute layer geometry ---
    uint64_t layer_bytes = ((file_size / n_layers) / bs) * bs;  // align down
    if (layer_bytes == 0) {
        fprintf(stderr, "  ERROR: File too small for %d layers at %dK blocks\n", n_layers, g_block_size_kb);
        dsFile->Release(); factory->Release(); device->Release(); dxgiFactory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }
    int blocks_per_layer = (int)(layer_bytes / bs);

    fprintf(stderr, "  File size:     %.2f GB\n", file_size / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "  Layer size:    %.2f MB (%d x %dK blocks)\n",
            layer_bytes / (1024.0 * 1024.0), blocks_per_layer, g_block_size_kb);
    fprintf(stderr, "  Total layers:  %d\n", n_layers);

    // --- Allocate 2 GPU buffers (ping-pong in VRAM) ---
    ID3D12Resource * gpuBuf[2] = { nullptr, nullptr };
    for (int b = 0; b < 2; b++) {
        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

        D3D12_RESOURCE_DESC resDesc = {};
        resDesc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
        resDesc.Width            = layer_bytes;
        resDesc.Height           = 1;
        resDesc.DepthOrArraySize = 1;
        resDesc.MipLevels        = 1;
        resDesc.SampleDesc.Count = 1;
        resDesc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        hr = device->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &resDesc,
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&gpuBuf[b]));
        if (FAILED(hr)) {
            fprintf(stderr, "  ERROR: CreateCommittedResource[%d] failed: 0x%08lx (%.1f MB)\n",
                    b, (unsigned long)hr, layer_bytes / (1024.0 * 1024.0));
            if (b > 0 && gpuBuf[0]) gpuBuf[0]->Release();
            dsFile->Release(); factory->Release(); device->Release(); dxgiFactory->Release();
            if (com_inited) CoUninitialize();
            return false;
        }
    }
    fprintf(stderr, "  GPU buffers:   2 x %.2f MB (D3D12_HEAP_TYPE_DEFAULT = VRAM)\n",
            layer_bytes / (1024.0 * 1024.0));

    // --- DirectStorage queue sized for one full layer ---
    int queue_cap = 1;
    while (queue_cap < blocks_per_layer && queue_cap * 2 <= (int)DSTORAGE_MAX_QUEUE_CAPACITY)
        queue_cap *= 2;

    DSTORAGE_QUEUE_DESC queueDesc = {};
    queueDesc.Capacity   = (UINT16)queue_cap;
    queueDesc.Priority   = DSTORAGE_PRIORITY_HIGH;
    queueDesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    queueDesc.Device     = device;  // enables NVMe → GPU DMA

    IDStorageQueue * queue = nullptr;
    hr = factory->CreateQueue(&queueDesc, IID_PPV_ARGS(&queue));
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: CreateQueue failed: 0x%08lx\n", (unsigned long)hr);
        gpuBuf[0]->Release(); gpuBuf[1]->Release();
        dsFile->Release(); factory->Release(); device->Release(); dxgiFactory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }

    // Sync: prefer EnqueueSetEvent (DS 1.3+), fall back to StatusArray
    IDStorageQueue1 * queue1 = nullptr;
    queue->QueryInterface(IID_PPV_ARGS(&queue1));

    HANDLE loadEvent[2] = { CreateEvent(NULL, FALSE, FALSE, NULL),
                            CreateEvent(NULL, FALSE, FALSE, NULL) };

    IDStorageStatusArray * statusArray = nullptr;
    if (!queue1) {
        hr = factory->CreateStatusArray(2, "prefill-sync", IID_PPV_ARGS(&statusArray));
        if (FAILED(hr)) {
            fprintf(stderr, "  ERROR: CreateStatusArray failed: 0x%08lx\n", (unsigned long)hr);
            queue->Release(); gpuBuf[0]->Release(); gpuBuf[1]->Release();
            dsFile->Release(); factory->Release(); device->Release(); dxgiFactory->Release();
            if (com_inited) CoUninitialize();
            return false;
        }
    }

    fprintf(stderr, "  Queue cap:     %d  (blocks/layer: %d)\n", queue_cap, blocks_per_layer);
    fprintf(stderr, "  Sync:          %s\n", queue1 ? "EnqueueSetEvent" : "StatusArray");

    // --- Helper: submit one layer's DMA into gpuBuf[buf_idx] ---
    auto submit_layer_dma = [&](int il, int buf_idx) {
        uint64_t layer_offset = (uint64_t)il * layer_bytes;
        int blocks_left = blocks_per_layer;
        int block_i = 0;

        while (blocks_left > 0) {
            int batch = std::min(queue_cap, blocks_left);

            for (int i = 0; i < batch; i++) {
                DSTORAGE_REQUEST req = {};
                req.Options.SourceType        = DSTORAGE_REQUEST_SOURCE_FILE;
                req.Options.DestinationType   = DSTORAGE_REQUEST_DESTINATION_BUFFER;
                req.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
                req.Source.File.Source         = dsFile;
                req.Source.File.Offset        = layer_offset + (uint64_t)(block_i + i) * bs;
                req.Source.File.Size          = (UINT32)bs;
                req.Destination.Buffer.Resource = gpuBuf[buf_idx];
                req.Destination.Buffer.Offset   = (UINT64)(block_i + i) * bs;
                req.Destination.Buffer.Size     = (UINT32)bs;
                req.UncompressedSize            = (UINT32)bs;

                queue->EnqueueRequest(&req);
            }

            block_i     += batch;
            blocks_left -= batch;

            // Mid-layer drain: if more blocks remain, signal + submit + wait
            if (blocks_left > 0) {
                if (queue1) {
                    queue1->EnqueueSetEvent(loadEvent[buf_idx]);
                    queue->Submit();
                    WaitForSingleObject(loadEvent[buf_idx], INFINITE);
                } else {
                    queue->EnqueueStatus(statusArray, buf_idx);
                    queue->Submit();
                    while (!statusArray->IsComplete(buf_idx)) { YieldProcessor(); }
                }
            }
        }

        // Final signal for this layer
        if (queue1) {
            queue1->EnqueueSetEvent(loadEvent[buf_idx]);
        } else {
            queue->EnqueueStatus(statusArray, buf_idx);
        }
        queue->Submit();
    };

    auto wait_layer_dma = [&](int buf_idx) {
        if (queue1) {
            WaitForSingleObject(loadEvent[buf_idx], INFINITE);
        } else {
            while (!statusArray->IsComplete(buf_idx)) { YieldProcessor(); }
        }
    };

    // --- Timing arrays ---
    std::vector<double> dma_times(n_layers, 0.0);   // submit → completion
    std::vector<double> wait_times(n_layers, 0.0);   // blocked waiting for DMA
    std::vector<double> compute_times(n_layers, 0.0); // simulated compute

    fprintf(stderr, "\n  Running 2-buffer prefill pipeline (%d layers)...\n\n", n_layers);

    // --- Pre-load layer 0 into buffer 0 ---
    auto t_pipeline_start = std::chrono::high_resolution_clock::now();
    auto t_submit = std::chrono::high_resolution_clock::now();
    submit_layer_dma(0, 0);

    uint64_t total_bytes_dma = 0;

    // --- Pipeline loop ---
    for (int il = 0; il < n_layers; il++) {
        int cur_buf = il % 2;

        // 1. Wait for current layer's DMA to complete
        auto t_wait_start = std::chrono::high_resolution_clock::now();
        wait_layer_dma(cur_buf);
        auto t_wait_end = std::chrono::high_resolution_clock::now();

        wait_times[il] = std::chrono::duration<double, std::milli>(t_wait_end - t_wait_start).count();
        dma_times[il]  = std::chrono::duration<double, std::milli>(t_wait_end - t_submit).count();
        total_bytes_dma += layer_bytes;

        // 2. Start loading NEXT layer into alternate buffer (overlaps with compute)
        if (il + 1 < n_layers) {
            t_submit = std::chrono::high_resolution_clock::now();
            submit_layer_dma(il + 1, (il + 1) % 2);
        }

        // 3. Simulate GPU compute on current buffer
        //    In real pipeline: CUDA kernels run on gpuBuf[cur_buf] data
        //    Here: busy-spin for --compute-ms to measure overlap effectiveness
        if (g_compute_ms > 0.0f) {
            auto t_compute_start = std::chrono::high_resolution_clock::now();
            auto deadline = t_compute_start + std::chrono::microseconds((int)(g_compute_ms * 1000.0f));
            while (std::chrono::high_resolution_clock::now() < deadline) {
                YieldProcessor();
            }
            auto t_compute_end = std::chrono::high_resolution_clock::now();
            compute_times[il] = std::chrono::duration<double, std::milli>(t_compute_end - t_compute_start).count();
        }

        if (g_trace) {
            fprintf(stderr, "    layer %3d: buf[%d] dma=%.1fms wait=%.1fms compute=%.1fms\n",
                    il, cur_buf, dma_times[il], wait_times[il], compute_times[il]);
        }
    }

    auto t_pipeline_end = std::chrono::high_resolution_clock::now();
    double pipeline_elapsed = std::chrono::duration<double>(t_pipeline_end - t_pipeline_start).count();

    // --- Error check ---
    DSTORAGE_ERROR_RECORD ds_err = {};
    queue->RetrieveErrorRecord(&ds_err);
    if (FAILED(ds_err.FirstFailure.HResult)) {
        fprintf(stderr, "  WARNING: DirectStorage error: 0x%08lx\n",
                (unsigned long)ds_err.FirstFailure.HResult);
    }

    // --- Aggregate stats ---
    double total_gb = (double)total_bytes_dma / (1024.0 * 1024.0 * 1024.0);
    double bw       = total_gb / pipeline_elapsed;

    double sum_dma = 0, max_dma = 0, min_dma = 1e9;
    double sum_wait = 0, sum_compute = 0;
    int    stall_count = 0;  // layers where wait > 0.1ms (GPU would stall)

    for (int il = 0; il < n_layers; il++) {
        sum_dma     += dma_times[il];
        sum_wait    += wait_times[il];
        sum_compute += compute_times[il];
        if (dma_times[il] > max_dma) max_dma = dma_times[il];
        if (dma_times[il] < min_dma) min_dma = dma_times[il];
        // After layer 0, if we waited more than 0.1ms the GPU would have stalled
        if (il > 0 && wait_times[il] > 5.0) stall_count++;  // 5ms: ignore fence overhead, catch real DMA stalls
    }
    double avg_dma     = sum_dma / n_layers;
    double avg_wait    = sum_wait / n_layers;
    double avg_compute = g_compute_ms > 0 ? (sum_compute / n_layers) : 0;

    // Overlap efficiency: fraction of DMA time hidden behind compute
    // Perfect overlap = 0 wait time after layer 0
    double overlap_eff = 0.0;
    if (g_compute_ms > 0 && n_layers > 1) {
        double stall_after_0 = sum_wait - wait_times[0]; // exclude layer 0 cold-start
        double total_dma_after_0 = sum_dma - dma_times[0];
        if (total_dma_after_0 > 0) {
            overlap_eff = 1.0 - (stall_after_0 / total_dma_after_0);
            if (overlap_eff < 0) overlap_eff = 0;
        }
    }

    // Token throughput estimates (8K context, various model sizes)
    // tok/s = 8192 / (n_layers * per_layer_time_s)
    // per_layer_time = max(compute_time, dma_time) with perfect overlap
    double dma_only_tks = 0;
    if (avg_dma > 0) {
        dma_only_tks = 8192.0 / (n_layers * avg_dma / 1000.0);
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "  ====== DIRECTSTORAGE 2-BUFFER PREFILL RESULTS ======\n");
    fprintf(stderr, "  Layers:         %d x %.2f MB = %.2f GB\n",
            n_layers, layer_bytes / (1024.0 * 1024.0), total_gb);
    fprintf(stderr, "  Block size:     %d KiB\n", g_block_size_kb);
    fprintf(stderr, "  Pipeline time:  %.3f s\n", pipeline_elapsed);
    fprintf(stderr, "  Bandwidth:      %.2f GB/s\n", bw);
    fprintf(stderr, "  Per-layer DMA:  avg=%.1fms  min=%.1fms  max=%.1fms\n",
            avg_dma, min_dma, max_dma);
    fprintf(stderr, "  Per-layer wait: avg=%.1fms (time GPU stalls if compute=0)\n", avg_wait);
    if (g_compute_ms > 0) {
        fprintf(stderr, "  Simulated compute: %.1fms/layer  overlap=%.0f%%  stalls=%d/%d\n",
                avg_compute, overlap_eff * 100.0, stall_count, n_layers - 1);
    }
    fprintf(stderr, "  DMA ceiling:    %.0f tok/s @ 8K ctx (bottleneck if compute << DMA)\n",
            dma_only_tks);
    fprintf(stderr, "  ---\n");
    fprintf(stderr, "  Model estimates (8K context, compute_ms/layer from real measurements):\n");
    // 27B Q6_K: 48 layers, ~440 MB/layer, ~170ms compute/layer @ 1000 t/s
    double est_27b_dma = (440.0 * 1024 * 1024) / (bw * 1024 * 1024 * 1024) * 1000.0;
    double est_27b_bottleneck = std::max(170.0, est_27b_dma);
    double est_27b_tks = 8192.0 / (48 * est_27b_bottleneck / 1000.0);
    fprintf(stderr, "    27B Q6_K (48L, 440MB/L): DMA=%.0fms  compute=170ms  → %.0f tok/s\n",
            est_27b_dma, est_27b_tks);
    // 70B Q4_K_M: 80 layers, ~550 MB/layer, ~200ms compute/layer
    double est_70b_dma = (550.0 * 1024 * 1024) / (bw * 1024 * 1024 * 1024) * 1000.0;
    double est_70b_bottleneck = std::max(200.0, est_70b_dma);
    double est_70b_tks = 8192.0 / (80 * est_70b_bottleneck / 1000.0);
    fprintf(stderr, "    70B Q4_K (80L, 550MB/L): DMA=%.0fms  compute=200ms  → %.0f tok/s\n",
            est_70b_dma, est_70b_tks);
    fprintf(stderr, "  ====================================================\n");

    // --- Cleanup ---
    if (loadEvent[0]) CloseHandle(loadEvent[0]);
    if (loadEvent[1]) CloseHandle(loadEvent[1]);
    if (statusArray) statusArray->Release();
    if (queue1) queue1->Release();
    queue->Release();
    gpuBuf[0]->Release();
    gpuBuf[1]->Release();
    dsFile->Release();
    factory->Release();
    device->Release();
    dxgiFactory->Release();
    if (com_inited) CoUninitialize();

    return true;
}

#if defined(LLAMA_CUDA_INTEROP)

// ---------------------------------------------------------------------------
// DirectStorage + CUDA interop test — NVMe → D3D12 buffer → CUDA device ptr
//
// Proves end-to-end: DirectStorage loads data into a D3D12 committed resource,
// CUDA imports that resource via cudaImportExternalMemory, gets a device
// pointer, and reads the data back to host for verification.
//
// Sync path: D3D12 fence → cudaImportExternalSemaphore → GPU-side wait.
// This is the zero-CPU-stall path needed for real prefill integration.
//
// The test also runs a 2-buffer overlap pipeline with CUDA readback to prove
// the full DMA + interop + compute pattern works end to end.
//
// Activated with: --dstorage-cuda <path>
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "  CUDA ERROR: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        return false; \
    } \
} while(0)

static bool test_dstorage_cuda_interop(const char * path) {
    fprintf(stderr, "test_dstorage_cuda_interop: D3D12-CUDA interop pipeline (%s)\n", path);

    const int    n_layers = g_nlayers;
    const size_t bs       = (size_t)g_block_size_kb * 1024;

    fprintf(stderr, "  layers=%d  bs=%dK  compute-ms=%.1f\n", n_layers, g_block_size_kb, g_compute_ms);

    // --- Verify CUDA device is available ---
    int cuda_device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&cuda_device_count));
    if (cuda_device_count == 0) {
        fprintf(stderr, "  ERROR: No CUDA devices found\n");
        return false;
    }

    cudaDeviceProp cuda_props;
    CUDA_CHECK(cudaGetDeviceProperties(&cuda_props, 0));
    fprintf(stderr, "  CUDA device: %s (%.0f MB VRAM)\n",
            cuda_props.name, cuda_props.totalGlobalMem / (1024.0 * 1024.0));

    // --- COM + DXGI + D3D12 setup ---
    HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    bool com_inited = SUCCEEDED(hr);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
        fprintf(stderr, "  ERROR: CoInitializeEx failed: 0x%08lx\n", (unsigned long)hr);
        return false;
    }

    IDXGIFactory4 * dxgiFactory = nullptr;
    hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory));
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: CreateDXGIFactory1 failed: 0x%08lx\n", (unsigned long)hr);
        if (com_inited) CoUninitialize();
        return false;
    }

    // Find the same GPU that CUDA is using (match by LUID)
    IDXGIAdapter1 * adapter = nullptr;
    for (UINT i = 0; dxgiFactory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) { adapter->Release(); adapter = nullptr; continue; }
        fprintf(stderr, "  D3D12 GPU: %ls (VRAM: %.0f MB)\n",
                desc.Description, desc.DedicatedVideoMemory / (1024.0 * 1024.0));
        break;
    }
    if (!adapter) {
        fprintf(stderr, "  ERROR: No hardware GPU adapter found\n");
        dxgiFactory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }

    ID3D12Device * device = nullptr;
    hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
    adapter->Release();
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: D3D12CreateDevice failed: 0x%08lx\n", (unsigned long)hr);
        dxgiFactory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }

    // --- DirectStorage factory + file ---
    IDStorageFactory * factory = nullptr;
    hr = DStorageGetFactory(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: DStorageGetFactory failed: 0x%08lx\n", (unsigned long)hr);
        device->Release(); dxgiFactory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }

    int wlen = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);
    std::vector<wchar_t> wpath(wlen);
    MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath.data(), wlen);

    IDStorageFile * dsFile = nullptr;
    hr = factory->OpenFile(wpath.data(), IID_PPV_ARGS(&dsFile));
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: OpenFile failed: 0x%08lx\n", (unsigned long)hr);
        factory->Release(); device->Release(); dxgiFactory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }

    BY_HANDLE_FILE_INFORMATION finfo;
    hr = dsFile->GetFileInformation(&finfo);
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: GetFileInformation failed: 0x%08lx\n", (unsigned long)hr);
        dsFile->Release(); factory->Release(); device->Release(); dxgiFactory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }
    uint64_t file_size = ((uint64_t)finfo.nFileSizeHigh << 32) | finfo.nFileSizeLow;

    // --- Layer geometry ---
    uint64_t layer_bytes = ((file_size / n_layers) / bs) * bs;  // align down
    if (layer_bytes == 0) {
        fprintf(stderr, "  ERROR: File too small for %d layers at %dK blocks\n", n_layers, g_block_size_kb);
        dsFile->Release(); factory->Release(); device->Release(); dxgiFactory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }
    int blocks_per_layer = (int)(layer_bytes / bs);

    fprintf(stderr, "  File size:     %.2f GB\n", file_size / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "  Layer size:    %.2f MB (%d x %dK blocks)\n",
            layer_bytes / (1024.0 * 1024.0), blocks_per_layer, g_block_size_kb);
    fprintf(stderr, "  Total layers:  %d\n", n_layers);

    // --- Allocate 2 shared D3D12 buffers for CUDA interop ---
    //
    // Approach A (NVIDIA CUDA sample pattern): CreateCommittedResource with
    // D3D12_HEAP_FLAG_SHARED + D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.
    // CUDA imports via cudaExternalMemoryHandleTypeD3D12Resource.
    //
    // Approach B (fallback): CreateHeap(SHARED) + CreatePlacedResource.
    // CUDA imports via cudaExternalMemoryHandleTypeD3D12Heap.
    //
    uint64_t heap_size = (layer_bytes + 65535) & ~((uint64_t)65535);  // 64KB aligned

    bool use_heap = false;
    ID3D12Heap * gpuHeap[2] = { nullptr, nullptr };
    ID3D12Resource * gpuBuf[2] = { nullptr, nullptr };

    // -- Approach A: Committed resource + SHARED + UAV --
    {
        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

        D3D12_RESOURCE_DESC resDesc = {};
        resDesc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
        resDesc.Width            = heap_size;
        resDesc.Height           = 1;
        resDesc.DepthOrArraySize = 1;
        resDesc.MipLevels        = 1;
        resDesc.SampleDesc.Count = 1;
        resDesc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        resDesc.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        hr = device->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_SHARED, &resDesc,
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&gpuBuf[0]));

        if (SUCCEEDED(hr)) {
            hr = device->CreateCommittedResource(
                &heapProps, D3D12_HEAP_FLAG_SHARED, &resDesc,
                D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&gpuBuf[1]));
            if (FAILED(hr)) {
                gpuBuf[0]->Release(); gpuBuf[0] = nullptr;
            }
        }

        if (FAILED(hr)) {
            fprintf(stderr, "  NOTE: CommittedResource+SHARED failed (0x%08lx), trying shared heap...\n",
                    (unsigned long)hr);
            gpuBuf[0] = gpuBuf[1] = nullptr;
            use_heap = true;
        }
    }

    // -- Approach B: Shared heap + placed resource (fallback) --
    if (use_heap) {
        for (int b = 0; b < 2; b++) {
            D3D12_HEAP_DESC heapDesc = {};
            heapDesc.SizeInBytes     = heap_size;
            heapDesc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
            heapDesc.Flags           = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;

            hr = device->CreateHeap(&heapDesc, IID_PPV_ARGS(&gpuHeap[b]));
            if (FAILED(hr)) {
                fprintf(stderr, "  ERROR: CreateHeap[%d] SHARED failed: 0x%08lx (%.1f MB)\n",
                        b, (unsigned long)hr, heap_size / (1024.0 * 1024.0));
                if (b > 0 && gpuHeap[0]) gpuHeap[0]->Release();
                dsFile->Release(); factory->Release(); device->Release(); dxgiFactory->Release();
                if (com_inited) CoUninitialize();
                return false;
            }

            D3D12_RESOURCE_DESC resDesc = {};
            resDesc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
            resDesc.Width            = heap_size;
            resDesc.Height           = 1;
            resDesc.DepthOrArraySize = 1;
            resDesc.MipLevels        = 1;
            resDesc.SampleDesc.Count = 1;
            resDesc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            resDesc.Flags            = D3D12_RESOURCE_FLAG_NONE;

            hr = device->CreatePlacedResource(
                gpuHeap[b], 0, &resDesc,
                D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&gpuBuf[b]));
            if (FAILED(hr)) {
                fprintf(stderr, "  ERROR: CreatePlacedResource[%d] failed: 0x%08lx (%.1f MB)\n",
                        b, (unsigned long)hr, heap_size / (1024.0 * 1024.0));
                gpuHeap[b]->Release();
                if (b > 0) { gpuBuf[0]->Release(); gpuHeap[0]->Release(); }
                dsFile->Release(); factory->Release(); device->Release(); dxgiFactory->Release();
                if (com_inited) CoUninitialize();
                return false;
            }
        }
    }

    fprintf(stderr, "  D3D12 bufs:    2 x %.2f MB (%s)\n",
            heap_size / (1024.0 * 1024.0),
            use_heap ? "shared heap + placed resource" : "committed resource + SHARED");

    // --- Create shared HANDLEs for CUDA import ---
    // Committed resource: share the resource; Shared heap: share the heap
    HANDLE sharedHandles[2] = { NULL, NULL };
    for (int b = 0; b < 2; b++) {
        ID3D12DeviceChild * shareTarget = use_heap
            ? static_cast<ID3D12DeviceChild *>(gpuHeap[b])
            : static_cast<ID3D12DeviceChild *>(gpuBuf[b]);
        hr = device->CreateSharedHandle(shareTarget, nullptr, GENERIC_ALL, nullptr, &sharedHandles[b]);
        if (FAILED(hr)) {
            fprintf(stderr, "  ERROR: CreateSharedHandle[%d] failed: 0x%08lx\n", b, (unsigned long)hr);
            for (int j = 0; j < b; j++) CloseHandle(sharedHandles[j]);
            for (int j = 0; j < 2; j++) {
                if (gpuBuf[j])  gpuBuf[j]->Release();
                if (gpuHeap[j]) gpuHeap[j]->Release();
            }
            dsFile->Release(); factory->Release(); device->Release(); dxgiFactory->Release();
            if (com_inited) CoUninitialize();
            return false;
        }
    }

    // --- Import D3D12 memory into CUDA ---
    cudaExternalMemory_t extMem[2] = {};
    void * cudaDevPtr[2] = { nullptr, nullptr };

    for (int b = 0; b < 2; b++) {
        cudaExternalMemoryHandleDesc memDesc = {};
        memDesc.type                = use_heap
            ? cudaExternalMemoryHandleTypeD3D12Heap
            : cudaExternalMemoryHandleTypeD3D12Resource;
        memDesc.handle.win32.handle = sharedHandles[b];
        memDesc.size                = heap_size;
        memDesc.flags               = use_heap ? 0 : cudaExternalMemoryDedicated;

        CUDA_CHECK(cudaImportExternalMemory(&extMem[b], &memDesc));

        cudaExternalMemoryBufferDesc bufDesc = {};
        bufDesc.offset = 0;
        bufDesc.size   = layer_bytes;

        CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&cudaDevPtr[b], extMem[b], &bufDesc));
    }
    fprintf(stderr, "  CUDA import:   buf[0]=%p  buf[1]=%p\n", cudaDevPtr[0], cudaDevPtr[1]);

    // --- Create D3D12 fence with SHARED flag for cross-API sync ---
    ID3D12Fence * d3dFence = nullptr;
    hr = device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&d3dFence));
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: CreateFence SHARED failed: 0x%08lx\n", (unsigned long)hr);
        for (int b = 0; b < 2; b++) {
            cudaDestroyExternalMemory(extMem[b]);
            CloseHandle(sharedHandles[b]);
            gpuBuf[b]->Release();
            if (gpuHeap[b]) gpuHeap[b]->Release();
        }
        dsFile->Release(); factory->Release(); device->Release(); dxgiFactory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }

    HANDLE fenceSharedHandle = NULL;
    hr = device->CreateSharedHandle(d3dFence, nullptr, GENERIC_ALL, nullptr, &fenceSharedHandle);
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: CreateSharedHandle (fence) failed: 0x%08lx\n", (unsigned long)hr);
        d3dFence->Release();
        for (int b = 0; b < 2; b++) {
            cudaDestroyExternalMemory(extMem[b]);
            CloseHandle(sharedHandles[b]);
            gpuBuf[b]->Release();
            if (gpuHeap[b]) gpuHeap[b]->Release();
        }
        dsFile->Release(); factory->Release(); device->Release(); dxgiFactory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }

    // --- Import fence into CUDA as external semaphore ---
    cudaExternalSemaphore_t extSem = {};
    {
        cudaExternalSemaphoreHandleDesc semDesc = {};
        semDesc.type                = cudaExternalSemaphoreHandleTypeD3D12Fence;
        semDesc.handle.win32.handle = fenceSharedHandle;

        CUDA_CHECK(cudaImportExternalSemaphore(&extSem, &semDesc));
    }
    fprintf(stderr, "  D3D12 fence:   imported as CUDA external semaphore\n");

    // --- Create CUDA stream for compute/readback ---
    cudaStream_t cudaStream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&cudaStream));

    // --- DirectStorage queue (GPU destination) ---
    int queue_cap = 1;
    while (queue_cap < blocks_per_layer && queue_cap * 2 <= (int)DSTORAGE_MAX_QUEUE_CAPACITY)
        queue_cap *= 2;

    DSTORAGE_QUEUE_DESC queueDesc = {};
    queueDesc.Capacity   = (UINT16)queue_cap;
    queueDesc.Priority   = DSTORAGE_PRIORITY_HIGH;
    queueDesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    queueDesc.Device     = device;

    IDStorageQueue * queue = nullptr;
    hr = factory->CreateQueue(&queueDesc, IID_PPV_ARGS(&queue));
    if (FAILED(hr)) {
        fprintf(stderr, "  ERROR: CreateQueue failed: 0x%08lx\n", (unsigned long)hr);
        cudaStreamDestroy(cudaStream);
        cudaDestroyExternalSemaphore(extSem);
        CloseHandle(fenceSharedHandle);
        d3dFence->Release();
        for (int b = 0; b < 2; b++) {
            cudaDestroyExternalMemory(extMem[b]);
            CloseHandle(sharedHandles[b]);
            gpuBuf[b]->Release();
            if (gpuHeap[b]) gpuHeap[b]->Release();
        }
        dsFile->Release(); factory->Release(); device->Release(); dxgiFactory->Release();
        if (com_inited) CoUninitialize();
        return false;
    }

    fprintf(stderr, "  DS queue:      cap=%d  blocks/layer=%d\n", queue_cap, blocks_per_layer);

    // --- Phase 1: Single-layer verification (prove interop works) ---
    fprintf(stderr, "\n  === Phase 1: Single-layer DMA + CUDA readback verification ===\n");
    {
        // Read reference data from file using fread
        std::vector<uint8_t> ref_data(layer_bytes);
        FILE * f = fopen(path, "rb");
        if (!f) {
            fprintf(stderr, "  ERROR: cannot open file for reference read\n");
            // cleanup below
        } else {
            size_t n = fread(ref_data.data(), 1, layer_bytes, f);
            fclose(f);
            if (n != layer_bytes) {
                fprintf(stderr, "  WARNING: fread got %zu of %llu bytes\n", n, (unsigned long long)layer_bytes);
            }
        }

        // Submit DMA for layer 0 into gpuBuf[0]
        uint64_t fence_val = 1;
        {
            int blocks_left = blocks_per_layer;
            int block_i = 0;
            while (blocks_left > 0) {
                int batch = std::min(queue_cap, blocks_left);
                for (int i = 0; i < batch; i++) {
                    DSTORAGE_REQUEST req = {};
                    req.Options.SourceType        = DSTORAGE_REQUEST_SOURCE_FILE;
                    req.Options.DestinationType   = DSTORAGE_REQUEST_DESTINATION_BUFFER;
                    req.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
                    req.Source.File.Source         = dsFile;
                    req.Source.File.Offset        = (uint64_t)(block_i + i) * bs;
                    req.Source.File.Size          = (UINT32)bs;
                    req.Destination.Buffer.Resource = gpuBuf[0];
                    req.Destination.Buffer.Offset   = (UINT64)(block_i + i) * bs;
                    req.Destination.Buffer.Size     = (UINT32)bs;
                    req.UncompressedSize            = (UINT32)bs;
                    queue->EnqueueRequest(&req);
                }
                block_i     += batch;
                blocks_left -= batch;

                if (blocks_left > 0) {
                    // Mid-layer fence signal + wait (CPU side, for phase 1 only)
                    queue->EnqueueSignal(d3dFence, fence_val);
                    queue->Submit();
                    // CPU wait for this mid-layer batch
                    if (d3dFence->GetCompletedValue() < fence_val) {
                        HANDLE evt = CreateEvent(NULL, FALSE, FALSE, NULL);
                        d3dFence->SetEventOnCompletion(fence_val, evt);
                        WaitForSingleObject(evt, INFINITE);
                        CloseHandle(evt);
                    }
                    fence_val++;
                }
            }
        }

        // Signal fence when DMA is done
        queue->EnqueueSignal(d3dFence, fence_val);
        queue->Submit();

        // GPU-side wait: CUDA stream waits for D3D12 fence
        {
            cudaExternalSemaphoreWaitParams waitParams = {};
            waitParams.params.fence.value = fence_val;
            CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&extSem, &waitParams, 1, cudaStream));
        }

        // CUDA readback: copy from CUDA device ptr to host
        std::vector<uint8_t> readback(layer_bytes);
        CUDA_CHECK(cudaMemcpyAsync(readback.data(), cudaDevPtr[0], layer_bytes,
                                    cudaMemcpyDeviceToHost, cudaStream));
        CUDA_CHECK(cudaStreamSynchronize(cudaStream));

        // Verify data matches fread reference
        bool data_ok = (memcmp(readback.data(), ref_data.data(), layer_bytes) == 0);
        if (!data_ok) {
            // Find first mismatch for diagnostics
            for (size_t i = 0; i < layer_bytes; i++) {
                if (readback[i] != ref_data[i]) {
                    fprintf(stderr, "    MISMATCH at byte %zu: got 0x%02x expected 0x%02x\n",
                            i, readback[i], ref_data[i]);
                    break;
                }
            }
            fprintf(stderr, "  FAIL: CUDA readback does not match fread reference!\n");
        } else {
            fprintf(stderr, "    PASS: %.2f MB verified — DirectStorage → D3D12 → CUDA readback matches fread\n",
                    layer_bytes / (1024.0 * 1024.0));
        }

        if (!data_ok) {
            // Cleanup and bail
            queue->Release();
            cudaStreamDestroy(cudaStream);
            cudaDestroyExternalSemaphore(extSem);
            CloseHandle(fenceSharedHandle);
            d3dFence->Release();
            for (int b = 0; b < 2; b++) {
                cudaDestroyExternalMemory(extMem[b]);
                CloseHandle(sharedHandles[b]);
                gpuBuf[b]->Release();
                if (gpuHeap[b]) gpuHeap[b]->Release();
            }
            dsFile->Release(); factory->Release(); device->Release(); dxgiFactory->Release();
            if (com_inited) CoUninitialize();
            return false;
        }

        fence_val++;  // advance for phase 2
    }

    // --- Phase 2: 2-buffer overlap pipeline with CUDA readback ---
    fprintf(stderr, "\n  === Phase 2: 2-buffer overlap pipeline (%d layers, CUDA GPU-side sync) ===\n\n",
            n_layers);

    uint64_t fence_val = 100;  // start fresh, well past phase 1
    std::vector<double> dma_times(n_layers, 0.0);
    std::vector<double> wait_times(n_layers, 0.0);
    std::vector<double> compute_times(n_layers, 0.0);
    uint64_t total_bytes_dma = 0;

    // Helper: submit one layer's DMA into gpuBuf[buf_idx], signal fence
    auto submit_layer_dma = [&](int il, int buf_idx) -> uint64_t {
        uint64_t layer_offset = (uint64_t)il * layer_bytes;
        int blocks_left = blocks_per_layer;
        int block_i = 0;

        while (blocks_left > 0) {
            int batch = std::min(queue_cap, blocks_left);
            for (int i = 0; i < batch; i++) {
                DSTORAGE_REQUEST req = {};
                req.Options.SourceType        = DSTORAGE_REQUEST_SOURCE_FILE;
                req.Options.DestinationType   = DSTORAGE_REQUEST_DESTINATION_BUFFER;
                req.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
                req.Source.File.Source         = dsFile;
                req.Source.File.Offset        = layer_offset + (uint64_t)(block_i + i) * bs;
                req.Source.File.Size          = (UINT32)bs;
                req.Destination.Buffer.Resource = gpuBuf[buf_idx];
                req.Destination.Buffer.Offset   = (UINT64)(block_i + i) * bs;
                req.Destination.Buffer.Size     = (UINT32)bs;
                req.UncompressedSize            = (UINT32)bs;
                queue->EnqueueRequest(&req);
            }
            block_i     += batch;
            blocks_left -= batch;

            if (blocks_left > 0) {
                queue->EnqueueSignal(d3dFence, fence_val);
                queue->Submit();
                // CPU wait for mid-layer drain (DS queue capacity constraint)
                if (d3dFence->GetCompletedValue() < fence_val) {
                    HANDLE evt = CreateEvent(NULL, FALSE, FALSE, NULL);
                    d3dFence->SetEventOnCompletion(fence_val, evt);
                    WaitForSingleObject(evt, INFINITE);
                    CloseHandle(evt);
                }
                fence_val++;
            }
        }

        // Final fence signal for this layer
        uint64_t layer_fence_val = fence_val++;
        queue->EnqueueSignal(d3dFence, layer_fence_val);
        queue->Submit();
        return layer_fence_val;
    };

    // Pre-load layer 0 into buffer 0
    auto t_pipeline_start = std::chrono::high_resolution_clock::now();
    auto t_submit = std::chrono::high_resolution_clock::now();
    uint64_t pending_fence = submit_layer_dma(0, 0);

    for (int il = 0; il < n_layers; il++) {
        int cur_buf = il % 2;

        // 1. GPU-side wait: CUDA stream waits for DirectStorage fence
        auto t_wait_start = std::chrono::high_resolution_clock::now();
        {
            cudaExternalSemaphoreWaitParams waitParams = {};
            waitParams.params.fence.value = pending_fence;
            cudaWaitExternalSemaphoresAsync(&extSem, &waitParams, 1, cudaStream);
            cudaStreamSynchronize(cudaStream);  // for timing accuracy
        }
        auto t_wait_end = std::chrono::high_resolution_clock::now();

        wait_times[il] = std::chrono::duration<double, std::milli>(t_wait_end - t_wait_start).count();
        dma_times[il]  = std::chrono::duration<double, std::milli>(t_wait_end - t_submit).count();
        total_bytes_dma += layer_bytes;

        // 2. Start loading NEXT layer into alternate buffer
        if (il + 1 < n_layers) {
            t_submit = std::chrono::high_resolution_clock::now();
            pending_fence = submit_layer_dma(il + 1, (il + 1) % 2);
        }

        // 3. Simulate compute on current buffer
        //    In real pipeline: CUDA kernels run on cudaDevPtr[cur_buf]
        if (g_compute_ms > 0.0f) {
            auto t_compute_start = std::chrono::high_resolution_clock::now();
            auto deadline = t_compute_start + std::chrono::microseconds((int)(g_compute_ms * 1000.0f));
            while (std::chrono::high_resolution_clock::now() < deadline) {
                YieldProcessor();
            }
            auto t_compute_end = std::chrono::high_resolution_clock::now();
            compute_times[il] = std::chrono::duration<double, std::milli>(t_compute_end - t_compute_start).count();
        }

        if (g_trace) {
            fprintf(stderr, "    layer %3d: buf[%d] cuda=%p dma=%.1fms wait=%.1fms compute=%.1fms\n",
                    il, cur_buf, cudaDevPtr[cur_buf], dma_times[il], wait_times[il], compute_times[il]);
        }
    }

    auto t_pipeline_end = std::chrono::high_resolution_clock::now();
    double pipeline_elapsed = std::chrono::duration<double>(t_pipeline_end - t_pipeline_start).count();

    // --- Error check ---
    DSTORAGE_ERROR_RECORD ds_err = {};
    queue->RetrieveErrorRecord(&ds_err);
    if (FAILED(ds_err.FirstFailure.HResult)) {
        fprintf(stderr, "  WARNING: DirectStorage error: 0x%08lx\n",
                (unsigned long)ds_err.FirstFailure.HResult);
    }

    // --- Aggregate stats ---
    double total_gb = (double)total_bytes_dma / (1024.0 * 1024.0 * 1024.0);
    double bw       = total_gb / pipeline_elapsed;

    double sum_dma = 0, max_dma = 0, min_dma = 1e9;
    double sum_wait = 0, sum_compute = 0;
    int    stall_count = 0;

    for (int il = 0; il < n_layers; il++) {
        sum_dma     += dma_times[il];
        sum_wait    += wait_times[il];
        sum_compute += compute_times[il];
        if (dma_times[il] > max_dma) max_dma = dma_times[il];
        if (dma_times[il] < min_dma) min_dma = dma_times[il];
        if (il > 0 && wait_times[il] > 5.0) stall_count++;  // 5ms: ignore fence overhead, catch real DMA stalls
    }
    double avg_dma     = sum_dma / n_layers;
    double avg_wait    = sum_wait / n_layers;
    double avg_compute = g_compute_ms > 0 ? (sum_compute / n_layers) : 0;

    double overlap_eff = 0.0;
    if (g_compute_ms > 0 && n_layers > 1) {
        double stall_after_0 = sum_wait - wait_times[0];
        double total_dma_after_0 = sum_dma - dma_times[0];
        if (total_dma_after_0 > 0) {
            overlap_eff = 1.0 - (stall_after_0 / total_dma_after_0);
            if (overlap_eff < 0) overlap_eff = 0;
        }
    }

    double dma_only_tks = 0;
    if (avg_dma > 0) {
        dma_only_tks = 8192.0 / (n_layers * avg_dma / 1000.0);
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "  ====== DIRECTSTORAGE + CUDA INTEROP RESULTS ======\n");
    fprintf(stderr, "  Layers:         %d x %.2f MB = %.2f GB\n",
            n_layers, layer_bytes / (1024.0 * 1024.0), total_gb);
    fprintf(stderr, "  Block size:     %d KiB\n", g_block_size_kb);
    fprintf(stderr, "  Pipeline time:  %.3f s\n", pipeline_elapsed);
    fprintf(stderr, "  Bandwidth:      %.2f GB/s\n", bw);
    fprintf(stderr, "  Per-layer DMA:  avg=%.1fms  min=%.1fms  max=%.1fms\n",
            avg_dma, min_dma, max_dma);
    fprintf(stderr, "  Per-layer wait: avg=%.1fms (CUDA GPU-side fence wait)\n", avg_wait);
    if (g_compute_ms > 0) {
        fprintf(stderr, "  Simulated compute: %.1fms/layer  overlap=%.0f%%  stalls=%d/%d\n",
                avg_compute, overlap_eff * 100.0, stall_count, n_layers - 1);
    }
    fprintf(stderr, "  Sync path:      DirectStorage → D3D12 fence → CUDA extSemaphore (GPU-side)\n");
    fprintf(stderr, "  CUDA ptrs:      buf[0]=%p  buf[1]=%p\n", cudaDevPtr[0], cudaDevPtr[1]);
    fprintf(stderr, "  DMA ceiling:    %.0f tok/s @ 8K ctx\n", dma_only_tks);
    fprintf(stderr, "  ---\n");
    fprintf(stderr, "  Model estimates (8K context):\n");
    double est_27b_dma = (440.0 * 1024 * 1024) / (bw * 1024 * 1024 * 1024) * 1000.0;
    double est_27b_bottleneck = std::max(170.0, est_27b_dma);
    double est_27b_tks = 8192.0 / (48 * est_27b_bottleneck / 1000.0);
    fprintf(stderr, "    27B Q6_K (48L, 440MB/L): DMA=%.0fms  compute=170ms  -> %.0f tok/s\n",
            est_27b_dma, est_27b_tks);
    double est_70b_dma = (550.0 * 1024 * 1024) / (bw * 1024 * 1024 * 1024) * 1000.0;
    double est_70b_bottleneck = std::max(200.0, est_70b_dma);
    double est_70b_tks = 8192.0 / (80 * est_70b_bottleneck / 1000.0);
    fprintf(stderr, "    70B Q4_K (80L, 550MB/L): DMA=%.0fms  compute=200ms  -> %.0f tok/s\n",
            est_70b_dma, est_70b_tks);
    fprintf(stderr, "  ====================================================\n");

    // --- Cleanup ---
    queue->Release();
    cudaStreamDestroy(cudaStream);
    cudaDestroyExternalSemaphore(extSem);
    CloseHandle(fenceSharedHandle);
    d3dFence->Release();
    for (int b = 0; b < 2; b++) {
        cudaDestroyExternalMemory(extMem[b]);
        CloseHandle(sharedHandles[b]);
        gpuBuf[b]->Release();
        if (gpuHeap[b]) gpuHeap[b]->Release();
    }
    dsFile->Release();
    factory->Release();
    device->Release();
    dxgiFactory->Release();
    if (com_inited) CoUninitialize();

    return true;
}

#endif // LLAMA_CUDA_INTEROP

// ---------------------------------------------------------------------------
// Test: DirectStorage weight fetcher pipeline (wf API)
//
// Exercises the full wf_create → wf_start → wf_wait_layer_gpu pipeline
// with WF_MODE_DSTORAGE.  Verifies that the DirectStorage backend in
// llama-weight-fetcher.cpp (wf_dstorage_init, wf_dstorage_submit_layer,
// fence sync, pipeline thread) works end-to-end.
//
// Activated with: --dstorage-wf <path>
// ---------------------------------------------------------------------------

#if defined(LLAMA_CUDA_INTEROP)

static bool test_dstorage_wf_pipeline_once(const char * path, int cycle) {
    fprintf(stderr, "test_dstorage_wf_pipeline: cycle %d/%d with DirectStorage (%s)\n",
            cycle + 1, g_wf_cycles, path);

    const int    n_layers = g_nlayers;
    const size_t bs       = (size_t)g_block_size_kb * 1024;

    fprintf(stderr, "  layers=%d  bs=%dK  compute-ms=%.1f\n", n_layers, g_block_size_kb, g_compute_ms);

    // --- Get file size ---
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "  ERROR: cannot open %s\n", path);
        return false;
    }
#ifdef _WIN32
    _fseeki64(f, 0, SEEK_END);
    size_t file_size = (size_t)_ftelli64(f);
#else
    fseek(f, 0, SEEK_END);
    size_t file_size = (size_t)ftell(f);
#endif
    fclose(f);

    const size_t data_offset = 16 * 1024 * 1024;  // skip GGUF header
    size_t usable = file_size > data_offset ? file_size - data_offset : 0;
    size_t layer_size = (usable / n_layers) & ~((size_t)65535);  // 64KB aligned for D3D12
    if (g_wf_layer_mb > 0) {
        size_t cap = ((size_t) g_wf_layer_mb * 1024 * 1024) & ~((size_t) 65535);
        layer_size = std::min(layer_size, cap);
    }
    if (layer_size == 0) {
        fprintf(stderr, "  ERROR: file too small for %d layers\n", n_layers);
        return false;
    }
    size_t total_data = (size_t)n_layers * layer_size;
    double total_gb = total_data / (1024.0 * 1024.0 * 1024.0);

    fprintf(stderr, "  File size:    %.2f GB\n", file_size / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "  Layer size:   %.2f MB (%d x %dK blocks)\n",
            layer_size / (1024.0 * 1024.0), (int)((layer_size + bs - 1) / bs), g_block_size_kb);
    fprintf(stderr, "  Total:        %d layers x %.2f MB = %.2f GB\n",
            n_layers, layer_size / (1024.0 * 1024.0), total_gb);

    // --- Create wf context with DSTORAGE mode ---
    wf_params p = wf_params_default();
    p.model_path      = path;
    p.n_layers        = n_layers;
    p.max_layer_bytes = layer_size;
    p.mode            = WF_MODE_DSTORAGE;
    p.io_block_size   = bs;
    p.io_threads      = g_threads;
    p.io_queue_depth  = g_queue_depth;
    p.n_gpu_buffers   = g_gpu_bufs;
    p.n_ram_buffers   = g_ram_bufs;
    p.prefetch_depth  = g_gpu_bufs;  // match GPU buffer count (no RAM staging)
    p.trace           = g_trace;
    p.post_mode       = WF_POST_KEEP_FETCHER;

    wf_context * wf = wf_create(&p);
    if (!wf) {
        fprintf(stderr, "  ERROR: wf_create failed (DirectStorage init may have failed)\n");
        return false;
    }

    // Check effective mode
    wf_transfer_mode mode = wf_get_effective_mode(wf);
    fprintf(stderr, "  Effective mode: %s\n",
            mode == WF_MODE_DSTORAGE ? "DirectStorage" :
            mode == WF_MODE_DIRECT_IO ? "Direct I/O (DS fallback)" :
            mode == WF_MODE_MMAP ? "mmap" : "unknown");

    if (mode != WF_MODE_DSTORAGE) {
        fprintf(stderr, "  WARNING: DirectStorage not active — test will still run but via fallback\n");
    }

    // Production supplies ggml-owned ping-pong buffers through
    // wf_set_gpu_buf_ptr(). This standalone driver has no ggml backend, so
    // create equivalent external CUDA targets explicitly.
    std::vector<void *> compute_bufs(g_gpu_bufs, nullptr);
    for (int bi = 0; bi < g_gpu_bufs; ++bi) {
        cudaError_t alloc_err = cudaMalloc(&compute_bufs[bi], layer_size);
        if (alloc_err != cudaSuccess) {
            fprintf(stderr, "  ERROR: cudaMalloc compute buffer %d failed: %s\n",
                    bi, cudaGetErrorString(alloc_err));
            wf_destroy(wf);
            for (void * ptr : compute_bufs) if (ptr) cudaFree(ptr);
            return false;
        }
        wf_set_gpu_buf_ptr(wf, bi, compute_bufs[bi]);
    }

    // --- Register layers ---
    for (int il = 0; il < n_layers; il++) {
        wf_weight_span span;
        memset(&span, 0, sizeof(span));
        span.tensor_name = "layer_weights";
        span.file_offset = data_offset + il * layer_size;
        span.nbytes      = layer_size;
        span.file_idx    = 0;

        wf_register_layer(wf, il, &span, 1, layer_size);
    }

    fprintf(stderr, "\n  === Running wf pipeline (DirectStorage backend) ===\n");

    // --- Run pipeline ---
    auto t_start = std::chrono::high_resolution_clock::now();
    wf_start(wf);

    std::vector<double> wait_times(n_layers, 0);
    int stall_count = 0;

    for (int il = 0; il < n_layers; il++) {
        auto tw0 = std::chrono::high_resolution_clock::now();

        void * ptr = wf_wait_layer_gpu(wf, il);
        if (!ptr) {
            fprintf(stderr, "  ERROR: wf_wait_layer_gpu returned NULL for layer %d\n", il);
            wf_stop(wf);
            wf_destroy(wf);
            for (void * compute_buf : compute_bufs) cudaFree(compute_buf);
            return false;
        }

        auto tw1 = std::chrono::high_resolution_clock::now();
        double wait_ms = std::chrono::duration<double, std::milli>(tw1 - tw0).count();
        wait_times[il] = wait_ms;

        // GPU-side fence sync (ensures DMA complete before compute)
        wf_cuda_wait_upload(wf, il);

        // Force a CUDA read from the transferred allocation. This makes the
        // D3D12-to-CUDA ordering path observable to Compute Sanitizer instead
        // of merely testing host-side fence completion.
        unsigned char sample[16] = {};
        cudaError_t sample_err = cudaMemcpy(sample, ptr, sizeof(sample), cudaMemcpyDeviceToHost);
        if (sample_err != cudaSuccess) {
            fprintf(stderr, "  ERROR: cycle %d layer %d CUDA sample failed: %s\n",
                    cycle + 1, il, cudaGetErrorString(sample_err));
            wf_stop(wf);
            wf_destroy(wf);
            for (void * compute_buf : compute_bufs) cudaFree(compute_buf);
            return false;
        }

        // Simulated compute
        if (g_compute_ms > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds((int)(g_compute_ms * 1000)));
        }

        wf_release_layer(wf, il);

        if (il > 0 && wait_ms > 5.0) stall_count++;

        if (g_trace || (il < 3) || (il == n_layers - 1)) {
            fprintf(stderr, "    layer %d: wait=%.2fms  ptr=%p\n", il, wait_ms, ptr);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double pipeline_s = std::chrono::duration<double>(t_end - t_start).count();
    double bw = total_gb / pipeline_s;

    // Stats
    double sum_wait = 0, min_wait = 1e9, max_wait = 0;
    for (int il = 0; il < n_layers; il++) {
        sum_wait += wait_times[il];
        if (wait_times[il] < min_wait) min_wait = wait_times[il];
        if (wait_times[il] > max_wait) max_wait = wait_times[il];
    }
    double avg_wait = sum_wait / n_layers;

    fprintf(stderr, "\n");
    fprintf(stderr, "  ====== WF PIPELINE + DIRECTSTORAGE RESULTS ======\n");
    fprintf(stderr, "  Layers:         %d x %.2f MB = %.2f GB\n",
            n_layers, layer_size / (1024.0 * 1024.0), total_gb);
    fprintf(stderr, "  Block size:     %d KiB\n", g_block_size_kb);
    fprintf(stderr, "  GPU buffers:    %d (ping-pong)\n", g_gpu_bufs);
    fprintf(stderr, "  Pipeline time:  %.3f s\n", pipeline_s);
    fprintf(stderr, "  Bandwidth:      %.2f GB/s\n", bw);
    fprintf(stderr, "  Per-layer wait: avg=%.1fms  min=%.1fms  max=%.1fms\n",
            avg_wait, min_wait, max_wait);
    if (g_compute_ms > 0) {
        fprintf(stderr, "  Simulated compute: %.1fms/layer  stalls=%d/%d\n",
                g_compute_ms, stall_count, n_layers - 1);
    }
    fprintf(stderr, "  Mode:           %s\n",
            mode == WF_MODE_DSTORAGE ? "DirectStorage (NVMe → D3D12 → CUDA)" : "fallback");
    fprintf(stderr, "  ====================================================\n");

    // Print wf stats
    wf_print_stats(wf);

    // Cleanup
    wf_stop(wf);
    wf_destroy(wf);
    for (void * compute_buf : compute_bufs) cudaFree(compute_buf);

    fprintf(stderr, "  PASS cycle %d/%d\n", cycle + 1, g_wf_cycles);
    return true;
}

static bool test_dstorage_wf_pipeline(const char * path) {
    for (int cycle = 0; cycle < g_wf_cycles; ++cycle) {
        if (!test_dstorage_wf_pipeline_once(path, cycle)) return false;
    }
    return true;
}

#endif // LLAMA_CUDA_INTEROP

#endif // _WIN32 && LLAMA_DIRECTSTORAGE

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * split_path = nullptr;

    bool raw_io_mode = false;
    const char * raw_io_path = nullptr;
    bool dstorage_mode = false;
    const char * dstorage_path = nullptr;
    bool dstorage_gpu_mode = false;
    const char * dstorage_gpu_path = nullptr;
    bool dstorage_prefill_mode = false;
    const char * dstorage_prefill_path = nullptr;
    bool dstorage_cuda_mode = false;
    const char * dstorage_cuda_path = nullptr;
    bool dstorage_wf_mode = false;
    const char * dstorage_wf_path = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--trace") == 0) {
            g_trace = true;
        } else if (strcmp(argv[i], "--dstorage-wf") == 0) {
            dstorage_wf_mode = true;
            if (i + 1 < argc && argv[i+1][0] != '-') dstorage_wf_path = argv[++i];
        } else if (strcmp(argv[i], "--dstorage-cuda") == 0) {
            dstorage_cuda_mode = true;
            if (i + 1 < argc && argv[i+1][0] != '-') dstorage_cuda_path = argv[++i];
        } else if (strcmp(argv[i], "--dstorage-prefill") == 0) {
            dstorage_prefill_mode = true;
            if (i + 1 < argc && argv[i+1][0] != '-') dstorage_prefill_path = argv[++i];
        } else if (strcmp(argv[i], "--compute-ms") == 0) {
            if (i + 1 < argc) g_compute_ms = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--dstorage-gpu") == 0) {
            dstorage_gpu_mode = true;
            if (i + 1 < argc && argv[i+1][0] != '-') dstorage_gpu_path = argv[++i];
        } else if (strcmp(argv[i], "--dstorage") == 0) {
            dstorage_mode = true;
            if (i + 1 < argc && argv[i+1][0] != '-') dstorage_path = argv[++i];
        } else if (strcmp(argv[i], "--raw-io") == 0) {
            raw_io_mode = true;
            if (i + 1 < argc && argv[i+1][0] != '-') raw_io_path = argv[++i];
        } else if (strcmp(argv[i], "--model") == 0 || strcmp(argv[i], "-m") == 0) {
            if (i + 1 < argc) model_path = argv[++i];
        } else if (strcmp(argv[i], "--split") == 0 || strcmp(argv[i], "-s") == 0) {
            if (i + 1 < argc) split_path = argv[++i];
        } else if (strcmp(argv[i], "--bs") == 0) {
            if (i + 1 < argc) g_block_size_kb = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 || strcmp(argv[i], "-t") == 0) {
            if (i + 1 < argc) g_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--qd") == 0 || strcmp(argv[i], "-q") == 0) {
            if (i + 1 < argc) g_queue_depth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ram-bufs") == 0) {
            if (i + 1 < argc) g_ram_bufs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gpu-bufs") == 0) {
            if (i + 1 < argc) g_gpu_bufs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--nlayers") == 0 || strcmp(argv[i], "-n") == 0) {
            if (i + 1 < argc) g_nlayers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--wf-cycles") == 0) {
            if (i + 1 < argc) g_wf_cycles = std::max(1, atoi(argv[++i]));
        } else if (strcmp(argv[i], "--wf-layer-mb") == 0) {
            if (i + 1 < argc) g_wf_layer_mb = std::max(0, atoi(argv[++i]));
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            fprintf(stderr, "Usage: %s [options]\n", argv[0]);
            fprintf(stderr, "  --split <path>   First split GGUF file\n");
            fprintf(stderr, "  --model <path>   Single GGUF file\n");
            fprintf(stderr, "  --bs <KiB>       Block size in KiB (default: %d)\n", g_block_size_kb);
            fprintf(stderr, "  --threads <N>    I/O threads (default: %d)\n", g_threads);
            fprintf(stderr, "  --qd <N>         Queue depth per handle (default: %d)\n", g_queue_depth);
            fprintf(stderr, "  --ram-bufs <N>   Pinned RAM buffers (default: %d)\n", g_ram_bufs);
            fprintf(stderr, "  --gpu-bufs <N>   GPU staging buffers (default: %d)\n", g_gpu_bufs);
            fprintf(stderr, "  --nlayers <N>    Layers per file (fewer = bigger reads, default: %d)\n", g_nlayers);
            fprintf(stderr, "  --wf-cycles <N>  Recreate WF N times to exercise persistent DS reuse (default: %d)\n", g_wf_cycles);
            fprintf(stderr, "  --wf-layer-mb <N> Cap each --dstorage-wf layer to N MiB (default: unlimited)\n");
            fprintf(stderr, "  --raw-io <path>  Raw IOCP read benchmark (no GPU pipeline)\n");
#if defined(LLAMA_DIRECTSTORAGE)
            fprintf(stderr, "  --dstorage <path>         DirectStorage read benchmark (BypassIO → RAM)\n");
            fprintf(stderr, "  --dstorage-gpu <path>     DirectStorage GPU-direct (NVMe → GPU VRAM, zero RAM)\n");
            fprintf(stderr, "  --dstorage-prefill <path> 2-buffer prefill pipeline (NVMe → GPU overlap test)\n");
            fprintf(stderr, "  --dstorage-cuda <path>    D3D12-CUDA interop pipeline (NVMe → D3D12 → CUDA)\n");
            fprintf(stderr, "  --dstorage-wf <path>      Weight fetcher pipeline via DirectStorage backend\n");
            fprintf(stderr, "  --compute-ms <ms>         Simulated per-layer compute time (default: 0)\n");
#endif
            fprintf(stderr, "  --trace          Enable verbose logging\n");
            return 0;
        }
    }

    fprintf(stderr, "=== Weight Fetcher Tests ===\n");
    fprintf(stderr, "  I/O config: bs=%dK  threads=%d  qd=%d  ram=%d  gpu=%d  nlayers=%d\n\n",
            g_block_size_kb, g_threads, g_queue_depth, g_ram_bufs, g_gpu_bufs, g_nlayers);

#ifdef _WIN32
    // Raw I/O benchmark — runs INSTEAD of normal tests
    if (raw_io_mode) {
        const char * io_path = raw_io_path ? raw_io_path : (split_path ? split_path : model_path);
        if (!io_path) {
            fprintf(stderr, "ERROR: --raw-io needs a file path (or use with --split/--model)\n");
            return 1;
        }
        fprintf(stderr, "\n--- Raw IOCP I/O Benchmark ---\n\n");
        test_raw_io(io_path);
        return 0;
    }
#else
    if (raw_io_mode) {
        fprintf(stderr, "ERROR: --raw-io only supported on Windows (IOCP)\n");
        return 1;
    }
#endif

    // DirectStorage memory benchmark — runs INSTEAD of normal tests
    if (dstorage_mode) {
#if defined(_WIN32) && defined(LLAMA_DIRECTSTORAGE)
        const char * ds_path = dstorage_path ? dstorage_path : (split_path ? split_path : model_path);
        if (!ds_path) {
            fprintf(stderr, "ERROR: --dstorage needs a file path (or use with --split/--model)\n");
            return 1;
        }
        fprintf(stderr, "\n--- DirectStorage Benchmark (memory destination) ---\n\n");
        test_directstorage_io(ds_path);
        return 0;
#else
        fprintf(stderr, "ERROR: --dstorage requires Windows build with -DLLAMA_DIRECTSTORAGE=ON\n");
        fprintf(stderr, "  Run setup_directstorage.bat first, then reconfigure CMake.\n");
        return 1;
#endif
    }

    // DirectStorage GPU-direct benchmark — NVMe → GPU VRAM, zero RAM
    if (dstorage_gpu_mode) {
#if defined(_WIN32) && defined(LLAMA_DIRECTSTORAGE)
        const char * dsg_path = dstorage_gpu_path ? dstorage_gpu_path : (split_path ? split_path : model_path);
        if (!dsg_path) {
            fprintf(stderr, "ERROR: --dstorage-gpu needs a file path (or use with --split/--model)\n");
            return 1;
        }
        fprintf(stderr, "\n--- DirectStorage GPU-Direct Benchmark (NVMe → GPU VRAM) ---\n\n");
        test_directstorage_gpu(dsg_path);
        return 0;
#else
        fprintf(stderr, "ERROR: --dstorage-gpu requires Windows build with -DLLAMA_DIRECTSTORAGE=ON\n");
        fprintf(stderr, "  Run setup_directstorage.bat first, then reconfigure CMake.\n");
        return 1;
#endif
    }

    // DirectStorage 2-buffer prefill pipeline test — NVMe → GPU VRAM with overlap
    if (dstorage_prefill_mode) {
#if defined(_WIN32) && defined(LLAMA_DIRECTSTORAGE)
        const char * dsp_path = dstorage_prefill_path ? dstorage_prefill_path : (split_path ? split_path : model_path);
        if (!dsp_path) {
            fprintf(stderr, "ERROR: --dstorage-prefill needs a file path (or use with --split/--model)\n");
            return 1;
        }
        fprintf(stderr, "\n--- DirectStorage 2-Buffer Prefill Pipeline (NVMe → GPU VRAM overlap) ---\n\n");
        test_dstorage_prefill(dsp_path);
        return 0;
#else
        fprintf(stderr, "ERROR: --dstorage-prefill requires Windows build with -DLLAMA_DIRECTSTORAGE=ON\n");
        fprintf(stderr, "  Run setup_directstorage.bat first, then reconfigure CMake.\n");
        return 1;
#endif
    }

    // DirectStorage + CUDA interop — D3D12 → CUDA device pointers with GPU-side fence sync
    if (dstorage_cuda_mode) {
#if defined(_WIN32) && defined(LLAMA_DIRECTSTORAGE) && defined(LLAMA_CUDA_INTEROP)
        const char * dsc_path = dstorage_cuda_path ? dstorage_cuda_path : (split_path ? split_path : model_path);
        if (!dsc_path) {
            fprintf(stderr, "ERROR: --dstorage-cuda needs a file path (or use with --split/--model)\n");
            return 1;
        }
        fprintf(stderr, "\n--- DirectStorage + CUDA Interop (NVMe → D3D12 → CUDA) ---\n\n");
        test_dstorage_cuda_interop(dsc_path);
        return 0;
#else
        fprintf(stderr, "ERROR: --dstorage-cuda requires Windows build with DirectStorage + CUDA\n");
        fprintf(stderr, "  Need: -DLLAMA_DIRECTSTORAGE=ON (run setup_directstorage.bat) + -DGGML_CUDA=ON\n");
        return 1;
#endif
    }

    // DirectStorage weight fetcher pipeline — wf_create → wf_start → wf_wait_layer_gpu with DS backend
    if (dstorage_wf_mode) {
#if defined(_WIN32) && defined(LLAMA_DIRECTSTORAGE) && defined(LLAMA_CUDA_INTEROP)
        const char * dsw_path = dstorage_wf_path ? dstorage_wf_path : (split_path ? split_path : model_path);
        if (!dsw_path) {
            fprintf(stderr, "ERROR: --dstorage-wf needs a file path (or use with --split/--model)\n");
            return 1;
        }
        fprintf(stderr, "\n--- Weight Fetcher Pipeline + DirectStorage Backend ---\n\n");
        test_dstorage_wf_pipeline(dsw_path);
        return 0;
#else
        fprintf(stderr, "ERROR: --dstorage-wf requires Windows build with DirectStorage + CUDA\n");
        fprintf(stderr, "  Need: -DLLAMA_DIRECTSTORAGE=ON (run setup_directstorage.bat) + -DGGML_CUDA=ON\n");
        return 1;
#endif
    }

    // Core unit tests (always run)
    test_create_destroy();
    test_register_layers();
    test_mmap_fallback();
    test_direct_io();
    test_moe_filtering();
    test_pipeline_throughput();
    test_pass_reset();

    // Real GGUF test (only if --model is provided)
    if (model_path) {
        fprintf(stderr, "\n--- Real GGUF Weight Fetch ---\n\n");
        test_real_gguf(model_path);
    } else {
        fprintf(stderr, "\nSkipping real GGUF test (pass --model <path> to enable)\n");
    }

    // Split GGUF test (only if --split is provided)
    if (split_path) {
        fprintf(stderr, "\n--- Split GGUF Full-Model Read ---\n\n");
        test_real_gguf_split(split_path);
    } else {
        fprintf(stderr, "\nSkipping split GGUF test (pass --split <first-split-path> to enable)\n");
    }

    fprintf(stderr, "\n=== Results: %d/%d passed, %d failed ===\n",
            g_tests_passed, g_tests_run, g_tests_failed);

    return g_tests_failed > 0 ? 1 : 0;
}
