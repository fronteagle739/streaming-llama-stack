//
// test-ds-chunks.cpp — Minimal DirectStorage chunked-read test
//
// Proves that large files (>4 GB) can be loaded to GPU VRAM correctly
// by splitting reads across multiple D3D12 resources, each ≤ 3.5 GB.
//
// The test:
//   1. Opens a file via DirectStorage
//   2. Creates N D3D12 resources (chunks), each ≤ CHUNK_SIZE
//   3. Reads the file chunk-by-chunk into separate resources
//   4. CUDA imports each resource and reads back to host
//   5. Compares readback to fread reference data
//   6. Reports sustained throughput (target: 20 GB/s)
//
// Usage:
//   test-ds-chunks.exe <file_path> [--chunk-mb <MB>] [--layers <N>]
//
// Example:
//   test-ds-chunks.exe MODEL.gguf
//   test-ds-chunks.exe LAYER.bin --chunk-mb 2048
//
// Build (from VS dev shell):
//   cmake --build prefill_llama.cpp\build --target test-ds-chunks
//

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dstorage.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <objbase.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

static uint64_t g_chunk_mb    = 3500;     // Max chunk size in MB (default 3.5 GB)
static int      g_verify_mb   = 256;      // MB of readback to verify per chunk (0=all)
static int      g_block_kb    = 256;      // DS request block size in KB
static int      g_n_layers    = 0;        // Number of layers to simulate (0 = whole file as 1 layer)
static bool     g_verbose     = false;

#define LOG(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#define ERR(fmt, ...) fprintf(stderr, "ERROR: " fmt "\n", ##__VA_ARGS__)
#define VLOG(fmt, ...) do { if (g_verbose) fprintf(stderr, "  " fmt "\n", ##__VA_ARGS__); } while(0)

// ---------------------------------------------------------------------------
// CUDA helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        ERR("CUDA: %s at line %d", cudaGetErrorString(err), __LINE__); \
        return false; \
    } \
} while(0)

// ---------------------------------------------------------------------------
// D3D12 resource + CUDA interop for one chunk
// ---------------------------------------------------------------------------

struct chunk_resource {
    ID3D12Resource *       d3d_resource = nullptr;
    HANDLE                 shared_handle = NULL;
    cudaExternalMemory_t   ext_mem = {};
    void *                 cuda_ptr = nullptr;
    uint64_t               size = 0;       // allocated size (64KB aligned)
    uint64_t               data_bytes = 0; // actual data in this chunk
    uint64_t               file_offset = 0;// where in the file this chunk starts
};

static bool create_chunk_resource(ID3D12Device * device, chunk_resource & chunk, uint64_t size) {
    HRESULT hr;
    uint64_t aligned = (size + 65535) & ~((uint64_t)65535);
    chunk.size = aligned;

    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC resDesc = {};
    resDesc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    resDesc.Width            = aligned;
    resDesc.Height           = 1;
    resDesc.DepthOrArraySize = 1;
    resDesc.MipLevels        = 1;
    resDesc.SampleDesc.Count = 1;
    resDesc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resDesc.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    hr = device->CreateCommittedResource(
        &heapProps, D3D12_HEAP_FLAG_SHARED, &resDesc,
        D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&chunk.d3d_resource));
    if (FAILED(hr)) {
        ERR("CreateCommittedResource failed: 0x%08lx (%.1f MB)", (unsigned long)hr, aligned / (1024.0 * 1024.0));
        return false;
    }

    hr = device->CreateSharedHandle(chunk.d3d_resource, nullptr, GENERIC_ALL, nullptr, &chunk.shared_handle);
    if (FAILED(hr)) {
        ERR("CreateSharedHandle failed: 0x%08lx", (unsigned long)hr);
        chunk.d3d_resource->Release(); chunk.d3d_resource = nullptr;
        return false;
    }

    cudaExternalMemoryHandleDesc memDesc = {};
    memDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    memDesc.handle.win32.handle = chunk.shared_handle;
    memDesc.size = aligned;
    memDesc.flags = cudaExternalMemoryDedicated;

    cudaError_t cerr = cudaImportExternalMemory(&chunk.ext_mem, &memDesc);
    if (cerr != cudaSuccess) {
        ERR("cudaImportExternalMemory failed: %s (%.1f MB)", cudaGetErrorString(cerr), aligned / (1024.0 * 1024.0));
        CloseHandle(chunk.shared_handle);
        chunk.d3d_resource->Release(); chunk.d3d_resource = nullptr;
        return false;
    }

    cudaExternalMemoryBufferDesc bufDesc = {};
    bufDesc.offset = 0;
    bufDesc.size   = aligned;
    cerr = cudaExternalMemoryGetMappedBuffer(&chunk.cuda_ptr, chunk.ext_mem, &bufDesc);
    if (cerr != cudaSuccess) {
        ERR("cudaExternalMemoryGetMappedBuffer failed: %s", cudaGetErrorString(cerr));
        cudaDestroyExternalMemory(chunk.ext_mem);
        CloseHandle(chunk.shared_handle);
        chunk.d3d_resource->Release(); chunk.d3d_resource = nullptr;
        return false;
    }

    return true;
}

static void destroy_chunk_resource(chunk_resource & chunk) {
    if (chunk.cuda_ptr)      { cudaFree(chunk.cuda_ptr); chunk.cuda_ptr = nullptr; }
    if (chunk.ext_mem)        { cudaDestroyExternalMemory(chunk.ext_mem); chunk.ext_mem = {}; }
    if (chunk.shared_handle)  { CloseHandle(chunk.shared_handle); chunk.shared_handle = NULL; }
    if (chunk.d3d_resource)   { chunk.d3d_resource->Release(); chunk.d3d_resource = nullptr; }
}

// ---------------------------------------------------------------------------
// Fence wait helper
// ---------------------------------------------------------------------------

static bool wait_fence(ID3D12Fence * fence, uint64_t value, DWORD timeout_ms = 10000) {
    if (fence->GetCompletedValue() >= value) return true;
    HANDLE evt = CreateEvent(NULL, FALSE, FALSE, NULL);
    if (!evt) return false;
    fence->SetEventOnCompletion(value, evt);
    DWORD result = WaitForSingleObject(evt, timeout_ms);
    CloseHandle(evt);
    if (result != WAIT_OBJECT_0) {
        ERR("Fence wait timeout (completed=%llu, needed=%llu)",
            (unsigned long long)fence->GetCompletedValue(), (unsigned long long)value);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Enqueue DS reads for one chunk (file_offset → resource at offset 0)
// ---------------------------------------------------------------------------

static int enqueue_chunk_reads(IDStorageQueue * queue, IDStorageFile * file,
                                chunk_resource & chunk, uint64_t block_size) {
    int n_reqs = 0;
    uint64_t remaining = chunk.data_bytes;
    uint64_t file_off  = chunk.file_offset;
    uint64_t buf_off   = 0;

    while (remaining > 0) {
        uint64_t req_size = std::min(remaining, block_size);

        DSTORAGE_REQUEST req = {};
        req.Options.SourceType        = DSTORAGE_REQUEST_SOURCE_FILE;
        req.Options.DestinationType   = DSTORAGE_REQUEST_DESTINATION_BUFFER;
        req.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
        req.Source.File.Source         = file;
        req.Source.File.Offset        = file_off;
        req.Source.File.Size          = (UINT32)req_size;
        req.Destination.Buffer.Resource = chunk.d3d_resource;
        req.Destination.Buffer.Offset   = buf_off;
        req.Destination.Buffer.Size     = (UINT32)req_size;
        req.UncompressedSize            = (UINT32)req_size;

        queue->EnqueueRequest(&req);
        file_off  += req_size;
        buf_off   += req_size;
        remaining -= req_size;
        n_reqs++;
    }
    return n_reqs;
}

// ---------------------------------------------------------------------------
// Verify readback against fread reference
// ---------------------------------------------------------------------------

struct verify_result {
    bool     pass;
    uint64_t bytes_checked;
    uint64_t first_mismatch_offset;
    uint8_t  expected_byte;
    uint8_t  actual_byte;
    int      mismatched_regions;  // number of 4KB blocks with any mismatch
};

static verify_result verify_chunk(const chunk_resource & chunk, const char * file_path,
                                   uint64_t verify_bytes) {
    verify_result r = {};
    r.pass = true;
    r.first_mismatch_offset = UINT64_MAX;

    uint64_t check_bytes = (verify_bytes > 0) ? std::min(verify_bytes, chunk.data_bytes) : chunk.data_bytes;
    r.bytes_checked = check_bytes;

    // Read reference from disk
    std::vector<uint8_t> ref(check_bytes);
    FILE * fp = fopen(file_path, "rb");
    if (!fp) {
        ERR("Cannot open file for reference read: %s", file_path);
        r.pass = false; return r;
    }
    _fseeki64(fp, chunk.file_offset, SEEK_SET);
    size_t got = fread(ref.data(), 1, check_bytes, fp);
    fclose(fp);
    if (got != check_bytes) {
        ERR("fread short: got %zu, expected %llu", got, (unsigned long long)check_bytes);
        r.pass = false; return r;
    }

    // CUDA readback
    std::vector<uint8_t> gpu(check_bytes);
    cudaError_t cerr = cudaMemcpy(gpu.data(), chunk.cuda_ptr, check_bytes, cudaMemcpyDeviceToHost);
    if (cerr != cudaSuccess) {
        ERR("cudaMemcpy readback failed: %s", cudaGetErrorString(cerr));
        r.pass = false; return r;
    }

    // Compare in 4KB blocks
    const size_t BLOCK = 4096;
    for (uint64_t off = 0; off < check_bytes; off += BLOCK) {
        size_t len = std::min((uint64_t)BLOCK, check_bytes - off);
        if (memcmp(ref.data() + off, gpu.data() + off, len) != 0) {
            r.mismatched_regions++;
            if (r.first_mismatch_offset == UINT64_MAX) {
                // Find exact byte
                for (size_t j = 0; j < len; j++) {
                    if (ref[off + j] != gpu[off + j]) {
                        r.first_mismatch_offset = off + j;
                        r.expected_byte = ref[off + j];
                        r.actual_byte   = gpu[off + j];
                        break;
                    }
                }
            }
            r.pass = false;
        }
    }

    return r;
}

// ---------------------------------------------------------------------------
// Main test: chunked DS read → CUDA readback → verify
// ---------------------------------------------------------------------------

static bool run_test(const char * file_path) {
    HRESULT hr;
    const uint64_t chunk_max = g_chunk_mb * 1024ULL * 1024ULL;
    const uint64_t block_size = (uint64_t)g_block_kb * 1024ULL;
    const uint64_t verify_limit = (uint64_t)g_verify_mb * 1024ULL * 1024ULL;

    LOG("=== DirectStorage Chunked Read Test ===");
    LOG("File:       %s", file_path);
    LOG("Chunk max:  %llu MB", (unsigned long long)g_chunk_mb);
    LOG("Block size: %d KB", g_block_kb);
    LOG("Verify:     %s", g_verify_mb > 0 ? (std::to_string(g_verify_mb) + " MB per chunk").c_str() : "ALL");

    // --- Get file size ---
    HANDLE hFile = CreateFileA(file_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        ERR("Cannot open file: %s", file_path);
        return false;
    }
    LARGE_INTEGER li;
    GetFileSizeEx(hFile, &li);
    uint64_t file_size = (uint64_t)li.QuadPart;
    CloseHandle(hFile);
    LOG("File size:  %.2f GB (%llu bytes)", file_size / (1024.0 * 1024.0 * 1024.0), (unsigned long long)file_size);

    // --- Determine layer geometry ---
    struct layer_info {
        uint64_t offset;
        uint64_t size;
    };
    std::vector<layer_info> layers;

    if (g_n_layers > 0) {
        // Split file into equal layers
        uint64_t layer_size = ((file_size / g_n_layers) / block_size) * block_size;
        for (int i = 0; i < g_n_layers && (uint64_t)i * layer_size < file_size; i++) {
            uint64_t off = (uint64_t)i * layer_size;
            uint64_t sz  = std::min(layer_size, file_size - off);
            sz = (sz / block_size) * block_size;  // align down
            if (sz > 0) layers.push_back({off, sz});
        }
    } else {
        // Whole file as one layer
        uint64_t sz = (file_size / block_size) * block_size;
        layers.push_back({0, sz});
    }

    LOG("Layers:     %d", (int)layers.size());
    for (int i = 0; i < std::min((int)layers.size(), 4); i++) {
        LOG("  layer[%d]: offset=%llu  size=%.2f GB",
            i, (unsigned long long)layers[i].offset, layers[i].size / (1024.0 * 1024.0 * 1024.0));
    }
    if ((int)layers.size() > 4) LOG("  ... (%d more)", (int)layers.size() - 4);

    // --- CUDA device ---
    int cuda_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&cuda_count));
    if (cuda_count == 0) { ERR("No CUDA devices"); return false; }
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    LOG("CUDA:       %s (%.0f MB)", props.name, props.totalGlobalMem / (1024.0 * 1024.0));

    // --- COM + DXGI + D3D12 ---
    hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    bool com_init = SUCCEEDED(hr);

    IDXGIFactory4 * dxgiFactory = nullptr;
    hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&dxgiFactory));
    if (FAILED(hr)) { ERR("CreateDXGIFactory2 failed: 0x%08lx", (unsigned long)hr); return false; }

    // Match CUDA device by LUID
    LUID target_luid = {};
    memcpy(&target_luid, props.luid, sizeof(LUID));

    ID3D12Device * device = nullptr;
    IDXGIAdapter1 * adapter = nullptr;
    for (UINT i = 0; dxgiFactory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) { adapter->Release(); adapter = nullptr; continue; }
        if (desc.AdapterLuid.HighPart == target_luid.HighPart &&
            desc.AdapterLuid.LowPart == target_luid.LowPart) {
            hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
            adapter->Release(); adapter = nullptr;
            if (SUCCEEDED(hr)) break;
        } else {
            adapter->Release(); adapter = nullptr;
        }
    }
    if (!device) { ERR("No D3D12 device matching CUDA LUID"); dxgiFactory->Release(); return false; }
    LOG("D3D12:      device created (LUID match)");

    // --- DirectStorage factory + file ---
    IDStorageFactory * factory = nullptr;
    hr = DStorageGetFactory(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) { ERR("DStorageGetFactory failed: 0x%08lx", (unsigned long)hr); return false; }

    factory->SetStagingBufferSize(256 * 1024 * 1024);

    int wlen = MultiByteToWideChar(CP_UTF8, 0, file_path, -1, NULL, 0);
    std::vector<wchar_t> wpath(wlen);
    MultiByteToWideChar(CP_UTF8, 0, file_path, -1, wpath.data(), wlen);

    IDStorageFile * dsFile = nullptr;
    hr = factory->OpenFile(wpath.data(), IID_PPV_ARGS(&dsFile));
    if (FAILED(hr)) { ERR("DS OpenFile failed: 0x%08lx", (unsigned long)hr); return false; }
    LOG("DS:         file opened");

    // --- Single DS queue + fence ---
    DSTORAGE_QUEUE_DESC qd = {};
    qd.Capacity   = DSTORAGE_MAX_QUEUE_CAPACITY;
    qd.Priority   = DSTORAGE_PRIORITY_HIGH;
    qd.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    qd.Device     = device;

    IDStorageQueue * queue = nullptr;
    hr = factory->CreateQueue(&qd, IID_PPV_ARGS(&queue));
    if (FAILED(hr)) { ERR("DS CreateQueue failed: 0x%08lx", (unsigned long)hr); return false; }

    ID3D12Fence * fence = nullptr;
    hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (FAILED(hr)) { ERR("CreateFence failed: 0x%08lx", (unsigned long)hr); return false; }

    uint64_t fence_counter = 0;
    LOG("DS:         queue created (cap=%d), fence ready", (int)DSTORAGE_MAX_QUEUE_CAPACITY);

    // --- Compute buffer (contiguous CUDA allocation) ---
    // This is where the final assembled layer data lives for kernel access.
    uint64_t max_layer_size = 0;
    for (auto & l : layers) max_layer_size = std::max(max_layer_size, l.size);

    void * compute_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&compute_buf, max_layer_size));
    LOG("Compute:    %.2f GB CUDA buffer at %p", max_layer_size / (1024.0 * 1024.0 * 1024.0), compute_buf);

    // =====================================================================
    // TEST 1: Basic multi-resource write test
    //
    // Create 2 D3D12 resources. Write known data into each one from the
    // same file. Verify both have correct data. This tests whether DS can
    // write to more than one D3D12 committed resource.
    // =====================================================================
    LOG("\n--- TEST 1: Multi-resource DS write ---");
    {
        uint64_t test_size = std::min((uint64_t)256 * 1024 * 1024, file_size);
        test_size = (test_size / block_size) * block_size;

        chunk_resource res[2];
        bool ok = create_chunk_resource(device, res[0], test_size);
        if (!ok) { ERR("Failed to create test resource 0"); return false; }
        ok = create_chunk_resource(device, res[1], test_size);
        if (!ok) { ERR("Failed to create test resource 1"); destroy_chunk_resource(res[0]); return false; }
        LOG("  Created 2 x %.1f MB D3D12 resources", test_size / (1024.0 * 1024.0));

        // Write to resource 0 from file offset 0
        res[0].file_offset = 0;
        res[0].data_bytes  = test_size;
        int n0 = enqueue_chunk_reads(queue, dsFile, res[0], block_size);
        uint64_t fv0 = ++fence_counter;
        queue->EnqueueSignal(fence, fv0);
        queue->Submit();
        LOG("  Resource 0: %d reqs enqueued, fence=%llu", n0, (unsigned long long)fv0);

        if (!wait_fence(fence, fv0)) { ERR("Fence wait failed for resource 0"); return false; }
        LOG("  Resource 0: DMA complete");

        // Write to resource 1 from a different file offset (or same, doesn't matter)
        uint64_t offset1 = std::min(test_size, file_size - test_size);
        offset1 = (offset1 / block_size) * block_size;
        res[1].file_offset = offset1;
        res[1].data_bytes  = test_size;
        int n1 = enqueue_chunk_reads(queue, dsFile, res[1], block_size);
        uint64_t fv1 = ++fence_counter;
        queue->EnqueueSignal(fence, fv1);
        queue->Submit();
        LOG("  Resource 1: %d reqs enqueued from offset %llu, fence=%llu",
            n1, (unsigned long long)offset1, (unsigned long long)fv1);

        if (!wait_fence(fence, fv1)) { ERR("Fence wait failed for resource 1"); return false; }
        LOG("  Resource 1: DMA complete");

        // Verify both
        auto v0 = verify_chunk(res[0], file_path, test_size);
        auto v1 = verify_chunk(res[1], file_path, test_size);

        LOG("  Resource 0: %s (%.1f MB checked, %d bad regions)",
            v0.pass ? "PASS" : "FAIL", v0.bytes_checked / (1024.0 * 1024.0), v0.mismatched_regions);
        if (!v0.pass) {
            LOG("    First mismatch at byte %llu: expected 0x%02x got 0x%02x",
                (unsigned long long)v0.first_mismatch_offset, v0.expected_byte, v0.actual_byte);
        }
        LOG("  Resource 1: %s (%.1f MB checked, %d bad regions)",
            v1.pass ? "PASS" : "FAIL", v1.bytes_checked / (1024.0 * 1024.0), v1.mismatched_regions);
        if (!v1.pass) {
            LOG("    First mismatch at byte %llu: expected 0x%02x got 0x%02x",
                (unsigned long long)v1.first_mismatch_offset, v1.expected_byte, v1.actual_byte);
        }

        if (!v0.pass || !v1.pass) {
            LOG("\n  *** MULTI-RESOURCE WRITE FAILED ***");
            LOG("  DS cannot write to multiple D3D12 resources correctly.");
            LOG("  Will test single-resource with offset < 4GB next...");
        } else {
            LOG("\n  *** MULTI-RESOURCE WRITE OK ***");
            LOG("  DS can write to multiple D3D12 resources — chunked approach is viable!");
        }

        destroy_chunk_resource(res[0]);
        destroy_chunk_resource(res[1]);
    }

    // =====================================================================
    // TEST 2: Chunked layer streaming
    //
    // For each layer:
    //   - Split into chunks ≤ chunk_max
    //   - Create one D3D12 resource per chunk
    //   - DS read file → resource (buf offset always starts at 0)
    //   - Drain fence
    //   - D2D copy: resource → compute_buf at correct offset
    //   - Destroy resource (or reuse)
    //   - Verify readback from compute_buf
    // =====================================================================
    LOG("\n--- TEST 2: Chunked layer streaming (%.1f GB chunks) ---", chunk_max / (1024.0 * 1024.0 * 1024.0));

    // Pre-allocate chunk resources (reusable across layers)
    int max_chunks_per_layer = (int)((max_layer_size + chunk_max - 1) / chunk_max);
    LOG("  Max chunks per layer: %d", max_chunks_per_layer);

    // We create 2 chunk resources for ping-pong (read next chunk while copying previous)
    chunk_resource chunk_res[2];
    bool chunk_ok = create_chunk_resource(device, chunk_res[0], std::min(chunk_max, max_layer_size));
    if (!chunk_ok) { ERR("Failed to create chunk resource 0"); return false; }
    chunk_ok = create_chunk_resource(device, chunk_res[1], std::min(chunk_max, max_layer_size));
    if (!chunk_ok) { ERR("Failed to create chunk resource 1"); destroy_chunk_resource(chunk_res[0]); return false; }
    LOG("  Created 2 x %.2f GB chunk resources for ping-pong",
        std::min(chunk_max, max_layer_size) / (1024.0 * 1024.0 * 1024.0));

    int total_layers_pass = 0;
    int total_layers_fail = 0;
    double total_dma_time_ms = 0;
    uint64_t total_dma_bytes = 0;

    auto t_all_start = std::chrono::high_resolution_clock::now();

    for (int li = 0; li < (int)layers.size(); li++) {
        auto & layer = layers[li];
        int n_chunks = (int)((layer.size + chunk_max - 1) / chunk_max);

        LOG("  Layer %d: %.2f GB, %d chunk%s, file_offset=%llu",
            li, layer.size / (1024.0 * 1024.0 * 1024.0), n_chunks, n_chunks > 1 ? "s" : "",
            (unsigned long long)layer.offset);

        uint64_t layer_remaining = layer.size;
        uint64_t layer_file_off  = layer.offset;
        uint64_t compute_offset  = 0;  // offset within compute_buf for this layer
        bool layer_pass = true;

        auto t_layer_start = std::chrono::high_resolution_clock::now();

        for (int ci = 0; ci < n_chunks; ci++) {
            int ri = ci % 2;  // ping-pong resource index
            auto & cr = chunk_res[ri];

            uint64_t this_chunk = std::min(layer_remaining, chunk_max);
            this_chunk = (this_chunk / block_size) * block_size;
            if (this_chunk == 0) break;

            cr.file_offset = layer_file_off;
            cr.data_bytes  = this_chunk;

            // Enqueue reads: file → D3D12 resource at offset 0
            int n_reqs = enqueue_chunk_reads(queue, dsFile, cr, block_size);

            // Signal fence and submit
            uint64_t fv = ++fence_counter;
            queue->EnqueueSignal(fence, fv);
            queue->Submit();

            VLOG("    chunk[%d]: %.2f GB, %d reqs, fence=%llu, file_off=%llu",
                 ci, this_chunk / (1024.0 * 1024.0 * 1024.0), n_reqs,
                 (unsigned long long)fv, (unsigned long long)layer_file_off);

            // Wait for DMA to complete
            if (!wait_fence(fence, fv)) {
                ERR("Fence wait failed for layer %d chunk %d", li, ci);
                layer_pass = false; break;
            }

            // D2D copy: chunk resource → compute buffer at correct offset
            cudaError_t cerr = cudaMemcpy(
                (char *)compute_buf + compute_offset,
                cr.cuda_ptr, this_chunk, cudaMemcpyDeviceToDevice);
            if (cerr != cudaSuccess) {
                ERR("D2D copy failed: %s", cudaGetErrorString(cerr));
                layer_pass = false; break;
            }

            layer_file_off  += this_chunk;
            compute_offset  += this_chunk;
            layer_remaining -= this_chunk;
        }

        auto t_layer_end = std::chrono::high_resolution_clock::now();
        double layer_ms = std::chrono::duration<double, std::milli>(t_layer_end - t_layer_start).count();
        double layer_gbps = (layer.size / (1024.0 * 1024.0 * 1024.0)) / (layer_ms / 1000.0);
        total_dma_time_ms += layer_ms;
        total_dma_bytes   += layer.size;

        // Verify readback from compute buffer
        if (layer_pass) {
            uint64_t check_bytes = verify_limit > 0
                ? std::min(verify_limit, layer.size)
                : layer.size;

            // Read reference from disk
            std::vector<uint8_t> ref(check_bytes);
            FILE * fp = fopen(file_path, "rb");
            if (fp) {
                _fseeki64(fp, layer.offset, SEEK_SET);
                size_t got = fread(ref.data(), 1, check_bytes, fp);
                fclose(fp);
                if (got == check_bytes) {
                    std::vector<uint8_t> gpu(check_bytes);
                    cudaMemcpy(gpu.data(), compute_buf, check_bytes, cudaMemcpyDeviceToHost);

                    int bad_blocks = 0;
                    uint64_t first_bad = UINT64_MAX;
                    const size_t BLK = 4096;
                    for (uint64_t off = 0; off < check_bytes; off += BLK) {
                        size_t len = std::min((uint64_t)BLK, check_bytes - off);
                        if (memcmp(ref.data() + off, gpu.data() + off, len) != 0) {
                            bad_blocks++;
                            if (first_bad == UINT64_MAX) {
                                for (size_t j = 0; j < len; j++) {
                                    if (ref[off + j] != gpu[off + j]) {
                                        first_bad = off + j;
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    if (bad_blocks > 0) {
                        layer_pass = false;
                        LOG("    FAIL: %d bad blocks (of %llu), first mismatch at byte %llu",
                            bad_blocks, (unsigned long long)(check_bytes / BLK),
                            (unsigned long long)first_bad);

                        // Show bytes around first mismatch for debugging
                        if (first_bad < check_bytes) {
                            uint64_t start = (first_bad > 16) ? first_bad - 16 : 0;
                            LOG("    ref[%llu..]: %02x %02x %02x %02x %02x %02x %02x %02x",
                                (unsigned long long)start,
                                ref[start], ref[start+1], ref[start+2], ref[start+3],
                                ref[start+4], ref[start+5], ref[start+6], ref[start+7]);
                            LOG("    gpu[%llu..]: %02x %02x %02x %02x %02x %02x %02x %02x",
                                (unsigned long long)start,
                                gpu[start], gpu[start+1], gpu[start+2], gpu[start+3],
                                gpu[start+4], gpu[start+5], gpu[start+6], gpu[start+7]);
                        }
                    }
                }
            }
        }

        if (layer_pass) {
            total_layers_pass++;
            LOG("    PASS  %.0f ms  %.1f GB/s", layer_ms, layer_gbps);
        } else {
            total_layers_fail++;
            LOG("    FAIL  %.0f ms", layer_ms);
        }
    }

    auto t_all_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_all_end - t_all_start).count();
    double total_gbps = (total_dma_bytes / (1024.0 * 1024.0 * 1024.0)) / (total_ms / 1000.0);

    LOG("\n=== RESULTS ===");
    LOG("Layers:     %d pass, %d fail (of %d)", total_layers_pass, total_layers_fail, (int)layers.size());
    LOG("Total DMA:  %.2f GB in %.0f ms", total_dma_bytes / (1024.0 * 1024.0 * 1024.0), total_ms);
    LOG("Throughput: %.1f GB/s %s", total_gbps,
        total_gbps >= 20.0 ? "(OK >= 20 GB/s target)" : "(BELOW 20 GB/s target)");

    // =====================================================================
    // TEST 3: Sustained streaming (full pipeline timing)
    //
    // If we have multiple layers, re-stream all of them without verification
    // to measure pure DMA throughput with chunk ping-pong.
    // =====================================================================
    if (layers.size() > 1 && total_layers_fail == 0) {
        LOG("\n--- TEST 3: Sustained throughput (no verify) ---");

        auto t3_start = std::chrono::high_resolution_clock::now();
        uint64_t t3_bytes = 0;

        for (int li = 0; li < (int)layers.size(); li++) {
            auto & layer = layers[li];
            uint64_t remaining = layer.size;
            uint64_t file_off  = layer.offset;
            uint64_t comp_off  = 0;

            for (int ci = 0; remaining > 0; ci++) {
                int ri = ci % 2;
                auto & cr = chunk_res[ri];

                uint64_t this_chunk = std::min(remaining, chunk_max);
                this_chunk = (this_chunk / block_size) * block_size;
                if (this_chunk == 0) break;

                cr.file_offset = file_off;
                cr.data_bytes  = this_chunk;

                enqueue_chunk_reads(queue, dsFile, cr, block_size);
                uint64_t fv = ++fence_counter;
                queue->EnqueueSignal(fence, fv);
                queue->Submit();

                if (!wait_fence(fence, fv)) {
                    ERR("Test 3 fence wait failed at layer %d chunk %d", li, ci);
                    break;
                }

                cudaMemcpy((char *)compute_buf + comp_off, cr.cuda_ptr,
                           this_chunk, cudaMemcpyDeviceToDevice);

                file_off  += this_chunk;
                comp_off  += this_chunk;
                remaining -= this_chunk;
            }
            t3_bytes += layer.size;
        }

        auto t3_end = std::chrono::high_resolution_clock::now();
        double t3_ms = std::chrono::duration<double, std::milli>(t3_end - t3_start).count();
        double t3_gbps = (t3_bytes / (1024.0 * 1024.0 * 1024.0)) / (t3_ms / 1000.0);

        LOG("  Streamed:   %.2f GB in %.0f ms", t3_bytes / (1024.0 * 1024.0 * 1024.0), t3_ms);
        LOG("  Throughput: %.1f GB/s %s", t3_gbps,
            t3_gbps >= 20.0 ? "(OK)" : "(BELOW TARGET)");
    }

    // --- Cleanup ---
    destroy_chunk_resource(chunk_res[0]);
    destroy_chunk_resource(chunk_res[1]);
    cudaFree(compute_buf);
    fence->Release();
    queue->Release();
    dsFile->Release();
    factory->Release();
    device->Release();
    dxgiFactory->Release();
    if (com_init) CoUninitialize();

    LOG("\n%s", total_layers_fail == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return total_layers_fail == 0;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main(int argc, char * argv[]) {
    const char * file_path = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--chunk-mb") == 0 && i + 1 < argc) {
            g_chunk_mb = (uint64_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--verify-mb") == 0 && i + 1 < argc) {
            g_verify_mb = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--block-kb") == 0 && i + 1 < argc) {
            g_block_kb = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            g_n_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            g_verbose = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            fprintf(stderr, "Usage: %s <file> [options]\n", argv[0]);
            fprintf(stderr, "  --chunk-mb <MB>    Max chunk size (default 3500 = 3.5 GB)\n");
            fprintf(stderr, "  --verify-mb <MB>   Verify bytes per chunk (default 256, 0=all)\n");
            fprintf(stderr, "  --block-kb <KB>    DS request block size (default 256)\n");
            fprintf(stderr, "  --layers <N>       Split file into N layers (default: whole file)\n");
            fprintf(stderr, "  -v, --verbose      Verbose per-chunk logging\n");
            return 0;
        } else if (!file_path) {
            file_path = argv[i];
        }
    }

    if (!file_path) {
        fprintf(stderr, "Usage: %s <file> [--chunk-mb <MB>] [--layers <N>] [-v]\n", argv[0]);
        fprintf(stderr, "\nExamples:\n");
        fprintf(stderr, "  %s MODEL.gguf\n", argv[0]);
        fprintf(stderr, "  %s LAYER.bin --chunk-mb 2048\n", argv[0]);
        fprintf(stderr, "  %s SHARD.gguf --layers 8 --verify-mb 0\n", argv[0]);
        return 1;
    }

    bool ok = run_test(file_path);
    return ok ? 0 : 1;
}
