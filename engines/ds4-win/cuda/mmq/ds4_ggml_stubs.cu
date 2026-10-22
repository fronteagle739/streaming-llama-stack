// SPDX-License-Identifier: MIT
// Implementations of the ggml-API stubs declared in ds4_ggml_stubs.h plus
// the bodies of ggml_backend_cuda_context / ggml_cuda_info /
// new_pool_for_device that the vendored common.cuh declares without
// defining.
//
// CUDA uses stream-ordered allocation. HIP retains stream-local allocations
// because returning live MMQ scratch to its asynchronous pool is unreliable.

#include "common.cuh"   // pulls in ds4_ggml_stubs.h via redirect headers

#if defined(GGML_USE_HIP)
#include "vendors/hip.h"
#include "../../ds4_rocm_memory.h"
#else
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

#if defined(_WIN32)
#include "../../ds4_win32.h"
#endif

// ----------------------------------------------------------------------------
// Device info singleton.
//
// Common.cuh declares `const ggml_cuda_device_info & ggml_cuda_info();` -
// we provide the body. The struct layout (common.cuh:1091) is:
//   { int device_count; cuda_device_info devices[GGML_CUDA_MAX_DEVICES];
//     std::array<float, GGML_CUDA_MAX_DEVICES> default_tensor_split; }
// where cuda_device_info has { cc, nsm, smpb, smpbo, integrated, vmm,
// vmm_granularity, total_vram, warp_size, supports_cooperative_launch }.
// ----------------------------------------------------------------------------

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info;
    static std::once_flag once;
    std::call_once(once, []{
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess) {
            fprintf(stderr, "ggml_cuda_info: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
            count = 0;
        }
        if (count > GGML_CUDA_MAX_DEVICES) count = GGML_CUDA_MAX_DEVICES;
        info.device_count = count;
        for (int i = 0; i < count; i++) {
            cudaDeviceProp p;
            CUDA_CHECK(cudaGetDeviceProperties(&p, i));
#if defined(GGML_USE_HIP)
            unsigned gfx_id = 0;
            if (sscanf(p.gcnArchName, "gfx%x", &gfx_id) == 1) {
                info.devices[i].cc = GGML_CUDA_CC_OFFSET_AMD + (int)gfx_id;
            } else {
                info.devices[i].cc = GGML_CUDA_CC_OFFSET_AMD + p.major * 0x100 + p.minor * 0x10;
            }
#else
            info.devices[i].cc                          = p.major * 100 + p.minor * 10;
#endif
            info.devices[i].nsm                         = p.multiProcessorCount;
            info.devices[i].smpb                        = p.sharedMemPerBlock;
#if defined(GGML_USE_HIP)
            info.devices[i].smpbo                       = p.sharedMemPerBlock;
#else
            info.devices[i].smpbo                       = p.sharedMemPerBlockOptin;
#endif
            info.devices[i].integrated                  = p.integrated != 0;
            info.devices[i].vmm                         = false;
            info.devices[i].vmm_granularity             = 0;
            info.devices[i].total_vram                  = p.totalGlobalMem;
            info.devices[i].warp_size                   = p.warpSize;
            info.devices[i].supports_cooperative_launch = p.cooperativeLaunch != 0;
        }
    });
    return info;
}

// ggml_cuda_get_device / ggml_cuda_set_device are declared (not defined) in
// common.cuh. We provide thin wrappers.

int ggml_cuda_get_device() {
    int dev = 0;
    (void)cudaGetDevice(&dev);
    return dev;
}

void ggml_cuda_set_device(int device) {
    int cur = -1;
    (void)cudaGetDevice(&cur);
    if (cur != device) {
        CUDA_CHECK(cudaSetDevice(device));
    }
}

int64_t ggml_time_us() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();
}

// ----------------------------------------------------------------------------
// ggml_cuda_error: invoked by the CUDA_CHECK / CUBLAS_CHECK macros defined
// in common.cuh on the error path. Marked [[noreturn]] in the declaration
// (common.cuh:155) - abort() satisfies that contract.
// ----------------------------------------------------------------------------

[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "CUDA error: %s\n  call: %s\n  in: %s at %s:%d\n", msg, stmt, func, file, line);
    fflush(stderr);
    abort();
}

// ----------------------------------------------------------------------------
// Concrete device scratch pool.
// ----------------------------------------------------------------------------

namespace {

/* Thread-local stream that ds4_naive_pool uses for cudaMallocAsync /
 * cudaFreeAsync.  Defaults to cudaStreamPerThread (preserves prior
 * behaviour).  Step 8 / CUDA Graphs sets this to the capture stream
 * before allocating, so the alloc node lives on the captured stream
 * and capture is not invalidated.  Set via ds4_pool_set_stream() from
 * ds4_mmq.cu wrappers; an explicit stream=0 means the legacy default
 * stream so pool ops stay ordered with legacy-stream kernels. */
static thread_local cudaStream_t t_ds4_pool_stream = cudaStreamPerThread;

} // anonymous namespace

extern "C" void ds4_pool_set_stream(cudaStream_t stream) {
    t_ds4_pool_stream = stream;
}

extern "C" cudaStream_t ds4_pool_get_stream(void) {
    return t_ds4_pool_stream;
}

namespace {

static bool ds4_pool_trace_enabled() {
    const char *value = getenv("DS4_MMQ_POOL_TRACE");
    return value && value[0] == '1' && value[1] == '\0';
}

#if !defined(GGML_USE_HIP)
/* Query backing separately from requested/used bytes. Async frees can leave
 * backing retained by the default pool, outside logical allocation totals.
 * These observations do not synchronize, trim or configure the allocator. */
static cudaError_t ds4_pool_statistics(int device, cudaMemPool_t *pool, uint64_t stats[4],
                                      int status[4]) {
    for (unsigned i = 0; i < 4; ++i) { stats[i] = 0; status[i] = -1; }
    *pool = nullptr;
    const cudaError_t result = cudaDeviceGetDefaultMemPool(pool, device);
    if (result != cudaSuccess) return result;
    const cudaMemPoolAttr attributes[] = {
        cudaMemPoolAttrReservedMemCurrent, cudaMemPoolAttrReservedMemHigh,
        cudaMemPoolAttrUsedMemCurrent, cudaMemPoolAttrUsedMemHigh};
    for (unsigned i = 0; i < 4; ++i)
        status[i] = (int)cudaMemPoolGetAttribute(*pool, attributes[i], &stats[i]);
    return result;
}

static void ds4_pool_trace(const char *phase, int device, cudaStream_t stream,
                          void *ptr, size_t requested, size_t reported_actual,
                          cudaError_t allocation_status) {
    if (!ds4_pool_trace_enabled()) return;
    cudaMemPool_t pool;
    uint64_t stats[4];
    int status[4];
    const cudaError_t pool_status = ds4_pool_statistics(device, &pool, stats, status);
    fprintf(stderr,
        "ds4: MMQ_POOL_TRACE phase=%s allocator=cudaMallocAsync/default "
        "device=%d stream=%p ptr=%p requested_bytes=%llu reported_actual_bytes=%llu "
        "operation_status=%d default_pool=%p pool_status=%d "
        "reserved_current=%llu reserved_high=%llu used_current=%llu used_high=%llu "
        "attribute_status=%d,%d,%d,%d\n",
        phase, device, (void *)stream, ptr, (unsigned long long)requested,
        (unsigned long long)reported_actual, (int)allocation_status, (void *)pool,
        (int)pool_status, (unsigned long long)stats[0], (unsigned long long)stats[1],
        (unsigned long long)stats[2], (unsigned long long)stats[3],
        status[0], status[1], status[2], status[3]);
    fflush(stderr);
#if defined(_WIN32)
    ds4_win32_host_memory_snapshot(phase);
#endif
}
#endif

struct ds4_naive_pool : public ggml_cuda_pool {
    int device;

    explicit ds4_naive_pool(int device) : device(device) {}

#if defined(GGML_USE_HIP)
    struct cached_block {
        void *ptr;
        size_t size;
        cudaStream_t stream;
    };

    std::mutex mutex;
    std::vector<cached_block> available;
    std::vector<void *> owned;
#endif

    void * alloc(size_t size, size_t * actual_size) override {
        ggml_cuda_set_device(device);
#if defined(GGML_USE_HIP)
        {
            std::lock_guard<std::mutex> guard(mutex);
            size_t best = available.size();
            for (size_t i = 0; i < available.size(); i++) {
                if (available[i].stream != t_ds4_pool_stream ||
                    available[i].size < size) {
                    continue;
                }
                if (best == available.size() ||
                    available[i].size < available[best].size) {
                    best = i;
                }
            }
            if (best != available.size()) {
                const cached_block block = available[best];
                available[best] = available.back();
                available.pop_back();
                if (actual_size) *actual_size = block.size;
                return block.ptr;
            }
        }

        void * ptr = nullptr;
        CUDA_CHECK(ds4_rocm_allocation_fits(size) ? hipSuccess : hipErrorOutOfMemory);
        CUDA_CHECK(cudaMallocAsync(&ptr, size, t_ds4_pool_stream));
        {
            std::lock_guard<std::mutex> guard(mutex);
            owned.push_back(ptr);
        }
        if (actual_size) *actual_size = size;
        return ptr;
#else
        void * ptr = nullptr;
        ds4_pool_trace("MMQ-alloc-before", device, t_ds4_pool_stream, ptr, size, 0, cudaSuccess);
#if defined(_WIN32)
        if (!ds4_win32_host_memory_check_commit(size, "MMQ cudaMallocAsync backing store")) {
            fprintf(stderr, "ds4: HOST_MEMORY_ADMISSION_REFUSED MMQ async scratch bytes=%llu\n",
                    (unsigned long long)size);
            fflush(stderr);
            ExitProcess(72);
        }
        if (size >= 64u * 1048576u) {
            fprintf(stderr, "ds4: HOST_MEMORY_ALLOCATION kind=MMQ_cudaMallocAsync bytes=%llu\n",
                    (unsigned long long)size);
            ds4_win32_host_memory_snapshot("MMQ cudaMallocAsync");
        }
#endif
        const cudaError_t result = cudaMallocAsync(&ptr, size, t_ds4_pool_stream);
        ds4_pool_trace("MMQ-alloc-after", device, t_ds4_pool_stream, ptr,
                       size, result == cudaSuccess ? size : 0, result);
        CUDA_CHECK(result);
        if (actual_size) *actual_size = size;
        return ptr;
#endif
    }

    void free(void * ptr, size_t size) override {
        if (!ptr) return;
        ggml_cuda_set_device(device);
#if defined(GGML_USE_HIP)
        std::lock_guard<std::mutex> guard(mutex);
        available.push_back({ptr, size, t_ds4_pool_stream});
#else
        ds4_pool_trace("MMQ-free-before", device, t_ds4_pool_stream, ptr, size, size, cudaSuccess);
        const cudaError_t result = cudaFreeAsync(ptr, t_ds4_pool_stream);
        ds4_pool_trace("MMQ-free-after", device, t_ds4_pool_stream, ptr, size, size, result);
        CUDA_CHECK(result);
#endif
    }

#if defined(GGML_USE_HIP)
    ~ds4_naive_pool() override {
        ggml_cuda_set_device(device);
        (void)cudaDeviceSynchronize();
        for (void *ptr : owned) (void)cudaFree(ptr);
    }
#endif
};

} // anonymous namespace

/* Default-off model-free test boundary. A trusted fixture must explicitly
 * enable it and establish the source-enforced <=4 GiB Windows private cap
 * before these helpers initialize CUDA or allocate real MMQ scratch. */
static bool ds4_pool_fixture_enabled() {
#if defined(_WIN32) && !defined(GGML_USE_HIP)
    const char *value = getenv("DS4_MMQ_POOL_TEST");
    return value && value[0] == '1' && value[1] == '\0';
#else
    return false;
#endif
}

extern "C" void *ds4_mmq_pool_test_create(int device) {
    if (!ds4_pool_fixture_enabled() || device < 0) return nullptr;
#if defined(_WIN32) && !defined(GGML_USE_HIP)
    ds4_win32_host_memory_status status;
    if (!ds4_win32_host_memory_begin() || !ds4_win32_host_memory_read(&status) ||
        status.process_limit_bytes > (UINT64_C(4) << 30)) return nullptr;
    return new ds4_naive_pool(device);
#else
    return nullptr;
#endif
}

extern "C" void *ds4_mmq_pool_test_alloc(void *pool, size_t bytes, void *stream, size_t *actual) {
    if (!ds4_pool_fixture_enabled() || !pool || !bytes) return nullptr;
    ds4_pool_set_stream((cudaStream_t)stream);
    return ((ds4_naive_pool *)pool)->alloc(bytes, actual);
}

extern "C" int ds4_mmq_pool_test_free(void *pool, void *ptr, size_t bytes, void *stream) {
    if (!ds4_pool_fixture_enabled() || !pool || !ptr) return 0;
    ds4_pool_set_stream((cudaStream_t)stream);
    ((ds4_naive_pool *)pool)->free(ptr, bytes);
    return 1;
}

extern "C" int ds4_mmq_pool_test_stats(int device, uint64_t stats[4]) {
    if (!ds4_pool_fixture_enabled() || !stats) return 0;
#if !defined(GGML_USE_HIP)
    cudaMemPool_t pool;
    int status[4];
    if (ds4_pool_statistics(device, &pool, stats, status) != cudaSuccess) return 0;
    for (unsigned i = 0; i < 4; ++i) if (status[i] != (int)cudaSuccess) return 0;
    return 1;
#else
    return 0;
#endif
}

extern "C" int ds4_mmq_pool_test_destroy(void *pool) {
    if (!ds4_pool_fixture_enabled() || !pool) return 0;
    delete (ds4_naive_pool *)pool;
    return 1;
}

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int device, int /*stream_no*/) {
    return std::unique_ptr<ggml_cuda_pool>(new ds4_naive_pool(device));
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    if (copy_event) {
        (void)cudaEventDestroy(copy_event);
        copy_event = nullptr;
    }
    // streams[][], cublas_handles[], and pools[][] are owned-by-value
    // (cudaStream_t and cublasHandle_t are opaque handles - destroying the
    // context "should" tear them down, but in our shim ds4 manages streams
    // externally and we leave them alone. The pools auto-destruct via
    // unique_ptr.).
}
