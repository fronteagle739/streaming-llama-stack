/* DirectStorage transport for the CUDA SSD-streaming expert path. See ds4_dstorage.h for rationale.
 *
 * Compiled by cl as C++17 and kept out of the nvcc translation unit deliberately: dstorage.h pulls
 * in d3d12.h/unknwn.h, which nvcc has no reason to parse.
 */
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <io.h>
#include <dxgi1_4.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <algorithm>

#include <dstorage.h>
#include <cuda_runtime_api.h>

#include "ds4_dstorage.h"

namespace {

/* Batch bookkeeping counts physical requests, after span splitting. Queue
 * entries are submitted automatically at half capacity (or explicitly by Submit),
 * and EnqueueRequest can block when full. Keep self-test waves below half of
 * physical capacity; production planners use conservative logical capacity.
 * https://learn.microsoft.com/windows/win32/dstorage/dstorage/nf-dstorage-idstoragequeue-enqueuerequest
 */
UINT16 kQueueCapacity = DSTORAGE_MAX_QUEUE_CAPACITY;
DSTORAGE_PRIORITY g_demand_priority = DSTORAGE_PRIORITY_HIGH;
uint32_t g_completion_spin_us = 0;
uint32_t g_read_parts = 8;
uint32_t g_requested_read_parts = 8;

/* Qualified split-read default; 0/1 restore one request per expert span.
 * Parse before any queue is created; malformed opt-ins fail initialization. */
bool latency_controls_from_env(void) {
    struct Control { const char *name; unsigned lo, hi, value; } controls[] = {
        {"DS4_DS_QUEUE_CAPACITY", DSTORAGE_MIN_QUEUE_CAPACITY, DSTORAGE_MAX_QUEUE_CAPACITY,
         DSTORAGE_MAX_QUEUE_CAPACITY},
        {"DS4_DS_COMPLETION_SPIN_US", 0, 10000, 0},
        {"DS4_DS_READ_PARTS", 0, 64, g_requested_read_parts}
    };
    for (auto &c : controls) {
        const char *s = getenv(c.name);
        if (!s || !*s) continue;
        char *end = NULL;
        const unsigned long v = strtoul(s, &end, 10);
        if (*s < '0' || *s > '9' || end == s || !end || *end || v < c.lo || v > c.hi) {
            fprintf(stderr, "ds4: invalid %s=%s (expected %u..%u)\n", c.name,s,c.lo,c.hi);
            return false;
        }
        c.value = (unsigned)v;
    }
    if ((controls[0].value & (controls[0].value - 1)) != 0) {
        fprintf(stderr,"ds4: DS4_DS_QUEUE_CAPACITY must be a power of two\n");
        return false;
    }
    kQueueCapacity = (UINT16)controls[0].value;
    g_completion_spin_us = controls[1].value;
    g_read_parts = controls[2].value <= 1 ? 1 : controls[2].value;
    g_demand_priority = DSTORAGE_PRIORITY_HIGH;
    const char *priority = getenv("DS4_DS_PRIORITY");
    if (priority && *priority) {
        if (!strcmp(priority,"normal")) g_demand_priority = DSTORAGE_PRIORITY_NORMAL;
        else if (!strcmp(priority,"high")) g_demand_priority = DSTORAGE_PRIORITY_HIGH;
        else if (!strcmp(priority,"realtime")) g_demand_priority = DSTORAGE_PRIORITY_REALTIME;
        else if (!strcmp(priority,"low")) g_demand_priority = DSTORAGE_PRIORITY_LOW;
        else { fprintf(stderr,"ds4: invalid DS4_DS_PRIORITY=%s\n",priority);return false; }
    }
    return true;
}

DWORD wait_for_demand(HANDLE event) {
    if (g_completion_spin_us) {
        LARGE_INTEGER start, frequency, current;
        QueryPerformanceFrequency(&frequency);
        QueryPerformanceCounter(&start);
        const LONGLONG ticks = (frequency.QuadPart * g_completion_spin_us) / 1000000;
        do {
            const DWORD result = WaitForSingleObject(event, 0);
            if (result != WAIT_TIMEOUT) return result;
            YieldProcessor();
            QueryPerformanceCounter(&current);
        } while (current.QuadPart - start.QuadPart < ticks);
    }
    return WaitForSingleObject(event, 30000);
}

typedef HRESULT(WINAPI *PFN_DStorageGetFactory)(REFIID riid, void **ppv);
typedef HRESULT(WINAPI *PFN_DStorageSetConfiguration1)(
    DSTORAGE_CONFIGURATION1 const *configuration);

HMODULE                g_dll            = NULL;
IDStorageFactory      *g_factory        = NULL;
IDStorageFile         *g_file           = NULL;
IDStorageQueue1       *g_queue          = NULL;   /* Queue1: EnqueueSetEvent lives here */
#define DS4_DS_MAX_GPU_QUEUES 8
IDStorageQueue1       *g_gpu_queues[DS4_DS_MAX_GPU_QUEUES] = {};
uint32_t               g_gpu_queue_count = 0;
uint32_t               g_gpu_queue_next  = 0;
ID3D12Device          *g_d3d_device     = NULL;
ID3D12Resource        *g_gpu_arena      = NULL;
cudaExternalMemory_t   g_gpu_extmem     = NULL;
void                  *g_gpu_cuda_ptr   = NULL;
uint64_t               g_gpu_arena_size = 0;
ID3D12Fence           *g_gpu_fence      = NULL;
cudaExternalSemaphore_t g_gpu_semaphore = NULL;
uint64_t               g_gpu_fence_next = 0;
IDStorageQueue1       *g_last_sync_gpu_queue = NULL;   /* queue of the last GPU-destination submit_wait */
#define DS4_DS_PREFILL_ARENAS 3
ID3D12Resource        *g_prefill_arenas[DS4_DS_PREFILL_ARENAS] = {};
cudaExternalMemory_t   g_prefill_extmem[DS4_DS_PREFILL_ARENAS] = {};
void                  *g_prefill_cuda_ptr[DS4_DS_PREFILL_ARENAS] = {};
uint64_t               g_prefill_arena_size[DS4_DS_PREFILL_ARENAS] = {};
uint32_t               g_prefill_arena_count = 0;
HANDLE                 g_event          = NULL;
IDStorageQueue1 *g_background_queue=NULL;
HANDLE g_background_event=NULL;
bool g_background_busy=false, g_background_saw_demand=false;
uint64_t g_background_bytes=0,g_background_submitted_total=0;
LARGE_INTEGER g_background_started={};
std::vector<DSTORAGE_REQUEST> g_background_requests;

int                    g_init_attempted = 0;
int                    g_init_ok        = 0;
uint64_t               g_file_id_low    = 0;   /* file index, to notice an fd pointing elsewhere */
uint64_t               g_file_id_high   = 0;
uint32_t               g_volume_serial  = 0;
std::vector<DSTORAGE_REQUEST> g_batch;
uint64_t               g_batch_bytes    = 0;
bool                   g_batch_gpu      = false;
uint64_t               g_file_size      = 0;   /* for scattering selftest offsets over the whole file */
int                    g_config_attempted = 0;

static uint32_t request_parts(uint64_t offset, uint32_t bytes) {
    // Preserve small/unaligned requests exactly. Never pad a production read
    // into its neighbor: only split complete 4KiB units of the original span.
    return g_read_parts > 1 && (offset & 4095u) == 0 && (bytes & 4095u) == 0 &&
           bytes / 4096u >= g_read_parts ? g_read_parts : 1;
}

static void append_request_parts(const DSTORAGE_REQUEST &original, bool gpu, uint32_t parts) {
    const uint32_t total = original.Source.File.Size;
    const uint32_t blocks = total / 4096u;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < parts; ++i) {
        DSTORAGE_REQUEST req = original;
        // Distribute remainder sectors, avoiding one disproportionately slow
        // final request when using many small subreads.
        const uint32_t bytes = parts == 1 ? total :
            (blocks / parts + (i < blocks % parts ? 1u : 0u)) * 4096u;
        req.Source.File.Offset += offset;
        req.Source.File.Size = req.UncompressedSize = bytes;
        if (gpu) {
            req.Destination.Buffer.Offset += offset;
            req.Destination.Buffer.Size = bytes;
        } else {
            req.Destination.Memory.Buffer = (char *)original.Destination.Memory.Buffer + offset;
            req.Destination.Memory.Size = bytes;
        }
        g_batch.push_back(req);
        offset += bytes;
    }
    g_batch_bytes += total;
}

uint32_t staging_mb_from_env(void) {
    uint32_t mb = 256;                       /* documented default; see ds4_dstorage.h */
    const char *env = getenv("DS4_DS_STAGING_MB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(env, &end, 10);
        if (end != env && end && *end == '\0' && v >= 8 && v <= 4096) mb = (uint32_t)v;
    }
    return mb;
}

/* Resolve the full path of the file behind a CRT fd, and its identity, so a later fd swap (the MTP
 * model does exactly this) can be detected rather than silently reading the wrong file. */
bool path_and_identity_from_fd(int fd, std::wstring &out_path,
                               uint64_t &id_low, uint64_t &id_high, uint32_t &volser) {
    intptr_t osfh = _get_osfhandle(fd);
    if (osfh == -1 || (HANDLE)osfh == INVALID_HANDLE_VALUE) return false;
    HANDLE h = (HANDLE)osfh;

    BY_HANDLE_FILE_INFORMATION info;
    memset(&info, 0, sizeof(info));
    if (!GetFileInformationByHandle(h, &info)) return false;
    id_low  = ((uint64_t)info.nFileIndexHigh << 32) | (uint64_t)info.nFileIndexLow;
    g_file_size = ((uint64_t)info.nFileSizeHigh << 32) | (uint64_t)info.nFileSizeLow;
    id_high = 0;
    volser  = (uint32_t)info.dwVolumeSerialNumber;

    DWORD need = GetFinalPathNameByHandleW(h, NULL, 0, FILE_NAME_NORMALIZED);
    if (need == 0) return false;
    out_path.assign(need + 1, L'\0');
    DWORD got = GetFinalPathNameByHandleW(h, &out_path[0], need + 1, FILE_NAME_NORMALIZED);
    if (got == 0 || got > need) return false;
    out_path.resize(got);
    return true;
}

void gpu_arena_release(void) {
    if (g_gpu_semaphore) {
        (void)cudaDestroyExternalSemaphore(g_gpu_semaphore);
        g_gpu_semaphore = NULL;
    }
    if (g_gpu_fence) {
        g_gpu_fence->Release();
        g_gpu_fence = NULL;
    }
    g_gpu_fence_next = 0;
    if (g_gpu_cuda_ptr) {
        (void)cudaFree(g_gpu_cuda_ptr);
        g_gpu_cuda_ptr = NULL;
    }
    if (g_gpu_extmem) {
        (void)cudaDestroyExternalMemory(g_gpu_extmem);
        g_gpu_extmem = NULL;
    }
    if (g_gpu_arena) {
        g_gpu_arena->Release();
        g_gpu_arena = NULL;
    }
    g_gpu_arena_size = 0;
}

void prefill_arenas_release(void) {
    for (uint32_t i = 0; i < DS4_DS_PREFILL_ARENAS; i++) {
        if (g_prefill_cuda_ptr[i]) {
            (void)cudaFree(g_prefill_cuda_ptr[i]);
            g_prefill_cuda_ptr[i] = NULL;
        }
        if (g_prefill_extmem[i]) {
            (void)cudaDestroyExternalMemory(g_prefill_extmem[i]);
            g_prefill_extmem[i] = NULL;
        }
        if (g_prefill_arenas[i]) {
            g_prefill_arenas[i]->Release();
            g_prefill_arenas[i] = NULL;
        }
        g_prefill_arena_size[i] = 0;
    }
    g_prefill_arena_count = 0;
    if (g_gpu_semaphore) {
        (void)cudaDestroyExternalSemaphore(g_gpu_semaphore);
        g_gpu_semaphore = NULL;
    }
    if (g_gpu_fence) {
        g_gpu_fence->Release();
        g_gpu_fence = NULL;
    }
    g_gpu_fence_next = 0;
}

bool create_shared_fence(void) {
    if (g_gpu_fence && g_gpu_semaphore) return true;
    if (!g_d3d_device) return false;
    HRESULT hr = g_d3d_device->CreateFence(0, D3D12_FENCE_FLAG_SHARED,
                                           __uuidof(ID3D12Fence),
                                           (void **)&g_gpu_fence);
    if (FAILED(hr) || !g_gpu_fence) return false;
    HANDLE shared = NULL;
    hr = g_d3d_device->CreateSharedHandle(g_gpu_fence, NULL, GENERIC_ALL,
                                           NULL, &shared);
    if (FAILED(hr) || !shared) return false;
    cudaExternalSemaphoreHandleDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    desc.handle.win32.handle = shared;
    const cudaError_t err = cudaImportExternalSemaphore(&g_gpu_semaphore, &desc);
    CloseHandle(shared);
    return err == cudaSuccess;
}

void release_all(void) {
    if(!ds4_ds_background_finish()) abort();
    if(g_background_queue) {g_background_queue->Release();g_background_queue=NULL;}
    if(g_background_event) {CloseHandle(g_background_event);g_background_event=NULL;}

    for (uint32_t qi = 0; qi < g_gpu_queue_count; qi++) {
        if (g_gpu_queues[qi]) { g_gpu_queues[qi]->Release(); g_gpu_queues[qi] = NULL; }
    }
    g_gpu_queue_count = 0;
    g_gpu_queue_next  = 0;
    if (g_queue)   { g_queue->Release();   g_queue = NULL; }
    if (g_file)    { g_file->Release();    g_file = NULL; }
    if (g_factory) { g_factory->Release(); g_factory = NULL; }
    if (g_event)   { CloseHandle(g_event); g_event = NULL; }
    g_batch.clear();
    g_batch_bytes = 0;
    g_batch_gpu = false;
    g_init_ok = 0;
}

/* A completion event can fire even when a request FAILED. The weight-fetcher learned this the hard
 * way: the error record must be consulted before any consumer touches the destination memory. */
bool wave_succeeded(IDStorageQueue1 *queue, size_t request_count) {
    if (!queue) return false;
    HANDLE err = queue->GetErrorEvent();
    if (err == NULL) return true;
    if (WaitForSingleObject(err, 0) != WAIT_OBJECT_0) return true;   /* not signalled -> no error */

    DSTORAGE_ERROR_RECORD rec;
    memset(&rec, 0, sizeof(rec));
    queue->RetrieveErrorRecord(&rec);
    fprintf(stderr,
            "ds4: DirectStorage request FAILED: hr=0x%08lX, %u of %u requests failed\n",
            (unsigned long)rec.FirstFailure.HResult,
            (unsigned)rec.FailureCount,
            (unsigned)request_count);
    return false;
}

bool ensure_d3d_device(void) {
    if (g_d3d_device) return true;

    int cuda_device = 0;
    cudaDeviceProp prop;
    memset(&prop, 0, sizeof(prop));
    cudaError_t cerr = cudaGetDevice(&cuda_device);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "ds4: DirectStorage GPU init: cudaGetDevice failed: %s\n",
                cudaGetErrorString(cerr));
        return false;
    }
    cerr = cudaGetDeviceProperties(&prop, cuda_device);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "ds4: DirectStorage GPU init: cudaGetDeviceProperties failed: %s\n",
                cudaGetErrorString(cerr));
        return false;
    }

    IDXGIFactory4 *factory = NULL;
    HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void **)&factory);
    if (FAILED(hr) || !factory) {
        fprintf(stderr, "ds4: DirectStorage GPU init: CreateDXGIFactory1 failed hr=0x%08lX\n",
                (unsigned long)hr);
        return false;
    }

    for (UINT i = 0; ; i++) {
        IDXGIAdapter1 *adapter = NULL;
        hr = factory->EnumAdapters1(i, &adapter);
        if (hr == DXGI_ERROR_NOT_FOUND) break;
        if (FAILED(hr) || !adapter) continue;
        DXGI_ADAPTER_DESC1 desc;
        memset(&desc, 0, sizeof(desc));
        if (SUCCEEDED(adapter->GetDesc1(&desc)) &&
            memcmp(&desc.AdapterLuid, prop.luid, sizeof(desc.AdapterLuid)) == 0) {
            hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0,
                                   __uuidof(ID3D12Device), (void **)&g_d3d_device);
            adapter->Release();
            break;
        }
        adapter->Release();
    }
    factory->Release();

    if (!g_d3d_device || FAILED(hr)) {
        fprintf(stderr,
                "ds4: DirectStorage GPU init: no D3D12 adapter matched CUDA device %d LUID\n",
                cuda_device);
        if (g_d3d_device) { g_d3d_device->Release(); g_d3d_device = NULL; }
        return false;
    }
    if (g_d3d_device->GetNodeCount() != 1) {
        fprintf(stderr, "ds4: DirectStorage GPU init: linked-node D3D12 adapters are unsupported\n");
        g_d3d_device->Release();
        g_d3d_device = NULL;
        return false;
    }
    return true;
}

uint32_t gpu_queue_count_from_env(void) {
    uint32_t n = 1;
    const char *env = getenv("DS4_DS_GPU_QUEUES");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(env, &end, 10);
        if (end != env && end && *end == 0 && v >= 1) {
            n = (v > DS4_DS_MAX_GPU_QUEUES) ? (uint32_t)DS4_DS_MAX_GPU_QUEUES : (uint32_t)v;
        }
    }
    return n;
}

bool ensure_gpu_queue(void) {
    if (g_gpu_queue_count != 0) return true;
    if (!g_factory || !ensure_d3d_device()) return false;
    DSTORAGE_QUEUE_DESC desc;
    memset(&desc, 0, sizeof(desc));
    desc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    desc.Capacity = kQueueCapacity;
    desc.Priority = g_demand_priority;
    desc.Name = "ds4-expert-stream-gpu";
    desc.Device = g_d3d_device;
    /* [2026-08-18] FIXED: every extra queue used to land in slot 1 and leak.
     *
     * The old body wrote `g_gpu_queues[g_gpu_queue_count + 1] = q`, but this function early-returns
     * when g_gpu_queue_count != 0, so at this point it is ALWAYS 0 - making the index a constant 1.
     * Each iteration therefore overwrote the previous queue's pointer (leaking the interface) and
     * g_gpu_queue_count was then set to 1 below regardless, so the extra queues were never visible
     * to anything that dispatches work.
     *
     * Net effect: DS4_DS_GPU_QUEUES (or whatever gpu_queue_count_from_env reads) has never done
     * anything except leak N-2 queue objects. Any past measurement that believed it was varying GPU
     * queue count was varying nothing - re-run those before trusting them.
     *
     * Pack from slot 1 upward and count what was actually created, so a mid-loop failure leaves a
     * dense array rather than a hole. */
    const uint32_t want = gpu_queue_count_from_env();
    uint32_t extra_created = 0;
    for (uint32_t qi = 1; qi < want && qi < (uint32_t)DS4_DS_MAX_GPU_QUEUES; qi++) {
        IDStorageQueue1 *q = NULL;
        if (SUCCEEDED(g_factory->CreateQueue(&desc, __uuidof(IDStorageQueue1), (void **)&q)) && q) {
            g_gpu_queues[1 + extra_created] = q;
            extra_created++;
        }
    }
    const HRESULT hr = g_factory->CreateQueue(
        &desc, __uuidof(IDStorageQueue1), (void **)&g_gpu_queues[0]);
    if (FAILED(hr) || !g_gpu_queues[0]) {
        fprintf(stderr, "ds4: DirectStorage GPU init: CreateQueue failed hr=0x%08lX\n",
                (unsigned long)hr);
        return false;
    }
    g_gpu_queue_count = 1;
    for (uint32_t qi = 1; qi < DS4_DS_MAX_GPU_QUEUES; qi++) {
        if (g_gpu_queues[qi]) g_gpu_queue_count = qi + 1;
    }
    return true;
}

}  /* namespace */

extern "C" int ds4_ds_available(void) { return g_init_ok; }
extern "C" int ds4_ds_configure_read_parts(uint32_t parts) {
    if(parts>64 || (g_init_attempted && parts!=g_requested_read_parts)) {
        fprintf(stderr,"ds4: read parts must be 0..64; changing live transport settings is refused\n");return 0;
    }
    if(g_init_attempted)return 1;
    g_requested_read_parts=parts;
    return latency_controls_from_env()?1:0;
}

extern "C" uint32_t ds4_ds_batch_count(void) { return (uint32_t)g_batch.size(); }
extern "C" uint64_t ds4_ds_batch_bytes(void) { return g_batch_bytes; }
extern "C" uint32_t ds4_ds_batch_capacity(void) {
    // Decode plans reserve logical tensor spans before addresses are known.
    // Budget the worst-case expansion so wide waves cannot overfill the queue.
    return (uint32_t)kQueueCapacity / g_read_parts;
}

extern "C" void ds4_ds_batch_reset(void) {
    g_batch.clear();
    g_batch_bytes = 0;
    g_batch_gpu = false;
}

extern "C" int ds4_ds_init_from_fd(int fd) {
    std::wstring path;
    uint64_t id_low = 0, id_high = 0;
    uint32_t volser = 0;
    if (!path_and_identity_from_fd(fd, path, id_low, id_high, volser)) {
        fprintf(stderr,
                "ds4: DirectStorage init: cannot resolve a path for fd %d (GetLastError=%lu)\n",
                fd, (unsigned long)GetLastError());
        return 0;
    }

    /* Already up for THIS exact file? Reuse. Different file (e.g. the MTP model swapped in)? Tear
     * down and rebuild rather than keep reading the previous one. */
    if (g_init_ok) {
        if (id_low == g_file_id_low && id_high == g_file_id_high && volser == g_volume_serial) {
            return 1;
        }
        fprintf(stderr, "ds4: DirectStorage: model file changed, re-initialising\n");
        release_all();
        g_init_attempted = 0;
    }
    if (g_init_attempted && !g_init_ok) return 0;   /* never retry a known-broken init per token */
    g_init_attempted = 1;
    if (!latency_controls_from_env()) return 0;

    if (!g_dll) {
        /* App-local load: dstorage.dll sits beside the exe and pulls dstoragecore.dll from there. */
        wchar_t exe[MAX_PATH];
        DWORD n = GetModuleFileNameW(NULL, exe, MAX_PATH);
        if (n == 0 || n >= MAX_PATH) {
            fprintf(stderr, "ds4: DirectStorage init: GetModuleFileNameW failed (%lu)\n",
                    (unsigned long)GetLastError());
            return 0;
        }
        wchar_t *slash = wcsrchr(exe, L'\\');
        if (slash) *(slash + 1) = L'\0';
        std::wstring dll = std::wstring(exe) + L"dstorage.dll";
        g_dll = LoadLibraryW(dll.c_str());
        if (!g_dll) {
            fprintf(stderr,
                    "ds4: DirectStorage init: cannot load dstorage.dll from the executable "
                    "directory (GetLastError=%lu).\n"
                    "ds4:   dstorage.dll and dstoragecore.dll must sit beside ds4.exe.\n"
                    "ds4:   Re-run windows_build.ps1, or pass --ssd-streaming-disk-io pread to "
                    "explicitly use the old serial path.\n",
                    (unsigned long)GetLastError());
            return 0;
        }
    }

    /* [2026-08-20 FIX] CONFIGURE DIRECTSTORAGE UNCONDITIONALLY.
     *
     * This block used to run only `if (threads_env && threads_env[0])`, so in the SHIPPED default -
     * where nobody sets DS4_DS_SUBMIT_THREADS - DStorageSetConfiguration1 was NEVER CALLED AT ALL.
     * Every field it governs was left at whatever the runtime chose, and we had no record of what
     * that was. Three of those fields decide whether the fast path is used:
     *
     *     ForceMappingLayer    TRUE forces the slow compatibility path
     *     DisableBypassIO      TRUE disables the kernel bypass fast path
     *     ForceFileBuffering   TRUE routes reads through the OS cache
     *
     * We want all three FALSE, and we now SAY SO explicitly rather than inheriting them. The struct
     * is zeroed first, so every field we do not name is FALSE/0 by construction.
     *
     * NumSubmitThreads: 0 means "runtime heuristic" and is the documented default here. It stays
     * overridable via DS4_DS_SUBMIT_THREADS (0..64) because decode submits many small scattered
     * batches and the best value is workload- and core-count-dependent - but the CALL no longer
     * depends on the override being present.
     *
     * Must run before the first DStorageGetFactory call. */
    if (!g_config_attempted) {
        g_config_attempted = 1;
        unsigned long threads = 0;            /* documented default: runtime heuristic */
        int threads_from_env = 0;
        const char *threads_env = getenv("DS4_DS_SUBMIT_THREADS");
        if (threads_env && threads_env[0]) {
            char *end = NULL;
            const unsigned long v = strtoul(threads_env, &end, 10);
            if (end != threads_env && end && *end == '\0' && v <= 64) {
                threads = v;
                threads_from_env = 1;
            } else {
                fprintf(stderr,
                        "ds4: ignoring invalid DS4_DS_SUBMIT_THREADS=%s (expected 0..64); "
                        "using default 0 (runtime heuristic)\n",
                        threads_env);
            }
        }
        PFN_DStorageSetConfiguration1 pSetConfiguration1 =
            (PFN_DStorageSetConfiguration1)GetProcAddress(
                g_dll, "DStorageSetConfiguration1");
        if (pSetConfiguration1) {
            DSTORAGE_CONFIGURATION1 config;
            memset(&config, 0, sizeof(config));
            config.NumSubmitThreads   = (UINT32)threads;
            config.ForceMappingLayer  = FALSE;   /* stated, not inherited */
            config.DisableBypassIO    = FALSE;   /* stated, not inherited */
            config.ForceFileBuffering = FALSE;   /* stated, not inherited */
            const HRESULT config_hr = pSetConfiguration1(&config);
            if (SUCCEEDED(config_hr)) {
                fprintf(stderr,
                        "ds4: [DS4_DS_CONFIG_V1] DirectStorage configured: submit_threads=%lu (%s) "
                        "force_mapping_layer=0 disable_bypass_io=0 force_file_buffering=0\n",
                        threads, threads_from_env ? "env" : "default");
            } else {
                fprintf(stderr,
                        "ds4: [DS4_DS_CONFIG_V1] SetConfiguration1(threads=%lu) FAILED hr=0x%08lX; "
                        "runtime defaults retained and NOT known\n",
                        threads, (unsigned long)config_hr);
            }
        } else {
            fprintf(stderr,
                    "ds4: [DS4_DS_CONFIG_V1] runtime has no DStorageSetConfiguration1; "
                    "bypass-IO/mapping-layer/file-buffering state is the runtime's and NOT known\n");
        }
    }

    PFN_DStorageGetFactory pGetFactory =
        (PFN_DStorageGetFactory)GetProcAddress(g_dll, "DStorageGetFactory");
    if (!pGetFactory) {
        fprintf(stderr, "ds4: DirectStorage init: dstorage.dll has no DStorageGetFactory export\n");
        return 0;
    }

    HRESULT hr = pGetFactory(__uuidof(IDStorageFactory), (void **)&g_factory);
    if (FAILED(hr) || !g_factory) {
        fprintf(stderr, "ds4: DirectStorage init: DStorageGetFactory failed hr=0x%08lX\n",
                (unsigned long)hr);
        release_all();
        return 0;
    }

    const uint32_t staging_mb = staging_mb_from_env();
    hr = g_factory->SetStagingBufferSize(staging_mb * 1024u * 1024u);
    if (FAILED(hr)) {
        fprintf(stderr, "ds4: DirectStorage init: SetStagingBufferSize(%u MiB) failed hr=0x%08lX\n",
                staging_mb, (unsigned long)hr);
        release_all();
        return 0;
    }

    hr = g_factory->OpenFile(path.c_str(), __uuidof(IDStorageFile), (void **)&g_file);
    if (FAILED(hr) || !g_file) {
        fprintf(stderr, "ds4: DirectStorage init: OpenFile failed hr=0x%08lX\n",
                (unsigned long)hr);
        release_all();
        return 0;
    }

    DSTORAGE_QUEUE_DESC desc;
    memset(&desc, 0, sizeof(desc));
    desc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    desc.Capacity   = kQueueCapacity;
    desc.Priority   = g_demand_priority;
    desc.Name       = "ds4-expert-stream";
    desc.Device     = NULL;      /* null device is legal precisely because we target MEMORY */

    hr = g_factory->CreateQueue(&desc, __uuidof(IDStorageQueue1), (void **)&g_queue);
    if (FAILED(hr) || !g_queue) {
        fprintf(stderr, "ds4: DirectStorage init: CreateQueue failed hr=0x%08lX\n",
                (unsigned long)hr);
        release_all();
        return 0;
    }

    g_event = CreateEventW(NULL, FALSE, FALSE, NULL);
    if (!g_event) {
        fprintf(stderr, "ds4: DirectStorage init: CreateEvent failed (%lu)\n",
                (unsigned long)GetLastError());
        release_all();
        return 0;
    }

    g_file_id_low   = id_low;
    g_file_id_high  = id_high;
    g_volume_serial = volser;
    g_init_ok       = 1;
    g_batch.reserve(kQueueCapacity);
    fprintf(stderr,
            "ds4: DirectStorage ready (queue capacity %u, staging %u MiB, memory destinations)\n",
            (unsigned)kQueueCapacity, staging_mb);
    fprintf(stderr,"ds4: DirectStorage demand priority=%d completion_spin_us=%u read_parts=%u [RTXDISKLAT_CONFIG_V1]\n",
            (int)g_demand_priority,g_completion_spin_us,g_read_parts);
    return 1;
}

extern "C" int ds4_ds_batch_add(uint64_t file_offset, uint32_t bytes, void *dst_host) {
    if (!g_init_ok || !dst_host || bytes == 0) return 0;
    if (file_offset > g_file_size || bytes > g_file_size-file_offset ||
        (uintptr_t)dst_host > UINTPTR_MAX-bytes) return 0;
    const uint32_t parts = request_parts(file_offset,bytes);
    // Atomic admission: a refusal must not leave a partly appended slice.
    if (g_batch.size() + parts > (size_t)kQueueCapacity) return 0;
    if (!g_batch.empty() && g_batch_gpu) return 0;

    DSTORAGE_REQUEST req;
    memset(&req, 0, sizeof(req));
    req.Options.SourceType         = DSTORAGE_REQUEST_SOURCE_FILE;
    req.Options.DestinationType    = DSTORAGE_REQUEST_DESTINATION_MEMORY;
    req.Options.CompressionFormat  = DSTORAGE_COMPRESSION_FORMAT_NONE;
    req.Source.File.Source         = g_file;
    req.Source.File.Offset         = file_offset;
    req.Source.File.Size           = bytes;
    req.Destination.Memory.Buffer  = dst_host;
    req.Destination.Memory.Size    = bytes;
    req.UncompressedSize           = bytes;   /* uncompressed: must equal the source size */

    append_request_parts(req,false,parts);
    return 1;
}

/* [2026-08-20] Is the GPU-destination arena RESOURCE still alive?
 *
 * ds4_ds_prefill_arenas_create() calls gpu_arena_release() to make room for R4's prefill arenas,
 * which destroys g_gpu_arena and the CUDA external-memory mapping derived from it. The CUDA side
 * caches its own pointer (g_ds_gpu_prefetch_arena) and had no way to learn the resource had gone,
 * so it kept reporting the GPU-destination path as available while every ds4_ds_batch_add_gpu()
 * refused with !g_gpu_arena. Measured cost of that disagreement: the decode prepass abandoned all
 * 43 layers per token and fell back to synchronous pread - 206 calls / 499 MiB / 162 ms per token
 * at 3.01 GB/s against DirectStorage's 11.99 GiB/s, i.e. -59.6% decode from request 2 onward. */
extern "C" int ds4_ds_gpu_arena_alive(void) { return g_gpu_arena != NULL; }

extern "C" int ds4_ds_gpu_arena_create(uint64_t bytes, void **out_cuda_ptr) {
    if (out_cuda_ptr) *out_cuda_ptr = NULL;
    if (!g_init_ok || bytes == 0 || !out_cuda_ptr || !ensure_gpu_queue()) return 0;
    if (g_gpu_arena && g_gpu_arena_size == bytes && g_gpu_cuda_ptr) {
        *out_cuda_ptr = g_gpu_cuda_ptr;
        return 1;
    }
    gpu_arena_release();

    D3D12_HEAP_PROPERTIES heap;
    memset(&heap, 0, sizeof(heap));
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = 1;
    heap.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC resource;
    memset(&resource, 0, sizeof(resource));
    resource.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resource.Width = bytes;
    resource.Height = 1;
    resource.DepthOrArraySize = 1;
    resource.MipLevels = 1;
    resource.Format = DXGI_FORMAT_UNKNOWN;
    resource.SampleDesc.Count = 1;
    resource.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    HRESULT hr = g_d3d_device->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_SHARED, &resource, D3D12_RESOURCE_STATE_COMMON,
        NULL, __uuidof(ID3D12Resource), (void **)&g_gpu_arena);
    if (FAILED(hr) || !g_gpu_arena) {
        fprintf(stderr,
                "ds4: DirectStorage GPU init: CreateCommittedResource(%.0f MiB) failed hr=0x%08lX\n",
                (double)bytes / 1048576.0, (unsigned long)hr);
        gpu_arena_release();
        return 0;
    }

    HANDLE shared = NULL;
    hr = g_d3d_device->CreateSharedHandle(g_gpu_arena, NULL, GENERIC_ALL, NULL, &shared);
    if (FAILED(hr) || !shared) {
        fprintf(stderr, "ds4: DirectStorage GPU init: CreateSharedHandle failed hr=0x%08lX\n",
                (unsigned long)hr);
        gpu_arena_release();
        return 0;
    }

    cudaExternalMemoryHandleDesc ext_desc;
    memset(&ext_desc, 0, sizeof(ext_desc));
    ext_desc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    ext_desc.handle.win32.handle = shared;
    ext_desc.size = bytes;
    ext_desc.flags = cudaExternalMemoryDedicated;
    cudaError_t cerr = cudaImportExternalMemory(&g_gpu_extmem, &ext_desc);
    CloseHandle(shared);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "ds4: DirectStorage GPU init: cudaImportExternalMemory failed: %s\n",
                cudaGetErrorString(cerr));
        gpu_arena_release();
        return 0;
    }

    cudaExternalMemoryBufferDesc buffer_desc;
    memset(&buffer_desc, 0, sizeof(buffer_desc));
    buffer_desc.size = bytes;
    cerr = cudaExternalMemoryGetMappedBuffer(&g_gpu_cuda_ptr, g_gpu_extmem, &buffer_desc);
    if (cerr != cudaSuccess) {
        fprintf(stderr,
                "ds4: DirectStorage GPU init: cudaExternalMemoryGetMappedBuffer failed: %s\n",
                cudaGetErrorString(cerr));
        gpu_arena_release();
        return 0;
    }
    g_gpu_arena_size = bytes;

    hr = g_d3d_device->CreateFence(0, D3D12_FENCE_FLAG_SHARED,
                                   __uuidof(ID3D12Fence),
                                   (void **)&g_gpu_fence);
    if (FAILED(hr) || !g_gpu_fence) {
        fprintf(stderr, "ds4: DirectStorage GPU fence creation failed hr=0x%08lX\n",
                (unsigned long)hr);
        gpu_arena_release();
        return 0;
    }
    shared = NULL;
    hr = g_d3d_device->CreateSharedHandle(g_gpu_fence, NULL, GENERIC_ALL,
                                           NULL, &shared);
    if (FAILED(hr) || !shared) {
        fprintf(stderr, "ds4: DirectStorage GPU fence sharing failed hr=0x%08lX\n",
                (unsigned long)hr);
        gpu_arena_release();
        return 0;
    }
    cudaExternalSemaphoreHandleDesc semaphore_desc;
    memset(&semaphore_desc, 0, sizeof(semaphore_desc));
    semaphore_desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    semaphore_desc.handle.win32.handle = shared;
    cerr = cudaImportExternalSemaphore(&g_gpu_semaphore, &semaphore_desc);
    CloseHandle(shared);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "ds4: DirectStorage CUDA fence import failed: %s\n",
                cudaGetErrorString(cerr));
        gpu_arena_release();
        return 0;
    }
    *out_cuda_ptr = g_gpu_cuda_ptr;
    fprintf(stderr, "ds4: DirectStorage GPU arena ready (%.0f MiB, D3D12/CUDA shared)\n",
            (double)bytes / 1048576.0);
    return 1;
}

extern "C" void ds4_ds_gpu_arena_release(void) { gpu_arena_release(); }

/* DS4COMBO option C: DirectStorage straight into the borrowed prefill loader's expert slots.
 * Each slot pool (gate, up, down) is ONE shared D3D12 heap imported into CUDA as a single
 * contiguous mapping (the D3D12 -> CUDA direction, as the GPU arena; nothing touches CUDA VMM).
 * DirectStorage buffer destinations past 4 GiB inside one resource are not honoured (b23 C1:
 * every slot above offset 4 GiB read all-zero), so the heap carries overlapping placed buffer
 * WINDOWS every 2 GiB, each 3 GiB long: a request for heap offset o goes to window o / 2 GiB at
 * a local offset below 2 GiB, and never spans two windows (an expert part is < 8 MiB).
 * Independent of g_gpu_arena and g_prefill_arenas; release requires no batch in flight. */
#define DS4_DS_SLOT_POOLS 3
#define DS4_DS_SLOT_WINDOWS 8
static const uint64_t kSlotWindowStride = UINT64_C(2) << 30;
static const uint64_t kSlotWindowBytes = UINT64_C(3) << 30;
static ID3D12Heap            *g_slot_heap[DS4_DS_SLOT_POOLS] = {};
static ID3D12Resource        *g_slot_win[DS4_DS_SLOT_POOLS][DS4_DS_SLOT_WINDOWS] = {};
static uint32_t               g_slot_nwin[DS4_DS_SLOT_POOLS] = {};
static cudaExternalMemory_t   g_slot_ext[DS4_DS_SLOT_POOLS] = {};
static void                  *g_slot_ptr[DS4_DS_SLOT_POOLS] = {};
static uint64_t               g_slot_bytes[DS4_DS_SLOT_POOLS] = {};

static void slot_pool_release(int i) {
    if (g_slot_ext[i]) { (void)cudaDestroyExternalMemory(g_slot_ext[i]); g_slot_ext[i] = NULL; }
    ULONG left[DS4_DS_SLOT_WINDOWS + 1] = {};
    for (uint32_t w = 0; w < DS4_DS_SLOT_WINDOWS; w++)
        if (g_slot_win[i][w]) { left[w] = g_slot_win[i][w]->Release(); g_slot_win[i][w] = NULL; }
    if (g_slot_heap[i]) { left[DS4_DS_SLOT_WINDOWS] = g_slot_heap[i]->Release(); g_slot_heap[i] = NULL; }
    if (g_slot_nwin[i])
        fprintf(stderr, "DS4COMBO_DS_SLOTS pool %d refcounts after Release: windows %lu %lu heap %lu\n", i + 1,
                (unsigned long)left[0], (unsigned long)left[1], (unsigned long)left[DS4_DS_SLOT_WINDOWS]);
    g_slot_nwin[i] = 0;
    g_slot_ptr[i] = NULL;
    g_slot_bytes[i] = 0;
}

/* Returns pool handle 1..3 (and the CUDA pointer) or 0 with a logged reason. */
extern "C" int ds4_ds_slot_pool_create(uint64_t bytes, void **out_cuda_ptr) {
    if (out_cuda_ptr) *out_cuda_ptr = NULL;
    if (!g_init_ok || bytes == 0 || !out_cuda_ptr || !ensure_gpu_queue()) {
        fprintf(stderr, "DS4COMBO_DS_SLOTS refused: DirectStorage GPU queue unavailable\n");
        return 0;
    }
    int i = 0;
    while (i < DS4_DS_SLOT_POOLS && g_slot_heap[i]) i++;
    if (i == DS4_DS_SLOT_POOLS) { fprintf(stderr, "DS4COMBO_DS_SLOTS refused: all pools in use\n"); return 0; }
    const uint64_t heap_bytes = (bytes + 0xFFFFu) & ~UINT64_C(0xFFFF);
    const uint32_t nwin = heap_bytes <= kSlotWindowBytes ? 1u :
        (uint32_t)((heap_bytes - kSlotWindowBytes + kSlotWindowStride - 1) / kSlotWindowStride) + 1u;
    D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT va;
    memset(&va, 0, sizeof(va));
    if (nwin > DS4_DS_SLOT_WINDOWS ||
        FAILED(g_d3d_device->CheckFeatureSupport(D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT, &va, sizeof(va))) ||
        va.MaxGPUVirtualAddressBitsPerResource < 33) {
        fprintf(stderr, "DS4COMBO_DS_SLOTS refused: %.2f GiB needs %u windows / VA bits %u\n",
                (double)bytes / 1073741824.0, nwin, (unsigned)va.MaxGPUVirtualAddressBitsPerResource);
        return 0;
    }
    D3D12_HEAP_DESC hd;
    memset(&hd, 0, sizeof(hd));
    hd.SizeInBytes = heap_bytes;
    hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    hd.Properties.CreationNodeMask = 1;
    hd.Properties.VisibleNodeMask = 1;
    hd.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    hd.Flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
    HRESULT hr = g_d3d_device->CreateHeap(&hd, __uuidof(ID3D12Heap), (void **)&g_slot_heap[i]);
    if (FAILED(hr) || !g_slot_heap[i]) {
        fprintf(stderr, "DS4COMBO_DS_SLOTS refused: CreateHeap(%.2f GiB) hr=0x%08lX\n",
                (double)heap_bytes / 1073741824.0, (unsigned long)hr);
        slot_pool_release(i);
        return 0;
    }
    for (uint32_t w = 0; w < nwin; w++) {
        const uint64_t at = (uint64_t)w * kSlotWindowStride;
        D3D12_RESOURCE_DESC resource;
        memset(&resource, 0, sizeof(resource));
        resource.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        resource.Width = std::min(kSlotWindowBytes, heap_bytes - at);
        resource.Height = 1;
        resource.DepthOrArraySize = 1;
        resource.MipLevels = 1;
        resource.Format = DXGI_FORMAT_UNKNOWN;
        resource.SampleDesc.Count = 1;
        resource.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        hr = g_d3d_device->CreatePlacedResource(g_slot_heap[i], at, &resource, D3D12_RESOURCE_STATE_COMMON,
                                                NULL, __uuidof(ID3D12Resource), (void **)&g_slot_win[i][w]);
        if (FAILED(hr) || !g_slot_win[i][w]) {
            fprintf(stderr, "DS4COMBO_DS_SLOTS refused: CreatePlacedResource(window %u) hr=0x%08lX\n",
                    w, (unsigned long)hr);
            slot_pool_release(i);
            return 0;
        }
    }
    g_slot_nwin[i] = nwin;
    ID3D12Device1 *device1 = NULL;
    if (SUCCEEDED(g_d3d_device->QueryInterface(__uuidof(ID3D12Device1), (void **)&device1)) && device1) {
        ID3D12Pageable *pageable = g_slot_heap[i];
        const D3D12_RESIDENCY_PRIORITY priority = D3D12_RESIDENCY_PRIORITY_HIGH;
        (void)device1->SetResidencyPriority(1, &pageable, &priority);
        device1->Release();
    }
    HANDLE shared = NULL;
    hr = g_d3d_device->CreateSharedHandle(g_slot_heap[i], NULL, GENERIC_ALL, NULL, &shared);
    if (FAILED(hr) || !shared) {
        fprintf(stderr, "DS4COMBO_DS_SLOTS refused: CreateSharedHandle hr=0x%08lX\n", (unsigned long)hr);
        slot_pool_release(i);
        return 0;
    }
    cudaExternalMemoryHandleDesc ext_desc;
    memset(&ext_desc, 0, sizeof(ext_desc));
    ext_desc.type = cudaExternalMemoryHandleTypeD3D12Heap;
    ext_desc.handle.win32.handle = shared;
    ext_desc.size = heap_bytes;
    ext_desc.flags = 0;
    cudaError_t cerr = cudaImportExternalMemory(&g_slot_ext[i], &ext_desc);
    CloseHandle(shared);
    if (cerr == cudaSuccess) {
        cudaExternalMemoryBufferDesc buffer_desc;
        memset(&buffer_desc, 0, sizeof(buffer_desc));
        buffer_desc.size = heap_bytes;
        cerr = cudaExternalMemoryGetMappedBuffer(&g_slot_ptr[i], g_slot_ext[i], &buffer_desc);
    }
    if (cerr != cudaSuccess || !g_slot_ptr[i]) {
        fprintf(stderr, "DS4COMBO_DS_SLOTS refused: CUDA heap import failed: %s\n", cudaGetErrorString(cerr));
        (void)cudaGetLastError();
        slot_pool_release(i);
        return 0;
    }
    g_slot_bytes[i] = bytes;
    *out_cuda_ptr = g_slot_ptr[i];
    fprintf(stderr, "DS4COMBO_DS_SLOTS pool %d ready: %.2f GiB shared D3D12 heap imported into CUDA, "
                    "%u buffer windows (2 GiB stride, 3 GiB each)\n",
            i + 1, (double)heap_bytes / 1073741824.0, nwin);
    return i + 1;
}

/* Pool handle owning [ptr, ptr+bytes), else 0. */
extern "C" int ds4_ds_slot_pool_find(const void *ptr, uint64_t bytes, uint64_t *offset_out) {
    for (int i = 0; i < DS4_DS_SLOT_POOLS; i++) {
        const char *base = (const char *)g_slot_ptr[i];
        if (!base || (const char *)ptr < base) continue;
        const uint64_t off = (uint64_t)((const char *)ptr - base);
        if (off <= g_slot_bytes[i] && bytes <= g_slot_bytes[i] - off) {
            if (offset_out) *offset_out = off;
            return i + 1;
        }
    }
    return 0;
}

/* The caller has drained every DS batch and synchronised CUDA before releasing. */
extern "C" void ds4_ds_slot_pool_release(int handle) {
    if (handle >= 1 && handle <= DS4_DS_SLOT_POOLS) slot_pool_release(handle - 1);
}

extern "C" int ds4_ds_batch_add_slot(uint64_t file_offset, uint32_t bytes, int handle, uint64_t offset) {
    if (!g_init_ok || handle < 1 || handle > DS4_DS_SLOT_POOLS || !g_slot_heap[handle - 1] || bytes == 0 ||
        g_gpu_queue_count == 0) return 0;
    const int i = handle - 1;
    if (file_offset > g_file_size || bytes > g_file_size - file_offset) return 0;
    if (offset > g_slot_bytes[i] || bytes > g_slot_bytes[i] - offset) return 0;
    /* The last window runs to the heap end (local offset < 3 GiB). */
    const uint32_t w = (uint32_t)std::min<uint64_t>(offset / kSlotWindowStride, g_slot_nwin[i] ? g_slot_nwin[i] - 1 : 0);
    const uint64_t local = offset - (uint64_t)w * kSlotWindowStride;
    if (w >= g_slot_nwin[i] || !g_slot_win[i][w] ||
        local + bytes > g_slot_win[i][w]->GetDesc().Width) return 0;
    const uint32_t parts = request_parts(file_offset, bytes);
    if (g_batch.size() + parts > (size_t)kQueueCapacity) return -1;   /* full: submit, then retry */
    if (!g_batch.empty() && !g_batch_gpu) return 0;
    DSTORAGE_REQUEST req;
    memset(&req, 0, sizeof(req));
    req.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    req.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
    req.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
    req.Source.File.Source = g_file;
    req.Source.File.Offset = file_offset;
    req.Source.File.Size = bytes;
    req.Destination.Buffer.Resource = g_slot_win[i][w];
    req.Destination.Buffer.Offset = local;
    req.Destination.Buffer.Size = bytes;
    req.UncompressedSize = bytes;
    append_request_parts(req, true, parts);
    g_batch_gpu = true;
    return 1;
}

extern "C" int ds4_ds_prefill_arenas_create(
        const uint64_t *bytes, uint32_t count, void **out_cuda_ptrs) {
    if (!bytes || !out_cuda_ptrs || count == 0 ||
        count > DS4_DS_PREFILL_ARENAS || !g_init_ok || !ensure_gpu_queue()) return 0;
    gpu_arena_release();
    prefill_arenas_release();

    D3D12_HEAP_PROPERTIES heap;
    memset(&heap, 0, sizeof(heap));
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = 1;
    heap.VisibleNodeMask = 1;
    for (uint32_t i = 0; i < count; i++) {
        out_cuda_ptrs[i] = NULL;
        if (bytes[i] == 0) {
            prefill_arenas_release();
            return 0;
        }
        D3D12_RESOURCE_DESC resource;
        memset(&resource, 0, sizeof(resource));
        resource.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        resource.Width = bytes[i];
        resource.Height = 1;
        resource.DepthOrArraySize = 1;
        resource.MipLevels = 1;
        resource.Format = DXGI_FORMAT_UNKNOWN;
        resource.SampleDesc.Count = 1;
        resource.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        HRESULT hr = g_d3d_device->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_SHARED, &resource,
            D3D12_RESOURCE_STATE_COMMON, NULL,
            __uuidof(ID3D12Resource), (void **)&g_prefill_arenas[i]);
        if (FAILED(hr) || !g_prefill_arenas[i]) {
            fprintf(stderr,
                    "ds4: DirectStorage prefill slot %u resource (%.0f MiB) failed hr=0x%08lX\n",
                    i, (double)bytes[i] / 1048576.0, (unsigned long)hr);
            prefill_arenas_release();
            return 0;
        }
        HANDLE shared = NULL;
        hr = g_d3d_device->CreateSharedHandle(g_prefill_arenas[i], NULL,
                                               GENERIC_ALL, NULL, &shared);
        if (FAILED(hr) || !shared) {
            prefill_arenas_release();
            return 0;
        }
        cudaExternalMemoryHandleDesc ext_desc;
        memset(&ext_desc, 0, sizeof(ext_desc));
        ext_desc.type = cudaExternalMemoryHandleTypeD3D12Resource;
        ext_desc.handle.win32.handle = shared;
        ext_desc.size = bytes[i];
        ext_desc.flags = cudaExternalMemoryDedicated;
        cudaError_t cerr = cudaImportExternalMemory(&g_prefill_extmem[i], &ext_desc);
        CloseHandle(shared);
        if (cerr != cudaSuccess) {
            prefill_arenas_release();
            return 0;
        }
        cudaExternalMemoryBufferDesc buffer_desc;
        memset(&buffer_desc, 0, sizeof(buffer_desc));
        buffer_desc.size = bytes[i];
        cerr = cudaExternalMemoryGetMappedBuffer(
            &g_prefill_cuda_ptr[i], g_prefill_extmem[i], &buffer_desc);
        if (cerr != cudaSuccess || !g_prefill_cuda_ptr[i]) {
            prefill_arenas_release();
            return 0;
        }
        g_prefill_arena_size[i] = bytes[i];
        out_cuda_ptrs[i] = g_prefill_cuda_ptr[i];
    }
    g_prefill_arena_count = count;
    if (!create_shared_fence()) {
        fprintf(stderr, "ds4: DirectStorage prefill shared fence creation failed\n");
        prefill_arenas_release();
        return 0;
    }
    fprintf(stderr,
            "ds4: DirectStorage prefill arenas ready (%u separate D3D12/CUDA resources)\n",
            count);
    return 1;
}

extern "C" void ds4_ds_prefill_arenas_release(void) {
    prefill_arenas_release();
}

struct ds4_ds_prefill_wave {
    std::vector<DSTORAGE_REQUEST> reqs;
    IDStorageQueue1 *last_queue = NULL;
    uint64_t bytes = 0;
};
static std::vector<ds4_ds_prefill_wave> g_prefill_waves;

static void ds4_ds_stats_record_prefill_wave(uint64_t bytes, size_t nreq);

extern "C" void ds4_ds_prefill_waves_release(void) {
    g_prefill_waves.clear();
}

extern "C" int ds4_ds_prefill_waves_create(uint32_t count) {
    if (!g_init_ok || g_prefill_arena_count == 0 ||
        !g_gpu_fence || !g_gpu_semaphore || count == 0) {
        return 0;
    }
    g_prefill_waves.clear();
    g_prefill_waves.resize(count);
    return 1;
}

extern "C" int ds4_ds_prefill_wave_build(
        uint32_t index, const ds4_ds_wave_span *spans, uint32_t count) {
    if (index >= g_prefill_waves.size() || !spans || count == 0 ||
        count > (uint32_t)kQueueCapacity) return 0;
    ds4_ds_prefill_wave &wave = g_prefill_waves[index];
    wave.reqs.clear();
    wave.bytes = 0;
    wave.reqs.reserve(count);
    for (uint32_t i = 0; i < count; i++) {
        if (spans[i].bytes == 0 ||
            spans[i].arena_slot >= g_prefill_arena_count ||
            spans[i].arena_offset > g_prefill_arena_size[spans[i].arena_slot] ||
            spans[i].bytes >
                g_prefill_arena_size[spans[i].arena_slot] - spans[i].arena_offset) {
            wave.reqs.clear();
            wave.bytes = 0;
            return 0;
        }
        DSTORAGE_REQUEST req;
        memset(&req, 0, sizeof(req));
        req.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        req.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
        req.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
        req.Source.File.Source = g_file;
        req.Source.File.Offset = spans[i].file_offset;
        req.Source.File.Size = spans[i].bytes;
        req.Destination.Buffer.Resource =
            g_prefill_arenas[spans[i].arena_slot];
        req.Destination.Buffer.Offset = spans[i].arena_offset;
        req.Destination.Buffer.Size = spans[i].bytes;
        req.UncompressedSize = spans[i].bytes;
        wave.reqs.push_back(req);
        wave.bytes += spans[i].bytes;
    }
    return 1;
}

extern "C" int ds4_ds_prefill_wave_submit(
        uint32_t index, uint64_t *out_fence_value) {
    if (out_fence_value) *out_fence_value = 0;
    if (index >= g_prefill_waves.size() || g_gpu_queue_count == 0 ||
        !g_gpu_fence) return 0;
    ds4_ds_prefill_wave &wave = g_prefill_waves[index];
    if (wave.reqs.empty()) return 0;
    /* All R4 waves share one monotonically-valued fence. Keep their signals
     * on one ordered queue: round-robin queues can complete value N+1 before
     * N, which makes CUDA's wait for N succeed before N's bytes have landed. */
    IDStorageQueue1 *queue = g_gpu_queues[0];
    for (size_t i = 0; i < wave.reqs.size(); i++) {
        queue->EnqueueRequest(&wave.reqs[i]);
    }
    const uint64_t value = ++g_gpu_fence_next;
    queue->EnqueueSignal(g_gpu_fence, value);
    queue->Submit();
    wave.last_queue = queue;
    ds4_ds_stats_record_prefill_wave(wave.bytes, wave.reqs.size());
    if (out_fence_value) *out_fence_value = value;
    return 1;
}

extern "C" int ds4_ds_prefill_wave_cuda_wait(
        uint64_t fence_value, void *cuda_stream) {
    if (!g_gpu_semaphore || fence_value == 0) return 0;
    cudaExternalSemaphoreWaitParams params;
    memset(&params, 0, sizeof(params));
    params.params.fence.value = fence_value;
    const cudaError_t err = cudaWaitExternalSemaphoresAsync(
        &g_gpu_semaphore, &params, 1, (cudaStream_t)cuda_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: DirectStorage CUDA fence wait failed: %s\n",
                cudaGetErrorString(err));
        return 0;
    }
    return 1;
}

/* DS4COMBO diagnostic: order CUDA work on `cuda_stream` after every GPU write
 * of the last GPU-destination submit_wait batch, through the shared D3D12
 * fence and its CUDA external semaphore (the R4 prefill-wave mechanism). The
 * signal is enqueued on that batch's own queue, so it follows its requests. */
extern "C" int ds4_ds_gpu_fence_wait_last(void *cuda_stream) {
    if (!g_init_ok || !g_gpu_fence || !g_gpu_semaphore || !g_last_sync_gpu_queue) return 0;
    const uint64_t value = ++g_gpu_fence_next;
    g_last_sync_gpu_queue->EnqueueSignal(g_gpu_fence, value);
    g_last_sync_gpu_queue->Submit();
    return ds4_ds_prefill_wave_cuda_wait(value, cuda_stream);
}

extern "C" int ds4_ds_prefill_waves_check_errors(void) {
    for (size_t i = 0; i < g_prefill_waves.size(); i++) {
        const ds4_ds_prefill_wave &wave = g_prefill_waves[i];
        if (wave.last_queue && !wave_succeeded(wave.last_queue, wave.reqs.size())) {
            return 0;
        }
    }
    return 1;
}

/* [2026-08-20] DS4_GPU_ADD_REASON_V1.
 * This function has seven ways to return 0 and NONE of them logged anything, so a refusal here
 * surfaced only as `skipped++; break` in the CUDA prepass, which abandons the layer and drops every
 * expert to a synchronous pread. Measured: on requests 2+ the prepass enters all 43 layers, stages
 * ZERO experts and skips 32, with a healthy 256 MiB capacity - i.e. this function is refusing and we
 * could not tell which branch. The reason code makes that observable; 0 means "succeeded". */
int g_ds_gpu_add_fail_reason = 0;
extern "C" int ds4_ds_gpu_add_fail_reason(void) { return g_ds_gpu_add_fail_reason; }

extern "C" int ds4_ds_batch_add_gpu(
        uint64_t file_offset, uint32_t bytes, uint64_t arena_offset) {
    if (!g_init_ok)            { g_ds_gpu_add_fail_reason = 1; return 0; }
    if (!g_gpu_arena)          { g_ds_gpu_add_fail_reason = 2; return 0; }
    if (bytes == 0)            { g_ds_gpu_add_fail_reason = 3; return 0; }
    if (g_gpu_queue_count == 0){ g_ds_gpu_add_fail_reason = 4; return 0; }
    if (file_offset > g_file_size || bytes > g_file_size-file_offset) {
                                 g_ds_gpu_add_fail_reason = 8; return 0; }
    if (arena_offset > g_gpu_arena_size || bytes > g_gpu_arena_size - arena_offset) {
                                 g_ds_gpu_add_fail_reason = 5; return 0; }
    const uint32_t parts = request_parts(file_offset,bytes);
    if (g_batch.size() + parts > (size_t)kQueueCapacity) {
                                 g_ds_gpu_add_fail_reason = 6; return 0; }
    if (!g_batch.empty() && !g_batch_gpu) {
                                 g_ds_gpu_add_fail_reason = 7; return 0; }
    g_ds_gpu_add_fail_reason = 0;

    DSTORAGE_REQUEST req;
    memset(&req, 0, sizeof(req));
    req.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    req.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
    req.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
    req.Source.File.Source = g_file;
    req.Source.File.Offset = file_offset;
    req.Source.File.Size = bytes;
    req.Destination.Buffer.Resource = g_gpu_arena;
    req.Destination.Buffer.Offset = arena_offset;
    req.Destination.Buffer.Size = bytes;
    req.UncompressedSize = bytes;

    append_request_parts(req,true,parts);
    g_batch_gpu = true;
    return 1;
}

/* [2026-08-16] SPLIT submit from wait.
 *
 * WHY: profiling a 16K prefill showed the `load` phase is 93.5% of streaming prefill time (26,492 ms
 * of 28,346 ms over 172 layer-batches) while the GPU sat ~80% idle and the RAID ran at ~15% of its
 * 24 GB/s. Nothing was saturated - the pipeline simply alternated fetch and compute, because the
 * ONLY exit from a batch was submit-and-block. DirectStorage is built for deep queues
 * (kQueueCapacity is already DSTORAGE_MAX_QUEUE_CAPACITY = 8192), so the queue was never the limit;
 * the call shape was.
 *
 * ds4_ds_batch_submit_wait() is preserved below as submit(); wait(), so every existing caller is
 * bit-for-bit unchanged. Nothing overlaps yet - see the note on ds4_ds_batch_submit().
 */
/* [2026-08-16] TWO-SLOT RING so a second batch can be built while the first is in flight.
 *
 * `g_batch` stays the FILL buffer, so ds4_ds_batch_add/count/reset are untouched. submit() moves the
 * filled batch into a free in-flight slot (std::move keeps the DSTORAGE_REQUEST objects alive for
 * the queue, which reads them asynchronously) and clears `g_batch` so the caller can immediately
 * build the next layer's batch. wait() drains the OLDEST outstanding slot, FIFO.
 *
 * Two slots is the minimum that overlaps layer N+1's fetch with layer N's compute, and the sweep
 * already showed 1792 MiB per slot (3584 total, host staging - not VRAM) is measured-safe, so each
 * slot keeps the known-best size rather than halving to 896 (which costs 9% on its own). */
/* [2026-08-18] DEPTH IS NOW A KNOB. Measured: every DirectStorage rate this program reports equals
 * batch_size / batch_latency exactly (24.8 MiB/1.928 ms = 12.55 GiB/s; 33.0/2.246 = 14.35; a 1.4 GiB
 * load batch = 14.4), i.e. the queue completes ONE batch at a time. Batch SIZE has already saturated -
 * a 1.4 GiB batch still gives 14.7 GiB/s - so the only way to the array's ~24 GB/s is MORE BATCHES IN
 * FLIGHT. Two slots overlaps layer N+1 with layer N; deeper pipelining (N+2, N+3) needs more.
 * The compile-time array is grown; the RUNTIME depth defaults to 2, so behaviour is unchanged unless
 * DS4_DS_PIPELINE_DEPTH is set. */
#define DS4_DS_SLOTS 8
struct ds4_ds_slow_tags {
    uint64_t token = UINT64_MAX;
    uint32_t first_layer = UINT32_MAX;
    uint32_t last_layer = UINT32_MAX;
    int      phase_decode = 0;
    int      token_valid = 0;
    int      valid = 0;
};

static int      g_ds_slow_tags_enabled = 0;
static int      g_ds_phase_decode = 0;
static uint64_t g_ds_profile_token = UINT64_MAX;
static int      g_ds_profile_token_valid = 0;
static ds4_ds_slow_tags g_ds_sync_tags;

struct ds4_ds_slot {
    std::vector<DSTORAGE_REQUEST> reqs;
    HANDLE        event = NULL;
    IDStorageQueue1 *queue = NULL;
    uint64_t      bytes = 0;
    LARGE_INTEGER t0    = {};
    bool          busy  = false;
    ds4_ds_slow_tags slow_tags;
};
static ds4_ds_slot g_slots[DS4_DS_SLOTS];
static uint32_t    g_slot_next  = 0;   /* next slot to fill    */
static uint32_t    g_slot_drain = 0;   /* oldest slot to wait  */
static uint32_t    g_slot_busy  = 0;   /* how many outstanding */

/* Opt-in, bounded RAM trace. Emitted only at stats dump, never to disk on the
 * timed fetch path. Batch latency includes overlap; wait_ms is the observed
 * blocking stall and must be reported separately. Multi-layer waves retain
 * their complete range rather than being attributed to an invented layer. */
struct ds4_ds_layer_sample {
    uint64_t token, bytes;
    uint32_t first_layer, last_layer, requests, epoch;
    double batch_ms, wait_ms;
};
static std::vector<ds4_ds_layer_sample> g_layer_samples;
static uint32_t g_layer_epoch = 0;
static bool layer_trace_enabled(void) {
    static const bool enabled=[](){const char *s=getenv("DS4_DS_LAYER_TRACE");return s&&!strcmp(s,"1");}();
    return enabled;
}
extern "C" int ds4_ds_layer_trace_enabled(void) { return layer_trace_enabled() ? 1 : 0; }
static void layer_trace_dump(void) {
    for (const auto &s:g_layer_samples)
        fprintf(stderr,"RTXDISKLAT_LAYER epoch=%u token=%llu layer=%u-%u bytes=%llu requests=%u batch_ms=%.9g wait_ms=%.9g\n",
                s.epoch,(unsigned long long)s.token,s.first_layer,s.last_layer,
                (unsigned long long)s.bytes,s.requests,s.batch_ms,s.wait_ms);
    g_layer_samples.clear();
}

extern "C" void ds4_ds_stats_set_token(uint64_t token_index) {
    if (g_ds_slow_tags_enabled) {
        g_ds_profile_token = token_index;
        g_ds_profile_token_valid = 1;
    }
}

extern "C" void ds4_ds_stats_set_batch_layer(
        uint32_t first_layer, uint32_t last_layer) {
    if (!g_ds_slow_tags_enabled) return;
    ds4_ds_slow_tags tags;
    tags.token = g_ds_profile_token;
    tags.first_layer = first_layer;
    tags.last_layer = last_layer;
    tags.phase_decode = g_ds_phase_decode;
    tags.token_valid = g_ds_profile_token_valid;
    tags.valid = 1;
    if (g_slot_busy < DS4_DS_SLOTS && !g_slots[g_slot_next].busy) {
        g_slots[g_slot_next].slow_tags = tags;
    }
    g_ds_sync_tags = tags;
}

/* Runtime pipeline depth, clamped to DS4_DS_SLOTS. Cached: this is read on the per-layer path and a bare
 * getenv there costs a measured 1.5 us/call. */
extern "C" uint32_t ds4_ds_pipeline_depth(void) {
    static uint32_t cached = 0;
    if (cached == 0) {
        cached = 2;
        const char *env = getenv("DS4_DS_PIPELINE_DEPTH");
        if (env && env[0]) {
            char *end = NULL;
            unsigned long v = strtoul(env, &end, 10);
            if (end != env && end && *end == 0 && v >= 1) {
                cached = (v > DS4_DS_SLOTS) ? (uint32_t)DS4_DS_SLOTS : (uint32_t)v;
            }
        }
        if (cached != 2) {
            fprintf(stderr, "ds4: DirectStorage pipeline depth = %u slots (max %d)\n",
                    cached, DS4_DS_SLOTS);
        }
    }
    return cached;
}

extern "C" uint32_t ds4_ds_pipeline_capacity(void) {
    return DS4_DS_SLOTS;
}

/* ── FETCH-RATE AND LATENCY INSTRUMENTATION ─────────────────────────────────────────────────────────
 *
 * WHY: every fetch-rate number this program has quoted was wrong in the same way. "3.5 of 24 GiB/s" was
 * computed as bytes divided by TOTAL WALL TIME, which averages over the long stretches when DirectStorage
 * is idle because the pipeline is serialised (measured: routed_moe 30.6 s = 8.9 copy + 11.0 DS + 10.7
 * compute). That figure therefore describes DUTY CYCLE, not throughput, and understates the achieved rate
 * by roughly 3x. Independently, the decode arithmetic proves the disk must already be sustaining ~8.5
 * GiB/s: one MoE layer needs 6 x 6.75 MiB = 40.5 MiB, the layer budget at 4.5 tok/s is 5.29 ms, and 40.5
 * MiB in 5.29 ms IS 7.5 GiB/s. A "15% utilised" disk cannot be doing that.
 *
 * So this records bytes and time SPENT INSIDE DirectStorage, giving the instantaneous rate directly.
 *
 * AND LATENCY, WHICH IS THE SHARPER QUESTION. MoE routing is data-dependent: layer L+1's experts are
 * unknowable until layer L has computed, so each layer's fetch sits on the critical path, 42 times per
 * token. That is a serial dependency chain, and bandwidth cannot fix a latency chain. At the 10 tok/s
 * goal the per-layer budget is 2.38 ms, of which the bytes cost 1.65 ms even at a full 24 GiB/s - leaving
 * 0.73 ms for routing, issue, wait, copy and compute combined. So per-batch LATENCY and the number of
 * batches per layer matter as much as GiB/s, and neither has ever been measured here.
 *
 * Cost: two integer adds and one comparison per batch, on a path that already calls
 * QueryPerformanceCounter twice. Nothing is added to any per-expert or per-token path. */
static uint64_t g_ds_total_bytes = 0;
static double   g_ds_total_ms    = 0.0;
static uint64_t g_ds_batches     = 0;
static uint64_t g_ds_requests    = 0;
static uint64_t g_ds_prefill_wave_bytes = 0;
static uint64_t g_ds_prefill_waves      = 0;
static uint64_t g_ds_prefill_wave_requests = 0;
static double   g_ds_max_ms      = 0.0;
static uint64_t g_ds_lat_us_hist[20] = {0};   /* log2 buckets of per-batch latency, microseconds */
static uint64_t g_ds_req_hist[20]    = {0};   /* log2 buckets of requests per batch */
static int      g_ds_stats_registered = 0;
struct ds4_ds_stats_snapshot {
    uint64_t total_bytes = 0;
    double   total_ms = 0.0;
    uint64_t batches = 0;
    uint64_t requests = 0;
    uint64_t prefill_wave_bytes = 0;
    uint64_t prefill_waves = 0;
    uint64_t prefill_wave_requests = 0;
    double   max_ms = 0.0;
    uint64_t lat_us_hist[20] = {0};
    uint64_t req_hist[20] = {0};
};
/* Per-phase totals for the current request: [0] prefill, [1] decode. The live
 * counters are banked into the phase being left at every phase change, so a
 * prefill -> decode -> prefill -> decode sequence (setup/checkpoint requests
 * between the prefill and generate) keeps the prefill bytes instead of
 * replacing them with the last, often empty, prefill span. */
static ds4_ds_stats_snapshot g_ds_phase_acc[2];
static int g_ds_phase_tracking_started = 0;

static void ds4_ds_stats_capture(ds4_ds_stats_snapshot *snapshot) {
    if (!snapshot) return;
    snapshot->total_bytes = g_ds_total_bytes;
    snapshot->total_ms = g_ds_total_ms;
    snapshot->batches = g_ds_batches;
    snapshot->requests = g_ds_requests;
    snapshot->prefill_wave_bytes = g_ds_prefill_wave_bytes;
    snapshot->prefill_waves = g_ds_prefill_waves;
    snapshot->prefill_wave_requests = g_ds_prefill_wave_requests;
    snapshot->max_ms = g_ds_max_ms;
    memcpy(snapshot->lat_us_hist, g_ds_lat_us_hist, sizeof(g_ds_lat_us_hist));
    memcpy(snapshot->req_hist, g_ds_req_hist, sizeof(g_ds_req_hist));
}

static void ds4_ds_stats_reset_live(void) {
    g_ds_total_bytes = 0;
    g_ds_total_ms = 0.0;
    g_ds_batches = 0;
    g_ds_requests = 0;
    g_ds_prefill_wave_bytes = 0;
    g_ds_prefill_waves = 0;
    g_ds_prefill_wave_requests = 0;
    g_ds_max_ms = 0.0;
    memset(g_ds_lat_us_hist, 0, sizeof(g_ds_lat_us_hist));
    memset(g_ds_req_hist, 0, sizeof(g_ds_req_hist));
}

static void ds4_ds_stats_merge(ds4_ds_stats_snapshot *into, const ds4_ds_stats_snapshot *add) {
    into->total_bytes += add->total_bytes;
    into->total_ms += add->total_ms;
    into->batches += add->batches;
    into->requests += add->requests;
    into->prefill_wave_bytes += add->prefill_wave_bytes;
    into->prefill_waves += add->prefill_waves;
    into->prefill_wave_requests += add->prefill_wave_requests;
    if (add->max_ms > into->max_ms) into->max_ms = add->max_ms;
    for (int i = 0; i < 20; i++) {
        into->lat_us_hist[i] += add->lat_us_hist[i];
        into->req_hist[i] += add->req_hist[i];
    }
}

/* Move the live counters into the current phase's request total. */
static void ds4_ds_stats_bank_live(void) {
    ds4_ds_stats_snapshot live;
    ds4_ds_stats_capture(&live);
    ds4_ds_stats_merge(&g_ds_phase_acc[g_ds_phase_decode ? 1 : 0], &live);
    ds4_ds_stats_reset_live();
}

static void ds4_ds_stats_dump_snapshot(
        const ds4_ds_stats_snapshot *snapshot, const char *phase) {
    if (!snapshot || (snapshot->batches == 0 && snapshot->prefill_waves == 0)) return;
    const uint64_t total_bytes = snapshot->total_bytes + snapshot->prefill_wave_bytes;
    const uint64_t total_batches = snapshot->batches + snapshot->prefill_waves;
    const uint64_t total_requests = snapshot->requests + snapshot->prefill_wave_requests;
    const double gib = (double)total_bytes / 1073741824.0;
    if (phase) {
        fprintf(stderr,
                "ds4: DirectStorage phase=%s totals: %.3f GiB in %.1f ms inside DS across %llu batches / %llu requests [DS4_DS_PHASE_STATS_V1]\n",
                phase,
                gib,
                snapshot->total_ms,
                (unsigned long long)total_batches,
                (unsigned long long)total_requests);
    } else {
        fprintf(stderr,
                "ds4: DirectStorage totals: %.3f GiB in %.1f ms inside DS across %llu batches / %llu requests\n",
                gib,
                snapshot->total_ms,
                (unsigned long long)total_batches,
                (unsigned long long)total_requests);
    }
    if (snapshot->prefill_waves != 0) {
        fprintf(stderr,
                "ds4: DirectStorage async prefill waves included in totals: "
                "%.3f GiB across %llu waves / %llu requests "
                "[DS4_R7_PREFILL_WAVE_ACCOUNTING]\n",
                (double)snapshot->prefill_wave_bytes / 1073741824.0,
                (unsigned long long)snapshot->prefill_waves,
                (unsigned long long)snapshot->prefill_wave_requests);
    }
    if (snapshot->total_ms > 0.0 && snapshot->total_bytes != 0) {
        fprintf(stderr,
                "ds4: DirectStorage%s%s INSTANTANEOUS rate %.2f GiB/s "
                "(timed batch bytes / time-inside-DS; async prefill waves overlap compute and are excluded)\n",
                phase ? " phase=" : "",
                phase ? phase : "",
                ((double)snapshot->total_bytes / 1073741824.0) /
                    (snapshot->total_ms / 1000.0));
    }
    if (snapshot->batches == 0) return;
    fprintf(stderr,
            "ds4: DirectStorage%s%s per-batch: mean %.3f ms, max %.3f ms, mean %.1f requests, mean %.2f MiB\n",
            phase ? " phase=" : "",
            phase ? phase : "",
            snapshot->total_ms / (double)snapshot->batches,
            snapshot->max_ms,
            (double)snapshot->requests / (double)snapshot->batches,
            (double)snapshot->total_bytes / (double)snapshot->batches / 1048576.0);
    fprintf(stderr, "ds4: DirectStorage%s%s batch latency histogram (us, log2 buckets):",
            phase ? " phase=" : "", phase ? phase : "");
    for (int i = 0; i < 20; i++) {
        if (snapshot->lat_us_hist[i]) {
            fprintf(stderr, " [%d-%d)=%llu", 1 << i, 1 << (i + 1),
                    (unsigned long long)snapshot->lat_us_hist[i]);
        }
    }
    fprintf(stderr, "\nds4: DirectStorage%s%s requests-per-batch histogram (log2 buckets):",
            phase ? " phase=" : "", phase ? phase : "");
    for (int i = 0; i < 20; i++) {
        if (snapshot->req_hist[i]) {
            fprintf(stderr, " [%d-%d)=%llu", 1 << i, 1 << (i + 1),
                    (unsigned long long)snapshot->req_hist[i]);
        }
    }
    fprintf(stderr, "\n");
}

extern "C" void ds4_ds_stats_dump(void) {
    layer_trace_dump();
    ds4_ds_stats_snapshot live;
    ds4_ds_stats_capture(&live);
    if (!g_ds_phase_tracking_started) {
        ds4_ds_stats_dump_snapshot(&live, NULL);
        return;
    }
    ds4_ds_stats_bank_live();
    ds4_ds_stats_dump_snapshot(&g_ds_phase_acc[0], "prefill");
    ds4_ds_stats_dump_snapshot(&g_ds_phase_acc[1], "decode");

    /* The server invokes this function at request completion while holding its
     * inference mutex. Consume the request-local snapshots there so the next
     * request begins cleanly; atexit remains only the CLI fallback. */
    g_ds_phase_acc[0] = g_ds_phase_acc[1] = ds4_ds_stats_snapshot();
    g_ds_phase_tracking_started = 0;
    g_ds_slow_tags_enabled = 0;
    g_ds_profile_token_valid = 0;
}

extern "C" void ds4_ds_stats_set_phase(int decode) {
    g_ds_slow_tags_enabled = 1;
    const int next_decode = decode != 0;
    if (layer_trace_enabled() && g_layer_samples.capacity() == 0)
        g_layer_samples.reserve(65536);
    if (!next_decode && g_ds_phase_decode && layer_trace_enabled()) ++g_layer_epoch;
    if (!g_ds_phase_tracking_started) {
        ds4_ds_stats_reset_live();
        g_ds_phase_acc[0] = g_ds_phase_acc[1] = ds4_ds_stats_snapshot();
        g_ds_phase_tracking_started = 1;
    } else if (next_decode != g_ds_phase_decode) {
        ds4_ds_stats_bank_live();
    }
    g_ds_sync_tags.valid = 0;
    for (uint32_t i = 0; i < DS4_DS_SLOTS; i++) {
        if (!g_slots[i].busy) g_slots[i].slow_tags.valid = 0;
    }
    g_ds_phase_decode = next_decode;
    g_ds_profile_token = UINT64_MAX;
    g_ds_profile_token_valid = 0;
}

static void ds4_ds_stats_record_prefill_wave(uint64_t bytes, size_t nreq) {
    if (!g_ds_stats_registered) {
        g_ds_stats_registered = 1;
        atexit(ds4_ds_stats_dump);
    }
    g_ds_prefill_wave_bytes += bytes;
    g_ds_prefill_waves += 1;
    g_ds_prefill_wave_requests += (uint64_t)nreq;
}

/* SLOW-BATCH LOG.  MEASURED 2026-08-18: ~0.7% of batches (46 of 6900) consume ~35% of ALL DirectStorage
 * time - 65-520 ms each, moving ~25 MiB, i.e. 0.05-0.4 GiB/s against the 12.55 GiB/s the SAME queue
 * sustains on the bulk in the SAME run.  Those are stalls, not slow reads, and they mean the 35.9 ms/token
 * "average wait" the overlap work targeted is an artefact of aggregation over a few dozen events.
 *
 * The two histograms above CANNOT settle what causes them, because they are MARGINAL: 43 batches carry
 * >=256 requests and 46 are slow, which is suggestive but does not pair a latency to a request count.
 * This does - it prints the triple for one batch, which is the whole question.  It also prints a wall
 * timestamp, which answers the prior question: if the slow batches cluster in the first seconds they are
 * warm-up and can be excluded from every measurement; if they are spread through decode they ARE the
 * decode bottleneck.
 *
 * Volume is ~46 lines per run, so there is no measurable overhead and nothing like the per-token sync
 * profiler that degrades a run sequence -55%. The base slow line and wait_ms remain default ON. Rich
 * phase/token/layer tags reuse the existing CUDA decode-profile gate; their context writes do not exist
 * in an unprofiled run, and wall time, queue lookup and formatting stay below the slow-only branch. */
static double ds4_ds_wall_s(void) {
    static LARGE_INTEGER freq = {0};
    static LARGE_INTEGER t0   = {0};
    LARGE_INTEGER now;
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&t0);
    }
    QueryPerformanceCounter(&now);
    return (double)(now.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
}

static int ds4_ds_stats_queue_id(IDStorageQueue1 *queue) {
    if (queue == g_queue) return 0;
    for (uint32_t i = 0; i < g_gpu_queue_count; i++) {
        if (queue == g_gpu_queues[i]) return (int)i;
    }
    return -1;
}

static void ds4_ds_stats_record(
        uint64_t bytes,
        double ms,
        double wait_ms,
        size_t nreq,
        IDStorageQueue1 *queue,
        const ds4_ds_slow_tags *tags) {
    if (!g_ds_stats_registered) { g_ds_stats_registered = 1; atexit(ds4_ds_stats_dump); }
    if (layer_trace_enabled() && tags && tags->valid && tags->phase_decode && tags->token_valid) {
        if (g_layer_samples.size() >= 65536) {
            fprintf(stderr,"ds4: RTXDISKLAT layer trace exceeded its 65536-row RAM budget\n");
            abort();
        }
        g_layer_samples.push_back({tags->token,bytes,tags->first_layer,tags->last_layer,
                                  (uint32_t)nreq,g_layer_epoch,ms,wait_ms});
    }
    if (ms >= 8.0) {
        const double wall_s = ds4_ds_wall_s();
        const double rate = ms > 0.0 ?
            ((double)bytes / 1073741824.0) / (ms / 1000.0) : 0.0;
        if (g_ds_slow_tags_enabled && tags && tags->valid) {
            const char *destination = queue == g_queue ? "host" : "GPU-dest";
            const uint32_t inflight_depth = g_slot_busy + 1u;
            fprintf(stderr,
                    "ds4: DS SLOW batch ordinal=%llu t=%.2fs lat=%.3fms wait_ms=%.3f requests=%llu bytes=%.2fMiB rate=%.3fGiB/s phase=%s token=%lld layer=%u-%u dest=%s queue_id=%d inflight=%u [DS4_DS_SLOW_TAGS_V1]\n",
                    (unsigned long long)(g_ds_batches + 1),
                    wall_s,
                    ms,
                    wait_ms,
                    (unsigned long long)nreq,
                    (double)bytes / 1048576.0,
                    rate,
                    tags->phase_decode ? "decode" : "prefill",
                    tags->token_valid ? (long long)tags->token : -1ll,
                    tags->first_layer,
                    tags->last_layer,
                    destination,
                    ds4_ds_stats_queue_id(queue),
                    inflight_depth);
        } else {
            fprintf(stderr,
                    "ds4: DS SLOW batch ordinal=%llu t=%.2fs lat=%.3fms wait_ms=%.3f requests=%llu bytes=%.2fMiB rate=%.3fGiB/s\n",
                    (unsigned long long)(g_ds_batches + 1),
                    wall_s,
                    ms,
                    wait_ms,
                    (unsigned long long)nreq,
                    (double)bytes / 1048576.0,
                    rate);
        }
    }
    g_ds_total_bytes += bytes;
    g_ds_total_ms    += ms;
    g_ds_batches     += 1;
    g_ds_requests    += (uint64_t)nreq;
    if (ms > g_ds_max_ms) g_ds_max_ms = ms;
    double us = ms * 1000.0;
    int b = 0;
    while (b < 19 && us >= 2.0) { us /= 2.0; b++; }
    g_ds_lat_us_hist[b]++;
    size_t n = nreq; int rb = 0;
    while (rb < 19 && n >= 2) { n >>= 1; rb++; }
    g_ds_req_hist[rb]++;
}

extern "C" uint32_t ds4_ds_batch_inflight(void) { return g_slot_busy; }

extern "C" int ds4_ds_batch_oldest_ready(void) {
    if (g_slot_busy == 0) return -1;
    const ds4_ds_slot &s = g_slots[g_slot_drain];
    if (!s.busy || !s.event) return -1;
    const DWORD wait = WaitForSingleObject(s.event, 0);
    if (wait == WAIT_OBJECT_0) return 1;
    return wait == WAIT_TIMEOUT ? 0 : -1;
}

/* Explicit diagnostic only: separate CPU enqueue and Submit API intervals.
 * Does not label asynchronous submit-to-completion time as CPU work. */
static bool ds4_ds_submit_trace_enabled(void) {
    static const bool enabled=[](){const char *s=getenv("DS4_CUDA_DS_SUBMIT_TRACE");return s && !strcmp(s,"1");}();
    return enabled;
}
static void ds4_ds_submit_trace(const char *path, uint64_t bytes, size_t requests,
        const LARGE_INTEGER &enqueue_start, const LARGE_INTEGER &submit_start,
        const LARGE_INTEGER &submit_end) {
    LARGE_INTEGER freq;QueryPerformanceFrequency(&freq);
    fprintf(stderr,"ds4: DSSUBMIT path=%s bytes=%llu requests=%llu enqueue_ms=%.6f submit_ms=%.6f token=%lld\n",
        path,(unsigned long long)bytes,(unsigned long long)requests,
        (double)(submit_start.QuadPart-enqueue_start.QuadPart)*1000.0/freq.QuadPart,
        (double)(submit_end.QuadPart-submit_start.QuadPart)*1000.0/freq.QuadPart,
        g_ds_profile_token_valid?(long long)g_ds_profile_token:-1ll);
}

extern "C" int ds4_ds_batch_submit(void) {
    if (!g_init_ok) return 0;
    if (g_batch.empty()) return 1;
    if (g_slot_busy >= DS4_DS_SLOTS) {
        fprintf(stderr, "ds4: DirectStorage submit with all %d slots in flight\n", DS4_DS_SLOTS);
        return 0;
    }
    if(g_background_busy && !g_background_saw_demand) {
        LARGE_INTEGER now,freq;QueryPerformanceCounter(&now);QueryPerformanceFrequency(&freq);
        fprintf(stderr,"ds4: affinity-background token=%llu idle_window_ms=%.6f next_demand_while_background=%u\n",
            (unsigned long long)g_ds_profile_token,
            (double)(now.QuadPart-g_background_started.QuadPart)*1000.0/freq.QuadPart,
            WaitForSingleObject(g_background_event,0)!=WAIT_OBJECT_0);
        g_background_saw_demand=true;
    }
    ds4_ds_slot &s = g_slots[g_slot_next];
    if (!s.event) {
        /* Manual-reset makes ds4_ds_batch_oldest_ready() a non-consuming
         * readiness probe. Reset explicitly before every slot reuse. */
        s.event = CreateEventW(NULL, TRUE, FALSE, NULL);
        if (!s.event) {
            fprintf(stderr, "ds4: DirectStorage slot event creation failed\n");
            return 0;
        }
    }
    if (!ResetEvent(s.event)) {
        fprintf(stderr, "ds4: DirectStorage slot event reset failed\n");
        return 0;
    }
    QueryPerformanceCounter(&s.t0);
    s.reqs  = std::move(g_batch);          /* requests must outlive Submit(); the slot owns them */
    s.bytes = g_batch_bytes;
    if (g_batch_gpu && g_gpu_queue_count != 0) {
        s.queue = g_gpu_queues[g_gpu_queue_next];
        g_gpu_queue_next = (g_gpu_queue_next + 1u) % g_gpu_queue_count;
    } else {
        s.queue = g_queue;
    }
    const bool trace_submit=ds4_ds_submit_trace_enabled();
    LARGE_INTEGER enqueue_start={},submit_start={},submit_end={};
    if(trace_submit)QueryPerformanceCounter(&enqueue_start);
    for (size_t i = 0; i < s.reqs.size(); i++) {
        s.queue->EnqueueRequest(&s.reqs[i]);
    }
    s.queue->EnqueueSetEvent(s.event);
    if(trace_submit)QueryPerformanceCounter(&submit_start);
    s.queue->Submit();
    if(trace_submit){QueryPerformanceCounter(&submit_end);ds4_ds_submit_trace("split",s.bytes,s.reqs.size(),enqueue_start,submit_start,submit_end);}
    s.busy = true;
    g_slot_busy++;
    g_slot_next = (g_slot_next + 1u) % DS4_DS_SLOTS;

    g_batch.clear();                        /* moved-from; restore capacity for the next layer */
    g_batch.reserve(kQueueCapacity);
    g_batch_bytes = 0;
    g_batch_gpu = false;
    return 1;
}

extern "C" int ds4_ds_batch_wait(
        uint64_t *out_bytes, double *out_ms, double *out_wait_ms) {
    if (!g_init_ok) return 0;
    if (g_slot_busy == 0) {
        if (out_bytes) *out_bytes = 0;
        if (out_ms)    *out_ms    = 0.0;
        if (out_wait_ms) *out_wait_ms = 0.0;
        return 1;
    }
    ds4_ds_slot &s = g_slots[g_slot_drain];
    LARGE_INTEGER freq, wait_t0, t1;
    QueryPerformanceFrequency(&freq);

    /* A 30 s ceiling: a stalled queue is a fatal condition, not something to wait out silently. */
    QueryPerformanceCounter(&wait_t0);
    const DWORD wait = wait_for_demand(s.event);
    QueryPerformanceCounter(&t1);
    const double wait_ms =
        (double)(t1.QuadPart - wait_t0.QuadPart) * 1000.0 / (double)freq.QuadPart;
    if (out_wait_ms) *out_wait_ms = wait_ms;
    const size_t nreq = s.reqs.size();
    s.busy = false;
    s.reqs.clear();
    g_slot_busy--;
    g_slot_drain = (g_slot_drain + 1u) % DS4_DS_SLOTS;

    if (wait != WAIT_OBJECT_0) {
        fprintf(stderr,
                "ds4: DirectStorage batch of %u requests did not complete within 30 s (wait=%lu)\n",
                (unsigned)nreq, (unsigned long)wait);
        return 0;
    }
    if (!wave_succeeded(s.queue, nreq)) return 0;
    {
        /* Instrument the SPLIT path too, so the 2-slot ring is already measured on the day it goes live.
         * Note this timer spans submit->completion, so once the ring actually overlaps fetch with compute
         * these latencies will include time the GPU was busy - which is exactly what we want to see:
         * overlap should make the instantaneous rate rise while per-batch latency stays flat. */
        const double batch_ms =
            (double)(t1.QuadPart - s.t0.QuadPart) * 1000.0 / (double)freq.QuadPart;
        ds4_ds_stats_record(s.bytes, batch_ms, wait_ms, nreq,
                            s.queue, &s.slow_tags);
        if (out_bytes) *out_bytes = s.bytes;
        if (out_ms) *out_ms = batch_ms;
    }
    return 1;
}

extern "C" int ds4_ds_batch_submit_wait(
        uint64_t *out_bytes, double *out_ms, double *out_wait_ms) {
    if (!g_init_ok) return 0;
    if (g_batch.empty()) {
        if (out_bytes) *out_bytes = 0;
        if (out_ms)    *out_ms    = 0.0;
        if (out_wait_ms) *out_wait_ms = 0.0;
        return 1;
    }

    LARGE_INTEGER freq, t0, wait_t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);

    IDStorageQueue1 *queue = g_queue;
    if (g_batch_gpu && g_gpu_queue_count != 0) {
        queue = g_gpu_queues[g_gpu_queue_next];
        g_gpu_queue_next = (g_gpu_queue_next + 1u) % g_gpu_queue_count;
    }
    if (!queue) return 0;
    const bool trace_submit=ds4_ds_submit_trace_enabled();
    LARGE_INTEGER enqueue_start={},submit_start={},submit_end={};
    if(trace_submit)QueryPerformanceCounter(&enqueue_start);
    for (size_t i = 0; i < g_batch.size(); i++) {
        queue->EnqueueRequest(&g_batch[i]);
    }
    queue->EnqueueSetEvent(g_event);
    if (g_batch_gpu) g_last_sync_gpu_queue = queue;
    if(trace_submit)QueryPerformanceCounter(&submit_start);
    queue->Submit();
    if(trace_submit){QueryPerformanceCounter(&submit_end);ds4_ds_submit_trace("sync",g_batch_bytes,g_batch.size(),enqueue_start,submit_start,submit_end);}

    /* A 30 s ceiling: a stalled queue is a fatal condition, not something to wait out silently. */
    QueryPerformanceCounter(&wait_t0);
    const DWORD wait = wait_for_demand(g_event);
    QueryPerformanceCounter(&t1);
    const double wait_ms =
        (double)(t1.QuadPart - wait_t0.QuadPart) * 1000.0 / (double)freq.QuadPart;
    if (out_wait_ms) *out_wait_ms = wait_ms;

    if (wait != WAIT_OBJECT_0) {
        fprintf(stderr,
                "ds4: DirectStorage batch of %u requests did not complete within 30 s (wait=%lu)\n",
                (unsigned)g_batch.size(), (unsigned long)wait);
        g_batch.clear();
        g_batch_bytes = 0;
        return 0;
    }
    if (!wave_succeeded(queue, g_batch.size())) {
        g_batch.clear();
        g_batch_bytes = 0;
        return 0;
    }

    const double batch_ms =
        (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)freq.QuadPart;
    /* Record unconditionally. Every caller passes NULL/NULL, so these numbers were being computed and
     * thrown away - which is why no correct fetch-rate figure has ever existed for this engine. */
    ds4_ds_stats_record(g_batch_bytes, batch_ms, wait_ms, g_batch.size(),
                        queue, &g_ds_sync_tags);
    if (out_bytes) *out_bytes = g_batch_bytes;
    if (out_ms) *out_ms = batch_ms;
    g_batch.clear();
    g_batch_bytes = 0;
    g_batch_gpu = false;
    return 1;
}


extern "C" void ds4_ds_get_byte_counters(uint64_t *demand,double *demand_ms,uint64_t *background) {
    if(demand)*demand=g_ds_total_bytes;if(demand_ms)*demand_ms=g_ds_total_ms;
    if(background)*background=g_background_submitted_total;
}
extern "C" int ds4_ds_background_submit(const ds4_ds_background_span *spans, uint32_t count) {
    if(!g_init_ok || !spans || !count || count>12 || g_background_busy ||
        ds4_ds_batch_inflight()!=0 || !g_batch.empty()) return 0;
    if(!g_background_queue) {
        DSTORAGE_QUEUE_DESC desc={};desc.SourceType=DSTORAGE_REQUEST_SOURCE_FILE;
        desc.Capacity=256;desc.Priority=DSTORAGE_PRIORITY_LOW;desc.Name="ds4-displaced-background";
        HRESULT hr=g_factory->CreateQueue(&desc,__uuidof(IDStorageQueue1),(void **)&g_background_queue);
        if(FAILED(hr) || !g_background_queue) {fprintf(stderr,"ds4: background LOW DS queue creation failed\n");return -1;}
        g_background_event=CreateEventW(NULL,TRUE,FALSE,NULL);
        if(!g_background_event) return -1;
        fprintf(stderr,"ds4: background promotion uses separate LOW DirectStorage queue; demand remains HIGH\n");
    }
    g_background_requests.clear();g_background_bytes=0;
    const uint32_t chunk=1024u*1024u;
    for(uint32_t i=0;i<count;i++) {
        const auto &span=spans[i];
        if(!span.destination || !span.bytes || span.offset>g_file_size || span.bytes>g_file_size-span.offset) return -1;
        for(uint32_t done=0;done<span.bytes;) {
            uint32_t n=span.bytes-done;if(n>chunk)n=chunk;
            DSTORAGE_REQUEST request={};
            request.Options.SourceType=DSTORAGE_REQUEST_SOURCE_FILE;
            request.Options.DestinationType=DSTORAGE_REQUEST_DESTINATION_MEMORY;
            request.Source.File.Source=g_file;request.Source.File.Offset=span.offset+done;request.Source.File.Size=n;
            request.Destination.Memory.Buffer=(char*)span.destination+done;request.Destination.Memory.Size=n;
            request.UncompressedSize=n;g_background_requests.push_back(request);g_background_bytes+=n;done+=n;
        }
    }
    if(g_background_requests.size()+1u>256u || !ResetEvent(g_background_event)) return -1;
    QueryPerformanceCounter(&g_background_started);g_background_saw_demand=false;
    for(const auto &request:g_background_requests) g_background_queue->EnqueueRequest(&request);
    g_background_queue->EnqueueSetEvent(g_background_event);g_background_queue->Submit();g_background_busy=true;g_background_submitted_total+=g_background_bytes;
    return 1;
}
extern "C" int ds4_ds_background_poll(uint64_t *bytes, double *ms) {
    if(!g_background_busy)return 0;
    DWORD status=WaitForSingleObject(g_background_event,0);
    if(status==WAIT_TIMEOUT)return 0;
    if(status!=WAIT_OBJECT_0)return -1;
    g_background_busy=false;
    if(!wave_succeeded(g_background_queue,g_background_requests.size()))return -1;
    LARGE_INTEGER now,freq;QueryPerformanceCounter(&now);QueryPerformanceFrequency(&freq);
    if(bytes)*bytes=g_background_bytes;
    if(ms)*ms=(double)(now.QuadPart-g_background_started.QuadPart)*1000.0/freq.QuadPart;
    g_background_requests.clear();return 1;
}
extern "C" int ds4_ds_background_finish(void) {
    if(!g_background_busy)return 1;
    if(WaitForSingleObject(g_background_event,30000)!=WAIT_OBJECT_0) {
        fprintf(stderr,"ds4: background DS drain exceeded 30s; refusing to free live buffers\n");return 0;
    }
    return ds4_ds_background_poll(NULL,NULL)==1;
}
/* Submission counters are credited only after every physical request succeeds. */
static int selftest_submit(uint64_t expected,uint64_t *completed,const char *leg,uint32_t index) {
    uint64_t bytes=0;
    if(!ds4_ds_batch_submit_wait(&bytes,NULL,NULL) || bytes!=expected) {
        fprintf(stderr,"ds4: DirectStorage selftest: %s incomplete at %u (expected=%llu completed=%llu)\n",
            leg,index,(unsigned long long)expected,(unsigned long long)bytes);return 0;
    }
    *completed+=bytes;return 1;
}
extern "C" int ds4_ds_selftest(int fd, uint64_t start_offset, uint64_t span_bytes,
                               uint32_t chunk_bytes, uint32_t batch) {
    if (!ds4_ds_init_from_fd(fd)) return 0;
    if (chunk_bytes == 0 || span_bytes < chunk_bytes ||
        span_bytes/chunk_bytes > UINT32_MAX || start_offset > g_file_size ||
        chunk_bytes > g_file_size-start_offset) {
        fprintf(stderr,"ds4: DirectStorage selftest: invalid source span/chunk count\n");return 0;
    }
    if (batch == 0) batch = 1;
    const uint32_t requested_batch=batch;
    const uint32_t safe_batch=ds4_ds_batch_capacity()/2u;
    if (!safe_batch) {fprintf(stderr,"ds4: DirectStorage selftest: no safe physical capacity\n");return 0;}
    if (batch > safe_batch) batch=safe_batch;
    fprintf(stderr,"ds4: selftest requested_batch=%u effective_batch=%u physical_capacity=%u parts=%u\n",
        requested_batch,batch,(unsigned)kQueueCapacity,g_read_parts);

    const uint32_t chunks = (uint32_t)(span_bytes / chunk_bytes);
    const uint64_t total  = (uint64_t)chunks * chunk_bytes;
    MEMORYSTATUSEX memory={};memory.dwLength=sizeof(memory);
    const uint64_t reserve=2ull*1024*1024*1024;
    if (total > SIZE_MAX/3u || !GlobalMemoryStatusEx(&memory) ||
        memory.ullAvailPhys <= reserve || total > (memory.ullAvailPhys-reserve)/3u ||
        (uint64_t)chunks*4u*sizeof(uint64_t) > memory.ullAvailPhys-reserve-3u*total) {
        fprintf(stderr,"ds4: DirectStorage selftest: buffers/offsets exceed available physical RAM minus2GiB\n");return 0;
    }
    uint64_t qd1_completed=0,batch_completed=0,pread_completed=0;


    /* Offsets must SCATTER, not run contiguously.
     *
     * A contiguous span flatters every transport: the array prefetches it and even plain pread hit
     * 3.2-3.4 GB/s, while the same pread manages only ~1.8 GB/s during real decode. Routed experts
     * live at essentially random offsets across a 90 GB file, so a sequential probe measures a
     * workload ds4 never runs. Deterministic LCG, so the two transports read the SAME offsets in
     * the same order and the comparison stays apples-to-apples. */
    /* THREE DISJOINT offset sets, one per transport.
     *
     * Reading the same offsets with each transport in turn is invalid: whichever runs last reads
     * data the earlier passes already pulled into the Windows page cache. In the first version pread
     * ran third and was flattered by exactly that. Correctness and timing therefore get separate
     * passes -- a small same-offset pass proves byte equality, and the timed pass gives each
     * transport its own offsets drawn from the same distribution, so none warms another's cache. */
    std::vector<uint64_t> offsets(chunks);       /* DS QD1   */
    std::vector<uint64_t> offsets_b(chunks);     /* DS batch */
    std::vector<uint64_t> offsets_p(chunks);     /* pread    */
    std::vector<uint64_t> offsets_g(chunks);     /* DS GPU destination */
    {
        const uint64_t file_span = g_file_size;
        const uint64_t slots=(file_span-start_offset)/chunk_bytes;
        /* Vary the offsets on EVERY invocation. With a fixed seed the second selftest re-read the
         * first one's offsets straight out of the Windows page cache and reported 6.99 GB/s against
         * 2.35 for the same code -- a RAM measurement wearing a disk measurement's clothes. Each run
         * must touch fresh regions of the file. */
        static uint64_t s_selftest_run = 0;
        s_selftest_run++;
        uint64_t rng = 0x9E3779B97F4A7C15ull ^ ((uint64_t)chunks * 0x100000001B3ull)
                                             ^ (s_selftest_run * 0xD1B54A32D192ED03ull);
        const uint64_t nslots = (slots != 0 ? slots : 1ull);
        for (uint32_t k = 0; k < chunks; k++) {
            rng = rng * 6364136223846793005ull + 1442695040888963407ull;
            offsets[k]   = start_offset + ((rng >> 17) % nslots) * (uint64_t)chunk_bytes;
            rng = rng * 6364136223846793005ull + 1442695040888963407ull;
            offsets_b[k] = start_offset + ((rng >> 17) % nslots) * (uint64_t)chunk_bytes;
            rng = rng * 6364136223846793005ull + 1442695040888963407ull;
            offsets_p[k] = start_offset + ((rng >> 17) % nslots) * (uint64_t)chunk_bytes;
            rng = rng * 6364136223846793005ull + 1442695040888963407ull;
            offsets_g[k] = start_offset + ((rng >> 17) % nslots) * (uint64_t)chunk_bytes;
        }
    }

    char *buf_qd1   = (char *)malloc(total);
    char *buf_batch = (char *)malloc(total);
    char *buf_pread = (char *)malloc(total);
    if (!buf_qd1 || !buf_batch || !buf_pread) {
        fprintf(stderr, "ds4: DirectStorage selftest: cannot allocate %.2f GiB x3\n",
                (double)total / 1073741824.0);
        free(buf_qd1); free(buf_batch); free(buf_pread);
        return 0;
    }

    LARGE_INTEGER freq, a, b;
    QueryPerformanceFrequency(&freq);

    /* --- queue depth 1: one request per submit, exactly what the decode path does today --- */
    QueryPerformanceCounter(&a);
    for (uint32_t i = 0; i < chunks; i++) {
        ds4_ds_batch_reset();
        if (!ds4_ds_batch_add(offsets[i],chunk_bytes,buf_qd1+(uint64_t)i*chunk_bytes)) {
            fprintf(stderr,"ds4: DirectStorage selftest: QD1 add refused at chunk %u\n",i);
            ds4_ds_batch_reset();free(buf_qd1);free(buf_batch);free(buf_pread);return 0;
        }
        if (!selftest_submit(chunk_bytes,&qd1_completed,"QD1",i)) {
            fprintf(stderr, "ds4: DirectStorage selftest: QD1 submit failed at chunk %u\n", i);
            free(buf_qd1); free(buf_batch); free(buf_pread);
            return 0;
        }
    }
    QueryPerformanceCounter(&b);
    const double qd1_s = (double)(b.QuadPart - a.QuadPart) / (double)freq.QuadPart;

    /* --- batched: `batch` requests per submit --- */
    QueryPerformanceCounter(&a);
    uint32_t i = 0;
    while (i < chunks) {
        ds4_ds_batch_reset();
        uint32_t n = 0;
        while (n < batch && i < chunks) {
            if (!ds4_ds_batch_add(offsets_b[i],chunk_bytes,buf_batch+(uint64_t)i*chunk_bytes)) {
                fprintf(stderr,"ds4: DirectStorage selftest: batch add refused at chunk %u (physical=%u)\n",i,ds4_ds_batch_count());
                ds4_ds_batch_reset();free(buf_qd1);free(buf_batch);free(buf_pread);return 0;
            }
            i++; n++;
        }
        if (!selftest_submit((uint64_t)n*chunk_bytes,&batch_completed,"batch",i)) {
            fprintf(stderr, "ds4: DirectStorage selftest: batched submit failed near chunk %u\n", i);
            free(buf_qd1); free(buf_batch); free(buf_pread);
            return 0;
        }
    }
    QueryPerformanceCounter(&b);
    const double batch_s = (double)(b.QuadPart - a.QuadPart) / (double)freq.QuadPart;

    /* --- pread reference, for both timing and byte equality --- */
    QueryPerformanceCounter(&a);
    for (uint32_t k = 0; k < chunks; k++) {
        const uint64_t off  = offsets_p[k];
        const size_t   want = (size_t)chunk_bytes;
        const uint64_t done = (uint64_t)k * chunk_bytes;
        OVERLAPPED ov;
        memset(&ov, 0, sizeof(ov));
        ov.Offset     = (DWORD)(off & 0xffffffffu);
        ov.OffsetHigh = (DWORD)((off >> 32) & 0xffffffffu);
        DWORD got = 0;
        if (!ReadFile((HANDLE)_get_osfhandle(fd), buf_pread + done, (DWORD)want, &got, &ov) ||
            got != chunk_bytes) {
            fprintf(stderr, "ds4: DirectStorage selftest: reference pread failed at %llu\n",
                    (unsigned long long)off);
            free(buf_qd1); free(buf_batch); free(buf_pread);
            return 0;
        }
        pread_completed += got;
    }
    QueryPerformanceCounter(&b);
    const double pread_s = (double)(b.QuadPart - a.QuadPart) / (double)freq.QuadPart;

    /* Correctness, measured separately from speed: a small pass where DS and pread read the SAME
     * offsets. The timed passes above deliberately use disjoint offsets, so their buffers cannot be
     * compared to each other -- conflating the two is what made the first version invalid. */
    int match_qd1 = 1, match_batch = 1;
    {
        const uint32_t vchunks = chunks < 8u ? chunks : 8u;
        uint64_t verified=0;
        for(uint32_t begin=0;begin<vchunks;) {
            ds4_ds_batch_reset();const uint32_t n=(vchunks-begin<batch?vchunks-begin:batch);
            for(uint32_t k=begin;k<begin+n;k++) {
                if(!ds4_ds_batch_add(offsets[k],chunk_bytes,buf_batch+(uint64_t)k*chunk_bytes)) {
                    fprintf(stderr,"ds4: DirectStorage selftest: correctness add refused at chunk %u\n",k);
                    ds4_ds_batch_reset();free(buf_qd1);free(buf_batch);free(buf_pread);return 0;
                }
            }
            if(!selftest_submit((uint64_t)n*chunk_bytes,&verified,"correctness",begin)) {
                free(buf_qd1);free(buf_batch);free(buf_pread);return 0;
            }
            begin+=n;
        }
        for (uint32_t k = 0; k < vchunks; k++) {
            OVERLAPPED ov;
            memset(&ov, 0, sizeof(ov));
            ov.Offset     = (DWORD)(offsets[k] & 0xffffffffu);
            ov.OffsetHigh = (DWORD)((offsets[k] >> 32) & 0xffffffffu);
            DWORD got = 0;
            if (!ReadFile((HANDLE)_get_osfhandle(fd), buf_pread + (uint64_t)k * chunk_bytes,
                          chunk_bytes, &got, &ov) || got != chunk_bytes) {
                fprintf(stderr, "ds4: DirectStorage selftest: verification pread failed\n");
                free(buf_qd1); free(buf_batch); free(buf_pread);
                return 0;
            }
        }
        match_batch = memcmp(buf_batch, buf_pread, (size_t)vchunks * chunk_bytes) == 0;
        match_qd1=memcmp(buf_qd1,buf_pread,(size_t)vchunks*chunk_bytes)==0;
    }
    if(qd1_completed!=total || batch_completed!=total || pread_completed!=total) {
        fprintf(stderr,"ds4: DirectStorage selftest: incomplete byte accounting\n");
        free(buf_qd1);free(buf_batch);free(buf_pread);return 0;
    }
    fprintf(stderr,"ds4: [DS4_SELFTEST_BYTES_V2] requested=%llu qd1=%llu batch=%llu pread=%llu\n",
        (unsigned long long)total,(unsigned long long)qd1_completed,
        (unsigned long long)batch_completed,(unsigned long long)pread_completed);
    const double gib = (double)total / 1073741824.0;

    /* [2026-08-20] A rate above the hardware ceiling means the TRANSFER DID NOT HAPPEN.
     *
     * Observed: the GPU line printed 10737418.24 GB/s. Working back from a 1 GiB total, that is an
     * elapsed time of ~1e-7 s — 100 NANOSECONDS for a gibibyte. QueryPerformanceCounter resolves far
     * below that, so this is NOT timer granularity: the requests were skipped or failed silently and
     * the code reported the non-event as spectacular throughput.
     *
     * The existing `x > 0.0 ? x : 1.0` guards only catch EXACTLY zero, which is the one case that
     * cannot occur. Clamping is also wrong here — it would turn an impossible number into a
     * PLAUSIBLE one, which is strictly worse: a reader can spot 10737418 GB/s, but not a fabricated
     * 35.0. So an implausible result is reported AS A FAILURE and the rate is suppressed.
     *
     * Ceiling: this drive measures ~35 GB/s in CrystalDiskMark, and prefill reaches ~15.7 GB/s
     * through this path. 100 GB/s is far above anything reachable and far below the 1e6-scale
     * garbage, so it separates the two cleanly without needing tuning. */
    const double DS4_SELFTEST_IMPLAUSIBLE_GBS = 100.0;
    struct ds4_rate {
        static double gbs(uint64_t bytes, double seconds) {
            return seconds > 0.0 ? (double)bytes / seconds / 1e9 : 0.0;
        }
        static int implausible(uint64_t bytes, double seconds, double ceiling) {
            const double r = gbs(bytes, seconds);
            return (seconds <= 0.0 || r > ceiling) ? 1 : 0;
        }
    };
    const int rate_bad =
        ds4_rate::implausible(pread_completed, pread_s, DS4_SELFTEST_IMPLAUSIBLE_GBS) ||
        ds4_rate::implausible(qd1_completed, qd1_s,   DS4_SELFTEST_IMPLAUSIBLE_GBS) ||
        ds4_rate::implausible(batch_completed, batch_s, DS4_SELFTEST_IMPLAUSIBLE_GBS);
    if (rate_bad) {
        fprintf(stderr,
                "ds4: [DS4_SELFTEST_IMPLAUSIBLE_V1] a leg completed faster than the hardware allows "
                "(pread=%.9f s qd1=%.9f s batch=%.9f s for %.2f GiB) — the transfer DID NOT HAPPEN. "
                "RATES SUPPRESSED; SELFTEST FAILED.\n",
                pread_s, qd1_s, batch_s, gib);
    }

    if(!rate_bad)fprintf(stderr,
            "ds4: === DirectStorage selftest: %.2f GiB in %u x %.2f MiB chunks ===\n"
            "ds4:   pread   (QD1, current path) %6.2f s = %5.2f GB/s\n"
            "ds4:   DS      (QD1)               %6.2f s = %5.2f GB/s\n"
            "ds4:   DS      (batch of %-4u)     %6.2f s = %5.2f GB/s   <-- %.2fx over DS QD1\n"
            "ds4:   byte-identical to pread: DS-QD1 %s, DS-batched %s\n",
            gib, chunks, (double)chunk_bytes / 1048576.0,
            pread_s,  (double)pread_completed / pread_s  / 1e9,
            qd1_s,    (double)qd1_completed / qd1_s    / 1e9,
            batch,
            batch_s,  (double)batch_completed / batch_s  / 1e9,
            qd1_s / (batch_s > 0.0 ? batch_s : 1.0),
            match_qd1 ? "YES" : "NO",
            match_batch ? "YES" : "NO");

    int match_gpu = 1,gpu_rate_bad=0;
    uint64_t gpu_completed=0;
    const char *gpu_test = getenv("DS4_DS_GPU_SELFTEST");
    if (gpu_test && gpu_test[0]) {
        void *gpu_ptr = NULL;
        match_gpu = ds4_ds_gpu_arena_create(total, &gpu_ptr);
        QueryPerformanceCounter(&a);
        uint32_t gi = 0;
        while (match_gpu && gi < chunks) {
            ds4_ds_batch_reset();
            uint32_t n = 0;
            while (n < batch && gi < chunks) {
                if (!ds4_ds_batch_add_gpu(offsets_g[gi], chunk_bytes,
                                          (uint64_t)gi * chunk_bytes)) {
                    fprintf(stderr,"ds4: DirectStorage selftest: GPU add refused at chunk %u (physical=%u)\n",gi,ds4_ds_batch_count());
                    ds4_ds_batch_reset();match_gpu=0;break;
                }
                gi++; n++;
            }
            if (match_gpu && !selftest_submit((uint64_t)n*chunk_bytes,&gpu_completed,"GPU",gi)) match_gpu=0;
        }
        QueryPerformanceCounter(&b);
        const double gpu_s = (double)(b.QuadPart - a.QuadPart) / (double)freq.QuadPart;
        if(match_gpu && (gi!=chunks || gpu_completed!=total))match_gpu=0;
        fprintf(stderr,"ds4: [DS4_SELFTEST_GPU_BYTES_V2] requested=%llu completed=%llu\n",
            (unsigned long long)total,(unsigned long long)gpu_completed);
        if (match_gpu) {
            const cudaError_t cerr = cudaMemcpy(buf_batch, gpu_ptr, (size_t)total,
                                                cudaMemcpyDeviceToHost);
            if (cerr != cudaSuccess) {
                fprintf(stderr, "ds4: DirectStorage GPU selftest readback failed: %s\n",
                        cudaGetErrorString(cerr));
                match_gpu = 0;
            }
        }
        for (uint32_t k = 0; match_gpu && k < chunks; k++) {
            OVERLAPPED ov;
            memset(&ov, 0, sizeof(ov));
            ov.Offset = (DWORD)(offsets_g[k] & 0xffffffffu);
            ov.OffsetHigh = (DWORD)((offsets_g[k] >> 32) & 0xffffffffu);
            DWORD got = 0;
            if (!ReadFile((HANDLE)_get_osfhandle(fd),
                          buf_pread + (uint64_t)k * chunk_bytes,
                          chunk_bytes, &got, &ov) || got != chunk_bytes) {
                match_gpu = 0;
            }
        }
        if (match_gpu) match_gpu = memcmp(buf_batch, buf_pread, (size_t)total) == 0;
        /* An implausible GPU rate is the case actually observed in the wild: 10737418.24 GB/s,
         * i.e. ~1e-7 s for 1 GiB. Report it as a NON-EVENT rather than a number. Two separate
         * fprintf calls, not one with a conditional format string - the two formats take different
         * argument counts and a single varargs call would be undefined behaviour. */
        gpu_rate_bad=match_gpu && ds4_rate::implausible(gpu_completed,gpu_s,DS4_SELFTEST_IMPLAUSIBLE_GBS);
        if(!match_gpu) {
            fprintf(stderr,"ds4: DirectStorage GPU selftest FAILED; no throughput reported\n");
        } else if (gpu_rate_bad) {
            fprintf(stderr,
                    "ds4:   DS GPU  (batch of %-4u)     %.9f s = TRANSFER DID NOT HAPPEN "
                    "(faster than hardware allows); byte-identical %s\n",
                    batch, gpu_s, match_gpu ? "YES" : "NO");
        } else {
            fprintf(stderr,
                    "ds4:   DS GPU  (batch of %-4u)     %6.2f s = %5.2f GB/s; byte-identical %s\n",
                    batch, gpu_s, ds4_rate::gbs(gpu_completed, gpu_s), match_gpu ? "YES" : "NO");
        }
        ds4_ds_gpu_arena_release();
    }

    free(buf_qd1); free(buf_batch); free(buf_pread);
    return (match_qd1 && match_batch && match_gpu && !rate_bad && !gpu_rate_bad) ? 1 : 0;
}

extern "C" void ds4_ds_shutdown(void) {
    release_all();
    /* Any surviving prefill arena holds a reference on the D3D12 device, so
     * release waves and arenas before the decode arena and the device. */
    ds4_ds_prefill_waves_release();
    std::vector<ds4_ds_prefill_wave>().swap(g_prefill_waves);
    prefill_arenas_release();
    gpu_arena_release();
    if (g_d3d_device) {
        const ULONG refs = g_d3d_device->Release();
        g_d3d_device = NULL;
        fprintf(stderr, "DS4COMBO_DS_SHUTDOWN d3d12_device_refcount_after_release=%lu\n",
                (unsigned long)refs);
        fflush(stderr);
    }
    if (g_dll) { FreeLibrary(g_dll); g_dll = NULL; }
    g_init_attempted = 0;
}
