// SPDX-License-Identifier: MIT
// Tensor-layout adapter around the pinned Basalt expert kernels; no alternate matrix math.
#include "qfn_basalt_nvfp4.h"
#include "mmq-bridge.hpp"
#include "calibrated-a4.hpp"
#include "basalt/prefill/moe_fp4.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace {
namespace fp4 = basalt::prefill::fp4;
namespace native = basalt::kernels;
constexpr int E = 2560, FF = 640, K = 10, EXPERTS = 512, WINDOW = 16384;
// Upstream native SwiGLU launches ceil(nt*K*FF/256) blocks on grid.x.
static_assert(K * FF % 256 == 0);
constexpr int MAX_NATIVE_TOKENS = INT_MAX / (K * FF / 256);
static_assert(MAX_NATIVE_TOKENS == 85899345);
constexpr uint64_t BLOB = 2764816, DOWN = 1843200, TAIL = 2764800;
std::atomic<bool> prepared[64]{};

#define DEVICE_REQUIRE(condition, message) \
    do { if (!(condition)) { printf("qfn NVFP4: %s\n", message); asm volatile("trap;"); return; } } while (0)

struct Arena {
    uint8_t * base;
    uint64_t used = 0;
    bool valid = true;
    void * take(uint64_t bytes) {
        if (bytes > UINT64_MAX - 255 || used > UINT64_MAX - ((bytes + 255) & ~uint64_t(255))) {
            valid = false;
            return nullptr;
        }
        const uint64_t at = used;
        used += (bytes + 255) & ~uint64_t(255);
        return base ? base + at : nullptr;
    }
};

struct Short {
    void * xq;
    void * scratch;
    unsigned long long * ptr;
    int32_t * start, * dst, * tok, * count;
};
Short short_layout(Arena & a, int nt) {
    const int64_t rows = int64_t(nt) * K;
    Short s;
    s.xq = a.take(uint64_t(nt) * (E / 32) * 36);
    s.scratch = a.take(native::native_expert_scratch_bytes(rows, FF));
    s.ptr = static_cast<unsigned long long *>(a.take(uint64_t(rows) * 8));
    s.start = static_cast<int32_t *>(a.take(uint64_t(rows + 1) * 4));
    s.dst = static_cast<int32_t *>(a.take(uint64_t(rows) * 4));
    s.tok = static_cast<int32_t *>(a.take(uint64_t(rows) * 4));
    s.count = static_cast<int32_t *>(a.take(4));
    return s;
}
struct Long {
    void * ax, * ah;
    float * sx, * sh, * h, * dm;
    int32_t * tok, * scatter, * bounds, * tiles;
};
Long long_layout(Arena & a, int nt) {
    const int64_t rows = int64_t(nt) * K;
    Long s;
    s.ax = a.take(fp4::token_bytes(nt, E));
    s.sx = static_cast<float *>(a.take(uint64_t(nt) * 4));
    s.h = static_cast<float *>(a.take(uint64_t(rows) * FF * 4));
    s.ah = a.take(fp4::a4_bytes(rows, FF, EXPERTS));
    s.sh = static_cast<float *>(a.take(fp4::scale_count(rows, EXPERTS) * 4));
    s.dm = static_cast<float *>(a.take(uint64_t(rows) * E * 4));
    s.tok = static_cast<int32_t *>(a.take(uint64_t(rows) * 4));
    s.scatter = static_cast<int32_t *>(a.take(uint64_t(rows) * 4));
    s.bounds = static_cast<int32_t *>(a.take((EXPERTS + 1) * 4));
    s.tiles = static_cast<int32_t *>(a.take((EXPERTS + 1) * 4));
    return s;
}
bool is_fp4(int nt, int mode) { return mode == QFN_NVFP4_W4A4 || (mode == QFN_NVFP4_AUTO && nt >= 1024); }
bool shape_ok(int nt, int top_k, uint32_t slots, int mode) {
    if(mode==QFN_NVFP4_CALIBRATED_DECODE)return nt>=1&&nt<=4&&top_k==K&&slots>0&&slots<=INT_MAX;
    return nt > 0 && nt <= INT_MAX / K && top_k == K && slots > 0 && slots <= INT_MAX &&
           mode >= QFN_NVFP4_AUTO && mode <= QFN_NVFP4_CALIBRATED_A4 &&
           ((is_fp4(nt, mode)||mode==QFN_NVFP4_MMQ_Q8||mode==QFN_NVFP4_CALIBRATED_A4) ? slots == EXPERTS : nt <= MAX_NATIVE_TOKENS);
}

__global__ void validate(const int32_t * ids, const uint8_t * raw, int rows, uint32_t slots, bool unique) {
    const int r = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int32_t id = ids[r];
    DEVICE_REQUIRE(id >= 0 && uint32_t(id) < slots, "NVFP4 physical slot is outside the bank/slab");
    if (unique) for (int j = r - r % K; j < r; ++j)
        DEVICE_REQUIRE(ids[j] != id, "NVFP4 W4A4 requires distinct logical experts within one token");
    const float * tail = reinterpret_cast<const float *>(raw + uint64_t(id) * BLOB + TAIL);
    DEVICE_REQUIRE(isfinite(tail[0]) && tail[0] > 0 && isfinite(tail[1]) && tail[1] > 0 &&
                   isfinite(tail[2]) && tail[2] > 0, "NVFP4 FP32 scale tail is not finite and positive");
}
__global__ void setup_short(const int32_t * ids, const uint8_t * raw, int rows, Short s) {
    const int r = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r == 0) { *s.count = rows; s.start[rows] = rows; }
    if (r >= rows) return;
    s.ptr[r] = reinterpret_cast<unsigned long long>(raw + uint64_t(ids[r]) * BLOB);
    s.start[r] = r;
    s.dst[r] = r;
    s.tok[r] = r / K;
}

// The existing ggml mm_ids_helper generic K path, with one warp per expert and token-only gathers.
// Adapted from ggml/src/ggml-cuda/mmid.cu (MIT, Copyright 2023-2026 The ggml authors).
__global__ void group_bank(const int32_t * ids, int nt, int32_t * tok, int32_t * scatter, int32_t * bounds) {
    extern __shared__ uint32_t packed[];
    const int expert = blockIdx.x, lane = threadIdx.x;
    int lower = 0, count = 0;
    for (int t = 0; t < nt; ++t) {
        const int id = lane < K ? ids[t * K + lane] : INT_MAX;
        lower += id < expert;
        const bool mine = id == expert;
        if (mine) packed[count] = uint32_t(t) | (uint32_t(lane) << 22);
        count += __any_sync(0xffffffffu, mine);
    }
    for (int delta = 16; delta; delta >>= 1) lower += __shfl_down_sync(0xffffffffu, lower, delta);
    lower = __shfl_sync(0xffffffffu, lower, 0);
    __syncwarp();
    for (int i = lane; i < count; i += 32) {
        const uint32_t v = packed[i];
        tok[lower + i] = int32_t(v & 0x003fffffu);
        scatter[lower + i] = tok[lower + i] * K + int32_t(v >> 22);
    }
    if (lane == 0) {
        bounds[expert] = lower;
        if (expert == EXPERTS - 1) bounds[EXPERTS] = lower + count;
    }
}
__global__ void scatter_rows(const float * sorted, const int32_t * map, float * out, int64_t values) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < values) out[int64_t(map[i / E]) * E + i % E] = sorted[i];
}
int device(int & dev) {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return int(err);
    if (dev < 0 || dev >= 64) return int(cudaErrorInvalidDevice);
    int major = 0, minor = 0;
    err = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    if (err == cudaSuccess) err = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    if (err != cudaSuccess) return int(err);
    return major == 12 && minor == 0 ? 0 : int(cudaErrorInvalidDevice);
}
} // namespace

namespace qfn_mmq {
void group(const int32_t * ids,const uint8_t * raw,int nt,int32_t * tok,int32_t * order,
           int32_t * bounds,void * stream) {
    const int rows=nt*K;const auto cs=static_cast<cudaStream_t>(stream);
    validate<<<(rows+255)/256,256,0,cs>>>(ids,raw,rows,EXPERTS,true);
    group_bank<<<EXPERTS,32,nt*sizeof(uint32_t),cs>>>(ids,nt,tok,order,bounds);
}
void scatter(const float * sorted,const int32_t * map,float * out,int64_t values,void * stream) {
    scatter_rows<<<(values+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(sorted,map,out,values);
}
}

extern "C" int qfn_basalt_nvfp4_abi(void) { return 1; }

extern "C" uint64_t qfn_basalt_nvfp4_workspace_bytes(int nt, int top_k, uint32_t slots, int mode) {
    if (!shape_ok(nt, top_k, slots, mode)) return 0;
    if(mode==QFN_NVFP4_CALIBRATED_DECODE)return qfn_calibrated::decode_workspace_bytes(nt);
    if(mode==QFN_NVFP4_MMQ_Q8) return qfn_mmq::workspace_bytes(nt);
    if(mode==QFN_NVFP4_CALIBRATED_A4) return qfn_calibrated::workspace_bytes(nt);
    Arena a{nullptr};
    if (is_fp4(nt, mode)) (void) long_layout(a, std::min(nt, WINDOW));
    else (void) short_layout(a, nt);
    return a.valid ? a.used : 0;
}

extern "C" int qfn_basalt_nvfp4_prepare(void * stream) {
    int dev = 0;
    if (int err = device(dev)) return err;
    if (prepared[dev].load(std::memory_order_acquire)) return 0;
    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    cudaStreamCaptureStatus capturing;
    cudaError_t err = cudaStreamIsCapturing(cs, &capturing);
    if (err != cudaSuccess) return int(err);
    if (capturing != cudaStreamCaptureStatusNone) return int(cudaErrorStreamCaptureUnsupported);
    err = cudaFuncSetAttribute(group_bank, cudaFuncAttributeMaxDynamicSharedMemorySize, WINDOW * sizeof(uint32_t));
    if (err != cudaSuccess) return int(err);
    // Zero-route launches configure both FP4 specializations and native decode kernels without reading weights.
    Arena sizes{nullptr};
    const Short ignored = short_layout(sizes, 1);
    (void) ignored;
    const uint64_t xoff = sizes.used;
    sizes.take(E * 4);
    const uint64_t boff = sizes.used;
    sizes.take(8);
    const uint64_t axoff = sizes.used;
    sizes.take(fp4::token_bytes(1, E));
    const uint64_t ahoff = sizes.used;
    sizes.take(fp4::a4_bytes(1, FF, 1));
    const uint64_t gsoff = sizes.used;
    sizes.take(fp4::scale_count(1, 1) * 4);
    const uint64_t tileoff = sizes.used;
    sizes.take(8);
    uint8_t * tiny = nullptr;
    err = cudaMalloc(reinterpret_cast<void **>(&tiny), size_t(sizes.used));
    if (err != cudaSuccess) return int(err);
    err = cudaMemsetAsync(tiny, 0, size_t(sizes.used), cs);
    if (err == cudaSuccess) {
        Arena a{tiny};
        Short s = short_layout(a, 1);
        native::quantize_q8_1_rows(reinterpret_cast<float *>(tiny + xoff), 1, E, s.xq, cs);
        const auto layout = native::native_expert_layout(40, 40, E, FF);
        native::native_expert_grouped(layout, s.ptr, s.start, s.count, s.dst, s.tok, 1, 1, s.xq,
                                      s.scratch, nullptr, cs);
        fp4::quantize_tokens(reinterpret_cast<float *>(tiny + xoff), 1, E, E,
                             tiny + axoff, reinterpret_cast<float *>(tiny + gsoff), cs);
        fp4::plan(reinterpret_cast<int32_t *>(tiny + boff), 1, reinterpret_cast<int32_t *>(tiny + tileoff), cs);
        fp4::quantize(reinterpret_cast<float *>(tiny + xoff), nullptr,
                      reinterpret_cast<int32_t *>(tiny + boff), reinterpret_cast<int32_t *>(tiny + tileoff), 1,
                      tiny + ahoff, reinterpret_cast<float *>(tiny + gsoff), FF, FF, 1, cs);
        fp4::Product p;
        p.n = 1; p.max_rows = 1; p.bounds = reinterpret_cast<int32_t *>(tiny + boff);
        p.w_rows = 2 * FF; p.w_cols = E; p.swiglu = true;
        p.ids = reinterpret_cast<int32_t *>(tiny + boff);
        p.a4 = tiny + axoff;
        fp4::run(p, cs);
        p.w_rows = E; p.w_cols = FF; p.swiglu = false; p.ids = nullptr;
        fp4::run(p, cs);
        validate<<<1, 256, 0, cs>>>(nullptr, nullptr, 0, 0, false);
        setup_short<<<1, 256, 0, cs>>>(nullptr, nullptr, 0, s);
        group_bank<<<EXPERTS, 32, 0, cs>>>(nullptr, 0, nullptr, nullptr, reinterpret_cast<int32_t *>(s.scratch));
        scatter_rows<<<1, 256, 0, cs>>>(nullptr, nullptr, nullptr, 0);
        err = cudaGetLastError();
    }
    if (err == cudaSuccess) err = cudaStreamSynchronize(cs);
    const cudaError_t freed = cudaFree(tiny);
    if (err == cudaSuccess) err = freed;
    if (err == cudaSuccess) {
        qfn_mmq::prepare(stream);
        err=static_cast<cudaError_t>(qfn_calibrated::prepare(stream));
        if(err==cudaSuccess)err=cudaStreamSynchronize(cs);
    }
    if(err==cudaSuccess) {
        prepared[dev].store(true, std::memory_order_release);
    }
    return int(err);
}

extern "C" int qfn_basalt_nvfp4_run(const float * x, const int32_t * ids, const uint8_t * raw,
    uint64_t stride, uint32_t slots, int nt, int top_k, float * out, void * workspace,
    uint64_t bytes, void * stream, int mode) {
    if(mode==QFN_NVFP4_CALIBRATED_A4||mode==QFN_NVFP4_CALIBRATED_DECODE)return int(cudaErrorInvalidValue); // explicit aux export required
    const uint64_t need = qfn_basalt_nvfp4_workspace_bytes(nt, top_k, slots, mode);
    if (!need || !x || !ids || !raw || !out || !workspace || bytes < need || stride != BLOB ||
        (reinterpret_cast<uintptr_t>(workspace) & 255) || (reinterpret_cast<uintptr_t>(raw) & 15) ||
        (reinterpret_cast<uintptr_t>(x) & 15) || (reinterpret_cast<uintptr_t>(out) & 15) ||
        (reinterpret_cast<uintptr_t>(ids) & 3))
        return int(cudaErrorInvalidValue);
    int dev = 0;
    cudaError_t status = cudaGetDevice(&dev);
    if (status != cudaSuccess) return int(status);
    if (dev < 0 || dev >= 64 || !prepared[dev].load(std::memory_order_acquire)) return int(cudaErrorNotReady);
    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    if(mode==QFN_NVFP4_MMQ_Q8) return qfn_mmq::run(x,ids,raw,nt,out,workspace,bytes,stream);
    if (!is_fp4(nt, mode)) {
        Arena a{static_cast<uint8_t *>(workspace)};
        const Short s = short_layout(a, nt);
        const int rows = nt * K;
        validate<<<(int64_t(rows) + 255) / 256, 256, 0, cs>>>(ids, raw, rows, slots, false);
        setup_short<<<(int64_t(rows) + 255) / 256, 256, 0, cs>>>(ids, raw, rows, s);
        native::quantize_q8_1_rows(x, nt, E, s.xq, cs);
        const auto layout = native::native_expert_layout(40, 40, E, FF);
        native::native_expert_grouped(layout, s.ptr, s.start, s.count, s.dst, s.tok, rows, rows,
                                      s.xq, s.scratch, out, cs);
    } else {
        for (int first = 0; first < nt; first += WINDOW) {
            const int count = std::min(WINDOW, nt - first), rows = count * K;
            Arena a{static_cast<uint8_t *>(workspace)};
            const Long s = long_layout(a, count);
            const float * xx = x + int64_t(first) * E;
            const int32_t * ii = ids + int64_t(first) * K;
            validate<<<(rows + 255) / 256, 256, 0, cs>>>(ii, raw, rows, slots, true);
            group_bank<<<EXPERTS, 32, count * sizeof(uint32_t), cs>>>(ii, count, s.tok, s.scatter, s.bounds);
            fp4::plan(s.bounds, EXPERTS, s.tiles, cs);
            fp4::quantize_tokens(xx, count, E, E, s.ax, s.sx, cs);
            for (int g = 0; g < EXPERTS; g += fp4::kMaxExperts) {
                fp4::Product p;
                for (int j = 0; j < fp4::kMaxExperts; ++j) p.blob[j] = raw + uint64_t(g + j) * BLOB;
                p.n = fp4::kMaxExperts; p.w_rows = 2 * FF; p.w_cols = E; p.swiglu = true;
                p.ids = s.tok; p.bounds = s.bounds + g; p.tokens = count;
                p.a4 = s.ax; p.gs = s.sx; p.tail_off = TAIL; p.dst = s.h; p.ld_dst = FF; p.max_rows = count;
                fp4::run(p, cs);
            }
            fp4::quantize(s.h, nullptr, s.bounds, s.tiles, EXPERTS, s.ah, s.sh, FF, FF, rows, cs);
            for (int g = 0; g < EXPERTS; g += fp4::kMaxExperts) {
                fp4::Product p;
                for (int j = 0; j < fp4::kMaxExperts; ++j) p.blob[j] = raw + uint64_t(g + j) * BLOB;
                p.n = fp4::kMaxExperts; p.w_rows = E; p.w_cols = FF; p.w_off = DOWN;
                p.bounds = s.bounds + g; p.tile_off = s.tiles + g;
                p.a4 = s.ah; p.gs = s.sh; p.dst = s.dm; p.ld_dst = E; p.max_rows = count;
                fp4::run(p, cs);
            }
            scatter_rows<<<(int64_t(rows) * E + 255) / 256, 256, 0, cs>>>(s.dm, s.scatter,
                out + int64_t(first) * K * E, int64_t(rows) * E);
        }
    }
    return int(cudaGetLastError());
}

extern "C" int qfn_basalt_nvfp4_calibrated_abi(void) { return 1; }
extern "C" int qfn_basalt_nvfp4_run_calibrated(const float * x,const int32_t * ids,
    const uint8_t * raw,const float * aux,uint64_t stride,uint32_t slots,int nt,int top_k,
    float * out,void * workspace,uint64_t bytes,void * stream) {
    const auto need=qfn_basalt_nvfp4_workspace_bytes(nt,top_k,slots,QFN_NVFP4_CALIBRATED_A4);
    if(!need||!x||!ids||!raw||!aux||!out||!workspace||bytes<need||stride!=BLOB||
       (reinterpret_cast<uintptr_t>(workspace)&255)||(reinterpret_cast<uintptr_t>(raw)&15)||
       (reinterpret_cast<uintptr_t>(x)&15)||(reinterpret_cast<uintptr_t>(out)&15)||
       (reinterpret_cast<uintptr_t>(ids)&3)||(reinterpret_cast<uintptr_t>(aux)&3))return int(cudaErrorInvalidValue);
    int dev=0;const auto err=cudaGetDevice(&dev);if(err!=cudaSuccess)return int(err);
    if(dev<0||dev>=64||!prepared[dev].load(std::memory_order_acquire))return int(cudaErrorNotReady);
    return qfn_calibrated::run(x,ids,raw,aux,nt,out,workspace,stream);
}

extern "C" int qfn_basalt_nvfp4_calibrated_decode_abi(void) { return 1; }
extern "C" int qfn_basalt_nvfp4_run_calibrated_decode(const float * x,const int32_t * ids,
    const int32_t * logical,const uint8_t * raw,const float * aux,uint64_t stride,uint32_t slots,
    int nt,int top_k,float * out,void * workspace,uint64_t bytes,void * stream) {
    const auto need=qfn_basalt_nvfp4_workspace_bytes(nt,top_k,slots,QFN_NVFP4_CALIBRATED_DECODE);
    if(!need||!x||!ids||!logical||!raw||!aux||!out||!workspace||bytes<need||stride!=BLOB||
       (reinterpret_cast<uintptr_t>(workspace)&255)||(reinterpret_cast<uintptr_t>(raw)&15)||
       (reinterpret_cast<uintptr_t>(x)&15)||(reinterpret_cast<uintptr_t>(out)&15)||
       (reinterpret_cast<uintptr_t>(ids)&3)||(reinterpret_cast<uintptr_t>(logical)&3)||
       (reinterpret_cast<uintptr_t>(aux)&3))return int(cudaErrorInvalidValue);
    int dev=0;const auto err=cudaGetDevice(&dev);if(err!=cudaSuccess)return int(err);
    if(dev<0||dev>=64||!prepared[dev].load(std::memory_order_acquire))return int(cudaErrorNotReady);
    return qfn_calibrated::run_decode(x,ids,logical,raw,slots,aux,nt,out,workspace,stream);
}
