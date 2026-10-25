#include "llama-dsv41-cache.h"
#include "llama-dsv41-ctx-cache-plan.h"

#include "ggml-alloc.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>
#include <cstring>

namespace {
constexpr std::array<uint32_t, 4> sources {{2, 8, 14, 20}};

uint32_t checked_context(uint32_t context) {
    if (!context || context > 1048576) {
        throw std::runtime_error("V4.1 cache context must be in 1..1048576");
    }
    return context;
}
}

uint64_t llama_dsv41_cache_storage::required_bytes(uint32_t context,bool compressed_bf16,bool capture_dspark) {
    checked_context(context);
    const uint64_t ratio2 = (uint64_t(context) + 1) / 2;
    const uint64_t compressed_rows = 3*ratio2 + context;
    // 40 canonical raw rings, four KV/index owners and three pool2 tails.
    // Every tensor is a multiple of 512 bytes, so backend padding is normally
    // zero. The actual buffer size is checked against admission as well.
    return 40ull*128*512*4 + compressed_rows*(512+128)*(compressed_bf16?2:4) + 3ull*2*512*4 + (capture_dspark?llama_dsv41_dspark_capture::bytes:0);
}

int llama_dsv41_cache_storage::source_index(uint32_t layer) {
    for (size_t i = 0; i < sources.size(); ++i) {
        if (sources[i] == layer) { return (int) i; }
    }
    return -1;
}

llama_dsv41_cache_storage::llama_dsv41_cache_storage(
        const std::array<int32_t, 40> & kv_owners, uint32_t context,
        ggml_backend_buffer_type_t buft, uint64_t admitted_bytes,bool compressed_bf16,bool capture_dspark,uint32_t grow_step) :
        context(checked_context(context)),compressed_bf16(compressed_bf16),grow_step(grow_step),admitted(admitted_bytes) {
    if (!buft) {
        throw std::runtime_error("canonical V4.1 cache requires an explicit backend");
    }
    if (required_bytes(context,compressed_bf16,capture_dspark) > admitted_bytes) {
        throw std::runtime_error("canonical V4.1 cache exceeds admitted memory budget");
    }
    for (uint32_t il = 0; il < 40; ++il) {
        owners[il] = kv_owners[il];
        if (il < 2) {
            if (owners[il] != -1) {
                throw std::runtime_error("V4.1 raw-only layer has a compressed owner");
            }
            continue;
        }
        const int index = source_index((uint32_t) owners[il]);
        const int expected = il < 8 ? 0 : il < 14 ? 1 : il < 20 ? 2 : 3;
        if (index != expected) {
            throw std::runtime_error("V4.1 cache ownership differs from the native Flash layout");
        }
    }

    ggml_init_params params {128*ggml_tensor_overhead(), nullptr, true};
    tensors.reset(ggml_init(params));
    if (!tensors) { throw std::runtime_error("cannot create V4.1 cache tensor context"); }
    if (grow_step) {
#ifdef GGML_USE_CUDA
        const auto why = llama_dsv41_ctx_cache::refusal(context, compressed_bf16, true, grow_step, 2ull << 20, admitted_bytes);
        if (!why.empty()) { throw std::runtime_error(why); }
        grow_tensors.reset(ggml_init(params));
        if (!grow_tensors) { throw std::runtime_error("cannot create V4.1 growable cache tensor context"); }
#else
        throw std::runtime_error("growable V4.1 context cache requires CUDA");
#endif
    }
    ggml_context * owner_ctx = grow_step ? grow_tensors.get() : tensors.get();
    for (uint32_t il = 0; il < 40; ++il) {
        raw_ring[il] = ggml_new_tensor_2d(tensors.get(), GGML_TYPE_F32, 512, 128);
        ggml_format_name(raw_ring[il], "dsv41_raw_l%u", il);
    }
    for (size_t i = 0; i < sources.size(); ++i) {
        const uint32_t rows = i < 3 ? (context + 1)/2 : context;
        compressed_cache[i] = ggml_new_tensor_2d(owner_ctx, compressed_bf16?GGML_TYPE_BF16:GGML_TYPE_F32, 512, rows);
        index_cache[i] = ggml_new_tensor_2d(owner_ctx, compressed_bf16?GGML_TYPE_BF16:GGML_TYPE_F32, 128, rows);
        ggml_format_name(compressed_cache[i], "dsv41_compressed_l%u", sources[i]);
        ggml_format_name(index_cache[i], "dsv41_index_l%u", sources[i]);
        if (i < 3) {
            carry_kv[i] = ggml_new_tensor_1d(tensors.get(), GGML_TYPE_F32, 512);
            carry_score[i] = ggml_new_tensor_1d(tensors.get(), GGML_TYPE_F32, 512);
            ggml_format_name(carry_kv[i], "dsv41_pool_kv_l%u", sources[i]);
            ggml_format_name(carry_score[i], "dsv41_pool_score_l%u", sources[i]);
        }
    }
    if(capture_dspark){
        dspark_ring=ggml_new_tensor_3d(tensors.get(),GGML_TYPE_F32,5120,3,128);
        ggml_set_name(dspark_ring,"dsv41_dspark_physical_ring");
    }
    // Query actual backend alignment before allocating the buffer. This is a
    // strict ceiling, not an allocation followed by an oversized-load check.
    const size_t actual = ggml_backend_alloc_ctx_tensors_from_buft_size(tensors.get(), buft);
    if (actual > admitted_bytes) { throw std::runtime_error("aligned V4.1 cache exceeds admitted memory budget"); }
    buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(tensors.get(), buft));
    if (!buffer) { throw std::runtime_error("cannot allocate admitted V4.1 cache buffer"); }
#ifdef GGML_USE_CUDA
    if (grow_step) {
        // One VA reservation; each owner tensor on its own granularity-aligned range.
        auto cu = [](CUresult r, const char * what) { if (r != CUDA_SUCCESS) throw std::runtime_error(std::string("growable cache: ") + what + " failed"); };
        if (cudaGetDevice(&grow_device) != cudaSuccess) { throw std::runtime_error("growable cache: no current CUDA device"); }
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = grow_device;
        size_t g = 0;
        cu(cuMemGetAllocationGranularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "cuMemGetAllocationGranularity");
        grow_granularity = g;
        const auto why = llama_dsv41_ctx_cache::refusal(context, compressed_bf16, true, grow_step, grow_granularity, admitted_bytes);
        if (!why.empty()) { throw std::runtime_error(why); }
        for (size_t i = 0; i < 4; ++i) { grow_tensor[2*i] = compressed_cache[i]; grow_tensor[2*i+1] = index_cache[i]; }
        for (size_t k = 0; k < grow_tensor.size(); ++k) {
            grow_offset[k] = grow_reserved; grow_full[k] = llama_dsv41_ctx_cache::round_up(ggml_nbytes(grow_tensor[k]), g);
            grow_reserved += grow_full[k];
        }
        CUdeviceptr va = 0;
        cu(cuMemAddressReserve(&va, grow_reserved, g, 0, 0), "cuMemAddressReserve");
        grow_base = reinterpret_cast<void *>(va);
        grow_buffer.reset(ggml_backend_cuda_buffer_from_ptr(grow_device, grow_base, grow_reserved));
        if (!grow_buffer) { throw std::runtime_error("growable cache: CUDA buffer wrapper failed"); }
        for (size_t k = 0; k < grow_tensor.size(); ++k) {
            if (ggml_backend_tensor_alloc(grow_buffer.get(), grow_tensor[k], static_cast<char *>(grow_base) + grow_offset[k]) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("growable cache: tensor binding failed");
            }
        }
    }
#endif
    clear();
    if (grow_step) {
        std::string error;
        if (!ensure_tokens(1, "open", nullptr, error)) { throw std::runtime_error(error); }
    }
    std::fprintf(stderr, "LLAMADECODE_CTX_CACHE precision=%s context=%u bytes=%llu allocated_bytes=%llu growable=%d step=%u granularity=%llu mapped_tokens=%u reachable_tokens=%u admitted_bytes=%llu\n",
        compressed_bf16 ? "bf16" : "f32", context, (unsigned long long) required_bytes(context, compressed_bf16, capture_dspark),
        (unsigned long long) allocated_bytes(), grow_step ? 1 : 0, grow_step, (unsigned long long) grow_granularity, mapped_tokens(),
        grow_step ? llama_dsv41_ctx_cache::reachable_tokens(context, grow_step, compressed_bf16, grow_granularity, admitted_bytes) : context,
        (unsigned long long) admitted_bytes);
    std::fflush(stderr);
}

llama_dsv41_cache_storage::~llama_dsv41_cache_storage() {
#ifdef GGML_USE_CUDA
    if (grow_base) {
        if (execution_backend) { ggml_backend_synchronize(execution_backend); }
        cudaDeviceSynchronize();
        grow_buffer.reset();
        const CUdeviceptr va = reinterpret_cast<CUdeviceptr>(grow_base);
        for (const auto & m : grow_maps) {
            cuMemUnmap(va + m.offset, m.bytes);
            cuMemRelease(static_cast<CUmemGenericAllocationHandle>(m.handle));
        }
        cuMemAddressFree(va, grow_reserved);
    }
#endif
}

uint64_t llama_dsv41_cache_storage::allocated_bytes() const {
    uint64_t mapped = 0;
    for (const auto & m : grow_maps) { mapped += m.bytes; }
    return ggml_backend_buffer_get_size(buffer.get()) + mapped;
}

uint32_t llama_dsv41_cache_storage::mapped_tokens() const { return grow_step ? grow_tokens : context; }

uint64_t llama_dsv41_cache_storage::next_step_bytes() const {
    return grow_step ? llama_dsv41_ctx_cache::next_step_bytes(context, grow_tokens, grow_step, compressed_bf16, grow_granularity) : 0;
}

void llama_dsv41_cache_storage::clear_grown_locked() {
#ifdef GGML_USE_CUDA
    for (size_t k = 0; k < grow_tensor.size(); ++k) {
        if (grow_mapped[k] && cudaMemset(static_cast<char *>(grow_base) + grow_offset[k], 0, grow_mapped[k]) != cudaSuccess) {
            throw std::runtime_error("growable cache: clear failed");
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) { throw std::runtime_error("growable cache: clear synchronize failed"); }
#endif
}

bool llama_dsv41_cache_storage::ensure_tokens(uint32_t tokens, const char * phase, bool * grew, std::string & error) {
    std::lock_guard<std::mutex>lock(execution_mutex);
    return ensure_tokens_locked(tokens, phase, grew, error);
}

bool llama_dsv41_cache_storage::ensure_tokens_locked(uint32_t tokens, const char * phase, bool * grew, std::string & error) {
    if (grew) { *grew = false; }
    error.clear();
    if (!grow_step) { return true; }
    if (tokens > context) { error = "growable cache: " + std::to_string(tokens) + " positions exceed the context " + std::to_string(context); return false; }
    const uint32_t target = llama_dsv41_ctx_cache::round_tokens(context, grow_step, tokens);
    if (target <= grow_tokens) { return true; }
#ifdef GGML_USE_CUDA
    const uint64_t before = allocated_bytes();
    const uint64_t want_total = ggml_backend_buffer_get_size(buffer.get()) +
        llama_dsv41_ctx_cache::mapped_bytes(context, target, compressed_bf16, grow_granularity);
    if (want_total > admitted) {
        error = "growable cache: mapping " + std::to_string(target) + " positions needs " + std::to_string(want_total) +
            " bytes, over the admitted " + std::to_string(admitted);
        return false;
    }
    if (execution_backend) { ggml_backend_synchronize(execution_backend); }
    if (cudaSetDevice(grow_device) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) { error = "growable cache: device synchronize failed"; return false; }
    const CUdeviceptr va = reinterpret_cast<CUdeviceptr>(grow_base);
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = grow_device;
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE; access.location.id = grow_device; access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    for (size_t k = 0; k < grow_tensor.size(); ++k) {
        const uint32_t owner = uint32_t(k / 2);
        const uint64_t width = k % 2 ? 128 : 512;
        const uint64_t rows = std::min(llama_dsv41_ctx_cache::owner_rows(owner, target), uint64_t(grow_tensor[k]->ne[1]));
        const uint64_t want = std::min(llama_dsv41_ctx_cache::round_up(rows * width * ggml_type_size(grow_tensor[k]->type), grow_granularity), grow_full[k]);
        if (want <= grow_mapped[k]) { continue; }
        const uint64_t offset = grow_offset[k] + grow_mapped[k], bytes = want - grow_mapped[k];
        CUmemGenericAllocationHandle handle = 0;
        if (cuMemCreate(&handle, bytes, &prop, 0) != CUDA_SUCCESS) {
            error = "growable cache: out of device memory mapping " + std::to_string(bytes) + " bytes at " + std::to_string(target) + " positions";
            return false;
        }
        if (cuMemMap(va + offset, bytes, 0, handle, 0) != CUDA_SUCCESS) { cuMemRelease(handle); error = "growable cache: cuMemMap failed"; return false; }
        if (cuMemSetAccess(va + offset, bytes, &access, 1) != CUDA_SUCCESS) {
            cuMemUnmap(va + offset, bytes); cuMemRelease(handle); error = "growable cache: cuMemSetAccess failed"; return false;
        }
        grow_maps.push_back({offset, bytes, uint64_t(handle)});
        // Same zeroed contents the fixed buffer's clear gives these rows.
        if (cudaMemset(static_cast<char *>(grow_base) + offset, 0, bytes) != cudaSuccess) { error = "growable cache: zero fill failed"; return false; }
        grow_mapped[k] = want;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) { error = "growable cache: map synchronize failed"; return false; }
    const uint32_t from = grow_tokens;
    grow_tokens = target;
    if (grew) { *grew = true; }
    std::fprintf(stderr, "LLAMADECODE_CTX_CACHE_GROW phase=%s need_tokens=%u mapped_tokens=%u->%u step_bytes=%llu allocated_bytes=%llu next_step_bytes=%llu context=%u precision=%s\n",
        phase ? phase : "?", tokens, from, target, (unsigned long long) (allocated_bytes() - before), (unsigned long long) allocated_bytes(),
        (unsigned long long) next_step_bytes(), context, compressed_bf16 ? "bf16" : "f32");
    std::fflush(stderr);
    return true;
#else
    (void) phase;
    error = "growable cache requires CUDA";
    return false;
#endif
}

void llama_dsv41_cache_storage::clear() {
    std::lock_guard<std::mutex>lock(execution_mutex);
    if(execution_backend)ggml_backend_synchronize(execution_backend);
    ggml_backend_buffer_clear(buffer.get(),0);
    if(grow_step)clear_grown_locked();
    if(execution_backend)ggml_backend_synchronize(execution_backend);
    dspark_completion.clear();
    digest_memo.invalidate();
    execution_poisoned=false;
}

bool llama_dsv41_cache_storage::dspark_ready(uint32_t position) const {
    std::lock_guard<std::mutex>lock(execution_mutex);
    return !execution_poisoned&&dspark_ring&&dspark_completion.ready(position);
}

ggml_tensor * llama_dsv41_cache_storage::raw(uint32_t layer) const {
    if (layer >= raw_ring.size()) { throw std::out_of_range("V4.1 raw layer"); }
    return raw_ring[layer];
}

ggml_tensor * llama_dsv41_cache_storage::compressed(uint32_t layer) const {
    if (layer >= owners.size() || owners[layer] < 0) { throw std::out_of_range("V4.1 compressed reader"); }
    return compressed_cache[(size_t) source_index((uint32_t) owners[layer])];
}

ggml_tensor * llama_dsv41_cache_storage::index(uint32_t layer) const {
    if (layer >= owners.size() || owners[layer] < 0) { throw std::out_of_range("V4.1 index reader"); }
    return index_cache[(size_t) source_index((uint32_t) owners[layer])];
}

ggml_tensor * llama_dsv41_cache_storage::pool_kv(uint32_t layer) const {
    const int i = source_index(layer);
    if (i < 0 || i >= 3) { throw std::out_of_range("V4.1 pool2 owner"); }
    return carry_kv[(size_t) i];
}

ggml_tensor * llama_dsv41_cache_storage::pool_score(uint32_t layer) const {
    const int i = source_index(layer);
    if (i < 0 || i >= 3) { throw std::out_of_range("V4.1 pool2 owner"); }
    return carry_score[(size_t) i];
}

bool llama_dsv41_cache_storage::read_native_span(
        const llama_dsv41::state_span & span, uint64_t offset, void * dst,
        size_t bytes, std::string & error) const {
    std::lock_guard<std::mutex>lock(execution_mutex);
    if(execution_poisoned){error="V4.1 cache requires clear after an incomplete computation";return false;}
    return read_native_span_locked(span,offset,dst,bytes,error,true);
}

bool llama_dsv41_cache_storage::read_native_span_locked(
        const llama_dsv41::state_span & span, uint64_t offset, void * dst,
        size_t bytes, std::string & error, bool require_capture, size_t limit) const {
    using llama_dsv41::state_kind;
    ggml_tensor * tensor = nullptr;
    switch (span.kind) {
        case state_kind::raw_window:
            if (span.index < raw_ring.size()) { tensor = raw_ring[span.index]; }
            break;
        case state_kind::compressed:
            if (span.index < compressed_cache.size()) { tensor = compressed_cache[span.index]; }
            break;
        case state_kind::index:
            if (span.index < index_cache.size()) { tensor = index_cache[span.index]; }
            break;
        case state_kind::pool_kv:
            if (span.index < carry_kv.size()) { tensor = carry_kv[span.index]; }
            break;
        case state_kind::pool_score:
            if (span.index < carry_score.size()) { tensor = carry_score[span.index]; }
            break;
        case state_kind::dspark_hidden:
            if(span.index||!dspark_ring||(require_capture&&(!dspark_completion.ready(dspark_completion.ends[0])||
               span.bytes!=uint64_t(std::min(dspark_completion.ends[0],128u))*3*5120*4))){
                error="DSpark hidden frontier is incomplete or does not match its canonical span";return false;
            }
            tensor=dspark_ring;break;
    }
    if (!tensor || (!dst && bytes) || bytes > limit ||
        span.bytes > uint64_t(ggml_nelements(tensor))*4 || offset > span.bytes || bytes > span.bytes-offset) {
        error = "invalid or unbounded canonical V4.1 cache export span";
        return false;
    }
    if (bytes) {
        if(tensor->type==GGML_TYPE_BF16&&limit>1024*1024&&offset%4==0&&bytes%4==0){
            // Owner digests: packed BF16 into the front of dst, widened in place
            // from the back (element i's F32 bytes never overlap unread BF16).
            const size_t first=size_t(offset/4),count=bytes/4;auto*p=static_cast<unsigned char*>(dst);
            ggml_backend_tensor_get(tensor,p,first*2,count*2);
            for(size_t i=count;i-->0;){uint16_t b;std::memcpy(&b,p+2*i,2);const uint32_t v=uint32_t(b)<<16;std::memcpy(p+4*i,&v,4);}
        }else if(tensor->type==GGML_TYPE_BF16){
            // Canonical wire remains F32, including arbitrary byte slices.
            // At most 262145 elements: <1.51MiB combined bounded host scratch.
            const size_t first=size_t(offset/4),skip=size_t(offset%4),count=(skip+bytes+3)/4;
            std::vector<ggml_bf16_t> packed(count);std::vector<uint32_t> canonical(count);
            ggml_backend_tensor_get(tensor,packed.data(),first*2,count*2);
            for(size_t i=0;i<count;++i)canonical[i]=uint32_t(packed[i].bits)<<16;
            std::memcpy(dst,reinterpret_cast<const unsigned char*>(canonical.data())+skip,bytes);
        }else ggml_backend_tensor_get(tensor, dst, (size_t) offset, bytes);
    }
    error.clear();
    return true;
}
