#include "llama-dsv41-cache.h"

#include "ggml-alloc.h"

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
        ggml_backend_buffer_type_t buft, uint64_t admitted_bytes,bool compressed_bf16,bool capture_dspark) : context(checked_context(context)),compressed_bf16(compressed_bf16) {
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
    for (uint32_t il = 0; il < 40; ++il) {
        raw_ring[il] = ggml_new_tensor_2d(tensors.get(), GGML_TYPE_F32, 512, 128);
        ggml_format_name(raw_ring[il], "dsv41_raw_l%u", il);
    }
    for (size_t i = 0; i < sources.size(); ++i) {
        const uint32_t rows = i < 3 ? (context + 1)/2 : context;
        compressed_cache[i] = ggml_new_tensor_2d(tensors.get(), compressed_bf16?GGML_TYPE_BF16:GGML_TYPE_F32, 512, rows);
        index_cache[i] = ggml_new_tensor_2d(tensors.get(), compressed_bf16?GGML_TYPE_BF16:GGML_TYPE_F32, 128, rows);
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
    clear();
}

uint64_t llama_dsv41_cache_storage::allocated_bytes() const {
    return ggml_backend_buffer_get_size(buffer.get());
}

void llama_dsv41_cache_storage::clear() {
    std::lock_guard<std::mutex>lock(execution_mutex);
    if(execution_backend)ggml_backend_synchronize(execution_backend);
    ggml_backend_buffer_clear(buffer.get(),0);
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
        size_t bytes, std::string & error, bool require_capture) const {
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
    if (!tensor || (!dst && bytes) || bytes > 1024*1024 ||
        span.bytes > uint64_t(ggml_nelements(tensor))*4 || offset > span.bytes || bytes > span.bytes-offset) {
        error = "invalid or unbounded canonical V4.1 cache export span";
        return false;
    }
    if (bytes) {
        if(tensor->type==GGML_TYPE_BF16){
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
