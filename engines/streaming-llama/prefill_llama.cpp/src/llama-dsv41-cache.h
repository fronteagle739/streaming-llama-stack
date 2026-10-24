#pragma once

#include "llama-dsv41-wire.h"
#include "llama-dsv41-delta.h"
#include "llama-dsv41-dspark-capture.h"
#include "ggml-cpp.h"

#include <array>
#include <mutex>

// Canonical native V4.1 storage. Each physical compressor/index cache exists
// once, irrespective of the number of reader layers. Values remain F32 so
// native BF16/QAT boundaries do not acquire an additional F16 conversion.
// Optional BF16 physical compressed/index storage retains the existing QAT
// output exactly; raw rings and unrounded pool tails always remain F32.
// Allocation requires an explicit byte ceiling from resource admission.
class llama_dsv41_cache_storage {
public:
    llama_dsv41_cache_storage(const std::array<int32_t, 40> & kv_owners, uint32_t context,
                             ggml_backend_buffer_type_t buft, uint64_t admitted_bytes,bool compressed_bf16=false,bool capture_dspark=false);

    static uint64_t required_bytes(uint32_t context,bool compressed_bf16=false,bool capture_dspark=false);
    uint64_t allocated_bytes() const;
    uint32_t capacity() const { return context; }
    bool compact_compressed() const { return compressed_bf16; }
    void clear();
    bool captures_dspark() const { return dspark_ring!=nullptr; }
    bool dspark_ready(uint32_t position) const;

    ggml_tensor * raw(uint32_t layer) const;
    ggml_tensor * compressed(uint32_t reader_layer) const;
    ggml_tensor * index(uint32_t reader_layer) const;
    ggml_tensor * pool_kv(uint32_t source_layer) const;
    ggml_tensor * pool_score(uint32_t source_layer) const;

    // The caller must synchronize and certify its completed graph frontier
    // before exporting. This method copies only a bounded requested slice;
    // write_native_payload() limits requests to 1 MiB. It never stages a
    // table- or context-sized host copy.
    bool read_native_span(const llama_dsv41::state_span & span,
                          uint64_t offset, void * dst, size_t bytes,
                          std::string & error) const;

    // Import verified canonical state into this existing allocation. Hashes
    // correspond exactly to make_wire_plan().spans in order. Full restore is
    // allowed only before the cache has been bound to a compute backend.
    // Delta admission verifies every base owner before the first write and
    // every target owner before publication. Any failure poisons the cache.
    // Callers separately own/authenticate token IDs, logits and session identity.
    bool restore_native_frontier(const llama_dsv41::wire_frontier & target,
        const std::vector<std::string> & target_hashes,
        const llama_dsv41::wire_read_span & read, std::string & error);
    // Explicit replacement after the caller retires the previous executor.
    // Synchronizes, invalidates and replaces every owner; no stale base reuse.
    bool replace_native_frontier(const llama_dsv41::wire_frontier & target,
        const std::vector<std::string> & target_hashes,
        const llama_dsv41::wire_read_span & read, std::string & error);
    bool apply_native_delta(const llama_dsv41::wire_frontier & base,
        const llama_dsv41::wire_frontier & target,
        const std::vector<std::string> & base_hashes,
        const std::vector<std::string> & target_hashes,
        const llama_dsv41::wire_read_span & read, std::string & error);
    // Restore a previously captured prefix's mutable rings/tails only. The
    // append-only compressed/index prefix remains resident and is re-hashed.
    bool rewind_native_frontier(const llama_dsv41::wire_frontier & base,
        const llama_dsv41::wire_frontier & target,
        const std::vector<std::string> & base_hashes,
        const std::vector<std::string> & target_hashes,
        const llama_dsv41::wire_read_span & read, std::string & error);

private:
    friend class llama_dsv41_tile_executor;
    // Shared by ALL executors and bounded native readbacks. Once bound, the
    // compute backend must outlive this cache. Clear synchronizes that backend
    // before invalidating a failed frontier or resetting physical state.
    mutable std::mutex execution_mutex;
    bool execution_poisoned=false;
    ggml_backend_t execution_backend=nullptr;
    const uint32_t context;
    const bool compressed_bf16;
    std::array<int32_t, 40> owners {};
    std::array<ggml_tensor *, 40> raw_ring {};
    std::array<ggml_tensor *, 4> compressed_cache {};
    std::array<ggml_tensor *, 4> index_cache {};
    std::array<ggml_tensor *, 3> carry_kv {};
    std::array<ggml_tensor *, 3> carry_score {};
    ggml_tensor * dspark_ring=nullptr;
    llama_dsv41_dspark_capture::completion dspark_completion;
    ggml_context_ptr tensors;
    ggml_backend_buffer_ptr buffer;

    static int source_index(uint32_t source_layer);
    bool read_native_span_locked(const llama_dsv41::state_span &, uint64_t,
        void *, size_t, std::string &, bool require_capture) const;
    bool write_native_span_locked(const llama_dsv41::state_span &, uint64_t,
        const void *, size_t, std::string &);
    bool apply_native_update(const llama_dsv41::wire_frontier * base,
        const llama_dsv41::wire_frontier & target,
        const std::vector<std::string> & base_hashes,
        const std::vector<std::string> & target_hashes,
        const llama_dsv41::wire_read_span & read, std::string & error, bool rewind=false, bool replace=false);
};
