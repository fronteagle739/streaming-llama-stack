#pragma once

#include "llama-dsv41-wire.h"
#include "llama-dsv41-delta.h"
#include "llama-dsv41-dspark-capture.h"
#include "llama-dsv41-owner-digest.h"
#include "ggml-cpp.h"

#include <array>
#include <mutex>
#include <string>
#include <vector>

// Canonical native V4.1 storage. Each physical compressor/index cache exists
// once, irrespective of the number of reader layers. Values remain F32 so
// native BF16/QAT boundaries do not acquire an additional F16 conversion.
// Optional BF16 physical compressed/index storage retains the existing QAT
// output exactly; raw rings and unrounded pool tails always remain F32.
// Allocation requires an explicit byte ceiling from resource admission.
class llama_dsv41_cache_storage {
public:
    // grow_step 0: one fixed full-context buffer (unchanged). Otherwise the
    // compressed/index owners are CUDA VMM ranges mapped in grow_step-token
    // steps (LLAMADECODE growable context cache; plain CUDA VMM, no D3D12).
    llama_dsv41_cache_storage(const std::array<int32_t, 40> & kv_owners, uint32_t context,
                             ggml_backend_buffer_type_t buft, uint64_t admitted_bytes,bool compressed_bf16=false,bool capture_dspark=false,
                             uint32_t grow_step=0);
    ~llama_dsv41_cache_storage();

    static uint64_t required_bytes(uint32_t context,bool compressed_bf16=false,bool capture_dspark=false);
    uint64_t allocated_bytes() const;
    uint32_t capacity() const { return context; }
    bool compact_compressed() const { return compressed_bf16; }
    // Growable cache (grow_step != 0): mapped positions, mapped physical bytes,
    // and the physical bytes of the next step (0 at full context).
    uint32_t growth_step() const { return grow_step; }
    uint32_t mapped_tokens() const;
    uint64_t next_step_bytes() const;
    // Map the owners to cover `tokens` positions (rounded up to the step).
    // No-op for a fixed cache or when already covered; logs each grow event.
    bool ensure_tokens(uint32_t tokens, const char * phase, bool * grew, std::string & error);
    // A carry loan may only coexist with this bounded physical cache mapping.
    // Clear after the oversized hidden bank is freed, before decode/append.
    void prefill_ceiling(uint32_t tokens,uint64_t physical_bytes=0);
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

    // The frontier's owner SHA-256s (the wire owner_sha256 values). With the
    // default incremental verify (llama-dsv41-owner-digest.h) only rows added
    // since the last verified digest are read; if `expected` is given and the
    // incremental result differs, the owners are re-hashed in full.
    // LLAMADECODE_OWNER_VERIFY=full always re-hashes every owner.
    bool owner_digests(const llama_dsv41::wire_frontier & frontier, std::vector<std::string> & out,
        const std::vector<std::string> * expected, std::string & error);
    // Drop the incremental digest memo (any path that may rewrite old rows).
    void forget_owner_digests();
    // Owner-digest read staging admitted by the job's host plan (sweep
    // admission, owner_digest_staging). Pinned staging is allocated only up to
    // this; before admission the digests use at most 2 MiB of pageable slots.
    void admit_owner_digest_staging(uint64_t bytes);

private:
    llama_dsv41::owner_digest_memo digest_memo;
    // Owner-digest read slots: pinned host memory of the cache's device
    // (owner_digest_memo::staging_bytes(), 32 MiB), allocated on first use.
    ggml_backend_buffer_ptr digest_staging;
    bool digest_staging_tried=false;
    uint64_t digest_staging_admitted=0;
    // Owner digests for verify/inspect under execution_mutex (synchronized).
    std::vector<std::string> owner_digests_locked(const llama_dsv41::wire_plan & plan,
        const std::vector<std::string> * expected, const char * phase, bool require_capture);
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
    // Growable owners: one VA reservation, eight granularity-aligned ranges.
    const uint32_t grow_step;
    uint32_t grow_tokens=0;
    uint64_t grow_granularity=0,grow_reserved=0,admitted=0;
    int grow_device=0;
    void * grow_base=nullptr;
    std::array<ggml_tensor *, 8> grow_tensor {};
    std::array<uint64_t, 8> grow_offset {}, grow_full {}, grow_mapped {};
    struct grow_mapping { uint64_t offset, bytes, handle; };
    std::vector<grow_mapping> grow_maps;
    ggml_context_ptr grow_tensors;
    ggml_backend_buffer_ptr grow_buffer;
    bool ensure_tokens_locked(uint32_t tokens, const char * phase, bool * grew, std::string & error);
    uint32_t transient_ceiling=0;
    void clear_grown_locked();

    static int source_index(uint32_t source_layer);
    // limit > 1 MiB (owner digests): BF16 owners are read packed into dst and
    // widened to F32 in place (same bytes as the bounded scratch path).
    bool read_native_span_locked(const llama_dsv41::state_span &, uint64_t,
        void *, size_t, std::string &, bool require_capture, size_t limit = 1024*1024) const;
    bool write_native_span_locked(const llama_dsv41::state_span &, uint64_t,
        const void *, size_t, std::string &);
    bool apply_native_update(const llama_dsv41::wire_frontier * base,
        const llama_dsv41::wire_frontier & target,
        const std::vector<std::string> & base_hashes,
        const std::vector<std::string> & target_hashes,
        const llama_dsv41::wire_read_span & read, std::string & error, bool rewind=false, bool replace=false);
};
