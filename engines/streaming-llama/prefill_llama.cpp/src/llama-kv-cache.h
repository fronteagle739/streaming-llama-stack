#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"

#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;

//
// llama_kv_cache
//

class llama_kv_cache : public llama_memory_i {
public:
    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }

        // check if indices are contiguous starting from head()
        bool is_contiguous() const {
            if (idxs.empty() || idxs[0].empty()) {
                return true;
            }
            if (idxs.size() > 1) {
                return false;
            }
            const uint32_t h = idxs[0][0];
            for (size_t i = 0; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != h + i) {
                    return false;
                }
            }
            return true;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    llama_kv_cache(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                         bool   sparse_mla,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse);

    // DeepSeek-V4: overload taking explicit hparams so DSV4 sub-caches can be
    // built with per-sub-cache modified hparams (K-only / indexer dims / ratios).
    llama_kv_cache(
            const llama_model & model,
          const llama_hparams & hparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                         bool   sparse_mla,   // dsv4-merge: fork's sparse indexer-K flag threads through
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse);

    ~llama_kv_cache() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    bool get_v_trans() const { return v_trans; }
    void set_v_trans(bool v) { v_trans = v; }

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache specific API
    //

    uint32_t get_size()     const;
    uint32_t get_n_stream() const;

    // Update k_stream/v_stream view data pointers from their parent tensors.
    // Call after alloc_graph which may re-point the parent tensor's data.
    void sync_view_pointers();

    bool get_has_shift() const;

    //
    // per-layer KV data pointer access (for layer-major KV RAM double-buffer)
    //

    // Get/set raw data pointers for a model layer's K/V tensors.
    // Uses map_layer_ids internally to resolve cache layer index.
    // Check if a model layer index has KV data in this cache.
    // Returns false for recurrent layers in hybrid models.
    bool   has_layer(int32_t il) const;
    void * get_k_data(int32_t il) const;
    void * get_v_data(int32_t il) const;
    void   set_k_data(int32_t il, void * data, ggml_backend_buffer_t buf = nullptr);
    void   set_v_data(int32_t il, void * data, ggml_backend_buffer_t buf = nullptr);
    size_t get_k_layer_bytes(int32_t il) const;
    size_t get_v_layer_bytes(int32_t il) const;
    size_t get_k_idx_layer_bytes(int32_t il) const;

    // Get K/V tensor pointers for direct access (used by KV-RAM download)
    ggml_tensor * get_k_tensor(int32_t il) const;
    ggml_tensor * get_v_tensor(int32_t il) const;
    ggml_tensor * get_k_idx_tensor(int32_t il) const;

    // DeepSeek-V4: raw storage-layer accessors used by llama_kv_cache_dsv4
    std::vector<uint32_t> get_layer_ids() const;
    ggml_tensor * get_k_storage(int32_t il) const;

    // Like set_k/v_data but also updates stream view tensor data pointers.
    // Required for KV-RAM export where we temporarily redirect tensors to CPU RAM.
    void set_k_data_all(int32_t il, void * data, ggml_backend_buffer_t buf);
    void set_v_data_all(int32_t il, void * data, ggml_backend_buffer_t buf);
    void set_k_idx_data_all(int32_t il, void * data, ggml_backend_buffer_t buf);
    void * get_k_idx_data(int32_t il) const;

    // Free the GPU (device) buffer, nulling tensor buffer/data pointers.
    // Returns the buffer type for later reallocation. Returns nullptr if
    // no device buffer is found. GPU must be synchronized before calling.
    ggml_backend_buffer_type_t free_device_buffer();

    // Reallocate the device buffer using the given buffer type, reassigning
    // tensor buffer/data pointers. Returns true on success.
    bool realloc_device_buffer(ggml_backend_buffer_type_t buft);

    //
    // graph_build API
    //

    uint32_t get_n_kv(const slot_info & sinfo) const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;

    // Phase 3.5-lite (iter 29): indexer K cache accessors for GLM_DSA sparse MLA.
    // Returns a view / store op for the per-layer indexer K cache. Callable only when
    // hparams.indexer_n_head > 0 AND layer.k_idx was allocated (otherwise returns nullptr).
    // Layout parallels get_k / cpy_k but operates on layer.k_idx (single-head, fp16).
    ggml_tensor * get_k_idx(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * cpy_k_idx(ggml_context * ctx, ggml_tensor * k_idx_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    bool update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch);

    // STREAMING CONTINUATION (2026-09-10): mark cells [0, n) as used by seq_id at positions 0..n-1
    // without data, so that a batch continuing a prefix the host/NVMe tier holds is placed by
    // find_slot at cell == position (the layer-major prefill's row layout). Cells already used
    // must already carry exactly that position and seq. Returns false if it cannot.
    bool reserve_prefix_cells(llama_seq_id seq_id, uint32_t n);

    // STREAMED EXPORT (2026-09-10): the byte layout state_write_data emits for stream strm (types and
    // row sizes per layer, in layer order), so the NVMe tier can stream the same bytes from its row
    // files. has_k_idx_state mirrors state_write_data's decision.
    struct export_layer_layout {
        uint32_t il = 0;
        int32_t  k_type = 0;
        uint64_t k_size_row = 0;
        bool     k_idx_present = false;
        int32_t  k_idx_type = 0;
        uint64_t k_idx_size_row = 0;
        bool     v_present = false;
        int32_t  v_type = 0;
        uint64_t v_size_row = 0;
    };
    bool export_layout(uint32_t strm, std::vector<export_layer_layout> & out, bool & has_k_idx_state) const;

    // True when seq_id's cells in its stream are exactly [0, n) with cell i at position i and no
    // cell of that seq beyond; strm_out receives the stream. This is the layout the layer-major
    // prefill leaves behind (cell == position), and what a streamed export assumes.
    bool seq_cells_identity(llama_seq_id seq_id, uint32_t n, uint32_t & strm_out) const;

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    // DSV4 / DeepSeek lightning-indexer QK Hadamard rotation (upstream PR 21038).
    // Returns nullptr unless attn_rot_k/attn_rot_v is enabled (arch-neutral: false for all
    // existing archs, so these are no-ops for them).
    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void fill_input_k_idxs(std::vector<int64_t> & dst, int64_t n_total, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void fill_input_v_idxs(std::vector<int64_t> & dst, int64_t n_total, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn, float soft_mask_alpha = 0.0f, float soft_mask_tau = 0.0f) const;
    bool fill_input_kq_mask_f16(std::vector<uint16_t> & dst, int64_t n_kv, const llama_ubatch * ubatch, bool causal_attn, float soft_mask_alpha = 0.0f, float soft_mask_tau = 0.0f) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

private:
    const llama_model & model;
    const llama_hparams & hparams;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        // Phase 3.5-lite (iter 29): indexer K cache for GLM_DSA sparse MLA.
        // Allocated only when hparams.indexer_n_head > 0; nullptr otherwise.
        // Shape: [indexer_head_size, kv_size, n_stream] (fp16 — indexer scoring
        // accuracy matters for correct topk picks; q8_0 would degrade selection).
        ggml_tensor * k_idx = nullptr;
        std::vector<ggml_tensor *> k_idx_stream;
    };

    bool v_trans = true;  // the value tensor is transposed

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;

    // required padding
    const uint32_t n_pad = 1;

    // SWA
    const uint32_t n_swa = 0;

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;
    // Streaming KV-RAM mode: the host KV buffer is never read (the pass points K/V at the GPU staging and
    // the kv_ram mirror holds the rows), so it is left demand-zero instead of cleared (clearing made a
    // full-context duplicate resident: 4.4 GB at 64K cells for Kimi-K2.7).  LLAMA_PREFILL_KV_RAM_HOST_CLEAR=1 restores.
    bool lazy_host_buffers = false;

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    // DSV4 / DeepSeek lightning-indexer QK Hadamard rotation (upstream PR 21038).
    // attn_rot_k/v stay false for all non-DEEPSEEK4 archs -> byte-identical behavior for them.
    bool attn_rot_k = false;
    bool attn_rot_v = false;

    // common per-head key/value dimension across all layers (-1 if not uniform, 0 if unset)
    int32_t n_embd_head_k_all = 0;
    int32_t n_embd_head_v_all = 0;

    // pre-computed Hadamard matrices (host), keyed by rotation size
    std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    std::vector<llama_kv_cells> v_cells;

    // maps from a sequence id to a stream id
    std::vector<uint32_t> seq_to_stream;

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    std::vector<kv_layer> layers;

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);
};

class llama_kv_cache_context : public llama_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    // used for errors
    llama_kv_cache_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_context(
            llama_kv_cache * kv);

    // used to create an update context
    llama_kv_cache_context(
            llama_kv_cache * kv,
            llama_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch processing context from a batch
    llama_kv_cache_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_context specific API
    //

    uint32_t get_n_kv() const;
    int32_t  get_n_kv_max() const { return n_kv_max; }
    bool     has_per_ubatch_nkv() const { return !n_kv_per_ubatch.empty(); }
    uint32_t get_n_kv_for_ubatch(size_t u) const {
        return u < n_kv_per_ubatch.size() ? n_kv_per_ubatch[u] : n_kv;
    }

    // Override n_kv (e.g. to n_kv_max for graph build). seek() will restore per-ubatch values.
    void set_n_kv(int32_t nkv) { n_kv = nkv; }

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be layed out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    // Phase 3.5-lite (iter 29): indexer K cache wrappers.
    // Forwarded to llama_kv_cache::{get,cpy}_k_idx with the current slot_info.
    ggml_tensor * get_k_idx(ggml_context * ctx, int32_t il) const;
    ggml_tensor * cpy_k_idx(ggml_context * ctx, ggml_tensor * k_idx_cur, ggml_tensor * k_idxs, int32_t il) const;

    // Iter 34: full-cache indexer K view (size = n_kv_max). Used by sparse MLA so the
    // scoring tensor has a CONSTANT shape across sched_reserve and runtime — required
    // because top_k = min(indexer_top_k, ne[0]) becomes data-dependent if we use n_kv,
    // breaking graph alloc (cast_op stored at reserve-time small ne, dispatched at
    // runtime against larger sparse_k).
    ggml_tensor * get_k_idx_max(ggml_context * ctx, int32_t il) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    // DSV4 / DeepSeek lightning-indexer QK Hadamard rotation (upstream PR 21038).
    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void fill_input_k_idxs(std::vector<int64_t> & dst, int64_t n_total, const llama_ubatch * ubatch) const;
    void fill_input_v_idxs(std::vector<int64_t> & dst, int64_t n_total, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn, float soft_mask_alpha = 0.0f, float soft_mask_tau = 0.0f) const;
    bool fill_input_kq_mask_f16(std::vector<uint16_t> & dst, int64_t n_kv, const llama_ubatch * ubatch, bool causal_attn, float soft_mask_alpha = 0.0f, float soft_mask_tau = 0.0f) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    // Layer-major support: reposition the cursor to a specific ubatch index
    // without re-applying to the KV cache. This allows re-iterating over
    // ubatches for set_inputs (attention mask, KV indices) after all ubatches
    // have been applied in the initial pass.
    void seek(size_t idx) override;

    size_t n_ubatches() const override { return ubatches.size(); }

private:
    llama_memory_status status;

    llama_kv_cache * kv;
    llama_context * lctx;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;
    int32_t n_kv_max = 0;  // max n_kv across all ubatches (for graph build)
    std::vector<uint32_t> n_kv_per_ubatch;  // per-ubatch n_kv for layer-major
};
