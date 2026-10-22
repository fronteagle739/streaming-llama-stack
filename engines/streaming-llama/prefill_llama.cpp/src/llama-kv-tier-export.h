#pragma once
// Streams the KVARTIF1 export straight from the NVMe tier's row files (docs/KV_TIER_DESIGN.md, P2b).
//
// The stream is the byte sequence llama_kv_cache::state_write would produce for the cells [0, n_cells)
// of one sequence (n_stream, per-stream cell_count, cell meta, v_trans, n_layer word, K per layer,
// k_idx per layer, V per layer), followed by the artifact's tail sections (prompt text, token ids),
// generated in chunks on demand: the RTX never assembles the artifact in RAM, and the transport pulls
// each chunk as it sends. The header is built up front with the payload size known from the layout;
// the artifact carries LLAMA_KV_ARTIFACT_FLAG_NO_CRC (a CRC over the payload would need a second pass
// over the files).
#include "llama-kv-tier.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_kv_tier_export_layer {
    uint32_t il = 0;               // model layer index == tier layer index
    int32_t  k_type = 0;
    uint64_t k_size_row = 0;
    bool     k_idx_present = false;
    int32_t  k_idx_type = 0;
    uint64_t k_idx_size_row = 0;
    bool     v_present = false;
    int32_t  v_type = 0;
    uint64_t v_size_row = 0;
};

struct llama_kv_tier_export {
    // Lay the stream out. Fails (err set) when the tier cannot serve it byte-exact: a row size that
    // differs from the cache's, fewer rows than n_cells, or a transposed V (not streamable from
    // cell-major files).
    // base_prefix (delta shipping): the decoder holds cells [0, base_prefix); only [base_prefix, n_cells)
    // are streamed, with their true positions in the cell meta.
    bool plan(std::shared_ptr<llama_kv_tier> tier, uint32_t n_cells, uint32_t base_prefix, uint32_t n_stream, uint32_t strm,
              int32_t seq_id, uint32_t v_trans, bool has_k_idx_state,
              const std::vector<llama_kv_tier_export_layer> & layers, std::string & err);

    // The bytes after the KV state (prompt text, token ids) and the artifact header.
    void set_tail(std::vector<uint8_t> tail);
    void set_header(std::vector<uint8_t> header);

    uint64_t kv_state_bytes() const { return kv_state_bytes_; }
    uint64_t payload_bytes()  const { return kv_state_bytes_ + tail_.size(); }
    uint64_t total_bytes()    const { return header_.size() + payload_bytes(); }
    uint32_t n_cells()        const { return n_cells_; }
    uint32_t base_prefix()    const { return base_; }
    uint32_t shipped_cells()  const { return n_cells_ - base_; }

    // Fill dst with the next bytes: at most cap, and at least one unless the stream is finished
    // (n_out == 0 then). Row-file segments stop at row boundaries, so n_out may be below cap while
    // bytes remain. Sequential; reset() rewinds.
    bool pull(uint8_t * dst, size_t cap, size_t & n_out, std::string & err);
    void reset();
    uint64_t bytes_pulled() const { return pulled_; }
    bool finished() const { return pulled_ >= total_bytes(); }

private:
    enum seg_kind : int { SEG_BYTES = 0, SEG_META = 1, SEG_TIER = 2 };
    struct segment {
        seg_kind             kind = SEG_BYTES;
        uint64_t             size = 0;
        std::vector<uint8_t> bytes;      // SEG_BYTES
        int                  il = -1;    // SEG_TIER
        int                  tier_kind = 0;
        uint64_t             row_bytes = 0;
        uint64_t             n_rows = 0;
    };
    void push_bytes(const void * p, size_t n);
    void push_meta();
    void push_tier(int il, int kind, uint64_t row_bytes, uint64_t n_rows);
    void build_segments();

    std::shared_ptr<llama_kv_tier> tier_;
    uint32_t n_cells_ = 0, base_ = 0, n_stream_ = 1, strm_ = 0;
    int32_t  seq_id_ = 0;
    uint32_t v_trans_ = 1;
    bool     has_k_idx_ = false;
    std::vector<llama_kv_tier_export_layer> layers_;
    std::vector<uint8_t> header_, tail_;
    uint64_t kv_state_bytes_ = 0;

    std::vector<segment> segs_;
    size_t   seg_i_ = 0;
    uint64_t seg_off_ = 0;
    uint64_t pulled_ = 0;
    std::vector<uint8_t> scratch_;
};
