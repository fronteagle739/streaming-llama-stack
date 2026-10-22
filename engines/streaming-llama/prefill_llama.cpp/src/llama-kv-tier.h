#pragma once

// NVMe KV tier for the streaming (layer-major) prefill. See docs/KV_TIER_DESIGN.md.
//
// One append-only row store per conversation:
//     <LLAMA_KV_TIER_DIR>/<model_fp>/<conv_id>/manifest.json
//                                             /L<il>.k     rows [n_tokens x k_row_bytes]
//                                             /L<il>.v     rows (dense archs only)
//                                             /L<il>.kidx  rows (DSA indexer, sparse archs)
// Rows are written exactly once, at their position, when the layer that produced them has finished
// (the copyout worker), and are read back per layer for continuations, the export to the decoder,
// and a restore after a relaunch. Nothing is rewritten; a conversation is only ever appended to or
// deleted whole.

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

struct llama_kv_tier {
    enum kind : int { K = 0, V = 1, KIDX = 2 };

    // LLAMA_KV_TIER_DIR set and non-empty.
    static bool        enabled();
    static std::string root_dir();

    llama_kv_tier() = default;
    ~llama_kv_tier();
    llama_kv_tier(const llama_kv_tier &) = delete;
    llama_kv_tier & operator=(const llama_kv_tier &) = delete;

    // Create/open a conversation directory and its per-layer files. row_bytes[kind][il] == 0 means
    // that kind has no rows for that layer (V for MLA archs, KIDX for dense archs).
    bool open(const std::string & model_fp, const std::string & conv_id, int n_layer,
              const std::vector<size_t> & k_row_bytes, const std::vector<size_t> & v_row_bytes,
              const std::vector<size_t> & kidx_row_bytes, std::string & err);
    void close();
    bool is_open() const { return !layers_.empty(); }

    // Re-open a conversation directory written earlier (by this or a previous server process):
    // reads manifest.json for the layout and row counts and opens the files for append.
    bool open_existing(const std::string & dir, std::string & err);
    // The prompt tokens the rows cover, written to tokens.bin (int32); the first n_tokens() are
    // meaningful. Written once per pass; a follow-up rewrites this small file, not the rows.
    bool write_tokens(const int32_t * tokens, size_t n, std::string & err);
    // Find the stored conversation under <root>/<model_fp> whose tokens share the longest prefix
    // with `tokens` (at least lcp_min). Scans the newest `max_scan` conversations.
    static bool lookup(const std::string & model_fp, const int32_t * tokens, size_t n, uint32_t lcp_min,
                       std::string & best_dir, int64_t & best_lcp, size_t max_scan = 64);

    // Append rows [row0, row0 + n) of layer il. row0 must equal the rows already stored for that
    // file (append-only; the CRC is a running one). Thread-safe (the copyout worker calls this).
    bool write_rows(int il, kind k, int64_t row0, int64_t n, const uint8_t * src, std::string & err);
    // Read rows [row0, row0 + n) of layer il into dst (n * row_bytes bytes).
    bool read_rows(int il, kind k, int64_t row0, int64_t n, uint8_t * dst, std::string & err);

    int64_t  rows(int il, kind k) const;
    size_t   row_bytes(int il, kind k) const;
    uint32_t crc(int il, kind k) const;

    // Record the pass: the number of prompt tokens the rows now cover, plus per-file rows/CRC.
    bool commit(int64_t n_tokens, std::string & err);
    // A follow-up that reuses fewer tokens than the store holds (the chat template retokenizes the
    // boundary) continues from there: rows beyond n are forgotten and the next write appends at n.
    // The few forgotten rows are rewritten later; the running CRCs are invalidated (set to 0).
    bool truncate_to(int64_t n_rows, std::string & err);

    const std::string & path() const { return path_; }
    const std::string & conv_id() const { return conv_id_; }
    int64_t  n_tokens() const { return n_tokens_; }
    int      n_layer() const { return (int) layers_.size(); }
    uint64_t bytes_written() const { return bytes_written_; }
    uint64_t bytes_read() const { return bytes_read_; }
    bool     failed() const { return failed_; }
    const std::string & last_error() const { return last_error_; }

    static uint32_t crc32_update(uint32_t crc, const uint8_t * data, size_t n);

    // Eviction under a cap (LLAMA_KV_TIER_MAX_GB, default 64): when a conversation is opened, the
    // model's conversations are summed and the oldest (by manifest mtime) are removed until the rest
    // fit; the conversation being opened is never removed. Returns the number removed.
    static uint64_t cap_bytes();
    static size_t   evict_to_cap(const std::string & model_fp, const std::string & keep_dir,
                                 uint64_t cap, uint64_t * total_after);

private:
    struct layer_files {
        FILE *   f[3]         = {nullptr, nullptr, nullptr};
        size_t   row_bytes[3] = {0, 0, 0};
        int64_t  n_rows[3]    = {0, 0, 0};
        uint32_t crc[3]       = {0, 0, 0};
    };
    static const char * kind_ext(kind k);
    bool write_manifest(std::string & err);

    std::string root_, model_fp_, conv_id_, path_;
    std::vector<layer_files> layers_;
    int64_t n_tokens_ = 0;
    uint64_t bytes_written_ = 0;
    uint64_t bytes_read_ = 0;
    bool failed_ = false;
    std::string last_error_;
    mutable std::mutex mu_;
};
