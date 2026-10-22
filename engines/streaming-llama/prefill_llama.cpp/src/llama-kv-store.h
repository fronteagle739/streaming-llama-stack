#pragma once

// ---------------------------------------------------------------------------
// Persistent (oMLX-style) KV-artifact store for the RTX streaming-prefill fork.
//
// Engine A / Phases A1+A2 of front-eagle-stack's
// `.planning/PERSISTENT_KV_CACHE_PLAN_2026-06-10.md`.
//
// What it does: caches the existing v1.5+ KV artifact bytes VERBATIM (header +
// canonical prompt token IDs + KV payload) to disk, keyed by a content
// fingerprint, so a prompt prefix prefilled in a PRIOR process (e.g. before the
// agent loop tore down + rebuilt the prefill server around a diffusion render,
// or before a Mac receiver restart) is restored in milliseconds and only the
// divergent suffix is re-prefilled. The artifact bytes are the SAME ones the
// Mac receiver already consumes — no new format, no new transport.
//
// Layout (plan §3):
//   <dir>/index.jsonl                     append-only index, one JSON row/entry
//   <dir>/<model_fp>/<chain_hash>.kva     the verbatim artifact bytes
//
// Keying:
//   model_fp   = artifact FNV-1a fingerprint  XOR  XXH64(model path+size+mtime)
//                XOR  KV-layout bits (type_k/v, v_trans, is_mla, flash layout).
//                Image-bearing prompts are EXCLUDED (plan §3, risk §7.3).
//   chain_hash = rolling XXH64 over 256-token chunks seeded by model_fp; mirrors
//                tools/mlx_vlm_decode/prompt_cache_store.py exactly so the two
//                engines' docs align (CHUNK_SIZE_TOKENS=256,
//                h[0]=xxh64(chunk0, seed=model_fp), h[i]=xxh64(chunk_i, seed=h[i-1])).
//
// Reuse (plan §3, §6 A2): longest-prefix — walk the request's chunk chain to the
// deepest indexed entry, then TOKEN-EXACT verify against the artifact's embedded
// canonical prompt token IDs (hash = accelerator, tokens = truth) + CRC32.
// Fail-open to normal streaming on any mismatch (risk §7.4).
//
// Eviction: LRU by size cap (whole artifacts only). Index compaction on boot.
// Reads are sequential/unbuffered — the artifact is NEVER mmap-walked (31.4 GB
// box; page-cache thrash precedent, risk §7.5).
// ---------------------------------------------------------------------------

#include <cstdint>
#include <string>
#include <vector>

#include "llama-kv-artifact.h"

// One row of the on-disk index (mirrors a line of index.jsonl).
struct llama_kv_store_entry {
    std::string           model_fp_hex;   // <model_fp> dir component
    std::string           chain_hash_hex; // <chain_hash>.kva basename
    uint32_t              n_tokens   = 0;  // canonical prompt token count (verify length)
    uint32_t              streamed_prefix = 0; // streamed_prefix_token_count covered by KV
    uint64_t              bytes      = 0;  // artifact byte size on disk
    double                last_used  = 0;  // unix seconds, for LRU
    uint32_t              crc32      = 0;  // artifact payload CRC32 (from summary)
    std::vector<uint64_t> chunk_hashes;    // rolling 256-tok chunk chain (A2 longest-prefix)
    std::vector<int32_t>  token_ids;       // canonical prompt token IDs (token-exact verify truth)
};

// Result of a successful longest-prefix lookup.
struct llama_kv_store_hit {
    std::vector<uint8_t> artifact_bytes;       // verbatim .kva bytes (header+payload)
    std::vector<int32_t> prefix_tokens;        // canonical prefix token IDs (== streamed_prefix length)
    uint32_t             streamed_prefix = 0;  // # leading cells the KV covers
    uint32_t             n_tokens        = 0;  // full canonical prompt token count of the entry
    std::string          chain_hash_hex;       // matched entry id (for logging)
};

class llama_kv_store {
public:
    // Resolve the store directory: explicit `dir` wins, else %LLAMA_KV_CACHE_DIR%.
    // Empty/unset = feature fully OFF (every op is a cheap no-op; enabled()==false).
    // `max_gb` <= 0 falls back to %LLAMA_KV_CACHE_MAX_GB% or the 200 GB default.
    explicit llama_kv_store(const std::string & dir = std::string(), double max_gb = -1.0);

    bool enabled() const { return enabled_; }
    const std::string & dir() const { return dir_; }
    double max_gb() const { return max_bytes_ / double(1024ull * 1024 * 1024); }

    // ----- model_fp -----------------------------------------------------
    // Compose the model fingerprint from the model description string (arch +
    // params + quant, hashed — same input the server uses for pd_model_fp), the
    // GGUF file identity (path+size+mtime), and the KV layout bits. Different
    // layouts / models / files => different namespace dir. Token-ID-exact verify
    // + layout bits in the key together cover the tokenizer-change risk (§7.2).
    static uint64_t compute_model_fp(const std::string & model_desc,
                                     const std::string & model_path,
                                     uint16_t type_k, uint16_t type_v,
                                     uint8_t v_trans, uint8_t is_mla,
                                     uint8_t flash_layout);

    // ----- chain hash (mirrors prompt_cache_store.py) -------------------
    static std::vector<uint64_t> chunk_hashes(const std::vector<int32_t> & token_ids,
                                              uint64_t seed_model_fp);

    // ----- write (A1 write hook) ----------------------------------------
    // Store `artifact_bytes` (already serialized, includes embedded canonical
    // tokens) under model_fp/chain_hash. Write-once per (model_fp, deepest chunk
    // hash) at >= the stored length. Returns true if stored or already present.
    // Image-bearing artifacts (img flags set) are skipped by the caller; this
    // method additionally refuses when token_ids is empty.
    bool put(uint64_t model_fp,
             const std::vector<int32_t> & canonical_tokens,
             uint32_t streamed_prefix,
             uint32_t artifact_crc32,
             const std::vector<uint8_t> & artifact_bytes);

    // ----- read (A1/A2 read hook) ---------------------------------------
    // Longest-prefix lookup for `request_tokens` under `model_fp`. On a verified
    // hit (chain prefix + token-exact + CRC), fills `out` and returns the LCP
    // length (>= lcp_min). Returns 0 on miss (caller falls open to streaming).
    // Multiple prefix lengths for one conversation may coexist (A2): the deepest
    // token-exact entry wins.
    uint32_t lookup(uint64_t model_fp,
                    const std::vector<int32_t> & request_tokens,
                    uint32_t lcp_min,
                    llama_kv_store_hit & out);

    // ----- stats / maintenance ------------------------------------------
    struct stats {
        uint64_t entries     = 0;
        uint64_t bytes       = 0;
        uint64_t hits        = 0;
        uint64_t misses      = 0;
        uint64_t writes      = 0;
        uint64_t evictions   = 0;
    };
    stats snapshot() const;

private:
    bool                              enabled_ = false;
    std::string                       dir_;
    uint64_t                          max_bytes_ = 0;
    std::vector<llama_kv_store_entry> index_;
    mutable stats                     stats_;

    std::string index_path() const;
    std::string model_dir(uint64_t model_fp) const;
    std::string artifact_path(uint64_t model_fp, const std::string & chain_hash_hex) const;

    // boot: parse index.jsonl, drop rows whose .kva is missing, compact.
    void load_and_compact();
    void rewrite_index() const;          // atomic full rewrite (after eviction/compaction)
    void append_index_row(const llama_kv_store_entry & e) const;

    // sequential/unbuffered read of an artifact file (no mmap). false on error.
    bool read_artifact_file(const std::string & path, std::vector<uint8_t> & out) const;

    void evict_to_cap();                 // LRU whole-artifact eviction to <= max_bytes_

    uint64_t current_bytes() const;
};

// Process-wide shared store. Constructed once (lazily) from %LLAMA_KV_CACHE_DIR%
// + %LLAMA_KV_CACHE_MAX_GB% (or the --kv-cache-dir flag, which set those envs
// before the first access). Returns the same instance for the write hook
// (llama-context.cpp) and the read hook (server-context.cpp). When the env is
// empty/unset the store reports enabled()==false and every op is a no-op, so
// the feature is fully off with zero behavior change.
llama_kv_store & llama_kv_store_instance();
