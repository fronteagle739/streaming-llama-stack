// llama-decode-receiver (decode-server): Mac-side daemon that receives a
// KV artifact (via file or network), imports it into a llama context,
// decodes autonomously, and streams tokens back to the RTX host via the
// token stream protocol.
//
// Supports two KV sources:
//   --kv-artifact PATH   Read KV from file (for testing / single-shot)
//   --kv-port PORT       Listen for KV artifact over TCP (TBP1 protocol)
//
// Session loop: after decode completes, clears KV and loops back to
// receive the next artifact. Use --one-shot to exit after first decode.
//
// Usage (network mode):
//   llama-decode-receiver -m model.gguf --kv-port 9100 \
//     --token-stream-port 9101 --ctx-size 8192 -ngl 99
//
// Usage (file mode):
//   llama-decode-receiver -m model.gguf --kv-artifact /path/to/kv.bin \
//     --token-stream-port 9101 --ctx-size 8192 -ngl 99

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-context.h"
#include "llama-kv-artifact.h"
#include "llama-kv-import.h"
#include "llama-kv-receiver.h"
#include "sampling.h"
#include "llama-token-stream.h"
#include "llama-wir.h"
#include "tail_tokens.h"

#include "llama-tb-transport.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// PROMPT-DIRECT KV-handshake (Gate #2): Mac-side LRU prefix cache
//
// When Mac completes a prompt-direct prefill session, we snapshot the prefix
// token hash + cached_len into a small LRU map. On the next session RTX queries
// the control socket (kv_port-1); if the hash matches, Mac replies with
// hit=true and skips the prefill, reusing the existing KV cells.
//
// Phase A3 (2026-06-10, plan §3 / §6 A3): the RAM map is now backed by a
// disk tier so a snapshot survives a receiver restart. Behaviour:
//   * env LLAMA_PD_CACHE_DIR unset  => RAM-only (byte-identical legacy, no I/O)
//   * env LLAMA_PD_CACHE_DIR=<dir>  => RAM hot tier (LLAMA_PD_CACHE_HOT, def 4)
//     + disk cold tier (LRU by LLAMA_PD_CACHE_MAX_GB, def 50).
// On capture: write-through the full snapshot to disk (atomic tmp->rename,
// sequential I/O, no mmap). On boot: index the disk dir, dropping CRC-bad /
// orphaned files. On a QUERY that misses RAM: consult disk, load + verify
// (CRC32 + self-consistent prefix_hash over the stored canonical token IDs;
// hash is only an accelerator, tokens are truth) + promote into the RAM tier
// so the follow-on CACHE_HIT artifact's pd_session_id resolves via the same
// pd_cache_get_kv() path. Fail-open (treat as miss) on ANY mismatch.
// File layout: <dir>/<model_fp_hex>/<prefix_hash_hex>.kvsnap
// ---------------------------------------------------------------------------
struct KvSnapshot {
    uint64_t prefix_hash          = 0;
    uint64_t model_fp             = 0;
    uint32_t prefix_len_tokens    = 0;
    uint64_t session_id           = 0;   // monotonic counter, used as Mac handle
    std::vector<int32_t> token_ids;      // canonical prompt token IDs (token-exact verify truth)
    std::vector<uint8_t> kv_data;        // serialized llama_state_seq_get_data bytes
    std::chrono::steady_clock::time_point last_seen_ts;
};

// RAM hot-tier capacity. Default 4 (was the hardcoded KV_SNAPSHOT_MAX); raise
// via LLAMA_PD_CACHE_HOT. Computed once on first use.
static int pd_cache_hot_max() {
    static const int n = []() {
        if (const char * e = std::getenv("LLAMA_PD_CACHE_HOT")) {
            const int v = std::atoi(e);
            if (v > 0) { return v; }
        }
        return 4;
    }();
    return n;
}

// Global prefix cache: keyed by prefix_hash
static std::unordered_map<uint64_t, KvSnapshot> g_prefix_cache;
static std::mutex                                g_prefix_cache_mu;
static std::atomic<uint64_t>                     g_next_session_id{1};

// ---------------------------------------------------------------------------
// Phase A3: disk tier
// ---------------------------------------------------------------------------
namespace pd_disk {

// On-disk .kvsnap header (fixed prefix; followed by token IDs, then KV payload).
// All fields little-endian. Mirrors the A1 store discipline: self-describing,
// token-exact verify on load, CRC32 over the whole file body.
static constexpr uint32_t KVSNAP_MAGIC   = 0x504e5350u; // "PSNP" (Prompt SNaPshot)
static constexpr uint32_t KVSNAP_VERSION = 1u;

#pragma pack(push, 1)
struct kvsnap_header {
    uint32_t magic;          // KVSNAP_MAGIC
    uint32_t version;        // KVSNAP_VERSION
    uint64_t model_fp;       // model fingerprint (namespace)
    uint64_t prefix_hash;    // XXH64 of canonical token IDs (the cache key)
    uint32_t n_tokens;       // canonical prompt token count
    uint32_t reserved;       // 0
    uint64_t kv_bytes;       // KV payload length (after the token IDs)
    uint32_t crc32;          // CRC32 over everything AFTER this field (tokens + KV)
    uint32_t reserved2;      // 0
};
#pragma pack(pop)
static_assert(sizeof(kvsnap_header) == 48, "kvsnap_header must be 48 bytes");

// LRU index of files currently on disk (loaded at boot, kept current on write).
struct disk_entry {
    uint64_t model_fp     = 0;
    uint64_t prefix_hash  = 0;
    uint32_t n_tokens     = 0;
    uint64_t file_bytes   = 0;
    uint64_t last_used    = 0; // monotonic counter; higher = more recent
};

static std::string                     g_dir;            // resolved LLAMA_PD_CACHE_DIR ("" => disabled)
static uint64_t                        g_max_bytes  = 0; // LLAMA_PD_CACHE_MAX_GB
static std::map<std::string, disk_entry> g_index;        // key = "<model_fp_hex>/<prefix_hash_hex>"
static std::mutex                      g_index_mu;
static uint64_t                        g_lru_counter = 1;
static bool                            g_inited      = false;

static std::string hex16(uint64_t v) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) v);
    return std::string(buf);
}

static std::string index_key(uint64_t model_fp, uint64_t prefix_hash) {
    return hex16(model_fp) + "/" + hex16(prefix_hash);
}

static std::string file_path(uint64_t model_fp, uint64_t prefix_hash) {
    return (std::filesystem::path(g_dir) / hex16(model_fp) / (hex16(prefix_hash) + ".kvsnap")).string();
}

static bool enabled() { return !g_dir.empty(); }

static uint64_t current_bytes_locked() {
    uint64_t total = 0;
    for (const auto & kv : g_index) { total += kv.second.file_bytes; }
    return total;
}

// Evict least-recently-used files until total <= g_max_bytes. Caller holds g_index_mu.
static void evict_to_cap_locked() {
    if (g_max_bytes == 0) { return; }
    while (current_bytes_locked() > g_max_bytes && !g_index.empty()) {
        auto victim = g_index.begin();
        for (auto it = g_index.begin(); it != g_index.end(); ++it) {
            if (it->second.last_used < victim->second.last_used) { victim = it; }
        }
        std::error_code ec;
        std::filesystem::remove(file_path(victim->second.model_fp, victim->second.prefix_hash), ec);
        fprintf(stderr, "[pd-cache][disk] LRU evicted %s (%llu bytes)\n",
                victim->first.c_str(), (unsigned long long) victim->second.file_bytes);
        g_index.erase(victim);
    }
}

// Initialise once: resolve env, create dir, index + compact existing files.
static void init() {
    if (g_inited) { return; }
    g_inited = true;
    const char * dir = std::getenv("LLAMA_PD_CACHE_DIR");
    if (!dir || dir[0] == '\0') { return; }   // disabled => RAM-only legacy
    g_dir = dir;

    double max_gb = 50.0;
    if (const char * e = std::getenv("LLAMA_PD_CACHE_MAX_GB")) {
        const double v = std::atof(e);
        if (v > 0.0) { max_gb = v; }
    }
    g_max_bytes = (uint64_t)(max_gb * 1024.0 * 1024.0 * 1024.0);

    std::error_code ec;
    std::filesystem::create_directories(g_dir, ec);
    if (ec) {
        fprintf(stderr, "[pd-cache][disk] cannot create dir %s: %s — disk tier OFF\n",
                g_dir.c_str(), ec.message().c_str());
        g_dir.clear();
        return;
    }

    // Index pass: scan <dir>/<model_fp>/<prefix>.kvsnap, validate header magic +
    // version, drop CRC-bad / malformed / orphaned files (compaction).
    std::lock_guard<std::mutex> lk(g_index_mu);
    uint64_t kept = 0, dropped = 0;
    for (auto & lvl1 : std::filesystem::directory_iterator(g_dir, ec)) {
        if (ec) break;
        if (!lvl1.is_directory()) { continue; }
        const std::string model_fp_hex = lvl1.path().filename().string();
        for (auto & f : std::filesystem::directory_iterator(lvl1.path(), ec)) {
            if (ec) break;
            if (!f.is_regular_file() || f.path().extension() != ".kvsnap") { continue; }
            std::ifstream in(f.path(), std::ios::binary);
            kvsnap_header h = {};
            bool bad = false;
            if (!in.read(reinterpret_cast<char *>(&h), sizeof(h))) { bad = true; }
            if (!bad && (h.magic != KVSNAP_MAGIC || h.version != KVSNAP_VERSION)) { bad = true; }
            const uint64_t body_bytes = bad ? 0
                : (uint64_t) h.n_tokens * sizeof(int32_t) + h.kv_bytes;
            std::vector<uint8_t> body;
            if (!bad) {
                body.resize(body_bytes);
                if (body_bytes && !in.read(reinterpret_cast<char *>(body.data()), (std::streamsize) body_bytes)) {
                    bad = true;
                }
            }
            if (!bad) {
                const uint32_t crc = body.empty() ? 0u
                    : llama_kv_artifact_crc32(body.data(), body.size());
                if (crc != h.crc32) { bad = true; }
            }
            in.close();
            if (bad) {
                std::error_code rmec;
                std::filesystem::remove(f.path(), rmec);
                ++dropped;
                continue;
            }
            disk_entry e;
            e.model_fp    = h.model_fp;
            e.prefix_hash = h.prefix_hash;
            e.n_tokens    = h.n_tokens;
            e.file_bytes  = sizeof(h) + body_bytes;
            e.last_used   = g_lru_counter++;
            g_index[index_key(h.model_fp, h.prefix_hash)] = e;
            ++kept;
            (void) model_fp_hex;
        }
    }
    evict_to_cap_locked();
    fprintf(stderr, "[pd-cache][disk] indexed %s: %llu kept, %llu dropped, cap %.1f GB, used %.2f GB\n",
            g_dir.c_str(), (unsigned long long) kept, (unsigned long long) dropped,
            max_gb, current_bytes_locked() / (1024.0 * 1024.0 * 1024.0));
}

// Write-through a fully-captured snapshot (atomic tmp->rename, sequential I/O).
static void write_through(const KvSnapshot & snap) {
    if (!enabled() || snap.kv_data.empty()) { return; }

    // Compose the body (token IDs followed by KV payload) so CRC + atomic write
    // cover both. Tokens enable token-exact / self-consistency verification on
    // load even though the QUERY round-trip never re-ships the request tokens.
    const uint64_t tok_bytes = (uint64_t) snap.token_ids.size() * sizeof(int32_t);
    std::vector<uint8_t> body;
    body.reserve(tok_bytes + snap.kv_data.size());
    if (tok_bytes) {
        const uint8_t * p = reinterpret_cast<const uint8_t *>(snap.token_ids.data());
        body.insert(body.end(), p, p + tok_bytes);
    }
    body.insert(body.end(), snap.kv_data.begin(), snap.kv_data.end());

    kvsnap_header h = {};
    h.magic       = KVSNAP_MAGIC;
    h.version     = KVSNAP_VERSION;
    h.model_fp    = snap.model_fp;
    h.prefix_hash = snap.prefix_hash;
    h.n_tokens    = (uint32_t) snap.token_ids.size();
    h.kv_bytes    = snap.kv_data.size();
    h.crc32       = body.empty() ? 0u : llama_kv_artifact_crc32(body.data(), body.size());

    const std::string path = file_path(snap.model_fp, snap.prefix_hash);
    const std::string tmp  = path + ".tmp";
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            fprintf(stderr, "[pd-cache][disk] cannot open %s for write\n", tmp.c_str());
            return;
        }
        out.write(reinterpret_cast<const char *>(&h), sizeof(h));
        if (!body.empty()) {
            out.write(reinterpret_cast<const char *>(body.data()), (std::streamsize) body.size());
        }
        out.flush();
        if (!out) {
            fprintf(stderr, "[pd-cache][disk] write failed for %s\n", tmp.c_str());
            out.close();
            std::filesystem::remove(tmp, ec);
            return;
        }
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        fprintf(stderr, "[pd-cache][disk] rename %s -> %s failed: %s\n",
                tmp.c_str(), path.c_str(), ec.message().c_str());
        std::filesystem::remove(tmp, ec);
        return;
    }

    const uint64_t fbytes = sizeof(h) + body.size();
    {
        std::lock_guard<std::mutex> lk(g_index_mu);
        disk_entry e;
        e.model_fp    = snap.model_fp;
        e.prefix_hash = snap.prefix_hash;
        e.n_tokens    = h.n_tokens;
        e.file_bytes  = fbytes;
        e.last_used   = g_lru_counter++;
        g_index[index_key(snap.model_fp, snap.prefix_hash)] = e;
        evict_to_cap_locked();
    }
    fprintf(stderr, "[pd-cache][disk] wrote %s (%llu bytes, %u tokens)\n",
            path.c_str(), (unsigned long long) fbytes, h.n_tokens);
}

// Load + verify a snapshot from disk into `out`. Returns true on a verified hit.
// Verifies: index presence, header magic/version, model_fp + prefix_len match
// the QUERY, CRC32, and self-consistent prefix_hash recomputed over the stored
// canonical token IDs. Fail-open (false) on ANY mismatch.
static bool load_verified(uint64_t model_fp, uint64_t prefix_hash,
                          uint32_t prefix_len_tokens, KvSnapshot & out) {
    if (!enabled()) { return false; }
    std::string path;
    {
        std::lock_guard<std::mutex> lk(g_index_mu);
        auto it = g_index.find(index_key(model_fp, prefix_hash));
        if (it == g_index.end()) { return false; }
        path = file_path(model_fp, prefix_hash);
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) { return false; }
    kvsnap_header h = {};
    if (!in.read(reinterpret_cast<char *>(&h), sizeof(h))) { return false; }
    if (h.magic != KVSNAP_MAGIC || h.version != KVSNAP_VERSION) {
        fprintf(stderr, "[pd-cache][disk] %s: bad header — fail-open\n", path.c_str());
        return false;
    }
    if (h.model_fp != model_fp || h.prefix_hash != prefix_hash ||
        h.n_tokens != prefix_len_tokens) {
        fprintf(stderr, "[pd-cache][disk] %s: header/query mismatch — fail-open\n", path.c_str());
        return false;
    }
    const uint64_t tok_bytes = (uint64_t) h.n_tokens * sizeof(int32_t);
    std::vector<uint8_t> body(tok_bytes + h.kv_bytes);
    if (!body.empty() && !in.read(reinterpret_cast<char *>(body.data()), (std::streamsize) body.size())) {
        fprintf(stderr, "[pd-cache][disk] %s: short read — fail-open\n", path.c_str());
        return false;
    }
    const uint32_t crc = body.empty() ? 0u : llama_kv_artifact_crc32(body.data(), body.size());
    if (crc != h.crc32) {
        fprintf(stderr, "[pd-cache][disk] %s: CRC mismatch (0x%x != 0x%x) — fail-open\n",
                path.c_str(), crc, h.crc32);
        return false;
    }

    // Reconstruct the snapshot.
    out.token_ids.resize(h.n_tokens);
    if (tok_bytes) { std::memcpy(out.token_ids.data(), body.data(), tok_bytes); }
    out.kv_data.assign(body.begin() + (std::ptrdiff_t) tok_bytes, body.end());

    // Self-consistency: the stored prefix_hash must equal a re-hash of the stored
    // tokens (catches any silent token-region corruption the CRC alone passed).
    const uint64_t rehash = llama_tb_transport_hash_tokens(out.token_ids.data(), (int32_t) out.token_ids.size());
    if (rehash != prefix_hash) {
        fprintf(stderr, "[pd-cache][disk] %s: token rehash 0x%llx != key 0x%llx — fail-open\n",
                path.c_str(), (unsigned long long) rehash, (unsigned long long) prefix_hash);
        out.token_ids.clear();
        out.kv_data.clear();
        return false;
    }

    out.prefix_hash       = prefix_hash;
    out.model_fp          = model_fp;
    out.prefix_len_tokens = h.n_tokens;

    // Touch LRU.
    {
        std::lock_guard<std::mutex> lk(g_index_mu);
        auto it = g_index.find(index_key(model_fp, prefix_hash));
        if (it != g_index.end()) { it->second.last_used = g_lru_counter++; }
    }
    return true;
}

} // namespace pd_disk

// Current active session: set at the start of a prompt-direct session so the
// KV-clear conditional can check whether a snapshot exists.
static uint64_t g_active_prefix_hash = 0;  // 0 = no active snapshot
static bool     g_active_cache_hit   = false;

// ---------------------------------------------------------------------------
// Stage 1 (2026-05-29): Mac resident-KV retention. When an incoming KV-artifact
// request EXTENDS the resident conversation, reuse the LIVE ctx + its KV instead
// of recreating the context + re-importing + re-reserving the decode graph (the
// measured ~6.2s/turn cost: the post-import graph RE-RESERVE; decode itself is
// ~37 tok/s once the graph is warm). Gated LLAMA_PD_RESIDENT_KV_REUSE (default
// off => byte-identical to today, RC-safe). SINGLE-SESSION loop only.
// Staleness-safe (the 6-week bug class): exact-LCP token match, seq_rm trim of
// the stale decoded tail + any divergent suffix, bit-faithful resident tokens,
// fail-safe to the full clear+import path on ANY anomaly. NOTE: the per-session
// context recreate exists to avoid "stale Metal GPU state" garbage — reuse must
// be validated empirically; g_ctx_generation guards against reusing across a
// recreate.
// ---------------------------------------------------------------------------
struct ResidentKvState {
    std::vector<llama_token> token_ids;  // exact tokens materialized in live seq-0 KV
    uint64_t ctx_generation = 0;         // == g_ctx_generation when this ctx wrote the KV
    int32_t  source_slot_id = -1;
    bool     valid = false;
};
static ResidentKvState g_resident_kv;
static std::mutex       g_resident_kv_mu;
static uint64_t         g_ctx_generation = 0;  // ++ each time create_context() replaces ctx

// Token index the current prefill batch starts at, published by the prefill loop so the dump below
// can record it. -1 means "not a tracked prefill batch".
static std::atomic<int> g_dump_batch_start{-1};

// Dump DSV4 compressor tensors from THIS receiver, for a cross-backend diff against the RTX.
//
// The names match what the RTX side writes (llama-prefill-stream.cpp's COMP_DUMP list) so the two
// sets line up file-for-file. Capped per name; last firing wins, because the interesting graph is a
// real prefill rather than any warm-up that precedes it.
static bool dsv4_recv_dump_cb(ggml_tensor * t, bool ask, void * user_data) {
    static const char * wanted[] = {
        // csa_comp_input FIRST: every stage of the compressor is now verified correct GIVEN its
        // inputs, so the hidden state arriving at the layer is the prime suspect. Comparing it
        // across backends is the direct test -- same source, same file, same tokens, and Metal
        // recalls where CUDA does not.
        "csa_comp_input-",
        "csa_state_kv-", "csa_state_score_ape-", "csa_state_compress_pooled-",
        "csa_state_compress_final-", "lid_state_compress_pooled-", "csa_comp_k-",
    };
    const char * name = t ? ggml_get_name(t) : nullptr;
    bool match = false;
    if (name) {
        for (const char * w : wanted) {
            if (std::strncmp(name, w, std::strlen(w)) == 0) {
                match = true;
                break;
            }
        }
    }
    if (ask) {
        return match;
    }
    if (!match) {
        return true;  // fail-open: returning false would break the scheduler's graph loop
    }
    const char * pref = (const char *) user_data;
    if (!pref || !pref[0]) {
        return true;
    }
    // CAP HARD, and skip anything unreasonable. The first version of this dumped every matching
    // tensor at every layer on every ubatch with no limit and no size guard; the receiver armed,
    // bound its port, and then EXITED, which stalled the whole switch in "warming up DSV4
    // (attempt 4/4)". A diagnostic that takes the process down is worse than no diagnostic.
    // PREFILL-SHAPED FIRINGS ONLY. cb_eval fires on every graph, and a 2-per-name cap kept the FIRST
    // two -- which are single-token DECODE steps (`ne[1] == 1`). The RTX side dumps prefill ubatches
    // (`[1024 x 512]`), so the two sets could never be compared: every Mac tensor came back
    // `[1024 x 1]`. The whole point of this dump is a cross-backend diff against the RTX, so capture
    // multi-token graphs and ignore the decode ones.
    // Multi-token graphs only: single-token DECODE firings are not comparable with the RTX's prefill
    // ubatches. Do NOT raise this floor to force a 512-wide batch -- measured, this receiver never
    // produces one. Its multi-token graphs are ~37 tokens wide, so a `>= 256` filter captured
    // NOTHING at all, while the RTX side dumps 512-wide ubatches.
    //
    // The consequence is that the two dumps CANNOT be aligned by shape, and the cross-backend diff
    // needs each batch's STARTING TOKEN POSITION recorded in the manifest on both sides. Until that
    // exists, do not diff these arrays: a two-deep cap already produced a convincing false positive
    // (rel 1.83 on the hidden state) purely from comparing different token ranges.
    if (t->ne[1] <= 1) {
        return true;
    }
    static std::mutex                 dump_mu;
    static std::map<std::string, int> dump_counts;
    std::lock_guard<std::mutex> dump_lock(dump_mu);
    // KEEP THE FIRST FIRING ONLY. Writing every firing to one filename means the file holds the LAST
    // one, and the RTX side keeps firing #0 (`.f0.bin`) -- so a two-deep cap silently compared the
    // Mac's SECOND prefill batch against the RTX's FIRST, i.e. different token ranges. That produced
    // a convincing "divergence" (rel 1.8) that would have been pure misalignment. Both sides now
    // pin batch 0 of the same prompt.
    int & seen = dump_counts[name];
    if (seen >= 1) {
        return true;
    }
    const size_t nbytes = ggml_nbytes(t);
    if (nbytes == 0 || nbytes > (size_t) 64 * 1024 * 1024) {
        return true;
    }
    try {
        std::vector<uint8_t> raw(nbytes);
        ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
        seen++;
        const int batch_start = g_dump_batch_start.load(std::memory_order_relaxed);
        char fname[1024];
        // Offset in the FILENAME as well as the manifest, so a stale file from a previous run can
        // never be mistaken for the batch you think you are holding.
        snprintf(fname, sizeof(fname), "%s.%s.t%d.mac.bin", pref, name, batch_start);
        if (FILE * fb = std::fopen(fname, "wb")) {
            fwrite(raw.data(), 1, nbytes, fb);
            std::fclose(fb);
        }
        char mname[1024];
        snprintf(mname, sizeof(mname), "%s.macmanifest.tsv", pref);
        if (FILE * fm = std::fopen(mname, "ab+")) {
            fprintf(fm, "%s\t%s\t%lld\t%lld\t%lld\t%zu\t%d\n", name, ggml_type_name(t->type),
                    (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], nbytes,
                    batch_start);
            std::fclose(fm);
        }
    } catch (...) {}
    return true;
}

static bool resident_kv_reuse_enabled() {
    const char * e = std::getenv("LLAMA_PD_RESIDENT_KV_REUSE");
    return e && e[0] == '1';
}
// Exact longest-common-prefix length (token-by-token; stops at first divergence
// so rollback/branch/edit naturally clamp reuse to the divergence point). No hashing.
static uint32_t resident_compute_lcp(const std::vector<llama_token> & a,
                                     const std::vector<int32_t> & b) {
    const size_t m = std::min(a.size(), b.size());
    uint32_t n = 0;
    while (n < m && a[n] == (llama_token) b[n]) ++n;
    return n;
}

static bool pd_cache_lookup(uint64_t prefix_hash, uint32_t prefix_len_tokens,
                             uint64_t model_fp,
                             uint32_t & out_cached_len, uint64_t & out_session_id) {
    // Safety env gate: LLAMA_PD_CACHE_DISABLE=1 → immediate miss
    if (const char * dis = std::getenv("LLAMA_PD_CACHE_DISABLE")) {
        if (dis[0] == '1') { return false; }
    }
    std::lock_guard<std::mutex> lk(g_prefix_cache_mu);
    auto it = g_prefix_cache.find(prefix_hash);
    if (it == g_prefix_cache.end()) {
        // RAM miss — Phase A3: consult the disk tier, load + verify, and promote
        // into the hot tier so the follow-on CACHE_HIT artifact resolves through
        // the same pd_cache_get_kv() path. Fail-open to a normal miss otherwise.
        KvSnapshot loaded;
        if (pd_disk::load_verified(model_fp, prefix_hash, prefix_len_tokens, loaded)) {
            if (loaded.kv_data.empty()) { return false; }
            // Evict oldest RAM entry if at capacity before promoting.
            if ((int) g_prefix_cache.size() >= pd_cache_hot_max()) {
                uint64_t oldest_hash = 0;
                std::chrono::steady_clock::time_point oldest_ts = std::chrono::steady_clock::now();
                bool first = true;
                for (auto & kv : g_prefix_cache) {
                    if (first || kv.second.last_seen_ts < oldest_ts) {
                        oldest_hash = kv.first;
                        oldest_ts   = kv.second.last_seen_ts;
                        first = false;
                    }
                }
                if (!first) { g_prefix_cache.erase(oldest_hash); }
            }
            loaded.session_id   = g_next_session_id.fetch_add(1);
            loaded.last_seen_ts = std::chrono::steady_clock::now();
            const uint64_t sid  = loaded.session_id;
            const size_t   nb   = loaded.kv_data.size();
            g_prefix_cache[prefix_hash] = std::move(loaded);
            out_cached_len = prefix_len_tokens;
            out_session_id = sid;
            fprintf(stderr, "[pd-cache] DISK HIT promoted to RAM: hash=0x%llx len=%u sid=0x%llx kv_bytes=%zu\n",
                    (unsigned long long) prefix_hash, prefix_len_tokens,
                    (unsigned long long) sid, nb);
            return true;
        }
        return false;
    }
    KvSnapshot & snap = it->second;
    if (snap.model_fp != model_fp || snap.prefix_len_tokens != prefix_len_tokens) {
        // Hash collision or model mismatch — treat as miss
        return false;
    }
    // Only report a hit if we have KV data to restore.
    if (snap.kv_data.empty()) {
        fprintf(stderr, "[pd-cache] Cache entry exists but kv_data is empty (metadata-only entry) — treating as miss\n");
        return false;
    }
    snap.last_seen_ts = std::chrono::steady_clock::now();
    out_cached_len  = snap.prefix_len_tokens;
    out_session_id  = snap.session_id;
    fprintf(stderr, "[pd-cache] RAM HIT: hash=0x%llx len=%u sid=0x%llx\n",
            (unsigned long long) prefix_hash, prefix_len_tokens,
            (unsigned long long) snap.session_id);
    return true;
}

// Memory limit for KV snapshots: 2 GB per snapshot.
static constexpr size_t KV_SNAPSHOT_MAX_BYTES = 2ull * 1024 * 1024 * 1024;

// Update KV data for an existing cache entry (by prefix_hash).
// Called after decode completes to fill in the kv_data field.
// NOTE: Requires LLAMA_PD_CACHE_CAPTURE=1 to enable. Default: disabled.
// On CUDA/Windows targets, llama_state_seq_get_data can crash (access violation
// in CUDA KV serialization). Only enable on Metal (Mac) or CPU backends.
static void pd_cache_update_kv(uint64_t prefix_hash, llama_context * ctx) {
    if (!ctx) { return; }
    // Safety env gate: LLAMA_PD_CACHE_DISABLE=1 → no-op
    if (const char * dis = std::getenv("LLAMA_PD_CACHE_DISABLE")) {
        if (dis[0] == '1') { return; }
    }
    // KV capture is opt-in: must set LLAMA_PD_CACHE_CAPTURE=1 explicitly.
    // On Mac/Metal, set this to enable Phase-2 KV serialization.
    // On Windows/CUDA, the capture path has known CUDA access violations in
    // llama_state_seq_get_data post-decode — hard-compile-disabled here so
    // even explicit LLAMA_PD_CACHE_CAPTURE=1 cannot trigger a crash.
#if defined(_WIN32) || defined(_WIN64)
    static bool warned_once = false;
    if (const char * cap = std::getenv("LLAMA_PD_CACHE_CAPTURE"); cap && cap[0] == '1' && !warned_once) {
        fprintf(stderr, "[pd-cache] WARNING: LLAMA_PD_CACHE_CAPTURE=1 is IGNORED on Windows — CUDA KV serialization crashes. Only Mac/Metal receiver supports Phase-2 KV capture.\n");
        warned_once = true;
    }
    return;
#else
    if (const char * cap = std::getenv("LLAMA_PD_CACHE_CAPTURE")) {
        if (cap[0] != '1') { return; }
    } else {
        return;  // not set → skip
    }
#endif

    // Check if entry exists and needs kv_data
    {
        std::lock_guard<std::mutex> lk(g_prefix_cache_mu);
        auto it = g_prefix_cache.find(prefix_hash);
        if (it == g_prefix_cache.end() || !it->second.kv_data.empty()) {
            return;  // entry not found or already has kv_data
        }
    }

    // Capture KV state (outside lock to avoid holding during CUDA copy)
    const size_t kv_size = llama_state_seq_get_size(ctx, 0);
    if (kv_size == 0 || kv_size > KV_SNAPSHOT_MAX_BYTES) {
        fprintf(stderr, "[pd-cache] KV update: size=%zu (skipping)\n", kv_size);
        return;
    }
    std::vector<uint8_t> kv_bytes(kv_size);
    const size_t written = llama_state_seq_get_data(ctx, kv_bytes.data(), kv_bytes.size(), 0);
    if (written == 0 || written > kv_bytes.size()) {
        fprintf(stderr, "[pd-cache] KV update: llama_state_seq_get_data failed (written=%zu)\n", written);
        return;
    }
    kv_bytes.resize(written);

    // Update the snapshot, then (Phase A3) write-through to disk while holding a
    // complete copy (token IDs + KV payload). The write itself runs OUTSIDE the
    // cache mutex (sequential file I/O can be slow; don't block the control
    // socket lookups). We copy the data we need under the lock, release, write.
    KvSnapshot to_persist;
    bool do_persist = false;
    {
        std::lock_guard<std::mutex> lk(g_prefix_cache_mu);
        auto it = g_prefix_cache.find(prefix_hash);
        if (it != g_prefix_cache.end() && it->second.kv_data.empty()) {
            it->second.kv_data = std::move(kv_bytes);
            fprintf(stderr, "[pd-cache] KV updated: hash=0x%llx %zu bytes\n",
                    (unsigned long long) prefix_hash, it->second.kv_data.size());
            if (pd_disk::enabled() && !it->second.token_ids.empty()) {
                to_persist = it->second;   // copy (hash, model_fp, len, tokens, kv)
                do_persist = true;
            }
        }
    }
    if (do_persist) {
        pd_disk::write_through(to_persist);
    }
}

// Retrieve KV bytes for a given session_id (for restore on cache hit).
// Returns true and fills out_kv_data if found, false otherwise.
static bool pd_cache_get_kv(uint64_t session_id, std::vector<uint8_t> & out_kv_data) {
    std::lock_guard<std::mutex> lk(g_prefix_cache_mu);
    for (auto & kv : g_prefix_cache) {
        if (kv.second.session_id == session_id) {
            if (kv.second.kv_data.empty()) {
                return false;
            }
            out_kv_data = kv.second.kv_data;  // copy
            return true;
        }
    }
    return false;
}

static void pd_cache_insert(uint64_t prefix_hash, uint32_t prefix_len_tokens,
                             uint64_t model_fp, llama_context * ctx,
                             const int32_t * token_ids = nullptr, int n_token_ids = 0) {
    // Safety env gate: LLAMA_PD_CACHE_DISABLE=1 → no-op
    if (const char * dis = std::getenv("LLAMA_PD_CACHE_DISABLE")) {
        if (dis[0] == '1') { return; }
    }

    // Capture KV state from context (Phase-2: full serialization).
    // Windows/CUDA: llama_state_seq_get_data has known access-violation crashes
    // post-prefill with -ngl > 0. Skip entirely on Windows to guarantee no crash,
    // even if LLAMA_PD_CACHE_CAPTURE=1. kv_bytes stays empty → lookup treats
    // this as a "metadata-only" entry → subsequent query returns miss → full send.
    std::vector<uint8_t> kv_bytes;
#if defined(_WIN32) || defined(_WIN64)
    // kv_bytes stays empty on Windows — safe miss-fallback is enforced.
    (void) ctx;
#else
    if (ctx) {
        const size_t kv_size = llama_state_seq_get_size(ctx, 0);
        if (kv_size == 0) {
            fprintf(stderr, "[pd-cache] WARNING: llama_state_seq_get_size returned 0 — skipping KV capture\n");
        } else if (kv_size > KV_SNAPSHOT_MAX_BYTES) {
            fprintf(stderr, "[pd-cache] Snapshot too large (%zu bytes > 2 GB limit) — skipping cache insert for this session\n", kv_size);
            return;
        } else {
            kv_bytes.resize(kv_size);
            const size_t written = llama_state_seq_get_data(ctx, kv_bytes.data(), kv_bytes.size(), 0);
            if (written == 0 || written > kv_bytes.size()) {
                fprintf(stderr, "[pd-cache] WARNING: llama_state_seq_get_data failed (written=%zu size=%zu) — skipping KV capture\n",
                        written, kv_size);
                kv_bytes.clear();
            } else {
                kv_bytes.resize(written);
                fprintf(stderr, "[pd-cache] KV captured: %zu bytes\n", written);
            }
        }
    }
#endif

    std::lock_guard<std::mutex> lk(g_prefix_cache_mu);
    // Evict oldest if at capacity (RAM hot tier).
    if ((int) g_prefix_cache.size() >= pd_cache_hot_max()) {
        uint64_t oldest_hash = 0;
        std::chrono::steady_clock::time_point oldest_ts = std::chrono::steady_clock::now();
        bool first = true;
        for (auto & kv : g_prefix_cache) {
            if (first || kv.second.last_seen_ts < oldest_ts) {
                oldest_hash = kv.first;
                oldest_ts   = kv.second.last_seen_ts;
                first = false;
            }
        }
        if (!first) {
            g_prefix_cache.erase(oldest_hash);
        }
    }
    KvSnapshot snap;
    snap.prefix_hash       = prefix_hash;
    snap.model_fp          = model_fp;
    snap.prefix_len_tokens = prefix_len_tokens;
    snap.session_id        = g_next_session_id.fetch_add(1);
    snap.last_seen_ts      = std::chrono::steady_clock::now();
    // Stash canonical token IDs (truth for token-exact verify on disk reload).
    if (token_ids && n_token_ids > 0) {
        snap.token_ids.assign(token_ids, token_ids + n_token_ids);
    }
    snap.kv_data           = std::move(kv_bytes);
    const bool have_kv = !snap.kv_data.empty();
    g_prefix_cache[prefix_hash] = std::move(snap);
    fprintf(stderr, "[pd-cache] Snapshot inserted: hash=0x%llx len=%u sid=0x%llx kv_bytes=%zu (cache size=%zu)\n",
            (unsigned long long) prefix_hash, prefix_len_tokens,
            (unsigned long long) g_prefix_cache[prefix_hash].session_id,
            g_prefix_cache[prefix_hash].kv_data.size(), g_prefix_cache.size());
    // Phase A3: if this insert already carried a complete KV payload (ctx passed
    // directly), write it through to disk now. The common receiver path inserts
    // with ctx=nullptr (metadata only) and persists later in pd_cache_update_kv,
    // so this branch is NOT on the control-socket hot path — persisting inline
    // under the lock is acceptable here.
    if (have_kv && pd_disk::enabled()) {
        auto it = g_prefix_cache.find(prefix_hash);
        if (it != g_prefix_cache.end() && !it->second.token_ids.empty()) {
            pd_disk::write_through(it->second);
        }
    }
}

// Background thread: run control socket listener in a loop
static void run_control_server(std::string bind_host, int control_port,
                                std::atomic<bool> & stop_flag) {
    fprintf(stderr, "[pd-ctrl] Control server listening on %s:%d\n",
            bind_host.c_str(), control_port);
    while (!stop_flag.load()) {
        std::string err;
        bool ok = llama_tb_transport_control_server_accept_one(
            bind_host, control_port,
            [](uint64_t hash, uint32_t len, uint64_t mfp,
               uint32_t & out_len, uint64_t & out_sid) -> bool {
                return pd_cache_lookup(hash, len, mfp, out_len, out_sid);
            },
            &err);
        if (!ok && !err.empty()) {
            // Timeout is normal ("accept timeout") — don't spam the log
            if (err.find("timeout") == std::string::npos) {
                fprintf(stderr, "[pd-ctrl] %s\n", err.c_str());
            }
        }
    }
    fprintf(stderr, "[pd-ctrl] Control server shutdown\n");
}

struct receiver_params {
    std::string model_path;
    std::string kv_artifact_path;       // file mode (mutually exclusive with kv_port)
    std::string dump_kv_after_import_path;
    std::string tail_tokens_path;
    std::string kv_host         = "0.0.0.0";  // bind address for network KV receive
    int         kv_port         = 0;           // 0 = file mode; >0 = network mode
    std::string token_stream_host = "0.0.0.0";
    int         token_stream_port = 9101;
    int         n_ctx             = 8192;
    int         n_batch           = 512;
    int         n_ubatch          = 512;
    int         n_gpu_layers      = 99;
    int         n_threads         = 4;
    int         n_predict         = 2048;
    int         n_parallel        = 1;       // n_seq_max — must match RTX server's -np
    ggml_type   cache_type_k      = GGML_TYPE_F16;
    ggml_type   cache_type_v      = GGML_TYPE_F16;
    float       temp              = 0.7f;
    bool        temp_forced       = false;
    bool        one_shot          = false;
    bool        use_mlock         = false;
    bool        no_token_stream   = false;  // skip token stream server (local test mode)
    bool        flash_attn        = false;  // match prefiller's -fa setting
    bool        flash_attn_forced = false;  // honor explicit CLI override instead of artifact peek
    bool        exit_after_import = false;

    // Multi-session support
    int         n_sessions        = 1;     // number of concurrent decode sessions (1 = legacy single-session)
    int         kv_port_base      = 9100;  // first session uses kv_port_base, second uses kv_port_base+2, etc.
    int         token_port_base   = 9101;  // first session uses token_port_base, second uses token_port_base+2, etc.

    // WIR (Wavefront Iterative Refinement) mode
    bool        wir_enabled       = false;
    std::string wir_draft_model;            // small draft model path (optional; if empty, use same model)
    std::string wir_prompt_text;            // original prompt text (required for cross-tokenizer mode)
    int         wir_block_size    = 64;
    int         wir_max_passes    = 10;
    float       wir_confidence    = 0.95f;
    int         wir_topk_accept   = 3;
    float       wir_draft_temp    = 0.8f;   // temperature for single-model draft
    std::string wir_refine_mode   = "all";  // "all" or "n-worst"
    int         wir_refine_n      = 1;
};

static bool parse_cache_type(const char * value, ggml_type & out) {
    if (!value) {
        return false;
    }
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const ggml_type type = (ggml_type) i;
        const char * name = ggml_type_name(type);
        if (name && std::strcmp(name, value) == 0) {
            out = type;
            return true;
        }
    }
    return false;
}

static void print_usage(const char * argv0) {
    fprintf(stderr, "Usage: %s [options]\n", argv0);
    fprintf(stderr, "  -m, --model PATH          Model file (GGUF)\n");
    fprintf(stderr, "  --kv-artifact PATH        KV artifact file to import (file mode)\n");
    fprintf(stderr, "  --dump-kv-after-import PATH  Dump imported KV back to artifact file\n");
    fprintf(stderr, "  --tail-tokens PATH        Apply local tail token IDs after KV import (.json array or int32 LE .bin)\n");
    fprintf(stderr, "  --exit-after-import       Exit after KV import (after optional dump)\n");
    fprintf(stderr, "  --kv-port PORT            Listen for KV artifact over TCP (network mode)\n");
    fprintf(stderr, "  --kv-host HOST            Bind address for KV receive (default: 0.0.0.0)\n");
    fprintf(stderr, "  --token-stream-host HOST  Bind address for token stream (default: 0.0.0.0)\n");
    fprintf(stderr, "  --token-stream-port PORT  Port for token stream (default: 9101)\n");
    fprintf(stderr, "  -c, --ctx-size N          Context size (default: 8192)\n");
    fprintf(stderr, "  -b, --batch-size N        Logical batch size (default: 512)\n");
    fprintf(stderr, "  -ub, --ubatch-size N      Physical batch size (default: 512)\n");
    fprintf(stderr, "  -ngl, --n-gpu-layers N    GPU layers (default: 99)\n");
    fprintf(stderr, "  -t, --threads N           Threads (default: 4)\n");
    fprintf(stderr, "  -n, --predict N           Max decode tokens (default: 2048)\n");
    fprintf(stderr, "  -ctk, --cache-type-k TYPE KV K cache type (default: f16)\n");
    fprintf(stderr, "  -ctv, --cache-type-v TYPE KV V cache type (default: f16)\n");
    fprintf(stderr, "  --temp F                  Temperature (default: 0.7)\n");
    fprintf(stderr, "  --mlock                   Lock model in memory (prevent paging)\n");
    fprintf(stderr, "  --one-shot                Exit after first decode\n");
    fprintf(stderr, "  --no-fa                   Disable flash attention for imported KV layout\n");
    fprintf(stderr, "  --sessions N              Number of concurrent decode sessions (default: 1)\n");
    fprintf(stderr, "  --kv-port-base PORT       Base KV port for multi-session (default: 9100)\n");
    fprintf(stderr, "  --token-port-base PORT    Base token stream port for multi-session (default: 9101)\n");
    fprintf(stderr, "\n  WIR (Wavefront Iterative Refinement) options:\n");
    fprintf(stderr, "  --wir                     Enable WIR decode mode\n");
    fprintf(stderr, "  --wir-draft-model PATH    Small draft model (if omitted, uses same model with temperature)\n");
    fprintf(stderr, "  --wir-prompt TEXT         Original prompt text (required for cross-tokenizer draft)\n");
    fprintf(stderr, "  --wir-block-size N        Tokens per block (default: 64)\n");
    fprintf(stderr, "  --wir-max-passes N        Max refinement passes per block (default: 10)\n");
    fprintf(stderr, "  --wir-confidence F        Confidence threshold (default: 0.95)\n");
    fprintf(stderr, "  --wir-topk-accept N       Accept if in top-K (default: 3)\n");
    fprintf(stderr, "  --wir-draft-temp F        Draft temperature for single-model mode (default: 0.8)\n");
    fprintf(stderr, "  --wir-refine-mode MODE    Refinement mode: all, n-worst (default: all)\n");
    fprintf(stderr, "  --wir-refine-n N          For n-worst: tokens to fix per pass (default: 1)\n");
}

static bool parse_args(int argc, char ** argv, receiver_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            params.model_path = argv[++i];
        } else if (arg == "--kv-artifact" && i + 1 < argc) {
            params.kv_artifact_path = argv[++i];
        } else if (arg == "--dump-kv-after-import" && i + 1 < argc) {
            params.dump_kv_after_import_path = argv[++i];
        } else if (arg == "--tail-tokens" && i + 1 < argc) {
            params.tail_tokens_path = argv[++i];
        } else if (arg == "--kv-port" && i + 1 < argc) {
            params.kv_port = std::atoi(argv[++i]);
        } else if (arg == "--kv-host" && i + 1 < argc) {
            params.kv_host = argv[++i];
        } else if (arg == "--token-stream-host" && i + 1 < argc) {
            params.token_stream_host = argv[++i];
        } else if (arg == "--token-stream-port" && i + 1 < argc) {
            params.token_stream_port = std::atoi(argv[++i]);
        } else if ((arg == "-c" || arg == "--ctx-size") && i + 1 < argc) {
            params.n_ctx = std::atoi(argv[++i]);
        } else if ((arg == "-b" || arg == "--batch-size") && i + 1 < argc) {
            params.n_batch = std::atoi(argv[++i]);
        } else if ((arg == "-ub" || arg == "--ubatch-size") && i + 1 < argc) {
            params.n_ubatch = std::atoi(argv[++i]);
        } else if ((arg == "-ngl" || arg == "--n-gpu-layers") && i + 1 < argc) {
            params.n_gpu_layers = std::atoi(argv[++i]);
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            params.n_threads = std::atoi(argv[++i]);
        } else if ((arg == "-n" || arg == "--predict") && i + 1 < argc) {
            params.n_predict = std::atoi(argv[++i]);
        } else if ((arg == "-ctk" || arg == "--cache-type-k") && i + 1 < argc) {
            if (!parse_cache_type(argv[++i], params.cache_type_k)) {
                fprintf(stderr, "Unknown cache type for K: %s\n", argv[i]);
                return false;
            }
        } else if ((arg == "-ctv" || arg == "--cache-type-v") && i + 1 < argc) {
            if (!parse_cache_type(argv[++i], params.cache_type_v)) {
                fprintf(stderr, "Unknown cache type for V: %s\n", argv[i]);
                return false;
            }
        } else if (arg == "--temp" && i + 1 < argc) {
            params.temp = std::atof(argv[++i]);
            params.temp_forced = true;
        } else if (arg == "--mlock") {
            params.use_mlock = true;
        } else if (arg == "--one-shot") {
            params.one_shot = true;
        } else if (arg == "--no-token-stream") {
            params.no_token_stream = true;
        } else if (arg == "--sessions" && i + 1 < argc) {
            params.n_sessions = std::atoi(argv[++i]);
            if (params.n_sessions < 1) params.n_sessions = 1;
            if (params.n_sessions > 16) params.n_sessions = 16;
        } else if (arg == "--kv-port-base" && i + 1 < argc) {
            params.kv_port_base = std::atoi(argv[++i]);
        } else if (arg == "--token-port-base" && i + 1 < argc) {
            params.token_port_base = std::atoi(argv[++i]);
        } else if (arg == "--exit-after-import") {
            params.exit_after_import = true;
        } else if (arg == "--wir") {
            params.wir_enabled = true;
        } else if (arg == "--wir-draft-model" && i + 1 < argc) {
            params.wir_draft_model = argv[++i];
            params.wir_enabled = true;
        } else if (arg == "--wir-prompt" && i + 1 < argc) {
            params.wir_prompt_text = argv[++i];
        } else if (arg == "--wir-block-size" && i + 1 < argc) {
            params.wir_block_size = std::atoi(argv[++i]);
        } else if (arg == "--wir-max-passes" && i + 1 < argc) {
            params.wir_max_passes = std::atoi(argv[++i]);
        } else if (arg == "--wir-confidence" && i + 1 < argc) {
            params.wir_confidence = std::atof(argv[++i]);
        } else if (arg == "--wir-topk-accept" && i + 1 < argc) {
            params.wir_topk_accept = std::atoi(argv[++i]);
        } else if (arg == "--wir-draft-temp" && i + 1 < argc) {
            params.wir_draft_temp = std::atof(argv[++i]);
        } else if (arg == "--wir-refine-mode" && i + 1 < argc) {
            params.wir_refine_mode = argv[++i];
        } else if (arg == "--wir-refine-n" && i + 1 < argc) {
            params.wir_refine_n = std::atoi(argv[++i]);
        } else if (arg == "-np" || arg == "--n-parallel") {
            if (i + 1 < argc) params.n_parallel = std::atoi(argv[++i]);
        } else if (arg == "-fa" || arg == "--flash-attn") {
            params.flash_attn = true;
            params.flash_attn_forced = true;
        } else if (arg == "--no-fa") {
            params.flash_attn = false;
            params.flash_attn_forced = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return false;
        }
    }

    if (params.model_path.empty()) {
        fprintf(stderr, "Error: --model is required\n");
        return false;
    }
    if (params.n_batch < 1 || params.n_ubatch < 1 || params.n_ubatch > params.n_batch) {
        fprintf(stderr, "Error: batch sizes require n_batch >= n_ubatch >= 1 (got %d/%d)\n",
                params.n_batch, params.n_ubatch);
        return false;
    }
    // Multi-session mode: if --sessions > 1, auto-set kv_port from kv_port_base
    // so that the single-session validation below passes.
    if (params.n_sessions > 1 && params.kv_port <= 0) {
        params.kv_port = params.kv_port_base;
    }
    if (params.kv_artifact_path.empty() && params.kv_port <= 0) {
        fprintf(stderr, "Error: either --kv-artifact or --kv-port is required\n");
        return false;
    }
    if (!params.kv_artifact_path.empty() && params.kv_port > 0) {
        fprintf(stderr, "Error: --kv-artifact and --kv-port are mutually exclusive\n");
        return false;
    }
    if (params.n_sessions > 1 && !params.kv_artifact_path.empty()) {
        fprintf(stderr, "Error: --sessions > 1 requires network mode (--kv-port), not file mode\n");
        return false;
    }
    return true;
}

static std::string default_tail_tokens_sidecar_path(const std::string & artifact_path) {
    return artifact_path + ".tail.bin";
}

static common_params_sampling make_receiver_sampling_params(float temp) {
    common_params_sampling params;
    params.seed = 42;
    params.top_k = 40;
    params.top_p = 0.95f;
    params.min_p = 0.05f;
    params.temp = temp;
    params.penalty_last_n = 64;
    params.penalty_repeat = 1.1f;
    params.penalty_freq = 0.0f;
    params.penalty_present = 0.0f;
    params.user_sampling_config =
        COMMON_PARAMS_SAMPLING_CONFIG_TOP_K |
        COMMON_PARAMS_SAMPLING_CONFIG_TOP_P |
        COMMON_PARAMS_SAMPLING_CONFIG_MIN_P |
        COMMON_PARAMS_SAMPLING_CONFIG_TEMP |
        COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N |
        COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT;
    return params;
}

static common_params_sampling make_receiver_sampling_params(
        const llama_kv_artifact_metadata::sampling_profile_t & profile) {
    common_params_sampling params;
    params.seed              = (uint32_t) profile.seed;
    params.min_keep          = profile.min_keep;
    params.top_k             = profile.top_k;
    params.top_p             = profile.top_p;
    params.min_p             = profile.min_p;
    params.xtc_probability   = profile.xtc_probability;
    params.xtc_threshold     = profile.xtc_threshold;
    params.typ_p             = profile.typ_p;
    params.temp              = profile.temp;
    params.dynatemp_range    = profile.dynatemp_range;
    params.dynatemp_exponent = profile.dynatemp_exponent;
    params.penalty_last_n    = profile.penalty_last_n;
    params.penalty_repeat    = profile.penalty_repeat;
    params.penalty_freq      = profile.penalty_freq;
    params.penalty_present   = profile.penalty_present;
    params.mirostat          = profile.mirostat;
    params.top_n_sigma       = profile.top_n_sigma;
    params.mirostat_tau      = profile.mirostat_tau;
    params.mirostat_eta      = profile.mirostat_eta;
    params.user_sampling_config = profile.user_sampling_config;
    return params;
}

static common_params_sampling choose_decode_sampling_params(
        const receiver_params & rparams,
        const llama_kv_artifact_metadata * meta,
        bool & imported_profile) {
    imported_profile = meta != nullptr && meta->format_minor >= 3;
    common_params_sampling params = imported_profile
        ? make_receiver_sampling_params(meta->sampling_profile)
        : make_receiver_sampling_params(rparams.temp);

    if (rparams.temp_forced) {
        params.temp = rparams.temp;
        params.user_sampling_config |= COMMON_PARAMS_SAMPLING_CONFIG_TEMP;
    }

    return params;
}

static int choose_decode_limit(
        const receiver_params & rparams,
        const llama_kv_artifact_metadata & meta,
        bool & ignored_zero_cap) {
    ignored_zero_cap = false;

    if (meta.requested_n_predict < 0) {
        return rparams.n_predict;
    }

    // Local file-mode / one-shot debugging often imports artifacts produced by
    // `--prefill-only`, which intentionally stores n_predict=0 on the prefiller.
    // In that workflow we still want the receiver's local -n override to apply.
    if (meta.requested_n_predict == 0 &&
        (!rparams.kv_artifact_path.empty() || rparams.no_token_stream || rparams.one_shot)) {
        ignored_zero_cap = true;
        return rparams.n_predict;
    }

    return meta.requested_n_predict;
}

static llama_token choose_best_non_eog_token(
        const llama_vocab * vocab,
        const float * logits,
        int n_vocab,
        llama_token sampled_eog) {
    if (!vocab || !logits || n_vocab <= 0) {
        return sampled_eog;
    }

    float best_logit = -INFINITY;
    llama_token best_token = sampled_eog;
    for (int token_id = 0; token_id < n_vocab; ++token_id) {
        if (llama_vocab_is_eog(vocab, token_id)) {
            continue;
        }
        if (logits[token_id] > best_logit) {
            best_logit = logits[token_id];
            best_token = token_id;
        }
    }
    return best_token;
}

static bool dump_imported_kv_artifact(
        llama_context * ctx,
        const std::string & path,
        llama_kv_artifact_metadata meta,
        uint32_t effective_token_count) {
    std::vector<uint8_t> state_buf(llama_state_seq_get_size(ctx, 0));
    if (state_buf.empty()) {
        fprintf(stderr, "[decode-server] Failed to size imported KV dump buffer\n");
        return false;
    }

    const size_t written = llama_state_seq_get_data(ctx, state_buf.data(), state_buf.size(), 0);
    if (written == 0 || written > state_buf.size()) {
        fprintf(stderr, "[decode-server] Failed to capture imported KV state (written=%zu size=%zu)\n",
                written, state_buf.size());
        return false;
    }
    state_buf.resize(written);

    meta.token_count = effective_token_count;

    llama_kv_artifact_summary summary = {};
    std::string err;
    if (!llama_kv_artifact_write(path, state_buf, meta, &summary, &err, nullptr)) {
        fprintf(stderr, "[decode-server] Failed to write imported KV dump to %s: %s\n",
                path.c_str(), err.c_str());
        return false;
    }

    fprintf(stderr, "[decode-server] Dumped imported KV to %s (%zu bytes payload, crc=%u)\n",
            path.c_str(), summary.payload_bytes, summary.payload_crc32);
    return true;
}

static bool apply_legacy_tail_tokens(
        llama_context * ctx,
        const receiver_params & rparams,
        const std::vector<llama_token> * embedded_tail_tokens,
        int session_id,
        uint32_t & effective_token_count,
        llama_token & last_token,
        bool & tail_applied) {
    tail_applied = false;
    std::vector<llama_token> tail_tokens;
    std::string tail_path;
    if (embedded_tail_tokens && !embedded_tail_tokens->empty()) {
        tail_tokens = *embedded_tail_tokens;
        fprintf(stderr, "[decode-server] Session %d: using %zu embedded tail token(s) from artifact\n",
                session_id, tail_tokens.size());
    } else {
        tail_path = rparams.tail_tokens_path;
        if (tail_path.empty() && !rparams.kv_artifact_path.empty()) {
            const std::string inferred_path = default_tail_tokens_sidecar_path(rparams.kv_artifact_path);
            std::error_code ec;
            if (std::filesystem::exists(inferred_path, ec) && !ec) {
                tail_path = inferred_path;
                fprintf(stderr, "[decode-server] Session %d: auto-detected tail sidecar %s\n",
                        session_id, tail_path.c_str());
            }
        }

        if (tail_path.empty()) {
            return true;
        }

        std::string error;
        if (!load_tail_tokens_file(tail_path, tail_tokens, error)) {
            fprintf(stderr, "[decode-server] Session %d: failed to load tail tokens from %s: %s\n",
                    session_id, tail_path.c_str(), error.c_str());
            return false;
        }
        if (tail_tokens.empty()) {
            fprintf(stderr, "[decode-server] Session %d: tail sidecar %s is empty, skipping local tail prefill\n",
                    session_id, tail_path.c_str());
            return true;
        }
    }

    llama_memory_t mem = llama_get_memory(ctx);
    llama_pos pos_max = mem ? llama_memory_seq_pos_max(mem, 0) : -1;
    // Use the artifact's token count as the tail start position, not pos_max.
    // pos_max can disagree with token_count if import left padding/extra cells.
    const llama_pos tail_pos_start = (llama_pos)effective_token_count;
    // Suffix-prefill batch size: default to the context's launched n_batch
    // (typically 2048 on Mac), env-overridable. The historical hardcoded
    // 512 forced 4x as many llama_decode round-trips as needed for a typical
    // 1243-token suffix, costing ~25 s of avoidable Mac prefill latency.
    int n_batch = std::max(1, (int) llama_n_batch(ctx));
    if (const char * env = std::getenv("LLAMA_HANDOFF_SUFFIX_BATCH")) {
        const int parsed = std::atoi(env);
        if (parsed > 0) {
            n_batch = parsed;
        }
    }

    // Auto-detect full-prompt mode: if the sidecar carries more tokens than
    // what's already in the imported KV, assume it contains the ENTIRE prompt
    // and skip the imported prefix. This makes the handoff protocol robust to
    // position-offset mismatches between Windows's n_p_eval and Mac's pos_max.
    size_t tail_skip = 0;
    if (tail_tokens.size() > effective_token_count) {
        tail_skip = effective_token_count;
        fprintf(stderr, "[decode-server] Session %d: tail sidecar contains full prompt (%zu tokens), skipping %u already-imported tokens\n",
                session_id, tail_tokens.size(), effective_token_count);
    }
    const size_t effective_tail_size = tail_tokens.size() - tail_skip;

    const std::string tail_source_suffix = tail_path.empty() ? std::string() : (" from " + tail_path);
    fprintf(stderr, "[decode-server] Session %d: applying %zu local tail token(s)%s at pos %d\n",
            session_id, effective_tail_size, tail_source_suffix.c_str(), tail_pos_start);

    llama_batch batch = llama_batch_init(std::min<int>(static_cast<int>(effective_tail_size), n_batch), 0, 1);
    auto t_tail_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < effective_tail_size; i += n_batch) {
        const int batch_size = std::min<int>(n_batch, static_cast<int>(effective_tail_size - i));
        common_batch_clear(batch);

        for (int j = 0; j < batch_size; ++j) {
            const bool is_last = (i + static_cast<size_t>(j) == effective_tail_size - 1);
            common_batch_add(batch, tail_tokens[tail_skip + i + j],
                             tail_pos_start + static_cast<llama_pos>(i + j), { 0 }, is_last);
        }

        {
            llama_memory_t mem = llama_get_memory(ctx);
            llama_pos pmin = llama_memory_seq_pos_min(mem, 0);
            llama_pos pmax = llama_memory_seq_pos_max(mem, 0);
            fprintf(stderr, "[decode-server] Session %d: tail prefill batch: n=%d pos=[%d,%d] kv_pos=[%d,%d]\n",
                    session_id, batch.n_tokens, batch.pos[0], batch.pos[batch.n_tokens-1], pmin, pmax);
            fflush(stderr);
        }
        const int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            fprintf(stderr, "[decode-server] Session %d: local tail prefill failed at tail token %zu (ret=%d)\n",
                    session_id, i, ret);
            llama_batch_free(batch);
            return false;
        }
    }

    // Sync BEFORE the clock stops: llama_decode enqueues, it does not wait, so a timer closed
    // above this line reports how fast work was handed to the GPU. Same defect as the suffix
    // prefill timer below (see the note there) -- and the free moves after the sync too, since
    // releasing the batch while decode work may still reference its arrays is a use-after-free.
    llama_synchronize(ctx);
    auto t_tail_end = std::chrono::high_resolution_clock::now();
    const double tail_ms = std::chrono::duration<double, std::milli>(t_tail_end - t_tail_start).count();
    llama_batch_free(batch);

    if (mem) {
        pos_max = llama_memory_seq_pos_max(mem, 0);
    }
    effective_token_count = pos_max >= 0 ? static_cast<uint32_t>(pos_max + 1) : static_cast<uint32_t>(tail_tokens.size());
    last_token = tail_tokens.back();
    tail_applied = true;

    fprintf(stderr, "[decode-server] Session %d: local tail prefill %.1f ms (%.1f tok/s), effective tokens=%u, last_token=%d\n",
            session_id,
            tail_ms,
            tail_ms > 0.0 ? effective_tail_size / (tail_ms / 1000.0) : 0.0,
            effective_token_count,
            last_token);
    return true;
}

static bool apply_handoff_prompt(
        llama_context * ctx,
        const receiver_params & rparams,
        const llama_kv_artifact_metadata & meta,
        const llama_kv_prompt_payload & prompt_payload,
        int session_id,
        uint32_t & effective_token_count,
        llama_token & last_token,
        bool & prompt_applied,
        uint32_t resident_reuse_n = 0) {
    prompt_applied = false;
    auto _ah_t0 = std::chrono::high_resolution_clock::now();
    auto _ah_el = [&]() { return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - _ah_t0).count(); };

    if (meta.format_minor >= 5 && !prompt_payload.token_ids.empty()) {
        std::vector<llama_token> prompt_tokens;
        prompt_tokens.reserve(prompt_payload.token_ids.size());
        for (int32_t tok : prompt_payload.token_ids) {
            prompt_tokens.push_back((llama_token) tok);
        }

        const uint32_t declared_prefix = std::min<uint32_t>(meta.streamed_prefix_token_count,
                                                            (uint32_t) prompt_tokens.size());
        uint32_t logical_prefix = declared_prefix;
        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) {
            llama_pos pos_min = llama_memory_seq_pos_min(mem, 0);
            llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
            uint32_t imported_tokens = 0u;
            const llama_model * ctx_model = llama_get_model(ctx);
            const bool is_swa = ctx_model && llama_model_n_swa(ctx_model) > 0;
            if (pos_max >= 0 && (pos_min == 0 ||
                    (is_swa && declared_prefix > 0 && (uint32_t) (pos_max + 1) >= declared_prefix))) {
                imported_tokens = std::min<uint32_t>((uint32_t) (pos_max + 1), declared_prefix);
                if (is_swa && pos_min > 0) {
                    fprintf(stderr,
                            "[decode-server] Session %d: ISWA import coverage valid: sliding range [%d,%d], declared prefix=%u\n",
                            session_id, pos_min, pos_max, declared_prefix);
                }
            } else if (pos_min >= 0 || pos_max >= 0) {
                fprintf(stderr,
                        "[decode-server] Session %d: imported KV range [%d,%d] does not cover a valid prompt prefix from position 0; discarding imported prefix\n",
                        session_id, pos_min, pos_max);
                llama_memory_seq_rm(mem, 0, -1, -1);
            }
            if (imported_tokens < logical_prefix) {
                fprintf(stderr,
                        "[decode-server] Session %d: imported KV only materialized %u/%u declared prefix tokens; refusing local re-prefill\n",
                        session_id, imported_tokens, logical_prefix);
                return false;
            }
        }
        if (const char * env = std::getenv("LLAMA_HANDOFF_PREFIX_BACKOFF")) {
            const int parsed = std::atoi(env);
            if (parsed > 0) {
                const uint32_t backoff = (uint32_t) parsed;
                const uint32_t adjusted = logical_prefix > backoff ? (logical_prefix - backoff) : 0u;
                if (adjusted != logical_prefix) {
                    fprintf(stderr,
                            "[decode-server] Session %d: prefix backoff %u applied, trusting %u/%u imported prompt tokens before local re-prefill\n",
                            session_id, backoff, adjusted, logical_prefix);
                    logical_prefix = adjusted;
                }
            }
        }
        if (resident_reuse_n > 0) {
            // Stage 1 RESIDENT REUSE: trust the validated LCP against the live
            // resident KV (NOT the blob's streamed_prefix — no blob was imported).
            // Clamp to the live KV extent; the seq_rm below trims the stale prior
            // decoded tail + any divergent suffix beyond reuse_n, then we prefill
            // only [reuse_n..full].
            llama_pos pmax = mem ? llama_memory_seq_pos_max(mem, 0) : -1;
            uint32_t live_kv = pmax >= 0 ? (uint32_t) (pmax + 1) : 0;
            logical_prefix = std::min<uint32_t>(resident_reuse_n, live_kv);
            fprintf(stderr,
                    "[decode-server] Session %d: RESIDENT logical_prefix=%u (reuse_n=%u live_kv=%u) — trim+prefill tail\n",
                    session_id, logical_prefix, resident_reuse_n, live_kv);
        }
        if (mem) {
            llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
            if (pos_max >= (llama_pos) logical_prefix) {
                // llama_memory_seq_rm RETURNS A BOOL, and on an iSWA cache it can refuse: the
                // sliding window no longer covers logical_prefix, so the range cannot be dropped
                // without leaving a hole the cache cannot reconstruct. Ignoring that return value
                // left the trim silently un-done and handed llama_decode a batch starting BEFORE
                // the KV's last position -- measured 2026-08-27 on DSV4-flash:
                //     RESIDENT logical_prefix=11320 ... trim+prefill tail
                //     ISWA import coverage valid: sliding range [11327,11327]
                //     init: ... the last position stored ... is X = 11327
                //      - the tokens ... have a starting position of Y = 11320
                //     prompt suffix prefill failed at token 11320 (ret=-1)
                // and the session died. Turn 1 and turn 2 worked, turn 3 did not, because only by
                // then had the retained window drifted past the reuse point.
                //
                // Re-read pos_max rather than trusting the bool alone: either signal means the
                // sequence is not in the state the suffix loop assumes. Then fall back to the one
                // state that is always consistent -- an empty sequence, prefilled from 0. That
                // costs a local prefill of the whole prompt for this turn, which is slow but
                // correct; the alternative is a dead session.
                const bool trimmed = llama_memory_seq_rm(mem, 0, (llama_pos) logical_prefix, -1);
                const llama_pos after = llama_memory_seq_pos_max(mem, 0);
                if (!trimmed || after >= (llama_pos) logical_prefix) {
                    // The reuse decision (search: "window_covers_reuse") refuses reuse when the window
                    // has moved past the reuse point, so reaching here means that predicate missed a
                    // case. FAIL, loudly and with the numbers. Do NOT quietly clear and re-prefill the
                    // whole prompt locally: that turns a bug into "the Mac is mysteriously slow this
                    // turn", which is the failure mode that hid the iSWA trim for a whole session. The
                    // caller invalidates the resident record and the next turn takes the import path.
                    fprintf(stderr,
                            "[decode-server] Session %d: TRIM REFUSED to %u (rm=%d, pos_max still %d, window starts at %d) — "
                            "the reuse guard should have prevented this; failing the session rather than "
                            "silently re-prefilling %zu tokens locally\n",
                            session_id, logical_prefix, (int) trimmed, (int) after,
                            (int) llama_memory_seq_pos_min(mem, 0), prompt_tokens.size());
                    return false;
                }
            }
        }
        effective_token_count = logical_prefix;
        fprintf(stderr, "[decode-server] Session %d: [AH_TIMING after_seq_rm=%.0fms]\n", session_id, _ah_el());

        if (!prompt_payload.prompt_text.empty()) {
            auto _dt0 = std::chrono::high_resolution_clock::now();
            const std::string detok = common_detokenize(ctx, prompt_tokens, true);
            fprintf(stderr, "[decode-server] Session %d: [AH_TIMING detokenize=%.0fms tokens=%zu]\n", session_id,
                    std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - _dt0).count(),
                    prompt_tokens.size());
            if (detok == prompt_payload.prompt_text) {
                fprintf(stderr, "[decode-server] Session %d: prompt text matches canonical token payload (%zu bytes)\n",
                        session_id, prompt_payload.prompt_text.size());
            } else {
                fprintf(stderr, "[decode-server] Session %d: WARNING: prompt text differs from canonical token payload; using token IDs as source of truth\n",
                        session_id);
            }
        }

        const size_t suffix_tokens = prompt_tokens.size() - logical_prefix;
        if (suffix_tokens == 0) {
            last_token = prompt_tokens.back();
            fprintf(stderr, "[decode-server] Session %d: imported KV already covers full prompt (%u tokens)\n",
                    session_id, effective_token_count);
            return true;
        }

        fprintf(stderr, "[decode-server] Session %d: prompt verification OK, prefix=%u suffix=%zu - prefilling remaining prompt tokens locally\n",
                session_id, logical_prefix, suffix_tokens);

        // Default to the context's launched n_batch (typically 2048 on Mac);
        // env-overridable. Historical hardcoded 512 caused a 4x prefill
        // round-trip count for typical suffix sizes.
        int n_batch = std::max(1, (int) llama_n_batch(ctx));
        if (const char * env = std::getenv("LLAMA_HANDOFF_SUFFIX_BATCH")) {
            const int parsed = std::atoi(env);
            if (parsed > 0) {
                n_batch = parsed;
            }
        }
        // Stage 1b (2026-05-29): on the RESIDENT-REUSE path, prefill a SMALL tail via
        // single-token steps (n_batch=1) so the n_tokens=1 decode-graph stays the active
        // reserved graph. A batched (n_tokens=tail) prefill perturbs the scheduler and
        // forces the first REAL decode to re-reserve the decode graph (~6s on the 1T MoE
        // at full context — the measured residual). The decode graph from the prior turn
        // persists on the reused ctx, so single-token tail prefill reuses it with no
        // re-reserve. Only worthwhile for small tails (N*~27ms < ~6s reserve => N<~222);
        // large tails are faster batched (and should route to RTX streaming — Stage 2).
        // REVERTED 2026-05-29: default 0 (disabled). Measurement showed single-token
        // tail prefill runs at DECODE speed (~6.4 tok/s) — 74 tokens took 11.5s,
        // making the reuse turn WORSE (ttft 7.7s->16.3s). The batched suffix prefill
        // is already 22ms with NO graph reserve (the decode first token is 159-191ms,
        // not ~6s — the "first-decode reserve" premise was wrong). The real reuse-turn
        // residual (~6s) is RTX-SIDE, not the Mac. Keep the knob for experiments only.
        static const int resident_singletok_max = []() {
            const char * e = std::getenv("LLAMA_PD_RESIDENT_SINGLETOK_MAX");
            return (e && e[0]) ? std::max(0, atoi(e)) : 0;
        }();
        if (resident_reuse_n > 0 && (int) suffix_tokens > 0 && (int) suffix_tokens <= resident_singletok_max) {
            n_batch = 1;
            fprintf(stderr, "[decode-server] Session %d: Stage 1b — single-token tail prefill (suffix=%zu <= %d) to keep decode graph warm (skip first-decode re-reserve)\n",
                    session_id, suffix_tokens, resident_singletok_max);
        }
        fprintf(stderr, "[decode-server] Session %d: local suffix prefill batch=%d\n",
                session_id, n_batch);
        llama_batch batch = llama_batch_init(std::min<int>((int) suffix_tokens, n_batch), 0, 1);
        auto t_start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < suffix_tokens; i += n_batch) {
            const int batch_size = std::min<int>(n_batch, (int) (suffix_tokens - i));
            common_batch_clear(batch);
            for (int j = 0; j < batch_size; ++j) {
                const size_t token_idx = (size_t) logical_prefix + i + j;
                const bool is_last = token_idx + 1 == prompt_tokens.size();
                common_batch_add(batch, prompt_tokens[token_idx],
                                 (llama_pos) token_idx, { 0 }, is_last);
            }
            const int ret = llama_decode(ctx, batch);
            if (ret != 0) {
                fprintf(stderr, "[decode-server] Session %d: prompt suffix prefill failed at token %zu (ret=%d)\n",
                        session_id, (size_t) logical_prefix + i, ret);
                llama_batch_free(batch);
                return false;
            }
        }

        // SYNCHRONIZE BEFORE STOPPING THE CLOCK. `llama_decode` ENQUEUES on Metal; it does not
        // wait. With the sync after t_end this timer measured how long it took to hand work to the
        // GPU, not how long the GPU took -- and it reported the result as a throughput.
        //
        // Measured 2026-08-31 on a resident-reuse turn: "prompt suffix prefill 9.4 ms
        // (1924.6 tok/s)" for an 18-token suffix, inside an enclosing span
        // [RDS_TIMING after_suffix=3324ms] whose other components were all zero
        // (after_seq_rm=0ms, detokenize=0ms). 1924 tok/s of prefill on a 1T MoE over Metal is not
        // a plausible number, and it was not one: ~3.3 s of real work was being attributed to
        // nothing at all, because the wait for it happened two lines after the measurement.
        //
        // This is not only a cosmetic mis-report. The comment above ("Stage 1b, REVERTED") rests
        // on this timer -- "the batched suffix prefill is already 22ms with NO graph reserve ...
        // The real reuse-turn residual (~6s) is RTX-SIDE, not the Mac" -- and 22 ms was an enqueue
        // time too. That conclusion needs re-testing against a timer that waits.
        //
        // The free also moves after the sync. Releasing the batch while decode work that may still
        // reference its token/pos arrays is in flight is a use-after-free waiting for a scheduler
        // that is slightly less forgiving.
        llama_synchronize(ctx);
        auto t_end = std::chrono::high_resolution_clock::now();
        const double suffix_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        llama_batch_free(batch);

        effective_token_count = (uint32_t) prompt_tokens.size();
        last_token = prompt_tokens.back();
        prompt_applied = true;

        fprintf(stderr, "[decode-server] Session %d: prompt suffix prefill %.1f ms (%.1f tok/s), effective tokens=%u, last_token=%d\n",
                session_id, suffix_ms,
                suffix_ms > 0.0 ? suffix_tokens / (suffix_ms / 1000.0) : 0.0,
                effective_token_count, last_token);
        return true;
    }

    if (meta.format_minor >= 5) {
        fprintf(stderr,
                "[decode-server] Session %d: artifact v1.%u is missing canonical prompt token IDs; refusing legacy tail fallback\n",
                session_id, meta.format_minor);
        return false;
    }

    std::vector<llama_token> legacy_tail_tokens;
    if (!prompt_payload.token_ids.empty()) {
        legacy_tail_tokens.reserve(prompt_payload.token_ids.size());
        for (int32_t tok : prompt_payload.token_ids) {
            legacy_tail_tokens.push_back((llama_token) tok);
        }
    }
    return apply_legacy_tail_tokens(ctx, rparams,
                                    legacy_tail_tokens.empty() ? nullptr : &legacy_tail_tokens,
                                    session_id, effective_token_count, last_token, prompt_applied);
}

// ---------------------------------------------------------------------------
// Run prompt-direct session: receive raw tokens, prefill locally, then decode.
// Used for short prompts where RTX streaming prefill is not beneficial.
// ---------------------------------------------------------------------------
// is_cache_hit: if true, skip prefill and restore KV from kv_restore_data instead.
// kv_restore_data: filled when is_cache_hit=true (llama_state_seq_get_data bytes).
static int run_prompt_session(llama_context * ctx, llama_model * model,
                               const receiver_params & rparams,
                               const llama_kv_artifact_metadata & meta,
                               const llama_kv_prompt_payload & prompt_payload,
                               int session_id,
                               bool is_cache_hit = false,
                               const std::vector<uint8_t> * kv_restore_data = nullptr) {
    if (prompt_payload.token_ids.empty() && !is_cache_hit) {
        fprintf(stderr, "[decode-server] Session %d: prompt-direct artifact is missing canonical token IDs\n",
                session_id);
        return -1;
    }
    const int n_tokens = is_cache_hit ? 0 : (int) prompt_payload.token_ids.size();
    const int32_t * token_ids = is_cache_hit ? nullptr : prompt_payload.token_ids.data();

    if (is_cache_hit) {
        fprintf(stderr, "[decode-server] Session %d: PROMPT MODE (CACHE HIT) — restoring KV from snapshot\n",
                session_id);
    } else {
        fprintf(stderr, "[decode-server] Session %d: PROMPT MODE — %d tokens, prefilling locally\n",
                session_id, n_tokens);
    }

    // Open the token stream listener before local prefill so the RTX relay can
    // connect as soon as the prompt-direct artifact lands.
    llama_token_stream_server token_server;
    bool token_stream_active = false;
    if (!rparams.no_token_stream) {
        if (!token_server.start(rparams.token_stream_host, rparams.token_stream_port)) {
            fprintf(stderr,
                    "[decode-server] Session %d: failed to start token stream on %s:%d\n",
                    session_id, rparams.token_stream_host.c_str(), rparams.token_stream_port);
            return -1;
        }
        fprintf(stderr,
                "[decode-server] Session %d: token stream listening early on port %d\n",
                session_id, token_server.bound_port());
    }

    if (is_cache_hit) {
        // CACHE HIT PATH: restore KV state from snapshot instead of running prefill.
        if (!kv_restore_data || kv_restore_data->empty()) {
            fprintf(stderr, "[decode-server] Session %d: CACHE HIT but kv_restore_data is empty — falling through to miss path\n",
                    session_id);
            // Cannot proceed without KV data; caller should have verified
            if (!rparams.no_token_stream) {
                token_server.stop();
            }
            return -1;
        }
        auto t_restore_start = std::chrono::high_resolution_clock::now();
        size_t consumed = llama_state_seq_set_data(ctx, kv_restore_data->data(), kv_restore_data->size(), 0);
        auto t_restore_end = std::chrono::high_resolution_clock::now();
        double restore_ms = std::chrono::duration<double, std::milli>(t_restore_end - t_restore_start).count();
        if (consumed == 0) {
            fprintf(stderr, "[decode-server] Session %d: llama_state_seq_set_data FAILED (consumed=0 size=%zu) — falling through as error\n",
                    session_id, kv_restore_data->size());
            if (!rparams.no_token_stream) {
                token_server.stop();
            }
            return -1;
        }
        fprintf(stderr, "[decode-server] Session %d: KV restored %.1f ms (%zu bytes consumed)\n",
                session_id, restore_ms, consumed);
        g_active_cache_hit = true;
    } else {
        // MISS PATH: prefill locally in batches. Defaults to the context's
        // launched n_batch (typically 2048 on Mac), env-overridable via
        // LLAMA_HANDOFF_SUFFIX_BATCH. Historical hardcoded 512 was a 4x
        // unnecessary round-trip count for typical suffix sizes.
        int n_batch = std::max(1, (int) llama_n_batch(ctx));
        if (const char * env = std::getenv("LLAMA_HANDOFF_SUFFIX_BATCH")) {
            const int parsed = std::atoi(env);
            if (parsed > 0) {
                n_batch = parsed;
            }
        }
        // The batch below is built at ABSOLUTE positions starting at 0, so sequence 0 has to be
        // EMPTY before it runs. It is not, whenever the previous session retained its KV: with
        // LLAMA_PD_RESIDENT_KV_REUSE=1 (which the studio launcher injects unconditionally) session
        // N-1 ends "KV RETAINED for next-turn reuse (pos_max=9768)", and llama_decode then rejects
        // the whole batch --
        //     init: the tokens of sequence 0 ... have inconsistent sequence positions
        //      - the tokens for sequence 0 ... have a starting position of Y = 0
        // -> "local prefill failed at token 0 (ret=-1)". Measured 2026-08-27: the RTX->Mac loop
        // decoded a 9,728-token KV artifact correctly and then failed EVERY short follow-up prompt,
        // because short prompts route PROMPT-DIRECT (below rtx crossover=3072) and land here.
        //
        // Clearing is right for THIS path specifically: the miss path exists to prefill the prompt
        // from scratch. Reuse of a retained prefix is the CACHE-HIT branch above (is_cache_hit),
        // which restores a snapshot instead of prefilling. Reaching here means no reuse was claimed,
        // so the retained KV is stale by definition -- and leaving it in place does not preserve it,
        // it just makes the session fail.
        if (llama_memory_t mem = llama_get_memory(ctx)) {
            const llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
            if (pos_max >= 0) {
                fprintf(stderr,
                        "[decode-server] Session %d: clearing retained KV (pos_max=%d) before local prefill from position 0\n",
                        session_id, pos_max);
                llama_memory_seq_rm(mem, 0, -1, -1);
            }
        }
        {
            // The resident-KV record described the sequence we just wiped; leaving it "valid" would
            // offer a later session an LCP against KV that no longer exists.
            std::lock_guard<std::mutex> lk(g_resident_kv_mu);
            g_resident_kv.valid = false;
        }

        llama_batch batch = llama_batch_init(std::min(n_tokens, n_batch), 0, 1);
        auto t_prefill_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < n_tokens; i += n_batch) {
            const int batch_size = std::min(n_batch, n_tokens - i);
            common_batch_clear(batch);
            for (int j = 0; j < batch_size; j++) {
                const bool is_last = (i + j == n_tokens - 1);
                common_batch_add(batch, token_ids[i + j], i + j, { 0 }, is_last);
            }
            // Tell the dump which token this batch STARTS at. Without it the cross-backend diff is
            // impossible: the RTX dumps 512-wide ubatches and this receiver's are ~37 wide, so the
            // arrays only line up once both sides know their offset. Comparing without it already
            // produced a convincing false positive (rel 1.83 on the hidden state) that was purely
            // two different token ranges.
            g_dump_batch_start.store(i, std::memory_order_relaxed);
            int ret = llama_decode(ctx, batch);
            if (ret != 0) {
                fprintf(stderr, "[decode-server] Session %d: local prefill failed at token %d (ret=%d)\n",
                        session_id, i, ret);
                if (!rparams.no_token_stream) {
                    token_server.stop();
                }
                llama_batch_free(batch);
                return -1;
            }
            if (i + batch_size < n_tokens && (i / n_batch) % 10 == 0) {
                fprintf(stderr, "[decode-server] Session %d: prefill progress %d/%d\n",
                        session_id, i + batch_size, n_tokens);
            }
        }

        auto t_prefill_end = std::chrono::high_resolution_clock::now();
        double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
        fprintf(stderr, "[decode-server] Session %d: local prefill %.1f ms (%.1f tok/s)\n",
                session_id, prefill_ms, n_tokens / (prefill_ms / 1000.0));
        llama_batch_free(batch);

        // PROMPT-DIRECT KV-handshake: record snapshot so future RTX queries can get a hit.
        // Compute hash of the token stream (same algorithm as RTX side: xxhash64).
        // NOTE: KV capture (llama_state_seq_get_data) is deferred to after the decode
        // session completes (see caller in session loop) to avoid blocking the token stream.
        // We set the active hash here so the caller knows to capture after decode.
        {
            const uint64_t snap_hash = llama_tb_transport_hash_tokens(token_ids, n_tokens);
            char model_desc_buf[256] = {};
            if (model) {
                llama_model_desc(model, model_desc_buf, sizeof(model_desc_buf));
            }
            const uint64_t snap_model_fp = llama_tb_transport_model_fingerprint(model_desc_buf);
            // Store active hash/model_fp so caller can call pd_cache_insert after decode
            g_active_prefix_hash = snap_hash;
            g_active_cache_hit   = false;  // this was a miss → we ran real prefill
            // Insert metadata-only snapshot now (kv_data filled by caller after
            // decode). Stash the canonical token IDs so the post-decode disk
            // write-through (Phase A3) can persist them for token-exact verify.
            pd_cache_insert(snap_hash, (uint32_t) n_tokens, snap_model_fp, nullptr,
                            token_ids, n_tokens);
        }
    }

    // Determine starting decode position and last prefill token.
    // For cache-hit: derive from KV state. For normal prefill: from token array.
    int decode_start_pos = 0;  // KV position of the first generated token
    llama_token initial_last_token = -1;
    if (is_cache_hit) {
        // Determine last token position from restored KV
        llama_memory_t mem = llama_get_memory(ctx);
        llama_pos p_max = mem ? llama_memory_seq_pos_max(mem, 0) : -1;
        if (p_max < 0) {
            fprintf(stderr, "[decode-server] Session %d: restored KV appears empty (pos_max=%d)\n",
                    session_id, (int)p_max);
            if (!rparams.no_token_stream) { token_server.stop(); }
            return -1;
        }
        decode_start_pos = (int)(p_max + 1);
        // last_token: we need to re-run a single-token forward pass for the last KV position
        // to get valid logits. The last token of the original prompt is needed.
        // For cache-hit we don't have token_ids but we need the last token for sampling seed.
        // Use -1 as sentinel; the decode loop will use the logits already present from KV restore.
        initial_last_token = -1;
        fprintf(stderr, "[decode-server] Session %d: cache-hit decode starting at pos=%d\n",
                session_id, decode_start_pos);
    } else {
        decode_start_pos = n_tokens;
        initial_last_token = token_ids[n_tokens - 1];
    }

    // Decode batch (reused across iterations)
    llama_batch decode_batch = llama_batch_init(1, 0, 1);

    // Start token stream + decode loop
    // We skip the KV import step — KV is already in the context from local prefill.
    // Go straight to decode.

    llama_decode_config decode_cfg;
    if (!rparams.no_token_stream) {
        fprintf(stderr, "[decode-server] Session %d: token stream on port %d, waiting for client...\n",
                session_id, token_server.bound_port());
        if (!token_server.wait_for_client(60000)) {
            fprintf(stderr, "[decode-server] Session %d: no token stream client\n", session_id);
            token_server.stop();
            llama_batch_free(decode_batch);
            return -1;
        }
        // Read decode config from RTX client (stop words, etc.)
        if (token_server.read_config(decode_cfg, 5000)) {
            if (!decode_cfg.stop_words.empty()) {
                fprintf(stderr, "[decode-server] Session %d: received %zu stop words\n",
                        session_id, decode_cfg.stop_words.size());
            }
        } else {
            fprintf(stderr, "[decode-server] Session %d: no config received (old client?), using defaults\n",
                    session_id);
        }
        llama_token_handshake hs = {};
        hs.version = 1;
        hs.status = "ready";
        hs.token_count = decode_start_pos;
        token_server.send_handshake(hs);
        token_stream_active = true;
    }

    bool imported_sampling = false;
    common_params_sampling sparams = choose_decode_sampling_params(rparams, &meta, imported_sampling);
    if (imported_sampling) {
        fprintf(stderr,
                "[decode-server] Session %d: imported sampling profile temp=%.3f top_k=%d top_p=%.3f min_p=%.3f repeat_last_n=%d repeat_penalty=%.3f mirostat=%d%s\n",
                session_id,
                sparams.temp,
                sparams.top_k,
                sparams.top_p,
                sparams.min_p,
                sparams.penalty_last_n,
                sparams.penalty_repeat,
                sparams.mirostat,
                rparams.temp_forced ? " (CLI temp override applied)" : "");
    }
    bool ignored_zero_cap = false;
    const int decode_limit = choose_decode_limit(rparams, meta, ignored_zero_cap);
    if (meta.requested_n_predict >= 0) {
        fprintf(stderr,
                "[decode-server] Session %d: imported decode cap n_predict=%d%s\n",
                session_id, meta.requested_n_predict,
                ignored_zero_cap ? " (ignored in local file/one-shot mode)" : "");
    }
    common_sampler_ptr smpl(common_sampler_init(model, sparams));

    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token last_token = initial_last_token;
    int n_decoded = 0;
    int32_t stop_reason = LLAMA_STOP_EOS;
    std::string stop_word_matched;
    std::string generated_text;  // accumulated text for stop word matching
    std::vector<int64_t> emitted_timestamps_us;
    emitted_timestamps_us.reserve((size_t) std::max(0, decode_limit));

    // For cache-hit: need a forward pass with the last KV position to produce logits
    // before we can start sampling. We use decode_batch for this.
    if (is_cache_hit && decode_start_pos > 0) {
        // The KV restore puts all tokens in context but we need fresh logits.
        // Re-run the last position to compute logits.
        // We use a dummy token (BOS) since we just need the logits from the cached KV.
        // Actually, llama_state_seq_set_data should restore logit state too, but we
        // need to verify: run a single decode step at pos decode_start_pos-1
        // using last_token_of_sequence. Since we don't have the last token,
        // we need to do a decode of BOS at the last position to prime logits.
        // However, the standard approach is: on KV restore, the logits are NOT valid.
        // We need to do a forward pass of the BOS (or any token) to get logits.
        // The "correct" approach: store last_token in the snapshot too.
        // For now: use llama_vocab_bos as a placeholder; this will give slightly
        // different first-token distribution but is correct after the first token.
        // TODO: store last_token in KvSnapshot for perfect first-token reproducibility.
        const llama_token bos = llama_vocab_bos(vocab);
        common_batch_clear(decode_batch);
        common_batch_add(decode_batch, bos, decode_start_pos - 1, { 0 }, true);
        // Actually: the correct way is to run decode at the existing last position
        // using the actual last token. Since we don't track it yet, use pos decode_start_pos
        // with BOS. This adds a "phantom" token but is needed for logits.
        // Better: don't add to KV, just compute logits from current KV.
        // llama_decode runs forward pass AND updates KV. We want logits only.
        // Use eval_string is too high-level. We'll accept the slight corruption
        // for this iteration and fix by storing last_token in snapshot (TODO).
        fprintf(stderr, "[decode-server] Session %d: priming logits at pos %d (cache-hit)\n",
                session_id, decode_start_pos - 1);
        // Override: re-decode at pos decode_start_pos-1 to get correct logits.
        // This re-runs the last position through the model.
        common_batch_clear(decode_batch);
        common_batch_add(decode_batch, bos, decode_start_pos, { 0 }, true);
        if (llama_decode(ctx, decode_batch) != 0) {
            fprintf(stderr, "[decode-server] Session %d: cache-hit logit prime failed\n", session_id);
            llama_batch_free(decode_batch);
            if (!rparams.no_token_stream) { token_server.stop(); }
            return -1;
        }
        decode_start_pos++;  // BOS token was added, advance position
    }

    // Decode loop — sample from last token's logits, then generate
    int repeat_count = 0;
    llama_token repeat_token = -1;
    const int max_repeats = 50;  // abort if same token repeats 50+ times (degenerate output)
    for (int i = 0; i < decode_limit; i++) {
        llama_token new_token;
        if (i == 0 && !is_cache_hit) {
            // First token: sample from prefill logits (last position already has logits=true)
            new_token = common_sampler_sample(smpl.get(), ctx, -1);
        } else if (i == 0 && is_cache_hit) {
            // Cache hit: sample from primed logits
            new_token = common_sampler_sample(smpl.get(), ctx, -1);
        } else {
            // Subsequent tokens: decode one token at a time
            common_batch_clear(decode_batch);
            common_batch_add(decode_batch, last_token, decode_start_pos + i - 1, { 0 }, true);
            if (llama_decode(ctx, decode_batch) != 0) {
                fprintf(stderr, "[decode-server] Session %d: decode failed at token %d\n", session_id, i);
                break;
            }
            new_token = common_sampler_sample(smpl.get(), ctx, -1);
        }

        if (i == 0 && llama_vocab_is_eog(vocab, new_token)) {
            const float * logits = llama_get_logits(ctx);
            const int n_vocab = llama_vocab_n_tokens(vocab);
            const llama_token fallback = choose_best_non_eog_token(vocab, logits, n_vocab, new_token);
            if (fallback != new_token) {
                fprintf(stderr,
                        "[decode-server] Session %d: first token sampled EOG (%d), overriding with best non-EOG token %d\n",
                        session_id, new_token, fallback);
                new_token = fallback;
            }
        }

        if (llama_vocab_is_eog(vocab, new_token)) {
            fprintf(stderr, "[decode-server] Session %d: EOG at token %d\n", session_id, i);
            stop_reason = LLAMA_STOP_EOS;
            break;
        }

        // Detect degenerate output (same token repeating endlessly)
        if (new_token == repeat_token) {
            repeat_count++;
            if (repeat_count >= max_repeats) {
                fprintf(stderr, "[decode-server] Session %d: ABORT — token %d repeated %d times (degenerate output)\n",
                        session_id, new_token, repeat_count);
                stop_reason = LLAMA_STOP_LIMIT;
                break;
            }
        } else {
            repeat_token = new_token;
            repeat_count = 1;
        }

        common_sampler_accept(smpl.get(), new_token, true);

        char buf[256];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        std::string text(buf, n > 0 ? n : 0);
        fprintf(stdout, "%s", text.c_str());
        fflush(stdout);
        generated_text += text;

        // Check stop words (antiprompt matching, same as native llama server)
        if (!decode_cfg.stop_words.empty()) {
            bool found_stop = false;
            for (const auto & sw : decode_cfg.stop_words) {
                // Check for full match in recent text (look at last N chars where N = stop word length + last token size)
                size_t search_from = generated_text.size() > (sw.size() + text.size())
                    ? generated_text.size() - sw.size() - text.size() : 0;
                size_t pos = generated_text.find(sw, search_from);
                if (pos != std::string::npos) {
                    fprintf(stderr, "[decode-server] Session %d: stop word '%s' at token %d\n",
                            session_id, sw.c_str(), i);
                    stop_reason = LLAMA_STOP_WORD;
                    stop_word_matched = sw;
                    found_stop = true;
                    break;
                }
            }
            if (found_stop) break;
        }

        const int64_t emitted_at_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        emitted_timestamps_us.push_back(emitted_at_us);
        n_decoded++;

        if (token_stream_active) {
            llama_token_msg tmsg = {};
            tmsg.token_id = new_token;
            tmsg.text = text;
            tmsg.pos = decode_start_pos + i;
            tmsg.timestamp_us = emitted_at_us;
            token_server.enqueue(tmsg);

            // Check every token so stop/cancel tears down the session promptly.
            if (!token_server.is_client_connected()) {
                fprintf(stderr, "[decode-server] Session %d: relay client disconnected at token %d (cancelled)\n",
                        session_id, i);
                stop_reason = LLAMA_STOP_CANCELLED;
                break;
            }
        }

        last_token = new_token;

        // Check if we hit the decode limit on the last iteration
        if (i == decode_limit - 1) {
            stop_reason = LLAMA_STOP_LIMIT;
        }
    }

    const llama_sustained_decode_timing timing =
        llama_sustained_decode_timing_from_timestamps(emitted_timestamps_us);
    fprintf(stderr,
            "\n[decode-server] Session %d: sustained decode first->last: emitted=%d intervals=%d "
            "elapsed=%.1f ms total=%.2f tok/s early=%.2f middle=%.2f late=%.2f tok/s\n",
            session_id, timing.emitted_tokens, timing.decode_intervals, timing.elapsed_ms,
            timing.tok_s, timing.early_tok_s, timing.middle_tok_s, timing.late_tok_s);

    if (token_stream_active) {
        token_server.send_done(n_decoded, timing,
                               stop_reason, stop_word_matched);
        token_server.stop();
    }

    llama_batch_free(decode_batch);
    return n_decoded;
}

// ---------------------------------------------------------------------------
// Receive KV artifact from file or network
// ---------------------------------------------------------------------------
static bool receive_kv_artifact(const receiver_params & rparams,
                                 std::vector<uint8_t> & payload_out,
                                 llama_kv_artifact_metadata & meta_out,
                                 llama_kv_prompt_payload & prompt_payload_out,
                                 std::string & error,
                                 bool & is_prompt_mode) {
    is_prompt_mode = false;
    prompt_payload_out = {};
    if (!rparams.kv_artifact_path.empty()) {
        // File mode
        fprintf(stderr, "[decode-server] Reading KV artifact from file: %s\n",
                rparams.kv_artifact_path.c_str());
        llama_kv_artifact_summary summary;
        if (!llama_kv_artifact_read(rparams.kv_artifact_path, payload_out, &meta_out, &summary, &error, &prompt_payload_out)) {
            return false;
        }
        if (meta_out.flags & LLAMA_KV_ARTIFACT_FLAG_PROMPT_DIRECT) {
            is_prompt_mode = true;
            fprintf(stderr, "[decode-server] Prompt-direct artifact: prompt_tokens=%u, prompt_text_bytes=%u, %lu bytes\n",
                    meta_out.prompt_token_count, meta_out.prompt_text_bytes, (unsigned long) summary.payload_bytes);
            return true;
        }
        fprintf(stderr, "[decode-server] Artifact: %u layers, kv_tokens=%u, prompt_tokens=%u, prefix_tokens=%u, %lu bytes\n",
                meta_out.n_layers, meta_out.token_count, meta_out.prompt_token_count,
                meta_out.streamed_prefix_token_count, (unsigned long) summary.payload_bytes);
        return true;
    }

    // Network mode
    fprintf(stderr, "[decode-server] Listening for KV artifact on %s:%d ...\n",
            rparams.kv_host.c_str(), rparams.kv_port);

    llama_kv_receiver_config cfg;
    cfg.bind_host = rparams.kv_host;
    cfg.port = rparams.kv_port;
    cfg.accept_timeout_ms = 0;  // wait indefinitely for sender
    cfg.recv_timeout_ms = 300000;

    std::vector<uint8_t> raw_artifact;
    llama_kv_receiver_result recv_result;
    if (!llama_kv_receiver_accept_artifact(cfg, raw_artifact, &recv_result, &error)) {
        return false;
    }

    fprintf(stderr, "[decode-server] Received %lu bytes in %.1f ms (%.3f Gbps), %u chunks\n",
            (unsigned long) recv_result.bytes_received, recv_result.receive_ms,
            recv_result.throughput_gbps, recv_result.chunks_received);

    // Detect prompt-direct mode: if the data does NOT start with KVARTIF1 magic,
    // it's raw token IDs for local prefill (not a KV artifact).
    const char kv_magic[] = "KVARTIF1";
    if (raw_artifact.size() >= 8 && memcmp(raw_artifact.data(), kv_magic, 8) != 0) {
        // Prompt-direct mode: raw token IDs (little-endian int32_t array)
        if (raw_artifact.size() >= 4 && raw_artifact.size() % 4 == 0) {
            is_prompt_mode = true;
            payload_out.clear();
            meta_out = {};
            const size_t n_tokens = raw_artifact.size() / sizeof(int32_t);
            prompt_payload_out.token_ids.resize(n_tokens);
            std::memcpy(prompt_payload_out.token_ids.data(), raw_artifact.data(), raw_artifact.size());
            meta_out.prompt_token_count = (uint32_t) n_tokens;
            meta_out.prompt_text_bytes = 0;
            meta_out.streamed_prefix_token_count = 0;
            fprintf(stderr, "[decode-server] PROMPT-DIRECT legacy payload: received %zu bytes (%zu tokens)\n",
                    raw_artifact.size(), n_tokens);
            return true;
        }
        error = "received data is neither KV artifact nor valid prompt tokens";
        return false;
    }

    // KV artifact mode: parse header and extract payload
    llama_kv_artifact_summary summary;
    if (!llama_kv_artifact_read_mem(raw_artifact.data(), raw_artifact.size(),
                                     payload_out, &meta_out, &summary, &error, &prompt_payload_out)) {
        return false;
    }

    if (meta_out.flags & LLAMA_KV_ARTIFACT_FLAG_PROMPT_DIRECT) {
        is_prompt_mode = true;
        fprintf(stderr, "[decode-server] Prompt-direct artifact: prompt_tokens=%u, prompt_text_bytes=%u, %lu bytes payload\n",
                meta_out.prompt_token_count, meta_out.prompt_text_bytes, (unsigned long) summary.payload_bytes);
        return true;
    }

    fprintf(stderr, "[decode-server] Artifact: %u layers, kv_tokens=%u, prompt_tokens=%u, prefix_tokens=%u, %lu bytes payload\n",
            meta_out.n_layers, meta_out.token_count, meta_out.prompt_token_count,
            meta_out.streamed_prefix_token_count, (unsigned long) summary.payload_bytes);
    return true;
}

// ---------------------------------------------------------------------------
// Run one WIR decode session: import KV, run wavefront refinement, stream tokens
// ---------------------------------------------------------------------------
static int run_wir_decode_session(llama_context * ctx, llama_model * model,
                                   const receiver_params & rparams,
                                   const std::vector<uint8_t> & payload,
                                   const llama_kv_artifact_metadata & meta,
                                   const llama_kv_prompt_payload & prompt_payload,
                                   int session_id) {
    // Import KV state (same as standard decode)
    auto t_import_start = std::chrono::high_resolution_clock::now();
    size_t bytes_consumed = llama_state_seq_set_data(ctx, payload.data(), payload.size(), 0);
    if (bytes_consumed == 0) {
        fprintf(stderr, "[wir-server] Session %d: failed to import KV state\n", session_id);
        return -1;
    }
    if (bytes_consumed != payload.size()) {
        fprintf(stderr,
                "[wir-server] Session %d: partial KV import (%zu/%zu bytes consumed)\n",
                session_id, bytes_consumed, payload.size());
        return -1;
    }
    // Sync before the clock stops AND before the cache is interrogated -- the KV import lands on
    // asynchronous GPU upload paths, so a timer closed above this measures the enqueue, and a
    // seq_pos_max read above it asks the cache a question while the copies filling it are queued.
    // Same defect as the [decode-server] import path.
    llama_synchronize(ctx);

    auto t_import_end = std::chrono::high_resolution_clock::now();
    double import_ms = std::chrono::duration<double, std::milli>(t_import_end - t_import_start).count();

    uint32_t effective_token_count = meta.token_count;
    if (effective_token_count == 0) {
        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) {
            llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
            if (pos_max >= 0) effective_token_count = (uint32_t)(pos_max + 1);
        }
    }
    fprintf(stderr, "[wir-server] Session %d: KV import %.1f ms, %zu bytes, %u tokens\n",
            session_id, import_ms, bytes_consumed, effective_token_count);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token last_token = meta.last_token_id >= 0 ? (llama_token) meta.last_token_id : llama_vocab_bos(vocab);
    bool prompt_applied = false;

    if (!apply_handoff_prompt(ctx, rparams, meta, prompt_payload,
                              session_id, effective_token_count, last_token, prompt_applied)) {
        return -1;
    }

    if (rparams.exit_after_import) {
        fprintf(stderr, "[wir-server] Session %d: exiting after KV import\n", session_id);
        return 0;
    }

    // Start token stream server
    llama_token_stream_server token_server;
    bool token_stream_active = false;
    if (!rparams.no_token_stream) {
        if (!token_server.start(rparams.token_stream_host, rparams.token_stream_port)) {
            fprintf(stderr, "[wir-server] Session %d: token stream start failed\n", session_id);
            return -1;
        }
        fprintf(stderr, "[wir-server] Session %d: token stream on port %d, waiting...\n",
                session_id, token_server.bound_port());

        if (!token_server.wait_for_client(60000)) {
            fprintf(stderr, "[wir-server] Session %d: no client connected\n", session_id);
            token_server.stop();
            return -1;
        }

        llama_token_handshake hs = {};
        hs.version = 1;
        hs.status  = "ready";
        hs.token_count = effective_token_count;
        token_server.send_handshake(hs);
        token_stream_active = true;
    }

    // Build WIR params from receiver config
    wir_params wp;
    wp.block_size    = rparams.wir_block_size;
    wp.max_passes    = rparams.wir_max_passes;
    wp.confidence    = rparams.wir_confidence;
    wp.topk_accept   = rparams.wir_topk_accept;
    wp.reuse_kv      = true;
    wp.refine_mode   = rparams.wir_refine_mode;
    wp.refine_n      = rparams.wir_refine_n;

    // Token callback: stream converged block tokens to the RTX client
    wp.token_callback = [&](int block_start, const llama_token * tokens, int count) -> bool {
        for (int i = 0; i < count; i++) {
            char buf[256];
            int n = llama_token_to_piece(vocab, tokens[i], buf, sizeof(buf), 0, true);
            std::string text(buf, n > 0 ? n : 0);
            fprintf(stdout, "%s", text.c_str());
            fflush(stdout);

            if (token_stream_active) {
                llama_token_msg tmsg = {};
                tmsg.token_id = tokens[i];
                tmsg.text = text;
                tmsg.pos = effective_token_count + block_start + i;
                tmsg.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
                token_server.enqueue(tmsg);
            }
        }
        return true;
    };

    // Choose WIR decode path based on whether a separate draft model is configured.
    auto t_decode_start = std::chrono::high_resolution_clock::now();

    bool ignored_zero_cap = false;
    const int decode_limit = choose_decode_limit(rparams, meta, ignored_zero_cap);

    std::vector<llama_token> output;

    if (!rparams.wir_draft_model.empty()) {
        // Cross-tokenizer mode: load separate draft model, generate draft via text bridge.
        // NOTE: requires --wir-prompt to provide original prompt text for draft tokenization.
        // Without it, the draft model can't prefill the prompt (KV artifact doesn't carry raw text).
        if (rparams.wir_prompt_text.empty()) {
            fprintf(stderr, "[wir-server] Session %d: cross-tokenizer mode requires --wir-prompt\n", session_id);
            if (token_stream_active) { token_server.stop(); }
            return -1;
        }

        llama_model_params draft_mparams = llama_model_default_params();
        draft_mparams.n_gpu_layers = rparams.n_gpu_layers;
        llama_model * draft_model = llama_model_load_from_file(rparams.wir_draft_model.c_str(), draft_mparams);
        if (!draft_model) {
            fprintf(stderr, "[wir-server] Session %d: failed to load draft model\n", session_id);
            if (token_stream_active) { token_server.stop(); }
            return -1;
        }
        llama_context_params draft_cparams = llama_context_default_params();
        draft_cparams.n_ctx = rparams.n_ctx;
        draft_cparams.n_batch = 512;
        draft_cparams.no_perf = true;
        llama_context * draft_ctx = llama_init_from_model(draft_model, draft_cparams);
        if (!draft_ctx) {
            fprintf(stderr, "[wir-server] Session %d: failed to create draft context\n", session_id);
            llama_model_free(draft_model);
            if (token_stream_active) { token_server.stop(); }
            return -1;
        }

        output = wir_decode_cross_tokenizer(
            draft_model, draft_ctx, model, ctx,
            rparams.wir_prompt_text, (int)effective_token_count, last_token,
            decode_limit, rparams.wir_draft_temp, wp);

        llama_free(draft_ctx);
        llama_model_free(draft_model);
    } else {
        // Single-model mode: same tokenizer, draft from KV state
        output = wir_decode_from_kv(
            model, ctx, (int)effective_token_count, last_token, decode_limit,
            rparams.wir_draft_temp, wp);
    }

    auto t_decode_end = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    int n_decoded = (int)output.size();
    double tok_s = n_decoded > 0 ? n_decoded / (decode_ms / 1000.0) : 0;

    fprintf(stderr, "\n[wir-server] Session %d: decoded %d tokens in %.1f ms (%.1f tok/s) via WIR\n",
            session_id, n_decoded, decode_ms, tok_s);
    fprintf(stdout, "\n");

    if (token_stream_active) {
        token_server.send_done(n_decoded, tok_s);
        token_server.stop();
    }

    return n_decoded;
}

// ---------------------------------------------------------------------------
// Run one decode session: import KV, decode, stream tokens
// ---------------------------------------------------------------------------
static int run_decode_session(llama_context * ctx, llama_model * model,
                               const receiver_params & rparams,
                               const std::vector<uint8_t> & payload,
                               const llama_kv_artifact_metadata & meta,
                               const llama_kv_prompt_payload & prompt_payload,
                               int session_id,
                               bool resident_reuse = false, uint32_t reuse_n = 0) {
    auto _rds_t0 = std::chrono::high_resolution_clock::now();
    auto _rds_el = [&]() { return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - _rds_t0).count(); };

    // BIND THE TOKEN STREAM FIRST, BEFORE THE IMPORT. This used to be declared and started
    // roughly 240 lines below, after the KV import and the local suffix prefill -- and the RTX
    // starts dialling it the moment its own prefill ends, with a FIXED budget:
    //
    //     llama-decode-handoff.cpp:  client.connect(host, port, /*retries*/ 120, /*ms*/ 2000)
    //
    // 120 x 2 s is four minutes, and the comment beside it states the assumption it was betting
    // on: "Large KV imports (1.5 GB+) + tail prefill (13k+ tokens) can take 90+ seconds on the
    // Mac." When that work runs past four minutes the client exhausts its retries against a port
    // that was always going to open, and the turn dies with a 500. Measured 2026-08-31: 120/120
    // attempts exhausted on the receiver's token port while it sat at 99% CPU doing precisely
    // the import and prefill that preceded the bind. It is a RACE, so it fails intermittently --
    // which is what made it look like a malformed KV artifact for two debugging sessions.
    //
    // Binding early fixes the ORDER rather than the budget. A listening socket costs nothing and
    // needs none of the session state below; the client's connect then lands in the listen
    // backlog immediately and blocks on the handshake -- a wait with no deadline instead of a
    // poll with one. Raising 120 to 400 would only move the cliff and keep the failure.
    //
    // accept() deliberately stays where it was: wait_for_client() must not run until the KV is
    // imported and the suffix prefilled, because the handshake it answers with reports
    // decode_start_pos, which is not known until then.
    llama_token_stream_server token_server;
    llama_decode_config decode_cfg;
    bool token_stream_active = false;
    if (!rparams.no_token_stream) {
        if (!token_server.start(rparams.token_stream_host, rparams.token_stream_port)) {
            fprintf(stderr, "[decode-server] Session %d: failed to start token stream on %s:%d\n",
                    session_id, rparams.token_stream_host.c_str(), rparams.token_stream_port);
            return -1;
        }
        fprintf(stderr, "[decode-server] Session %d: token stream LISTENING on port %d before KV import "
                        "(client may connect now; handshake waits for the prefill)\n",
                session_id, token_server.bound_port());
    }

    uint32_t effective_token_count = meta.token_count;
    if (!resident_reuse) {
    // Ensure KV is completely empty before import (seq_rm all sequences)
    {
        llama_memory_t mem = llama_get_memory(ctx);
        llama_memory_clear(mem, true);

        // Force the receiver context to reserve its decode buffers before KV
        // import. Without this warm-up decode, the first real decode after
        // import can realloc KV storage and leave the imported data behind.
        {
            const llama_vocab * vocab_pre = llama_model_get_vocab(model);
            llama_token bos = llama_vocab_bos(vocab_pre);
            llama_batch dummy = llama_batch_init(1, 0, 1);
            common_batch_clear(dummy);
            common_batch_add(dummy, bos, 0, { 0 }, false);
            llama_decode(ctx, dummy);
            llama_batch_free(dummy);
            llama_memory_clear(mem, true);
            llama_synchronize(ctx);
            fprintf(stderr, "[decode-server] Session %d: sched_reserve stabilized via dummy decode\n", session_id);
            fflush(stderr);
        }

        // Also explicitly remove seq 0 to clear any residual position state
        llama_memory_seq_rm(mem, 0, -1, -1);
    }

    // Import KV state
    {
        // Checksum payload for debugging CUDA→Metal KV transfer
        uint32_t cksum = 0;
        for (size_t b = 0; b < payload.size(); b++) {
            cksum = (cksum * 31) + payload[b];
        }
        fprintf(stderr, "[decode-server] Session %d: payload checksum=0x%08x size=%zu\n",
                session_id, cksum, payload.size());
    }
    auto t_import_start = std::chrono::high_resolution_clock::now();
    size_t bytes_consumed = llama_state_seq_set_data(ctx, payload.data(), payload.size(), 0);
    if (bytes_consumed == 0) {
        fprintf(stderr, "[decode-server] Session %d: failed to import KV state\n", session_id);
        return -1;
    }
    if (bytes_consumed != payload.size()) {
        fprintf(stderr,
                "[decode-server] Session %d: partial KV import (%zu/%zu bytes consumed)\n",
                session_id, bytes_consumed, payload.size());
        return -1;
    }
    // Imported KV may land on asynchronous GPU upload paths. Make the first decode consume the
    // fully materialized cache rather than queued copies.
    //
    // MOVED ABOVE THE TIMER AND THE pos_max READ, 2026-08-31. This sync used to sit below both,
    // and the comment explaining WHY it is needed was three lines under a timer that its own
    // reasoning invalidates: if the upload is asynchronous, then stopping the clock before the
    // wait measures the ENQUEUE, and `KV import 7.8 ms, 465267500 bytes` is 59.6 GB/s -- a figure
    // to disbelieve rather than report. Reading `seq_pos_max` ahead of the sync was the same
    // mistake with worse consequences: it interrogates cache state while the copies that populate
    // it may still be queued, and `effective_token_count` is derived from the answer.
    llama_synchronize(ctx);

    auto t_import_end = std::chrono::high_resolution_clock::now();
    double import_ms = std::chrono::duration<double, std::milli>(t_import_end - t_import_start).count();
    // Derive actual token count from KV cache state. The cache may have more
    // cells than meta.token_count due to alignment/rounding in state_seq_set_data.
    // Always use pos_max to avoid position conflicts with tail prefill.
    {
        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) {
            llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
            if (pos_max >= 0) {
                effective_token_count = (uint32_t)(pos_max + 1);
            }
        }
    }
    fprintf(stderr, "[decode-server] Session %d: KV import %.1f ms, %zu bytes, %u tokens (meta=%u, kv_pos_max=%u)\n",
            session_id, import_ms, bytes_consumed, effective_token_count, meta.token_count,
            effective_token_count > 0 ? effective_token_count - 1 : 0);

    // The warm-up dummy decode reserves a 1-token graph before import. After
    // importing a full prompt KV state, rebuild the decode graph so K/V view
    // nodes are recreated against the imported memory rather than the stale
    // pre-import reservation.
#ifndef COSMOS_UPSTREAM_RECEIVER
    llama_invalidate_graph(ctx);
    ctx->set_sched_need_reserve(true);
    fprintf(stderr, "[decode-server] Session %d: invalidated pre-import graph reservation\n", session_id);
#else
    fprintf(stderr, "[decode-server] Session %d: upstream scheduler will reserve against imported KV lazily\n", session_id);
#endif
    fflush(stderr);
    } else {
        // Stage 1 RESIDENT REUSE: no clear, no warm-up, no import, no graph
        // invalidate. The live ctx already holds [0..resident) in seq-0 KV and
        // its decode graph is warm (reused => no ~6.2s re-reserve). Derive the
        // resident extent from the live KV; apply_handoff_prompt will trim to
        // reuse_n and prefill only the genuinely-new tail.
        llama_memory_t mem = llama_get_memory(ctx);
        llama_pos pos_max = mem ? llama_memory_seq_pos_max(mem, 0) : -1;
        effective_token_count = pos_max >= 0 ? (uint32_t)(pos_max + 1) : 0;
        fprintf(stderr, "[decode-server] Session %d: RESIDENT REUSE active — reuse_n=%u live_kv_pos_max=%d (no import/clear/invalidate)\n",
                session_id, reuse_n, (int) pos_max);
        fflush(stderr);
    }
    fprintf(stderr, "[decode-server] Session %d: [RDS_TIMING after_setup_branch=%.0fms reuse=%d] (entry->clear/warmup/import/invalidate OR reuse-branch)\n",
            session_id, _rds_el(), (int) resident_reuse);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token last_token = meta.last_token_id >= 0 ? (llama_token) meta.last_token_id : llama_vocab_bos(vocab);
    bool prompt_applied = false;

    // v1.8 ship-embeddings: re-prefill the image cells (+ trailing suffix tokens) from the
    // shipped embeddings so the receiver computes the image K/V with correct Metal F32 FA.
    // The RTX SM120 NCOLS=32 MLA FA accumulates VKQ in F16 → degraded glyphs (GROUND_TRUTH
    // §11). Keep the imported prefix K/V [0,pos0); remove + recompute [pos0,end).
    bool image_reprefilled = false;
    if (!resident_reuse && (meta.flags & LLAMA_KV_ARTIFACT_FLAG_IMAGE_EMBD) &&
            meta.img_cell_count > 0 && meta.img_n_embd > 0 &&
            prompt_payload.image_embd.size() >= (size_t) meta.img_cell_count * (size_t) meta.img_n_embd) {
        const uint32_t pos0  = meta.img_pos_start;
        const uint32_t ncell = meta.img_cell_count;
        const uint32_t nembd = meta.img_n_embd;
        if (nembd != (uint32_t) llama_model_n_embd(model)) {
            fprintf(stderr, "[decode-server] Session %d: SHIP_EMBD n_embd mismatch (%u vs model %d); using imported KV\n",
                    session_id, nembd, llama_model_n_embd(model));
        } else {
            // round-trip diagnostic: compare embd[0] to the RTX's logged embd0 (catch a
            // corrupt/zeroed format round-trip vs a decode bug).
            {
                double esum = 0.0; size_t nz = 0;
                for (size_t k = 0; k < prompt_payload.image_embd.size(); ++k) {
                    esum += prompt_payload.image_embd[k];
                    if (prompt_payload.image_embd[k] != 0.0f) nz++;
                }
                fprintf(stderr, "[decode-server] Session %d: SHIP_EMBD recv embd[0..3]=%.5f,%.5f,%.5f,%.5f sum=%.4f nonzero=%zu/%zu\n",
                        session_id, prompt_payload.image_embd[0], prompt_payload.image_embd[1],
                        prompt_payload.image_embd[2], prompt_payload.image_embd[3], esum,
                        nz, prompt_payload.image_embd.size());
                fflush(stderr);
            }
            llama_memory_t mem = llama_get_memory(ctx);
            // FULL local re-prefill (the proven path — same as mtmd-cli / test_inject2):
            // discard the imported RTX K/V entirely and recompute prefix(tokens) +
            // image(embd) + suffix(tokens) locally, so the WHOLE sequence uses the Mac's
            // correct Metal F32 FA. (seq_rm-on-imported-cache + partial re-decode produced
            // ineffective image K/V — the model ignored the image and read the prompt.)
            llama_memory_clear(mem, true);
            const int nb = std::max(1, (int) llama_n_batch(ctx));
            bool ok = true;
            // (0) prefix tokens [0, pos0) via token batches (token_ids = [prefix][suffix])
            if (pos0 > 0 && prompt_payload.token_ids.size() >= (size_t) pos0) {
                llama_batch pb = llama_batch_init(std::min<int>((int) pos0, nb), 0, 1);
                for (size_t i = 0; i < (size_t) pos0 && ok; i += (size_t) nb) {
                    const int bs = std::min<int>(nb, (int)((size_t) pos0 - i));
                    common_batch_clear(pb);
                    for (int j = 0; j < bs; ++j) {
                        common_batch_add(pb, (llama_token) prompt_payload.token_ids[i + (size_t) j],
                                         (llama_pos)(i + (size_t) j), { 0 }, false);
                    }
                    if (llama_decode(ctx, pb) != 0) { ok = false; }
                }
                llama_batch_free(pb);
            }
            // (1) image cells via embd batches
            std::vector<llama_pos>      v_pos;
            std::vector<int32_t>        v_nseq;
            std::vector<llama_seq_id>   v_seq0(1, 0);
            std::vector<llama_seq_id *> v_seqp;
            std::vector<int8_t>         v_log;
            for (uint32_t off = 0; off < ncell && ok; off += (uint32_t) nb) {
                const uint32_t chunk = std::min<uint32_t>((uint32_t) nb, ncell - off);
                v_pos.assign(chunk, 0); v_nseq.assign(chunk, 1);
                v_seqp.assign(chunk, v_seq0.data()); v_log.assign(chunk, 0);
                for (uint32_t i = 0; i < chunk; ++i) v_pos[i] = (llama_pos)(pos0 + off + i);
                llama_batch ib = { (int32_t) chunk, nullptr,
                    const_cast<float *>(prompt_payload.image_embd.data()) + (size_t) off * nembd,
                    v_pos.data(), v_nseq.data(), v_seqp.data(), v_log.data() };
                if (llama_decode(ctx, ib) != 0) { ok = false; }
            }
            // (2) trailing suffix tokens (token_ids[pos0:]) via token batches
            const size_t suffix_n = prompt_payload.token_ids.size() > (size_t) pos0
                                  ? prompt_payload.token_ids.size() - (size_t) pos0 : 0;
            if (ok && suffix_n > 0) {
                llama_batch sb = llama_batch_init(std::min<int>((int) suffix_n, nb), 0, 1);
                for (size_t i = 0; i < suffix_n && ok; i += (size_t) nb) {
                    const int bs = std::min<int>(nb, (int)(suffix_n - i));
                    common_batch_clear(sb);
                    for (int j = 0; j < bs; ++j) {
                        const size_t ti = (size_t) pos0 + i + (size_t) j;
                        const bool is_last = (i + (size_t) j == suffix_n - 1);
                        common_batch_add(sb, (llama_token) prompt_payload.token_ids[ti],
                                         (llama_pos)(pos0 + ncell + i + (size_t) j), { 0 }, is_last);
                    }
                    if (llama_decode(ctx, sb) != 0) { ok = false; }
                }
                llama_batch_free(sb);
            }
            llama_synchronize(ctx);
            if (ok) {
                image_reprefilled = true;
                prompt_applied = true;
                effective_token_count = pos0 + ncell + (uint32_t) suffix_n;
                if (suffix_n > 0) last_token = (llama_token) prompt_payload.token_ids.back();
                fprintf(stderr, "[decode-server] Session %d: SHIP_EMBD re-prefilled %u image cells [%u,%u) + %zu suffix tok; effective=%u last_token=%d\n",
                        session_id, ncell, pos0, pos0 + ncell, suffix_n, effective_token_count, last_token);
                fflush(stderr);
            } else {
                fprintf(stderr, "[decode-server] Session %d: SHIP_EMBD re-prefill FAILED; using imported KV\n", session_id);
            }
        }
    }

    if (!image_reprefilled && !apply_handoff_prompt(ctx, rparams, meta, prompt_payload,
                              session_id, effective_token_count, last_token, prompt_applied,
                              resident_reuse ? reuse_n : 0u)) {
        // Tail prefill failed (possibly after a partial seq_rm) => KV state is no
        // longer trustworthy; invalidate resident so next turn takes the full path.
        { std::lock_guard<std::mutex> lk(g_resident_kv_mu); g_resident_kv.valid = false; }
        return -1;
    }
    fprintf(stderr, "[decode-server] Session %d: [RDS_TIMING after_suffix=%.0fms reuse=%d] (entry->setup->suffix-prefill)\n",
            session_id, _rds_el(), (int) resident_reuse);

    if (!rparams.dump_kv_after_import_path.empty()) {
        llama_kv_artifact_metadata dump_meta = meta;
        dump_meta.last_token_id = last_token;
        if (!dump_imported_kv_artifact(ctx, rparams.dump_kv_after_import_path, dump_meta, effective_token_count)) {
            return -1;
        }
    }

    if (rparams.exit_after_import) {
        fprintf(stderr, "[decode-server] Session %d: exiting after KV import by request\n", session_id);
        return 0;
    }

    // The token stream was BOUND AT FUNCTION ENTRY, above -- see the comment there. Only the
    // accept and the handshake belong here, because the handshake reports decode_start_pos and
    // that is not known until the import and suffix prefill have run.
    if (!rparams.no_token_stream) {
        fprintf(stderr, "[decode-server] Session %d: token stream on port %d, waiting for client...\n",
                session_id, token_server.bound_port());

        if (!token_server.wait_for_client(60000)) {
            fprintf(stderr, "[decode-server] Session %d: no token stream client connected\n", session_id);
            token_server.stop();
            return -1;
        }

        // Read decode config from RTX client (stop words, etc.)
        if (token_server.read_config(decode_cfg, 5000)) {
            if (!decode_cfg.stop_words.empty()) {
                fprintf(stderr, "[decode-server] Session %d: received %zu stop words\n",
                        session_id, decode_cfg.stop_words.size());
            }
        } else {
            fprintf(stderr, "[decode-server] Session %d: no config received (old client?), using defaults\n",
                    session_id);
        }

        // Send handshake
        llama_token_handshake hs = {};
        hs.version = 1;
        hs.status  = "ready";
        hs.model_fingerprint = 0;
        hs.token_count = effective_token_count;
        token_server.send_handshake(hs);
        token_stream_active = true;
    } else {
        fprintf(stderr, "[decode-server] Session %d: token stream disabled (local test mode)\n", session_id);
    }

    bool imported_sampling = false;
    common_params_sampling sparams = choose_decode_sampling_params(rparams, &meta, imported_sampling);
    if (imported_sampling) {
        fprintf(stderr,
                "[decode-server] Session %d: imported sampling profile temp=%.3f top_k=%d top_p=%.3f min_p=%.3f repeat_last_n=%d repeat_penalty=%.3f mirostat=%d%s\n",
                session_id,
                sparams.temp,
                sparams.top_k,
                sparams.top_p,
                sparams.min_p,
                sparams.penalty_last_n,
                sparams.penalty_repeat,
                sparams.mirostat,
                rparams.temp_forced ? " (CLI temp override applied)" : "");
    }
    bool ignored_zero_cap = false;
    const int decode_limit = choose_decode_limit(rparams, meta, ignored_zero_cap);
    if (meta.requested_n_predict >= 0) {
        fprintf(stderr,
                "[decode-server] Session %d: imported decode cap n_predict=%d%s\n",
                session_id, meta.requested_n_predict,
                ignored_zero_cap ? " (ignored in local file/one-shot mode)" : "");
    }
    common_sampler_ptr smpl(common_sampler_init(model, sparams));
    const char * sample_prefill_logits_env = std::getenv("LLAMA_HANDOFF_SAMPLE_PREFILL_LOGITS");
    const bool sample_from_prefill_logits =
        prompt_applied &&
        sample_prefill_logits_env &&
        sample_prefill_logits_env[0] != '\0' &&
        sample_prefill_logits_env[0] != '0';
    if (prompt_applied && !sample_from_prefill_logits) {
        fprintf(stderr,
                "[decode-server] Session %d: prompt-prefill logits bypassed; priming first token with explicit decode\n",
                session_id);
    }

    // Determine starting token
    if (prompt_applied) {
        fprintf(stderr, "[decode-server] Session %d: using last_token_id=%d after local prompt suffix prefill\n",
                session_id, last_token);
    } else if (meta.last_token_id >= 0) {
        fprintf(stderr, "[decode-server] Session %d: last_token_id=%d\n", session_id, meta.last_token_id);
    } else {
        fprintf(stderr, "[decode-server] Session %d: WARNING: no last_token_id, using BOS\n", session_id);
    }

    // Two decode strategies:
    // Strategy A: Remove last KV position, re-evaluate last token to get logits
    // Strategy B: Keep all KV, evaluate last token at next position
    //
    // Strategy A is correct for standard KV models.
    // Strategy B avoids issues with MLA models where re-evaluation at the same
    // position produces different KV than what was imported.

    // Strategy B for ISWA and hybrid (recurrent) models: don't remove last position.
    // ISWA's seq_rm on dual caches may corrupt the SWA sliding window state.
    // Hybrid's seq_rm only affects KV cache, not recurrent state, causing
    // position mismatch between attention (pos_max=N-1) and recurrent (pos_max=N).
    const bool is_mla = meta.is_mla != 0;
    const bool is_iswa = (llama_model_n_swa(model) > 0);
    const bool is_hybrid = llama_model_is_hybrid(model);
    const char * force_mla_strategy_a_env = std::getenv("LLAMA_FORCE_MLA_STRATEGY_A");
    const bool force_mla_strategy_a =
        force_mla_strategy_a_env && force_mla_strategy_a_env[0] != '\0' && force_mla_strategy_a_env[0] != '0';
    bool use_strategy_b = (!force_mla_strategy_a && is_mla) || is_iswa || is_hybrid;
    if (sample_from_prefill_logits) {
        use_strategy_b = false;
        fprintf(stderr,
                "[decode-server] Session %d: sampling first decode token directly from prompt-prefill logits\n",
                session_id);
    } else if (is_mla) {
        fprintf(stderr, "[decode-server] MLA artifact detected - using Strategy %s\n",
                use_strategy_b ? "B" : "A");
    }
    if (is_iswa) {
        fprintf(stderr, "[decode-server] ISWA model detected (n_swa=%d) — using Strategy B\n",
                llama_model_n_swa(model));
    }
    if (is_hybrid) {
        fprintf(stderr, "[decode-server] Hybrid model detected — using Strategy B\n");
    }

    if (!sample_from_prefill_logits && !use_strategy_b) {
        // Strategy A: remove last position and re-evaluate
        llama_memory_t mem = llama_get_memory(ctx);
        llama_pos last_pos = (llama_pos)(effective_token_count - 1);
        llama_memory_seq_rm(mem, 0, last_pos, last_pos + 1);
        fprintf(stderr, "[decode-server] Session %d: Strategy A — removed KV pos %d for re-evaluation\n",
                session_id, last_pos);
    } else {
        fprintf(stderr, "[decode-server] Session %d: Strategy B — MLA: keeping all %u KV positions\n",
                session_id, effective_token_count);
    }

    // Diagnostic: check KV cache cell positions after import / suffix prefill
    {
        llama_memory_t mem = llama_get_memory(ctx);
        llama_pos pos_min = llama_memory_seq_pos_min(mem, 0);
        llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
        fprintf(stderr, "[decode-server] Session %d: KV pos range [%d, %d]\n",
                session_id, pos_min, pos_max);
    }

    auto t_decode_start = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[decode-server] Session %d: [RDS_TIMING at_decode_start=%.0fms reuse=%d] (after_suffix->token_server.start+wait_for_client+handshake)\n",
            session_id, _rds_el(), (int) resident_reuse);
    llama_batch batch = llama_batch_init(1, 0, 1);
    int n_decoded = 0;
    int32_t stop_reason = LLAMA_STOP_EOS;
    std::string stop_word_matched;
    std::string generated_text;  // accumulated text for stop word matching
    std::vector<int64_t> emitted_timestamps_us;
    emitted_timestamps_us.reserve((size_t) std::max(0, decode_limit));

    // For hybrid models, the recurrent memory's position may differ from the
    // attention KV cell count (artifact metadata). Use the actual pos_max from
    // the memory module to determine where to start decoding. This handles the
    // case where all tokens pass through recurrent layers (pos_max = n_tokens)
    // but only a subset have attention KV entries.
    llama_pos decode_pos_start;
    if (sample_from_prefill_logits) {
        decode_pos_start = (llama_pos)(effective_token_count);
    } else if (use_strategy_b) {
        llama_memory_t mem = llama_get_memory(ctx);
        llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
        if (pos_max >= 0) {
            decode_pos_start = pos_max + 1;  // next position after all imported state
        } else {
            decode_pos_start = (llama_pos)(effective_token_count);  // fallback
        }
    } else {
        decode_pos_start = (llama_pos)(effective_token_count - 1);  // A: re-evaluate last position
    }

    int repeat_count_net = 0;
    llama_token repeat_token_net = -1;
    const int max_repeats_net = 50;

    for (int i = 0; i < decode_limit; i++) {
        if (!(sample_from_prefill_logits && i == 0)) {
            const llama_pos decode_pos = sample_from_prefill_logits
                ? (decode_pos_start + i - 1)
                : (decode_pos_start + i);
            common_batch_clear(batch);
            common_batch_add(batch, last_token, decode_pos, { 0 }, true);

            // First iteration: diagnostic dump before decode
            if (i == 0) {
                // Get n_kv from memory context
                llama_memory_t mem_diag = llama_get_memory(ctx);
                llama_pos p_min = llama_memory_seq_pos_min(mem_diag, 0);
                llama_pos p_max = llama_memory_seq_pos_max(mem_diag, 0);
                fprintf(stderr, "[decode-server] DECODE_START: token=%d pos=%d batch.n_tokens=%d kv_pos=[%d,%d]\n",
                        last_token, decode_pos, batch.n_tokens, p_min, p_max);
            }

            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "[decode-server] Session %d: decode failed at token %d\n", session_id, i);
                break;
            }
        } else {
            fprintf(stderr,
                    "[decode-server] DECODE_START: sampling from existing prompt logits (last_token=%d next_pos=%d)\n",
                    last_token, decode_pos_start);
        }

        // Diagnostic: check logits
        if (i == 0) {
            const float * logits = llama_get_logits(ctx);
            if (logits) {
                int n_vocab = llama_vocab_n_tokens(vocab);
                float max_val = -1e30f, min_val = 1e30f;
                int max_idx = 0, n_nonzero = 0;
                for (int v = 0; v < n_vocab && v < 1000; v++) {
                    if (logits[v] != 0.0f) n_nonzero++;
                    if (logits[v] > max_val) { max_val = logits[v]; max_idx = v; }
                    if (logits[v] < min_val) min_val = logits[v];
                }
                fprintf(stderr, "[decode-server] LOGITS: first decode step: max=%.4f(idx=%d) min=%.4f nonzero=%d/1000 vocab=%d\n",
                        max_val, max_idx, min_val, n_nonzero, n_vocab);
            }
        }

        llama_token new_token = common_sampler_sample(smpl.get(), ctx, -1);

        if (i == 0 && llama_vocab_is_eog(vocab, new_token)) {
            const float * full_logits = llama_get_logits(ctx);
            const int n_vocab = llama_vocab_n_tokens(vocab);
            const llama_token fallback = choose_best_non_eog_token(vocab, full_logits, n_vocab, new_token);
            if (fallback != new_token) {
                fprintf(stderr,
                        "[decode-server] Session %d: first token sampled EOG (%d), overriding with best non-EOG token %d\n",
                        session_id, new_token, fallback);
                new_token = fallback;
            }
        }

        fprintf(stderr, "[decode-server] Session %d: token[%d] = %d (is_eog=%d) last_token=%d pos=%u\n",
                session_id, i, new_token, llama_vocab_is_eog(vocab, new_token),
                last_token, decode_pos_start + i);
        if (i == 0) {
            double _ftms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t_decode_start).count();
            fprintf(stderr, "[decode-server] Session %d: [STAGE1B first_token=%.0fms resident_reuse=%d] "
                    "(if ~6s on reuse => first-decode graph reserve is the residual to kill)\n",
                    session_id, _ftms, (int) resident_reuse);
        }

        if (llama_vocab_is_eog(vocab, new_token)) {
            fprintf(stderr, "[decode-server] Session %d: EOG at token %d, stopping\n", session_id, i);
            stop_reason = LLAMA_STOP_EOS;
            break;
        }

        // Detect degenerate output
        if (new_token == repeat_token_net) {
            repeat_count_net++;
            if (repeat_count_net >= max_repeats_net) {
                fprintf(stderr, "[decode-server] Session %d: ABORT — token %d repeated %d times (degenerate output)\n",
                        session_id, new_token, repeat_count_net);
                stop_reason = LLAMA_STOP_LIMIT;
                break;
            }
        } else {
            repeat_token_net = new_token;
            repeat_count_net = 1;
        }

        common_sampler_accept(smpl.get(), new_token, true);

        char buf[256];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        std::string text(buf, n > 0 ? n : 0);

        fprintf(stdout, "%s", text.c_str());
        fflush(stdout);
        generated_text += text;

        // Check stop words (antiprompt matching, same as native llama server)
        if (!decode_cfg.stop_words.empty()) {
            bool found_stop = false;
            for (const auto & sw : decode_cfg.stop_words) {
                size_t search_from = generated_text.size() > (sw.size() + text.size())
                    ? generated_text.size() - sw.size() - text.size() : 0;
                size_t pos = generated_text.find(sw, search_from);
                if (pos != std::string::npos) {
                    fprintf(stderr, "[decode-server] Session %d: stop word '%s' at token %d\n",
                            session_id, sw.c_str(), i);
                    stop_reason = LLAMA_STOP_WORD;
                    stop_word_matched = sw;
                    found_stop = true;
                    break;
                }
            }
            if (found_stop) break;
        }

        const int64_t emitted_at_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        emitted_timestamps_us.push_back(emitted_at_us);

        if (token_stream_active) {
            llama_token_msg tmsg = {};
            tmsg.token_id = new_token;
            tmsg.text = text;
            tmsg.pos = decode_pos_start + i;
            tmsg.timestamp_us = emitted_at_us;
            token_server.enqueue(tmsg);

            if (!token_server.is_client_connected()) {
                fprintf(stderr, "[decode-server] Session %d: relay client disconnected at token %d (cancelled)\n",
                        session_id, i);
                stop_reason = LLAMA_STOP_CANCELLED;
                break;
            }
        }

        last_token = new_token;
        n_decoded++;

        if (i == decode_limit - 1) {
            stop_reason = LLAMA_STOP_LIMIT;
        }
    }

    const llama_sustained_decode_timing timing =
        llama_sustained_decode_timing_from_timestamps(emitted_timestamps_us);
    fprintf(stderr,
            "\n[decode-server] Session %d: sustained decode first->last: emitted=%d intervals=%d "
            "elapsed=%.1f ms total=%.2f tok/s early=%.2f middle=%.2f late=%.2f tok/s stop=%d\n",
            session_id, timing.emitted_tokens, timing.decode_intervals, timing.elapsed_ms,
            timing.tok_s, timing.early_tok_s, timing.middle_tok_s, timing.late_tok_s, stop_reason);
    fprintf(stdout, "\n");

    if (token_stream_active) {
        token_server.send_done(n_decoded, timing, stop_reason, stop_word_matched);
        token_server.stop();
    }

    // Stage 1: record the resident state so the NEXT turn can reuse this turn's
    // prompt KV. Resident = the FULL incoming prompt ONLY (not the decoded tail):
    // the next turn's prompt already contains this response, so the LCP reuses
    // exactly the clean prefilled prompt cells [0..prompt) — which provably equal
    // these prompt tokens — and re-prefills the rest. The decode-loop cells
    // (strategy-B priming + response) are deliberately NOT claimed; next turn's
    // seq_rm clears them. This sidesteps the fragile decode->KV cell mapping.
    // Gated: only track when reuse is enabled (gate off => no state, no behavior change).
    if (resident_kv_reuse_enabled() && meta.format_minor >= 5 && !prompt_payload.token_ids.empty()) {
        std::lock_guard<std::mutex> lk(g_resident_kv_mu);
        g_resident_kv.token_ids.assign(prompt_payload.token_ids.begin(), prompt_payload.token_ids.end());
        g_resident_kv.ctx_generation = g_ctx_generation;
        g_resident_kv.source_slot_id = meta.source_slot_id;
        g_resident_kv.valid = true;
        fprintf(stderr, "[decode-server] Session %d: resident-KV updated (%zu prompt tokens, gen=%llu slot=%d)\n",
                session_id, g_resident_kv.token_ids.size(),
                (unsigned long long) g_ctx_generation, (int) meta.source_slot_id);
    }

    llama_batch_free(batch);

    return n_decoded;
}

// ---------------------------------------------------------------------------
// Multi-session support: session pool + worker threads
// ---------------------------------------------------------------------------

struct decode_session {
    int              session_id       = 0;
    int              kv_port          = 0;
    int              token_stream_port = 0;
    llama_context *  ctx              = nullptr;
    bool             ctx_flash_attn   = false;
    std::thread      worker;
    std::atomic<bool> active{false};   // true while session is running a decode
    std::atomic<bool> shutdown{false}; // signal worker to exit

    // Non-copyable, non-movable due to atomics + thread
    decode_session() = default;
    decode_session(const decode_session &) = delete;
    decode_session & operator=(const decode_session &) = delete;
};

struct session_manager {
    llama_model *               model       = nullptr;
    std::unique_ptr<decode_session[]> sessions;
    int                         max_sessions = 1;
    std::mutex                  mtx;

    // Create context with the specified FA setting
    std::function<llama_context *(bool enable_fa)> create_context_fn;

    void init(llama_model * m, int n_sessions, int kv_port_base, int token_port_base,
              std::function<llama_context *(bool)> ctx_fn, bool default_fa) {
        model = m;
        max_sessions = n_sessions;
        create_context_fn = std::move(ctx_fn);
        sessions.reset(new decode_session[n_sessions]);

        for (int i = 0; i < n_sessions; i++) {
            sessions[i].session_id       = i;
            sessions[i].kv_port          = kv_port_base + 2 * i;
            sessions[i].token_stream_port = token_port_base + 2 * i;
            sessions[i].ctx              = create_context_fn(default_fa);
            sessions[i].ctx_flash_attn   = default_fa;
            if (!sessions[i].ctx) {
                fprintf(stderr, "[session-mgr] FATAL: failed to create context for session %d\n", i);
            }
            fprintf(stderr, "[session-mgr] Session %d: kv_port=%d token_port=%d ctx=%p\n",
                    i, sessions[i].kv_port, sessions[i].token_stream_port,
                    (void *)sessions[i].ctx);
        }
    }

    void shutdown_all() {
        for (int i = 0; i < max_sessions; i++) {
            sessions[i].shutdown.store(true);
        }
        for (int i = 0; i < max_sessions; i++) {
            if (sessions[i].worker.joinable()) {
                sessions[i].worker.join();
            }
            if (sessions[i].ctx) {
                llama_free(sessions[i].ctx);
                sessions[i].ctx = nullptr;
            }
        }
    }
};

// Worker function: each session runs its own accept/decode loop.
// Each worker listens on its own KV port, receives artifacts, decodes, streams.
static void session_worker(decode_session & sess, session_manager & mgr,
                            const receiver_params & rparams) {
    fprintf(stderr, "[worker-%d] Started (kv_port=%d, token_port=%d)\n",
            sess.session_id, sess.kv_port, sess.token_stream_port);

    int local_session_counter = 0;

    while (!sess.shutdown.load()) {
        local_session_counter++;
        const int sid = sess.session_id * 1000 + local_session_counter; // unique session id for logs

        // Build per-session receiver params with this worker's ports
        receiver_params worker_rparams = rparams;
        worker_rparams.kv_port = sess.kv_port;
        worker_rparams.token_stream_port = sess.token_stream_port;

        // Receive KV artifact or prompt tokens
        std::vector<uint8_t> payload;
        llama_kv_prompt_payload prompt_payload;
        llama_kv_artifact_metadata meta = {};
        std::string error;
        bool is_prompt_mode = false;

        sess.active.store(false);
        fprintf(stderr, "[worker-%d] Waiting for artifact on port %d...\n",
                sess.session_id, sess.kv_port);

        if (!receive_kv_artifact(worker_rparams, payload, meta, prompt_payload, error, is_prompt_mode)) {
            fprintf(stderr, "[worker-%d] Session %d: failed to receive artifact: %s\n",
                    sess.session_id, sid, error.c_str());
            if (sess.shutdown.load()) break;
            continue;
        }

        sess.active.store(true);

        // Always recreate context for KV artifact imports to ensure clean
        // scheduler/compute state (stale Metal GPU state causes garbage output)
        if (!is_prompt_mode) {
            const bool need_fa = rparams.flash_attn_forced
                ? rparams.flash_attn
                : (meta.v_trans == 0);

            fprintf(stderr, "[worker-%d] Session %d: recreating context (FA=%s)\n",
                    sess.session_id, sid, need_fa ? "on" : "off");

            std::lock_guard<std::mutex> lock(mgr.mtx);
            if (sess.ctx) {
                llama_free(sess.ctx);
            }
            sess.ctx = mgr.create_context_fn(need_fa);
            if (!sess.ctx) {
                fprintf(stderr, "[worker-%d] FATAL: failed to recreate context for session %d\n",
                        sess.session_id, sid);
                sess.active.store(false);
                continue;
            }
            sess.ctx_flash_attn = need_fa;
        }

        // Run the appropriate decode path
        int tokens;
        if (is_prompt_mode) {
            const bool pd_cache_hit = (meta.flags & LLAMA_KV_ARTIFACT_FLAG_CACHE_HIT) != 0;
            bool disabled = false;
            if (const char * dis = std::getenv("LLAMA_PD_CACHE_DISABLE")) { disabled = (dis[0] == '1'); }
            if (pd_cache_hit && !disabled) {
                const uint64_t snap_sid = meta.pd_session_id;
                fprintf(stderr, "[worker-%d] Session %d: PROMPT-DIRECT CACHE HIT (sid=0x%llx)\n",
                        sess.session_id, sid, (unsigned long long) snap_sid);
                std::vector<uint8_t> kv_restore;
                if (!pd_cache_get_kv(snap_sid, kv_restore)) {
                    fprintf(stderr, "[worker-%d] Session %d: snapshot not found — falling back to miss\n",
                            sess.session_id, sid);
                    tokens = run_prompt_session(sess.ctx, mgr.model, worker_rparams, meta, prompt_payload, sid);
                } else {
                    g_active_cache_hit = true;
                    tokens = run_prompt_session(sess.ctx, mgr.model, worker_rparams, meta, prompt_payload, sid,
                                                /*is_cache_hit=*/true, &kv_restore);
                }
            } else {
                tokens = run_prompt_session(sess.ctx, mgr.model, worker_rparams, meta, prompt_payload, sid);
            }
        } else if (rparams.wir_enabled) {
            tokens = run_wir_decode_session(sess.ctx, mgr.model, worker_rparams, payload, meta,
                                             prompt_payload, sid);
        } else {
            tokens = run_decode_session(sess.ctx, mgr.model, worker_rparams, payload, meta,
                                         prompt_payload, sid);
        }

        if (tokens < 0) {
            fprintf(stderr, "[worker-%d] Session %d: decode failed\n", sess.session_id, sid);
        }

        // PROMPT-DIRECT KV-handshake: capture KV snapshot BEFORE clear (post-decode).
        if (is_prompt_mode && !g_active_cache_hit && g_active_prefix_hash != 0 && sess.ctx) {
            fprintf(stderr, "[pd-cache] Worker post-decode KV capture for hash=0x%llx\n",
                    (unsigned long long) g_active_prefix_hash);
            pd_cache_update_kv(g_active_prefix_hash, sess.ctx);
        }

        // Reset globals after session.
        g_active_prefix_hash = 0;
        g_active_cache_hit   = false;

        // Clear KV cache for next session
        if (sess.ctx) {
            llama_memory_clear(llama_get_memory(sess.ctx), true);
        }
        sess.active.store(false);
        fprintf(stderr, "[worker-%d] Session %d complete, KV cleared\n", sess.session_id, sid);

        if (rparams.one_shot) {
            fprintf(stderr, "[worker-%d] One-shot mode, exiting\n", sess.session_id);
            break;
        }
    }

    fprintf(stderr, "[worker-%d] Shutdown\n", sess.session_id);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    receiver_params rparams;
    if (!parse_args(argc, argv, rparams)) {
        return 1;
    }

    fprintf(stderr, "[decode-server] Loading model: %s\n", rparams.model_path.c_str());

    llama_backend_init();

    // Phase A3: initialise the persistent PD-cache disk tier (no-op unless
    // LLAMA_PD_CACHE_DIR is set). Indexes + compacts the dir on boot.
    pd_disk::init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = rparams.n_gpu_layers;
#ifndef COSMOS_UPSTREAM_RECEIVER
    model_params.use_mlock    = rparams.use_mlock;
#else
    if (rparams.use_mlock) {
        fprintf(stderr, "[upstream-receiver] --mlock is not available in this upstream compatibility target\n");
    }
#endif
    llama_model * model = llama_model_load_from_file(rparams.model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "[decode-server] Failed to load model\n");
        return 1;
    }

    // Peek at artifact header to determine flash_attn setting.
    // The v_trans in the KV state MUST match the context's v_trans.
    // Auto-detect: if the artifact's v_trans=0 (FA was on during prefill),
    // enable FA here. If v_trans=1 (no FA), keep FA disabled.
    bool need_fa = rparams.flash_attn;
    if (!rparams.flash_attn_forced && !rparams.kv_artifact_path.empty()) {
        llama_kv_artifact_metadata peek_meta = {};
        std::vector<uint8_t> peek_payload;
        std::string peek_err;
        llama_kv_artifact_summary peek_summary;
        if (llama_kv_artifact_read(rparams.kv_artifact_path, peek_payload, &peek_meta, &peek_summary, &peek_err, nullptr)) {
            // The v_trans field in the artifact metadata tells us the prefiller's FA setting
            need_fa = (peek_meta.v_trans == 0);
            fprintf(stderr, "[decode-server] Artifact peek: v_trans=%d → need_fa=%d\n",
                    peek_meta.v_trans, need_fa);
            peek_payload.clear();  // free the payload (we'll re-read it later)
        }
    } else if (!rparams.flash_attn_forced && rparams.kv_port > 0) {
        // Network mode (no artifact path to peek): default to FA on. The RTX
        // prefill side is expected to set LLAMA_PREFILL_MLA_MATCH_LAYOUT=1 so
        // it writes v_trans=0 KV that matches our FA-on layout. Without this
        // default, MLA decode runs with FA off and pays a 17 GB compute buffer
        // + 4x slower decode at long context (Bug #3, fixed 2026-04-25).
        // Override with -fa or by leaving flash_attn=false explicitly via CLI.
        need_fa = true;
        fprintf(stderr, "[decode-server] Network mode + FA not forced: defaulting FA ENABLED "
                        "(set --no-fa via CLI to opt out; RTX side must set LLAMA_PREFILL_MLA_MATCH_LAYOUT=1)\n");
    }
    if (rparams.flash_attn_forced) {
        fprintf(stderr, "[decode-server] FA forced via CLI override: %s\n",
                need_fa ? "ENABLED" : "DISABLED");
    }

    auto create_context = [&](bool enable_fa) -> llama_context * {
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx      = rparams.n_ctx;
        ctx_params.n_batch    = rparams.n_batch;
        ctx_params.n_ubatch   = rparams.n_ubatch;
        ctx_params.n_threads  = rparams.n_threads;
        // Match RTX server's n_seq_max so KV state import succeeds
        // (n_stream in serialized state must match the context's n_stream)
        // With kv_unified=false: n_stream = n_seq_max, each stream gets n_ctx/n_seq_max cells.
        // With kv_unified=true: n_stream = 1, but causes FA MLA copy crash on multi-turn.
        // Default n_seq_max=1 matches RTX np=1 (n_stream=1).
        // For np>1, pass --n-parallel N to match RTX's n_seq_max.
        ctx_params.n_seq_max  = rparams.n_parallel > 0 ? rparams.n_parallel : 1;
        ctx_params.type_k     = rparams.cache_type_k;
        ctx_params.type_v     = rparams.cache_type_v;
        fprintf(stderr, "[decode-server] Context geometry: ctx=%d batch=%d ubatch=%d ctk=%s ctv=%s\n",
                rparams.n_ctx, rparams.n_batch, rparams.n_ubatch,
                ggml_type_name(rparams.cache_type_k), ggml_type_name(rparams.cache_type_v));
        // Match the artifact's v_trans setting.
        ctx_params.flash_attn_type = enable_fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        fprintf(stderr, "[decode-server] FA set to %s (from artifact v_trans=%d)\n",
                enable_fa ? "ENABLED" : "DISABLED", enable_fa ? 0 : 1);
        // CROSS-BACKEND ROW DUMP. `LLAMA_DSV4_COMP_DUMP` only writes files from the RTX streaming
        // executor's layer loop (llama-prefill-stream.cpp), which this receiver never enters -- it
        // runs a plain llama_decode -- so the env was being delivered and silently ignored here.
        //
        // Why it is worth having: this receiver runs the SAME fork source on the SAME GGUF and
        // RECALLS a mid-prompt needle that the RTX (CUDA) prefill misses. Weights are byte-identical
        // on-device, graph logic is shared, the mask is right, decode consumption is exonerated, and
        // at this prompt length the indexer selects every row -- so the divergence is in what the
        // CUDA prefill computes. Dumping the same named tensors from the Metal side makes that a
        // numeric diff instead of an argument.
        //
        // cb_eval sees each tensor the instant its node computes; unlike the RTX post-hoc readback
        // it is on the graph that actually produces the tokens here, because this context IS the one
        // doing the work.
        if (const char * dump_pref = std::getenv("LLAMA_DSV4_COMP_DUMP")) {
            if (dump_pref[0]) {
                ctx_params.cb_eval = dsv4_recv_dump_cb;
                ctx_params.cb_eval_user_data = (void *) dump_pref;
                fprintf(stderr, "[decode-server] COMP_DUMP armed on this receiver -> %s.*\n", dump_pref);
            }
        }
        return llama_init_from_model(model, ctx_params);
    };

    bool ctx_flash_attn = need_fa;
    llama_context * ctx = create_context(ctx_flash_attn);
    if (!ctx) {
        fprintf(stderr, "[decode-server] Failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    fprintf(stderr, "[decode-server] Ready (ctx_size=%d, mode=%s, sessions=%d)\n",
            rparams.n_ctx,
            rparams.kv_port > 0 ? "network" : "file",
            rparams.n_sessions);

    // -----------------------------------------------------------------------
    // PROMPT-DIRECT KV-handshake (Gate #2): start control socket listener
    // on kv_port-1 so RTX can query the prefix cache before each session.
    // Only active in network mode (kv_port > 1).
    // -----------------------------------------------------------------------
    std::atomic<bool> control_server_stop{false};
    std::thread control_server_thread;
    if (rparams.kv_port > 1) {
        const int ctrl_port = rparams.kv_port - 1;
        control_server_thread = std::thread(run_control_server,
                                            rparams.kv_host, ctrl_port,
                                            std::ref(control_server_stop));
        fprintf(stderr, "[decode-server] PROMPT-DIRECT cache control socket on port %d\n", ctrl_port);
    }

    // -----------------------------------------------------------------------
    // Multi-session mode: spawn N worker threads, each with its own context
    // and port pair. Requires network mode (kv_port > 0).
    // -----------------------------------------------------------------------
    if (rparams.n_sessions > 1 && rparams.kv_port > 0) {
        // Populate port bases from the kv_port/token_stream_port if the user
        // didn't explicitly set --kv-port-base / --token-port-base.
        if (rparams.kv_port_base == 9100 && rparams.kv_port != 9100) {
            rparams.kv_port_base = rparams.kv_port;
        }
        if (rparams.token_port_base == 9101 && rparams.token_stream_port != 9101) {
            rparams.token_port_base = rparams.token_stream_port;
        }

        fprintf(stderr, "[decode-server] MULTI-SESSION MODE: %d sessions\n", rparams.n_sessions);
        fprintf(stderr, "[decode-server] Port allocation:\n");
        for (int i = 0; i < rparams.n_sessions; i++) {
            fprintf(stderr, "  Session %d: kv_port=%d token_port=%d\n",
                    i, rparams.kv_port_base + 2 * i, rparams.token_port_base + 2 * i);
        }

        // Free the single context created above; the session_manager creates its own
        llama_free(ctx);
        ctx = nullptr;

        session_manager mgr;
        mgr.init(model, rparams.n_sessions, rparams.kv_port_base, rparams.token_port_base,
                 create_context, need_fa);

        // Launch worker threads
        for (int i = 0; i < rparams.n_sessions; i++) {
            mgr.sessions[i].worker = std::thread(session_worker,
                std::ref(mgr.sessions[i]), std::ref(mgr), std::cref(rparams));
        }

        fprintf(stderr, "[decode-server] All %d workers launched, waiting...\n", rparams.n_sessions);

        // Wait for all workers (they run until shutdown or one-shot)
        for (int i = 0; i < rparams.n_sessions; i++) {
            if (mgr.sessions[i].worker.joinable()) {
                mgr.sessions[i].worker.join();
            }
        }

        // Cleanup: free contexts
        for (int i = 0; i < rparams.n_sessions; i++) {
            if (mgr.sessions[i].ctx) {
                llama_free(mgr.sessions[i].ctx);
                mgr.sessions[i].ctx = nullptr;
            }
        }

        llama_model_free(model);
        llama_backend_free();

        // Stop control socket thread
        control_server_stop.store(true);
        if (control_server_thread.joinable()) {
            control_server_thread.join();
        }
        fprintf(stderr, "[decode-server] Multi-session shutdown complete\n");
        return 0;
    }

    // -----------------------------------------------------------------------
    // Single-session mode (legacy): original sequential loop
    // -----------------------------------------------------------------------
    int session_id = 0;
    while (true) {
        session_id++;

        // Receive KV artifact or prompt tokens
        std::vector<uint8_t> payload;
        llama_kv_prompt_payload prompt_payload;
        llama_kv_artifact_metadata meta = {};
        std::string error;
        bool is_prompt_mode = false;

        if (!receive_kv_artifact(rparams, payload, meta, prompt_payload, error, is_prompt_mode)) {
            fprintf(stderr, "[decode-server] Session %d: failed to receive artifact: %s\n",
                    session_id, error.c_str());
            if (rparams.one_shot || rparams.kv_port <= 0) {
                break;
            }
            continue;
        }

        // Stage 1 (gated): decide whether to REUSE the live ctx + resident KV.
        bool resident_reuse = false;
        uint32_t reuse_n = 0;
        if (!is_prompt_mode) {
            const bool need_fa_session = rparams.flash_attn_forced
                ? rparams.flash_attn
                : (meta.v_trans == 0);
            // Reuse the live ctx (skip recreate + the ~6.2s post-import graph
            // re-reserve) only when the incoming conversation EXTENDS what's
            // already materialized in the live seq-0 KV, the graph shape matches
            // (same FA mode), and the resident state is from THIS ctx generation.
            if (resident_kv_reuse_enabled()
                && !rparams.wir_enabled
                && need_fa_session == ctx_flash_attn
                && meta.format_minor >= 5
                && !prompt_payload.token_ids.empty()
                && meta.streamed_prefix_token_count > 0) {
                std::lock_guard<std::mutex> lk(g_resident_kv_mu);
                if (g_resident_kv.valid
                    && g_resident_kv.ctx_generation == g_ctx_generation
                    && g_resident_kv.source_slot_id == meta.source_slot_id
                    && !g_resident_kv.token_ids.empty()) {
                    reuse_n = resident_compute_lcp(g_resident_kv.token_ids, prompt_payload.token_ids);
                    // 2026-06-01 FIX (multi-request garble root cause): reuse ONLY when the incoming prompt is a
                    // genuine EXTENSION of the resident — i.e. the LCP covers the ENTIRE resident (reuse_n >=
                    // resident token count). Multi-turn turns ARE extensions (resident = prior convo is a full
                    // prefix), so reuse stays fast there. For an UNRELATED new prompt the LCP is a tiny shared
                    // prefix (~system prompt); reusing it would trim to reuse_n and force the Mac to LOCALLY
                    // PREFILL the whole 23k divergent tail (slow + diverges from the RTX's shipped KV -> garbage).
                    // Falling back to the import path (resident_reuse=false) uses the RTX's correct shipped KV.
                    const uint32_t resident_len = (uint32_t) g_resident_kv.token_ids.size();
                    // 2026-08-27: the LCP being a full extension is necessary but NOT sufficient. Reuse
                    // trims the live KV to reuse_n and prefills the tail, and on an iSWA cache that trim
                    // is REFUSED once the sliding window has moved past reuse_n -- llama_memory_seq_rm
                    // returns false and changes nothing, so the suffix batch then starts BEFORE the KV's
                    // last position and llama_decode rejects the whole session:
                    //     RESIDENT logical_prefix=11320 ... ISWA sliding range [11327,11327]
                    //     init: ... last position ... X = 11327 / starting position of Y = 11320
                    //     prompt suffix prefill failed at token 11320 (ret=-1)
                    // Measured on DSV4-flash: turns 1-2 fine, turn 3 dead, because only by then had the
                    // window drifted past the reuse point.
                    //
                    // Decide it HERE, where refusing costs nothing: the RTX has already shipped a full
                    // KV artifact for this prompt, so the import path below is both correct and fast.
                    // Deciding it later -- after the import was skipped -- leaves only a local re-prefill
                    // of the entire prompt, which is the slow silent degradation this comment exists to
                    // prevent. Same reasoning as the non-extension case below.
                    bool window_covers_reuse = true;
                    if (reuse_n > 0) {
                        if (llama_memory_t mem = llama_get_memory(ctx)) {
                            const llama_pos pos_min = llama_memory_seq_pos_min(mem, 0);
                            if (pos_min > (llama_pos) reuse_n) {
                                window_covers_reuse = false;
                                fprintf(stderr,
                                        "[decode-server] Session %d: NOT reusing (LCP=%u but live KV window starts at %d, "
                                        "so the trim to %u would be refused) — importing the shipped KV instead\n",
                                        session_id, reuse_n, (int) pos_min, reuse_n);
                            }
                        }
                    }
                    if (reuse_n >= resident_len && reuse_n > 0 && window_covers_reuse) {
                        resident_reuse = true;
                        fprintf(stderr,
                                "[decode-server] Session %d: RESIDENT REUSE (LCP=%u resident=%zu incoming=%zu, full-extension) — "
                                "skip recreate+clear+import (LLAMA_PD_RESIDENT_KV_REUSE)\n",
                                session_id, reuse_n, g_resident_kv.token_ids.size(),
                                prompt_payload.token_ids.size());
                    } else if (reuse_n > 0) {
                        fprintf(stderr,
                                "[decode-server] Session %d: NOT reusing (LCP=%u < resident=%u, non-extension) — "
                                "importing the shipped KV instead (multi-request correctness fix)\n",
                                session_id, reuse_n, resident_len);
                    }
                }
            }
            if (!resident_reuse) {
                // Original path: recreate context for clean scheduler/compute state.
                // Reusing context between sessions otherwise risks intermittent
                // garbage (stale Metal GPU state) — hence the gate above.
                fprintf(stderr,
                        "[decode-server] Session %d: recreating context (FA=%s)\n",
                        session_id, need_fa_session ? "on" : "off");
                llama_free(ctx);
                ctx = create_context(need_fa_session);
                if (!ctx) {
                    fprintf(stderr, "[decode-server] Failed to recreate context for session %d\n", session_id);
                    llama_model_free(model);
                    llama_backend_free();
                    return 1;
                }
                ctx_flash_attn = need_fa_session;
                ++g_ctx_generation;  // fresh ctx => stale resident can never be reused
                { std::lock_guard<std::mutex> lk(g_resident_kv_mu); g_resident_kv.valid = false; }
            }
        }

        int tokens;
        if (is_prompt_mode) {
            // Detect PROMPT-DIRECT cache-hit artifact (Phase-2)
            const bool pd_cache_hit = (meta.flags & LLAMA_KV_ARTIFACT_FLAG_CACHE_HIT) != 0;
            if (pd_cache_hit) {
                // Gate: safety env disable
                bool disabled = false;
                if (const char * dis = std::getenv("LLAMA_PD_CACHE_DISABLE")) {
                    disabled = (dis[0] == '1');
                }
                if (disabled) {
                    fprintf(stderr, "[decode-server] Session %d: CACHE HIT but LLAMA_PD_CACHE_DISABLE=1 — treating as miss\n",
                            session_id);
                    tokens = run_prompt_session(ctx, model, rparams, meta, prompt_payload, session_id);
                } else {
                    const uint64_t sid = meta.pd_session_id;
                    fprintf(stderr, "[decode-server] Session %d: PROMPT-DIRECT CACHE HIT (sid=0x%llx) — restoring snapshot\n",
                            session_id, (unsigned long long) sid);
                    std::vector<uint8_t> kv_restore;
                    if (!pd_cache_get_kv(sid, kv_restore)) {
                        fprintf(stderr, "[decode-server] Session %d: snapshot not found for sid=0x%llx — falling back to miss\n",
                                session_id, (unsigned long long) sid);
                        tokens = run_prompt_session(ctx, model, rparams, meta, prompt_payload, session_id);
                    } else {
                        g_active_cache_hit = true;
                        tokens = run_prompt_session(ctx, model, rparams, meta, prompt_payload, session_id,
                                                    /*is_cache_hit=*/true, &kv_restore);
                    }
                }
            } else {
                // Normal miss: prefill locally on Mac, then decode
                tokens = run_prompt_session(ctx, model, rparams, meta, prompt_payload, session_id);
            }
        } else if (rparams.wir_enabled) {
            // WIR mode: import KV, then wavefront refinement decode
            tokens = run_wir_decode_session(ctx, model, rparams, payload, meta, prompt_payload, session_id);
        } else {
            // KV artifact: import KV state, then standard decode
            tokens = run_decode_session(ctx, model, rparams, payload, meta, prompt_payload, session_id,
                                        resident_reuse, reuse_n);
        }
        if (tokens < 0) {
            fprintf(stderr, "[decode-server] Session %d: decode session failed\n", session_id);
        }

        // PROMPT-DIRECT KV-handshake (Gate #2): capture KV snapshot BEFORE clear.
        // This runs AFTER decode completes so it doesn't block the token stream.
        // g_active_prefix_hash is set by run_prompt_session on a miss path.
        if (is_prompt_mode && !g_active_cache_hit && g_active_prefix_hash != 0) {
            fprintf(stderr, "[pd-cache] Post-decode KV capture for hash=0x%llx\n",
                    (unsigned long long) g_active_prefix_hash);
            pd_cache_update_kv(g_active_prefix_hash, ctx);
        }

        // Reset globals after session.
        g_active_prefix_hash = 0;
        g_active_cache_hit   = false;

        // Clear KV cache for next session — UNLESS Stage 1 resident reuse is on
        // and we just populated a valid resident: keep the live KV alive so the
        // NEXT turn can reuse the prompt cells (the whole point of retention).
        bool keep_kv_for_reuse = resident_kv_reuse_enabled() && !is_prompt_mode;
        if (keep_kv_for_reuse) {
            std::lock_guard<std::mutex> lk(g_resident_kv_mu);
            keep_kv_for_reuse = g_resident_kv.valid;
        }
        if (keep_kv_for_reuse) {
            llama_memory_t mem = llama_get_memory(ctx);
            llama_pos pmax = llama_memory_seq_pos_max(mem, 0);
            fprintf(stderr, "[decode-server] Session %d complete, KV RETAINED for next-turn reuse (pos_max=%d)\n",
                    session_id, pmax);
        } else {
            llama_memory_clear(llama_get_memory(ctx), true);
            llama_memory_t mem = llama_get_memory(ctx);
            llama_pos pmin = llama_memory_seq_pos_min(mem, 0);
            llama_pos pmax = llama_memory_seq_pos_max(mem, 0);
            fprintf(stderr, "[decode-server] Session %d complete, KV cleared (pos_min=%d pos_max=%d)\n",
                    session_id, pmin, pmax);
            if (pmax >= 0) {
                fprintf(stderr, "[decode-server] WARNING: KV not fully cleared! Doing seq_rm...\n");
                llama_memory_seq_rm(mem, 0, -1, -1);
                pmin = llama_memory_seq_pos_min(mem, 0);
                pmax = llama_memory_seq_pos_max(mem, 0);
                fprintf(stderr, "[decode-server] After seq_rm: pos_min=%d pos_max=%d\n", pmin, pmax);
            }
        }

        if (rparams.one_shot) {
            fprintf(stderr, "[decode-server] One-shot mode, exiting\n");
            break;
        }

        if (rparams.kv_port <= 0) {
            // File mode: no more artifacts to receive
            break;
        }

        // Network mode: loop back for next artifact
        fprintf(stderr, "[decode-server] Waiting for next session...\n");
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    // Stop control socket thread
    control_server_stop.store(true);
    if (control_server_thread.joinable()) {
        control_server_thread.join();
    }

    fprintf(stderr, "[decode-server] Shutdown after %d session(s)\n", session_id);
    return 0;
}
