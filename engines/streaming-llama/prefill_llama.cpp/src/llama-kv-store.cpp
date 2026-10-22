#include "llama-kv-store.h"

// xxhash header-only: XXH_INLINE_ALL makes all symbols static/inline so this is
// ODR-safe alongside the other TUs that include it (llama-tb-transport.cpp,
// server-context.cpp). Used for the 256-token rolling chain hash that mirrors
// tools/mlx_vlm_decode/prompt_cache_store.py.
#define XXH_INLINE_ALL
#include "../examples/gguf-hash/deps/xxhash/xxhash.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

// KV_STORE_BUILD_MARKER_A1A2_2026_06_10 — grep the linked binary for this string
// to confirm this TU's object actually made it into the build (per the
// touch+grep-after-.cuh discipline; here it just proves the .cpp linked).
static const char * const kv_store_build_marker = "KV_STORE_BUILD_MARKER_A1A2_2026_06_10";

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Small helpers.
// ---------------------------------------------------------------------------

namespace {

constexpr uint32_t CHUNK_SIZE_TOKENS = 256; // plan §3 / prompt_cache_store.py

double now_seconds() {
    return double(std::time(nullptr));
}

std::string to_hex64(uint64_t v) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) v);
    return std::string(buf);
}

bool env_dir(std::string & out) {
    const char * e = std::getenv("LLAMA_KV_CACHE_DIR");
    if (e && e[0] != '\0') {
        out = e;
        return true;
    }
    return false;
}

double env_max_gb() {
    const char * e = std::getenv("LLAMA_KV_CACHE_MAX_GB");
    if (e && e[0] != '\0') {
        char * end = nullptr;
        double v = std::strtod(e, &end);
        if (end != e && v > 0.0) {
            return v;
        }
    }
    return 200.0; // plan §3 default
}

// --- minimal JSON helpers (no external dep) ---

std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

// Extract a string-valued field "key":"..." (no escapes inside for our keys).
bool json_get_str(const std::string & row, const std::string & key, std::string & out) {
    std::string pat = "\"" + key + "\":\"";
    size_t p = row.find(pat);
    if (p == std::string::npos) return false;
    p += pat.size();
    size_t q = row.find('"', p);
    if (q == std::string::npos) return false;
    out = row.substr(p, q - p);
    return true;
}

bool json_get_u64(const std::string & row, const std::string & key, uint64_t & out) {
    std::string pat = "\"" + key + "\":";
    size_t p = row.find(pat);
    if (p == std::string::npos) return false;
    p += pat.size();
    if (p < row.size() && row[p] == '"') return false; // string, not number
    out = std::strtoull(row.c_str() + p, nullptr, 10);
    return true;
}

bool json_get_double(const std::string & row, const std::string & key, double & out) {
    std::string pat = "\"" + key + "\":";
    size_t p = row.find(pat);
    if (p == std::string::npos) return false;
    p += pat.size();
    out = std::strtod(row.c_str() + p, nullptr);
    return true;
}

// Parse a flat integer array field "key":[a,b,c] into `out`. T is the element type.
template <typename T>
bool json_get_int_array(const std::string & row, const std::string & key, std::vector<T> & out) {
    std::string pat = "\"" + key + "\":[";
    size_t p = row.find(pat);
    if (p == std::string::npos) return false;
    p += pat.size();
    size_t q = row.find(']', p);
    if (q == std::string::npos) return false;
    out.clear();
    const char * s = row.c_str() + p;
    const char * end = row.c_str() + q;
    while (s < end) {
        char * np = nullptr;
        // unsigned for chain hashes, signed for token ids — strtoull then cast
        unsigned long long v = std::strtoull(s, &np, 10);
        if (np == s) {
            // try signed (token ids can be negative in principle)
            long long sv = std::strtoll(s, &np, 10);
            if (np == s) break;
            out.push_back((T) sv);
        } else {
            out.push_back((T) v);
        }
        s = np;
        while (s < end && (*s == ',' || *s == ' ')) ++s;
    }
    return true;
}

std::string json_int_array_u64(const std::vector<uint64_t> & v) {
    std::ostringstream os;
    os << '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ',';
        os << v[i];
    }
    os << ']';
    return os.str();
}

std::string json_int_array_i32(const std::vector<int32_t> & v) {
    std::ostringstream os;
    os << '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ',';
        os << (long long) v[i];
    }
    os << ']';
    return os.str();
}

uint32_t lcp_tokens(const std::vector<int32_t> & a, const std::vector<int32_t> & b) {
    uint32_t n = (uint32_t) std::min(a.size(), b.size());
    uint32_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

} // namespace

// ---------------------------------------------------------------------------
// model_fp / chain hash.
// ---------------------------------------------------------------------------

uint64_t llama_kv_store::compute_model_fp(const std::string & model_desc,
                                          const std::string & model_path,
                                          uint16_t type_k, uint16_t type_v,
                                          uint8_t v_trans, uint8_t is_mla,
                                          uint8_t flash_layout) {
    // arch/params/quant identity from the model description string (the same
    // string the server hashes for pd_model_fp). 0 if empty.
    uint64_t artifact_model_fingerprint =
        model_desc.empty() ? 0 : XXH64(model_desc.data(), model_desc.size(), /*seed=*/0);

    // file identity: XXH64 over path + size + mtime (cheap, no content read).
    uint64_t file_id = 0;
    {
        std::string blob = model_path;
        std::error_code ec;
        uintmax_t sz = 0;
        int64_t mt = 0;
        if (!model_path.empty()) {
            sz = fs::file_size(model_path, ec);
            if (ec) sz = 0;
            ec.clear();
            auto t = fs::last_write_time(model_path, ec);
            if (!ec) {
                mt = (int64_t) t.time_since_epoch().count();
            }
        }
        blob.push_back('\0');
        blob.append(reinterpret_cast<const char *>(&sz), sizeof(sz));
        blob.append(reinterpret_cast<const char *>(&mt), sizeof(mt));
        file_id = XXH64(blob.data(), blob.size(), /*seed=*/0);
    }

    // KV-layout bits packed into a small word, then hashed for diffusion.
    uint64_t layout_word =
        (uint64_t) type_k |
        ((uint64_t) type_v << 16) |
        ((uint64_t) v_trans << 32) |
        ((uint64_t) is_mla << 40) |
        ((uint64_t) flash_layout << 48);
    uint64_t layout_id = XXH64(&layout_word, sizeof(layout_word), /*seed=*/0);

    return artifact_model_fingerprint ^ file_id ^ layout_id;
}

std::vector<uint64_t> llama_kv_store::chunk_hashes(const std::vector<int32_t> & token_ids,
                                                   uint64_t seed_model_fp) {
    // Mirrors prompt_cache_store.py::chunk_hashes — only WHOLE 256-tok chunks are
    // chained; a trailing partial chunk is intentionally not a link (longest-
    // prefix operates at chunk granularity, the token-exact verify covers the
    // partial remainder). Token ids packed as little-endian uint32 (same as the
    // python struct.pack("<I")).
    std::vector<uint64_t> out;
    uint64_t h = seed_model_fp;
    const size_t n_full = token_ids.size() / CHUNK_SIZE_TOKENS;
    std::vector<uint32_t> buf(CHUNK_SIZE_TOKENS);
    for (size_t c = 0; c < n_full; ++c) {
        const size_t lo = c * CHUNK_SIZE_TOKENS;
        for (size_t i = 0; i < CHUNK_SIZE_TOKENS; ++i) {
            buf[i] = (uint32_t) token_ids[lo + i];
        }
        h = XXH64(buf.data(), buf.size() * sizeof(uint32_t), /*seed=*/h);
        out.push_back(h);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Construction + boot.
// ---------------------------------------------------------------------------

llama_kv_store::llama_kv_store(const std::string & dir, double max_gb) {
    std::string d = dir;
    if (d.empty()) {
        env_dir(d); // empty/unset => disabled
    }
    if (d.empty()) {
        enabled_ = false; // feature fully off, zero behavior change
        return;
    }
    dir_ = d;
    double gb = max_gb > 0.0 ? max_gb : env_max_gb();
    max_bytes_ = (uint64_t) (gb * 1024.0 * 1024.0 * 1024.0);

    std::error_code ec;
    fs::create_directories(dir_, ec);
    if (ec) {
        fprintf(stderr, "[kv-store] disabled: cannot create dir %s: %s\n",
                dir_.c_str(), ec.message().c_str());
        fflush(stderr);
        enabled_ = false;
        return;
    }
    enabled_ = true;
    load_and_compact();
    fprintf(stderr,
            "[kv-store] enabled dir=%s max_gb=%.1f entries=%llu bytes=%.1fMB "
            "(marker=%s)\n",
            dir_.c_str(), gb, (unsigned long long) index_.size(),
            current_bytes() / (1024.0 * 1024.0), kv_store_build_marker);
    fflush(stderr);
}

std::string llama_kv_store::index_path() const {
    return (fs::path(dir_) / "index.jsonl").string();
}

std::string llama_kv_store::model_dir(uint64_t model_fp) const {
    return (fs::path(dir_) / to_hex64(model_fp)).string();
}

std::string llama_kv_store::artifact_path(uint64_t model_fp, const std::string & chain_hash_hex) const {
    return (fs::path(model_dir(model_fp)) / (chain_hash_hex + ".kva")).string();
}

void llama_kv_store::load_and_compact() {
    index_.clear();
    std::ifstream ifs(index_path());
    if (!ifs.is_open()) {
        return; // fresh store
    }
    std::string line;
    bool need_rewrite = false;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        llama_kv_store_entry e;
        std::string mh, ch;
        if (!json_get_str(line, "model_fp", mh) || !json_get_str(line, "chain_hash", ch)) {
            need_rewrite = true;
            continue;
        }
        e.model_fp_hex   = mh;
        e.chain_hash_hex = ch;
        uint64_t u = 0;
        if (json_get_u64(line, "n_tokens", u))       e.n_tokens = (uint32_t) u;
        if (json_get_u64(line, "streamed_prefix", u)) e.streamed_prefix = (uint32_t) u;
        if (json_get_u64(line, "bytes", u))          e.bytes = u;
        if (json_get_u64(line, "crc32", u))          e.crc32 = (uint32_t) u;
        double lu = 0;
        if (json_get_double(line, "last_used", lu))  e.last_used = lu;
        json_get_int_array<uint64_t>(line, "chunk_hashes", e.chunk_hashes);
        json_get_int_array<int32_t>(line, "token_ids", e.token_ids);

        // drop rows whose artifact file is missing (atomic-write crash / manual rm)
        std::string ap = (fs::path(dir_) / mh / (ch + ".kva")).string();
        std::error_code ec;
        if (!fs::exists(ap, ec)) {
            need_rewrite = true;
            continue;
        }
        // de-dup: keep the row with the larger n_tokens for a given key
        bool replaced = false;
        for (auto & ex : index_) {
            if (ex.model_fp_hex == e.model_fp_hex && ex.chain_hash_hex == e.chain_hash_hex) {
                need_rewrite = true;
                if (e.n_tokens >= ex.n_tokens) ex = e;
                replaced = true;
                break;
            }
        }
        if (!replaced) index_.push_back(std::move(e));
    }
    ifs.close();
    if (need_rewrite) {
        rewrite_index();
    }
    evict_to_cap();
}

void llama_kv_store::rewrite_index() const {
    if (!enabled_) return;
    std::string tmp = index_path() + ".tmp";
    {
        std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
        if (!ofs.is_open()) {
            fprintf(stderr, "[kv-store] rewrite_index: cannot open %s\n", tmp.c_str());
            fflush(stderr);
            return;
        }
        for (const auto & e : index_) {
            std::ostringstream row;
            row << "{\"model_fp\":\"" << e.model_fp_hex << "\","
                << "\"chain_hash\":\"" << e.chain_hash_hex << "\","
                << "\"n_tokens\":" << e.n_tokens << ","
                << "\"streamed_prefix\":" << e.streamed_prefix << ","
                << "\"bytes\":" << e.bytes << ","
                << "\"last_used\":" << (long long) e.last_used << ","
                << "\"crc32\":" << e.crc32 << ","
                << "\"chunk_hashes\":" << json_int_array_u64(e.chunk_hashes) << ","
                << "\"token_ids\":" << json_int_array_i32(e.token_ids) << "}";
            ofs << row.str() << "\n";
        }
        ofs.flush();
    }
    std::error_code ec;
    fs::rename(tmp, index_path(), ec);
    if (ec) {
        // fall back to copy+remove
        fs::copy_file(tmp, index_path(), fs::copy_options::overwrite_existing, ec);
        fs::remove(tmp, ec);
    }
}

void llama_kv_store::append_index_row(const llama_kv_store_entry & e) const {
    std::ofstream ofs(index_path(), std::ios::binary | std::ios::app);
    if (!ofs.is_open()) return;
    std::ostringstream row;
    row << "{\"model_fp\":\"" << e.model_fp_hex << "\","
        << "\"chain_hash\":\"" << e.chain_hash_hex << "\","
        << "\"n_tokens\":" << e.n_tokens << ","
        << "\"streamed_prefix\":" << e.streamed_prefix << ","
        << "\"bytes\":" << e.bytes << ","
        << "\"last_used\":" << (long long) e.last_used << ","
        << "\"crc32\":" << e.crc32 << ","
        << "\"chunk_hashes\":" << json_int_array_u64(e.chunk_hashes) << ","
        << "\"token_ids\":" << json_int_array_i32(e.token_ids) << "}";
    ofs << row.str() << "\n";
}

// ---------------------------------------------------------------------------
// I/O — sequential/unbuffered artifact read (NEVER mmap; page-cache thrash).
// ---------------------------------------------------------------------------

bool llama_kv_store::read_artifact_file(const std::string & path, std::vector<uint8_t> & out) const {
#ifdef _WIN32
    // FILE_FLAG_SEQUENTIAL_SCAN hint so the cache manager evicts aggressively
    // and does not retain the (large) artifact in the page cache (risk §7.5).
    FILE * f = nullptr;
    // _fsopen with "rbS" -> sequential scan optimization on Windows CRT.
    f = fopen(path.c_str(), "rbS");
    if (!f) f = fopen(path.c_str(), "rb");
#else
    FILE * f = fopen(path.c_str(), "rb");
#endif
    if (!f) return false;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return false; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return false; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return false; }
    out.resize((size_t) sz);
    size_t rd = sz > 0 ? fread(out.data(), 1, (size_t) sz, f) : 0;
    fclose(f);
    if (rd != (size_t) sz) { out.clear(); return false; }
    return true;
}

// ---------------------------------------------------------------------------
// put (write hook).
// ---------------------------------------------------------------------------

bool llama_kv_store::put(uint64_t model_fp,
                         const std::vector<int32_t> & canonical_tokens,
                         uint32_t streamed_prefix,
                         uint32_t artifact_crc32,
                         const std::vector<uint8_t> & artifact_bytes) {
    if (!enabled_ || canonical_tokens.empty() || artifact_bytes.empty()) {
        return false;
    }

    std::vector<uint64_t> chashes = chunk_hashes(canonical_tokens, model_fp);
    std::string chain_hex;
    if (!chashes.empty()) {
        chain_hex = to_hex64(chashes.back());
    } else {
        // sub-chunk prompt: hash the whole token sequence (matched via short scan)
        std::vector<uint32_t> buf(canonical_tokens.size());
        for (size_t i = 0; i < canonical_tokens.size(); ++i) buf[i] = (uint32_t) canonical_tokens[i];
        chain_hex = to_hex64(XXH64(buf.data(), buf.size() * sizeof(uint32_t), model_fp));
    }
    std::string mh = to_hex64(model_fp);

    // write-once: if an equal-or-longer entry already exists at this key, touch.
    for (auto & ex : index_) {
        if (ex.model_fp_hex == mh && ex.chain_hash_hex == chain_hex &&
            ex.n_tokens >= (uint32_t) canonical_tokens.size()) {
            ex.last_used = now_seconds();
            return true;
        }
    }

    std::error_code ec;
    fs::create_directories(model_dir(model_fp), ec);
    if (ec) {
        fprintf(stderr, "[kv-store] put: cannot create %s: %s\n",
                model_dir(model_fp).c_str(), ec.message().c_str());
        fflush(stderr);
        return false;
    }

    // atomic write: tmp -> rename.
    std::string ap = artifact_path(model_fp, chain_hex);
    std::string tmp = ap + ".tmp";
    {
        std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
        if (!ofs.is_open()) {
            fprintf(stderr, "[kv-store] put: cannot open %s\n", tmp.c_str());
            fflush(stderr);
            return false;
        }
        ofs.write(reinterpret_cast<const char *>(artifact_bytes.data()),
                  (std::streamsize) artifact_bytes.size());
        if (!ofs) {
            fprintf(stderr, "[kv-store] put: write failed %s\n", tmp.c_str());
            fflush(stderr);
            ofs.close();
            fs::remove(tmp, ec);
            return false;
        }
        ofs.flush();
    }
    fs::rename(tmp, ap, ec);
    if (ec) {
        fs::copy_file(tmp, ap, fs::copy_options::overwrite_existing, ec);
        fs::remove(tmp, ec);
    }

    llama_kv_store_entry e;
    e.model_fp_hex   = mh;
    e.chain_hash_hex = chain_hex;
    e.n_tokens       = (uint32_t) canonical_tokens.size();
    e.streamed_prefix = streamed_prefix;
    e.bytes          = artifact_bytes.size();
    e.last_used      = now_seconds();
    e.crc32          = artifact_crc32;
    e.chunk_hashes   = std::move(chashes);
    e.token_ids      = canonical_tokens;

    // replace any shorter existing row at this exact key, else append.
    bool replaced = false;
    for (auto & ex : index_) {
        if (ex.model_fp_hex == mh && ex.chain_hash_hex == chain_hex) {
            ex = e;
            replaced = true;
            break;
        }
    }
    if (!replaced) index_.push_back(e);

    if (replaced) {
        rewrite_index();      // can't append over an existing key without dup
    } else {
        append_index_row(e);  // fast path: write-once append
    }
    stats_.writes++;
    evict_to_cap();

    fprintf(stderr,
            "[kv-store] STORED model_fp=%s chain=%s n_tokens=%u prefix=%u bytes=%.2fMB crc=%08x\n",
            mh.c_str(), chain_hex.c_str(), e.n_tokens, streamed_prefix,
            e.bytes / (1024.0 * 1024.0), artifact_crc32);
    fflush(stderr);
    return true;
}

// ---------------------------------------------------------------------------
// lookup (read hook) — longest-prefix + token-exact + CRC verify.
// ---------------------------------------------------------------------------

uint32_t llama_kv_store::lookup(uint64_t model_fp,
                                const std::vector<int32_t> & request_tokens,
                                uint32_t lcp_min,
                                llama_kv_store_hit & out) {
    if (!enabled_ || request_tokens.empty()) {
        return 0;
    }
    std::string mh = to_hex64(model_fp);
    std::vector<uint64_t> cand = chunk_hashes(request_tokens, model_fp);

    // A2 longest-prefix: walk the chunk chain from the deepest chunk back,
    // picking the deepest indexed entry whose chunk chain is a prefix of ours.
    const llama_kv_store_entry * best = nullptr;
    for (size_t depth = cand.size(); depth > 0; --depth) {
        std::string chx = to_hex64(cand[depth - 1]);
        for (const auto & e : index_) {
            if (e.model_fp_hex != mh || e.chain_hash_hex != chx) continue;
            if (e.chunk_hashes.size() < depth) continue;
            bool ok = true;
            for (size_t i = 0; i < depth; ++i) {
                if (e.chunk_hashes[i] != cand[i]) { ok = false; break; }
            }
            if (ok) { best = &e; break; }
        }
        if (best) break;
    }

    // Fallback: sub-chunk entries (< 256 tok, no chain link) via direct token
    // prefix scan — pick the longest usable prefix.
    if (!best) {
        for (const auto & e : index_) {
            if (e.model_fp_hex != mh) continue;
            if (e.n_tokens >= CHUNK_SIZE_TOKENS) continue;
            uint32_t l = lcp_tokens(request_tokens, e.token_ids);
            if (l >= lcp_min && (!best || e.n_tokens > best->n_tokens)) {
                best = &e;
            }
        }
    }
    if (!best) {
        stats_.misses++;
        return 0;
    }

    // TOKEN-EXACT verify (hash = accelerator, tokens = truth, risk §7.4).
    uint32_t lcp = lcp_tokens(request_tokens, best->token_ids);
    if (lcp < lcp_min) {
        stats_.misses++;
        return 0;
    }
    // need at least one new token for logits — handled by caller's threshold,
    // but never let the prefix == the entire request (no suffix to prefill).
    if (lcp >= (uint32_t) request_tokens.size()) {
        if (request_tokens.size() <= 1) { stats_.misses++; return 0; }
        lcp = (uint32_t) request_tokens.size() - 1;
        if (lcp < lcp_min) { stats_.misses++; return 0; }
    }

    // Read the artifact bytes (sequential, no mmap) + CRC verify.
    std::string ap = artifact_path(model_fp, best->chain_hash_hex);
    std::vector<uint8_t> bytes;
    if (!read_artifact_file(ap, bytes)) {
        fprintf(stderr, "[kv-store] lookup: read failed %s (dropping)\n", ap.c_str());
        fflush(stderr);
        // drop the dead entry
        std::string drop_chain = best->chain_hash_hex;
        index_.erase(std::remove_if(index_.begin(), index_.end(),
                     [&](const llama_kv_store_entry & e) {
                         return e.model_fp_hex == mh && e.chain_hash_hex == drop_chain;
                     }), index_.end());
        rewrite_index();
        stats_.misses++;
        return 0;
    }

    // Parse the artifact to verify CRC + extract canonical tokens (the truth).
    std::vector<uint8_t>         payload;
    llama_kv_artifact_metadata   meta = {};
    llama_kv_artifact_summary    summary = {};
    llama_kv_prompt_payload      pp = {};
    std::string                  perr;
    bool parsed = llama_kv_artifact_read_mem(bytes.data(), bytes.size(),
                                             payload, &meta, &summary, &perr, &pp);
    if (!parsed) {
        fprintf(stderr, "[kv-store] lookup: artifact parse failed (%s); fail-open\n", perr.c_str());
        fflush(stderr);
        stats_.misses++;
        return 0;
    }
    if (best->crc32 != 0 && summary.payload_crc32 != best->crc32) {
        fprintf(stderr, "[kv-store] lookup: CRC mismatch (index=%08x artifact=%08x); fail-open\n",
                best->crc32, summary.payload_crc32);
        fflush(stderr);
        stats_.misses++;
        return 0;
    }
    // Re-verify LCP against the artifact's OWN embedded canonical tokens (not just
    // the index copy) — the index could in theory be stale vs the .kva bytes.
    {
        uint32_t lcp_art = lcp_tokens(request_tokens, pp.token_ids);
        if (lcp_art < lcp) lcp = lcp_art;
        if (lcp < lcp_min) {
            fprintf(stderr, "[kv-store] lookup: artifact-token LCP=%u < min=%u; fail-open\n", lcp, lcp_min);
            fflush(stderr);
            stats_.misses++;
            return 0;
        }
        if (lcp >= (uint32_t) request_tokens.size()) {
            lcp = (uint32_t) request_tokens.size() - 1;
        }
    }

    // The KV the artifact carries covers `streamed_prefix` leading cells. We can
    // only reuse cells the artifact actually has KV for. The installed prefix is
    // min(lcp, streamed_prefix) — beyond that there are no KV cells to import.
    uint32_t reuse = std::min<uint32_t>(lcp, best->streamed_prefix ? best->streamed_prefix
                                                                   : meta.streamed_prefix_token_count);
    if (reuse < lcp_min) {
        // KV coverage shorter than the min reuse; not worth a handoff.
        stats_.misses++;
        return 0;
    }

    out.artifact_bytes = std::move(bytes);
    out.prefix_tokens.assign(request_tokens.begin(), request_tokens.begin() + reuse);
    out.streamed_prefix = reuse;
    out.n_tokens        = best->n_tokens;
    out.chain_hash_hex  = best->chain_hash_hex;

    // touch LRU (persisted on next rewrite/eviction).
    const_cast<llama_kv_store_entry *>(best)->last_used = now_seconds();
    stats_.hits++;

    fprintf(stderr,
            "[kv-store] HIT model_fp=%s chain=%s n_tokens=%u reuse=%u (req=%zu, suffix=%zu to prefill)\n",
            mh.c_str(), best->chain_hash_hex.c_str(), best->n_tokens, reuse,
            request_tokens.size(), request_tokens.size() - reuse);
    fflush(stderr);
    return reuse;
}

// ---------------------------------------------------------------------------
// eviction + stats.
// ---------------------------------------------------------------------------

uint64_t llama_kv_store::current_bytes() const {
    uint64_t t = 0;
    for (const auto & e : index_) t += e.bytes;
    return t;
}

void llama_kv_store::evict_to_cap() {
    if (!enabled_ || max_bytes_ == 0) return;
    if (current_bytes() <= max_bytes_) return;
    // LRU by last_used; evict whole artifacts until under cap.
    bool changed = false;
    while (current_bytes() > max_bytes_ && !index_.empty()) {
        // find LRU
        size_t lru = 0;
        for (size_t i = 1; i < index_.size(); ++i) {
            if (index_[i].last_used < index_[lru].last_used) lru = i;
        }
        const auto & v = index_[lru];
        std::error_code ec;
        // reconstruct path from hex (model_fp dir + chain.kva)
        std::string ap = (fs::path(dir_) / v.model_fp_hex / (v.chain_hash_hex + ".kva")).string();
        fs::remove(ap, ec);
        fprintf(stderr, "[kv-store] EVICT model_fp=%s chain=%s (%.2fMB, lru)\n",
                v.model_fp_hex.c_str(), v.chain_hash_hex.c_str(), v.bytes / (1024.0 * 1024.0));
        fflush(stderr);
        index_.erase(index_.begin() + lru);
        stats_.evictions++;
        changed = true;
    }
    if (changed) rewrite_index();
}

llama_kv_store::stats llama_kv_store::snapshot() const {
    stats s = stats_;
    s.entries = index_.size();
    s.bytes   = current_bytes();
    return s;
}

// ---------------------------------------------------------------------------
// Process-wide shared instance (env-driven, lazily constructed once).
// ---------------------------------------------------------------------------

llama_kv_store & llama_kv_store_instance() {
    // Constructed on first use from %LLAMA_KV_CACHE_DIR% / %LLAMA_KV_CACHE_MAX_GB%.
    // The --kv-cache-dir / --kv-cache-max-gb flags set those envs at arg-parse
    // time (before any prefill/handoff), so the singleton picks them up here.
    static llama_kv_store store; // empty dir => disabled no-op
    return store;
}
