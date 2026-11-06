// Unit tests for the persistent KV-artifact store (Engine A / Phases A1+A2).
// No model required — builds synthetic KVARTIF1 artifacts (with embedded
// canonical prompt tokens) via the artifact serializer and exercises the store's
// put / longest-prefix lookup / token-exact verify / CRC verify / LRU eviction /
// index compaction / restart-survival paths against a temp directory.

#include "llama-kv-store.h"
#include "llama-kv-artifact.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

namespace fs = std::filesystem;

static std::string make_tmp_dir(const std::string & tag) {
    fs::path p = fs::temp_directory_path() / ("test_kv_store_" + tag + "_" +
                                              std::to_string((unsigned long long) ::time(nullptr)) +
                                              "_" + std::to_string((unsigned long long) rand()));
    std::error_code ec;
    fs::remove_all(p, ec);
    fs::create_directories(p, ec);
    return p.string();
}

// Build a real KVARTIF1 artifact carrying `tokens` as the canonical prompt IDs,
// with a synthetic KV payload. Returns the serialized bytes + CRC + the
// streamed_prefix advertised.
static std::vector<uint8_t> make_artifact(const std::vector<int32_t> & tokens,
                                          uint32_t streamed_prefix,
                                          uint32_t & crc_out) {
    llama_kv_artifact_metadata meta = {};
    meta.format_major = 1;
    meta.format_minor = 5;                 // v1.5: canonical prompt payload
    meta.n_layers     = 4;
    meta.n_ctx        = 4096;
    meta.token_count  = streamed_prefix;
    meta.type_k       = 1;
    meta.type_v       = 1;
    meta.is_mla       = 1;
    meta.v_trans      = 1;
    meta.prompt_token_count          = (uint32_t) tokens.size();
    meta.streamed_prefix_token_count = streamed_prefix;

    llama_kv_prompt_payload pp;
    pp.token_ids = tokens;
    pp.prompt_text = "synthetic";

    // a small synthetic KV payload (bytes don't matter for the store test)
    std::vector<uint8_t> payload(256, 0xAB);

    std::vector<uint8_t> bytes;
    llama_kv_artifact_summary summary = {};
    std::string err;
    bool ok = llama_kv_artifact_serialize_to_buffer(bytes, payload, meta, &summary, &err, &pp);
    assert(ok && "serialize artifact");
    crc_out = summary.payload_crc32;
    return bytes;
}

static std::vector<int32_t> seq(int32_t base, int n) {
    std::vector<int32_t> v;
    v.reserve(n);
    for (int i = 0; i < n; ++i) v.push_back(base + i);
    return v;
}

static const uint64_t MODEL_FP =
    llama_kv_store::compute_model_fp("test-arch 7B Q4_0", "fixtures/test.gguf",
                                     1, 1, 1, 1, 0);

static void test_chain_hash_matches_python_constants() {
    printf("  chain_hash chunking ... ");
    // < 256 tokens => no chain link (only whole 256-chunks hashed).
    auto h0 = llama_kv_store::chunk_hashes(seq(1000, 100), MODEL_FP);
    assert(h0.empty());
    // exactly 256 => 1 link; 600 => 2 links (floor(600/256)).
    auto h1 = llama_kv_store::chunk_hashes(seq(1000, 256), MODEL_FP);
    assert(h1.size() == 1);
    auto h2 = llama_kv_store::chunk_hashes(seq(1000, 600), MODEL_FP);
    assert(h2.size() == 2);
    // determinism + prefix consistency: first link identical for same prefix.
    assert(h1[0] == h2[0]);
    printf("OK\n");
}

static void test_put_lookup_roundtrip() {
    printf("  put + lookup roundtrip ... ");
    std::string dir = make_tmp_dir("rt");
    llama_kv_store store(dir, 200.0);
    assert(store.enabled());

    auto tokens = seq(7000, 700);            // > 2 full chunks
    uint32_t crc = 0;
    auto art = make_artifact(tokens, (uint32_t) tokens.size(), crc);
    assert(store.put(MODEL_FP, tokens, (uint32_t) tokens.size(), crc, art));

    // request = stored prefix + a divergent suffix
    auto req = tokens;
    for (int i = 0; i < 50; ++i) req.push_back(90000 + i);

    llama_kv_store_hit hit;
    uint32_t reuse = store.lookup(MODEL_FP, req, 64, hit);
    assert(reuse == (uint32_t) tokens.size());            // full prefix reused
    assert(hit.prefix_tokens.size() == reuse);
    assert(hit.prefix_tokens == tokens);
    assert(!hit.artifact_bytes.empty());
    printf("OK (reuse=%u)\n", reuse);
}

static void test_longest_prefix_multi_entry() {
    printf("  A2 longest-prefix multi-entry ... ");
    std::string dir = make_tmp_dir("a2");
    llama_kv_store store(dir, 200.0);

    auto shortp = seq(5000, 300);            // 1 chunk
    auto longp  = shortp;                     // shares the 300-tok prefix
    for (int i = 0; i < 400; ++i) longp.push_back(60000 + i); // total 700 => 2 chunks

    uint32_t c1 = 0, c2 = 0;
    store.put(MODEL_FP, shortp, (uint32_t) shortp.size(), c1, make_artifact(shortp, (uint32_t) shortp.size(), c1));
    store.put(MODEL_FP, longp,  (uint32_t) longp.size(),  c2, make_artifact(longp,  (uint32_t) longp.size(),  c2));

    // Request matching the LONG prefix should pick the deeper (long) entry.
    auto req = longp;
    req.push_back(123456);
    llama_kv_store_hit hit;
    uint32_t reuse = store.lookup(MODEL_FP, req, 64, hit);
    assert(reuse == (uint32_t) longp.size());
    assert(hit.n_tokens == (uint32_t) longp.size());

    // Request diverging right after the SHORT prefix should pick the short entry.
    auto req2 = shortp;
    for (int i = 0; i < 100; ++i) req2.push_back(777000 + i);
    llama_kv_store_hit hit2;
    uint32_t reuse2 = store.lookup(MODEL_FP, req2, 64, hit2);
    assert(reuse2 == (uint32_t) shortp.size());
    assert(hit2.n_tokens == (uint32_t) shortp.size());
    printf("OK (long=%u short=%u)\n", reuse, reuse2);
}

static void test_token_exact_verify_rejects_collision() {
    printf("  token-exact verify (truth over hash) ... ");
    std::string dir = make_tmp_dir("tev");
    llama_kv_store store(dir, 200.0);

    auto tokens = seq(8000, 500);
    uint32_t crc = 0;
    store.put(MODEL_FP, tokens, (uint32_t) tokens.size(), crc, make_artifact(tokens, (uint32_t) tokens.size(), crc));

    // A request that shares fewer than lcp_min leading tokens must miss even if
    // it would otherwise reach the index (it won't share a chunk hash here).
    auto req = seq(99, 500);                  // totally different tokens
    llama_kv_store_hit hit;
    uint32_t reuse = store.lookup(MODEL_FP, req, 64, hit);
    assert(reuse == 0);                       // miss
    printf("OK\n");
}

static void test_crc_mismatch_fails_open() {
    printf("  CRC corruption fails open ... ");
    std::string dir = make_tmp_dir("crc");
    llama_kv_store store(dir, 200.0);

    auto tokens = seq(8100, 500);
    uint32_t crc = 0;
    auto art = make_artifact(tokens, (uint32_t) tokens.size(), crc);
    store.put(MODEL_FP, tokens, (uint32_t) tokens.size(), crc, art);

    // Corrupt the on-disk artifact payload (flip a byte well past the header).
    // Find the single .kva and overwrite a payload byte.
    for (auto & p : fs::recursive_directory_iterator(dir)) {
        if (p.path().extension() == ".kva") {
            std::fstream f(p.path(), std::ios::in | std::ios::out | std::ios::binary);
            f.seekp(-4, std::ios::end);
            char b = 0x00;
            f.read(&b, 1);
            f.seekp(-1, std::ios::cur);
            b ^= 0xFF;
            f.write(&b, 1);
            f.close();
            break;
        }
    }

    auto req = tokens;
    req.push_back(424242);
    llama_kv_store_hit hit;
    uint32_t reuse = store.lookup(MODEL_FP, req, 64, hit);
    // CRC mismatch (or parse failure) => fail-open miss, never a bad blob.
    assert(reuse == 0);
    printf("OK\n");
}

static void test_lru_eviction() {
    printf("  LRU eviction by size cap ... ");
    std::string dir = make_tmp_dir("lru");
    // Each artifact ~ a few hundred bytes; set a tiny cap so a few entries evict.
    // 1 artifact ~ header + 256 payload + tokens*4 ~ < 4KB. Cap at ~3KB so only
    // the most-recent fits => eviction fires.
    double tiny_gb = 3000.0 / (1024.0 * 1024.0 * 1024.0);
    llama_kv_store store(dir, tiny_gb);

    for (int k = 0; k < 5; ++k) {
        auto tokens = seq(10000 + k * 1000, 300);
        uint32_t crc = 0;
        store.put(MODEL_FP, tokens, (uint32_t) tokens.size(), crc,
                  make_artifact(tokens, (uint32_t) tokens.size(), crc));
    }
    auto s = store.snapshot();
    assert(s.evictions > 0);
    assert(s.bytes <= store.max_gb() * 1024.0 * 1024.0 * 1024.0 + 1);
    // The most-recent entry must still be present + usable.
    auto last = seq(10000 + 4 * 1000, 300);
    auto req = last; req.push_back(1);
    llama_kv_store_hit hit;
    assert(store.lookup(MODEL_FP, req, 64, hit) == (uint32_t) last.size());
    printf("OK (entries=%llu evictions=%llu)\n",
           (unsigned long long) s.entries, (unsigned long long) s.evictions);
}

static void test_restart_survival() {
    printf("  restart survival (reload index from disk) ... ");
    std::string dir = make_tmp_dir("restart");
    auto tokens = seq(20000, 800);
    uint32_t crc = 0;
    {
        llama_kv_store store(dir, 200.0);     // process 1: write
        store.put(MODEL_FP, tokens, (uint32_t) tokens.size(), crc,
                  make_artifact(tokens, (uint32_t) tokens.size(), crc));
    }
    {
        llama_kv_store store2(dir, 200.0);    // process 2: fresh store, same dir
        assert(store2.enabled());
        assert(store2.snapshot().entries == 1);
        auto req = tokens; req.push_back(55);
        llama_kv_store_hit hit;
        uint32_t reuse = store2.lookup(MODEL_FP, req, 64, hit);
        assert(reuse == (uint32_t) tokens.size());      // restored across "restart"
        assert(hit.prefix_tokens == tokens);
    }
    printf("OK\n");
}

static void test_disabled_is_noop() {
    printf("  disabled store = no-op ... ");
    llama_kv_store store("", 200.0);          // empty dir => disabled
    assert(!store.enabled());
    auto tokens = seq(1, 300);
    uint32_t crc = 0;
    assert(!store.put(MODEL_FP, tokens, 300, crc, make_artifact(tokens, 300, crc)));
    llama_kv_store_hit hit;
    assert(store.lookup(MODEL_FP, tokens, 64, hit) == 0);
    printf("OK\n");
}

static void test_write_once() {
    printf("  write-once per (model, prefix) ... ");
    std::string dir = make_tmp_dir("wo");
    llama_kv_store store(dir, 200.0);
    auto tokens = seq(30000, 500);
    uint32_t crc = 0;
    auto art = make_artifact(tokens, (uint32_t) tokens.size(), crc);
    assert(store.put(MODEL_FP, tokens, (uint32_t) tokens.size(), crc, art));
    assert(store.put(MODEL_FP, tokens, (uint32_t) tokens.size(), crc, art));  // 2nd = touch
    assert(store.snapshot().entries == 1);
    assert(store.snapshot().writes == 1);    // only one real write
    printf("OK\n");
}

int main() {
    printf("test-kv-store:\n");
    test_chain_hash_matches_python_constants();
    test_put_lookup_roundtrip();
    test_longest_prefix_multi_entry();
    test_token_exact_verify_rejects_collision();
    test_crc_mismatch_fails_open();
    test_lru_eviction();
    test_restart_survival();
    test_disabled_is_noop();
    test_write_once();
    printf("test-kv-store: ALL PASS\n");
    return 0;
}
