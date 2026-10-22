// Self-contained unit test for kvblob_v2 codec.
// Builds standalone (only depends on the kvblob_v2.{h,cpp} pair). Run via:
//   build/bin/test-kvblob-v2
// Exit code 0 = all tests passed; non-zero = first failing test number.

#include "kvblob_v2.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define TEST_ASSERT(expr)                                                      \
    do {                                                                       \
        if (!(expr)) {                                                         \
            std::fprintf(stderr, "ASSERT FAILED: %s\n  at %s:%d\n", #expr,     \
                         __FILE__, __LINE__);                                  \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)


static int g_test_id = 0;

#define RUN_TEST(name)                                                         \
    do {                                                                       \
        ++g_test_id;                                                           \
        std::printf("[%2d] %s ... ", g_test_id, #name);                        \
        std::fflush(stdout);                                                   \
        name();                                                                \
        std::printf("OK\n");                                                   \
    } while (0)


// Helper: build an opaque blob from N synthetic seqs.
static std::string build_blob(const std::vector<llama_seq_id> & seq_ids,
                              const std::vector<size_t> & per_seq_size) {
    std::vector<std::vector<uint8_t>> per_seq_opaque;
    std::vector<std::pair<uint32_t, uint32_t>> per_seq_pos;
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        std::vector<uint8_t> buf(per_seq_size[i]);
        // Fill with a recognizable pattern keyed by seq_id so we can verify
        // round-trip after parse.
        for (size_t j = 0; j < buf.size(); ++j) {
            buf[j] = static_cast<uint8_t>((seq_ids[i] * 31 + j) & 0xFF);
        }
        per_seq_opaque.push_back(std::move(buf));
        per_seq_pos.emplace_back(0u, static_cast<uint32_t>(per_seq_size[i] / 16));
    }
    return kvblob_v2_build_opaque(
        KVBLOB_V2_PRODUCER_LLAMA_CPP,
        /*n_layers=*/4, /*n_kv_heads=*/2, /*head_dim=*/64,
        KVBLOB_V2_DTYPE_F16, KVBLOB_V2_DTYPE_F16,
        /*mla_compressed=*/false,
        seq_ids, per_seq_opaque, per_seq_pos);
}


static void test_magic_and_version() {
    auto blob = build_blob({0, 1, 2}, {32, 64, 96});
    TEST_ASSERT(blob.size() >= KVBLOB_V2_HEADER_SIZE);
    TEST_ASSERT(static_cast<uint8_t>(blob[0]) == 'K');
    TEST_ASSERT(static_cast<uint8_t>(blob[1]) == 'V');
    TEST_ASSERT(static_cast<uint8_t>(blob[2]) == 'B');
    TEST_ASSERT(static_cast<uint8_t>(blob[3]) == '2');
}


static void test_basic_roundtrip() {
    std::vector<llama_seq_id> seq_ids = {3, 7, 19};
    std::vector<size_t> per_seq_size = {128, 256, 64};
    auto blob = build_blob(seq_ids, per_seq_size);

    kvblob_v2_header h{};
    std::vector<kvblob_v2_seq_desc> descs;
    std::string err;
    TEST_ASSERT(kvblob_v2_parse(blob, h, descs, err));
    TEST_ASSERT(err.empty());

    TEST_ASSERT(h.magic == KVBLOB_V2_MAGIC);
    TEST_ASSERT(h.version == 2);
    TEST_ASSERT(h.producer_id == KVBLOB_V2_PRODUCER_LLAMA_CPP);
    TEST_ASSERT((h.producer_flags & KVBLOB_V2_FLAG_LAYOUT_OPAQUE) != 0);
    TEST_ASSERT(h.n_seqs == 3);

    TEST_ASSERT(descs.size() == 3);
    for (size_t i = 0; i < descs.size(); ++i) {
        TEST_ASSERT(descs[i].seq_id == seq_ids[i]);
        TEST_ASSERT(descs[i].bytes_per_layer == per_seq_size[i]);
    }
}


static void test_seq_ids_preserved_verbatim() {
    // Phase 316z architectural fix: producer's [3, 7, 19] is exactly
    // receiver's [3, 7, 19]. No implicit remap to 0, no other shifts.
    std::vector<llama_seq_id> seq_ids = {3, 7, 19};
    auto blob = build_blob(seq_ids, {32, 32, 32});

    kvblob_v2_header h{};
    std::vector<kvblob_v2_seq_desc> descs;
    std::string err;
    TEST_ASSERT(kvblob_v2_parse(blob, h, descs, err));
    TEST_ASSERT(descs[0].seq_id == 3);
    TEST_ASSERT(descs[1].seq_id == 7);
    TEST_ASSERT(descs[2].seq_id == 19);
}


static void test_payload_byte_pattern_roundtrip() {
    // Verify the per-seq opaque bytes survive the wire byte-for-byte.
    std::vector<llama_seq_id> seq_ids = {5, 11};
    std::vector<size_t> sizes = {128, 256};
    auto blob = build_blob(seq_ids, sizes);

    kvblob_v2_header h{};
    std::vector<kvblob_v2_seq_desc> descs;
    std::string err;
    TEST_ASSERT(kvblob_v2_parse(blob, h, descs, err));

    for (size_t i = 0; i < descs.size(); ++i) {
        const auto & d = descs[i];
        TEST_ASSERT(d.k_byte_offset + d.bytes_per_layer <= blob.size());
        const uint8_t * payload = reinterpret_cast<const uint8_t *>(
            blob.data() + d.k_byte_offset);
        for (uint64_t j = 0; j < d.bytes_per_layer; ++j) {
            uint8_t expected = static_cast<uint8_t>((seq_ids[i] * 31 + j) & 0xFF);
            TEST_ASSERT(payload[j] == expected);
        }
    }
}


static void test_descriptor_offsets_distinct_and_sequential() {
    // Layout invariant: per-seq opaque blobs are concatenated in
    // descriptor-table order. Offsets must be strictly increasing.
    std::vector<llama_seq_id> seq_ids = {0, 1, 2, 3, 4};
    std::vector<size_t> sizes = {64, 128, 256, 512, 32};
    auto blob = build_blob(seq_ids, sizes);

    kvblob_v2_header h{};
    std::vector<kvblob_v2_seq_desc> descs;
    std::string err;
    TEST_ASSERT(kvblob_v2_parse(blob, h, descs, err));

    uint64_t prev_end = h.payload_offset;
    for (size_t i = 0; i < descs.size(); ++i) {
        TEST_ASSERT(descs[i].k_byte_offset == prev_end);
        prev_end = descs[i].k_byte_offset + descs[i].bytes_per_layer;
    }
}


static void test_remap_header_parse_basic() {
    auto m = kvblob_v2_parse_seq_id_remap_header("3=100,7=200,19=300");
    TEST_ASSERT(m.size() == 3);
    TEST_ASSERT(m[3] == 100);
    TEST_ASSERT(m[7] == 200);
    TEST_ASSERT(m[19] == 300);
}


static void test_remap_header_parse_empty() {
    auto m = kvblob_v2_parse_seq_id_remap_header("");
    TEST_ASSERT(m.empty());
}


static void test_remap_header_parse_whitespace() {
    auto m = kvblob_v2_parse_seq_id_remap_header("  3=100  ,  7  =  200  ");
    TEST_ASSERT(m.size() == 2);
    TEST_ASSERT(m[3] == 100);
    TEST_ASSERT(m[7] == 200);
}


static void test_remap_header_parse_malformed() {
    bool threw = false;
    try {
        (void) kvblob_v2_parse_seq_id_remap_header("3,7=200");
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    TEST_ASSERT(threw);
}


static void test_remap_header_format() {
    std::map<llama_seq_id, llama_seq_id> m;
    m[3] = 100;
    m[7] = 200;
    auto s = kvblob_v2_format_seq_id_remap_header(m);
    // Map iterates sorted by key so the output is deterministic.
    TEST_ASSERT(s == "3=100,7=200");
}


static void test_remap_header_roundtrip() {
    std::map<llama_seq_id, llama_seq_id> orig;
    orig[1] = 1001;
    orig[2] = 1002;
    orig[42] = 1042;
    auto s = kvblob_v2_format_seq_id_remap_header(orig);
    auto parsed = kvblob_v2_parse_seq_id_remap_header(s);
    TEST_ASSERT(parsed == orig);
}


static void test_parse_rejects_bad_magic() {
    std::string bad(KVBLOB_V2_HEADER_SIZE, '\0');
    bad[0] = 'X';  // bad magic
    kvblob_v2_header h{};
    std::vector<kvblob_v2_seq_desc> descs;
    std::string err;
    TEST_ASSERT(!kvblob_v2_parse(bad, h, descs, err));
    TEST_ASSERT(err.find("magic") != std::string::npos);
}


static void test_parse_rejects_short_body() {
    std::string tiny(10, '\0');
    kvblob_v2_header h{};
    std::vector<kvblob_v2_seq_desc> descs;
    std::string err;
    TEST_ASSERT(!kvblob_v2_parse(tiny, h, descs, err));
    TEST_ASSERT(err.find("too short") != std::string::npos);
}


static void test_thousand_seqs_roundtrip() {
    // Scale check matching the user's 1000-slot architectural goal. Verifies
    // the descriptor table doesn't quietly cap or truncate at any small N.
    std::vector<llama_seq_id> seq_ids;
    std::vector<size_t> sizes;
    seq_ids.reserve(1000);
    sizes.reserve(1000);
    for (int i = 0; i < 1000; ++i) {
        seq_ids.push_back(i);
        sizes.push_back(32);
    }
    auto blob = build_blob(seq_ids, sizes);

    kvblob_v2_header h{};
    std::vector<kvblob_v2_seq_desc> descs;
    std::string err;
    TEST_ASSERT(kvblob_v2_parse(blob, h, descs, err));
    TEST_ASSERT(h.n_seqs == 1000);
    TEST_ASSERT(descs.size() == 1000);
    for (int i = 0; i < 1000; ++i) {
        TEST_ASSERT(descs[i].seq_id == i);
    }
}


int main() {
    std::printf("Running kvblob_v2 codec tests...\n");
    RUN_TEST(test_magic_and_version);
    RUN_TEST(test_basic_roundtrip);
    RUN_TEST(test_seq_ids_preserved_verbatim);
    RUN_TEST(test_payload_byte_pattern_roundtrip);
    RUN_TEST(test_descriptor_offsets_distinct_and_sequential);
    RUN_TEST(test_remap_header_parse_basic);
    RUN_TEST(test_remap_header_parse_empty);
    RUN_TEST(test_remap_header_parse_whitespace);
    RUN_TEST(test_remap_header_parse_malformed);
    RUN_TEST(test_remap_header_format);
    RUN_TEST(test_remap_header_roundtrip);
    RUN_TEST(test_parse_rejects_bad_magic);
    RUN_TEST(test_parse_rejects_short_body);
    RUN_TEST(test_thousand_seqs_roundtrip);
    std::printf("\nALL OK  %d/%d kvblob_v2 codec tests\n", g_test_id, g_test_id);
    return 0;
}
