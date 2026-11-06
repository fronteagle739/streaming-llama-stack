#include "llama-upstream-dsa-wire.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

template <typename T>
void append(std::vector<uint8_t> & dst, T value) {
    const auto * bytes = reinterpret_cast<const uint8_t *>(&value);
    dst.insert(dst.end(), bytes, bytes + sizeof(value));
}

void append_bytes(std::vector<uint8_t> & dst, std::initializer_list<uint8_t> bytes) {
    dst.insert(dst.end(), bytes.begin(), bytes.end());
}

template <typename T>
T read(const std::vector<uint8_t> & src, size_t & offset) {
    assert(offset + sizeof(T) <= src.size());
    T value {};
    std::memcpy(&value, src.data() + offset, sizeof(value));
    offset += sizeof(value);
    return value;
}

void expect_bytes(const std::vector<uint8_t> & src, size_t & offset,
                  std::initializer_list<uint8_t> expected) {
    assert(offset + expected.size() <= src.size());
    for (uint8_t value : expected) {
        assert(src[offset++] == value);
    }
}

void append_cell(std::vector<uint8_t> & dst, int32_t pos,
                 std::initializer_list<int32_t> seq_ids) {
    append<int32_t>(dst, pos);
    append<uint32_t>(dst, static_cast<uint32_t>(seq_ids.size()));
    for (int32_t seq_id : seq_ids) {
        append<int32_t>(dst, seq_id);
    }
}

void append_layer(std::vector<uint8_t> & dst, int32_t type, uint64_t row_size,
                  std::initializer_list<uint8_t> payload) {
    append<int32_t>(dst, type);
    append<uint64_t>(dst, row_size);
    append_bytes(dst, payload);
}

std::vector<uint8_t> make_legacy_payload() {
    constexpr uint32_t state_ext_bit = 0x80000000u;

    std::vector<uint8_t> src;
    append<uint32_t>(src, 2); // n_stream

    // Stream 0: two cells, two main MLA layers, one lightning-indexer layer.
    append<uint32_t>(src, 2);
    append_cell(src, 7, {0});
    append_cell(src, 8, {0, 3});
    append<uint32_t>(src, 0); // v_trans: FA layout
    append<uint32_t>(src, state_ext_bit | 2u);
    append<uint32_t>(src, 1); // state_ext_has_k_idx
    append_layer(src, 1, 2, {0x10, 0x11, 0x12, 0x13});
    append_layer(src, 1, 2, {0x20, 0x21, 0x22, 0x23});
    append<uint32_t>(src, 1);
    // Two F16 indexer rows. Current upstream stores these after an
    // orthonormal Walsh-Hadamard rotation; the fork stores them before it.
    append_layer(src, 1, 8, {
        0x00, 0x3c, 0x00, 0x40, 0x00, 0x42, 0x00, 0x44,
        0x00, 0x44, 0x00, 0x42, 0x00, 0x40, 0x00, 0x3c});
    append<uint32_t>(src, 0);

    // Stream 1: one cell with the same cache geometry.
    append<uint32_t>(src, 1);
    append_cell(src, 4, {1});
    append<uint32_t>(src, 0);
    append<uint32_t>(src, state_ext_bit | 2u);
    append<uint32_t>(src, 1);
    append_layer(src, 1, 2, {0x40, 0x41});
    append_layer(src, 1, 2, {0x50, 0x51});
    append<uint32_t>(src, 1);
    append_layer(src, 1, 8, {
        0x00, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00});
    append<uint32_t>(src, 0);

    return src;
}

std::vector<uint8_t> make_q8_payload() {
    constexpr uint32_t state_ext_bit = 0x80000000u;
    std::vector<uint8_t> src;
    append<uint32_t>(src, 1);
    append<uint32_t>(src, 1);
    append_cell(src, 0, {0});
    append<uint32_t>(src, 0);
    append<uint32_t>(src, state_ext_bit | 1u);
    append<uint32_t>(src, 1);
    append_layer(src, 1, 2, {0x00, 0x00});
    append<uint32_t>(src, 1);
    append<int32_t>(src, 8);   // GGML_TYPE_Q8_0
    append<uint64_t>(src, 34); // one 32-value Q8_0 block
    append<uint16_t>(src, 0x3c00); // scale 1.0
    append<int8_t>(src, 1);
    for (int i = 1; i < 32; ++i) {
        append<int8_t>(src, 0);
    }
    return src;
}

void test_converts_fork_glm_dsa_to_current_upstream_layout() {
    std::vector<uint8_t> dst;
    std::string error;
    assert(llama_upstream_dsa_wire_convert(make_legacy_payload(), 3, {0}, dst, error));
    assert(error.empty());

    size_t off = 0;
    assert(read<uint32_t>(dst, off) == 0xaf143cd8u);
    assert(read<int32_t>(dst, off) == 3);

    // Upstream cache 1: ordinary MLA K state, with the fork extension removed.
    assert(read<uint32_t>(dst, off) == 2);
    assert(read<uint32_t>(dst, off) == 2);
    assert(read<int32_t>(dst, off) == 7);
    assert(read<uint32_t>(dst, off) == 1);
    assert(read<int32_t>(dst, off) == 0);
    assert(read<int32_t>(dst, off) == 8);
    assert(read<uint32_t>(dst, off) == 2);
    assert(read<int32_t>(dst, off) == 0);
    assert(read<int32_t>(dst, off) == 3);
    assert(read<uint32_t>(dst, off) == 0);
    assert(read<uint32_t>(dst, off) == 2);
    assert(read<int32_t>(dst, off) == 1);
    assert(read<uint64_t>(dst, off) == 2);
    expect_bytes(dst, off, {0x10, 0x11, 0x12, 0x13});
    assert(read<int32_t>(dst, off) == 1);
    assert(read<uint64_t>(dst, off) == 2);
    expect_bytes(dst, off, {0x20, 0x21, 0x22, 0x23});

    assert(read<uint32_t>(dst, off) == 1);
    assert(read<int32_t>(dst, off) == 4);
    assert(read<uint32_t>(dst, off) == 1);
    assert(read<int32_t>(dst, off) == 1);
    assert(read<uint32_t>(dst, off) == 0);
    assert(read<uint32_t>(dst, off) == 2);
    assert(read<int32_t>(dst, off) == 1);
    assert(read<uint64_t>(dst, off) == 2);
    expect_bytes(dst, off, {0x40, 0x41});
    assert(read<int32_t>(dst, off) == 1);
    assert(read<uint64_t>(dst, off) == 2);
    expect_bytes(dst, off, {0x50, 0x51});

    // Upstream cache 2: standard indexer-K cache using the same cell metadata.
    assert(read<uint32_t>(dst, off) == 2);
    assert(read<uint32_t>(dst, off) == 2);
    assert(read<int32_t>(dst, off) == 7);
    assert(read<uint32_t>(dst, off) == 1);
    assert(read<int32_t>(dst, off) == 0);
    assert(read<int32_t>(dst, off) == 8);
    assert(read<uint32_t>(dst, off) == 2);
    assert(read<int32_t>(dst, off) == 0);
    assert(read<int32_t>(dst, off) == 3);
    assert(read<uint32_t>(dst, off) == 0);
    assert(read<uint32_t>(dst, off) == 1);
    assert(read<int32_t>(dst, off) == 1);
    assert(read<uint64_t>(dst, off) == 8);
    // H4/sqrt(4) * [1,2,3,4] = [5,-1,-2,0]
    // H4/sqrt(4) * [4,3,2,1] = [5,1,2,0]
    expect_bytes(dst, off, {
        0x00, 0x45, 0x00, 0xbc, 0x00, 0xc0, 0x00, 0x00,
        0x00, 0x45, 0x00, 0x3c, 0x00, 0x40, 0x00, 0x00});

    assert(read<uint32_t>(dst, off) == 1);
    assert(read<int32_t>(dst, off) == 4);
    assert(read<uint32_t>(dst, off) == 1);
    assert(read<int32_t>(dst, off) == 1);
    assert(read<uint32_t>(dst, off) == 0);
    assert(read<uint32_t>(dst, off) == 1);
    assert(read<int32_t>(dst, off) == 1);
    assert(read<uint64_t>(dst, off) == 8);
    // H4/sqrt(4) * [1,0,0,0] = [.5,.5,.5,.5]
    expect_bytes(dst, off, {
        0x00, 0x38, 0x00, 0x38, 0x00, 0x38, 0x00, 0x38});
    assert(off == dst.size());
}

void test_rejects_truncated_payload() {
    auto src = make_legacy_payload();
    src.pop_back();
    std::vector<uint8_t> dst;
    std::string error;
    assert(!llama_upstream_dsa_wire_convert(src, 0, {0}, dst, error));
    assert(!error.empty());
    assert(dst.empty());
}

void test_rejects_payload_without_indexer_extension() {
    auto src = make_legacy_payload();
    // First stream's n_layer_word follows n_stream, cell_count, and 28 bytes of metadata.
    const size_t n_layer_word_offset = 4 + 4 + 28 + 4;
    uint32_t plain_layers = 2;
    std::memcpy(src.data() + n_layer_word_offset, &plain_layers, sizeof(plain_layers));

    std::vector<uint8_t> dst;
    std::string error;
    assert(!llama_upstream_dsa_wire_convert(src, 0, {0}, dst, error));
    assert(error.find("indexer") != std::string::npos);
    assert(dst.empty());
}

void test_glm53_default_indexer_cadence_matches_current_upstream() {
    const auto layers = llama_upstream_glm_dsa_indexer_layers(78, 1048576);
    assert(layers.size() == 21);
    assert(layers[0] == 0 && layers[1] == 1 && layers[2] == 2);
    for (size_t i = 3; i < layers.size(); ++i) {
        assert(layers[i] == 6 + 4 * (i - 3));
    }
    assert(layers.back() == 74);

    const auto pre52 = llama_upstream_glm_dsa_indexer_layers(5, 131072);
    assert((pre52 == std::vector<uint32_t> {0, 1, 2, 3, 4}));
}

void test_q8_indexer_rows_are_dequantized_rotated_and_requantized() {
    std::vector<uint8_t> dst;
    std::string error;
    assert(llama_upstream_dsa_wire_convert(make_q8_payload(), 0, {0}, dst, error));
    assert(error.empty());

    size_t off = 0;
    assert(read<uint32_t>(dst, off) == 0xaf143cd8u);
    assert(read<int32_t>(dst, off) == 0);

    // Main MLA cache.
    assert(read<uint32_t>(dst, off) == 1);
    assert(read<uint32_t>(dst, off) == 1);
    assert(read<int32_t>(dst, off) == 0);
    assert(read<uint32_t>(dst, off) == 1);
    assert(read<int32_t>(dst, off) == 0);
    assert(read<uint32_t>(dst, off) == 0);
    assert(read<uint32_t>(dst, off) == 1);
    assert(read<int32_t>(dst, off) == 1);
    assert(read<uint64_t>(dst, off) == 2);
    expect_bytes(dst, off, {0x00, 0x00});

    // Indexer cache: H32/sqrt(32) rotates [1,0,...] to one constant
    // row. Q8_0 therefore stores a non-zero scale and 32 equal +127 quants.
    assert(read<uint32_t>(dst, off) == 1);
    assert(read<uint32_t>(dst, off) == 1);
    assert(read<int32_t>(dst, off) == 0);
    assert(read<uint32_t>(dst, off) == 1);
    assert(read<int32_t>(dst, off) == 0);
    assert(read<uint32_t>(dst, off) == 0);
    assert(read<uint32_t>(dst, off) == 1);
    assert(read<int32_t>(dst, off) == 8);
    assert(read<uint64_t>(dst, off) == 34);
    assert(read<uint16_t>(dst, off) != 0);
    for (int i = 0; i < 32; ++i) {
        assert(read<int8_t>(dst, off) == 127);
    }
    assert(off == dst.size());
}

} // namespace

int main() {
    std::puts("test-upstream-dsa-wire");
    test_converts_fork_glm_dsa_to_current_upstream_layout();
    test_rejects_truncated_payload();
    test_rejects_payload_without_indexer_extension();
    test_glm53_default_indexer_cadence_matches_current_upstream();
    test_q8_indexer_rows_are_dequantized_rotated_and_requantized();
    std::puts("all upstream DSA wire tests passed");
    return 0;
}
