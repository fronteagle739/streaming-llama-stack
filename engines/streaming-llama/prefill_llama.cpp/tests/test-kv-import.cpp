// TDD: KV scatter map + import unit tests — written before implementation.
// Tests scatter map offset computation with synthetic parameters.
// Integration tests (actual model load + import) are in test-e2e-handoff.cpp.

#include "llama-kv-import.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <numeric>

#undef NDEBUG
#include <cassert>

// ---------------------------------------------------------------------------
// Scatter map offset computation tests (no model needed)
// ---------------------------------------------------------------------------

static void test_scatter_map_basic() {
    printf("  scatter_map_basic ... ");

    // Simulate a simple model:
    //   2 layers, n_embd_k_gqa=128, n_embd_v_gqa=128
    //   type_k = type_v = f16 (element size = 2 bytes)
    //   v_trans = false, is_mla = false
    //   kv_size = 512, token_count = 100

    kv_import_model_info info = {};
    info.n_layers = 2;
    info.kv_size  = 512;
    info.v_trans  = false;
    info.is_mla   = false;
    info.type_k_size = 2;  // fp16
    info.type_v_size = 2;  // fp16
    info.layers.resize(2);
    info.layers[0] = { 0, 128, 128 };  // layer 0: cache_idx=0, n_embd_k=128, n_embd_v=128
    info.layers[1] = { 1, 128, 128 };  // layer 1: cache_idx=1, n_embd_k=128, n_embd_v=128

    kv_scatter_map map;
    uint32_t token_count = 100;
    assert(kv_scatter_map_build(info, token_count, map));

    // With v_trans=false, 2 layers, no MLA:
    //   4 entries total (K0, V0, K1, V1)
    assert(map.entries.size() == 4);

    // K row size = 128 * 2 = 256 bytes
    // V row size = 128 * 2 = 256 bytes
    // K block per layer = 100 * 256 = 25600 bytes
    // V block per layer = 100 * 256 = 25600 bytes

    size_t k_row = 128 * 2;  // 256
    size_t k_block = 100 * k_row;  // 25600
    size_t v_block = 100 * k_row;  // same dimensions

    // Entry 0: K layer 0
    assert(map.entries[0].layer_idx == 0);
    assert(map.entries[0].tensor_kind == 0);  // K
    assert(map.entries[0].artifact_offset == 0);
    assert(map.entries[0].tensor_offset == 0);
    assert(map.entries[0].bytes == k_block);

    // Entry 1: V layer 0
    assert(map.entries[1].layer_idx == 0);
    assert(map.entries[1].tensor_kind == 1);  // V
    assert(map.entries[1].artifact_offset == k_block);
    assert(map.entries[1].tensor_offset == 0);
    assert(map.entries[1].bytes == v_block);

    // Entry 2: K layer 1
    assert(map.entries[2].layer_idx == 1);
    assert(map.entries[2].tensor_kind == 0);  // K
    assert(map.entries[2].artifact_offset == k_block + v_block);
    assert(map.entries[2].tensor_offset == 0);
    assert(map.entries[2].bytes == k_block);

    // Entry 3: V layer 1
    assert(map.entries[3].layer_idx == 1);
    assert(map.entries[3].tensor_kind == 1);  // V
    assert(map.entries[3].artifact_offset == 2 * k_block + v_block);
    assert(map.entries[3].tensor_offset == 0);
    assert(map.entries[3].bytes == v_block);

    // Total payload size
    size_t total = 2 * k_block + 2 * v_block;
    assert(map.total_payload_bytes == total);

    printf("OK\n");
}

static void test_scatter_map_mla() {
    printf("  scatter_map_mla ... ");

    // MLA model: V stored in K, no separate V data
    kv_import_model_info info = {};
    info.n_layers = 2;
    info.kv_size  = 512;
    info.v_trans  = false;
    info.is_mla   = true;
    info.type_k_size = 2;
    info.type_v_size = 2;
    info.layers.resize(2);
    info.layers[0] = { 0, 256, 128 };
    info.layers[1] = { 1, 256, 128 };

    kv_scatter_map map;
    assert(kv_scatter_map_build(info, 100, map));

    // MLA: only K entries, no V
    assert(map.entries.size() == 2);
    assert(map.entries[0].tensor_kind == 0);  // K
    assert(map.entries[1].tensor_kind == 0);  // K

    size_t k_block = 100 * 256 * 2;
    assert(map.entries[0].bytes == k_block);
    assert(map.entries[1].bytes == k_block);
    assert(map.entries[1].artifact_offset == k_block);

    printf("OK\n");
}

static void test_scatter_map_vtrans() {
    printf("  scatter_map_vtrans ... ");

    // V-transposed: V needs per-row scatter entries
    kv_import_model_info info = {};
    info.n_layers = 1;
    info.kv_size  = 512;
    info.v_trans  = true;
    info.is_mla   = false;
    info.type_k_size = 2;
    info.type_v_size = 2;
    info.layers.resize(1);
    info.layers[0] = { 0, 128, 64 };

    uint32_t token_count = 50;
    kv_scatter_map map;
    assert(kv_scatter_map_build(info, token_count, map));

    // K: 1 entry (contiguous)
    // V: when v_trans, one entry per token row → 50 entries
    // Total: 1 + 50 = 51
    assert(map.entries.size() == 51);

    // K entry
    size_t k_block = 50 * 128 * 2;
    assert(map.entries[0].layer_idx == 0);
    assert(map.entries[0].tensor_kind == 0);
    assert(map.entries[0].bytes == k_block);

    // V entries: each copies n_embd_v * type_v_size = 64 * 2 = 128 bytes
    // Artifact: V is row-major [n_embd_v, token_count] = stored as token_count rows
    //           of n_embd_v * type_v_size bytes each.
    // Cache:    V is [kv_size, n_embd_v] when transposed.
    //           Row i in artifact → column i in cache → offset i * n_embd_v * type_v_size
    //           BUT since cache is [kv_size, n_embd_v], element (pos, j) is at
    //           pos * n_embd_v * type_v_size + j * type_v_size
    //           which is pos * v_row_size
    size_t v_elem_row = 64 * 2;  // 128 bytes per token's V data
    for (uint32_t i = 0; i < token_count; i++) {
        auto & e = map.entries[1 + i];
        assert(e.layer_idx == 0);
        assert(e.tensor_kind == 1);
        assert(e.artifact_offset == k_block + i * v_elem_row);
        // In transposed cache: row i is at offset i * n_embd_v_padded * type_v_size
        // Since we're using n_embd_v_gqa (possibly padded), the tensor_offset is i * stride
        assert(e.bytes == v_elem_row);
    }

    printf("OK\n");
}

static void test_scatter_map_variable_embd() {
    printf("  scatter_map_variable_embd ... ");

    // Different n_embd_k/v per layer
    kv_import_model_info info = {};
    info.n_layers = 2;
    info.kv_size  = 256;
    info.v_trans  = false;
    info.is_mla   = false;
    info.type_k_size = 2;
    info.type_v_size = 2;
    info.layers.resize(2);
    info.layers[0] = { 0, 128, 64 };   // layer 0: K=128, V=64
    info.layers[1] = { 1, 256, 128 };  // layer 1: K=256, V=128

    kv_scatter_map map;
    assert(kv_scatter_map_build(info, 100, map));

    // 4 entries: K0, V0, K1, V1
    assert(map.entries.size() == 4);

    size_t k0 = 100 * 128 * 2;   // 25600
    size_t v0 = 100 * 64 * 2;    // 12800
    size_t k1 = 100 * 256 * 2;   // 51200
    size_t v1 = 100 * 128 * 2;   // 25600

    assert(map.entries[0].bytes == k0);
    assert(map.entries[1].bytes == v0);
    assert(map.entries[1].artifact_offset == k0);
    assert(map.entries[2].bytes == k1);
    assert(map.entries[2].artifact_offset == k0 + v0);
    assert(map.entries[3].bytes == v1);
    assert(map.entries[3].artifact_offset == k0 + v0 + k1);

    assert(map.total_payload_bytes == k0 + v0 + k1 + v1);

    printf("OK\n");
}

static void test_scatter_map_token_count_validation() {
    printf("  scatter_map_token_count_validation ... ");

    kv_import_model_info info = {};
    info.n_layers = 1;
    info.kv_size  = 512;
    info.v_trans  = false;
    info.is_mla   = false;
    info.type_k_size = 2;
    info.type_v_size = 2;
    info.layers.resize(1);
    info.layers[0] = { 0, 128, 128 };

    kv_scatter_map map;

    // token_count == kv_size → should fail (no room for decode)
    assert(!kv_scatter_map_build(info, 512, map));

    // token_count > kv_size → should fail
    assert(!kv_scatter_map_build(info, 1000, map));

    // token_count = kv_size - 1 → should succeed
    assert(kv_scatter_map_build(info, 511, map));

    // token_count = 0 → should succeed (empty import)
    assert(kv_scatter_map_build(info, 0, map));
    assert(map.entries.empty());

    printf("OK\n");
}

// ---------------------------------------------------------------------------
// Artifact metadata validation tests
// ---------------------------------------------------------------------------

static void test_artifact_metadata_validation() {
    printf("  artifact_metadata_validation ... ");

    kv_import_model_info info = {};
    info.n_layers = 2;
    info.kv_size  = 512;
    info.v_trans  = false;
    info.is_mla   = false;
    info.type_k_size = 2;
    info.type_v_size = 2;
    info.layers.resize(2);
    info.layers[0] = { 0, 128, 128 };
    info.layers[1] = { 1, 128, 128 };
    info.model_fingerprint = 12345;

    // Good metadata
    kv_import_artifact_info art = {};
    art.format_major = 1;
    art.n_layers = 2;
    art.token_count = 100;
    art.v_trans = false;
    art.is_mla = false;
    art.model_fingerprint = 12345;

    std::string error;
    assert(kv_import_validate(info, art, false, &error));

    // Wrong format major
    art.format_major = 2;
    assert(!kv_import_validate(info, art, false, &error));
    art.format_major = 1;

    // Wrong layer count
    art.n_layers = 3;
    assert(!kv_import_validate(info, art, false, &error));
    art.n_layers = 2;

    // token_count >= kv_size
    art.token_count = 512;
    assert(!kv_import_validate(info, art, false, &error));
    art.token_count = 100;

    // v_trans mismatch
    art.v_trans = true;
    assert(!kv_import_validate(info, art, false, &error));
    art.v_trans = false;

    // Fingerprint mismatch without allow flag
    art.model_fingerprint = 99999;
    assert(!kv_import_validate(info, art, false, &error));

    // Fingerprint mismatch WITH allow flag
    assert(kv_import_validate(info, art, true, &error));

    // Fingerprint == 0 (unknown) always passes
    art.model_fingerprint = 0;
    assert(kv_import_validate(info, art, false, &error));

    printf("OK\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    printf("test-kv-import: scatter map tests\n");
    test_scatter_map_basic();
    test_scatter_map_mla();
    test_scatter_map_vtrans();
    test_scatter_map_variable_embd();
    test_scatter_map_token_count_validation();

    printf("test-kv-import: validation tests\n");
    test_artifact_metadata_validation();

    printf("\ntest-kv-import: all tests OK\n");
    return 0;
}
