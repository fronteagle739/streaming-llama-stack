// TDD: KV artifact header v1.1 tests — written before implementation.
// Tests artifact write/read round-trip with new fields:
//   - last_token_id (G6)
//   - v_trans, is_mla, model_fingerprint (G7)
//   - backwards compatibility with v1.0 files
//
// Uses small synthetic payloads — no model needed.

#include "llama-kv-artifact.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

namespace fs = std::filesystem;

static std::string tmp_path(const std::string & name) {
    fs::path p = fs::temp_directory_path() / ("test_kv_artifact_" + name + ".bin");
    return p.string();
}

// ---------------------------------------------------------------------------
// Test: basic round-trip with v1.0 fields
// ---------------------------------------------------------------------------
static void test_roundtrip_basic() {
    printf("  roundtrip_basic ... ");

    auto path = tmp_path("basic");
    std::vector<uint8_t> payload = { 0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04 };

    llama_kv_artifact_metadata meta = {};
    meta.format_major = 1;
    meta.format_minor = 0;
    meta.n_layers     = 32;
    meta.n_ctx        = 4096;
    meta.token_count  = 512;
    meta.type_k       = 1;   // fp16
    meta.type_v       = 1;
    meta.flags        = LLAMA_KV_ARTIFACT_FLAG_TRANSPORTED;

    llama_kv_artifact_summary summary = {};
    std::string error;
    assert(llama_kv_artifact_write(path, payload, meta, &summary, &error));
    assert(summary.payload_bytes == 8);
    assert(summary.payload_crc32 != 0);

    // Read back
    std::vector<uint8_t> payload_out;
    llama_kv_artifact_metadata meta_out = {};
    llama_kv_artifact_summary summary_out = {};
    assert(llama_kv_artifact_read(path, payload_out, &meta_out, &summary_out, &error));

    assert(payload_out == payload);
    assert(meta_out.format_major == 1);
    assert(meta_out.format_minor == 0);
    assert(meta_out.n_layers == 32);
    assert(meta_out.n_ctx == 4096);
    assert(meta_out.token_count == 512);
    assert(meta_out.type_k == 1);
    assert(meta_out.type_v == 1);
    assert(meta_out.flags == LLAMA_KV_ARTIFACT_FLAG_TRANSPORTED);
    assert(summary_out.payload_bytes == 8);
    assert(summary_out.payload_crc32 == summary.payload_crc32);

    fs::remove(path);
    printf("OK\n");
}

// ---------------------------------------------------------------------------
// Test: round-trip with last_token_id (G6)
// ---------------------------------------------------------------------------
static void test_roundtrip_last_token_id() {
    printf("  roundtrip_last_token_id ... ");

    auto path = tmp_path("last_token");
    std::vector<uint8_t> payload = { 0x42 };

    llama_kv_artifact_metadata meta = {};
    meta.format_major = 1;
    meta.format_minor = 1;
    meta.n_layers     = 4;
    meta.token_count  = 100;
    meta.last_token_id = 31337;

    std::string error;
    assert(llama_kv_artifact_write(path, payload, meta, nullptr, &error));

    std::vector<uint8_t> payload_out;
    llama_kv_artifact_metadata meta_out = {};
    assert(llama_kv_artifact_read(path, payload_out, &meta_out, nullptr, &error));

    assert(meta_out.last_token_id == 31337);
    assert(payload_out.size() == 1);
    assert(payload_out[0] == 0x42);

    fs::remove(path);
    printf("OK\n");
}

// ---------------------------------------------------------------------------
// Test: round-trip with v1.1 extension fields (G7)
// ---------------------------------------------------------------------------
static void test_roundtrip_v1_1_fields() {
    printf("  roundtrip_v1_1_fields ... ");

    auto path = tmp_path("v1_1");
    std::vector<uint8_t> payload(1024, 0xAA);  // 1KB test payload

    llama_kv_artifact_metadata meta = {};
    meta.format_major      = 1;
    meta.format_minor      = 1;
    meta.n_layers          = 61;
    meta.n_ctx             = 32768;
    meta.token_count       = 8192;
    meta.type_k            = 1;
    meta.type_v            = 1;
    meta.flags             = LLAMA_KV_ARTIFACT_FLAG_PROGRESSIVE;
    meta.last_token_id     = 42;
    meta.v_trans           = 1;
    meta.is_mla            = 0;
    meta.model_fingerprint = 0xDEADBEEFCAFEBABEull;

    std::string error;
    assert(llama_kv_artifact_write(path, payload, meta, nullptr, &error));

    std::vector<uint8_t> payload_out;
    llama_kv_artifact_metadata meta_out = {};
    assert(llama_kv_artifact_read(path, payload_out, &meta_out, nullptr, &error));

    assert(meta_out.format_major == 1);
    assert(meta_out.format_minor == 1);
    assert(meta_out.n_layers == 61);
    assert(meta_out.n_ctx == 32768);
    assert(meta_out.token_count == 8192);
    assert(meta_out.type_k == 1);
    assert(meta_out.type_v == 1);
    assert(meta_out.flags == LLAMA_KV_ARTIFACT_FLAG_PROGRESSIVE);
    assert(meta_out.last_token_id == 42);
    assert(meta_out.v_trans == 1);
    assert(meta_out.is_mla == 0);
    assert(meta_out.model_fingerprint == 0xDEADBEEFCAFEBABEull);
    assert(payload_out == payload);

    fs::remove(path);
    printf("OK\n");
}

// ---------------------------------------------------------------------------
// Test: MLA model artifact round-trip
// ---------------------------------------------------------------------------
static void test_roundtrip_mla() {
    printf("  roundtrip_mla ... ");

    auto path = tmp_path("mla");
    std::vector<uint8_t> payload(64, 0xBB);

    llama_kv_artifact_metadata meta = {};
    meta.format_major      = 1;
    meta.format_minor      = 1;
    meta.n_layers          = 28;
    meta.token_count       = 50;
    meta.last_token_id     = -1;  // not stored
    meta.v_trans           = 0;
    meta.is_mla            = 1;
    meta.model_fingerprint = 999;

    std::string error;
    assert(llama_kv_artifact_write(path, payload, meta, nullptr, &error));

    std::vector<uint8_t> payload_out;
    llama_kv_artifact_metadata meta_out = {};
    assert(llama_kv_artifact_read(path, payload_out, &meta_out, nullptr, &error));

    assert(meta_out.is_mla == 1);
    assert(meta_out.last_token_id == -1);
    assert(meta_out.model_fingerprint == 999);

    fs::remove(path);
    printf("OK\n");
}

// ---------------------------------------------------------------------------
// Test: backwards compat — v1.0 written file read by v1.1 reader
// ---------------------------------------------------------------------------
static void test_backwards_compat_v1_0_read() {
    printf("  backwards_compat_v1_0_read ... ");

    // Write a file with format_minor=0 (no extension fields)
    auto path = tmp_path("v1_0_compat");
    std::vector<uint8_t> payload = { 0x11, 0x22 };

    llama_kv_artifact_metadata meta = {};
    meta.format_major = 1;
    meta.format_minor = 0;  // old format
    meta.n_layers     = 4;
    meta.token_count  = 10;
    // v1.0 fields: last_token_id, v_trans, is_mla, model_fingerprint are all zero/defaults

    std::string error;
    assert(llama_kv_artifact_write(path, payload, meta, nullptr, &error));

    // Read with v1.1 reader — extension fields should have safe defaults
    std::vector<uint8_t> payload_out;
    llama_kv_artifact_metadata meta_out = {};
    assert(llama_kv_artifact_read(path, payload_out, &meta_out, nullptr, &error));

    assert(meta_out.format_minor == 0);
    assert(meta_out.last_token_id == -1);       // default: not stored
    assert(meta_out.v_trans == 0);               // default
    assert(meta_out.is_mla == 0);                // default
    assert(meta_out.model_fingerprint == 0);     // default: unknown
    assert(payload_out == payload);

    fs::remove(path);
    printf("OK\n");
}

// ---------------------------------------------------------------------------
// Test: empty payload
// ---------------------------------------------------------------------------
static void test_empty_payload() {
    printf("  empty_payload ... ");

    auto path = tmp_path("empty");
    std::vector<uint8_t> payload;

    llama_kv_artifact_metadata meta = {};
    meta.format_major = 1;
    meta.format_minor = 1;
    meta.n_layers     = 1;
    meta.token_count  = 0;

    llama_kv_artifact_summary summary = {};
    std::string error;
    assert(llama_kv_artifact_write(path, payload, meta, &summary, &error));
    assert(summary.payload_bytes == 0);

    std::vector<uint8_t> payload_out;
    llama_kv_artifact_metadata meta_out = {};
    assert(llama_kv_artifact_read(path, payload_out, &meta_out, nullptr, &error));
    assert(payload_out.empty());
    assert(meta_out.token_count == 0);

    fs::remove(path);
    printf("OK\n");
}

// ---------------------------------------------------------------------------
// Test: CRC validation catches corruption
// ---------------------------------------------------------------------------
static void test_crc_corruption_detected() {
    printf("  crc_corruption_detected ... ");

    auto path = tmp_path("corrupt");
    std::vector<uint8_t> payload = { 0x01, 0x02, 0x03, 0x04, 0x05 };

    llama_kv_artifact_metadata meta = {};
    meta.format_major = 1;
    meta.format_minor = 1;
    meta.n_layers     = 1;
    meta.token_count  = 1;

    std::string error;
    assert(llama_kv_artifact_write(path, payload, meta, nullptr, &error));

    // Corrupt one byte in the payload region of the file
    {
        FILE * f = fopen(path.c_str(), "r+b");
        assert(f);
        // Seek past header to first payload byte and flip it
        fseek(f, 0, SEEK_END);
        long file_size = ftell(f);
        // Payload starts after header, corrupt the last byte
        fseek(f, file_size - 1, SEEK_SET);
        uint8_t byte = 0xFF;
        fwrite(&byte, 1, 1, f);
        fclose(f);
    }

    std::vector<uint8_t> payload_out;
    llama_kv_artifact_metadata meta_out = {};
    bool ok = llama_kv_artifact_read(path, payload_out, &meta_out, nullptr, &error);
    assert(!ok);  // CRC mismatch should cause failure
    assert(error.find("crc") != std::string::npos || error.find("CRC") != std::string::npos);

    fs::remove(path);
    printf("OK\n");
}

// ---------------------------------------------------------------------------
// Test: large payload (stress test)
// ---------------------------------------------------------------------------
static void test_large_payload() {
    printf("  large_payload ... ");

    auto path = tmp_path("large");
    // 1MB payload — small enough for a test, large enough to exercise chunked I/O
    std::vector<uint8_t> payload(1024 * 1024);
    for (size_t i = 0; i < payload.size(); i++) {
        payload[i] = (uint8_t)(i & 0xFF);
    }

    llama_kv_artifact_metadata meta = {};
    meta.format_major      = 1;
    meta.format_minor      = 1;
    meta.n_layers          = 32;
    meta.token_count       = 4096;
    meta.last_token_id     = 12345;
    meta.v_trans           = 1;
    meta.model_fingerprint = 0x123456789ABCDEFull;

    llama_kv_artifact_summary summary = {};
    std::string error;
    assert(llama_kv_artifact_write(path, payload, meta, &summary, &error));
    assert(summary.payload_bytes == 1024 * 1024);

    std::vector<uint8_t> payload_out;
    llama_kv_artifact_metadata meta_out = {};
    llama_kv_artifact_summary summary_out = {};
    assert(llama_kv_artifact_read(path, payload_out, &meta_out, &summary_out, &error));

    assert(payload_out == payload);
    assert(meta_out.last_token_id == 12345);
    assert(meta_out.v_trans == 1);
    assert(meta_out.model_fingerprint == 0x123456789ABCDEFull);
    assert(summary_out.payload_crc32 == summary.payload_crc32);

    fs::remove(path);
    printf("OK\n");
}

// ---------------------------------------------------------------------------
// Test: read_mem round-trip (write to file, read file bytes, parse from memory)
// ---------------------------------------------------------------------------
static void test_read_mem_roundtrip() {
    printf("  read_mem_roundtrip ... ");

    auto path = tmp_path("readmem");
    std::vector<uint8_t> payload = { 0xCA, 0xFE, 0xBA, 0xBE };

    llama_kv_artifact_metadata meta = {};
    meta.format_major      = 1;
    meta.format_minor      = 1;
    meta.n_layers          = 8;
    meta.token_count       = 200;
    meta.last_token_id     = 777;
    meta.v_trans           = 1;
    meta.model_fingerprint = 0xABCDEF0123456789ull;

    std::string error;
    assert(llama_kv_artifact_write(path, payload, meta, nullptr, &error));

    // Read the whole file into memory
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    assert(ifs.is_open());
    size_t file_size = (size_t)ifs.tellg();
    ifs.seekg(0);
    std::vector<uint8_t> file_data(file_size);
    ifs.read((char*)file_data.data(), (std::streamsize)file_size);
    ifs.close();

    // Parse from memory
    std::vector<uint8_t> payload_out;
    llama_kv_artifact_metadata meta_out = {};
    if (!llama_kv_artifact_read_mem(file_data.data(), file_data.size(),
                                     payload_out, &meta_out, nullptr, &error)) {
        fprintf(stderr, "FAIL: read_mem error: %s (file_size=%zu)\n", error.c_str(), file_size);
        assert(false);
    }

    assert(payload_out == payload);
    assert(meta_out.n_layers == 8);
    assert(meta_out.token_count == 200);
    assert(meta_out.last_token_id == 777);
    assert(meta_out.v_trans == 1);
    assert(meta_out.model_fingerprint == 0xABCDEF0123456789ull);

    // Test truncated buffer
    assert(!llama_kv_artifact_read_mem(file_data.data(), 10, payload_out, nullptr, nullptr, &error));

    fs::remove(path);
    printf("OK\n");
}

// ---------------------------------------------------------------------------
// Test: v1.5 prompt payload round-trip
// ---------------------------------------------------------------------------
static void test_roundtrip_prompt_payload() {
    printf("  roundtrip_prompt_payload ... ");

    auto path = tmp_path("prompt_payload");
    std::vector<uint8_t> payload = { 0x10, 0x20, 0x30, 0x40 };

    llama_kv_artifact_metadata meta = {};
    meta.format_major                = 1;
    meta.format_minor                = 5;
    meta.n_layers                    = 78;
    meta.n_ctx                       = 131072;
    meta.token_count                 = 16000;
    meta.last_token_id               = 4242;
    meta.flags                       = LLAMA_KV_ARTIFACT_FLAG_TRANSPORTED;

    llama_kv_prompt_payload prompt = {};
    prompt.prompt_text = "<s>hello from canonical text";
    prompt.token_ids   = { 1, 2, 3, 4, 5, 6 };

    meta.prompt_token_count          = (uint32_t) prompt.token_ids.size();
    meta.prompt_text_bytes           = (uint32_t) prompt.prompt_text.size();
    meta.streamed_prefix_token_count = 4;

    std::string error;
    assert(llama_kv_artifact_write(path, payload, meta, nullptr, &error, &prompt));

    std::vector<uint8_t> payload_out;
    llama_kv_artifact_metadata meta_out = {};
    llama_kv_prompt_payload prompt_out = {};
    assert(llama_kv_artifact_read(path, payload_out, &meta_out, nullptr, &error, &prompt_out));

    assert(payload_out == payload);
    assert(meta_out.format_minor == 5);
    assert(meta_out.prompt_token_count == prompt.token_ids.size());
    assert(meta_out.prompt_text_bytes == prompt.prompt_text.size());
    assert(meta_out.streamed_prefix_token_count == 4);
    assert(prompt_out.prompt_text == prompt.prompt_text);
    assert(prompt_out.token_ids == prompt.token_ids);

    fs::remove(path);
    printf("OK\n");
}

// ---------------------------------------------------------------------------
// Test: v1.5 prompt payload read_mem round-trip
// ---------------------------------------------------------------------------
static void test_read_mem_prompt_payload() {
    printf("  read_mem_prompt_payload ... ");

    auto path = tmp_path("readmem_prompt_payload");
    std::vector<uint8_t> payload = { 0xA1, 0xB2, 0xC3 };

    llama_kv_artifact_metadata meta = {};
    meta.format_major                = 1;
    meta.format_minor                = 5;
    meta.n_layers                    = 12;
    meta.token_count                 = 32;
    meta.prompt_token_count          = 5;
    meta.streamed_prefix_token_count = 3;

    llama_kv_prompt_payload prompt = {};
    prompt.prompt_text = "abcde";
    prompt.token_ids   = { 11, 22, 33, 44, 55 };
    meta.prompt_text_bytes = (uint32_t) prompt.prompt_text.size();

    std::string error;
    assert(llama_kv_artifact_write(path, payload, meta, nullptr, &error, &prompt));

    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    assert(ifs.is_open());
    size_t file_size = (size_t) ifs.tellg();
    ifs.seekg(0);
    std::vector<uint8_t> file_data(file_size);
    ifs.read((char *) file_data.data(), (std::streamsize) file_size);
    ifs.close();

    std::vector<uint8_t> payload_out;
    llama_kv_artifact_metadata meta_out = {};
    llama_kv_prompt_payload prompt_out = {};
    assert(llama_kv_artifact_read_mem(file_data.data(), file_data.size(),
                                      payload_out, &meta_out, nullptr, &error, &prompt_out));

    assert(payload_out == payload);
    assert(meta_out.format_minor == 5);
    assert(meta_out.prompt_token_count == 5);
    assert(meta_out.prompt_text_bytes == 5);
    assert(meta_out.streamed_prefix_token_count == 3);
    assert(prompt_out.prompt_text == prompt.prompt_text);
    assert(prompt_out.token_ids == prompt.token_ids);

    fs::remove(path);
    printf("OK\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("test-kv-artifact: round-trip tests\n");
    test_roundtrip_basic();
    test_roundtrip_last_token_id();
    test_roundtrip_v1_1_fields();
    test_roundtrip_mla();
    test_roundtrip_prompt_payload();
    test_backwards_compat_v1_0_read();
    test_empty_payload();

    printf("test-kv-artifact: integrity tests\n");
    test_crc_corruption_detected();
    test_large_payload();

    printf("test-kv-artifact: memory reader tests\n");
    test_read_mem_roundtrip();
    test_read_mem_prompt_payload();

    printf("\ntest-kv-artifact: all tests OK\n");
    return 0;
}
