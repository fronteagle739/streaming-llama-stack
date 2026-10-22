#pragma once
#include <cstdint>
// Explicit loader scheduling experiment. Original 16-column arithmetic order.
// This binds the tested object, not the immutable baseline object proof.
namespace llama_dsv41_native_loader_resource_proof {
constexpr unsigned columns=16,loader_unroll=1,maximum_tile_rows=4096;
constexpr uint64_t minimum_pool_bytes=2097152,minimum_runtime_bytes=2097152;
constexpr bool operator_resource_fixture_passed=true,full_model_qualified=false;
constexpr const char* source_sha256="f75b97ecd2120b5cbb54c0ffcf1adcf53903f87bd044eb42ec4ca37fdb80fa46";
constexpr const char* object_sha256="6f60719ac67f2161a9f80b25c5c24826c9f7e922906f35d08c551b5ee55e3885";
constexpr const char* fixture_sha256="e5ee138680d86e54c96fe2af0f0ebba567a527be0ca176fbf393f676ce289b65";
constexpr const char* phase_sha256="f7168498f146a8156cece19f3a7663d1f0dec6e08835b134f100c65dd7fc2053";
}
