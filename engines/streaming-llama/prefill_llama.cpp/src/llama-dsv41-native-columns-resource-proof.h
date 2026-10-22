#pragma once
#include <cstdint>
// Separate explicit32-column schedule experiment. It does not replace the
// immutable16-column proof or claim full-model/native-envelope qualification.
namespace llama_dsv41_native_columns_resource_proof {
constexpr unsigned columns=32,maximum_tile_rows=4096;
constexpr uint64_t minimum_pool_bytes=2097152,minimum_runtime_bytes=2097152;
constexpr bool operator_resource_fixture_passed=true,full_model_qualified=false;
constexpr const char* source_sha256="b7a902d5063f61f6132f837fde357d5a4711b162152ff798047ed668d86d960a";
constexpr const char* object_sha256="97114a7c6c067a429b6d1d407c7692abaece41503c64a214a6ffc8cc1f38ff35";
constexpr const char* fixture_sha256="1fe40ab6c4ca9fcb3cd13e90c116b4ecd37edc22c74511487eeaad413da7eed2";
constexpr const char* phase_sha256="4e75f5e989823ba8790fb92f2b05d53d4de36b6855b391848e92a013d15c8cc5";
// 282loader hashes,68operator cases,204candidate calls vs29688serial MMF calls.
// Exact per-column output bytes, with original Q4_K and native F16 rounding.
}
