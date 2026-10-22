#pragma once
#include <cstdint>
// Separate TEST-only performance resource envelope. This does not qualify
// numerics, widen the correctness profile, or enable a shipping preset.
namespace llama_dsv41_native_half_perf_resource_proof {
constexpr uint32_t maximum_tile_rows=4096;
constexpr uint64_t minimum_pool_reserve_bytes=2097152,minimum_code_runtime_reserve_bytes=2097152;
constexpr bool resource_fixture_passed=true,numerical_qualification_passed=false;
constexpr const char * fixture_receipt_sha256="835f73e43590608fe82e06466b55ed0fad75fc5a8bdfd2a665218b69d966cdb5";
constexpr const char * phase_receipt_sha256="9d9fe5e90411bdf7a6fbe74cb8a017dce193433ce9a17ef215ff58f418a9e6b6";
constexpr const char * mathematical_source_sha256="8a5305489a7331b02538cd83bee69a329c77f242254b05fb4e290e34841e099a";
// 282 loader checks, 34 counts x2 geometries,204 native calls and29688 serial
// reference calls. Max observed ctx pool2MiB; first-loader global delta2MiB.
// Global counter differences do not establish allocation ownership.
}
