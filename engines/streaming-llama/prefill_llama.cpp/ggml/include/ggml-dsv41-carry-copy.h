#pragma once
// Internal C++ CUDA utility. No llama object, engine, file or allocator dependency.
#include "ggml-backend.h"
#include <cstdint>
namespace ggml_dsv41_copy {
constexpr uint64_t record_bytes=92176,page_bytes=2953216,header_bytes=4096;
// Opt-in compact record (OPUS22): BF16/F16 hidden field 40960 B instead of the
// F32 81920 B field; pre, selected and candidate IDs follow unchanged in size.
constexpr uint64_t compact_record_bytes=51216,compact_page_bytes=1642496;
constexpr bool known_geometry(uint64_t record,uint64_t page){
    return (record==record_bytes&&page==page_bytes)||(record==compact_record_bytes&&page==compact_page_bytes);}
constexpr uint32_t page_rows=32,maximum_rows=1048576,maximum_tile_rows=4096;
}
struct ggml_dsv41_ram_slice {
    void *base=nullptr;
    uint64_t bytes=0;
    uint32_t start=0,count=0;
    uint64_t record_bytes=ggml_dsv41_copy::record_bytes,page_bytes=ggml_dsv41_copy::page_bytes;
};
struct ggml_dsv41_validation {
    const float *residual=nullptr,*pre=nullptr,*mean=nullptr,*engram=nullptr;
    const int32_t *selected=nullptr,*candidates=nullptr;
    uint64_t residual_n=0,pre_n=0,mean_n=0,engram_n=0,selected_n=0,candidates_n=0;
    uint32_t compressed_rows=0;
    // OPUS22 round 8: a BF16 residual (bit patterns) validated in place.
    const uint16_t *residual_bf16=nullptr;uint64_t residual_bf16_n=0;
};
GGML_BACKEND_API int ggml_dsv41_cuda_validate(const ggml_dsv41_validation &,void *device_flags,void *stream);
GGML_BACKEND_API int ggml_dsv41_cuda_copy_ram(bool to_device,const ggml_dsv41_ram_slice &,
    uint64_t record_offset,uint64_t row_bytes,void *device,void *stream);
// OPUS22: decode original Engram rows (256 E4M3 codes + 8 E8M0 scales, 264 B)
// on the GPU, bit-identical to llama_dsv41::engram_decode_row. An invalid code
// or scale writes NaN and an overflow writes Inf, so the existing finite
// validator of the same tile rejects it before any layer compute.
GGML_BACKEND_API int ggml_dsv41_cuda_engram_decode(const void *raw,float *output,uint64_t rows,void *stream);
// Explicit offline TEST scheduling experiment. No kernel or precision change.
GGML_BACKEND_API int ggml_dsv41_cuda_blocking_sync(int ordinal,uint32_t *before,uint32_t *after);
