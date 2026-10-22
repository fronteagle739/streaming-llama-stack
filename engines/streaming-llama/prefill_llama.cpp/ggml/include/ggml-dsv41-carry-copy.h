#pragma once
// Internal C++ CUDA utility. No llama object, engine, file or allocator dependency.
#include "ggml-backend.h"
#include <cstdint>
namespace ggml_dsv41_copy {
constexpr uint64_t record_bytes=92176,page_bytes=2953216,header_bytes=4096;
constexpr uint32_t page_rows=32,maximum_rows=1048576,maximum_tile_rows=4096;
}
struct ggml_dsv41_ram_slice {
    void *base=nullptr;
    uint64_t bytes=0;
    uint32_t start=0,count=0;
};
struct ggml_dsv41_validation {
    const float *residual=nullptr,*pre=nullptr,*mean=nullptr,*engram=nullptr;
    const int32_t *selected=nullptr,*candidates=nullptr;
    uint64_t residual_n=0,pre_n=0,mean_n=0,engram_n=0,selected_n=0,candidates_n=0;
    uint32_t compressed_rows=0;
};
GGML_BACKEND_API int ggml_dsv41_cuda_validate(const ggml_dsv41_validation &,void *device_flags,void *stream);
GGML_BACKEND_API int ggml_dsv41_cuda_copy_ram(bool to_device,const ggml_dsv41_ram_slice &,
    uint64_t record_offset,uint64_t row_bytes,void *device,void *stream);
// Explicit offline TEST scheduling experiment. No kernel or precision change.
GGML_BACKEND_API int ggml_dsv41_cuda_blocking_sync(int ordinal,uint32_t *before,uint32_t *after);
