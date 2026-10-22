#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

// Shared adapter arithmetic. It creates no factory, buffer, queue or environment
// override. Physical admission and actual DirectStorage qualification are separate.
namespace llama_wf_bounded {
constexpr uint64_t mib=1024ull*1024;
constexpr uint64_t profile_staging_bytes=1500*mib;
constexpr uint64_t small_staging_bytes=250*mib;
constexpr uint64_t baseline_staging_bytes=3500*mib;
inline uint64_t add(uint64_t a,uint64_t b) {
    if(a>UINT64_MAX-b)throw std::overflow_error("Weight fetch policy overflow");
    return a+b;
}
inline uint64_t align(uint64_t value,uint64_t alignment) {
    if(!alignment||(alignment&(alignment-1)))throw std::invalid_argument("Weight fetch policy alignment");
    return add(value,alignment-1)&~(alignment-1);
}
inline size_t auto_io_block(size_t maximum_layer_bytes) {
    // Same policy as the established adapter: target1000 requests/layer,
    // round up4KiB, clamp512KiB..32MiB. Avoid max+999 overflow.
    const uint64_t block=align(maximum_layer_bytes/1000+(maximum_layer_bytes%1000!=0),4096);
    return size_t(std::max<uint64_t>(512*1024,std::min<uint64_t>(32*mib,block)));
}
struct plan {
    uint32_t buffers;
    uint64_t compute_buffer_bytes,staging_bytes,compute_and_staging_bytes;
    size_t request_bytes;
};
inline plan make(uint64_t maximum_stage_bytes,uint32_t buffers,uint64_t staging_limit) {
    if(!maximum_stage_bytes||maximum_stage_bytes>SIZE_MAX||buffers<1||buffers>2||
       (staging_limit!=baseline_staging_bytes&&staging_limit!=profile_staging_bytes&&staging_limit!=small_staging_bytes))
        throw std::invalid_argument("Unqualified weight fetch profile");
    const uint64_t compute=align(add(maximum_stage_bytes,std::max<uint64_t>(maximum_stage_bytes/100,64*1024)),65536);
    const uint64_t staging=align(std::min(compute,staging_limit),65536);
    if(compute>UINT64_MAX/buffers)throw std::overflow_error("Weight fetch buffer multiplication");
    return {buffers,compute,staging,add(compute*buffers,staging),auto_io_block(size_t(maximum_stage_bytes))};
}
inline bool exact_two_buffer_profile(uint64_t maximum_stage_bytes,uint32_t buffers,
                                     uint64_t staging_limit,size_t request_bytes) {
    return maximum_stage_bytes&&maximum_stage_bytes<=SIZE_MAX&&buffers==2&&
           staging_limit==profile_staging_bytes&&request_bytes==auto_io_block(size_t(maximum_stage_bytes));
}
// Imported CUDA views alias the D3D allocation. CUDA's free-byte query
// double-counts those views on this WDDM driver; NVML and bounded allocation/
// canary tests establish the physical charge without disabling admission.
inline uint64_t shared_heap_physical_bytes(uint64_t compute_bytes,uint32_t buffers) {
    if(!compute_bytes||buffers<1||buffers>2)throw std::invalid_argument("Shared compute heap geometry");
    const uint64_t heap=align(compute_bytes,256*mib);
    if(heap>UINT64_MAX/buffers)throw std::overflow_error("Shared compute heap physical charge overflow");
    return heap*buffers;
}
inline bool exact_relay_profile(uint64_t maximum_stage_bytes,uint32_t buffers,
                               uint64_t staging_limit,size_t request_bytes) {
    return maximum_stage_bytes&&maximum_stage_bytes<=SIZE_MAX&&buffers>=1&&buffers<=2&&
           (staging_limit==small_staging_bytes||(buffers==2&&staging_limit==profile_staging_bytes))&&
           request_bytes==auto_io_block(size_t(maximum_stage_bytes));
}
}
