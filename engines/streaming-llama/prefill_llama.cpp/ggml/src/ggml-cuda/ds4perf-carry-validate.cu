#include "../../include/ggml-dsv41-carry-copy.h"
#include <cuda_runtime.h>
#include <algorithm>

int ggml_dsv41_cuda_blocking_sync(int ordinal,uint32_t *before,uint32_t *after){
    if(ordinal<0||!before||!after)return int(cudaErrorInvalidValue);
    auto e=cudaSetDevice(ordinal);if(e!=cudaSuccess)return int(e);
    unsigned flags=0;e=cudaGetDeviceFlags(&flags);if(e!=cudaSuccess)return int(e);*before=flags;
    e=cudaSetDeviceFlags((flags&~unsigned(cudaDeviceScheduleMask))|cudaDeviceScheduleBlockingSync);
    if(e!=cudaSuccess)return int(e);
    e=cudaGetDeviceFlags(&flags);if(e!=cudaSuccess)return int(e);*after=flags;
    return int((flags&cudaDeviceScheduleMask)==cudaDeviceScheduleBlockingSync?cudaSuccess:cudaErrorInvalidValue);
}

namespace {
__device__ bool nonfinite(float x){return (__float_as_uint(x)&0x7f800000u)==0x7f800000u;}
__global__ void validation_failure_gate(const uint32_t *flags){
    if (*flags) {
        printf("OTHER22_GPU_VALIDATION_FAILED flags=%u; nonfinite carry or invalid compact IDs\n",*flags);
        // This path is a terminal producer error, never a sanitized input or
        // a successful completion. Do not let consumers see invalid indices.
        asm volatile("trap;");
    }
}
__global__ void validate_kernel(ggml_dsv41_validation p,uint32_t *flags,uint64_t maximum){
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=maximum)return;
    uint32_t bad=0;
    if(i<p.residual_n&&nonfinite(p.residual[i]))bad|=1;
    if(i<p.pre_n&&nonfinite(p.pre[i]))bad|=2;
    if(i<p.mean_n&&nonfinite(p.mean[i]))bad|=4;
    if(i<p.engram_n&&nonfinite(p.engram[i]))bad|=8;
    if(i<p.selected_n){const int32_t id=p.selected[i];if(id<-1||id>=int32_t(p.compressed_rows))bad|=16;}
    if(i<p.candidates_n){const int32_t id=p.candidates[i];if(id<0||id>=int32_t((p.compressed_rows+7)/8))bad|=32;}
    if(bad)atomicOr(flags,bad);
}
}
int ggml_dsv41_cuda_validate(const ggml_dsv41_validation &p,void *device_flags,void *stream){
    constexpr uint64_t limit=uint64_t(5120)*4*4096;
    const uint64_t maximum=std::max({p.residual_n,p.pre_n,p.mean_n,p.engram_n,p.selected_n,p.candidates_n});
    if(!device_flags||!maximum||maximum>limit||p.compressed_rows>1048576||
       (p.residual_n&&!p.residual)||(p.pre_n&&!p.pre)||(p.mean_n&&!p.mean)||(p.engram_n&&!p.engram)||
       (p.selected_n&&!p.selected)||(p.candidates_n&&!p.candidates))return int(cudaErrorInvalidValue);
    const auto s=static_cast<cudaStream_t>(stream);
    const auto status=cudaMemsetAsync(device_flags,0,4,s);if(status!=cudaSuccess)return int(status);
    validate_kernel<<<unsigned((maximum+255)/256),256,0,s>>>(p,static_cast<uint32_t*>(device_flags),maximum);
    return int(cudaGetLastError());
}

int ggml_dsv41_cuda_validate_async(const ggml_dsv41_validation &p,void *flags,void *stream){
    const int status=ggml_dsv41_cuda_validate(p,flags,stream);
    if(status)return status;
    validation_failure_gate<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const uint32_t*>(flags));
    return int(cudaGetLastError());
}

int ggml_dsv41_cuda_copy_ram(bool to_device,const ggml_dsv41_ram_slice &slice,
        uint64_t offset,uint64_t width,void *device,void *stream){
    using namespace ggml_dsv41_copy;
    if(!slice.base||!device||!slice.count||slice.count>maximum_tile_rows||slice.start>maximum_rows||slice.count>maximum_rows-slice.start||
       !width||offset>record_bytes||width>record_bytes-offset)return int(cudaErrorInvalidValue);
    auto address=[&](uint32_t relative){const uint64_t row=uint64_t(slice.start)+relative;return header_bytes+(row/page_rows)*page_bytes+(row%page_rows)*record_bytes;};
    const uint64_t last=address(slice.count-1);
    if(last>slice.bytes||record_bytes>slice.bytes-last||last>UINTPTR_MAX-record_bytes||uintptr_t(slice.base)>UINTPTR_MAX-last-record_bytes)return int(cudaErrorInvalidValue);
    for(uint32_t done=0;done<slice.count;){
        const uint32_t rows=std::min(slice.count-done,page_rows-(slice.start+done)%page_rows);
        auto *host=static_cast<char*>(slice.base)+address(done)+offset;auto *gpu=static_cast<char*>(device)+uint64_t(done)*width;
        const auto s=static_cast<cudaStream_t>(stream);
        const auto status=to_device?cudaMemcpy2DAsync(gpu,width,host,record_bytes,width,rows,cudaMemcpyHostToDevice,s):
            cudaMemcpy2DAsync(host,record_bytes,gpu,width,width,rows,cudaMemcpyDeviceToHost,s);
        if(status!=cudaSuccess)return int(status);done+=rows;
    }
    return int(cudaSuccess);
}
