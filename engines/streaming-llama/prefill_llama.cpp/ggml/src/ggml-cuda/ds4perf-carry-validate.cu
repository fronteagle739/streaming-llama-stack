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
// Integer-exact form of engram_decode_row: the value M*2^E (M<=15) rounded to
// F32 (round-half-even, including subnormals) and then to BF16, with no
// floating arithmetic that -use_fast_math could flush.
__device__ __forceinline__ uint32_t engram_value_bits(uint32_t code,uint32_t scale){
    if((code&127u)==127u||scale==255u)return 0x7fc00000u;
    const uint32_t exponent=(code>>3)&15u,mantissa=code&7u;
    const uint32_t m=exponent?8u+mantissa:mantissa;
    const int e=(exponent?int(exponent)-10:-9)+int(scale)-127;
    uint32_t bits=0;
    if(m){
        const int top=31-__clz(m),x=top+e;
        if(x>127)bits=0x7f800000u;
        else if(x>=-126)bits=(uint32_t(x+127)<<23)|((m<<(23-top))&0x7fffffu);
        else{const int shift=e+149;
            if(shift>=0)bits=m<<shift;
            else{const int s=-shift;
                if(s<5){const uint32_t q=m>>s,r=m&((1u<<s)-1u),h=1u<<(s-1);bits=q+((r>h||(r==h&&(q&1u)))?1u:0u);}}}
    }
    if(code&128u)bits|=0x80000000u;
    if((bits&0x7f800000u)!=0x7f800000u)bits=(bits+0x7fffu+((bits>>16)&1u))&0xffff0000u;
    return bits;
}
__global__ void engram_decode_kernel(const uint8_t *raw,uint32_t *output,uint64_t n){
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n)return;
    const uint8_t *row=raw+(i>>8)*264;const uint32_t j=uint32_t(i&255u);
    output[i]=engram_value_bits(row[j],row[256+(j>>5)]);
}
__device__ bool nonfinite(float x){return (__float_as_uint(x)&0x7f800000u)==0x7f800000u;}
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

int ggml_dsv41_cuda_copy_ram(bool to_device,const ggml_dsv41_ram_slice &slice,
        uint64_t offset,uint64_t width,void *device,void *stream){
    using namespace ggml_dsv41_copy;
    if(!known_geometry(slice.record_bytes,slice.page_bytes))return int(cudaErrorInvalidValue);
    const uint64_t record_bytes=slice.record_bytes,page_bytes=slice.page_bytes;
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

int ggml_dsv41_cuda_engram_decode(const void *raw,float *output,uint64_t rows,void *stream){
    if(!raw||!output||!rows||rows>uint64_t(4096)*24)return int(cudaErrorInvalidValue);
    const uint64_t n=rows*256;
    engram_decode_kernel<<<unsigned((n+255)/256),256,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t*>(raw),reinterpret_cast<uint32_t*>(output),n);
    return int(cudaGetLastError());
}
