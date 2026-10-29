// SPDX-License-Identifier: MIT
// Streaming adapter to the pinned Apache-2.0 FlashInfer grouped GEMM.
#include "flashinfer_bridge.h"
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#define CUTE_DSL_CUDA_ERROR_CHECK(err) do { auto e=(err); if(e!=cudaSuccess) { \
    std::fprintf(stderr,"FlashInfer AOT CUDA failure: %s\n",cudaGetErrorString(e)); std::abort(); } } while(0)
#include "fi_mxfp4_2304_5120.h"
#include "fi_mxfp4_5120_2304.h"

struct native_block { uint8_t scale; uint8_t codes[16]; };
static_assert(sizeof(native_block)==17,"Canonical GGML MXFP4 block");
static __global__ void rearrange_weights(const native_block *input,uint8_t *values,uint32_t *scales,const int32_t *bounds,int n,int k) {
    const int expert=blockIdx.z;
    if(bounds[expert]==bounds[expert+1])return;
    const size_t local=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t expert_blocks=size_t(n)*(k/32);
    if(local>=expert_blocks)return;
    const size_t i=size_t(expert)*expert_blocks+local;
    const auto b=input[i];
    uint32_t packed_values[4]={0,0,0,0};
    #pragma unroll
    for(int j=0;j<16;++j) {
        const int lo=2*j,hi=lo+1;
        const unsigned a=(b.codes[lo&15]>>(lo>=16?4:0))&15;
        const unsigned c=(b.codes[hi&15]>>(hi>=16?4:0))&15;
        packed_values[j/4]|=uint32_t(a|(c<<4))<<((j%4)*8);
    }
    reinterpret_cast<uint4*>(values)[i]=make_uint4(packed_values[0],packed_values[1],packed_values[2],packed_values[3]);
    if((i&3)==0){
        const int kb=local%(k/32),column=local/(k/32);
        uint32_t packed=0;
        #pragma unroll
        for(int j=0;j<4;++j)packed|=uint32_t(input[i+j].scale)<<(8*j);
        scales[(size_t(expert)*(k/128)+kb/4)*n+column]=packed;
    }
}
static __global__ void pack_activations(const float *input,const int32_t *gather,const int32_t *bounds,
    __nv_fp8_e4m3 *values,uint8_t *scales,int rows,int k,int sf_rows,int64_t input_stride) {
    const int block=int(blockIdx.x)*8+threadIdx.x/32,lane=threadIdx.x%32;
    if(block>=rows*(k/32))return;
    const int row=block/(k/32),kb=block%(k/32);
    const float value=input[int64_t(gather[row])*input_stride+kb*32+lane];
    if(!isfinite(value))asm("trap;");
    float maximum=fabsf(value);
    #pragma unroll
    for(int distance=16;distance;distance/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,distance));
    int exponent=maximum==0?0:ilogbf(maximum)-8;
    if(maximum>ldexpf(448.f,exponent))++exponent;
    exponent=max(-127,min(127,exponent));
    values[size_t(row)*k+kb*32+lane]=__nv_fp8_e4m3(ldexpf(value,-exponent));
    if(lane==0){
        int low=0,high=384;
        while(low<high){const int mid=(low+high)/2;if(bounds[mid+1]<=row)low=mid+1;else high=mid;}
        const int expert=low;
        const int sf_start=(bounds[expert]+expert*3)/4*4;
        const int sf_row=sf_start+row-bounds[expert];
        scales[(size_t(kb/4)*sf_rows+sf_row)*4+kb%4]=uint8_t(exponent+127);
    }
}
static __global__ void scatter_output(const __nv_bfloat16 *input,const int32_t *scatter,float *output,
    int rows,int n,int64_t slot_stride,int64_t token_stride) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(rows)*n)return;
    const int route=scatter[i/n];
    output[int64_t(route/6)*token_stride+int64_t(route%6)*slot_stride+i%n]=__bfloat162float(input[i]);
}
struct workspace_view { void *data; };
#if defined(_WIN32)
#define BRIDGE_EXPORT extern "C" __declspec(dllexport)
#else
#define BRIDGE_EXPORT extern "C" __attribute__((visibility("default")))
#endif
BRIDGE_EXPORT int slstack_flashinfer_mxfp4_abi(){return 4;}
BRIDGE_EXPORT int slstack_flashinfer_mxfp4(const void *weights,const float *input,
    const int32_t *gather,const int32_t *scatter,const int32_t *bounds,float *output,
    int rows,int n,int k,int64_t input_stride,int64_t slot_stride,int64_t token_stride,void *workspace,uint64_t workspace_bytes,cudaStream_t stream) {
    if(rows<=0||rows>16384*6||!((n==2304&&k==5120)||(n==5120&&k==2304)))return int(cudaErrorInvalidValue);
    int device;cudaDeviceProp properties;
    auto error=cudaGetDevice(&device);if(error!=cudaSuccess)return int(error);
    error=cudaGetDeviceProperties(&properties,device);if(error!=cudaSuccess)return int(error);
    if(properties.major!=12||properties.minor!=0)return int(cudaErrorInvalidDeviceFunction);
    const int sf_rows=(rows+384*3+127)/128*128;
    const auto offsets=ggml_dsv41_flashinfer::layout(rows,n,k);
    const size_t sa=size_t(sf_rows)*k/32;
    if(!workspace||(reinterpret_cast<uintptr_t>(workspace)&255)||workspace_bytes>(5120ull<<20)||workspace_bytes<offsets.bytes){
        std::fprintf(stderr,"FlashInfer caller workspace admission failed: bytes=%llu required=%llu\n",static_cast<unsigned long long>(workspace_bytes),static_cast<unsigned long long>(offsets.bytes));
        return int(cudaErrorMemoryAllocation);
    }
    auto *base=static_cast<uint8_t*>(workspace);
    workspace_view w{base+offsets.weight},sfb{base+offsets.weight_scale},a{base+offsets.activation},sfa{base+offsets.activation_scale},out{base+offsets.output};
    #define TRY_CUDA(call) do{error=(call);if(error!=cudaSuccess)return int(error);}while(0)
    TRY_CUDA(cudaMemsetAsync(sfa.data,0,sa,stream));
    const size_t blocks=size_t(n)*(k/32);
    rearrange_weights<<<dim3(unsigned((blocks+255)/256),1,384),256,0,stream>>>(static_cast<const native_block*>(weights),static_cast<uint8_t*>(w.data),static_cast<uint32_t*>(sfb.data),bounds,n,k);
    TRY_CUDA(cudaGetLastError());
    pack_activations<<<(rows*(k/32)+7)/8,256,0,stream>>>(input,gather,bounds,static_cast<__nv_fp8_e4m3*>(a.data),static_cast<uint8_t*>(sfa.data),rows,k,sf_rows,input_stride);
    TRY_CUDA(cudaGetLastError());
    int result;
    if(n==2304){
        static fi_mxfp4_2304_5120_Kernel_Module_t module=[](){fi_mxfp4_2304_5120_Kernel_Module_t m{};fi_mxfp4_2304_5120_Kernel_Module_Load(&m);return m;}();
        fi_mxfp4_2304_5120_Tensor_gA_t aa{a.data,{rows}};
        fi_mxfp4_2304_5120_Tensor_gB_t bb{w.data};
        fi_mxfp4_2304_5120_Tensor_gSFA_t as{sfa.data,{sf_rows},{sf_rows}};
        fi_mxfp4_2304_5120_Tensor_gSFB_t bs{sfb.data};
        fi_mxfp4_2304_5120_Tensor_gD_t dd{out.data,{rows}};
        fi_mxfp4_2304_5120_Tensor_offsets_t offsets{const_cast<int32_t*>(bounds)};
        result=cute_dsl_fi_mxfp4_2304_5120_wrapper(&module,&aa,&bb,&as,&bs,&dd,&offsets,stream);
    }else{
        static fi_mxfp4_5120_2304_Kernel_Module_t module=[](){fi_mxfp4_5120_2304_Kernel_Module_t m{};fi_mxfp4_5120_2304_Kernel_Module_Load(&m);return m;}();
        fi_mxfp4_5120_2304_Tensor_gA_t aa{a.data,{rows}};
        fi_mxfp4_5120_2304_Tensor_gB_t bb{w.data};
        fi_mxfp4_5120_2304_Tensor_gSFA_t as{sfa.data,{sf_rows},{sf_rows}};
        fi_mxfp4_5120_2304_Tensor_gSFB_t bs{sfb.data};
        fi_mxfp4_5120_2304_Tensor_gD_t dd{out.data,{rows}};
        fi_mxfp4_5120_2304_Tensor_offsets_t offsets{const_cast<int32_t*>(bounds)};
        result=cute_dsl_fi_mxfp4_5120_2304_wrapper(&module,&aa,&bb,&as,&bs,&dd,&offsets,stream);
    }
    if(result)return result;
    TRY_CUDA(cudaGetLastError());
    scatter_output<<<unsigned((size_t(rows)*n+255)/256),256,0,stream>>>(static_cast<__nv_bfloat16*>(out.data),scatter,output,rows,n,slot_stride,token_stride);
    return int(cudaGetLastError());
}
