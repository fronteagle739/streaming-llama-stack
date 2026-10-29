// SPDX-License-Identifier: MIT
// Experimental routed FFN with a single expert-sorted routing map.
#include "flashinfer_native_bridge.h"
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#define CUTE_DSL_CUDA_ERROR_CHECK(err) do { auto e=(err); if(e!=cudaSuccess){ \
    std::fprintf(stderr,"FlashInfer native CUDA failure: %s\n",cudaGetErrorString(e));std::abort();} }while(0)
#include "fi_mxfp4_2304_5120.h"
#include "fi_mxfp4_5120_2304.h"

static int gemm(const void *weights,void *a,void *sa,void *out,const int32_t *bounds,
                int rows,int n,int k,int sf_rows,cudaStream_t stream){
    void *w=const_cast<void*>(weights);
    void *sb=static_cast<uint8_t*>(w)+slstack_fi_native::scale_offset(n,k);
    if(n==2304){
        static fi_mxfp4_2304_5120_Kernel_Module_t module=[](){fi_mxfp4_2304_5120_Kernel_Module_t m{};fi_mxfp4_2304_5120_Kernel_Module_Load(&m);return m;}();
        fi_mxfp4_2304_5120_Tensor_gA_t aa{a,{rows}};
        fi_mxfp4_2304_5120_Tensor_gB_t bb{w};
        fi_mxfp4_2304_5120_Tensor_gSFA_t as{sa,{sf_rows},{sf_rows}};
        fi_mxfp4_2304_5120_Tensor_gSFB_t bs{sb};
        fi_mxfp4_2304_5120_Tensor_gD_t dd{out,{rows}};
        fi_mxfp4_2304_5120_Tensor_offsets_t offsets{const_cast<int32_t*>(bounds)};
        return cute_dsl_fi_mxfp4_2304_5120_wrapper(&module,&aa,&bb,&as,&bs,&dd,&offsets,stream);
    }
    static fi_mxfp4_5120_2304_Kernel_Module_t module=[](){fi_mxfp4_5120_2304_Kernel_Module_t m{};fi_mxfp4_5120_2304_Kernel_Module_Load(&m);return m;}();
    fi_mxfp4_5120_2304_Tensor_gA_t aa{a,{rows}};
    fi_mxfp4_5120_2304_Tensor_gB_t bb{w};
    fi_mxfp4_5120_2304_Tensor_gSFA_t as{sa,{sf_rows},{sf_rows}};
    fi_mxfp4_5120_2304_Tensor_gSFB_t bs{sb};
    fi_mxfp4_5120_2304_Tensor_gD_t dd{out,{rows}};
    fi_mxfp4_5120_2304_Tensor_offsets_t offsets{const_cast<int32_t*>(bounds)};
    return cute_dsl_fi_mxfp4_5120_2304_wrapper(&module,&aa,&bb,&as,&bs,&dd,&offsets,stream);
}
static __device__ int scale_row(int row,const int32_t *bounds){
    int low=0,high=384;
    while(low<high){int mid=(low+high)/2;if(bounds[mid+1]<=row)low=mid+1;else high=mid;}
    return (bounds[low]+low*3)/4*4+row-bounds[low];
}
static __device__ void quantize(float value,int row,int kb,int k,int sf_rows,const int32_t *bounds,
                               __nv_fp8_e4m3 *a,uint8_t *sa){
    if(!isfinite(value))asm("trap;");
    float maximum=fabsf(value);
    #pragma unroll
    for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
    int exponent=maximum==0?0:ilogbf(maximum)-8;
    if(maximum>ldexpf(448.f,exponent))++exponent;
    exponent=max(-127,min(127,exponent));
    int lane=threadIdx.x%32;
    a[size_t(row)*k+kb*32+lane]=__nv_fp8_e4m3(ldexpf(value,-exponent));
    if(!lane)sa[(size_t(kb/4)*sf_rows+scale_row(row,bounds))*4+kb%4]=uint8_t(exponent+127);
}
static __global__ void gather(const float *input,const int32_t *routes,const int32_t *bounds,
                             __nv_fp8_e4m3 *a,uint8_t *sa,int rows,int sf_rows){
    int block=int(blockIdx.x)*8+threadIdx.x/32,lane=threadIdx.x%32;
    if(block>=rows*160)return;
    int row=block/160,kb=block%160;
    float value=input[size_t(routes[row]/6)*5120+kb*32+lane];
    quantize(value,row,kb,5120,sf_rows,bounds,a,sa);
}
static __global__ void mid_pack(const __nv_bfloat16 *gate,const __nv_bfloat16 *up,const float *weights,
                               const int32_t *routes,const int32_t *bounds,__nv_fp8_e4m3 *a,uint8_t *sa,
                               int rows,int sf_rows,float limit){
    int block=int(blockIdx.x)*8+threadIdx.x/32,lane=threadIdx.x%32;
    if(block>=rows*72)return;
    int row=block/72,kb=block%72;size_t i=size_t(row)*2304+kb*32+lane;
    float g=fminf(__bfloat162float(gate[i]),limit);
    float u=fmaxf(-limit,fminf(__bfloat162float(up[i]),limit));
    // Preserve the existing separate SiLU, multiply, weight and F16 store.
    float value=__fmul_rn(__fmul_rn(g/(1.f+expf(-g)),u),weights[routes[row]]);
    value=__half2float(__float2half_rn(value));
    quantize(value,row,kb,2304,sf_rows,bounds,a,sa);
}
static __global__ void inverse_routes(const int32_t *routes,int32_t *inverse,int rows){
    int row=int(blockIdx.x)*blockDim.x+threadIdx.x;if(row<rows)inverse[routes[row]]=row;
}
static __global__ void reduce(const __nv_bfloat16 *down,const int32_t *inverse,float *output,int tokens){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(tokens)*5120)return;
    int token=i/5120,column=i%5120;
    float result=__bfloat162float(down[size_t(inverse[token*6])*5120+column]);
    #pragma unroll
    for(int slot=1;slot<6;++slot)result=__fadd_rn(result,__bfloat162float(down[size_t(inverse[token*6+slot])*5120+column]));
    output[i]=result;
}
#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif
EXPORT int slstack_flashinfer_native_ffn_abi(){return 1;}
EXPORT uint64_t slstack_flashinfer_native_ffn_workspace(int tokens){
    return tokens>0&&tokens<=65536?slstack_fi_native::layout(tokens).bytes+slstack_fi_native::aligned(uint64_t(tokens)*6*4):0;
}
EXPORT int slstack_flashinfer_native_ffn(const void *gate,const void *up,const void *down,
    const float *input,const float *route_weights,const int32_t *routes,const int32_t *bounds,
    float *output,int tokens,float limit,void *workspace,uint64_t bytes,cudaStream_t stream){
    if(!gate||!up||!down||!input||!route_weights||!routes||!bounds||!output||!workspace||
       tokens<1||tokens>65536||!std::isfinite(limit)||limit<=0||
       (reinterpret_cast<uintptr_t>(workspace)&255)||bytes<slstack_flashinfer_native_ffn_workspace(tokens))return int(cudaErrorInvalidValue);
    int device;cudaDeviceProp properties;auto error=cudaGetDevice(&device);if(error!=cudaSuccess)return int(error);
    error=cudaGetDeviceProperties(&properties,device);if(error!=cudaSuccess)return int(error);
    if(properties.major!=12||properties.minor!=0)return int(cudaErrorInvalidDeviceFunction);
    const int rows=tokens*6,sf_rows=(rows+384*3+127)/128*128;
    const auto layout=slstack_fi_native::layout(tokens);auto *base=static_cast<uint8_t*>(workspace);
    auto *a=reinterpret_cast<__nv_fp8_e4m3*>(base);
    auto *sa=base+layout.scale;
    auto *g=reinterpret_cast<__nv_bfloat16*>(base+layout.gate),*u=reinterpret_cast<__nv_bfloat16*>(base+layout.up);
    auto *result=reinterpret_cast<__nv_bfloat16*>(base+layout.down);
    auto *inverse=reinterpret_cast<int32_t*>(base+layout.bytes);
    #define TRY(call) do{int rc=int(call);if(rc)return rc;}while(0)
    TRY(cudaMemsetAsync(sa,0,size_t(sf_rows)*160,stream));
    gather<<<(rows*160+7)/8,256,0,stream>>>(input,routes,bounds,a,sa,rows,sf_rows);
    TRY(cudaGetLastError());
    TRY(gemm(gate,a,sa,g,bounds,rows,2304,5120,sf_rows,stream));
    TRY(gemm(up,a,sa,u,bounds,rows,2304,5120,sf_rows,stream));
    TRY(cudaMemsetAsync(sa,0,size_t(sf_rows)*72,stream));
    mid_pack<<<(rows*72+7)/8,256,0,stream>>>(g,u,route_weights,routes,bounds,a,sa,rows,sf_rows,limit);
    TRY(cudaGetLastError());
    TRY(gemm(down,a,sa,result,bounds,rows,5120,2304,sf_rows,stream));
    inverse_routes<<<(rows+255)/256,256,0,stream>>>(routes,inverse,rows);
    reduce<<<unsigned((size_t(tokens)*5120+255)/256),256,0,stream>>>(result,inverse,output,tokens);
    return int(cudaGetLastError());
}
