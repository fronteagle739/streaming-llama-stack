// SPDX-License-Identifier: MIT
#include "dense_fp8_bridge.h"
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#define CUTE_DSL_CUDA_ERROR_CHECK(err) do{auto e=(err);if(e!=cudaSuccess){std::fprintf(stderr,"Dense FP8 AOT: %s\n",cudaGetErrorString(e));std::abort();}}while(0)
#include "fi_mxfp8_1280_5120.h"
#include "fi_mxfp8_32768_1280.h"
#include "fi_mxfp8_512_5120.h"
#include "fi_mxfp8_128_512.h"
#include "fi_mxfp8_5120_8192.h"
#include "fi_mxfp8_2304_5120.h"
#include "fi_mxfp8_5120_2304.h"
struct q8_block {__half d;int8_t q[32];};
static_assert(sizeof(q8_block)==34,"GGML Q8_0 layout");
static uint64_t align256(uint64_t n){return (n+255)/256*256;}
static __device__ size_t sf_index(int row,int kb,int k){
    // CUTLASS BlockScaledBasicChunk: ((32,4),(32,4)) strides
    // ((16,4),(0,1)), tiled in K-major order (2,1,3).
    return (size_t(row/128)*(k/128)+kb/4)*512+(row%32)*16+((row%128)/32)*4+kb%4;
}
static __device__ int exponent(float maximum){
    int result=maximum==0?0:ilogbf(maximum)-8;
    if(maximum>ldexpf(448.f,result))++result;
    return max(-127,min(127,result));
}
static __global__ void pack_weights(const void *input,int format,__nv_fp8_e4m3 *b,uint8_t *sf,int n,int k){
    const int kb=blockIdx.x,first=blockIdx.y*32;
    float values[4],maximum=0;
    for(int j=0;j<4;++j){int at=threadIdx.x+j*256,row=first+at/32,column=kb*32+at%32;
        if(format==0){auto q=static_cast<const q8_block*>(input)[size_t(row)*(k/32)+kb];values[j]=__half2float(q.d)*float(q.q[at%32]);}
        else values[j]=float(static_cast<const __nv_fp8_e4m3*>(input)[size_t(row)*k+column]);
        if(!isfinite(values[j]))asm("trap;");maximum=fmaxf(maximum,fabsf(values[j]));
    }
    for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
    __shared__ float warp[8];if(threadIdx.x%32==0)warp[threadIdx.x/32]=maximum;__syncthreads();
    maximum=warp[threadIdx.x%8];for(int d=4;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
    const int power=format==0?exponent(maximum):int(static_cast<const uint8_t*>(input)[size_t(n)*k+size_t(first/32)*(k/32)+kb])-127;
    if(power==128)asm("trap;");
    for(int j=0;j<4;++j){int at=threadIdx.x+j*256,row=first+at/32,column=kb*32+at%32;
        if(format==0)b[size_t(row)*k+column]=__nv_fp8_e4m3(ldexpf(values[j],-power));
        if(at%32==0)sf[sf_index(row,kb,k)]=uint8_t(power+127);
    }
}
static __global__ void pack_input(const float *input,__nv_fp8_e4m3 *a,uint8_t *sf,float *alpha,int m,int valid_m,int k,int64_t stride){
    int block=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;
    if(block==0&&lane==0)*alpha=1.f;
    if(block>=m*(k/32))return;int row=block/(k/32),kb=block%(k/32);
    float value=row<valid_m?input[int64_t(row)*stride+kb*32+lane]:0.f;if(!isfinite(value))asm("trap;");
    float maximum=fabsf(value);for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
    int power=exponent(maximum);a[size_t(row)*k+kb*32+lane]=__nv_fp8_e4m3(ldexpf(value,-power));
    if(!lane)sf[sf_index(row,kb,k)]=uint8_t(power+127);
}
static __global__ void finish(const __nv_bfloat16 *source,float *out,size_t count){
    size_t i=size_t(blockIdx.x)*256+threadIdx.x;if(i<count)out[i]=__bfloat162float(source[i]);
}
#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif
EXPORT int slstack_dense_fp8_abi(){return 1;}
EXPORT int slstack_dense_fp8(const void *weights,int format,const float *input,float *out,int m,int n,int k,int64_t stride,
                            void *workspace,uint64_t bytes,cudaStream_t stream){
    if(!weights||!input||!out||!workspace||(reinterpret_cast<uintptr_t>(workspace)&255)||bytes<(512ull<<20)||
       m<1||m>4096||stride<k||(format!=0&&format!=1))return int(cudaErrorInvalidValue);
    if(!((n==1280&&k==5120)||(n==32768&&k==1280)||(n==512&&k==5120)||(n==128&&k==512)||
         (n==5120&&k==8192)||(n==2304&&k==5120)||(n==5120&&k==2304)))return int(cudaErrorInvalidValue);
    int device;cudaDeviceProp properties;auto error=cudaGetDevice(&device);if(error!=cudaSuccess)return int(error);
    error=cudaGetDeviceProperties(&properties,device);if(error!=cudaSuccess)return int(error);
    if(properties.major!=12||properties.minor!=0)return int(cudaErrorInvalidDeviceFunction);
    const int padded_m=(m+15)/16*16;
    auto *base=static_cast<uint8_t*>(workspace);uint64_t offset=0;
    auto *b=reinterpret_cast<__nv_fp8_e4m3*>(base+offset);offset+=align256(uint64_t(n)*k);
    auto *sb=base+offset;offset+=align256(uint64_t(n)*k/32);
    auto *a=reinterpret_cast<__nv_fp8_e4m3*>(base+offset);offset+=align256(uint64_t(padded_m)*k);
    auto *sa=base+offset;offset+=align256(uint64_t((padded_m+127)/128*128)*k/32);
    auto *result=reinterpret_cast<__nv_bfloat16*>(base+offset);offset+=align256(uint64_t(padded_m)*n*2);
    auto *alpha=reinterpret_cast<float*>(base+offset);offset+=256;if(offset>bytes)return int(cudaErrorMemoryAllocation);
    #define TRY(call) do{int rc=int(call);if(rc)return rc;}while(0)
    TRY(cudaMemsetAsync(sa,0,uint64_t((padded_m+127)/128*128)*k/32,stream));
    pack_weights<<<dim3(k/32,n/32),256,0,stream>>>(weights,format,b,sb,n,k);TRY(cudaGetLastError());
    pack_input<<<(padded_m*(k/32)+7)/8,256,0,stream>>>(input,a,sa,alpha,padded_m,m,k,stride);TRY(cudaGetLastError());
    if(format==1)b=reinterpret_cast<__nv_fp8_e4m3*>(const_cast<void*>(weights));
    #define RUN(N,K) if(n==N&&k==K){ \
        static fi_mxfp8_##N##_##K##_Kernel_Module_t module=[](){fi_mxfp8_##N##_##K##_Kernel_Module_t x{};fi_mxfp8_##N##_##K##_Kernel_Module_Load(&x);return x;}(); \
        fi_mxfp8_##N##_##K##_Tensor_a_t aa{a,{padded_m},{int64_t(padded_m)*k}};fi_mxfp8_##N##_##K##_Tensor_b_t bb{b}; \
        fi_mxfp8_##N##_##K##_Tensor_sa_t as{sa};fi_mxfp8_##N##_##K##_Tensor_sb_t bs{sb}; \
        fi_mxfp8_##N##_##K##_Tensor_out_t dd{result,{padded_m},{int64_t(padded_m)*n}};fi_mxfp8_##N##_##K##_Tensor_alpha_t al{alpha}; \
        TRY(cute_dsl_fi_mxfp8_##N##_##K##_wrapper(&module,&aa,&bb,&as,&bs,&dd,&al,stream));}
    RUN(1280,5120)RUN(32768,1280)RUN(512,5120)RUN(128,512)RUN(5120,8192)RUN(2304,5120)RUN(5120,2304)
    finish<<<unsigned((uint64_t(m)*n+255)/256),256,0,stream>>>(result,out,uint64_t(m)*n);
    return int(cudaGetLastError());
}
