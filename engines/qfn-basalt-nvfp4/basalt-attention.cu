// SPDX-License-Identifier: MIT
// Native F16 contract around pinned Basalt prompt and64-cell decode arithmetic.
// Off-tree candidate: no default/cache/selector/other DLL API changes.
#include "qfn_basalt_attention.h"
#include "vendor/basalt/include/basalt/kernels/pdl.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <atomic>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

static_assert(sizeof(qfn_basalt_attn_desc)==120);
static_assert(offsetof(qfn_basalt_attn_desc,q_head_stride)==32);
namespace {
constexpr int D=256;
struct Args {
    qfn_basalt_attn_desc d{};
    const float * q=nullptr,* gate=nullptr;
    const void * k=nullptr,* v=nullptr;
    const int32_t * ids=nullptr,* pos=nullptr;
    float * out=nullptr,* scratch=nullptr;
    int n_chunks=0;uint64_t stride=0,first=0;
};
__host__ __device__ const unsigned char * bytes(const void * p) {return static_cast<const unsigned char *>(p);}
__device__ const float * qptr(const Args & a,int t,int h,int d) {
    return reinterpret_cast<const float *>(bytes(a.q)+uint64_t(t)*a.d.q_token_stride+uint64_t(h)*a.d.q_head_stride)+d;
}
__device__ float * optr(const Args & a,int t,int h,int d) {
    return reinterpret_cast<float *>(reinterpret_cast<unsigned char *>(a.out)+uint64_t(t)*a.d.out_token_stride+uint64_t(h)*a.d.out_head_stride)+d;
}
#include "basalt-attention-prompt.cuh"
#include "basalt-attention-decode.cuh"

std::atomic<bool> ready[64]{};
int shared_limit[64]{};
bool mul(uint64_t a,uint64_t b,uint64_t & out) {if(b&&a>UINT64_MAX/b)return false;out=a*b;return true;}
bool add(uint64_t a,uint64_t b,uint64_t & out) {if(a>UINT64_MAX-b)return false;out=a+b;return true;}
bool span(int a,uint64_t sa,int b,uint64_t sb,uint64_t width,uint64_t & out) {
    if(a==0||b==0){out=0;return true;}
    uint64_t x,y;
    return mul(uint64_t(a-1),sa,x)&&mul(uint64_t(b-1),sb,y)&&add(x,y,x)&&add(x,width,out)&&out<=PTRDIFF_MAX;
}
bool shape(const qfn_basalt_attn_desc & d) {
    if(d.mode<0||d.mode>1||d.nt<=0||d.n_heads<=0||d.n_kv_heads<=0||d.n_kv_heads>65535||
       int64_t(d.n_kv_heads)*12!=d.n_heads||d.n_kv<0||d.cap<0||!std::isfinite(d.scale))return false;
    if((d.q_head_stride|d.q_token_stride|d.k_head_stride|d.k_cell_stride|d.v_head_stride|d.v_cell_stride)&15)return false;
    if((d.gate_head_stride|d.gate_token_stride|d.ids_token_stride)&3)return false;
    if((d.out_head_stride|d.out_token_stride)&7)return false;
    uint64_t outspan=0;
    if(d.out_head_stride<D*4||!span(d.n_heads,d.out_head_stride,1,0,D*4,outspan)||d.out_token_stride<outspan)return false;
    uint64_t n=0;
    if(!span(d.nt,d.q_token_stride,d.n_heads,d.q_head_stride,D*4,n)||
       !span(d.nt,d.gate_token_stride,d.n_heads,d.gate_head_stride,D*4,n)||
       !span(d.n_kv,d.k_cell_stride,d.n_kv_heads,d.k_head_stride,D*2,n)||
       !span(d.n_kv,d.v_cell_stride,d.n_kv_heads,d.v_head_stride,D*2,n)||
       !span(d.nt,d.ids_token_stride,d.cap,4,4,n)||
       !span(d.nt,d.out_token_stride,d.n_heads,d.out_head_stride,D*4,n))return false;
    const uint64_t chunks=(uint64_t(d.cap)+63)/64;
    return d.mode!=QFN_ATTN_DECODE64||chunks*uint64_t(d.n_heads)<=INT_MAX; // published partial index uses signed int
}
uint64_t query(const qfn_basalt_attn_desc & d) {
    if(!shape(d))return UINT64_MAX;
    if(d.mode==QFN_ATTN_PROMPT||d.cap==0||d.n_kv==0)return 0;
    const uint64_t chunks=(uint64_t(d.cap)+63)/64;
    uint64_t n;
    if(!mul(chunks,uint64_t(d.n_heads)*(D+2),n)||!add(n,64,n)||!mul(n,uint64_t(d.nt)*4,n)||n>PTRDIFF_MAX)return UINT64_MAX;
    return n;
}
bool pointer_span(const void * p,uint64_t n) {return n==0||(p&&uintptr_t(p)<=UINTPTR_MAX-n);}
__global__ void finish(Args a,bool zero) {
    if(a.d.nt==0)return;
    const int t=int(a.first+blockIdx.y),h=blockIdx.x,d=threadIdx.x;
    float * out=optr(a,t,h,d);float v=zero?0.0f:*out;
    if(a.gate) {
        const float * gate=reinterpret_cast<const float *>(bytes(a.gate)+uint64_t(t)*a.d.gate_token_stride+uint64_t(h)*a.d.gate_head_stride)+d;
        v*=1.0f/(1.0f+expf(-*gate)); // exact native optional-gate expression
    }
    *out=v;
}
} // namespace
extern "C" int qfn_basalt_attn_abi(void) {return 1;}
extern "C" uint64_t qfn_basalt_attn_workspace_bytes(const qfn_basalt_attn_desc * d) {return d?query(*d):UINT64_MAX;}
extern "C" int qfn_basalt_attn_prepare(void * stream) {
    int dev=0;auto e=cudaGetDevice(&dev);if(e!=cudaSuccess)return int(e);
    if(dev<0||dev>=64)return int(cudaErrorInvalidDevice);
    if(ready[dev].load(std::memory_order_acquire))return 0;
    const auto cs=static_cast<cudaStream_t>(stream);cudaStreamCaptureStatus status;
    e=cudaStreamIsCapturing(cs,&status);if(e!=cudaSuccess)return int(e);
    if(status!=cudaStreamCaptureStatusNone)return int(cudaErrorStreamCaptureUnsupported);
    int major=0,minor=0,limit=0;
    e=cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,dev);
    if(e==cudaSuccess)e=cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,dev);
    if(e==cudaSuccess)e=cudaDeviceGetAttribute(&limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev);
    if(e!=cudaSuccess)return int(e);
    if(major!=12||minor!=0)return int(cudaErrorInvalidDevice); // same actual fatbin target as this DLL
    if(limit<int(sizeof(prompt::Smem<0>))||limit<int(sizeof(decode::ChunkSmem<0>)))return int(cudaErrorNotSupported);
    e=cudaFuncSetAttribute(prompt::kernel<0>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(prompt::Smem<0>));
    if(e==cudaSuccess)e=cudaFuncSetAttribute(decode::chunk_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(decode::ChunkSmem<0>));
    if(e==cudaSuccess)e=cudaFuncSetAttribute(decode::merge_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,limit);
    if(e!=cudaSuccess)return int(e);
    // Cache capability/module lookups outside capture; never silently drop PDL.
    if(!basalt::kernels::pdl_kernel_ok(reinterpret_cast<const void *>(decode::chunk_kernel))||
       !basalt::kernels::pdl_kernel_ok(reinterpret_cast<const void *>(decode::merge_kernel)))return int(cudaErrorNotSupported);
    Args a;
    prompt::kernel<0><<<1,128,sizeof(prompt::Smem<0>),cs>>>(a);
    {
        basalt::kernels::PdlScope scope(true,true);
        basalt::kernels::launch_pdl(decode::chunk_kernel,dim3(1),dim3(256),sizeof(decode::ChunkSmem<0>),cs,a);
        basalt::kernels::launch_pdl(decode::merge_kernel,dim3(1),dim3(256),32,cs,a);
    }
    finish<<<1,256,0,cs>>>(a,true);
    e=cudaGetLastError();if(e==cudaSuccess)e=cudaStreamSynchronize(cs);
    if(e==cudaSuccess){shared_limit[dev]=limit;ready[dev].store(true,std::memory_order_release);}
    return int(e);
}
extern "C" int qfn_basalt_attn_run(const float * q,const float * gate,const void * k,const void * v,
    const int32_t * ids,const int32_t * pos,float * out,const qfn_basalt_attn_desc * desc,
    void * workspace,uint64_t bytes_count,void * stream) {
    if(!desc)return int(cudaErrorInvalidValue);
    const auto d=*desc;const auto need=query(d);
    if(need==UINT64_MAX||!q||!pos||!out||(d.cap>0&&!ids)||(d.n_kv>0&&(!k||!v))||
       (uintptr_t(q)&15)||(uintptr_t(out)&7)||(uintptr_t(pos)&3)||(uintptr_t(ids)&3)||
       (uintptr_t(k)&15)||(uintptr_t(v)&15)||(uintptr_t(gate)&3)||
       (need&&(!workspace||(uintptr_t(workspace)&255)||bytes_count<need)))return int(cudaErrorInvalidValue);
    uint64_t n=0;
    span(d.nt,d.q_token_stride,d.n_heads,d.q_head_stride,D*4,n);if(!pointer_span(q,n))return int(cudaErrorInvalidValue);
    span(d.nt,d.out_token_stride,d.n_heads,d.out_head_stride,D*4,n);if(!pointer_span(out,n))return int(cudaErrorInvalidValue);
    span(d.n_kv,d.k_cell_stride,d.n_kv_heads,d.k_head_stride,D*2,n);if(!pointer_span(k,n))return int(cudaErrorInvalidValue);
    span(d.n_kv,d.v_cell_stride,d.n_kv_heads,d.v_head_stride,D*2,n);if(!pointer_span(v,n))return int(cudaErrorInvalidValue);
    span(d.nt,d.ids_token_stride,d.cap,4,4,n);if(!pointer_span(ids,n))return int(cudaErrorInvalidValue);
    if(gate){span(d.nt,d.gate_token_stride,d.n_heads,d.gate_head_stride,D*4,n);if(!pointer_span(gate,n))return int(cudaErrorInvalidValue);}
    if(!pointer_span(pos,uint64_t(d.nt)*4)||!pointer_span(workspace,need))return int(cudaErrorInvalidValue);
    int dev=0;auto e=cudaGetDevice(&dev);if(e!=cudaSuccess)return int(e);
    if(dev<0||dev>=64||!ready[dev].load(std::memory_order_acquire))return int(cudaErrorNotReady);
    const uint64_t chunks=(uint64_t(d.cap)+63)/64;
    const uint64_t merge_bytes=(2*chunks+8)*4;
    if(d.mode==QFN_ATTN_DECODE64&&need&&merge_bytes>uint64_t(shared_limit[dev]))return int(cudaErrorInvalidConfiguration);
    Args a;a.d=d;a.q=q;a.gate=gate;a.k=k;a.v=v;a.ids=ids;a.pos=pos;a.out=out;
    a.scratch=static_cast<float *>(workspace);a.n_chunks=int(chunks);a.stride=chunks*d.n_heads*(D+2)+64;
    const auto cs=static_cast<cudaStream_t>(stream);
    for(int64_t first=0;first<d.nt;first+=65535) {
        const auto count=unsigned(std::min(int64_t(65535),int64_t(d.nt)-first));a.first=uint64_t(first);
        const bool zero=d.cap==0||d.n_kv==0;
        if(!zero) {
            if(d.mode==QFN_ATTN_PROMPT)prompt::kernel<0><<<dim3(count,unsigned(d.n_kv_heads)),128,sizeof(prompt::Smem<0>),cs>>>(a);
            else {
                basalt::kernels::PdlScope scope(true,true);
                basalt::kernels::launch_pdl(decode::chunk_kernel,dim3(unsigned(chunks),unsigned(d.n_kv_heads),count),dim3(256),sizeof(decode::ChunkSmem<0>),cs,a);
                basalt::kernels::launch_pdl(decode::merge_kernel,dim3(unsigned(d.n_heads),count),dim3(256),merge_bytes,cs,a);
            }
        }
        if(zero||gate)finish<<<dim3(unsigned(d.n_heads),count),256,0,cs>>>(a,zero);
        e=cudaGetLastError();if(e!=cudaSuccess)return int(e);
    }
    return 0;
}
