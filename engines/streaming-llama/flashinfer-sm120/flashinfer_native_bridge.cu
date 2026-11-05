// SPDX-License-Identifier: MIT
// Experimental routed FFN with a single expert-sorted routing map.
#include "flashinfer_native_bridge.h"
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cuda.h>
#define CUTE_DSL_CUDA_ERROR_CHECK(err) do { auto e=(err); if(e!=cudaSuccess){ \
    std::fprintf(stderr,"FlashInfer native CUDA failure: %s\n",cudaGetErrorString(e));std::abort();} }while(0)
#include "fi_mxfp4_2304_5120.h"
#include "fi_mxfp4_5120_2304.h"
#ifdef SLSTACK_FI_SLOT_LAYOUT
#include "fi_mxfp4_2304_5120_slots.h"
#include "fi_mxfp4_5120_2304_slots.h"
#endif

static int gemm(const void *weights,void *a,void *sa,void *out,const int32_t *bounds,
                int rows,int n,int k,int sf_rows,cudaStream_t stream,bool slots=false){
    void *w=const_cast<void*>(weights);
    void *sb=static_cast<uint8_t*>(w)+(slots?uint64_t(n)*k/2:slstack_fi_native::scale_offset(n,k));
#ifdef SLSTACK_FI_SLOT_LAYOUT
    if(slots){
        if(n==2304){
            static fi_mxfp4_2304_5120_slots_Kernel_Module_t module=[](){fi_mxfp4_2304_5120_slots_Kernel_Module_t m{};fi_mxfp4_2304_5120_slots_Kernel_Module_Load(&m);return m;}();
            fi_mxfp4_2304_5120_slots_Tensor_gA_t aa{a,{rows}};
            fi_mxfp4_2304_5120_slots_Tensor_gB_t bb{w};
            fi_mxfp4_2304_5120_slots_Tensor_gSFA_t as{sa,{sf_rows},{sf_rows}};
            fi_mxfp4_2304_5120_slots_Tensor_gSFB_t bs{sb};
            fi_mxfp4_2304_5120_slots_Tensor_gD_t dd{out,{rows}};
            fi_mxfp4_2304_5120_slots_Tensor_offsets_t offsets{const_cast<int32_t*>(bounds)};
            return cute_dsl_fi_mxfp4_2304_5120_slots_wrapper(&module,&aa,&bb,&as,&bs,&dd,&offsets,stream);
        }
        static fi_mxfp4_5120_2304_slots_Kernel_Module_t module=[](){fi_mxfp4_5120_2304_slots_Kernel_Module_t m{};fi_mxfp4_5120_2304_slots_Kernel_Module_Load(&m);return m;}();
        fi_mxfp4_5120_2304_slots_Tensor_gA_t aa{a,{rows}};
        fi_mxfp4_5120_2304_slots_Tensor_gB_t bb{w};
        fi_mxfp4_5120_2304_slots_Tensor_gSFA_t as{sa,{sf_rows},{sf_rows}};
        fi_mxfp4_5120_2304_slots_Tensor_gSFB_t bs{sb};
        fi_mxfp4_5120_2304_slots_Tensor_gD_t dd{out,{rows}};
        fi_mxfp4_5120_2304_slots_Tensor_offsets_t offsets{const_cast<int32_t*>(bounds)};
        return cute_dsl_fi_mxfp4_5120_2304_slots_wrapper(&module,&aa,&bb,&as,&bs,&dd,&offsets,stream);
    }
#else
    if(slots)return int(cudaErrorNotSupported);
#endif
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
static __global__ void scale_rows(const int32_t *bounds,int32_t *row_offsets){
    const int expert=int(blockIdx.x),first=bounds[expert],end=bounds[expert+1];
    const int aligned=(first+expert*3)/4*4;
    for(int row=first+int(threadIdx.x);row<end;row+=int(blockDim.x))
        row_offsets[row]=aligned+row-first;
}
static __device__ void quantize(float value,int row,int kb,int k,int sf_rows,const int32_t *row_offsets,
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
    if(!lane)sa[(size_t(kb/4)*sf_rows+row_offsets[row])*4+kb%4]=uint8_t(exponent+127);
}
static __global__ void gather(const float *input,const int32_t *routes,const int32_t *row_offsets,
                             __nv_fp8_e4m3 *a,uint8_t *sa,int rows,int sf_rows){
    int block=int(blockIdx.x)*8+threadIdx.x/32,lane=threadIdx.x%32;
    if(block>=rows*160)return;
    int row=block/160,kb=block%160;
    float value=input[size_t(routes[row]/6)*5120+kb*32+lane];
    quantize(value,row,kb,5120,sf_rows,row_offsets,a,sa);
}
static __global__ void mid_pack(const __nv_bfloat16 *gate,const __nv_bfloat16 *up,const float *weights,
                               const int32_t *routes,const int32_t *row_offsets,__nv_fp8_e4m3 *a,uint8_t *sa,
                               int rows,int sf_rows,float limit){
    int block=int(blockIdx.x)*8+threadIdx.x/32,lane=threadIdx.x%32;
    if(block>=rows*72)return;
    int row=block/72,kb=block%72;size_t i=size_t(row)*2304+kb*32+lane;
    float g=fminf(__bfloat162float(gate[i]),limit);
    float u=fmaxf(-limit,fminf(__bfloat162float(up[i]),limit));
    // Preserve the existing separate SiLU, multiply, weight and F16 store.
    float value=__fmul_rn(__fmul_rn(g/(1.f+expf(-g)),u),weights[routes[row]]);
    value=__half2float(__float2half_rn(value));
    quantize(value,row,kb,2304,sf_rows,row_offsets,a,sa);
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
static __global__ void writer_dots(const float*a,const __nv_fp8_e4m3*activation,const uint8_t*scales,
        const int32_t*bounds,const int32_t*row_offsets,float*dots,int rank,int sf_rows){
    const int row=int(blockIdx.x),d=int(blockIdx.y);int low=0,high=384;
    while(low+1<high){const int mid=(low+high)/2;if(bounds[mid]<=row)low=mid;else high=mid;}
    const int expert=low;float sum=0;
    for(int k=threadIdx.x;k<2304;k+=blockDim.x){const int kb=k/32;
        const uint8_t exponent=scales[(size_t(kb/4)*sf_rows+row_offsets[row])*4+kb%4];
        const float x=ldexpf(float(activation[size_t(row)*2304+k]),int(exponent)-127);
        sum+=a[(size_t(expert)*rank+d)*2304+k]*x;
    }
    __shared__ float partial[128];partial[threadIdx.x]=sum;__syncthreads();
    for(int step=64;step;step/=2){if(threadIdx.x<step)partial[threadIdx.x]+=partial[threadIdx.x+step];__syncthreads();}
    if(!threadIdx.x)dots[size_t(row)*rank+d]=partial[0];
}
static __global__ void reduce_writer(const __nv_bfloat16*down,const int32_t*inverse,const float*dots,
        const float*vectors,float*output,int tokens,int rank){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(tokens)*5120)return;
    const int token=int(i/5120),column=int(i%5120);float result=0;
    for(int slot=0;slot<6;++slot){const int row=inverse[token*6+slot];float value=__bfloat162float(down[size_t(row)*5120+column]);
        for(int d=0;d<rank;++d)value+=(-vectors[size_t(d)*5121]*vectors[size_t(d)*5121+1+column])*dots[size_t(row)*rank+d];
        result=slot?__fadd_rn(result,value):value;
    }
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
#ifdef SLSTACK_FI_SLOT_LAYOUT
// Refuse a bad packed-slot call before any kernel can read outside the pool.
static int slot_pool_fault(const char *what,long long a,long long b){
    std::fprintf(stderr,"DSVFIX_SLOT_BOUNDS_FAULT %s: %lld versus %lld\n",what,a,b);
    return int(cudaErrorInvalidValue);
}
static int check_slot_pool(const void *weights,const char *name,int n,int k,int slots){
    CUdeviceptr base=0;size_t size=0;
    const auto start=reinterpret_cast<CUdeviceptr>(weights);
    const auto error=cuMemGetAddressRange(&base,&size,start);
    if(error!=CUDA_SUCCESS){
        std::fprintf(stderr,"DSVFIX_SLOT_BOUNDS_FAULT %s: address range query failed with %d\n",name,int(error));
        return int(cudaErrorInvalidValue);
    }
    const auto needed=uint64_t(slots)*(uint64_t(n)*k/2+uint64_t(n)*k/32);
    const auto available=uint64_t(base+size-start);
    return needed>available?slot_pool_fault(name,(long long)needed,(long long)available):0;
}
static int check_slot_bounds(const int32_t *bounds,int rows,int slots,cudaStream_t stream){
    int32_t host[385];
    int error=int(cudaMemcpyAsync(host,bounds,sizeof(host),cudaMemcpyDeviceToHost,stream));
    if(!error)error=int(cudaStreamSynchronize(stream));
    if(error)return error;
    if(host[0]!=0)return slot_pool_fault("first bound",host[0],0);
    for(int e=0;e<384;++e)if(host[e+1]<host[e])return slot_pool_fault("descending bounds",host[e+1],host[e]);
    if(host[384]!=rows)return slot_pool_fault("total routed rows",host[384],rows);
    if(host[slots]!=rows)return slot_pool_fault("rows routed beyond the slot pool",host[slots],rows);
    return 0;
}
#endif
static int native_ffn(const void *gate,const void *up,const void *down,
    const float *input,const float *route_weights,const int32_t *routes,const int32_t *bounds,
    float *output,int tokens,float limit,void *workspace,uint64_t bytes,cudaStream_t stream,const slstack_fi_writer_delta*delta,int weight_slots=0){
    const bool slots=weight_slots>0;
    if(!gate||!up||!down||!input||!route_weights||!routes||!bounds||!output||!workspace||
       tokens<1||tokens>65536||!std::isfinite(limit)||limit<=0||
       (reinterpret_cast<uintptr_t>(workspace)&255)||bytes<slstack_flashinfer_native_ffn_workspace(tokens))return int(cudaErrorInvalidValue);
    if(delta&&(delta->version!=1||!delta->rank||delta->rank>8||!delta->a||!delta->vectors||!delta->dots))return int(cudaErrorInvalidValue);
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
    // Optional stage trace for component checks: a timed-out kernel is named by
    // the last completed stage. Slot path only; the full-plane path is untouched.
    static const bool trace_enabled=[]{char *v=nullptr;size_t l=0;const bool on=!_dupenv_s(&v,&l,"SLSTACK_FI_STAGE_TRACE")&&v;std::free(v);return on;}();
    const bool trace=slots&&trace_enabled;
    auto stage_start=std::chrono::steady_clock::now();
    #define STAGE(name) do{if(trace){TRY(cudaStreamSynchronize(stream));const auto now_=std::chrono::steady_clock::now();\
        std::fprintf(stderr,"DSVFIX_STAGE %s ms=%.3f\n",name,std::chrono::duration<double,std::milli>(now_-stage_start).count());std::fflush(stderr);stage_start=now_;}}while(0)
#ifdef SLSTACK_FI_SLOT_LAYOUT
    if(slots){
        if(weight_slots>384)return slot_pool_fault("weight slots",weight_slots,384);
        TRY(check_slot_pool(gate,"gate pool",2304,5120,weight_slots));
        TRY(check_slot_pool(up,"up pool",2304,5120,weight_slots));
        TRY(check_slot_pool(down,"down pool",5120,2304,weight_slots));
        TRY(check_slot_bounds(bounds,rows,weight_slots,stream));
        STAGE("bounds_checked");
    }
#endif
    // Reuse the late inverse-map storage for padded scale rows. Every expert's
    // interval is disjoint, including empty experts. Both packing kernels use
    // this map before inverse_routes replaces it on the same CUDA stream.
    scale_rows<<<384,256,0,stream>>>(bounds,inverse);
    TRY(cudaGetLastError());
    TRY(cudaMemsetAsync(sa,0,size_t(sf_rows)*160,stream));
    gather<<<(rows*160+7)/8,256,0,stream>>>(input,routes,inverse,a,sa,rows,sf_rows);
    TRY(cudaGetLastError());
    STAGE("gather");
    TRY(gemm(gate,a,sa,g,bounds,rows,2304,5120,sf_rows,stream,slots));
    STAGE("gate");
    TRY(gemm(up,a,sa,u,bounds,rows,2304,5120,sf_rows,stream,slots));
    STAGE("up");
    TRY(cudaMemsetAsync(sa,0,size_t(sf_rows)*72,stream));
    mid_pack<<<(rows*72+7)/8,256,0,stream>>>(g,u,route_weights,routes,inverse,a,sa,rows,sf_rows,limit);
    TRY(cudaGetLastError());
    STAGE("mid_pack");
    TRY(gemm(down,a,sa,result,bounds,rows,5120,2304,sf_rows,stream,slots));
    STAGE("down");
    if(delta){writer_dots<<<dim3(rows,delta->rank),128,0,stream>>>(delta->a,a,sa,bounds,inverse,delta->dots,int(delta->rank),sf_rows);TRY(cudaGetLastError());}
    inverse_routes<<<(rows+255)/256,256,0,stream>>>(routes,inverse,rows);
    if(delta)reduce_writer<<<unsigned((size_t(tokens)*5120+255)/256),256,0,stream>>>(result,inverse,delta->dots,delta->vectors,output,tokens,int(delta->rank));
    else reduce<<<unsigned((size_t(tokens)*5120+255)/256),256,0,stream>>>(result,inverse,output,tokens);
    return int(cudaGetLastError());
}
EXPORT int slstack_flashinfer_native_ffn(const void*g,const void*u,const void*d,const float*x,const float*w,
    const int32_t*r,const int32_t*b,float*y,int n,float limit,void*scratch,uint64_t bytes,cudaStream_t stream){
    return native_ffn(g,u,d,x,w,r,b,y,n,limit,scratch,bytes,stream,nullptr);
}
EXPORT int slstack_flashinfer_native_ffn_rank_v1(const void*g,const void*u,const void*d,const float*x,const float*w,
    const int32_t*r,const int32_t*b,float*y,int n,float limit,void*scratch,uint64_t bytes,cudaStream_t stream,const slstack_fi_writer_delta*delta){
    return native_ffn(g,u,d,x,w,r,b,y,n,limit,scratch,bytes,stream,delta);
}
#ifdef SLSTACK_FI_SLOT_LAYOUT
EXPORT int slstack_flashinfer_native_ffn_slots_v2(const void*g,const void*u,const void*d,const float*x,const float*w,
    const int32_t*r,const int32_t*b,float*y,int n,float limit,void*scratch,uint64_t bytes,cudaStream_t stream,const slstack_fi_writer_delta*delta,int weight_slots){
    if(weight_slots<1)return slot_pool_fault("weight slots",weight_slots,1);
    return native_ffn(g,u,d,x,w,r,b,y,n,limit,scratch,bytes,stream,delta,weight_slots);
}
#endif
