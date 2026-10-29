#pragma once
#include "ggml-dsv41-tilelang-layout.h"
#include <cuda.h>
#include <cuda_bf16.h>
#include <mutex>

static void dsv41_tilelang_driver(CUresult code,const char *operation){
    if(code!=CUDA_SUCCESS){const char *why=nullptr;cuGetErrorString(code,&why);
        GGML_ABORT("TileLang %s failed: %s",operation,why?why:"unknown CUDA driver error");}
}
static CUfunction dsv41_tilelang_function(){
    static CUmodule module=nullptr;static CUfunction function=nullptr;static std::once_flag initialized;
    std::call_once(initialized,[]{
        const char *path=std::getenv("LLAMA_DSV41_TILELANG_CUBIN");
        GGML_ASSERT(path&&*path);
        dsv41_tilelang_driver(cuModuleLoad(&module,path),"load pinned SM120 image");
        dsv41_tilelang_driver(cuModuleGetFunction(&function,module,"sparse_attn_kernel__kernel"),"resolve sparse attention");
        dsv41_tilelang_driver(cuFuncSetAttribute(function,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,86016),"admit86KiB shared memory");
    });
    return function;
}
static __global__ void dsv41_tilelang_kv(__nv_bfloat16 *out,const half *raw,const half *comp,
                                        uint64_t raw_elements,uint64_t elements){
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=elements)return;
    out[i]=__float2bfloat16_rn(__half2float(i<raw_elements?raw[i]:comp[i-raw_elements]));
}
static __global__ void dsv41_tilelang_query(__nv_bfloat16 *out,const float *query,uint32_t first,uint32_t rows,uint32_t head){
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*16*512)return;
    const uint32_t row=i/(16*512),d=i%(16*512);
    out[i]=__float2bfloat16_rn(query[(uint64_t(first+row)*64+head)*512+d]);
}
static __global__ void dsv41_tilelang_query_all(__nv_bfloat16 *out,const float *query,uint32_t elements){
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<elements)out[i]=__float2bfloat16_rn(query[i]);
}
static __global__ void dsv41_tilelang_output(float *out,const __nv_bfloat16 *input,uint32_t first,uint32_t rows,uint32_t head){
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*16*512)return;
    const uint32_t row=i/(16*512),d=i%(16*512);
    out[(uint64_t(first+row)*64+head)*512+d]=__bfloat162float(input[i]);
}
static __global__ void dsv41_tilelang_indices(int32_t *out,uint32_t first,uint32_t rows,uint32_t queries,
        uint32_t raw_rows,uint32_t raw_min,uint32_t comp_rows,
        const ggml_dsv41_tilelang_record *records,const uint32_t *counts,uint32_t stride){
    const uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
    if(!ggml_dsv41_tilelang_ids(out+uint64_t(row)*640,first+row,queries,raw_rows,raw_min,comp_rows,records,counts,stride))
        asm volatile("trap;"); // fail loudly; never truncate an invalid sparse list
}
static void dsv41_tilelang_attention(ggml_backend_cuda_context &ctx,ggml_tensor *dst,
        const half *raw,const half *comp,uint32_t raw_rows,uint32_t raw_min,uint32_t comp_rows){
    GGML_ASSERT(ggml_cuda_info().devices[ctx.device].cc==1200&&ctx.device==0);
    const uint32_t queries=uint32_t(dst->ne[2]);GGML_ASSERT(queries&&queries<=4096&&comp_rows<=131072);
    const char *variant=std::getenv("LLAMA_DSV41_TILELANG_ATTENTION");
    if(!variant)variant="2";
    if(variant&&variant[0]=='2'&&!variant[1]){
        static CUmodule module=nullptr;static CUfunction function=nullptr;static std::once_flag initialized;
        std::call_once(initialized,[]{
            const char *path=std::getenv("LLAMA_DSV41_TILELANG_CUBIN");
            if(path&&*path){
                dsv41_tilelang_driver(cuModuleLoad(&module,path),"load native-layout SM120 image override");
            }else{
                // Pinned, licensed, ahead-of-time image: no runtime compiler,
                // asset search path or external deployment file is required.
                alignas(16) static const unsigned char image[]={
#include "dsv41-tilelang-sm120.inc"
                };
                dsv41_tilelang_driver(cuModuleLoadData(&module,image),"load embedded SM120 image");
            }
            dsv41_tilelang_driver(cuModuleGetFunction(&function,module,"native_sparse_attn_kernel__kernel"),"resolve native-layout sparse attention");
            dsv41_tilelang_driver(cuFuncSetAttribute(function,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,86016),"admit86KiB shared memory");
            fprintf(stderr,"KERNELS2_TILELANG_NATIVE active=1 query_tile=all head_group=16 stages=1 kv_staging=once query_staging=once output_copy=0\n");
        });
        const uint64_t kv_elements=uint64_t(raw_rows+comp_rows)*512,q_elements=uint64_t(queries)*64*512;
        // Includes the caller's existing F16 mirrors. Fits the already admitted768MiB pool.
        GGML_ASSERT(kv_elements*4+q_elements*2+uint64_t(queries)*640*4<=640ull*1024*1024);
        ggml_cuda_pool_alloc<__nv_bfloat16> staged(ctx.pool(),kv_elements+q_elements);
        auto *keys=staged.get(),*q=staged.get()+kv_elements;
        dsv41_tilelang_kv<<<uint32_t((kv_elements+255)/256),256,0,ctx.stream()>>>(keys,raw,comp,uint64_t(raw_rows)*512,kv_elements);
        dsv41_tilelang_query_all<<<uint32_t((q_elements+255)/256),256,0,ctx.stream()>>>(q,(const float*)dst->src[0]->data,uint32_t(q_elements));
        ggml_cuda_pool_alloc<int32_t> ids(ctx.pool(),uint64_t(queries)*640);
        dsv41_tilelang_indices<<<(queries+127)/128,128,0,ctx.stream()>>>(ids.get(),0,queries,queries,raw_rows,raw_min,comp_rows,
            (const ggml_dsv41_tilelang_record*)dst->src[3]->data,(const uint32_t*)dst->src[4]->data,uint32_t(dst->src[3]->ne[1]));
        const float *sink=(const float*)dst->src[5]->data;
        float *out=(float*)dst->data;const auto *selected=ids.get();int m=int(queries),n=int(raw_rows+comp_rows);
        void *args[]={&selected,&keys,&out,&q,&sink,&m,&n};
        dsv41_tilelang_driver(cuLaunchKernel(function,queries,4,1,256,1,1,86016,(CUstream)ctx.stream(),args,nullptr),"launch native-layout sparse attention");
        CUDA_CHECK(cudaGetLastError());return;
    }
    const uint64_t kv_elements=uint64_t(raw_rows+comp_rows)*512;
    // Together with the existing half mirrors this is <=384MiB, within the
    // existing768MiB backend temporary reserve. Process queries128 at a time.
    GGML_ASSERT(kv_elements*4+2ull*128*16*512*2+128*640*4<=384ull*1024*1024);
    ggml_cuda_pool_alloc<__nv_bfloat16> kv(ctx.pool(),kv_elements),q(ctx.pool(),128*16*512),o(ctx.pool(),128*16*512);
    ggml_cuda_pool_alloc<int32_t> ids(ctx.pool(),128*640);
    const auto stream=ctx.stream();const auto kernel=dsv41_tilelang_function();
    dsv41_tilelang_kv<<<uint32_t((kv_elements+255)/256),256,0,stream>>>(kv.get(),raw,comp,uint64_t(raw_rows)*512,kv_elements);
    for(uint32_t first=0;first<queries;first+=128){
        const uint32_t rows=std::min(128u,queries-first);
        dsv41_tilelang_indices<<<(rows+127)/128,128,0,stream>>>(ids.get(),first,rows,queries,raw_rows,raw_min,comp_rows,
            (const ggml_dsv41_tilelang_record*)dst->src[3]->data,(const uint32_t*)dst->src[4]->data,uint32_t(dst->src[3]->ne[1]));
        for(uint32_t head=0;head<64;head+=16){
            dsv41_tilelang_query<<<(rows*16*512+255)/256,256,0,stream>>>(q.get(),(const float*)dst->src[0]->data,first,rows,head);
            const float *sinks=(const float*)dst->src[5]->data+head;auto *keys=kv.get();auto *input=q.get();auto *output=o.get();auto *selected=ids.get();
            int b=1,m=int(rows),n=int(raw_rows+comp_rows),topk=640;
            void *args[]={&sinks,&keys,&output,&input,&selected,&b,&m,&n,&topk};
            dsv41_tilelang_driver(cuLaunchKernel(kernel,rows,1,1,256,1,1,86016,(CUstream)stream,args,nullptr),"launch sparse attention");
            dsv41_tilelang_output<<<(rows*16*512+255)/256,256,0,stream>>>((float*)dst->data,o.get(),first,rows,head);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    fprintf(stderr,"PERFEXP_TILELANG_ATTENTION queries=%u raw_rows=%u compressed_rows=%u query_block=128 head_group=16 extra_scratch_bytes=%llu\n",
        queries,raw_rows,comp_rows,(unsigned long long)(kv_elements*2+2ull*128*16*512*2+128*640*4));
}
