// Operator-only bounded reuse of cuBLAS grouped GEMM. No production dispatch.
// Routing/gather/scatter and Q4_K conversion are the existing expert fallback.
#include "common.cuh"
#include "convert.cuh"
#include "getrows.cuh"
#include "ggml-dsv41-expert-cublas.h"
#include <array>
#include <vector>
#include <algorithm>

void ggml_cuda_dsv41_expert_grouped_trial(ggml_backend_cuda_context &ctx,ggml_tensor *dst,
        uint32_t panel,uint64_t admitted,bool tf32){
    GGML_ASSERT(ggml_dsv41_expert_cublas_valid(dst)&&panel&&panel<=16);
    const auto*w=dst->src[0];const auto*x=dst->src[1];const auto*ids=dst->src[2];
    const int k=int(w->ne[0]),n=int(w->ne[1]),tokens=int(x->ne[2]),slots=tokens*6,channels=int(x->ne[1]);
    const uint64_t required=uint64_t(slots)*k*(tf32?6:2)+uint64_t(slots)*n*(tf32?4:2)+
        uint64_t(k)*n*(tf32?4*panel+2:2*panel)+uint64_t(slots)*8+uint64_t(panel)*3*sizeof(void*)+2ull*1024*1024;
    GGML_ASSERT(required<=admitted);
    auto stream=ctx.stream();std::vector<int32_t> host_ids(slots),maps(2*slots);std::array<int32_t,385> bounds{};
    CUDA_CHECK(cudaMemcpyAsync(host_ids.data(),ids->data,size_t(slots)*4,cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for(int t=0;t<tokens;++t)for(int rank=0;rank<6;++rank){const int e=host_ids[t*6+rank];
        GGML_ASSERT(e>=0&&e<384);for(int p=0;p<rank;++p)GGML_ASSERT(host_ids[t*6+p]!=e);++bounds[e+1];}
    for(int e=1;e<=384;++e)bounds[e]+=bounds[e-1];auto cursor=bounds;
    for(int t=0;t<tokens;++t)for(int rank=0;rank<6;++rank){const int original=t*6+rank,e=host_ids[original],sorted=cursor[e]++;
        maps[sorted]=t*channels+rank%channels;maps[slots+original]=sorted;}
    ggml_cuda_pool_alloc<int32_t> device_maps(ctx.pool(),size_t(2)*slots);
    ggml_cuda_pool_alloc<half> sorted_half(ctx.pool(),size_t(slots)*k);
    ggml_cuda_pool_alloc<char> weights(ctx.pool(),uint64_t(k)*n*panel*(tf32?4:2));
    ggml_cuda_pool_alloc<char> sorted_y(ctx.pool(),uint64_t(slots)*n*(tf32?4:2));
    ggml_cuda_pool_alloc<void*> pointers(ctx.pool(),size_t(panel)*3);
    ggml_cuda_pool_alloc<float> sorted_float(ctx.pool());ggml_cuda_pool_alloc<half> one_half(ctx.pool());
    if(tf32){sorted_float.alloc(size_t(slots)*k);one_half.alloc(size_t(k)*n);}
    CUDA_CHECK(cudaMemcpyAsync(device_maps.get(),maps.data(),size_t(slots)*8,cudaMemcpyHostToDevice,stream));
    get_rows_cuda(x->data,GGML_TYPE_F32,device_maps.get(),sorted_half.get(),GGML_TYPE_F16,
        k,x->nb[1],x->nb[2],x->nb[3],slots,1,1,4,size_t(slots)*4,size_t(slots)*4,
        size_t(k)*2,size_t(slots)*k*2,size_t(slots)*k*2,stream);
    const auto to_half=ggml_get_to_fp16_cuda(GGML_TYPE_Q4_K);
    const auto to_float=ggml_get_to_fp32_cuda(GGML_TYPE_F16);
    GGML_ASSERT(to_half&&to_float);
    if(tf32)to_float(sorted_half.get(),sorted_float.get(),int64_t(slots)*k,stream);
    auto handle=ctx.cublas_handle();CUBLAS_CHECK(cublasSetStream(handle,stream));
    CUBLAS_CHECK(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST));
    std::array<cublasOperation_t,16> transa{},transb{};
    std::array<int,16> ms{},ns{},ks{},lda{},ldb{},ldc{},sizes{};
    std::array<float,16> alphas{},betas{};
    std::array<void*,48> host_pointers{};
    for(int first=0;first<384;){
        int groups=0;
        while(first<384&&groups<int(panel)){
            const int e=first++,count=bounds[e+1]-bounds[e];if(!count)continue;
            void *weight=weights.get()+uint64_t(groups)*k*n*(tf32?4:2);
            if(tf32){to_half((const char*)w->data+size_t(e)*w->nb[2],one_half.get(),int64_t(k)*n,stream);
                to_float(one_half.get(),static_cast<float*>(weight),int64_t(k)*n,stream);}
            else to_half((const char*)w->data+size_t(e)*w->nb[2],static_cast<half*>(weight),int64_t(k)*n,stream);
            transa[groups]=CUBLAS_OP_T;transb[groups]=CUBLAS_OP_N;
            ms[groups]=n;ns[groups]=count;ks[groups]=k;lda[groups]=k;ldb[groups]=k;ldc[groups]=n;sizes[groups]=1;
            alphas[groups]=1.f;betas[groups]=0.f;
            host_pointers[groups]=weight;
            host_pointers[panel+groups]=tf32?static_cast<void*>(sorted_float.get()+size_t(bounds[e])*k):static_cast<void*>(sorted_half.get()+size_t(bounds[e])*k);
            host_pointers[2*panel+groups]=sorted_y.get()+uint64_t(bounds[e])*n*(tf32?4:2);
            ++groups;
        }
        if(!groups)continue;
        CUDA_CHECK(cudaMemcpyAsync(pointers.get(),host_pointers.data(),size_t(panel)*3*sizeof(void*),cudaMemcpyHostToDevice,stream));
        CUBLAS_CHECK(cublasGemmGroupedBatchedEx(handle,transa.data(),transb.data(),ms.data(),ns.data(),ks.data(),alphas.data(),
            (const void*const*)pointers.get(),tf32?CUDA_R_32F:CUDA_R_16F,lda.data(),
            (const void*const*)(pointers.get()+panel),tf32?CUDA_R_32F:CUDA_R_16F,ldb.data(),betas.data(),
            pointers.get()+2*panel,tf32?CUDA_R_32F:CUDA_R_16F,ldc.data(),groups,sizes.data(),
            tf32?CUBLAS_COMPUTE_32F_FAST_TF32:CUBLAS_COMPUTE_32F));
        // Stack parameter/pointer arrays are reused only after this invocation
        // has consumed them. The same admitted panel is then reused, not grown.
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    get_rows_cuda(sorted_y.get(),tf32?GGML_TYPE_F32:GGML_TYPE_F16,device_maps.get()+slots,dst->data,GGML_TYPE_F32,
        n,size_t(n)*(tf32?4:2),size_t(slots)*n*(tf32?4:2),size_t(slots)*n*(tf32?4:2),
        slots,1,1,4,size_t(slots)*4,size_t(slots)*4,dst->nb[1],dst->nb[2],dst->nb[3],stream);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));
}
