#pragma once
#include "mma.cuh"
#include "ggml-dsv41-index-fp4-policy.h"
#include "../ggml-dsv41-quant.h"

// Experimental representation-preserving indexer. The input has already had
// the model's FP4/E8M0 round trip. Packing must restore EVERY input exactly;
// a conservative integer-range certificate also proves every dot partial sum
// exact in FP32. Otherwise the caller uses its original BF16 cuBLAS path.
static __global__ void dsv41_index_fp4_pack(const float *input,uint32_t *codes,
        uint32_t *scales,int rows,int *proof,int side){
    const int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;
    if(row>=rows)return;
    int low=INT_MAX,high=INT_MIN;uint32_t scale_word=0;
    for(int group=0;group<4;++group){
        const float x=input[uint64_t(row)*128+group*32+lane];
        float maximum=fabsf(x);
        for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
        const float scale=ggml_dsv41_scale(maximum,2);
        const int exponent=int((__float_as_uint(scale)>>23)&255);
        if(exponent<=0||exponent>=255||!isfinite(x))atomicOr(proof,1);
        const float magnitude=__fdiv_rn(fabsf(x),scale);
        int code=0;for(int c=1;c<8;++c)if(magnitude==ggml_dsv41_e2m1_value(c))code=c;
        const float restored=ggml_dsv41_e2m1_value(code)*scale;
        if(restored!=fabsf(x))atomicOr(proof,2);
        code|=signbit(x)?8:0;
        uint32_t word=0;
        for(int d=0;d<8;++d)word|=uint32_t(__shfl_sync(0xffffffff,code,(lane/8)*8+d))<<(4*d);
        if(lane%8==0)codes[uint64_t(row)*16+group*4+lane/8]=word;
        scale_word|=uint32_t(exponent)<<(group*8);
        const uint32_t bits=__float_as_uint(x)&0x7fffffff;
        if(bits){const int e=int(bits>>23)-127;
            low=min(low,e-23+__ffs(int((bits&0x7fffff)|0x800000))-1);high=max(high,e);}
    }
    for(int d=16;d;d/=2){low=min(low,__shfl_down_sync(0xffffffff,low,d));high=max(high,__shfl_down_sync(0xffffffff,high,d));}
    if(!lane){scales[row]=scale_word;if(low!=INT_MAX){atomicMax(proof+1+side,high-low);
        atomicMin(proof+3+side,low);atomicMax(proof+5+side,high);}}
}

static __global__ void dsv41_index_fp4_scores(float *out,const uint32_t *query,const uint32_t *keys,
        const uint32_t *qs,const uint32_t *ks,const float *weights,const int32_t *visible,
        const float *mask,int rows){
    using namespace ggml_cuda_mma;
    // ggml's MMA fragment helpers use threadIdx.x as the lane index.
    const int lane=threadIdx.x,warp=threadIdx.y,token=blockIdx.y;
    const int first=(blockIdx.x*4+warp)*8;
    __shared__ float dots[4][32][8];
    for(int head=0;head<32;head+=16){
        tile<16,8,float> c;
        for(int k=0;k<2;++k){
            tile<16,8,int>a;tile<8,8,int>b;
            // A uses the ldmatrix.x4 register order (not the accumulator's
            // generic 16x8 int coordinates). Read that order directly from
            // global packed rows, avoiding a second shared-memory tile.
            for(int l=0;l<a.ne;++l)a.x[l]=int(query[(uint64_t(token)*32+head+(l%2)*8+lane/4)*16+k*8+(l/2)*4+lane%4]);
            for(int l=0;l<b.ne;++l){const int key=first+b.get_i(l);
                b.x[l]=key<rows?int(keys[uint64_t(key)*16+k*8+b.get_j(l)]):0;}
            const int qrow=head+lane/4+(lane%2)*8,key=first+lane/4;
            const uint32_t as=(qs[token*32+qrow]>>(k*16))&65535;
            const uint32_t bs=key<rows?(ks[key]>>(k*16))&65535:0;
            mma_block_scaled(c,a,b,as,bs);
        }
        for(int l=0;l<c.ne;++l)dots[warp][head+c.get_i(l)][c.get_j(l)]=c.x[l];
    }
    __syncwarp();
    if(lane<8&&first+lane<rows){const int key=first+lane;float result=0;
        if(key>=visible[token]||(mask&&mask[uint64_t(token)*((rows+7)/8)+key/8]!=0.f))result=-CUDART_INF_F;
        else for(int h=0;h<32;++h)result=__fmaf_rn(fmaxf(__fmul_rn(dots[warp][h][lane],1.f/64.f),0.f),weights[token*32+h],result);
        out[uint64_t(token)*rows+key]=result;
    }
}

static __global__ void dsv41_index_fp4_init_proof(int *proof) {
    const int i=threadIdx.x;
    if(i<7)proof[i]=i<3?0:(i<5?INT_MAX:INT_MIN);
}
#if CUDART_VERSION >= 12080
static __global__ void dsv41_index_fp4_condition(const int *proof,cudaGraphConditionalHandle handle,int tokens,int rows) {
    if(threadIdx.x)return;
    const bool exact=ggml_dsv41_index_fp4_certified(proof);
    cudaGraphSetConditional(handle,exact?1:0);
    printf("PERFEXP_INDEX_FP4_GPU_CONDITIONAL tokens=%d keys=%d exact=%d invalid=%d spans=%d,%d original_bf16_else=1\n",
        tokens,rows,int(exact),proof[0],proof[1],proof[2]);
}
#endif
template<typename Fallback>
static bool dsv41_index_fp4(ggml_backend_cuda_context&ctx,ggml_tensor*dst,Fallback fallback){
    if(!ggml_dsv41_index_fp4_enabled(std::getenv("LLAMA_DSV41_INDEX_FP4")))return false;
    GGML_ASSERT(ggml_cuda_info().devices[ctx.device].cc==1200);
    const int tokens=int(dst->ne[1]),rows=int(dst->ne[0]),qrows=tokens*32;
    cudaStreamCaptureStatus entering=cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(),&entering));
    const char *queued=std::getenv("LLAMA_DSV41_ASYNC_LAYER");
    const bool local_capture=entering==cudaStreamCaptureStatusNone&&queued&&queued[0]=='1'&&!queued[1];
    if(local_capture){
#if CUDART_VERSION >= 12080
        GGML_ASSERT(ggml_cuda_context_abi_size()==sizeof(ctx)&&"CUDA context layout mismatch; rebuild with consistent graph flags");
        ctx.reap_dsv41_graphs();
        GGML_ASSERT(ctx.dsv41_deferred_graphs.size()<128&&"Conditional index graph metadata exceeded its bounded queue");
        // Eager queued layers need the same GPU certificate/IF-ELSE decision
        // as outer capture. Capture only this index operation, not the whole
        // changing layer graph. Both numerical branches remain executable.
        (void)ctx.cublas_handle();
        CUDA_CHECK(cudaStreamBeginCapture(ctx.stream(),cudaStreamCaptureModeRelaxed));
#else
        GGML_ABORT("Asynchronous FP4/BF16 selection requires CUDA12.8 or newer");
#endif
    }
    const uint64_t bytes=uint64_t(qrows+rows)*68+7*4;
    GGML_ASSERT(bytes<=64ull*1024*1024&&"FP4 index workspace exceeds existing64MiB reserve");
    ggml_cuda_pool_alloc<uint32_t>q(ctx.pool(),uint64_t(qrows)*16),k(ctx.pool(),uint64_t(rows)*16),qs(ctx.pool(),qrows),ks(ctx.pool(),rows);
    ggml_cuda_pool_alloc<int>proof(ctx.pool(),7);
    int host[7]={0,0,0,INT_MAX,INT_MAX,INT_MIN,INT_MIN};const auto stream=ctx.stream();
    cudaStreamCaptureStatus capturing=cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream,&capturing));
    if(capturing==cudaStreamCaptureStatusActive) {
#if CUDART_VERSION >= 12080
        // Keep the same representation/integer-sum certificate and both
        // arithmetic paths. The GPU selects the branch on every replay.
        dsv41_index_fp4_init_proof<<<1,32,0,stream>>>(proof.get());
        dsv41_index_fp4_pack<<<(qrows+3)/4,128,0,stream>>>((const float*)dst->src[0]->data,q.get(),qs.get(),qrows,proof.get(),0);
        dsv41_index_fp4_pack<<<(rows+3)/4,128,0,stream>>>((const float*)dst->src[2]->data,k.get(),ks.get(),rows,proof.get(),1);
        cudaGraph_t graph=nullptr;cudaGraphConditionalHandle condition;
        CUDA_CHECK(cudaStreamGetCaptureInfo(stream,&capturing,nullptr,&graph,nullptr,nullptr,nullptr));
        CUDA_CHECK(cudaGraphConditionalHandleCreate(&condition,graph,0,0));
        dsv41_index_fp4_condition<<<1,32,0,stream>>>(proof.get(),condition,tokens,rows);
        const cudaGraphNode_t *dependencies=nullptr;size_t dependency_count=0;
        CUDA_CHECK(cudaStreamGetCaptureInfo(stream,&capturing,nullptr,&graph,&dependencies,nullptr,&dependency_count));
        cudaGraphNodeParams params{};params.type=cudaGraphNodeTypeConditional;
        params.conditional.handle=condition;params.conditional.type=cudaGraphCondTypeIf;params.conditional.size=2;
        cudaGraphNode_t branch;
        CUDA_CHECK(cudaGraphAddNode(&branch,graph,dependencies,nullptr,dependency_count,&params));
        cudaStream_t body;
        CUDA_CHECK(cudaStreamCreateWithFlags(&body,cudaStreamNonBlocking));
        for(int arm=0;arm<2;++arm) {
            CUDA_CHECK(cudaStreamBeginCaptureToGraph(body,params.conditional.phGraph_out[arm],nullptr,nullptr,0,cudaStreamCaptureModeRelaxed));
            if(arm==0) {
                dsv41_index_fp4_scores<<<dim3((rows+31)/32,tokens),dim3(32,4),0,body>>>((float*)dst->data,q.get(),k.get(),qs.get(),ks.get(),
                    (const float*)dst->src[1]->data,(const int32_t*)dst->src[3]->data,dst->src[4]?(const float*)dst->src[4]->data:nullptr,rows);
                CUDA_CHECK(cudaGetLastError());
            } else {
                // The branches are mutually exclusive and depend on the
                // completed certificate. Reuse the FP4 packing storage for
                // the original <=64MiB BF16 workspace instead of charging
                // both branches simultaneously. VMM mappings remain live;
                // the captured true branch retains its actual addresses.
                auto release=[](auto &allocation) {
                    allocation.pool->free(allocation.ptr,allocation.actual_size);
                    allocation.ptr=nullptr;
                };
                release(proof);release(ks);release(qs);release(k);release(q);
                fallback(body);
            }
            CUDA_CHECK(cudaStreamEndCapture(body,nullptr));
        }
        CUDA_CHECK(cudaStreamDestroy(body));
        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(),stream));
        CUDA_CHECK(cudaStreamUpdateCaptureDependencies(stream,&branch,nullptr,1,cudaStreamSetCaptureDependencies));
        if(local_capture){
            cudaGraph_t captured=nullptr;cudaGraphExec_t executable=nullptr;
            CUDA_CHECK(cudaStreamEndCapture(stream,&captured));
            CUDA_CHECK(cudaGraphInstantiate(&executable,captured,nullptr,nullptr,0));
            CUDA_CHECK(cudaGraphLaunch(executable,stream));
            // WDDM graph destruction can wait for an in-flight execution.
            // Retire handles only after an event query proves completion.
            cudaEvent_t done;CUDA_CHECK(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(done,stream));ctx.dsv41_deferred_graphs.push_back({captured,executable,done});
        }
        return true;
#else
        GGML_ABORT("FP4/BF16 conditional capture requires CUDA 12.8 or newer; use the original eager policy on older toolkits");
#endif
    }
    CUDA_CHECK(cudaMemcpyAsync(proof.get(),host,sizeof(host),cudaMemcpyHostToDevice,stream));
    dsv41_index_fp4_pack<<<(qrows+3)/4,128,0,stream>>>((const float*)dst->src[0]->data,q.get(),qs.get(),qrows,proof.get(),0);
    dsv41_index_fp4_pack<<<(rows+3)/4,128,0,stream>>>((const float*)dst->src[2]->data,k.get(),ks.get(),rows,proof.get(),1);
    CUDA_CHECK(cudaMemcpyAsync(host,proof.get(),sizeof(host),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const bool exact=ggml_dsv41_index_fp4_certified(host);
    if(!exact){fprintf(stderr,"PERFEXP_INDEX_FP4_FALLBACK tokens=%d keys=%d invalid=%d spans=%d,%d original_bf16=1\n",tokens,rows,host[0],host[1],host[2]);return false;}
    dsv41_index_fp4_scores<<<dim3((rows+31)/32,tokens),dim3(32,4),0,stream>>>((float*)dst->data,q.get(),k.get(),qs.get(),ks.get(),
        (const float*)dst->src[1]->data,(const int32_t*)dst->src[3]->data,dst->src[4]?(const float*)dst->src[4]->data:nullptr,rows);
    CUDA_CHECK(cudaGetLastError());
    fprintf(stderr,"PERFEXP_INDEX_FP4_MMA tokens=%d keys=%d spans=%d,%d packed_bytes=%llu certificate=exact_integer_sum original_epilogue=1\n",tokens,rows,host[1],host[2],(unsigned long long)bytes);
    return true;
}
