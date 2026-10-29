#include "dsv41-layer-fusion.cuh"
#include "ggml-dsv41-hc-collapse.h"
#include "../ggml-dsv41-ops.h"
#include <cstring>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

namespace {
struct binding_graph {
    dsv41_layer_norm_args *device=nullptr,*host=nullptr;
    cudaGraphExec_t executable[2]={};
    cudaEvent_t done[64]={};bool used[64]={};
    cudaStream_t stream;
    uint64_t launches=0,captures=0;
    explicit binding_graph(cudaStream_t s):stream(s){
        CUDA_CHECK(cudaMalloc(&device,sizeof(*device)));
        CUDA_CHECK(cudaMallocHost(&host,64*sizeof(*host)));
        for(auto &event:done)CUDA_CHECK(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
    }
    ~binding_graph(){
        // Own completion events rather than the backend stream's lifetime.
        // They remain valid even when the backend retires its stream first.
        for(unsigned i=0;i<64;++i)if(used[i])CUDA_CHECK(cudaEventSynchronize(done[i]));
        for(auto e:executable)if(e)CUDA_CHECK(cudaGraphExecDestroy(e));
        for(auto e:done)CUDA_CHECK(cudaEventDestroy(e));
        CUDA_CHECK(cudaFreeHost(host));CUDA_CHECK(cudaFree(device));
    }
    void run(const dsv41_layer_norm_args &a){
        const int kind=a.width<1024?0:1;
        if(!executable[kind]){
            cudaGraph_t graph;cudaGraphNode_t node;CUDA_CHECK(cudaGraphCreate(&graph,0));
            dsv41_layer_norm_args ignored{};void *params[]={&ignored,&device};
            cudaKernelNodeParams launch{};
            launch.func=kind==0?(void*)dsv41_layer_norm_kernel<256,true>:(void*)dsv41_layer_norm_kernel<1024,true>;
            launch.gridDim=dim3(4096);launch.blockDim=dim3(kind==0?256:1024);
            launch.sharedMemBytes=(5120+32)*4;launch.kernelParams=params;
            CUDA_CHECK(cudaGraphAddKernelNode(&node,graph,nullptr,0,&launch));
            CUDA_CHECK(cudaGraphInstantiate(&executable[kind],graph,nullptr,nullptr,0));
            CUDA_CHECK(cudaGraphDestroy(graph));++captures;
        }
        const unsigned slot=unsigned(launches%64);
        // A completed event protects each pinned descriptor's source lifetime.
        // This ring is bounded and never waits or overwrites an in-flight slot.
        if(used[slot])GGML_ASSERT(cudaEventQuery(done[slot])==cudaSuccess&&"Binding table exceeded64 in-flight submissions");
        host[slot]=a;
        CUDA_CHECK(cudaMemcpyAsync(device,host+slot,sizeof(a),cudaMemcpyHostToDevice,stream));
        CUDA_CHECK(cudaGraphLaunch(executable[kind],stream));
        CUDA_CHECK(cudaEventRecord(done[slot],stream));used[slot]=true;++launches;
        std::fprintf(stderr,"OTHER22_BINDING_GRAPH launches=%llu captures=%llu pointer_table=1 mutable_rows=1\n",
            (unsigned long long)launches,(unsigned long long)captures);
    }
};
struct binding_graph_pool:ggml_cuda_layer_graph_cache {
    std::unordered_map<cudaStream_t,std::unique_ptr<binding_graph>> streams;
    size_t executable_count()const override{size_t count=0;for(const auto &s:streams)for(auto e:s.second->executable)count+=bool(e);return count;}
    void run(cudaStream_t stream,const dsv41_layer_norm_args &a){
        auto &graph=streams[stream];
        if(!graph){GGML_ASSERT(streams.size()<=GGML_CUDA_MAX_STREAMS);graph=std::make_unique<binding_graph>(stream);}
        graph->run(a);
    }
};
}

bool ggml_cuda_dsv41_layer_norm_match(const ggml_cgraph *g,int i){
    const char *mode=std::getenv("LLAMA_DSV41_LAYER_FUSION");
    if(!mode||!std::strcmp(mode,"0"))return false;
    GGML_ASSERT((!std::strcmp(mode,"1")||!std::strcmp(mode,"2"))&&"Layer fusion requires0/1/2");
    if(i+3>=g->n_nodes)return false;
    auto *hc=g->nodes[i],*norm=g->nodes[i+1],*mul=g->nodes[i+2],*out=g->nodes[i+3];
    if(!ggml_dsv41_hc_collapse_valid(hc)||norm->op!=GGML_OP_RMS_NORM||norm->src[0]!=hc||
        mul->op!=GGML_OP_MUL||mul->src[0]!=norm||!mul->src[1]||
        out->op!=GGML_OP_DSV4_ACT_QAT||out->op_params[0]!=GGML_DSV41_QAT_BF16||out->src[0]!=mul)return false;
    const auto *w=mul->src[1];
    if(hc->ne[0]>5120||w->type!=GGML_TYPE_F32||w->ne[0]!=hc->ne[0]||w->ne[1]!=1||w->ne[2]!=1||w->ne[3]!=1||
        norm->type!=GGML_TYPE_F32||mul->type!=GGML_TYPE_F32||out->type!=GGML_TYPE_F32||
        !ggml_are_same_shape(hc,norm)||!ggml_are_same_shape(norm,mul)||!ggml_are_same_shape(mul,out)||
        !ggml_is_contiguous(w)||!ggml_is_contiguous(norm)||!ggml_is_contiguous(mul)||!ggml_is_contiguous(out))return false;
    return ggml_can_fuse_subgraph(g,i,{GGML_OP_MAP_CUSTOM3,GGML_OP_RMS_NORM,GGML_OP_MUL,GGML_OP_DSV4_ACT_QAT},{i+3});
}

void ggml_cuda_dsv41_layer_norm(ggml_backend_cuda_context &ctx,ggml_cgraph *g,int i){
    GGML_ASSERT(ggml_cuda_context_abi_size()==sizeof(ctx)&&"CUDA context layout mismatch; rebuild with consistent graph flags");
    auto *hc=g->nodes[i],*norm=g->nodes[i+1],*mul=g->nodes[i+2],*out=g->nodes[i+3];
    dsv41_layer_norm_args a{(const float*)hc->src[0]->data,(const float*)hc->src[1]->data,
        (const float*)mul->src[1]->data,(float*)out->data,int(out->ne[0]),int(ggml_nrows(out)),0};
    std::memcpy(&a.eps,norm->op_params,4);
    const char *mode=std::getenv("LLAMA_DSV41_LAYER_FUSION");
    if(mode&&!std::strcmp(mode,"2")&&ctx.cuda_graph_disable_depth.load(std::memory_order_acquire)==0){
        const char *policy=std::getenv("GGML_CUDA_DISABLE_GRAPHS");
        GGML_ASSERT((!policy||!std::strcmp(policy,"0"))&&"Pointer-table mode requires CUDA graphs enabled");
        cudaStreamCaptureStatus status;CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(),&status));
        GGML_ASSERT(status==cudaStreamCaptureStatusNone&&"Pointer-table fusion graph owns its capture; select mode1 for outer graph composition");
        if(!ctx.dsv41_layer_graph)ctx.dsv41_layer_graph=std::make_shared<binding_graph_pool>();
        std::static_pointer_cast<binding_graph_pool>(ctx.dsv41_layer_graph)->run(ctx.stream(),a);
    }else if(a.width<1024)dsv41_layer_norm_kernel<256><<<std::min(a.rows,4096),256,(a.width+32)*4,ctx.stream()>>>(a);
    else dsv41_layer_norm_kernel<1024><<<std::min(a.rows,4096),1024,(a.width+32)*4,ctx.stream()>>>(a);
    CUDA_CHECK(cudaGetLastError());
    std::fprintf(stderr,"OTHER22_LAYER_FUSION rows=%d width=%d HC_BF16_RMS_weight_BF16=1\n",a.rows,a.width);
}
#else
bool ggml_cuda_dsv41_layer_norm_match(const ggml_cgraph *,int){
    const char *mode=std::getenv("LLAMA_DSV41_LAYER_FUSION");
    GGML_ASSERT((!mode||!std::strcmp(mode,"0"))&&"Layer fusion requires CUDA; original HIP/MUSA operations remain available");
    return false;
}
void ggml_cuda_dsv41_layer_norm(ggml_backend_cuda_context &,ggml_cgraph *,int){GGML_ABORT("Layer fusion requires CUDA");}
#endif
