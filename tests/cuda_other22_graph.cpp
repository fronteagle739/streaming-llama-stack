#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#include "../engines/streaming-llama/prefill_llama.cpp/src/llama-dsv41-graph.h"
#include "ggml-dsv41-bf16-vector.h"
#include "ggml-dsv41-index-cublas.h"
#include "../engines/streaming-llama/prefill_llama.cpp/ggml/src/ggml-dsv41-quant.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
static void require(bool v,const char*why) {if(!v){std::fprintf(stderr,"FAIL %s\n",why);std::exit(2);}}
int main() {
    _putenv_s("GGML_CUDA_DISABLE_GRAPHS","");_putenv_s("LLAMA_DSV41_CUDA_GRAPHS","1");
    _putenv_s("LLAMA_DSV41_BF16_VECTOR","1");
    auto backend=ggml_backend_cuda_init(0);require(backend,"backend init");
    require(ggml_backend_cuda_graphs_reserve(backend,64ull<<20),"pre-map admitted temporary workspace");
    constexpr int count=512;
    uint64_t calls=ggml_backend_cuda_dsv41_bf16_vector_queued();
    for(int plan=0;plan<2;++plan) {
        auto ctx=ggml_init({1<<20,nullptr,true});require(ctx,"metadata allocation");
        auto a=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,64,8),b=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,64,8);
        auto scale=ggml_scale(ctx,ggml_add(ctx,a,b),.25f);
        auto out=ggml_dsv4_act_qat(ctx,scale,GGML_DSV41_QAT_BF16,0);
        auto graph=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(graph,out);
        auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);require(buffer,"tensor allocation");
        std::vector<float>av(count),bv(count),actual(count);
        for(int step=0;step<9;++step) {
            float factor=step<3?.25f:float(step+1)/8.f;
            std::memcpy(scale->op_params,&factor,4);
            for(int i=0;i<count;++i){av[i]=float((i*13+step*7)%997-498)/64.f;bv[i]=float((i*19+step*11)%113-56)/32.f;}
            ggml_backend_tensor_set(a,av.data(),0,count*4);ggml_backend_tensor_set(b,bv.data(),0,count*4);
            require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"graph computation");
            ggml_backend_tensor_get(out,actual.data(),0,count*4);
            require(ggml_backend_cuda_dsv41_bf16_vector_queued()==++calls,"capture/replay dispatch ledger");
            for(int i=0;i<count;++i) {volatile float sum=av[i]+bv[i];float expected=ggml_dsv41_bf16(sum*factor);
                require(!std::memcmp(&expected,&actual[i],4),"full output bitwise oracle");}
            require(ggml_backend_cuda_graphs_count(backend)==1,"bounded executable count");
        }
        // Destroy all captured tensor metadata/buffers, then rebind a new plan.
        // Changed pointers must update before launch; stale graph inputs fail the oracle.
        ggml_backend_buffer_free(buffer);ggml_free(ctx);
    }
    {
        _putenv_s("LLAMA_DSV41_INDEX_FP4","1");
        auto ctx=ggml_init({1<<20,nullptr,true});require(ctx,"index metadata");
        constexpr int tokens=3,rows=17;
        auto q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,128,32,tokens);
        auto w=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,32,tokens);
        auto k=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,128,rows);
        auto v=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,tokens);
        auto mask=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,(rows+7)/8,tokens);
        auto out=ggml_dsv41_index_score(ctx,q,w,k,v,mask);
        require(ggml_dsv41_index_bf16_tag(out),"index BF16 tag");
        auto graph=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(graph,out);
        auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);require(buffer,"index buffers");
        std::vector<float>qv(128*32*tokens),kv(128*rows),wv(32*tokens,1.f/32.f),mv(((rows+7)/8)*tokens,0.f);
        int32_t vv[tokens]={7,13,17};mv[3]=-INFINITY;
        std::vector<float>reference(rows*tokens),actual(rows*tokens);
        auto queued=ggml_backend_cuda_dsv41_index_cublas_queued();
        for(int step=0;step<4;++step) {
            for(size_t i=0;i<qv.size();++i)qv[i]=step%2?(.3f+float(i%4)/8.f):float(1+i%4)/2.f;
            for(size_t i=0;i<kv.size();++i)kv[i]=float(1+(i+step)%4)/2.f;
            ggml_backend_tensor_set(q,qv.data(),0,qv.size()*4);ggml_backend_tensor_set(k,kv.data(),0,kv.size()*4);
            ggml_backend_tensor_set(w,wv.data(),0,wv.size()*4);ggml_backend_tensor_set(v,vv,0,sizeof(vv));ggml_backend_tensor_set(mask,mv.data(),0,mv.size()*4);
            ggml_backend_cuda_graphs_disable_push(backend);
            require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"eager index oracle");
            ggml_backend_cuda_graphs_disable_pop(backend);
            ggml_backend_tensor_get(out,reference.data(),0,reference.size()*4);
            require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"conditional index graph");
            ggml_backend_tensor_get(out,actual.data(),0,actual.size()*4);
            require(ggml_backend_cuda_dsv41_index_cublas_queued()==(queued+=2),"conditional index dispatch accounting");
            require(!std::memcmp(reference.data(),actual.data(),actual.size()*4),"both conditional branches match full eager output");
        }
        ggml_backend_buffer_free(buffer);ggml_free(ctx);
    }
    ggml_backend_free(backend);
    auto cpu=ggml_backend_cpu_init();require(cpu,"CPU backend");
    for(int threads:{1,7})for(int width:{31,5120}) {
        ggml_backend_cpu_set_n_threads(cpu,threads);
        constexpr int rows=17;
        std::vector<float>r(width*4*rows),p(4*rows),reference(width*rows),actual(width*rows);
        for(size_t i=0;i<r.size();++i)r[i]=float(int(i%127)-63)/32.f;
        for(size_t i=0;i<p.size();++i)p[i]=float(int(i%31)-15)/8.f;
        for(int fused=0;fused<2;++fused) {
            _putenv_s("LLAMA_DSV41_HC_COLLAPSE_FUSION",fused?"1":"0");
            auto ctx=ggml_init({1<<20,nullptr,true});require(ctx,"CPU metadata");
            auto residual=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,width,4,rows),pre=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4,rows);
            llama_dsv41_graph_ops ops(ctx,1e-20f,1e-6f,20);
            auto out=ops.collapse(residual,pre);
            auto graph=ggml_new_graph_custom(ctx,64,false);ggml_build_forward_expand(graph,out);
            auto buffer=ggml_backend_alloc_ctx_tensors(ctx,cpu);require(buffer,"CPU tensor allocation");
            ggml_backend_tensor_set(residual,r.data(),0,r.size()*4);ggml_backend_tensor_set(pre,p.data(),0,p.size()*4);
            require(ggml_backend_graph_compute(cpu,graph)==GGML_STATUS_SUCCESS,"CPU collapse graph");
            auto &values=fused?actual:reference;ggml_backend_tensor_get(out,values.data(),0,values.size()*4);
            ggml_backend_buffer_free(buffer);ggml_free(ctx);
        }
        require(!std::memcmp(reference.data(),actual.data(),actual.size()*4),"CPU custom callback matches original graph, all output values");
    }
    ggml_backend_free(cpu);
    std::puts("PASS 18 graph/counter checks, 4 FP4/BF16 conditional cases and 4 threaded CPU fallback graph comparisons");return 0;
}
