#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "../engines/streaming-llama/prefill_llama.cpp/src/llama-dsv41-graph.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
static void require(bool value,const char *message){if(!value){std::fprintf(stderr,"FAIL %s\n",message);std::exit(2);}}
int main(){
    _putenv_s("GGML_CUDA_DISABLE_GRAPHS","0");
    auto backend=ggml_backend_cuda_init(0);require(backend,"CUDA backend");
    int checks=0;
    for(int width:{31,1024,5120})for(int rows:{1,17,4096,8192,16384}){
        std::vector<float> r(size_t(width)*4*rows),p(rows*4),w(width),reference(size_t(width)*rows),actual(reference.size());
        for(size_t j=0;j<r.size();++j)r[j]=float(int((j*12347+99)%65521)-32760)/8192.f;
        for(size_t j=0;j<p.size();++j)p[j]=float(int((j*13+7)%127)-63)/32.f;
        for(int j=0;j<width;++j)w[j]=float(j%31-15)/8.f;
        r[0]=-0.f;r[1]=std::ldexp(1.f,-140);r[2]=1.00390625f;r[3]=1.01171875f;
        // New metadata/buffers per arm prove pointer-independent replay, not
        // reuse of stale tensor bindings. Rows, epsilon and values also change.
        for(int mode=0;mode<3;++mode){
            _putenv_s("LLAMA_DSV41_LAYER_FUSION",mode==0?"0":mode==1?"1":"2");
            auto ctx=ggml_init({1<<20,nullptr,true});require(ctx,"metadata");
            auto residual=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,width,4,rows);
            auto pre=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4,rows),weight=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,width);
            llama_dsv41_graph_ops ops(ctx,rows==17?1e-6f:1e-20f,1e-6f,20);
            auto out=ops.norm(ops.collapse(residual,pre),weight);
            auto graph=ggml_new_graph_custom(ctx,64,false);ggml_build_forward_expand(graph,out);
            auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);require(buffer,"device buffers");
            ggml_backend_tensor_set(residual,r.data(),0,r.size()*4);ggml_backend_tensor_set(pre,p.data(),0,p.size()*4);
            ggml_backend_tensor_set(weight,w.data(),0,w.size()*4);
            if(mode<2)ggml_backend_cuda_graphs_disable_push(backend);
            require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"layer norm graph");
            if(mode<2)ggml_backend_cuda_graphs_disable_pop(backend);
            ggml_backend_tensor_get(out,actual.data(),0,actual.size()*4);
            if(mode==0)reference=actual;
            else {if(std::memcmp(reference.data(),actual.data(),actual.size()*4)){
                for(size_t j=0;j<actual.size();++j)if(std::memcmp(&reference[j],&actual[j],4)){
                    std::fprintf(stderr,"mismatch mode=%d width=%d rows=%d index=%zu reference=%.9g actual=%.9g\n",mode,width,rows,j,reference[j],actual[j]);break;}
                require(false,"all fused output bits match original CUDA reduction");}++checks;}
            ggml_backend_buffer_free(buffer);ggml_free(ctx);
        }
    }
    require(ggml_backend_cuda_graphs_count(backend)==2,"both block-size pointer-table executables counted");
    require(ggml_backend_cuda_graphs_clear(backend)==2,"pointer-table executable retirement counted");
    require(ggml_backend_cuda_graphs_count(backend)==0,"graph clear releases pointer tables and both graphs");
    ggml_backend_free(backend);
    std::printf("PASS %d full-output fused/changed-binding CUDA comparisons\n",checks);
}
