// CPU numerical fixture for all common routed formats. No model or GPU.
#include "llama-dsv41-quant-types.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void need(bool ok,const char*message){if(!ok)throw std::runtime_error(message);}
int main(int argc,char**argv){try{
    need(argc==2,"Expected scratch GGUF filename");
    for(auto type:{GGML_TYPE_Q4_K,GGML_TYPE_IQ2_XXS,GGML_TYPE_Q2_K,GGML_TYPE_MXFP4}){
        auto*ctx=ggml_init({16*1024*1024,nullptr,false});need(ctx,"CPU context");
        constexpr int k=256,n=8,experts=2,tokens=3;
        auto*w=ggml_new_tensor_3d(ctx,type,k,n,experts);ggml_set_name(w,"blk.0.ffn_gate_exps.weight");
        std::vector<float>weights(k*n*experts),importance(k,1.f),decoded(k);
        for(size_t i=0;i<weights.size();++i)weights[i]=0.15f*std::sin(float(i)*0.03f)+0.07f*std::cos(float(i)*0.07f);
        need(ggml_quantize_chunk(type,weights.data(),w->data,0,n*experts,k,importance.data())==ggml_nbytes(w),"Quantized bytes");
        // Exercise GGUF metadata + payload round trip on a tiny real file.
        auto*file=gguf_init_empty();gguf_set_val_str(file,"general.architecture","deepseek41");gguf_add_tensor(file,w);
        need(gguf_write_to_file(file,argv[1],false),"Tiny GGUF write");gguf_free(file);
        ggml_context*loaded=nullptr;file=gguf_init_from_file(argv[1],{false,&loaded});need(file&&loaded,"Tiny GGUF read");
        auto*copy=ggml_get_tensor(loaded,w->name);need(copy&&copy->type==type&&ggml_nbytes(copy)==ggml_nbytes(w)&&!std::memcmp(copy->data,w->data,ggml_nbytes(w)),"GGUF type/bytes round trip");
        auto*x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,k,1,tokens);auto*ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,2,tokens);
        for(int i=0;i<k*tokens;++i)static_cast<float*>(x->data)[i]=std::sin(float(i)*0.019f);
        for(int i=0;i<2*tokens;++i)static_cast<int32_t*>(ids->data)[i]=(i+i/2)%2;
        auto*out=ggml_mul_mat_id(ctx,copy,x,ids);auto*graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out);
        need(ggml_graph_compute_with_ctx(ctx,graph,2)==GGML_STATUS_SUCCESS,"CPU MMID dispatch");
        double squared_error=0,squared_reference=0;
        for(int token=0;token<tokens;++token)for(int slot=0;slot<2;++slot)for(int row=0;row<n;++row){
            const int expert=static_cast<int32_t*>(ids->data)[2*token+slot];
            const char*bytes=static_cast<const char*>(copy->data)+expert*copy->nb[2]+row*copy->nb[1];
            ggml_get_type_traits(type)->to_float(bytes,decoded.data(),k);
            double expected=0;for(int col=0;col<k;++col)expected+=decoded[col]*static_cast<float*>(x->data)[token*k+col];
            const float actual=static_cast<float*>(out->data)[token*2*n+slot*n+row];
            need(std::isfinite(actual),"Finite MMID output");squared_error+=(actual-expected)*(actual-expected);squared_reference+=expected*expected;
        }
        const double relative=std::sqrt(squared_error/squared_reference);need(relative<0.025,"MMID disagrees with independently dequantized dot products");
        std::printf("PASS %s GGUF roundtrip + CPU MMID relative RMSE %.8f\n",ggml_type_name(type),relative);
        gguf_free(file);ggml_free(loaded);ggml_free(ctx);
    }
    need(llama_dsv41_quant::triplet(GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XXS,GGML_TYPE_Q2_K),"IQ2/Q2 triplet");
    need(llama_dsv41_quant::triplet(GGML_TYPE_MXFP4,GGML_TYPE_MXFP4,GGML_TYPE_MXFP4),"MXFP4 triplet");
    need(!llama_dsv41_quant::triplet(GGML_TYPE_Q4_K,GGML_TYPE_MXFP4,GGML_TYPE_Q4_K),"Mixed gate/up rejected");
    try{llama_dsv41_quant::check("blk.7.ffn_up_exps.weight",GGML_TYPE_Q6_K,GGML_TYPE_Q4_K);need(false,"Unsupported type accepted");}
    catch(const std::runtime_error&e){need(std::strstr(e.what(),"blk.7.ffn_up_exps.weight")&&std::strstr(e.what(),"q6_K"),"Named type diagnostic");}
    std::puts("PASS tensor policy and unsupported tensor/type diagnostic");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
