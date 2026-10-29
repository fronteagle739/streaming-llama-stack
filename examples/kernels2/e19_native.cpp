// Component-only bridge: preload identical CUDA inputs before timing. No model I/O.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include "ggml-dsv41-index-cublas.h"
#include "ggml-dsv41-shared-sparse.h"
#include "ggml-dsv41-hc-sinkhorn.h"
#include "llama-dsv41-graph.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define API extern "C" __declspec(dllexport)
static thread_local std::string error;
static void check(bool condition,const char *message){if(!condition)throw std::runtime_error(message);}
static void cuda_check(cudaError_t code){check(code==cudaSuccess,cudaGetErrorString(code));}
struct plan {
    ggml_backend_t backend=nullptr;
    ggml_context *ctx=nullptr;
    ggml_gallocr_t allocator=nullptr;
    ggml_cgraph *graph=nullptr;
    std::vector<ggml_backend_buffer_t> borrowed;
    std::vector<ggml_tensor*> outputs;
    ~plan(){
        if(backend)ggml_backend_synchronize(backend);
        if(allocator)ggml_gallocr_free(allocator);
        for(auto *buffer:borrowed)ggml_backend_buffer_free(buffer);
        if(ctx)ggml_free(ctx);
        if(backend)ggml_backend_free(backend);
    }
    ggml_tensor *input(ggml_type type,std::initializer_list<int64_t> shape,void *pointer){
        cudaPointerAttributes attributes{};cuda_check(cudaPointerGetAttributes(&attributes,pointer));
        check(attributes.type==cudaMemoryTypeDevice&&attributes.device==0,"Input must be on selected CUDA device");
        std::vector<int64_t> dimensions(shape);
        auto *tensor=ggml_new_tensor(ctx,type,int(dimensions.size()),dimensions.data());
        auto buft=ggml_backend_cuda_buffer_type(0);
        auto *buffer=ggml_backend_buft_alloc_buffer(buft,ggml_backend_buft_get_alloc_size(buft,tensor));
        check(buffer,"Native input allocation refused");borrowed.push_back(buffer);
        check(ggml_backend_tensor_alloc(buffer,tensor,ggml_backend_buffer_get_base(buffer))==GGML_STATUS_SUCCESS,"Native input initialization failed");
        cuda_check(cudaMemcpy(tensor->data,pointer,ggml_nbytes(tensor),cudaMemcpyDeviceToDevice));
        ggml_set_input(tensor);return tensor;
    }
    void finish(){
        graph=ggml_new_graph_custom(ctx,2048,false);
        for(auto *tensor:outputs){ggml_set_output(tensor);ggml_build_forward_expand(graph,tensor);}
        allocator=ggml_gallocr_new(ggml_backend_cuda_buffer_type(0));
        check(ggml_gallocr_alloc_graph(allocator,graph),"Native component allocation refused");
    }
    void run(){check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"Native component failed");}
};
API const char *e19_error(){return error.c_str();}
API void *e19_create(int kind,int rows,int keys,int start,void **inputs){try{
    check(rows>0&&rows<=65536&&keys>=0&&keys<=100001&&start>=0&&start<=100000,"Invalid component dimensions");
    auto p=std::make_unique<plan>();
    p->backend=ggml_backend_cuda_init(0);check(p->backend,"CUDA backend unavailable");
    ggml_backend_cuda_graphs_disable_push(p->backend);
    p->ctx=ggml_init({16u*1024*1024,nullptr,true});check(p->ctx,"Metadata allocation failed");
    auto *ctx=p->ctx;llama_dsv41_graph_ops ops(ctx,1e-20f,1e-6f,20);
    if(kind==0){
        p->outputs={ops.norm(p->input(GGML_TYPE_F32,{5120,rows},inputs[0]),p->input(GGML_TYPE_F32,{5120},inputs[1]))};
    }else if(kind==1){
        auto result=ops.mixes(p->input(GGML_TYPE_F32,{5120,4,rows},inputs[0]),
            p->input(GGML_TYPE_F32,{20480,24},inputs[1]),p->input(GGML_TYPE_F32,{3},inputs[2]),p->input(GGML_TYPE_F32,{24},inputs[3]));
        p->outputs={result.pre,result.post,result.comb};
    }else if(kind==5||kind==6||kind==7){
        check(kind!=5||rows<=4096,"Native Sinkhorn requires microtiles <=4096 rows");
        auto *projected=p->input(GGML_TYPE_F32,{24,rows},inputs[0]);
        auto *scale=p->input(GGML_TYPE_F32,{3},inputs[1]);
        auto *base=p->input(GGML_TYPE_F32,{24},inputs[2]);
        if(kind==6||kind==7){
            auto cpu=[](ggml_tensor*,const ggml_tensor*,const ggml_tensor*,const ggml_tensor*,int,int,void*){GGML_ABORT("GPU-only component");};
            auto *packed=ggml_map_custom3(ctx,projected,scale,base,cpu,1,nullptr);
            packed->op_params[7]=GGML_DSV41_HC_SINKHORN_MAGIC;
            packed->op_params[6]=kind==7?2:1;
            p->outputs={ggml_view_2d(ctx,packed,4,rows,16,0),ggml_view_2d(ctx,packed,4,rows,16,rows*16),
                ggml_view_3d(ctx,packed,4,4,rows,16,64,rows*32)};
            p->finish();return p.release();
        }
        auto affine=[&](int offset,int length,int si){
            auto *part=ggml_view_2d(ctx,projected,length,rows,projected->nb[1],offset*4);
            return ggml_add(ctx,ggml_mul(ctx,part,ggml_view_1d(ctx,scale,1,si*4)),ggml_view_1d(ctx,base,length,offset*4));
        };
        auto *eps=ggml_fill(ctx,ggml_new_tensor_1d(ctx,GGML_TYPE_F32,1),1e-6f);
        auto *pre=ggml_add(ctx,ggml_sigmoid(ctx,affine(0,4,0)),eps);
        auto *post=ggml_scale(ctx,ggml_sigmoid(ctx,affine(4,4,1)),2.f);
        auto *comb=ggml_add(ctx,ggml_soft_max(ctx,ggml_reshape_3d(ctx,affine(8,16,2),4,4,rows)),eps);
        auto columns=[&]{auto *t=ggml_cont(ctx,ggml_permute(ctx,comb,1,0,2,3));
            comb=ggml_div(ctx,comb,ggml_permute(ctx,ggml_add(ctx,ggml_sum_rows(ctx,t),eps),1,0,2,3));};
        columns();for(int i=1;i<20;++i){comb=ggml_div(ctx,comb,ggml_add(ctx,ggml_sum_rows(ctx,comb),eps));columns();}
        p->outputs={pre,post,comb};
    }else if(kind==2){
        check(keys>0&&keys<=32768,"Invalid dense output width");
        auto *weight=p->input(GGML_TYPE_BF16,{5120,keys},inputs[1]);
        auto *product=ggml_mul_mat(ctx,weight,p->input(GGML_TYPE_F32,{5120,rows},inputs[0]));
        ggml_mul_mat_set_prec(product,GGML_PREC_F32);p->outputs={ops.bf16(product)};
    }else if(kind==3){
        check(rows<=4096&&keys>0,"Native indexer requires <=4096 query rows");
        auto *out=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,keys,rows);
        out->op=GGML_OP_DSV41_INDEX_SCORE;
        out->src[0]=p->input(GGML_TYPE_F32,{128,32,rows},inputs[0]);
        out->src[1]=p->input(GGML_TYPE_F32,{32,rows},inputs[1]);
        out->src[2]=p->input(GGML_TYPE_F32,{128,keys},inputs[2]);
        out->src[3]=p->input(GGML_TYPE_I32,{rows},inputs[3]);
        check(ggml_dsv41_index_bf16_tag(out),"Native index contract failed");
        p->outputs={out};
    }else if(kind==4){
        check(rows<=4096&&keys>0,"Native attention requires <=4096 query rows");
        auto *out=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,512,64,rows);
        out->op=GGML_OP_FLASH_ATTN_EXT_SPARSE;
        out->op_params[2]=512;out->op_params[3]=start;out->op_params[4]=GGML_DSV41_TOKEN_TILE_MAGIC;
        out->src[0]=p->input(GGML_TYPE_F32,{512,64,rows},inputs[0]);
        out->src[1]=p->input(GGML_TYPE_F32,{512,rows+std::min(start,127)},inputs[1]);
        out->src[2]=p->input(GGML_TYPE_F32,{512,keys},inputs[2]);
        out->src[3]=p->input(GGML_TYPE_I32,{2,2048,(rows+3)/4},inputs[3]);
        out->src[4]=p->input(GGML_TYPE_I32,{(rows+3)/4},inputs[4]);
        out->src[5]=p->input(GGML_TYPE_F32,{64},inputs[5]);
        check(ggml_dsv41_token_tile_valid(out),"Native attention contract failed");p->outputs={out};
    }else throw std::runtime_error("Unknown component");
    p->finish();return p.release();
}catch(const std::exception &e){error=e.what();return nullptr;}}
API int e19_run(void *handle,int iterations,float *ms){try{
    auto &p=*static_cast<plan*>(handle);check(iterations>0&&iterations<=100,"Invalid iteration count");
    ggml_backend_synchronize(p.backend);
    const auto stream=static_cast<cudaStream_t>(ggml_backend_cuda_get_stream(p.backend));
    cudaEvent_t first,last;cuda_check(cudaEventCreate(&first));cuda_check(cudaEventCreate(&last));
    cuda_check(cudaEventRecord(first,stream));
    for(int i=0;i<iterations;++i)p.run();
    cuda_check(cudaEventRecord(last,stream));cuda_check(cudaEventSynchronize(last));
    cuda_check(cudaEventElapsedTime(ms,first,last));*ms/=iterations;
    cuda_check(cudaEventDestroy(first));cuda_check(cudaEventDestroy(last));return 0;
}catch(const std::exception &e){error=e.what();return 1;}}
API int e19_copy(void *handle,int index,void *destination,size_t bytes){try{
    auto &p=*static_cast<plan*>(handle);check(index>=0&&size_t(index)<p.outputs.size(),"Invalid output index");
    auto *tensor=p.outputs[index];check(bytes==ggml_nbytes(tensor),"Output geometry mismatch");
    ggml_backend_synchronize(p.backend);cuda_check(cudaMemcpy(destination,tensor->data,bytes,cudaMemcpyDeviceToDevice));return 0;
}catch(const std::exception &e){error=e.what();return 1;}}
API void e19_destroy(void *handle){delete static_cast<plan*>(handle);}
