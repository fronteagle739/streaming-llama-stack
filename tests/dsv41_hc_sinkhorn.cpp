#include "ggml-dsv41-hc-sinkhorn.h"
#include <cstdio>
#include <stdexcept>
#include <initializer_list>
#include <string>
static void check_at(bool ok,int line){if(!ok)throw std::runtime_error("HC specialization contract mismatch at line"+std::to_string(line));}
#define check(ok) check_at(bool(ok),__LINE__)
int main(){try{
    for(int rows:{1,3,8,127,128,2541,4096,65536}){
        auto *ctx=ggml_init({65536,nullptr,true});check(ctx);
        auto *mix=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,24,rows);
        auto *scale=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,3);
        auto *base=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,24);
        auto *out=ggml_dup_tensor(ctx,mix);out->op=GGML_OP_MAP_CUSTOM3;
        out->src[0]=mix;out->src[1]=scale;out->src[2]=base;
        out->op_params[7]=GGML_DSV41_HC_SINKHORN_MAGIC;
        for(int mode:{0,1,2}){out->op_params[6]=mode;check(ggml_dsv41_hc_sinkhorn_valid(out));}
        out->op_params[6]=3;check(!ggml_dsv41_hc_sinkhorn_valid(out));out->op_params[6]=2;
        out->op_params[7]=0;check(!ggml_dsv41_hc_sinkhorn_valid(out));out->op_params[7]=GGML_DSV41_HC_SINKHORN_MAGIC;
        scale->ne[0]=4;check(!ggml_dsv41_hc_sinkhorn_valid(out));scale->ne[0]=3;
        mix->nb[0]+=4;check(!ggml_dsv41_hc_sinkhorn_valid(out));mix->nb[0]-=4;
        out->ne[1]=65537;mix->ne[1]=65537;check(!ggml_dsv41_hc_sinkhorn_valid(out));
        ggml_free(ctx);
    }
    std::puts("PASS HC specialization: dimensions, modes, strides, output shape and private dispatch tag");
}catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
