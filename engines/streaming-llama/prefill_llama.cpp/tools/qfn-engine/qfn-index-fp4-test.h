// Independent CPU quantize/dequantize oracle for the explicit DSV FP4 candidate.
#pragma once
#include "../../ggml/src/ggml-dsv41-quant.h"
static void test_qfn_index_fp4(const backend_pair &bp,int T,ggml_type kt){
    constexpr int P=1025,S=2;
    auto *ctx=ggml_init({128*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true});
    auto *qp=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,128,5,T+1,S);
    auto *kp=ggml_new_tensor_4d(ctx,kt,136,1,P+1,S);
    auto *wp=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,8,T+2,1,S);
    auto *q=ggml_view_4d(ctx,qp,128,4,T,S,qp->nb[1],qp->nb[2],qp->nb[3],0);
    auto *k=ggml_view_4d(ctx,kp,128,1,P,S,kp->nb[1],kp->nb[2],kp->nb[3],0);
    auto *w=ggml_view_4d(ctx,wp,4,T,1,S,wp->nb[1],wp->nb[2],wp->nb[3],0);
    auto *out=ggml_qfn_qsa_score(ctx,q,k,w,0.08838834765f,0.5f);out->op_params[1]=2;
    auto *gf=ggml_new_graph(ctx);ggml_build_forward_expand(gf,out);
    auto buf=ggml_backend_alloc_ctx_tensors(ctx,bp.cuda);
    if(!buf){report("DSV FP4 indexer",false,"allocation failed");ggml_free(ctx);return;}
    std::mt19937 gen(719+T);std::uniform_real_distribution<float>rnd(-0.7f,0.7f);
    std::vector<float>qv(ggml_nelements(qp)),kv(ggml_nelements(kp)),wv(ggml_nelements(wp));
    for(auto &x:qv)x=rnd(gen);for(auto &x:kv)x=rnd(gen);for(auto &x:wv)x=rnd(gen);
    std::vector<ggml_fp16_t>kh(kv.size());std::vector<float>actual(ggml_nelements(out)),last;
    auto quant=[](const float *x,float *y){for(int g=0;g<4;++g){float a=0;
        for(int d=0;d<32;++d)a=std::max(a,std::fabs(x[g*32+d]));
        const float scale=ggml_dsv41_scale(a,2);
        for(int d=0;d<32;++d)y[g*32+d]=ggml_dsv41_e2m1(x[g*32+d]/scale)*scale;}};
    bool ok=true;double worst=0;
    for(int phase=0;phase<3;++phase){
        if(phase==2){last=actual;for(auto &x:qv)x=x*.5f+.125f;for(auto &x:kv)x=x*.75f-.0625f;}
        std::vector<float>kr=kv;
        if(kt==GGML_TYPE_F16){for(size_t i=0;i<kv.size();++i){kh[i]=ggml_fp32_to_fp16(kv[i]);kr[i]=ggml_fp16_to_fp32(kh[i]);}
            ggml_backend_tensor_set(kp,kh.data(),0,kh.size()*2);}
        else ggml_backend_tensor_set(kp,kv.data(),0,kv.size()*4);
        ggml_backend_tensor_set(qp,qv.data(),0,qv.size()*4);ggml_backend_tensor_set(wp,wv.data(),0,wv.size()*4);
        if(phase==1)ggml_backend_cuda_graphs_disable_pop(bp.cuda);
        ok=(phase?ggml_backend_cuda_graph_compute_static(bp.cuda,gf):ggml_backend_graph_compute(bp.cuda,gf))==GGML_STATUS_SUCCESS&&ok;
        ggml_backend_synchronize(bp.cuda);ggml_backend_tensor_get(out,actual.data(),0,actual.size()*4);
        for(int s=0;s<S;++s)for(int t=0;t<T;++t)for(int p=0;p<P;++p){
            float kk[128],qq[128],score=0;quant(kr.data()+s*(P+1)*136+p*136,kk);
            for(int h=0;h<4;++h){quant(qv.data()+s*(T+1)*5*128+t*5*128+h*128,qq);
                float dot=0;for(int d=0;d<128;++d)dot+=qq[d]*kk[d];
                score+=std::max(dot*0.08838834765f,0.f)*wv[s*(T+2)*8+t*8+h];}
            const float expected=score*.5f,a=actual[(s*T+t)*P+p];
            const double error=std::fabs(double(a)-expected);worst=std::max(worst,error);
            ok=std::isfinite(a)&&error<=2e-6*(1+std::fabs(expected))&&ok;
        }
        if(phase==2)ok=actual!=last&&ok;
    }
    ggml_backend_cuda_graphs_disable_push(bp.cuda);
    report("DSV FP4 indexer T="+std::to_string(T)+" type="+std::to_string(int(kt)),ok,
        "1025 pools, two padded streams, CPU quantized oracle, eager/capture/changed replay; maxabs="+std::to_string(worst));
    ggml_backend_buffer_free(buf);ggml_free(ctx);
}
