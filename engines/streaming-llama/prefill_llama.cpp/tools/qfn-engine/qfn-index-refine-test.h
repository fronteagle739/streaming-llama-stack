// 100K pool geometry, partial causal rows, ties and changed-input graph replay.
#pragma once
static void test_qfn_index_refine(const backend_pair &bp,int T,ggml_type kt){
    constexpr int P=25001;
    auto *ctx=ggml_init({256*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true});
    auto *q=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,128,4,T,1);
    auto *k=ggml_new_tensor_4d(ctx,kt,128,1,P,1);
    auto *w=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4,T);
    auto *pos=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,T);
    auto *approx=ggml_qfn_qsa_score(ctx,q,k,w,1.f,1.f);
    approx->op_params[1]=3;approx->src[3]=pos;approx->op_params[4]=512;approx->op_params[5]=4;
    auto *scalar=ggml_lightning_indexer(ctx,q,k,w,1.f,1.f);
    auto *a=ggml_qfn_qsa_select(ctx,ggml_reshape_2d(ctx,approx,P,T),pos,512,4);
    auto *b=ggml_qfn_qsa_select(ctx,ggml_reshape_2d(ctx,scalar,P,T),pos,512,4);
    auto *gf=ggml_new_graph(ctx);ggml_build_forward_expand(gf,a);ggml_build_forward_expand(gf,b);
    auto buf=ggml_backend_alloc_ctx_tensors(ctx,bp.cuda);
    if(!buf){report("TC refined indexer",false,"allocation failed");ggml_free(ctx);return;}
    char error[512]={};bool ok=ggml_backend_cuda_qfn_qsa_score_available(error,sizeof(error))&&ggml_backend_cuda_qfn_qsa_score_prepare(nullptr)==cudaSuccess;
    std::mt19937 gen(9182+T);std::uniform_real_distribution<float>dist(-1.f,1.f);
    auto qv=randv(128*4*T),kv=randv(128*P),wv=randv(4*T);
    std::vector<ggml_fp16_t>kh(kv.size());std::vector<int32_t>pv(T),av(ggml_nelements(a)),bv(av.size());
    for(int t=0;t<T;++t)pv[t]=T==1?100000:t==0?1535:t==1?4095:100000-(T-1-t)*97;
    for(int phase=0;phase<3;++phase){
        if(phase==2){for(auto &x:qv)x=dist(gen);for(auto &x:wv)x=dist(gen);}
        if(phase==0){for(auto &x:kv)x=dist(gen);for(int p=1;p<P;p+=7)std::copy(kv.begin(),kv.begin()+128,kv.begin()+p*128);}
        if(phase==2)for(int p=0;p<P;p+=3)std::fill(kv.begin()+p*128,kv.begin()+(p+1)*128,0.f);
        std::fill(kv.end()-128,kv.end(),std::numeric_limits<float>::quiet_NaN()); // invisible final pool must not affect bound
        ggml_backend_tensor_set(q,qv.data(),0,qv.size()*4);ggml_backend_tensor_set(w,wv.data(),0,wv.size()*4);
        ggml_backend_tensor_set(pos,pv.data(),0,pv.size()*4);
        if(kt==GGML_TYPE_F16){for(size_t i=0;i<kv.size();++i)kh[i]=ggml_fp32_to_fp16(kv[i]);ggml_backend_tensor_set(k,kh.data(),0,kh.size()*2);}
        else ggml_backend_tensor_set(k,kv.data(),0,kv.size()*4);
        if(phase==1)ggml_backend_cuda_graphs_disable_pop(bp.cuda);
        ok=(phase?ggml_backend_cuda_graph_compute_static(bp.cuda,gf):ggml_backend_graph_compute(bp.cuda,gf))==GGML_STATUS_SUCCESS&&ok;
        ggml_backend_synchronize(bp.cuda);ggml_backend_tensor_get(a,av.data(),0,av.size()*4);ggml_backend_tensor_get(b,bv.data(),0,bv.size()*4);
        size_t diff=0;for(size_t i=0;i<av.size();++i)diff+=av[i]!=bv[i];
        std::printf("TC_REFINE T=%d type=%d phase=%d selected_id_differences=%zu/%zu\n",T,int(kt),phase,diff,av.size());ok=diff==0&&ok;
    }
    ggml_backend_cuda_graphs_disable_push(bp.cuda);
    report("TC boundary refinement T="+std::to_string(T)+" type="+std::to_string(int(kt)),ok,"25001 pools, causal top512, duplicates, signed heads, zeros, eager/capture/changed replay");
    ggml_backend_buffer_free(buf);ggml_free(ctx);
}
