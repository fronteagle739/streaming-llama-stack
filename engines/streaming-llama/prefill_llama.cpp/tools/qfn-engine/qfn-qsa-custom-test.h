// Real GGML-to-pinned-DLL wiring checks. Kernel arithmetic has independent proofs.
#pragma once
#include "ggml-qfn-qsa-score.h"
#include "ggml-qfn-attention.h"
#include "../../../../qfn-basalt-nvfp4/qfn_basalt_qsa.h"
#include "../../../../qfn-basalt-nvfp4/qfn_basalt_attention.h"

static HMODULE qfn_test_adjacent_dll() {
    wchar_t path[32768];
    const auto n=GetModuleFileNameW(nullptr,path,32768);
    if (!n || n>=32768) return nullptr;
    wchar_t * slash=wcsrchr(path,L'\\');if (!slash) return nullptr;
    wcscpy_s(slash+1,32768-size_t(slash+1-path),L"qfn-basalt-nvfp4.dll");
    return LoadLibraryExW(path,nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
}

static void test_qsa_score_custom(const backend_pair & bp,int T,int ns,ggml_type key_type) {
    const std::string name="QSA score CUSTOM T="+std::to_string(T)+" streams="+std::to_string(ns)+
        (key_type==GGML_TYPE_F16?" F16":" F32");
    char error[512]={};
    auto dll=qfn_test_adjacent_dll();
    auto direct=dll?reinterpret_cast<decltype(&qfn_basalt_qsa_run)>(GetProcAddress(dll,"qfn_basalt_qsa_run")):nullptr;
    if (!direct || !ggml_backend_cuda_qfn_qsa_score_available(error,sizeof(error)) ||
        ggml_backend_cuda_qfn_qsa_score_prepare(nullptr)!=cudaSuccess) {
        report(name,false,std::string("independent score admission: ")+error);if(dll)FreeLibrary(dll);return;
    }
    constexpr int pools=257;
    auto * ctx=ggml_init({128*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true});
    // Padded parents make every supplied outer stride observable, including stream nb[3].
    auto * qp=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,128,5,T+1,ns);
    auto * kp=ggml_new_tensor_4d(ctx,key_type,136,1,pools+1,ns);
    auto * wp=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,8,T+2,1,ns);
    auto * q=ggml_view_4d(ctx,qp,128,4,T,ns,qp->nb[1],qp->nb[2],qp->nb[3],0);
    auto * k=ggml_view_4d(ctx,kp,128,1,pools,ns,kp->nb[1],kp->nb[2],kp->nb[3],0);
    auto * w=ggml_view_4d(ctx,wp,4,T,1,ns,wp->nb[1],wp->nb[2],wp->nb[3],0);
    auto * out=ggml_qfn_qsa_score(ctx,q,k,w,0.08838834765f,0.5f);
    auto * reference=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,pools,T,1,ns);
    auto * gf=ggml_new_graph(ctx);ggml_build_forward_expand(gf,out);
    auto buf=ggml_backend_alloc_ctx_tensors(ctx,bp.cuda);
    if (!buf) {report(name,false,"allocation failed");ggml_free(ctx);FreeLibrary(dll);return;}
    std::mt19937 local_rng(7981+T+ns);std::uniform_real_distribution<float> dist(-0.7f,0.7f);
    std::vector<float> qv(ggml_nelements(qp)),kv(ggml_nelements(kp)),wv(ggml_nelements(wp));
    for(auto & x:qv)x=dist(local_rng);for(auto & x:kv)x=dist(local_rng);for(auto & x:wv)x=dist(local_rng);
    std::vector<ggml_fp16_t> kh(kv.size());
    auto upload=[&] {
        ggml_backend_tensor_set(qp,qv.data(),0,qv.size()*4);
        ggml_backend_tensor_set(wp,wv.data(),0,wv.size()*4);
        if(key_type==GGML_TYPE_F16){for(size_t i=0;i<kv.size();++i)kh[i]=ggml_fp32_to_fp16(kv[i]);
            ggml_backend_tensor_set(kp,kh.data(),0,kh.size()*2);}
        else ggml_backend_tensor_set(kp,kv.data(),0,kv.size()*4);
        ggml_backend_synchronize(bp.cuda);
    };
    std::vector<float> expected(ggml_nelements(out)),actual(expected.size()),previous;
    bool ok=true;
    const auto queued=ggml_backend_cuda_qfn_qsa_score_queued();
    auto check=[&](const char * phase) {
        ggml_backend_synchronize(bp.cuda);ggml_backend_tensor_get(out,actual.data(),0,actual.size()*4);
        const bool pass=finite(actual)&&actual==expected;
        std::printf("QSA_SCORE_CUSTOM T=%d streams=%d type=%d phase=%s maxabs=%.9g bit_equal=%d\n",
            T,ns,int(key_type),phase,maxabs(actual,expected),int(pass));return pass;
    };
    const void * stable_q=q->data,* stable_k=k->data,* stable_w=w->data;
    for(int phase=0;phase<2;++phase) {
        if(phase){previous=expected;for(auto & x:qv)x=x*0.875f+0.03125f;
            for(auto & x:wv)x=x*0.75f-0.0625f;for(auto & x:kv)x=x*0.5f+0.125f;}
        upload();
        const int status=direct((const float *)q->data,k->data,(const float *)w->data,(float *)reference->data,
            T,pools,ns,key_type==GGML_TYPE_F16?0:1,q->nb[1],q->nb[2],q->nb[3],k->nb[2],k->nb[3],
            w->nb[1],w->nb[3],reference->nb[1],reference->nb[3],0.08838834765f,0.5f,nullptr,0);
        ok=status==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess && ok;
        ggml_backend_tensor_get(reference,expected.data(),0,expected.size()*4);
        ok=finite(expected) && (!phase || previous!=expected) && ok;
        ok=q->data==stable_q && k->data==stable_k && w->data==stable_w && ok;
        if(!phase){ok=ggml_backend_graph_compute(bp.cuda,gf)==GGML_STATUS_SUCCESS&&ok;ok=check("eager")&&ok;
            ggml_backend_cuda_graphs_disable_pop(bp.cuda);}
        ok=ggml_backend_cuda_graph_compute_static(bp.cuda,gf)==GGML_STATUS_SUCCESS&&ok;
        ok=check(phase?"changed-replay":"capture")&&ok;
    }
    ggml_backend_cuda_graphs_disable_push(bp.cuda);
    ok=ggml_backend_cuda_qfn_qsa_score_queued()>=queued+3 && ok;
    report(name,ok,"padded q/k/w stream strides; direct-DLL all-output bit equality; changed inputs; required graph replay; dispatch witness");
    ggml_backend_buffer_free(buf);ggml_free(ctx);FreeLibrary(dll);
}

static void test_qsa_attention_custom(const backend_pair & bp,int T,bool gated) {
    const std::string name="QSA attention Basalt T="+std::to_string(T)+(gated?" gated":" no-gate");
    char error[512]={};auto dll=qfn_test_adjacent_dll();
    auto direct=dll?reinterpret_cast<decltype(&qfn_basalt_attn_run)>(GetProcAddress(dll,"qfn_basalt_attn_run")):nullptr;
    if(!direct || !ggml_backend_cuda_qfn_attention_available(error,sizeof(error)) ||
        ggml_backend_cuda_qfn_attention_prepare(nullptr)!=cudaSuccess) {
        report(name,false,std::string("attention admission: ")+error);if(dll)FreeLibrary(dll);return;
    }
    constexpr int D=256,H=24,Hkv=2,N=257,cap=67;
    auto * ctx=ggml_init({128*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true});
    auto * qp=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,H+1,T+1);
    auto * gp=gated?ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,H+2,T+2):nullptr;
    auto * kp=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,D+8,Hkv+1,N+1);
    auto * vp=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,D+16,Hkv+2,N+2);
    auto * ip=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,cap+3,T);
    auto * q=ggml_view_3d(ctx,qp,D,H,T,qp->nb[1],qp->nb[2],0);
    auto * gate=gp?ggml_view_3d(ctx,gp,D,H,T,gp->nb[1],gp->nb[2],0):nullptr;
    auto * k=ggml_view_3d(ctx,kp,D,Hkv,N,kp->nb[1],kp->nb[2],0);
    auto * v=ggml_view_3d(ctx,vp,D,Hkv,N,vp->nb[1],vp->nb[2],0);
    auto * ids=ggml_view_2d(ctx,ip,cap,T,ip->nb[1],0);
    auto * pos=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,T);
    const uint64_t bytes=ggml_backend_cuda_qfn_attention_workspace_bytes(T,H,Hkv,N,cap,1.0f/16.0f);
    if(bytes==UINT64_MAX){report(name,false,"scratch query invalid");ggml_free(ctx);FreeLibrary(dll);return;}
    auto * scratch=T<=4?ggml_new_tensor_1d(ctx,GGML_TYPE_I8,int64_t(std::max(uint64_t(1),bytes))):nullptr;
    auto * out=ggml_qfn_attention(ctx,q,gate,k,v,ids,pos,1.0f/16.0f,scratch);
    auto * reference=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,H,T);
    auto * gf=ggml_new_graph(ctx);ggml_build_forward_expand(gf,out);
    auto buf=ggml_backend_alloc_ctx_tensors(ctx,bp.cuda);
    if(!buf){report(name,false,"allocation failed");ggml_free(ctx);FreeLibrary(dll);return;}
    std::mt19937 local_rng(1543+T);std::uniform_real_distribution<float> dist(-0.8f,0.8f);
    std::vector<float> qv(ggml_nelements(qp)),gv(gp?ggml_nelements(gp):0),kf(ggml_nelements(kp)),vf(ggml_nelements(vp));
    for(auto & x:qv)x=dist(local_rng);for(auto & x:gv)x=dist(local_rng);
    for(auto & x:kf)x=dist(local_rng);for(auto & x:vf)x=dist(local_rng);
    std::vector<ggml_fp16_t> kh(kf.size()),vh(vf.size());
    for(size_t i=0;i<kf.size();++i)kh[i]=ggml_fp32_to_fp16(kf[i]);
    for(size_t i=0;i<vf.size();++i)vh[i]=ggml_fp32_to_fp16(vf[i]);
    std::vector<int32_t> iv(ggml_nelements(ip),-1),pv(T);
    for(int t=0;t<T;++t){pv[t]=180+t;for(int j=0;j<cap;++j)
        iv[size_t(t)*(cap+3)+j]=j%11==2?-1:(j%13==4?N+3:(j*7)%N);}
    ggml_backend_tensor_set(kp,kh.data(),0,kh.size()*2);ggml_backend_tensor_set(vp,vh.data(),0,vh.size()*2);
    ggml_backend_tensor_set(ip,iv.data(),0,iv.size()*4);ggml_backend_tensor_set(pos,pv.data(),0,pv.size()*4);
    qfn_basalt_attn_desc d{};d.mode=T<=4?QFN_ATTN_DECODE64:QFN_ATTN_PROMPT;
    d.nt=T;d.n_heads=H;d.n_kv_heads=Hkv;d.n_kv=N;d.cap=cap;d.scale=1.0f/16.0f;
    d.q_head_stride=q->nb[1];d.q_token_stride=q->nb[2];
    d.gate_head_stride=gate?gate->nb[1]:0;d.gate_token_stride=gate?gate->nb[2]:0;
    d.k_head_stride=k->nb[1];d.k_cell_stride=k->nb[2];d.v_head_stride=v->nb[1];d.v_cell_stride=v->nb[2];
    d.ids_token_stride=ids->nb[1];d.out_head_stride=reference->nb[1];d.out_token_stride=reference->nb[2];
    std::vector<float> expected(ggml_nelements(out)),actual(expected.size()),previous;
    bool ok=true;const void * stable_q=q->data;
    auto check=[&](const char * phase){ggml_backend_synchronize(bp.cuda);ggml_backend_tensor_get(out,actual.data(),0,actual.size()*4);
        const bool pass=finite(actual)&&actual==expected;
        std::printf("QSA_ATTN_CUSTOM T=%d gated=%d phase=%s maxabs=%.9g bit_equal=%d\n",T,int(gated),phase,maxabs(actual,expected),int(pass));return pass;};
    for(int phase=0;phase<2;++phase){
        if(phase){previous=expected;for(auto & x:qv)x=x*0.75f+0.125f;for(auto & x:gv)x=x*0.5f-0.25f;}
        ggml_backend_tensor_set(qp,qv.data(),0,qv.size()*4);if(gp)ggml_backend_tensor_set(gp,gv.data(),0,gv.size()*4);
        ggml_backend_synchronize(bp.cuda);
        const int status=direct((const float *)q->data,gate?(const float *)gate->data:nullptr,k->data,v->data,
            (const int32_t *)ids->data,(const int32_t *)pos->data,(float *)reference->data,&d,
            scratch?scratch->data:nullptr,bytes,nullptr);
        ok=status==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess && ok;
        ggml_backend_tensor_get(reference,expected.data(),0,expected.size()*4);
        ok=finite(expected) && (!phase || previous!=expected) && q->data==stable_q && ok;
        if(!phase){ok=ggml_backend_graph_compute(bp.cuda,gf)==GGML_STATUS_SUCCESS&&ok;ok=check("eager")&&ok;
            ggml_backend_cuda_graphs_disable_pop(bp.cuda);}
        ok=ggml_backend_cuda_graph_compute_static(bp.cuda,gf)==GGML_STATUS_SUCCESS&&ok;
        ok=check(phase?"changed-replay":"capture")&&ok;
    }
    ggml_backend_cuda_graphs_disable_push(bp.cuda);
    report(name,ok,"padded independent KV/gate/selection strides; invalid-ID holes; complete direct-DLL bit equality; changed-input static replay");
    ggml_backend_buffer_free(buf);ggml_free(ctx);FreeLibrary(dll);
}
