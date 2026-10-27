// Scalar oracles for the GLM numerical bisect. No model or scheduler callback.
// Change inputs between repeated graph executions so CUDA replay cannot pass
// by returning a stale output. KDA has its own larger scalar matrix.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GLM53_WITH_CUDA
#include "ggml-cuda.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

struct test_graph {
    ggml_context * ctx;
    ggml_cgraph * graph;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_backend_t backend;
    explicit test_graph(ggml_backend_t b) : backend(b) {
        ctx = ggml_init({2*1024*1024, nullptr, true});
        graph = ggml_new_graph(ctx);
    }
    void alloc(ggml_tensor * out) {
        ggml_build_forward_expand(graph, out);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) std::abort();
    }
    bool compute() { return ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS; }
    ~test_graph() { ggml_backend_buffer_free(buffer); ggml_free(ctx); }
};

static float value(int i, int phase) {
    return .7f*std::sin(.131f*float(i+1)+.37f*phase) + .3f*std::cos(.071f*float(i+3));
}
static float input_max=0;
static size_t input_zero=0,input_inf=0;
static void put(ggml_tensor * t, const std::vector<float> & v) {
    for (float x:v) {
        if (std::isnan(x) || x==INFINITY || (std::isfinite(x)&&std::abs(x)>1e30f)) std::abort();
        if (std::isfinite(x)) input_max=std::max(input_max,std::abs(x));
        else ++input_inf; // only intentional negative-infinity masks
        input_zero+=x==0.f;
    }
    ggml_backend_tensor_set(t,v.data(),0,v.size()*sizeof(float));
}
static bool compare(ggml_tensor * out, const std::vector<float> & expected, float & error) {
    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(out,actual.data(),0,actual.size()*sizeof(float));
    for (size_t i=0; i<actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) return false;
        error=std::max(error,std::abs(actual[i]-expected[i]));
    }
    return true;
}
static bool report(ggml_backend_t b, const char * op, int tokens, int precision, float error, bool ok) {
    ok = ok && error < 1e-5f;
    std::printf("OP backend=%s op=%s tokens=%d precision=%s max_abs=%g %s\n",
        ggml_backend_name(b),op,tokens,precision ? "F32" : "default",double(error),ok?"PASS":"FAIL");
    std::fflush(stdout);
    return ok;
}

static bool matmul(ggml_backend_t backend, int tokens, bool routed, bool broadcast, int precision, int width=64) {
    constexpr int rows=32, used=2, experts=8;
    const int channels=routed ? (broadcast?1:used) : 1;
    test_graph g(backend);
    auto * weights=ggml_new_tensor_3d(g.ctx,GGML_TYPE_F32,width,rows,routed?experts:1);
    auto * x=routed ? ggml_new_tensor_3d(g.ctx,GGML_TYPE_F32,width,channels,tokens)
                     : ggml_new_tensor_2d(g.ctx,GGML_TYPE_F32,width,tokens);
    auto * ids=routed ? ggml_new_tensor_2d(g.ctx,GGML_TYPE_I32,used,tokens):nullptr;
    auto * out=routed ? ggml_mul_mat_id(g.ctx,weights,x,ids):ggml_mul_mat(g.ctx,weights,x);
    out->op_params[0]=precision ? GGML_PREC_F32 : GGML_PREC_DEFAULT;
    g.alloc(out);
    std::vector<float> w(ggml_nelements(weights)),v(ggml_nelements(x)),expected(ggml_nelements(out));
    std::vector<int32_t> routes(used*tokens);
    float error=0;bool ok=true;
    for(int phase=0;phase<4;++phase) {
        for(size_t i=0;i<w.size();++i)w[i]=value(int(i),phase);
        for(size_t i=0;i<v.size();++i)v[i]=value(int(i)+113,phase+2);
        put(weights,w);put(x,v);
        for(int t=0;t<tokens;++t)for(int s=0;s<used;++s)routes[t*used+s]=(3*t+2*s+phase)%experts;
        if(routed)ggml_backend_tensor_set(ids,routes.data(),0,routes.size()*sizeof(int32_t));
        for(int t=0;t<tokens;++t)for(int s=0;s<(routed?used:1);++s)for(int r=0;r<rows;++r) {
            double sum=0;
            const int expert=routed?routes[t*used+s]:0;
            for(int k=0;k<width;++k)sum+=double(w[(expert*rows+r)*width+k])*v[(t*channels+s%channels)*width+k];
            expected[(t*(routed?used:1)+s)*rows+r]=float(sum);
        }
        ok=g.compute() && compare(out,expected,error) && ok;
    }
    return report(backend,routed?(broadcast?"moe_broadcast":"moe_channels"):"matmul",tokens,precision,error,ok);
}

static bool mla(ggml_backend_t backend, int tokens, int precision) {
    constexpr int d=16, cells=24, heads=4;
    test_graph g(backend);
    auto * q=ggml_new_tensor_3d(g.ctx,GGML_TYPE_F32,d,tokens,heads);
    auto * k=ggml_new_tensor_2d(g.ctx,GGML_TYPE_F32,d,cells);
    auto * v=ggml_new_tensor_2d(g.ctx,GGML_TYPE_F32,cells,d);
    auto * mask=ggml_new_tensor_2d(g.ctx,GGML_TYPE_F32,cells,tokens);
    auto * scores=ggml_mul_mat(g.ctx,k,q);
    auto * probs=ggml_soft_max_ext(g.ctx,scores,mask,1.f/std::sqrt(float(d)),0.f);
    auto * out=ggml_mul_mat(g.ctx,v,probs);
    scores->op_params[0]=out->op_params[0]=precision?GGML_PREC_F32:GGML_PREC_DEFAULT;
    g.alloc(out);
    std::vector<float> qv(ggml_nelements(q)),kv(ggml_nelements(k)),vv(ggml_nelements(v));
    std::vector<float> mv(cells*tokens),expected(ggml_nelements(out));
    float error=0;bool ok=true;
    for(int phase=0;phase<4;++phase) {
        for(size_t i=0;i<qv.size();++i)qv[i]=value(int(i),phase);
        for(size_t i=0;i<kv.size();++i)kv[i]=value(int(i)+71,phase+1);
        for(size_t i=0;i<vv.size();++i)vv[i]=value(int(i)+43,phase+2);
        for(int t=0;t<tokens;++t)for(int c=0;c<cells;++c)mv[t*cells+c]=c<=std::min(cells-1,t+phase)?0.f:-INFINITY;
        put(q,qv);put(k,kv);put(v,vv);put(mask,mv);
        for(int h=0;h<heads;++h)for(int t=0;t<tokens;++t) {
            double logits[cells],maximum=-INFINITY,total=0;
            for(int c=0;c<cells;++c) {
                double dot=0;for(int i=0;i<d;++i)dot+=double(kv[c*d+i])*qv[(h*tokens+t)*d+i];
                logits[c]=dot/std::sqrt(double(d))+mv[t*cells+c];maximum=std::max(maximum,logits[c]);
            }
            for(auto & z:logits){z=std::exp(z-maximum);total+=z;}
            for(int i=0;i<d;++i){double sum=0;for(int c=0;c<cells;++c)sum+=vv[i*cells+c]*logits[c]/total;expected[(h*tokens+t)*d+i]=float(sum);}
        }
        ok=g.compute() && compare(out,expected,error) && ok;
    }
    return report(backend,"mla_mask_softmax_value",tokens,precision,error,ok);
}

static bool hc(ggml_backend_t backend, int tokens) {
    constexpr int width=17,streams=4,mixdim=24,iterations=20;constexpr float eps=1e-6f;
    test_graph g(backend);
    auto * x=ggml_new_tensor_3d(g.ctx,GGML_TYPE_F32,width,streams,tokens);
    auto * weights=ggml_new_tensor_2d(g.ctx,GGML_TYPE_F32,streams,tokens);
    auto * gate=ggml_new_tensor_3d(g.ctx,GGML_TYPE_F32,width,streams,tokens);
    auto * mixes=ggml_new_tensor_2d(g.ctx,GGML_TYPE_F32,mixdim,tokens);
    auto * scale=ggml_new_tensor_1d(g.ctx,GGML_TYPE_F32,3);
    auto * base=ggml_new_tensor_1d(g.ctx,GGML_TYPE_F32,mixdim);
    auto * pre=ggml_dsv4_hc_pre(g.ctx,x,weights);
    auto * gated=ggml_dsv4_hc_pre_gated(g.ctx,x,gate,.25f);
    auto * comb=ggml_dsv4_hc_comb(g.ctx,mixes,scale,base,eps,iterations);
    auto * post=ggml_dsv4_hc_post(g.ctx,pre,x,weights,comb);
    ggml_build_forward_expand(g.graph,gated);g.alloc(post);
    std::vector<float> xv(ggml_nelements(x)),wv(ggml_nelements(weights)),gv(xv.size()),mx(ggml_nelements(mixes)),bv(mixdim),sc{.7f,.9f,1.1f};
    std::vector<float> ep(width*tokens),eg(ep.size()),ec(16*tokens),eo(xv.size());
    bool ok=true;float error=0;
    for(int phase=0;phase<4;++phase) {
        for(size_t i=0;i<xv.size();++i){xv[i]=value(int(i),phase);gv[i]=value(int(i),phase+2);}
        for(size_t i=0;i<wv.size();++i)wv[i]=value(int(i)+13,phase);
        for(size_t i=0;i<mx.size();++i)mx[i]=value(int(i)+37,phase);
        for(int i=0;i<mixdim;++i)bv[i]=value(i+53,phase+1);
        put(x,xv);put(weights,wv);put(gate,gv);put(mixes,mx);put(scale,sc);put(base,bv);
        for(int t=0;t<tokens;++t) {
            double c[16];
            for(int s=0;s<streams;++s){
                double maximum=-INFINITY,sum=0;
                for(int a=0;a<streams;++a){int j=a+streams*s;c[j]=mx[t*mixdim+8+j]*sc[2]+bv[8+j];maximum=std::max(maximum,c[j]);}
                for(int a=0;a<streams;++a){int j=a+streams*s;c[j]=std::exp(c[j]-maximum);sum+=c[j];}
                for(int a=0;a<streams;++a)c[a+streams*s]=c[a+streams*s]/sum+eps;
            }
            for(int iter=0;iter<iterations;++iter){
                if(iter){for(int s=0;s<streams;++s){double sum=eps;for(int a=0;a<streams;++a)sum+=c[a+streams*s];for(int a=0;a<streams;++a)c[a+streams*s]/=sum;}}
                for(int a=0;a<streams;++a){double sum=eps;for(int s=0;s<streams;++s)sum+=c[a+streams*s];for(int s=0;s<streams;++s)c[a+streams*s]/=sum;}
            }
            for(int j=0;j<16;++j)ec[t*16+j]=float(c[j]);
            for(int i=0;i<width;++i){
                double sum=0,sg=0;for(int s=0;s<streams;++s){int j=(t*streams+s)*width+i;sum+=double(xv[j])*wv[t*streams+s];sg+=xv[j]/(1.+std::exp(-double(gv[j])));}
                ep[t*width+i]=float(sum);eg[t*width+i]=float(.25*sg);
                for(int a=0;a<streams;++a){double z=sum*wv[t*streams+a];for(int s=0;s<streams;++s)z+=xv[(t*streams+s)*width+i]*c[a+streams*s];eo[(t*streams+a)*width+i]=float(z);}
            }
        }
        ok=g.compute() && compare(pre,ep,error) && compare(gated,eg,error) && compare(comb,ec,error) && compare(post,eo,error) && ok;
    }
    return report(backend,"mhc_pre_gated_comb_post",tokens,0,error,ok);
}

static bool indexer(ggml_backend_t backend, int tokens) {
    // Pool scoring/top-k: signed head weights and masked future pools; assert
    // selected SETS because the top-k API intentionally leaves order unspecified.
    constexpr int heads=4,pools=12,selected=2;
    test_graph g(backend);
    auto * dots=ggml_new_tensor_3d(g.ctx,GGML_TYPE_F32,heads,tokens,pools);
    auto * weights=ggml_new_tensor_2d(g.ctx,GGML_TYPE_F32,heads,tokens);
    auto * mask=ggml_new_tensor_2d(g.ctx,GGML_TYPE_F32,pools,tokens);
    auto * sums=ggml_sum_rows(g.ctx,ggml_mul(g.ctx,ggml_relu(g.ctx,dots),weights));
    auto * score=ggml_add(g.ctx,ggml_cont(g.ctx,ggml_permute(g.ctx,sums,2,1,0,3)),mask);
    auto * top=ggml_top_k(g.ctx,score,selected);g.alloc(top);
    std::vector<float> dv(ggml_nelements(dots)),wv(ggml_nelements(weights)),mv(pools*tokens),expected(pools*tokens);
    std::vector<int32_t> ids(selected*tokens);float error=0;bool ok=true;
    for(int phase=0;phase<4;++phase){
        for(size_t i=0;i<dv.size();++i)dv[i]=phase==0?0.f:value(int(i),phase);
        for(size_t i=0;i<wv.size();++i)wv[i]=value(int(i)+17,phase+1);
        for(int t=0;t<tokens;++t)for(int p=0;p<pools;++p){
            bool visible=p<std::min(pools,selected+t/4);mv[t*pools+p]=visible?0.f:-INFINITY;
            double sum=0;for(int h=0;h<heads;++h)sum+=std::max(0.f,dv[(p*tokens+t)*heads+h])*double(wv[t*heads+h]);
            expected[t*pools+p]=float(sum)+mv[t*pools+p];
        }
        put(dots,dv);put(weights,wv);put(mask,mv);ok=g.compute()&&ok;
        std::vector<float> actual(expected.size());ggml_backend_tensor_get(score,actual.data(),0,actual.size()*4);
        ggml_backend_tensor_get(top,ids.data(),0,ids.size()*4);
        for(int t=0;t<tokens;++t){
            for(int p=0;p<pools;++p){int i=t*pools+p;if(std::isfinite(expected[i])){ok=std::isfinite(actual[i])&&ok;error=std::max(error,std::abs(actual[i]-expected[i]));}else ok=actual[i]==-INFINITY&&ok;}
            std::vector<int> order(pools);for(int p=0;p<pools;++p)order[p]=p;
            std::stable_sort(order.begin(),order.end(),[&](int a,int b){return expected[t*pools+a]>expected[t*pools+b];});
            // Exact zero ties must choose the same lowest-index pools on both
            // backends. The API leaves only the output order unspecified.
            int a=ids[t*selected],b=ids[t*selected+1];
            ok=a>=0&&a<pools&&b>=0&&b<pools&&a!=b&&ok;
            if(a>=0&&a<pools&&b>=0&&b<pools){ok=std::min(a,b)==std::min(order[0],order[1])&&std::max(a,b)==std::max(order[0],order[1])&&ok;}
        }
    }
    return report(backend,"indexer_score_topk_mask",tokens,0,error,ok);
}

static bool tied_pools(ggml_backend_t backend) {
    constexpr int pools=64,tokens=4;
    test_graph g(backend);
    auto * score=ggml_new_tensor_2d(g.ctx,GGML_TYPE_F32,pools,tokens);
    auto * order=ggml_argsort(g.ctx,score,GGML_SORT_ORDER_DESC);
    auto * top=ggml_top_k(g.ctx,score,2);
    ggml_build_forward_expand(g.graph,order);g.alloc(top);
    std::vector<float> values(pools*tokens);
    std::vector<int32_t> sorted(pools*tokens),selected(2*tokens);
    bool ok=true;
    for(int phase=0;phase<4;++phase) {
        for(int t=0;t<tokens;++t) {
            for(int p=0;p<pools;++p)values[t*pools+p]=p<21?-1.f-float(p)*.1f:-INFINITY;
            // The actual zero plateau at failing token 85 was {0,4,6}.
            values[t*pools+phase]=0.f;
            values[t*pools+4+phase]=-0.f;
            values[t*pools+6+phase]=0.f;
        }
        put(score,values);ok=g.compute()&&ok;
        ggml_backend_tensor_get(order,sorted.data(),0,sorted.size()*4);
        ggml_backend_tensor_get(top,selected.data(),0,selected.size()*4);
        for(int t=0;t<tokens;++t) {
            std::vector<int32_t> expected(pools);for(int p=0;p<pools;++p)expected[p]=p;
            std::stable_sort(expected.begin(),expected.end(),[&](int a,int b){return values[t*pools+a]>values[t*pools+b];});
            for(int p=0;p<pools;++p)ok=sorted[t*pools+p]==expected[p]&&ok;
            int a=selected[t*2],b=selected[t*2+1];
            ok=std::min(a,b)==phase&&std::max(a,b)==4+phase&&ok;
        }
    }
    return report(backend,"indexer_exact_zero_ties",tokens,0,0.f,ok);
}

int main(int argc,char ** argv){
    ggml_backend_t backend=nullptr;
    bool cuda=argc>1&&std::strcmp(argv[1],"--cuda")==0;
#ifdef GLM53_WITH_CUDA
    if(cuda)backend=ggml_backend_cuda_init(0);
#endif
    if(!cuda)backend=ggml_backend_cpu_init();
    if(!backend)return 2;
    std::printf("EXPERIMENT replay_input_phases=4 graphs_disabled_env=%s\n",
        std::getenv("GGML_CUDA_DISABLE_GRAPHS")?std::getenv("GGML_CUDA_DISABLE_GRAPHS"):"unset");
    bool ok=true;
    for(int precision:{0,1})for(int tokens:{1,4,16,17,64}){
        ok=matmul(backend,tokens,false,true,precision)&&ok;
        ok=matmul(backend,tokens,true,true,precision)&&ok;
        ok=matmul(backend,tokens,true,false,precision)&&ok;
        ok=mla(backend,tokens,precision)&&ok;
    }
    // Odd-width fallback must remain legal when CUDA graph capture is enabled.
    for(int precision:{0,1})ok=matmul(backend,1,true,true,precision,63)&&ok;
    for(int tokens:{1,17,65}){ok=hc(backend,tokens)&&ok;ok=indexer(backend,tokens)&&ok;}
    ok=tied_pools(backend)&&ok;
    std::printf("INPUT abs_max=%g nan=0 intentional_mask_inf=%zu above_1e30=0 zeros=%zu\n",double(input_max),input_inf,input_zero);
    ggml_backend_free(backend);
    return ok?0:1;
}
