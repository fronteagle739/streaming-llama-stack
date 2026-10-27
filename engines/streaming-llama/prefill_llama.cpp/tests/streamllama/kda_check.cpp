// GLM KDA output and recurrent carry against an independent scalar recurrence.
// No model, tokenizer, scheduler callbacks or reduced-precision matmul is used.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GLM53_WITH_CUDA
#include "ggml-cuda.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static bool check(ggml_backend_t backend, int d, int heads, int tokens, int seqs, bool broadcast, int snapshots=1) {
    const int qheads = broadcast ? heads/2 : heads;
    const int qseqs = broadcast ? 1 : seqs;
    ggml_init_params params{2*1024*1024, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, qheads, tokens, qseqs);
    auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, qheads, tokens, qseqs);
    auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, tokens, seqs);
    auto * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, tokens, seqs);
    auto * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, heads, tokens, seqs);
    auto * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, d, heads, seqs);
    auto * out = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, snapshots);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); return false; }
    std::vector<float> qv(ggml_nelements(q)), kv(qv.size()), vv(ggml_nelements(v)), gv(vv.size());
    std::vector<float> bv(ggml_nelements(beta)), sv(ggml_nelements(state));
    for (size_t i=0; i<qv.size(); ++i) {
        qv[i] = std::sin(float(i)*.13f)/std::sqrt(float(d));
        kv[i] = std::cos(float(i)*.17f)/std::sqrt(float(d));
    }
    for (size_t i=0; i<vv.size(); ++i) {
        vv[i] = std::sin(float(i)*.11f);
        gv[i] = -.01f - .9f*std::abs(std::sin(float(i)*.07f));
    }
    for (size_t i=0; i<bv.size(); ++i) bv[i] = .1f+.7f*std::abs(std::cos(float(i)*.19f));
    for (size_t i=0; i<sv.size(); ++i) sv[i] = .1f*std::sin(float(i)*.23f); // nonzero, asymmetric carry
    ggml_backend_tensor_set(q,qv.data(),0,ggml_nbytes(q));
    ggml_backend_tensor_set(k,kv.data(),0,ggml_nbytes(k));
    ggml_backend_tensor_set(v,vv.data(),0,ggml_nbytes(v));
    ggml_backend_tensor_set(g,gv.data(),0,ggml_nbytes(g));
    ggml_backend_tensor_set(beta,bv.data(),0,ggml_nbytes(beta));
    ggml_backend_tensor_set(state,sv.data(),0,ggml_nbytes(state));
    std::vector<float> expected(ggml_nelements(out)), actual(expected.size());
    for (int s=0; s<seqs; ++s) for (int h=0; h<heads; ++h) {
        float * carry = sv.data()+(s*heads+h)*d*d;
        for (int t=0; t<tokens; ++t) {
            const int qi = (((broadcast ? 0 : s)*tokens+t)*qheads+h%qheads)*d;
            const int vi = ((s*tokens+t)*heads+h)*d;
            const float b = bv[(s*tokens+t)*heads+h];
            for (int col=0; col<d; ++col) {
                double projection=0;
                for (int row=0; row<d; ++row) {
                    carry[col*d+row] *= std::exp(gv[vi+row]);
                    projection += double(carry[col*d+row])*kv[qi+row];
                }
                const float delta = (vv[vi+col]-float(projection))*b;
                double output=0;
                for (int row=0; row<d; ++row) {
                    carry[col*d+row] += kv[qi+row]*delta;
                    output += double(carry[col*d+row])*qv[qi+row];
                }
                expected[vi+col] = float(output)/std::sqrt(float(d));
            }
            const int slot = tokens-1-t;
            if (snapshots>1 && slot<snapshots) {
                std::copy(carry,carry+d*d,expected.begin()+vv.size()+slot*sv.size()+(s*heads+h)*d*d);
            }
        }
    }
    if (snapshots==1) std::copy(sv.begin(),sv.end(),expected.begin()+vv.size());
    bool ok = ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS;
    if (ok) ggml_backend_tensor_get(out,actual.data(),0,ggml_nbytes(out));
    float output_error=0, state_error=0;
    for (size_t i=0; i<actual.size(); ++i) {
        ok = ok && std::isfinite(actual[i]);
        float & error = i<vv.size() ? output_error : state_error;
        error = std::max(error,std::abs(actual[i]-expected[i]));
    }
    ok = ok && output_error<1e-5f && state_error<1e-5f;
    std::printf("KDA backend=%s dim=%d heads=%d tokens=%d seqs=%d broadcast=%d snapshots=%d output_max_abs=%g state_max_abs=%g %s\n",
        ggml_backend_name(backend),d,heads,tokens,seqs,int(broadcast),snapshots,double(output_error),double(state_error),ok?"PASS":"FAIL");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main(int argc,char ** argv) {
    if (argc>2 || (argc==2 && std::strcmp(argv[1],"--cuda")!=0)) {
        std::fprintf(stderr,"usage: %s [--cuda]\n",argv[0]);
        return 2;
    }
    const bool cuda = argc>1 && std::strcmp(argv[1],"--cuda")==0;
    ggml_backend_t backend = nullptr;
#ifdef GLM53_WITH_CUDA
    if (cuda) backend=ggml_backend_cuda_init(0);
#endif
    if (!cuda) { backend=ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(backend,2); }
    if (!backend) { std::fprintf(stderr,"Requested backend unavailable\n"); return 2; }
    bool ok=true;
    for (int d : {16,128}) for (int t : {1,4,17,65}) {
        ok=check(backend,d,4,t,2,true)&&ok;
    }
    // Actual Flash KDA head shape; small token count keeps the scalar oracle bounded.
    ok=check(backend,128,64,4,1,false)&&ok;
    ok=check(backend,128,4,4,2,true,3)&&ok;
    ggml_backend_free(backend);
    return ok?0:1;
}
