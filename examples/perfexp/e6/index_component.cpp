#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-dsv41-index-cublas.h"
#include "ggml-dsv41-quant.h"
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;
static void check(bool ok,const char*why){if(!ok)throw std::runtime_error(why);}
int main(){try{
    _putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");auto backend=ggml_backend_cuda_init(0);check(backend,"CUDA backend unavailable");
    std::mt19937 rng(719);std::uniform_real_distribution<float>random(-2.f,2.f);
    json results=json::array();bool passed=true;
    for(int fixture=0;fixture<4;++fixture){const int queries=fixture<2?3:128,rows=fixture<2?17:100001;
        auto*ctx=ggml_init({8*1024*1024,nullptr,true});check(ctx,"Metadata allocation failed");
        auto*q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,128,32,queries);
        auto*k=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,128,rows);
        auto*w=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,32,queries);
        auto*v=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,queries);
        auto*m=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,(rows+7)/8,queries);
        auto*out=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,rows,queries);ggml_set_output(out);
        out->op=GGML_OP_DSV41_INDEX_SCORE;out->src[0]=q;out->src[1]=w;out->src[2]=k;out->src[3]=v;out->src[4]=m;
        check(ggml_dsv41_index_bf16_tag(out),"Invalid synthetic operator");
        auto*graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out);
        auto*buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);check(buffer,"GPU allocation failed");
        std::vector<float>qhost(size_t(ggml_nelements(q))),khost(size_t(ggml_nelements(k)));
        for(auto*values:{&qhost,&khost}){for(auto&x:*values)x=random(rng);
            for(size_t i=0;i<values->size();i+=128)ggml_dsv41_quantize_row(values->data()+i,values->data()+i,128,2);}
        // Unrepresentable input must fall back, without changing the baseline.
        if(fixture%2)qhost[0]=.123456789f;
        ggml_backend_tensor_set(q,qhost.data(),0,qhost.size()*4);ggml_backend_tensor_set(k,khost.data(),0,khost.size()*4);
        std::vector<float>weights(size_t(ggml_nelements(w)));for(auto&x:weights)x=random(rng)/8;
        ggml_backend_tensor_set(w,weights.data(),0,weights.size()*4);
        std::vector<int32_t>visible(queries,rows);visible[0]=0;visible[1]=rows/2;
        ggml_backend_tensor_set(v,visible.data(),0,visible.size()*4);
        std::vector<float>mask(size_t(ggml_nelements(m)),0.f);for(size_t i=4;i<mask.size();i+=19)mask[i]=-INFINITY;
        ggml_backend_tensor_set(m,mask.data(),0,mask.size()*4);
        json row={{"queries",queries},{"keys",rows},{"fallback_expected",bool(fixture%2)}};
        std::vector<float>reference,actual;
        for(int candidate=0;candidate<2;++candidate){_putenv_s("LLAMA_DSV41_INDEX_FP4",candidate?"1":"0");
            auto compute=[&]{check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"Operator failed");ggml_backend_synchronize(backend);};
            compute();compute();std::vector<float>output(size_t(ggml_nelements(out)));ggml_backend_tensor_get(out,output.data(),0,output.size()*4);
            cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);std::vector<float>times;
            for(int repeat=0;repeat<3;++repeat){cudaEventRecord(a);compute();cudaEventRecord(b);cudaEventSynchronize(b);float ms=0;cudaEventElapsedTime(&ms,a,b);times.push_back(ms);}
            cudaEventDestroy(a);cudaEventDestroy(b);std::sort(times.begin(),times.end());row[candidate?"fp4_ms":"bf16_ms"]=times[1];
            (candidate?actual:reference)=std::move(output);
        }
        size_t mismatches=0,finite=0;double maximum=0;
        for(size_t i=0;i<actual.size();++i){if(std::memcmp(&actual[i],&reference[i],4))++mismatches;
            if(std::isfinite(reference[i])){check(std::isfinite(actual[i]),"Nonfinite unmasked output");++finite;maximum=std::max(maximum,std::abs(double(actual[i])-reference[i]));}
            else check(actual[i]==reference[i],"Masked score differs");}
        row.update({{"bit_exact",mismatches==0},{"mismatches",mismatches},{"finite_scores",finite},{"max_abs",maximum}});
        passed&=mismatches==0;results.push_back(row);std::cout<<row.dump()<<std::endl;
        ggml_backend_buffer_free(buffer);ggml_free(ctx);
    }
    ggml_backend_free(backend);std::cout<<json{{"component_only",true},{"passed",passed},{"results",results}}.dump()<<std::endl;return passed?0:2;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
