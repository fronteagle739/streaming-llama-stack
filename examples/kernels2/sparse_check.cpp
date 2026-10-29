#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-dsv41-shared-sparse.h"
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;
static void check(bool ok,const char*why){if(!ok)throw std::runtime_error(why);}
int main(){try{
    _putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");auto backend=ggml_backend_cuda_init(0);check(backend,"CUDA backend unavailable");
    std::mt19937 rng(17);std::uniform_real_distribution<float> random(-.5f,.5f);
    bool passed=true;json results=json::array();
    for(int queries:{3,128}){
        const int start=queries==3?0:100000,comp_rows=queries==3?1:100001,top=queries==3?0:512,tiles=(queries+3)/4;
        auto*ctx=ggml_init({16*1024*1024,nullptr,true});check(ctx,"Metadata allocation failed");
        auto*q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,512,64,queries);
        auto*raw=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,512,queries+std::min(start,127));
        auto*comp=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,512,comp_rows);
        auto*records=ggml_new_tensor_3d(ctx,GGML_TYPE_I32,2,top?4*top:1,tiles);
        auto*counts=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,tiles);auto*sinks=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,64);
        auto*out=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,512,64,queries);ggml_set_output(out);
        out->op=GGML_OP_FLASH_ATTN_EXT_SPARSE;out->op_params[2]=top;out->op_params[3]=start;out->op_params[4]=GGML_DSV41_TOKEN_TILE_MAGIC;
        out->src[0]=q;out->src[1]=raw;out->src[2]=comp;out->src[3]=records;out->src[4]=counts;out->src[5]=sinks;
        check(ggml_dsv41_token_tile_valid(out),"Synthetic native operator contract invalid");
        auto*graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out);
        auto*buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);check(buffer,"GPU allocation failed");
        for(auto*t:{q,raw,comp,sinks}){std::vector<float>values(size_t(ggml_nelements(t)));
            for(auto&v:values){v=random(rng);if(t==raw||t==comp)v=std::round(v*32)/32;}
            if(t==comp)std::fill(values.begin(),values.begin()+512,0.f);
            ggml_backend_tensor_set(t,values.data(),0,values.size()*4);}
        std::vector<int32_t>record_data(size_t(ggml_nelements(records)),0),count_data(tiles,4*top);
        if(top)for(int t=0;t<queries;++t)for(int j=0;j<top;++j){const size_t at=size_t(t)*top*2+j*2;
            record_data[at]=1+(t*997+j*13)%(comp_rows-1);record_data[at+1]=(t==0&&j%9==0)?0:1<<(t%4);}
        ggml_backend_tensor_set(records,record_data.data(),0,record_data.size()*4);ggml_backend_tensor_set(counts,count_data.data(),0,count_data.size()*4);
        std::vector<float>reference,actual;json row={{"queries",queries},{"compressed_rows",comp_rows},{"raw_start",start}};
        for(int candidate=0;candidate<2;++candidate){_putenv_s("LLAMA_DSV41_TILELANG_ATTENTION",candidate?"2":"0");
            auto compute=[&]{check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"Operator failed");ggml_backend_synchronize(backend);};
            compute();compute();std::vector<float>output(size_t(ggml_nelements(out)));ggml_backend_tensor_get(out,output.data(),0,output.size()*4);
            cudaEvent_t a,b;check(cudaEventCreate(&a)==cudaSuccess&&cudaEventCreate(&b)==cudaSuccess,"Timing events failed");
            std::vector<float>times;for(int repeat=0;repeat<3;++repeat){cudaEventRecord(a);compute();cudaEventRecord(b);cudaEventSynchronize(b);float ms=0;cudaEventElapsedTime(&ms,a,b);times.push_back(ms);}
            cudaEventDestroy(a);cudaEventDestroy(b);std::sort(times.begin(),times.end());row[candidate?"tilelang_ms":"native_ms"]=times[1];
            (candidate?actual:reference)=std::move(output);
        }
        double error=0,norm=0,max_abs=0;for(size_t i=0;i<actual.size();++i){check(std::isfinite(actual[i])&&std::isfinite(reference[i]),"Nonfinite output");
            double delta=double(actual[i])-reference[i];error+=delta*delta;norm+=double(reference[i])*reference[i];max_abs=std::max(max_abs,std::abs(delta));}
        const double relative=std::sqrt(error/std::max(norm,1e-300));row.update({{"relative_l2",relative},{"max_abs",max_abs},{"finite",true}});
        passed&=relative<.02&&max_abs<.02;results.push_back(row);std::cout<<row.dump()<<std::endl;
        ggml_backend_buffer_free(buffer);ggml_free(ctx);
    }
    ggml_backend_free(backend);std::cout<<json{{"component_only",true},{"passed",passed},{"results",results}}.dump()<<std::endl;return passed?0:2;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
