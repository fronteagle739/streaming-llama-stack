// E7 component only. Read one actual Q4_K expert directly into bounded RAM;
// never write weights, and never infer full-engine speed/quality from this graph.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;
static void check(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
struct fixture{std::vector<unsigned char> gate,up,down;};
static fixture read_weights(const json&m){
    const auto path=std::filesystem::u8path(m.at("model").get<std::string>());
    check(std::filesystem::file_size(path)==m.at("model_bytes").get<uint64_t>(),"Model size changed");
    std::ifstream file(path,std::ios::binary);check(bool(file),"Model unavailable");
    fixture f;std::vector<unsigned char>*buffers[]={&f.gate,&f.up,&f.down};
    for(size_t i=0;i<3;++i){const auto&t=m.at("tensors").at(i);const uint64_t bytes=t.at("bytes"),offset=t.at("offset");
        check(bytes==6635520&&offset+bytes<=std::filesystem::file_size(path),"Unbounded/non-Q4 expert extent");
        buffers[i]->resize(size_t(bytes));file.seekg(std::streamoff(offset));file.read(reinterpret_cast<char*>(buffers[i]->data()),std::streamsize(bytes));
        check(uint64_t(file.gcount())==bytes,"Short expert read");}
    return f;
}
static std::pair<double,std::vector<float>> run(ggml_backend_t backend,const fixture&f,int rows){
    ggml_init_params params{16*1024*1024,nullptr,true};auto*ctx=ggml_init(params);check(ctx,"Context allocation failed");
    auto*gate=ggml_new_tensor_2d(ctx,GGML_TYPE_Q4_K,5120,2304);
    auto*up=ggml_new_tensor_2d(ctx,GGML_TYPE_Q4_K,5120,2304);
    auto*down=ggml_new_tensor_2d(ctx,GGML_TYPE_Q4_K,2304,5120);
    auto*x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,5120,rows);
    auto*a=ggml_mul_mat(ctx,gate,x);auto*b=ggml_mul_mat(ctx,up,x);
    a=ggml_cast(ctx,ggml_cast(ctx,a,GGML_TYPE_F16),GGML_TYPE_F32);
    b=ggml_cast(ctx,ggml_cast(ctx,b,GGML_TYPE_F16),GGML_TYPE_F32);
    auto*mid=ggml_mul(ctx,ggml_silu(ctx,ggml_clamp(ctx,a,-INFINITY,10.f)),ggml_clamp(ctx,b,-10.f,10.f));
    mid=ggml_cast(ctx,ggml_cast(ctx,mid,GGML_TYPE_F16),GGML_TYPE_F32);
    auto*y=ggml_mul_mat(ctx,down,mid);ggml_set_output(y);
    auto*graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);
    auto*buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);check(buffer,"Backend allocation failed");
    check(ggml_nbytes(gate)==f.gate.size()&&ggml_nbytes(up)==f.up.size()&&ggml_nbytes(down)==f.down.size(),"Expert shape changed");
    ggml_backend_tensor_set(gate,f.gate.data(),0,f.gate.size());ggml_backend_tensor_set(up,f.up.data(),0,f.up.size());ggml_backend_tensor_set(down,f.down.data(),0,f.down.size());
    std::vector<float>input(size_t(5120)*rows);
    for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i%8191)*.03125f)*.2f;
    ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
    auto compute=[&]{check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"Expert component compute failed");ggml_backend_synchronize(backend);};
    compute();compute();std::vector<double> times;
    for(int repeat=0;repeat<3;++repeat){auto start=std::chrono::steady_clock::now();compute();times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());}
    std::sort(times.begin(),times.end());std::vector<float>output(size_t(5120)*rows);
    ggml_backend_tensor_get(y,output.data(),0,output.size()*4);
    for(float v:output)check(std::isfinite(v),"Nonfinite expert output");
    ggml_backend_buffer_free(buffer);ggml_free(ctx);return {times[1],std::move(output)};
}
int main(int argc,char**argv){try{
    check(argc==2,"Supply a bounded expert manifest");std::ifstream file(std::filesystem::u8path(argv[1]));json manifest;file>>manifest;
    auto weights=read_weights(manifest);auto cpu=ggml_backend_cpu_init();check(cpu,"CPU backend unavailable");
    ggml_backend_cpu_set_n_threads(cpu,8);auto params=ggml_threadpool_params_default(8);auto pool=ggml_threadpool_new(&params);
    check(pool,"CPU threadpool unavailable");ggml_backend_cpu_set_threadpool(cpu,pool);
    auto gpu=ggml_backend_cuda_init(0);check(gpu,"Exclusive configured CUDA GPU unavailable");
    json results=json::array();
    for(int rows:{1,32,128,256}){auto a=run(cpu,weights,rows);auto b=run(gpu,weights,rows);double error=0,norm=0,max_abs=0;
        for(size_t i=0;i<a.second.size();++i){const double delta=double(a.second[i])-b.second[i];error+=delta*delta;norm+=double(a.second[i])*a.second[i];max_abs=std::max(max_abs,std::abs(delta));}
        results.push_back({{"rows",rows},{"cpu_ms",a.first},{"gpu_ms",b.first},{"relative_l2",std::sqrt(error/std::max(norm,1e-300))},{"max_abs",max_abs},{"finite",true}});
        std::cout<<results.back().dump()<<std::endl;
    }
    ggml_backend_free(gpu);ggml_backend_free(cpu);ggml_threadpool_free(pool);
    std::cout<<json{{"component_only",true},{"cpu_threads",8},{"weights_bytes",19906560},{"results",results}}.dump()<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
