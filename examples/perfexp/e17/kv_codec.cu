#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>
#include "nlohmann/json.hpp"
#include "llama-dsv41-kv-fp4-codec.h"
using json=nlohmann::json;
static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__global__ void decode(const uint8_t*input,float*output,size_t values,int format,bool packed){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=values)return;
    const unsigned block=format==3?16:32;
    const size_t group=i/block,within=i%block,stride=packed?block/2+1:block+1;
    const uint8_t scale=input[group*stride];
    const unsigned code=packed?(input[group*stride+1+within/2]>>(within%2*4))&15:input[group*stride+1+within];
    const float s=format==3?ggml_dsv41_e4m3_value(scale):ggml_dsv41_bits_float(uint32_t(scale)<<23);
    output[i]=ggml_dsv41_bf16(copysignf(ggml_dsv41_e2m1_value(code&7),code&8?-1.f:1.f)*s);
}
int main(int argc,char**argv){
    json result={{"experiment","E17"},{"integrated_cache",false},{"status","RUNNING"}};
    try{
        const bool quick=argc==5&&std::string(argv[4])=="--check";
        if(argc!=4&&!quick)throw std::runtime_error("manifest samples.f32 result.json required");
        const auto manifest=json::parse(std::ifstream(argv[1]));std::ifstream input(argv[2],std::ios::binary);
        result["source_position"]=manifest.at("kv_samples").at("position");result["segments"]=json::array();
        for(const auto&segment:manifest.at("kv_samples").at("segments")){
            const size_t source_values=segment.at("bytes").get<size_t>()/4;
            std::vector<float>source(source_values);input.seekg(segment.at("offset").get<uint64_t>());
            input.read(reinterpret_cast<char*>(source.data()),std::streamsize(source_values*4));
            if(!input)throw std::runtime_error("Truncated KV sample");
            const int format=segment.at("kind")=="compressed"?3:2,block=format==3?16:32;
            const size_t width=segment.at("width");
            const std::string kind=segment.at("kind");
            if((kind!="compressed"&&kind!="index")||width!=(format==3?512:128)||!source_values||
               source_values>128*width||source_values%block)throw std::runtime_error("Invalid bounded KV sample geometry");
            const size_t values=quick?source_values:size_t(32768)*width;
            std::vector<uint8_t>packed(values/block*(block/2+1)),eight(values/block*(block+1));
            for(size_t i=0;i<values;i+=block)
                llama_dsv41_kv_fp4::encode_block(source.data()+i%source_values,packed.data()+i/block*(block/2+1),eight.data()+i/block*(block+1),format);
            size_t free=0,total=0;check(cudaMemGetInfo(&free,&total));
            const size_t needed=packed.size()+eight.size()+values*4;
            if(needed>(512ull<<20)||free<needed+(512ull<<20))throw std::runtime_error("KV component VRAM admission refused");
            uint8_t*codes=nullptr;float*out=nullptr;check(cudaMalloc(&codes,eight.size()));check(cudaMalloc(&out,values*4));
            std::vector<float>decoded(values);
            json row=segment;row["replicated_values"]=values;row["fp4_bytes"]=packed.size();row["eight_bit_code_bytes"]=eight.size();
            bool exact=true;
            for(int arm=0;arm<2;++arm){
                const auto&bytes=arm?packed:eight;check(cudaMemcpy(codes,bytes.data(),bytes.size(),cudaMemcpyHostToDevice));
                for(int warm=0;warm<(quick?0:3);++warm)decode<<<unsigned((values+255)/256),256>>>(codes,out,values,format,arm!=0);
                check(cudaGetLastError());check(cudaDeviceSynchronize());
                cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));check(cudaEventRecord(start));
                for(int repeat=0;repeat<(quick?1:100);++repeat)decode<<<unsigned((values+255)/256),256>>>(codes,out,values,format,arm!=0);
                check(cudaEventRecord(end));check(cudaEventSynchronize(end));float ms=0;check(cudaEventElapsedTime(&ms,start,end));
                row[arm?"fp4_decode_ms":"eight_bit_decode_ms"]=ms/(quick?1:100);
                check(cudaMemcpy(decoded.data(),out,values*4,cudaMemcpyDeviceToHost));
                for(size_t i=0;i<values;++i)if(ggml_dsv41_float_bits(decoded[i])!=ggml_dsv41_float_bits(source[i%source_values])){exact=false;break;}
                check(cudaEventDestroy(start));check(cudaEventDestroy(end));
            }
            check(cudaFree(out));check(cudaFree(codes));row["bit_exact_against_canonical_samples"]=exact;
            result["segments"].push_back(row);if(!exact)throw std::runtime_error("Packed KV sample is not lossless");
            std::printf("E17 segment %s/%u exact\n",segment.at("kind").get<std::string>().c_str(),segment.at("index").get<unsigned>());std::fflush(stdout);
        }
        result["status"]="PASS";
    }catch(const std::exception&e){result["status"]="FAIL";result["error"]=e.what();}
    if(argc>=4)std::ofstream(argv[3])<<result.dump(2)<<"\n";
    std::printf("%s\n",result.dump().c_str());return result["status"]=="PASS"?0:1;
}
