#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "nlohmann/json.hpp"
#include "llama-dsv41-kv-fp4-codec.h"
using json=nlohmann::json;
int main(int argc,char**argv){try{
 if(argc!=4)throw std::runtime_error("manifest samples output required");
 // Safety gates: signed zero round trips; noncanonical and nonfinite values refuse.
 for(int format:{2,3}){
  float zero[32]={};for(int i=0;i<32;i+=2)zero[i]=-0.f;
  uint8_t packed[17]={},eight[33]={};llama_dsv41_kv_fp4::encode_block(zero,packed,eight,format);
  for(int i=0;i<(format==3?16:32);++i)if((eight[1+i]&8)!=(i%2?0:8))throw std::runtime_error("Signed zero lost");
  for(float bad:{NAN,INFINITY,0.12345f}){zero[0]=bad;bool refused=false;
   try{llama_dsv41_kv_fp4::encode_block(zero,packed,eight,format);}catch(const std::invalid_argument&){refused=true;}
   if(!refused)throw std::runtime_error("Invalid codec input was not refused");}
 }
 json m=json::parse(std::ifstream(argv[1])),out={{"status","PASS"},{"component_only",true},{"boundary_refusals_pass",true},{"segments",json::array()}};
 std::ifstream f(argv[2],std::ios::binary);
 for(auto row:m.at("kv_samples").at("segments")){
  const std::string kind=row.at("kind");const int format=kind=="compressed"?3:2,n=format==3?16:32;
  const size_t bytes=row.at("bytes"),width=row.at("width");
  if((kind!="compressed"&&kind!="index")||width!=(format==3?512:128)||!bytes||bytes>128*width*4||bytes%(n*4))throw std::runtime_error("Invalid bounded KV segment");
  std::vector<float>v(bytes/4);f.seekg(row.at("offset").get<uint64_t>());f.read(reinterpret_cast<char*>(v.data()),bytes);if(!f)throw std::runtime_error("Truncated samples");
  size_t changed=0;double max_error=0;std::vector<uint8_t>packed(v.size()/n*(n/2+1)),eight(v.size()/n*(n+1));
  for(size_t i=0;i<v.size();i+=n){
   auto*p=packed.data()+i/n*(n/2+1);auto*e=eight.data()+i/n*(n+1);
   llama_dsv41_kv_fp4::encode_block(v.data()+i,p,e,format);
   const float scale=format==3?ggml_dsv41_e4m3_value(p[0]):ggml_dsv41_bits_float(uint32_t(p[0])<<23);
   for(int j=0;j<n;++j){const unsigned c=(p[1+j/2]>>(4*(j%2)))&15;
    if(c!=e[1+j])throw std::runtime_error("Packed/eight-bit code mismatch");
    const float value=ggml_dsv41_bf16(std::copysign(ggml_dsv41_e2m1_value(c&7),c&8?-1.f:1.f)*scale);
    if(ggml_dsv41_float_bits(value)!=ggml_dsv41_float_bits(v[i+j])){++changed;max_error=std::max(max_error,double(std::fabs(value-v[i+j])));}
   }
  }
  row["source_values"]=v.size();row["packed_bytes"]=packed.size();row["eight_bit_code_bytes"]=eight.size();row["changed_values"]=changed;row["max_abs_error"]=max_error;out["segments"].push_back(row);
  if(changed)out["status"]="FAIL";
 }
 std::ofstream(argv[3])<<out.dump(2)<<"\n";std::cout<<out.dump()<<std::endl;return out["status"]=="PASS"?0:1;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
