#pragma once
#include "../ggml/src/ggml-cuda/ds4split-q4k-half-mmid-plan.h"
#include "llama-dsv41-native-half-resource-proof.h"
#include "llama-dsv41-native-half-perf-resource-proof.h"
#include "../ggml/include/ggml-dsv41-expert-cublas.h"
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace llama_dsv41_expert_profile {
static_assert(llama_dsv41_native_half_perf_resource_proof::minimum_pool_reserve_bytes==llama_dsv41_native_half_resource_proof::minimum_pool_reserve_bytes&&
              llama_dsv41_native_half_perf_resource_proof::minimum_code_runtime_reserve_bytes==llama_dsv41_native_half_resource_proof::minimum_code_runtime_reserve_bytes,
              "Update selected-profile reserves if measured resource envelopes differ");
enum class kind:uint32_t { existing=0,native_half_q4k_v1=1,q8_1_activation_rounding=3,native_half_shared_sparse=4,q8_1_shared_sparse=5,native_half_fattn=6,q8_1_fattn=7
#if defined(DSV41_NATIVE_HALF_PERF_TEST) && DSV41_NATIVE_HALF_PERF_TEST == 1
 ,native_half_q4k_perf_v1=2
#endif
 };
struct policy { kind profile=kind::existing;bool dense_inputs_f16=false,index_bf16=false,token_tile_attention=false,expert_f16_cublas=false;uint64_t admitted_pool_bytes=0,admitted_code_runtime_bytes=0;bool sorted_expert_routing=false,reuse_expert_routing_map=false,hc_bf16_epilogue=false,expert_mxfp4=false,dense_activation_q8_1=false,index_key_windows=false,attention_kv_f16_table=false; };
inline uint64_t add(uint64_t a,uint64_t b){if(b>std::numeric_limits<uint64_t>::max()-a)throw std::runtime_error("native-half resource sum overflow");return a+b;}
inline bool performance(const policy &p){
#if defined(DSV41_NATIVE_HALF_PERF_TEST) && DSV41_NATIVE_HALF_PERF_TEST == 1
 return p.profile==kind::native_half_q4k_perf_v1||p.profile==kind::native_half_shared_sparse||p.profile==kind::native_half_fattn;
#else
 (void)p;return false;
#endif
}
inline bool activation_rounding_q8_1(const policy &p){return p.profile==kind::q8_1_activation_rounding||p.profile==kind::q8_1_shared_sparse||p.profile==kind::q8_1_fattn;}
inline bool shared_sparse(const policy &p){return p.profile==kind::native_half_shared_sparse||p.profile==kind::q8_1_shared_sparse;}
inline bool shared_flash_attention(const policy &p){return p.profile==kind::native_half_fattn||p.profile==kind::q8_1_fattn;}
inline const char * name(const policy &p){return activation_rounding_q8_1(p)?"q8_1_activation_rounding":performance(p)?"native_half_q4k_perf_v1":p.profile==kind::native_half_q4k_v1?"native_half_q4k_v1":"existing";}
inline uint32_t maximum_rows(const policy &p){return activation_rounding_q8_1(p)?16384:shared_sparse(p)?4096:performance(p)?llama_dsv41_native_half_perf_resource_proof::maximum_tile_rows:32;}
inline const char * resource_receipt(const policy &p){return performance(p)?llama_dsv41_native_half_perf_resource_proof::fixture_receipt_sha256:llama_dsv41_native_half_resource_proof::fixture_receipt_sha256;}
inline bool native(const policy & p){
 if(performance(p)||activation_rounding_q8_1(p)||shared_sparse(p))return true;
 if(p.profile!=kind::existing&&p.profile!=kind::native_half_q4k_v1)throw std::runtime_error("unknown explicit DS4.1 expert profile");
 return p.profile==kind::native_half_q4k_v1;
}
inline kind parse(const char * name){
 if(name&&std::strcmp(name,"q8_1_activation_rounding")==0)return kind::q8_1_activation_rounding;
#if defined(DSV41_NATIVE_HALF_PERF_TEST) && DSV41_NATIVE_HALF_PERF_TEST == 1
 if(name&&std::strcmp(name,"native_half_q4k_perf_v1")==0)return kind::native_half_q4k_perf_v1;
#endif
 if(name&&std::strcmp(name,"existing")==0)return kind::existing;
 if(name&&std::strcmp(name,"native_half_q4k_v1")==0)return kind::native_half_q4k_v1;
 throw std::runtime_error("unknown explicit DS4.1 expert profile name");
}
inline bool natural_half(const policy &p){return native(p)&&!activation_rounding_q8_1(p)&&!p.expert_f16_cublas;}
inline void validate(const policy & p,uint32_t tokens,const char * boundary_capture_flag){
 // Original-Q8 MMQ products and remaining F16 products are disjoint after the
 // native graph rewrite. Their explicit activation toggles may be combined.
 const bool selected=native(p);
 if(p.expert_mxfp4&&(!activation_rounding_q8_1(p)||p.expert_f16_cublas||p.sorted_expert_routing||tokens>4096))throw std::runtime_error("Expert FP4 requires the explicit shared expert lane, no sorted routing and<=4096tokens");
 if(p.reuse_expert_routing_map&&!p.sorted_expert_routing)throw std::runtime_error("GPU routing-map reuse requires explicit count/prefix/scatter builder");
 if(p.sorted_expert_routing&&!activation_rounding_q8_1(p))throw std::runtime_error("Count/prefix/scatter routing requires explicit shared Q8_1 expert selection");
 if(p.expert_f16_cublas&&(!selected||activation_rounding_q8_1(p)||p.admitted_pool_bytes<ggml_dsv41_expert_cublas_workspace(tokens)))
  throw std::runtime_error("F16 expert cuBLAS requires explicit native TEST selection, no Q8_1 rounding and bounded actual workspace admission");
 if(!selected){if(p.admitted_pool_bytes||p.admitted_code_runtime_bytes)throw std::runtime_error("existing expert profile has no native-half reserve");return;}
#if !defined(DSV41_NATIVE_HALF_TEST) || DSV41_NATIVE_HALF_TEST != 1
 throw std::runtime_error("native-half requires explicit TEST build; ordinary builds remain default off");
#endif
 using namespace llama_dsv41_native_half_resource_proof;
 if(!actual_fixture_reviewed||std::strlen(fixture_receipt_sha256)!=64)throw std::runtime_error("native-half actual282/operator/resource fixture proof pending review");
 if(performance(p)&&(!llama_dsv41_native_half_perf_resource_proof::resource_fixture_passed||
    std::strlen(llama_dsv41_native_half_perf_resource_proof::fixture_receipt_sha256)!=64||
    std::strcmp(mathematical_source_sha256,llama_dsv41_native_half_perf_resource_proof::mathematical_source_sha256)!=0))
    throw std::runtime_error("native-half performance resource proof invalid");
 if(!tokens||tokens>maximum_rows(p))throw std::runtime_error("native-half tile exceeds selected explicit TEST resource envelope");
 if(activation_rounding_q8_1(p)&&tokens>4096){
   if(tokens>16384||p.admitted_pool_bytes<(tokens>8192?640ull:320ull)*1024*1024||p.admitted_code_runtime_bytes<minimum_code_runtime_reserve_bytes||(boundary_capture_flag&&*boundary_capture_flag))throw std::runtime_error("8192-row shared MMQ requires320MiB explicit temporary admission and no capture");
   return;
  }
  const auto logical=ds4split_q4k_half::plan(tokens,384).bytes;
 if(!minimum_pool_reserve_bytes||p.admitted_pool_bytes<logical||p.admitted_pool_bytes<minimum_pool_reserve_bytes||
    p.admitted_code_runtime_bytes<minimum_code_runtime_reserve_bytes)throw std::runtime_error("native-half actual pool/code reserve underadmitted");
 // A future honest native-half capture arm is separate. Never silently bypass
 // requested old MMQ/projection observers or canonical replay injection.
 if(boundary_capture_flag&&*boundary_capture_flag)throw std::runtime_error("native-half natural profile refuses existing boundary/replay/projection capture; explicit native-half observer arm not implemented");
 (void)add(p.admitted_pool_bytes,p.admitted_code_runtime_bytes);
}
inline uint64_t extra_device(const policy & p){return native(p)?add(p.admitted_pool_bytes,p.admitted_code_runtime_bytes):0;}
}
