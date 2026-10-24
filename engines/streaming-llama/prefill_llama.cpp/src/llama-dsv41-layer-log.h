#pragma once
// LLAMADECODE: per-layer diagnostics are opt-in during a decode step.
// A 1-row decode token used to emit ~190 KB of per-layer DS4PERF lines, which
// floods a bounded parent diagnostic pipe on long decodes. While a decode-step
// scope is active, per-layer lines print only with LLAMADECODE_LAYER_LOG=1;
// the step prints one summary line instead. Prefill logging is unchanged.
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

inline std::atomic<int> &llama_dsv41_decode_quiet_depth(){static std::atomic<int>depth{0};return depth;}
inline bool llama_dsv41_layer_log_on(){
    if(llama_dsv41_decode_quiet_depth().load(std::memory_order_acquire)==0)return true;
    static const bool forced=[]{const char*e=std::getenv("LLAMADECODE_LAYER_LOG");return e&&e[0]=='1'&&!e[1];}();
    return forced;
}
inline void llama_dsv41_layer_log(const char *format,...){
    if(!llama_dsv41_layer_log_on())return;
    va_list args;va_start(args,format);std::vfprintf(stderr,format,args);va_end(args);
}
struct llama_dsv41_decode_quiet_scope{
    llama_dsv41_decode_quiet_scope(){llama_dsv41_decode_quiet_depth().fetch_add(1,std::memory_order_acq_rel);}
    ~llama_dsv41_decode_quiet_scope(){llama_dsv41_decode_quiet_depth().fetch_sub(1,std::memory_order_acq_rel);}
    llama_dsv41_decode_quiet_scope(const llama_dsv41_decode_quiet_scope&)=delete;
    llama_dsv41_decode_quiet_scope&operator=(const llama_dsv41_decode_quiet_scope&)=delete;
};
