#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "ggml-dsv41-flashinfer-workspace.h"
namespace llama_dsv41_flashinfer {
inline uint64_t reserve_bytes(){return ggml_dsv41_flashinfer::selected_reserve_bytes();}
inline bool requested(){const char *v=std::getenv("LLAMA_DSV41_FLASHINFER_MXFP4");return v&&!std::strcmp(v,"1");}
inline uint64_t workspace_bytes(uint64_t n,uint64_t k,uint64_t rows){
    (void)n;(void)k;(void)rows;
    // GGML legacy-pool allocation look-ahead, plus AOT module headroom.
    return ggml_dsv41_flashinfer::align256((ggml_dsv41_flashinfer::selected_workspace_bytes()+255)*105/100+1)+16ull*1024*1024;
}
}
