#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
namespace llama_dsv41_flashinfer {
constexpr uint64_t reserve_bytes=3328ull*1024*1024;
inline bool requested(){const char *v=std::getenv("LLAMA_DSV41_FLASHINFER_MXFP4");return v&&!std::strcmp(v,"1");}
inline uint64_t workspace_bytes(uint64_t n,uint64_t k,uint64_t rows){
    const uint64_t sf_rows=(rows+384*3+127)/128*128;
    return 384*n*k/2+384*n*k/32+rows*k+sf_rows*k/32+rows*n*2+5*256+16ull*1024*1024;
}
}
