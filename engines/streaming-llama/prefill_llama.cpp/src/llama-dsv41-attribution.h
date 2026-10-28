#pragma once
// Opt-in, synchronizing E2c diagnostic. Same record grammar as ds4-win's
// DS4COMBO_V41_DUMP: magic, layer, absolute position, point, word count, payload.
// At most16 requested positions; only one row is ever read to host at a time.
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

inline std::vector<uint32_t> llama_dsv41_attribution_positions() {
    const char *path=std::getenv("DS4COMBO_V41_DUMP");if(!path||!*path)return {};
    const char *text=std::getenv("DS4COMBO_V41_DUMP_POS");
    if(!text||!*text)text="0,128,275,549";
    std::vector<uint32_t> out;
    while(*text){
        char *end=nullptr;const unsigned long long n=std::strtoull(text,&end,10);
        if(end==text||n>std::numeric_limits<uint32_t>::max()||out.size()==16||(*end&&*end!=','))
            throw std::runtime_error("E2c attribution positions must be at most16 unsigned positions");
        out.push_back(uint32_t(n));text=*end?end+1:end;
    }
    return out;
}
inline bool llama_dsv41_attribution_active(uint32_t first,uint32_t count){
    for(auto p:llama_dsv41_attribution_positions())if(p>=first&&uint64_t(p)<uint64_t(first)+count)return true;
    return false;
}
inline void llama_dsv41_attribution_write(ggml_tensor *tensor,uint32_t first,uint32_t count,
                                         uint32_t layer,uint32_t point,uint32_t words){
    if(!llama_dsv41_attribution_active(first,count))return;
    if(!tensor||!ggml_is_contiguous(tensor)||ggml_nbytes(tensor)!=uint64_t(count)*words*4)
        throw std::runtime_error("E2c attribution tensor shape/storage mismatch");
    const char *path=std::getenv("DS4COMBO_V41_DUMP");
    FILE *file=std::fopen(path,"ab");if(!file)throw std::runtime_error("E2c attribution output unavailable");
    bool ok=true;std::vector<uint32_t> row(words);
    for(auto p:llama_dsv41_attribution_positions())if(p>=first&&uint64_t(p)<uint64_t(first)+count){
        ggml_backend_tensor_get(tensor,row.data(),uint64_t(p-first)*words*4,words*4);
        const uint32_t head[]={0x52313444u,layer,p,point,words};
        ok=ok&&std::fwrite(head,sizeof(head),1,file)==1&&std::fwrite(row.data(),4,words,file)==words;
    }
    ok=std::fclose(file)==0&&ok;if(!ok)throw std::runtime_error("E2c attribution write failed");
}
