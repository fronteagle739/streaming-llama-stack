#pragma once
#include "ggml-backend.h"
#include "ggml-impl.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

struct llama_dsv41_image_span {
    uint32_t position=0,count=0;
    std::filesystem::path file;
};

inline bool llama_dsv41_image_row(const std::vector<llama_dsv41_image_span>&spans,uint32_t position){
    for(const auto&s:spans)if(position>=s.position&&position-s.position<s.count)return true;
    return false;
}

// Image prompts use bounded128-row seed tiles. Read only the source rows for
// that tile; neither the full embedding table nor all image blocks enter RAM.
inline std::vector<float> llama_dsv41_seed_rows(ggml_tensor*embedding,const std::vector<int32_t>&ids,
        uint32_t position,const std::vector<llama_dsv41_image_span>&spans){
    constexpr size_t width=5120;
    if(ids.empty()||ids.size()>128)throw std::runtime_error("Vision seed exceeds128-row host budget");
    std::vector<float>out(ids.size()*width);
    std::vector<unsigned char>packed(embedding->nb[1]);
    const auto*traits=ggml_get_type_traits(embedding->type);
    for(size_t row=0;row<ids.size();++row){
        const uint32_t at=position+uint32_t(row);float*dst=out.data()+row*width;bool image=false;
        for(const auto&s:spans)if(at>=s.position&&at-s.position<s.count){
            if(ids[row]!=129264)throw std::runtime_error("Image span does not cover native placeholder IDs");
            std::ifstream file(s.file,std::ios::binary);file.seekg(48+uint64_t(at-s.position)*width*4);
            if(!file.read(reinterpret_cast<char*>(dst),width*4))throw std::runtime_error("Truncated projected image row");
            image=true;break;
        }
        if(!image){
            if(ids[row]<0||ids[row]>=129280)throw std::runtime_error("Vision text row token out of range");
            ggml_backend_tensor_get(embedding,packed.data(),uint64_t(ids[row])*embedding->nb[1],packed.size());
            if(embedding->type==GGML_TYPE_F32)std::memcpy(dst,packed.data(),width*4);
            else if(traits->to_float)traits->to_float(packed.data(),dst,width);
            else throw std::runtime_error("Text embedding has no exact row decoder");
        }
        if(!std::all_of(dst,dst+width,[](float x){return std::isfinite(x);}))throw std::runtime_error("Nonfinite image/text embedding");
    }
    return out;
}
