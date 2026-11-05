#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

// Isolated encoder worker: no language weights or KV are opened here. The
// launcher must retire its language process first, as the DSV vision path does.
struct qfn_vision_result {
    uint32_t count=0,width=0,grid_height=0,grid_width=0;
    uint32_t image_width=0,image_height=0;
    std::array<uint8_t,32> fingerprint{};
    std::vector<float> values;
    double preprocess_seconds=0,load_seconds=0,encode_seconds=0,unload_seconds=0;
    uint64_t free_before=0,free_after_load=0,free_after_unload=0;
};
bool qfn_vision_encode_file(const std::string &model,const std::string &image,
    uint32_t minimum,uint32_t maximum,int threads,const std::string &backend,bool flash_attention,
    qfn_vision_result &result,std::string &error);
