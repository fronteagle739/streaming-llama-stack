#pragma once
// E8 opt-in placement policy. Scores change cache placement only, never routes.
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace llama_dsv41_recency_lfu {
class scores {
    std::array<uint32_t,40*384> frequency{};
    uint64_t tokens=0;
public:
    void decay() { for(auto & value:frequency)value=(value+1)/2; }
    void tick(uint32_t half_life) {
        if(!half_life)throw std::invalid_argument("LFU half life must be positive");
        if(tokens++%half_life==0)decay();
    }
    void touch(int32_t key) {
        if(key<0||size_t(key)>=frequency.size())throw std::out_of_range("LFU expert key");
        auto & value=frequency[size_t(key)];
        if(value<std::numeric_limits<uint32_t>::max()-256)value+=256;
    }
    uint32_t get(int32_t key)const {
        return key<0 ? 0 : frequency.at(size_t(key));
    }
    // 25% admission margin prevents near-equal experts cycling through slots.
    bool admit(int32_t candidate,int32_t victim)const {
        return victim<0 || uint64_t(get(candidate))*4>uint64_t(get(victim))*5;
    }
};
}
