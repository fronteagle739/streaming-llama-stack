#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

// Complement of the bytes that a complete, verified DirectStorage fill writes.
// This is destination geometry only; the caller independently proves that every
// request covers its source span exactly once before using the result.
namespace llama_wf_zero_gaps {
struct interval { uint64_t offset, bytes; };
inline bool cover(const interval & span,uint64_t offset,uint64_t bytes,uint64_t & covered){
    if(!bytes||covered>span.bytes||bytes>span.bytes-covered||
       span.offset>UINT64_MAX-covered||offset!=span.offset+covered)return false;
    covered+=bytes;return true;
}
inline bool plan(uint64_t extent, const std::vector<interval> & writes,
                 std::vector<interval> & gaps, std::string & error) {
    gaps.clear();error.clear();
    auto fail=[&](const char * why){gaps.clear();error=why;return false;};
    if (!extent || writes.empty() || writes.size()>8192)
        return fail("Invalid bounded direct-write extent/span count");
    auto sorted=writes;
    std::sort(sorted.begin(),sorted.end(),[](const interval&a,const interval&b){return a.offset<b.offset;});
    uint64_t end=0;
    for(const auto&s:sorted){
        if(!s.bytes || s.offset>extent || s.bytes>extent-s.offset || s.offset<end)
            return fail("Overlapping, empty or out-of-range direct-write span");
        if(s.offset>end)gaps.push_back({end,s.offset-end});
        end=s.offset+s.bytes;
    }
    if(end<extent)gaps.push_back({end,extent-end});
    return true;
}
}
