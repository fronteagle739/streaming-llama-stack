#pragma once
#include <cstdint>
#include <stdexcept>
namespace ds4perf_output_budget {
constexpr uint64_t gib=1024ull*1024*1024,legacy_floor=30*gib;
// Exact native wire plan + fixed receipt cap are admitted separately. This
// branch admits no file carry or other outstanding disk payloads.
inline uint64_t floor(bool ram,bool native,uint64_t carry_file,uint64_t carry_pending,uint64_t output_pending){
 return ram&&native&&!carry_file&&!carry_pending&&!output_pending?gib:legacy_floor;
}
inline uint64_t required(uint64_t limit,uint64_t written,uint64_t pending,uint64_t reserve){
 if(written>limit||(reserve!=gib&&reserve!=legacy_floor))throw std::runtime_error("Invalid native output budget");
 uint64_t need=limit-written;
 if(pending>UINT64_MAX-need)throw std::runtime_error("Output pending capacity overflow");need+=pending;
 if(reserve>UINT64_MAX-need)throw std::runtime_error("Output reserve capacity overflow");return need+reserve;
}
inline bool fits(uint64_t available,uint64_t limit,uint64_t written,uint64_t pending,uint64_t reserve){
 return available>=required(limit,written,pending,reserve);
}
}
