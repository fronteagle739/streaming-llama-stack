#pragma once
#include <climits>

inline bool ggml_dsv41_index_fp4_enabled(const char *value){
    return !value||!value[0]||value[0]=='1';
}
// proof = invalid bits, maximum per-row spans Q/K, minimum low bits Q/K,
// maximum high bits Q/K. 128 terms require seven additional magnitude bits.
// At most16 product-span bits + seven sum bits fit the24-bit significand.
inline bool ggml_dsv41_index_fp4_certified(const int *proof){
    const bool emptyq=proof[3]==INT_MAX,emptyk=proof[4]==INT_MAX;
    return proof[0]==0&&(emptyq||emptyk||(proof[1]+proof[2]<=16&&
        proof[3]+proof[4]>=-126&&proof[5]+proof[6]<=120));
}
