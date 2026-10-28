#pragma once
#include <cstdlib>
#include <cstring>
// E2c candidate: retain decode expert placement but execute the original sweep
// graph for a bounded append. Never changes KV type or the arithmetic policy.
inline bool llama_dsv41_resident_append_enabled(){
    const char *value=std::getenv("LLAMADECODE_APPEND_RESIDENT_SWEEP");
    return value&&!std::strcmp(value,"1");
}
constexpr unsigned llama_dsv41_resident_append_limit=32;
