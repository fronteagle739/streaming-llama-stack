#pragma once
#include "ggml.h"
#include <stdexcept>
#include <string>

// Intersection of the combined producer's GGML path and ds4 CUDA decode.
// Auxiliary tensors retain the architecture's required representation.
namespace llama_dsv41_quant {
inline bool dense(ggml_type t) {
    return t==GGML_TYPE_Q8_0 || t==GGML_TYPE_Q4_K || t==GGML_TYPE_Q4_0;
}
inline bool triplet(ggml_type gate,ggml_type up,ggml_type down) {
    return gate==up && ((gate==GGML_TYPE_Q4_K&&down==GGML_TYPE_Q4_K) ||
        (gate==GGML_TYPE_IQ2_XXS&&down==GGML_TYPE_Q2_K) ||
        (gate==GGML_TYPE_MXFP4&&down==GGML_TYPE_MXFP4));
}
inline bool supported(ggml_type actual,ggml_type original) {
    if(original==GGML_TYPE_Q4_K) return actual==GGML_TYPE_Q4_K ||
        actual==GGML_TYPE_IQ2_XXS || actual==GGML_TYPE_Q2_K || actual==GGML_TYPE_MXFP4;
    return original==GGML_TYPE_Q8_0?dense(actual):actual==original;
}
inline void check(const std::string&name,ggml_type actual,ggml_type original) {
    if(!supported(actual,original)) throw std::runtime_error("Unsupported DSV4.1 tensor "+name+
        " type "+ggml_type_name(actual)+" ("+std::to_string(int(actual))+"); no common prefill/decode kernel");
}
}
