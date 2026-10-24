#pragma once
// Admission of the explicit F16 dense toggle for one layer graph (LLAMADECODE B0).
// Prefill-sized steps must tag at least one allowlisted projection, or the
// toggle is refused as inert. Steps at or below the shape rule's row limit
// (1-row decode, short tails) cannot be tagged by construction; for them the
// proof is the converse: no projection may carry the tag.
#include "ggml-dsv41-dense-half.h"
#include <cstdint>

inline const char *llama_dsv41_dense_half_admission(uint32_t rows,unsigned accepted){
    if(int64_t(rows)<=ggml_dsv41_dense_half_max_untagged_rows)
        return accepted==0?nullptr:"F16 dense tag applied at a row count the shape rule excludes";
    return accepted>0?nullptr:"Explicit F16 dense toggle matched no admitted projection";
}
