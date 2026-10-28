#pragma once
// E17 component experiment only. These are the model's existing QAT formats:
// compressed KV uses E2M1/E4M3 per 16; index keys E2M1/E8M0 per 32.
// No shipping cache constructor or preset selects this experimental codec.
#include "../ggml/src/ggml-dsv41-quant.h"
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace llama_dsv41_kv_fp4 {
inline uint8_t scale_code(float scale,int format){
    if(format==2){
        const auto bits=ggml_dsv41_float_bits(scale);
        if((bits&0x807fffffu)||!(bits&0x7f800000u)||(bits&0x7f800000u)==0x7f800000u)
            throw std::invalid_argument("Invalid E8M0 KV scale");
        return uint8_t(bits>>23);
    }
    for(int i=0;i<=126;++i)if(ggml_dsv41_e4m3_value(i)==scale)return uint8_t(i);
    throw std::invalid_argument("Invalid E4M3 KV scale");
}
inline uint8_t value_code(float value,float scale){
    const float magnitude=ggml_dsv41_e2m1(ggml_dsv41_divide(fabsf(value),scale));
    for(int i=0;i<8;++i)if(ggml_dsv41_e2m1_value(i)==magnitude)
        return uint8_t(i|((ggml_dsv41_float_bits(value)>>28)&8u));
    throw std::invalid_argument("Invalid E2M1 KV code");
}
inline void encode_block(const float*input,uint8_t*packed,uint8_t*eight_bit,int format){
    if(format!=2&&format!=3)throw std::invalid_argument("Unsupported KV codec format");
    const int block=format==3?16:32;
    float amax=0;
    for(int i=0;i<block;++i){
        if(!std::isfinite(input[i]))throw std::invalid_argument("Nonfinite KV sample");
        if(ggml_dsv41_float_bits(input[i])!=ggml_dsv41_float_bits(ggml_dsv41_bf16(input[i])))
            throw std::invalid_argument("KV input is not canonical BF16 QAT output");
        amax=fmaxf(amax,fabsf(input[i]));
    }
    const float scale=ggml_dsv41_scale(amax,format);
    packed[0]=eight_bit[0]=scale_code(scale,format);
    for(int i=0;i<block;i+=2){
        const uint8_t lo=value_code(ggml_dsv41_bf16(input[i]),scale),hi=value_code(ggml_dsv41_bf16(input[i+1]),scale);
        for(int j=0;j<2;++j){
            const uint8_t code=j?hi:lo;
            const float restored=ggml_dsv41_bf16(copysignf(ggml_dsv41_e2m1_value(code&7),code&8?-1.f:1.f)*scale);
            if(ggml_dsv41_float_bits(restored)!=ggml_dsv41_float_bits(input[i+j]))
                throw std::invalid_argument("KV block cannot be packed losslessly");
        }
        packed[1+i/2]=uint8_t(lo|(hi<<4));eight_bit[1+i]=lo;eight_bit[2+i]=hi;
    }
}
}
