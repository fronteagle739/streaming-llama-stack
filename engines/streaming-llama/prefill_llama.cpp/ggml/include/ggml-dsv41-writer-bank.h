#pragma once
#include "ggml-backend.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace dsv41_writer_bank {
struct bank {
    uint32_t count=0;
    std::array<uint8_t,32> identity{};
    std::string sha256;
    std::vector<float> pack;
    uint64_t mask[2]{};
    void load(const std::vector<uint8_t>&bytes,const std::string&digest){
        auto check=[](bool ok,const char *why){if(!ok)throw std::runtime_error(why);};
        check(sha256.empty()&&bytes.size()>=48&&digest.size()==64,"Invalid/repeated immutable direction bank");
        auto u32=[&](size_t off){uint32_t v=0;for(unsigned i=0;i<4;++i)v|=uint32_t(bytes[off+i])<<(8*i);return v;};
        count=u32(4);check(u32(0)==1&&count>0&&count<=8&&u32(8)==40&&u32(12)==5120&&bytes.size()==48+size_t(40)*2*count*5121*4,"Direction bank packet geometry differs");
        std::memcpy(identity.data(),bytes.data()+16,32);pack.resize((bytes.size()-48)/4);std::memcpy(pack.data(),bytes.data()+48,bytes.size()-48);
        bool signal=false;
        for(size_t off=0;off<pack.size();off+=5121){
            const auto strength=pack[off];check(std::isfinite(strength)&&strength>=0&&strength<=4,"Direction strength outside0..4");
            double norm=0;for(size_t i=1;i<=5120;++i){check(std::isfinite(pack[off+i]),"Nonfinite direction vector");norm+=double(pack[off+i])*pack[off+i];}
            check(norm<=1e-12||std::abs(norm-1)<=1e-3,"Direction vector must be unit or zero");
            if(strength&&norm>1e-12){const auto writer=(off/5121)/count;mask[writer%2]|=uint64_t(1)<<(writer/2);signal=true;}
        }
        check(signal,"Empty direction bank must use ordinary off binding");sha256=digest;
    }
    bool active(uint32_t layer,uint32_t site)const{return layer<40&&site<2&&count&&(mask[site]&(uint64_t(1)<<layer));}
};
inline bank current;
inline bool requested(const ggml_tensor*t){return t&&t->op==GGML_OP_CUSTOM&&t->op_params[0]==0x57445231;}
inline bool valid(const ggml_tensor*t){
    if(!requested(t)||t->type!=GGML_TYPE_F32||t->ne[0]!=5120||t->ne[1]<=0||t->ne[2]!=1||t->ne[3]!=1||t->op_params[1]<0||t->op_params[1]>=40||t->op_params[2]<0||t->op_params[2]>1)return false;
    const auto *base=t->src[0],*weight=t->src[1],*input=t->src[2];
    return base&&weight&&input&&base->type==GGML_TYPE_F32&&weight->ne[1]==5120&&weight->ne[2]==1&&weight->ne[3]==1&&input->ne[0]==weight->ne[0]&&input->ne[1]==t->ne[1]&&input->ne[2]==1&&input->ne[3]==1&&
        !std::memcmp(base->ne,t->ne,sizeof(t->ne))&&ggml_is_contiguous(base)&&ggml_is_contiguous(input);
}
inline ggml_tensor *writer(ggml_context*ctx,ggml_tensor*base,ggml_tensor*weight,ggml_tensor*input,uint32_t layer,uint32_t site){
    if(!current.active(layer,site))return base;
    auto *out=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,5120,base->ne[1]);out->op=GGML_OP_CUSTOM;
    out->op_params[0]=0x57445231;out->op_params[1]=int32_t(layer);out->op_params[2]=int32_t(site);
    out->src[0]=base;out->src[1]=weight;out->src[2]=input;
    if(!valid(out))throw std::runtime_error("Invalid GPU direction writer graph");return out;
}
}
