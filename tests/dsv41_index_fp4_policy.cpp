#include "ggml-dsv41-index-fp4-policy.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
static void check(bool ok){if(!ok)throw std::runtime_error("FP4 index proof failed");}
int main(){try{
    check(ggml_dsv41_index_fp4_enabled(nullptr));check(ggml_dsv41_index_fp4_enabled(""));
    check(ggml_dsv41_index_fp4_enabled("1"));check(!ggml_dsv41_index_fp4_enabled("0"));
    int proof[7]={0,8,8,-63,-63,60,60};check(ggml_dsv41_index_fp4_certified(proof));
    proof[0]=2;check(!ggml_dsv41_index_fp4_certified(proof));proof[0]=0;
    proof[1]=9;check(!ggml_dsv41_index_fp4_certified(proof));proof[1]=8;
    proof[3]=-64;check(!ggml_dsv41_index_fp4_certified(proof));proof[3]=-63;
    proof[5]=61;check(!ggml_dsv41_index_fp4_certified(proof));proof[5]=60;
    proof[3]=INT_MAX;proof[5]=INT_MIN;check(ggml_dsv41_index_fp4_certified(proof));
    std::mt19937 rng(4181);size_t cases=0;
    // Integer oracle for the proof's full product range, broader than the
    // actual FP4 product alphabet. Signed, canceling and permuted sums.
    for(int span=0;span<=16;++span)for(int low:{-126,-80,-12,0,16,64,100})for(int repeat=0;repeat<32;++repeat){
        const uint32_t mask=(1u<<(span+1))-1;std::array<int32_t,128>terms;
        int64_t exact=0;for(auto&x:terms){x=int32_t(rng()&mask);if(rng()&1)x=-x;exact+=x;}
        const float expected=std::ldexp(float(exact),low);
        for(int order=0;order<3;++order){float actual=0;for(auto x:terms)actual+=std::ldexp(float(x),low);
            check(std::isfinite(actual)&&actual==expected);std::shuffle(terms.begin(),terms.end(),rng);}
        ++cases;
    }
    // One bit outside the certificate permits an unrepresentable odd sum.
    int64_t outside=127ll*262143+262142;check(int64_t(float(outside))!=outside);
    std::cout<<"PASS FP4 index range certificate "<<cases<<" integer-oracle cases, three accumulation orders, boundaries and default/off toggle\n";
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
