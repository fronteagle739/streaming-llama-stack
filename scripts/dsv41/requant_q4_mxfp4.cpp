// Q4_K -> MXFP4 speed experiment. GGML scalar algorithms, MIT licence.
// Derived from ggml-quants.c (dequantize_row_q4_K, quantize_row_mxfp4_ref).
// No GGML runtime or GPU allocation. Compile with precise floating point.
#include <cstdint>
#include <cstring>
#include <cmath>
#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif
static float half(uint16_t h) {
    const unsigned sign=(unsigned(h)&0x8000)<<16, e=(h>>10)&31, m=h&1023;
    if (!e) return std::ldexp(float(m),-24)*(sign?-1.f:1.f);
    unsigned bits=sign|((e==31?255:e+112)<<23)|(m<<13);
    float v;std::memcpy(&v,&bits,4);return v;
}
static void scale_min(int j,const uint8_t*q,int &s,int &m) {
    if(j<4){s=q[j]&63;m=q[j+4]&63;}
    else{s=(q[j+4]&15)|((q[j-4]>>6)<<4);m=(q[j+4]>>4)|((q[j]>>6)<<4);}
}
// Each 32-value subblock has only 16 possible decoded values. Find its
// actual amax, then map those 16 values once with GGML's strict first-tie rule.
EXPORT int requant_q4_mxfp4(const uint8_t*src,uint8_t*dst,int64_t blocks,int threads) {
    if(!src||!dst||blocks<0||threads<1||threads>16)return 1;
    int invalid=0;
    #pragma omp parallel for num_threads(threads) reduction(|:invalid)
    for(int64_t b=0;b<blocks;++b) {
        const uint8_t *in=src+b*144;
        uint16_t h1,h2;std::memcpy(&h1,in,2);std::memcpy(&h2,in+2,2);
        const float d=half(h1),dm=half(h2);
        if(!std::isfinite(d)||!std::isfinite(dm)){invalid|=1;continue;}
        for(int j=0;j<8;++j) {
            int sc,m;scale_min(j,in+4,sc,m);
            const float ds=d*sc,ms=dm*m;
            const uint8_t*q=in+16+(j/2)*32;
            float values[16],amax=0;
            for(int v=0;v<16;++v)values[v]=ds*v-ms;
            uint8_t codes[32];
            for(int k=0;k<32;++k){codes[k]=j%2?q[k]>>4:q[k]&15;amax=std::fmax(amax,std::fabs(values[codes[k]]));}
            const int exponent=amax>0?int(std::floor(std::log2(amax)))-2+127:0;
            if(exponent<0||exponent>254){invalid|=1;continue;}
            const float inverse_scale=std::ldexp(1.f,127-exponent);
            uint8_t mapping[16];
            for(int v=0;v<16;++v) {
                // Power-of-two normalization is exact for finite Q4_K
                // operands. Strict midpoint comparisons retain GGML's
                // first-code tie rule, including unsigned zero.
                const float x=std::fabs(values[v])*inverse_scale;
                const unsigned c=(x>.25f)+(x>.75f)+(x>1.25f)+(x>1.75f)+
                                 (x>2.5f)+(x>3.5f)+(x>5.f);
                mapping[v]=uint8_t(c+(values[v]<0&&c?8:0));
            }
            uint8_t*out=dst+(b*8+j)*17;out[0]=uint8_t(exponent);
            for(int k=0;k<16;++k)out[k+1]=mapping[codes[k]]|(mapping[codes[k+16]]<<4);
        }
    }
    return invalid;
}
