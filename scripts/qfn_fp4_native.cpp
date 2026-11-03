// CPU-only Q2_0 -> MXFP4 proxy. Uses the fork's verbatim GGML reference
// dequantizer/quantizer via a build-generated include; no model runtime.
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cassert>
#define EXPORT extern "C" __declspec(dllexport)
#define GGML_RESTRICT
#include "qfn_fp4_reference.inc"

EXPORT void qfn_dequant(const void* src, float* dst, int64_t count) {
    dequantize_row_q2_0(static_cast<const block_q2_0*>(src),dst,count);
}
EXPORT void qfn_quant(const float* src, void* dst, int64_t count) {
    quantize_row_mxfp4_ref(src,static_cast<block_mxfp4*>(dst),count);
}
EXPORT int qfn_convert(const uint8_t* src, uint8_t* dst, int64_t blocks, int threads) {
    if(!src||!dst||blocks<0||threads<1||threads>4)return 1;
    int invalid=0;
    #pragma omp parallel for num_threads(threads) reduction(|:invalid)
    for(int64_t b=0;b<blocks;++b) {
        uint16_t h;std::memcpy(&h,src+b*18,2);const float d=half(h);
        if(!std::isfinite(d)){invalid|=1;continue;}
        const uint8_t* codes=src+b*18+2;
        for(int s=0;s<2;++s) {
            float amax=0;uint8_t q[32];
            for(int j=0;j<32;++j){const int k=s*32+j;q[j]=(codes[k/4]>>((k%4)*2))&3;
                amax=std::fmax(amax,std::fabs((int(q[j])-1)*d));}
            const int e=amax>0?int(std::floor(std::log2(amax)))-2+127:0;
            if(e<0||e>254){invalid|=1;continue;}
            const float scale=GGML_E8M0_TO_FP32_HALF(e);
            uint8_t map[4];for(int j=0;j<4;++j)map[j]=uint8_t(best_index_mxfp4((j-1)*d,scale));
            uint8_t* out=dst+(b*2+s)*17;out[0]=uint8_t(e);
            for(int j=0;j<16;++j)out[j+1]=map[q[j]]|(map[q[j+16]]<<4);
        }
    }
    return invalid;
}
