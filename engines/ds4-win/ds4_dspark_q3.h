/* Q3_K bit layout from ggml CUDA dequantize_q3_K (MIT, ggml contributors).
 * Returns integer quant and signed scale; caller applies the stored FP16 d.
 * Shared host/device tests compare all256 values with ggml's CPU decoder. */
#ifndef DS4_DSPARK_Q3_H
#define DS4_DSPARK_Q3_H
#include <stdint.h>
#ifdef __CUDACC__
#define DS4_DSPARK_Q3_FN __host__ __device__
#else
#define DS4_DSPARK_Q3_FN
#endif
static DS4_DSPARK_Q3_FN int ds4_dspark_q3_value(const uint8_t *block,unsigned index,int *scale) {
    const unsigned n=index/128u,j=(index%128u)/32u,l=index%32u;
    const unsigned is=8u*n+2u*j+l/16u;const uint8_t *s=block+96;
    unsigned us=is<4 ? (s[is]&15u)|(((s[is+8]>>0)&3u)<<4) :
                is<8 ? (s[is]&15u)|(((s[is+4]>>2)&3u)<<4) :
                is<12 ? (s[is-8]>>4)|(((s[is]>>4)&3u)<<4) :
                        (s[is-8]>>4)|(((s[is-4]>>6)&3u)<<4);
    *scale=(int)us-32;
    return (int)((block[32u+32u*n+l]>>(2u*j))&3u)-((block[l]&(1u<<(4u*n+j)))?0:4);
}
#undef DS4_DSPARK_Q3_FN
#endif
