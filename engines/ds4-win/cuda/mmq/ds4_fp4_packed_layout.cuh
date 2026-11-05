// Shared address rules for one FlashInfer SM120 MXFP4 expert cache slot.
// The slot contains all adjacent E2M1 values, then [K/128][N][4] E8M0 scales.
// No repack or host weight read is involved.
#ifndef DS4_FP4_PACKED_LAYOUT_CUH
#define DS4_FP4_PACKED_LAYOUT_CUH

static __host__ __device__ __forceinline__ uint64_t ds4_fp4p_scale_offset(
        uint32_t rows, uint32_t cols, uint32_t row, uint32_t block32) {
    return (uint64_t)rows*(cols/2u)+((uint64_t)(block32/4u)*rows+row)*4u+block32%4u;
}

static __device__ __forceinline__ float ds4_fp4p_weight_planes(
        const unsigned char *values,const unsigned char *scales,uint32_t rows,uint32_t cols,
        uint32_t row, uint32_t col) {
    const uint8_t pair=values[(uint64_t)row*(cols/2u)+col/2u];
    const uint32_t code=(pair>>(4u*(col&1u)))&15u;
    const uint32_t exponent=scales[ds4_fp4p_scale_offset(rows,cols,row,col/32u)-(uint64_t)rows*(cols/2u)];
    const float scale=__uint_as_float(exponent?exponent<<23:0x00400000u);
    const uint32_t mag=code&7u;
    const float value=mag<4u?(float)mag*.5f:
        (float)(2u+((mag-4u)&1u))*(float)(1u<<((mag-4u)>>1));
    return scale*((code&8u)?-value:value);
}
static __device__ __forceinline__ float ds4_fp4p_weight(
        const unsigned char *source,uint32_t rows,uint32_t cols,uint32_t row,uint32_t col){
    return ds4_fp4p_weight_planes(source,source+(uint64_t)rows*(cols/2u),rows,cols,row,col);
}
#endif
