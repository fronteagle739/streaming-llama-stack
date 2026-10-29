#pragma once
#include "common.cuh"
#include "../ggml-dsv41-quant.h"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

struct dsv41_layer_norm_args {
    const float *residual,*pre,*weight;
    float *output;
    int width,rows;
    float eps;
};
static __device__ float dsv41_layer_mul_rn(float x,float y){float z;asm("mul.rn.f32 %0,%1,%2;":"=f"(z):"f"(x),"f"(y));return z;}
static __device__ float dsv41_layer_add_rn(float x,float y){float z;asm("add.rn.f32 %0,%1,%2;":"=f"(z):"f"(x),"f"(y));return z;}

// Preserve the HC BF16 boundary, the ordinary CUDA RMS reduction tree, both
// rounded multiplies and the final BF16 boundary. One CTA owns an entire row.
template<int block_size,bool indirect=false>
static __global__ void dsv41_layer_norm_kernel(dsv41_layer_norm_args direct,
        const dsv41_layer_norm_args *table=nullptr){
    const auto a=indirect?*table:direct;
    extern __shared__ float shared[];
    float *values=shared+32;
    for(int row=blockIdx.x;row<a.rows;row+=gridDim.x){
        float sum=0.f;
        for(int d=threadIdx.x;d<a.width;d+=block_size){
            float x=dsv41_layer_mul_rn(a.residual[(int64_t(row)*4)*a.width+d],a.pre[row*4]);
            #pragma unroll
            for(int c=1;c<4;++c)x=dsv41_layer_add_rn(x,dsv41_layer_mul_rn(a.residual[(int64_t(row)*4+c)*a.width+d],a.pre[row*4+c]));
            x=ggml_dsv41_bf16(x);values[d]=x;
            // norm.cu is built with the backend's -use_fast_math policy.
            // Keep its FTZ FMA and approximate divide/rsqrt explicitly while
            // HC above retains its separately qualified RN/no-FTZ contract.
            asm("fma.rn.ftz.f32 %0,%1,%1,%2;":"=f"(sum):"f"(x),"f"(sum));
        }
        sum=block_reduce<block_reduce_method::SUM,block_size>(sum,shared);
        float mean,shifted,scale;
        asm("div.approx.ftz.f32 %0,%1,%2;":"=f"(mean):"f"(sum),"f"(float(a.width)));
        asm("add.rn.ftz.f32 %0,%1,%2;":"=f"(shifted):"f"(mean),"f"(a.eps));
        asm("rsqrt.approx.ftz.f32 %0,%1;":"=f"(scale):"f"(shifted));
        for(int d=threadIdx.x;d<a.width;d+=block_size){
            float normalized,weighted;
            asm("mul.rn.ftz.f32 %0,%1,%2;":"=f"(normalized):"f"(scale),"f"(values[d]));
            asm("mul.rn.ftz.f32 %0,%1,%2;":"=f"(weighted):"f"(normalized),"f"(a.weight[d]));
            a.output[int64_t(row)*a.width+d]=ggml_dsv41_bf16(weighted);
        }
        __syncthreads();
    }
}
#endif

bool ggml_cuda_dsv41_layer_norm_match(const ggml_cgraph *,int);
void ggml_cuda_dsv41_layer_norm(ggml_backend_cuda_context &,ggml_cgraph *,int);
