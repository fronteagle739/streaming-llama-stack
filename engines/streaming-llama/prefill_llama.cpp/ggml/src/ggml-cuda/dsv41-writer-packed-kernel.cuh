#pragma once
// Projection of the writer vectors through packed down weights. Shared with the
// component checker, so the check runs the production kernel and not a copy.
#include "../../../../../ds4-win/cuda/mmq/ds4_fp4_packed_layout.cuh"
__global__ static void dsv41_writer_packed_a_kernel(float*a,const uint8_t*values,const uint8_t*scales,const float*pack,int layer,int rank,bool slots){
    const int col=int(blockIdx.x)*32+threadIdx.x,expert=int(blockIdx.y);const int input=2304,output=5120;
    const float *vectors=pack+(size_t(layer)*2+1)*rank*5121;
    const size_t stride=size_t(output)*input/2+size_t(output)*input/32;
    const auto *v=values+size_t(expert)*(slots?stride:size_t(output)*input/2),*s=scales+size_t(expert)*(slots?stride:size_t(input/32)*output);
    float sums[8]={};
    if(col<input)for(int row=threadIdx.y;row<output;row+=8){const float weight=ds4_fp4p_weight_planes(v,s,output,input,row,col);
        for(int d=0;d<rank;++d)sums[d]+=vectors[size_t(d)*5121+1+row]*weight;}
    __shared__ float partial[8][8][32];for(int d=0;d<rank;++d)partial[d][threadIdx.y][threadIdx.x]=sums[d];__syncthreads();
    if(!threadIdx.y&&col<input)for(int d=0;d<rank;++d){float sum=0;for(int r=0;r<8;++r)sum+=partial[d][r][threadIdx.x];a[(size_t(expert)*rank+d)*input+col]=sum;}
}
