// SPDX-License-Identifier: MIT
// Integration oracle: canonical stored weights -> layout adapter -> FlashInfer -> route scatter.
#include "flashinfer_bridge.h"
#include <cuda_bf16.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
struct native_block {uint8_t scale,codes[16];};
__host__ __device__ static unsigned hash32(unsigned e,unsigned row,unsigned k){
    unsigned x=(e+1)*0x9e3779b9u^(row+1)*0x85ebca6bu^(k+1)*0xc2b2ae35u;
    x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;return x;
}
__host__ __device__ static int pattern(int block){return block%4==3?0:block%4;}
static void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));std::exit(2);}}
__global__ static void fill_weights(native_block *out,size_t count,int n,int k){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
    int kb=i%(k/32),column=i/(k/32)%n,e=i/(size_t(n)*(k/32));native_block b;
    b.scale=127+pattern(kb);
    for(int j=0;j<16;++j)b.codes[j]=(hash32(e,column,kb*32+j)&15)|((hash32(e,column,kb*32+j+16)&15)<<4);
    out[i]=b;
}
__global__ static void fill_input(float *out,int rows,int k,const int *bounds){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(rows)*k)return;
    int row=i/k,column=i%k,e=0;while(bounds[e+1]<=row)++e;
    int m=row-bounds[e],pp=pattern(column/32);
    out[size_t(rows-1-row)*k+column]=ldexpf(int(hash32(e,m,column)%5)-2,(pp+e+m)%3);
}
int main(int argc,char **argv){
    // Supply the verified GGUF and absolute tensor data offset as two extra
    // arguments. Only four independently referenced expert matrices are read.
    if(argc!=7)return 1;int n=std::atoi(argv[2]),k=std::atoi(argv[3]);
    if(!((n==2304&&k==5120)||(n==5120&&k==2304)))return 1;
    std::vector<int> counts(384),bounds(385,0);std::ifstream file(argv[4],std::ios::binary);
    if(!file.read(reinterpret_cast<char*>(counts.data()),384*4)||file.peek()!=std::char_traits<char>::eof())return 1;
    for(int e=0;e<384;++e){if(counts[e]<0||counts[e]>4096)return 1;bounds[e+1]=bounds[e]+counts[e];}
    int rows=bounds.back();if(rows<1||rows>4096*6||rows%6)return 1;
    std::vector<int>active;for(int e=0;e<384;++e)if(counts[e])active.push_back(e);
    const int hottest=std::max_element(counts.begin(),counts.end())-counts.begin();
    std::vector<std::vector<native_block>> native_weights(384);
    std::ifstream gguf(argv[5],std::ios::binary);const auto tensor_offset=std::strtoull(argv[6],nullptr,10);
    if(!gguf||!tensor_offset)return 1;
    const size_t expert_blocks=size_t(n)*k/32;
    for(int e:{active.front(),active[active.size()/2],active.back(),hottest}){
        if(!native_weights[e].empty())continue;
        native_weights[e].resize(expert_blocks);
        gguf.seekg(tensor_offset+size_t(e)*expert_blocks*sizeof(native_block));
        if(!gguf.read(reinterpret_cast<char*>(native_weights[e].data()),expert_blocks*sizeof(native_block)))return 1;
        for(const auto &b:native_weights[e])if(b.scale==255)return 1;
    }
    auto library=LoadLibraryA(argv[1]);if(!library){std::fprintf(stderr,"LoadLibrary failed: %lu\n",GetLastError());return 2;}
    auto call=reinterpret_cast<slstack_flashinfer_mxfp4_fn>(GetProcAddress(library,"slstack_flashinfer_mxfp4"));if(!call)return 2;
    auto abi=reinterpret_cast<int(*)()>(GetProcAddress(library,"slstack_flashinfer_mxfp4_abi"));if(!abi||abi()!=3)return 2;
    native_block *weights;float *input,*output;int *gather,*scatter,*gpu_bounds;void *workspace;
    check(cudaMalloc(&workspace,ggml_dsv41_flashinfer::workspace_bytes));
    size_t blocks=size_t(384)*n*k/32;
    check(cudaMalloc(&weights,blocks*sizeof(native_block)));check(cudaMalloc(&input,size_t(rows)*k*4));check(cudaMalloc(&output,size_t(rows)*n*4));
    check(cudaMalloc(&gather,rows*4));check(cudaMalloc(&scatter,rows*4));check(cudaMalloc(&gpu_bounds,385*4));
    std::vector<int> hg(rows),hs(rows);for(int i=0;i<rows;++i){hg[i]=rows-1-i;hs[i]=(i+13)%rows;}
    check(cudaMemcpy(gather,hg.data(),rows*4,cudaMemcpyHostToDevice));check(cudaMemcpy(scatter,hs.data(),rows*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(gpu_bounds,bounds.data(),385*4,cudaMemcpyHostToDevice));
    fill_weights<<<unsigned((blocks+255)/256),256>>>(weights,blocks,n,k);
    for(int e=0;e<384;++e)if(!native_weights[e].empty())
        check(cudaMemcpy(weights+size_t(e)*expert_blocks,native_weights[e].data(),expert_blocks*sizeof(native_block),cudaMemcpyHostToDevice));
    fill_input<<<unsigned((size_t(rows)*k+255)/256),256>>>(input,rows,k,gpu_bounds);
    check(cudaMemset(output,0xff,size_t(rows)*n*4));check(cudaDeviceSynchronize());
    check(static_cast<cudaError_t>(call(weights,input,gather,scatter,gpu_bounds,output,rows,n,k,k,n,n*6,workspace,ggml_dsv41_flashinfer::workspace_bytes,nullptr)));
    check(cudaDeviceSynchronize());
    std::vector<float> actual(size_t(rows)*n);check(cudaMemcpy(actual.data(),output,actual.size()*4,cudaMemcpyDeviceToHost));
    for(float value:actual)if(!std::isfinite(value)){std::fprintf(stderr,"Nonfinite/unwritten output\n");return 3;}
    const float lut[16]={0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6};
    int checked=0;double maximum=0;
    for(int e:{active.front(),active[active.size()/2],active.back(),hottest})for(int m:{0,counts[e]/2,counts[e]-1})for(int column:{0,n/2,n-1}){
        double reference=0;
        for(int q=0;q<k;++q){
            const int pp=pattern(q/32);const auto &b=native_weights[e][size_t(column)*(k/32)+q/32];
            const int code=(b.codes[q%16]>>(q%32>=16?4:0))&15;
            reference+=std::ldexp(double(int(hash32(e,m,q)%5)-2)*lut[code],int(b.scale)-127+(pp+e+m)%3);
        }
        float expected=__bfloat162float(__float2bfloat16_rn(float(reference)));
        float value=actual[size_t(hs[bounds[e]+m])*n+column];maximum=std::max(maximum,std::abs(double(value)-expected));++checked;
    }
    if(maximum){std::fprintf(stderr,"Adapter oracle mismatch: %.9g\n",maximum);return 3;}
    std::printf("{\"status\":\"PASS\",\"N\":%d,\"K\":%d,\"rows\":%d,\"weight_source\":\"stored-MXFP4-proxy-GGUF\",\"full_output_finite\":true,\"sampled_double_reference_elements\":%d,\"max_absolute_error\":0,\"gather_and_scatter_permuted\":true}\n",n,k,rows,checked);
    check(cudaFree(workspace));check(cudaFree(weights));check(cudaFree(input));check(cudaFree(output));check(cudaFree(gather));check(cudaFree(scatter));check(cudaFree(gpu_bounds));return 0;
}
