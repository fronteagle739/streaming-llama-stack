#include "ggml-dsv41-carry-copy.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <limits>
__global__ void poison_after_validation(float *x){*x=999.f;}
int main(int argc,char **argv){
    const int bad=argc>1?std::atoi(argv[1]):0;
    float *values;int32_t *ids;uint32_t *flags;
    if(cudaMalloc(&values,20*sizeof(float))||cudaMalloc(&ids,2*sizeof(int32_t))||cudaMalloc(&flags,4))return 2;
    float host[20]={};int32_t index[2]={-1,0};
    if(bad>=1&&bad<=4)host[bad-1]=std::numeric_limits<float>::infinity();
    if(bad==5)index[0]=8;if(bad==6)index[1]=-1;
    if(cudaMemcpy(values,host,sizeof(host),cudaMemcpyHostToDevice)||cudaMemcpy(ids,index,sizeof(index),cudaMemcpyHostToDevice))return 3;
    ggml_dsv41_validation p;p.residual=values;p.pre=values+1;p.mean=values+2;p.engram=values+3;
    p.residual_n=p.pre_n=p.mean_n=p.engram_n=p.selected_n=p.candidates_n=1;
    p.selected=ids;p.candidates=ids+1;p.compressed_rows=8;
    cudaStream_t stream;if(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking))return 4;
    for(int step=0;step<(bad?1:100);++step)if(ggml_dsv41_cuda_validate_async(p,flags,stream))return 5;
    poison_after_validation<<<1,1,0,stream>>>(values+19);
    const auto status=cudaStreamSynchronize(stream);
    if(bad){if(status==cudaSuccess)return 6;std::printf("PASS invalid field %d rejected on GPU: %s\n",bad,cudaGetErrorString(status));return 0;}
    if(status||cudaMemcpy(host,values,sizeof(host),cudaMemcpyDeviceToHost)||host[19]!=999.f)return 7;
    cudaStreamDestroy(stream);cudaFree(values);cudaFree(ids);cudaFree(flags);
    std::puts("PASS100 queued complete validators and dependent consumer; no intermediate host waits");
}
