// SPDX-License-Identifier: MIT
#include "dense_fp8_bridge.h"
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define NOMINMAX
#include <windows.h>
struct q8_block {__half d;int8_t q[32];};
static void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));std::exit(2);}}
static __host__ __device__ unsigned hash(unsigned i){i^=i>>16;i*=0x7feb352du;i^=i>>15;i*=0x846ca68bu;return i^(i>>16);}
static __host__ __device__ int power(int row,int kb){return (row/32+kb)%5-3;}
static __host__ __device__ int code(int row,int q){return int(hash(unsigned(row)*127+q)%15)-7;}
static __host__ __device__ float xvalue(int row,int q){return ldexpf(float(int(hash(unsigned(row)*83+q)%15)-7),(row+q/32)%3-2);}
static __global__ void fill(q8_block *q8,uint8_t *native,float *x,int m,int n,int k){
    size_t i=size_t(blockIdx.x)*256+threadIdx.x;
    if(i<size_t(n)*k/32){int row=i/(k/32),kb=i%(k/32);q8_block b;b.d=__float2half(ldexpf(1.f,power(row,kb)+row%3));
        for(int j=0;j<32;++j){int c=kb*32+j;b.q[j]=int8_t(code(row,c));native[size_t(row)*k+c]=__nv_fp8_e4m3(float(code(row,c))).__x;}
        q8[i]=b;
        if(row%32==0)native[size_t(n)*k+size_t(row/32)*(k/32)+kb]=uint8_t(power(row,kb)+127);
    }
    if(i<size_t(m)*k)x[i]=xvalue(i/k,i%k);
}
static __global__ void finite(const float *x,size_t count,unsigned long long *errors){size_t i=size_t(blockIdx.x)*256+threadIdx.x;if(i<count&&!isfinite(x[i]))atomicAdd(errors,1ull);}
int main(int argc,char **argv){
    if(argc!=5)return 1;int m=std::atoi(argv[2]),n=std::atoi(argv[3]),k=std::atoi(argv[4]);
    if(m<1||m>4096||n<128||n%128||k<128||k%128)return 1;
    auto lib=LoadLibraryA(argv[1]);if(!lib)return 2;
    auto abi=reinterpret_cast<int(*)()>(GetProcAddress(lib,"slstack_dense_fp8_abi"));
    auto call=reinterpret_cast<slstack_dense_fp8_fn>(GetProcAddress(lib,"slstack_dense_fp8"));if(!abi||abi()!=1||!call)return 2;
    q8_block *q8;uint8_t *native;float *x,*out;void *workspace;unsigned long long *errors;
    check(cudaMalloc(&q8,size_t(n)*k/32*34));check(cudaMalloc(&native,size_t(n)*k+size_t(n)*k/1024));
    check(cudaMalloc(&x,size_t(m)*k*4));check(cudaMalloc(&out,size_t(m)*n*4));check(cudaMalloc(&workspace,512ull<<20));check(cudaMalloc(&errors,8));
    size_t count=size_t(n)*k/32;if(size_t(m)*k>count)count=size_t(m)*k;
    fill<<<unsigned((count+255)/256),256>>>(q8,native,x,m,n,k);check(cudaDeviceSynchronize());
    int checked=0;float timing[2]={};
    for(int format=0;format<2;++format){
        check(cudaMemset(out,0xff,size_t(m)*n*4));check(cudaMemset(errors,0,8));
        auto weights=format?static_cast<const void*>(native):static_cast<const void*>(q8);
        check(static_cast<cudaError_t>(call(weights,format,x,out,m,n,k,k,workspace,512ull<<20,nullptr)));check(cudaDeviceSynchronize());
        finite<<<unsigned((size_t(m)*n+255)/256),256>>>(out,size_t(m)*n,errors);
        check(cudaDeviceSynchronize());unsigned long long nonfinite;check(cudaMemcpy(&nonfinite,errors,8,cudaMemcpyDeviceToHost));if(nonfinite)return 3;
        for(int row:{0,m/2,m-1})for(int column:{0,n/2,n-1}){
            double reference=0;
            for(int q=0;q<k;++q)reference+=std::ldexp(double(code(column,q)),power(column,q/32)+(format?0:column%3))*double(xvalue(row,q));
            float expected=__bfloat162float(__float2bfloat16_rn(float(reference))),actual;
            check(cudaMemcpy(&actual,out+size_t(row)*n+column,4,cudaMemcpyDeviceToHost));
            if(std::memcmp(&actual,&expected,4)){std::fprintf(stderr,"Dense FP8 mismatch fmt%d [%d,%d] %.9g vs%.9g\n",format,row,column,actual,expected);return 3;}++checked;
        }
        cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));check(cudaEventRecord(start));
        for(int repeat=0;repeat<10;++repeat)check(static_cast<cudaError_t>(call(weights,format,x,out,m,n,k,k,workspace,512ull<<20,nullptr)));
        check(cudaEventRecord(end));check(cudaEventSynchronize(end));check(cudaEventElapsedTime(&timing[format],start,end));timing[format]/=10;
        check(cudaEventDestroy(start));check(cudaEventDestroy(end));
    }
    std::printf("{\"status\":\"PASS\",\"M\":%d,\"N\":%d,\"K\":%d,\"sampled_double_reference_elements\":%d,\"full_output_finite_and_written\":true,\"q8_proxy_ms\":%.6f,\"native_fp8_ms\":%.6f,\"native_fp8_tflops\":%.6f}\n",m,n,k,checked,timing[0],timing[1],2.*m*n*k/timing[1]/1.e9);
    check(cudaFree(q8));check(cudaFree(native));check(cudaFree(x));check(cudaFree(out));check(cudaFree(workspace));check(cudaFree(errors));return 0;
}
