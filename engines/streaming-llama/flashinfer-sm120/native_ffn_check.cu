// SPDX-License-Identifier: MIT
// Full-output comparison with the separately built, qualified ABI4 adapter.
#include "flashinfer_native_bridge.h"
#include "flashinfer_bridge.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define NOMINMAX
#include <windows.h>
struct block {uint8_t scale,codes[16];};
static void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));std::exit(2);}}
static __host__ __device__ unsigned hash(unsigned i){i^=i>>16;i*=0x7feb352du;i^=i>>15;i*=0x846ca68bu;return i^(i>>16);}
// Fill BOTH representations independently: canonical split nibbles versus
// adjacent nibble values and K-group/N/4 scales. No production repacker is used.
static __global__ void fill(block *canonical,uint8_t *native,int n,int k,int kind){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x,total=size_t(384)*n*(k/32);if(i>=total)return;
    int kb=i%(k/32),column=i/(k/32)%n,expert=i/(size_t(n)*(k/32));
    block b;b.scale=uint8_t(116+(kb+expert+kind)%3);
    for(int j=0;j<16;++j){
        unsigned lo=hash(unsigned(i*32+j)+kind*73)&15,hi=hash(unsigned(i*32+j+16)+kind*73)&15;
        b.codes[j]=lo|(hi<<4);
    }
    canonical[i]=b;
    for(int j=0;j<16;++j){int a=2*j,c=a+1;
        uint8_t lo=(b.codes[a&15]>>(a>=16?4:0))&15,hi=(b.codes[c&15]>>(c>=16?4:0))&15;
        native[i*16+j]=lo|(hi<<4);
    }
    native[size_t(384)*n*k/2+((size_t(expert)*(k/128)+kb/4)*n+column)*4+kb%4]=b.scale;
}
static __global__ void inputs(float *x,float *weights,int tokens){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<size_t(tokens)*5120)x[i]=float(int(hash(unsigned(i))%513)-256)/128.f;
    if(i<size_t(tokens)*6)weights[i]=float(1+hash(unsigned(i))%7)/32.f;
}
static __global__ void middle(const float *gate,const float *up,const float *weights,float *mid,size_t count){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
    float g=fminf(gate[i],10.f),u=fmaxf(-10.f,fminf(up[i],10.f));
    mid[i]=__half2float(__float2half_rn(__fmul_rn(__fmul_rn(g/(1.f+expf(-g)),u),weights[i/2304])));
}
static __global__ void reduce_reference(const float *down,float *output,int tokens){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(tokens)*5120)return;
    int token=i/5120,c=i%5120;float result=down[size_t(token*6)*5120+c];
    for(int slot=1;slot<6;++slot)result=__fadd_rn(result,down[size_t(token*6+slot)*5120+c]);output[i]=result;
}
static __global__ void compare(const float *reference,const float *actual,size_t count,unsigned long long *errors){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
    if(!isfinite(actual[i]))atomicAdd(errors+1,1ull);
    if(reference&&__float_as_uint(reference[i])!=__float_as_uint(actual[i]))atomicAdd(errors,1ull);
}
int main(int argc,char **argv){
    if(argc!=4)return 1;int tokens=std::atoi(argv[3]);if(tokens<1||tokens>65536)return 1;
    auto library=LoadLibraryA(argv[1]),legacy=LoadLibraryA(argv[2]);if(!library||!legacy)return 2;
    auto call=reinterpret_cast<slstack_fi_native_ffn_fn>(GetProcAddress(library,"slstack_flashinfer_native_ffn"));
    auto abi=reinterpret_cast<int(*)()>(GetProcAddress(library,"slstack_flashinfer_native_ffn_abi"));
    auto size=reinterpret_cast<uint64_t(*)(int)>(GetProcAddress(library,"slstack_flashinfer_native_ffn_workspace"));
    auto old=reinterpret_cast<slstack_flashinfer_mxfp4_fn>(GetProcAddress(legacy,"slstack_flashinfer_mxfp4"));
    if(!call||!size||!old||!abi||abi()!=1)return 2;
    int rows=tokens*6;std::vector<int>routes(rows),gather(rows),bounds(385),expert(rows);
    for(int r=0;r<rows;++r){routes[r]=r;expert[r]=((r/6)*7+(r%6)*61)%384;}
    std::stable_sort(routes.begin(),routes.end(),[&](int a,int b){return expert[a]<expert[b];});
    for(int r=0;r<rows;++r){gather[r]=routes[r]/6;bounds[expert[r]+1]++;}
    for(int e=0;e<384;++e)bounds[e+1]+=bounds[e];
    int *dr,*dg,*db;float *x,*rw,*output;void *workspace;uint8_t *native[3];block *canonical;
    unsigned long long *errors;check(cudaMalloc(&errors,16));check(cudaMemset(errors,0,16));
    check(cudaMalloc(&dr,rows*4));check(cudaMalloc(&dg,rows*4));check(cudaMalloc(&db,385*4));
    check(cudaMemcpy(dr,routes.data(),rows*4,cudaMemcpyHostToDevice));check(cudaMemcpy(dg,gather.data(),rows*4,cudaMemcpyHostToDevice));check(cudaMemcpy(db,bounds.data(),385*4,cudaMemcpyHostToDevice));
    check(cudaMalloc(&x,size_t(tokens)*5120*4));check(cudaMalloc(&rw,rows*4));check(cudaMalloc(&output,size_t(tokens)*5120*4));check(cudaMalloc(&workspace,size(tokens)));
    inputs<<<unsigned((size_t(tokens)*5120+255)/256),256>>>(x,rw,tokens);
    size_t blocks=size_t(384)*2304*5120/32,weight_bytes=blocks*17;
    check(cudaMalloc(&canonical,weight_bytes));
    for(int kind=0;kind<3;++kind){check(cudaMalloc(&native[kind],weight_bytes));int n=kind==2?5120:2304,k=kind==2?2304:5120;
        fill<<<unsigned((blocks+255)/256),256>>>(canonical,native[kind],n,k,kind);}
    check(cudaDeviceSynchronize());
    check(cudaMemset(output,0xff,size_t(tokens)*5120*4));
    check(static_cast<cudaError_t>(call(native[0],native[1],native[2],x,rw,dr,db,output,tokens,10.f,workspace,size(tokens),nullptr)));
    check(cudaDeviceSynchronize());
    cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
    check(cudaEventRecord(begin));
    for(int repeat=0;repeat<5;++repeat)
        check(static_cast<cudaError_t>(call(native[0],native[1],native[2],x,rw,dr,db,output,tokens,10.f,workspace,size(tokens),nullptr)));
    check(cudaEventRecord(end));check(cudaEventSynchronize(end));float elapsed_ms=0;check(cudaEventElapsedTime(&elapsed_ms,begin,end));
    elapsed_ms/=5;check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
    float *reference=nullptr;
    if(tokens<=16384){
        // Retire native workspace before allocating the intentionally larger reference.
        check(cudaFree(workspace));workspace=nullptr;
        const auto old_bytes=ggml_dsv41_flashinfer::layout(rows,5120,2304).bytes;
        void *old_workspace;check(cudaMalloc(&old_workspace,old_bytes));
        float *g,*u,*mid,*all;check(cudaMalloc(&g,size_t(rows)*2304*4));check(cudaMalloc(&u,size_t(rows)*2304*4));check(cudaMalloc(&mid,size_t(rows)*2304*4));
        for(int kind=0;kind<2;++kind){fill<<<unsigned((blocks+255)/256),256>>>(canonical,native[kind],2304,5120,kind);
            check(static_cast<cudaError_t>(old(canonical,x,dg,dr,db,kind?u:g,rows,2304,5120,5120,2304,2304*6,old_workspace,old_bytes,nullptr)));}
        middle<<<unsigned((size_t(rows)*2304+255)/256),256>>>(g,u,rw,mid,size_t(rows)*2304);
        check(cudaFree(g));check(cudaFree(u));check(cudaMalloc(&all,size_t(rows)*5120*4));
        fill<<<unsigned((blocks+255)/256),256>>>(canonical,native[2],5120,2304,2);
        check(static_cast<cudaError_t>(old(canonical,mid,dr,dr,db,all,rows,5120,2304,2304,5120,5120*6,old_workspace,old_bytes,nullptr)));
        check(cudaMalloc(&reference,size_t(tokens)*5120*4));reduce_reference<<<unsigned((size_t(tokens)*5120+255)/256),256>>>(all,reference,tokens);
        compare<<<unsigned((size_t(tokens)*5120+255)/256),256>>>(reference,output,size_t(tokens)*5120,errors);
        check(cudaDeviceSynchronize());check(cudaFree(old_workspace));check(cudaFree(mid));check(cudaFree(all));check(cudaFree(reference));
    }else compare<<<unsigned((size_t(tokens)*5120+255)/256),256>>>(nullptr,output,size_t(tokens)*5120,errors);
    check(cudaDeviceSynchronize());unsigned long long observed[2];check(cudaMemcpy(observed,errors,16,cudaMemcpyDeviceToHost));
    std::printf("{\"status\":\"%s\",\"tokens\":%d,\"workspace_bytes\":%llu,\"compared_elements\":%llu,\"different_bits\":%llu,\"nonfinite\":%llu,\"runtime_weight_repack_bytes\":0,\"native_ffn_ms\":%.6f,\"effective_tflops\":%.6f,\"timed_repeats\":5}\n",
        observed[0]||observed[1]?"FAIL":"PASS",tokens,(unsigned long long)size(tokens),tokens<=16384?(unsigned long long)tokens*5120:0,observed[0],observed[1],elapsed_ms,3.*2*2304*5120*rows/elapsed_ms/1.e9);
    for(auto*p:native)check(cudaFree(p));check(cudaFree(canonical));if(workspace)check(cudaFree(workspace));
    check(cudaFree(x));check(cudaFree(rw));check(cudaFree(output));check(cudaFree(dr));check(cudaFree(dg));check(cudaFree(db));check(cudaFree(errors));
    return observed[0]||observed[1]?3:0;
}
