#pragma once
#include "ggml-dsv41-cast-vector.h"
#include <cstdlib>
#include <type_traits>

extern "C" void ggml_cuda_dsv41_cast_vector_note(uint64_t elements,bool noncontiguous);
template<typename T> struct alignas(4*sizeof(T)) dsv41_cast4 { T v[4]; };

template<typename S,typename D,bool CONTIG>
static __global__ void dsv41_cast_vector4(const S*x,D*y,int64_t n0,int64_t n1,int64_t n2,
        int64_t s1,int64_t s2,int64_t s3) {
    const int64_t col=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=n0/4)return;
    int64_t ix=col,iy=col;
    if constexpr(!CONTIG){
        const int64_t i1=blockIdx.y,i2=blockIdx.z%n2,i3=blockIdx.z/n2;
        ix=(i3*s3+i2*s2+i1*s1)/4+col;iy=((i3*n2+i2)*n1+i1)*(n0/4)+col;
    }
    const dsv41_cast4<S> v=((const dsv41_cast4<S>*)x)[ix];
    const dsv41_cast4<D> out{{ggml_cuda_cast<D>(v.v[0]),ggml_cuda_cast<D>(v.v[1]),
                             ggml_cuda_cast<D>(v.v[2]),ggml_cuda_cast<D>(v.v[3])}};
    ((dsv41_cast4<D>*)y)[iy]=out;
}

template<typename S,typename D>
static bool dsv41_try_cast_vector(const void*x,D*y,int64_t n0,int64_t n1,int64_t n2,int64_t n3,
        int64_t s1,int64_t s2,int64_t s3,cudaStream_t stream) {
    constexpr bool sf=std::is_same<S,float>::value,df=std::is_same<D,float>::value;
    constexpr bool sh=std::is_same<S,half>::value,sb=std::is_same<S,nv_bfloat16>::value;
    constexpr bool dh=std::is_same<D,half>::value,db=std::is_same<D,nv_bfloat16>::value;
    if constexpr((sf&&(dh||db))||(df&&(sh||sb))){
        const char*e=std::getenv("LLAMA_DSV41_CAST_VECTOR");if(!(e&&e[0]=='1'&&e[1]=='\0'))return false;
        constexpr ggml_type st=sf?GGML_TYPE_F32:sh?GGML_TYPE_F16:GGML_TYPE_BF16;
        constexpr ggml_type dt=df?GGML_TYPE_F32:dh?GGML_TYPE_F16:GGML_TYPE_BF16;
        if(!ggml_dsv41_cast_vector_geometry(st,dt,x,y,n0,n1,n2,n3,s1,s2,s3))return false;
        const int64_t total=n0*n1*n2*n3;
        const bool contig=s1==n0&&s2==n0*n1&&s3==n0*n1*n2;
        if(contig){
            const uint64_t blocks=(uint64_t(total/4)+255)/256;if(blocks>=UINT_MAX)return false;
            dsv41_cast_vector4<S,D,true><<<unsigned(blocks),256,0,stream>>>((const S*)x,y,total,1,1,total,total,total);
        }else{
            if(n1>65535||n2>65535/n3)return false;
            const uint64_t blocks=(uint64_t(n0/4)+255)/256;if(blocks>=UINT_MAX)return false;
            dsv41_cast_vector4<S,D,false><<<dim3(unsigned(blocks),unsigned(n1),unsigned(n2*n3)),256,0,stream>>>(
                (const S*)x,y,n0,n1,n2,s1,s2,s3);
        }
        CUDA_CHECK(cudaGetLastError());ggml_cuda_dsv41_cast_vector_note(uint64_t(total),!contig);return true;
    }
    return false;
}
