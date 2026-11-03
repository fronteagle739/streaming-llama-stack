// SPDX-License-Identifier: MIT
// Adapted from Basalt a04edb960cf3b7e5df2db2d70169ca615bcfbf21
// src/kernels/cuda/qsa_select.cu:158-269,1062-1101 (batched3xTF32 tile).
// Original bytes/hash retained in vendor/basalt and PROVENANCE.json.
// Differences: actual F16/F32 input/byte strides, weights/scales, full matrix,
// explicit prepare/admission; no Basalt causal/dead-tail/selection behavior.
// Mode1 oracle arithmetic from current ggml lightning-indexer.cu (MIT), whose
// snapshot hash is recorded in QSA_PROVENANCE.json. Retains lane+j*32/shfl_down.
// Notices: vendor/basalt/LICENSE and THIRD_PARTY_LICENSES.
#include "qfn_basalt_qsa.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>

namespace {
constexpr int D=128, H=4, QT=16, NB=32, ITER=4, QS=H*D+4, KS=D+4;
constexpr int SMEM=(QT*QS+NB*KS)*sizeof(float);
static_assert(SMEM==49920);
std::atomic<bool> ready[64]{};
struct Args {
    const float * q; const void * k; const float * w; float * out;
    int nt, np, ns;
    uint64_t qh, qt, qs, kp, ks, wt, ws, ot, os;
    float se, sh;
};
template<class T> __device__ float key(const Args & a,int s,int pool,int d) {
    const auto * row=reinterpret_cast<const T *>(static_cast<const char *>(a.k)+uint64_t(s)*a.ks+uint64_t(pool)*a.kp);
    if constexpr (sizeof(T)==2) return __half2float(row[d]);
    else return row[d];
}
__device__ float query(const Args & a,int s,int token,int head,int d) {
    return *reinterpret_cast<const float *>(reinterpret_cast<const char *>(a.q)+uint64_t(s)*a.qs+
             uint64_t(token)*a.qt+uint64_t(head)*a.qh+uint64_t(d)*4);
}
__device__ float weight(const Args & a,int s,int token,int head) {
    return *reinterpret_cast<const float *>(reinterpret_cast<const char *>(a.w)+uint64_t(s)*a.ws+
             uint64_t(token)*a.wt+uint64_t(head)*4);
}
__device__ void store(const Args & a,int s,int token,int pool,float score) {
    *reinterpret_cast<float *>(reinterpret_cast<char *>(a.out)+uint64_t(s)*a.os+
             uint64_t(token)*a.ot+uint64_t(pool)*4)=score;
}
__device__ __forceinline__ uint32_t tf32_hi(float x) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ void mma(float* c,const uint32_t* a,const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template<class T> __global__ __launch_bounds__(128) void tc_kernel(Args a) {
    extern __shared__ __align__(16) float sm[];
    float * sQ=sm, * sK=sm+QT*QS;
    const int t=threadIdx.x,lane=t&31,warp=t>>5,gid=lane>>2,tig=lane&3;
    const int q0=blockIdx.y*QT,s=blockIdx.z;
    // Scalar input loads intentionally support all actual byte-aligned outer strides.
    for(int i=t;i<QT*H*D;i+=128) {
        const int r=i/(H*D),c=i%(H*D);
        sQ[r*QS+c]=q0+r<a.nt?query(a,s,q0+r,c/D,c%D):0.0f;
    }
    __syncthreads();
    for(int it=0;it<ITER;++it) {
        const int64_t b0=(int64_t(blockIdx.x)*ITER+it)*NB;
        if(b0>=a.np) break;
        __syncthreads();
        for(int i=t;i<NB*D;i+=128) {
            const int r=i/D,c=i%D;
            sK[r*KS+c]=b0+r<a.np?key<T>(a,s,int(b0+r),c):0.0f;
        }
        __syncthreads();
        float acc[H][4]{};
        const float * kr=sK+(warp*8+gid)*KS;
#pragma unroll 4
        for(int k0=0;k0<D;k0+=8) {
            const float kx0=kr[k0+tig],kx1=kr[k0+tig+4];
            uint32_t bh[2],bl[2];
            bh[0]=tf32_hi(kx0);bh[1]=tf32_hi(kx1);
            bl[0]=tf32_hi(kx0-__uint_as_float(bh[0]));
            bl[1]=tf32_hi(kx1-__uint_as_float(bh[1]));
#pragma unroll
            for(int h=0;h<H;++h) {
                const float * qa=sQ+h*D+k0+tig;
                const float x0=qa[gid*QS],x1=qa[(gid+8)*QS],x2=qa[gid*QS+4],x3=qa[(gid+8)*QS+4];
                uint32_t ah[4],al[4];
                ah[0]=tf32_hi(x0);ah[1]=tf32_hi(x1);ah[2]=tf32_hi(x2);ah[3]=tf32_hi(x3);
                al[0]=tf32_hi(x0-__uint_as_float(ah[0]));
                al[1]=tf32_hi(x1-__uint_as_float(ah[1]));
                al[2]=tf32_hi(x2-__uint_as_float(ah[2]));
                al[3]=tf32_hi(x3-__uint_as_float(ah[3]));
                mma(acc[h],al,bh);mma(acc[h],ah,bl);mma(acc[h],ah,bh);
            }
        }
        const int64_t bc=b0+warp*8+2*tig;
#pragma unroll
        for(int half=0;half<2;++half) {
            const int qi=q0+gid+half*8;
            if(qi>=a.nt) continue;
#pragma unroll
            for(int j=0;j<2;++j) {
                const int64_t pool=bc+j;
                if(pool>=a.np) continue;
                float score=0;
#pragma unroll
                for(int h=0;h<H;++h) score+=fmaxf(acc[h][half*2+j]*a.se,0.0f)*weight(a,s,qi,h);
                store(a,s,qi,int(pool),score*a.sh);
            }
        }
    }
}
template<class T> __global__ __launch_bounds__(256) void oracle_kernel(Args a,int first) {
    if(a.nt==0) return;
    __shared__ float sq[H*D],sw[H];
    const int token=first+blockIdx.y,s=blockIdx.z,lane=threadIdx.x&31,warp=threadIdx.x>>5;
    for(int i=threadIdx.x;i<H*D;i+=256) sq[i]=query(a,s,token,i/D,i%D);
    for(int h=threadIdx.x;h<H;h+=256) sw[h]=weight(a,s,token,h);
    __syncthreads();
    const int64_t pool=int64_t(blockIdx.x)*8+warp;
    if(pool>=a.np) return;
    float kr[8];
#pragma unroll
    for(int j=0;j<8;++j) kr[j]=lane+j*32<D?key<T>(a,s,int(pool),lane+j*32):0.0f;
    float score=0;
#pragma unroll
    for(int h=0;h<H;++h) {
        float part=0;
#pragma unroll
        for(int j=0;j<8;++j) if(lane+j*32<D) part+=sq[h*D+lane+j*32]*kr[j];
        part+=__shfl_down_sync(0xffffffff,part,16);
        part+=__shfl_down_sync(0xffffffff,part,8);
        part+=__shfl_down_sync(0xffffffff,part,4);
        part+=__shfl_down_sync(0xffffffff,part,2);
        part+=__shfl_down_sync(0xffffffff,part,1);
        if(lane==0) score+=fmaxf(part*a.se,0.0f)*sw[h];
    }
    if(lane==0) store(a,s,token,int(pool),score*a.sh);
}
bool extent(uintptr_t p,uint64_t initial,uint64_t n,uint64_t stride,uint64_t & total) {
    if(n && stride>(UINT64_MAX-initial)/n) return false;
    total=initial+n*stride;
    return total<=UINTPTR_MAX-p;
}
bool spans_ok(const Args & a,int kt) {
    uint64_t qbytes,kbytes,wbytes,obytes;
    if(!extent(uintptr_t(a.q),D*4,H-1,a.qh,qbytes)||
       !extent(uintptr_t(a.q),qbytes,a.nt-1,a.qt,qbytes)||
       !extent(uintptr_t(a.q),qbytes,a.ns-1,a.qs,qbytes)||
       !extent(uintptr_t(a.k),D*(kt==0?2:4),a.np-1,a.kp,kbytes)||
       !extent(uintptr_t(a.k),kbytes,a.ns-1,a.ks,kbytes)||
       !extent(uintptr_t(a.w),H*4,a.nt-1,a.wt,wbytes)||
       !extent(uintptr_t(a.w),wbytes,a.ns-1,a.ws,wbytes)||
       !extent(uintptr_t(a.out),uint64_t(a.np)*4,a.nt-1,a.ot,obytes)||
       !extent(uintptr_t(a.out),obytes,a.ns-1,a.os,obytes)) return false;
    // Inputs may broadcast; caller owns cross-axis output non-overlap for transposed layouts.
    return (a.nt==1 || a.ot>=uint64_t(a.np)*4) && (a.ns==1 || a.os>=uint64_t(a.np)*4);
}
} // namespace

extern "C" int qfn_basalt_qsa_abi(void) { return 1; }
extern "C" int qfn_basalt_qsa_prepare(void * stream) {
    int dev=0,major=0;
    cudaError_t err=cudaGetDevice(&dev);
    if(err!=cudaSuccess) return int(err);
    if(dev<0||dev>=64) return int(cudaErrorInvalidDevice);
    if(ready[dev].load(std::memory_order_acquire)) return 0;
    err=cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,dev);
    if(err!=cudaSuccess) return int(err);
    if(major<8) return int(cudaErrorInvalidDevice);
    cudaStream_t cs=static_cast<cudaStream_t>(stream);
    cudaStreamCaptureStatus capture;
    err=cudaStreamIsCapturing(cs,&capture);
    if(err!=cudaSuccess) return int(err);
    if(capture!=cudaStreamCaptureStatusNone) return int(cudaErrorStreamCaptureUnsupported);
    err=cudaFuncSetAttribute(tc_kernel<__half>,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);
    if(err==cudaSuccess) err=cudaFuncSetAttribute(tc_kernel<float>,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);
    if(err!=cudaSuccess) return int(err);
    Args zero{};
    tc_kernel<__half><<<1,128,SMEM,cs>>>(zero);
    tc_kernel<float><<<1,128,SMEM,cs>>>(zero);
    oracle_kernel<__half><<<1,256,0,cs>>>(zero,0);
    oracle_kernel<float><<<1,256,0,cs>>>(zero,0);
    err=cudaGetLastError();
    if(err==cudaSuccess) err=cudaStreamSynchronize(cs);
    if(err==cudaSuccess) ready[dev].store(true,std::memory_order_release);
    return int(err);
}
extern "C" int qfn_basalt_qsa_run(const float * q,const void * k,const float * w,float * out,
    int nt,int np,int ns,int kt,uint64_t qh,uint64_t qt,uint64_t qs,uint64_t kp,uint64_t ks,
    uint64_t wt,uint64_t ws,uint64_t ot,uint64_t os,float se,float sh,void * stream,int mode) {
    Args a{q,k,w,out,nt,np,ns,qh,qt,qs,kp,ks,wt,ws,ot,os,se,sh};
    if(!q||!k||!w||!out||nt<=0||nt>65535*QT||np<=0||ns<=0||ns>65535||kt<0||kt>1||mode<0||mode>1||
       !std::isfinite(se)||!std::isfinite(sh)||
       ((uintptr_t(q)|uintptr_t(w)|uintptr_t(out)|qh|qt|qs|wt|ws|ot|os)&3)||
       ((uintptr_t(k)|kp|ks)&uint64_t(kt==0?1:3))||!spans_ok(a,kt)) return int(cudaErrorInvalidValue);
    int dev=0;
    cudaError_t err=cudaGetDevice(&dev);
    if(err!=cudaSuccess) return int(err);
    if(dev<0||dev>=64||!ready[dev].load(std::memory_order_acquire)) return int(cudaErrorNotReady);
    cudaStream_t cs=static_cast<cudaStream_t>(stream);
    if(mode==0) {
        const dim3 grid(unsigned((int64_t(np)+NB*ITER-1)/(NB*ITER)),unsigned((nt+QT-1)/QT),unsigned(ns));
        if(kt==0) tc_kernel<__half><<<grid,128,SMEM,cs>>>(a);
        else tc_kernel<float><<<grid,128,SMEM,cs>>>(a);
    } else {
        for(int first=0;first<nt;first+=65535) {
            const dim3 grid(unsigned((int64_t(np)+7)/8),unsigned(nt-first<65535?nt-first:65535),unsigned(ns));
            if(kt==0) oracle_kernel<__half><<<grid,256,0,cs>>>(a,first);
            else oracle_kernel<float><<<grid,256,0,cs>>>(a,first);
        }
    }
    return int(cudaGetLastError());
}
