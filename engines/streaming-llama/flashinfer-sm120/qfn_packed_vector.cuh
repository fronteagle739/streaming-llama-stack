// SPDX-License-Identifier: MIT
// Packed-vector implementation adapted from
// engines/ds4-win/cuda/mmq/ds4_mmq_fp4_packed.cuh: ds4_fp4p_expand,
// ds4_fp4p_ldw and its fused gate/up schedule. Accumulation follows Basalt's
// native iq_kernels.cu row_dot_multi<40>/warp_sum rather than DSV's half warp:
// per16 integer dot, native lane-stride32 and XOR16/8/4/2/1 float reduction.
// QFN has UE4M3 per16 scales, role globals and top10 instead of MXFP4/top6.
// The Q8_1 quantizer is the original native-Q8 quantizer (Basalt iq_kernels.cu),
// including its finite-scale handling; no expert bytes are converted/copied.
#pragma once
#include "../../qfn-basalt-nvfp4/vendor/basalt/include/basalt/kernels/q8_1_finite.hpp"

struct QfnVectorQ8 { half2 ds; int8_t qs[32]; };
static_assert(sizeof(QfnVectorQ8)==36,"native Q8_1 block ABI");
struct QfnVector {
    QfnVectorQ8 * xq, * hq;
    float * hidden, * up;
};
QfnVector vector_layout(Arena & a,int nt) {
    QfnVector s{};
    s.xq=static_cast<QfnVectorQ8 *>(a.take(uint64_t(nt)*(E/32)*36));
    s.hidden=static_cast<float *>(a.take(uint64_t(nt)*K*F*4));
    s.up=static_cast<float *>(a.take(uint64_t(nt)*K*F*4));
    s.hq=static_cast<QfnVectorQ8 *>(a.take(uint64_t(nt)*K*(F/32)*36));
    return s;
}

__device__ __constant__ uint32_t qfn_vector_values[4]={
    0x03020100u,0x0c080604u,0xfdfeff00u,0xf4f8fafcu};
static __device__ __forceinline__ int2 qfn_vector_expand(const uint32_t q4) {
    const uint32_t *t32=qfn_vector_values;
    const uint32_t sel=0x32103210u|((q4&0x88888888u)>>1);
    const uint32_t lo0=__byte_perm(t32[0],t32[1],q4);
    const uint32_t hi0=__byte_perm(t32[2],t32[3],q4);
    const uint32_t lo1=__byte_perm(t32[0],t32[1],q4>>16);
    const uint32_t hi1=__byte_perm(t32[2],t32[3],q4>>16);
    return make_int2((int)__byte_perm(lo0,hi0,sel),(int)__byte_perm(lo1,hi1,sel>>16));
}
static __device__ __forceinline__ uint32_t qfn_vector_ldw(const uint32_t * p) {
    uint32_t r;
    asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];":"=r"(r):"l"(p));
    return r;
}
static __device__ __forceinline__ float qfn_vector_sum(float v) {
    for(int offset=16;offset>0;offset>>=1)v+=__shfl_xor_sync(0xffffffffu,v,offset);
    return v;
}
// Native UE4M3 ignores the sign bit and maps 0x7f to zero, matching
// basalt nvfp4_scale_dev; doubled E2M1 integer codes account for factor 0.5.
static __device__ __forceinline__ float qfn_vector_scale(uint8_t s) {
    s&=127;
    if(s==0||s==127)return 0.0f;
    const uint32_t exponent=s>>3,mantissa=s&7;
    if(exponent==0)return float(mantissa)*(1.0f/1024.0f);
    return __uint_as_float(((exponent+119)<<23)|(mantissa<<20));
}
static __device__ __forceinline__ float qfn_vector_term(
        const uint32_t * w,const int * q,float d,float scale) {
    // Packed adjacent nibbles replace native low/high nibble gathers only.
    // Keep Split<40>::apply's complete integer16 sum before ANY float scale,
    // including its qword order0,2,1,3 and left-associated d8*d4*integer.
    const int2 lo=qfn_vector_expand(qfn_vector_ldw(w));
    const int2 hi=qfn_vector_expand(qfn_vector_ldw(w+1));
    int sum=__dp4a(lo.x,q[0],0);
    sum=__dp4a(hi.x,q[2],sum);
    sum=__dp4a(lo.y,q[1],sum);
    sum=__dp4a(hi.y,q[3],sum);
    return d*scale*float(sum);
}
__global__ void qfn_vector_quantize(const float * x,QfnVectorQ8 * y,uint64_t count) {
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count)return; // all supported dimensions/counts are warp multiples
    const float xi=x[i];float maximum=fabsf(xi),sum=xi;
    for(int offset=16;offset>0;offset>>=1) {
        maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,offset));
        sum+=__shfl_xor_sync(0xffffffffu,sum,offset);
    }
    const float d=basalt::kernels::q8_1_finite(maximum/127.0f);
    y[i/32].qs[i%32]=basalt::kernels::q8_1_quant(xi,d,maximum);
    if(i%32==0)y[i/32].ds=basalt::kernels::q8_1_ds(d,sum);
}
// One full warp per row, matching native GU row_dot_multi's lane assignment.
template<bool FUSED> __global__ void qfn_vector_gate_up(const uint8_t * bank,const int32_t * ids,
        const QfnVectorQ8 * xq,float * hidden,float * up_out,int routes) {
    if(int(blockIdx.y)>=routes)return; // zero-route module warmup
    constexpr int NB=E/32,R=8;
    __shared__ int sx[NB*8];
    __shared__ float sd[NB];
    const int route=blockIdx.y,p=ids[route],n0=blockIdx.x*R;
    const QfnVectorQ8 * xb=xq+uint64_t(route/K)*NB;
    for(int i=threadIdx.x;i<NB*8;i+=256)sx[i]=reinterpret_cast<const int *>(xb[i>>3].qs)[i&7];
    for(int b=threadIdx.x;b<NB;b+=256)sd[b]=__low2float(xb[b].ds);
    __syncthreads();
    const int lane=threadIdx.x&31,row=n0+(threadIdx.x>>5);
    const uint8_t * expert=bank+uint64_t(p)*BLOB;
    const uint8_t * gate=expert,*up=expert+ROLE;
    const uint32_t * wg=reinterpret_cast<const uint32_t *>(gate+uint64_t(row)*(E/2));
    const uint32_t * wu=reinterpret_cast<const uint32_t *>(up+uint64_t(row)*(E/2));
    float g=0,u=0;
    for(int b=lane;b<E/16;b+=32) {
        const uint64_t si=(uint64_t(b/4)*F+row)*4+(b&3);
        const uint8_t * sg=gate+uint64_t(F)*E/2+si;
        const uint8_t * su=up+uint64_t(F)*E/2+si;
        g+=qfn_vector_term(wg+2*b,sx+(b/2)*8+(b&1)*4,sd[b/2],qfn_vector_scale(sg[0]));
        u+=qfn_vector_term(wu+2*b,sx+(b/2)*8+(b&1)*4,sd[b/2],qfn_vector_scale(su[0]));
    }
    g=qfn_vector_sum(g);u=qfn_vector_sum(u);
    if(lane==0) {
        const float * tail=reinterpret_cast<const float *>(expert+TAIL);
        g*=tail[0];u*=tail[1];
        const uint64_t i=uint64_t(route)*F+row;
        if constexpr(FUSED)hidden[i]=__fmul_rn(g/(1.0f+__expf(-g)),u);
        else { hidden[i]=g;up_out[i]=u; }
    }
}
__global__ void qfn_vector_swiglu(float * gate,const float * up,uint64_t count) {
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)gate[i]=__fmul_rn(gate[i]/(1.0f+__expf(-gate[i])),up[i]);
}
// Preserve the engine's full per-route output ABI; its existing router-weight
// multiply and ordered top10 sum stay outside this operator.
__global__ void qfn_vector_down(const uint8_t * bank,const int32_t * ids,
        const QfnVectorQ8 * hq,float * out,int routes) {
    if(int(blockIdx.y)>=routes)return;
    constexpr int NB=F/32,R=8;
    __shared__ int sx[NB*8];
    __shared__ float sd[NB];
    const int route=blockIdx.y,p=ids[route],n0=blockIdx.x*R;
    const QfnVectorQ8 * xb=hq+uint64_t(route)*NB;
    for(int i=threadIdx.x;i<NB*8;i+=256)sx[i]=reinterpret_cast<const int *>(xb[i>>3].qs)[i&7];
    for(int b=threadIdx.x;b<NB;b+=256)sd[b]=__low2float(xb[b].ds);
    __syncthreads();
    const int lane=threadIdx.x&31,row=n0+(threadIdx.x>>5);
    const uint8_t * expert=bank+uint64_t(p)*BLOB,*down=expert+2*ROLE;
    const uint32_t * w=reinterpret_cast<const uint32_t *>(down+uint64_t(row)*(F/2));
    float result=0;
    // Native's down sub8 optimization is algebraically scheduled to reproduce
    // this full-warp lane32 tree bit-for-bit (iq_kernels.cu:1163-1221).
    for(int b=lane;b<F/16;b+=32) {
        const uint64_t si=(uint64_t(b/4)*E+row)*4+(b&3);
        const uint8_t * s=down+uint64_t(E)*F/2+si;
        result+=qfn_vector_term(w+2*b,sx+(b/2)*8+(b&1)*4,sd[b/2],qfn_vector_scale(s[0]));
    }
    result=qfn_vector_sum(result);
    if(lane==0)out[uint64_t(route)*E+row]=result*reinterpret_cast<const float *>(expert+TAIL)[2];
}
