// SPDX-License-Identifier: Apache-2.0
// Quantizer/layout adapted from Basalt a04edb9 src/prefill/fp4/moe_fp4.cu,
// originally adapted from Infernix (Ian Ranson and contributors), Apache-2.0.
// See vendor/basalt/THIRD_PARTY_LICENSES. Changed: calibrated per-expert/per-role
// global scales, separate gate/up products, original-H down quantization.
// Matrix products call the unchanged Basalt FP4 tensor-core implementation.
#include "calibrated-a4.hpp"
#include "mmq-bridge.hpp"
#include "basalt/prefill/moe_fp4.hpp"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cstdio>

namespace qfn_calibrated {
namespace {
namespace fp4=basalt::prefill::fp4;
constexpr int E=2560,F=640,K=10,NE=512,WINDOW=16384;
constexpr uint64_t BLOB=2764816,UP=921600,DOWN=1843200,TAIL=2764800;
constexpr int CODE=4096,ASTAGE=4608;
struct Arena {
    uint8_t * base;uint64_t used=0;
    void * take(uint64_t n) { const auto at=used;used+=(n+255)&~uint64_t(255);return base?base+at:nullptr; }
};
struct Layout { float * h,* ud,* gs;uint8_t * a4;int32_t * tok,* order,* bounds,* tiles; };
Layout layout(Arena & a,int nt,int banks=NE) {
    const int rows=nt*K;Layout s;
    s.h=static_cast<float *>(a.take(uint64_t(rows)*F*4)); // gate, then original H
    s.ud=static_cast<float *>(a.take(uint64_t(rows)*E*4)); // up, then down product
    s.a4=static_cast<uint8_t *>(a.take(fp4::a4_bytes(rows,E,banks))); // gate/up then H
    s.gs=static_cast<float *>(a.take(fp4::scale_count(rows,banks)*4));
    s.tok=static_cast<int32_t *>(a.take(uint64_t(rows)*4));
    s.order=static_cast<int32_t *>(a.take(uint64_t(rows)*4));
    s.bounds=static_cast<int32_t *>(a.take((banks+1)*4));
    s.tiles=static_cast<int32_t *>(a.take((banks+1)*4));
    return s;
}
#define REQUIRE_STATIC(condition,message) do { if(!(condition)) { printf("QFN calibratedA4: %s\n",message);asm volatile("trap;");return; } } while(0)
__global__ void validate_scales(const float * aux,int count) {
    const int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count) REQUIRE_STATIC(isfinite(aux[i])&&aux[i]>0,"input_scale must be finite and positive");
}
__device__ __forceinline__ int swz(int rr,int lb) { return (((lb>>4)^((rr>>1)&3))<<4)+(lb&15); }
// Same UE4M3/RNE E2M1 conversion and tile layout as pinned Basalt quantize_kernel.
// g is the raw exported input_scale = calibrated_amax/(6*448), never amax(row).
template<int NBL> __global__ void quantize_static(const float * x,const int32_t * ids,
        const int32_t * bounds,const int32_t * tiles,const float * aux,int role,
        uint8_t * a4,float * gs,int cols,int64_t ld,int rows,int banks=NE) {
    const int lane=threadIdx.x&31,r=int(blockIdx.x)*(blockDim.x>>5)+(threadIdx.x>>5);
    if(r>=rows) return;
    int lo=0,hi=banks;
    while(hi-lo>1) { const int mid=(lo+hi)>>1;if(bounds[mid]<=r)lo=mid;else hi=mid; }
    const int64_t pos=int64_t(tiles[lo])*64+(r-bounds[lo]),tile=pos/64;
    const int rr=int(pos%64),nkt=cols/128,nb=cols/16;
    const float g=aux[lo*3+role],inv_g=1.0f/g;
    if(lane==0)gs[pos]=g;
    const float * xr=x+int64_t(ids?ids[r]:r)*ld;
    uint8_t * tbase=a4+tile*(int64_t(nkt)*ASTAGE);
    #pragma unroll
    for(int i=0;i<NBL;++i) {
        const int b=lane+32*i;if(b>=nb)continue;
        const float4 * p=reinterpret_cast<const float4 *>(xr+16*b);
        float v[16],m=0;
        #pragma unroll
        for(int q=0;q<4;++q) {
            const float4 z=p[q];v[4*q]=z.x;v[4*q+1]=z.y;v[4*q+2]=z.z;v[4*q+3]=z.w;
            m=fmaxf(m,fmaxf(fmaxf(fabsf(z.x),fabsf(z.y)),fmaxf(fabsf(z.z),fabsf(z.w))));
        }
        const __nv_fp8_storage_t sb=__nv_cvt_float_to_fp8(m*inv_g*(1.0f/6.0f),__NV_SATFINITE,__NV_E4M3);
        const float sd=__half2float(__half(__nv_cvt_fp8_to_halfraw(sb,__NV_E4M3)));
        const float inv=sd>0?1.0f/(sd*g):0;
        uint32_t words[2]={0,0};
        #pragma unroll
        for(int j=0;j<8;++j) {
            unsigned short c;
            asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; cvt.u16.u8 %0, t; }"
                :"=h"(c):"f"(v[j]*inv),"f"(v[j+8]*inv));
            words[j>>2]|=uint32_t(c&0xff)<<(8*(j&3));
        }
        const int kt=b/8,lb=(b%8)*8;
        uint8_t * ktp=tbase+int64_t(kt)*ASTAGE;
        *reinterpret_cast<uint2 *>(ktp+rr*64+swz(rr,lb))=make_uint2(words[0],words[1]);
        ktp[CODE+rr*8+(b%8)]=uint8_t(sb);
    }
}
__global__ void swiglu_original_h(float * gate,const float * up,const int32_t * ids,
                                 const int32_t * order,const uint8_t * raw,int rows) {
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=int64_t(rows)*F)return;
    const int r=int(i/F),e=ids[order[r]];
    const float4 s=*reinterpret_cast<const float4 *>(raw+uint64_t(e)*BLOB+TAIL);
    const float g=gate[i]*s.x,u=up[i]*s.y;
    gate[i]=g/(1.0f+__expf(-g))*u; // NO s_down before calibrated hidden quantization
}
__global__ void scatter_down(const float * down,const int32_t * ids,const int32_t * order,
                              const uint8_t * raw,float * out,int64_t values) {
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=values)return;
    const int r=int(i/E),original=order[r],e=ids[original];
    const float sd=*reinterpret_cast<const float *>(raw+uint64_t(e)*BLOB+TAIL+8);
    out[int64_t(original)*E+i%E]=down[i]*sd;
}
void quant(const float * x,const int32_t * ids,const Layout & s,const float * aux,int role,int cols,int rows,cudaStream_t cs,int banks=NE) {
    const int grid=(rows+7)/8;
    if(cols==E)quantize_static<5><<<grid,256,0,cs>>>(x,ids,s.bounds,s.tiles,aux,role,s.a4,s.gs,cols,cols,rows,banks);
    else quantize_static<2><<<grid,256,0,cs>>>(x,ids,s.bounds,s.tiles,aux,role,s.a4,s.gs,cols,cols,rows,banks);
}
void product(const uint8_t * raw,const Layout & s,int nt,int role,void * stream,int banks=NE) {
    for(int g=0;g<banks;g+=fp4::kMaxExperts) {
        fp4::Product p;
        p.n=std::min(fp4::kMaxExperts,banks-g);
        for(int j=0;j<p.n;++j)p.blob[j]=raw+uint64_t(g+j)*BLOB;p.w_off=role==0?0:role==1?UP:DOWN;
        p.w_rows=role==2?E:F;p.w_cols=role==2?F:E;
        p.bounds=s.bounds+g;p.tile_off=s.tiles+g;p.a4=s.a4;p.gs=s.gs;
        p.dst=role==0?s.h:s.ud;p.ld_dst=role==2?E:F;p.max_rows=nt;
        fp4::run(p,stream);
    }
}

// Decode uses independent physical payload IDs and canonical logical calibration IDs.
// One compact bank per selected route keeps every payload/tail and avoids host readback,
// cache mutation, device pointer-table ABI changes, and CUDA calls from a doorbell worker.
struct Compact { uint8_t * raw;float * aux;int32_t * ids;Layout work; };
Compact compact_layout(Arena & a,int nt) {
    const int routes=nt*K;Compact c;
    c.raw=static_cast<uint8_t *>(a.take(uint64_t(routes)*BLOB));
    c.aux=static_cast<float *>(a.take(uint64_t(routes)*3*4));
    c.ids=static_cast<int32_t *>(a.take(uint64_t(routes)*4));
    c.work=layout(a,nt,routes);return c;
}
__global__ void gather_decode(const int32_t * physical,const int32_t * logical,
        const uint8_t * raw,uint32_t slots,const float * aux,Compact c,int routes) {
    const int r=int(blockIdx.y);if(r>=routes)return;
    const int p=physical[r],l=logical[r];
    REQUIRE_STATIC(p>=0&&uint32_t(p)<slots,"decode physical slot outside bank");
    REQUIRE_STATIC(l>=0&&l<NE,"decode logical expert outside calibration table");
    for(int j=r-r%K;j<r;++j) {
        REQUIRE_STATIC(logical[j]!=l,"decode logical experts must be distinct per token");
    }
    const auto * source=raw+uint64_t(p)*BLOB;
    const float * tail=reinterpret_cast<const float *>(source+TAIL);
    REQUIRE_STATIC(isfinite(tail[0])&&tail[0]>0&&isfinite(tail[1])&&tail[1]>0&&
                   isfinite(tail[2])&&tail[2]>0,"decode raw scale tail invalid");
    for(int role=0;role<3;++role)
        REQUIRE_STATIC(isfinite(aux[l*3+role])&&aux[l*3+role]>0,"decode input_scale invalid");
    if(blockIdx.x==0&&threadIdx.x==0) {
        c.ids[r]=r;c.work.tok[r]=r/K;c.work.order[r]=r;c.work.bounds[r]=r;
        if(r==routes-1)c.work.bounds[routes]=routes;
        for(int role=0;role<3;++role)c.aux[r*3+role]=aux[l*3+role];
    }
    const auto * src=reinterpret_cast<const uint4 *>(source);
    auto * dst=reinterpret_cast<uint4 *>(c.raw+uint64_t(r)*BLOB);
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<BLOB/16;
        i+=uint64_t(gridDim.x)*blockDim.x)dst[i]=src[i]; // all bytes, including the complete16-byte tail
}

} // namespace
uint64_t workspace_bytes(int nt) {
    if(nt<=0||nt>INT_MAX/K)return 0;
    Arena a{nullptr};(void)layout(a,std::min(nt,WINDOW));return a.used;
}
uint64_t decode_workspace_bytes(int nt) {
    if(nt<1||nt>4)return 0;
    Arena a{nullptr};(void)compact_layout(a,nt);return a.used;
}
int prepare(void * stream) {
    const auto cs=static_cast<cudaStream_t>(stream);
    validate_scales<<<1,256,0,cs>>>(nullptr,0);
    gather_decode<<<dim3(32,1),256,0,cs>>>(nullptr,nullptr,nullptr,0,nullptr,Compact{},0);
    quantize_static<5><<<1,256,0,cs>>>(nullptr,nullptr,nullptr,nullptr,nullptr,0,nullptr,nullptr,E,E,0);
    quantize_static<2><<<1,256,0,cs>>>(nullptr,nullptr,nullptr,nullptr,nullptr,2,nullptr,nullptr,F,F,0);
    swiglu_original_h<<<1,256,0,cs>>>(nullptr,nullptr,nullptr,nullptr,nullptr,0);
    scatter_down<<<1,256,0,cs>>>(nullptr,nullptr,nullptr,nullptr,nullptr,0);
    return int(cudaGetLastError());
}
int run(const float * x,const int32_t * ids,const uint8_t * raw,const float * aux,
        int nt,float * out,void * workspace,void * stream) {
    const auto cs=static_cast<cudaStream_t>(stream);
    validate_scales<<<6,256,0,cs>>>(aux,NE*3);
    for(int first=0;first<nt;first+=WINDOW) {
        const int n=std::min(WINDOW,nt-first),rows=n*K;
        const float * xx=x+int64_t(first)*E;const int32_t * ii=ids+int64_t(first)*K;
        Arena a{static_cast<uint8_t *>(workspace)};const Layout s=layout(a,n);
        qfn_mmq::group(ii,raw,n,s.tok,s.order,s.bounds,stream);
        fp4::plan(s.bounds,NE,s.tiles,stream);
        quant(xx,s.tok,s,aux,0,E,rows,cs);product(raw,s,n,0,stream);
        quant(xx,s.tok,s,aux,1,E,rows,cs);product(raw,s,n,1,stream);
        swiglu_original_h<<<(int64_t(rows)*F+255)/256,256,0,cs>>>(s.h,s.ud,ii,s.order,raw,rows);
        quant(s.h,nullptr,s,aux,2,F,rows,cs);product(raw,s,n,2,stream);
        scatter_down<<<(int64_t(rows)*E+255)/256,256,0,cs>>>(s.ud,ii,s.order,raw,out+int64_t(first)*K*E,int64_t(rows)*E);
        const auto err=cudaGetLastError();if(err!=cudaSuccess)return int(err);
    }
    return 0;
}

int run_decode(const float * x,const int32_t * physical_ids,const int32_t * logical_ids,
        const uint8_t * raw,uint32_t slots,const float * aux,int nt,
        float * out,void * workspace,void * stream) {
    if(!decode_workspace_bytes(nt))return int(cudaErrorInvalidValue);
    const auto cs=static_cast<cudaStream_t>(stream);const int rows=nt*K;
    Arena a{static_cast<uint8_t *>(workspace)};const Compact c=compact_layout(a,nt);
    validate_scales<<<6,256,0,cs>>>(aux,NE*3); // validate the complete declared field, including currently unselected experts
    gather_decode<<<dim3(32,rows),256,0,cs>>>(physical_ids,logical_ids,raw,slots,aux,c,rows);
    fp4::plan(c.work.bounds,rows,c.work.tiles,stream);
    quant(x,c.work.tok,c.work,c.aux,0,E,rows,cs,rows);product(c.raw,c.work,1,0,stream,rows);
    quant(x,c.work.tok,c.work,c.aux,1,E,rows,cs,rows);product(c.raw,c.work,1,1,stream,rows);
    swiglu_original_h<<<(int64_t(rows)*F+255)/256,256,0,cs>>>(c.work.h,c.work.ud,c.ids,c.work.order,c.raw,rows);
    quant(c.work.h,nullptr,c.work,c.aux,2,F,rows,cs,rows);product(c.raw,c.work,1,2,stream,rows);
    scatter_down<<<(int64_t(rows)*E+255)/256,256,0,cs>>>(c.work.ud,c.ids,c.work.order,c.raw,out,int64_t(rows)*E);
    return int(cudaGetLastError());
}

} // namespace qfn_calibrated
