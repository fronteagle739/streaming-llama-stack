#pragma once
// OPUS22 r10: hand-written SM120 sparse MLA attention for the DSV4.1 token tile.
// Same contract as the native-layout TileLang image (q [m,64,512], kv BF16 [n,512],
// ids [m,640] with -1 = invalid, sinks [64], out F32 [m,64,512]); same arithmetic
// (BF16 MMA with F32 accumulation, scale 512^-0.5, online softmax over 32-key
// tiles, BF16 probabilities for P*V, sink added to the denominator). Differences:
// 32 heads per CTA (each key row gathered twice per query instead of four times),
// cp.async double-buffered key tiles, the query read as F32 and rounded to BF16 on
// load (no staged query copy), and split-K score accumulators. Summation order
// differs from the TileLang image, so results match to F32 rounding, not bitwise.
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

namespace dsv41_sparse_hand {
constexpr int D=512,HG=32,KT=32,TOPK=640,NTILE=TOPK/KT,THREADS=256;
constexpr int ROWB=D*2;
constexpr int Q_BYTES=HG*ROWB,K_BYTES=KT*ROWB,P_BYTES=HG*KT*2;
constexpr int STAT_FLOATS=HG*4+HG+HG;
constexpr int SMEM=Q_BYTES+2*K_BYTES+P_BYTES+STAT_FLOATS*4;
static_assert(SMEM<=101376,"sparse hand kernel exceeds the SM120 opt-in shared memory");

__device__ __forceinline__ uint32_t smem_u32(const void *p){return uint32_t(__cvta_generic_to_shared(p));}
__device__ __forceinline__ uint32_t swz(uint32_t row,uint32_t chunk){return row*ROWB+((chunk^(row&7u))<<4);}
__device__ __forceinline__ uint32_t pswz(uint32_t row,uint32_t chunk){return row*(KT*2)+((chunk^((row>>1)&3u))<<4);}
__device__ __forceinline__ void ldsm_x4(uint32_t addr,uint32_t &a,uint32_t &b,uint32_t &c,uint32_t &d){
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n":"=r"(a),"=r"(b),"=r"(c),"=r"(d):"r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t addr,uint32_t &a,uint32_t &b,uint32_t &c,uint32_t &d){
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n":"=r"(a),"=r"(b),"=r"(c),"=r"(d):"r"(addr));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4],uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ void cp_async16(uint32_t dst,const void *src,bool live){
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"::"r"(dst),"l"(src),"r"(live?16:0));
}
__device__ __forceinline__ void cp_commit(){asm volatile("cp.async.commit_group;\n"::);}
template<int N> __device__ __forceinline__ void cp_wait(){asm volatile("cp.async.wait_group %0;\n"::"n"(N));}
__device__ __forceinline__ float ex2(float x){float y;asm("ex2.approx.ftz.f32 %0, %1;":"=f"(y):"f"(x));return y;}
__device__ __forceinline__ uint32_t pack_bf16(float lo,float hi){
    __nv_bfloat162 v=__floats2bfloat162_rn(lo,hi);return *reinterpret_cast<uint32_t*>(&v);
}

// grid: m*2 CTAs (query-major, the two head groups of one query adjacent), 256 threads.
__global__ void __launch_bounds__(THREADS,1) sparse_attn_hg32(
        const int32_t *__restrict__ ids,const __nv_bfloat16 *__restrict__ kv,float *__restrict__ out,
        const float *__restrict__ q,const float *__restrict__ sinks,int m){
    extern __shared__ __align__(128) unsigned char smem[];
    unsigned char *qs=smem,*ks=smem+Q_BYTES,*ps=smem+Q_BYTES+2*K_BYTES;
    float *mx=reinterpret_cast<float*>(ps+P_BYTES);float *corr_s=mx+HG*4,*mfin=corr_s+HG;
    const int query=blockIdx.x>>1,group=blockIdx.x&1;if(query>=m)return;
    const int tid=threadIdx.x,warp=tid>>5,lane=tid&31;
    const int32_t *qids=ids+size_t(query)*TOPK;
    // key tile loader: thread -> key row tid>>3, chunks (tid&7)+8i
    const int lrow=tid>>3,lch=tid&7;
    auto issue=[&](int tile,int stage){
        const int id=__ldg(qids+tile*KT+lrow);const bool live=id>=0;
        const unsigned char *src=reinterpret_cast<const unsigned char*>(kv+size_t(live?id:0)*D);
        const uint32_t base=smem_u32(ks+stage*K_BYTES);
#pragma unroll
        for(int i=0;i<8;++i){const int c=lch+8*i;cp_async16(base+swz(lrow,c),src+c*16,live);}
    };
    issue(0,0);cp_commit();
    // query: F32 -> BF16 (round to nearest even, identical to the staged copy)
    {
        const float4 *src=reinterpret_cast<const float4*>(q+(size_t(query)*64+group*HG)*D);
#pragma unroll 4
        for(int k=0;k<16;++k){
            const int i=tid+THREADS*k,row=i>>7,f4=i&127;const float4 v=__ldg(src+i);
            uint2 packed;packed.x=pack_bf16(v.x,v.y);packed.y=pack_bf16(v.z,v.w);
            *reinterpret_cast<uint2*>(qs+swz(row,f4>>1)+(f4&1)*8)=packed;
        }
    }
    const float scale=0.044194173824159216f,L2E=1.4426950408889634f;
    // S-phase role: rb rows rb*16.., keys nt*8..
    const int rb=warp>>2,nt=warp&3,g=lane>>2,qd=lane&3;
    const int r0=rb*16+g,r1=r0+8;
    float m0=-1e30f,m1=-1e30f,l0=0.f,l1=0.f;
    // PV-phase role: output columns warp*64.. for all 32 rows
    float o[2][8][4];
#pragma unroll
    for(int a=0;a<2;++a)
#pragma unroll
        for(int b=0;b<8;++b)
#pragma unroll
            for(int c=0;c<4;++c)o[a][b][c]=0.f;
    const uint32_t qbase=smem_u32(qs),pbase=smem_u32(ps);
    for(int tile=0;tile<NTILE;++tile){
        const int stage=tile&1;
        if(tile+1<NTILE)issue(tile+1,stage^1);
        cp_commit();cp_wait<1>();__syncthreads();
        const uint32_t kbase=smem_u32(ks+stage*K_BYTES);
        // ---- scores: 16x8 per warp, K=512 split over 4 accumulators
        const int key0=nt*8+2*qd;
        const bool v0=__ldg(qids+tile*KT+key0)>=0,v1=__ldg(qids+tile*KT+key0+1)>=0;
        float acc[4][4];
#pragma unroll
        for(int s=0;s<4;++s){acc[s][0]=acc[s][1]=acc[s][2]=acc[s][3]=0.f;}
#pragma unroll
        for(int k2=0;k2<16;++k2){
            uint32_t b0,b1,b2,b3;
            ldsm_x4(kbase+swz(nt*8+(lane&7),k2*4+(lane>>3)),b0,b1,b2,b3);
            uint32_t a0,a1,a2,a3;
            ldsm_x4(qbase+swz(rb*16+(lane&15),k2*4+(lane>>4)),a0,a1,a2,a3);
            mma_bf16(acc[(2*k2)&3],a0,a1,a2,a3,b0,b1);
            ldsm_x4(qbase+swz(rb*16+(lane&15),k2*4+2+(lane>>4)),a0,a1,a2,a3);
            mma_bf16(acc[(2*k2+1)&3],a0,a1,a2,a3,b2,b3);
        }
        float s[4];
#pragma unroll
        for(int c=0;c<4;++c){
            const float t=(acc[0][c]+acc[1][c])+(acc[2][c]+acc[3][c]);
            s[c]=((c&1)?v1:v0)?t*scale:__int_as_float(0xff800000u);
        }
        float t0=fmaxf(s[0],s[1]),t1=fmaxf(s[2],s[3]);
        t0=fmaxf(t0,__shfl_xor_sync(0xffffffffu,t0,1));t0=fmaxf(t0,__shfl_xor_sync(0xffffffffu,t0,2));
        t1=fmaxf(t1,__shfl_xor_sync(0xffffffffu,t1,1));t1=fmaxf(t1,__shfl_xor_sync(0xffffffffu,t1,2));
        if(qd==0){mx[r0*4+nt]=t0;mx[r1*4+nt]=t1;}
        __syncthreads();
        {
            const float n0=fmaxf(m0,fmaxf(fmaxf(mx[r0*4],mx[r0*4+1]),fmaxf(mx[r0*4+2],mx[r0*4+3])));
            const float n1=fmaxf(m1,fmaxf(fmaxf(mx[r1*4],mx[r1*4+1]),fmaxf(mx[r1*4+2],mx[r1*4+3])));
            const float c0=ex2((m0-n0)*L2E),c1=ex2((m1-n1)*L2E);
            const float nb0=n0*L2E,nb1=n1*L2E;
            const float p0=ex2(fmaf(s[0],L2E,-nb0)),p1=ex2(fmaf(s[1],L2E,-nb0));
            const float p2=ex2(fmaf(s[2],L2E,-nb1)),p3=ex2(fmaf(s[3],L2E,-nb1));
            l0=fmaf(l0,c0,p0+p1);l1=fmaf(l1,c1,p2+p3);m0=n0;m1=n1;
            const int col=nt*8+2*qd;
            *reinterpret_cast<uint32_t*>(ps+pswz(r0,col>>3)+(col&7)*2)=pack_bf16(p0,p1);
            *reinterpret_cast<uint32_t*>(ps+pswz(r1,col>>3)+(col&7)*2)=pack_bf16(p2,p3);
            if(nt==0&&qd==0){corr_s[r0]=c0;corr_s[r1]=c1;}
        }
        __syncthreads();
        // ---- O = O*corr + P*V, columns warp*64..+64
        {
            float cr[2][2];
#pragma unroll
            for(int a=0;a<2;++a){cr[a][0]=corr_s[a*16+g];cr[a][1]=corr_s[a*16+g+8];}
#pragma unroll
            for(int a=0;a<2;++a)
#pragma unroll
                for(int b=0;b<8;++b){o[a][b][0]*=cr[a][0];o[a][b][1]*=cr[a][0];o[a][b][2]*=cr[a][1];o[a][b][3]*=cr[a][1];}
#pragma unroll
            for(int kk=0;kk<2;++kk){
                uint32_t pa[2][4];
#pragma unroll
                for(int a=0;a<2;++a)ldsm_x4(pbase+pswz(a*16+(lane&15),kk*2+(lane>>4)),pa[a][0],pa[a][1],pa[a][2],pa[a][3]);
                const int mi=lane>>3,kr=kk*16+(mi&1)*8+(lane&7);
#pragma unroll
                for(int b=0;b<8;b+=2){
                    uint32_t b0,b1,b2,b3;
                    ldsm_x4_t(kbase+swz(kr,warp*8+b+(mi>>1)),b0,b1,b2,b3);
#pragma unroll
                    for(int a=0;a<2;++a){
                        mma_bf16(o[a][b],pa[a][0],pa[a][1],pa[a][2],pa[a][3],b0,b1);
                        mma_bf16(o[a][b+1],pa[a][0],pa[a][1],pa[a][2],pa[a][3],b2,b3);
                    }
                }
            }
        }
        __syncthreads();
    }
    // ---- denominators: quad sums, then the four key-column warps, plus the sink
    l0+=__shfl_xor_sync(0xffffffffu,l0,1);l0+=__shfl_xor_sync(0xffffffffu,l0,2);
    l1+=__shfl_xor_sync(0xffffffffu,l1,1);l1+=__shfl_xor_sync(0xffffffffu,l1,2);
    if(qd==0){mx[r0*4+nt]=l0;mx[r1*4+nt]=l1;if(nt==0){mfin[r0]=m0;mfin[r1]=m1;}}
    __syncthreads();
#pragma unroll
    for(int a=0;a<2;++a)
#pragma unroll
        for(int h=0;h<2;++h){
            const int row=a*16+g+h*8,head=group*HG+row;
            const float total=((mx[row*4]+mx[row*4+1])+(mx[row*4+2]+mx[row*4+3]))+ex2((__ldg(sinks+head)-mfin[row])*L2E);
            float *dst=out+(size_t(query)*64+head)*D+warp*64+2*qd;
#pragma unroll
            for(int b=0;b<8;++b){
                float2 v;v.x=o[a][b][2*h]/total;v.y=o[a][b][2*h+1]/total;
                *reinterpret_cast<float2*>(dst+b*8)=v;
            }
        }
}
} // namespace dsv41_sparse_hand
