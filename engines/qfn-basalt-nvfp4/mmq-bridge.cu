// SPDX-License-Identifier: MIT
// Basalt a04edb9 MMQ NVFP4 adapter. Core MMQ/quantizer/host files are unchanged.
// GU/down GPU packing fixes raw blob16B tails; SwiGLU math follows moe_mmq.cu.
#include "mmq-bridge.hpp"
#include "vendor/basalt/src/prefill/mmq/mmq.cuh"
#include "vendor/basalt/src/prefill/mmq/quantize.cuh"
#include <atomic>
#include <algorithm>
#include <cstdio>
#include <climits>

namespace qfn_mmq {
namespace {
constexpr int E=2560,F=640,K=10,NE=512,NG=16,WINDOW=16384,RESERVED_SMS=256;
constexpr uint64_t GU=1843200,DN=921600,BLOB=2764816,TAIL=2764800,GUARD=4096;
constexpr uint64_t FIXUP=uint64_t(RESERVED_SMS)*128*128*4;
std::atomic<int> prepared_status[64]{}; // 0=not prepared, 1=ready, otherwise CUDA status+1
struct Arena {
    uint8_t * base;uint64_t used=0;
    void * take(uint64_t n) { const auto at=used;used+=(n+255)&~uint64_t(255);return base?base+at:nullptr; }
};
uint64_t qbytes(int rows,int cols) { return uint64_t(rows)*((cols+511)/512*512)/128*144+128*144; }
struct Layout {
    uint8_t * stage,* xh,* hq,* weights,* fix;
    int32_t * tok,* order,* ident,* bounds;
};
Layout layout(Arena & a,int nt) {
    const int r=nt*K;Layout s;
    s.stage=static_cast<uint8_t *>(a.take(uint64_t(r)*E*4)); // GU then Dm
    s.xh=static_cast<uint8_t *>(a.take(std::max(qbytes(r,E),uint64_t(r)*F*4))); // Xq then H
    s.hq=static_cast<uint8_t *>(a.take(qbytes(r,F)));
    s.weights=static_cast<uint8_t *>(a.take(NG*GU+GUARD));
    s.tok=static_cast<int32_t *>(a.take(uint64_t(r)*4));
    s.order=static_cast<int32_t *>(a.take(uint64_t(r)*4));
    s.ident=static_cast<int32_t *>(a.take(uint64_t(r+128)*4));
    s.bounds=static_cast<int32_t *>(a.take((NE+1)*4));
    s.fix=static_cast<uint8_t *>(a.take(FIXUP));
    return s;
}
struct FixedPool final:ggml_cuda_pool {
    void * p;size_t capacity;bool used=false;
    FixedPool(void * ptr,size_t bytes):p(ptr),capacity(bytes){}
    void * alloc(size_t n,size_t * actual) override {
        if(used||n>capacity) { std::fprintf(stderr,"QFN MMQ fixed pool overflow: request%zu capacity%zu busy%d\n",n,capacity,int(used));std::abort(); }
        used=true;*actual=capacity;return p;
    }
    void free(void * ptr,size_t bytes) override {
        if(ptr!=p||bytes!=capacity||!used) { std::fprintf(stderr,"QFN MMQ fixed pool ownership error\n");std::abort(); }
        used=false;
    }
};
struct BoundContext {
    FixedPool pool;ggml_backend_cuda_context ctx;
    BoundContext(int dev,void * p,size_t bytes):pool(p,bytes),ctx(dev) { ctx.pools[dev].reset(&pool); }
    ~BoundContext() { ctx.pools[ctx.device].release(); }
};
__global__ void identity(int32_t * ids,int rows) {
    const int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<rows+128) ids[i]=i<rows?i:0;
}
__global__ void swiglu(const float * gu,float * h,const int32_t * original,const int32_t * order,
                        const uint8_t * raw,int rows) {
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=int64_t(rows)*F) return;
    const int r=int(i/F),c=int(i%F),e=original[order[r]];
    const float4 sc=*reinterpret_cast<const float4 *>(raw+uint64_t(e)*BLOB+TAIL);
    const float g=gu[int64_t(r)*2*F+c]*sc.x,u=gu[int64_t(r)*2*F+F+c]*sc.y;
    h[i]=g/(1.0f+__expf(-g))*u*sc.z;
}
mmq_args product(const void * w,const void * xq,const int32_t * bounds,const int32_t * ids,
                 float * dst,int rows,int count,int opt,bool down) {
    const int nc=down?F:E,nr=down?E:2*F;
    return {static_cast<const char *>(w),GGML_TYPE_NVFP4,static_cast<const int *>(xq),ids,bounds,dst,
            nc,nr,rows,nc/64,rows,nr,
            NG,NG,int64_t((down?DN:GU)/36),0,0,
            1,1,0,0,0,count,opt};
}
void quantize(const float * x,const int32_t * ids,void * q,int cols,int rows,cudaStream_t cs) {
    quantize_mmq_q8_1_cuda(x,ids,q,GGML_TYPE_NVFP4,cols,cols,int64_t(rows)*cols,int64_t(rows)*cols,
                         (cols+511)/512*512,rows,1,1,cs);
}
int run_window(const float * x,const int32_t * ids,const uint8_t * raw,int nt,float * out,
                 uint8_t * workspace,cudaStream_t cs,int dev) {
    const int rows=nt*K;Arena arena{workspace};const Layout s=layout(arena,nt);
    BoundContext bound(dev,s.fix,FIXUP);
    group(ids,raw,nt,s.tok,s.order,s.bounds,cs);
    identity<<<(rows+128+255)/256,256,0,cs>>>(s.ident,rows);
    quantize(x,s.tok,s.xh,E,rows,cs);
    cudaError_t err=cudaGetLastError();
    if(err!=cudaSuccess) return int(err);
    err=cudaMemsetAsync(s.xh+qbytes(rows,E)-128*144,0,128*144,cs);
    if(err!=cudaSuccess) return int(err);
    // max_rows is the full window, never the average; opt_rows selects a tile only.
    const int opt=std::max(1,(rows+NE-1)/NE);
    for(int g=0;g<NE;g+=NG) {
        err=cudaMemcpy2DAsync(s.weights,GU,raw+uint64_t(g)*BLOB,BLOB,GU,NG,cudaMemcpyDeviceToDevice,cs);
        if(err==cudaSuccess) err=cudaMemsetAsync(s.weights+NG*GU,0,GUARD,cs);
        if(err!=cudaSuccess) return int(err);
        const auto a=product(s.weights,s.xh,s.bounds+g,s.ident,reinterpret_cast<float *>(s.stage),rows,nt,opt,false);
        mul_mat_q_case<GGML_TYPE_NVFP4>(bound.ctx,a,cs);
    }
    swiglu<<<(int64_t(rows)*F+255)/256,256,0,cs>>>(reinterpret_cast<const float *>(s.stage),
        reinterpret_cast<float *>(s.xh),ids,s.order,raw,rows);
    quantize(reinterpret_cast<const float *>(s.xh),nullptr,s.hq,F,rows,cs);
    err=cudaGetLastError();
    if(err!=cudaSuccess) return int(err);
    err=cudaMemsetAsync(s.hq+qbytes(rows,F)-128*144,0,128*144,cs);
    if(err!=cudaSuccess) return int(err);
    for(int g=0;g<NE;g+=NG) {
        err=cudaMemcpy2DAsync(s.weights,DN,raw+uint64_t(g)*BLOB+GU,BLOB,DN,NG,cudaMemcpyDeviceToDevice,cs);
        if(err==cudaSuccess) err=cudaMemsetAsync(s.weights+NG*DN,0,GUARD,cs);
        if(err!=cudaSuccess) return int(err);
        const auto a=product(s.weights,s.hq,s.bounds+g,s.ident,reinterpret_cast<float *>(s.stage),rows,nt,opt,true);
        mul_mat_q_case<GGML_TYPE_NVFP4>(bound.ctx,a,cs);
    }
    scatter(reinterpret_cast<const float *>(s.stage),s.order,out,int64_t(rows)*E,cs);
    return int(cudaGetLastError());
}
int do_prepare(cudaStream_t cs,int dev) {
    // Mode3 is bounded; a larger device must not remove modes0/1/2 capability.
    if(dev>=GGML_CUDA_MAX_DEVICES) return int(cudaErrorNotSupported);
    const auto & info=ggml_cuda_info();
    if(dev>=info.device_count||info.devices[dev].nsm<=0) return int(cudaErrorInvalidDevice);
    if(info.devices[dev].nsm>RESERVED_SMS) return int(cudaErrorNotSupported);
    bool fits=false;
    for(int j=8;j<=128;j+=8) {
        const auto config=ggml_cuda_mmq_get_config(GGML_TYPE_NVFP4,j,false);
        if(config.type!=GGML_TYPE_COUNT&&mmq_get_nbytes_shared(config)<=info.devices[dev].smpbo) fits=true;
    }
    if(!fits) return int(cudaErrorNotSupported);
    uint8_t * mem=nullptr;
    constexpr uint64_t QOFF=FIXUP+E*4,BOFF=QOFF+32768,IOFF=BOFF+256,SIZE=IOFF+1024;
    cudaError_t err=cudaMalloc(reinterpret_cast<void **>(&mem),SIZE);
    if(err!=cudaSuccess) return int(err);
    err=cudaMemsetAsync(mem,0,SIZE,cs);
    if(err==cudaSuccess) {
        BoundContext bound(dev,mem,FIXUP);
        quantize(reinterpret_cast<const float *>(mem+FIXUP),nullptr,mem+QOFF,E,1,cs);
        for(int cols=0;cols<2;++cols) for(int opt=8;opt<=128;opt+=8) {
            const auto a=product(mem+FIXUP,mem+QOFF,reinterpret_cast<const int32_t *>(mem+BOFF),
                                 reinterpret_cast<const int32_t *>(mem+IOFF),nullptr,1,128,opt,cols!=0);
            mul_mat_q_case<GGML_TYPE_NVFP4>(bound.ctx,a,cs);
        }
        swiglu<<<1,256,0,cs>>>(nullptr,nullptr,nullptr,nullptr,nullptr,0);
        identity<<<1,256,0,cs>>>(reinterpret_cast<int32_t *>(mem+IOFF),0);
        err=cudaGetLastError();
    }
    if(err==cudaSuccess) err=cudaStreamSynchronize(cs);
    const auto freed=cudaFree(mem);
    return int(err==cudaSuccess?freed:err);
}
} // namespace

uint64_t workspace_bytes(int nt) {
    if(nt<=0||nt>INT_MAX/K) return 0;
    Arena arena{nullptr};(void)layout(arena,std::min(nt,WINDOW));return arena.used;
}
void prepare(void * stream) {
    int dev=0;auto err=cudaGetDevice(&dev);
    if(err!=cudaSuccess||dev<0||dev>=64) return;
    if(prepared_status[dev].load(std::memory_order_acquire)) return;
    const int status=do_prepare(static_cast<cudaStream_t>(stream),dev);
    prepared_status[dev].store(status==0?1:status+1,std::memory_order_release);
    if(status) cudaGetLastError(); // status retained for mode3; don't poison unrelated legacy launches.
    if(status) std::fprintf(stderr,"QFN MMQ mode3 unavailable: %s; existing modes0/1/2 remain available\n",cudaGetErrorString(static_cast<cudaError_t>(status)));
}
int run(const float * x,const int32_t * ids,const uint8_t * raw,int nt,float * out,
        void * workspace,uint64_t bytes,void * stream) {
    if(bytes<workspace_bytes(nt)) return int(cudaErrorInvalidValue);
    int dev=0;auto err=cudaGetDevice(&dev);
    if(err!=cudaSuccess) return int(err);
    if(dev<0||dev>=64) return int(cudaErrorInvalidDevice);
    const int state=prepared_status[dev].load(std::memory_order_acquire);
    if(state!=1) return state?state-1:int(cudaErrorNotReady);
    for(int first=0;first<nt;first+=WINDOW) {
        const int count=std::min(WINDOW,nt-first);
        const int status=run_window(x+int64_t(first)*E,ids+int64_t(first)*K,raw,count,
            out+int64_t(first)*K*E,static_cast<uint8_t *>(workspace),static_cast<cudaStream_t>(stream),dev);
        if(status) return status;
    }
    return 0;
}
} // namespace qfn_mmq
