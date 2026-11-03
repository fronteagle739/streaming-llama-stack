// SPDX-License-Identifier: MIT
// DSV grouped native NVFP4 adapter; calibrated quantizer arithmetic follows
// qfn-basalt-nvfp4/calibrated-a4.cu (Basalt/Infernix, Apache-2.0).
#define QFN_NATIVE_NVFP4_BUILD
#include "qfn_native_nvfp4_bridge.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <string>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "../../qfn-basalt-nvfp4/qfn_basalt_nvfp4.h"
#define CUTE_DSL_CUDA_ERROR_CHECK(err) do { auto e=(err);if(e!=cudaSuccess){ \
    std::fprintf(stderr,"QFN native NVFP4 CUDA failure: %s\n",cudaGetErrorString(e));std::abort();} }while(0)
#include "fi_nvfp4_640_2560.h"
#include "fi_nvfp4_2560_640.h"

namespace {
constexpr int E=2560,F=640,K=10,NE=512,WINDOW=16384;
constexpr uint64_t BLOB=2764816,ROLE=921600,TAIL=2764800;
std::atomic<bool> prepared[64]{};
std::atomic<bool> shared_gu_reported{};
fi_nvfp4_640_2560_Kernel_Module_t gu_module[64]{};
fi_nvfp4_2560_640_Kernel_Module_t down_module[64]{};
// A process selects the short mode6 middle-stage schedule before admission.
// Keep the original two-kernel arithmetic available for measured A/B.
int fused_mid_policy(){
    static const int value=[](){
        char option[8]{};SetLastError(ERROR_SUCCESS);
        const DWORD n=GetEnvironmentVariableA("QFN_NATIVE_NVFP4_FUSED_MID",option,sizeof(option));
        if(!n&&GetLastError()==ERROR_ENVVAR_NOT_FOUND)return 0; // measured: the short schedule is slower in prefill, decode unchanged
        if(n==1&&(option[0]=='0'||option[0]=='1'))return int(option[0]-'0');
        std::fprintf(stderr,"QFN_NATIVE_NVFP4_FUSED_MID must be exactly 0 or 1 before admission\n");return -1;
    }();return value;
}
int group_parallel_policy(){
    static const int value=[](){
        char option[8]{};SetLastError(ERROR_SUCCESS);
        const DWORD n=GetEnvironmentVariableA("QFN_NATIVE_NVFP4_GROUP_PARALLEL",option,sizeof(option));
        if(!n&&GetLastError()==ERROR_ENVVAR_NOT_FOUND)return 0;
        if(n==1&&(option[0]=='0'||option[0]=='1'))return int(option[0]-'0');
        std::fprintf(stderr,"QFN_NATIVE_NVFP4_GROUP_PARALLEL must be exactly 0 or 1 before admission\n");return -1;
    }();return value;
}
int share_gu_input_policy(){
    static const int value=[](){
        char option[8]{};SetLastError(ERROR_SUCCESS);
        const DWORD n=GetEnvironmentVariableA("QFN_NATIVE_NVFP4_SHARE_GU_INPUT",option,sizeof(option));
        if(!n&&GetLastError()==ERROR_ENVVAR_NOT_FOUND)return 0;
        if(n==1&&(option[0]=='0'||option[0]=='1'))return int(option[0]-'0');
        std::fprintf(stderr,"QFN_NATIVE_NVFP4_SHARE_GU_INPUT must be exactly 0 or 1 before admission\n");return -1;
    }();return value;
}
int decode_residual_policy(){
    static const int value=[](){
        char option[8]{};SetLastError(ERROR_SUCCESS);
        const DWORD n=GetEnvironmentVariableA("QFN_NATIVE_NVFP4_DECODE_RESIDUAL",option,sizeof(option));
        if(!n&&GetLastError()==ERROR_ENVVAR_NOT_FOUND)return 0;
        if(n==1&&(option[0]=='0'||option[0]=='1'))return int(option[0]-'0');
        std::fprintf(stderr,"QFN_NATIVE_NVFP4_DECODE_RESIDUAL must be exactly 0 or 1 before admission\n");return -1;
    }();return value;
}
int decode_vector_policy(){
    static const int value=[](){
        char option[8]{};SetLastError(ERROR_SUCCESS);
        const DWORD n=GetEnvironmentVariableA("QFN_NATIVE_NVFP4_DECODE_VECTOR",option,sizeof(option));
        if(!n&&GetLastError()==ERROR_ENVVAR_NOT_FOUND)return 0;
        if(n==1&&(option[0]=='0'||option[0]=='1'))return int(option[0]-'0');
        std::fprintf(stderr,"QFN_NATIVE_NVFP4_DECODE_VECTOR must be exactly 0 or 1 before admission\n");return -1;
    }();return value;
}

// Resolve only the stable adjacent original DLL; never alter its file/ABI.
struct Original {
    HMODULE module=nullptr;
    decltype(&qfn_basalt_nvfp4_workspace_bytes) bytes=nullptr;
    decltype(&qfn_basalt_nvfp4_prepare) prepare=nullptr;
    decltype(&qfn_basalt_nvfp4_run) run=nullptr;
    decltype(&qfn_basalt_nvfp4_run_calibrated) calibrated=nullptr;
    decltype(&qfn_basalt_nvfp4_run_calibrated_decode) decode=nullptr;
    Original() {
        HMODULE self=nullptr;wchar_t path[32768];
        if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
               reinterpret_cast<LPCWSTR>(&qfn_native_nvfp4_abi),&self))return;
        DWORD n=GetModuleFileNameW(self,path,32768);if(!n||n>=32768)return;
        std::wstring name(path,n);const auto split=name.find_last_of(L"\\/");
        if(split==std::wstring::npos)return;
        name.resize(split+1);name+=L"qfn-basalt-nvfp4.dll";
        module=LoadLibraryW(name.c_str());if(!module)return;
        bytes=reinterpret_cast<decltype(bytes)>(GetProcAddress(module,"qfn_basalt_nvfp4_workspace_bytes"));
        prepare=reinterpret_cast<decltype(prepare)>(GetProcAddress(module,"qfn_basalt_nvfp4_prepare"));
        run=reinterpret_cast<decltype(run)>(GetProcAddress(module,"qfn_basalt_nvfp4_run"));
        calibrated=reinterpret_cast<decltype(calibrated)>(GetProcAddress(module,"qfn_basalt_nvfp4_run_calibrated"));
        decode=reinterpret_cast<decltype(decode)>(GetProcAddress(module,"qfn_basalt_nvfp4_run_calibrated_decode"));
    }
    bool valid() const {return bytes&&prepare&&run&&calibrated&&decode;}
};
Original & original(){static Original o;return o;}
struct Arena {
    uint8_t * base;uint64_t used=0;
    void * take(uint64_t n){const uint64_t at=used;used+=(n+255)&~uint64_t(255);return base?base+at:nullptr;}
};
#include "qfn_packed_vector.cuh"
struct Native {
    uint8_t * a,* sa;float * gate,* product;
    uint8_t * low_a,* low_sa;float * low_product;
    int32_t * order,* bounds,* scale_row;
    int sf;
};
Native native_layout(Arena & a,int nt,bool residual=false){
    const int rows=nt*K;Native s{};s.sf=(rows+NE*3+127)/128*128;
    s.a=static_cast<uint8_t *>(a.take(uint64_t(rows)*E/2));
    s.sa=static_cast<uint8_t *>(a.take(uint64_t(s.sf)*E/16));
    s.gate=static_cast<float *>(a.take(uint64_t(rows)*F*4));
    s.product=static_cast<float *>(a.take(uint64_t(rows)*E*4));
    s.order=static_cast<int32_t *>(a.take(uint64_t(rows)*4));
    s.bounds=static_cast<int32_t *>(a.take((NE+1)*4));
    s.scale_row=static_cast<int32_t *>(a.take(uint64_t(rows)*4));
    if(residual){
        s.low_a=static_cast<uint8_t *>(a.take(uint64_t(rows)*E/2));
        s.low_sa=static_cast<uint8_t *>(a.take(uint64_t(s.sf)*E/16));
        s.low_product=static_cast<float *>(a.take(uint64_t(rows)*E*4));
    }
    return s;
}
// Short target-decode calls use one packed group per route. This keeps the
// existing 512-group AOT descriptors while permitting arbitrary cache slots:
// only groups [0,nt*K) have rows, and their payloads are copied losslessly.
// Repeated physical placeholders retain separate route/logical identities.
struct CompactNative {
    uint8_t * bank;int32_t * physical,* logical;
};
CompactNative compact_native_layout(Arena & a,int nt){
    const uint64_t rows=uint64_t(nt)*K;CompactNative s{};
    s.bank=static_cast<uint8_t *>(a.take(rows*BLOB));
    s.physical=static_cast<int32_t *>(a.take(rows*4));
    s.logical=static_cast<int32_t *>(a.take(rows*4));return s;
}
bool compact(int nt,int mode){return nt<=4&&(mode==0||mode==2||mode==5);}
struct Inverse {
    uint8_t * raw;int32_t * ids,* logical,* calibration_map;
    float * mapped_calibration;void * work;uint64_t bytes;uint32_t slots;
};
Inverse inverse_layout(Arena & a,int nt,uint32_t slots,int mode){
    Inverse s{};const bool small=compact(nt,mode);s.slots=small?uint32_t(nt*K):slots;
    s.raw=static_cast<uint8_t *>(a.take(uint64_t(s.slots)*BLOB));
    if(small){s.ids=static_cast<int32_t *>(a.take(uint64_t(nt)*K*4));s.logical=static_cast<int32_t *>(a.take(uint64_t(nt)*K*4));}
    if(mode==4){
        s.calibration_map=static_cast<int32_t *>(a.take(NE*4));
        s.mapped_calibration=static_cast<float *>(a.take(NE*3*4));
    }
    s.bytes=original().bytes(nt,K,s.slots,mode);s.work=a.take(s.bytes);return s;
}
#define NEED_DEVICE(condition,message) do {if(!(condition)){printf("QFN native NVFP4: %s\n",message);asm volatile("trap;");return;}}while(0)
__global__ void validate_calibration(const float * calibration,int count,bool shared_gu){
    const int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count){
        NEED_DEVICE(isfinite(calibration[i])&&calibration[i]>0,"calibration must contain all512x3 finitepositive globals");
        if(shared_gu&&i%3==0)
            NEED_DEVICE(__float_as_uint(calibration[i])==__float_as_uint(calibration[i+1]),
                "QFN_NATIVE_NVFP4_SHARE_GU_INPUT=1 requires bit-identical gate/up calibration for every expert");
    }
}
__global__ void validate_routes(const int32_t * ids,const int32_t * logical,const uint8_t * bank,
                              uint32_t slots,int rows,bool unique,bool calibrated){
    const int r=int(blockIdx.x)*blockDim.x+threadIdx.x;if(r>=rows)return;
    const int p=ids[r],l=logical?logical[r]:p;
    NEED_DEVICE(p>=0&&uint32_t(p)<slots,"physical slot outside packed bank");
    if(calibrated)NEED_DEVICE(l>=0&&l<NE,"logical calibration expert outside512 bank");
    if(unique)for(int j=r-r%K;j<r;++j){
        NEED_DEVICE(ids[j]!=p,"grouped arithmetic needs distinct physical experts per token");
        if(calibrated)NEED_DEVICE((logical?logical[j]:ids[j])!=l,"calibrated arithmetic needs distinct logical experts per token");
    }
    const float * tail=reinterpret_cast<const float *>(bank+uint64_t(p)*BLOB+TAIL);
    NEED_DEVICE(isfinite(tail[0])&&tail[0]>0&&isfinite(tail[1])&&tail[1]>0&&isfinite(tail[2])&&tail[2]>0,"packed FP32 scale tail invalid");
}
__global__ void compact_packed_routes(const uint8_t * bank,const int32_t * physical,
        const int32_t * logical,uint32_t slots,int rows,uint8_t * compact,
        int32_t * compact_ids,int32_t * canonical){
    const int r=int(blockIdx.y);if(r>=rows)return;
    const int p=physical[r],l=logical?logical[r]:p;
    NEED_DEVICE(p>=0&&uint32_t(p)<slots,"compact source physical slot outside packed bank");
    NEED_DEVICE(l>=0&&l<NE,"compact logical calibration expert outside512 bank");
    const auto * source=reinterpret_cast<const uint4 *>(bank+uint64_t(p)*BLOB);
    auto * target=reinterpret_cast<uint4 *>(compact+uint64_t(r)*BLOB);
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<BLOB/16;i+=uint64_t(gridDim.x)*blockDim.x)
        target[i]=source[i];
    if(blockIdx.x==0&&threadIdx.x==0){compact_ids[r]=r;canonical[r]=l;}
}
__global__ void compact_route_plan(int rows,int32_t * order,int32_t * bounds,int32_t * scale_row){
    if(!rows)return;
    const int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<=NE)bounds[i]=min(i,rows);
    if(i<rows){order[i]=i;scale_row[i]=4*i;}
}

// Inverse of the lossless packing operation: recover each original36-byte
// block (four scale bytes, then four j/j+8 nibble sub-blocks). No requantization.
__global__ void inverse_bank(const uint8_t * bank,uint8_t * raw,const int32_t * ids,
                            const int32_t * logical,int32_t * gathered,int32_t * canonical,
                            uint32_t slots,int banks,bool selected){
    if(int(blockIdx.y)>=banks)return;
    for(int64_t index=blockIdx.y;index<banks;index+=gridDim.y){
    const int dst=int(index);
    const int role=int(blockIdx.z),p=selected?ids[dst]:dst;
    NEED_DEVICE(p>=0&&uint32_t(p)<slots,"inverse source physical slot outside packed bank");
    const int n=role==2?E:F,k=role==2?F:E,nkb=k/64;
    const uint8_t * source=bank+uint64_t(p)*BLOB+uint64_t(role)*ROLE;
    uint8_t * target=raw+uint64_t(dst)*BLOB+uint64_t(role)*ROLE;
    for(int b=int(blockIdx.x)*blockDim.x+threadIdx.x;b<n*nkb;b+=int(gridDim.x)*blockDim.x){
        const int row=b/nkb,kb=b%nkb;uint8_t * block=target+uint64_t(b)*36;
        #pragma unroll
        for(int s=0;s<4;++s){
            block[s]=source[uint64_t(n)*k/2+(uint64_t(kb)*n+row)*4+s];
            #pragma unroll
            for(int j=0;j<8;++j){
                const int a=kb*64+s*16+j,bv=a+8;
                const uint8_t q0=source[uint64_t(row)*(k/2)+a/2];
                const uint8_t q1=source[uint64_t(row)*(k/2)+bv/2];
                block[4+s*8+j]=uint8_t(((q0>>((a&1)*4))&15)|(((q1>>((bv&1)*4))&15)<<4));
            }
        }
    }
    if(blockIdx.x==0&&threadIdx.x==0&&role==0){
        *reinterpret_cast<uint4 *>(raw+uint64_t(dst)*BLOB+TAIL)=
            *reinterpret_cast<const uint4 *>(bank+uint64_t(p)*BLOB+TAIL);
        if(selected){gathered[dst]=dst;canonical[dst]=logical?logical[dst]:p;}
    }
    }
}

// The compatibility mode4 ABI indexes calibration by physical bank ID. Retain
// optional canonical IDs by materializing the equivalent calibration table,
// checking that a physical payload is never assigned conflicting identities.
__global__ void map_calibration(const int32_t * physical,const int32_t * logical,
                               int32_t * map,int rows){
    const int r=int(blockIdx.x)*blockDim.x+threadIdx.x;if(r>=rows)return;
    const int p=physical[r],l=logical[r];
    NEED_DEVICE(p>=0&&p<NE&&l>=0&&l<NE,"mode4 physical/logical calibration ID outside512 bank");
    const int previous=atomicCAS(map+p,-1,l);
    NEED_DEVICE(previous==-1||previous==l,"one physical payload has conflicting logical calibration IDs");
}
__global__ void gather_calibration(const float * calibration,const int32_t * map,float * mapped,int count){
    const int i=int(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
    const int expert=i/3,l=map[expert];mapped[i]=calibration[(l<0?expert:l)*3+i%3];
}

// Existing ggml/Basalt expert grouping: stable token/route order inside each
// expert, one warp per expert, with unique routes already validated above.
__global__ void group_bank(const int32_t * ids,int nt,int32_t * order,int32_t * bounds){
    if(!nt)return;
    extern __shared__ uint32_t routes[];
    const int expert=int(blockIdx.x),lane=threadIdx.x;int lower=0,count=0;
    for(int t=0;t<nt;++t){
        const int id=lane<K?ids[t*K+lane]:INT_MAX;lower+=id<expert;
        const bool mine=id==expert;if(mine)routes[count]=uint32_t(t*K+lane);
        count+=__any_sync(0xffffffffu,mine);
    }
    for(int d=16;d;d/=2)lower+=__shfl_down_sync(0xffffffffu,lower,d);
    lower=__shfl_sync(0xffffffffu,lower,0);__syncwarp();
    for(int i=lane;i<count;i+=32)order[lower+i]=int32_t(routes[i]);
    if(lane==0){bounds[expert]=lower;if(expert==NE-1)bounds[NE]=lower+count;}
}
__global__ void padded_scale_rows(const int32_t * bounds,int32_t * scale_row,int experts){
    if(int(blockIdx.x)>=experts)return;
    const int expert=int(blockIdx.x),first=bounds[expert],end=bounds[expert+1];
    const int aligned=(first+expert*3)/4*4;
    for(int r=first+int(threadIdx.x);r<end;r+=int(blockDim.x))scale_row[r]=aligned+r-first;
}

// Same calibrated UE4M3/RNE arithmetic as the original adapter, with adjacent
// codes and DSV scale planes instead of Basalt's swizzled64-row activation tiles.
template<int COLS> __global__ void quantize_calibrated(const float * x,const int32_t * physical,
        const int32_t * logical,const int32_t * order,const int32_t * scale_row,
        const float * calibration,int role,uint8_t * a,uint8_t * sa,int rows,int sf){
    const int lane=threadIdx.x&31,r=int(blockIdx.x)*(blockDim.x/32)+threadIdx.x/32;
    if(r>=rows)return;
    const int route=order[r],p=physical[route],l=logical?logical[route]:p;
    const float g=calibration[l*3+role],inv_g=1.0f/g;
    const float * input=x+uint64_t(role==2?r:route/K)*COLS;
    #pragma unroll
    for(int b=lane;b<COLS/16;b+=32){
        float v[16],maximum=0;
        const float4 * src=reinterpret_cast<const float4 *>(input+16*b);
        #pragma unroll
        for(int j=0;j<4;++j){const float4 q=src[j];
            v[j*4]=q.x;v[j*4+1]=q.y;v[j*4+2]=q.z;v[j*4+3]=q.w;
            maximum=fmaxf(maximum,fmaxf(fmaxf(fabsf(q.x),fabsf(q.y)),fmaxf(fabsf(q.z),fabsf(q.w))));
        }
        const __nv_fp8_storage_t sb=__nv_cvt_float_to_fp8(maximum*inv_g*(1.0f/6.0f),__NV_SATFINITE,__NV_E4M3);
        const float sd=__half2float(__half(__nv_cvt_fp8_to_halfraw(sb,__NV_E4M3)));
        const float inv=sd>0?1.0f/(sd*g):0;
        #pragma unroll
        for(int j=0;j<8;++j){unsigned short q;
            asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; cvt.u16.u8 %0, t; }"
                :"=h"(q):"f"(v[j*2]*inv),"f"(v[j*2+1]*inv));
            a[uint64_t(r)*(COLS/2)+b*8+j]=uint8_t(q);
        }
        sa[(uint64_t(b/4)*sf+scale_row[r])*4+b%4]=uint8_t(sb);
    }
}
__global__ void swiglu_original_h(float * gate,const float * up,const int32_t * physical,
        const int32_t * logical,const int32_t * order,const uint8_t * bank,const float * calibration,int rows){
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=uint64_t(rows)*F)return;
    const int route=order[i/F],p=physical[route],l=logical?logical[route]:p;
    const float * tail=reinterpret_cast<const float *>(bank+uint64_t(p)*BLOB+TAIL);
    const float g=(gate[i]*calibration[l*3])*tail[0],u=(up[i]*calibration[l*3+1])*tail[1];
    gate[i]=g/(1.0f+__expf(-g))*u;
}

// Contiguous token intervals and an exclusive match-count prefix preserve the
// original token/route order. All arithmetic here is bounded integer indexing.
// The original warp kernel remains available; this candidate uses fixed 2KiB
// shared storage even at the full 16384-token window.
__global__ void group_bank_parallel(const int32_t * ids,int nt,int32_t * order,int32_t * bounds){
    if(!nt)return;
    __shared__ int lower[256],matches[256];
    const int expert=int(blockIdx.x),tid=int(threadIdx.x);
    const int first=nt*tid/256,last=nt*(tid+1)/256;
    int below=0,count=0;
    for(int t=first;t<last;++t){
        for(int k=0;k<K;++k){const int id=ids[t*K+k];below+=id<expert;count+=id==expert;}
    }
    lower[tid]=below;matches[tid]=count;__syncthreads();
    for(int stride=128;stride;stride/=2){
        if(tid<stride)lower[tid]+=lower[tid+stride];
        __syncthreads();
    }
    for(int offset=1;offset<256;offset*=2){
        const int previous=tid>=offset?matches[tid-offset]:0;
        __syncthreads();
        matches[tid]+=previous;__syncthreads();
    }
    int destination=lower[0]+(tid?matches[tid-1]:0);
    if(tid==0){bounds[expert]=lower[0];if(expert==NE-1)bounds[NE]=lower[0]+matches[255];}
    for(int t=first;t<last;++t){
        for(int k=0;k<K;++k)if(ids[t*K+k]==expert)order[destination++]=t*K+k;
    }
}
// DSV's mid_pack schedule, retaining QFN's original-H calibration and all
// independent NVIDIA globals. Each lane owns one complete16-value scale block;
// the arithmetic/UE4M3 and E2M1 RNE conversions match the unfused kernels.
__global__ void swiglu_pack_calibrated(const float * gate,const float * up,const int32_t * physical,
        const int32_t * logical,const int32_t * order,const int32_t * scale_row,
        const uint8_t * bank,const float * calibration,uint8_t * a,uint8_t * sa,int rows,int sf){
    const int lane=threadIdx.x&31,r=int(blockIdx.x)*(blockDim.x/32)+threadIdx.x/32;
    if(r>=rows)return;
    const int route=order[r],p=physical[route],l=logical?logical[route]:p;
    const float * tail=reinterpret_cast<const float *>(bank+uint64_t(p)*BLOB+TAIL);
    const float sg=calibration[l*3],su=calibration[l*3+1],sd_global=calibration[l*3+2];
    const float inv_global=1.0f/sd_global;
    for(int b=lane;b<F/16;b+=32){
        float v[16],maximum=0;
        #pragma unroll
        for(int j=0;j<16;++j){
            const uint64_t i=uint64_t(r)*F+b*16+j;
            const float g=(gate[i]*sg)*tail[0],u=(up[i]*su)*tail[1];
            v[j]=g/(1.0f+__expf(-g))*u;maximum=fmaxf(maximum,fabsf(v[j]));
        }
        const __nv_fp8_storage_t sb=__nv_cvt_float_to_fp8(maximum*inv_global*(1.0f/6.0f),__NV_SATFINITE,__NV_E4M3);
        const float sd=__half2float(__half(__nv_cvt_fp8_to_halfraw(sb,__NV_E4M3)));
        const float inv=sd>0?1.0f/(sd*sd_global):0;
        #pragma unroll
        for(int j=0;j<8;++j){unsigned short q;
            asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; cvt.u16.u8 %0, t; }"
                :"=h"(q):"f"(v[j*2]*inv),"f"(v[j*2+1]*inv));
            a[uint64_t(r)*(F/2)+b*8+j]=uint8_t(q);
        }
        sa[(uint64_t(b/4)*sf+scale_row[r])*4+b%4]=uint8_t(sb);
    }
}
__global__ void scatter_down(const float * product,const int32_t * physical,const int32_t * logical,
        const int32_t * order,const uint8_t * bank,const float * calibration,float * out,uint64_t values){
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=values)return;
    const int route=order[i/E],p=physical[route],l=logical?logical[route]:p;
    const float down=*reinterpret_cast<const float *>(bank+uint64_t(p)*BLOB+TAIL+8);
    out[uint64_t(route)*E+i%E]=(product[i]*calibration[l*3+2])*down;
}
// Experimental second activation component. The high component is unchanged;
// quantize its F32 reconstruction error with the SAME independent role global.
// Weight values/scales and all original calibration bytes remain untouched.
// FUSED_H reconstructs original H without changing the fused high-pack schedule.
template<int COLS,bool FUSED_H> __global__ void quantize_residual(const float * x,
        const float * gate,const float * up,const int32_t * physical,const int32_t * logical,
        const int32_t * order,const int32_t * scale_row,const uint8_t * bank,
        const float * calibration,int role,const uint8_t * high,const uint8_t * high_sa,
        uint8_t * low,uint8_t * low_sa,int rows,int sf){
    const int lane=threadIdx.x&31,r=int(blockIdx.x)*(blockDim.x/32)+threadIdx.x/32;
    if(r>=rows)return;
    const int route=order[r],p=physical[route],l=logical?logical[route]:p;
    const float global=calibration[l*3+role],inv_global=1.0f/global;
    const float * input=x+uint64_t(role==2?r:route/K)*COLS;
    const float * tail=reinterpret_cast<const float *>(bank+uint64_t(p)*BLOB+TAIL);
    const float levels[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
    for(int b=lane;b<COLS/16;b+=32){
        const uint64_t si=(uint64_t(b/4)*sf+scale_row[r])*4+b%4;
        const float high_scale=__half2float(__half(__nv_cvt_fp8_to_halfraw(high_sa[si],__NV_E4M3)));
        float values[16],maximum=0;
        #pragma unroll
        for(int j=0;j<16;++j){
            float value;
            if constexpr(FUSED_H){
                const uint64_t i=uint64_t(r)*F+b*16+j;
                const float g=(gate[i]*calibration[l*3])*tail[0];
                const float u=(up[i]*calibration[l*3+1])*tail[1];
                value=g/(1.0f+__expf(-g))*u;
            }else value=input[b*16+j];
            const uint8_t packed=high[uint64_t(r)*(COLS/2)+b*8+j/2];
            const int code=(packed>>((j&1)*4))&15;
            const float decoded=(code&8)?-levels[code&7]:levels[code&7];
            values[j]=value-((decoded*high_scale)*global);
            NEED_DEVICE(isfinite(values[j]),"activation residual must remain finite");
            maximum=fmaxf(maximum,fabsf(values[j]));
        }
        const __nv_fp8_storage_t sb=__nv_cvt_float_to_fp8(maximum*inv_global*(1.0f/6.0f),__NV_SATFINITE,__NV_E4M3);
        const float scale=__half2float(__half(__nv_cvt_fp8_to_halfraw(sb,__NV_E4M3)));
        const float inv=scale>0?1.0f/(scale*global):0;
        #pragma unroll
        for(int j=0;j<8;++j){unsigned short q;
            asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; cvt.u16.u8 %0, t; }"
                :"=h"(q):"f"(values[j*2]*inv),"f"(values[j*2+1]*inv));
            low[uint64_t(r)*(COLS/2)+b*8+j]=uint8_t(q);
        }
        low_sa[si]=uint8_t(sb);
    }
}
__global__ void add_residual_product(float * high,const float * low,uint64_t count){
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)high[i]+=low[i];
}
int gemm(const uint8_t * bank,Native & s,int rows,int role,int dev,cudaStream_t stream){
    uint8_t * w=const_cast<uint8_t *>(bank)+uint64_t(role)*ROLE;
    const int n=role==2?E:F,k=role==2?F:E;
    void * sb=w+uint64_t(n)*k/2;float * out=role==0?s.gate:s.product;
    if(role!=2){
        fi_nvfp4_640_2560_Tensor_gA_t aa{s.a,{rows}};
        fi_nvfp4_640_2560_Tensor_gB_t bb{w};
        fi_nvfp4_640_2560_Tensor_gSFA_t as{s.sa,{s.sf},{s.sf}};
        fi_nvfp4_640_2560_Tensor_gSFB_t bs{sb};
        fi_nvfp4_640_2560_Tensor_gD_t dd{out,{rows}};
        fi_nvfp4_640_2560_Tensor_offsets_t offsets{s.bounds};
        return cute_dsl_fi_nvfp4_640_2560_wrapper(&gu_module[dev],&aa,&bb,&as,&bs,&dd,&offsets,stream);
    }
    fi_nvfp4_2560_640_Tensor_gA_t aa{s.a,{rows}};
    fi_nvfp4_2560_640_Tensor_gB_t bb{w};
    fi_nvfp4_2560_640_Tensor_gSFA_t as{s.sa,{s.sf},{s.sf}};
    fi_nvfp4_2560_640_Tensor_gSFB_t bs{sb};
    fi_nvfp4_2560_640_Tensor_gD_t dd{out,{rows}};
    fi_nvfp4_2560_640_Tensor_offsets_t offsets{s.bounds};
    return cute_dsl_fi_nvfp4_2560_640_wrapper(&down_module[dev],&aa,&bb,&as,&bs,&dd,&offsets,stream);
}
int residual_product(const uint8_t * bank,Native & s,int rows,int role,int dev,cudaStream_t stream){
    Native low=s;low.a=s.low_a;low.sa=s.low_sa;
    low.gate=s.low_product;low.product=s.low_product;
    const int status=gemm(bank,low,rows,role,dev,stream);if(status)return status;
    const uint64_t count=uint64_t(rows)*(role==2?E:F);
    add_residual_product<<<(count+255)/256,256,0,stream>>>(role==0?s.gate:s.product,s.low_product,count);
    return int(cudaGetLastError());
}
bool shape_ok(int nt,uint32_t slots,int mode){
    if(nt<1||nt>INT_MAX/K||!slots||slots>INT_MAX||mode<0||mode>6)return false;
    if(mode==6)return slots==NE||nt<=4;
    if(mode==5)return nt<=4;
    return true; // Original workspace query retains its explicit mode constraints.
}
} // namespace

extern "C" int qfn_native_nvfp4_abi(void){return 1;}
extern "C" uint64_t qfn_native_nvfp4_packed_workspace_bytes(int nt,uint32_t slots,int mode){
    const int parallel=group_parallel_policy();if(parallel<0)return 0;
    const int shared_gu=share_gu_input_policy();if(shared_gu<0)return 0;
    if(shared_gu&&nt>4&&mode!=6){
        std::fprintf(stderr,"QFN_NATIVE_NVFP4_SHARE_GU_INPUT=1 requires mode6 for long requests (nt>4); refusing to ignore input-sharing policy\n");return 0;
    }
    if(parallel&&nt>4&&mode!=6){
        std::fprintf(stderr,"QFN_NATIVE_NVFP4_GROUP_PARALLEL=1 requires mode6 for long requests (nt>4); refusing to ignore grouping policy\n");return 0;
    }
    if(!shape_ok(nt,slots,mode))return 0;Arena a{nullptr};
    const int residual=decode_residual_policy();if(residual<0)return 0;
    const int vector=decode_vector_policy();if(vector<0)return 0;
    if(vector&&mode!=6){
        std::fprintf(stderr,"QFN_NATIVE_NVFP4_DECODE_VECTOR=1 requires explicit mode6 arithmetic\n");return 0;
    }
    if(vector&&residual){
        std::fprintf(stderr,"QFN vector Q8_1 activations and FP4 residual activations are distinct arithmetic modes; select one\n");return 0;
    }
    if(vector&&nt<=4){if(fused_mid_policy()<0)return 0;vector_layout(a,nt);return a.used;}
    if(residual&&nt<=4&&mode!=6){
        std::fprintf(stderr,"QFN_NATIVE_NVFP4_DECODE_RESIDUAL=1 requires short mode6 arithmetic\n");return 0;
    }
    if(mode==6){if(fused_mid_policy()<0)return 0;native_layout(a,std::min(nt,WINDOW),residual&&nt<=4);if(nt<=4)compact_native_layout(a,nt);return a.used;}
    if(!original().valid())return 0;
    const auto s=inverse_layout(a,nt,slots,mode);return s.bytes?a.used:0;
}
extern "C" int qfn_native_nvfp4_prepare(void * stream){
    if(decode_residual_policy()<0||decode_vector_policy()<0||group_parallel_policy()<0||share_gu_input_policy()<0)return int(cudaErrorInvalidValue);
    int dev=-1;auto err=cudaGetDevice(&dev);if(err!=cudaSuccess)return int(err);
    if(dev<0||dev>=64)return int(cudaErrorInvalidDevice);
    if(prepared[dev].load(std::memory_order_acquire))return 0;
    std::fprintf(stderr,"[qfn] native NVFP4 route grouping: parallel=%d, long mode6 only; original warp and short compact paths retained\n",group_parallel_policy());
    std::fprintf(stderr,"[qfn] native NVFP4 shared gate/up input: %d, long mode6 only; bit-identical role calibration required, both GEMMs retained\n",share_gu_input_policy());
    auto cs=static_cast<cudaStream_t>(stream);cudaStreamCaptureStatus capture;
    err=cudaStreamIsCapturing(cs,&capture);if(err!=cudaSuccess)return int(err);
    if(capture!=cudaStreamCaptureStatusNone)return int(cudaErrorStreamCaptureUnsupported);
    int major=0,minor=0;err=cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,dev);
    if(err==cudaSuccess)err=cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,dev);
    if(err!=cudaSuccess)return int(err);if(major!=12||minor!=0)return int(cudaErrorInvalidDeviceFunction);
    if(!original().valid())return int(cudaErrorSharedObjectSymbolNotFound);
    if(int status=original().prepare(stream))return status;
    fi_nvfp4_640_2560_Kernel_Module_Load(&gu_module[dev]);
    fi_nvfp4_2560_640_Kernel_Module_Load(&down_module[dev]);
    err=cudaFuncSetAttribute(group_bank,cudaFuncAttributeMaxDynamicSharedMemorySize,WINDOW*4);
    if(err!=cudaSuccess)return int(err);
    validate_calibration<<<1,256,0,cs>>>(nullptr,0,false);
    validate_routes<<<1,256,0,cs>>>(nullptr,nullptr,nullptr,0,0,false,false);
    inverse_bank<<<1,256,0,cs>>>(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0,0,false);
    compact_packed_routes<<<1,256,0,cs>>>(nullptr,nullptr,nullptr,0,0,nullptr,nullptr,nullptr);
    compact_route_plan<<<1,256,0,cs>>>(0,nullptr,nullptr,nullptr);
    map_calibration<<<1,256,0,cs>>>(nullptr,nullptr,nullptr,0);
    gather_calibration<<<1,256,0,cs>>>(nullptr,nullptr,nullptr,0);
    group_bank<<<1,32,0,cs>>>(nullptr,0,nullptr,nullptr);
    group_bank_parallel<<<1,256,0,cs>>>(nullptr,0,nullptr,nullptr);
    padded_scale_rows<<<1,256,0,cs>>>(nullptr,nullptr,0);
    quantize_calibrated<E><<<1,256,0,cs>>>(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0,nullptr,nullptr,0,0);
    quantize_calibrated<F><<<1,256,0,cs>>>(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,2,nullptr,nullptr,0,0);
    swiglu_original_h<<<1,256,0,cs>>>(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0);
    swiglu_pack_calibrated<<<1,256,0,cs>>>(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0,0);
    scatter_down<<<1,256,0,cs>>>(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0);
    quantize_residual<E,false><<<1,256,0,cs>>>(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0,nullptr,nullptr,nullptr,nullptr,0,0);
    quantize_residual<F,false><<<1,256,0,cs>>>(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,2,nullptr,nullptr,nullptr,nullptr,0,0);
    quantize_residual<F,true><<<1,256,0,cs>>>(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,2,nullptr,nullptr,nullptr,nullptr,0,0);
    add_residual_product<<<1,256,0,cs>>>(nullptr,nullptr,0);
    qfn_vector_quantize<<<1,256,0,cs>>>(nullptr,nullptr,0);
    qfn_vector_gate_up<true><<<1,256,0,cs>>>(nullptr,nullptr,nullptr,nullptr,nullptr,0);
    qfn_vector_gate_up<false><<<1,256,0,cs>>>(nullptr,nullptr,nullptr,nullptr,nullptr,0);
    qfn_vector_swiglu<<<1,256,0,cs>>>(nullptr,nullptr,0);
    qfn_vector_down<<<1,256,0,cs>>>(nullptr,nullptr,nullptr,nullptr,0);
    err=cudaGetLastError();if(err==cudaSuccess)err=cudaStreamSynchronize(cs);
    if(err==cudaSuccess)prepared[dev].store(true,std::memory_order_release);return int(err);
}
#define TRY_CUDA(call) do {int status=int(call);if(status)return status;}while(0)
extern "C" int qfn_native_nvfp4_group_plan(const int32_t * ids,int nt,int parallel,
        int32_t * order,int32_t * bounds,void * stream){
    if(nt<1||nt>WINDOW||(parallel!=0&&parallel!=1)||!ids||!order||!bounds||
       (reinterpret_cast<uintptr_t>(ids)&3)||(reinterpret_cast<uintptr_t>(order)&3)||
       (reinterpret_cast<uintptr_t>(bounds)&3))return int(cudaErrorInvalidValue);
    int dev=-1;TRY_CUDA(cudaGetDevice(&dev));
    if(dev<0||dev>=64||!prepared[dev].load(std::memory_order_acquire))return int(cudaErrorNotReady);
    const auto cs=static_cast<cudaStream_t>(stream);
    if(parallel)group_bank_parallel<<<NE,256,0,cs>>>(ids,nt,order,bounds);
    else group_bank<<<NE,32,nt*4,cs>>>(ids,nt,order,bounds);
    return int(cudaGetLastError());
}
extern "C" int qfn_native_nvfp4_unpack(const uint8_t * bank,const int32_t * physical,
        const int32_t * logical,uint32_t slots,uint32_t count,uint8_t * raw,uint64_t bytes,
        int32_t * gathered,int32_t * canonical,void * stream){
    const uint32_t banks=count?count:slots;
    if(!slots||slots>INT_MAX||!banks||banks>INT_MAX||!bank||!raw||bytes<uint64_t(banks)*BLOB||
       (reinterpret_cast<uintptr_t>(bank)&15)||(reinterpret_cast<uintptr_t>(raw)&15)||
       (reinterpret_cast<uintptr_t>(physical)&3)||(reinterpret_cast<uintptr_t>(logical)&3)||
       (reinterpret_cast<uintptr_t>(gathered)&3)||(reinterpret_cast<uintptr_t>(canonical)&3)||
       (count?(!physical||!gathered||!canonical):(physical||logical||gathered||canonical)))return int(cudaErrorInvalidValue);
    int dev=-1;TRY_CUDA(cudaGetDevice(&dev));
    if(dev<0||dev>=64||!prepared[dev].load(std::memory_order_acquire))return int(cudaErrorNotReady);
    inverse_bank<<<dim3(32,std::min(banks,uint32_t(65535)),3),256,0,static_cast<cudaStream_t>(stream)>>>(
        bank,raw,physical,logical,gathered,canonical,slots,int(banks),count!=0);
    return int(cudaGetLastError());
}
extern "C" int qfn_native_nvfp4_run(const float * x,const int32_t * physical,const int32_t * logical,
        const uint8_t * bank,const float * calibration,uint32_t slots,int nt,int mode,float * out,
        void * workspace,uint64_t bytes,void * stream){
    const auto need=qfn_native_nvfp4_packed_workspace_bytes(nt,slots,mode);
    if(!need||!x||!physical||!bank||!out||!workspace||bytes<need||
       (reinterpret_cast<uintptr_t>(workspace)&255)||(reinterpret_cast<uintptr_t>(bank)&15)||
       (reinterpret_cast<uintptr_t>(x)&15)||(reinterpret_cast<uintptr_t>(out)&15)||
       (reinterpret_cast<uintptr_t>(physical)&3)||(reinterpret_cast<uintptr_t>(logical)&3)||
       (mode>=4&&(!calibration||(reinterpret_cast<uintptr_t>(calibration)&3))))return int(cudaErrorInvalidValue);
    int dev=-1;TRY_CUDA(cudaGetDevice(&dev));
    if(dev<0||dev>=64||!prepared[dev].load(std::memory_order_acquire))return int(cudaErrorNotReady);
    auto cs=static_cast<cudaStream_t>(stream);
    const bool shared_gu=mode==6&&nt>4&&share_gu_input_policy()==1;
    if(mode>=4){validate_calibration<<<6,256,0,cs>>>(calibration,NE*3,shared_gu);TRY_CUDA(cudaGetLastError());}
    if(mode==6&&nt<=4&&decode_vector_policy()==1){
        Arena a{static_cast<uint8_t *>(workspace)};auto s=vector_layout(a,nt);
        const int rows=nt*K;
        validate_routes<<<(rows+255)/256,256,0,cs>>>(physical,logical,bank,slots,rows,false,true);
        qfn_vector_quantize<<<(uint64_t(nt)*E+255)/256,256,0,cs>>>(x,s.xq,uint64_t(nt)*E);
        if(fused_mid_policy()==1)
            qfn_vector_gate_up<true><<<dim3(F/8,rows),256,0,cs>>>(bank,physical,s.xq,s.hidden,s.up,rows);
        else {
            qfn_vector_gate_up<false><<<dim3(F/8,rows),256,0,cs>>>(bank,physical,s.xq,s.hidden,s.up,rows);
            qfn_vector_swiglu<<<(uint64_t(rows)*F+255)/256,256,0,cs>>>(s.hidden,s.up,uint64_t(rows)*F);
        }
        qfn_vector_quantize<<<(uint64_t(rows)*F+255)/256,256,0,cs>>>(s.hidden,s.hq,uint64_t(rows)*F);
        qfn_vector_down<<<dim3(E/8,rows),256,0,cs>>>(bank,physical,s.hq,out,rows);
        return int(cudaGetLastError());
    }
    if(mode!=6){
        Arena a{static_cast<uint8_t *>(workspace)};const auto s=inverse_layout(a,nt,slots,mode);
        const bool small=compact(nt,mode);
        inverse_bank<<<dim3(32,std::min(s.slots,uint32_t(65535)),3),256,0,cs>>>(bank,s.raw,physical,logical,s.ids,s.logical,slots,int(s.slots),small);
        TRY_CUDA(cudaGetLastError());
        if(mode==5)return original().decode(x,s.ids,s.logical,s.raw,calibration,BLOB,s.slots,nt,K,out,s.work,s.bytes,stream);
        if(mode==4){
            const float * aux=calibration;
            if(logical){
                validate_routes<<<(uint64_t(nt)*K+255)/256,256,0,cs>>>(physical,logical,bank,slots,nt*K,true,true);
                TRY_CUDA(cudaMemsetAsync(s.calibration_map,0xff,NE*4,cs));
                map_calibration<<<(uint64_t(nt)*K+255)/256,256,0,cs>>>(physical,logical,s.calibration_map,nt*K);
                gather_calibration<<<6,256,0,cs>>>(calibration,s.calibration_map,s.mapped_calibration,NE*3);
                TRY_CUDA(cudaGetLastError());aux=s.mapped_calibration;
            }
            return original().calibrated(x,physical,s.raw,aux,BLOB,s.slots,nt,K,out,s.work,s.bytes,stream);
        }
        return original().run(x,small?s.ids:physical,s.raw,BLOB,s.slots,nt,K,out,s.work,s.bytes,stream,mode);
    }
    for(int first=0;first<nt;first+=WINDOW){
        const int n=std::min(WINDOW,nt-first),rows=n*K;
        const bool residual=nt<=4&&decode_residual_policy()==1;
        Arena a{static_cast<uint8_t *>(workspace)};auto s=native_layout(a,n,residual);
        const float * xx=x+uint64_t(first)*E;const int32_t * ids=physical+uint64_t(first)*K;
        const int32_t * lids=logical?logical+uint64_t(first)*K:nullptr;
        const uint8_t * routed_bank=bank;
        if(nt<=4){
            auto compact=compact_native_layout(a,n);
            validate_routes<<<(rows+255)/256,256,0,cs>>>(ids,lids,bank,slots,rows,false,true);
            compact_packed_routes<<<dim3(32,rows),256,0,cs>>>(bank,ids,lids,slots,rows,
                compact.bank,compact.physical,compact.logical);
            compact_route_plan<<<(NE+1+255)/256,256,0,cs>>>(rows,s.order,s.bounds,s.scale_row);
            ids=compact.physical;lids=compact.logical;routed_bank=compact.bank;
        }else{
            validate_routes<<<(rows+255)/256,256,0,cs>>>(ids,lids,bank,slots,rows,true,true);
            if(group_parallel_policy()==1)group_bank_parallel<<<NE,256,0,cs>>>(ids,n,s.order,s.bounds);
            else group_bank<<<NE,32,n*4,cs>>>(ids,n,s.order,s.bounds);
            padded_scale_rows<<<NE,256,0,cs>>>(s.bounds,s.scale_row,NE);
        }
        TRY_CUDA(cudaGetLastError());
        TRY_CUDA(cudaMemsetAsync(s.sa,0,uint64_t(s.sf)*E/16,cs));
        quantize_calibrated<E><<<(rows+7)/8,256,0,cs>>>(xx,ids,lids,s.order,s.scale_row,calibration,0,s.a,s.sa,rows,s.sf);
        TRY_CUDA(cudaGetLastError());TRY_CUDA(gemm(routed_bank,s,rows,0,dev,cs));
        if(residual){
            TRY_CUDA(cudaMemsetAsync(s.low_sa,0,uint64_t(s.sf)*E/16,cs));
            quantize_residual<E,false><<<(rows+7)/8,256,0,cs>>>(xx,nullptr,nullptr,ids,lids,s.order,s.scale_row,routed_bank,calibration,0,s.a,s.sa,s.low_a,s.low_sa,rows,s.sf);
            TRY_CUDA(cudaGetLastError());TRY_CUDA(residual_product(routed_bank,s,rows,0,dev,cs));
        }
        // Gate GEMM only reads these activation/scale planes. With identical
        // role globals the up quantizer would write precisely the same bytes.
        if(shared_gu){
            if(!shared_gu_reported.exchange(true,std::memory_order_relaxed))
                std::fprintf(stderr,"[qfn] native NVFP4 shared gate/up quantizer reuse engaged: nt=%d window=%d rows=%d, identical input bytes retained across read-only gate GEMM\n",nt,n,rows);
        }else
            quantize_calibrated<E><<<(rows+7)/8,256,0,cs>>>(xx,ids,lids,s.order,s.scale_row,calibration,1,s.a,s.sa,rows,s.sf);
        TRY_CUDA(cudaGetLastError());TRY_CUDA(gemm(routed_bank,s,rows,1,dev,cs));
        if(residual){
            quantize_residual<E,false><<<(rows+7)/8,256,0,cs>>>(xx,nullptr,nullptr,ids,lids,s.order,s.scale_row,routed_bank,calibration,1,s.a,s.sa,s.low_a,s.low_sa,rows,s.sf);
            TRY_CUDA(cudaGetLastError());TRY_CUDA(residual_product(routed_bank,s,rows,1,dev,cs));
        }
        TRY_CUDA(cudaMemsetAsync(s.sa,0,uint64_t(s.sf)*F/16,cs));
        if(nt<=4&&fused_mid_policy()==1){
            swiglu_pack_calibrated<<<(rows+7)/8,256,0,cs>>>(s.gate,s.product,ids,lids,s.order,
                s.scale_row,routed_bank,calibration,s.a,s.sa,rows,s.sf);
        }else{
            swiglu_original_h<<<(uint64_t(rows)*F+255)/256,256,0,cs>>>(s.gate,s.product,ids,lids,s.order,routed_bank,calibration,rows);
            TRY_CUDA(cudaGetLastError());
            quantize_calibrated<F><<<(rows+7)/8,256,0,cs>>>(s.gate,ids,lids,s.order,s.scale_row,calibration,2,s.a,s.sa,rows,s.sf);
        }
        TRY_CUDA(cudaGetLastError());
        if(residual){
            TRY_CUDA(cudaMemsetAsync(s.low_sa,0,uint64_t(s.sf)*F/16,cs));
            if(fused_mid_policy()==1)
                quantize_residual<F,true><<<(rows+7)/8,256,0,cs>>>(s.gate,s.gate,s.product,ids,lids,s.order,s.scale_row,routed_bank,calibration,2,s.a,s.sa,s.low_a,s.low_sa,rows,s.sf);
            else quantize_residual<F,false><<<(rows+7)/8,256,0,cs>>>(s.gate,nullptr,nullptr,ids,lids,s.order,s.scale_row,routed_bank,calibration,2,s.a,s.sa,s.low_a,s.low_sa,rows,s.sf);
            TRY_CUDA(cudaGetLastError());
        }
        TRY_CUDA(gemm(routed_bank,s,rows,2,dev,cs));
        if(residual)TRY_CUDA(residual_product(routed_bank,s,rows,2,dev,cs));
        scatter_down<<<(uint64_t(rows)*E+255)/256,256,0,cs>>>(s.product,ids,lids,s.order,routed_bank,calibration,out+uint64_t(first)*K*E,uint64_t(rows)*E);
        TRY_CUDA(cudaGetLastError());
    }
    return 0;
}
