// SPDX-License-Identifier: MIT
// Full-output comparison with the separately built, qualified ABI4 adapter.
#include "flashinfer_native_bridge.h"
#include "flashinfer_bridge.h"
#include "../prefill_llama.cpp/ggml/src/ggml-cuda/dsv41-writer-packed-kernel.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define NOMINMAX
#include <windows.h>
struct block {uint8_t scale,codes[16];};
static void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));std::exit(2);}}
static __host__ __device__ unsigned hash(unsigned i){i^=i>>16;i*=0x7feb352du;i^=i>>15;i*=0x846ca68bu;return i^(i>>16);}
// Fill BOTH representations independently: canonical split nibbles versus
// adjacent nibble values and K-group/N/4 scales. No production repacker is used.
static __global__ void fill(block *canonical,uint8_t *native,int n,int k,int kind,bool slots=false,int mult=73){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x,total=size_t(slots?265:384)*n*(k/32);if(i>=total)return;
    int kb=i%(k/32),column=i/(k/32)%n,expert=i/(size_t(n)*(k/32));
    const int source=slots?(expert*mult+19)%384:expert;
    const size_t source_i=(size_t(source)*n+column)*(k/32)+kb;
    block b;b.scale=uint8_t(116+(kb+source+kind)%3);
    for(int j=0;j<16;++j){
        unsigned lo=hash(unsigned(source_i*32+j)+kind*73)&15,hi=hash(unsigned(source_i*32+j+16)+kind*73)&15;
        b.codes[j]=lo|(hi<<4);
    }
    canonical[source_i]=b;
    for(int j=0;j<16;++j){int a=2*j,c=a+1;
        uint8_t lo=(b.codes[a&15]>>(a>=16?4:0))&15,hi=(b.codes[c&15]>>(c>=16?4:0))&15;
        const size_t slot_first=size_t(expert)*(size_t(n)*k/2+size_t(n)*k/32);
        native[(slots?slot_first+(size_t(column)*(k/32)+kb)*16:i*16)+j]=lo|(hi<<4);
    }
    const size_t scale_at=slots?size_t(expert)*(size_t(n)*k/2+size_t(n)*k/32)+size_t(n)*k/2+(size_t(kb/4)*n+column)*4+kb%4:
        size_t(384)*n*k/2+((size_t(expert)*(k/128)+kb/4)*n+column)*4+kb%4;
    native[scale_at]=b.scale;
}
static __global__ void fill_pack(float *pack,size_t count){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)pack[i]=float(int(hash(unsigned(i)+977u)%1025)-512)/4096.f;
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
    if(argc<4||argc>6||(argc==6&&std::strcmp(argv[5],"sparse")&&std::strcmp(argv[5],"slots")&&std::strcmp(argv[5],"slots-writer")))return 1;
    const bool writer=argc==6&&!std::strcmp(argv[5],"slots-writer");
    const bool slots=writer||(argc==6&&!std::strcmp(argv[5],"slots"));
    int tokens=std::atoi(argv[3]);if(tokens<1||tokens>65536)return 1;
    auto library=LoadLibraryA(argv[1]),legacy=LoadLibraryA(argv[2]);if(!library||!legacy)return 2;
    auto call=reinterpret_cast<slstack_fi_native_ffn_fn>(GetProcAddress(library,"slstack_flashinfer_native_ffn"));
    auto slot_call=reinterpret_cast<slstack_fi_native_ffn_slots_fn>(GetProcAddress(library,"slstack_flashinfer_native_ffn_slots_v2"));
    auto abi=reinterpret_cast<int(*)()>(GetProcAddress(library,"slstack_flashinfer_native_ffn_abi"));
    auto size=reinterpret_cast<uint64_t(*)(int)>(GetProcAddress(library,"slstack_flashinfer_native_ffn_workspace"));
    auto old=reinterpret_cast<slstack_flashinfer_mxfp4_fn>(GetProcAddress(legacy,"slstack_flashinfer_mxfp4"));
    auto old_abi=reinterpret_cast<int(*)()>(GetProcAddress(legacy,"slstack_flashinfer_mxfp4_abi"));
    if(!call||!size||!old||!old_abi||old_abi()!=4||!abi||abi()!=1||(slots&&!slot_call))return 2;
    int rows=tokens*6;std::vector<int>routes(rows),gather(rows),bounds(385),expert(rows);
    for(int r=0;r<rows;++r){routes[r]=r;expert[r]=slots?((r/6)*7+(r%6)*61)%265:argc==6?r%6:((r/6)*7+(r%6)*61)%384;}
    if(slots)expert[0]=264; // exercise the last physical slot beside empty groups265..383
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
    if(writer){
        // Steered slot output against the existing full-plane writer path, bit for bit.
        // Two slot arrangements: the projection must follow each slot replacement.
        auto rank_call=reinterpret_cast<slstack_fi_native_ffn_rank_fn>(GetProcAddress(library,"slstack_flashinfer_native_ffn_rank_v1"));
        if(!rank_call||!slot_call)return 2;
        const int rank=3,layer=3,physical=265;const size_t pack_count=size_t(layer*2+2)*rank*5121;
        const size_t out_bytes=size_t(tokens)*5120*4,elements=size_t(tokens)*5120;
        float *pack,*a_slot,*a_plane,*dots,*steered,*planes,*plain;uint8_t *slot[3],*plane[3];
        check(cudaMalloc(&pack,pack_count*4));check(cudaMalloc(&a_slot,size_t(384)*rank*2304*4));check(cudaMalloc(&a_plane,size_t(384)*rank*2304*4));
        check(cudaMalloc(&dots,size_t(rows)*rank*4));check(cudaMalloc(&steered,out_bytes));check(cudaMalloc(&planes,out_bytes));check(cudaMalloc(&plain,out_bytes));
        fill_pack<<<unsigned((pack_count+255)/256),256>>>(pack,pack_count);
        const float *vectors=pack+(size_t(layer)*2+1)*rank*5121;
        for(int kind=0;kind<3;++kind){check(cudaMalloc(&slot[kind],weight_bytes/384*physical));check(cudaMalloc(&plane[kind],weight_bytes));
            int n=kind==2?5120:2304,k=kind==2?2304:5120;
            fill<<<unsigned((blocks+255)/256),256>>>(canonical,plane[kind],n,k,kind,false);}
        dsv41_writer_packed_a_kernel<<<dim3((2304+31)/32,384),dim3(32,8)>>>(a_plane,plane[2],plane[2]+size_t(384)*5120*2304/2,pack,layer,rank,false);
        check(cudaDeviceSynchronize());
        unsigned long long totals[3]={0,0,0};size_t compared=0;
        const std::vector<int> base_expert=expert;
        const int multipliers[2]={73,35};
        for(int pass=0;pass<2;++pass){
            const int mult=multipliers[pass];
            for(int kind=0;kind<3;++kind){int n=kind==2?5120:2304,k=kind==2?2304:5120;
                fill<<<unsigned((blocks+255)/256),256>>>(canonical,slot[kind],n,k,kind,true,mult);}
            dsv41_writer_packed_a_kernel<<<dim3((2304+31)/32,unsigned(physical)),dim3(32,8)>>>(a_slot,slot[2],slot[2]+size_t(5120)*2304/2,pack,layer,rank,true);
            check(cudaDeviceSynchronize());
            auto upload=[&](const std::vector<int>&ids){
                for(int r=0;r<rows;++r)routes[r]=r;
                std::stable_sort(routes.begin(),routes.end(),[&](int a,int b){return ids[a]<ids[b];});
                std::fill(bounds.begin(),bounds.end(),0);
                for(int r=0;r<rows;++r)bounds[ids[r]+1]++;
                for(int e=0;e<384;++e)bounds[e+1]+=bounds[e];
                check(cudaMemcpy(dr,routes.data(),rows*4,cudaMemcpyHostToDevice));check(cudaMemcpy(db,bounds.data(),385*4,cudaMemcpyHostToDevice));};
            upload(base_expert);
            slstack_fi_writer_delta delta{1,uint32_t(rank),a_slot,vectors,dots};
            check(cudaMemset(steered,0xff,out_bytes));check(cudaMemset(plain,0xff,out_bytes));
            check(static_cast<cudaError_t>(slot_call(slot[0],slot[1],slot[2],x,rw,dr,db,steered,tokens,10.f,workspace,size(tokens),nullptr,&delta,physical)));
            check(static_cast<cudaError_t>(slot_call(slot[0],slot[1],slot[2],x,rw,dr,db,plain,tokens,10.f,workspace,size(tokens),nullptr,nullptr,physical)));
            check(cudaDeviceSynchronize());
            std::vector<int> sources(rows);for(int r=0;r<rows;++r)sources[r]=(base_expert[r]*mult+19)%384;
            upload(sources);
            slstack_fi_writer_delta plane_delta{1,uint32_t(rank),a_plane,vectors,dots};
            check(cudaMemset(planes,0xff,out_bytes));
            check(static_cast<cudaError_t>(rank_call(plane[0],plane[1],plane[2],x,rw,dr,db,planes,tokens,10.f,workspace,size(tokens),nullptr,&plane_delta)));
            check(cudaDeviceSynchronize());
            unsigned long long observed[2];
            check(cudaMemset(errors,0,16));
            compare<<<unsigned((elements+255)/256),256>>>(planes,steered,elements,errors);
            check(cudaDeviceSynchronize());check(cudaMemcpy(observed,errors,16,cudaMemcpyDeviceToHost));
            totals[0]+=observed[0];totals[1]+=observed[1];compared+=elements;
            check(cudaMemset(errors,0,16));
            compare<<<unsigned((elements+255)/256),256>>>(plain,steered,elements,errors);
            check(cudaDeviceSynchronize());check(cudaMemcpy(observed,errors,16,cudaMemcpyDeviceToHost));
            totals[2]+=observed[0];
            std::printf("{\"pass\":%d,\"slot_multiplier\":%d,\"running_different_bits\":%llu,\"running_nonfinite\":%llu,\"running_elements_changed_by_writer\":%llu}\n",pass,mult,totals[0],totals[1],totals[2]);std::fflush(stdout);
        }
        const bool pass=!totals[0]&&!totals[1]&&totals[2]>0;
        std::printf("{\"status\":\"%s\",\"mode\":\"slots-writer\",\"physical_slots\":%d,\"rank\":%d,\"tokens\":%d,\"compared_elements\":%llu,\"different_bits\":%llu,\"nonfinite\":%llu,\"elements_changed_by_writer\":%llu}\n",
            pass?"PASS":"FAIL",physical,rank,tokens,(unsigned long long)compared,totals[0],totals[1],totals[2]);
        return pass?0:3;
    }
    for(int kind=0;kind<3;++kind){check(cudaMalloc(&native[kind],slots?weight_bytes/384*265:weight_bytes));int n=kind==2?5120:2304,k=kind==2?2304:5120;
        fill<<<unsigned((blocks+255)/256),256>>>(canonical,native[kind],n,k,kind,slots);}
    check(cudaDeviceSynchronize());
    check(cudaMemset(output,0xff,size_t(tokens)*5120*4));
    auto invoke=[&](){return slots?slot_call(native[0],native[1],native[2],x,rw,dr,db,output,tokens,10.f,workspace,size(tokens),nullptr,nullptr,265):
        call(native[0],native[1],native[2],x,rw,dr,db,output,tokens,10.f,workspace,size(tokens),nullptr);};
    check(static_cast<cudaError_t>(invoke()));
    check(cudaDeviceSynchronize());
    cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
    check(cudaEventRecord(begin));
    for(int repeat=0;repeat<5;++repeat)
        check(static_cast<cudaError_t>(invoke()));
    check(cudaEventRecord(end));check(cudaEventSynchronize(end));float elapsed_ms=0;check(cudaEventElapsedTime(&elapsed_ms,begin,end));
    elapsed_ms/=5;check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
    float *reference=nullptr;
    unsigned long long *plane_errors;check(cudaMalloc(&plane_errors,16));unsigned long long plane_observed[2]={0,0},plane_compared=0;
    if(slots){
        for(auto&e:expert)e=(e*73+19)%384;
        std::stable_sort(routes.begin(),routes.end(),[&](int a,int b){return expert[a]<expert[b];});
        std::fill(bounds.begin(),bounds.end(),0);
        for(int r=0;r<rows;++r){gather[r]=routes[r]/6;bounds[expert[r]+1]++;}
        for(int e=0;e<384;++e)bounds[e+1]+=bounds[e];
        check(cudaMemcpy(dr,routes.data(),rows*4,cudaMemcpyHostToDevice));check(cudaMemcpy(dg,gather.data(),rows*4,cudaMemcpyHostToDevice));check(cudaMemcpy(db,bounds.data(),385*4,cudaMemcpyHostToDevice));
    }
    if(tokens<=16384){
        // Retire native workspace before allocating the intentionally larger reference.
        check(cudaFree(workspace));workspace=nullptr;
        const auto old_bytes=ggml_dsv41_flashinfer::layout(rows,5120,2304).bytes;
        void *old_workspace;check(cudaMalloc(&old_workspace,old_bytes));
        float *g,*u,*mid,*all;check(cudaMalloc(&g,size_t(rows)*2304*4));check(cudaMalloc(&u,size_t(rows)*2304*4));check(cudaMalloc(&mid,size_t(rows)*2304*4));
        for(int kind=0;kind<2;++kind){fill<<<unsigned((blocks+255)/256),256>>>(canonical,native[kind],2304,5120,kind,slots);
            check(static_cast<cudaError_t>(old(canonical,x,dg,dr,db,kind?u:g,rows,2304,5120,5120,2304,2304*6,old_workspace,old_bytes,nullptr)));}
        middle<<<unsigned((size_t(rows)*2304+255)/256),256>>>(g,u,rw,mid,size_t(rows)*2304);
        check(cudaFree(g));check(cudaFree(u));check(cudaMalloc(&all,size_t(rows)*5120*4));
        fill<<<unsigned((blocks+255)/256),256>>>(canonical,native[2],5120,2304,2,slots);
        check(static_cast<cudaError_t>(old(canonical,mid,dr,dr,db,all,rows,5120,2304,2304,5120,5120*6,old_workspace,old_bytes,nullptr)));
        check(cudaMalloc(&reference,size_t(tokens)*5120*4));reduce_reference<<<unsigned((size_t(tokens)*5120+255)/256),256>>>(all,reference,tokens);
        compare<<<unsigned((size_t(tokens)*5120+255)/256),256>>>(reference,output,size_t(tokens)*5120,errors);
        check(cudaDeviceSynchronize());check(cudaFree(old_workspace));check(cudaFree(mid));check(cudaFree(all));check(cudaFree(reference));
        if(slots){
            // Direct differential: the same experts, loaded as full planes, through
            // the qualified full-plane path. Routes and bounds above name source experts.
            for(int kind=0;kind<3;++kind){check(cudaFree(native[kind]));check(cudaMalloc(&native[kind],weight_bytes));
                int n=kind==2?5120:2304,k=kind==2?2304:5120;
                fill<<<unsigned((blocks+255)/256),256>>>(canonical,native[kind],n,k,kind,false);}
            check(cudaDeviceSynchronize());
            float *planes;check(cudaMalloc(&planes,size_t(tokens)*5120*4));check(cudaMemset(planes,0xff,size_t(tokens)*5120*4));
            check(cudaMalloc(&workspace,size(tokens)));
            check(static_cast<cudaError_t>(call(native[0],native[1],native[2],x,rw,dr,db,planes,tokens,10.f,workspace,size(tokens),nullptr)));
            check(cudaMemset(plane_errors,0,16));
            compare<<<unsigned((size_t(tokens)*5120+255)/256),256>>>(planes,output,size_t(tokens)*5120,plane_errors);
            check(cudaDeviceSynchronize());check(cudaMemcpy(plane_observed,plane_errors,16,cudaMemcpyDeviceToHost));
            check(cudaFree(planes));plane_compared=size_t(tokens)*5120;
        }
    }else if(argc>=5){
        // Large-batch differential check with the independently retained
        // pre-optimization ABI1 bridge, reusing the same bounded workspace.
        auto control_library=LoadLibraryA(argv[4]);if(!control_library)return 2;
        auto control=reinterpret_cast<slstack_fi_native_ffn_fn>(GetProcAddress(control_library,"slstack_flashinfer_native_ffn"));
        auto control_abi=reinterpret_cast<int(*)()>(GetProcAddress(control_library,"slstack_flashinfer_native_ffn_abi"));
        if(!control||!control_abi||control_abi()!=1)return 2;
        check(cudaMalloc(&reference,size_t(tokens)*5120*4));
        check(cudaMemset(reference,0xff,size_t(tokens)*5120*4));
        check(static_cast<cudaError_t>(control(native[0],native[1],native[2],x,rw,dr,db,reference,tokens,10.f,workspace,size(tokens),nullptr)));
        compare<<<unsigned((size_t(tokens)*5120+255)/256),256>>>(reference,output,size_t(tokens)*5120,errors);
        check(cudaDeviceSynchronize());check(cudaFree(reference));
    }else compare<<<unsigned((size_t(tokens)*5120+255)/256),256>>>(nullptr,output,size_t(tokens)*5120,errors);
    check(cudaDeviceSynchronize());unsigned long long observed[2];check(cudaMemcpy(observed,errors,16,cudaMemcpyDeviceToHost));
    std::printf("{\"status\":\"%s\",\"layout\":\"%s\",\"physical_experts\":%d,\"reference_abi\":4,\"tokens\":%d,\"workspace_bytes\":%llu,\"compared_elements\":%llu,\"different_bits\":%llu,\"nonfinite\":%llu,\"fullplane_compared_elements\":%llu,\"fullplane_different_bits\":%llu,\"fullplane_nonfinite\":%llu,\"runtime_weight_repack_bytes\":0,\"native_ffn_ms\":%.6f,\"effective_tflops\":%.6f,\"timed_repeats\":5}\n",
        observed[0]||observed[1]||plane_observed[0]||plane_observed[1]?"FAIL":"PASS",slots?"slot-interleaved":"full-planes",slots?265:384,tokens,(unsigned long long)size(tokens),(tokens<=16384||argc>=5)?(unsigned long long)tokens*5120:0,observed[0],observed[1],plane_compared,plane_observed[0],plane_observed[1],elapsed_ms,3.*2*2304*5120*rows/elapsed_ms/1.e9);
    for(auto*p:native)check(cudaFree(p));check(cudaFree(canonical));if(workspace)check(cudaFree(workspace));
    check(cudaFree(x));check(cudaFree(rw));check(cudaFree(output));check(cudaFree(dr));check(cudaFree(dg));check(cudaFree(db));check(cudaFree(errors));
    check(cudaFree(plane_errors));
    return observed[0]||observed[1]||plane_observed[0]||plane_observed[1]?3:0;
}
