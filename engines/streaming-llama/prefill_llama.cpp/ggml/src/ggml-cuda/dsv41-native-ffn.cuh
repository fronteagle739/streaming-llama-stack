#include "dispatch-counter.cuh"
// SPDX-License-Identifier: MIT
#pragma once
#include "ggml-dsv41-native-ffn.h"
#include "mmid.cuh"
#include "ggml-dsv41-flashinfer-workspace.h"
#include "../../../../flashinfer-sm120/flashinfer_native_bridge.h"
#include <atomic>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
static ggml_cuda_dispatch_counter dsv41_native_ffn_calls{0};
extern "C" uint64_t ggml_backend_cuda_dsv41_native_ffn_queued(){return dsv41_native_ffn_calls.load(std::memory_order_acquire);}
static slstack_fi_native_ffn_fn dsv41_native_ffn_bridge(){
    static auto function=[](){
#ifdef _WIN32
        const char *path=std::getenv("LLAMA_DSV41_FLASHINFER_NATIVE_FFN_DLL");
        GGML_ASSERT(path&&path[0]&&(path[1]==':'||path[0]=='\\')&&"Explicit absolute native FFN DLL required");
        auto library=LoadLibraryExA(path,nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        GGML_ASSERT(library&&"Native FFN DLL failed to load");
        auto abi=reinterpret_cast<int(*)()>(GetProcAddress(library,"slstack_flashinfer_native_ffn_abi"));
        auto call=reinterpret_cast<slstack_fi_native_ffn_fn>(GetProcAddress(library,"slstack_flashinfer_native_ffn"));
        GGML_ASSERT(abi&&abi()==1&&call&&"Native FFN ABI mismatch");return call;
#else
        GGML_ABORT("Native FFN bridge is qualified on Windows SM120 only");return slstack_fi_native_ffn_fn(nullptr);
#endif
    }();return function;
}
static void ggml_cuda_op_dsv41_native_ffn(ggml_backend_cuda_context&ctx,ggml_tensor *t){
    GGML_ASSERT(ggml_dsv41_native_ffn_valid(t)&&ggml_cuda_info().devices[ctx.device].cc==1200);
    int tokens=int(t->ne[1]),rows=tokens*6;auto stream=ctx.stream();
    const uint64_t maximum_tokens=ggml_dsv41_flashinfer::batch_tokens(),maximum_routes=maximum_tokens*6;
    GGML_ASSERT(uint64_t(tokens)<=maximum_tokens);
    ggml_cuda_pool_alloc<int32_t> gather(ctx.pool(),maximum_routes),routes(ctx.pool(),maximum_routes),bounds(ctx.pool(),385);
    ggml_cuda_launch_mm_ids_helper(static_cast<const int32_t*>(t->src[2]->data),gather.get(),routes.get(),bounds.get(),384,tokens,6,1,6,1,stream);
    CUDA_CHECK(cudaGetLastError());
    const auto bytes=slstack_fi_native::layout(maximum_tokens).bytes+slstack_fi_native::aligned(maximum_routes*4);
    ggml_cuda_pool_alloc<char> workspace(ctx.pool(),bytes+255);
    void *aligned=reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(workspace.get())+255)&~uintptr_t(255));
    float limit;std::memcpy(&limit,t->op_params+2,sizeof(limit));
    CUDA_CHECK(static_cast<cudaError_t>(dsv41_native_ffn_bridge()(t->src[3]->data,t->src[4]->data,t->src[5]->data,
        static_cast<const float*>(t->src[0]->data),static_cast<const float*>(t->src[1]->data),routes.get(),bounds.get(),
        static_cast<float*>(t->data),tokens,limit,aligned,bytes,stream)));
    CUDA_CHECK(cudaGetLastError());dsv41_native_ffn_calls.fetch_add(1,std::memory_order_release);
}
