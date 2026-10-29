// SPDX-License-Identifier: MIT
#pragma once
#include "ggml-dsv41-dense-fp8.h"
#include "../../../../flashinfer-sm120/dense_fp8_bridge.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
static std::atomic<uint64_t> dense_fp8_calls{0};
extern "C" uint64_t ggml_backend_cuda_dsv41_dense_fp8_queued(){return dense_fp8_calls.load(std::memory_order_acquire);}
static slstack_dense_fp8_fn dsv41_dense_fp8_bridge(){
    static auto call=[](){
#ifdef _WIN32
        const char *path=std::getenv("LLAMA_DSV41_DENSE_FP8_DLL");
        GGML_ASSERT(path&&path[0]&&(path[1]==':'||path[0]=='\\')&&"Explicit absolute dense FP8 DLL required");
        auto lib=LoadLibraryExA(path,nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        GGML_ASSERT(lib&&"Dense FP8 DLL load failed");
        auto abi=reinterpret_cast<int(*)()>(GetProcAddress(lib,"slstack_dense_fp8_abi"));
        auto fn=reinterpret_cast<slstack_dense_fp8_fn>(GetProcAddress(lib,"slstack_dense_fp8"));
        GGML_ASSERT(abi&&abi()==1&&fn&&"Dense FP8 ABI mismatch");return fn;
#else
        GGML_ABORT("Dense FP8 AOT bridge requires Windows SM120");return slstack_dense_fp8_fn(nullptr);
#endif
    }();return call;
}
static void ggml_cuda_dsv41_dense_fp8(ggml_backend_cuda_context&ctx,ggml_tensor *t){
    GGML_ASSERT(ggml_dsv41_dense_fp8_eligible(t)&&ggml_cuda_info().devices[ctx.device].cc==1200);
    constexpr uint64_t bytes=512ull<<20;
    ggml_cuda_pool_alloc<char> raw(ctx.pool(),bytes+255);
    auto aligned=reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(raw.get())+255)&~uintptr_t(255));
    const bool native=ggml_dsv41_dense_fp8_native_weight(t->src[0]);
    GGML_ASSERT((t->op_params[5]==ggml_dsv41_dense_fp8_magic)==native);
    CUDA_CHECK(static_cast<cudaError_t>(dsv41_dense_fp8_bridge()(t->src[0]->data,native?1:0,static_cast<const float*>(t->src[1]->data),
        static_cast<float*>(t->data),int(t->ne[1]),int(t->ne[0]),int(t->src[0]->ne[0]),int64_t(t->src[1]->nb[1]/4),aligned,bytes,ctx.stream())));
    CUDA_CHECK(cudaGetLastError());dense_fp8_calls.fetch_add(1,std::memory_order_release);
}
