#pragma once
#include <cstdlib>
#include "ggml-dsv41-flashinfer-workspace.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
using dsv41_flashinfer_fn=int (*)(const void*,const float*,const int32_t*,const int32_t*,const int32_t*,float*,
    int,int,int,int64_t,int64_t,int64_t,void*,uint64_t,cudaStream_t);
static bool dsv41_flashinfer_requested(){
    const char *value=std::getenv("LLAMA_DSV41_FLASHINFER_MXFP4");
    return value&&std::strcmp(value,"1")==0;
}
static dsv41_flashinfer_fn dsv41_flashinfer_bridge(){
    static dsv41_flashinfer_fn function=[](){
#ifdef _WIN32
        const char *path=std::getenv("LLAMA_DSV41_FLASHINFER_MXFP4_DLL");
        GGML_ASSERT(path&&path[0]&&"Explicit FlashInfer bridge DLL path required");
        GGML_ASSERT((path[1]==':'||path[0]=='\\')&&"FlashInfer DLL path must be absolute");
        HMODULE library=LoadLibraryExA(path,nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        GGML_ASSERT(library&&"FlashInfer bridge DLL failed to load");
        auto abi=reinterpret_cast<int(*)()>(GetProcAddress(library,"slstack_flashinfer_mxfp4_abi"));
        auto call=reinterpret_cast<dsv41_flashinfer_fn>(GetProcAddress(library,"slstack_flashinfer_mxfp4"));
        GGML_ASSERT(abi&&abi()==3&&call&&"FlashInfer bridge ABI mismatch");
        return call;
#else
        GGML_ABORT("This optional FlashInfer AOT bridge is currently qualified on Windows only");
        return dsv41_flashinfer_fn(nullptr);
#endif
    }();
    return function;
}
