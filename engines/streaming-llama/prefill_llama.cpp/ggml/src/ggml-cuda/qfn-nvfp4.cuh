// SPDX-License-Identifier: MIT
#pragma once
#include "ggml-qfn-nvfp4.h"
#include "../../../../../qfn-basalt-nvfp4/qfn_basalt_nvfp4.h"
#include "../../../../flashinfer-sm120/qfn_native_nvfp4_bridge.h"
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// Same explicit C ABI boundary as dsv41-native-ffn.cuh, with a single installed adjacent DLL.
struct qfn_nvfp4_bridge {
    decltype(&qfn_basalt_nvfp4_workspace_bytes) workspace = nullptr;
    decltype(&qfn_basalt_nvfp4_prepare) prepare = nullptr;
    decltype(&qfn_basalt_nvfp4_run) run = nullptr;
    decltype(&qfn_basalt_nvfp4_calibrated_abi) calibrated_abi = nullptr;
    decltype(&qfn_basalt_nvfp4_run_calibrated) calibrated_run = nullptr;
    decltype(&qfn_basalt_nvfp4_calibrated_decode_abi) decode_abi = nullptr;
    decltype(&qfn_basalt_nvfp4_run_calibrated_decode) decode_run = nullptr;
    std::string error;
};
static const qfn_nvfp4_bridge & qfn_nvfp4_functions() {
    static const qfn_nvfp4_bridge result = [] {
        qfn_nvfp4_bridge b;
#ifdef _WIN32
        wchar_t executable[32768];
        const DWORD n = GetModuleFileNameW(nullptr, executable, 32768);
        if (!n || n >= 32768) { b.error = "QFN_NVFP4_DLL_PATH: cannot resolve executable path"; return b; }
        std::wstring path(executable, n);
        const auto slash = path.find_last_of(L"\\/");
        if (slash == std::wstring::npos) { b.error = "QFN_NVFP4_DLL_PATH: executable path is not absolute"; return b; }
        path.resize(slash + 1);
        path += L"qfn-basalt-nvfp4.dll";
        HMODULE library = LoadLibraryExW(path.c_str(), nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!library) { b.error = "QFN_NVFP4_DLL_MISSING: adjacent qfn-basalt-nvfp4.dll load failed, Windows error " +
                                  std::to_string(GetLastError()); return b; }
        const auto abi = reinterpret_cast<decltype(&qfn_basalt_nvfp4_abi)>(GetProcAddress(library, "qfn_basalt_nvfp4_abi"));
        b.workspace = reinterpret_cast<decltype(b.workspace)>(GetProcAddress(library, "qfn_basalt_nvfp4_workspace_bytes"));
        b.prepare = reinterpret_cast<decltype(b.prepare)>(GetProcAddress(library, "qfn_basalt_nvfp4_prepare"));
        b.run = reinterpret_cast<decltype(b.run)>(GetProcAddress(library, "qfn_basalt_nvfp4_run"));
        b.calibrated_abi = reinterpret_cast<decltype(b.calibrated_abi)>(GetProcAddress(library, "qfn_basalt_nvfp4_calibrated_abi"));
        b.calibrated_run = reinterpret_cast<decltype(b.calibrated_run)>(GetProcAddress(library, "qfn_basalt_nvfp4_run_calibrated"));
        b.decode_abi = reinterpret_cast<decltype(b.decode_abi)>(GetProcAddress(library, "qfn_basalt_nvfp4_calibrated_decode_abi"));
        b.decode_run = reinterpret_cast<decltype(b.decode_run)>(GetProcAddress(library, "qfn_basalt_nvfp4_run_calibrated_decode"));
        if (!abi || abi() != 1 || !b.workspace || !b.prepare || !b.run)
            b.error = "QFN_NVFP4_ABI_MISMATCH: adjacent DLL must provide ABI1 workspace/prepare/run";
#else
        b.error = "QFN_NVFP4_PLATFORM: native bridge requires Windows CUDA SM120";
#endif
        return b;
    }();
    return result;
}
extern "C" int ggml_backend_cuda_qfn_nvfp4_available(char * error, size_t capacity) {
    const auto & b = qfn_nvfp4_functions();
    if (error && capacity) {
        const size_t n = std::min(capacity - 1, b.error.size());
        std::memcpy(error, b.error.data(), n); error[n] = '\0';
    }
    return b.error.empty();
}
extern "C" int ggml_backend_cuda_qfn_nvfp4_calibrated_available(char * error, size_t capacity) {
    const auto & b = qfn_nvfp4_functions();
    const std::string message = !b.error.empty() ? b.error :
        (!b.calibrated_abi || !b.calibrated_run || b.calibrated_abi() != 1)
        ? "QFN_NVFP4_CALIBRATED_ABI: explicit calibrated mode requires calibrated ABI1/run exports" : "";
    if (error && capacity) { const size_t n = std::min(capacity - 1, message.size());
        std::memcpy(error, message.data(), n); error[n] = '\0'; }
    return message.empty();
}
extern "C" int ggml_backend_cuda_qfn_nvfp4_calibrated_decode_available(char * error,size_t capacity) {
    const auto & b=qfn_nvfp4_functions();
    const std::string message=!b.error.empty()?b.error:
        (!b.decode_abi||!b.decode_run||b.decode_abi()!=1)
        ?"QFN_NVFP4_CALIBRATED_DECODE_ABI: slot-aware calibrated decode ABI1/run required":"";
    if(error&&capacity) { const size_t n=std::min(capacity-1,message.size());
        std::memcpy(error,message.data(),n);error[n]='\0'; }
    return message.empty();
}
extern "C" uint64_t ggml_backend_cuda_qfn_nvfp4_workspace_bytes(int tokens, int top_k, uint32_t slots, int mode) {
    const auto & b = qfn_nvfp4_functions();
    return b.error.empty() ? b.workspace(tokens, top_k, slots, mode) : 0;
}
extern "C" int ggml_backend_cuda_qfn_nvfp4_prepare(void * stream) {
    const auto & b = qfn_nvfp4_functions();
    return b.error.empty() ? b.prepare(stream) : int(cudaErrorSharedObjectSymbolNotFound);
}
struct qfn_native_nvfp4_bridge {
    decltype(&qfn_native_nvfp4_packed_workspace_bytes) workspace = nullptr;
    decltype(&qfn_native_nvfp4_prepare) prepare = nullptr;
    decltype(&qfn_native_nvfp4_run) run = nullptr;
    std::string error;
};
static const qfn_native_nvfp4_bridge & qfn_native_nvfp4_functions() {
    static const auto result = [] {
        qfn_native_nvfp4_bridge b;
#ifdef _WIN32
        wchar_t executable[32768];
        const DWORD n=GetModuleFileNameW(nullptr,executable,32768);
        if (!n || n>=32768) { b.error="QFN_NATIVE_NVFP4_PATH: executable resolution failed";return b; }
        std::wstring path(executable,n);const auto slash=path.find_last_of(L"\\/");
        if (slash==std::wstring::npos) { b.error="QFN_NATIVE_NVFP4_PATH: absolute executable path required";return b; }
        path.resize(slash+1);path+=L"qfn-native-nvfp4.dll";
        auto library=LoadLibraryExW(path.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!library) { b.error="QFN_NATIVE_NVFP4_DLL: adjacent qfn-native-nvfp4.dll load failed: "+std::to_string(GetLastError());return b; }
        auto abi=reinterpret_cast<decltype(&qfn_native_nvfp4_abi)>(GetProcAddress(library,"qfn_native_nvfp4_abi"));
        b.workspace=reinterpret_cast<decltype(b.workspace)>(GetProcAddress(library,"qfn_native_nvfp4_packed_workspace_bytes"));
        b.prepare=reinterpret_cast<decltype(b.prepare)>(GetProcAddress(library,"qfn_native_nvfp4_prepare"));
        b.run=reinterpret_cast<decltype(b.run)>(GetProcAddress(library,"qfn_native_nvfp4_run"));
        if (!abi || abi()!=1 || !b.workspace || !b.prepare || !b.run) b.error="QFN_NATIVE_NVFP4_ABI: ABI1 prepare/workspace/run required";
#else
        b.error="QFN_NATIVE_NVFP4_PLATFORM: Windows CUDA SM120 required";
#endif
        return b;
    }();
    return result;
}
extern "C" int ggml_backend_cuda_qfn_native_nvfp4_available(char * error,size_t capacity) {
    const auto & b=qfn_native_nvfp4_functions();
    if (error && capacity) { const auto n=std::min(capacity-1,b.error.size());std::memcpy(error,b.error.data(),n);error[n]='\0'; }
    return b.error.empty();
}
extern "C" uint64_t ggml_backend_cuda_qfn_native_nvfp4_workspace_bytes(int tokens,uint32_t slots,int mode) {
    const auto & b=qfn_native_nvfp4_functions();return b.error.empty()?b.workspace(tokens,slots,mode):0;
}
extern "C" int ggml_backend_cuda_qfn_native_nvfp4_prepare(void * stream) {
    const auto & b=qfn_native_nvfp4_functions();return b.error.empty()?b.prepare(stream):int(cudaErrorSharedObjectSymbolNotFound);
}
static void ggml_cuda_op_qfn_nvfp4(ggml_backend_cuda_context & ctx, ggml_tensor * t) {
    GGML_ASSERT(ggml_qfn_nvfp4_valid(t) && ggml_cuda_info().devices[ctx.device].cc == 1200);
    if (t->op_params[3] == 1) {
        const auto & b=qfn_native_nvfp4_functions();
        GGML_ASSERT(b.error.empty());
        const auto needed=b.workspace(int(t->ne[2]),uint32_t(t->src[2]->ne[1]),t->op_params[2]);
        GGML_ASSERT(needed && needed<=uint64_t(t->src[3]->ne[0]) && !(reinterpret_cast<uintptr_t>(t->src[3]->data)&255));
        CUDA_CHECK(static_cast<cudaError_t>(b.run(static_cast<const float *>(t->src[0]->data),
            static_cast<const int32_t *>(t->src[1]->data),t->src[5]?static_cast<const int32_t *>(t->src[5]->data):nullptr,
            static_cast<const uint8_t *>(t->src[2]->data),t->src[4]?static_cast<const float *>(t->src[4]->data):nullptr,
            uint32_t(t->src[2]->ne[1]),int(t->ne[2]),t->op_params[2],static_cast<float *>(t->data),
            t->src[3]->data,uint64_t(t->src[3]->ne[0]),ctx.stream())));
        CUDA_CHECK(cudaGetLastError());return;
    }
    const auto & b = qfn_nvfp4_functions();
    GGML_ASSERT(b.error.empty() && "QFN NVFP4 bridge preflight was not completed");
    const int tokens = int(t->ne[2]), mode = t->op_params[2];
    const uint32_t slots = uint32_t(t->src[2]->ne[1]);
    const uint64_t needed = b.workspace(tokens, 10, slots, mode);
    GGML_ASSERT(needed && needed <= uint64_t(t->src[3]->ne[0]) &&
                !(reinterpret_cast<uintptr_t>(t->src[3]->data) & 255));
    if (mode == 5) {
        GGML_ASSERT(b.decode_abi && b.decode_run && b.decode_abi()==1);
        CUDA_CHECK(static_cast<cudaError_t>(b.decode_run(static_cast<const float *>(t->src[0]->data),
            static_cast<const int32_t *>(t->src[1]->data),static_cast<const int32_t *>(t->src[5]->data),
            static_cast<const uint8_t *>(t->src[2]->data),static_cast<const float *>(t->src[4]->data),
            uint64_t(t->src[2]->ne[0]),slots,tokens,10,static_cast<float *>(t->data),
            t->src[3]->data,uint64_t(t->src[3]->ne[0]),ctx.stream())));
    } else if (mode == 4) {
        GGML_ASSERT(b.calibrated_abi && b.calibrated_run && b.calibrated_abi() == 1);
        CUDA_CHECK(static_cast<cudaError_t>(b.calibrated_run(static_cast<const float *>(t->src[0]->data),
            static_cast<const int32_t *>(t->src[1]->data), static_cast<const uint8_t *>(t->src[2]->data),
            static_cast<const float *>(t->src[4]->data), uint64_t(t->src[2]->ne[0]), slots, tokens, 10,
            static_cast<float *>(t->data), t->src[3]->data, uint64_t(t->src[3]->ne[0]), ctx.stream())));
    } else {
    CUDA_CHECK(static_cast<cudaError_t>(b.run(static_cast<const float *>(t->src[0]->data),
        static_cast<const int32_t *>(t->src[1]->data), static_cast<const uint8_t *>(t->src[2]->data),
        uint64_t(t->src[2]->ne[0]), slots, tokens, 10, static_cast<float *>(t->data), t->src[3]->data,
        uint64_t(t->src[3]->ne[0]), ctx.stream(), mode)));
    }
    CUDA_CHECK(cudaGetLastError());
}
