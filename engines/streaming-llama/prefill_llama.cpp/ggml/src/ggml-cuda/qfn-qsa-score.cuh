// SPDX-License-Identifier: MIT
#pragma once
#include "ggml-qfn-qsa-score.h"
#include "dispatch-counter.cuh"
#include "qfn-index-fp4.cuh"
#include "qfn-index-refine.cuh"
#include "../../../../../qfn-basalt-nvfp4/qfn_basalt_qsa.h"
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
struct qfn_qsa_score_bridge {
    decltype(&qfn_basalt_qsa_prepare) prepare = nullptr;
    decltype(&qfn_basalt_qsa_run) run = nullptr;
    std::string error;
};
static const qfn_qsa_score_bridge & qfn_qsa_score_functions() {
    static const qfn_qsa_score_bridge result = [] {
        qfn_qsa_score_bridge b;
#ifdef _WIN32
        wchar_t executable[32768];
        const DWORD n = GetModuleFileNameW(nullptr, executable, 32768);
        if (!n || n >= 32768) { b.error = "QFN_QSA_DLL_PATH: executable path unavailable"; return b; }
        std::wstring path(executable, n); const auto slash = path.find_last_of(L"\\/");
        if (slash == std::wstring::npos) { b.error = "QFN_QSA_DLL_PATH: absolute directory unavailable"; return b; }
        path.resize(slash + 1); path += L"qfn-basalt-nvfp4.dll";
        const HMODULE library = LoadLibraryExW(path.c_str(), nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!library) { b.error = "QFN_QSA_DLL_MISSING: adjacent DLL failed to load, Windows error " + std::to_string(GetLastError()); return b; }
        const auto abi = reinterpret_cast<decltype(&qfn_basalt_qsa_abi)>(GetProcAddress(library, "qfn_basalt_qsa_abi"));
        b.prepare = reinterpret_cast<decltype(b.prepare)>(GetProcAddress(library, "qfn_basalt_qsa_prepare"));
        b.run = reinterpret_cast<decltype(b.run)>(GetProcAddress(library, "qfn_basalt_qsa_run"));
        if (!abi || abi() != 1 || !b.prepare || !b.run) b.error = "QFN_QSA_ABI_MISMATCH: independent score ABI1 prepare/run required";
#else
        b.error = "QFN_QSA_PLATFORM: Basalt tensor-core scores require Windows CUDA SM120";
#endif
        return b;
    }(); return result;
}
static ggml_cuda_dispatch_counter qfn_qsa_score_queued{0};
extern "C" uint64_t ggml_backend_cuda_qfn_qsa_score_queued() { return qfn_qsa_score_queued.load(std::memory_order_acquire); }
extern "C" int ggml_backend_cuda_qfn_qsa_score_available(char * error, size_t capacity) {
    const auto & b = qfn_qsa_score_functions();
    if (error && capacity) { const size_t n = std::min(capacity - 1, b.error.size()); std::memcpy(error, b.error.data(), n); error[n] = '\0'; }
    return b.error.empty();
}
extern "C" int ggml_backend_cuda_qfn_qsa_score_prepare(void * stream) {
    const auto & b = qfn_qsa_score_functions();
    return b.error.empty() ? b.prepare(stream) : int(cudaErrorSharedObjectSymbolNotFound);
}
static void ggml_cuda_op_qfn_qsa_score(ggml_backend_cuda_context & ctx, ggml_tensor * t) {
    GGML_ASSERT(ggml_qfn_qsa_score_valid(t) && ggml_cuda_info().devices[ctx.device].cc == 1200);
    if(t->op_params[1]==2){qfn_index_fp4(ctx,t); qfn_qsa_score_queued.fetch_add(1,std::memory_order_release);return;}
    const auto & b = qfn_qsa_score_functions(); GGML_ASSERT(b.error.empty());
    const auto * q = t->src[0], * k = t->src[1], * w = t->src[2];
    float a, scale; std::memcpy(&a, t->op_params + 2, sizeof a); std::memcpy(&scale, t->op_params + 3, sizeof scale);
    CUDA_CHECK((cudaError_t) b.run((const float *) q->data, k->data, (const float *) w->data, (float *) t->data,
        int(q->ne[2]), int(k->ne[2]), int(q->ne[3]), k->type == GGML_TYPE_F16 ? 0 : 1,
        q->nb[1], q->nb[2], q->nb[3], k->nb[2], k->nb[3], w->nb[1], w->nb[3],
        t->nb[1], t->nb[3], a, scale, ctx.stream(), 0));
    if(t->op_params[1]==3)qfn_index_refine(ctx,t);
    CUDA_CHECK(cudaGetLastError()); qfn_qsa_score_queued.fetch_add(1, std::memory_order_release);
}
