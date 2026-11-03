// SPDX-License-Identifier: MIT
#pragma once
#include "ggml-qfn-attention.h"
#include "../../../../../qfn-basalt-nvfp4/qfn_basalt_attention.h"
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

struct qfn_attention_bridge {
    decltype(&qfn_basalt_attn_workspace_bytes) workspace=nullptr;
    decltype(&qfn_basalt_attn_prepare) prepare=nullptr;
    decltype(&qfn_basalt_attn_run) run=nullptr;
    std::string error;
};
static const qfn_attention_bridge & qfn_attention_functions() {
    static const auto result=[] {
        qfn_attention_bridge b;
#ifdef _WIN32
        wchar_t executable[32768];const DWORD n=GetModuleFileNameW(nullptr,executable,32768);
        if (!n || n>=32768) {b.error="QFN_ATTENTION_DLL_PATH: cannot resolve executable path";return b;}
        std::wstring path(executable,n);const auto slash=path.find_last_of(L"\\/");
        if (slash==std::wstring::npos) {b.error="QFN_ATTENTION_DLL_PATH: executable path must be absolute";return b;}
        path.resize(slash+1);path+=L"qfn-basalt-nvfp4.dll";
        const auto library=LoadLibraryExW(path.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!library) {b.error="QFN_ATTENTION_DLL_MISSING: adjacent DLL load failed, Windows error "+std::to_string(GetLastError());return b;}
        const auto abi=reinterpret_cast<decltype(&qfn_basalt_attn_abi)>(GetProcAddress(library,"qfn_basalt_attn_abi"));
        b.workspace=reinterpret_cast<decltype(b.workspace)>(GetProcAddress(library,"qfn_basalt_attn_workspace_bytes"));
        b.prepare=reinterpret_cast<decltype(b.prepare)>(GetProcAddress(library,"qfn_basalt_attn_prepare"));
        b.run=reinterpret_cast<decltype(b.run)>(GetProcAddress(library,"qfn_basalt_attn_run"));
        if (!abi || abi()!=1 || !b.workspace || !b.prepare || !b.run)
            b.error="QFN_ATTENTION_ABI: explicit Basalt attention requires independent attention ABI1/prepare/query/run";
#else
        b.error="QFN_ATTENTION_PLATFORM: explicit Basalt attention requires Windows CUDA SM120";
#endif
        return b;
    }();return result;
}
extern "C" int ggml_backend_cuda_qfn_attention_available(char * error,size_t capacity) {
    const auto & b=qfn_attention_functions();if (error && capacity) {
        const size_t n=std::min(capacity-1,b.error.size());std::memcpy(error,b.error.data(),n);error[n]=0;
    }return b.error.empty();
}
extern "C" uint64_t ggml_backend_cuda_qfn_attention_workspace_bytes(int tokens,int heads,int kv_heads,int n_kv,int cap,float scale) {
    const auto & b=qfn_attention_functions();if (!b.error.empty()) return UINT64_MAX;
    qfn_basalt_attn_desc d{};d.mode=tokens<=4?QFN_ATTN_DECODE64:QFN_ATTN_PROMPT;
    d.nt=tokens;d.n_heads=heads;d.n_kv_heads=kv_heads;d.n_kv=n_kv;d.cap=cap;d.scale=scale;
    d.q_head_stride=d.gate_head_stride=d.out_head_stride=1024;
    d.q_token_stride=d.gate_token_stride=d.out_token_stride=uint64_t(heads)*1024;
    d.k_head_stride=d.v_head_stride=512;d.k_cell_stride=d.v_cell_stride=uint64_t(kv_heads)*512;
    d.ids_token_stride=uint64_t(cap)*4;return b.workspace(&d);
}
extern "C" int ggml_backend_cuda_qfn_attention_prepare(void * stream) {
    const auto & b=qfn_attention_functions();return b.error.empty()?b.prepare(stream):int(cudaErrorSharedObjectSymbolNotFound);
}
static void ggml_cuda_op_qfn_basalt_attention(ggml_backend_cuda_context & ctx,ggml_tensor * t) {
    GGML_ASSERT(ggml_qfn_attention_valid(t) && ggml_cuda_info().devices[ctx.device].cc==1200);
    const auto & b=qfn_attention_functions();GGML_ASSERT(b.error.empty());
    const auto * q=t->src[0],* gate=t->src[1],* k=t->src[2],* v=t->src[3],* ids=t->src[4],* pos=t->src[5],* scratch=t->src[6];
    qfn_basalt_attn_desc d{};d.nt=int(q->ne[2]);d.n_heads=int(q->ne[1]);d.n_kv_heads=int(k->ne[1]);
    d.n_kv=int(k->ne[2]);d.cap=int(ids->ne[0]);d.scale=ggml_get_op_params_f32(t,0);
    d.mode=d.nt<=4?QFN_ATTN_DECODE64:QFN_ATTN_PROMPT;
    d.q_head_stride=q->nb[1];d.q_token_stride=q->nb[2];
    d.gate_head_stride=gate?gate->nb[1]:0;d.gate_token_stride=gate?gate->nb[2]:0;
    d.k_head_stride=k->nb[1];d.k_cell_stride=k->nb[2];d.v_head_stride=v->nb[1];d.v_cell_stride=v->nb[2];
    d.ids_token_stride=ids->nb[1];d.out_head_stride=t->nb[1];d.out_token_stride=t->nb[2];
    const uint64_t needed=b.workspace(&d);GGML_ASSERT(needed!=UINT64_MAX && (!needed || (scratch && needed<=uint64_t(scratch->ne[0]))));
    CUDA_CHECK(static_cast<cudaError_t>(b.run(static_cast<const float *>(q->data),gate?static_cast<const float *>(gate->data):nullptr,
        k->data,v->data,static_cast<const int32_t *>(ids->data),static_cast<const int32_t *>(pos->data),static_cast<float *>(t->data),
        &d,scratch?scratch->data:nullptr,scratch?uint64_t(scratch->ne[0]):0,ctx.stream())));
    CUDA_CHECK(cudaGetLastError());
}
