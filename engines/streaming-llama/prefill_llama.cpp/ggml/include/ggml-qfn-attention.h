// SPDX-License-Identifier: MIT
#pragma once
#include "ggml-backend.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

inline bool ggml_qfn_attention_requested(const ggml_tensor * t) {
    return t && t->op==GGML_OP_QFN_QSA_ATTN && t->op_params[1]==0x51415341;
}
inline bool ggml_qfn_attention_span(const ggml_tensor * t,size_t alignment) {
    if (!t || (t->data && (uintptr_t(t->data)&(alignment-1)))) return false;
    uint64_t span=ggml_type_size(t->type);
    for (int i=0;i<4;++i) {
        if (t->ne[i]<0) return false;
        if (!t->ne[i]) return true;
        const uint64_t n=uint64_t(t->ne[i]-1);
        if (n && t->nb[i]>(uint64_t(PTRDIFF_MAX)-span)/n) return false;
        span+=n*t->nb[i];
    }
    return !t->data || uintptr_t(t->data)<=UINTPTR_MAX-span;
}
inline bool ggml_qfn_attention_valid(const ggml_tensor * t) {
    if (!ggml_qfn_attention_requested(t) || t->type!=GGML_TYPE_F32 || t->ne[0]!=256 ||
        t->ne[1]<1 || t->ne[2]<1 || t->ne[2]>INT32_MAX || t->ne[3]!=1 || t->nb[0]!=4 ||
        (t->nb[1]|t->nb[2])&7 || t->nb[1]<1024 ||
        uint64_t(t->ne[1]-1)>(UINT64_MAX-1024)/t->nb[1] ||
        t->nb[2]<uint64_t(t->ne[1]-1)*t->nb[1]+1024 || !ggml_qfn_attention_span(t,8)) return false;
    const auto * q=t->src[0],* gate=t->src[1],* k=t->src[2],* v=t->src[3],* ids=t->src[4],* pos=t->src[5];
    if (!q || !k || !v || !ids || !pos || q->type!=GGML_TYPE_F32 ||
        k->type!=GGML_TYPE_F16 || v->type!=GGML_TYPE_F16 || ids->type!=GGML_TYPE_I32 || pos->type!=GGML_TYPE_I32 ||
        !ggml_are_same_shape(q,t) || q->nb[0]!=4 || (q->nb[1]|q->nb[2])&15 ||
        k->ne[0]!=256 || k->ne[1]<1 || k->ne[1]>65535 || k->ne[2]<0 || k->ne[2]>INT32_MAX || k->ne[3]!=1 ||
        int64_t(k->ne[1])*12!=q->ne[1] || !ggml_are_same_shape(k,v) || k->nb[0]!=2 || v->nb[0]!=2 ||
        (k->nb[1]|k->nb[2]|v->nb[1]|v->nb[2])&15 ||
        ids->ne[0]<0 || ids->ne[0]>INT32_MAX || ids->ne[1]!=q->ne[2] || ids->ne[2]!=1 || ids->ne[3]!=1 ||
        ids->nb[0]!=4 || (ids->nb[1]&3) || !ggml_is_contiguous(pos) || pos->ne[0]!=q->ne[2] ||
        pos->ne[1]!=1 || pos->ne[2]!=1 || pos->ne[3]!=1 ||
        !ggml_qfn_attention_span(q,16) || !ggml_qfn_attention_span(k,16) || !ggml_qfn_attention_span(v,16) ||
        !ggml_qfn_attention_span(ids,4) || !ggml_qfn_attention_span(pos,4)) return false;
    if (gate && (gate->type!=GGML_TYPE_F32 || !ggml_are_same_shape(gate,q) || gate->nb[0]!=4 ||
        (gate->nb[1]|gate->nb[2])&3 || !ggml_qfn_attention_span(gate,4))) return false;
    float scale;std::memcpy(&scale,t->op_params,sizeof(scale));if (!std::isfinite(scale)) return false;
    if (q->ne[2]>4) return !t->src[6]; // published prompt has no scratch field
    const auto * scratch=t->src[6];
    if (!scratch || scratch->type!=GGML_TYPE_I8 || !ggml_is_contiguous(scratch) || scratch->ne[0]<1 ||
        scratch->ne[1]!=1 || scratch->ne[2]!=1 || scratch->ne[3]!=1 || !ggml_qfn_attention_span(scratch,256)) return false;
    const uint64_t chunks=(uint64_t(ids->ne[0])+63)/64;
    if (chunks*uint64_t(q->ne[1])>INT32_MAX) return false;
    const uint64_t need=(ids->ne[0] && k->ne[2]) ? uint64_t(q->ne[2])*4*(chunks*uint64_t(q->ne[1])*258+64) : 0;
    return uint64_t(scratch->ne[0])>=need;
}
inline ggml_tensor * ggml_qfn_attention(ggml_context * ctx,ggml_tensor * q,ggml_tensor * gate,
    ggml_tensor * k,ggml_tensor * v,ggml_tensor * ids,ggml_tensor * pos,float scale,ggml_tensor * scratch) {
    auto * t=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,q->ne[0],q->ne[1],q->ne[2]);
    t->op=GGML_OP_QFN_QSA_ATTN;std::memcpy(t->op_params,&scale,sizeof(scale));t->op_params[1]=0x51415341;
    t->src[0]=q;t->src[1]=gate;t->src[2]=k;t->src[3]=v;t->src[4]=ids;t->src[5]=pos;t->src[6]=scratch;
    GGML_ASSERT(ggml_qfn_attention_valid(t));return t;
}
extern "C" GGML_BACKEND_API int ggml_backend_cuda_qfn_attention_available(char * error,size_t capacity);
extern "C" GGML_BACKEND_API uint64_t ggml_backend_cuda_qfn_attention_workspace_bytes(int tokens,int heads,int kv_heads,int n_kv,int cap,float scale);
extern "C" GGML_BACKEND_API int ggml_backend_cuda_qfn_attention_prepare(void * stream);
