// SPDX-License-Identifier: MIT
#pragma once
#include "ggml-backend.h"
#include <cstdint>
#include <cstring>
#include <limits>

// Raw Basalt NVFP4 expert rows. This is a narrowly tagged operation, not a ggml quantization type.
// x[F32,E,T], ids[I32,K,T], blobs[I8,2764816,physical_slots], persistent scratch[I8].
// Result F32[E,K,T] is UNWEIGHTED and in the original router-entry order.
inline bool ggml_qfn_nvfp4_requested(const ggml_tensor * t) {
    return t && t->op == GGML_OP_CUSTOM && t->op_params[0] == 0x514e4634 && t->op_params[1] == 1;
}
inline bool ggml_qfn_nvfp4_valid(const ggml_tensor * t) {
    if (!ggml_qfn_nvfp4_requested(t) || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) ||
        t->ne[0] != 2560 || t->ne[1] != 10 || t->ne[2] < 1 || t->ne[2] > INT32_MAX || t->ne[3] != 1 ||
        t->op_params[2] < 0 || t->op_params[2] > 6 || t->op_params[3] < 0 || t->op_params[3] > 1 ||
        (t->op_params[2] == 6 && t->op_params[3] != 1)) return false;
    const auto * x = t->src[0], * ids = t->src[1], * blobs = t->src[2], * scratch = t->src[3];
    if (!x || !ids || !blobs || !scratch || x->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
        blobs->type != GGML_TYPE_I8 || scratch->type != GGML_TYPE_I8 ||
        x->ne[0] != 2560 || x->ne[1] != t->ne[2] || x->ne[2] != 1 || x->ne[3] != 1 ||
        ids->ne[0] != 10 || ids->ne[1] != t->ne[2] || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        blobs->ne[0] != 2764816 || blobs->ne[1] < 1 || uint64_t(blobs->ne[1]) > UINT32_MAX ||
        blobs->ne[2] != 1 || blobs->ne[3] != 1 || scratch->ne[0] < 1 || scratch->ne[1] != 1 ||
        scratch->ne[2] != 1 || scratch->ne[3] != 1 || !ggml_is_contiguous(x) || !ggml_is_contiguous(ids) ||
        !ggml_is_contiguous(blobs) || !ggml_is_contiguous(scratch)) return false;
    if (t->op_params[2] == 4 || t->op_params[2] == 5 || t->op_params[2] == 6) {
        const auto * scales = t->src[4];
        if (!scales || scales->type != GGML_TYPE_F32 || scales->ne[0] != 3 || scales->ne[1] != 512 ||
            scales->ne[2] != 1 || scales->ne[3] != 1 || !ggml_is_contiguous(scales) ||
            (scales->data && (reinterpret_cast<uintptr_t>(scales->data) & 3))) return false;
    } else if (t->src[4]) return false; // an unexpected calibration field is never ignored
    if (t->op_params[2] == 5 || (t->op_params[2] == 6 && t->src[5])) {
        const auto * logical=t->src[5];
        if(t->ne[2]>4 || !logical || logical->type!=GGML_TYPE_I32 || !ggml_is_contiguous(logical) ||
           logical->ne[0]!=10 || logical->ne[1]!=t->ne[2] || logical->ne[2]!=1 || logical->ne[3]!=1)
            return false;
    } else if(t->src[5])return false; // unexpected logical calibration IDs are never ignored
    const bool w4a4 = t->op_params[2] == 6 || t->op_params[2] == 4 || t->op_params[2] == 3 || t->op_params[2] == 1 || (t->op_params[2] == 0 && t->ne[2] >= 1024);
    return !w4a4 || blobs->ne[1] == 512 || (t->op_params[2] == 6 && t->ne[2] <= 4 && t->src[5]);
}
inline ggml_tensor * ggml_qfn_nvfp4_experts(ggml_context * ctx, ggml_tensor * x, ggml_tensor * ids,
                                          ggml_tensor * blobs, ggml_tensor * scratch, int mode = 0, ggml_tensor * input_scale = nullptr, ggml_tensor * logical_ids = nullptr, bool packed = false) {
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    if (!ggml_is_contiguous(ids)) ids = ggml_cont(ctx, ids);
    if (logical_ids && !ggml_is_contiguous(logical_ids)) logical_ids = ggml_cont(ctx, logical_ids);
    auto * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2560, ids->ne[0], x->ne[1]);
    t->op = GGML_OP_CUSTOM;
    t->op_params[0] = 0x514e4634;
    t->op_params[1] = 1;
    t->op_params[2] = mode;
    t->op_params[3] = packed ? 1 : 0;
    t->src[0] = x; t->src[1] = ids; t->src[2] = blobs; t->src[3] = scratch; t->src[4] = input_scale; t->src[5] = logical_ids;
    GGML_ASSERT(ggml_qfn_nvfp4_valid(t));
    return t;
}

// Preflight only loads/checks the fixed adjacent DLL; prepare is called once before any graph capture.
extern "C" GGML_BACKEND_API int ggml_backend_cuda_qfn_nvfp4_available(char * error, size_t capacity);
extern "C" GGML_BACKEND_API int ggml_backend_cuda_qfn_nvfp4_calibrated_available(char * error, size_t capacity);
extern "C" GGML_BACKEND_API int ggml_backend_cuda_qfn_nvfp4_calibrated_decode_available(char * error, size_t capacity);
extern "C" GGML_BACKEND_API uint64_t ggml_backend_cuda_qfn_nvfp4_workspace_bytes(int tokens, int top_k,
                                                                               uint32_t slots, int mode);
extern "C" GGML_BACKEND_API int ggml_backend_cuda_qfn_nvfp4_prepare(void * stream);
extern "C" GGML_BACKEND_API int ggml_backend_cuda_qfn_native_nvfp4_available(char * error, size_t capacity);
extern "C" GGML_BACKEND_API uint64_t ggml_backend_cuda_qfn_native_nvfp4_workspace_bytes(int tokens, uint32_t slots, int mode);
extern "C" GGML_BACKEND_API int ggml_backend_cuda_qfn_native_nvfp4_prepare(void * stream);
