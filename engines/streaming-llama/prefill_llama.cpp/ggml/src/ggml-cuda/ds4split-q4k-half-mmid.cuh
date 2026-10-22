#pragma once
// SOURCE candidate. Explicit entry only; original operator/dispatch is untouched.
#include "common.cuh"
#include "ds4split-q4k-half-mmid-plan.h"
struct ds4split_q4k_half_launch {
    int kernel_family=0,device=-1,cc=0,nwarps=0,rows_per_block=0,columns_per_block=0;
    int64_t tokens=0,experts=0,maximum_tasks=0;
    size_t workspace_requested_bytes=0,shared_bytes=0;
};
bool ggml_cuda_supports_q4k_half_mmid(const ggml_tensor*,const ggml_tensor*,const ggml_tensor*,const ggml_tensor*,int);
// Same actual map/task stages as the operator, exposed for bounded map-only cases.
void ggml_cuda_q4k_half_prepare_tasks(const int32_t*,int,int,int,void*,size_t,cudaStream_t);
// Caller-owned workspace makes model-free map/task readback possible before reuse.
// This entry performs no host readback, stream synchronization or full-matrix expansion.
void ggml_cuda_mul_mat_q4k_half_id_workspace(ggml_backend_cuda_context&,const ggml_tensor*,const ggml_tensor*,const ggml_tensor*,ggml_tensor*,void*,size_t,ds4split_q4k_half_launch*);
// Explicit four-expert panel + once-per-product shared half-RHS gather.
void ggml_cuda_mul_mat_q4k_half_panel_id_workspace(ggml_backend_cuda_context&,const ggml_tensor*,const ggml_tensor*,const ggml_tensor*,ggml_tensor*,void*,size_t,ds4split_q4k_half_launch*);
void ggml_cuda_mul_mat_q4k_half_id(ggml_backend_cuda_context&,const ggml_tensor*,const ggml_tensor*,const ggml_tensor*,ggml_tensor*,ds4split_q4k_half_launch*);
// Model-free proof only: same device loader, natural K-fastest F16 output.
void ggml_cuda_q4k_half_loader_golden(const void*,half*,int64_t,int64_t,cudaStream_t);
