// Vendored from streaming-llama ggml/src/ggml-cuda (mmf.cuh helpers, convert.cuh cast,
// ds4split-q4k-half-mmid-plan.h) for the native-half Q4_K MMID port. Only the pieces the
// kernels use; arithmetic unchanged.
#pragma once
#include "common.cuh"
#include "mma.cuh"
#include "mmid.cuh"
#include <cstdint>
#include <stdexcept>
using namespace ggml_cuda_mma;
/* The ds4 stub's GGML_UNUSED_VARS takes at most 12 names; the kernels' non-MMA branch
 * passes 22. Use ggml.h's own C fallback form here (no code generated). */
#undef GGML_UNUSED_VARS
#define GGML_UNUSED_VARS(...) do { (void)sizeof((__VA_ARGS__, 0)); } while (0)
#define MMF_ROWS_PER_BLOCK 32
#define MMF_ROWS_PER_BLOCK_CDNA 64
static __forceinline__ int64_t mmf_get_max_block_size(int cc) { return GGML_CUDA_CC_IS_CDNA(cc) ? 512 : 256; }
static __forceinline__ int mmf_get_padding(int cc) { return GGML_CUDA_CC_IS_CDNA(cc) ? 2 : 4; }
static constexpr __device__ int mmf_get_padding() {
#if defined(AMD_MFMA_AVAILABLE)
    return 2;
#else
    return 4;
#endif
}
// Pure bounds and logical layout for the explicit Q4_K/half-RHS MMID family.
#include <cstdint>
#include <stdexcept>
namespace ds4split_q4k_half {
constexpr int task_columns=16,maximum_tokens=4096,maximum_experts=384,selected_ranks=6;
#if defined(__CUDACC__)
#define DS4SPLIT_Q4K_COORD_HD __host__ __device__
#else
#define DS4SPLIT_Q4K_COORD_HD
#endif
struct pair_coordinates { int block,group,payload; };
DS4SPLIT_Q4K_COORD_HD constexpr pair_coordinates natural_pair(int p){
    return {p/256,(p%256)/32,((p%256)/64)*32+(p%32)};
}
#undef DS4SPLIT_Q4K_COORD_HD
struct workspace_plan {
    uint64_t slots,maximum_tasks,src_map,dst_map,expert_bounds,task_expert,task_col_base,task_count,bytes;
};
inline workspace_plan plan(int64_t tokens,int64_t experts){
    if(tokens<1||tokens>maximum_tokens||experts<selected_ranks||experts>maximum_experts)throw std::runtime_error("Q4_K half MMID explicit token/expert bounds");
    const uint64_t slots=uint64_t(tokens)*selected_ranks,nonempty=slots<uint64_t(experts)?slots:uint64_t(experts);
    const uint64_t maximum_tasks=(slots+uint64_t(task_columns-1)*nonempty)/task_columns;
    workspace_plan p{};p.slots=slots;p.maximum_tasks=maximum_tasks;p.src_map=0;p.dst_map=slots*4;p.expert_bounds=p.dst_map+slots*4;
    p.task_expert=p.expert_bounds+uint64_t(experts+1)*4;p.task_col_base=p.task_expert+maximum_tasks*4;p.task_count=p.task_col_base+maximum_tasks*4;p.bytes=p.task_count+4;
    return p;
}
}
