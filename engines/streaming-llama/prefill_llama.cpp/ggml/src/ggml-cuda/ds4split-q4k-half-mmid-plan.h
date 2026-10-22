#pragma once
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
