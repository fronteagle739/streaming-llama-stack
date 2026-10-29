#include "llama-dsv41-ffn.h"
#include "boundary-capture.h"
#include "ggml-ds4-native-half.h"
#include "llama-dsv41-flashinfer-budget.h"
#include <cmath>

llama_dsv41_ffn_graph::llama_dsv41_ffn_graph(ggml_context * ctx,llama_dsv41_graph_ops & ops,float limit,float route_scale) :
    ctx(ctx),ops(ops),limit(limit),route_scale(route_scale) {
    GGML_ASSERT(ctx && std::isfinite(limit) && limit>0 && std::isfinite(route_scale) && route_scale>0);
}
ggml_tensor * llama_dsv41_ffn_graph::mm(ggml_tensor * w,ggml_tensor * input,bool round) const {
    auto * out=ggml_mul_mat(ctx,w,input);ggml_mul_mat_set_prec(out,GGML_PREC_F32);return round?ops.bf16(out):out;
}
ggml_tensor * llama_dsv41_ffn_graph::swiglu(ggml_tensor * gate,ggml_tensor * up) const {
    // Native clamps the gate BEFORE SiLU, only above; up is clamped on both
    // sides. Clamping SiLU itself is a different operation for negative gates.
    gate=ggml_clamp(ctx,gate,-INFINITY,limit);up=ggml_clamp(ctx,up,-limit,limit);
    return ggml_swiglu_split(ctx,gate,up);
}

llama_dsv41_ffn_graph::result llama_dsv41_ffn_graph::build(
        const weights & w,ggml_tensor * input,ggml_tensor * token_kinds) const {
    return build_impl(w,input,token_kinds,false);
}
llama_dsv41_ffn_graph::result llama_dsv41_ffn_graph::build_native_half(
        const weights & w,ggml_tensor * input,ggml_tensor * token_kinds) const {
    return build_impl(w,input,token_kinds,true);
}
llama_dsv41_ffn_graph::result llama_dsv41_ffn_graph::routed_impl(
        const weights & w,ggml_tensor * input,ggml_tensor * token_kinds,bool native_half,bool native_router_partitions) const {
    const int64_t count=input->ne[1],width=input->ne[0],experts=w.router->ne[1];
    constexpr int64_t used=6;
    GGML_ASSERT(input->type==GGML_TYPE_F32 && input->ne[2]==1 && input->ne[3]==1 && count>0 && count<=(native_half?4096:16384));
    GGML_ASSERT(experts==384 && w.router->ne[0]==width && w.text_bias && ggml_nelements(w.text_bias)==experts);
    GGML_ASSERT(token_kinds && token_kinds->type==GGML_TYPE_I32 && ggml_nelements(token_kinds)==count);
    // The original vision-capable checkpoint supplies both bias vectors. Fail
    // if one is missing rather than silently routing image rows as text.
    GGML_ASSERT(w.image_bias && ggml_nelements(w.image_bias)==experts);
    ggml_tensor * bias=ggml_concat(ctx,ggml_reshape_2d(ctx,w.text_bias,experts,1),
        ggml_reshape_2d(ctx,w.image_bias,experts,1),1);
    bias=ggml_get_rows(ctx,bias,token_kinds);
    // Expert grouping must not implicitly widen the sensitive router GEMM.
    // Reuse the original attention-sized F32 products before joining logits.
    ggml_tensor * router_logits=nullptr;
    const int64_t router_rows=native_router_partitions?2048:4096;
    if((native_router_partitions||llama_dsv41_flashinfer::requested())&&count>router_rows){
        GGML_ASSERT(!native_router_partitions||(native_half&&count<=4096));
        for(int64_t first=0;first<count;first+=router_rows){
            const int64_t rows=std::min<int64_t>(router_rows,count-first);
            auto * part=ggml_view_2d(ctx,input,width,rows,input->nb[1],size_t(first)*input->nb[1]);
            auto * logits=mm(w.router,part,false);
            router_logits=router_logits?ggml_concat(ctx,router_logits,logits,1):logits;
        }
    }else router_logits=mm(w.router,input,false);
    auto * probabilities=ggml_sqrt(ctx,ggml_softplus(ctx,router_logits));
    auto * selected=ggml_cont(ctx,ggml_argsort_top_k(ctx,ggml_add(ctx,probabilities,bias),used));
    ggml_set_name(selected,"dsv41_ffn_selected_experts");
    // Selection includes the bias; route weights deliberately do not.
    auto * weights=ggml_get_rows(ctx,ggml_reshape_3d(ctx,probabilities,1,experts,count),selected);
    weights=ggml_reshape_2d(ctx,weights,used,count);
    auto * sum=ggml_clamp(ctx,ggml_sum_rows(ctx,weights),6.103515625e-5f,INFINITY);
    weights=ggml_scale(ctx,ggml_div(ctx,weights,sum),route_scale);
    ggml_set_name(weights,"dsv41_ffn_selected_weights");
    input=dsv41_boundary::replacement(ctx,input,"expert_input");
    selected=dsv41_boundary::replacement(ctx,selected,"route_ids");
    weights=dsv41_boundary::replacement(ctx,weights,"route_weights");
    dsv41_boundary::stage(selected,"route_ids");
    dsv41_boundary::stage(weights,"route_weights");
    auto * x=ggml_reshape_3d(ctx,input,width,1,count);
    ggml_tensor * product_ids=selected;
    if(w.expert_slots){
        GGML_ASSERT(w.expert_slots->type==GGML_TYPE_I32&&w.expert_slots->ne[0]==used&&w.expert_slots->ne[1]==count);
        product_ids=ggml_cpy(ctx,selected,w.expert_slots);ggml_set_name(product_ids,"dsv41_expert_slot_copy");
    }
    auto * gate=ggml_mul_mat_id(ctx,w.gate,x,product_ids);
    auto * up=ggml_mul_mat_id(ctx,w.up,x,product_ids);
    if(native_half){
        GGML_ASSERT(ggml_ds4_native_half_tag(gate,GGML_DS4_NATIVE_HALF_GATE));
        GGML_ASSERT(ggml_ds4_native_half_tag(up,GGML_DS4_NATIVE_HALF_UP));
    }
    auto * mid=swiglu(gate,up);
    // Native Q4_K pair-SwiGLU stores half(silu(gate)*up*route_weight).
    // Round weighted mid to F16 storage, then expose F32 to the same down MMID.
    mid=ggml_mul(ctx,mid,ggml_reshape_3d(ctx,weights,1,used,count));
    mid=ggml_cast(ctx,ggml_cast(ctx,mid,GGML_TYPE_F16),GGML_TYPE_F32);
    auto * all=ggml_mul_mat_id(ctx,w.down,mid,product_ids);
    if(native_half)GGML_ASSERT(ggml_ds4_native_half_tag(all,GGML_DS4_NATIVE_HALF_DOWN));
    ggml_tensor * routed=nullptr;
    for(int64_t i=0;i<used;++i){
        auto * one=ggml_view_2d(ctx,all,width,count,all->nb[2],i*all->nb[1]);
        routed=routed?ggml_add(ctx,routed,one):one;
    }
    dsv41_boundary::stage(routed,"routed_ffn_f32");
    return {routed,selected,weights};
}

ggml_tensor *llama_dsv41_ffn_graph::build_shared(const weights &w,ggml_tensor *input) const {
    GGML_ASSERT(input->ne[1]>0&&input->ne[1]<=4096);
    auto * shared_gate=mm(w.shared_gate,input,true);
    auto * shared_up=mm(w.shared_up,input,true);
    auto * shared_mid=ops.bf16(swiglu(shared_gate,shared_up));
    auto * shared=mm(w.shared_down,shared_mid,true);
    dsv41_boundary::stage(shared,"shared_ffn_bf16");return shared;
}

llama_dsv41_ffn_graph::result llama_dsv41_ffn_graph::build_routed_pair(
        const weights &w,ggml_tensor *input,ggml_tensor *kinds,bool native_half)const{
    return routed_impl(w,input,kinds,native_half,native_half);
}

llama_dsv41_ffn_graph::result llama_dsv41_ffn_graph::build_impl(
        const weights &w,ggml_tensor *input,ggml_tensor *kinds,bool native_half)const{
    GGML_ASSERT(input->ne[1]>0&&input->ne[1]<=4096);
    const auto routed=routed_impl(w,input,kinds,native_half);
    auto *shared=build_shared(w,input);
    return {ggml_add(ctx,routed.output,shared),routed.selected,routed.route_weights};
}
