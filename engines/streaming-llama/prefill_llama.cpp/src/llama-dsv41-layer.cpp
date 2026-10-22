#include "llama-dsv41-layer.h"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

llama_dsv41_layer_graph::llama_dsv41_layer_graph(ggml_context * ctx,ggml_cgraph * graph,llama_dsv41_graph_ops & ops,
        float swiglu_limit,float expert_scale) : ctx(ctx),graph(graph),ops(ops),swiglu_limit(swiglu_limit),expert_scale(expert_scale) {
    GGML_ASSERT(ctx && graph);
}
ggml_tensor * llama_dsv41_layer_graph::publish(ggml_tensor * persistent,ggml_tensor * rows,ggml_tensor * ids) const {
    GGML_ASSERT(persistent && rows && ids && (persistent->type==GGML_TYPE_F32||persistent->type==GGML_TYPE_BF16) && rows->type==GGML_TYPE_F32);
    if(persistent->type==GGML_TYPE_BF16){
        // Only the existing compressor QAT result may enter compact storage.
        // Its final BF16 rounding is part of both original QAT modes already.
        const ggml_tensor*producer=rows;
        while(producer->op==GGML_OP_RESHAPE||producer->op==GGML_OP_VIEW)producer=producer->src[0];
        GGML_ASSERT(producer&&producer->op==GGML_OP_DSV4_ACT_QAT&&
            (producer->op_params[0]==GGML_DSV41_QAT_FP4_E8M0||producer->op_params[0]==GGML_DSV41_QAT_FP4_E4M3));
    }
    GGML_ASSERT(ids->type==GGML_TYPE_I64 && ggml_nelements(ids)==rows->ne[1]);
    auto * output=ggml_set_rows(ctx,persistent,rows,ids);
    output->data=persistent->data;output->buffer=persistent->buffer;
    return output;
}
ggml_tensor * llama_dsv41_layer_graph::prefix(ggml_tensor * persistent,uint32_t rows,bool preserve_storage) const {
    GGML_ASSERT(persistent && rows>0 && rows<=persistent->ne[1] && persistent->ne[2]==1 && persistent->ne[3]==1);
    auto * view=ggml_view_2d(ctx,persistent,persistent->ne[0],rows,persistent->nb[1],0);
    view->data=persistent->data;view->buffer=persistent->buffer;
    return !preserve_storage&&view->type==GGML_TYPE_BF16?ggml_cast(ctx,view,GGML_TYPE_F32):view;
}
ggml_tensor * llama_dsv41_layer_graph::keep(ggml_tensor * value) const {
    if(value){
        // Pin a materialized result, not just a reshape/view whose base can be
        // recycled before the executor's bounded GPU/SSD carry transfer.
        value=ggml_cont(ctx,value);ggml_set_output(value);ggml_build_forward_expand(graph,value);
    }
    return value;
}

llama_dsv41_layer_graph::result llama_dsv41_layer_graph::build(uint32_t layer,const llama_dsv41_tile_plan & p,
        uint32_t compressed_rows,const weights & w,const inputs & in,const cache & state) const {
    return build(layer,p,compressed_rows,w,in,state,{});
}
llama_dsv41_layer_graph::attention_part llama_dsv41_layer_graph::build_attention(uint32_t layer,const llama_dsv41_tile_plan & p,
        uint32_t compressed_rows,const weights & w,const inputs & in,const cache & state,
        const llama_dsv41_expert_profile::policy & expert) const {
    llama_dsv41_expert_profile::validate(expert,p.count,std::getenv("DSV41_BOUNDARY_CAPTURE"));
    GGML_ASSERT(layer<40 && p.ratio==(layer<2?0u:layer<20?2u:1u));
    GGML_ASSERT(in.residual->ne[0]==5120 && in.residual->ne[1]==4 && in.residual->ne[2]==p.count);
    GGML_ASSERT(in.previous_pre && in.previous_pre->ne[0]==4 && in.previous_pre->ne[1]==p.count);
    GGML_ASSERT(p.ratio!=0 || compressed_rows==0);
    if(p.ratio) GGML_ASSERT(compressed_rows>=uint32_t(p.compressed_visible.back()));
    ggml_tensor * residual=in.residual;
    if(layer==1 || layer==14){
        GGML_ASSERT(in.engram_rows && in.text_mask && w.engram_wkv && w.engram_q_norm && w.engram_k_norm);
        residual=ops.engram(residual,in.engram_rows,w.engram_wkv,w.engram_q_norm,w.engram_k_norm,in.text_mask);
    }
    ggml_tensor * capture=layer>=37?ops.mean_for_capture(residual):nullptr;
    const auto attn=ops.prepare(residual,in.previous_pre,w.attn_fn,w.attn_scale,w.attn_base,w.attn_norm);
    llama_dsv41_attention_graph attention(ctx,ops);
    auto * frequencies=p.ratio?in.compressed_frequencies:in.raw_frequencies;
    const auto projected=attention.project(w.attention,attn.input,in.positions,frequencies);
    ggml_tensor * compressed=state.compressed,*index=state.index;
    ggml_tensor * next_kv=nullptr,*next_score=nullptr;
    const bool source=layer==2 || layer==8 || layer==14 || layer==20;
    const bool query_source=source || layer==24 || layer==28 || layer==32 || layer==36;
    if(source){
        const auto publication=attention.compress(w.attention,attn.input,p,state.previous_pool_kv,state.previous_pool_score,
            in.publication_positions,in.compressed_frequencies);
        next_kv=publication.next_pool_kv;next_score=publication.next_pool_score;
        if(p.published_count){
            // Appended rows cannot overlap any previous visible compressed
            // row. Reads depend on these aliased writes via their graph input.
            GGML_ASSERT(p.published_first+p.published_count<=compressed_rows);
            compressed=publish(compressed,publication.kv,in.publication_ids);
            index=publish(index,publication.index,in.publication_ids);
        }
    }
    ggml_tensor * selected=nullptr,*selected_mask=nullptr,*candidates=in.candidate_ids;
    if(compressed_rows){
        GGML_ASSERT(in.visible && compressed && index);
        if(query_source){
            const auto iq=attention.project_index(w.attention,projected.qr,attn.input,in.positions,in.compressed_frequencies);
            const auto selection=ops.index_select(iq.q,iq.weights,prefix(index,compressed_rows,expert.index_key_windows&&compressed_rows>131072),in.visible,layer,
                layer>20?in.candidate_ids:nullptr,expert.index_key_windows?graph:nullptr);
            selected=selection.rows;selected_mask=selection.mask;
            if(layer==20)candidates=selection.candidates;
        }else{
            selected=in.previous_selected;
            GGML_ASSERT(selected && (layer<24 || in.candidate_ids));
            // Carry only compact IDs between layers. Reconstruct this tiny
            // selected-row mask from causal visibility and source20 candidates.
            selected_mask=ops.selection_mask(selected,in.visible,layer>=24?in.candidate_ids:nullptr,compressed_rows);
        }
    }
    auto * raw=ops.raw_snapshot(state.raw,projected.raw,in.history_ids);
    const char * gather_half=std::getenv("LLAMA_DSV41_GATHER_HALF");
    const bool preserve_gather_storage=gather_half&&!std::strcmp(gather_half,"1");
    auto * heads=ops.attention_heads(projected.q,raw,in.raw_ids,in.raw_mask,
        compressed_rows?prefix(compressed,compressed_rows,expert.attention_kv_f16_table||preserve_gather_storage):nullptr,selected,selected_mask,w.attention.attn_sinks,
        llama_dsv41_expert_profile::shared_sparse(expert),llama_dsv41_expert_profile::shared_flash_attention(expert),
        expert.token_tile_attention,p.start,expert.attention_kv_f16_table);
    // This exact head result is the dependency protecting the old raw ring.
    auto *committed=ops.raw_commit(state.raw,projected.raw,heads,p.start);
    ggml_build_forward_expand(graph,committed);
    auto * after_attn=ops.complete(attention.finish(w.attention,heads,in.positions,frequencies),attn);
    const auto ffn=ops.prepare(after_attn,attn.next.pre,w.ffn_fn,w.ffn_scale,w.ffn_base,w.ffn_norm);
    cache after{committed,compressed,index,state.previous_pool_kv,state.previous_pool_score};
    return {ffn,selected,selected_mask,candidates,next_kv,next_score,capture,after};
}

llama_dsv41_layer_graph::result llama_dsv41_layer_graph::build(uint32_t layer,const llama_dsv41_tile_plan &p,
        uint32_t compressed_rows,const weights &w,const inputs &in,const cache &state,
        const llama_dsv41_expert_profile::policy &expert)const{
    const auto part=build_attention(layer,p,compressed_rows,w,in,state,expert);
    const auto &ffn=part.ffn;
    llama_dsv41_ffn_graph feed_forward(ctx,ops,swiglu_limit,expert_scale);
    const auto block=llama_dsv41_expert_profile::natural_half(expert)?
        feed_forward.build_native_half(w.ffn,ffn.input,in.token_kinds):feed_forward.build(w.ffn,ffn.input,in.token_kinds);
    auto * output=ops.complete(block.output,ffn);
    // Pool tails are copied to persistent state only AFTER graph completion;
    // pinning the raw projection avoids overwriting the preceding odd tail
    // while the current pooling node still consumes it.
    return {keep(output),keep(ffn.next.pre),keep(part.selected),keep(part.mask),keep(part.candidates),
            keep(part.next_kv),keep(part.next_score),keep(part.capture)};
}

llama_dsv41_layer_graph::result llama_dsv41_layer_graph::build_pair(uint32_t layer,const llama_dsv41_tile_plan &p,
        uint32_t compressed_rows,const weights &w,const inputs &in,const cache &state,
        const llama_dsv41_expert_profile::policy &expert)const{
    const bool native_pair=llama_dsv41_expert_profile::natural_half(expert);
    const uint32_t attention_rows=native_pair?2048u:4096u;
    const char * reuse_option=std::getenv("LLAMA_DSV41_REUSE_CHILD_RAW_METADATA");
    GGML_ASSERT(!reuse_option||!std::strcmp(reuse_option,"0")||!std::strcmp(reuse_option,"1"));
    const bool reuse_child=reuse_option&&!std::strcmp(reuse_option,"1");
    // Child attention consumes the parent raw-ID/mask tensor slices below.
    // Reuse the existing planner's exact aligned-window omission mode;
    // all positions, history, publication, visibility and pool tails remain.
    const auto raw_metadata=[&](uint32_t start){return reuse_child&&start>=128&&start%128==0;};
    GGML_ASSERT((native_pair||llama_dsv41_expert_profile::activation_rounding_q8_1(expert))&&p.count>0&&p.count<=(native_pair?4096u:16384u));
    if(p.count<=attention_rows)return build(layer,p,compressed_rows,w,in,state,expert);
    if(p.count>8192){
        GGML_ASSERT(p.start%4096==0&&in.pair_history_ids&&ggml_nelements(in.pair_history_ids)==127);
        auto slice=[&](ggml_tensor*t,int dim,uint32_t begin,uint32_t rows)->ggml_tensor*{
            if(!t)return nullptr;GGML_ASSERT(dim>=0&&dim<=2&&begin+rows<=t->ne[dim]);
            if(dim==0)return ggml_view_1d(ctx,t,rows,size_t(begin)*t->nb[0]);
            if(dim==1)return ggml_view_2d(ctx,t,t->ne[0],rows,t->nb[1],size_t(begin)*t->nb[1]);
            return ggml_view_3d(ctx,t,t->ne[0],t->ne[1],rows,t->nb[1],t->nb[2],size_t(begin)*t->nb[2]);
        };
        std::vector<attention_part> parts;std::vector<uint32_t> counts;cache current=state;
        for(uint32_t offset=0;offset<p.count;offset+=4096){
            const uint32_t count=std::min(4096u,p.count-offset);llama_dsv41_tile_plan child(p.start+offset,count,p.context,p.ratio,4096,raw_metadata(p.start+offset));
            GGML_ASSERT(offset+count==p.count||!child.next_pool_tail);
            inputs v=in;v.residual=slice(in.residual,2,offset,count);v.previous_pre=slice(in.previous_pre,1,offset,count);
            v.positions=slice(in.positions,0,offset,count);v.raw_ids=slice(in.raw_ids,1,offset,count);v.raw_mask=slice(in.raw_mask,1,offset,count);
            if(offset){v.history_ids=in.pair_history_ids;GGML_ASSERT(child.history_rows==127);
                const float shift=-float(offset+p.history_rows-child.history_rows);
                v.raw_ids=ggml_cast(ctx,ggml_scale_bias(ctx,ggml_cast(ctx,v.raw_ids,GGML_TYPE_F32),1.f,shift),GGML_TYPE_I32);}
            v.visible=slice(in.visible,0,offset,count);
            if(p.ratio){const uint32_t at=child.published_first-p.published_first;
                v.publication_ids=slice(in.publication_ids,0,at,child.published_count);v.publication_positions=slice(in.publication_positions,0,at,child.published_count);}
            v.previous_selected=slice(in.previous_selected,1,offset,count);v.candidate_ids=slice(in.candidate_ids,1,offset,count);
            v.engram_rows=slice(in.engram_rows,1,offset,count);v.text_mask=slice(in.text_mask,0,offset,count);v.token_kinds=slice(in.token_kinds,0,offset,count);
            parts.push_back(build_attention(layer,child,compressed_rows,w,v,current,expert));current=parts.back().after;counts.push_back(count);
        }
        auto join=[&](const std::vector<ggml_tensor*>&ts,int dim){ggml_tensor*value=nullptr;bool seen=false,present=false;
            for(auto*t:ts){if(!seen){present=bool(t);seen=true;}GGML_ASSERT(bool(t)==present);if(t)value=value?ggml_concat(ctx,value,t,dim):t;}return value;};
        std::vector<ggml_tensor*> inputs,outputs,pre,selected,masks,candidates,captures;
        for(const auto&part:parts)inputs.push_back(part.ffn.input);
        llama_dsv41_ffn_graph feed_forward(ctx,ops,swiglu_limit,expert_scale);
        const auto routed=feed_forward.build_routed_pair(w.ffn,join(inputs,1),in.token_kinds);
        uint32_t offset=0;
        for(size_t i=0;i<parts.size();++i){const auto&part=parts[i];auto*one=ggml_view_2d(ctx,routed.output,5120,counts[i],routed.output->nb[1],size_t(offset)*routed.output->nb[1]);
            outputs.push_back(ops.complete(ggml_add(ctx,one,feed_forward.build_shared(w.ffn,part.ffn.input)),part.ffn));
            pre.push_back(part.ffn.next.pre);selected.push_back(part.selected);masks.push_back(part.mask);candidates.push_back(part.candidates);captures.push_back(part.capture);offset+=counts[i];}
        return {keep(join(outputs,2)),keep(join(pre,1)),keep(join(selected,1)),keep(join(masks,1)),keep(join(candidates,1)),keep(parts.back().next_kv),keep(parts.back().next_score),keep(join(captures,1))};
    }
    const llama_dsv41_tile_plan first(p.start,attention_rows,p.context,p.ratio,attention_rows,raw_metadata(p.start)),second(p.start+attention_rows,p.count-attention_rows,p.context,p.ratio,attention_rows,raw_metadata(p.start+attention_rows));
    GGML_ASSERT(!first.next_pool_tail&&in.pair_history_ids&&ggml_nelements(in.pair_history_ids)==second.history_rows);
    auto slice=[&](ggml_tensor *t,int dim,uint32_t begin,uint32_t rows)->ggml_tensor*{
        if(!t)return nullptr;GGML_ASSERT(dim>=0&&dim<=2&&begin+rows<=t->ne[dim]);
        if(dim==0)return ggml_view_1d(ctx,t,rows,size_t(begin)*t->nb[0]);
        if(dim==1)return ggml_view_2d(ctx,t,t->ne[0],rows,t->nb[1],size_t(begin)*t->nb[1]);
        return ggml_view_3d(ctx,t,t->ne[0],t->ne[1],rows,t->nb[1],t->nb[2],size_t(begin)*t->nb[2]);
    };
    auto sub=[&](const llama_dsv41_tile_plan &child,uint32_t offset){
        inputs v=in;v.residual=slice(in.residual,2,offset,child.count);v.previous_pre=slice(in.previous_pre,1,offset,child.count);
        v.positions=slice(in.positions,0,offset,child.count);v.raw_ids=slice(in.raw_ids,1,offset,child.count);v.raw_mask=slice(in.raw_mask,1,offset,child.count);
        if(offset){v.history_ids=in.pair_history_ids;
            const float shift=-float(offset+p.history_rows-child.history_rows);
            v.raw_ids=ggml_cast(ctx,ggml_scale_bias(ctx,ggml_cast(ctx,v.raw_ids,GGML_TYPE_F32),1.f,shift),GGML_TYPE_I32);}
        v.visible=slice(in.visible,0,offset,child.count);
        if(p.ratio){const uint32_t at=child.published_first-p.published_first;
            v.publication_ids=slice(in.publication_ids,0,at,child.published_count);v.publication_positions=slice(in.publication_positions,0,at,child.published_count);}
        v.previous_selected=slice(in.previous_selected,1,offset,child.count);v.candidate_ids=slice(in.candidate_ids,1,offset,child.count);
        v.engram_rows=slice(in.engram_rows,1,offset,child.count);v.text_mask=slice(in.text_mask,0,offset,child.count);v.token_kinds=slice(in.token_kinds,0,offset,child.count);
        return v;
    };
    const auto left=build_attention(layer,first,compressed_rows,w,sub(first,0),state,expert);
    const auto right=build_attention(layer,second,compressed_rows,w,sub(second,attention_rows),left.after,expert);
    llama_dsv41_ffn_graph feed_forward(ctx,ops,swiglu_limit,expert_scale);
    const auto routed=feed_forward.build_routed_pair(w.ffn,ggml_concat(ctx,left.ffn.input,right.ffn.input,1),in.token_kinds,native_pair);
    auto finish=[&](const attention_part &part,uint32_t offset,uint32_t rows){
        auto *one=ggml_view_2d(ctx,routed.output,5120,rows,routed.output->nb[1],size_t(offset)*routed.output->nb[1]);
        return ops.complete(ggml_add(ctx,one,feed_forward.build_shared(w.ffn,part.ffn.input)),part.ffn);
    };
    auto join=[&](ggml_tensor*a,ggml_tensor*b,int dim){GGML_ASSERT(bool(a)==bool(b));return a?ggml_concat(ctx,a,b,dim):nullptr;};
    auto *output=ggml_concat(ctx,finish(left,0,attention_rows),finish(right,attention_rows,second.count),2);
    return {keep(output),keep(ggml_concat(ctx,left.ffn.next.pre,right.ffn.next.pre,1)),keep(join(left.selected,right.selected,1)),
        keep(join(left.mask,right.mask,1)),keep(join(left.candidates,right.candidates,1)),keep(right.next_kv),keep(right.next_score),keep(join(left.capture,right.capture,1))};
}
