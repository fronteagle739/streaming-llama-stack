#pragma once
#include "ggml-dsv41-stream-topk.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
namespace llama_dsv41_detail {
// Slabs arrive in increasing global-ID order. A prefix's equal-score IDs are
// already ascending and precede every new ID. Thus compact merge tie order
// equals global-ID tie order. The tagged existing selector refuses NaNs.
//
// Deferred merge (LLAMA_DSV41_STREAM_TOPK_DEFER, default 1): per-window
// candidates are appended to one pending buffer in window order and the
// tagged selector merges only when the buffer would exceed its 8192-column
// limit, and once at the end. Pending columns stay in global-ID order (the
// sorted prefix holds only smaller IDs, each window's local selection is
// index-ascending among ties, a raw window is iota order), so every merge
// sees the same (score desc, global ID asc) total order: identical IDs.
// A window no wider than the limit is appended raw (selecting all of it is a
// permutation). 0 restores the per-window running merge.
inline bool stream_topk_defer_enabled(){
    static const bool enabled=[]{
        const char*env=std::getenv("LLAMA_DSV41_STREAM_TOPK_DEFER");
        if(env&&std::strcmp(env,"0")&&std::strcmp(env,"1"))GGML_ABORT("LLAMA_DSV41_STREAM_TOPK_DEFER must be 0 or 1");
        return !env||env[0]=='1';
    }();
    return enabled;
}
class streaming_topk {
    static constexpr int64_t merge_columns=8192;
    ggml_context*ctx;ggml_cgraph*graph;uint32_t limit,next=0;bool defer=stream_topk_defer_enabled();
    ggml_tensor*values_=nullptr,*ids_=nullptr;bool sorted_=true;
    ggml_tensor*gather(ggml_tensor*source,ggml_tensor*positions){
        GGML_ASSERT(ggml_is_contiguous(source)&&ggml_is_contiguous(positions));
        auto*result=ggml_get_rows(ctx,ggml_reshape_3d(ctx,source,1,source->ne[0],source->ne[1]),positions);
        return ggml_reshape_2d(ctx,result,positions->ne[0],positions->ne[1]);
    }
    ggml_tensor*select(ggml_tensor*scores,uint32_t k){
        auto*t=ggml_top_k(ctx,scores,int(k));GGML_ASSERT(ggml_dsv41_stream_topk_tag(t));return t;
    }
    void merge(){
        auto*chosen=select(values_,uint32_t(std::min<int64_t>(limit,values_->ne[0])));
        ids_=gather(ids_,chosen);values_=gather(values_,chosen);sorted_=true;
        ggml_build_forward_expand(graph,values_);ggml_build_forward_expand(graph,ids_);
    }
    void append_deferred(ggml_tensor*scores,uint32_t first){
        ggml_tensor*values,*ids;
        if(scores->ne[0]<=int64_t(limit)){
            values=scores;
            auto*iota=ggml_reshape_2d(ctx,ggml_arange(ctx,float(first),float(first+uint32_t(scores->ne[0])),1.f),scores->ne[0],1);
            ids=ggml_cast(ctx,ggml_repeat(ctx,iota,scores),GGML_TYPE_I32);
        }else{
            auto*positions=select(scores,limit);values=gather(scores,positions);
            ids=first?ggml_cast(ctx,ggml_scale_bias(ctx,ggml_cast(ctx,positions,GGML_TYPE_F32),1.f,float(first)),GGML_TYPE_I32):positions;
        }
        if(values_){
            GGML_ASSERT(values_->ne[1]==scores->ne[1]);
            if(values_->ne[0]+values->ne[0]>merge_columns)merge();
            values_=ggml_concat(ctx,values_,values,0);ids_=ggml_concat(ctx,ids_,ids,0);
        }else{values_=values;ids_=ids;}
        sorted_=false;GGML_ASSERT(values_->ne[0]<=merge_columns);
    }
public:
    streaming_topk(ggml_context*c,ggml_cgraph*g,uint32_t k):ctx(c),graph(g),limit(k){GGML_ASSERT(c&&g&&k&&k<=2048);}
    void append(ggml_tensor*scores,uint32_t first){
        GGML_ASSERT(scores&&scores->type==GGML_TYPE_F32&&ggml_is_contiguous(scores)&&scores->ne[0]>0&&scores->ne[0]<=8192&&
            scores->ne[1]>0&&scores->ne[1]<=4096&&scores->ne[2]==1&&scores->ne[3]==1&&first==next&&uint64_t(next)+scores->ne[0]<=1048576);
        if(defer){append_deferred(scores,first);next+=uint32_t(scores->ne[0]);
            ggml_build_forward_expand(graph,values_);ggml_build_forward_expand(graph,ids_);return;}
        const auto local_count=uint32_t(std::min<int64_t>(limit,scores->ne[0]));auto*positions=select(scores,local_count);
        auto*values=gather(scores,positions);auto*ids=first?ggml_cast(ctx,ggml_scale_bias(ctx,ggml_cast(ctx,positions,GGML_TYPE_F32),1.f,float(first)),GGML_TYPE_I32):positions;
        if(values_){
            GGML_ASSERT(values_->ne[1]==scores->ne[1]);
            values=ggml_concat(ctx,values_,values,0);ids=ggml_concat(ctx,ids_,ids,0);
            auto*chosen=select(values,uint32_t(std::min<int64_t>(limit,values->ne[0])));ids=gather(ids,chosen);values=gather(values,chosen);
        }
        values_=values;ids_=ids;next+=uint32_t(scores->ne[0]);
        // Expand BOTH row/block accumulators once per slab at the caller. This
        // preserves their interleaved score lifetimes under the existing gallocr.
        ggml_build_forward_expand(graph,values_);ggml_build_forward_expand(graph,ids_);
    }
    // The final merge of a deferred buffer; a no-op when already merged.
    void finish(){if(values_&&!sorted_)merge();}
    ggml_tensor*values(){finish();return values_;}ggml_tensor*ids(){finish();return ids_;}
    uint32_t consumed()const{return next;}
};
}
