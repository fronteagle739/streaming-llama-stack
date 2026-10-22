#pragma once
#include "ggml-dsv41-stream-topk.h"
#include <algorithm>
namespace llama_dsv41_detail {
// Slabs arrive in increasing global-ID order. A prefix's equal-score IDs are
// already ascending and precede every new ID. Thus compact merge tie order
// equals global-ID tie order. The tagged existing selector refuses NaNs.
class streaming_topk {
    ggml_context*ctx;ggml_cgraph*graph;uint32_t limit,next=0;
    ggml_tensor*values_=nullptr,*ids_=nullptr;
    ggml_tensor*gather(ggml_tensor*source,ggml_tensor*positions){
        GGML_ASSERT(ggml_is_contiguous(source)&&ggml_is_contiguous(positions));
        auto*result=ggml_get_rows(ctx,ggml_reshape_3d(ctx,source,1,source->ne[0],source->ne[1]),positions);
        return ggml_reshape_2d(ctx,result,positions->ne[0],positions->ne[1]);
    }
    ggml_tensor*select(ggml_tensor*scores,uint32_t k){
        auto*t=ggml_top_k(ctx,scores,int(k));GGML_ASSERT(ggml_dsv41_stream_topk_tag(t));return t;
    }
public:
    streaming_topk(ggml_context*c,ggml_cgraph*g,uint32_t k):ctx(c),graph(g),limit(k){GGML_ASSERT(c&&g&&k&&k<=2048);}
    void append(ggml_tensor*scores,uint32_t first){
        GGML_ASSERT(scores&&scores->type==GGML_TYPE_F32&&ggml_is_contiguous(scores)&&scores->ne[0]>0&&scores->ne[0]<=8192&&
            scores->ne[1]>0&&scores->ne[1]<=4096&&scores->ne[2]==1&&scores->ne[3]==1&&first==next&&uint64_t(next)+scores->ne[0]<=1048576);
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
    ggml_tensor*values()const{return values_;}ggml_tensor*ids()const{return ids_;}
    uint32_t consumed()const{return next;}
};
}
