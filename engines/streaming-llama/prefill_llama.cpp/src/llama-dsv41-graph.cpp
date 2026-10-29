#include "llama-dsv41-profile-stage.h"
#include "ggml-dsv41-pv-view-f32.h"
#include "llama-dsv41-graph.h"
#include "boundary-capture.h"
#include "ggml-dsv41-shared-sparse.h"
#include "ggml-dsv41-hc-expand.h"
#include "ggml-dsv41-hc-half.h"
#include "ggml-dsv41-hc-sinkhorn.h"
#include "../ggml/src/ggml-dsv41-ops.h"
#include "llama-dsv41-stream-topk.h"
#include "ggml-dsv41-index-window.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "ggml-dsv41-masked-half-inplace.h"
#include "ggml-dsv41-gather-half.h"
#include "ggml-dsv41-split-f32.h"

#include <cmath>
#include <algorithm>
#include <utility>

// CPU fallback for the imported hc=4 Sinkhorn specialization. Output sections
// are pre[rows,4], post[rows,4], comb[rows,4,4] in one24*rows allocation.
static void dsv41_hc_sinkhorn_cpu(ggml_tensor *dst,const ggml_tensor *mixes,
        const ggml_tensor *scale,const ggml_tensor *base,int ith,int nth,void*){
    const auto rows=mixes->ne[1];auto *out=(float*)dst->data;
    const auto *m=(const float*)mixes->data,*s=(const float*)scale->data,*b=(const float*)base->data;
    for(int64_t row=ith;row<rows;row+=nth){
        for(int j=0;j<4;++j){
            out[row*4+j]=1.f/(1.f+std::exp(-(m[row*24+j]*s[0]+b[j])))+1e-6f;
            out[rows*4+row*4+j]=2.f/(1.f+std::exp(-(m[row*24+j+4]*s[1]+b[j+4])));
        }
        float v[16];for(int i=0;i<16;++i)v[i]=m[row*24+i+8]*s[2]+b[i+8];
        for(int r=0;r<4;++r){
            const auto maximum=*std::max_element(v+r*4,v+r*4+4);float sum=0;
            for(int c=0;c<4;++c){v[r*4+c]=std::exp(v[r*4+c]-maximum);sum+=v[r*4+c];}
            for(int c=0;c<4;++c)v[r*4+c]=v[r*4+c]/sum+1e-6f;
        }
        auto columns=[&]{for(int c=0;c<4;++c){float sum=1e-6f;for(int r=0;r<4;++r)sum+=v[r*4+c];
            for(int r=0;r<4;++r)v[r*4+c]/=sum;}};
        columns();for(int iteration=1;iteration<20;++iteration){
            for(int r=0;r<4;++r){float sum=1e-6f;for(int c=0;c<4;++c)sum+=v[r*4+c];for(int c=0;c<4;++c)v[r*4+c]/=sum;}
            columns();
        }
        std::copy(v,v+16,out+rows*8+row*16);
    }
}

// CPU reference for the same original expansion expression. CUDA recognizes
// only the explicit private tag and reuses ds4's existing no-FMA body. The
// ordinary MAP_CUSTOM3 API keeps the graph valid on CPU without a new op enum.
static void dsv41_hc_expand_cpu(ggml_tensor*dst,const ggml_tensor*residual,
        const ggml_tensor*block,const ggml_tensor*mixes,int ith,int nth,void*){
    const auto width=residual->ne[0],total=ggml_nelements(dst);
    const auto*r=(const float*)residual->data,*b=(const float*)block->data,*m=(const float*)mixes->data;
    auto*out=(float*)dst->data;
    for(int64_t at=ith;at<total;at+=nth){
        const auto d=at%width,copy=(at/width)%4,t=at/(width*4);
        float acc=ggml_dsv41_rope_product(b[t*width+d],m[t*20+copy]);
        for(int64_t src=0;src<4;++src)acc=ggml_dsv41_add_rn(acc,
            ggml_dsv41_rope_product(r[(t*4+src)*width+d],m[t*20+4+copy+src*4]));
        out[at]=ggml_dsv41_hc_bf16_requested(dst)?ggml_dsv41_bf16(acc):acc;
    }
}

llama_dsv41_graph_ops::llama_dsv41_graph_ops(
        ggml_context * ctx, float norm_eps, float hc_eps, uint32_t sinkhorn_iters, observer tag) :
    ctx(ctx), norm_eps(norm_eps), hc_eps(hc_eps), sinkhorn_iters(sinkhorn_iters), observe(std::move(tag)) {
    GGML_ASSERT(ctx && std::isfinite(norm_eps) && norm_eps > 0.f);
    GGML_ASSERT(std::isfinite(hc_eps) && hc_eps > 0.f && sinkhorn_iters > 0);
}

ggml_tensor * llama_dsv41_graph_ops::tagged(ggml_tensor * tensor, const char * name) const {
    tensor=dsv41_boundary::replacement(ctx,tensor,name);
    if (observe) observe(tensor, name);
    else ggml_set_name(tensor, name);
    return tensor;
}

ggml_tensor * llama_dsv41_graph_ops::bf16(ggml_tensor * input) const {
    GGML_ASSERT(input->type == GGML_TYPE_F32);
    // Rounding an existing BF16 round trip again is idempotent. Keep its
    // original tensor (and capture boundary), rather than allocating two more
    // casts for the large attention-head result.
    if(ggml_dsv41_hc_bf16_requested(input))return input;
    if ((input->op == GGML_OP_DSV4_ACT_QAT &&
         input->op_params[0] == GGML_DSV41_QAT_BF16) ||
        (input->op == GGML_OP_CPY && input->src[0] &&
         input->src[0]->type == GGML_TYPE_BF16)) return input;
    // Reuse the existing exact V4.1 BF16 QAT operator, which stores F32 directly.
    // Strided inputs retain the original general cast implementation.
    if (ggml_is_contiguous(input))
        return ggml_dsv4_act_qat(ctx,input,GGML_DSV41_QAT_BF16,0);
    // A genuine BF16 round trip; no intervening F16 exponent truncation.
    // Both backends already implement these casts, independently of V4 QAT.
    return ggml_cast(ctx, ggml_cast(ctx, input, GGML_TYPE_BF16), GGML_TYPE_F32);
}

ggml_tensor * llama_dsv41_graph_ops::norm(ggml_tensor * input, ggml_tensor * weight) const {
    llama_dsv41_profile_stage profile_stage(ctx,"norm");
    ggml_tensor * result = ggml_mul(ctx, ggml_rms_norm(ctx, input, norm_eps), weight);
    if (observe) observe(result, "dsv41_weighted_norm_f32");
    return tagged(bf16(result), "dsv41_weighted_norm_bf16");
}

llama_dsv41_graph_ops::seed llama_dsv41_graph_ops::seed_embeddings(ggml_tensor * embeddings) const {
    GGML_ASSERT(embeddings && embeddings->type == GGML_TYPE_F32);
    GGML_ASSERT(embeddings->ne[0] > 0 && embeddings->ne[1] > 0 && embeddings->ne[1] <= 4096);
    GGML_ASSERT(embeddings->ne[2] == 1 && embeddings->ne[3] == 1);
    const int64_t width = embeddings->ne[0], count = embeddings->ne[1];
    auto * rows = ggml_reshape_3d(ctx, ggml_cont(ctx, embeddings), width, 1, count);
    auto * residual = tagged(ggml_repeat_4d(ctx, rows, width, 4, count, 1), "dsv41_seed_residual");
    auto * first = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, count), 1.f);
    auto * zero = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, count), 0.f);
    auto * pre = tagged(ggml_concat(ctx, first, zero, 0), "dsv41_seed_pre");
    return {residual, pre};
}

ggml_tensor * llama_dsv41_graph_ops::logits(ggml_tensor * residual, ggml_tensor * final_pre,
        ggml_tensor * output_norm, ggml_tensor * output_weight) const {
    GGML_ASSERT(residual && final_pre && output_norm && output_weight);
    GGML_ASSERT(residual->ne[2] > 0 && residual->ne[2] <= 4096);
    GGML_ASSERT(ggml_nelements(output_norm) == residual->ne[0]);
    GGML_ASSERT(output_weight->ne[0] == residual->ne[0] && output_weight->ne[1] > 0);
    GGML_ASSERT(output_weight->ne[2] == 1 && output_weight->ne[3] == 1);
    auto * normalized = norm(collapse(residual, final_pre), output_norm);
    auto * result = ggml_mul_mat(ctx, output_weight, normalized);
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    return tagged(result, "dsv41_head_logits_f32");
}

ggml_tensor * llama_dsv41_graph_ops::weighted_sum(ggml_tensor * residual, ggml_tensor * weights) const {
    llama_dsv41_profile_stage profile_stage(ctx,"hc_pre");
    GGML_ASSERT(residual->type == GGML_TYPE_F32 && residual->ne[1] == 4 && residual->ne[3] == 1);
    const int64_t width = residual->ne[0], count = residual->ne[2];
    GGML_ASSERT(weights->type == GGML_TYPE_F32 && weights->ne[0] == 4 && weights->ne[1] == count);
    ggml_tensor * result = nullptr;
    for (int64_t copy = 0; copy < 4; ++copy) {
        ggml_tensor * x = ggml_view_2d(ctx, residual, width, count, residual->nb[2], copy * residual->nb[1]);
        ggml_tensor * w = ggml_view_2d(ctx, weights, 1, count, weights->nb[1], copy * weights->nb[0]);
        ggml_tensor * product = ggml_mul(ctx, x, w);
        result = result ? ggml_add(ctx, result, product) : product;
    }
    return result;
}

ggml_tensor * llama_dsv41_graph_ops::collapse(ggml_tensor * residual, ggml_tensor * previous_pre) const {
    llama_dsv41_profile_stage profile_stage(ctx,"hc_pre");
    return tagged(bf16(weighted_sum(residual, previous_pre)), "dsv41_hc_pre_bf16");
}

ggml_tensor * llama_dsv41_graph_ops::mean_for_capture(ggml_tensor * residual) const {
    ggml_tensor * weights = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, residual->ne[2]), .25f);
    // Native DSpark stores the actual F32 weighted mean, without another BF16 round.
    return tagged(weighted_sum(residual, weights), "dsv41_dspark_layer_mean");
}

ggml_tensor * llama_dsv41_graph_ops::rope(
        ggml_tensor * input, ggml_tensor * positions, ggml_tensor * native_frequencies, bool inverse) const {
    return tagged(ggml_dsv41_rope(ctx, input, positions, native_frequencies, inverse),
                  inverse ? "dsv41_inverse_rope_bf16_tail" : "dsv41_rope_bf16_tail");
}

ggml_tensor * llama_dsv41_graph_ops::gather_per_query(ggml_tensor * table, ggml_tensor * ids) const {
    llama_dsv41_profile_stage profile_stage(ctx,"attention_gather");
    GGML_ASSERT(table->type == GGML_TYPE_F32 && table->ne[2] == 1 && table->ne[3] == 1);
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->ne[2] == 1 && ids->ne[3] == 1);
    GGML_ASSERT(ids->ne[0] > 0 && ids->ne[1] > 0 && ids->ne[1] <= 4096);
    ggml_tensor * flat = ggml_reshape_1d(ctx, ggml_cont(ctx, ids), ids->ne[0] * ids->ne[1]);
    return ggml_reshape_3d(ctx, ggml_get_rows(ctx, table, flat), table->ne[0], ids->ne[0], ids->ne[1]);
}

ggml_tensor * llama_dsv41_graph_ops::raw_snapshot(
        ggml_tensor * ring, ggml_tensor * current, ggml_tensor * history_ids) const {
    llama_dsv41_profile_stage profile_stage(ctx,"cache_publish");
    GGML_ASSERT(ring->type == GGML_TYPE_F32 && ring->ne[0] == 512 && ring->ne[1] == 128);
    GGML_ASSERT(current->type == GGML_TYPE_F32 && current->ne[0] == 512 && current->ne[1] > 0 && current->ne[1] <= 4096);
    GGML_ASSERT(ring->ne[2] == 1 && ring->ne[3] == 1 && current->ne[2] == 1 && current->ne[3] == 1);
    ggml_tensor * result = current;
    if (history_ids) {
        GGML_ASSERT(history_ids->type == GGML_TYPE_I32 && history_ids->ne[0] > 0 && history_ids->ne[0] <= 127);
        GGML_ASSERT(ggml_nelements(history_ids) == history_ids->ne[0]);
        result = ggml_concat(ctx, ggml_get_rows(ctx, ring, history_ids), current, 1);
    }
    return tagged(result, "dsv41_raw_chronological_snapshot");
}

ggml_tensor * llama_dsv41_graph_ops::raw_commit(
        ggml_tensor * ring, ggml_tensor * current, ggml_tensor * attention_complete, uint32_t start) const {
    llama_dsv41_profile_stage profile_stage(ctx,"cache_publish");
    if(current->ne[1]<=32)
        return tagged(ggml_dsv41_raw_commit(ctx, ring, current, attention_complete, start), "dsv41_raw_ring_commit");
    // Attention consumed the chronological snapshot. Only the final128 rows
    // survive in the ring. Reuse the existing <=32-row operation in dependency
    // order, so no CUDA blocks race to overwrite the same modulo128 slot.
    const int64_t first=std::max<int64_t>(0,current->ne[1]-128);
    auto * done=attention_complete;
    auto * result=ring;
    for(int64_t row=first;row<current->ne[1];row+=32){
        auto * rows=ggml_cont(ctx,ggml_view_2d(ctx,current,512,std::min<int64_t>(32,current->ne[1]-row),current->nb[1],row*current->nb[1]));
        result=ggml_dsv41_raw_commit(ctx,ring,rows,done,start+uint32_t(row));done=result;
    }
    return tagged(result,"dsv41_raw_ring_commit");
}

ggml_tensor * llama_dsv41_graph_ops::candidate_mask(ggml_tensor * candidates, int64_t blocks) const {
    llama_dsv41_profile_stage profile_stage(ctx,"index_select");
    GGML_ASSERT(candidates->type==GGML_TYPE_I32 && candidates->ne[0]>0 && candidates->ne[0]<=2048 && candidates->ne[1]>0 && candidates->ne[1]<=4096);
    GGML_ASSERT(candidates->ne[2]==1 && candidates->ne[3]==1 && blocks>0 && blocks<=131072);
    const int64_t queries=candidates->ne[1],top=candidates->ne[0];
    // [1,blocks,queries] writes one scalar per selected block. This mask is
    // tile-local; only the compact candidate IDs go to layer-major SSD carry.
    ggml_tensor * all=ggml_fill(ctx,ggml_new_tensor_3d(ctx,GGML_TYPE_F32,1,blocks,queries),-INFINITY);
    ggml_tensor * zero=ggml_fill(ctx,ggml_new_tensor_3d(ctx,GGML_TYPE_F32,1,top,queries),0.f);
    return ggml_reshape_2d(ctx,ggml_set_rows(ctx,all,zero,ggml_cont(ctx,candidates)),blocks,queries);
}

llama_dsv41_graph_ops::selection llama_dsv41_graph_ops::index_select(
        ggml_tensor * q,ggml_tensor * weights,ggml_tensor * keys,ggml_tensor * visible,
        uint32_t source_layer,ggml_tensor * prior_candidates,ggml_cgraph * stream_graph) const {
    llama_dsv41_profile_stage profile_stage(ctx,"index_select");
    GGML_ASSERT(source_layer==2 || source_layer==8 || source_layer==14 || source_layer==20 ||
                source_layer==24 || source_layer==28 || source_layer==32 || source_layer==36);
    GGML_ASSERT((source_layer>20)==(prior_candidates!=nullptr));
    const int64_t blocks=(keys->ne[1]+7)/8;
    ggml_tensor * mask=prior_candidates?candidate_mask(prior_candidates,blocks):nullptr;
    if(stream_graph&&keys->ne[1]>131072){
        GGML_ASSERT((keys->type==GGML_TYPE_F32||keys->type==GGML_TYPE_BF16)&&ggml_is_contiguous(keys)&&keys->ne[1]<=1048576);
        const bool single=q->ne[2]==1;
        llama_dsv41_detail::streaming_topk row_top(ctx,stream_graph,512),block_top(ctx,stream_graph,2048);
        ggml_tensor * single_scores=nullptr,*single_blocks=nullptr;uint32_t windows=0;
        for(uint32_t first=0;first<uint64_t(keys->ne[1]);first+=8192){
            const uint32_t count=uint32_t(std::min<int64_t>(8192,keys->ne[1]-first));
            auto*window=ggml_cont(ctx,ggml_view_2d(ctx,keys,128,count,keys->nb[1],size_t(first)*keys->nb[1]));
            if(window->type==GGML_TYPE_BF16)window=ggml_cast(ctx,window,GGML_TYPE_F32);
            auto*local_visible=ggml_cast(ctx,ggml_clamp(ctx,ggml_scale_bias(ctx,ggml_cast(ctx,visible,GGML_TYPE_F32),1.f,-float(first)),0.f,float(count)),GGML_TYPE_I32);
            auto*local_mask=mask?ggml_cont(ctx,ggml_view_2d(ctx,mask,(count+7)/8,q->ne[2],mask->nb[1],size_t(first/8)*4)):nullptr;
            auto*scores=tagged(ggml_dsv41_index_score(ctx,q,weights,window,local_visible,local_mask),"dsv41_index_window_scores_f32");
            GGML_ASSERT(ggml_dsv41_index_window_tag(scores,first));
            if(single){single_scores=single_scores?ggml_concat(ctx,single_scores,scores,0):scores;ggml_build_forward_expand(stream_graph,single_scores);}
            else row_top.append(scores,first);
            if(source_layer==20){
                auto*b=ggml_dsv41_candidate_blocks(ctx,scores,visible);GGML_ASSERT(ggml_dsv41_index_window_tag(b,first));
                if(single){single_blocks=single_blocks?ggml_concat(ctx,single_blocks,b,0):b;ggml_build_forward_expand(stream_graph,single_blocks);}
                else block_top.append(b,first/8);
            }
            ++windows;
        }
        // Reducer IDs may be views into gathered merge buffers. Materialize the
        // same owning CONT outputs as the ordinary selector before another
        // attention part expands its stream into this graph.
        auto*rows=tagged(ggml_cont(ctx,single?ggml_top_k(ctx,single_scores,512):row_top.ids()),"dsv41_selected_rows");
        auto*candidates=source_layer==20?tagged(ggml_cont(ctx,single?ggml_top_k(ctx,single_blocks,2048):block_top.ids()),"dsv41_candidate_block_ids"):prior_candidates;
        ggml_build_forward_expand(stream_graph,rows);if(source_layer==20)ggml_build_forward_expand(stream_graph,candidates);
        auto*selected_mask=tagged(ggml_dsv41_selected_mask(ctx,rows,visible,mask),"dsv41_selected_causal_mask");
        std::fprintf(stderr,"DS4PERF_INDEX_STREAM_GRAPH source=%u keys=%lld queries=%lld windows=%u score_window_rows=8192 maximum_dense_score_bytes=%llu single_query_original_selector=%u interleaved_row_block_expansion=1\n",source_layer,(long long)keys->ne[1],(long long)q->ne[2],windows,(unsigned long long)(uint64_t(single?keys->ne[1]:8192)*q->ne[2]*4),unsigned(single));
        return {rows,selected_mask,candidates};
    }
    if(keys->type==GGML_TYPE_BF16)keys=ggml_cast(ctx,keys,GGML_TYPE_F32);
    ggml_tensor * scores=tagged(ggml_dsv41_index_score(ctx,q,weights,keys,visible,mask),"dsv41_index_scores_f32");
    ggml_tensor * candidates=prior_candidates;
    if(source_layer==20){
        ggml_tensor * block_scores=tagged(ggml_dsv41_candidate_blocks(ctx,scores,visible),"dsv41_candidate_block_scores");
        candidates=tagged(ggml_cont(ctx,ggml_top_k(ctx,block_scores,int(std::min<int64_t>(2048,blocks)))),"dsv41_candidate_block_ids");
    }
    // Deliberately uses unfiltered scores at source20. Its candidate result
    // only constrains sources24/28/32/36, never its own rows or readers21-23.
    ggml_tensor * rows=tagged(ggml_cont(ctx,ggml_top_k(ctx,scores,int(std::min<int64_t>(512,keys->ne[1])))),"dsv41_selected_rows");
    ggml_tensor * selected_mask=tagged(ggml_dsv41_selected_mask(ctx,rows,visible,mask),"dsv41_selected_causal_mask");
    return {rows,selected_mask,candidates};
}

ggml_tensor * llama_dsv41_graph_ops::selection_mask(ggml_tensor * selected,ggml_tensor * visible,
        ggml_tensor * candidates,int64_t compressed_rows) const {
    GGML_ASSERT(compressed_rows>0 && compressed_rows<=1048576);
    return tagged(ggml_dsv41_selected_mask(ctx,selected,visible,
        candidates?candidate_mask(candidates,(compressed_rows+7)/8):nullptr),"dsv41_reused_selected_mask");
}

ggml_tensor * llama_dsv41_graph_ops::attention_heads(
        ggml_tensor * query, ggml_tensor * raw_table, ggml_tensor * raw_ids,
        ggml_tensor * raw_mask, ggml_tensor * compressed_table,
        ggml_tensor * selected_ids, ggml_tensor * selected_mask, ggml_tensor * sinks, bool shared_sparse, bool shared_flash_attention,
        bool token_tile_attention,uint32_t start,bool attention_kv_f16_table) const {
    query=dsv41_boundary::replacement(ctx,query,"attention_query_input");
    const int64_t width = query->ne[0], heads = query->ne[1], count = query->ne[2];
    GGML_ASSERT(query->type == GGML_TYPE_F32 && query->ne[3] == 1 && count > 0 && count <= 4096);
    GGML_ASSERT(width > 0 && width <= 512 && heads > 0 && heads <= 64);
    GGML_ASSERT(raw_ids->ne[0] <= 128 && raw_ids->ne[1] == count && raw_table->ne[0] == width);
    GGML_ASSERT(raw_mask->type == GGML_TYPE_F32 && ggml_are_same_shape(raw_ids, raw_mask));
    GGML_ASSERT(sinks->type == GGML_TYPE_F32 && ggml_nelements(sinks) == heads);
    GGML_ASSERT((compressed_table != nullptr) == (selected_ids != nullptr));
    GGML_ASSERT((compressed_table != nullptr) == (selected_mask != nullptr));
    GGML_ASSERT(!attention_kv_f16_table||token_tile_attention);
    if(token_tile_attention){
        GGML_ASSERT(!shared_sparse&&!shared_flash_attention&&width==512&&heads==64);
        const int64_t tiles=(count+3)/4,top=selected_ids?selected_ids->ne[0]:0;
        auto *zero=ggml_fill(ctx,ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width,1),0.f);
        if(attention_kv_f16_table)zero=ggml_cast(ctx,zero,GGML_TYPE_F16);
        // Flatten this concatenation: the existing F32 dim1 CUDA kernel uses
        // grid.y=rows, whose65535 limit cannot hold the64K table plus dummy row.
        auto *table=zero;
        if(compressed_table){
            auto *source=ggml_is_contiguous(compressed_table)?compressed_table:ggml_cont(ctx,compressed_table);
            if(attention_kv_f16_table){
                GGML_ASSERT(source->type==GGML_TYPE_F32||source->type==GGML_TYPE_BF16);
                source=ggml_cast(ctx,source,GGML_TYPE_F16);
            }
            table=ggml_reshape_2d(ctx,ggml_concat(ctx,ggml_reshape_1d(ctx,zero,width),
                ggml_reshape_1d(ctx,source,ggml_nelements(source)),0),width,source->ne[1]+1);
        }
        ggml_tensor *records=nullptr,*union_ids=nullptr;
        if(top){
            // Zero is a real finite dummy KV row. Invalid records use it with
            // a zero query mask, so masked NaNs never enter the MMA PV input.
            auto *ids=ggml_clamp(ctx,ggml_add(ctx,ggml_scale_bias(ctx,ggml_cast(ctx,selected_ids,GGML_TYPE_F32),1.f,1.f),selected_mask),0.f,float(compressed_table->ne[1]));
            if(merge_selected_rows){
                GGML_ASSERT(merge_ratio==1||merge_ratio==2);
                union_ids=ggml_cast(ctx,ggml_scale_bias(ctx,ids,1.f,-1.f),GGML_TYPE_I32);
                // Same admitted record capacity. The private CUDA operator
                // fills these scratch leaves before consuming their records.
                records=ggml_new_tensor_3d(ctx,GGML_TYPE_I32,2,4*top,tiles);ggml_set_input(records);
            }else{
            auto *one=ggml_fill(ctx,ggml_new_tensor_2d(ctx,GGML_TYPE_F32,1,1),1.f);
            auto *bits=one;
            for(float bit:{2.f,4.f,8.f})bits=ggml_concat(ctx,bits,ggml_fill(ctx,ggml_new_tensor_2d(ctx,GGML_TYPE_F32,1,1),bit),1);
            bits=ggml_reshape_2d(ctx,ggml_repeat_4d(ctx,bits,1,4,tiles,1),1,4*tiles);
            if(count<4*tiles)bits=ggml_view_2d(ctx,bits,1,count,bits->nb[1],0);
            auto *valid=ggml_step(ctx,ggml_add(ctx,selected_mask,one));
            auto *query_mask=ggml_mul(ctx,valid,bits);
            records=ggml_concat(ctx,ggml_reshape_3d(ctx,ids,1,top,count),ggml_reshape_3d(ctx,query_mask,1,top,count),0);
            if(count<4*tiles)records=ggml_pad(ctx,records,0,0,int(4*tiles-count),0);
            records=ggml_reshape_3d(ctx,ggml_cast(ctx,records,GGML_TYPE_I32),2,4*top,tiles);
            }
        }else{
            records=ggml_cast(ctx,ggml_fill(ctx,ggml_new_tensor_3d(ctx,GGML_TYPE_F32,2,1,tiles),0.f),GGML_TYPE_I32);
        }
        auto *counts=ggml_cast(ctx,ggml_fill(ctx,ggml_new_tensor_1d(ctx,GGML_TYPE_F32,tiles),float(4*top)),GGML_TYPE_I32);
        if(merge_selected_rows&&top){counts=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,tiles);ggml_set_input(counts);}
        auto *result=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,width,heads,count);
        result->op=GGML_OP_FLASH_ATTN_EXT_SPARSE;
        result->src[0]=ggml_is_contiguous(query)?query:ggml_cont(ctx,query);
        result->src[1]=ggml_is_contiguous(raw_table)?raw_table:ggml_cont(ctx,raw_table);
        result->src[2]=table;result->src[3]=records;result->src[4]=counts;result->src[5]=sinks;
        result->op_params[2]=int32_t(top);result->op_params[3]=int32_t(start);result->op_params[4]=GGML_DSV41_TOKEN_TILE_MAGIC;
        result->op_params[5]=int32_t(merge_selected_rows);result->op_params[6]=int32_t(merge_ratio);result->src[6]=union_ids;
        result->op_params[7]=int32_t(attention_kv_f16_table);
        GGML_ASSERT(ggml_dsv41_token_tile_valid(result));
        tagged(result,"dsv41_ds4_token_tile_f16_probabilities");
        dsv41_boundary::stage(result,"attention_heads_f32");
        auto *rounded=tagged(bf16(result),"dsv41_attention_heads_bf16");
        dsv41_boundary::stage(rounded,"attention_heads_bf16");return rounded;
    }
    if(shared_sparse){
        // Reuse GLM v5 native-F32 attention over a compact full-table view.
        // Canonical cache storage, selected IDs and causal masks stay unchanged.
        GGML_ASSERT(width==512 && raw_table->type==GGML_TYPE_F32);
        auto * table=raw_table;
        auto * ids=raw_ids;
        auto * mask=raw_mask;
        if(compressed_table){
            table=ggml_concat(ctx,raw_table,compressed_table,1);
            auto * shifted=ggml_cast(ctx,ggml_scale_bias(ctx,ggml_cast(ctx,selected_ids,GGML_TYPE_F32),1.f,float(raw_table->ne[1])),GGML_TYPE_I32);
            ids=ggml_concat(ctx,raw_ids,shifted,0);
            mask=ggml_concat(ctx,raw_mask,selected_mask,0);
        }
        auto * result=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,width,heads,count);
        result->op=GGML_OP_FLASH_ATTN_EXT_SPARSE;
        result->src[0]=ggml_cont(ctx,query);
        result->src[1]=ggml_reshape_3d(ctx,ggml_cont(ctx,table),width,1,table->ne[1]);
        result->src[2]=result->src[1];
        result->src[3]=ggml_cont(ctx,ids);result->src[4]=ggml_cont(ctx,mask);result->src[5]=sinks;
        result->op_params[2]=int32_t(ids->ne[0]);result->op_params[4]=GGML_DSV41_SHARED_SPARSE_MAGIC;
        GGML_ASSERT(ggml_dsv41_shared_sparse_valid(result));
        tagged(result,"dsv41_shared_sparse_f32");
        dsv41_boundary::stage(result,"attention_heads_f32");
        auto * rounded=tagged(bf16(result),"dsv41_attention_heads_bf16");
        dsv41_boundary::stage(rounded,"attention_heads_bf16");return rounded;
    }
    const auto exact_option=[](const char*name){const char*v=std::getenv(name);if(!v||!*v)return false;GGML_ASSERT(!std::strcmp(v,"0")||!std::strcmp(v,"1"));return v[0]=='1';};
    const bool compact_gather=exact_option("LLAMA_DSV41_COMPACT_KV_GATHER");
    const bool masked_inplace=exact_option("LLAMA_DSV41_MASKED_HALF_INPLACE");
    const bool fused_gather=exact_option("LLAMA_DSV41_GATHER_HALF");
    GGML_ASSERT(!fused_gather || (compact_gather && !observe && !shared_flash_attention));
    ggml_tensor * gathered = nullptr;
    ggml_tensor * mask = raw_mask;
    if (compressed_table) {
        GGML_ASSERT(compressed_table->ne[0] == width && selected_ids->ne[0] <= 512 && selected_ids->ne[1] == count);
        GGML_ASSERT(selected_mask->type == GGML_TYPE_F32 && ggml_are_same_shape(selected_ids, selected_mask));
        mask = ggml_concat(ctx, mask, selected_mask, 0);
    }
    if(compact_gather && compressed_table){
        // Reuse the shared-sparse path's combined table and shifted IDs.
        // Concatenate the small canonical tables BEFORE gathering instead of
        // concatenating two per-query expansions. Flatten as the existing
        // token-tile path does, avoiding CUDA CONCAT's65535-row grid.y limit.
        GGML_ASSERT(raw_table->type==GGML_TYPE_F32);
        GGML_ASSERT(compressed_table->type==GGML_TYPE_F32||compressed_table->type==GGML_TYPE_BF16);
        GGML_ASSERT(raw_table->ne[1]+compressed_table->ne[1]<=16777216);
        auto * raw=ggml_is_contiguous(raw_table)?raw_table:ggml_cont(ctx,raw_table);
        auto * shifted=ggml_cast(ctx,ggml_scale_bias(ctx,ggml_cast(ctx,selected_ids,GGML_TYPE_F32),1.f,float(raw->ne[1])),GGML_TYPE_I32);
        auto * ids=ggml_concat(ctx,raw_ids,shifted,0);
        if(fused_gather){
            auto * comp=ggml_is_contiguous(compressed_table)?compressed_table:ggml_cont(ctx,compressed_table);
            gathered=ggml_dsv41_gather_half(ctx,raw,ggml_cont(ctx,ids),ggml_cont(ctx,mask),comp);
        }else{
            auto * comp=compressed_table->type==GGML_TYPE_F32?compressed_table:ggml_cast(ctx,compressed_table,GGML_TYPE_F32);
            comp=ggml_is_contiguous(comp)?comp:ggml_cont(ctx,comp);
            auto * table=ggml_reshape_2d(ctx,ggml_concat(ctx,ggml_reshape_1d(ctx,raw,ggml_nelements(raw)),ggml_reshape_1d(ctx,comp,ggml_nelements(comp)),0),width,raw->ne[1]+comp->ne[1]);
            gathered=gather_per_query(table,ids);
        }
        tagged(gathered,"dsv41_compact_kv_gather");
    }else{
        gathered=fused_gather?ggml_dsv41_gather_half(ctx,ggml_cont(ctx,raw_table),ggml_cont(ctx,raw_ids),ggml_cont(ctx,mask)):gather_per_query(raw_table,raw_ids);
        if(compressed_table)gathered=ggml_concat(ctx,gathered,gather_per_query(compressed_table,selected_ids),1);
    }
    if(!fused_gather)tagged(gathered, "dsv41_selected_kv_f32");
    // Native V4.1 Metal attention reads half Q/K/V, but retains float scores,
    // probabilities, accumulators and sinks. Round only the bounded compute
    // inputs; never convert or mutate the canonical full-context F32 caches.
    // Native indexed attention skips invalid rows before loading half KV.
    // Zero masked rows conditionally, rather than multiplying Inf/NaN by zero.
    if(!fused_gather)gathered = tagged(ggml_dsv41_masked_half(ctx,
                       ggml_is_contiguous(gathered)?gathered:ggml_cont(ctx,gathered),
                       ggml_is_contiguous(mask)?mask:ggml_cont(ctx,mask)),
                      "dsv41_selected_kv_native_f16");
    if(fused_gather)tagged(gathered,"dsv41_gather_native_half_f32_store");
    if(masked_inplace&&!fused_gather)GGML_ASSERT(ggml_dsv41_tag_masked_half_inplace(gathered));
    if(shared_flash_attention){
        // The shared 512/512 MMA instance requires >4 query rows per sequence;
        // each token here has its own selected KV set. Preserve that geometry
        // and the original F32 per-head sinks, and let fattn select its tile path.
        const auto keys=gathered->ne[1],padded_keys=((keys+255)/256)*256;
        auto * padded=ggml_cast(ctx,gathered,GGML_TYPE_F16);
        if(padded_keys!=keys){
            // Padding is masked -Inf, so its finite data is unobservable.
            // Reuse the existing byte-wise F16 CONCAT instead of a full F32 PAD.
            const auto gap=padded_keys-keys;
            auto * tail=padded;
            while(tail->ne[1]<gap)tail=ggml_concat(ctx,tail,tail,1);
            if(tail->ne[1]>gap)tail=ggml_view_3d(ctx,tail,width,gap,count,tail->nb[1],tail->nb[2],0);
            padded=ggml_concat(ctx,padded,tail,1);
        }
        auto * kv=ggml_reshape_4d(ctx,padded,width,padded_keys,1,count);
        auto * q=ggml_reshape_4d(ctx,ggml_cont(ctx,ggml_cast(ctx,ggml_cast(ctx,query,GGML_TYPE_F16),GGML_TYPE_F32)),width,1,heads,count);
        auto * padded_mask=mask;
        if(padded_keys!=keys){
            // STEP is1 exactly for valid keys; LOG(0) gives the -Inf tail.
            auto * tail=ggml_log(ctx,ggml_step(ctx,ggml_scale_bias(ctx,ggml_arange(ctx,0.f,float(padded_keys),1.f),-1.f,float(keys))));
            padded_mask=ggml_add(ctx,ggml_pad(ctx,mask,int(padded_keys-keys),0,0,0),tail);
        }
        auto * fa_mask=ggml_reshape_4d(ctx,ggml_cast(ctx,ggml_cont(ctx,padded_mask),GGML_TYPE_F16),padded_keys,1,1,count);
        auto * result=ggml_flash_attn_ext(ctx,q,kv,kv,fa_mask,1.f/sqrtf(float(width)),0.f,0.f);
        ggml_flash_attn_ext_add_sinks(result,sinks);
        ggml_flash_attn_ext_set_prec(result,GGML_PREC_F32);
        result=ggml_reshape_3d(ctx,result,width,heads,count);
        tagged(result,"dsv41_shared_fattn");
        dsv41_boundary::stage(result,"attention_heads_f32");
        auto * rounded=tagged(bf16(result),"dsv41_attention_heads_bf16");
        dsv41_boundary::stage(rounded,"attention_heads_bf16");return rounded;
    }
    ggml_tensor * rounded_query = tagged(ggml_cast(ctx, ggml_cast(ctx, query, GGML_TYPE_F16), GGML_TYPE_F32),
                                        "dsv41_query_native_f16");
    ggml_tensor * logits = ggml_mul_mat(ctx, gathered, rounded_query);
    ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
    const auto split_enabled=[](const char * name) {
        const char * value=std::getenv(name);
        GGML_ASSERT(!value||!std::strcmp(value,"0")||!std::strcmp(value,"1"));
        return value&&!std::strcmp(value,"1");
    };
    const bool split_qk=split_enabled("LLAMA_DSV41_SPLIT_F32_QK");
    const bool split_pv=split_enabled("LLAMA_DSV41_SPLIT_F32_PV");
    if(split_qk)ggml_dsv41_split_f32_tag(logits);
    // SOFT_MAX's sink axis is ne[2] (heads), so put queries on ne[1].
    logits = ggml_cont(ctx, ggml_permute(ctx, logits, 0, 2, 1, 3));
    tagged(logits, "dsv41_sparse_logits_f32");
    // [keys,queries] broadcasts over the head axis without a dense repeat.
    ggml_tensor * probabilities = ggml_soft_max_ext(ctx, logits, ggml_cont(ctx, mask), 1.f / sqrtf(float(width)), 0.f);
    ggml_soft_max_add_sinks(probabilities, sinks);
    tagged(probabilities, "dsv41_sparse_probabilities_f32");
    const bool strided_pv=split_enabled("LLAMA_DSV41_PV_STRIDED_PROBABILITIES");
    probabilities = ggml_permute(ctx, probabilities, 0, 2, 1, 3);
    if(!strided_pv)probabilities=ggml_cont(ctx,probabilities);
    const char * pv_view_option=std::getenv("LLAMA_DSV41_PV_VIEW_F32");
    GGML_ASSERT(!pv_view_option || !std::strcmp(pv_view_option,"0") || !std::strcmp(pv_view_option,"1"));
    const bool pv_view=pv_view_option && !std::strcmp(pv_view_option,"1");
    GGML_ASSERT(!strided_pv||(pv_view&&!split_pv));
    ggml_tensor * result=nullptr;
    if(pv_view) {
        result=ggml_dsv41_pv_view_f32(ctx,gathered,probabilities);
        if(split_pv)ggml_dsv41_split_f32_pv_tag(result);
        tagged(result,"dsv41_pv_view_f32");
    } else {
        ggml_tensor * values = ggml_cont(ctx, ggml_permute(ctx, gathered, 1, 0, 2, 3));
        result = ggml_mul_mat(ctx, values, probabilities);
        ggml_mul_mat_set_prec(result, GGML_PREC_F32);
        if(split_pv)ggml_dsv41_split_f32_tag(result);
    }
    // Native ds4 rounds every head before the inverse unit-magnitude rotation.
    dsv41_boundary::stage(result,"attention_heads_f32");
    auto *rounded=tagged(bf16(result), "dsv41_attention_heads_bf16");
    dsv41_boundary::stage(rounded,"attention_heads_bf16");return rounded;
}

ggml_tensor * llama_dsv41_graph_ops::sinkhorn(ggml_tensor * comb) const {
    llama_dsv41_profile_stage profile_stage(ctx,"hc_comb");
    comb = ggml_soft_max(ctx, comb);
    ggml_tensor * eps = ggml_fill(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1), hc_eps);
    comb = ggml_add(ctx, comb, eps);
    auto columns = [&]() {
        ggml_tensor * transposed = ggml_cont(ctx, ggml_permute(ctx, comb, 1, 0, 2, 3));
        ggml_tensor * sum = ggml_add(ctx, ggml_sum_rows(ctx, transposed), eps);
        comb = ggml_div(ctx, comb, ggml_permute(ctx, sum, 1, 0, 2, 3));
    };
    columns();
    for (uint32_t i = 1; i < sinkhorn_iters; ++i) {
        comb = ggml_div(ctx, comb, ggml_add(ctx, ggml_sum_rows(ctx, comb), eps));
        columns();
    }
    return comb;
}

llama_dsv41_graph_ops::mix llama_dsv41_graph_ops::mixes(
        ggml_tensor * residual, ggml_tensor * fn, ggml_tensor * scale, ggml_tensor * base) const {
    llama_dsv41_profile_stage profile_stage(ctx,"hc_mix");
    GGML_ASSERT(residual->ne[1] == 4 && residual->ne[3] == 1 && ggml_is_contiguous(residual));
    const int64_t count = residual->ne[2], width = residual->ne[0];
    GGML_ASSERT(fn->ne[0] == 4 * width && fn->ne[1] == 24);
    GGML_ASSERT(ggml_nelements(scale) == 3 && ggml_nelements(base) == 24);
    ggml_tensor * flat = tagged(ggml_reshape_2d(ctx, residual, 4 * width, count), "dsv41_hc_flat");
    // Native HC projection consumes a F32 plain RMS norm, not a BF16-rounded one.
    ggml_tensor * flat_norm = tagged(ggml_rms_norm(ctx, flat, norm_eps), "dsv41_hc_flat_norm");
    ggml_tensor * projection_input = flat_norm;
    // Native ds4.c::ds41_hc_mix_batch selects the F16 matrix path only for
    // count > DS4_TP_BATCH_MAX_ROWS (8) and F16 HC weights. Its Metal
    // kernel_mul_mm_f16_f32_scaled (metal/dense.metal, around line 2898)
    // rounds x*RMS_scale to F16 at RHS staging, with F32 accumulation.
    // Preserve that numerical boundary before the unchanged F32 GEMM;
    // small/scalar batches and other weight types retain their F32 input.
    if (count > 8 && fn->type == GGML_TYPE_F16) {
        auto *half=ggml_cast(ctx,flat_norm,GGML_TYPE_F16);
        projection_input = tagged(hc_projection_f16_gemm?half:ggml_cast(ctx,half,GGML_TYPE_F32),
                                  "dsv41_hc_projection_rhs_f16");
    }
    ggml_tensor * projected = ggml_mul_mat(ctx, fn, projection_input);
    ggml_mul_mat_set_prec(projected, GGML_PREC_F32);
    if(hc_projection_f16_gemm&&count>8&&fn->type==GGML_TYPE_F16)GGML_ASSERT(ggml_dsv41_hc_half_tag(projected));
    tagged(projected, "dsv41_hc_mixes");
    const char *fused_hc=std::getenv("LLAMA_DSV41_FUSED_HC");
#ifdef GGML_USE_CUDA
    if(!fused_hc)fused_hc="2";
#endif
    if(fused_hc&&(fused_hc[0]=='1'||fused_hc[0]=='2')&&!fused_hc[1]&&sinkhorn_iters==20&&hc_eps==1e-6f&&count<=65536){
        auto *packed=ggml_map_custom3(ctx,projected,scale,base,dsv41_hc_sinkhorn_cpu,GGML_N_TASKS_MAX,nullptr);
        packed->op_params[7]=GGML_DSV41_HC_SINKHORN_MAGIC;
        packed->op_params[6]=fused_hc[0]-'0';
        GGML_ASSERT(ggml_dsv41_hc_sinkhorn_valid(packed));ggml_set_name(packed,"dsv41_fused_hc_sinkhorn");
        mix fused;
        fused.pre=tagged(ggml_view_2d(ctx,packed,4,count,4*sizeof(float),0),"dsv41_hc_next_pre");
        fused.post=tagged(ggml_view_2d(ctx,packed,4,count,4*sizeof(float),count*4*sizeof(float)),"dsv41_hc_post");
        fused.comb=tagged(ggml_view_3d(ctx,packed,4,4,count,4*sizeof(float),16*sizeof(float),count*8*sizeof(float)),"dsv41_hc_comb");
        return fused;
    }
    auto affine = [&](int64_t offset, int64_t length, int64_t scale_index) {
        ggml_tensor * part = ggml_view_2d(ctx, projected, length, count, projected->nb[1], offset * projected->nb[0]);
        ggml_tensor * s = ggml_view_1d(ctx, scale, 1, scale_index * scale->nb[0]);
        ggml_tensor * b = ggml_view_1d(ctx, base, length, offset * base->nb[0]);
        return ggml_add(ctx, ggml_mul(ctx, part, s), b);
    };
    ggml_tensor * eps = ggml_fill(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1), hc_eps);
    mix result;
    result.pre = tagged(ggml_add(ctx, ggml_sigmoid(ctx, affine(0, 4, 0)), eps), "dsv41_hc_next_pre");
    result.post = tagged(ggml_scale(ctx, ggml_sigmoid(ctx, affine(4, 4, 1)), 2.f), "dsv41_hc_post");
    result.comb = tagged(sinkhorn(ggml_reshape_3d(ctx, affine(8, 16, 2), 4, 4, count)), "dsv41_hc_comb");
    return result;
}

ggml_tensor * llama_dsv41_graph_ops::expand(
        ggml_tensor * block, ggml_tensor * residual, const mix & current) const {
    llama_dsv41_profile_stage profile_stage(ctx,"hc_post");
    const int64_t width = residual->ne[0], count = residual->ne[2];
    GGML_ASSERT(residual->ne[1] == 4 && residual->ne[3] == 1);
    GGML_ASSERT(block->ne[0] == width && block->ne[1] == count && block->ne[2] == 1 && block->ne[3] == 1);
    // Native attention output / routed+shared FFN sums have a BF16 boundary
    // before HC expansion, even when the native producer folds that store.
    block = tagged(bf16(block), "dsv41_block_before_expand_bf16");
    if(ggml_is_contiguous(block)&&ggml_is_contiguous(residual)&&ggml_is_contiguous(current.post)&&ggml_is_contiguous(current.comb)){
        static_assert(sizeof(void*)==8&&GGML_MAX_OP_PARAMS>=32,"Private HC tag follows the24-byte custom3 callback parameters");
        auto*mixes=ggml_concat(ctx,current.post,ggml_reshape_2d(ctx,current.comb,16,count),0);
        auto*out=ggml_map_custom3(ctx,residual,block,mixes,dsv41_hc_expand_cpu,GGML_N_TASKS_MAX,nullptr);
        out->op_params[7]=GGML_DSV41_HC_EXPAND_MAGIC;
        GGML_ASSERT(ggml_dsv41_hc_expand_valid(out));
        if(hc_bf16_epilogue){out->op_params[6]=GGML_DSV41_HC_BF16_MAGIC;return tagged(out,"dsv41_hc_expand_bf16");}
        return tagged(bf16(out),"dsv41_hc_expand_bf16");
    }
    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < 4; ++dst) {
        ggml_tensor * post = ggml_view_2d(ctx, current.post, 1, count, current.post->nb[1], dst * current.post->nb[0]);
        ggml_tensor * acc = ggml_mul(ctx, block, post);
        for (int64_t src = 0; src < 4; ++src) {
            ggml_tensor * r = ggml_view_2d(ctx, residual, width, count, residual->nb[2], src * residual->nb[1]);
            ggml_tensor * c = ggml_view_2d(ctx, current.comb, 1, count, current.comb->nb[2],
                dst * current.comb->nb[0] + src * current.comb->nb[1]);
            acc = ggml_add(ctx, acc, ggml_mul(ctx, r, c));
        }
        acc = ggml_reshape_3d(ctx, acc, width, 1, count);
        out = out ? ggml_concat(ctx, out, acc, 1) : acc;
    }
    return tagged(bf16(out), "dsv41_hc_expand_bf16");
}

llama_dsv41_graph_ops::sublayer llama_dsv41_graph_ops::prepare(
        ggml_tensor * residual, ggml_tensor * previous_pre,
        ggml_tensor * fn, ggml_tensor * scale, ggml_tensor * base, ggml_tensor * norm_weight) const {
    llama_dsv41_profile_stage profile_stage(ctx,"hc_prepare");
    residual=dsv41_boundary::replacement(ctx,residual,"dsv41_hc_input_residual");
    mix next = mixes(residual, fn, scale, base);
    ggml_tensor * input = norm(collapse(residual, previous_pre), norm_weight);
    return {residual, next, input};
}

ggml_tensor * llama_dsv41_graph_ops::complete(ggml_tensor * block, const sublayer & layer) const {
    return expand(block, layer.residual, layer.next);
}

ggml_tensor * llama_dsv41_graph_ops::engram(
        ggml_tensor * residual, ggml_tensor * gathered_rows, ggml_tensor * wkv,
        ggml_tensor * q_weight, ggml_tensor * k_weight, ggml_tensor * text_mask) const {
    llama_dsv41_profile_stage profile_stage(ctx,"engram");
    GGML_ASSERT(gathered_rows->ne[1] == residual->ne[2] && gathered_rows->ne[2] == 1 && gathered_rows->ne[3] == 1);
    GGML_ASSERT(wkv->ne[0] == gathered_rows->ne[0] && wkv->ne[1] == residual->ne[0] * 5);
    ggml_tensor * input = tagged(bf16(gathered_rows), "dsv41_engram_rows_bf16");
    ggml_tensor * projected = ggml_mul_mat(ctx, wkv, input);
    ggml_mul_mat_set_prec(projected, GGML_PREC_F32);
    projected = tagged(bf16(projected), "dsv41_engram_kv_bf16");
    projected = ggml_reshape_3d(ctx, projected, residual->ne[0], 5, residual->ne[2]);
    return tagged(ggml_dsv41_engram_add(ctx, residual, projected, q_weight, k_weight, text_mask, norm_eps), "dsv41_engram_out");
}
