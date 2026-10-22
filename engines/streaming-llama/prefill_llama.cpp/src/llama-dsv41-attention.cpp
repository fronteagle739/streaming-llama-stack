#include "llama-dsv41-attention.h"
#include "boundary-capture.h"

llama_dsv41_attention_graph::llama_dsv41_attention_graph(ggml_context * ctx, llama_dsv41_graph_ops & ops) : ctx(ctx), ops(ops) {
    GGML_ASSERT(ctx);
}

ggml_tensor * llama_dsv41_attention_graph::mm(ggml_tensor * weight, ggml_tensor * input, bool round) const {
    GGML_ASSERT(weight && input && input->type==GGML_TYPE_F32);
    ggml_tensor * result=ggml_mul_mat(ctx,weight,input);
    ggml_mul_mat_set_prec(result,GGML_PREC_F32);
    return round?ops.bf16(result):result;
}
ggml_tensor * llama_dsv41_attention_graph::qat(ggml_tensor * input,ggml_dsv4_qat_mode mode) const {
    GGML_ASSERT(mode>=GGML_DSV41_QAT_BF16 && mode<=GGML_DSV41_QAT_FP8_E8M0);
    return ggml_dsv4_act_qat(ctx,ggml_cont(ctx,input),mode,0);
}

llama_dsv41_attention_graph::projected llama_dsv41_attention_graph::project(
        const weights & w,ggml_tensor * input,ggml_tensor * positions,ggml_tensor * frequencies) const {
    const int64_t count=input->ne[1];
    GGML_ASSERT(w.q_b->ne[1]%512==0 && w.kv->ne[1]==512 && input->ne[2]==1 && input->ne[3]==1);
    const int64_t heads=w.q_b->ne[1]/512;
    GGML_ASSERT(heads>0 && heads<=64 && count>0 && count<=4096);
    auto * qr=ops.norm(mm(w.q_a,input,true),w.q_norm);
    // V4.1 has no V4 post-q_b RMS norm and no Hadamard on either stream.
    auto * q=ggml_reshape_3d(ctx,mm(w.q_b,qr,true),512,heads,count);
    q=ops.rope(q,positions,frequencies);
    dsv41_boundary::stage(q,"q_after_rope_f32");
    // Identical production operations; capture observes existing boundaries.
    auto * kv_projection=mm(w.kv,input,false);
    dsv41_boundary::stage(kv_projection,"kv_projection_f32");
    auto * kv_bf16=ops.bf16(kv_projection);
    dsv41_boundary::stage(kv_bf16,"kv_projection_bf16");
    auto * raw=ops.norm(kv_bf16,w.kv_norm);
    raw=ops.rope(ggml_reshape_3d(ctx,raw,512,1,count),positions,frequencies);
    dsv41_boundary::stage(raw,"kv_rope_bf16");
    raw=qat(raw,GGML_DSV41_QAT_FP8_E8M0); // all layers, including raw-only0/1
    dsv41_boundary::stage(raw,"kv_fp8_e8m0");
    auto *raw_rows=ggml_reshape_2d(ctx,raw,512,count);
    raw_rows=dsv41_boundary::replacement(ctx,raw_rows,"attention_raw_kv_input");
    return {qr,q,raw_rows};
}

llama_dsv41_attention_graph::compressed llama_dsv41_attention_graph::compress(
        const weights & w,ggml_tensor * input,const llama_dsv41_tile_plan & p,
        ggml_tensor * previous_kv,ggml_tensor * previous_score,
        ggml_tensor * publication_positions,ggml_tensor * frequencies) const {
    GGML_ASSERT(p.ratio==1 || p.ratio==2);
    GGML_ASSERT(input->ne[1]==p.count && w.compressor && w.compressor->ne[1]==512);
    ggml_tensor * projected=mm(w.compressor,input,p.ratio==1);
    ggml_tensor * scores=nullptr,*latent=nullptr,*next_kv=nullptr,*next_score=nullptr;
    if(p.ratio==2){
        GGML_ASSERT(w.gate && w.gate->ne[1]==512);
        GGML_ASSERT(!p.previous_pool_tail || (previous_kv && previous_score));
        scores=mm(w.gate,input,false);
        if(p.next_pool_tail){
            // These are the original F32 projections/scores, not their pooled
            // or BF16-rounded result. They remain live until carry publication.
            next_kv=ggml_view_1d(ctx,projected,512,(p.count-1)*projected->nb[1]);
            next_score=ggml_view_1d(ctx,scores,512,(p.count-1)*scores->nb[1]);
        }
        if(p.published_count)latent=ggml_dsv41_pool2(ctx,projected,scores,
            p.previous_pool_tail?previous_kv:nullptr,p.previous_pool_tail?previous_score:nullptr);
    }else latent=projected;
    if(!p.published_count)return {nullptr,nullptr,next_kv,next_score};
    GGML_ASSERT(publication_positions && ggml_nelements(publication_positions)==p.published_count);
    GGML_ASSERT(latent->ne[1]==p.published_count);
    // Shared latent is normalized/BF16 BEFORE the index-key projection and
    // before its own rotation. Indexer never reads the rotated KV latent.
    latent=ops.norm(latent,w.compressor_norm);
    auto * index=ops.norm(mm(w.index_k,latent,true),w.index_norm);
    GGML_ASSERT(index->ne[0]==128);
    index=ops.rope(ggml_reshape_3d(ctx,index,128,1,p.published_count),publication_positions,frequencies);
    index=qat(index,GGML_DSV41_QAT_FP4_E8M0);
    auto * kv=ops.rope(ggml_reshape_3d(ctx,latent,512,1,p.published_count),publication_positions,frequencies);
    kv=qat(kv,GGML_DSV41_QAT_FP4_E4M3);
    return {ggml_reshape_2d(ctx,kv,512,p.published_count),ggml_reshape_2d(ctx,index,128,p.published_count),next_kv,next_score};
}

llama_dsv41_attention_graph::index_query llama_dsv41_attention_graph::project_index(
        const weights & w,ggml_tensor * qr,ggml_tensor * input,ggml_tensor * positions,ggml_tensor * frequencies) const {
    const int64_t count=input->ne[1];
    GGML_ASSERT(w.index_q && w.index_q->ne[1]==128*32 && w.index_weights && w.index_weights->ne[1]==32);
    auto * q=ggml_reshape_3d(ctx,mm(w.index_q,qr,true),128,32,count);
    q=qat(ops.rope(q,positions,frequencies),GGML_DSV41_QAT_FP4_E8M0);
    // Native projection is BF16; score reduction applies the1/64 scale after
    // each head dot and before multiplying by this head weight.
    return {q,mm(w.index_weights,input,true)};
}

ggml_tensor * llama_dsv41_attention_graph::finish(
        const weights & w,ggml_tensor * heads,ggml_tensor * positions,ggml_tensor * frequencies) const {
    GGML_ASSERT(heads->ne[0]==512 && heads->ne[3]==1 && w.out_a && w.out_b);
    // Native GGUF/this loader stores out_a flattened[4096,8192]. Restore
    // [4096,1024,8] for the grouped product without copying the weight bytes.
    GGML_ASSERT(w.out_a->ne[2]==1 && w.out_a->ne[3]==1 && w.out_a->ne[0]>0);
    GGML_ASSERT((512*heads->ne[1])%w.out_a->ne[0]==0);
    const int64_t groups=512*heads->ne[1]/w.out_a->ne[0],count=heads->ne[2];
    GGML_ASSERT(groups>0 && heads->ne[1]%groups==0 && w.out_a->ne[1]%groups==0);
    const int64_t rank=w.out_a->ne[1]/groups;
    auto * out_a=ggml_reshape_3d(ctx,w.out_a,w.out_a->ne[0],rank,groups);
    auto * rotated=ops.rope(ops.bf16(heads),positions,frequencies,true);
    auto * grouped=ggml_reshape_3d(ctx,rotated,w.out_a->ne[0],groups,count);
    grouped=ggml_permute(ctx,grouped,0,2,1,3);
    auto * low=mm(out_a,grouped,true); // native out_a has its own BF16 store
    low=ggml_cont_2d(ctx,ggml_permute(ctx,low,0,2,1,3),rank*groups,count);
    // complete()/expand() owns the final block BF16 boundary.
    return mm(w.out_b,low,false);
}
