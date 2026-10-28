/* Included by ds4_decode_adapter.inc after the native GLM graph definitions. */
#include "ds4_glm_external.h"
static struct {
    bool open, bound;
    uint32_t context, position;
    ds4_model model;
    ds4_weights weights;
    ds4_glm_gpu_graph graph;
    ds4_gpu_tensor view[DS4_MAX_LAYER][6];
} g_ds4glm;

/* Called also by ds4x_open: shape/backend globals cannot represent two models. */
static int ds4glm_is_open(void) { return g_ds4glm.open; }

DS4X_EXPORT int ds4glm_open(const char *path, uint32_t context,
        uint64_t scratch_limit, char *error, size_t cap) {
    if (g_ds4glm.open || g_ds4x.open || !path || context < 128u ||
        context > 1048576u || !scratch_limit || scratch_limit > 256u*1024u*1024u) {
        ds4x_set_error(error,cap,"GLM external open requires an exclusive model, context 128..1048576 and scratch budget <=256 MiB");return 0;
    }
    model_open(&g_ds4glm.model,path,false,false);
    config_validate_model(&g_ds4glm.model);
    if (!ds4_model_is_glm53()) {
        model_close(&g_ds4glm.model);
        ds4x_set_error(error,cap,"GLM external decode requires glm5-next");return 0;
    }
    weights_bind(&g_ds4glm.weights,&g_ds4glm.model,false,0,0,true,false);
    if (!ds4_gpu_init()) {
        model_close(&g_ds4glm.model);ds4x_set_error(error,cap,"GLM CUDA backend unavailable");return 0;
    }
    ds4_gpu_set_quality(false);ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_external_decode_mode(1);
    g_ds4glm_scratch_limit=scratch_limit;g_ds4glm_scratch_bytes=0;
    g_ds4glm_external_alloc=true;
    const bool ok=glm_graph_alloc(&g_ds4glm.graph,&g_ds4glm.model,
            &g_ds4glm.weights,(int)context,false,false);
    g_ds4glm_external_alloc=false;
    if (!ok) {
        glm_graph_free(&g_ds4glm.graph);model_close(&g_ds4glm.model);
        ds4_gpu_external_decode_mode(0);memset(&g_ds4glm,0,sizeof(g_ds4glm));
        ds4x_set_error(error,cap,"GLM external scratch allocation failed or exceeded budget");return 0;
    }
    g_ds4glm.context=context;g_ds4glm.open=true;
    fprintf(stderr,"GLM_EXTERNAL_OPEN context=%u scratch_bytes=%llu external_weights=1 external_state=1 prefill_workspace=0\n",
        context,(unsigned long long)g_ds4glm_scratch_bytes);
    return 1;
}

DS4X_EXPORT int ds4glm_register_weight(uint64_t offset,uint64_t bytes,void *ptr) {
    return g_ds4glm.open && ptr && bytes &&
        ds4_gpu_register_external_range(g_ds4glm.model.map,offset,bytes,ptr);
}
DS4X_EXPORT int ds4glm_replace_weight(uint64_t offset,uint64_t bytes,void *ptr) {
    return g_ds4glm.open && ptr && bytes &&
        ds4_gpu_replace_external_range(g_ds4glm.model.map,offset,bytes,ptr);
}
DS4X_EXPORT uint64_t ds4glm_drop_weights(void) {
    if (!g_ds4glm.open) return 0;
    ds4_gpu_decode_graphs_invalidate();
    return ds4_gpu_drop_external_ranges();
}
DS4X_EXPORT void ds4glm_set_expert_hook(ds4_gpu_external_expert_hook hook,void *user) {
    if (g_ds4glm.open) ds4_gpu_set_external_expert_hook(hook,user);
}

static bool ds4glm_span_valid(ds4glm_span span,uint64_t expected) {
    return expected ? span.data && span.bytes >= expected : !span.data && !span.bytes;
}

DS4X_EXPORT int ds4glm_bind_state(const ds4glm_state *state,char *error,size_t cap) {
    ds4_glm_gpu_graph *g=&g_ds4glm.graph;
    if (!g_ds4glm.open || !state || state->abi!=DS4GLM_ABI || !state->layer ||
        state->layers!=g->layer_count || state->context!=g_ds4glm.context ||
        state->position>=state->context || !glm_graph_compact_cache_is_f16()) {
        ds4x_set_error(error,cap,"GLM external state ABI/layers/context/position/F16 precision mismatch");return 0;
    }
    const uint64_t projection=(uint64_t)DS4_N_KDA_HEAD*DS4_N_KDA_HEAD_DIM;
    /* Validate the entire bank before replacing any borrowed pointer. */
    for (uint32_t il=0;il<state->layers;++il) {
        const ds4glm_layer_state *s=&state->layer[il];
        const bool kda=ds4_glm53_layer_is_kda(il);
        const ds4glm_span spans[6]={s->conv,s->recurrent,s->latent,s->pooled_key,s->tail_key,s->tail_gate};
        const uint64_t expected[6]={
            kda ? 3u*(DS4_N_KDA_CONV-1u)*projection*sizeof(float) : 0,
            kda ? projection*DS4_N_KDA_HEAD_DIM*sizeof(float) : 0,
            kda ? 0 : (uint64_t)state->context*DS4_N_KV_LORA*sizeof(uint16_t),
            kda ? 0 : (uint64_t)glm53_graph_indexer_pool_cap(state->context)*DS4_N_INDEXER_HEAD_DIM*sizeof(uint16_t),
            kda ? 0 : (uint64_t)DS4_GLM53_INDEX_POOL_SIZE*DS4_N_INDEXER_HEAD_DIM*sizeof(float),
            kda ? 0 : (uint64_t)DS4_GLM53_INDEX_POOL_SIZE*DS4_N_INDEXER_HEAD_DIM*sizeof(float)};
        for (uint32_t j=0;j<6;++j) if (!ds4glm_span_valid(spans[j],expected[j])) {
            if (error && cap) snprintf(error,cap,"GLM external state layer %u owner %u invalid: bytes=%llu expected=%llu",il,j,
                (unsigned long long)spans[j].bytes,(unsigned long long)expected[j]);return 0;
        }
    }
    if (!ds4_gpu_synchronize()) {ds4x_set_error(error,cap,"GLM state bind synchronization failed");return 0;}
    for (uint32_t il=0;il<state->layers;++il) {
        const ds4glm_layer_state *s=&state->layer[il];
        const ds4glm_span spans[6]={s->conv,s->recurrent,s->latent,s->pooled_key,s->tail_key,s->tail_gate};
        ds4_gpu_tensor **target[6]={&g->layer_kda_conv_state[il],&g->layer_kda_recurrent_state[il],
            &g->layer_kv_lora_cache[il],&g->layer_indexer_key_cache[il],
            &g->layer_indexer_tail_k[il],&g->layer_indexer_tail_gate[il]};
        for (uint32_t j=0;j<6;++j) {
            ds4x_view(&g_ds4glm.view[il][j],spans[j].data,spans[j].bytes);
            *target[j]=spans[j].data ? &g_ds4glm.view[il][j] : NULL;
        }
    }
    g_ds4glm.position=state->position;g_ds4glm.bound=true;
    fprintf(stderr,"GLM_EXTERNAL_BIND position=%u layers=%u precision=f16 borrowed=1\n",state->position,state->layers);
    return 1;
}

DS4X_EXPORT int ds4glm_decode(int32_t token,uint32_t position,float *logits) {
    if (!g_ds4glm.open || !g_ds4glm.bound || !logits || token<0 ||
        token>=(int32_t)DS4_N_VOCAB || position!=g_ds4glm.position || position>=g_ds4glm.context) return 0;
    /* A failed graph may have partially advanced recurrence. Require a fresh
     * caller state binding before another decode, rather than silently retry. */
    g_ds4glm.bound=false;
    const bool ok=glm_graph_forward_token(&g_ds4glm.graph,&g_ds4glm.model,
        &g_ds4glm.weights,token,NULL,position,NULL,logits,false);
    if (ok) {g_ds4glm.position=position+1u;g_ds4glm.bound=true;}
    return ok;
}

DS4X_EXPORT void ds4glm_close(void) {
    if (!g_ds4glm.open) return;
    ds4_gpu_synchronize();
    ds4_glm_gpu_graph *g=&g_ds4glm.graph;
    for (uint32_t il=0;il<DS4_MAX_LAYER;++il) {
        g->layer_kda_conv_state[il]=g->layer_kda_recurrent_state[il]=NULL;
        g->layer_kv_lora_cache[il]=g->layer_indexer_key_cache[il]=NULL;
        g->layer_indexer_tail_k[il]=g->layer_indexer_tail_gate[il]=NULL;
    }
    glm_graph_free(g);ds4_gpu_drop_external_ranges();
    ds4_gpu_set_external_expert_hook(NULL,NULL);ds4_gpu_set_external_residency_hook(NULL,NULL);
    ds4_gpu_external_decode_mode(0);model_close(&g_ds4glm.model);
    memset(&g_ds4glm,0,sizeof(g_ds4glm));
}
