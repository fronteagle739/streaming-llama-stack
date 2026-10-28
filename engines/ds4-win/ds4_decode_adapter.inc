/* LLAMADECODE: ds4-win's own V4.1 single-token decode (ds41_graph_step and
 * everything it calls, unchanged) driven on another engine's state.
 *
 * Built only with -DDS4_EXTERNAL_DECODE=1 (ds4-decode.dll). Normal ds4-win
 * builds do not include this file. The caller (the streaming-llama ULTRA
 * producer) owns:
 *   - every weight the decode reads: it registers each tensor's file offset
 *     against its own device copy (ds4_gpu_register_external_range); an
 *     unregistered weight is a refusal, never a host mapping;
 *   - the canonical V4.1 state: raw rings, compressed/index caches and pool
 *     tails, bound as views (same layout and values as ds4-win's own);
 *   - the routed experts: a hook returns cache-pool bases and slot ids.
 * ds4-win allocates its per-token scratch. The optional versioned host-cache
 * bridge reuses ds4's pinned cache for producer-owned expert victims. No
 * standalone streamer, prefill workspace or draft model starts here. */
#ifdef DS4_EXTERNAL_DECODE
#if defined(_WIN32)
#define DS4X_EXPORT __declspec(dllexport)
#else
#define DS4X_EXPORT __attribute__((visibility("default")))
#endif

static struct {
    bool open;
    ds4_model model;
    ds4_weights weights;
    ds41_gpu_graph graph;
    ds4_gpu_tensor window[40], compressed[4], index_cache[4], previous_kv[4], previous_score[4];
    ds4_gpu_tensor *spare_kv, *spare_score;   /* ratio-1 owner 3 keeps no pool tail */
} g_ds4x;

static void ds4x_set_error(char *error, size_t cap, const char *message) {
    if (error && cap) snprintf(error, cap, "%s", message);
}

/* Isolated vision worker: reuse Studio's original encoder, preprocessing,
 * sentinel embeddings and V4.1 image layout, without loading an LLM session. */
DS4X_EXPORT int ds4x_vision_file(const char *model_path, const char *vision_path,
        const char *image_path, const char *output_path, char *error, size_t cap) {
    if (g_ds4x.open || !model_path || !vision_path || !image_path || !output_path) {
        ds4x_set_error(error,cap,"Vision encoding requires an isolated worker");return 0;
    }
    ds4_engine *e=calloc(1,sizeof(*e));if(!e){ds4x_set_error(error,cap,"Vision metadata allocation failed");return 0;}ds4_vision_embedding embedding={0};ds4_vision_span span={0};ds4_tokens tokens={0};
    model_open(&e->model,model_path,false,false);config_validate_model(&e->model);
    if (DS4_MODEL_FAMILY!=DS4_MODEL_FAMILY_DEEPSEEK41) {
        model_close(&e->model);free(e);ds4x_set_error(error,cap,"V4.1 vision model required");return 0;
    }
    model_open(&e->vision_model,vision_path,true,false);
    deepseek4_vision_weights_bind(&e->deepseek4_vision_weights,&e->model,&e->vision_model);
    e->backend=DS4_BACKEND_CUDA;e->vision_kind=DS4_VISION_DEEPSEEK4;e->vision_ready=true;
    e->vision_image_token=e->vision_start_token=e->vision_end_token=129264;
    const uint64_t map_offset=e->vision_model.tensor_data_pos,map_size=e->vision_model.size-map_offset;
    void *device=NULL;uint64_t fetched=0;double fetch_ms=0,wait_ms=0;
    int ok=ds4_gpu_init()&&ds4_ds_init_from_fd(e->vision_model.fd)&&ds4_ds_gpu_arena_create(map_size,&device);
    if(ok){
        ds4_ds_batch_reset();
        for(uint64_t off=0;ok&&off<map_size;off+=8u*1024u*1024u){
            const uint32_t count=(uint32_t)((map_size-off)<8u*1024u*1024u?map_size-off:8u*1024u*1024u);
            ok=ds4_ds_batch_add_gpu(map_offset+off,count,off);
        }
        if(ok)ok=ds4_ds_batch_submit_wait(&fetched,&fetch_ms,&wait_ms)&&fetched==map_size&&
            ds4_ds_gpu_fence_wait_last(NULL)&&ds4_gpu_synchronize();
        if(ok)ok=ds4_gpu_register_external_range(e->vision_model.map,map_offset,map_size,device);
    }
    if(ok){
        ds4_gpu_external_decode_mode(1);e->metal_ready=true;e->vision_map_ready=true;
        fprintf(stderr,"LLAMADECODE_VISION_FETCH backend=directstorage destination=gpu bytes=%llu milliseconds=%.3f host_weight_staging=0\n",(unsigned long long)fetched,fetch_ms);
        ok=ds4_engine_vision_encode_file(e,image_path,&embedding,error,cap);
    }else ds4x_set_error(error,cap,"DirectStorage GPU vision fetch failed; host-copy fallback refused");
    if(ok)ok=ds4_prompt_append_vision(e,&tokens,&span,&embedding,error,cap);
    FILE *out=NULL;
    if(ok){
        if(!span.embedding.token_count || span.embedding.token_count>32768 || span.token_start ||
           tokens.len!=(int)span.embedding.token_count){ds4x_set_error(error,cap,"Invalid V4.1 projected image block");ok=0;}
        else out=fopen(output_path,"wbx");
        if(ok&&!out){ds4x_set_error(error,cap,"Cannot create fresh vision embedding file");ok=0;}
    }
    if(ok){
        const uint32_t header[4]={0x49564c53,1,span.embedding.token_count,5120};
        const size_t values=(size_t)span.embedding.token_count*5120;
        ok=fwrite(header,sizeof(header),1,out)==1&&fwrite(span.embedding.fingerprint,32,1,out)==1&&
            fwrite(span.embedding.data,sizeof(float),values,out)==values;
        if(!ok)ds4x_set_error(error,cap,"Vision embedding write failed");
    }
    if(out&&fclose(out))ok=0;
    ds4_vision_embedding_free(&embedding);ds4_vision_embedding_free(&span.embedding);free(tokens.v);
    ds4_gpu_synchronize();ds4_gpu_external_decode_mode(0);ds4_ds_gpu_arena_release();ds4_ds_shutdown();
    fprintf(stderr,"LLAMADECODE_VISION_RELEASE encoder_weights_released=1 next=llm_prefill\n");
    model_close(&e->vision_model);model_close(&e->model);free(e);return ok;
}

DS4X_EXPORT int ds4x_open(const char *model_path, uint32_t ctx, char *error, size_t cap) {
    if (g_ds4x.open) { ds4x_set_error(error, cap, "ds4x already open"); return 0; }
    if (!model_path || !ctx) { ds4x_set_error(error, cap, "ds4x_open requires a model and context"); return 0; }
    model_open(&g_ds4x.model, model_path, false, false);
    config_validate_model(&g_ds4x.model);
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_DEEPSEEK41) {
        model_close(&g_ds4x.model);
        ds4x_set_error(error, cap, "ds4x supports DeepSeek V4.1 only");
        return 0;
    }
    weights_bind(&g_ds4x.weights, &g_ds4x.model, false, 0, 0, true, false);
    if (!ds4_gpu_init()) { model_close(&g_ds4x.model); ds4x_set_error(error, cap, "ds4 CUDA backend unavailable"); return 0; }
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_external_decode_mode(1);   /* refuse any weight not registered by the caller */
    g_ds4x_external_alloc = true;
    const bool ok = ds41_graph_alloc(&g_ds4x.graph, &g_ds4x.model, &g_ds4x.weights, model_path, ctx, false);
    g_ds4x_external_alloc = false;
    if (!ok) {
        ds41_graph_free(&g_ds4x.graph);
        model_close(&g_ds4x.model);
        ds4x_set_error(error, cap, "ds4x scratch allocation failed");
        return 0;
    }
    g_ds4x.spare_kv = ds4_gpu_tensor_alloc(512u * sizeof(float));
    g_ds4x.spare_score = ds4_gpu_tensor_alloc(512u * sizeof(float));
    if (!g_ds4x.spare_kv || !g_ds4x.spare_score) { ds4x_set_error(error, cap, "ds4x pool-tail scratch failed"); return 0; }
    g_ds4x.open = true;
    fprintf(stderr, "LLAMADECODE_DS4X_OPEN ctx=%u model_map=%p external_weights=1 external_state=1 streaming=0\n",
            ctx, (const void *)g_ds4x.model.map);
    return 1;
}

/* The model map pointer the caller passes with every registered range. */
DS4X_EXPORT const void *ds4x_model_map(void) { return g_ds4x.open ? g_ds4x.model.map : NULL; }

DS4X_EXPORT int ds4x_register_weight(uint64_t file_offset, uint64_t bytes, void *device_ptr) {
    if (!g_ds4x.open || !device_ptr || !bytes) return 0;
    return ds4_gpu_register_external_range(g_ds4x.model.map, file_offset, bytes, device_ptr);
}

/* LLAMADECODE keep-open (LLAMADECODE_DS4X_KEEP_OPEN): the decoder stays open
 * across a caller bank rebuild (prefill weight sweep). The caller then moves
 * only the weights whose device copy moved (same file range), or drops all of
 * its ranges and registers the new set. Scratch, model map, Engram tables and
 * bound-state views are kept; the caller rebinds its state before decoding.
 * Callers invoke these only between decode steps (no ds4 work in flight). */
DS4X_EXPORT int ds4x_replace_weight(uint64_t file_offset, uint64_t bytes, void *device_ptr) {
    if (ds41_single_hc_graphs_on()) ds4_gpu_decode_graphs_invalidate();
    if (!g_ds4x.open || !device_ptr || !bytes) return 0;
    return ds4_gpu_replace_external_range(g_ds4x.model.map, file_offset, bytes, device_ptr);
}

DS4X_EXPORT uint64_t ds4x_drop_weights(void) {
    if (ds41_single_hc_graphs_on()) ds4_gpu_decode_graphs_invalidate();
    if (!g_ds4x.open) return 0;
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    ds4_gpu_decode_graphs_invalidate();   /* captured islands hold weight pointers (unused at tp_world 1) */
#endif
    return ds4_gpu_drop_external_ranges();
}

static void ds4x_view(ds4_gpu_tensor *t, void *ptr, uint64_t bytes) {
    t->ptr = ptr; t->bytes = bytes; t->owner = 0; t->device_id = -1;
}

/* raw[40]: [128][512] F32 rings; compressed/index[4] with their row counts;
 * pool_kv/pool_score[3] for the ratio-2 owners. cache_bf16: the compressed/
 * index owners are BF16 (LLAMADECODE --ctx-cache-precision bf16). */
DS4X_EXPORT int ds4x_bind_state_ex(void *const *raw, void *const *compressed, void *const *index,
                                   const uint32_t *rows, void *const *pool_kv, void *const *pool_score,
                                   uint32_t cache_bf16) {
    if (!g_ds4x.open || !raw || !compressed || !index || !rows || !pool_kv || !pool_score || cache_bf16 > 1u) return 0;
    if (cache_bf16 && g_ds4x.graph.batch.residual) {
        fprintf(stderr, "LLAMADECODE_DS4X_BIND_REFUSED reason=bf16 owner caches with the multi-row prefill workspace\n");
        return 0;
    }
    const uint64_t element = cache_bf16 ? sizeof(uint16_t) : sizeof(float);
    ds41_gpu_graph *g = &g_ds4x.graph;
    for (uint32_t i = 0; i < 40; i++) {
        if (!raw[i]) return 0;
        ds4x_view(&g_ds4x.window[i], raw[i], 128u * 512u * sizeof(float));
        g->window[i] = &g_ds4x.window[i];
    }
    for (uint32_t i = 0; i < 4; i++) {
        if (!compressed[i] || !index[i] || !rows[i]) return 0;
        ds4x_view(&g_ds4x.compressed[i], compressed[i], (uint64_t)rows[i] * 512u * element);
        ds4x_view(&g_ds4x.index_cache[i], index[i], (uint64_t)rows[i] * 128u * element);
        g->compressed[i] = &g_ds4x.compressed[i];
        g->index_cache[i] = &g_ds4x.index_cache[i];
        if (i < 3) {
            if (!pool_kv[i] || !pool_score[i]) return 0;
            ds4x_view(&g_ds4x.previous_kv[i], pool_kv[i], 512u * sizeof(float));
            ds4x_view(&g_ds4x.previous_score[i], pool_score[i], 512u * sizeof(float));
            g->previous_kv[i] = &g_ds4x.previous_kv[i];
            g->previous_score[i] = &g_ds4x.previous_score[i];
        } else {
            g->previous_kv[i] = g_ds4x.spare_kv;
            g->previous_score[i] = g_ds4x.spare_score;
        }
    }
    if (g_ds41_cache_bf16 != (cache_bf16 != 0u))
        fprintf(stderr, "LLAMADECODE_DS4X_CACHE_PRECISION owners=%s\n", cache_bf16 ? "bf16" : "f32");
    g_ds41_cache_bf16 = cache_bf16 != 0u;
    return 1;
}

DS4X_EXPORT int ds4x_bind_state(void *const *raw, void *const *compressed, void *const *index,
                                const uint32_t *rows, void *const *pool_kv, void *const *pool_score) {
    return ds4x_bind_state_ex(raw, compressed, index, rows, pool_kv, pool_score, 0u);
}

DS4X_EXPORT void ds4x_set_expert_hook(ds4_gpu_external_expert_hook hook, void *user) {
    ds4_gpu_set_external_expert_hook(hook, user);
}

/* Optional: in-place second-tier reads for kernel-native packed FP4 experts. */
DS4X_EXPORT void ds4x_set_expert_hook_l2(ds4_gpu_external_expert_hook_l2 hook, void *user) {
    ds4_gpu_set_external_expert_hook_l2(hook, user);
}

/* B5: ds4-win's cache-affinity bias (selection only; original combine weights) on
 * the caller's cache residency. mean 0 disables bias; calibration can observe.
 * Same configure call, warmup,
 * ramp and per-model spread calibration as the native engine. */
DS4X_EXPORT int ds4x_set_affinity(float mean, uint32_t warmup, uint32_t ramp, const char *spread_file,
                                  ds4_gpu_external_residency_hook residency, void *user) {
    if (!g_ds4x.open || !isfinite(mean) || mean < 0.0f || mean > 1.36125f) return 0;
    ds4_gpu_set_external_residency_hook(residency, user);
    const char *calibrate=getenv("DS4_CACHE_AFFINITY_CALIBRATE");
    if (!ds4_gpu_cache_affinity_configure(mean, warmup, ramp, spread_file, calibrate, 0, 0u, 0)) return 0;
    g_cache_affinity_observe=calibrate && *calibrate;
    if ((mean > 0.0f || g_cache_affinity_observe) &&
        !ds4_gpu_cache_affinity_prepare_model(g_ds4x.model.map, g_ds4x.model.size, DS4_N_EXPERT, DS4_N_LAYER)) return 0;
    g_cache_affinity_bias = mean;
    g_ds4x_affinity = (mean > 0.0f || g_cache_affinity_observe) && residency;
    ds4_gpu_external_affinity_phase(g_ds4x_affinity ? 1 : 0);
    fprintf(stderr, "LLAMADECODE_DS4X_AFFINITY mean=%.6g warmup=%u ramp=%u spread=%s enabled=%d\n",
            mean, warmup, ramp, spread_file ? spread_file : "", (int)g_ds4x_affinity);
    return 1;
}

/* One token at `position`, with the caller's Engram history (newest first).
 * Writes the full-vocabulary logits. The device state advances in place. */
DS4X_EXPORT int ds4x_decode(int token, uint32_t position, const int32_t *history_tail, uint32_t history_count,
                            float *logits) {
    if (!g_ds4x.open || !logits || !history_tail || history_count != DS4_ENGRAM_NGRAM - 1) return 0;
    ds41_gpu_graph *g = &g_ds4x.graph;
    g->pos = position;
    g->valid = true;
    for (uint32_t i = 0; i < DS4_ENGRAM_NGRAM - 1; i++) g->history.tail[i] = history_tail[i];
    g_ds41_hc_encode_s = 0.0; g_ds41_hc_calls = g_ds41_hc_replays = 0;
    (void)ds4_gpu_stage_mark(9);
    const bool ok = ds41_graph_step(g, &g_ds4x.model, &g_ds4x.weights, token, logits);
    (void)ds4_gpu_stage_mark(8);
    ds4_gpu_stage_flush(position);
    if (ds41_hc_graph_profile_on()) {
        fprintf(stderr, "PERFEXP_HC_GRAPH_TOKEN position=%u calls=%u replays=%u encode_ms=%.6f enabled=%d\n",
                position, g_ds41_hc_calls, g_ds41_hc_replays, g_ds41_hc_encode_s * 1000.0,
                (int)ds41_single_hc_graphs_on()); fflush(stderr);
    }
    return ok ? 1 : 0;
}

/* LLAMADECODE multi-row append: ds4-win's own layer-major prefill
 * (ds41_graph_prefill, the resident path native ds4-win uses for >= 8-row
 * appends) on the bound state. The workspace holds `rows` rows; it is the
 * native prefill workspace (ds41_graph_ensure_prefill_workspace) sized small
 * instead of the 2048+-row default. Routed experts still come from the
 * caller's cache through the expert hook, one union per row group. */
DS4X_EXPORT int ds4x_prefill_workspace(uint32_t rows, char *error, size_t cap) {
    if (!g_ds4x.open) { ds4x_set_error(error, cap, "ds4x not open"); return 0; }
    if (rows < 2u || rows > 2048u) { ds4x_set_error(error, cap, "ds4x prefill workspace rows must be 2..2048"); return 0; }
    if (g_ds41_cache_bf16) { ds4x_set_error(error, cap, "ds4x multi-row prefill reads F32 owner caches; refused with bf16 owners"); return 0; }
    ds41_gpu_graph *g = &g_ds4x.graph;
    if (g->batch.residual && g->prefill_cap == rows) return 1;
    ds41_graph_free_prefill_workspace(g);
    free(g->prefill_ids); free(g->rows_view); ds4_gpu_tensor_free(g->prefill_tokens);
    g->prefill_ids = NULL; g->rows_view = NULL; g->prefill_tokens = NULL;
    g->prefill_cap = rows;
    g->prefill_ids = malloc((size_t)rows * sizeof(*g->prefill_ids));
    g->rows_view = calloc(rows, sizeof(*g->rows_view));
    g->prefill_tokens = ds4_gpu_tensor_alloc((uint64_t)rows * sizeof(int32_t));
    if (!g->prefill_ids || !g->rows_view || !g->prefill_tokens || !ds41_graph_ensure_prefill_workspace(g)) {
        ds4x_set_error(error, cap, "ds4x prefill workspace allocation failed");
        return 0;
    }
    fprintf(stderr, "LLAMADECODE_DS4X_PREFILL_WORKSPACE rows=%u\n", rows);
    return 1;
}

/* `count` (2..workspace rows) teacher-forced tokens at `position` with the
 * caller's Engram history (newest first); writes the frontier logits only.
 * The device state advances in place exactly as `count` ds4x_decode calls. */
DS4X_EXPORT int ds4x_prefill(const int32_t *tokens, uint32_t count, uint32_t position,
                             const int32_t *history_tail, uint32_t history_count, float *logits) {
    ds41_gpu_graph *g = &g_ds4x.graph;
    if (!g_ds4x.open || !tokens || !logits || !history_tail || history_count != DS4_ENGRAM_NGRAM - 1 ||
        count < 2u || !g->batch.residual || count > g->prefill_cap || position > g->ctx || count > g->ctx - position) return 0;
    if (g_ds41_cache_bf16) { fprintf(stderr, "LLAMADECODE_DS4X_PREFILL_FAILED reason=bf16 owner caches\n"); return 0; }
    g->pos = position;
    g->valid = true;
    for (uint32_t i = 0; i < DS4_ENGRAM_NGRAM - 1; i++) g->history.tail[i] = history_tail[i];
    if (!ds41_graph_prefill(g, &g_ds4x.model, &g_ds4x.weights, tokens, count, NULL, NULL, 0, NULL, NULL, NULL, 0)) {
        fprintf(stderr, "LLAMADECODE_DS4X_PREFILL_FAILED count=%u position=%u\n", count, position);
        return 0;
    }
    return ds41_graph_logits(g, &g_ds4x.model, &g_ds4x.weights, logits) ? 1 : 0;
}

DS4X_EXPORT void ds4x_close(void) {
    if (!g_ds4x.open) return;
    ds4_gpu_cache_affinity_finish();
    ds41_gpu_graph *g = &g_ds4x.graph;
    for (uint32_t i = 0; i < 40; i++) g->window[i] = NULL;
    for (uint32_t i = 0; i < 4; i++) {
        g->compressed[i] = g->index_cache[i] = NULL;
        g->previous_kv[i] = g->previous_score[i] = NULL;
    }
    ds41_graph_free(g);
    ds4_gpu_tensor_free(g_ds4x.spare_kv);
    ds4_gpu_tensor_free(g_ds4x.spare_score);
    ds4_gpu_set_external_expert_hook(NULL, NULL);
    ds4_gpu_set_external_residency_hook(NULL, NULL);
    ds4_gpu_external_affinity_phase(0);
    g_ds4x_affinity = false; g_cache_affinity_bias = 0.0f;g_cache_affinity_observe=false;
    ds4_gpu_external_decode_mode(0);
    model_close(&g_ds4x.model);
    memset(&g_ds4x, 0, sizeof(g_ds4x));
}
#endif
