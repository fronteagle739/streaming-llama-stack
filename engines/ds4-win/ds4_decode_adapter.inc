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
 * ds4-win allocates only its per-token scratch. No streaming, SSD tiers,
 * host caches, prefill workspace or draft model start here. */
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

static void ds4x_view(ds4_gpu_tensor *t, void *ptr, uint64_t bytes) {
    t->ptr = ptr; t->bytes = bytes; t->owner = 0; t->device_id = -1;
}

/* raw[40]: [128][512] F32 rings; compressed/index[4] with their row counts;
 * pool_kv/pool_score[3] for the ratio-2 owners. */
DS4X_EXPORT int ds4x_bind_state(void *const *raw, void *const *compressed, void *const *index,
                                const uint32_t *rows, void *const *pool_kv, void *const *pool_score) {
    if (!g_ds4x.open || !raw || !compressed || !index || !rows || !pool_kv || !pool_score) return 0;
    ds41_gpu_graph *g = &g_ds4x.graph;
    for (uint32_t i = 0; i < 40; i++) {
        if (!raw[i]) return 0;
        ds4x_view(&g_ds4x.window[i], raw[i], 128u * 512u * sizeof(float));
        g->window[i] = &g_ds4x.window[i];
    }
    for (uint32_t i = 0; i < 4; i++) {
        if (!compressed[i] || !index[i] || !rows[i]) return 0;
        ds4x_view(&g_ds4x.compressed[i], compressed[i], (uint64_t)rows[i] * 512u * sizeof(float));
        ds4x_view(&g_ds4x.index_cache[i], index[i], (uint64_t)rows[i] * 128u * sizeof(float));
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
    return 1;
}

DS4X_EXPORT void ds4x_set_expert_hook(ds4_gpu_external_expert_hook hook, void *user) {
    ds4_gpu_set_external_expert_hook(hook, user);
}

/* B5: ds4-win's cache-affinity bias (selection only; original combine weights) on
 * the caller's cache residency. mean 0 disables. Same configure call, warmup,
 * ramp and per-model spread calibration as the native engine. */
DS4X_EXPORT int ds4x_set_affinity(float mean, uint32_t warmup, uint32_t ramp, const char *spread_file,
                                  ds4_gpu_external_residency_hook residency, void *user) {
    if (!g_ds4x.open || !isfinite(mean) || mean < 0.0f || mean > 1.36125f) return 0;
    ds4_gpu_set_external_residency_hook(residency, user);
    if (!ds4_gpu_cache_affinity_configure(mean, warmup, ramp, spread_file, NULL, 0, 0u, 0)) return 0;
    if (mean > 0.0f &&
        !ds4_gpu_cache_affinity_prepare_model(g_ds4x.model.map, g_ds4x.model.size, DS4_N_EXPERT, DS4_N_LAYER)) return 0;
    g_cache_affinity_bias = mean;
    g_ds4x_affinity = mean > 0.0f && residency;
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
    return ds41_graph_step(g, &g_ds4x.model, &g_ds4x.weights, token, logits) ? 1 : 0;
}

DS4X_EXPORT void ds4x_close(void) {
    if (!g_ds4x.open) return;
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
    g_ds4x_affinity = false; g_cache_affinity_bias = 0.0f;
    ds4_gpu_external_decode_mode(0);
    model_close(&g_ds4x.model);
    memset(&g_ds4x, 0, sizeof(g_ds4x));
}
#endif
