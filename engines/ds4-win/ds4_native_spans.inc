typedef struct {
    ds4_gpu_tensor *tensor;
    uint64_t bytes;
    uint32_t kind,index;
} ds41_state_span;

/* The selection masks are rebuilt by each index-source layer on the next
 * token. Only owner caches, windows and unfinished compression pairs persist. */
static uint32_t ds41_state_spans(ds41_gpu_graph *g, uint32_t pos,
                                 ds41_state_span spans[54]) {
    uint32_t n = 0;
    const uint32_t raw = pos < 128u ? pos : 128u;
    for (uint32_t il = 0; il < 40; il++)
        spans[n++] = (ds41_state_span){g->window[il], (uint64_t)raw * 512u * 4u, DSV41_C_RAW, il};
    for (uint32_t i = 0; i < 4; i++) {
        const uint32_t live = pos / (i < 3 ? 2u : 1u);
        spans[n++] = (ds41_state_span){g->compressed[i], (uint64_t)live * 512u * 4u, DSV41_C_COMPRESSED, i};
        spans[n++] = (ds41_state_span){g->index_cache[i], (uint64_t)live * 128u * 4u, DSV41_C_INDEX, i};
        if (i < 3 && (pos & 1u)) {
            spans[n++] = (ds41_state_span){g->previous_kv[i], 512u * 4u, DSV41_C_POOL_KV, i};
            spans[n++] = (ds41_state_span){g->previous_score[i], 512u * 4u, DSV41_C_POOL_SCORE, i};
        }
    }
    return n;
}

