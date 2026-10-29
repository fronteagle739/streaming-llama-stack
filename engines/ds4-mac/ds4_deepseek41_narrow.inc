/* Exact local-weight readers ported from f48c5e2e20b1aaae1c79735a8457225d2469ae73. */
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
#include <dispatch/dispatch.h>
#endif
/* NARROW_WEIGHTS: narrow decode copies built at load from the GGUF being loaded (after Ivan Fioravanti's
 * builder, extended to the indexer q_b): the router as BF16 (the F32 tensor's high halves, every low half zero),
 * and the Engram projections and the index-source indexer q_b as E4M3 codes with one E8M0 byte per 32x32 block, every
 * F16 weight re-encoded exactly at its block scale. A family whose tensors do not narrow exactly keeps its wide reads.
 * Exact copies only. Model-wide, built once by the
 * first session, freed with the engine. */
typedef struct {
    ds4_gpu_tensor *codes, *e8m0;   /* [rows][cols] codes; [rows/32][cols/32] bytes, e + 127 */
} ds41_fp8_copy;

#define DS41_NARROW_LAYERS 64u   /* >= DS4_N_LAYER (a runtime value) */
static struct {
    const ds4_model *owner;
    ds4_gpu_tensor *router[DS41_NARROW_LAYERS];
    ds41_fp8_copy engram[2], indexer[DS41_NARROW_LAYERS];
} g_ds41_narrow;

static void ds41_fp8_copy_free(ds41_fp8_copy *c) {
    ds4_gpu_tensor_free(c->codes);
    ds4_gpu_tensor_free(c->e8m0);
    memset(c, 0, sizeof(*c));
}

static void ds41_narrow_free(void) {
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    if (g_ds41_narrow.owner) ds4_gpu_weight_residency(NULL, 0);
#endif
    for (uint32_t il = 0; il < DS41_NARROW_LAYERS; il++) {
        ds4_gpu_tensor_free(g_ds41_narrow.router[il]);
        ds41_fp8_copy_free(&g_ds41_narrow.indexer[il]);
    }
    for (uint32_t i = 0; i < 2u; i++) ds41_fp8_copy_free(&g_ds41_narrow.engram[i]);
    memset(&g_ds41_narrow, 0, sizeof(g_ds41_narrow));
}

#if defined(__APPLE__) && defined(__aarch64__) && !defined(DS4_NO_GPU)
/* E4M3 code of q when q is exactly representable (|q| <= 448, three mantissa bits, subnormals in steps of 2^-9). */
static bool ds41_e4m3_code(float q, uint8_t *code) {
    const uint8_t sign = signbit(q) ? 0x80u : 0u;
    const float a = fabsf(q);
    if (a == 0.0f) { *code = sign; return true; }
    if (!(a <= 448.0f)) return false;
    int e;
    const float f = frexpf(a, &e);   /* a = f * 2^e, f in [0.5, 1) */
    if (e - 1 >= -6) {
        const float mm = f * 16.0f - 8.0f;   /* 1.mmm: mm in [0, 8) */
        if (mm != floorf(mm)) return false;
        *code = (uint8_t)(sign | (uint8_t)((e - 1 + 7) << 3) | (uint8_t)mm);
        return true;
    }
    const float mm = a * 512.0f;
    if (mm != floorf(mm) || mm < 1.0f || mm > 7.0f) return false;
    *code = (uint8_t)(sign | (uint8_t)mm);
    return true;
}

typedef struct {
    const __fp16 *src;
    uint64_t cols;
    uint8_t *codes, *e8m0;
    bool failed;   /* shared by the dispatch_apply workers: atomic accesses only */
} ds41_e4m3_job;

/* One row of 32x32 blocks: the scale that fits the block's largest value, or one step either side, must hold every
 * weight exactly; its E8M0 byte is e + 127 (never 255). */
static void ds41_e4m3_block_row(void *context, size_t br) {
    ds41_e4m3_job *job = context;
    const uint64_t cols = job->cols, bc = cols / 32u;
    for (uint64_t bj = 0; bj < bc; bj++) {
        if (__atomic_load_n(&job->failed, __ATOMIC_RELAXED)) return;   /* another row already failed the copy */
        float amax = 0.0f;
        for (uint32_t r = 0; r < 32u; r++)
            for (uint32_t j = 0; j < 32u; j++) {
                const float v = fabsf((float)job->src[(br * 32u + r) * cols + bj * 32u + j]);
                if (v > amax) amax = v;
            }
        const int e0 = amax > 0.0f ? (int)ceilf(log2f(amax / 448.0f)) : 0;
        bool fit = false;
        for (int d = 0; !fit && d < 3; d++) {
            const int e = e0 + (d == 1 ? 1 : d == 2 ? -1 : 0);
            if (e < -127 || e > 127) continue;
            const float inv = ldexpf(1.0f, -e);
            fit = true;
            for (uint32_t r = 0; fit && r < 32u; r++)
                for (uint32_t j = 0; fit && j < 32u; j++) {
                    const uint64_t i = (br * 32u + r) * cols + bj * 32u + j;
                    fit = ds41_e4m3_code((float)job->src[i] * inv, &job->codes[i]);
                }
            if (fit) job->e8m0[br * bc + bj] = (uint8_t)(e + 127);
        }
        if (!fit) { __atomic_store_n(&job->failed, true, __ATOMIC_RELAXED); return; }
    }
}

/* The F16 tensor t [rows][cols] as E4M3 codes + E8M0 block bytes, or false (nothing kept) unless every block is exact. */
static bool ds41_fp8_copy_build(const ds4_model *m, const ds4_tensor *t, ds41_fp8_copy *out) {
    memset(out, 0, sizeof(*out));
    if (!t || t->type != DS4_TENSOR_F16 || t->dim[0] % 32u || t->dim[1] % 32u) return false;
    const uint64_t cols = t->dim[0], rows = t->dim[1], nb = (rows / 32u) * (cols / 32u);
    ds41_fp8_copy c = {ds4_gpu_tensor_alloc_weights(rows * cols), ds4_gpu_tensor_alloc_weights(nb)};
    ds41_e4m3_job job = {(const __fp16 *)(m->map + t->abs_offset), cols,
        c.codes ? ds4_gpu_tensor_contents(c.codes) : NULL, c.e8m0 ? ds4_gpu_tensor_contents(c.e8m0) : NULL, false};
    bool ok = job.codes && job.e8m0;
    if (ok) dispatch_apply_f(rows / 32u, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &job, ds41_e4m3_block_row);
    ok = ok && !__atomic_load_n(&job.failed, __ATOMIC_RELAXED);   /* dispatch_apply_f returned: every row is done */
    if (!ok) { ds41_fp8_copy_free(&c); return false; }
    *out = c;
    return true;
}
#endif

/* Build the copies for model m once (single-host, non-streaming sessions). */
static void ds41_narrow_build(const ds4_model *m, const ds4_weights *w) {
#if defined(__APPLE__) && defined(__aarch64__) && !defined(DS4_NO_GPU)
    /* Build independently of the runtime reader switch so /debug/levers can
     * toggle it in either direction without an inert enable-after-load case. */
    if (g_ds41_narrow.owner == m || DS4_N_LAYER > DS41_NARROW_LAYERS) return;
    ds41_narrow_free();
    g_ds41_narrow.owner = m;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_tensor *t = w->layer[il].ffn_gate_inp;
        if (!t) continue;
        const uint64_t n = t->dim[0] * t->dim[1];
        const uint32_t *src = (const uint32_t *)(m->map + t->abs_offset);
        bool exact = t->type == DS4_TENSOR_F32;
        for (uint64_t i = 0; exact && i < n; i++) exact = (src[i] & 0xffffu) == 0;
        ds4_gpu_tensor *dst = exact ? ds4_gpu_tensor_alloc_weights(n * sizeof(uint16_t)) : NULL;
        uint16_t *o = dst ? ds4_gpu_tensor_contents(dst) : NULL;
        if (!o) {
            ds4_gpu_tensor_free(dst);
            for (uint32_t j = 0; j < il; j++) { ds4_gpu_tensor_free(g_ds41_narrow.router[j]); g_ds41_narrow.router[j] = NULL; }
            break;
        }
        for (uint64_t i = 0; i < n; i++) o[i] = (uint16_t)(src[i] >> 16);
        g_ds41_narrow.router[il] = dst;
    }
    for (uint32_t i = 0; i < 2u; i++) {
        ds41_fp8_copy *c = &g_ds41_narrow.engram[i];
        const ds4_layer_weights *l = &w->layer[i == 0 ? 1 : 14];
        if (!ds41_fp8_copy_build(m, l->engram_kv, c)) {
            ds41_fp8_copy_free(&g_ds41_narrow.engram[0]);
            ds41_fp8_copy_free(&g_ds41_narrow.engram[1]);
            break;
        }
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!ds41_index_source(il)) continue;
        ds41_fp8_copy *c = &g_ds41_narrow.indexer[il];
        const ds4_layer_weights *l = &w->layer[il];
        if (!ds41_fp8_copy_build(m, l->indexer_attn_q_b, c)) {
            for (uint32_t j = 0; j < DS41_NARROW_LAYERS; j++) ds41_fp8_copy_free(&g_ds41_narrow.indexer[j]);
            break;
        }
    }
    ds4_gpu_tensor *all[DS41_NARROW_LAYERS * 3u + 4u];
    uint32_t n_all = 0;
    for (uint32_t il = 0; il < DS41_NARROW_LAYERS; il++) {
        if (g_ds41_narrow.router[il]) all[n_all++] = g_ds41_narrow.router[il];
        if (g_ds41_narrow.indexer[il].codes) {
            all[n_all++] = g_ds41_narrow.indexer[il].codes;
            all[n_all++] = g_ds41_narrow.indexer[il].e8m0;
        }
    }
    for (uint32_t i = 0; i < 2u; i++)
        if (g_ds41_narrow.engram[i].codes) {
            all[n_all++] = g_ds41_narrow.engram[i].codes;
            all[n_all++] = g_ds41_narrow.engram[i].e8m0;
        }
    if (!ds4_gpu_weight_residency(all, n_all))
        fprintf(stderr, "ds4: exact narrow weight residency request failed; tracked buffers remain valid\n");
    uint32_t routers = 0, indexers = 0, engrams = 0;
    for (uint32_t il = 0; il < DS41_NARROW_LAYERS; il++) {
        routers += g_ds41_narrow.router[il] != NULL;
        indexers += g_ds41_narrow.indexer[il].codes != NULL;
    }
    for (uint32_t i = 0; i < 2u; i++) engrams += g_ds41_narrow.engram[i].codes != NULL;
    fprintf(stderr, "ds4: exact local narrow copies: router=%u/%u indexer=%u Engram=%u/2; other tensors keep original readers\n",
            routers, (uint32_t)DS4_N_LAYER, indexers, engrams);
#else
    (void)m; (void)w;
#endif
}

/* The decode readers of the copies: true when a copy served the matvec. */
static DS4_MAYBE_UNUSED bool ds41_narrow_on(const ds41_gpu_graph *g, const ds4_model *m) {
    return g_ds41_narrow.owner == m && g->tp_world == 1 && !g->streaming && !g->quality &&
        g_ds41_levers.narrow_weights;
}

static bool ds41_narrow_router_mv(ds41_gpu_graph *g, const ds4_model *m, uint32_t il) {
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    return ds41_narrow_on(g, m) && il < DS41_NARROW_LAYERS && g_ds41_narrow.router[il] &&
        ds4_gpu_matmul_bf16_router_weights(g->route_logits, g_ds41_narrow.router[il], DS4_N_EMBD, DS4_N_EXPERT, g->norm);
#else
    (void)g; (void)m; (void)il;
    return false;
#endif
}

static bool ds41_narrow_engram_mv(ds41_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *l, uint32_t table) {
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    const ds41_fp8_copy *c = &g_ds41_narrow.engram[table];
    return ds41_narrow_on(g, m) && c->codes &&
        ds4_gpu_matmul_engram_fp8_weights(g->engram_kv, c->codes, c->e8m0, (uint32_t)l->engram_kv->dim[0],
                                          (uint32_t)l->engram_kv->dim[1], ds41_engram_input(g, table));
#else
    (void)g; (void)m; (void)l; (void)table;
    return false;
#endif
}

static bool ds41_narrow_indexer_mv(ds41_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *l, uint32_t il) {
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    const ds41_fp8_copy *c = il < DS41_NARROW_LAYERS ? &g_ds41_narrow.indexer[il] : NULL;
    return ds41_narrow_on(g, m) && c && c->codes &&
        ds4_gpu_dsv41_project_fp8_bf16_weights(g->index_q, c->codes, c->e8m0, (uint32_t)l->indexer_attn_q_b->dim[0],
                                               (uint32_t)l->indexer_attn_q_b->dim[1], g->qr);
#else
    (void)g; (void)m; (void)l; (void)il;
    return false;
#endif
}
