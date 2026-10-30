/* Bounded CPU fixture for the upstream/native wire adapter. No model, GPU,
 * server or filesystem artifact; the temporary streams are anonymous. */
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef __APPLE__
#define __APPLE__ 1
#endif
#define DS4_N_VOCAB 129280u
#define DS4_SESSION_PAYLOAD_MAGIC 0x34565344u
#define DS4_SESSION_PAYLOAD_VERSION 2u
#define DS4_SESSION_IO_CHUNK 65536u
#define DS4_V41_BF16 1
typedef struct { unsigned char *data; uint64_t bytes; } ds4_gpu_tensor;
typedef struct { uint32_t target_layers[3]; } draft_weights;
typedef struct {
    const draft_weights *weights;
    ds4_gpu_tensor *target[3], *mean; uint32_t positions[3][128];
    bool kv_valid; uint32_t kv_end, verify_count, rewind_end;
} ds41_dspark;
typedef struct {
    uint32_t ctx, pos, tp_world; bool valid, sel_done, sel_want, kg_want, index_bf16_ok;
    ds41_dspark *dspark; ds4_gpu_tensor *logits;
    ds4_gpu_tensor *window[40], *compressed[4], *index_cache[4], *previous_kv[4], *previous_score[4];
    struct { int32_t tail[3]; } history; uint32_t *token_map;
} ds41_gpu_graph;
typedef struct { ds4_gpu_tensor *tensor; uint64_t bytes; } ds41_state_span;
typedef struct { int *v, len, cap; } ds4_tokens;
typedef struct {
    ds41_gpu_graph ds41_graph; bool ds41_graph_ready, checkpoint_valid, mtp_draft_valid, sel_bound;
    ds4_tokens checkpoint; float *logits; bool (*cancel)(void *); void *cancel_ud;
} ds4_session;
static unsigned reset_calls, derived_calls, invalidations;
static int commands_open;
static int ds4_gpu_begin_commands(void) { assert(!commands_open); commands_open = 1; return 1; }
static int ds4_gpu_end_commands(void) { assert(commands_open); commands_open = 0; return 1; }
static int ds4_gpu_tensor_read(ds4_gpu_tensor *t, uint64_t off, void *p, size_t n) {
    if (!t || off > t->bytes || n > t->bytes - off) return 0;
    memcpy(p, t->data + off, n); return 1;
}
static int ds4_gpu_tensor_write(ds4_gpu_tensor *t, uint64_t off, const void *p, size_t n) {
    if (!t || off > t->bytes || n > t->bytes - off) return 0;
    memcpy(t->data + off, p, n); return 1;
}
static int ds4_gpu_tensor_copy(ds4_gpu_tensor *d, uint64_t off, const ds4_gpu_tensor *s, uint64_t src, uint64_t n) {
    assert(commands_open); /* Metal copies require an owned command buffer. */
    if (!d || !s || off > d->bytes || n > d->bytes - off || src > s->bytes || n > s->bytes - src) return 0;
    memmove(d->data + off, s->data + src, (size_t)n); return 1;
}
static uint32_t bf16(uint32_t v) { return (v + 0x7fffu + ((v >> 16) & 1u)) & 0xffff0000u; }
static int ds4_gpu_dsv41_quantize(ds4_gpu_tensor *t, uint32_t width, uint32_t rows, int kind) {
    assert(kind == DS4_V41_BF16); uint64_t count = (uint64_t)width * rows;
    if (count * 4u > t->bytes) return 0;
    for (uint64_t i = 0; i < count; ++i) { uint32_t v; memcpy(&v, t->data + i * 4, 4); v = bf16(v); memcpy(t->data + i * 4, &v, 4); }
    return 1;
}
static int ds4_gpu_synchronize(void) { return 1; }
static void ds41_graph_reset(ds41_gpu_graph *g) {
    ++reset_calls; g->pos = 0; g->valid = true;
    memset(g->dspark->positions, 0xff, sizeof(g->dspark->positions));
    g->dspark->kv_valid = false; g->dspark->kv_end = 0;
    g->dspark->verify_count = g->dspark->rewind_end = 0;
}
static void ds41_index_bf16_restore(ds41_gpu_graph *g, uint32_t pos) {
    assert(g->pos == pos); ++derived_calls; g->index_bf16_ok = true;
}
static void ds4_session_dspark_capture_invalidate(ds4_session *s) { (void)s; ++invalidations; }
static void ds4_tokens_free(ds4_tokens *t) { free(t->v); memset(t, 0, sizeof(*t)); }
static void token_vec_push(ds4_tokens *t, int v) {
    if (t->len == t->cap) { t->cap = t->cap ? t->cap * 2 : 256; t->v = realloc(t->v, (size_t)t->cap * 4); assert(t->v); }
    t->v[t->len++] = v;
}
static void payload_set_err(char *e, size_t n, const char *m) { if (e && n) snprintf(e, n, "%s", m); }
static int payload_read_bytes(FILE *f, void *p, uint64_t n, uint64_t *remaining, char *e, size_t cap) {
    if (n > *remaining || fread(p, 1, (size_t)n, f) != n) { payload_set_err(e, cap, "short payload"); return 1; }
    *remaining -= n; return 0;
}
static int payload_read_u32(FILE *f, uint32_t *v, uint64_t *remaining, char *e, size_t cap) {
    unsigned char p[4]; if (payload_read_bytes(f, p, 4, remaining, e, cap)) return 1;
    *v = (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; return 0;
}
static int payload_write_bytes(FILE *f, const void *p, uint64_t n, char *e, size_t cap) {
    if (fwrite(p, 1, (size_t)n, f) != n) { payload_set_err(e, cap, "write failed"); return 1; } return 0;
}
static int payload_write_u32(FILE *f, uint32_t v, char *e, size_t cap) {
    unsigned char p[4]; for (unsigned i = 0; i < 4; ++i) p[i] = (unsigned char)(v >> (8u * i));
    return payload_write_bytes(f, p, 4, e, cap);
}
static int payload_write_tensor_span(FILE *f, ds4_gpu_tensor *t, uint64_t off, uint64_t bytes,
                                     unsigned char *page, size_t chunk, char *e, size_t cap) {
    while (bytes) {
        size_t n = bytes < chunk ? (size_t)bytes : chunk;
        if (!ds4_gpu_tensor_read(t, off, page, n) || payload_write_bytes(f, page, n, e, cap)) return 1;
        bytes -= n; off += n;
    }
    return 0;
}
static uint32_t ds41_state_spans(ds41_gpu_graph *g, uint32_t pos, ds41_state_span out[54]) {
    uint32_t n = 0, raw = pos < 128 ? pos : 128;
    for (unsigned i = 0; i < 40; ++i) out[n++] = (ds41_state_span){g->window[i], (uint64_t)raw * 512u * 4u};
    for (unsigned i = 0; i < 4; ++i) {
        uint32_t live = pos / (i < 3 ? 2 : 1);
        out[n++] = (ds41_state_span){g->compressed[i], (uint64_t)live * 512u * 4u};
        out[n++] = (ds41_state_span){g->index_cache[i], (uint64_t)live * 128u * 4u};
        if (i < 3 && (pos & 1)) {
            out[n++] = (ds41_state_span){g->previous_kv[i], 512u * 4u};
            out[n++] = (ds41_state_span){g->previous_score[i], 512u * 4u};
        }
    }
    return n;
}
#include "../engines/ds4-mac/ds4_native_upstream.inc"

static ds4_gpu_tensor *tensor(uint64_t bytes, unsigned seed) {
    ds4_gpu_tensor *t = calloc(1, sizeof(*t)); assert(t); t->bytes = bytes;
    t->data = malloc((size_t)bytes); assert(t->data);
    for (uint64_t i = 0; i < bytes; ++i) t->data[i] = (unsigned char)(seed + i * 17u + (i >> 8));
    return t;
}
static void tensor_free(ds4_gpu_tensor *t) { free(t->data); free(t); }
static void init(ds4_session *s, uint32_t pos) {
    memset(s, 0, sizeof(*s)); ds41_gpu_graph *g = &s->ds41_graph;
    static const draft_weights weights = {{37, 38, 39}};
    g->ctx = 256; g->pos = pos; g->tp_world = 1; g->valid = true;
    g->dspark = calloc(1, sizeof(*g->dspark)); assert(g->dspark); g->dspark->weights = &weights;
    g->token_map = calloc(DS4_N_VOCAB, sizeof(uint32_t)); assert(g->token_map);
    for (uint32_t i = 0; i < DS4_N_VOCAB; ++i) g->token_map[i] = i + 3;
    for (unsigned i = 0; i < 3; ++i) g->dspark->target[i] = tensor(128u * 5120u * 4u, 17u + i * 41u);
    g->dspark->mean = tensor(128u * 5120u * 4u, 0);
    /* Finite signed values exercise ties, carry into exponent and exact BF16
     * values while remaining valid model-state inputs. */
    for (unsigned stage = 0; stage < 3; ++stage)
        for (uint32_t i = 0; i < 128u * 5120u; ++i) {
            uint32_t bits = ((i & 1u) << 31) | 0x3f000000u | ((i * 7919u + stage * 32768u) & 0x007fffffu);
            memcpy(g->dspark->target[stage]->data + (uint64_t)i * 4u, &bits, 4);
        }
    for (unsigned i = 0; i < 40; ++i) g->window[i] = tensor(128u * 512u * 4u, i);
    for (unsigned i = 0; i < 4; ++i) {
        g->compressed[i] = tensor(256u * 512u * 4u, 55 + i);
        g->index_cache[i] = tensor(256u * 128u * 4u, 67 + i);
        g->previous_kv[i] = tensor(512u * 4u, 71 + i);
        g->previous_score[i] = tensor(512u * 4u, 79 + i);
    }
    g->logits = tensor(DS4_N_VOCAB * 4u, 0);
    s->logits = calloc(DS4_N_VOCAB, sizeof(float)); assert(s->logits);
    for (uint32_t i = 0; i < DS4_N_VOCAB; ++i) s->logits[i] = (float)i / 129280.0f;
    for (uint32_t i = 0; i < pos; ++i) token_vec_push(&s->checkpoint, (int)i);
    assert(ds41_native_commit(s, &s->checkpoint));
    s->ds41_graph_ready = s->checkpoint_valid = true;
}
static void close_session(ds4_session *s) {
    ds41_gpu_graph *g = &s->ds41_graph;
    for (unsigned i = 0; i < 3; ++i) tensor_free(g->dspark->target[i]);
    for (unsigned i = 0; i < 40; ++i) tensor_free(g->window[i]);
    for (unsigned i = 0; i < 4; ++i) {
        tensor_free(g->compressed[i]); tensor_free(g->index_cache[i]);
        tensor_free(g->previous_kv[i]); tensor_free(g->previous_score[i]);
    }
    tensor_free(g->logits); tensor_free(g->dspark->mean); free(g->dspark); free(g->token_map); free(s->logits); ds4_tokens_free(&s->checkpoint);
}
static void test_all_ring_words_and_seed_order(ds4_session *s) {
    ds41_gpu_graph *g = &s->ds41_graph; const uint64_t ring_bytes = 128u * 3u * 5120u * 4u;
    unsigned char *wire = malloc((size_t)ring_bytes); assert(wire);
    for (uint64_t off = 0; off < ring_bytes;) {
        size_t n = (size_t)(ring_bytes - off); if (n > 65536) n = 65536;
        assert(ds41_native_owner_io(g, 131, 0, off, wire + off, n, false)); off += n;
    }
    for (uint32_t slot = 0; slot < 128; ++slot)
        for (unsigned stage = 0; stage < 3; ++stage)
            assert(!memcmp(wire + ((uint64_t)slot * 3u + stage) * 5120u * 4u,
                           g->dspark->target[stage]->data + (uint64_t)slot * 5120u * 4u, 5120u * 4u));
    const uint32_t starts[] = {0, 3, 126, 127, 128, 130};
    const uint32_t counts[] = {1, 2, 6, 127, 128};
    ds4_gpu_tensor *input = tensor(ring_bytes, 0);
    for (unsigned a = 0; a < sizeof(starts) / sizeof(*starts); ++a)
        for (unsigned b = 0; b < sizeof(counts) / sizeof(*counts); ++b) {
            uint32_t count = counts[b];
            assert(ds4_gpu_begin_commands());
            for (uint32_t i = 0; i < count; ++i)
                for (unsigned stage = 0; stage < 3; ++stage)
                    assert(ds4_gpu_tensor_copy(input, ((uint64_t)i * 3u + stage) * 5120u * 4u,
                        g->dspark->target[stage], ((starts[a] + i) % 128u) * 5120u * 4u, 5120u * 4u));
            assert(ds4_gpu_dsv41_quantize(input, 3u * 5120u, count, DS4_V41_BF16));
            assert(ds4_gpu_end_commands());
            for (uint32_t i = 0; i < count; ++i)
                for (unsigned stage = 0; stage < 3; ++stage)
                    for (uint32_t j = 0; j < 5120; ++j) {
                        uint32_t raw, actual;
                        memcpy(&raw, wire + (((uint64_t)(starts[a] + i) % 128u * 3u + stage) * 5120u + j) * 4u, 4);
                        memcpy(&actual, input->data + (((uint64_t)i * 3u + stage) * 5120u + j) * 4u, 4);
                        assert(actual == bf16(raw));
                    }
        }
    /* Verify rollback's ordinary tensor copies restore every raw word, even
     * across slot127/0, before any seed quantization sees it. */
    unsigned char saved[5120u * 4u];
    for (unsigned stage = 0; stage < 3; ++stage)
        for (uint32_t position = 126; position < 132; ++position) {
            uint64_t off = position % 128u * 5120u * 4u;
            assert(ds4_gpu_tensor_read(g->dspark->target[stage], off, saved, sizeof(saved)));
            memset(g->dspark->target[stage]->data + off, 0xa5, sizeof(saved));
            assert(ds4_gpu_tensor_write(g->dspark->target[stage], off, saved, sizeof(saved)));
        }
    for (uint32_t slot = 0; slot < 128; ++slot)
        for (unsigned stage = 0; stage < 3; ++stage)
            assert(!memcmp(wire + ((uint64_t)slot * 3u + stage) * 5120u * 4u,
                           g->dspark->target[stage]->data + (uint64_t)slot * 5120u * 4u, 5120u * 4u));
    tensor_free(input); free(wire);
}
static void test_mode2_staging(ds4_session *s) {
    unsigned char page[65536]; char err[128] = {0};
    for (unsigned stage = 0; stage < 3; ++stage) {
        FILE *f = tmpfile(); assert(f);
        ds4_gpu_tensor *raw = s->ds41_graph.dspark->target[stage];
        unsigned char *before = malloc((size_t)raw->bytes); assert(before); memcpy(before, raw->data, (size_t)raw->bytes);
        assert(!ds41_native_write_mode2_ring(f, s->ds41_graph.dspark, stage, page, err, sizeof(err)));
        assert((uint64_t)ftell(f) == raw->bytes); rewind(f);
        for (uint64_t off = 0; off < raw->bytes; off += 4) {
            uint32_t loaded, expected; assert(fread(&loaded, 4, 1, f) == 1);
            memcpy(&expected, before + off, 4);
            assert(loaded == bf16(expected)); assert(bf16(loaded) == loaded);
        }
        assert(!memcmp(before, raw->data, (size_t)raw->bytes)); free(before); fclose(f);
    }
}
static void test_owner_layout(ds4_session *s) {
    ds41_gpu_graph *g = &s->ds41_graph;
    dsv41_c_plan delta;
    assert(dsv41_c_delta(g->ctx, 126, 131, &delta));
    assert(delta.span_count == 55 && delta.slices[0].span == 0 &&
           delta.slices[0].offset == 126u * 3u * 5120u * 4u &&
           delta.slices[0].bytes == 2u * 3u * 5120u * 4u &&
           delta.slices[1].span == 0 && delta.slices[1].offset == 0 &&
           delta.slices[1].bytes == 3u * 3u * 5120u * 4u);
    assert(!dsv41_c_delta(g->ctx, 131, 126, &delta));
    unsigned char actual[65536], expected[65536], replacement[65536];
    const uint64_t offsets[] = {0, 20477, 40957, 61437, 127u * 61440u};
    for (unsigned k = 0; k < sizeof(offsets) / sizeof(*offsets); ++k) {
        const uint64_t off = offsets[k], extent = 128u * 61440u;
        size_t n = sizeof(actual); if (n > extent - off) n = (size_t)(extent - off);
        for (size_t i = 0; i < n; ++i) {
            const uint64_t address = off + i, row = address / 20480u;
            expected[i] = g->dspark->target[row % 3u]->data[row / 3u * 20480u + address % 20480u];
            replacement[i] = (unsigned char)(i ^ 0xa5u);
        }
        assert(ds41_native_owner_io(g, 131, 0, off, actual, n, false));
        assert(!memcmp(actual, expected, n));
        assert(ds41_native_owner_io(g, 131, 0, off, replacement, n, true));
        for (size_t i = 0; i < n; ++i) {
            const uint64_t address = off + i, row = address / 20480u;
            assert(g->dspark->target[row % 3u]->data[row / 3u * 20480u + address % 20480u] == replacement[i]);
        }
        assert(ds41_native_owner_io(g, 131, 0, off, expected, n, true));
    }
    assert(!ds41_native_owner_io(g, 131, 0, 128u * 61440u, actual, 1, false));
    assert(!ds41_native_owner_io(g, 131, 55, 0, actual, 1, false));
    assert(!ds41_native_owner_io(g, 131, 0, 0, actual, 65537, false));
    assert(ds41_native_positions(g, 131)); g->dspark->positions[1][130 % 128] = 0;
    assert(!ds41_native_positions(g, 131)); g->dspark->positions[1][130 % 128] = 130;
}
static void test_payload(ds4_session *s) {
    FILE *f = tmpfile(); assert(f); char err[128] = {0};
    assert(!ds41_native_save_payload(s, f, err, sizeof(err)));
    const uint64_t expected = ds41_native_payload_body_bytes(&s->ds41_graph, 131);
    assert((uint64_t)ftell(f) == 52u + expected); rewind(f);
    uint32_t h[13]; uint64_t remaining = 52u + expected;
    for (unsigned i = 0; i < 13; ++i) assert(!payload_read_u32(f, h + i, &remaining, err, sizeof(err)));
    s->sel_bound = true; s->ds41_graph.sel_done = s->ds41_graph.sel_want = s->ds41_graph.kg_want = true;
    s->ds41_graph.dspark->kv_valid = true;
    assert(!ds41_native_load_payload(s, f, h, remaining, err, sizeof(err)));
    assert(s->checkpoint_valid && ds41_native_positions(&s->ds41_graph, 131));
    assert(!s->sel_bound && !s->ds41_graph.sel_done && !s->ds41_graph.sel_want && !s->ds41_graph.kg_want);
    assert(!s->ds41_graph.dspark->kv_valid);
    assert(!memcmp(s->ds41_graph.logits->data, s->logits, DS4_N_VOCAB * 4u));
    assert(s->ds41_graph.history.tail[0] == 133 && reset_calls >= 2 && derived_calls >= 2 && invalidations == 1);
    assert(!fseek(f, 52, SEEK_SET)); h[3] = 2;
    assert(ds41_native_load_payload(s, f, h, expected, err, sizeof(err)));
    assert(s->checkpoint_valid); h[3] = 3;
    assert(!fseek(f, 52, SEEK_SET)); assert(!payload_write_u32(f, 130, err, sizeof(err)));
    assert(!fseek(f, 52, SEEK_SET));
    assert(ds41_native_load_payload(s, f, h, expected, err, sizeof(err)));
    assert(s->checkpoint_valid);
    assert(!fseek(f, 52, SEEK_SET)); assert(!payload_write_u32(f, 131, err, sizeof(err)));
    const uint32_t bad_values[] = {0x7fc00000u, 0x7f800000u, 0xff800000u, 0x7f800001u};
    for (unsigned i = 0; i < sizeof(bad_values) / sizeof(*bad_values); ++i) {
        assert(!fseek(f, 60 + 131 * 4, SEEK_SET));
        assert(!payload_write_u32(f, bad_values[i], err, sizeof(err)));
        assert(!fseek(f, 52, SEEK_SET)); err[0] = 0;
        assert(ds41_native_load_payload(s, f, h, expected, err, sizeof(err)));
        assert(!s->checkpoint_valid && !s->ds41_graph.valid);
        assert(strstr(err, "nonfinite native mode3 logits"));
    }
    fclose(f);
}
int main(void) {
    ds4_session s; init(&s, 131); test_owner_layout(&s); test_all_ring_words_and_seed_order(&s);
    test_mode2_staging(&s); test_payload(&s); close_session(&s);
    puts("native upstream mode3 owner layout, snapshot, and sampler invalidation PASS"); return 0;
}
