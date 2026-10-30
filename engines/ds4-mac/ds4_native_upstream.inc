/* Native-C mode3 is an external wire contract, not upstream's mode2 snapshot.
 * Its first owner interleaves [ring slot][target stage][embedding]. Upstream
 * owns three separate stage tensors. Translate bounded I/O, never duplicate
 * the ring or replace the upstream graph. Include after ds41_state_spans. */
#ifndef DS4_NATIVE_UPSTREAM_INC
#define DS4_NATIVE_UPSTREAM_INC
#include "dsv41-native-c-layout.h"

#if defined(__APPLE__) && !defined(DS4_NO_GPU)
static bool ds41_native_draft(const ds41_gpu_graph *g) {
    if (!g || g->tp_world != 1 || !g->dspark || !g->dspark->weights) return false;
    for (unsigned i = 0; i < 3; ++i)
        if (!g->dspark->target[i] || g->dspark->weights->target_layers[i] != 37u + i)
            return false;
    return true;
}

static bool ds41_native_positions(const ds41_gpu_graph *g, uint32_t pos) {
    if (!ds41_native_draft(g) || !pos || pos >= g->ctx) return false;
    const uint32_t first = pos > 128u ? pos - 128u : 0u;
    for (unsigned stage = 0; stage < 3; ++stage)
        for (uint32_t p = first; p < pos; ++p)
            if (g->dspark->positions[stage][p % 128u] != p) return false;
    return true;
}

static bool ds41_native_geometry(ds41_gpu_graph *g, uint32_t pos,
                                 ds41_state_span target[54],
                                 dsv41_c_span wire[55], uint32_t *count) {
    if (!ds41_native_draft(g) || !dsv41_c_layout(g->ctx, pos, wire, count)) return false;
    if (ds41_state_spans(g, pos, target) + 1u != *count) return false;
    for (uint32_t i = 1; i < *count; ++i)
        if (target[i - 1u].bytes != wire[i].bytes) return false;
    return true;
}

static int ds41_native_owner_io(ds41_gpu_graph *g, uint32_t pos, uint32_t owner,
                                uint64_t offset, void *buffer, size_t bytes, bool write) {
    ds41_state_span target[54]; dsv41_c_span wire[55]; uint32_t count = 0;
    if (!buffer || !bytes || bytes > 65536u ||
        !ds41_native_geometry(g, pos, target, wire, &count) || owner >= count ||
        offset > wire[owner].bytes || bytes > wire[owner].bytes - offset) return 0;
    if (owner) {
        ds4_gpu_tensor *t = target[owner - 1u].tensor;
        return write ? ds4_gpu_tensor_write(t, offset, buffer, bytes) :
                       ds4_gpu_tensor_read(t, offset, buffer, bytes);
    }
    const uint64_t row_bytes = 5120u * 4u;
    unsigned char *p = buffer;
    while (bytes) {
        const uint64_t row = offset / row_bytes, within = offset % row_bytes;
        const unsigned stage = (unsigned)(row % 3u);
        const uint64_t tensor_offset = (row / 3u) * row_bytes + within;
        size_t n = (size_t)(row_bytes - within); if (n > bytes) n = bytes;
        int ok = write ? ds4_gpu_tensor_write(g->dspark->target[stage], tensor_offset, p, n) :
                         ds4_gpu_tensor_read(g->dspark->target[stage], tensor_offset, p, n);
        if (!ok) return 0;
        bytes -= n; offset += n; p += n;
    }
    return 1;
}

/* A replaced frontier invalidates both CPU and device sampling summaries.
 * The caller keeps checkpoint_valid false until this completes. */
static int ds41_native_commit(ds4_session *s, const ds4_tokens *tokens) {
    ds41_gpu_graph *g = &s->ds41_graph;
    if (!tokens || tokens->len <= 0 || !ds41_native_draft(g)) return 0;
    const uint32_t pos = (uint32_t)tokens->len;
    s->sel_bound = false;
    if (!ds4_gpu_tensor_write(g->logits, 0, s->logits, (uint64_t)DS4_N_VOCAB * 4u) ||
        !ds4_gpu_synchronize()) return 0;
    ds41_graph_reset(g);
    g->sel_done = g->sel_want = g->kg_want = false;
    for (uint32_t i = 0; i < 3 && i < pos; ++i)
        g->history.tail[i] = (int32_t)g->token_map[tokens->v[pos - 1u - i]];
    g->pos = pos;
    const uint32_t first = pos > 128u ? pos - 128u : 0u;
    for (unsigned stage = 0; stage < 3; ++stage)
        for (uint32_t p = first; p < pos; ++p)
            g->dspark->positions[stage][p % 128u] = p;
    /* Rebuild checked derived views from authoritative F32 owners. Upstream
     * retains its F32 scorer when an imported key is not BF16-exact. */
    ds41_index_bf16_restore(g, pos);
    return 1;
}

static uint64_t ds41_native_payload_body_bytes(ds41_gpu_graph *g, uint32_t pos) {
    ds41_state_span target[54]; dsv41_c_span wire[55]; uint32_t count = 0;
    if (!ds41_native_geometry(g, pos, target, wire, &count)) return 0;
    uint64_t bytes = 8u + ((uint64_t)pos + DS4_N_VOCAB) * 4u;
    for (uint32_t i = 0; i < count; ++i) bytes += wire[i].bytes;
    return bytes;
}

/* Keep upstream mode2 snapshots consumable by unmodified upstream readers.
 * Its stored feature ring is BF16-valued F32, whereas mode3 needs the raw
 * F32 mean. Reuse the existing capture scratch under the session owner; do
 * not round the authoritative target ring or allocate another ring. */
static int ds41_native_write_mode2_ring(FILE *fp, ds41_dspark *d, unsigned stage,
                                        unsigned char *page, char *err, size_t cap) {
    const uint64_t bytes = 128u * 5120u * 4u;
    if (!d || stage >= 3 || !d->mean || !ds4_gpu_synchronize() ||
        !ds4_gpu_begin_commands()) {
        payload_set_err(err, cap, "failed to stage upstream mode2 feature ring"); return 1;
    }
    const int encoded = ds4_gpu_tensor_copy(d->mean, 0, d->target[stage], 0, bytes) &&
        ds4_gpu_dsv41_quantize(d->mean, 5120u, 128u, DS4_V41_BF16);
    const int completed = ds4_gpu_end_commands();
    if (!encoded || !completed) {
        payload_set_err(err, cap, "failed to encode upstream mode2 feature ring"); return 1;
    }
    return payload_write_tensor_span(fp, d->mean, 0, bytes, page, DS4_SESSION_IO_CHUNK, err, cap);
}

/* Explicit mode3 export for the native protocol. Ordinary upstream mode1/2
 * save selection stays in ds41_save_payload; this helper does not replace it. */
static int ds41_native_save_payload(ds4_session *s, FILE *fp, char *err, size_t cap) {
    ds41_gpu_graph *g = &s->ds41_graph; const uint32_t pos = g->pos;
    if (!s->ds41_graph_ready || !s->checkpoint_valid || !g->valid ||
        pos != (uint32_t)s->checkpoint.len || !ds41_native_positions(g, pos) ||
        !ds4_gpu_synchronize()) {
        payload_set_err(err, cap, "native mode3 export requires a complete captured frontier"); return 1;
    }
    const uint32_t header[15] = {DS4_SESSION_PAYLOAD_MAGIC, DS4_SESSION_PAYLOAD_VERSION,
        g->ctx, 3u, 128u, 128u, g->ctx + 1u, pos, 40u, 512u, 128u, DS4_N_VOCAB,
        0x413431u, pos, pos < 128u ? pos : 128u};
    for (unsigned i = 0; i < 15; ++i)
        if (payload_write_u32(fp, header[i], err, cap)) return 1;
    for (uint32_t i = 0; i < pos; ++i)
        if (payload_write_u32(fp, (uint32_t)s->checkpoint.v[i], err, cap)) return 1;
    if (payload_write_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * 4u, err, cap)) return 1;
    ds41_state_span target[54]; dsv41_c_span wire[55]; uint32_t count = 0;
    if (!ds41_native_geometry(g, pos, target, wire, &count)) return 1;
    unsigned char page[65536];
    for (uint32_t i = 0; i < count; ++i)
        for (uint64_t off = 0; off < wire[i].bytes;) {
            size_t n = (size_t)(wire[i].bytes - off); if (n > sizeof(page)) n = sizeof(page);
            if ((s->cancel && s->cancel(s->cancel_ud)) ||
                !ds41_native_owner_io(g, pos, i, off, page, n, false) ||
                payload_write_bytes(fp, page, n, err, cap)) return 1;
            off += n;
        }
    return 0;
}

static int ds41_native_load_payload(ds4_session *s, FILE *fp, const uint32_t *h,
                                    uint64_t remaining, char *err, size_t cap) {
    ds41_gpu_graph *g = &s->ds41_graph;
    const uint32_t pos = h[7];
    const uint64_t expected = ds41_native_payload_body_bytes(g, pos);
    if (!s->ds41_graph_ready || !expected || !pos || pos >= h[2] || h[2] > 1048576u ||
        h[3] != 3u || h[4] != 128u || h[5] != 128u || h[6] != h[2] + 1u ||
        h[8] != 40u || h[9] != 512u || h[10] != 128u || h[11] != DS4_N_VOCAB ||
        h[12] != 0x413431u || remaining != expected) {
        payload_set_err(err, cap, "invalid native mode3 snapshot dimensions or size"); return 1;
    }
    uint32_t end = 0, length = 0;
    if (payload_read_u32(fp, &end, &remaining, err, cap) ||
        payload_read_u32(fp, &length, &remaining, err, cap) || end != pos ||
        length != (pos < 128u ? pos : 128u)) {
        payload_set_err(err, cap, "invalid native mode3 captured frontier"); return 1;
    }
    ds4_tokens tokens = {0}; int rc = 1;
    for (uint32_t i = 0; i < pos; ++i) {
        uint32_t token;
        if (payload_read_u32(fp, &token, &remaining, err, cap) || token >= DS4_N_VOCAB) {
            payload_set_err(err, cap, "invalid native mode3 token"); goto done;
        }
        token_vec_push(&tokens, (int)token);
    }
    if (!ds4_gpu_synchronize()) goto done;
    s->checkpoint_valid = false; g->valid = false; s->sel_bound = false;
    s->mtp_draft_valid = false; ds4_session_dspark_capture_invalidate(s);
    if (payload_read_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * 4u, &remaining, err, cap)) goto done;
    /* -ffast-math may fold isfinite away. Validate the serialized exponent
     * bits, exactly as the existing native artifact reader does. */
    for (uint32_t i = 0; i < DS4_N_VOCAB; ++i) {
        uint32_t bits; memcpy(&bits, s->logits + i, sizeof(bits));
        if ((bits & 0x7f800000u) == 0x7f800000u) {
            payload_set_err(err, cap, "nonfinite native mode3 logits"); goto done;
        }
    }
    ds41_state_span target[54]; dsv41_c_span wire[55]; uint32_t count = 0;
    if (!ds41_native_geometry(g, pos, target, wire, &count)) goto done;
    unsigned char page[65536];
    for (uint32_t i = 0; i < count; ++i) {
        for (uint64_t off = 0; off < wire[i].bytes;) {
            size_t n = (size_t)(wire[i].bytes - off); if (n > sizeof(page)) n = sizeof(page);
            if ((s->cancel && s->cancel(s->cancel_ud)) ||
                payload_read_bytes(fp, page, n, &remaining, err, cap) ||
                !ds41_native_owner_io(g, pos, i, off, page, n, true)) goto done;
            off += n;
        }
    }
    if (remaining || !ds41_native_commit(s, &tokens)) goto done;
    ds4_tokens_free(&s->checkpoint); s->checkpoint = tokens; memset(&tokens, 0, sizeof(tokens));
    s->checkpoint_valid = true; rc = 0;
done:
    ds4_tokens_free(&tokens);
    if (rc && err && cap && !err[0]) payload_set_err(err, cap, "native mode3 import failed; checkpoint invalid");
    return rc;
}
#endif
#endif
