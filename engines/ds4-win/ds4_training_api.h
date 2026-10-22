/* Authenticated frozen-prefix adapter for the existing streamed LoRA recipe.
 * Input is the established exact little-endian token-ID representation.
 * Output is the native final normalized head input, F32 little-endian.
 * No model, adapter, output-path or process argument is accepted over HTTP. */
#ifndef DS4_TRAINING_API_H
#define DS4_TRAINING_API_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

static bool ds4_training_tokens_le32(const unsigned char *bytes, size_t size,
                                    int *tokens, uint32_t *count, uint32_t vocab) {
    if (!bytes || !tokens || !count || size == 0 || size % 4u || size > 256u * 4u || !vocab) return false;
    *count = (uint32_t)(size / 4u);
    for (uint32_t i = 0; i < *count; ++i) {
        const unsigned char *p = bytes + 4u * i;
        uint32_t value = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                         ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        if (value >= vocab) return false;
        tokens[i] = (int)value;
    }
    return true;
}

#ifndef DS4_TRAINING_PARSER_ONLY
static void ds4_training_send_tokens(server *s, int fd, const ds4_tokens *tokens) {
    if (!tokens || tokens->len < 1 || tokens->len > 256) {
        http_error(fd, s->enable_cors, 400, "training tokens must have length 1..256");return;
    }
    unsigned char data[1024];
    for (int i = 0; i < tokens->len; ++i) {
        const uint32_t t = (uint32_t)tokens->v[i];
        for (int b = 0; b < 4; ++b) data[4*i+b] = (unsigned char)(t >> (8*b));
    }
    buf header = {0};
    buf_printf(&header, "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", (size_t)tokens->len * 4);
    if (send_all(fd, header.ptr, header.len)) send_all(fd, data, (size_t)tokens->len * 4);
    buf_free(&header);
}

static void ds4_training_prompt(server *s, int fd, const http_request *hr) {
    request req; char error[256] = {0};
    if (hr->body_len > 16384 || !parse_chat_request(s->engine, s, hr->body,
            1, 256, &req, error, sizeof(error))) {
        http_error(fd, s->enable_cors, 400, error[0] ? error : "training prompt exceeds bound");return;
    }
    ds4_training_send_tokens(s, fd, &req.prompt);request_free(&req);
}

static void ds4_training_text(server *s, int fd, const http_request *hr) {
    if (!hr->body_len || hr->body_len > 16384 || strlen(hr->body) != hr->body_len) {
        http_error(fd, s->enable_cors, 400, "bounded UTF-8 answer text required");return;
    }
    ds4_tokens tokens = {0};ds4_tokenize_text(s->engine, hr->body, &tokens);
    ds4_training_send_tokens(s, fd, &tokens);ds4_tokens_free(&tokens);
}

static void ds4_training_features(server *s, int fd, const http_request *hr) {
    int tokens[256]; uint32_t count = 0;
    const int dim = ds4_engine_output_features_dim(s->engine);
    if (!dim || !ds4_training_tokens_le32((const unsigned char *)hr->body, hr->body_len,
                                         tokens, &count, (uint32_t)ds4_engine_vocab_size(s->engine))) {
        http_error(fd, s->enable_cors, 400, "original DSV4.1 and 1..256 exact LE32 token IDs required");
        return;
    }
    const size_t bytes = (size_t)count * (size_t)dim * sizeof(float);
    float *features = malloc(bytes);
    ds4_session *session = NULL;
    char error[256] = {0}; int rc = 1;
    if (!features) {http_error(fd, s->enable_cors, 500, "bounded feature buffer allocation failed");return;}
    pthread_mutex_lock(&s->inference_mu);
    if (!s->training_features || s->slot_count != 1 || s->batched_mode ||
        !s->slots || !(session = s->slots[0].session)) {
        snprintf(error, sizeof(error), "dedicated single-slot --training-features provider required");
    } else {
        ds4_tokens first = {0}; ds4_tokens_push(&first, tokens[0]);
        rc = ds4_session_sync(session, &first, error, sizeof(error));
        ds4_tokens_free(&first);
        for (uint32_t i = 0; !rc && i < count; ++i) {
            if (i) rc = ds4_session_eval(session, tokens[i], error, sizeof(error));
            if (!rc && ds4_session_copy_output_features(session, features + (size_t)i * dim, dim) != dim) {
                snprintf(error, sizeof(error), "native output feature readback failed");rc = 1;
            }
        }
    }
    pthread_mutex_unlock(&s->inference_mu);
    if (rc) {free(features);http_error(fd, s->enable_cors, 500, error);return;}
    buf header = {0};
    buf_printf(&header, "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: %zu\r\nX-Slstack-Hidden-Dim: %d\r\nX-Slstack-Token-Count: %u\r\nConnection: close\r\n\r\n", bytes, dim, count);
    if (send_all(fd, header.ptr, header.len)) send_all(fd, features, bytes);
    fprintf(stderr, "TRAINING_FEATURES native_frozen_prefix=1 tokens=%u hidden=%d bytes=%zu base_read_only=1\n", count, dim, bytes);
    fflush(stderr);buf_free(&header);free(features);
}
#endif
#endif
