#include "ds4_peer.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include "ds4_win32.h"
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

#define DS4_PEER_IO_TIMEOUT_SEC 120   /* see peer_io_timeout_sec(): 10 s killed a live generation */
#define DS4_PEER_MAX_FRAME_BODY (1024u * 1024u)
#define DS4_PEER_KV_CHUNK_BYTES (64u * 1024u)
#define DS4_PEER_MAX_PAYLOAD_BYTES (UINT64_C(16) * 1024u * 1024u * 1024u)
#define DS4_PEER_MAX_ERROR_TEXT 4096u

#define DS4_PEER_VALID_PAYLOAD_VERSIONS \
    (DS4_PEER_PAYLOAD_V2 | DS4_PEER_PAYLOAD_V3)
#define DS4_PEER_VALID_POLICY_FLAGS \
    (DS4_PEER_POLICY_F32_CANONICAL | DS4_PEER_POLICY_ALLOW_INEXACT_KV | \
     DS4_PEER_POLICY_ALLOW_WEIGHT_MISMATCH)

typedef struct {
    uint8_t *data;
    size_t len;
    size_t cap;
} ds4_peer_buffer;

typedef struct {
    const uint8_t *data;
    size_t len;
    size_t pos;
} ds4_peer_reader;

typedef struct {
    uint32_t type;
    uint8_t *body;
    size_t body_len;
} ds4_peer_frame;

struct ds4_peer_listener {
    int fd;
    int stopping;
    int active_fd;
    pthread_mutex_t mu;
    pthread_t thread;
    ds4_peer_capabilities local;
    ds4_peer_service service;
};

static bool peer_listener_is_stopping(ds4_peer_listener *listener) {
    if (!listener) return true;
    pthread_mutex_lock(&listener->mu);
    const bool stopping = listener->stopping != 0;
    pthread_mutex_unlock(&listener->mu);
    return stopping;
}

static void peer_set_error(char *err, size_t errlen, const char *fmt, ...) {
    if (!err || errlen == 0) return;
    va_list ap;
    va_start(ap, fmt);
    (void)vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
    err[errlen - 1] = '\0';
}

static void peer_log(const char *kind, const char *text) {
    fprintf(stderr, "ds4-peer: %s%s%s\n",
            kind && kind[0] ? kind : "",
            kind && kind[0] ? ": " : "",
            text ? text : "");
    fflush(stderr);
}

static size_t peer_strnlen(const char *s, size_t maxlen) {
    size_t n = 0;
    if (!s) return 0;
    while (n < maxlen && s[n]) n++;
    return n;
}

static const char *peer_format_name(uint32_t format) {
    switch (format) {
    case 0: return "fp32";
    case 1: return "fp16";
    case 2: return "q8";
    default: return "invalid";
    }
}

static const char *peer_role_name(uint32_t role) {
    switch (role) {
    case DS4_PEER_ROLE_PREFILL: return "prefill";
    case DS4_PEER_ROLE_DECODE: return "decode";
    default: return "none";
    }
}

static const char *peer_transfer_name(uint32_t mode) {
    return mode == DS4_PEER_TRANSFER_NATIVE ? "native" : "canonical";
}

void ds4_peer_capabilities_init(ds4_peer_capabilities *caps) {
    if (!caps) return;
    memset(caps, 0, sizeof(*caps));
    caps->protocol_version = DS4_PEER_PROTOCOL_VERSION;
}

int ds4_peer_parse_endpoint(const char *text, ds4_peer_endpoint *out,
                            char *err, size_t errlen) {
    if (!text || !out) {
        peer_set_error(err, errlen, "DS4P endpoint is missing");
        return 1;
    }
    const char *colon = strrchr(text, ':');
    if (!colon || colon == text || !colon[1]) {
        peer_set_error(err, errlen,
                       "invalid DS4P endpoint '%s'; expected host:port", text);
        return 1;
    }
    size_t host_len = (size_t)(colon - text);
    if (host_len >= sizeof(out->host)) {
        peer_set_error(err, errlen, "DS4P endpoint host is too long");
        return 1;
    }
    char *end = NULL;
    errno = 0;
    long port = strtol(colon + 1, &end, 10);
    if (errno != 0 || end == colon + 1 || *end != '\0' ||
        port <= 0 || port > 65535) {
        peer_set_error(err, errlen,
                       "invalid DS4P endpoint port in '%s'", text);
        return 1;
    }
    memset(out, 0, sizeof(*out));
    memcpy(out->host, text, host_len);
    out->port = (uint16_t)port;
    return 0;
}

static int peer_endpoint_address(const ds4_peer_endpoint *endpoint,
                                 struct sockaddr_in *out,
                                 char *err, size_t errlen) {
    if (!endpoint || !out) {
        peer_set_error(err, errlen, "DS4P endpoint is missing");
        return 1;
    }
    memset(out, 0, sizeof(*out));
    out->sin_family = AF_INET;
    out->sin_port = htons(endpoint->port);
    const char *host = endpoint->host;
    if (!strcmp(host, "localhost")) host = "127.0.0.1";
    if (inet_pton(AF_INET, host, &out->sin_addr) != 1) {
        peer_set_error(err, errlen,
                       "DS4P endpoint host '%s' must be an IPv4 literal or localhost",
                       endpoint->host);
        return 1;
    }
    return 0;
}

/* HOW LONG A PEER MAY GO QUIET BEFORE WE CALL THE LINK DEAD.
 *
 * The 10 s default killed a healthy generation. Measured 2026-08-24, RTX-prefill/Mac-decode: turn 1
 * decoded at ~40 t/s, slowed to 15 t/s while the RTX was streaming experts from SSD, and the Mac's
 * socket read hit EAGAIN -- "DS4P receive failed: Resource temporarily unavailable". The owner saw
 * that as `finish=error error="DS4P receive failed: connection reset"` at 441 tokens, and the session
 * could not even be retained afterwards ("turn-close could not retain session").
 *
 * A quiet socket is NOT a dead peer on this link: the prefilling side streams tens of GiB of experts
 * per request and can legitimately stall for longer than a fixed 10 s while it does. Treating that as
 * a reset loses the turn AND the session.
 *
 * Default raised to 120 s and made tunable, so a genuinely dead peer is still detected -- just not on
 * a threshold shorter than the work the other side is legitimately doing.
 */
static int peer_io_timeout_sec(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_PEER_IO_TIMEOUT_SEC");
        int v = env ? atoi(env) : 0;
        cached = v > 0 ? v : DS4_PEER_IO_TIMEOUT_SEC;
    }
    return cached;
}

static void peer_set_socket_timeout(int fd) {
    struct timeval tv;
    memset(&tv, 0, sizeof(tv));
    tv.tv_sec = peer_io_timeout_sec();
    (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

static int peer_open_listener(const ds4_peer_endpoint *endpoint,
                              char *err, size_t errlen) {
    struct sockaddr_in addr;
    if (peer_endpoint_address(endpoint, &addr, err, errlen) != 0) return -1;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        peer_set_error(err, errlen, "DS4P socket failed: %s", strerror(errno));
        return -1;
    }
    int yes = 1;
    (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        peer_set_error(err, errlen, "DS4P bind %s:%u failed: %s",
                       endpoint->host, (unsigned)endpoint->port, strerror(errno));
        close(fd);
        return -1;
    }
    if (listen(fd, 16) != 0) {
        peer_set_error(err, errlen, "DS4P listen %s:%u failed: %s",
                       endpoint->host, (unsigned)endpoint->port, strerror(errno));
        close(fd);
        return -1;
    }
    return fd;
}

static int peer_open_connection(const ds4_peer_endpoint *endpoint,
                                char *err, size_t errlen) {
    struct sockaddr_in addr;
    if (peer_endpoint_address(endpoint, &addr, err, errlen) != 0) return -1;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        peer_set_error(err, errlen, "DS4P socket failed: %s", strerror(errno));
        return -1;
    }
    peer_set_socket_timeout(fd);
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        peer_set_error(err, errlen, "DS4P connect %s:%u failed: %s",
                       endpoint->host, (unsigned)endpoint->port, strerror(errno));
        close(fd);
        return -1;
    }
    return fd;
}

static void peer_store_u32(uint8_t *dst, uint32_t value) {
    dst[0] = (uint8_t)value;
    dst[1] = (uint8_t)(value >> 8);
    dst[2] = (uint8_t)(value >> 16);
    dst[3] = (uint8_t)(value >> 24);
}

static uint32_t peer_load_u32(const uint8_t *src) {
    return (uint32_t)src[0] |
           ((uint32_t)src[1] << 8) |
           ((uint32_t)src[2] << 16) |
           ((uint32_t)src[3] << 24);
}

static void peer_store_u64(uint8_t *dst, uint64_t value) {
    for (uint32_t i = 0; i < 8; i++) dst[i] = (uint8_t)(value >> (8u * i));
}

static uint64_t peer_load_u64(const uint8_t *src) {
    uint64_t value = 0;
    for (uint32_t i = 0; i < 8; i++) value |= (uint64_t)src[i] << (8u * i);
    return value;
}

typedef struct {
    uint32_t state[8];
    uint64_t bytes;
    uint8_t block[64];
    size_t used;
} peer_sha256;

static const uint32_t peer_sha256_round_constants[64] = {
    UINT32_C(0x428a2f98), UINT32_C(0x71374491), UINT32_C(0xb5c0fbcf), UINT32_C(0xe9b5dba5),
    UINT32_C(0x3956c25b), UINT32_C(0x59f111f1), UINT32_C(0x923f82a4), UINT32_C(0xab1c5ed5),
    UINT32_C(0xd807aa98), UINT32_C(0x12835b01), UINT32_C(0x243185be), UINT32_C(0x550c7dc3),
    UINT32_C(0x72be5d74), UINT32_C(0x80deb1fe), UINT32_C(0x9bdc06a7), UINT32_C(0xc19bf174),
    UINT32_C(0xe49b69c1), UINT32_C(0xefbe4786), UINT32_C(0x0fc19dc6), UINT32_C(0x240ca1cc),
    UINT32_C(0x2de92c6f), UINT32_C(0x4a7484aa), UINT32_C(0x5cb0a9dc), UINT32_C(0x76f988da),
    UINT32_C(0x983e5152), UINT32_C(0xa831c66d), UINT32_C(0xb00327c8), UINT32_C(0xbf597fc7),
    UINT32_C(0xc6e00bf3), UINT32_C(0xd5a79147), UINT32_C(0x06ca6351), UINT32_C(0x14292967),
    UINT32_C(0x27b70a85), UINT32_C(0x2e1b2138), UINT32_C(0x4d2c6dfc), UINT32_C(0x53380d13),
    UINT32_C(0x650a7354), UINT32_C(0x766a0abb), UINT32_C(0x81c2c92e), UINT32_C(0x92722c85),
    UINT32_C(0xa2bfe8a1), UINT32_C(0xa81a664b), UINT32_C(0xc24b8b70), UINT32_C(0xc76c51a3),
    UINT32_C(0xd192e819), UINT32_C(0xd6990624), UINT32_C(0xf40e3585), UINT32_C(0x106aa070),
    UINT32_C(0x19a4c116), UINT32_C(0x1e376c08), UINT32_C(0x2748774c), UINT32_C(0x34b0bcb5),
    UINT32_C(0x391c0cb3), UINT32_C(0x4ed8aa4a), UINT32_C(0x5b9cca4f), UINT32_C(0x682e6ff3),
    UINT32_C(0x748f82ee), UINT32_C(0x78a5636f), UINT32_C(0x84c87814), UINT32_C(0x8cc70208),
    UINT32_C(0x90befffa), UINT32_C(0xa4506ceb), UINT32_C(0xbef9a3f7), UINT32_C(0xc67178f2),
};

static uint32_t peer_sha256_rotr(uint32_t value, uint32_t count) {
    return (value >> count) | (value << (32u - count));
}

static void peer_sha256_block(peer_sha256 *ctx, const uint8_t block[64]) {
    uint32_t words[64];
    for (uint32_t i = 0; i < 16; i++) {
        const uint8_t *p = block + (size_t)i * 4u;
        words[i] = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
                   ((uint32_t)p[2] << 8) | (uint32_t)p[3];
    }
    for (uint32_t i = 16; i < 64; i++) {
        const uint32_t s0 = peer_sha256_rotr(words[i - 15], 7) ^
                            peer_sha256_rotr(words[i - 15], 18) ^
                            (words[i - 15] >> 3);
        const uint32_t s1 = peer_sha256_rotr(words[i - 2], 17) ^
                            peer_sha256_rotr(words[i - 2], 19) ^
                            (words[i - 2] >> 10);
        words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }

    uint32_t a = ctx->state[0];
    uint32_t b = ctx->state[1];
    uint32_t c = ctx->state[2];
    uint32_t d = ctx->state[3];
    uint32_t e = ctx->state[4];
    uint32_t f = ctx->state[5];
    uint32_t g = ctx->state[6];
    uint32_t h = ctx->state[7];
    for (uint32_t i = 0; i < 64; i++) {
        const uint32_t s1 = peer_sha256_rotr(e, 6) ^ peer_sha256_rotr(e, 11) ^
                            peer_sha256_rotr(e, 25);
        const uint32_t choose = (e & f) ^ ((~e) & g);
        const uint32_t temp1 = h + s1 + choose + peer_sha256_round_constants[i] + words[i];
        const uint32_t s0 = peer_sha256_rotr(a, 2) ^ peer_sha256_rotr(a, 13) ^
                            peer_sha256_rotr(a, 22);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = s0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

static void peer_sha256_init(peer_sha256 *ctx) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->state[0] = UINT32_C(0x6a09e667);
    ctx->state[1] = UINT32_C(0xbb67ae85);
    ctx->state[2] = UINT32_C(0x3c6ef372);
    ctx->state[3] = UINT32_C(0xa54ff53a);
    ctx->state[4] = UINT32_C(0x510e527f);
    ctx->state[5] = UINT32_C(0x9b05688c);
    ctx->state[6] = UINT32_C(0x1f83d9ab);
    ctx->state[7] = UINT32_C(0x5be0cd19);
}

static void peer_sha256_update(peer_sha256 *ctx, const void *data, size_t len) {
    const uint8_t *p = data;
    ctx->bytes += (uint64_t)len;
    while (len != 0) {
        const size_t room = sizeof(ctx->block) - ctx->used;
        const size_t take = len < room ? len : room;
        memcpy(ctx->block + ctx->used, p, take);
        ctx->used += take;
        p += take;
        len -= take;
        if (ctx->used == sizeof(ctx->block)) {
            peer_sha256_block(ctx, ctx->block);
            ctx->used = 0;
        }
    }
}

static void peer_sha256_final(peer_sha256 *ctx,
                              uint8_t out[DS4_PEER_SHA256_BYTES]) {
    const uint64_t bits = ctx->bytes << 3;
    ctx->block[ctx->used++] = UINT8_C(0x80);
    if (ctx->used > 56u) {
        memset(ctx->block + ctx->used, 0, sizeof(ctx->block) - ctx->used);
        peer_sha256_block(ctx, ctx->block);
        ctx->used = 0;
    }
    memset(ctx->block + ctx->used, 0, 56u - ctx->used);
    for (uint32_t i = 0; i < 8; i++) {
        ctx->block[63u - i] = (uint8_t)(bits >> (8u * i));
    }
    peer_sha256_block(ctx, ctx->block);
    for (uint32_t i = 0; i < 8; i++) {
        out[(size_t)i * 4u] = (uint8_t)(ctx->state[i] >> 24);
        out[(size_t)i * 4u + 1u] = (uint8_t)(ctx->state[i] >> 16);
        out[(size_t)i * 4u + 2u] = (uint8_t)(ctx->state[i] >> 8);
        out[(size_t)i * 4u + 3u] = (uint8_t)ctx->state[i];
    }
}

void ds4_peer_sha256(const void *data, size_t len,
                     uint8_t out[DS4_PEER_SHA256_BYTES]) {
    peer_sha256 ctx;
    peer_sha256_init(&ctx);
    if (len != 0 && data) peer_sha256_update(&ctx, data, len);
    peer_sha256_final(&ctx, out);
}

static int peer_write_all(int fd, const void *data, size_t len,
                          char *err, size_t errlen) {
    const uint8_t *p = data;
    size_t pos = 0;
    while (pos < len) {
        size_t remaining = len - pos;
        if (remaining > (size_t)INT_MAX) remaining = (size_t)INT_MAX;
        ssize_t n = send(fd, p + pos, remaining, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            peer_set_error(err, errlen, "DS4P send failed: %s", strerror(errno));
            return 1;
        }
        pos += (size_t)n;
    }
    return 0;
}

/* Returns 1 for a complete read, 0 for a clean EOF before any bytes, and -1
 * for an incomplete or failed read. */
static int peer_read_all(int fd, void *data, size_t len, bool eof_ok,
                         char *err, size_t errlen) {
    uint8_t *p = data;
    size_t pos = 0;
    while (pos < len) {
        size_t remaining = len - pos;
        if (remaining > (size_t)INT_MAX) remaining = (size_t)INT_MAX;
        ssize_t n = recv(fd, p + pos, remaining, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n == 0) {
            if (pos == 0 && eof_ok) return 0;
            peer_set_error(err, errlen, "DS4P peer closed during a frame");
            return -1;
        }
        if (n < 0) {
            peer_set_error(err, errlen, "DS4P receive failed: %s", strerror(errno));
            return -1;
        }
        pos += (size_t)n;
    }
    return 1;
}

static int peer_send_frame(int fd, uint32_t type, const void *body,
                           size_t body_len, char *err, size_t errlen) {
    if (body_len > DS4_PEER_MAX_FRAME_BODY) {
        peer_set_error(err, errlen, "DS4P frame body is too large");
        return 1;
    }
    uint8_t header[16];
    peer_store_u32(header, DS4_PEER_MAGIC);
    peer_store_u32(header + 4, type);
    peer_store_u64(header + 8, (uint64_t)body_len);
    if (peer_write_all(fd, header, sizeof(header), err, errlen) != 0) return 1;
    if (body_len && peer_write_all(fd, body, body_len, err, errlen) != 0) return 1;
    return 0;
}

static void peer_frame_free(ds4_peer_frame *frame) {
    if (!frame) return;
    free(frame->body);
    memset(frame, 0, sizeof(*frame));
}

/* Returns 1 for a frame, 0 for a clean connection close, and -1 for error. */
static int peer_recv_frame(int fd, ds4_peer_frame *frame,
                           char *err, size_t errlen) {
    uint8_t header[16];
    memset(frame, 0, sizeof(*frame));
    int rc = peer_read_all(fd, header, sizeof(header), true, err, errlen);
    if (rc <= 0) return rc;
    if (peer_load_u32(header) != DS4_PEER_MAGIC) {
        peer_set_error(err, errlen, "DS4P frame magic mismatch");
        return -1;
    }
    uint64_t body_len = peer_load_u64(header + 8);
    if (body_len > DS4_PEER_MAX_FRAME_BODY || body_len > SIZE_MAX) {
        peer_set_error(err, errlen, "DS4P frame body length is invalid");
        return -1;
    }
    frame->type = peer_load_u32(header + 4);
    frame->body_len = (size_t)body_len;
    if (!frame->body_len) return 1;
    frame->body = malloc(frame->body_len);
    if (!frame->body) {
        peer_set_error(err, errlen, "out of memory receiving DS4P frame");
        return -1;
    }
    rc = peer_read_all(fd, frame->body, frame->body_len, false, err, errlen);
    if (rc != 1) {
        peer_frame_free(frame);
        return -1;
    }
    return 1;
}

static int peer_send_error(int fd, const char *text, char *err, size_t errlen) {
    size_t len = text ? strlen(text) : 0;
    if (len == 0 || len > DS4_PEER_MAX_ERROR_TEXT) {
        peer_set_error(err, errlen, "DS4P error text is invalid");
        return 1;
    }
    int rc = peer_send_frame(fd, DS4_PEER_FRAME_ERROR, text, len, err, errlen);
    if (rc == 0) peer_log("sent ERROR", text);
    return rc;
}

static int peer_read_error_frame(const ds4_peer_frame *frame,
                                 char *err, size_t errlen) {
    if (!frame || frame->body_len == 0 || frame->body_len > DS4_PEER_MAX_ERROR_TEXT) {
        peer_set_error(err, errlen, "DS4P ERROR frame has invalid text");
        return 1;
    }
    char text[DS4_PEER_MAX_ERROR_TEXT + 1u];
    memcpy(text, frame->body, frame->body_len);
    text[frame->body_len] = '\0';
    peer_set_error(err, errlen, "%s", text);
    peer_log("received ERROR", text);
    return 1;
}

static bool peer_buffer_reserve(ds4_peer_buffer *buf, size_t add) {
    if (add > SIZE_MAX - buf->len) return false;
    size_t need = buf->len + add;
    if (need <= buf->cap) return true;
    size_t cap = buf->cap ? buf->cap : 128u;
    while (cap < need) {
        if (cap > SIZE_MAX / 2u) {
            cap = need;
            break;
        }
        cap *= 2u;
    }
    uint8_t *next = realloc(buf->data, cap);
    if (!next) return false;
    buf->data = next;
    buf->cap = cap;
    return true;
}

static bool peer_buffer_bytes(ds4_peer_buffer *buf, const void *data, size_t len) {
    if (!peer_buffer_reserve(buf, len)) return false;
    if (len) memcpy(buf->data + buf->len, data, len);
    buf->len += len;
    return true;
}

static bool peer_buffer_u32(ds4_peer_buffer *buf, uint32_t value) {
    uint8_t bytes[4];
    peer_store_u32(bytes, value);
    return peer_buffer_bytes(buf, bytes, sizeof(bytes));
}

static bool peer_buffer_u64(ds4_peer_buffer *buf, uint64_t value) {
    uint8_t bytes[8];
    peer_store_u64(bytes, value);
    return peer_buffer_bytes(buf, bytes, sizeof(bytes));
}

static bool peer_buffer_f32(ds4_peer_buffer *buf, float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return peer_buffer_u32(buf, bits);
}

static void peer_buffer_free(ds4_peer_buffer *buf) {
    if (!buf) return;
    free(buf->data);
    memset(buf, 0, sizeof(*buf));
}

static bool peer_reader_bytes(ds4_peer_reader *reader, void *out, size_t len) {
    if (reader->pos > reader->len || len > reader->len - reader->pos) return false;
    if (len && out) memcpy(out, reader->data + reader->pos, len);
    reader->pos += len;
    return true;
}

static bool peer_reader_u32(ds4_peer_reader *reader, uint32_t *out) {
    uint8_t bytes[4];
    if (!peer_reader_bytes(reader, bytes, sizeof(bytes))) return false;
    *out = peer_load_u32(bytes);
    return true;
}

static bool peer_reader_u64(ds4_peer_reader *reader, uint64_t *out) {
    uint8_t bytes[8];
    if (!peer_reader_bytes(reader, bytes, sizeof(bytes))) return false;
    *out = peer_load_u64(bytes);
    return true;
}

static bool peer_reader_f32(ds4_peer_reader *reader, float *out) {
    uint32_t bits = 0;
    if (!peer_reader_u32(reader, &bits)) return false;
    memcpy(out, &bits, sizeof(bits));
    return true;
}

static int peer_capabilities_sane(const ds4_peer_capabilities *caps,
                                  char *err, size_t errlen) {
    if (!caps) {
        peer_set_error(err, errlen, "DS4P capabilities are missing");
        return 1;
    }
    if (caps->role > DS4_PEER_ROLE_DECODE) {
        peer_set_error(err, errlen, "DS4P peer role is invalid");
        return 1;
    }
    if (caps->policy_flags & ~DS4_PEER_VALID_POLICY_FLAGS) {
        peer_set_error(err, errlen, "DS4P capability policy flags are invalid");
        return 1;
    }
    if (caps->n_layer == 0 || caps->n_layer > DS4_PEER_MAX_LAYERS) {
        peer_set_error(err, errlen, "DS4P layer count is invalid");
        return 1;
    }
    if (caps->n_embd == 0 || caps->n_vocab == 0 || caps->n_head_dim == 0 ||
        caps->n_indexer_head_dim == 0) {
        peer_set_error(err, errlen, "DS4P model shape is invalid");
        return 1;
    }
    if (peer_strnlen(caps->model_name, sizeof(caps->model_name)) ==
        sizeof(caps->model_name)) {
        peer_set_error(err, errlen, "DS4P model name is not terminated");
        return 1;
    }
    if (caps->ctx_size == 0 || caps->raw_window == 0 ||
        caps->raw_cap < caps->raw_window || caps->comp_cap == 0) {
        peer_set_error(err, errlen, "DS4P runtime KV layout is invalid");
        return 1;
    }
    if (caps->attn_comp_cache_format > 2u) {
        peer_set_error(err, errlen, "DS4P compressed-cache format is invalid");
        return 1;
    }
    if ((caps->supported_payload_versions & ~DS4_PEER_VALID_PAYLOAD_VERSIONS) ||
        !(caps->supported_payload_versions & DS4_PEER_VALID_PAYLOAD_VERSIONS)) {
        peer_set_error(err, errlen, "DS4P supported payload versions are invalid");
        return 1;
    }
    if (caps->weights_fingerprint == 0) {
        peer_set_error(err, errlen, "DS4P weights fingerprint is missing");
        return 1;
    }
    return 0;
}

/* HELLO capability body, encoded in this exact order:
 * protocol, role/policy, model family/variant/shape, ratio vector, routed
 * quantization, runtime KV layout/format/backend/payload support, tensor
 * directory fingerprint, and model name. */
static int peer_encode_capabilities(const ds4_peer_capabilities *caps,
                                    ds4_peer_buffer *buf,
                                    char *err, size_t errlen) {
    if (peer_capabilities_sane(caps, err, errlen) != 0) return 1;
    size_t name_len = peer_strnlen(caps->model_name, sizeof(caps->model_name));
    bool ok =
        peer_buffer_u32(buf, caps->protocol_version) &&
        peer_buffer_u32(buf, caps->role) &&
        peer_buffer_u32(buf, caps->policy_flags) &&
        peer_buffer_u32(buf, caps->model_family) &&
        peer_buffer_u32(buf, caps->model_variant) &&
        peer_buffer_u32(buf, caps->n_layer) &&
        peer_buffer_u32(buf, caps->n_embd) &&
        peer_buffer_u32(buf, caps->n_vocab) &&
        peer_buffer_u32(buf, caps->n_head_dim) &&
        peer_buffer_u32(buf, caps->n_indexer_head_dim) &&
        peer_buffer_u32(buf, caps->n_swa) &&
        peer_buffer_u32(buf, caps->n_layer);
    for (uint32_t i = 0; ok && i < caps->n_layer; i++) {
        ok = peer_buffer_u32(buf, caps->compress_ratios[i]);
    }
    ok = ok &&
        peer_buffer_u32(buf, caps->routed_quant_bits) &&
        peer_buffer_u32(buf, caps->ctx_size) &&
        peer_buffer_u32(buf, caps->raw_window) &&
        peer_buffer_u32(buf, caps->raw_cap) &&
        peer_buffer_u32(buf, caps->comp_cap) &&
        peer_buffer_u32(buf, caps->attn_comp_cache_format) &&
        peer_buffer_u32(buf, caps->backend) &&
        peer_buffer_u32(buf, caps->supported_payload_versions) &&
        peer_buffer_u64(buf, caps->weights_fingerprint) &&
        peer_buffer_u32(buf, (uint32_t)name_len) &&
        peer_buffer_bytes(buf, caps->model_name, name_len);
    if (!ok) {
        peer_set_error(err, errlen, "out of memory encoding DS4P capabilities");
        return 1;
    }
    return 0;
}

static int peer_decode_capabilities(ds4_peer_reader *reader,
                                    ds4_peer_capabilities *caps,
                                    char *err, size_t errlen) {
    ds4_peer_capabilities_init(caps);
    uint32_t ratio_count = 0;
    uint32_t model_name_len = 0;
    bool ok =
        peer_reader_u32(reader, &caps->protocol_version) &&
        peer_reader_u32(reader, &caps->role) &&
        peer_reader_u32(reader, &caps->policy_flags) &&
        peer_reader_u32(reader, &caps->model_family) &&
        peer_reader_u32(reader, &caps->model_variant) &&
        peer_reader_u32(reader, &caps->n_layer) &&
        peer_reader_u32(reader, &caps->n_embd) &&
        peer_reader_u32(reader, &caps->n_vocab) &&
        peer_reader_u32(reader, &caps->n_head_dim) &&
        peer_reader_u32(reader, &caps->n_indexer_head_dim) &&
        peer_reader_u32(reader, &caps->n_swa) &&
        peer_reader_u32(reader, &ratio_count);
    if (!ok) {
        peer_set_error(err, errlen, "truncated DS4P capabilities");
        return 1;
    }
    if (ratio_count != caps->n_layer || ratio_count > DS4_PEER_MAX_LAYERS) {
        peer_set_error(err, errlen,
                       "DS4P compression-ratio vector length is invalid");
        return 1;
    }
    for (uint32_t i = 0; i < ratio_count; i++) {
        if (!peer_reader_u32(reader, &caps->compress_ratios[i])) {
            peer_set_error(err, errlen, "truncated DS4P compression-ratio vector");
            return 1;
        }
    }
    ok =
        peer_reader_u32(reader, &caps->routed_quant_bits) &&
        peer_reader_u32(reader, &caps->ctx_size) &&
        peer_reader_u32(reader, &caps->raw_window) &&
        peer_reader_u32(reader, &caps->raw_cap) &&
        peer_reader_u32(reader, &caps->comp_cap) &&
        peer_reader_u32(reader, &caps->attn_comp_cache_format) &&
        peer_reader_u32(reader, &caps->backend) &&
        peer_reader_u32(reader, &caps->supported_payload_versions) &&
        peer_reader_u64(reader, &caps->weights_fingerprint) &&
        peer_reader_u32(reader, &model_name_len);
    if (!ok || model_name_len >= sizeof(caps->model_name) ||
        !peer_reader_bytes(reader, caps->model_name, model_name_len)) {
        peer_set_error(err, errlen, "truncated or invalid DS4P model name");
        return 1;
    }
    caps->model_name[model_name_len] = '\0';
    return peer_capabilities_sane(caps, err, errlen);
}

static int peer_decode_hello(const ds4_peer_frame *frame,
                             ds4_peer_capabilities *caps,
                             char *err, size_t errlen) {
    if (frame->type != DS4_PEER_FRAME_HELLO) {
        peer_set_error(err, errlen, "expected DS4P HELLO, received frame type %u",
                       frame->type);
        return 1;
    }
    ds4_peer_reader reader = {.data = frame->body, .len = frame->body_len};
    if (peer_decode_capabilities(&reader, caps, err, errlen) != 0) return 1;
    if (reader.pos != reader.len) {
        peer_set_error(err, errlen, "DS4P HELLO has trailing bytes");
        return 1;
    }
    return 0;
}

static int peer_encode_ack(const ds4_peer_capabilities *caps,
                           uint32_t payload_version, uint32_t transfer_mode,
                           ds4_peer_buffer *buf,
                           char *err, size_t errlen) {
    if (peer_encode_capabilities(caps, buf, err, errlen) != 0) return 1;
    if (!peer_buffer_u32(buf, payload_version) ||
        !peer_buffer_u32(buf, transfer_mode)) {
        peer_set_error(err, errlen, "out of memory encoding DS4P HELLO_ACK");
        return 1;
    }
    return 0;
}

static int peer_decode_ack(const ds4_peer_frame *frame,
                           ds4_peer_handshake_result *ack,
                           char *err, size_t errlen) {
    if (frame->type != DS4_PEER_FRAME_HELLO_ACK) {
        peer_set_error(err, errlen,
                       "expected DS4P HELLO_ACK, received frame type %u", frame->type);
        return 1;
    }
    memset(ack, 0, sizeof(*ack));
    ds4_peer_reader reader = {.data = frame->body, .len = frame->body_len};
    if (peer_decode_capabilities(&reader, &ack->peer, err, errlen) != 0 ||
        !peer_reader_u32(&reader, &ack->payload_version) ||
        !peer_reader_u32(&reader, &ack->transfer_mode) ||
        reader.pos != reader.len) {
        if (!err || !err[0]) {
            peer_set_error(err, errlen, "truncated DS4P HELLO_ACK");
        }
        return 1;
    }
    if ((ack->payload_version != 0 &&
         ack->payload_version != DS4_PEER_PAYLOAD_VERSION_V2 &&
         ack->payload_version != DS4_PEER_PAYLOAD_VERSION_V3) ||
        ack->transfer_mode > DS4_PEER_TRANSFER_NATIVE) {
        peer_set_error(err, errlen, "DS4P HELLO_ACK selected an invalid transfer mode");
        return 1;
    }
    return 0;
}

static bool peer_capabilities_equal(const ds4_peer_capabilities *a,
                                    const ds4_peer_capabilities *b) {
    if (a->protocol_version != b->protocol_version ||
        a->role != b->role ||
        a->policy_flags != b->policy_flags ||
        a->model_family != b->model_family ||
        a->model_variant != b->model_variant ||
        a->n_layer != b->n_layer ||
        a->n_embd != b->n_embd ||
        a->n_vocab != b->n_vocab ||
        a->n_head_dim != b->n_head_dim ||
        a->n_indexer_head_dim != b->n_indexer_head_dim ||
        a->n_swa != b->n_swa ||
        a->routed_quant_bits != b->routed_quant_bits ||
        a->ctx_size != b->ctx_size ||
        a->raw_window != b->raw_window ||
        a->raw_cap != b->raw_cap ||
        a->comp_cap != b->comp_cap ||
        a->attn_comp_cache_format != b->attn_comp_cache_format ||
        a->backend != b->backend ||
        a->supported_payload_versions != b->supported_payload_versions ||
        a->weights_fingerprint != b->weights_fingerprint ||
        strcmp(a->model_name, b->model_name) != 0) {
        return false;
    }
    return memcmp(a->compress_ratios, b->compress_ratios,
                  (size_t)a->n_layer * sizeof(a->compress_ratios[0])) == 0;
}

static bool peer_policy_both(const ds4_peer_capabilities *a,
                             const ds4_peer_capabilities *b,
                             uint32_t bit) {
    return (a->policy_flags & bit) != 0 && (b->policy_flags & bit) != 0;
}

static int peer_effective_role(const ds4_peer_capabilities *local,
                               const ds4_peer_capabilities *remote,
                               uint32_t *out_role,
                               char *err, size_t errlen) {
    if (local->role != DS4_PEER_ROLE_NONE && remote->role != DS4_PEER_ROLE_NONE &&
        local->role != remote->role) {
        peer_set_error(err, errlen,
                       "peer role conflict: local=%s peer=%s",
                       peer_role_name(local->role), peer_role_name(remote->role));
        return 1;
    }
    uint32_t role = local->role != DS4_PEER_ROLE_NONE ? local->role : remote->role;
    if (role == DS4_PEER_ROLE_NONE) {
        peer_set_error(err, errlen,
                       "peer role is missing; owner must specify --peer-role prefill|decode");
        return 1;
    }
    *out_role = role;
    return 0;
}

static int peer_validate_layout(const ds4_peer_capabilities *local,
                                const ds4_peer_capabilities *remote,
                                uint32_t role,
                                char *err, size_t errlen) {
    const ds4_peer_capabilities *owner =
        local->role != DS4_PEER_ROLE_NONE ? local : remote;
    const ds4_peer_capabilities *peer = owner == local ? remote : local;
    const ds4_peer_capabilities *sender =
        role == DS4_PEER_ROLE_PREFILL ? peer : owner;
    const ds4_peer_capabilities *receiver =
        role == DS4_PEER_ROLE_PREFILL ? owner : peer;
    const char *direction = peer_role_name(role);

    if (sender->raw_window != receiver->raw_window) {
        peer_set_error(err, errlen,
                       "raw-window mismatch for %s transfer: sender=%u receiver=%u",
                       direction, sender->raw_window, receiver->raw_window);
        return 1;
    }
    if (receiver->ctx_size < sender->ctx_size) {
        peer_set_error(err, errlen,
                       "context capacity mismatch for %s transfer: receiver ctx_size=%u is smaller than sender ctx_size=%u",
                       direction, receiver->ctx_size, sender->ctx_size);
        return 1;
    }
    if (receiver->raw_cap < sender->raw_window) {
        peer_set_error(err, errlen,
                       "raw-capacity mismatch for %s transfer: receiver raw_cap=%u is smaller than required raw_window=%u",
                       direction, receiver->raw_cap, sender->raw_window);
        return 1;
    }
    if (receiver->comp_cap < sender->comp_cap) {
        peer_set_error(err, errlen,
                       "compressed-cache capacity mismatch for %s transfer: receiver comp_cap=%u is smaller than sender comp_cap=%u",
                       direction, receiver->comp_cap, sender->comp_cap);
        return 1;
    }
    return 0;
}

static int peer_negotiate(const ds4_peer_capabilities *local,
                          const ds4_peer_capabilities *remote,
                          uint32_t *out_payload_version,
                          uint32_t *out_transfer_mode,
                          char *err, size_t errlen) {
    if (local->protocol_version != remote->protocol_version) {
        peer_set_error(err, errlen,
                       "protocol version mismatch: local=%u peer=%u",
                       local->protocol_version, remote->protocol_version);
        return 1;
    }
    if (local->model_family != remote->model_family) {
        peer_set_error(err, errlen,
                       "model family mismatch: local=%u peer=%u",
                       local->model_family, remote->model_family);
        return 1;
    }
    if (local->model_variant != remote->model_variant) {
        peer_set_error(err, errlen,
                       "model variant mismatch: local=%u peer=%u",
                       local->model_variant, remote->model_variant);
        return 1;
    }
    if (local->n_layer != remote->n_layer) {
        peer_set_error(err, errlen,
                       "layer count mismatch: local=%u peer=%u",
                       local->n_layer, remote->n_layer);
        return 1;
    }
    if (local->n_embd != remote->n_embd) {
        peer_set_error(err, errlen,
                       "embedding dimension mismatch: local=%u peer=%u",
                       local->n_embd, remote->n_embd);
        return 1;
    }
    if (local->n_vocab != remote->n_vocab) {
        peer_set_error(err, errlen,
                       "vocabulary size mismatch: local=%u peer=%u",
                       local->n_vocab, remote->n_vocab);
        return 1;
    }
    if (local->n_head_dim != remote->n_head_dim) {
        peer_set_error(err, errlen,
                       "attention head dimension mismatch: local=%u peer=%u",
                       local->n_head_dim, remote->n_head_dim);
        return 1;
    }
    if (local->n_indexer_head_dim != remote->n_indexer_head_dim) {
        peer_set_error(err, errlen,
                       "indexer head dimension mismatch: local=%u peer=%u",
                       local->n_indexer_head_dim, remote->n_indexer_head_dim);
        return 1;
    }
    if (local->n_swa != remote->n_swa) {
        peer_set_error(err, errlen,
                       "sliding-window size mismatch: local=%u peer=%u",
                       local->n_swa, remote->n_swa);
        return 1;
    }
    for (uint32_t i = 0; i < local->n_layer; i++) {
        if (local->compress_ratios[i] != remote->compress_ratios[i]) {
            peer_set_error(err, errlen,
                           "compress-ratio vector mismatch at layer %u: local=%u peer=%u",
                           i, local->compress_ratios[i], remote->compress_ratios[i]);
            return 1;
        }
    }
    if (local->routed_quant_bits != remote->routed_quant_bits) {
        peer_set_error(err, errlen,
                       "routed quantization mismatch: local=%u peer=%u",
                       local->routed_quant_bits, remote->routed_quant_bits);
        return 1;
    }
    if (local->weights_fingerprint != remote->weights_fingerprint &&
        !peer_policy_both(local, remote, DS4_PEER_POLICY_ALLOW_WEIGHT_MISMATCH)) {
        peer_set_error(err, errlen,
                       "weights fingerprint mismatch: local=0x%016" PRIx64
                       " peer=0x%016" PRIx64,
                       local->weights_fingerprint, remote->weights_fingerprint);
        return 1;
    }

    uint32_t role = DS4_PEER_ROLE_NONE;
    if (peer_effective_role(local, remote, &role, err, errlen) != 0 ||
        peer_validate_layout(local, remote, role, err, errlen) != 0) {
        return 1;
    }

    uint32_t common_payload_versions =
        local->supported_payload_versions & remote->supported_payload_versions;
    bool same_format =
        local->attn_comp_cache_format == remote->attn_comp_cache_format;
    if (same_format && (common_payload_versions & DS4_PEER_PAYLOAD_V3)) {
        *out_payload_version = DS4_PEER_PAYLOAD_VERSION_V3;
        *out_transfer_mode = DS4_PEER_TRANSFER_NATIVE;
        return 0;
    }
    if (!(common_payload_versions & DS4_PEER_PAYLOAD_V2)) {
        peer_set_error(err, errlen,
                       "no common supported payload version for DS4P transfer");
        return 1;
    }
    if (!peer_policy_both(local, remote, DS4_PEER_POLICY_F32_CANONICAL)) {
        if (!same_format) {
            peer_set_error(err, errlen,
                           "compressed-cache format mismatch: local=%s peer=%s; enable --peer-kv-f32-canonical on both peers",
                           peer_format_name(local->attn_comp_cache_format),
                           peer_format_name(remote->attn_comp_cache_format));
        } else {
            peer_set_error(err, errlen,
                           "native payload v3 is unavailable on one peer; enable --peer-kv-f32-canonical on both peers");
        }
        return 1;
    }
    if ((local->attn_comp_cache_format == 2u ||
         remote->attn_comp_cache_format == 2u) &&
        !peer_policy_both(local, remote, DS4_PEER_POLICY_ALLOW_INEXACT_KV)) {
        peer_set_error(err, errlen,
                       "q8 canonical fallback requires --allow-inexact-kv on both peers");
        return 1;
    }
    *out_payload_version = DS4_PEER_PAYLOAD_VERSION_V2;
    *out_transfer_mode = DS4_PEER_TRANSFER_CANONICAL;
    return 0;
}

static int peer_send_hello(int fd, const ds4_peer_capabilities *caps,
                           char *err, size_t errlen) {
    ds4_peer_buffer body = {0};
    int rc = peer_encode_capabilities(caps, &body, err, errlen);
    if (rc == 0) rc = peer_send_frame(fd, DS4_PEER_FRAME_HELLO,
                                     body.data, body.len, err, errlen);
    peer_buffer_free(&body);
    return rc;
}

static int peer_send_ack(int fd, const ds4_peer_capabilities *caps,
                         uint32_t payload_version, uint32_t transfer_mode,
                         char *err, size_t errlen) {
    ds4_peer_buffer body = {0};
    int rc = peer_encode_ack(caps, payload_version, transfer_mode,
                             &body, err, errlen);
    if (rc == 0) rc = peer_send_frame(fd, DS4_PEER_FRAME_HELLO_ACK,
                                     body.data, body.len, err, errlen);
    peer_buffer_free(&body);
    return rc;
}

static int peer_refuse(int fd, const char *reason, char *err, size_t errlen) {
    peer_set_error(err, errlen, "%s", reason);
    char send_err[256] = {0};
    if (peer_send_error(fd, reason, send_err, sizeof(send_err)) != 0) {
        peer_log("ERROR", reason);
    }
    return 1;
}

static bool peer_file_rewind(FILE *file) {
    if (!file) return false;
#if defined(_WIN32)
    return _fseeki64(file, 0, SEEK_SET) == 0;
#else
    return fseeko(file, 0, SEEK_SET) == 0;
#endif
}

static void peer_payload_source_release(ds4_peer_payload_source *source) {
    if (!source) return;
    if (source->release) {
        source->release(source->opaque, source->file);
    } else if (source->file) {
        fclose(source->file);
    }
    memset(source, 0, sizeof(*source));
}

static int peer_send_kv_file(int fd, FILE *file, uint64_t bytes,
                             uint64_t test_mutate_wire_byte,
                             char *err, size_t errlen) {
    if (!file || bytes == 0 || bytes > DS4_PEER_MAX_PAYLOAD_BYTES) {
        peer_set_error(err, errlen, "DS4P staged KV payload size is invalid");
        return 1;
    }
    if (test_mutate_wire_byte != DS4_PEER_NO_MUTATION &&
        test_mutate_wire_byte >= bytes) {
        peer_set_error(err, errlen, "DS4P test mutation byte is outside the KV payload");
        return 1;
    }
    if (!peer_file_rewind(file)) {
        peer_set_error(err, errlen, "failed to rewind staged DS4P KV payload");
        return 1;
    }

    uint8_t begin[8];
    peer_store_u64(begin, bytes);
    if (peer_send_frame(fd, DS4_PEER_FRAME_KV_BEGIN, begin, sizeof(begin),
                        err, errlen) != 0) {
        return 1;
    }

    peer_sha256 sha;
    peer_sha256_init(&sha);
    uint8_t chunk[DS4_PEER_KV_CHUNK_BYTES];
    uint64_t remaining = bytes;
    uint64_t offset = 0;
    while (remaining != 0) {
        const size_t want = remaining > sizeof(chunk) ?
            sizeof(chunk) : (size_t)remaining;
        if (fread(chunk, 1, want, file) != want) {
            peer_set_error(err, errlen, "failed to read staged DS4P KV payload");
            return 1;
        }
        peer_sha256_update(&sha, chunk, want);
        if (test_mutate_wire_byte != DS4_PEER_NO_MUTATION &&
            test_mutate_wire_byte >= offset &&
            test_mutate_wire_byte - offset < (uint64_t)want) {
            chunk[(size_t)(test_mutate_wire_byte - offset)] ^= UINT8_C(0x01);
        }
        if (peer_send_frame(fd, DS4_PEER_FRAME_KV_DATA, chunk, want,
                            err, errlen) != 0) {
            return 1;
        }
        remaining -= want;
        offset += want;
    }

    uint8_t digest[DS4_PEER_SHA256_BYTES];
    peer_sha256_final(&sha, digest);
    if (peer_send_frame(fd, DS4_PEER_FRAME_KV_END, digest, sizeof(digest),
                        err, errlen) != 0) {
        return 1;
    }

    ds4_peer_frame frame = {0};
    int rc = peer_recv_frame(fd, &frame, err, errlen);
    if (rc == 0) {
        peer_set_error(err, errlen, "DS4P peer closed before KV_ACK");
        return 1;
    }
    if (rc < 0) return 1;
    if (frame.type == DS4_PEER_FRAME_ERROR) {
        (void)peer_read_error_frame(&frame, err, errlen);
        peer_frame_free(&frame);
        return 1;
    }
    const bool acknowledged = frame.type == DS4_PEER_FRAME_KV_ACK &&
                              frame.body_len == sizeof(digest) &&
                              memcmp(frame.body, digest, sizeof(digest)) == 0;
    peer_frame_free(&frame);
    if (!acknowledged) {
        peer_set_error(err, errlen, "DS4P peer sent an invalid KV_ACK");
        return 1;
    }
    peer_log("KV", "payload sent and SHA-256 acknowledged");
    return 0;
}

static int peer_recv_kv_file(int fd, FILE **out_file, uint64_t *out_bytes,
                             uint8_t out_digest[DS4_PEER_SHA256_BYTES],
                             char *err, size_t errlen) {
    if (out_file) *out_file = NULL;
    if (out_bytes) *out_bytes = 0;

    ds4_peer_frame frame = {0};
    int rc = peer_recv_frame(fd, &frame, err, errlen);
    if (rc == 0) {
        peer_set_error(err, errlen, "DS4P peer closed before KV_BEGIN");
        return 1;
    }
    if (rc < 0) return 1;
    if (frame.type == DS4_PEER_FRAME_ERROR) {
        (void)peer_read_error_frame(&frame, err, errlen);
        peer_frame_free(&frame);
        return 1;
    }
    if (frame.type != DS4_PEER_FRAME_KV_BEGIN || frame.body_len != 8u) {
        peer_frame_free(&frame);
        return peer_refuse(fd, "DS4P expected KV_BEGIN with payload length", err, errlen);
    }
    const uint64_t expected_bytes = peer_load_u64(frame.body);
    peer_frame_free(&frame);
    if (expected_bytes == 0 || expected_bytes > DS4_PEER_MAX_PAYLOAD_BYTES) {
        return peer_refuse(fd, "DS4P KV payload length is invalid", err, errlen);
    }

    FILE *file = tmpfile();
    if (!file) {
        return peer_refuse(fd, "DS4P could not create a temporary KV payload file", err, errlen);
    }
    peer_sha256 sha;
    peer_sha256_init(&sha);
    uint64_t received = 0;
    while (received < expected_bytes) {
        memset(&frame, 0, sizeof(frame));
        rc = peer_recv_frame(fd, &frame, err, errlen);
        if (rc == 0) {
            fclose(file);
            peer_set_error(err, errlen, "DS4P peer closed during KV payload");
            return 1;
        }
        if (rc < 0) {
            fclose(file);
            return 1;
        }
        if (frame.type == DS4_PEER_FRAME_ERROR) {
            (void)peer_read_error_frame(&frame, err, errlen);
            peer_frame_free(&frame);
            fclose(file);
            return 1;
        }
        if (frame.type != DS4_PEER_FRAME_KV_DATA || frame.body_len == 0 ||
            (uint64_t)frame.body_len > expected_bytes - received) {
            peer_frame_free(&frame);
            fclose(file);
            return peer_refuse(fd, "DS4P KV_DATA frame length is invalid", err, errlen);
        }
        if (fwrite(frame.body, 1, frame.body_len, file) != frame.body_len) {
            peer_frame_free(&frame);
            fclose(file);
            return peer_refuse(fd, "DS4P could not write temporary KV payload", err, errlen);
        }
        peer_sha256_update(&sha, frame.body, frame.body_len);
        received += frame.body_len;
        peer_frame_free(&frame);
    }

    memset(&frame, 0, sizeof(frame));
    rc = peer_recv_frame(fd, &frame, err, errlen);
    if (rc == 0) {
        fclose(file);
        peer_set_error(err, errlen, "DS4P peer closed before KV_END");
        return 1;
    }
    if (rc < 0) {
        fclose(file);
        return 1;
    }
    if (frame.type == DS4_PEER_FRAME_ERROR) {
        (void)peer_read_error_frame(&frame, err, errlen);
        peer_frame_free(&frame);
        fclose(file);
        return 1;
    }
    if (frame.type != DS4_PEER_FRAME_KV_END ||
        frame.body_len != DS4_PEER_SHA256_BYTES) {
        peer_frame_free(&frame);
        fclose(file);
        return peer_refuse(fd, "DS4P expected KV_END with SHA-256", err, errlen);
    }

    uint8_t digest[DS4_PEER_SHA256_BYTES];
    peer_sha256_final(&sha, digest);
    const bool matches = memcmp(frame.body, digest, sizeof(digest)) == 0;
    peer_frame_free(&frame);
    if (!matches) {
        fclose(file);
        return peer_refuse(fd, "DS4P KV payload SHA-256 mismatch", err, errlen);
    }
    if (fflush(file) != 0 || !peer_file_rewind(file)) {
        fclose(file);
        return peer_refuse(fd, "DS4P could not finalize temporary KV payload", err, errlen);
    }
    if (peer_send_frame(fd, DS4_PEER_FRAME_KV_ACK, digest, sizeof(digest),
                        err, errlen) != 0) {
        fclose(file);
        return 1;
    }
    if (out_file) *out_file = file;
    else fclose(file);
    if (out_bytes) *out_bytes = expected_bytes;
    if (out_digest) memcpy(out_digest, digest, sizeof(digest));
    peer_log("KV", "payload received and SHA-256 verified");
    return 0;
}

static int peer_decode_prefill_request(const ds4_peer_frame *frame,
                                       int **out_tokens, uint32_t *out_count,
                                       char *err, size_t errlen) {
    if (out_tokens) *out_tokens = NULL;
    if (out_count) *out_count = 0;
    if (!frame || frame->type != DS4_PEER_FRAME_PREFILL_REQ ||
        frame->body_len < 4u) {
        peer_set_error(err, errlen, "DS4P PREFILL_REQ body is invalid");
        return 1;
    }
    ds4_peer_reader reader = {.data = frame->body, .len = frame->body_len};
    uint32_t count = 0;
    if (!peer_reader_u32(&reader, &count) || count == 0 ||
        count > (DS4_PEER_MAX_FRAME_BODY - 4u) / 4u ||
        reader.len - reader.pos != (size_t)count * 4u) {
        peer_set_error(err, errlen, "DS4P PREFILL_REQ token vector is invalid");
        return 1;
    }
    int *tokens = malloc((size_t)count * sizeof(*tokens));
    if (!tokens) {
        peer_set_error(err, errlen, "out of memory decoding DS4P PREFILL_REQ");
        return 1;
    }
    for (uint32_t i = 0; i < count; i++) {
        uint32_t encoded = 0;
        if (!peer_reader_u32(&reader, &encoded)) {
            free(tokens);
            peer_set_error(err, errlen, "DS4P PREFILL_REQ token vector is truncated");
            return 1;
        }
        tokens[i] = (int)(int32_t)encoded;
    }
    *out_tokens = tokens;
    *out_count = count;
    return 0;
}

static int peer_handle_prefill_request(int fd, const ds4_peer_frame *frame,
                                       const ds4_peer_handshake_result *result,
                                       const ds4_peer_service *service,
                                       char *err, size_t errlen) {
    if (!result || result->payload_version != DS4_PEER_PAYLOAD_VERSION_V3 ||
        result->transfer_mode != DS4_PEER_TRANSFER_NATIVE) {
        return peer_refuse(fd,
                           "DS4P PREFILL_REQ requires negotiated native payload v3",
                           err, errlen);
    }
    if (result->peer.role != DS4_PEER_ROLE_PREFILL || !service ||
        !service->prefill) {
        return peer_refuse(fd, "DS4P PREFILL_REQ service is unavailable", err, errlen);
    }
    int *tokens = NULL;
    uint32_t count = 0;
    if (peer_decode_prefill_request(frame, &tokens, &count, err, errlen) != 0) {
        char reason[512];
        snprintf(reason, sizeof(reason), "%s", err && err[0] ? err :
                 "invalid DS4P PREFILL_REQ");
        return peer_refuse(fd, reason, err, errlen);
    }
    ds4_peer_payload_source source = {
        .test_mutate_wire_byte = DS4_PEER_NO_MUTATION,
    };
    int rc = service->prefill(service->opaque, tokens, count, &source,
                              err, errlen);
    free(tokens);
    if (rc != 0 || !source.file || source.bytes == 0) {
        char reason[512];
        snprintf(reason, sizeof(reason), "%s", err && err[0] ? err :
                 "DS4P prefill service did not produce a staged payload");
        peer_payload_source_release(&source);
        return peer_refuse(fd, reason, err, errlen);
    }
    rc = peer_send_kv_file(fd, source.file, source.bytes,
                           source.test_mutate_wire_byte, err, errlen);
    peer_payload_source_release(&source);
    return rc;
}

uint32_t ds4_peer_token_common_prefix(const int *a, uint32_t a_count,
                                      const int *b, uint32_t b_count) {
    const uint32_t limit = a_count < b_count ? a_count : b_count;
    if ((!a && a_count) || (!b && b_count)) return 0;
    uint32_t common = 0;
    while (common < limit && a[common] == b[common]) common++;
    return common;
}

static uint32_t peer_session_max_tokens(bool has_expected_common) {
    const size_t fixed = 8u + (has_expected_common ? 12u : 4u);
    return (uint32_t)((DS4_PEER_MAX_FRAME_BODY - fixed) / 4u);
}

static int peer_encode_session_tokens(ds4_peer_buffer *body,
                                      uint64_t session_id,
                                      const int *tokens, uint32_t n_tokens,
                                      bool has_expected_common,
                                      uint32_t expected_common,
                                      ds4_peer_session_resume_mode mode,
                                      char *err, size_t errlen) {
    const bool suffix = mode == DS4_PEER_SESSION_RESUME_SUFFIX;
    const bool mode_valid = !has_expected_common ||
        mode == DS4_PEER_SESSION_RESUME_FULL || suffix;
    if (!body || session_id == 0 || !tokens ||
        (!suffix && n_tokens == 0) ||
        n_tokens > peer_session_max_tokens(has_expected_common) ||
        !mode_valid ||
        (has_expected_common && !suffix && expected_common > n_tokens) ||
        (has_expected_common && suffix &&
         expected_common > UINT32_MAX - n_tokens)) {
        peer_set_error(err, errlen, "DS4P session token timeline is invalid");
        return 1;
    }
    if (!peer_buffer_u64(body, session_id) ||
        (has_expected_common && !peer_buffer_u32(body, expected_common)) ||
        (has_expected_common && !peer_buffer_u32(body, (uint32_t)mode)) ||
        !peer_buffer_u32(body, n_tokens)) {
        peer_set_error(err, errlen, "out of memory encoding DS4P session timeline");
        return 1;
    }
    for (uint32_t i = 0; i < n_tokens; i++) {
        if (!peer_buffer_u32(body, (uint32_t)tokens[i])) {
            peer_set_error(err, errlen,
                           "out of memory encoding DS4P session timeline");
            return 1;
        }
    }
    return 0;
}

static int peer_decode_session_tokens(const ds4_peer_frame *frame,
                                      uint32_t expected_type,
                                      bool has_expected_common,
                                      uint64_t *out_session_id,
                                      int **out_tokens, uint32_t *out_count,
                                      uint32_t *out_expected_common,
                                      ds4_peer_session_resume_mode *out_mode,
                                      char *err, size_t errlen) {
    if (out_session_id) *out_session_id = 0;
    if (out_tokens) *out_tokens = NULL;
    if (out_count) *out_count = 0;
    if (out_expected_common) *out_expected_common = 0;
    if (!frame || frame->type != expected_type) {
        peer_set_error(err, errlen, "DS4P session frame type is invalid");
        return 1;
    }
    const size_t fixed = 8u + (has_expected_common ? 12u : 4u);
    if (frame->body_len < fixed) {
        peer_set_error(err, errlen, "DS4P session token timeline is truncated");
        return 1;
    }
    ds4_peer_reader reader = {.data = frame->body, .len = frame->body_len};
    uint64_t session_id = 0;
    uint32_t expected_common = 0;
    uint32_t encoded_mode = DS4_PEER_SESSION_RESUME_FULL;
    uint32_t count = 0;
    if (!peer_reader_u64(&reader, &session_id) ||
        (has_expected_common && !peer_reader_u32(&reader, &expected_common)) ||
        (has_expected_common && !peer_reader_u32(&reader, &encoded_mode)) ||
        !peer_reader_u32(&reader, &count) || session_id == 0 || count == 0 ||
        count > peer_session_max_tokens(has_expected_common) ||
        (has_expected_common &&
         encoded_mode != DS4_PEER_SESSION_RESUME_FULL &&
         encoded_mode != DS4_PEER_SESSION_RESUME_SUFFIX) ||
        (has_expected_common &&
         encoded_mode == DS4_PEER_SESSION_RESUME_FULL &&
         expected_common > count) ||
        (has_expected_common &&
         encoded_mode == DS4_PEER_SESSION_RESUME_SUFFIX &&
         expected_common > UINT32_MAX - count) ||
        reader.len - reader.pos != (size_t)count * 4u) {
        peer_set_error(err, errlen, "DS4P session token timeline is invalid");
        return 1;
    }
    int *tokens = malloc((size_t)count * sizeof(*tokens));
    if (!tokens) {
        peer_set_error(err, errlen, "out of memory decoding DS4P session timeline");
        return 1;
    }
    for (uint32_t i = 0; i < count; i++) {
        uint32_t encoded = 0;
        if (!peer_reader_u32(&reader, &encoded)) {
            free(tokens);
            peer_set_error(err, errlen, "DS4P session token timeline is truncated");
            return 1;
        }
        tokens[i] = (int)(int32_t)encoded;
    }
    if (out_session_id) *out_session_id = session_id;
    if (out_tokens) *out_tokens = tokens;
    else free(tokens);
    if (out_count) *out_count = count;
    if (out_expected_common) *out_expected_common = expected_common;
    if (out_mode) *out_mode = (ds4_peer_session_resume_mode)encoded_mode;
    return 0;
}

static bool peer_session_reply_sane(const ds4_peer_session_reply *reply,
                                    uint32_t n_logits) {
    return reply && reply->session_id != 0 &&
           reply->cache_source <= DS4_PEER_SESSION_CACHE_CHECKPOINT &&
           n_logits <= (DS4_PEER_MAX_FRAME_BODY - 32u) / 4u;
}

static int peer_send_session_reply(int fd, uint32_t type,
                                   const ds4_peer_session_reply *reply,
                                   const float *logits, uint32_t n_logits,
                                   char *err, size_t errlen) {
    if (!peer_session_reply_sane(reply, n_logits) || (n_logits && !logits)) {
        peer_set_error(err, errlen, "DS4P session reply is invalid");
        return 1;
    }
    ds4_peer_buffer body = {0};
    bool ok = peer_buffer_u64(&body, reply->session_id) &&
              peer_buffer_u64(&body, reply->evals) &&
              peer_buffer_u32(&body, reply->cache_source) &&
              peer_buffer_u32(&body, reply->cached_tokens) &&
              peer_buffer_u32(&body, reply->prefill_tokens) &&
              peer_buffer_u32(&body, n_logits);
    for (uint32_t i = 0; ok && i < n_logits; i++) {
        ok = peer_buffer_f32(&body, logits[i]);
    }
    if (!ok) {
        peer_buffer_free(&body);
        peer_set_error(err, errlen, "out of memory encoding DS4P session reply");
        return 1;
    }
    const int rc = peer_send_frame(fd, type, body.data, body.len, err, errlen);
    peer_buffer_free(&body);
    return rc;
}

static int peer_decode_session_reply(const ds4_peer_frame *frame,
                                     uint32_t expected_type,
                                     bool require_logits,
                                     ds4_peer_session_reply *out_reply,
                                     float *logits, uint32_t logits_cap,
                                     uint32_t *out_logits,
                                     char *err, size_t errlen) {
    if (out_reply) memset(out_reply, 0, sizeof(*out_reply));
    if (out_logits) *out_logits = 0;
    if (!frame || frame->type != expected_type || frame->body_len < 32u) {
        peer_set_error(err, errlen, "DS4P session reply has an unexpected frame type");
        return 1;
    }
    ds4_peer_reader reader = {.data = frame->body, .len = frame->body_len};
    ds4_peer_session_reply reply = {0};
    uint32_t n_logits = 0;
    if (!peer_reader_u64(&reader, &reply.session_id) ||
        !peer_reader_u64(&reader, &reply.evals) ||
        !peer_reader_u32(&reader, &reply.cache_source) ||
        !peer_reader_u32(&reader, &reply.cached_tokens) ||
        !peer_reader_u32(&reader, &reply.prefill_tokens) ||
        !peer_reader_u32(&reader, &n_logits) ||
        !peer_session_reply_sane(&reply, n_logits) ||
        (require_logits && n_logits == 0) || n_logits > logits_cap ||
        (n_logits && !logits) ||
        reader.len - reader.pos != (size_t)n_logits * 4u) {
        peer_set_error(err, errlen, "DS4P session reply is invalid");
        return 1;
    }
    for (uint32_t i = 0; i < n_logits; i++) {
        if (!peer_reader_f32(&reader, &logits[i])) {
            peer_set_error(err, errlen, "DS4P session reply logits are truncated");
            return 1;
        }
    }
    if (out_reply) *out_reply = reply;
    if (out_logits) *out_logits = n_logits;
    return 0;
}

static int peer_recv_session_reply(int fd, uint32_t expected_type,
                                   bool require_logits,
                                   ds4_peer_session_reply *reply,
                                   float *logits, uint32_t logits_cap,
                                   uint32_t *out_logits,
                                   char *err, size_t errlen) {
    ds4_peer_frame frame = {0};
    const int rc = peer_recv_frame(fd, &frame, err, errlen);
    if (rc == 0) {
        peer_set_error(err, errlen, "DS4P peer closed before session reply");
        return 1;
    }
    if (rc < 0) return 1;
    if (frame.type == DS4_PEER_FRAME_ERROR) {
        (void)peer_read_error_frame(&frame, err, errlen);
        peer_frame_free(&frame);
        return 1;
    }
    const int decode_rc = peer_decode_session_reply(
        &frame, expected_type, require_logits, reply, logits, logits_cap,
        out_logits, err, errlen);
    peer_frame_free(&frame);
    return decode_rc;
}

static int peer_validate_timeline_reply(const ds4_peer_session_reply *reply,
                                        uint64_t session_id,
                                        uint32_t n_tokens,
                                        uint32_t expected_common,
                                        char *err, size_t errlen) {
    if (!reply || reply->session_id != session_id ||
        reply->cached_tokens != expected_common ||
        reply->prefill_tokens != n_tokens - expected_common) {
        peer_set_error(err, errlen,
                       "DS4P session prefix accounting mismatch: id=%llu cached=%u expected_cached=%u prefill=%u expected_prefill=%u",
                       (unsigned long long)(reply ? reply->session_id : 0),
                       reply ? reply->cached_tokens : 0,
                       expected_common,
                       reply ? reply->prefill_tokens : 0,
                       n_tokens - expected_common);
        return 1;
    }
    return 0;
}

static int peer_decode_session_id(const ds4_peer_frame *frame,
                                  uint32_t expected_type,
                                  uint64_t *out_session_id,
                                  char *err, size_t errlen) {
    if (out_session_id) *out_session_id = 0;
    if (!frame || frame->type != expected_type || frame->body_len != 8u) {
        peer_set_error(err, errlen, "DS4P session id frame is invalid");
        return 1;
    }
    const uint64_t session_id = peer_load_u64(frame->body);
    if (session_id == 0) {
        peer_set_error(err, errlen, "DS4P session id must be non-zero");
        return 1;
    }
    if (out_session_id) *out_session_id = session_id;
    return 0;
}

static int peer_require_decode_session(const ds4_peer_handshake_result *result,
                                       const ds4_peer_service *service,
                                       char *err, size_t errlen) {
    if (!result || result->payload_version != DS4_PEER_PAYLOAD_VERSION_V3 ||
        result->transfer_mode != DS4_PEER_TRANSFER_NATIVE ||
        result->peer.role != DS4_PEER_ROLE_DECODE || !service) {
        peer_set_error(err, errlen,
                       "DS4P decode session requires negotiated native-v3 role=decode");
        return 1;
    }
    return 0;
}

static int peer_handle_session_new(int fd, const ds4_peer_frame *frame,
                                   const ds4_peer_handshake_result *result,
                                   const ds4_peer_service *service,
                                   char *err, size_t errlen) {
    if (peer_require_decode_session(result, service, err, errlen) != 0 ||
        !service->session_new) {
        return peer_refuse(fd, err[0] ? err : "DS4P SESSION_NEW service is unavailable",
                           err, errlen);
    }
    uint64_t session_id = 0;
    int *tokens = NULL;
    uint32_t n_tokens = 0;
    if (peer_decode_session_tokens(frame, DS4_PEER_FRAME_SESSION_NEW, false,
                                   &session_id, &tokens, &n_tokens, NULL,
                                   NULL,
                                   err, errlen) != 0) {
        return peer_refuse(fd, err[0] ? err : "invalid DS4P SESSION_NEW", err, errlen);
    }
    FILE *payload = NULL;
    uint64_t payload_bytes = 0;
    const int payload_rc = peer_recv_kv_file(fd, &payload, &payload_bytes,
                                             NULL, err, errlen);
    if (payload_rc != 0) {
        free(tokens);
        return 1;
    }
    float *logits = malloc(DS4_PEER_MAX_FRAME_BODY);
    if (!logits) {
        free(tokens);
        fclose(payload);
        return peer_refuse(fd, "out of memory serving DS4P SESSION_NEW", err, errlen);
    }
    ds4_peer_session_reply reply = {0};
    uint32_t n_logits = 0;
    int rc = service->session_new(service->opaque, session_id, tokens, n_tokens,
                                  payload, payload_bytes, &reply, logits,
                                  DS4_PEER_MAX_FRAME_BODY / (uint32_t)sizeof(float),
                                  &n_logits, err, errlen);
    free(tokens);
    fclose(payload);
    if (rc == 0) {
        rc = peer_validate_timeline_reply(&reply, session_id, n_tokens, 0,
                                          err, errlen);
    }
    if (rc == 0) {
        rc = peer_send_session_reply(fd, DS4_PEER_FRAME_SESSION_NEW_ACK,
                                     &reply, logits, n_logits, err, errlen);
    }
    free(logits);
    if (rc != 0) {
        return peer_refuse(fd, err[0] ? err : "DS4P SESSION_NEW failed", err, errlen);
    }
    return 0;
}

static int peer_handle_session_resume(int fd, const ds4_peer_frame *frame,
                                      const ds4_peer_handshake_result *result,
                                      const ds4_peer_service *service,
                                      char *err, size_t errlen) {
    if (peer_require_decode_session(result, service, err, errlen) != 0 ||
        !service->session_resume) {
        return peer_refuse(fd, err[0] ? err : "DS4P SESSION_RESUME service is unavailable",
                           err, errlen);
    }
    uint64_t session_id = 0;
    int *tokens = NULL;
    uint32_t n_tokens = 0;
    uint32_t expected_common = 0;
    ds4_peer_session_resume_mode mode = DS4_PEER_SESSION_RESUME_FULL;
    if (peer_decode_session_tokens(frame, DS4_PEER_FRAME_SESSION_RESUME, true,
                                   &session_id, &tokens, &n_tokens,
                                   &expected_common, &mode, err, errlen) != 0) {
        return peer_refuse(fd, err[0] ? err : "invalid DS4P SESSION_RESUME", err, errlen);
    }
    float *logits = malloc(DS4_PEER_MAX_FRAME_BODY);
    if (!logits) {
        free(tokens);
        return peer_refuse(fd, "out of memory serving DS4P SESSION_RESUME", err, errlen);
    }
    ds4_peer_session_reply reply = {0};
    uint32_t n_logits = 0;
    const uint32_t total_tokens = mode == DS4_PEER_SESSION_RESUME_SUFFIX ?
        expected_common + n_tokens : n_tokens;
    int rc = service->session_resume(service->opaque, session_id, tokens,
                                     n_tokens, expected_common, mode,
                                     &reply, logits,
                                     DS4_PEER_MAX_FRAME_BODY / (uint32_t)sizeof(float),
                                     &n_logits, err, errlen);
    free(tokens);
    if (rc == 0) {
        rc = peer_validate_timeline_reply(&reply, session_id, total_tokens,
                                          expected_common, err, errlen);
    }
    if (rc == 0) {
        rc = peer_send_session_reply(fd, DS4_PEER_FRAME_SESSION_RESUME_ACK,
                                     &reply, logits, n_logits, err, errlen);
    }
    free(logits);
    if (rc != 0) {
        return peer_refuse(fd, err[0] ? err : "DS4P SESSION_RESUME failed", err, errlen);
    }
    return 0;
}

static int peer_handle_session_id_request(
        int fd, const ds4_peer_frame *frame, uint32_t request_type,
        uint32_t reply_type, const ds4_peer_handshake_result *result,
        const ds4_peer_service *service, bool is_cancel,
        char *err, size_t errlen) {
    if (peer_require_decode_session(result, service, err, errlen) != 0 ||
        (is_cancel ? !service->session_cancel : !service->session_drop)) {
        return peer_refuse(fd, err[0] ? err : "DS4P session service is unavailable",
                           err, errlen);
    }
    uint64_t session_id = 0;
    if (peer_decode_session_id(frame, request_type, &session_id, err, errlen) != 0) {
        return peer_refuse(fd, err[0] ? err : "invalid DS4P session id", err, errlen);
    }
    ds4_peer_session_reply reply = {0};
    const int rc = is_cancel
        ? service->session_cancel(service->opaque, session_id, &reply, err, errlen)
        : service->session_drop(service->opaque, session_id, &reply, err, errlen);
    if (rc != 0 || reply.session_id != session_id) {
        return peer_refuse(fd, err[0] ? err : "DS4P session operation failed", err, errlen);
    }
    if (peer_send_session_reply(fd, reply_type, &reply, NULL, 0, err, errlen) != 0) {
        return 1;
    }
    return 0;
}

static int peer_handle_session_eval(int fd, const ds4_peer_frame *frame,
                                    const ds4_peer_handshake_result *result,
                                    const ds4_peer_service *service,
                                    char *err, size_t errlen) {
    if (peer_require_decode_session(result, service, err, errlen) != 0 ||
        !service->session_eval) {
        return peer_refuse(fd, err[0] ? err : "DS4P EVAL service is unavailable", err, errlen);
    }
    if (!frame || frame->type != DS4_PEER_FRAME_EVAL_REQ || frame->body_len != 12u) {
        return peer_refuse(fd, "DS4P EVAL_REQ body is invalid", err, errlen);
    }
    const uint64_t session_id = peer_load_u64(frame->body);
    const int token = (int)(int32_t)peer_load_u32(frame->body + 8u);
    if (session_id == 0) {
        return peer_refuse(fd, "DS4P EVAL_REQ session id must be non-zero", err, errlen);
    }
    float *logits = malloc(DS4_PEER_MAX_FRAME_BODY);
    if (!logits) return peer_refuse(fd, "out of memory serving DS4P EVAL_REQ", err, errlen);
    ds4_peer_session_reply reply = {0};
    uint32_t n_logits = 0;
    int rc = service->session_eval(service->opaque, session_id, token, &reply,
                                   logits,
                                   DS4_PEER_MAX_FRAME_BODY / (uint32_t)sizeof(float),
                                   &n_logits, err, errlen);
    if (rc == 0 && reply.session_id != session_id) {
        peer_set_error(err, errlen, "DS4P EVAL reply session id does not match request");
        rc = 1;
    }
    if (rc == 0) {
        rc = peer_send_session_reply(fd, DS4_PEER_FRAME_EVAL_LOGITS,
                                     &reply, logits, n_logits, err, errlen);
    }
    free(logits);
    if (rc != 0) return peer_refuse(fd, err[0] ? err : "DS4P EVAL_REQ failed", err, errlen);
    return 0;
}

static int peer_verify_ack(const ds4_peer_handshake_result *proposal,
                           const ds4_peer_handshake_result *ack,
                           const ds4_peer_capabilities *hello_peer,
                           char *err, size_t errlen) {
    if (!peer_capabilities_equal(hello_peer, &ack->peer)) {
        peer_set_error(err, errlen,
                       "peer HELLO_ACK capabilities changed during handshake");
        return 1;
    }
    if (ack->payload_version != proposal->payload_version ||
        ack->transfer_mode != proposal->transfer_mode) {
        peer_set_error(err, errlen,
                       "peer HELLO_ACK negotiation mismatch: local=v%u/%s peer=v%u/%s",
                       proposal->payload_version,
                       peer_transfer_name(proposal->transfer_mode),
                       ack->payload_version,
                       peer_transfer_name(ack->transfer_mode));
        return 1;
    }
    return 0;
}

static void peer_log_ready(const ds4_peer_handshake_result *result) {
    char line[256];
    snprintf(line, sizeof(line),
             "ready peer=%s peer_role=%s payload=v%u mode=%s format=%s",
             result->peer.model_name,
             peer_role_name(result->peer.role),
             result->payload_version,
             peer_transfer_name(result->transfer_mode),
             peer_format_name(result->peer.attn_comp_cache_format));
    peer_log("READY", line);
}

static int peer_handshake_client_fd(int fd,
                                    const ds4_peer_capabilities *local,
                                    ds4_peer_handshake_result *result,
                                    char *err, size_t errlen) {
    if (peer_capabilities_sane(local, err, errlen) != 0) return 1;
    if (peer_send_hello(fd, local, err, errlen) != 0) return 1;

    ds4_peer_frame frame = {0};
    int rc = peer_recv_frame(fd, &frame, err, errlen);
    if (rc == 0) {
        peer_set_error(err, errlen, "DS4P peer closed before HELLO");
        return 1;
    }
    if (rc < 0) return 1;
    if (frame.type == DS4_PEER_FRAME_ERROR) {
        (void)peer_read_error_frame(&frame, err, errlen);
        peer_frame_free(&frame);
        return 1;
    }
    ds4_peer_capabilities peer_hello;
    if (peer_decode_hello(&frame, &peer_hello, err, errlen) != 0) {
        char reason[512];
        snprintf(reason, sizeof(reason), "%s", err && err[0] ? err : "invalid DS4P HELLO");
        peer_frame_free(&frame);
        return peer_refuse(fd, reason, err, errlen);
    }
    peer_frame_free(&frame);

    ds4_peer_handshake_result proposal = {.peer = peer_hello};
    if (peer_negotiate(local, &peer_hello,
                       &proposal.payload_version, &proposal.transfer_mode,
                       err, errlen) != 0) {
        char reason[512];
        snprintf(reason, sizeof(reason), "%s", err && err[0] ? err : "DS4P negotiation failed");
        return peer_refuse(fd, reason, err, errlen);
    }
    if (peer_send_ack(fd, local, proposal.payload_version, proposal.transfer_mode,
                      err, errlen) != 0) {
        return 1;
    }

    rc = peer_recv_frame(fd, &frame, err, errlen);
    if (rc == 0) {
        peer_set_error(err, errlen, "DS4P peer closed before HELLO_ACK");
        return 1;
    }
    if (rc < 0) return 1;
    if (frame.type == DS4_PEER_FRAME_ERROR) {
        (void)peer_read_error_frame(&frame, err, errlen);
        peer_frame_free(&frame);
        return 1;
    }
    ds4_peer_handshake_result ack;
    if (peer_decode_ack(&frame, &ack, err, errlen) != 0) {
        char reason[512];
        snprintf(reason, sizeof(reason), "%s", err && err[0] ? err : "invalid DS4P HELLO_ACK");
        peer_frame_free(&frame);
        return peer_refuse(fd, reason, err, errlen);
    }
    peer_frame_free(&frame);
    if (peer_verify_ack(&proposal, &ack, &peer_hello, err, errlen) != 0) {
        char reason[512];
        snprintf(reason, sizeof(reason), "%s", err && err[0] ? err : "DS4P HELLO_ACK refused");
        return peer_refuse(fd, reason, err, errlen);
    }
    if (result) *result = proposal;
    peer_log_ready(&proposal);
    return 0;
}

static int peer_handshake_server_fd(int fd,
                                    const ds4_peer_capabilities *local,
                                    ds4_peer_handshake_result *result,
                                    char *err, size_t errlen) {
    if (peer_capabilities_sane(local, err, errlen) != 0) return 1;

    ds4_peer_frame frame = {0};
    int rc = peer_recv_frame(fd, &frame, err, errlen);
    if (rc == 0) {
        peer_set_error(err, errlen, "DS4P peer closed before HELLO");
        return 1;
    }
    if (rc < 0) return 1;
    if (frame.type == DS4_PEER_FRAME_ERROR) {
        (void)peer_read_error_frame(&frame, err, errlen);
        peer_frame_free(&frame);
        return 1;
    }
    ds4_peer_capabilities peer_hello;
    if (peer_decode_hello(&frame, &peer_hello, err, errlen) != 0) {
        char reason[512];
        snprintf(reason, sizeof(reason), "%s", err && err[0] ? err : "invalid DS4P HELLO");
        peer_frame_free(&frame);
        return peer_refuse(fd, reason, err, errlen);
    }
    peer_frame_free(&frame);

    /* Always send our HELLO before a capability refusal so both directions
     * have exchanged the full runtime description for diagnostics. */
    if (peer_send_hello(fd, local, err, errlen) != 0) return 1;

    ds4_peer_handshake_result proposal = {.peer = peer_hello};
    if (peer_negotiate(local, &peer_hello,
                       &proposal.payload_version, &proposal.transfer_mode,
                       err, errlen) != 0) {
        char reason[512];
        snprintf(reason, sizeof(reason), "%s", err && err[0] ? err : "DS4P negotiation failed");
        return peer_refuse(fd, reason, err, errlen);
    }

    rc = peer_recv_frame(fd, &frame, err, errlen);
    if (rc == 0) {
        peer_set_error(err, errlen, "DS4P peer closed before HELLO_ACK");
        return 1;
    }
    if (rc < 0) return 1;
    if (frame.type == DS4_PEER_FRAME_ERROR) {
        (void)peer_read_error_frame(&frame, err, errlen);
        peer_frame_free(&frame);
        return 1;
    }
    ds4_peer_handshake_result ack;
    if (peer_decode_ack(&frame, &ack, err, errlen) != 0) {
        char reason[512];
        snprintf(reason, sizeof(reason), "%s", err && err[0] ? err : "invalid DS4P HELLO_ACK");
        peer_frame_free(&frame);
        return peer_refuse(fd, reason, err, errlen);
    }
    peer_frame_free(&frame);
    if (peer_verify_ack(&proposal, &ack, &peer_hello, err, errlen) != 0) {
        char reason[512];
        snprintf(reason, sizeof(reason), "%s", err && err[0] ? err : "DS4P HELLO_ACK refused");
        return peer_refuse(fd, reason, err, errlen);
    }
    if (peer_send_ack(fd, local, proposal.payload_version, proposal.transfer_mode,
                      err, errlen) != 0) {
        return 1;
    }
    if (result) *result = proposal;
    peer_log_ready(&proposal);
    return 0;
}

static int peer_control_loop(int fd,
                             const ds4_peer_handshake_result *result,
                             const ds4_peer_service *service,
                             char *err, size_t errlen) {
    for (;;) {
        ds4_peer_frame frame = {0};
        int rc = peer_recv_frame(fd, &frame, err, errlen);
        if (rc == 0) return 0;
        if (rc < 0) return 1;
        if (service && service->session_sweep) {
            service->session_sweep(service->opaque);
        }
        if (frame.type == DS4_PEER_FRAME_ERROR) {
            (void)peer_read_error_frame(&frame, err, errlen);
            peer_frame_free(&frame);
            return 1;
        }
        if (frame.type == DS4_PEER_FRAME_PING) {
            rc = peer_send_frame(fd, DS4_PEER_FRAME_PING,
                                 frame.body, frame.body_len, err, errlen);
            peer_frame_free(&frame);
            if (rc != 0) return 1;
            peer_log("PING", "echoed");
            continue;
        }
        if (frame.type == DS4_PEER_FRAME_PREFILL_REQ) {
            rc = peer_handle_prefill_request(fd, &frame, result, service,
                                             err, errlen);
            peer_frame_free(&frame);
            if (rc != 0) return 1;
            continue;
        }
        if (frame.type == DS4_PEER_FRAME_SESSION_NEW) {
            rc = peer_handle_session_new(fd, &frame, result, service,
                                         err, errlen);
            peer_frame_free(&frame);
            if (rc != 0) return 1;
            continue;
        }
        if (frame.type == DS4_PEER_FRAME_SESSION_RESUME) {
            rc = peer_handle_session_resume(fd, &frame, result, service,
                                            err, errlen);
            peer_frame_free(&frame);
            if (rc != 0) return 1;
            continue;
        }
        if (frame.type == DS4_PEER_FRAME_SESSION_DROP) {
            rc = peer_handle_session_id_request(
                fd, &frame, DS4_PEER_FRAME_SESSION_DROP,
                DS4_PEER_FRAME_SESSION_DROP_ACK, result, service, false,
                err, errlen);
            peer_frame_free(&frame);
            if (rc != 0) return 1;
            continue;
        }
        if (frame.type == DS4_PEER_FRAME_SESSION_CANCEL) {
            rc = peer_handle_session_id_request(
                fd, &frame, DS4_PEER_FRAME_SESSION_CANCEL,
                DS4_PEER_FRAME_SESSION_CANCEL_ACK, result, service, true,
                err, errlen);
            peer_frame_free(&frame);
            if (rc != 0) return 1;
            continue;
        }
        if (frame.type == DS4_PEER_FRAME_EVAL_REQ) {
            rc = peer_handle_session_eval(fd, &frame, result, service,
                                          err, errlen);
            peer_frame_free(&frame);
            if (rc != 0) return 1;
            continue;
        }
        char reason[160];
        snprintf(reason, sizeof(reason),
                 "DS4P frame type %u is not implemented", frame.type);
        peer_frame_free(&frame);
        return peer_refuse(fd, reason, err, errlen);
    }
}

int ds4_peer_ping(int fd, const void *body, size_t body_len,
                  char *err, size_t errlen) {
    if (body_len > DS4_PEER_MAX_FRAME_BODY) {
        peer_set_error(err, errlen, "DS4P PING body is too large");
        return 1;
    }
    if (peer_send_frame(fd, DS4_PEER_FRAME_PING, body, body_len, err, errlen) != 0) {
        return 1;
    }
    ds4_peer_frame frame = {0};
    int rc = peer_recv_frame(fd, &frame, err, errlen);
    if (rc == 0) {
        peer_set_error(err, errlen, "DS4P peer closed before PING reply");
        return 1;
    }
    if (rc < 0) return 1;
    if (frame.type == DS4_PEER_FRAME_ERROR) {
        (void)peer_read_error_frame(&frame, err, errlen);
        peer_frame_free(&frame);
        return 1;
    }
    bool match = frame.type == DS4_PEER_FRAME_PING &&
                 frame.body_len == body_len &&
                 (!body_len || memcmp(frame.body, body, body_len) == 0);
    peer_frame_free(&frame);
    if (!match) {
        peer_set_error(err, errlen, "DS4P PING reply did not echo the request");
        return 1;
    }
    peer_log("PING", "reply received");
    return 0;
}

int ds4_peer_request_prefill(int fd,
                             const ds4_peer_handshake_result *negotiated,
                             const int *tokens, uint32_t n_tokens,
                             FILE **out_payload, uint64_t *out_payload_bytes,
                             uint8_t out_sha256[DS4_PEER_SHA256_BYTES],
                             char *err, size_t errlen) {
    if (out_payload) *out_payload = NULL;
    if (out_payload_bytes) *out_payload_bytes = 0;
    if (!negotiated ||
        negotiated->payload_version != DS4_PEER_PAYLOAD_VERSION_V3 ||
        negotiated->transfer_mode != DS4_PEER_TRANSFER_NATIVE) {
        peer_set_error(err, errlen,
                       "DS4P PREFILL_REQ requires negotiated native payload v3");
        return 1;
    }
    if (!tokens || n_tokens == 0 ||
        n_tokens > (DS4_PEER_MAX_FRAME_BODY - 4u) / 4u) {
        peer_set_error(err, errlen, "DS4P PREFILL_REQ token vector is invalid");
        return 1;
    }
    ds4_peer_buffer body = {0};
    if (!peer_buffer_u32(&body, n_tokens)) {
        peer_buffer_free(&body);
        peer_set_error(err, errlen, "out of memory encoding DS4P PREFILL_REQ");
        return 1;
    }
    for (uint32_t i = 0; i < n_tokens; i++) {
        if (!peer_buffer_u32(&body, (uint32_t)tokens[i])) {
            peer_buffer_free(&body);
            peer_set_error(err, errlen, "out of memory encoding DS4P PREFILL_REQ");
            return 1;
        }
    }
    const int send_rc = peer_send_frame(fd, DS4_PEER_FRAME_PREFILL_REQ,
                                        body.data, body.len, err, errlen);
    peer_buffer_free(&body);
    if (send_rc != 0) return 1;
    return peer_recv_kv_file(fd, out_payload, out_payload_bytes, out_sha256,
                             err, errlen);
}

static int peer_require_native_session_client(
        const ds4_peer_handshake_result *negotiated,
        char *err, size_t errlen) {
    if (!negotiated ||
        negotiated->payload_version != DS4_PEER_PAYLOAD_VERSION_V3 ||
        negotiated->transfer_mode != DS4_PEER_TRANSFER_NATIVE) {
        peer_set_error(err, errlen,
                       "DS4P decode session requires negotiated native payload v3");
        return 1;
    }
    return 0;
}

int ds4_peer_session_new(
        int fd, const ds4_peer_handshake_result *negotiated,
        uint64_t session_id, const int *tokens, uint32_t n_tokens,
        FILE *payload, uint64_t payload_bytes,
        ds4_peer_session_reply *reply, float *logits, uint32_t logits_cap,
        uint32_t *out_logits, char *err, size_t errlen) {
    if (reply) memset(reply, 0, sizeof(*reply));
    if (out_logits) *out_logits = 0;
    if (peer_require_native_session_client(negotiated, err, errlen) != 0 ||
        !payload || payload_bytes == 0 || payload_bytes > DS4_PEER_MAX_PAYLOAD_BYTES) {
        if (!err || !err[0]) {
            peer_set_error(err, errlen, "DS4P SESSION_NEW payload is invalid");
        }
        return 1;
    }
    ds4_peer_buffer body = {0};
    int rc = peer_encode_session_tokens(&body, session_id, tokens, n_tokens,
                                        false, 0,
                                        DS4_PEER_SESSION_RESUME_FULL,
                                        err, errlen);
    if (rc == 0) {
        rc = peer_send_frame(fd, DS4_PEER_FRAME_SESSION_NEW,
                             body.data, body.len, err, errlen);
    }
    peer_buffer_free(&body);
    if (rc != 0) return 1;
    if (peer_send_kv_file(fd, payload, payload_bytes, DS4_PEER_NO_MUTATION,
                          err, errlen) != 0) {
        return 1;
    }
    rc = peer_recv_session_reply(fd, DS4_PEER_FRAME_SESSION_NEW_ACK, true,
                                 reply, logits, logits_cap, out_logits,
                                 err, errlen);
    if (rc == 0) {
        rc = peer_validate_timeline_reply(reply, session_id, n_tokens, 0,
                                          err, errlen);
    }
    if (rc == 0 && reply->cache_source != DS4_PEER_SESSION_CACHE_NEW) {
        peer_set_error(err, errlen,
                       "DS4P SESSION_NEW reply did not report cache_source=new");
        rc = 1;
    }
    return rc;
}

int ds4_peer_session_resume(
        int fd, const ds4_peer_handshake_result *negotiated,
        uint64_t session_id, const int *tokens, uint32_t n_tokens,
        uint32_t expected_common, ds4_peer_session_resume_mode mode,
        ds4_peer_session_reply *reply,
        float *logits, uint32_t logits_cap, uint32_t *out_logits,
        char *err, size_t errlen) {
    if (reply) memset(reply, 0, sizeof(*reply));
    if (out_logits) *out_logits = 0;
    if (peer_require_native_session_client(negotiated, err, errlen) != 0) return 1;
    const uint32_t total_tokens = mode == DS4_PEER_SESSION_RESUME_SUFFIX ?
        expected_common + n_tokens : n_tokens;
    ds4_peer_buffer body = {0};
    int rc = peer_encode_session_tokens(&body, session_id, tokens, n_tokens,
                                        true, expected_common, mode,
                                        err, errlen);
    if (rc == 0) {
        rc = peer_send_frame(fd, DS4_PEER_FRAME_SESSION_RESUME,
                             body.data, body.len, err, errlen);
    }
    peer_buffer_free(&body);
    if (rc != 0) return 1;
    rc = peer_recv_session_reply(fd, DS4_PEER_FRAME_SESSION_RESUME_ACK, true,
                                 reply, logits, logits_cap, out_logits,
                                 err, errlen);
    if (rc == 0) {
        rc = peer_validate_timeline_reply(reply, session_id, total_tokens,
                                          expected_common, err, errlen);
    }
    return rc;
}

int ds4_peer_session_eval(
        int fd, uint64_t session_id, int token,
        ds4_peer_session_reply *reply, float *logits, uint32_t logits_cap,
        uint32_t *out_logits, char *err, size_t errlen) {
    if (reply) memset(reply, 0, sizeof(*reply));
    if (out_logits) *out_logits = 0;
    if (session_id == 0) {
        peer_set_error(err, errlen, "DS4P EVAL_REQ session id must be non-zero");
        return 1;
    }
    uint8_t body[12];
    peer_store_u64(body, session_id);
    peer_store_u32(body + 8u, (uint32_t)token);
    if (peer_send_frame(fd, DS4_PEER_FRAME_EVAL_REQ, body, sizeof(body),
                        err, errlen) != 0) {
        return 1;
    }
    const int rc = peer_recv_session_reply(fd, DS4_PEER_FRAME_EVAL_LOGITS,
                                           true, reply, logits, logits_cap,
                                           out_logits, err, errlen);
    if (rc == 0 && (!reply || reply->session_id != session_id)) {
        peer_set_error(err, errlen, "DS4P EVAL reply session id does not match request");
        return 1;
    }
    return rc;
}

static int peer_session_id_request(int fd, uint32_t request_type,
                                   uint32_t reply_type, uint64_t session_id,
                                   ds4_peer_session_reply *reply,
                                   char *err, size_t errlen) {
    if (reply) memset(reply, 0, sizeof(*reply));
    if (session_id == 0) {
        peer_set_error(err, errlen, "DS4P session id must be non-zero");
        return 1;
    }
    uint8_t body[8];
    peer_store_u64(body, session_id);
    if (peer_send_frame(fd, request_type, body, sizeof(body), err, errlen) != 0) {
        return 1;
    }
    const int rc = peer_recv_session_reply(fd, reply_type, false, reply,
                                           NULL, 0, NULL, err, errlen);
    if (rc == 0 && (!reply || reply->session_id != session_id)) {
        peer_set_error(err, errlen, "DS4P session reply id does not match request");
        return 1;
    }
    return rc;
}

int ds4_peer_session_cancel(int fd, uint64_t session_id,
                            ds4_peer_session_reply *reply,
                            char *err, size_t errlen) {
    return peer_session_id_request(fd, DS4_PEER_FRAME_SESSION_CANCEL,
                                   DS4_PEER_FRAME_SESSION_CANCEL_ACK,
                                   session_id, reply, err, errlen);
}

int ds4_peer_session_drop(int fd, uint64_t session_id,
                          ds4_peer_session_reply *reply,
                          char *err, size_t errlen) {
    return peer_session_id_request(fd, DS4_PEER_FRAME_SESSION_DROP,
                                   DS4_PEER_FRAME_SESSION_DROP_ACK,
                                   session_id, reply, err, errlen);
}

int ds4_peer_connect_handshake(const char *endpoint,
                               const ds4_peer_capabilities *local,
                               ds4_peer_handshake_result *result,
                               int *out_fd,
                               char *err, size_t errlen) {
    if (out_fd) *out_fd = -1;
    ds4_peer_endpoint parsed;
    if (ds4_peer_parse_endpoint(endpoint, &parsed, err, errlen) != 0) return 1;
    int fd = peer_open_connection(&parsed, err, errlen);
    if (fd < 0) return 1;
    if (peer_handshake_client_fd(fd, local, result, err, errlen) != 0) {
        close(fd);
        return 1;
    }
    if (out_fd) {
        *out_fd = fd;
    } else {
        close(fd);
    }
    return 0;
}

static int peer_serve_once_with_service(const char *endpoint,
                                        const ds4_peer_capabilities *local,
                                        const ds4_peer_service *service,
                                        ds4_peer_handshake_result *result,
                                        char *err, size_t errlen) {
    ds4_peer_endpoint parsed;
    if (ds4_peer_parse_endpoint(endpoint, &parsed, err, errlen) != 0) return 1;
    int lfd = peer_open_listener(&parsed, err, errlen);
    if (lfd < 0) return 1;
    int fd;
    do {
        fd = accept(lfd, NULL, NULL);
    } while (fd < 0 && errno == EINTR);
    close(lfd);
    if (fd < 0) {
        peer_set_error(err, errlen, "DS4P accept failed: %s", strerror(errno));
        return 1;
    }
    peer_set_socket_timeout(fd);
    ds4_peer_handshake_result negotiated = {0};
    int rc = peer_handshake_server_fd(fd, local, &negotiated, err, errlen);
    if (rc == 0) rc = peer_control_loop(fd, &negotiated, service, err, errlen);
    close(fd);
    if (rc == 0 && result) *result = negotiated;
    return rc;
}

int ds4_peer_serve_once(const char *endpoint,
                        const ds4_peer_capabilities *local,
                        ds4_peer_handshake_result *result,
                        char *err, size_t errlen) {
    return peer_serve_once_with_service(endpoint, local, NULL, result,
                                        err, errlen);
}

int ds4_peer_serve_prefill_once(const char *endpoint,
                                const ds4_peer_capabilities *local,
                                const ds4_peer_service *service,
                                ds4_peer_handshake_result *result,
                                char *err, size_t errlen) {
    if (!service || !service->prefill) {
        peer_set_error(err, errlen, "DS4P PREFILL_REQ service is missing");
        return 1;
    }
    return peer_serve_once_with_service(endpoint, local, service, result,
                                        err, errlen);
}

static void *peer_listener_main(void *arg) {
    ds4_peer_listener *listener = arg;
    while (!peer_listener_is_stopping(listener)) {
        const int lfd = listener->fd;
        if (lfd < 0) break;
        struct pollfd ready = {
            .fd = lfd,
            .events = POLLIN,
        };
        const int poll_rc = poll(&ready, 1, 200);
        if (peer_listener_is_stopping(listener)) break;
        if (poll_rc == 0) {
            /* TTL eviction must not depend on a later client frame: an idle
             * decode peer releases its retained KV on the bounded poll tick. */
            if (listener->service.session_sweep) {
                listener->service.session_sweep(listener->service.opaque);
            }
            continue;
        }
        if (poll_rc < 0) {
            if (errno == EINTR) continue;
            char line[256];
            snprintf(line, sizeof(line), "listener poll failed: %s", strerror(errno));
            peer_log("ERROR", line);
            continue;
        }
        int fd = accept(lfd, NULL, NULL);
        if (fd < 0) {
            if (peer_listener_is_stopping(listener)) break;
            if (errno == EINTR) continue;
            char line[256];
            snprintf(line, sizeof(line), "accept failed: %s", strerror(errno));
            peer_log("ERROR", line);
            continue;
        }
        pthread_mutex_lock(&listener->mu);
        const bool stopping = listener->stopping != 0;
        if (!stopping) listener->active_fd = fd;
        pthread_mutex_unlock(&listener->mu);
        if (stopping) {
            close(fd);
            break;
        }
        peer_set_socket_timeout(fd);
        char err[512] = {0};
        ds4_peer_handshake_result result;
        int rc = peer_handshake_server_fd(fd, &listener->local, &result,
                                          err, sizeof(err));
        if (rc == 0) {
            rc = peer_control_loop(fd, &result, &listener->service,
                                   err, sizeof(err));
        }
        if (rc != 0 && err[0]) peer_log("ERROR", err);
        pthread_mutex_lock(&listener->mu);
        if (listener->active_fd == fd) listener->active_fd = -1;
        pthread_mutex_unlock(&listener->mu);
        close(fd);
    }
    return NULL;
}

ds4_peer_listener *ds4_peer_listener_start(const char *endpoint,
                                            const ds4_peer_capabilities *local,
                                            char *err, size_t errlen) {
    return ds4_peer_listener_start_service(endpoint, local, NULL, err, errlen);
}

ds4_peer_listener *ds4_peer_listener_start_service(
        const char *endpoint, const ds4_peer_capabilities *local,
        const ds4_peer_service *service, char *err, size_t errlen) {
    if (peer_capabilities_sane(local, err, errlen) != 0) return NULL;
    if (service && !service->prefill && !service->session_new &&
        !service->session_resume && !service->session_drop &&
        !service->session_eval && !service->session_cancel) {
        peer_set_error(err, errlen, "DS4P listener service is invalid");
        return NULL;
    }
    ds4_peer_endpoint parsed;
    if (ds4_peer_parse_endpoint(endpoint, &parsed, err, errlen) != 0) return NULL;
    int fd = peer_open_listener(&parsed, err, errlen);
    if (fd < 0) return NULL;
    ds4_peer_listener *listener = calloc(1, sizeof(*listener));
    if (!listener) {
        close(fd);
        peer_set_error(err, errlen, "out of memory starting DS4P listener");
        return NULL;
    }
    listener->fd = fd;
    listener->active_fd = -1;
    listener->local = *local;
    if (service) listener->service = *service;
    if (pthread_mutex_init(&listener->mu, NULL) != 0) {
        close(fd);
        free(listener);
        peer_set_error(err, errlen, "failed to initialize DS4P listener mutex");
        return NULL;
    }
    if (pthread_create(&listener->thread, NULL, peer_listener_main, listener) != 0) {
        peer_set_error(err, errlen, "failed to start DS4P listener thread");
        close(fd);
        pthread_mutex_destroy(&listener->mu);
        free(listener);
        return NULL;
    }
    char line[160];
    snprintf(line, sizeof(line), "listening on %s:%u", parsed.host,
             (unsigned)parsed.port);
    peer_log("LISTEN", line);
    return listener;
}

void ds4_peer_listener_stop(ds4_peer_listener *listener) {
    if (!listener) return;
    pthread_mutex_lock(&listener->mu);
    listener->stopping = 1;
    const int active_fd = listener->active_fd;
    pthread_mutex_unlock(&listener->mu);
    /* The listener itself polls with a bounded timeout.  Only an accepted
     * connection needs shutdown to wake a blocking control-frame read; the
     * listening socket remains owned by its thread until join has completed. */
    if (active_fd >= 0) {
#if defined(_WIN32)
        (void)shutdown(active_fd, SD_BOTH);
#else
        (void)shutdown(active_fd, SHUT_RDWR);
#endif
    }
    (void)pthread_join(listener->thread, NULL);
    if (listener->fd >= 0) close(listener->fd);
    listener->fd = -1;
    pthread_mutex_destroy(&listener->mu);
    free(listener);
}
