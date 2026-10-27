#ifndef DS4_PEER_H
#define DS4_PEER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* DS4P carries capability negotiation, topology-A native-v3 KV handoff, and
 * the topology-B owner-sampled decode session protocol.  Session ids are
 * owner-selected non-zero u64 values: a peer must refuse an unknown resume or
 * a duplicate new id rather than silently creating a different conversation. */

#define DS4_PEER_PROTOCOL_VERSION UINT32_C(1)
#define DS4_PEER_MAGIC            UINT32_C(0x50345344) /* "DS4P", little endian */

#define DS4_PEER_MAX_LAYERS       128u
#define DS4_PEER_MODEL_NAME_MAX   128u
#define DS4_PEER_ENDPOINT_HOST_MAX 64u
#define DS4_PEER_ENDPOINT_TEXT_MAX 80u
#define DS4_PEER_SHA256_BYTES     32u
#define DS4_PEER_NO_MUTATION      UINT64_MAX

#define DS4_PEER_PAYLOAD_V2       UINT32_C(0x00000004)
#define DS4_PEER_PAYLOAD_V3       UINT32_C(0x00000008)
#define DS4_PEER_PAYLOAD_VERSION_V2 UINT32_C(2)
#define DS4_PEER_PAYLOAD_VERSION_V3 UINT32_C(3)

typedef enum {
    DS4_PEER_FRAME_HELLO = 1,
    DS4_PEER_FRAME_HELLO_ACK = 2,
    DS4_PEER_FRAME_ERROR = 3,
    DS4_PEER_FRAME_PING = 4,
    DS4_PEER_FRAME_PREFILL_REQ = 5,
    DS4_PEER_FRAME_KV_BEGIN = 6,
    DS4_PEER_FRAME_KV_DATA = 7,
    DS4_PEER_FRAME_KV_END = 8,
    DS4_PEER_FRAME_KV_ACK = 9,
    DS4_PEER_FRAME_SESSION_NEW = 10,
    DS4_PEER_FRAME_SESSION_NEW_ACK = 11,
    DS4_PEER_FRAME_SESSION_RESUME = 12,
    DS4_PEER_FRAME_SESSION_RESUME_ACK = 13,
    DS4_PEER_FRAME_SESSION_DROP = 14,
    DS4_PEER_FRAME_SESSION_DROP_ACK = 15,
    DS4_PEER_FRAME_EVAL_REQ = 16,
    DS4_PEER_FRAME_EVAL_LOGITS = 17,
    DS4_PEER_FRAME_SESSION_CANCEL = 18,
    DS4_PEER_FRAME_SESSION_CANCEL_ACK = 19,
    DS4_PEER_FRAME_SESSION_STATS = 20,
} ds4_peer_frame_type;

typedef enum {
    DS4_PEER_ROLE_NONE = 0,
    /* The owner receives a prefilled payload from the peer. */
    DS4_PEER_ROLE_PREFILL = 1,
    /* The owner sends a payload to the peer for remote evaluation. */
    DS4_PEER_ROLE_DECODE = 2,
} ds4_peer_role;

typedef enum {
    DS4_PEER_TRANSFER_CANONICAL = 0,
    DS4_PEER_TRANSFER_NATIVE = 1,
} ds4_peer_transfer_mode;

enum {
    /* Both peers must advertise these policy bits before an exception is used. */
    DS4_PEER_POLICY_F32_CANONICAL = 1u << 0,
    DS4_PEER_POLICY_ALLOW_INEXACT_KV = 1u << 1,
    DS4_PEER_POLICY_ALLOW_WEIGHT_MISMATCH = 1u << 2,
};

/* All fields are explicitly encoded little-endian; this in-memory layout is
 * never sent as a C struct.  n_layer compression-ratio entries are sent. */
typedef struct {
    uint32_t protocol_version;
    uint32_t role;
    uint32_t policy_flags;

    uint32_t model_family;
    uint32_t model_variant;
    char model_name[DS4_PEER_MODEL_NAME_MAX];
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_vocab;
    uint32_t n_head_dim;
    uint32_t n_indexer_head_dim;
    uint32_t n_swa;
    uint32_t compress_ratios[DS4_PEER_MAX_LAYERS];
    uint32_t routed_quant_bits;

    uint32_t ctx_size;
    uint32_t raw_window;
    uint32_t raw_cap;
    uint32_t comp_cap;
    uint32_t attn_comp_cache_format;
    uint32_t backend;
    uint32_t supported_payload_versions;
    uint64_t weights_fingerprint;
} ds4_peer_capabilities;

typedef struct {
    char host[DS4_PEER_ENDPOINT_HOST_MAX];
    uint16_t port;
} ds4_peer_endpoint;

typedef struct {
    uint32_t payload_version;
    uint32_t transfer_mode;
    ds4_peer_capabilities peer;
} ds4_peer_handshake_result;

typedef struct ds4_peer_listener ds4_peer_listener;

/* The data plane deliberately passes a staged, seekable file rather than a
 * memory buffer: a full native payload is GiB-scale at production contexts.
 * release owns both source->file and source->opaque.  The mutation field is
 * only for a harness to prove the receiver rejects a byte changed in flight;
 * production services must leave it as DS4_PEER_NO_MUTATION. */
typedef struct {
    FILE *file;
    uint64_t bytes;
    uint64_t test_mutate_wire_byte;
    void *opaque;
    void (*release)(void *opaque, FILE *file);
} ds4_peer_payload_source;

/* The peer returns these counters on every session-open/resume.  They are
 * protocol facts, not owner-side estimates: tests use prefill_tokens to prove
 * that an edit replayed exactly the suffix after the validated common prefix. */
typedef enum {
    DS4_PEER_SESSION_CACHE_NONE = 0,
    DS4_PEER_SESSION_CACHE_NEW = 1,
    DS4_PEER_SESSION_CACHE_LIVE_EXTEND = 2,
    DS4_PEER_SESSION_CACHE_CHECKPOINT = 3,
} ds4_peer_session_cache_source;

/* SESSION_RESUME carries either a complete proposed timeline (for an edit) or
 * only the append suffix.  The latter is valid solely when expected_common is
 * the peer's complete live prefix, and keeps ordinary multi-turn wire traffic
 * proportional to the new turn. */
typedef enum {
    DS4_PEER_SESSION_RESUME_FULL = 0,
    DS4_PEER_SESSION_RESUME_SUFFIX = 1,
} ds4_peer_session_resume_mode;

typedef struct {
    uint64_t session_id;
    uint64_t evals;
    uint32_t cache_source;
    uint32_t cached_tokens;
    uint32_t prefill_tokens;
} ds4_peer_session_reply;

/* A decode-session service retains mutable peer state after the TCP client
 * disconnects.  Each callback must either provide a complete reply (and, for
 * new/resume/eval, current logits) or return a specific error.  The supplied
 * logits buffer belongs to ds4_peer.c and is valid only for the callback.
 * session_sweep is called before each control frame so the model server can
 * enforce its configured idle-TTL eviction policy. */
typedef int (*ds4_peer_session_new_fn)(
        void *opaque, uint64_t session_id, const int *tokens,
        uint32_t n_tokens, FILE *payload, uint64_t payload_bytes,
        ds4_peer_session_reply *reply, float *logits, uint32_t logits_cap,
        uint32_t *out_logits, char *err, size_t errlen);
typedef int (*ds4_peer_session_resume_fn)(
        void *opaque, uint64_t session_id, const int *tokens,
        uint32_t n_tokens, uint32_t expected_common,
        ds4_peer_session_resume_mode mode,
        ds4_peer_session_reply *reply, float *logits, uint32_t logits_cap,
        uint32_t *out_logits, char *err, size_t errlen);
typedef int (*ds4_peer_session_drop_fn)(
        void *opaque, uint64_t session_id, ds4_peer_session_reply *reply,
        char *err, size_t errlen);
typedef int (*ds4_peer_session_eval_fn)(
        void *opaque, uint64_t session_id, int token,
        ds4_peer_session_reply *reply, float *logits, uint32_t logits_cap,
        uint32_t *out_logits, char *err, size_t errlen);
typedef int (*ds4_peer_session_cancel_fn)(
        void *opaque, uint64_t session_id, ds4_peer_session_reply *reply,
        char *err, size_t errlen);
typedef void (*ds4_peer_session_sweep_fn)(void *opaque);

typedef int (*ds4_peer_prefill_fn)(void *opaque, const int *tokens,
                                   uint32_t n_tokens,
                                   ds4_peer_payload_source *out,
                                   char *err, size_t errlen);

typedef struct {
    ds4_peer_prefill_fn prefill;
    ds4_peer_session_new_fn session_new;
    ds4_peer_session_resume_fn session_resume;
    ds4_peer_session_drop_fn session_drop;
    ds4_peer_session_eval_fn session_eval;
    ds4_peer_session_cancel_fn session_cancel;
    ds4_peer_session_sweep_fn session_sweep;
    void *opaque;
} ds4_peer_service;

void ds4_peer_capabilities_init(ds4_peer_capabilities *caps);
int ds4_peer_parse_endpoint(const char *text, ds4_peer_endpoint *out,
                            char *err, size_t errlen);

/* Opens a TCP connection, completes the symmetric HELLO/HELLO_ACK exchange,
 * and leaves the connected descriptor in out_fd on success.  The caller owns
 * the descriptor and must close it. */
int ds4_peer_connect_handshake(const char *endpoint,
                               const ds4_peer_capabilities *local,
                               ds4_peer_handshake_result *result,
                               int *out_fd,
                               char *err, size_t errlen);

/* Bind, accept exactly one connection, complete the handshake, then service
 * PING until the peer closes the socket.  This is the no-model G2 harness API. */
int ds4_peer_serve_once(const char *endpoint,
                        const ds4_peer_capabilities *local,
                        ds4_peer_handshake_result *result,
                        char *err, size_t errlen);

/* Model-free service entry point used by the topology-A listener and G3.
 * It accepts exactly one connection, then services one native-v3 prefill
 * request before the owner closes the connection. */
int ds4_peer_serve_prefill_once(const char *endpoint,
                                const ds4_peer_capabilities *local,
                                const ds4_peer_service *service,
                                ds4_peer_handshake_result *result,
                                char *err, size_t errlen);

/* A normal server listener accepts successive Phase-2 control connections.
 * It owns its listener thread and may be stopped safely during server teardown. */
ds4_peer_listener *ds4_peer_listener_start(const char *endpoint,
                                           const ds4_peer_capabilities *local,
                                           char *err, size_t errlen);
ds4_peer_listener *ds4_peer_listener_start_service(
        const char *endpoint, const ds4_peer_capabilities *local,
        const ds4_peer_service *service, char *err, size_t errlen);
void ds4_peer_listener_stop(ds4_peer_listener *listener);

/* PING is the only post-handshake control frame in Phase 2.  The listener
 * echoes its body verbatim, making the framing path observable without KV. */
int ds4_peer_ping(int fd, const void *body, size_t body_len,
                  char *err, size_t errlen);

/* Request a peer-owned topology-A prefill after a successful native-v3
 * handshake.  On success out_payload is a rewound temporary file owned by
 * the caller and out_sha256 is the digest verified before KV_ACK. */
int ds4_peer_request_prefill(int fd,
                             const ds4_peer_handshake_result *negotiated,
                             const int *tokens, uint32_t n_tokens,
                             FILE **out_payload, uint64_t *out_payload_bytes,
                             uint8_t out_sha256[DS4_PEER_SHA256_BYTES],
                             char *err, size_t errlen);

/* Topology-B decode sessions. SESSION_NEW sends an owner-staged native-v3
 * payload and its token truth, then receives the peer's current logits.
 * SESSION_RESUME sends either the complete new token timeline for an edit, or
 * just an append suffix for the common live-extension case.  The peer must
 * validate the independently calculated boundary and returns measured
 * cached/prefill token counts. */
int ds4_peer_session_new(
        int fd, const ds4_peer_handshake_result *negotiated,
        uint64_t session_id, const int *tokens, uint32_t n_tokens,
        FILE *payload, uint64_t payload_bytes,
        ds4_peer_session_reply *reply, float *logits, uint32_t logits_cap,
        uint32_t *out_logits, char *err, size_t errlen);
int ds4_peer_session_resume(
        int fd, const ds4_peer_handshake_result *negotiated,
        uint64_t session_id, const int *tokens, uint32_t n_tokens,
        uint32_t expected_common, ds4_peer_session_resume_mode mode,
        ds4_peer_session_reply *reply,
        float *logits, uint32_t logits_cap, uint32_t *out_logits,
        char *err, size_t errlen);
int ds4_peer_session_eval(
        int fd, uint64_t session_id, int token,
        ds4_peer_session_reply *reply, float *logits, uint32_t logits_cap,
        uint32_t *out_logits, char *err, size_t errlen);
int ds4_peer_session_cancel(int fd, uint64_t session_id,
                            ds4_peer_session_reply *reply,
                            char *err, size_t errlen);
int ds4_peer_session_drop(int fd, uint64_t session_id,
                          ds4_peer_session_reply *reply,
                          char *err, size_t errlen);

/* The arithmetic contract used by SESSION_RESUME.  It is public so an owner
 * and a harness can independently calculate and assert the same boundary. */
uint32_t ds4_peer_token_common_prefix(const int *a, uint32_t a_count,
                                      const int *b, uint32_t b_count);

/* Small public helper so the G3 harness can independently compare the sent,
 * received, and advertised DS4P SHA-256 values. */
void ds4_peer_sha256(const void *data, size_t len,
                     uint8_t out[DS4_PEER_SHA256_BYTES]);

#endif /* DS4_PEER_H */
