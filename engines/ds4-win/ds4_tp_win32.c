/* Windows stubs for the tensor-parallel transport (ds4_tp.h).
 *
 * The real implementation, ds4_tp.c, is POSIX-only: netdb.h, sys/uio.h,
 * netinet/in.h, netinet/tcp.h, pthread.h and stdatomic.h, plus libibverbs for
 * the RDMA path.  None of that compiles under MSVC, so ds4_tp.c is excluded
 * from windows_build.ps1 and this file supplies the ~42 externals it would
 * otherwise define.  Same arrangement as ds4_distributed.c, which is likewise
 * replaced here by ds4_distributed_win32.c.
 *
 * THESE STUBS FAIL LOUDLY ON PURPOSE.  Tensor parallelism splits one model
 * across two machines that must stay in lockstep; a stub that quietly returned
 * "success" would let the engine believe a peer rank was computing half of
 * every gate when nothing is on the wire at all.  That produces silently wrong
 * logits, which is far worse than not starting.  So:
 *
 *   - ds4_tp_create() never yields a handle; there is no such thing as a live
 *     ds4_tp on this build, and every entry point that takes one is therefore
 *     unreachable in practice.  They are still implemented, and they all
 *     report failure rather than pretending the exchange happened.
 *   - the CLI parser rejects the tensor-parallel flags with a clear message
 *     instead of accepting and ignoring them, so `--tensor-parallel` cannot
 *     look like it took effect.
 *   - a run that never asks for TP is completely unaffected: the parser
 *     returns "not matched", and validation/adoption succeed silently.
 *
 * Porting the real transport is a separate, much larger job (Winsock, an
 * overlapped/IOCP replacement for the readv/writev framing, a pthread and
 * C11-atomics shim, and either dropping RDMA or binding NetworkDirect).
 */

#include "ds4_tp.h"

#if defined(_WIN32)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DS4_TP_WIN32_UNSUPPORTED \
    "tensor parallelism is not supported in the native Windows build"

/* Longer form, printed once directly to stderr on the CLI path so the reason
 * is visible even though the caller prints its own short diagnostic too. */
#define DS4_TP_WIN32_UNSUPPORTED_LONG                                      \
    "ds4: " DS4_TP_WIN32_UNSUPPORTED " (ds4_tp.c is POSIX-only and is not " \
    "compiled here)\n"

static void tp_win32_set_err(char *err, size_t errlen, const char *msg) {
    if (err && errlen) snprintf(err, errlen, "%s", msg);
}

/* Reports the failure through whichever channel the caller offers, and always
 * leaves a trace on stderr for callers that discard `err`. */
static void tp_win32_fail(char *err, size_t errlen, const char *msg) {
    tp_win32_set_err(err, errlen, msg);
    if (!err || !errlen) fprintf(stderr, "ds4: %s\n", msg);
}

/* ------------------------------------------------------------------------
 * Options and CLI.
 * --------------------------------------------------------------------- */

bool ds4_tp_enabled(const ds4_tp_options *opt) {
    /* Matches upstream: reflects what was asked for, not what is possible.
     * Nothing here can make a requested role actually run. */
    return opt && opt->role != DS4_TP_NONE;
}

void ds4_tp_usage(FILE *fp) {
    if (!fp) return;
    fprintf(fp, "Tensor parallelism (--tensor-parallel) is not available in "
                "the native Windows build.\n");
}

/* Every flag ds4_tp.c would consume.  All of them are tensor-parallel only --
 * no other parser in the tree claims them -- so matching one means the user
 * really did ask for TP, and the right answer is a hard error. */
static const char *const tp_win32_flags[] = {
    "--tensor-parallel",
    "--tensor-parallel-token-prefill",
    "--transport",
    "--rdma-device",
    "--rdma-gid-index",
    "--debug-hash",
};

int ds4_tp_parse_cli_arg(
        const char *arg,
        int *index,
        int argc,
        char **argv,
        ds4_tp_options *opt,
        char *err,
        size_t errlen) {
    (void)index;
    (void)argc;
    (void)argv;
    (void)opt;
    if (!arg) return DS4_TP_CLI_NOT_MATCHED;
    for (size_t i = 0; i < sizeof(tp_win32_flags) / sizeof(*tp_win32_flags); i++) {
        if (strcmp(arg, tp_win32_flags[i]) != 0) continue;
        /* Deliberately loud: the caller prints `err` and exits, but print the
         * reason as well so it is never mistaken for a typo in the flag. */
        fputs(DS4_TP_WIN32_UNSUPPORTED_LONG, stderr);
        {
            char msg[192];
            snprintf(msg, sizeof(msg),
                     "%s is a tensor-parallel option, which this build does "
                     "not support", arg);
            tp_win32_set_err(err, errlen, msg);
        }
        /* `*index` is left alone: any value argument stays unconsumed, which
         * is harmless because the caller aborts. */
        return DS4_TP_CLI_ERROR;
    }
    /* Not a TP flag -- ordinary single-GPU runs never notice this file. */
    return DS4_TP_CLI_NOT_MATCHED;
}

/* Returns 1 on success, 0 on error (inverted relative to the CLI parser --
 * this matches ds4_tp.c and ds4_cli.c's `if (!ds4_tp_adopt_...)` check). */
int ds4_tp_adopt_distributed_options(
        ds4_tp_options *tp,
        ds4_distributed_options *dist,
        char *err,
        size_t errlen) {
    if (!tp || !dist || !tp->requested) return 1; /* nothing asked for */
    tp_win32_fail(err, errlen, DS4_TP_WIN32_UNSUPPORTED);
    return 0;
}

/* Returns 1 when the option set is runnable, 0 with `err` filled otherwise. */
int ds4_tp_validate_engine_options(
        const ds4_engine_options *opt,
        char *err,
        size_t errlen) {
    if (!opt) return 1;
    if (ds4_tp_enabled(&opt->tp) || opt->tp.requested ||
        opt->tp.transport != DS4_TP_TRANSPORT_AUTO || opt->tp.rdma_device ||
        opt->tp.rdma_gid_index_set || opt->tp.glm_token_prefill ||
        opt->tp.debug_hash != 0) {
        tp_win32_fail(err, errlen, DS4_TP_WIN32_UNSUPPORTED);
        return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------------
 * Slab layout.  Sizes and offsets describe a slab that is never allocated,
 * because the connection that would use it cannot be created.  Zero is
 * returned rather than a plausible-looking layout so that any caller that
 * reached here despite ds4_tp_create() failing cannot get a usable address.
 * --------------------------------------------------------------------- */

uint64_t ds4_tp_slab_bytes(uint32_t n_layer, uint32_t n_embd) {
    (void)n_layer;
    (void)n_embd;
    return 0;
}

uint64_t ds4_tp_slab_out_offset(const ds4_tp *tp, uint32_t layer, uint32_t gate) {
    (void)tp; (void)layer; (void)gate;
    return 0;
}

uint64_t ds4_tp_slab_in_offset(const ds4_tp *tp, uint32_t layer, uint32_t gate) {
    (void)tp; (void)layer; (void)gate;
    return 0;
}

uint64_t ds4_tp_slab_batch_out_offset(const ds4_tp *tp, uint32_t layer) {
    (void)tp; (void)layer;
    return 0;
}

uint64_t ds4_tp_slab_batch_in_offset(const ds4_tp *tp, uint32_t layer) {
    (void)tp; (void)layer;
    return 0;
}

uint64_t ds4_tp_slab_gpu_flags_offset(const ds4_tp *tp) {
    (void)tp;
    return 0;
}

int ds4_tp_attach_slab(ds4_tp *tp, void *base, char *err, size_t errlen) {
    (void)tp;
    (void)base;
    tp_win32_fail(err, errlen, DS4_TP_WIN32_UNSUPPORTED);
    return 0; /* 0 == failure, per ds4_tp.h */
}

/* ------------------------------------------------------------------------
 * Connection bring-up and handle accessors.
 * --------------------------------------------------------------------- */

int ds4_tp_create(
        ds4_tp **out,
        const ds4_tp_options *opt,
        const ds4_tp_identity *id,
        char *err,
        size_t errlen) {
    (void)opt;
    (void)id;
    if (out) *out = NULL;
    tp_win32_fail(err, errlen, DS4_TP_WIN32_UNSUPPORTED);
    return 0; /* 0 == failure, per ds4_tp.c */
}

void ds4_tp_free(ds4_tp *tp) {
    /* Nothing is ever allocated, and NULL is the only value that can reach
     * here (ds4_cli.c calls this on the error path with a NULL handle). */
    (void)tp;
}

int ds4_tp_rank(const ds4_tp *tp) {
    (void)tp;
    return -1; /* not a valid rank; 0 would read as "leader" */
}

bool ds4_tp_is_rdma(const ds4_tp *tp) {
    (void)tp;
    return false;
}

uint32_t ds4_tp_peer_ctx(const ds4_tp *tp) {
    (void)tp;
    return 0;
}

bool ds4_tp_failed(const ds4_tp *tp) {
    /* NULL matches upstream's answer for "no context".  A non-NULL handle
     * cannot exist on this build, so if one ever shows up it is broken by
     * definition and must be reported as failed. */
    return tp != NULL;
}

void ds4_tp_mark_failed(ds4_tp *tp) {
    (void)tp; /* already permanently failed */
}

/* ------------------------------------------------------------------------
 * Gate exchanges.  Return 0 on failure, per ds4_tp.h; ds4.c turns that into
 * ds4_tp_mark_failed() and aborts the decode rather than using stale halves.
 * --------------------------------------------------------------------- */

int ds4_tp_gate_exchange(ds4_tp *tp, uint32_t layer, uint32_t gate, uint64_t seq) {
    (void)tp; (void)layer; (void)gate; (void)seq;
    return 0;
}

int ds4_tp_batch_gate_exchange(ds4_tp *tp, uint32_t layer, uint32_t rows,
                               uint64_t seq) {
    (void)tp; (void)layer; (void)rows; (void)seq;
    return 0;
}

int ds4_tp_big_gate_exchange(ds4_tp *tp, uint32_t layer, uint64_t seq,
                             const void *out, void *in, uint64_t bytes) {
    (void)tp; (void)layer; (void)seq; (void)out; (void)in; (void)bytes;
    return 0;
}

/* ------------------------------------------------------------------------
 * Lockstep mirroring.  All of these return 0 == transport failure; none of
 * them may return success, because success would tell the leader that a
 * worker rank had accepted the command.
 * --------------------------------------------------------------------- */

int ds4_tp_send_session_create(ds4_tp *tp, uint64_t session_id, int ctx_size) {
    (void)tp; (void)session_id; (void)ctx_size;
    return 0;
}

int ds4_tp_send_session_destroy(ds4_tp *tp, uint64_t session_id) {
    (void)tp; (void)session_id;
    return 0;
}

int ds4_tp_send_sync(ds4_tp *tp, uint64_t session_id,
                     const int *tokens, uint32_t n_tokens) {
    (void)tp; (void)session_id; (void)tokens; (void)n_tokens;
    return 0;
}

int ds4_tp_send_eval(ds4_tp *tp, uint64_t session_id,
                     uint64_t seq, int token) {
    (void)tp; (void)session_id; (void)seq; (void)token;
    return 0;
}

int ds4_tp_send_rewind(ds4_tp *tp, uint64_t session_id, int pos) {
    (void)tp; (void)session_id; (void)pos;
    return 0;
}

int ds4_tp_send_invalidate(ds4_tp *tp, uint64_t session_id) {
    (void)tp; (void)session_id;
    return 0;
}

int ds4_tp_send_eval_batch(ds4_tp *tp, const ds4_tp_batch_item *items,
                           uint32_t count) {
    (void)tp; (void)items; (void)count;
    return 0;
}

int ds4_tp_send_mixed_batch(ds4_tp *tp, uint64_t prefill_session_id,
                            const int *prompt, uint32_t prompt_count,
                            const ds4_tp_batch_item *items,
                            uint32_t count) {
    (void)tp; (void)prefill_session_id; (void)prompt; (void)prompt_count;
    (void)items; (void)count;
    return 0;
}

int ds4_tp_send_command_ack(ds4_tp *tp, uint64_t session_id, int status) {
    (void)tp; (void)session_id; (void)status;
    return 0;
}

int ds4_tp_wait_command_ack(ds4_tp *tp, uint64_t session_id,
                            const char *operation, char *err, size_t errlen) {
    (void)tp;
    (void)session_id;
    (void)operation;
    tp_win32_set_err(err, errlen, DS4_TP_WIN32_UNSUPPORTED);
    return 0;
}

int ds4_tp_send_stop(ds4_tp *tp) {
    (void)tp;
    return 0;
}

/* ------------------------------------------------------------------------
 * Worker side.
 * --------------------------------------------------------------------- */

int ds4_tp_recv_command(
        ds4_tp *tp,
        ds4_tp_command *command,
        char *err,
        size_t errlen) {
    (void)tp;
    if (command) {
        memset(command, 0, sizeof(*command));
        command->type = DS4_TP_FRAME_ERROR;
    }
    tp_win32_set_err(err, errlen, DS4_TP_WIN32_UNSUPPORTED);
    return 0;
}

void ds4_tp_command_free(ds4_tp_command *command) {
    /* Mirrors upstream, including tolerating NULL.  The frees are kept so
     * that a command filled in by some other code path is still released. */
    if (!command) return;
    free(command->tokens);
    free(command->items);
    memset(command, 0, sizeof(*command));
    command->type = DS4_TP_FRAME_ERROR;
}

/* 0 == transport failure (never -1, which would mean a real hash mismatch). */
int ds4_tp_hash_check(ds4_tp *tp, uint64_t seq, uint64_t hash, char *err,
                      size_t errlen) {
    (void)tp;
    (void)seq;
    (void)hash;
    tp_win32_set_err(err, errlen, DS4_TP_WIN32_UNSUPPORTED);
    return 0;
}

/* ------------------------------------------------------------------------
 * Vocab-split logits and speculative verify mirroring.
 * --------------------------------------------------------------------- */

int ds4_tp_send_logits_half(ds4_tp *tp, const float *half, uint32_t count) {
    (void)tp; (void)half; (void)count;
    return 0;
}

int ds4_tp_recv_logits_half(ds4_tp *tp, float *half, uint32_t count) {
    /* The caller must not treat `half` as populated; returning 0 makes it
     * abandon the token instead of sampling over an untouched buffer. */
    (void)tp; (void)half; (void)count;
    return 0;
}

int ds4_tp_send_verify(ds4_tp *tp, uint64_t session_id,
                       const int *drafts, uint32_t n) {
    (void)tp; (void)session_id; (void)drafts; (void)n;
    return 0;
}

int ds4_tp_send_verify_commit(ds4_tp *tp, int32_t full_accept, int32_t replay_n) {
    (void)tp; (void)full_accept; (void)replay_n;
    return 0;
}

int ds4_tp_recv_verify_commit(ds4_tp *tp, int32_t *full_accept, int32_t *replay_n) {
    (void)tp;
    /* Never report an acceptance the leader did not make. */
    if (full_accept) *full_accept = 0;
    if (replay_n) *replay_n = 0;
    return 0;
}

/* Process exit code: non-zero == the worker did not run. */
int ds4_tp_worker_run(ds4_engine *engine, const ds4_tp_options *opt) {
    (void)engine;
    (void)opt;
    fputs(DS4_TP_WIN32_UNSUPPORTED_LONG, stderr);
    return 1;
}

/* New upstream commands retain the explicit unsupported transport result. */
void ds4_tp_detach_slab(ds4_tp *tp) { (void)tp; }

int ds4_tp_send_sync_multimodal(ds4_tp *tp, uint64_t session_id,
                                const int *tokens, uint32_t n_tokens,
                                const ds4_vision_span *images, uint32_t image_count) {
    (void)tp; (void)session_id; (void)tokens; (void)n_tokens;
    (void)images; (void)image_count;
    return 0;
}

int ds4_tp_send_glm_mtp(ds4_tp *tp, uint64_t session_id,
                       uint64_t seq, int token, int limit) {
    (void)tp; (void)session_id; (void)seq; (void)token; (void)limit;
    return 0;
}

int ds4_tp_wait_command_status(ds4_tp *tp, uint64_t session_id, int *status,
                               const char *operation, char *err, size_t errlen) {
    (void)tp; (void)session_id; (void)operation;
    if (status) *status = 1;
    tp_win32_set_err(err, errlen, DS4_TP_WIN32_UNSUPPORTED);
    return 0;
}

int ds4_tp_sync_checkpoint(ds4_tp *tp, uint32_t point, int current, int total,
                            bool requested, bool *cancelled) {
    (void)tp; (void)point; (void)current; (void)total; (void)requested;
    if (cancelled) *cancelled = true;
    return 0;
}

#endif /* _WIN32 */
