#include "ds4_distributed.h"

#if defined(_WIN32)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void ds4_dist_set_err(char *err, size_t errlen, const char *msg) {
    if (err && errlen) snprintf(err, errlen, "%s", msg);
}

bool ds4_dist_enabled(const ds4_dist_options *opt) {
    return opt && opt->role != DS4_DISTRIBUTED_NONE;
}

ds4_dist_options *ds4_dist_options_create(void) {
    return (ds4_dist_options *)calloc(1, sizeof(ds4_dist_options));
}

void ds4_dist_options_free(ds4_dist_options *opt) {
    free(opt);
}

void ds4_dist_usage(FILE *fp) {
    fprintf(fp, "\nDistributed inference is not available in the native Windows build.\n");
}

ds4_dist_cli_parse_result ds4_dist_parse_cli_arg(
        const char *arg,
        int *index,
        int argc,
        char **argv,
        ds4_dist_options *opt,
        char *err,
        size_t errlen) {
    (void)index;
    (void)argc;
    (void)argv;
    (void)opt;
    if (!arg) return DS4_DIST_CLI_NOT_MATCHED;
    if (strcmp(arg, "--role") == 0 ||
        strcmp(arg, "--listen") == 0 ||
        strcmp(arg, "--connect") == 0 ||
        strcmp(arg, "--layers") == 0 ||
        strcmp(arg, "--dist-debug") == 0 ||
        strcmp(arg, "--dist-replay-check") == 0 ||
        strcmp(arg, "--dist-prefill-chunk") == 0 ||
        strcmp(arg, "--dist-prefill-window") == 0 ||
        strcmp(arg, "--dist-activation-bits") == 0) {
        ds4_dist_set_err(err, errlen,
                         "distributed inference is not available in the native Windows build");
        return DS4_DIST_CLI_ERROR;
    }
    return DS4_DIST_CLI_NOT_MATCHED;
}

int ds4_dist_prepare_engine_options(
        const ds4_dist_options *opt,
        ds4_engine_options *engine,
        char *err,
        size_t errlen) {
    if (ds4_dist_enabled(opt)) {
        ds4_dist_set_err(err, errlen,
                         "distributed inference is not available in the native Windows build");
        return 1;
    }
    if (engine) memset(&engine->distributed, 0, sizeof(engine->distributed));
    return 0;
}

int ds4_dist_session_create(
        ds4_dist_session **out,
        ds4_engine *engine,
        const ds4_dist_options *opt,
        ds4_session *owner,
        int ctx_size,
        char *err,
        size_t errlen) {
    (void)engine;
    (void)opt;
    (void)owner;
    (void)ctx_size;
    if (out) *out = NULL;
    ds4_dist_set_err(err, errlen,
                     "distributed inference is not available in the native Windows build");
    return 1;
}

void ds4_dist_session_free(ds4_dist_session *d) {
    (void)d;
}

int ds4_dist_session_route_ready(ds4_dist_session *d, char *err, size_t errlen) {
    (void)d;
    ds4_dist_set_err(err, errlen,
                     "distributed inference is not available in the native Windows build");
    return -1;
}

int ds4_dist_session_sync(
        ds4_dist_session *d,
        ds4_session *owner,
        const ds4_tokens *checkpoint,
        const ds4_tokens *prompt,
        float *logits,
        char *err,
        size_t errlen) {
    (void)d;
    (void)owner;
    (void)checkpoint;
    (void)prompt;
    (void)logits;
    ds4_dist_set_err(err, errlen,
                     "distributed inference is not available in the native Windows build");
    return 1;
}

int ds4_dist_session_eval(
        ds4_dist_session *d,
        ds4_session *owner,
        const ds4_tokens *checkpoint,
        int token,
        float *logits,
        char *err,
        size_t errlen) {
    (void)d;
    (void)owner;
    (void)checkpoint;
    (void)token;
    (void)logits;
    ds4_dist_set_err(err, errlen,
                     "distributed inference is not available in the native Windows build");
    return 1;
}

int ds4_dist_session_save_payload(
        ds4_dist_session *d,
        ds4_session *owner,
        FILE *fp,
        char *err,
        size_t errlen) {
    (void)d;
    (void)owner;
    (void)fp;
    ds4_dist_set_err(err, errlen,
                     "distributed inference is not available in the native Windows build");
    return 1;
}

int ds4_dist_session_load_payload(
        ds4_dist_session *d,
        ds4_session *owner,
        FILE *fp,
        uint64_t payload_bytes,
        char *err,
        size_t errlen) {
    (void)d;
    (void)owner;
    (void)fp;
    (void)payload_bytes;
    ds4_dist_set_err(err, errlen,
                     "distributed inference is not available in the native Windows build");
    return 1;
}

int ds4_dist_run(ds4_engine *engine,
                 const ds4_dist_options *opt,
                 const ds4_dist_generation_options *gen) {
    (void)engine;
    (void)opt;
    (void)gen;
    fprintf(stderr,
            "ds4: distributed inference is not available in the native Windows build\n");
    return 1;
}

#endif /* _WIN32 */
