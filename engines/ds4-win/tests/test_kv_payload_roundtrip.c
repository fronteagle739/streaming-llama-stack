/*
 * Model-backed payload gate for the DSV4 session serializer.
 *
 * Run the complete Phase 0 + Phase 1 matrix with:
 *   DS4_TEST_MODEL=/path/to/model.gguf ./test-kv-payload-roundtrip --all-formats
 *
 * Each case drives the engine API directly: sync -> stage -> new session ->
 * load -> greedy argmax/eval.  The canonical v2 path remains the disk-cache
 * compatibility check; native v3 is exercised independently for every cache
 * row format.
 */

#include "ds4.h"
#include "ds4_gpu.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32)
#include <sys/types.h>
#endif

#define TEST_CONTEXT_DEFAULT 512
#define TEST_DECODE_STEPS 16
#define TEST_IO_CHUNK (64u * 1024u)
#define TEST_ERR_CAP 512

typedef struct {
    const char *name;
    const char *env_name;
    uint32_t format;
} payload_format_case;

typedef struct {
    bool equal;
    uint64_t first_difference;
    uint8_t left_byte;
    uint8_t right_byte;
    uint64_t left_bytes;
    uint64_t right_bytes;
} payload_compare;

static const payload_format_case g_formats[] = {
    { "fp16", "DS4_CUDA_ATTN_COMP_CACHE_F16", DS4_GPU_COMP_KV_FORMAT_F16 },
    { "q8",   "DS4_CUDA_ATTN_COMP_CACHE_Q8",  DS4_GPU_COMP_KV_FORMAT_Q8 },
    { "fp32", "DS4_CUDA_ATTN_COMP_CACHE_F32", DS4_GPU_COMP_KV_FORMAT_F32 },
};

static int cache_bytes_from_env(const char *name, uint64_t default_gib,
                                uint64_t *bytes_out);

static void failf(const char *label, const char *detail) {
    fprintf(stderr, "FAIL: %s: %s\n", label, detail);
}

static bool file_seek(FILE *fp, uint64_t offset) {
#if defined(_WIN32)
    return _fseeki64(fp, (int64_t)offset, SEEK_SET) == 0;
#else
    return fseeko(fp, (off_t)offset, SEEK_SET) == 0;
#endif
}

static void put_u32_le(uint8_t out[4], uint32_t value) {
    out[0] = (uint8_t)value;
    out[1] = (uint8_t)(value >> 8);
    out[2] = (uint8_t)(value >> 16);
    out[3] = (uint8_t)(value >> 24);
}

static uint32_t get_u32_le(const uint8_t in[4]) {
    return (uint32_t)in[0] |
           ((uint32_t)in[1] << 8) |
           ((uint32_t)in[2] << 16) |
           ((uint32_t)in[3] << 24);
}

static int set_env_value(const char *name, const char *value) {
#if defined(_WIN32)
    return _putenv_s(name, value ? value : "") == 0 ? 0 : 1;
#else
    return value ? setenv(name, value, 1) : unsetenv(name);
#endif
}

static int select_cache_format(const payload_format_case *format) {
    const char *const names[] = {
        "DS4_CUDA_ATTN_COMP_CACHE",
        "DS4_ATTN_COMP_CACHE",
        "DS4_CUDA_ATTN_COMP_CACHE_F16",
        "DS4_CUDA_ATTN_COMP_CACHE_Q8",
        "DS4_CUDA_ATTN_COMP_CACHE_F32",
    };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        if (set_env_value(names[i], NULL) != 0) return 1;
    }
    return set_env_value(format->env_name, "1");
}

static FILE *copy_payload_prefix(const ds4_session_payload_file *payload,
                                 uint64_t bytes) {
    if (!payload || !payload->path || bytes > payload->bytes) return NULL;
    FILE *src = fopen(payload->path, "rb");
    if (!src) return NULL;
    FILE *dst = tmpfile();
    if (!dst) {
        fclose(src);
        return NULL;
    }

    uint8_t buffer[TEST_IO_CHUNK];
    uint64_t remaining = bytes;
    int failed = 0;
    while (remaining != 0) {
        const size_t n = remaining > sizeof(buffer)
            ? sizeof(buffer) : (size_t)remaining;
        if (fread(buffer, 1, n, src) != n || fwrite(buffer, 1, n, dst) != n) {
            failed = 1;
            break;
        }
        remaining -= n;
    }
    if (fclose(src) != 0 || fflush(dst) != 0 || !file_seek(dst, 0)) failed = 1;
    if (failed) {
        fclose(dst);
        return NULL;
    }
    return dst;
}

static int staged_read_u32(const ds4_session_payload_file *payload,
                           uint32_t field, uint32_t *value) {
    if (!payload || !value || field > UINT64_MAX / sizeof(uint32_t)) return 1;
    FILE *fp = fopen(payload->path, "rb");
    uint8_t bytes[4];
    int rc = 1;
    if (fp && file_seek(fp, (uint64_t)field * sizeof(uint32_t)) &&
        fread(bytes, 1, sizeof(bytes), fp) == sizeof(bytes)) {
        *value = get_u32_le(bytes);
        rc = 0;
    }
    if (fp) fclose(fp);
    return rc;
}

static int mutate_flip_byte(FILE *fp, uint64_t offset) {
    if (!fp || !file_seek(fp, offset)) return 1;
    const int original = fgetc(fp);
    if (original == EOF || !file_seek(fp, offset)) return 1;
    const uint8_t flipped = (uint8_t)original ^ UINT8_C(0x01);
    if (fwrite(&flipped, 1, 1, fp) != 1 || fflush(fp) != 0) return 1;
    return file_seek(fp, 0) ? 0 : 1;
}

static int mutate_u32(FILE *fp, uint64_t offset, uint32_t value) {
    uint8_t bytes[4];
    put_u32_le(bytes, value);
    if (!fp || !file_seek(fp, offset)) return 1;
    if (fwrite(bytes, 1, sizeof(bytes), fp) != sizeof(bytes) || fflush(fp) != 0) {
        return 1;
    }
    return file_seek(fp, 0) ? 0 : 1;
}

static int compare_staged_payloads(const ds4_session_payload_file *left,
                                   const ds4_session_payload_file *right,
                                   payload_compare *result) {
    if (!left || !right || !result) return 1;
    memset(result, 0, sizeof(*result));
    result->left_bytes = left->bytes;
    result->right_bytes = right->bytes;

    FILE *lf = fopen(left->path, "rb");
    FILE *rf = fopen(right->path, "rb");
    if (!lf || !rf) {
        if (lf) fclose(lf);
        if (rf) fclose(rf);
        return 1;
    }

    uint8_t lbuf[TEST_IO_CHUNK];
    uint8_t rbuf[TEST_IO_CHUNK];
    const uint64_t common = left->bytes < right->bytes ? left->bytes : right->bytes;
    uint64_t offset = 0;
    int rc = 0;
    while (offset < common) {
        const size_t n = common - offset > sizeof(lbuf)
            ? sizeof(lbuf) : (size_t)(common - offset);
        if (fread(lbuf, 1, n, lf) != n || fread(rbuf, 1, n, rf) != n) {
            rc = 1;
            break;
        }
        for (size_t i = 0; i < n; i++) {
            if (lbuf[i] != rbuf[i]) {
                result->first_difference = offset + i;
                result->left_byte = lbuf[i];
                result->right_byte = rbuf[i];
                goto compared;
            }
        }
        offset += n;
    }
    if (rc == 0 && left->bytes == right->bytes) {
        result->equal = true;
    } else if (rc == 0) {
        result->first_difference = common;
    }

compared:
    if (fclose(lf) != 0 || fclose(rf) != 0) rc = 1;
    return rc;
}

static int load_staged_payload(ds4_session *session,
                               const ds4_session_payload_file *payload,
                               char *err, size_t errlen) {
    FILE *fp = fopen(payload->path, "rb");
    if (!fp) {
        snprintf(err, errlen, "failed to open staged payload");
        return 1;
    }
    const int rc = ds4_session_load_payload(session, fp, payload->bytes, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        snprintf(err, errlen, "failed to close staged payload");
        return 1;
    }
    return rc;
}

static int expect_load_refusal(ds4_engine *engine, int context,
                               FILE *payload, uint64_t payload_bytes,
                               const char *expected_error,
                               const char *label) {
    ds4_session *target = NULL;
    char err[TEST_ERR_CAP] = {0};
    if (!payload || ds4_session_create(&target, engine, context) != 0) {
        failf(label, "could not create mutation target session");
        return 1;
    }
    if (!file_seek(payload, 0)) {
        ds4_session_free(target);
        failf(label, "could not rewind mutation payload");
        return 1;
    }
    const int rc = ds4_session_load_payload(target, payload, payload_bytes,
                                            err, sizeof(err));
    ds4_session_free(target);
    if (rc == 0) {
        failf(label, "mutated payload unexpectedly loaded");
        return 1;
    }
    if (!strstr(err, expected_error)) {
        fprintf(stderr, "FAIL: %s: expected error containing '%s', got '%s'\n",
                label, expected_error, err);
        return 1;
    }
    fprintf(stderr, "%s=REFUSED error=%s\n", label, err);
    return 0;
}

static int assert_noop_sync_preserves_logits(ds4_session *session,
                                             const ds4_tokens *prompt,
                                             int vocab) {
    float *before = malloc((size_t)vocab * sizeof(*before));
    float *after = malloc((size_t)vocab * sizeof(*after));
    char err[TEST_ERR_CAP] = {0};
    int rc = 1;
    if (!before || !after) {
        failf("no-op sync", "could not allocate logits buffers");
        goto done;
    }
    const int common = ds4_session_common_prefix(session, prompt);
    if (common != prompt->len) {
        fprintf(stderr, "FAIL: no-op sync: common=%d prompt_len=%d\n", common, prompt->len);
        goto done;
    }
    if (ds4_session_copy_logits(session, before, vocab) != vocab) {
        failf("no-op sync", "could not copy logits before sync");
        goto done;
    }
    if (ds4_session_sync(session, prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL: no-op sync: ds4_session_sync failed: %s\n", err);
        goto done;
    }
    if (ds4_session_copy_logits(session, after, vocab) != vocab) {
        failf("no-op sync", "could not copy logits after sync");
        goto done;
    }
    if (memcmp(before, after, (size_t)vocab * sizeof(*before)) != 0) {
        failf("no-op sync", "restored logits changed after common == len sync");
        goto done;
    }
    fprintf(stderr, "NO_OP_SYNC_LOGITS=PASS common=%d\n", common);
    rc = 0;

done:
    free(after);
    free(before);
    return rc;
}

static int assert_resume_sequence(ds4_session *continuous,
                                  ds4_session *restored,
                                  int steps) {
    char err[TEST_ERR_CAP] = {0};
    for (int step = 0; step < steps; step++) {
        const int expected = ds4_session_argmax(continuous);
        const int actual = ds4_session_argmax(restored);
        if (expected < 0 || actual < 0 || expected != actual) {
            fprintf(stderr,
                    "FAIL: resume sequence: step=%d continuous=%d restored=%d\n",
                    step, expected, actual);
            return 1;
        }
        if (ds4_session_eval(continuous, expected, err, sizeof(err)) != 0) {
            fprintf(stderr, "FAIL: resume sequence: continuous eval step=%d: %s\n",
                    step, err);
            return 1;
        }
        if (ds4_session_eval(restored, actual, err, sizeof(err)) != 0) {
            fprintf(stderr, "FAIL: resume sequence: restored eval step=%d: %s\n",
                    step, err);
            return 1;
        }
    }
    fprintf(stderr, "RESUME_ARGMAX_EVAL=PASS steps=%d\n", steps);
    return 0;
}

static int build_prompt(ds4_engine *engine, int context, ds4_tokens *out) {
    const char prefix[] = "Explain why a serialized accelerator cache needs exact framing.";
    const char word[] = " checkpoint";
    const size_t prefix_bytes = sizeof(prefix) - 1u;
    const size_t word_bytes = sizeof(word) - 1u;
    memset(out, 0, sizeof(*out));

    for (int repeats = 132; repeats <= 220; repeats++) {
        const size_t bytes = prefix_bytes + (size_t)repeats * word_bytes + 1u;
        char *text = malloc(bytes);
        if (!text) return 1;
        memcpy(text, prefix, prefix_bytes);
        char *cursor = text + prefix_bytes;
        for (int i = 0; i < repeats; i++) {
            memcpy(cursor, word, word_bytes);
            cursor += word_bytes;
        }
        *cursor = '\0';
        ds4_encode_chat_prompt(engine, NULL, text, DS4_THINK_NONE, out);
        free(text);
        if (out->len > 128 && out->len + TEST_DECODE_STEPS < context &&
            (out->len % 128) != 0) {
            return 0;
        }
        ds4_tokens_free(out);
    }
    failf("prompt", "could not construct a non-boundary prompt inside the test context");
    return 1;
}

static int run_v2_mutation_twins(ds4_engine *engine, int context,
                                 const ds4_session_payload_file *payload,
                                 const payload_format_case *format) {
    FILE *flipped = copy_payload_prefix(payload, payload->bytes);
    if (!flipped) {
        failf("G0 byte-flip twin", "could not copy staged payload");
        return 1;
    }
    int rc = 0;
    if (mutate_flip_byte(flipped, 0) != 0 ||
        expect_load_refusal(engine, context, flipped, payload->bytes,
                            "unsupported session payload version",
                            "G0_MUTATION_BYTE_FLIP") != 0) {
        rc = 1;
    }
    fclose(flipped);
    if (rc != 0) return 1;

    if (payload->bytes < 2) {
        failf("G0 truncate twin", "staged payload is too short to truncate");
        return 1;
    }
    FILE *truncated = copy_payload_prefix(payload, payload->bytes - 1u);
    if (!truncated) {
        failf("G0 truncate twin", "could not create truncated payload");
        return 1;
    }
    char label[96];
    snprintf(label, sizeof(label), "G0_MUTATION_TRUNCATE_%s", format->name);
    rc = expect_load_refusal(engine, context, truncated, payload->bytes - 1u,
                             "truncated session payload", label);
    fclose(truncated);
    return rc;
}

static int run_v3_format_mutation(ds4_engine *engine, int context,
                                  const ds4_session_payload_file *payload,
                                  const payload_format_case *format) {
    const uint32_t wrong_format = format->format == DS4_GPU_COMP_KV_FORMAT_F32
        ? DS4_GPU_COMP_KV_FORMAT_F16 : DS4_GPU_COMP_KV_FORMAT_F32;
    FILE *misdeclared = copy_payload_prefix(payload, payload->bytes);
    if (!misdeclared) {
        failf("G1 format twin", "could not copy native payload");
        return 1;
    }
    const uint64_t format_offset = 13u * sizeof(uint32_t);
    int rc = 0;
    if (mutate_u32(misdeclared, format_offset, wrong_format) != 0 ||
        expect_load_refusal(engine, context, misdeclared, payload->bytes,
                            "native session payload compressed-cache format does not match local format",
                            "G1_MUTATION_MISDECLARED_FORMAT") != 0) {
        rc = 1;
    }
    fclose(misdeclared);
    return rc;
}

static int run_roundtrip(ds4_engine *engine, int context,
                         const ds4_tokens *prompt,
                         const payload_format_case *format,
                         bool native_comp_rows) {
    const char *kind = native_comp_rows ? "native-v3" : "canonical-v2";
    ds4_session *continuous = NULL;
    ds4_session *restored = NULL;
    ds4_session_payload_file staged = {0};
    ds4_session_payload_file restaged = {0};
    char err[TEST_ERR_CAP] = {0};
    int rc = 1;

    if (ds4_session_create(&continuous, engine, context) != 0) {
        failf(kind, "could not create continuous session");
        goto done;
    }
    if (ds4_session_sync(continuous, prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL: %s: ds4_session_sync: %s\n", kind, err);
        goto done;
    }

    const uint64_t expected_bytes = native_comp_rows
        ? ds4_session_native_payload_bytes(continuous)
        : ds4_session_payload_bytes(continuous);
    const int stage_rc = native_comp_rows
        ? ds4_session_stage_native_payload(continuous, &staged, err, sizeof(err))
        : ds4_session_stage_payload(continuous, &staged, err, sizeof(err));
    if (stage_rc != 0) {
        fprintf(stderr, "FAIL: %s: payload staging: %s\n", kind, err);
        goto done;
    }
    if (expected_bytes == 0 || staged.bytes != expected_bytes) {
        fprintf(stderr,
                "FAIL: %s: staged bytes=%" PRIu64 " expected=%" PRIu64 "\n",
                kind, staged.bytes, expected_bytes);
        goto done;
    }

    if (native_comp_rows) {
        uint32_t version = 0;
        uint32_t declared_format = 0;
        uint32_t flags = 0;
        if (staged_read_u32(&staged, 1, &version) != 0 ||
            staged_read_u32(&staged, 13, &declared_format) != 0 ||
            staged_read_u32(&staged, 14, &flags) != 0 ||
            version != DS4_SESSION_PAYLOAD_VERSION_V3 ||
            declared_format != format->format ||
            flags != DS4_SESSION_PAYLOAD_FLAG_NATIVE_COMP_ROWS) {
            failf(kind, "native header is not the expected v3 format declaration");
            goto done;
        }
        if (run_v3_format_mutation(engine, context, &staged, format) != 0) goto done;
    } else if (run_v2_mutation_twins(engine, context, &staged, format) != 0) {
        goto done;
    }

    if (ds4_session_create(&restored, engine, context) != 0) {
        failf(kind, "could not create restored session");
        goto done;
    }
    if (load_staged_payload(restored, &staged, err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL: %s: ds4_session_load_payload: %s\n", kind, err);
        goto done;
    }

    const int restage_rc = native_comp_rows
        ? ds4_session_stage_native_payload(restored, &restaged, err, sizeof(err))
        : ds4_session_stage_payload(restored, &restaged, err, sizeof(err));
    if (restage_rc != 0) {
        fprintf(stderr, "FAIL: %s: restored payload staging: %s\n", kind, err);
        goto done;
    }

    payload_compare comparison;
    if (compare_staged_payloads(&staged, &restaged, &comparison) != 0) {
        failf(kind, "could not byte-compare staged payloads");
        goto done;
    }
    if (comparison.equal) {
        fprintf(stderr,
                "%s_RESTAGE format=%s result=IDENTICAL bytes=%" PRIu64 "\n",
                native_comp_rows ? "G1" : "CANONICAL",
                format->name, comparison.left_bytes);
    } else {
        fprintf(stderr,
                "%s_RESTAGE format=%s result=DIFFERENT left_bytes=%" PRIu64
                " right_bytes=%" PRIu64 " first_offset=%" PRIu64
                " left=0x%02x right=0x%02x\n",
                native_comp_rows ? "G1" : "CANONICAL",
                format->name, comparison.left_bytes, comparison.right_bytes,
                comparison.first_difference, comparison.left_byte, comparison.right_byte);
    }
    if (native_comp_rows && !comparison.equal) {
        failf("G1 native round trip", "native payload bytes differed after load and re-save");
        goto done;
    }

    if (assert_noop_sync_preserves_logits(restored, prompt,
                                          ds4_engine_vocab_size(engine)) != 0) {
        goto done;
    }
    if (assert_resume_sequence(continuous, restored, TEST_DECODE_STEPS) != 0) {
        goto done;
    }
    fprintf(stderr,
            "%s=PASS format=%s prompt_tokens=%d payload_bytes=%" PRIu64 "\n",
            native_comp_rows ? "G1" : "G0", format->name, prompt->len, staged.bytes);
    rc = 0;

done:
    ds4_session_payload_file_free(&restaged);
    ds4_session_payload_file_free(&staged);
    ds4_session_free(restored);
    ds4_session_free(continuous);
    return rc;
}

static int run_format_case(const payload_format_case *format, int context,
                           bool phase0, bool phase1) {
    if (select_cache_format(format) != 0) {
        failf("format selection", "could not set compressed-cache environment");
        return 1;
    }
    fprintf(stderr, "PAYLOAD_FORMAT=%s %s=1\n", format->name, format->env_name);

    const char *model_path = getenv("DS4_TEST_MODEL");
    if (!model_path || !model_path[0]) {
        failf("payload roundtrip", "DS4_TEST_MODEL is not set");
        return 1;
    }
    uint64_t ssd_cache_bytes = 0;
    uint64_t ram_cache_bytes = 0;
    /* Keep correctness runs below the server preset: the 90 GB Flash model
     * needs a 1.75 GiB decode arena in addition to the selected-expert cache. */
    if (cache_bytes_from_env("DS4_PAYLOAD_TEST_SSD_CACHE_GIB", 16u,
                             &ssd_cache_bytes) != 0 ||
        cache_bytes_from_env("DS4_PAYLOAD_TEST_RAM_CACHE_GIB", 4u,
                             &ram_cache_bytes) != 0) {
        return 1;
    }
    fprintf(stderr,
            "PAYLOAD_STREAMING ssd_cache_gib=%.0f ram_cache_gib=%.0f\n",
            (double)ssd_cache_bytes / 1073741824.0,
            (double)ram_cache_bytes / 1073741824.0);
    ds4_engine_options options = {
        .model_path = model_path,
#if defined(__APPLE__)
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .n_threads = 1,
        .context_size = context,
        .ssd_streaming = true,
        .ssd_streaming_cache_bytes = ssd_cache_bytes,
        .ram_streaming_cache_bytes = ram_cache_bytes,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &options) != 0 || !engine) {
        failf("payload roundtrip", "could not open test engine");
        return 1;
    }

    ds4_tokens prompt = {0};
    int rc = build_prompt(engine, context, &prompt);
    if (rc == 0 && (phase0 || phase1)) {
        /* A v2 load is retained in Phase 1 so the compatibility promise is a
         * behavioural gate rather than a source-level claim. */
        rc = run_roundtrip(engine, context, &prompt, format, false);
    }
    if (rc == 0 && phase1) {
        rc = run_roundtrip(engine, context, &prompt, format, true);
    }
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return rc;
}

static const payload_format_case *find_format(const char *name) {
    for (size_t i = 0; i < sizeof(g_formats) / sizeof(g_formats[0]); i++) {
        if (strcmp(name, g_formats[i].name) == 0) return &g_formats[i];
    }
    return NULL;
}

static int test_context_from_env(void) {
    const char *value = getenv("DS4_PAYLOAD_TEST_CONTEXT");
    if (!value || !value[0]) return TEST_CONTEXT_DEFAULT;
    char *end = NULL;
    const long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 384 || parsed > 65536) {
        fprintf(stderr,
                "FAIL: DS4_PAYLOAD_TEST_CONTEXT must be an integer from 384 to 65536\n");
        return 0;
    }
    return (int)parsed;
}

static int cache_bytes_from_env(const char *name, uint64_t default_gib,
                                uint64_t *bytes_out) {
    const char *value = getenv(name);
    uint64_t gib = default_gib;
    if (value && value[0]) {
        char *end = NULL;
        const unsigned long long parsed = strtoull(value, &end, 10);
        if (end == value || *end != '\0' || parsed == 0 || parsed > 64u) {
            fprintf(stderr, "FAIL: %s must be an integer number of GiB from 1 to 64\n", name);
            return 1;
        }
        gib = (uint64_t)parsed;
    }
    *bytes_out = gib * UINT64_C(1024) * UINT64_C(1024) * UINT64_C(1024);
    return 0;
}

int main(int argc, char **argv) {
    bool all_formats = false;
    bool phase0 = true;
    bool phase1 = true;
    bool requested_phase0 = false;
    bool requested_phase1 = false;
    const payload_format_case *selected = &g_formats[0];

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--all-formats") == 0) {
            all_formats = true;
        } else if (strcmp(argv[i], "--phase0") == 0) {
            requested_phase0 = true;
        } else if (strcmp(argv[i], "--phase1") == 0) {
            requested_phase1 = true;
        } else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            selected = find_format(argv[++i]);
            if (!selected) {
                fprintf(stderr, "FAIL: unknown payload format '%s'\n", argv[i]);
                return 1;
            }
        } else {
            fprintf(stderr,
                    "usage: %s [--format fp16|q8|fp32] [--all-formats] [--phase0|--phase1]\n",
                    argv[0]);
            return 1;
        }
    }

    if (requested_phase0 || requested_phase1) {
        phase0 = requested_phase0;
        phase1 = requested_phase1;
    }

    const int context = test_context_from_env();
    if (context == 0) return 1;
    if (all_formats) {
        for (size_t i = 0; i < sizeof(g_formats) / sizeof(g_formats[0]); i++) {
            if (run_format_case(&g_formats[i], context, phase0, phase1) != 0) return 1;
        }
    } else if (run_format_case(selected, context, phase0, phase1) != 0) {
        return 1;
    }

    fprintf(stderr, "KV_PAYLOAD_ROUNDTRIP=PASS\n");
    return 0;
}
