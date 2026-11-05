#ifndef DS4_NATIVE_SESSION_H
#define DS4_NATIVE_SESSION_H
#include "ds4.h"
#include "dsv41-native-c-layout.h"
#include "qfn-native-layout.h"

/* A synchronous lease over the existing complete mode3 frontier. The caller
 * holds the session's generation owner throughout the callback. Views and
 * callbacks must not escape, re-enter or mutate the session. No tensor type
 * crosses this API. read_span returns one on success and caps reads at64KiB. */
typedef struct {
    uint32_t context,position,capture_end,capture_length,span_count;
    const int32_t *tokens;
    const float *logits;
    const dsv41_c_span *spans;
    int (*read_span)(void*,uint32_t,uint64_t,void*,size_t);
    void *state;
} ds4_native_view;
typedef int (*ds4_native_visit_fn)(void*,const ds4_native_view*,char*,size_t);
typedef int (*ds4_native_read_fn)(void*,void*,size_t);
typedef int (*ds4_native_write_fn)(void*,const void*,size_t);
typedef struct ds4_qfn_anchor ds4_qfn_anchor;
int ds4_qfn_anchor_save(ds4_session*,ds4_qfn_anchor**,uint64_t,uint64_t*,char*,size_t);
int ds4_qfn_anchor_check(ds4_session*,const ds4_qfn_anchor*,uint64_t*,char*,size_t);
int ds4_qfn_anchor_restore(ds4_session*,const ds4_qfn_anchor*,char*,size_t);
void ds4_qfn_anchor_free(ds4_qfn_anchor*);
uint64_t ds4_qfn_anchor_test_bytes(const ds4_qfn_anchor*);
int ds4_qfn_anchor_test_write(const ds4_qfn_anchor*,ds4_native_write_fn,void*);
/* QFN adapter over the same bounded stream callbacks. Layout and MTP frontier
 * are family-specific; the DSV APIs and owner hashes below are unchanged. */
int ds4_qfn_session_pair(ds4_session*,char*,size_t);
int ds4_qfn_session_shape(ds4_session*,qfn_native_shape*,uint32_t *mtp,char*,size_t);
int ds4_qfn_session_delta_write(ds4_session*,uint32_t,uint32_t,uint64_t,
    ds4_native_write_fn,void*,uint64_t*,char*,size_t);
int ds4_qfn_session_delta_apply(ds4_session*,uint32_t,uint32_t,uint32_t,uint32_t,
    uint64_t,uint64_t,ds4_native_read_fn,void*,char*,size_t);

/* Engine convention: zero means success. Unsupported backends fail; these
 * APIs never rebuild tokens, run prefill or use a second cache. */
int ds4_session_native_visit(ds4_session*,ds4_native_visit_fn,void*,char*,size_t);

/* Apply only the planned ranges from a forward stream. Both verifiers are
 * mandatory: base before mutation, target against actual unpublished owner
 * bytes before commit. The adapter allocates at most4MiB tokens +64KiB page;
 * the caller separately budgets incoming IDs/logits and protocol metadata.
 * Any failure after mutation leaves the checkpoint invalid. */
int ds4_session_native_apply_delta(ds4_session*,uint32_t base,uint32_t target,
    const int32_t *new_tokens,const float *logits,ds4_native_read_fn,void *stream,
    ds4_native_visit_fn verify_base,void *base_identity,
    ds4_native_visit_fn verify_target,void *target_identity,char*,size_t);

/* Restore a saved earlier prefix without copying append-only cache rows or
 * evaluating any tokens. Mutable rings/tails arrive in native span order. */
int ds4_session_native_rewind(ds4_session*,uint32_t target,const float*logits,
    ds4_native_read_fn,void *stream,ds4_native_visit_fn verify_base,void *base_identity,
    ds4_native_visit_fn verify_target,void *target_identity,char*,size_t);
#endif
