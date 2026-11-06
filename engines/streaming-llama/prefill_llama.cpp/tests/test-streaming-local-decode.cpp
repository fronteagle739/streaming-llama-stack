// TDD: streaming local decode selector tests.
// Pure logic: no model, GPU, DirectStorage, or network required.

#include "llama-streaming-local-decode.h"
#include "llama.h"

#include <cassert>
#include <cstdio>

static llama_streaming_local_decode_inputs base_inputs() {
    llama_streaming_local_decode_inputs in = {};
    in.batch_tokens = 1;
    in.decode_mode = LLAMA_PREFILL_DECODE_MODE_AUTO;
    return in;
}

static void test_non_wf_small_batch_uses_normal_decode() {
    printf("  non_wf_small_batch_uses_normal_decode ... ");
    auto in = base_inputs();

    const auto d = llama_streaming_local_decode_select(in);

    assert(d.action == LLAMA_STREAMING_LOCAL_ACTION_NORMAL_DECODE);
    assert(!d.clear_wf_prefill_done);
    assert(!d.bypass_wf_skip);
    printf("OK\n");
}

static void test_large_batch_uses_streaming_prefill() {
    printf("  large_batch_uses_streaming_prefill ... ");
    auto in = base_inputs();
    in.should_stream_batch = true;

    const auto d = llama_streaming_local_decode_select(in);

    assert(d.action == LLAMA_STREAMING_LOCAL_ACTION_STREAMING_PREFILL);
    assert(!d.clear_wf_prefill_done);
    printf("OK\n");
}

static void test_wf_sentinel_small_batch_is_skipped_without_mode() {
    printf("  wf_sentinel_small_batch_is_skipped_without_mode ... ");
    auto in = base_inputs();
    in.wf_active = true;

    const auto d = llama_streaming_local_decode_select(in);

    assert(d.action == LLAMA_STREAMING_LOCAL_ACTION_SKIP_DECODE);
    assert(!d.bypass_wf_skip);
    printf("OK\n");
}

static void test_initial_wf_stream_local_streams_small_batch() {
    printf("  initial_wf_stream_local_streams_small_batch ... ");
    auto in = base_inputs();
    in.wf_active = true;
    in.decode_mode = LLAMA_PREFILL_DECODE_MODE_STREAM_LOCAL;

    const auto d = llama_streaming_local_decode_select(in);

    assert(d.action == LLAMA_STREAMING_LOCAL_ACTION_STREAMING_LOCAL_DECODE);
    assert(!d.clear_wf_prefill_done);
    assert(!d.bypass_wf_skip);
    printf("OK\n");
}

static void test_stream_local_speculative_mtp_exposes_target_and_draft_plan() {
    printf("  stream_local_speculative_mtp_exposes_target_and_draft_plan ... ");
    auto in = base_inputs();
    in.wf_active = true;
    in.decode_mode = LLAMA_PREFILL_DECODE_MODE_STREAM_LOCAL;
    in.mtp_mode = LLAMA_MTP_MODE_SPECULATIVE;
    in.nextn_predict_layers = 1;
    in.mtp_max_draft = 16;

    const auto d = llama_streaming_local_decode_select(in);

    assert(d.action == LLAMA_STREAMING_LOCAL_ACTION_STREAMING_LOCAL_DECODE);
    assert(d.mtp_plan.speculative);
    assert(d.mtp_plan.requires_target_graph);
    assert(d.mtp_plan.requires_draft_graph);
    assert(d.mtp_plan.target_graph_role == LLAMA_MTP_GRAPH_ROLE_TARGET);
    assert(d.mtp_plan.draft_graph_role == LLAMA_MTP_GRAPH_ROLE_DRAFT);
    assert(d.mtp_plan.max_draft == LLAMA_MTP_DEFAULT_MAX_DRAFT);
    printf("OK\n");
}

static void test_stream_local_speculative_mtp_honors_smaller_runtime_draft() {
    printf("  stream_local_speculative_mtp_honors_smaller_runtime_draft ... ");
    auto in = base_inputs();
    in.wf_active = true;
    in.decode_mode = LLAMA_PREFILL_DECODE_MODE_STREAM_LOCAL;
    in.mtp_mode = LLAMA_MTP_MODE_SPECULATIVE;
    in.nextn_predict_layers = 1;
    in.mtp_max_draft = 4;

    const auto d = llama_streaming_local_decode_select(in);

    assert(d.action == LLAMA_STREAMING_LOCAL_ACTION_STREAMING_LOCAL_DECODE);
    assert(d.mtp_plan.speculative);
    assert(d.mtp_plan.max_draft == 4);
    printf("OK\n");
}

static void test_stream_local_speculative_mtp_requires_nextn_layers() {
    printf("  stream_local_speculative_mtp_requires_nextn_layers ... ");
    auto in = base_inputs();
    in.wf_active = true;
    in.decode_mode = LLAMA_PREFILL_DECODE_MODE_STREAM_LOCAL;
    in.mtp_mode = LLAMA_MTP_MODE_SPECULATIVE;
    in.nextn_predict_layers = 0;
    in.mtp_max_draft = 7;

    const auto d = llama_streaming_local_decode_select(in);

    assert(d.action == LLAMA_STREAMING_LOCAL_ACTION_STREAMING_LOCAL_DECODE);
    assert(!d.mtp_plan.speculative);
    assert(!d.mtp_plan.requires_target_graph);
    assert(!d.mtp_plan.requires_draft_graph);
    assert(d.mtp_plan.max_draft == 0);
    printf("OK\n");
}

static void test_post_prefill_handoff_bypasses_wf_skip() {
    printf("  post_prefill_handoff_bypasses_wf_skip ... ");
    auto in = base_inputs();
    in.wf_active = true;
    in.wf_prefill_done = true;
    in.handoff_pending = true;

    const auto d = llama_streaming_local_decode_select(in);

    assert(d.action == LLAMA_STREAMING_LOCAL_ACTION_HANDOFF_DECODE);
    assert(d.bypass_wf_skip);
    assert(!d.clear_wf_prefill_done);
    printf("OK\n");
}

static void test_post_prefill_stream_local_streams_small_decode_batch() {
    printf("  post_prefill_stream_local_streams_small_decode_batch ... ");
    auto in = base_inputs();
    in.wf_active = true;
    in.wf_prefill_done = true;
    in.decode_mode = LLAMA_PREFILL_DECODE_MODE_STREAM_LOCAL;

    const auto d = llama_streaming_local_decode_select(in);

    assert(d.action == LLAMA_STREAMING_LOCAL_ACTION_STREAMING_LOCAL_DECODE);
    assert(d.clear_wf_prefill_done);
    assert(!d.bypass_wf_skip);
    printf("OK\n");
}

static void test_post_prefill_stream_local_overrides_stale_handoff() {
    printf("  post_prefill_stream_local_overrides_stale_handoff ... ");
    auto in = base_inputs();
    in.wf_active = true;
    in.wf_prefill_done = true;
    in.handoff_pending = true;
    in.decode_mode = LLAMA_PREFILL_DECODE_MODE_STREAM_LOCAL;

    const auto d = llama_streaming_local_decode_select(in);

    assert(d.action == LLAMA_STREAMING_LOCAL_ACTION_STREAMING_LOCAL_DECODE);
    assert(d.clear_wf_prefill_done);
    assert(!d.bypass_wf_skip);
    printf("OK\n");
}

static void test_post_prefill_streaming_continuation_takes_priority() {
    printf("  post_prefill_streaming_continuation_takes_priority ... ");
    auto in = base_inputs();
    in.wf_active = true;
    in.wf_prefill_done = true;
    in.should_stream_batch = true;
    in.decode_mode = LLAMA_PREFILL_DECODE_MODE_STREAM_LOCAL;
    in.handoff_pending = true;

    const auto d = llama_streaming_local_decode_select(in);

    assert(d.action == LLAMA_STREAMING_LOCAL_ACTION_STREAMING_PREFILL);
    assert(d.clear_wf_prefill_done);
    assert(!d.bypass_wf_skip);
    printf("OK\n");
}

static void test_inside_streaming_decode_never_recurses() {
    printf("  inside_streaming_decode_never_recurses ... ");
    auto in = base_inputs();
    in.wf_active = true;
    in.wf_prefill_done = true;
    in.inside_streaming_decode = true;
    in.should_stream_batch = true;
    in.decode_mode = LLAMA_PREFILL_DECODE_MODE_STREAM_LOCAL;

    const auto d = llama_streaming_local_decode_select(in);

    assert(d.action == LLAMA_STREAMING_LOCAL_ACTION_NORMAL_DECODE);
    assert(!d.clear_wf_prefill_done);
    printf("OK\n");
}

int main() {
    printf("test-streaming-local-decode: selector tests\n");
    test_non_wf_small_batch_uses_normal_decode();
    test_large_batch_uses_streaming_prefill();
    test_wf_sentinel_small_batch_is_skipped_without_mode();
    test_initial_wf_stream_local_streams_small_batch();
    test_stream_local_speculative_mtp_exposes_target_and_draft_plan();
    test_stream_local_speculative_mtp_honors_smaller_runtime_draft();
    test_stream_local_speculative_mtp_requires_nextn_layers();
    test_post_prefill_handoff_bypasses_wf_skip();
    test_post_prefill_stream_local_streams_small_decode_batch();
    test_post_prefill_stream_local_overrides_stale_handoff();
    test_post_prefill_streaming_continuation_takes_priority();
    test_inside_streaming_decode_never_recurses();

    printf("\ntest-streaming-local-decode: all tests OK\n");
    return 0;
}
