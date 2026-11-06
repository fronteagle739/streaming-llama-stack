#include "llama-mtp-policy.h"

#include "llama.h"

#include <cassert>

static void test_glm52_one_nextn_layer() {
    const auto range = llama_mtp_resolve_layer_range(79, 1);
    assert(range.base_start == 0);
    assert(range.base_end == 78);
    assert(range.mtp_start == 78);
    assert(range.mtp_end == 79);

    assert(!llama_mtp_is_nextn_layer(77, 79, 1));
    assert(llama_mtp_is_nextn_layer(78, 79, 1));
}

static void test_off_mode_preserves_legacy_base_layer_graph() {
    assert(llama_mtp_clamp_layer_end(-1, 79, 1, LLAMA_MTP_MODE_OFF) == 78);
    assert(llama_mtp_clamp_layer_end(79, 79, 1, LLAMA_MTP_MODE_OFF) == 78);
    assert(llama_mtp_clamp_layer_end(12, 79, 1, LLAMA_MTP_MODE_OFF) == 12);
}

static void test_enabled_modes_allow_nextn_layer() {
    assert(llama_mtp_clamp_layer_end(-1, 79, 1, LLAMA_MTP_MODE_VALIDATE) == 79);
    assert(llama_mtp_clamp_layer_end(-1, 79, 1, LLAMA_MTP_MODE_DRAFT) == 79);
    assert(llama_mtp_clamp_layer_end(-1, 79, 1, LLAMA_MTP_MODE_SPECULATIVE) == 79);
    assert(llama_mtp_mode_enabled(LLAMA_MTP_MODE_VALIDATE));
    assert(llama_mtp_mode_enabled(LLAMA_MTP_MODE_DRAFT));
    assert(llama_mtp_mode_enabled(LLAMA_MTP_MODE_SPECULATIVE));
    assert(!llama_mtp_mode_enabled(LLAMA_MTP_MODE_OFF));
}

static void test_target_graph_stays_base_head_under_speculative_mode() {
    assert(!llama_mtp_graph_uses_nextn(LLAMA_MTP_MODE_SPECULATIVE, LLAMA_MTP_GRAPH_ROLE_TARGET));
    assert(llama_mtp_effective_layer_count(
        79, 1, LLAMA_MTP_MODE_SPECULATIVE, LLAMA_MTP_GRAPH_ROLE_TARGET) == 78);
    assert(llama_mtp_clamp_layer_end_for_role(
        -1, 79, 1, LLAMA_MTP_MODE_SPECULATIVE, LLAMA_MTP_GRAPH_ROLE_TARGET) == 78);
    assert(llama_mtp_clamp_layer_end_for_role(
        79, 79, 1, LLAMA_MTP_MODE_SPECULATIVE, LLAMA_MTP_GRAPH_ROLE_TARGET) == 78);
}

static void test_draft_graph_uses_nextn_under_speculative_mode() {
    assert(llama_mtp_graph_uses_nextn(LLAMA_MTP_MODE_SPECULATIVE, LLAMA_MTP_GRAPH_ROLE_DRAFT));
    assert(llama_mtp_effective_layer_count(
        79, 1, LLAMA_MTP_MODE_SPECULATIVE, LLAMA_MTP_GRAPH_ROLE_DRAFT) == 79);
    assert(llama_mtp_clamp_layer_end_for_role(
        -1, 79, 1, LLAMA_MTP_MODE_SPECULATIVE, LLAMA_MTP_GRAPH_ROLE_DRAFT) == 79);
}

static void test_validate_role_can_build_full_nextn_graph() {
    assert(llama_mtp_graph_uses_nextn(LLAMA_MTP_MODE_VALIDATE, LLAMA_MTP_GRAPH_ROLE_VALIDATE));
    assert(llama_mtp_clamp_layer_end_for_role(
        -1, 79, 1, LLAMA_MTP_MODE_VALIDATE, LLAMA_MTP_GRAPH_ROLE_VALIDATE) == 79);
    assert(!llama_mtp_graph_uses_nextn(LLAMA_MTP_MODE_VALIDATE, LLAMA_MTP_GRAPH_ROLE_TARGET));
    assert(llama_mtp_clamp_layer_end_for_role(
        -1, 79, 1, LLAMA_MTP_MODE_VALIDATE, LLAMA_MTP_GRAPH_ROLE_TARGET) == 78);
}

static void test_default_graph_roles_keep_speculative_target_graph_clean() {
    assert(llama_mtp_default_graph_role(LLAMA_MTP_MODE_OFF) == LLAMA_MTP_GRAPH_ROLE_TARGET);
    assert(llama_mtp_default_graph_role(LLAMA_MTP_MODE_SPECULATIVE) == LLAMA_MTP_GRAPH_ROLE_TARGET);
    assert(llama_mtp_default_graph_role(LLAMA_MTP_MODE_DRAFT) == LLAMA_MTP_GRAPH_ROLE_DRAFT);
    assert(llama_mtp_default_graph_role(LLAMA_MTP_MODE_VALIDATE) == LLAMA_MTP_GRAPH_ROLE_VALIDATE);
}

static void test_native_drafter_selection_requires_speculative_nextn_without_external_draft() {
    assert(llama_mtp_should_use_native_drafter(LLAMA_MTP_MODE_SPECULATIVE, 1, false));
    assert(!llama_mtp_should_use_native_drafter(LLAMA_MTP_MODE_SPECULATIVE, 1, true));
    assert(!llama_mtp_should_use_native_drafter(LLAMA_MTP_MODE_SPECULATIVE, 0, false));
    assert(!llama_mtp_should_use_native_drafter(LLAMA_MTP_MODE_DRAFT, 1, false));
    assert(!llama_mtp_should_use_native_drafter(LLAMA_MTP_MODE_VALIDATE, 1, false));
    assert(!llama_mtp_should_use_native_drafter(LLAMA_MTP_MODE_OFF, 1, false));
}

static void test_streaming_native_drafter_uses_stream_local_path() {
    assert(llama_mtp_should_use_native_draft_context(LLAMA_MTP_MODE_SPECULATIVE, 1, false, false));
    assert(!llama_mtp_should_use_stream_local_drafter(LLAMA_MTP_MODE_SPECULATIVE, 1, false, false));

    assert(!llama_mtp_should_use_native_draft_context(LLAMA_MTP_MODE_SPECULATIVE, 1, false, true));
    assert(llama_mtp_should_use_stream_local_drafter(LLAMA_MTP_MODE_SPECULATIVE, 1, false, true));

    assert(!llama_mtp_should_use_native_draft_context(LLAMA_MTP_MODE_SPECULATIVE, 1, true, false));
    assert(!llama_mtp_should_use_stream_local_drafter(LLAMA_MTP_MODE_SPECULATIVE, 1, true, true));
}

static void test_target_hidden_capture_only_for_single_token_speculative_decode() {
    assert(llama_mtp_should_capture_target_hidden(LLAMA_MTP_MODE_SPECULATIVE, 1, 1, 1));
    assert(!llama_mtp_should_capture_target_hidden(LLAMA_MTP_MODE_SPECULATIVE, 1, 2, 2));
    assert(!llama_mtp_should_capture_target_hidden(LLAMA_MTP_MODE_SPECULATIVE, 1, 1, 0));
    assert(!llama_mtp_should_capture_target_hidden(LLAMA_MTP_MODE_SPECULATIVE, 0, 1, 1));
    assert(!llama_mtp_should_capture_target_hidden(LLAMA_MTP_MODE_DRAFT, 1, 1, 1));
    assert(!llama_mtp_should_capture_target_hidden(LLAMA_MTP_MODE_VALIDATE, 1, 1, 1));
    assert(!llama_mtp_should_capture_target_hidden(LLAMA_MTP_MODE_OFF, 1, 1, 1));
}

static void test_glm52_speculative_plan_caps_to_seven_steps() {
    const auto plan = llama_mtp_build_decode_plan(LLAMA_MTP_MODE_SPECULATIVE, 1, 16);
    assert(plan.native_requested);
    assert(plan.has_nextn);
    assert(plan.speculative);
    assert(plan.requires_target_graph);
    assert(plan.requires_draft_graph);
    assert(!plan.requires_validate_graph);
    assert(plan.target_graph_role == LLAMA_MTP_GRAPH_ROLE_TARGET);
    assert(plan.draft_graph_role == LLAMA_MTP_GRAPH_ROLE_DRAFT);
    assert(plan.max_draft == LLAMA_MTP_DEFAULT_MAX_DRAFT);
}

static void test_glm52_speculative_plan_allows_smaller_runtime_draft() {
    const auto plan = llama_mtp_build_decode_plan(LLAMA_MTP_MODE_SPECULATIVE, 1, 4);
    assert(plan.speculative);
    assert(plan.max_draft == 4);
}

static void test_diagnostic_plans_are_not_speculative() {
    const auto draft = llama_mtp_build_decode_plan(LLAMA_MTP_MODE_DRAFT, 1, 7);
    assert(!draft.speculative);
    assert(draft.requires_draft_graph);
    assert(!draft.requires_target_graph);

    const auto validate = llama_mtp_build_decode_plan(LLAMA_MTP_MODE_VALIDATE, 1, 7);
    assert(!validate.speculative);
    assert(validate.requires_validate_graph);
    assert(!validate.requires_draft_graph);
}

static void test_non_mtp_models_are_unchanged() {
    const auto range = llama_mtp_resolve_layer_range(61, 0);
    assert(range.base_end == 61);
    assert(range.mtp_start == 61);
    assert(range.mtp_end == 61);
    assert(llama_mtp_clamp_layer_end(-1, 61, 0, LLAMA_MTP_MODE_OFF) == 61);
    assert(llama_mtp_clamp_layer_end(-1, 61, 0, LLAMA_MTP_MODE_DRAFT) == 61);
    assert(!llama_mtp_is_nextn_layer(60, 61, 0));

    const auto plan = llama_mtp_build_decode_plan(LLAMA_MTP_MODE_SPECULATIVE, 0, 7);
    assert(plan.native_requested);
    assert(!plan.has_nextn);
    assert(!plan.speculative);
    assert(plan.max_draft == 0);
}

int main() {
    test_glm52_one_nextn_layer();
    test_off_mode_preserves_legacy_base_layer_graph();
    test_enabled_modes_allow_nextn_layer();
    test_target_graph_stays_base_head_under_speculative_mode();
    test_draft_graph_uses_nextn_under_speculative_mode();
    test_validate_role_can_build_full_nextn_graph();
    test_default_graph_roles_keep_speculative_target_graph_clean();
    test_native_drafter_selection_requires_speculative_nextn_without_external_draft();
    test_streaming_native_drafter_uses_stream_local_path();
    test_target_hidden_capture_only_for_single_token_speculative_decode();
    test_glm52_speculative_plan_caps_to_seven_steps();
    test_glm52_speculative_plan_allows_smaller_runtime_draft();
    test_diagnostic_plans_are_not_speculative();
    test_non_mtp_models_are_unchanged();
    return 0;
}
