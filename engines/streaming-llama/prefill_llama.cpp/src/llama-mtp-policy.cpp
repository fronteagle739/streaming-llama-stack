#include "llama-mtp-policy.h"

#include "llama.h"

#include <algorithm>

bool llama_mtp_mode_enabled(int32_t mtp_mode) {
    return mtp_mode == LLAMA_MTP_MODE_VALIDATE ||
           mtp_mode == LLAMA_MTP_MODE_DRAFT ||
           mtp_mode == LLAMA_MTP_MODE_SPECULATIVE;
}

bool llama_mtp_graph_uses_nextn(int32_t mtp_mode, int32_t graph_role) {
    if (!llama_mtp_mode_enabled(mtp_mode)) {
        return false;
    }

    switch (graph_role) {
        case LLAMA_MTP_GRAPH_ROLE_DRAFT:
            return mtp_mode == LLAMA_MTP_MODE_DRAFT ||
                   mtp_mode == LLAMA_MTP_MODE_SPECULATIVE;
        case LLAMA_MTP_GRAPH_ROLE_VALIDATE:
            return true;
        case LLAMA_MTP_GRAPH_ROLE_TARGET:
        default:
            return false;
    }
}

int32_t llama_mtp_default_graph_role(int32_t mtp_mode) {
    switch (mtp_mode) {
        case LLAMA_MTP_MODE_DRAFT:
            return LLAMA_MTP_GRAPH_ROLE_DRAFT;
        case LLAMA_MTP_MODE_VALIDATE:
            return LLAMA_MTP_GRAPH_ROLE_VALIDATE;
        case LLAMA_MTP_MODE_SPECULATIVE:
        case LLAMA_MTP_MODE_OFF:
        default:
            return LLAMA_MTP_GRAPH_ROLE_TARGET;
    }
}

int32_t llama_mtp_clamp_max_draft(int32_t requested_max_draft, int32_t nextn_predict_layers) {
    if (nextn_predict_layers <= 0) {
        return 0;
    }

    const int32_t requested = requested_max_draft > 0 ?
        requested_max_draft :
        LLAMA_MTP_DEFAULT_MAX_DRAFT;
    return std::max<int32_t>(1, std::min<int32_t>(requested, LLAMA_MTP_DEFAULT_MAX_DRAFT));
}

bool llama_mtp_should_use_native_drafter(int32_t mtp_mode, int32_t nextn_predict_layers, bool has_external_draft) {
    return mtp_mode == LLAMA_MTP_MODE_SPECULATIVE &&
           nextn_predict_layers > 0 &&
           !has_external_draft;
}

bool llama_mtp_should_use_native_draft_context(int32_t mtp_mode, int32_t nextn_predict_layers, bool has_external_draft, bool prefill_streaming) {
    return llama_mtp_should_use_native_drafter(mtp_mode, nextn_predict_layers, has_external_draft) &&
           !prefill_streaming;
}

bool llama_mtp_should_use_stream_local_drafter(int32_t mtp_mode, int32_t nextn_predict_layers, bool has_external_draft, bool prefill_streaming) {
    return llama_mtp_should_use_native_drafter(mtp_mode, nextn_predict_layers, has_external_draft) &&
           prefill_streaming;
}

bool llama_mtp_should_capture_target_hidden(int32_t mtp_mode, int32_t nextn_predict_layers, int32_t n_tokens, int32_t n_outputs) {
    return mtp_mode == LLAMA_MTP_MODE_SPECULATIVE &&
           nextn_predict_layers > 0 &&
           n_tokens == 1 &&
           n_outputs == 1;
}

llama_mtp_decode_plan llama_mtp_build_decode_plan(int32_t mtp_mode, int32_t nextn_predict_layers, int32_t requested_max_draft) {
    llama_mtp_decode_plan plan;
    plan.native_requested = llama_mtp_mode_enabled(mtp_mode);
    plan.has_nextn = nextn_predict_layers > 0;

    if (!plan.native_requested) {
        plan.reason = "MTP mode off";
        return plan;
    }

    if (!plan.has_nextn) {
        plan.reason = "model has no NextN layers";
        return plan;
    }

    plan.max_draft = llama_mtp_clamp_max_draft(requested_max_draft, nextn_predict_layers);

    switch (mtp_mode) {
        case LLAMA_MTP_MODE_SPECULATIVE:
            plan.speculative = true;
            plan.requires_target_graph = true;
            plan.requires_draft_graph = true;
            plan.reason = "native MTP speculative decode";
            break;
        case LLAMA_MTP_MODE_DRAFT:
            plan.requires_draft_graph = true;
            plan.reason = "native MTP draft graph diagnostic";
            break;
        case LLAMA_MTP_MODE_VALIDATE:
            plan.requires_validate_graph = true;
            plan.reason = "native MTP validate graph diagnostic";
            break;
        default:
            plan.max_draft = 0;
            plan.reason = "MTP mode off";
            break;
    }

    return plan;
}

int32_t llama_mtp_base_layer_count(int32_t n_layer, int32_t nextn_predict_layers) {
    if (n_layer <= 0) {
        return 0;
    }
    if (nextn_predict_layers <= 0) {
        return n_layer;
    }
    return std::max<int32_t>(0, n_layer - nextn_predict_layers);
}

int32_t llama_mtp_effective_layer_count(int32_t n_layer, int32_t nextn_predict_layers, int32_t mtp_mode, int32_t graph_role) {
    return llama_mtp_graph_uses_nextn(mtp_mode, graph_role) ?
        std::max<int32_t>(0, n_layer) :
        llama_mtp_base_layer_count(n_layer, nextn_predict_layers);
}

llama_mtp_layer_range llama_mtp_resolve_layer_range(int32_t n_layer, int32_t nextn_predict_layers) {
    llama_mtp_layer_range range;
    range.base_start = 0;
    range.base_end   = llama_mtp_base_layer_count(n_layer, nextn_predict_layers);
    range.mtp_start  = range.base_end;
    range.mtp_end    = std::max<int32_t>(range.mtp_start, n_layer);
    return range;
}

int32_t llama_mtp_clamp_layer_end(int32_t requested_end, int32_t n_layer, int32_t nextn_predict_layers, int32_t mtp_mode) {
    const int32_t max_end = llama_mtp_mode_enabled(mtp_mode) ?
        std::max<int32_t>(0, n_layer) :
        llama_mtp_base_layer_count(n_layer, nextn_predict_layers);
    if (requested_end < 0) {
        return max_end;
    }
    return std::max<int32_t>(0, std::min<int32_t>(requested_end, max_end));
}

int32_t llama_mtp_clamp_layer_end_for_role(int32_t requested_end, int32_t n_layer, int32_t nextn_predict_layers, int32_t mtp_mode, int32_t graph_role) {
    const int32_t max_end = llama_mtp_effective_layer_count(
        n_layer, nextn_predict_layers, mtp_mode, graph_role);
    if (requested_end < 0) {
        return max_end;
    }
    return std::max<int32_t>(0, std::min<int32_t>(requested_end, max_end));
}

bool llama_mtp_is_nextn_layer(int32_t layer, int32_t n_layer, int32_t nextn_predict_layers) {
    const llama_mtp_layer_range range = llama_mtp_resolve_layer_range(n_layer, nextn_predict_layers);
    return layer >= range.mtp_start && layer < range.mtp_end;
}
