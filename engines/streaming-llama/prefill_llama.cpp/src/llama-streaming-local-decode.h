#pragma once

#include "llama-mtp-policy.h"

#include <cstdint>

enum llama_streaming_local_decode_action {
    LLAMA_STREAMING_LOCAL_ACTION_NORMAL_DECODE = 0,
    LLAMA_STREAMING_LOCAL_ACTION_STREAMING_PREFILL,
    LLAMA_STREAMING_LOCAL_ACTION_STREAMING_LOCAL_DECODE,
    LLAMA_STREAMING_LOCAL_ACTION_HANDOFF_DECODE,
    LLAMA_STREAMING_LOCAL_ACTION_SKIP_DECODE,
};

struct llama_streaming_local_decode_inputs {
    bool wf_active               = false;
    bool wf_prefill_done         = false;
    bool inside_streaming_decode = false;
    bool should_stream_batch     = false;
    bool handoff_pending         = false;
    int32_t decode_mode          = 0;
    int32_t batch_tokens         = 0;
    int32_t mtp_mode             = 0;
    int32_t nextn_predict_layers = 0;
    int32_t mtp_max_draft        = 0;
};

struct llama_streaming_local_decode_decision {
    llama_streaming_local_decode_action action = LLAMA_STREAMING_LOCAL_ACTION_NORMAL_DECODE;
    bool clear_wf_prefill_done = false;
    bool bypass_wf_skip        = false;
    llama_mtp_decode_plan mtp_plan = {};
    const char * reason        = "normal decode";
};

llama_streaming_local_decode_decision llama_streaming_local_decode_select(
    const llama_streaming_local_decode_inputs & inputs);
