#include "llama-streaming-local-decode.h"
#include "llama.h"

llama_streaming_local_decode_decision llama_streaming_local_decode_select(
    const llama_streaming_local_decode_inputs & in) {

    llama_streaming_local_decode_decision d = {};
    d.mtp_plan = llama_mtp_build_decode_plan(
        in.mtp_mode, in.nextn_predict_layers, in.mtp_max_draft);

    if (in.inside_streaming_decode) {
        d.action = LLAMA_STREAMING_LOCAL_ACTION_NORMAL_DECODE;
        d.reason = "already inside streaming decode";
        return d;
    }

    if (in.wf_active && in.wf_prefill_done) {
        if (in.should_stream_batch) {
            d.action = LLAMA_STREAMING_LOCAL_ACTION_STREAMING_PREFILL;
            d.clear_wf_prefill_done = true;
            d.reason = "additional streaming prefill batch";
            return d;
        }

        if (in.decode_mode == LLAMA_PREFILL_DECODE_MODE_STREAM_LOCAL) {
            d.action = LLAMA_STREAMING_LOCAL_ACTION_STREAMING_LOCAL_DECODE;
            d.clear_wf_prefill_done = true;
            d.reason = "stream_local requested after WF prefill";
            return d;
        }

        if (in.handoff_pending) {
            d.action = LLAMA_STREAMING_LOCAL_ACTION_HANDOFF_DECODE;
            d.bypass_wf_skip = true;
            d.reason = "handoff pending after WF prefill";
            return d;
        }

        d.action = LLAMA_STREAMING_LOCAL_ACTION_SKIP_DECODE;
        d.reason = "WF prefill done with no handoff or stream_local mode";
        return d;
    }

    if (in.should_stream_batch) {
        d.action = LLAMA_STREAMING_LOCAL_ACTION_STREAMING_PREFILL;
        d.reason = "batch meets streaming threshold";
        return d;
    }

    if (in.wf_active &&
        in.decode_mode == LLAMA_PREFILL_DECODE_MODE_STREAM_LOCAL &&
        in.batch_tokens > 0) {
        d.action = LLAMA_STREAMING_LOCAL_ACTION_STREAMING_LOCAL_DECODE;
        d.reason = "stream_local requested with WF sentinel weights";
        return d;
    }

    if (in.wf_active) {
        d.action = LLAMA_STREAMING_LOCAL_ACTION_SKIP_DECODE;
        d.reason = "WF sentinel weights active for non-streaming batch";
        return d;
    }

    d.action = LLAMA_STREAMING_LOCAL_ACTION_NORMAL_DECODE;
    d.reason = "normal decode";
    return d;
}
