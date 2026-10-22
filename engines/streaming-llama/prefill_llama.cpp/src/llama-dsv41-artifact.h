#pragma once
#include "llama-dsv41-delta.h"
#include "llama-kv-artifact.h"

namespace llama_dsv41 {
using artifact_read = std::function<bool(uint64_t,void*,size_t,std::string&)>;

// Reuses the existing artifact v2.1 DELTA header and native payload header.
// Body: native target header, NEW token IDs, full logits, make_delta_plan
// slices in their existing order. No full-context duplicate or new transport.
// Full model/session authentication and per-owner/token/logit SHA checks are
// mandatory in the caller. This adapter only binds geometry and routing IDs.
class delta_artifact_sender {
public:
    bool init(const wire_frontier&base,const wire_frontier&target,
        uint64_t session,uint64_t model,wire_payload_reader&source,std::string&error);
    uint64_t bytes()const{return ready?total:0;}
    bool read(uint64_t offset,void*dst,size_t count,std::string&error);
private:
    struct range{uint64_t offset,source,bytes;};
    bool ready=false;
    uint64_t total=0;
    std::vector<uint8_t> header;
    std::vector<range> ranges;
    wire_payload_reader*source=nullptr;
};

class delta_artifact_receiver {
public:
    // Header/token/logit reads are forward-only and bounded64KiB. State reads
    // in delta_plan order are also forward-only, allowing an existing HTTP or
    // TCP stream without a full-payload buffer. Default cap is decode128;
    // prefill callers must explicitly select their <=49152-token chunk cap.
    bool init(uint64_t bytes,const artifact_read&source,const wire_frontier&base,
        uint64_t session,uint64_t model,std::string&error,uint32_t token_limit=128);
    const delta_plan&plan()const{return delta;}
    const std::vector<int32_t>&tokens()const{return new_tokens;}
    const std::vector<float>&logits()const{return new_logits;}
    bool read_span(const state_span&span,uint64_t offset,void*dst,size_t bytes,std::string&error);
private:
    bool ready=false;
    delta_plan delta;
    std::vector<uint64_t> offsets;
    std::vector<int32_t> new_tokens;
    std::vector<float> new_logits;
    artifact_read source;
};
}
