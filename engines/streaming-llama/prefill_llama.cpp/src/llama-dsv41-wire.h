#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <array>
#include <mutex>

// The native ds4 V4.1 session payload, not llama.cpp's sequence-state format.
// Pinned reference: stack-backend c8aa547e, ds4.c ds41_state_spans /
// ds41_save_payload. No graph/backend dependencies: callers provide already
// completed, model-compatible F32 state, in native ring storage order.
namespace llama_dsv41 {

enum class state_kind { dspark_hidden, raw_window, compressed, index, pool_kv, pool_score };
struct state_span {
    state_kind kind;
    uint32_t index;
    uint64_t bytes;
};

struct wire_frontier {
    uint32_t context = 0;
    uint32_t position = 0;
    // 1: ordinary decode, 2: legacy DSpark capture, 3: capture with frontier.
    uint32_t mode = 1;
    uint32_t capture_end = 0;
    uint32_t capture_length = 0;
};

struct wire_plan {
    wire_frontier frontier;
    uint64_t bytes = 0;
    std::vector<state_span> spans;
};

bool make_wire_plan(const wire_frontier & frontier, wire_plan & plan, std::string & error);

using wire_write = std::function<bool(const void *, size_t)>;
using wire_read_span = std::function<bool(const state_span &, uint64_t offset, void *, size_t)>;

// Bounded serialization (at most 1 MiB scratch). All token/logit/frontier
// validation occurs before the first write. A failed writer/read invalidates
// the output; callers must publish only after complete success. The graph
// owner must synchronize and certify a complete frontier before calling.
bool write_native_payload(const wire_frontier & frontier,
                           const int32_t * tokens, size_t token_count,
                           const float * logits, size_t logits_count,
                           const wire_read_span & read_span, const wire_write & write,
                           std::string & error);

// Random-access pull adapter for the existing bounded streaming transport.
// Header metadata is tiny; tokens/logits/cache stay in their completed source
// storage. They must remain immutable/alive until transport completes. State
// readbacks are serialized even if transport channels pull concurrently. An
// outer pull may be up to16MiB (transport-owned buffer); each cache read stays
// <=1MiB and no additional pull-sized allocation occurs in this adapter.
class wire_payload_reader {
public:
    bool init(const wire_frontier & frontier,const int32_t * tokens,size_t token_count,
              const float * logits,size_t logits_count,const wire_read_span & read_span,std::string & error);
    uint64_t bytes() const;
    bool read(uint64_t offset,void * dst,size_t bytes,std::string & error);
private:
    mutable std::mutex mutex;
    bool ready=false;
    wire_plan plan;
    std::array<uint8_t,60> header{};
    size_t header_bytes=0;
    const uint8_t * tokens=nullptr,*logits=nullptr;
    uint64_t token_bytes=0,logit_bytes=0;
    wire_read_span read_span;
};

} // namespace llama_dsv41
