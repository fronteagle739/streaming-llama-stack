#pragma once

#include "ggml.h"
#include <cstdlib>
#include <cstring>

// Explicit QFN experiment; upstream behavior is preserved when unset. Use the
// same accumulation mode in ordinary MMQ and both routed MMQ launch paths.
inline bool qfn_mmq_stream_k_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("QFN_MMQ_STREAM_K");
        if (!value) return true;
        GGML_ASSERT(std::strcmp(value, "0") == 0 || std::strcmp(value, "1") == 0);
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}
