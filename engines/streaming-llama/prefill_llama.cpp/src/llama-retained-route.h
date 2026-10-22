#pragma once

#include "llama-arch.h"

#include <cstdlib>

// Kept as a small, header-only policy so it can be unit-tested without constructing
// a llama_context (which requires a loaded model and initialized backends).
inline bool llama_retained_route_capable_for(llm_arch arch, bool is_mla) {
    switch (arch) {
        case LLM_ARCH_DEEPSEEK4:
            return true;
        case LLM_ARCH_GLM_DSA:
            // GLM-DSA retained reuse is safe only for its MLA cache layout.
            // LLM_ARCH_DEEPSEEK2 (Kimi) remains out of scope until separately validated.
            return is_mla;
        case LLM_ARCH_MINIMAX_M3: {
            // Match MiniMax-M3 graph selection exactly: any non-empty value whose
            // first character is not '0' enables sparse attention.
            const char * sparse_env = std::getenv("LLAMA_M3_SPARSE");
            const bool m3_sparse = sparse_env && sparse_env[0] != '\0' && sparse_env[0] != '0';
            return !m3_sparse && !is_mla;
        }
        default:
            return false;
    }
}
