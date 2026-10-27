#include "llama-wir.h"

#include <cstdio>

std::vector<llama_token> wir_decode_from_kv(
        llama_model *, llama_context *, int, llama_token, int, float,
        const wir_params &, const float *, int) {
    std::fprintf(stderr, "[upstream-receiver] WIR is not available in the upstream compatibility target\n");
    return {};
}

std::vector<llama_token> wir_decode_cross_tokenizer(
        llama_model *, llama_context *, llama_model *, llama_context *,
        const std::string &, int, llama_token, int, float, const wir_params &) {
    std::fprintf(stderr, "[upstream-receiver] cross-tokenizer WIR is not available in the upstream compatibility target\n");
    return {};
}
