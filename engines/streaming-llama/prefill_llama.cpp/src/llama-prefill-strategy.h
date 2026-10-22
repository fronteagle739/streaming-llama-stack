#pragma once

#include <cmath>
#include <cstdint>

enum llama_prefill_strategy {
    LLAMA_PREFILL_STRATEGY_AUTO = 0,
    LLAMA_PREFILL_STRATEGY_MAC  = 1,  // Force Mac RPC prefill + decode
    LLAMA_PREFILL_STRATEGY_RTX  = 2,  // Force RTX streaming prefill + Mac decode
};

struct llama_prefill_strategy_params {
    llama_prefill_strategy strategy = LLAMA_PREFILL_STRATEGY_AUTO;
    int32_t crossover_tokens       = 0;     // 0 = auto-compute from hardware
    float   rtx_streaming_floor_ms = 0.0f;  // 0 = estimate from model size
    float   mac_prefill_tok_s      = 0.0f;  // 0 = use default 300 tok/s
    float   pcie_bandwidth_gbs     = 64.0f; // PCIe 5.0 per GPU
    float   rtx_compute_tok_s      = 2000.0f; // RTX compute rate estimate
};

struct llama_prefill_strategy_decision {
    llama_prefill_strategy chosen;
    int32_t     crossover_tokens;
    float       estimated_mac_ms;
    float       estimated_rtx_ms;
    const char *reason;
};

// Select prefill strategy based on prompt length and hardware estimates.
// Pure function — no side effects, no I/O.
llama_prefill_strategy_decision llama_prefill_strategy_select(
    const llama_prefill_strategy_params & params,
    int32_t n_prompt_tokens,
    float   model_size_gb);
