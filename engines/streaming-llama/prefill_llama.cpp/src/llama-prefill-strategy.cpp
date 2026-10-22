#include "llama-prefill-strategy.h"

#include <algorithm>
#include <cmath>

llama_prefill_strategy_decision llama_prefill_strategy_select(
    const llama_prefill_strategy_params & params,
    int32_t n_prompt_tokens,
    float   model_size_gb) {

    llama_prefill_strategy_decision d = {};
    d.crossover_tokens = 0;
    d.estimated_mac_ms = 0.0f;
    d.estimated_rtx_ms = 0.0f;

    // Forced modes
    if (params.strategy == LLAMA_PREFILL_STRATEGY_MAC) {
        d.chosen = LLAMA_PREFILL_STRATEGY_MAC;
        d.reason = "forced by --prefill-strategy mac";
        return d;
    }
    if (params.strategy == LLAMA_PREFILL_STRATEGY_RTX) {
        d.chosen = LLAMA_PREFILL_STRATEGY_RTX;
        d.reason = "forced by --prefill-strategy rtx";
        return d;
    }

    // Auto mode: compute crossover
    const float mac_tok_s  = params.mac_prefill_tok_s > 0.0f ? params.mac_prefill_tok_s : 300.0f;
    const float pcie_bw    = params.pcie_bandwidth_gbs > 0.0f ? params.pcie_bandwidth_gbs : 64.0f;
    const float rtx_tok_s  = params.rtx_compute_tok_s > 0.0f ? params.rtx_compute_tok_s : 2000.0f;

    int32_t crossover = params.crossover_tokens;
    if (crossover <= 0) {
        // RTX streaming floor = model_size / PCIe bandwidth
        const float rtx_floor_s = model_size_gb / pcie_bw;
        crossover = (int32_t)(rtx_floor_s * mac_tok_s);
        crossover = std::max(crossover, (int32_t)100);  // minimum sensible threshold
    }
    d.crossover_tokens = crossover;

    // Estimate times for both paths
    d.estimated_mac_ms = (n_prompt_tokens / mac_tok_s) * 1000.0f;

    const float rtx_floor_ms = params.rtx_streaming_floor_ms > 0.0f
        ? params.rtx_streaming_floor_ms
        : (model_size_gb / pcie_bw) * 1000.0f;
    const float rtx_compute_ms = (n_prompt_tokens / rtx_tok_s) * 1000.0f;
    d.estimated_rtx_ms = std::max(rtx_floor_ms, rtx_compute_ms);

    // Decision
    if (n_prompt_tokens < crossover) {
        d.chosen = LLAMA_PREFILL_STRATEGY_MAC;
        d.reason = "prompt < crossover threshold";
    } else {
        d.chosen = LLAMA_PREFILL_STRATEGY_RTX;
        d.reason = "prompt >= crossover threshold";
    }

    return d;
}
