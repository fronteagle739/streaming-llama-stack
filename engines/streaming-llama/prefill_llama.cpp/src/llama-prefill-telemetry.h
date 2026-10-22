#pragma once

#include <cstdint>
#include <string>
#include <vector>

//
// Per-layer telemetry for streaming prefill
//
// This module provides timing and metrics collection for each layer
// during streaming prefill mode, enabling analysis of:
// - Weight load/unload times
// - Per-layer compute times
// - Bottleneck identification
//

struct llama_layer_telemetry {
    int     layer_idx;
    int64_t t_load_us;     // time to load layer weights into GPU buffer
    int64_t t_compute_us;  // time for layer forward pass
    int64_t t_unload_us;   // time to unload layer weights from buffer
    size_t  weight_bytes;  // size of layer weights
};

// JSON output for metrics analysis
std::string llama_telemetry_to_json(const std::vector<llama_layer_telemetry> & layers);

// Console output for debugging
void llama_telemetry_print(const std::vector<llama_layer_telemetry> & layers);

// Compute aggregate statistics
struct llama_telemetry_stats {
    float avg_load_ms;
    float avg_compute_ms;
    float avg_unload_ms;
    float total_time_ms;
    float load_overhead_pct;    // percentage of time spent loading
    float unload_overhead_pct;  // percentage of time spent unloading
};

llama_telemetry_stats llama_telemetry_compute_stats(const std::vector<llama_layer_telemetry> & layers);
