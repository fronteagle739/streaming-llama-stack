#include "llama-prefill-telemetry.h"

#include <cstdio>
#include <iomanip>
#include <sstream>

std::string llama_telemetry_to_json(const std::vector<llama_layer_telemetry> & layers) {
    std::ostringstream oss;
    oss << "{\n  \"layers\": [\n";

    for (size_t i = 0; i < layers.size(); ++i) {
        const auto & l = layers[i];
        oss << "    {\n"
            << "      \"layer\": " << l.layer_idx << ",\n"
            << "      \"load_us\": " << l.t_load_us << ",\n"
            << "      \"compute_us\": " << l.t_compute_us << ",\n"
            << "      \"unload_us\": " << l.t_unload_us << ",\n"
            << "      \"weight_bytes\": " << l.weight_bytes << "\n"
            << "    }";
        if (i < layers.size() - 1) {
            oss << ",";
        }
        oss << "\n";
    }

    oss << "  ]\n}";
    return oss.str();
}

void llama_telemetry_print(const std::vector<llama_layer_telemetry> & layers) {
    printf("\n=== Streaming Prefill Layer Telemetry ===\n");
    printf("%-6s %12s %12s %12s %12s\n", "Layer", "Load(ms)", "Compute(ms)", "Unload(ms)", "Size(MB)");
    printf("--------------------------------------------------------------\n");

    for (const auto & l : layers) {
        printf("%-6d %12.2f %12.2f %12.2f %12.2f\n", l.layer_idx, l.t_load_us / 1000.0, l.t_compute_us / 1000.0,
               l.t_unload_us / 1000.0, l.weight_bytes / (1024.0 * 1024.0));
    }

    auto stats = llama_telemetry_compute_stats(layers);
    printf("--------------------------------------------------------------\n");
    printf("Avg:   %12.2f %12.2f %12.2f\n", stats.avg_load_ms, stats.avg_compute_ms, stats.avg_unload_ms);
    printf("Total: %12.2f ms (load: %.1f%%, unload: %.1f%%)\n", stats.total_time_ms, stats.load_overhead_pct,
           stats.unload_overhead_pct);
}

llama_telemetry_stats llama_telemetry_compute_stats(const std::vector<llama_layer_telemetry> & layers) {
    llama_telemetry_stats stats = {};

    if (layers.empty()) {
        return stats;
    }

    int64_t total_load_us    = 0;
    int64_t total_compute_us = 0;
    int64_t total_unload_us  = 0;

    for (const auto & l : layers) {
        total_load_us += l.t_load_us;
        total_compute_us += l.t_compute_us;
        total_unload_us += l.t_unload_us;
    }

    int64_t total_us = total_load_us + total_compute_us + total_unload_us;
    size_t  n        = layers.size();

    stats.avg_load_ms         = (total_load_us / (float) n) / 1000.0f;
    stats.avg_compute_ms      = (total_compute_us / (float) n) / 1000.0f;
    stats.avg_unload_ms       = (total_unload_us / (float) n) / 1000.0f;
    stats.total_time_ms       = total_us / 1000.0f;
    stats.load_overhead_pct   = (total_us > 0) ? 100.0f * total_load_us / total_us : 0.0f;
    stats.unload_overhead_pct = (total_us > 0) ? 100.0f * total_unload_us / total_us : 0.0f;

    return stats;
}
