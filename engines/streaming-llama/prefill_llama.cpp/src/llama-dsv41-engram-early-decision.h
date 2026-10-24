#pragma once
// LLAMADECODE_ENGRAM_EARLY=auto (default): start the early Engram row load
// only when its predicted wall fits the gap before layer 1 needs table 0.
// Measured (bin-eg, 2026-10-24): the loader sustains ~100K random 4 KiB reads/s
// while weights stream. Early wins when the whole load fits the gap (16K cold:
// 488,608 requests, 4.99 s, done before layer 0 ended, prefill -5 s) and loses
// when it does not (43K: 1.42M requests, 13.7 s, +3 s weight wait).
// 0 forces off (the pre-37cad00 path), 1 forces on. Pure; unit-tested.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace llama_dsv41_engram_early {

enum class mode { off, on, automatic };

// Unset or empty: auto. Anything else than 0/1/auto is an error, not off.
inline mode parse_mode(const char *e) {
    if (!e || !*e || !std::strcmp(e, "auto")) return mode::automatic;
    if (!std::strcmp(e, "0")) return mode::off;
    if (!std::strcmp(e, "1")) return mode::on;
    throw std::invalid_argument("LLAMADECODE_ENGRAM_EARLY must be 0, 1 or auto");
}

constexpr double default_iops = 100000.0;
// Cold launch: from the early start (after admission, before RAM/DS setup and
// the first weight fetch) to layer 1. The 16K gate's 4.99 s load finished
// inside it; at 43K table 0 was needed 3.6 s after the start while the load
// contended with the weight stream. 5 s keeps 16K early and 23K+ ordinary.
constexpr double cold_gap_s = 5.0;

// LLAMADECODE_ENGRAM_IOPS: sustained random-read rate, requests per second.
inline double parse_iops(const char *e) {
    if (!e || !*e) return default_iops;
    char *end = nullptr;
    const double v = std::strtod(e, &end);
    if (!end || *end || !(v >= 1000.0 && v <= 1.0e8))
        throw std::invalid_argument("LLAMADECODE_ENGRAM_IOPS must be a rate in [1000, 1e8] requests/s");
    return v;
}

struct decision {
    uint64_t requests = 0;
    double predicted_s = 0, gap_s = 0;
    bool early = false;
};

inline decision decide(uint64_t requests, double iops, double gap_s) {
    if (!(iops > 0)) throw std::invalid_argument("Engram early decision needs a positive rate");
    decision d;
    d.requests = requests;
    d.predicted_s = double(requests) / iops;
    d.gap_s = gap_s > 0 ? gap_s : 0;
    d.early = d.gap_s > 0 && d.predicted_s <= d.gap_s;
    return d;
}

} // namespace llama_dsv41_engram_early
