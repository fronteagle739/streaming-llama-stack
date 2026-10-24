#pragma once
// LLAMADECODE_ENGRAM_EARLY=auto (default): start the early Engram row load
// only when its predicted wall fits the gap before layer 1 needs table 0.
// Measured (bin-eg, 2026-10-24): the loader sustains ~100K random 4 KiB reads/s
// while weights stream. Early wins when the whole load fits the gap (16K cold:
// 488,608 requests, 4.99 s, done before layer 0 ended, prefill -5 s) and loses
// when it does not (43K: 1.42M requests, 13.7 s, +3 s weight wait).
// 0 forces the sweep-start arm (the early multi-ring loader, started at sweep
// start as in c080518), 1 forces the early start, pause forces the pause arm:
// the early load runs alone and the weight DirectStorage stream holds layers
// >= 1 until both tables are loaded. Auto: early when the load fits the gap,
// otherwise the sweep-start arm (never pause: measured 43K cold 0 36.05 s,
// pause 40.39 s, auto-pause 41.44 s). Every arm loads through the early
// loader, so the row cache is consulted on every turn. Pure; unit-tested.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace llama_dsv41_engram_early {

enum class mode { off, on, pause, automatic };
enum class arm { early, sweep_start };

// Unset or empty: auto. Anything else than 0/1/pause/auto is an error, not off.
inline mode parse_mode(const char *e) {
    if (!e || !*e || !std::strcmp(e, "auto")) return mode::automatic;
    if (!std::strcmp(e, "0")) return mode::off;
    if (!std::strcmp(e, "1")) return mode::on;
    if (!std::strcmp(e, "pause")) return mode::pause;
    throw std::invalid_argument("LLAMADECODE_ENGRAM_EARLY must be 0, 1, pause or auto");
}

constexpr double default_iops = 100000.0;
// Cold launch: from the early start (after admission, before RAM/DS setup and
// the first weight fetch) to layer 1. The 16K gate's 4.99 s load finished
// inside it; at 43K table 0 was needed 3.6 s after the start while the load
// contended with the weight stream. 5 s keeps 16K early; 23K+ start at sweep start.
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
    bool early = false;           // the load fits the gap
    arm chosen = arm::sweep_start; // auto: early when it fits, else sweep start (arm 0)
};
inline const char *arm_name(arm a) { return a == arm::early ? "early" : "sweep_start"; }

inline decision decide(uint64_t requests, double iops, double gap_s) {
    if (!(iops > 0)) throw std::invalid_argument("Engram early decision needs a positive rate");
    decision d;
    d.requests = requests;
    d.predicted_s = double(requests) / iops;
    d.gap_s = gap_s > 0 ? gap_s : 0;
    d.early = d.gap_s > 0 && d.predicted_s <= d.gap_s;
    d.chosen = d.early ? arm::early : arm::sweep_start;
    return d;
}

} // namespace llama_dsv41_engram_early
