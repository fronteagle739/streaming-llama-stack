#pragma once
// LLAMADECODE open-phase timing (log only, always on, no behaviour change).
// One line per llama_dsv41_sweep::open:
//   LLAMADECODE_OPEN_PHASES ok=1 total_ms=T <step>_ms=.. [<step>.<sub>_ms=.. <step>.<key>=..] ... unaccounted_ms=U
// Steps are listed in the order they start; nested steps are dotted under their
// parent. unaccounted_ms = total - the sum of the top-level steps. Only the thread
// that opened records (other threads' calls are not timed here).
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <thread>
#include <vector>
// Physical-RAM figures come from the caller (llama-dsv41-locked-ram.h); this
// header stays free of <windows.h> so its include order cannot change any TU.

namespace llama_dsv41_open_phases {
using clock = std::chrono::steady_clock;
inline double ms_since(clock::time_point t) { return std::chrono::duration<double, std::milli>(clock::now() - t).count(); }

struct recorder {
    bool active = false;
    std::thread::id owner;
    clock::time_point start;
    double top_ms = 0;
    std::vector<std::string> path;     // open scope names (outermost first)
    std::vector<std::string> fields;   // in start order
};
inline recorder &rec() { static recorder r; return r; }
inline bool mine() { const auto &r = rec(); return r.active && r.owner == std::this_thread::get_id(); }
inline std::string prefix() {
    std::string out;
    for (const auto &p : rec().path) { out += p; out += '.'; }
    return out;
}
// A key=value field under the current scope.
inline void note(const char *key, uint64_t value) {
    if (!mine()) return;
    rec().fields.push_back(prefix() + key + "=" + std::to_string(value));
}
inline void note_ms(const char *key, double value) {
    if (!mine()) return;
    char b[48]; std::snprintf(b, sizeof b, "_ms=%.1f", value);
    rec().fields.push_back(prefix() + key + b);
}

// A timed step: <path.name>_ms, in start order.
class scope {
    bool on = false;
    size_t index = 0;
    std::string key;
    clock::time_point started;
public:
    explicit scope(const char *name) {
        if (!mine()) return;
        auto &r = rec();
        on = true; key = prefix() + name; index = r.fields.size();
        r.fields.emplace_back();   // reserved: keeps start order
        r.path.emplace_back(name);
        started = clock::now();
    }
    scope(const scope &) = delete;
    scope &operator=(const scope &) = delete;
    ~scope() {
        if (!on) return;
        auto &r = rec();
        const double ms = ms_since(started);
        r.path.pop_back();
        if (r.path.empty()) r.top_ms += ms;
        char b[48]; std::snprintf(b, sizeof b, "_ms=%.1f", ms);
        if (index < r.fields.size()) r.fields[index] = key + b;
    }
};

// One per open: begins on construction, prints the line on destruction.
class session {
    bool on = false;
public:
    bool ok = false;
    session() {
        auto &r = rec();
        if (r.active) return;   // nested open on the same process: the outer one records
        r = recorder{}; r.active = true; r.owner = std::this_thread::get_id(); r.start = clock::now(); on = true;
    }
    session(const session &) = delete;
    session &operator=(const session &) = delete;
    ~session() {
        if (!on) return;
        auto &r = rec();
        const double total = ms_since(r.start);
        std::string line;
        for (const auto &f : r.fields) if (!f.empty()) { line += ' '; line += f; }
        std::fprintf(stderr, "LLAMADECODE_OPEN_PHASES ok=%d total_ms=%.1f%s unaccounted_ms=%.1f\n", int(ok), total, line.c_str(), total - r.top_ms);
        std::fflush(stderr);
        r.active = false;
    }
};
} // namespace llama_dsv41_open_phases
