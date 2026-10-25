#pragma once
// LLAMADECODE test-only fault injection (off unless LLAMADECODE_TEST_FAULT is set):
//   LLAMADECODE_TEST_FAULT=<kind>@<layer>   kind = main_throw | worker_throw | bad_alloc
// Fires once, at the start of <layer> in the first multi-token sweep that reaches it:
//   main_throw   std::runtime_error thrown on the sweep (main session) thread
//   bad_alloc    std::bad_alloc thrown on the sweep thread
//   worker_throw std::runtime_error thrown on a std::async worker thread and
//                marshalled to the sweep thread by future::get() (the pattern of
//                the Engram lookahead worker), so no std::terminate
// Used to prove that every producer abort reaches the client as a reason.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <new>
#include <stdexcept>
#include <string>

namespace llama_dsv41_test_fault {

enum class kind { none, main_throw, worker_throw, bad_alloc };
struct spec { kind k = kind::none; uint32_t layer = 0; };

inline const char *name(kind k) {
    return k == kind::main_throw ? "main_throw" : k == kind::worker_throw ? "worker_throw" : k == kind::bad_alloc ? "bad_alloc" : "none";
}
// "<kind>@<layer>", layer 0..1023. Empty/null = none. Anything else throws.
inline spec parse(const char *value) {
    spec out;
    if (!value || !*value) return out;
    const char *at = std::strchr(value, '@');
    const std::string k = at ? std::string(value, at) : std::string(value);
    if (k == "main_throw") out.k = kind::main_throw;
    else if (k == "worker_throw") out.k = kind::worker_throw;
    else if (k == "bad_alloc") out.k = kind::bad_alloc;
    char *end = nullptr;
    const unsigned long layer = at ? std::strtoul(at + 1, &end, 10) : 0;
    if (out.k == kind::none || !at || !at[1] || !end || *end || layer > 1023)
        throw std::runtime_error("LLAMADECODE_TEST_FAULT must be <main_throw|worker_throw|bad_alloc>@<layer>");
    out.layer = uint32_t(layer);
    return out;
}
inline const spec &armed() { static const spec s = parse(std::getenv("LLAMADECODE_TEST_FAULT")); return s; }
inline std::atomic<bool> fired{false};

// Startup: validate and log the armed fault (nothing when unset).
inline void announce() {
    const auto &s = armed();
    if (s.k == kind::none) return;
    std::fprintf(stderr, "LLAMADECODE_TEST_FAULT_ARMED kind=%s layer=%u fires=once first_multi_token_sweep=1 test_only=1\n", name(s.k), s.layer);
    std::fflush(stderr);
}

// Called at the start of each layer of a sweep of `tokens` tokens.
inline void at_layer(uint32_t layer, size_t tokens, const spec &s = armed()) {
    if (s.k == kind::none || layer != s.layer || tokens < 2 || fired.exchange(true)) return;
    std::fprintf(stderr, "LLAMADECODE_TEST_FAULT_FIRED kind=%s layer=%u tokens=%zu\n", name(s.k), layer, tokens);
    std::fflush(stderr);
    char reason[160];
    if (s.k == kind::bad_alloc) throw std::bad_alloc();
    if (s.k == kind::main_throw) {
        std::snprintf(reason, sizeof reason, "LLAMADECODE_TEST_FAULT main_throw at layer %u on the sweep thread (test-only injected fault)", layer);
        throw std::runtime_error(reason);
    }
    std::snprintf(reason, sizeof reason, "LLAMADECODE_TEST_FAULT worker_throw at layer %u on a worker thread (test-only injected fault)", layer);
    const std::string text(reason);
    auto worker = std::async(std::launch::async, [text] { throw std::runtime_error(text); });
    worker.get();   // rethrows the worker's exception on this thread
}

} // namespace llama_dsv41_test_fault
