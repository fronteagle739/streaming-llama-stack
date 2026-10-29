#pragma once

// Diagnostic counter for the dispatch thread, not elapsed wall time. A delta
// excludes blocked/descheduled time, but includes active driver polling during
// synchronization. It does not count the provider's separate workers.
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

inline double llama_dsv41_host_thread_cpu_ms() {
#if defined(_WIN32)
    FILETIME created, exited, kernel, user;
    if (!GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user)) return -1;
    const auto ticks = [](const FILETIME & t) {
        return (static_cast<unsigned long long>(t.dwHighDateTime) << 32) | t.dwLowDateTime;
    };
    return double(ticks(kernel) + ticks(user)) / 10000.0;
#else
    return -1; // No wall-time substitution on unsupported platforms.
#endif
}
