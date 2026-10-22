#pragma once
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifndef PSAPI_VERSION
#define PSAPI_VERSION 2
#endif
#include <psapi.h>
#include <cstdint>
#include <cstdio>
#include <limits>

// Physical host residency and WDDM backing commitment are independent budgets.
// A caller supplies its complete bounded host/device plan before CUDA starts.
namespace llama_dsv41_host_memory {
constexpr uint64_t reserve = UINT64_C(1) << 30;
struct state {
    uint64_t available = 0, total = 0, commit = 0, commit_limit = 0;
    uint64_t private_bytes = 0, private_peak = 0;
};
inline uint64_t private_limit = 0;
inline HANDLE job = nullptr;
inline bool read(state &s) {
    MEMORYSTATUSEX m{}; m.dwLength = sizeof(m);
    PERFORMANCE_INFORMATION p{}; p.cb = sizeof(p);
    PROCESS_MEMORY_COUNTERS_EX c{}; c.cb = sizeof(c);
    if (!GlobalMemoryStatusEx(&m) || !GetPerformanceInfo(&p, sizeof(p)) ||
        !GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&c, sizeof(c))) return false;
    s = {m.ullAvailPhys, m.ullTotalPhys, uint64_t(p.CommitTotal) * p.PageSize,
        uint64_t(p.CommitLimit) * p.PageSize, c.PrivateUsage, c.PeakPagefileUsage};
    return true;
}
inline bool fits(const state &s, uint64_t host, uint64_t device) {
    if (s.available < reserve || host > s.available - reserve ||
        s.commit_limit < reserve || s.commit > s.commit_limit - reserve) return false;
    const uint64_t remaining = s.commit_limit - reserve - s.commit;
    return host <= remaining && device <= remaining - host &&
        (!private_limit || (s.private_bytes <= private_limit &&
            host <= private_limit - s.private_bytes &&
            device <= private_limit - s.private_bytes - host));
}
inline void report(const char *event, const state &s, uint64_t host = 0, uint64_t device = 0) {
    char text[768];
    const int n = snprintf(text, sizeof(text),
        "DSV41_HOST_MEMORY_%s host_plan_bytes=%llu device_plan_bytes=%llu available_physical_bytes=%llu "
        "physical_total_bytes=%llu system_commit_bytes=%llu system_commit_limit_bytes=%llu "
        "private_bytes=%llu private_peak_bytes=%llu private_limit_bytes=%llu reserve_bytes=%llu\n",
        event, (unsigned long long)host, (unsigned long long)device,
        (unsigned long long)s.available, (unsigned long long)s.total,
        (unsigned long long)s.commit, (unsigned long long)s.commit_limit,
        (unsigned long long)s.private_bytes, (unsigned long long)s.private_peak,
        (unsigned long long)private_limit, (unsigned long long)reserve);
    if (n > 0) { DWORD done; WriteFile(GetStdHandle(STD_ERROR_HANDLE), text,
        DWORD(size_t(n) < sizeof(text) ? size_t(n) : sizeof(text) - 1), &done, nullptr); }
}
inline bool live() {
    state s;
    if (!read(s)) { report("QUERY_FAILED", s); return false; }
    if (!fits(s, 0, 0)) { report("REFUSED_LIVE_RESERVE", s); return false; }
    return true;
}
inline DWORD WINAPI watch(void *) {
    for (;;) {
        Sleep(250);
        if (!live()) { TerminateProcess(GetCurrentProcess(), 78); return 78; }
    }
}
inline bool begin(uint64_t host, uint64_t device) {
    // Single-threaded before CUDA/producer workers; no restart or cap changes.
    if (job) return false;
    state s;
    if (!read(s)) { report("QUERY_FAILED", s); return false; }
    if (!fits(s, host, device)) { report("REFUSED_PLAN", s, host, device); return false; }
    const uint64_t increment = s.commit_limit - reserve - s.commit;
    if (increment > UINT64_MAX - s.private_bytes) return false;
    private_limit = s.private_bytes + increment;
    HANDLE next = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limit{};
    limit.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
    limit.ProcessMemoryLimit = SIZE_T(private_limit);
    if (!next || !SetInformationJobObject(next, JobObjectExtendedLimitInformation, &limit, sizeof(limit)) ||
        !AssignProcessToJobObject(next, GetCurrentProcess())) {
        if (next) CloseHandle(next);
        report("REFUSED_JOB", s, host, device); return false;
    }
    job = next; // Process-lifetime nested Job; outer owner retains kill-on-close.
    HANDLE thread = CreateThread(nullptr, 65536, watch, nullptr, STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
    if (!thread) { report("REFUSED_WATCHDOG", s, host, device); return false; }
    CloseHandle(thread); report("ADMITTED", s, host, device);
    return true;
}
}
#endif
