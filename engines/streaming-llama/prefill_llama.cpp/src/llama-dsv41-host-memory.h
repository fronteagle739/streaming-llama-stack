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
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <limits>

// Physical host residency and WDDM backing commitment are independent budgets.
// A caller supplies its complete bounded host/device plan before CUDA starts.
namespace llama_dsv41_host_memory {
constexpr uint64_t reserve = UINT64_C(512) << 20;
struct state {
    uint64_t available = 0, total = 0, commit = 0, commit_limit = 0;
    uint64_t private_bytes = 0, private_peak = 0;
};
inline uint64_t private_limit = 0;
inline uint64_t runtime_min_free_bytes = UINT64_C(512) << 20;
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
        "private_bytes=%llu private_peak_bytes=%llu private_limit_bytes=%llu reserve_bytes=%llu runtime_min_free_bytes=%llu\n",
        event, (unsigned long long)host, (unsigned long long)device,
        (unsigned long long)s.available, (unsigned long long)s.total,
        (unsigned long long)s.commit, (unsigned long long)s.commit_limit,
        (unsigned long long)s.private_bytes, (unsigned long long)s.private_peak,
        (unsigned long long)private_limit, (unsigned long long)reserve, (unsigned long long)runtime_min_free_bytes);
    if (n > 0) { DWORD done; WriteFile(GetStdHandle(STD_ERROR_HANDLE), text,
        DWORD(size_t(n) < sizeof(text) ? size_t(n) : sizeof(text) - 1), &done, nullptr); }
}
// Runtime available physical RAM floor, independent of admission's reserve.
// Runtime floor defaults to 512 MiB. 0 disables
// it; the existing paging stop remains enabled in either case.
inline std::atomic<bool> floor_stop{false}, memory_query_failed{false};
inline std::atomic<uint64_t> stopped_available{0};
inline bool configure_runtime_min_free_gib(const char *text) {
    if (!text || !*text || job) return false;
    char *end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end || !std::isfinite(value) || value < 0 || value >= 17179869184.0) return false;
    runtime_min_free_bytes = uint64_t(value * double(UINT64_C(1) << 30));
    return true;
}
inline bool floor_exceeded(const state &s) { return runtime_min_free_bytes && s.available < runtime_min_free_bytes; }
inline bool check_runtime_floor(const state &s) {
    if (floor_stop.load()) return false;
    if (!floor_exceeded(s)) return true;
    stopped_available.store(s.available); floor_stop.store(true);
    report("MIN_FREE_STOP", s);
    return false;
}
constexpr uint64_t paging_limit = UINT64_C(256000000);   // studio.job_paging.GLOBAL_PAGING_LIMIT
inline std::atomic<bool> paging_stop{false};
inline std::atomic<uint64_t> paging_baseline{UINT64_MAX}, paging_growth_peak{0}, low_memory_events{0};
inline bool stopped() { return floor_stop.load() || memory_query_failed.load() || paging_stop.load(); }
inline void stop_reason(char *text, size_t capacity) {
    if (floor_stop.load()) std::snprintf(text, capacity,
        "Producer stopped: available physical RAM %llu bytes below --runtime-min-free-gib=%g (%llu bytes)",
        (unsigned long long)stopped_available.load(), double(runtime_min_free_bytes) / double(UINT64_C(1) << 30),
        (unsigned long long)runtime_min_free_bytes);
    else if (memory_query_failed.load()) std::snprintf(text, capacity, "Producer stopped: physical RAM query failed while runtime floor enabled");
    else std::snprintf(text, capacity, "Producer stopped: paging detected or paging query failed (pagefile growth limit 256 MB)");
}
// Pure verdicts (tests/dsv41_host_memory_paging.cpp).
inline uint64_t paging_growth(uint64_t baseline, uint64_t now) { return now > baseline ? now - baseline : 0; }
inline bool paging_exceeded(uint64_t baseline, uint64_t now, uint64_t limit = paging_limit) {
    return paging_growth(baseline, now) > limit;
}
inline bool low_memory(const state &s) { return !fits(s, 0, 0); }
// Global pagefile bytes in use (EnumPageFilesW TotalInUse pages; same source as
// studio.job_paging.paging_bytes). Private commitment is not paging.
inline BOOL CALLBACK pagefile_visit(LPVOID context, PENUM_PAGE_FILE_INFORMATION info, LPCWSTR) {
    *static_cast<uint64_t *>(context) += uint64_t(info->TotalInUse); return TRUE;
}
inline bool pagefile_in_use(uint64_t &bytes) {
    uint64_t pages = 0; SYSTEM_INFO si{}; GetSystemInfo(&si);
    if (!EnumPageFilesW(pagefile_visit, &pages)) return false;
    bytes = pages * uint64_t(si.dwPageSize); return true;
}
inline void report_paging(const char *event, uint64_t baseline, uint64_t now) {
    std::fprintf(stderr, "DSV41_HOST_MEMORY_%s pagefile_in_use_bytes=%llu baseline_bytes=%llu growth_bytes=%llu limit_bytes=%llu low_memory_events=%llu\n",
        event, (unsigned long long)now, (unsigned long long)baseline, (unsigned long long)paging_growth(baseline, now),
        (unsigned long long)paging_limit, (unsigned long long)low_memory_events.load());
    std::fflush(stderr);
}
inline bool set_paging_baseline() {
    uint64_t now = 0; if (!pagefile_in_use(now)) return false;
    paging_baseline.store(now); report_paging("PAGING_BASELINE", now, now); return true;
}
// Latch the graceful stop on either guard; sampled at most every 250 ms.
inline bool live() {
    if (stopped()) return false;
    static std::atomic<int64_t> last{0};
    const int64_t t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (t - last.load() < 250) return true;
    last.store(t);
    state s;
    const bool readable = read(s);
    if (!readable && runtime_min_free_bytes) {
        report("QUERY_FAILED", s); memory_query_failed.store(true); return false;
    }
    if (readable && !check_runtime_floor(s)) return false;
    if (readable && low_memory(s) && low_memory_events.fetch_add(1) == 0) report("LOW_MEMORY_RECORDED", s);
    uint64_t now = 0;
    if (!pagefile_in_use(now)) { report("PAGING_QUERY_FAILED", s); paging_stop.store(true); return false; }
    uint64_t base = paging_baseline.load();
    if (base == UINT64_MAX) { paging_baseline.store(now); base = now; }
    const uint64_t growth = paging_growth(base, now);
    if (growth > paging_growth_peak.load()) paging_growth_peak.store(growth);
    if (paging_exceeded(base, now)) { report_paging("PAGING_STOP", base, now); paging_stop.store(true); return false; }
    return true;
}
// Samples RAM and paging at 4 Hz. It only flags the stop: the producer stops
// gracefully at its next boundary (should_stop); never a TerminateProcess.
inline DWORD WINAPI watch(void *) {
    for (;;) {
        Sleep(250);
        if (!live()) return 78;
    }
}
inline bool begin(uint64_t host, uint64_t device) {
    // Single-threaded before CUDA/producer workers; no restart or cap changes.
    if (job) return false;
    state s;
    if (!read(s)) { report("QUERY_FAILED", s); return false; }
    if (!fits(s, host, device)) { report("REFUSED_PLAN", s, host, device); return false; }
    if (!set_paging_baseline()) { report("PAGING_QUERY_FAILED", s, host, device); return false; }
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
