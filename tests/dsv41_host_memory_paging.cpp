// Owner paging-only rule for the producer's runtime host-memory watch
// (llama-dsv41-host-memory.h): admission may check fit; at runtime only real
// paging (global pagefile usage > 256,000,000 bytes over the launch baseline,
// studio.job_paging.GLOBAL_PAGING_LIMIT) stops the producer; low free RAM is
// recorded, not fatal.
#include "llama-dsv41-host-memory.h"
#include <cstdio>

namespace hm = llama_dsv41_host_memory;
static int failures = 0;
static void expect(bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

int main() {
    const uint64_t G = 1ull << 30, base = 3 * G;
    expect(hm::paging_limit == 256000000ull, "limit = studio GLOBAL_PAGING_LIMIT (256 MB)");
    expect(!hm::paging_exceeded(base, base), "no growth: run");
    expect(!hm::paging_exceeded(base, base - G), "pagefile shrank below baseline: run (growth 0)");
    expect(hm::paging_growth(base, base - G) == 0, "growth never negative");
    expect(!hm::paging_exceeded(base, base + 256000000ull), "growth exactly at the limit: run");
    expect(hm::paging_exceeded(base, base + 256000001ull), "growth one byte over the limit: stop");
    expect(hm::paging_exceeded(0, 300000000ull), "zero baseline, 300 MB paged: stop");

    // Low free RAM / commit near its limit with zero paging is recorded, not fatal.
    hm::state low{};
    low.available = 512ull << 20; low.total = 32 * G; low.commit = 40 * G; low.commit_limit = 128 * G;
    expect(hm::low_memory(low), "512 MiB free is below the 1 GiB reserve (recorded)");
    hm::state ok{};
    ok.available = 12 * G; ok.total = 32 * G; ok.commit = 40 * G; ok.commit_limit = 128 * G;
    expect(!hm::low_memory(ok), "12 GiB free is not low");
    hm::state commit{};
    commit.available = 12 * G; commit.total = 32 * G; commit.commit = 128 * G - (512ull << 20); commit.commit_limit = 128 * G;
    expect(hm::low_memory(commit), "commit within 1 GiB of the limit (recorded)");
    // Admission still checks fit (begin uses fits with the plan).
    expect(!hm::fits(ok, 13 * G, 0), "admission: 13 GiB host plan does not fit 12 GiB free");
    expect(hm::fits(ok, 10 * G, 0), "admission: 10 GiB host plan fits");

    // live() on this machine (read-only queries): low RAM never stops it, paging below the limit runs.
    uint64_t now = 0;
    expect(hm::pagefile_in_use(now), "EnumPageFilesW pagefile usage readable");
    hm::paging_baseline.store(now);
    expect(hm::live() && !hm::paging_stop.load(), "live(): runs at its own baseline");
    hm::paging_baseline.store(now > 300000000ull ? now - 300000000ull : 0);
    if (now > 300000000ull) {
        // Force a fresh sample past the 250 ms throttle.
        Sleep(300);
        expect(!hm::live() && hm::paging_stop.load(), "live(): 300 MB over the baseline stops (graceful flag)");
        expect(!hm::live(), "stop latches");
    } else {
        std::printf("SKIP live stop sample (pagefile in use %llu bytes < 300 MB)\n", (unsigned long long)now);
    }
    std::printf("%s %d failures\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
