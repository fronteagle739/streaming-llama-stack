// CPU-only verdicts for both runtime guards. No engine or GPU is launched.
#include "llama-dsv41-host-memory.h"
#include <cstdio>
#include <cstring>
#include <initializer_list>

namespace hm = llama_dsv41_host_memory;
static int failures = 0;
static void expect(bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

int main() {
    const uint64_t G = 1ull << 30, base = 3 * G;
    expect(hm::runtime_min_free_bytes == G / 2, "runtime floor defaults to 512 MiB");
    for (const char *bad : {"", "-1", "nan", "inf", "4oops", "17179869184"})
        expect(!hm::configure_runtime_min_free_gib(bad), "invalid floor rejected before admission");
    expect(hm::configure_runtime_min_free_gib("6.5") && hm::runtime_min_free_bytes == 6 * G + G / 2, "fractional GiB supported");
    expect(hm::configure_runtime_min_free_gib("4.0"), "explicit 4 GiB supported");
    expect(hm::paging_limit == 256000000ull, "limit = studio GLOBAL_PAGING_LIMIT (256 MB)");
    expect(!hm::paging_exceeded(base, base), "no growth: run");
    expect(!hm::paging_exceeded(base, base - G), "pagefile shrank below baseline: run (growth 0)");
    expect(hm::paging_growth(base, base - G) == 0, "growth never negative");
    expect(!hm::paging_exceeded(base, base + 256000000ull), "growth exactly at the limit: run");
    expect(hm::paging_exceeded(base, base + 256000001ull), "growth one byte over the limit: stop");
    expect(hm::paging_exceeded(0, 300000000ull), "zero baseline, 300 MB paged: stop");

    // The admission reserve remains independent of the runtime floor.
    hm::state low{};
    low.available = (512ull << 20)-1; low.total = 32 * G; low.commit = 40 * G; low.commit_limit = 128 * G;
    expect(hm::low_memory(low), "below the 512 MiB admission reserve (recorded)");
    hm::state ok{};
    ok.available = 12 * G; ok.total = 32 * G; ok.commit = 40 * G; ok.commit_limit = 128 * G;
    expect(!hm::low_memory(ok), "12 GiB free is not low");
    hm::state commit{};
    commit.available = 12 * G; commit.total = 32 * G; commit.commit = 128 * G - (512ull << 20)+1; commit.commit_limit = 128 * G;
    expect(hm::low_memory(commit), "commit within 512 MiB of the limit (recorded)");
    // Admission still checks fit (begin uses fits with the plan).
    expect(!hm::fits(ok, 13 * G, 0), "admission: 13 GiB host plan does not fit 12 GiB free");
    expect(hm::fits(ok, 10 * G, 0), "admission: 10 GiB host plan fits");

    // live() on this machine (read-only queries): paging below the limit runs.
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
    hm::paging_stop.store(false);
    hm::state fake{}; fake.available = 4 * G;
    expect(hm::check_runtime_floor(fake), "exactly at floor: run");
    fake.available = 3 * G;
    expect(hm::floor_exceeded(fake), "fake available reader below floor: stop verdict");
    expect(hm::configure_runtime_min_free_gib("0"), "0 disables floor");
    expect(hm::check_runtime_floor(fake) && !hm::floor_stop.load(), "disabled floor allows low available RAM");
    expect(hm::paging_exceeded(base, base + 256000001ull), "paging still stops with floor disabled");
    expect(hm::configure_runtime_min_free_gib("4"), "restore 4 GiB");
    expect(!hm::check_runtime_floor(fake) && hm::floor_stop.load(), "low RAM latches existing graceful stop flag");
    char reason[384]; hm::stop_reason(reason, sizeof(reason));
    expect(std::strstr(reason, "available physical RAM") && std::strstr(reason, "--runtime-min-free-gib=4"), "RAM stop names its reason and startup setting");
    expect(!hm::live(), "latched floor stop reaches should_stop via live()");
    std::printf("%s %d failures\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
