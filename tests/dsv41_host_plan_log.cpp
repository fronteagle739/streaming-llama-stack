// LLAMADECODE host-plan cost log: headroom arithmetic and line format.
// Numbers from the live Studio fh2 run of 2026-10-25 (admitted_host 6576668672,
// row cache 195035136 B of 195148800 B headroom, owner_digest_staging 33554432).
#include "llama-dsv41-host-plan-log.h"
#include <cstdio>

namespace h = llama_dsv41_host_plan_log;
static int failures = 0;
static void expect(bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

int main() {
    const uint64_t admitted = 6576668672ull, headroom = 195148800ull, base = admitted - headroom;
    const uint64_t provider = 1290818048ull, carry = 826897408ull, tile = 147260928ull, staging = 33554432ull;
    const uint64_t rest = base - provider - carry - tile - staging;
    const std::vector<h::component> parts = {{"provider", provider}, {"carry", carry}, {"tile", tile}, {"owner_digest_staging", staging}, {"other", rest}};
    const auto s = h::line(admitted, base, parts, 195035136ull);
    std::printf("%s\n", s.c_str());
    expect(s.find("sum_matches=1") != std::string::npos, "components sum to the planned base");
    expect(s.find("row_cache_headroom=195148800 ") != std::string::npos, "row-cache headroom = admitted - planned base");
    expect(s.find("owner_digest_staging:228703232") != std::string::npos, "without the 32 MiB staging the headroom is 32 MiB larger");
    expect(s.find("row_cache_bytes=195035136 ") != std::string::npos, "row-cache bytes are reported unchanged");
    expect(h::headroom(10, 20) == 0 && h::headroom(20, 10) == 10, "headroom never underflows");
    const auto bad = h::line(admitted, base, {{"provider", provider}}, 0);
    expect(bad.find("sum_matches=0") != std::string::npos, "a missing component shows sum_matches=0");
    const auto over = h::line(100, 150, {{"a", 100}, {"b", 50}}, 0);
    expect(over.find("row_cache_headroom=0 ") != std::string::npos && over.find("a:50") != std::string::npos && over.find("b:0") != std::string::npos,
           "over-planned base: headroom 0; without a component, what is left");
    std::printf("%s %d failures\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
