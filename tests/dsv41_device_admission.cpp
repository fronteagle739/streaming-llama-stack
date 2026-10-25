// LLAMADECODE measured device plan: admitted = min(job bound, measured free - reserve);
// the combined plan must fit under it; the context cache is charged at what it
// can hold for its context (<= --cache-bytes). Numbers are the RTX 5090 runs of
// 2026-10-25 (lazy free 32495370240, EAGER free 32252100608, job 31377412860).
#include "llama-dsv41-device-admission.h"
#include <cstdio>

namespace a = llama_dsv41_device_admission;
static int failures = 0;
static void expect(bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

int main() {
    const uint64_t gib = 1ull << 30, job = 31377412860ull, lazy_free = 32495370240ull, eager_free = 32252100608ull;
    // measured below the bound: admitted = free - reserve
    auto r = a::admit(eager_free, job, gib, 29000000000ull);
    expect(r.ok && r.admitted == eager_free - gib, "measured free - reserve below the job bound is the admission");
    // measured above the bound: the bound wins
    r = a::admit(34190458880ull, job, gib, job);
    expect(r.ok && r.admitted == job, "measured free - reserve above the job bound: admitted = job bound");
    r = a::admit(lazy_free, job, gib, 31376429820ull);
    expect(r.ok && r.admitted == job, "today's lazy free admits today's plan at the full job bound");
    // refusal below the minimum: the plan does not fit the measured admission
    r = a::admit(eager_free, job, gib, 31376429820ull);
    expect(!r.ok && r.admitted == eager_free - gib && r.reason.find("exceeds the measured admission") != std::string::npos,
           "plan above min(job, free - reserve) is refused, with the numbers");
    r = a::admit(eager_free, job, gib, eager_free - gib);
    expect(r.ok, "plan exactly at the measured admission fits");
    r = a::admit(gib, job, gib, 1);
    expect(!r.ok && r.admitted == 0, "free not above the reserve is refused");
    r = a::admit(eager_free, 0, gib, 1);
    expect(!r.ok, "no job bound is refused");
    // context cache charge
    const uint64_t cache_args = 3225419776ull, need_262k_f32 = 1688219648ull, need_1m_bf16 = 3208654848ull;
    const uint64_t c262 = a::cache_charge(cache_args, need_262k_f32), c1m = a::cache_charge(cache_args, need_1m_bf16);
    expect(c262 == need_262k_f32 + 17ull * 1024 * 1024, "262K f32 cache charged at its full-context bytes + 8 x 2 MiB rounding + 1 MiB");
    expect(c1m == cache_args, "1M bf16 cache: full need + rounding exceeds --cache-bytes, charged at --cache-bytes");
    expect(a::cache_charge(1000, 5000) == 1000, "charge never exceeds --cache-bytes");
    // today's args under EAGER
    const uint64_t today = 31376429820ull;   // planned_device with the cache at --cache-bytes
    const uint64_t plan262 = today - cache_args + c262, plan1m = today - cache_args + c1m;
    const auto e262 = a::admit(eager_free, job, gib, plan262), e1m = a::admit(eager_free, job, gib, plan1m);
    std::printf("EAGER: 262K plan %llu vs admitted %llu (slack %lld); 1M plan %llu vs admitted %llu (short %lld)\n",
        (unsigned long long)plan262, (unsigned long long)e262.admitted, (long long)(e262.admitted - plan262),
        (unsigned long long)plan1m, (unsigned long long)e1m.admitted, (long long)(plan1m - e1m.admitted));
    expect(e262.ok, "EAGER 262K f32: today's args fit the measured admission (cache charged exactly)");
    expect(!e1m.ok, "EAGER 1M bf16: today's args do not fit (no component freed; refused loudly, not shrunk)");
    std::printf("%s %d failures\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
