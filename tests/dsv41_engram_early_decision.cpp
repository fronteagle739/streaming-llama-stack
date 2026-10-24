// LLAMADECODE_ENGRAM_EARLY=auto: the pure early-start decision
// (planned requests / LLAMADECODE_ENGRAM_IOPS <= gap) and its env parsing,
// checked against the bin-eg gate's measured request counts.
#include "llama-dsv41-engram-early-decision.h"
#include <cstdio>
#include <stdexcept>
#include <string>

namespace ed = llama_dsv41_engram_early;
static int failures = 0;
static void expect(bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}
template <class F> static bool throws(F f) {
    try { f(); } catch (const std::invalid_argument &) { return true; }
    return false;
}

int main() {
    expect(ed::parse_mode(nullptr) == ed::mode::automatic, "unset LLAMADECODE_ENGRAM_EARLY is auto");
    expect(ed::parse_mode("") == ed::mode::automatic, "empty is auto");
    expect(ed::parse_mode("auto") == ed::mode::automatic, "auto");
    expect(ed::parse_mode("0") == ed::mode::off, "0 forces off");
    expect(ed::parse_mode("1") == ed::mode::on, "1 forces on");
    expect(ed::parse_mode("pause") == ed::mode::pause, "pause forces the pause arm");
    expect(throws([] { ed::parse_mode("true"); }) && throws([] { ed::parse_mode("2"); }) && throws([] { ed::parse_mode("AUTO"); }),
           "other values are an error, not a silent off");

    expect(ed::parse_iops(nullptr) == 100000.0 && ed::parse_iops("") == 100000.0, "IOPS default 100000");
    expect(ed::parse_iops("250000") == 250000.0, "IOPS override");
    expect(throws([] { ed::parse_iops("0"); }) && throws([] { ed::parse_iops("12k"); }) && throws([] { ed::parse_iops("-5"); }) &&
           throws([] { ed::parse_iops("999"); }), "IOPS out of range or malformed is an error");

    // Measured requests (bin-eg gate, 2026-10-24).
    const auto r16 = ed::decide(488608, ed::default_iops, ed::cold_gap_s);
    expect(r16.early && r16.requests == 488608 && r16.predicted_s > 4.88 && r16.predicted_s < 4.89 && r16.gap_s == 5.0,
           "16K cold (488,608 requests, 4.89 s <= 5 s): early (measured -5 s)");
    const auto r43 = ed::decide(1419303, ed::default_iops, ed::cold_gap_s);
    expect(!r43.early && r43.predicted_s > 14.1, "43K cold (1.42M requests, 14.2 s > 5 s): does not fit (measured +3 s early)");
    expect(r16.chosen == ed::arm::early && r43.chosen == ed::arm::pause, "auto arm: 16K early, 43K pause (not the late load)");
    expect(std::string(ed::arm_name(ed::arm::early)) == "early" && std::string(ed::arm_name(ed::arm::pause)) == "pause", "arm names");
    // 23.7K: 651,121 requests (ordinary planner, coalesced ~5% fewer) -> 6.2-6.5 s: ordinary.
    expect(!ed::decide(620000, ed::default_iops, ed::cold_gap_s).early, "23.7K cold: ordinary");
    // Warm append: gap = the completed pass's layer-0 wall (545-token base: 0.601 s).
    expect(!ed::decide(1418962, ed::default_iops, 0.601).early, "43K warm append after a 545-token base: ordinary");
    const auto small = ed::decide(4734, ed::default_iops, 0.601);
    expect(small.early && small.predicted_s < 0.05, "545-token append (4,734 requests): early");
    expect(ed::decide(60000, ed::default_iops, 0.6).early && !ed::decide(60001, ed::default_iops, 0.6).early, "boundary: predicted == gap starts");
    expect(!ed::decide(0, ed::default_iops, 0).early && !ed::decide(10, ed::default_iops, -1).early, "no measured gap: ordinary");
    expect(ed::decide(1419303, 300000, ed::cold_gap_s).early, "a faster measured rate admits 43K");
    expect(throws([] { ed::decide(1, 0, 1); }), "zero rate rejected");

    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
