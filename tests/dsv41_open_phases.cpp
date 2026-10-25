// LLAMADECODE_OPEN_PHASES line format: start order, dotted nesting, notes, unaccounted.
#include "llama-dsv41-open-phases.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

namespace ph = llama_dsv41_open_phases;
static int passed = 0, failed = 0;
static void expect(bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    ok ? ++passed : ++failed;
}
static bool has(const std::string &s, const char *x) { return s.find(x) != std::string::npos; }

int main(int argc, char **argv) {
    const std::string path = std::string(argc > 1 ? argv[1] : "open_phases_test.log");
    if (!std::freopen(path.c_str(), "w", stderr)) return 2;
    {
        ph::session open;
        { ph::scope a("first"); std::this_thread::sleep_for(std::chrono::milliseconds(20));
            { ph::scope b("inner"); ph::note("bytes", 4096); ph::note_ms("virtuallock", 1.5); } }
        { ph::scope c("second"); }
        std::thread([] { ph::scope other("other_thread"); }).join();   // not the opening thread: not recorded
        std::this_thread::sleep_for(std::chrono::milliseconds(10));   // unaccounted
        open.ok = true;
    }
    { ph::scope outside("outside"); }   // no open: not recorded, no line
    { ph::session failed_open; ph::scope x("x"); }
    std::fflush(stderr);
    std::freopen("CON", "w", stderr);
    std::ifstream in(path);
    std::stringstream all; all << in.rdbuf();
    const std::string text = all.str();
    const auto first_line = text.substr(0, text.find('\n'));
    const auto second_line = text.substr(text.find('\n') + 1);
    expect(first_line.rfind("LLAMADECODE_OPEN_PHASES ok=1 total_ms=", 0) == 0, "one line per open, ok=1 on success");
    const auto p1 = first_line.find(" first_ms="), p2 = first_line.find(" first.inner_ms="), p3 = first_line.find(" first.inner.bytes=4096"),
               p4 = first_line.find(" first.inner.virtuallock_ms=1.5"), p5 = first_line.find(" second_ms="), p6 = first_line.find(" unaccounted_ms=");
    expect(p1 != std::string::npos && p1 < p2 && p2 < p3 && p3 < p4 && p4 < p5 && p5 < p6 && p6 != std::string::npos,
           "steps in start order, nested dotted, notes under their scope, unaccounted last");
    expect(!has(text, "other_thread") && !has(text, "outside"), "other threads and calls outside open are not recorded");
    double total = 0, first = 0, second = 0, unaccounted = 0;
    std::sscanf(first_line.c_str() + first_line.find("total_ms=") + 9, "%lf", &total);
    std::sscanf(first_line.c_str() + p1 + 10, "%lf", &first);
    std::sscanf(first_line.c_str() + p5 + 11, "%lf", &second);
    std::sscanf(first_line.c_str() + p6 + 16, "%lf", &unaccounted);
    expect(first >= 19 && unaccounted >= 9 && unaccounted <= total && std::abs(total - first - second - unaccounted) < 0.5,
           "unaccounted = total - top-level steps (nested not double-counted)");
    expect(second_line.rfind("LLAMADECODE_OPEN_PHASES ok=0 ", 0) == 0 && has(second_line, " x_ms="), "a failed open still prints its line (ok=0)");
    std::printf("dsv41_open_phases: %d PASS, %d FAIL\n", passed, failed);
    return failed ? 1 : 0;
}
