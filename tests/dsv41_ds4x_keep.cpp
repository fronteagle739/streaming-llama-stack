// LLAMADECODE keep-open: the re-register plan after a prefill weight sweep rebuilt
// the provider banks (ds4 decoder kept open; only moved weights re-registered).
#include "llama-dsv41-ds4x-keep.h"
#include <cstdio>

namespace k = llama_dsv41_ds4x_keep;
static int failures = 0;
static void expect(bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}
static void *p(uintptr_t v) { return reinterpret_cast<void *>(v); }

int main() {
    const std::vector<k::resident> held{{100, 10, p(0x1000)}, {200, 20, p(0x2000)}, {300, 30, p(0x3000)}};
    auto same = k::reregister_plan(held, held);
    expect(same.delta && same.moved.empty() && same.unchanged == 3, "same banks: nothing re-registered");
    auto rebased = held; for (auto &r : rebased) r.device = p(reinterpret_cast<uintptr_t>(r.device) + 0x100000);
    auto all = k::reregister_plan(held, rebased);
    expect(all.delta && all.moved.size() == 3 && all.unchanged == 0, "new heap base: every pointer moved in place");
    auto one = held; one[1].device = p(0x9000);
    auto d1 = k::reregister_plan(held, one);
    expect(d1.delta && d1.moved.size() == 1 && d1.moved[0] == 1 && d1.unchanged == 2, "one moved tensor: only it is replaced");
    auto grown = held; grown.push_back({400, 40, p(0x4000)});
    auto g = k::reregister_plan(held, grown);
    expect(!g.delta && g.moved.size() == 4 && g.unchanged == 0, "tensor added: full drop + register");
    auto resized = held; resized[2].bytes = 31;
    auto r = k::reregister_plan(held, resized);
    expect(!r.delta && r.moved.size() == 3, "size changed: full drop + register");
    auto reordered = held; std::swap(reordered[0], reordered[1]);
    auto o = k::reregister_plan(held, reordered);
    expect(!o.delta && o.moved.size() == 3, "order changed: full drop + register");
    auto first = k::reregister_plan({}, held);
    expect(!first.delta && first.moved.size() == 3, "nothing held (after an x_close): register all");
    auto empty = k::reregister_plan({}, {});
    expect(empty.delta && empty.moved.empty() && empty.unchanged == 0, "empty");
    expect(k::keep_open_from_env(nullptr), "env unset: keep-open ON (default)");
    expect(!k::keep_open_from_env("0"), "env =0: old x_close + x_open path");
    expect(k::keep_open_from_env("1"), "env =1: keep-open ON");
    expect(k::keep_open_from_env(""), "env empty: keep-open ON (default)");
    std::printf("%s %d failures\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
