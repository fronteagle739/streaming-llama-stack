// TDD: Prefill strategy selector tests — written before implementation.
// Pure logic tests — no model or GPU needed.

#include "llama-prefill-strategy.h"

#include <cmath>
#include <cstdio>

#undef NDEBUG
#include <cassert>

// ---------------------------------------------------------------------------
// Test: forced strategies ignore prompt length
// ---------------------------------------------------------------------------
static void test_forced_mac() {
    printf("  forced_mac ... ");
    llama_prefill_strategy_params p = {};
    p.strategy = LLAMA_PREFILL_STRATEGY_MAC;

    auto d = llama_prefill_strategy_select(p, 100000, 500.0f);
    assert(d.chosen == LLAMA_PREFILL_STRATEGY_MAC);

    // Even with a huge prompt, forced Mac still wins
    d = llama_prefill_strategy_select(p, 1, 500.0f);
    assert(d.chosen == LLAMA_PREFILL_STRATEGY_MAC);

    printf("OK\n");
}

static void test_forced_rtx() {
    printf("  forced_rtx ... ");
    llama_prefill_strategy_params p = {};
    p.strategy = LLAMA_PREFILL_STRATEGY_RTX;

    auto d = llama_prefill_strategy_select(p, 10, 500.0f);
    assert(d.chosen == LLAMA_PREFILL_STRATEGY_RTX);

    printf("OK\n");
}

// ---------------------------------------------------------------------------
// Test: auto mode picks Mac for short prompts
// ---------------------------------------------------------------------------
static void test_auto_short_prompt_picks_mac() {
    printf("  auto_short_prompt_picks_mac ... ");
    llama_prefill_strategy_params p = {};
    p.strategy = LLAMA_PREFILL_STRATEGY_AUTO;
    p.mac_prefill_tok_s = 300.0f;
    p.pcie_bandwidth_gbs = 64.0f;

    // 500 GB model on PCIe 5.0 = ~7.8s streaming floor
    // At 300 tok/s Mac rate, crossover = 7.8 * 300 = ~2340 tokens
    // 100 tokens << crossover → Mac
    auto d = llama_prefill_strategy_select(p, 100, 500.0f);
    assert(d.chosen == LLAMA_PREFILL_STRATEGY_MAC);
    assert(d.crossover_tokens > 0);

    printf("OK\n");
}

// ---------------------------------------------------------------------------
// Test: auto mode picks RTX for long prompts
// ---------------------------------------------------------------------------
static void test_auto_long_prompt_picks_rtx() {
    printf("  auto_long_prompt_picks_rtx ... ");
    llama_prefill_strategy_params p = {};
    p.strategy = LLAMA_PREFILL_STRATEGY_AUTO;
    p.mac_prefill_tok_s = 300.0f;
    p.pcie_bandwidth_gbs = 64.0f;

    // Same 500 GB model, crossover ~2340 tokens
    // 8000 tokens >> crossover → RTX
    auto d = llama_prefill_strategy_select(p, 8000, 500.0f);
    assert(d.chosen == LLAMA_PREFILL_STRATEGY_RTX);

    printf("OK\n");
}

// ---------------------------------------------------------------------------
// Test: explicit crossover override
// ---------------------------------------------------------------------------
static void test_explicit_crossover() {
    printf("  explicit_crossover ... ");
    llama_prefill_strategy_params p = {};
    p.strategy = LLAMA_PREFILL_STRATEGY_AUTO;
    p.crossover_tokens = 1000;

    // 999 tokens < 1000 → Mac
    auto d = llama_prefill_strategy_select(p, 999, 500.0f);
    assert(d.chosen == LLAMA_PREFILL_STRATEGY_MAC);
    assert(d.crossover_tokens == 1000);

    // 1000 tokens >= 1000 → RTX
    d = llama_prefill_strategy_select(p, 1000, 500.0f);
    assert(d.chosen == LLAMA_PREFILL_STRATEGY_RTX);
    assert(d.crossover_tokens == 1000);

    printf("OK\n");
}

// ---------------------------------------------------------------------------
// Test: small model has low streaming floor → low crossover
// ---------------------------------------------------------------------------
static void test_small_model_low_crossover() {
    printf("  small_model_low_crossover ... ");
    llama_prefill_strategy_params p = {};
    p.strategy = LLAMA_PREFILL_STRATEGY_AUTO;
    p.mac_prefill_tok_s = 500.0f;
    p.pcie_bandwidth_gbs = 64.0f;

    // 5 GB model: floor = 5/64 = 0.078s, crossover = 0.078 * 500 = 39 tokens
    // But minimum crossover is 100
    auto d = llama_prefill_strategy_select(p, 50, 5.0f);
    assert(d.crossover_tokens >= 100);  // clamped minimum

    printf("OK\n");
}

// ---------------------------------------------------------------------------
// Test: estimates are reasonable
// ---------------------------------------------------------------------------
static void test_estimates_reasonable() {
    printf("  estimates_reasonable ... ");
    llama_prefill_strategy_params p = {};
    p.strategy = LLAMA_PREFILL_STRATEGY_AUTO;
    p.mac_prefill_tok_s = 200.0f;
    p.pcie_bandwidth_gbs = 64.0f;
    p.rtx_compute_tok_s = 2000.0f;

    auto d = llama_prefill_strategy_select(p, 4000, 500.0f);

    // Mac estimate: 4000/200 * 1000 = 20000 ms
    assert(d.estimated_mac_ms > 19000.0f && d.estimated_mac_ms < 21000.0f);

    // RTX estimate: max(floor, compute) = max(500/64*1000, 4000/2000*1000) = max(7812, 2000) = 7812 ms
    assert(d.estimated_rtx_ms > 7000.0f && d.estimated_rtx_ms < 8500.0f);

    printf("OK\n");
}

// ---------------------------------------------------------------------------
// Test: default params produce sane results
// ---------------------------------------------------------------------------
static void test_default_params() {
    printf("  default_params ... ");
    llama_prefill_strategy_params p = {};  // all defaults

    // Should not crash, should produce a decision
    auto d = llama_prefill_strategy_select(p, 1000, 100.0f);
    assert(d.chosen == LLAMA_PREFILL_STRATEGY_MAC || d.chosen == LLAMA_PREFILL_STRATEGY_RTX);
    assert(d.reason != nullptr);
    assert(d.crossover_tokens > 0);

    printf("OK\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("test-prefill-strategy: forced mode tests\n");
    test_forced_mac();
    test_forced_rtx();

    printf("test-prefill-strategy: auto mode tests\n");
    test_auto_short_prompt_picks_mac();
    test_auto_long_prompt_picks_rtx();
    test_explicit_crossover();
    test_small_model_low_crossover();
    test_estimates_reasonable();
    test_default_params();

    printf("\ntest-prefill-strategy: all tests OK\n");
    return 0;
}
