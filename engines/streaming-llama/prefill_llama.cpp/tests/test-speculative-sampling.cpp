#include "speculative-sampling.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static llama_token_data tok(int id, float p) {
    return llama_token_data{ id, 0.0f, p };
}

static bool near(float a, float b) {
    return std::fabs(a - b) < 1e-6f;
}

static void test_accept_probability_uses_target_over_draft() {
    printf("  accept_probability_uses_target_over_draft ... ");
    const std::vector<llama_token_data> target = { tok(1, 0.30f), tok(2, 0.70f) };
    const std::vector<llama_token_data> draft  = { tok(1, 0.60f), tok(2, 0.40f) };

    const auto d = common_speculative_accept_decide(target, draft, 1, 0.49f);

    assert(d.accepted);
    assert(near(d.p_target, 0.30f));
    assert(near(d.p_draft, 0.60f));
    assert(near(d.p_accept, 0.50f));
    printf("OK\n");
}

static void test_reject_when_uniform_exceeds_ratio() {
    printf("  reject_when_uniform_exceeds_ratio ... ");
    const std::vector<llama_token_data> target = { tok(1, 0.30f), tok(2, 0.70f) };
    const std::vector<llama_token_data> draft  = { tok(1, 0.60f), tok(2, 0.40f) };

    const auto d = common_speculative_accept_decide(target, draft, 1, 0.51f);

    assert(!d.accepted);
    assert(near(d.p_accept, 0.50f));
    printf("OK\n");
}

static void test_ratio_clamps_to_one() {
    printf("  ratio_clamps_to_one ... ");
    const std::vector<llama_token_data> target = { tok(1, 0.70f), tok(2, 0.30f) };
    const std::vector<llama_token_data> draft  = { tok(1, 0.20f), tok(2, 0.80f) };

    const auto d = common_speculative_accept_decide(target, draft, 1, 1.0f);

    assert(d.accepted);
    assert(near(d.p_accept, 1.0f));
    printf("OK\n");
}

static void test_residual_subtracts_by_token_id() {
    printf("  residual_subtracts_by_token_id ... ");
    const std::vector<llama_token_data> target = { tok(1, 0.50f), tok(2, 0.50f) };
    const std::vector<llama_token_data> draft  = { tok(2, 0.40f), tok(1, 0.10f) };

    const auto residual = common_speculative_residual_distribution(target, draft);

    assert(residual.size() == 2);
    assert(residual[0].id == 1);
    assert(residual[1].id == 2);
    assert(near(residual[0].p, 0.80f));
    assert(near(residual[1].p, 0.20f));
    printf("OK\n");
}

static void test_residual_omits_negative_mass() {
    printf("  residual_omits_negative_mass ... ");
    const std::vector<llama_token_data> target = { tok(1, 0.20f), tok(2, 0.80f) };
    const std::vector<llama_token_data> draft  = { tok(1, 0.50f), tok(2, 0.25f) };

    const auto residual = common_speculative_residual_distribution(target, draft);

    assert(residual.size() == 1);
    assert(residual[0].id == 2);
    assert(near(residual[0].p, 1.0f));
    printf("OK\n");
}

static void test_sample_distribution_uses_cumulative_mass() {
    printf("  sample_distribution_uses_cumulative_mass ... ");
    const std::vector<llama_token_data> dist = { tok(10, 0.25f), tok(20, 0.50f), tok(30, 0.25f) };

    assert(common_speculative_sample_distribution(dist, 0.00f) == 10);
    assert(common_speculative_sample_distribution(dist, 0.26f) == 20);
    assert(common_speculative_sample_distribution(dist, 0.99f) == 30);
    printf("OK\n");
}

int main() {
    printf("test-speculative-sampling: rejection sampling helpers\n");
    test_accept_probability_uses_target_over_draft();
    test_reject_when_uniform_exceeds_ratio();
    test_ratio_clamps_to_one();
    test_residual_subtracts_by_token_id();
    test_residual_omits_negative_mass();
    test_sample_distribution_uses_cumulative_mass();

    printf("\ntest-speculative-sampling: all tests OK\n");
    return 0;
}
