#include "speculative-sampling.h"

#include <algorithm>
#include <cmath>

static float common_speculative_clean_prob(float p) {
    return std::isfinite(p) && p > 0.0f ? p : 0.0f;
}

float common_speculative_probability_for_token(
        const std::vector<llama_token_data> & dist,
        llama_token token) {
    for (const auto & item : dist) {
        if (item.id == token) {
            return common_speculative_clean_prob(item.p);
        }
    }
    return 0.0f;
}

float common_speculative_probability_for_token(
        const llama_token_data_array & dist,
        llama_token token) {
    for (size_t i = 0; i < dist.size; ++i) {
        if (dist.data[i].id == token) {
            return common_speculative_clean_prob(dist.data[i].p);
        }
    }
    return 0.0f;
}

common_speculative_accept_decision common_speculative_accept_decide(
        const std::vector<llama_token_data> & target_dist,
        const std::vector<llama_token_data> & draft_dist,
        llama_token draft_token,
        float uniform01) {
    common_speculative_accept_decision result;
    result.p_target = common_speculative_probability_for_token(target_dist, draft_token);
    result.p_draft  = common_speculative_probability_for_token(draft_dist, draft_token);

    if (result.p_target <= 0.0f || result.p_draft <= 0.0f) {
        result.p_accept = 0.0f;
        result.accepted = false;
        return result;
    }

    result.p_accept = std::min(1.0f, result.p_target / result.p_draft);
    result.accepted = uniform01 <= result.p_accept;
    return result;
}

std::vector<llama_token_data> common_speculative_residual_distribution(
        const std::vector<llama_token_data> & target_dist,
        const std::vector<llama_token_data> & draft_dist) {
    std::vector<llama_token_data> residual;
    residual.reserve(target_dist.size());

    float sum = 0.0f;
    for (const auto & target : target_dist) {
        llama_token_data item = target;
        const float p_target = common_speculative_clean_prob(target.p);
        const float p_draft  = common_speculative_probability_for_token(draft_dist, target.id);
        item.p = std::max(0.0f, p_target - p_draft);
        if (item.p > 0.0f) {
            sum += item.p;
            residual.push_back(item);
        }
    }

    if (sum <= 0.0f) {
        residual.clear();
        return residual;
    }

    for (auto & item : residual) {
        item.p /= sum;
    }
    return residual;
}

llama_token common_speculative_sample_distribution(
        const std::vector<llama_token_data> & dist,
        float uniform01) {
    float sum = 0.0f;
    for (const auto & item : dist) {
        sum += common_speculative_clean_prob(item.p);
    }
    if (sum <= 0.0f) {
        return LLAMA_TOKEN_NULL;
    }

    const float u = std::min(1.0f, std::max(0.0f, uniform01));
    const float threshold = u * sum;
    float cumulative = 0.0f;
    llama_token last = LLAMA_TOKEN_NULL;

    for (const auto & item : dist) {
        const float p = common_speculative_clean_prob(item.p);
        if (p <= 0.0f) {
            continue;
        }
        cumulative += p;
        last = item.id;
        if (threshold <= cumulative) {
            return item.id;
        }
    }

    return last;
}
