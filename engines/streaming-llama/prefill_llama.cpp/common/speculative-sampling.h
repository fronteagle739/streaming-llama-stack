#pragma once

#include "llama.h"

#include <vector>

struct common_speculative_accept_decision {
    bool  accepted = false;
    float p_target = 0.0f;
    float p_draft  = 0.0f;
    float p_accept = 0.0f;
};

float common_speculative_probability_for_token(
        const std::vector<llama_token_data> & dist,
        llama_token token);

float common_speculative_probability_for_token(
        const llama_token_data_array & dist,
        llama_token token);

common_speculative_accept_decision common_speculative_accept_decide(
        const std::vector<llama_token_data> & target_dist,
        const std::vector<llama_token_data> & draft_dist,
        llama_token draft_token,
        float uniform01);

std::vector<llama_token_data> common_speculative_residual_distribution(
        const std::vector<llama_token_data> & target_dist,
        const std::vector<llama_token_data> & draft_dist);

llama_token common_speculative_sample_distribution(
        const std::vector<llama_token_data> & dist,
        float uniform01);
