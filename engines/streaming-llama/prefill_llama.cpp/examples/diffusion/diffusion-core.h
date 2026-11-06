#pragma once

// Shared diffusion runtime used by BOTH llama-diffusion-cli (one-shot) and llama-diffusion-server
// (persistent). The entropy-bound + self-conditioning denoiser is the PROVEN DiffusionGemma sampler;
// it lives here (compiled once into diffusion-core) so both binaries call the identical bytes.

#include "llama.h"

#include <cstdint>

typedef bool (*diffusion_step_callback_t)(int32_t             step,
                                          int32_t             total_steps,
                                          const llama_token * tokens,
                                          int32_t             n_tokens,
                                          void *              user_data);

// Entropy-bound denoiser params (DiffusionGemma-style block diffusion). Defaults are the proven values.
struct diffusion_eb_params {
    // Default limit is 24 denoising steps; adaptive stopping can finish sooner.
    // Override with LLAMA_DG_STEPS / --diffusion-steps.
    int32_t max_denoising_steps  = 24;
    float   t_min                = 0.4f;   // temperature at the last step
    float   t_max                = 0.8f;   // temperature at the first step
    float   entropy_bound        = 0.1f;   // accept lowest-entropy tokens within this MI bound
    int32_t stability_threshold  = 1;      // steps the argmax canvas must hold to count as stable
    float   confidence_threshold = 0.10f;  // adaptive stop: stop once argmax is stable AND mean canvas entropy
                                           // < this. 0.005 was so strict it NEVER fired (always ran the full
                                           // step cap); 0.10 fires per-prompt (WW1: 15 steps, coherent) — the
                                           // argmax-stability gate is the real convergence signal, this is the
                                           // entropy safety. Prompt-adaptive: harder prompts run more steps.
    int32_t seed                 = 0;
    int32_t max_length           = 0;      // n_input + canvas_length
    bool    self_conditioning    = false;  // feed each step's canvas logits back into the next (DiffusionGemma)
    bool    resume_from_canvas   = false;  // seed the canvas from output_tokens[n_input..] (a prior draft)
                                           // instead of random init — the keystone for guided RE-denoising:
                                           // re-noise only a LLAMA_DG_RENOISE_DEPTH fraction, refine the rest.

    diffusion_step_callback_t step_callback           = nullptr;
    void *                    step_callback_user_data = nullptr;
};

// Entropy-bound denoiser for DiffusionGemma-style block-diffusion canvas models (the model's TRAINED
// scheme). The canvas is RANDOM-initialized; each step samples a candidate per position, accepts the
// lowest-entropy positions within a mutual-information bound, renoises the rest under a linear temperature
// schedule, with adaptive stopping and optional self-conditioning. Writes the stable argmax canvas into
// output_tokens[n_input .. max_length). Mirrors the reference transformers EntropyBoundSampler.
void diffusion_generate_entropy_bound(llama_context *             ctx,
                                      const llama_token *         input_tokens,
                                      llama_token *               output_tokens,
                                      int32_t                     n_input,
                                      const diffusion_eb_params & params,
                                      int32_t &                   n_generated);
