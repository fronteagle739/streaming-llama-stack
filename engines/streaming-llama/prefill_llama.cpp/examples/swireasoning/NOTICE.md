# Attribution

This project (`llama-swir` + `server-swir.h`) is an independent, training-free
implementation of the **SwiReasoning** decoding method for **llama.cpp**. It builds on,
and is grateful to, the following prior work.

## Method

- **SwiReasoning: Switch-Thinking in Latent and Explicit for Pareto-Superior Reasoning LLMs**
  Dachuan Shi, Abedelkadir Asi, Keying Li, Xiangchi Yuan, Leyan Pan, Wenke Lee, Wen Xiao.
  arXiv:2510.05069 — https://arxiv.org/abs/2510.05069
  Reference implementation: https://github.com/sdc17/SwiReasoning

  The entropy-trend switch state machine and the alpha/beta interpolation here mirror that
  reference (`generation_utils.py` line references are preserved in the code comments). The
  contribution of *this* repo is porting the mechanism — in particular faithful soft-embedding
  (`probs @ E`) injection — into the llama.cpp decode path, which the reference targets via
  HuggingFace/vLLM only.

## Engine

- **llama.cpp** — https://github.com/ggml-org/llama.cpp (MIT). The base inference engine;
  `llama-swir` is a standard llama.cpp example and links only the public `libllama` /
  `common` APIs (plus `src/llama-model.h` to read the `token_embd.weight` tensor).

- **llama-cpp-turboquant** (TheTom) — https://github.com/TheTom/llama-cpp-turboquant — the
  turboquant fork this work was originally developed and validated against (TurboQuant KV-cache
  compression, itself crediting Google's TurboQuant). None of the turboquant KV-compression code
  is included here; credited as the development host.

## Implementation

Soft embeddings use the full vocabulary `probs @ E`, without L2 normalization or
top-k truncation. The alpha/beta switch and overthinking controls share the same
decode path. `--swir-selftest` checks the embedding round trip.
