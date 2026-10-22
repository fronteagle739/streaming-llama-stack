#pragma once

#include <array>

// Only the canonical Flash-1017 geometry admitted by load_hparams is supported.
// Return the exact native Mac table, rather than recomputing powf on another CRT.
std::array<float, 32> llama_dsv41_rope_frequencies(bool compressed);
