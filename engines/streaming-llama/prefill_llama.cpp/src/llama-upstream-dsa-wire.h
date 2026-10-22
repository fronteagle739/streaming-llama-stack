#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Convert this fork's GLM-DSA sequence-state payload (one MLA cache with the
// lightning-indexer K rows stored in its state extension) into the current
// upstream llama.cpp sequence-state wire layout:
//
//   upstream envelope + standard MLA cache + standard indexer-K cache
//
// Current upstream also stores lightning-indexer K after an orthonormal
// Walsh-Hadamard rotation, while this fork stores the unrotated rows. The
// converter applies that rotation for F16 and Q8_0 indexer caches.
//
// The conversion is intentionally limited to the MLA/GLM representation:
// there are no V-cache rows and n_pos_per_embd == 1. It fails closed on an
// unexpected extension, cache geometry mismatch, or trailing bytes.
bool llama_upstream_dsa_wire_convert(
        const std::vector<uint8_t> & fork_payload,
        int32_t                      source_seq_id,
        const std::vector<uint32_t> & upstream_indexer_layers,
        std::vector<uint8_t> &       upstream_payload,
        std::string &                error);

// Mirror current upstream's fallback when a GLM GGUF does not carry
// attention.indexer_types. GLM before 5.2 uses a full indexer on every layer;
// 5.2/5.3 use full layers 0,1,2,6,10,... and share between them.
std::vector<uint32_t> llama_upstream_glm_dsa_indexer_layers(
        uint32_t n_layer,
        uint32_t n_ctx_train);
