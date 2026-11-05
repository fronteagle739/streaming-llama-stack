#pragma once
// QFN: reader for the Basalt .basalt pack of Qwen3.8 Flash Next (the container is described in Basalt's
// include/basalt/core/model_format.hpp, MIT). Only the pieces this engine streams or keeps resident are exposed:
//   * the `gguf` section: every non-expert source tensor at an absolute file offset (dense weights, the n-gram table)
//   * the `experts` section: raw [gate | up | down] rows, plus NVFP4's 16-byte FP32 scale tail
//   * `mtp/*`: Basalt's tuned MTP drafter (dense tensors by HF name; experts in Basalt's plane layout)
//   * `profile`: (layer, expert) pairs ranked by routing frequency
// Payload reads are limited to small text/JSON and the explicitly described activation calibration.
#include "ggml.h"

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct qfn_tensor {
    std::string name;
    ggml_type   type = GGML_TYPE_COUNT;
    int         n_dims = 0;
    int64_t     ne[4] = {1, 1, 1, 1};
    uint64_t    offset = 0;   // absolute file offset
    uint64_t    bytes  = 0;
};

struct qfn_hparams {
    uint32_t n_layer = 0, n_embd = 0, n_vocab = 0, n_ctx_train = 0;
    uint32_t n_head = 0, n_head_kv = 0, head_dim = 0;
    uint32_t n_rot = 0;
    int      rope_sections[4] = {0, 0, 0, 0};
    float    rope_freq_base = 0, rms_eps = 0;
    uint32_t n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_ff_shexp = 0;
    uint32_t ssm_conv = 0, ssm_state = 0, ssm_group = 0, ssm_dt_rank = 0, ssm_inner = 0;
    uint32_t full_attn_interval = 0;
    uint32_t hc = 0, hc_low_rank = 0;
    uint32_t idx_heads = 0, idx_dim = 0, idx_top_k = 0, kpool = 0;
    std::vector<uint32_t> compress_ratios;
    // PLE n-gram embedding
    std::vector<uint32_t> ple_layers;
    uint32_t ple_ngram = 0, ple_heads_per_ngram = 0, ple_conv = 0, ple_dim = 0, ple_n_heads = 0;
    uint32_t ple_eos = 0, ple_image = 0;
    std::vector<uint64_t> ple_mult, ple_head_off, ple_head_vocab;
    // tokenizer ids
    uint32_t tok_bos = 0, tok_eos = 0, tok_pad = 0;
    bool is_recurrent(uint32_t il) const { return (il + 1) % full_attn_interval != 0; }
    bool is_ple(uint32_t il) const { for (uint32_t p : ple_layers) if (p == il) return true; return false; }
};

struct qfn_expert_layer {
    // Preserve on-disk type IDs independently of this tree's ggml enum. Basalt type 40 is NVFP4;
    // ggml has no NVFP4 kernels here yet, so gu_type/d_type remain COUNT for that descriptor.
    uint32_t gu_type_id = GGML_TYPE_COUNT, d_type_id = GGML_TYPE_COUNT;
    ggml_type gu_type = GGML_TYPE_COUNT, d_type = GGML_TYPE_COUNT;
    uint64_t  offset = 0;      // absolute file offset of expert 0's blob
    uint64_t  blob = 0;        // bytes per expert: gate | up | down [| scale tail]
    uint64_t  gate_bytes = 0, down_bytes = 0;
    uint64_t  tail_offset = 0, tail_bytes = 0;  // relative to each blob; FP32 {s_gate, s_up, s_down, 0}
    bool is_nvfp4() const { return gu_type_id == 40 || d_type_id == 40; }
};

struct qfn_pack {
    bool native_nvfp4_packed = false;
    std::string  path;
    std::wstring wpath;
    uint64_t     file_bytes = 0;
    uint32_t     version = 0, expert_layout_version = 0;
    std::map<std::string, std::pair<uint64_t, uint64_t>> sections;  // name -> (offset, bytes)
    std::map<std::string, qfn_tensor> tensors;                       // the gguf section
    std::vector<qfn_expert_layer> experts;                            // [n_layer]
    qfn_hparams hp;
    // Opaque checkpoint metadata never enters the ordinary GGUF tensor map.
    // Full JSON and section extents preserve unknown provenance/calibration fields.
    std::string checkpoint_json, expert_input_scale_index_json;
    std::string expert_input_scale_sha256;
    uint64_t expert_input_scale_offset = 0;
    std::vector<float> expert_input_scale; // raw LE_F32 [layer,expert,gate/up/down], NOT reciprocals
    std::vector<int32_t> profile_rank;                                // [n_layer * n_expert], rank or -1
    std::vector<std::pair<uint16_t, uint16_t>> profile_pairs;         // (layer, expert) in rank order
    std::string chat_template;
    std::vector<std::string> vocab;                                   // token strings (tokenizer.ggml.tokens)
    std::vector<std::string> merges;                                  // tokenizer.ggml.merges
    std::vector<int32_t> token_type;                                  // tokenizer.ggml.token_type
    std::string tok_model, tok_pre;                                   // tokenizer.ggml.model / .pre
    std::string tok_normalizer_json; // exact source JSON string; absent remains empty
    bool tok_normalize_nfc = false;
    bool add_bos = false;
    // Authenticated source generation policy is independent of tokenizer token roles.
    std::string generation_config_json, generation_defaults_json = "{}";
    bool generation_eos_explicit = false, generation_bos_explicit = false, generation_pad_explicit = false;
    int32_t generation_bos_id = -1, generation_pad_id = -1;
    std::vector<int32_t> generation_eos;
    // Also used by model-free strict metadata fixtures. Empty raw means legacy source absence.
    void configure_generation(const std::string & raw, const std::string & gguf_defaults = "{}", const std::string & base_defaults = "{}");
    std::vector<int32_t> eos_tokens() const;
    static bool generation_self_test(std::string & err);
    // MTP drafter (present when the pack carries mtp/*): dense tensors by HF name at absolute offsets, the expert
    // section (Basalt plane layout, one blob per expert), the draft head's vocabulary
    std::map<std::string, qfn_tensor> mtp_tensors;
    uint64_t mtp_experts_offset = 0, mtp_expert_blob = 0;
    qfn_expert_layer mtp_expert_layout;  // the copied Q2_0 drafter can differ from NVFP4 main experts
    std::vector<int32_t> draft_vocab;

    bool open(const std::string & path, std::string & err);
    // Parser support is distinct from runtime support; call before CUDA/cache/weight allocations.
    bool runtime_supported(std::string & err) const;
    // Explicit bounded inspection only: open() never reads expert payloads.
    bool expert_scale_tail(uint32_t layer, uint32_t expert, std::array<float, 4> & scales, std::string & err) const;
    const qfn_tensor * find(const std::string & name) const;
    const qfn_tensor & get(const std::string & name) const;           // throws when absent
    std::pair<uint64_t, uint64_t> section(const std::string & name) const;  // throws when absent
};
