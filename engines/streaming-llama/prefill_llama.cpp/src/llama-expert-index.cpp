#include "llama-expert-index.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <array>
#include <filesystem>
#include <string>

namespace {

constexpr int DSV4_MOE_LAYERS = 43;
constexpr int DSV4_EXPERTS    = 256;

// S0 decode recon, verified against the DSV4-flash Q2_K-XL GGUF headers:
// gate/up are Q2_K and down is Q3_K. One routed-expert slab totals 9,109,504 bytes.
constexpr std::array<uint64_t, 3> DSV4_EXPERT_SPANS = {
    2752512,
    2752512,
    3604480,
};

constexpr std::array<const char *, 3> DSV4_EXPERT_ROLE_NAMES = {
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
};

bool get_shard_path(const llama_model * model, int file_idx, std::filesystem::path & path) {
    if (!model->split_file_paths.empty()) {
        if (file_idx < 0 || static_cast<size_t>(file_idx) >= model->split_file_paths.size()) {
            LLAMA_LOG_ERROR("%s: shard index %d is outside split_file_paths\n", __func__, file_idx);
            return false;
        }
        path = model->split_file_paths[file_idx];
        return true;
    }

    if (file_idx != 0 || model->primary_model_path.empty()) {
        return false;
    }

    path = model->primary_model_path;
    return true;
}

} // namespace

bool dsv4_expert_slab_index(
        const llama_model * model,
        int                 layer,
        int                 expert,
        expert_slab_loc     out[3]) {
    if (model == nullptr || out == nullptr) {
        LLAMA_LOG_ERROR("%s: model and output must be non-null\n", __func__);
        return false;
    }
    if (model->arch != LLM_ARCH_DEEPSEEK4) {
        LLAMA_LOG_ERROR("%s: model architecture is not deepseek4\n", __func__);
        return false;
    }
    if (layer < 0 || layer >= DSV4_MOE_LAYERS || static_cast<size_t>(layer) >= model->layers.size()) {
        LLAMA_LOG_ERROR("%s: layer %d is outside the DSV4 MoE range [0, %d)\n", __func__, layer, DSV4_MOE_LAYERS);
        return false;
    }
    if (expert < 0 || expert >= DSV4_EXPERTS) {
        LLAMA_LOG_ERROR("%s: expert %d is outside the DSV4 range [0, %d)\n", __func__, expert, DSV4_EXPERTS);
        return false;
    }

    const llama_layer & model_layer = model->layers[layer];
    const std::array<const ggml_tensor *, 3> tensors = {
        model_layer.ffn_gate_exps,
        model_layer.ffn_up_exps,
        model_layer.ffn_down_exps,
    };
    std::array<expert_slab_loc, 3> result = {};

    for (size_t role = 0; role < tensors.size(); ++role) {
        const ggml_tensor * tensor = tensors[role];
        if (tensor == nullptr) {
            LLAMA_LOG_ERROR(
                    "%s: tensor blk.%d.%s.weight was not found in the model\n",
                    __func__, layer, DSV4_EXPERT_ROLE_NAMES[role]);
            return false;
        }

        const std::string tensor_name = tensor->name;
        const auto loc_it = model->weight_file_locs.find(tensor_name);
        if (loc_it == model->weight_file_locs.end()) {
            LLAMA_LOG_ERROR("%s: tensor %s was not found in weight_file_locs\n", __func__, tensor_name.c_str());
            return false;
        }

        const uint64_t stride = tensor->nb[2];
        if (tensor->ne[2] != DSV4_EXPERTS || stride != DSV4_EXPERT_SPANS[role]) {
            LLAMA_LOG_ERROR(
                    "%s: tensor %s has expert geometry ne[2]=%lld nb[2]=%llu; expected %d and %llu\n",
                    __func__, tensor_name.c_str(), static_cast<long long>(tensor->ne[2]),
                    static_cast<unsigned long long>(stride), DSV4_EXPERTS,
                    static_cast<unsigned long long>(DSV4_EXPERT_SPANS[role]));
            return false;
        }

        const uint64_t tensor_bytes = ggml_nbytes(tensor);
        if (tensor_bytes != stride * DSV4_EXPERTS) {
            LLAMA_LOG_ERROR(
                    "%s: tensor %s size %llu does not equal nb[2] * %d\n",
                    __func__, tensor_name.c_str(), static_cast<unsigned long long>(tensor_bytes), DSV4_EXPERTS);
            return false;
        }

        if (!dsv4_expert_slab_loc_from_geometry(
                    loc_it->second.file_idx,
                    loc_it->second.file_offset,
                    stride,
                    tensor_bytes,
                    expert,
                    DSV4_EXPERT_SPANS[role],
                    &result[role])) {
            LLAMA_LOG_ERROR("%s: tensor %s has invalid expert-slab offset geometry\n", __func__, tensor_name.c_str());
            return false;
        }

        const uint64_t slab_begin = result[role].file_offset;
        const uint64_t slab_end   = slab_begin + result[role].span_bytes;
        const uint64_t tensor_end = static_cast<uint64_t>(loc_it->second.file_offset) + tensor_bytes;

        std::filesystem::path shard_path;
        if (get_shard_path(model, loc_it->second.file_idx, shard_path)) {
            std::error_code ec;
            const uint64_t shard_end = std::filesystem::file_size(shard_path, ec);
            if (ec) {
                LLAMA_LOG_ERROR("%s: could not stat shard %s: %s\n", __func__, shard_path.string().c_str(), ec.message().c_str());
                return false;
            }
            if (tensor_end > shard_end || slab_end > shard_end) {
                LLAMA_LOG_ERROR(
                        "%s: tensor %s expert %d range [%llu, %llu) crosses shard %s end %llu\n",
                        __func__, tensor_name.c_str(), expert,
                        static_cast<unsigned long long>(slab_begin), static_cast<unsigned long long>(slab_end),
                        shard_path.string().c_str(), static_cast<unsigned long long>(shard_end));
                return false;
            }
        } else if (!model->primary_model_path.empty() || !model->split_file_paths.empty()) {
            LLAMA_LOG_ERROR("%s: no path for shard index %u used by tensor %s\n", __func__, loc_it->second.file_idx, tensor_name.c_str());
            return false;
        }
    }

    for (size_t role = 0; role < result.size(); ++role) {
        out[role] = result[role];
    }
    return true;
}
