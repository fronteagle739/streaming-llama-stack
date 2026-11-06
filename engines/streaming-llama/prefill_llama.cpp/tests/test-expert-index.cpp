#include "llama-expert-index.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

constexpr int N_LAYERS = 43;
constexpr int N_EXPERTS = 256;

constexpr std::array<uint64_t, 3> EXPECTED_SPANS = {
    2752512,
    2752512,
    3604480,
};

constexpr std::array<const char *, 3> ROLE_NAMES = {
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
};

#define REQUIRE(condition, message) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "FAIL: %s:%d: %s\n", __func__, __LINE__, message); \
        return false; \
    } \
} while (0)

struct ggml_context_deleter {
    void operator()(ggml_context * ctx) const {
        ggml_free(ctx);
    }
};

struct gguf_context_deleter {
    void operator()(gguf_context * ctx) const {
        gguf_free(ctx);
    }
};

bool test_synthetic_model_index() {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 3 + 16 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    std::unique_ptr<ggml_context, ggml_context_deleter> ctx(ggml_init(params));
    REQUIRE(ctx != nullptr, "create GGML metadata context");

    const std::array<ggml_tensor *, 3> tensors = {
        ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q2_K, 4096, 2048, N_EXPERTS),
        ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q2_K, 4096, 2048, N_EXPERTS),
        ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q3_K, 2048, 4096, N_EXPERTS),
    };

    constexpr std::array<int, 3> sample_layers = {0, 21, 42};
    constexpr std::array<int, 4> sample_experts = {0, 1, 128, 255};
    for (int layer : sample_layers) {
        for (int expert : sample_experts) {
            for (size_t role = 0; role < 3; ++role) {
                const ggml_tensor * tensor = tensors[role];
                const uint64_t base = (UINT64_C(1) << 20) +
                                      static_cast<uint64_t>(layer * 3 + role) * (UINT64_C(1) << 30);
                const int file_idx = (layer + static_cast<int>(role)) % 3;
                expert_slab_loc loc = {};
                REQUIRE(dsv4_expert_slab_loc_from_geometry(
                                file_idx, base, tensor->nb[2], ggml_nbytes(tensor), expert,
                                EXPECTED_SPANS[role], &loc),
                        "index sampled expert geometry");
                REQUIRE(loc.span_bytes == EXPECTED_SPANS[role], "exact S0 role span");
                REQUIRE(loc.span_bytes == tensor->nb[2], "span equals independent GGML nb[2]");
                REQUIRE(
                        loc.file_offset == base + static_cast<uint64_t>(expert) * tensor->nb[2],
                        "offset equals base plus expert times nb[2]");
                REQUIRE(loc.file_idx == file_idx, "role keeps its shard index");

                if (expert < N_EXPERTS - 1) {
                    expert_slab_loc next = {};
                    REQUIRE(dsv4_expert_slab_loc_from_geometry(
                                    file_idx, base, tensor->nb[2], ggml_nbytes(tensor), expert + 1,
                                    EXPECTED_SPANS[role], &next),
                            "index consecutive expert");
                    REQUIRE(
                            next.file_offset == loc.file_offset + loc.span_bytes,
                            "consecutive experts are contiguous");
                    REQUIRE(next.file_idx == loc.file_idx, "consecutive experts stay in one shard");
                }
            }
        }
    }

    expert_slab_loc ignored = {};
    REQUIRE(!dsv4_expert_slab_loc_from_geometry(0, 0, EXPECTED_SPANS[0], EXPECTED_SPANS[0] * N_EXPERTS,
                                                 -1, EXPECTED_SPANS[0], &ignored),
            "reject negative expert");
    REQUIRE(!dsv4_expert_slab_loc_from_geometry(0, 0, EXPECTED_SPANS[0], EXPECTED_SPANS[0] * N_EXPERTS,
                                                 N_EXPERTS, EXPECTED_SPANS[0], &ignored),
            "reject expert past DSV4 range");
    REQUIRE(!dsv4_expert_slab_loc_from_geometry(0, 0, EXPECTED_SPANS[0] + 1, EXPECTED_SPANS[0] * N_EXPERTS,
                                                 1, EXPECTED_SPANS[0], &ignored),
            "reject stride mismatch");
    REQUIRE(!dsv4_expert_slab_loc_from_geometry(0, 0, EXPECTED_SPANS[0], EXPECTED_SPANS[0] * (N_EXPERTS - 1),
                                                 1, EXPECTED_SPANS[0], &ignored),
            "reject packed tensor size mismatch");

    std::fprintf(stderr, "test_synthetic_model_index: PASS\n");
    return true;
}

bool test_real_gguf_headers(const char * path_value) {
    namespace fs = std::filesystem;

    fs::path input(path_value);
    std::vector<fs::path> shards;
    std::error_code ec;
    if (fs::is_directory(input, ec)) {
        for (const fs::directory_entry & entry : fs::directory_iterator(input)) {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                shards.push_back(entry.path());
            }
        }
    } else {
        shards.push_back(input);
    }
    std::sort(shards.begin(), shards.end());
    REQUIRE(!shards.empty(), "find GGUF shards");

    struct parsed_shard {
        fs::path path;
        std::unique_ptr<ggml_context, ggml_context_deleter> tensor_ctx;
        std::unique_ptr<gguf_context, gguf_context_deleter> gguf_ctx;
    };
    std::vector<parsed_shard> parsed;
    for (const fs::path & path : shards) {
        ggml_context * tensor_ctx = nullptr;
        gguf_init_params params = {
            /*.no_alloc =*/ true,
            /*.ctx      =*/ &tensor_ctx,
        };
        gguf_context * gguf_ctx = gguf_init_from_file(path.string().c_str(), params);
        REQUIRE(gguf_ctx != nullptr && tensor_ctx != nullptr, "parse GGUF header without tensor payload");
        parsed.push_back({
            path,
            std::unique_ptr<ggml_context, ggml_context_deleter>(tensor_ctx),
            std::unique_ptr<gguf_context, gguf_context_deleter>(gguf_ctx),
        });
    }

    constexpr std::array<int, 3> sample_layers = {0, 21, 42};
    constexpr std::array<int, 4> sample_experts = {0, 1, 128, 255};
    std::set<size_t> sampled_file_indices;
    for (int layer : sample_layers) {
        for (size_t role = 0; role < ROLE_NAMES.size(); ++role) {
            const std::string name = "blk." + std::to_string(layer) + "." + ROLE_NAMES[role] + ".weight";
            bool found = false;
            for (size_t file_idx = 0; file_idx < parsed.size(); ++file_idx) {
                const int64_t tensor_id = gguf_find_tensor(parsed[file_idx].gguf_ctx.get(), name.c_str());
                if (tensor_id < 0) {
                    continue;
                }

                const ggml_tensor * tensor = ggml_get_tensor(parsed[file_idx].tensor_ctx.get(), name.c_str());
                REQUIRE(tensor != nullptr, "GGUF tensor geometry exists");
                REQUIRE(tensor->ne[2] == N_EXPERTS, "GGUF expert dimension is 256");
                REQUIRE(tensor->nb[2] == EXPECTED_SPANS[role], "GGUF nb[2] equals exact S0 span");
                REQUIRE(gguf_get_tensor_size(parsed[file_idx].gguf_ctx.get(), tensor_id) == EXPECTED_SPANS[role] * N_EXPERTS,
                        "GGUF packed tensor size equals 256 expert spans");

                const uint64_t base = gguf_get_data_offset(parsed[file_idx].gguf_ctx.get()) +
                                      gguf_get_tensor_offset(parsed[file_idx].gguf_ctx.get(), tensor_id);
                REQUIRE(base + gguf_get_tensor_size(parsed[file_idx].gguf_ctx.get(), tensor_id) <=
                                fs::file_size(parsed[file_idx].path),
                        "packed tensor stays inside its GGUF shard");
                for (int expert : sample_experts) {
                    const uint64_t offset = base + static_cast<uint64_t>(expert) * tensor->nb[2];
                    REQUIRE(offset + tensor->nb[2] <= base + gguf_get_tensor_size(parsed[file_idx].gguf_ctx.get(), tensor_id),
                            "sampled slab stays inside packed tensor");
                }
                sampled_file_indices.insert(file_idx);
                found = true;
                break;
            }
            REQUIRE(found, name.c_str());
        }
    }
    REQUIRE(sampled_file_indices.size() == parsed.size(), "sample matrix covers all GGUF shard boundaries");

    std::fprintf(stderr, "test_real_gguf_headers: PASS (%zu shards, header-only)\n", parsed.size());
    return true;
}

} // namespace

int main() {
    if (!test_synthetic_model_index()) {
        return 1;
    }

    const char * real_gguf = std::getenv("LLAMA_DSV4_EXPERT_INDEX_GGUF");
    if (real_gguf != nullptr && real_gguf[0] != '\0') {
        if (!test_real_gguf_headers(real_gguf)) {
            return 1;
        }
    } else {
        std::fprintf(stderr, "test_real_gguf_headers: SKIP (set LLAMA_DSV4_EXPERT_INDEX_GGUF)\n");
    }

    std::fprintf(stderr, "PASS: DSV4 expert-slab index CPU tests\n");
    return 0;
}
