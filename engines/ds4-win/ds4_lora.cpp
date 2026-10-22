/* Dense-output Safetensors adapter loader; does not write or map base weights. */
#include "ds4_lora.h"
#include "external/nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void ds4_output_lora_free(ds4_output_lora *adapter) {
    if (!adapter) return;
    free(adapter->a); free(adapter->b);
    memset(adapter, 0, sizeof(*adapter));
}

extern "C" int ds4_output_lora_read(const char *path, const char *base_sha256,
        uint32_t hidden, uint32_t vocab, ds4_output_lora *out,
        char *error, size_t error_cap) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    try {
        if (!path || !base_sha256 || strlen(base_sha256) != 64 || !hidden || !vocab)
            throw std::runtime_error("adapter requires a verified base SHA-256 and output geometry");
        for (size_t i = 0; i < 64; ++i)
            if (!strchr("0123456789abcdef", base_sha256[i]))
                throw std::runtime_error("invalid base SHA-256");
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        const auto length = input.tellg();
        if (!input || length < 8 || length > 128 * 1024 * 1024)
            throw std::runtime_error("adapter file missing or outside 128 MiB bound");
        input.seekg(0);
        unsigned char prefix[8]; input.read((char *)prefix, 8);
        uint64_t header_size = 0;
        for (unsigned i = 0; i < 8; ++i) header_size |= uint64_t(prefix[i]) << (8 * i);
        if (!header_size || header_size > 1024 * 1024 || header_size > uint64_t(length) - 8)
            throw std::runtime_error("invalid bounded Safetensors header");
        std::string header(size_t(header_size), '\0');
        input.read(&header[0], std::streamsize(header_size));
        auto j = nlohmann::json::parse(header);
        if (!j.is_object() || j.size() != 3)
            throw std::runtime_error("only lm_head A/B and metadata are supported");
        const auto &metadata = j.at("__metadata__");
        if (metadata.at("slstack_arch").get<std::string>() != "deepseek41" ||
            metadata.at("base_sha256").get<std::string>() != base_sha256)
            throw std::runtime_error("adapter architecture or base SHA-256 mismatch");
        size_t consumed = 0;
        const auto rank_text = metadata.at("rank").get<std::string>();
        const auto rank = std::stoul(rank_text, &consumed);
        if (consumed != rank_text.size() || !rank || rank > 64)
            throw std::runtime_error("adapter rank must be in [1,64]");
        const auto alpha_text = metadata.at("alpha").get<std::string>();
        const auto alpha = std::stof(alpha_text, &consumed);
        if (consumed != alpha_text.size() || !std::isfinite(alpha) || alpha <= 0 || alpha > 4096)
            throw std::runtime_error("adapter alpha outside finite (0,4096] bound");
        struct Span { uint64_t first, last; } spans[2];
        const char *names[] = {"base_model.model.lm_head.lora_A.weight", "base_model.model.lm_head.lora_B.weight"};
        const uint64_t counts[] = {rank * hidden, uint64_t(vocab) * rank};
        float **destinations[] = {&out->a, &out->b};
        for (unsigned t = 0; t < 2; ++t) {
            const auto &tensor = j.at(names[t]);
            const std::vector<uint64_t> wanted = t == 0 ? std::vector<uint64_t>{rank, hidden} : std::vector<uint64_t>{vocab, rank};
            if (tensor.at("dtype") != "F32" || tensor.at("shape").get<std::vector<uint64_t>>() != wanted)
                throw std::runtime_error("adapter dtype or matrix shape mismatch");
            const auto offsets = tensor.at("data_offsets").get<std::vector<uint64_t>>();
            if (offsets.size() != 2 || offsets[0] > offsets[1] ||
                offsets[1] - offsets[0] != counts[t] * sizeof(float) ||
                offsets[1] > uint64_t(length) - 8 - header_size)
                throw std::runtime_error("adapter data span outside file");
            spans[t] = {offsets[0], offsets[1]};
            *destinations[t] = (float *)malloc(size_t(counts[t] * sizeof(float)));
            if (!*destinations[t]) throw std::bad_alloc();
            input.seekg(std::streamoff(8 + header_size + offsets[0]));
            input.read((char *)*destinations[t], std::streamsize(counts[t] * sizeof(float)));
            if (!input) throw std::runtime_error("short adapter tensor read");
            for (uint64_t i = 0; i < counts[t]; ++i)
                if (!std::isfinite((*destinations[t])[i])) throw std::runtime_error("non-finite adapter weight");
        }
        if (spans[0].first > spans[1].first) std::swap(spans[0], spans[1]);
        if (spans[0].first || spans[0].last != spans[1].first ||
            spans[1].last != uint64_t(length) - 8 - header_size)
            throw std::runtime_error("overlapping, sparse or trailing adapter data");
        out->rank = uint32_t(rank); out->alpha = alpha;
        return 1;
    } catch (const std::exception &exc) {
        ds4_output_lora_free(out);
        if (error && error_cap) snprintf(error, error_cap, "%s", exc.what());
        return 0;
    }
}
