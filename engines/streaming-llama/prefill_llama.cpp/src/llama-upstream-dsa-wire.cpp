#include "llama-upstream-dsa-wire.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {

constexpr uint32_t upstream_sequence_magic = 0xaf143cd8u;
constexpr uint32_t state_ext_bit = 0x80000000u;
constexpr uint32_t state_ext_has_k_idx = 0x00000001u;
constexpr int32_t ggml_type_f16 = 1;
constexpr int32_t ggml_type_q8_0 = 8;
constexpr size_t q8_0_values_per_block = 32;
constexpr size_t q8_0_block_bytes = sizeof(uint16_t) + q8_0_values_per_block;

struct byte_span {
    size_t offset = 0;
    size_t size   = 0;
};

struct layer_span {
    byte_span encoded;
    int32_t type = 0;
    uint64_t row_size = 0;
    size_t payload_offset = 0;
};

struct stream_layout {
    uint32_t cell_count = 0;
    byte_span metadata;
    uint32_t v_trans = 0;
    uint32_t n_layer = 0;
    std::vector<layer_span> main_layers;
    std::vector<layer_span> indexer_layers_by_main_layer;
    std::vector<bool> indexer_present;
};

class checked_reader {
public:
    explicit checked_reader(const std::vector<uint8_t> & bytes) : bytes(bytes) {}

    template <typename T>
    T read(const char * label) {
        require(sizeof(T), label);
        T value {};
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        offset += sizeof(value);
        return value;
    }

    byte_span take(size_t size, const char * label) {
        require(size, label);
        byte_span result { offset, size };
        offset += size;
        return result;
    }

    size_t position() const { return offset; }

private:
    void require(size_t size, const char * label) const {
        if (size > bytes.size() - offset) {
            throw std::runtime_error(std::string("truncated GLM DSA state while reading ") + label);
        }
    }

    const std::vector<uint8_t> & bytes;
    size_t offset = 0;
};

size_t checked_payload_size(uint64_t row_size, uint32_t cell_count, const char * label) {
    if (row_size > std::numeric_limits<size_t>::max() ||
        (cell_count != 0 && row_size > std::numeric_limits<size_t>::max() / cell_count)) {
        throw std::runtime_error(std::string("overflow in GLM DSA ") + label + " payload size");
    }
    return static_cast<size_t>(row_size) * cell_count;
}

layer_span read_layer(checked_reader & reader, uint32_t cell_count, const char * label) {
    const size_t start = reader.position();
    const int32_t type = reader.read<int32_t>(label);
    const uint64_t row_size = reader.read<uint64_t>(label);
    const byte_span payload = reader.take(checked_payload_size(row_size, cell_count, label), label);
    return { { start, reader.position() - start }, type, row_size, payload.offset };
}

template <typename T>
void append(std::vector<uint8_t> & dst, T value) {
    const auto * bytes = reinterpret_cast<const uint8_t *>(&value);
    dst.insert(dst.end(), bytes, bytes + sizeof(value));
}

void append_span(std::vector<uint8_t> & dst, const std::vector<uint8_t> & src, byte_span span) {
    dst.insert(dst.end(), src.begin() + static_cast<std::ptrdiff_t>(span.offset),
               src.begin() + static_cast<std::ptrdiff_t>(span.offset + span.size));
}

float f16_to_f32(uint16_t half) {
    const uint32_t sign = static_cast<uint32_t>(half & 0x8000u) << 16;
    uint32_t exponent = (half >> 10) & 0x1fu;
    uint32_t mantissa = half & 0x03ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            uint32_t shift = 0;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x03ffu;
            bits = sign | ((127u - 14u - shift) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint16_t f32_to_f16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    const uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) {
        return static_cast<uint16_t>(sign | (magnitude > 0x7f800000u ? 0x7e00u : 0x7c00u));
    }

    int32_t exponent = static_cast<int32_t>(magnitude >> 23) - 127 + 15;
    uint32_t mantissa = magnitude & 0x007fffffu;
    if (exponent <= 0) {
        if (exponent < -10) {
            return sign;
        }
        mantissa |= 0x00800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exponent);
        uint32_t rounded = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) {
            ++rounded;
        }
        return static_cast<uint16_t>(sign | rounded);
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }

    uint32_t rounded = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
        ++rounded;
        if (rounded == 0x0400u) {
            rounded = 0;
            ++exponent;
            if (exponent >= 31) {
                return static_cast<uint16_t>(sign | 0x7c00u);
            }
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint16_t>(exponent) << 10) | rounded);
}

void hadamard_rotate(std::vector<float> & row) {
    const size_t n = row.size();
    if (n == 0 || (n & (n - 1)) != 0) {
        throw std::runtime_error("current-upstream GLM DSA indexer row is not a power of two");
    }
    for (size_t stride = 1; stride < n; stride *= 2) {
        for (size_t base = 0; base < n; base += stride * 2) {
            for (size_t i = 0; i < stride; ++i) {
                const float a = row[base + i];
                const float b = row[base + stride + i];
                row[base + i] = a + b;
                row[base + stride + i] = a - b;
            }
        }
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(n));
    for (float & value : row) {
        value *= scale;
    }
}

void append_upstream_indexer_layer(
        std::vector<uint8_t> & dst,
        const std::vector<uint8_t> & src,
        const layer_span & layer,
        uint32_t cell_count) {
    size_t n = 0;
    if (layer.type == ggml_type_f16 && layer.row_size % sizeof(uint16_t) == 0) {
        n = static_cast<size_t>(layer.row_size / sizeof(uint16_t));
    } else if (layer.type == ggml_type_q8_0 && layer.row_size % q8_0_block_bytes == 0) {
        n = static_cast<size_t>(layer.row_size / q8_0_block_bytes) * q8_0_values_per_block;
    } else {
        throw std::runtime_error("current-upstream GLM DSA Hadamard conversion supports only F16 or Q8_0 indexer K");
    }
    std::vector<float> row(n);
    append<int32_t>(dst, layer.type);
    append<uint64_t>(dst, layer.row_size);
    for (uint32_t cell = 0; cell < cell_count; ++cell) {
        const size_t row_offset = layer.payload_offset + static_cast<size_t>(cell) * layer.row_size;
        if (layer.type == ggml_type_f16) {
            for (size_t i = 0; i < n; ++i) {
                uint16_t half = 0;
                std::memcpy(&half, src.data() + row_offset + i * sizeof(half), sizeof(half));
                row[i] = f16_to_f32(half);
            }
        } else {
            const size_t block_count = n / q8_0_values_per_block;
            for (size_t block = 0; block < block_count; ++block) {
                const size_t block_offset = row_offset + block * q8_0_block_bytes;
                uint16_t half_scale = 0;
                std::memcpy(&half_scale, src.data() + block_offset, sizeof(half_scale));
                const float scale = f16_to_f32(half_scale);
                for (size_t i = 0; i < q8_0_values_per_block; ++i) {
                    const int8_t quant = static_cast<int8_t>(src[block_offset + sizeof(uint16_t) + i]);
                    row[block * q8_0_values_per_block + i] = scale * quant;
                }
            }
        }
        hadamard_rotate(row);
        if (layer.type == ggml_type_f16) {
            for (float value : row) {
                append<uint16_t>(dst, f32_to_f16(value));
            }
        } else {
            const size_t block_count = n / q8_0_values_per_block;
            for (size_t block = 0; block < block_count; ++block) {
                float amax = 0.0f;
                for (size_t i = 0; i < q8_0_values_per_block; ++i) {
                    amax = std::max(amax, std::fabs(row[block * q8_0_values_per_block + i]));
                }
                const float scale = amax / 127.0f;
                append<uint16_t>(dst, f32_to_f16(scale));
                const float inverse_scale = scale == 0.0f ? 0.0f : 1.0f / scale;
                for (size_t i = 0; i < q8_0_values_per_block; ++i) {
                    const float scaled = row[block * q8_0_values_per_block + i] * inverse_scale;
                    const int quant = std::max(-127, std::min(127, static_cast<int>(std::round(scaled))));
                    append<int8_t>(dst, static_cast<int8_t>(quant));
                }
            }
        }
    }
}

void append_cache(
        std::vector<uint8_t> & dst,
        const std::vector<uint8_t> & src,
        const std::vector<stream_layout> & streams,
        const std::vector<uint32_t> * indexer_layers) {
    append<uint32_t>(dst, static_cast<uint32_t>(streams.size()));
    for (const auto & stream : streams) {
        append<uint32_t>(dst, stream.cell_count);
        if (stream.cell_count == 0) {
            continue;
        }

        append_span(dst, src, stream.metadata);
        append<uint32_t>(dst, stream.v_trans);
        if (indexer_layers) {
            append<uint32_t>(dst, static_cast<uint32_t>(indexer_layers->size()));
            for (uint32_t layer : *indexer_layers) {
                append_upstream_indexer_layer(
                    dst, src, stream.indexer_layers_by_main_layer[layer], stream.cell_count);
            }
        } else {
            append<uint32_t>(dst, static_cast<uint32_t>(stream.main_layers.size()));
            for (const layer_span & layer : stream.main_layers) {
                append_span(dst, src, layer.encoded);
            }
        }
    }
}

} // namespace

bool llama_upstream_dsa_wire_convert(
        const std::vector<uint8_t> & fork_payload,
        int32_t source_seq_id,
        const std::vector<uint32_t> & upstream_indexer_layers,
        std::vector<uint8_t> & upstream_payload,
        std::string & error) {
    upstream_payload.clear();
    error.clear();

    try {
        checked_reader reader(fork_payload);
        const uint32_t n_stream = reader.read<uint32_t>("stream count");
        if (n_stream == 0 || n_stream > 1024) {
            throw std::runtime_error("invalid GLM DSA stream count");
        }

        std::vector<stream_layout> streams;
        streams.reserve(n_stream);
        uint32_t expected_n_layer = 0;
        uint32_t expected_v_trans = 0;
        std::vector<bool> expected_indexer_present;
        bool have_geometry = false;

        for (uint32_t stream_index = 0; stream_index < n_stream; ++stream_index) {
            stream_layout stream;
            stream.cell_count = reader.read<uint32_t>("cell count");
            if (stream.cell_count == 0) {
                streams.push_back(std::move(stream));
                continue;
            }

            const size_t metadata_start = reader.position();
            for (uint32_t cell = 0; cell < stream.cell_count; ++cell) {
                (void) reader.read<int32_t>("cell position");
                const uint32_t n_seq_id = reader.read<uint32_t>("cell sequence count");
                if (n_seq_id > 1024) {
                    throw std::runtime_error("invalid GLM DSA cell sequence count");
                }
                (void) reader.take(static_cast<size_t>(n_seq_id) * sizeof(int32_t), "cell sequence ids");
            }
            stream.metadata = { metadata_start, reader.position() - metadata_start };

            stream.v_trans = reader.read<uint32_t>("V transposition flag");
            const uint32_t n_layer_word = reader.read<uint32_t>("layer count");
            if ((n_layer_word & state_ext_bit) == 0) {
                throw std::runtime_error("GLM DSA state has no lightning-indexer extension");
            }
            stream.n_layer = n_layer_word & ~state_ext_bit;
            if (stream.n_layer == 0 || stream.n_layer > 1024) {
                throw std::runtime_error("invalid GLM DSA layer count");
            }

            const uint32_t state_ext_flags = reader.read<uint32_t>("state extension flags");
            if ((state_ext_flags & state_ext_has_k_idx) == 0) {
                throw std::runtime_error("GLM DSA state has no lightning-indexer data");
            }
            if ((state_ext_flags & ~state_ext_has_k_idx) != 0) {
                throw std::runtime_error("unsupported GLM DSA state extension flags");
            }

            stream.main_layers.reserve(stream.n_layer);
            for (uint32_t layer = 0; layer < stream.n_layer; ++layer) {
                stream.main_layers.push_back(read_layer(reader, stream.cell_count, "MLA K layer"));
            }

            stream.indexer_present.reserve(stream.n_layer);
            stream.indexer_layers_by_main_layer.resize(stream.n_layer);
            for (uint32_t layer = 0; layer < stream.n_layer; ++layer) {
                const bool present = reader.read<uint32_t>("indexer presence") != 0;
                stream.indexer_present.push_back(present);
                if (present) {
                    stream.indexer_layers_by_main_layer[layer] =
                        read_layer(reader, stream.cell_count, "indexer K layer");
                }
            }
            if (!stream.indexer_present.empty() &&
                    std::find(stream.indexer_present.begin(), stream.indexer_present.end(), true) ==
                        stream.indexer_present.end()) {
                throw std::runtime_error("GLM DSA state contains no lightning-indexer layers");
            }

            if (!have_geometry) {
                expected_n_layer = stream.n_layer;
                expected_v_trans = stream.v_trans;
                expected_indexer_present = stream.indexer_present;
                have_geometry = true;
            } else if (stream.n_layer != expected_n_layer ||
                       stream.v_trans != expected_v_trans ||
                       stream.indexer_present != expected_indexer_present) {
                throw std::runtime_error("GLM DSA cache geometry differs between streams");
            }

            streams.push_back(std::move(stream));
        }

        if (!have_geometry) {
            throw std::runtime_error("GLM DSA state has no populated streams");
        }
        if (reader.position() != fork_payload.size()) {
            throw std::runtime_error("unexpected trailing bytes in GLM DSA state (V rows or incompatible format)");
        }

        if (upstream_indexer_layers.empty()) {
            throw std::runtime_error("current-upstream GLM DSA indexer layer list is empty");
        }
        uint32_t previous = 0;
        bool first = true;
        for (uint32_t layer : upstream_indexer_layers) {
            if (layer >= expected_n_layer || (!first && layer <= previous)) {
                throw std::runtime_error("invalid current-upstream GLM DSA indexer layer list");
            }
            for (const auto & stream : streams) {
                if (stream.cell_count != 0 && !stream.indexer_present[layer]) {
                    throw std::runtime_error("fork state is missing a current-upstream GLM DSA indexer layer");
                }
            }
            previous = layer;
            first = false;
        }

        // The output duplicates only compact cell metadata and component headers;
        // all large K payload spans are copied exactly once.
        upstream_payload.reserve(fork_payload.size() + 8 + n_stream * 32);
        append<uint32_t>(upstream_payload, upstream_sequence_magic);
        append<int32_t>(upstream_payload, source_seq_id);
        append_cache(upstream_payload, fork_payload, streams, nullptr);
        append_cache(upstream_payload, fork_payload, streams, &upstream_indexer_layers);
        return true;
    } catch (const std::exception & ex) {
        upstream_payload.clear();
        error = ex.what();
        return false;
    }
}

std::vector<uint32_t> llama_upstream_glm_dsa_indexer_layers(
        uint32_t n_layer,
        uint32_t n_ctx_train) {
    std::vector<uint32_t> layers;
    layers.reserve(n_layer);
    const bool pre_glm_52 = n_ctx_train < 1048576u;
    for (uint32_t layer = 0; layer < n_layer; ++layer) {
        const bool full = pre_glm_52 || layer < 3u || ((layer - 2u) % 4u == 0u);
        if (full) {
            layers.push_back(layer);
        }
    }
    return layers;
}
