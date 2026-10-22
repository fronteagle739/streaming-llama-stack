#include "llama-dsv41-engram.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace llama_dsv41 {
namespace {
bool fail(std::string & error, const char * message) {
    error = message;
    return false;
}
}

bool engram_layout::validate(std::string & error) const {
    if (token_map.empty() || !compressed_vocab_size ||
        compressed_vocab_size > INT32_MAX || pad_id >= compressed_vocab_size) {
        return fail(error, "invalid Engram vocabulary");
    }
    for (uint32_t id : token_map) {
        if (id >= compressed_vocab_size) return fail(error, "Engram token map out of range");
    }
    for (size_t layer = 0; layer < engram_layers; ++layer) {
        for (uint64_t multiplier : multipliers[layer]) {
            if (!(multiplier & 1u) || multiplier > uint64_t(INT64_MAX) / compressed_vocab_size) {
                return fail(error, "invalid Engram hash multiplier");
            }
        }
        uint64_t total = 0;
        for (uint32_t prime : primes[layer]) {
            if (prime < 2) return fail(error, "invalid Engram hash modulus");
            total += prime;
        }
        if (total != rows[layer]) return fail(error, "Engram row count mismatch");
    }
    error.clear();
    return true;
}

bool engram_hash(const engram_layout & l, engram_history & history,
                 const int32_t * tokens, const uint8_t * mask, size_t count,
                 engram_rows * output, std::string & error) {
    if (l.token_map.empty() || !l.compressed_vocab_size ||
        (count && (!tokens || !output)) || count > SIZE_MAX / sizeof(engram_rows)) {
        return fail(error, "invalid Engram hash input");
    }
    for (int32_t id : history.tail) {
        if (id < -1 || (id >= 0 && uint32_t(id) >= l.compressed_vocab_size)) {
            return fail(error, "invalid Engram history");
        }
    }
    for (size_t i = 0; i < count; ++i) {
        if (tokens[i] < 0 || size_t(tokens[i]) >= l.token_map.size() ||
            l.token_map[tokens[i]] >= l.compressed_vocab_size) {
            return fail(error, "Engram token out of range");
        }
    }
    // Check moduli even when the caller has not run the full load-time gate.
    for (const auto & layer : l.primes) {
        for (uint32_t prime : layer) if (prime < 2) return fail(error, "invalid Engram hash modulus");
    }
    for (size_t i = 0; i < count; ++i) {
        const int32_t current = mask && !mask[i] ? -1 : int32_t(l.token_map[tokens[i]]);
        uint32_t ids[4];
        bool blocked = false;
        for (size_t j = 0; j < 4; ++j) {
            const int32_t id = j ? history.tail[j - 1] : current;
            blocked |= id == -1;
            ids[j] = blocked ? l.pad_id : uint32_t(id);
        }
        for (size_t layer = 0; layer < engram_layers; ++layer) {
            uint64_t hash = uint64_t(ids[0]) * l.multipliers[layer][0];
            uint32_t offset = 0;
            for (size_t j = 1; j < 4; ++j) {
                hash ^= uint64_t(ids[j]) * l.multipliers[layer][j];
                for (size_t head = 0; head < 8; ++head) {
                    const size_t column = (j - 1) * 8 + head;
                    const uint32_t prime = l.primes[layer][column];
                    output[i][layer][column] = uint32_t(hash % prime) + offset;
                    offset += prime;
                }
            }
        }
        history.tail = {{current, history.tail[0], history.tail[1]}};
    }
    error.clear();
    return true;
}

bool engram_decode_row(const uint8_t * row, float * output, std::string & error) {
    if (!row || !output) return fail(error, "invalid Engram row buffer");
    // Stage locally: malformed rows never expose a partially decoded vector.
    float decoded[engram_width];
    for (size_t j = 0; j < engram_width; ++j) {
        const uint8_t code = row[j], scale = row[engram_width + j / 32];
        if ((code & 127u) == 127u || scale == 255u) {
            return fail(error, "invalid Engram FP8 value or scale");
        }
        const int exponent = (code >> 3) & 15, mantissa = code & 7;
        float value = exponent ? std::ldexp(float(8 + mantissa), exponent - 10)
                               : std::ldexp(float(mantissa), -9);
        if (code & 128u) value = -value;
        value = std::ldexp(value, int(scale) - 127);
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
        std::memcpy(&value, &bits, sizeof(value));
        if ((bits & 0x7f800000u) == 0x7f800000u) return fail(error, "Engram FP8 row overflow");
        decoded[j] = value;
    }
    std::memcpy(output, decoded, sizeof(decoded));
    error.clear();
    return true;
}

bool engram_read_batch(const engram_read_at & read_at, uint64_t file_bytes,
                       uint64_t table_offset, uint32_t table_rows,
                       const uint32_t * rows, size_t tokens, size_t row_stride,
                       float * output, std::string & error) {
    const uint64_t table_bytes = uint64_t(table_rows) * engram_row_bytes;
    if (!read_at || !table_rows || table_offset > uint64_t(INT64_MAX) ||
        table_bytes > uint64_t(INT64_MAX) - table_offset ||
        table_offset > file_bytes || table_bytes > file_bytes - table_offset ||
        (tokens && (!rows || !output || row_stride < engram_columns)) ||
        tokens > SIZE_MAX / (engram_columns * engram_width * sizeof(float)) ||
        (tokens && tokens - 1 > (SIZE_MAX / sizeof(uint32_t) - engram_columns) / row_stride)) {
        return fail(error, "invalid Engram table range or batch");
    }
    // Validate all indices before any disk read or output write.
    for (size_t i = 0; i < tokens; ++i) {
        for (size_t j = 0; j < engram_columns; ++j) {
            if (rows[i * row_stride + j] >= table_rows) return fail(error, "Engram row out of range");
        }
    }
    struct request { uint32_t row; uint32_t output; };
    constexpr size_t batch_tokens = 2048; // <=384 KiB request workspace
    std::vector<request> requests(std::min(tokens, batch_tokens) * engram_columns);
    uint8_t raw[engram_row_bytes];
    float decoded[engram_width];
    for (size_t start = 0; start < tokens; start += batch_tokens) {
        const size_t count = std::min(tokens - start, batch_tokens) * engram_columns;
        for (size_t i = 0; i < count; ++i) {
            requests[i] = {rows[(start + i / engram_columns) * row_stride + i % engram_columns], uint32_t(i)};
        }
        std::sort(requests.begin(), requests.begin() + count,
                  [](const request & a, const request & b) { return a.row < b.row; });
        uint32_t previous = UINT32_MAX;
        for (size_t i = 0; i < count; ++i) {
            const request & request = requests[i];
            if (request.row != previous) {
                if (!read_at(table_offset + uint64_t(request.row) * engram_row_bytes, raw, sizeof(raw))) {
                    return fail(error, "Engram row read failed");
                }
                if (!engram_decode_row(raw, decoded, error)) return false;
                previous = request.row;
            }
            std::memcpy(output + (start * engram_columns + request.output) * engram_width,
                        decoded, sizeof(decoded));
        }
    }
    error.clear();
    return true;
}
} // namespace llama_dsv41
