#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// Original DeepSeek-V4.1 Engram representation. This is not a GGML I8
// matrix: every row contains 256 E4M3 values followed by eight E8M0 scales.
// Keep the two ~101 GB tables on disk and fetch only the indexed rows.
namespace llama_dsv41 {

constexpr size_t engram_layers = 2;
constexpr size_t engram_columns = 24;
constexpr size_t engram_width = 256;
constexpr size_t engram_row_bytes = 264;
using engram_rows = std::array<std::array<uint32_t, engram_columns>, engram_layers>;

struct engram_layout {
    std::vector<uint32_t> token_map;
    uint32_t compressed_vocab_size = 0;
    uint32_t pad_id = 0;
    std::array<uint32_t, engram_layers> rows {};
    std::array<std::array<uint64_t, 4>, engram_layers> multipliers {};
    std::array<std::array<uint32_t, engram_columns>, engram_layers> primes {};

    bool validate(std::string & error) const;
};

struct engram_history {
    // Compressed token IDs, newest first. -1 breaks ngrams at an image/mask.
    std::array<int32_t, 3> tail {{-1, -1, -1}};
};

// Validate the whole input before changing history or output. A mask value of
// zero breaks all ngrams through that position; the graph must also suppress
// that position's Engram contribution. layout.validate() is a load-time gate.
bool engram_hash(const engram_layout & layout, engram_history & history,
                 const int32_t * tokens, const uint8_t * mask, size_t count,
                 engram_rows * output, std::string & error);

// Decode with the source model's BF16 rounding, without quantizing/rebuilding
// the original table. Reject E4M3 NaN, reserved E8M0 scale, or overflow.
bool engram_decode_row(const uint8_t * row, float * output, std::string & error);

// A bounded random-read interface, usable by native file I/O or a storage
// queue. It must fill exactly `bytes` or return false. No mmap/table arena.
using engram_read_at = std::function<bool(uint64_t offset, void * dst, size_t bytes)>;
bool engram_read_batch(const engram_read_at & read_at, uint64_t file_bytes,
                       uint64_t table_offset, uint32_t table_rows,
                       const uint32_t * rows, size_t tokens, size_t row_stride,
                       float * output, std::string & error);

} // namespace llama_dsv41
