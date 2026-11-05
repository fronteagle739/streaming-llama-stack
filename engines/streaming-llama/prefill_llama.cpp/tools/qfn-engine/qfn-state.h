#pragma once
// Complete Mac ds4 Qwen payload v2, family 0x51573802. Readers expose ggml owners.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include "../../../../ds4-mac/qfn-native-layout.h"
namespace qfn_state {
struct shape {
    uint32_t context, tile, rows, layers, embd, hc, head, kv_heads, vocab;
    uint32_t interval, ssm_dim, ssm_heads, conv_channels, conv_history;
    uint32_t index_dim, pool, ple_history, eos;
    bool mtp;
    uint32_t mtp_rows = UINT32_MAX; // UINT32_MAX uses the full frontier
};
inline uint32_t draft_rows(const shape &s) { return s.mtp ? (s.mtp_rows == UINT32_MAX ? s.rows : s.mtp_rows) : 0; }
using reader = std::function<void(const std::string &, uint32_t, uint64_t, void *, size_t)>;
using half_to_float = std::function<float(uint16_t)>;
using writer = std::function<void(const void *, size_t)>;
using position_reader=std::function<std::array<int32_t,4>(uint32_t)>;
using input = std::function<void(void *,size_t)>;
inline qfn_native_shape native_shape(const shape &s) {
    return {s.layers,s.interval,s.embd,s.hc,s.head,s.kv_heads,s.vocab,s.ssm_dim,s.ssm_heads,
        s.conv_channels,s.conv_history,s.index_dim,s.pool,s.ple_history,s.context,uint32_t(s.mtp)};
}
inline void require(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
inline uint64_t payload_bytes(const shape &s, bool narrow_index = false) {
    require(s.rows && s.rows <= s.context && s.layers && s.layers <= 256 &&
        s.embd && s.embd <= 65536 && s.hc && s.hc <= 64 && s.head && s.head <= 4096 &&
        s.kv_heads && s.kv_heads <= 256 && s.vocab && s.vocab <= 1000000 &&
        s.interval && s.ssm_dim && s.ssm_dim <= 1024 && s.ssm_heads && s.ssm_heads <= 1024 &&
        s.conv_channels && s.conv_channels <= 1000000 && s.conv_history && s.conv_history <= 64 &&
        s.index_dim && s.index_dim <= 4096 && s.pool == 4 && s.ple_history && s.ple_history <= 64 &&
        s.eos < s.vocab && draft_rows(s) <= s.rows, "Invalid QFN/Mac state geometry");
    uint64_t n = 52 + uint64_t(s.rows) * 4 + uint64_t(s.vocab) * 4 + 4;
    for (uint32_t il = 0; il < s.layers + uint32_t(s.mtp); ++il) {
        if (il < s.layers && (il + 1) % s.interval)
            n += uint64_t(s.ssm_dim) * s.ssm_dim * s.ssm_heads * 4 + uint64_t(s.conv_history) * s.conv_channels * 4;
        else {
            const uint32_t rows = il == s.layers ? draft_rows(s) : s.rows;
            n += uint64_t(rows) * s.kv_heads * s.head * 4 +
                 uint64_t(rows) * s.index_dim * (narrow_index ? 2 : 4) + uint64_t(rows / 4) * s.index_dim * 2;
        }
    }
    return n + uint64_t(s.ple_history) * s.embd * s.hc * 4 + 32 + 4 + uint64_t(s.rows) * 16;
}
inline void write_to(const writer &out, const shape &s, const std::vector<int32_t> &tokens,
                  const std::vector<float> &logits, uint64_t max_bytes,
                  const reader &read, const half_to_float &decode_half, bool narrow_index = false,
                  int32_t rotary_delta=0,const position_reader &positions={}) {
    const uint64_t expected = payload_bytes(s, narrow_index);
    require(out && expected <= max_bytes, "QFN state exceeds explicit payload byte budget");
    require(tokens.size() == s.rows && logits.size() == s.vocab, "QFN snapshot token/logit extent differs");
    for (int32_t t : tokens) require(t >= 0 && uint32_t(t) < s.vocab, "Invalid snapshot token");
    for (float f : logits) require(std::isfinite(f), "Nonfinite snapshot logits");
    const uint32_t endian = 1;
    require(*reinterpret_cast<const uint8_t *>(&endian) == 1 && sizeof(float) == 4, "QFN state requires LE IEEE FP32");
    uint64_t written = 0;
    auto put = [&](const void *p, size_t bytes) {
        require(bytes <= expected - written, "QFN state write exceeds extent");
        out(p, bytes);
        written += bytes;
    };
    auto u32 = [&](uint32_t n) { uint8_t b[4] = {uint8_t(n), uint8_t(n >> 8), uint8_t(n >> 16), uint8_t(n >> 24)}; put(b, 4); };
    const uint32_t h[] = {0x34565344, 2, s.context, s.tile, s.context, s.context,
        s.embd * s.hc, s.rows, s.layers + uint32_t(s.mtp), s.head, s.index_dim, s.vocab, 0x51573802};
    for (auto n : h) u32(n);
    for (int32_t t : tokens) u32(uint32_t(t));
    put(logits.data(), logits.size() * 4);
    u32(draft_rows(s)); // caller flushes pending MTP owners first
    std::vector<uint8_t> buf(1 << 20);
    auto span = [&](const char *owner, uint32_t layer, uint64_t bytes) {
        for (uint64_t off = 0; off < bytes;) {
            const size_t n = size_t(std::min<uint64_t>(buf.size(), bytes - off));
            read(owner, layer, off, buf.data(), n); put(buf.data(), n); off += n;
        }
    };
    // ggml histories [channel][time] -> Metal histories [time][channel].
    auto history = [&](const char *owner, uint32_t layer, uint32_t time, uint32_t channels) {
        const uint32_t block = std::min<uint32_t>(4096, uint32_t(buf.size() / (time * 4)));
        std::vector<float> row(block);
        for (uint32_t t = 0; t < time; ++t) for (uint32_t c = 0; c < channels; c += block) {
            const uint32_t n = std::min(block, channels - c);
            read(owner, layer, uint64_t(c) * time * 4, buf.data(), size_t(n) * time * 4);
            for (uint32_t j = 0; j < n; ++j) std::memcpy(&row[j], buf.data() + (size_t(j) * time + t) * 4, 4);
            put(row.data(), size_t(n) * 4);
        }
    };
    for (uint32_t il = 0; il < s.layers + uint32_t(s.mtp); ++il) {
        if (il < s.layers && (il + 1) % s.interval) {
            span("gdn", il, uint64_t(s.ssm_dim) * s.ssm_dim * s.ssm_heads * 4);
            history("conv", il, s.conv_history, s.conv_channels);
        } else {
            const uint32_t rows = il == s.layers ? draft_rows(s) : s.rows;
            span("key", il, uint64_t(rows) * s.kv_heads * s.head * 2);
            span("value", il, uint64_t(rows) * s.kv_heads * s.head * 2);
            const uint64_t elements = uint64_t(rows) * s.index_dim;
            std::vector<uint16_t> half(1 << 18); std::vector<float> full(narrow_index ? 0 : half.size());
            for (uint64_t off = 0; off < elements;) {
                const size_t n = size_t(std::min<uint64_t>(half.size(), elements - off));
                read("index", il, off * 2, half.data(), n * 2);
                if (narrow_index) put(half.data(), n * 2);
                else {
                    for (size_t j = 0; j < n; ++j) full[j] = decode_half(half[j]);
                    put(full.data(), n * 4);
                }
                off += n;
            }
            span("pool", il, uint64_t(rows / 4) * s.index_dim * 2);
        }
    }
    history("ple", 0, s.ple_history, s.embd * s.hc);
    for (uint32_t i = 0; i < 8; ++i) u32(i < s.rows ? uint32_t(tokens[s.rows - 1 - i]) : s.eos);
    u32(uint32_t(rotary_delta));
    for (uint32_t i = 0; i < s.rows; ++i) {
        if(positions){const auto row=positions(i);for(auto value:row)u32(uint32_t(value));}
        else {u32(i);u32(i);u32(i);u32(0);}
    }
    require(written == expected, "QFN snapshot byte count differs");
}
inline void write(FILE *out, const shape &s, const std::vector<int32_t> &tokens,
                  const std::vector<float> &logits, uint64_t max_bytes,
                  const reader &read, const half_to_float &decode_half,
                  int32_t rotary_delta=0,const position_reader &positions={}) {
    require(out != nullptr, "QFN snapshot file is null");
    write_to([&](const void *p,size_t n) { require(fwrite(p,1,n,out)==n,"QFN state write failed"); },
             s,tokens,logits,max_bytes,read,decode_half,false,rotary_delta,positions);
    require(fflush(out)==0,"QFN snapshot flush failed");
}
} // namespace qfn_state
