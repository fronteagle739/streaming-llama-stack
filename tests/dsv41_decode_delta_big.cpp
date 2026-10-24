// rtxmac-producer-2: one apply_decode_delta may carry a whole local append
// (up to decode_delta_max_tokens = 49152). Byte parity: a single big delta and
// the same rows as a stream of <=128-token deltas rebuild the identical state,
// equal to the canonical target, with equal owner/token/logit SHA256s. The
// receiver's default (decode 128) cap still refuses a big delta; the explicit
// maximum accepts it and refuses one token more.
#include "llama-dsv41-artifact.h"
#include "live-session-checkpoint.h"
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

namespace ld = llama_dsv41;
static int failures = 0;
static void expect(bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}
static constexpr uint32_t ctx = 262144, vocab = 129280;
static uint64_t row_bytes(ld::state_kind k) {
    switch (k) {
        case ld::state_kind::raw_window: return 512u * 4u;
        case ld::state_kind::dspark_hidden: return 3u * 5120u * 4u;
        case ld::state_kind::compressed: return 512u * 4u;
        case ld::state_kind::index: return 128u * 4u;
        default: return 0;
    }
}
static uint8_t mix(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    uint64_t x = a * 0x9E3779B97F4A7C15ull ^ (b + 0x632BE59BD9B4E019ull) * 0xBF58476D1CE4E5B9ull ^ c * 0x94D049BB133111EBull ^ d;
    x ^= x >> 31; x *= 0xD6E8FEB86659FD93ull; x ^= x >> 29;
    return uint8_t(x);
}
// Canonical synthetic state at frontier position p: ring slot r holds the row
// of the latest position q < p with q % 128 == r; compressed/index row i is
// fixed once written; pool tails depend on the whole position.
static void canonical(uint32_t p, const ld::state_span &s, uint64_t off, void *dst, size_t n) {
    auto *o = static_cast<uint8_t *>(dst);
    const uint64_t rb = row_bytes(s.kind);
    for (size_t i = 0; i < n; ++i) {
        const uint64_t at = off + i;
        uint64_t key = 0;
        if (s.kind == ld::state_kind::raw_window || s.kind == ld::state_kind::dspark_hidden) {
            const uint64_t slot = at / rb;
            key = (p - 1) - ((p - 1 + 128 - slot) % 128); // latest q < p with q % 128 == slot
        } else if (s.kind == ld::state_kind::compressed || s.kind == ld::state_kind::index) {
            key = at / rb;
        } else {
            key = p;
        }
        o[i] = mix(uint64_t(s.kind) * 64 + s.index, key, at % (rb ? rb : 1u << 20), rb ? 0 : at);
    }
}
static ld::wire_frontier frontier(uint32_t p, uint32_t mode) { return {ctx, p, mode, mode == 3 ? p : 0, mode == 3 ? std::min(p, 128u) : 0}; }
static std::vector<int32_t> token_ids(uint32_t n) { std::vector<int32_t> t(n); for (uint32_t i = 0; i < n; ++i) t[i] = int32_t((i * 7919u) % vocab); return t; }
static std::vector<float> logits_at(uint32_t p) { std::vector<float> l(vocab); for (uint32_t i = 0; i < vocab; ++i) l[i] = float((i + p) % 977) * 0.01f; return l; }

using key = std::pair<int, uint32_t>;
struct state {
    uint32_t position = 0;
    std::map<key, std::vector<uint8_t>> spans;
};
static state build(uint32_t p, uint32_t mode) {
    ld::wire_plan plan; std::string error; state s; s.position = p;
    if (!ld::make_wire_plan(frontier(p, mode), plan, error)) throw std::runtime_error(error);
    for (const auto &sp : plan.spans) { auto &v = s.spans[{int(sp.kind), sp.index}]; v.resize(size_t(sp.bytes)); canonical(p, sp, 0, v.data(), v.size()); }
    return s;
}
// One delta base -> target through sender/receiver; applied onto s.
static bool apply(state &s, uint32_t mode, uint32_t target, uint32_t limit, std::string &error, bool apply_state = true) {
    const auto base_f = frontier(s.position, mode), target_f = frontier(target, mode);
    const auto tokens = token_ids(target); const auto logits = logits_at(target);
    ld::wire_payload_reader source;
    if (!source.init(target_f, tokens.data(), tokens.size(), logits.data(), logits.size(),
            [&](const ld::state_span &sp, uint64_t off, void *dst, size_t n) { canonical(target, sp, off, dst, n); return true; }, error)) return false;
    ld::delta_artifact_sender sender;
    if (!sender.init(base_f, target_f, 17, 23, source, error)) return false;
    ld::delta_artifact_receiver receiver;
    auto stream = [&](uint64_t off, void *dst, size_t n, std::string &why) { return sender.read(off, dst, n, why); };
    if (!receiver.init(sender.bytes(), stream, base_f, 17, 23, error, limit)) return false;
    if (receiver.tokens() != std::vector<int32_t>(tokens.begin() + s.position, tokens.end()) || receiver.logits() != logits) { error = "token/logit mismatch"; return false; }
    if (!apply_state) return true;
    ld::wire_plan plan; ld::make_wire_plan(target_f, plan, error);
    std::map<key, std::vector<uint8_t>> next;
    for (const auto &sp : plan.spans) { auto &v = next[{int(sp.kind), sp.index}]; auto it = s.spans.find({int(sp.kind), sp.index});
        if (it != s.spans.end()) v = it->second; v.resize(size_t(sp.bytes), 0xEE); }
    for (const auto &slice : receiver.plan().slices) {
        auto &v = next[{int(slice.span.kind), slice.span.index}];
        for (uint64_t off = 0; off < slice.bytes;) {
            const size_t n = size_t(std::min<uint64_t>(65536, slice.bytes - off));
            if (!receiver.read_span(slice.span, slice.offset + off, v.data() + slice.offset + off, n, error)) return false;
            off += n;
        }
    }
    s.spans = std::move(next); s.position = target; return true;
}
static dsv41_live::checkpoint identity(const state &s, uint32_t mode) {
    const auto tokens = token_ids(s.position); const auto logits = logits_at(s.position); std::string error;
    ld::wire_payload_reader r;
    if (!r.init(frontier(s.position, mode), tokens.data(), tokens.size(), logits.data(), logits.size(),
            [&](const ld::state_span &sp, uint64_t off, void *dst, size_t n) { const auto &v = s.spans.at({int(sp.kind), sp.index});
                if (off > v.size() || n > v.size() - off) return false; std::memcpy(dst, v.data() + off, n); return true; }, error))
        throw std::runtime_error(error);
    return dsv41_live::capture(r, {});
}

int main() {
    std::printf("decode_delta_max_tokens=%u chunk=%u byte_cap(128)=%llu byte_cap(49152)=%llu\n", ld::decode_delta_max_tokens, ld::decode_delta_chunk_tokens,
        (unsigned long long)ld::decode_delta_byte_cap(128), (unsigned long long)ld::decode_delta_byte_cap(49152));
    for (uint32_t mode : {1u, 3u})
        for (auto range : {std::pair<uint32_t, uint32_t>{5, 134}, {127, 1000}, {300, 301 + 4096}, {1, 1 + 49152}}) {
            if (mode == 3 && range.second - range.first > 4097) continue; // mode 3 big case covered at 4096
            const uint32_t b = range.first, t = range.second;
            char what[160];
            state big = build(b, mode), chunked = build(b, mode);
            std::string error;
            ld::delta_plan plan; ld::make_delta_plan(frontier(b, mode), frontier(t, mode), plan, error);
            const uint64_t artifact = 204 + (mode == 3 ? 60 : 52) + uint64_t(t - b) * 4 + uint64_t(vocab) * 4 + plan.state_bytes;
            std::snprintf(what, sizeof what, "mode=%u %u->%u bytes=%llu within byte cap", mode, b, t, (unsigned long long)artifact);
            expect(artifact <= ld::decode_delta_byte_cap(t - b), what);
            bool ok = apply(big, mode, t, ld::decode_delta_max_tokens, error);
            std::snprintf(what, sizeof what, "mode=%u %u->%u single delta (%u tokens) accepted at the advertised maximum", mode, b, t, t - b);
            expect(ok, what); if (!ok) std::printf("  error: %s\n", error.c_str());
            uint32_t pieces = 0;
            while (ok && chunked.position < t) {
                ok = apply(chunked, mode, std::min(t, chunked.position + ld::decode_delta_chunk_tokens), ld::decode_delta_chunk_tokens, error); ++pieces;
            }
            std::snprintf(what, sizeof what, "mode=%u %u->%u same rows as %u deltas of <=128", mode, b, t, pieces);
            expect(ok, what);
            const state truth = build(t, mode);
            std::snprintf(what, sizeof what, "mode=%u %u->%u big == chunked == canonical target, byte for byte", mode, b, t);
            expect(ok && big.spans == chunked.spans && big.spans == truth.spans, what);
            if (ok) {
                const auto a = identity(big, mode), c = identity(chunked, mode), z = identity(truth, mode);
                std::snprintf(what, sizeof what, "mode=%u %u->%u owner/token/logit SHA256 equal", mode, b, t);
                expect(a.describe() == c.describe() && a.describe() == z.describe() && a.owners == z.owners, what);
            }
            if (t - b > 128) {
                state probe = build(b, mode);
                std::snprintf(what, sizeof what, "mode=%u %u->%u default decode cap (128) still refuses it", mode, b, t);
                expect(!apply(probe, mode, t, 128, error, false), what);
            }
        }
    {
        state probe = build(1, 1); std::string error;
        expect(!apply(probe, 1, 2 + ld::decode_delta_max_tokens, ld::decode_delta_max_tokens, error, false), "one token above the maximum is refused");
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
