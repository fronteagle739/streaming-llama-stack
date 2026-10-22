#include "llama-kv-artifact.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>

namespace {

constexpr std::array<char, 8> LLAMA_KV_ARTIFACT_MAGIC = { 'K', 'V', 'A', 'R', 'T', 'I', 'F', '1' };

// Base header (format_minor == 0): 44 bytes.
// Extended header (format_minor >= 1): 60 bytes.
// Sampling header (format_minor >= 3): 84 bytes.
// Request header (format_minor >= 4): 4 bytes.
// Prompt header (format_minor >= 5): 12 bytes.
// Pack to 1-byte alignment to ensure no compiler padding in wire format.
#pragma pack(push, 1)
struct kv_artifact_header_v1_base {
    char     magic[8];        // 8
    uint16_t format_major;    // 2
    uint16_t format_minor;    // 2
    uint32_t n_layers;        // 4
    uint32_t n_ctx;           // 4
    uint32_t token_count;     // 4
    uint16_t type_k;          // 2
    uint16_t type_v;          // 2
    uint32_t flags;           // 4
    uint64_t payload_bytes;   // 8
    uint32_t payload_crc32;   // 4
};                            // total: 44

struct kv_artifact_header_v1_ext {
    int32_t  last_token_id;     // 4
    uint8_t  v_trans;           // 1
    uint8_t  is_mla;            // 1
    uint16_t reserved_0;        // 2
    uint64_t model_fingerprint; // 8
};                              // total: 16

struct kv_artifact_header_v1_sampling {
    uint64_t user_sampling_config; // 8
    int32_t  seed;                 // 4
    int32_t  min_keep;             // 4
    int32_t  top_k;                // 4
    int32_t  penalty_last_n;       // 4
    int32_t  mirostat;             // 4
    float    top_p;                // 4
    float    min_p;                // 4
    float    typ_p;                // 4
    float    temp;                 // 4
    float    dynatemp_range;       // 4
    float    dynatemp_exponent;    // 4
    float    penalty_repeat;       // 4
    float    penalty_freq;         // 4
    float    penalty_present;      // 4
    float    xtc_probability;      // 4
    float    xtc_threshold;        // 4
    float    top_n_sigma;          // 4
    float    mirostat_tau;         // 4
    float    mirostat_eta;         // 4
};                                 // total: 84

struct kv_artifact_header_v1_request {
    int32_t requested_n_predict;   // 4
};                                 // total: 4

struct kv_artifact_header_v1_prompt {
    uint32_t prompt_token_count;          // 4
    uint32_t prompt_text_bytes;           // 4
    uint32_t streamed_prefix_token_count; // 4
};                                        // total: 12

struct kv_artifact_header_v1_cache_hit {
    uint64_t pd_session_id;               // 8 — Mac-side LRU snapshot handle
};                                        // total: 8

// Phase 5.3 (2026-05-12) — v1.7 extension: per-slot multi-slot demux metadata.
struct kv_artifact_header_v1_slot {
    int32_t source_slot_id;               // 4 — RTX-side slot index; -1 = legacy
};                                        // total: 4

// ship-embeddings (2026-06-01) — v1.8 extension: image-embeddings re-prefill metadata.
// When img_cell_count>0 the stored payload carries img_cell_count*img_n_embd f32
// image embeddings appended AFTER the prompt text+tokens.
struct kv_artifact_header_v1_image {
    uint32_t img_cell_count;              // 4 — # image cells (embedding rows); 0 = none
    uint32_t img_n_embd;                  // 4 — embedding dim per cell
    uint32_t img_pos_start;               // 4 — abs sequence position of first image cell
    uint32_t reserved_img;                // 4 — reserved (0)
};                                        // total: 16
// v2.0 (MTP-on-KV-inject, 2026-06-24): per-token last-layer hidden states (fp16) appended at the very END of the
// payload (after raw_image), sized mtp_hidden_n_tokens*mtp_hidden_n_embd uint16. Lets the receiver warm the MTP sidecar.
struct kv_artifact_header_v2_mtp {
    uint32_t mtp_hidden_n_tokens;         // 4 — # hidden rows (= streamed prefix token count); 0 = none
    uint32_t mtp_hidden_n_embd;           // 4 — hidden dim per row (= model n_embd)
};                                        // total: 8

// v2.1 (format_minor >= 10) — delta shipping (2026-09-10)
struct kv_artifact_header_v2_delta {
    uint32_t base_prefix_cells;           // 4 — cells [0, base) the decoder holds; the payload's cells start at base
    uint32_t reserved_delta;              // 4
};                                        // total: 8
#pragma pack(pop)

static_assert(sizeof(kv_artifact_header_v1_base) == 44, "base header must be 44 bytes");
static_assert(sizeof(kv_artifact_header_v1_ext)  == 16, "ext header must be 16 bytes");
static_assert(sizeof(kv_artifact_header_v1_sampling) == 84, "sampling header must be 84 bytes");
static_assert(sizeof(kv_artifact_header_v1_request) == 4, "request header must be 4 bytes");
static_assert(sizeof(kv_artifact_header_v1_prompt) == 12, "prompt header must be 12 bytes");
static_assert(sizeof(kv_artifact_header_v1_cache_hit) == 8, "cache_hit header must be 8 bytes");
static_assert(sizeof(kv_artifact_header_v1_slot) == 4, "slot header must be 4 bytes");
static_assert(sizeof(kv_artifact_header_v1_image) == 16, "image header must be 16 bytes");
static_assert(sizeof(kv_artifact_header_v2_mtp) == 8, "mtp header must be 8 bytes");
static_assert(sizeof(kv_artifact_header_v2_delta) == 8, "delta header must be 8 bytes");

static constexpr size_t HEADER_BASE_SIZE      = 44;
static constexpr size_t HEADER_EXT_SIZE       = 16;
static constexpr size_t HEADER_SAMPLING_SIZE  = 84;
static constexpr size_t HEADER_REQUEST_SIZE   = 4;
static constexpr size_t HEADER_PROMPT_SIZE    = 12;
static constexpr size_t HEADER_CACHE_HIT_SIZE = 8;
static constexpr size_t HEADER_SLOT_SIZE      = 4;
static constexpr size_t HEADER_IMAGE_SIZE     = 16;
static constexpr size_t HEADER_MTP_SIZE       = 8;
static constexpr size_t HEADER_DELTA_SIZE     = 8;
static constexpr size_t HEADER_V1_1_SIZE = HEADER_BASE_SIZE + HEADER_EXT_SIZE;               // 60
static constexpr size_t HEADER_V1_3_SIZE = HEADER_V1_1_SIZE + HEADER_SAMPLING_SIZE;          // 144
static constexpr size_t HEADER_V1_4_SIZE = HEADER_V1_3_SIZE + HEADER_REQUEST_SIZE;           // 148
static constexpr size_t HEADER_V1_5_SIZE = HEADER_V1_4_SIZE + HEADER_PROMPT_SIZE;            // 160
static constexpr size_t HEADER_V1_6_SIZE = HEADER_V1_5_SIZE + HEADER_CACHE_HIT_SIZE;         // 168
static constexpr size_t HEADER_V1_7_SIZE = HEADER_V1_6_SIZE + HEADER_SLOT_SIZE;              // 172
static constexpr size_t HEADER_V1_8_SIZE = HEADER_V1_7_SIZE + HEADER_IMAGE_SIZE;             // 188

template <typename T>
bool read_exact(std::ifstream &ifs, T &value) {
    ifs.read(reinterpret_cast<char *>(&value), sizeof(T));
    return (bool) ifs;
}

template <typename T>
bool write_exact(std::ofstream &ofs, const T &value) {
    ofs.write(reinterpret_cast<const char *>(&value), sizeof(T));
    return (bool) ofs;
}

template <typename T>
void buffer_append(std::vector<uint8_t> &buf, const T &value) {
    const uint8_t * ptr = reinterpret_cast<const uint8_t *>(&value);
    buf.insert(buf.end(), ptr, ptr + sizeof(T));
}

void set_error(std::string *error, std::string msg) {
    if (error) {
        *error = std::move(msg);
    }
}

}  // namespace

uint32_t llama_kv_artifact_crc32(const uint8_t *data, size_t size) {
    static uint32_t table[256];
    static bool table_init = false;
    if (!table_init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int j = 0; j < 8; ++j) {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1u)) : (c >> 1u);
            }
            table[i] = c;
        }
        table_init = true;
    }

    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < size; ++i) {
        crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8u);
    }
    return crc ^ 0xFFFFFFFFu;
}

static void kv_artifact_append_header(std::vector<uint8_t> & buffer_out, const llama_kv_artifact_metadata & meta,
                                      uint64_t payload_bytes, uint32_t payload_crc32, uint32_t flags,
                                      size_t raw_image_bytes, size_t reserve_payload) {
    kv_artifact_header_v1_base base = {};
    std::memcpy(base.magic, LLAMA_KV_ARTIFACT_MAGIC.data(), LLAMA_KV_ARTIFACT_MAGIC.size());
    base.format_major  = meta.format_major;
    base.format_minor  = meta.format_minor;
    base.n_layers      = meta.n_layers;
    base.n_ctx         = meta.n_ctx;
    base.token_count   = meta.token_count;
    base.type_k        = meta.type_k;
    base.type_v        = meta.type_v;
    base.flags         = flags;
    base.payload_bytes = payload_bytes;
    base.payload_crc32 = payload_crc32;

    // Pre-size the buffer so a single allocation covers the whole artifact.
    size_t header_bytes = sizeof(base);
    if (meta.format_minor >= 1) header_bytes += sizeof(kv_artifact_header_v1_ext);
    if (meta.format_minor >= 3) header_bytes += sizeof(kv_artifact_header_v1_sampling);
    if (meta.format_minor >= 4) header_bytes += sizeof(kv_artifact_header_v1_request);
    if (meta.format_minor >= 5) header_bytes += sizeof(kv_artifact_header_v1_prompt);
    if (meta.format_minor >= 6) header_bytes += sizeof(kv_artifact_header_v1_cache_hit);
    if (meta.format_minor >= 7) header_bytes += sizeof(kv_artifact_header_v1_slot);
    if (meta.format_minor >= 8) header_bytes += sizeof(kv_artifact_header_v1_image);
    if (meta.format_minor >= 9) header_bytes += sizeof(kv_artifact_header_v2_mtp);
    if (meta.format_minor >= 10) header_bytes += sizeof(kv_artifact_header_v2_delta);
    buffer_out.clear();
    buffer_out.reserve(header_bytes + (size_t) std::min<uint64_t>(payload_bytes, (uint64_t) reserve_payload));

    buffer_append(buffer_out, base);

    if (meta.format_minor >= 1) {
        kv_artifact_header_v1_ext ext = {};
        ext.last_token_id     = meta.last_token_id;
        ext.v_trans           = meta.v_trans;
        ext.is_mla            = meta.is_mla;
        ext.reserved_0        = meta.reserved_0;
        ext.model_fingerprint = meta.model_fingerprint;
        buffer_append(buffer_out, ext);
    }

    if (meta.format_minor >= 3) {
        kv_artifact_header_v1_sampling sampling = {};
        sampling.user_sampling_config = meta.sampling_profile.user_sampling_config;
        sampling.seed                 = meta.sampling_profile.seed;
        sampling.min_keep             = meta.sampling_profile.min_keep;
        sampling.top_k                = meta.sampling_profile.top_k;
        sampling.penalty_last_n       = meta.sampling_profile.penalty_last_n;
        sampling.mirostat             = meta.sampling_profile.mirostat;
        sampling.top_p                = meta.sampling_profile.top_p;
        sampling.min_p                = meta.sampling_profile.min_p;
        sampling.typ_p                = meta.sampling_profile.typ_p;
        sampling.temp                 = meta.sampling_profile.temp;
        sampling.dynatemp_range       = meta.sampling_profile.dynatemp_range;
        sampling.dynatemp_exponent    = meta.sampling_profile.dynatemp_exponent;
        sampling.penalty_repeat       = meta.sampling_profile.penalty_repeat;
        sampling.penalty_freq         = meta.sampling_profile.penalty_freq;
        sampling.penalty_present      = meta.sampling_profile.penalty_present;
        sampling.xtc_probability      = meta.sampling_profile.xtc_probability;
        sampling.xtc_threshold        = meta.sampling_profile.xtc_threshold;
        sampling.top_n_sigma          = meta.sampling_profile.top_n_sigma;
        sampling.mirostat_tau         = meta.sampling_profile.mirostat_tau;
        sampling.mirostat_eta         = meta.sampling_profile.mirostat_eta;
        buffer_append(buffer_out, sampling);
    }

    if (meta.format_minor >= 4) {
        kv_artifact_header_v1_request request = {};
        request.requested_n_predict = meta.requested_n_predict;
        buffer_append(buffer_out, request);
    }

    if (meta.format_minor >= 5) {
        kv_artifact_header_v1_prompt prompt = {};
        prompt.prompt_token_count          = meta.prompt_token_count;
        prompt.prompt_text_bytes           = meta.prompt_text_bytes;
        prompt.streamed_prefix_token_count = meta.streamed_prefix_token_count;
        buffer_append(buffer_out, prompt);
    }

    if (meta.format_minor >= 6) {
        kv_artifact_header_v1_cache_hit ch = {};
        ch.pd_session_id = meta.pd_session_id;
        buffer_append(buffer_out, ch);
    }

    if (meta.format_minor >= 7) {
        kv_artifact_header_v1_slot slot = {};
        slot.source_slot_id = meta.source_slot_id;
        buffer_append(buffer_out, slot);
    }

    if (meta.format_minor >= 8) {
        kv_artifact_header_v1_image image = {};
        image.img_cell_count = meta.img_cell_count;
        image.img_n_embd     = meta.img_n_embd;
        image.img_pos_start  = meta.img_pos_start;
        image.reserved_img   = static_cast<uint32_t>(raw_image_bytes);  // v1.9: raw image byte count (RAW_IMAGE flag)
        buffer_append(buffer_out, image);
    }

    if (meta.format_minor >= 9) {
        kv_artifact_header_v2_mtp mtp = {};
        mtp.mtp_hidden_n_tokens = meta.mtp_hidden_n_tokens;
        mtp.mtp_hidden_n_embd   = meta.mtp_hidden_n_embd;
        buffer_append(buffer_out, mtp);
    }
    if (meta.format_minor >= 10) {
        kv_artifact_header_v2_delta delta = {};
        delta.base_prefix_cells = meta.base_prefix_cells;
        buffer_append(buffer_out, delta);
    }

}

bool llama_kv_artifact_serialize_header(std::vector<uint8_t> &            header_out,
                                        const llama_kv_artifact_metadata &meta,
                                        uint64_t                          payload_bytes,
                                        uint32_t                          payload_crc32,
                                        uint32_t                          extra_flags,
                                        size_t                            raw_image_bytes) {
    kv_artifact_append_header(header_out, meta, payload_bytes, payload_crc32, meta.flags | extra_flags, raw_image_bytes, 0);
    return true;
}

bool llama_kv_artifact_serialize_to_buffer(std::vector<uint8_t> &            buffer_out,
                                           const std::vector<uint8_t> &      payload,
                                           const llama_kv_artifact_metadata &meta,
                                           llama_kv_artifact_summary *       summary,
                                           std::string *                     error,
                                           const llama_kv_prompt_payload *   prompt_payload) {
    (void) error;  // No I/O failures to report at this layer.

    std::vector<uint8_t> stored_payload;
    const bool use_prompt_payload = prompt_payload != nullptr &&
        (!prompt_payload->token_ids.empty() || !prompt_payload->prompt_text.empty());
    const size_t prompt_text_bytes  = use_prompt_payload ? prompt_payload->prompt_text.size() : 0;
    const size_t prompt_token_bytes = use_prompt_payload ? prompt_payload->token_ids.size() * sizeof(int32_t) : 0;
    // v1.8 ship-embeddings: image embeddings appended AFTER text+tokens. Gated on format_minor>=8.
    const bool   use_image_embd     = meta.format_minor >= 8 && prompt_payload != nullptr &&
                                      !prompt_payload->image_embd.empty();
    const size_t image_embd_bytes   = use_image_embd ? prompt_payload->image_embd.size() * sizeof(float) : 0;
    // v1.9 small-image-forward: raw image bytes appended at the very END (after image_embd).
    const bool   use_raw_image      = meta.format_minor >= 8 && prompt_payload != nullptr &&
                                      !prompt_payload->raw_image.empty();
    const size_t raw_image_bytes    = use_raw_image ? prompt_payload->raw_image.size() : 0;
    // v2.0 MTP-on-KV-inject: fp16 per-token hidden states at the very END (after raw_image). Gated on format_minor>=9.
    const bool   use_mtp_hidden     = meta.format_minor >= 9 && prompt_payload != nullptr &&
                                      !prompt_payload->mtp_hidden.empty();
    const size_t mtp_hidden_bytes   = use_mtp_hidden ? prompt_payload->mtp_hidden.size() * sizeof(uint16_t) : 0;
    stored_payload.reserve(payload.size() + prompt_text_bytes + prompt_token_bytes + image_embd_bytes + raw_image_bytes + mtp_hidden_bytes);
    stored_payload.insert(stored_payload.end(), payload.begin(), payload.end());
    if (use_prompt_payload && prompt_text_bytes > 0) {
        const uint8_t * text_ptr = reinterpret_cast<const uint8_t *>(prompt_payload->prompt_text.data());
        stored_payload.insert(stored_payload.end(), text_ptr, text_ptr + prompt_text_bytes);
    }
    if (use_prompt_payload && prompt_token_bytes > 0) {
        const uint8_t * token_ptr = reinterpret_cast<const uint8_t *>(prompt_payload->token_ids.data());
        stored_payload.insert(stored_payload.end(), token_ptr, token_ptr + prompt_token_bytes);
    }
    if (use_image_embd && image_embd_bytes > 0) {
        const uint8_t * embd_ptr = reinterpret_cast<const uint8_t *>(prompt_payload->image_embd.data());
        stored_payload.insert(stored_payload.end(), embd_ptr, embd_ptr + image_embd_bytes);
    }
    if (use_raw_image && raw_image_bytes > 0) {
        stored_payload.insert(stored_payload.end(),
                              prompt_payload->raw_image.begin(), prompt_payload->raw_image.end());
    }
    if (use_mtp_hidden && mtp_hidden_bytes > 0) {
        const uint8_t * mtp_ptr = reinterpret_cast<const uint8_t *>(prompt_payload->mtp_hidden.data());
        stored_payload.insert(stored_payload.end(), mtp_ptr, mtp_ptr + mtp_hidden_bytes);
    }

    const uint32_t flags = meta.flags | (use_raw_image ? LLAMA_KV_ARTIFACT_FLAG_RAW_IMAGE : 0u)
                                    | (use_mtp_hidden ? LLAMA_KV_ARTIFACT_FLAG_MTP_HIDDEN : 0u);
    const uint32_t payload_crc32 = stored_payload.empty() ? 0u : llama_kv_artifact_crc32(stored_payload.data(), stored_payload.size());
    kv_artifact_append_header(buffer_out, meta, (uint64_t) stored_payload.size(), payload_crc32, flags, raw_image_bytes, stored_payload.size());
    if (summary) {
        summary->payload_bytes = (uint64_t) stored_payload.size();
        summary->payload_crc32 = payload_crc32;
    }
    if (!stored_payload.empty()) {
        buffer_out.insert(buffer_out.end(), stored_payload.begin(), stored_payload.end());
    }
    return true;
}

bool llama_kv_artifact_write(const std::string &               path,
                             const std::vector<uint8_t> &      payload,
                             const llama_kv_artifact_metadata &meta,
                             llama_kv_artifact_summary *       summary,
                             std::string *                     error,
                             const llama_kv_prompt_payload *   prompt_payload) {
    std::vector<uint8_t> buffer;
    if (!llama_kv_artifact_serialize_to_buffer(buffer, payload, meta, summary, error, prompt_payload)) {
        return false;
    }

    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        set_error(error, "failed to open kv artifact for write: " + path);
        return false;
    }
    if (!buffer.empty()) {
        ofs.write(reinterpret_cast<const char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        if (!ofs) {
            set_error(error, "failed to write kv artifact payload: " + path);
            return false;
        }
    }
    return true;
}

bool llama_kv_artifact_read(const std::string &         path,
                            std::vector<uint8_t> &      payload_out,
                            llama_kv_artifact_metadata *meta_out,
                            llama_kv_artifact_summary * summary_out,
                            std::string *               error,
                            llama_kv_prompt_payload *   prompt_payload_out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        set_error(error, "failed to open kv artifact for read: " + path);
        return false;
    }

    kv_artifact_header_v1_base base = {};
    if (!read_exact(ifs, base)) {
        set_error(error, "failed to read kv artifact header: " + path);
        return false;
    }

    if (!std::equal(std::begin(base.magic), std::end(base.magic), LLAMA_KV_ARTIFACT_MAGIC.begin())) {
        set_error(error, "invalid kv artifact magic: " + path);
        return false;
    }

    // Read extension header if format_minor >= 1
    kv_artifact_header_v1_ext ext = {};
    ext.last_token_id = -1;  // safe default for v1.0 files
    if (base.format_minor >= 1) {
        if (!read_exact(ifs, ext)) {
            set_error(error, "failed to read kv artifact ext header: " + path);
            return false;
        }
    }

    kv_artifact_header_v1_sampling sampling = {};
    if (base.format_minor >= 3) {
        if (!read_exact(ifs, sampling)) {
            set_error(error, "failed to read kv artifact sampling header: " + path);
            return false;
        }
    }

    kv_artifact_header_v1_request request = {};
    request.requested_n_predict = -1;
    if (base.format_minor >= 4) {
        if (!read_exact(ifs, request)) {
            set_error(error, "failed to read kv artifact request header: " + path);
            return false;
        }
    }

    kv_artifact_header_v1_prompt prompt = {};
    if (base.format_minor >= 5) {
        if (!read_exact(ifs, prompt)) {
            set_error(error, "failed to read kv artifact prompt header: " + path);
            return false;
        }
    }

    kv_artifact_header_v1_cache_hit cache_hit = {};
    if (base.format_minor >= 6) {
        if (!read_exact(ifs, cache_hit)) {
            set_error(error, "failed to read kv artifact cache_hit header: " + path);
            return false;
        }
    }

    kv_artifact_header_v1_slot slot = {};
    slot.source_slot_id = -1;
    if (base.format_minor >= 7) {
        if (!read_exact(ifs, slot)) {
            set_error(error, "failed to read kv artifact slot header: " + path);
            return false;
        }
    }

    kv_artifact_header_v1_image image_hdr = {};
    if (base.format_minor >= 8) {
        if (!read_exact(ifs, image_hdr)) {
            set_error(error, "failed to read kv artifact image header: " + path);
            return false;
        }
    }
    kv_artifact_header_v2_mtp mtp_hdr = {};
    if (base.format_minor >= 9) {
        if (!read_exact(ifs, mtp_hdr)) {
            set_error(error, "failed to read kv artifact mtp header: " + path);
            return false;
        }
    }
    kv_artifact_header_v2_delta delta_hdr = {};
    if (base.format_minor >= 10) {
        if (!read_exact(ifs, delta_hdr)) {
            set_error(error, "failed to read kv artifact delta header: " + path);
            return false;
        }
    }

    std::vector<uint8_t> stored_payload(static_cast<size_t>(base.payload_bytes));
    if (!stored_payload.empty()) {
        ifs.read(reinterpret_cast<char *>(stored_payload.data()), static_cast<std::streamsize>(stored_payload.size()));
        if (!ifs) {
            set_error(error, "failed to read kv artifact payload: " + path);
            return false;
        }
    }

    const uint32_t crc = stored_payload.empty() ? 0u : llama_kv_artifact_crc32(stored_payload.data(), stored_payload.size());
    if (!(base.flags & LLAMA_KV_ARTIFACT_FLAG_NO_CRC) && crc != base.payload_crc32) {
        set_error(error, "kv artifact crc mismatch: expected " + std::to_string(base.payload_crc32) +
                             " got " + std::to_string(crc));
        return false;
    }

    const size_t tail_token_count = (base.format_minor >= 2 && base.format_minor < 5) ? ext.reserved_0 : 0;
    const size_t tail_bytes = tail_token_count * sizeof(int32_t);
    const size_t prompt_text_bytes = base.format_minor >= 5 ? prompt.prompt_text_bytes : 0;
    const size_t prompt_token_bytes = base.format_minor >= 5 ? (size_t) prompt.prompt_token_count * sizeof(int32_t) : 0;
    const size_t img_embd_bytes = base.format_minor >= 8 ? (size_t) image_hdr.img_cell_count * image_hdr.img_n_embd * sizeof(float) : 0;
    // v1.9 small-image-forward: raw image bytes at the very END (byte count in image_hdr.reserved_img).
    const size_t img_raw_bytes = (base.format_minor >= 8 && (base.flags & LLAMA_KV_ARTIFACT_FLAG_RAW_IMAGE))
                                     ? (size_t) image_hdr.reserved_img : 0;
    // v2.0 MTP-on-KV-inject: fp16 hidden states at the very END (after raw_image).
    const size_t mtp_hidden_bytes = (base.format_minor >= 9 && (base.flags & LLAMA_KV_ARTIFACT_FLAG_MTP_HIDDEN))
                                     ? (size_t) mtp_hdr.mtp_hidden_n_tokens * mtp_hdr.mtp_hidden_n_embd * sizeof(uint16_t) : 0;
    const size_t aux_bytes = (base.format_minor >= 5 ? (prompt_text_bytes + prompt_token_bytes) : tail_bytes) + img_embd_bytes + img_raw_bytes + mtp_hidden_bytes;
    if (aux_bytes > stored_payload.size()) {
        set_error(error, base.format_minor >= 5
                             ? "kv artifact embedded prompt payload exceeds payload size"
                             : "kv artifact embedded tail exceeds payload size");
        return false;
    }

    const size_t kv_payload_bytes = stored_payload.size() - aux_bytes;
    payload_out.assign(stored_payload.begin(), stored_payload.begin() + kv_payload_bytes);
    if (prompt_payload_out) {
        prompt_payload_out->token_ids.clear();
        prompt_payload_out->prompt_text.clear();
        if (base.format_minor >= 5) {
            if (prompt_text_bytes > 0) {
                prompt_payload_out->prompt_text.assign(
                    reinterpret_cast<const char *>(stored_payload.data() + kv_payload_bytes),
                    prompt_text_bytes);
            }
            if (prompt.prompt_token_count > 0) {
                prompt_payload_out->token_ids.resize(prompt.prompt_token_count);
                std::memcpy(prompt_payload_out->token_ids.data(),
                            stored_payload.data() + kv_payload_bytes + prompt_text_bytes,
                            prompt_token_bytes);
            }
        } else if (tail_token_count > 0) {
            prompt_payload_out->token_ids.resize(tail_token_count);
            std::memcpy(prompt_payload_out->token_ids.data(), stored_payload.data() + kv_payload_bytes, tail_bytes);
        }
        // v1.8 ship-embeddings: image embeddings live after [K/V][text][tokens].
        prompt_payload_out->image_embd.clear();
        if (img_embd_bytes > 0) {
            const size_t embd_off = kv_payload_bytes + prompt_text_bytes + prompt_token_bytes;
            prompt_payload_out->image_embd.resize((size_t) image_hdr.img_cell_count * image_hdr.img_n_embd);
            std::memcpy(prompt_payload_out->image_embd.data(), stored_payload.data() + embd_off, img_embd_bytes);
        }
        // v1.9 small-image-forward: raw image bytes after [K/V][text][tokens][image_embd].
        prompt_payload_out->raw_image.clear();
        if (img_raw_bytes > 0) {
            const size_t raw_off = kv_payload_bytes + prompt_text_bytes + prompt_token_bytes + img_embd_bytes;
            prompt_payload_out->raw_image.assign(stored_payload.begin() + raw_off,
                                                 stored_payload.begin() + raw_off + img_raw_bytes);
        }
        // v2.0 MTP-on-KV-inject: fp16 hidden states after [K/V][text][tokens][image_embd][raw_image].
        prompt_payload_out->mtp_hidden.clear();
        if (mtp_hidden_bytes > 0) {
            const size_t mtp_off = kv_payload_bytes + prompt_text_bytes + prompt_token_bytes + img_embd_bytes + img_raw_bytes;
            prompt_payload_out->mtp_hidden.resize((size_t) mtp_hdr.mtp_hidden_n_tokens * mtp_hdr.mtp_hidden_n_embd);
            std::memcpy(prompt_payload_out->mtp_hidden.data(), stored_payload.data() + mtp_off, mtp_hidden_bytes);
        }
    }

    if (meta_out) {
        meta_out->format_major      = base.format_major;
        meta_out->format_minor      = base.format_minor;
        meta_out->n_layers          = base.n_layers;
        meta_out->n_ctx             = base.n_ctx;
        meta_out->token_count       = base.token_count;
        meta_out->type_k            = base.type_k;
        meta_out->type_v            = base.type_v;
        meta_out->flags             = base.flags;
        meta_out->last_token_id     = ext.last_token_id;
        meta_out->v_trans           = ext.v_trans;
        meta_out->is_mla            = ext.is_mla;
        meta_out->reserved_0        = ext.reserved_0;
        meta_out->model_fingerprint = ext.model_fingerprint;
        meta_out->sampling_profile.user_sampling_config = sampling.user_sampling_config;
        meta_out->sampling_profile.seed                 = sampling.seed;
        meta_out->sampling_profile.min_keep             = sampling.min_keep;
        meta_out->sampling_profile.top_k                = sampling.top_k;
        meta_out->sampling_profile.penalty_last_n       = sampling.penalty_last_n;
        meta_out->sampling_profile.mirostat             = sampling.mirostat;
        meta_out->sampling_profile.top_p                = sampling.top_p;
        meta_out->sampling_profile.min_p                = sampling.min_p;
        meta_out->sampling_profile.typ_p                = sampling.typ_p;
        meta_out->sampling_profile.temp                 = sampling.temp;
        meta_out->sampling_profile.dynatemp_range       = sampling.dynatemp_range;
        meta_out->sampling_profile.dynatemp_exponent    = sampling.dynatemp_exponent;
        meta_out->sampling_profile.penalty_repeat       = sampling.penalty_repeat;
        meta_out->sampling_profile.penalty_freq         = sampling.penalty_freq;
        meta_out->sampling_profile.penalty_present      = sampling.penalty_present;
        meta_out->sampling_profile.xtc_probability      = sampling.xtc_probability;
        meta_out->sampling_profile.xtc_threshold        = sampling.xtc_threshold;
        meta_out->sampling_profile.top_n_sigma          = sampling.top_n_sigma;
        meta_out->sampling_profile.mirostat_tau         = sampling.mirostat_tau;
        meta_out->sampling_profile.mirostat_eta         = sampling.mirostat_eta;
        meta_out->requested_n_predict                   = request.requested_n_predict;
        meta_out->prompt_token_count                    = prompt.prompt_token_count;
        meta_out->prompt_text_bytes                     = prompt.prompt_text_bytes;
        meta_out->streamed_prefix_token_count           = prompt.streamed_prefix_token_count;
        meta_out->pd_session_id                         = cache_hit.pd_session_id;
        meta_out->source_slot_id                        = slot.source_slot_id;
        meta_out->img_cell_count                        = image_hdr.img_cell_count;
        meta_out->img_n_embd                            = image_hdr.img_n_embd;
        meta_out->img_pos_start                         = image_hdr.img_pos_start;
        meta_out->mtp_hidden_n_tokens                   = mtp_hdr.mtp_hidden_n_tokens;
        meta_out->mtp_hidden_n_embd                     = mtp_hdr.mtp_hidden_n_embd;
        meta_out->base_prefix_cells                     = delta_hdr.base_prefix_cells;
    }
    if (summary_out) {
        summary_out->payload_bytes = base.payload_bytes;
        summary_out->payload_crc32 = base.payload_crc32;
    }
    return true;
}

static bool read_artifact_memory(const uint8_t *             data,
                                size_t                      size,
                                std::vector<uint8_t> &      payload_out,
                                llama_kv_artifact_metadata *meta_out,
                                llama_kv_artifact_summary * summary_out,
                                std::string *               error,
                                llama_kv_prompt_payload *   prompt_payload_out, size_t *header_only) {
    if (!data || size < HEADER_BASE_SIZE) {
        set_error(error, "kv artifact buffer too small for header");
        return false;
    }

    kv_artifact_header_v1_base base = {};
    std::memcpy(&base, data, HEADER_BASE_SIZE);

    if (!std::equal(std::begin(base.magic), std::end(base.magic), LLAMA_KV_ARTIFACT_MAGIC.begin())) {
        set_error(error, "invalid kv artifact magic in memory buffer");
        return false;
    }

    size_t hdr_size = HEADER_BASE_SIZE;
    kv_artifact_header_v1_ext ext = {};
    ext.last_token_id = -1;
    if (base.format_minor >= 1) {
        if (size < HEADER_V1_1_SIZE) {
            set_error(error, "kv artifact buffer too small for v1.1 header");
            return false;
        }
        std::memcpy(&ext, data + HEADER_BASE_SIZE, HEADER_EXT_SIZE);
        hdr_size = HEADER_V1_1_SIZE;
    }

    kv_artifact_header_v1_sampling sampling = {};
    if (base.format_minor >= 3) {
        if (size < HEADER_V1_3_SIZE) {
            set_error(error, "kv artifact buffer too small for v1.3 header");
            return false;
        }
        std::memcpy(&sampling, data + HEADER_V1_1_SIZE, HEADER_SAMPLING_SIZE);
        hdr_size = HEADER_V1_3_SIZE;
    }

    kv_artifact_header_v1_request request = {};
    request.requested_n_predict = -1;
    if (base.format_minor >= 4) {
        if (size < HEADER_V1_4_SIZE) {
            set_error(error, "kv artifact buffer too small for v1.4 header");
            return false;
        }
        std::memcpy(&request, data + HEADER_V1_3_SIZE, HEADER_REQUEST_SIZE);
        hdr_size = HEADER_V1_4_SIZE;
    }

    kv_artifact_header_v1_prompt prompt = {};
    if (base.format_minor >= 5) {
        if (size < HEADER_V1_5_SIZE) {
            set_error(error, "kv artifact buffer too small for v1.5 header");
            return false;
        }
        std::memcpy(&prompt, data + HEADER_V1_4_SIZE, HEADER_PROMPT_SIZE);
        hdr_size = HEADER_V1_5_SIZE;
    }

    kv_artifact_header_v1_cache_hit cache_hit = {};
    if (base.format_minor >= 6) {
        if (size < HEADER_V1_6_SIZE) {
            set_error(error, "kv artifact buffer too small for v1.6 header");
            return false;
        }
        std::memcpy(&cache_hit, data + HEADER_V1_5_SIZE, HEADER_CACHE_HIT_SIZE);
        hdr_size = HEADER_V1_6_SIZE;
    }

    kv_artifact_header_v1_slot slot = {};
    slot.source_slot_id = -1;
    if (base.format_minor >= 7) {
        if (size < HEADER_V1_7_SIZE) {
            set_error(error, "kv artifact buffer too small for v1.7 header");
            return false;
        }
        std::memcpy(&slot, data + HEADER_V1_6_SIZE, HEADER_SLOT_SIZE);
        hdr_size = HEADER_V1_7_SIZE;
    }

    kv_artifact_header_v1_image image_hdr = {};
    if (base.format_minor >= 8) {
        if (size < HEADER_V1_8_SIZE) {
            set_error(error, "kv artifact buffer too small for v1.8 header");
            return false;
        }
        std::memcpy(&image_hdr, data + HEADER_V1_7_SIZE, HEADER_IMAGE_SIZE);
        hdr_size = HEADER_V1_8_SIZE;
    }
    kv_artifact_header_v2_mtp mtp_hdr = {};
    if (base.format_minor >= 9) {
        if (size < HEADER_V1_8_SIZE + HEADER_MTP_SIZE) {
            set_error(error, "kv artifact buffer too small for v2.0 mtp header");
            return false;
        }
        std::memcpy(&mtp_hdr, data + HEADER_V1_8_SIZE, HEADER_MTP_SIZE);
        hdr_size = HEADER_V1_8_SIZE + HEADER_MTP_SIZE;
    }
    kv_artifact_header_v2_delta delta_hdr = {};
    if (base.format_minor >= 10) {
        if (size < HEADER_V1_8_SIZE + HEADER_MTP_SIZE + HEADER_DELTA_SIZE) {
            set_error(error, "kv artifact buffer too small for v2.1 delta header");
            return false;
        }
        std::memcpy(&delta_hdr, data + HEADER_V1_8_SIZE + HEADER_MTP_SIZE, HEADER_DELTA_SIZE);
        hdr_size = HEADER_V1_8_SIZE + HEADER_MTP_SIZE + HEADER_DELTA_SIZE;
    }

    if (header_only) *header_only=hdr_size;
    else {
    const size_t payload_bytes = static_cast<size_t>(base.payload_bytes);
    if (base.payload_bytes>SIZE_MAX || hdr_size>size || payload_bytes>size-hdr_size) {
        set_error(error, "kv artifact buffer too small for payload: need " +
                         std::to_string(hdr_size + payload_bytes) + " have " + std::to_string(size));
        return false;
    }

    const uint8_t * payload_ptr = data + hdr_size;
    std::vector<uint8_t> stored_payload(payload_ptr, payload_ptr + payload_bytes);

    const bool no_crc = (base.flags & LLAMA_KV_ARTIFACT_FLAG_NO_CRC) != 0;
    const uint32_t crc = (payload_bytes > 0 && !no_crc) ? llama_kv_artifact_crc32(stored_payload.data(), stored_payload.size()) : 0u;
    if (!no_crc && crc != base.payload_crc32) {
        set_error(error, "kv artifact crc mismatch: expected " + std::to_string(base.payload_crc32) +
                             " got " + std::to_string(crc));
        return false;
    }

    const size_t tail_token_count = (base.format_minor >= 2 && base.format_minor < 5) ? ext.reserved_0 : 0;
    const size_t tail_bytes = tail_token_count * sizeof(int32_t);
    const size_t prompt_text_bytes = base.format_minor >= 5 ? prompt.prompt_text_bytes : 0;
    const size_t prompt_token_bytes = base.format_minor >= 5 ? (size_t) prompt.prompt_token_count * sizeof(int32_t) : 0;
    const size_t img_embd_bytes = base.format_minor >= 8 ? (size_t) image_hdr.img_cell_count * image_hdr.img_n_embd * sizeof(float) : 0;
    // v1.9 small-image-forward: raw image bytes at the very END (byte count in image_hdr.reserved_img).
    const size_t img_raw_bytes = (base.format_minor >= 8 && (base.flags & LLAMA_KV_ARTIFACT_FLAG_RAW_IMAGE))
                                     ? (size_t) image_hdr.reserved_img : 0;
    // v2.0 MTP-on-KV-inject: fp16 hidden states at the very END (after raw_image).
    const size_t mtp_hidden_bytes = (base.format_minor >= 9 && (base.flags & LLAMA_KV_ARTIFACT_FLAG_MTP_HIDDEN))
                                     ? (size_t) mtp_hdr.mtp_hidden_n_tokens * mtp_hdr.mtp_hidden_n_embd * sizeof(uint16_t) : 0;
    const size_t aux_bytes = (base.format_minor >= 5 ? (prompt_text_bytes + prompt_token_bytes) : tail_bytes) + img_embd_bytes + img_raw_bytes + mtp_hidden_bytes;
    if (aux_bytes > stored_payload.size()) {
        set_error(error, base.format_minor >= 5
                             ? "kv artifact embedded prompt payload exceeds payload size"
                             : "kv artifact embedded tail exceeds payload size");
        return false;
    }

    const size_t kv_payload_bytes = stored_payload.size() - aux_bytes;
    payload_out.assign(stored_payload.begin(), stored_payload.begin() + kv_payload_bytes);
    if (prompt_payload_out) {
        prompt_payload_out->token_ids.clear();
        prompt_payload_out->prompt_text.clear();
        if (base.format_minor >= 5) {
            if (prompt_text_bytes > 0) {
                prompt_payload_out->prompt_text.assign(
                    reinterpret_cast<const char *>(stored_payload.data() + kv_payload_bytes),
                    prompt_text_bytes);
            }
            if (prompt.prompt_token_count > 0) {
                prompt_payload_out->token_ids.resize(prompt.prompt_token_count);
                std::memcpy(prompt_payload_out->token_ids.data(),
                            stored_payload.data() + kv_payload_bytes + prompt_text_bytes,
                            prompt_token_bytes);
            }
        } else if (tail_token_count > 0) {
            prompt_payload_out->token_ids.resize(tail_token_count);
            std::memcpy(prompt_payload_out->token_ids.data(), stored_payload.data() + kv_payload_bytes, tail_bytes);
        }
        // v1.8 ship-embeddings: image embeddings live after [K/V][text][tokens].
        prompt_payload_out->image_embd.clear();
        if (img_embd_bytes > 0) {
            const size_t embd_off = kv_payload_bytes + prompt_text_bytes + prompt_token_bytes;
            prompt_payload_out->image_embd.resize((size_t) image_hdr.img_cell_count * image_hdr.img_n_embd);
            std::memcpy(prompt_payload_out->image_embd.data(), stored_payload.data() + embd_off, img_embd_bytes);
        }
        // v1.9 small-image-forward: raw image bytes after [K/V][text][tokens][image_embd].
        prompt_payload_out->raw_image.clear();
        if (img_raw_bytes > 0) {
            const size_t raw_off = kv_payload_bytes + prompt_text_bytes + prompt_token_bytes + img_embd_bytes;
            prompt_payload_out->raw_image.assign(stored_payload.begin() + raw_off,
                                                 stored_payload.begin() + raw_off + img_raw_bytes);
        }
        // v2.0 MTP-on-KV-inject: fp16 hidden states after [K/V][text][tokens][image_embd][raw_image].
        prompt_payload_out->mtp_hidden.clear();
        if (mtp_hidden_bytes > 0) {
            const size_t mtp_off = kv_payload_bytes + prompt_text_bytes + prompt_token_bytes + img_embd_bytes + img_raw_bytes;
            prompt_payload_out->mtp_hidden.resize((size_t) mtp_hdr.mtp_hidden_n_tokens * mtp_hdr.mtp_hidden_n_embd);
            std::memcpy(prompt_payload_out->mtp_hidden.data(), stored_payload.data() + mtp_off, mtp_hidden_bytes);
        }
    }

    } // Header-only callers never read or allocate payload bytes.

    if (meta_out) {
        meta_out->format_major      = base.format_major;
        meta_out->format_minor      = base.format_minor;
        meta_out->n_layers          = base.n_layers;
        meta_out->n_ctx             = base.n_ctx;
        meta_out->token_count       = base.token_count;
        meta_out->type_k            = base.type_k;
        meta_out->type_v            = base.type_v;
        meta_out->flags             = base.flags;
        meta_out->last_token_id     = ext.last_token_id;
        meta_out->v_trans           = ext.v_trans;
        meta_out->is_mla            = ext.is_mla;
        meta_out->reserved_0        = ext.reserved_0;
        meta_out->model_fingerprint = ext.model_fingerprint;
        meta_out->sampling_profile.user_sampling_config = sampling.user_sampling_config;
        meta_out->sampling_profile.seed                 = sampling.seed;
        meta_out->sampling_profile.min_keep             = sampling.min_keep;
        meta_out->sampling_profile.top_k                = sampling.top_k;
        meta_out->sampling_profile.penalty_last_n       = sampling.penalty_last_n;
        meta_out->sampling_profile.mirostat             = sampling.mirostat;
        meta_out->sampling_profile.top_p                = sampling.top_p;
        meta_out->sampling_profile.min_p                = sampling.min_p;
        meta_out->sampling_profile.typ_p                = sampling.typ_p;
        meta_out->sampling_profile.temp                 = sampling.temp;
        meta_out->sampling_profile.dynatemp_range       = sampling.dynatemp_range;
        meta_out->sampling_profile.dynatemp_exponent    = sampling.dynatemp_exponent;
        meta_out->sampling_profile.penalty_repeat       = sampling.penalty_repeat;
        meta_out->sampling_profile.penalty_freq         = sampling.penalty_freq;
        meta_out->sampling_profile.penalty_present      = sampling.penalty_present;
        meta_out->sampling_profile.xtc_probability      = sampling.xtc_probability;
        meta_out->sampling_profile.xtc_threshold        = sampling.xtc_threshold;
        meta_out->sampling_profile.top_n_sigma          = sampling.top_n_sigma;
        meta_out->sampling_profile.mirostat_tau         = sampling.mirostat_tau;
        meta_out->sampling_profile.mirostat_eta         = sampling.mirostat_eta;
        meta_out->requested_n_predict                   = request.requested_n_predict;
        meta_out->prompt_token_count                    = prompt.prompt_token_count;
        meta_out->prompt_text_bytes                     = prompt.prompt_text_bytes;
        meta_out->streamed_prefix_token_count           = prompt.streamed_prefix_token_count;
        meta_out->pd_session_id                         = cache_hit.pd_session_id;
        meta_out->source_slot_id                        = slot.source_slot_id;
        meta_out->img_cell_count                        = image_hdr.img_cell_count;
        meta_out->img_n_embd                            = image_hdr.img_n_embd;
        meta_out->img_pos_start                         = image_hdr.img_pos_start;
        meta_out->mtp_hidden_n_tokens                   = mtp_hdr.mtp_hidden_n_tokens;
        meta_out->mtp_hidden_n_embd                     = mtp_hdr.mtp_hidden_n_embd;
        meta_out->base_prefix_cells                     = delta_hdr.base_prefix_cells;
    }
    if (summary_out) {
        summary_out->payload_bytes = base.payload_bytes;
        summary_out->payload_crc32 = base.payload_crc32;
    }
    return true;
}


bool llama_kv_artifact_read_mem(const uint8_t *data,size_t size,
        std::vector<uint8_t>&payload,llama_kv_artifact_metadata*meta,
        llama_kv_artifact_summary*summary,std::string*error,
        llama_kv_prompt_payload*prompt) {
    return read_artifact_memory(data,size,payload,meta,summary,error,prompt,nullptr);
}

bool llama_kv_artifact_parse_header(const uint8_t*data,size_t size,
        llama_kv_artifact_metadata&meta,llama_kv_artifact_summary&summary,
        size_t&header_bytes,std::string*error) {
    meta={};summary={};header_bytes=0;
    if(size>4096){set_error(error,"kv artifact header exceeds bounded4096-byte parser");return false;}
    std::vector<uint8_t> unused;
    llama_kv_artifact_metadata parsed;llama_kv_artifact_summary measured;size_t consumed=0;
    if(!read_artifact_memory(data,size,unused,&parsed,&measured,error,nullptr,&consumed))return false;
    meta=parsed;summary=measured;header_bytes=consumed;return true;
}
