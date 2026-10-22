#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct llama_kv_prompt_payload {
    std::vector<int32_t> token_ids;
    std::string          prompt_text;
    // --- v1.8 (ship-embeddings): image embeddings for receiver-side re-prefill ---
    // img_cell_count * img_n_embd floats, row-major (cell-major). Empty = no image.
    // The receiver re-prefills these cells via an embd batch (Metal F32 FA) instead of
    // injecting the RTX's degraded F16-VKQ image K/V. See GROUND_TRUTH §11.
    std::vector<float>   image_embd;
    // --- v1.9 (small-image-forward, 2026-06-01): RAW image bytes (PNG/JPEG) ---
    // When non-empty (+ FLAG_RAW_IMAGE), the receiver processes this SMALL image prompt
    // END-TO-END LOCALLY via mlx_vlm generate() on these raw bytes (no RTX clip/prefill
    // degradation). prompt_text carries the question. See GROUND_TRUTH §11 (USER 2026-06-01).
    std::vector<uint8_t> raw_image;
    // --- v2.0 (MTP-on-KV-inject, 2026-06-24): per-token LAST-LAYER hidden states (fp16) for the streamed prefix,
    // appended at the very END of the payload (after raw_image). mtp_hidden_n_tokens * mtp_hidden_n_embd uint16 values,
    // row-major [token][embd]. Lets the receiver warm the MTP sidecar (layer-78 NextN) over a KV-inject session, which
    // otherwise lacks the prompt's hidden states and so disables MTP. Gated on FLAG_MTP_HIDDEN + format_minor >= 9.
    std::vector<uint16_t> mtp_hidden;
};

struct llama_kv_artifact_metadata {
    uint16_t format_major = 1;
    uint16_t format_minor = 0;
    uint32_t n_layers = 0;
    uint32_t n_ctx = 0;
    uint32_t token_count = 0;
    uint16_t type_k = 0;
    uint16_t type_v = 0;
    uint32_t flags = 0;
    // --- v1.1 extension (format_minor >= 1) ---
    int32_t  last_token_id     = -1;  // last prefilled token; -1 = not stored (use BOS)
    uint8_t  v_trans           = 0;   // 1 if V cache is transposed
    uint8_t  is_mla            = 0;   // 1 if MLA model (V stored in K)
    uint16_t reserved_0        = 0;   // format_minor >= 2 legacy embedded tail token count
    uint64_t model_fingerprint = 0;   // FNV-1a hash of model architecture; 0 = unknown
    // --- v1.3 extension (format_minor >= 3) ---
    struct sampling_profile_t {
        uint64_t user_sampling_config = 0;
        int32_t  seed                 = 0;
        int32_t  min_keep             = 0;
        int32_t  top_k                = 40;
        int32_t  penalty_last_n       = 64;
        int32_t  mirostat             = 0;
        float    top_p                = 0.95f;
        float    min_p                = 0.05f;
        float    typ_p                = 1.00f;
        float    temp                 = 0.80f;
        float    dynatemp_range       = 0.00f;
        float    dynatemp_exponent    = 1.00f;
        float    penalty_repeat       = 1.00f;
        float    penalty_freq         = 0.00f;
        float    penalty_present      = 0.00f;
        float    xtc_probability      = 0.00f;
        float    xtc_threshold        = 0.10f;
        float    top_n_sigma          = -1.00f;
        float    mirostat_tau         = 5.00f;
        float    mirostat_eta         = 0.10f;
    } sampling_profile = {};
    int32_t requested_n_predict = -1; // per-request decode cap; -1 = receiver default
    // --- v1.5 extension (format_minor >= 5) ---
    uint32_t prompt_token_count         = 0; // canonical full prompt token count
    uint32_t prompt_text_bytes          = 0; // canonical prompt UTF-8 bytes
    uint32_t streamed_prefix_token_count = 0; // logical prompt prefix already covered by imported KV
    // --- v1.6 extension (format_minor >= 6) ---
    uint64_t pd_session_id              = 0; // PROMPT-DIRECT cache hit: Mac-side snapshot handle (0 = not a hit)
    // --- v1.7 extension (format_minor >= 7) — Phase 5.3 (2026-05-12) ---
    int32_t  source_slot_id             = -1; // RTX-side slot index that produced this blob.
                                              // -1 = unset (legacy single-slot path).
                                              // Mac receiver may use this to demultiplex
                                              // concurrent multi-slot blobs into separate
                                              // decode sessions when kv_unified is true.
                                              // Without Mac-side support, Mac ignores this
                                              // and behaves as legacy seq_id=0.
    // --- v1.8 extension (format_minor >= 8) — ship-embeddings (2026-06-01) ---
    // When img_cell_count > 0, the prompt payload carries img_cell_count*img_n_embd f32
    // image embeddings; the receiver re-prefills those cells [img_pos_start, +img_cell_count)
    // via an embd batch (correct Metal F32 FA) instead of using the RTX's F16-VKQ image K/V.
    uint32_t img_cell_count             = 0;  // # image cells (embedding rows); 0 = no image embd
    uint32_t img_n_embd                 = 0;  // embedding dim per cell (= model n_embd)
    uint32_t img_pos_start              = 0;  // absolute sequence position of the first image cell
    // --- v2.0 extension (format_minor >= 9) — MTP-on-KV-inject (2026-06-24) ---
    uint32_t mtp_hidden_n_tokens        = 0;  // # hidden rows (= streamed prefix token count); 0 = no MTP hidden states
    uint32_t mtp_hidden_n_embd          = 0;  // hidden dim per row (= model n_embd)
    // --- v2.1 extension (format_minor >= 10) — delta shipping (2026-09-10) ---
    // The decoder already holds cells [0, base_prefix_cells) of this sequence (it said so when the
    // prefill server asked); the payload's cells start at that position and are appended there.
    uint32_t base_prefix_cells          = 0;
};

struct llama_kv_artifact_summary {
    uint64_t payload_bytes = 0;
    uint32_t payload_crc32 = 0;
};

// Bit flags for llama_kv_artifact_metadata::flags.
constexpr uint32_t LLAMA_KV_ARTIFACT_FLAG_PROGRESSIVE   = 1u << 0;
constexpr uint32_t LLAMA_KV_ARTIFACT_FLAG_TRANSPORTED   = 1u << 1;
constexpr uint32_t LLAMA_KV_ARTIFACT_FLAG_PROMPT_DIRECT = 1u << 2;
// CACHE_HIT: prompt-direct artifact where Mac already holds the KV (no re-prefill needed).
// pd_session_id field (format_minor >= 6) identifies which snapshot to restore.
constexpr uint32_t LLAMA_KV_ARTIFACT_FLAG_CACHE_HIT     = 1u << 3;
// IMAGE_EMBD (v1.8, ship-embeddings): the prompt payload carries image embeddings the
// receiver must re-prefill (img_cell_count/img_n_embd/img_pos_start in metadata).
constexpr uint32_t LLAMA_KV_ARTIFACT_FLAG_IMAGE_EMBD    = 1u << 4;
// RAW_IMAGE (v1.9, small-image-forward): the prompt payload carries the raw image bytes
// (PNG/JPEG) at the very END; the receiver runs mlx_vlm generate() on them locally. The
// raw-image byte count is stored in the v1.8 image header's reserved_img (4th) field.
constexpr uint32_t LLAMA_KV_ARTIFACT_FLAG_RAW_IMAGE     = 1u << 5;
// MTP_HIDDEN (v2.0, 2026-06-24): the payload carries fp16 per-token last-layer hidden states at the very END (after
// raw_image), sized mtp_hidden_n_tokens*mtp_hidden_n_embd. The receiver warms the MTP sidecar with them over a
// KV-inject session (which otherwise lacks the prompt hidden states). See mtp_hidden_n_tokens/n_embd in metadata.
constexpr uint32_t LLAMA_KV_ARTIFACT_FLAG_MTP_HIDDEN    = 1u << 6;
// NO_CRC (2026-09-10, streamed export): payload_crc32 is 0 and was not computed. The RTX streams the
// payload straight from the NVMe tier's row files while the header has already gone out; a CRC would
// need a second pass over the files. Readers skip the check when this bit is set.
constexpr uint32_t LLAMA_KV_ARTIFACT_FLAG_NO_CRC        = 1u << 7;
// DELTA (v2.1, 2026-09-10): base_prefix_cells > 0; the payload carries only the cells the decoder lacks.
constexpr uint32_t LLAMA_KV_ARTIFACT_FLAG_DELTA         = 1u << 8;
// Native DSV4.1 payload adapter. Uses the existing v2.1 delta envelope;
// the payload is interpreted by its model-specific native state adapter.
constexpr uint32_t LLAMA_KV_ARTIFACT_FLAG_NATIVE_DSV41  = 1u << 9;

bool llama_kv_artifact_write(const std::string &               path,
                             const std::vector<uint8_t> &      payload,
                             const llama_kv_artifact_metadata &meta,
                             llama_kv_artifact_summary *       summary,
                             std::string *                     error,
                             const llama_kv_prompt_payload *   prompt_payload = nullptr);

// Serialize the same byte sequence as llama_kv_artifact_write but into a
// memory buffer (no file I/O). Output is bit-identical to the file path so
// downstream readers (llama_kv_artifact_read_mem on the receiver) can
// consume it transparently. Used by the in-memory transport path that
// avoids the SSD round-trip on every prefill request.
bool llama_kv_artifact_serialize_to_buffer(std::vector<uint8_t> &            buffer_out,
                                           const std::vector<uint8_t> &      payload,
                                           const llama_kv_artifact_metadata &meta,
                                           llama_kv_artifact_summary *       summary,
                                           std::string *                     error,
                                           const llama_kv_prompt_payload *   prompt_payload = nullptr);

// Header only (the same bytes serialize_to_buffer emits before the payload) for a payload whose size is
// known and whose bytes are produced elsewhere (the streamed export). extra_flags is OR-ed into
// meta.flags; raw_image_bytes fills the v1.8 image header's reserved field as the serializer does.
bool llama_kv_artifact_serialize_header(std::vector<uint8_t> &            header_out,
                                        const llama_kv_artifact_metadata &meta,
                                        uint64_t                          payload_bytes,
                                        uint32_t                          payload_crc32,
                                        uint32_t                          extra_flags,
                                        size_t                            raw_image_bytes = 0);

bool llama_kv_artifact_read(const std::string &         path,
                            std::vector<uint8_t> &      payload_out,
                            llama_kv_artifact_metadata *meta_out,
                            llama_kv_artifact_summary * summary_out,
                            std::string *               error,
                            llama_kv_prompt_payload *   prompt_payload_out = nullptr);

uint32_t llama_kv_artifact_crc32(const uint8_t * data, size_t size);

// Parse only the existing bounded header, without reading/allocating payload.
// This certifies header syntax only; the caller must validate identity, sizes,
// model-specific semantics and payload integrity before state publication.
bool llama_kv_artifact_parse_header(const uint8_t *data, size_t size,
    llama_kv_artifact_metadata &meta, llama_kv_artifact_summary &summary,
    size_t &header_bytes, std::string *error);

// Read artifact from memory buffer instead of file.
bool llama_kv_artifact_read_mem(const uint8_t *             data,
                                size_t                      size,
                                std::vector<uint8_t> &      payload_out,
                                llama_kv_artifact_metadata *meta_out,
                                llama_kv_artifact_summary * summary_out,
                                std::string *               error,
                                llama_kv_prompt_payload *   prompt_payload_out = nullptr);
