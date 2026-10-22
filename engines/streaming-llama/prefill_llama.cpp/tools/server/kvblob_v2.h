#pragma once

// KVBlob v2 — backend-agnostic KV-cache wire format for prefill->decode handoff.
//
// Wire spec lives at <runtime-path>  Python
// reference impl: <runtime-path>
//
// Header (72 bytes, little-endian):
//   uint32  magic           = 0x3242564B  ('KVB2' bytes-on-disk)
//   uint32  version         = 2
//   uint32  producer_id     (0=llama_cpp, 1=vllm, 2=ik_llama, 0xFF..=unknown)
//   uint32  producer_flags  (bitfield — see KVBLOB_V2_FLAG_* below)
//   uint32  n_layers
//   uint32  n_kv_heads
//   uint32  head_dim
//   uint32  k_dtype         (1=f16, 2=bf16, 3=f32, 16=q8_0, 17=q4_0)
//   uint32  v_dtype
//   uint32  mla_compressed  (0/1, alias of FLAG_MLA_LATENT)
//   uint32  n_seqs
//   uint32  total_positions
//   uint32  payload_offset
//   uint32  payload_bytes
//   uint32  reserved[4]
//
// Per-seq descriptor (48 bytes):
//   int32   seq_id
//   uint32  pos_start
//   uint32  pos_end
//   uint64  k_byte_offset       — absolute from blob[0]
//   uint64  v_byte_offset       — absolute from blob[0]
//   uint64  bytes_per_layer     — for OPAQUE layout, this is the seq's
//                                 total opaque-payload byte count (V offset
//                                 is unused).
//
// Multi-slot correctness rule: the consumer MUST iterate the descriptor
// table and place each (seq_id, pos) pair into its own logical slot.
// There is no "everything goes to seq 0" code path.  This is the
// architectural fix for the Phase 316z bug.

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#define KVBLOB_V2_MAGIC          0x3242564Bu  // 'KVB2' little-endian
#define KVBLOB_V2_VERSION        2u
#define KVBLOB_V2_HEADER_SIZE    72u
// Descriptor wire size: 4 (seq_id) + 4 (pos_start) + 4 (pos_end) +
// 8 (k_byte_offset) + 8 (v_byte_offset) + 8 (bytes_per_layer) = 36 bytes.
// Python reference: struct.calcsize("<iIIQQQ") == 36.
#define KVBLOB_V2_SEQDESC_SIZE   36u

// producer_id values
#define KVBLOB_V2_PRODUCER_LLAMA_CPP 0u
#define KVBLOB_V2_PRODUCER_VLLM      1u
#define KVBLOB_V2_PRODUCER_IK_LLAMA  2u

// producer_flags bitfield
#define KVBLOB_V2_FLAG_LAYOUT_OPAQUE  0x1u  // payload = native opaque blobs
#define KVBLOB_V2_FLAG_MLA_LATENT     0x2u  // alias of mla_compressed=1

// dtype ids
#define KVBLOB_V2_DTYPE_F16   1u
#define KVBLOB_V2_DTYPE_BF16  2u
#define KVBLOB_V2_DTYPE_F32   3u
#define KVBLOB_V2_DTYPE_Q8_0  16u
#define KVBLOB_V2_DTYPE_Q4_0  17u

struct kvblob_v2_header {
    uint32_t magic;
    uint32_t version;
    uint32_t producer_id;
    uint32_t producer_flags;
    uint32_t n_layers;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t k_dtype;
    uint32_t v_dtype;
    uint32_t mla_compressed;
    uint32_t n_seqs;
    uint32_t total_positions;
    uint32_t payload_offset;
    uint32_t payload_bytes;
    uint32_t reserved[4];
};

struct kvblob_v2_seq_desc {
    int32_t  seq_id;
    uint32_t pos_start;
    uint32_t pos_end;
    uint64_t k_byte_offset;
    uint64_t v_byte_offset;
    uint64_t bytes_per_layer;
};

// Parse a KVBlob v2 byte stream.  Returns true on success and fills
// `out_header` and `out_descs`.  On failure, `out_error` carries a
// human-readable message and the function returns false.  No allocation
// touches `body` — the descriptor table is parsed by offset arithmetic.
bool kvblob_v2_parse(const std::string & body,
                     kvblob_v2_header & out_header,
                     std::vector<kvblob_v2_seq_desc> & out_descs,
                     std::string & out_error);

// Build an OPAQUE-layout KVBlob v2 from an array of (seq_id, opaque_bytes)
// pairs.  Each opaque_bytes payload is the result of
// llama_state_seq_get_data_ext for that seq.  The wire layout:
//
//   [header][descs][seq0_opaque][seq1_opaque]...[seqN_opaque]
//
// The descriptor's bytes_per_layer carries the seq's full opaque size,
// k_byte_offset points at the start of that seq's opaque block,
// v_byte_offset is unused (set to 0).  pos_start/pos_end bracket the seq's
// known position range (best-effort; receiver doesn't need them for the
// opaque path).
//
// `n_layers`, `n_kv_heads`, `head_dim`, `k_dtype`, `v_dtype`,
// `mla_compressed` populate the header so consumers that DO understand
// the canonical layout can decline if dtype/MLA mismatch.
std::string kvblob_v2_build_opaque(uint32_t producer_id,
                                   uint32_t n_layers,
                                   uint32_t n_kv_heads,
                                   uint32_t head_dim,
                                   uint32_t k_dtype,
                                   uint32_t v_dtype,
                                   bool mla_compressed,
                                   const std::vector<llama_seq_id> & seq_ids,
                                   const std::vector<std::vector<uint8_t>> & per_seq_opaque,
                                   const std::vector<std::pair<uint32_t, uint32_t>> & per_seq_pos);

// Parse the X-KVBlob-SeqId-Remap header value: "src=dst,src=dst,..."
// Returns an empty map when the value is empty.  Throws std::invalid_argument
// on malformed input.
std::map<llama_seq_id, llama_seq_id>
kvblob_v2_parse_seq_id_remap_header(const std::string & header_value);

// Format a remap map back into the wire string form.  Useful for
// pass-through proxies and tests.
std::string kvblob_v2_format_seq_id_remap_header(
    const std::map<llama_seq_id, llama_seq_id> & remap);
