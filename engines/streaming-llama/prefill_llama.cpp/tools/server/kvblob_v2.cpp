#include "kvblob_v2.h"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace {

inline void write_u32(std::string & dst, uint32_t v) {
    char buf[4];
    buf[0] = static_cast<char>(v & 0xFF);
    buf[1] = static_cast<char>((v >> 8) & 0xFF);
    buf[2] = static_cast<char>((v >> 16) & 0xFF);
    buf[3] = static_cast<char>((v >> 24) & 0xFF);
    dst.append(buf, 4);
}

inline void write_i32(std::string & dst, int32_t v) {
    write_u32(dst, static_cast<uint32_t>(v));
}

inline void write_u64(std::string & dst, uint64_t v) {
    char buf[8];
    for (int i = 0; i < 8; ++i) {
        buf[i] = static_cast<char>((v >> (8 * i)) & 0xFF);
    }
    dst.append(buf, 8);
}

inline uint32_t read_u32(const char * p) {
    return  (static_cast<uint8_t>(p[0]))
          | (static_cast<uint32_t>(static_cast<uint8_t>(p[1])) << 8)
          | (static_cast<uint32_t>(static_cast<uint8_t>(p[2])) << 16)
          | (static_cast<uint32_t>(static_cast<uint8_t>(p[3])) << 24);
}

inline int32_t read_i32(const char * p) {
    return static_cast<int32_t>(read_u32(p));
}

inline uint64_t read_u64(const char * p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<uint64_t>(static_cast<uint8_t>(p[i])) << (8 * i);
    }
    return v;
}

} // namespace

bool kvblob_v2_parse(const std::string & body,
                     kvblob_v2_header & out_header,
                     std::vector<kvblob_v2_seq_desc> & out_descs,
                     std::string & out_error) {
    if (body.size() < KVBLOB_V2_HEADER_SIZE) {
        out_error = "KVBlob v2 body too short for header";
        return false;
    }
    const char * p = body.data();
    out_header.magic           = read_u32(p +  0);
    out_header.version         = read_u32(p +  4);
    out_header.producer_id     = read_u32(p +  8);
    out_header.producer_flags  = read_u32(p + 12);
    out_header.n_layers        = read_u32(p + 16);
    out_header.n_kv_heads      = read_u32(p + 20);
    out_header.head_dim        = read_u32(p + 24);
    out_header.k_dtype         = read_u32(p + 28);
    out_header.v_dtype         = read_u32(p + 32);
    out_header.mla_compressed  = read_u32(p + 36);
    out_header.n_seqs          = read_u32(p + 40);
    out_header.total_positions = read_u32(p + 44);
    out_header.payload_offset  = read_u32(p + 48);
    out_header.payload_bytes   = read_u32(p + 52);
    for (int i = 0; i < 4; ++i) {
        out_header.reserved[i] = read_u32(p + 56 + i * 4);
    }
    if (out_header.magic != KVBLOB_V2_MAGIC) {
        out_error = "bad magic (not 'KVB2')";
        return false;
    }
    if (out_header.version != KVBLOB_V2_VERSION) {
        out_error = "unsupported KVBlob version " + std::to_string(out_header.version);
        return false;
    }
    const size_t need_descs = static_cast<size_t>(out_header.n_seqs) * KVBLOB_V2_SEQDESC_SIZE;
    if (body.size() < KVBLOB_V2_HEADER_SIZE + need_descs) {
        out_error = "KVBlob v2 body too short for descriptor table";
        return false;
    }
    out_descs.clear();
    out_descs.reserve(out_header.n_seqs);
    for (uint32_t i = 0; i < out_header.n_seqs; ++i) {
        const char * dp = p + KVBLOB_V2_HEADER_SIZE + i * KVBLOB_V2_SEQDESC_SIZE;
        kvblob_v2_seq_desc d{};
        d.seq_id          = read_i32(dp + 0);
        d.pos_start       = read_u32(dp + 4);
        d.pos_end         = read_u32(dp + 8);
        d.k_byte_offset   = read_u64(dp + 12);
        d.v_byte_offset   = read_u64(dp + 20);
        d.bytes_per_layer = read_u64(dp + 28);
        out_descs.push_back(d);
    }
    return true;
}

std::string kvblob_v2_build_opaque(uint32_t producer_id,
                                   uint32_t n_layers,
                                   uint32_t n_kv_heads,
                                   uint32_t head_dim,
                                   uint32_t k_dtype,
                                   uint32_t v_dtype,
                                   bool mla_compressed,
                                   const std::vector<llama_seq_id> & seq_ids,
                                   const std::vector<std::vector<uint8_t>> & per_seq_opaque,
                                   const std::vector<std::pair<uint32_t, uint32_t>> & per_seq_pos) {
    if (seq_ids.size() != per_seq_opaque.size() ||
        seq_ids.size() != per_seq_pos.size()) {
        throw std::invalid_argument(
            "kvblob_v2_build_opaque: seq_ids, per_seq_opaque, and per_seq_pos "
            "must have the same length");
    }

    const uint32_t n_seqs = static_cast<uint32_t>(seq_ids.size());
    const uint32_t payload_offset = KVBLOB_V2_HEADER_SIZE + n_seqs * KVBLOB_V2_SEQDESC_SIZE;

    // Compute per-seq absolute offsets in the payload.
    std::vector<uint64_t> abs_offsets(n_seqs);
    uint64_t cursor = payload_offset;
    uint32_t total_positions = 0;
    for (uint32_t i = 0; i < n_seqs; ++i) {
        abs_offsets[i] = cursor;
        cursor += per_seq_opaque[i].size();
        total_positions += per_seq_pos[i].second - per_seq_pos[i].first;
    }
    const uint32_t payload_bytes = static_cast<uint32_t>(cursor - payload_offset);

    std::string out;
    out.reserve(static_cast<size_t>(cursor));

    // Header
    write_u32(out, KVBLOB_V2_MAGIC);
    write_u32(out, KVBLOB_V2_VERSION);
    write_u32(out, producer_id);
    write_u32(out, KVBLOB_V2_FLAG_LAYOUT_OPAQUE | (mla_compressed ? KVBLOB_V2_FLAG_MLA_LATENT : 0u));
    write_u32(out, n_layers);
    write_u32(out, n_kv_heads);
    write_u32(out, head_dim);
    write_u32(out, k_dtype);
    write_u32(out, v_dtype);
    write_u32(out, mla_compressed ? 1u : 0u);
    write_u32(out, n_seqs);
    write_u32(out, total_positions);
    write_u32(out, payload_offset);
    write_u32(out, payload_bytes);
    for (int i = 0; i < 4; ++i) write_u32(out, 0u);

    // Descriptor table
    for (uint32_t i = 0; i < n_seqs; ++i) {
        write_i32(out, seq_ids[i]);
        write_u32(out, per_seq_pos[i].first);
        write_u32(out, per_seq_pos[i].second);
        write_u64(out, abs_offsets[i]);
        write_u64(out, 0);                                  // v_byte_offset unused for opaque
        write_u64(out, static_cast<uint64_t>(per_seq_opaque[i].size()));
    }

    // Payload: per-seq opaque blobs concatenated in the same order as descriptors
    for (uint32_t i = 0; i < n_seqs; ++i) {
        out.append(reinterpret_cast<const char *>(per_seq_opaque[i].data()),
                   per_seq_opaque[i].size());
    }
    return out;
}

std::map<llama_seq_id, llama_seq_id>
kvblob_v2_parse_seq_id_remap_header(const std::string & header_value) {
    std::map<llama_seq_id, llama_seq_id> out;
    if (header_value.empty()) {
        return out;
    }
    std::stringstream ss(header_value);
    std::string piece;
    while (std::getline(ss, piece, ',')) {
        // trim
        auto a = piece.find_first_not_of(" \t");
        auto b = piece.find_last_not_of(" \t");
        if (a == std::string::npos) continue;
        piece = piece.substr(a, b - a + 1);
        const auto eq = piece.find('=');
        if (eq == std::string::npos) {
            throw std::invalid_argument(
                "kvblob_v2_parse_seq_id_remap_header: missing '=' in piece '" +
                piece + "'");
        }
        const std::string lhs = piece.substr(0, eq);
        const std::string rhs = piece.substr(eq + 1);
        try {
            out[static_cast<llama_seq_id>(std::stoi(lhs))]
                = static_cast<llama_seq_id>(std::stoi(rhs));
        } catch (const std::exception & e) {
            throw std::invalid_argument(
                std::string("kvblob_v2_parse_seq_id_remap_header: bad number: ") +
                e.what());
        }
    }
    return out;
}

std::string kvblob_v2_format_seq_id_remap_header(
    const std::map<llama_seq_id, llama_seq_id> & remap) {
    std::string out;
    bool first = true;
    for (const auto & kv : remap) {
        if (!first) out.push_back(',');
        first = false;
        out += std::to_string(kv.first);
        out.push_back('=');
        out += std::to_string(kv.second);
    }
    return out;
}
