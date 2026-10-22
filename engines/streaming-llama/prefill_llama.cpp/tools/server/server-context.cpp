#include "server-context.h"
#include "server-common.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "kvblob_v2.h"

#include "common.h"
#include "llama.h"
#include "llama-token-stream.h"
#include "llama-tb-transport.h"
#include "llama-context.h"
#include "llama-mtp-policy.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstddef>
#include <cinttypes>
#include <memory>
#include <filesystem>

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

// ---------------------------------------------------------------------------
// PROMPT-DIRECT KV-handshake (Gate #2) — RTX side helpers
//
// xxhash: use XXH_INLINE_ALL so all symbols are static/inline.
// Safe to include in multiple TUs without ODR conflicts.
// ---------------------------------------------------------------------------
#define XXH_INLINE_ALL
#include "../../examples/gguf-hash/deps/xxhash/xxhash.h"

// TBP1 control socket protocol constants (must match llama-tb-transport.cpp)
static constexpr uint32_t PD_TBP_MAGIC       = 0x54425031u; // "TBP1"
static constexpr uint16_t PD_MSG_QUERY        = 30;
static constexpr uint16_t PD_MSG_REPLY        = 31;

struct pd_prefix_cache_reply {
    bool     hit;
    uint32_t cached_len_tokens;
    uint64_t session_id;
};

// Inline helper: xxhash64 of int32_t token array
static uint64_t pd_hash_tokens(const int32_t * tokens, int32_t n) {
    if (!tokens || n <= 0) { return 0; }
    return XXH64(tokens, (size_t)n * sizeof(int32_t), 0);
}

// Option B (LLAMA_DSV4_MAC_RETAIN_HANDLE): FNV-1a over int32 token IDs, computed
// BYTE-FOR-BYTE the way the Mac receiver's _hash_token_ids does (little-endian per
// token, 4 bytes each). This is DELIBERATELY not pd_hash_tokens (XXH64) — the two
// sides must agree, and the Mac uses FNV-1a. The result is masked to 63 bits so it
// round-trips losslessly through the token-stream JSON transport, whose numeric parse
// (json_get_int -> strtoll) is signed int64 and would clamp any value > INT64_MAX.
// The Mac masks its emitted hash identically (_send_done retained_prefix_hash).
static uint64_t mac_fnv1a_tokens(const int32_t * t, int32_t n) {
    if (!t || n <= 0) { return 0; }
    uint64_t h = 0xCBF29CE484222325ULL;
    for (int32_t i = 0; i < n; i++) {
        uint32_t v = (uint32_t) t[i];
        for (int s = 0; s < 32; s += 8) {
            h ^= (v >> s) & 0xFF;
            h *= 0x100000001B3ULL;
        }
    }
    return h & 0x7FFFFFFFFFFFFFFFULL;
}

static void splice_append_lineage_field(std::string & dst, const std::string & value) {
    dst += std::to_string(value.size());
    dst.push_back(':');
    dst.append(value);
}

static void splice_append_lineage_message(
        std::string & dst, const std::string & role, const std::string & content) {
    splice_append_lineage_field(dst, role);
    splice_append_lineage_field(dst, content);
}

// Inline helper: xxhash64 of model description string
static uint64_t pd_model_fingerprint(const char * desc) {
    if (!desc || !desc[0]) { return 0; }
    return XXH64(desc, strlen(desc), 0);
}

// Socket helpers (self-contained, no link deps)
#ifndef _WIN32
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int pd_sock_t;
static constexpr pd_sock_t PD_SOCK_INVALID = -1;
#define PD_CLOSE(fd) ::close(fd)
#else
#ifndef _WINSOCK_INCLUDED_
#include <winsock2.h>
#include <ws2tcpip.h>
#endif
typedef SOCKET pd_sock_t;
static const pd_sock_t PD_SOCK_INVALID = INVALID_SOCKET;
#define PD_CLOSE(fd) ::closesocket(fd)
#endif

static void pd_append_u16(std::vector<uint8_t> & v, uint16_t x) {
    v.push_back((uint8_t)(x & 0xFF)); v.push_back((uint8_t)(x >> 8));
}
static void pd_append_u32(std::vector<uint8_t> & v, uint32_t x) {
    for (int i = 0; i < 4; ++i) { v.push_back((uint8_t)(x >> (8*i))); }
}
static void pd_append_u64(std::vector<uint8_t> & v, uint64_t x) {
    for (int i = 0; i < 8; ++i) { v.push_back((uint8_t)(x >> (8*i))); }
}

static uint32_t pd_crc32(const uint8_t * data, size_t len) {
    // Use llama-kv-artifact CRC32 (compiled into llama.lib) via the artifact header.
    // We just include a trivial implementation inline to avoid any link dep.
    uint32_t crc = 0xFFFFFFFFu;
    static const uint32_t tbl[256] = {
        0,          0x77073096, 0xEE0E612C, 0x990951BA, 0x076DC419, 0x706AF48F,
        0xE963A535, 0x9E6495A3, 0x0EDB8832, 0x79DCB8A4, 0xE0D5E91B, 0x97D2D988,
        0x09B64C2B, 0x7EB17CBF, 0xE7B82D08, 0x90BF1C35, 0x1DB71064, 0x6AB020F2,
        0xF3B97148, 0x84BE41DE, 0x1ADAD47D, 0x6DDDE4EB, 0xF4D4B551, 0x83D385C7,
        0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC, 0x14015C4F, 0x63066CD9,
        0xFA0F3D63, 0x8D080DF5, 0x3B6E20C8, 0x4C69105E, 0xD56041E4, 0xA2677172,
        0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B, 0x35B5A8FA, 0x42B2986C,
        0xDBBBC9D6, 0xACBCF940, 0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59,
        0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116, 0x21B4F927, 0x56B3C423,
        0xCFBA9599, 0xB8BDA50F, 0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924,
        0x2F6F7C87, 0x58684C11, 0xC1611DAB, 0xB6662D3D, 0x76DC4190, 0x01DB7106,
        0x98D220BC, 0xEFD5102A, 0x71B18589, 0x06B6B51F, 0x9FBFE4A5, 0xE8B8D433,
        0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818, 0x7F6A0DBB, 0x086D3D2D,
        0x91646C97, 0xE6635C01, 0x6B6B51F4, 0x1C6C6162, 0x856530D8, 0xF262004E,
        0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457, 0x65B0D9C6, 0x12B7E950,
        0x8BBEB8EA, 0xFCB9887C, 0x62DD1D7F, 0x15DA2D49, 0x8CD37CF3, 0xFBD44C65,
        0x4DB26158, 0x3AB551CE, 0xA3BC0074, 0xD4BB30E2, 0x4ADFA541, 0x3DD895D7,
        0xA4D1C46D, 0xD3D6F4FB, 0x4369E96A, 0x346ED9FC, 0xAD678846, 0xDA60B8D0,
        0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7CC9, 0x5005713C, 0x270241AA,
        0xBE0B1010, 0xC90C2086, 0x5768B525, 0x206F85B3, 0xB966D409, 0xCE61E49F,
        0x5EDEF90E, 0x29D9C998, 0xB0D09822, 0xC7D7A8B4, 0x59B33D17, 0x2EB40D81,
        0xB7BD5C3B, 0xC0BA6CAD, 0xEDB88320, 0x9ABFB3B6, 0x03B6E20C, 0x74B1D29A,
        0xEAD54739, 0x9DD277AF, 0x04DB2615, 0x73DC1683, 0xE3630B12, 0x94643B84,
        0x0D6D6A3E, 0x7A6A5AA8, 0xE40ECF0B, 0x9309FF9D, 0x0A00AE27, 0x7D079EB1,
        0xF00F9344, 0x8708A3D2, 0x1E01F268, 0x6906C2FE, 0xF762575D, 0x806567CB,
        0x196C3671, 0x6E6B06E7, 0xFED41B76, 0x89D32BE0, 0x10DA7A5A, 0x67DD4ACC,
        0xF9B9DF6F, 0x8EBEEFF9, 0x17B7BE43, 0x60B08ED5, 0xD6D6A3E8, 0xA1D1937E,
        0x38D8C2C4, 0x4FDFF252, 0xD1BB67F1, 0xA6BC5767, 0x3FB506DD, 0x48B2364B,
        0xD80D2BDA, 0xAF0A1B4C, 0x36034AF6, 0x41047A60, 0xDF60EFC3, 0xA8670955,
        0x316658EF, 0x466CA279, 0xB40BBE37, 0xC30C8EA1, 0x5A05DF1B, 0x2D02EF8D
    };
    for (size_t i = 0; i < len; ++i) {
        crc = tbl[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

// Build a minimal TBP1 frame with the given payload and send it over fd.
// Returns false on error. Simplified version of send_tbp_frame() from
// llama-tb-transport.cpp — keeps server-context.cpp self-contained.
static bool pd_send_tbp_frame(pd_sock_t fd, uint16_t msg_type,
                               const uint8_t * payload, size_t payload_bytes) {
    std::vector<uint8_t> hdr;
    hdr.reserve(52);
    pd_append_u32(hdr, PD_TBP_MAGIC);
    pd_append_u16(hdr, 1);                  // proto major
    pd_append_u16(hdr, 0);                  // proto minor
    pd_append_u16(hdr, msg_type);
    pd_append_u16(hdr, 0);                  // flags
    pd_append_u32(hdr, 52u);               // hdr_size
    pd_append_u32(hdr, (uint32_t)payload_bytes);
    pd_append_u64(hdr, 0);                  // session_id
    pd_append_u64(hdr, 0);                  // stream_id
    pd_append_u64(hdr, 0);                  // seq_no
    const size_t hdr_crc_off = hdr.size();
    pd_append_u32(hdr, 0u);                // hdr crc placeholder
    pd_append_u32(hdr, payload_bytes > 0 ? pd_crc32(payload, payload_bytes) : 0u);
    const uint32_t hdr_crc = pd_crc32(hdr.data(), hdr.size());
    hdr[hdr_crc_off+0] = (uint8_t)(hdr_crc & 0xFF);
    hdr[hdr_crc_off+1] = (uint8_t)((hdr_crc >> 8) & 0xFF);
    hdr[hdr_crc_off+2] = (uint8_t)((hdr_crc >> 16) & 0xFF);
    hdr[hdr_crc_off+3] = (uint8_t)((hdr_crc >> 24) & 0xFF);

    size_t off = 0;
    const size_t total = hdr.size();
    while (off < total) {
#ifndef _WIN32
        const ssize_t n = ::send(fd, hdr.data() + off, total - off, 0);
#else
        const int n = ::send(fd, (const char*)(hdr.data() + off), (int)(total - off), 0);
#endif
        if (n <= 0) { return false; }
        off += (size_t)n;
    }
    if (payload_bytes > 0 && payload) {
        off = 0;
        while (off < payload_bytes) {
#ifndef _WIN32
            const ssize_t n = ::send(fd, payload + off, payload_bytes - off, 0);
#else
            const int n = ::send(fd, (const char*)(payload + off), (int)(payload_bytes - off), 0);
#endif
            if (n <= 0) { return false; }
            off += (size_t)n;
        }
    }
    return true;
}

// Receive exactly 'size' bytes from fd with a timeout_ms deadline.
static bool pd_recv_all(pd_sock_t fd, uint8_t * dst, size_t size, int timeout_ms) {
    size_t off = 0;
    while (off < size) {
        fd_set rfds; FD_ZERO(&rfds); FD_SET(fd, &rfds);
        struct timeval tv = {};
        tv.tv_sec  = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
#ifndef _WIN32
        if (::select((int)fd + 1, &rfds, nullptr, nullptr, &tv) <= 0) { return false; }
        const ssize_t n = ::recv(fd, dst + off, size - off, 0);
#else
        if (::select(0, &rfds, nullptr, nullptr, &tv) <= 0) { return false; }
        const int n = ::recv(fd, (char*)(dst + off), (int)(size - off), 0);
#endif
        if (n <= 0) { return false; }
        off += (size_t)n;
    }
    return true;
}

// Query the Mac decode receiver's prefix cache via control socket.
// Returns true and fills *reply on success (hit or miss).
// Returns false on connection failure or timeout (treat as miss, fall back).
static bool pd_query_prefix_cache(const std::string & host, int ctrl_port,
                                   uint64_t prefix_hash, uint32_t prefix_len,
                                   uint64_t model_fp, pd_prefix_cache_reply * reply) {
    static bool s_warned_once = false;
    if (!reply || host.empty() || ctrl_port <= 0) { return false; }
    // Safety env gate: LLAMA_PD_CACHE_DISABLE=1 → immediate miss (no query sent)
    if (const char * dis = std::getenv("LLAMA_PD_CACHE_DISABLE")) {
        if (dis[0] == '1') {
            *reply = {false, 0, 0};
            return false;
        }
    }
    *reply = {false, 0, 0};

#ifdef _WIN32
    { WSADATA w; static bool init_done = false;
      if (!init_done) { WSAStartup(MAKEWORD(2,2), &w); init_done = true; } }
#endif

    // Connect
    const std::string port_str = std::to_string(ctrl_port);
    struct addrinfo hints = {};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo * ai = nullptr;
    if (::getaddrinfo(host.c_str(), port_str.c_str(), &hints, &ai) != 0 || !ai) {
        if (!s_warned_once) {
            fprintf(stderr, "[pd-cache] PROMPT-DIRECT cache query: getaddrinfo failed for %s:%d — falling back\n",
                    host.c_str(), ctrl_port);
            s_warned_once = true;
        }
        return false;
    }
    pd_sock_t fd = PD_SOCK_INVALID;
    for (auto * cur = ai; cur; cur = cur->ai_next) {
        fd = ::socket(cur->ai_family, cur->ai_socktype, cur->ai_protocol);
        if (fd == PD_SOCK_INVALID) { continue; }
        if (::connect(fd, cur->ai_addr, (int)cur->ai_addrlen) == 0) { break; }
        PD_CLOSE(fd); fd = PD_SOCK_INVALID;
    }
    ::freeaddrinfo(ai);
    if (fd == PD_SOCK_INVALID) {
        if (!s_warned_once) {
            fprintf(stderr, "[pd-cache] PROMPT-DIRECT cache query: connect to %s:%d failed — falling back to full send\n",
                    host.c_str(), ctrl_port);
            s_warned_once = true;
        }
        return false;
    }
    s_warned_once = false; // reset on success

    // Build and send TBP_MSG_QUERY_PREFIX_CACHE payload (20 bytes)
    std::vector<uint8_t> qpay;
    qpay.reserve(20);
    pd_append_u64(qpay, prefix_hash);
    pd_append_u32(qpay, prefix_len);
    pd_append_u64(qpay, model_fp);
    if (!pd_send_tbp_frame(fd, PD_MSG_QUERY, qpay.data(), qpay.size())) {
        fprintf(stderr, "[pd-cache] Failed to send cache query frame\n");
        PD_CLOSE(fd); return false;
    }

    // Receive TBP1 reply header (52 bytes) with 50 ms timeout
    constexpr int TIMEOUT_MS = 50;
    uint8_t hdr[52] = {};
    if (!pd_recv_all(fd, hdr, 52, TIMEOUT_MS)) {
        fprintf(stderr, "[pd-cache] PROMPT-DIRECT cache query timed out (%d ms) — falling back to full send\n",
                TIMEOUT_MS);
        PD_CLOSE(fd); return false;
    }

    // Parse header fields
    const uint16_t msg_type    = (uint16_t)hdr[8] | ((uint16_t)hdr[9] << 8);
    const uint32_t payload_len = (uint32_t)hdr[12] | ((uint32_t)hdr[13] << 8) |
                                  ((uint32_t)hdr[14] << 16) | ((uint32_t)hdr[15] << 24);

    if (msg_type != PD_MSG_REPLY) {
        fprintf(stderr, "[pd-cache] Unexpected reply msg_type=%u\n", (unsigned)msg_type);
        PD_CLOSE(fd); return false;
    }
    if (payload_len < 13) {
        fprintf(stderr, "[pd-cache] Reply payload too short: %u bytes\n", payload_len);
        PD_CLOSE(fd); return false;
    }

    // Read payload
    std::vector<uint8_t> rpay(payload_len);
    if (!pd_recv_all(fd, rpay.data(), payload_len, TIMEOUT_MS)) {
        PD_CLOSE(fd); return false;
    }
    PD_CLOSE(fd);

    reply->hit = (rpay[0] != 0);
    reply->cached_len_tokens = (uint32_t)rpay[1] | ((uint32_t)rpay[2] << 8) |
                                ((uint32_t)rpay[3] << 16) | ((uint32_t)rpay[4] << 24);
    reply->session_id = 0;
    for (int i = 0; i < 8; ++i) {
        reply->session_id |= ((uint64_t)rpay[5+i]) << (8*i);
    }
    fprintf(stderr, "[pd-cache] Cache reply: hit=%d cached_len=%u sid=0x%llx\n",
            (int)reply->hit, reply->cached_len_tokens,
            (unsigned long long)reply->session_id);
    return true;
}

using json = nlohmann::ordered_json;

constexpr int HTTP_POLLING_SECONDS = 1;

static llama_kv_artifact_metadata::sampling_profile_t make_handoff_sampling_profile(const common_params_sampling & sampling) {
    llama_kv_artifact_metadata::sampling_profile_t profile;
    profile.user_sampling_config = sampling.user_sampling_config;
    profile.seed                 = (int32_t) sampling.seed;
    profile.min_keep             = sampling.min_keep;
    profile.top_k                = sampling.top_k;
    profile.penalty_last_n       = sampling.penalty_last_n;
    profile.mirostat             = sampling.mirostat;
    profile.top_p                = sampling.top_p;
    profile.min_p                = sampling.min_p;
    profile.typ_p                = sampling.typ_p;
    profile.temp                 = sampling.temp;
    profile.dynatemp_range       = sampling.dynatemp_range;
    profile.dynatemp_exponent    = sampling.dynatemp_exponent;
    profile.penalty_repeat       = sampling.penalty_repeat;
    profile.penalty_freq         = sampling.penalty_freq;
    profile.penalty_present      = sampling.penalty_present;
    profile.xtc_probability      = sampling.xtc_probability;
    profile.xtc_threshold        = sampling.xtc_threshold;
    profile.top_n_sigma          = sampling.top_n_sigma;
    profile.mirostat_tau         = sampling.mirostat_tau;
    profile.mirostat_eta         = sampling.mirostat_eta;
    return profile;
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

enum server_state {
    SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,          // Server is ready and model is loaded
};

struct route_reuse_retention_record {
    uint64_t model_layout_fingerprint = 0;
    uint64_t slot_generation = 0;
    std::vector<llama_token> prefix_tokens;
    uint64_t prefix_hash = 0;
    size_t prefix_count = 0;
    uint64_t blob_generation = 0;
    bool physical_state_valid = false;
    bool valid = false;

    void clear() {
        model_layout_fingerprint = 0;
        slot_generation = 0;
        prefix_tokens.clear();
        prefix_hash = 0;
        prefix_count = 0;
        blob_generation = 0;
        physical_state_valid = false;
        valid = false;
    }
};

struct mac_splice_state {
    bool valid = false;
    uint64_t generation = 0;
    llama_tokens canonical_tokens;
    uint32_t retained_count = 0;
    uint64_t retained_hash = 0;
    std::string expected_message_prefix;
    std::string assistant_text;
    std::string model_fingerprint;
    std::string template_fingerprint;
    std::string render_fingerprint;
    llama_tokens logical_prompt_sent;

    void clear() {
        valid = false;
        ++generation;
        canonical_tokens.clear();
        retained_count = 0;
        retained_hash = 0;
        expected_message_prefix.clear();
        assistant_text.clear();
        model_fingerprint.clear();
        template_fingerprint.clear();
        render_fingerprint.clear();
        logical_prompt_sent.clear();
    }
};

struct server_slot {
    int id;

    // TODO: change to unique_ptrs for consistency:
    llama_context * ctx = nullptr;

    // multimodal
    mtmd_context * mctx = nullptr;

    common_speculative * spec = nullptr;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_keep      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;

    int32_t n_prompt_tokens_cache     = 0;
    int32_t n_prompt_tokens_processed = 0;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    llama_tokens generated_tokens;

    // idx of draft tokens in the main batch
    // non-empty if we went to evaluate draft tokens
    // ref: https://github.com/ggml-org/llama.cpp/pull/17808
    std::vector<int32_t> i_batch_dft;

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;

    // --reasoning-budget N: how many tokens have been generated inside the thinking block, and
    // whether we have already forced it closed (so we only ever do it once per generation).
    int    n_reasoning_tokens      = 0;
    bool   reasoning_budget_closed = false;
    bool   reasoning_open          = false;
    size_t reasoning_scan_from     = 0;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;
    uint64_t route_reuse_generation = 1;
    route_reuse_retention_record route_reuse_record;
    // Default-empty and never populated unless LLAMA_DSV4_SPLICE_GENERATED_IDS=1;
    // therefore gate-off slot lifetime/reset behavior is unchanged.
    mac_splice_state mac_splice;

    void prompt_save(server_prompt_cache & prompt_cache) const {
        GGML_ASSERT(prompt.data.size() == 0);

        const size_t cur_size = llama_state_seq_get_size_ext(ctx, id, 0);

        SRV_WRN(" - saving prompt with length %d, total state size = %.3f MiB\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size);
        if (cur == nullptr) {
            return;
        }

        llama_state_seq_get_data_ext(ctx, cur->data.data(), cur_size, id, 0);
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear(bool allow_processing) {
        if (!allow_processing) {
            GGML_ASSERT(!is_processing());
        }

        // TURN-2 REUSE FIX (2026-07-18): the release-time prompt_clear (release()->prompt_clear for
        // child slots or LLAMA_SERVER_RELEASE_FREES_SLOT_KV) fires AFTER a streaming remote-handoff has
        // exported+migrated+retained the KV but BEFORE commit_route_reuse_record() commits it — wiping the
        // fully-valid retained candidate (slot.prompt.tokens + ctx generation/prefix/resident) so the commit
        // saw candidate_ready=0 and turn-2 ALWAYS re-prefilled. Preserve a valid retained-KV candidate here;
        // it is validated (prefix hash + fingerprint + generation) before any reuse, so keeping it is safe.
        if (ctx) {
            auto * ctx_i = static_cast<llama_context *>(ctx);
            if (ctx_i->is_route_reuse_enabled() && ctx_i->route_reuse_candidate_ready()) {
                SLT_INF(*this, "prompt_clear: preserving valid retained-KV candidate (%zu tokens) for cross-turn reuse\n",
                        prompt.tokens.size());
                return;
            }
        }

        SLT_INF(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        route_reuse_record.clear();
        if (++route_reuse_generation == 0) {
            ++route_reuse_generation;
        }
        llama_memory_seq_rm(llama_get_memory(ctx), id, -1, -1);
        if (ctx) {
            auto * ctx_i = static_cast<llama_context *>(ctx);
            if (ctx_i->is_route_reuse_enabled()) {
                ctx_i->invalidate_route_reuse_state();
                ctx_i->clear_kv_ram_state();
            }
        }
        prompt.tokens.clear();
        prompt.data.clear();
        prompt.checkpoints.clear();
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token  sampled; // in speculative mode, this is the last accepted token
    llama_tokens drafted;
    std::vector<std::vector<llama_token_data>> draft_dists;
    bool native_mtp_stream_local = false;

    // stats
    size_t n_sent_text = 0; // number of sent text character

    int64_t t_start_process_prompt;
    int64_t t_start_generation;

    double t_prompt_processing; // ms
    double t_token_generation;  // ms

    // Authoritative emitted-token timing supplied by a remote Mac receiver.
    double relay_decode_elapsed_ms = 0.0;
    int32_t relay_decode_intervals = 0;
    double relay_decode_tok_s = 0.0;
    double relay_decode_tok_s_early = 0.0;
    double relay_decode_tok_s_middle = 0.0;
    double relay_decode_tok_s_late = 0.0;

    std::function<void(int /* id_slot */)> callback_on_release;

    // Speculative decoding stats
    int32_t n_draft_total = 0;      // Total draft tokens generated
    int32_t n_draft_accepted = 0;   // Draft tokens actually accepted

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        n_prompt_tokens_cache = 0;

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;

        n_reasoning_tokens      = 0;
        reasoning_budget_closed = false;
        reasoning_open          = false;
        reasoning_scan_from     = 0;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        drafted.clear();
        draft_dists.clear();
        i_batch_dft.clear();
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        // clear speculative decoding stats
        n_draft_total = 0;
        n_draft_accepted = 0;
        relay_decode_elapsed_ms = 0.0;
        relay_decode_intervals = 0;
        relay_decode_tok_s = 0.0;
        relay_decode_tok_s_early = 0.0;
        relay_decode_tok_s_middle = 0.0;
        relay_decode_tok_s_late = 0.0;

        task_prev = std::move(task);
        task.reset();

        llama_set_sampler(ctx, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_INF(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    bool can_split() const {
        GGML_ASSERT(task);

        return
            !task->need_embd() ||
            (llama_get_memory(ctx) && llama_pooling_type(ctx) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type && are_lora_equal(lora, other_slot.lora);
    }

    bool has_budget(const common_params & global_params) {
        GGML_ASSERT(task);

        if (task->params.n_predict == -1 && global_params.n_predict == -1) {
            return true; // limitless
        }

        n_remaining = -1;

        if (task->params.n_predict != -1) {
            n_remaining = task->params.n_predict - n_decoded;
        } else if (global_params.n_predict != -1) {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0; // no budget
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return !!spec || native_mtp_stream_local;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        int n_draft_max = task->params.speculative.n_max;

        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        n_draft_max = std::min(n_draft_max, n_ctx - prompt.n_tokens() - 2);

        if (n_remaining > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        if (n_draft_max < task->params.speculative.n_min) {
            SLT_DBG(*this, "the max possible draft is too small: %d < %d - skipping speculative decoding\n", n_draft_max, task->params.speculative.n_min);
            n_draft_max = 0;
        }

        return n_draft_max;
    }

    llama_tokens draft_native_mtp_stream_local(
            const common_params_speculative & params_spec,
            int n_draft_max,
            std::vector<std::vector<llama_token_data>> & out_dists) {
        llama_tokens result;
        out_dists.clear();

        if (n_draft_max <= 0) {
            return result;
        }

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t n_embd  = llama_model_n_embd(model);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        if (n_embd <= 0 || n_vocab <= 0) {
            SLT_WRN(*this, "native MTP stream-local disabled for bad dims n_embd=%d n_vocab=%d\n", n_embd, n_vocab);
            return result;
        }

        std::vector<float> hidden((size_t) n_embd);
        std::vector<float> next_hidden((size_t) n_embd);
        std::vector<float> logits((size_t) n_vocab);

        llama_pos hidden_pos = -1;
        llama_seq_id hidden_seq = id;
        const int32_t got_hidden = llama_get_mtp_last_hidden(ctx, hidden.data(), n_embd, &hidden_pos, &hidden_seq);
        if (got_hidden < 0) {
            SLT_DBG(*this, "native MTP stream-local warmup skipped: no captured target hidden (%d)\n", got_hidden);
            return result;
        }

        common_sampler_ptr draft_smpl(common_sampler_clone(smpl.get()));
        if (!draft_smpl) {
            SLT_WRN(*this, "%s", "native MTP stream-local failed to clone sampler\n");
            return result;
        }

        llama_token token = sampled;
        llama_pos token_pos = prompt.tokens.pos_next();

        result.reserve((size_t) n_draft_max);
        out_dists.reserve((size_t) n_draft_max);

        for (int i = 0; i < n_draft_max; ++i) {
            const int32_t ret = llama_mtp_draft_step(
                ctx,
                hidden.data(),
                token,
                (int32_t) token_pos,
                id,
                logits.data(),
                next_hidden.data(),
                n_embd);
            if (ret != 0) {
                SLT_WRN(*this, "native MTP stream-local draft step failed at i=%d ret=%d\n", i, ret);
                break;
            }

            const llama_token drafted_id = common_sampler_sample_from_logits(
                draft_smpl.get(), logits.data(), n_vocab, true);
            if (drafted_id == LLAMA_TOKEN_NULL) {
                SLT_WRN(*this, "native MTP stream-local sampler returned null token at i=%d\n", i);
                break;
            }
            auto * cur_p = common_sampler_get_candidates(draft_smpl.get(), true);
            if (!cur_p || !cur_p->data || cur_p->size == 0 || cur_p->selected < 0) {
                SLT_WRN(*this, "native MTP stream-local sampler returned no candidates at i=%d\n", i);
                break;
            }

            const std::vector<llama_token_data> draft_dist(cur_p->data, cur_p->data + cur_p->size);

            common_sampler_accept(draft_smpl.get(), drafted_id, true);
            result.push_back(drafted_id);
            out_dists.push_back(draft_dist);

            if (params_spec.n_max <= (int) result.size()) {
                break;
            }

            hidden.swap(next_hidden);
            token = drafted_id;
            ++token_pos;
        }

        return result;
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used        =  ggml_time_us();
            t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            if (task->is_child()) {
                prompt_clear(false);
            }

            // Phase 329 (2026-05-05): when LLAMA_SERVER_RELEASE_FREES_SLOT_KV=1
            // is set, also clear the parent slot's KV on release. This is
            // required for kv_unified mode at high NP (=64+) where cells
            // accumulate across slots in stream 0; without explicit eviction
            // find_slot can't recycle cells from completed-but-not-cleared
            // seqs (no SWA → SWA-evict path doesn't fire). Trade-off: loses
            // cross-request prompt cache for this slot. Default off; enable
            // only when running with high NP and you don't need cross-call
            // prompt caching for individual slots (fenrir's shared-prefix
            // architecture handles caching at a different layer).
            if (!task->is_child()) {
                const char * release_clear_env = std::getenv("LLAMA_SERVER_RELEASE_FREES_SLOT_KV");
                if (release_clear_env && release_clear_env[0] && release_clear_env[0] != '0') {
                    SLT_DBG(*this, "%s", "Phase 329: clearing slot KV on release\n");
                    prompt_clear(false);
                }
            }

            reset();

            callback_on_release(id);
        }
    }

    result_timings get_timings() const {
        result_timings timings;
        timings.cache_n = n_prompt_tokens_cache;

        timings.prompt_n            = n_prompt_tokens_processed;
        timings.prompt_ms           = t_prompt_processing;
        timings.prompt_per_token_ms = t_prompt_processing / n_prompt_tokens_processed;
        timings.prompt_per_second   = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        timings.predicted_n            = n_decoded;
        timings.predicted_ms           = t_token_generation;
        timings.predicted_per_token_ms = t_token_generation / n_decoded;
        timings.predicted_per_second   = 1e3 / t_token_generation * n_decoded;

        timings.decode_elapsed_ms = relay_decode_elapsed_ms;
        timings.decode_intervals = relay_decode_intervals;
        timings.decode_tok_s = relay_decode_tok_s;
        timings.decode_tok_s_early = relay_decode_tok_s_early;
        timings.decode_tok_s_middle = relay_decode_tok_s_middle;
        timings.decode_tok_s_late = relay_decode_tok_s_late;

        // Add speculative metrics
        if (n_draft_total > 0) {
            timings.draft_n          = n_draft_total;
            timings.draft_n_accepted = n_draft_accepted;
        }

        return timings;
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings() const {
        const double t_prompt        =       t_prompt_processing / n_prompt_tokens_processed;
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        const double t_gen        =       t_token_generation / n_decoded;
        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        SLT_INF(*this,
                "\n"
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_processing, n_prompt_tokens_processed, t_prompt, n_prompt_second,
                t_token_generation, n_decoded, t_gen, n_gen_second,
                t_prompt_processing + t_token_generation, n_prompt_tokens_processed + n_decoded);

        if (n_draft_total > 0) {
            const float draft_ratio = (float) n_draft_accepted / n_draft_total;
            SLT_CNT(*this,
                    "draft acceptance rate = %0.5f (%5d accepted / %5d generated)\n",
                    draft_ratio, n_draft_accepted, n_draft_total
            );
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = {
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining},
                    {"n_decoded",      n_decoded},
                }
            };

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx, true);
                res["generated"] = generated_text;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        llama_memory_seq_rm(llama_get_memory(ctx), other.id,     -1, -1);
        llama_memory_seq_cp(llama_get_memory(ctx), id, other.id, -1, -1);

        other.n_decoded   = n_decoded;
        other.n_remaining = n_remaining;
        other.i_batch     = i_batch;

        other.t_start_process_prompt    = t_start_process_prompt;
        other.t_prompt_processing       = t_prompt_processing;
        other.n_prompt_tokens_cache     = n_prompt_tokens_cache;
        other.n_prompt_tokens_processed = n_prompt_tokens_processed;

        other.prompt = prompt.clone();
        other.init_sampler();
    }
};



//
// server_metrics
//

struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    void init() {
        t_start = ggml_time_us();
    }

    void on_prompt_eval(const server_slot & slot) {
        n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
        n_prompt_tokens_processed       += slot.n_prompt_tokens_processed;
        t_prompt_processing             += slot.t_prompt_processing;
        t_prompt_processing_total       += slot.t_prompt_processing;

        n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
    }

    void on_prediction(const server_slot & slot) {
        n_tokens_predicted_total   += slot.n_decoded;
        n_tokens_predicted         += slot.n_decoded;
        t_tokens_generation        += slot.t_token_generation;
        t_tokens_generation_total  += slot.t_token_generation;
    }

    void on_decoded(const std::vector<server_slot> & slots) {
        n_decode_total++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                n_busy_slots_total++;
            }
            n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
    }
};


//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model = nullptr;
    mtmd_context * mctx = nullptr;
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

    // Phase 327 (2026-05-05): public accessor for /v1/seq_cp endpoint.
    // Used from server_routes::init_routes lambda. Caller must ensure
    // the context is alive (not in sleep state) before dereferencing.
    llama_context * get_ctx_unsafe() const { return ctx; }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    // Token relay buffer: tokens from Mac are buffered here during handoff,
    // then sent to HTTP client after handoff completes (can't send during
    // llama_decode — conflicts with batch result queue).
    struct relay_token_t {
        int32_t token_id;
        std::string text;
        bool done;
    };
    std::vector<relay_token_t> relay_token_buffer;
    server_slot * handoff_slot = nullptr;  // active slot during decode handoff (for streaming SSE)
    std::atomic<bool> handoff_cancelled{false};

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx = nullptr;

    llama_batch batch {};

    llama_model_ptr model_dft;

    bool add_bos_token  = true;

    int32_t n_ctx; // total context for all clients / slots

    // slots / clients
    std::vector<server_slot> slots;
    bool route_reuse_enabled = false;
    bool mac_retain_handle_enabled = false;
    bool mac_splice_enabled = false;

    int slots_debug = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

    json json_webui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API

    bool sleeping = false;

    void destroy() {
        llama_init.reset();
        ctx = nullptr;
        model = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;

        // Clear any sampling context
        for (server_slot & slot : slots) {
            common_speculative_free(slot.spec);
            slot.spec = nullptr;
        }

        llama_batch_free(batch);
    }

    bool route_reuse_record_complete(const server_slot & slot) const {
        if (!route_reuse_enabled || !ctx) {
            return false;
        }

        const auto * ctx_i = static_cast<const llama_context *>(ctx);
        const auto & record = slot.route_reuse_record;
        if (!record.valid || !record.physical_state_valid ||
                record.slot_generation != slot.route_reuse_generation ||
                record.prefix_count == 0 || record.prefix_tokens.size() != record.prefix_count ||
                slot.prompt.tokens.size() != record.prefix_count ||
                record.model_layout_fingerprint != ctx_i->compute_kv_store_model_fp() ||
                record.blob_generation != ctx_i->route_reuse_blob_generation() ||
                !ctx_i->route_reuse_candidate_ready()) {
            return false;
        }

        const uint64_t hash = pd_hash_tokens(
                reinterpret_cast<const int32_t *>(record.prefix_tokens.data()),
                (int32_t) record.prefix_tokens.size());
        if (hash != record.prefix_hash) {
            return false;
        }
        for (size_t i = 0; i < record.prefix_count; ++i) {
            if (slot.prompt.tokens[(int) i] != record.prefix_tokens[i]) {
                return false;
            }
        }
        return true;
    }

    void invalidate_route_reuse(server_slot * slot, const char * reason, bool clear_prompt) {
        // Environment registry (2026-07-28):
        //   LLAMA_SERVER_CHUNK_ROUTE_FIX=0
        //       Restore legacy invalidation order and task_n-batch_n reuse
        //       accounting for A/B. The default-on path makes an intentional
        //       prompt clear authoritative and separates cached-prefix reuse
        //       from an intra-request >n_batch continuation.
        //   LLAMA_SERVER_ROUTE_REUSE_PROBE=1
        //       Log the slot-token length, exact LCP, effective LCP, batch
        //       position, and live slot-memory position at route boundaries.
        static const bool chunk_route_fix = []() {
            const char * e = std::getenv("LLAMA_SERVER_CHUNK_ROUTE_FIX");
            return !(e && e[0] == '0');
        }();

        // prompt_clear() preserves route_reuse_candidate_ready(). Invalidate
        // that candidate first when this caller explicitly requested a clear;
        // otherwise prompt_clear() returns without clearing slot tokens/KV.
        bool ctx_invalidated_first = false;
        if (chunk_route_fix && clear_prompt && ctx && route_reuse_enabled) {
            auto * ctx_i = static_cast<llama_context *>(ctx);
            ctx_i->invalidate_route_reuse_state();
            ctx_i->clear_kv_ram_state();
            if (mac_retain_handle_enabled) {
                ctx_i->clear_mac_retained_handle();
            }
            ctx_invalidated_first = true;
        }

        if (route_reuse_enabled) {
            SRV_WRN("DSV4 route-reuse invalidated: %s\n", reason ? reason : "unspecified failure");
        }

        if (slot) {
            slot->route_reuse_record.clear();
            if (clear_prompt) {
                slot->prompt_clear(false);
            } else if (++slot->route_reuse_generation == 0) {
                ++slot->route_reuse_generation;
            }
        }

        if (!ctx_invalidated_first && ctx && route_reuse_enabled) {
            auto * ctx_i = static_cast<llama_context *>(ctx);
            ctx_i->invalidate_route_reuse_state();
            ctx_i->clear_kv_ram_state();
            if (mac_retain_handle_enabled) {
                ctx_i->clear_mac_retained_handle();
            }
        }
    }

    bool commit_route_reuse_record(server_slot & slot) {
        if (!route_reuse_enabled || !ctx) {
            return false;
        }

        auto * ctx_i = static_cast<llama_context *>(ctx);
        const auto & prefix = ctx_i->get_prefill_kv_state_cache_prefix_tokens();
        const uint64_t fingerprint = ctx_i->compute_kv_store_model_fp();
        const uint64_t blob_generation = ctx_i->route_reuse_blob_generation();
        fprintf(stderr, "*** REUSE_COMMIT_DIAG: ctx=%p candidate_ready=%d fingerprint=%llu blob_generation=%llu "
                "prefix_size=%zu slot_tokens=%zu\n",
                (const void *) ctx_i,
                (int) ctx_i->route_reuse_candidate_ready(),
                (unsigned long long) fingerprint, (unsigned long long) blob_generation,
                prefix.size(), slot.prompt.tokens.size());
        fflush(stderr);
        if (!ctx_i->route_reuse_candidate_ready() || fingerprint == 0 ||
                blob_generation == 0 || prefix.empty() || prefix.size() > slot.prompt.tokens.size()) {
            return false;
        }

        for (size_t i = 0; i < prefix.size(); ++i) {
            if (slot.prompt.tokens[(int) i] != prefix[i]) {
                return false;
            }
        }

        route_reuse_retention_record record;
        record.model_layout_fingerprint = fingerprint;
        record.slot_generation = slot.route_reuse_generation;
        record.prefix_tokens = prefix;
        record.prefix_count = prefix.size();
        record.prefix_hash = pd_hash_tokens(
                reinterpret_cast<const int32_t *>(record.prefix_tokens.data()),
                (int32_t) record.prefix_tokens.size());
        record.blob_generation = blob_generation;
        record.physical_state_valid = true;
        record.valid = true;

        // The logical slot state is exactly the physical blob coverage. Mac-only
        // tail/decode tokens must never inflate the next LCP claim.
        slot.prompt.tokens.clear();
        slot.prompt.tokens.insert(record.prefix_tokens);
        slot.prompt.data.clear();
        slot.prompt.checkpoints.clear();
        slot.route_reuse_record = std::move(record);

        SLT_INF(slot,
                "route-reuse: committed complete retention record (prefix=%zu hash=0x%" PRIx64
                " blob_gen=%" PRIu64 " slot_gen=%" PRIu64 ")\n",
                slot.route_reuse_record.prefix_count,
                slot.route_reuse_record.prefix_hash,
                slot.route_reuse_record.blob_generation,
                slot.route_reuse_record.slot_generation);
        return true;
    }

    void record_mac_retained_handle(server_slot & slot, bool relay_succeeded) {
        if (!mac_retain_handle_enabled || !ctx) {
            return;
        }

        auto * ctx_i = static_cast<llama_context *>(ctx);
        const uint32_t count = relay_succeeded
            ? ctx_i->get_last_relay_retained_token_count() : 0;
        const uint64_t hash = relay_succeeded
            ? (ctx_i->get_last_relay_retained_prefix_hash() & 0x7fffffffffffffffULL) : 0;
        if (!relay_succeeded || count == 0 || slot.task == nullptr) {
            ctx_i->clear_mac_retained_handle();
            if (relay_succeeded) {
                SRV_WRN("%s", "Mac-retain relay returned no usable handle; cleared stale handle [MAC_RETAIN_HANDLE]\n");
            }
            return;
        }

        const int prompt_n = (int) slot.task->n_tokens();
        std::vector<llama_token> prefix;
        prefix.reserve((size_t) std::min<int>((int) count, prompt_n));
        for (int i = 0; i < prompt_n && (uint32_t) i < count; ++i) {
            const llama_token token = slot.task->tokens[i];
            if (token == LLAMA_TOKEN_NULL) {
                ctx_i->clear_mac_retained_handle();
                SRV_WRN("%s", "Mac-retain handle contained a media prefix; cleared handle [MAC_RETAIN_HANDLE]\n");
                return;
            }
            prefix.push_back(token);
        }

        const bool covers = prefix.size() == count;
        uint64_t local_hash = 0;
        if (covers) {
            std::vector<int32_t> prefix32(prefix.begin(), prefix.end());
            local_hash = mac_fnv1a_tokens(prefix32.data(), (int32_t) prefix32.size());
        }
        ctx_i->set_mac_retained_handle(count, hash, prefix.data(), prefix.size());
        SRV_INF("Mac-retain handle recorded (count=%u hash=0x%" PRIx64
                " prompt_lead=%zu covers=%d hash_ok=%d) [MAC_RETAIN_HANDLE]\n",
                count, hash, prefix.size(), covers ? 1 : 0,
                covers && local_hash == hash ? 1 : 0);
    }

    void commit_mac_splice_state(server_slot & slot, bool relay_succeeded) {
        // Gate-off no-op: no state inspection or mutation unless the effective
        // default-off gate is enabled.
        if (!mac_splice_enabled || !ctx || !slot.task ||
                slot.task->splice_route_kind == server_task::SPLICE_ROUTE_NONE) {
            return;
        }

        auto & state = slot.mac_splice;
        auto * ctx_i = static_cast<llama_context *>(ctx);
        llama_tokens candidate = state.logical_prompt_sent;
        std::string assistant_text;
        size_t generated_n = 0;
        for (const auto & rt : relay_token_buffer) {
            if (!rt.done) {
                candidate.push_back((llama_token) rt.token_id);
                assistant_text += rt.text;
                ++generated_n;
            }
        }
        std::vector<int32_t> candidate32(candidate.begin(), candidate.end());
        const uint32_t returned_count = relay_succeeded
                ? ctx_i->get_last_relay_retained_token_count() : 0;
        const uint64_t returned_hash = relay_succeeded
                ? (ctx_i->get_last_relay_retained_prefix_hash() & 0x7fffffffffffffffULL) : 0;
        const uint64_t candidate_hash = candidate32.empty()
                ? 0 : mac_fnv1a_tokens(candidate32.data(), (int32_t) candidate32.size());
        const bool witness_ok = relay_succeeded && !candidate.empty() && generated_n > 0 &&
                candidate.size() == returned_count && candidate_hash == returned_hash;

        SRV_INF("splice witness: generated_n=%zu candidate_n=%zu candidate_hash=0x%" PRIx64
                " returned_n=%u returned_hash=0x%" PRIx64 " result=%s\n",
                generated_n, candidate.size(), candidate_hash, returned_count, returned_hash,
                witness_ok ? "commit" : "clear");
        if (!witness_ok) {
            state.clear();
            return;
        }

        state.valid = true;
        ++state.generation;
        state.canonical_tokens = std::move(candidate);
        state.retained_count = returned_count;
        state.retained_hash = returned_hash;
        state.assistant_text = assistant_text;
        state.expected_message_prefix = slot.task->splice_request_lineage;
        splice_append_lineage_message(state.expected_message_prefix, "assistant", assistant_text);
        state.model_fingerprint = slot.task->params.oaicompat_model;
        state.template_fingerprint = slot.task->splice_template_fingerprint;
        state.render_fingerprint = slot.task->splice_template_family;
        state.logical_prompt_sent.clear();
    }

    bool should_cancel_active_handoff(const char * stage) {
        if (handoff_slot == nullptr || handoff_slot->task == nullptr) {
            if (!handoff_cancelled) {
                fprintf(stderr, "*** RELAY_CB: cancelling relay (%s): no active handoff slot\n", stage);
                fflush(stderr);
            }
            handoff_cancelled = true;
            return true;
        }

        if (!queue_results.is_waiting_task_id(handoff_slot->task->id)) {
            if (!handoff_cancelled) {
                fprintf(stderr, "*** RELAY_CB: cancelling relay (%s): task %d is no longer waiting\n",
                        stage, handoff_slot->task->id);
                fflush(stderr);
            }
            handoff_cancelled = true;
            return true;
        }

        return false;
    }

    void cleanup_cancelled_handoff(bool clear_prompt_cache, bool clear_local_seq_kv) {
        if (handoff_slot == nullptr) {
            relay_token_buffer.clear();
            return;
        }

        fprintf(stderr, "*** SERVER: handoff cancelled for slot %d\n", handoff_slot->id);
        fflush(stderr);

        // A cancelled relay cannot publish an exact-chain witness. Gate-off is
        // unchanged because the state is never populated when splice is disabled.
        if (mac_splice_enabled) {
            handoff_slot->mac_splice.clear();
        }

        // Narrow cleanup to the cancelled slot only. Releasing all processing
        // slots was too broad — under n_parallel > 1 it could tear down
        // unrelated in-flight work.
        if (handoff_slot->is_processing()) {
            metrics.on_prediction(*handoff_slot);
            handoff_slot->release();
        }
        if (route_reuse_enabled) {
            invalidate_route_reuse(handoff_slot, "handoff cancelled", true);
        } else if (clear_prompt_cache) {
            handoff_slot->prompt_clear(false);
        }

        if (clear_local_seq_kv && llama_get_memory(ctx)) {
            llama_memory_seq_rm(llama_get_memory(ctx), handoff_slot->id, -1, -1);
        }

        // Clear KV-RAM staging state so the next request rebuilds from scratch
        // rather than reusing stale per-layer pointers from the cancelled
        // handoff. This is still global scope because KV-RAM staging is
        // context-wide, but it's safe when only one slot is active at a time —
        // which is the current deployment (Mac decode receiver is single-
        // session). Revisit if n_parallel > 1 decode is ever added.
        bool any_other_slot_active = false;
        for (auto & slot : slots) {
            if (&slot != handoff_slot && slot.is_processing()) {
                any_other_slot_active = true;
                break;
            }
        }
        if (!any_other_slot_active) {
            auto * ctx_internal = static_cast<llama_context *>(ctx);
            ctx_internal->clear_kv_ram_state();
        } else {
            fprintf(stderr,
                "*** SERVER: cancel cleanup preserving KV-RAM state — other slots "
                "still processing; per-slot KV-RAM scoping not yet implemented\n");
            fflush(stderr);
        }

        relay_token_buffer.clear();
        handoff_slot = nullptr;
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(const common_params & params) {
        bool is_resume = sleeping;

        SRV_INF("loading model '%s'\n", params.model.path.c_str());
        fprintf(stderr, "*** SRV: load_model entered\n");

        params_base = params;

        llama_init = common_init_from_params(params_base);
        fprintf(stderr, "*** SRV: common_init_from_params done\n");

        model = llama_init->model();
        ctx   = llama_init->context();
        fprintf(stderr, "*** SRV: model=%p ctx=%p\n", (void*)model, (void*)ctx);

        // Set up token relay callback for streaming prefill → Mac decode → HTTP SSE.
        // When handoff_slot is set and the request has stream=true, tokens are sent
        // as SSE partial responses in real-time. Otherwise they're buffered and sent
        // as a single final response after handoff completes.
        if (ctx) {
            auto * ctx_internal = (llama_context *)ctx;
            ctx_internal->set_token_relay_callback(
                [this](int32_t token_id, const std::string & text, bool done) -> bool {
                    if (should_cancel_active_handoff(done ? "done" : "before-buffer")) {
                        return false;
                    }

                    relay_token_buffer.push_back({token_id, text, done});

                    // Stream tokens to HTTP client in real-time if streaming is enabled
                    if (handoff_slot && handoff_slot->task && handoff_slot->task->params.stream && !done) {
                        handoff_slot->n_decoded++;
                        completion_token_output tkn;
                        tkn.tok = token_id;
                        tkn.text_to_send = text;
                        send_partial_response(*handoff_slot, tkn, false);

                        if (should_cancel_active_handoff("after-send")) {
                            return false;
                        }
                    } else if (!done && handoff_slot) {
                        // Non-streaming: just buffer (already done above)
                        static int relay_log_count = 0;
                        if (relay_log_count++ < 3) {
                            fprintf(stderr, "*** RELAY_CB: token %d buffered (stream=%d slot=%p)\n",
                                    token_id, handoff_slot->task ? (int)handoff_slot->task->params.stream : -1,
                                    (void*)handoff_slot);
                            fflush(stderr);
                        }
                    }
                    return true;
                });
        }

        if (model == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        fprintf(stderr, "*** SRV: getting vocab/n_ctx/bos...\n");
        vocab = llama_model_get_vocab(model);

        n_ctx = llama_n_ctx(ctx);

        add_bos_token = llama_vocab_get_add_bos(vocab);
        fprintf(stderr, "*** SRV: vocab=%p n_ctx=%d add_bos=%d\n", (void*)vocab, n_ctx, (int)add_bos_token);

        const int32_t nextn_predict_layers = llama_model_n_nextn_predict_layers(model);
        const bool has_external_draft = params_base.speculative.has_external_dft();
        const bool use_native_mtp_draft = llama_mtp_should_use_native_draft_context(
            params_base.mtp_mode, nextn_predict_layers, has_external_draft, params_base.prefill_streaming);
        const bool use_stream_local_mtp_draft = llama_mtp_should_use_stream_local_drafter(
            params_base.mtp_mode, nextn_predict_layers, has_external_draft, params_base.prefill_streaming);

        if (use_native_mtp_draft) {
            const auto & params_spec = params_base.speculative;
            auto params_dft = params_base;

            params_dft.n_parallel   = 1;
            params_dft.n_ctx        = params_spec.n_ctx == 0 ? llama_n_ctx_seq(ctx) : params_spec.n_ctx;
            params_dft.n_batch      = llama_n_ctx_seq(ctx);
            params_dft.devices      = params_spec.devices;
            params_dft.n_gpu_layers = params_spec.n_gpu_layers;
            params_dft.cache_type_k = params_spec.cache_type_k;
            params_dft.cache_type_v = params_spec.cache_type_v;

            params_dft.prefill_streaming = false;
            params_dft.prefill_decode_mode = LLAMA_PREFILL_DECODE_MODE_AUTO;
            params_dft.mtp_mode = LLAMA_MTP_MODE_DRAFT;
            params_dft.speculative.n_max = llama_mtp_clamp_max_draft(
                params_spec.n_max, nextn_predict_layers);

            if (params_spec.cpuparams.n_threads > 0) {
                params_dft.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
                params_dft.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
            }

            params_base.speculative.n_max = params_dft.speculative.n_max;
            params_base.speculative.model_dft = model;
            params_base.speculative.cparams_dft = common_context_params_to_llama(params_dft);
            params_base.speculative.native_mtp = true;

            SRV_INF("using native NextN/MTP draft context from target model (nextn=%d, max_draft=%d)\n",
                    nextn_predict_layers, params_base.speculative.n_max);
        } else if (use_stream_local_mtp_draft) {
            params_base.speculative.n_max = llama_mtp_clamp_max_draft(
                params_base.speculative.n_max, nextn_predict_layers);
            params_base.speculative.native_mtp = true;
            params_base.speculative.native_mtp_stream_local = true;

            SRV_INF("using stream-local native NextN/MTP drafter from target context (nextn=%d, max_draft=%d)\n",
                    nextn_predict_layers, params_base.speculative.n_max);
        } else if (has_external_draft) {
            SRV_INF("loading draft model '%s'\n", params_base.speculative.mparams_dft.path.c_str());

            const auto & params_spec = params_base.speculative;

            auto params_dft = params_base;

            params_dft.n_parallel   = 1;
            params_dft.n_ctx        = params_spec.n_ctx == 0 ? llama_n_ctx_seq(ctx) : params_spec.n_ctx;
            params_dft.n_batch      = llama_n_ctx_seq(ctx);
            params_dft.devices      = params_spec.devices;
            params_dft.model        = params_spec.mparams_dft;
            params_dft.n_gpu_layers = params_spec.n_gpu_layers;
            params_dft.cache_type_k = params_spec.cache_type_k;
            params_dft.cache_type_v = params_spec.cache_type_v;

            // Phase 335 (2026-05-05): drafter must NOT use the target's streaming
            // prefill / WF / KV-RAM machinery. The drafter is fully GPU-resident
            // (-ngld 99) and small (~600 MB for jukofyork Kimi-K2 0.6B drafter).
            // Without these overrides, the drafter inherits -ps from params_base
            // which causes its layer weights to be no_alloc/dummy → null-pointer
            // crash on first decode.
            params_dft.prefill_streaming = false;
            // Also clear WF/KV-RAM env-driven overrides for the drafter context.
            // The model load path checks LLAMA_NO_HOST and friends; we can't
            // unset them per-context but the prefill_streaming=false flag is
            // the gate that activates WF mode in llama_model_load_data. Confirmed
            // empirically that LLAMA_NO_HOST=1 alone doesn't break a non-streaming
            // model load.
            SRV_INF("Phase 335: drafter context will use prefill_streaming=false "
                    "(target uses %s)\n", params_base.prefill_streaming ? "true" : "false");

            if (params_spec.cpuparams.n_threads > 0) {
                params_dft.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
                params_dft.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
            }

            params_dft.tensor_buft_overrides = params_spec.tensor_buft_overrides;

            auto mparams_dft = common_model_params_to_llama(params_dft);

            model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
            if (model_dft == nullptr) {
                SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                return false;
            }

            params_base.speculative.model_dft = model_dft.get();
            params_base.speculative.cparams_dft = common_context_params_to_llama(params_dft);
        }

        std::string & mmproj_path = params_base.mmproj.path;
        if (!mmproj_path.empty()) {
            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            mtmd_context_params mparams = mtmd_context_params_default();

            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }

            if (params_base.speculative.type != COMMON_SPECULATIVE_TYPE_NONE) {
                params_base.speculative.type =  COMMON_SPECULATIVE_TYPE_NONE;
                SRV_WRN("%s\n", "speculative decoding is not supported by multimodal, it will be disabled");
            }
        }

        fprintf(stderr, "*** SRV: checking memory shift capability...\n");
        if (!llama_memory_can_shift(llama_get_memory(ctx))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        // Route reuse owns context-wide handoff/blob/live-memory state. Until
        // those objects are per-slot and target-affine, fail closed for any
        // parallel or multi-target configuration.
        {
            const char * env = std::getenv("LLAMA_DSV4_ROUTE_REUSE");
            const bool requested = env && env[0] != '\0' && env[0] != '0';
            auto * ctx_i = static_cast<llama_context *>(ctx);
            const int32_t configured_remote_sessions = ctx_i->get_prefill_decode_remote_nodes();
            const int32_t planned_remote_targets = std::max<int32_t>(
                    1, ctx_i->get_decode_handoff_plan().remote_nodes);
            const bool unsafe_parallel = params_base.n_parallel != 1;
            const bool unsafe_multi_session = configured_remote_sessions != 1;
            const bool unsafe_multi_target = planned_remote_targets != 1;

            route_reuse_enabled = requested && !unsafe_parallel &&
                    !unsafe_multi_session && !unsafe_multi_target;
            ctx_i->set_route_reuse_enabled(route_reuse_enabled);

            if (requested && !route_reuse_enabled) {
                SRV_WRN("WARNING: LLAMA_DSV4_ROUTE_REUSE requested but DISABLED fail-closed "
                        "(n_parallel=%d remote_sessions=%d remote_targets=%d); "
                        "route reuse currently requires exactly one slot and one remote target\n",
                        params_base.n_parallel, configured_remote_sessions, planned_remote_targets);
            }
            SRV_INF("LLAMA_DSV4_ROUTE_REUSE requested=%d effective=%d "
                    "(n_parallel=%d remote_sessions=%d remote_targets=%d)\n",
                    requested ? 1 : 0, route_reuse_enabled ? 1 : 0,
                    params_base.n_parallel, configured_remote_sessions, planned_remote_targets);

            const char * mac_env = std::getenv("LLAMA_DSV4_MAC_RETAIN_HANDLE");
            const bool mac_requested = mac_env && mac_env[0] != '\0' && mac_env[0] != '0';
            const char * splice_env = std::getenv("LLAMA_DSV4_SPLICE_GENERATED_IDS");
            const bool splice_requested = splice_env && std::strcmp(splice_env, "1") == 0;
            const bool incompatible = mac_requested && splice_requested;
            mac_retain_handle_enabled = mac_requested && route_reuse_enabled && !incompatible;
            mac_splice_enabled = splice_requested && route_reuse_enabled &&
                    ctx_i->retained_route_capable() && !incompatible;
            // Gate-off no-op: no request metadata is consumed and no per-slot
            // state is touched unless this effective flag is true.
            if (incompatible) {
                SRV_ERR("%s", "LLAMA_DSV4_SPLICE_GENERATED_IDS and LLAMA_DSV4_MAC_RETAIN_HANDLE "
                        "are mutually exclusive; disabling both optimized paths (full baseline)\n");
            }
            if (mac_requested) {
                SRV_INF("LLAMA_DSV4_MAC_RETAIN_HANDLE requested=%d effective=%d\n",
                        mac_requested ? 1 : 0, mac_retain_handle_enabled ? 1 : 0);
            }
            if (splice_requested) {
                SRV_INF("LLAMA_DSV4_SPLICE_GENERATED_IDS requested=%d effective=%d\n",
                        splice_requested ? 1 : 0, mac_splice_enabled ? 1 : 0);
            }
        }

        // setup slots
        fprintf(stderr, "*** SRV: initializing slots, n_parallel=%d\n", params_base.n_parallel);

        const int n_ctx_train = llama_model_n_ctx_train(model);

        int n_ctx_slot = llama_n_ctx_seq(ctx);
        fprintf(stderr, "*** SRV: n_ctx_slot=%d n_ctx_train=%d\n", n_ctx_slot, n_ctx_train);
        if (n_ctx_slot > n_ctx_train) {
            SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n", n_ctx_slot, n_ctx_train);
            n_ctx_slot = n_ctx_train;
        }

        slots.clear();

        // Skip speculative compat check when weight fetcher is active — the test
        // decode in common_speculative_is_compat() accesses layer weight tensors which
        // have sentinel data (0x1) in no_alloc mode, causing access violations.
        //
        // Phase 6.3 (2026-05-12): if a drafter is explicitly configured AND wf_active,
        // trust the user — assume spec-decode is compatible. Without this the WF path
        // entirely loses spec-decoding (Phase 6.3 was getting n_drafted=0). The test
        // decode that crashed is the only thing this protects against; skipping it
        // means a wrong drafter MAY crash on first speculation, but a right drafter
        // will work.
        bool can_spec = false;
        {
            const char * wf_env = getenv("LLAMA_PREFILL_WF");
            const bool wf_active = wf_env && wf_env[0] != '\0' && strcmp(wf_env, "0") != 0;
            if (params_base.speculative.native_mtp_stream_local) {
                can_spec = true;
                fprintf(stderr, "*** SRV: stream-local native MTP drafter configured - assuming spec-decode compatible\n");
            } else if (wf_active) {
                const bool drafter_configured = params_base.speculative.has_dft();
                if (drafter_configured) {
                    can_spec = true;
                    fprintf(stderr, "*** SRV: wf_active + drafter configured — trusting user, assuming spec-decode compatible (Phase 6.3)\n");
                } else {
                    fprintf(stderr, "*** SRV: skipping speculative compat check (wf_active, no drafter)\n");
                }
            } else if (params_base.speculative.native_mtp) {
                can_spec = true;
                fprintf(stderr, "*** SRV: native MTP drafter configured - assuming spec-decode compatible\n");
            } else {
                fprintf(stderr, "*** SRV: checking speculative compat...\n");
                can_spec = common_speculative_is_compat(ctx);
                fprintf(stderr, "*** SRV: can_spec=%d\n", (int)can_spec);
            }
        }
        if (!can_spec) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot slot;

            slot.id    = i;
            slot.ctx   = ctx;
            slot.n_ctx = n_ctx_slot;

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            // try speculative decoding
            if (can_spec) {
                if (params_base.speculative.native_mtp_stream_local) {
                    if (mctx) {
                        SRV_ERR("%s\n", "native MTP stream-local speculative decoding is not supported with multimodal");
                        return false;
                    }
                    slot.native_mtp_stream_local = true;
                    SLT_INF(slot, "%s", "native MTP stream-local speculative decoding initialized\n");
                } else {
                    slot.spec = common_speculative_init(params_base.speculative, slot.ctx);
                    if (slot.spec) {
                        if (mctx) {
                            SRV_ERR("%s\n", "speculative decoding is not supported with multimodal");
                            return false;
                        }
                        SLT_INF(slot, "%s", "speculative decoding context initialized\n");
                    } else {
                        SLT_INF(slot, "%s", "speculative decoding context not initialized\n");
                    }
                }
            }

            SLT_INF(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.reset();

            slots.push_back(std::move(slot));
            fprintf(stderr, "*** SRV: slot %d created\n", i);
        }
        fprintf(stderr, "*** SRV: all slots created\n");

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("slots debug = %d\n", slots_debug);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx);
            batch = llama_batch_init(std::max(n_batch, params_base.n_parallel), 0, 1);
        }

        int32_t effective_cache_ram_mib = params_base.cache_ram_mib;
        {
            const char * wf_env = getenv("LLAMA_PREFILL_WF");
            const bool wf_active = wf_env && wf_env[0] != '\0' && strcmp(wf_env, "0") != 0;
            const char * force_prompt_cache_env = getenv("LLAMA_SERVER_FORCE_PROMPT_CACHE");
            const bool force_prompt_cache =
                force_prompt_cache_env && force_prompt_cache_env[0] != '\0' && strcmp(force_prompt_cache_env, "0") != 0;
            const bool unsafe_prompt_cache =
                wf_active && model &&
                (llama_model_is_recurrent(model) || llama_model_is_hybrid(model));
            if (unsafe_prompt_cache && effective_cache_ram_mib != 0 && !force_prompt_cache) {
                SRV_WRN("%s", "prompt cache auto-disabled: unsafe with streaming weight fetcher on recurrent/hybrid models\n");
                SRV_WRN("%s", "use `--cache-ram 0` explicitly if you want to silence this warning\n");
                effective_cache_ram_mib = 0;
            } else if (unsafe_prompt_cache && effective_cache_ram_mib != 0 && force_prompt_cache) {
                SRV_WRN("%s", "prompt cache force-enabled by LLAMA_SERVER_FORCE_PROMPT_CACHE\n");
            }
        }

        if (effective_cache_ram_mib != 0) {
            if (effective_cache_ram_mib < 0) {
                SRV_WRN("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_WRN("prompt cache is enabled, size limit: %d MiB\n", effective_cache_ram_mib);
            }
            SRV_WRN("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(effective_cache_ram_mib, n_ctx);
        } else {
            SRV_WRN("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_WRN("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (!params_base.model_alias.empty()) {
            // user explicitly specified model name
            model_name = params_base.model_alias;
        } else if (!params_base.model.name.empty()) {
            // use model name in registry format (for models in cache)
            model_name = params_base.model.name;
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        fprintf(stderr, "*** SRV: about to call init() is_resume=%d\n", (int)is_resume);
        if (!is_resume) {
            bool ok = init();
            fprintf(stderr, "*** SRV: init() returned %d\n", (int)ok);
            return ok;
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(model != nullptr);
        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task) {
            process_single_task(std::move(task));
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        // populate webui settings
        {
            if (!params_base.webui_config_json.empty()) {
                try {
                    json_webui_settings = json::parse(params_base.webui_config_json);
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse webui config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;

            try {
                chat_templates = common_chat_templates_init(model, params_base.chat_template);

                LOG_INF("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // thinking is enabled if:
            // 1. It's not explicitly disabled (reasoning_budget == 0)
            // 2. The chat template supports it
            const bool enable_thinking = params_base.use_jinja && params_base.reasoning_budget != 0 && common_chat_templates_support_enable_thinking(chat_templates.get());
            SRV_INF("%s: chat template, thinking = %d\n", __func__, enable_thinking);

            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* media_path            */ params_base.media_path,
            };
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        if (route_reuse_enabled) {
            for (server_slot & slot : slots) {
                if (slot.is_processing()) {
                    continue;
                }
                bool task_matches_record = route_reuse_record_complete(slot) &&
                        task.tokens.size() >= slot.route_reuse_record.prefix_count;
                if (task_matches_record) {
                    for (size_t i = 0; i < slot.route_reuse_record.prefix_count; ++i) {
                        if (task.tokens[(int) i] != slot.route_reuse_record.prefix_tokens[i]) {
                            task_matches_record = false;
                            break;
                        }
                    }
                }
                if ((!slot.prompt.tokens.empty() || slot.route_reuse_record.valid) &&
                        !task_matches_record) {
                    invalidate_route_reuse(&slot, "incomplete or stale retention record at reuse claim", true);
                }
            }
        }

        // find the slot that has at least n% prompt similarity
        if (ret == nullptr && slot_prompt_similarity != 0.0f) {
            float sim_best = 0;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const float sim_cur = float(tokens.get_common_prefix(task.tokens)) / task.tokens.size();

                // select the current slot if the criteria match
                if (sim_cur > sim_best && sim_cur > slot_prompt_similarity) {
                    sim_best = sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                SLT_INF(*ret, "selected slot by LCP similarity, sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                        sim_best, slot_prompt_similarity, f_keep);

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // select the current slot if the criteria match
                if (!ret || slot.t_last_used <= t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 "\n", t_last);

                update_cache = true;
            }
        }

        if (ret) {
            const auto & tokens = ret->prompt.tokens;

            update_cache = update_cache && prompt_cache;

            // cache prompts only for completion tasks
            update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;

            // don't update the cache if the slot's context is empty
            update_cache = update_cache && tokens.size() > 0;

            // TODO: mtmd does not support prompt cache
            update_cache = update_cache && (ret->mctx == nullptr);

            // Skip prompt cache when streaming prefill is active — the KV
            // tensors are on freed GPU buffers after prefill cleanup.
            // KV-REUSE PART B exception (LLAMA_DSV4_KV_REUSE / LLAMA_DSV4_ROUTE_REUSE):
            // DSV4 stream_local and the gated remote-Mac route end every retained
            // request with migrate_kv_cache_to_cpu(), so at save time the KV lives in a live
            // CPU-backed module — the historic freed-GPU-buffer hazard does not apply. Allow
            // the save there so the LCP match + seeding path can reuse the prefix next turn.
            {
                auto * ctx_i = static_cast<llama_context *>(ctx);
                const char * kv_env = std::getenv("LLAMA_DSV4_KV_REUSE");
                const bool kv_reuse_on =
                        (kv_env && kv_env[0] != '\0' && kv_env[0] != '0') || route_reuse_enabled;
                const bool dsv4_cpu_resident_kv = kv_reuse_on && ctx_i &&
                        ctx_i->dsv4_kv_cpu_resident();
                update_cache = update_cache && (!(ctx_i && ctx_i->is_prefill_streaming()) || dsv4_cpu_resident_kv);
            }

            if (update_cache) {
                SRV_WRN("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                ret->prompt_save(*prompt_cache);

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear(false);
                }

                prompt_cache->update();

                SRV_WRN("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear(false);

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        // LLAMA_DSV4_SPLICE_GENERATED_IDS: select a transient, complete prompt
        // only after every proof passes. Gate off skips this block entirely, so
        // task.tokens, slot state, route selection, and configuration are unchanged.
        if (mac_splice_enabled) {
            auto clear_splice = [&](const char * reason) {
                if (slot.mac_splice.valid || !slot.mac_splice.logical_prompt_sent.empty()) {
                    SLT_INF(slot, "splice canonical cleared: %s\n", reason);
                }
                slot.mac_splice.clear();
            };

            task.splice_route_kind = server_task::SPLICE_ROUTE_NONE;
            task.splice_prompt_override.clear();
            const bool request_shape = task.splice_shape_eligible && task.params.stream &&
                    task.params.n_cmpl == 1 && !task.is_child() && task.params.lora.empty() &&
                    !task.tokens.has_media() && ctx &&
                    static_cast<llama_context *>(ctx)->retained_route_capable();
            bool token_clean = request_shape;
            for (int i = 0; token_clean && i < task.n_tokens(); ++i) {
                token_clean = task.tokens[i] != LLAMA_TOKEN_NULL;
            }

            if (request_shape && token_clean && task.splice_seed_eligible) {
                clear_splice("eligible new seed replaces prior singleton lineage");
                if ((size_t) task.n_tokens() * sizeof(int32_t) + 4096u < 1024u * 1024u &&
                        task.n_tokens() < slot.n_ctx) {
                    task.splice_route_kind = server_task::SPLICE_ROUTE_FULL_SEED;
                }
            } else if (request_shape && token_clean && !task.splice_tail_tokens.empty()) {
                const auto & state = slot.mac_splice;
                std::vector<int32_t> canonical32(state.canonical_tokens.begin(), state.canonical_tokens.end());
                const bool local_witness = state.valid && state.retained_count == state.canonical_tokens.size() &&
                        !canonical32.empty() && mac_fnv1a_tokens(canonical32.data(), (int32_t) canonical32.size()) == state.retained_hash;
                const bool lineage_ok = local_witness && task.splice_request_prefix == state.expected_message_prefix;
                {   // SPLICE-LINEAGE diagnostic: pinpoint the divergence when the continuation lineage fails
                    const char * _l = std::getenv("LLAMA_DSV4_SPLICE_GENERATED_IDS");
                    if (_l && _l[0] == '1' && local_witness && !lineage_ok) {
                        const std::string & a = task.splice_request_prefix;
                        const std::string & b = state.expected_message_prefix;
                        size_t d = 0; while (d < a.size() && d < b.size() && a[d] == b[d]) ++d;
                        const size_t from = d > 40 ? d - 40 : 0;
                        fprintf(stderr, "*** SPLICE-LINEAGE mismatch: req_len=%zu exp_len=%zu diverge_at=%zu req='%.80s' exp='%.80s'\n",
                                a.size(), b.size(), d, a.c_str() + from, b.c_str() + from);
                    }
                }
                const bool fingerprint_ok = local_witness &&
                        task.params.oaicompat_model == state.model_fingerprint &&
                        task.splice_template_fingerprint == state.template_fingerprint &&
                        task.splice_template_family == state.render_fingerprint;
                const size_t sent_n = state.canonical_tokens.size() + task.splice_tail_tokens.size();
                const bool size_ok = state.canonical_tokens.size() >= 64 && sent_n < (size_t) slot.n_ctx &&
                        sent_n * sizeof(int32_t) + 4096u < 1024u * 1024u;
                if (local_witness && lineage_ok && fingerprint_ok && size_ok) {
                    task.splice_prompt_override = state.canonical_tokens;
                    task.splice_prompt_override.insert(task.splice_prompt_override.end(),
                            task.splice_tail_tokens.begin(), task.splice_tail_tokens.end());
                    task.splice_route_kind = server_task::SPLICE_ROUTE_CONTINUATION;
                } else {
                    clear_splice(!local_witness ? "local witness mismatch" :
                            (!lineage_ok ? "structured lineage mismatch" :
                            (!fingerprint_ok ? "model/template fingerprint mismatch" : "shape/context/payload mismatch")));
                }
            } else {
                clear_splice("unsupported intervening request");
            }
        }

        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_INF(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.tokens.clear();
                } else {
                    SLT_INF(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            slot.smpl.reset(common_sampler_init(model, task.params.sampling));

            if (slot.smpl == nullptr) {
                // for now, the only error that may happen here is invalid grammar
                send_error(task, "Failed to parse grammar", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_logits = task.params.sampling.n_probs > 0;

            bool backend_sampling = true;

            backend_sampling &= task.params.sampling.backend_sampling;

            // TODO: speculative decoding requires multiple samples per batch - not supported yet
            backend_sampling &= !(slot.spec && task.params.speculative.n_max > 0);

            // TODO: getting post/pre sampling logits is not yet supported with backend sampling
            backend_sampling &= !need_logits;

            // TODO: tmp until backend sampling is fully implemented
            if (backend_sampling) {
                llama_set_sampler(ctx, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx, slot.id, nullptr);
            }

            SLT_INF(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
        } else {
            slot.smpl.reset();
        }

        slot.task = std::make_unique<const server_task>(std::move(task));

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated_text += token_str;

        // --reasoning-budget N: close an over-long thinking block and continue into the visible
        // answer. Two things matter here. It counts REASONING tokens, not total tokens. And it
        // closes the block with the format's own end marker rather than merely stopping, because a
        // block that is truncated without being closed leaves the client stuck in the reasoning
        // pane -- a bug this server has shipped once before.
        //
        // `reasoning_budget_closed` is the "stop doing any of this" flag: it is set when we force
        // the close, when the model closes the block itself, and when no block has appeared in the
        // first tokens. Without that last case a model that never thinks would pay a find() over
        // the whole of generated_text on every token, which is quadratic in the answer length.
        if (slot.task->params.reasoning_budget > 0 && !slot.reasoning_budget_closed) {
            const auto markers = common_chat_reasoning_markers_for(slot.task->params.chat_parser_params.format);
            if (!slot.reasoning_open) {
                if (slot.task->params.chat_parser_params.thinking_forced_open) {
                    slot.reasoning_open      = true;
                    slot.reasoning_scan_from = 0;
                } else {
                    const size_t start_pos = slot.generated_text.find(markers.start);
                    if (start_pos != std::string::npos) {
                        slot.reasoning_open      = true;
                        slot.reasoning_scan_from = start_pos + markers.start.size();
                    } else if (slot.n_decoded > 32) {
                        slot.reasoning_budget_closed = true; // this generation has no thinking block
                    }
                }
            }
            if (slot.reasoning_open && !slot.reasoning_budget_closed) {
                if (slot.generated_text.find(markers.end, slot.reasoning_scan_from) != std::string::npos) {
                    slot.reasoning_budget_closed = true; // the model closed it on its own, within budget
                } else if (++slot.n_reasoning_tokens >= slot.task->params.reasoning_budget) {
                    slot.generated_text         += markers.end;
                    slot.reasoning_budget_closed = true;
                    SLT_INF(slot, "reasoning budget of %d tokens reached; closed the thinking block with \"%s\"\n",
                            slot.task->params.reasoning_budget, markers.end.c_str());
                }
            }
        }

        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_decoded = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), slot.n_decoded, slot.n_ctx);
        }

        // check the limits
        if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(params_base)) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_decoded = %d, n_predict = %d\n", slot.n_decoded, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_decoded = %d, n_indent = %d\n", slot.n_decoded, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && (ggml_time_us() - slot.t_start_generation > 1000.0f*slot.task->params.t_max_predict_ms)) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_decoded = %d, t_max_predict_ms = %d ms\n", slot.n_decoded, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_decoded = %d, n_remaining = %d, next token: %5d '%s'\n", slot.n_decoded, slot.n_remaining, result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            // TODO: optimize this with min-p optimization
            std::vector<llama_token_data> cur = get_token_probabilities(ctx, idx);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    // if multimodal is enabled, send an error and return false
    bool check_no_mtmd(const int id_task) {
        if (mctx) {
            send_error(id_task, "This feature is not supported by multimodal", ERROR_TYPE_NOT_SUPPORTED);
            return false;
        }
        return true;
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.n_prompt_tokens_cache;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = (ggml_time_us() - slot.t_start_process_prompt) / 1000;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded           = slot.n_decoded;
        res->n_prompt_tokens     = slot.task->n_tokens();
        res->post_sampling_probs = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->timings = slot.get_timings();
        }

        queue_results.send(std::move(res));
    }

    void add_relay_decode_timings(result_timings & timings) const {
        // The handoff callback owns the authoritative receiver counters. Copy them at the final
        // response boundary as well as through server_slot::get_timings(): prompt-direct completion
        // bypasses the ordinary token-generation accounting, and an older incremental build was
        // observed returning an uninitialised native predicted_ms while silently omitting these
        // otherwise-valid counters. This final-boundary copy is additive and is a no-op for local
        // decode because the handoff context reports zero intervals there.
        const auto * ctx_ho = static_cast<const llama_context *>(ctx);
        const int32_t intervals = ctx_ho->get_last_relay_decode_intervals();
        if (intervals <= 0) {
            return;
        }
        timings.decode_elapsed_ms = ctx_ho->get_last_relay_decode_elapsed_ms();
        timings.decode_intervals = intervals;
        timings.decode_tok_s = ctx_ho->get_last_relay_decode_tok_s();
        timings.decode_tok_s_early = ctx_ho->get_last_relay_decode_tok_s_early();
        timings.decode_tok_s_middle = ctx_ho->get_last_relay_decode_tok_s_middle();
        timings.decode_tok_s_late = ctx_ho->get_last_relay_decode_tok_s_late();
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index           = slot.task->index;
        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = "";
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated_text);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->timings         = slot.get_timings();
        add_relay_decode_timings(res->timings);
        res->prompt          = slot.task->tokens.detokenize(ctx, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated           = slot.truncated;
        res->n_decoded           = slot.n_decoded;
        res->n_prompt_tokens     = slot.task->n_tokens();
        res->n_tokens_cached     = slot.prompt.n_tokens();
        res->has_new_line        = slot.has_new_line;
        res->stopping_word       = slot.stopping_word;
        res->stop                = slot.stop;
        res->post_sampling_probs = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(ctx, i);
            } else {
                embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
            if (free_slots.size() >= n_slots_needed) {
                break;
            }
        }
        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_INF("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    void process_single_task(server_task && task) {
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_slot = task.id_slot;
                    const int id_task = task.id;

                    server_slot * slot = id_slot != -1
                                            ? get_slot_by_id(id_slot)
                                            : get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    json slots_data = json::array();

                    int n_idle_slots       = 0;
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        json slot_data = slot.to_json(slots_debug == 0);

                        if (slot.is_processing()) {
                            n_processing_slots++;
                        } else {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot_data);
                    }
                    SRV_DBG("n_idle_slots = %d, n_processing_slots = %d\n", n_idle_slots, n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->slots_data          = std::move(slots_data);
                    res->n_idle_slots        = n_idle_slots;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->t_start             = metrics.t_start;

                    res->n_prompt_tokens_processed_total = metrics.n_prompt_tokens_processed_total;
                    res->t_prompt_processing_total       = metrics.t_prompt_processing_total;
                    res->n_tokens_predicted_total        = metrics.n_tokens_predicted_total;
                    res->t_tokens_generation_total       = metrics.t_tokens_generation_total;

                    res->n_tokens_max = metrics.n_tokens_max;

                    res->n_prompt_tokens_processed = metrics.n_prompt_tokens_processed;
                    res->t_prompt_processing       = metrics.t_prompt_processing;
                    res->n_tokens_predicted        = metrics.n_tokens_predicted;
                    res->t_tokens_generation       = metrics.t_tokens_generation;

                    res->n_decode_total          = metrics.n_decode_total;
                    res->n_busy_slots_total      = metrics.n_busy_slots_total;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }

                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const size_t token_count = slot->prompt.tokens.size();
                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    const llama_tokens & tokens = slot->prompt.tokens.get_text_tokens();
                    const size_t nwrite = llama_state_seq_save_file(ctx, filepath.c_str(), slot->id, tokens.data(), token_count);

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = token_count;
                    res->n_bytes  = nwrite;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    if (!check_no_mtmd(task.id)) break;
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    llama_tokens tokens;
                    tokens.resize(slot->n_ctx);
                    size_t token_count = 0;
                    size_t nread = llama_state_seq_load_file(ctx, filepath.c_str(), slot->id, tokens.data(), tokens.size(), &token_count);
                    if (nread == 0) {
                        slot->prompt.tokens.clear(); // KV may already been invalidated?
                        send_error(task, "Unable to restore slot, no available space in KV cache or invalid slot save file", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    tokens.resize(token_count);
                    slot->prompt.tokens.clear();
                    slot->prompt.tokens.insert(tokens);

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = token_count;
                    res->n_bytes  = nread;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear(false);

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_INF("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_HANDOFF_EXPORT:
                {
                    fprintf(stderr, "*** HANDOFF_EXPORT: case enter, task.id=%d, n_ids=%zu\n",
                        task.id, task.handoff_action.request_ids.size());
                    fflush(stderr);
                    // Phase G (2026-05-06): KV reads run on the main loop where
                    // they're safe.  Direct call from the request thread
                    // crashed in this fork's streaming-prefill llama-context.
                    auto res = std::make_unique<server_task_result_handoff_export>();
                    res->id = task.id;
                    // Drain any in-flight GPU work before reading KV state.
                    // state_seq_get_size_ext does NOT internally synchronize.
                    llama_synchronize(ctx);
                    fprintf(stderr, "*** HANDOFF_EXPORT: post-sync, ctx=%p\n", (void*)ctx); fflush(stderr);

                    std::vector<llama_seq_id> seq_ids;
                    std::vector<std::vector<uint8_t>> per_seq_blobs;
                    std::vector<std::pair<uint32_t, uint32_t>> per_seq_pos;
                    seq_ids.reserve(task.handoff_action.request_ids.size());
                    per_seq_blobs.reserve(task.handoff_action.request_ids.size());
                    per_seq_pos.reserve(task.handoff_action.request_ids.size());

                    bool failed = false;
                    // Use the MLA-aware llama_handoff_capture_kv (Phase G,
                    // 2026-05-06) which sets v_trans on attn caches before
                    // state_seq access — required for streaming-prefill
                    // DeepSeek-V2 / Kimi-style models.
                    for (int32_t raw_sid : task.handoff_action.request_ids) {
                        const llama_seq_id sid = static_cast<llama_seq_id>(raw_sid);
                        const llama_pos pos_max = llama_memory_seq_pos_max(
                            llama_get_memory(ctx), sid);
                        fprintf(stderr, "*** HANDOFF_EXPORT: sid=%d pos_max=%d\n", (int)sid, (int)pos_max); fflush(stderr);
                        if (pos_max < 0) {
                            res->status_code = 404;
                            res->error = std::string("seq_id ") +
                                std::to_string(sid) + " has no KV (pos_max=-1)";
                            failed = true;
                            break;
                        }
                        const size_t need = llama_handoff_capture_kv(ctx, sid, nullptr, 0);
                        fprintf(stderr, "*** HANDOFF_EXPORT: sid=%d need=%zu\n", (int)sid, need); fflush(stderr);
                        if (need == 0) {
                            res->status_code = 500;
                            res->error = std::string("handoff_capture_kv sizing failed for ") + std::to_string(sid);
                            failed = true;
                            break;
                        }
                        std::vector<uint8_t> buf(need);
                        const size_t got = llama_handoff_capture_kv(ctx, sid, buf.data(), need);
                        fprintf(stderr, "*** HANDOFF_EXPORT: sid=%d got=%zu\n", (int)sid, got); fflush(stderr);
                        if (got != need) {
                            res->status_code = 500;
                            res->error = std::string("handoff_capture_kv short read for ") + std::to_string(sid);
                            failed = true;
                            break;
                        }
                        // FEATURE (RAM-artifact mode, user 2026-05-29): when
                        // LLAMA_PREFILL_KV_ARTIFACT_PATH is set, also persist the captured
                        // streaming-KV blob to a file artifact (<path>/kv_seq<sid>.artifact).
                        // This is the SAME serialization the split path ships over TCP
                        // (llama_handoff_capture_kv, which DOES capture the KV-RAM streaming
                        // state — unlike the standard slot-save). A normal-decode process can
                        // then load this artifact and decode (prefill->artifact->normal decode).
                        // Default off => no behavior change. Validation pending (needs one
                        // streaming-prefill pass to generate the blob).
                        {
                            static const char * artifact_dir = std::getenv("LLAMA_PREFILL_KV_ARTIFACT_PATH");
                            if (artifact_dir && artifact_dir[0] != '\0') {
                                char apath[1024];
                                snprintf(apath, sizeof(apath), "%s/kv_seq%d.artifact", artifact_dir, (int) sid);
                                FILE * af = fopen(apath, "wb");
                                if (af) {
                                    const size_t wrote = fwrite(buf.data(), 1, buf.size(), af);
                                    fclose(af);
                                    fprintf(stderr, "*** KV_ARTIFACT: wrote %zu/%zu bytes for seq %d -> %s\n",
                                            wrote, buf.size(), (int) sid, apath);
                                    fflush(stderr);
                                } else {
                                    fprintf(stderr, "*** KV_ARTIFACT: FAILED to open %s for write\n", apath);
                                    fflush(stderr);
                                }
                            }
                        }
                        seq_ids.push_back(sid);
                        per_seq_blobs.emplace_back(std::move(buf));
                        per_seq_pos.emplace_back(
                            0u, static_cast<uint32_t>(pos_max + 1));
                    }

                    if (!failed) {
                        const struct llama_model * model_p = llama_get_model(ctx);
                        const int32_t n_layer    = llama_model_n_layer(model_p);
                        const int32_t n_head_kv  = llama_model_n_head_kv(model_p);
                        const int32_t n_head     = llama_model_n_head(model_p);
                        const int32_t n_embd     = llama_model_n_embd(model_p);
                        const int32_t head_dim   = (n_head > 0) ? (n_embd / n_head) : 0;
                        try {
                            res->kvblob_bytes = kvblob_v2_build_opaque(
                                KVBLOB_V2_PRODUCER_LLAMA_CPP,
                                static_cast<uint32_t>(n_layer),
                                static_cast<uint32_t>(n_head_kv),
                                static_cast<uint32_t>(head_dim),
                                KVBLOB_V2_DTYPE_F16,
                                KVBLOB_V2_DTYPE_F16,
                                /*mla_compressed=*/false,
                                seq_ids, per_seq_blobs, per_seq_pos);
                            res->n_seqs = static_cast<int32_t>(seq_ids.size());
                            res->status_code = 200;
                        } catch (const std::exception & e) {
                            res->status_code = 500;
                            res->error = std::string("kvblob_v2_build_opaque failed: ") + e.what();
                        }
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_HANDOFF_IMPORT:
                {
                    auto res = std::make_unique<server_task_result_handoff_import>();
                    res->id = task.id;

                    kvblob_v2_header h{};
                    std::vector<kvblob_v2_seq_desc> descs;
                    std::string parse_err;
                    if (!kvblob_v2_parse(task.handoff_action.handoff_blob,
                                         h, descs, parse_err)) {
                        res->status_code = 400;
                        res->error = "KVBlob v2 parse failed: " + parse_err;
                        queue_results.send(std::move(res));
                        break;
                    }

                    // Phase 316z architectural guard: pre-flight every target
                    // before writing ANY.  Conflict -> 409, never silent
                    // overwrite.
                    bool conflict = false;
                    for (const auto & d : descs) {
                        llama_seq_id target = d.seq_id;
                        auto it = task.handoff_action.handoff_remap.find(d.seq_id);
                        if (it != task.handoff_action.handoff_remap.end()) {
                            target = static_cast<llama_seq_id>(it->second);
                        }
                        if (llama_memory_seq_pos_max(llama_get_memory(ctx),
                                                     target) >= 0) {
                            res->status_code = 409;
                            res->error = std::string("seq_id ") +
                                std::to_string(target) + " already occupied";
                            conflict = true;
                            break;
                        }
                    }
                    if (conflict) {
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (!(h.producer_flags & KVBLOB_V2_FLAG_LAYOUT_OPAQUE)) {
                        res->status_code = 501;
                        res->error = "canonical KVBlob v2 layout not yet supported "
                                     "on llama.cpp consumer; producer must set "
                                     "KVBLOB_V2_FLAG_LAYOUT_OPAQUE";
                        queue_results.send(std::move(res));
                        break;
                    }

                    bool write_failed = false;
                    for (const auto & d : descs) {
                        llama_seq_id target = d.seq_id;
                        auto it = task.handoff_action.handoff_remap.find(d.seq_id);
                        if (it != task.handoff_action.handoff_remap.end()) {
                            target = static_cast<llama_seq_id>(it->second);
                        }
                        if (d.k_byte_offset + d.bytes_per_layer >
                            task.handoff_action.handoff_blob.size()) {
                            res->status_code = 400;
                            res->error = std::string("seq ") +
                                std::to_string(d.seq_id) +
                                " payload offset/size exceeds blob bounds";
                            write_failed = true;
                            break;
                        }
                        const uint8_t * data = reinterpret_cast<const uint8_t *>(
                            task.handoff_action.handoff_blob.data() +
                            d.k_byte_offset);
                        const size_t set = llama_state_seq_set_data_ext(
                            ctx, data,
                            static_cast<size_t>(d.bytes_per_layer),
                            target, 0);
                        if (set != d.bytes_per_layer) {
                            res->status_code = 500;
                            res->error = std::string("state_seq_set short write for ") +
                                std::to_string(target);
                            write_failed = true;
                            break;
                        }
                    }

                    if (!write_failed) {
                        res->status_code = 200;
                        res->n_seqs = static_cast<int32_t>(descs.size());
                    }
                    queue_results.send(std::move(res));
                } break;
        }
    }

    void update_slots() {
        bool handoff_needed = false;  // deferred handoff flag (ret==2 from streaming prefill)

        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_INF("%s", "all slots are idle\n");

                return;
            }
        }

        {
            SRV_DBG("%s", "posting NEXT_RESPONSE\n");

            server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
            task.id = queue_tasks.get_new_id();
            queue_tasks.post(std::move(task));
        }

        // apply context-shift if needed
        // TODO: simplify and improve
        for (server_slot & slot : slots) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    continue;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    continue;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                const int n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                llama_memory_seq_rm (llama_get_memory(ctx), slot.id, n_keep            , n_keep + n_discard);
                llama_memory_seq_add(llama_get_memory(ctx), slot.id, n_keep + n_discard, slot.prompt.n_tokens(), -n_discard);

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_text_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.tokens.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        }

        // start populating the batch for this iteration
        common_batch_clear(batch);

        // track if given slot can be batched with slots already in the batch
        server_slot * slot_batched = nullptr;

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        // first, add sampled tokens from any ongoing sequences
        for (auto & slot : slots) {
            if (slot.state != SLOT_STATE_GENERATING) {
                continue;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                continue;
            }

            // generate draft tokens in speculative decoding mode
            // TODO: rework to have a single draft llama_context shared across all slots [TAG_SERVER_SPEC_REWORK]
            //       perform the speculative drafting for all sequences at the same time in a single batch
            const int n_draft_max = slot.get_n_draft_max();
            if (n_draft_max > 0) {
                if (mctx) {
                    // we should never reach this, as speculative is automatically disabled if mmproj is loaded
                    GGML_ABORT("not supported by multimodal");
                }

                const auto & params_spec = slot.task->params.speculative;

                llama_tokens draft;
                std::vector<std::vector<llama_token_data>> draft_dists;
                if (slot.native_mtp_stream_local) {
                    draft = slot.draft_native_mtp_stream_local(params_spec, n_draft_max, draft_dists);
                } else {
                    const llama_tokens & cached_text_tokens = slot.prompt.tokens.get_text_tokens();
                    draft = common_speculative_draft(slot.spec, params_spec, cached_text_tokens, slot.sampled);
                    draft_dists = common_speculative_get_draft_dists(slot.spec);
                }

                if (draft.size() > (size_t) n_draft_max) {
                    SLT_WRN(slot, "draft size %d exceeds max %d, truncating\n", (int) draft.size(), n_draft_max);
                    draft.resize(n_draft_max);
                }
                if (draft_dists.size() > draft.size()) {
                    draft_dists.resize(draft.size());
                }
                if (draft_dists.size() != draft.size()) {
                    draft_dists.clear();
                }

                // add the sampled token to the batch
                slot.i_batch_dft.push_back(batch.n_tokens);
                common_batch_add(batch, slot.sampled, slot.prompt.tokens.pos_next(), { slot.id }, true);
                slot.prompt.tokens.push_back(slot.sampled);

                if (slot.task->params.speculative.n_min > (int) draft.size()) {
                    SLT_DBG(slot, "ignoring small draft: %d < %d\n", (int) draft.size(), slot.task->params.speculative.n_min);
                    // fallback to normal decoding
                    slot.i_batch = slot.i_batch_dft[0];
                    slot.drafted.clear();
                    slot.draft_dists.clear();
                    slot.i_batch_dft.clear();
                } else {
                    // keep track of total number of drafted tokens tested
                    slot.n_draft_total += draft.size();

                    // add all drafted tokens to the batch
                    for (size_t i = 0; i < draft.size(); i++) {
                        slot.i_batch_dft.push_back(batch.n_tokens);
                        common_batch_add(batch, draft[i], slot.prompt.tokens.pos_next(), { slot.id }, true);
                        slot.prompt.tokens.push_back(draft[i]);
                    }
                    slot.drafted = std::move(draft);
                    slot.draft_dists = std::move(draft_dists);
                }
            } else {
                // no speculative decoding
                slot.i_batch = batch.n_tokens;

                common_batch_add(batch, slot.sampled, slot.prompt.tokens.pos_next(), { slot.id }, true);

                slot.prompt.tokens.push_back(slot.sampled);

                SLT_DBG(slot, "slot decode token, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                        slot.n_ctx, slot.prompt.n_tokens(), slot.truncated);
            }
        }

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx);
        int32_t n_ubatch = llama_n_ubatch(ctx);

        float  alora_scale       = -1.0f;
        size_t alora_disabled_id = 0;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.n_tokens == 0) {
            // Phase 5.3 (2026-05-12): serial-prefill mode. Under streaming-prefill
            // (LLAMA_PREFILL_TB_ENABLE-style stack), the per-layer kv_ram buffer
            // is shared across slots and indexed by token position [0..N-1].
            // Concurrent multi-slot prefill writes overwrite each other at the
            // same positions — last-writer-wins, prior slot's cells lost.
            // Empirical proof: KV_SEQ_SWEEP at N=2 concurrent shows seq=0 with
            // only 1 prompt's cells; both Mac sessions import the same blob.
            //
            // Mode: only ONE slot may have prompt tokens in any single batch
            // when streaming-prefill is active. Concurrent client requests stay
            // queued and are processed across successive update_slots iterations
            // (one slot per iteration). Wall time per slot is unchanged; total
            // wall scales linearly with N (correct semantics over throughput).
            //
            // Override via --prefill-mode parallel (future phase) once the
            // streaming-prefill engine grows per-slot kv_ram offsets.
            auto * ctx_for_mode = static_cast<llama_context *>(ctx);
            const bool prefill_streaming_serial =
                ctx_for_mode && ctx_for_mode->is_prefill_streaming();
            bool prompt_slot_claimed = false;

            for (auto & slot : slots) {
                if (!slot.is_processing()) {
                    continue;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    continue;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    continue;
                }

                // Phase 5.3 serial-prefill mode: skip additional slots that
                // have prompt work pending if one slot already claimed the
                // prompt batch in this iteration.
                const bool has_prompt_work =
                    slot.state == SLOT_STATE_PROCESSING_PROMPT ||
                    slot.state == SLOT_STATE_STARTED;
                if (prefill_streaming_serial && has_prompt_work && prompt_slot_claimed) {
                    SLT_DBG(slot, "%s",
                            "serial-prefill: deferring (another slot's prompt is the current batch)\n");
                    continue;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    if (prefill_streaming_serial) {
                        prompt_slot_claimed = true;
                    }
                    const auto & input_tokens = slot.task->tokens;

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.t_start_process_prompt = ggml_time_us();
                        slot.t_start_generation = 0;

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_INF(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;
                        const size_t slot_tokens_before_match = slot.prompt.tokens.size();
                        int raw_common_prefix = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            continue;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx)) {
                            send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                continue;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                continue;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                continue;
                            }

                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);
                                raw_common_prefix = n_past;

                                // 2026-06-01 vision cache-robustness (STEP 1b): an image request MUST be
                                // fully re-processed so the VISION-PS block (~line 3270) fires, re-encodes
                                // the image, and re-stashes the handoff image embeddings. On a REPEATED
                                // identical image request, get_common_prefix's mtmd branch matches the
                                // image chunk and advances n_past PAST the image cells -> VISION-PS never
                                // fires -> no embd stash -> no-embd export -> Mac re-prefills nothing ->
                                // garbage. Force a full re-prefill when THIS request carries media. Text /
                                // multi-turn text cache reuse is untouched (guard only fires for images).
                                if (n_past > 0) {
                                    bool req_has_media = input_tokens.has_media();
                                    if (!req_has_media) {
                                        const int n_req_tok = (int) input_tokens.size();
                                        for (int t = 0; t < n_req_tok; ++t) {
                                            if (input_tokens[t] == LLAMA_TOKEN_NULL) { req_has_media = true; break; }
                                        }
                                    }
                                    if (req_has_media) {
                                        SLT_INF(slot, "vision cache-robustness: disabling prompt-cache reuse for image request (was n_past=%d) to force VISION-PS re-encode + embd stash\n", n_past);
                                        n_past = 0;
                                    }
                                }

                                // -----------------------------------------------------------
                                // PERSISTENT KV STORE — read hook (Engine A / Phases A1+A2).
                                // When THIS process has no live in-RAM KV blob and no usable
                                // in-process prompt-cache prefix (e.g. fresh after a server
                                // rebuild / restart), do a longest-prefix lookup in the disk
                                // store via the exported llama_context shim (store internals
                                // stay inside llama.dll). On a verified hit it installs the
                                // restored artifact bytes as the retained KV blob + prefix
                                // tokens; here we seed slot.prompt.tokens to exactly the matched
                                // prefix and set n_past = reuse, so the batch narrows to the
                                // suffix and the EXISTING viable-as-prefix CACHED-PREFIX HANDOFF
                                // logic (~line 3900) runs unmodified. Image requests are excluded
                                // (the v1.8 ship-embeddings path handles those).
                                {
                                    auto * ctx_kvs = static_cast<llama_context *>(ctx);
                                    bool req_has_media2 = input_tokens.has_media();
                                    if (!req_has_media2) {
                                        const int n_req_tok = (int) input_tokens.size();
                                        for (int t = 0; t < n_req_tok; ++t) {
                                            if (input_tokens[t] == LLAMA_TOKEN_NULL) { req_has_media2 = true; break; }
                                        }
                                    }
                                    // 2026-07-21 GOAL#2 GLM read-guard fix (glm-kvstore-miss.md §5): a resident
                                    // in-RAM blob only blocks the disk lookup when it is VIABLE FOR THIS request
                                    // (its prefix tokens prefix-match input_tokens) — a stale blob from a
                                    // different conversation is dropped later anyway (cell-count mismatch) and
                                    // must not permanently shadow the persistent store (the GLM interleave miss).
                                    // Arch-agnostic; also covers DSV4 configs without LLAMA_DSV4_ROUTE_REUSE.
                                    bool blob_blocks_lookup = ctx_kvs && ctx_kvs->has_kv_blob_cache();
                                    if (blob_blocks_lookup) {
                                        const auto & bp = ctx_kvs->get_prefill_kv_state_cache_prefix_tokens();
                                        bool viable = !bp.empty() && bp.size() <= (size_t) input_tokens.size();
                                        for (size_t t = 0; viable && t < bp.size(); ++t) {
                                            viable = input_tokens[(int) t] == bp[t];
                                        }
                                        blob_blocks_lookup = viable;
                                    }
                                    // NVMe KV TIER RESTORE (2026-09-10, docs/KV_TIER_DESIGN.md): the
                                    // conversation's rows live in per-conversation files on D:, written
                                    // once. Every streaming request hands its tokens over (recorded next
                                    // to the rows); when this process holds no cached prefix the tier is
                                    // asked for the longest stored prefix, the slot is seeded to it and the
                                    // pass uploads those rows from the files and prefills only the tail.
                                    // The .kva store (whole-artifact rewrites) is bypassed on a tier hit.
                                    bool tier_hit = false;
                                    if (ctx_kvs && ctx_kvs->kv_tier_enabled() && ctx_kvs->is_prefill_streaming() &&
                                        !req_has_media2 && (int) input_tokens.size() > n_past) {
                                        std::vector<llama_token> req_tokens_t;
                                        req_tokens_t.reserve(input_tokens.size());
                                        for (int t = 0; t < (int) input_tokens.size(); ++t) {
                                            const llama_token tok = input_tokens[t];
                                            if (tok == LLAMA_TOKEN_NULL) break;
                                            req_tokens_t.push_back(tok);
                                        }
                                        ctx_kvs->set_kv_tier_prompt_tokens(req_tokens_t);
                                        const uint32_t lcp_min_t = 64;
                                        if (req_tokens_t.size() > lcp_min_t) {
                                            const uint32_t reuse_t = ctx_kvs->kv_tier_restore_prefix(req_tokens_t, lcp_min_t);
                                            if (reuse_t > (uint32_t) n_past && reuse_t < req_tokens_t.size()) {
                                                std::vector<llama_token> prefix_tokens_t(req_tokens_t.begin(), req_tokens_t.begin() + reuse_t);
                                                slot.prompt.tokens.clear();
                                                slot.prompt.tokens.insert(prefix_tokens_t);
                                                n_past = (int) reuse_t;
                                                tier_hit = true;
                                                SLT_INF(slot,
                                                        "KV tier restore HIT: reuse=%u of %zu req tokens; prefix rows come from the tier files, "
                                                        "RTX prefills tail [%u..]\n",
                                                        reuse_t, req_tokens_t.size(), reuse_t);
                                            }
                                        }
                                    }
                                    // The NVMe tier supersedes the .kva store: a store-restored RAM blob would
                                    // be migrated into a CPU KV module that virtual KV never allocates (the
                                    // 2026-09-10 warm-up crash, GGML_ASSERT tensor not allocated).
                                    const bool can_store_lookup = !tier_hit &&
                                        ctx_kvs && ctx_kvs->kv_store_enabled() && !ctx_kvs->kv_tier_enabled() &&
                                        ctx_kvs->is_prefill_streaming() &&
                                        !blob_blocks_lookup && !req_has_media2 &&
                                        (int) input_tokens.size() > n_past;
                                    if (can_store_lookup) {
                                        std::vector<llama_token> req_tokens;
                                        req_tokens.reserve(input_tokens.size());
                                        for (int t = 0; t < (int) input_tokens.size(); ++t) {
                                            const llama_token tok = input_tokens[t];
                                            if (tok == LLAMA_TOKEN_NULL) break;
                                            req_tokens.push_back(tok);
                                        }
                                        // Require >= lcp_min reuse with at least one suffix token
                                        // (the store enforces a non-empty suffix internally).
                                        const uint32_t lcp_min = 64;
                                        if (req_tokens.size() > lcp_min) {
                                            std::vector<llama_token> prefix_tokens;
                                            uint32_t reuse = ctx_kvs->kv_store_install_prefix(
                                                req_tokens, lcp_min, prefix_tokens);
                                            if (reuse > (uint32_t) n_past && reuse < req_tokens.size() &&
                                                prefix_tokens.size() == reuse) {
                                                // Seed the slot's cached prefix to EXACTLY the reuse
                                                // length so n_past == slot.prompt.tokens.size()
                                                // (skips the pos_min stale-cache check) and the
                                                // batch narrows to the suffix downstream.
                                                slot.prompt.tokens.clear();
                                                slot.prompt.tokens.insert(prefix_tokens);
                                                n_past = (int) reuse;
                                                SLT_INF(slot,
                                                        "persistent KV store HIT: reuse=%u of %zu req tokens; "
                                                        "installed disk blob, Mac prefills tail [%u..]\n",
                                                        reuse, req_tokens.size(), reuse);
                                            } else if (reuse > 0) {
                                                // Installed a blob that we ultimately won't use here
                                                // (e.g. reuse <= n_past) — clear it so it can't leak
                                                // into a later unrelated export.
                                                ctx_kvs->clear_prefill_kv_state_cache();
                                            }
                                        }
                                    }
                                }

                                // 2026-05-29 LLAMA_PREFILL_CAP_NPAST_AT_BLOB (default off): for the
                                // streaming-prefill handoff, KV beyond the RTX's retained blob lives only
                                // on the Mac (decoded there; the receiver clears KV per-request). Claiming
                                // the full common-prefix as n_past makes the handoff blob non-viable at
                                // turn3+ (blob=turn1 cells, common-prefix includes turn2's Mac-only cells)
                                // -> Mac full re-prefill stall. Cap n_past at the RTX blob's matching
                                // coverage so the batch tail (Mac-only portion + new delta) is prefilled by
                                // the Mac via the existing handoff tail-prefill path (bounded < crossover).
                                static const bool cap_npast_at_blob = []() {
                                    const char * e = std::getenv("LLAMA_PREFILL_CAP_NPAST_AT_BLOB");
                                    return e && e[0] != '\0' && e[0] != '0';
                                }();
                                if (cap_npast_at_blob && n_past > 0 && ctx &&
                                    ctx->is_prefill_streaming() && ctx->has_kv_blob_cache()) {
                                    const auto & blob_prefix = ctx->get_prefill_kv_state_cache_prefix_tokens();
                                    const int blob_cells = (int) blob_prefix.size();
                                    if (blob_cells > 0 && blob_cells < n_past) {
                                        bool match = true;
                                        for (int t = 0; t < blob_cells; ++t) {
                                            if (blob_prefix[t] != input_tokens[t]) { match = false; break; }
                                        }
                                        if (match) {
                                            SLT_INF(slot, "cap n_past %d -> %d (RTX blob coverage; Mac prefills bounded tail)\n", n_past, blob_cells);
                                            n_past = blob_cells;
                                        }
                                    }
                                }

                                // 2026-07-21 STREAMED-CONTINUATION GUARD (doc 6.16d/6.16f): DSV4's compressed
                                // caches cannot resume from an arbitrary unaligned prefix — the comp state was
                                // cleared and a mid-block continuation mis-binds cells => deterministic garbled
                                // output (proven: identical-prompt warm request reused a 9-token prefix and
                                // decoded symbol soup; zero-prefix A/B clean). For STREAMING-prefill requests,
                                // generic common-prefix reuse is only safe when n_past lands exactly on the
                                // retained streamed-prefix boundary (the validated blob/route-reuse prefix,
                                // e.g. the working relay turn-2 cache_n=4096). Otherwise cap to that boundary,
                                // or force a full re-prefill — the streamed pass re-evaluates in bulk so the
                                // lost reuse is negligible. LLAMA_PREFILL_ALLOW_UNALIGNED_REUSE=1 disables.
                                {
                                    static const bool allow_unaligned = []() {
                                        const char * e = std::getenv("LLAMA_PREFILL_ALLOW_UNALIGNED_REUSE");
                                        return e && e[0] != '\0' && e[0] != '0';
                                    }();
                                    if (!allow_unaligned && n_past > 0 && ctx && ctx->is_prefill_streaming()) {
                                        int viable = 0;
                                        if (ctx->has_kv_blob_cache() || ctx->route_reuse_candidate_ready()) {
                                            const auto & bp = ctx->get_prefill_kv_state_cache_prefix_tokens();
                                            const int bc = (int) bp.size();
                                            if (bc > 0 && bc <= n_past && bc <= (int) input_tokens.size()) {
                                                bool match = true;
                                                for (int t = 0; t < bc; ++t) {
                                                    if (bp[t] != input_tokens[t]) { match = false; break; }
                                                }
                                                if (match) viable = bc;
                                            }
                                        }
                                        // NVMe KV tier (2026-09-10): a prefix whose rows the tier holds is
                                        // exactly as viable as a retained blob -- the pass uploads those rows
                                        // and continues at cell == position (reserve_prefill_prefix_cells).
                                        if (n_past != viable && ctx->kv_tier_prefix_ready((uint32_t) n_past)) {
                                            SLT_INF(slot, "streamed-continuation guard: prefix of %d tokens is held by the NVMe tier; continuing from it\n", n_past);
                                            viable = n_past;
                                        } else if (n_past != viable) {
                                            // The tier holds a shorter prefix (the Mac prefilled the rest on a
                                            // PROMPT-DIRECT turn): continue from the rows it has rather than
                                            // re-prefilling the whole prompt.
                                            const uint32_t held = ctx->kv_tier_held_rows();
                                            if (held >= 64 && (int) held < n_past && (int) held > viable) {
                                                SLT_INF(slot, "streamed-continuation guard: the NVMe tier holds the first %u of the %d reused tokens; continuing from there\n", held, n_past);
                                                viable = (int) held;
                                            }
                                        }
                                        if (n_past != viable) {
                                            SLT_INF(slot, "streamed-continuation guard: unaligned prefix reuse (n_past=%d, viable streamed prefix=%d) would mis-bind DSV4compressed state — %s\n",
                                                    n_past, viable, viable > 0 ? "capping to the retained streamed prefix" : "forcing full re-prefill");
                                            n_past = viable;
                                        }
                                    }
                                }

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_INF(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            llama_memory_seq_rm (llama_get_memory(ctx), slot.id, head_p, head_c);
                                            llama_memory_seq_add(llama_get_memory(ctx), slot.id, head_c, head_c + n_match, kv_shift);

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            // note: when n_swa == 0, the model does not use SWA, which is equivalent to a window of 1
                            const auto n_swa = std::max(1, llama_model_n_swa(model));

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, n_past - n_swa);

                            // note: disallow with mtmd contexts for now
                            //       https://github.com/ggml-org/llama.cpp/issues/17043
                            if (!mctx && n_past > 0 && n_past < slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx), slot.id);
                                // Phase 316m (2026-05-04): for streaming-prefill / WF
                                // mode, KV lives in kv_ram_layers (host-pinned RAM)
                                // not the GPU memory module, so seq_pos_min returns
                                // -1 even when the cached prefix is fully valid.
                                // Setting LLAMA_SERVER_TRUST_PROMPT_CACHE_KV_RAM=1
                                // bypasses the false-positive "stale cache" check
                                // and trusts slot.prompt.tokens. Kimi cache_prompt
                                // never produced a real speedup before this.
                                const char * trust_env = std::getenv(
                                    "LLAMA_SERVER_TRUST_PROMPT_CACHE_KV_RAM");
                                const bool trust_kv_ram = trust_env && trust_env[0]
                                    && trust_env[0] != '0';
                                if (pos_min == -1 && !trust_kv_ram) {
                                    SLT_WRN(slot,
                                            "dropping stale prompt cache: n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n",
                                            n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    slot.prompt_clear(true);
                                    n_past = 0;
                                } else if (pos_min == -1 && trust_kv_ram) {
                                    SLT_INF(slot,
                                            "Phase 316m: trusting prompt cache despite pos_min=-1 (KV-RAM mode). n_past=%d slot.prompt.tokens.size()=%d seq_id=%d\n",
                                            n_past, (int) slot.prompt.tokens.size(), slot.id);
                                } else {
                                    // when the prompt prefix does not match, print the tokens around the mismatch
                                    // this is useful for debugging prompt caching
                                    if (slots_debug) {
                                        const int np0 = std::max<int>(n_past - 4, 0);
                                        const int np1 = std::min<int>(n_past + 6, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                        std::stringstream ss0;
                                        std::stringstream ss1;

                                        std::stringstream st0;
                                        std::stringstream st1;

                                        ss0 << "old: ... ";
                                        ss1 << "new: ... ";

                                        for (int i = np0; i < np1; i++) {
                                            if (i == n_past) {
                                                ss0 << " | ";
                                                ss1 << " | ";
                                            }

                                            {
                                                const auto token = slot.prompt.tokens[i];
                                                const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx, token) : "[mtmd]";
                                                ss0 << piece;
                                                st0 << std::setw(8) << token;
                                            }

                                            {
                                                const auto token = slot.task->tokens[i];
                                                const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx, token) : "[mtmd]";
                                                ss1 << piece;
                                                st1 << std::setw(8) << token;
                                            }
                                        }

                                        SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                        SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                        SLT_WRN(slot, "%s\n", st0.str().c_str());
                                        SLT_WRN(slot, "%s\n", st1.str().c_str());
                                    }

                                    if (pos_min > pos_min_thold) {
                                        // TODO: support can be added in the future when corresponding vision models get released
                                        GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                        SLT_WRN(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d, n_swa = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min, n_swa);

                                        // search for a context checkpoint
                                        const auto it = std::find_if(
                                            slot.prompt.checkpoints.rbegin(),
                                            slot.prompt.checkpoints.rend(),
                                            [&](const auto & cur) {
                                                // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                                return cur.pos_min < pos_min_thold;
                                            }
                                        );

                                        bool do_reset = it == slot.prompt.checkpoints.rend();

                                        if (!do_reset) {
                                            // restore the context checkpoint
                                            const size_t checkpoint_size = it->data.size();
                                            const size_t n = llama_state_seq_set_data_ext(ctx, it->data.data(), checkpoint_size, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                                            if (n != checkpoint_size) {
                                                SLT_ERR(slot, "failed to restore context checkpoint (pos_min = %d, pos_max = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, (float) checkpoint_size / 1024 / 1024);
                                                do_reset = true;
                                                //printf("[DEBUG] `do_reset` was set to `true` after failing to restore a checkpoint");
                                            } else {
                                                n_past = std::min(n_past, std::max(it->pos_min + 1, it->pos_max));
                                                SLT_WRN(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, (float) checkpoint_size / 1024 / 1024);
                                            }
                                        }

                                        if (do_reset) {
                                            SLT_WRN(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                    "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                            n_past = 0;
                                        }
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_min > pos_min_thold
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_min > pos_min_thold) {
                                        SLT_WRN(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_swa = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, n_swa, (float) cur.data.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        if (const char * e = std::getenv("LLAMA_SERVER_ROUTE_REUSE_PROBE");
                                e && e[0] != '\0' && e[0] != '0') {
                            const auto seq_pos_before_rm = llama_memory_seq_pos_max(
                                    llama_get_memory(ctx), slot.id);
                            fprintf(stderr,
                                    "*** ROUTE_REUSE_PROBE: phase=start slot=%d "
                                    "slot_tokens_before=%zu task_tokens=%d raw_match=%d "
                                    "effective_match=%d seq_pos_before_rm=%lld\n",
                                    slot.id, slot_tokens_before_match,
                                    slot.task->n_tokens(), raw_common_prefix, n_past,
                                    (long long) seq_pos_before_rm);
                            fflush(stderr);
                        }

                        slot.n_prompt_tokens_cache     = n_past;
                        slot.n_prompt_tokens_processed = 0;

                        slot.prompt.tokens.keep_first(n_past);

                        // STREAMING CONTINUATION (2026-09-10). After a streamed turn's handoff the
                        // retained prefix's CELLS are gone from the cache (only the slot's tokens and the
                        // host/NVMe copy of its KV survive), so a follow-up that reuses n_past tokens had
                        // its new tokens placed at cells 0.. while their positions started at n_past.
                        // The layer-major pass lays rows out by position: the new K rows overwrote the
                        // uploaded prefix rows and the position window stayed empty (K_IDXS_DUMP
                        // first4=[0,1,2,3] on a pos-6043 continuation; KV_DL_STAGING_MAP [0,6049)+[49151]).
                        // Re-reserve cells [0, n_past) for this seq so find_slot places the batch at
                        // cell == position; their rows come from the mirror / tier upload.
                        {
                            auto * ctx_rs = static_cast<llama_context *>(ctx);
                            if (n_past > 0 && ctx_rs && ctx_rs->is_prefill_streaming()) {
                                const llama_pos pmax_rs = llama_memory_seq_pos_max(llama_get_memory(ctx), slot.id);
                                if (pmax_rs + 1 < (llama_pos) n_past) {
                                    const bool ok = ctx_rs->reserve_prefill_prefix_cells(slot.id, (uint32_t) n_past);
                                    fprintf(stderr,
                                            "*** STREAM_CONTINUATION: cache held positions up to %lld for seq %d but the reused "
                                            "prefix is %d tokens; %s [PREFIX_CELLS]\n",
                                            (long long) pmax_rs, slot.id, n_past,
                                            ok ? "reserved the prefix cells so new tokens land at cell == position"
                                               : "RESERVE FAILED (the pass will not line up)");
                                    fflush(stderr);
                                }
                            }
                        }

                        // send initial 0% progress update if needed
                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream && slot.task->params.return_progress) {
                            send_partial_response(slot, {}, true);
                        }
                    }

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.n_tokens + slot.task->n_tokens() > n_batch) {
                            continue;
                        }
                    }

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_INF(slot, "n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    if (!llama_memory_seq_rm(llama_get_memory(ctx), slot.id, p0, -1)) {
                        SLT_WRN(slot, "failed to truncate tokens with position >= %d - clearing the memory\n", p0);

                        slot.prompt_clear(true);

                        // there is no common part left
                        slot.n_prompt_tokens_cache = 0;
                    }

                    // streaming_prefill_active: hoisted here (also used below for the tail
                    // logic) so the image path can route through the -ps streaming executor.
                    auto * ctx_internal = static_cast<llama_context *>(ctx);
                    const bool streaming_prefill_active = ctx_internal && ctx_internal->is_prefill_streaming();

                    // check if we should process the image
                    if (slot.prompt.n_tokens() < slot.task->n_tokens() && input_tokens[slot.prompt.n_tokens()] == LLAMA_TOKEN_NULL) {
                        const auto & img_chunk = input_tokens.find_chunk(slot.prompt.n_tokens());
                        const size_t n_img = mtmd_input_chunk_get_n_tokens(img_chunk.get());
                        if (streaming_prefill_active) {
                            // PATH A (2026-05-31 vision): the non-streaming process_chunk ends in
                            // llama_decode on a model that is NOT resident (the -ps weight-fetcher
                            // streams 460GB). Instead: clip/MoonViT-encode the image here, then
                            // prefill the image embeddings through the STREAMING executor — build an
                            // embd-only llama_batch (token=nullptr) for the whole image span; an
                            // embd batch with n_tokens>=prefill_min_stream_batch_tokens routes via
                            // should_use_streaming_for_batch -> llama_prefill_stream -> image KV.
                            if (mtmd_encode_chunk(mctx, img_chunk.get()) != 0) {
                                SLT_ERR(slot, "failed to encode image (streaming), n_img = %zu\n", n_img);
                                send_error(slot, "failed to encode image", ERROR_TYPE_SERVER);
                                slot.release();
                                continue;
                            }
                            float * img_embd = mtmd_get_output_embd(mctx);
                            // 2026-05-31 vision diag: stats over the image embeddings to
                            // tell a broken vision tower (NaN/all-zero/huge) from plausible
                            // output (mean~0, modest std). n_embd = LM embedding dim (7168).
                            if (img_embd) {
                                const int n_embd_img = llama_model_n_embd(model);
                                const size_t n_vals = (size_t) n_img * (size_t) n_embd_img;
                                double sum = 0.0, sumsq = 0.0; float vmin = img_embd[0], vmax = img_embd[0];
                                size_t n_nan = 0, n_zero = 0;
                                for (size_t i = 0; i < n_vals; ++i) {
                                    float v = img_embd[i];
                                    if (v != v) { n_nan++; continue; }
                                    sum += v; sumsq += (double) v * v;
                                    if (v < vmin) vmin = v; if (v > vmax) vmax = v;
                                    if (v == 0.0f) n_zero++;
                                }
                                const double mean = sum / (double) n_vals;
                                const double var  = sumsq / (double) n_vals - mean * mean;
                                fprintf(stderr, "*** VISION_EMBD_STATS: n_img=%zu n_embd=%d mean=%.5f std=%.5f min=%.4f max=%.4f nan=%zu zero=%zu/%zu [VISION_ROUTE_V2]\n",
                                        n_img, n_embd_img, mean, var > 0 ? std::sqrt(var) : 0.0, vmin, vmax, n_nan, n_zero, n_vals);
                                fflush(stderr);
                                // 2026-05-31 embedding-parity probe: dump the projected image embeddings
                                // for element-wise comparison vs the mlx-vlm reference vision tower.
                                if (std::getenv("LLAMA_VISION_DUMP_EMBD")) {
                                    FILE * df = fopen("diagnostics/rtx_img_embd.bin", "wb");
                                    if (df) {
                                        int32_t hdr[2] = { (int32_t) n_img, n_embd_img };
                                        fwrite(hdr, sizeof(int32_t), 2, df);
                                        fwrite(img_embd, sizeof(float), n_vals, df);
                                        fclose(df);
                                        fprintf(stderr, "*** VISION_EMBD_DUMP: wrote diagnostics/rtx_img_embd.bin (%zu floats)\n", n_vals);
                                        fflush(stderr);
                                    }
                                }
                            }
                            const llama_pos pos0 = slot.prompt.tokens.pos_next();
                            std::vector<llama_pos>      v_pos(n_img);
                            std::vector<int32_t>        v_nseq(n_img, 1);
                            std::vector<llama_seq_id>   v_seq0(1, slot.id);
                            std::vector<llama_seq_id *> v_seqp(n_img + 1, nullptr);
                            std::vector<int8_t>         v_log(n_img, 0);
                            for (size_t i = 0; i < n_img; ++i) {
                                v_pos[i]  = pos0 + (llama_pos) i;
                                v_seqp[i] = v_seq0.data();
                            }
                            llama_batch image_batch = {
                                /*n_tokens =*/ (int32_t) n_img,
                                /*token    =*/ nullptr,
                                /*embd     =*/ img_embd,
                                /*pos      =*/ v_pos.data(),
                                /*n_seq_id =*/ v_nseq.data(),
                                /*seq_id   =*/ v_seqp.data(),
                                /*logits   =*/ v_log.data(),
                            };
                            // IMAGE-DIRECT (opt-in LLAMA_SKIP_IMAGE_PREFILL): skip the wasted image
                            // LM-prefill (the ~51s weight-stream of the 361 image cells). Both decoders
                            // re-prefill the image — the llama-receiver clears + re-prefills from the
                            // shipped embd (Metal F32), and small-img-forward (MLX) uses raw_image via
                            // generate() — so the RTX image K/V (and any sparse-cache effect on the text
                            // suffix) is UNUSED downstream. The clip embd is already captured above (so
                            // the ship-embd handoff + the wait-loop match by image cells still work).
                            // res==2 = the expected "streaming prefill complete (no logits)" success code.
                            static const bool skip_img_pf = []() { const char * e = std::getenv("LLAMA_SKIP_IMAGE_PREFILL"); return e && e[0] && e[0] != '0'; }();
                            int32_t res;
                            if (skip_img_pf) {
                                res = 2;
                                SLT_INF(slot, "*** IMAGE-DIRECT: SKIPPED image LM-prefill n_img=%zu pos0=%d (LLAMA_SKIP_IMAGE_PREFILL — saves the wasted weight-stream)\n",
                                        n_img, (int) pos0);
                            } else {
                                res = llama_decode(ctx, image_batch);
                                SLT_INF(slot, "*** VISION-PS: image encode+stream-prefill n_img=%zu pos0=%d embd0=%.5f res=%d\n",
                                        n_img, (int) pos0, img_embd ? img_embd[0] : 0.0f, res);
                            }
                            // res==2 = streaming prefill pass complete (no logits) — that is the
                            // EXPECTED success code from the -ps executor (same ret==2 the batch
                            // loop handles), NOT an error. Only res<0 / res==1 are real failures.
                            if (res != 0 && res != 2) {
                                SLT_ERR(slot, "failed to stream-prefill image, res = %d\n", res);
                                send_error(slot, "failed to prefill image", ERROR_TYPE_SERVER);
                                slot.release();
                                continue;
                            }
                            // v1.8 ship-embeddings (opt-in LLAMA_SHIP_IMAGE_EMBD): stash the
                            // image embeddings so the handoff artifact ships them; the Mac
                            // receiver re-prefills these cells with correct Metal F32 FA (the
                            // RTX SM120 NCOLS=32 MLA FA accumulates VKQ in F16 and loses the
                            // image's fine glyph detail — see GROUND_TRUTH §11).
                            if (img_embd && ctx_internal && std::getenv("LLAMA_SHIP_IMAGE_EMBD")) {
                                ctx_internal->set_prefill_handoff_image_embd(
                                    img_embd, (uint32_t) n_img,
                                    (uint32_t) llama_model_n_embd(model), (uint32_t) pos0);
                                SLT_INF(slot, "*** SHIP_EMBD: stashed %zu image cells (n_embd=%d) for handoff at pos0=%d\n",
                                        n_img, llama_model_n_embd(model), (int) pos0);
                            }
                        } else {
                            // non-streaming path: original mtmd process_chunk (encode + llama_decode)
                            const llama_pos pos0_ns = slot.prompt.tokens.pos_next();
                            size_t n_tokens_out = 0;
                            int32_t res = input_tokens.process_chunk(ctx, mctx, slot.prompt.n_tokens(), slot.prompt.tokens.pos_next(), slot.id, n_tokens_out);
                            if (res != 0) {
                                SLT_ERR(slot, "failed to process image, res = %d\n", res);
                                send_error(slot, "failed to process image", ERROR_TYPE_SERVER);
                                slot.release();
                                continue;
                            }
                            // v1.8 ship-embeddings: stash the image embeddings on the NON-streaming
                            // path too (mirrors the VISION-PS stash). When is_prefill_streaming() is
                            // false, the image takes this path; without a stash here the handoff ships
                            // a no-embd artifact the receiver can't fix → garbage (the flakiness root).
                            if (ctx_internal && std::getenv("LLAMA_SHIP_IMAGE_EMBD")) {
                                float * img_embd_ns = mtmd_get_output_embd(mctx);
                                if (img_embd_ns) {
                                    ctx_internal->set_prefill_handoff_image_embd(
                                        img_embd_ns, (uint32_t) n_img,
                                        (uint32_t) llama_model_n_embd(model), (uint32_t) pos0_ns);
                                    SLT_INF(slot, "*** SHIP_EMBD: (non-streaming) stashed %zu image cells (n_embd=%d) at pos0=%d\n",
                                            n_img, llama_model_n_embd(model), (int) pos0_ns);
                                }
                            }
                        }

                        // v1.9 small-image-forward (opt-in LLAMA_SMALL_IMG_FORWARD): stash the RAW
                        // image bytes + the bare question so the receiver runs mlx_vlm generate()
                        // locally (sidesteps the RTX F16-VKQ entirely). The MLX dispatch checks
                        // raw_image FIRST. prompt_text = the question (override at export). (USER 2026-06-01.)
                        if (ctx_internal && std::getenv("LLAMA_SMALL_IMG_FORWARD") &&
                                !input_tokens.raw_image_files.empty()) {
                            // v1.10 multi-turn: pack ALL images as b"MIMG"+[N:u32]+[size_i:u32]*N+[blobs]
                            // so the receiver sees EVERY image (was raw_image_files[0] only). question_text
                            // now carries the structured vlm_conversation JSON (full history).
                            std::vector<uint8_t> packed = {'M','I','M','G'};
                            auto pu32 = [&](uint32_t v){ packed.push_back((uint8_t)(v&0xFF)); packed.push_back((uint8_t)((v>>8)&0xFF)); packed.push_back((uint8_t)((v>>16)&0xFF)); packed.push_back((uint8_t)((v>>24)&0xFF)); };
                            pu32((uint32_t) input_tokens.raw_image_files.size());
                            for (auto & im : input_tokens.raw_image_files) pu32((uint32_t) im.size());
                            for (auto & im : input_tokens.raw_image_files) packed.insert(packed.end(), im.begin(), im.end());
                            ctx_internal->set_prefill_handoff_raw_image(packed.data(), packed.size(), input_tokens.question_text);
                            SLT_INF(slot, "*** SHIP_RAW_IMAGE: packed %zu images (%zu bytes) + conv='%.40s' (multi-turn)\n",
                                    input_tokens.raw_image_files.size(), packed.size(), input_tokens.question_text.c_str());
                        }

                        slot.n_prompt_tokens_processed += (int32_t) n_img;
                        slot.prompt.tokens.push_back(img_chunk.get()); // copy
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model uses SWA and we are not using `swa_full`
                    // - the model architecture is marked as recurrent or hybrid
                    //
                    // TODO: try to make this conditional on the context or the memory module, instead of the model type
                    do_checkpoint = do_checkpoint && (
                            llama_model_is_recurrent(model) ||
                            llama_model_is_hybrid(model) ||
                            (llama_model_n_swa(model) > 0 && !params_base.swa_full)
                            );

                    bool checkpoint_prompt_has_media = false;
                    if (slot.task) {
                        checkpoint_prompt_has_media = slot.task->tokens.has_media();
                        if (!checkpoint_prompt_has_media) {
                            for (size_t t = 0; t < slot.task->tokens.size(); ++t) {
                                if (slot.task->tokens[t] == LLAMA_TOKEN_NULL) {
                                    checkpoint_prompt_has_media = true;
                                    break;
                                }
                            }
                        }
                    }

                    // SWA partial checkpoints are a text-cache optimization. With Gemma4
                    // vision prompts, the multimodal KV span has embedding-backed image
                    // cells and the partial state-size walk can fault before first decode.
                    if (do_checkpoint && checkpoint_prompt_has_media && llama_model_n_swa(model) > 0 && !params_base.swa_full) {
                        do_checkpoint = false;
                        SLT_INF(slot, "%s", "skipping SWA context checkpoint for multimodal prompt\n");
                    }

                    // (ctx_internal / streaming_prefill_active are computed above — hoisted
                    //  before the image block so vision can route through the -ps executor)

                    // Multi-batch tail optimization: only hand off a genuinely
                    // final short suffix to the Mac. Larger remainders must stay
                    // on RTX so the receiver only ever prefills the last ragged
                    // tail that FA cannot safely cover.
                    // Guard: n_prompt_tokens_processed > 0 means THIS request has
                    // had at least one streaming prefill pass complete. Without
                    // this, the trigger fires on a fresh request whose only
                    // prior tokens are cache-reused from the previous task
                    // (slot.prompt.n_tokens() picks up system-token prefixes),
                    // causing the whole prompt to be dumped on Mac as tail.
                    const bool no_remote_tail_handoff =
                        std::getenv("LLAMA_PREFILL_NO_REMOTE_TAIL_HANDOFF") != nullptr ||
                        ctx_internal->get_prefill_decode_mode() ==
                            LLAMA_PREFILL_DECODE_MODE_STREAM_LOCAL;
                    // 2026-05-31 vision: the suffix optimization hands the remaining
                    // [n_tokens..task) span to the Mac to prefill locally. That is INVALID
                    // when the remaining span contains an image placeholder (LLAMA_TOKEN_NULL):
                    // the Mac receives only raw text tokens (the export breaks at NULL) and
                    // CANNOT vision-encode the image -> it hallucinates. Detect an unprocessed
                    // image in the remaining span and keep the full prefill on RTX so the image
                    // block (VISION-PS, above) encodes+streams it and the full image KV exports.
                    bool remaining_has_image = false;
                    for (int t = slot.prompt.n_tokens(); t < slot.task->n_tokens(); ++t) {
                        if (input_tokens[t] == LLAMA_TOKEN_NULL) { remaining_has_image = true; break; }
                    }
                    if (remaining_has_image) {
                        fprintf(stderr, "*** VISION-SUFFIX-GUARD: remaining span [%d..%d) has an image; "
                                "NOT handing off suffix to Mac — keeping full prefill on RTX [VISION_ROUTE_V2]\n",
                                slot.prompt.n_tokens(), (int) slot.task->n_tokens());
                        fflush(stderr);
                    }
                    if (!no_remote_tail_handoff && !remaining_has_image &&
                        streaming_prefill_active && ctx_internal->is_wf_prefill_done() &&
                        slot.n_prompt_tokens_processed > 0 &&  // THIS request has streamed
                        slot.prompt.n_tokens() > 0 &&
                        slot.prompt.n_tokens() < slot.task->n_tokens()) {
                        const int remaining = slot.task->n_tokens() - slot.prompt.n_tokens();
                        const int tail_threshold = std::max(1, (int) llama_n_ubatch(ctx) - 1);

                        if (remaining > 0 && remaining <= tail_threshold) {
                            std::vector<llama_token> full_prompt_for_handoff;
                            full_prompt_for_handoff.reserve(slot.task->n_tokens());
                            for (int t = 0; t < slot.task->n_tokens(); t++) {
                                llama_token tok = input_tokens[t];
                                if (tok == LLAMA_TOKEN_NULL) {
                                    // 2026-05-31 vision EXPORT FIX: image placeholder cells have no
                                    // token id in the stream. Their KV is ALREADY computed + serialized
                                    // in the exported blob (pos covers them), so DON'T break here —
                                    // breaking truncated streamed_prefix to the pre-image text count and
                                    // made the Mac re-prefill the image positions locally (it has no
                                    // pixels -> garbage). Emit the model's media-pad id so the advertised
                                    // prefix COVERS the image cells; the receiver imports them from the
                                    // blob and re-prefills only the genuine text suffix after the image.
                                    tok = 163606; // Kimi-VL <|media_pad|>
                                }
                                full_prompt_for_handoff.push_back(tok);
                            }

                            if (full_prompt_for_handoff.empty()) {
                                SLT_ERR(slot, "%s\n", "suffix handoff requested but full prompt payload was empty");
                                continue;
                            }

                            const uint32_t streamed_prefix = (uint32_t) std::min<int>(
                                slot.prompt.n_tokens(), (int) full_prompt_for_handoff.size());
                            ctx_internal->set_prefill_handoff_prompt(
                                full_prompt_for_handoff.data(), full_prompt_for_handoff.size(), std::string(), streamed_prefix);

                            fprintf(stderr,
                                "*** SERVER: multi-batch suffix optimization: %d remaining tokens "
                                "(< %d threshold); handing off streamed prefix (%u/%zu tokens) and full prompt "
                                "so Mac can verify KV and prefill only the final suffix locally\n",
                                (int) remaining, tail_threshold, streamed_prefix, full_prompt_for_handoff.size());
                            fflush(stderr);

                            // slot.prompt remains the local-KV prefix only. The
                            // Mac reconstructs the full prompt from the explicit
                            // handoff payload and locally prefills the suffix.
                            slot.state = SLOT_STATE_DONE_PROMPT;
                            slot.n_decoded = 0;
                            slot.i_batch = -1;
                            slot.init_sampler();
                            ctx_internal->arm_decode_handoff_for_cached_prefill();
                            handoff_needed = true;
                            continue;

                            // Collect FULL prompt tokens into the tail sidecar.
                            // The Mac receiver auto-skips already-imported tokens
                            // (by comparing sidecar length to imported KV size),
                            // which makes the handoff protocol robust against any
                            // position-offset mismatch between Windows n_p_eval
                            // and Mac pos_max. This is equivalent to standard
                            // llama.cpp server cache_prompt behaviour: send the
                            // whole prompt, let the receiver detect common prefix.
                            std::vector<llama_token> full_prompt;
                            full_prompt.reserve(slot.task->n_tokens());
                            for (int t = 0; t < slot.task->n_tokens(); t++) {
                                llama_token tok = input_tokens[t];
                                if (tok == LLAMA_TOKEN_NULL) break;
                                full_prompt.push_back(tok);
                            }
                            // Update slot bookkeeping with only the NEW tokens
                            // processed in this step (the tokens not already in
                            // slot.prompt from the previous streaming pass).
                            for (int t = slot.prompt.n_tokens(); t < slot.task->n_tokens(); t++) {
                                llama_token tok = input_tokens[t];
                                if (tok == LLAMA_TOKEN_NULL) break;
                                slot.prompt.tokens.push_back(tok);
                                slot.n_prompt_tokens_processed++;
                            }

                            if (!full_prompt.empty()) {
                                ctx_internal->set_prefill_tail_tokens(full_prompt.data(), full_prompt.size());
                                fprintf(stderr,
                                    "*** SERVER: multi-ubatch tail optimization: %d remaining tokens "
                                    "(< %d threshold); sending full prompt (%zu tokens) as tail sidecar — "
                                    "Mac will skip imported prefix and prefill %d new tokens\n",
                                    (int)remaining, tail_threshold, full_prompt.size(), (int)remaining);
                                fflush(stderr);

                                // Mark prompt as fully consumed so handoff proceeds
                                slot.state = SLOT_STATE_DONE_PROMPT;
                                slot.n_decoded = 0;
                                slot.i_batch = -1;  // no logits from this batch
                                slot.init_sampler();
                                // No local decode will run for this request, so
                                // arm the deferred export/handoff explicitly.
                                // Without this, trigger_handoff() sees no pending
                                // handoff and becomes a no-op, leaving the server
                                // to report a misleading "returned no tokens"
                                // failure on the next follow-up turn.
                                ctx_internal->arm_decode_handoff_for_cached_prefill();
                                handoff_needed = true;
                                continue;  // skip batching, go straight to handoff
                            }
                        }
                    }

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.n_tokens < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output
                        common_batch_add(batch,
                            cur_tok,
                            slot.prompt.tokens.pos_next(),
                            { slot.id },
                            slot.task->need_embd());
                        slot.prompt.tokens.push_back(cur_tok);

                        slot.n_prompt_tokens_processed++;

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        if (do_checkpoint && !streaming_prefill_active &&
                            slot.task->n_tokens() - slot.prompt.n_tokens() == 64) {
                            break;
                        }
                    }

                    // SLT_INF(slot, "new slot.prompt.tokens: %s\n", slot.slot.prompt.tokens.str().c_str());

                    SLT_INF(slot, "prompt processing progress, n_tokens = %d, batch.n_tokens = %d, progress = %f\n", slot.prompt.n_tokens(), batch.n_tokens, (float) slot.prompt.n_tokens() / slot.task->n_tokens());

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.n_tokens > 0);

                        // Extract logits only for the last token.
                        // Iter 35e: previously skipped for ALL streaming-prefill paths because
                        // output-head weights are sentinel data when WF is active (Kimi/MiniMax
                        // NVMe streaming with prefill_llama_v2). For NON-WF streaming prefill
                        // (GLM-5 local decode), the output head IS valid and we MUST mark logits
                        // so the sampler doesn't crash with `get_logits_ith: invalid logits id`.
                        // The KV-staging-overlap concern is only relevant when WF is active.
                        {
                            auto * ctx_internal = (llama_context *)ctx;
                            // Iter 35e: original comment claimed setting logits[N-1]=true
                            // would alias the GPU output-projection buffer with KV staging.
                            // The Sonnet investigation showed this is the ONLY case where
                            // sampler reaches `get_logits_ith` — without the marker, all
                            // local-decode-after-streaming-prefill flows crash. Allow opt-out
                            // via LLAMA_SERVER_SKIP_LOGITS_MARK=1 if KV corruption surfaces.
                            const char * skip_env = std::getenv("LLAMA_SERVER_SKIP_LOGITS_MARK");
                            const bool skip = skip_env && skip_env[0] != '\0' && skip_env[0] != '0';
                            if (!skip) {
                                batch.logits[batch.n_tokens - 1] = true;
                            }
                        }

                        slot.n_decoded = 0;
                        slot.i_batch   = batch.n_tokens - 1;

                        SLT_INF(slot, "prompt done, n_tokens = %d, batch.n_tokens = %d\n", slot.prompt.n_tokens(), batch.n_tokens);

                        slot.init_sampler();

                        const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx), slot.id);
                        const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx), slot.id);

                        // no need for empty or small checkpoints
                        do_checkpoint = do_checkpoint && (pos_min >= 0 && pos_max >= 64);

                        // no need to create checkpoints that are too close together
                        do_checkpoint = do_checkpoint && (slot.prompt.checkpoints.empty() || pos_max > slot.prompt.checkpoints.back().pos_max + 64);

                        if (do_checkpoint) {
                            while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
                                // make room for the new checkpoint, if needed
                                const auto & cur = slot.prompt.checkpoints.front();

                                SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, size = %.3f MiB)\n",
                                        cur.pos_min, cur.pos_max, (float) cur.data.size() / 1024 / 1024);

                                slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
                            }

                            const size_t checkpoint_size = llama_state_seq_get_size_ext(ctx, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                            auto & cur = slot.prompt.checkpoints.emplace_back(server_prompt_checkpoint{
                                /*.pos_min = */ pos_min,
                                /*.pos_max = */ pos_max,
                                /*.data    = */ std::vector<uint8_t>(checkpoint_size),
                            });

                            llama_state_seq_get_data_ext(ctx, cur.data.data(), checkpoint_size, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                            SLT_WRN(slot, "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, size = %.3f MiB)\n",
                                    (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min, cur.pos_max, (float) cur.data.size() / 1024 / 1024);
                        }
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }

                if (batch.n_tokens >= n_batch) {
                    break;
                }
            }
        }

        SRV_DBG("decoding batch, n_tokens = %d\n", batch.n_tokens);

        if (slot_batched) {
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx, slot_batched->task->need_embd());
        }

        if (batch.n_tokens == 0) {
            SRV_WRN("%s", "no tokens to decode\n");
        }

        int32_t i_next = 0;

        // When weight fetcher is active (streaming prefill mode), the KV export
        // path needs seq_id=0 cells. The decode receiver imports with seq_id=0.
        // Remap batch seq_ids from slot.id to 0 before the decode call, making
        // the server path identical to the CLI path. This avoids the fragile
        // seq_cp workaround in capture_kv_state_blob.
        // After the batch loop, if handoff succeeded, the slot is released anyway
        // (decode happens on Mac, not locally). If handoff didn't trigger,
        // we remap back to slot.id.
        const char * wf_env = getenv("LLAMA_PREFILL_WF");
        const bool wf_active = wf_env && wf_env[0] != '\0' && strcmp(wf_env, "0") != 0;
        // Phase 316z (2026-05-05): the seq_id->0 remap was added for the Mac
        // handoff path (KV export needs seq_id=0). For RTX-only KIMI_NP>1, this
        // remap CORRUPTS multi-slot inference: every slot's data lands at
        // seq_id=0 in the unified KV cache, plus the seq_rm at remap time
        // wipes prior slots' data. Phase 316w bisect proved KIMI_NP=1 produces
        // coherent output and KIMI_NP>1 produces gibberish — this remap is the
        // culprit. Set LLAMA_SERVER_NO_REMAP_SEQ=1 to skip the remap when
        // running RTX-only (no Mac handoff) so multi-slot KV stays separated
        // by actual seq_id.
        const char * no_remap_env = getenv("LLAMA_SERVER_NO_REMAP_SEQ");
        const bool no_remap_seq = no_remap_env && no_remap_env[0] != '\0'
            && strcmp(no_remap_env, "0") != 0;
        llama_seq_id original_slot_seq = -1;
        bool seq_remapped = false;
        if (wf_active && !no_remap_seq) {
            // Find the slot that owns this batch and remap its seq_ids to 0
            for (auto & slot : slots) {
                if (slot.is_processing() && slot.id != 0) {
                    original_slot_seq = slot.id;
                    // Clear seq 0 before remapping — may have stale data from a previous handoff
                    auto * const seq_memory = llama_get_memory(ctx);
                    llama_memory_seq_rm(seq_memory, 0, -1, -1);
                    // [2026-07-28 STREAMING_CONTINUATION_SEQ_REHYDRATE_V1, codex-v6] Chunked
                    // streaming (>n_batch prompts) runs each chunk as seq 0, then copies the
                    // cells back to slot.id and REMOVES them from seq 0 before returning to the
                    // scheduler. Chunk 2+ is relabeled slot.id -> 0 below, but without copying
                    // the retained prefix BACK to seq 0 the causal mask (llama-kv-cache.cpp
                    // seq-membership check) rejects every prior-chunk K/V/k_idx cell despite
                    // intact bytes -> current-chunk locality retained, history LOST (GPU-proven:
                    // 43k prompts degenerate + confabulated recall while <=24576 is clean).
                    // Rehydrate seq 0 from the slot before the remap. Metadata-only on unified
                    // KV. Escape hatch: LLAMA_PREFILL_CONTINUATION_SEQ_REHYDRATE=0 for A/B.
                    static const bool continuation_seq_rehydrate = []() {
                        const char * e = getenv("LLAMA_PREFILL_CONTINUATION_SEQ_REHYDRATE");
                        return !(e && e[0] == '0');
                    }();
                    const llama_pos continuation_pos0 =
                        batch.pos && batch.n_tokens > 0 ? batch.pos[0] : 0;
                    if (continuation_seq_rehydrate && continuation_pos0 > 0) {
                        const llama_pos source_pos_max =
                            llama_memory_seq_pos_max(seq_memory, slot.id);
                        llama_memory_seq_cp(seq_memory, slot.id, 0, -1, -1);
                        fprintf(stderr,
                                "*** SERVER: rehydrated streaming continuation seq %d -> 0 "
                                "(pos0=%d source_pos_max=%d) [STREAMING_CONTINUATION_SEQ_REHYDRATE_V1]\n",
                                slot.id, (int) continuation_pos0, (int) source_pos_max);
                        fflush(stderr);
                    }
                    // Remap all seq_id entries in the batch from slot.id to 0
                    for (int32_t t = 0; t < batch.n_tokens; t++) {
                        if (batch.seq_id && batch.seq_id[t]) {
                            for (int32_t s = 0; s < batch.n_seq_id[t]; s++) {
                                if (batch.seq_id[t][s] == slot.id) {
                                    batch.seq_id[t][s] = 0;
                                }
                            }
                        }
                    }
                    seq_remapped = true;
                    fprintf(stderr, "*** SERVER: remapped batch seq_id %d → 0 for streaming prefill\n",
                            original_slot_seq); fflush(stderr);
                    break;
                }
            }
        }

        bool prompt_direct_done = false;
        bool prompt_direct_route_record_committed = false;

        // Prompt-direct mode: for short prompts, send raw tokens to Mac for
        // local prefill + decode instead of RTX streaming prefill.
        // The decode-receiver detects non-KVARTIF1 data and runs local prefill.
        {
            auto * ctx_internal = (llama_context *)ctx;
            // The effective Mac-retain gate is initialized once and inherits
            // route-reuse's one-slot/one-session/one-target fail-closed rules.
            // Threshold for prompt-direct vs RTX streaming prefill.
            // Default: all prompts go through prompt-direct (Mac local prefill)
            // because the RTX streaming path has a sched_reserve compute buffer
            // overlap that corrupts KV for the server binary (CLI works fine).
            // Override via --prefill-crossover N to enable RTX streaming for
            // prompts > N tokens (once the sched_reserve issue is fixed).
            const int crossover = ctx_internal->get_prefill_crossover_tokens();
            int threshold = crossover > 0 ? crossover : 1000;
            if (crossover <= 0 &&
                ctx_internal->get_prefill_decode_mode() == LLAMA_PREFILL_DECODE_MODE_SPLIT_THUNDERBOLT) {
                // DEFAULT ONLY. An explicitly configured --prefill-crossover is now honoured exactly,
                // including values BELOW 3072 (owner, 2026-09-03: "the crossover shouldn't be limited
                // to minimum 3072", wanting to tune it in the 2k-10k range).
                //
                // The floor used to apply unconditionally, which made the knob silently INERT below
                // 3072: 23 of the 48 Mac-side presets carry crossover=1, and every one of them was
                // actually running at 3072. The preset said one thing and the server did another, and
                // the only way to see the real value was the per-request
                // "*** DEBUG PROMPT-DIRECT ... threshold=%d" line.
                //
                // The floor's own reasoning still holds for the UNSET case: in split-thunderbolt mode
                // short prompts are more reliable through prompt-direct than through the current RTX
                // streaming fallback paths, and 3072 is 2 ubatches at ub=1536 (64- and 32-aligned).
                // A previous floor of 6144 was too aggressive for interactive multi-turn chat, where
                // most follow-ups sit in the 3k-6k range and paid RTX streaming overhead for no gain.
                // So a preset that never configured a crossover keeps exactly today's behaviour; one
                // that DID configure it now gets what it asked for.
                threshold = std::max(threshold, 3072);
            }

            // Detect cache-reuse. If the current batch is smaller than the
            // full task, the server narrowed batch.n_tokens to the delta
            // because it matched a cached prefix against slot.prompt.tokens.
            // Those `n_past` cached tokens are live in Windows' KV-RAM but
            // NOT on Mac (Mac cleared its KV at prior session end).
            const bool legacy_cache_reuse_narrowed_batch =
                (slot_batched != nullptr) &&
                (slot_batched->task != nullptr) &&
                ((int) slot_batched->task->n_tokens() > batch.n_tokens);
            const int legacy_n_past_cached = legacy_cache_reuse_narrowed_batch
                ? (int) slot_batched->task->n_tokens() - batch.n_tokens
                : 0;

            static const bool chunk_route_fix = []() {
                const char * e = std::getenv("LLAMA_SERVER_CHUNK_ROUTE_FIX");
                return !(e && e[0] == '0');
            }();

            // n_prompt_tokens_cache is the exact START-time LCP. batch.pos[0]
            // is the amount already present before this particular batch,
            // including earlier chunks from the same request.
            const int batch_prefix_pos =
                batch.pos && batch.n_tokens > 0
                    ? std::max(0, (int) batch.pos[0])
                    : 0;
            const int recorded_cache_prefix =
                slot_batched
                    ? std::max(0, slot_batched->n_prompt_tokens_cache)
                    : 0;
            const int n_past_cached = chunk_route_fix
                ? std::min(recorded_cache_prefix, batch_prefix_pos)
                : legacy_n_past_cached;
            const bool cache_reuse_narrowed_batch = chunk_route_fix
                ? n_past_cached > 0
                : legacy_cache_reuse_narrowed_batch;
            const bool chunked_prompt_continuation =
                chunk_route_fix && batch_prefix_pos > n_past_cached;

            if (const char * e = std::getenv("LLAMA_SERVER_ROUTE_REUSE_PROBE");
                    e && e[0] != '\0' && e[0] != '0') {
                const auto seq_pos = slot_batched
                    ? llama_memory_seq_pos_max(
                            llama_get_memory(ctx), slot_batched->id)
                    : -1;
                fprintf(stderr,
                        "*** ROUTE_REUSE_PROBE: phase=decision slot=%d "
                        "slot_tokens=%zu task_tokens=%d batch_tokens=%d "
                        "batch_pos0=%d cache_match=%d chunk_continuation=%d "
                        "seq_pos_max=%lld\n",
                        slot_batched ? slot_batched->id : -1,
                        slot_batched ? slot_batched->prompt.tokens.size() : 0,
                        slot_batched && slot_batched->task
                            ? slot_batched->task->n_tokens() : -1,
                        batch.n_tokens, batch_prefix_pos, n_past_cached,
                        (int) chunked_prompt_continuation, (long long) seq_pos);
                fflush(stderr);
            }

            // CACHED-PREFIX HANDOFF only viable when Windows has a retained
            // KV blob AND its prefix tokens match the current request's
            // first n_past tokens. Validating the match catches the case
            // where an unrelated request's prior handoff overwrote the blob
            // (shipping stale bytes would produce @@@@ garbage on Mac).
            // streamed_prefix to ship in the CACHED-PREFIX HANDOFF: how many of
            // the full prompt's leading cells the exported KV blob actually
            // covers. Defaults to n_past_cached (exact-match turn2 case). The
            // viability-as-prefix path below narrows it to the blob's true
            // extent when the blob is a strict prefix of the reused n_past.
            int cached_prefix_export_count = n_past_cached;
            bool cached_prefix_handoff_viable = false;
            if (cache_reuse_narrowed_batch && ctx_internal->has_kv_blob_cache() && slot_batched && slot_batched->task) {
                const auto & cached_prefix = ctx_internal->get_prefill_kv_state_cache_prefix_tokens();
                // 2026-05-29 LLAMA_PREFILL_VIABILITY_AS_PREFIX (default off, RC-safe):
                // accept the retained blob as a STRICT PREFIX of the reused n_past.
                // At turn3+ the reused common-prefix (n_past_cached) spans turn1's
                // streamed cells PLUS turn2's Mac-only cells, so the retained blob
                // (turn1 only, e.g. 14336) is SMALLER than n_past_cached (e.g. 14671).
                // The old Phase 67 rule required EXACT equality and dropped the blob,
                // forcing a full re-stream. With the gate on we instead export the
                // blob (covering its cached_prefix.size() leading cells) and ship
                // streamed_prefix = cached_prefix.size(); the Mac imports those cells
                // and locally prefills the bounded tail [cached_prefix.size()..full]
                // (the same stream+tail mechanism turn1 uses). A blob LARGER than
                // n_past_cached is still genuinely stale (extra cells from a longer
                // prior prompt) and is dropped regardless of the gate.
                // Default-ON when the persistent KV store is enabled (plan §3):
                // a disk-restored blob is intentionally a STRICT PREFIX of the
                // reused n_past at turn3+, so prefix-subset acceptance is required
                // for the store to deliver its win. Explicit env still overrides.
                const bool viability_as_prefix = [&]() {
                    const char * e = std::getenv("LLAMA_PREFILL_VIABILITY_AS_PREFIX");
                    if (e && e[0] != '\0') {
                        return e[0] != '0';
                    }
                    return ctx_internal && ctx_internal->kv_store_enabled();
                }();
                const int blob_cells = (int) cached_prefix.size();
                const bool exact_match    = (blob_cells == n_past_cached);
                const bool prefix_subset  = viability_as_prefix && blob_cells > 0 && blob_cells < n_past_cached;
                if (exact_match || prefix_subset) {
                    // Validate only the first blob_cells tokens (the blob's coverage).
                    const int n_check = exact_match ? n_past_cached : blob_cells;
                    bool match = true;
                    for (int t = 0; t < n_check; ++t) {
                        if (cached_prefix[t] != slot_batched->task->tokens[t]) {
                            match = false;
                            break;
                        }
                    }
                    cached_prefix_handoff_viable = match;
                    if (match) {
                        // Ship streamed_prefix = the blob's true extent so the Mac
                        // imports exactly blob_cells cells and prefills the tail.
                        cached_prefix_export_count = blob_cells;
                        if (prefix_subset) {
                            fprintf(stderr, "*** SERVER: KV blob viable-as-prefix (blob=%d cells <= n_past=%d); export blob + Mac prefills tail [%d..]\n",
                                    blob_cells, n_past_cached, blob_cells);
                            fflush(stderr);
                        }
                    } else {
                        // Blob is for a different prompt. Drop it so streaming
                        // rebuilds cleanly.
                        ctx_internal->clear_prefill_kv_state_cache();
                        fprintf(stderr, "*** SERVER: KV blob mismatch (different prior request); dropped cache, will re-stream\n");
                        fflush(stderr);
                    }
                } else {
                    // Blob size > n_past_cached (stale extra cells from a longer
                    // prior prompt) OR (gate off) any inequality. Drop and re-stream.
                    fprintf(stderr, "*** SERVER: KV blob cell-count mismatch (blob=%zu cells, need=%d cells); dropped cache, will re-stream\n",
                            cached_prefix.size(), n_past_cached);
                    fflush(stderr);
                    ctx_internal->clear_prefill_kv_state_cache();
                }
            }

            const bool dbg_has_media =
                slot_batched && slot_batched->task && slot_batched->task->tokens.has_media();
            fprintf(stderr, "*** DEBUG PROMPT-DIRECT: streaming=%d n_tokens=%d threshold=%d token_ptr=%p cache_reuse=%d n_past=%d blob=%d viable=%d has_media=%d task_tok=%d [VISION_ROUTE_V2]\n",
                    ctx_internal->is_prefill_streaming(), batch.n_tokens, threshold, (void*)batch.token,
                    cache_reuse_narrowed_batch ? 1 : 0, n_past_cached,
                    ctx_internal->has_kv_blob_cache() ? 1 : 0,
                    cached_prefix_handoff_viable ? 1 : 0,
                    dbg_has_media ? 1 : 0,
                    (slot_batched && slot_batched->task) ? (int) slot_batched->task->tokens.size() : -1);
            fflush(stderr);

            // Route by DELTA size. Three flavors:
            //   - Fresh prompt, delta < threshold → raw-token PROMPT-DIRECT
            //     (Mac prefills the whole small prompt from scratch).
            //   - Cache-reuse + KV-RAM populated, delta < threshold →
            //     CACHED-PREFIX HANDOFF. Skip Windows prefill entirely.
            //     Re-serialize kv_ram_layers (the n_past cells from the
            //     prior streaming pass, still alive in host RAM) + full
            //     prompt payload with streamed_prefix_count = n_past.
            //     Mac imports, verifies, prefills the tiny delta (~150 tok/s).
            //   - Cache-reuse but KV-RAM empty → fall through to streaming.
            //     (kv_ram was cleared — divergent prompt. We need to rebuild.)
            // Server doesn't run its own prefill in the first two paths —
            // the delta is tiny, Mac prefills much faster than Windows'
            // weight-fetcher spin-up cost.
            // Phase 66 fix: Always allow small-delta routing to Mac when SUFFIX (delta)
            // is below threshold. Three sub-cases:
            //   - Cache reuse + viable blob → CACHED-PREFIX HANDOFF (export blob, Mac
            //     imports KV, re-prefills delta only)
            //   - Cache reuse, no usable blob → PROMPT-DIRECT (send full prompt; Mac
            //     re-prefills entire thing locally; slower than CACHED-PREFIX but
            //     much faster than RTX streaming)
            //   - No cache reuse → PROMPT-DIRECT (fresh prompt)
            // Without this fix, multi-turn conversations with PROMPT-DIRECT-only
            // history have no blob and were forced into RTX streaming on every
            // turn after the first one that triggered cache reuse.
            // 2026-05-31 vision path A: image requests MUST NOT go PROMPT-DIRECT — that
            // path skips KV capture and would drop the image-derived kv_ram (the decoder
            // can't vision-encode raw placeholder tokens -> garbage). Force them onto the
            // full streaming export+handoff (which ships kv_ram incl the image cells).
            // Robust media detection: has_media() (the map_idx_to_media accessor) MAY be
            // empty at this stage depending on how intake stored the chunk, so also scan
            // the task tokens for the image placeholder (LLAMA_TOKEN_NULL marks an mtmd
            // chunk position). Either signal forces the image off PROMPT-DIRECT.
            bool req_has_media = false;
            if (slot_batched && slot_batched->task) {
                req_has_media = slot_batched->task->tokens.has_media();
                if (!req_has_media) {
                    const int n_task_tok = (int) slot_batched->task->tokens.size();
                    for (int t = 0; t < n_task_tok; ++t) {
                        if (slot_batched->task->tokens[t] == LLAMA_TOKEN_NULL) {
                            req_has_media = true;
                            break;
                        }
                    }
                }
            }
            // IMG-PROMPT-DIRECT (opt-in LLAMA_IMG_PROMPT_DIRECT): a SMALL image request with a raw
            // image captured can skip the RTX LM-prefill ENTIRELY and ship PROMPT-DIRECT-style. We
            // still CLIP here (fast vision-tower, ~secs, NOT the 51s LM-stream) so the embd ships —
            // the export then reuses the ship-embd logic ([prefix][suffix]+embd+raw_image) and the
            // wait-loop matches via cand_img (no MLX change). Both decoders work; ~14s vs 38s.
            static const bool img_pd_env = []() { const char * e = std::getenv("LLAMA_IMG_PROMPT_DIRECT"); return e && e[0] && e[0] != '0'; }();
            const bool small_img_pd = req_has_media && img_pd_env && slot_batched && slot_batched->task &&
                                      !slot_batched->task->tokens.raw_image_files.empty();
            // NVMe KV tier continuation: the prefix rows live in the tier files, so the RTX uploads
            // them and prefills the tail whatever its size; the blob-based cached-prefix handoff has
            // nothing to ship and prompt-direct would make the Mac redo the whole prompt.
            // Delta routing (2026-09-10): a small delta whose prefix the decoder already holds resident
            // goes PROMPT-DIRECT: the Mac prefills the few new tokens against its retained cache in well
            // under a second, while the RTX's streaming pass is a full weight-stream over every layer
            // (~30 s) however small the delta (measured: a 1-token re-entry cost 41 s with the tier
            // continuation). The decoder is asked over the KV port; no answer means "continue here".
            // The Mac's local prefill slows with the context it attends over (measured 101 tok/s at
            // 6k, 22 tok/s at 60k: 137 s for a 3034-token delta the RTX streams in ~10 s), so above a
            // context limit only tiny deltas (a re-entry, a short question) go to the Mac.
            static const int pd_max_tokens = []() {
                const char * e = std::getenv("LLAMA_KV_DELTA_PD_MAX_TOKENS");
                const int v = e ? atoi(e) : 0;
                return v > 0 ? v : 64;
            }();
            static const int pd_max_ctx = []() {
                const char * e = std::getenv("LLAMA_KV_DELTA_PD_MAX_CTX");
                const int v = e ? atoi(e) : 0;
                return v > 0 ? v : 16384;
            }();
            const bool pd_size_ok = batch.n_tokens <= pd_max_tokens || n_past_cached <= pd_max_ctx;
            bool decoder_holds_prefix = false;
            if (cache_reuse_narrowed_batch && n_past_cached >= 64 && batch.n_tokens > 0 && batch.n_tokens < threshold &&
                pd_size_ok && batch.token && !req_has_media && slot_batched && slot_batched->task &&
                ctx_internal->kv_tier_prefix_ready((uint32_t) n_past_cached)) {
                std::vector<llama_token> prefix_toks;
                prefix_toks.reserve((size_t) n_past_cached);
                for (int t = 0; t < n_past_cached; ++t) {
                    const llama_token tok = slot_batched->task->tokens[t];
                    if (tok == LLAMA_TOKEN_NULL) break;
                    prefix_toks.push_back(tok);
                }
                if ((int) prefix_toks.size() == n_past_cached) {
                    const uint32_t held = ctx_internal->kv_decoder_resident_prefix(prefix_toks.data(), prefix_toks.size());
                    decoder_holds_prefix = held >= (uint32_t) n_past_cached;
                    fprintf(stderr, "*** KV_DELTA: routing: the decoder holds %u of the %d-token prefix; the %d-token delta goes %s [DELTA_ROUTE]\n",
                            held, n_past_cached, batch.n_tokens,
                            decoder_holds_prefix ? "PROMPT-DIRECT (the Mac prefills it)" : "to the RTX (tier continuation)");
                    fflush(stderr);
                }
            }
            const bool tier_continuation = !decoder_holds_prefix && cache_reuse_narrowed_batch &&
                ctx_internal->kv_tier_prefix_ready((uint32_t) std::max(0, n_past_cached));
            if (tier_continuation) {
                fprintf(stderr, "*** KV_TIER: continuation over %d prefix rows from the tier; the %d-token tail streams on the RTX [TIER_ROUTE]\n",
                        n_past_cached, batch.n_tokens);
                fflush(stderr);
            }
            const bool small_delta_path = !tier_continuation &&
                (batch.n_tokens > 0 && batch.n_tokens < threshold && batch.token && (!req_has_media || small_img_pd));
            // Gate-off no-op: both predicates are false, leaving the existing
            // threshold/CACHED-PREFIX/streaming decision byte-for-byte unchanged.
            const bool use_spliced_prompt_direct = mac_splice_enabled && slot_batched && slot_batched->task &&
                    slot_batched->task->splice_route_kind == server_task::SPLICE_ROUTE_CONTINUATION &&
                    !slot_batched->task->splice_prompt_override.empty();
            const bool use_splice_seed_prompt_direct = mac_splice_enabled && slot_batched && slot_batched->task &&
                    slot_batched->task->splice_route_kind == server_task::SPLICE_ROUTE_FULL_SEED &&
                    small_delta_path;
            // Option B: cross-turn retained-KV continuation (LLAMA_DSV4_MAC_RETAIN_HANDLE).
            // The prior turn recorded the Mac's retained-prefix handle {count, hash, prefix
            // tokens}. If that handle validates against the LEADING tokens of THIS turn's
            // incoming prompt, ship only the delta [count:] and tell the Mac to reuse its
            // first `count` retained tokens as the prefix. Validation is the hard correctness
            // guard: full-prefix FNV-1a hash match AND a direct token compare of the stored
            // prefix — we NEVER ship delta-only on a hash-only match. Any mismatch (topic
            // change / stale-or-foreign cache) drops the handle and falls back to full send.
            bool     use_mac_retained_prefix = false;
            uint32_t mac_reuse_count         = 0;
            uint64_t mac_reuse_hash          = 0;
            if (mac_retain_handle_enabled && !req_has_media &&
                slot_batched && slot_batched->task) {
                const uint32_t stored_count = ctx_internal->get_mac_retained_count();
                const auto &   stored_prefix = ctx_internal->get_mac_retained_prefix_tokens();
                const int      full_n = (int) slot_batched->task->n_tokens();
                if (stored_count > 0 && !stored_prefix.empty()) {
                    const size_t checked_n = std::min(
                        (size_t) stored_count, (size_t) std::max(full_n, 0));
                    std::vector<int32_t> lead;
                    lead.reserve(checked_n);
                    bool clean = true;
                    for (size_t t = 0; t < checked_n; ++t) {
                        const llama_token tok = slot_batched->task->tokens[t];
                        if (tok == LLAMA_TOKEN_NULL) { clean = false; break; }
                        lead.push_back((int32_t) tok);
                    }
                    if (clean) {
                        // (1) full-prefix FNV-1a hash (validates the Mac's generated tail too,
                        //     which the RTX never stored — it arrives in this turn's prompt).
                        const bool full_hash_ok = lead.size() == (size_t) stored_count &&
                            full_n > (int) stored_count &&
                            mac_fnv1a_tokens(lead.data(), (int32_t) stored_count)
                                == ctx_internal->get_mac_retained_hash();
                        // (2) HARD GUARD: direct token compare of the stored prefix.
                        bool tok_ok = !stored_prefix.empty() &&
                            stored_prefix.size() <= lead.size() &&
                            full_n > (int) stored_prefix.size();
                        for (size_t t = 0; tok_ok && t < stored_prefix.size(); ++t) {
                            if (stored_prefix[t] != (llama_token) lead[t]) { tok_ok = false; }
                        }
                        if (full_hash_ok && tok_ok) {
                            use_mac_retained_prefix = true;
                            mac_reuse_count         = stored_count;
                            mac_reuse_hash          = ctx_internal->get_mac_retained_hash() & 0x7fffffffffffffffULL;
                        } else if (!full_hash_ok && tok_ok && stored_prefix.size() < (size_t) stored_count) {
                            std::vector<int32_t> verified_lead(stored_prefix.begin(), stored_prefix.end());
                            use_mac_retained_prefix = true;
                            mac_reuse_count         = (uint32_t) verified_lead.size();
                            mac_reuse_hash          = mac_fnv1a_tokens(
                                verified_lead.data(), (int32_t) verified_lead.size());
                            fprintf(stderr,
                                "*** SERVER: MAC-RETAINED selected reuse=%u full_count=%u "
                                "reason=full-hash-miss,prefix-lead-match [MAC_RETAIN_HANDLE]\n",
                                mac_reuse_count, stored_count);
                            fflush(stderr);
                        } else {
                            ctx_internal->clear_mac_retained_handle();
                            fprintf(stderr,
                                "*** SERVER: Mac-retain handle REJECTED (hash_ok=%d tok_ok=%d) — full PROMPT-DIRECT [MAC_RETAIN_HANDLE]\n",
                                (int) full_hash_ok, (int) tok_ok);
                            fflush(stderr);
                        }
                    }
                }
                if (stored_count > 0 && !use_mac_retained_prefix &&
                        ctx_internal->get_mac_retained_count() > 0) {
                    ctx_internal->clear_mac_retained_handle();
                    fprintf(stderr,
                            "*** SERVER: Mac-retain handle REJECTED (count/prefix/media validation) [MAC_RETAIN_HANDLE]\n");
                    fflush(stderr);
                }
            }
            const bool use_cached_prefix_handoff =
                cache_reuse_narrowed_batch && cached_prefix_handoff_viable;
            const bool cached_prefix_prompt_direct =
                use_cached_prefix_handoff && []() {
                    const char * env = std::getenv("LLAMA_PREFILL_CACHED_PREFIX_PROMPT_DIRECT");
                    return env && env[0] != '\0' && env[0] != '0';
                }();
            const bool export_cached_prefix_kv =
                use_cached_prefix_handoff && !cached_prefix_prompt_direct && !use_mac_retained_prefix &&
                !use_spliced_prompt_direct && !use_splice_seed_prompt_direct;
            GGML_ASSERT(!(export_cached_prefix_kv && mac_reuse_count > 0));
            const bool prefer_gated_cached_prefix =
                mac_retain_handle_enabled && export_cached_prefix_kv;
            const uint32_t live_streamed_prefix =
                ctx_internal->get_prefill_handoff_streamed_prefix_count();
            const bool continuation_prefix_ready = chunk_route_fix
                ? (cache_reuse_narrowed_batch || chunked_prompt_continuation) &&
                    batch_prefix_pos >= (int) live_streamed_prefix
                : cache_reuse_narrowed_batch &&
                    n_past_cached >= (int) live_streamed_prefix;
            const bool streaming_prefill_continuation =
                ctx_internal->is_prefill_streaming() &&
                ctx_internal->has_decode_handoff_pending() &&
                continuation_prefix_ready &&
                !use_mac_retained_prefix &&
                !use_cached_prefix_handoff &&
                live_streamed_prefix > 0;
            if (route_reuse_enabled && slot_batched && slot_batched->task) {
                const char * route_chosen = nullptr;
                if (use_spliced_prompt_direct) {
                    route_chosen = "PROMPT-DIRECT SPLICE";
                } else if (use_splice_seed_prompt_direct) {
                    route_chosen = "PROMPT-DIRECT SPLICE-SEED";
                } else if (use_mac_retained_prefix && !streaming_prefill_continuation) {
                    route_chosen = "PROMPT-DIRECT MAC-RETAINED";
                } else if ((small_delta_path || prefer_gated_cached_prefix) && !streaming_prefill_continuation) {
                    route_chosen = export_cached_prefix_kv
                        ? "CACHED-PREFIX HANDOFF"
                        : (cached_prefix_prompt_direct ? "PROMPT-DIRECT CACHED-PREFIX" : "PROMPT-DIRECT");
                } else {
                    route_chosen =
                        (cache_reuse_narrowed_batch || chunked_prompt_continuation)
                        ? "STREAMING CONTINUATION"
                        : "STREAMING PREFILL";
                }
                SRV_INF("DSV4 route-reuse decision: reused_n=%d delta_n=%d route=%s\n",
                        use_mac_retained_prefix ? (int) mac_reuse_count : n_past_cached,
                        use_mac_retained_prefix
                            ? ((int) slot_batched->task->n_tokens() - (int) mac_reuse_count)
                            : batch.n_tokens,
                        route_chosen);
            }
            if (small_delta_path && streaming_prefill_continuation) {
                fprintf(stderr,
                        "*** SERVER: PROMPT-DIRECT skipped for streamed-prefill continuation "
                        "(delta=%d, n_past=%d, streamed_prefix=%u); keeping tail on RTX\n",
                        batch.n_tokens, n_past_cached, live_streamed_prefix);
                fflush(stderr);
            }
            if (use_spliced_prompt_direct || use_splice_seed_prompt_direct ||
                    ((small_delta_path || use_mac_retained_prefix || prefer_gated_cached_prefix) &&
                    !streaming_prefill_continuation)) {
                const char * route_tag = use_spliced_prompt_direct
                    ? "PROMPT-DIRECT SPLICE"
                    : (use_splice_seed_prompt_direct
                        ? "PROMPT-DIRECT SPLICE-SEED"
                        : (use_mac_retained_prefix
                            ? "PROMPT-DIRECT MAC-RETAINED"
                            : (export_cached_prefix_kv
                                ? "CACHED-PREFIX HANDOFF"
                                : (cached_prefix_prompt_direct ? "PROMPT-DIRECT CACHED-PREFIX" : "PROMPT-DIRECT"))));
                const int route_delta = use_mac_retained_prefix && slot_batched && slot_batched->task
                    ? (int) slot_batched->task->n_tokens() - (int) mac_reuse_count
                    : batch.n_tokens;
                fprintf(stderr, "*** SERVER: %s — delta=%d tokens < %d threshold%s, handing off to Mac\n",
                        route_tag, route_delta, threshold,
                        export_cached_prefix_kv
                            ? (std::string(" (n_past=") + std::to_string(n_past_cached) +
                               " cached KV cells will be exported)").c_str()
                            : (cache_reuse_narrowed_batch
                                ? std::string(" (cache reuse without viable blob — full prompt re-prefill)").c_str()
                                : "")); fflush(stderr);

                // Write token IDs to temp file (legacy — only used by pure
                // prompt-direct path which sends raw tokens).
                const char * tmp_env = std::getenv("TMP");
                std::string tmp_path = std::string(tmp_env ? tmp_env : "/tmp") +
                                       "/llama_prompt_" + std::to_string(ggml_time_us()) + ".bin";
                if (!export_cached_prefix_kv && !use_mac_retained_prefix &&
                        !use_spliced_prompt_direct && !use_splice_seed_prompt_direct) {
                    FILE * f = fopen(tmp_path.c_str(), "wb");
                    if (f) { fwrite(batch.token, sizeof(int32_t), batch.n_tokens, f); fclose(f); }
                }

                std::string kv_host = ctx_internal->get_prefill_kv_host();
                int kv_port = ctx_internal->get_prefill_kv_port();
                std::remove(tmp_path.c_str());

                ctx_internal->clear_prefill_handoff_prompt();
                if (use_spliced_prompt_direct || use_splice_seed_prompt_direct) {
                    // Complete prompt, zero KV, zero retained-count semantics.
                    // This branch is unreachable gate-off and never enters FIX-B.
                    ctx_internal->set_mac_retained_reuse_count(0);
                    ctx_internal->set_mac_retained_reuse_hash(0);
                    ctx_internal->prompt_direct_export = true;
                    ctx_internal->set_prompt_direct_tokens_only(true);
                    ctx_internal->arm_decode_handoff_for_prompt_direct();
                } else if (export_cached_prefix_kv) {
                    // CACHED-PREFIX HANDOFF: export KV to Mac. The retained
                    // prefill_kv_state_cache from the prior streaming handoff
                    // holds EXACTLY the n_past cells Mac needs (prior turn's
                    // exported blob, bytes-identical). Export will reuse it
                    // via the `!prefill_kv_state_cache.empty()` branch
                    // instead of doing a fresh capture from live llama_memory
                    // (which doesn't have the streaming-specific KV layout
                    // Mac expects and would crash in capture).
                    ctx_internal->prompt_direct_export = false;
                    if (mac_retain_handle_enabled) {
                        ctx_internal->clear_mac_retained_handle();
                    }
                    ctx_internal->arm_decode_handoff_for_cached_prefill();
                } else if (use_mac_retained_prefix) {
                    // Option B MAC-RETAINED continuation: ship ONLY the delta (the leading
                    // `mac_reuse_count` tokens are trimmed from the handoff prompt below) and
                    // arm the reuse count so the config carries "reuse N retained tokens" to
                    // the Mac. Skip Gate #2 (that's a different, KV-handshake mechanism);
                    // this path relies on the Mac's own Phase-74 retained cache.
                    ctx_internal->set_mac_retained_reuse_count(mac_reuse_count);
                    ctx_internal->set_mac_retained_reuse_hash(mac_reuse_hash);
                    ctx_internal->prompt_direct_export = true;
                    ctx_internal->arm_decode_handoff_for_prompt_direct();
                } else {
                    // Fresh PROMPT-DIRECT: raw tokens, no KV.
                    // Gate #2: query Mac's prefix cache before sending full token stream.
                    // Uses control socket at kv_port-1. On miss/timeout, falls back to full send.
                    bool pd_cache_hit = false;
                    if (kv_port > 1 && slot_batched && slot_batched->task) {
                        // Collect full prompt token IDs for hashing.
                        std::vector<int32_t> pd_tokens;
                        pd_tokens.reserve(slot_batched->task->n_tokens());
                        for (int t = 0; t < (int) slot_batched->task->n_tokens(); ++t) {
                            const llama_token tok = slot_batched->task->tokens[t];
                            if (tok == LLAMA_TOKEN_NULL) { break; }
                            pd_tokens.push_back((int32_t) tok);
                        }
                        if (!pd_tokens.empty()) {
                            // Compute prefix hash and model fingerprint locally via
                            // xxhash (inline in this TU — no link dependency).
                            const uint64_t pd_prefix_hash = pd_hash_tokens(
                                pd_tokens.data(), (int32_t) pd_tokens.size());
                            char pd_model_desc[256] = {};
                            if (model) {
                                llama_model_desc(model, pd_model_desc, sizeof(pd_model_desc));
                            }
                            const uint64_t pd_model_fp = pd_model_fingerprint(pd_model_desc);

                            pd_prefix_cache_reply pd_reply{};
                            const bool pd_queried = pd_query_prefix_cache(
                                kv_host, kv_port - 1,
                                pd_prefix_hash, (uint32_t) pd_tokens.size(), pd_model_fp,
                                &pd_reply);
                            if (pd_queried && pd_reply.hit &&
                                pd_reply.cached_len_tokens == (uint32_t) pd_tokens.size()) {
                                // Full cache hit: Mac already has prefill KV for this prompt.
                                // Skip token send; tell Mac to resume from its cached session.
                                fprintf(stderr, "*** SERVER: PROMPT-DIRECT cache HIT (hash=0x%llx, sid=0x%llx) — skipping token send\n",
                                        (unsigned long long) pd_prefix_hash,
                                        (unsigned long long) pd_reply.session_id); fflush(stderr);
                                ctx_internal->prompt_direct_export = false;
                                ctx_internal->arm_decode_handoff_for_prompt_direct_cache_hit(pd_reply.session_id);
                                pd_cache_hit = true;
                            }
                        }
                    }
                    if (!pd_cache_hit) {
                        ctx_internal->prompt_direct_export = true;
                        ctx_internal->arm_decode_handoff_for_prompt_direct();
                    }
                }
                ctx_internal->update_n_p_eval(use_spliced_prompt_direct
                    ? (int) slot_batched->task->splice_prompt_override.size()
                    : (use_splice_seed_prompt_direct
                        ? (int) slot_batched->task->n_tokens()
                        : (cache_reuse_narrowed_batch
                            ? (int) slot_batched->task->n_tokens()
                            : batch.n_tokens)));

                // llama_decode_trigger_handoff does the full cycle:
                // export (raw tokens) → TCP transport → token relay → buffer
                // The relay callback populates relay_token_buffer.
                handoff_slot = nullptr;
                for (auto & slot : slots) {
                    if (slot.is_processing()) {
                        handoff_slot = &slot;
                        slot.t_start_generation = ggml_time_us();
                        slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                        metrics.on_prompt_eval(slot);
                        break;
                    }
                }

                std::vector<llama_token> mac_retry_full_prompt;
                std::string mac_retry_prompt_text;

                if (handoff_slot && handoff_slot->task) {
                    auto * ctx_ho = static_cast<llama_context *>(ctx);
                    ctx->set_prefill_sampling_profile(
                        make_handoff_sampling_profile(handoff_slot->task->params.sampling));
                    ctx->set_prefill_requested_n_predict(handoff_slot->task->params.n_predict);
                    ctx_ho->set_handoff_slot_id(handoff_slot->id);
                    ctx_ho->set_prefill_handoff_capture_seq_id(handoff_slot->id);
                    ctx_ho->set_decode_stop_words(handoff_slot->task->params.antiprompt);

                    // 2026-05-31 vision EXPORT FIX: for an image request the prompt token
                    // stream has LLAMA_TOKEN_NULL placeholders at the image cells. Breaking
                    // at the first NULL truncated the advertised prompt to the pre-image
                    // text (19 tok) and shipped streamed_prefix=0, so the Mac re-prefilled
                    // the image positions locally (no pixels -> garbage). The image KV is
                    // already stream-prefilled into the exported blob, so instead emit the
                    // media-pad id for image cells (preserve the full count) and advertise
                    // streamed_prefix = the RTX-prefilled cell count so the Mac imports the
                    // image KV and re-prefills only the genuine text suffix after the image.
                    const bool ho_has_media = handoff_slot->task->tokens.has_media();
                    bool ho_saw_image = false;
                    std::vector<llama_token> full_prompt;
                    const llama_tokens * selected_splice = use_spliced_prompt_direct
                            ? &handoff_slot->task->splice_prompt_override : nullptr;
                    const size_t selected_n = selected_splice
                            ? selected_splice->size() : (size_t) handoff_slot->task->n_tokens();
                    full_prompt.reserve(selected_n);
                    for (size_t t = 0; t < selected_n; ++t) {
                        llama_token tok = selected_splice ? (*selected_splice)[t] : handoff_slot->task->tokens[t];
                        if (tok == LLAMA_TOKEN_NULL) {
                            if (ho_has_media) {
                                tok = 163606; // Kimi-VL <|media_pad|>
                                ho_saw_image = true;
                            } else {
                                break;
                            }
                        }
                        full_prompt.push_back(tok);
                    }
                    if (use_spliced_prompt_direct || use_splice_seed_prompt_direct) {
                        // Exact complete vector used by the post-relay witness;
                        // task.tokens remains the untouched full fallback.
                        handoff_slot->mac_splice.logical_prompt_sent = full_prompt;
                    }

                    // streamed_prefix_count: for fresh prompt-direct it's 0
                    // (no KV on Mac). For cache-reuse it's n_past (cells
                    // exported from live Windows KV — Mac imports them and
                    // prefills only the delta).
                    uint32_t ho_streamed_prefix =
                        export_cached_prefix_kv ? (uint32_t) cached_prefix_export_count : 0u;
                    // Option B MAC-RETAINED: trim the leading `mac_reuse_count` tokens so the
                    // handoff ships ONLY the delta continuation. streamed_prefix stays 0 (no KVA
                    // import) — the Mac splices onto its own retained cache via reuse_retained_count.
                    if (use_mac_retained_prefix && !ho_has_media &&
                        full_prompt.size() > (size_t) mac_reuse_count) {
                        mac_retry_full_prompt = full_prompt;
                        mac_retry_prompt_text = handoff_slot->task->tokens.detokenize(ctx, true);
                        fprintf(stderr, "*** SERVER: MAC-RETAINED expected hash=0x%llx [MAC_RETAIN_HANDLE]\n",
                                (unsigned long long) mac_reuse_hash);
                        full_prompt.erase(full_prompt.begin(),
                                          full_prompt.begin() + (size_t) mac_reuse_count);
                        fprintf(stderr,
                            "*** SERVER: MAC-RETAINED continuation — reuse=%u delta=%zu cache_n=%u [MAC_RETAIN_HANDLE]\n",
                            mac_reuse_count, full_prompt.size(), mac_reuse_count);
                        fflush(stderr);
                    }
                    if (ho_has_media && ho_saw_image) {
                        // The RTX stream-prefilled text+image into the exported blob; advertise
                        // that full extent (handoff_slot->prompt.n_tokens() = prefilled cells).
                        ho_streamed_prefix = (uint32_t) handoff_slot->prompt.n_tokens();
                        fprintf(stderr, "*** VISION-HANDOFF: full_prompt=%zu streamed_prefix=%u (image KV imported, Mac prefills text suffix only) [VISION_ROUTE_V2]\n",
                                full_prompt.size(), ho_streamed_prefix);
                        fflush(stderr);
                    }
                    ctx_ho->set_prefill_handoff_prompt(
                        full_prompt.empty() ? nullptr : full_prompt.data(),
                        full_prompt.size(),
                        (use_spliced_prompt_direct || use_splice_seed_prompt_direct)
                            ? std::string() : handoff_slot->task->tokens.detokenize(ctx, true),
                        ho_streamed_prefix);
                    if (use_spliced_prompt_direct || use_splice_seed_prompt_direct) {
                        SRV_INF("route=%s canonical_n=%zu tail_n=%zu sent_n=%zu token_bytes=%zu "
                                "artifact_bytes=pending kv_bytes=0 lineage=ok boundary=ok witness=pending\n",
                                use_spliced_prompt_direct ? "PROMPT-DIRECT SPLICE" : "PROMPT-DIRECT SPLICE-SEED",
                                use_spliced_prompt_direct ? handoff_slot->mac_splice.canonical_tokens.size() : 0,
                                use_spliced_prompt_direct ? handoff_slot->task->splice_tail_tokens.size() : 0,
                                full_prompt.size(), full_prompt.size() * sizeof(int32_t));
                    }

                    // IMG-PROMPT-DIRECT: clip the image HERE (fast vision-tower, NOT the 51s LM-stream)
                    // and stash the embd + raw_image + question. The export reuses the ship-embd logic
                    // ([prefix][suffix] + embd + raw_image, image range skipped via img_cell_count); the
                    // wait-loop matches via cand_img; the MLX dispatches via raw_image -> generate(). No
                    // RTX LM-prefill at all. (USER 2026-06-01 text-skip: ~14s vs 38s.)
                    if (small_img_pd) {
                        auto & st = handoff_slot->task->tokens;
                        int img_pos = -1;
                        for (int t = 0; t < (int) st.size(); ++t) {
                            if (st[t] == LLAMA_TOKEN_NULL) { img_pos = t; break; }
                        }
                        if (img_pos >= 0) {
                            const auto & img_chunk = st.find_chunk((size_t) img_pos);
                            if (img_chunk && mtmd_encode_chunk(mctx, img_chunk.get()) == 0) {
                                float * embd = mtmd_get_output_embd(mctx);
                                const size_t n_img = mtmd_input_chunk_get_n_tokens(img_chunk.get());
                                if (embd && n_img > 0) {
                                    ctx_ho->set_prefill_handoff_image_embd(embd, (uint32_t) n_img,
                                        (uint32_t) llama_model_n_embd(model), (uint32_t) img_pos);
                                }
                            }
                        }
                        if (!st.raw_image_files.empty()) {
                            // v1.10 multi-turn: pack ALL images (b"MIMG"+[N][sizes][blobs]) — was [0] only.
                            // st.question_text now carries the structured vlm_conversation JSON (full history).
                            std::vector<uint8_t> packed = {'M','I','M','G'};
                            auto pu32 = [&](uint32_t v){ packed.push_back((uint8_t)(v&0xFF)); packed.push_back((uint8_t)((v>>8)&0xFF)); packed.push_back((uint8_t)((v>>16)&0xFF)); packed.push_back((uint8_t)((v>>24)&0xFF)); };
                            pu32((uint32_t) st.raw_image_files.size());
                            for (auto & im : st.raw_image_files) pu32((uint32_t) im.size());
                            for (auto & im : st.raw_image_files) packed.insert(packed.end(), im.begin(), im.end());
                            ctx_ho->set_prefill_handoff_raw_image(packed.data(), packed.size(), st.question_text);
                            fprintf(stderr, "*** IMG-PROMPT-DIRECT: packed %zu images (%zu bytes) [VISION_ROUTE_V2]\n",
                                    st.raw_image_files.size(), packed.size());
                        }
                        fprintf(stderr, "*** IMG-PROMPT-DIRECT: clipped+stashed (no RTX LM-prefill), img_pos=%d [VISION_ROUTE_V2]\n", img_pos);
                        fflush(stderr);
                    }
                }

                handoff_cancelled = false;
                relay_token_buffer.clear();
                bool send_ok = llama_decode_trigger_handoff(ctx);
                // Option B: the reuse count was consumed by the plan build inside the trigger
                // (refresh_decode_handoff_plan). Reset it so it can never leak into a later,
                // unrelated handoff. Harmless no-op when it was never armed.
                ctx_internal->set_mac_retained_reuse_count(0);
                ctx_internal->set_mac_retained_reuse_hash(0);

                // A retained-prefix NACK is a pre-decode safety result. Retry
                // exactly once with an independently validated physical prefix;
                // never let the delta fall through to local decode.
                if (use_mac_retained_prefix && ctx_internal->get_last_relay_retained_prefix_miss()) {
                    ctx_internal->clear_mac_retained_handle();
                    relay_token_buffer.clear();

                    uint32_t fallback_prefix_count = 0;
                    const auto physical_prefix_matches = [&]() {
                        const auto & prefix = ctx_internal->get_prefill_kv_state_cache_prefix_tokens();
                        if (!ctx_internal->has_kv_blob_cache() || prefix.empty() ||
                                prefix.size() >= mac_retry_full_prompt.size()) {
                            return false;
                        }
                        for (size_t i = 0; i < prefix.size(); ++i) {
                            if (prefix[i] != mac_retry_full_prompt[i]) {
                                return false;
                            }
                        }
                        fallback_prefix_count = (uint32_t) prefix.size();
                        return true;
                    };

                    bool have_physical_fallback = physical_prefix_matches();
                    if (!have_physical_fallback && ctx_internal->kv_store_enabled() &&
                            mac_retry_full_prompt.size() > 64) {
                        std::vector<llama_token> disk_prefix;
                        fallback_prefix_count = ctx_internal->kv_store_install_prefix(
                            mac_retry_full_prompt, 64, disk_prefix);
                        have_physical_fallback = fallback_prefix_count > 0 &&
                            fallback_prefix_count < mac_retry_full_prompt.size() &&
                            disk_prefix.size() == fallback_prefix_count;
                    }

                    if (have_physical_fallback) {
                        fprintf(stderr,
                                "*** SERVER: retained-prefix-miss; retrying once via CACHED-PREFIX HANDOFF (prefix=%u full=%zu)\n",
                                fallback_prefix_count, mac_retry_full_prompt.size());
                        fflush(stderr);
                        ctx_internal->clear_prefill_handoff_prompt();
                        ctx_internal->prompt_direct_export = false;
                        ctx_internal->set_mac_retained_reuse_count(0);
                        ctx_internal->set_mac_retained_reuse_hash(0);
                        GGML_ASSERT(ctx_internal->get_mac_retained_reuse_count() == 0);
                        ctx_internal->arm_decode_handoff_for_cached_prefill();
                        ctx_internal->set_prefill_sampling_profile(
                            make_handoff_sampling_profile(handoff_slot->task->params.sampling));
                        ctx_internal->set_prefill_requested_n_predict(handoff_slot->task->params.n_predict);
                        ctx_internal->set_decode_stop_words(handoff_slot->task->params.antiprompt);
                        ctx_internal->set_handoff_slot_id(handoff_slot->id);
                        ctx_internal->set_prefill_handoff_capture_seq_id(handoff_slot->id);
                        ctx_internal->set_prefill_handoff_prompt(
                            mac_retry_full_prompt.data(), mac_retry_full_prompt.size(),
                            mac_retry_prompt_text, fallback_prefix_count);
                        ctx_internal->update_n_p_eval((int) mac_retry_full_prompt.size());
                        send_ok = llama_decode_trigger_handoff(ctx);
                        ctx_internal->set_mac_retained_reuse_count(0);
                        ctx_internal->set_mac_retained_reuse_hash(0);
                        if (ctx_internal->get_last_relay_retained_prefix_miss()) {
                            send_ok = false;
                        }
                    } else {
                        fprintf(stderr,
                                "*** SERVER: retained-prefix-miss with no viable physical fallback; refusing bare-delta decode\n");
                        fflush(stderr);
                        send_ok = false;
                    }
                }

                std::remove(tmp_path.c_str());

                if (handoff_cancelled) {
                    cleanup_cancelled_handoff(true, true);
                    handoff_cancelled = false;
                    prompt_direct_done = true;
                } else if (send_ok && !relay_token_buffer.empty()) {
                    fprintf(stderr, "*** SERVER: prompt-direct SUCCESS (%zu tokens relayed)\n",
                            relay_token_buffer.size()); fflush(stderr);

                    // Get stop reason from Mac decode relay result
                    int pd_content_tokens = 0;
                    for (const auto & rt : relay_token_buffer) {
                        if (!rt.done) pd_content_tokens++;
                    }
                    auto * pd_ctx = static_cast<llama_context *>(ctx);
                    const int32_t pd_mac_stop = pd_ctx->get_last_relay_stop_reason();
                    stop_type pd_stop_type = STOP_TYPE_EOS;
                    if (pd_mac_stop == LLAMA_STOP_WORD) {
                        pd_stop_type = STOP_TYPE_WORD;
                    } else if (pd_mac_stop == LLAMA_STOP_LIMIT) {
                        pd_stop_type = STOP_TYPE_LIMIT;
                    }

                    // Send HTTP response from buffered relay tokens
                    for (auto & slot : slots) {
                        if (slot.is_processing()) {
                            slot.n_decoded = (int32_t)pd_content_tokens;
                            slot.relay_decode_elapsed_ms = pd_ctx->get_last_relay_decode_elapsed_ms();
                            slot.relay_decode_intervals = pd_ctx->get_last_relay_decode_intervals();
                            slot.relay_decode_tok_s = pd_ctx->get_last_relay_decode_tok_s();
                            slot.relay_decode_tok_s_early = pd_ctx->get_last_relay_decode_tok_s_early();
                            slot.relay_decode_tok_s_middle = pd_ctx->get_last_relay_decode_tok_s_middle();
                            slot.relay_decode_tok_s_late = pd_ctx->get_last_relay_decode_tok_s_late();
                            if (slot.task && slot.task->params.stream) {
                                slot.stop = pd_stop_type;
                                send_final_response(slot);
                            } else {
                                std::string full_text;
                                for (auto & rt : relay_token_buffer) {
                                    if (rt.done) break;
                                    full_text += rt.text;
                                }
                                auto res = std::make_unique<server_task_result_cmpl_final>();
                                res->id      = slot.task->id;
                                res->index   = slot.task->index;
                                res->content = full_text;
                                res->n_decoded         = slot.n_decoded;
                                res->n_prompt_tokens   = slot.task->n_tokens();
                                res->verbose           = slot.task->params.verbose;
                                res->res_type          = slot.task->params.res_type;
                                res->oaicompat_model   = slot.task->params.oaicompat_model;
                                res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;
                                slot.stop = pd_stop_type;
                                res->stop = slot.stop;
                                res->timings = slot.get_timings();
                                queue_results.send(std::move(res));
                            }
                            metrics.on_prediction(slot);
                            // Commit only after the Mac's full-chain count/hash
                            // matches logical_prompt_sent || every non-done relay ID.
                            commit_mac_splice_state(slot, true);
                            record_mac_retained_handle(slot, true);
                            slot.release();
                            // Clear prompt cache AFTER release — the server's KV
                            // is empty (Mac prefilled, not RTX). Without this,
                            // the next turn tries to reuse nonexistent KV.
                            // Commit only the real cached-prefix route. A plain
                            // prompt-direct request has no local physical KV.
                            if (export_cached_prefix_kv && commit_route_reuse_record(slot)) {
                                prompt_direct_route_record_committed = true;
                            } else if (export_cached_prefix_kv && route_reuse_enabled) {
                                invalidate_route_reuse(&slot,
                                        "cached-prefix handoff completed without a complete physical retention candidate",
                                        true);
                            } else {
                                // Genuine prompt-direct has no local KV to retain LOCALLY.
                                // Option B (LLAMA_DSV4_MAC_RETAIN_HANDLE): the Mac DOES retain
                                // its Phase-74 cache across turns and just reported the prefix
                                // (prompt+generated) it now holds. Record that handle so the
                                // NEXT turn can route a cached-prefix continuation (Step 2 wires
                                // the delta-only send). RECORD ONLY here — shipping + prompt_clear
                                // are unchanged, so this is byte-identical when the gate is off.
                                if (false && mac_retain_handle_enabled && slot.task) {
                                    const uint32_t rc = ctx_internal->get_last_relay_retained_token_count();
                                    if (rc > 0) {
                                        const uint64_t rh = ctx_internal->get_last_relay_retained_prefix_hash();
                                        const int pn = (int) slot.task->n_tokens();
                                        // Leading prompt tokens we currently have (stop at any
                                        // media placeholder). For a growing chat pn < rc (the Mac's
                                        // rc counts THIS turn's generated tokens, which the RTX prompt
                                        // does not carry) so we store what we have; the full hash +
                                        // direct-token validation runs next turn in the route block,
                                        // where the retained prefix arrives inside the incoming prompt.
                                        std::vector<llama_token> pref;
                                        pref.reserve((size_t) std::min<int>((int) rc, pn));
                                        bool clean = true;
                                        for (int t = 0; t < pn && (uint32_t) t < rc; ++t) {
                                            const llama_token tok = slot.task->tokens[t];
                                            if (tok == LLAMA_TOKEN_NULL) { clean = false; break; }
                                            pref.push_back(tok);
                                        }
                                        // Same-turn round-trip check: only possible when this turn's
                                        // prompt actually covers rc tokens (e.g. an exact re-ask).
                                        const bool covers = clean && (int) pref.size() == (int) rc;
                                        uint64_t lead_hash = 0;
                                        if (covers) {
                                            std::vector<int32_t> lead32(pref.begin(), pref.end());
                                            lead_hash = mac_fnv1a_tokens(lead32.data(), (int32_t) lead32.size());
                                        }
                                        const bool hash_ok = covers && (lead_hash == rh);
                                        ctx_internal->set_mac_retained_handle(rc, rh, pref.data(), pref.size());
                                        fprintf(stderr,
                                            "*** SERVER: Mac-retain handle recorded (count=%u hash=0x%llx "
                                            "prompt_lead=%zu covers=%d hash_ok=%d) cache_n=%u [MAC_RETAIN_HANDLE]\n",
                                            rc, (unsigned long long) rh, pref.size(),
                                            (int) covers, (int) hash_ok, rc);
                                        fflush(stderr);
                                    }
                                }
                                if (!use_mac_retained_prefix) {
                                    slot.prompt_clear(false);
                                }
                            }
                        }
                    }
                    // Clear only this slot's KV — Mac prefilled, not RTX.
                    // Don't clear other slots' KV (they may have valid catch-up state).
                    if (!prompt_direct_route_record_committed) {
                        llama_memory_seq_rm(llama_get_memory(ctx), handoff_slot ? handoff_slot->id : 0, -1, -1);
                    }
                    relay_token_buffer.clear();
                    handoff_slot = nullptr;
                    prompt_direct_done = true;
                } else {
                    const char * reason = relay_token_buffer.empty()
                        ? "prompt-direct returned no tokens (Mac prefill may have timed out)"
                        : "prompt-direct handoff failed";
                    fprintf(stderr, "*** SERVER: prompt-direct FAILED (%s)\n", reason);
                    fflush(stderr);
                    // Send error to client instead of falling through to streaming
                    // (streaming KV export is not yet working for KV-RAM mode)
                    for (auto & slot : slots) {
                        if (slot.is_processing()) {
                            send_error(slot, reason);
                            if (mac_splice_enabled) {
                                slot.mac_splice.clear();
                            }
                            slot.release();
                            invalidate_route_reuse(&slot, reason, true);
                        }
                    }
                    relay_token_buffer.clear();
                    handoff_slot = nullptr;
                    prompt_direct_done = true;
                }
            }

            // Clear the prompt-direct KV state cache so it doesn't get reused
            // by the next request's KV export (prompt-direct writes raw tokens,
            // not actual KV state — reusing it corrupts the next handoff).
            if (prompt_direct_done) {
                auto * ctx_pd = static_cast<llama_context *>(ctx);
                // 2026-05-29: a CACHED-PREFIX HANDOFF (export_cached_prefix_kv)
                // just retained a REAL KV blob in host RAM tagged with its
                // prefix tokens. The next turn reuses it (exact-match or
                // viable-as-prefix) so the Mac prefills only the bounded tail
                // instead of a full re-stream. Unconditionally clearing it here
                // — the original behavior, meant for PROMPT-DIRECT raw-token
                // artifacts (not real KV; reusing them corrupts the next
                // export) — starves turn3+ of the seed blob and forces a full
                // re-prefill EVERY follow-up turn. So skip the clear when the
                // just-completed handoff exported a real KV blob AND the
                // retain-after-handoff opt-in is set. Default off → identical
                // to the prior always-clear behavior (RC-safe). For genuine
                // prompt-direct (export_cached_prefix_kv == false) we always
                // clear, exactly as before.
                if (!prompt_direct_route_record_committed && !use_mac_retained_prefix) {
                    ctx_pd->invalidate_route_reuse_state();
                }
                ctx_pd->clear_prefill_handoff_prompt();
                ctx_pd->prompt_direct_export = false;
            }
        }

        // Phase 132: per-request dynamic batch/ubatch sizing for streaming-prefill.
        // Keep the RTX prefix FA-clean and exactly tiled, but do not let exact
        // divisor math choose arbitrary 512-aligned ubatches. On Kimi MLA FA,
        // 1024 and 3072 have both produced coherent-looking K rows with corrupt
        // decode, while partial 4096 tails produce zero active rows. Prefer the
        // proven power-of-two family (4096, then 2048) and let Mac prefill a
        // bounded suffix under the crossover threshold.
        bool         _dyn_ub_applied = false;
        uint32_t     _saved_n_ubatch = 0;
        uint32_t     _saved_n_batch  = 0;
        // Phase 261 (2026-05-03): user-algo's batch + tail computed inside
        // dynamic-UB block. _dyn_dispatch_batch < batch.n_tokens triggers
        // dispatch-loop clip; tail goes to Mac via handoff.
        uint32_t     _dyn_dispatch_batch = 0;  // 0 = no clip
        uint32_t     _dyn_dispatch_tail  = 0;
        // Phase 266 (2026-05-04): pad-and-trim algorithm needs to know
        // pad_amount separately (for K-cell trim before export). Distinct from
        // _dyn_dispatch_tail (which is the Mac-side tail to prefill).
        uint32_t     _dyn_pad_amount    = 0;
        if (!prompt_direct_done && std::getenv("LLAMA_PREFILL_DYNAMIC_UB") != nullptr) {
            auto * ctx_dub = static_cast<llama_context *>(ctx);
            _saved_n_ubatch = ctx_dub->get_n_ubatch_cached();
            _saved_n_batch  = ctx_dub->get_n_batch_cached();
            if (batch.n_tokens >= 512) {
                // Dynamic UB keeps the RTX work as one FA-clean batch prefix.
                // The microbatch must divide it exactly; prior "close to
                // target" sizing allowed final partial microbatches such as
                // 7*4096+3072, which reproduced stale/zero KV rows.
                auto align_down = [](uint32_t v, uint32_t alignment) -> uint32_t {
                    return v - (v % alignment);
                };
                auto parse_env_u32 = [](const char * name, uint32_t fallback) -> uint32_t {
                    const char * value = std::getenv(name);
                    if (value == nullptr || value[0] == '\0') {
                        return fallback;
                    }
                    char * end = nullptr;
                    const unsigned long parsed = std::strtoul(value, &end, 10);
                    return end != value && parsed > 0 ? (uint32_t) parsed : fallback;
                };
                struct dyn_ub_choice {
                    uint32_t batch = 0;
                    uint32_t ubatch = 0;
                    uint32_t tail = 0;
                    bool found = false;
                };
                auto choose_power2_tiling = [&](uint32_t prompt_n, uint32_t target, uint32_t cap,
                                                uint32_t min_ubatch, uint32_t tail_budget) -> dyn_ub_choice {
                    constexpr uint32_t grid = 512u;
                    dyn_ub_choice best;
                    if (prompt_n < grid) {
                        return best;
                    }

                    target     = align_down(target, grid);
                    cap        = align_down(cap, grid);
                    min_ubatch = align_down(min_ubatch, grid);
                    if (target < grid) {
                        target = grid;
                    }
                    if (cap < grid) {
                        cap = grid;
                    }
                    if (min_ubatch < grid) {
                        min_ubatch = grid;
                    }
                    if (cap > target) {
                        cap = target;
                    }

                    std::vector<uint32_t> preferred;
                    for (uint32_t u = std::min(cap, target); u >= min_ubatch; u /= 2) {
                        if (u % grid == 0) {
                            preferred.push_back(u);
                        }
                        if (u == min_ubatch || u < min_ubatch * 2) {
                            break;
                        }
                    }
                    if (preferred.empty()) {
                        preferred.push_back(std::min(cap, target));
                    }

                    for (uint32_t u : preferred) {
                        if (u == 0) {
                            continue;
                        }
                        const uint32_t batch_candidate = prompt_n - (prompt_n % u);
                        if (batch_candidate == 0 || batch_candidate % grid != 0) {
                            continue;
                        }
                        const uint32_t tail = prompt_n - batch_candidate;
                        if (tail > tail_budget) {
                            continue;
                        }
                        if (std::getenv("LLAMA_PREFILL_REJECT_TINY_TAIL") && tail > 0 && tail < u) {
                            continue;
                        }
                        if (!best.found || batch_candidate > best.batch ||
                                (batch_candidate == best.batch && u > best.ubatch)) {
                            best = dyn_ub_choice{ batch_candidate, u, tail, true };
                        }
                    }
                    return best;
                };
                auto choose_any_exact_tiling = [&](uint32_t prompt_n, uint32_t target, uint32_t cap,
                                                   uint32_t min_ubatch, uint32_t tail_budget) -> dyn_ub_choice {
                    constexpr uint32_t grid = 512u;
                    dyn_ub_choice best;
                    uint32_t best_diff = 0xffffffffu;
                    uint32_t best_tail = 0xffffffffu;

                    target     = std::max<uint32_t>(grid, align_down(target, grid));
                    cap        = std::max<uint32_t>(grid, align_down(std::min(cap, target), grid));
                    min_ubatch = std::max<uint32_t>(grid, align_down(min_ubatch, grid));

                    const uint32_t first_batch = align_down(prompt_n, grid);
                    const uint32_t min_batch = prompt_n > tail_budget
                        ? align_down(prompt_n - tail_budget, grid)
                        : grid;
                    for (uint32_t batch_candidate = first_batch; batch_candidate >= min_batch && batch_candidate >= grid;
                            batch_candidate -= grid) {
                        const uint32_t tail = prompt_n - batch_candidate;
                        for (uint32_t u = min_ubatch; u <= cap; u += grid) {
                            if (batch_candidate % u != 0) {
                                continue;
                            }
                            if (std::getenv("LLAMA_PREFILL_REJECT_TINY_TAIL") && tail > 0 && tail < u) {
                                continue;
                            }
                            const uint32_t diff = u > target ? u - target : target - u;
                            if (!best.found || diff < best_diff ||
                                    (diff == best_diff && (tail < best_tail ||
                                     (tail == best_tail && u > best.ubatch)))) {
                                best = dyn_ub_choice{ batch_candidate, u, tail, true };
                                best_diff = diff;
                                best_tail = tail;
                            }
                        }
                        if (batch_candidate == grid) {
                            break;
                        }
                    }
                    return best;
                };
                // Opt-in: allow a partial final microbatch on RTX. With this
                // enabled, the dynamic UB picks the largest 512-aligned prefix
                // (regardless of ub divisibility) at the configured target
                // ubatch. The prefill-stream FA-aligned prefix logic processes
                // the resulting partial last microbatch on RTX (saving the
                // Mac local-prefill cost). The KV-RAM zero-row guard added
                // alongside this catches the historical silent corruption
                // case so failures surface as a hard export error rather than
                // garbage decode. Use only after verifying coherence on a
                // probe prompt.
                auto choose_partial_ub_tiling = [&](uint32_t prompt_n, uint32_t target, uint32_t cap,
                                                    uint32_t /*min_ubatch*/, uint32_t /*tail_budget*/) -> dyn_ub_choice {
                    constexpr uint32_t grid = 512u;
                    dyn_ub_choice best;

                    target = std::max<uint32_t>(grid, align_down(target, grid));
                    cap    = std::max<uint32_t>(grid, align_down(std::min(cap, target), grid));

                    const uint32_t batch_candidate = align_down(prompt_n, grid);
                    if (batch_candidate < grid) {
                        return best;
                    }
                    const uint32_t u    = std::min(cap, target);
                    const uint32_t tail = prompt_n - batch_candidate;

                    // Decline this strategy when the chosen batch is NOT an exact
                    // multiple of the ubatch. decode_layer_major builds a single
                    // graph sized to the largest ubatch (see llama-context.cpp
                    // PHASE 2 comment) and patches FA tensor ne[1] per-ubatch in
                    // the layer loop, but K SET_ROWS / k_idxs are NOT patched.
                    // Diag (LLAMA_K_SET_ROWS_DIAG=1, commit f86853fc) confirmed
                    // only ne=1 (warmup) and ne=ubatch graphs build — the partial
                    // last microbatch reuses the ubatch-sized graph and dispatches
                    // (ubatch - n_tokens) stale SET_ROWS row writes through stale
                    // k_idxs entries, producing the deterministic 12988-zero
                    // striped clobber documented across 8 ruled-out env-knob
                    // variants. Falling back to power2/exact-divisor here keeps
                    // the partial tail off RTX (handed to the receiver via
                    // remote-tail-handoff) until the K SET_ROWS path is patched
                    // per-ubatch the same way FA already is.
                    if (u > 0 && (batch_candidate % u) != 0) {
                        return best;
                    }
                    best = dyn_ub_choice{ batch_candidate, u, tail, true };
                    return best;
                };
                const char * _cap_env = std::getenv("LLAMA_PREFILL_DYNAMIC_UB_MAX");
                uint32_t _cap = _cap_env ? (uint32_t) std::strtoul(_cap_env, nullptr, 10) : 16384u;
                _cap = align_down(_cap, 512u);
                if (_cap == 0) {
                    _cap = 512;
                }
                uint32_t _target = _saved_n_ubatch > 0 ? _saved_n_ubatch : 4096u;
                _target = align_down(_target, 512u);
                if (_target == 0) {
                    _target = 512;
                }
                uint32_t _min_good_ubatch = parse_env_u32("LLAMA_PREFILL_DYNAMIC_UB_MIN", 2048u);
                _min_good_ubatch = align_down(_min_good_ubatch, 512u);
                if (_min_good_ubatch == 0) {
                    _min_good_ubatch = 512;
                }
                uint32_t _tail_budget = parse_env_u32("LLAMA_PREFILL_DYNAMIC_TAIL_MAX",
                        (uint32_t) std::max(3072, ctx_dub->get_prefill_crossover_tokens()));
                if (_tail_budget < 511u) {
                    _tail_budget = 511u;
                }

                const uint32_t prompt_n = (uint32_t) batch.n_tokens;
                uint32_t _dyn_batch  = align_down(prompt_n, 512u);
                uint32_t _dyn_ubatch = _saved_n_ubatch;
                uint32_t _tail       = prompt_n - _dyn_batch;

                if (_dyn_batch == 0) {
                    // Prompt smaller than one target ubatch — let the
                    // existing streaming loop handle it untouched.
                    _dyn_ubatch = _saved_n_ubatch;
                    _dyn_batch  = _saved_n_batch;
                } else {
                    const bool _allow_partial_ub = parse_env_u32("LLAMA_PREFILL_DYNAMIC_ALLOW_PARTIAL_UB", 0u) != 0u;
                    dyn_ub_choice choice;
                    const char * _strategy = "fallback";

                    // Phase 259/260 (2026-05-03): user-spec'd algorithm for clean
                    // chunking. For ANY prompt size:
                    //   1. n_ub = ceil(prompt_n / base_ub_target)
                    //   2. ub = align_down(prompt_n / n_ub, 256)
                    //   3. batch = ub * n_ub
                    //   4. tail = prompt_n - batch (ragged tail goes to Mac)
                    // Worked example from user: prompt=31001, base_ub=4096:
                    //   n_ub = ceil(31001/4096) = 8
                    //   ub = align_down(31001/8, 256) = align_down(3875, 256) = 3840
                    //   batch = 3840 × 8 = 30720
                    //   tail = 31001 - 30720 = 281 ragged tokens for Mac
                    // For prompts ≤ base_ub: single ubatch matching prompt size
                    // (no chunking, no tail).
                    {
                        constexpr uint32_t grid_256 = 256u;
                        // Phase 266 (2026-05-04): user's pad-and-trim algorithm.
                        // For prompt_n, pick:
                        //   1. n_ub = ceil(prompt_n / base_ub_target)
                        //   2. ub = ceil(prompt_n / n_ub) rounded UP to 256-grid
                        //   3. batch = ub × n_ub (≥ prompt_n; difference is padding)
                        //   4. pad_amount = batch - prompt_n
                        // Then RTX processes batch tokens (with pad_amount safe
                        // pad tokens appended) as N uniform 256-aligned ubatches
                        // — no FA-split, no tail-deferral, no multi-ubatch geometry
                        // bug. K cells for padding positions are TRIMMED before
                        // export. Mac receives full prompt + complete trimmed KV.
                        // Mode select via LLAMA_PREFILL_TILING_ALGO=pad-and-trim
                        // (default) or =tail-to-mac (Phase 260 algo).
                        const char * algo_env = std::getenv("LLAMA_PREFILL_TILING_ALGO");
                        const bool use_pad_trim = !algo_env ||
                                                  algo_env[0] == '\0' ||
                                                  std::strcmp(algo_env, "pad-and-trim") == 0;
                        if (use_pad_trim) {
                            // Phase 278 (2026-05-04): support LLAMA_PREFILL_LOCK_UB
                            // env to force ub == _target exactly (no division).
                            // Used by adaptive WF=2 mode where ub must equal the
                            // FA-validated power-of-2 sweet spot (4096) for the
                            // MoE FA kernel correctness. Without lock, pad-and-trim
                            // divides prompt evenly producing non-power-2 ub that
                            // breaks Kimi MLA decode coherence.
                            const char * lock_ub_env = std::getenv("LLAMA_PREFILL_LOCK_UB");
                            const bool lock_ub = lock_ub_env && lock_ub_env[0] != '\0' &&
                                                 lock_ub_env[0] != '0';
                            uint32_t n_ub, ub;
                            if (lock_ub) {
                                ub = _target;
                                n_ub = (prompt_n + ub - 1) / ub;  // ceil
                            } else {
                                n_ub = (prompt_n + _target - 1) / _target;
                                uint32_t min_ub = (prompt_n + n_ub - 1) / n_ub;
                                ub = ((min_ub + grid_256 - 1) / grid_256) * grid_256;
                                if (ub == 0) ub = grid_256;
                            }
                            const uint32_t batch_u = ub * n_ub;
                            const uint32_t pad_amount = batch_u - prompt_n;
                            // tail field repurposed: 0 means batch covers full
                            // prompt (no Mac-side tail prefill needed). pad_amount
                            // stored in _dyn_dispatch_tail for trim logic.
                            //
                            // Phase 3.2 GLM-5.1 16k crash fix (2026-05-12): pad-and-
                            // trim can pick batch_u > _saved_n_batch (initial -b),
                            // e.g. prompt_n=16626 + 9x2048 -> batch=18432 vs
                            // saved=18000. The server's llama_batch was allocated
                            // once at startup with capacity _saved_n_batch, and
                            // set_n_batch on the ctx doesn't resize it.
                            // common_batch_add then asserts at common.cpp:1616
                            // ("llama_batch size exceeded"). Defer this case to
                            // fallback strategies that produce smaller batches.
                            if (batch_u <= _saved_n_batch) {
                                choice = dyn_ub_choice{ batch_u, ub, 0, true };
                                _dyn_pad_amount = pad_amount;  // for trim at export
                                _strategy = "pad-and-trim";
                                fprintf(stderr,
                                        "*** PAD_AND_TRIM: prompt_n=%u n_ub=%u ub=%u batch=%u pad=%u\n",
                                        prompt_n, n_ub, ub, batch_u, pad_amount);
                                fflush(stderr);
                            } else {
                                fprintf(stderr,
                                        "*** PAD_AND_TRIM: SKIPPED — batch=%u > saved_n_batch=%u; falling back\n",
                                        batch_u, _saved_n_batch);
                                fflush(stderr);
                            }
                        } else if (prompt_n <= _target) {
                            choice = dyn_ub_choice{ prompt_n, prompt_n, 0, true };
                            _strategy = "user-algo-single-ub";
                        } else {
                            const uint32_t n_ub = (prompt_n + _target - 1) / _target;
                            uint32_t ub = (prompt_n / n_ub) & ~(grid_256 - 1);
                            if (ub == 0) ub = grid_256;
                            const uint32_t batch_u = ub * n_ub;
                            const uint32_t tail_u = prompt_n - batch_u;
                            choice = dyn_ub_choice{ batch_u, ub, tail_u, true };
                            _strategy = "user-algo";
                        }
                    }
                    if (!choice.found && _allow_partial_ub) {
                        choice = choose_partial_ub_tiling(prompt_n, _target, _cap, _min_good_ubatch, _tail_budget);
                        if (choice.found) _strategy = "partial-ub";
                    }
                    if (!choice.found) {
                        choice = choose_power2_tiling(prompt_n, _target, _cap, _min_good_ubatch, _tail_budget);
                        if (choice.found) _strategy = "power2";
                    }
                    if (!choice.found) {
                        choice = choose_any_exact_tiling(prompt_n, _target, _cap, _min_good_ubatch, _tail_budget);
                        if (choice.found) _strategy = "exact-divisor";
                    }
                    if (!choice.found) {
                        // Last-resort fallback. Phase 259: clamp tail to <256
                        // (sub-grid). If standard chunking would leave a bigger
                        // tail, prefer a single oversized ubatch over the buggy
                        // chunk-with-tail dispatch path.
                        const uint32_t fb_tail = prompt_n - _dyn_batch;
                        if (fb_tail >= 256u) {
                            // Reject the bad chunk-with-tail and use single ubatch
                            choice = dyn_ub_choice{ prompt_n, prompt_n, 0, true };
                            _strategy = "fallback-single-ub-no-tail";
                        } else {
                            choice = dyn_ub_choice{ _dyn_batch, std::min<uint32_t>(_target, _dyn_batch),
                                fb_tail, true };
                            while (choice.ubatch > 512u && choice.batch % choice.ubatch != 0) {
                                choice.ubatch /= 2;
                            }
                        }
                    }
                    _dyn_batch  = choice.batch;
                    _dyn_ubatch = choice.ubatch;
                    _tail       = choice.tail;
                    // Phase 261: hoist for dispatch-loop clipping
                    _dyn_dispatch_batch = _dyn_batch;
                    _dyn_dispatch_tail  = _tail;
                    fprintf(stderr,
                            "*** PREFILL_DYNAMIC_UB_STRATEGY: chose=%s partial_ub_opt_in=%d batch=%u ubatch=%u tail=%u\n",
                            _strategy, _allow_partial_ub ? 1 : 0, _dyn_batch, _dyn_ubatch, _tail);
                    fflush(stderr);
                }

                const uint32_t _min_near_ub = 512u;     // logged for transparency
                const uint32_t _max_near_ub = std::min<uint32_t>(_cap, _dyn_batch);
                if (_dyn_batch != _saved_n_batch || _dyn_ubatch != _saved_n_ubatch) {
                    fprintf(stderr,
                            "*** PREFILL_DYNAMIC_UB: prompt=%d n_batch=%u n_ubatch=%u target_ub=%u near=[%u,%u] cap=%u min_good=%u tail=%u tail_budget=%u (saved ub=%u b=%u)\n",
                            batch.n_tokens, _dyn_batch, _dyn_ubatch, _target, _min_near_ub, _max_near_ub, _cap,
                            _min_good_ubatch, _tail, _tail_budget,
                            _saved_n_ubatch, _saved_n_batch);
                    fflush(stderr);
                    // Drain pending CUDA work before resizing the pool. cuMemSetAccess
                    // fails with "device not ready" if there are outstanding async ops.
                    llama_synchronize(ctx);
                    ctx_dub->set_n_batch(_dyn_batch);
                    ctx_dub->set_n_ubatch(_dyn_ubatch);
                    ctx_dub->set_sched_need_reserve(true);
                    n_batch = (int32_t) _dyn_batch;
                    _dyn_ub_applied = true;
                }
            }
        }

        // Phase 267 (2026-05-04): pad-and-trim — append _dyn_pad_amount safe
        // pad tokens to batch so streaming-prefill processes a 256-aligned batch
        // with uniform 256-aligned ubatches (no FA-split, no partial last ubatch
        // that triggers the K-value bug). K cells for padding positions are
        // trimmed at export (Phase 268 — TODO).
        // Pad token: BOS is safe (model has natural handling). Use seq_id from
        // the active slot. Positions continue sequentially.
        if (!prompt_direct_done && _dyn_pad_amount > 0 && batch.n_tokens > 0) {
            // Determine the seq_id and starting position from the last batch token
            const int32_t last_idx = batch.n_tokens - 1;
            const llama_seq_id seq_id = batch.seq_id[last_idx][0];
            const llama_pos start_pos = batch.pos[last_idx] + 1;
            const llama_vocab * vocab = llama_model_get_vocab(model);
            const llama_token pad_tok = vocab ? llama_vocab_bos(vocab) : LLAMA_TOKEN_NULL;
            const llama_token safe_pad = (pad_tok != LLAMA_TOKEN_NULL) ? pad_tok : 0;
            const int32_t orig_n = batch.n_tokens;
            for (uint32_t k = 0; k < _dyn_pad_amount; ++k) {
                common_batch_add(batch, safe_pad, start_pos + (llama_pos)k, { seq_id }, false);
            }
            fprintf(stderr,
                    "*** PAD_INJECT: appended %u pad tokens (id=%d) at pos=[%d..%d]; batch.n_tokens %d → %d\n",
                    _dyn_pad_amount, (int)safe_pad, (int)start_pos,
                    (int)(start_pos + (llama_pos)_dyn_pad_amount - 1),
                    orig_n, batch.n_tokens);
            fflush(stderr);
        }

        // process the created batch of tokens (skip if prompt-direct handled it)
        // Phase 261 (2026-05-03): clip dispatch to user-algo's _dyn_dispatch_batch
        // so RTX only processes the 256-aligned batch portion. The ragged tail
        // stays in slot.task and is routed to Mac via the existing handoff
        // logic — Mac compares prompt vs received-KV size and prefills the diff.
        // A stream_local request has no Mac tail consumer. Clipping it at the
        // aligned prefix arms the remote-handoff path with transport disabled,
        // which completes as a no-op and returns zero tokens. Keep the full
        // prompt local; decode_streaming already supports the final ragged
        // chunk and preserves its logits/KV for local generation.
        const bool _stream_local_dispatch =
                static_cast<llama_context *>(ctx)->get_prefill_decode_mode() ==
                LLAMA_PREFILL_DECODE_MODE_STREAM_LOCAL;
        const int32_t _dispatch_limit = (!_stream_local_dispatch &&
                                          _dyn_dispatch_batch > 0 &&
                                          _dyn_dispatch_tail > 0 &&
                                          (int32_t) _dyn_dispatch_batch < batch.n_tokens)
                                         ? (int32_t) _dyn_dispatch_batch
                                         : batch.n_tokens;
        if (_dispatch_limit < batch.n_tokens) {
            fprintf(stderr,
                    "*** PHASE_261_CLIP: dispatch limited to %d tokens (user-algo batch=%u tail=%u → tail goes to Mac)\n",
                    _dispatch_limit, _dyn_dispatch_batch, _dyn_dispatch_tail);
            fflush(stderr);
        }
        for (int32_t i = 0; !prompt_direct_done && i < _dispatch_limit; i = i_next) {
            const int32_t n_tokens = std::min(n_batch, _dispatch_limit - i);

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
            };

            auto resume_i_after_streamed_prefix = [&](int prefix_n, const char * label) -> int32_t {
                const int32_t view_pos0 = batch.pos ? (int32_t) batch.pos[i] : i;
                const int32_t consumed_in_view = (int32_t) prefix_n - view_pos0;
                if (consumed_in_view <= 0 || consumed_in_view > n_tokens) {
                    fprintf(stderr,
                            "*** SERVER: %s prefix=%d does not map into current batch view "
                            "(view_pos0=%d n_tokens=%d i=%d); advancing attempted chunk\n",
                            label, prefix_n, view_pos0, n_tokens, i);
                    fflush(stderr);
                    return i + n_tokens;
                }
                const int32_t next_i = i + consumed_in_view;
                if (next_i < i + n_tokens) {
                    fprintf(stderr,
                            "*** SERVER: %s consumed %d/%d tokens in current view; "
                            "resuming RTX stream at batch offset %d (prompt pos %d)\n",
                            label, consumed_in_view, n_tokens, next_i, prefix_n);
                    fflush(stderr);
                }
                return next_i;
            };

            const int ret = llama_decode(ctx, batch_view);

            metrics.on_decoded(slots);

            if (ret != 0) {
                {
                    std::string err;

                    if (n_batch == 1 && ret == 1) {
                        // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                        //       need to remove the tokens from the current batch too
                        err = "Context size has been exceeded.";
                    }

                    if (ret == -1) {
                        err = "Invalid input batch.";
                    }

                    if (ret < -1) {
                        // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                        err = "Compute error.";
                    }

                    // ret == 2: streaming prefill complete, no logits computed.
                    // Trigger the decode handoff: export KV → send to Mac → relay tokens.
                    if (ret == 2) {
                        auto * ctx_sp = static_cast<llama_context *>(ctx);
                        // Phase 268 (2026-05-04): pad-and-trim — if pad-and-trim
                        // injected pad tokens (Phase 267), trim the streamed
                        // count down by pad_amount so the export only includes
                        // K cells for real prompt positions, not padding ones.
                        if (_dyn_pad_amount > 0) {
                            const uint32_t raw_count = ctx_sp->get_prefill_handoff_streamed_prefix_count();
                            const uint32_t trimmed = (raw_count > _dyn_pad_amount)
                                                    ? (raw_count - _dyn_pad_amount)
                                                    : raw_count;
                            ctx_sp->set_prefill_handoff_streamed_prefix_count(trimmed);
                            fprintf(stderr,
                                    "*** PAD_TRIM_EXPORT: trimming streamed_prefix %u → %u (pad=%u)\n",
                                    raw_count, trimmed, _dyn_pad_amount);
                            fflush(stderr);
                        }
                        const uint32_t streamed_prefix = ctx_sp->get_prefill_handoff_streamed_prefix_count();
                        if (slot_batched && streamed_prefix > 0 && slot_batched->task) {
                            const int task_n = slot_batched->task->n_tokens();
                            const int keep_n = std::min<int>((int) streamed_prefix, task_n);
                            const int prompt_n_before = (int) slot_batched->prompt.n_tokens();
                            if (keep_n < prompt_n_before) {
                                const int n_suffix = prompt_n_before - keep_n;
                                slot_batched->prompt.tokens.keep_first((size_t) keep_n);
                                slot_batched->n_prompt_tokens_processed =
                                    std::max(0, slot_batched->n_prompt_tokens_processed - n_suffix);
                                fprintf(stderr, "*** SERVER: FA split handed %d suffix tokens to Mac, "
                                        "rolled back slot.prompt to %d tokens (local KV)\n",
                                        n_suffix, (int) slot_batched->prompt.n_tokens());
                                fflush(stderr);
                            } else if (keep_n > prompt_n_before) {
                                int extended = 0;
                                const auto & task_tokens = slot_batched->task->tokens;
                                for (int t = prompt_n_before; t < keep_n; ++t) {
                                    const llama_token tok = task_tokens[t];
                                    if (tok == LLAMA_TOKEN_NULL) {
                                        break;
                                    }
                                    slot_batched->prompt.tokens.push_back(tok);
                                    slot_batched->n_prompt_tokens_processed++;
                                    extended++;
                                }
                                if (extended > 0) {
                                    fprintf(stderr,
                                            "*** SERVER: streaming continued; extended slot.prompt by %d tokens to %d (local KV)\n",
                                            extended, (int) slot_batched->prompt.n_tokens());
                                    fflush(stderr);
                                }
                            }

                            const int remaining_prompt =
                                (int) slot_batched->task->n_tokens() - keep_n;
                            const int tail_limit = std::max(1, (int) llama_n_ubatch(ctx) - 1);
                            const bool force_local_tail_env = std::getenv("LLAMA_PREFILL_NO_REMOTE_TAIL_HANDOFF") != nullptr;
                            const bool force_local_tail = force_local_tail_env && remaining_prompt > 1;
                            // 2026-05-31 vision: if the un-streamed remainder [keep_n..task) contains
                            // an image placeholder (LLAMA_TOKEN_NULL), the Mac cannot prefill it (no
                            // pixel data). Keep streaming on RTX so update_slots re-enters and the image
                            // block (VISION-PS) encodes+streams it before any handoff. [VISION_ROUTE_V2]
                            bool remaining_has_image_a = false;
                            for (int t = keep_n; t < (int) slot_batched->task->n_tokens(); ++t) {
                                if (slot_batched->task->tokens[t] == LLAMA_TOKEN_NULL) { remaining_has_image_a = true; break; }
                            }
                            if (remaining_has_image_a) {
                                fprintf(stderr, "*** VISION-RET2-GUARD-A: remaining [%d..%d) has image; continuing RTX prefill\n",
                                        keep_n, (int) slot_batched->task->n_tokens()); fflush(stderr);
                            }
                            if (remaining_prompt > 0 && (force_local_tail || remaining_prompt > tail_limit || remaining_has_image_a)) {
                                if (seq_remapped && original_slot_seq > 0) {
                                    llama_memory_seq_cp(llama_get_memory(ctx), 0, original_slot_seq, -1, -1);
                                    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
                                    seq_remapped = false;
                                    fprintf(stderr,
                                            "*** SERVER: restored KV seq 0 -> %d before continuing local streaming\n",
                                            original_slot_seq);
                                    fflush(stderr);
                                }
                                fprintf(stderr,
                                    "*** SERVER: keeping %d-token streamed prefix on RTX and continuing local streaming for remaining %d prompt tokens (tail_limit=%d force_local=%d)\n",
                                    keep_n, remaining_prompt, tail_limit, (int) force_local_tail_env);
                                fflush(stderr);
                                i_next = resume_i_after_streamed_prefix(keep_n, "streamed-prefix");
                                n_batch = llama_n_batch(ctx);
                                continue;
                            }
                            fprintf(stderr,
                                "*** SERVER: handoff suffix prefill: keeping %d-token streamed prefix in local KV "
                                "and delegating the final %d prompt tokens to Mac local prefill after verification\n",
                                (int) slot_batched->prompt.n_tokens(),
                                remaining_prompt > 0 ? remaining_prompt : 0);
                            fflush(stderr);

                            slot_batched->state = SLOT_STATE_DONE_PROMPT;
                            slot_batched->n_decoded = 0;
                            slot_batched->i_batch = -1;
                            slot_batched->init_sampler();
                            ctx_sp->arm_decode_handoff_for_cached_prefill();
                            handoff_needed = true;
                            break;
                        }

                        // If the MLA FA split handed tail tokens to the Mac receiver,
                        // those tokens are NOT in local KV. Roll back the slot's prompt
                        // tracker so the next batch starts at the correct position.
                        const auto & tail_vec = ctx_sp->get_prefill_tail_tokens_vec();
                        if (!tail_vec.empty() && slot_batched) {
                            const int n_tail = (int)tail_vec.size();
                            const size_t keep_n = slot_batched->prompt.n_tokens() - n_tail;
                            slot_batched->prompt.tokens.keep_first(keep_n);
                            slot_batched->n_prompt_tokens_processed -= n_tail;
                            fprintf(stderr, "*** SERVER: FA split handed %d tail tokens to Mac, "
                                    "rolled back slot.prompt to %d tokens (local KV)\n",
                                    n_tail, (int)slot_batched->prompt.n_tokens());
                            fflush(stderr);

                            // Check remaining tokens. Only embed as tail if small enough
                            // for the Mac to prefill locally at reasonable speed (~500 tok/s).
                            // Larger remainders should go through another RTX streaming pass
                            // (~400 tok/s vs Mac's ~48 tok/s for large contexts).
                            // tail_vec already represents prompt tokens that the
                            // receiver will prefill locally. Don't append those
                            // same tokens again as "remaining" tail, or the Mac
                            // sees them twice and decode desyncs immediately.
                            const int accounted_prompt =
                                (int) slot_batched->prompt.n_tokens() + n_tail;
                            const int remaining_prompt =
                                (int) slot_batched->task->n_tokens() - accounted_prompt;
                            const int local_remaining_prompt =
                                (int) slot_batched->task->n_tokens() - (int) slot_batched->prompt.n_tokens();
                            const int tail_limit = std::max(1, (int) llama_n_ubatch(ctx) - 1);
                            // Same gate as the suffix block above: large remainders stay
                            // on RTX, but small final tails are delegated to Mac prefill.
                            const bool force_local_tail_env2 = std::getenv("LLAMA_PREFILL_NO_REMOTE_TAIL_HANDOFF") != nullptr;
                            const bool force_local_tail2 = force_local_tail_env2 && local_remaining_prompt > 1;
                            // 2026-05-31 vision: same image guard as the ret==2 path above — never
                            // delegate an un-streamed image span to the Mac. [VISION_ROUTE_V2]
                            bool remaining_has_image_b = false;
                            for (int t = (int) slot_batched->prompt.n_tokens(); t < (int) slot_batched->task->n_tokens(); ++t) {
                                if (slot_batched->task->tokens[t] == LLAMA_TOKEN_NULL) { remaining_has_image_b = true; break; }
                            }
                            if (remaining_has_image_b) {
                                fprintf(stderr, "*** VISION-RET2-GUARD-B: remaining [%d..%d) has image; continuing RTX prefill\n",
                                        (int) slot_batched->prompt.n_tokens(), (int) slot_batched->task->n_tokens()); fflush(stderr);
                            }
                            if (force_local_tail2 || remaining_prompt > tail_limit || remaining_has_image_b) {
                                // Too many remaining tokens for Mac local prefill.
                                // Continue batching on RTX immediately instead of
                                // falling out of the loop and waiting for another
                                // scheduler turn to resume prompt processing.
                                if (seq_remapped && original_slot_seq > 0) {
                                    llama_memory_seq_cp(llama_get_memory(ctx), 0, original_slot_seq, -1, -1);
                                    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
                                    seq_remapped = false;
                                    fprintf(stderr,
                                            "*** SERVER: restored KV seq 0 -> %d before continuing local streaming\n",
                                            original_slot_seq);
                                    fflush(stderr);
                                }
                                fprintf(stderr,
                                    "*** SERVER: tail handoff deferred; keeping %d-token streamed prefix on RTX and continuing local streaming for remaining %d prompt tokens (tail_limit=%d force_local=%d)\n",
                                    (int) slot_batched->prompt.n_tokens(), local_remaining_prompt, tail_limit, (int) force_local_tail_env2);
                                fflush(stderr);
                                i_next = resume_i_after_streamed_prefix((int) slot_batched->prompt.n_tokens(), "tail-deferred");
                                n_batch = llama_n_batch(ctx);
                                continue;
                            }
                            fprintf(stderr,
                                "*** SERVER: handoff suffix prefill: keeping %d-token streamed prefix in local KV "
                                "and delegating the final %d prompt tokens to Mac local prefill after verification\n",
                                (int) slot_batched->prompt.n_tokens(),
                                remaining_prompt > 0 ? remaining_prompt : 0);
                            fflush(stderr);

                            // slot.prompt continues to reflect only the prefix
                            // that exists in local KV. The Mac receives the full
                            // prompt payload and reconstructs the suffix locally.
                            slot_batched->state = SLOT_STATE_DONE_PROMPT;
                            slot_batched->n_decoded = 0;
                            slot_batched->i_batch = -1;
                            slot_batched->init_sampler();
                            ctx_sp->arm_decode_handoff_for_cached_prefill();
                            handoff_needed = true;
                            break;

                            // Replace the tail sidecar with the FULL prompt.
                            // Mac receiver auto-skips the imported prefix (by
                            // comparing sidecar length to imported KV cell count)
                            // and prefills only the new suffix tokens. This mirrors
                            // standard llama.cpp server cache_prompt semantics and
                            // removes any position-offset coupling between
                            // Windows n_p_eval and Mac pos_max.
                            const auto & task_tokens = slot_batched->task->tokens;
                            std::vector<llama_token> full_prompt;
                            full_prompt.reserve(slot_batched->task->n_tokens());
                            for (int t = 0; t < slot_batched->task->n_tokens(); t++) {
                                llama_token tok = task_tokens[t];
                                if (tok == LLAMA_TOKEN_NULL) break;
                                full_prompt.push_back(tok);
                            }
                            // Update slot bookkeeping for the NEW tokens (those
                            // beyond accounted_prompt, i.e. pass-2 remainder).
                            if (remaining_prompt > 0) {
                                for (int t = accounted_prompt; t < slot_batched->task->n_tokens(); t++) {
                                    llama_token tok = task_tokens[t];
                                    if (tok == LLAMA_TOKEN_NULL) break;
                                    slot_batched->prompt.tokens.push_back(tok);
                                    slot_batched->n_prompt_tokens_processed++;
                                }
                            }
                            if (!full_prompt.empty()) {
                                auto * ctx_tail = static_cast<llama_context *>(ctx);
                                ctx_tail->set_prefill_tail_tokens(full_prompt.data(), full_prompt.size());
                                fprintf(stderr,
                                    "*** SERVER: handoff tail sidecar: sending full prompt (%zu tokens) — "
                                    "Mac will skip imported prefix (~%d cells) and prefill the %d new tokens\n",
                                    full_prompt.size(), (int)slot_batched->prompt.n_tokens() - remaining_prompt,
                                    remaining_prompt > 0 ? remaining_prompt : 0);
                                fflush(stderr);
                            }
                            // Mark prompt as done and trigger handoff
                            slot_batched->state = SLOT_STATE_DONE_PROMPT;
                            slot_batched->n_decoded = 0;
                            slot_batched->i_batch = -1;
                            slot_batched->init_sampler();
                            handoff_needed = true;
                            break;
                        }

                        if (i + n_tokens < batch.n_tokens) {
                            fprintf(stderr, "*** SERVER: ret=2 for chunk [%d,%d) — continuing with remaining prompt tokens\n",
                                    i, i + n_tokens); fflush(stderr);
                            i_next = i + n_tokens;
                            n_batch = llama_n_batch(ctx);
                            continue;
                        }
                        fprintf(stderr, "*** SERVER: ret=2 — deferring handoff to after batch loop\n"); fflush(stderr);
                        handoff_needed = true;
                        break;
                    }

                    if (!err.empty()) {
                        SRV_ERR("%s i = %d, n_batch = %d, ret = %d\n", err.c_str(), i, n_batch, ret);

                        for (auto & slot : slots) {
                            if (slot.is_processing()) {
                                send_error(slot, err);
                                slot.release();

                                // note: it's complicated to keep track of how much of the current batch has been
                                //       processed before the error occurred, so we simply clear the entire context
                                slot.prompt_clear(false);
                            }
                        }

                        break;
                    }
                }

                // retry with half the batch size to try to find a free slot in the KV cache
                if (!try_clear_idle_slots()) {
                    n_batch /= 2;
                }

                SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, i = %d, n_batch = %d, ret = %d\n", i, n_batch, ret);

                continue; // continue loop of n_batch
            }

            // move the head of the batch forward with the number of tokens we just processed
            i_next = i + n_tokens;

            // on successful decode, restore the original batch size
            n_batch = llama_n_batch(ctx);

            // Phase 261/263 (2026-05-03): debug + handoff trigger when dispatch
            // was clipped to user-algo batch.
            if (_dispatch_limit < batch.n_tokens && i_next >= _dispatch_limit) {
                fprintf(stderr,
                        "*** PHASE_261_DEBUG: clip hit. i_next=%d limit=%d ret=%d slot_batched=%p slot_state=%d\n",
                        i_next, _dispatch_limit, ret,
                        (void*)slot_batched,
                        slot_batched ? (int)slot_batched->state : -1);
                fflush(stderr);
                if (ret == 0 && slot_batched) {
                    auto * ctx_sp = static_cast<llama_context *>(ctx);
                    fprintf(stderr,
                            "*** PHASE_261_HANDOFF: arming handoff (tail %u → Mac)\n",
                            _dyn_dispatch_tail);
                    fflush(stderr);
                    slot_batched->state = SLOT_STATE_DONE_PROMPT;
                    slot_batched->n_decoded = 0;
                    slot_batched->i_batch = -1;
                    slot_batched->init_sampler();
                    ctx_sp->arm_decode_handoff_for_cached_prefill();
                    handoff_needed = true;
                    break;
                }
            }

            // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
            for (auto & slot : slots) {
                if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                    std::vector<server_slot *> children;
                    for (auto & other : slots) {
                        if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                            children.push_back(&other);
                        }
                    }

                    // all children slots should already launched by launch_slots_with_parent_task()
                    // copy state to the child slots
                    for (auto & child : children) {
                        SLT_INF(slot, " - copying state to child %d\n", child->id);

                        GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                        slot.copy_state_to(*child);
                        child->state = SLOT_STATE_DONE_PROMPT;
                    }
                }
            }

            for (auto & slot : slots) {
                // optionally send prompt processing progress
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task->params.stream && slot.task->params.return_progress) {
                        send_partial_response(slot, {}, true);
                    }
                }

                if (slot.i_batch < (int) i || slot.i_batch >= (int) (i + n_tokens)) {
                    continue; // continue loop of slots
                }

                if (slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                        // prompt evaluated for embedding
                        send_embedding(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                        send_rerank(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    GGML_ASSERT(slot.task->need_sampling());

                    // prefill_only / n_predict=0: prompt done, no generation needed.
                    // Report timings and release immediately — do NOT sample (logits
                    // may be unavailable when weight fetcher is active).
                    if (!slot.has_budget(params_base)) {
                        slot.n_decoded = 0;
                        slot.t_start_generation = ggml_time_us();
                        slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                        metrics.on_prompt_eval(slot);
                        slot.print_timings();
                        send_final_response(slot);
                        metrics.on_prediction(slot);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    // [2026-07-27 FRESH-STREAMING-PREFILL HANDOFF GUARD] If a decode handoff was armed
                    // during streaming prefill (set in decode_streaming, llama-context.cpp:6993), do NOT
                    // fall through to the local sampler. For M3/Kimi WF streaming the output-head weights
                    // are sentinel data so LLAMA_SERVER_SKIP_LOGITS_MARK=1 left batch.logits[n-1] unmarked;
                    // and for split_thunderbolt Mac transport configured via kv_port, llama_decode returns
                    // 0 (not 2) because the return-2 gate (llama-context.cpp:19120-19123) is narrower than
                    // the streaming handoff check (llama-prefill-stream.cpp:3228-3235). Without this guard a
                    // fresh single-server-batch >3072 request reaches common_sampler_sample() with
                    // i_batch=n-1 whose logits were never computed -> get_logits_ith assert (sampling.cpp:149).
                    // Defer to the deferred-handoff block (below, guarded by handoff_needed) exactly like the
                    // CACHED-PREFIX small-delta path does. Only fires when a handoff is pending (M3/Kimi fresh
                    // WF streaming); GLM (non-WF, logits marked), DSV4 (satisfies the return-2 gate), and
                    // stream_local (no handoff pending) are byte-unchanged. Root-cause: sol-xhigh diagnosis in
                    // .planning/debug/m3-fault-diagnosis.md. init_sampler() already ran at DONE_PROMPT above.
                    if (((llama_context *) ctx)->has_decode_handoff_pending()) {
                        slot.i_batch   = -1;
                        slot.n_decoded = 0;
                        handoff_needed = true;
                        continue; // never sample the WF sentinel head; the Mac relay decodes
                    }

                    // prompt evaluated for next-token prediction

                    // Phase 268 COMPLETION — decode pad-trim (LLAMA_PREFILL_KV_PAD_TRIM=1, default OFF).
                    // The streaming pad-and-trim (Phase 267) appends _dyn_pad_amount BOS pad tokens to
                    // 256-align the prefill (e.g. real 9860 -> 9984), leaving the KV memory with pad cells
                    // past the real prompt (pos_max overshoots slot.prompt.n_tokens()). Phase 268 trims only
                    // the EXPORT count on the Mac-handoff (ret==2) path; that branch break's out before here.
                    // For LOCAL (wir_local) decode we reach this point with the UNtrimmed KV, so the next
                    // generation token's position is non-consecutive and balloc->init rejects it
                    // (Y=real != X+1=padded). Drop the pad cells [real_end, end) so local decode continues.
                    {
                        auto * ctx_pt = (llama_context *) ctx;
                        // Default ON: the streaming pad-and-trim (Phase 267) ALWAYS appends pad tokens to
                        // 256-align the prefill, so local decode ALWAYS needs the KV-memory trim to keep
                        // positions consecutive. Mac-handoff path breaks out at ret==2 before reaching here,
                        // so this is local-decode-only and Mac-safe. Opt out with LLAMA_PREFILL_KV_PAD_TRIM=0.
                        static const bool kv_pad_trim = []() {
                            const char * e = std::getenv("LLAMA_PREFILL_KV_PAD_TRIM");
                            return !(e && e[0] == '0');
                        }();
                        if (kv_pad_trim && ctx_pt && ctx_pt->is_prefill_streaming()) {
                            const llama_pos real_end = (llama_pos) slot.prompt.n_tokens();
                            const llama_pos pm       = llama_memory_seq_pos_max(llama_get_memory(ctx), slot.id);
                            if (real_end > 0 && pm >= real_end) {
                                SLT_INF(slot, "PAD_TRIM_KV: seq %d pos_max=%d > real_end=%d — dropping %d pad cell(s) for local decode\n",
                                        slot.id, (int) pm, (int) real_end, (int)(pm - real_end + 1));
                                llama_memory_seq_rm(llama_get_memory(ctx), slot.id, real_end, -1);
                            }
                        }
                    }

                    slot.state = SLOT_STATE_GENERATING;

                    if (slot.can_speculate()) {
                        common_speculative_begin(slot.spec, slot.prompt.tokens.get_text_tokens());
                    }
                } else if (slot.state != SLOT_STATE_GENERATING) {
                    continue; // continue loop of slots
                }

                if (slot.i_batch_dft.size() > 0) {
                    continue; // sample using speculative decoding
                }

                const int tok_idx = slot.i_batch - i;

                llama_token id = common_sampler_sample(slot.smpl.get(), ctx, tok_idx);

                slot.i_batch = -1;

                common_sampler_accept(slot.smpl.get(), id, true);

                // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
                const int64_t t_current = ggml_time_us();

                slot.n_decoded += 1;

                if (slot.n_decoded == 1) {
                    slot.t_start_generation = t_current;
                    slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                    metrics.on_prompt_eval(slot);
                }

                slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                completion_token_output result;
                result.tok          = id;
                result.text_to_send = common_token_to_piece(ctx, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

                if (slot.task->params.sampling.n_probs > 0) {
                    populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
                }

                if (!process_token(result, slot)) {
                    // release slot because of stop condition
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                    slot.release();

                    continue;
                }
            }

            // speculative decoding - main model sample and accept
            for (auto & slot : slots) {
                if (slot.state != SLOT_STATE_GENERATING || slot.i_batch_dft.empty()) {
                    continue;
                }

                const size_t n_draft = slot.drafted.size();

                // the accepted tokens from the speculation
                const auto ids = slot.draft_dists.empty() ?
                    common_sampler_sample_and_accept_n(slot.smpl.get(), ctx, slot.i_batch_dft, slot.drafted) :
                    common_sampler_sample_and_accept_n(slot.smpl.get(), ctx, slot.i_batch_dft, slot.drafted, slot.draft_dists);
                slot.i_batch_dft.clear();
                slot.drafted.clear();
                slot.draft_dists.clear();

                const int64_t t_current = ggml_time_us();

                slot.n_decoded += ids.size();

                slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                // update how many tokens out of those tested were accepted
                slot.n_draft_accepted += ids.size() - 1;

                // inform the speculative decoding about the number of accepted tokens
                if (slot.spec) {
                    common_speculative_accept(slot.spec, ids.size() - 1);
                }

                // rollback to the state before sampling the draft tokens
                slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);

                // add accepted tokens to the prompt
                slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});
                slot.sampled = ids.back(); // last accepted token

                llama_memory_seq_rm(llama_get_memory(ctx), slot.id, slot.prompt.n_tokens(), -1);

                for (size_t i = 0; i < ids.size(); ++i) {
                    completion_token_output result;

                    result.tok          = ids[i];
                    result.text_to_send = common_token_to_piece(ctx, result.tok, accept_special_token(slot, result.tok));
                    result.prob         = 1.0f; // set later

                    // TODO: set result.probs

                    if (!process_token(result, slot)) {
                        slot.print_timings();
                        send_final_response(slot);
                        metrics.on_prediction(slot);
                        slot.release();

                        break;
                    }
                }

                SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) ids.size() - 1, (int) n_draft, slot.prompt.n_tokens());
            }
        }

        // Phase 132: restore original n_ubatch / n_batch after the dynamic-ub override.
        if (_dyn_ub_applied) {
            auto * ctx_dub = static_cast<llama_context *>(ctx);
            ctx_dub->set_n_ubatch(_saved_n_ubatch);
            ctx_dub->set_n_batch(_saved_n_batch);
            ctx_dub->set_sched_need_reserve(true);
            fprintf(stderr, "*** PREFILL_DYNAMIC_UB: restored n_ubatch=%u n_batch=%u\n",
                    _saved_n_ubatch, _saved_n_batch);
            fflush(stderr);
        }

        // Deferred handoff: process outside the batch loop to avoid result queue conflicts.
        // The handoff exports KV to Mac, waits for decode, and relays tokens back.
        // With streaming enabled, tokens are sent as SSE partial responses in real-time.
        do_handoff:
        if (handoff_needed) {
            bool prompt_still_pending = false;
            for (auto & slot : slots) {
                if (slot.is_processing() && slot.state == SLOT_STATE_PROCESSING_PROMPT) {
                    prompt_still_pending = true;
                    break;
                }
            }

            if (prompt_still_pending) {
                fprintf(stderr, "*** SERVER: deferred handoff postponed - prompt processing is not complete yet\n");
                fflush(stderr);
            } else {
            fprintf(stderr, "*** SERVER: deferred handoff — triggering now\n"); fflush(stderr);

            // Find the active slot for the handoff — needed for streaming SSE
            handoff_slot = nullptr;
            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    handoff_slot = &slot;
                    slot.t_start_generation = ggml_time_us();
                    slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                    metrics.on_prompt_eval(slot);
                    break;
                }
            }

            if (handoff_slot && handoff_slot->task) {
                auto * ctx_ho = static_cast<llama_context *>(ctx);
                ctx->set_prefill_sampling_profile(
                    make_handoff_sampling_profile(handoff_slot->task->params.sampling));
                ctx->set_prefill_requested_n_predict(handoff_slot->task->params.n_predict);
                ctx_ho->set_handoff_slot_id(handoff_slot->id);
                ctx_ho->set_prefill_handoff_capture_seq_id(handoff_slot->id);

                // Pass stop words (antiprompts + chat template stops) to Mac
                ctx_ho->set_decode_stop_words(handoff_slot->task->params.antiprompt);

                // 2026-05-31 vision EXPORT FIX: image requests carry LLAMA_TOKEN_NULL
                // placeholders at the image cells. Breaking here truncated the advertised
                // prompt to the pre-image text (19 tok), so the Mac re-prefilled the image
                // positions locally (no pixels -> garbage). The image KV is already
                // stream-prefilled into the exported blob, so emit the media-pad id for
                // image cells (preserve the full count); streamed_prefix below already
                // advertises the RTX-prefilled cell count so the Mac imports the image KV
                // and re-prefills only the genuine text suffix after the image.
                const bool ho_has_media = handoff_slot->task->tokens.has_media();
                std::vector<llama_token> full_prompt;
                full_prompt.reserve(handoff_slot->task->n_tokens());
                for (int t = 0; t < handoff_slot->task->n_tokens(); ++t) {
                    llama_token tok = handoff_slot->task->tokens[t];
                    if (tok == LLAMA_TOKEN_NULL) {
                        if (ho_has_media) {
                            tok = 163606; // Kimi-VL <|media_pad|>
                        } else {
                            break;
                        }
                    }
                    full_prompt.push_back(tok);
                }

                // Phase 81: when local-tail continuation extended the streamed prefix
                // past the slot.prompt rollback (LLAMA_PREFILL_NO_REMOTE_TAIL_HANDOFF=1),
                // use the actual streamed count from the context, not the rolled-back
                // slot.prompt.n_tokens (which would understate kv_tokens and force Mac to
                // re-prefill the tail cross-stack → garbage).
                const uint32_t streamed_in_ctx = ctx_ho->get_prefill_handoff_streamed_prefix_count();
                const uint32_t slot_prompt_n   = (uint32_t) handoff_slot->prompt.n_tokens();
                const uint32_t streamed_prefix_for_handoff = std::max(streamed_in_ctx, slot_prompt_n);
                if (ho_has_media) {
                    fprintf(stderr, "*** VISION-HANDOFF(deferred): full_prompt=%zu streamed_prefix=%u slot_prompt_n=%u [VISION_ROUTE_V2]\n",
                            full_prompt.size(), streamed_prefix_for_handoff, slot_prompt_n);
                    fflush(stderr);
                }
                ctx_ho->set_prefill_handoff_prompt(
                    full_prompt.empty() ? nullptr : full_prompt.data(),
                    full_prompt.size(),
                    handoff_slot->task->tokens.detokenize(ctx, true),
                    streamed_prefix_for_handoff);
                // A long splice seed correctly took RTX streaming prefill instead of
                // PROMPT-DIRECT. Preserve its exact logical prompt so the returned
                // Mac retained-count/hash can arm the same continuation witness used
                // by a short prompt-direct seed.
                if (mac_splice_enabled && handoff_slot->task->splice_route_kind ==
                        server_task::SPLICE_ROUTE_FULL_SEED) {
                    handoff_slot->mac_splice.logical_prompt_sent = full_prompt;
                }
            }

            handoff_cancelled = false;
            relay_token_buffer.clear();
            handoff_slot->stop = STOP_TYPE_NONE;  // reset before handoff
            if (mac_retain_handle_enabled) {
                static_cast<llama_context *>(ctx)->clear_mac_retained_handle();
            }
            bool handoff_ok = llama_decode_trigger_handoff(ctx);
            static_cast<llama_context *>(ctx)->set_mac_retained_reuse_count(0);
            static_cast<llama_context *>(ctx)->set_mac_retained_reuse_hash(0);

            if (handoff_cancelled) {
                cleanup_cancelled_handoff(true, true);
            } else if (!handoff_ok || relay_token_buffer.empty()) {
                // NAME THE ACTUAL FAILURE. This reported "KV export may have failed — check disk
                // space" for EVERY cause, and the relay already knows better: llama_context records
                // the relay's own status, which distinguishes "failed to connect to token stream at
                // HOST:PORT", "token stream handshake timeout", "retained-prefix-miss", a read
                // timeout and "no token stream endpoint configured".
                //
                // Measured 2026-08-31 on a preset-3 turn: the prefill completed in 28 s, every KV
                // chunk was sent, and the RTX then spent 240 s exhausting 120 connect retries to the
                // Mac's token relay on <peer-ip>:9101 — after which it blamed disk space, on a
                // machine with 60 GB free. A suggested cause that is wrong is worse than none: it
                // sent the reader to the wrong machine AND the wrong subsystem, and the log line
                // holding the real answer was 20 lines above it.
                const std::string relay_status =
                    static_cast<llama_context *>(ctx)->get_last_relay_status();
                const std::string reason_str =
                    !relay_status.empty()
                        ? "Decode handoff failed: " + relay_status
                        : (relay_token_buffer.empty()
                            // No status AND no tokens is the only state where the export itself is
                            // still a live suspect, so that guess survives only here.
                            ? std::string("Decode handoff returned no tokens and the relay reported "
                                          "no status (the KV export may have failed)")
                            : std::string("Decode handoff to remote node failed"));
                const char * reason = reason_str.c_str();
                fprintf(stderr, "*** SERVER: decode handoff FAILED (%s)\n", reason); fflush(stderr);
                handoff_slot = nullptr;
                for (auto & slot : slots) {
                    if (slot.is_processing()) {
                        send_error(slot, reason);
                        if (mac_splice_enabled) {
                            slot.mac_splice.clear();
                        }
                        slot.release();
                        invalidate_route_reuse(&slot, reason, true);
                    }
                }
                relay_token_buffer.clear();
            } else {
                fprintf(stderr, "*** SERVER: decode handoff SUCCESS (%zu buffered tokens)\n",
                        relay_token_buffer.size()); fflush(stderr);

                // Get stop reason from Mac decode relay result
                int n_content_tokens = 0;
                for (const auto & rt : relay_token_buffer) {
                    if (!rt.done) n_content_tokens++;
                }
                auto * ctx_ho = static_cast<llama_context *>(ctx);
                const int32_t mac_stop = ctx_ho->get_last_relay_stop_reason();
                stop_type handoff_stop_type = STOP_TYPE_EOS;
                if (mac_stop == LLAMA_STOP_WORD) {
                    handoff_stop_type = STOP_TYPE_WORD;
                    if (handoff_slot) {
                        handoff_slot->stopping_word = ctx_ho->get_last_relay_stopping_word();
                    }
                } else if (mac_stop == LLAMA_STOP_LIMIT) {
                    handoff_stop_type = STOP_TYPE_LIMIT;
                }

                for (auto & slot : slots) {
                    if (slot.is_processing()) {
                        slot.n_decoded = (int32_t)n_content_tokens;
                        slot.relay_decode_elapsed_ms = ctx_ho->get_last_relay_decode_elapsed_ms();
                        slot.relay_decode_intervals = ctx_ho->get_last_relay_decode_intervals();
                        slot.relay_decode_tok_s = ctx_ho->get_last_relay_decode_tok_s();
                        slot.relay_decode_tok_s_early = ctx_ho->get_last_relay_decode_tok_s_early();
                        slot.relay_decode_tok_s_middle = ctx_ho->get_last_relay_decode_tok_s_middle();
                        slot.relay_decode_tok_s_late = ctx_ho->get_last_relay_decode_tok_s_late();

                        if (slot.task && slot.task->params.stream) {
                            slot.stop = handoff_stop_type;
                            send_final_response(slot);
                        } else {
                            std::string full_text;
                            for (auto & rt : relay_token_buffer) {
                                if (rt.done) break;
                                full_text += rt.text;
                            }
                            auto res = std::make_unique<server_task_result_cmpl_final>();
                            res->id    = slot.task->id;
                            res->index = slot.task->index;
                            res->content = full_text;
                            res->n_decoded           = slot.n_decoded;
                            res->n_prompt_tokens     = slot.task->n_tokens();
                            res->verbose             = slot.task->params.verbose;
                            res->res_type            = slot.task->params.res_type;
                            res->oaicompat_model     = slot.task->params.oaicompat_model;
                            res->oaicompat_cmpl_id   = slot.task->params.oaicompat_cmpl_id;
                            slot.stop = handoff_stop_type;
                            res->stop = slot.stop;
                            res->timings = slot.get_timings();
                            add_relay_decode_timings(res->timings);
                            queue_results.send(std::move(res));
                        }

                        metrics.on_prediction(slot);

                        // Don't modify slot.prompt — it reflects what's in local KV
                        // (the FA prefix only). The tail and assistant tokens are
                        // NOT in local KV. Next turn, the server's prompt cache
                        // will match the FA prefix, then re-prefill everything
                        // after it (tail + assistant + new user message).
                        commit_mac_splice_state(slot, true);
                        record_mac_retained_handle(slot, true);
                        slot.release();
                        if (route_reuse_enabled && !commit_route_reuse_record(slot)) {
                            // FIX (B) 2026-07-18: clear_prompt=false. The comment above says slot.prompt reflects
                            // the FA prefix in local KV and next turn's prompt cache should MATCH it — but
                            // clear_prompt=true WIPED it, forcing turn-2 to full re-prefill (reused_n=0). Keep the
                            // slot tokens so turn-2's common-prefix match yields reused_n>0 -> STREAMING CONTINUATION
                            // (delta-only prefill = O(new tokens)) instead of STREAMING PREFILL. The DSV4 route-reuse
                            // record is still invalidated (ctx state), we just don't destroy the prompt-cache prefix.
                            invalidate_route_reuse(&slot,
                                    "remote handoff completed without a complete export/publish/migration record",
                                    false);
                        }
                    }
                }
                relay_token_buffer.clear();
            }

            handoff_cancelled = false;

            // Restore KV seq_id mapping: streaming prefill remapped slot.id → 0.
            // Copy KV from seq 0 back to the slot's seq_id so prompt cache works.
            if (seq_remapped && original_slot_seq > 0) {
                llama_memory_seq_cp(llama_get_memory(ctx), 0, original_slot_seq, -1, -1);
                llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
                fprintf(stderr, "*** SERVER: restored KV seq 0 → %d after handoff\n",
                        original_slot_seq); fflush(stderr);
            }

            handoff_slot = nullptr;
            }
        }

        SRV_DBG("%s", "run slots completed\n");
    }

    int get_slot_n_ctx() {
        return slots.back().n_ctx;
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(const common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx, eos_id, true) : "";

    return server_context_meta {
        /* build_info             */ build_info,
        /* model_name             */ impl->model_name,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* json_webui_settings    */ impl->json_webui_settings,
        /* slot_n_ctx             */ impl->get_slot_n_ctx(),
        /* pooling_type           */ llama_pooling_type(impl->ctx),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model),
        /* model_n_params         */ llama_model_n_params(impl->model),
        /* model_size             */ llama_model_size(impl->model),
    };
}



// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_http_res {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};



//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            // v1.9 small-image-forward: forward the bare user question text for the receiver's mlx_vlm generate().
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files,
                                                 // v1.10 multi-turn: prefer the structured vlm_conversation JSON
                                                 // (full history) over the bare latest question.
                                                 data.value("vlm_conversation", data.value("raw_user_text", std::string()))));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_task::params_from_json_cmpl(
                    ctx_server.vocab,
                    params,
                    meta->slot_n_ctx,
                    data);
            task.id_slot = json_value(data, "id_slot", -1);

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // Gate-off no-op: parser emits none of these internal keys unless
            // LLAMA_DSV4_SPLICE_GENERATED_IDS=1 and both probes/shape checks pass.
            if (data.value("__splice_shape_eligible", false)) {
                task.splice_shape_eligible       = true;
                task.splice_seed_eligible        = data.value("__splice_seed_eligible", false);
                task.splice_request_lineage      = data.value("__splice_request_lineage", std::string());
                task.splice_request_prefix       = data.value("__splice_request_prefix", std::string());
                task.splice_tail_text            = data.value("__splice_tail_text", std::string());
                task.splice_template_fingerprint = data.value("__splice_template_fingerprint", std::string());
                task.splice_template_family      = data.value("__splice_template_family", std::string());

                if (!task.splice_seed_eligible && !task.splice_tail_text.empty()) {
                    task.splice_tail_tokens = tokenize_mixed(
                            ctx_server.vocab, json(task.splice_tail_text), false, true);
                    bool exact_tail = !task.splice_tail_tokens.empty() &&
                            task.splice_tail_tokens.size() <= (size_t) task.n_tokens();
                    const size_t tail_at = exact_tail
                            ? (size_t) task.n_tokens() - task.splice_tail_tokens.size() : 0;
                    for (size_t t = 0; exact_tail && t < task.splice_tail_tokens.size(); ++t) {
                        exact_tail = task.tokens[tail_at + t] == task.splice_tail_tokens[t] &&
                                task.splice_tail_tokens[t] != LLAMA_TOKEN_NULL;
                    }
                    const std::string first_piece = exact_tail
                            ? common_token_to_piece(ctx_server.vocab, task.splice_tail_tokens.front(), true)
                            : std::string();
                    const bool boundary_token_ok = task.splice_template_family == "glm"
                            ? first_piece == "<|user|>"
                            : first_piece == "<｜end▁of▁sentence｜>";
                    const bool scaffold_ok = task.splice_template_family == "glm"
                            ? task.splice_tail_text.find("<|assistant|>") != std::string::npos
                            : task.splice_tail_text.find("<｜Assistant｜>") != std::string::npos;
                    if (!exact_tail || !boundary_token_ok || !scaffold_ok) {
                        task.splice_shape_eligible = false;
                        task.splice_tail_tokens.clear();
                    }
                }
            }

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->next = [res_this = res.get(), res_type, &req](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            try {
                if (req.should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                auto result = rd.next(req.should_stop);
                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(req.should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        };
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
        json all_metrics_def = json {
            {"counter", {{
                    {"name",  "prompt_tokens_total"},
                    {"help",  "Number of prompt tokens processed."},
                    {"value",  (uint64_t) res_task->n_prompt_tokens_processed_total}
            }, {
                    {"name",  "prompt_seconds_total"},
                    {"help",  "Prompt process time"},
                    {"value",  (uint64_t) res_task->t_prompt_processing_total / 1.e3}
            }, {
                    {"name",  "tokens_predicted_total"},
                    {"help",  "Number of generation tokens processed."},
                    {"value",  (uint64_t) res_task->n_tokens_predicted_total}
            }, {
                    {"name",  "tokens_predicted_seconds_total"},
                    {"help",  "Predict process time"},
                    {"value",  (uint64_t) res_task->t_tokens_generation_total / 1.e3}
            }, {
                    {"name",  "n_decode_total"},
                    {"help",  "Total number of llama_decode() calls"},
                    {"value",  res_task->n_decode_total}
            }, {
                    {"name",  "n_tokens_max"},
                    {"help",  "Largest observed n_tokens."},
                    {"value",  res_task->n_tokens_max}
            }, {
                    {"name",  "n_busy_slots_per_decode"},
                    {"help",  "Average number of busy slots per llama_decode() call"},
                    {"value",  (float) res_task->n_busy_slots_total / std::max((float) res_task->n_decode_total, 1.f)}
            }}},
            {"gauge", {{
                    {"name",  "prompt_tokens_seconds"},
                    {"help",  "Average prompt throughput in tokens/s."},
                    {"value",  res_task->n_prompt_tokens_processed ? 1.e3 / res_task->t_prompt_processing * res_task->n_prompt_tokens_processed : 0.}
            },{
                    {"name",  "predicted_tokens_seconds"},
                    {"help",  "Average generation throughput in tokens/s."},
                    {"value",  res_task->n_tokens_predicted ? 1.e3 / res_task->t_tokens_generation * res_task->n_tokens_predicted : 0.}
            },{
                    {"name",  "requests_processing"},
                    {"help",  "Number of requests processing."},
                    {"value",  (uint64_t) res_task->n_processing_slots}
            },{
                    {"name",  "requests_deferred"},
                    {"help",  "Number of requests deferred."},
                    {"value",  (uint64_t) res_task->n_tasks_deferred}
            }}}
        };

        std::stringstream prometheus;

        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();

            for (const auto & metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");

                auto value = json_value(metric_def, "value", 0.);
                prometheus << "# HELP llamacpp:" << name << " " << help  << "\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\n"
                            << "llamacpp:"        << name << " " << value << "\n";
            }
        }

        res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->t_start);
        res->content_type = "text/plain; version=0.0.4";
        res->status = 200;
        res->data = prometheus.str();
        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto * res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->slots_data);
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        task_params tparams;
        tparams.sampling = params.sampling;
        json default_generation_settings_for_props = json {
            { "params", tparams.to_json(true) },
            { "n_ctx",  meta->slot_n_ctx },
        };

        std::string tmpl_default = common_chat_templates_source(meta->chat_params.tmpls.get(), "");
        std::string tmpl_tools   = common_chat_templates_source(meta->chat_params.tmpls.get(), "tool_use");

        json props = {
            { "default_generation_settings", default_generation_settings_for_props },
            { "total_slots",                 params.n_parallel },
            { "model_alias",                 meta->model_name },
            { "model_path",                  meta->model_path },
            { "modalities",                  json {
                {"vision", meta->has_inp_image},
                {"audio",  meta->has_inp_audio},
            } },
            { "endpoint_slots",              params.endpoint_slots },
            { "endpoint_props",              params.endpoint_props },
            { "endpoint_metrics",            params.endpoint_metrics },
            { "webui",                       params.webui },
            { "webui_settings",              meta->json_webui_settings },
            { "chat_template",               tmpl_default },
            { "chat_template_caps",          meta->chat_template_caps },
            { "bos_token",                   meta->bos_token_str },
            { "eos_token",                   meta->eos_token_str },
            { "build_info",                  meta->build_info },
            { "is_sleeping",                 queue_tasks.is_sleeping() },
        };
        if (params.use_jinja) {
            if (!tmpl_tools.empty()) {
                props["chat_template_tool_use"] = tmpl_tools;
            }
        }
        res->ok(props);
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    // Phase 327 (2026-05-05): /v1/seq_cp endpoint. Calls llama_memory_seq_cp
    // to clone a source sequence's KV cache into a destination sequence at
    // zero GPU memory cost (under kv_unified=1, n_stream=1: pure metadata
    // op adding dst seq_id to existing cells; under multi-stream mode it
    // copies cells across streams).
    //
    // Body: {"src": int, "dst": int, "p0": int (-1 = 0), "p1": int (-1 = end)}
    // Returns: {"success": true} or error
    this->post_seq_cp = [this](const server_http_req & req) {
        auto res = create_response();
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res->error(format_error_response("invalid JSON body", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!body.contains("src") || !body.contains("dst")) {
            res->error(format_error_response("body must contain 'src' and 'dst' integer seq_ids", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        const llama_seq_id src = (llama_seq_id) body.value("src", -1);
        const llama_seq_id dst = (llama_seq_id) body.value("dst", -1);
        const llama_pos p0 = (llama_pos) body.value("p0", -1);
        const llama_pos p1 = (llama_pos) body.value("p1", -1);
        if (src < 0 || dst < 0) {
            res->error(format_error_response("src and dst must be non-negative", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        llama_memory_seq_cp(llama_get_memory(ctx_server.get_ctx_unsafe()), src, dst, p0, p1);
        res->ok({{"success", true}, {"src", src}, {"dst", dst}, {"p0", p0}, {"p1", p1}});
        return res;
    };

    // Phase G (2026-05-06): KVBlob v2 export endpoint.
    // Body JSON: {"request_ids":[3,7,19], "p0":-1, "p1":-1}
    // Returns: KVBlob v2 byte stream (LAYOUT_OPAQUE — payload is per-seq
    // llama_state_seq blobs).  See <runtime-path>
    //
    // The KV reads run on the main loop via SERVER_TASK_TYPE_HANDOFF_EXPORT;
    // direct call from this thread crashes in this fork's streaming-prefill
    // llama-context.  Pattern mirrors handle_slots_save (line ~6160).
    this->post_handoff_export_kvblob_v2 = [this](const server_http_req & req) {
        auto res = create_response();
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res->error(format_error_response("invalid JSON body", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!body.contains("request_ids") || !body["request_ids"].is_array()) {
            res->error(format_error_response("body must contain 'request_ids' array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<int32_t> request_ids;
        request_ids.reserve(body["request_ids"].size());
        for (const auto & v : body["request_ids"]) {
            if (!v.is_number_integer()) {
                res->error(format_error_response("request_ids must be integers", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            request_ids.push_back(v.get<int>());
        }

        // Post a SERVER_TASK_TYPE_HANDOFF_EXPORT task to the main loop so
        // KV reads happen on the thread that owns the cache.  Mirrors
        // handle_slots_save (line ~6160).
        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_HANDOFF_EXPORT);
            task.id = rd.get_new_id();
            task.handoff_action.request_ids = std::move(request_ids);
            rd.post_task(std::move(task));
        }
        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        auto * exp = dynamic_cast<server_task_result_handoff_export*>(result.get());
        if (exp == nullptr) {
            res->error(format_error_response("internal: bad result type from HANDOFF_EXPORT", ERROR_TYPE_SERVER));
            return res;
        }
        if (exp->status_code != 200) {
            res->status = exp->status_code;
            res->data = safe_json_to_str({{"error", exp->error}});
            return res;
        }
        res->status = 200;
        res->content_type = "application/octet-stream";
        res->data = std::move(exp->kvblob_bytes);
        return res;
    };

    // Phase G (2026-05-06): KVBlob v2 import endpoint.
    // Body: raw KVBlob v2 bytes
    // Header X-KVBlob-SeqId-Remap: optional "src=dst,src=dst"
    // Returns: {"ok":true,"n_seqs":N}, or HTTP 409 on remap conflict.
    //
    // Like the export endpoint, KV writes run on the main loop via
    // SERVER_TASK_TYPE_HANDOFF_IMPORT.
    this->post_handoff_import_kvblob_v2 = [this](const server_http_req & req) {
        auto res = create_response();

        // Parse remap header (case-insensitive lookup).
        std::map<llama_seq_id, llama_seq_id> remap;
        for (const auto & kv : req.headers) {
            std::string k_lower = kv.first;
            std::transform(k_lower.begin(), k_lower.end(), k_lower.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (k_lower == "x-kvblob-seqid-remap") {
                try {
                    remap = kvblob_v2_parse_seq_id_remap_header(kv.second);
                } catch (const std::exception & e) {
                    res->error(format_error_response(
                        std::string("bad X-KVBlob-SeqId-Remap header: ") + e.what(),
                        ERROR_TYPE_INVALID_REQUEST));
                    return res;
                }
                break;
            }
        }

        // Cheap up-front parse so obviously-bad blobs return 400 without
        // burning a task slot.
        kvblob_v2_header h{};
        std::vector<kvblob_v2_seq_desc> tmp_descs;
        std::string parse_err;
        if (!kvblob_v2_parse(req.body, h, tmp_descs, parse_err)) {
            res->error(format_error_response(
                "KVBlob v2 parse failed: " + parse_err, ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_HANDOFF_IMPORT);
            task.id = rd.get_new_id();
            task.handoff_action.handoff_blob = req.body;  // copy; main loop reads
            // Convert llama_seq_id -> int32_t for the task struct.
            for (const auto & kv : remap) {
                task.handoff_action.handoff_remap[static_cast<int32_t>(kv.first)]
                    = static_cast<int32_t>(kv.second);
            }
            rd.post_task(std::move(task));
        }
        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        auto * imp = dynamic_cast<server_task_result_handoff_import*>(result.get());
        if (imp == nullptr) {
            res->error(format_error_response("internal: bad result type from HANDOFF_IMPORT", ERROR_TYPE_SERVER));
            return res;
        }
        res->status = imp->status_code;
        if (imp->status_code == 200) {
            res->data = safe_json_to_str({{"ok", true}, {"n_seqs", imp->n_seqs}});
        } else {
            res->data = safe_json_to_str({{"error", imp->error}});
        }
        return res;
    };

    this->get_api_show = [this](const server_http_req &) {
        auto res = create_response();
        std::string tmpl_default = common_chat_templates_source(meta->chat_params.tmpls.get(), "");
        json data = {
            {
                "model_info", {
                    { "llama.context_length", meta->slot_n_ctx },
                }
            },
            {"modelfile", ""},
            {"parameters", ""},
            {"template", tmpl_default},
            {"details", {
                {"parent_model", ""},
                {"format", "gguf"},
                {"family", ""},
                {"families", {""}},
                {"parameter_size", ""},
                {"quantization_level", ""}
            }},
            {"model_info", ""},
            {"capabilities", meta->has_mtmd ? json({"completion","multimodal"}) : json({"completion"})}
        };

        res->ok(data);
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_text_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    auto parse_completion_body = [this](json body, std::vector<raw_buffer> & files) {
        const bool raw_prompt = json_value(body, "raw_prompt", false);
        const bool can_template = meta && meta->chat_params.tmpls && body.contains("prompt") && body.at("prompt").is_string();

        if (!raw_prompt && can_template) {
            json chat_body = body;
            chat_body["messages"] = json::array({
                json{
                    {"role", "user"},
                    {"content", body.at("prompt")}
                }
            });
            chat_body.erase("prompt");
            chat_body.erase("raw_prompt");

            SRV_INF("%s", "completion request normalized through chat template\n");
            return oaicompat_chat_params_parse(chat_body, meta->chat_params, files);
        }

        return oaicompat_completion_params_parse(body);
    };

    this->post_completions = [this, parse_completion_body](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        json body = json::parse(req.body);
        json body_parsed = parse_completion_body(body, files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this, parse_completion_body](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        json body = json::parse(req.body);
        json body_parsed = parse_completion_body(body, files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = convert_responses_to_chatcmpl(json::parse(req.body));
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = convert_anthropic_to_oai(json::parse(req.body));
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = convert_anthropic_to_oai(json::parse(req.body));
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);

        json prompt = body_parsed.at("prompt");
        llama_tokens tokens = tokenize_mixed(ctx_server.vocab, prompt, true, true);
        res->ok({{"input_tokens", static_cast<int>(tokens.size())}});
        return res;
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        json models = {
            {"models", {
                {
                    {"name",  meta->model_name},
                    {"model", meta->model_name},
                    {"modified_at", ""},
                    {"size", ""},
                    {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                    {"type", "model"},
                    {"description", ""},
                    {"tags", {""}},
                    {"capabilities", meta->has_mtmd ? json({"completion","multimodal"}) : json({"completion"})},
                    {"parameters", ""},
                    {"details", {
                        {"parent_model", ""},
                        {"format", "gguf"},
                        {"family", ""},
                        {"families", {""}},
                        {"parameter_size", ""},
                        {"quantization_level", ""}
                    }}
                }
            }},
            {"object", "list"},
            {"data", {
                {
                    {"id",       meta->model_name},
                    {"object",   "model"},
                    {"created",  std::time(0)},
                    {"owned_by", "llamacpp"},
                    {"meta",     {
                        {"vocab_type",  meta->model_vocab_type},
                        {"n_vocab",     meta->model_vocab_n_tokens},
                        {"n_ctx_train", meta->model_n_ctx_train},
                        {"n_embd",      meta->model_n_embd_inp},
                        {"n_params",    meta->model_n_params},
                        {"size",        meta->model_size},
                    }},
                },
            }}
        };

        res->ok(models);
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model, ctx_server.vocab, ctx_server.mctx, query, documents[i]);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = 2; // default to Euclidean/L2 norm
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize");
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}
