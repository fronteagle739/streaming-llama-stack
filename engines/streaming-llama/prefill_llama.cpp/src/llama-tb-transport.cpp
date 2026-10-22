#include "llama-tb-transport.h"
#include "llama-kv-artifact.h"

// xxhash header-only: used for prefix_hash and model_fingerprint.
// XXH_INLINE_ALL makes all symbols static/inline — safe to include in multiple
// translation units without ODR conflicts (each TU gets its own copy).
#define XXH_INLINE_ALL
#include "../examples/gguf-hash/deps/xxhash/xxhash.h"

#include <mutex>
#include <condition_variable>
#include <atomic>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <net/if.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int tb_socket_t;
static constexpr tb_socket_t TB_SOCKET_INVALID = -1;
#define TB_SOCKET_CLOSE(fd) ::close(fd)
#else
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
typedef SOCKET tb_socket_t;
static const tb_socket_t TB_SOCKET_INVALID = INVALID_SOCKET;
#define TB_SOCKET_CLOSE(fd) ::closesocket(fd)

static bool tb_winsock_init() {
    static bool done = false;
    if (!done) {
        WSADATA wsa;
        done = (WSAStartup(MAKEWORD(2, 2), &wsa) == 0);
    }
    return done;
}
#endif

namespace fs = std::filesystem;

namespace {

bool env_truthy(const char *v) {
    if (!v) {
        return false;
    }
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

std::string env_str(const char * key, const std::string & dflt = "") {
    if (const char * v = std::getenv(key)) {
        if (v[0] != '\0') {
            return std::string(v);
        }
    }
    return dflt;
}

int env_i32(const char * key, int dflt) {
    if (const char * v = std::getenv(key)) {
        if (v[0] != '\0') {
            return std::atoi(v);
        }
    }
    return dflt;
}

size_t env_size(const char * key, size_t dflt) {
    if (const char * v = std::getenv(key)) {
        if (v[0] != '\0') {
            const long long n = std::atoll(v);
            if (n > 0) {
                return (size_t) n;
            }
        }
    }
    return dflt;
}

std::string trim_copy(std::string s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

std::vector<std::string> split_csv(const std::string & csv) {
    std::vector<std::string> out;
    if (csv.empty()) {
        return out;
    }
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim_copy(item);
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

std::string resolve_transport_mode(const llama_tb_transfer_options & options);

std::string normalize_transport_mode(std::string mode) {
    mode = trim_copy(mode);
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    if (mode == "tb-direct") {
        return "rdma";
    }
    if (mode == "tb-ethernet" || mode == "ethernet") {
        return "tcp";
    }
    return mode;
}

void append_unique(std::vector<std::string> & out, const std::string & value) {
    if (value.empty()) {
        return;
    }
    if (std::find(out.begin(), out.end(), value) == out.end()) {
        out.push_back(value);
    }
}

bool env_truthy_default(const char * key, bool dflt) {
    const char * v = std::getenv(key);
    if (!v || v[0] == '\0') {
        return dflt;
    }
    return env_truthy(v);
}

#ifdef __APPLE__
bool run_command_capture(const char * cmd, std::string & out) {
    out.clear();
    FILE * pipe = ::popen(cmd, "r");
    if (!pipe) {
        return false;
    }
    std::array<char, 512> buf = {};
    while (std::fgets(buf.data(), (int) buf.size(), pipe) != nullptr) {
        out.append(buf.data());
    }
    const int rc = ::pclose(pipe);
    return rc == 0;
}

std::vector<std::string> detect_macos_thunderbolt_devices() {
    std::vector<std::string> devices;
    std::string out;
    if (!run_command_capture("networksetup -listallhardwareports", out)) {
        return devices;
    }

    std::stringstream ss(out);
    std::string line;
    std::string current_port;
    while (std::getline(ss, line)) {
        const std::string trimmed = trim_copy(line);
        if (trimmed.rfind("Hardware Port:", 0) == 0) {
            current_port = trim_copy(trimmed.substr(strlen("Hardware Port:")));
            continue;
        }
        if (trimmed.rfind("Device:", 0) == 0) {
            const std::string dev = trim_copy(trimmed.substr(strlen("Device:")));
            const std::string port = lower_copy(current_port);
            if (!dev.empty() && port.find("thunderbolt") != std::string::npos &&
                    port.find("bridge") == std::string::npos) {
                append_unique(devices, dev);
            }
            current_port.clear();
        }
    }

    return devices;
}

std::vector<std::string> collect_interface_ips(const std::vector<std::string> & ifnames) {
    std::vector<std::string> out;
    if (ifnames.empty()) {
        return out;
    }

#ifdef _WIN32
    // ifaddrs/getifaddrs not available on Windows — RDMA bind addr resolution
    // not supported. Return empty; TCP transport still works via explicit endpoints.
    return out;
#else
    struct ifaddrs * ifaddr = nullptr;
    if (::getifaddrs(&ifaddr) != 0 || !ifaddr) {
        return out;
    }

    for (struct ifaddrs * cur = ifaddr; cur != nullptr; cur = cur->ifa_next) {
        if (!cur->ifa_name || !cur->ifa_addr) {
            continue;
        }
        if (!(cur->ifa_flags & IFF_UP)) {
            continue;
        }
        if (std::find(ifnames.begin(), ifnames.end(), cur->ifa_name) == ifnames.end()) {
            continue;
        }

        char buf[INET6_ADDRSTRLEN] = {};
        if (cur->ifa_addr->sa_family == AF_INET) {
            const struct sockaddr_in * sin = (const struct sockaddr_in *) cur->ifa_addr;
            if (::inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf)) == nullptr) {
                continue;
            }
        } else if (cur->ifa_addr->sa_family == AF_INET6) {
            const struct sockaddr_in6 * sin6 = (const struct sockaddr_in6 *) cur->ifa_addr;
            if (::inet_ntop(AF_INET6, &sin6->sin6_addr, buf, sizeof(buf)) == nullptr) {
                continue;
            }
        } else {
            continue;
        }

        const std::string ip(buf);
        if (ip.empty() || ip == "127.0.0.1" || ip == "::1") {
            continue;
        }
        append_unique(out, ip);
    }

    ::freeifaddrs(ifaddr);
    return out;
#endif  // _WIN32
}

std::vector<std::string> detect_macos_rdma_bind_addrs() {
    std::vector<std::string> ifnames;

    // Prefer explicit rdma_* interfaces when available.
    {
        struct ifaddrs * ifaddr = nullptr;
        if (::getifaddrs(&ifaddr) == 0 && ifaddr) {
            for (struct ifaddrs * cur = ifaddr; cur != nullptr; cur = cur->ifa_next) {
                if (!cur->ifa_name) {
                    continue;
                }
                const std::string name(cur->ifa_name);
                if (name.rfind("rdma_", 0) == 0) {
                    append_unique(ifnames, name);
                }
            }
            ::freeifaddrs(ifaddr);
        }
    }

    // Mirror exo's mapping from Thunderbolt hardware ports to rdma_<enX> names.
    const std::vector<std::string> tb_devices = detect_macos_thunderbolt_devices();
    for (const std::string & dev : tb_devices) {
        append_unique(ifnames, "rdma_" + dev);
        append_unique(ifnames, dev);
    }

    return collect_interface_ips(ifnames);
}
#endif // __APPLE__

std::vector<std::string> resolve_auto_bind_addrs(const llama_tb_transfer_options & options) {
    if (!env_truthy_default("LLAMA_PREFILL_KV_AUTO_BIND", true)) {
        return {};
    }

    const std::string mode = resolve_transport_mode(options);
    if (!(mode == "auto" || mode == "rdma" || mode == "mixed")) {
        return {};
    }

#ifdef __APPLE__
    return detect_macos_rdma_bind_addrs();
#else
    return {};
#endif
}

std::string default_session_root() {
    if (const char *v = std::getenv("LLAMA_PREFILL_TB_DIR")) {
        if (v[0] != '\0') {
            return std::string(v);
        }
    }
    return "/tmp/llama_tb_sessions";
}

std::string default_transport_endpoint() {
    return env_str("LLAMA_PREFILL_TB_ENDPOINT", "");
}

std::string default_rdma_endpoint() {
    return env_str("LLAMA_PREFILL_KV_RDMA_ENDPOINT", default_transport_endpoint());
}

std::string default_ip_endpoint() {
    const std::string host = env_str("LLAMA_PREFILL_KV_HOST", "");
    const int port = env_i32("LLAMA_PREFILL_KV_PORT", 0);
    if (!host.empty() && port > 0) {
        return host + ":" + std::to_string(port);
    }
    return "";
}

std::string default_tcp_endpoint() {
    return env_str("LLAMA_PREFILL_KV_TCP_ENDPOINT", default_ip_endpoint());
}

uint64_t unix_now_us() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

double calc_throughput_gbps(uint64_t bytes, double transfer_ms) {
    if (bytes == 0 || transfer_ms <= 0.0) {
        return 0.0;
    }
    return ((double) bytes * 8.0) / (transfer_ms * 1.0e6);
}

std::string resolve_transport_mode(const llama_tb_transfer_options & options) {
    std::string mode = !options.transport_mode.empty() ? options.transport_mode :
                       env_str("LLAMA_PREFILL_KV_TRANSPORT", "auto");
    mode = normalize_transport_mode(mode);
    return mode.empty() ? "auto" : mode;
}

std::vector<std::string> resolve_transport_endpoints(const llama_tb_transfer_options & options,
                                                     const std::string & mode) {
    std::vector<std::string> eps;

    const auto append_unique = [&](const std::string & ep) {
        if (ep.empty()) {
            return;
        }
        if (std::find(eps.begin(), eps.end(), ep) == eps.end()) {
            eps.push_back(ep);
        }
    };

    if (!options.endpoint.empty()) {
        append_unique(options.endpoint);
    }

    const std::string kv_peers = options.kv_peer_addrs.empty() ? env_str("LLAMA_PREFILL_KV_PEER_ADDRS", "") : options.kv_peer_addrs;
    for (const auto & ep : split_csv(kv_peers)) {
        append_unique(ep);
    }

    if (mode == "rdma") {
        append_unique(default_rdma_endpoint());
    } else if (mode == "tcp") {
        if (!options.kv_host.empty() && options.kv_port > 0) {
            append_unique(options.kv_host + ":" + std::to_string(options.kv_port));
        }
        append_unique(default_tcp_endpoint());
    } else if (mode == "mixed") {
        append_unique(default_rdma_endpoint());
        if (!options.kv_host.empty() && options.kv_port > 0) {
            append_unique(options.kv_host + ":" + std::to_string(options.kv_port));
        }
        append_unique(default_tcp_endpoint());
    } else if (mode == "auto") {
        append_unique(default_rdma_endpoint());
        append_unique(default_tcp_endpoint());
    }

    return eps;
}

void set_error(std::string *error, const std::string &msg) {
    if (error) {
        *error = msg;
    }
}

bool copy_file_binary(const fs::path &src, const fs::path &dst, std::string *error) {
    std::error_code ec;
    fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
    if (!ec) {
        return true;
    }

    // Fallback stream copy for filesystems where copy_file may fail.
    std::ifstream ifs(src, std::ios::binary);
    if (!ifs.is_open()) {
        set_error(error, "failed to open source file for transport copy: " + src.string());
        return false;
    }
    std::ofstream ofs(dst, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        set_error(error, "failed to open destination file for transport copy: " + dst.string());
        return false;
    }
    ofs << ifs.rdbuf();
    if (!ofs.good()) {
        set_error(error, "failed to copy transport payload to: " + dst.string());
        return false;
    }
    return true;
}

constexpr uint16_t TBP_PROTO_MAJOR = 1;
constexpr uint16_t TBP_PROTO_MINOR = 0;
constexpr uint32_t TBP_MAGIC       = 0x54425031u; // "TBP1"

enum tbp_msg_type : uint16_t {
    TBP_MSG_HELLO                = 1,
    TBP_MSG_SESSION_START        = 3,
    TBP_MSG_KV_SEGMENT_BEGIN     = 7,
    TBP_MSG_KV_CHUNK             = 8,
    TBP_MSG_KV_SEGMENT_END       = 9,
    TBP_MSG_KV_ACK               = 10,
    TBP_MSG_KV_DONE              = 12,
    // PROMPT-DIRECT KV-handshake (Gate #2): control socket messages
    // Sent over a separate control socket on kv_port-1 to keep the
    // unidirectional TBP1 stream protocol unchanged.
    TBP_MSG_QUERY_PREFIX_CACHE   = 30,  // RTX  → Mac
    TBP_MSG_PREFIX_CACHE_REPLY   = 31,  // Mac  → RTX
};

void append_u16_le(std::vector<uint8_t> & out, uint16_t v) {
    out.push_back((uint8_t) ((v >> 0) & 0xFFu));
    out.push_back((uint8_t) ((v >> 8) & 0xFFu));
}

void append_u32_le(std::vector<uint8_t> & out, uint32_t v) {
    out.push_back((uint8_t) ((v >> 0) & 0xFFu));
    out.push_back((uint8_t) ((v >> 8) & 0xFFu));
    out.push_back((uint8_t) ((v >> 16) & 0xFFu));
    out.push_back((uint8_t) ((v >> 24) & 0xFFu));
}

void append_u64_le(std::vector<uint8_t> & out, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        out.push_back((uint8_t) ((v >> (8 * i)) & 0xFFu));
    }
}

void patch_u32_le(std::vector<uint8_t> & out, size_t offset, uint32_t v) {
    if (offset + 4 > out.size()) {
        return;
    }
    out[offset + 0] = (uint8_t) ((v >> 0) & 0xFFu);
    out[offset + 1] = (uint8_t) ((v >> 8) & 0xFFu);
    out[offset + 2] = (uint8_t) ((v >> 16) & 0xFFu);
    out[offset + 3] = (uint8_t) ((v >> 24) & 0xFFu);
}

uint64_t hash_session_id(const std::string & sid) {
    // FNV-1a 64-bit hash to derive stable wire session IDs from string IDs.
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : sid) {
        h ^= (uint64_t) c;
        h *= 1099511628211ull;
    }
    return h;
}

struct tbp_rx_frame {
    uint16_t msg_type = 0;
    uint16_t flags = 0;
    uint64_t session_id = 0;
    uint64_t stream_id = 0;
    uint64_t seq_no = 0;
    std::vector<uint8_t> payload;
};

bool send_all_bytes(tb_socket_t fd, const uint8_t * data, size_t size, std::string * error) {
    size_t off = 0;
    while (off < size) {
#ifndef _WIN32
        const ssize_t n = ::send(fd, data + off, size - off, 0);
#else
        const int n = ::send(fd, (const char*)(data + off), (int)(size - off), 0);
#endif
        if (n <= 0) {
            set_error(error, "transport send failed");
            return false;
        }
        off += (size_t) n;
    }
    return true;
}

bool recv_all_bytes_timeout(tb_socket_t fd, uint8_t * dst, size_t size, int timeout_ms, std::string * error) {
    size_t off = 0;
    while (off < size) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(fd, &rfds);

        struct timeval tv = {};
        tv.tv_sec  = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
#ifndef _WIN32
        const int sel = ::select((int)fd + 1, &rfds, nullptr, nullptr, &tv);
#else
        const int sel = ::select(0, &rfds, nullptr, nullptr, &tv);
#endif
        if (sel <= 0) {
            set_error(error, sel == 0 ? "transport receive timeout" : "transport receive select failed");
            return false;
        }

#ifndef _WIN32
        const ssize_t n = ::recv(fd, dst + off, size - off, 0);
#else
        const int n = ::recv(fd, (char*)(dst + off), (int)(size - off), 0);
#endif
        if (n <= 0) {
            set_error(error, "transport receive failed");
            return false;
        }
        off += (size_t) n;
    }
    return true;
}

bool send_tbp_frame(tb_socket_t fd, uint16_t msg_type, uint16_t flags, uint64_t session_id, uint64_t stream_id, uint64_t seq_no,
                    const uint8_t * payload, size_t payload_bytes, std::string * error) {
    // KV chunks carry no payload CRC unless LLAMA_PREFILL_TB_CHUNK_CRC=1 (2026-09-10): the byte-wise
    // CRC over every chunk ran in series with the socket write and capped a 6.3 GB export at ~330 MB/s
    // on a wire that carries ~750; no receiver verified it (the Mac reads and discards the field).
    static const bool chunk_crc = []() {
        const char * e = std::getenv("LLAMA_PREFILL_TB_CHUNK_CRC");
        return e && e[0] != 0 && e[0] != '0';
    }();
    const bool want_crc = payload_bytes > 0 && payload != nullptr && (msg_type != TBP_MSG_KV_CHUNK || chunk_crc);
    const uint32_t payload_crc = want_crc ? llama_kv_artifact_crc32(payload, payload_bytes) : 0u;

    std::vector<uint8_t> hdr;
    hdr.reserve(52);
    append_u32_le(hdr, TBP_MAGIC);
    append_u16_le(hdr, TBP_PROTO_MAJOR);
    append_u16_le(hdr, TBP_PROTO_MINOR);
    append_u16_le(hdr, msg_type);
    append_u16_le(hdr, flags);
    append_u32_le(hdr, 52u);
    append_u32_le(hdr, (uint32_t) payload_bytes);
    append_u64_le(hdr, session_id);
    append_u64_le(hdr, stream_id);
    append_u64_le(hdr, seq_no);
    const size_t hdr_crc_offset = hdr.size();
    append_u32_le(hdr, 0u); // header CRC placeholder
    append_u32_le(hdr, payload_crc);
    const uint32_t hdr_crc = llama_kv_artifact_crc32(hdr.data(), hdr.size());
    patch_u32_le(hdr, hdr_crc_offset, hdr_crc);

    if (!send_all_bytes(fd, hdr.data(), hdr.size(), error)) {
        return false;
    }
    if (payload_bytes > 0 && payload != nullptr) {
        if (!send_all_bytes(fd, payload, payload_bytes, error)) {
            return false;
        }
    }
    return true;
}

bool recv_tbp_frame_timeout(tb_socket_t fd, int timeout_ms, tbp_rx_frame * out, std::string * error) {
    std::array<uint8_t, 52> hdr = {};
    if (!recv_all_bytes_timeout(fd, hdr.data(), hdr.size(), timeout_ms, error)) {
        return false;
    }

    auto rd_u16 = [&](size_t off) -> uint16_t {
        return (uint16_t) hdr[off + 0] | ((uint16_t) hdr[off + 1] << 8);
    };
    auto rd_u32 = [&](size_t off) -> uint32_t {
        return (uint32_t) hdr[off + 0] |
               ((uint32_t) hdr[off + 1] << 8) |
               ((uint32_t) hdr[off + 2] << 16) |
               ((uint32_t) hdr[off + 3] << 24);
    };
    auto rd_u64 = [&](size_t off) -> uint64_t {
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) {
            v |= (uint64_t) hdr[off + i] << (8 * i);
        }
        return v;
    };

    const uint32_t magic = rd_u32(0);
    if (magic != TBP_MAGIC) {
        set_error(error, "invalid TBP frame magic while waiting for ACK");
        return false;
    }

    const uint32_t payload_bytes = rd_u32(16);
    if (out) {
        out->msg_type = rd_u16(8);
        out->flags = rd_u16(10);
        out->session_id = rd_u64(20);
        out->stream_id = rd_u64(28);
        out->seq_no = rd_u64(36);
        out->payload.resize(payload_bytes);
    }

    if (payload_bytes > 0) {
        std::vector<uint8_t> tmp;
        uint8_t * dst = nullptr;
        if (out) {
            dst = out->payload.data();
        } else {
            tmp.resize(payload_bytes);
            dst = tmp.data();
        }
        if (!recv_all_bytes_timeout(fd, dst, payload_bytes, timeout_ms, error)) {
            return false;
        }
    }

    return true;
}

bool parse_host_port(const std::string & endpoint, std::string * host_out, std::string * port_out) {
    const size_t colon = endpoint.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= endpoint.size()) {
        return false;
    }
    *host_out = endpoint.substr(0, colon);
    *port_out = endpoint.substr(colon + 1);
    return !host_out->empty() && !port_out->empty();
}

tb_socket_t connect_endpoint(const std::string & endpoint,
                     const std::string & bind_addr,
                     int sock_send_buf,
                     int sock_recv_buf,
                     std::string * error) {
#ifdef _WIN32
    if (!tb_winsock_init()) {
        set_error(error, "winsock init failed");
        return TB_SOCKET_INVALID;
    }
#endif
    std::string host;
    std::string port;
    if (!parse_host_port(endpoint, &host, &port)) {
        set_error(error, "invalid transport endpoint, expected host:port: " + endpoint);
        return TB_SOCKET_INVALID;
    }

    struct addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo * ai = nullptr;
    const int gai_rc = ::getaddrinfo(host.c_str(), port.c_str(), &hints, &ai);
    if (gai_rc != 0) {
#ifndef _WIN32
        set_error(error, "getaddrinfo failed for endpoint " + endpoint + ": " + std::string(gai_strerror(gai_rc)));
#else
        set_error(error, "getaddrinfo failed for endpoint " + endpoint + " (error " + std::to_string(gai_rc) + ")");
#endif
        return TB_SOCKET_INVALID;
    }

    tb_socket_t fd = TB_SOCKET_INVALID;
    for (struct addrinfo * cur = ai; cur != nullptr; cur = cur->ai_next) {
        fd = ::socket(cur->ai_family, cur->ai_socktype, cur->ai_protocol);
        if (fd == TB_SOCKET_INVALID) {
            continue;
        }
#ifndef _WIN32
        if (sock_send_buf > 0) {
            (void) ::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sock_send_buf, sizeof(sock_send_buf));
        }
        if (sock_recv_buf > 0) {
            (void) ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &sock_recv_buf, sizeof(sock_recv_buf));
        }
#else
        if (sock_send_buf > 0) {
            (void) ::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, (const char*)&sock_send_buf, sizeof(sock_send_buf));
        }
        if (sock_recv_buf > 0) {
            (void) ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, (const char*)&sock_recv_buf, sizeof(sock_recv_buf));
        }
#endif

        if (!bind_addr.empty()) {
            struct addrinfo bind_hints = {};
            bind_hints.ai_family = cur->ai_family;
            bind_hints.ai_socktype = cur->ai_socktype;
            bind_hints.ai_flags = AI_PASSIVE;
            struct addrinfo * bind_ai = nullptr;
            if (::getaddrinfo(bind_addr.c_str(), nullptr, &bind_hints, &bind_ai) == 0) {
                if (bind_ai && ::bind(fd, bind_ai->ai_addr, (int)bind_ai->ai_addrlen) != 0) {
                    // non-fatal for mixed local test paths; continue without bind pinning
                }
                if (bind_ai) {
                    ::freeaddrinfo(bind_ai);
                }
            }
        }

        if (::connect(fd, cur->ai_addr, (int)cur->ai_addrlen) == 0) {
            break;
        }
        TB_SOCKET_CLOSE(fd);
        fd = TB_SOCKET_INVALID;
    }
    ::freeaddrinfo(ai);

    if (fd == TB_SOCKET_INVALID) {
        set_error(error, "failed to connect transport endpoint: " + endpoint);
    }
    return fd;
}

bool send_artifact_over_tcp(const uint8_t * src_data, size_t src_size, const std::string & src_name,
                            const std::vector<std::string> & endpoints, const llama_tb_transfer_options & options,
                            uint64_t payload_bytes, llama_tb_transfer_result * result, std::string * error,
                            const llama_tb_pull_fn * pull = nullptr) {
    if (endpoints.empty()) {
        set_error(error, "no TCP transport endpoints resolved");
        return false;
    }

    struct pending_chunk {
        uint64_t             seq_no = 0;
        std::vector<uint8_t> data;
    };

    struct stream_conn {
        tb_socket_t fd = TB_SOCKET_INVALID;
        uint64_t    stream_id = 0;
        uint64_t    inflight_bytes = 0;
        std::string endpoint;
        std::string bind_addr;
        std::deque<pending_chunk> pending_chunks;
    };

    const std::string sid = options.session_id.empty() ? "session_default" : options.session_id;
    const uint64_t wire_sid = hash_session_id(sid);
    uint64_t seq = 0;
    const std::string mode = resolve_transport_mode(options);
    const bool is_rdma_transport = mode == "rdma";

    const bool ack_required = env_truthy(std::getenv("LLAMA_PREFILL_TB_ACK_REQUIRED"));
    const bool ack_strict   = env_truthy(std::getenv("LLAMA_PREFILL_TB_ACK_STRICT"));
    const uint64_t max_inflight_bytes = std::max<uint64_t>(
        1ull << 20,
        options.max_inflight_bytes > 0 ? (uint64_t) options.max_inflight_bytes :
            (uint64_t) env_size("LLAMA_PREFILL_TB_MAX_INFLIGHT_BYTES", 268435456)
    );
    const int ack_timeout_ms = std::max(10, env_i32("LLAMA_PREFILL_TB_ACK_TIMEOUT_MS", 250));
    const std::string balance = !options.kv_balance.empty() ? options.kv_balance : env_str("LLAMA_PREFILL_KV_BALANCE", "roundrobin");
    std::vector<std::string> bind_addrs = split_csv(
        !options.kv_bind_addrs.empty()
            ? options.kv_bind_addrs
            : (is_rdma_transport
                   ? env_str("LLAMA_PREFILL_KV_RDMA_BIND_ADDRS", env_str("LLAMA_PREFILL_KV_BIND_ADDRS", ""))
                   : env_str("LLAMA_PREFILL_KV_TCP_BIND_ADDRS", env_str("LLAMA_PREFILL_KV_BIND_ADDRS", "")))
    );
    if (bind_addrs.empty() && is_rdma_transport) {
        bind_addrs = resolve_auto_bind_addrs(options);
    }
    const int stream_count = std::max<int>(1, options.stream_count > 0 ? options.stream_count :
                                              env_i32("LLAMA_PREFILL_TB_STREAMS",
                                                      env_i32("LLAMA_PREFILL_KV_STREAMS", 1)));
    const int sock_send_buf = options.socket_send_buf > 0 ? options.socket_send_buf :
                              env_i32("LLAMA_PREFILL_TB_SOCKET_SEND_BUF", env_i32("LLAMA_PREFILL_KV_SOCKET_SEND_BUF", 0));
    const int sock_recv_buf = options.socket_recv_buf > 0 ? options.socket_recv_buf :
                              env_i32("LLAMA_PREFILL_TB_SOCKET_RECV_BUF", env_i32("LLAMA_PREFILL_KV_SOCKET_RECV_BUF", 0));

    auto close_all = [&](std::vector<stream_conn> & streams) {
        for (auto & s : streams) {
            if (s.fd != TB_SOCKET_INVALID) {
                TB_SOCKET_CLOSE(s.fd);
                s.fd = TB_SOCKET_INVALID;
            }
        }
    };

    std::vector<stream_conn> streams;
    streams.reserve((size_t) stream_count);
    for (int i = 0; i < stream_count; ++i) {
        const std::string & endpoint = endpoints[(size_t) i % endpoints.size()];
        const std::string bind_addr = bind_addrs.empty() ? "" : bind_addrs[(size_t) i % bind_addrs.size()];
        tb_socket_t fd = TB_SOCKET_INVALID;
        // Default 60 x 1s. Overridable because the receiver may still be LOADING the decode model when the
        // RTX side connects — a ~432GB GGUF (MiniMax-M3) takes several minutes to come up on :9100, and a
        // 60s window made every such switch fail (observed 2026-07-11). The studio launcher sets
        // LLAMA_TB_CONNECT_RETRIES per preset for huge Mac-side models.
        static const int max_retries = [] {
            const char * env = getenv("LLAMA_TB_CONNECT_RETRIES");
            if (env && env[0]) {
                const int v = atoi(env);
                if (v >= 1 && v <= 3600) return v;
            }
            return 60;
        }();
        for (int retry = 0; retry < max_retries; retry++) {
            fd = connect_endpoint(endpoint, bind_addr, sock_send_buf, sock_recv_buf, error);
            if (fd != TB_SOCKET_INVALID) break;
            if (retry < max_retries - 1) {
                fprintf(stderr, "[transport] connect to %s failed (attempt %d/%d), retrying in 1s...\n",
                        endpoint.c_str(), retry + 1, max_retries);
                fflush(stderr);
#ifdef _WIN32
                Sleep(1000);
#else
                usleep(1000000);
#endif
            }
        }
        if (fd == TB_SOCKET_INVALID) {
            close_all(streams);
            return false;
        }
        streams.push_back(stream_conn{
            /*fd=*/fd,
            /*stream_id=*/(uint64_t) (i + 1),
            /*inflight_bytes=*/0,
            /*endpoint=*/endpoint,
            /*bind_addr=*/bind_addr,
            /*pending_chunks=*/{},
        });
    }

    auto send_frame_str = [&](stream_conn & stream, uint16_t msg_type, uint64_t stream_id, const std::string & payload) -> bool {
        return send_tbp_frame(stream.fd, msg_type, 0, wire_sid, stream_id, seq++,
                              reinterpret_cast<const uint8_t *>(payload.data()), payload.size(), error);
    };

    const std::string role_payload = "node_role=prefill-server;capabilities=bulk,progressive,multistream";
    for (auto & stream : streams) {
        if (!send_frame_str(stream, TBP_MSG_HELLO, 0, role_payload)) {
            close_all(streams);
            return false;
        }
    }

    std::ostringstream session_payload;
    session_payload << "mode=" << (options.progressive ? "progressive" : "bulk")
                    << ";artifact=" << src_name
                    << ";bytes=" << payload_bytes
                    << ";remote_nodes=" << (int)std::max(1, options.remote_nodes)
                    << ";execution_mode=" << (options.execution_mode.empty() ? "coupled" : options.execution_mode)
                    << ";streams=" << stream_count
                    << ";ack_required=" << (ack_required ? 1 : 0)
                    << ";balance=" << balance;
    for (auto & stream : streams) {
        if (!send_frame_str(stream, TBP_MSG_SESSION_START, 0, session_payload.str())) {
            close_all(streams);
            return false;
        }
    }

    std::ostringstream seg_begin;
    seg_begin << "segment_id=0;payload_bytes=" << payload_bytes
              << ";chunk_bytes_nominal=" << std::max<size_t>(1, options.chunk_bytes)
              << ";stream_count=" << stream_count;
    for (auto & stream : streams) {
        if (!send_frame_str(stream, TBP_MSG_KV_SEGMENT_BEGIN, stream.stream_id, seg_begin.str())) {
            close_all(streams);
            return false;
        }
    }

    const size_t chunk_bytes = std::max<size_t>(
        1, options.chunk_bytes > 0 ? options.chunk_bytes :
        env_size("LLAMA_PREFILL_TB_STREAM_CHUNK_BYTES", env_size("LLAMA_PREFILL_KV_STREAM_CHUNK_BYTES", 4 * 1024 * 1024))
    );
    if (!pull && ((src_data == nullptr && src_size > 0) || (src_size == 0 && payload_bytes > 0))) {
        set_error(error, "send_artifact_over_tcp: empty source buffer for non-zero payload");
        close_all(streams);
        return false;
    }
    if (pull && streams.size() != 1) {
        set_error(error, "send_artifact_over_tcp: a pulled stream needs exactly one TCP stream (chunks are reassembled in arrival order)");
        close_all(streams);
        return false;
    }
    size_t src_offset = 0;
    std::vector<uint8_t> chunk(chunk_bytes);
    uint32_t chunks_sent = 0;
    // Pulled streams (2026-09-10): the producer reads the next chunks from the NVMe tier on its own
    // thread into a small ring while this thread sends, so the disk read and the socket write overlap
    // instead of alternating (measured 319 MB/s alternating on a 6.3 GB export; the wire does ~750).
    struct ring_slot { std::vector<uint8_t> buf; size_t n = 0; bool ready = false; bool last = false; };
    const size_t ring_n = pull ? 4 : 0;
    std::vector<ring_slot> ring(ring_n);
    for (auto & r : ring) r.buf.resize(chunk_bytes);
    std::mutex ring_mu;
    std::condition_variable ring_cv;
    std::atomic<bool> ring_stop{false};
    std::string ring_err;
    bool ring_failed = false;
    size_t ring_head = 0;   // next slot the consumer takes
    std::thread reader;
    if (pull) {
        reader = std::thread([&]() {
            size_t tail = 0;          // next slot the producer fills
            size_t produced = 0;      // bytes produced so far
            while (!ring_stop.load()) {
                std::unique_lock<std::mutex> lk(ring_mu);
                ring_cv.wait(lk, [&]() { return ring_stop.load() || !ring[tail].ready; });
                if (ring_stop.load()) break;
                lk.unlock();
                const size_t want = std::min(chunk_bytes, src_size - produced);
                size_t n = 0;
                std::string perr;
                bool ok = want > 0 && (*pull)(ring[tail].buf.data(), want, n, &perr);
                lk.lock();
                if (!ok || n == 0) {
                    ring_failed = true;
                    ring_err = !ok ? perr : ("pull ended at offset " + std::to_string(produced) + " of " + std::to_string(src_size));
                    ring[tail].ready = true;
                    ring[tail].n = 0;
                    ring[tail].last = true;
                    ring_cv.notify_all();
                    break;
                }
                produced += n;
                ring[tail].n = n;
                ring[tail].last = produced >= src_size;
                ring[tail].ready = true;
                ring_cv.notify_all();
                if (ring[tail].last) break;
                tail = (tail + 1) % ring_n;
            }
        });
    }
    auto stop_reader = [&]() {
        if (reader.joinable()) {
            ring_stop.store(true);
            ring_cv.notify_all();
            reader.join();
        }
    };
    uint32_t retransmit_chunks = 0;
    uint64_t window_stalls_ms = 0;
    size_t rr_index = 0;
    const auto choose_stream_idx = [&]() -> size_t {
        if (streams.size() == 1) {
            return 0;
        }
        if (balance == "hash") {
            return (size_t) (seq % streams.size());
        }
        if (balance == "adaptive") {
            size_t best = 0;
            uint64_t best_inflight = streams[0].inflight_bytes;
            for (size_t i = 1; i < streams.size(); ++i) {
                if (streams[i].inflight_bytes < best_inflight) {
                    best = i;
                    best_inflight = streams[i].inflight_bytes;
                }
            }
            return best;
        }
        size_t idx = rr_index % streams.size();
        rr_index = (rr_index + 1) % streams.size();
        return idx;
    };

    const auto transfer_t0 = std::chrono::steady_clock::now();
    uint64_t first_chunk_send_unix_us = 0;
    uint64_t last_chunk_send_unix_us = 0;

    const auto release_acked_chunks = [](stream_conn & stream, uint64_t ack_seq_no) {
        uint64_t released = 0;
        while (!stream.pending_chunks.empty() && stream.pending_chunks.front().seq_no <= ack_seq_no) {
            released += (uint64_t) stream.pending_chunks.front().data.size();
            stream.pending_chunks.pop_front();
        }
        if (released >= stream.inflight_bytes) {
            stream.inflight_bytes = 0;
        } else {
            stream.inflight_bytes -= released;
        }
    };

    while (src_offset < src_size) {
        const size_t remaining = src_size - src_offset;
        size_t n = std::min(chunk.size(), remaining);
        if (n == 0) {
            break;
        }
        const uint8_t * send_ptr = chunk.data();
        if (pull) {
            std::unique_lock<std::mutex> lk(ring_mu);
            ring_cv.wait(lk, [&]() { return ring[ring_head].ready; });
            if (ring_failed && ring[ring_head].n == 0) {
                const std::string e = ring_err;
                lk.unlock();
                set_error(error, "send_artifact_over_tcp: pull failed at offset " + std::to_string(src_offset) + ": " + e);
                stop_reader();
                close_all(streams);
                return false;
            }
            n = ring[ring_head].n;
            send_ptr = ring[ring_head].buf.data();
            lk.unlock();
        } else {
            std::memcpy(chunk.data(), src_data + src_offset, n);
        }
        src_offset += n;

        stream_conn & stream = streams[choose_stream_idx()];
        const uint64_t chunk_seq_no = seq++;
        if (!send_tbp_frame(stream.fd, TBP_MSG_KV_CHUNK, 0, wire_sid, stream.stream_id, chunk_seq_no, send_ptr, (size_t) n, error)) {
            stop_reader();
            close_all(streams);
            return false;
        }
        if (pull) {
            std::lock_guard<std::mutex> lk(ring_mu);
            ring[ring_head].ready = false;
            ring_head = (ring_head + 1) % ring_n;
            ring_cv.notify_all();
        }
        const uint64_t now_us = unix_now_us();
        if (first_chunk_send_unix_us == 0) {
            first_chunk_send_unix_us = now_us;
        }
        last_chunk_send_unix_us = now_us;
        stream.inflight_bytes += (uint64_t) n;
        // Keep the retransmit copy ONLY when the ACK path can actually use it.
        //
        // `pending.data` is readable from exactly two places, the NACK-retransmit lookup and
        // release_acked_chunks, and BOTH live inside the block gated on
        // `options.progressive && ack_required && inflight >= max_inflight`. In production
        // LLAMA_PREFILL_TB_ACK_REQUIRED is unset, so that branch never executes: nothing reads
        // these copies, and nothing frees them either -- pending_chunks.clear() is inside the same
        // dead branch. Every chunk therefore accumulated for the whole transfer.
        //
        // Measured 2026-09-03 on a real 832,518,542-byte handoff: 199 chunks of ~4.2 MB, so ~832 MB
        // of retained heap, a second memcpy per chunk, and a first-touch page fault over all of it,
        // for data no code path could ever consume. The transport moved 332 MB/s where a plain TCP
        // socket over the same wire (bridge100 -> en0, 10GbE) measured 750 MB/s both-sides-agreeing.
        //
        // The log signature of the dead branch is `retransmit_chunks=0 window_stalls_ms=0` -- both
        // counters only advance inside it, so those zeros meant "never ran", not "clean transfer".
        //
        // With acks ON this is byte-for-byte the previous behaviour.
        if (options.progressive && ack_required) {
            pending_chunk pending = {};
            pending.seq_no = chunk_seq_no;
            pending.data.assign(send_ptr, send_ptr + (size_t) n);
            stream.pending_chunks.emplace_back(std::move(pending));
        }
        ++chunks_sent;

        if (options.progressive && ack_required && stream.inflight_bytes >= max_inflight_bytes) {
            const auto t0 = std::chrono::steady_clock::now();
            bool acked = false;
            int ignored_frames = 0;
            while (!acked && ignored_frames < 8) {
                tbp_rx_frame ack;
                std::string ack_err;
                if (!recv_tbp_frame_timeout(stream.fd, ack_timeout_ms, &ack, &ack_err)) {
                    if (ack_strict) {
                        set_error(error, "strict ACK pacing failed: " + ack_err);
                        close_all(streams);
                        return false;
                    }
                    break;
                }

                const bool is_ack = ack.msg_type == TBP_MSG_KV_ACK;
                const bool sid_ok = ack.session_id == wire_sid;
                const bool stream_ok = ack.stream_id == stream.stream_id;
                const std::string ack_payload(ack.payload.begin(), ack.payload.end());
                const bool is_nack_payload = ack_payload.find("nack=") != std::string::npos;
                const bool is_ok_payload = ack_payload.empty() || ack_payload.find("ack=1") != std::string::npos;
                const bool is_negative_payload = is_nack_payload || (!ack_payload.empty() && !is_ok_payload);

                if (is_ack && sid_ok && stream_ok) {
                    if (is_negative_payload) {
                        auto it = std::find_if(stream.pending_chunks.begin(), stream.pending_chunks.end(),
                                               [&](const pending_chunk & p) { return p.seq_no == ack.seq_no; });
                        if (it != stream.pending_chunks.end()) {
                            if (!send_tbp_frame(stream.fd,
                                                TBP_MSG_KV_CHUNK,
                                                0,
                                                wire_sid,
                                                stream.stream_id,
                                                it->seq_no,
                                                it->data.data(),
                                                it->data.size(),
                                                error)) {
                                close_all(streams);
                                return false;
                            }
                            ++retransmit_chunks;
                            const uint64_t resend_now_us = unix_now_us();
                            if (first_chunk_send_unix_us == 0) {
                                first_chunk_send_unix_us = resend_now_us;
                            }
                            last_chunk_send_unix_us = resend_now_us;
                            continue;
                        }
                        if (ack_strict) {
                            set_error(error, "strict ACK pacing received NACK for unknown chunk seq");
                            close_all(streams);
                            return false;
                        }
                        ++ignored_frames;
                        continue;
                    }

                    if (is_ok_payload) {
                        release_acked_chunks(stream, ack.seq_no);
                        acked = true;
                        break;
                    }
                }

                if (ack_strict) {
                    std::ostringstream oss;
                    oss << "strict ACK pacing rejected frame: msg=" << ack.msg_type
                        << " sid=" << ack.session_id
                        << " stream=" << ack.stream_id
                        << " expected_sid=" << wire_sid
                        << " expected_stream=" << stream.stream_id;
                    set_error(error, oss.str());
                    close_all(streams);
                    return false;
                }
                ++ignored_frames;
            }

            const auto t1 = std::chrono::steady_clock::now();
            window_stalls_ms += (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

            if (acked) {
                // inflight/pending already released by ACK seq handling.
            } else if (!ack_strict) {
                stream.inflight_bytes = 0;
                stream.pending_chunks.clear();
            }
        }
    }

    stop_reader();
    std::ostringstream seg_end;
    seg_end << "segment_id=0;chunks_sent=" << chunks_sent;
    for (auto & stream : streams) {
        if (!send_frame_str(stream, TBP_MSG_KV_SEGMENT_END, stream.stream_id, seg_end.str())) {
            close_all(streams);
            return false;
        }
        if (!send_frame_str(stream, TBP_MSG_KV_DONE, stream.stream_id, "done=1")) {
            close_all(streams);
            return false;
        }
    }

    close_all(streams);

    const auto transfer_t1 = std::chrono::steady_clock::now();
    const double transfer_ms = (double) std::chrono::duration_cast<std::chrono::microseconds>(transfer_t1 - transfer_t0).count() / 1000.0;
    const double throughput_gbps = calc_throughput_gbps(payload_bytes, transfer_ms);

    if (result) {
        result->session_path = std::string(is_rdma_transport ? "rdma://" : "tcp://") + endpoints.front() + "/" + sid;
        result->transport_mode = mode.empty() ? "tcp" : mode;
        result->transport_backend = is_rdma_transport ? "rdma" : "tcp";
        result->bytes_sent = payload_bytes;
        result->chunks_sent = chunks_sent;
        result->progressive = options.progressive;
        result->stream_count = stream_count;
        result->interface_count = (int32_t) std::max<size_t>(endpoints.size(), bind_addrs.size());
        result->retransmit_chunks = retransmit_chunks;
        result->window_stalls_ms = window_stalls_ms;
        result->transfer_ms = transfer_ms;
        result->throughput_gbps = throughput_gbps;
        result->first_chunk_send_unix_us = first_chunk_send_unix_us;
        result->last_chunk_send_unix_us = last_chunk_send_unix_us;
    }
    return true;
}

}  // namespace

uint64_t llama_tb_transport_hash_tokens(const int32_t * tokens, int32_t n_tokens) {
    if (!tokens || n_tokens <= 0) {
        return 0;
    }
    return XXH64(tokens, (size_t) n_tokens * sizeof(int32_t), /*seed=*/0);
}

uint64_t llama_tb_transport_model_fingerprint(const char * model_desc) {
    if (!model_desc || model_desc[0] == '\0') {
        return 0;
    }
    return XXH64(model_desc, strlen(model_desc), /*seed=*/0);
}

bool llama_tb_transport_enabled() {
    return env_truthy(std::getenv("LLAMA_PREFILL_TB_ENABLE"));
}

bool llama_tb_transport_send_artifact(const std::string &          artifact_path,
                                      const llama_tb_transfer_options & options,
                                      llama_tb_transfer_result *    result,
                                      std::string *                 error) {
    if (!llama_tb_transport_enabled()) {
        set_error(error, "thunderbolt transport disabled (set LLAMA_PREFILL_TB_ENABLE=1)");
        return false;
    }

    fs::path src = artifact_path;
    if (!fs::exists(src)) {
        set_error(error, "artifact does not exist: " + artifact_path);
        return false;
    }

    const fs::path root = options.session_dir.empty() ? fs::path(default_session_root()) : fs::path(options.session_dir);
    const std::string sid = options.session_id.empty() ? "session_default" : options.session_id;
    const fs::path session = root / sid;

    std::error_code ec;
    fs::create_directories(session, ec);
    if (ec) {
        set_error(error, "failed to create transport session dir: " + session.string());
        return false;
    }

    const uint64_t payload_bytes = fs::file_size(src, ec);
    if (ec) {
        set_error(error, "failed to stat artifact: " + artifact_path);
        return false;
    }

    uint32_t chunks_sent = 0;
    llama_tb_transfer_result resolved_result;
    std::string backend_desc = "filesystem";
    std::string endpoint_desc = "<none>";
    std::string resolved_transport_mode = "disabled";
    bool transfer_ok = false;

    const std::string selected_mode = resolve_transport_mode(options);
    const bool fallback_enabled = options.transport_fallback || env_truthy(std::getenv("LLAMA_PREFILL_KV_TRANSPORT_FALLBACK"));
    std::vector<std::string> mode_order;
    const std::array<const char *, 2> chain = {"rdma", "tcp"};
    if (selected_mode == "auto" || selected_mode == "mixed") {
        mode_order.assign(chain.begin(), chain.end());
    } else if (selected_mode == "disabled" || selected_mode == "rdma" || selected_mode == "tcp") {
        mode_order.push_back(selected_mode);
        if (fallback_enabled && selected_mode != "disabled") {
            for (const char * m : chain) {
                const std::string ms(m);
                if (std::find(mode_order.begin(), mode_order.end(), ms) == mode_order.end()) {
                    mode_order.push_back(ms);
                }
            }
        }
    } else {
        set_error(error, "unknown transport mode '" + selected_mode +
                             "' (allowed: auto,rdma,tcp,mixed,disabled; aliases: tb-direct->rdma, tb-ethernet/ethernet->tcp)");
        return false;
    }

    const bool can_try_multiple_modes = selected_mode == "auto" || selected_mode == "mixed" || fallback_enabled;

    std::string last_transport_error;
    for (const std::string & mode : mode_order) {
        if (mode == "disabled") {
            transfer_ok = true;
            backend_desc = "filesystem";
            endpoint_desc = "<none>";
            resolved_transport_mode = "disabled";
            break;
        }

        const std::vector<std::string> endpoints = resolve_transport_endpoints(options, mode);
        const bool use_tcp_transport = !endpoints.empty();
        if (use_tcp_transport) {
            llama_tb_transfer_options mode_opts = options;
            mode_opts.transport_mode = mode;
            mode_opts.stream_count = std::max<int32_t>(
                1, options.stream_count > 0 ? options.stream_count :
                env_i32("LLAMA_PREFILL_TB_STREAMS", env_i32("LLAMA_PREFILL_KV_STREAMS", 1))
            );
            mode_opts.chunk_bytes = std::max<size_t>(
                1, options.chunk_bytes > 0 ? options.chunk_bytes :
                env_size("LLAMA_PREFILL_TB_STREAM_CHUNK_BYTES", env_size("LLAMA_PREFILL_KV_STREAM_CHUNK_BYTES", 4 * 1024 * 1024))
            );
            mode_opts.max_inflight_bytes = std::max<size_t>(
                (size_t) (1ull << 20),
                options.max_inflight_bytes > 0 ? options.max_inflight_bytes :
                env_size("LLAMA_PREFILL_TB_MAX_INFLIGHT_BYTES", env_size("LLAMA_PREFILL_KV_MAX_INFLIGHT_BYTES", 256 * 1024 * 1024))
            );
            mode_opts.socket_send_buf = options.socket_send_buf > 0 ? options.socket_send_buf :
                env_i32("LLAMA_PREFILL_TB_SOCKET_SEND_BUF", env_i32("LLAMA_PREFILL_KV_SOCKET_SEND_BUF", 0));
            mode_opts.socket_recv_buf = options.socket_recv_buf > 0 ? options.socket_recv_buf :
                env_i32("LLAMA_PREFILL_TB_SOCKET_RECV_BUF", env_i32("LLAMA_PREFILL_KV_SOCKET_RECV_BUF", 0));
            mode_opts.kv_balance = options.kv_balance.empty() ? env_str("LLAMA_PREFILL_KV_BALANCE", "roundrobin") : options.kv_balance;
            mode_opts.kv_bind_addrs = options.kv_bind_addrs.empty() ? env_str("LLAMA_PREFILL_KV_BIND_ADDRS", "") : options.kv_bind_addrs;

            // Read the file once into RAM and send from memory. This keeps
            // both file-based and memory-based callers on the same TCP loop.
            // For the wear-protection path, callers should prefer
            // llama_tb_transport_send_artifact_bytes which skips the file
            // entirely.
            std::vector<uint8_t> file_buffer;
            file_buffer.resize((size_t) payload_bytes);
            std::ifstream src_ifs(src, std::ios::binary);
            if (!src_ifs.is_open()) {
                set_error(error, "failed to open artifact for transport: " + artifact_path);
                return false;
            }
            if (payload_bytes > 0) {
                src_ifs.read(reinterpret_cast<char *>(file_buffer.data()), (std::streamsize) payload_bytes);
                if ((uint64_t) src_ifs.gcount() != payload_bytes) {
                    set_error(error, "short read of artifact: " + artifact_path);
                    return false;
                }
            }
            const std::string src_name = src.filename().string();
            std::string send_err;
            if (send_artifact_over_tcp(file_buffer.data(), file_buffer.size(), src_name,
                                       endpoints, mode_opts, payload_bytes, &resolved_result, &send_err)) {
                chunks_sent = resolved_result.chunks_sent;
                transfer_ok = true;
                backend_desc = resolved_result.transport_backend.empty() ? "tcp" : resolved_result.transport_backend;
                endpoint_desc = endpoints.front();
                resolved_transport_mode = mode;
                break;
            }
            last_transport_error = send_err.empty() ? ("send failed for mode " + mode) : send_err;
            if (!can_try_multiple_modes) {
                set_error(error, last_transport_error);
                return false;
            }
        } else {
            if (!can_try_multiple_modes) {
                set_error(error, "transport mode '" + mode + "' selected but no endpoint resolved");
                return false;
            }
            last_transport_error = "transport mode '" + mode + "' unavailable (no endpoint resolved)";
            continue;
        }
    }

    if (!transfer_ok) {
        if (last_transport_error.empty()) {
            if (selected_mode == "auto") {
                last_transport_error = "auto transport resolution failed: no usable rdma/tcp endpoint";
            } else {
                last_transport_error = "all transport modes failed";
            }
        }
        set_error(error, last_transport_error);
        return false;
    }

    if (backend_desc == "filesystem") {
        const auto transfer_t0 = std::chrono::steady_clock::now();
        uint64_t first_chunk_send_unix_us = 0;
        uint64_t last_chunk_send_unix_us = 0;
        std::vector<std::pair<uint64_t, uint32_t>> chunk_spans;
        if (!options.progressive) {
            first_chunk_send_unix_us = unix_now_us();
            if (!copy_file_binary(src, session / "kv_bulk.bin", error)) {
                return false;
            }
            last_chunk_send_unix_us = unix_now_us();
            chunks_sent = 1;
        } else {
            const size_t chunk_bytes = std::max<size_t>(1, options.chunk_bytes);
            std::ifstream ifs(src, std::ios::binary);
            if (!ifs.is_open()) {
                set_error(error, "failed to open artifact for progressive send: " + artifact_path);
                return false;
            }

            fs::path chunks_dir = session / "kv_chunks";
            fs::create_directories(chunks_dir, ec);
            if (ec) {
                set_error(error, "failed to create progressive chunk dir: " + chunks_dir.string());
                return false;
            }

            std::vector<char> chunk(chunk_bytes);
            uint64_t offset = 0;
            while (ifs) {
                ifs.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
                const std::streamsize n = ifs.gcount();
                if (n <= 0) {
                    break;
                }

                std::ostringstream name;
                name << "chunk_" << chunks_sent << ".bin";
                std::ofstream ofs(chunks_dir / name.str(), std::ios::binary | std::ios::trunc);
                if (!ofs.is_open()) {
                    set_error(error, "failed to open chunk file for write");
                    return false;
                }
                ofs.write(chunk.data(), n);
                if (!ofs.good()) {
                    set_error(error, "failed to write chunk file");
                    return false;
                }
                const uint64_t now_us = unix_now_us();
                if (first_chunk_send_unix_us == 0) {
                    first_chunk_send_unix_us = now_us;
                }
                last_chunk_send_unix_us = now_us;
                chunk_spans.emplace_back(offset, (uint32_t) n);
                offset += (uint64_t) n;
                ++chunks_sent;
            }

            std::ofstream manifest(chunks_dir / "manifest.tsv", std::ios::trunc);
            if (!manifest.is_open()) {
                set_error(error, "failed to write progressive chunk manifest");
                return false;
            }
            manifest << "chunk_index\toffset\tbytes\n";
            for (uint32_t i = 0; i < chunk_spans.size(); ++i) {
                manifest << i << "\t" << chunk_spans[i].first << "\t" << chunk_spans[i].second << "\n";
            }
        }

        resolved_result.session_path = session.string();
        resolved_result.bytes_sent = payload_bytes;
        resolved_result.chunks_sent = chunks_sent;
        resolved_result.progressive = options.progressive;
        resolved_result.stream_count = 1;
        resolved_result.interface_count = 0;
        resolved_result.retransmit_chunks = 0;
        const auto transfer_t1 = std::chrono::steady_clock::now();
        resolved_result.transfer_ms =
            (double) std::chrono::duration_cast<std::chrono::microseconds>(transfer_t1 - transfer_t0).count() / 1000.0;
        resolved_result.throughput_gbps = calc_throughput_gbps(payload_bytes, resolved_result.transfer_ms);
        resolved_result.first_chunk_send_unix_us = first_chunk_send_unix_us;
        resolved_result.last_chunk_send_unix_us = last_chunk_send_unix_us;
    }
    resolved_result.transport_mode = resolved_transport_mode;
    resolved_result.transport_backend = backend_desc;

    {
        std::ofstream meta(session / "session.meta", std::ios::trunc);
        if (!meta.is_open()) {
            set_error(error, "failed to open session.meta for write");
            return false;
        }
        meta << "mode=" << (options.progressive ? "progressive" : "bulk") << "\n";
        meta << "artifact=" << src.filename().string() << "\n";
        meta << "bytes=" << payload_bytes << "\n";
        meta << "chunks=" << chunks_sent << "\n";
        meta << "remote_nodes=" << (int)std::max(1, options.remote_nodes) << "\n";
        meta << "execution_mode=" << (options.execution_mode.empty() ? "coupled" : options.execution_mode) << "\n";
        meta << "transport_mode=" << resolved_transport_mode << "\n";
        meta << "transport_backend=" << backend_desc << "\n";
        meta << "endpoint=" << endpoint_desc << "\n";
    }

    if (env_truthy(std::getenv("LLAMA_PREFILL_TB_LOOPBACK_ACK"))) {
        std::ofstream(session / "ack.ok", std::ios::trunc) << "ok\n";
    }

    if (result) {
        *result = resolved_result;
    }

    return true;
}

static bool tb_transport_send_artifact_impl(const uint8_t *              data,
                                            size_t                       size,
                                            const llama_tb_pull_fn *     pull,
                                            const llama_tb_transfer_options & options,
                                            llama_tb_transfer_result *   result,
                                            std::string *                error) {
    if (!llama_tb_transport_enabled()) {
        set_error(error, "thunderbolt transport disabled (set LLAMA_PREFILL_TB_ENABLE=1)");
        return false;
    }
    if (!pull && data == nullptr && size > 0) {
        set_error(error, "send_artifact_bytes: null buffer with non-zero size");
        return false;
    }

    const fs::path root = options.session_dir.empty() ? fs::path(default_session_root()) : fs::path(options.session_dir);
    const std::string sid = options.session_id.empty() ? "session_default" : options.session_id;
    const fs::path session = root / sid;

    std::error_code ec;
    fs::create_directories(session, ec);
    if (ec) {
        set_error(error, "failed to create transport session dir: " + session.string());
        return false;
    }

    const uint64_t payload_bytes = (uint64_t) size;
    const std::string src_name = pull ? "kv_tier_stream.kva" : "kv_in_memory.kva";

    uint32_t chunks_sent = 0;
    llama_tb_transfer_result resolved_result;
    std::string backend_desc = "filesystem";
    std::string endpoint_desc = "<none>";
    std::string resolved_transport_mode = "disabled";
    bool transfer_ok = false;

    const std::string selected_mode = resolve_transport_mode(options);
    const bool fallback_enabled = options.transport_fallback || env_truthy(std::getenv("LLAMA_PREFILL_KV_TRANSPORT_FALLBACK"));
    std::vector<std::string> mode_order;
    const std::array<const char *, 2> chain = {"rdma", "tcp"};
    if (selected_mode == "auto" || selected_mode == "mixed") {
        mode_order.assign(chain.begin(), chain.end());
    } else if (selected_mode == "disabled" || selected_mode == "rdma" || selected_mode == "tcp") {
        mode_order.push_back(selected_mode);
        if (fallback_enabled && selected_mode != "disabled") {
            for (const char * m : chain) {
                const std::string ms(m);
                if (std::find(mode_order.begin(), mode_order.end(), ms) == mode_order.end()) {
                    mode_order.push_back(ms);
                }
            }
        }
    } else {
        set_error(error, "unknown transport mode '" + selected_mode +
                             "' (allowed: auto,rdma,tcp,mixed,disabled)");
        return false;
    }

    const bool can_try_multiple_modes = selected_mode == "auto" || selected_mode == "mixed" || fallback_enabled;
    std::string last_transport_error;

    for (const std::string & mode : mode_order) {
        if (mode == "disabled") {
            // Memory-only mode with disabled transport is not useful; require
            // a real network endpoint. Fall through to fail with a clear error.
            last_transport_error = "send_artifact_bytes does not support transport_mode=disabled (file-based filesystem fallback elided to avoid SSD wear)";
            continue;
        }

        const std::vector<std::string> endpoints = resolve_transport_endpoints(options, mode);
        if (endpoints.empty()) {
            if (!can_try_multiple_modes) {
                set_error(error, "transport mode '" + mode + "' selected but no endpoint resolved");
                return false;
            }
            last_transport_error = "transport mode '" + mode + "' unavailable (no endpoint resolved)";
            continue;
        }

        llama_tb_transfer_options mode_opts = options;
        mode_opts.transport_mode = mode;
        mode_opts.stream_count = std::max<int32_t>(
            1, options.stream_count > 0 ? options.stream_count :
            env_i32("LLAMA_PREFILL_TB_STREAMS", env_i32("LLAMA_PREFILL_KV_STREAMS", 1)));
        mode_opts.chunk_bytes = std::max<size_t>(
            1, options.chunk_bytes > 0 ? options.chunk_bytes :
            env_size("LLAMA_PREFILL_TB_STREAM_CHUNK_BYTES", env_size("LLAMA_PREFILL_KV_STREAM_CHUNK_BYTES", 4 * 1024 * 1024)));
        mode_opts.max_inflight_bytes = std::max<size_t>(
            (size_t) (1ull << 20),
            options.max_inflight_bytes > 0 ? options.max_inflight_bytes :
            env_size("LLAMA_PREFILL_TB_MAX_INFLIGHT_BYTES", env_size("LLAMA_PREFILL_KV_MAX_INFLIGHT_BYTES", 256 * 1024 * 1024)));
        mode_opts.socket_send_buf = options.socket_send_buf > 0 ? options.socket_send_buf :
            env_i32("LLAMA_PREFILL_TB_SOCKET_SEND_BUF", env_i32("LLAMA_PREFILL_KV_SOCKET_SEND_BUF", 0));
        mode_opts.socket_recv_buf = options.socket_recv_buf > 0 ? options.socket_recv_buf :
            env_i32("LLAMA_PREFILL_TB_SOCKET_RECV_BUF", env_i32("LLAMA_PREFILL_KV_SOCKET_RECV_BUF", 0));
        mode_opts.kv_balance = options.kv_balance.empty() ? env_str("LLAMA_PREFILL_KV_BALANCE", "roundrobin") : options.kv_balance;
        mode_opts.kv_bind_addrs = options.kv_bind_addrs.empty() ? env_str("LLAMA_PREFILL_KV_BIND_ADDRS", "") : options.kv_bind_addrs;

        if (pull) {
            mode_opts.stream_count = 1;
        }
        std::string send_err;
        if (send_artifact_over_tcp(data, size, src_name,
                                   endpoints, mode_opts, payload_bytes,
                                   &resolved_result, &send_err, pull)) {
            chunks_sent = resolved_result.chunks_sent;
            transfer_ok = true;
            backend_desc = resolved_result.transport_backend.empty() ? "tcp" : resolved_result.transport_backend;
            endpoint_desc = endpoints.front();
            resolved_transport_mode = mode;
            break;
        }
        last_transport_error = send_err.empty() ? ("send failed for mode " + mode) : send_err;
        if (!can_try_multiple_modes) {
            set_error(error, last_transport_error);
            return false;
        }
    }

    if (!transfer_ok) {
        set_error(error, last_transport_error.empty() ? std::string("all transport modes failed") : last_transport_error);
        return false;
    }

    resolved_result.transport_mode = resolved_transport_mode;
    resolved_result.transport_backend = backend_desc;

    if (env_truthy(std::getenv("LLAMA_PREFILL_TB_LOOPBACK_ACK"))) {
        std::ofstream(session / "ack.ok", std::ios::trunc) << "ok\n";
    }

    if (result) {
        *result = resolved_result;
    }
    (void) chunks_sent;
    (void) endpoint_desc;
    return true;
}

bool llama_tb_transport_send_artifact_bytes(const uint8_t *              data,
                                             size_t                       size,
                                             const llama_tb_transfer_options & options,
                                             llama_tb_transfer_result *   result,
                                             std::string *                error) {
    return tb_transport_send_artifact_impl(data, size, nullptr, options, result, error);
}

bool llama_tb_transport_query_prefix(const llama_tb_transfer_options & options,
                                     const int32_t *              tokens,
                                     size_t                       n_tokens,
                                     int                          timeout_ms,
                                     uint32_t &                   cached_len,
                                     std::string *                error) {
    cached_len = 0;
    if (!llama_tb_transport_enabled()) {
        set_error(error, "thunderbolt transport disabled");
        return false;
    }
    llama_tb_transfer_options opts = options;
    opts.transport_mode = "tcp";
    const std::vector<std::string> endpoints = resolve_transport_endpoints(opts, "tcp");
    if (endpoints.empty()) {
        set_error(error, "no tcp endpoint resolved for the prefix query");
        return false;
    }
    tb_socket_t fd = connect_endpoint(endpoints.front(), "", 0, 0, error);
    if (fd == TB_SOCKET_INVALID) {
        return false;
    }
    const uint64_t wire_sid = hash_session_id(options.session_id.empty() ? std::string("prefix_query") : options.session_id);
    const std::string role_payload = "node_role=prefill-server;capabilities=bulk,progressive,multistream,delta";
    uint64_t seq = 0;
    bool ok = send_tbp_frame(fd, TBP_MSG_HELLO, 0, wire_sid, 0, seq++,
                             reinterpret_cast<const uint8_t *>(role_payload.data()), role_payload.size(), error) &&
              send_tbp_frame(fd, TBP_MSG_QUERY_PREFIX_CACHE, 0, wire_sid, 0, seq++,
                             reinterpret_cast<const uint8_t *>(tokens), n_tokens * sizeof(int32_t), error);
    tbp_rx_frame rx;
    if (ok) {
        ok = recv_tbp_frame_timeout(fd, timeout_ms, &rx, error);
    }
    TB_SOCKET_CLOSE(fd);
    if (!ok) {
        return false;
    }
    if (rx.msg_type != TBP_MSG_PREFIX_CACHE_REPLY || rx.payload.size() < 5) {
        set_error(error, "unexpected prefix query reply (msg " + std::to_string(rx.msg_type) + ", " +
                         std::to_string(rx.payload.size()) + " bytes)");
        return false;
    }
    const uint8_t hit = rx.payload[0];
    const uint32_t len = (uint32_t) rx.payload[1] | ((uint32_t) rx.payload[2] << 8) |
                         ((uint32_t) rx.payload[3] << 16) | ((uint32_t) rx.payload[4] << 24);
    cached_len = hit ? len : 0u;
    return true;
}

bool llama_tb_transport_send_artifact_stream(uint64_t                     total_bytes,
                                             const llama_tb_pull_fn &     pull,
                                             const llama_tb_transfer_options & options,
                                             llama_tb_transfer_result *   result,
                                             std::string *                error) {
    if (!pull) {
        set_error(error, "send_artifact_stream: no pull callback");
        return false;
    }
    return tb_transport_send_artifact_impl(nullptr, (size_t) total_bytes, &pull, options, result, error);
}

void llama_tb_transport_weight_relay_copy(const uint8_t *src, uint8_t *dst, size_t size) {
    if (size == 0 || src == dst) {
        return;
    }

    // Explicit two-hop copy (src -> staging -> dst) to model relay overhead deterministically.
    std::vector<uint8_t> staging(size);
    std::copy_n(src, size, staging.data());
    std::copy_n(staging.data(), size, dst);
}

bool llama_tb_transport_fetch_weight_chunk(const uint8_t * src, size_t size, std::vector<uint8_t> & dst, std::string * error) {
    dst.clear();
    if (size == 0) {
        return true;
    }
    if (!src) {
        set_error(error, "null source pointer for weight chunk fetch");
        return false;
    }

    const char * mode_env = std::getenv("LLAMA_PREFILL_TB_WEIGHT_SOURCE");
    const std::string mode = mode_env && mode_env[0] != '\0' ? std::string(mode_env) : "relay";
    const int latency_us =
        std::max(0, std::atoi(std::getenv("LLAMA_PREFILL_TB_WEIGHT_LATENCY_US") ?
                              std::getenv("LLAMA_PREFILL_TB_WEIGHT_LATENCY_US") :
                              "0"));
    auto maybe_latency = [&]() {
        if (latency_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(latency_us));
        }
    };

    dst.resize(size);

    if (mode == "direct") {
        std::copy_n(src, size, dst.data());
        maybe_latency();
        return true;
    }

    if (mode == "remote_file") {
        const char * root_env = std::getenv("LLAMA_PREFILL_TB_WEIGHT_DIR");
        const fs::path root = (root_env && root_env[0] != '\0') ? fs::path(root_env) : fs::path("/tmp/llama_tb_weight_cache");
        std::error_code ec;
        fs::create_directories(root, ec);
        if (ec) {
            set_error(error, "failed to create remote weight dir: " + root.string());
            return false;
        }

        const uint32_t crc = llama_kv_artifact_crc32(src, size);
        const fs::path slab = root / ("chunk_" + std::to_string(crc) + "_" + std::to_string(size) + ".bin");
        if (!fs::exists(slab)) {
            const bool allow_populate = env_truthy(std::getenv("LLAMA_PREFILL_TB_WEIGHT_POPULATE"));
            if (!allow_populate) {
                set_error(error, "remote weight chunk missing (set LLAMA_PREFILL_TB_WEIGHT_POPULATE=1 to seed cache): " +
                                   slab.string());
                return false;
            }
            std::ofstream ofs(slab, std::ios::binary | std::ios::trunc);
            if (!ofs.is_open()) {
                set_error(error, "failed to seed remote weight chunk: " + slab.string());
                return false;
            }
            ofs.write(reinterpret_cast<const char *>(src), (std::streamsize) size);
            if (!ofs.good()) {
                set_error(error, "failed to write seeded remote weight chunk: " + slab.string());
                return false;
            }
        }

        std::ifstream ifs(slab, std::ios::binary);
        if (!ifs.is_open()) {
            set_error(error, "failed to open remote weight chunk: " + slab.string());
            return false;
        }
        ifs.read(reinterpret_cast<char *>(dst.data()), (std::streamsize) size);
        if (!ifs || (size_t) ifs.gcount() != size) {
            set_error(error, "failed to read full remote weight chunk: " + slab.string());
            return false;
        }
        maybe_latency();
        return true;
    }

    // Default mode is relay: explicit two-hop copy that can later be replaced
    // with transport-backed fetch without touching the upload pipeline.
    llama_tb_transport_weight_relay_copy(src, dst.data(), size);
    maybe_latency();
    return true;
}

// Send raw prompt tokens for local prefill on the decode receiver.
// Writes token data to a temp file and uses the existing TCP transport.
// The receiver detects non-KV data (missing KVARTIF1 magic) and treats
// it as raw token IDs for local prefill.
bool llama_tb_transport_send_prompt(const int32_t *               tokens,
                                    int32_t                       n_tokens,
                                    const llama_tb_transfer_options & options,
                                    llama_tb_transfer_result *    result,
                                    std::string *                 error) {
    if (n_tokens <= 0 || !tokens) {
        set_error(error, "no tokens to send");
        return false;
    }

    // Write token IDs to a temporary file
    const std::string tmp_path = std::string(std::getenv("TMP") ? std::getenv("TMP") : "/tmp") +
                                 "/llama_prompt_tokens_" + std::to_string(unix_now_us()) + ".bin";
    {
        std::ofstream ofs(tmp_path, std::ios::binary);
        if (!ofs.is_open()) {
            set_error(error, "failed to create temp prompt file: " + tmp_path);
            return false;
        }
        ofs.write(reinterpret_cast<const char *>(tokens), n_tokens * sizeof(int32_t));
    }

    // Use the existing transport with the temp file
    bool ok = llama_tb_transport_send_artifact(tmp_path, options, result, error);

    // Clean up temp file
    std::error_code ec;
    fs::remove(tmp_path, ec);

    return ok;
}

// ---------------------------------------------------------------------------
// Mac-side: PROMPT-DIRECT KV-handshake control socket server (one round-trip)
//
// This function binds a listen socket on bind_host:control_port, accepts one
// RTX connection, reads TBP_MSG_QUERY_PREFIX_CACHE, calls lookup_fn, sends
// TBP_MSG_PREFIX_CACHE_REPLY, and returns. Designed to be called in a loop
// from a background thread in decode-receiver/main.cpp.
// ---------------------------------------------------------------------------
bool llama_tb_transport_control_server_accept_one(
    const std::string &             bind_host,
    int                             control_port,
    llama_tb_prefix_cache_lookup_fn lookup_fn,
    std::string *                   error)
{
#ifdef _WIN32
    if (!tb_winsock_init()) {
        set_error(error, "winsock init failed");
        return false;
    }
#endif

    // Create listen socket
    tb_socket_t listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd == TB_SOCKET_INVALID) {
        set_error(error, "control server: socket creation failed");
        return false;
    }

#ifndef _WIN32
    {
        int opt = 1;
        ::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    }
#else
    {
        const char opt = 1;
        ::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    }
#endif

    // Bind
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t) control_port);
    if (bind_host.empty() || bind_host == "0.0.0.0") {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        ::inet_pton(AF_INET, bind_host.c_str(), &addr.sin_addr);
    }
    if (::bind(listen_fd, (struct sockaddr *) &addr, sizeof(addr)) != 0) {
        set_error(error, "control server: bind failed on port " + std::to_string(control_port));
        TB_SOCKET_CLOSE(listen_fd);
        return false;
    }
    if (::listen(listen_fd, 4) != 0) {
        set_error(error, "control server: listen failed");
        TB_SOCKET_CLOSE(listen_fd);
        return false;
    }

    // Accept one connection (blocking, with 5s timeout between sessions)
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(listen_fd, &rfds);
    struct timeval tv = {};
    tv.tv_sec  = 5;
    tv.tv_usec = 0;
#ifndef _WIN32
    const int sel = ::select((int) listen_fd + 1, &rfds, nullptr, nullptr, &tv);
#else
    const int sel = ::select(0, &rfds, nullptr, nullptr, &tv);
#endif
    if (sel <= 0) {
        // Timeout (no RTX connected) — normal; caller loops back
        TB_SOCKET_CLOSE(listen_fd);
        set_error(error, sel == 0 ? "control server: accept timeout" : "control server: accept select error");
        return false;
    }
    tb_socket_t client_fd = ::accept(listen_fd, nullptr, nullptr);
    TB_SOCKET_CLOSE(listen_fd);
    if (client_fd == TB_SOCKET_INVALID) {
        set_error(error, "control server: accept failed");
        return false;
    }

    // Read TBP_MSG_QUERY_PREFIX_CACHE frame (2 s timeout)
    tbp_rx_frame rx = {};
    std::string recv_err;
    if (!recv_tbp_frame_timeout(client_fd, 2000, &rx, &recv_err)) {
        fprintf(stderr, "[pd-ctrl] Failed to receive query frame: %s\n", recv_err.c_str());
        TB_SOCKET_CLOSE(client_fd);
        return false;
    }

    if (rx.msg_type != TBP_MSG_QUERY_PREFIX_CACHE) {
        fprintf(stderr, "[pd-ctrl] Unexpected msg_type=%u from RTX (expected %u)\n",
                (unsigned) rx.msg_type, (unsigned) TBP_MSG_QUERY_PREFIX_CACHE);
        TB_SOCKET_CLOSE(client_fd);
        return false;
    }

    // Parse query payload: prefix_hash(8) + prefix_len(4) + model_fp(8) = 20 bytes
    if (rx.payload.size() < 20) {
        fprintf(stderr, "[pd-ctrl] Query payload too short: %zu bytes\n", rx.payload.size());
        TB_SOCKET_CLOSE(client_fd);
        return false;
    }
    const uint8_t * qp = rx.payload.data();
    uint64_t prefix_hash = 0;
    for (int i = 0; i < 8; ++i) { prefix_hash |= ((uint64_t) qp[i]) << (8 * i); }
    const uint32_t prefix_len = (uint32_t) qp[8] | ((uint32_t) qp[9] << 8) | ((uint32_t) qp[10] << 16) | ((uint32_t) qp[11] << 24);
    uint64_t model_fp = 0;
    for (int i = 0; i < 8; ++i) { model_fp |= ((uint64_t) qp[12 + i]) << (8 * i); }

    fprintf(stderr, "[pd-ctrl] Query: hash=0x%llx prefix_len=%u model_fp=0x%llx\n",
            (unsigned long long) prefix_hash, prefix_len, (unsigned long long) model_fp);

    // Call the lookup function provided by decode-receiver
    uint32_t out_cached_len = 0;
    uint64_t out_session_id = 0;
    bool hit = false;
    if (lookup_fn) {
        hit = lookup_fn(prefix_hash, prefix_len, model_fp, out_cached_len, out_session_id);
    }

    // Build TBP_MSG_PREFIX_CACHE_REPLY payload: hit(1) + cached_len(4) + session_id(8) = 13 bytes
    std::vector<uint8_t> reply_payload;
    reply_payload.reserve(13);
    reply_payload.push_back(hit ? 1u : 0u);
    append_u32_le(reply_payload, out_cached_len);
    append_u64_le(reply_payload, out_session_id);

    std::string send_err;
    if (!send_tbp_frame(client_fd, TBP_MSG_PREFIX_CACHE_REPLY, 0, 0, 0, 0,
                        reply_payload.data(), reply_payload.size(), &send_err)) {
        fprintf(stderr, "[pd-ctrl] Failed to send reply: %s\n", send_err.c_str());
        TB_SOCKET_CLOSE(client_fd);
        return false;
    }

    TB_SOCKET_CLOSE(client_fd);
    fprintf(stderr, "[pd-ctrl] Replied: hit=%d cached_len=%u session_id=0x%llx\n",
            (int) hit, out_cached_len, (unsigned long long) out_session_id);
    return true;
}
