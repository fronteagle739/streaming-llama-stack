#include "llama-kv-receiver.h"
#include "llama-kv-artifact.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef _WIN32
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <sys/socket.h>
#  include <unistd.h>
#  define RECV_SOCKET_CLOSE(fd) ::close(fd)
typedef int recv_socket_t;
static constexpr recv_socket_t RECV_SOCKET_INVALID = -1;
#else
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "Ws2_32.lib")
#  define RECV_SOCKET_CLOSE(fd) ::closesocket(fd)
typedef SOCKET recv_socket_t;
static const recv_socket_t RECV_SOCKET_INVALID = INVALID_SOCKET;

static bool recv_winsock_init() {
    static bool done = false;
    if (!done) {
        WSADATA wsa;
        done = (WSAStartup(MAKEWORD(2, 2), &wsa) == 0);
    }
    return done;
}
#endif

namespace {

// TBP1 wire format constants (must match llama-tb-transport.cpp)
constexpr uint32_t TBP_MAGIC             = 0x54425031u;
constexpr size_t   TBP_HEADER_SIZE       = 52;
constexpr uint16_t TBP_MSG_HELLO         = 1;
constexpr uint16_t TBP_MSG_SESSION_START = 3;
constexpr uint16_t TBP_MSG_KV_SEG_BEGIN  = 7;
constexpr uint16_t TBP_MSG_KV_CHUNK      = 8;
constexpr uint16_t TBP_MSG_KV_SEG_END    = 9;
constexpr uint16_t TBP_MSG_KV_DONE       = 12;

static uint16_t rd_u16(const uint8_t * p) { return (uint16_t)p[0] | ((uint16_t)p[1] << 8); }
static uint32_t rd_u32(const uint8_t * p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

void set_error(std::string * err, std::string msg) {
    if (err) *err = std::move(msg);
}

bool recv_all(recv_socket_t fd, uint8_t * dst, size_t size, int timeout_ms, std::string * error) {
    size_t off = 0;
    while (off < size) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(fd, &rfds);
        struct timeval tv = {};
        tv.tv_sec  = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
#ifndef _WIN32
        int sel = select((int)fd + 1, &rfds, nullptr, nullptr, &tv);
#else
        int sel = select(0, &rfds, nullptr, nullptr, &tv);
#endif
        if (sel <= 0) {
            set_error(error, sel == 0 ? "recv timeout" : "recv select error");
            return false;
        }
#ifndef _WIN32
        ssize_t n = ::recv(fd, dst + off, size - off, 0);
#else
        int n = ::recv(fd, (char*)(dst + off), (int)(size - off), 0);
#endif
        if (n <= 0) {
            set_error(error, "recv failed or connection closed");
            return false;
        }
        off += (size_t)n;
    }
    return true;
}

recv_socket_t create_listen(const std::string & host, int port, std::string * error) {
#ifdef _WIN32
    if (!recv_winsock_init()) {
        set_error(error, "winsock init failed");
        return RECV_SOCKET_INVALID;
    }
#endif
    recv_socket_t fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == RECV_SOCKET_INVALID) {
        set_error(error, "socket creation failed");
        return RECV_SOCKET_INVALID;
    }
#ifndef _WIN32
    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#else
    const char opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t)port);
    inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        set_error(error, "bind failed on " + host + ":" + std::to_string(port));
        RECV_SOCKET_CLOSE(fd);
        return RECV_SOCKET_INVALID;
    }
    if (listen(fd, 1) != 0) {
        set_error(error, "listen failed");
        RECV_SOCKET_CLOSE(fd);
        return RECV_SOCKET_INVALID;
    }
    return fd;
}

recv_socket_t accept_timeout(recv_socket_t listen_fd, int timeout_ms, std::string * error) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(listen_fd, &rfds);
    struct timeval tv = {};
    struct timeval * tv_ptr = nullptr;
    if (timeout_ms > 0) {
        tv.tv_sec  = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        tv_ptr = &tv;
    }
    // timeout_ms <= 0 → tv_ptr is NULL → select blocks indefinitely
#ifndef _WIN32
    int ret = select((int)listen_fd + 1, &rfds, nullptr, nullptr, tv_ptr);
#else
    int ret = select(0, &rfds, nullptr, nullptr, tv_ptr);
#endif
    if (ret <= 0) {
        set_error(error, ret == 0 ? "accept timeout" : "accept select error");
        return RECV_SOCKET_INVALID;
    }
    recv_socket_t client = accept(listen_fd, nullptr, nullptr);
    if (client == RECV_SOCKET_INVALID) {
        set_error(error, "accept failed");
    }
    return client;
}

}  // namespace

bool llama_kv_receiver_accept_artifact(
    const llama_kv_receiver_config & config,
    std::vector<uint8_t> &           artifact_out,
    llama_kv_receiver_result *       result,
    std::string *                    error) {

    artifact_out.clear();
    if (result) *result = {};

    // Listen
    recv_socket_t listen_fd = create_listen(config.bind_host, config.port, error);
    if (listen_fd == RECV_SOCKET_INVALID) return false;
    fprintf(stderr, "[kv-receiver] Bound and listening on %s:%d\n",
            config.bind_host.c_str(), config.port);
    fflush(stderr);

    // Accept
    recv_socket_t client_fd = accept_timeout(listen_fd, config.accept_timeout_ms, error);
    RECV_SOCKET_CLOSE(listen_fd);
    if (client_fd == RECV_SOCKET_INVALID) return false;

    auto cleanup = [&]() { RECV_SOCKET_CLOSE(client_fd); };

    auto t0 = std::chrono::steady_clock::now();

    // State machine: receive TBP1 frames
    enum { WAIT_HELLO, WAIT_SESSION, WAIT_SEG_BEGIN, RECV_CHUNKS, WAIT_SEG_END, WAIT_DONE, COMPLETE } state = WAIT_HELLO;

    uint64_t expected_payload_bytes = 0;
    std::vector<uint8_t> reassembled;
    uint32_t chunks_received = 0;
    std::string session_info;

    const int frame_timeout_ms = config.recv_timeout_ms;

    while (state != COMPLETE) {
        // Read 52-byte frame header
        uint8_t hdr[TBP_HEADER_SIZE];
        if (!recv_all(client_fd, hdr, TBP_HEADER_SIZE, frame_timeout_ms, error)) {
            cleanup();
            return false;
        }

        uint32_t magic = rd_u32(hdr + 0);
        if (magic != TBP_MAGIC) {
            set_error(error, "invalid TBP1 magic");
            cleanup();
            return false;
        }

        uint16_t msg_type     = rd_u16(hdr + 8);
        uint32_t payload_size = rd_u32(hdr + 16);

        // Read payload
        std::vector<uint8_t> frame_payload(payload_size);
        if (payload_size > 0) {
            if (!recv_all(client_fd, frame_payload.data(), payload_size, frame_timeout_ms, error)) {
                cleanup();
                return false;
            }

            // CRC check
            if (config.verify_crc && payload_size > 0) {
                uint32_t expected_crc = rd_u32(hdr + 48);
                uint32_t actual_crc   = llama_kv_artifact_crc32(frame_payload.data(), payload_size);
                if (expected_crc != actual_crc) {
                    set_error(error, "TBP1 frame payload CRC mismatch");
                    cleanup();
                    return false;
                }
            }
        }

        // Dispatch
        switch (msg_type) {
            case TBP_MSG_HELLO:
                if (state == WAIT_HELLO) state = WAIT_SESSION;
                break;

            case TBP_MSG_SESSION_START: {
                if (state == WAIT_SESSION) {
                    // Parse session payload for expected bytes
                    session_info.assign(frame_payload.begin(), frame_payload.end());
                    // Look for "bytes=N" in the key=value payload
                    auto pos = session_info.find("bytes=");
                    if (pos != std::string::npos) {
                        expected_payload_bytes = (uint64_t) std::strtoull(
                            session_info.c_str() + pos + 6, nullptr, 10);
                        reassembled.reserve((size_t) expected_payload_bytes);
                    }
                    state = WAIT_SEG_BEGIN;
                }
                break;
            }

            case TBP_MSG_KV_SEG_BEGIN:
                if (state == WAIT_SEG_BEGIN) state = RECV_CHUNKS;
                break;

            case TBP_MSG_KV_CHUNK:
                if (state == RECV_CHUNKS) {
                    reassembled.insert(reassembled.end(),
                                       frame_payload.begin(), frame_payload.end());
                    chunks_received++;
                }
                break;

            case TBP_MSG_KV_SEG_END:
                if (state == RECV_CHUNKS) state = WAIT_DONE;
                break;

            case TBP_MSG_KV_DONE:
                state = COMPLETE;
                break;

            default:
                // Skip unknown frame types
                break;
        }
    }

    cleanup();

    auto t1 = std::chrono::steady_clock::now();
    double ms = (double) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;

    artifact_out = std::move(reassembled);

    if (result) {
        result->bytes_received  = artifact_out.size();
        result->chunks_received = chunks_received;
        result->receive_ms      = ms;
        result->session_info    = session_info;
        // Detect prompt-direct mode from session metadata (mode=prompt)
        result->is_prompt_mode  = (session_info.find("mode=prompt") != std::string::npos);
        if (ms > 0) {
            result->throughput_gbps = ((double) artifact_out.size() * 8.0) / (ms * 1e6);
        }
    }

    return true;
}
