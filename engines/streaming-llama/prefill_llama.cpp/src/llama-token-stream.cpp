#include "llama-token-stream.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>

// ---------------------------------------------------------------------------
// Platform socket includes
// ---------------------------------------------------------------------------

#ifndef _WIN32
#  include <arpa/inet.h>
#  include <errno.h>
#  include <fcntl.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <poll.h>
#  include <sys/socket.h>
#  include <unistd.h>
#  define SOCKET_CLOSE(fd) ::close(fd)
#  define SOCKET_INVALID   (-1)
typedef int socket_fd_t;
#else
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "Ws2_32.lib")
#  define SOCKET_CLOSE(fd) ::closesocket(fd)
#  define SOCKET_INVALID   ((socket_fd_t)INVALID_SOCKET)
typedef SOCKET socket_fd_t;

static bool winsock_ensure_init() {
    static bool done = false;
    if (!done) {
        WSADATA wsa;
        done = (WSAStartup(MAKEWORD(2, 2), &wsa) == 0);
    }
    return done;
}
#endif

// ---------------------------------------------------------------------------
// JSON helpers — hand-rolled, no dependency.
// ---------------------------------------------------------------------------

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

static std::string json_unescape(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            switch (s[i + 1]) {
                case '"':  out += '"';  i++; break;
                case '\\': out += '\\'; i++; break;
                case 'n':  out += '\n'; i++; break;
                case 'r':  out += '\r'; i++; break;
                case 't':  out += '\t'; i++; break;
                default:   out += s[i]; break;
            }
        } else {
            out += s[i];
        }
    }
    return out;
}

// Extract a string value for a key from a JSON object string.
// Returns "" if not found.  Very naive — good enough for our fixed format.
static std::string json_get_str(const std::string & json, const std::string & key) {
    std::string needle = "\"" + key + "\":\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos += needle.size();
    std::string val;
    for (size_t i = pos; i < json.size(); i++) {
        if (json[i] == '\\' && i + 1 < json.size()) {
            val += json[i];
            val += json[i + 1];
            i++;
        } else if (json[i] == '"') {
            break;
        } else {
            val += json[i];
        }
    }
    return json_unescape(val);
}

// Extract a numeric value (int64_t) for a key.
static bool json_get_int(const std::string & json, const std::string & key, int64_t & out) {
    std::string needle = "\"" + key + "\":";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return false;
    pos += needle.size();
    // skip whitespace
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    char * end = nullptr;
    long long val = strtoll(json.c_str() + pos, &end, 10);
    if (end == json.c_str() + pos) return false;
    out = (int64_t) val;
    return true;
}

static bool json_get_double(const std::string & json, const std::string & key, double & out) {
    std::string needle = "\"" + key + "\":";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return false;
    pos += needle.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    char * end = nullptr;
    double val = strtod(json.c_str() + pos, &end);
    if (end == json.c_str() + pos) return false;
    out = val;
    return true;
}

static bool json_get_bool(const std::string & json, const std::string & key, bool & out) {
    std::string needle = "\"" + key + "\":";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return false;
    pos += needle.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    if (json.compare(pos, 4, "true") == 0)  { out = true;  return true; }
    if (json.compare(pos, 5, "false") == 0) { out = false; return true; }
    return false;
}

// ---------------------------------------------------------------------------
// JSON serialization
// ---------------------------------------------------------------------------

static double decode_interval_rate(
        const std::vector<int64_t> & timestamps_us,
        int32_t begin_interval,
        int32_t end_interval) {
    if (begin_interval < 0 || end_interval <= begin_interval ||
        (size_t) end_interval >= timestamps_us.size()) {
        return 0.0;
    }
    const int64_t elapsed_us = timestamps_us[(size_t) end_interval] -
                               timestamps_us[(size_t) begin_interval];
    if (elapsed_us <= 0) {
        return 0.0;
    }
    return (double) (end_interval - begin_interval) * 1000000.0 / (double) elapsed_us;
}

llama_sustained_decode_timing llama_sustained_decode_timing_from_timestamps(
        const std::vector<int64_t> & emitted_timestamps_us) {
    llama_sustained_decode_timing timing = {};
    timing.emitted_tokens = (int32_t) emitted_timestamps_us.size();
    if (emitted_timestamps_us.size() < 2) {
        return timing;
    }

    timing.decode_intervals = timing.emitted_tokens - 1;
    const int64_t elapsed_us = emitted_timestamps_us.back() - emitted_timestamps_us.front();
    if (elapsed_us > 0) {
        timing.elapsed_ms = (double) elapsed_us / 1000.0;
        timing.tok_s = (double) timing.decode_intervals * 1000000.0 / (double) elapsed_us;
    }

    const int32_t first_end = (timing.decode_intervals + 2) / 3;
    const int32_t middle_end = (2 * timing.decode_intervals + 2) / 3;
    timing.early_tok_s = decode_interval_rate(emitted_timestamps_us, 0, first_end);
    timing.middle_tok_s = decode_interval_rate(emitted_timestamps_us, first_end, middle_end);
    timing.late_tok_s = decode_interval_rate(
        emitted_timestamps_us, middle_end, timing.decode_intervals);
    return timing;
}

std::string llama_token_msg_to_json(const llama_token_msg & msg) {
    std::ostringstream ss;
    if (msg.done) {
        ss << "{\"done\":true"
           << ",\"tokens_generated\":" << msg.tokens_generated
           << ",\"decode_tok_s\":" << msg.decode_tok_s
           << ",\"decode_elapsed_ms\":" << msg.decode_elapsed_ms
           << ",\"decode_intervals\":" << msg.decode_intervals
           << ",\"decode_tok_s_early\":" << msg.decode_tok_s_early
           << ",\"decode_tok_s_middle\":" << msg.decode_tok_s_middle
           << ",\"decode_tok_s_late\":" << msg.decode_tok_s_late
           << ",\"stop_reason\":" << msg.stop_reason;
        if (!msg.stopping_word.empty()) {
            ss << ",\"stopping_word\":\"" << json_escape(msg.stopping_word) << "\"";
        }
        if (msg.retained_token_count > 0) {
            ss << ",\"retained_token_count\":" << msg.retained_token_count
               << ",\"retained_prefix_hash\":"
               << (msg.retained_prefix_hash & 0x7fffffffffffffffULL);
        }
        ss << "}";
    } else {
        ss << "{\"token_id\":" << msg.token_id
           << ",\"text\":\"" << json_escape(msg.text) << "\""
           << ",\"pos\":" << msg.pos
           << ",\"timestamp_us\":" << msg.timestamp_us
           << ",\"dropped_count\":" << msg.dropped_count
           << "}";
    }
    return ss.str();
}

bool llama_token_msg_from_json(const std::string & json, llama_token_msg & msg) {
    msg = {};

    bool is_done = false;
    if (json_get_bool(json, "done", is_done) && is_done) {
        msg.done = true;
        int64_t tg = 0;
        if (json_get_int(json, "tokens_generated", tg)) msg.tokens_generated = (int32_t) tg;
        json_get_double(json, "decode_tok_s", msg.decode_tok_s);
        json_get_double(json, "decode_elapsed_ms", msg.decode_elapsed_ms);
        int64_t di = 0;
        if (json_get_int(json, "decode_intervals", di)) msg.decode_intervals = (int32_t) di;
        json_get_double(json, "decode_tok_s_early", msg.decode_tok_s_early);
        json_get_double(json, "decode_tok_s_middle", msg.decode_tok_s_middle);
        json_get_double(json, "decode_tok_s_late", msg.decode_tok_s_late);
        int64_t sr = 0;
        if (json_get_int(json, "stop_reason", sr)) msg.stop_reason = (int32_t) sr;
        msg.stopping_word = json_get_str(json, "stopping_word");
        // Option B (cross-turn retained-KV): the Mac reports the token prefix it now holds in
        // its Phase-74 retained cache. Purely additive — absent on a legacy receiver (fields stay 0).
        int64_t rtc = 0;
        if (json_get_int(json, "retained_token_count", rtc) && rtc > 0) {
            msg.retained_token_count = (uint32_t) rtc;
            int64_t rph = 0;
            // Same 64-bit-through-int64 convention the handshake uses for model_fingerprint.
            if (json_get_int(json, "retained_prefix_hash", rph)) {
                msg.retained_prefix_hash = (uint64_t) rph & 0x7fffffffffffffffULL;
            }
        }
        return true;
    }

    int64_t v = 0;
    if (json_get_int(json, "token_id", v)) msg.token_id = (int32_t) v;
    msg.text = json_get_str(json, "text");
    if (json_get_int(json, "pos", v)) msg.pos = (int32_t) v;
    if (json_get_int(json, "timestamp_us", v)) msg.timestamp_us = v;
    if (json_get_int(json, "dropped_count", v)) msg.dropped_count = (int32_t) v;

    return true;
}

std::string llama_token_handshake_to_json(const llama_token_handshake & hs) {
    std::ostringstream ss;
    ss << "{\"version\":" << hs.version
       << ",\"status\":\"" << json_escape(hs.status) << "\""
       << ",\"model_fingerprint\":" << hs.model_fingerprint
       << ",\"token_count\":" << hs.token_count
       << "}";
    return ss.str();
}

bool llama_token_handshake_from_json(const std::string & json, llama_token_handshake & hs) {
    hs = {};
    int64_t v = 0;
    if (json_get_int(json, "version", v)) hs.version = (int32_t) v;
    hs.status = json_get_str(json, "status");
    if (json_get_int(json, "model_fingerprint", v)) hs.model_fingerprint = (uint64_t) v;
    if (json_get_int(json, "token_count", v)) hs.token_count = (uint32_t) v;
    return hs.version > 0;
}

std::string llama_decode_config_to_json(const llama_decode_config & cfg) {
    std::ostringstream ss;
    ss << "{\"stop_words\":[";
    for (size_t i = 0; i < cfg.stop_words.size(); i++) {
        if (i > 0) ss << ",";
        ss << "\"" << json_escape(cfg.stop_words[i]) << "\"";
    }
    ss << "],\"max_tokens\":" << cfg.max_tokens
       << ",\"prompt_token_count\":" << cfg.prompt_token_count
       << ",\"prompt_hash\":" << cfg.prompt_hash;
    if (cfg.reuse_retained_count > 0) {
        ss << ",\"reuse_retained_count\":" << cfg.reuse_retained_count
           << ",\"reuse_retained_prefix_hash\":"
           << (cfg.reuse_retained_prefix_hash & 0x7fffffffffffffffULL);
    }
    ss << "}";
    return ss.str();
}

bool llama_decode_config_from_json(const std::string & json, llama_decode_config & cfg) {
    cfg = {};
    int64_t max_tokens = -1;
    if (json_get_int(json, "max_tokens", max_tokens)) {
        cfg.max_tokens = (int32_t) max_tokens;
    }
    int64_t prompt_token_count = 0;
    if (json_get_int(json, "prompt_token_count", prompt_token_count)) {
        cfg.prompt_token_count = (uint32_t) prompt_token_count;
    }
    int64_t prompt_hash = 0;
    if (json_get_int(json, "prompt_hash", prompt_hash)) {
        cfg.prompt_hash = (uint64_t) prompt_hash;
    }
    int64_t reuse_retained_count = 0;
    if (json_get_int(json, "reuse_retained_count", reuse_retained_count) && reuse_retained_count > 0) {
        cfg.reuse_retained_count = (uint32_t) reuse_retained_count;
        int64_t reuse_retained_prefix_hash = 0;
        if (json_get_int(json, "reuse_retained_prefix_hash", reuse_retained_prefix_hash)) {
            cfg.reuse_retained_prefix_hash =
                (uint64_t) reuse_retained_prefix_hash & 0x7fffffffffffffffULL;
        }
    }

    // Simple JSON array parser for stop_words
    size_t arr_start = json.find("\"stop_words\"");
    if (arr_start == std::string::npos) return true; // no stop words is valid
    arr_start = json.find('[', arr_start);
    if (arr_start == std::string::npos) return false;
    size_t arr_end = json.find(']', arr_start);
    if (arr_end == std::string::npos) return false;

    std::string arr = json.substr(arr_start + 1, arr_end - arr_start - 1);
    size_t pos = 0;
    while (pos < arr.size()) {
        size_t q1 = arr.find('"', pos);
        if (q1 == std::string::npos) break;
        size_t q2 = q1 + 1;
        std::string word;
        while (q2 < arr.size()) {
            if (arr[q2] == '\\' && q2 + 1 < arr.size()) {
                char c = arr[q2 + 1];
                if (c == '"') word += '"';
                else if (c == '\\') word += '\\';
                else if (c == 'n') word += '\n';
                else if (c == 't') word += '\t';
                else { word += '\\'; word += c; }
                q2 += 2;
            } else if (arr[q2] == '"') {
                break;
            } else {
                word += arr[q2++];
            }
        }
        if (!word.empty()) {
            cfg.stop_words.push_back(word);
        }
        pos = q2 + 1;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Ring buffer
// ---------------------------------------------------------------------------

llama_token_ring::llama_token_ring() : buf_(CAPACITY) {}

bool llama_token_ring::push(const llama_token_msg & msg) {
    std::lock_guard<std::mutex> lock(mtx_);
    bool overwrite = false;
    if (count_ == CAPACITY) {
        // overwrite oldest
        tail_ = (tail_ + 1) % CAPACITY;
        count_--;
        dropped_++;
        overwrite = true;
    }
    buf_[head_] = msg;
    head_ = (head_ + 1) % CAPACITY;
    count_++;
    return !overwrite;
}

bool llama_token_ring::pop(llama_token_msg & msg) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (count_ == 0) return false;
    msg = std::move(buf_[tail_]);
    tail_ = (tail_ + 1) % CAPACITY;
    count_--;
    return true;
}

size_t llama_token_ring::size() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return count_;
}

int32_t llama_token_ring::total_dropped() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return dropped_;
}

// ---------------------------------------------------------------------------
// Socket helpers (cross-platform)
// ---------------------------------------------------------------------------

static socket_fd_t create_listen_socket(const std::string & host, int port) {
#ifdef _WIN32
    if (!winsock_ensure_init()) return SOCKET_INVALID;
#endif
    socket_fd_t fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == SOCKET_INVALID) return SOCKET_INVALID;

#ifndef _WIN32
    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#else
    const char opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t) port);
    inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

    if (bind(fd, (struct sockaddr *) &addr, sizeof(addr)) != 0) {
        SOCKET_CLOSE(fd);
        return SOCKET_INVALID;
    }
    if (listen(fd, 1) != 0) {
        SOCKET_CLOSE(fd);
        return SOCKET_INVALID;
    }
    return fd;
}

static int get_bound_port(socket_fd_t fd) {
    struct sockaddr_in addr = {};
    socklen_t len = sizeof(addr);
    if (getsockname(fd, (struct sockaddr *) &addr, &len) != 0) return -1;
    return ntohs(addr.sin_port);
}

static socket_fd_t accept_with_timeout(socket_fd_t listen_fd, int timeout_ms) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(listen_fd, &rfds);
    struct timeval tv = {};
    tv.tv_sec  = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
#ifndef _WIN32
    int ret = select((int)listen_fd + 1, &rfds, nullptr, nullptr, &tv);
#else
    int ret = select(0, &rfds, nullptr, nullptr, &tv);  // Windows ignores nfds
#endif
    if (ret <= 0) return SOCKET_INVALID;
    return accept(listen_fd, nullptr, nullptr);
}

static socket_fd_t connect_to(const std::string & host, int port) {
#ifdef _WIN32
    if (!winsock_ensure_init()) return SOCKET_INVALID;
#endif
    socket_fd_t fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == SOCKET_INVALID) return SOCKET_INVALID;

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t) port);
    inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

    if (::connect(fd, (struct sockaddr *) &addr, sizeof(addr)) != 0) {
        SOCKET_CLOSE(fd);
        return SOCKET_INVALID;
    }

    // disable Nagle for low-latency token delivery
#ifndef _WIN32
    int opt = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
#else
    const char opt = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
#endif

    return fd;
}

static bool send_all(socket_fd_t fd, const void * data, size_t len) {
    const char * p = (const char *) data;
    size_t sent = 0;
    while (sent < len) {
#ifndef _WIN32
        ssize_t n = ::send(fd, p + sent, len - sent, MSG_NOSIGNAL);
#else
        int n = ::send(fd, p + sent, (int)(len - sent), 0);
#endif
        if (n <= 0) return false;
        sent += (size_t) n;
    }
    return true;
}

static bool recv_with_timeout(socket_fd_t fd, void * buf, size_t len, int timeout_ms, size_t & bytes_read) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(fd, &rfds);
    struct timeval tv = {};
    tv.tv_sec  = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
#ifndef _WIN32
    int ret = select((int)fd + 1, &rfds, nullptr, nullptr, &tv);
#else
    int ret = select(0, &rfds, nullptr, nullptr, &tv);
#endif
    if (ret <= 0) return false;
#ifndef _WIN32
    ssize_t n = ::recv(fd, buf, len, 0);
#else
    int n = ::recv(fd, (char*)buf, (int)len, 0);
#endif
    if (n <= 0) return false;
    bytes_read = (size_t) n;
    return true;
}

// ---------------------------------------------------------------------------
// Token stream server
// ---------------------------------------------------------------------------

bool llama_token_stream_server::start(const std::string & host, int port) {
    listen_fd_ = create_listen_socket(host, port);
    if (listen_fd_ == SOCKET_INVALID) return false;
    port_ = get_bound_port(listen_fd_);
    running_ = true;
    return true;
}

bool llama_token_stream_server::wait_for_client(int timeout_ms) {
    client_fd_ = accept_with_timeout(listen_fd_, timeout_ms);
    if (client_fd_ == SOCKET_INVALID) return false;
#ifndef _WIN32
    int opt = 1;
    setsockopt(client_fd_, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
#else
    const char opt = 1;
    setsockopt(client_fd_, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
#endif
    return true;
}

bool llama_token_stream_server::read_config(llama_decode_config & cfg, int timeout_ms) {
    cfg = {};
    // Read one JSON line from the client (RTX sends config after connecting)
#ifdef _WIN32
    if (client_fd_ == ~(uintptr_t)0) return false;
    auto fd = (SOCKET)client_fd_;
#else
    if (client_fd_ < 0) return false;
    auto fd = client_fd_;
#endif
    std::string buf;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        char c;
#ifdef _WIN32
        int n = ::recv(fd, &c, 1, 0);
#else
        ssize_t n = ::recv(fd, &c, 1, 0);
#endif
        if (n <= 0) {
            if (n == 0) return false; // connection closed
#ifdef _WIN32
            if (WSAGetLastError() == WSAEWOULDBLOCK) continue;
#else
            if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
#endif
            return false;
        }
        if (c == '\n') {
            return llama_decode_config_from_json(buf, cfg);
        }
        buf += c;
    }
    return false; // timeout
}

bool llama_token_stream_server::send_handshake(const llama_token_handshake & hs) {
    return send_line(llama_token_handshake_to_json(hs));
}

bool llama_token_stream_server::enqueue(const llama_token_msg & msg) {
    ring_.push(msg);
    if (!sender_.joinable()) {
        sender_ = std::thread(&llama_token_stream_server::sender_loop, this);
    }
    return true;
}

bool llama_token_stream_server::send_done(int32_t tokens_generated, double decode_tok_s,
                                           int32_t stop_reason, const std::string & stopping_word) {
    llama_sustained_decode_timing timing = {};
    timing.emitted_tokens = tokens_generated;
    timing.decode_intervals = tokens_generated > 0 ? tokens_generated - 1 : 0;
    timing.tok_s = decode_tok_s;
    return send_done(tokens_generated, timing, stop_reason, stopping_word);
}

bool llama_token_stream_server::send_done(int32_t tokens_generated,
                                           const llama_sustained_decode_timing & timing,
                                           int32_t stop_reason, const std::string & stopping_word) {
    llama_token_msg done = {};
    done.done = true;
    done.tokens_generated = tokens_generated;
    done.decode_tok_s = timing.tok_s;
    done.decode_elapsed_ms = timing.elapsed_ms;
    done.decode_intervals = timing.decode_intervals;
    done.decode_tok_s_early = timing.early_tok_s;
    done.decode_tok_s_middle = timing.middle_tok_s;
    done.decode_tok_s_late = timing.late_tok_s;
    done.stop_reason = stop_reason;
    done.stopping_word = stopping_word;
    ring_.push(done);
    done_queued_ = true;

    // start sender if not running, or wait for it to drain
    if (!sender_.joinable()) {
        sender_ = std::thread(&llama_token_stream_server::sender_loop, this);
    }
    // wait for sender to finish draining
    if (sender_.joinable()) {
        sender_.join();
    }
    return true;
}

void llama_token_stream_server::stop() {
    running_ = false;
    if (sender_.joinable()) sender_.join();
    if (client_fd_ != SOCKET_INVALID) { SOCKET_CLOSE(client_fd_); client_fd_ = SOCKET_INVALID; }
    if (listen_fd_ != SOCKET_INVALID) { SOCKET_CLOSE(listen_fd_); listen_fd_ = SOCKET_INVALID; }
}

int llama_token_stream_server::bound_port() const {
    return port_;
}

bool llama_token_stream_server::is_client_connected() const {
#ifdef _WIN32
    if (client_fd_ == ~(uintptr_t)0) return false;
    // Set non-blocking for peek
    u_long mode = 1;
    ioctlsocket((SOCKET)client_fd_, FIONBIO, &mode);
    char probe;
    int r = recv((SOCKET)client_fd_, &probe, 1, MSG_PEEK);
    mode = 0;
    ioctlsocket((SOCKET)client_fd_, FIONBIO, &mode);
    if (r == 0) return false;  // clean disconnect
    if (r < 0) {
        int err = WSAGetLastError();
        if (err == WSAECONNRESET || err == WSAENOTCONN) return false;
    }
#else
    if (client_fd_ == -1) return false;
    char probe;
    int r = recv(client_fd_, &probe, 1, MSG_PEEK | MSG_DONTWAIT);
    if (r == 0) return false;
    if (r < 0) {
        if (errno == ECONNRESET || errno == ENOTCONN) return false;
    }
#endif
    return true;
}

void llama_token_stream_server::sender_loop() {
    while (running_) {
        llama_token_msg msg = {};
        if (ring_.pop(msg)) {
            std::string line = llama_token_msg_to_json(msg);
            if (!send_line(line)) {
                break;  // client disconnected
            }
            if (msg.done) {
                break;
            }
        } else {
            // ring empty, brief sleep to avoid busy-wait
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}

bool llama_token_stream_server::send_line(const std::string & line) {
    if (client_fd_ == SOCKET_INVALID) return false;
    std::string data = line + "\n";
    return send_all(client_fd_, data.data(), data.size());
}

// ---------------------------------------------------------------------------
// Token stream client
// ---------------------------------------------------------------------------

bool llama_token_stream_client::connect(const std::string & host, int port,
                                         int retries, int retry_ms) {
    // EXPONENTIAL BACKOFF, capped at retry_ms, instead of a flat retry_ms on every attempt.
    //
    // The two cases this connect covers have wildly different timescales. The caller in
    // llama-decode-handoff.cpp passes (120, 2000) for the SLOW case -- a 1.5 GB+ KV import plus a
    // 13k-token tail prefill on the Mac, 90+ seconds -- and that ~240 s ceiling is deliberate, so it
    // is preserved here. But the COMMON case is a bind race that resolves in milliseconds: the Mac
    // binds 9101 PER SESSION, so the RTX's first attempt can land before the receiver reaches its
    // bind. Measured on production 2026-09-03: 4 of 25 handoffs logged "attempt 1/120 failed" and
    // ZERO ever reached "exhausted all" -- every occurrence succeeded on the very next try, having
    // slept a flat 2000 ms waiting for something that was already ready. ~2 s of dead latency on
    // 16% of turns.
    //
    // Backing off from 25 ms and doubling to the cap costs 25 ms in that common case rather than
    // 2000 ms, while the first seven attempts still span only ~3.2 s and the full 120-attempt budget
    // still lands near 229 s -- the same order as the 240 s the caller asked for, so a genuinely slow
    // Mac import is not cut short.
    // NOTE: no std::min here -- <windows.h> defines min/max as MACROS, and std::min(...) then
    // expands to garbage (MSVC C2589 "illegal token on right side of '::'"). Caught by compiling
    // this TU before committing.
    int delay_ms = retry_ms < 25 ? retry_ms : 25;
    for (int attempt = 0; attempt < retries; attempt++) {
        fd_ = connect_to(host, port);
        if (fd_ != SOCKET_INVALID) return true;
        if (attempt < 3 || attempt % 10 == 0) {
            fprintf(stderr, "[token-stream] connect %s:%d attempt %d/%d failed, retrying in %dms...\n",
                    host.c_str(), port, attempt + 1, retries, delay_ms);
            fflush(stderr);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        delay_ms = (delay_ms * 2 > retry_ms) ? retry_ms : delay_ms * 2;
    }
    fprintf(stderr, "[token-stream] connect %s:%d exhausted all %d retries\n", host.c_str(), port, retries);
    fflush(stderr);
    return false;
}

bool llama_token_stream_client::send_config(const llama_decode_config & cfg) {
    if (fd_ == SOCKET_INVALID) return false;
    std::string data = llama_decode_config_to_json(cfg) + "\n";
    return send_all(fd_, data.data(), data.size());
}

bool llama_token_stream_client::read_handshake(llama_token_handshake & hs, int timeout_ms) {
    std::string line;
    if (!read_line(line, timeout_ms)) return false;
    return llama_token_handshake_from_json(line, hs);
}

bool llama_token_stream_client::read_msg(llama_token_msg & msg, int timeout_ms) {
    std::string line;
    if (!read_line(line, timeout_ms)) return false;
    return llama_token_msg_from_json(line, msg);
}

void llama_token_stream_client::disconnect() {
    if (fd_ != SOCKET_INVALID) { SOCKET_CLOSE(fd_); fd_ = SOCKET_INVALID; }
    buf_.clear();
}

bool llama_token_stream_client::read_line(std::string & line, int timeout_ms) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    while (true) {
        // check if we already have a complete line in the buffer
        auto nl = buf_.find('\n');
        if (nl != std::string::npos) {
            line = buf_.substr(0, nl);
            buf_.erase(0, nl + 1);
            return true;
        }

        // compute remaining timeout
        auto now = std::chrono::steady_clock::now();
        int remaining = (int) std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        if (remaining <= 0) return false;

        char tmp[4096];
        size_t n = 0;
        if (!recv_with_timeout(fd_, tmp, sizeof(tmp), remaining, n)) return false;
        buf_.append(tmp, n);
    }
}
