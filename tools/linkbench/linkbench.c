/*
 * linkbench - serial request/response round-trip benchmark for a point-to-point
 * link (RTX <-> Mac Studio). One C file, Windows (Winsock) and macOS/Linux.
 *
 * The side that accepts connections is the "server"; the side that connects is
 * the "client". Either side can be the requester (the side that times round
 * trips): the client sends a hello describing the run, and with --reverse the
 * server drives the requests and the client answers. That lets every direction
 * be measured with only one listening machine.
 *
 *   linkbench server --proto tcp|udp --bind IP --port N
 *   linkbench client --proto tcp|udp --host IP --port N [options]
 *
 * Client options:
 *   --sizes 8192,14336,28672  message bytes each way (request = response)
 *   --iters 10000 --warmup 500 --gap-us 50   (gap = busy wait between trips)
 *   --nodelay        TCP_NODELAY on both ends
 *   --quickack       Windows: SIO_TCP_SET_ACK_FREQUENCY=1 (no delayed ACK)
 *   --spin           busy-poll non-blocking receive on both ends
 *   --prio           high-priority thread on both ends (Windows TIME_CRITICAL +
 *                    HIGH class; macOS QOS_CLASS_USER_INTERACTIVE)
 *   --cpu N          pin the local thread to logical CPU N (Windows only)
 *   --bufs N         SO_SNDBUF/SO_RCVBUF bytes on both ends
 *   --chunk N        UDP datagram size incl. header (default 1472)
 *   --rto-us N       UDP retransmit timeout (default 2000)
 *   --reverse        server is the requester; client answers
 *   --label TEXT     tag for the RESULT line
 *
 * Output: one "RESULT ..." line per size with p50/p90/p99/p99.9/max/mean in
 * microseconds and, for UDP, retransmits.
 *
 * UDP framing: every datagram carries {magic, seq, idx, n, req_len, resp_len}.
 * The responder reassembles a request by seq, answers once, and re-sends its
 * cached answer when it sees a retransmitted request for the last seq. The
 * requester retransmits the whole request after --rto-us without a complete
 * answer. That is the loss detection; it never reorders across seq numbers.
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <mstcpip.h>
#include <windows.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET sock_t;
#define BAD_SOCK INVALID_SOCKET
#define closesock closesocket
static int would_block(void) { int e = WSAGetLastError(); return e == WSAEWOULDBLOCK; }
static int64_t now_ns(void) {
    static LARGE_INTEGER f; LARGE_INTEGER c;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (int64_t)((double)c.QuadPart * 1e9 / (double)f.QuadPart);
}
#else
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#ifdef __APPLE__
#include <pthread/qos.h>
#endif
typedef int sock_t;
#define BAD_SOCK (-1)
#define closesock close
static int would_block(void) { return errno == EAGAIN || errno == EWOULDBLOCK; }
static int64_t now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t)t.tv_sec * 1000000000LL + t.tv_nsec;
}
#endif

#define F_NODELAY  1u
#define F_SPIN     2u
#define F_PRIO     4u
#define F_REVERSE  8u
#define F_QUICKACK 16u

#define M_HELLO 0x314c484cu /* LHL1 */
#define M_ACK   0x314b414cu /* LAK1 */
#define M_MSG   0x314d424cu /* LBM1 */
#define M_UDP   0x3155424cu /* LBU1 */
#define M_RES   0x3152424cu /* LBR1 */
#define MAXSZ   8
#define MAXMSG  (1 << 20)
#define MAXCHUNKS 1024

typedef struct {
    uint32_t magic, flags, iters, warmup, gap_us, nsizes, chunk, rto_us, bufs;
    uint32_t sizes[MAXSZ];
} hello_t;

typedef struct { uint32_t magic, seq, req_len, resp_len; } mhdr_t;
typedef struct { uint32_t magic, seq; uint16_t idx, n; uint32_t req_len, resp_len; } uhdr_t;

static void die(const char *m) {
#ifdef _WIN32
    fprintf(stderr, "linkbench: %s (wsa=%d)\n", m, WSAGetLastError());
#else
    perror(m);
#endif
    exit(1);
}

static void set_nonblock(sock_t s, int on) {
#ifdef _WIN32
    u_long v = on; ioctlsocket(s, FIONBIO, &v);
#else
    int fl = fcntl(s, F_GETFL, 0);
    fcntl(s, F_SETFL, on ? (fl | O_NONBLOCK) : (fl & ~O_NONBLOCK));
#endif
}

static void apply_prio(uint32_t flags, int cpu) {
#ifdef _WIN32
    if (flags & F_PRIO) {
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    } else {
        SetPriorityClass(GetCurrentProcess(), NORMAL_PRIORITY_CLASS);
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
    }
    if (cpu >= 0) SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu);
#else
    (void)cpu;
#ifdef __APPLE__
    pthread_set_qos_class_self_np((flags & F_PRIO) ? QOS_CLASS_USER_INTERACTIVE : QOS_CLASS_DEFAULT, 0);
#endif
#endif
}

static void sock_opts(sock_t s, uint32_t flags, uint32_t bufs, int tcp) {
    int one = 1;
    if (tcp && (flags & F_NODELAY)) setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char *)&one, sizeof one);
    if (bufs) {
        int b = (int)bufs;
        setsockopt(s, SOL_SOCKET, SO_SNDBUF, (const char *)&b, sizeof b);
        setsockopt(s, SOL_SOCKET, SO_RCVBUF, (const char *)&b, sizeof b);
    }
#ifdef _WIN32
    if (tcp && (flags & F_QUICKACK)) {
        DWORD freq = 1, ret = 0;
        WSAIoctl(s, SIO_TCP_SET_ACK_FREQUENCY, &freq, sizeof freq, NULL, 0, &ret, NULL, NULL);
    }
#endif
}

/* ---- TCP helpers ---- */
static int send_all(sock_t s, const char *b, size_t n) {
    while (n) {
        int k = send(s, b, (int)(n > 0x40000000 ? 0x40000000 : n), 0);
        if (k < 0) { if (would_block()) continue; return -1; }
        b += k; n -= (size_t)k;
    }
    return 0;
}
/* spin != 0: socket is non-blocking and we busy-poll */
static int recv_all(sock_t s, char *b, size_t n) {
    while (n) {
        int k = recv(s, b, (int)n, 0);
        if (k == 0) return -1;
        if (k < 0) { if (would_block()) continue; return -1; }
        b += k; n -= (size_t)k;
    }
    return 0;
}

static int cmp_i64(const void *a, const void *b) {
    int64_t x = *(const int64_t *)a, y = *(const int64_t *)b; return (x > y) - (x < y);
}

static void busy_wait_us(uint32_t us) {
    if (!us) return;
    int64_t end = now_ns() + (int64_t)us * 1000;
    while (now_ns() < end) {}
}

static int format_result(char *out, size_t cap, const char *label, const char *proto, uint32_t flags,
                         uint32_t size, int64_t *rtt, uint32_t n, uint64_t retx, uint64_t lost) {
    qsort(rtt, n, sizeof *rtt, cmp_i64);
    double sum = 0; for (uint32_t i = 0; i < n; i++) sum += (double)rtt[i];
#define P(q) ((double)rtt[(uint32_t)((double)(n - 1) * (q))] / 1000.0)
    return snprintf(out, cap,
        "RESULT label=%s proto=%s flags=%s%s%s%s%s size=%u n=%u p50_us=%.1f p90_us=%.1f p99_us=%.1f p999_us=%.1f "
        "max_us=%.1f min_us=%.1f mean_us=%.1f retx=%llu lost=%llu\n",
        label, proto, (flags & F_NODELAY) ? "N" : "-", (flags & F_QUICKACK) ? "Q" : "-",
        (flags & F_SPIN) ? "S" : "-", (flags & F_PRIO) ? "P" : "-", (flags & F_REVERSE) ? "R" : "-",
        size, n, P(0.5), P(0.9), P(0.99), P(0.999), P(1.0), P(0.0), sum / n / 1000.0,
        (unsigned long long)retx, (unsigned long long)lost);
#undef P
}

/* ---- TCP request/response ---- */
static void tcp_requester(sock_t s, const hello_t *h, const char *label, char *res, size_t rescap) {
    char *buf = malloc(MAXMSG + 64), *rbuf = malloc(MAXMSG + 64);
    int64_t *rtt = malloc(sizeof(int64_t) * h->iters);
    size_t used = 0; uint32_t seq = 0;
    memset(buf, 0x5a, MAXMSG + 64);
    for (uint32_t z = 0; z < h->nsizes; z++) {
        uint32_t size = h->sizes[z];
        for (uint32_t i = 0; i < h->warmup + h->iters; i++) {
            mhdr_t *m = (mhdr_t *)buf;
            m->magic = M_MSG; m->seq = ++seq; m->req_len = size; m->resp_len = size;
            busy_wait_us(h->gap_us);
            int64_t t0 = now_ns();
            if (send_all(s, buf, sizeof(mhdr_t) + size)) die("send");
            mhdr_t r;
            if (recv_all(s, (char *)&r, sizeof r)) die("recv hdr");
            if (r.magic != M_MSG || r.seq != seq || r.resp_len > MAXMSG) { fprintf(stderr, "bad reply\n"); exit(1); }
            if (recv_all(s, rbuf, r.resp_len)) die("recv body");
            int64_t t1 = now_ns();
            if (i >= h->warmup) rtt[i - h->warmup] = t1 - t0;
        }
        used += format_result(res + used, rescap - used, label, "tcp", h->flags, size, rtt, h->iters, 0, 0);
    }
    mhdr_t end = { M_MSG, 0, 0, 0 };
    send_all(s, (char *)&end, sizeof end);
    free(buf); free(rbuf); free(rtt);
}

static void tcp_responder(sock_t s) {
    char *buf = malloc(MAXMSG + 64);
    memset(buf, 0xa5, MAXMSG + 64);
    for (;;) {
        mhdr_t m;
        if (recv_all(s, (char *)&m, sizeof m)) break;
        if (m.magic != M_MSG || m.seq == 0 || m.req_len > MAXMSG || m.resp_len > MAXMSG) break;
        if (recv_all(s, buf + sizeof(mhdr_t), m.req_len)) break;
        mhdr_t *r = (mhdr_t *)buf;
        r->magic = M_MSG; r->seq = m.seq; r->req_len = 0; r->resp_len = m.resp_len;
        if (send_all(s, buf, sizeof(mhdr_t) + m.resp_len)) break;
    }
    free(buf);
}

/* ---- UDP request/response ---- */
typedef struct sockaddr_storage addr_t;

static int udp_recv(sock_t s, char *b, int cap, addr_t *from, uint32_t flags, int64_t deadline_ns) {
    socklen_t fl = sizeof *from;
    for (;;) {
        int k = recvfrom(s, b, cap, 0, (struct sockaddr *)from, &fl);
        if (k >= 0) return k;
        if (!would_block()) {
#ifdef _WIN32
            if (WSAGetLastError() == WSAECONNRESET) continue;
#endif
            return -1;
        }
        if (flags & F_SPIN) { if (deadline_ns && now_ns() > deadline_ns) return -2; continue; }
        int64_t left = deadline_ns ? deadline_ns - now_ns() : 1000000000LL;
        if (left <= 0) return -2;
        fd_set rd; FD_ZERO(&rd); FD_SET(s, &rd);
        struct timeval tv = { (long)(left / 1000000000LL), (long)((left % 1000000000LL) / 1000) };
        select((int)s + 1, &rd, NULL, NULL, &tv);
    }
}

static void udp_send_msg(sock_t s, const addr_t *to, uint32_t seq, uint32_t req_len, uint32_t resp_len,
                         uint32_t body_len, uint32_t chunk, char *dgram) {
    uint32_t pay = chunk - (uint32_t)sizeof(uhdr_t);
    uint32_t n = body_len ? (body_len + pay - 1) / pay : 1;
    socklen_t tl = to->ss_family == AF_INET ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
    for (uint32_t i = 0; i < n; i++) {
        uhdr_t *u = (uhdr_t *)dgram;
        u->magic = M_UDP; u->seq = seq; u->idx = (uint16_t)i; u->n = (uint16_t)n;
        u->req_len = req_len; u->resp_len = resp_len;
        uint32_t len = body_len - i * pay; if (len > pay) len = pay; if (!body_len) len = 0;
        while (sendto(s, dgram, (int)(sizeof(uhdr_t) + len), 0, (const struct sockaddr *)to, tl) < 0) {
            if (!would_block()) die("sendto");
        }
    }
}

static void udp_requester(sock_t s, const addr_t *peer, const hello_t *h, const char *label, char *res, size_t rescap) {
    char *dg = malloc(70000), *rx = malloc(70000);
    int64_t *rtt = malloc(sizeof(int64_t) * h->iters);
    static uint8_t got[MAXCHUNKS];
    size_t used = 0; uint32_t seq = 0x1000;
    memset(dg, 0x5a, 70000);
    for (uint32_t z = 0; z < h->nsizes; z++) {
        uint32_t size = h->sizes[z]; uint64_t retx = 0, lost = 0;
        for (uint32_t i = 0; i < h->warmup + h->iters; i++) {
            ++seq;
            busy_wait_us(h->gap_us);
            int64_t t0 = now_ns();
            udp_send_msg(s, peer, seq, size, size, size, h->chunk, dg);
            uint32_t have = 0, need = 0xffffffffu; int tries = 0;
            memset(got, 0, sizeof got);
            int64_t deadline = now_ns() + (int64_t)h->rto_us * 1000;
            while (have != need) {
                addr_t from;
                int k = udp_recv(s, rx, 70000, &from, h->flags, deadline);
                if (k == -2) {
                    if (++tries > 50) { lost++; break; }  /* message given up */
                    retx++;
                    udp_send_msg(s, peer, seq, size, size, size, h->chunk, dg);
                    deadline = now_ns() + (int64_t)h->rto_us * 1000;
                    continue;
                }
                if (k < (int)sizeof(uhdr_t)) continue;
                uhdr_t *u = (uhdr_t *)rx;
                if (u->magic != M_UDP || u->seq != seq || u->idx >= MAXCHUNKS) continue;
                need = u->n;
                if (!got[u->idx]) { got[u->idx] = 1; have++; }
            }
            int64_t t1 = now_ns();
            if (i >= h->warmup) rtt[i - h->warmup] = t1 - t0;
            if (lost >= 10) { /* peer gone: stop instead of retrying for minutes */
                used += snprintf(res + used, rescap - used, "FAILED label=%s proto=udp size=%u lost=%llu at=%u\n",
                                 label, size, (unsigned long long)lost, i);
                goto out;
            }
        }
        used += format_result(res + used, rescap - used, label, "udp", h->flags, size, rtt, h->iters, retx, lost);
    }
out:
    free(dg); free(rx); free(rtt);
}

/* Answers requests until it receives a datagram with seq 0 (end) or a non-UDP magic. */
static void udp_responder(sock_t s, uint32_t flags, uint32_t chunk, addr_t *peer_out) {
    char *dg = malloc(70000), *rx = malloc(70000);
    static uint8_t got[MAXCHUNKS];
    uint32_t cur = 0, have = 0, done = 0, last_resp = 0;
    addr_t from;
    memset(dg, 0xa5, 70000);
    for (;;) {
        int k = udp_recv(s, rx, 70000, &from, flags, now_ns() + 30LL * 1000000000LL);
        if (k == -2) break; /* idle 30 s: run is over */
        if (k < (int)sizeof(uhdr_t)) continue;
        uhdr_t *u = (uhdr_t *)rx;
        if (u->magic != M_UDP) continue;
        if (u->seq == 0) break;
        if (u->idx >= MAXCHUNKS) continue;
        if (u->seq == done) { /* retransmitted request: resend cached answer once per retransmit */
            if (u->idx == 0) udp_send_msg(s, &from, done, 0, last_resp, last_resp, chunk, dg);
            continue;
        }
        if (u->seq != cur) { cur = u->seq; have = 0; memset(got, 0, sizeof got); }
        if (!got[u->idx]) { got[u->idx] = 1; have++; }
        if (have == u->n) {
            done = cur; last_resp = u->resp_len;
            udp_send_msg(s, &from, done, 0, last_resp, last_resp, chunk, dg);
        }
    }
    if (peer_out) *peer_out = from;
    free(dg); free(rx);
}

/* ---- setup ---- */
static int resolve(const char *host, int port, addr_t *a) {
    memset(a, 0, sizeof *a);
    struct sockaddr_in *v4 = (struct sockaddr_in *)a;
    v4->sin_family = AF_INET; v4->sin_port = htons((uint16_t)port);
    return inet_pton(AF_INET, host, &v4->sin_addr) == 1 ? 0 : -1;
}

static void usage(void) {
    fprintf(stderr, "usage: linkbench server --proto tcp|udp --bind IP --port N [--cpu N]\n"
                    "       linkbench client --proto tcp|udp --host IP --port N [--sizes a,b,c] [--iters N]\n"
                    "         [--warmup N] [--gap-us N] [--nodelay] [--quickack] [--spin] [--prio] [--cpu N]\n"
                    "         [--bufs N] [--chunk N] [--rto-us N] [--reverse] [--label TEXT]\n");
    exit(2);
}

static void run_server(int udp, const char *bind_ip, int port, int cpu) {
    addr_t a; if (resolve(bind_ip, port, &a)) usage();
    sock_t ls = socket(AF_INET, udp ? SOCK_DGRAM : SOCK_STREAM, 0);
    int one = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof one);
    if (bind(ls, (struct sockaddr *)&a, sizeof(struct sockaddr_in))) die("bind");
    if (!udp && listen(ls, 4)) die("listen");
    fprintf(stderr, "linkbench server %s %s:%d ready\n", udp ? "udp" : "tcp", bind_ip, port);
    char *res = malloc(65536);
    for (;;) {
        hello_t h; sock_t s = ls; addr_t peer;
        if (!udp) {
            s = accept(ls, NULL, NULL);
            if (s == BAD_SOCK) continue;
            set_nonblock(s, 0);
            if (recv_all(s, (char *)&h, sizeof h) || h.magic != M_HELLO) { closesock(s); continue; }
        } else {
            set_nonblock(s, 0);
            socklen_t pl = sizeof peer;
            int k = recvfrom(s, (char *)&h, sizeof h, 0, (struct sockaddr *)&peer, &pl);
            if (k != (int)sizeof h || h.magic != M_HELLO) continue;
        }
        if (h.nsizes > MAXSZ) h.nsizes = MAXSZ;
        apply_prio(h.flags, cpu);
        sock_opts(s, h.flags, h.bufs, !udp);
        uint32_t ack = M_ACK;
        if (!udp) send_all(s, (char *)&ack, 4);
        else sendto(s, (char *)&ack, 4, 0, (struct sockaddr *)&peer, sizeof(struct sockaddr_in));
        set_nonblock(s, (udp || (h.flags & F_SPIN)) ? 1 : 0); /* UDP always non-blocking: waits use select with a deadline */
        if (h.flags & F_REVERSE) {
            res[0] = 0;
            if (!udp) tcp_requester(s, &h, "rev", res, 65536);
            else { udp_requester(s, &peer, &h, "rev", res, 65536); }
            set_nonblock(s, 0);
            uint32_t hdr[2] = { M_RES, (uint32_t)strlen(res) + 1 };
            if (!udp) { send_all(s, (char *)hdr, 8); send_all(s, res, hdr[1]); }
            else {
                char end[sizeof(uhdr_t)] = { 0 }; ((uhdr_t *)end)->magic = M_UDP;
                for (int r = 0; r < 3; r++) sendto(s, end, sizeof end, 0, (struct sockaddr *)&peer, sizeof(struct sockaddr_in));
                char pkt[8192]; memcpy(pkt, hdr, 8); memcpy(pkt + 8, res, hdr[1] < 8000 ? hdr[1] : 8000);
                for (int r = 0; r < 5; r++) sendto(s, pkt, 8 + (int)(hdr[1] < 8000 ? hdr[1] : 8000), 0,
                                                   (struct sockaddr *)&peer, sizeof(struct sockaddr_in));
            }
            fputs(res, stderr);
        } else {
            if (!udp) tcp_responder(s);
            else udp_responder(s, h.flags, h.chunk, NULL);
        }
        if (!udp) closesock(s);
        else { /* drop leftovers (duplicate hellos, end markers) so a stale peer is never picked up */
            char junk[2048]; set_nonblock(s, 1);
            while (recv(s, junk, sizeof junk, 0) >= 0 || !would_block()) {}
            set_nonblock(s, 0);
        }
        apply_prio(0, -1);
    }
}

static void run_client(int udp, const char *host, int port, hello_t *h, int cpu, const char *label) {
    addr_t a; if (resolve(host, port, &a)) usage();
    sock_t s = socket(AF_INET, udp ? SOCK_DGRAM : SOCK_STREAM, 0);
    apply_prio(h->flags, cpu);
    sock_opts(s, h->flags, h->bufs, !udp);
    if (connect(s, (struct sockaddr *)&a, sizeof(struct sockaddr_in))) die("connect");
    if (!udp) {
        if (send_all(s, (char *)h, sizeof *h)) die("hello");
        uint32_t ack; if (recv_all(s, (char *)&ack, 4) || ack != M_ACK) die("hello ack");
    } else {
        uint32_t ack = 0;
        for (int t = 0; t < 20 && ack != M_ACK; t++) {
            send(s, (char *)h, sizeof *h, 0);
            fd_set rd; FD_ZERO(&rd); FD_SET(s, &rd); struct timeval tv = { 0, 200000 };
            if (select((int)s + 1, &rd, NULL, NULL, &tv) > 0) recv(s, (char *)&ack, 4, 0);
        }
        if (ack != M_ACK) die("udp hello ack");
    }
    set_nonblock(s, (udp || (h->flags & F_SPIN)) ? 1 : 0);
    char *res = malloc(65536); res[0] = 0;
    if (h->flags & F_REVERSE) {
        if (!udp) {
            tcp_responder(s);  /* returns at the end marker */
            set_nonblock(s, 0);
            uint32_t hdr[2];
            if (recv_all(s, (char *)hdr, 8) || hdr[0] != M_RES || hdr[1] > 65536) die("result");
            if (recv_all(s, res, hdr[1])) die("result body");
        } else {
            udp_responder(s, h->flags, h->chunk, NULL); /* returns at the seq-0 end marker */
            set_nonblock(s, 0);
            char pkt[8192];
            for (;;) {
                fd_set rd; FD_ZERO(&rd); FD_SET(s, &rd); struct timeval tv = { 5, 0 };
                if (select((int)s + 1, &rd, NULL, NULL, &tv) <= 0) die("udp result");
                int k = recv(s, pkt, sizeof pkt - 1, 0);
                if (k > 8 && ((uint32_t *)pkt)[0] == M_RES) { pkt[k] = 0; memcpy(res, pkt + 8, (size_t)(k - 8) + 1); break; }
            }
        }
        /* print with the local label */
        char *p = res;
        while (*p) {
            char *nl = strchr(p, '\n'); if (!nl) break; *nl = 0;
            char *rest = strstr(p, " proto=");
            printf("RESULT label=%s%s\n", label, rest ? rest : "");
            p = nl + 1;
        }
    } else {
        if (!udp) tcp_requester(s, h, label, res, 65536);
        else {
            udp_requester(s, &a, h, label, res, 65536);
            char end[sizeof(uhdr_t)] = { 0 }; uhdr_t *u = (uhdr_t *)end; u->magic = M_UDP; u->seq = 0;
            for (int r = 0; r < 3; r++) send(s, end, sizeof end, 0);
        }
        fputs(res, stdout);
    }
    fflush(stdout);
    closesock(s);
}

int main(int argc, char **argv) {
#ifdef _WIN32
    WSADATA w; WSAStartup(MAKEWORD(2, 2), &w);
    setvbuf(stdout, NULL, _IONBF, 0);
#endif
    if (argc < 2) usage();
    int server = !strcmp(argv[1], "server");
    if (!server && strcmp(argv[1], "client")) usage();
    int udp = 0, port = 0, cpu = -1;
    const char *host = "127.0.0.1", *label = "run";
    hello_t h; memset(&h, 0, sizeof h);
    h.magic = M_HELLO; h.iters = 10000; h.warmup = 500; h.gap_us = 50; h.chunk = 1472; h.rto_us = 2000;
    h.nsizes = 3; h.sizes[0] = 8192; h.sizes[1] = 14336; h.sizes[2] = 28672;
    for (int i = 2; i < argc; i++) {
        const char *k = argv[i]; const char *v = i + 1 < argc ? argv[i + 1] : NULL;
#define ARG(name) (!strcmp(k, name) && v && (++i, 1))
        if (ARG("--proto")) udp = !strcmp(v, "udp");
        else if (ARG("--bind") || ARG("--host")) host = v;
        else if (ARG("--port")) port = atoi(v);
        else if (ARG("--cpu")) cpu = atoi(v);
        else if (ARG("--iters")) h.iters = (uint32_t)atoi(v);
        else if (ARG("--warmup")) h.warmup = (uint32_t)atoi(v);
        else if (ARG("--gap-us")) h.gap_us = (uint32_t)atoi(v);
        else if (ARG("--chunk")) h.chunk = (uint32_t)atoi(v);
        else if (ARG("--rto-us")) h.rto_us = (uint32_t)atoi(v);
        else if (ARG("--bufs")) h.bufs = (uint32_t)atoi(v);
        else if (ARG("--label")) label = v;
        else if (ARG("--sizes")) {
            h.nsizes = 0;
            for (const char *p = v; *p && h.nsizes < MAXSZ;) {
                h.sizes[h.nsizes++] = (uint32_t)strtoul(p, (char **)&p, 10);
                if (*p == ',') p++;
            }
        }
        else if (!strcmp(k, "--nodelay")) h.flags |= F_NODELAY;
        else if (!strcmp(k, "--quickack")) h.flags |= F_QUICKACK;
        else if (!strcmp(k, "--spin")) h.flags |= F_SPIN;
        else if (!strcmp(k, "--prio")) h.flags |= F_PRIO;
        else if (!strcmp(k, "--reverse")) h.flags |= F_REVERSE;
        else usage();
#undef ARG
    }
    if (!port) usage();
    for (uint32_t z = 0; z < h.nsizes; z++) if (h.sizes[z] > MAXMSG) usage();
    if (h.chunk < 64 || h.chunk > 65000) usage();
    if (server) run_server(udp, host, port, cpu);
    else run_client(udp, host, port, &h, cpu, label);
    return 0;
}
