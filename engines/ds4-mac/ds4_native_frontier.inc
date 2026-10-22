/* Private server adapter. Enabling this source is not qualification or preset
 * activation: startup also requires a separately pinned qualification artifact.
 * It consumes native final token IDs and imports into the existing slot only. */
#ifndef DS4_NATIVE_FRONTIER_ENABLE
#define DS4_NATIVE_FRONTIER_ENABLE 0
#endif
typedef struct ds4_nf_config ds4_nf_config;

#if DS4_NATIVE_FRONTIER_ENABLE || defined(DS4_NATIVE_FRONTIER_TEST)
#define DS4_NF_MODEL_SHA "2cc6bc254f22a0238020e61e931c0cac865218aada1eed9c0036d9f4031c6e5f"
#define DS4_NF_JSON_CAP 16384
#define DS4_NF_QUALIFICATION_CAP (1024 * 1024)
#define DS4_NF_RESERVE (UINT64_C(30) * 1024 * 1024 * 1024)
#define DS4_NF_CHUNK (1024 * 1024)

typedef struct {
    char schema[64], job_id[33], state[16], model_sha256[65];
    char tokens_sha256[65], payload_sha256[65], producer_binary_sha256[65];
    char producer_source_revision[41], qualification_sha256[65];
    char receiver_binary_sha256[65], renderer_sha256[65], tokenization[64];
    uint64_t context, position, mode, wire_version, payload_bytes;
    uint64_t receiver_model_size, receiver_model_device;
    uint64_t receiver_model_inode, receiver_model_mtime_ns;
    bool native_parity_passed, performance_passed;
    uint64_t present;
} ds4_nf_fields;

/* Protocol identity integers never pass through the server's double parser.
 * Reject signs, leading zeros, fractions, exponent notation and overflow. */
static bool ds4_nf_u64(const char **p, uint64_t *out) {
    json_ws(p);
    const char *s = *p;
    if (*s < '0' || *s > '9' || (*s == '0' && s[1] >= '0' && s[1] <= '9'))
        return false;
    uint64_t v = 0;
    do {
        unsigned n = (unsigned)(*s++ - '0');
        if (v > (UINT64_MAX - n) / 10) return false;
        v = v * 10 + n;
    } while (*s >= '0' && *s <= '9');
    if (*s && !strchr(" \r\n\t,}", *s)) return false;
    *out = v; *p = s;
    return true;
}

static bool ds4_nf_ascii(const char **p, char *out, size_t cap) {
    json_ws(p);
    const char *start = *p;
    if (*start++ != '"') return false;
    const char *end = start;
    while (*end && *end != '"') {
        if ((unsigned char)*end < 32 || (unsigned char)*end >= 127 || *end == '\\')
            return false;
        end++;
    }
    if (*end != '"' || (size_t)(end - start) >= cap) return false;
    char *decoded = NULL;
    if (!json_string(p, &decoded)) return false;
    memcpy(out, decoded, (size_t)(end - start) + 1);
    free(decoded);
    return true;
}

static bool ds4_nf_hex(const char *s, size_t n) {
    if (strlen(s) != n) return false;
    for (size_t i = 0; i < n; i++)
        if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f')))
            return false;
    return true;
}

/* Reuse the server JSON parser, but keep identity/length fields exact and reject
 * duplicate known keys. Unknown fields (including bounded sample evidence) are
 * parsed only after the whole qualification file's external SHA is verified. */
static bool ds4_nf_parse_bounded(const char *text,size_t len,ds4_nf_fields *out,size_t cap) {
    if (!text || !out || len > cap || memchr(text, 0, len)) return false;
    memset(out, 0, sizeof(*out));
    const char *p = text;
    json_ws(&p);
    if (*p++ != '{') return false;
    json_ws(&p);
    if (*p == '}') return false;
    for (;;) {
        char key[80];
        if (!ds4_nf_ascii(&p, key, sizeof(key))) return false;
        json_ws(&p);
        if (*p++ != ':') return false;
        int bit = -1;
        bool ok = true;
#define NF_STRING(name, index) if (!strcmp(key, #name)) { bit = index; ok = ds4_nf_ascii(&p, out->name, sizeof(out->name)); }
#define NF_UINT(name, index) if (!strcmp(key, #name)) { bit = index; ok = ds4_nf_u64(&p, &out->name); }
#define NF_BOOL(name, index) if (!strcmp(key, #name)) { bit = index; ok = json_bool(&p, &out->name); }
        NF_STRING(schema, 0)
        else NF_STRING(job_id, 1)
        else NF_STRING(state, 2)
        else NF_STRING(model_sha256, 3)
        else NF_STRING(tokens_sha256, 4)
        else NF_STRING(payload_sha256, 5)
        else NF_STRING(producer_binary_sha256, 6)
        else NF_STRING(producer_source_revision, 7)
        else NF_STRING(qualification_sha256, 8)
        else NF_UINT(context, 9)
        else NF_UINT(position, 10)
        else NF_UINT(mode, 11)
        else NF_UINT(wire_version, 12)
        else NF_UINT(payload_bytes, 13)
        else NF_STRING(receiver_binary_sha256, 14)
        else NF_STRING(renderer_sha256, 15)
        else NF_STRING(tokenization, 16)
        else NF_UINT(receiver_model_size, 17)
        else NF_UINT(receiver_model_device, 18)
        else NF_UINT(receiver_model_inode, 19)
        else NF_UINT(receiver_model_mtime_ns, 20)
        else NF_BOOL(native_parity_passed, 21)
        else NF_BOOL(performance_passed, 22)
        else ok = json_skip_value(&p);
#undef NF_STRING
#undef NF_UINT
#undef NF_BOOL
        if (!ok || (bit >= 0 && (out->present & (UINT64_C(1) << bit)))) return false;
        if (bit >= 0) out->present |= UINT64_C(1) << bit;
        json_ws(&p);
        if (*p == '}') { p++; break; }
        if (*p++ != ',') return false;
    }
    json_ws(&p);
    return (size_t)(p - text) == len;
}

static bool ds4_nf_parse(const char *text,size_t len,ds4_nf_fields *out) {
    return ds4_nf_parse_bounded(text,len,out,DS4_NF_JSON_CAP);
}

static bool ds4_nf_have(const ds4_nf_fields *f, uint64_t mask) {
    return (f->present & mask) == mask;
}

static uint64_t ds4_nf_payload_bytes(uint32_t pos) {
    uint64_t raw = pos < 128 ? pos : 128;
    return 52 + ((uint64_t)pos + 129280) * 4 + 40 * raw * 512 * 4 +
           (3 * ((uint64_t)pos / 2) + pos) * (512 + 128) * 4 +
           ((pos & 1) ? 3 * 2 * 512 * 4 : 0);
}

static uint32_t ds4_nf_le32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static bool ds4_nf_header(const unsigned char h[52], uint32_t ctx,
                          uint32_t pos, uint64_t bytes) {
    const uint32_t words[13] = {0x34565344, 2, ctx, 1, 128, 128,
                               ctx + 1, pos, 40, 512, 128, 129280, 0x413431};
    if (ctx < 2 || ctx > 1048576 || pos == 0 || pos >= ctx ||
        bytes != ds4_nf_payload_bytes(pos)) return false;
    for (int i = 0; i < 13; i++)
        if (ds4_nf_le32(h + i * 4) != words[i]) return false;
    return true;
}

static bool ds4_nf_same_identity(const ds4_native_frontier_model_identity *a,
                                 const ds4_native_frontier_model_identity *b) {
    return a->device == b->device && a->inode == b->inode &&
           a->size == b->size && a->mtime_ns == b->mtime_ns;
}

static bool ds4_nf_completed(const ds4_nf_fields *f,const char *job,
                              const char *tokens,int ctx,int pos,
                              const char *binary,const char *source,const char *qualification) {
    uint64_t required=(UINT64_C(1)<<14)-1;
    return ds4_nf_have(f,required) && !strcmp(f->schema,"dsv41-native-offload-v1") &&
        !strcmp(f->job_id,job) && !strcmp(f->state,"completed") && ds4_nf_hex(job,32) &&
        !strcmp(f->model_sha256,DS4_NF_MODEL_SHA) && !strcmp(f->tokens_sha256,tokens) &&
        ctx>=2 && ctx<=1048576 && pos>0 && pos<ctx &&
        f->context==(uint64_t)ctx && f->position==(uint64_t)pos && f->mode==1 && f->wire_version==2 &&
        f->payload_bytes==ds4_nf_payload_bytes((uint32_t)pos) && ds4_nf_hex(f->payload_sha256,64) &&
        !strcmp(f->producer_binary_sha256,binary) &&
        !strcmp(f->producer_source_revision,source) &&
        !strcmp(f->qualification_sha256,qualification);
}
#endif

#if DS4_NATIVE_FRONTIER_ENABLE && defined(__APPLE__)
#include <CommonCrypto/CommonDigest.h>
#include <sys/file.h>
#include <sys/statvfs.h>
#include <mach-o/dyld.h>

struct ds4_nf_config {
    struct sockaddr_in broker;
    char host[16], token[513], qualification_sha256[65];
    char producer_binary_sha256[65], producer_source_revision[41];
    ds4_native_frontier_model_identity model;
    int dirfd;
    struct stat dir_stat;
    char directory[PATH_MAX];
    uint32_t minimum_tokens, maximum_tokens, deadline_seconds;
};
typedef bool (*ds4_nf_cancel_fn)(void *);
typedef struct {
    int fd;
    double deadline, last_activity;
    ds4_nf_cancel_fn cancelled;
    void *cancel_ud;
} ds4_nf_io;

static double ds4_nf_now(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC,&t)) return 1e30;
    return (double)t.tv_sec+(double)t.tv_nsec/1e9;
}

static int ds4_nf_error(char *err, size_t cap, const char *message) {
    snprintf(err, cap, "native frontier: %s", message);
    return -1;
}

static void ds4_nf_digest(const unsigned char digest[32], char out[65]) {
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) { out[2*i] = hex[digest[i] >> 4]; out[2*i+1] = hex[digest[i] & 15]; }
    out[64] = 0;
}

/* CommonCrypto is part of libSystem on Darwin. Only these incremental SHA calls
 * are used; no shell utility, external process or model bytes are loaded here. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
static void ds4_nf_hash_bytes(const void *bytes,size_t length,char out[65]) {
    unsigned char digest[32];CC_SHA256(bytes,(CC_LONG)length,digest);
    ds4_nf_digest(digest,out);
}

static bool ds4_nf_hash_file(int fd, uint64_t cap, char out[65], ds4_nf_io *guard) {
    struct stat before, after;
    if (fstat(fd, &before) || !S_ISREG(before.st_mode) || before.st_size < 0 ||
        (uint64_t)before.st_size > cap) return false;
    unsigned char *page = malloc(DS4_NF_CHUNK);
    if (!page) return false;
    CC_SHA256_CTX hash; CC_SHA256_Init(&hash);
    bool ok = true;
    for (uint64_t off = 0; off < (uint64_t)before.st_size;) {
        if (guard && (ds4_nf_now() >= guard->deadline ||
            (guard->cancelled && guard->cancelled(guard->cancel_ud)))) {ok=false;break;}
        size_t want = (uint64_t)before.st_size - off > DS4_NF_CHUNK ?
                      DS4_NF_CHUNK : (size_t)((uint64_t)before.st_size - off);
        ssize_t got = pread(fd, page, want, (off_t)off);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) { ok = false; break; }
        CC_SHA256_Update(&hash, page, (CC_LONG)got); off += (uint64_t)got;
    }
    unsigned char digest[32]; CC_SHA256_Final(digest, &hash);
    free(page);
    if (fstat(fd, &after) || before.st_dev != after.st_dev ||
        before.st_ino != after.st_ino || before.st_size != after.st_size ||
        before.st_mtimespec.tv_sec != after.st_mtimespec.tv_sec ||
        before.st_mtimespec.tv_nsec != after.st_mtimespec.tv_nsec) ok = false;
    if (ok) ds4_nf_digest(digest, out);
    return ok;
}

static bool ds4_nf_token_hash(const ds4_tokens *tokens, char out[65]) {
    CC_SHA256_CTX hash; CC_SHA256_Init(&hash);
    for (int i = 0; i < tokens->len; i++) {
        uint32_t t = (uint32_t)tokens->v[i];
        if (t >= 129280) return false;
        unsigned char v[4] = {t, t >> 8, t >> 16, t >> 24};
        CC_SHA256_Update(&hash, v, 4);
    }
    unsigned char digest[32]; CC_SHA256_Final(digest, &hash);
    ds4_nf_digest(digest, out); return true;
}
#pragma clang diagnostic pop

/* Walk each component with O_NOFOLLOW. Directory identity is checked again
 * before create and before import; anchored openat never follows a rename. */
static int ds4_nf_physical_dir(const char *path) {
    if (!path || path[0] != '/' || strlen(path) >= PATH_MAX) return -1;
    char copy[PATH_MAX]; snprintf(copy, sizeof(copy), "%s", path);
    int fd = open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    char *cursor = copy + 1;
    while (fd >= 0 && *cursor) {
        char *slash = strchr(cursor, '/');
        if (slash) *slash = 0;
        if (!*cursor || !strcmp(cursor, ".") || !strcmp(cursor, "..")) { close(fd); return -1; }
        int next = openat(fd, cursor, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        close(fd); fd = next;
        if (!slash) break;
        cursor = slash + 1;
        if (!*cursor) { if (fd >= 0) close(fd); return -1; }
    }
    return fd;
}

static bool ds4_nf_directory(ds4_nf_config *c) {
    int fd = ds4_nf_physical_dir(c->directory);
    struct stat st;
    bool ok = fd >= 0 && !fstat(fd, &st) && st.st_dev == c->dir_stat.st_dev &&
              st.st_ino == c->dir_stat.st_ino && st.st_uid == geteuid() &&
              !(st.st_mode & 0077);
    if (fd >= 0) close(fd);
    return ok;
}

static bool ds4_nf_space(ds4_nf_config *c, uint64_t bytes) {
    struct statvfs st;
    if (!ds4_nf_directory(c) || fstatvfs(c->dirfd, &st) || !st.f_frsize ||
        st.f_bavail > UINT64_MAX / st.f_frsize) return false;
    uint64_t available = (uint64_t)st.f_bavail * st.f_frsize;
    return bytes <= UINT64_MAX - DS4_NF_RESERVE && available >= bytes + DS4_NF_RESERVE;
}

static bool ds4_nf_pair_token(const char *path,char token[513]) {
    if(!path || path[0]!='/' || strlen(path)>=PATH_MAX)return false;
    char parent[PATH_MAX];snprintf(parent,sizeof(parent),"%s",path);
    char *base=strrchr(parent,'/');
    if(!base || !base[1])return false;
    const char *name=path+(base-parent)+1;
    if(base==parent)base[1]=0;else *base=0;
    int dir=ds4_nf_physical_dir(parent);
    if(dir<0)return false;
    int fd=openat(dir,name,O_RDONLY|O_NOFOLLOW|O_CLOEXEC);close(dir);
    struct stat before,after;char text[515];
    bool ok=fd>=0 && !fstat(fd,&before) && S_ISREG(before.st_mode) &&
        before.st_uid==geteuid() && !(before.st_mode&0077) && before.st_nlink==1 &&
        before.st_size>0 && before.st_size<=514 && !flock(fd,LOCK_SH|LOCK_NB);
    if(ok) {
        ssize_t n=pread(fd,text,(size_t)before.st_size,0);
        ok=n==before.st_size && !fstat(fd,&after) && before.st_dev==after.st_dev &&
            before.st_ino==after.st_ino && before.st_size==after.st_size &&
            before.st_mtimespec.tv_sec==after.st_mtimespec.tv_sec &&
            before.st_mtimespec.tv_nsec==after.st_mtimespec.tv_nsec;
        if(ok) {
            if(n && text[n-1]=='\n')n--;
            if(n && text[n-1]=='\r')n--;
            ok=n>0 && n<=512;
            for(ssize_t i=0;ok && i<n;i++)if((unsigned char)text[i]<33 || (unsigned char)text[i]>126)ok=false;
            if(ok) {memcpy(token,text,(size_t)n);token[n]=0;}
        }
    }
    memset(text,0,sizeof(text));if(fd>=0)close(fd);return ok;
}

static bool ds4_nf_wait(ds4_nf_io *io, short events) {
    for (;;) {
        if (ds4_nf_now() >= io->deadline || ds4_nf_now() - io->last_activity >= 10 ||
            (io->cancelled && io->cancelled(io->cancel_ud))) return false;
        struct pollfd p = {.fd = io->fd, .events = events};
        int r = poll(&p, 1, 100);
        if (r < 0 && errno == EINTR) continue;
        if (r < 0 || (r > 0 && (p.revents & (POLLERR | POLLNVAL)))) return false;
        if (r > 0 && (p.revents & (events | POLLHUP))) return true;
    }
}

static bool ds4_nf_send(ds4_nf_io *io, const void *ptr, size_t n) {
    const unsigned char *p = ptr;
    while (n) {
        if (!ds4_nf_wait(io, POLLOUT)) return false;
        ssize_t r = send(io->fd, p, n, 0);
        if (r < 0 && (errno == EINTR || errno == EAGAIN)) continue;
        if (r <= 0) return false;
        io->last_activity = ds4_nf_now();
        p += r; n -= (size_t)r;
    }
    return true;
}

static bool ds4_nf_read(ds4_nf_io *io, void *ptr, size_t n) {
    unsigned char *p = ptr;
    while (n) {
        if (!ds4_nf_wait(io, POLLIN)) return false;
        ssize_t r = recv(io->fd, p, n, 0);
        if (r < 0 && (errno == EINTR || errno == EAGAIN)) continue;
        if (r <= 0) return false;
        io->last_activity = ds4_nf_now();
        p += r; n -= (size_t)r;
    }
    return true;
}

static bool ds4_nf_connect(ds4_nf_config *c, ds4_nf_io *io) {
    io->last_activity = ds4_nf_now();
    io->fd = socket(AF_INET, SOCK_STREAM, 0);
    if (io->fd < 0) return false;
    int flags = fcntl(io->fd, F_GETFL, 0), error = 0;
    socklen_t len = sizeof(error);
    if (flags < 0 || fcntl(io->fd, F_SETFL, flags | O_NONBLOCK)) return false;
    int rc = connect(io->fd, (struct sockaddr *)&c->broker, sizeof(c->broker));
    if (rc && errno != EINPROGRESS) return false;
    double whole_deadline=io->deadline;
    if(io->deadline>ds4_nf_now()+2)io->deadline=ds4_nf_now()+2;
    bool ok=ds4_nf_wait(io,POLLOUT) && !getsockopt(io->fd,SOL_SOCKET,SO_ERROR,&error,&len) && !error;
    io->deadline=whole_deadline;return ok;
}

/* One request per connection. Header reads are bounded bytewise so no payload
 * bytes disappear in a header buffer; large bodies are always streamed. */
static bool ds4_nf_response(ds4_nf_io *io, int *status, uint64_t *length) {
    char h[8193]; size_t n = 0;
    while (n < sizeof(h) - 1) {
        if (!ds4_nf_read(io, h + n, 1)) return false;
        n++;
        if (n >= 4 && !memcmp(h + n - 4, "\r\n\r\n", 4)) break;
    }
    h[n] = 0;
    if (n < 4 || memcmp(h + n - 4, "\r\n\r\n", 4) || memchr(h, 0, n)) return false;
    char *line = strstr(h, "\r\n");
    if (!line) return false;
    *line = 0;
    if (strlen(h) < 12 || strncmp(h, "HTTP/1.1 ", 9) ||
        h[9] < '1' || h[9] > '5' || h[10] < '0' || h[10] > '9' ||
        h[11] < '0' || h[11] > '9' || (h[12] && h[12] != ' ')) return false;
    *status = (h[9]-'0')*100+(h[10]-'0')*10+h[11]-'0';
    bool have = false;
    for (line += 2; *line;) {
        char *end = strstr(line, "\r\n"); if (!end) return false;
        *end = 0; if (!*line) break;
        char *colon = strchr(line, ':');
        if (!colon || colon == line || *line == ' ' || *line == '\t') return false;
        *colon++ = 0;
        for (char *v = line; *v; v++) if (!isalnum((unsigned char)*v) && *v != '-') return false;
        while (*colon == ' ' || *colon == '\t') colon++;
        if (!strcasecmp(line, "Transfer-Encoding") || !strcasecmp(line, "Content-Encoding") ||
            !strcasecmp(line, "Location")) return false;
        if (!strcasecmp(line, "Content-Length")) {
            const char *value = colon;
            if (have || !ds4_nf_u64(&value, length)) return false;
            json_ws(&value); if (*value) return false; have = true;
        }
        line = end + 2;
    }
    return have;
}

static bool ds4_nf_request(ds4_nf_config *c, ds4_nf_io *io, const char *method,
                            const char *job, bool payload, const ds4_tokens *tokens,
                            int context, const char *token_sha, int *status,
                            uint64_t *length) {
    char h[2304];
    int n = snprintf(h, sizeof(h),
        "%s /dsv41-native-offload/jobs/%s%s HTTP/1.1\r\nHost: %s:%u\r\n"
        "X-Studio-Token: %s\r\nConnection: close\r\nContent-Length: %llu\r\n",
        method, job, payload ? "/payload" : "", c->host, ntohs(c->broker.sin_port),
        c->token, (unsigned long long)(tokens ? (uint64_t)tokens->len * 4 : 0));
    if (n < 0 || (size_t)n >= sizeof(h)) return false;
    if (tokens) {
        int k = snprintf(h+n, sizeof(h)-(size_t)n,
            "Content-Type: application/vnd.studio.dsv41.tokens-le32\r\n"
            "X-DSV41-Model-SHA256: %s\r\nX-DSV41-Tokens-SHA256: %s\r\n"
            "X-DSV41-Context: %d\r\nX-DSV41-Mode: 1\r\nX-DSV41-Tokens: %d\r\n",
            DS4_NF_MODEL_SHA, token_sha, context, tokens->len);
        if (k < 0 || (size_t)k >= sizeof(h)-(size_t)n) return false;
        n += k;
    }
    if ((size_t)n + 2 >= sizeof(h)) return false;
    h[n++] = '\r'; h[n++] = '\n';
    if (!ds4_nf_connect(c, io) || !ds4_nf_send(io, h, (size_t)n)) return false;
    if (tokens) {
        unsigned char page[4096];
        for (int off = 0; off < tokens->len;) {
            int count = tokens->len - off > 1024 ? 1024 : tokens->len-off;
            for (int i=0;i<count;i++) {
                uint32_t t=(uint32_t)tokens->v[off+i];
                for (int b=0;b<4;b++) page[4*i+b]=(unsigned char)(t>>(8*b));
            }
            if (!ds4_nf_send(io,page,(size_t)count*4)) return false;
            off += count;
        }
    }
    return ds4_nf_response(io, status, length);
}

static bool ds4_nf_json_body(ds4_nf_io *io, uint64_t length, ds4_nf_fields *f) {
    if (!length || length > DS4_NF_JSON_CAP) return false;
    char *text = malloc((size_t)length + 1); if (!text) return false;
    bool ok = ds4_nf_read(io, text, (size_t)length);
    text[length] = 0;
    if (ok) ok = ds4_nf_parse(text, (size_t)length, f);
    free(text); return ok;
}

static void ds4_nf_close(ds4_nf_config *c) {
    if (!c) return;
    if (c->dirfd >= 0) close(c->dirfd);
    memset(c->token, 0, sizeof(c->token)); free(c);
}

static const char *ds4_nf_env(const char *name) {
    const char *v = getenv(name); return v && *v ? v : NULL;
}

static bool ds4_nf_env_uint(const char *name, uint32_t *out, uint32_t min, uint32_t max) {
    const char *value = ds4_nf_env(name); uint64_t n;
    if (!value) return false;
    if (!ds4_nf_u64(&value, &n) || *value || n < min || n > max) return false;
    *out = (uint32_t)n; return true;
}

static int ds4_nf_setup(ds4_engine *engine, const ds4_engine_options *opt,
                        ds4_nf_config **out, char *err, size_t cap) {
    *out = NULL;
    const char *enable = ds4_nf_env("DS4_NATIVE_FRONTIER");
    if (!enable || !strcmp(enable,"0")) return 0;
    if (strcmp(enable,"1") || !ds4_engine_is_deepseek41(engine) ||
        opt->backend != DS4_BACKEND_METAL || opt->dspark || opt->mtp_path ||
        ds4_engine_has_mtp(engine) || opt->directional_steering_file || opt->load_slice ||
        opt->tp.role != DS4_TP_NONE || opt->distributed.role != DS4_DISTRIBUTED_NONE)
        return ds4_nf_error(err,cap,"enabled hook requires original serial Metal mode 1");
    const char *host=ds4_nf_env("DS4_NATIVE_FRONTIER_IPV4");
    /* Activation may provision the already paired RTX credential in a private
     * file; never infer that this Mac's local install token is accepted there. */
    const char *token_file=ds4_nf_env("DS4_NATIVE_FRONTIER_PAIR_TOKEN_FILE");
    const char *directory=ds4_nf_env("DS4_NATIVE_FRONTIER_DIRECTORY");
    const char *qualification=ds4_nf_env("DS4_NATIVE_FRONTIER_QUALIFICATION");
    const char *pin=ds4_nf_env("DS4_NATIVE_FRONTIER_QUALIFICATION_SHA256");
    const char *renderer=ds4_nf_env("DS4_NATIVE_FRONTIER_RENDERER_SHA256");
    if (!host || !token_file || !directory || !qualification || !pin ||
        !renderer || !ds4_nf_hex(renderer,64) ||
        strlen(host)>15 || !ds4_nf_hex(pin,64))
        return ds4_nf_error(err,cap,"missing fixed broker/auth/qualification configuration");
    ds4_nf_config *c=calloc(1,sizeof(*c)); if (!c) return ds4_nf_error(err,cap,"allocation failed");
    c->dirfd=-1; uint32_t port;
    if (!ds4_nf_pair_token(token_file,c->token) ||
        !ds4_nf_env_uint("DS4_NATIVE_FRONTIER_PORT",&port,1,65535) ||
        !ds4_nf_env_uint("DS4_NATIVE_FRONTIER_MIN_TOKENS",&c->minimum_tokens,1,1048575) ||
        !ds4_nf_env_uint("DS4_NATIVE_FRONTIER_MAX_TOKENS",&c->maximum_tokens,c->minimum_tokens,1048575) ||
        !ds4_nf_env_uint("DS4_NATIVE_FRONTIER_DEADLINE_SECONDS",&c->deadline_seconds,1,7200) ||
        inet_pton(AF_INET,host,&c->broker.sin_addr)!=1 || strlen(directory)>=sizeof(c->directory)) goto bad;
    c->broker.sin_family=AF_INET; c->broker.sin_port=htons((uint16_t)port);
    snprintf(c->host,sizeof(c->host),"%s",host);
    snprintf(c->directory,sizeof(c->directory),"%s",directory);
    snprintf(c->qualification_sha256,sizeof(c->qualification_sha256),"%s",pin);
    c->dirfd=ds4_nf_physical_dir(directory);
    if (c->dirfd<0 || fstat(c->dirfd,&c->dir_stat) || !ds4_nf_directory(c) ||
        !ds4_engine_native_frontier_identity(engine,&c->model)) goto bad;
    /* Qualification is small, read-only, externally pinned, and cannot introduce
     * a broker address or model path. Native identity is compared to the fd that
     * engine_open actually mapped, not a later pathname reopen. */
    int qfd=open(qualification,O_RDONLY|O_NOFOLLOW|O_CLOEXEC);
    char sha[65]; struct stat st; ds4_nf_fields f;
    bool ok=qfd>=0 && !fstat(qfd,&st) && st.st_size>0 && st.st_size<=DS4_NF_QUALIFICATION_CAP &&
            !flock(qfd,LOCK_SH|LOCK_NB);
    char *text=ok?malloc((size_t)st.st_size+1):NULL;
    if (ok && !text) ok=false;
    if (ok) {
        ssize_t got=pread(qfd,text,(size_t)st.st_size,0); text[st.st_size]=0;
        ok=got==st.st_size;
        if(ok) {
            ds4_nf_hash_bytes(text,(size_t)st.st_size,sha);
            ok=!strcmp(sha,pin) && ds4_nf_parse_bounded(text,(size_t)st.st_size,&f,DS4_NF_QUALIFICATION_CAP);
        }
    }
    free(text); if(qfd>=0)close(qfd);
    uint64_t required=(UINT64_C(1)<<0)|(UINT64_C(1)<<3)|(UINT64_C(1)<<6)|(UINT64_C(1)<<7)|
        (UINT64_C(1)<<11)|(UINT64_C(1)<<12)|(((UINT64_C(1)<<23)-1)&~((UINT64_C(1)<<14)-1));
    if (!ok || !ds4_nf_have(&f,required) || strcmp(f.schema,"dsv41-native-offload-qualification-v1") ||
        strcmp(f.model_sha256,DS4_NF_MODEL_SHA) || f.mode!=1 || f.wire_version!=2 ||
        strcmp(f.tokenization,"native-final-token-ids-le32-v1") ||
        !f.native_parity_passed || !f.performance_passed ||
        !ds4_nf_hex(f.producer_binary_sha256,64) || !ds4_nf_hex(f.producer_source_revision,40) ||
        !ds4_nf_hex(f.receiver_binary_sha256,64) || strcmp(f.renderer_sha256,renderer) ||
        f.receiver_model_size!=c->model.size || f.receiver_model_device!=c->model.device ||
        f.receiver_model_inode!=c->model.inode || f.receiver_model_mtime_ns!=c->model.mtime_ns) goto bad;
    char executable[PATH_MAX]; uint32_t exelen=sizeof(executable);
    if (_NSGetExecutablePath(executable,&exelen)) goto bad;
    int exe=open(executable,O_RDONLY|O_CLOEXEC);
    ok=exe>=0 && ds4_nf_hash_file(exe,UINT64_C(128)*1024*1024,sha,NULL) && !strcmp(sha,f.receiver_binary_sha256);
    if(exe>=0)close(exe); if(!ok)goto bad;
    snprintf(c->producer_binary_sha256,sizeof(c->producer_binary_sha256),"%s",f.producer_binary_sha256);
    snprintf(c->producer_source_revision,sizeof(c->producer_source_revision),"%s",f.producer_source_revision);
    *out=c; return 0;
bad:
    ds4_nf_close(c);
    return ds4_nf_error(err,cap,"broker configuration or pinned qualification failed validation");
}

static bool ds4_nf_bound(ds4_nf_config *c, ds4_engine *engine) {
    ds4_native_frontier_model_identity now;
    return c && ds4_engine_native_frontier_identity(engine,&now) && ds4_nf_same_identity(&now,&c->model);
}

/* Returns an already verified read-only lease. No partial download or invalid
 * native header reaches load_payload. Retained files are evidence, never removed. */
static int ds4_nf_fetch(ds4_nf_config *c,const ds4_tokens *tokens,int ctx,
                        ds4_nf_cancel_fn cancel,void *ud,FILE **payload,uint64_t *bytes,
                        struct stat *lease_identity,char *err,size_t cap) {
    *payload=NULL; char job[33],token_sha[65]; unsigned char nonce[16];
    if(!random_bytes(nonce,sizeof(nonce)) || !ds4_nf_token_hash(tokens,token_sha))
        return ds4_nf_error(err,cap,"nonce or token validation failed");
    for(int i=0;i<16;i++)snprintf(job+2*i,3,"%02x",nonce[i]);
    double deadline=ds4_nf_now()+c->deadline_seconds;
    ds4_nf_io io={.fd=-1,.deadline=deadline,.cancelled=cancel,.cancel_ud=ud};
    int status;uint64_t length;ds4_nf_fields f;bool submitted=false,ok=false;
    int fd=-1;FILE *file=NULL;char name[64];snprintf(name,sizeof(name),"native-frontier-%s.partial",job);
    submitted=true; /* Even an invalid/lost response may already have queued it. */
    if(!ds4_nf_request(c,&io,"POST",job,false,tokens,ctx,token_sha,&status,&length) ||
       status!=202 || !ds4_nf_json_body(&io,length,&f))goto done;
    close(io.fd);io.fd=-1;
    for(;;) {
        if(!ds4_nf_have(&f,7) || strcmp(f.schema,"dsv41-native-offload-v1") || strcmp(f.job_id,job))goto done;
        if(!strcmp(f.state,"completed"))break;
        if(strcmp(f.state,"queued") && strcmp(f.state,"running"))goto done;
        if(ds4_nf_now()>=deadline || cancel(ud))goto done;
        (void)poll(NULL,0,100);
        if(!ds4_nf_request(c,&io,"GET",job,false,NULL,ctx,NULL,&status,&length) ||
           status!=200 || !ds4_nf_json_body(&io,length,&f))goto done;
        close(io.fd);io.fd=-1;
    }
    if(!ds4_nf_completed(&f,job,token_sha,ctx,tokens->len,c->producer_binary_sha256,
                         c->producer_source_revision,c->qualification_sha256) ||
       !ds4_nf_space(c,f.payload_bytes))goto done;
    /* Recheck physical parent and actual remaining volume budget immediately
     * before exclusive create. openat is anchored to the retained parent. */
    if(!ds4_nf_directory(c) || !ds4_nf_space(c,f.payload_bytes))goto done;
    fd=openat(c->dirfd,name,O_CREAT|O_EXCL|O_RDWR|O_NOFOLLOW|O_CLOEXEC,0600);
    struct stat created;
    if(fd<0 || fstat(fd,&created) || !S_ISREG(created.st_mode) || created.st_nlink!=1 ||
       created.st_uid!=geteuid() || created.st_size!=0 || flock(fd,LOCK_EX|LOCK_NB))goto done;
    fprintf(stderr,"ds4-server: native frontier retains %s/%s\n",c->directory,name);
    if(!ds4_nf_request(c,&io,"GET",job,true,NULL,ctx,NULL,&status,&length) ||
       status!=200 || length!=f.payload_bytes)goto done;
    unsigned char *page=malloc(DS4_NF_CHUNK);if(!page)goto done;
    bool received=true;
    for(uint64_t off=0;off<length;) {
        size_t n=length-off>DS4_NF_CHUNK?DS4_NF_CHUNK:(size_t)(length-off);
        if(!ds4_nf_space(c,length-off) || !ds4_nf_read(&io,page,n)) {received=false;break;}
        for(size_t written=0;written<n;) {
            ssize_t k=write(fd,page+written,n-written);
            if(k<0 && errno==EINTR)continue;
            if(k<=0) {received=false;break;}
            written+=(size_t)k;
        }
        if(!received)break;off+=n;
    }
    free(page);if(!received || fsync(fd) || cancel(ud) || !ds4_nf_directory(c))goto done;
    char sha[65];struct stat finished,named;
    if(!ds4_nf_hash_file(fd,length,sha,&io) || strcmp(sha,f.payload_sha256) ||
       fstat(fd,&finished) || fstatat(c->dirfd,name,&named,AT_SYMLINK_NOFOLLOW) ||
       finished.st_dev!=created.st_dev || finished.st_ino!=created.st_ino ||
       named.st_dev!=created.st_dev || named.st_ino!=created.st_ino ||
       finished.st_size!=(off_t)length || named.st_size!=finished.st_size)goto done;
    unsigned char header[52],v[4096];
    if(pread(fd,header,52,0)!=52 || !ds4_nf_header(header,(uint32_t)ctx,(uint32_t)tokens->len,length))goto done;
    for(int off=0;off<tokens->len;) {
        int count=tokens->len-off>1024?1024:tokens->len-off;
        if(cancel(ud) || ds4_nf_now()>=deadline ||
           pread(fd,v,(size_t)count*4,52+(off_t)off*4)!=(ssize_t)count*4)goto done;
        for(int i=0;i<count;i++)if(ds4_nf_le32(v+4*i)!=(uint32_t)tokens->v[off+i])goto done;
        off+=count;
    }
    /* Reopen only our exact O_EXCL inode read-only, and rehash across the
     * close/reopen boundary. Identity is rechecked again by the import owner. */
    int ro=openat(c->dirfd,name,O_RDONLY|O_NOFOLLOW|O_CLOEXEC);
    if(ro<0)goto done;
    struct stat read_stat;
    bool reopen_ok=!fstat(ro,&read_stat) && read_stat.st_dev==created.st_dev &&
        read_stat.st_ino==created.st_ino && read_stat.st_size==(off_t)length;
    if(reopen_ok) {close(fd);fd=-1;reopen_ok=!flock(ro,LOCK_SH|LOCK_NB) &&
        ds4_nf_hash_file(ro,length,sha,&io) && !strcmp(sha,f.payload_sha256) &&
        !fstat(ro,lease_identity) && lease_identity->st_nlink==1;}
    if(!reopen_ok) {close(ro);goto done;}
    file=fdopen(ro,"rb");if(!file) {close(ro);goto done;}
    if(cancel(ud) || ds4_nf_now()>=deadline)goto done;
    *payload=file;file=NULL;*bytes=length;ok=true;
done:
    if(io.fd>=0)close(io.fd);
    if(fd>=0)close(fd);if(file)fclose(file);
    if(!ok && submitted) {
        ds4_nf_io stop={.fd=-1,.deadline=ds4_nf_now()+2};
        (void)ds4_nf_request(c,&stop,"DELETE",job,false,NULL,ctx,NULL,&status,&length);
        if(stop.fd>=0)close(stop.fd);
    }
    return ok?0:ds4_nf_error(err,cap,"broker failed/cancelled or frontier validation rejected; no refill fallback");
}
#else
static int ds4_nf_setup(ds4_engine *engine,const ds4_engine_options *opt,
                        ds4_nf_config **out,char *err,size_t cap) {
    (void)engine;(void)opt;*out=NULL;
    const char *enable=getenv("DS4_NATIVE_FRONTIER");
    if(enable && strcmp(enable,"0")) {
        snprintf(err,cap,"native frontier: disabled in this build (requires qualified Apple mode-1 build)");
        return -1;
    }
    return 0;
}
static void ds4_nf_close(ds4_nf_config *c) {(void)c;}
#endif
