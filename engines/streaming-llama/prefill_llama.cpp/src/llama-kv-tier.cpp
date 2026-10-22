#include "llama-kv-tier.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#define KV_TIER_FSEEK _fseeki64
#define KV_TIER_MKDIR(p) _mkdir(p)
#else
#include <unistd.h>
#define KV_TIER_FSEEK fseeko
#define KV_TIER_MKDIR(p) mkdir(p, 0755)
#endif

// ---- helpers -----------------------------------------------------------------------------------

static bool kv_tier_mkdir_p(const std::string & path) {
    std::string cur;
    for (size_t i = 0; i < path.size(); ++i) {
        const char c = path[i];
        cur.push_back(c);
        const bool sep = (c == '/' || c == '\\');
        if (sep || i + 1 == path.size()) {
            std::string d = sep ? cur.substr(0, cur.size() - 1) : cur;
            if (d.empty() || (d.size() == 2 && d[1] == ':')) {
                continue;
            }
            struct stat st;
            if (stat(d.c_str(), &st) != 0) {
                if (KV_TIER_MKDIR(d.c_str()) != 0 && errno != EEXIST) {
                    return false;
                }
            }
        }
    }
    return true;
}

static uint32_t kv_tier_crc_table[256];
static bool     kv_tier_crc_table_ready = false;

static void kv_tier_crc_init() {
    if (kv_tier_crc_table_ready) {
        return;
    }
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int j = 0; j < 8; ++j) {
            c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        }
        kv_tier_crc_table[i] = c;
    }
    kv_tier_crc_table_ready = true;
}

uint32_t llama_kv_tier::crc32_update(uint32_t crc, const uint8_t * data, size_t n) {
    kv_tier_crc_init();
    uint32_t c = crc ^ 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) {
        c = kv_tier_crc_table[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

// ---- llama_kv_tier -----------------------------------------------------------------------------

bool llama_kv_tier::enabled() {
    const char * e = std::getenv("LLAMA_KV_TIER_DIR");
    return e && e[0] != '\0';
}

std::string llama_kv_tier::root_dir() {
    const char * e = std::getenv("LLAMA_KV_TIER_DIR");
    return e ? std::string(e) : std::string();
}

const char * llama_kv_tier::kind_ext(kind k) {
    switch (k) {
        case K:    return "k";
        case V:    return "v";
        case KIDX: return "kidx";
    }
    return "bin";
}

llama_kv_tier::~llama_kv_tier() {
    close();
}

bool llama_kv_tier::open(const std::string & model_fp, const std::string & conv_id, int n_layer,
                         const std::vector<size_t> & k_row_bytes, const std::vector<size_t> & v_row_bytes,
                         const std::vector<size_t> & kidx_row_bytes, std::string & err) {
    close();
    root_ = root_dir();
    if (root_.empty()) {
        err = "LLAMA_KV_TIER_DIR is not set";
        return false;
    }
    model_fp_ = model_fp;
    conv_id_  = conv_id;
    path_ = root_ + "/" + model_fp + "/" + conv_id;
    if (!kv_tier_mkdir_p(path_)) {
        err = "cannot create " + path_;
        return false;
    }
    layers_.resize((size_t) n_layer);
    for (int il = 0; il < n_layer; ++il) {
        layer_files & lf = layers_[(size_t) il];
        lf.row_bytes[K]    = il < (int) k_row_bytes.size()    ? k_row_bytes[(size_t) il]    : 0;
        lf.row_bytes[V]    = il < (int) v_row_bytes.size()    ? v_row_bytes[(size_t) il]    : 0;
        lf.row_bytes[KIDX] = il < (int) kidx_row_bytes.size() ? kidx_row_bytes[(size_t) il] : 0;
        for (int k = 0; k < 3; ++k) {
            if (lf.row_bytes[k] == 0) {
                continue;
            }
            const std::string fn = path_ + "/L" + std::to_string(il) + "." + kind_ext((kind) k);
            FILE * f = std::fopen(fn.c_str(), "w+b");
            if (!f) {
                err = "cannot create " + fn;
                close();
                return false;
            }
            lf.f[k] = f;
        }
    }
    n_tokens_ = 0;
    bytes_written_ = bytes_read_ = 0;
    failed_ = false;
    last_error_.clear();
    if (!write_manifest(err)) {
        return false;
    }
    {
        uint64_t after = 0;
        const uint64_t cap = cap_bytes();
        const size_t n_evicted = evict_to_cap(model_fp, path_, cap, &after);
        if (n_evicted > 0) {
            fprintf(stderr, "*** KV_TIER: evicted %zu older conversation(s) to stay under %.0f GB; %.1f GB kept\n",
                    n_evicted, cap / (1024.0 * 1024.0 * 1024.0), after / (1024.0 * 1024.0 * 1024.0));
            fflush(stderr);
        }
    }
    return true;
}

uint64_t llama_kv_tier::cap_bytes() {
    const char * e = std::getenv("LLAMA_KV_TIER_MAX_GB");
    double gb = 200.0;   // a 1M-token GLM-5.3 conversation is 107 GB; the cap must hold one whole conversation
    if (e && e[0] != 0) {
        const double v = std::atof(e);
        if (v > 0) {
            gb = v;
        }
    }
    return (uint64_t) (gb * 1024.0 * 1024.0 * 1024.0);
}

size_t llama_kv_tier::evict_to_cap(const std::string & model_fp, const std::string & keep_dir,
                                   uint64_t cap, uint64_t * total_after) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path base = fs::path(root_dir()) / model_fp;
    if (!fs::is_directory(base, ec)) {
        if (total_after) *total_after = 0;
        return 0;
    }
    struct conv { fs::file_time_type t; fs::path p; uint64_t bytes; bool keep; };
    std::vector<conv> convs;
    const fs::path keep_p = fs::path(keep_dir);
    uint64_t total = 0;
    for (const auto & e : fs::directory_iterator(base, ec)) {
        if (!e.is_directory(ec)) {
            continue;
        }
        uint64_t bytes = 0;
        for (const auto & f : fs::directory_iterator(e.path(), ec)) {
            if (f.is_regular_file(ec)) {
                bytes += (uint64_t) f.file_size(ec);
            }
        }
        fs::file_time_type t = fs::file_time_type::min();
        if (fs::exists(e.path() / "manifest.json", ec)) {
            t = fs::last_write_time(e.path() / "manifest.json", ec);
        }
        bool keep = false;
        if (!keep_dir.empty()) {
            keep = fs::equivalent(e.path(), keep_p, ec);
        }
        convs.push_back({t, e.path(), bytes, keep});
        total += bytes;
    }
    // newest first; the kept conversation is treated as newest
    std::sort(convs.begin(), convs.end(), [](const conv & a, const conv & b) {
        if (a.keep != b.keep) return a.keep;
        return a.t > b.t;
    });
    size_t n_evicted = 0;
    uint64_t running = 0;
    for (const auto & c : convs) {
        if (c.keep || running + c.bytes <= cap) {
            running += c.bytes;
            continue;
        }
        fs::remove_all(c.p, ec);
        if (ec) {
            fprintf(stderr, "*** KV_TIER: could not evict %s: %s\n", c.p.string().c_str(), ec.message().c_str());
            fflush(stderr);
            running += c.bytes;
            continue;
        }
        fprintf(stderr, "*** KV_TIER: evicted %s (%.1f GB)\n", c.p.filename().string().c_str(), c.bytes / (1024.0 * 1024.0 * 1024.0));
        fflush(stderr);
        n_evicted++;
    }
    if (total_after) *total_after = running;
    (void) total;
    return n_evicted;
}

void llama_kv_tier::close() {
    if (!layers_.empty()) {
        std::string err;
        write_manifest(err);       // the last word on rows and CRCs before the handles go
    }
    std::lock_guard<std::mutex> lk(mu_);
    for (auto & lf : layers_) {
        for (int k = 0; k < 3; ++k) {
            if (lf.f[k]) {
                std::fclose(lf.f[k]);
                lf.f[k] = nullptr;
            }
        }
    }
    layers_.clear();
}

bool llama_kv_tier::write_rows(int il, kind k, int64_t row0, int64_t n, const uint8_t * src, std::string & err) {
    std::lock_guard<std::mutex> lk(mu_);
    if (il < 0 || il >= (int) layers_.size()) {
        err = "layer out of range";
        return false;
    }
    layer_files & lf = layers_[(size_t) il];
    if (!lf.f[k] || lf.row_bytes[k] == 0) {
        err = "no file for this kind";
        return false;
    }
    if (n <= 0) {
        return true;
    }
    if (row0 != lf.n_rows[k]) {
        // The store is append-only: a write anywhere but the end would either rewrite rows (the
        // wear the design forbids) or leave a hole the CRC could not follow.
        err = "non-append write: layer " + std::to_string(il) + " kind " + std::to_string((int) k) +
              " row0=" + std::to_string(row0) + " stored=" + std::to_string(lf.n_rows[k]);
        failed_ = true;
        last_error_ = err;
        return false;
    }
    const size_t bytes = (size_t) n * lf.row_bytes[k];
    if (KV_TIER_FSEEK(lf.f[k], (int64_t) row0 * (int64_t) lf.row_bytes[k], SEEK_SET) != 0) {
        err = "seek failed";
        failed_ = true;
        last_error_ = err;
        return false;
    }
    if (std::fwrite(src, 1, bytes, lf.f[k]) != bytes) {
        err = "write failed (disk full?)";
        failed_ = true;
        last_error_ = err;
        return false;
    }
    std::fflush(lf.f[k]);
    lf.crc[k] = crc32_update(lf.crc[k], src, bytes);
    lf.n_rows[k] += n;
    bytes_written_ += bytes;
    if (k == K && lf.n_rows[K] > n_tokens_) {
        n_tokens_ = lf.n_rows[K];      // the rows are the tokens covered
    }
    return true;
}

bool llama_kv_tier::read_rows(int il, kind k, int64_t row0, int64_t n, uint8_t * dst, std::string & err) {
    std::lock_guard<std::mutex> lk(mu_);
    if (il < 0 || il >= (int) layers_.size()) {
        err = "layer out of range";
        return false;
    }
    layer_files & lf = layers_[(size_t) il];
    if (!lf.f[k] || lf.row_bytes[k] == 0) {
        err = "no file for this kind";
        return false;
    }
    if (n <= 0) {
        return true;
    }
    if (row0 < 0 || row0 + n > lf.n_rows[k]) {
        err = "read past the stored rows: layer " + std::to_string(il) + " rows " + std::to_string(row0) + "+" +
              std::to_string(n) + " stored=" + std::to_string(lf.n_rows[k]);
        return false;
    }
    const size_t bytes = (size_t) n * lf.row_bytes[k];
    if (KV_TIER_FSEEK(lf.f[k], (int64_t) row0 * (int64_t) lf.row_bytes[k], SEEK_SET) != 0) {
        err = "seek failed";
        return false;
    }
    if (std::fread(dst, 1, bytes, lf.f[k]) != bytes) {
        err = "short read";
        return false;
    }
    bytes_read_ += bytes;
    return true;
}

int64_t llama_kv_tier::rows(int il, kind k) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (il < 0 || il >= (int) layers_.size()) {
        return 0;
    }
    return layers_[(size_t) il].n_rows[k];
}

size_t llama_kv_tier::row_bytes(int il, kind k) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (il < 0 || il >= (int) layers_.size()) {
        return 0;
    }
    return layers_[(size_t) il].row_bytes[k];
}

uint32_t llama_kv_tier::crc(int il, kind k) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (il < 0 || il >= (int) layers_.size()) {
        return 0;
    }
    return layers_[(size_t) il].crc[k];
}

// ---- manifest parsing (the manifest is written by write_manifest; a flat scan is enough) ----------

static bool kv_tier_read_file(const std::string & path, std::string & out) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) {
        out.append(buf, n);
    }
    std::fclose(f);
    return true;
}

static long long kv_tier_json_int(const std::string & s, const char * key, long long dflt) {
    const std::string k = std::string("\"") + key + "\":";
    const size_t p = s.find(k);
    if (p == std::string::npos) {
        return dflt;
    }
    return std::strtoll(s.c_str() + p + k.size(), nullptr, 10);
}

bool llama_kv_tier::open_existing(const std::string & dir, std::string & err) {
    close();
    std::string manifest;
    if (!kv_tier_read_file(dir + "/manifest.json", manifest)) {
        err = "no manifest in " + dir;
        return false;
    }
    const int n_layer = (int) kv_tier_json_int(manifest, "n_layer", 0);
    if (n_layer <= 0) {
        err = "manifest without layers in " + dir;
        return false;
    }
    root_ = root_dir();
    path_ = dir;
    {
        // <root>/<model_fp>/<conv_id>
        const size_t slash = dir.find_last_of("/\\");
        conv_id_ = slash == std::string::npos ? dir : dir.substr(slash + 1);
        const size_t slash2 = slash == std::string::npos ? std::string::npos : dir.find_last_of("/\\", slash - 1);
        model_fp_ = (slash != std::string::npos && slash2 != std::string::npos) ? dir.substr(slash2 + 1, slash - slash2 - 1) : "";
    }
    layers_.resize((size_t) n_layer);
    // per-layer entries: {"il": N, "k": [row_bytes, n_rows, crc], "v": [...], "kidx": [...]}
    size_t pos = 0;
    for (int il = 0; il < n_layer; ++il) {
        const std::string key = "{\"il\": " + std::to_string(il) + ",";
        pos = manifest.find(key, pos);
        if (pos == std::string::npos) {
            err = "manifest is missing layer " + std::to_string(il);
            close();
            return false;
        }
        layer_files & lf = layers_[(size_t) il];
        const char * names[3] = {"\"k\": [", "\"v\": [", "\"kidx\": ["};
        for (int k = 0; k < 3; ++k) {
            const size_t q = manifest.find(names[k], pos);
            if (q == std::string::npos) {
                continue;
            }
            const char * c = manifest.c_str() + q + std::strlen(names[k]);
            char * end = nullptr;
            lf.row_bytes[k] = (size_t) std::strtoull(c, &end, 10);
            lf.n_rows[k]    = std::strtoll(end + 1, &end, 10);
            lf.crc[k]       = (uint32_t) std::strtoull(end + 1, &end, 10);
        }
        for (int k = 0; k < 3; ++k) {
            if (lf.row_bytes[k] == 0) {
                continue;
            }
            const std::string fn = path_ + "/L" + std::to_string(il) + "." + kind_ext((kind) k);
            FILE * f = std::fopen(fn.c_str(), "r+b");
            if (!f) {
                err = "cannot open " + fn;
                close();
                return false;
            }
            lf.f[k] = f;
        }
    }
    n_tokens_ = kv_tier_json_int(manifest, "n_tokens", 0);
    bytes_written_ = 0;
    bytes_read_ = 0;
    failed_ = false;
    last_error_.clear();
    return true;
}

bool llama_kv_tier::write_tokens(const int32_t * tokens, size_t n, std::string & err) {
    if (!is_open()) {
        err = "store not open";
        return false;
    }
    const std::string tmp = path_ + "/tokens.bin.tmp";
    const std::string fin = path_ + "/tokens.bin";
    FILE * f = std::fopen(tmp.c_str(), "wb");
    if (!f) {
        err = "cannot write " + tmp;
        return false;
    }
    const bool ok = n == 0 || std::fwrite(tokens, sizeof(int32_t), n, f) == n;
    std::fclose(f);
    if (!ok) {
        err = "short write to " + tmp;
        return false;
    }
    std::remove(fin.c_str());
    if (std::rename(tmp.c_str(), fin.c_str()) != 0) {
        err = "cannot rename tokens.bin";
        return false;
    }
    return true;
}

bool llama_kv_tier::lookup(const std::string & model_fp, const int32_t * tokens, size_t n, uint32_t lcp_min,
                           std::string & best_dir, int64_t & best_lcp, size_t max_scan) {
    namespace fs = std::filesystem;
    best_dir.clear();
    best_lcp = 0;
    if (!enabled() || n == 0) {
        return false;
    }
    const fs::path base = fs::path(root_dir()) / model_fp;
    std::error_code ec;
    if (!fs::is_directory(base, ec)) {
        return false;
    }
    std::vector<std::pair<fs::file_time_type, fs::path>> dirs;
    for (const auto & e : fs::directory_iterator(base, ec)) {
        if (!e.is_directory(ec)) {
            continue;
        }
        if (!fs::exists(e.path() / "tokens.bin", ec)) {
            continue;
        }
        dirs.emplace_back(fs::last_write_time(e.path() / "manifest.json", ec), e.path());
    }
    std::sort(dirs.begin(), dirs.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
    if (dirs.size() > max_scan) {
        dirs.resize(max_scan);
    }
    for (const auto & d : dirs) {
        std::string manifest;
        if (!kv_tier_read_file((d.second / "manifest.json").string(), manifest)) {
            continue;
        }
        const long long n_tok = kv_tier_json_int(manifest, "n_tokens", 0);
        if (n_tok <= (long long) lcp_min) {
            continue;
        }
        FILE * f = std::fopen((d.second / "tokens.bin").string().c_str(), "rb");
        if (!f) {
            continue;
        }
        const size_t n_cmp = (size_t) std::min<long long>(n_tok, (long long) n);
        std::vector<int32_t> stored(n_cmp);
        const size_t got = std::fread(stored.data(), sizeof(int32_t), n_cmp, f);
        std::fclose(f);
        size_t lcp = 0;
        while (lcp < got && stored[lcp] == tokens[lcp]) {
            lcp++;
        }
        if ((int64_t) lcp > best_lcp && lcp >= lcp_min) {
            best_lcp = (int64_t) lcp;
            best_dir = d.second.string();
        }
    }
    return best_lcp >= (int64_t) lcp_min;
}

bool llama_kv_tier::truncate_to(int64_t n_rows, std::string & err) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (n_rows < 0) {
            err = "negative row count";
            return false;
        }
        for (auto & lf : layers_) {
            for (int k = 0; k < 3; ++k) {
                if (lf.f[k] && lf.n_rows[k] > n_rows) {
                    lf.n_rows[k] = n_rows;
                    lf.crc[k] = 0;
                }
            }
        }
        if (n_tokens_ > n_rows) {
            n_tokens_ = n_rows;
        }
    }
    return write_manifest(err);
}

bool llama_kv_tier::commit(int64_t n_tokens, std::string & err) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        n_tokens_ = n_tokens;
    }
    return write_manifest(err);
}

bool llama_kv_tier::write_manifest(std::string & err) {
    std::lock_guard<std::mutex> lk(mu_);
    const std::string tmp = path_ + "/manifest.json.tmp";
    const std::string fin = path_ + "/manifest.json";
    FILE * f = std::fopen(tmp.c_str(), "wb");
    if (!f) {
        err = "cannot write " + tmp;
        return false;
    }
    std::fprintf(f, "{\n  \"version\": 1,\n  \"model_fp\": \"%s\",\n  \"conv_id\": \"%s\",\n  \"n_tokens\": %lld,\n  \"n_layer\": %d,\n  \"layers\": [\n",
                 model_fp_.c_str(), conv_id_.c_str(), (long long) n_tokens_, (int) layers_.size());
    for (size_t il = 0; il < layers_.size(); ++il) {
        const layer_files & lf = layers_[il];
        std::fprintf(f, "    {\"il\": %d, \"k\": [%zu, %lld, %u], \"v\": [%zu, %lld, %u], \"kidx\": [%zu, %lld, %u]}%s\n",
                     (int) il,
                     lf.row_bytes[K], (long long) lf.n_rows[K], lf.crc[K],
                     lf.row_bytes[V], (long long) lf.n_rows[V], lf.crc[V],
                     lf.row_bytes[KIDX], (long long) lf.n_rows[KIDX], lf.crc[KIDX],
                     il + 1 < layers_.size() ? "," : "");
    }
    std::fprintf(f, "  ],\n  \"bytes_written\": %llu\n}\n", (unsigned long long) bytes_written_);
    std::fclose(f);
    std::remove(fin.c_str());
    if (std::rename(tmp.c_str(), fin.c_str()) != 0) {
        err = "cannot rename manifest";
        return false;
    }
    return true;
}
