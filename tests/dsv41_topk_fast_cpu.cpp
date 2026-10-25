// LLAMADECODE: CPU gate for the fast exact top-512 select (--topk-kernel fast).
//
// current_pipeline(): a CPU model of today's GPU selector for nrows>1, k=512,
// ncols>8192 (ggml-cuda top-k.cu + argsort.cu): 8192-column tiles, each the CUB
// radix sort of packed (ordered value, ~index) keys - or, for a tile holding a
// NaN, subnormal or invalid id, the original bitonic comparison network with
// FTZ comparisons - keeping 512, then merge passes over the candidate ids until
// one tile remains. Both tile variants are modelled (radix=LLAMA_DSV41_TOPK_RADIX
// on; network=off).
// fast_select(): the kernel's algorithm, sequentially, using the SAME
// dsv41-topk-select.cuh helpers the CUDA kernel uses: 4 x 8-bit radix select on
// ordered value bits, index-ordered collection, final sort of 512 packed keys.
// A NaN/subnormal row is reported dirty: on the GPU it then runs the unchanged
// current pipeline (row-filtered), so its output is the current one by construction.
//
// Checks, per synthetic row: fast == current(radix) == current(network) ==
// documented oracle (value desc with -0 == +0, index asc) for clean rows; dirty
// rows are flagged. With a directory argument it also replays recorded inputs
// (LLAMADECODE_TOPK_DUMP files) and requires both models to equal the ids the
// GPU produced.
#include "../engines/streaming-llama/prefill_llama.cpp/ggml/src/ggml-cuda/dsv41-topk-select.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace S = dsv41_topk_select;
static int failures = 0, checks = 0;
static void expect(bool ok, const std::string &what) {
    ++checks;
    if (!ok) { ++failures; std::printf("FAIL %s\n", what.c_str()); }
}
static uint32_t bits_of(float v) { uint32_t b; std::memcpy(&b, &v, 4); return b; }
static float ftz(float v) {  // CUDA -use_fast_math comparisons flush subnormals to signed zero
    const uint32_t b = bits_of(v);
    if ((b & 0x7fffffffu) && (b & 0x7fffffffu) < 0x00800000u) { const uint32_t z = b & 0x80000000u; float r; std::memcpy(&r, &z, 4); return r; }
    return v;
}

// ---- CPU model of the current selector -------------------------------------
// top_k_candidate_before (argsort.cu) evaluated with FTZ float comparisons.
static bool before(const float *x, int lhs, int rhs) {
    if (lhs < 0) return false;
    if (rhs < 0) return true;
    const float a = ftz(x[lhs]), b = ftz(x[rhs]);
    return a > b || (a == b && lhs < rhs);
}
// top_k_sort_candidates_strided: the bitonic network over width_pad slots.
static void network(int width_pad, const float *x, std::vector<int> &c) {
    for (int k = 2; k <= width_pad; k *= 2)
        for (int j = k / 2; j > 0; j /= 2)
            for (int col = 0; col < width_pad; ++col) {
                const int peer = col ^ j;
                if (peer > col) {
                    const bool descending = (col & k) == 0;
                    const bool swap = descending ? before(x, c[peer], c[col]) : before(x, c[col], c[peer]);
                    if (swap) std::swap(c[col], c[peer]);
                }
            }
}
static int next_pow2(int v) { int p = 1; while (p < v) p *= 2; return p; }
// One 8192-tile of k_dsv41_topk_radix_tiles (radix=true) or k_top_k_f32_i32_tiles (radix=false).
static void tile_select(const float *x, const int *ids, int tile_width, bool radix, int *out) {
    const int padded = next_pow2(tile_width);
    std::vector<int> c(size_t(radix ? 8192 : padded), -1);
    for (int i = 0; i < tile_width; ++i) c[size_t(i)] = ids[i];
    bool odd = false;
    for (int i = 0; i < tile_width; ++i) {
        const int id = c[size_t(i)];
        if (id < -1) odd = true;
        if (id >= 0 && S::classify(bits_of(x[id]))) odd = true;
    }
    if (!radix || odd) {
        std::vector<int> n((size_t)padded, -1);
        for (int i = 0; i < padded && i < int(c.size()); ++i) n[size_t(i)] = c[size_t(i)];
        network(padded, x, n);
        for (int i = 0; i < S::k; ++i) out[i] = i < padded ? n[size_t(i)] : -1;
        return;
    }
    std::vector<uint64_t> keys(8192, 0);
    for (int i = 0; i < 8192; ++i) {
        const int id = c[size_t(i)];
        if (id < 0) continue;  // dsv41_topk_pack_key: invalid id -> 0
        const float v = x[id];
        keys[size_t(i)] = S::pack(S::ordered(bits_of(ftz(v) == 0.0f ? 0.0f : v)), id);
    }
    std::sort(keys.begin(), keys.end(), std::greater<uint64_t>());
    for (int i = 0; i < S::k; ++i) out[i] = int(0xffffffffu - uint32_t(keys[size_t(i)]));
}
// ggml_cuda_op_top_k: first pass over columns, merge passes over candidates.
static std::vector<int> current_pipeline(const float *x, int ncols, bool radix) {
    const int T = 8192, k = S::k;
    int width = ncols;
    std::vector<int> cand((size_t)ncols);
    std::iota(cand.begin(), cand.end(), 0);
    do {
        const int ntiles = (width + T - 1) / T;
        std::vector<int> next(size_t(ntiles) * k);
        for (int t = 0; t < ntiles; ++t)
            tile_select(x, cand.data() + size_t(t) * T, std::min(T, width - t * T), radix, next.data() + size_t(t) * k);
        cand.swap(next);
        width = ntiles * k;
    } while (width > k);
    return cand;
}

// ---- the fast kernel's algorithm -------------------------------------------
static bool fast_select(const float *x, int ncols, std::vector<int> &out) {
    for (int c = 0; c < ncols; ++c) if (S::classify(bits_of(x[c]))) return false;  // dirty
    std::vector<uint32_t> o((size_t)ncols);
    for (int c = 0; c < ncols; ++c) o[size_t(c)] = S::ordered(bits_of(x[c]));
    S::state st{0u, 0u, uint32_t(S::k)};
    for (int shift = 24;; shift -= 8) {
        uint32_t hist[256] = {};
        for (int c = 0; c < ncols; ++c) if (S::in_region(st, o[size_t(c)])) ++hist[(o[size_t(c)] >> shift) & 0xffu];
        uint32_t incl = 0, bin = 0, above_bin = 0, in_bin = 0; bool found = false;
        for (int b = 255; b >= 0; --b) {
            const uint32_t excl = incl; incl += hist[b];
            if (!found && excl < st.need && st.need <= incl) { bin = uint32_t(b); above_bin = excl; in_bin = hist[b]; found = true; }
        }
        if (!found) return expect(false, "fast: no bin holds the k-th value"), false;
        if (S::advance(st, shift, bin, above_bin, in_bin)) break;
    }
    std::vector<uint64_t> sel;
    uint32_t taken = 0;
    for (int c = 0; c < ncols; ++c) {
        const uint32_t v = o[size_t(c)];
        if (S::above(st, v)) sel.push_back(S::pack(v, c));
        else if (S::in_region(st, v) && taken < st.need) { sel.push_back(S::pack(v, c)); ++taken; }
    }
    expect(sel.size() == size_t(S::k), "fast: exactly 512 winners");
    std::sort(sel.begin(), sel.end(), std::greater<uint64_t>());
    out.resize(sel.size());
    for (size_t i = 0; i < sel.size(); ++i) out[i] = S::unpack_id(sel[i]);
    return true;
}

// Documented semantics for a clean row: value descending (-0 == +0), index ascending.
static std::vector<int> oracle(const float *x, int ncols) {
    std::vector<int> id((size_t)ncols);
    std::iota(id.begin(), id.end(), 0);
    std::stable_sort(id.begin(), id.end(), [&](int a, int b) { return x[a] > x[b]; });
    id.resize(size_t(S::k));
    return id;
}

static void check_row(const std::vector<float> &row, const std::string &name, bool expect_dirty) {
    const int n = int(row.size());
    const auto radix = current_pipeline(row.data(), n, true);
    std::vector<int> fast;
    const bool clean = fast_select(row.data(), n, fast);
    expect(clean == !expect_dirty, name + ": dirty classification");
    if (!clean) return;  // GPU: this row runs the unchanged current pipeline
    const auto net = current_pipeline(row.data(), n, false);
    expect(fast == radix, name + ": fast == current (radix tiles), ids and order");
    expect(net == radix, name + ": current radix == current network");
    expect(radix == oracle(row.data(), n), name + ": current == documented order");
}

static int replay(const char *dir) {
    int rows = 0, files = 0;
    std::vector<std::filesystem::path> paths;
    for (const auto &e : std::filesystem::directory_iterator(dir))
        if (e.path().extension() == ".bin" && e.path().filename().string().rfind("topk-", 0) == 0) paths.push_back(e.path());
    std::sort(paths.begin(), paths.end());
    for (const auto &p : paths) {
        FILE *f = std::fopen(p.string().c_str(), "rb");
        uint32_t h[7];
        if (!f || std::fread(h, 4, 7, f) != 7 || h[0] != 0x31444b54u || h[4] != uint32_t(S::k)) { expect(false, "dump header " + p.string()); if (f) std::fclose(f); continue; }
        ++files;
        std::vector<float> s(h[3]); std::vector<int> ids(h[4]);
        for (uint32_t i = 0; i < h[6]; ++i) {
            uint32_t r;
            if (std::fread(&r, 4, 1, f) != 1 || std::fread(s.data(), 4, s.size(), f) != s.size() || std::fread(ids.data(), 4, ids.size(), f) != ids.size()) { expect(false, "dump body " + p.string()); break; }
            const std::string name = p.filename().string() + " row " + std::to_string(r);
            const auto cur = current_pipeline(s.data(), int(s.size()), true);
            expect(cur == ids, name + ": CPU current model == recorded GPU ids");
            std::vector<int> fast;
            if (fast_select(s.data(), int(s.size()), fast)) expect(fast == ids, name + ": fast == recorded GPU ids");
            ++rows;
        }
        std::fclose(f);
    }
    std::printf("replayed %d recorded rows from %d files\n", rows, files);
    return files;
}

int main(int argc, char **argv) {
    std::mt19937 rng(1017);
    std::normal_distribution<float> gauss(0.f, 1.f);
    const float inf = std::numeric_limits<float>::infinity();
    for (const int n : {8193, 10832, 16384, 23417, 40000, 131072}) {
        const std::string N = "ncols=" + std::to_string(n);
        std::vector<float> row((size_t)n);
        for (auto &v : row) v = gauss(rng);
        check_row(row, N + " gaussian", false);
        for (auto &v : row) v = std::round(gauss(rng) * 8.f) / 8.f;
        check_row(row, N + " coarse ties", false);
        // Indexer-like: sum over 32 heads of w_h * relu(dot), BF16-rounded dots: exact zeros and ties.
        for (auto &v : row) {
            float s = 0.f;
            for (int h = 0; h < 32; ++h) { float d = gauss(rng); uint32_t b = bits_of(d) & 0xffff0000u; std::memcpy(&d, &b, 4); s += (0.02f + 0.01f * h) * std::max(0.f, d - 1.5f); }
            v = s * (1.f / 64.f);
        }
        check_row(row, N + " indexer-like", false);
        for (const int visible : {0, 1, 100, 511, 512, 513, 4096}) {  // causal mask: -inf tail
            if (visible > n) continue;
            for (int c = 0; c < n; ++c) row[size_t(c)] = c < visible ? gauss(rng) : -inf;
            check_row(row, N + " visible=" + std::to_string(visible), false);
        }
        std::fill(row.begin(), row.end(), 1.0f);
        check_row(row, N + " all equal", false);
        for (int c = 0; c < n; ++c) row[size_t(c)] = (c % 3 == 0) ? -0.0f : 0.0f;
        for (int c = 0; c < 300; ++c) row[size_t(rng() % unsigned(n))] = std::fabs(gauss(rng)) + 0.5f;
        check_row(row, N + " signed zeros at the cut", false);
        for (int c = 0; c < n; ++c) row[size_t(c)] = (c % 7 == 0) ? inf : ((c % 5 == 0) ? -inf : gauss(rng));
        check_row(row, N + " infinities", false);
        for (int c = 0; c < n; ++c) row[size_t(c)] = 1.0f + float(rng() % 4096) * 1.1920929e-7f;  // low-bit ties
        check_row(row, N + " low-bit values", false);
        for (int c = 0; c < n; ++c) row[size_t(c)] = -std::fabs(gauss(rng)) * 1e30f;
        check_row(row, N + " large negatives", false);
        for (auto &v : row) v = gauss(rng);
        row[size_t(rng() % unsigned(n))] = std::numeric_limits<float>::quiet_NaN();
        check_row(row, N + " one NaN", true);
        for (auto &v : row) v = gauss(rng);
        row[size_t(rng() % unsigned(n))] = 1e-40f;
        check_row(row, N + " one subnormal", true);
        for (auto &v : row) v = gauss(rng);
        row[size_t(n - 1)] = -1e-41f;
        check_row(row, N + " negative subnormal last", true);
    }
    if (argc > 1 && replay(argv[1]) == 0) expect(false, std::string("no recorded dumps in ") + argv[1]);
    std::printf("%d checks, %d failures\n%s\n", checks, failures, failures ? "FAILED" : "ALL PASS");
    return failures ? 1 : 0;
}
