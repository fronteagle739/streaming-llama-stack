// LLAMADECODE: GPU gate + micro-benchmark for the fast exact top-512 select.
// Runs ggml_top_k(k=512) on the CUDA backend with the selector current
// (ggml_backend_cuda_dsv41_topk_fast_set(0)) and fast (1) over the same multi-row
// inputs and requires byte-identical ids for EVERY row: synthetic rows (gaussian,
// coarse ties, indexer-like, causal -inf tails, signed zeros, infinities,
// low-bit ties) mixed in one call with NaN/subnormal rows (those take the
// row-filtered current pipeline). Optional argument: a LLAMADECODE_TOPK_DUMP
// directory, whose recorded rows are replayed the same way and also compared
// with the ids recorded by the producer. Then times 4096-row calls at the 43K
// (10,832 keys) and 93K (23,417 keys) indexer shapes in both modes.
// Run with LLAMA_DSV41_TOPK_RADIX=1 (the producer's current selector).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-dsv41-topk.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <random>
#include <string>
#include <vector>

static int failures = 0;
static void expect(bool ok, const std::string &what) { std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str()); if (!ok) ++failures; }

struct runner {
    ggml_backend_t backend;
    std::vector<int32_t> run(const std::vector<float> &all, int64_t ncols, int64_t nrows, bool fast, int reps, double *ms) {
        ggml_backend_cuda_dsv41_topk_fast_set(fast ? 1 : 0);
        ggml_init_params params{16 * 1024 * 1024, nullptr, true};
        ggml_context *ctx = ggml_init(params);
        ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
        ggml_tensor *y = ggml_top_k(ctx, x, 512);
        ggml_cgraph *g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, y);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_tensor_set(x, all.data(), 0, all.size() * 4);
        const uint64_t q0 = ggml_backend_cuda_dsv41_topk_fast_queued();
        ggml_backend_graph_compute(backend, g);  // warm
        double best = 1e30;
        for (int r = 0; r < reps; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(backend, g);
            ggml_backend_synchronize(backend);
            best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        if (ms) *ms = best;
        const uint64_t launched = ggml_backend_cuda_dsv41_topk_fast_queued() - q0;
        expect(launched == (fast && nrows > 1 && ncols > 8192 ? uint64_t(reps + 1) : 0u), std::string("fast kernel launches ") + (fast ? "on" : "off"));
        std::vector<int32_t> out(size_t(512 * nrows));
        ggml_backend_tensor_get(y, out.data(), 0, out.size() * 4);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_cuda_dsv41_topk_fast_set(0);
        return out;
    }
};

static void compare(runner &R, const std::vector<float> &all, int64_t ncols, int64_t nrows, const std::string &name, const std::vector<int32_t> *recorded) {
    const auto cur = R.run(all, ncols, nrows, false, 1, nullptr);
    const uint64_t d0 = ggml_backend_cuda_dsv41_topk_fast_dirty_rows();
    const auto fast = R.run(all, ncols, nrows, true, 1, nullptr);
    const uint64_t dirty = (ggml_backend_cuda_dsv41_topk_fast_dirty_rows() - d0) / 2;  // warm + 1 rep
    int bad_rows = 0;
    for (int64_t r = 0; r < nrows; ++r) if (std::memcmp(cur.data() + r * 512, fast.data() + r * 512, 512 * 4)) ++bad_rows;
    expect(bad_rows == 0, name + ": fast ids byte-identical to current, all rows (fallback rows " + std::to_string(dirty) + ", differing rows " + std::to_string(bad_rows) + ")");
    if (recorded) expect(*recorded == cur, name + ": current == ids recorded by the producer");
}

int main(int argc, char **argv) {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::printf("FAIL no CUDA backend\n"); return 2; }
    const char *radix = std::getenv("LLAMA_DSV41_TOPK_RADIX");
    std::printf("current selector: %s\n", radix && radix[0] == '1' ? "radix tiles (LLAMA_DSV41_TOPK_RADIX=1)" : "bitonic tiles");
    runner R{backend};
    std::mt19937 rng(1017);
    std::normal_distribution<float> gauss(0.f, 1.f);
    const float inf = std::numeric_limits<float>::infinity();
    for (const int64_t n : {8193, 10832, 16384, 23417, 40000, 131072}) {
        const int64_t rows = 48;
        std::vector<float> all(size_t(n * rows));
        for (int64_t r = 0; r < rows; ++r) {
            float *row = all.data() + r * n;
            const int kind = int(r % 12);
            for (int64_t c = 0; c < n; ++c) {
                float v = gauss(rng);
                switch (kind) {
                    case 1: v = std::round(v * 8.f) / 8.f; break;
                    case 2: { float s = 0.f; for (int h = 0; h < 32; ++h) { float d = gauss(rng); uint32_t b; std::memcpy(&b, &d, 4); b &= 0xffff0000u; std::memcpy(&d, &b, 4); s += (0.02f + 0.01f * h) * std::max(0.f, d - 1.5f); } v = s / 64.f; break; }
                    case 3: v = c < 100 ? v : -inf; break;
                    case 4: v = c < 512 ? v : -inf; break;
                    case 5: v = c < 513 ? v : -inf; break;
                    case 6: v = c < int64_t(r * 997 % n) ? v : -inf; break;
                    case 7: v = (c % 3 == 0) ? -0.0f : 0.0f; if (rng() % 64 == 0) v = std::fabs(v) + 0.5f; break;
                    case 8: v = (c % 7 == 0) ? inf : ((c % 5 == 0) ? -inf : v); break;
                    case 9: v = 1.0f + float(rng() % 4096) * 1.1920929e-7f; break;
                    default: break;
                }
                row[c] = v;
            }
            if (kind == 10) row[rng() % unsigned(n)] = std::numeric_limits<float>::quiet_NaN();
            if (kind == 11) row[rng() % unsigned(n)] = 1e-40f;
        }
        compare(R, all, n, rows, "ncols=" + std::to_string(n) + " mixed 48 rows", nullptr);
    }
    if (argc > 1) {
        int files = 0;
        std::vector<std::filesystem::path> paths;
        for (const auto &e : std::filesystem::directory_iterator(argv[1]))
            if (e.path().extension() == ".bin" && e.path().filename().string().rfind("topk-", 0) == 0) paths.push_back(e.path());
        std::sort(paths.begin(), paths.end());
        for (const auto &p : paths) {
            FILE *f = std::fopen(p.string().c_str(), "rb");
            uint32_t h[7];
            if (!f || std::fread(h, 4, 7, f) != 7 || h[0] != 0x31444b54u || h[4] != 512) { expect(false, "dump header " + p.string()); if (f) std::fclose(f); continue; }
            std::vector<float> all(size_t(h[3]) * h[6]); std::vector<int32_t> ids(size_t(512) * h[6]);
            bool ok = true;
            for (uint32_t i = 0; i < h[6] && ok; ++i) {
                uint32_t r;
                ok = std::fread(&r, 4, 1, f) == 1 && std::fread(all.data() + size_t(i) * h[3], 4, h[3], f) == h[3] && std::fread(ids.data() + size_t(i) * 512, 4, 512, f) == 512;
            }
            std::fclose(f);
            expect(ok, "dump body " + p.string());
            if (ok && h[6] > 1) { compare(R, all, h[3], h[6], p.filename().string(), &ids); ++files; }
        }
        expect(files > 0, "replayed recorded dumps");
    }
    for (const int64_t n : {10832, 23417}) {  // 43K / 93K indexer shapes, one 4096-row tile
        const int64_t rows = 4096;
        std::vector<float> all(size_t(n * rows));
        for (int64_t r = 0; r < rows; ++r) {
            const int64_t visible = std::min<int64_t>(n, (n - 1024) + r / 4);  // the last tile of the sweep
            for (int64_t c = 0; c < n; ++c) {
                float s = 0.f;
                if (c < visible) { const float d = gauss(rng); s = std::max(0.f, d - 1.f) * 0.5f + 0.01f * d; }
                all[size_t(r * n + c)] = c < visible ? s : -inf;
            }
        }
        double cur_ms = 0, fast_ms = 0;
        const auto cur = R.run(all, n, rows, false, 5, &cur_ms);
        const auto fast = R.run(all, n, rows, true, 5, &fast_ms);
        expect(cur == fast, "timing shape ncols=" + std::to_string(n) + " identical");
        std::printf("TIMING ncols=%lld rows=%lld current_ms=%.3f fast_ms=%.3f speedup=%.2fx\n", (long long) n, (long long) rows, cur_ms, fast_ms, cur_ms / std::max(fast_ms, 1e-9));
    }
    ggml_backend_free(backend);
    std::printf(failures ? "FAILED %d\n" : "ALL PASS\n", failures);
    return failures ? 1 : 0;
}
