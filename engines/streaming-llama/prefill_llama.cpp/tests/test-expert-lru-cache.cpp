#include "llama-expert-lru-cache.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int N_LAYERS = 43;
constexpr int TOP_K = 6;

#define REQUIRE(condition, message) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "FAIL: %s:%d: %s\n", __func__, __LINE__, message); \
        return false; \
    } \
} while (0)

struct oracle_case {
    const char * file_name;
    size_t       capacity;
    uint64_t     tokens;
    uint64_t     accesses;
    uint64_t     hits;
    uint64_t     misses;
    uint64_t     evictions;
};

// Oracle values for the global expert LRU policy, generated with:
//   cache = TierCache("p1", capacity)
//   for token in load_trace(path).tokens:
//       for layer, experts in enumerate(token.routes):
//           cache.lookup_layer(layer, experts)
// and recorded from cache.total for capacities 64, 256, and 2357 slabs.
constexpr oracle_case ORACLE_CASES[] = {
    { "code_4096_greedy_0.tsv",        64, 64, 16512,     0, 16512, 16448 },
    { "code_4096_greedy_0.tsv",       256, 64, 16512, 10824,  5688,  5432 },
    { "code_4096_greedy_0.tsv",      2357, 64, 16512, 16236,   276,     0 },
    { "math_4096_greedy_0.tsv",        64, 64, 16512,     0, 16512, 16448 },
    { "math_4096_greedy_0.tsv",       256, 64, 16512, 10824,  5688,  5432 },
    { "math_4096_greedy_0.tsv",      2357, 64, 16512, 16236,   276,     0 },
    { "needle_60000_greedy_0.tsv",     64, 64, 16512,     0, 16512, 16448 },
    { "needle_60000_greedy_0.tsv",    256, 64, 16512, 10824,  5688,  5432 },
    { "needle_60000_greedy_0.tsv",   2357, 64, 16512, 16236,   276,     0 },
    { "prose_4096_greedy_0.tsv",       64, 64, 16512,     0, 16512, 16448 },
    { "prose_4096_greedy_0.tsv",      256, 64, 16512, 10824,  5688,  5432 },
    { "prose_4096_greedy_0.tsv",     2357, 64, 16512, 16236,   276,     0 },
    { "structured_4096_greedy_0.tsv",  64, 64, 16512,     0, 16512, 16448 },
    { "structured_4096_greedy_0.tsv", 256, 64, 16512, 10824,  5688,  5432 },
    { "structured_4096_greedy_0.tsv",2357, 64, 16512, 16236,   276,     0 },
};

using routes = std::array<std::array<int, TOP_K>, N_LAYERS>;

bool parse_routes(const std::string & value, routes & result) {
    std::istringstream layers(value);
    std::string layer_text;
    int expected_layer = 0;
    while (std::getline(layers, layer_text, ';')) {
        if (expected_layer >= N_LAYERS) {
            return false;
        }
        const size_t colon = layer_text.find(':');
        if (colon == std::string::npos || std::stoi(layer_text.substr(0, colon)) != expected_layer) {
            return false;
        }
        std::istringstream experts(layer_text.substr(colon + 1));
        std::string expert_text;
        int count = 0;
        while (std::getline(experts, expert_text, ',')) {
            if (count >= TOP_K) {
                return false;
            }
            result[expected_layer][count++] = std::stoi(expert_text);
        }
        if (count != TOP_K) {
            return false;
        }
        ++expected_layer;
    }
    return expected_layer == N_LAYERS;
}

bool load_trace(const std::filesystem::path & path, std::vector<routes> & tokens) {
    std::ifstream input(path);
    REQUIRE(input.good(), "open trace");

    std::string line;
    REQUIRE(static_cast<bool>(std::getline(input, line)), "read trace header");
    REQUIRE(line.rfind("# llama-dsv4-route-trace-v1", 0) == 0, "trace v1 magic");
    REQUIRE(line.find("layers=43") != std::string::npos, "trace declares 43 layers");
    REQUIRE(line.find("top_k=6") != std::string::npos, "trace declares top_k 6");

    bool saw_columns = false;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (!saw_columns) {
            REQUIRE(line == "step\tpos\ttoken\troutes", "trace column header");
            saw_columns = true;
            continue;
        }
        const size_t routes_column = line.find_last_of('\t');
        REQUIRE(routes_column != std::string::npos, "trace token has routes column");
        routes token_routes = {};
        REQUIRE(parse_routes(line.substr(routes_column + 1), token_routes), "parse token routes");
        tokens.push_back(token_routes);
    }
    REQUIRE(saw_columns, "trace has column header");
    REQUIRE(!tokens.empty(), "trace has token rows");
    return true;
}

bool test_atomic_classification_and_duplicate_misses() {
    expert_lru_cache cache(8);
    const std::array<int, TOP_K> repeated = { 7, 7, 7, 7, 7, 7 };
    const auto cold = cache.lookup(3, repeated);
    for (const auto & access : cold) {
        REQUIRE(access.classification == expert_lru_classification::miss, "duplicate cold routes classify as misses");
        REQUIRE(access.slot == 0, "duplicate misses converge on first slot");
    }
    REQUIRE(cache.stats().misses == TOP_K, "every snapshot miss is counted");
    REQUIRE(cache.stats().evictions == 0, "duplicate insertions do not evict");

    const auto warm = cache.lookup(3, repeated);
    for (const auto & access : warm) {
        REQUIRE(access.classification == expert_lru_classification::hit, "duplicate warm routes classify as hits");
        REQUIRE(access.slot == 0, "warm route keeps slot");
    }
    REQUIRE(cache.stats().hits == TOP_K, "every warm access is counted");
    return true;
}

bool test_global_lru_eviction_and_reset() {
    expert_lru_cache cache(6);
    const std::array<int, TOP_K> first = { 0, 1, 2, 3, 4, 5 };
    const auto filled = cache.lookup(0, first);
    for (int i = 0; i < TOP_K; ++i) {
        REQUIRE(filled[i].slot == i, "cold fill assigns ascending abstract slots");
    }

    const std::array<int, TOP_K> second = { 0, 1, 2, 3, 4, 6 };
    const auto update = cache.lookup(0, second);
    REQUIRE(update[5].classification == expert_lru_classification::miss, "new expert misses");
    REQUIRE(update[5].evicted, "full cache evicts");
    REQUIRE((update[5].evicted_key == expert_lru_key{ 0, 5 }), "hits touched first make expert 5 global LRU");
    REQUIRE(update[5].slot == 5, "eviction reuses victim slot");

    const std::array<int, TOP_K> other_layer = { 10, 10, 10, 10, 10, 10 };
    const auto global = cache.lookup(1, other_layer);
    REQUIRE((global[0].evicted_key == expert_lru_key{ 0, 0 }), "LRU is global across layers");

    cache.reset(0);
    const auto disabled = cache.lookup(2, first);
    REQUIRE(cache.capacity() == 0 && cache.size() == 0, "zero-capacity reset disables residency");
    REQUIRE(cache.stats().misses == TOP_K && cache.stats().evictions == 0, "zero cache counts misses without eviction");
    for (const auto & access : disabled) {
        REQUIRE(access.slot == -1, "zero cache has no abstract slot");
    }
    return true;
}

bool test_real_trace_oracle(const std::filesystem::path & trace_root) {
    size_t present_required = 0;
    size_t checked = 0;
    std::string loaded_name;
    std::vector<routes> tokens;

    for (const oracle_case & expected : ORACLE_CASES) {
        const std::filesystem::path path = trace_root / expected.file_name;
        if (!std::filesystem::is_regular_file(path)) {
            if (expected.capacity == 64) {
                std::fprintf(stderr, "SKIP: missing pilot trace %s\n", path.string().c_str());
            }
            continue;
        }
        if (expected.capacity == 64 &&
                (std::string(expected.file_name).rfind("prose_", 0) == 0 ||
                 std::string(expected.file_name).rfind("code_", 0) == 0 ||
                 std::string(expected.file_name).rfind("math_", 0) == 0)) {
            ++present_required;
        }
        if (loaded_name != expected.file_name) {
            tokens.clear();
            REQUIRE(load_trace(path, tokens), "load real pilot trace");
            loaded_name = expected.file_name;
        }
        REQUIRE(tokens.size() == expected.tokens, "token count matches oracle input");

        expert_lru_cache cache(expected.capacity);
        for (const routes & token : tokens) {
            for (int layer = 0; layer < N_LAYERS; ++layer) {
                cache.lookup(layer, token[layer]);
            }
        }
        const expert_lru_stats & actual = cache.stats();
        REQUIRE(actual.accesses == expected.accesses, "access count matches sim_lru.py P1");
        REQUIRE(actual.hits == expected.hits, "hit count matches sim_lru.py P1");
        REQUIRE(actual.misses == expected.misses, "miss count matches sim_lru.py P1");
        REQUIRE(actual.evictions == expected.evictions, "eviction count matches sim_lru.py P1");

        const double hit_rate = actual.accesses == 0 ? 0.0 :
                100.0 * static_cast<double>(actual.hits) / static_cast<double>(actual.accesses);
        std::fprintf(stderr,
                "PASS: %-31s capacity=%4zu tokens=%llu hits=%llu misses=%llu evictions=%llu hit-rate=%.6f%%\n",
                expected.file_name, expected.capacity,
                static_cast<unsigned long long>(expected.tokens),
                static_cast<unsigned long long>(actual.hits),
                static_cast<unsigned long long>(actual.misses),
                static_cast<unsigned long long>(actual.evictions), hit_rate);
        ++checked;
    }

    REQUIRE(present_required == 3, "prose, code, and math pilot traces are required");
    REQUIRE(checked >= 9, "all capacities checked for required traces");
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 3 || std::string(argv[1]) != "--trace-root") {
        std::fprintf(stderr, "Usage: test-expert-lru-cache --trace-root TRACE_DIRECTORY\n");
        return 2;
    }
    if (!test_atomic_classification_and_duplicate_misses() ||
        !test_global_lru_eviction_and_reset() ||
        !test_real_trace_oracle(argv[2])) {
        return 1;
    }
    std::fprintf(stderr, "PASS: DSV4 expert LRU policy matches sim_lru.py P1\n");
    return 0;
}
