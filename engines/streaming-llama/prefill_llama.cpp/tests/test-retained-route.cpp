#include "llama-retained-route.h"

#include <cstdlib>
#include <string>

#undef NDEBUG
#include <cassert>

namespace {

class scoped_sparse_env {
public:
    explicit scoped_sparse_env(const char * value) {
        const char * current = std::getenv("LLAMA_M3_SPARSE");
        if (current != nullptr) {
            original = current;
            had_original = true;
        }
        set(value);
    }

    ~scoped_sparse_env() {
        set(had_original ? original.c_str() : nullptr);
    }

    scoped_sparse_env(const scoped_sparse_env &) = delete;
    scoped_sparse_env & operator=(const scoped_sparse_env &) = delete;

private:
    static void set(const char * value) {
#ifdef _WIN32
        const int rc = _putenv_s("LLAMA_M3_SPARSE", value != nullptr ? value : "");
#else
        const int rc = value != nullptr
            ? setenv("LLAMA_M3_SPARSE", value, 1)
            : unsetenv("LLAMA_M3_SPARSE");
#endif
        assert(rc == 0);
    }

    std::string original;
    bool had_original = false;
};

void test_existing_architectures_unchanged() {
    assert(llama_retained_route_capable_for(LLM_ARCH_DEEPSEEK4, false));
    assert(llama_retained_route_capable_for(LLM_ARCH_GLM_DSA, true));
    assert(!llama_retained_route_capable_for(LLM_ARCH_LLAMA, false));
}

void test_dense_minimax_m3_capable_when_sparse_env_unset() {
    const scoped_sparse_env env(nullptr);
    assert(llama_retained_route_capable_for(LLM_ARCH_MINIMAX_M3, false));
}

void test_dense_minimax_m3_capable_when_sparse_env_empty_or_zero() {
    {
        const scoped_sparse_env env("");
        assert(llama_retained_route_capable_for(LLM_ARCH_MINIMAX_M3, false));
    }
    {
        const scoped_sparse_env env("0");
        assert(llama_retained_route_capable_for(LLM_ARCH_MINIMAX_M3, false));
    }
}

void test_minimax_m3_rejects_sparse_mode() {
    const scoped_sparse_env env("1");
    assert(!llama_retained_route_capable_for(LLM_ARCH_MINIMAX_M3, false));
}

void test_minimax_m3_rejects_mla() {
    const scoped_sparse_env env(nullptr);
    assert(!llama_retained_route_capable_for(LLM_ARCH_MINIMAX_M3, true));
}

void test_minimax_m3_sparse_env_uses_graph_builder_semantics() {
    {
        const scoped_sparse_env env("true");
        assert(!llama_retained_route_capable_for(LLM_ARCH_MINIMAX_M3, false));
    }
    {
        const scoped_sparse_env env("2");
        assert(!llama_retained_route_capable_for(LLM_ARCH_MINIMAX_M3, false));
    }
    {
        const scoped_sparse_env env("0enabled");
        assert(llama_retained_route_capable_for(LLM_ARCH_MINIMAX_M3, false));
    }
}

} // namespace

int main() {
    test_existing_architectures_unchanged();
    test_dense_minimax_m3_capable_when_sparse_env_unset();
    test_dense_minimax_m3_capable_when_sparse_env_empty_or_zero();
    test_minimax_m3_rejects_sparse_mode();
    test_minimax_m3_rejects_mla();
    test_minimax_m3_sparse_env_uses_graph_builder_semantics();
    return 0;
}
