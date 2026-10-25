#pragma once
// Diagnostic scaffolding only; compile alongside the three boundary-* variants.
// Unbuilt owner-review candidate: explicit HC/expert seed injection, no qualification.
#include "ggml.h"
#include "ggml-backend.h"
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace dsv41_boundary {
constexpr uint64_t maximum_capture_bytes = 32ull * 1024 * 1024;
std::string sha256(const void *data, size_t bytes);
// Reuse the same digest for bounded canonical state verification. No capture
// configuration or environment is consulted; each read is at most64KiB.
std::string sha256_stream(uint64_t bytes,
    const std::function<bool(uint64_t,void *,size_t)> & read);
// Resumable SHA-256 (same digest): a running state can be copied, extended
// with more bytes and finalized any number of times. sha256_hex() finalizes a
// copy; the state itself keeps accepting bytes. For a byte string A||B,
// sha256_hex(update(update(begin(),A),B)) == sha256(A||B) exactly.
struct sha256_state {
    uint32_t h[8] = {};
    std::array<unsigned char,64> block{};
    uint64_t total = 0;
    size_t pending = 0;
};
sha256_state sha256_begin();
void sha256_update(sha256_state &state, const void *data, size_t bytes);
std::string sha256_hex(const sha256_state &state);
struct config {
    std::string nonce, source_base_revision, binary_sha256, variant_manifest_sha256;
    std::string residual_sha256, previous_pre_sha256;
    std::string replay_arm, replay_inputs_sha256, weight_proof_sha256, attention_contract_sha256;
    uint64_t capture_bytes = 0;
};
// Strict, bounded owner-file grammar; no HTTP/request path is accepted.
config parse_config(const std::string &text);
struct record {
    std::string stage;
    ggml_type type = GGML_TYPE_F32;
    int64_t ne[4] = {};
    uint64_t bytes = 0;
    bool has_token0_dim94 = false;
    uint32_t token0_dim94_bits = 0;
    std::string sha256;
};
using block_sink = std::function<void(const char *, const void *, size_t)>;

class capture {
public:
    capture(uint32_t layer, uint32_t start, uint32_t count,
            const std::vector<float> &residual, const std::vector<float> &previous_pre);
    ~capture();
    bool active() const;
    void add(ggml_tensor *, const char *stage);
    void observe(ggml_tensor *, const char *name);
    // Creates independent output copies before the graph allocator plans reuse.
    void retain(ggml_context *, ggml_cgraph *);
    uint64_t bytes() const;
    std::string replay_arm() const;
    ggml_tensor *replacement(ggml_context *,ggml_tensor *,const char *);
    void upload_replacements();
    void verify_weight(const char *,const ggml_tensor *);
    void verify_weight_proof_complete();
    void verify_attention_contract(const std::vector<int32_t>&,const std::vector<float>&);
    void write(ggml_backend_t, const ggml_tensor *kv_weight,
               const ggml_tensor *attention_norm, const ggml_tensor *kv_norm);
#ifdef DSV41_BOUNDARY_CAPTURE_TEST
    static std::unique_ptr<capture> for_test(uint32_t rows, uint64_t cap);
    static std::unique_ptr<capture> for_replay_test(const std::string &config_text,const std::string &inputs_text,
        const std::string &proof_text,const std::function<std::string(const std::string &,size_t)> &loader);
    std::vector<record> read_for_test(ggml_backend_t, const block_sink &);
#endif
private:
    struct implementation;
    std::unique_ptr<implementation> p;
    explicit capture(std::unique_ptr<implementation>);
    void load_replay(const std::string &,const std::string &,
        const std::function<std::string(const std::string &,size_t)> &);
};

// Scoped only around this executor's graph construction. It never changes math.
class scope {
public:
    explicit scope(capture *);
    ~scope();
    scope(const scope &) = delete;
    scope &operator=(const scope &) = delete;
private:
    capture *previous;
};
void stage(ggml_tensor *, const char *);
ggml_tensor *replacement(ggml_context *,ggml_tensor *,const char *);
}
