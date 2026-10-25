#pragma once

#include "llama-dsv41-carry.h"
#include "llama-dsv41-weights.h"
#include "llama-dsv41-wire.h"
#include "llama-dsv41-executor.h"

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// A single original-1017 text prefill. This owns the provider, canonical cache,
// executor and admitted RAM or exclusive SSD carry; none can be mutated by a transport caller.
// The externally owned backend must outlive this object and must not be used
// concurrently by another job. This class does not enable an inference preset.
class llama_dsv41_sweep {
public:
    struct config {
        uint32_t context=0;
        uint64_t admitted_cache_bytes=0,admitted_graph_bytes=0;
        // Total job ceilings, including the provider's separate actual budget.
        uint64_t admitted_device_bytes=0,admitted_host_bytes=0;
        // Explicit one-bank CUDA hidden residency; RAM mirror/readback guards stay.
        uint64_t admitted_hidden_bytes=0;
        // Explicit pinned, direct-page CUDA copies with GPU validation. The
        // single RAM bank stays authoritative; ordinary vectors remain default.
        uint64_t admitted_pinned_device_bytes=0,admitted_cast_device_bytes=0;
        uint64_t admitted_index_device_bytes=0;
        uint64_t admitted_prefetch_device_bytes=0;
        bool audit_casts=false,coalesce_dense=false,merge_attention_selected_rows=false,hc_projection_f16_gemm=false,dense_mxfp4=false,batch_seed_ram=false,pair_ffn=false;
        bool reuse_graph=false,reuse_layer_graph=false;
        bool engram_tile_lookahead=false,resident_bf16=false,async_inputs=false,resident_ids=false,resident_metadata=false;
        std::shared_ptr<llama_dsv41_fixed_storage> fixed;
        bool compressed_cache_bf16=false;
        // LLAMADECODE growable context cache: 0 = fixed full-context buffer;
        // otherwise the VMM growth step in tokens (a multiple of 2048).
        uint32_t ctx_cache_grow_step=0;
        bool capture_dspark=false;
        bool stateful_chunks=false;
        // Logical scheduling step within the unchanged admitted maximum tile.
        uint32_t compute_step_tokens=0;
        // Explicit continuation policy:0 rejects unaligned paired appends;
        // 2048 keeps the fixed owner but uses single attention graphs.
        uint32_t unaligned_append_step_tokens=0;
        uint32_t diagnostic_layers=0;
        std::function<void(uint32_t,uint32_t,uint32_t,const void*,uint64_t,const ggml_tensor*)> diagnostic_carry;
        llama_dsv41_carry_store::config carry;
        llama_dsv41_expert_profile::policy expert; // default existing, no extra reserve
    };
    using cancelled=std::function<bool()>;
    // Synchronous transport lease. Publish only after export_native returns
    // true. The reader and its source pointers may not
    // escape this callback. All its worker reads must finish before it returns.
    // Do not call run/export_native recursively from inside the callback.
    using consumer=std::function<bool(llama_dsv41::wire_payload_reader &,std::string &)>;

    llama_dsv41_sweep();
    ~llama_dsv41_sweep();
    llama_dsv41_sweep(const llama_dsv41_sweep&)=delete;
    llama_dsv41_sweep&operator=(const llama_dsv41_sweep&)=delete;

    struct native_base {
        llama_dsv41::wire_frontier frontier;
        std::vector<int32_t> tokens;
        std::vector<float> logits;
        std::string tokens_sha256,logits_sha256;
        std::vector<std::string> owner_sha256;
        llama_dsv41::wire_read_span read;
    };
    // Optional base is an authenticated native mode1 snapshot, streamed into
    // the existing cache before binding the new executor. Caller separately
    // admits its token/logit metadata and holds the same request/owner epoch.
    bool open(const llama_dsv41_weight_provider::config &weights,const config &options,std::string &error,
              llama_dsv41_sweep *completed_one_token_warmup=nullptr,
              llama_dsv41_sweep *completed_prefix=nullptr,
              const native_base *restored_base=nullptr);
    // Immutable token IDs and, only with explicit RAM carry admission, its
    // single bank scale with context. Graph readbacks obey the explicit RAM
    // compute tile bound; physical carry pages and embedding seeds stay32.
    // Failed/cancelled/opened objects cannot start a replacement request.
    bool run(const std::vector<int32_t> &tokens,const cancelled &stop,std::string &error);
    // DS4COMBO producer-warm: on an opened, not yet run sweep, start the
    // Engram row cache of the NEXT request (tokens known up front) while this
    // pass runs. base_tail is the restored base's last <=3 tokens, if any.
    // DS4COMBO fast switch: on a completed fixed owner, release all device
    // memory (ggml switch arena, DS pipeline) and the carry RAM pages while the
    // process keeps its CUDA context, model lease and plans; resume maps fresh
    // zeroed memory at the same addresses. A resumed owner may only be passed
    // as `warmup` to open() together with a restored native base (rebase).
    bool suspend_device(std::string &error);
    bool resume_device(std::string &error);
    bool start_next_engram(const std::vector<int32_t> &tokens,const std::vector<int32_t> &base_tail,std::string &error);
    // LLAMADECODE_ENGRAM_EARLY=1|auto: on a completed stateful owner, start the
    // next append's Engram rows (its own history tail) before that append's
    // open; the append's run adopts them. =0: returns true and does nothing.
    // Auto: only when requests/IOPS fits this pass's layer-0 wall.
    bool start_append_engram(const std::vector<int32_t> &tokens,std::string &error);
    // An early Engram cache is queued on this sweep's provider (cold: after open).
    bool early_engram_queued() const;
    struct run_timing {double prompt_processing_ms=0,export_preparation_ms=0;};
    // Actual sequential clock boundaries, available only after successful run.
    run_timing timing() const;
    // Mode1 by default; explicit DSpark capture exports completed mode3.
    // Serial export pins the entire
    // completed frontier; failure or cancellation permanently revokes export.
    bool export_native(const consumer &send,std::string &error);
    // The completed frontier's owner SHA-256s, equal to the owner digests of
    // its export_native payload. Incremental by default (only rows appended
    // since the last verified digest are read; `expected` mismatch re-hashes
    // in full); LLAMADECODE_OWNER_VERIFY=full re-hashes every owner.
    bool owner_digests(std::vector<std::string> &out,const std::vector<std::string> *expected,std::string &error);
    struct decode_delta {
        llama_dsv41::wire_frontier target;
        std::vector<int32_t> evaluated_tokens;
        std::vector<float> logits;
        std::string base_tokens_sha256,target_tokens_sha256,target_logits_sha256;
        std::vector<std::string> base_owner_sha256,target_owner_sha256;
    };
    // Accept1..128 evaluated Mac tokens together with their canonical state.
    // The transport must bind/authenticate model and session identity. Token,
    // position, logit and every native owner hash are checked here; any failure
    // revokes this sweep. No evaluation or re-prefill is performed by this API.
    bool apply_decode_delta(const decode_delta & update,
        const llama_dsv41::wire_read_span & read,std::string &error);
    struct rewind_anchor {
        llama_dsv41::wire_frontier frontier;
        llama_dsv41::engram_history history;
        std::string tokens_sha256,logits_sha256;
        std::vector<std::string> owner_sha256;
        std::vector<float> logits;
        std::vector<std::vector<unsigned char>> mutable_owners;
    };
    // One bounded RAM anchor: only overwritten rings/tails and logits, never
    // the append-only context-sized compressed/index cache. No model eval.
    bool capture_rewind_anchor(rewind_anchor &,std::string &error);
    bool restore_rewind_anchor(const rewind_anchor &,
        const std::vector<std::string> &current_owner_sha256,std::string &error);
    // Thread-safe, nonblocking request; computation observes this between
    // bounded pages/stages and every native cache readback.
    void cancel();
    bool ready() const;
    // LLAMADECODE B0: the completed frontier's next-token logits (F32, full
    // vocabulary) for local greedy decode. Copies under the owner lock; fails
    // unless the sweep is completed and published.
    bool completed_logits(std::vector<float> &out,std::string &error) const;
    // LLAMADECODE ds4-decode: device addresses of the canonical V4.1 state so
    // ds4-win's own decode can advance it in place, and the host bookkeeping
    // of one token that external decode evaluated (tokens, logits, Engram
    // history; no cache write: the device state was advanced in place).
    struct device_state {
        std::array<void*,40> raw{};std::array<void*,4> compressed{},index{};std::array<uint32_t,4> rows{};
        std::array<void*,3> pool_kv{},pool_score{};
        bool bf16=false;   // compressed/index owners stored as BF16 (--ctx-cache-precision bf16)
    };
    bool device_state_views(device_state &out,std::string &error) const;
    bool engram_tail(std::array<int32_t,3> &tail,std::string &error) const;
    bool commit_external_token(int32_t token,const std::vector<float> &logits,std::string &error);
    // LLAMADECODE multi-row append: count (1..128) evaluated tokens, logits of the last one.
    bool commit_external_tokens(const int32_t *tokens,size_t count,const std::vector<float> &logits,std::string &error);
    // The provider of this completed owner (resident decode weights, expert cache).
    llama_dsv41_weight_provider *provider_for_external_decode();
    // LLAMADECODE: decode-sized fixed plan for an external decoder (no executor runs).
    bool compact_for_external_decode(std::string &error);
    // LLAMADECODE growable context cache: map the owner caches to cover `tokens`
    // positions before an external decoder writes there (no-op when fixed).
    bool ensure_context(uint32_t tokens,const char *phase,bool *grew,std::string &error);
    // Physical bytes of the next growth step (0 when fixed or at full context).
    uint64_t context_next_step_bytes() const;
private:
    struct implementation;
    std::unique_ptr<implementation>p;
};
