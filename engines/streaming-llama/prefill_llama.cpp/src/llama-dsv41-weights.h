#pragma once

#include "llama-dsv41-layer.h"
#include "llama-dsv41-engram.h"
#include "ggml-cpp.h"
#include <array>
#include <memory>
#include <string>
#include <vector>

// Original native-1017 GGUF -> bounded DirectStorage layer staging. Metadata
// tensors never mmap or allocate their complete file payload. Engram tables
// remain opaque, original row264 bytes on disk and are excluded from stages.
// This provider does not enable the main model inference gate.
class llama_dsv41_weight_provider {
public:
    struct config {
        std::string model_path;
        // Caller must first verify this complete original checkpoint's SHA.
        // Physical file size must equal the verified receipt, not a partial.
        uint64_t verified_file_bytes=0;
        // Identity bound to that completed SHA verification on Windows:
        // volume serial, 128-bit FileIdInfo identifier, Unix last-write ns.
        // A stale same-sized file must be reverified, never trusted by size.
        uint64_t verified_volume=0,verified_mtime_ns=0;
        std::array<uint8_t,16> verified_file_id{};
        std::string verified_sha256;
        ggml_backend_t backend=nullptr;
        // Includes compute buffers plus DS GPU staging and factory host stage.
        uint64_t admitted_device_bytes=0,admitted_host_bytes=0;
        uint32_t buffers=1;
        // Source-pinned candidate profile, independently admitted before allocate.
        // Zero/default retains the measured baseline, not an environment override.
        uint64_t staging_limit_bytes=0;
        bool auto_request_size=false;
        bool async_staging_drain=false;
        bool compute_direct=false;
        // Whole-prompt original row264 cache on its own IoRing. Zero is off.
        uint64_t engram_prefetch_host_bytes=0;
        uint32_t engram_prefetch_tokens=0;
        bool zero_padding_only=false;
        uint32_t ds_submit_threads=0;
        uint32_t ds_gpu_ring_slots=0;
    };
    struct capacity {
        uint64_t maximum_stage_bytes=0,compute_buffer_bytes=0;
        uint64_t device_bytes=0,host_bytes=0;
        uint64_t staging_bytes=0,request_bytes=0;
    };

    llama_dsv41_weight_provider();
    ~llama_dsv41_weight_provider();
    llama_dsv41_weight_provider(const llama_dsv41_weight_provider&)=delete;
    llama_dsv41_weight_provider&operator=(const llama_dsv41_weight_provider&)=delete;
    llama_dsv41_weight_provider(llama_dsv41_weight_provider&&) noexcept;
    llama_dsv41_weight_provider&operator=(llama_dsv41_weight_provider&&) noexcept;
    // Windows+CUDA+DS only. Caller owns backend for the full provider lifetime.
    // inspect retains the read-only physical file lease and metadata only;
    // no DS factory or GPU buffer allocation. Admit combined job budgets from
    // budget() BEFORE allocate(). open() is the standalone inspect+allocate.
    bool inspect(const config &options,std::string &error);
    bool allocate(std::string &error);
    bool open(const config &options,std::string &error);
    // Consume only a fully completed pass, retaining the original model lease,
    // heaps, requests and imported pointers. Start is deferred to the timed run.
    bool restart_prompt(const config &options,std::string &error);
    bool start_request(std::string &error);
    capacity budget() const;
    ggml_backend_t backend() const;
    bool prepare_engram_prompt(const std::vector<int32_t> &tokens,
         const std::function<bool()> &cancelled,std::string &error,llama_dsv41::engram_history initial={});
    bool engram_tile(uint32_t index,llama_dsv41::engram_history &history,
        const int32_t *tokens,const uint8_t *text_mask,size_t count,
        std::vector<float> &output,std::string &error);
    // Advance decoded text history with the existing hash/token-map logic.
    // No table fetch or model computation. Only a completed released pass may
    // accept at most128 already evaluated tokens per synchronization batch.
    bool advance_decoded_engram(llama_dsv41::engram_history &history,
        const int32_t *tokens,size_t count,std::string &error);
    // Strict order: embedding stage, layers0..39, output head. One lease at a
    // time; release synchronizes the actual backend before any buffer reuse.
    bool acquire_embeddings(ggml_tensor *&embedding,std::string &error);
    bool acquire_layer(uint32_t layer,llama_dsv41_layer_graph::weights &weights,std::string &error);
    bool acquire_head(ggml_tensor *&norm,ggml_tensor *&output,std::string &error);
    bool release(std::string &error);
    void invalidate();
private:
    struct implementation;
    std::unique_ptr<implementation>p;
};
