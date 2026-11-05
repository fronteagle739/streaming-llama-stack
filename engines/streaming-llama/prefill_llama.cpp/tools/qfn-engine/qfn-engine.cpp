// QFN: Qwen3.8 Flash Next streaming engine. The math follows upstream ggml-org llama.cpp
// src/models/qwen4exp.cpp (a91eab67d) op for op; the differences are where the data lives:
//   * dense weights resident in VRAM, routed experts streamed per layer through the stack's DirectStorage weight
//     fetcher (two banks, one contiguous 675 MiB read per layer), the n-gram (PLE) rows read through the stack's
//     IoRing Engram loader and decoded on the GPU;
//   * prefill is layer-major over the whole prompt in row tiles (each layer's experts are read once per prompt);
//   * QSA attends with a gather-sparse kernel over the selected cells (upstream masks a dense attention).
#include "qfn-engine.h"
#include "qfn-state.h"
#include <io.h>
#include <fcntl.h>
#include <share.h>
#include <sys/stat.h>
#include "llama.h"

#include <filesystem>
#include <fstream>

#include "ggml-cuda.h"
#include "ggml-qfn-nvfp4.h"
#include "ggml-qfn-expert-reduce.h"
#include "ggml-qfn-attention.h"
#include "ggml-qfn-qsa-score.h"
#include "ggml-qfn-service.h"
#include "llama-weight-fetcher.h"
#include "llama-wf-bounded-policy.h"
#include "llama-dsv41-engram-prefetch.h"
#include "llama-dsv41-engram-early-decision.h"

#define NOMINMAX
#include <windows.h>
#include <objbase.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cerrno>
#include <cstdarg>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <map>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace {

// Physical free VRAM of device 0 (the job sees one device); CUDA's free counts imported D3D12 heaps twice.
bool nvml_free_bytes(uint64_t & free_bytes) {
    free_bytes = 0;
    static HMODULE nvml = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!nvml) return false;
    struct memory { unsigned long long total, free, used; };
    using init_t = int (*)();
    using by_uuid_t = int (*)(const char *, void **);
    using info_t = int (*)(void *, memory *);
    static auto init = reinterpret_cast<init_t>(GetProcAddress(nvml, "nvmlInit_v2"));
    static auto by_uuid = reinterpret_cast<by_uuid_t>(GetProcAddress(nvml, "nvmlDeviceGetHandleByUUID"));
    static auto info = reinterpret_cast<info_t>(GetProcAddress(nvml, "nvmlDeviceGetMemoryInfo"));
    static const int ready = init ? init() : -1;
    if (ready != 0 || !by_uuid || !info) return false;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return false;
    const auto * u = reinterpret_cast<const unsigned char *>(prop.uuid.bytes);
    char name[64];
    std::snprintf(name, sizeof(name), "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    void * device = nullptr;
    memory m{};
    if (by_uuid(name, &device) != 0 || info(device, &m) != 0) return false;
    free_bytes = m.free;
    return true;
}

void require(bool ok, const std::string & what) {
    if (!ok) throw std::runtime_error(what);
}

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

struct file_reader {
    HANDLE h = INVALID_HANDLE_VALUE;
    uint64_t size = 0;
    explicit file_reader(const std::wstring & path, bool sequential) {
        h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                        sequential ? FILE_FLAG_SEQUENTIAL_SCAN : FILE_FLAG_RANDOM_ACCESS, nullptr);
        require(h != INVALID_HANDLE_VALUE, "cannot open model file");
        LARGE_INTEGER s{};
        require(GetFileSizeEx(h, &s) != 0, "cannot size model file");
        size = (uint64_t) s.QuadPart;
    }
    ~file_reader() { if (h != INVALID_HANDLE_VALUE) CloseHandle(h); }
    void read(uint64_t off, void * dst, uint64_t n) const {
        uint8_t * d = (uint8_t *) dst;
        while (n) {
            OVERLAPPED ov{};
            ov.Offset = (DWORD) (off & 0xFFFFFFFFu);
            ov.OffsetHigh = (DWORD) (off >> 32);
            const DWORD want = (DWORD) std::min<uint64_t>(n, 1u << 30);
            DWORD got = 0;
            require(ReadFile(h, d, want, &got, &ov) && got == want, "short read from model file");
            d += got; off += got; n -= got;
        }
    }
};

}  // namespace

// ================================================================================================ impl
// a graph input that no node reads is never allocated: skip it instead of writing through a null buffer
static void set_input(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    if (t && t->buffer) ggml_backend_tensor_set(t, data, offset, size);
}

struct qfn_engine::impl {
    std::function<void(uint32_t,uint32_t,uint64_t)> prefill_progress;
    // Extend the existing synchronized diagnostic profiler at request boundaries.
    // Counts are CPU graph nodes, not an inferred CUDA kernel-launch count.
    bool phase_profile = false;
    uint64_t phase_graphs = 0, phase_nodes = 0;
    double phase_embedding_ms = 0, phase_ple_ms = 0;
    void profile_begin() {
        const char * e = std::getenv("LLAMA_CUDA_OP_PROFILE");
        phase_profile = e && e[0] == '1';
        phase_graphs = phase_nodes = 0;
        phase_embedding_ms = phase_ple_ms = 0;
    }
    void profile_end(const char * phase) {
        if (!phase_profile) return;
        log("QFN_PHASE_PROFILE phase=%s graphs=%llu nodes=%llu backend=CUDA scheduler_splits=0 cpu_backend_ops=0 cpu_embedding_ms=%.6f ple_host_ms=%.6f cumulative_gpu_profile=1\n",
            phase, (unsigned long long) phase_graphs, (unsigned long long) phase_nodes,
            phase_embedding_ms, phase_ple_ms);
        ggml_cuda_op_profile_dump_now();
    }
    qfn_options opt;
    std::vector<qfn_image_span> images;
    std::array<int32_t,4> image_position(int64_t position) const {
        int64_t delta=0;
        for(const auto &image:images) {
            if(position<image.position)break;
            if(position<int64_t(image.position)+image.count) {
                const auto row=uint32_t(position-image.position);
                const auto base=int32_t(int64_t(image.position)+delta);
                return {base,base+int32_t(row/image.grid_width),base+int32_t(row%image.grid_width),0};
            }
            delta-=int64_t(image.count)-std::max(image.grid_height,image.grid_width);
        }
        const auto value=int32_t(position+delta);return {value,value,value,0};
    }
    int32_t image_delta() const {
        int64_t delta=0;
        for(const auto &image:images)delta-=int64_t(image.count)-std::max(image.grid_height,image.grid_width);
        return int32_t(delta);
    }
    void image_embedding(int32_t token,int64_t position,float *dst) const {
        for(const auto &image:images)if(position>=image.position&&position<int64_t(image.position)+image.count) {
            std::memcpy(dst,image.values.data()+size_t(position-image.position)*hp->n_embd,size_t(hp->n_embd)*4);return;
        }
        ggml_get_type_traits(tok_embd_type)->to_float(tok_embd+size_t(token)*tok_embd_row,dst,hp->n_embd);
    }
    void image_positions(std::vector<int32_t> &out,int64_t count,int64_t first,int64_t stride=1) const {
        for(int64_t i=0;i<count;i++) {
            const auto value=image_position(first+i*stride);
            for(int axis=0;axis<4;axis++)out[size_t(axis*count+i)]=value[axis];
        }
    }
    qfn_pack pack;
    const qfn_hparams * hp = nullptr;
    ggml_backend_t gpu = nullptr;
    bool nvfp4 = false;
    uint64_t nvfp4_scratch_bytes = 0;
    ggml_context * nvfp4_ctx = nullptr;
    ggml_backend_buffer_t nvfp4_buf = nullptr;
    ggml_tensor * nvfp4_input_scale = nullptr; // persistent F32[3,NE,NL], allocated before cache admission
    ggml_tensor * nvfp4_scratch = nullptr; // engine-owned, unchanged address across prefill/decode/v3 capture/replay
    ggml_context * nvfp4_prefill_ctx = nullptr;
    ggml_backend_buffer_t nvfp4_prefill_buf = nullptr;
    ggml_tensor * nvfp4_prefill_scratch = nullptr;
    uint64_t nvfp4_prefill_bytes = 0;
    uint64_t nvfp4_placeholder_read_bytes = 0;

    // resident weights
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    std::map<std::string, ggml_tensor *> wmap;
    std::vector<qfn_layer_w> layers;
    bool paired_gdn = false;
    bool fused_expert_reduce = true;
    std::vector<ggml_tensor *> paired_gdn_weights; // original adjacent Q6_K rows, no device copy
    ggml_tensor *output = nullptr, *out_hc_norm = nullptr, *out_hc_down = nullptr, *out_hc_up = nullptr;
    // Read-only original token embedding bytes, dequantized per tile. Mapping
    // the complete table keeps every token available without an anonymous
    // whole-table copy beside the early Engram IO arena.
    HANDLE tok_embd_mapping = nullptr;
    void * tok_embd_view = nullptr;
    const uint8_t * tok_embd = nullptr;
    ggml_type tok_embd_type = GGML_TYPE_COUNT;
    size_t tok_embd_row = 0;

    // routed experts: two DirectStorage banks
    wf_context * wf = nullptr;
    uint64_t layer_bytes = 0, bank_bytes = 0;
    void * bank_ptr[2] = {nullptr, nullptr};
    ggml_backend_buffer_t bank_buf[2] = {nullptr, nullptr};
    ggml_context * ectx = nullptr;
    ggml_tensor *gate_exps[2] = {}, *up_exps[2] = {}, *down_exps[2] = {};
    bool wf_running = false;

    // per-sequence state (VRAM)
    ggml_context * sctx = nullptr;
    ggml_backend_buffer_t sbuf = nullptr;
    ggml_context * carry_ctx = nullptr;                     // prefill-only hidden rows: freed at the decode handoff
    ggml_backend_buffer_t carry_buf = nullptr;
    bool carry_in_slab = false;                             // the carry occupies the slab's tail slots during prefill
    uint32_t seed_slots = 0;                                // slots the prefill may seed: those before the carry
    ggml_tensor * carry = nullptr;                          // F32 (or BF16 with QFN_CARRY_BF16=1) [n_embd*hc, n_ctx]
    bool carry_bf16 = false;
    std::vector<ggml_tensor *> kcache, vcache, idx_raw, idx_pool;   // per layer (QSA layers only)
    std::vector<ggml_tensor *> gdn_state, conv_state;               // per layer (GDN layers only)
    ggml_tensor * ple_conv = nullptr;                       // F32 [hist, hc*n_embd]

    // Separate layer-major MoE pass. All staging remains F32 even with BF16 carry.
    ggml_context * pfctx = nullptr;
    ggml_backend_buffer_t pfbuf = nullptr;
    ggml_tensor * pf_x = nullptr, * pf_ids = nullptr, * pf_w = nullptr, * pf_inj = nullptr, * pf_res = nullptr;
    bool moe_pass = false;
    int64_t pf_first = 0;

    // per-request PLE rows (VRAM, IQ4_NL [ple_dim, n_unique])
    ggml_context * pctx = nullptr;
    ggml_backend_buffer_t pbuf = nullptr;
    ggml_tensor * ple_rows = nullptr;
    // The same ownership contract as the DSV4.1 provider's early_engram job:
    // immutable tokens, shared row loader, cancellation, captured failure, joined
    // before adoption or before any pack/backend/fetcher resource is destroyed.
    struct ple_request {
        std::vector<int32_t> tokens;
        int64_t p0 = 0;
        std::vector<uint32_t> idx;
        llama_dsv41_engram_prompt_cache cache;
        std::vector<std::unique_ptr<llama_dsv41_engram_prompt_cache>> shards;
        std::vector<uint32_t> joined_ids;
        std::vector<uint8_t> joined_rows;
        double child_kernel_ms=0,child_user_ms=0;uint64_t child_cycles=0;
        bool child_cpu_available=true,parallel=false,parallel_scan=false;
        std::atomic<bool> cancel{false}, sweep_started{false};
        std::exception_ptr failure;
        std::thread worker;
        bool synchronous = false;
        double started_ms = 0, wall_ms = 0;
        ~ple_request() {
            cancel.store(true);
            if (worker.joinable()) worker.join();
        }
    };
    std::unique_ptr<ple_request> ple_pending;
    bool ple_prompt_overlap = true;
    ggml_context * attention_ctx = nullptr;
    ggml_backend_buffer_t attention_buf = nullptr;
    ggml_tensor * attention_scratch = nullptr;
    uint64_t attention_scratch_bytes = 0;


    ggml_gallocr_t galloc = nullptr;

    // decode expert cache: a VRAM slab of expert slots (gate|up|down rows, the pack's blob layout)
    uint32_t n_slots = 0;
    ggml_context * cctx = nullptr;
    ggml_backend_buffer_t cbuf = nullptr;
    void * slab = nullptr;
    ggml_tensor *slab_gate = nullptr, *slab_up = nullptr, *slab_down = nullptr;
    ggml_tensor * resident = nullptr;                 // F32 [n_expert, n_layer]: 1 when the expert is in the slab
    std::vector<float> resident_host;
    std::vector<float> replay_routing_resident;
    bool resident_dirty = true;
    std::vector<int32_t> slot_of;                     // [n_layer * n_expert] -> slot or -1
    std::vector<int32_t> owner;                       // [n_slots] -> layer * n_expert + expert, or -1
    std::vector<double> lfu;                          // [n_slots] recency-weighted use count
    std::vector<uint64_t> lfu_step;                   // [n_slots] step of the last update
    uint64_t step = 0;
    std::vector<uint32_t> prompt_hist;                // [n_layer * n_expert] prefill routing counts
    cudaEvent_t staging_free[2] = {nullptr, nullptr};
    int staging_next = 0;
    // decode state carried between the two halves of a layer (router -> experts)
    ggml_context * dctx = nullptr;
    ggml_backend_buffer_t dbuf = nullptr;
    ggml_tensor *dec_res = nullptr, *dec_x = nullptr, *dec_inj = nullptr, *dec_w = nullptr, *dec_sel = nullptr;
    ggml_tensor * dec_hout = nullptr;                 // F32 [n_embd*hc, DEC_NMAX]: final residual of each decoded token
    static constexpr int DEC_NMAX = 4;                // tokens per decode step (1 + the MTP drafts verified at once)
    // multi-token decode rollback: per recurrent layer, the states after each token but the last (allocated with MTP),
    // and the conv / n-gram conv inputs of the step
    ggml_context * snap_ctx = nullptr;
    ggml_backend_buffer_t snap_buf = nullptr;
    std::vector<ggml_tensor *> gdn_snap, conv_hist;
    ggml_tensor * ple_hist = nullptr;
    int last_step_n = 0;                              // tokens of the last decode step (what rollback() may undo)

    // MTP drafter: one QSA block (layer index n_layer in layers / kcache / idx caches), its 512 experts resident in
    // VRAM (relaid from Basalt's plane layout to gate|up|down blobs), the head restricted to the draft vocabulary
    bool mtp_ok = false;
    bool handover_draft_pending = false;
    ggml_context *paired_ctx = nullptr;
    ggml_backend_buffer_t paired_buffer = nullptr;
    std::array<ggml_tensor *,4> paired_mtp{};
    uint32_t paired_rows = 0;
    struct paired_anchor_state {
        qfn_native_plan plan{};std::vector<uint8_t> owners;
        std::vector<int32_t> tokens;std::vector<float> logits;std::vector<qfn_image_span> images;
    } paired_anchor;
    int mtp_drafts = 3;
    ggml_context * mctx = nullptr;
    ggml_backend_buffer_t mbuf = nullptr, mexp_buf = nullptr;
    ggml_tensor *mtp_fc_e = nullptr, *mtp_fc_h = nullptr, *mtp_enorm = nullptr, *mtp_hnorm = nullptr;
    ggml_tensor *mtp_head_norm = nullptr, *mtp_head_down = nullptr, *mtp_head_up = nullptr, *mtp_dhead = nullptr;
    ggml_tensor *mtp_gate = nullptr, *mtp_up = nullptr, *mtp_down = nullptr;
    ggml_tensor *mtp_hin = nullptr, *mtp_hout = nullptr;   // F32 [n_embd*hc, DEC_NMAX+1]: draft-layer h in / out
    std::vector<int32_t> draft_ids;                      // draft index -> token id
    std::vector<float> h_last;                           // the target's final residual of the last sequence token
    uint64_t mtp_rounds = 0, mtp_drafted = 0, mtp_accepted = 0;

    // Speculative execution can warm physical slots, but only canonical kept rows
    // commit routing/LFU/step state. A fixed-point repair retains the full batch graph.
    struct cache_policy {
        std::vector<int32_t> slot_of, owner, ram_of, ram_owner;
        std::vector<double> lfu;
        std::vector<uint64_t> lfu_step, ram_age;
        std::vector<float> resident, routing;
        uint64_t step = 0, ram_clock = 0;
        bool dirty = false;
    };
    bool verify_active = false;
    ggml_context * verify_ctx = nullptr;
    ggml_backend_buffer_t verify_buf = nullptr;
    ggml_tensor * verify_resident = nullptr; // F32 [NE,DEC_NMAX,NL], stable graph input
    void * verify_state = nullptr;
    size_t verify_state_bytes = 0;
    std::vector<ggml_tensor *> verify_state_tensors;
    std::vector<int32_t> verify_selected;
    std::vector<cache_policy> verify_prefix;
    bool verify_checkpoint_ready = false;

    // decode v3 (static per-step graphs + device slot map + doorbell)
    struct v3graph {
        int n = 0, nbp = 1;
        float bias = 0.0f;
        bool snap = false, logical_policy = false;
        ggml_context * ctx = nullptr;
        ggml_cgraph * gf = nullptr;
        ggml_gallocr_t ga = nullptr;
        cudaGraph_t future_parent = nullptr;
        cudaGraphExec_t future_parent_exec = nullptr;
        void * future_model = nullptr;
        uint64_t future_capture = 0;
        std::vector<cudaGraphNode_t> future_waits;
        ggml_tensor *emb = nullptr, *pos = nullptr, *pos4 = nullptr, *kv_idx = nullptr, *mem_idx = nullptr,
                    *pool_idx = nullptr, *pool_pos4 = nullptr, *seq = nullptr, *logits = nullptr;
    };
    std::vector<v3graph *> v3g;
    bool v3_on = false, v3_replay = false;
    bool v3_future = false;
    ggml_qfn_service_record * service_h = nullptr;
    ggml_qfn_service_record * service_d = nullptr;
    ggml_qfn_service_control * service_control_h = nullptr;
    ggml_qfn_service_control * service_control_d = nullptr;
    cudaExternalSemaphore_t service_semaphore = nullptr; // provider-owned
    cudaGraph_t service_graph = nullptr;
    // Shared only by serialized model graphs, admitted before cache sizing and
    // retained until every parent retires. Keep12MiB as the qualified control;
    // CUDA13.2 recommends32MiB on SM12x, subject to isolated qualification.
    size_t service_blas_bytes = 12ull << 20;
    void * service_blas = nullptr;
    cudaGraphExec_t service_exec = nullptr;
    std::vector<cudaGraphNode_t> service_waits;
    uint64_t service_first = 0, service_last = 0;
    bool service_epoch = false, service_live = false, service_unproven = false;
    void * ram_device = nullptr;
    void * replay_state = nullptr;
    int32_t * replay_miss_latch = nullptr;
    size_t replay_state_bytes = 0;
    std::vector<ggml_tensor *> replay_tensors;

    void alloc_replay_state() {
        if (replay_state) return;
        for (auto * t : gdn_state) if (t) replay_tensors.push_back(t);
        for (auto * t : conv_state) if (t) replay_tensors.push_back(t);
        if (ple_conv) replay_tensors.push_back(ple_conv);
        for (auto * t : replay_tensors) replay_state_bytes += ggml_nbytes(t);
        require(replay_state_bytes > 0 && cudaMalloc(&replay_state, replay_state_bytes) == cudaSuccess,
                "cannot allocate decode replay pre-step recurrent state");
        log("[qfn] decode replay replay state: %.1f MiB, %zu tensors\n", replay_state_bytes / 1048576.0, replay_tensors.size());
    }

    void copy_replay_state(bool restore) {
        size_t off = 0;
        for (auto * t : replay_tensors) {
            void * saved = (char *) replay_state + off;
            require(cudaMemcpyAsync(restore ? t->data : saved, restore ? saved : t->data, ggml_nbytes(t),
                                    cudaMemcpyDeviceToDevice, stream()) == cudaSuccess, "replay state copy failed");
            off += ggml_nbytes(t);
        }
    }
    ggml_context * v3ctx = nullptr;
    ggml_backend_buffer_t v3buf = nullptr;
    ggml_tensor * slot_tab = nullptr;          // I32 [n_expert, n_layer]: device mirror of slot_of
    ggml_tensor * sel_log = nullptr;           // I32 [K*DEC_NMAX, n_layer]: the step's routing (host LFU after the step)
    bool slot_tab_dirty = true;
    ggml_qfn_bell * bell_h = nullptr;
    void * bell_d = nullptr;
    cudaStream_t side = nullptr;
    cudaEvent_t bell_completion = nullptr;
    bool doorbell_poisoned = false;
    void require_live_context() const {
        require(!doorbell_poisoned, "QFN_DOORBELL_RESET_REQUIRED: failed mode1 state is unavailable; fresh prefill is required before append/feed/decode/sampling/state queries");
    }
    std::thread bell_thread;
    std::atomic<bool> bell_stop{false};
    std::atomic<uint64_t> bell_served{0}, bell_miss{0}, bell_bytes{0};
    std::atomic<double> bell_ms{0.0};
    std::string bell_err;
    std::mutex bell_err_mutex;
    uint64_t bell_warm_bytes = 0;
    double bell_warm_ms = 0;
    std::atomic<int> bell_start_ready{0};
    struct miss_placement_parts {
        double bank_wait_ms = 0, disk_wall_ms = 0, payload_enqueue_ms = 0;
        double ds_ms[4] = {0,0,0,0}; // submit, fence, verify, CUDA external-fence handoff
        uint64_t requests = 0;
    };
    int32_t v3_seq = 0;
    std::vector<int32_t> seq;                         // tokens of the sequence so far (PLE history)
    ggml_tensor * last_selected = nullptr;            // prefill: the tile's routing (read back for the histogram)
    std::vector<int32_t> eog;
    std::mt19937_64 rng{1};
    bool sampling_started = false;
    uint64_t sampling_seed = 1;

    ~impl() {
        cancel_prefill();
        v3_shutdown(); // an in-flight timed-out service still owns RAM and staging metadata
        release_reference_weights();
        if(paired_buffer)ggml_backend_buffer_free(paired_buffer);
        if(paired_ctx)ggml_free(paired_ctx);
        destroy_future_service();
        if (pfbuf) ggml_backend_buffer_free(pfbuf);
        if (pfctx) ggml_free(pfctx);
        if (ram) cudaFreeHost(ram);
        if (bell_completion) cudaEventDestroy(bell_completion);
        if (side) cudaStreamDestroy(side);
        if (bell_h) cudaFreeHost((void *) bell_h);
        for (auto * G : v3g) { destroy_future_parent(*G); if (G->ga) ggml_gallocr_free(G->ga); if (G->ctx) ggml_free(G->ctx); delete G; }
        if (replay_state) cudaFree(replay_state);
        if (verify_state) cudaFree(verify_state);
        if (verify_buf) ggml_backend_buffer_free(verify_buf);
        if (verify_ctx) ggml_free(verify_ctx);
        if (replay_miss_latch) cudaFree(replay_miss_latch);
        if (v3buf) ggml_backend_buffer_free(v3buf);
        if (v3ctx) ggml_free(v3ctx);
        for (auto & e : staging_free) if (e) cudaEventDestroy(e);
        if (dbuf) ggml_backend_buffer_free(dbuf);
        if (dctx) ggml_free(dctx);
        if (cbuf) ggml_backend_buffer_free(cbuf);
        if (cctx) ggml_free(cctx);
        if (wf) { wf_stop(wf); wf_quiesce(wf); wf_destroy(wf); wf = nullptr; }
        for (auto * b : bank_buf) if (b) ggml_backend_buffer_free(b);
        if (nvfp4_buf) ggml_backend_buffer_free(nvfp4_buf);
        if (nvfp4_ctx) ggml_free(nvfp4_ctx);
        if (nvfp4_prefill_buf) ggml_backend_buffer_free(nvfp4_prefill_buf);
        if (nvfp4_prefill_ctx) ggml_free(nvfp4_prefill_ctx);
        if (galloc) ggml_gallocr_free(galloc);
        if (attention_buf) ggml_backend_buffer_free(attention_buf);
        if (attention_ctx) ggml_free(attention_ctx);
        if (pbuf) ggml_backend_buffer_free(pbuf);
        if (pctx) ggml_free(pctx);
        free_carry();
        if (sbuf) ggml_backend_buffer_free(sbuf);
        if (sctx) ggml_free(sctx);
        if (ectx) ggml_free(ectx);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
        if (gpu) ggml_backend_free(gpu);
        if (tok_embd_view) UnmapViewOfFile(tok_embd_view);
        if (tok_embd_mapping) CloseHandle(tok_embd_mapping);
    }

    void log(const char * fmt, ...) const {
        if (!opt.verbose) return;
        va_list ap;
        va_start(ap, fmt);
        std::vfprintf(stderr, fmt, ap);
        va_end(ap);
        std::fflush(stderr);
    }

    ggml_tensor * W(const std::string & name, bool required = true) {
        auto it = wmap.find(name);
        if (it == wmap.end()) {
            require(!required, "missing weight " + name);
            return nullptr;
        }
        return it->second;
    }

    // ------------------------------------------------------------------------------------------ load
    void load_weights() {
        const double t0 = now_ms();
        std::vector<const qfn_tensor *> dense;
        for (const auto & kv : pack.tensors) {
            const auto & t = kv.second;
            if (t.name == "per_layer_token_embd.weight" || t.name == "token_embd.weight") continue;
            dense.push_back(&t);
        }
        std::sort(dense.begin(), dense.end(), [](const qfn_tensor * a, const qfn_tensor * b) { return a->offset < b->offset; });
        ggml_init_params ip = { (dense.size() + (paired_gdn ? hp->n_layer : 0) + 16) * ggml_tensor_overhead(), nullptr, true };
        wctx = ggml_init(ip);
        for (const auto * t : dense) {
            ggml_tensor * g = ggml_new_tensor(wctx, t->type, t->n_dims, t->ne);
            ggml_set_name(g, t->name.c_str());
            wmap[t->name] = g;
        }
        wbuf = ggml_backend_alloc_ctx_tensors(wctx, gpu);
        require(wbuf != nullptr, "cannot allocate the resident weights in VRAM");
        ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        // read in large sequential spans
        file_reader f(pack.wpath, true);
        std::vector<uint8_t> chunk;
        // Cold Engram IO now overlaps this upload. Bound its staging even for
        // an individual large tensor; every byte is uploaded at its original
        // offset, without a second whole-tensor host copy.
        constexpr uint64_t chunk_limit = 8ull << 20;
        uint64_t total = 0;
        size_t i = 0;
        while (i < dense.size()) {
            const uint64_t start = dense[i]->offset;
            size_t j = i;
            uint64_t end = start;
            while (j < dense.size() && dense[j]->offset + dense[j]->bytes - start <= chunk_limit) {
                end = std::max(end, dense[j]->offset + dense[j]->bytes);
                ++j;
            }
            if (j == i) {
                const auto * t = dense[i];
                for (uint64_t off = 0; off < t->bytes; off += chunk_limit) {
                    const size_t len = size_t(std::min(chunk_limit, t->bytes - off));
                    chunk.resize(len);
                    f.read(t->offset + off, chunk.data(), len);
                    ggml_backend_tensor_set(wmap[t->name], chunk.data(), off, len);
                }
                total += t->bytes;
                ++i;
                continue;
            }
            chunk.resize(end - start);
            f.read(start, chunk.data(), end - start);
            for (size_t k = i; k < j; ++k) {
                ggml_backend_tensor_set(wmap[dense[k]->name], chunk.data() + (dense[k]->offset - start), 0, dense[k]->bytes);
                total += dense[k]->bytes;
            }
            if ((total >> 30) != ((total - (end - start)) >> 30)) log("[qfn] resident weights: %.2f GiB read\n", total / 1073741824.0);
            i = j;
        }
        // The same original quantized rows remain CPU inputs for prefill,
        // decode, MTP and v3. File-backed pages use the model volume, never
        // page-file backing; no token, row or precision is omitted.
        const qfn_tensor & te = pack.get("token_embd.weight");
        SYSTEM_INFO system_info{};
        GetSystemInfo(&system_info);
        const uint64_t base = te.offset / system_info.dwAllocationGranularity * system_info.dwAllocationGranularity;
        tok_embd_mapping = CreateFileMappingW(f.h, nullptr, PAGE_READONLY, 0, 0, nullptr);
        require(tok_embd_mapping != nullptr, "cannot map the complete original token embedding table");
        tok_embd_view = MapViewOfFile(tok_embd_mapping, FILE_MAP_READ, DWORD(base >> 32), DWORD(base), SIZE_T(te.offset - base + te.bytes));
        require(tok_embd_view != nullptr, "cannot view the complete original token embedding table");
        tok_embd = static_cast<const uint8_t *>(tok_embd_view) + (te.offset - base);
        tok_embd_type = te.type;
        tok_embd_row = ggml_row_size(te.type, te.ne[0]);
        log("[qfn] resident weights: %zu tensors, %.2f GiB in VRAM (+%.2f GiB read-only mapped token embedding), %.1f s\n",
            dense.size(), total / 1073741824.0, te.bytes / 1073741824.0, (now_ms() - t0) / 1000.0);

        layers.resize(hp->n_layer);
        for (uint32_t il = 0; il < hp->n_layer; ++il) {
            auto & L = layers[il];
            const std::string b = "blk." + std::to_string(il) + ".";
            L.hc_attn_norm   = W(b + "hc_attn_norm.weight");
            L.hc_attn_down   = W(b + "hc_attn_down.weight");
            L.hc_attn_up     = W(b + "hc_attn_up.weight");
            L.hc_attn_inject = W(b + "hc_attn_inject.weight");
            L.hc_ffn_norm    = W(b + "hc_ffn_norm.weight");
            L.hc_ffn_down    = W(b + "hc_ffn_down.weight");
            L.hc_ffn_up      = W(b + "hc_ffn_up.weight");
            L.hc_ffn_inject  = W(b + "hc_ffn_inject.weight");
            if (hp->is_recurrent(il)) {
                L.attn_qkv   = W(b + "attn_qkv.weight");
                L.attn_gate  = W(b + "attn_gate.weight");
                L.ssm_a      = W(b + "ssm_a");
                L.ssm_alpha  = W(b + "ssm_alpha.weight");
                L.ssm_beta   = W(b + "ssm_beta.weight");
                L.ssm_conv1d = W(b + "ssm_conv1d.weight");
                L.ssm_dt     = W(b + "ssm_dt.bias");
                L.ssm_norm   = W(b + "ssm_norm.weight");
                L.ssm_out    = W(b + "ssm_out.weight");
            } else {
                L.wq = W(b + "attn_q.weight");
                L.wk = W(b + "attn_k.weight");
                L.wv = W(b + "attn_v.weight");
                L.wo = W(b + "attn_output.weight");
                L.q_norm = W(b + "attn_q_norm.weight");
                L.k_norm = W(b + "attn_k_norm.weight");
                L.idx_k_proj = W(b + "indexer.k_proj.weight");
                L.idx_q_proj = W(b + "indexer.q_proj.weight");
                L.idx_k_norm = W(b + "indexer.k_norm.weight");
                L.idx_q_norm = W(b + "indexer.q_norm.weight");
            }
            if (hp->is_ple(il)) {
                L.ple_key        = W(b + "ple_key.weight");
                L.ple_value      = W(b + "ple_value.weight");
                L.ple_norm_key   = W(b + "ple_norm_key.weight");
                L.ple_norm_query = W(b + "ple_norm_query.weight");
                L.ple_norm_conv  = W(b + "ple_norm_conv.weight");
                L.ple_conv1d     = W(b + "ple_conv1d.weight");
            }
            L.ffn_gate_inp       = W(b + "ffn_gate_inp.weight");
            L.ffn_gate_inp_shexp = W(b + "ffn_gate_inp_shexp.weight");
            L.ffn_gate_shexp     = W(b + "ffn_gate_shexp.weight");
            L.ffn_up_shexp       = W(b + "ffn_up_shexp.weight");
            L.ffn_down_shexp     = W(b + "ffn_down_shexp.weight");
        }
        output      = W("output.weight");
        out_hc_norm = W("output_hc_norm.weight");
        out_hc_down = W("output_hc_down.weight");
        out_hc_up   = W("output_hc_up.weight");
        if (paired_gdn) prepare_paired_gdn();
    }

    void prepare_paired_gdn() {
        paired_gdn_weights.assign(hp->n_layer, nullptr);
        const uintptr_t base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(wbuf));
        const size_t capacity = ggml_backend_buffer_get_size(wbuf);
        unsigned pairs = 0;
        for (uint32_t il = 0; il < hp->n_layer; ++il) {
            if (!hp->is_recurrent(il)) continue;
            const auto * qkv = layers[il].attn_qkv;
            const auto * gate = layers[il].attn_gate;
            const std::string label = "QFN_PAIRED_GDN layer " + std::to_string(il) + ": ";
            const int64_t qkv_rows = int64_t(hp->ssm_state) * (2 * int64_t(hp->ssm_group) + hp->ssm_dt_rank);
            const int64_t gate_rows = int64_t(hp->ssm_state) * hp->ssm_dt_rank;
            require(qkv && gate && qkv->type == GGML_TYPE_Q6_K && gate->type == GGML_TYPE_Q6_K &&
                    qkv->ne[0] == hp->n_embd && gate->ne[0] == qkv->ne[0] &&
                    qkv->ne[1] == qkv_rows && gate->ne[1] == gate_rows &&
                    qkv->ne[2] == 1 && qkv->ne[3] == 1 && gate->ne[2] == 1 && gate->ne[3] == 1 &&
                    ggml_is_contiguous(qkv) && ggml_is_contiguous(gate),
                    label + "requires original contiguous Q6_K qkv/gate matrices; refusing fallback");
            const uintptr_t first = reinterpret_cast<uintptr_t>(qkv->data);
            const uintptr_t second = reinterpret_cast<uintptr_t>(gate->data);
            const size_t qkv_bytes = ggml_nbytes(qkv), gate_bytes = ggml_nbytes(gate);
            require(qkv->buffer == wbuf && gate->buffer == wbuf && first >= base && second >= first &&
                    second - first == qkv_bytes && first - base <= capacity &&
                    qkv_bytes <= capacity - (first - base) &&
                    gate_bytes <= capacity - (first - base) - qkv_bytes,
                    label + "original GPU rows are not adjacent within the resident buffer; refusing fallback");
            auto * whole = ggml_new_tensor_2d(wctx, GGML_TYPE_Q6_K, qkv->ne[0], qkv_rows + gate_rows);
            ggml_format_name(whole, "blk.%u.attn_qkv_gate_paired.weight", il);
            // Binding a quantized tensor can clear backend padding. An alias
            // must require none, so no neighboring resident bytes are touched.
            require(ggml_backend_buffer_get_alloc_size(wbuf, whole) == ggml_nbytes(whole),
                    label + "combined matrix requires backend padding; refusing an alias that could write neighboring weights");
            require(ggml_nbytes(whole) == qkv_bytes + gate_bytes &&
                    ggml_backend_tensor_alloc(wbuf, whole, qkv->data) == GGML_STATUS_SUCCESS,
                    label + "cannot bind the original combined weight span");
            paired_gdn_weights[il] = whole;
            ++pairs;
        }
        require(pairs != 0, "QFN_PAIRED_GDN requires at least one recurrent layer; refusing no-op");
        log("[qfn] paired GDN: %u original Q6_K qkv/gate GPU spans admitted; one product per pair, no weight copies; numerical qualification required\n", pairs);
    }

    // ------------------------------------------------------------------------------------------ experts
    void admit_qsa_indexer() {
        if (opt.qsa_indexer == "auto") {
            cudaDeviceProp device{};
            require(cudaGetDeviceProperties(&device, 0) == cudaSuccess, "QFN_QSA_AUTO: cannot query selected CUDA device");
            const bool matched = device.major == 12 && device.minor == 0 && hp->idx_dim == 128 && hp->idx_heads == 4;
            opt.qsa_indexer = matched ? "basalt-tc-refine-prefill" : "scalar";
            log("[qfn] QSA indexer auto resolves %s: device SM%d%d, H%u/D%u; matched SM120 H4/D128 uses refined TC for >=64 query rows, scalar otherwise\n",
                opt.qsa_indexer.c_str(), device.major, device.minor, hp->idx_heads, hp->idx_dim);
        }
        require(opt.qsa_indexer == "scalar" || opt.qsa_indexer == "basalt-tc-refine-prefill" || opt.qsa_indexer == "basalt-tc-refine" || opt.qsa_indexer == "basalt-tc" || opt.qsa_indexer == "basalt-tc-prefill" || opt.qsa_indexer == "dsv-fp4",
                "invalid QSA indexer policy");
        if (opt.qsa_indexer == "scalar") return;
        require(hp->idx_dim == 128 && hp->idx_heads == 4, "QFN_QSA_GEOMETRY: basalt-tc requires D128/H4; refusing fallback");
        require(opt.tile > 0 && opt.tile <= 1048560 && opt.n_ctx > 0 && hp->kpool > 0 &&
                uint64_t(opt.n_ctx) / hp->kpool <= INT32_MAX, "QFN_QSA_SHAPE: context/tile exceed score C ABI");
        if (opt.qsa_indexer == "dsv-fp4") return;
        char error[512] = {};
        require(ggml_backend_cuda_qfn_qsa_score_available(error, sizeof error), error);
    }
    void prepare_qsa_indexer() {
        if (opt.qsa_indexer == "scalar") return;
        cudaDeviceProp device{};
        require(cudaGetDeviceProperties(&device, 0) == cudaSuccess && device.major == 12 && device.minor == 0,
                "QFN_QSA_PLATFORM: basalt-tc requires SM120; refusing fallback");
        if (opt.qsa_indexer == "dsv-fp4") {
            log("[qfn] QSA indexer dsv-fp4: E2M1/E8M0 per32 Q/K, SM120 block-scaled tensor cores, H4/D128, full pools, no fallback; tolerance candidate\n");
            return;
        }
        require(ggml_backend_cuda_qfn_qsa_score_prepare(nullptr) == cudaSuccess, "QFN_QSA_PREPARE: independent score initialization failed");
        if(opt.qsa_indexer=="basalt-tc-refine" || opt.qsa_indexer=="basalt-tc-refine-prefill")log("[qfn] TC causal top-k scalar boundary refinement engaged: error envelope 1e-4, all pools retained\n");
        log("[qfn] QSA indexer %s: full scores, unchanged split/selector, TC minimum query rows %d; "
            "prefill policies retain scalar below64; every explicit arithmetic policy remains available\n",
            opt.qsa_indexer.c_str(), (opt.qsa_indexer == "basalt-tc-prefill" || opt.qsa_indexer == "basalt-tc-refine-prefill") ? 64 : 1);
    }
    void admit_nvfp4() {
        const char * shared_gu_env = std::getenv("QFN_NATIVE_NVFP4_SHARE_GU_INPUT");
        require(!shared_gu_env || !std::strcmp(shared_gu_env, "0") || !std::strcmp(shared_gu_env, "1"),
                "QFN_NATIVE_NVFP4_SHARE_GU_INPUT must be 0 or 1");
        const bool shared_gu = shared_gu_env && !std::strcmp(shared_gu_env, "1");
        const char * residual_env = std::getenv("QFN_NATIVE_NVFP4_DECODE_RESIDUAL");
        require(!residual_env || !std::strcmp(residual_env, "0") || !std::strcmp(residual_env, "1"),
                "QFN_NATIVE_NVFP4_DECODE_RESIDUAL must be 0 or 1");
        const bool residual = residual_env && !std::strcmp(residual_env, "1");
        const char * vector_env = std::getenv("QFN_NATIVE_NVFP4_DECODE_VECTOR");
        require(!vector_env || !std::strcmp(vector_env, "0") || !std::strcmp(vector_env, "1") || !std::strcmp(vector_env, "2") || !std::strcmp(vector_env, "3") || !std::strcmp(vector_env, "4"),
                "QFN_NATIVE_NVFP4_DECODE_VECTOR must be 0 through 4");
        const bool vector_decode = vector_env && std::strcmp(vector_env, "0");
        for (const auto & e : pack.experts) nvfp4 = nvfp4 || e.is_nvfp4();
        require(!shared_gu || (nvfp4 && pack.native_nvfp4_packed),
                "QFN_NATIVE_NVFP4_SHARE_GU_INPUT=1 requires native-packed NVFP4; refusing ignored input-sharing option");
        require(!residual || (nvfp4 && pack.native_nvfp4_packed && opt.nvfp4_decode_mode == 6 && opt.nvfp4_decode_mode_explicit),
                "QFN_NATIVE_NVFP4_DECODE_RESIDUAL1 requires native-packed NVFP4 and explicit decode mode6; refusing ignored arithmetic option");
        require(!vector_decode || (nvfp4 && pack.native_nvfp4_packed && opt.nvfp4_decode_mode == 6 && opt.nvfp4_decode_mode_explicit),
                "QFN_NATIVE_NVFP4_DECODE_VECTOR1 requires native-packed NVFP4 and explicit decode mode6; refusing ignored arithmetic option");
        require(!vector_decode || !residual,
                "QFN_NATIVE_NVFP4_DECODE_VECTOR1 conflicts with DECODE_RESIDUAL1; select one activation arithmetic");
        if (vector_decode) log("[qfn] packed FP4 vector policy=%s: %s activations, original UE4M3 per16 weight scales/tails; persistent workspace\n",vector_env,std::strcmp(vector_env,"1")?"F32":"dynamic Q8_1");
        if (!nvfp4) {
            require(!opt.reference_prefill_max_tokens,"Reference small-prefix mode requires native-packed NVFP4 experts");
            require(!opt.nvfp4_mode_explicit, "--nvfp4-mode requires NVFP4 main experts; refusing to ignore it");
            require(!opt.nvfp4_decode_mode_explicit, "--nvfp4-decode-mode requires NVFP4 main experts; refusing to ignore it");
            return;
        }
        // Lossless packed prefill is qualified against the calibrated raw reference.
        // Decode/indexer candidates failed the owner full-model gate and stay explicit.
        if (pack.native_nvfp4_packed && !opt.nvfp4_mode_explicit) opt.nvfp4_mode = 6;
        require((opt.nvfp4_mode >= 0 && opt.nvfp4_mode <= 4) || opt.nvfp4_mode==6, "invalid NVFP4 prefill arithmetic mode");
        require(opt.nvfp4_mode!=6 || pack.native_nvfp4_packed,
                "QFN_NATIVE_NVFP4_LAYOUT: dsv-native requires the lossless native-packed v7 model");
        require(!shared_gu || opt.nvfp4_mode == 6,
                "QFN_NATIVE_NVFP4_SHARE_GU_INPUT=1 requires prefill mode6; original modes remain available with flag0");
        require(opt.nvfp4_decode_mode==2 || opt.nvfp4_decode_mode==5 || opt.nvfp4_decode_mode==6,"invalid NVFP4 decode arithmetic mode");
        require(opt.nvfp4_decode_mode!=6 || pack.native_nvfp4_packed,
                "QFN_NATIVE_NVFP4_LAYOUT: dsv-native decode requires the lossless native-packed v7 model");
        require(hp->n_embd == 2560 && hp->n_ff_exp == 640 && hp->n_expert == 512 && hp->n_expert_used == 10,
                "QFN_NVFP4_FORMAT: Basalt bridge requires E2560 FF640 experts512 top10");
        for (const auto & e : pack.experts)
            require(e.gu_type_id == 40 && e.d_type_id == 40 && e.blob == 2764816 && e.gate_bytes == 921600 &&
                    e.down_bytes == 921600 && e.tail_offset == 2764800 && e.tail_bytes == 16,
                    "QFN_NVFP4_FORMAT: all main expert layers must carry exact NVFP4 matrices and 16-byte tails");
        require(!pack.mtp_expert_blob || (pack.mtp_expert_layout.gu_type == GGML_TYPE_Q2_0 &&
                pack.mtp_expert_layout.d_type == GGML_TYPE_Q2_0 && !pack.mtp_expert_layout.tail_bytes),
                "QFN_NVFP4_MTP_FORMAT: copied drafter must retain its independently described Q2_0 planes");
        if (shared_gu) {
            const auto & scales = pack.expert_input_scale;
            require(scales.size() == size_t(hp->n_layer) * hp->n_expert * 3,
                    "QFN_NATIVE_NVFP4_SHARE_GU_INPUT=1 requires every exported role calibration");
            for (size_t i = 0; i < scales.size(); i += 3)
                require(std::memcmp(&scales[i], &scales[i + 1], sizeof(float)) == 0,
                        "QFN_NATIVE_NVFP4_SHARE_GU_INPUT=1 requires bit-identical gate/up globals at layer " +
                        std::to_string(i / (hp->n_expert * 3)) + " expert " + std::to_string(i / 3 % hp->n_expert));
            log("[qfn] shared native FP4 gate/up input admitted: %zu bit-identical calibration pairs; independent down globals; long prefill mode6 only\n",
                scales.size() / 3);
        }
        require(opt.reference_prefill_max_tokens<=1024,"Reference prefill token bound is invalid");
        require(!opt.reference_prefill_max_tokens||(pack.native_nvfp4_packed&&opt.nvfp4_mode==6&&vector_decode),
                "Reference small-prefix mode requires packed mode6 and explicit vector1; no arithmetic option is ignored");
        char error[1024] = {};
        require(ggml_backend_cuda_qfn_nvfp4_available(error, sizeof(error)) != 0, error);
        if (pack.native_nvfp4_packed)
            require(ggml_backend_cuda_qfn_native_nvfp4_available(error,sizeof(error)) != 0,error);
        if (opt.nvfp4_mode == 4 || opt.nvfp4_mode == 6 || opt.nvfp4_decode_mode == 5 || opt.nvfp4_decode_mode == 6) {
            require(pack.expert_input_scale.size() == size_t(hp->n_layer) * hp->n_expert * 3,
                    "QFN_NVFP4_CALIBRATION_MISSING: calibrated-w4a4 requires validated experts.input_scale");
            if(opt.nvfp4_mode==4)
                require(ggml_backend_cuda_qfn_nvfp4_calibrated_available(error, sizeof(error)) != 0, error);
            if(opt.nvfp4_decode_mode==5)
                require(ggml_backend_cuda_qfn_nvfp4_calibrated_decode_available(error, sizeof(error)) != 0, error);
        }
        require(opt.tile > 0 && opt.tile <= INT32_MAX && uint64_t(hp->n_layer) * hp->n_expert <= UINT32_MAX,
                "QFN_NVFP4_WORKSPACE: tile/physical slot count overflows the bridge ABI");
        const uint32_t slots = hp->n_layer * hp->n_expert;
        require(opt.n_ctx > 0 && opt.n_ctx <= INT32_MAX, "QFN_NVFP4_WORKSPACE: context exceeds bridge ABI");
        const uint64_t prefill = nvfp4_workspace(int(opt.tile),512,opt.nvfp4_mode);
        const uint64_t decode = nvfp4_workspace(DEC_NMAX,slots,opt.nvfp4_decode_mode);
        require(prefill && decode, "QFN_NVFP4_WORKSPACE: DLL refused complete prefill/decode geometry");
        nvfp4_scratch_bytes = decode;
        require(nvfp4_scratch_bytes <= uint64_t(INT64_MAX), "QFN_NVFP4_WORKSPACE: byte count overflows ggml tensor");
    }

    uint64_t nvfp4_workspace(int tokens,uint32_t slots,int mode) const {
        return pack.native_nvfp4_packed ? ggml_backend_cuda_qfn_native_nvfp4_workspace_bytes(tokens,slots,mode) :
            ggml_backend_cuda_qfn_nvfp4_workspace_bytes(tokens,10,slots,mode);
    }
    void alloc_nvfp4_workspace() {
        if (!nvfp4) return;
        require((pack.native_nvfp4_packed ? ggml_backend_cuda_qfn_native_nvfp4_prepare(stream()) :
                ggml_backend_cuda_qfn_nvfp4_prepare(stream())) == int(cudaSuccess),
                "QFN_NVFP4_PREPARE: Basalt kernel/module preparation failed before graph capture");
        ggml_init_params ip = { 3 * ggml_tensor_overhead(), nullptr, true };
        nvfp4_ctx = ggml_init(ip);
        require(nvfp4_ctx != nullptr, "cannot create NVFP4 scratch context");
        nvfp4_scratch = ggml_new_tensor_1d(nvfp4_ctx, GGML_TYPE_I8, int64_t(nvfp4_scratch_bytes));
        if (!pack.expert_input_scale.empty())
            nvfp4_input_scale = ggml_new_tensor_3d(nvfp4_ctx, GGML_TYPE_F32, 3, hp->n_expert, hp->n_layer);
        nvfp4_buf = ggml_backend_alloc_ctx_tensors(nvfp4_ctx, gpu);
        require(nvfp4_buf && !(reinterpret_cast<uintptr_t>(nvfp4_scratch->data) & 255),
                "QFN_NVFP4_WORKSPACE: cannot allocate complete aligned persistent workspace");
        ggml_set_input(nvfp4_scratch);
        if (nvfp4_input_scale) {
            ggml_backend_tensor_set(nvfp4_input_scale, pack.expert_input_scale.data(), 0,
                                    pack.expert_input_scale.size() * sizeof(float));
            ggml_set_input(nvfp4_input_scale);
            log("[qfn] NVIDIA calibrated input_scale: %zu raw F32 scalars, SHA256 %s; stable GPU bytes %zu before cache sizing\n",
                pack.expert_input_scale.size(), pack.expert_input_scale_sha256.c_str(), ggml_nbytes(nvfp4_input_scale));
        }
        log("[qfn] NVFP4 Basalt ABI1: raw matrices plus all 16 tail bytes; unweighted expert rows, "
            "prefill mode %d (0 auto, 1 W4A4, 2 native-q8, 3 MMQ, 4 calibrated-w4a4, 6 dsv-native), decode mode %d (2 native-q8, 5 calibrated-w4a4, 6 dsv-native); persistent decode workspace %llu bytes; "
            "router weighting, K sums and shared expert unchanged\n", opt.nvfp4_mode, opt.nvfp4_decode_mode,
            (unsigned long long) nvfp4_scratch_bytes);
        if (pack.native_nvfp4_packed)
            log("[qfn] native-packed NVFP4 ABI1: lossless DSV values/scales layout, mode6 grouped NVFP4; reference modes invert on GPU; no weight requantization\n");
    }

    uint64_t nvfp4_prefill_required(int64_t rows) const {
        if (!nvfp4) return 0;
        require(rows > 0 && rows <= INT32_MAX, "QFN_NVFP4_WORKSPACE: invalid prefill row count");
        uint64_t bytes = nvfp4_workspace(int(rows),512,opt.nvfp4_mode);
        if(reference_prefill_active)for(int n=1;n<=std::min<int64_t>(4,rows);n++) {
            const uint64_t part=nvfp4_workspace(n,512,6);require(part,"Reference vector workspace geometry refused");
            bytes=std::max(bytes,part);
        }
        require(bytes && bytes <= uint64_t(INT64_MAX), "QFN_NVFP4_WORKSPACE: prefill scratch overflow/unsupported geometry");
        return bytes;
    }

    void ensure_nvfp4_prefill_workspace(int64_t rows) {
        const uint64_t bytes = nvfp4_prefill_required(rows);
        if (!bytes || bytes <= nvfp4_prefill_bytes) return;
        ggml_backend_synchronize(gpu); // only a prefill phase admission edge, never inside a captured decode graph
        if (nvfp4_prefill_buf) ggml_backend_buffer_free(nvfp4_prefill_buf);
        if (nvfp4_prefill_ctx) ggml_free(nvfp4_prefill_ctx);
        nvfp4_prefill_buf = nullptr; nvfp4_prefill_ctx = nullptr; nvfp4_prefill_scratch = nullptr;
        nvfp4_prefill_bytes = 0;
        ggml_init_params ip = {2 * ggml_tensor_overhead(), nullptr, true};
        nvfp4_prefill_ctx = ggml_init(ip);
        require(nvfp4_prefill_ctx != nullptr, "cannot create NVFP4 prefill workspace context");
        nvfp4_prefill_scratch = ggml_new_tensor_1d(nvfp4_prefill_ctx, GGML_TYPE_I8, int64_t(bytes));
        nvfp4_prefill_buf = ggml_backend_alloc_ctx_tensors(nvfp4_prefill_ctx, gpu);
        require(nvfp4_prefill_buf && !(reinterpret_cast<uintptr_t>(nvfp4_prefill_scratch->data) & 255),
                "cannot allocate complete aligned NVFP4 prefill workspace at admitted chunk size");
        nvfp4_prefill_bytes = bytes;
        ggml_set_input(nvfp4_prefill_scratch);
        log("[qfn] NVFP4 prefill workspace: rows %lld, mode %d, %llu bytes; decode pointer unchanged\n",
            (long long) rows, opt.nvfp4_mode, (unsigned long long) bytes);
    }

    void free_nvfp4_prefill_workspace() {
        if (!nvfp4_prefill_buf) return;
        ggml_backend_synchronize(gpu);
        ggml_backend_buffer_free(nvfp4_prefill_buf);
        ggml_free(nvfp4_prefill_ctx);
        nvfp4_prefill_buf = nullptr; nvfp4_prefill_ctx = nullptr; nvfp4_prefill_scratch = nullptr;
        nvfp4_prefill_bytes = 0;
    }

    void open_experts() {
        layer_bytes = pack.experts[0].blob * hp->n_expert;
        for (const auto & e : pack.experts) require(e.blob * hp->n_expert == layer_bytes, "expert layers differ in size");
        // a little room past the last expert: the MMQ K loop over a 640-wide down row reads up to one 256-value
        // iteration beyond it; that tail is zeroed once and never written by DirectStorage
        bank_bytes = layer_bytes + (64u << 10);
        wf_params prm = wf_params_default();
        if (!opt.experts_path.empty()) {
            // same container, same offsets: its header (section table) must equal the pack's
            file_reader a(pack.wpath, false), b(std::filesystem::path(opt.experts_path).wstring(), false);
            require(a.size == b.size, "the experts file differs in size from the pack");
            std::vector<uint8_t> ha(65536), hb(65536);
            a.read(0, ha.data(), ha.size()); b.read(0, hb.data(), hb.size());
            require(ha == hb, "the experts file header differs from the pack");
            log("[qfn] routed experts from %s\n", opt.experts_path.c_str());
        }
        prm.model_path = opt.experts_path.empty() ? pack.path.c_str() : opt.experts_path.c_str();
        prm.gpu_backend = gpu;
        prm.mode = WF_MODE_DSTORAGE;
        prm.n_layers = (int) hp->n_layer;
        prm.n_gpu_buffers = 2;
        prm.n_ram_buffers = 0;
        prm.max_layer_bytes = (size_t) bank_bytes;
        // the fetcher admits only its qualified relay profiles (DSV4.1 settings): the auto request size for the stage
        // (stage/1000 rounded to 4 KiB, 512 KiB..32 MiB: about 708 KiB here, above the measured 256 KiB RAID floor), the
        // small 250 MiB staging profile, asynchronous staging drain, direct compute banks
        prm.io_block_size = llama_wf_bounded::auto_io_block((size_t) bank_bytes);
        prm.ds_staging_limit_bytes = (size_t) llama_wf_bounded::small_staging_bytes;
        prm.ds_async_drain = true;
        prm.ds_compute_direct = true;
        wf = wf_create_required_dstorage(&prm);
        require(wf != nullptr, "DirectStorage expert streaming initialization failed (required; no fallback)");
        require(wf_is_ds_direct(wf) && wf_get_n_gpu_bufs(wf) == 2, "DirectStorage direct banks were not allocated");
        ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
        ectx = ggml_init(ip);
        const auto & e0 = pack.experts[0];
        for (int b = 0; b < 2; ++b) {
            bank_ptr[b] = wf_get_gpu_buf_ptr(wf, b);
            require(bank_ptr[b] != nullptr, "missing DirectStorage bank");
            bank_buf[b] = ggml_backend_cuda_buffer_from_ptr(wf_get_cuda_device_index(wf), bank_ptr[b], (size_t) bank_bytes);
            require(bank_buf[b] != nullptr, "cannot wrap a DirectStorage bank");
            ggml_backend_buffer_set_usage(bank_buf[b], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            // zero the tail past the experts once (see bank_bytes)
            std::vector<uint8_t> zeros(bank_bytes - layer_bytes, 0);
            ggml_tensor * tail = ggml_new_tensor_1d(ectx, GGML_TYPE_I8, (int64_t) zeros.size());
            ggml_backend_tensor_alloc(bank_buf[b], tail, (char *) bank_ptr[b] + layer_bytes);
            ggml_backend_tensor_set(tail, zeros.data(), 0, zeros.size());
            auto mk = [&](ggml_type type, int64_t ne0, int64_t ne1, uint64_t off) {
                ggml_tensor * t = ggml_new_tensor_3d(ectx, type, ne0, ne1, hp->n_expert);
                t->nb[1] = ggml_row_size(type, ne0);
                t->nb[2] = e0.blob;
                t->nb[3] = e0.blob * hp->n_expert;
                require(ggml_backend_tensor_alloc(bank_buf[b], t, (char *) bank_ptr[b] + off) == GGML_STATUS_SUCCESS,
                        "cannot bind expert tensors to a bank");
                return t;
            };
            if (nvfp4) {
                gate_exps[b] = ggml_new_tensor_2d(ectx, GGML_TYPE_I8, int64_t(e0.blob), hp->n_expert);
                require(ggml_backend_tensor_alloc(bank_buf[b], gate_exps[b], bank_ptr[b]) == GGML_STATUS_SUCCESS,
                        "cannot bind complete NVFP4 raw expert bank including scale tails");
            } else {
                gate_exps[b] = mk(e0.gu_type, hp->n_embd, hp->n_ff_exp, 0);
                up_exps[b]   = mk(e0.gu_type, hp->n_embd, hp->n_ff_exp, e0.gate_bytes);
                down_exps[b] = mk(e0.d_type, hp->n_ff_exp, hp->n_embd, 2 * e0.gate_bytes);
            }
        }
        for (uint32_t il = 0; il < hp->n_layer; ++il) {
            wf_weight_span span{};
            span.tensor_name = "qfn.experts";
            span.file_offset = pack.experts[il].offset;
            span.nbytes = (size_t) layer_bytes;
            span.alloc_size = 0;
            span.mmap_addr = nullptr;
            span.file_idx = 0;
            span.is_expert = false;
            span.expert_idx = -1;
            span.alloc_offset = 0;
            wf_register_layer(wf, (int) il, &span, 1, (size_t) layer_bytes);
        }
        require(!wf_has_failed(wf), "expert layer registration failed");
        log("[qfn] expert streaming: 2 DirectStorage banks of %.1f MiB, %u layers of %u experts (%.2f GiB total)\n",
            bank_bytes / 1048576.0, hp->n_layer, hp->n_expert, layer_bytes * (double) hp->n_layer / 1073741824.0);
    }

    // ------------------------------------------------------------------------------------------ state
    // Decode never reads the prefill carry (it has dec_res/dec_x); only MTP's draft layer does.
    void alloc_carry() {
        if (carry) return;
        ggml_init_params ip = { 4 * ggml_tensor_overhead(), nullptr, true };
        carry_ctx = ggml_init(ip);
        carry = ggml_new_tensor_2d(carry_ctx, carry_bf16 ? GGML_TYPE_BF16 : GGML_TYPE_F32, (int64_t) hp->n_embd * hp->hc, (int64_t) opt.n_ctx);
        carry_buf = ggml_backend_alloc_ctx_tensors(carry_ctx, gpu);
        require(carry_buf != nullptr, "cannot allocate the prefill carry in VRAM (lower --ctx)");
        ggml_backend_buffer_clear(carry_buf, 0);
    }
    void free_carry() {
        if (carry_in_slab) return;
        if (carry_buf) ggml_backend_buffer_free(carry_buf);
        if (carry_ctx) ggml_free(carry_ctx);
        carry_buf = nullptr; carry_ctx = nullptr; carry = nullptr;
    }

    void alloc_state() {
        const int64_t n_ctx = opt.n_ctx;
        const int64_t E = hp->n_embd, HC = hp->hc;
        const uint32_t nl = hp->n_layer;
        ggml_init_params ip = { (8 * nl + 16) * ggml_tensor_overhead(), nullptr, true };
        sctx = ggml_init(ip);
        { const char * cb = std::getenv("QFN_CARRY_BF16"); carry_bf16 = cb && *cb == '1'; }
        kcache.assign(nl, nullptr); vcache.assign(nl, nullptr); idx_raw.assign(nl, nullptr); idx_pool.assign(nl, nullptr);
        gdn_state.assign(nl, nullptr); conv_state.assign(nl, nullptr);
        const int64_t conv_ch = (int64_t) hp->ssm_state * hp->ssm_group * 2 + (int64_t) hp->ssm_state * hp->ssm_dt_rank;
        for (uint32_t il = 0; il < nl; ++il) {
            if (hp->is_recurrent(il)) {
                gdn_state[il]  = ggml_new_tensor_4d(sctx, GGML_TYPE_F32, hp->ssm_state, hp->ssm_state, hp->ssm_dt_rank, 1);
                conv_state[il] = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, hp->ssm_conv - 1, conv_ch);
            } else {
                kcache[il]   = ggml_new_tensor_3d(sctx, GGML_TYPE_F16, hp->head_dim, hp->n_head_kv, n_ctx);
                vcache[il]   = ggml_new_tensor_3d(sctx, GGML_TYPE_F16, hp->head_dim, hp->n_head_kv, n_ctx);
                // F16 like upstream's indexer cache (it takes the attention K-cache type): keys round before pooling
                idx_raw[il]  = ggml_new_tensor_2d(sctx, GGML_TYPE_F16, hp->idx_dim, n_ctx);
                idx_pool[il] = ggml_new_tensor_2d(sctx, GGML_TYPE_F16, hp->idx_dim, std::max<int64_t>(1, n_ctx / hp->kpool));
            }
        }
        ple_conv = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, (int64_t) (hp->ple_conv - 1) * hp->ple_ngram, E * HC);
        sbuf = ggml_backend_alloc_ctx_tensors(sctx, gpu);
        require(sbuf != nullptr, "cannot allocate the sequence state in VRAM (lower --ctx)");
        log("[qfn] sequence state: n_ctx %lld, %.2f GiB in VRAM (prefill carry separate)\n", (long long) n_ctx,
            ggml_backend_buffer_get_size(sbuf) / 1073741824.0);
    }

    // ------------------------------------------------------------------------------------------ PLE rows
    // upstream llm_graph_input_qwen4exp_ple::set_input, for a fresh sequence of text tokens
    void ple_hash(const std::vector<int32_t> & tok, std::vector<uint32_t> & idx) const {
        const int64_t n = (int64_t) tok.size(), ng = hp->ple_ngram, nh = hp->ple_n_heads, per = hp->ple_heads_per_ngram;
        const int64_t eos = hp->ple_eos;
        idx.resize((size_t) (n * nh));
        std::vector<int64_t> ctx(ng);
        for (int64_t i = 0; i < n; ++i) {
            ctx[0] = tok[i];
            bool cut = false;
            for (int64_t s = 1; s < ng; ++s) {
                const int64_t t = (i - s >= 0 && !cut) ? tok[i - s] : -1;
                cut = cut || t < 0 || t == eos;
                ctx[s] = cut ? eos : t;
            }
            for (int64_t g = 2; g <= ng; ++g) {
                uint64_t mixed = (uint64_t) ctx[0] * hp->ple_mult[0];
                for (int64_t j = 1; j < g; ++j) mixed ^= (uint64_t) ctx[j] * hp->ple_mult[j];
                const int64_t base = (g - 2) * per;
                for (int64_t h = 0; h < per; ++h) {
                    const int64_t hi = base + h;
                    idx[(size_t) (i * nh + hi)] = (uint32_t) (mixed % hp->ple_head_vocab[hi] + hp->ple_head_off[hi]);
                }
            }
        }
    }

    void validate_prompt(const std::vector<int32_t> & tokens, int64_t past = 0) const {
        require(!tokens.empty() && uint64_t(past) + tokens.size() <= opt.n_ctx, "prompt longer than the engine context");
        for (int32_t t : tokens) require(t >= 0 && uint32_t(t) < hp->n_vocab, "token id out of range");
        for(const auto &image:images) {
            require(uint64_t(image.position)+image.count<=uint64_t(past)+tokens.size(),"image span exceeds its prompt");
            const auto first=std::max<int64_t>(past,image.position),end=std::min<int64_t>(past+tokens.size(),uint64_t(image.position)+image.count);
            for(int64_t i=first;i<end;i++)require(uint32_t(tokens[size_t(i-past)])==hp->ple_image,"image span does not cover image placeholder tokens");
        }
    }

    void start_weight_stream(bool pause) {
        if (wf_running) return;
        wf_hold_from(wf, pause ? 1 : INT_MAX);
        wf_start(wf);
        wf_running = true;
        require(!wf_has_failed(wf), "expert stream start failed");
        llama_dsv41_engram_weight_stream(true);
    }

    void cancel_prefill() {
        // Join while both the verified file and the fetcher are still alive.
        ple_pending.reset();
        if (wf_running) {
            wf_hold_from(wf, INT_MAX);
            wf_stop(wf);
            wf_quiesce(wf);
            wf_running = false;
        }
        llama_dsv41_engram_weight_stream(false);
    }

    void queue_ple(const std::vector<int32_t> & seq_all, int64_t p0, bool cold) {
        require(!ple_pending, "a PLE request is already pending");
        const bool early = ple_prompt_overlap && llama_dsv41_engram_early_enabled();
        const bool pause = early && llama_dsv41_engram_early_pause();
        // Unlike the original QFN path, even a synchronous diagnostic starts
        // the expert stream first. Pause uses the shared fetcher's layer hold.
        start_weight_stream(pause);
        ple_pending.reset(new ple_request);
        auto * job = ple_pending.get();
        job->tokens = seq_all;
        job->p0 = p0;
        job->sweep_started.store(!cold);
        job->started_ms = now_ms();
        auto load = [this, job, cold, pause] {
            const HANDLE thread=GetCurrentThread();const int previous=GetThreadPriority(thread);
            const int requested=opt.ple_thread_above_normal?THREAD_PRIORITY_ABOVE_NORMAL:THREAD_PRIORITY_NORMAL;
            FILETIME created{},exited{},kernel0{},user0{},kernel1{},user1{};ULONG64 cycles0=0,cycles1=0;
            const bool cpu0=GetThreadTimes(thread,&created,&exited,&kernel0,&user0)&&QueryThreadCycleTime(thread,&cycles0);
            bool changed=false;
            try {
                require(previous!=THREAD_PRIORITY_ERROR_RETURN,"PLE worker priority query failed");
                if(previous!=requested){require(SetThreadPriority(thread,requested)!=0,"PLE worker priority selection failed");changed=true;}
                const int64_t h0 = std::max<int64_t>(0, job->p0 - int64_t(hp->ple_ngram) + 1);
                const std::vector<int32_t> tok(job->tokens.begin() + h0, job->tokens.end());
                const int64_t skip = job->p0 - h0;
                const qfn_tensor & pt = pack.get("per_layer_token_embd.weight");
                const uint64_t row_bytes = ggml_row_size(pt.type, pt.ne[0]);
                ple_hash(tok, job->idx);
                job->idx.erase(job->idx.begin(), job->idx.begin() + skip * hp->ple_n_heads);
                uint64_t planned=0;bool scan=false;
                if(opt.ple_io_workers==1) {
                    std::array<std::vector<uint32_t>,2> ids{job->idx,{}};
                    job->cache.set_submit_low_water(opt.ple_submit_low_water);
                    job->cache.set_completion_poll_us(opt.ple_poll_us);
                    job->cache.prepare_rows_early(std::move(ids),{pt.offset,0},uint32_t(row_bytes),
                        {uint64_t(pt.ne[1]),0},opt.host_budget);
                    planned=job->cache.planned_requests();scan=job->cache.scan_geometry();
                } else {
                    job->parallel=true;
                    const auto bounds=std::minmax_element(job->idx.begin(),job->idx.end());
                    require(bounds.first!=job->idx.end(),"PLE parallel load requires nonempty rows");
                    const uint64_t low=*bounds.first,width=(uint64_t(*bounds.second)-low+opt.ple_io_workers)/opt.ple_io_workers;
                    std::vector<std::vector<uint32_t>> parts(opt.ple_io_workers);
                    for(uint32_t id:job->idx)parts[(uint64_t(id)-low)/width].push_back(id);
                    // Reserve the maximum flattened output charge before admitting
                    // any shard. No loader or row count is silently dropped to fit.
                    const uint64_t reserve=uint64_t(job->idx.size())*(row_bytes+4);
                    require(reserve<opt.host_budget,"PLE parallel merge exceeds explicit host budget");
                    const uint64_t each=(opt.host_budget-reserve)/opt.ple_io_workers;
                    for(auto &part:parts)if(!part.empty()) {
                        auto cache=std::make_unique<llama_dsv41_engram_prompt_cache>();
                        cache->set_submit_low_water(opt.ple_submit_low_water);
                        cache->set_completion_poll_us(opt.ple_poll_us);
                        std::array<std::vector<uint32_t>,2> ids{std::move(part),{}};
                        cache->prepare_rows_early(std::move(ids),{pt.offset,0},uint32_t(row_bytes),{uint64_t(pt.ne[1]),0},each);
                        planned+=cache->planned_requests();scan=scan||cache->scan_geometry();
                        job->shards.push_back(std::move(cache));
                    }
                    job->parallel_scan=scan;
                }
                // Preserve the DSV4.1 auto arm: scans always start early;
                // random loads that cannot fit the cold gap start at prefill.
                if (cold && llama_dsv41_engram_early_auto()) {
                    const auto d = llama_dsv41_engram_early::decide(planned,
                            llama_dsv41_engram_early_iops(), llama_dsv41_engram_early::cold_gap_s);
                    if (!scan && !d.early) {
                        log("[qfn] PLE early auto: random load deferred to sweep start (predicted %.3f s)\n", d.predicted_s);
                        while (!job->sweep_started.load() && !job->cancel.load())
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                }
                if(!job->parallel) {
                    file_reader verified(pack.wpath,false);
                    job->cache.load_early(pack.wpath.c_str(),verified.h,verified.size,[job]{return job->cancel.load();});
                } else {
                    struct measure {double kernel=0,user=0;uint64_t cycles=0;bool available=false;};
                    std::vector<measure> cpu(job->shards.size());std::vector<std::exception_ptr> failures(job->shards.size());
                    auto read_part=[&](size_t i){
                        HANDLE self=GetCurrentThread();const int old=GetThreadPriority(self);
                        FILETIME a{},b{},k0{},u0{},k1{},u1{};ULONG64 c0=0,c1=0;
                        const bool first=GetThreadTimes(self,&a,&b,&k0,&u0)&&QueryThreadCycleTime(self,&c0);bool changed=false;
                        try {
                            require(old!=THREAD_PRIORITY_ERROR_RETURN,"PLE shard priority query failed");
                            if(old!=requested){require(SetThreadPriority(self,requested)!=0,"PLE shard priority selection failed");changed=true;}
                            file_reader verified(pack.wpath,false);
                            job->shards[i]->load_early(pack.wpath.c_str(),verified.h,verified.size,[job]{return job->cancel.load();});
                        }catch(...){failures[i]=std::current_exception();job->cancel.store(true);}
                        if(changed&&!SetThreadPriority(self,old))failures[i]=std::make_exception_ptr(std::runtime_error("PLE shard priority restoration failed"));
                        cpu[i].available=first&&GetThreadTimes(self,&a,&b,&k1,&u1)&&QueryThreadCycleTime(self,&c1);
                        auto ticks=[](const FILETIME&t){return (uint64_t(t.dwHighDateTime)<<32)|t.dwLowDateTime;};
                        if(cpu[i].available){cpu[i].kernel=double(ticks(k1)-ticks(k0))/10000.0;cpu[i].user=double(ticks(u1)-ticks(u0))/10000.0;cpu[i].cycles=c1-c0;}
                    };
                    std::vector<std::thread> readers;readers.reserve(job->shards.size()-1);
                    try{for(size_t i=1;i<job->shards.size();i++)readers.emplace_back(read_part,i);read_part(0);}
                    catch(...){job->cancel.store(true);for(auto &reader:readers)reader.join();throw;}
                    for(auto &reader:readers)reader.join();
                    for(size_t i=0;i<failures.size();i++)if(failures[i])std::rethrow_exception(failures[i]);
                    for(size_t i=1;i<cpu.size();i++){job->child_cpu_available=job->child_cpu_available&&cpu[i].available;job->child_kernel_ms+=cpu[i].kernel;job->child_user_ms+=cpu[i].user;job->child_cycles+=cpu[i].cycles;}
                    size_t count=0;for(const auto &part:job->shards)count+=part->table_ids(0).size();
                    job->joined_ids.reserve(count);job->joined_rows.resize(count*row_bytes);size_t at=0;
                    for(const auto &part:job->shards) {
                        const auto &ids=part->table_ids(0);job->joined_ids.insert(job->joined_ids.end(),ids.begin(),ids.end());
                        std::memcpy(job->joined_rows.data()+at*row_bytes,part->table_rows(0),ids.size()*row_bytes);at+=ids.size();
                    }
                    require(std::is_sorted(job->joined_ids.begin(),job->joined_ids.end())&&
                        std::adjacent_find(job->joined_ids.begin(),job->joined_ids.end())==job->joined_ids.end(),"PLE shard ranges overlap or are unordered");
                    log("[qfn] PLE parallel owners: requested=%u active=%zu unique_rows=%zu merged_bytes=%zu admitted_host_bytes=%llu\n",
                        opt.ple_io_workers,job->shards.size(),count,job->joined_rows.size(),(unsigned long long)opt.host_budget);
                }
            } catch (...) { job->failure = std::current_exception(); }
            if(changed&&!SetThreadPriority(thread,previous))job->failure=std::make_exception_ptr(std::runtime_error("PLE worker priority restoration failed"));
            const bool cpu1=GetThreadTimes(thread,&created,&exited,&kernel1,&user1)&&QueryThreadCycleTime(thread,&cycles1);
            auto ticks=[](const FILETIME &t){return (uint64_t(t.dwHighDateTime)<<32)|t.dwLowDateTime;};
            const bool measured=cpu0&&cpu1&&job->child_cpu_available;
            log("[qfn] PLE worker: priority=%s cpu_times_available=%d kernel_ms=%.3f user_ms=%.3f cycles=%llu\n",
                opt.ple_thread_above_normal?"above-normal":"normal",int(measured),
                measured?double(ticks(kernel1)-ticks(kernel0))/10000.0+job->child_kernel_ms:-1.0,
                measured?double(ticks(user1)-ticks(user0))/10000.0+job->child_user_ms:-1.0,
                (unsigned long long)(measured?cycles1-cycles0+job->child_cycles:0));
            job->wall_ms = now_ms() - job->started_ms;
            if (pause) wf_hold_from(wf, INT_MAX);
        };
        log("[qfn] PLE early queued: at=%s async=%d weight_pass_at_start=1\n", cold ? "cold_open" : "tokens_arrived", int(early));
        if (early) job->worker = std::thread(load);
        else { job->synchronous = true; load(); }
    }

    // Only the main thread touches the GPU. The joined shared loader supplies
    // exactly the original sorted unique rows and token/head slot map.
    void join_ple(std::vector<int32_t> & slot, double & read_wait_ms, double & upload_ms) {
        require(bool(ple_pending), "missing PLE request at the first PLE layer");
        auto & job = *ple_pending;
        const double joined = now_ms();
        if (job.worker.joinable()) job.worker.join();
        read_wait_ms = now_ms() - joined + (job.synchronous ? job.wall_ms : 0);
        if (job.failure) std::rethrow_exception(job.failure);
        auto & cache = job.cache;
        const auto & idx = job.idx;
        const qfn_tensor & pt = pack.get("per_layer_token_embd.weight");
        const uint64_t row_bytes = ggml_row_size(pt.type, pt.ne[0]);
        const double upload_started = now_ms();
        const auto & uids = job.parallel?job.joined_ids:cache.table_ids(0);
        const uint8_t * rows = job.parallel?job.joined_rows.data():cache.table_rows(0);
        slot.resize(idx.size());
        const uint32_t threads=uint32_t(std::min<size_t>(opt.ple_slot_threads,idx.size()));
        auto slots=[&](uint32_t part){
            const size_t begin=idx.size()*part/threads,end=idx.size()*(part+1)/threads;
            for(size_t i=begin;i<end;i++)slot[i]=int32_t(std::lower_bound(uids.begin(),uids.end(),idx[i])-uids.begin());
        };
        std::vector<std::thread> workers;workers.reserve(threads?threads-1:0);
        try{for(uint32_t part=1;part<threads;part++)workers.emplace_back(slots,part);if(threads)slots(0);}
        catch(...){for(auto &worker:workers)worker.join();throw;}
        for(auto &worker:workers)worker.join();
        const double slots_ms=now_ms()-upload_started;
        if (pbuf) { ggml_backend_buffer_free(pbuf); pbuf = nullptr; }
        if (pctx) { ggml_free(pctx); pctx = nullptr; }
        ggml_init_params ip = { 2 * ggml_tensor_overhead(), nullptr, true };
        pctx = ggml_init(ip);
        ple_rows = ggml_new_tensor_2d(pctx, pt.type, pt.ne[0], (int64_t) uids.size());
        pbuf = ggml_backend_alloc_ctx_tensors(pctx, gpu);
        require(pbuf != nullptr, "cannot allocate the prompt's PLE rows in VRAM");
        ggml_backend_tensor_set(ple_rows, rows, 0, uids.size() * row_bytes);
        if (!opt.ple_row_dump.empty()) {
            std::ofstream dump(opt.ple_row_dump, std::ios::binary);
            require(bool(dump), "cannot open uploaded PLE row dump");
            std::vector<uint8_t> chunk(1 << 20);
            const size_t bytes = uids.size() * row_bytes;
            for (size_t off = 0; off < bytes; off += chunk.size()) {
                const size_t len = std::min(chunk.size(), bytes - off);
                ggml_backend_tensor_get(ple_rows, chunk.data(), off, len);
                dump.write(reinterpret_cast<const char *>(chunk.data()), len);
            }
            dump.close();
            require(bool(dump), "cannot write uploaded PLE row dump");
            log("[qfn] uploaded PLE row dump: %zu bytes to %s\n", bytes, opt.ple_row_dump.c_str());
        }
        log("[qfn] PLE rows: %zu lookups, %zu unique rows (%.1f MiB), %.0f ms (%s geometry)\n",
            idx.size(), uids.size(), uids.size() * row_bytes / 1048576.0, job.wall_ms,
            job.parallel?(job.parallel_scan?"parallel scan":"parallel random"):(cache.scan_geometry()?"scan":"random"));
        upload_ms = now_ms() - upload_started;
        log("[qfn] PLE slot map: threads=%u rows=%zu milliseconds=%.3f remaining_upload_ms=%.3f\n",
            threads,idx.size(),slots_ms,upload_ms-slots_ms);
        log("[qfn] PLE overlap: load_wall_ms=%.3f exposed_read_wait_ms=%.3f upload_and_slots_ms=%.3f\n",
            job.wall_ms, read_wait_ms, upload_ms);
        ple_pending.reset();
    }

    // ------------------------------------------------------------------------------------------ graph pieces
    struct gctx {
        ggml_context * ctx;
        ggml_cgraph * gf;
    };

    // ---- exact multi-token decode: per-token splits of the ops whose kernels depend on the column count
    bool split_tokens = false;
    bool reference_prefill_active = false;
    std::vector<ggml_backend_buffer_t> reference_weight_buffers;
    std::vector<uint8_t> reference_quantized;
    std::vector<float> reference_float_row;
    std::vector<ggml_fp16_t> reference_half;
    void release_reference_weights() {
        if(!reference_weight_buffers.empty())ggml_backend_synchronize(gpu);
        for(auto *buffer:reference_weight_buffers)ggml_backend_buffer_free(buffer);
        reference_weight_buffers.clear();
    }
    void retain_reference_host_buffers() {
        const size_t bytes=reference_quantized.capacity()+reference_float_row.capacity()*sizeof(float)+
            reference_half.capacity()*sizeof(ggml_fp16_t);
        // Keep this bounded host arena for the engine lifetime. Freeing a
        // pinned heap arena can turn its holes into inaccessible private pages
        // which cannot be locked by the following residency scan. Reuse keeps
        // the actual host coefficients readable and eligible for VirtualLock;
        // the independent zero-owned-paging guard is unchanged.
        log("[qfn] reference host workspace retained: %zu bytes quant=%p/%zu row=%p/%zu half=%p/%zu; graph GPU coefficients released; guard unchanged\n",bytes,
            reference_quantized.data(),reference_quantized.capacity(),reference_float_row.data(),reference_float_row.capacity()*sizeof(float),
            reference_half.data(),reference_half.capacity()*sizeof(ggml_fp16_t));
    }
    uint32_t reference_prefill_rows=0;
    size_t graph_node_capacity() const {
        if(!reference_prefill_active)return 8192;
        size_t nodes=8192;while(nodes<size_t(reference_prefill_rows)*128)nodes*=2;return nodes;
    }
    ggml_tensor * reference_columns(ggml_context *ctx,ggml_tensor *weight,ggml_tensor *x) {
        ggml_tensor *out=nullptr;
        // Native MMVF retains F32 inputs/accumulation with either coefficient
        // storage type. NVIDIA admits one F16 column or three F32 columns.
        const int64_t columns=weight->type==GGML_TYPE_F16?1:3;
        for(int64_t first=0;first<x->ne[1];first+=columns) {
            const int64_t count=std::min<int64_t>(columns,x->ne[1]-first);
            // Keep every broadcast dimension and the actual token/batch strides.
            auto *part=ggml_view_4d(ctx,x,x->ne[0],count,x->ne[2],x->ne[3],x->nb[1],x->nb[2],x->nb[3],size_t(first)*x->nb[1]);
            auto *product=ggml_mul_mat(ctx,weight,part);ggml_mul_mat_set_prec(product,GGML_PREC_F32);
            out=out?ggml_concat(ctx,out,product,1):product;
        }
        require(out,"Reference matrix has no input columns");return out;
    }

    ggml_tensor * reference_matmul(ggml_context *ctx,ggml_tensor *w,ggml_tensor *x) {
        if(!reference_prefill_active)return ggml_mul_mat(ctx,w,x);
        if(w->type==GGML_TYPE_F32||w->type==GGML_TYPE_F16) {
            return reference_columns(ctx,w,x);
        }
        require(w->ne[2]==1&&w->ne[3]==1&&(w->type==GGML_TYPE_BF16||w->type==GGML_TYPE_F16||ggml_is_quantized(w->type)),
            "Reference prefill matrix has unsupported native geometry/type");
        // Bound host conversion rows to32MiB, retaining original values. The
        // existing graph reserve admits these temporary GPU matrices; original
        // model weights and the complete head on disk remain unchanged.
        const int64_t block=std::max<int64_t>(1,std::min<int64_t>(8192,(32ll<<20)/(w->ne[0]*4)));
        ggml_tensor *out=nullptr;
        for(int64_t first=0;first<w->ne[1];first+=block) {
            const int64_t count=std::min(block,w->ne[1]-first);
            auto *view=ggml_view_2d(ctx,w,w->ne[0],count,w->nb[1],size_t(first)*w->nb[1]);
            // The accepted CUDA backend does not implement Q6_K GET_ROWS.
            // Materialize the original values with its existing CPU decoder,
            // Use bounded reusable host buffers and upload the exact Mac-F16 values
            // as F16 coefficients, preserving F32 activations and accumulation.
            // GPU buffers live through this graph and are freed before the next.
            const auto *traits=ggml_get_type_traits(w->type);
            require(w->data&&traits&&traits->to_float,"Reference weight has no readable native row decoder");
            const size_t source_row=ggml_row_size(w->type,w->ne[0]);
            reference_quantized.resize(size_t(count)*source_row);
            reference_float_row.resize(size_t(w->ne[0]));
            reference_half.resize(size_t(count*w->ne[0]));
            if(w->nb[1]==source_row) {
                require(cudaMemcpy(reference_quantized.data(),(const char*)w->data+size_t(first)*w->nb[1],
                    size_t(count)*source_row,cudaMemcpyDeviceToHost)==cudaSuccess,"Reference weight readback failed");
            } else for(int64_t row=0;row<count;row++) {
                require(cudaMemcpy(reference_quantized.data()+size_t(row)*source_row,
                    (const char*)w->data+size_t(first+row)*w->nb[1],source_row,cudaMemcpyDeviceToHost)==cudaSuccess,
                    "Reference strided weight readback failed");
            }
            for(int64_t row=0;row<count;row++) {
                float *values=reference_float_row.data();
                traits->to_float(reference_quantized.data()+size_t(row)*source_row,values,w->ne[0]);
                ggml_fp32_to_fp16_row(values,reference_half.data()+size_t(row*w->ne[0]),w->ne[0]);
            }
            auto *dense=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,w->ne[0],count);
            const auto type=ggml_backend_get_default_buffer_type(gpu);
            auto *buffer=ggml_backend_buft_alloc_buffer(type,ggml_backend_buft_get_alloc_size(type,dense));
            require(buffer,"Reference weight upload exceeded admitted graph VRAM");
            reference_weight_buffers.push_back(buffer);
            require(ggml_backend_tensor_alloc(buffer,dense,ggml_backend_buffer_get_base(buffer))==GGML_STATUS_SUCCESS,
                "Reference weight tensor allocation failed");
            ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_tensor_set(dense,reference_half.data(),0,reference_half.size()*sizeof(ggml_fp16_t));
            auto *part=reference_columns(ctx,dense,x);
            out=out?ggml_concat(ctx,out,part,0):part;
        }
        require(out,"Reference prefill matrix has no rows");return out;
    }
    ggml_tensor * MM(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
        if (!split_tokens || ggml_n_dims(x) != 2 || x->ne[1] <= 1) return reference_matmul(ctx, w, x);
        ggml_tensor * out = nullptr;
        for (int64_t t = 0; t < x->ne[1]; ++t) {
            ggml_tensor * xt = ggml_cont(ctx, ggml_view_2d(ctx, x, x->ne[0], 1, x->nb[1], (size_t) t * x->nb[1]));
            ggml_tensor * yt = reference_matmul(ctx, w, xt);
            out = out ? ggml_concat(ctx, out, yt, 1) : yt;
        }
        return out;
    }
    ggml_tensor * MMID(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, ggml_tensor * ids, bool capture_rows = false) {
        if ((!split_tokens && !capture_rows) || x->ne[2] <= 1) return ggml_mul_mat_id(ctx, w, x, ids);
        ggml_tensor * out = nullptr;
        for (int64_t t = 0; t < x->ne[2]; ++t) {
            ggml_tensor * xt = ggml_cont(ctx, ggml_view_3d(ctx, x, x->ne[0], x->ne[1], 1, x->nb[1], x->nb[2], (size_t) t * x->nb[2]));
            ggml_tensor * it = ggml_cont(ctx, ggml_view_2d(ctx, ids, ids->ne[0], 1, ids->nb[1], (size_t) t * ids->nb[1]));
            ggml_tensor * yt = ggml_mul_mat_id(ctx, w, xt, it);
            out = out ? ggml_concat(ctx, out, yt, 2) : yt;
        }
        return out;
    }
    ggml_tensor * LIDX(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * w, float a, float b, ggml_tensor *positions) {
        auto score = [&](ggml_tensor * query, ggml_tensor * keys, ggml_tensor * weights, ggml_tensor *pos) {
            if (opt.qsa_indexer == "basalt-tc-refine" ||
                (opt.qsa_indexer == "basalt-tc-refine-prefill" && query->ne[2] >= 64)) {
                auto *result=ggml_qfn_qsa_score(ctx,query,keys,weights,a,b);
                result->op_params[1]=3;result->src[3]=pos;
                result->op_params[4]=int(hp->idx_top_k/hp->kpool);result->op_params[5]=int(hp->kpool);
                return result;
            }
            if (opt.qsa_indexer == "dsv-fp4") {
                auto *result = ggml_qfn_qsa_score(ctx, query, keys, weights, a, b);
                result->op_params[1] = 2; // distinct explicit arithmetic policy
                return result;
            }
            const bool tensor_core = opt.qsa_indexer == "basalt-tc" ||
                (opt.qsa_indexer == "basalt-tc-prefill" && query->ne[2] >= 64);
            return tensor_core ? ggml_qfn_qsa_score(ctx, query, keys, weights, a, b)
                               : ggml_lightning_indexer(ctx, query, keys, weights, a, b);
        };
        if (!split_tokens || q->ne[2] <= 1) return score(q, k, w, positions);
        ggml_tensor * out = nullptr;
        for (int64_t t = 0; t < q->ne[2]; ++t) {
            ggml_tensor * qt = ggml_cont(ctx, ggml_view_4d(ctx, q, q->ne[0], q->ne[1], 1, 1, q->nb[1], q->nb[2], q->nb[3], (size_t) t * q->nb[2]));
            ggml_tensor * wt = ggml_cont(ctx, ggml_view_2d(ctx, w, w->ne[0], 1, w->nb[1], (size_t) t * w->nb[1]));
            ggml_tensor *pt=ggml_view_1d(ctx,positions,1,size_t(t)*sizeof(int32_t));
            ggml_tensor * yt = score(qt, k, wt,pt);
            yt = ggml_reshape_2d(ctx, yt, ggml_nelements(yt), 1);
            out = out ? ggml_concat(ctx, out, yt, 1) : yt;
        }
        return out;
    }

    // grouped RMSNorm over each hc stream, then the low-rank gate and the mean over the streams
    ggml_tensor * hc_mix(gctx & g, ggml_tensor * x, ggml_tensor * w_norm, ggml_tensor * w_down, ggml_tensor * w_up,
                         ggml_tensor * w_inject, ggml_tensor ** inject_out) {
        ggml_context * ctx = g.ctx;
        const int64_t E = hp->n_embd, HC = hp->hc, nt = x->ne[2];
        ggml_tensor * xn = ggml_mul(ctx, ggml_rms_norm(ctx, x, hp->rms_eps), ggml_reshape_2d(ctx, w_norm, E, HC));
        xn = ggml_reshape_2d(ctx, xn, E * HC, nt);
        ggml_tensor * lo = ggml_scale(ctx, MM(ctx, w_down, xn), 1.0f / (float) HC);
        lo = ggml_silu(ctx, lo);
        ggml_tensor * gate = MM(ctx, w_up, lo);
        if (fused_hc()) {
            // one kernel, same products and summation order as the unfused graph below
            ggml_tensor * mixed = ggml_qfn_hc_mix(ctx, ggml_reshape_3d(ctx, xn, E, HC, nt), gate, 1.0f / (float) HC);
            if (inject_out) *inject_out = w_inject ? ggml_scale(ctx, MM(ctx, w_inject, xn), 1.0f / (float) HC) : nullptr;
            return mixed;
        }
        ggml_tensor * gated = ggml_mul(ctx, xn, ggml_sigmoid(ctx, gate));
        gated = ggml_reshape_3d(ctx, gated, E, HC, nt);
        ggml_tensor * mixed = ggml_cont(ctx, ggml_view_2d(ctx, gated, E, nt, ggml_row_size(gated->type, E) * HC, 0));
        for (int64_t c = 1; c < HC; ++c) {
            ggml_tensor * s = ggml_view_2d(ctx, gated, E, nt, ggml_row_size(gated->type, E) * HC, ggml_row_size(gated->type, E) * c);
            mixed = ggml_add(ctx, mixed, s);
        }
        mixed = ggml_scale(ctx, mixed, 1.0f / (float) HC);
        if (inject_out) {
            *inject_out = w_inject ? ggml_scale(ctx, MM(ctx, w_inject, xn), 1.0f / (float) HC) : nullptr;
        }
        return mixed;
    }

    // Qualified at 100K: fused HC is the default; QFN_FUSED_HC=0 retains the reference path.
    bool fused_hc() const {
        static const bool on = [] { const char * e = std::getenv("QFN_FUSED_HC"); return !e || *e == '1'; }();
        return on;
    }

    ggml_tensor * hc_combine(gctx & g, ggml_tensor * residual, ggml_tensor * block_out, ggml_tensor * inject) {
        ggml_context * ctx = g.ctx;
        const int64_t E = hp->n_embd, HC = hp->hc, nt = residual->ne[2];
        if (fused_hc()) {
            auto C = [&](ggml_tensor * t) { return ggml_is_contiguous(t) ? t : ggml_cont(ctx, t); };
            return ggml_qfn_hc_combine(ctx, C(residual), C(block_out), C(inject));
        }
        ggml_tensor * w = ggml_scale(ctx, ggml_sigmoid(ctx, inject), 2.0f);
        w = ggml_reshape_3d(ctx, w, 1, HC, nt);
        ggml_tensor * b = ggml_reshape_3d(ctx, block_out, E, 1, nt);
        b = ggml_repeat_4d(ctx, b, E, HC, nt, 1);
        return ggml_add(ctx, residual, ggml_mul(ctx, b, w));
    }

    ggml_tensor * rope(gctx & g, ggml_tensor * x, ggml_tensor * pos4) {
        int sections[4];
        for (int i = 0; i < 4; ++i) sections[i] = hp->rope_sections[i];
        return ggml_rope_multi(g.ctx, x, pos4, nullptr, (int) hp->n_rot, sections, GGML_ROPE_TYPE_IMROPE,
                               (int) hp->n_ctx_train, hp->rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    }

    // gated delta net layer (upstream build_layer_attn_linear), state carried in VRAM across tiles
    ggml_tensor * gdn_layer(gctx & g, uint32_t il, ggml_tensor * x, bool snapshots = false) {
        ggml_context * ctx = g.ctx;
        const auto & L = layers[il];
        const int64_t nt = x->ne[1];
        const int64_t S = hp->ssm_state, Hk = hp->ssm_group, Hv = hp->ssm_dt_rank;
        const int64_t conv_ch = S * Hk * 2 + S * Hv;
        ggml_tensor * qkv;
        ggml_tensor * z;
        if (paired_gdn) {
            require(il < paired_gdn_weights.size() && paired_gdn_weights[il],
                    "QFN_PAIRED_GDN: recurrent layer has no admitted combined weight span");
            ggml_tensor * both = MM(ctx, paired_gdn_weights[il], x);
            ggml_format_name(both, "qfn_paired_gdn_%u", il);
            // Each token has [qkv | gate] rows. Keep that actual token stride,
            // then materialize the original contiguous inputs to all existing
            // convolution, gate, snapshot and replay consumers.
            qkv = ggml_cont(ctx, ggml_view_2d(ctx, both, conv_ch, nt, both->nb[1], 0));
            z = ggml_cont(ctx, ggml_view_2d(ctx, both, S * Hv, nt, both->nb[1], size_t(conv_ch) * sizeof(float)));
        } else {
            qkv = MM(ctx, L.attn_qkv, x);                            // [conv_ch, nt]
            z   = MM(ctx, L.attn_gate, x);                           // [S*Hv, nt]
        }
        ggml_tensor * beta  = ggml_reshape_4d(ctx, MM(ctx, L.ssm_beta, x), 1, Hv, nt, 1);
        ggml_tensor * alpha = ggml_reshape_3d(ctx, MM(ctx, L.ssm_alpha, x), Hv, nt, 1);
        beta = ggml_sigmoid(ctx, beta);
        ggml_tensor * gate = ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, alpha, L.ssm_dt)), L.ssm_a);
        gate = ggml_reshape_4d(ctx, gate, 1, Hv, nt, 1);
        // conv history + this tile, then keep the last (kernel-1) columns for the next tile
        ggml_tensor * cs = conv_state[il];
        ggml_tensor * conv_in = ggml_concat(ctx, ggml_reshape_3d(ctx, cs, cs->ne[0], conv_ch, 1),
                                            ggml_transpose(ctx, ggml_reshape_3d(ctx, qkv, conv_ch, nt, 1)), 0);
        ggml_tensor * tail = ggml_view_3d(ctx, conv_in, cs->ne[0], conv_ch, 1, conv_in->nb[1], conv_in->nb[2],
                                          ggml_row_size(conv_in->type, conv_in->ne[0] - cs->ne[0]));
        ggml_tensor * conv_out = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_in, L.ssm_conv1d));  // [conv_ch, nt, 1]
        if (snapshots) {
            ggml_tensor * h = conv_hist[il];
            ggml_build_forward_expand(g.gf, ggml_cpy(ctx, ggml_cont(ctx, conv_in),
                                                     ggml_view_3d(ctx, h, conv_in->ne[0], conv_ch, 1, h->nb[1], h->nb[1] * conv_ch, 0)));
        }
        ggml_build_forward_expand(g.gf, ggml_cpy(ctx, ggml_cont(ctx, tail), ggml_reshape_3d(ctx, cs, cs->ne[0], conv_ch, 1)));
        const size_t nb1 = ggml_row_size(conv_out->type, conv_ch);
        ggml_tensor * q = ggml_view_4d(ctx, conv_out, S, Hk, nt, 1, ggml_row_size(conv_out->type, S), nb1, nb1 * nt, 0);
        ggml_tensor * k = ggml_view_4d(ctx, conv_out, S, Hk, nt, 1, ggml_row_size(conv_out->type, S), nb1, nb1 * nt,
                                       ggml_row_size(conv_out->type, S * Hk));
        ggml_tensor * v = ggml_view_4d(ctx, conv_out, S, Hv, nt, 1, ggml_row_size(conv_out->type, S), nb1, nb1 * nt,
                                       ggml_row_size(conv_out->type, 2 * S * Hk));
        const float eps = hp->rms_eps;
        q = ggml_scale(ctx, ggml_rms_norm(ctx, q, eps / (float) S), 1.0f / sqrtf((float) S));
        k = ggml_scale(ctx, ggml_rms_norm(ctx, k, eps / (float) S), 1.0f / sqrtf((float) S));
        ggml_tensor * st = gdn_state[il];
        ggml_tensor * out;
        if (snapshots && nt > 1) {
            // one token at a time; the state after every token but the last goes to its snapshot slot
            ggml_tensor * cur_state = st;
            out = nullptr;
            auto tok_view = [&](ggml_tensor * a, int64_t t) {
                return ggml_cont(ctx, ggml_view_4d(ctx, a, a->ne[0], a->ne[1], 1, 1, a->nb[1], a->nb[2], a->nb[3], t * a->nb[2]));
            };
            for (int64_t t = 0; t < nt; ++t) {
                ggml_tensor * o = ggml_gated_delta_net(ctx, tok_view(q, t), tok_view(k, t), tok_view(v, t), tok_view(gate, t),
                                                       tok_view(beta, t), cur_state, 1);
                ggml_tensor * ot = ggml_view_4d(ctx, o, S, Hv, 1, 1, ggml_row_size(o->type, S), ggml_row_size(o->type, S * Hv),
                                                ggml_row_size(o->type, S * Hv), 0);
                ggml_tensor * sv = ggml_view_1d(ctx, o, S * S * Hv, ggml_row_size(o->type, S * Hv));
                if (t + 1 < nt) {
                    ggml_tensor * slot = ggml_view_1d(ctx, gdn_snap[il], S * S * Hv, (size_t) t * gdn_snap[il]->nb[1]);
                    ggml_build_forward_expand(g.gf, ggml_cpy(ctx, sv, slot));
                } else {
                    ggml_build_forward_expand(g.gf, ggml_cpy(ctx, sv, ggml_reshape_1d(ctx, st, S * S * Hv)));
                }
                cur_state = ggml_reshape_4d(ctx, sv, S, S, Hv, 1);
                out = out ? ggml_concat(ctx, out, ggml_cont(ctx, ot), 2) : ggml_cont(ctx, ot);
            }
        } else {
            ggml_tensor * out_all = ggml_gated_delta_net(ctx, q, k, v, gate, beta, st, 1);
            out = ggml_view_4d(ctx, out_all, S, Hv, nt, 1, ggml_row_size(out_all->type, S),
                               ggml_row_size(out_all->type, S * Hv), ggml_row_size(out_all->type, S * Hv * nt), 0);
            ggml_tensor * new_state = ggml_view_1d(ctx, out_all, S * S * Hv, ggml_row_size(out_all->type, S * Hv * nt));
            ggml_build_forward_expand(g.gf, ggml_cpy(ctx, new_state, ggml_reshape_1d(ctx, st, S * S * Hv)));
        }
        ggml_tensor * z4 = ggml_reshape_4d(ctx, z, S, Hv, nt, 1);
        ggml_tensor * normed = ggml_mul(ctx, ggml_rms_norm(ctx, out, eps), L.ssm_norm);
        normed = ggml_mul(ctx, normed, ggml_sigmoid(ctx, z4));
        return MM(ctx, L.ssm_out, ggml_reshape_2d(ctx, normed, S * Hv, nt));
    }

    void admit_attention() {
        require(opt.qsa_attention==0 || opt.qsa_attention==1,"invalid QSA attention policy");
        if (!opt.qsa_attention) return;
        require(hp->head_dim==256 && hp->n_head_kv>0 && uint64_t(hp->n_head_kv)*12==hp->n_head &&
                hp->n_head_kv<=65535 && hp->full_attn_interval>0 && hp->full_attn_interval<=hp->n_layer,
                "QFN_ATTENTION_FORMAT: Basalt requires native F16KV/D256/H=12Hkv QSA geometry");
        require(opt.n_ctx>0 && opt.n_ctx<=INT32_MAX && opt.tile>0 && opt.tile<=INT32_MAX &&
                hp->kpool>1 && hp->idx_top_k/hp->kpool>0,"QFN_ATTENTION_GEOMETRY: context/tile/selection is outside the ABI");
        const uint64_t cap=uint64_t(hp->idx_top_k/hp->kpool)*hp->kpool+hp->kpool-1;
        require(cap<=INT32_MAX,"QFN_ATTENTION_GEOMETRY: selected-cell capacity overflows the ABI");
        char error[1024]={};require(ggml_backend_cuda_qfn_attention_available(error,sizeof(error))!=0,error);
        for (int nt=1;nt<=DEC_NMAX;++nt) {
            const uint64_t bytes=ggml_backend_cuda_qfn_attention_workspace_bytes(nt,int(hp->n_head),int(hp->n_head_kv),int(opt.n_ctx),int(cap),1.0f/16.0f);
            require(bytes!=UINT64_MAX && bytes>0 && bytes<=uint64_t(INT64_MAX),"QFN_ATTENTION_WORKSPACE: complete short-query geometry refused");
            attention_scratch_bytes=std::max(attention_scratch_bytes,bytes);
        }
    }
    void alloc_attention_workspace() {
        if (!opt.qsa_attention) return;
        require(ggml_backend_cuda_qfn_attention_prepare(stream())==int(cudaSuccess),"QFN_ATTENTION_PREPARE: module/PDL preparation failed before capture");
        ggml_init_params ip={2*ggml_tensor_overhead(),nullptr,true};attention_ctx=ggml_init(ip);
        require(attention_ctx!=nullptr,"cannot create persistent attention workspace context");
        attention_scratch=ggml_new_tensor_1d(attention_ctx,GGML_TYPE_I8,int64_t(attention_scratch_bytes));
        attention_buf=ggml_backend_alloc_ctx_tensors(attention_ctx,gpu);
        require(attention_buf && !(reinterpret_cast<uintptr_t>(attention_scratch->data)&255),"QFN_ATTENTION_WORKSPACE: persistent aligned allocation failed");
        ggml_set_input(attention_scratch);
        log("[qfn] QSA attention Basalt ABI1: prompt T>4, decode64 T<=4; persistent decode scratch %llu bytes before cache sizing; native F16KV/positions/gates retained\n",
            (unsigned long long)attention_scratch_bytes);
    }
    ggml_tensor * attention(ggml_context * ctx,ggml_tensor * q,ggml_tensor * gate,ggml_tensor * k,ggml_tensor * v,
        ggml_tensor * ids,ggml_tensor * pos,float scale) {
        if (!opt.qsa_attention) return ggml_qfn_qsa_attn(ctx,q,gate,k,v,ids,pos,scale);
        return ggml_qfn_attention(ctx,q,gate,k,v,ids,pos,scale,q->ne[2]<=DEC_NMAX?attention_scratch:nullptr);
    }

    // Qwen sparse attention layer (upstream build_layer_attn + build_qsa_sel, gather-sparse attention)
    ggml_tensor * qsa_layer(gctx & g, uint32_t il, ggml_tensor * x, ggml_tensor * pos, ggml_tensor * pos4,
                            ggml_tensor * pool_pos4, int64_t t0, int64_t nt, bool caches_only = false) {
        ggml_context * ctx = g.ctx;
        const auto & L = layers[il];
        const int64_t D = hp->head_dim, H = hp->n_head, Hkv = hp->n_head_kv, Di = hp->idx_dim, Hi = hp->idx_heads;
        const int64_t R = hp->kpool;
        // ---- indexer: raw keys into the cache, then pool every block this tile completes
        ggml_tensor * k_raw = MM(ctx, L.idx_k_proj, x);                                  // [Di, nt]
        ggml_tensor * raw_dst = ggml_view_2d(ctx, idx_raw[il], Di, nt, idx_raw[il]->nb[1], idx_raw[il]->nb[1] * t0);
        ggml_build_forward_expand(g.gf, ggml_cpy(ctx, k_raw, raw_dst));
        const int64_t b_lo = t0 / R, b_hi = (t0 + nt) / R;
        if (b_hi > b_lo) {
            const int64_t nn = b_hi - b_lo;
            ggml_tensor * rows = ggml_view_3d(ctx, idx_raw[il], Di, R, nn, idx_raw[il]->nb[1], idx_raw[il]->nb[1] * R,
                                              idx_raw[il]->nb[1] * R * b_lo);
            ggml_tensor * pooled = nullptr;
            for (int64_t i = 0; i < R; ++i) {
                ggml_tensor * s = ggml_cast(ctx, ggml_view_2d(ctx, rows, Di, nn, rows->nb[2], i * rows->nb[1]), GGML_TYPE_F32);
                pooled = pooled ? ggml_add(ctx, pooled, s) : s;
            }
            pooled = ggml_scale(ctx, pooled, 1.0f / (float) R);
            pooled = ggml_mul(ctx, ggml_rms_norm(ctx, pooled, hp->rms_eps), L.idx_k_norm);
            pooled = ggml_reshape_3d(ctx, pooled, Di, 1, nn);
            pooled = rope(g, pooled, pool_pos4);
            ggml_tensor * dst = ggml_view_2d(ctx, idx_pool[il], Di, nn, idx_pool[il]->nb[1], idx_pool[il]->nb[1] * b_lo);
            ggml_build_forward_expand(g.gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, pooled, Di, nn), dst));
        }
        if (caches_only) {
            // Exact cone: nothing after the last layer reads these rows' attention output; decode reads the caches.
            ggml_tensor * k = ggml_reshape_3d(ctx, MM(ctx, L.wk, x), D, Hkv, nt);
            k = ggml_mul(ctx, ggml_rms_norm(ctx, k, hp->rms_eps), L.k_norm);
            ggml_tensor * v = ggml_reshape_3d(ctx, MM(ctx, L.wv, x), D, Hkv, nt);
            k = rope(g, k, pos4);
            ggml_build_forward_expand(g.gf, ggml_cpy(ctx, k, ggml_view_3d(ctx, kcache[il], D, Hkv, nt, kcache[il]->nb[1], kcache[il]->nb[2], kcache[il]->nb[2] * t0)));
            ggml_build_forward_expand(g.gf, ggml_cpy(ctx, v, ggml_view_3d(ctx, vcache[il], D, Hkv, nt, vcache[il]->nb[1], vcache[il]->nb[2], vcache[il]->nb[2] * t0)));
            return nullptr;
        }
        ggml_tensor * qi = ggml_reshape_3d(ctx, MM(ctx, L.idx_q_proj, x), Di, Hi, nt);
        qi = ggml_mul(ctx, ggml_rms_norm(ctx, qi, hp->rms_eps), L.idx_q_norm);
        qi = rope(g, qi, pos4);
        // score the visible pools; a token sees the pools b < (pos+1)/R, so the tile needs b < b_hi
        ggml_tensor * score;
        if (b_hi > 0) {
            ggml_tensor * keys = ggml_view_4d(ctx, idx_pool[il], Di, 1, b_hi, 1, idx_pool[il]->nb[1], idx_pool[il]->nb[1],
                                              idx_pool[il]->nb[1] * b_hi, 0);
            ggml_tensor * wts = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Hi, nt), 1.0f / sqrtf((float) Di));
            score = LIDX(ctx, ggml_reshape_4d(ctx, qi, Di, Hi, nt, 1), keys, wts, 1.0f, 1.0f, pos);
            score = ggml_reshape_2d(ctx, score, b_hi, nt);
        } else {
            score = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, nt), 0.0f);
        }
        ggml_tensor * sel = ggml_qfn_qsa_select(ctx, score, pos, (int) (hp->idx_top_k / R), (int) R);
        // ---- attention projections
        ggml_tensor * qfull = MM(ctx, L.wq, x);                                           // [2*D*H, nt]
        ggml_tensor * q = ggml_view_3d(ctx, qfull, D, H, nt, ggml_element_size(qfull) * D * 2, ggml_element_size(qfull) * D * 2 * H, 0);
        q = ggml_mul(ctx, ggml_rms_norm(ctx, q, hp->rms_eps), L.q_norm);
        ggml_tensor * gate = ggml_view_3d(ctx, qfull, D, H, nt, ggml_element_size(qfull) * D * 2, ggml_element_size(qfull) * D * 2 * H,
                                          ggml_element_size(qfull) * D);
        gate = ggml_cont(ctx, gate);
        ggml_tensor * k = ggml_reshape_3d(ctx, MM(ctx, L.wk, x), D, Hkv, nt);
        k = ggml_mul(ctx, ggml_rms_norm(ctx, k, hp->rms_eps), L.k_norm);
        ggml_tensor * v = ggml_reshape_3d(ctx, MM(ctx, L.wv, x), D, Hkv, nt);
        q = rope(g, q, pos4);
        k = rope(g, k, pos4);
        ggml_tensor * kd = ggml_view_3d(ctx, kcache[il], D, Hkv, nt, kcache[il]->nb[1], kcache[il]->nb[2], kcache[il]->nb[2] * t0);
        ggml_tensor * vd = ggml_view_3d(ctx, vcache[il], D, Hkv, nt, vcache[il]->nb[1], vcache[il]->nb[2], vcache[il]->nb[2] * t0);
        ggml_build_forward_expand(g.gf, ggml_cpy(ctx, k, kd));
        ggml_build_forward_expand(g.gf, ggml_cpy(ctx, v, vd));
        ggml_tensor * kc = ggml_view_3d(ctx, kcache[il], D, Hkv, t0 + nt, kcache[il]->nb[1], kcache[il]->nb[2], 0);
        ggml_tensor * vc = ggml_view_3d(ctx, vcache[il], D, Hkv, t0 + nt, vcache[il]->nb[1], vcache[il]->nb[2], 0);
        ggml_tensor * att = attention(ctx, ggml_cont(ctx, q), gate, kc, vc, sel, pos, 1.0f / sqrtf((float) D));
        dbg(g, il, x, "attn_in");
        dbg(g, il, qfull, "Qcur_full");
        dbg(g, il, v, "Vcur");
        dbg(g, il, ggml_sigmoid(ctx, gate), "gate_sigmoid");
        dbg(g, il, att, "attn_gated");
        ggml_tensor * ao = MM(ctx, L.wo, ggml_reshape_2d(ctx, att, D * H, nt));
        dbg(g, il, ao, "attn_output");
        return ao;
    }

    // PLE n-gram embedding block (upstream build_ple), conv history carried in VRAM across tiles
    ggml_tensor * ple_block(gctx & g, uint32_t il, ggml_tensor * hidden, ggml_tensor * emb, bool snapshots = false) {
        ggml_context * ctx = g.ctx;
        const auto & L = layers[il];
        const int64_t E = hp->n_embd, HC = hp->hc, nt = hidden->ne[2], HD = E * HC;
        ggml_tensor * key   = MM(ctx, L.ple_key, emb);
        ggml_tensor * value = MM(ctx, L.ple_value, emb);
        auto gnorm = [&](ggml_tensor * t, ggml_tensor * w) {
            ggml_tensor * r = ggml_reshape_3d(ctx, t, E, HC, nt);
            return ggml_mul(ctx, ggml_rms_norm(ctx, r, hp->rms_eps), ggml_reshape_2d(ctx, w, E, HC));
        };
        key = gnorm(key, L.ple_norm_key);
        ggml_tensor * query = gnorm(hidden, L.ple_norm_query);
        ggml_tensor * s = ggml_sum_rows(ctx, ggml_mul(ctx, key, query));
        s = ggml_scale(ctx, s, 1.0f / sqrtf((float) E));
        ggml_tensor * mag  = ggml_sqrt(ctx, ggml_clamp(ctx, ggml_abs(ctx, s), 1e-6f, INFINITY));
        ggml_tensor * gate = ggml_sigmoid(ctx, ggml_mul(ctx, ggml_sgn(ctx, s), mag));
        ggml_tensor * v3 = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, value, E, 1, nt), E, HC, nt, 1);
        ggml_tensor * gated = ggml_mul(ctx, v3, gate);
        ggml_tensor * normalized = gnorm(ggml_reshape_2d(ctx, gated, HD, nt), L.ple_norm_conv);
        normalized = ggml_reshape_2d(ctx, normalized, HD, nt);
        const int64_t kern = hp->ple_conv, dil = hp->ple_ngram, hist = (kern - 1) * dil;
        ggml_tensor * padded = ggml_concat(ctx, ggml_reshape_3d(ctx, ple_conv, hist, HD, 1),
                                           ggml_transpose(ctx, ggml_reshape_3d(ctx, normalized, HD, nt, 1)), 0);  // [hist+nt, HD, 1]
        ggml_tensor * tail = ggml_view_3d(ctx, padded, hist, HD, 1, padded->nb[1], padded->nb[2], ggml_row_size(padded->type, nt));
        if (snapshots) {
            ggml_build_forward_expand(g.gf, ggml_cpy(ctx, ggml_cont(ctx, padded),
                ggml_view_3d(ctx, ple_hist, padded->ne[0], HD, 1, ple_hist->nb[1], ple_hist->nb[1] * HD, 0)));
        }
        ggml_build_forward_expand(g.gf, ggml_cpy(ctx, ggml_cont(ctx, tail), ggml_reshape_3d(ctx, ple_conv, hist, HD, 1)));
        ggml_tensor * conv = nullptr;
        for (int64_t kk = 0; kk < kern; ++kk) {
            const int64_t start = hist - (kern - 1 - kk) * dil;
            ggml_tensor * shifted = ggml_cont(ctx, ggml_transpose(ctx,
                ggml_view_3d(ctx, padded, nt, HD, 1, padded->nb[1], padded->nb[2], ggml_row_size(padded->type, start))));
            ggml_tensor * wk = ggml_cont(ctx, ggml_view_2d(ctx, L.ple_conv1d, 1, HD, L.ple_conv1d->nb[1], kk * L.ple_conv1d->nb[0]));
            wk = ggml_reshape_1d(ctx, wk, HD);
            if (wk->type != GGML_TYPE_F32) wk = ggml_cast(ctx, wk, GGML_TYPE_F32);
            ggml_tensor * term = ggml_mul(ctx, shifted, wk);
            conv = conv ? ggml_add(ctx, conv, term) : term;
        }
        conv = ggml_silu(ctx, conv);
        conv = ggml_reshape_3d(ctx, ggml_cont(ctx, conv), E, HC, nt);
        return ggml_add(ctx, hidden, ggml_add(ctx, gated, conv));
    }

    // router (upstream build_moe_ffn gating: softmax probabilities, normalised top-k weights). With a residency bias
    // the choice runs on logits + bias * resident (softmax is monotone, so bias 0 would pick the same experts; the
    // probabilities themselves stay the selection input at bias 0, exactly as upstream). Weights always come from
    // the unbiased probabilities.
    struct route { ggml_tensor * selected; ggml_tensor * weights; };
    // Affinity spreads. DSV4.1 scales its penalty by each layer's mean top-1..top-k probability spread, measured
    // at bias zero; its mean spread is 0.6327. QFN_AFFINITY_SPREADS names this model's file, with the same format.
    std::vector<float> affinity_spread;
    std::vector<double> calibrate_sum;
    std::vector<uint32_t> calibrate_n;
    bool affinity_calibrating = false;
    ggml_tensor * calibrate_probs = nullptr;
    float affinity_factor(uint32_t il) {
        if (affinity_spread.empty()) {
            // Measured at bias zero on the exact 100K prompt (presets/qfn-affinity-spreads.txt).
            static const float measured[48] = { 0.0501226f, 0.0330423f, 0.0345657f, 0.0180449f, 0.0197601f, 0.0310545f, 0.0361451f, 0.0335465f, 0.0231881f, 0.0177413f, 0.0178266f, 0.0169741f, 0.0259258f, 0.0236602f, 0.0160583f, 0.0171706f, 0.0128187f, 0.0150019f, 0.0307255f, 0.0164983f, 0.0186828f, 0.032143f, 0.038412f, 0.0242357f, 0.0265036f, 0.0172981f, 0.0173106f, 0.0148644f, 0.022158f, 0.0193617f, 0.0170604f, 0.0133291f, 0.0120101f, 0.0153931f, 0.0241819f, 0.0151216f, 0.012669f, 0.0208197f, 0.0362306f, 0.01562f, 0.0153134f, 0.016045f, 0.017656f, 0.0106593f, 0.0118067f, 0.0118347f, 0.009906f, 0.0204335f };
            affinity_spread.assign(hp->n_layer, 0.0212f);
            if (hp->n_layer == 48) affinity_spread.assign(measured, measured + 48);
            if (const char * path = std::getenv("QFN_AFFINITY_SPREADS"); path && *path) {
                std::ifstream in(path);
                std::string magic, id;
                unsigned long long bytes = 0;
                uint32_t experts = 0, nl = 0;
                in >> magic >> bytes >> experts >> nl >> id;
                require(in && magic == "DS4AFFINITY1" && experts == hp->n_expert && nl == hp->n_layer, "QFN_AFFINITY_SPREADS: wrong format or model");
                for (uint32_t l = 0; l < nl; ++l) {
                    uint32_t index = 0, samples = 0;
                    float spread = 0;
                    in >> index >> spread >> samples;
                    require(in && index == l && samples >= 16 && spread > 0, "QFN_AFFINITY_SPREADS: invalid layer row");
                    affinity_spread[l] = spread;
                }
            }
        }
        return affinity_spread[il] / 0.6327f;
    }
    void calibrate_take(uint32_t il) {
        const int64_t NE = hp->n_expert, K = hp->n_expert_used;
        std::vector<float> row((size_t) NE);
        ggml_backend_tensor_get(calibrate_probs, row.data(), 0, row.size() * sizeof(float));
        std::partial_sort(row.begin(), row.begin() + K, row.end(), std::greater<float>());
        calibrate_sum[il] += row[0] - row[(size_t) K - 1];
        ++calibrate_n[il];
    }
    void calibrate_write() {
        const char * path = std::getenv("QFN_AFFINITY_CALIBRATE");
        std::ofstream out(path);
        out << "DS4AFFINITY1 " << pack.experts[0].blob * hp->n_expert * hp->n_layer << " " << hp->n_expert << " " << hp->n_layer << " qfn\n";
        char line[64];
        for (uint32_t l = 0; l < hp->n_layer; ++l) {
            std::snprintf(line, sizeof(line), "%u %.9g %u\n", l, calibrate_sum[l] / std::max<uint32_t>(1, calibrate_n[l]), calibrate_n[l]);
            out << line;
        }
        require((bool) out, "cannot write QFN_AFFINITY_CALIBRATE");
    }

    route router(gctx & g, uint32_t il, ggml_tensor * x, float bias) {
        ggml_context * ctx = g.ctx;
        const auto & L = layers[il];
        const int64_t nt = x->ne[1], NE = hp->n_expert, K = hp->n_expert_used;
        ggml_tensor * logits = MM(ctx, L.ffn_gate_inp, x);                 // [NE, nt]
        ggml_tensor * probs = ggml_soft_max(ctx, logits);
        ggml_tensor * choose = probs;
        if (bias > 0.0f) {
            ggml_tensor * r = verify_active
                ? ggml_view_2d(ctx, verify_resident, NE, nt, verify_resident->nb[1], (size_t) il * verify_resident->nb[2])
                : ggml_view_1d(ctx, resident, NE, (size_t) il * NE * sizeof(float));
            // DSV4.1 cache affinity: the penalty applies to router probabilities, per-layer calibrated.
            // QFN_AFFINITY_LOGIT=1 keeps the earlier logit-unit bias.
            static const bool logit_units = [] { const char * e = std::getenv("QFN_AFFINITY_LOGIT"); return e && *e == '1'; }();
            choose = logit_units ? ggml_add(ctx, logits, ggml_scale(ctx, r, bias))
                                 : ggml_add(ctx, probs, ggml_scale(ctx, r, bias * affinity_factor(il)));
        }
        if (affinity_calibrating) {
            ggml_set_output(probs);
            ggml_build_forward_expand(g.gf, probs);
            calibrate_probs = probs;
        }
        ggml_tensor * selected = ggml_cont(ctx, ggml_argsort_top_k(ctx, choose, (int) K));   // contiguous [K, nt]: read back and mapped by row           // [K, nt]
        ggml_tensor * weights = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, NE, nt), selected);  // [1, K, nt]
        weights = ggml_reshape_2d(ctx, weights, K, nt);
        ggml_tensor * wsum = ggml_clamp(ctx, ggml_sum_rows(ctx, weights), 6.103515625e-5f, INFINITY);
        weights = ggml_reshape_3d(ctx, ggml_div(ctx, weights, wsum), 1, K, nt);
        return { selected, weights };
    }

    // routed experts over any expert array (the streamed bank in prefill, the cache slab in decode, indexed by
    // slot) + the gated shared expert
    ggml_tensor * experts(gctx & g, uint32_t il, ggml_tensor * x, ggml_tensor * ids, ggml_tensor * weights,
                          ggml_tensor * gate_t, ggml_tensor * up_t, ggml_tensor * down_t, bool capture_rows = false,
                          ggml_tensor * logical_ids = nullptr) {
        ggml_context * ctx = g.ctx;
        const auto & L = layers[il];
        const int64_t E = hp->n_embd, nt = x->ne[1], K = hp->n_expert_used;
        ggml_tensor * ex = nullptr;
        if (gate_t->type == GGML_TYPE_I8) {
            require(nvfp4 && nvfp4_scratch && !up_t && !down_t,
                    "QFN_NVFP4_NODE: raw expert view requires the admitted bridge and persistent scratch");
            const bool decode_slots = gate_t == slab_gate;
            ggml_tensor * work = decode_slots ? nvfp4_scratch : nvfp4_prefill_scratch;
            require(work != nullptr, "QFN_NVFP4_NODE: prefill phase workspace was not admitted");
            const int mode = decode_slots ? opt.nvfp4_decode_mode : opt.nvfp4_mode;
            require(!(mode==5 || (mode==6 && decode_slots)) || (logical_ids && nt>=1 && nt<=DEC_NMAX),
                    "QFN_NVFP4_CALIBRATED_DECODE: original logical selection ids are required");
            ggml_tensor * input_scale = nullptr;
            if (mode == 4 || mode == 5 || mode == 6) {
                require(nvfp4_input_scale && il < hp->n_layer, "QFN_NVFP4_CALIBRATION: stable layer scalars unavailable");
                input_scale = ggml_view_2d(ctx, nvfp4_input_scale, 3, hp->n_expert,
                    nvfp4_input_scale->nb[1], size_t(il) * nvfp4_input_scale->nb[2]);
            }
            if(reference_prefill_active&&!decode_slots) {
                require(mode==6&&pack.native_nvfp4_packed,"Reference prefill requires the actual packed vector arithmetic");
                for(int64_t first=0;first<nt;first+=4) {
                    const int64_t count=std::min<int64_t>(4,nt-first);
                    auto *part_x=ggml_view_2d(ctx,x,E,count,x->nb[1],size_t(first)*x->nb[1]);
                    auto *part_ids=ggml_view_2d(ctx,ids,K,count,ids->nb[1],size_t(first)*ids->nb[1]);
                    auto *part=ggml_qfn_nvfp4_experts(ctx,part_x,part_ids,gate_t,work,mode,input_scale,nullptr,true);
                    ex=ex?ggml_concat(ctx,ex,part,2):part;
                }
            } else ex = ggml_qfn_nvfp4_experts(ctx, x, ids, gate_t, work, mode, input_scale,
                (mode==5 || (mode==6 && decode_slots)) ? logical_ids : nullptr,pack.native_nvfp4_packed);
        } else {
            ggml_tensor * cur = ggml_reshape_3d(ctx, x, E, 1, nt);
            ggml_tensor * up   = MMID(ctx, up_t, cur, ids, capture_rows);                // [n_ff, K, nt]
            ggml_tensor * gt   = MMID(ctx, gate_t, cur, ids, capture_rows);
            ggml_tensor * act  = ggml_swiglu_split(ctx, gt, up);
            ex = MMID(ctx, down_t, act, ids, capture_rows);                              // [E, K, nt]
        }
        ggml_tensor * moe;
        if (fused_expert_reduce) {
            moe = ggml_qfn_expert_reduce(ctx, ex, weights);
        } else {
            ex = ggml_mul(ctx, ex, weights);
            moe = ggml_view_2d(ctx, ex, E, nt, ex->nb[2], 0);
            for (int64_t i = 1; i < K; ++i) moe = ggml_add(ctx, moe, ggml_view_2d(ctx, ex, E, nt, ex->nb[2], i * ex->nb[1]));
        }
        ggml_tensor * su = MM(ctx, L.ffn_up_shexp, x);
        ggml_tensor * sg = MM(ctx, L.ffn_gate_shexp, x);
        ggml_tensor * sh = MM(ctx, L.ffn_down_shexp, ggml_swiglu_split(ctx, sg, su));
        ggml_tensor * shared_gate = ggml_sigmoid(ctx, MM(ctx, L.ffn_gate_inp_shexp, x));  // [1, nt]
        sh = ggml_mul(ctx, sh, shared_gate);
        return ggml_add(ctx, moe, sh);
    }

    ggml_tensor * ffn(gctx & g, uint32_t il, ggml_tensor * x, int bank) {
        route r = router(g, il, x, 0.0f);
        last_selected = r.selected;
        ggml_set_output(r.selected);
        return experts(g, il, x, r.selected, r.weights, gate_exps[bank], up_exps[bank], down_exps[bank]);
    }

    // ------------------------------------------------------------------------------------------ execution
    ggml_context * new_graph_ctx() {
        release_reference_weights();
        const size_t nodes=graph_node_capacity();
        ggml_init_params ip = { (reference_prefill_active?nodes*2:4096) * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true };
        return ggml_init(ip);
    }

    size_t graph_high_water = 0;
    void compute(ggml_cgraph * gf) {
        if (phase_profile) { ++phase_graphs; phase_nodes += ggml_graph_n_nodes(gf); }
        require(ggml_gallocr_alloc_graph(galloc, gf), "graph allocation failed (VRAM)");
        const size_t bytes = ggml_gallocr_get_buffer_size(galloc, 0);
        if (bytes > graph_high_water) {
            graph_high_water = bytes;
            log("[qfn] graph allocator high-water: %zu bytes (%.2f MiB)\n", bytes, bytes / 1048576.0);
        }
    }

    double pf_build_ms = 0, pf_stage_ms = 0, pf_gpu_ms = 0, pf_read_ms = 0;   // prefill time ledger
    // debug: intermediates of one layer, printed after its tile graph (QFN_DUMP_LAYER)
    int dump_layer = -2;
    std::vector<std::pair<std::string, ggml_tensor *>> dbg_list;
    void dbg(gctx & g, uint32_t il, ggml_tensor * t, const char * name) {
        if (dump_layer == -2) { const char * d = std::getenv("QFN_DUMP_LAYER"); dump_layer = d && *d ? std::atoi(d) : -1; }
        if ((int) il != dump_layer || t->type != GGML_TYPE_F32) return;
        ggml_tensor * c = ggml_cont(g.ctx, t);
        ggml_set_output(c);
        ggml_build_forward_expand(g.gf, c);
        dbg_list.push_back({ std::string(name) + "-" + std::to_string(il), c });
    }
    void dbg_print(int64_t nt) {
        for (auto & d : dbg_list) {
            ggml_tensor * t = d.second;
            const int64_t n = ggml_nelements(t);
            std::vector<float> v((size_t) n);
            ggml_backend_tensor_get(t, v.data(), 0, (size_t) n * 4);
            const int64_t per = n / nt;
            for (int64_t k = 0; k < std::min<int64_t>(nt, 4); ++k) {
                double s = 0, l2 = 0;
                const float * r = v.data() + k * per;
                for (int64_t i = 0; i < per; ++i) { s += r[i]; l2 += (double) r[i] * r[i]; }
                std::fprintf(stderr, "DUMP %s tok %lld n %lld sum %.6g l2 %.6g v %.6g %.6g %.6g %.6g\n", d.first.c_str(), (long long) k,
                             (long long) per, s, std::sqrt(l2), r[0], per > 1 ? r[1] : 0.0f, per > 2 ? r[2] : 0.0f, per > 3 ? r[3] : 0.0f);
            }
        }
        dbg_list.clear();
    }
    int64_t pf_keep_from = 0;  // the first prompt row whose final-layer output is read; rows before it stop at the caches
    void run_layer_tile(uint32_t il, int bank, int64_t t0, int64_t nt, const std::vector<int32_t> & tok,
                        const std::vector<int32_t> & ple_slot, int64_t p0) {
        const double tb = now_ms();
        ggml_context * ctx = new_graph_ctx();
        gctx g{ ctx, ggml_new_graph_custom(ctx, graph_node_capacity(), false) };
        const int64_t E = hp->n_embd, HC = hp->hc;
        const bool caches_only = il + 1 == hp->n_layer && t0 + nt <= pf_keep_from && !hp->is_recurrent(il) && !moe_pass;
        ggml_tensor * pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nt);
        ggml_tensor * pos4 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * nt);
        ggml_set_input(pos); ggml_set_input(pos4);
        ggml_tensor * emb = nullptr, * ple_idx = nullptr, * pool_pos4 = nullptr;
        ggml_tensor * carry_tile = ggml_view_2d(ctx, carry, E * HC, nt, carry->nb[1], carry->nb[1] * t0);
        ggml_tensor * res;
        if (il == 0) {
            emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, nt);
            ggml_set_input(emb);
            res = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, emb, E, 1, nt), E, HC, nt, 1);
        } else {
            res = ggml_reshape_3d(ctx, carry_bf16 ? ggml_cast(ctx, carry_tile, GGML_TYPE_F32) : carry_tile, E, HC, nt);
        }
        if (hp->is_ple(il)) {
            ple_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nt * hp->ple_n_heads);
            ggml_set_input(ple_idx);
            ggml_tensor * pe = ggml_get_rows(ctx, ple_rows, ple_idx);                 // [ple_dim, nt*heads]
            pe = ggml_reshape_2d(ctx, pe, (int64_t) hp->ple_dim * hp->ple_n_heads, nt);
            res = ple_block(g, il, res, pe);
        }
        ggml_tensor * inj = nullptr;
        ggml_tensor * cur = hc_mix(g, res, layers[il].hc_attn_norm, layers[il].hc_attn_down, layers[il].hc_attn_up,
                                   layers[il].hc_attn_inject, &inj);
        if (hp->is_recurrent(il)) {
            cur = gdn_layer(g, il, cur);
        } else {
            const int64_t R = hp->kpool, b_lo = t0 / R, b_hi = (t0 + nt) / R;
            if (b_hi > b_lo) {
                pool_pos4 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * (b_hi - b_lo));
                ggml_set_input(pool_pos4);
            }
            cur = qsa_layer(g, il, cur, pos, pos4, pool_pos4, t0, nt, caches_only);
        }
        if (!caches_only) {
        res = hc_combine(g, res, cur, inj);
        dbg(g, il, res, "attn_res");
        cur = hc_mix(g, res, layers[il].hc_ffn_norm, layers[il].hc_ffn_down, layers[il].hc_ffn_up, layers[il].hc_ffn_inject, &inj);
        dbg(g, il, cur, "ffn_in");
        if (moe_pass) {
            const int64_t off = t0 - pf_first;
            auto tile2 = [&](ggml_tensor * t) { return ggml_view_2d(ctx, t, t->ne[0], nt, t->nb[1], off * t->nb[1]); };
            route r = router(g, il, cur, 0.0f);
            last_selected = r.selected;
            ggml_set_output(r.selected);
            ggml_build_forward_expand(g.gf, ggml_cpy(ctx, cur, tile2(pf_x)));
            ggml_build_forward_expand(g.gf, ggml_cpy(ctx, r.selected, tile2(pf_ids)));
            ggml_build_forward_expand(g.gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, r.weights, hp->n_expert_used, nt), tile2(pf_w)));
            ggml_build_forward_expand(g.gf, ggml_cpy(ctx, inj, tile2(pf_inj)));
            // F32 carry can hold attention residual directly. BF16 carry needs
            // separate F32 storage to avoid an extra rounding before hc_combine.
            ggml_tensor * dst = pf_res ? tile2(pf_res) : carry_tile;
            ggml_build_forward_expand(g.gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, res, E * HC, nt), dst));
        } else {
            cur = ffn(g, il, cur, bank);
            dbg(g, il, cur, "ffn_out");
            res = hc_combine(g, res, cur, inj);
            ggml_build_forward_expand(g.gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, res, E * HC, nt), carry_tile));
        }
        }
        compute(g.gf);
        const double ts = now_ms();
        pf_build_ms += ts - tb;
        // inputs
        std::vector<int32_t> pv(nt), p4(4 * nt, 0);
        for (int64_t i = 0; i < nt; ++i) { pv[i] = (int32_t) (t0 + i); p4[i] = p4[nt + i] = p4[2 * nt + i] = (int32_t) (t0 + i); }
        set_input(pos, pv.data(), 0, pv.size() * 4);
        if(!images.empty())image_positions(p4,nt,t0);
                set_input(pos4, p4.data(), 0, p4.size() * 4);
        if (emb) {
            const double embedding_start = phase_profile ? now_ms() : 0;
            std::vector<float> e((size_t) (E * nt));
            const auto * traits = ggml_get_type_traits(tok_embd_type);
            for (int64_t i = 0; i < nt; ++i) {
                image_embedding(tok[t0+i],t0+i,e.data()+i*E);
            }
            set_input(emb, e.data(), 0, e.size() * 4);
            if (phase_profile) phase_embedding_ms += now_ms() - embedding_start;
        }
        if (ple_idx) {
            set_input(ple_idx, ple_slot.data() + (t0 - p0) * hp->ple_n_heads, 0, (size_t) nt * hp->ple_n_heads * 4);
        }
        if (pool_pos4) {
            const int64_t R = hp->kpool, b_lo = t0 / R, nn = pool_pos4->ne[0] / 4;
            std::vector<int32_t> pp(4 * nn, 0);
            for (int64_t i = 0; i < nn; ++i) pp[i] = pp[nn + i] = pp[2 * nn + i] = (int32_t) ((b_lo + i) * R);
            if(!images.empty())image_positions(pp,nn,b_lo*R,R);
                set_input(pool_pos4, pp.data(), 0, pp.size() * 4);
        }
        const double tg = now_ms();
        pf_stage_ms += tg - ts;
        require(ggml_backend_graph_compute(gpu, g.gf) == GGML_STATUS_SUCCESS, "layer graph compute failed");
        const double tr = now_ms();
        pf_gpu_ms += tr - tg;
        if (last_selected && !prompt_hist.empty()) {
            const int64_t K = hp->n_expert_used;
            std::vector<int32_t> sel((size_t) (K * nt));
            ggml_backend_tensor_get(last_selected, sel.data(), 0, sel.size() * 4);
            uint32_t * h = prompt_hist.data() + (size_t) il * hp->n_expert;
            for (int32_t e : sel) if (e >= 0 && (uint32_t) e < hp->n_expert) ++h[e];
        }
        last_selected = nullptr;
        if (!dbg_list.empty()) dbg_print(nt);
        ggml_free(ctx);
        pf_read_ms += now_ms() - tr;
    }

    void alloc_moe_pass(int64_t first, int64_t count) {
        pf_first = first;
        ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
        pfctx = ggml_init(ip);
        pf_x = ggml_new_tensor_2d(pfctx, GGML_TYPE_F32, hp->n_embd, count);
        pf_ids = ggml_new_tensor_2d(pfctx, GGML_TYPE_I32, hp->n_expert_used, count);
        pf_w = ggml_new_tensor_2d(pfctx, GGML_TYPE_F32, hp->n_expert_used, count);
        pf_inj = ggml_new_tensor_2d(pfctx, GGML_TYPE_F32, hp->hc, count);
        if (carry_bf16) pf_res = ggml_new_tensor_2d(pfctx, GGML_TYPE_F32, (int64_t) hp->n_embd * hp->hc, count);
        pfbuf = ggml_backend_alloc_ctx_tensors(pfctx, gpu);
        require(pfbuf != nullptr, "cannot allocate whole-prompt MoE staging (VRAM)");
        log("[qfn] separate MoE staging: %zu bytes, %lld tokens, F32 arithmetic\n",
            ggml_backend_buffer_get_size(pfbuf), (long long) count);
    }

    void free_moe_pass() {
        ggml_backend_synchronize(gpu);
        if (pfbuf) ggml_backend_buffer_free(pfbuf);
        if (pfctx) ggml_free(pfctx);
        pfbuf = nullptr; pfctx = nullptr;
        pf_x = pf_ids = pf_w = pf_inj = pf_res = nullptr;
    }

    void build_moe_pass(gctx & g, uint32_t il, int bank, int64_t first, int64_t count, bool debug) {
        ggml_context * ctx = g.ctx;
        const int64_t E = hp->n_embd, HC = hp->hc, K = hp->n_expert_used;
        const int64_t off = first - pf_first;
        auto tile2 = [&](ggml_tensor * t) { return ggml_view_2d(ctx, t, t->ne[0], count, t->nb[1], off * t->nb[1]); };
        ggml_tensor * x = tile2(pf_x), * ids = tile2(pf_ids);
        ggml_tensor * weights = ggml_reshape_3d(ctx, tile2(pf_w), 1, K, count);
        ggml_tensor * out = experts(g, il, x, ids, weights, gate_exps[bank], up_exps[bank], down_exps[bank]);
        if (debug) dbg(g, il, out, "ffn_out");
        ggml_tensor * dst = ggml_view_2d(ctx, carry, E * HC, count, carry->nb[1], first * carry->nb[1]);
        ggml_tensor * res = ggml_reshape_3d(ctx, pf_res ? tile2(pf_res) : dst, E, HC, count);
        out = hc_combine(g, res, out, tile2(pf_inj));
        ggml_build_forward_expand(g.gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, out, E * HC, count), dst));
    }

    size_t moe_graph_bytes(int64_t count) {
        ggml_context * ctx = new_graph_ctx();
        gctx g{ctx, ggml_new_graph_custom(ctx, graph_node_capacity(), false)};
        build_moe_pass(g, 0, 0, pf_first, count, false);
        ggml_gallocr_t probe = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
        size_t bytes = 0;
        ggml_gallocr_reserve_n_size(probe, g.gf, nullptr, nullptr, &bytes);
        ggml_gallocr_free(probe);
        ggml_free(ctx);
        require(bytes > 0, "MoE size-only allocation plan is empty");
        return bytes;
    }

    void run_moe_pass(uint32_t il, int bank, int64_t first, int64_t count) {
        const double tb = now_ms();
        ggml_context * ctx = new_graph_ctx();
        gctx g{ctx, ggml_new_graph_custom(ctx, graph_node_capacity(), false)};
        build_moe_pass(g, il, bank, first, count, true);
        compute(g.gf);
        const double tg = now_ms();
        pf_build_ms += tg - tb;
        require(ggml_backend_graph_compute(gpu, g.gf) == GGML_STATUS_SUCCESS, "separate MoE graph compute failed");
        pf_gpu_ms += now_ms() - tg;
        if (!dbg_list.empty()) dbg_print(count);
        ggml_free(ctx);
    }

    // output head over the final hidden streams of the given positions
    void run_head(const std::vector<int64_t> & rows, std::vector<float> & logits) {
        ggml_context * ctx = new_graph_ctx();
        gctx g{ ctx, ggml_new_graph_custom(ctx, graph_node_capacity(), false) };
        const int64_t E = hp->n_embd, HC = hp->hc, m = (int64_t) rows.size();
        ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, m);
        ggml_set_input(idx);
        ggml_tensor * h = ggml_get_rows(ctx, carry, idx);                                  // [E*HC, m]
        h = ggml_reshape_3d(ctx, h, E, HC, m);
        ggml_tensor * cur = hc_mix(g, h, out_hc_norm, out_hc_down, out_hc_up, nullptr, nullptr);
        ggml_tensor * out = reference_matmul(ctx, output, cur);                                // [n_vocab, m]
        ggml_set_output(out);
        ggml_build_forward_expand(g.gf, out);
        compute(g.gf);
        std::vector<int32_t> iv(rows.begin(), rows.end());
        ggml_backend_tensor_set(idx, iv.data(), 0, iv.size() * 4);
        require(ggml_backend_graph_compute(gpu, g.gf) == GGML_STATUS_SUCCESS, "head graph compute failed");
        logits.resize((size_t) ggml_nelements(out));
        ggml_backend_tensor_get(out, logits.data(), 0, logits.size() * 4);
        ggml_free(ctx);
    }

    // ------------------------------------------------------------------------------------------ expert cache
    cudaStream_t stream() const { return (cudaStream_t) ggml_backend_cuda_get_stream(gpu); }

    void alloc_cache(bool restoring = false) {
        const uint64_t blob = pack.experts[0].blob;
        const uint32_t NE = hp->n_expert, NL = hp->n_layer;
        size_t free_b = 0, total_b = 0;
        require(cudaMemGetInfo(&free_b, &total_b) == cudaSuccess, "cannot query VRAM");
        const char * gr = std::getenv("QFN_GRAPH_RESERVE_MIB");
        const uint64_t graph_reserve = (gr && *gr ? (uint64_t) std::atoll(gr) : 3072ull) << 20;
        require(!opt.reference_prefill_max_tokens||graph_reserve>=uint64_t(hp->n_vocab)*hp->n_embd*2+(256ull<<20),
                "Reference prefill requires its exact F16 coefficient head plus256MiB in the explicit graph reserve");
        uint64_t avail = free_b > graph_reserve + opt.vram_margin + (64u << 10) ? free_b - graph_reserve - opt.vram_margin - (64u << 10) : 0;
        const char * rc = std::getenv("QFN_PREFILL_RELEASE_CACHE");
        // Without MTP the carry lives in the slab's tail: the same memory serves prefill and then decode, so nothing
        // is freed or reloaded at the handoff and the prefill seeds the hot experts from its banks.
        // Explicit slot counts select the same slab layout as automatic sizing; they only fix its capacity.
        const bool tail_carry = !restoring && !mtp_ok && !(rc && *rc == '1');
        if ((restoring || tail_carry) && !opt.cache_slots) {
            // As DSV4.1's decode cache offer: CUDA's free counts the imported D3D12 banks twice, so the cap is NVML's
            // physical free memory minus a small reserve and the margin for prefill and decode graphs.
            uint64_t nv_free = 0;
            require(nvml_free_bytes(nv_free), "the decode cache sizing needs NVML physical free VRAM");
            const char * rs = std::getenv("QFN_VRAM_RESERVE_MIB");
            const char * mg = std::getenv("QFN_VRAM_MARGIN_MIB");
            const uint64_t keep = ((rs && *rs ? (uint64_t) std::atoll(rs) : 128ull) + (mg && *mg ? (uint64_t) std::atoll(mg) : (restoring ? 512ull : 1100ull))) << 20;
            avail = nv_free > keep + (64u << 10) ? nv_free - keep - (64u << 10) : 0;
            log("[qfn] decode handoff VRAM: NVML free %.2f GiB (CUDA free %.2f GiB), kept %.2f GiB\n",
                nv_free / 1073741824.0, free_b / 1073741824.0, keep / 1073741824.0);
        }
        if (!restoring || !opt.cache_slots) n_slots = opt.cache_slots ? opt.cache_slots : (uint32_t) std::min<uint64_t>(avail / blob, (uint64_t) NL * NE);
        seed_slots = n_slots;
        require(n_slots >= 4 * hp->n_expert_used, "not enough free VRAM for a decode expert cache (lower --ctx)");
        // a refused allocation (another process took VRAM since the query) steps down 256 MiB at a time (DSV4.1)
        ggml_tensor * raw = nullptr;
        for (;;) {
            ggml_init_params ip = { 16 * ggml_tensor_overhead(), nullptr, true };
            cctx = ggml_init(ip);
            raw = ggml_new_tensor_1d(cctx, GGML_TYPE_I8, (int64_t) (n_slots * blob + (64u << 10)));
            resident = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, NE, NL);
            cbuf = ggml_backend_alloc_ctx_tensors(cctx, gpu);
            if (cbuf != nullptr) break;
            ggml_free(cctx);
            cctx = nullptr;
            const uint32_t step = (uint32_t) std::max<uint64_t>(1, (256ull << 20) / blob);
            require(!opt.cache_slots && n_slots > step + 4 * hp->n_expert_used, "cannot allocate the decode expert cache at the required capacity");
            n_slots -= step;
            log("[qfn] decode expert cache allocation refused; stepping down to %u slots\n", n_slots);
        }
        ggml_backend_buffer_set_usage(cbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        slab = raw->data;
        require(cudaMemsetAsync((char *) slab + n_slots * blob, 0, 64u << 10, stream()) == cudaSuccess, "cache tail clear failed");
        if (tail_carry) {
            carry = ggml_new_tensor_2d(cctx, carry_bf16 ? GGML_TYPE_BF16 : GGML_TYPE_F32, (int64_t) hp->n_embd * hp->hc, (int64_t) opt.n_ctx);
            const uint32_t tail = (uint32_t) ((ggml_nbytes(carry) + blob - 1) / blob);
            require(n_slots > tail + 4 * hp->n_expert_used, "not enough free VRAM for the prefill carry and a decode expert cache (lower --ctx)");
            seed_slots = n_slots - tail;
            require(ggml_backend_tensor_alloc(cbuf, carry, (char *) slab + (size_t) seed_slots * blob) == GGML_STATUS_SUCCESS, "cannot bind the carry to the slab tail");
            carry_in_slab = true;
            log("[qfn] prefill carry in the last %u of %u cache slots; prefill seeds %u\n", tail, n_slots, seed_slots);
        }
        const auto & e0 = pack.experts[0];
        auto mk = [&](ggml_type type, int64_t ne0, int64_t ne1, uint64_t off) {
            ggml_tensor * t = ggml_new_tensor_3d(cctx, type, ne0, ne1, n_slots);
            t->nb[1] = ggml_row_size(type, ne0);
            t->nb[2] = blob;
            t->nb[3] = blob * n_slots;
            require(ggml_backend_tensor_alloc(cbuf, t, (char *) slab + off) == GGML_STATUS_SUCCESS, "cannot bind the cache slab");
            return t;
        };
        if (nvfp4) {
            slab_gate = ggml_new_tensor_2d(cctx, GGML_TYPE_I8, int64_t(blob), n_slots);
            require(ggml_backend_tensor_alloc(cbuf, slab_gate, slab) == GGML_STATUS_SUCCESS,
                    "cannot bind complete NVFP4 physical cache slots including scale tails");
            // Record-only v3 maps unresolved IDs to physical slot zero and discards that attempt.
            // Seed real bytes even for a cold/empty cache so the doomed graph never reads undefined tails.
            // Metadata stays nonresident; this is not an accepted cache hit or a substitute expert result.
            wf_extent first{};
            first.file_offset = pack.experts[0].offset;
            first.bank_offset = 0; first.nbytes = blob; first.file_idx = 0;
            uint64_t read_bytes = 0; double read_ms = 0;
            require(wf_fetch_extents_direct(wf, 0, &first, 1, stream(), &read_bytes, &read_ms) && !wf_has_failed(wf),
                    "QFN_NVFP4_SLOT_ZERO: cannot initialize record-only miss placeholder with actual expert bytes");
            require(cudaMemcpyAsync(slab, bank_ptr[0], blob, cudaMemcpyDeviceToDevice, stream()) == cudaSuccess &&
                    cudaStreamSynchronize(stream()) == cudaSuccess, "NVFP4 slot-zero initialization failed");
            log("[qfn] NVFP4 record-only slot0: actual layer0/expert0 blob including tails, %llu read bytes, %.3f ms; "
                "not resident, included in load/restore TTFT\n", (unsigned long long) read_bytes, read_ms);
            nvfp4_placeholder_read_bytes = read_bytes;
        } else {
            slab_gate = mk(e0.gu_type, hp->n_embd, hp->n_ff_exp, 0);
            slab_up   = mk(e0.gu_type, hp->n_embd, hp->n_ff_exp, e0.gate_bytes);
            slab_down = mk(e0.d_type, hp->n_ff_exp, hp->n_embd, 2 * e0.gate_bytes);
        }
        slot_of.assign((size_t) NL * NE, -1);
        owner.assign(n_slots, -1);
        lfu.assign(n_slots, 0.0);
        lfu_step.assign(n_slots, 0);
        resident_host.assign((size_t) NL * NE, 0.0f);
        resident_dirty = true;
        if (!restoring) prompt_hist.assign((size_t) NL * NE, 0);
        if (!restoring) {
        for (auto & e : staging_free) require(cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess, "event create failed");
        // decode carry between the router half and the expert half of a layer
        ggml_init_params dp = { (16 + 2 * (size_t) hp->n_layer) * ggml_tensor_overhead(), nullptr, true };
        dctx = ggml_init(dp);
        dec_res  = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, (int64_t) hp->n_embd * hp->hc, DEC_NMAX);
        dec_hout = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, (int64_t) hp->n_embd * hp->hc, DEC_NMAX);
        dec_x   = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, hp->n_embd, DEC_NMAX);
        dec_inj = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, hp->hc, DEC_NMAX);
        dec_w   = ggml_new_tensor_3d(dctx, GGML_TYPE_F32, 1, hp->n_expert_used, DEC_NMAX);
        dec_sel = ggml_new_tensor_2d(dctx, GGML_TYPE_I32, hp->n_expert_used, DEC_NMAX);
        {
            // rollback windows: each recurrent layer's conv input of the last multi-token step, the n-gram conv input
            const int64_t conv_ch = (int64_t) hp->ssm_state * hp->ssm_group * 2 + (int64_t) hp->ssm_state * hp->ssm_dt_rank;
            conv_hist.assign(hp->n_layer, nullptr);
            for (uint32_t il = 0; il < hp->n_layer; ++il) {
                if (hp->is_recurrent(il)) conv_hist[il] = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, hp->ssm_conv - 1 + DEC_NMAX, conv_ch);
            }
            ple_hist = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, (int64_t) (hp->ple_conv - 1) * hp->ple_ngram + DEC_NMAX,
                                          (int64_t) hp->n_embd * hp->hc);
        }
        dbuf = ggml_backend_alloc_ctx_tensors(dctx, gpu);
        require(dbuf != nullptr, "cannot allocate the decode carry");
        }
        log("[qfn] decode expert cache: %u slots of %.2f MiB = %.2f GiB (%.1f%% of the %u experts), %.2f GiB VRAM left for graphs\n",
            n_slots, blob / 1048576.0, n_slots * (double) blob / 1073741824.0, 100.0 * n_slots / ((double) NL * NE), NL * NE,
            (free_b - n_slots * (double) blob) / 1073741824.0);
    }

    // DSV4.1 phase edge: synchronize, retire cached decode graphs and slab;
    // recreate the same capacity only after the prefill allocator is released.
    void release_cache_for_prefill() {
        v3_shutdown();
        ggml_backend_synchronize(gpu);
        if (side) require(cudaStreamSynchronize(side) == cudaSuccess, "cache release side-stream sync failed");
        if (replay_miss_latch) require(cudaFree(replay_miss_latch) == cudaSuccess, "cache release replay-latch free failed");
        replay_miss_latch = nullptr;
        size_t free_before = 0, total_before = 0;
        require(cudaMemGetInfo(&free_before, &total_before) == cudaSuccess, "prefill edge VRAM query failed");
        const size_t released_bytes = cbuf ? ggml_backend_buffer_get_size(cbuf) : 0;
        if (carry_in_slab) { carry = nullptr; carry_in_slab = false; }
        for (auto * G : v3g) { destroy_future_parent(*G); if (G->ga) ggml_gallocr_free(G->ga); if (G->ctx) ggml_free(G->ctx); delete G; }
        v3g.clear();
        if (v3buf) ggml_backend_buffer_free(v3buf);
        if (v3ctx) ggml_free(v3ctx);
        v3buf = nullptr; v3ctx = nullptr; slot_tab = sel_log = nullptr;
        if (bell_completion) cudaEventDestroy(bell_completion);
        if (side) cudaStreamDestroy(side);
        if (bell_h) cudaFreeHost((void *) bell_h);
        side = nullptr; bell_completion = nullptr; bell_h = nullptr; bell_d = nullptr; bell_err.clear();
        if (cbuf) ggml_backend_buffer_free(cbuf);
        if (cctx) ggml_free(cctx);
        cbuf = nullptr; cctx = nullptr; slab = nullptr;
        slab_gate = slab_up = slab_down = resident = nullptr;
        size_t free_b = 0, total_b = 0;
        require(cudaMemGetInfo(&free_b, &total_b) == cudaSuccess, "cache release VRAM query failed");
        log("[qfn] prefill edge released decode cache: %zu bytes, capacity %u preserved; free %.2f -> %.2f GiB\n",
            released_bytes, n_slots, free_before / 1073741824.0, free_b / 1073741824.0);
    }

    void restore_cache_after_prefill() {
        const double t0 = now_ms();
        // Oversized prefill graphs must not survive the phase edge (DSV4.1).
        ggml_backend_synchronize(gpu);
        ggml_gallocr_free(galloc);
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
        graph_high_water = 0;
        if (!mtp_ok) free_carry();
        alloc_cache(true);
        uint64_t bytes = nvfp4_placeholder_read_bytes;
        for (uint32_t il = 0; il < hp->n_layer; ++il) bytes += seed_layer(il, il & 1, true);
        ggml_backend_synchronize(gpu);
        log("[qfn] decode cache reload/reseed: %.3f ms, %llu bytes from experts pack through DirectStorage banks, %u slots; counts toward TTFT\n",
            now_ms() - t0, (unsigned long long) bytes, n_slots);
    }

    void reset_cache() {
        std::fill(slot_of.begin(), slot_of.end(), -1);
        std::fill(owner.begin(), owner.end(), -1);
        std::fill(lfu.begin(), lfu.end(), 0.0);
        std::fill(lfu_step.begin(), lfu_step.end(), 0);
        std::fill(resident_host.begin(), resident_host.end(), 0.0f);
        std::fill(prompt_hist.begin(), prompt_hist.end(), 0);
        resident_dirty = true;
        slot_tab_dirty = true;
        step = 0;
    }

    // After a layer's prefill tiles: copy its most-routed experts (prompt counts, then the pack profile) from the
    // streamed bank into a fair share of the slab. D2D on the compute stream, before the bank is released.
    uint64_t seed_layer(uint32_t il, int bank, bool reload = false) {
        const uint32_t NE = hp->n_expert, NL = hp->n_layer;
        const uint64_t blob = pack.experts[0].blob;
        const uint32_t quota = seed_slots / NL + (il < seed_slots % NL ? 1u : 0u);
        uint32_t first = 0;
        for (uint32_t l = 0; l < il; ++l) first += seed_slots / NL + (l < seed_slots % NL ? 1u : 0u);
        const uint32_t * h = prompt_hist.data() + (size_t) il * NE;
        std::vector<uint32_t> order(NE);
        for (uint32_t e = 0; e < NE; ++e) order[e] = e;
        auto prank = [&](uint32_t e) {
            const int32_t r = pack.profile_rank.empty() ? -1 : pack.profile_rank[(size_t) il * NE + e];
            return r < 0 ? INT32_MAX : r;
        };
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            return h[a] != h[b] ? h[a] > h[b] : prank(a) < prank(b); });
        // The seed may fill only part of the slab: each seeded byte is a copy on the compute stream; misses fill the rest.
        static const uint32_t reload_pct = [] { const char * e = std::getenv("QFN_CACHE_RELOAD_PCT"); return e && *e ? (uint32_t) std::atoi(e) : 100u; }();
        const uint32_t count = std::min((uint32_t) ((uint64_t) quota * reload_pct / 100), NE);
        uint64_t bytes = 0;
        if (reload && count) {
            // Reuse QFN's bounded direct-extent loader and DSV4.1's prompt
            // selection -> pack -> cache phase lifecycle. No host weight copy.
            std::vector<wf_extent> ext;
            for (uint32_t i = 0; i < count; ++i) {
                wf_extent e{};
                e.file_offset = pack.experts[il].offset + (uint64_t) order[i] * blob;
                e.bank_offset = (uint64_t) i * blob; e.nbytes = blob; e.file_idx = 0;
                ext.push_back(e);
            }
            require(cudaStreamSynchronize(stream()) == cudaSuccess, "reseed bank reuse sync failed");
            double ms = 0;
            require(wf_fetch_extents_direct(wf, bank, ext.data(), (int) ext.size(), stream(), &bytes, &ms) && !wf_has_failed(wf),
                "decode cache reseed DirectStorage read failed");
        }
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t e = order[i], slot = first + i;
            require(cudaMemcpyAsync((char *) slab + slot * blob, (const char *) bank_ptr[bank] + (uint64_t) (reload ? i : e) * blob, blob,
                                    cudaMemcpyDeviceToDevice, stream()) == cudaSuccess, "cache seed copy failed");
            slot_of[(size_t) il * NE + e] = (int32_t) slot;
            owner[slot] = (int32_t) (il * NE + e);
            // QFN_SEED_LFU_MAX caps the prompt count: uncapped counts outlive the decode's own working set.
            static const double seed_lfu_max = [] { const char * e = std::getenv("QFN_SEED_LFU_MAX"); return e && *e ? std::atof(e) : 1.0; }();
            lfu[slot] = seed_lfu_max > 0 ? std::min((double) h[e], seed_lfu_max) : (double) h[e];
            lfu_step[slot] = 0;
            resident_host[(size_t) il * NE + e] = 1.0f;
        }
        resident_dirty = true;
        slot_tab_dirty = true;
        return bytes;
    }

    // Same values as exp2(-dt / half-life), from a table: the victim scan evaluates every slot for each layer.
    mutable std::vector<double> lfu_pow;
    double lfu_factor(uint64_t dt) const {
        if (dt >= lfu_pow.size()) {
            if (dt >= (1u << 20)) return std::exp2(-(double) dt / std::max(1.0f, opt.lfu_half_life));
            const size_t from = lfu_pow.size();
            lfu_pow.resize((size_t) dt + 4096);
            for (size_t i = from; i < lfu_pow.size(); ++i) lfu_pow[i] = std::exp2(-(double) i / std::max(1.0f, opt.lfu_half_life));
        }
        return lfu_pow[(size_t) dt];
    }
    double lfu_now(uint32_t slot) const {
        return lfu[slot] * lfu_factor(step - lfu_step[slot]);
    }
    void lfu_touch(uint32_t slot) {
        lfu[slot] = lfu_now(slot) + 1.0;
        lfu_step[slot] = step;
    }

    // ---- miss placement (decode v2 and v3). Victims: lowest recency-weighted use, never an expert of this layer's
    // current selection. Misses come from the RAM tier when present there, otherwise from NVMe through DirectStorage
    // (adjacent experts merged into one extent) into a staging bank; all copies run on `strm`.
    uint32_t ram_slots = 0;
    uint8_t * ram = nullptr;                          // pinned [ram_slots * blob]
    std::vector<int32_t> ram_of, ram_owner;           // key -> RAM slot or -1; RAM slot -> key or -1
    std::vector<uint64_t> ram_age;
    uint64_t ram_clock = 0, ram_hits = 0;

    void alloc_ram_tier() {
        const char * g = std::getenv("QFN_RAM_TIER_GIB");
        const double gib = g && *g ? std::atof(g) : 0.0;
        ram_of.assign((size_t) hp->n_layer * hp->n_expert, -1);
        if (gib <= 0) return;
        const uint64_t blob = pack.experts[0].blob;
        ram_slots = (uint32_t) ((gib * 1073741824.0) / (double) blob);
        require(cudaHostAlloc((void **) &ram, (size_t) ram_slots * blob,
                             v3_future ? cudaHostAllocMapped : cudaHostAllocDefault) == cudaSuccess,
                "cannot pin the RAM expert tier (lower QFN_RAM_TIER_GIB)");
        if (v3_future) require(cudaHostGetDevicePointer(&ram_device, ram, 0) == cudaSuccess,
                               "QFN_FUTURE_SERVICE: configured RAM tier cannot be device-mapped; capacity retained, admission failed");
        ram_owner.assign(ram_slots, -1);
        ram_age.assign(ram_slots, 0);
        log("[qfn] RAM expert tier: %u slots (%.2f GiB pinned), exclusive with VRAM\n", ram_slots, ram_slots * (double) blob / 1073741824.0);
    }

    std::vector<uint32_t> miss_victims(uint32_t il, size_t count, const int32_t * sel, int n,
                                     const std::vector<uint8_t> * replay_pins = nullptr) const {
        const uint32_t NE = hp->n_expert;
        require(!replay_pins || replay_pins->size() == slot_of.size(), "invalid replay pin table");
        // One pass: the `count` lowest (value, slot) pairs, the same choice as repeated minimum searches.
        std::vector<std::pair<double, uint32_t>> best;
        best.reserve(count + 1);
        for (uint32_t s = 0; s < n_slots && count; ++s) {
            const int32_t o = owner[s];
            const double v = o < 0 ? -1.0 : lfu_now(s);
            if (best.size() == count && !(v < best.back().first)) continue;
            if (o >= 0 && replay_pins && (*replay_pins)[size_t(o)]) continue;
            if (o >= 0 && (uint32_t) o / NE == il) {
                bool used = false;
                for (int k = 0; k < n; ++k) if ((int32_t) (o % NE) == sel[k]) { used = true; break; }
                if (used) continue;
            }
            auto at = best.end();
            while (at != best.begin() && v < (at - 1)->first) --at;
            best.insert(at, { v, s });
            if (best.size() > count) best.pop_back();
        }
        require(best.size() == count, replay_pins ? "no evictable expert slot while retaining replay prefix" : "no evictable expert slot");
        std::vector<uint32_t> victims;
        victims.reserve(count);
        for (const auto & b : best) victims.push_back(b.second);
        return victims;
    }

    // Only the prefix through the first miss has valid activations. Rows after it,
    // and unused token columns in each DEC_NMAX row, must never acquire pins.
    void pin_replay_prefix(std::vector<uint8_t> & pins, const std::vector<int32_t> & selected,
                           uint32_t last, int n) const {
        require(last < hp->n_layer && n >= 1 && n <= DEC_NMAX && pins.size() == slot_of.size() &&
                selected.size() == size_t(hp->n_layer) * DEC_NMAX * hp->n_expert_used, "invalid replay pin prefix");
        for (uint32_t il = 0; il <= last; ++il) for (uint32_t k = 0; k < hp->n_expert_used * n; ++k) {
            const int32_t e = selected[size_t(il) * DEC_NMAX * hp->n_expert_used + k];
            require(e >= 0 && uint32_t(e) < hp->n_expert, "invalid expert in replay pin prefix");
            pins[size_t(il) * hp->n_expert + e] = 1;
        }
    }

    // returns the bytes read from NVMe; `touched` collects every key whose slot changed (for the v3 device table)
    uint64_t place_misses(uint32_t il, const std::vector<int32_t> & miss, const int32_t * sel, int n, cudaStream_t strm,
                          std::vector<size_t> * touched, miss_placement_parts * parts = nullptr,
                          const std::vector<uint8_t> * replay_pins = nullptr) {
        const uint32_t NE = hp->n_expert;
        const uint64_t blob = pack.experts[il].blob;
        const auto victims = miss_victims(il, miss.size(), sel, n, replay_pins);
        // misses held by the RAM tier are copied back; the rest are read from NVMe
        std::vector<int32_t> from_ram((size_t) miss.size(), -1);
        std::vector<int32_t> disk;
        for (size_t m = 0; m < miss.size(); ++m) {
            const size_t key = (size_t) il * NE + miss[m];
            if (ram_slots && ram_of[key] >= 0) from_ram[m] = ram_of[key];
            else disk.push_back(miss[m]);
        }
        // demote each victim with an owner into the RAM tier (oldest RAM slot not being promoted now)
        if (ram_slots) {
            for (uint32_t s : victims) {
                const int32_t o = owner[s];
                if (o < 0) continue;
                int32_t rs = -1;
                uint64_t oldest = UINT64_MAX;
                for (uint32_t r = 0; r < ram_slots; ++r) {
                    if (std::find(from_ram.begin(), from_ram.end(), (int32_t) r) != from_ram.end()) continue;
                    const uint64_t age = ram_owner[r] < 0 ? 0 : ram_age[r] + 1;
                    if (age < oldest) { oldest = age; rs = (int32_t) r; }
                }
                if (rs < 0) break;
                if (ram_owner[rs] >= 0) ram_of[(size_t) ram_owner[rs]] = -1;
                require(cudaMemcpyAsync(ram + (size_t) rs * blob, (const char *) slab + (size_t) s * blob, blob,
                                        cudaMemcpyDeviceToHost, strm) == cudaSuccess, "RAM tier demotion failed");
                ram_owner[rs] = o;
                ram_of[(size_t) o] = rs;
                ram_age[rs] = ++ram_clock;
            }
        }
        uint64_t bytes = 0;
        int b = -1;
        std::vector<uint64_t> staged((size_t) miss.size(), 0);
        if (!disk.empty()) {
            b = staging_next;
            staging_next ^= 1;
            const double bank_started = now_ms();
            require(cudaEventSynchronize(staging_free[b]) == cudaSuccess, "staging wait failed");
            if (parts) parts->bank_wait_ms += now_ms() - bank_started;
            std::vector<wf_extent> ext;
            uint64_t off = 0;
            for (size_t m = 0; m < miss.size(); ++m) {
                if (from_ram[m] >= 0) continue;
                staged[m] = off;
                const uint64_t fo = pack.experts[il].offset + (uint64_t) miss[m] * blob;
                if (!ext.empty() && ext.back().file_offset + ext.back().nbytes == fo) {
                    ext.back().nbytes += blob;
                } else {
                    wf_extent x{};
                    x.file_offset = fo; x.bank_offset = off; x.nbytes = blob; x.file_idx = 0;
                    ext.push_back(x);
                }
                off += blob;
            }
            double before[4] = {}, after[4] = {}, ms = 0;
            uint64_t requests_before = 0, requests_after = 0;
            if (parts) wf_extent_fetch_stats(before, &requests_before, nullptr);
            const double disk_started = now_ms();
            require(wf_fetch_extents_direct(wf, b, ext.data(), (int) ext.size(), strm, &bytes, &ms) && !wf_has_failed(wf),
                    "expert miss DirectStorage read failed");
            if (parts) {
                parts->disk_wall_ms += now_ms() - disk_started;
                wf_extent_fetch_stats(after, &requests_after, nullptr);
                for (int j = 0; j < 4; ++j) parts->ds_ms[j] += after[j] - before[j];
                parts->requests += requests_after - requests_before;
            }
        }
        const double payload_started = now_ms();
        for (size_t m = 0; m < miss.size(); ++m) {
            const uint32_t s = victims[m];
            if (owner[s] >= 0) {
                slot_of[(size_t) owner[s]] = -1;
                resident_host[(size_t) owner[s]] = 0.0f;
                if (touched) touched->push_back((size_t) owner[s]);
            }
            if (from_ram[m] >= 0) {
                const int32_t rs = from_ram[m];
                require(cudaMemcpyAsync((char *) slab + (size_t) s * blob, ram + (size_t) rs * blob, blob,
                                        cudaMemcpyHostToDevice, strm) == cudaSuccess, "RAM tier promotion failed");
                ++ram_hits;
            } else {
                require(cudaMemcpyAsync((char *) slab + (size_t) s * blob, (const char *) bank_ptr[b] + staged[m], blob,
                                        cudaMemcpyDeviceToDevice, strm) == cudaSuccess, "miss copy failed");
            }
            const size_t key = (size_t) il * NE + miss[m];
            slot_of[key] = (int32_t) s;
            owner[s] = (int32_t) key;
            resident_host[key] = 1.0f;
            if (touched) touched->push_back(key);
            lfu[s] = 0.0;
            lfu_step[s] = step;
            lfu_touch(s);
        }
        // promoted keys leave the RAM tier (exclusive)
        for (size_t m = 0; m < miss.size(); ++m) {
            if (from_ram[m] < 0) continue;
            const size_t key = (size_t) il * NE + miss[m];
            if (ram_of[key] == from_ram[m]) { ram_of[key] = -1; ram_owner[(size_t) from_ram[m]] = -1; ram_age[(size_t) from_ram[m]] = 0; }
        }
        if (b >= 0) require(cudaEventRecord(staging_free[b], strm) == cudaSuccess, "staging event failed");
        if (parts) parts->payload_enqueue_ms += now_ms() - payload_started;
        return bytes;
    }

    // Makes the selected experts of layer il resident; returns their slots. Misses are read with DirectStorage
    // (adjacent experts merged into one extent, one request per extent part >= 256 KiB) into a staging bank and
    // copied to the victims' slots.
    void ensure_resident(uint32_t il, const int32_t * sel, int n, int32_t * slots, qfn_decode_stats & st) {
        const uint32_t NE = hp->n_expert;
        const uint64_t blob = pack.experts[il].blob;
        std::vector<int32_t> miss;
        for (int k = 0; k < n; ++k) {
            const int32_t e = sel[k];
            require(e >= 0 && (uint32_t) e < NE, "router selected an invalid expert");
            const int32_t s = slot_of[(size_t) il * NE + e];
            if (s >= 0) { lfu_touch((uint32_t) s); ++st.expert_hits; }
            else if (std::find(miss.begin(), miss.end(), e) == miss.end()) miss.push_back(e);
        }
        if (!miss.empty()) {
            const double t0 = now_ms();
            std::sort(miss.begin(), miss.end());
            const uint64_t bytes = place_misses(il, miss, sel, n, stream(), nullptr);
            st.expert_misses += miss.size();
            st.miss_bytes += bytes;
            st.miss_ms += now_ms() - t0;
            resident_dirty = true;
            slot_tab_dirty = true;
        }
        for (int k = 0; k < n; ++k) slots[k] = slot_of[(size_t) il * NE + sel[k]];
    }

    // ------------------------------------------------------------------------------------------ PLE rows, decode
    HANDLE ple_file = INVALID_HANDLE_VALUE;
    uint8_t * ple_io = nullptr;   // 16 x 8 KiB aligned buffers
    ggml_tensor * ple_tok = nullptr, * ple_tok_idx = nullptr;

    void open_ple_reader() {
        ple_file = CreateFileW(pack.wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                               FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS, nullptr);
        require(ple_file != INVALID_HANDLE_VALUE, "cannot open the n-gram table for decode reads");
        ple_io = (uint8_t *) _aligned_malloc((size_t) hp->ple_n_heads * DEC_NMAX * 8192, 4096);
        require(ple_io != nullptr, "cannot allocate n-gram read buffers");
        const qfn_tensor & pt = pack.get("per_layer_token_embd.weight");
        ggml_init_params ip = { 4 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * c = ggml_init(ip);
        ple_tok = ggml_new_tensor_2d(c, pt.type, pt.ne[0], (int64_t) hp->ple_n_heads * DEC_NMAX);
        ple_tok_idx = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t) hp->ple_n_heads * DEC_NMAX);
        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors(c, gpu);
        require(b != nullptr, "cannot allocate decode n-gram rows");
        std::vector<int32_t> idx((size_t) hp->ple_n_heads * DEC_NMAX);
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int32_t) i;
        ggml_backend_tensor_set(ple_tok_idx, idx.data(), 0, idx.size() * 4);
        // c and b live as long as the engine (released with the process)
    }

    // the n-gram rows of the last n tokens of seq (16 each), read with overlapped unbuffered 4 KiB-aligned reads
    void load_ple_tokens(int n) {
        const qfn_tensor & pt = pack.get("per_layer_token_embd.weight");
        const uint64_t rb = ggml_row_size(pt.type, pt.ne[0]);
        std::vector<int32_t> tail(seq.end() - std::min<size_t>(seq.size(), hp->ple_ngram - 1 + (size_t) n), seq.end());
        std::vector<uint32_t> idx;
        ple_hash(tail, idx);
        const uint32_t nh = hp->ple_n_heads * (uint32_t) n;
        const uint32_t * mine = idx.data() + (tail.size() - (size_t) n) * hp->ple_n_heads;
        std::vector<OVERLAPPED> ov(nh);
        std::vector<HANDLE> ev(nh);
        std::vector<uint64_t> within(nh);
        for (uint32_t h = 0; h < nh; ++h) {
            const uint64_t off = pt.offset + (uint64_t) mine[h] * rb;
            const uint64_t al = off & ~4095ull;
            within[h] = off - al;
            const DWORD len = (DWORD) ((within[h] + rb + 4095) & ~4095ull);
            ev[h] = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            ov[h] = {};
            ov[h].Offset = (DWORD) (al & 0xFFFFFFFFu);
            ov[h].OffsetHigh = (DWORD) (al >> 32);
            ov[h].hEvent = ev[h];
            if (!ReadFile(ple_file, ple_io + (size_t) h * 8192, len, nullptr, &ov[h]))
                require(GetLastError() == ERROR_IO_PENDING, "n-gram row read failed");
        }
        std::vector<uint8_t> rows((size_t) nh * rb);
        for (uint32_t h = 0; h < nh; ++h) {
            DWORD got = 0;
            require(GetOverlappedResult(ple_file, &ov[h], &got, TRUE) != 0, "n-gram row read failed");
            CloseHandle(ev[h]);
            std::memcpy(rows.data() + h * rb, ple_io + (size_t) h * 8192 + within[h], rb);
        }
        ggml_backend_tensor_set(ple_tok, rows.data(), 0, rows.size());
    }

    void alloc_verify_state() {
        if (verify_state) return;
        for (auto * t : gdn_state) if (t) verify_state_tensors.push_back(t);
        for (auto * t : conv_state) if (t) verify_state_tensors.push_back(t);
        if (ple_conv) verify_state_tensors.push_back(ple_conv);
        for (auto * t : verify_state_tensors) verify_state_bytes += ggml_nbytes(t);
        require(verify_state_bytes && cudaMalloc(&verify_state, verify_state_bytes) == cudaSuccess,
                "cannot admit MTP canonical-repair recurrent checkpoint");
        verify_ctx = ggml_init({ 2 * ggml_tensor_overhead(), nullptr, true });
        require(verify_ctx != nullptr, "cannot allocate MTP routing metadata context");
        verify_resident = ggml_new_tensor_3d(verify_ctx, GGML_TYPE_F32, hp->n_expert, DEC_NMAX, hp->n_layer);
        ggml_set_input(verify_resident);
        verify_buf = ggml_backend_alloc_ctx_tensors(verify_ctx, gpu);
        require(verify_buf != nullptr, "cannot admit stable per-row MTP routing inputs");
        log("[qfn] MTP canonical repair admitted: recurrent checkpoint %zu bytes, stable routing %zu bytes; full batch graphs retained\n",
            verify_state_bytes, ggml_nbytes(verify_resident));
    }

    cache_policy capture_cache_policy() {
        cache_policy c;
        c.slot_of = slot_of; c.owner = owner; c.lfu = lfu; c.lfu_step = lfu_step;
        c.resident = resident_host; c.step = step; c.dirty = resident_dirty;
        c.ram_of = ram_of; c.ram_owner = ram_owner; c.ram_age = ram_age; c.ram_clock = ram_clock;
        c.routing.resize(resident_host.size());
        if (resident_dirty) c.routing = resident_host;
        else ggml_backend_tensor_get(resident, c.routing.data(), 0, c.routing.size() * sizeof(float));
        return c;
    }

    // Pure canonical policy, in the same order as ensure_resident/place_misses for ONE token.
    // Tensor publication intentionally matches v2, including rows preceding the first miss.
    void policy_layer(cache_policy & c, uint32_t il, const int32_t * selected, uint32_t count = 0) const {
        const uint32_t NE = hp->n_expert, K = count ? count : hp->n_expert_used;
        auto value = [&](uint32_t s) {
            return c.lfu[s] * std::exp2(-double(c.step - c.lfu_step[s]) / std::max(1.0f, opt.lfu_half_life));
        };
        auto touch = [&](uint32_t s) { c.lfu[s] = value(s) + 1.0; c.lfu_step[s] = c.step; };
        std::vector<int32_t> misses;
        for (uint32_t k = 0; k < K; ++k) {
            const int32_t e = selected[k];
            require(e >= 0 && uint32_t(e) < NE, "canonical cache simulation got invalid expert");
            const int32_t s = c.slot_of[size_t(il) * NE + e];
            if (s >= 0) touch(uint32_t(s));
            else if (std::find(misses.begin(), misses.end(), e) == misses.end()) misses.push_back(e);
        }
        std::sort(misses.begin(), misses.end());
        std::vector<uint32_t> victims;
        for (size_t m = 0; m < misses.size(); ++m) {
            double best = 1e300; int32_t victim = -1;
            for (uint32_t s = 0; s < c.owner.size(); ++s) {
                const int32_t key = c.owner[s];
                if (key >= 0 && uint32_t(key) / NE == il &&
                    std::find(selected, selected + K, key % NE) != selected + K) continue;
                if (std::find(victims.begin(), victims.end(), s) != victims.end()) continue;
                const double v = key < 0 ? -1.0 : value(s);
                if (v < best) { best = v; victim = int32_t(s); }
            }
            require(victim >= 0, "no evictable canonical expert slot"); victims.push_back(uint32_t(victim));
        }
        std::vector<int32_t> from_ram(misses.size(), -1);
        for (size_t m = 0; m < misses.size(); ++m)
            if (!c.ram_owner.empty()) from_ram[m] = c.ram_of[size_t(il) * NE + misses[m]];
        if (!c.ram_owner.empty()) for (uint32_t s : victims) {
            const int32_t key = c.owner[s]; if (key < 0) continue;
            int32_t rs = -1; uint64_t oldest = UINT64_MAX;
            for (uint32_t r = 0; r < c.ram_owner.size(); ++r) {
                if (std::find(from_ram.begin(), from_ram.end(), int32_t(r)) != from_ram.end()) continue;
                const uint64_t age = c.ram_owner[r] < 0 ? 0 : c.ram_age[r] + 1;
                if (age < oldest) { oldest = age; rs = int32_t(r); }
            }
            if (rs < 0) break;
            if (c.ram_owner[rs] >= 0) c.ram_of[c.ram_owner[rs]] = -1;
            c.ram_owner[rs] = key; c.ram_of[key] = rs; c.ram_age[rs] = ++c.ram_clock;
        }
        for (size_t m = 0; m < misses.size(); ++m) {
            const uint32_t s = victims[m];
            if (c.owner[s] >= 0) { c.slot_of[c.owner[s]] = -1; c.resident[c.owner[s]] = 0.0f; }
            const size_t key = size_t(il) * NE + misses[m];
            c.slot_of[key] = int32_t(s); c.owner[s] = int32_t(key); c.resident[key] = 1.0f;
            c.lfu[s] = 0.0; c.lfu_step[s] = c.step; touch(s);
        }
        for (size_t m = 0; m < misses.size(); ++m) if (from_ram[m] >= 0) {
            const size_t key = size_t(il) * NE + misses[m]; const int32_t r = from_ram[m];
            if (c.ram_of[key] == r) { c.ram_of[key] = -1; c.ram_owner[r] = -1; c.ram_age[r] = 0; }
        }
        if (!misses.empty()) c.dirty = true;
        if (c.dirty) std::copy(c.resident.begin() + size_t(il) * NE, c.resident.begin() + size_t(il + 1) * NE,
                               c.routing.begin() + size_t(il) * NE);
    }

    std::vector<float> simulate_verify_policy(const cache_policy & initial, int n) {
        cache_policy c = initial;
        const size_t NE = hp->n_expert, NL = hp->n_layer, K = hp->n_expert_used;
        std::vector<float> masks(NE * DEC_NMAX * NL, 0.0f);
        verify_prefix.clear(); verify_prefix.push_back(initial);
        for (int t = 0; t < n; ++t) {
            if (c.dirty) { c.routing = c.resident; c.dirty = false; }
            for (size_t il = 0; il < NL; ++il)
                std::copy(c.routing.begin() + il * NE, c.routing.begin() + (il + 1) * NE,
                          masks.begin() + (il * DEC_NMAX + t) * NE);
            for (uint32_t il = 0; il < NL; ++il)
                policy_layer(c, il, verify_selected.data() + (size_t(il) * DEC_NMAX + t) * K);
            c.dirty = false; ++c.step;
            verify_prefix.push_back(c);
        }
        return masks;
    }

    void verify_batch(const int32_t * toks, int n, int64_t p, float bias,
                      std::vector<float> & logits, qfn_decode_stats & stats) {
        require(n > 1 && n <= DEC_NMAX && verify_state && verify_resident, "MTP transaction was not admitted");
        const cache_policy initial = capture_cache_policy();
        verify_checkpoint_ready = false;
        verify_prefix.clear(); verify_prefix.push_back(initial);
        const size_t NE = hp->n_expert, NL = hp->n_layer;
        std::vector<float> masks(NE * DEC_NMAX * NL, 0.0f);
        for (size_t il = 0; il < NL; ++il) for (int t = 0; t < n; ++t)
            std::copy(initial.routing.begin() + il * NE, initial.routing.begin() + (il + 1) * NE,
                      masks.begin() + (il * DEC_NMAX + t) * NE);
        size_t off = 0;
        for (auto * tensor : verify_state_tensors) {
            require(cudaMemcpyAsync((char *) verify_state + off, tensor->data, ggml_nbytes(tensor), cudaMemcpyDeviceToDevice,
                                    stream()) == cudaSuccess, "MTP transaction checkpoint failed");
            off += ggml_nbytes(tensor);
        }
        verify_checkpoint_ready = true;
        ++stats.policy_batches;
        verify_selected.assign(size_t(hp->n_layer) * DEC_NMAX * hp->n_expert_used, -1);
        verify_active = true;
        try {
            for (int attempt = 0; attempt < n; ++attempt) {
                if (attempt) {
                    off = 0;
                    for (auto * tensor : verify_state_tensors) {
                        require(cudaMemcpyAsync(tensor->data, (const char *) verify_state + off, ggml_nbytes(tensor),
                                                cudaMemcpyDeviceToDevice, stream()) == cudaSuccess, "MTP canonical repair state restore failed");
                        off += ggml_nbytes(tensor);
                    }
                    ++stats.policy_replays;
                }
                ggml_backend_tensor_set(verify_resident, masks.data(), 0, masks.size() * sizeof(float));
                decode_n(toks, n, p, bias, logits, stats, true);
                const auto expected = simulate_verify_policy(initial, n);
                if (bias <= 0.0f || expected == masks) { verify_active = false; return; }
                ++stats.policy_conflicts;
                require(attempt + 1 < n, "MTP canonical routing did not converge within causal batch bound");
                masks = expected;
            }
        } catch (...) { verify_active = false; throw; }
        verify_active = false;
        throw std::runtime_error("MTP canonical routing repair exhausted batch bound");
    }

    // Restore displaced payloads from immutable extents, never a whole-cache copy.
    // Both VRAM and RAM tier slots retain canonical owners and recency state after keep.
    void commit_cache_policy(const cache_policy & c, qfn_decode_stats & stats) {
        const double start = now_ms();
        const uint64_t blob = pack.experts[0].blob;
        struct restore_row { int32_t key; uint32_t slot; bool host; };
        std::vector<restore_row> rows;
        for (uint32_t s = 0; s < c.owner.size(); ++s)
            if (c.owner[s] >= 0 && c.owner[s] != owner[s]) rows.push_back({ c.owner[s], s, false });
        for (uint32_t r = 0; r < c.ram_owner.size(); ++r)
            if (c.ram_owner[r] >= 0 && c.ram_owner[r] != ram_owner[r]) rows.push_back({ c.ram_owner[r], r, true });
        require(cudaStreamSynchronize(stream()) == cudaSuccess, "MTP cache commit prefetch sync failed");
        for (size_t base = 0; base < rows.size(); base += hp->n_expert) {
            const size_t count = std::min<size_t>(hp->n_expert, rows.size() - base);
            std::vector<wf_extent> ext;
            for (size_t j = 0; j < count; ++j) {
                const uint32_t key = uint32_t(rows[base + j].key), il = key / hp->n_expert, e = key % hp->n_expert;
                require(pack.experts[il].blob == blob, "MTP cache commit expert stride changed");
                wf_extent item{}; item.file_offset = pack.experts[il].offset + uint64_t(e) * blob;
                item.bank_offset = uint64_t(j) * blob; item.nbytes = blob; item.file_idx = 0; ext.push_back(item);
            }
            uint64_t bytes = 0; double ms = 0;
            require(wf_fetch_extents_direct(wf, 0, ext.data(), int(ext.size()), stream(), &bytes, &ms) && !wf_has_failed(wf),
                    "MTP canonical cache payload refetch failed");
            stats.miss_bytes += bytes; stats.policy_restore_bytes += bytes; stats.policy_restore_slots += count;
            for (size_t j = 0; j < count; ++j) {
                const auto & row = rows[base + j];
                void * dst = row.host ? (void *) (ram + size_t(row.slot) * blob) : (void *) ((char *) slab + size_t(row.slot) * blob);
                require(cudaMemcpyAsync(dst, (const char *) bank_ptr[0] + j * blob, blob,
                    row.host ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice, stream()) == cudaSuccess,
                    "MTP canonical cache payload restore failed");
                if (row.host) { ++stats.policy_restore_ram_slots; stats.policy_restore_ram_bytes += blob; }
            }
            require(cudaStreamSynchronize(stream()) == cudaSuccess, "MTP cache commit bank reuse sync failed");
        }
        slot_of = c.slot_of; owner = c.owner; lfu = c.lfu; lfu_step = c.lfu_step;
        resident_host = c.resident; step = c.step; resident_dirty = c.dirty;
        ram_of = c.ram_of; ram_owner = c.ram_owner; ram_age = c.ram_age; ram_clock = c.ram_clock;
        ggml_backend_tensor_set(resident, c.routing.data(), 0, c.routing.size() * sizeof(float));
        replay_routing_resident = c.routing;
        if (slot_tab) { ggml_backend_tensor_set(slot_tab, slot_of.data(), 0, slot_of.size() * sizeof(int32_t)); slot_tab_dirty = false; }
        else slot_tab_dirty = true;
        const double elapsed = now_ms() - start;
        stats.policy_restore_ms += elapsed;
        stats.miss_ms += elapsed;
        stats.route_ms += elapsed;
    }

    void commit_verify_cache(int keep, qfn_decode_stats & stats) {
        require(keep >= 0 && size_t(keep) < verify_prefix.size(), "invalid canonical MTP prefix commit");
        commit_cache_policy(verify_prefix[size_t(keep)], stats);
        verify_prefix.clear();
    }

    cache_policy replay_final_policy(cache_policy c, const std::vector<int32_t> & selected, int n) const {
        require(n >= 1 && n <= DEC_NMAX && selected.size() == size_t(hp->n_layer) * hp->n_expert_used * DEC_NMAX,
                "invalid replay canonical selection");
        if (c.dirty) { c.routing = c.resident; c.dirty = false; }
        for (uint32_t il = 0; il < hp->n_layer; ++il)
            policy_layer(c, il, selected.data() + size_t(il) * hp->n_expert_used * DEC_NMAX, hp->n_expert_used * n);
        c.dirty = false; ++c.step;
        return c;
    }

    // ------------------------------------------------------------------------------------------ decode step
    // n tokens (already appended to seq) at positions p..p+n-1 through the decode path. One graph per layer boundary:
    // the expert half of layer il-1 (slab, slot ids) and the attention + router half of layer il; the last graph is
    // the expert half of the last layer and the output head. logits: [n][n_vocab]. With snapshots (n > 1, MTP
    // verification) every recurrent state after each token is kept so rollback() can undo the tail exactly.
    void decode_n(const int32_t * toks, int n, int64_t p, float bias, std::vector<float> & logits, qfn_decode_stats & st,
                  bool snapshots) {
        if (v3_on) { decode_v3_n(toks, n, p, bias, logits, st, snapshots); return; }
        const int64_t E = hp->n_embd, HC = hp->hc, K = hp->n_expert_used, nt = n;
        require(n >= 1 && n <= DEC_NMAX, "decode step size out of range");
        require(!snapshots || n == 1 || !gdn_snap.empty(), "decode snapshots need the MTP buffers");
        if (resident_dirty) {
            ggml_backend_tensor_set(resident, resident_host.data(), 0, resident_host.size() * sizeof(float));
            resident_dirty = false;
        }
        if (!hp->ple_layers.empty()) {
            const double ple_start = phase_profile ? now_ms() : 0;
            load_ple_tokens(n);
            if (phase_profile) phase_ple_ms += now_ms() - ple_start;
        }
        const bool snap = snapshots && n > 1;
        static const bool exact_env = [] { const char * e = std::getenv("QFN_EXACT_SPLIT"); return e && *e == '1'; }();
        split_tokens = n > 1 && (snap || exact_env);
        std::vector<int32_t> slots((size_t) (K * nt)), sel((size_t) (K * nt));
        for (uint32_t il = 0; il <= hp->n_layer; ++il) {
            const double tb = now_ms();
            ggml_context * ctx = new_graph_ctx();
            gctx g{ ctx, ggml_new_graph_custom(ctx, graph_node_capacity(), false) };
            ggml_tensor * dres = ggml_view_2d(ctx, dec_res, E * HC, nt, dec_res->nb[1], 0);
            ggml_tensor * dx   = ggml_view_2d(ctx, dec_x, E, nt, dec_x->nb[1], 0);
            ggml_tensor * dinj = ggml_view_2d(ctx, dec_inj, HC, nt, dec_inj->nb[1], 0);
            ggml_tensor * dw   = ggml_view_3d(ctx, dec_w, 1, K, nt, dec_w->nb[1], dec_w->nb[2], 0);
            ggml_tensor * res = nullptr;
            ggml_tensor * slot_ids = nullptr, * logical_ids = nullptr;
            if (il > 0) {
                slot_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, K, nt);
                ggml_set_input(slot_ids);
                if(nvfp4 && (opt.nvfp4_decode_mode==5 || opt.nvfp4_decode_mode==6)) {
                    logical_ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,K,nt);
                    ggml_set_input(logical_ids);
                }
                ggml_tensor * out = experts(g, il - 1, dx, slot_ids, dw, slab_gate, slab_up, slab_down, false, logical_ids);
                res = hc_combine(g, ggml_reshape_3d(ctx, dres, E, HC, nt), out, dinj);
            }
            ggml_tensor * pos = nullptr, * pos4 = nullptr, * emb = nullptr, * pool_pos4 = nullptr, * logit_out = nullptr, * sel_out = nullptr;
            if (il < hp->n_layer) {
                pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nt);
                pos4 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * nt);
                ggml_set_input(pos); ggml_set_input(pos4);
                if (il == 0) {
                    emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, nt);
                    ggml_set_input(emb);
                    res = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, emb, E, 1, nt), E, HC, nt, 1);
                }
                if (hp->is_ple(il)) {
                    ggml_tensor * idx = ggml_view_1d(ctx, ple_tok_idx, (int64_t) hp->ple_n_heads * nt, 0);
                    ggml_tensor * pe = ggml_get_rows(ctx, ple_tok, idx);
                    pe = ggml_reshape_2d(ctx, pe, (int64_t) hp->ple_dim * hp->ple_n_heads, nt);
                    res = ple_block(g, il, res, pe, snap);
                }
                ggml_tensor * inj = nullptr;
                ggml_tensor * cur = hc_mix(g, res, layers[il].hc_attn_norm, layers[il].hc_attn_down, layers[il].hc_attn_up,
                                           layers[il].hc_attn_inject, &inj);
                if (hp->is_recurrent(il)) {
                    cur = gdn_layer(g, il, cur, snap);
                } else {
                    const int64_t R = hp->kpool, b_lo = p / R, b_hi = (p + nt) / R;
                    if (b_hi > b_lo) {
                        pool_pos4 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * (b_hi - b_lo));
                        ggml_set_input(pool_pos4);
                    }
                    cur = qsa_layer(g, il, cur, pos, pos4, pool_pos4, p, nt);
                }
                res = hc_combine(g, res, cur, inj);
                ggml_build_forward_expand(g.gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, res, E * HC, nt), dres));
                cur = hc_mix(g, res, layers[il].hc_ffn_norm, layers[il].hc_ffn_down, layers[il].hc_ffn_up, layers[il].hc_ffn_inject, &inj);
                ggml_build_forward_expand(g.gf, ggml_cpy(ctx, cur, dx));
                ggml_build_forward_expand(g.gf, ggml_cpy(ctx, inj, dinj));
                route r = router(g, il, cur, bias);
                ggml_build_forward_expand(g.gf, ggml_cpy(ctx, r.weights, dw));
                ggml_set_output(r.selected);
                ggml_build_forward_expand(g.gf, r.selected);
                sel_out = r.selected;
            } else {
                ggml_build_forward_expand(g.gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, res, E * HC, nt),
                                                         ggml_view_2d(ctx, dec_hout, E * HC, nt, dec_hout->nb[1], 0)));
                ggml_tensor * hcur = hc_mix(g, res, out_hc_norm, out_hc_down, out_hc_up, nullptr, nullptr);
                logit_out = MM(ctx, output, hcur);
                ggml_set_output(logit_out);
                ggml_build_forward_expand(g.gf, logit_out);
            }
            compute(g.gf);
            const double ts = now_ms();
            st.build_ms += ts - tb;
            if (slot_ids) set_input(slot_ids, slots.data(), 0, slots.size() * 4);
            // sel still holds the previous layer's canonical selection; slots is its physical mapping.
            if (logical_ids) set_input(logical_ids, sel.data(), 0, sel.size() * sizeof(int32_t));
            if (pos) {
                std::vector<int32_t> pv((size_t) nt), p4((size_t) (4 * nt), 0);
                for (int64_t i = 0; i < nt; ++i) {
                    pv[(size_t) i] = (int32_t) (p + i);
                    p4[(size_t) i] = p4[(size_t) (nt + i)] = p4[(size_t) (2 * nt + i)] = (int32_t) (p + i);
                }
                set_input(pos, pv.data(), 0, pv.size() * 4);
                if(!images.empty())image_positions(p4,nt,p);
                set_input(pos4, p4.data(), 0, p4.size() * 4);
            }
            if (emb) {
                const double embedding_start = phase_profile ? now_ms() : 0;
                std::vector<float> e((size_t) (E * nt));
                for (int64_t i = 0; i < nt; ++i) {
                    image_embedding(toks[i],p+i,e.data()+i*E);
                }
                set_input(emb, e.data(), 0, e.size() * 4);
                if (phase_profile) phase_embedding_ms += now_ms() - embedding_start;
            }
            if (pool_pos4) {
                const int64_t R = hp->kpool, b_lo = p / R, nn = pool_pos4->ne[0] / 4;
                std::vector<int32_t> pp((size_t) (4 * nn), 0);
                for (int64_t i = 0; i < nn; ++i) pp[(size_t) i] = pp[(size_t) (nn + i)] = pp[(size_t) (2 * nn + i)] = (int32_t) ((b_lo + i) * R);
                if(!images.empty())image_positions(pp,nn,b_lo*R,R);
                set_input(pool_pos4, pp.data(), 0, pp.size() * 4);
            }
            const double tg = now_ms();
            st.stage_ms += tg - ts;
            require(ggml_backend_graph_compute(gpu, g.gf) == GGML_STATUS_SUCCESS, "decode graph compute failed");
            const double tr = now_ms();
            st.gpu_ms += tr - tg;
            if (il < hp->n_layer) {
                ggml_backend_tensor_get(sel_out, sel.data(), 0, sel.size() * 4);
                if (affinity_calibrating && nt == 1) calibrate_take(il);
                if (verify_active) std::copy(sel.begin(), sel.end(), verify_selected.begin() + (size_t) il * K * DEC_NMAX);
                ensure_resident(il, sel.data(), (int) (K * nt), slots.data(), st);
                if (resident_dirty) {
                    ggml_backend_tensor_set(resident, resident_host.data() + (size_t) il * hp->n_expert,
                                            (size_t) il * hp->n_expert * sizeof(float), hp->n_expert * sizeof(float));
                }
            } else {
                logits.resize((size_t) (hp->n_vocab * nt));
                ggml_backend_tensor_get(logit_out, logits.data(), 0, logits.size() * 4);
            }
            st.route_ms += now_ms() - tr;
            ggml_free(ctx);
        }
        resident_dirty = false;
        split_tokens = false;
        last_step_n = snap ? n : 0;
        ++step;
    }

    // After a multi-token step with snapshots: keep only its first `keep` tokens (1 <= keep <= n). Restores the
    // recurrent states, conv windows and the n-gram conv window to "after token keep-1" and trims seq. The attention
    // and indexer caches need nothing: the rejected positions are overwritten before anything reads them.
    void rollback(int keep) {
        const int n = last_step_n;
        require(n > 0 && keep >= 1 && keep <= n, "rollback without a snapshot step");
        last_step_n = 0;
        if (keep == n) return;
        const int64_t S = hp->ssm_state, Hv = hp->ssm_dt_rank;
        for (uint32_t il = 0; il < hp->n_layer; ++il) {
            if (!hp->is_recurrent(il)) continue;
            require(cudaMemcpyAsync(gdn_state[il]->data, (const char *) gdn_snap[il]->data + (size_t) (keep - 1) * gdn_snap[il]->nb[1],
                                    (size_t) (S * S * Hv) * sizeof(float), cudaMemcpyDeviceToDevice, stream()) == cudaSuccess,
                    "state rollback failed");
            // conv state after token keep-1 = conv input columns keep .. keep+kernel-2
            ggml_tensor * h = conv_hist[il], * cs = conv_state[il];
            const int64_t w = cs->ne[0];
            require(cudaMemcpy2DAsync(cs->data, (size_t) w * sizeof(float), (const char *) h->data + (size_t) keep * sizeof(float),
                                      h->nb[1], (size_t) w * sizeof(float), (size_t) cs->ne[1],
                                      cudaMemcpyDeviceToDevice, stream()) == cudaSuccess, "conv rollback failed");
        }
        if (!hp->ple_layers.empty()) {
            const int64_t w = ple_conv->ne[0];
            require(cudaMemcpy2DAsync(ple_conv->data, (size_t) w * sizeof(float), (const char *) ple_hist->data + (size_t) keep * sizeof(float),
                                      ple_hist->nb[1], (size_t) w * sizeof(float), (size_t) ple_conv->ne[1],
                                      cudaMemcpyDeviceToDevice, stream()) == cudaSuccess, "n-gram conv rollback failed");
        }
        seq.resize(seq.size() - (size_t) (n - keep));
    }

    // ------------------------------------------------------------------------------------------ MTP drafter
    void load_mtp() {
        if (pack.mtp_tensors.empty()) { log("[qfn] the pack has no MTP block; speculative decoding off\n"); return; }
        const uint32_t M = hp->n_layer;   // the draft layer's index in the per-layer arrays
        const int64_t E = hp->n_embd, HC = hp->hc;
        auto T_ = [&](const char * n) -> const qfn_tensor & {
            auto it = pack.mtp_tensors.find(n);
            require(it != pack.mtp_tensors.end(), std::string("MTP tensor missing: ") + n);
            return it->second;
        };
        struct up { ggml_tensor * t; uint64_t off; uint64_t bytes; };
        std::vector<up> uploads;
        ggml_init_params ip = { 64 * ggml_tensor_overhead(), nullptr, true };
        mctx = ggml_init(ip);
        auto mk = [&](const char * n, int64_t ne0 = 0, int64_t ne1 = 0) {
            const qfn_tensor & t = T_(n);
            ggml_tensor * g = ne0 ? ggml_new_tensor_2d(mctx, t.type, ne0, ne1) :
                              (t.n_dims == 1 ? ggml_new_tensor_1d(mctx, t.type, t.ne[0]) : ggml_new_tensor_2d(mctx, t.type, t.ne[0], t.ne[1]));
            require(ggml_nbytes(g) == t.bytes, std::string("MTP tensor size differs: ") + n);
            uploads.push_back({ g, t.offset, t.bytes });
            return g;
        };
        layers.resize(M + 1);
        qfn_layer_w & L = layers[M];
        mtp_fc_e = mk("fc_embedding.weight");
        mtp_fc_h = mk("fc_hidden.weight");
        mtp_enorm = mk("pre_fc_norm_embedding.weight");
        mtp_hnorm = mk("pre_fc_norm_hidden.weight");
        mtp_head_norm = mk("hyper_connection_mixer.hc_norm.weight");
        mtp_head_down = mk("hyper_connection_mixer.input_mix_weight_down.weight");
        mtp_head_up   = mk("hyper_connection_mixer.input_mix_weight_up.weight");
        L.hc_attn_norm = mk("attn_hyper_connection.hc_norm.weight");
        L.hc_attn_down = mk("attn_hyper_connection.input_mix_weight_down.weight");
        L.hc_attn_up   = mk("attn_hyper_connection.input_mix_weight_up.weight");
        L.hc_attn_inject = mk("attn_hyper_connection.block_inject_weight.weight");
        L.hc_ffn_norm = mk("mlp_hyper_connection.hc_norm.weight");
        L.hc_ffn_down = mk("mlp_hyper_connection.input_mix_weight_down.weight");
        L.hc_ffn_up   = mk("mlp_hyper_connection.input_mix_weight_up.weight");
        L.hc_ffn_inject = mk("mlp_hyper_connection.block_inject_weight.weight");
        L.wq = mk("self_attn.q_proj.weight");
        L.wk = mk("self_attn.k_proj.weight");
        L.wv = mk("self_attn.v_proj.weight");
        L.wo = mk("self_attn.o_proj.weight");
        L.q_norm = mk("self_attn.q_norm.weight");
        L.k_norm = mk("self_attn.k_norm.weight");
        L.idx_q_norm = mk("self_attn.indexer.q_layernorm.weight");
        L.idx_k_norm = mk("self_attn.indexer.k_layernorm.weight");
        {
            // the fused indexer projection [q heads | k] (upstream's converter splits it at n_q rows)
            const qfn_tensor & t = T_("self_attn.indexer.index_qk_proj.weight");
            const int64_t n_q = (int64_t) hp->idx_heads * hp->idx_dim, n_k = hp->idx_dim;
            require(t.ne[1] == n_q + n_k, "MTP indexer projection rows differ");
            const uint64_t rb = ggml_row_size(t.type, t.ne[0]);
            L.idx_q_proj = ggml_new_tensor_2d(mctx, t.type, t.ne[0], n_q);
            L.idx_k_proj = ggml_new_tensor_2d(mctx, t.type, t.ne[0], n_k);
            uploads.push_back({ L.idx_q_proj, t.offset, rb * (uint64_t) n_q });
            uploads.push_back({ L.idx_k_proj, t.offset + rb * (uint64_t) n_q, rb * (uint64_t) n_k });
        }
        L.ffn_gate_inp = mk("mlp.gate.weight");
        L.ffn_gate_shexp = mk("mlp.shared_expert.gate_proj.weight");
        L.ffn_up_shexp = mk("mlp.shared_expert.up_proj.weight");
        L.ffn_down_shexp = mk("mlp.shared_expert.down_proj.weight");
        L.ffn_gate_inp_shexp = mk("mlp.shared_expert_gate.weight");
        // the draft head: output.weight rows of the draft vocabulary (all rows when the pack has no draft vocab)
        const qfn_tensor & ot = pack.get("output.weight");
        draft_ids = pack.draft_vocab;
        if (draft_ids.empty()) for (uint32_t v = 0; v < hp->n_vocab; ++v) draft_ids.push_back((int32_t) v);
        mtp_dhead = ggml_new_tensor_2d(mctx, ot.type, ot.ne[0], (int64_t) draft_ids.size());
        mtp_hin  = ggml_new_tensor_2d(mctx, GGML_TYPE_F32, E * HC, DEC_NMAX + 1);
        mtp_hout = ggml_new_tensor_2d(mctx, GGML_TYPE_F32, E * HC, DEC_NMAX + 1);
        mbuf = ggml_backend_alloc_ctx_tensors(mctx, gpu);
        require(mbuf != nullptr, "cannot allocate the MTP block in VRAM");
        {
            file_reader f(pack.wpath, false);
            std::vector<uint8_t> buf;
            for (const auto & u : uploads) {
                buf.resize(u.bytes);
                f.read(u.off, buf.data(), u.bytes);
                ggml_backend_tensor_set(u.t, buf.data(), 0, u.bytes);
            }
            const uint64_t rb = ggml_row_size(ot.type, ot.ne[0]);
            std::vector<uint8_t> row(rb), rows;
            rows.reserve(rb * 4096);
            for (size_t i = 0; i < draft_ids.size(); ++i) {
                f.read(ot.offset + (uint64_t) draft_ids[i] * rb, row.data(), rb);
                rows.insert(rows.end(), row.begin(), row.end());
                if (rows.size() >= rb * 4096 || i + 1 == draft_ids.size()) {
                    ggml_backend_tensor_set(mtp_dhead, rows.data(), (i + 1 - rows.size() / rb) * rb, rows.size());
                    rows.clear();
                }
            }
            // experts: Basalt's per-expert planes [codes(gate/up interleaved rows) | codes(down) | scales(gu) | scales(down)]
            // back to Q2_0 blocks in the gate | up | down blob layout the expert kernels read
            const auto & e0 = pack.mtp_expert_layout;
            require(e0.gu_type == GGML_TYPE_Q2_0 && e0.d_type == GGML_TYPE_Q2_0, "MTP expert relayout expects Q2_0");
            const uint64_t blob = pack.mtp_expert_blob, NE = hp->n_expert, FF = hp->n_ff_exp;
            const uint64_t gu_rows = 2 * FF, gu_blocks = E / 64, dn_rows = (uint64_t) E, dn_blocks = FF / 64;
            require(blob == (gu_rows * gu_blocks + dn_rows * dn_blocks) * 18, "MTP expert blob size differs");
            ggml_init_params xp = { 8 * ggml_tensor_overhead(), nullptr, true };
            ggml_context * xc = ggml_init(xp);
            ggml_tensor * raw = ggml_new_tensor_1d(xc, GGML_TYPE_I8, (int64_t) (blob * NE + (64u << 10)));
            mexp_buf = ggml_backend_alloc_ctx_tensors(xc, gpu);
            require(mexp_buf != nullptr, "cannot allocate the MTP experts in VRAM");
            ggml_backend_buffer_set_usage(mexp_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            std::vector<uint8_t> src(blob), dst(blob), zeros(64u << 10, 0);
            const uint64_t codes_gu = gu_rows * gu_blocks * 16, codes_dn = dn_rows * dn_blocks * 16, scales_gu = gu_rows * gu_blocks * 2;
            for (uint64_t e = 0; e < NE; ++e) {
                f.read(pack.mtp_experts_offset + e * blob, src.data(), blob);
                auto put = [&](uint64_t out_row, uint64_t blocks, const uint8_t * codes, const uint8_t * scales, uint8_t * base) {
                    for (uint64_t b = 0; b < blocks; ++b) {
                        uint8_t * o = base + (out_row * blocks + b) * 18;
                        std::memcpy(o, scales + b * 2, 2);
                        std::memcpy(o + 2, codes + b * 16, 16);
                    }
                };
                for (uint64_t r = 0; r < gu_rows; ++r) {
                    const uint8_t * c = src.data() + r * gu_blocks * 16;
                    const uint8_t * s = src.data() + codes_gu + codes_dn + r * gu_blocks * 2;
                    // interleaved row 2j = gate row j, 2j+1 = up row j
                    const uint64_t j = r / 2;
                    uint8_t * base = dst.data() + (r % 2 ? FF * gu_blocks * 18 : 0);
                    put(j, gu_blocks, c, s, base);
                }
                for (uint64_t r = 0; r < dn_rows; ++r) {
                    const uint8_t * c = src.data() + codes_gu + r * dn_blocks * 16;
                    const uint8_t * s = src.data() + codes_gu + codes_dn + scales_gu + r * dn_blocks * 2;
                    put(r, dn_blocks, c, s, dst.data() + 2 * FF * gu_blocks * 18);
                }
                ggml_backend_tensor_set(raw, dst.data(), e * blob, blob);
            }
            ggml_backend_tensor_set(raw, zeros.data(), NE * blob, zeros.size());
            auto mkx = [&](int64_t ne0, int64_t ne1, uint64_t off) {
                ggml_tensor * t = ggml_new_tensor_3d(xc, GGML_TYPE_Q2_0, ne0, ne1, (int64_t) NE);
                t->nb[1] = ggml_row_size(GGML_TYPE_Q2_0, ne0);
                t->nb[2] = blob;
                t->nb[3] = blob * NE;
                require(ggml_backend_tensor_alloc(mexp_buf, t, (char *) raw->data + off) == GGML_STATUS_SUCCESS, "cannot bind MTP experts");
                return t;
            };
            mtp_gate = mkx(E, (int64_t) FF, 0);
            mtp_up   = mkx(E, (int64_t) FF, e0.gate_bytes);
            mtp_down = mkx((int64_t) FF, E, 2 * e0.gate_bytes);
        }
        // the draft layer's caches
        const int64_t n_ctx = opt.n_ctx;
        ggml_init_params cp = { 8 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * cc = ggml_init(cp);
        kcache.resize(M + 1, nullptr); vcache.resize(M + 1, nullptr); idx_raw.resize(M + 1, nullptr); idx_pool.resize(M + 1, nullptr);
        kcache[M]   = ggml_new_tensor_3d(cc, GGML_TYPE_F16, hp->head_dim, hp->n_head_kv, n_ctx);
        vcache[M]   = ggml_new_tensor_3d(cc, GGML_TYPE_F16, hp->head_dim, hp->n_head_kv, n_ctx);
        idx_raw[M]  = ggml_new_tensor_2d(cc, GGML_TYPE_F16, hp->idx_dim, n_ctx);
        idx_pool[M] = ggml_new_tensor_2d(cc, GGML_TYPE_F16, hp->idx_dim, std::max<int64_t>(1, n_ctx / hp->kpool));
        require(ggml_backend_alloc_ctx_tensors(cc, gpu) != nullptr, "cannot allocate the MTP caches");
        h_last.assign((size_t) (E * HC), 0.0f);
        alloc_snapshots();
        alloc_verify_state(); // admitted before the cache is sized; never shrinks an explicit capacity
        if (const char * d = std::getenv("QFN_MTP_DRAFTS"); d && *d) mtp_drafts = std::max(0, std::min(DEC_NMAX - 1, std::atoi(d)));
        mtp_ok = true;
        log("[qfn] MTP drafter: %zu-token draft head, 512 resident experts (%.0f MiB), %d drafts per step\n",
            draft_ids.size(), (ggml_backend_buffer_get_size(mbuf) + ggml_backend_buffer_get_size(mexp_buf)) / 1048576.0, mtp_drafts);
    }

    // The draft layer over m tokens at positions p..p+m-1. Row i of its h input pairs with token i: by default rows
    // 0..m-1 of mtp_hin; with carry_row0 >= 0 the target's prefill residuals (carry rows carry_row0..), where a
    // negative first row is replaced by h0 (h_last, or zeros at position 0). Writes its residuals to mtp_hout (when
    // m <= its rows) and returns the draft-vocab logits of the last row.
    void mtp_run(const int32_t * toks, int m, int64_t p, std::vector<float> & dlogits, int64_t carry_row0 = INT64_MIN,
                 bool first_from_h0 = false, const float * h0 = nullptr, int image_token_shift = 0) {
        const int64_t E = hp->n_embd, HC = hp->hc, nt = m;
        const uint32_t M = hp->n_layer;
        ggml_context * ctx = new_graph_ctx();
        gctx g{ ctx, ggml_new_graph_custom(ctx, graph_node_capacity(), false) };
        ggml_tensor * emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, nt);
        ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nt);
        ggml_tensor * pos4 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * nt);
        ggml_set_input(emb); ggml_set_input(pos); ggml_set_input(pos4);
        ggml_tensor * h, * hfirst = nullptr;
        if (carry_row0 == INT64_MIN) {
            h = ggml_view_2d(ctx, mtp_hin, E * HC, nt, mtp_hin->nb[1], 0);
        } else if (!first_from_h0) {
            h = ggml_view_2d(ctx, carry, E * HC, nt, carry->nb[1], (size_t) carry_row0 * carry->nb[1]);
            if (carry_bf16) h = ggml_cast(ctx, h, GGML_TYPE_F32);
        } else {
            // row 0 from h0 (zeros when null), rows 1.. from carry rows carry_row0+1..
            hfirst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E * HC, 1);
            ggml_set_input(hfirst);
            if (nt > 1) {
                ggml_tensor * rest = ggml_view_2d(ctx, carry, E * HC, nt - 1, carry->nb[1], (size_t) (carry_row0 + 1) * carry->nb[1]);
                if (carry_bf16) rest = ggml_cast(ctx, rest, GGML_TYPE_F32);
                h = ggml_concat(ctx, hfirst, rest, 1);
            } else {
                h = hfirst;
            }
        }
        // pre_fc_norm_hidden is one RMSNorm over the whole hc-wide residual (weight [E*HC]), as in the HF module and in
        // Basalt, which self-distilled this pack's MTP layer with that forward (upstream llama.cpp normalizes per stream)
        ggml_tensor * hn = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_cont(ctx, h), hp->rms_eps), mtp_hnorm);   // [E*HC, nt]
        hn = ggml_reshape_3d(ctx, hn, E, HC, nt);
        ggml_tensor * en = ggml_mul(ctx, ggml_rms_norm(ctx, emb, hp->rms_eps), mtp_enorm);
        ggml_tensor * fe = ggml_mul_mat(ctx, mtp_fc_e, en);                                        // [E, nt]
        ggml_tensor * fh = ggml_mul_mat(ctx, mtp_fc_h, ggml_reshape_2d(ctx, hn, E, HC * nt));       // [E, HC*nt]
        ggml_tensor * res = ggml_add(ctx, ggml_reshape_3d(ctx, fh, E, HC, nt),
                                     ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, fe, E, 1, nt), E, HC, nt, 1));
        ggml_tensor * inj = nullptr;
        ggml_tensor * cur = hc_mix(g, res, layers[M].hc_attn_norm, layers[M].hc_attn_down, layers[M].hc_attn_up, layers[M].hc_attn_inject, &inj);
        const int64_t R = hp->kpool, b_lo = p / R, b_hi = (p + nt) / R;
        ggml_tensor * pool_pos4 = nullptr;
        if (b_hi > b_lo) { pool_pos4 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * (b_hi - b_lo)); ggml_set_input(pool_pos4); }
        cur = qsa_layer(g, M, cur, pos, pos4, pool_pos4, p, nt);
        res = hc_combine(g, res, cur, inj);
        cur = hc_mix(g, res, layers[M].hc_ffn_norm, layers[M].hc_ffn_down, layers[M].hc_ffn_up, layers[M].hc_ffn_inject, &inj);
        route r = router(g, M, cur, 0.0f);
        cur = experts(g, M, cur, r.selected, r.weights, mtp_gate, mtp_up, mtp_down);
        res = hc_combine(g, res, cur, inj);
        if (nt <= mtp_hout->ne[1]) {
            ggml_build_forward_expand(g.gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, res, E * HC, nt),
                                                     ggml_view_2d(ctx, mtp_hout, E * HC, nt, mtp_hout->nb[1], 0)));
        }
        ggml_tensor * probe_arg = nullptr;
        if (mtp_probe_out) {
            ggml_tensor * hall = hc_mix(g, res, mtp_head_norm, mtp_head_down, mtp_head_up, nullptr, nullptr);
            probe_arg = ggml_argmax(ctx, ggml_mul_mat(ctx, mtp_dhead, hall));          // [nt]
            ggml_set_output(probe_arg);
            ggml_build_forward_expand(g.gf, probe_arg);
        }
        ggml_tensor * last = ggml_view_3d(ctx, res, E, HC, 1, res->nb[1], res->nb[2], (size_t) (nt - 1) * res->nb[2]);
        ggml_tensor * hcur = hc_mix(g, ggml_cont(ctx, last), mtp_head_norm, mtp_head_down, mtp_head_up, nullptr, nullptr);
        ggml_tensor * lo = ggml_mul_mat(ctx, mtp_dhead, hcur);
        ggml_set_output(lo);
        ggml_build_forward_expand(g.gf, lo);
        compute(g.gf);
        std::vector<float> e((size_t) (E * nt));
        std::vector<int32_t> pv((size_t) nt), p4((size_t) (4 * nt), 0);
        for (int64_t i = 0; i < nt; ++i) {
            image_embedding(toks[i],p+i+image_token_shift,e.data()+i*E);
            pv[(size_t) i] = (int32_t) (p + i);
            p4[(size_t) i] = p4[(size_t) (nt + i)] = p4[(size_t) (2 * nt + i)] = (int32_t) (p + i);
        }
        set_input(emb, e.data(), 0, e.size() * 4);
        set_input(pos, pv.data(), 0, pv.size() * 4);
        if(!images.empty())image_positions(p4,nt,p);
                set_input(pos4, p4.data(), 0, p4.size() * 4);
        if (hfirst) {
            std::vector<float> z;
            if (!h0) z.assign((size_t) (E * HC), 0.0f);
            set_input(hfirst, h0 ? h0 : z.data(), 0, (size_t) (E * HC) * sizeof(float));
        }
        if (pool_pos4) {
            const int64_t nn = pool_pos4->ne[0] / 4;
            std::vector<int32_t> pp((size_t) (4 * nn), 0);
            for (int64_t i = 0; i < nn; ++i) pp[(size_t) i] = pp[(size_t) (nn + i)] = pp[(size_t) (2 * nn + i)] = (int32_t) ((b_lo + i) * R);
            if(!images.empty())image_positions(pp,nn,b_lo*R,R);
                set_input(pool_pos4, pp.data(), 0, pp.size() * 4);
        }
        require(ggml_backend_graph_compute(gpu, g.gf) == GGML_STATUS_SUCCESS, "MTP graph compute failed");
        dlogits.resize(draft_ids.size());
        ggml_backend_tensor_get(lo, dlogits.data(), 0, dlogits.size() * 4);
        if (probe_arg) {
            mtp_probe_out->resize((size_t) nt);
            ggml_backend_tensor_get(probe_arg, mtp_probe_out->data(), 0, (size_t) nt * 4);
        }
        ggml_free(ctx);
    }
    std::vector<int32_t> * mtp_probe_out = nullptr;

    // ---- draft-layer cache bookkeeping. Row q of the draft layer's caches is exact once it was computed from token q
    // and the target's residual of q-1. mtp_pend lists sequence tokens whose rows are still owed (positions
    // mtp_pend_p0..), with their h inputs already in mtp_hin rows 0..; h_last is the target residual of the last
    // sequence token.
    std::vector<int32_t> mtp_pend;
    int64_t mtp_pend_p0 = 0;

    qfn_native_shape paired_shape() const {
        const auto &h=*hp;
        return {h.n_layer,h.full_attn_interval,h.n_embd,h.hc,h.head_dim,h.n_head_kv,h.n_vocab,
            h.ssm_state,h.ssm_dt_rank,h.ssm_state*(2*h.ssm_group+h.ssm_dt_rank),h.ssm_conv-1,
            h.idx_dim,h.kpool,(h.ple_conv-1)*h.ple_ngram,opt.n_ctx,uint32_t(!pack.mtp_tensors.empty())};
    }
    void paired_allocate() {
        if(paired_buffer||!mtp_ok)return;
        const auto &h=*hp;
        paired_ctx=ggml_init({8*ggml_tensor_overhead(),nullptr,true});
        require(paired_ctx,"paired MTP metadata allocation failed");
        paired_mtp={ggml_new_tensor_3d(paired_ctx,GGML_TYPE_F16,h.head_dim,h.n_head_kv,opt.n_ctx),
            ggml_new_tensor_3d(paired_ctx,GGML_TYPE_F16,h.head_dim,h.n_head_kv,opt.n_ctx),
            ggml_new_tensor_2d(paired_ctx,GGML_TYPE_F16,h.idx_dim,opt.n_ctx),
            ggml_new_tensor_2d(paired_ctx,GGML_TYPE_F16,h.idx_dim,std::max<uint32_t>(1,opt.n_ctx/h.kpool))};
        paired_buffer=ggml_backend_alloc_ctx_tensors(paired_ctx,gpu);
        require(paired_buffer,"insufficient VRAM for paired MTP owners");
        ggml_backend_buffer_clear(paired_buffer,0);
    }
    void paired_prepare(const std::vector<float> &logits) {
        if(!mtp_ok)return;
        require(carry&&logits.size()==hp->n_vocab,"paired MTP needs target residuals/logits");
        paired_allocate();ggml_backend_synchronize(gpu);
        const uint32_t l=hp->n_layer;
        const std::array<ggml_tensor *,4> old={kcache[l],vcache[l],idx_raw[l],idx_pool[l]};
        kcache[l]=paired_mtp[0];vcache[l]=paired_mtp[1];idx_raw[l]=paired_mtp[2];idx_pool[l]=paired_mtp[3];
        auto restore=[&](){ggml_backend_synchronize(gpu);kcache[l]=old[0];vcache[l]=old[1];idx_raw[l]=old[2];idx_pool[l]=old[3];};
        try {
            const int64_t n=int64_t(seq.size()),start=paired_rows?paired_rows-1:0;
            require(start<n,"paired MTP frontier exceeds target");
            const int32_t parent=int32_t(std::max_element(logits.begin(),logits.end())-logits.begin());
            std::vector<float> dl;
            for(int64_t pos=start;pos<n;pos+=opt.tile) {
                const int count=int(std::min<int64_t>(opt.tile,n-pos));std::vector<int32_t> next(count);
                for(int j=0;j<count;j++)next[j]=pos+j+1<n?seq[size_t(pos+j+1)]:parent;
                mtp_run(next.data(),count,pos,dl,pos,false,nullptr,1);
            }
            restore();paired_rows=uint32_t(n);
            log("[qfn] paired MTP canonical rows %lld..%lld; target prompt not replayed\n",(long long)start,(long long)n);
        }catch(...){restore();paired_rows=0;throw;}
    }

    // Canonical owner I/O. Metal histories are time-major; ggml histories are
    // channel-major. Index/carry conversion is explicit and bounded, never a
    // reinterpretation of Mac F32 storage as CUDA F16/BF16 bytes.
    void paired_owner(const qfn_native_slice &r,void *data,bool write) {
        const uint64_t off=r.offset;const size_t n=size_t(r.bytes);ggml_tensor *t=nullptr;
        require(n&&n<=(size_t(16)<<20),"paired owner chunk exceeds bound");
        switch(r.owner) {
            case QFN_GDN:t=gdn_state.at(r.layer);break;
            case QFN_CONV:t=conv_state.at(r.layer);break;
            case QFN_KEY:t=r.layer==hp->n_layer?paired_mtp[0]:kcache.at(r.layer);break;
            case QFN_VALUE:t=r.layer==hp->n_layer?paired_mtp[1]:vcache.at(r.layer);break;
            case QFN_INDEX:t=r.layer==hp->n_layer?paired_mtp[2]:idx_raw.at(r.layer);break;
            case QFN_POOL:t=r.layer==hp->n_layer?paired_mtp[3]:idx_pool.at(r.layer);break;
            case QFN_PLE:t=ple_conv;break;
            case QFN_CARRY:t=carry;break;
            case QFN_PREV: {
                require(off==0&&n==32,"paired PLE token-history extent differs");
                int32_t expected[8];for(size_t i=0;i<8;i++)expected[i]=i<seq.size()?seq[seq.size()-1-i]:int32_t(hp->ple_eos);
                if(write)require(!std::memcmp(expected,data,32),"paired PLE history differs from target tokens");
                else std::memcpy(data,expected,32);return;
            }
            case QFN_ROPE: {
                require(off==0&&n==4,"paired rotary extent differs");const int32_t delta=image_delta();
                if(write)require(!std::memcmp(&delta,data,4),"paired rotary identity differs");else std::memcpy(data,&delta,4);return;
            }
            case QFN_POS: {
                require(off%16==0&&n%16==0,"paired position alignment differs");
                for(size_t i=0;i<n;i+=16) {const auto row=image_position(int64_t((off+i)/16));
                    if(write)require(!std::memcmp(row.data(),(char*)data+i,16),"paired image/position identity differs");
                    else std::memcpy((char*)data+i,row.data(),16);}
                return;
            }
            default:throw std::runtime_error("unknown paired owner");
        }
        require(t&&ggml_is_contiguous(t),"paired owner missing or strided");
        if(r.owner==QFN_CONV||r.owner==QFN_PLE) {
            const uint32_t time=r.owner==QFN_CONV?hp->ssm_conv-1:(hp->ple_conv-1)*hp->ple_ngram;
            require(off==0&&n==ggml_nbytes(t)&&n%4==0,"paired history must be complete");
            std::vector<float> native(n/4);const size_t channels=n/4/time;auto *canonical=static_cast<float *>(data);
            if(!write)ggml_backend_tensor_get(t,native.data(),0,n);
            for(size_t c=0;c<channels;c++)for(uint32_t j=0;j<time;j++)
                if(write)native[c*time+j]=canonical[size_t(j)*channels+c];else canonical[size_t(j)*channels+c]=native[c*time+j];
            if(write)ggml_backend_tensor_set(t,native.data(),0,n);return;
        }
        const bool half=r.owner==QFN_INDEX,bf=r.owner==QFN_CARRY&&carry_bf16;
        if(half||bf) {
            require(off%4==0&&n%4==0&&off/2<=ggml_nbytes(t)&&n/2<=ggml_nbytes(t)-off/2,"paired converted owner extent differs");
            std::vector<uint16_t> h(n/4);auto *f=static_cast<float *>(data);
            if(write) {
                for(size_t i=0;i<h.size();i++)h[i]=half?ggml_fp32_to_fp16(f[i]):ggml_fp32_to_bf16(f[i]).bits;
                for(uint16_t v:h)require(half?(v&0x7c00)!=0x7c00:(v&0x7f80)!=0x7f80,
                    "paired owner conversion overflowed native storage");
                ggml_backend_tensor_set(t,h.data(),size_t(off/2),n/2);
            } else {
                ggml_backend_tensor_get(t,h.data(),size_t(off/2),n/2);
                for(size_t i=0;i<h.size();i++)f[i]=half?ggml_fp16_to_fp32(h[i]):ggml_bf16_to_fp32(ggml_bf16_t{h[i]});
            }
        } else {
            require(off<=ggml_nbytes(t)&&n<=ggml_nbytes(t)-off,"paired owner extent exceeds allocation");
            if(write)ggml_backend_tensor_set(t,data,size_t(off),n);else ggml_backend_tensor_get(t,data,size_t(off),n);
        }
    }

    void mtp_flush() {
        if (mtp_pend.empty()) return;
        std::vector<float> dl;
        mtp_run(mtp_pend.data(), (int) mtp_pend.size(), mtp_pend_p0, dl);
        mtp_pend.clear();
    }

    // owe rows for tokens of the last decode step at positions p..p+n-1; token i pairs with dec_hout row base+i-1
    // (h_last when that is -1). Afterwards h_last = dec_hout row base+n-1.
    void remember_target_carry(int n, int64_t position, int base) {
        if (!mtp_ok || n == 0) return;
        require(carry && position >= 0 && position + n <= opt.n_ctx, "MTP export target history is unavailable");
        const size_t width = size_t(hp->n_embd) * hp->hc;
        if (!carry_bf16) {
            require(cudaMemcpyAsync((char *)carry->data + size_t(position) * carry->nb[1],
                    (char *)dec_hout->data + size_t(base) * dec_hout->nb[1], width * size_t(n) * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream()) == cudaSuccess, "MTP export residual history copy failed");
        } else {
            std::vector<float> full(width * n);
            std::vector<ggml_bf16_t> half(width * n);
            ggml_backend_tensor_get(dec_hout,full.data(),size_t(base)*dec_hout->nb[1],full.size()*sizeof(float));
            ggml_fp32_to_bf16_row(full.data(),half.data(),int64_t(full.size()));
            ggml_backend_tensor_set(carry,half.data(),size_t(position)*carry->nb[1],half.size()*sizeof(ggml_bf16_t));
        }
    }

    void mtp_owe_decoded(const int32_t * toks, int n, int64_t p, int base) {
        if (!mtp_ok) return;
        remember_target_carry(n,p,base);
        const size_t row = (size_t) hp->n_embd * hp->hc * sizeof(float);
        for (int i = 0; i < n; ++i) {
            if ((int64_t) mtp_pend.size() >= DEC_NMAX) mtp_flush();
            if (mtp_pend.empty()) mtp_pend_p0 = p + i;
            const size_t r = mtp_pend.size();
            if (base + i - 1 < 0) {
                ggml_backend_tensor_set(mtp_hin, h_last.data(), r * mtp_hin->nb[1], row);
            } else {
                require(cudaMemcpyAsync((char *) mtp_hin->data + r * mtp_hin->nb[1], (const char *) dec_hout->data + (size_t) (base + i - 1) * dec_hout->nb[1],
                                        row, cudaMemcpyDeviceToDevice, stream()) == cudaSuccess, "MTP h copy failed");
            }
            mtp_pend.push_back(toks[i]);
        }
        ggml_backend_tensor_get(dec_hout, h_last.data(), (size_t) (base + n - 1) * dec_hout->nb[1], row);
    }

    // after a layer-major prefill of positions p0..n-1: the draft layer over the same positions in tiles, h from carry
    void mtp_prefill(int64_t p0) {
        if (!mtp_ok) return;
        if (p0 == 0) { mtp_pend.clear(); std::fill(h_last.begin(), h_last.end(), 0.0f); }
        mtp_flush();
        const int64_t n = (int64_t) seq.size(), tile = opt.tile;
        std::vector<float> dl;
        for (int64_t t0 = p0; t0 < n; t0 += tile) {
            const int64_t m = std::min(tile, n - t0);
            // the first row pairs with the residual before p0: h_last (zeros at position 0); carry rows before p0
            // may be stale (decoded tokens never write carry)
            const char * pr = std::getenv("QFN_MTP_PROBE");
            std::vector<int32_t> arg;
            if (pr && *pr == '1') mtp_probe_out = &arg;
            mtp_run(seq.data() + t0, (int) m, t0, dl, t0 - 1, t0 == p0, (t0 == p0 && p0 > 0) ? h_last.data() : nullptr);
            mtp_probe_out = nullptr;
            if (!arg.empty()) {
                int64_t hit = 0, tot = 0;
                for (int64_t i = 0; i < m; ++i) {
                    if (t0 + i + 1 >= n) break;
                    ++tot;
                    if (draft_ids[(size_t) arg[(size_t) i]] == seq[(size_t) (t0 + i + 1)]) ++hit;
                }
                log("[qfn] MTP probe: draft argmax equals the next token at %lld of %lld rows (%.1f%%)\n", (long long) hit,
                    (long long) tot, 100.0 * hit / std::max<int64_t>(1, tot));
            }
        }
        carry_row(n - 1, h_last);
    }

    // one carry row as F32 on the host
    void carry_row(int64_t r, std::vector<float> & out) {
        const int64_t w = (int64_t) hp->n_embd * hp->hc;
        out.resize((size_t) w);
        if (!carry_bf16) { ggml_backend_tensor_get(carry, out.data(), (size_t) r * carry->nb[1], (size_t) w * sizeof(float)); return; }
        std::vector<ggml_bf16_t> b((size_t) w);
        ggml_backend_tensor_get(carry, b.data(), (size_t) r * carry->nb[1], (size_t) w * sizeof(ggml_bf16_t));
        ggml_bf16_to_fp32_row(b.data(), out.data(), w);
    }

    int32_t draft_pick(const std::vector<float> & dl) const {
        size_t best = 0;
        for (size_t i = 1; i < dl.size(); ++i) if (dl[i] > dl[best]) best = i;
        return draft_ids[best];
    }

    struct mtp_owed_prefix { int offset, count, base; bool first_only; };
    static mtp_owed_prefix owed_prefix_plan(bool drafted, int keep) {
        require(keep >= 1 && keep <= DEC_NMAX, "invalid MTP emitted-prefix bookkeeping");
        if (drafted && keep == 1) return {0, 0, 0, true};
        const int first = drafted ? 1 : 0;
        return {first, keep - first, first, false};
    }
    void owe_speculative_prefix(const std::vector<int32_t> & toks, int keep, int64_t p, bool drafted) {
        // The first target row is already drafted but still belongs to the
        // live transcript's export history, including callback interruption.
        if (drafted) remember_target_carry(1,p,0);
        const auto plan = owed_prefix_plan(drafted, keep);
        if (plan.first_only) ggml_backend_tensor_get(dec_hout, h_last.data(), 0,
                                                   size_t(hp->n_embd) * hp->hc * sizeof(float));
        else mtp_owe_decoded(toks.data() + plan.offset, plan.count, p + plan.offset, plan.base);
    }

    // Speculative decode (exact): drafts from the MTP layer, verified by one multi-token target step. Greedy keeps a
    // draft when it equals the target's choice; with temperature, standard speculative sampling with the draft as a
    // point-mass proposal (a draft d survives with the target probability p(d); a rejection samples p without d).
    bool decode_spec(int32_t first, int n_max, const qfn_sampling & s,
                     const std::function<bool(int32_t, const float *)> & on_token, qfn_decode_stats & st) {
        const double t0 = now_ms();
        const int64_t E = hp->n_embd, HC = hp->hc, V = hp->n_vocab;
        const size_t row = (size_t) (E * HC) * sizeof(float);
        int32_t y = first;
        int emitted = 0;
        std::vector<float> dl, logits;
        while (emitted < n_max) {
            const int64_t P = (int64_t) seq.size();
            require(P < (int64_t) opt.n_ctx, "context full");
            const int D = (int) std::max<int64_t>(0, std::min<int64_t>({ (int64_t) mtp_drafts, (int64_t) (n_max - emitted - 1),
                                                                        (int64_t) opt.n_ctx - P - 1 }));
            std::vector<int32_t> draft;
            if (D > 0) {
                // owed rows + y in one draft-layer step (y pairs with h_last), then one row per further draft
                if ((int64_t) mtp_pend.size() >= DEC_NMAX + 1) mtp_flush();
                if (mtp_pend.empty()) mtp_pend_p0 = P;
                ggml_backend_tensor_set(mtp_hin, h_last.data(), mtp_pend.size() * mtp_hin->nb[1], row);
                mtp_pend.push_back(y);
                const int rows = (int) mtp_pend.size();
                const double tm = now_ms();
                mtp_run(mtp_pend.data(), rows, mtp_pend_p0, dl);
                mtp_pend.clear();
                draft.push_back(draft_pick(dl));
                int last_row = rows - 1;
                for (int j = 1; j < D; ++j) {
                    require(cudaMemcpyAsync(mtp_hin->data, (const char *) mtp_hout->data + (size_t) last_row * mtp_hout->nb[1], row,
                                            cudaMemcpyDeviceToDevice, stream()) == cudaSuccess, "MTP h copy failed");
                    mtp_run(&draft.back(), 1, P + j, dl);
                    draft.push_back(draft_pick(dl));
                    last_row = 0;
                }
                st.mtp_ms += now_ms() - tm;
            }
            // verify [y, drafts] in one target step
            std::vector<int32_t> step_toks{ y };
            step_toks.insert(step_toks.end(), draft.begin(), draft.end());
            const int n = (int) step_toks.size();
            seq.insert(seq.end(), step_toks.begin(), step_toks.end());
            int a = 0, keep = 0;
            int32_t next = -1;
            bool stop = false, policy_committed = false;
            try {
                if (n > 1) verify_batch(step_toks.data(), n, P, s.residency_bias, logits, st);
                else decode_n(step_toks.data(), n, P, s.residency_bias, logits, st, false);
                for (;;) {
                    const float * L = logits.data() + (size_t) a * V;
                    if (a < (int) draft.size()) {
                        const int32_t d = draft[(size_t) a];
                        if (s.greedy()) {
                            const int32_t t = sample(L, s, P + a + 1);
                            if (t == d) { ++a; continue; }
                            next = t;
                        } else {
                            const double pd = prob_of(L, s, d, P + a + 1);
                            if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < pd) { ++a; continue; }
                            next = sample_excluding(L, s, d, P + a + 1);
                        }
                    } else {
                        next = sample(L, s, P + a + 1);
                    }
                    break;
                }
                ++mtp_rounds; mtp_drafted += draft.size(); mtp_accepted += (uint64_t) a;
                st.drafted += draft.size(); st.accepted += (uint64_t) a;
                // emit y and the accepted drafts; keep = tokens the sequence keeps
                for (int i = 0; i <= a; ++i) {
                    ++emitted; ++keep;
                    if (!on_token(step_toks[(size_t) i], logits.data() + (size_t) i * V)) { stop = true; break; }
                    if (emitted >= n_max) { stop = true; break; }
                    const int32_t after = i < a ? step_toks[(size_t) i + 1] : next;
                    if (std::find(eog.begin(), eog.end(), after) != eog.end()) { on_token(after, nullptr); stop = true; break; }
                }
                if (n > 1) rollback(keep);
                if (n > 1) commit_verify_cache(keep, st);
                policy_committed = true;
            } catch (...) {
                // Errors/cancelled callbacks commit exactly the rows already emitted.
                // A failure before emission restores the original recurrent/sequence state.
                if (n > 1 && !policy_committed && !verify_prefix.empty()) {
                    if (keep > 0 && last_step_n > 0) rollback(keep);
                    else if (keep == 0 && verify_checkpoint_ready) {
                        size_t off = 0;
                        for (auto * tensor : verify_state_tensors) {
                            require(cudaMemcpyAsync(tensor->data, (const char *) verify_state + off, ggml_nbytes(tensor),
                                    cudaMemcpyDeviceToDevice, stream()) == cudaSuccess, "MTP error rollback failed");
                            off += ggml_nbytes(tensor);
                        }
                        seq.resize(size_t(P)); last_step_n = 0;
                    }
                    commit_verify_cache(keep, st);
                }
                if (keep > 0) {
                    st.tokens += (uint64_t) keep;
                    owe_speculative_prefix(step_toks, keep, P, D > 0);
                }
                st.seconds += (now_ms() - t0) / 1000.0;
                throw;
            }
            st.tokens += (uint64_t) keep;
            // y's draft-layer row was computed with h_last (exact) when D > 0; the kept drafts owe theirs
            owe_speculative_prefix(step_toks, keep, P, D > 0);
            if (stop) break;
            y = next;
        }
        st.seconds += (now_ms() - t0) / 1000.0;
        return true;
    }

    // target probability of token d after the sampling filters
    double prob_of(const float * l, const qfn_sampling & s, int32_t d, int64_t history = -1) {
        std::vector<std::pair<int32_t, double>> dist;
        filtered(l, s, dist, history);
        for (auto & e : dist) if (e.first == d) return e.second;
        return 0.0;
    }
    int32_t sample_excluding(const float * l, const qfn_sampling & s, int32_t d, int64_t history = -1) {
        std::vector<std::pair<int32_t, double>> dist;
        filtered(l, s, dist, history);
        double tot = 0;
        for (auto & e : dist) if (e.first != d) tot += e.second;
        require(tot > 0, "excluded token has all target probability");
        double u = std::uniform_real_distribution<double>(0.0, tot)(rng);
        int32_t last = -1;
        for (const auto & e : dist) {
            if (e.first == d || e.second <= 0) continue;
            last = e.first; u -= e.second; if (u < 0) return e.first;
        }
        return last;
    }
    // One target distribution for ordinary and speculative decode. In a speculative
    // batch seq already includes unverified drafts: only history[0..history) exists
    // at this position. Rebuild the small upstream chain with that exact prefix.
    void filtered(const float * l, const qfn_sampling & s,
                  std::vector<std::pair<int32_t, double>> & out, int64_t history = -1) {
        const int64_t V = hp->n_vocab;
        const size_t end = history < 0 ? seq.size() : (size_t) history;
        require(end <= seq.size(), "sampling history beyond sequence");
        std::vector<llama_token_data> candidates((size_t) V);
        for (int32_t v = 0; v < V; ++v) candidates[(size_t) v] = {v, l[v], 0.0f};
        for (const auto & bias : s.logit_bias) {
            require(bias.first >= 0 && bias.first < V, "logit_bias token outside vocabulary");
            candidates[(size_t) bias.first].logit += bias.second;
        }
        auto deleter = [](llama_sampler * p) { llama_sampler_free(p); };
        std::unique_ptr<llama_sampler, decltype(deleter)> chain(
            llama_sampler_chain_init(llama_sampler_chain_default_params()), deleter);
        auto * penalties = llama_sampler_init_penalties(s.repeat_last_n < 0 ? (int) end : s.repeat_last_n,
            s.repeat_penalty, s.frequency_penalty, s.presence_penalty);
        const size_t window = s.repeat_last_n < 0 ? end : std::min(end, (size_t) s.repeat_last_n);
        for (size_t i = end - window; i < end; ++i) llama_sampler_accept(penalties, seq[i]);
        llama_sampler_chain_add(chain.get(), penalties);
        if (s.greedy()) {
            llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
        } else {
            // llama.cpp's standard order: penalties, top-k, typical, top-p, min-p, temperature.
            llama_sampler_chain_add(chain.get(), llama_sampler_init_top_k(s.top_k));
            llama_sampler_chain_add(chain.get(), llama_sampler_init_typical(s.typical_p, 1));
            llama_sampler_chain_add(chain.get(), llama_sampler_init_top_p(s.top_p, 1));
            llama_sampler_chain_add(chain.get(), llama_sampler_init_min_p(s.min_p, 1));
            llama_sampler_chain_add(chain.get(), llama_sampler_init_temp(s.temperature));
        }
        llama_token_data_array cur{candidates.data(), candidates.size(), -1, false};
        float input_max = -INFINITY;
        for (const auto & item : candidates) input_max = std::max(input_max, item.logit);
        require(std::isfinite(input_max), "no finite logits after logit_bias");
        llama_sampler_apply(chain.get(), &cur);
        require(cur.size > 0, "sampling filters removed every token");
        out.clear();
        if (s.greedy()) {
            require(cur.selected >= 0, "greedy sampler did not select a token");
            out.emplace_back(cur.data[cur.selected].id, 1.0);
            return;
        }
        float mx = -INFINITY;
        for (size_t i = 0; i < cur.size; ++i) mx = std::max(mx, cur.data[i].logit);
        require(std::isfinite(mx), "no finite logits after logit_bias and filters");
        double total = 0.0;
        for (size_t i = 0; i < cur.size; ++i) {
            const double p = std::exp((double) cur.data[i].logit - mx);
            out.emplace_back(cur.data[i].id, p); total += p;
        }
        for (auto & item : out) item.second /= total;
    }

    // ------------------------------------------------------------------------------------------ decode v3
    void prepare_future_service() {
        if (!v3_future || service_exec) return;
        if (const char * value = std::getenv("QFN_FUTURE_BLAS_WORKSPACE_MIB")) {
            char * end = nullptr;
            const long mib = std::strtol(value, &end, 10);
            require(*value && end && !*end && mib >= 1 && mib <= 256,
                    "QFN_FUTURE_BLAS_WORKSPACE_MIB must be an integer from1 to256");
            service_blas_bytes = size_t(mib) << 20;
        }
        log("[qfn] future service persistent cuBLAS workspace: %zu MiB; admitted before cache sizing, retained across warm parents\n",
                service_blas_bytes >> 20);
        require(hp->n_layer > 0 && hp->n_layer <= 64 &&
                uint64_t(hp->n_expert_used) * DEC_NMAX <= GGML_QFN_SERVICE_MAX,
                "QFN_FUTURE_SERVICE: layer/selection geometry exceeds checked descriptor; refusing truncation");
        size_t before = 0, total = 0, after = 0;
        require(cudaMemGetInfo(&before, &total) == cudaSuccess, "future service admission memory query failed");
        require(cudaMalloc(&service_blas, service_blas_bytes) == cudaSuccess,
                "future service persistent cuBLAS workspace allocation failed");
        void * semaphore = nullptr;
        require(wf_service_fence_prepare(wf, &semaphore), "QFN_FUTURE_SERVICE: dedicated provider fence preparation failed");
        service_semaphore = static_cast<cudaExternalSemaphore_t>(semaphore);
        const size_t descriptor_bytes = sizeof(*service_h) * hp->n_layer;
        require(cudaHostAlloc((void **) &service_h, descriptor_bytes, cudaHostAllocMapped) == cudaSuccess &&
                cudaHostGetDevicePointer((void **) &service_d, service_h, 0) == cudaSuccess,
                "QFN_FUTURE_SERVICE: complete mapped descriptor allocation failed");
        require(cudaHostAlloc((void **) &service_control_h, sizeof(*service_control_h), cudaHostAllocMapped) == cudaSuccess &&
                cudaHostGetDevicePointer((void **) &service_control_d, service_control_h, 0) == cudaSuccess,
                "QFN_FUTURE_SERVICE: mapped epoch control allocation failed");
        std::memset(service_h, 0, descriptor_bytes);
        std::memset(service_control_h, 0, sizeof(*service_control_h));
        if (!side) require(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking) == cudaSuccess,
                           "QFN_FUTURE_SERVICE: persistent service stream allocation failed");
        void * demote = nullptr, * fill = nullptr, * publish = nullptr;
        require(ggml_backend_cuda_qfn_service_prepare(&demote, &fill, &publish) == cudaSuccess,
                "QFN_FUTURE_SERVICE: copy/publication kernel preparation failed");
        require(cudaGraphCreate(&service_graph, 0) == cudaSuccess, "future service graph creation failed");
        service_waits.resize(hp->n_layer);
        cudaGraphNode_t previous = nullptr;
        for (uint32_t il = 0; il < hp->n_layer; ++il) {
            cudaExternalSemaphoreWaitParams value{}; value.params.fence.value = uint64_t(il) + 1;
            cudaExternalSemaphoreWaitNodeParams wait{&service_semaphore, &value, 1};
            require(cudaGraphAddExternalSemaphoresWaitNode(&service_waits[il], service_graph,
                    previous ? &previous : nullptr, previous ? 1 : 0, &wait) == cudaSuccess,
                    "future service graph external-wait node creation failed");
            previous = service_waits[il];
            auto * record = service_d + il;
            void * args[] = { &record };
            void * functions[] = { demote, fill, publish };
            for (int stage = 0; stage < 3; ++stage) {
                cudaKernelNodeParams kernel{};
                kernel.func = functions[stage]; kernel.kernelParams = args;
                kernel.gridDim = dim3(stage == 1 ? 16 : 1);
                kernel.blockDim = dim3(stage == 2 ? 1 : 256);
                cudaGraphNode_t node = nullptr;
                require(cudaGraphAddKernelNode(&node, service_graph, &previous, 1, &kernel) == cudaSuccess,
                        "future service graph copy/publication node creation failed");
                previous = node;
            }
        }
        require(cudaGraphInstantiate(&service_exec, service_graph, nullptr, nullptr, 0) == cudaSuccess &&
                cudaGraphUpload(service_exec, side) == cudaSuccess && cudaStreamSynchronize(side) == cudaSuccess,
                "QFN_FUTURE_SERVICE: persistent service graph admission/upload failed");
        require(cudaMemGetInfo(&after, &total) == cudaSuccess, "future service post-admission memory query failed");
        log("[qfn] future service admitted BEFORE cache sizing: %u layers, %zu mapped descriptor bytes, "
            "VRAM before=%zu after=%zu delta=%lld; all blob bytes/tails, RAM and MTP retained; explicit opt-in\n",
            hp->n_layer, descriptor_bytes + sizeof(*service_control_h), before, after,
            static_cast<long long>(before) - static_cast<long long>(after));
    }

    void abort_future_descriptor() {
        if (service_control_h)
            InterlockedExchange(reinterpret_cast<volatile LONG *>(&service_control_h->aborted), 1);
    }

    // The worker owns only CPU metadata/DS calls. Once it is joined, drain DS
    // BEFORE releasing inactive/aborted service waits, then prove both streams idle.
    void stop_future_service(bool abort) {
        if (abort) abort_future_descriptor();
        bell_stop.store(true, std::memory_order_release);
        if (bell_thread.joinable()) bell_thread.join();
        if (service_epoch) {
            if (!wf_service_finish_epoch(wf, service_last, abort)) {
                service_unproven = true;
                throw std::runtime_error("QFN_FUTURE_SERVICE_RELOAD_REQUIRED: storage retirement unproven; resources retained");
            }
            service_epoch = false;
        }
        if (service_live) {
            const auto main_status = cudaStreamSynchronize(stream());
            const auto side_status = side ? cudaStreamSynchronize(side) : cudaSuccess;
            if (main_status != cudaSuccess || side_status != cudaSuccess) {
                service_unproven = true;
                throw std::runtime_error("QFN_FUTURE_SERVICE_RELOAD_REQUIRED: main/service completion unproven; resources retained");
            }
            service_live = false;
        }
    }

    void destroy_future_parent(v3graph & G) {
        // Called only after the epoch and both graph branches have retired.
        if (G.future_parent_exec) cudaGraphExecDestroy(G.future_parent_exec);
        if (G.future_parent) cudaGraphDestroy(G.future_parent);
        G.future_parent_exec = nullptr; G.future_parent = nullptr;
        G.future_model = nullptr; G.future_capture = 0; G.future_waits.clear();
    }

    void prepare_future_parent(v3graph & G) {
        ggml_backend_cuda_graphs_disable_pop(gpu);
        const ggml_status prepared = ggml_backend_cuda_graph_prepare_static_with_workspace(
                gpu, G.gf, service_blas, service_blas_bytes);
        ggml_backend_cuda_graphs_disable_push(gpu);
        require(prepared == GGML_STATUS_SUCCESS, "future service model graph preparation failed");
        void * model = nullptr; uint64_t capture = 0;
        require(ggml_backend_cuda_graph_get_prepared(gpu, G.gf, &model, &capture),
                "future service prepared model graph is unavailable");
        if (G.future_parent_exec && G.future_model == model && G.future_capture == capture) return;
        // Child graphs snapshot model arguments. Rebuild only on recapture,
        // never retain stale tensor bindings after the borrowed handle expires.
        destroy_future_parent(G);
        require(cudaGraphClone(&G.future_parent, service_graph) == cudaSuccess,
                "future service parent template clone failed");
        G.future_waits.resize(service_waits.size());
        for (size_t il = 0; il < service_waits.size(); ++il)
            require(cudaGraphNodeFindInClone(&G.future_waits[il], service_waits[il], G.future_parent) == cudaSuccess,
                    "future service parent wait-node lookup failed");
        cudaGraphNode_t model_node = nullptr, completion_node = nullptr;
        const auto child_status = cudaGraphAddChildGraphNode(&model_node, G.future_parent, nullptr, 0,
                                                           static_cast<cudaGraph_t>(model));
        if (child_status != cudaSuccess) {
            size_t count = 0;
            cudaGraphGetNodes(static_cast<cudaGraph_t>(model), nullptr, &count);
            std::vector<cudaGraphNode_t> nodes(count);
            cudaGraphGetNodes(static_cast<cudaGraph_t>(model), nodes.data(), &count);
            std::map<int, size_t> types;
            for (auto node : nodes) {
                cudaGraphNodeType type;
                if (cudaGraphNodeGetType(node, &type) == cudaSuccess) ++types[int(type)];
            }
            std::string kinds;
            for (const auto & item : types) kinds += " " + std::to_string(item.first) + ":" + std::to_string(item.second);
            throw std::runtime_error("future service parent model child creation failed: " +
                    std::string(cudaGetErrorName(child_status)) + " (" + cudaGetErrorString(child_status) +
                    "), tokens=" + std::to_string(G.n) + ", node types=" + kinds);
        }
        // Depends ONLY on model completion. Parent completion also waits for
        // inactive service tickets, released by finish_epoch AFTER this event.
        require(cudaGraphAddEventRecordNode(&completion_node, G.future_parent, &model_node, 1,
                                           bell_completion) == cudaSuccess,
                "future service model-only completion node creation failed");
        require(cudaGraphInstantiate(&G.future_parent_exec, G.future_parent, nullptr, nullptr, 0) == cudaSuccess &&
                cudaGraphUpload(G.future_parent_exec, stream()) == cudaSuccess &&
                cudaStreamSynchronize(stream()) == cudaSuccess,
                "future service parent admission/upload failed");
        G.future_model = model; G.future_capture = capture;
        log("[qfn] future service parent prepared: model child + %zu direct service layers; one CUDA graph launch/token, capture=%llu\n",
            service_waits.size(), (unsigned long long) capture);
    }

    void destroy_future_service() {
        if (!service_h && !service_graph && !service_control_h && !service_blas) return;
        if (service_unproven) {
            std::fprintf(stderr, "QFN_FUTURE_SERVICE_RELOAD_REQUIRED: refusing to free possibly live graph/cache/RAM; process teardown\n");
            std::fflush(stderr); std::_Exit(3);
        }
        if (service_epoch || service_live) stop_future_service(true);
        // Preparation can have submitted an upload even if admission later
        // failed before the first epoch. It still owns graph arguments.
        if (side && cudaStreamSynchronize(side) != cudaSuccess) {
            std::fprintf(stderr, "QFN_FUTURE_SERVICE_RELOAD_REQUIRED: service upload completion unproven; process teardown\n");
            std::fflush(stderr); std::_Exit(3);
        }
        for (auto * G : v3g) destroy_future_parent(*G);
        if (service_exec) cudaGraphExecDestroy(service_exec);
        if (service_graph) cudaGraphDestroy(service_graph);
        if (service_h) cudaFreeHost(service_h);
        if (service_control_h) cudaFreeHost(service_control_h);
        if (service_blas) cudaFree(service_blas);
        service_blas = nullptr;
        service_exec = nullptr; service_graph = nullptr; service_h = service_d = nullptr;
        service_control_h = service_control_d = nullptr; service_waits.clear();
    }

    double bell_prepare_ms = 0, bell_fetch_ms = 0, bell_ack_ms = 0;  // host worker only; read after the token joins
    void serve_future_bell(int32_t request, uint32_t il, const std::vector<int32_t> & selected) {
        const double started = now_ms();
        try {
        const uint32_t NE = hp->n_expert;
        require(il < hp->n_layer && request == v3_seq * 64 + int32_t(il) &&
                !selected.empty() && selected.size() <= GGML_QFN_SERVICE_MAX,
                "QFN_FUTURE_SERVICE: invalid immutable miss request");
        const uint64_t blob = pack.experts[il].blob;
        std::vector<int32_t> miss;
        for (int32_t e : selected) {
            require(e >= 0 && uint32_t(e) < NE, "future service router selected invalid expert");
            if (slot_of[size_t(il) * NE + e] < 0 && std::find(miss.begin(), miss.end(), e) == miss.end()) miss.push_back(e);
        }
        require(!miss.empty(), "future service request has no missing expert");
        bell_miss += miss.size();
        std::sort(miss.begin(), miss.end());
        const auto victims = miss_victims(il, miss.size(), selected.data(), int(selected.size()));
        std::vector<int32_t> from_ram(miss.size(), -1);
        for (size_t m = 0; m < miss.size(); ++m)
            if (ram_slots) from_ram[m] = ram_of[size_t(il) * NE + miss[m]];
        auto next_ram_owner = ram_owner, next_ram_of = ram_of;
        auto next_ram_age = ram_age;
        uint64_t next_ram_clock = ram_clock;
        auto & record = service_h[il];
        require(!record.active && record.ticket == service_first + il, "future service descriptor reused while live");
        if (ram_slots) for (uint32_t s : victims) {
            const int32_t key = owner[s]; if (key < 0) continue;
            int32_t rs = -1; uint64_t oldest = UINT64_MAX;
            for (uint32_t r = 0; r < ram_slots; ++r) {
                if (std::find(from_ram.begin(), from_ram.end(), int32_t(r)) != from_ram.end()) continue;
                const uint64_t age = next_ram_owner[r] < 0 ? 0 : next_ram_age[r] + 1;
                if (age < oldest) { oldest = age; rs = int32_t(r); }
            }
            if (rs < 0) break;
            require(record.demotions < GGML_QFN_SERVICE_MAX && ram_device,
                    "future service RAM demotion descriptor/mapping invalid");
            record.demote[record.demotions++] = {uint64_t(reinterpret_cast<uintptr_t>(slab)) + uint64_t(s) * blob,
                uint64_t(reinterpret_cast<uintptr_t>(ram_device)) + uint64_t(rs) * blob, blob};
            if (next_ram_owner[rs] >= 0) next_ram_of[size_t(next_ram_owner[rs])] = -1;
            next_ram_owner[rs] = key; next_ram_of[size_t(key)] = rs; next_ram_age[rs] = ++next_ram_clock;
        }
        const bool disk = std::find(from_ram.begin(), from_ram.end(), -1) != from_ram.end();
        const int bank = staging_next;
        if (disk) staging_next ^= 1;
        std::vector<wf_extent> extents;
        uint64_t offset = 0;
        for (size_t m = 0; m < miss.size(); ++m) {
            const uint32_t slot = victims[m];
            uint64_t source = 0;
            if (from_ram[m] >= 0) {
                require(ram_device && uint32_t(from_ram[m]) < ram_slots, "future service RAM promotion mapping invalid");
                source = uint64_t(reinterpret_cast<uintptr_t>(ram_device)) + uint64_t(from_ram[m]) * blob;
            } else {
                require(offset <= bank_bytes && blob <= bank_bytes - offset, "future service complete blobs exceed staging bank");
                source = uint64_t(reinterpret_cast<uintptr_t>(bank_ptr[bank])) + offset;
                const uint64_t file = pack.experts[il].offset + uint64_t(miss[m]) * blob;
                if (!extents.empty() && extents.back().file_offset + extents.back().nbytes == file)
                    extents.back().nbytes += blob;
                else extents.push_back({file, offset, blob, 0});
                offset += blob;
            }
            record.fill[record.fills++] = {source,
                uint64_t(reinterpret_cast<uintptr_t>(slab)) + uint64_t(slot) * blob, blob};
            if (owner[slot] >= 0) record.update[record.updates++] = {uint64_t(owner[slot]), -1, 0};
            record.update[record.updates++] = {uint64_t(il) * NE + uint32_t(miss[m]), int32_t(slot), 0};
        }
        require(record.fills <= GGML_QFN_SERVICE_MAX && record.updates <= 2 * GGML_QFN_SERVICE_MAX && blob,
                "future service descriptor overflow/zero blob");
        record.active = 1;
        MemoryBarrier();
        uint64_t bytes = 0; double milliseconds = 0;
        const double fetch_started = now_ms();
        bell_prepare_ms += fetch_started - started;
        const bool loaded = wf_fetch_extents_service(wf, bank, extents.data(), int(extents.size()),
                                                     record.ticket, &bytes, &milliseconds);
        const double fetch_done = now_ms();
        bell_fetch_ms += fetch_done - fetch_started;
        // Count actual completed IO even if later GPU publication fails.
        bell_bytes += bytes;
        require(loaded, "QFN_FUTURE_SERVICE: real DirectStorage service submission failed");
        while (bell_h->ack != request && !bell_stop.load(std::memory_order_acquire) &&
               !service_control_h->aborted && !service_control_h->error && !bell_h->error) _mm_pause();
        require(bell_h->ack == request && !service_control_h->aborted && !service_control_h->error && !bell_h->error,
                "QFN_FUTURE_SERVICE: copy/publication did not acknowledge before abort/timeout");
        bell_ack_ms += now_ms() - fetch_done;
        // GPU publication proves all demotions/fills and bank reads completed.
        // Only now commit physical CPU metadata, once for this immutable request.
        for (size_t m = 0; m < miss.size(); ++m) {
            const uint32_t slot = victims[m];
            if (owner[slot] >= 0) { slot_of[size_t(owner[slot])] = -1; resident_host[size_t(owner[slot])] = 0.0f; }
            const size_t key = size_t(il) * NE + miss[m];
            slot_of[key] = int32_t(slot); owner[slot] = int32_t(key); resident_host[key] = 1.0f;
            lfu[slot] = 0; lfu_step[slot] = step; lfu_touch(slot);
            if (from_ram[m] >= 0) {
                ++ram_hits;
                if (next_ram_of[key] == from_ram[m]) {
                    next_ram_of[key] = -1; next_ram_owner[size_t(from_ram[m])] = -1; next_ram_age[size_t(from_ram[m])] = 0;
                }
            }
        }
        ram_of.swap(next_ram_of); ram_owner.swap(next_ram_owner); ram_age.swap(next_ram_age); ram_clock = next_ram_clock;
        ++bell_served; bell_ms.store(bell_ms.load() + now_ms() - started);
        } catch (...) {
            bell_ms.store(bell_ms.load() + now_ms() - started);
            throw;
        }
    }

    void execute_future_graph(v3graph & G) {
        require(service_exec && !service_epoch && !service_live && !service_unproven,
                "QFN_FUTURE_SERVICE: unavailable or live previous epoch");
        // No model computation occurs here. Every lazy allocation, capture and
        // upload finishes before any future external-wait graph is submitted.
        prepare_future_parent(G);
        try {
            require(wf_service_begin_epoch(wf, hp->n_layer, &service_first, &service_last), "future service epoch reservation failed");
            service_epoch = true;
            std::memset(service_control_h, 0, sizeof(*service_control_h));
            service_control_h->first_ticket = service_first; service_control_h->last_ticket = service_last;
            bell_h->req = -1; bell_h->ack = -1; bell_h->error = 0;
            for (uint32_t il = 0; il < hp->n_layer; ++il) {
                auto & record = service_h[il]; std::memset(&record, 0, sizeof(record));
                record.magic = GGML_QFN_SERVICE_MAGIC; record.ticket = service_first + il;
                record.table = uint64_t(reinterpret_cast<uintptr_t>(slot_tab->data));
                record.table_entries = uint64_t(hp->n_expert) * hp->n_layer;
                record.bell = uint64_t(reinterpret_cast<uintptr_t>(bell_d));
                record.control = uint64_t(reinterpret_cast<uintptr_t>(service_control_d));
                record.request = v3_seq * 64 + int32_t(il); record.layer = int32_t(il);
                cudaExternalSemaphoreWaitParams value{}; value.params.fence.value = record.ticket;
                cudaExternalSemaphoreWaitNodeParams wait{&service_semaphore, &value, 1};
                require(cudaGraphExecExternalSemaphoresWaitNodeSetParams(G.future_parent_exec, G.future_waits[il], &wait) == cudaSuccess,
                        "future service persistent wait ticket update failed");
            }
            MemoryBarrier();
            bell_stop.store(false, std::memory_order_release);
            { std::lock_guard<std::mutex> lock(bell_err_mutex); bell_err.clear(); }
            bell_thread = std::thread([this]() {
                const HRESULT apartment = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
                try {
                    require(SUCCEEDED(apartment), "future service CPU worker COM MTA admission failed");
                    int32_t seen = -1;
                    while (!bell_stop.load(std::memory_order_acquire) && !service_control_h->aborted) {
                        const int32_t request = bell_h->req;
                        if (request < 0 || request == seen) { _mm_pause(); continue; }
                        MemoryBarrier();
                        const int32_t layer = bell_h->layer, count = bell_h->n;
                        require(layer >= 0 && uint32_t(layer) < hp->n_layer && count > 0 && count <= GGML_QFN_SERVICE_MAX,
                                "future service immutable request geometry invalid");
                        const std::vector<int32_t> selected(bell_h->sel, bell_h->sel + count);
                        serve_future_bell(request, uint32_t(layer), selected);
                        seen = request;
                    }
                } catch (const std::exception & error) {
                    { std::lock_guard<std::mutex> lock(bell_err_mutex); bell_err = error.what(); }
                    abort_future_descriptor();
                }
                if (SUCCEEDED(apartment)) CoUninitialize();
            });
            service_live = true; // include any partially submitted launch failure
            require(ggml_backend_cuda_graph_launch_prepared_parent(gpu, G.gf,
                        static_cast<void *>(G.future_parent_exec), G.future_capture) == GGML_STATUS_SUCCESS,
                    "future service single parent graph launch failed");
            cudaError_t completion;
            while ((completion = cudaEventQuery(bell_completion)) == cudaErrorNotReady) std::this_thread::yield();
            require(completion == cudaSuccess, "future service main completion query failed");
            if (bell_h->error || service_control_h->aborted || service_control_h->error) {
                std::lock_guard<std::mutex> lock(bell_err_mutex);
                throw std::runtime_error("QFN_FUTURE_SERVICE: bounded miss timeout or service failure; fresh prefill required: " + bell_err);
            }
            stop_future_service(false);
            // Future normal loads use staging_free events; record after both
            // graphs retire, outside every live miss callback.
            for (auto event : staging_free) require(cudaEventRecord(event, stream()) == cudaSuccess,
                                                    "future service staging release event failed");
        } catch (...) {
            const auto original = std::current_exception();
            doorbell_poisoned = true;
            stop_future_service(true);
            std::rethrow_exception(original);
        }
    }

    // New streams have their own lazy external-fence/shared-VA association. Warm this exact
    // consumer outside a waiting graph; the load-time WF warmup predates creation of `side`.
    void warm_bell_stream() {
        require(side && wf, "doorbell warmup requires the live DirectStorage side stream");
        const double started = now_ms();
        ggml_backend_synchronize(gpu);
        const int bank = staging_next;
        require(cudaEventSynchronize(staging_free[bank]) == cudaSuccess, "doorbell warmup staging wait failed");
        void * probe = nullptr;
        require(cudaMalloc(&probe, 16) == cudaSuccess, "cannot allocate bounded doorbell warmup probe");
        try {
            wf_extent extent{};
            extent.file_offset = pack.experts[0].offset; extent.bank_offset = 0;
            extent.nbytes = pack.experts[0].blob; extent.file_idx = 0;
            uint64_t bytes = 0; double read_ms = 0;
            require(wf_fetch_extents_direct(wf, bank, &extent, 1, side, &bytes, &read_ms) && !wf_has_failed(wf),
                    "doorbell side-stream actual DirectStorage warmup failed");
            require(cudaMemcpyAsync(probe, bank_ptr[bank], 16, cudaMemcpyDeviceToDevice, side) == cudaSuccess &&
                    cudaEventRecord(staging_free[bank], side) == cudaSuccess &&
                    cudaStreamSynchronize(side) == cudaSuccess,
                    "doorbell side-stream fence/shared-VA warmup visibility failed");
            require(cudaFree(probe) == cudaSuccess, "cannot release doorbell warmup probe"); probe = nullptr;
            bell_warm_bytes += bytes;
            bell_warm_ms += now_ms() - started;
            log("[qfn] v3 doorbell actual side-stream warmup: %llu bytes %.3f ms (DS %.3f ms), timeout %d ms unchanged; counts toward TTFT and IO ledger\n",
                (unsigned long long) bytes, now_ms() - started, read_ms, bell_h->timeout_ms);
        } catch (...) { if (probe) cudaFree(probe); throw; }
    }

    void retire_poisoned_v3() {
        v3_shutdown();
        ggml_backend_synchronize(gpu);
        if (side) require(cudaStreamSynchronize(side) == cudaSuccess, "cache release side-stream sync failed");
        if (replay_miss_latch) require(cudaFree(replay_miss_latch) == cudaSuccess, "cache release replay-latch free failed");
        replay_miss_latch = nullptr;
        for (auto * G : v3g) { destroy_future_parent(*G); if (G->ga) ggml_gallocr_free(G->ga); if (G->ctx) ggml_free(G->ctx); delete G; }
        v3g.clear();
        if (v3buf) ggml_backend_buffer_free(v3buf);
        if (v3ctx) ggml_free(v3ctx);
        v3buf = nullptr; v3ctx = nullptr; slot_tab = sel_log = nullptr;
        if (bell_completion) cudaEventDestroy(bell_completion);
        if (side) cudaStreamDestroy(side);
        if (bell_h) cudaFreeHost((void *) bell_h);
        side = nullptr; bell_completion = nullptr; bell_h = nullptr; bell_d = nullptr; bell_err.clear();
    }

    void v3_init() {
        require_live_context();
        if (v3ctx) return;
        const uint32_t NE = hp->n_expert, NL = hp->n_layer, K = hp->n_expert_used;
        ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
        v3ctx = ggml_init(ip);
        slot_tab = ggml_new_tensor_2d(v3ctx, GGML_TYPE_I32, NE, NL);
        sel_log  = ggml_new_tensor_2d(v3ctx, GGML_TYPE_I32, (int64_t) K * DEC_NMAX, NL);
        v3buf = ggml_backend_alloc_ctx_tensors(v3ctx, gpu);
        require(v3buf != nullptr, "cannot allocate the v3 slot table");
        require(cudaHostAlloc((void **) &bell_h, sizeof(ggml_qfn_bell), cudaHostAllocMapped) == cudaSuccess, "cannot allocate the doorbell");
        require(cudaHostGetDevicePointer(&bell_d, bell_h, 0) == cudaSuccess, "cannot map the doorbell");
        std::memset((void *) bell_h, 0, sizeof(ggml_qfn_bell));
        bell_h->req = -1; bell_h->ack = -1; bell_h->error = 0;
        // bounded below the Windows GPU timeout (TDR, 2 s by default) so a stuck host cannot reset the shared GPU
        { const char * t = std::getenv("QFN_V3_BELL_MS"); bell_h->timeout_ms = t && *t ? std::max(1, std::min(1500, std::atoi(t))) : 500; }
        if (v3_future) {
            require(service_exec && service_h && service_control_h && !v3_replay,
                    "QFN_FUTURE_SERVICE: explicit mode unavailable; no alternate decode fallback");
            if (!side) require(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking) == cudaSuccess,
                               "future service stream re-creation failed");
            require(cudaEventCreateWithFlags(&bell_completion, cudaEventDisableTiming) == cudaSuccess,
                    "future service completion event allocation failed");
            log("[qfn] decode v3: one required CUDA model graph + persistent future service graph, CPU-only DS callback; canonical restore charged\n");
        } else if (v3_replay) {
            bell_h->timeout_ms = -1;
            require(cudaMalloc((void **) &replay_miss_latch, sizeof(int32_t)) == cudaSuccess,
                    "cannot allocate decode replay device atomic miss latch");
            alloc_replay_state();
            log("[qfn] decode replay device atomic latch; mapped miss record (no mapped-host atomics)\n");
            log("[qfn] decode replay decode v3: record-only misses, DirectStorage, restore + replay\n");
        } else {
            require(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking) == cudaSuccess, "cannot create the miss stream");
            require(cudaEventCreateWithFlags(&bell_completion, cudaEventDisableTiming) == cudaSuccess,
                    "cannot create the owned doorbell completion event before capture");
            bell_stop = false;
            bell_start_ready.store(0, std::memory_order_relaxed);
            bell_thread = std::thread([this]() {
                try {
                    require(cudaSetDevice(wf_get_cuda_device_index(wf)) == cudaSuccess,
                            "doorbell worker cannot attach to the engine CUDA device");
                    warm_bell_stream(); // warm BOTH this exact consumer stream and its real service thread
                    bell_start_ready.store(1, std::memory_order_release);
                    bell_loop();
                } catch (const std::exception & e) {
                    { std::lock_guard<std::mutex> lock(bell_err_mutex); bell_err = e.what(); }
                    bell_start_ready.store(-1, std::memory_order_release);
                }
            });
            while (bell_start_ready.load(std::memory_order_acquire) == 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (bell_start_ready.load(std::memory_order_acquire) < 0) {
                doorbell_poisoned = true;
                v3_shutdown();
                std::lock_guard<std::mutex> lock(bell_err_mutex);
                throw std::runtime_error("doorbell real side-stream warmup failed: " + bell_err);
            }
            log("[qfn] decode v3: static step graphs, device slot map, doorbell misses\n");
        }
    }

    void v3_shutdown() {
        if (v3_future && (service_epoch || service_live || service_unproven)) {
            if (service_unproven) {
                std::fprintf(stderr, "QFN_FUTURE_SERVICE_RELOAD_REQUIRED: completion remains unproven; process teardown required\n");
                std::fflush(stderr); std::_Exit(3);
            }
            stop_future_service(true);
            return;
        }
        if (!bell_thread.joinable()) return;
        bell_stop = true;
        bell_thread.join();
    }

    // Join before canonical exception restoration can touch the cache/RAM metadata.
    void quiesce_failed_bell(qfn_decode_stats & stats, uint64_t misses_before, uint64_t bytes_before, double ms_before) {
        doorbell_poisoned = true;
        v3_shutdown();
        if (side) require(cudaStreamSynchronize(side) == cudaSuccess, "failed doorbell side stream cannot quiesce; model reload required");
        stats.expert_misses += bell_miss.load() - misses_before;
        stats.miss_bytes += bell_bytes.load() - bytes_before;
        stats.miss_ms += bell_ms.load() - ms_before;
    }

    // the host half of the doorbell: serve each published miss, then acknowledge
    void bell_loop() {
        int32_t seen = -1;
        while (!bell_stop.load(std::memory_order_relaxed)) {
            const int32_t r = bell_h->req;
            if (r == seen || r < 0) { _mm_pause(); continue; }
            std::atomic_thread_fence(std::memory_order_acquire);
            const double tb = now_ms();
            const int32_t layer = bell_h->layer, count = bell_h->n;
            try {
                require(layer >= 0 && uint32_t(layer) < hp->n_layer && count > 0 && count <= 64 &&
                        r % 64 == layer, "invalid immutable doorbell miss record");
                const std::vector<int32_t> selected(bell_h->sel, bell_h->sel + count);
                serve_bell(r, uint32_t(layer), selected);
            } catch (const std::exception & e) {
                { std::lock_guard<std::mutex> lock(bell_err_mutex); bell_err = e.what(); }
                seen = r; // the bounded kernel timeout fails loudly; no false acknowledgement
                continue;
            }
            seen = r;
            std::atomic_thread_fence(std::memory_order_release);
            bell_h->ack = r;
            if (bell_served.load() <= 20) log("[qfn] v3 bell: request %d (layer %d, %d ids) served in %.1f ms\n", r, layer, count, now_ms() - tb);
        }
    }

    // place the published layer's missing experts: DirectStorage into a staging bank, copies to the victims' slots and
    // the slot-table entries on the side stream (the compute stream is held by the doorbell kernel meanwhile)
    void serve_bell(int32_t request, uint32_t il, const std::vector<int32_t> & sel) {
        const double t0 = now_ms();
        const uint32_t NE = hp->n_expert;
        const int n = int(sel.size());
        const uint64_t blob = pack.experts[il].blob;
        std::vector<int32_t> miss;
        for (int32_t e : sel) {
            require(e >= 0 && (uint32_t) e < NE, "router selected an invalid expert");
            if (slot_of[(size_t) il * NE + e] < 0 && std::find(miss.begin(), miss.end(), e) == miss.end()) miss.push_back(e);
        }
        if (miss.empty()) return;
        std::sort(miss.begin(), miss.end());
        std::vector<size_t> touched;
        miss_placement_parts parts;
        const uint64_t bytes = place_misses(il, miss, sel.data(), n, side, &touched, &parts);
        const double table_started = now_ms();
        for (size_t key : touched) {
            require(cudaMemcpyAsync((char *) slot_tab->data + key * 4, slot_of.data() + key, 4, cudaMemcpyHostToDevice, side) == cudaSuccess &&
                    cudaMemcpyAsync((char *) resident->data + key * 4, resident_host.data() + key, 4, cudaMemcpyHostToDevice, side) == cudaSuccess,
                    "slot table update failed");
        }
        const double sync_started = now_ms();
        require(cudaStreamSynchronize(side) == cudaSuccess, "miss stream sync failed");
        if (bell_served.load() < 20 || now_ms() - t0 > 100.0)
            log("[qfn] v3 bell parts req=%d layer=%u ids=%d misses=%zu bytes=%llu bank_wait_ms=%.3f disk_wall_ms=%.3f ds_submit_ms=%.3f ds_fence_ms=%.3f ds_verify_ms=%.3f cuda_handoff_ms=%.3f payload_enqueue_ms=%.3f table_enqueue_ms=%.3f side_sync_ms=%.3f requests=%llu total_ms=%.3f error_req=%d\n",
                request, il, n, miss.size(), (unsigned long long) bytes, parts.bank_wait_ms, parts.disk_wall_ms,
                parts.ds_ms[0], parts.ds_ms[1], parts.ds_ms[2], parts.ds_ms[3], parts.payload_enqueue_ms,
                sync_started - table_started, now_ms() - sync_started, (unsigned long long) parts.requests,
                now_ms() - t0, int(bell_h->error));
        bell_miss += miss.size();
        bell_bytes += bytes;
        bell_ms.store(bell_ms.load() + (now_ms() - t0));
        ++bell_served;
    }

    // QSA layer for the static decode graph: cache writes by row index, attention over the whole cache (the selection
    // ends at -1), indexer scores over every pool slot (the selection only reads the pools visible at each position),
    // and the step's blocks re-pooled from their members every step (an incomplete block is never read)
    ggml_tensor * qsa_layer_static(gctx & g, uint32_t il, ggml_tensor * x, const v3graph & G, int64_t nt) {
        ggml_context * ctx = g.ctx;
        const auto & L = layers[il];
        const int64_t D = hp->head_dim, H = hp->n_head, Hkv = hp->n_head_kv, Di = hp->idx_dim, Hi = hp->idx_heads, R = hp->kpool;
        const int64_t n_ctx = kcache[il]->ne[2], n_pool = idx_pool[il]->ne[1];
        ggml_tensor * k_raw = MM(ctx, L.idx_k_proj, x);                                  // [Di, nt]
        ggml_tensor * raw = ggml_set_rows(ctx, idx_raw[il], k_raw, G.kv_idx);
        ggml_tensor * rows = ggml_get_rows(ctx, raw, G.mem_idx);                                   // [Di, R*nbp] F32
        rows = ggml_reshape_3d(ctx, rows, Di, R, G.nbp);
        ggml_tensor * pooled = nullptr;
        for (int64_t i = 0; i < R; ++i) {
            ggml_tensor * s = ggml_view_2d(ctx, rows, Di, G.nbp, rows->nb[2], i * rows->nb[1]);
            pooled = pooled ? ggml_add(ctx, pooled, s) : ggml_cont(ctx, s);
        }
        pooled = ggml_scale(ctx, pooled, 1.0f / (float) R);
        pooled = ggml_mul(ctx, ggml_rms_norm(ctx, pooled, hp->rms_eps), L.idx_k_norm);
        pooled = rope(g, ggml_reshape_3d(ctx, pooled, Di, 1, G.nbp), G.pool_pos4);
        ggml_tensor * pool = ggml_set_rows(ctx, idx_pool[il], ggml_reshape_2d(ctx, pooled, Di, G.nbp), G.pool_idx);
        ggml_tensor * qi = ggml_reshape_3d(ctx, MM(ctx, L.idx_q_proj, x), Di, Hi, nt);
        qi = ggml_mul(ctx, ggml_rms_norm(ctx, qi, hp->rms_eps), L.idx_q_norm);
        qi = rope(g, qi, G.pos4);
        ggml_tensor * keys = ggml_view_4d(ctx, pool, Di, 1, n_pool, 1, pool->nb[1], pool->nb[1], pool->nb[1] * n_pool, 0);
        ggml_tensor * wts = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Hi, nt), 1.0f / sqrtf((float) Di));
        ggml_tensor * score = LIDX(ctx, ggml_reshape_4d(ctx, qi, Di, Hi, nt, 1), keys, wts, 1.0f, 1.0f, G.pos);
        score = ggml_reshape_2d(ctx, score, n_pool, nt);
        ggml_tensor * sel = ggml_qfn_qsa_select(ctx, score, G.pos, (int) (hp->idx_top_k / R), (int) R);
        ggml_tensor * qfull = MM(ctx, L.wq, x);
        ggml_tensor * q = ggml_view_3d(ctx, qfull, D, H, nt, ggml_element_size(qfull) * D * 2, ggml_element_size(qfull) * D * 2 * H, 0);
        q = ggml_mul(ctx, ggml_rms_norm(ctx, q, hp->rms_eps), L.q_norm);
        ggml_tensor * gate = ggml_cont(ctx, ggml_view_3d(ctx, qfull, D, H, nt, ggml_element_size(qfull) * D * 2,
                                                         ggml_element_size(qfull) * D * 2 * H, ggml_element_size(qfull) * D));
        ggml_tensor * k = ggml_reshape_3d(ctx, MM(ctx, L.wk, x), D, Hkv, nt);
        k = ggml_mul(ctx, ggml_rms_norm(ctx, k, hp->rms_eps), L.k_norm);
        ggml_tensor * v = ggml_reshape_3d(ctx, MM(ctx, L.wv, x), D, Hkv, nt);
        q = rope(g, q, G.pos4);
        k = rope(g, k, G.pos4);
        ggml_tensor * kc = ggml_set_rows(ctx, ggml_reshape_2d(ctx, kcache[il], D * Hkv, n_ctx), ggml_reshape_2d(ctx, k, D * Hkv, nt), G.kv_idx);
        ggml_tensor * vc = ggml_set_rows(ctx, ggml_reshape_2d(ctx, vcache[il], D * Hkv, n_ctx), ggml_reshape_2d(ctx, ggml_cont(ctx, v), D * Hkv, nt), G.kv_idx);
        kc = ggml_reshape_3d(ctx, kc, D, Hkv, n_ctx);
        vc = ggml_reshape_3d(ctx, vc, D, Hkv, n_ctx);
        ggml_tensor * att = attention(ctx, ggml_cont(ctx, q), gate, kc, vc, sel, G.pos, 1.0f / sqrtf((float) D));
        return MM(ctx, L.wo, ggml_reshape_2d(ctx, att, D * H, nt));
    }

    v3graph & v3_graph(int n, float bias, bool snap) {
        for (auto * G : v3g) if (G->n == n && G->bias == bias && G->snap == snap && G->logical_policy == verify_active) return *G;
        auto * G = new v3graph();
        G->n = n; G->bias = bias; G->snap = snap; G->logical_policy = verify_active; G->nbp = n > 1 ? 2 : 1;
        static const bool exact_env = [] { const char * e = std::getenv("QFN_EXACT_SPLIT"); return e && *e == '1'; }();
        split_tokens = n > 1 && (snap || exact_env);
        const int64_t E = hp->n_embd, HC = hp->hc, K = hp->n_expert_used, nt = n, R = hp->kpool;
        ggml_init_params ip = { 65536 * ggml_tensor_overhead() + ggml_graph_overhead_custom(65536, false), nullptr, true };
        G->ctx = ggml_init(ip);
        ggml_context * ctx = G->ctx;
        gctx g{ ctx, ggml_new_graph_custom(ctx, 65536, false) };
        G->emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, nt);
        G->pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nt);
        G->pos4 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * nt);
        G->kv_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, nt);
        G->mem_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, R * G->nbp);
        G->pool_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, G->nbp);
        G->pool_pos4 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * G->nbp);
        G->seq = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        for (ggml_tensor * t : { G->emb, G->pos, G->pos4, G->kv_idx, G->mem_idx, G->pool_idx, G->pool_pos4, G->seq }) {
            ggml_set_input(t);
            // INPUT reserves staging space but permits reuse after the last reader. A replay must retain the
            // original embeddings/positions/indices: OUTPUT pins these small allocations for the graph lifetime.
            ggml_set_output(t);
        }
        log("[qfn] decode replay pinned step inputs: embeddings, positions, indices and sequence survive replay\n");
        ggml_tensor * res = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, G->emb, E, 1, nt), E, HC, nt, 1);
        for (uint32_t il = 0; il < hp->n_layer; ++il) {
            if (hp->is_ple(il)) {
                ggml_tensor * idx = ggml_view_1d(ctx, ple_tok_idx, (int64_t) hp->ple_n_heads * nt, 0);
                ggml_tensor * pe = ggml_reshape_2d(ctx, ggml_get_rows(ctx, ple_tok, idx), (int64_t) hp->ple_dim * hp->ple_n_heads, nt);
                res = ple_block(g, il, res, pe, snap);
            }
            ggml_tensor * inj = nullptr;
            ggml_tensor * cur = hc_mix(g, res, layers[il].hc_attn_norm, layers[il].hc_attn_down, layers[il].hc_attn_up,
                                       layers[il].hc_attn_inject, &inj);
            cur = hp->is_recurrent(il) ? gdn_layer(g, il, cur, snap) : qsa_layer_static(g, il, cur, *G, nt);
            res = hc_combine(g, res, cur, inj);
            cur = hc_mix(g, res, layers[il].hc_ffn_norm, layers[il].hc_ffn_down, layers[il].hc_ffn_up, layers[il].hc_ffn_inject, &inj);
            route r = router(g, il, cur, bias);
            ggml_build_forward_expand(g.gf, ggml_cpy(ctx, ggml_reshape_1d(ctx, r.selected, K * nt),
                                                     ggml_view_1d(ctx, sel_log, K * nt, (size_t) il * sel_log->nb[1])));
            ggml_tensor * tab = ggml_view_1d(ctx, slot_tab, hp->n_expert, (size_t) il * slot_tab->nb[1]);
            ggml_tensor * slots = ggml_qfn_slot_map(ctx, r.selected, tab, G->seq, bell_d, (int) il);
            if (v3_replay) ggml_qfn_slot_map_set_record_latch(slots, replay_miss_latch);
            // One static graph still contains every token. Only indexed GEMMs are
            // packed into supported one-token nodes; dense/QSA/GDN batching stays intact.
            ggml_tensor * out = experts(g, il, cur, slots, r.weights, slab_gate, slab_up, slab_down, true, r.selected);
            res = hc_combine(g, res, out, inj);
        }
        ggml_build_forward_expand(g.gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, res, E * HC, nt),
                                                 ggml_view_2d(ctx, dec_hout, E * HC, nt, dec_hout->nb[1], 0)));
        ggml_tensor * hcur = hc_mix(g, res, out_hc_norm, out_hc_down, out_hc_up, nullptr, nullptr);
        G->logits = MM(ctx, output, hcur);
        ggml_set_output(G->logits);
        ggml_build_forward_expand(g.gf, G->logits);
        G->gf = g.gf;
        G->ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
        require(ggml_gallocr_alloc_graph(G->ga, G->gf), "cannot allocate a v3 decode graph (VRAM)");
        log("[qfn] decode v3 graph n=%d bias=%.3f: %d nodes, %.1f MiB\n", n, bias, ggml_graph_n_nodes(G->gf),
            ggml_gallocr_get_buffer_size(G->ga, 0) / 1048576.0);
        split_tokens = false;
        v3g.push_back(G);
        return *G;
    }

    void decode_v3_n(const int32_t * toks, int n, int64_t p, float bias, std::vector<float> & logits, qfn_decode_stats & st,
                     bool snapshots) {
        const int64_t E = hp->n_embd, K = hp->n_expert_used, R = hp->kpool, nt = n;
        const bool snap = snapshots && n > 1;
        require(!snap || !gdn_snap.empty(), "decode snapshots need the MTP buffers");
        const double tb = now_ms();
        const uint64_t warm_bytes_before = bell_warm_bytes;
        v3_init();
        // Warmup time is already in build_ms/TTFT; do not count its duration twice.
        // Actual expert bytes also enter the same existing IO ledger as other preparatory refetches.
        st.miss_bytes += bell_warm_bytes - warm_bytes_before;
        v3graph & G = v3_graph(n, bias, snap);
        st.build_ms += now_ms() - tb;
        const double ts = now_ms();
        // Pins are physical and last only for this full graph. The router keeps
        // its pre-step published rows (or verify_resident's canonical per-row masks),
        // never this temporary physical residency. V2 also publishes a row only
        // AFTER that row's router, so later-layer evictions cannot change its input.
        const bool canonical_replay = (v3_replay || v3_future) && !verify_active;
        const cache_policy initial_policy = canonical_replay ? capture_cache_policy() : cache_policy{};
        std::vector<uint8_t> replay_pins(v3_replay ? slot_of.size() : 0, 0);
        std::vector<int32_t> sl(size_t(K * DEC_NMAX) * hp->n_layer);
        if (!hp->ple_layers.empty()) load_ple_tokens(n);
        if (slot_tab_dirty) { ggml_backend_tensor_set(slot_tab, slot_of.data(), 0, slot_of.size() * 4); slot_tab_dirty = false; }
        if (resident_dirty) {
            ggml_backend_tensor_set(resident, resident_host.data(), 0, resident_host.size() * sizeof(float));
            if (v3_replay) replay_routing_resident = resident_host;
            if (v3_replay) log("[qfn] decode replay residency routing frozen across replay; v2 layer publication preserved\n");
            resident_dirty = false;
        }
        std::vector<float> e((size_t) (E * nt));
        std::vector<int32_t> pv((size_t) nt), p4((size_t) (4 * nt), 0);
        std::vector<int64_t> kv((size_t) nt);
        for (int64_t i = 0; i < nt; ++i) {
            image_embedding(toks[i],p+i,e.data()+i*E);
            pv[(size_t) i] = (int32_t) (p + i);
            p4[(size_t) i] = p4[(size_t) (nt + i)] = p4[(size_t) (2 * nt + i)] = (int32_t) (p + i);
            kv[(size_t) i] = p + i;
        }
        const int64_t b0 = p / R, b1 = (p + nt - 1) / R;
        std::vector<int32_t> mem((size_t) (R * G.nbp));
        std::vector<int64_t> pid((size_t) G.nbp);
        std::vector<int32_t> pp((size_t) (4 * G.nbp), 0);
        for (int j = 0; j < G.nbp; ++j) {
            const int64_t b = j == 0 ? b0 : b1;
            for (int64_t i = 0; i < R; ++i) mem[(size_t) (j * R + i)] = (int32_t) std::min<int64_t>(b * R + i, kcache.empty() ? 0 : (int64_t) opt.n_ctx - 1);
            pid[(size_t) j] = b;
            pp[(size_t) j] = pp[(size_t) (G.nbp + j)] = pp[(size_t) (2 * G.nbp + j)] = (int32_t) (b * R);
            if(!images.empty()){const auto v=image_position(b*R);for(int axis=0;axis<4;axis++)pp[size_t(axis*G.nbp+j)]=v[axis];}
        }
        require(v3_seq < (INT32_MAX - 63) / 64, "QFN_V3_SEQUENCE_EXHAUSTED: refusing request identity overflow; model reload required");
        const int32_t seqv = ++v3_seq;
        ggml_backend_tensor_set(G.emb, e.data(), 0, e.size() * 4);
        ggml_backend_tensor_set(G.pos, pv.data(), 0, pv.size() * 4);
        if(!images.empty())image_positions(p4,nt,p);
        ggml_backend_tensor_set(G.pos4, p4.data(), 0, p4.size() * 4);
        ggml_backend_tensor_set(G.kv_idx, kv.data(), 0, kv.size() * 8);
        ggml_backend_tensor_set(G.mem_idx, mem.data(), 0, mem.size() * 4);
        ggml_backend_tensor_set(G.pool_idx, pid.data(), 0, pid.size() * 8);
        ggml_backend_tensor_set(G.pool_pos4, pp.data(), 0, pp.size() * 4);
        ggml_backend_tensor_set(G.seq, &seqv, 0, 4);
        const uint64_t m0 = bell_miss.load(), by0 = bell_bytes.load();
        const double ms0 = bell_ms.load();
        bool future_stats_committed = false;
        try {
        if (v3_replay) copy_replay_state(false);
        uint32_t next_resident_row = hp->n_layer;
        auto finish_resident_rows = [&](uint32_t end) {
            // v2 publishes each row after that layer has routed, starting at the first miss. Preserve those
            // layer boundaries (including evictions of earlier layers), but publish only after final replay.
            if (next_resident_row < end) {
                const size_t begin = (size_t) next_resident_row * hp->n_expert, last = (size_t) end * hp->n_expert;
                std::copy(resident_host.begin() + begin, resident_host.begin() + last, replay_routing_resident.begin() + begin);
            }
        };
        st.stage_ms += now_ms() - ts;
        static const bool v3_cuda_graph = [] { const char * e = std::getenv("QFN_V3_CUDA_GRAPH"); return e && *e == '1'; }();
        for (uint32_t attempt = 0; ; ++attempt) {
            require(attempt < hp->n_layer + 1, "decode replay replay exhausted n_layer + 1 attempts");
            bell_h->error = 0;
            if (v3_replay) require(cudaMemsetAsync(replay_miss_latch, 0, sizeof(int32_t), stream()) == cudaSuccess,
                                   "cannot reset replay miss latch");
            const double tg = now_ms();
            ggml_status gs = GGML_STATUS_SUCCESS;
            if (v3_future) {
                try { execute_future_graph(G); }
                catch (...) { st.gpu_ms += now_ms() - tg; ++st.attempts; throw; }
            }
            else {
                if (v3_cuda_graph) ggml_backend_cuda_graphs_disable_pop(gpu);
                gs = v3_cuda_graph ? ggml_backend_cuda_graph_compute_static(gpu, G.gf)
                                   : ggml_backend_graph_compute(gpu, G.gf);
                if (v3_cuda_graph) ggml_backend_cuda_graphs_disable_push(gpu);
            }
            require(gs == GGML_STATUS_SUCCESS, "v3 decode graph compute failed");
            // Same completion edge, without a blocking CUDA runtime call holding the service launch lock.
            // Replay policy is unchanged. The persistent event is allocated before capture, recorded afterward.
            if (v3_future) { /* execute_future_graph already proved both graph streams idle */ }
            else if (v3_replay) ggml_backend_synchronize(gpu);
            else {
                require(bell_completion && cudaEventRecord(bell_completion, stream()) == cudaSuccess,
                        "cannot record mode1 graph completion");
                cudaError_t completion;
                while ((completion = cudaEventQuery(bell_completion)) == cudaErrorNotReady)
                    std::this_thread::yield();
                require(completion == cudaSuccess, "mode1 graph completion query failed");
            }
            st.gpu_ms += now_ms() - tg;
            ++st.attempts;
            if (bell_h->error == 0) break;
            if (!v3_replay) {
                quiesce_failed_bell(st, m0, by0, ms0);
                std::lock_guard<std::mutex> lock(bell_err_mutex);
                char msg[256];
                std::snprintf(msg, sizeof msg, "QFN_DOORBELL_TIMEOUT_RESET_REQUIRED: a decode miss was not served in time (error req %d, last req %d, ack %d, served %llu); fresh prefill required%s%s",
                              (int) bell_h->error, (int) bell_h->req, (int) bell_h->ack, (unsigned long long) bell_served.load(),
                              bell_err.empty() ? "" : ": ", bell_err.c_str());
                throw std::runtime_error(msg);
            }
            const double tm = now_ms();
            copy_replay_state(true);
            const uint32_t il = (uint32_t) bell_h->layer;
            const int ns = bell_h->n;
            require(il < hp->n_layer && ns > 0 && ns <= 64, "invalid replay miss record");
            std::vector<int32_t> selection(bell_h->sel, bell_h->sel + ns), misses;
            for (int32_t ex : selection) {
                require(ex >= 0 && (uint32_t) ex < hp->n_expert, "router selected an invalid expert");
                if (slot_of[(size_t) il * hp->n_expert + ex] < 0 &&
                    std::find(misses.begin(), misses.end(), ex) == misses.end()) misses.push_back(ex);
            }
            require(!misses.empty(), "replay miss record has no missing expert");
            std::sort(misses.begin(), misses.end());
            ggml_backend_tensor_get(sel_log, sl.data(), 0, sl.size() * sizeof(int32_t));
            pin_replay_prefix(replay_pins, sl, il, n);
            finish_resident_rows(il);
            next_resident_row = il;
            std::vector<size_t> touched;
            const uint64_t bytes = place_misses(il, misses, selection.data(), ns, stream(), &touched, nullptr, &replay_pins);
            for (size_t key : touched) {
                require(cudaMemcpyAsync((char *) slot_tab->data + key * 4, slot_of.data() + key, 4,
                                        cudaMemcpyHostToDevice, stream()) == cudaSuccess, "replay slot update failed");
            }
            // All attempts route against the same pre-step residency table, just as v2 does.
            require(cudaStreamSynchronize(stream()) == cudaSuccess, "replay miss sync failed");
            bell_miss += misses.size();
            bell_bytes += bytes;
            bell_ms.store(bell_ms.load() + now_ms() - tm);
            ++st.replays;
            if (st.replays <= 20 || attempt + 1 == hp->n_layer + 1) {
                log("[qfn] decode replay replay: attempt %u/%u, layer %u, misses %zu, total attempts %llu, replays %llu\n",
                    attempt + 1, hp->n_layer + 1, il, misses.size(),
                    (unsigned long long) st.attempts, (unsigned long long) st.replays);
                for (size_t key : touched) if (slot_of[key] < 0)
                    log("[qfn] decode replay eviction: layer %zu expert %zu\n", key / hp->n_expert, key % hp->n_expert);
            }
        }
        const double tr = now_ms();
        if (v3_replay && next_resident_row < hp->n_layer) {
            finish_resident_rows(hp->n_layer);
            ggml_backend_tensor_set(resident, replay_routing_resident.data(), 0, replay_routing_resident.size() * sizeof(float));
        }
        logits.resize((size_t) (hp->n_vocab * nt));
        ggml_backend_tensor_get(G.logits, logits.data(), 0, logits.size() * 4);
        // the host LFU sees every routed expert of the step
        ggml_backend_tensor_get(sel_log, sl.data(), 0, sl.size() * 4);
        if (verify_active) verify_selected = sl;
        const uint64_t misses = bell_miss.load() - m0;
        for (uint32_t il = 0; il < hp->n_layer; ++il) {
            for (int64_t i = 0; i < K * nt; ++i) {
                const int32_t ex = sl[(size_t) il * K * DEC_NMAX + (size_t) i];
                const int32_t s = slot_of[(size_t) il * hp->n_expert + ex];
                if (s >= 0) lfu_touch((uint32_t) s);
            }
        }
        st.route_ms += now_ms() - tr + bell_ms.load() - ms0;
        st.expert_misses += misses;
        st.expert_hits += (uint64_t) (K * nt) * hp->n_layer - std::min<uint64_t>(misses, (uint64_t) (K * nt) * hp->n_layer);
        st.miss_bytes += bell_bytes.load() - by0;
        st.miss_ms += bell_ms.load() - ms0;
        future_stats_committed = true;
        last_step_n = snap ? n : 0;
        // Temporary pins must not become a new LFU, residency-bias or RAM-tier
        // policy. Restore the ordinary V2 layer-major transaction, including any
        // displaced payloads, and charge its real reads/copies to the existing ledger.
        // MTP verification already commits its canonical token-major kept prefix.
        if (canonical_replay) commit_cache_policy(replay_final_policy(initial_policy, sl, n), st);
        else ++step;
        } catch (...) {
            if (v3_future) {
                doorbell_poisoned = true;
                if (service_epoch || service_live) stop_future_service(true);
                if (!future_stats_committed) {
                    st.expert_misses += bell_miss.load() - m0;
                    st.miss_bytes += bell_bytes.load() - by0;
                    st.miss_ms += bell_ms.load() - ms0;
                    st.route_ms += bell_ms.load() - ms0;
                }
            }
            throw;
        }
    }

    void alloc_snapshots() {
        if (!gdn_snap.empty()) return;
        const int64_t S = hp->ssm_state, Hv = hp->ssm_dt_rank;
        ggml_init_params ip = { (hp->n_layer + 4) * ggml_tensor_overhead(), nullptr, true };
        snap_ctx = ggml_init(ip);
        gdn_snap.assign(hp->n_layer, nullptr);
        for (uint32_t il = 0; il < hp->n_layer; ++il) {
            if (hp->is_recurrent(il)) gdn_snap[il] = ggml_new_tensor_2d(snap_ctx, GGML_TYPE_F32, S * S * Hv, DEC_NMAX - 1);
        }
        snap_buf = ggml_backend_alloc_ctx_tensors(snap_ctx, gpu);
        require(snap_buf != nullptr, "cannot allocate the decode state snapshots");
        log("[qfn] decode snapshots: %.1f MiB\n", ggml_backend_buffer_get_size(snap_buf) / 1048576.0);
    }

    int32_t sample(const float * l, const qfn_sampling & s, int64_t history = -1) {
        std::vector<std::pair<int32_t, double>> dist;
        filtered(l, s, dist, history);
        if (dist.size() == 1) return dist[0].first;
        double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
        int32_t last = -1;
        for (const auto & item : dist) {
            if (item.second <= 0) continue;
            last = item.first; u -= item.second; if (u < 0.0) return item.first;
        }
        return last;
    }

    bool feed(const std::vector<int32_t> & toks, std::vector<float> & logits, qfn_decode_stats & st, float bias) {
        require_live_context();
        const double t0 = now_ms();
        for (int32_t tok : toks) require(tok >= 0 && (uint32_t) tok < hp->n_vocab, "token id out of range");
        std::vector<float> all;
        for (size_t i = 0; i < toks.size(); i += DEC_NMAX) {
            const int n = (int) std::min<size_t>(DEC_NMAX, toks.size() - i);
            const int64_t p = (int64_t) seq.size();
            require(p + n <= (int64_t) opt.n_ctx, "context full");
            seq.insert(seq.end(), toks.begin() + (ptrdiff_t) i, toks.begin() + (ptrdiff_t) i + n);
            decode_n(toks.data() + i, n, p, bias, all, st, false);
            mtp_owe_decoded(toks.data() + i, n, p, 0);
            st.tokens += (uint64_t) n;
        }
        logits.assign(all.end() - hp->n_vocab, all.end());
        st.seconds += (now_ms() - t0) / 1000.0;
        return true;
    }

    bool decode(int32_t first, int n_max, const qfn_sampling & s,
                const std::function<bool(int32_t, const float *)> & on_token, qfn_decode_stats & st) {
        if (mtp_ok && mtp_drafts > 0) return decode_spec(first, n_max, s, on_token, st);
        const double t0 = now_ms();
        int32_t tok = first;
        std::vector<float> logits;
        for (int i = 0; i < n_max; ++i) {
            const int64_t p = (int64_t) seq.size();
            require(p < (int64_t) opt.n_ctx, "context full");
            seq.push_back(tok);
            decode_n(&tok, 1, p, s.residency_bias, logits, st, false);
            mtp_owe_decoded(&tok, 1, p, 0);
            ++st.tokens;
            if (!on_token(tok, logits.data())) break;
            if (i + 1 >= n_max) break;
            const int32_t next = sample(logits.data(), s);
            if (std::find(eog.begin(), eog.end(), next) != eog.end()) { on_token(next, nullptr); break; }
            tok = next;
        }
        st.seconds += (now_ms() - t0) / 1000.0;
        return true;
    }

    // fresh: a new sequence from position 0. Otherwise the tokens append to the live sequence (state kept).
    bool prefill(const std::vector<int32_t> & add, const qfn_logits_request & req, bool fresh) {
        const bool recovering = doorbell_poisoned;
        if (recovering) {
            require(fresh, "QFN_DOORBELL_RESET_REQUIRED: append cannot reset a failed mode1 context; use fresh prefill");
            require(!wf_has_failed(wf), "QFN_DOORBELL_RELOAD_REQUIRED: DirectStorage failed; reload the model");
            retire_poisoned_v3(); // retire only graphs/table/stream/event; keep exact expert capacity and owned buffers
            std::fill(ram_of.begin(), ram_of.end(), -1);
            std::fill(ram_owner.begin(), ram_owner.end(), -1);
            std::fill(ram_age.begin(), ram_age.end(), 0); ram_clock = 0;
            verify_prefix.clear(); verify_checkpoint_ready = false; last_step_n = 0;
            sampling_started = false;
            log("[qfn] mode1 failed context: fresh prefill rebuilds full recurrent/KV/MTP/cache state at unchanged capacity %u; poison clears only after success\n", n_slots);
        }
        if (fresh) {seq.clear();paired_rows=0;paired_anchor=paired_anchor_state{};}
        const int64_t p0 = (int64_t) seq.size();
        const int64_t n = p0 + (int64_t) add.size();
        require(!add.empty() && n <= (int64_t) opt.n_ctx, "prompt longer than the engine context");
        struct reference_scope {impl *owner;bool prior;uint32_t rows;~reference_scope(){owner->release_reference_weights();if(owner->reference_prefill_active)owner->retain_reference_host_buffers();owner->reference_prefill_active=prior;owner->reference_prefill_rows=rows;}} reference_reset{this,reference_prefill_active,reference_prefill_rows};
        reference_prefill_active=opt.reference_prefill_max_tokens&&add.size()<=opt.reference_prefill_max_tokens;
        reference_prefill_rows=reference_prefill_active?uint32_t(add.size()):0;
        if(reference_prefill_active) {
            uint64_t physical_free=0;require(nvml_free_bytes(physical_free),"Reference prefill physical VRAM query failed");
            const uint64_t required=uint64_t(hp->n_vocab)*hp->n_embd*2+(256ull<<20);
            require(physical_free>=required,"Reference prefill physical VRAM budget cannot admit its temporary head without spilling");
            log("[qfn] reference physical VRAM admitted: free=%llu required=%llu coefficients_f16=1 activations_f32=1\n",
                (unsigned long long)physical_free,(unsigned long long)required);
        }
        if(opt.reference_prefill_max_tokens)log("[qfn] reference prefill active=%d new_tokens=%zu max_tokens=%u dense_rows_budget=33554432 dense_arithmetic=f16-coefficients-f32-activations-accumulation dense_vector_columns=1_or_3 packed_vector_rows=4 weights_unchanged=1\n",
            int(reference_prefill_active),add.size(),opt.reference_prefill_max_tokens);

        validate_prompt(add, p0);
        const double t_start = now_ms();
        const bool has_ple = !hp->ple_layers.empty();
        // Warm serving and append start immediately at token arrival, before
        // clearing state or the decode cache. The cold CLI request was queued
        // in open(), before resident weight loading and state allocation.
        std::vector<int32_t> all = seq;
        all.insert(all.end(), add.begin(), add.end());
        if (has_ple) {
            if (ple_pending) {
                require(ple_pending->p0 == p0 && ple_pending->tokens == all, "cold PLE tokens differ from the prefill request");
                ple_pending->sweep_started.store(true);
            } else queue_ple(all, p0, false);
        } else start_weight_stream(false);
        if (fresh) {
            ggml_backend_buffer_clear(sbuf, 0);
            if (carry_buf) ggml_backend_buffer_clear(carry_buf, 0);
            if (carry_in_slab) ggml_backend_tensor_memset(carry, 0, 0, ggml_nbytes(carry));
        }
        if (cbuf) reset_cache();
        const char * mp = std::getenv("QFN_MOE_PASS");
        require(!mp || !*mp || std::strcmp(mp, "0") == 0 || std::strcmp(mp, "1") == 0, "QFN_MOE_PASS must be 0 or 1");
        moe_pass = mp && *mp == '1';
        const char * mc = std::getenv("QFN_MOE_CHUNK");
        require(!mc || !*mc || moe_pass, "QFN_MOE_CHUNK requires QFN_MOE_PASS=1; refusing to ignore the requested chunk");
        const char * rc = std::getenv("QFN_PREFILL_RELEASE_CACHE");
        require(!rc || !*rc || std::strcmp(rc, "0") == 0 || std::strcmp(rc, "1") == 0, "QFN_PREFILL_RELEASE_CACHE must be 0 or 1");
        const bool release_cache = rc && *rc ? *rc == '1' : moe_pass;
        if (release_cache) release_cache_for_prefill();
        alloc_carry();
        seq.insert(seq.end(), add.begin(), add.end());
        const std::vector<int32_t> & tok = seq;
        std::vector<int32_t> ple_slot;
        double ple_read_wait = 0, ple_upload = 0;
        bool ple_joined = !has_ple;
        // EARLY=0 or PROMPT_OVERLAP=0 explicitly retains synchronous adoption.
        if (has_ple && (!ple_prompt_overlap || !llama_dsv41_engram_early_enabled())) {
            join_ple(ple_slot, ple_read_wait, ple_upload);
            ple_joined = true;
        }
        const int64_t tile = opt.tile;
        require(tile > 0, "prefill tile must be positive");
        ensure_nvfp4_prefill_workspace(std::min(tile, n - p0));
        int64_t moe_chunk = n - p0;
        if (moe_pass) {
            alloc_moe_pass(p0, n - p0);
            size_t free_b = 0, total_b = 0;
            require(cudaMemGetInfo(&free_b, &total_b) == cudaSuccess, "MoE VRAM query failed");
            // Reuse ggml's exact size-only lifetime planner. The extra allowance
            // covers backend MMQ quantization/row maps and a fixed VRAM margin.
            auto fits = [&](int64_t count) {
                const uint64_t graph = moe_graph_bytes(count);
                const uint64_t wanted = nvfp4_prefill_required(count);
                const uint64_t extra = wanted > nvfp4_prefill_bytes ? wanted - nvfp4_prefill_bytes : 0;
                return graph + extra + (uint64_t) count * (16ull << 10) + (512ull << 20) <= free_b;
            };
            if (mc && *mc) {
                char * end = nullptr;
                errno = 0;
                moe_chunk = std::strtoll(mc, &end, 10);
                require(errno != ERANGE && end && !*end, "QFN_MOE_CHUNK must be a positive integer");
                require(moe_chunk > 0 && moe_chunk <= n - p0 && fits(moe_chunk), "QFN_MOE_CHUNK exceeds prompt/VRAM bounds or is not positive");
            } else if (!fits(moe_chunk)) {
                int64_t lo = 0, hi = (n - p0) / tile;
                while (lo < hi) {
                    const int64_t mid = (lo + hi + 1) / 2;
                    if (fits(mid * tile)) lo = mid; else hi = mid - 1;
                }
                moe_chunk = lo * tile;
            }
            require(moe_chunk > 0, "no VRAM for separate MoE pass");
            ensure_nvfp4_prefill_workspace(moe_chunk);
            log("[qfn] MoE size-only plan: %zu graph bytes + %llu backend allowance + 512 MiB margin, free %zu bytes\n",
                moe_graph_bytes(moe_chunk), (unsigned long long) (moe_chunk * (16ull << 10)), free_b);
            log("[qfn] separate MoE pass: attention tile %lld, expert chunk %lld, average %.3f rows/expert, stream-k unchanged\n",
                (long long) tile, (long long) moe_chunk, (double) moe_chunk * hp->n_expert_used / hp->n_expert);
        }
        double t_wait = 0, t_comp = 0;
        {
            static const bool cone = [] { const char * e = std::getenv("QFN_PREFILL_CONE"); return !(e && *e == '0'); }();
            const bool dump = std::getenv("QFN_DUMP_CARRY") && *std::getenv("QFN_DUMP_CARRY") == '1';
            pf_keep_from = !cone || mtp_ok || dump || req.all_top1 ? p0 : (req.full_from >= 0 ? std::min(req.full_from, n - 1) : n - 1);
        }
        for (uint32_t il = 0; il < hp->n_layer; ++il) {
            if (!ple_joined && hp->is_ple(il)) {
                // Layer 0 has no PLE dependency. Adopt before the first tile
                // of the first actual PLE layer, including future layouts.
                join_ple(ple_slot, ple_read_wait, ple_upload);
                ple_joined = true;
            }
            const double w0 = now_ms();
            void * base = wf_wait_layer_gpu(wf, (int) il);
            require(base != nullptr && !wf_has_failed(wf), "expert layer transfer failed");
            wf_cuda_wait_upload(wf, (int) il);
            const int bank = base == bank_ptr[0] ? 0 : (base == bank_ptr[1] ? 1 : -1);
            require(bank >= 0, "expert layer landed outside both banks");
            const double c0 = now_ms();
            t_wait += c0 - w0;
            for (int64_t t0 = p0; t0 < n; t0 += tile) {
                run_layer_tile(il, bank, t0, std::min(tile, n - t0), tok, ple_slot, p0);
            }
            if (moe_pass) {
                for (int64_t t0 = p0; t0 < n; t0 += moe_chunk) run_moe_pass(il, bank, t0, std::min(moe_chunk, n - t0));
            }
            if (cbuf) seed_layer(il, bank);
            if (const char * dd = std::getenv("QFN_DUMP_CARRY"); dd && *dd == '1') {
                // layer-by-layer check against the reference tool's DUMP l_last lines
                std::vector<float> r;
                const int kt = std::getenv("QFN_DUMP_TOKENS") ? std::atoi(std::getenv("QFN_DUMP_TOKENS")) : 4;
                for (int64_t t = p0; t < std::min<int64_t>(n, p0 + kt); ++t) {
                    carry_row(t, r);
                    double sum = 0, l2 = 0;
                    for (float x : r) { sum += x; l2 += (double) x * x; }
                    std::fprintf(stderr, "DUMP l_last-%u tok %lld n %zu sum %.6g l2 %.6g v %.6g %.6g %.6g %.6g\n", il, (long long) t, r.size(),
                                 sum, std::sqrt(l2), r[0], r[1], r[2], r[3]);
                }
            }
            t_comp += now_ms() - c0;
            wf_cuda_record_compute_done(wf, (int) il);
            wf_release_layer(wf, (int) il);
            require(!wf_has_failed(wf), "expert layer release failed");
            if(prefill_progress)prefill_progress(il+1,hp->n_layer,uint64_t(n-p0));
        }
        wf_stop(wf);
        wf_quiesce(wf);
        wf_running = false;
        if (moe_pass) free_moe_pass();
        free_nvfp4_prefill_workspace();
        llama_dsv41_engram_weight_stream(false);
        const double t_layers = now_ms();
        const double ple_exposed = ple_read_wait + ple_upload;
        // logits
        std::vector<int64_t> want;
        for (int64_t i = p0; i < n; ++i) if (req.all_top1 || (req.full_from >= 0 && i >= req.full_from)) want.push_back(i);
        std::vector<float> lg;
        for (size_t b = 0; b < want.size(); b += opt.head_batch) {
            std::vector<int64_t> rows(want.begin() + b, want.begin() + std::min(want.size(), b + opt.head_batch));
            run_head(rows, lg);
            const int64_t V = hp->n_vocab;
            for (size_t r = 0; r < rows.size(); ++r) {
                const float * l = lg.data() + r * V;
                if (req.all_top1 && req.on_top1) {
                    int32_t best = 0;
                    float mx = l[0];
                    for (int64_t v = 1; v < V; ++v) if (l[v] > mx) { mx = l[v]; best = (int32_t) v; }
                    double s = 0;
                    for (int64_t v = 0; v < V; ++v) s += std::exp((double) l[v] - mx);
                    req.on_top1(rows[r], best, (float) (mx + std::log(s)));
                }
                if (req.full_from >= 0 && rows[r] >= req.full_from && req.on_full) req.on_full(rows[r], l, V);
            }
        }
        log("[qfn] prefill ledger ms: tile graph build %.0f, input staging %.0f, GPU %.0f, routing readback %.0f\n",
            pf_build_ms, pf_stage_ms, pf_gpu_ms, pf_read_ms);
        pf_build_ms = pf_stage_ms = pf_gpu_ms = pf_read_ms = 0;
        log("[qfn] prefill %lld tokens: PLE %.0f ms, layers %.0f ms (expert wait %.0f ms, compute %.0f ms), head %.0f ms; "
            "%.1f tok/s over the layers\n", (long long) (n - p0), ple_exposed, t_layers - t_start - ple_exposed, t_wait, t_comp,
            now_ms() - t_layers, (n - p0) / ((t_layers - t_start) / 1000.0));
        handover_draft_pending = mtp_ok && opt.handover_prefill;
        if (mtp_ok && !handover_draft_pending) {
            const double tm = now_ms();
            mtp_prefill(p0);
            log("[qfn] MTP draft-layer prefill: %.0f ms\n", now_ms() - tm);
        }
        if (release_cache) restore_cache_after_prefill();
        if (recovering) doorbell_poisoned = false;
        if (opt.qsa_indexer != "scalar")
            log("[qfn] QSA candidate queued score dispatches: %llu (includes successful static graph replays)\n",
                (unsigned long long) ggml_backend_cuda_qfn_qsa_score_queued());
        return true;
    }
};

// ================================================================================================ public
qfn_engine::qfn_engine() : p(new impl) {}
qfn_engine::~qfn_engine() = default;

bool qfn_engine::set_vision(std::vector<qfn_image_span> images,uint32_t stable_prefix) {
    require(p->hp!=nullptr,"vision spans require a loaded QFN engine");
    uint64_t end=0;
    for(const auto &image:images) {
        require(image.count&&image.grid_height&&image.grid_width&&
            uint64_t(image.grid_height)*image.grid_width==image.count&&image.position>=end&&
            uint64_t(image.position)+image.count<=p->opt.n_ctx&&
            image.values.size()==size_t(image.count)*p->hp->n_embd,"invalid QFN projected image geometry");
        require(std::all_of(image.values.begin(),image.values.end(),[](float v){return std::isfinite(v);}),"nonfinite QFN image projection");
        end=uint64_t(image.position)+image.count;
    }
    bool changed=images.size()!=p->images.size();
    for(size_t i=0;!changed&&i<images.size();i++) {
        const auto &a=images[i],&b=p->images[i];
        changed=a.position!=b.position||a.count!=b.count||a.grid_height!=b.grid_height||
            a.grid_width!=b.grid_width||a.fingerprint!=b.fingerprint||a.values!=b.values;
    }
    if(stable_prefix) {
        size_t old=0,now=0;
        while(old<p->images.size()&&p->images[old].position<stable_prefix)old++;
        while(now<images.size()&&images[now].position<stable_prefix)now++;
        require(old==now,"paired vision append changes an existing prefix image");
        for(size_t i=0;i<old;i++) {
            const auto &a=images[i],&b=p->images[i];
            require(a.position==b.position&&a.count==b.count&&a.grid_height==b.grid_height&&
                a.grid_width==b.grid_width&&a.fingerprint==b.fingerprint&&a.values==b.values,
                "paired vision append changes a retained projection");
        }
    }
    p->images=std::move(images);return changed;
}

const qfn_hparams & qfn_engine::hparams() const { return *p->hp; }

bool qfn_engine::open(const qfn_options & opts, std::string & err, const std::vector<int32_t> & cold_prompt) {
    try {
        p->opt = opts;
        if (const char * e = std::getenv("QFN_FUSED_EXPERT_REDUCE")) {
            require(!std::strcmp(e, "0") || !std::strcmp(e, "1"), "QFN_FUSED_EXPERT_REDUCE must be 0 or 1");
            p->fused_expert_reduce = !std::strcmp(e, "1");
        }
        p->log("[qfn] fused expert weighted reduction: %d (F32 products, ascending slot sum)\n", int(p->fused_expert_reduce));
        if (const char * e = std::getenv("QFN_LFU_HALF_LIFE"); e && *e) p->opt.lfu_half_life = (float) std::atof(e);
        if (const char * e = std::getenv("QFN_PAIRED_GDN")) {
            require(!std::strcmp(e, "0") || !std::strcmp(e, "1"), "QFN_PAIRED_GDN must be 0 or 1");
            p->paired_gdn = !std::strcmp(e, "1");
        }
        // Shared Engram settings, QFN-scoped measured queue defaults.
        // Explicit values always win; retain the DSV4.1 auto geometry selector.
        // Native NVFP4 at 100K: 4 MiB / 64 KiB scans avoid wide-gap over-read.
        for (const auto & setting : {std::pair<const char *, const char *>{"LLAMADECODE_ENGRAM_EARLY", "1"},
                {"LLAMADECODE_ENGRAM_RINGS", "4"}, {"LLAMADECODE_ENGRAM_DEPTH", "128"},
                {"LLAMADECODE_ENGRAM_SCAN_RINGS", "1"}, {"LLAMADECODE_ENGRAM_SCAN_DEPTH", "64"},
                {"LLAMADECODE_ENGRAM_SCAN_KIB", "4096"}, {"LLAMADECODE_ENGRAM_SCAN_GAP_KIB", "64"},
                {"LLAMADECODE_ENGRAM_SCAN_GBPS", "15"}}) {
            if (!std::getenv(setting.first)) _putenv_s(setting.first, setting.second);
        }
        if (const char * e = std::getenv("LLAMA_DSV41_ENGRAM_PROMPT_OVERLAP")) {
            require(!std::strcmp(e, "0") || !std::strcmp(e, "1"), "LLAMA_DSV41_ENGRAM_PROMPT_OVERLAP must be 0 or 1");
            p->ple_prompt_overlap = !std::strcmp(e, "1");
        }
        (void) llama_dsv41_engram_early_enabled(); // validate even before a request arrives
        // expert-miss extents: one DirectStorage request per extent part (a 1.32 MiB expert is one request,
        // >= 256 KiB as the RAID measurements require); the fetcher's default splits into 8 parts
        if (!std::getenv("DS4_DS_READ_PARTS")) _putenv_s("DS4_DS_READ_PARTS", "8");
        require(p->pack.open(opts.model, err), err);
        p->hp = &p->pack.hp;
        // Native-packed NVFP4 decode default, the DSV4.1 method: packed-vector FP4 kernels on the bytes prefill streams,
        // one model graph and one service graph per token. Explicit settings win; MTP keeps the segmented path.
        if (p->pack.native_nvfp4_packed && !p->opt.nvfp4_decode_mode_explicit) {
            p->opt.nvfp4_decode_mode = 6;
            p->opt.nvfp4_decode_mode_explicit = true;
        }
        if (p->pack.native_nvfp4_packed && p->opt.nvfp4_decode_mode == 6) {
            const char * residual = std::getenv("QFN_NATIVE_NVFP4_DECODE_RESIDUAL");
            if (!std::getenv("QFN_NATIVE_NVFP4_DECODE_VECTOR") && !(residual && *residual == '1')) _putenv_s("QFN_NATIVE_NVFP4_DECODE_VECTOR", "1");
            const char * mtp = std::getenv("QFN_MTP");
            if (!(mtp && *mtp && *mtp != '0') && !std::getenv("QFN_DECODE_V3")) {
                _putenv_s("QFN_DECODE_V3", "1");
                _putenv_s("QFN_V3_FUTURE_SERVICE", "1");
                _putenv_s("QFN_V3_CUDA_GRAPH", "1");
            }
        }
        p->admit_qsa_indexer(); // independent of expert format; before CUDA/model allocations
        p->admit_nvfp4(); // DLL/ABI/format/workspace validation precedes CUDA and model allocations
        if (!p->nvfp4) require(p->pack.runtime_supported(err), err);
        p->admit_attention(); // independent DLL/ABI geometry preflight before CUDA/model allocation
        p->log("[qfn] pack %s: %u layers, %u experts top-%u, hidden %u x hc %u, vocab %u\n", opts.model.c_str(),
               p->hp->n_layer, p->hp->n_expert, p->hp->n_expert_used, p->hp->n_embd, p->hp->hc, p->hp->n_vocab);
        p->gpu = ggml_backend_cuda_init(0);
        require(p->gpu != nullptr, "CUDA backend init failed");
        p->prepare_qsa_indexer(); // before any graph capture; works with Q2 main experts
        p->alloc_nvfp4_workspace(); // before cache sizing and any required static graph capture
        ggml_backend_cuda_graphs_disable_push(p->gpu);  // per-tile graphs change shape: node-by-node path
        p->alloc_attention_workspace(); // stable decode scratch before model/cache autosizing and capture
        p->open_experts();
        if (!cold_prompt.empty()) {
            p->validate_prompt(cold_prompt);
            if (!p->hp->ple_layers.empty() && p->ple_prompt_overlap && llama_dsv41_engram_early_enabled())
                p->queue_ple(cold_prompt, 0, true);
        }
        p->load_weights();
        p->alloc_state();
        {
            // the drafter and its snapshots take their VRAM before the expert cache sizes itself from what is left
            const char * m = std::getenv("QFN_MTP");
            if (m && *m && *m != '0') p->load_mtp(); // owner default off; --mtp 1 remains fully supported
        }
        if (const char * v3 = std::getenv("QFN_DECODE_V3")) {
            p->v3_replay = std::strcmp(v3, "replay") == 0;
            p->v3_on = p->v3_replay || std::strcmp(v3, "1") == 0;
        }
        if (const char * future = std::getenv("QFN_V3_FUTURE_SERVICE")) {
            require(!std::strcmp(future, "0") || !std::strcmp(future, "1"), "QFN_V3_FUTURE_SERVICE must be 0 or 1");
            p->v3_future = !std::strcmp(future, "1");
            require(!p->v3_future || (p->v3_on && !p->v3_replay),
                    "QFN_V3_FUTURE_SERVICE1 requires QFN_DECODE_V3=1; no replay/eager substitution");
            const char * capture = std::getenv("QFN_V3_CUDA_GRAPH");
            require(!p->v3_future || (capture && !std::strcmp(capture, "1")),
                    "QFN_V3_FUTURE_SERVICE1 requires QFN_V3_CUDA_GRAPH1");
        }
        p->prepare_future_service(); // graph/fence/mappings admitted before automatic cache sizing
        if (p->v3_future) p->alloc_ram_tier(); // mapped RAM page tables are part of this mode's real admission cost
        p->alloc_cache();
        if (!p->v3_future) p->alloc_ram_tier();
        if (!p->hp->ple_layers.empty()) p->open_ple_reader();
        p->eog = p->pack.eos_tokens();
        p->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(p->gpu));
        return true;
    } catch (const std::exception & e) {
        p->cancel_prefill();
        err = e.what();
        return false;
    }
}

bool qfn_engine::decode(int32_t first, int n_max, const qfn_sampling & s,
                        const std::function<bool(int32_t, const float *)> & on_token, qfn_decode_stats & st, std::string & err) {
    const uint64_t ram_start = p->ram_hits;
    try {
        if (!p->sampling_started || p->sampling_seed != s.seed) begin_sampling(s);
        p->profile_begin();
        double ds_before[4] = {}, ds_after[4] = {};
        uint64_t requests_before = 0, requests_after = 0, batches_before = 0, batches_after = 0;
        wf_extent_fetch_stats(ds_before, &requests_before, &batches_before);
        p->bell_prepare_ms = p->bell_fetch_ms = p->bell_ack_ms = 0;
        if (const char * cal = std::getenv("QFN_AFFINITY_CALIBRATE"); cal && *cal && s.residency_bias == 0) {
            require(!p->v3_on, "QFN_AFFINITY_CALIBRATE uses the segmented decode path");
            p->affinity_calibrating = true;
            p->calibrate_sum.assign(p->hp->n_layer, 0.0);
            p->calibrate_n.assign(p->hp->n_layer, 0);
        }
        if (p->handover_draft_pending) {
            p->mtp_prefill(0);p->handover_draft_pending=false;
        }
        const bool ok = p->decode(first, n_max, s, on_token, st);
        if (p->affinity_calibrating) { p->affinity_calibrating = false; p->calibrate_write(); }
        wf_extent_fetch_stats(ds_after, &requests_after, &batches_after);
        p->log("[qfn] decode miss ledger: %d tokens max, %llu misses, %llu read batches, %llu requests, ms submit %.1f fence %.1f verify %.1f cuda %.1f; service prepare %.1f fetch %.1f gpu-ack %.1f; cache %u slots; graph high-water %.1f MiB\n",
               n_max, (unsigned long long) st.expert_misses, (unsigned long long) (batches_after - batches_before), (unsigned long long) (requests_after - requests_before),
               ds_after[0] - ds_before[0], ds_after[1] - ds_before[1], ds_after[2] - ds_before[2], ds_after[3] - ds_before[3], p->bell_prepare_ms, p->bell_fetch_ms, p->bell_ack_ms, p->n_slots, p->graph_high_water / 1048576.0);
        p->profile_end(s.residency_bias == 0 ? "decode-quality" : "decode-speed");
        st.ram_tier_hits += p->ram_hits - ram_start;
        p->sampling_started = false;
        if (p->opt.qsa_indexer != "scalar")
            p->log("[qfn] QSA candidate queued score dispatches after decode: %llu\n",
                   (unsigned long long) ggml_backend_cuda_qfn_qsa_score_queued());
        return ok;
    } catch (const std::exception & e) {
        st.ram_tier_hits += p->ram_hits - ram_start;
        p->sampling_started = false;
        err = e.what();
        return false;
    }
}

void qfn_engine::begin_sampling(const qfn_sampling & s) {
    p->require_live_context();
    p->rng.seed(s.seed);
    p->sampling_seed = s.seed;
    p->sampling_started = true;
}

int32_t qfn_engine::sample(const float * logits, const qfn_sampling & s) {
    p->require_live_context();
    if (!p->sampling_started || p->sampling_seed != s.seed) begin_sampling(s);
    return p->sample(logits, s);
}

bool qfn_engine::feed(const std::vector<int32_t> & tokens, std::vector<float> & last_logits, qfn_decode_stats & st, std::string & err, float bias) {
    const uint64_t ram_start = p->ram_hits;
    try {
        p->profile_begin();
        if (p->handover_draft_pending) {
            p->mtp_prefill(0);p->handover_draft_pending=false;
        }
        const bool ok = p->feed(tokens, last_logits, st, bias);
        p->profile_end("feed");
        st.ram_tier_hits += p->ram_hits - ram_start;
        if (p->opt.qsa_indexer != "scalar")
            p->log("[qfn] QSA candidate queued score dispatches after feed: %llu\n",
                   (unsigned long long) ggml_backend_cuda_qfn_qsa_score_queued());
        return ok;
    } catch (const std::exception & e) {
        st.ram_tier_hits += p->ram_hits - ram_start;
        err = e.what();
        return false;
    }
}

const std::vector<std::string> & qfn_engine::vocab() const { return p->pack.vocab; }

bool qfn_engine::is_eog(int32_t token) const { return std::find(p->eog.begin(), p->eog.end(), token) != p->eog.end(); }

bool qfn_engine::append(const std::vector<int32_t> & tokens, const qfn_logits_request & req, std::string & err) {
    try {
        p->profile_begin();
        const bool ok = p->prefill(tokens, req, false);
        p->profile_end("append");
        return ok;
    } catch (const std::exception & e) {
        p->cancel_prefill();
        err = e.what();
        return false;
    }
}

bool qfn_engine::reset_required() const { return p->doorbell_poisoned; }
int64_t qfn_engine::n_past() const { p->require_live_context(); return (int64_t) p->seq.size(); }
uint32_t qfn_engine::paired_mtp_rows() const {p->require_live_context();return p->paired_rows;}

bool qfn_engine::export_state(const std::string &path, const std::vector<float> &logits,
                              uint64_t max_bytes, uint64_t &bytes, std::string &err, bool narrow_index,
                              const std::function<void(const void *,size_t)> &sink,
                              const std::function<void(uint64_t,uint64_t,uint32_t)> &ready,
                              double *gpu_read_seconds) {
    bytes = 0;
    if(gpu_read_seconds)*gpu_read_seconds=0;
    FILE *fp = nullptr;
    try {
        p->require_live_context();
        require(!p->seq.empty() && p->seq.size() <= UINT32_MAX, "QFN snapshot needs a live sequence");
        const auto &h = *p->hp;
        require(h.ple_ngram <= 8 && h.kpool == 4, "QFN geometry differs from Mac codec");
        qfn_state::shape s{p->opt.n_ctx, p->opt.tile, uint32_t(p->seq.size()), h.n_layer,
            h.n_embd, h.hc, h.head_dim, h.n_head_kv, h.n_vocab, h.full_attn_interval,
            h.ssm_state, h.ssm_dt_rank, h.ssm_state * (2 * h.ssm_group + h.ssm_dt_rank),
            h.ssm_conv - 1, h.idx_dim, h.kpool, (h.ple_conv - 1) * h.ple_ngram, h.ple_eos,
            !p->pack.mtp_tensors.empty(), p->mtp_ok ? uint32_t(p->seq.size()) : 0};
        const uint64_t expected = qfn_state::payload_bytes(s, narrow_index);
        require(expected <= max_bytes && logits.size() == h.n_vocab, "QFN snapshot budget/logit extent differs");
        // Finish owed draft rows using retained target residuals, never skip them.
        if (p->mtp_ok && !p->handover_draft_pending) p->mtp_flush();
        ggml_backend_synchronize(p->gpu);
        p->paired_prepare(logits);
        auto read = [&](const std::string &owner, uint32_t il, uint64_t off, void *dst, size_t n) {
            ggml_tensor *t = nullptr;
            if (owner == "gdn") t = p->gdn_state.at(il);
            else if (owner == "conv") t = p->conv_state.at(il);
            else if (owner == "key") t = il == h.n_layer ? p->paired_mtp[0] : p->kcache.at(il);
            else if (owner == "value") t = il == h.n_layer ? p->paired_mtp[1] : p->vcache.at(il);
            else if (owner == "index") t = il == h.n_layer ? p->paired_mtp[2] : p->idx_raw.at(il);
            else if (owner == "pool") t = il == h.n_layer ? p->paired_mtp[3] : p->idx_pool.at(il);
            else if (owner == "ple") t = p->ple_conv;
            require(t && ggml_is_contiguous(t) && off <= ggml_nbytes(t) && n <= ggml_nbytes(t) - off,
                    "QFN snapshot owner missing, strided or too short");
            const double started=now_ms();
            ggml_backend_tensor_get(t, dst, size_t(off), n);
            if(gpu_read_seconds)*gpu_read_seconds+=(now_ms()-started)/1000.0;
        };
        if (sink) {
            if (ready) ready(expected,qfn_state::payload_bytes(s),qfn_state::draft_rows(s));
            qfn_state::write_to(sink,s,p->seq,logits,max_bytes,read,
                [](uint16_t x) { return ggml_fp16_to_fp32(x); },narrow_index,p->image_delta(),
                p->images.empty()?qfn_state::position_reader{}:qfn_state::position_reader{[&](uint32_t i){return p->image_position(i);}});
            bytes=expected;return true;
        }
        require(!narrow_index,"Narrow index is a stream encoding; file snapshots remain canonical");
        int fd = -1;
        const auto wide = std::filesystem::u8path(path).wstring();
        require(_wsopen_s(&fd, wide.c_str(), _O_CREAT | _O_EXCL | _O_WRONLY | _O_BINARY,
                         _SH_DENYRW, _S_IREAD | _S_IWRITE) == 0, "QFN snapshot output must be a new writable file");
        fp = _fdopen(fd, "wb");
        if (!fp) { _close(fd); throw std::runtime_error("QFN snapshot fdopen failed"); }
        qfn_state::write(fp, s, p->seq, logits, max_bytes, read,
                         [](uint16_t x) { return ggml_fp16_to_fp32(x); },p->image_delta(),
                         p->images.empty()?qfn_state::position_reader{}:qfn_state::position_reader{[&](uint32_t i){return p->image_position(i);}});
        const int closed = fclose(fp); fp = nullptr;
        require(closed == 0, "QFN snapshot close failed");

        bytes = expected;
        return true;
    } catch (const std::exception &e) {
        if (fp) fclose(fp);

        err = e.what(); return false;
    }
}

bool qfn_engine::export_delta(uint32_t base,uint32_t base_mtp,const std::vector<float> &logits,
        uint64_t budget,uint64_t &bytes,const std::function<void(const void *,size_t)> &sink,
        uint32_t &target_mtp,std::string &err) {
    bytes=0;
    try {
        p->require_live_context();require(sink&&logits.size()==p->hp->n_vocab,"delta requires sink and complete logits");
        for(float v:logits)require(std::isfinite(v),"nonfinite delta logits");
        const auto shape=p->paired_shape();qfn_native_plan plan{};
        target_mtp=p->mtp_ok?uint32_t(p->seq.size()):0;
        require(qfn_native_geometry(&shape,base,uint32_t(p->seq.size()),base_mtp,target_mtp,0,&plan)&&
            plan.bytes<=UINT64_MAX-40&&plan.bytes+40<=budget,"delta geometry or byte budget differs");
        p->paired_prepare(logits);
        const uint64_t extent=plan.bytes+40;
        const uint32_t header[10]={0x444e4651,1,p->opt.n_ctx,base,plan.target,base_mtp,target_mtp,0,
            uint32_t(extent),uint32_t(extent>>32)};
        sink(header,sizeof(header));sink(p->seq.data()+base,size_t(plan.target-base)*4);
        sink(logits.data(),logits.size()*4);
        std::vector<uint8_t> page(65536);
        for(uint32_t i=0;i<plan.count;i++) {
            const auto &r=plan.slices[i];
            if(r.owner==QFN_CONV||r.owner==QFN_PLE) {
                std::vector<float> history(size_t(r.bytes/4));p->paired_owner(r,history.data(),false);
                sink(history.data(),size_t(r.bytes));continue;
            }
            for(uint64_t at=0;at<r.bytes;) {
                const size_t n=size_t(std::min<uint64_t>(page.size(),r.bytes-at));
                const qfn_native_slice part={r.owner,r.layer,r.offset+at,n};
                p->paired_owner(part,page.data(),false);sink(page.data(),n);at+=n;
            }
        }
        bytes=extent;return true;
    }catch(const std::exception &e){err=e.what();return false;}
}

bool qfn_engine::apply_decode_delta(uint32_t base,uint32_t target,uint32_t base_mtp,uint32_t target_mtp,
        uint64_t bytes,uint64_t budget,const std::function<void(void *,size_t)> &read,
        std::vector<int32_t> &tokens,std::vector<float> &logits,std::string &err) {
    bool mutated=false;
    try {
        p->require_live_context();require(read&&p->seq.size()==base&&p->paired_rows==base_mtp,
            "reverse delta base owner frontier differs");
        const auto shape=p->paired_shape();qfn_native_plan plan{};
        require(qfn_native_geometry(&shape,base,target,base_mtp,target_mtp,1,&plan)&&
            plan.bytes<=UINT64_MAX-40&&bytes==plan.bytes+40&&bytes<=budget,"reverse delta extent or budget differs");
        uint32_t header[10];read(header,sizeof(header));
        require(header[0]==0x444e4651&&header[1]==1&&header[2]>=target&&header[3]==base&&
            header[4]==target&&header[5]==base_mtp&&header[6]==target_mtp&&header[7]==1&&
            ((uint64_t(header[9])<<32)|header[8])==bytes,"reverse delta header identity differs");
        std::vector<int32_t> add(target-base);std::vector<float> lg(shape.vocab);
        read(add.data(),add.size()*4);read(lg.data(),lg.size()*4);
        for(int32_t t:add)require(t>=0&&uint32_t(t)<shape.vocab,"reverse delta token outside vocabulary");
        for(float v:lg)require(std::isfinite(v),"nonfinite reverse delta logits");
        p->alloc_carry();p->paired_allocate();ggml_backend_synchronize(p->gpu);
        mutated=true;p->seq.insert(p->seq.end(),add.begin(),add.end());
        std::vector<uint8_t> page(65536);
        for(uint32_t i=0;i<plan.count;i++) {
            const auto &r=plan.slices[i];
            if(r.owner==QFN_CONV||r.owner==QFN_PLE) {
                std::vector<float> history(size_t(r.bytes/4));read(history.data(),size_t(r.bytes));
                for(float v:history)require(std::isfinite(v),"nonfinite reverse history owner");
                p->paired_owner(r,history.data(),true);continue;
            }
            for(uint64_t at=0;at<r.bytes;) {
                const size_t n=size_t(std::min<uint64_t>(page.size(),r.bytes-at));read(page.data(),n);
                if(r.owner==QFN_GDN||r.owner==QFN_INDEX||r.owner==QFN_CARRY) {
                    require(n%4==0,"reverse float owner alignment differs");
                    for(size_t j=0;j<n;j+=4){float v;std::memcpy(&v,page.data()+j,4);require(std::isfinite(v),"nonfinite reverse float owner");}
                }else if(r.owner==QFN_KEY||r.owner==QFN_VALUE||r.owner==QFN_POOL) {
                    require(n%2==0,"reverse half owner alignment differs");
                    for(size_t j=0;j<n;j+=2){uint16_t v;std::memcpy(&v,page.data()+j,2);require((v&0x7c00)!=0x7c00,"nonfinite reverse half owner");}
                }
                const qfn_native_slice part={r.owner,r.layer,r.offset+at,n};p->paired_owner(part,page.data(),true);at+=n;
            }
        }
        ggml_backend_synchronize(p->gpu);p->paired_rows=target_mtp;
        p->mtp_pend.clear();p->last_step_n=0;p->verify_checkpoint_ready=false;p->sampling_started=false;
        p->handover_draft_pending=p->mtp_ok; // source-aligned drafter is rebuilt only if CUDA local decode is requested
        if(p->mtp_ok)p->carry_row(target-1,p->h_last);
        tokens=p->seq;logits=std::move(lg);return true;
    }catch(const std::exception &e){if(mutated)p->doorbell_poisoned=true;err=e.what();return false;}
}

bool qfn_engine::save_paired_anchor(const std::vector<float> &logits,uint64_t budget,uint64_t &bytes,std::string &err) {
    bytes=0;
    try {
        p->require_live_context();const auto shape=p->paired_shape();impl::paired_anchor_state next;
        require(logits.size()==shape.vocab&&qfn_native_anchor_geometry(&shape,uint32_t(p->seq.size()),p->paired_rows,&next.plan),
            "paired anchor geometry/logits differ");
        require(next.plan.bytes+logits.size()*4+p->seq.size()*4<=budget,"paired anchor exceeds explicit host byte budget");
        next.tokens=p->seq;next.logits=logits;next.images=p->images;next.owners.resize(size_t(next.plan.bytes));
        ggml_backend_synchronize(p->gpu);size_t cursor=0;
        for(uint32_t i=0;i<next.plan.count;i++) {
            const auto &r=next.plan.slices[i];
            if(r.owner==QFN_CONV||r.owner==QFN_PLE){p->paired_owner(r,next.owners.data()+cursor,false);cursor+=size_t(r.bytes);continue;}
            for(uint64_t off=0;off<r.bytes;) {
                const size_t n=size_t(std::min<uint64_t>(65536,r.bytes-off));
                const qfn_native_slice part={r.owner,r.layer,r.offset+off,n};p->paired_owner(part,next.owners.data()+cursor,false);cursor+=n;off+=n;
            }
        }
        require(cursor==next.owners.size(),"paired anchor owner extent differs");
        bytes=next.owners.size()+next.tokens.size()*4+next.logits.size()*4;p->paired_anchor=std::move(next);return true;
    }catch(const std::exception &e){err=e.what();return false;}
}

bool qfn_engine::rewind_paired_anchor(std::vector<int32_t> &tokens,std::vector<float> &logits,std::string &err,bool test_owner_readback) {
    bool mutated=false;
    try {
        p->require_live_context();const auto &a=p->paired_anchor;
        require(!a.owners.empty()&&a.tokens.size()<=p->seq.size()&&std::equal(a.tokens.begin(),a.tokens.end(),p->seq.begin()),
            "paired rewind anchor prefix is unavailable");
        ggml_backend_synchronize(p->gpu);mutated=true;p->seq=a.tokens;p->images=a.images;size_t cursor=0;
        for(uint32_t i=0;i<a.plan.count;i++) {
            const auto &r=a.plan.slices[i];
            if(r.owner==QFN_CONV||r.owner==QFN_PLE){p->paired_owner(r,const_cast<uint8_t*>(a.owners.data()+cursor),true);cursor+=size_t(r.bytes);continue;}
            for(uint64_t off=0;off<r.bytes;) {
                const size_t n=size_t(std::min<uint64_t>(65536,r.bytes-off));
                const qfn_native_slice part={r.owner,r.layer,r.offset+off,n};p->paired_owner(part,const_cast<uint8_t*>(a.owners.data()+cursor),true);cursor+=n;off+=n;
            }
        }
        require(cursor==a.owners.size(),"paired rewind extent differs");ggml_backend_synchronize(p->gpu);
        if(test_owner_readback) {
            size_t checked=0;
            for(uint32_t i=0;i<a.plan.count;i++) {
                const auto &r=a.plan.slices[i];
                if(r.owner==QFN_CONV||r.owner==QFN_PLE) {
                    std::vector<uint8_t> page(size_t(r.bytes));p->paired_owner(r,page.data(),false);
                    require(!std::memcmp(page.data(),a.owners.data()+checked,page.size()),"test rewind history readback differs");checked+=page.size();continue;
                }
                std::vector<uint8_t> page(65536);
                for(uint64_t off=0;off<r.bytes;) {
                    const size_t n=size_t(std::min<uint64_t>(page.size(),r.bytes-off));
                    const qfn_native_slice part={r.owner,r.layer,r.offset+off,n};p->paired_owner(part,page.data(),false);
                    require(!std::memcmp(page.data(),a.owners.data()+checked,n),"test rewind actual owner readback differs");off+=n;checked+=n;
                }
            }
            require(checked==a.owners.size(),"test rewind readback extent differs");
            std::fprintf(stderr,"QFN_TEST_ANCHOR_READBACK bytes=%llu position=%llu actual_native_owners=1 matched=1\n",
                (unsigned long long)checked,(unsigned long long)a.tokens.size());
        }
        p->paired_rows=a.plan.target_mtp;p->mtp_pend.clear();p->handover_draft_pending=p->mtp_ok;
        p->last_step_n=0;p->sampling_started=false;p->verify_checkpoint_ready=false;
        if(p->mtp_ok)p->carry_row(int64_t(p->seq.size())-1,p->h_last);
        tokens=p->seq;logits=a.logits;return true;
    }catch(const std::exception &e){if(mutated)p->doorbell_poisoned=true;err=e.what();return false;}
}

void qfn_engine::set_prefill_progress(std::function<void(uint32_t,uint32_t,uint64_t)> progress) {
    p->prefill_progress=std::move(progress);
}

bool qfn_engine::prefill(const std::vector<int32_t> & tokens, const qfn_logits_request & req, std::string & err) {
    try {
        p->sampling_started = false;
        p->profile_begin();
        const bool ok = p->prefill(tokens, req, true);
        p->profile_end("prefill");
        return ok;
    } catch (const std::exception & e) {
        p->cancel_prefill();
        err = e.what();
        return false;
    }
}

bool qfn_engine::cache_policy_self_test(std::string & err) {
    try {
        qfn_hparams hp{}; hp.n_layer = 2; hp.n_expert = 4; hp.n_expert_used = 1;
        impl test; test.hp = &hp;
        int checks = 0;
        auto expect = [&](bool pass, const char * reason) { require(pass, reason); ++checks; };
        impl::cache_policy initial;
        initial.owner = {0, 4}; initial.slot_of = {0, -1, -1, -1, 1, -1, -1, -1};
        initial.lfu = {8, 1}; initial.lfu_step = {0, 0};
        initial.resident = initial.routing = {1, 0, 0, 0, 1, 0, 0, 0};
        initial.ram_owner = {1, 5}; initial.ram_of = {-1, 0, -1, -1, -1, 1, -1, -1};
        initial.ram_age = {7, 8}; initial.ram_clock = 8;
        const int32_t one = 1, zero = 0;
        auto c = initial;
        test.policy_layer(c, 0, &one);
        expect(c.owner == std::vector<int32_t>({0, 1}), "RAM promotion did not use lowest-LFU victim");
        expect(c.slot_of[4] == -1 && c.slot_of[1] == 1, "VRAM inverse ownership failed");
        expect(c.ram_owner == std::vector<int32_t>({-1, 4}) && c.ram_of[5] == -1 && c.ram_of[1] == -1,
               "RAM exclusivity/protected promoted slot failed");
        expect(c.ram_age[0] == 0 && c.ram_age[1] == 9 && c.ram_clock == 9, "RAM age/clock transition failed");
        expect(c.resident[4] == 0 && c.routing[4] == 1 && c.routing[1] == 1,
               "canonical v2 publication before later layer was changed");
        expect(c.lfu[1] == 1 && c.lfu_step[1] == 0, "new miss LFU touch differs from canonical one-token path");
        test.policy_layer(c, 1, &one);
        expect(c.owner == std::vector<int32_t>({0, 5}) && c.ram_owner == std::vector<int32_t>({1, 4}),
               "cross-layer disk miss/demotion transition failed");
        expect(c.routing[1] == 1 && c.resident[1] == 0 && c.routing[4] == 0 && c.routing[5] == 1,
               "later evictions did not retain exact v2 row-publication boundaries");
        expect(initial.owner == std::vector<int32_t>({0, 4}) && initial.ram_owner == std::vector<int32_t>({1, 5}),
               "speculative simulation mutated the original checkpoint");
        test.verify_selected = {1, 1, 0, 1, 1, 0, 1, 0}; // layer-major [layer, four tokens, K]
        const auto masks = test.simulate_verify_policy(initial, 4);
        expect(test.verify_prefix.size() == 5 && test.verify_prefix[1].step == 1 && test.verify_prefix[4].step == 4,
               "canonical step count depends on verification batch width");
        expect(test.verify_prefix[1].owner == std::vector<int32_t>({0, 5}) &&
               test.verify_prefix[1].ram_owner == std::vector<int32_t>({1, 4}), "prefix-only cache commit lost first-token state");
        expect(test.verify_prefix[1].lfu != test.verify_prefix[4].lfu &&
               test.verify_prefix[1].ram_clock < test.verify_prefix[4].ram_clock, "rejected tail LFU/RAM effects were kept in prefix");
        expect(masks[4] == test.verify_prefix[1].routing[0] && masks[5] == test.verify_prefix[1].routing[1],
               "next token did not get canonical per-row residency input");
        c = initial; c.routing.assign(8, 0); c.step = 64; c.lfu_step = {0, 0};
        test.policy_layer(c, 0, &zero);
        expect(std::fabs(c.lfu[0] - 5.0) < 1e-12 && c.lfu_step[0] == 64 && !c.dirty,
               "LFU canonical token-step decay or no-miss publication failed");
        expect(c.routing == std::vector<float>(8, 0.0f), "a hit-only token incorrectly published dirty rows");
        c = initial; c.routing.assign(8, 0); c.dirty = true;
        const auto dirty_masks = test.simulate_verify_policy(c, 2);
        expect(dirty_masks[0] == 1 && dirty_masks[4 * 4] == 1,
               "pre-step dirty bitmap did not publish before token zero");
        c = initial; c.owner[1] = -1; c.slot_of[4] = -1; c.resident[4] = 0; c.ram_owner.clear(); c.ram_age.clear();
        test.policy_layer(c, 0, &one);
        expect(c.owner == std::vector<int32_t>({0, 1}), "empty VRAM slot was not preferred without a RAM tier");
        // Reproduce the real 100K failure: newly loaded low-LFU experts from
        // two layers evict each other on every replay despite ample total slots.
        auto replay_fixture = [&](bool protect) {
            test.n_slots = 3; test.owner = {0, 2, 6};
            test.slot_of = {0, -1, 1, -1, -1, -1, 2, -1};
            test.lfu = {1, 100, 100}; test.lfu_step = {0, 0, 0}; test.step = 0;
            std::vector<uint8_t> pins(8, 0);
            std::vector<int32_t> selected(8, -1);
            for (int attempt = 0; attempt < 6; ++attempt) {
                uint32_t il = 0;
                while (il < hp.n_layer && test.slot_of[size_t(il) * hp.n_expert + 1] >= 0) {
                    selected[size_t(il) * impl::DEC_NMAX] = 1; ++il;
                }
                if (il == hp.n_layer) return attempt + 1;
                selected[size_t(il) * impl::DEC_NMAX] = 1;
                if (protect) test.pin_replay_prefix(pins, selected, il, 1);
                const uint32_t s = test.miss_victims(il, 1, &one, 1, protect ? &pins : nullptr)[0];
                test.slot_of[size_t(test.owner[s])] = -1;
                test.owner[s] = int32_t(il * hp.n_expert + 1); test.slot_of[size_t(test.owner[s])] = int32_t(s);
                test.lfu[s] = 1;
            }
            return 0;
        };
        expect(replay_fixture(false) == 0, "cross-layer fixture did not reproduce unpinned replay livelock");
        expect(replay_fixture(true) == 3 && test.slot_of[1] >= 0 && test.slot_of[5] >= 0 && test.n_slots == 3,
               "prefix pins did not finish both layers at unchanged cache capacity");
        expect(test.miss_victims(1, 1, &zero, 1)[0] == uint32_t(test.slot_of[1]),
               "temporary replay pins leaked into a later ordinary victim selection");
        std::vector<uint8_t> pins(8, 0);
        std::vector<int32_t> prefix_selected = {1, -1, -1, -1, -1, -1, -1, -1};
        test.pin_replay_prefix(pins, prefix_selected, 0, 1);
        expect(std::count(pins.begin(), pins.end(), uint8_t(1)) == 1 && pins[1],
               "replay pinned stale future layers or unused batch columns");
        prefix_selected = {0, 1, 2, 3, 0, 1, 2, 3};
        test.pin_replay_prefix(pins, prefix_selected, 1, 4);
        expect(std::count(pins.begin(), pins.end(), uint8_t(1)) == 8,
               "full batched replay did not retain all four tokens across both layers");
        bool exhausted = false;
        try { (void) test.miss_victims(1, 1, &zero, 1, &pins); }
        catch (const std::runtime_error & failure) {
            exhausted = std::strstr(failure.what(), "retaining replay prefix") != nullptr;
        }
        expect(exhausted, "insufficient replay capacity did not fail loudly with prefix retained");
        prefix_selected = {1, -1, -1, -1, 1, -1, -1, -1};
        const auto final_policy = test.replay_final_policy(initial, prefix_selected, 1);
        expect(final_policy.owner == std::vector<int32_t>({0, 5}) &&
               final_policy.ram_owner == std::vector<int32_t>({1, 4}) && final_policy.step == 1,
               "temporary replay layout replaced canonical VRAM/RAM ownership or step");
        expect(final_policy.routing == std::vector<float>({1, 1, 0, 0, 0, 1, 0, 0}) &&
               final_policy.resident == std::vector<float>({1, 0, 0, 0, 0, 1, 0, 0}) && !final_policy.dirty,
               "replay commit lost canonical post-router stale-row publication");
        c = initial; c.dirty = true; c.routing.assign(8, 0);
        const std::vector<int32_t> batch_selected = {1, 0, 1, 0, 1, 0, 1, 0};
        const auto batch_policy = test.replay_final_policy(c, batch_selected, 4);
        expect(batch_policy.owner == std::vector<int32_t>({5, 4}) && batch_policy.step == 1 &&
               batch_policy.lfu == std::vector<double>({1, 1}) && batch_policy.routing[1] == 1 && !batch_policy.dirty,
               "ordinary four-token replay was incorrectly committed as token-major MTP");
        expect(initial.owner == std::vector<int32_t>({0, 4}) && initial.routing == initial.resident,
               "replay canonical simulation mutated the router's initial logical residency");
        const auto first = impl::owed_prefix_plan(true, 1);
        expect(first.first_only && first.count == 0, "single kept drafted row incorrectly creates an owed token");
        const auto singleton = impl::owed_prefix_plan(false, 1);
        expect(!singleton.first_only && singleton.offset == 0 && singleton.count == 1 && singleton.base == 0,
               "non-drafted singleton lost its previous h_last pairing");
        const auto all = impl::owed_prefix_plan(true, 4);
        expect(!all.first_only && all.offset == 1 && all.count == 3 && all.base == 1,
               "complete accepted draft prefix owes wrong token or residual rows");
        // Mock the actual callback-throw prefix boundary using the same production plan.
        // Inputs beyond emitted rows must not appear in pending draft tokens/h_last.
        const std::vector<int32_t> tokens = {10, 11, 12, 13}, residuals = {100, 200, 300, 400};
        std::vector<int32_t> pending, paired_hidden;
        int keep = 0, last_hidden = 7;
        try {
            for (size_t i = 0; i < tokens.size(); ++i) {
                ++keep;
                if (i == 1) throw std::runtime_error("fixture callback threw after second emitted row");
            }
        } catch (const std::runtime_error &) {
            const auto plan = impl::owed_prefix_plan(true, keep);
            pending.assign(tokens.begin() + plan.offset, tokens.begin() + plan.offset + plan.count);
            for (int i = 0; i < plan.count; ++i) paired_hidden.push_back(residuals[plan.base + i - 1]);
            last_hidden = residuals[plan.base + plan.count - 1];
        }
        expect(keep == 2 && pending == std::vector<int32_t>({11}) && paired_hidden == std::vector<int32_t>({100}) &&
               last_hidden == 200, "callback throw resumed with stale or discarded draft hidden state");
        // The real MMID helper packs only indexed GEMMs for required static graphs.
        // Other callers retain the original batch node and all precision descriptors.
        for (auto type : {GGML_TYPE_Q2_0, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL})
            for (int tokens : {1, 2, 3, 4}) for (int rows : {1, 10}) {
                ggml_context * ctx = ggml_init({128 * ggml_tensor_overhead() + ggml_graph_overhead_custom(128, false), nullptr, true});
                require(ctx != nullptr, "cannot allocate capture packing fixture metadata");
                auto * weights = ggml_new_tensor_3d(ctx, type, 128, 128, 16);
                auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, rows, tokens);
                auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 10, tokens);
                auto * original = test.MMID(ctx, weights, x, ids);
                auto * packed = test.MMID(ctx, weights, x, ids, true);
                expect(original->op == GGML_OP_MUL_MAT_ID && original->ne[2] == tokens,
                       "ordinary MMID lost its original batching option");
                expect(packed->ne[0] == 128 && packed->ne[1] == 10 && packed->ne[2] == tokens,
                       "capture MMID changed output layout/router row order");
                auto * graph = ggml_new_graph_custom(ctx, 128, false); ggml_build_forward_expand(graph, packed);
                int indexed = 0;
                for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                    auto * node = ggml_graph_node(graph, i);
                    if (node->op != GGML_OP_MUL_MAT_ID) continue;
                    ++indexed;
                    require(node->ne[2] == 1 && node->src[0] == weights && node->src[0]->type == type &&
                            node->src[1]->ne[1] == rows && node->src[2]->ne[0] == 10 && node->src[2]->ne[1] == 1,
                            "capture MMID contains an unsupported shape or lost precision/input rows");
                }
                expect(indexed == tokens, "capture MMID did not retain every token in one graph");
                ggml_free(ctx);
            }
        // The production timeout quiescence must finish late payload/metadata writes BEFORE restoration.
        int mock_payload = 11;
        test.bell_stop.store(false);
        test.bell_thread = std::thread([&]() {
            while (!test.bell_stop.load()) std::this_thread::yield();
            mock_payload = 99;
            test.bell_miss += 2; test.bell_bytes += 128; test.bell_ms.store(3.0);
        });
        qfn_decode_stats failed_stats;
        test.quiesce_failed_bell(failed_stats, 0, 0, 0);
        expect(!test.bell_thread.joinable() && mock_payload == 99,
               "timeout rollback can overlap a late owned worker payload write");
        mock_payload = 11;
        expect(failed_stats.expert_misses == 2 && failed_stats.miss_bytes == 128 && failed_stats.miss_ms == 3.0 && mock_payload == 11,
               "timeout quiescence lost actual discarded IO or failed to restore after join");
        qfn_engine poisoned; poisoned.p->doorbell_poisoned = true;
        expect(poisoned.reset_required(), "poisoned state is not observable by the production adapter");
        for (int operation = 0; operation < 6; ++operation) {
            std::string failure; bool refused = false;
            try {
                qfn_sampling sampling; qfn_decode_stats stats; std::vector<float> logits;
                qfn_logits_request request;
                if (operation == 0) (void) poisoned.n_past();
                if (operation == 1) poisoned.begin_sampling(sampling);
                if (operation == 2) (void) poisoned.sample(nullptr, sampling);
                if (operation == 3) { refused = !poisoned.feed({0}, logits, stats, failure); }
                if (operation == 4) { refused = !poisoned.append({0}, request, failure); }
                if (operation == 5) { refused = !poisoned.decode(0, 1, sampling, [](int32_t,const float *) {return true;}, stats, failure); }
            } catch (const std::exception & error) { refused = true; failure = error.what(); }
            expect(refused && failure.find("QFN_DOORBELL_RESET_REQUIRED") != std::string::npos,
                   "poisoned production state/sampling/feed/append/decode path silently continued");
        }
        test.doorbell_poisoned = false;
        std::printf("MTP canonical cache policy PASS: %d checks, production policy/owed-prefix logic, no CUDA initialization\n", checks);
        return true;
    } catch (const std::exception & error) { err = error.what(); return false; }
}

bool qfn_engine::sampling_self_test(std::string & err) {
    try {
        qfn_hparams hp{}; hp.n_vocab = 4;
        impl test; test.hp = &hp;
        test.seq = {0, 1, 0, 2};
        const float logits[] = {4, 3, 2, 1};
        qfn_sampling s; s.top_k = 0; s.top_p = 1;
        using distribution = std::vector<std::pair<int32_t, double>>;
        distribution base, changed;
        test.filtered(logits, s, base);
        // The public sampler must advance between draws, and reset only at request start.
        qfn_engine api; api.p->hp = &hp; api.p->seq = test.seq;
        s.seed = 17;
        api.begin_sampling(s);
        std::vector<int32_t> draws;
        for (int i = 0; i < 128; ++i) draws.push_back(api.sample(logits, s));
        require(std::any_of(draws.begin(), draws.end(), [&](int32_t t) { return t != draws[0]; }), "public sample resets RNG on every draw");
        api.begin_sampling(s);
        for (int32_t t : draws) require(api.sample(logits, s) == t, "public request seed does not repeat");
        double total = 0;
        for (const auto & item : base) {
            total += item.second;
            require(std::abs(test.prob_of(logits, s, item.first) - item.second) < 1e-12, "prob_of differs from filtered");
        }
        require(std::abs(total - 1) < 1e-12, "distribution is not normalized");
        test.rng.seed(17);
        int counts[4] = {};
        for (int i = 0; i < 10000; ++i) ++counts[test.sample(logits, s)];
        for (const auto & item : base) require(std::abs(counts[item.first] / 10000.0 - item.second) < .02, "sample frequencies differ from filtered");
        for (int i = 0; i < 1000; ++i) require(test.sample_excluding(logits, s, 0) != 0, "excluded token sampled");
        std::fill(counts, counts + 4, 0);
        const double proposal_probability = test.prob_of(logits, s, 0);
        for (int i = 0; i < 10000; ++i) {
            const bool accepted = std::uniform_real_distribution<double>(0, 1)(test.rng) < proposal_probability;
            ++counts[accepted ? 0 : test.sample_excluding(logits, s, 0)];
        }
        for (const auto & item : base) require(std::abs(counts[item.first] / 10000.0 - item.second) < .02, "speculative acceptance changes target distribution");
        // do_sample=false is the same target distribution used by ordinary sampling and MTP acceptance.
        s.do_sample=false; s.temperature=0.8f;
        test.filtered(logits,s,changed);
        require(s.greedy() && s.temperature==0.8f && changed==std::vector<std::pair<int32_t,double>>{{0,1.0}},"do_sample=false failed shared greedy distribution");
        require(test.prob_of(logits,s,0)==1 && test.prob_of(logits,s,1)==0,"do_sample=false speculative acceptance probability changed");
        for(int i=0;i<32;++i) require(test.sample(logits,s)==0,"do_sample=false sampled stochastically");
        s.logit_bias[0]=-100;
        require(test.sample(logits,s)==1,"do_sample=false ignored logit bias");
        s.logit_bias.clear(); s.do_sample=true;
        s.temperature = 0;
        require(test.sample(logits, s) == 0, "greedy mismatch");
        s.logit_bias[0] = -100;
        require(test.sample(logits, s) == 1, "greedy ignores bias");
        s.logit_bias.clear(); s.temperature = 1; s.top_k = 1;
        require(test.sample(logits, s) == 0, "top-k 1 differs from greedy");
        s.top_k = 0; s.top_p = .01f;
        require(test.sample(logits, s) == 0, "top-p .01 differs from greedy");
        s.top_p = 1; s.min_p = 1;
        require(test.sample(logits, s) == 0, "min-p 1 differs from greedy");
        s.min_p = 0; s.typical_p = .01f;
        test.filtered(logits, s, changed);
        require(changed != base, "typical-p has no effect");
        s.typical_p = 1; s.repeat_penalty = 2;
        test.filtered(logits, s, changed);
        require(changed != base, "repeat penalty has no effect");
        s.repeat_last_n = 0;
        test.filtered(logits, s, changed);
        require(changed == base, "zero repeat window still applies penalties");
        s.repeat_last_n = -1; s.repeat_penalty = 1; s.presence_penalty = 5;
        test.filtered(logits, s, changed);
        require(changed != base, "presence penalty has no effect");
        s.presence_penalty = 0; s.frequency_penalty = 5;
        test.filtered(logits, s, changed);
        require(changed != base, "frequency penalty has no effect");
        // Changing the unverified suffix must not change a verified position's sampler.
        test.filtered(logits, s, base, 2);
        test.seq[2] = 3; test.seq[3] = 3;
        test.filtered(logits, s, changed, 2);
        require(changed == base, "unverified draft leaks into penalty history");
        test.filtered(logits, s, changed, 4);
        require(changed != base, "accepted history is missing from penalties");
        return true;
    } catch (const std::exception & e) { err = e.what(); return false; }
}
