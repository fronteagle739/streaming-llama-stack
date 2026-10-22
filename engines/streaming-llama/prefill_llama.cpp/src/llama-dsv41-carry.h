#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "../ggml/include/ggml-dsv41-carry-copy.h"

// One request's bounded, explicitly RAM- or SSD-backed layer-major carry. No existing file can be
// opened, resumed, truncated or replaced. Failed/incomplete files are retained.
// The store is deliberately not a reusable/persistent KV-frontier artifact.
class llama_dsv41_carry_store {
public:
    static constexpr uint32_t max_tokens = 1048576;
    static constexpr uint32_t page_tokens = 32;
    static constexpr uint32_t hidden_floats = 5120 * 4;
    static constexpr uint32_t pre_floats = 4;
    static constexpr uint32_t max_selected = 512;
    static constexpr uint32_t max_candidates = 2048;
    static constexpr uint64_t reserve_bytes = 30ull * 1024 * 1024 * 1024;
    static constexpr uint64_t ram_record_bytes = ggml_dsv41_copy::record_bytes;
    static constexpr uint64_t ram_page_bytes = ggml_dsv41_copy::page_bytes;
    static constexpr uint64_t ram_header_bytes = ggml_dsv41_copy::header_bytes;

    // A bounded borrowed slice of the existing single RAM bank. It may be used
    // only until its exact lease is completed/invalidated. The executor owns
    // registration and must finish all transfers before the store is destroyed.
    class ram_lease {
    public:
        ram_lease()=default;
        uint32_t start()const{return first;} uint32_t count()const{return rows;}
        uint32_t input_selected()const{return sw;} uint32_t input_candidates()const{return cw;}
        uint32_t output_selected()const{return next_sw;} uint32_t output_candidates()const{return next_cw;}
        uint8_t * row(uint32_t relative)const;
    private:
        uint8_t *base=nullptr;
        uint64_t bytes=0,store_id=0,generation=0;
        uint32_t first=0,rows=0,sw=0,cw=0,next_sw=0,next_cw=0;
        bool seeding=false;
        friend class llama_dsv41_carry_store;
        friend class llama_dsv41_tile_executor;
#ifdef DS4PERF_DIRECT_CARRY_TEST
        friend struct ds4perf_direct_carry_fixture;
#endif
    };
    // Read-only immediate next tile, borrowed ONLY while its parent direct
    // lease is live. No cursor advance and no completion credit. Executor must
    // finish the H2D read before completing that parent lease.
    class ram_lookahead {
    public:
        ram_lookahead()=default;
        uint32_t start()const{return lease.start();}
        uint32_t count()const{return lease.count();}
        const uint8_t *row(uint32_t relative)const{return lease.row(relative);}
    private:
        ram_lease lease;
        friend class llama_dsv41_carry_store;
        friend class llama_dsv41_tile_executor;
#ifdef DS4PERF_DIRECT_CARRY_TEST
        friend struct ds4perf_direct_carry_fixture;
#endif
    };
    // Constructible only by the executor after actual zero GPU validation flags
    // and completed D2H transfers, or explicit full-residency retention. No
    // public boolean can certify a live copy or a device-resident generation.
    class ram_completion {
    public:
        ram_completion(const ram_completion&)=default;
        ram_completion &operator=(const ram_completion&)=default;
        ~ram_completion()=default;
    private:
        ram_completion(const ram_lease &lease,bool selected_unchanged,bool candidates_unchanged,bool hidden_mirrored=true):
            lease(lease),selected_unchanged(selected_unchanged),candidates_unchanged(candidates_unchanged),hidden_mirrored(hidden_mirrored){}
        ram_lease lease;
        bool selected_unchanged=false,candidates_unchanged=false;
        bool hidden_mirrored=true;
        friend class llama_dsv41_carry_store;
        friend class llama_dsv41_tile_executor;
#ifdef DS4PERF_DIRECT_CARRY_TEST
        friend struct ds4perf_direct_carry_fixture;
#endif
    };

    struct budget {
        uint64_t file_bytes = 0; // Logical aligned bank extent in either mode.
        uint64_t page_buffer_bytes = 0;
        uint64_t bounded_host_bytes = 0;
        uint64_t required_free_bytes = 0;
        // Full one-bank extent retained in RAM; zero for the file mode.
        uint64_t resident_bytes = 0;
        uint64_t disk_bytes = 0; // Physical exclusive-new file; overflow only in hybrid mode.
    };
    class ram_backing {
    public:
        uint8_t *data()const{return storage.get();}
        uint64_t size()const{return bytes;}
    private:
        ram_backing()=default;
        std::shared_ptr<uint8_t[]> storage;
        uint64_t bytes=0;
        std::mutex owner;
        friend class llama_dsv41_carry_store;
    };
    struct config {
        // Absolute, exclusive-new file in an existing physical directory.
        std::string path;
        uint32_t tokens = 0;
        uint64_t admitted_file_bytes = 0;
        uint64_t admitted_host_bytes = 0;
        // Other not-yet-allocated payloads admitted on THIS actual volume.
        uint64_t outstanding_bytes = 0;
        // Explicit single-bank RAM mode. path/file admission are unused;
        // insufficient host admission fails without any file fallback.
        bool resident_ram = false;
        std::shared_ptr<ram_backing> shared_ram;
        // Direct borrowed RAM leases use no large vector tiles. Vector reads
        // are restricted to one page; seeding/head/export remain bounded.
        bool direct_ram_tiles = false;
        uint32_t max_tile_tokens = 32;
        // Explicit RAM-prefix/D-overflow mode, using the existing unbuffered
        // file owner and batched aligned tile buffer. No implicit fallback.
        uint32_t ram_prefix_tokens = 0;
        uint64_t disk_reserve_bytes = reserve_bytes;
        // Full existing VRAM residency only. Intermediate host hidden rows may
        // be stale; by default layer39 restores every row before head/export.
        bool defer_hidden_readback = false;
        // Full device bank only: restore final hidden rows after the required
        // head. Host reads/publication remain forbidden until every row is copied.
        bool export_hidden_after_head = false;
    };
    struct tile {
        uint32_t start = 0, count = 0;
        uint32_t selected_width = 0, candidate_width = 0;
        // Token-major: GGML [5120,4,count], [4,count], [width,count].
        std::vector<float> residual, previous_pre;
        std::vector<int32_t> selected, candidates;
    };

    llama_dsv41_carry_store();
    ~llama_dsv41_carry_store();
    llama_dsv41_carry_store(const llama_dsv41_carry_store &) = delete;
    llama_dsv41_carry_store & operator=(const llama_dsv41_carry_store &) = delete;

    // Pure capacity calculation; no allocation, file creation or model access.
    // Includes aligned physical pages, header, other pending payloads and the
    // fixed 30 GiB floor. No hidden second bank or temporary file is created.
    static bool plan(uint32_t tokens, uint64_t outstanding_bytes,
                     budget & result, std::string & error);
    static bool plan_resident_ram(uint32_t tokens, budget & result, std::string & error,
                                  uint32_t max_tile_tokens = 32);
    static bool plan_resident_ram_direct(uint32_t tokens,budget &result,std::string &error,
                                         uint32_t max_tile_tokens=32);
    static bool plan_hybrid(uint32_t tokens,uint32_t prefix_tokens,uint32_t max_tile_tokens,
                            uint64_t outstanding,uint64_t disk_reserve,budget &result,std::string &error);
    static std::shared_ptr<ram_backing> reserve_ram(uint32_t tokens,uint64_t admitted_host_bytes,uint32_t tile);
    bool create(const config & options, std::string & error);
    // Retire a completed one-token view before reusing its pre-registered bank.
    // Invalidate it first; live/seeding/incomplete leases cannot release credit.
    bool retire_shared_ram(std::string &error);

    // Seed contiguous pages once; ID widths must initially be zero. All four
    // hidden copies and all four previous-pre values are supplied explicitly.
    bool seed(const tile & input, std::string & error);
    bool acquire_ram_seed(uint32_t start,uint32_t count,ram_lease &lease,std::string &error);
    bool complete_ram_seed(const ram_lease &lease,const ram_completion &completion,std::string &error);
    bool finish_seed(std::string & error);

    // OUTPUT widths for this layer. Reads still return the OLD widths from
    // the last complete layer. A source may replace selected IDs only at
    // 2/8/14/20/24/28/32/36, and candidates only at 20. Other layers must
    // preserve the corresponding compact IDs exactly, including 21..23.
    bool begin_layer(uint32_t layer, uint32_t selected_width,
                     uint32_t candidate_width, std::string & error);
    // One lease at a time; start must equal the next uncommitted token and
    // the <=32-token range must not cross a physical 32-token page boundary.
    bool read_tile(uint32_t start, uint32_t count, tile & input, std::string & error);
    // Caller MUST finish/synchronize computation and capture its outputs
    // before this overwrites the leased input rows. Unconsumed neighboring
    // rows in the physical page are preserved. IDs are compact, never padded.
    bool write_tile(const tile & output, std::string & error);
    bool commit_layer(std::string & error);
    bool ram_storage(void * &base,uint64_t &bytes,std::string &error);
    bool acquire_ram_tile(uint32_t start,uint32_t count,ram_lease &lease,std::string &error);
    bool lookahead_ram_tile(const ram_lease &parent,uint32_t count,ram_lookahead &span,std::string &error);
    bool complete_ram_tile(const ram_lease &lease,const ram_completion &completion,std::string &error);
    bool acquire_ram_export(uint32_t start,uint32_t count,ram_lease &lease,std::string &error);
    bool complete_ram_export(const ram_lease &lease,const ram_completion &completion,std::string &error);

    // Available only after all 40 layers. Head computation may read any
    // bounded final range; it need not compute unused per-prompt logits.
    bool read_final_tile(uint32_t start, uint32_t count, tile & output,
                         std::string & error);
    // Caller MUST synchronize the actual head/native-cache frontier first.
    // This cannot certify partial layers, abandoned leases or failed sweeps.
    bool certify_frontier(std::string & error);
    bool frontier_ready() const;
    bool failed() const;
    uint32_t completed_layers() const;
    void invalidate();

private:
    struct implementation;
    std::unique_ptr<implementation> p;
};
