#pragma once
#include "llama-dsv41-engram.h"
#include <functional>
#include <memory>
class llama_dsv41_engram_prompt_cache {
    struct implementation;
    std::unique_ptr<implementation> p;
public:
    llama_dsv41_engram_prompt_cache();
    ~llama_dsv41_engram_prompt_cache();
    llama_dsv41_engram_prompt_cache(const llama_dsv41_engram_prompt_cache&)=delete;
    llama_dsv41_engram_prompt_cache&operator=(const llama_dsv41_engram_prompt_cache&)=delete;
    void prepare(const llama_dsv41::engram_layout&,const int32_t*,uint32_t,const std::array<uint64_t,2>&,uint64_t,llama_dsv41::engram_history initial={});
    uint64_t budget()const;
    void load(const wchar_t*,void *verified_file,uint64_t,const std::function<bool()>&);
    bool read(uint32_t,uint64_t,void*,size_t)const;
    // Table 0's rows complete and publish before table 1's, so a reader of
    // table 0 (layer 1) need not wait for table 1 (layer 14). Throws when the
    // load failed or stop() reports cancellation.
    void wait_table(uint32_t,const std::function<bool()>&)const;
};
