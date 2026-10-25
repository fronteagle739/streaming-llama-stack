#include "llama-dsv41-cache.h"
#include "boundary-capture.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

bool llama_dsv41_cache_storage::write_native_span_locked(
        const llama_dsv41::state_span & span,uint64_t offset,const void *src,
        size_t bytes,std::string &error) {
    using llama_dsv41::state_kind;
    ggml_tensor *tensor=nullptr;
    switch(span.kind) {
        case state_kind::raw_window:if(span.index<raw_ring.size())tensor=raw_ring[span.index];break;
        case state_kind::compressed:if(span.index<compressed_cache.size())tensor=compressed_cache[span.index];break;
        case state_kind::index:if(span.index<index_cache.size())tensor=index_cache[span.index];break;
        case state_kind::pool_kv:if(span.index<carry_kv.size())tensor=carry_kv[span.index];break;
        case state_kind::pool_score:if(span.index<carry_score.size())tensor=carry_score[span.index];break;
        case state_kind::dspark_hidden:if(!span.index)tensor=dspark_ring;break;
    }
    if(!tensor||(!src&&bytes)||bytes>65536||(offset&3)||(bytes&3)||
       span.bytes>uint64_t(ggml_nelements(tensor))*4||offset>span.bytes||bytes>span.bytes-offset) {
        error="Invalid bounded canonical V4.1 import slice";return false;
    }
    if(bytes&&tensor->type==GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> packed(bytes/4);
        for(size_t i=0;i<packed.size();++i) {
            uint32_t bits;std::memcpy(&bits,static_cast<const unsigned char*>(src)+4*i,4);
            if(bits&65535u){error="Native cache import would round a canonical F32 value";return false;}
            packed[i].bits=uint16_t(bits>>16);
        }
        ggml_backend_tensor_set(tensor,packed.data(),size_t(offset/2),bytes/2);
    } else if(bytes)ggml_backend_tensor_set(tensor,src,size_t(offset),bytes);
    error.clear();return true;
}

bool llama_dsv41_cache_storage::restore_native_frontier(
        const llama_dsv41::wire_frontier &target,const std::vector<std::string>&hashes,
        const llama_dsv41::wire_read_span&read,std::string&error) {
    return apply_native_update(nullptr,target,{},hashes,read,error);
}
bool llama_dsv41_cache_storage::apply_native_delta(
        const llama_dsv41::wire_frontier &base,const llama_dsv41::wire_frontier &target,
        const std::vector<std::string>&before,const std::vector<std::string>&after,
        const llama_dsv41::wire_read_span&read,std::string&error) {
    return apply_native_update(&base,target,before,after,read,error);
}
bool llama_dsv41_cache_storage::replace_native_frontier(
        const llama_dsv41::wire_frontier &target,const std::vector<std::string>&hashes,
        const llama_dsv41::wire_read_span&read,std::string&error) {
    return apply_native_update(nullptr,target,{},hashes,read,error,false,true);
}
bool llama_dsv41_cache_storage::apply_native_update(
        const llama_dsv41::wire_frontier *base,const llama_dsv41::wire_frontier &target,
        const std::vector<std::string>&before,const std::vector<std::string>&after,
        const llama_dsv41::wire_read_span&read,std::string&error,bool rewind,bool replace) {
    std::lock_guard<std::mutex>lock(execution_mutex);
    auto need=[](bool ok,const std::string&why){if(!ok)throw std::runtime_error(why);};
    // Restore/replace/rewind may rewrite rows the incremental memo covers.
    if(!base||rewind)digest_memo.invalidate();
    try {
        need(!execution_poisoned,"Native cache import requires a healthy cache");
        need(bool(read)&&target.context==context&&(target.mode==3)==bool(dspark_ring),"Native cache import context/mode mismatch");
        need(base||!execution_backend||replace,"Full native restore is refused after compute binding");
        need(!replace||!base,"Full replacement cannot assert a delta base");
        llama_dsv41::wire_plan target_plan,base_plan;
        need(llama_dsv41::make_wire_plan(target,target_plan,error),error);
        need(target.mode==1||(target.mode==3&&target.capture_length==std::min(target.position,128u)),"Native restore requires complete mode1/3 state");
        std::vector<llama_dsv41::state_slice>slices;
        auto check_hashes=[&](const llama_dsv41::wire_plan&plan,const std::vector<std::string>&hashes){
            need(hashes.size()==plan.spans.size(),"Native owner hash count mismatch");
            for(const auto&h:hashes)need(h.size()==64&&std::all_of(h.begin(),h.end(),[](char c){return(c>='0'&&c<='9')||(c>='a'&&c<='f');}),"Invalid native owner SHA256");
        };
        check_hashes(target_plan,after);
        if(base) {
            need(llama_dsv41::make_wire_plan(*base,base_plan,error),error);check_hashes(base_plan,before);
            need(!dspark_ring||dspark_completion.ready(base->position),"Native delta base DSpark position mismatch");
            if(rewind){
                need(base->context==target.context&&base->mode==target.mode&&target.position<base->position,
                    "Native rewind requires an earlier checkpoint in the same owner");
                for(const auto&s:target_plan.spans)
                    if(s.kind!=llama_dsv41::state_kind::compressed&&s.kind!=llama_dsv41::state_kind::index)
                        slices.push_back({s,0,s.bytes});
            }else{
                llama_dsv41::delta_plan plan;
                need(llama_dsv41::make_delta_plan(*base,target,plan,error),error);
                slices=std::move(plan.slices);
            }
        } else for(const auto&s:target_plan.spans)slices.push_back({s,0,s.bytes});
        if(execution_backend)ggml_backend_synchronize(execution_backend);
        auto verify=[&](const llama_dsv41::wire_plan&plan,const std::vector<std::string>&hashes){
            if(llama_dsv41::owner_verify_incremental()){
                // Same SHA-256 values; only rows added since the last verified
                // digest are read. A mismatch re-hashes every owner before failing.
                const auto actual=owner_digests_locked(plan,&hashes,&plan==&target_plan?"apply_target":"apply_base",false);
                for(size_t i=0;i<plan.spans.size();++i)need(actual[i]==hashes[i],"Native cache owner SHA256 mismatch at span"+std::to_string(i));
                return;
            }
            for(size_t i=0;i<plan.spans.size();++i){const auto&s=plan.spans[i];
                const auto actual=dsv41_boundary::sha256_stream(s.bytes,[&](uint64_t off,void*dst,size_t count){
                    return read_native_span_locked(s,off,dst,count,error,false);
                });
                need(actual==hashes[i],"Native cache owner SHA256 mismatch at span"+std::to_string(i));
            }
        };
        if(base)verify(base_plan,before);
        need(ensure_tokens_locked(target.position,base?(rewind?"rewind":"delta"):"import",nullptr,error),error);   // growable cache: map before the rows are written
        execution_poisoned=true;
        std::array<unsigned char,65536>page{};
        for(const auto&s:slices)for(uint64_t off=0;off<s.bytes;){
            const size_t count=size_t(std::min<uint64_t>(page.size(),s.bytes-off));
            need(read(s.span,s.offset+off,page.data(),count),"Native delta source read failed");
            need(write_native_span_locked(s.span,s.offset+off,page.data(),count,error),error);off+=count;
        }
        if(execution_backend)ggml_backend_synchronize(execution_backend);
        verify(target_plan,after);
        if(dspark_ring){dspark_completion.clear();dspark_completion.ends.fill(target.position);}
        execution_poisoned=false;error.clear();return true;
    }catch(const std::exception&failure){
        execution_poisoned=true;dspark_completion.clear();digest_memo.invalidate();error=failure.what();return false;
    }catch(...){
        execution_poisoned=true;dspark_completion.clear();digest_memo.invalidate();error="Unknown native cache import failure";return false;
    }
}

std::vector<std::string> llama_dsv41_cache_storage::owner_digests_locked(const llama_dsv41::wire_plan&plan,
        const std::vector<std::string>*expected,const char*phase,bool require_capture){
    std::string why;std::mutex why_lock;   // the memo reads from several threads
    auto read=[&](const llama_dsv41::state_span&s,uint64_t off,void*dst,size_t n){
        std::string e;if(read_native_span_locked(s,off,dst,n,e,require_capture,llama_dsv41::owner_digest_memo::chunk))return true;
        std::lock_guard<std::mutex>l(why_lock);if(why.empty())why=e;return false;
    };
    if(!digest_staging_tried){
        digest_staging_tried=true;
        ggml_backend_dev_t dev=buffer?ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer.get())):nullptr;
        ggml_backend_buffer_type_t host=dev?ggml_backend_dev_host_buffer_type(dev):nullptr;
        if(host)digest_staging.reset(ggml_backend_buft_alloc_buffer(host,llama_dsv41::owner_digest_memo::staging_bytes()));
        if(digest_staging)digest_memo.set_staging(static_cast<unsigned char*>(ggml_backend_buffer_get_base(digest_staging.get())),ggml_backend_buffer_get_size(digest_staging.get()));
        std::fprintf(stderr,"LLAMADECODE_OWNER_DIGEST_STAGING pinned=%d bytes=%llu chunk=%llu max_workers=%u sha=%s\n",int(bool(digest_staging)),
            (unsigned long long)(digest_staging?ggml_backend_buffer_get_size(digest_staging.get()):0),(unsigned long long)llama_dsv41::owner_digest_memo::chunk,
            llama_dsv41::owner_digest_memo::max_workers,
#if defined(_WIN32)
            "cng"
#else
            "portable"
#endif
            );
    }
    const auto started=std::chrono::steady_clock::now();
    const bool incremental=llama_dsv41::owner_verify_incremental(),memo=digest_memo.ready();
    uint64_t hashed=0,total=0;bool fallback=false;std::vector<std::string> out;
    for(const auto&s:plan.spans)total+=s.bytes;
    try{
        out=digest_memo.digests(plan,read,incremental,&hashed);
        if(incremental&&expected&&out!=*expected){
            fallback=true;uint64_t again=0;out=digest_memo.digests(plan,read,false,&again);hashed+=again;
        }
    }catch(const std::exception&e){digest_memo.invalidate();throw std::runtime_error(std::string(e.what())+(why.empty()?"":": "+why));}
    const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    std::fprintf(stderr,"LLAMADECODE_OWNER_DIGEST phase=%s mode=%s position=%u memo=%d hashed_bytes=%llu owner_bytes=%llu full_rehash_fallback=%d workers=%u ms=%.3f\n",
        phase,incremental?"incremental":"full",plan.frontier.position,int(memo),(unsigned long long)hashed,(unsigned long long)total,int(fallback),digest_memo.workers_used(),ms);
    if(fallback&&out==*expected)std::fprintf(stderr,"LLAMADECODE_OWNER_DIGEST_MEMO_STALE phase=%s position=%u full_rehash_matched=1\n",phase,plan.frontier.position);
    std::fflush(stderr);
    return out;
}

bool llama_dsv41_cache_storage::owner_digests(const llama_dsv41::wire_frontier&frontier,std::vector<std::string>&out,
        const std::vector<std::string>*expected,std::string&error){
    std::lock_guard<std::mutex>lock(execution_mutex);
    try{
        if(execution_poisoned)throw std::runtime_error("V4.1 cache requires clear after an incomplete computation");
        llama_dsv41::wire_plan plan;
        if(!llama_dsv41::make_wire_plan(frontier,plan,error))throw std::runtime_error(error);
        if(frontier.context!=context||(frontier.mode==3)!=bool(dspark_ring))throw std::runtime_error("Owner digest context/mode mismatch");
        if(expected&&expected->size()!=plan.spans.size())throw std::runtime_error("Owner digest expected hash count mismatch");
        if(execution_backend)ggml_backend_synchronize(execution_backend);
        out=owner_digests_locked(plan,expected,"inspect",true);error.clear();return true;
    }catch(const std::exception&e){digest_memo.invalidate();out.clear();error=e.what();return false;}
}

void llama_dsv41_cache_storage::forget_owner_digests(){
    std::lock_guard<std::mutex>lock(execution_mutex);digest_memo.invalidate();
}

bool llama_dsv41_cache_storage::rewind_native_frontier(
        const llama_dsv41::wire_frontier&base,const llama_dsv41::wire_frontier&target,
        const std::vector<std::string>&before,const std::vector<std::string>&after,
        const llama_dsv41::wire_read_span&read,std::string&error){
    return apply_native_update(&base,target,before,after,read,error,true);
}
