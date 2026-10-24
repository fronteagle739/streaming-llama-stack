/* Optional displaced-expert promotion. Single serialized foreground owner,
 * one LOW DS read batch, existing VRAM slots; no second cache or worker thread. */
#if defined(_WIN32)
struct cuda_affinity_background_batch {
    const void *model=NULL;uint64_t size=0,gate=0,up=0,down=0,gb=0,db=0;
    uint32_t layer=0,experts=0;std::vector<uint32_t> ids;
};
static cuda_affinity_background_batch g_affinity_pending,g_affinity_read;
static uint8_t *g_affinity_stage=NULL;
static uint64_t g_affinity_stage_bytes=0;
static cudaEvent_t g_affinity_copy_done=NULL;
static bool g_affinity_reading=false,g_affinity_copying=false;
static uint64_t g_affinity_last_actual[128][384]={},g_affinity_promoted[128][384]={};
static bool g_affinity_merit_seen[128][384]={};
static uint64_t g_affinity_bg_bytes=0,g_affinity_bg_admitted=0,g_affinity_bg_refused=0,
    g_affinity_bg_merit=0,g_affinity_bg_thrash=0,g_affinity_bg_no_window=0;
static uint64_t cuda_affinity_value(uint32_t l,uint32_t e) {
    return cuda_affinity_cache_priority(l,e);
}
static void cuda_affinity_thrash(uint32_t l,uint32_t e) {
    if(l>=128 || e>=384)return;
    uint64_t at=g_affinity_promoted[l][e];
    if(at && g_affinity_decoded+1u>=at && g_affinity_decoded+1u-at<8u) {
        g_affinity_displaced[l][e]/=2u;g_affinity_bg_thrash++;
    }
    g_affinity_promoted[l][e]=0;
}
static uint32_t cuda_affinity_victim(cuda_stream_expert_cache *cache,uint32_t l,uint32_t e) {
    uint32_t victim=UINT32_MAX;uint64_t lowest=UINT64_MAX;
    for(uint32_t i=0;i<cache->capacity;i++) {
        const auto &v=cache->slots[i];if(!v.valid)return i;
        if(v.layer>=128 || v.expert>=384)continue;
        uint64_t last=g_affinity_last_actual[v.layer][v.expert];
        if(last && g_affinity_decoded+1u>=last && g_affinity_decoded+1u-last<2u)continue;
        uint64_t value=cuda_affinity_value(v.layer,v.expert);
        if(value<lowest) {lowest=value;victim=i;}
    }
    return victim!=UINT32_MAX && cuda_affinity_value(l,e)>lowest?victim:UINT32_MAX;
}
static bool cuda_affinity_background_harvest(void) {
    if(!g_affinity_reading)return true;
    uint64_t bytes=0;double ms=0;int ready=ds4_ds_background_poll(&bytes,&ms);
    if(ready<0) {fprintf(stderr,"ds4: background DS read failed; no cache publication\n");return false;}
    if(!ready)return true;
    g_affinity_reading=false;
    auto &b=g_affinity_read;
    cuda_stream_expert_cache *cache=&g_stream_expert_caches[cuda_stream_expert_cache_class_index(b.gb,b.db)];
    const uint64_t record=2u*b.gb+b.db;
    for(uint32_t j=0;j<b.ids.size();j++) {
        uint32_t e=b.ids[j];
        if(!cache->valid || cache->gate_expert_bytes!=b.gb || cache->down_expert_bytes!=b.db ||
            cuda_stream_expert_cache_find(cache,b.model,b.size,b.layer,b.experts,e,
                b.gate,b.up,b.down,b.gb,b.db,false)>=0) {g_affinity_bg_refused++;continue;}
        uint32_t victim=cuda_affinity_victim(cache,b.layer,e);
        if(victim==UINT32_MAX) {g_affinity_bg_refused++;continue;}
        auto &old=cache->slots[victim];bool append=!old.valid;
        if(old.valid)cuda_affinity_thrash(old.layer,old.expert);
        old.valid=0;
        const uint8_t *host=g_affinity_stage+(uint64_t)j*record;
        if(!cuda_ok(cudaMemcpyAsync(cache->gate_ptr+(uint64_t)victim*b.gb,host,b.gb,cudaMemcpyHostToDevice,cuda_decode_stream()),"background gate upload") ||
           !cuda_ok(cudaMemcpyAsync(cache->up_ptr+(uint64_t)victim*b.gb,host+b.gb,b.gb,cudaMemcpyHostToDevice,cuda_decode_stream()),"background up upload") ||
           !cuda_ok(cudaMemcpyAsync(cache->down_ptr+(uint64_t)victim*b.db,host+2u*b.gb,b.db,cudaMemcpyHostToDevice,cuda_decode_stream()),"background down upload"))return false;
        cuda_stream_expert_cache_slot entry={};entry.valid=1;entry.model_map=b.model;entry.model_size=b.size;
        entry.layer=b.layer;entry.n_total_expert=b.experts;entry.expert=e;entry.gate_offset=b.gate;
        entry.up_offset=b.up;entry.down_offset=b.down;entry.gate_expert_bytes=b.gb;entry.down_expert_bytes=b.db;
        entry.age=++cache->tick;entry.route_rank=UINT32_MAX;old=entry;
        if(append)cache->count++;cache->rank_worst_valid=0;g_affinity_bg_admitted++;
        g_affinity_promoted[b.layer][e]=g_affinity_decoded+1u;g_affinity_merit_seen[b.layer][e]=false;
    }
    if(!cuda_ok(cudaEventRecord(g_affinity_copy_done,cuda_decode_stream()),"background upload fence"))return false;
    g_affinity_copying=true;
    fprintf(stderr,"ds4: affinity-background completed_bytes=%llu read_ms=%.6f admitted_total=%llu refused_total=%llu\n",
        (unsigned long long)bytes,ms,(unsigned long long)g_affinity_bg_admitted,(unsigned long long)g_affinity_bg_refused);
    return true;
}
static void cuda_affinity_background_finish(void) {
    /* Request/model boundary only. No foreground demand path waits for this read. */
    if(g_affinity_reading) {
        if(!ds4_ds_background_finish())abort();
        g_affinity_reading=false; /* discarded at request boundary; no late old-context admission */
    }
    if(g_affinity_copying && cudaEventSynchronize(g_affinity_copy_done)!=cudaSuccess)abort();
    g_affinity_copying=false;g_affinity_pending.ids.clear();g_affinity_read.ids.clear();
    if(g_affinity_stage) {cudaFreeHost(g_affinity_stage);g_affinity_stage=NULL;g_affinity_stage_bytes=0;}
    if(g_affinity_copy_done) {cudaEventDestroy(g_affinity_copy_done);g_affinity_copy_done=NULL;}
    if(g_affinity_bg_bytes || g_affinity_bg_admitted || g_affinity_bg_no_window)
        fprintf(stderr,"ds4: AFFINITY_BG_SUMMARY bytes=%llu admitted=%llu refused=%llu later_merit_wins=%llu thrash=%llu no_idle_window=%llu\n",
            (unsigned long long)g_affinity_bg_bytes,(unsigned long long)g_affinity_bg_admitted,
            (unsigned long long)g_affinity_bg_refused,(unsigned long long)g_affinity_bg_merit,
            (unsigned long long)g_affinity_bg_thrash,(unsigned long long)g_affinity_bg_no_window);
    g_affinity_bg_bytes=g_affinity_bg_admitted=g_affinity_bg_refused=g_affinity_bg_merit=g_affinity_bg_thrash=g_affinity_bg_no_window=0;
    memset(g_affinity_displaced,0,sizeof(g_affinity_displaced));memset(g_affinity_last_actual,0,sizeof(g_affinity_last_actual));
    memset(g_affinity_promoted,0,sizeof(g_affinity_promoted));memset(g_affinity_merit_seen,0,sizeof(g_affinity_merit_seen));
}
static bool cuda_affinity_background_prepare(uint64_t gb,uint64_t db) {
    if(!g_affinity_cap)return true;
    /* Foreground cache copies use the legacy stream. Decode graph streams are
     * blocking; the optional nonblocking shared-expert stream is unsupported. */
    if(!ds4_ds_available()) {fprintf(stderr,"ds4: background promotion requires DirectStorage; refusing silent fallback\n");return false;}
    if(g_dsv41_shared.active) {fprintf(stderr,"ds4: background promotion requires the serialized decode stream\n");return false;}
    const uint64_t needed=(2u*gb+db)*g_affinity_cap;
    if(needed>(UINT64_C(96)<<20)) {fprintf(stderr,"ds4: background staging exceeds 96MiB budget\n");return false;}
    if(g_affinity_stage_bytes<needed) {
        if(g_affinity_reading || g_affinity_copying)return false;
        if(g_affinity_stage)cudaFreeHost(g_affinity_stage);
        g_affinity_stage=NULL;g_affinity_stage_bytes=0;
        if(!cuda_ok(cudaMallocHost((void**)&g_affinity_stage,needed),"background pinned staging"))return false;
        g_affinity_stage_bytes=needed;
    }
    if(!g_affinity_copy_done && !cuda_ok(cudaEventCreateWithFlags(&g_affinity_copy_done,cudaEventDisableTiming),"background copy fence"))return false;
    return true;
}
static void cuda_affinity_background_note(const cuda_router_affinity_metrics &m,
        const cuda_router_residency &resident, const void *model,uint64_t size,
        uint32_t layer,uint32_t experts,uint64_t gate,uint64_t up,uint64_t down,uint64_t gb,uint64_t db) {
    if(!g_affinity_cap)return;
    for(uint32_t e=0;e<experts;e++) {
        if(g_affinity_promoted[layer][e] && !resident.tier[e])cuda_affinity_thrash(layer,e);
    }
    for(uint32_t j=0;j<m.selections;j++) {
        uint32_t e=(uint32_t)m.selected_ids[j];g_affinity_last_actual[layer][e]=g_affinity_decoded+1u;
        e=(uint32_t)m.original[j];
        if(g_affinity_promoted[layer][e] && resident.tier[e] && !g_affinity_merit_seen[layer][e]) {
            g_affinity_bg_merit++;g_affinity_merit_seen[layer][e]=true;
        }
    }
    g_affinity_pending.ids.clear();
    if(!g_affinity_current_biased)return;
    for(uint32_t j=0;j<m.displaced_count;j++) {
        uint32_t e=(uint32_t)m.displaced[j];
        if(g_affinity_displaced[layer][e]<65535u)g_affinity_displaced[layer][e]++;
        if(!resident.tier[e])g_affinity_pending.ids.push_back(e);
    }
    auto &b=g_affinity_pending;b.model=model;b.size=size;b.layer=layer;b.experts=experts;
    b.gate=gate;b.up=up;b.down=down;b.gb=gb;b.db=db;
    std::stable_sort(b.ids.begin(),b.ids.end(),[layer](uint32_t a,uint32_t c) {
        return cuda_affinity_value(layer,a)>cuda_affinity_value(layer,c);
    });
    if(b.ids.size()>g_affinity_cap)b.ids.resize(g_affinity_cap);
}
extern "C" int ds4_gpu_cache_affinity_background_tick(void) {
    if(!g_affinity_cap || !g_affinity_current_biased || !g_stream_expert_cache_phase_decode || g_affinity_pending.ids.empty())return 1;
    /* The async expert restore owns the single background DS slot until it completes. */
    if(ds4_gpu_expert_restore_async_state(NULL,0)==1)return 1;
    if(g_affinity_copying) {
        cudaError_t ready=cudaEventQuery(g_affinity_copy_done);
        if(ready==cudaErrorNotReady)return 1;
        if(!cuda_ok(ready,"background copy query"))return 0;
        g_affinity_copying=false;
    }
    if(g_affinity_reading)return 1;
    cudaError_t compute=cudaStreamQuery(cuda_decode_stream());
    if(compute==cudaSuccess || ds4_ds_batch_inflight()!=0 || ds4_ds_batch_count()!=0) {g_affinity_bg_no_window++;return 1;}
    if(compute!=cudaErrorNotReady)return 0;
    auto &b=g_affinity_pending;
    auto *cache=&g_stream_expert_caches[cuda_stream_expert_cache_class_index(b.gb,b.db)];
    if(!cache->valid)return 1;
    b.ids.erase(std::remove_if(b.ids.begin(),b.ids.end(),[&](uint32_t e) {
        return cuda_affinity_victim(cache,b.layer,e)==UINT32_MAX;
    }),b.ids.end());
    if(b.ids.empty())return 1;
    const uint64_t record=2u*b.gb+b.db,needed=record*b.ids.size();
    if(needed>g_affinity_stage_bytes || !g_affinity_copy_done)return 0;
    ds4_ds_background_span spans[12]={};uint32_t n=0;
    for(uint32_t j=0;j<b.ids.size();j++) {
        uint64_t e=b.ids[j];uint8_t *dst=g_affinity_stage+(uint64_t)j*record;
        if(b.gb>UINT32_MAX || b.db>UINT32_MAX)return 0;
        spans[n++]={b.gate+e*b.gb,(uint32_t)b.gb,dst};
        spans[n++]={b.up+e*b.gb,(uint32_t)b.gb,dst+b.gb};
        spans[n++]={b.down+e*b.db,(uint32_t)b.db,dst+2u*b.gb};
    }
    int submitted=ds4_ds_background_submit(spans,n);
    if(submitted<0)return 0;
    if(!submitted)return 1;
    g_affinity_read=b;g_affinity_pending.ids.clear();g_affinity_reading=true;g_affinity_bg_bytes+=needed;
    fprintf(stderr,"ds4: affinity-background submitted_step=%llu layer=%u experts=%zu bytes=%llu\n",
        (unsigned long long)g_affinity_decoded,b.layer,g_affinity_read.ids.size(),(unsigned long long)needed);
    return 1;
}
#else
static bool cuda_affinity_background_prepare(uint64_t,uint64_t) {return true;}
static void cuda_affinity_background_finish(void) {}
static bool cuda_affinity_background_harvest(void) {return true;}
static void cuda_affinity_background_note(const cuda_router_affinity_metrics &,const cuda_router_residency &,
        const void *,uint64_t,uint32_t,uint32_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t) {}
extern "C" int ds4_gpu_cache_affinity_background_tick(void) {return 1;}
#endif
