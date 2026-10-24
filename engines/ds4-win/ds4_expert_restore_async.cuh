/* DS4COMBO asynchronous expert-cache restore (DS4COMBO_RESTORE_ASYNC, default 1).
 *
 * The restore list is read on the existing LOW-priority DirectStorage background
 * queue (ds4_ds_background_submit/poll, the queue the affinity promotion uses)
 * into this module's own pinned staging, while decode step 0 fetches what it
 * needs on demand. Single foreground owner, no worker thread, ticked between
 * decode layers.
 *
 * Slot safety: the restore never reads into a cache slot. A batch is published
 * only after its DS read has completed, under a device synchronize, with the
 * slot bytes copied synchronously; the slot is registered (valid=1) only after
 * its bytes have landed. An expert already resident at publication (demand
 * fetched its own copy) is skipped, so at most one writer ever owns a slot.
 *
 * Priority: items are read hottest first; each item carries the age the
 * synchronous restore would have given it (list position), reserved on the
 * tier tick at start, so demand admissions stay newer. When a tier has no free
 * slot, an item replaces the least recent slot only when that slot is older
 * than the item; otherwise newer demand residency wins and the item is counted
 * as displaced (it would have been evicted by the same admissions anyway). */
#if defined(_WIN32)
struct cuda_restore_item {
    const void *model=NULL;uint64_t size=0,gate=0,up=0,down=0,gb=0,db=0,age=0;
    uint32_t layer=0,experts=0,expert=0,rank=UINT32_MAX;int host=0;
};
static std::vector<cuda_restore_item> g_rs_items;
static size_t g_rs_next=0,g_rs_done=0,g_rs_held_begin=0,g_rs_held_n=0,g_rs_read_begin=0,g_rs_read_n=0;
static int g_rs_read_buf=0,g_rs_held_buf=0;
static uint8_t *g_rs_stage[2]={NULL,NULL};
static uint64_t g_rs_stage_bytes=0,g_rs_bytes=0;
static bool g_rs_active=false,g_rs_reading=false,g_rs_held=false,g_rs_failed=false;
static char g_rs_why[200]={0};
static uint32_t g_rs_nv=0,g_rs_nh=0,g_rs_cleared=0;
static uint32_t g_rs_pub_v=0,g_rs_pub_h=0,g_rs_already=0,g_rs_displaced=0;
static double g_rs_t0=0,g_rs_tick_s=0;
static uint64_t g_rs_tier_base[2][DS4_CUDA_STREAM_EXPERT_CACHE_CLASSES]={};
static double cuda_rs_now(void) {
    LARGE_INTEGER c,f;QueryPerformanceCounter(&c);QueryPerformanceFrequency(&f);return (double)c.QuadPart/(double)f.QuadPart;
}
extern "C" void ds4_t1_mark(const char *point);
static void cuda_rs_release(void) {
    for(int i=0;i<2;i++)if(g_rs_stage[i]) {cudaFreeHost(g_rs_stage[i]);g_rs_stage[i]=NULL;}
    g_rs_stage_bytes=0;g_rs_items.clear();g_rs_items.shrink_to_fit();
}
static int cuda_rs_fail(const char *why) {
    if(!g_rs_failed) {
        g_rs_failed=true;snprintf(g_rs_why,sizeof(g_rs_why),"%s",why);
        fprintf(stderr,"DS4COMBO_EXPERT_RESTORE_FAILED vram=%u host=%u seconds=%.3f async=1 reason=%s\n",
            g_rs_nv,g_rs_nh,cuda_rs_now()-g_rs_t0,why);fflush(stderr);
    }
    if(g_rs_reading && ds4_ds_background_finish())g_rs_reading=false;
    if(!g_rs_reading)cuda_rs_release();
    g_rs_active=false;g_rs_held=false;return 0;
}
static uint64_t cuda_rs_record(const cuda_restore_item &it) {return 2u*it.gb+it.db;}
static bool cuda_rs_resident(const cuda_restore_item &it) {
    if(it.host) {
        auto *c=&g_stream_host_expert_caches[cuda_stream_host_expert_cache_class_index(it.gb,it.db)];
        return c->valid && cuda_stream_host_expert_cache_find(c,it.model,it.size,it.layer,it.experts,it.expert,it.gate,it.up,it.down,it.gb,it.db)>=0;
    }
    auto *c=&g_stream_expert_caches[cuda_stream_expert_cache_class_index(it.gb,it.db)];
    return c->valid && cuda_stream_expert_cache_find(c,it.model,it.size,it.layer,it.experts,it.expert,it.gate,it.up,it.down,it.gb,it.db,false)>=0;
}
/* Victim for an item of age `age`: a free slot, else the least recent slot when it
 * is older than the item; UINT32_MAX when newer residency should win. */
template<class Cache> static uint32_t cuda_rs_victim(Cache *c,uint64_t age) {
    uint32_t best=UINT32_MAX;uint64_t best_age=UINT64_MAX;
    for(uint32_t i=0;i<c->capacity;i++) {
        if(!c->slots[i].valid)return i;
        if(c->slots[i].age<best_age) {best_age=c->slots[i].age;best=i;}
    }
    return best!=UINT32_MAX && best_age<age?best:UINT32_MAX;
}
static int cuda_rs_publish(void) {
    if(!g_rs_held)return 1;
    g_rs_held=false;
    /* Every queued kernel or copy that may still read a victim slot finishes first. */
    if(g_ram_hot_prefetch.active && g_ram_hot_prefetch.ready &&
       !cuda_ok(cudaEventSynchronize(g_ram_hot_prefetch.ready),"restore RAM-hot fence"))return cuda_rs_fail("RAM-hot prefetch fence failed");
    if(!cuda_ok(cudaDeviceSynchronize(),"restore publish fence"))return cuda_rs_fail("device synchronize before publish failed");
    uint64_t off=0;
    for(size_t k=0;k<g_rs_held_n;k++) {
        const cuda_restore_item &it=g_rs_items[g_rs_held_begin+k];
        const uint8_t *src=g_rs_stage[g_rs_held_buf]+off;off+=cuda_rs_record(it);
        g_rs_done++;
        if(cuda_rs_resident(it)) {g_rs_already++;continue;}
        if(it.host) {
            const uint32_t ci=cuda_stream_host_expert_cache_class_index(it.gb,it.db);
            auto *c=&g_stream_host_expert_caches[ci];
            if(!c->valid || c->gate_expert_bytes!=it.gb || c->down_expert_bytes!=it.db)return cuda_rs_fail("RAM tier vanished during restore");
            const uint32_t s=cuda_rs_victim(c,g_rs_tier_base[1][ci]+it.age);
            if(s==UINT32_MAX) {g_rs_displaced++;continue;}
            auto &e=c->slots[s];const bool append=!e.valid;e.valid=0;
            memcpy(c->gate_ptr+(uint64_t)s*it.gb,src,it.gb);memcpy(c->up_ptr+(uint64_t)s*it.gb,src+it.gb,it.gb);
            memcpy(c->down_ptr+(uint64_t)s*it.db,src+2u*it.gb,it.db);
            cuda_stream_host_expert_cache_slot n={};n.valid=1;n.model_map=it.model;n.model_size=it.size;n.layer=it.layer;
            n.n_total_expert=it.experts;n.expert=it.expert;n.gate_offset=it.gate;n.up_offset=it.up;n.down_offset=it.down;
            n.gate_expert_bytes=it.gb;n.down_expert_bytes=it.db;n.age=g_rs_tier_base[1][ci]+it.age;e=n;
            if(append && c->count<c->capacity)c->count++;
            g_rs_pub_h++;
        } else {
            const uint32_t ci=cuda_stream_expert_cache_class_index(it.gb,it.db);
            auto *c=&g_stream_expert_caches[ci];
            if(!c->valid || c->gate_expert_bytes!=it.gb || c->down_expert_bytes!=it.db)return cuda_rs_fail("VRAM tier vanished during restore");
            const uint32_t s=cuda_rs_victim(c,g_rs_tier_base[0][ci]+it.age);
            if(s==UINT32_MAX) {g_rs_displaced++;continue;}
            auto &e=c->slots[s];const bool append=!e.valid;e.valid=0;
            if(!cuda_ok(cudaMemcpy(c->gate_ptr+(uint64_t)s*it.gb,src,it.gb,cudaMemcpyHostToDevice),"restore gate upload") ||
               !cuda_ok(cudaMemcpy(c->up_ptr+(uint64_t)s*it.gb,src+it.gb,it.gb,cudaMemcpyHostToDevice),"restore up upload") ||
               !cuda_ok(cudaMemcpy(c->down_ptr+(uint64_t)s*it.db,src+2u*it.gb,it.db,cudaMemcpyHostToDevice),"restore down upload"))
                return cuda_rs_fail("VRAM upload failed");
            /* cudaMemcpy from pinned host memory is synchronous: the bytes have landed here. */
            cuda_stream_expert_cache_slot n={};n.valid=1;n.model_map=it.model;n.model_size=it.size;n.layer=it.layer;
            n.n_total_expert=it.experts;n.expert=it.expert;n.gate_offset=it.gate;n.up_offset=it.up;n.down_offset=it.down;
            n.gate_expert_bytes=it.gb;n.down_expert_bytes=it.db;n.age=g_rs_tier_base[0][ci]+it.age;n.route_rank=it.rank;e=n;
            if(append && c->count<c->capacity)c->count++;
            c->rank_worst_valid=0;g_rs_pub_v++;
        }
    }
    return 1;
}
static int cuda_rs_submit(void) {
    if(g_rs_reading || g_rs_next>=g_rs_items.size())return 1;
    if(g_affinity_reading || g_affinity_copying)return 1;
    /* Skip items demand already made resident; they cost no read. */
    while(g_rs_next<g_rs_items.size() && cuda_rs_resident(g_rs_items[g_rs_next])) {g_rs_next++;g_rs_done++;g_rs_already++;}
    if(g_rs_next>=g_rs_items.size())return 1;
    const int buf=g_rs_held?1-g_rs_held_buf:g_rs_read_buf;
    ds4_ds_background_span spans[12]={};uint32_t n=0;size_t k=0;uint64_t off=0;
    while(g_rs_next+k<g_rs_items.size() && n+3u<=12u) {
        const cuda_restore_item &it=g_rs_items[g_rs_next+k];
        if(off+cuda_rs_record(it)>g_rs_stage_bytes)break;
        uint8_t *dst=g_rs_stage[buf]+off;
        spans[n++]={it.gate+(uint64_t)it.expert*it.gb,(uint32_t)it.gb,dst};
        spans[n++]={it.up+(uint64_t)it.expert*it.gb,(uint32_t)it.gb,dst+it.gb};
        spans[n++]={it.down+(uint64_t)it.expert*it.db,(uint32_t)it.db,dst+2u*it.gb};
        off+=cuda_rs_record(it);k++;
    }
    if(!k)return cuda_rs_fail("staging smaller than one expert");
    const int r=ds4_ds_background_submit(spans,n);
    if(r<0)return cuda_rs_fail("background DirectStorage submit failed");
    if(!r)return 1;  /* queue busy or a demand batch in flight: retry next tick */
    g_rs_reading=true;g_rs_read_buf=buf;g_rs_read_begin=g_rs_next;g_rs_read_n=k;g_rs_next+=k;g_rs_bytes+=off;
    return 1;
}
static void cuda_rs_complete_if_done(void) {
    if(!g_rs_active || g_rs_reading || g_rs_held || g_rs_done<g_rs_items.size())return;
    g_rs_active=false;
    fprintf(stderr,"DS4COMBO_EXPERT_RESTORE experts=%u vram=%u host=%u seconds=%.3f cleared_prefill_slots=%u async=1 "
        "published_vram=%u published_host=%u already_resident=%u displaced_by_demand=%u read_bytes=%llu tick_seconds=%.3f\n",
        g_rs_nv+g_rs_nh,g_rs_nv,g_rs_nh,cuda_rs_now()-g_rs_t0,g_rs_cleared,g_rs_pub_v,g_rs_pub_h,g_rs_already,
        g_rs_displaced,(unsigned long long)g_rs_bytes,g_rs_tick_s);fflush(stderr);
    ds4_t1_mark("restore_complete");
    cuda_rs_release();
}
static int cuda_rs_step(bool wait) {
    if(g_rs_reading) {
        uint64_t bytes=0;double ms=0;
        int ready=wait?(ds4_ds_background_finish()?1:-1):ds4_ds_background_poll(&bytes,&ms);
        if(ready<0)return cuda_rs_fail("background DirectStorage read failed");
        if(ready) {g_rs_reading=false;g_rs_held=true;g_rs_held_buf=g_rs_read_buf;g_rs_held_begin=g_rs_read_begin;g_rs_held_n=g_rs_read_n;}
    }
    if(!cuda_rs_submit())return 0;  /* next read overlaps this publication */
    if(!cuda_rs_publish())return 0;
    cuda_rs_complete_if_done();
    return 1;
}
extern "C" int ds4_gpu_expert_restore_async_available(char *why,size_t n) {
    /* The restore runs on the first decode eval, before any streamed read has
     * brought DirectStorage up (it initialises lazily, as the prefill pipeline
     * does below): bring it up here on the model file instead of falling back. */
    const int was_up=ds4_ds_available();
    if(!was_up && (g_model_fd<0 || !ds4_ds_init_from_fd(g_model_fd))) {
        snprintf(why,n,"DirectStorage unavailable (init on fd %d failed)",g_model_fd);return 0;}
    fprintf(stderr,"DS4COMBO_RESTORE_ASYNC_DS_READY lazy_init=%d\n",!was_up);
    if(g_dsv41_shared.active) {snprintf(why,n,"nonblocking shared-expert stream active");return 0;}
    return 1;
}
extern "C" void ds4_gpu_expert_restore_async_reset(void) {
    if(g_rs_reading)(void)ds4_ds_background_finish();
    cuda_rs_release();g_rs_next=g_rs_done=0;g_rs_active=g_rs_reading=g_rs_held=g_rs_failed=false;g_rs_why[0]=0;
    g_rs_pub_v=g_rs_pub_h=g_rs_already=g_rs_displaced=0;g_rs_bytes=0;g_rs_tick_s=0;g_rs_nv=g_rs_nh=0;
}
/* age = list position within its tier (1 = coldest), as the synchronous restore orders it. */
extern "C" int ds4_gpu_expert_restore_async_add(const ds4_gpu_stream_expert_table *t,uint32_t expert,int host,uint32_t age) {
    if(!t || !t->model_map || expert>=t->n_total_expert || !t->gate_expert_bytes || !t->down_expert_bytes ||
       t->gate_expert_bytes>UINT32_MAX || t->down_expert_bytes>UINT32_MAX)return 0;
    if(!host && t->expert_tiers && t->expert_tiers[expert]!=DS4_STREAM_EXPERT_TIER_CACHE)return 1;  /* as the VRAM seed */
    cuda_restore_item it;it.model=t->model_map;it.size=t->model_size;it.gate=t->gate_offset;it.up=t->up_offset;
    it.down=t->down_offset;it.gb=t->gate_expert_bytes;it.db=t->down_expert_bytes;it.layer=t->layer;
    it.experts=t->n_total_expert;it.expert=expert;it.host=host;it.age=age;
    it.rank=t->expert_ranks?t->expert_ranks[expert]:UINT32_MAX;
    try {g_rs_items.push_back(it);} catch(...) {return 0;}
    host?g_rs_nh++:g_rs_nv++;
    return 1;
}
extern "C" int ds4_gpu_expert_restore_async_start(uint32_t cleared,char *err,size_t n) {
    g_rs_cleared=cleared;g_rs_t0=cuda_rs_now();
    if(g_stream_prefill_borrow_slots) {snprintf(err,n,"restore start during a prefill borrow");return 0;}
    /* Hottest first: VRAM items newest-first, then host items newest-first. */
    std::stable_sort(g_rs_items.begin(),g_rs_items.end(),[](const cuda_restore_item &a,const cuda_restore_item &b) {
        if(a.host!=b.host)return a.host<b.host;return a.age>b.age;
    });
    uint64_t record=0;uint32_t want[2][DS4_CUDA_STREAM_EXPERT_CACHE_CLASSES]={};
    for(const auto &it:g_rs_items) {
        record=std::max(record,cuda_rs_record(it));
        if(it.host) {
            auto *c=cuda_stream_host_expert_cache_prepare(it.gb,it.db,0);
            if(!c || !c->capacity) {snprintf(err,n,"RAM tier unavailable at layer %u",it.layer);return 0;}
            want[1][cuda_stream_host_expert_cache_class_index(it.gb,it.db)]++;
        } else {
            auto *c=cuda_stream_expert_cache_prepare(it.gb,it.db,cuda_stream_expert_cache_configured_budget());
            if(!c || !c->capacity) {snprintf(err,n,"VRAM tier unavailable at layer %u",it.layer);return 0;}
            want[0][cuda_stream_expert_cache_class_index(it.gb,it.db)]++;
        }
    }
    for(uint32_t ci=0;ci<DS4_CUDA_STREAM_EXPERT_CACHE_CLASSES;ci++) {
        auto &v=g_stream_expert_caches[ci];auto &h=g_stream_host_expert_caches[ci];
        if(want[0][ci] && want[0][ci]>v.capacity) {snprintf(err,n,"VRAM tier %u < %u listed experts",v.capacity,want[0][ci]);return 0;}
        if(want[1][ci] && want[1][ci]>h.capacity) {snprintf(err,n,"RAM tier %u < %u listed experts",h.capacity,want[1][ci]);return 0;}
        /* Reserve the synchronous restore's ages so later demand admissions stay newer. */
        g_rs_tier_base[0][ci]=v.tick;g_rs_tier_base[1][ci]=h.tick;
        uint32_t maxv=0,maxh=0;
        for(const auto &it:g_rs_items) {
            if(it.host && cuda_stream_host_expert_cache_class_index(it.gb,it.db)==ci)maxh=std::max(maxh,(uint32_t)it.age);
            if(!it.host && cuda_stream_expert_cache_class_index(it.gb,it.db)==ci)maxv=std::max(maxv,(uint32_t)it.age);
        }
        v.tick+=maxv;h.tick+=maxh;
    }
    if(g_rs_items.empty()) {g_rs_active=true;cuda_rs_complete_if_done();return 1;}
    const uint64_t per=std::max(record,std::min<uint64_t>(record*4u,UINT64_C(96)<<20));
    for(int i=0;i<2;i++)
        if(!cuda_ok(cudaMallocHost((void**)&g_rs_stage[i],per),"restore pinned staging")) {cuda_rs_release();snprintf(err,n,"restore staging allocation failed");return 0;}
    g_rs_stage_bytes=per;g_rs_active=true;g_rs_read_buf=0;
    fprintf(stderr,"DS4COMBO_EXPERT_RESTORE_ASYNC_STARTED vram=%u host=%u staging_bytes=%llu per_read_experts<=4\n",
        g_rs_nv,g_rs_nh,(unsigned long long)(2u*per));fflush(stderr);
    if(!cuda_rs_submit())return 0;
    return 1;
}
/* Between decode layers. Never during a prefill borrow or a graph capture. */
extern "C" int ds4_gpu_expert_restore_async_tick(void) {
    if(!g_rs_active)return 1;
    if(g_stream_prefill_borrow_slots || g_decode_graph_capturing)return 1;
    const double t=cuda_rs_now();
    const int ok=cuda_rs_step(false);
    g_rs_tick_s+=cuda_rs_now()-t;
    return ok;
}
/* Completes the restore synchronously (request/phase boundary, snapshot). */
extern "C" int ds4_gpu_expert_restore_async_drain(const char *why) {
    if(!g_rs_active)return !g_rs_failed;
    fprintf(stderr,"DS4COMBO_EXPERT_RESTORE_ASYNC_DRAIN at=%s done=%zu/%zu\n",why?why:"",g_rs_done,g_rs_items.size());
    for(int stalled=0;g_rs_active;) {
        const size_t before=g_rs_done;const bool was_reading=g_rs_reading;
        if(!cuda_rs_step(true))return 0;
        stalled=(!was_reading && !g_rs_reading && g_rs_done==before)?stalled+1:0;
        if(g_rs_active && stalled>1000)return cuda_rs_fail("drain made no progress (background queue refused)");
        if(stalled)Sleep(1);
    }
    return !g_rs_failed;
}
extern "C" int ds4_gpu_expert_restore_async_state(char *why,size_t n) {
    if(g_rs_failed) {if(why)snprintf(why,n,"%s",g_rs_why);return -1;}
    return g_rs_active?1:0;
}
#else
extern "C" int ds4_gpu_expert_restore_async_available(char *why,size_t n) {snprintf(why,n,"requires Windows");return 0;}
extern "C" void ds4_gpu_expert_restore_async_reset(void) {}
extern "C" int ds4_gpu_expert_restore_async_add(const ds4_gpu_stream_expert_table *,uint32_t,int,uint32_t) {return 0;}
extern "C" int ds4_gpu_expert_restore_async_start(uint32_t,char *err,size_t n) {snprintf(err,n,"requires Windows");return 0;}
extern "C" int ds4_gpu_expert_restore_async_tick(void) {return 1;}
extern "C" int ds4_gpu_expert_restore_async_drain(const char *) {return 1;}
extern "C" int ds4_gpu_expert_restore_async_state(char *,size_t) {return 0;}
#endif
