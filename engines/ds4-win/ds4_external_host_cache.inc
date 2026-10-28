/* Reuse the standalone cache's allocator, allocation backoff, LRU/admission
 * and synchronous pinned H2D path. No model mapping or SSD reader is enabled.
 * This cache receives original bytes from the provider's VRAM victims. */
#ifdef DS4_EXTERNAL_DECODE
#include "ds4_external_host_cache.h"
static cuda_stream_host_expert_cache g_external_host_cache;
static cuda_stream_host_expert_cache g_external_host_stage;
static bool g_external_host_locked[3];
static std::vector<int32_t> g_external_host_index;
static bool g_external_host_busy=false;
static double g_external_upload_cost_ms=0;
static uint32_t g_external_upload_samples=0;
static void external_cpu_shutdown();

static void external_host_release_payload(void) {
    auto *c = &g_external_host_cache;
    char *p[3] = {c->gate_ptr,c->up_ptr,c->down_ptr};
    uint64_t n[3] = {c->gate_capacity,c->up_capacity,c->down_capacity};
    for(int k=0;k<3;++k) {
        if(g_external_host_locked[k]) VirtualUnlock(p[k],size_t(n[k]));
        g_external_host_locked[k]=false;
    }
    cuda_stream_host_expert_cache_release(c,0);
}

static bool external_host_allocate(cuda_stream_host_expert_cache *c,uint32_t n,uint64_t gate,uint64_t down,int pinned) {
    char *g=nullptr,*u=nullptr,*d=nullptr;uint64_t peak=0;
    const int previous=g_host_cache_pinned;
    const int ok=cuda_stream_host_expert_cache_try_alloc(n,gate,down,pinned,&g,&u,&d,&peak);
    g_host_cache_pinned=previous;
    if(!ok)return false;
    c->valid=1;c->capacity=n;c->gate_expert_bytes=gate;c->down_expert_bytes=down;
    c->gate_ptr=g;c->up_ptr=u;c->down_ptr=d;
    c->gate_capacity=c->up_capacity=uint64_t(n)*gate;c->down_capacity=uint64_t(n)*down;
    return true;
}

static void external_host_clear(void) {
    external_cpu_shutdown();
    external_host_release_payload();
    cuda_stream_host_expert_cache_release(&g_external_host_stage,1);
}

static uint32_t external_host_resize(uint64_t bytes, uint64_t gate, uint64_t down) {
    auto *cache = &g_external_host_cache;
    const uint64_t width = cuda_stream_expert_cache_expert_bytes(gate, down);
    if (!width || !gate || !down) return 0;
    const uint32_t wanted = (uint32_t)std::min<uint64_t>(bytes / width, DS4_CUDA_STREAM_EXPERT_MAX);
    if (cache->valid && cache->capacity == wanted && cache->gate_expert_bytes == gate && cache->down_expert_bytes == down)
        return wanted;
    /* Free before replacing: no two pinned generations, including on shrink. */
    external_host_clear();
    if(!wanted)return 0;
    g_external_host_index.assign(40u*384u,-1);
    /* WDDM can reject a multi-GiB CUDA-pinned tier despite abundant RAM.
     * Reuse ds4's pageable allocator for the payload, but VirtualLock every
     * page: this tier is never swappable. One expert's existing pinned cache
     * buffers stage both directions. No pageable pointer reaches CUDA. */
    if(!external_host_allocate(&g_external_host_stage,1,gate,down,1))return 0;
    for (uint32_t n = wanted; n; n = cuda_stream_host_expert_cache_shrunken_cap(n)) {
        std::vector<cuda_stream_host_expert_cache_slot> slots(n);
        if (!external_host_allocate(cache,n,gate,down,0)) continue;
        char *p[3]={cache->gate_ptr,cache->up_ptr,cache->down_ptr};
        uint64_t sizes[3]={cache->gate_capacity,cache->up_capacity,cache->down_capacity};
        bool locked=true;
        for(int k=0;k<3;++k){
            if(!VirtualLock(p[k],size_t(sizes[k]))){locked=false;break;}
            g_external_host_locked[k]=true;
        }
        if(!locked){external_host_release_payload();continue;}
        cache->slots.swap(slots);
        return n;
    }
    external_host_clear();
    return 0;
}

static int external_host_find(uint32_t layer, uint32_t expert) {
    if(layer>=40u||expert>=384u||!g_external_host_cache.valid)return -1;
    return g_external_host_index[layer*384u+expert];
}

static int external_host_upload(uint32_t index, char *const *dst, const uint64_t *sizes) {
    const auto began=std::chrono::steady_clock::now();
    auto *cache = &g_external_host_cache;
    if (!dst || !sizes || index >= cache->capacity || !cache->slots[index].valid) return 0;
    auto &slot = cache->slots[index];
    if (sizes[0] != slot.gate_expert_bytes || sizes[1] != sizes[0] || sizes[2] != slot.down_expert_bytes) return 0;
    auto *stage=&g_external_host_stage;
    const char *src[3]={cache->gate_ptr+uint64_t(index)*cache->gate_expert_bytes,
        cache->up_ptr+uint64_t(index)*cache->gate_expert_bytes,
        cache->down_ptr+uint64_t(index)*cache->down_expert_bytes};
    char *pinned[3]={stage->gate_ptr,stage->up_ptr,stage->down_ptr};
    for(int k=0;k<3;++k)memcpy(pinned[k],src[k],size_t(sizes[k]));
    int ok;
    if (sizes[0] == cache->gate_expert_bytes && sizes[2] == cache->down_expert_bytes) {
        ok = cuda_stream_host_expert_cache_copy_to_device(stage, 0, dst[0], dst[1], dst[2]);
    } else {
        /* Mixed quant layers share a maximum-width allocation, but transfer
         * only this layer's bytes, never its padding into a smaller L1 slot. */
        ok = 1;
        for (int k = 0; k < 3 && ok; ++k)
            ok = cuda_ok(cudaMemcpy(dst[k], pinned[k], size_t(sizes[k]), cudaMemcpyHostToDevice), "external RAM expert upload");
    }
    if (ok) slot.age = ++cache->tick;
    g_external_upload_cost_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count();
    ++g_external_upload_samples;
    return ok;
}

static int external_host_store(uint32_t layer, uint32_t expert, const char *const *src, const uint64_t *sizes) {
    if(layer>=40u||expert>=384u)return 0;
    auto *cache = &g_external_host_cache;
    if(g_external_host_busy)return 1; // borrowed by the concurrent CPU job
    if (!cache->valid || !src || !sizes || sizes[0] != sizes[1] ||
        sizes[0] > cache->gate_expert_bytes || sizes[2] > cache->down_expert_bytes) return 0;
    if (external_host_find(layer, expert) >= 0) return 1;
    const uint32_t index = cuda_stream_host_expert_cache_lru_slot(cache);
    auto &slot = cache->slots[index];
    if (!cuda_stream_host_expert_cache_admit(slot, layer, expert)) return 1;
    /* Each copy completes before publishing the key or reusing either slot. */
    char *dst[3] = {cache->gate_ptr + uint64_t(index)*cache->gate_expert_bytes,
        cache->up_ptr + uint64_t(index)*cache->gate_expert_bytes,
        cache->down_ptr + uint64_t(index)*cache->down_expert_bytes};
    const bool append = !slot.valid;
    if(slot.valid)g_external_host_index[slot.layer*384u+slot.expert]=-1;
    slot.valid = 0;
    char *pinned[3]={g_external_host_stage.gate_ptr,g_external_host_stage.up_ptr,g_external_host_stage.down_ptr};
    for (int k = 0; k < 3; ++k){
        if (!cuda_ok(cudaMemcpy(pinned[k], src[k], size_t(sizes[k]), cudaMemcpyDeviceToHost), "external RAM expert admission")) return 0;
        memcpy(dst[k],pinned[k],size_t(sizes[k]));
    }
    slot = {}; slot.valid = 1; slot.layer = layer; slot.expert = expert;
    slot.gate_expert_bytes = sizes[0]; slot.down_expert_bytes = sizes[2]; slot.age = ++cache->tick;
    g_external_host_index[layer*384u+expert]=int32_t(index);
    if (append) ++cache->count;
    return 1;
}

extern "C" __declspec(dllexport) int ds4x_host_cache_v1(ds4_external_host_cache_v1 *api) {
    if (!api) return 0;
    *api = {external_host_resize, external_host_find, external_host_upload, external_host_store, external_host_clear};
    return 1;
}
#endif
