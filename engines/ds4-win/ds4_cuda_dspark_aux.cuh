/* Optional resident DSpark expert placement on a second CUDA device. The
 * existing drafter, target verifier and cross-device pinned bounce are reused.
 * Only first-two-stage routed MoE dispatches execute there; shared projections,
 * attention, head, embeddings and stage2 remain on the target device. */
static bool cuda_dspark_aux_init_context(int physical_device) {
    if(g_dspark_aux_device>=0)return g_dspark_aux_device==physical_device;
    int count=0;cudaDeviceProp prop={};
    if(g_n_gpus!=1 || cudaGetDeviceCount(&count)!=cudaSuccess || physical_device<0 ||
       physical_device>=count || physical_device==g_gpu[0].device_id ||
       cudaGetDeviceProperties(&prop,physical_device)!=cudaSuccess) {
        fprintf(stderr,"ds4: DSpark auxiliary device unavailable; expose5090,3070 in that order\n");return false;
    }
    g_dspark_aux_device=physical_device;g_gpu[1]={};g_gpu[1].device_id=physical_device;
    bool ok=true;
    WITH_DEVICE(physical_device) {
        cudaStream_t stream=NULL;cudaEvent_t event=NULL;cublasHandle_t cublas=NULL;
        ok=cuda_ok(cudaStreamCreate(&stream),"DSpark auxiliary stream");g_gpu[1].stream=stream;
        if(ok)ok=cuda_ok(cudaEventCreateWithFlags(&event,cudaEventDisableTiming),"DSpark auxiliary event");g_gpu[1].boundary_event=event;
        if(ok)ok=cublas_ok(cublasCreate(&cublas),"DSpark auxiliary cublas");g_gpu[1].cublas=cublas;g_gpu[1].cublas_ready=ok;
        if(ok)(void)cublasSetMathMode(cublas,CUBLAS_DEFAULT_MATH);
    }
    /* Reuse the validated host-bounce path; no unvalidated GeForce P2P. */
    g_gpu_peer_ok[0][1]=g_gpu_peer_ok[1][0]=0;
    fprintf(stderr,"ds4: DSpark auxiliary %s sm_%d%d device=%d; existing pinned-host activation handoff\n",prop.name,prop.major,prop.minor,physical_device);
    return ok;
}

extern "C" int ds4_gpu_dspark_aux_map(const void *map,uint64_t size,uint64_t start,
        const uint64_t *offsets,const uint64_t *sizes,uint32_t count,int device) {
    if(!map || !offsets || !sizes || (count!=6 && count!=9) || !cuda_dspark_aux_init_context(device))return 0;
    struct range {uint64_t offset,bytes;};std::vector<range> ranges;
    uint64_t aux_bytes=0;
    for(uint32_t i=0;i<count;i++) {
        if(offsets[i]<start || offsets[i]>size || sizes[i]>size-offsets[i])return 0;
        ranges.push_back({offsets[i],sizes[i]});aux_bytes+=sizes[i];
    }
    std::sort(ranges.begin(),ranges.end(),[](const range &a,const range &b){return a.offset<b.offset;});
    uint64_t cursor=start;
    for(const auto &r:ranges){if(r.offset<cursor)return 0;cursor=r.offset+r.bytes;}
    size_t available=0,total=0;
    WITH_DEVICE(device) {if(!cuda_ok(cudaMemGetInfo(&available,&total),"DSpark auxiliary free VRAM"))return 0;}
    const uint64_t reserve=UINT64_C(512)<<20;
    if(available<aux_bytes+reserve+(UINT64_C(64)<<20)) {
        fprintf(stderr,"ds4: DSpark auxiliary weights need%llu bytes plus576MiB reserve, free=%llu; refusing spill\n",(unsigned long long)aux_bytes,(unsigned long long)available);return 0;
    }
    cursor=start;
    for(const auto &r:ranges) {
        if(r.offset>cursor && !ds4_gpu_set_aux_model_map_range(map,size,cursor,r.offset-cursor))return 0;
        bool ok=false;size_t before=g_model_ranges.size();
        WITH_DEVICE(device) {ok=ds4_gpu_set_aux_model_map_range(map,size,r.offset,r.bytes)!=0;}
        if(!ok || g_model_ranges.size()!=before+1)return 0;
        g_dspark_aux_allocations[g_model_ranges.back().device_ptr]=device;
        g_model_range_bytes-=r.bytes; /* target-GPU cache accounting excludes3070 weights */
        cursor=r.offset+r.bytes;
    }
    if(cursor<size && !ds4_gpu_set_aux_model_map_range(map,size,cursor,size-cursor))return 0;
    g_gpu[1].budget_bytes=available-reserve;
    fprintf(stderr,"ds4: DSpark split resident: auxiliary=%llu target=%llu bytes; no draft weight streaming\n",(unsigned long long)aux_bytes,(unsigned long long)(size-start-aux_bytes));
    return 1;
}

static bool cuda_dspark_aux_buffer(unsigned index,uint64_t bytes) {
    if(index>=8)return false;
    ds4_gpu_tensor *&t=g_dspark_aux_buffers[index];
    if(t && t->bytes>=bytes)return true;
    if(t)ds4_gpu_tensor_free(t);
    t=ds4_gpu_tensor_alloc_ptr_on(1,bytes);return t!=NULL;
}

/* Same public MoE signature, existing routed_moe_launch and existing tensor
 * handoff. No drafter, weights conversion, target routing or KV path here. */
extern "C" int ds4_gpu_routed_moe_batch_aux_tensor(ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,ds4_gpu_tensor *up,ds4_gpu_tensor *mid,ds4_gpu_tensor *down,
        const void *model_map,uint64_t model_size,uint64_t gate_offset,uint64_t up_offset,uint64_t down_offset,
        uint32_t gate_type,uint32_t down_type,uint64_t gate_expert_bytes,uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,uint64_t down_row_bytes,uint32_t expert_in_dim,uint32_t expert_mid_dim,
        uint32_t out_dim,const ds4_gpu_tensor *selected,const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,uint32_t n_expert,float clamp,const ds4_gpu_tensor *x,
        uint32_t layer_index,uint32_t n_tokens,bool *mid_is_f16,bool force_resident) {
    (void)gate;(void)up;(void)mid;(void)down;
    if(g_dspark_aux_device<0 || g_n_gpus!=1 || !force_resident || model_map!=g_support_host_base ||
       !n_tokens || n_tokens>16 || n_total_expert!=128 || n_expert!=3 ||
       !((gate_type==12 && down_type==12) || (gate_type==10 && down_type==10) || (gate_type==11 && down_type==11)))return 0;
    if(mid_is_f16)*mid_is_f16=false;
    bool shared=g_dsv41_shared.active;g_dsv41_shared.active=false;
    auto saved_peer=g_xdev_force_cuda_peer;auto saved_bounce=g_xdev_force_host_bounce;
    g_xdev_force_cuda_peer=0;g_xdev_force_host_bounce=1;
    bool ok=cuda_ok(cudaDeviceSynchronize(),"DSpark target handoff fence");
    g_n_gpus=2;g_current_logical_tier=-1;
    const uint64_t pairs=(uint64_t)n_tokens*n_expert;
    const uint64_t bytes[]={pairs*expert_mid_dim*4,pairs*expert_mid_dim*4,pairs*expert_mid_dim*4,
        pairs*out_dim*4,(uint64_t)n_tokens*out_dim*4,(uint64_t)n_tokens*expert_in_dim*4,pairs*4,pairs*4};
    for(unsigned i=0;ok && i<8;i++)ok=cuda_dspark_aux_buffer(i,bytes[i]);
    auto **b=g_dspark_aux_buffers;
    if(ok)ok=ds4_gpu_tensor_copy_xdev(b[5],x,bytes[5]) &&
        ds4_gpu_tensor_copy_xdev(b[6],selected,bytes[6]) && ds4_gpu_tensor_copy_xdev(b[7],weights,bytes[7]);
    if(ok)ok=ds4_gpu_set_current_device_fenced(1)==0;
    if(ok)ok=routed_moe_launch(b[4],b[0],b[1],b[2],b[3],model_map,model_size,
        gate_offset,up_offset,down_offset,gate_type,down_type,gate_expert_bytes,gate_row_bytes,
        down_expert_bytes,down_row_bytes,expert_in_dim,expert_mid_dim,out_dim,b[6],b[7],
        n_total_expert,n_expert,clamp,b[5],layer_index,n_tokens,0,0)!=0;
    if(ok)ok=cuda_ok(cudaDeviceSynchronize(),"DSpark auxiliary result fence") &&
        ds4_gpu_tensor_copy_xdev(out,b[4],bytes[4]);
    if(ds4_gpu_set_current_device_fenced(0)!=0)ok=false;
    if(ok)ok=cuda_ok(cudaDeviceSynchronize(),"DSpark target result fence");
    g_n_gpus=1;g_current_logical_tier=0;g_dsv41_shared.active=shared;
    g_xdev_force_cuda_peer=saved_peer;g_xdev_force_host_bounce=saved_bounce;
    if(!ok)fprintf(stderr,"ds4: resident auxiliary expert dispatch failed at draft stage%u; refusing fallback\n",layer_index);
    return ok;
}

static void cuda_dspark_aux_cleanup(void) {
    if(g_dspark_aux_device<0)return;
    const int saved=g_n_gpus;g_n_gpus=2;
    for(auto &t:g_dspark_aux_buffers){if(t)ds4_gpu_tensor_free(t);t=NULL;}
    WITH_DEVICE(g_dspark_aux_device) {
        if(g_gpu[1].scratch)(void)cudaFree(g_gpu[1].scratch);
        if(g_gpu[1].boundary_event)(void)cudaEventDestroy((cudaEvent_t)g_gpu[1].boundary_event);
        if(g_gpu[1].stream)(void)cudaStreamDestroy((cudaStream_t)g_gpu[1].stream);
        if(g_gpu[1].cublas)(void)cublasDestroy((cublasHandle_t)g_gpu[1].cublas);
    }
    g_gpu[1]={};g_dspark_aux_device=-1;g_n_gpus=saved;
    if(saved)(void)cudaSetDevice(g_gpu[0].device_id);
}

static bool g_dspark_full_scope=false,g_dspark_full_saved_shared=false;
static int g_dspark_full_saved_peer=0,g_dspark_full_saved_bounce=0;
extern "C" int ds4_gpu_dspark_aux_scope(int enter) {
    if(enter) {
        if(g_dspark_full_scope || g_dspark_aux_device<0 || g_n_gpus!=1)return 0;
        if(!cuda_ok(cudaDeviceSynchronize(),"full draft target fence"))return 0;
        g_dspark_full_saved_shared=g_dsv41_shared.active;g_dsv41_shared.active=false;
        g_dspark_full_saved_peer=g_xdev_force_cuda_peer;g_dspark_full_saved_bounce=g_xdev_force_host_bounce;
        g_xdev_force_cuda_peer=0;g_xdev_force_host_bounce=1;g_n_gpus=2;g_current_logical_tier=-1;
        g_dspark_full_scope=true;
        if(ds4_gpu_set_current_device_fenced(1)==0)return 1;
    }
    if(!g_dspark_full_scope)return 0;
    g_n_gpus=2;
    bool ok=cuda_ok(cudaDeviceSynchronize(),"full draft auxiliary fence");
    if(ds4_gpu_set_current_device_fenced(0)!=0)ok=false;
    g_n_gpus=1;g_current_logical_tier=0;g_dsv41_shared.active=g_dspark_full_saved_shared;
    g_xdev_force_cuda_peer=g_dspark_full_saved_peer;g_xdev_force_host_bounce=g_dspark_full_saved_bounce;
    g_dspark_full_scope=false;return ok;
}
extern "C" int ds4_gpu_dspark_aux_device(int logical) {
    if(!g_dspark_full_scope || (logical!=0 && logical!=1))return 0;
    /* The streaming target owns the single-device range cache, not the
     * multi-tier selective cache. Restore its native resolver during the
     * shared embedding/head operations; rejoin both devices before handoff. */
    g_n_gpus=2;
    if(ds4_gpu_set_current_device_fenced(logical)!=0)return 0;
    if(logical==0)g_n_gpus=1;
    return 1;
}
extern "C" ds4_gpu_tensor *ds4_gpu_dspark_aux_alloc(uint64_t bytes) {
    return g_dspark_full_scope ? ds4_gpu_tensor_alloc_ptr_on(1,bytes) : NULL;
}
extern "C" int ds4_gpu_dspark_aux_copy(ds4_gpu_tensor *dst,const ds4_gpu_tensor *src,uint64_t bytes) {
    return g_dspark_full_scope && ds4_gpu_tensor_copy_xdev(dst,src,bytes);
}
extern "C" int ds4_gpu_dspark_aux_map_all(const void *map,uint64_t size,uint64_t start,int device) {
    if(!map || start>=size || !cuda_dspark_aux_init_context(device))return 0;
    size_t available=0,total=0;
    WITH_DEVICE(device){if(!cuda_ok(cudaMemGetInfo(&available,&total),"full draft free VRAM"))return 0;}
    const uint64_t bytes=size-start,reserve=UINT64_C(576)<<20;
    if(available<bytes+reserve) {
        fprintf(stderr,"ds4: whole auxiliary draft needs%llu weight bytes plus576MiB; free=%llu; refusing spill\n",(unsigned long long)bytes,(unsigned long long)available);return 0;
    }
    bool ok=false;size_t previous=g_model_ranges.size();
    WITH_DEVICE(device){ok=ds4_gpu_set_aux_model_map_range(map,size,start,bytes)!=0;}
    if(!ok || g_model_ranges.size()!=previous+1)return 0;
    g_dspark_aux_allocations[g_model_ranges.back().device_ptr]=device;g_model_range_bytes-=bytes;
    g_gpu[1].budget_bytes=available-(UINT64_C(512)<<20);
    fprintf(stderr,"ds4: DSpark entire sidecar auxiliary resident: auxiliary=%llu target=0 bytes; shared parent embedding/head remain on target\n",(unsigned long long)bytes);
    return 1;
}
