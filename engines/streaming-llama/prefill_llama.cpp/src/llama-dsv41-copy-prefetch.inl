// Members of pinned_storage. One bounded device tile, no duplicate host bank.
ggml_context_ptr prefetch_context;
ggml_backend_buffer_ptr prefetch_buffer;
ggml_tensor *prefetch_hidden=nullptr,*prefetch_pre=nullptr;
cudaEvent_t prefetch_events[4]={nullptr,nullptr,nullptr,nullptr};
uint64_t prefetch_store=0,prefetch_parent_generation=0;
uint32_t prefetch_capacity=0,prefetch_layer=0,prefetch_first=0,prefetch_rows=0;
bool prefetch_valid=false,prefetch_compute_pending=false;

void prefetch_initialize(uint32_t rows,uint64_t admitted){
    require(rows&&admitted,"Prefetch requires an explicit device allowance");
    prefetch_context.reset(ggml_init({65536,nullptr,true}));require(bool(prefetch_context),"Prefetch metadata allocation failed");
    prefetch_hidden=ggml_new_tensor_2d(prefetch_context.get(),GGML_TYPE_F32,20480,rows);
    prefetch_pre=ggml_new_tensor_2d(prefetch_context.get(),GGML_TYPE_F32,4,rows);
    const auto buft=ggml_backend_get_default_buffer_type(backend);
    require(ggml_backend_alloc_ctx_tensors_from_buft_size(prefetch_context.get(),buft)<=admitted,"Prefetch device layout exceeds admission");
    size_t cuda_free=0,cuda_total=0;ok(cudaMemGetInfo(&cuda_free,&cuda_total),"Cannot inspect prefetch allocation GPU memory");
    std::fprintf(stderr,"DS4PERF_PREFETCH_ALLOCATION rows=%u planned_bytes=%llu cuda_free=%llu cuda_total=%llu\n",rows,(unsigned long long)ggml_backend_alloc_ctx_tensors_from_buft_size(prefetch_context.get(),buft),(unsigned long long)cuda_free,(unsigned long long)cuda_total);
    prefetch_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(prefetch_context.get(),buft));
    require(prefetch_buffer&&ggml_backend_buffer_get_size(prefetch_buffer.get())<=admitted,"Prefetch device allocation refused");
    for(auto&event:prefetch_events)ok(cudaEventCreate(&event),"Prefetch timestamp event allocation failed");
    prefetch_capacity=rows;
    std::fprintf(stderr,"[dsv41-copy-prefetch] capacity_rows=%u actual_device_bytes=%llu admitted_device_bytes=%llu host_bank_duplicate=0\n",
        rows,(unsigned long long)ggml_backend_buffer_get_size(prefetch_buffer.get()),(unsigned long long)admitted);
}
void prefetch_destroy_events(){for(auto event:prefetch_events)if(event)cleanup(cudaEventDestroy(event));}
void prefetch_take(const llama_dsv41_carry_store::ram_lease&current,uint32_t layer,ggml_tensor*hidden,ggml_tensor*pre){
    require(prefetch_valid&&prefetch_store==current.store_id&&prefetch_parent_generation<UINT64_MAX&&
        prefetch_parent_generation+1==current.generation&&prefetch_layer==layer&&prefetch_first==current.first&&
        prefetch_rows==current.rows&&current.rows<=prefetch_capacity,"Prefetched input differs from its exact next lease");
    require(hidden&&pre&&ggml_is_contiguous(hidden)&&ggml_is_contiguous(pre)&&
        ggml_nbytes(hidden)==uint64_t(current.rows)*81920&&ggml_nbytes(pre)==uint64_t(current.rows)*16,
        "Prefetch destination geometry changed");
    ok(cudaMemcpyAsync(hidden->data,prefetch_hidden->data,ggml_nbytes(hidden),cudaMemcpyDeviceToDevice,copies),"Prefetched hidden D2D failed");
    ok(cudaMemcpyAsync(pre->data,prefetch_pre->data,ggml_nbytes(pre),cudaMemcpyDeviceToDevice,copies),"Prefetched pre D2D failed");
    prefetch_valid=false;
    std::fprintf(stderr,"DS4PERF_PREFETCH_CONSUMED layer=%u start=%u rows=%u bytes=%llu exact_generation=1\n",
        layer,current.first,current.rows,(unsigned long long)current.rows*81936);
}
void prefetch_begin_compute(){ok(cudaEventRecord(prefetch_events[0],compute()),"Prefetch compute-start timestamp failed");}
void prefetch_queue_next(const llama_dsv41_carry_store::ram_lookahead&next,uint32_t layer){
    const auto&lease=next.lease;
    require(prefetch_buffer&&!prefetch_valid&&lease.rows&&lease.rows<=prefetch_capacity,"Prefetch slot is busy or too small");
    ok(cudaEventRecord(prefetch_events[1],compute()),"Prefetch compute-end timestamp failed");
    const auto pending=cudaEventQuery(prefetch_events[1]);
    require(pending==cudaSuccess||pending==cudaErrorNotReady,"Prefetch compute timestamp query failed");
    prefetch_compute_pending=pending==cudaErrorNotReady;
    ok(cudaEventRecord(prefetch_events[2],copies),"Prefetch copy-start timestamp failed");
    // Tail views change only metadata; allocation and canonical F32 rows stay
    // unchanged. Copy helpers validate every physical page/record extent.
    auto hidden=*prefetch_hidden,pre=*prefetch_pre;
    for(auto*t:{&hidden,&pre}){t->ne[1]=lease.rows;t->nb[2]=t->nb[1]*lease.rows;t->nb[3]=t->nb[2];}
    queue(true,lease,0,81920,&hidden);
#ifdef DSV41_NATIVE_HALF_TEST
    if(const char*failure=std::getenv("DS4PERF_TEST_PREFETCH_ABORT")){
        if(std::strcmp(failure,"1")==0&&prefetch_compute_pending){
            std::fprintf(stderr,"DS4PERF_TEST_PREFETCH_ABORT_INFLIGHT graph_pending=1 hidden_bytes=%llu\n",(unsigned long long)lease.rows*81920);
            throw std::runtime_error("DS4PERF_TEST_PREFETCH_ABORT_AFTER_HIDDEN_COPY");
        }
    }
#endif
    queue(true,lease,81920,16,&pre);
    ok(cudaEventRecord(prefetch_events[3],copies),"Prefetch copy-end timestamp failed");
    prefetch_store=lease.store_id;prefetch_parent_generation=lease.generation;
    prefetch_layer=layer;prefetch_first=lease.first;prefetch_rows=lease.rows;prefetch_valid=true;
}
void prefetch_measure(uint32_t layer,uint32_t start){
    // Called only after the existing compute synchronization AND copy-stream
    // finish. The borrowed next rows are no longer being read when the parent
    // carry lease is completed. Both event pairs share the selected GPU clock.
    float compute_end=0,copy_begin=0,copy_end=0;
    ok(cudaEventElapsedTime(&compute_end,prefetch_events[0],prefetch_events[1]),"Prefetch compute interval failed");
    ok(cudaEventElapsedTime(&copy_begin,prefetch_events[0],prefetch_events[2]),"Prefetch copy-start interval failed");
    ok(cudaEventElapsedTime(&copy_end,prefetch_events[0],prefetch_events[3]),"Prefetch copy-end interval failed");
    const float overlap=std::max(0.f,std::min(compute_end,copy_end)-std::max(0.f,copy_begin));
    std::fprintf(stderr,"DS4PERF_PREFETCH_OVERLAP layer=%u start=%u next=%u rows=%u bytes=%llu compute_gpu_ms=%.6f copy_offset_ms=%.6f copy_gpu_ms=%.6f overlap_gpu_ms=%.6f compute_pending_at_enqueue=%u\n",
        layer,start,prefetch_first,prefetch_rows,(unsigned long long)prefetch_rows*81936,
        compute_end,copy_begin,copy_end-copy_begin,overlap,unsigned(prefetch_compute_pending));
}
