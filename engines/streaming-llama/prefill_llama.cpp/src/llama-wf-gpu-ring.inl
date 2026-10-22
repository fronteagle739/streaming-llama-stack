// Windows/CUDA DirectStorage only. Reuses the original wave plan, resource,
// two compute banks, DMA thread and release scheduler; no file-I/O fallback.
static constexpr uint64_t wf_ring_cancel_tag=0x4453345045524635ull;
[[noreturn]] static void wf_ring_unsafe(const char* why) {
    WF_ERR("DS4PERF_RING_UNSAFE_QUIESCENCE: %s; refusing resource frees",why);
    fflush(stderr);TerminateProcess(GetCurrentProcess(),79);std::abort();
}
static bool wf_ring_error(wf_context*wf,const char*why){WF_ERR("DS4PERF_RING_REFUSED: %s",why);wf->ds_failed.store(true,std::memory_order_release);return false;}
static bool wf_ring_cuda(wf_context*wf,cudaError_t status,const char*why){
    if(status==cudaSuccess)return true;WF_ERR("DS4PERF_RING_CUDA: %s: %s",why,cudaGetErrorString(status));return wf_ring_error(wf,why);
}
static bool wf_ring_fence(wf_context*wf,wf_context::ds_ring_fence&f){
    if(FAILED(wf->d3d_device->CreateFence(0,D3D12_FENCE_FLAG_SHARED,IID_PPV_ARGS(&f.fence))))return wf_ring_error(wf,"CreateFence");
    if(FAILED(wf->d3d_device->CreateSharedHandle(f.fence,nullptr,GENERIC_ALL,nullptr,&f.handle)))return wf_ring_error(wf,"CreateSharedHandle");
    cudaExternalSemaphoreHandleDesc desc{};desc.type=cudaExternalSemaphoreHandleTypeD3D12Fence;desc.handle.win32.handle=f.handle;
    return wf_ring_cuda(wf,cudaImportExternalSemaphore(&f.semaphore,&desc),"Import fence");
}
static bool wf_ring_wait(wf_context*wf,wf_context::ds_ring_fence&f,uint64_t value,cudaStream_t stream){
    if(!value)return true;cudaExternalSemaphoreWaitParams p{};p.params.fence.value=value;
#ifdef DS4PERF_RING_HOST_READY_DIAGNOSTIC
    // Diagnostic only: keep all DS requests frontloaded and source-access fenced,
    // but reuse the original CPU fence wait before enqueueing a CUDA wait. This
    // isolates head-of-line blocking by a GPU wait on an unsignalled DS fence.
    if(!wf_ds_fence_wait(wf,value,f.fence))return false;
#endif
    return wf_ring_cuda(wf,cudaWaitExternalSemaphoresAsync(&f.semaphore,&p,1,stream),"Wait fence");
}
static bool wf_ring_signal(wf_context*wf,wf_context::ds_ring_fence&f,uint64_t value,cudaStream_t stream){
    cudaExternalSemaphoreSignalParams p{};p.params.fence.value=value;
    return wf_ring_cuda(wf,cudaSignalExternalSemaphoresAsync(&f.semaphore,&p,1,stream),"Signal fence");
}
static void wf_ring_event_bounded(cudaEvent_t event){
    const auto limit=std::chrono::steady_clock::now()+std::chrono::seconds(20);
    for(;;){const auto e=cudaEventQuery(event);if(e==cudaSuccess)return;
        if(e!=cudaErrorNotReady||std::chrono::steady_clock::now()>=limit)wf_ring_unsafe("CUDA event did not drain within20s");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));}
}
static bool wf_dstorage_ring_prepare(wf_context*wf){
    if(!wf->params.ds_gpu_ring_slots)return true;
    if(wf->ds_ring)return wf_ring_error(wf,"One-shot ring cannot restart");
    if(!wf->required_dstorage||wf->params.ds_gpu_ring_slots!=2||wf->gpu_bufs.size()!=2||!wf->copy_stream||!wf->cuda_compute_stream||
        !wf->layer_major_mode||wf->params.moe_mode!=WF_MOE_FETCH_ALL||wf_cache_enabled(wf)||wf->ds_direct_mode||wf->use_layer_files||
        wf->ds_waves.empty()||wf->ds_waves.size()>MAXIMUM_WAIT_OBJECTS)return wf_ring_error(wf,"Unsupported ring lifecycle/profile");
    wf->ds_ring.reset(new wf_context::ds_ring_state);auto&r=*wf->ds_ring;
    r.slot_bytes=(wf->ds_staging_size/2)&~uint64_t(65535);
    if(!r.slot_bytes)return wf_ring_error(wf,"Ring staging slot too small");
    uint64_t next=0;
    for(size_t layer=0;layer<wf->ds_waves.size();++layer){
        if(!wf->ds_mmap_spans[layer].empty()||wf->ds_waves[layer].empty())return wf_ring_error(wf,"Ring requires complete file-backed stages");
        uint64_t commands=2;
        for(const auto&wave:wf->ds_waves[layer]){
            if(wave.reqs.empty()||wave.ring_sequence!=next++||wave.nbytes>r.slot_bytes||wave.reqs.size()!=wave.req_span_idx.size())return wf_ring_error(wf,"Invalid immutable wave plan");
            commands+=wave.reqs.size()+1;r.requests+=wave.reqs.size();
            for(const auto&request:wave.reqs){
                if(request.Options.SourceType!=DSTORAGE_REQUEST_SOURCE_FILE||request.Options.DestinationType!=DSTORAGE_REQUEST_DESTINATION_BUFFER||
                    request.Destination.Buffer.Resource!=wf->ds_stage.d3d_resource||request.CancellationTag!=wf_ring_cancel_tag||
                    !request.Source.File.Size||request.Source.File.Size!=request.Destination.Buffer.Size)return wf_ring_error(wf,"Unexpected ring request");
                const uint64_t begin=(wave.ring_sequence%2)*r.slot_bytes;
                if(request.Destination.Buffer.Offset<begin||request.Destination.Buffer.Offset-begin>r.slot_bytes||
                    request.Destination.Buffer.Size>r.slot_bytes-(request.Destination.Buffer.Offset-begin))return wf_ring_error(wf,"Request crosses staging slot");
            }
        }
        if(commands>DSTORAGE_MAX_QUEUE_CAPACITY)return wf_ring_error(wf,"Stage list exceeds queue capacity; no partial enqueue fallback");
    }
    r.waves=next;
    if(!wf->gpu_bufs[0].compute_done||!wf->gpu_bufs[1].compute_done)return wf_ring_error(wf,"Missing existing compute-bank completion events");
    if(!wf_ring_fence(wf,r.read[0])||!wf_ring_fence(wf,r.read[1])||!wf_ring_fence(wf,r.drain))return false;
    if(!wf_ring_cuda(wf,cudaEventCreateWithFlags(&r.copy_cleanup,cudaEventDisableTiming),"Create copy cleanup event")||
       !wf_ring_cuda(wf,cudaEventCreateWithFlags(&r.compute_cleanup,cudaEventDisableTiming),"Create compute cleanup event"))return false;
    // Pack COMPLETE stage lists into the minimum greedy queue set. Capacity
    // includes every request and signal, so frontloading never blocks on GPU.
    std::vector<uint32_t> commands;
    for(const auto&stage:wf->ds_waves){uint32_t needed=0;for(const auto&wave:stage)needed+=uint32_t(wave.reqs.size())+1;
        if(commands.empty()||uint64_t(commands.back())+needed+2>DSTORAGE_MAX_QUEUE_CAPACITY)commands.push_back(0);
        r.stage_queue.push_back(commands.size()-1);commands.back()+=needed;
    }
    for(size_t group=0;group<commands.size();++group){
        IDStorageQueue3*q=nullptr;
        if(group==0){if(FAILED(wf->ds_queue->QueryInterface(IID_PPV_ARGS(&q))))return wf_ring_error(wf,"DirectStorage Queue3 unavailable");}
        else{DSTORAGE_QUEUE_DESC d{};d.SourceType=DSTORAGE_REQUEST_SOURCE_FILE;d.Priority=DSTORAGE_PRIORITY_NORMAL;d.Device=wf->d3d_device;
            uint32_t capacity=DSTORAGE_MIN_QUEUE_CAPACITY;while(capacity<commands[group]+2)capacity*=2;d.Capacity=UINT16(capacity);d.Name="DS4PERF packed frontloaded ring";
            if(FAILED(wf->ds_factory->CreateQueue(&d,IID_PPV_ARGS(&q))))return wf_ring_error(wf,"Cannot create bounded packed queue");}
        r.queues.push_back(q);HANDLE event=CreateEventW(nullptr,TRUE,FALSE,nullptr);if(!event)return wf_ring_error(wf,"Cannot create queue-drain event");r.ends.push_back(event);
    }
    r.armed=true;
    for(size_t layer=0;layer<wf->ds_waves.size();++layer){auto*q=r.queues[r.stage_queue[layer]];
        for(const auto&wave:wf->ds_waves[layer]){
            const uint64_t w=wave.ring_sequence;
            q->EnqueueRequests(wave.reqs.data(),UINT(wave.reqs.size()),r.drain.fence,w>=2?w-1:0,DSTORAGE_ENQUEUE_REQUEST_FLAG_FENCE_WAIT_BEFORE_SOURCE_ACCESS);
            // Each slot timeline is causally ordered: w+2 cannot read/signal
            // until CUDA consumed w and signalled drain(w+1).
            q->EnqueueSignal(r.read[w%2].fence,w/2+1);
        }
        if(layer+1==wf->ds_waves.size()||r.stage_queue[layer+1]!=r.stage_queue[layer])q->Submit();
    }
    r.prepared=true;
    WF_ERR("[dsv41-dma-ring] frontloaded_stages=%zu requests=%llu waves=%llu slots=2 slot_bytes=%llu host_wave_waits=0 host_4096_drains=0 source_access_fenced=1 queues=%zu bank_release=existing_cuda_event",
        wf->ds_waves.size(),(unsigned long long)r.requests,(unsigned long long)r.waves,(unsigned long long)r.slot_bytes,r.queues.size());
    return true;
}
static bool wf_dstorage_ring_transfer(wf_context*wf,int layer,int bank){
    auto&r=*wf->ds_ring;
    if(!r.prepared||layer!=int(r.transferred)||bank!=layer%2)return wf_ring_error(wf,"Ring stage/bank order changed");
    const auto start=std::chrono::high_resolution_clock::now();const auto stream=wf->copy_stream;auto*target=wf->gpu_bufs[bank].ptr;
    const bool trace=std::getenv("DS4PERF_RING_TRACE")!=nullptr;
    double wait_ms=0,copy_ms=0,signal_ms=0;
    auto timed=[&](double &counter,auto operation){const auto before=std::chrono::steady_clock::now();const auto result=operation();
        counter+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count();return result;};
    if(!target)return wf_ring_error(wf,"Missing ring compute bank");
    // Reuse the fetcher's existing CUDA compute_done event. The DMA scheduler
    // already observes its completion before entering. Bank reuse is CUDA-only;
    // a second D3D external compute fence is unnecessary. The slot drain fence
    // remains CUDA-signalled and gates every DirectStorage source access.
    if(layer>=2&&cudaEventQuery(wf->gpu_bufs[bank].compute_done)!=cudaSuccess)return wf_ring_error(wf,"Existing compute-bank event is not complete");
    const auto&waves=wf->ds_waves[layer];const uint64_t extent=waves.back().compute_offset+waves.back().nbytes;
    if(!timed(copy_ms,[&]{return wf_ring_cuda(wf,cudaMemsetAsync(target,0,size_t(extent),stream),"Zero ring target");}))return false;
    for(const auto&wave:waves){const uint64_t w=wave.ring_sequence;
        if(!timed(wait_ms,[&]{return wf_ring_wait(wf,r.read[w%2],w/2+1,stream);})||
           !timed(copy_ms,[&]{return wf_ring_cuda(wf,cudaMemcpyAsync((char*)target+wave.compute_offset,(char*)wf->ds_stage.cuda_ptr+(w%2)*r.slot_bytes,
               size_t(wave.nbytes),cudaMemcpyDeviceToDevice,stream),"Drain staging slot");})||
           !timed(signal_ms,[&]{return wf_ring_signal(wf,r.drain,w+1,stream);}))return false;
    }
    if(!wf_ring_cuda(wf,cudaEventRecord(wf->gpu_bufs[bank].upload_done,stream),"Record ring upload"))return false;
    const auto visibility_start=std::chrono::steady_clock::now();wf_ring_event_bounded(wf->gpu_bufs[bank].upload_done);
    if(trace)WF_ERR("[dsv41-dma-ring-api] stage=%d tid=%lu wait_api_ms=%.6f copy_api_ms=%.6f signal_api_ms=%.6f visibility_host_ms=%.6f",
        layer,GetCurrentThreadId(),wait_ms,copy_ms,signal_ms,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-visibility_start).count());
    if(!wf_dstorage_wave_succeeded(wf,r.queues[r.stage_queue[layer]],layer,-1))return false;
    wf->layers[layer].nvme_read_ms=std::chrono::duration<float,std::milli>(std::chrono::high_resolution_clock::now()-start).count();
    wf->stat_bytes_read_nvme.fetch_add(wf->layers[layer].total_bytes);++r.transferred;
#ifdef DS4PERF_RING_HOST_READY_DIAGNOSTIC
    const size_t host_wave_waits=waves.size();
#else
    const size_t host_wave_waits=0;
#endif
    WF_ERR("[dsv41-dma-ring] completed_stage=%d bank=%d waves=%zu host_wave_waits=%zu gpu_drain_signals=%zu",layer,bank,waves.size(),host_wave_waits,waves.size());
    return true;
}
static void wf_dstorage_ring_stop(wf_context*wf){
    if(!wf->ds_ring)return;auto&r=*wf->ds_ring;
    // Worker joined: no future copies can be submitted. Drain all queued CUDA
    // readers BEFORE cancellation releases source waits or any resource frees.
    if(r.armed){
        if(cudaEventRecord(r.copy_cleanup,wf->copy_stream)!=cudaSuccess||cudaEventRecord(r.compute_cleanup,wf->cuda_compute_stream)!=cudaSuccess)wf_ring_unsafe("Cannot record cleanup events");
        wf_ring_event_bounded(r.copy_cleanup);wf_ring_event_bounded(r.compute_cleanup);
        for(auto*q:r.queues)q->CancelRequestsWithTag(UINT64_MAX,wf_ring_cancel_tag);
        if(!wf_ring_signal(wf,r.drain,r.waves+2,wf->copy_stream))wf_ring_unsafe("Cannot release cancelled source waits");
        if(cudaEventRecord(r.copy_cleanup,wf->copy_stream)!=cudaSuccess)wf_ring_unsafe("Cannot record cancellation signal");wf_ring_event_bounded(r.copy_cleanup);
        for(size_t i=0;i<r.queues.size();++i){r.queues[i]->EnqueueSetEvent(r.ends[i]);r.queues[i]->Submit();}
        if(WaitForMultipleObjects(DWORD(r.ends.size()),r.ends.data(),TRUE,20000)!=WAIT_OBJECT_0)wf_ring_unsafe("DirectStorage queues did not drain within20s");
    }
    for(auto*q:r.queues)q->Release();for(auto e:r.ends)CloseHandle(e);
    for(auto*f:{&r.read[0],&r.read[1],&r.drain,&r.compute}){
        if(f->semaphore&&cudaDestroyExternalSemaphore(f->semaphore)!=cudaSuccess)wf_ring_unsafe("Cannot destroy drained semaphore");
        if(f->handle)CloseHandle(f->handle);if(f->fence)f->fence->Release();}
    if(r.copy_cleanup)cudaEventDestroy(r.copy_cleanup);if(r.compute_cleanup)cudaEventDestroy(r.compute_cleanup);
    WF_ERR("[dsv41-dma-ring] quiesced=1 transferred_stages=%u released_stages=%u queues=%zu unsafe_frees=0",r.transferred,r.released,r.queues.size());
    wf->ds_ring.reset();
}
