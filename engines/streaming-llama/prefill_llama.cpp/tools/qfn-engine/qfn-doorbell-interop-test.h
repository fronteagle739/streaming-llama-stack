// OFFTREE diagnostic candidate. Include only in the stable qfn-op-test translation unit.
#pragma once
#include <d3d12.h>
#include <dxgi1_4.h>
#include <dstorage.h>
#include <objbase.h>
#include <wrl/client.h>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <stdexcept>

extern "C" GGML_BACKEND_API int ggml_backend_cuda_qfn_doorbell_micro_prepare(void ** function);

static int test_doorbell_interop(const backend_pair & bp, const char * filename) {
    using Microsoft::WRL::ComPtr;
    using clock = std::chrono::steady_clock;
    constexpr uint64_t bytes = 1u << 20;
    auto require = [](bool ok, const char * what) { if (!ok) throw std::runtime_error(what); };
    auto cuda = [&](cudaError_t status, const char * what) { require(status == cudaSuccess, what); };
    auto hr = [&](HRESULT status, const char * what) { require(SUCCEEDED(status), what); };
    std::vector<uint8_t> expected(2 * bytes);
    std::ifstream input(filename, std::ios::binary);
    input.read((char *) expected.data(), expected.size());
    require(uint64_t(input.gcount()) == expected.size(), "microproof needs two retained 1MiB payloads");
    require(!std::equal(expected.begin(), expected.begin()+bytes, expected.begin()+bytes), "payload rounds must differ");

    ComPtr<IDXGIFactory4> dxgi;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12Heap> heap;
    ComPtr<ID3D12Resource> resource;
    ComPtr<ID3D12Fence> completed, ready;
    ComPtr<IDStorageFactory> storage;
    ComPtr<IDStorageQueue> queue;
    ComPtr<IDStorageFile> file;
    HANDLE heap_handle=nullptr, fence_handle=nullptr, completed_event=nullptr;
    cudaExternalMemory_t imported=nullptr;
    cudaExternalSemaphore_t semaphore=nullptr;
    void * bank=nullptr, * destination=nullptr, * bell_device=nullptr, * abort_device=nullptr;
    ggml_qfn_bell * bell=nullptr;
    volatile LONG * abort_host=nullptr;
    cudaStream_t service=nullptr;
    cudaEvent_t main_done=nullptr, service_done=nullptr;
    cudaGraph_t service_graph=nullptr;
    cudaGraphExec_t service_exec=nullptr;
    cudaGraphNode_t wait_node=nullptr, copy_node=nullptr;
    ggml_context * ctx=nullptr;
    ggml_backend_buffer_t buffer=nullptr;
    std::thread worker, watchdog;
    std::atomic<bool> round_done{false};
    uint64_t active_ticket=0;
    bool com_initialized=false, graph_enabled=false;
    auto cleanup = [&]() {
        // Same ticket as any pending normal signal: never rewind the dedicated fence.
        if (abort_host) InterlockedExchange(abort_host,1);
        if (ready && active_ticket) ready->Signal(active_ticket); // cancelled data is never consumed
        round_done.store(true);
        if (worker.joinable()) worker.join();
        if (watchdog.joinable()) watchdog.join();
        // SDK Close contract: no later request can complete after this returns.
        // Joining a worker whose CPU-fence wait failed alone does NOT prove the DS bank is idle.
        if (queue) queue->Close();
        if (graph_enabled) { ggml_backend_cuda_graphs_disable_push(bp.cuda); graph_enabled=false; }
        // After CPU abort+same-ticket release and DS Close, prove both streams idle.
        // Never free shared resources after an unproven asynchronous CUDA failure.
        const auto main_status=cudaStreamSynchronize(static_cast<cudaStream_t>(ggml_backend_cuda_get_stream(bp.cuda)));
        const auto service_status=service ? cudaStreamSynchronize(service) : cudaSuccess;
        if (main_status!=cudaSuccess || service_status!=cudaSuccess) {
            std::fprintf(stderr,"DOORBELL_INTEROP CLEANUP_UNPROVEN main=%d service=%d; process teardown required\n",int(main_status),int(service_status));
            std::fflush(stderr);std::_Exit(3);
        }
        if (service_exec) cudaGraphExecDestroy(service_exec);
        if (service_graph) cudaGraphDestroy(service_graph);
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
        if (main_done) cudaEventDestroy(main_done);
        if (service_done) cudaEventDestroy(service_done);
        if (service) cudaStreamDestroy(service);
        if (destination) cudaFree(destination);
        if (bank) cudaFree(bank);
        if (imported) cudaDestroyExternalMemory(imported);
        if (semaphore) cudaDestroyExternalSemaphore(semaphore);
        if (bell) cudaFreeHost(bell);
        if (abort_host) cudaFreeHost((void *) abort_host);
        if (completed_event) CloseHandle(completed_event);
        if (fence_handle) CloseHandle(fence_handle);
        if (heap_handle) CloseHandle(heap_handle);
        file.Reset();queue.Reset();storage.Reset();completed.Reset();ready.Reset();
        resource.Reset();heap.Reset();device.Reset();adapter.Reset();dxgi.Reset();
    };
    try {
        hr(CoInitializeEx(nullptr,COINIT_MULTITHREADED),"COM MTA initialization failed");
        com_initialized=true;
        int cuda_device=0; cudaDeviceProp cuda_properties{};
        cuda(cudaGetDevice(&cuda_device),"CUDA device query failed");
        cuda(cudaGetDeviceProperties(&cuda_properties,cuda_device),"selected CUDA WDDM properties unavailable");
        const unsigned node_mask=cuda_properties.luidDeviceNodeMask;
        require(node_mask!=0,"selected CUDA WDDM LUID node mask unavailable");
        LUID luid{}; std::memcpy(&luid,cuda_properties.luid,sizeof luid);
        hr(CreateDXGIFactory1(IID_PPV_ARGS(&dxgi)),"DXGI factory failed");
        hr(dxgi->EnumAdapterByLuid(luid,IID_PPV_ARGS(&adapter)),"D3D12 adapter must match selected CUDA device");
        hr(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&device)),"D3D12 device failed");
        D3D12_HEAP_DESC heap_desc{};
        heap_desc.SizeInBytes=2*bytes; heap_desc.Alignment=D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        heap_desc.Properties.Type=D3D12_HEAP_TYPE_DEFAULT;
        heap_desc.Properties.CreationNodeMask=heap_desc.Properties.VisibleNodeMask=node_mask;
        heap_desc.Flags=D3D12_HEAP_FLAG_SHARED|D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
        hr(device->CreateHeap(&heap_desc,IID_PPV_ARGS(&heap)),"shared GPU heap failed");
        D3D12_RESOURCE_DESC resource_desc{};
        resource_desc.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER; resource_desc.Width=bytes;
        resource_desc.Height=1; resource_desc.DepthOrArraySize=1; resource_desc.MipLevels=1;
        resource_desc.SampleDesc.Count=1; resource_desc.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        hr(device->CreatePlacedResource(heap.Get(),0,&resource_desc,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&resource)),"shared bank failed");
        hr(device->CreateSharedHandle(heap.Get(),nullptr,GENERIC_ALL,nullptr,&heap_handle),"heap handle failed");
        cudaExternalMemoryHandleDesc memory_desc{};
        memory_desc.type=cudaExternalMemoryHandleTypeD3D12Heap;
        memory_desc.handle.win32.handle=heap_handle; memory_desc.size=heap_desc.SizeInBytes;
        cuda(cudaImportExternalMemory(&imported,&memory_desc),"CUDA external heap import failed");
        cudaExternalMemoryBufferDesc mapped{}; mapped.size=bytes;
        cuda(cudaExternalMemoryGetMappedBuffer(&bank,imported,&mapped),"CUDA shared bank map failed");
        hr(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&completed)),"DS CPU completion fence failed");
        hr(device->CreateFence(0,D3D12_FENCE_FLAG_SHARED,IID_PPV_ARGS(&ready)),"doorbell ready fence failed");
        hr(device->CreateSharedHandle(ready.Get(),nullptr,GENERIC_ALL,nullptr,&fence_handle),"ready fence handle failed");
        cudaExternalSemaphoreHandleDesc semaphore_desc{};
        semaphore_desc.type=cudaExternalSemaphoreHandleTypeD3D12Fence;
        semaphore_desc.handle.win32.handle=fence_handle;
        cuda(cudaImportExternalSemaphore(&semaphore,&semaphore_desc),"CUDA external fence import failed");
        completed_event=CreateEventW(nullptr,FALSE,FALSE,nullptr);
        require(completed_event!=nullptr,"CPU completion event failed");
        hr(DStorageGetFactory(IID_PPV_ARGS(&storage)),"DirectStorage factory unavailable");
        DSTORAGE_QUEUE_DESC queue_desc{};
        queue_desc.SourceType=DSTORAGE_REQUEST_SOURCE_FILE; queue_desc.Capacity=128;
        queue_desc.Priority=DSTORAGE_PRIORITY_NORMAL; queue_desc.Device=device.Get();
        hr(storage->CreateQueue(&queue_desc,IID_PPV_ARGS(&queue)),"DirectStorage GPU queue failed");
        const std::wstring source=std::filesystem::path(filename).wstring();
        hr(storage->OpenFile(source.c_str(),IID_PPV_ARGS(&file)),"DirectStorage test payload open failed");

        cuda(cudaHostAlloc((void **)&bell,sizeof(*bell),cudaHostAllocMapped),"mapped request allocation failed");
        cuda(cudaHostGetDevicePointer(&bell_device,bell,0),"mapped request device pointer failed");
        cuda(cudaHostAlloc((void **)&abort_host,sizeof(LONG),cudaHostAllocMapped),"mapped abort allocation failed");
        cuda(cudaHostGetDevicePointer(&abort_device,(void *)abort_host,0),"mapped abort device pointer failed");
        cuda(cudaMalloc(&destination,bytes),"copy destination allocation failed");
        cuda(cudaStreamCreateWithFlags(&service,cudaStreamNonBlocking),"persistent service stream failed");
        cuda(cudaEventCreateWithFlags(&main_done,cudaEventDisableTiming),"main completion event failed");
        cuda(cudaEventCreateWithFlags(&service_done,cudaEventDisableTiming),"service completion event failed");
        ctx=ggml_init({96*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true});
        require(ctx!=nullptr,"microproof GGML context failed");
        auto * ids=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,1);
        auto * table=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,1);
        auto * sequence=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,1);
        auto * result=ggml_qfn_slot_map(ctx,ids,table,sequence,bell_device,0);
        auto * graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,result);
        buffer=ggml_backend_alloc_ctx_tensors(ctx,bp.cuda);
        require(buffer!=nullptr,"microproof GGML device inputs failed");
        const int32_t selected=0; int32_t slot=7, seq=1;
        std::memset(bell,0,sizeof(*bell));bell->req=bell->ack=-1;bell->timeout_ms=500;
        ggml_backend_tensor_set(ids,&selected,0,4);ggml_backend_tensor_set(table,&slot,0,4);
        ggml_backend_tensor_set(sequence,&seq,0,4);
        // Warm/capture the REAL mapped-wait kernel using a hit before any unsignalled external wait exists.
        ggml_backend_cuda_graphs_disable_pop(bp.cuda); graph_enabled=true;
        require(ggml_backend_cuda_graph_compute_static(bp.cuda,graph)==GGML_STATUS_SUCCESS,"main static capture warmup failed");
        ggml_backend_synchronize(bp.cuda);
        void * copy_function=nullptr;
        require(ggml_backend_cuda_qfn_doorbell_micro_prepare(&copy_function)==cudaSuccess && copy_function,
                "copy kernel must warm outside all future waits");
        cuda(cudaGraphCreate(&service_graph,0),"service graph create failed");
        cudaExternalSemaphoreWaitParams wait_value{};wait_value.params.fence.value=1;
        cudaExternalSemaphoreWaitNodeParams wait_params{&semaphore,&wait_value,1};
        cuda(cudaGraphAddExternalSemaphoresWaitNode(&wait_node,service_graph,nullptr,0,&wait_params),"future wait graph node failed");
        auto * table_pointer=static_cast<int32_t *>(table->data);
        auto * bell_pointer=static_cast<ggml_qfn_bell *>(bell_device);
        auto * abort_pointer=static_cast<volatile int32_t *>(abort_device);
        uint64_t copy_bytes=bytes;
        void * args[]={&bank,&destination,&copy_bytes,&table_pointer,&bell_pointer,&abort_pointer};
        cudaKernelNodeParams copy{};copy.func=copy_function;copy.gridDim=dim3(1);copy.blockDim=dim3(256);copy.kernelParams=args;
        cuda(cudaGraphAddKernelNode(&copy_node,service_graph,&wait_node,1,&copy),"service copy/publication graph node failed");
        cuda(cudaGraphInstantiate(&service_exec,service_graph,nullptr,nullptr,0),"service graph instantiate failed");

        bool passed=true;
        for (int round=0;round<3;++round) {
            const bool force_timeout=round==2;
            active_ticket=uint64_t(round+1);seq=round+1;slot=-1;
            std::memset(bell,0,sizeof(*bell));bell->req=bell->ack=-1;bell->timeout_ms=500;
            InterlockedExchange(abort_host,0);round_done.store(false);
            ggml_backend_tensor_set(table,&slot,0,4);ggml_backend_tensor_set(sequence,&seq,0,4);
            cuda(cudaMemset(destination,0xee,bytes),"destination poison initialization failed");
            wait_value.params.fence.value=active_ticket;
            cuda(cudaGraphExecExternalSemaphoresWaitNodeSetParams(service_exec,wait_node,&wait_params),"persistent wait ticket update failed");
            std::string worker_error;
            const auto started=clock::now();
            // Independent CPU watchdog also bounds a driver that blocks future-wait submission before request publication.
            watchdog=std::thread([&]() {
                while (!round_done.load()) {
                    if (bell->error || clock::now()-started>std::chrono::milliseconds(500)) {
                        InterlockedExchange(abort_host,1);ready->Signal(active_ticket);break;
                    }
                    std::this_thread::yield();
                }
            });
            worker=std::thread([&]() {
                try {
                    const HRESULT apartment=CoInitializeEx(nullptr,COINIT_MULTITHREADED);
                    struct leave_mta {
                        bool initialized;
                        ~leave_mta() { if (initialized) CoUninitialize(); }
                    } apartment_guard{SUCCEEDED(apartment)};
                    hr(apartment,"DirectStorage worker COM MTA initialization failed");
                    while (bell->req<0 && !*abort_host) std::this_thread::yield();
                    if (*abort_host || force_timeout) return;
                    const auto requested=clock::now();
                    DSTORAGE_REQUEST request{};
                    request.Options.SourceType=DSTORAGE_REQUEST_SOURCE_FILE;
                    request.Options.DestinationType=DSTORAGE_REQUEST_DESTINATION_BUFFER;
                    request.Source.File.Source=file.Get();request.Source.File.Offset=uint64_t(round)*bytes;
                    request.Source.File.Size=UINT32(bytes);request.Destination.Buffer.Resource=resource.Get();
                    request.Destination.Buffer.Size=UINT32(bytes);request.UncompressedSize=UINT32(bytes);
                    queue->EnqueueRequest(&request);queue->EnqueueSignal(completed.Get(),active_ticket);queue->Submit();
                    hr(completed->SetEventOnCompletion(active_ticket,completed_event),"DS CPU completion event failed");
                    require(WaitForSingleObject(completed_event,5000)==WAIT_OBJECT_0,"DS CPU fence did not complete");
                    require(WaitForSingleObject(queue->GetErrorEvent(),0)==WAIT_TIMEOUT,"DS reported a request error");
                    MemoryBarrier();
                    // NO cuda*/cu* call here. The real imported-fence wait is ALREADY in the service graph.
                    queue->EnqueueSignal(ready.Get(),active_ticket);queue->Submit();
                    std::printf("DOORBELL_INTEROP_CPU round=%d request=%d DS_and_validate_ms=%.6f callback_api=DS_CPU_only\n",
                        round,int(bell->req),std::chrono::duration<double,std::milli>(clock::now()-requested).count());
                } catch (const std::exception & error) {
                    worker_error=error.what();InterlockedExchange(abort_host,1);ready->Signal(active_ticket);
                }
            });
            const auto enqueued=clock::now();
            cuda(cudaGraphLaunch(service_exec,service),"future service graph launch failed");
            const double enqueue_ms=std::chrono::duration<double,std::milli>(clock::now()-enqueued).count();
            if (!*abort_host)
                require(ggml_backend_cuda_graph_compute_static(bp.cuda,graph)==GGML_STATUS_SUCCESS,"real mapped-wait static replay failed");
            cuda(cudaEventRecord(main_done,static_cast<cudaStream_t>(ggml_backend_cuda_get_stream(bp.cuda))),"main completion record failed");
            cuda(cudaEventRecord(service_done,service),"service completion record failed");
            cudaError_t state;
            while ((state=cudaEventQuery(main_done))==cudaErrorNotReady) std::this_thread::yield();
            cuda(state,"main completion query failed");
            while ((state=cudaEventQuery(service_done))==cudaErrorNotReady) std::this_thread::yield();
            cuda(state,"service completion query failed");
            round_done.store(true);worker.join();watchdog.join();
            std::vector<uint8_t> actual(bytes);
            cuda(cudaMemcpy(actual.data(),destination,bytes,cudaMemcpyDeviceToHost),"postcompletion copy readback failed");
            int32_t actual_slot=-1,published_slot=-2;ggml_backend_tensor_get(result,&actual_slot,0,4);
            ggml_backend_tensor_get(table,&published_slot,0,4);
            const double elapsed=std::chrono::duration<double,std::milli>(clock::now()-started).count();
            const bool copied=std::equal(actual.begin(),actual.end(),expected.begin()+size_t(round%2)*bytes);
            const bool untouched=std::all_of(actual.begin(),actual.end(),[](uint8_t value){return value==0xee;});
            const bool ok=force_timeout ? bell->req==seq*64 && bell->error==seq*64 && bell->ack==-1 && published_slot==-1 && untouched
                : worker_error.empty() && !*abort_host && bell->error==0 && bell->ack==seq*64 && actual_slot==7 && published_slot==7 && copied;
            passed=passed&&ok;
            std::printf("DOORBELL_INTEROP round=%d timeout_case=%d pass=%d req=%d ack=%d error=%d abort=%ld slot=%d published_slot=%d full_bytes_equal=%d untouched=%d future_enqueue_ms=%.6f elapsed_ms=%.6f worker_error=%s\n",
                round,int(force_timeout),int(ok),int(bell->req),int(bell->ack),int(bell->error),long(*abort_host),actual_slot,published_slot,
                int(copied),int(untouched),enqueue_ms,elapsed,worker_error.c_str());
            active_ticket=0;
        }
        cleanup();if (com_initialized) CoUninitialize();
        std::printf("DOORBELL_INTEROP %s: real DS bytes/tails, monotonic future wait-node updates, CPU-only miss callback, mapped500ms deadline\n",passed?"PASS":"FAIL");
        return passed?0:1;
    } catch (const std::exception & error) {
        cleanup();if (com_initialized) CoUninitialize();
        std::fprintf(stderr,"DOORBELL_INTEROP FAIL: %s\n",error.what());return 1;
    }
}
