// Windows diagnostic: pinned copies alone and alongside real DirectStorage
// reads of an existing expert extent. No model files/output files are created.
#define NOMINMAX
#include <windows.h>
#include <wrl/client.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <dstorage.h>
#include <cuda_runtime_api.h>
#include "llama-dsv41-host-memory.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <initializer_list>
using Microsoft::WRL::ComPtr;
using clock_type=std::chrono::steady_clock;
static void check(bool ok,const char*reason){if(!ok)throw std::runtime_error(reason);}
static void cu(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
static double ms(clock_type::time_point t){return std::chrono::duration<double,std::milli>(clock_type::now()-t).count();}
int wmain(int argc,wchar_t**argv){try{
    check(argc==3,"usage: carry-copy-bench existing.gguf aligned_expert_offset");
    check(llama_dsv41_host_memory::begin(2ull<<30,3ull<<30),"Host memory admission refused");
    const uint64_t offset=_wcstoui64(argv[2],nullptr,10);check(offset%4096==0,"Expert extent must be4KiB aligned");
    HANDLE model=CreateFileW(argv[1],GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
    check(model!=INVALID_HANDLE_VALUE,"Model is moving/writable or unavailable");
    BY_HANDLE_FILE_INFORMATION identity{};check(GetFileInformationByHandle(model,&identity),"Model identity unavailable");
    const uint64_t file_bytes=(uint64_t(identity.nFileSizeHigh)<<32)|identity.nFileSizeLow;
    check(offset<=file_bytes&&(4ull<<30)<=file_bytes-offset,"Expert read extent exceeds the selected file");
    cu(cudaSetDevice(0));cu(cudaFree(nullptr));
    cudaDeviceProp properties{};cu(cudaGetDeviceProperties(&properties,0));const char*luid=properties.luid;
    ComPtr<IDXGIFactory6> dxgi;check(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&dxgi))),"DXGI factory failed");
    ComPtr<IDXGIAdapter1> adapter;
    for(UINT i=0;;++i){ComPtr<IDXGIAdapter1> next;if(dxgi->EnumAdapters1(i,&next)==DXGI_ERROR_NOT_FOUND)break;
        DXGI_ADAPTER_DESC1 d{};check(SUCCEEDED(next->GetDesc1(&d)),"Adapter descriptor failed");
        if(!std::memcmp(&d.AdapterLuid,luid,8)){adapter=next;break;}}
    check(bool(adapter),"CUDA/D3D adapter LUID mismatch");
    ComPtr<ID3D12Device> device;check(SUCCEEDED(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&device))),"D3D device failed");
    D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;heap.CreationNodeMask=heap.VisibleNodeMask=1;
    D3D12_RESOURCE_DESC desc{};desc.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;desc.Width=1ull<<30;desc.Height=1;
    desc.DepthOrArraySize=1;desc.MipLevels=1;desc.SampleDesc.Count=1;desc.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> buffer;check(SUCCEEDED(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&buffer))),"DirectStorage GPU buffer failed");
    ComPtr<IDStorageFactory> factory;check(SUCCEEDED(DStorageGetFactory(IID_PPV_ARGS(&factory))),"DirectStorage factory failed");
    check(SUCCEEDED(factory->SetStagingBufferSize(128u<<20)),"DirectStorage staging cap failed");
    ComPtr<IDStorageFile> file;check(SUCCEEDED(factory->OpenFile(argv[1],IID_PPV_ARGS(&file))),"DirectStorage existing-file lease failed");
    DSTORAGE_QUEUE_DESC qd{};qd.Capacity=256;qd.Priority=DSTORAGE_PRIORITY_HIGH;qd.SourceType=DSTORAGE_REQUEST_SOURCE_FILE;qd.Device=device.Get();
    ComPtr<IDStorageQueue> queue;check(SUCCEEDED(factory->CreateQueue(&qd,IID_PPV_ARGS(&queue))),"DirectStorage queue failed");
    ComPtr<ID3D12Fence> fence;check(SUCCEEDED(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence))),"Fence creation failed");
    constexpr size_t size=64ull<<20;constexpr unsigned copies=32;
    void *host_in=nullptr,*host_out=nullptr,*dev_in=nullptr,*dev_out=nullptr;
    cu(cudaMallocHost(&host_in,size));cu(cudaMallocHost(&host_out,size));std::memset(host_in,0x5a,size);std::memset(host_out,0,size);
    cu(cudaMalloc(&dev_in,size));cu(cudaMalloc(&dev_out,size));cu(cudaMemset(dev_out,0x6b,size));
    cudaStream_t streams[2];cudaEvent_t begin[2],end[2];
    for(int i=0;i<2;++i){cu(cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking));cu(cudaEventCreate(&begin[i]));cu(cudaEventCreate(&end[i]));}
    unsigned serial=0;
    for(bool ds:{false,true})for(unsigned directions:{1u,2u,3u}){
        check(llama_dsv41_host_memory::live(),"Physical RAM/paging stop");const auto wall=clock_type::now();
        if(ds){
            for(uint64_t i=0;i<128;++i){DSTORAGE_REQUEST request{};request.Options.SourceType=DSTORAGE_REQUEST_SOURCE_FILE;
                request.Options.DestinationType=DSTORAGE_REQUEST_DESTINATION_BUFFER;request.Source.File.Source=file.Get();
                request.Source.File.Offset=offset+i*(32ull<<20);request.Source.File.Size=32u<<20;
                request.Destination.Buffer.Resource=buffer.Get();request.Destination.Buffer.Offset=(i%32)*(32ull<<20);
                request.Destination.Buffer.Size=32u<<20;request.UncompressedSize=32u<<20;queue->EnqueueRequest(&request);}
            queue->EnqueueSignal(fence.Get(),++serial);queue->Submit();
        }
        const bool busy_begin=ds&&fence->GetCompletedValue()<serial;
        for(unsigned d=0;d<2;++d)if(directions&(1u<<d)){
            cu(cudaEventRecord(begin[d],streams[d]));
            for(unsigned i=0;i<copies;++i)cu(cudaMemcpyAsync(d?host_out:dev_in,d?dev_out:host_in,size,d?cudaMemcpyDeviceToHost:cudaMemcpyHostToDevice,streams[d]));
            cu(cudaEventRecord(end[d],streams[d]));
        }
        float times[2]={};
        for(unsigned d=0;d<2;++d)if(directions&(1u<<d)){cu(cudaStreamSynchronize(streams[d]));cu(cudaEventElapsedTime(&times[d],begin[d],end[d]));}
        const bool busy_end=ds&&fence->GetCompletedValue()<serial;
        while(ds&&fence->GetCompletedValue()<serial){check(llama_dsv41_host_memory::live(),"Physical RAM/paging stop during DS");check(ms(wall)<60000,"DirectStorage stalled for60s");Sleep(1);}
        DSTORAGE_ERROR_RECORD error{};queue->RetrieveErrorRecord(&error);check(SUCCEEDED(error.FirstFailure.HResult),"DirectStorage request failed");
        const double wall_ms=ms(wall);const uint64_t copy_bytes=uint64_t(size)*copies;
        check(!(directions&2)||static_cast<unsigned char*>(host_out)[size-1]==0x6b,"D2H payload mismatch");
        std::printf("CARRY_BANDWIDTH ds=%u directions=%u copy_bytes_per_direction=%llu h2d_ms=%.6f d2h_ms=%.6f h2d_GB_s=%.6f d2h_GB_s=%.6f ds_bytes=%llu wall_ms=%.6f ds_busy_at_copy_begin=%u ds_busy_after_copies=%u\n",
            unsigned(ds),directions,(unsigned long long)copy_bytes,double(times[0]),double(times[1]),times[0]?double(copy_bytes)/times[0]/1e6:0,
            times[1]?double(copy_bytes)/times[1]/1e6:0,(unsigned long long)(ds?4ull<<30:0),wall_ms,unsigned(busy_begin),unsigned(busy_end));std::fflush(stdout);
    }
    for(unsigned d=0;d<2;++d){cu(cudaEventDestroy(begin[d]));cu(cudaEventDestroy(end[d]));cu(cudaStreamDestroy(streams[d]));}
    cu(cudaFree(dev_in));cu(cudaFree(dev_out));cu(cudaFreeHost(host_in));cu(cudaFreeHost(host_out));
    BY_HANDLE_FILE_INFORMATION after{};check(GetFileInformationByHandle(model,&after)&&identity.dwVolumeSerialNumber==after.dwVolumeSerialNumber&&
        identity.nFileIndexHigh==after.nFileIndexHigh&&identity.nFileIndexLow==after.nFileIndexLow&&identity.nFileSizeHigh==after.nFileSizeHigh&&
        identity.nFileSizeLow==after.nFileSizeLow&&!CompareFileTime(&identity.ftLastWriteTime,&after.ftLastWriteTime),"Model identity changed");
    CloseHandle(model);std::puts("PASS copy payload and unchanged read-only model identity");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL carry copy benchmark: %s\n",e.what());return 2;}}
