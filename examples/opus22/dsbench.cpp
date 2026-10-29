#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <dstorage.h>
#include <wrl/client.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
using Microsoft::WRL::ComPtr;
static void check(HRESULT hr, const char *what){if(FAILED(hr)){std::fprintf(stderr,"%s failed 0x%08lx\n",what,(unsigned long)hr);std::exit(2);}}
int main(int argc,char**argv){
    if(argc<6){std::fprintf(stderr,"usage: dsbench FILE OFFSET_GB SPAN_GB REQUEST_MB REPEATS [FENCE_MB]\n");return 1;}
    std::wstring path(argv[1],argv[1]+std::strlen(argv[1]));
    const uint64_t offset=uint64_t(std::atof(argv[2])*1e9)&~uint64_t(4095);
    const uint64_t span=uint64_t(std::atof(argv[3])*1e9)&~uint64_t(4095);
    const uint64_t req=uint64_t(std::atof(argv[4])*1048576)&~uint64_t(4095);
    const int repeats=std::atoi(argv[5]);
    const uint64_t fence_every=argc>6?uint64_t(std::atof(argv[6])*1048576):0;
    const int nq=argc>7?std::atoi(argv[7]):1;
    const int pause_ms=argc>8?std::atoi(argv[8]):0,keep_ms=argc>9?std::atoi(argv[9]):0;
    const bool host=argc>10&&!std::strcmp(argv[10],"host");const uint64_t ring=512ull<<20;
    void*hostbuf=host?VirtualAlloc(nullptr,ring,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE):nullptr;
    ComPtr<IDXGIFactory4> dxgi;check(CreateDXGIFactory1(IID_PPV_ARGS(&dxgi)),"dxgi");
    ComPtr<IDXGIAdapter1> adapter,best;SIZE_T bestmem=0;
    for(UINT i=0;dxgi->EnumAdapters1(i,&adapter)!=DXGI_ERROR_NOT_FOUND;++i){DXGI_ADAPTER_DESC1 d;adapter->GetDesc1(&d);if(d.DedicatedVideoMemory>bestmem){bestmem=d.DedicatedVideoMemory;best=adapter;}}
    ComPtr<ID3D12Device> dev;check(D3D12CreateDevice(best.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&dev)),"device");
    D3D12_HEAP_PROPERTIES hp{};hp.Type=D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd{};rd.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;rd.Width=span;rd.Height=1;rd.DepthOrArraySize=1;rd.MipLevels=1;rd.SampleDesc.Count=1;rd.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> buf;check(dev->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&rd,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&buf)),"buffer");
    ComPtr<IDStorageFactory> f;check(DStorageGetFactory(IID_PPV_ARGS(&f)),"factory");
    f->SetStagingBufferSize(256u*1024*1024);
    ComPtr<IDStorageFile> file;check(f->OpenFile(path.c_str(),IID_PPV_ARGS(&file)),"open");
    DSTORAGE_QUEUE_DESC qd{};qd.Capacity=DSTORAGE_MAX_QUEUE_CAPACITY;qd.Priority=DSTORAGE_PRIORITY_HIGH;qd.SourceType=DSTORAGE_REQUEST_SOURCE_FILE;qd.Device=dev.Get();
    ComPtr<IDStorageQueue> qs[8];for(int i=0;i<nq;++i)check(f->CreateQueue(&qd,IID_PPV_ARGS(&qs[i])),"queue");
    ComPtr<ID3D12Fence> fences[8];for(int i=0;i<nq;++i)check(dev->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fences[i])),"fence");
    HANDLE ev=CreateEvent(nullptr,FALSE,FALSE,nullptr);uint64_t fv=0;
    auto tiny=[&](){DSTORAGE_REQUEST rq{};rq.Options.SourceType=DSTORAGE_REQUEST_SOURCE_FILE;rq.Options.DestinationType=DSTORAGE_REQUEST_DESTINATION_BUFFER;
        rq.Source.File.Source=file.Get();rq.Source.File.Offset=offset;rq.Source.File.Size=4096;rq.Destination.Buffer.Resource=buf.Get();rq.Destination.Buffer.Offset=0;rq.Destination.Buffer.Size=4096;rq.UncompressedSize=4096;
        qs[0]->EnqueueRequest(&rq);qs[0]->EnqueueSignal(fences[0].Get(),++fv);qs[0]->Submit();HANDLE e=CreateEvent(nullptr,FALSE,FALSE,nullptr);fences[0]->SetEventOnCompletion(fv,e);WaitForSingleObject(e,INFINITE);CloseHandle(e);};
    for(int r=0;r<repeats;++r){
        if(r&&pause_ms){auto p0=std::chrono::steady_clock::now();
            while(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-p0).count()<pause_ms){
                if(keep_ms){tiny();Sleep(DWORD(keep_ms));}else Sleep(DWORD(pause_ms));}}
        auto t0=std::chrono::steady_clock::now();uint64_t since=0;int n=0;
        ++fv;
        for(uint64_t at=0,k=0;at<span;at+=req,++k){
            const uint64_t sz=std::min(req,span-at);auto&q=qs[k%nq];
            DSTORAGE_REQUEST rq{};rq.Options.SourceType=DSTORAGE_REQUEST_SOURCE_FILE;rq.Options.DestinationType=DSTORAGE_REQUEST_DESTINATION_BUFFER;
            rq.Source.File.Source=file.Get();rq.Source.File.Offset=offset+at;rq.Source.File.Size=UINT32(sz);
            if(host){rq.Options.DestinationType=DSTORAGE_REQUEST_DESTINATION_MEMORY;rq.Destination.Memory.Buffer=static_cast<char*>(hostbuf)+(at%ring);rq.Destination.Memory.Size=UINT32(sz);}
            else{rq.Destination.Buffer.Resource=buf.Get();rq.Destination.Buffer.Offset=at;rq.Destination.Buffer.Size=UINT32(sz);}rq.UncompressedSize=UINT32(sz);
            q->EnqueueRequest(&rq);++n;(void)since;(void)fence_every;
        }
        HANDLE evs[8];for(int i=0;i<nq;++i){qs[i]->EnqueueSignal(fences[i].Get(),fv);qs[i]->Submit();evs[i]=CreateEvent(nullptr,FALSE,FALSE,nullptr);check(fences[i]->SetEventOnCompletion(fv,evs[i]),"wait");}
        WaitForMultipleObjects(nq,evs,TRUE,INFINITE);for(int i=0;i<nq;++i)CloseHandle(evs[i]);
        auto&q=qs[0];
        DSTORAGE_ERROR_RECORD er{};q->RetrieveErrorRecord(&er);
        const double s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
        std::printf("dest=%s pause_ms=%d keep_ms=%d queues=%d repeat=%d requests=%d bytes=%llu seconds=%.4f GBps=%.2f failures=%u\n",host?"host":"gpu",pause_ms,keep_ms,nq,r,n,(unsigned long long)span,s,span/s/1e9,er.FailureCount);
    }
    return 0;
}

