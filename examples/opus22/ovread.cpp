// Unbuffered overlapped ReadFile bench: ovread FILE OFFSET_GB SPAN_GB REQUEST_MB QD
#define NOMINMAX
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
int main(int argc,char**argv){
    const uint64_t off=uint64_t(std::atof(argv[2])*1e9)&~4095ull,span=uint64_t(std::atof(argv[3])*1e9)&~4095ull,req=uint64_t(std::atof(argv[4])*1048576);const int qd=std::atoi(argv[5]);
    HANDLE f=CreateFileA(argv[1],GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_FLAG_NO_BUFFERING|FILE_FLAG_OVERLAPPED,nullptr);
    if(f==INVALID_HANDLE_VALUE){std::fprintf(stderr,"open %lu\n",GetLastError());return 2;}
    HANDLE port=CreateIoCompletionPort(f,nullptr,0,0);
    char*buf=(char*)VirtualAlloc(nullptr,req*qd,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE);
    std::vector<OVERLAPPED> ov(qd);uint64_t next=0,done=0;int inflight=0;
    auto issue=[&](int i){if(next>=span)return;ZeroMemory(&ov[i],sizeof(OVERLAPPED));uint64_t at=off+next;ov[i].Offset=DWORD(at);ov[i].OffsetHigh=DWORD(at>>32);
        DWORD sz=DWORD(std::min(req,span-next));next+=sz;if(!ReadFile(f,buf+uint64_t(i)*req,sz,nullptr,&ov[i])&&GetLastError()!=ERROR_IO_PENDING){std::fprintf(stderr,"read %lu\n",GetLastError());std::exit(3);}++inflight;};
    auto t0=std::chrono::steady_clock::now();
    for(int i=0;i<qd;++i)issue(i);
    while(inflight){DWORD n;ULONG_PTR k;LPOVERLAPPED o;if(!GetQueuedCompletionStatus(port,&n,&k,&o,INFINITE)){std::fprintf(stderr,"gqcs %lu\n",GetLastError());return 4;}
        --inflight;done+=n;issue(int(o-ov.data()));}
    double s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    std::printf("ovread qd=%d req_mb=%.2f bytes=%llu seconds=%.3f GBps=%.2f\n",qd,req/1048576.0,(unsigned long long)done,s,done/s/1e9);return 0;}
