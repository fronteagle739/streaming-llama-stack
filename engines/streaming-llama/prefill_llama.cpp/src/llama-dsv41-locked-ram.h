#pragma once
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <limits>
#include <string>
#include <chrono>
#include "llama-dsv41-open-phases.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

// One anonymous, page-locked RAM tier. Allocation/locking failure has no
// pageable or file-backed fallback; the caller supplies the physical budget.
namespace llama_dsv41_locked_ram {
struct lock_state {
    std::mutex mutex;
    uint64_t bytes=0;
#ifdef _WIN32
    SIZE_T original_min=0,original_max=0;
#endif
};
inline lock_state &state(){static lock_state value;return value;}
// DS4COMBO fast switch: give a locked allocation's physical pages back while the
// process stays alive (VA stays reserved, pointers stay valid), and later lock
// fresh zeroed pages at the same address. Contents are not preserved.
inline void suspend_pages(void *memory,uint64_t bytes){
    auto &record=state();std::lock_guard<std::mutex> guard(record.mutex);
    if(!memory||!bytes||bytes>record.bytes)throw std::runtime_error("Locked carry RAM suspend geometry differs");
#ifdef _WIN32
    if(!VirtualUnlock(memory,size_t(bytes))&&GetLastError()!=ERROR_NOT_LOCKED)throw std::runtime_error("Locked carry RAM unlock failed");
    if(!VirtualFree(memory,size_t(bytes),MEM_DECOMMIT))throw std::runtime_error("Locked carry RAM decommit failed");
#else
    if(munlock(memory,bytes)!=0||madvise(memory,bytes,MADV_DONTNEED)!=0)throw std::runtime_error("Locked carry RAM release failed");
#endif
    record.bytes-=bytes;
    std::fprintf(stderr,"DS4COMBO_CARRY_RAM_SUSPENDED bytes=%llu active_bytes=%llu same_address=1\n",(unsigned long long)bytes,(unsigned long long)record.bytes);
}
inline void resume_pages(void *memory,uint64_t bytes){
    auto &record=state();std::lock_guard<std::mutex> guard(record.mutex);
    if(!memory||!bytes)throw std::runtime_error("Locked carry RAM resume geometry differs");
#ifdef _WIN32
    MEMORYSTATUSEX available{};available.dwLength=sizeof(available);
    if(!GlobalMemoryStatusEx(&available)||bytes>available.ullAvailPhys)
        throw std::runtime_error("Locked carry RAM resume exceeds available physical memory");
    if(VirtualAlloc(memory,size_t(bytes),MEM_COMMIT,PAGE_READWRITE)!=memory)throw std::runtime_error("Locked carry RAM recommit at the same address failed");
    if(!VirtualLock(memory,size_t(bytes))){const DWORD error=GetLastError();VirtualFree(memory,size_t(bytes),MEM_DECOMMIT);
        throw std::runtime_error("Resumed carry RAM cannot be locked: Windows error "+std::to_string(error));}
#else
    if(mlock(memory,bytes)!=0)throw std::runtime_error("Resumed carry RAM cannot be locked");
#endif
    record.bytes+=bytes;
    std::fprintf(stderr,"DS4COMBO_CARRY_RAM_RESUMED bytes=%llu active_bytes=%llu same_address=1\n",(unsigned long long)bytes,(unsigned long long)record.bytes);
}
inline std::shared_ptr<uint8_t[]> allocate(uint64_t bytes,uint64_t admission) {
    if(!bytes||bytes>admission||bytes>std::numeric_limits<size_t>::max())
        throw std::runtime_error("Carry RAM admission failed before allocation");
#ifdef _WIN32
    SYSTEM_INFO system;GetSystemInfo(&system);const uint64_t page=system.dwPageSize;
#else
    const uint64_t page=uint64_t(sysconf(_SC_PAGESIZE));
#endif
    if(!page||bytes>std::numeric_limits<size_t>::max()-(page-1))
        throw std::runtime_error("Carry RAM alignment overflow");
    const size_t size=size_t((bytes+page-1)/page*page);
    if(size>admission)throw std::runtime_error("Aligned carry RAM exceeds admission");
    // LLAMADECODE open phases (log only): each locked allocation's steps and free RAM around it.
    namespace phases=llama_dsv41_open_phases;
    phases::scope timed("locked_ram");
    const auto lock_wait_started=phases::clock::now();
    auto &record=state();std::unique_lock<std::mutex> guard(record.mutex);
    const double mutex_ms=phases::ms_since(lock_wait_started);
    if(record.bytes>std::numeric_limits<uint64_t>::max()-size)
        throw std::runtime_error("Locked carry byte counter overflow");
    double virtualalloc_ms=0,setws_ms=0,virtuallock_ms=0;
    uint64_t avail_before=0,avail_after=0;
#ifdef _WIN32
    MEMORYSTATUSEX available{};available.dwLength=sizeof(available);
    if(!GlobalMemoryStatusEx(&available)||size>available.ullAvailPhys)
        throw std::runtime_error("Locked carry RAM exceeds available physical memory");
    avail_before=available.ullAvailPhys;
    SIZE_T minimum=0,maximum=0;
    if(!GetProcessWorkingSetSize(GetCurrentProcess(),&minimum,&maximum))
        throw std::runtime_error("Cannot inspect carry working-set lock allowance");
    if(!record.bytes){record.original_min=minimum;record.original_max=maximum;}
    constexpr uint64_t margin=64ull<<20;
    if(record.bytes+size>std::numeric_limits<SIZE_T>::max()-margin)
        throw std::runtime_error("Carry working-set allowance overflow");
    const SIZE_T wanted=SIZE_T(record.bytes+size+margin);
    auto step=phases::clock::now();
    void *memory=VirtualAlloc(nullptr,size,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE);
    virtualalloc_ms=phases::ms_since(step);
    if(!memory)throw std::runtime_error("Anonymous carry RAM allocation failed");
    // Same order and short-circuit as before; each call timed on its own.
    DWORD lock_error=0;step=phases::clock::now();
    bool locked=SetProcessWorkingSetSize(GetCurrentProcess(),minimum>wanted?minimum:wanted,maximum>wanted?maximum:wanted)!=0;
    if(!locked)lock_error=GetLastError();
    setws_ms=phases::ms_since(step);
    if(locked){step=phases::clock::now();locked=VirtualLock(memory,size)!=0;if(!locked)lock_error=GetLastError();virtuallock_ms=phases::ms_since(step);}
    if(!locked) {
        const DWORD error=lock_error;VirtualFree(memory,0,MEM_RELEASE);
        if(!record.bytes)SetProcessWorkingSetSize(GetCurrentProcess(),record.original_min,record.original_max);
        throw std::runtime_error("Carry RAM cannot be locked: Windows error "+std::to_string(error));
    }
#else
    void *memory=mmap(nullptr,size,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    if(memory==MAP_FAILED)throw std::runtime_error("Anonymous carry RAM allocation failed");
    if(mlock(memory,size)!=0){munmap(memory,size);throw std::runtime_error("Carry RAM cannot be locked");}
#endif
    record.bytes+=size;
    const uint64_t active_bytes=record.bytes;
    guard.unlock(); // shared_ptr failure invokes its deleter; never hold this mutex then.
    auto result=std::shared_ptr<uint8_t[]>(static_cast<uint8_t*>(memory),[size](uint8_t *pointer){
        auto &locked=state();std::lock_guard<std::mutex> guard(locked.mutex);
#ifdef _WIN32
        VirtualUnlock(pointer,size);VirtualFree(pointer,0,MEM_RELEASE);
#else
        munlock(pointer,size);munmap(pointer,size);
#endif
        locked.bytes-=size;
#ifdef _WIN32
        if(!locked.bytes) {
            SIZE_T current_min=0,current_max=0;
            if(GetProcessWorkingSetSize(GetCurrentProcess(),&current_min,&current_max))
                SetProcessWorkingSetSize(GetCurrentProcess(),locked.original_min,
                    current_max>locked.original_max?current_max:locked.original_max);
        }
#endif
    });
#ifdef _WIN32
    {MEMORYSTATUSEX after{};after.dwLength=sizeof(after);if(GlobalMemoryStatusEx(&after))avail_after=after.ullAvailPhys;}
#endif
    phases::note("bytes",size);phases::note("avail_before",avail_before);phases::note_ms("mutex",mutex_ms);phases::note_ms("virtualalloc",virtualalloc_ms);
    phases::note_ms("setws",setws_ms);phases::note_ms("virtuallock",virtuallock_ms);phases::note("avail_after",avail_after);
    std::fprintf(stderr,"CARRY_RAM_LOCKED bytes=%llu active_bytes=%llu anonymous=1 virtualalloc_ms=%.1f setws_ms=%.1f virtuallock_ms=%.1f avail_before=%llu avail_after=%llu\n",
        (unsigned long long)size,(unsigned long long)active_bytes,virtualalloc_ms,setws_ms,virtuallock_ms,(unsigned long long)avail_before,(unsigned long long)avail_after);
    return result;
}
}
