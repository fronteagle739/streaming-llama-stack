#include "../engines/streaming-llama/prefill_llama.cpp/src/llama-dsv41-locked-ram.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <utility>
static bool fail_next_control_allocation=false;
void *operator new(size_t bytes){if(std::exchange(fail_next_control_allocation,false))throw std::bad_alloc();if(void *p=std::malloc(bytes))return p;throw std::bad_alloc();}
void operator delete(void *p)noexcept{std::free(p);}
void operator delete(void *p,size_t)noexcept{std::free(p);}
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
int main(int argc,char **argv) {
    SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX);
    _set_abort_behavior(0,_WRITE_ABORT_MSG|_CALL_REPORTFAULT);
    const size_t bytes=argc>1?size_t(std::stoull(argv[1])):2*1024*1024;
    assert(bytes && bytes<=(8ull<<30));
    auto memory=llama_dsv41_locked_ram::allocate(bytes,bytes);
    assert(memory && memory[0]==0 && memory[bytes-1]==0);
    SYSTEM_INFO system;GetSystemInfo(&system);
    for(size_t offset=0;offset<bytes;offset+=system.dwPageSize) {
        PSAPI_WORKING_SET_EX_INFORMATION info{};info.VirtualAddress=memory.get()+offset;
        assert(QueryWorkingSetEx(GetCurrentProcess(),&info,sizeof(info)));
        assert(info.VirtualAttributes.Valid && info.VirtualAttributes.Locked);
    }
    MEMORY_BASIC_INFORMATION region{};assert(VirtualQuery(memory.get(),&region,sizeof(region)));
    assert(region.Type==MEM_PRIVATE && region.AllocationBase==memory.get());
    memory[0]=19;memory[bytes-1]=23;
    void *released=memory.get();memory.reset();
    assert(VirtualQuery(released,&region,sizeof(region)) && region.State==MEM_FREE);
    bool refused=false;try {auto wrong=llama_dsv41_locked_ram::allocate(bytes,bytes-1);}catch(const std::exception&){refused=true;}
    assert(refused);
    if(bytes<=64*1024*1024) {
    fail_next_control_allocation=true;bool allocation_failed=false;
    try {auto control=llama_dsv41_locked_ram::allocate(bytes,bytes);}catch(const std::bad_alloc&){allocation_failed=true;}
    assert(allocation_failed && llama_dsv41_locked_ram::state().bytes==0);
    auto first=llama_dsv41_locked_ram::allocate(bytes,bytes*2);
    auto second=llama_dsv41_locked_ram::allocate(bytes,bytes*2);
    first.reset();PSAPI_WORKING_SET_EX_INFORMATION second_info{};second_info.VirtualAddress=second.get();
    assert(QueryWorkingSetEx(GetCurrentProcess(),&second_info,sizeof(second_info)) && second_info.VirtualAttributes.Locked);
    second.reset();assert(llama_dsv41_locked_ram::state().bytes==0);
    }
    puts("PASS: carry pages are valid/locked/private, zeroed and writable; release and admission refusal verified");
}
#else
int main(){puts("SKIP: Windows working-set lock proof");}
#endif
