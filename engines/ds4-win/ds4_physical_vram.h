#pragma once
#ifdef _WIN32
#include <windows.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Shared physical device count for imported DirectStorage buffers. CUDA's
// WDDM free count can charge the same imported D3D12 allocation twice.
namespace ds4_physical_vram {
inline bool free_bytes(uint64_t &available,uint64_t &total) {
    available=total=0;
    static HMODULE library=LoadLibraryExW(L"nvml.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
    if(!library)return false;
    struct memory {unsigned long long total,free,used;};
    using init_t=int(*)();using by_uuid_t=int(*)(const char*,void**);using info_t=int(*)(void*,memory*);
    static auto init=reinterpret_cast<init_t>(GetProcAddress(library,"nvmlInit_v2"));
    static auto by_uuid=reinterpret_cast<by_uuid_t>(GetProcAddress(library,"nvmlDeviceGetHandleByUUID"));
    static auto info=reinterpret_cast<info_t>(GetProcAddress(library,"nvmlDeviceGetMemoryInfo"));
    static const int ready=init?init():-1;
    const char *uuid=getenv("CUDA_VISIBLE_DEVICES");
    if(ready!=0||!by_uuid||!info||!uuid||strncmp(uuid,"GPU-",4))return false;
    void *device=nullptr;memory result{};
    if(by_uuid(uuid,&device)!=0||info(device,&result)!=0||!result.total||result.free>result.total)return false;
    available=result.free;total=result.total;return true;
}
}
#endif
