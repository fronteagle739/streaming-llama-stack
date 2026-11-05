// QFN: engine serve mode. One JSON object per line on stdin, events as JSON lines on stdout. The slstack
// service owns HTTP, authentication and the GPU lease; the engine owns everything model-specific. Ops:
//   {"op":"chat","body":{OpenAI chat request}}  -> {"delta":{...}} ... {"done":true,"message":{...},"finish_reason":...}
//       rendered with the pack's own chat template and parsed (reasoning, Qwen3-Coder XML tool calls) by this tree's
//       common/chat (the llama-server path); deltas are OpenAI chat-completion chunk deltas.
//   {"op":"cancel"}                                   -> stops a running chat/generate at its next token (no reply)
//   {"op":"tokenize","text":S,"special":true}         -> {"tokens":[...]}
//   {"op":"detokenize","tokens":[...]}                -> {"text":S}
//   {"op":"prefill","tokens":[...]}                   -> {"n_past":N,"reused":R,"route":...,"seconds":S}
//       tokens is the whole conversation; when it extends the live sequence only the new suffix is processed
//       (short suffixes through the decode path, longer ones layer-major), otherwise a fresh prefill
//   {"op":"generate","max_tokens":N,"temperature":T,"top_p":P,"top_k":K,"min_p":M,"seed":S,"bias":B}
//                                                     -> {"token":id,"piece":S} ... {"done":true,"reason":...}
//   {"op":"stats"} -> engine figures; {"op":"quit"}
// Every reply carries "ok"; an error reply carries "error" and the engine stays usable when it can.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <bcrypt.h>
#include <psapi.h>
#include <tlhelp32.h>

#include "qfn-engine.h"
#include "qfn-tokenizer.h"

#include "chat.h"
#include "base64.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

// A hard working-set minimum is admission, not a guarantee against paging.
// Lock only resident process-private/COW host pages; never fault in device
// virtual addresses or read model weights as part of this operation. The
// normal aggregate RSS/floor/zero-paging guard remains independently active.
static void qfn_lock_host_resident(const qfn_options &opt) {
    if(opt.trim_host_heap_cache) {
        const auto started=std::chrono::steady_clock::now();
        // Documented HeapOptimizeResources releases only unused LFH caches
        // of this process. It does not select heap compatibility, corruption
        // handling, page protection, security policy or paging-observer rules.
        HEAP_OPTIMIZE_RESOURCES_INFORMATION info{};
        info.Version=HEAP_OPTIMIZE_RESOURCES_CURRENT_VERSION;
        if(!HeapSetInformation(nullptr,HeapOptimizeResources,&info,sizeof(info)))
            throw std::runtime_error("owned host heap cache optimization failed: "+std::to_string(GetLastError()));
        std::fprintf(stderr,"QFN_HOST_HEAP_CACHE_TRIM seconds=%.6f scope=current_process unused_only=1 guard_unchanged=1\n",
            std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count());
    }
    if(!opt.lock_host_resident)return;
    using query_fn=BOOL(WINAPI*)(HANDLE,PVOID,DWORD);
    static auto query=reinterpret_cast<query_fn>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"),"K32QueryWorkingSetEx"));
    if(!query)throw std::runtime_error("resident host page query is unavailable");
    SYSTEM_INFO system{};GetSystemInfo(&system);const size_t page=system.dwPageSize;
    /* Main generation owner only. Keep the bounded inventory buffers alive:
     * freeing them after pinning creates fresh heap metadata between scans,
     * exactly when a long prefill cannot return here to lock it again. */
    static std::vector<PSAPI_WORKING_SET_EX_INFORMATION> pages(4096);
    uintptr_t address=0;uint64_t admitted=0,added=0;
    const auto started=std::chrono::steady_clock::now();
    // CUDA's device virtual addresses can resemble resident private pages.
    // Obtain host ownership from Windows heaps and thread stacks instead of
    // attempting to lock the entire private virtual address space.
    static std::vector<std::pair<uintptr_t,uintptr_t>> ranges;ranges.clear();ranges.reserve(1u<<16);
    auto range=[&](const void *base,size_t size){
        if(!size)return;
        if(ranges.size()==ranges.capacity())throw std::runtime_error("host residency metadata exceeds bound");
        const auto lo=reinterpret_cast<uintptr_t>(base)&~uintptr_t(page-1);
        const auto hi=(reinterpret_cast<uintptr_t>(base)+size+page-1)&~uintptr_t(page-1);
        if(hi<=lo)throw std::runtime_error("host residency range overflow");ranges.emplace_back(lo,hi);
    };
    static std::vector<HANDLE> heaps;heaps.resize(GetProcessHeaps(0,nullptr));
    const DWORD heap_count=GetProcessHeaps(DWORD(heaps.size()),heaps.data());
    if(!heap_count||heap_count>heaps.size())throw std::runtime_error("host heap inventory changed");
    static std::vector<std::pair<uintptr_t,uintptr_t>> heap_allocations;heap_allocations.clear();heap_allocations.reserve(1u<<16);
    // Reserve before HeapLock/HeapWalk: diagnostics must not allocate while
    // enumerating the very same heap. The owner is serialized here.
    static std::vector<uintptr_t> reported_heaps,reported_stacks;
    reported_heaps.reserve(1u<<16);reported_stacks.reserve(1u<<16);
    auto heap_allocation=[&](const void *pointer) {
        const auto value=reinterpret_cast<uintptr_t>(pointer);
        auto after=std::upper_bound(heap_allocations.begin(),heap_allocations.end(),value,
            [](uintptr_t key,const auto &item){return key<item.first;});
        if(after!=heap_allocations.begin()&&value<std::prev(after)->second)return;
        MEMORY_BASIC_INFORMATION part{};
        if(!VirtualQuery(pointer,&part,sizeof(part))||!part.AllocationBase)throw std::runtime_error("owned heap allocation query failed");
        const auto base=reinterpret_cast<uintptr_t>(part.AllocationBase);uintptr_t end=base;
        while(VirtualQuery(reinterpret_cast<void*>(end),&part,sizeof(part))&&reinterpret_cast<uintptr_t>(part.AllocationBase)==base) {
            const auto next=reinterpret_cast<uintptr_t>(part.BaseAddress)+part.RegionSize;
            if(next<=end)throw std::runtime_error("non-progressing owned heap allocation");end=next;
        }
        if(heap_allocations.size()==heap_allocations.capacity())throw std::runtime_error("owned heap allocation count exceeds bound");
        range(reinterpret_cast<void*>(base),end-base);heap_allocations.insert(after,{base,end});
        if(reported_heaps.size()<reported_heaps.capacity()&&std::find(reported_heaps.begin(),reported_heaps.end(),base)==reported_heaps.end()) {
            reported_heaps.push_back(base);
            std::fprintf(stderr,"QFN_HOST_ALLOCATION kind=heap base=%p end=%p object=%p\n",reinterpret_cast<void*>(base),reinterpret_cast<void*>(end),pointer);
        }
    };
    for(DWORD i=0;i<heap_count;i++) {
        if(!HeapLock(heaps[i]))throw std::runtime_error("host heap ownership lock failed");
        DWORD error=0;
        try {
            PROCESS_HEAP_ENTRY entry{};
            while(HeapWalk(heaps[i],&entry)) {
                // LFH heaps need not emit PROCESS_HEAP_REGION markers.
                // Bind each object to its actual OS allocation and retain
                // that complete allocation once, including metadata/holes.
                if(entry.wFlags&(PROCESS_HEAP_REGION|PROCESS_HEAP_ENTRY_BUSY))heap_allocation(entry.lpData);
            }
            error=GetLastError();
        }catch(...){HeapUnlock(heaps[i]);throw;}
        if(!HeapUnlock(heaps[i])||error!=ERROR_NO_MORE_ITEMS)throw std::runtime_error("host heap inventory failed: "+std::to_string(error));
    }
    using thread_query_fn=LONG(WINAPI*)(HANDLE,ULONG,PVOID,ULONG,PULONG);
    const auto thread_query=reinterpret_cast<thread_query_fn>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"),"NtQueryInformationThread"));
    if(!thread_query)throw std::runtime_error("host stack query unavailable");
    struct basic_thread {LONG status;PVOID teb;HANDLE process,thread;ULONG_PTR affinity;LONG priority,base_priority;};
    HANDLE snapshot=CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD,0);
    if(snapshot==INVALID_HANDLE_VALUE)throw std::runtime_error("host thread inventory unavailable");
    THREADENTRY32 thread{};thread.dwSize=sizeof(thread);
    if(Thread32First(snapshot,&thread))do {
        if(thread.th32OwnerProcessID!=GetCurrentProcessId())continue;
        HANDLE handle=OpenThread(THREAD_QUERY_INFORMATION,FALSE,thread.th32ThreadID);if(!handle)continue;
        basic_thread info{};NT_TIB tib{};SIZE_T read=0;
        const bool ok=thread_query(handle,0,&info,sizeof(info),nullptr)>=0&&info.teb&&
            ReadProcessMemory(GetCurrentProcess(),info.teb,&tib,sizeof(tib),&read)&&read==sizeof(tib);
        DWORD code=0;const bool exited=GetExitCodeThread(handle,&code)&&code!=STILL_ACTIVE;CloseHandle(handle);
        if(!ok){if(exited)continue;CloseHandle(snapshot);throw std::runtime_error("owned host stack query failed");}
        range(tib.StackLimit,reinterpret_cast<uintptr_t>(tib.StackBase)-reinterpret_cast<uintptr_t>(tib.StackLimit));range(info.teb,page);
        const auto stack=reinterpret_cast<uintptr_t>(tib.StackBase);
        if(reported_stacks.size()<reported_stacks.capacity()&&std::find(reported_stacks.begin(),reported_stacks.end(),stack)==reported_stacks.end()) {
            reported_stacks.push_back(stack);
            MEMORY_BASIC_INFORMATION part{};VirtualQuery(tib.StackLimit,&part,sizeof(part));
            std::fprintf(stderr,"QFN_HOST_ALLOCATION kind=stack thread=%lu allocation=%p limit=%p base=%p teb=%p\n",thread.th32ThreadID,part.AllocationBase,tib.StackLimit,tib.StackBase,info.teb);
        }
    }while(Thread32Next(snapshot,&thread));
    CloseHandle(snapshot);
    MEMORY_BASIC_INFORMATION region{};unsigned noaccess_regions=0;
    while(VirtualQuery(reinterpret_cast<void*>(address),&region,sizeof(region))) {
        const auto base=reinterpret_cast<uintptr_t>(region.BaseAddress),end=base+region.RegionSize;
        if(end<=address)throw std::runtime_error("non-progressing host residency query");
        if(region.State==MEM_COMMIT&&region.Type==MEM_IMAGE)range(region.BaseAddress,region.RegionSize);
        // Bounded allocation-origin diagnostics only. The independent observer
        // still scans every eligible region, including inaccessible pages.
        if(region.State==MEM_COMMIT&&region.Type==MEM_PRIVATE&&(region.Protect&PAGE_NOACCESS)) {
            if(noaccess_regions<256)std::fprintf(stderr,"QFN_HOST_NOACCESS base=%p allocation=%p bytes=%zu\n",region.BaseAddress,region.AllocationBase,region.RegionSize);
            ++noaccess_regions;
        }
        address=end;
    }
    std::sort(ranges.begin(),ranges.end());size_t merged=0;
    for(const auto &item:ranges) {
        if(merged&&item.first<=ranges[merged-1].second)ranges[merged-1].second=std::max(ranges[merged-1].second,item.second);
        else ranges[merged++]=item;
    }
    ranges.resize(merged);
    // Population is restricted to actual HeapWalk-owned OS allocations, never
    // the CUDA/private-VA space or model mappings. Admit the entire prospective
    // host total before faulting demand-zero/standby heap pages into RAM.
    const int passes=opt.populate_host_heaps?2:1;
    uint64_t populated=0;
    for(int pass=0;pass<passes;pass++) {
    admitted=0;
    for(const auto &item:ranges)for(uintptr_t at=item.first;at<item.second;) {
        if(!VirtualQuery(reinterpret_cast<void*>(at),&region,sizeof(region)))throw std::runtime_error("owned host allocation disappeared");
        const auto end=std::min(item.second,reinterpret_cast<uintptr_t>(region.BaseAddress)+region.RegionSize);
        if(region.State==MEM_COMMIT&&!(region.Protect&(PAGE_GUARD|PAGE_NOACCESS))) {
            const auto heap_after=std::upper_bound(heap_allocations.begin(),heap_allocations.end(),at,
                [](uintptr_t key,const auto &v){return key<v.first;});
            const bool cpu_heap=opt.populate_host_heaps&&region.Type==MEM_PRIVATE&&heap_after!=heap_allocations.begin()&&
                at>=std::prev(heap_after)->first&&end<=std::prev(heap_after)->second;
            while(at<end) {
                const size_t count=std::min<size_t>(pages.size(),(end-at)/page);
                if(!count)break;
                for(size_t i=0;i<count;i++){pages[i].VirtualAddress=reinterpret_cast<void*>(at+i*page);pages[i].VirtualAttributes.Flags=0;}
                if(!query(GetCurrentProcess(),pages.data(),DWORD(count*sizeof(pages[0]))))
                    throw std::runtime_error("host residency page query failed: "+std::to_string(GetLastError()));
                size_t first=0;
                while(first<count) {
                    const auto managed=[&](size_t i){const auto f=pages[i].VirtualAttributes;return cpu_heap||(f.Valid&&!f.Shared);};
                    const auto selected=[&](size_t i){const auto f=pages[i].VirtualAttributes;return managed(i)&&!(f.Valid&&f.Locked);};
                    const auto paged=[&](size_t i){const auto f=pages[i].VirtualAttributes.Flags;return !(f&1)&&!(f&(1ull<<21))&&((f>>22)&3)==2;};
                    if(cpu_heap&&paged(first))throw std::runtime_error("owned readable heap page already has pagefile location; population refuses recovery");
                    if(managed(first))admitted+=page;
                    if(!selected(first)){first++;continue;}
                    size_t last=first+1;while(last<count&&selected(last)){
                        if(cpu_heap&&paged(last))throw std::runtime_error("owned readable heap page already has pagefile location; population refuses recovery");
                        admitted+=page;last++;
                    }
                    if(admitted>opt.host_budget)throw std::runtime_error("prospective private host pages exceed explicit engine budget");
                    if(pass==passes-1) {
                        if(!VirtualLock(pages[first].VirtualAddress,(last-first)*page))
                            throw std::runtime_error("admitted host residency lock failed: "+std::to_string(GetLastError()));
                        if(cpu_heap)for(size_t i=first;i<last;i++)if(!pages[i].VirtualAttributes.Valid)populated+=page;
                        added+=(last-first)*page;
                    }
                    first=last;
                }
                at+=count*page;
            }
        }
        at=end;
    }
    if(admitted>opt.host_budget)throw std::runtime_error("locked host pages exceed explicit engine budget");
    }
    if(opt.populate_host_heaps)std::fprintf(stderr,"QFN_HOST_HEAP_POPULATE newly_resident_bytes=%llu admitted_total=%llu budget=%llu protected_pages_unchanged=1 guard_unchanged=1\n",
        (unsigned long long)populated,(unsigned long long)admitted,(unsigned long long)opt.host_budget);
    std::fprintf(stderr,"QFN_HOST_PIN resident_private_bytes=%llu newly_locked_bytes=%llu seconds=%.6f budget=%llu guard_unchanged=1\n",
        (unsigned long long)admitted,(unsigned long long)added,
        std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count(),(unsigned long long)opt.host_budget);
    std::fprintf(stderr,"QFN_HOST_NOACCESS_SUMMARY regions=%u logged=%u guard_unchanged=1\n",noaccess_regions,std::min(noaccess_regions,256u));
}
#include <fstream>
#include <filesystem>

using json = nlohmann::ordered_json;

static qfn_image_span qfn_projected_span(const json &row,uint32_t width,uint32_t context) {
    if(!row.is_object()||row.size()!=6)throw std::runtime_error("unknown/missing projected image fields");
    qfn_image_span image;image.position=row.at("position").get<uint32_t>();
    image.count=row.at("count").get<uint32_t>();image.grid_height=row.at("grid_height").get<uint32_t>();
    image.grid_width=row.at("grid_width").get<uint32_t>();
    if(!image.count||uint64_t(image.position)+image.count>context||
        uint64_t(image.grid_height)*image.grid_width!=image.count)throw std::runtime_error("invalid image span geometry");
    const auto fingerprint=row.at("fingerprint").get<std::string>();
    if(fingerprint.size()!=64||fingerprint.find_first_not_of("0123456789abcdef")!=std::string::npos)
        throw std::runtime_error("invalid image pixel identity");
    for(size_t i=0;i<32;i++)image.fingerprint[i]=uint8_t(std::stoul(fingerprint.substr(i*2,2),nullptr,16));
    const auto &encoded=row.at("data").get_ref<const std::string&>();
    const auto expected=uint64_t(image.count)*width*4;
    if(encoded.size()!=((expected+2)/3)*4)throw std::runtime_error("projected image extent differs");
    const auto bytes=base64::decode(encoded,base64::alphabet::standard);
    if(bytes.size()!=expected||base64::encode(bytes)!=encoded)throw std::runtime_error("projected image block ended early or is not canonical base64");
    image.values.resize(size_t(image.count)*width);std::memcpy(image.values.data(),bytes.data(),bytes.size());return image;
}

std::string qfn_merge_sampling_defaults(const std::string & lower, const std::string & upper) {
    auto out=json::parse(lower); auto override=json::parse(upper);
    if(!out.is_object() || !override.is_object()) throw std::runtime_error("sampling defaults/overrides must be objects");
    // Preserve the parser's within-request precedence, while an explicit alias still overrides a lower default.
    for(const char * key:{"repeat_penalty","repetition_penalty","rep_pen"})
        if(override.contains(key) && !override[key].is_null()) { override["repeat_penalty"]=override[key]; break; }
    out.update(override);
    return out.dump();
}

qfn_sampling qfn_parse_sampling(const std::string & body,
    const std::function<std::vector<int32_t>(const std::string &)> & tokenize) {
    const auto b = json::parse(body);
    if(!b.is_object()) throw std::runtime_error("sampling request must be an object");
    qfn_sampling s;
    if(b.contains("do_sample")) {
        if(!b["do_sample"].is_boolean()) throw std::runtime_error("do_sample must be boolean");
        s.do_sample=b["do_sample"].get<bool>();
    }
    // Model token roles are global authenticated policy, not per-request sampler no-ops.
    for(const char * key:{"eos_token_id","bos_token_id","pad_token_id","generation_config"})
        if(b.contains(key)) throw std::runtime_error(std::string("Unsupported per-request generation override ")+key);
    // These are exposed by Studio's other backends, but not its DSV4.1 combo
    // contract. Refuse an explicit request rather than accepting a silent no-op.
    for (const char * key : {"mirostat", "mirostat_tau", "mirostat_eta", "min_keep",
            "xtc_probability", "xtc_threshold", "dynatemp_range", "dynatemp_exponent",
            "top_n_sigma", "dry_multiplier", "dry_base", "dry_allowed_length",
            "dry_penalty_last_n", "dry_sequence_breakers", "adaptive_target", "adaptive_decay",
            "samplers", "presence_context_size", "frequency_context_size"}) {
        if (b.contains(key) && !b[key].is_null())
            throw std::runtime_error(std::string("Unsupported operation: QFN does not implement sampler ") + key);
    }
    auto number = [&](const char * key, double fallback, double lo, double hi, bool integer = false) {
        if (!b.contains(key) || b[key].is_null()) return fallback;
        if (!b[key].is_number()) throw std::runtime_error(std::string(key) + " must be numeric");
        const double v = b[key].get<double>();
        if (!std::isfinite(v) || v < lo || v > hi || (integer && std::floor(v) != v))
            throw std::runtime_error(std::string("invalid ") + key);
        return v;
    };
    s.temperature = (float) number("temperature", s.temperature, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    s.top_p = (float) number("top_p", s.top_p, 0, 1);
    s.min_p = (float) number("min_p", s.min_p, 0, 1);
    s.typical_p = (float) number("typical_p", s.typical_p, 0, 1);
    s.top_k = (int) number("top_k", s.top_k, INT32_MIN, INT32_MAX, true);
    s.repeat_penalty = (float) number("repeat_penalty", number("repetition_penalty", number("rep_pen", 1, std::numeric_limits<float>::min(), std::numeric_limits<float>::max()), std::numeric_limits<float>::min(), std::numeric_limits<float>::max()), std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    s.repeat_last_n = (int) number("repeat_last_n", s.repeat_last_n, -1, INT32_MAX, true);
    s.presence_penalty = (float) number("presence_penalty", 0, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    s.frequency_penalty = (float) number("frequency_penalty", 0, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    s.residency_bias = (float) number("bias", 0, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    s.seed = (uint64_t) std::random_device{}();
    if (b.contains("seed") && !b["seed"].is_null()) {
        if (!b["seed"].is_number_integer()) throw std::runtime_error("seed must be an integer");
        if (b["seed"].is_number_unsigned() || b["seed"].get<int64_t>() >= 0) s.seed = b["seed"].get<uint64_t>();
        else if (b["seed"].get<int64_t>() != -1) throw std::runtime_error("seed must be nonnegative or -1 for automatic");
    }
    auto bias = [&](const json & key, const json & value, bool object_key = false) {
        float amount;
        if (value == false) amount = -INFINITY;
        else if (value.is_number() && std::isfinite(value.get<double>()) && std::abs(value.get<double>()) <= std::numeric_limits<float>::max()) amount = value.get<float>();
        else throw std::runtime_error("logit_bias values must be finite numbers or false");
        if (key.is_number_integer()) {
            if (key.get<int64_t>() < 0 || key.get<int64_t>() > INT32_MAX) throw std::runtime_error("logit_bias token id out of range");
            s.logit_bias[key.get<int32_t>()] = amount;
        }
        else if (key.is_string()) {
            const std::string text = key.get<std::string>();
            const bool numeric = object_key && !text.empty() && text.find_first_not_of("0123456789") == std::string::npos;
            if (numeric) s.logit_bias[(int32_t) std::stol(text)] = amount;
            else {
                const auto ids = tokenize(text);
                if (ids.empty()) throw std::runtime_error("logit_bias string has no tokens");
                for (int32_t id : ids) s.logit_bias[id] = amount;
            }
        } else throw std::runtime_error("logit_bias key must be a token id or string");
    };
    if (b.contains("logit_bias") && !b["logit_bias"].is_null()) {
        const auto & lb = b["logit_bias"];
        if (lb.is_object()) for (const auto & it : lb.items()) bias(it.key(), it.value(), true);
        else if (lb.is_array()) for (const auto & it : lb) {
            if (!it.is_array() || it.size() != 2) throw std::runtime_error("logit_bias requires [token, bias] pairs");
            bias(it[0], it[1]);
        }
        else throw std::runtime_error("logit_bias must be an object or pair array");
    }
    return s;
}

static std::vector<std::string> request_stops(const json & body) {
    std::vector<std::string> stops;
    if (!body.contains("stop") || body["stop"].is_null()) return stops;
    auto add = [&](const json & v) {
        if (!v.is_string() || v.get<std::string>().empty()) throw std::runtime_error("stop must contain nonempty strings");
        stops.push_back(v.get<std::string>());
    };
    if (body["stop"].is_array()) for (const auto & v : body["stop"]) add(v);
    else add(body["stop"]);
    return stops;
}

static int request_max_tokens(const json & body, int fallback) {
    for (const char * key : {"max_completion_tokens", "max_tokens"}) {
        if (!body.contains(key) || body[key].is_null()) continue;
        if (!body[key].is_number_integer()) throw std::runtime_error(std::string(key) + " must be an integer");
        const auto n = body[key].get<int64_t>();
        if (n < -1 || n > INT32_MAX) throw std::runtime_error(std::string("invalid ") + key);
        return n == -1 ? INT32_MAX : (int) n;
    }
    return fallback;
}

// transport: stdio, or one TCP client at a time on 127.0.0.1 (the engine stays loaded across reconnects)
namespace {
struct transport {
    SOCKET listener = INVALID_SOCKET, client = INVALID_SOCKET;
    std::string inbuf;
    std::mutex out_mu;
    bool sock = false;
    std::atomic<bool> paired{false};
    ~transport() { if(client!=INVALID_SOCKET)closesocket(client);if(listener!=INVALID_SOCKET)closesocket(listener); }

    void connect_state(const json &callback) {
        const auto host=callback.at("host").get<std::string>();
        const auto port=callback.at("port").get<int>();
        const auto nonce=callback.at("nonce").get<std::string>();
        sockaddr_in address{};address.sin_family=AF_INET;
        if(port<1||port>65535||inet_pton(AF_INET,host.c_str(),&address.sin_addr)!=1||
           nonce.size()!=32||nonce.find_first_not_of("0123456789abcdef")!=std::string::npos)
            throw std::runtime_error("invalid state callback identity/address");
        const uint32_t ip=ntohl(address.sin_addr.s_addr);
        if((ip>>24)!=10&&(ip>>24)!=127&&(ip>>16)!=0xc0a8&&(ip>>20)!=0xac1)
            throw std::runtime_error("direct state callback requires a private IPv4 address");
        address.sin_port=htons((u_short)port);client=socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);
        if(client==INVALID_SOCKET)throw std::runtime_error("state callback socket failed");
        // Outbound connection only: Windows retains its one loopback listener.
        u_long nonblock=1;ioctlsocket(client,FIONBIO,&nonblock);
        if(connect(client,(sockaddr*)&address,sizeof(address))!=0) {
            if(WSAGetLastError()!=WSAEWOULDBLOCK)throw std::runtime_error("state callback connection blocked/failed");
            fd_set writable,errors;FD_ZERO(&writable);FD_ZERO(&errors);FD_SET(client,&writable);FD_SET(client,&errors);
            timeval timeout{2,0};int error=0,len=sizeof(error);
            if(select(0,nullptr,&writable,&errors,&timeout)<=0||FD_ISSET(client,&errors)||
               getsockopt(client,SOL_SOCKET,SO_ERROR,(char*)&error,&len)||error)
                throw std::runtime_error("state callback connection blocked/failed");
        }
        nonblock=0;ioctlsocket(client,FIONBIO,&nonblock);
        const DWORD timeout=180000;setsockopt(client,SOL_SOCKET,SO_SNDTIMEO,(const char*)&timeout,sizeof(timeout));
        sock=true;
    }

    bool open_port(int port) {
        WSADATA w;
        if (WSAStartup(MAKEWORD(2, 2), &w) != 0) return false;
        listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (listener == INVALID_SOCKET) return false;
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons((u_short) port);
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (bind(listener, (sockaddr *) &a, sizeof(a)) != 0 || listen(listener, 1) != 0) return false;
        sock = true;
        return true;
    }
    bool accept_client() {
        client = accept(listener, nullptr, nullptr);
        paired = false;
        inbuf.clear();
        return client != INVALID_SOCKET;
    }
    void retire_client() {
        std::lock_guard<std::mutex> lock(out_mu);
        if(client!=INVALID_SOCKET)closesocket(client);
        client=INVALID_SOCKET;inbuf.clear();
    }
    // EOF leaves this connection owned until its queued and active commands
    // finish. A later client's replies must never share the old command owner.
    // one line (without the newline); false when the client closed
    bool getline(std::string & line) {
        if (!sock) return (bool) std::getline(std::cin, line);
        for (;;) {
            const size_t nl = inbuf.find('\n');
            if (nl != std::string::npos) {
                line = inbuf.substr(0, nl);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                inbuf.erase(0, nl + 1);
                return true;
            }
            char buf[65536];
            const int got = recv(client, buf, sizeof(buf), 0);
            if (got <= 0) return false;
            inbuf.append(buf, (size_t) got);
        }
    }
    void put(const std::string & s) {
        std::lock_guard<std::mutex> lk(out_mu);
        if (!sock) {
            std::fwrite(s.data(), 1, s.size(), stdout);
            std::fputc('\n', stdout);
            std::fflush(stdout);
            return;
        }
        if (client == INVALID_SOCKET) return;
        std::string m = s + "\n";
        size_t off = 0;
        while (off < m.size()) {
            const int sent = send(client, m.data() + off, (int) (m.size() - off), 0);
            if (sent <= 0) return;   // the client went away; the reader notices
            off += (size_t) sent;
        }
    }
    void binary(const void *p, size_t n) {
        std::lock_guard<std::mutex> lk(out_mu);
        if (!sock || client == INVALID_SOCKET) throw std::runtime_error("state streaming needs a live socket");
        const char *bytes=static_cast<const char *>(p);
        while(n) {
            int sent=send(client,bytes,int(std::min<size_t>(n,1<<20)),0);
            if(sent<=0)throw std::runtime_error("state socket closed while streaming");
            n-=sent;bytes+=sent;
        }
    }
    void read_binary(void *p,size_t n,uint32_t stall_seconds) {
        if(!sock||client==INVALID_SOCKET||!stall_seconds)throw std::runtime_error("delta upload requires live socket and stall budget");
        char *out=static_cast<char *>(p);auto progress=std::chrono::steady_clock::now();
        while(n) {
            if(!inbuf.empty()) {
                const size_t k=std::min(n,inbuf.size());std::memcpy(out,inbuf.data(),k);inbuf.erase(0,k);out+=k;n-=k;continue;
            }
            fd_set ready;FD_ZERO(&ready);FD_SET(client,&ready);timeval tick{1,0};
            const int rc=select(0,&ready,nullptr,nullptr,&tick);
            if(rc<0)throw std::runtime_error("delta socket polling failed");
            if(!rc){if(std::chrono::duration<double>(std::chrono::steady_clock::now()-progress).count()>stall_seconds)
                throw std::runtime_error("delta upload stalled");continue;}
            const int got=recv(client,out,int(std::min<size_t>(n,65536)),0);
            if(got<=0)throw std::runtime_error("delta upload closed before extent");
            out+=got;n-=size_t(got);progress=std::chrono::steady_clock::now();
        }
    }
};
transport g_io;

}  // namespace

static void emit(const json & j) {
    g_io.put(j.dump(-1, ' ', false, json::error_handler_t::replace));
}

// Opt-in test verification only. Exactly one incremental pass over bytes as
// they leave the engine; no producer file, checkpoint SHA pass or re-read.
struct state_hash {
    BCRYPT_ALG_HANDLE alg=nullptr;BCRYPT_HASH_HANDLE hash=nullptr;
    std::vector<unsigned char> object;
    explicit state_hash(bool enabled) {
        if(!enabled)return;
        ULONG bytes=0,got=0;
        if(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0 ||
           BCryptGetProperty(alg,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&bytes),sizeof(bytes),&got,0)<0)
            throw std::runtime_error("inline state SHA setup failed");
        object.resize(bytes);
        if(BCryptCreateHash(alg,&hash,object.data(),bytes,nullptr,0,0)<0)
            throw std::runtime_error("inline state SHA creation failed");
    }
    ~state_hash(){if(hash)BCryptDestroyHash(hash);if(alg)BCryptCloseAlgorithmProvider(alg,0);}
    void update(const void *p,size_t n){
        if(!hash)return;
        if(n>ULONG_MAX || BCryptHashData(hash,(PUCHAR)p,(ULONG)n,0)<0)
            throw std::runtime_error("inline state SHA update failed");
    }
    std::string finish(){
        if(!hash)return "";
        unsigned char digest[32];
        if(BCryptFinishHash(hash,digest,sizeof(digest),0)<0)throw std::runtime_error("inline state SHA finish failed");
        static const char hex[]="0123456789abcdef";std::string result;
        for(unsigned char byte:digest){result+=hex[byte>>4];result+=hex[byte&15];}return result;
    }
};

static double secs_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

// bytes at the end of s that start an incomplete UTF-8 sequence
static size_t utf8_incomplete_tail(const std::string & s) {
    const size_t n = s.size();
    for (size_t back = 1; back <= 4 && back <= n; ++back) {
        const unsigned char c = (unsigned char) s[n - back];
        if ((c & 0xC0) == 0x80) continue;          // continuation byte: keep looking for the lead byte
        size_t need = 1;
        if ((c & 0xE0) == 0xC0) need = 2; else if ((c & 0xF0) == 0xE0) need = 3; else if ((c & 0xF8) == 0xF0) need = 4;
        return need > back ? back : 0;
    }
    return 0;
}

// length of the longest suffix of text that is a proper prefix of some stop string (held back while streaming)
static size_t stop_prefix_tail(const std::string & text, const std::vector<std::string> & stops) {
    size_t best = 0;
    for (const auto & s : stops) {
        for (size_t k = std::min(s.size() - 1, text.size()); k > best; --k) {
            if (text.compare(text.size() - k, k, s, 0, k) == 0) { best = k; break; }
        }
    }
    return best;
}

namespace {

struct server {
    qfn_engine & eng;
    qfn_tokenizer & tok;
    common_chat_templates_ptr tmpls;
    size_t append_decode_max = 32;
    std::vector<int32_t> live;          // tokens the engine state holds
    std::vector<float> last_logits;     // logits after the last live token
    qfn_decode_stats total;
    std::atomic<bool> cancel{false};
    json defaults = json::object();

    server(qfn_engine & e, qfn_tokenizer & t) : eng(e), tok(t) {}

    // make the engine state hold exactly `want`, reusing the live prefix where possible
    json bring_to(const std::vector<int32_t> & want) {
        if (want.empty()) throw std::runtime_error("empty prompt");
        size_t common = 0;
        while (common < live.size() && common < want.size() && live[common] == want[common]) ++common;
        const auto t1 = std::chrono::steady_clock::now();
        std::string route, err;
        bool ok = true;
        qfn_logits_request lr;
        lr.full_from = (int64_t) want.size() - 1;
        lr.on_full = [&](int64_t, const float * l, int64_t V) { last_logits.assign(l, l + V); };
        if (common == live.size() && common == want.size() && !last_logits.empty()) {
            route = "none";
        } else if (common == live.size() && common > 0 && want.size() - common <= append_decode_max) {
            route = "decode";
            std::vector<int32_t> add(want.begin() + (ptrdiff_t) common, want.end());
            qfn_decode_stats st;
            ok = eng.feed(add, last_logits, st, err);
        } else if (common == live.size() && common > 0) {
            route = "layer-major-append";
            std::vector<int32_t> add(want.begin() + (ptrdiff_t) common, want.end());
            ok = eng.append(add, lr, err);
        } else {
            route = "layer-major";
            common = 0;
            ok = eng.prefill(want, lr, err);
        }
        if (!ok) { live.clear(); last_logits.clear(); throw std::runtime_error(err); }
        live = want;
        return { {"n_past", eng.n_past()}, {"reused", common}, {"new_tokens", want.size() - common}, {"route", route},
                 {"seconds", secs_since(t1)} };
    }

    void add_stats(const qfn_decode_stats & st) {
        total.tokens += st.tokens; total.expert_hits += st.expert_hits; total.expert_misses += st.expert_misses;
        total.miss_bytes += st.miss_bytes; total.seconds += st.seconds; total.miss_ms += st.miss_ms;
        total.drafted += st.drafted; total.accepted += st.accepted;
        total.policy_batches += st.policy_batches; total.policy_conflicts += st.policy_conflicts; total.policy_replays += st.policy_replays;
        total.policy_restore_bytes += st.policy_restore_bytes; total.policy_restore_slots += st.policy_restore_slots;
        total.policy_restore_ms += st.policy_restore_ms;
        total.ram_tier_hits += st.ram_tier_hits; total.policy_restore_ram_slots += st.policy_restore_ram_slots;
        total.policy_restore_ram_bytes += st.policy_restore_ram_bytes;
        total.attempts += st.attempts; total.replays += st.replays;
    }

    static json stats_json(const qfn_decode_stats & st) {
        return { {"decode_tokens", st.tokens}, {"decode_seconds", st.seconds},
                 {"tok_per_s", st.tokens / std::max(1e-9, st.seconds)}, {"expert_hits", st.expert_hits},
                 {"expert_misses", st.expert_misses}, {"miss_mib", st.miss_bytes / 1048576.0}, {"miss_ms", st.miss_ms},
                 {"mtp_drafted", st.drafted}, {"mtp_accepted", st.accepted},
                 {"mtp_policy_batches", st.policy_batches}, {"mtp_policy_conflicts", st.policy_conflicts},
                 {"mtp_policy_replays", st.policy_replays}, {"mtp_policy_restore_bytes", st.policy_restore_bytes},
                 {"mtp_policy_restore_slots", st.policy_restore_slots}, {"mtp_policy_restore_ms", st.policy_restore_ms},
                 {"ram_tier_hits", st.ram_tier_hits}, {"mtp_policy_restore_ram_slots", st.policy_restore_ram_slots},
                 {"mtp_policy_restore_ram_bytes", st.policy_restore_ram_bytes},
                 {"attempts", st.attempts}, {"replays", st.replays},
                 {"ledger_ms", { {"build", st.build_ms}, {"stage", st.stage_ms}, {"gpu", st.gpu_ms}, {"route", st.route_ms}, {"mtp", st.mtp_ms} }} };
    }

    qfn_sampling sampling_from(const json & b) {
        return qfn_parse_sampling(b.dump(), [&](const std::string & text) { return tok.tokenize(text, true); });
    }

    // ------------------------------------------------------------------------------------------- chat
    void chat(json body,const json &blocks=json::array()) {
        body=json::parse(qfn_merge_sampling_defaults(defaults.dump(),body.dump()));
        const auto t0 = std::chrono::steady_clock::now();
        if (!body.contains("messages") || !body["messages"].is_array()) throw std::runtime_error("messages must be an array");
        for(const auto &message:body["messages"])if(message.contains("content")&&message["content"].is_array())
            for(const auto &part:message["content"])if(part.is_object()&&(part.value("type","")=="image_url"||part.value("type","")=="input_image"))
                throw std::runtime_error("image content requires the guarded QFN vision adapter");
        common_chat_templates_inputs in;
        in.messages = common_chat_msgs_parse_oaicompat(body["messages"]);
        if (body.contains("tools") && !body["tools"].is_null()) in.tools = common_chat_tools_parse_oaicompat(body["tools"]);
        if (body.contains("tool_choice") && body["tool_choice"].is_string()) {
            in.tool_choice = common_chat_tool_choice_parse_oaicompat(body["tool_choice"].get<std::string>());
        } else if (body.contains("tool_choice") && body["tool_choice"].is_object()) {
            in.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        }
        in.parallel_tool_calls = body.value("parallel_tool_calls", false);
        in.add_generation_prompt = true;
        in.use_jinja = true;
        in.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) {
            for (const auto & it : body["chat_template_kwargs"].items()) in.chat_template_kwargs[it.key()] = it.value().dump();
        }
        if (body.contains("reasoning_effort") && body["reasoning_effort"].is_string() && !in.chat_template_kwargs.count("reasoning_effort")) {
            in.chat_template_kwargs["reasoning_effort"] = body["reasoning_effort"].dump();
        }
        auto et = in.chat_template_kwargs.find("enable_thinking");
        if (et != in.chat_template_kwargs.end()) {
            if (et->second == "true") in.enable_thinking = true;
            else if (et->second == "false") in.enable_thinking = false;
            else throw std::runtime_error("enable_thinking must be a boolean");
        }
        const common_chat_params cp = common_chat_templates_apply(tmpls.get(), in);
        common_chat_parser_params syn(cp);
        syn.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        const std::string open_think = "<think>\n";
        syn.thinking_forced_open = cp.thinking_forced_open ||
            (cp.prompt.size() >= open_think.size() && cp.prompt.compare(cp.prompt.size() - open_think.size(), open_think.size(), open_think) == 0);
        syn.parse_tool_calls = !in.tools.empty() && in.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
        if (!cp.parser.empty()) syn.parser.load(cp.parser);

        std::vector<std::string> stops = cp.additional_stops;
        const auto requested_stops = request_stops(body);
        stops.insert(stops.end(), requested_stops.begin(), requested_stops.end());
        stops.erase(std::remove_if(stops.begin(), stops.end(), [](const std::string & s) { return s.empty(); }), stops.end());
        const int max_tokens = request_max_tokens(body, 32768);
        const qfn_sampling s = sampling_from(body);
        std::vector<int32_t> prompt;std::vector<qfn_image_span> images;size_t cursor=0;
        if(!blocks.is_array()||blocks.size()>16)throw std::runtime_error("QFN accepts at most 16 image spans");
        for(const auto &block:blocks) {
            const auto marker=block.at("marker").get<std::string>();const auto at=cp.prompt.find(marker,cursor);
            if(marker.empty()||at==std::string::npos||cp.prompt.find(marker,at+marker.size())!=std::string::npos)
                throw std::runtime_error("image placeholder was altered by the chat template");
            auto text=tok.tokenize(cp.prompt.substr(cursor,at-cursor),true);prompt.insert(prompt.end(),text.begin(),text.end());
            prompt.push_back(248053);json span=block;span.erase("marker");span["position"]=prompt.size();
            auto image=qfn_projected_span(span,eng.hparams().n_embd,uint32_t(-1));
            prompt.insert(prompt.end(),image.count,248056);prompt.push_back(248054);images.push_back(std::move(image));
            cursor=at+marker.size();
        }
        auto tail=tok.tokenize(cp.prompt.substr(cursor),true);prompt.insert(prompt.end(),tail.begin(),tail.end());
        if(eng.set_vision(std::move(images))){live.clear();last_logits.clear();}
        const json pf = bring_to(prompt);
        emit({ {"ok", true}, {"event", "prefilled"}, {"prompt_tokens", prompt.size()}, {"prefill", pf},
               {"format", common_chat_format_name(cp.format)}, {"thinking_forced_open", syn.thinking_forced_open} });

        std::string text;
        common_chat_msg prev;
        std::vector<std::string> ids;
        std::mt19937_64 id_rng(s.seed ^ 0x9E3779B97F4A7C15ull);
        auto gen_id = [&]() {
            static const char * a = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
            std::string id = "call_";
            for (int i = 0; i < 24; ++i) id += a[id_rng() % 62];
            return id;
        };
        auto publish = [&](const common_chat_msg & msg) {
            for (const auto & d : common_chat_msg_diff::compute_diffs(prev, msg)) {
                emit({ {"ok", true}, {"delta", common_chat_msg_diff_to_json_oaicompat(d)} });
            }
            prev = msg;
        };
        auto parse_partial = [&]() {
            const std::string vis = text.substr(0, text.size() - std::max(utf8_incomplete_tail(text), stop_prefix_tail(text, stops)));
            try {
                common_chat_msg msg = common_chat_parse(vis, true, syn);
                msg.set_tool_call_ids(ids, gen_id);
                publish(msg);
            } catch (const std::exception &) {
                // an incomplete construct the parser cannot represent yet; the next token retries
            }
        };
        std::string finish = "length";
        int n_gen = 0;
        qfn_decode_stats st;
        std::string err;
        cancel = false;
        const auto td = std::chrono::steady_clock::now();
        eng.begin_sampling(s);
        const int32_t first = max_tokens > 0 ? eng.sample(last_logits.data(), s) : -1;
        bool ok = true;
        if (eng.is_eog(first)) {
            finish = "stop";
        } else if (max_tokens > 0) {
            ok = eng.decode(first, max_tokens, s, [&](int32_t t, const float * l) {
                if (!l) { finish = "stop"; return false; }          // the end-of-generation token
                live.push_back(t);
                last_logits.assign(l, l + eng.hparams().n_vocab);
                ++n_gen;
                const std::string piece = tok.piece(t);
                text += piece;
                for (const auto & stop : stops) {
                    const size_t from = text.size() > piece.size() + stop.size() ? text.size() - piece.size() - stop.size() : 0;
                    const size_t at = text.find(stop, from);
                    if (at != std::string::npos) { text.resize(at); finish = "stop"; return false; }
                }
                parse_partial();
                if (cancel) { finish = "cancelled"; return false; }
                return true;
            }, st, err);
        }
        add_stats(st);
        if (!ok) {
            if (eng.reset_required()) { live.clear(); last_logits.clear(); }
            throw std::runtime_error(err);
        }
        common_chat_msg msg;
        try {
            msg = common_chat_parse(text, false, syn);
        } catch (const std::exception &) {
            msg = common_chat_msg();
            msg.role = "assistant";
            msg.content = text;
        }
        msg.set_tool_call_ids(ids, gen_id);
        publish(msg);
        if (!msg.tool_calls.empty() && finish == "stop") finish = "tool_calls";
        const double dsec = secs_since(td);
        json timings = { {"prompt_n", pf["new_tokens"]}, {"cache_n", pf["reused"]}, {"prompt_ms", pf["seconds"].get<double>() * 1000.0},
                         {"predicted_n", n_gen}, {"predicted_ms", dsec * 1000.0},
                         {"predicted_per_second", n_gen / std::max(1e-9, dsec)} };
        emit({ {"ok", true}, {"done", true}, {"finish_reason", finish}, {"message", msg.to_json_oaicompat()},
               {"usage", { {"prompt_tokens", prompt.size()}, {"completion_tokens", n_gen}, {"total_tokens", prompt.size() + n_gen} }},
               {"timings", timings}, {"stats", stats_json(st)}, {"seconds", secs_since(t0)}, {"n_past", eng.n_past()} });
    }

    // --------------------------------------------------------------------------------------- generate
    void generate(json req) {
        req=json::parse(qfn_merge_sampling_defaults(defaults.dump(),req.dump()));
        if (last_logits.empty()) throw std::runtime_error("generate needs a prefill first");
        const qfn_sampling s = sampling_from(req);
        const int n_max = request_max_tokens(req, 256);
        const auto stops = request_stops(req);
        std::string text;
        qfn_decode_stats st;
        std::string err;
        cancel = false;
        const auto t1 = std::chrono::steady_clock::now();
        eng.begin_sampling(s);
        const int32_t first = n_max > 0 ? eng.sample(last_logits.data(), s) : -1;
        std::string reason = "length";
        bool ok = true;
        if (eng.is_eog(first)) {
            emit({ {"ok", true}, {"token", first}, {"piece", tok.piece(first)}, {"eog", true} });
            reason = "eog";
        } else if (n_max > 0) {
            ok = eng.decode(first, n_max, s, [&](int32_t t, const float * l) {
                if (!l) {
                    emit({ {"ok", true}, {"token", t}, {"piece", tok.piece(t)}, {"eog", true} });
                    reason = "eog";
                    return false;
                }
                live.push_back(t);
                last_logits.assign(l, l + eng.hparams().n_vocab);
                const auto piece = tok.piece(t);
                text += piece;
                bool stopped = false;
                for (const auto & stop : stops) if (text.find(stop) != std::string::npos) { stopped = true; break; }
                emit({ {"ok", true}, {"token", t}, {"piece", piece} });
                if (stopped) { reason = "stop"; return false; }
                if (cancel) { reason = "cancelled"; return false; }
                return true;
            }, st, err);
        }
        add_stats(st);
        if (!ok) {
            if (eng.reset_required()) { live.clear(); last_logits.clear(); }
            throw std::runtime_error(err);
        }
        json done = stats_json(st);
        done["ok"] = true; done["done"] = true; done["reason"] = reason; done["tokens"] = st.tokens;
        done["seconds"] = secs_since(t1); done["n_past"] = eng.n_past();
        emit(done);
    }
};

static std::string qfn_own_binary_identity() {
    wchar_t module[32768];const DWORD n=GetModuleFileNameW(nullptr,module,DWORD(std::size(module)));
    if(!n||n>=std::size(module))throw std::runtime_error("QFN producer executable path unavailable");
    std::ifstream file(std::filesystem::path(module),std::ios::binary);
    if(!file)throw std::runtime_error("QFN producer executable cannot be pinned");
    state_hash hash(true);char page[65536];
    while(file){file.read(page,sizeof(page));if(file.gcount())hash.update(page,size_t(file.gcount()));}
    if(!file.eof())throw std::runtime_error("QFN producer executable pin read failed");
    return hash.finish();
}

}  // namespace

int qfn_serve(qfn_options opt, const std::string & vocab_gguf, int port, const std::string & defaults) {
    std::string pair_token;
    if(!opt.pair_token_file.empty()) {
        std::ifstream token_file(std::filesystem::u8path(opt.pair_token_file),std::ios::binary);
        std::getline(token_file,pair_token);
        if(!pair_token.empty() && pair_token.back()=='\r')pair_token.pop_back();
        auto hex=[](const std::string &s){return s.size()==64 && s.find_first_not_of("0123456789abcdef")==std::string::npos;};
        if(!token_file || pair_token.size()<32 || pair_token.size()>512 ||
           !hex(opt.pair_model_sha256) || !hex(opt.pair_compatibility_sha256)) {
            std::fprintf(stderr,"qfn-engine: incomplete pairing token/model/settings identity\n");return 3;
        }
    } else if(!opt.pair_model_sha256.empty() || !opt.pair_compatibility_sha256.empty()) {
        std::fprintf(stderr,"qfn-engine: pairing identity requires --pair-token-file\n");return 3;
    }
    /* Functional installation identity, computed once before loading. Never a
     * state-integrity pass and never part of a measured handover. */
    const std::string producer_binary=pair_token.empty()?std::string{}:qfn_own_binary_identity();
    if (port > 0) {
        if (!g_io.open_port(port)) { std::fprintf(stderr, "qfn-engine: cannot listen on 127.0.0.1:%d\n", port); return 3; }
        std::fprintf(stderr, "qfn-engine: serving on 127.0.0.1:%d\n", port);
        if (!g_io.accept_client()) return 3;   // the front end connects at once and waits for ready
    }
    qfn_engine eng;
    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    if (!eng.open(opt, err)) { emit({ {"ok", false}, {"error", "open failed: " + err} }); return 3; }
    qfn_tokenizer tok;
    std::string chat_template, bos_text, eos_text;
    json model_defaults=json::object();
    {
        // the tokenizer and template come from the same pack metadata the engine already parsed
        qfn_pack pack;
        if (!pack.open(opt.model, err) || !tok.open(pack, vocab_gguf, err)) {
            emit({ {"ok", false}, {"error", "tokenizer: " + err} });
            return 3;
        }
        chat_template = pack.chat_template;
        model_defaults=json::parse(pack.generation_defaults_json);
        const int64_t bos=pack.generation_bos_explicit ? pack.generation_bos_id : int64_t(pack.hp.tok_bos);
        if(bos>=0 && uint64_t(bos)<pack.vocab.size()) bos_text=pack.vocab[size_t(bos)];
        if (pack.hp.tok_eos < pack.vocab.size()) eos_text = pack.vocab[pack.hp.tok_eos];
    }
    if (chat_template.empty()) { emit({ {"ok", false}, {"error", "the pack has no chat template"} }); return 3; }
    server srv(eng, tok);
    std::string paired_session,paired_epoch;
    uint64_t paired_sequence=1,paired_revision=0;uint32_t paired_mtp=0;
    json paired_synced,paired_pending;
    std::string paired_anchor_id;json paired_anchor_checkpoint;
    auto token_identity=[&](){state_hash h(true);h.update(srv.live.data(),srv.live.size()*4);return h.finish();};
    auto checkpoint=[&](){return json{{"position",eng.n_past()},{"mtp_rows",eng.paired_mtp_rows()},
        {"tokens_sha256",token_identity()},{"epoch",paired_epoch},{"revision",paired_revision}};};
    auto session_command=[&](const json &r){
        if(paired_session.empty()||r.at("session")!=paired_session||r.at("sequence")!=paired_sequence)
            throw std::runtime_error("paired session/ordered sequence differs");
    };
    srv.defaults = json::parse(qfn_merge_sampling_defaults(model_defaults.dump(),defaults));
    srv.tmpls = common_chat_templates_init(nullptr, chat_template, bos_text, eos_text);
    if (const char * am = std::getenv("QFN_APPEND_DECODE_MAX"); am && *am) srv.append_decode_max = (size_t) std::atoll(am);
    if(!opt.warmup_prompt.empty()) {
        auto warm=srv.bring_to(tok.tokenize(opt.warmup_prompt,true));
        std::fprintf(stderr,"QFN_WARMUP %s\n",warm.dump().c_str());
    }
    try{qfn_lock_host_resident(opt);}catch(const std::exception &e){std::fprintf(stderr,"QFN_HOST_PIN_FAILED %s\n",e.what());emit({{"ok",false},{"error",e.what()}});return 3;}
    emit({ {"ok", true}, {"event", "ready"}, {"n_ctx", opt.n_ctx}, {"n_vocab", eng.hparams().n_vocab},
           {"load_seconds", secs_since(t0)} });

    // stdin reader: "cancel" acts at once (the running op checks it per token); everything else queues
    std::mutex mu;
    std::condition_variable cv;
    struct command {std::string line;bool binary=false;};
    std::deque<command> queue;
    bool binary_pending=false;
    size_t outstanding_commands=0; // protected by mu, includes the executing command
    bool eof = false;
    std::thread reader([&]() {
        std::string line;
        for (;;) {
            if (!g_io.getline(line)) {
                if (!g_io.sock) break;
                srv.cancel = true; // the old request still owns all of its replies
                {
                    std::unique_lock<std::mutex> lock(mu);
                    cv.wait(lock,[&]{return outstanding_commands==0;});
                }
                g_io.retire_client();
                if (!g_io.accept_client()) break;
                continue;
            }
            if (line.find("\"cancel\"") != std::string::npos) {
                try { if (json::parse(line).value("op", "") == "cancel") { srv.cancel = true; continue; } } catch (...) {}
            }
            bool binary=false;
            try{binary=json::parse(line).value("op","")=="apply_decode_delta";}catch(...){}
            std::unique_lock<std::mutex> lk(mu);
            if(binary)binary_pending=true;
            queue.push_back({line,binary});++outstanding_commands;
            cv.notify_one();
            // The generation owner reads a bounded binary upload directly
            // from this socket; the line reader must not consume its bytes.
            if(binary)cv.wait(lk,[&]{return !binary_pending;});
        }
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_one();
    });
    reader.detach();

    for (;;) {
        std::string line;
        bool binary=false;
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return !queue.empty() || eof; });
            if (queue.empty()) break;
            line = std::move(queue.front().line);binary=queue.front().binary;
            queue.pop_front();
        }
        struct release_input {
            std::mutex &mu;std::condition_variable &cv;bool &pending;size_t &outstanding;bool active;
            ~release_input(){std::lock_guard<std::mutex> lock(mu);if(active)pending=false;--outstanding;cv.notify_all();}
        } input_release{mu,cv,binary_pending,outstanding_commands,binary};
        if (line.empty()) continue;
        json req;
        try { req = json::parse(line); } catch (...) { emit({ {"ok", false}, {"error", "bad json"} }); continue; }
        const std::string op = req.value("op", "");
        try {
            if(op=="pair") {
                qfn_lock_host_resident(opt);
                const auto candidate=req.at("token").get<std::string>();
                size_t different=candidate.size()^pair_token.size();
                for(size_t i=0;i<pair_token.size();i++)different|=(unsigned char)pair_token[i]^(i<candidate.size()?(unsigned char)candidate[i]:0);
                if(pair_token.empty() || different || req.at("model_sha256")!=opt.pair_model_sha256 ||
                   req.at("compatibility_sha256")!=opt.pair_compatibility_sha256)
                    throw std::runtime_error("pair token, model or settings compatibility differs");
                g_io.paired=true;
                emit({{"ok",true},{"event","paired"},{"family","qwen4exp"},{"kv_bits",16},
                    {"producer_binary_sha256",producer_binary},
                    {"vocab",eng.hparams().n_vocab},{"context",opt.n_ctx},
                    {"delta_budget_bytes",opt.host_budget},
                    {"session_id",paired_session},{"owner_epoch",paired_epoch},
                    {"model_sha256",opt.pair_model_sha256},{"compatibility_sha256",opt.pair_compatibility_sha256}});
                continue;
            }
            if(!pair_token.empty() && !g_io.paired)throw std::runtime_error("pair handshake required");
            if(!paired_session.empty()&&(op=="chat"||op=="prefill"||op=="generate"||op=="observe"||op=="export_state"||
                (op=="handover"&&!req.contains("session")))) {
                /* Standalone APIs keep working. Taking over their producer
                 * explicitly invalidates the earlier ordered paired owner. */
                std::fprintf(stderr,"QFN_PEER_OWNER_REPLACED operation=%s prior_position=%lld\n",op.c_str(),(long long)eng.n_past());
                paired_session.clear();paired_epoch.clear();paired_sequence=1;paired_revision=0;paired_mtp=0;
                paired_synced=nullptr;paired_pending=nullptr;paired_anchor_id.clear();paired_anchor_checkpoint=nullptr;
            }
            if(op=="checkpoint"||op=="acknowledge"||op=="apply_decode_delta"||op=="save_rewind"||op=="rewind") {
                session_command(req);
                if(op=="checkpoint") {
                    emit({{"ok",true},{"event","checkpoint"},{"checkpoint",checkpoint()},{"next_sequence",++paired_sequence}});continue;
                }
                if(op=="acknowledge") {
                    if(paired_pending.is_null()||req.at("checkpoint")!=paired_pending||checkpoint()!=paired_pending)
                        throw std::runtime_error("paired checkpoint acknowledgement differs from actual frontier");
                    paired_synced=paired_pending;paired_pending=nullptr;
                    emit({{"ok",true},{"event","acknowledged"},{"checkpoint",paired_synced},{"next_sequence",++paired_sequence}});continue;
                }
                if(!paired_pending.is_null()||paired_synced.is_null()||req.at("base")!=paired_synced||checkpoint()!=paired_synced)
                    throw std::runtime_error("reverse delta base has not been acknowledged");
                if(op=="save_rewind") {
                    const auto id=req.at("anchor_id").get<std::string>();
                    if(id.size()!=32||id.find_first_not_of("0123456789abcdef")!=std::string::npos)
                        throw std::runtime_error("invalid paired rewind anchor identity");
                    uint64_t bytes=0;std::string error;
                    if(!eng.save_paired_anchor(srv.last_logits,opt.host_budget,bytes,error))throw std::runtime_error(error);
                    qfn_lock_host_resident(opt);
                    paired_anchor_id=id;paired_anchor_checkpoint=checkpoint();
                    emit({{"ok",true},{"event","anchor_saved"},{"anchor_id",id},{"checkpoint",paired_anchor_checkpoint},
                        {"bytes",bytes},{"next_sequence",++paired_sequence}});continue;
                }
                if(op=="rewind") {
                    if(paired_anchor_id.empty()||req.at("anchor_id")!=paired_anchor_id||req.at("anchor")!=paired_anchor_checkpoint)
                        throw std::runtime_error("paired rewind anchor or saved checkpoint differs");
                    std::string error;
                    if(!eng.rewind_paired_anchor(srv.live,srv.last_logits,error,req.value("test_owner_readback",false))) {
                        if(eng.reset_required()){srv.live.clear();srv.last_logits.clear();}
                        throw std::runtime_error(error);
                    }
                    paired_mtp=eng.paired_mtp_rows();++paired_revision;paired_pending=checkpoint();
                    emit({{"ok",true},{"event","rewound"},{"anchor_id",paired_anchor_id},{"checkpoint",paired_pending},
                        {"next_sequence",++paired_sequence}});continue;
                }
                const auto base=req.at("base").at("position").get<uint32_t>();
                const auto bm=req.at("base").at("mtp_rows").get<uint32_t>();
                const auto target=req.at("target").get<uint32_t>(),tm=req.at("target_mtp_rows").get<uint32_t>();
                const auto bytes=req.at("bytes").get<uint64_t>(),budget=req.at("max_bytes").get<uint64_t>();
                const auto stall=req.at("stall_seconds").get<uint32_t>();
                if(!stall||stall>3600||!bytes||bytes>budget||budget>opt.host_budget||target<=base||target>opt.n_ctx)
                    throw std::runtime_error("reverse delta budget/frontier/stall policy refused");
                transport callback;transport *state=&g_io;
                if(req.contains("state_callback")) {
                    std::fprintf(stderr,"QFN_STATE_PHASE phase=reverse_callback_connect rows=%u\n",target);
                    callback.connect_state(req.at("state_callback"));state=&callback;
                    state->put(json{{"ok",true},{"event","state_paired"},{"token",pair_token},
                        {"nonce",req.at("state_callback").at("nonce")},{"model_sha256",opt.pair_model_sha256},
                        {"compatibility_sha256",opt.pair_compatibility_sha256}}.dump());
                }
                state->put(json{{"ok",true},{"event","delta_ready"},{"bytes",bytes}}.dump());
                const bool verify=req.value("verify_state",false);state_hash digest(verify);uint64_t consumed=0;std::string error;
                const bool ok=eng.apply_decode_delta(base,target,bm,tm,bytes,budget,[&](void *p,size_t n){
                    if(n>bytes-consumed)throw std::runtime_error("reverse stream exceeded claimed extent");
                    state->read_binary(p,n,stall);digest.update(p,n);consumed+=n;
                },srv.live,srv.last_logits,error);
                if(!ok||consumed!=bytes) {
                    paired_synced=nullptr;paired_pending=nullptr;
                    if(eng.reset_required()){srv.live.clear();srv.last_logits.clear();}
                    if(state==&g_io&&g_io.client!=INVALID_SOCKET){closesocket(g_io.client);g_io.client=INVALID_SOCKET;g_io.inbuf.clear();}
                    throw std::runtime_error(error.empty()?"reverse delta extent differed":error);
                }
                paired_mtp=tm;++paired_revision;paired_synced=checkpoint();
                if(req.at("tokens_sha256")!=paired_synced.at("tokens_sha256")) {
                    paired_synced=nullptr;throw std::runtime_error("reverse delta target token identity differs; session rebuild required");
                }
                state->put(json{{"ok",true},{"event","delta_applied"},{"bytes",bytes},{"checkpoint",paired_synced},
                    {"inline_sha256",digest.finish()},{"next_sequence",++paired_sequence}}.dump());continue;
            }
            if(op=="handover") {
                if(pair_token.empty() || !g_io.paired)throw std::runtime_error("handover requires an authenticated paired socket");
                const bool verify=req.value("verify_state",false);
                const auto encoding=req.value("wire_encoding",std::string("f16-index-v1"));
                if(encoding!="f16-index-v1"&&encoding!="f32-index-v1")throw std::runtime_error("unsupported QFN wire encoding");
                const bool narrow=encoding=="f16-index-v1";
                const auto tokens=req.at("tokens").get<std::vector<int32_t>>();
                const bool persistent=req.contains("session"),delta=persistent&&req.contains("base");
                uint32_t delta_base=0,delta_mtp=0;
                if(persistent) {
                    const auto id=req.at("session").get<std::string>();
                    if(id.size()!=32||id.find_first_not_of("0123456789abcdef")!=std::string::npos)
                        throw std::runtime_error("invalid paired conversation identity");
                    if(id!=paired_session) {
                        if(delta||req.at("sequence")!=1)throw std::runtime_error("new paired session requires full initial command sequence 1");
                        unsigned char nonce[16];if(BCryptGenRandom(nullptr,nonce,sizeof(nonce),BCRYPT_USE_SYSTEM_PREFERRED_RNG)<0)
                            throw std::runtime_error("paired owner epoch creation failed");
                        static const char hex[]="0123456789abcdef";paired_epoch.clear();
                        for(unsigned char v:nonce){paired_epoch+=hex[v>>4];paired_epoch+=hex[v&15];}
                        paired_session=id;paired_sequence=1;paired_revision=0;paired_mtp=0;paired_synced=nullptr;paired_pending=nullptr;
                        paired_anchor_id.clear();paired_anchor_checkpoint=nullptr;
                        // A new full owner can follow an interrupted binary
                        // import. Rebuild it from weights even when token IDs
                        // extend the old history; only an acknowledged delta
                        // may reuse the previous paired CUDA owner state.
                        srv.live.clear();srv.last_logits.clear();
                    }
                    session_command(req);
                    if(!paired_pending.is_null())throw std::runtime_error("previous checkpoint awaits acknowledgement");
                    if(delta) {
                        if(paired_synced.is_null()||req.at("base")!=paired_synced||checkpoint()!=paired_synced)
                            throw std::runtime_error("forward delta actual base differs");
                        delta_base=paired_synced.at("position").get<uint32_t>();delta_mtp=paired_synced.at("mtp_rows").get<uint32_t>();
                        if(tokens.size()<=delta_base||!std::equal(srv.live.begin(),srv.live.end(),tokens.begin()))
                            throw std::runtime_error("forward delta must extend the acknowledged token frontier");
                    }
                }
                std::vector<qfn_image_span> images;
                if(req.contains("vision")) {
                    const auto &vision=req.at("vision");
                    if(!vision.is_array()||vision.size()>16)throw std::runtime_error("QFN accepts at most 16 image spans");
                    for(const auto &row:vision)images.push_back(qfn_projected_span(row,eng.hparams().n_embd,opt.n_ctx));
                }
                if(eng.set_vision(std::move(images),delta?delta_base:0)&&!delta){
                    srv.live.clear();srv.last_logits.clear();
                }
                const auto budget=req.at("max_bytes").get<int64_t>();
                if(budget<=0)throw std::runtime_error("positive state byte budget required");
                // These paired branches continue before the common command
                // epilogue. Refresh residency for newly allocated request and
                // long-prefill graph/host pages without relaxing the guard.
                qfn_lock_host_resident(opt);
                struct progress_scope {
                    qfn_engine &engine;
                    ~progress_scope(){engine.set_prefill_progress({});}
                } progress_reset{eng};
                eng.set_prefill_progress([&](uint32_t layer,uint32_t layers,uint64_t rows){
                    emit({{"ok",true},{"event","prefill_progress"},{"layer",layer},{"layers",layers},{"work_rows",rows}});
                });
                auto pf=srv.bring_to(tokens);
                eng.set_prefill_progress({});
                qfn_lock_host_resident(opt);
                pf["ok"]=true;pf["event"]="prefilled";emit(pf);
                transport callback;transport *state=&g_io;
                if(req.contains("state_callback")) {
                    std::fprintf(stderr,"QFN_STATE_PHASE phase=callback_connect rows=%zu\n",tokens.size());
                    callback.connect_state(req.at("state_callback"));state=&callback;
                    callback.put(json({{"ok",true},{"event","state_paired"},{"token",pair_token},
                        {"nonce",req.at("state_callback").at("nonce")},{"model_sha256",opt.pair_model_sha256},
                        {"compatibility_sha256",opt.pair_compatibility_sha256}}).dump());
                }
                const auto state_emit=[&](const json &row){std::fprintf(stderr,"QFN_STATE_PHASE phase=%s rows=%zu\n",row.value("event",std::string("unknown")).c_str(),tokens.size());state->put(row.dump());};
                state_hash digest(verify);uint64_t bytes=0;std::string error;
                double gpu_read_s=0,socket_send_s=0,inline_hash_s=0;
                // The codec emits scalar header/token fields too. Coalesce them
                // into bounded pages rather than issuing hundreds of thousands
                // of four-byte socket sends for a long context.
                std::vector<unsigned char> page(1<<20);size_t buffered=0;
                auto flush=[&](){if(buffered){
                    auto clock=std::chrono::steady_clock::now();digest.update(page.data(),buffered);
                    if(verify)inline_hash_s+=secs_since(clock);
                    clock=std::chrono::steady_clock::now();state->binary(page.data(),buffered);socket_send_s+=secs_since(clock);buffered=0;
                }};
                const auto start=std::chrono::steady_clock::now();
                auto write=[&](const void *p,size_t n){
                        const auto *source=static_cast<const unsigned char *>(p);
                        while(n){size_t count=std::min(n,page.size()-buffered);std::memcpy(page.data()+buffered,source,count);
                            buffered+=count;source+=count;n-=count;if(buffered==page.size())flush();}
                    };
                auto ready=[&](uint64_t wire,uint64_t canonical,uint32_t mtp_rows){
                        paired_mtp=mtp_rows;
                        qfn_lock_host_resident(opt);
                        state_emit({{"ok",true},{"event","state_stream"},{"bytes",wire},{"canonical_bytes",canonical},
                            {"rows",eng.n_past()},{"mtp_rows",mtp_rows},{"context",opt.n_ctx},
                            {"wire_encoding",delta?"qfn-delta-f32-v1":encoding},{"verify_state",verify},{"export_prepare_seconds",secs_since(start)},
                            {"model_sha256",opt.pair_model_sha256},{"compatibility_sha256",opt.pair_compatibility_sha256}});
                    };
                bool exported=false;
                if(delta) {
                    bool header=true;
                    exported=eng.export_delta(delta_base,delta_mtp,srv.last_logits,uint64_t(budget),bytes,
                        [&](const void *p,size_t n){
                            if(header){uint32_t h[10];if(n!=sizeof(h))throw std::runtime_error("delta header extent differs");
                                std::memcpy(h,p,n);ready((uint64_t(h[9])<<32)|h[8],(uint64_t(h[9])<<32)|h[8],h[6]);header=false;}
                            write(p,n);
                        },paired_mtp,error);
                } else exported=eng.export_state("",srv.last_logits,uint64_t(budget),bytes,error,narrow,write,ready,&gpu_read_s);
                if(!exported)throw std::runtime_error(error);
                flush();
                if(persistent){++paired_revision;paired_pending=checkpoint();++paired_sequence;}
                state_emit({{"ok",true},{"event","state_streamed"},{"bytes",bytes},{"checkpoint",paired_pending},
                    {"next_sequence",paired_sequence},
                    {"inline_sha256",digest.finish()},{"export_and_send_seconds",secs_since(start)},
                    {"producer_gpu_read_seconds",gpu_read_s},{"producer_socket_send_seconds",socket_send_s},
                    {"producer_inline_hash_seconds",inline_hash_s}});
                qfn_lock_host_resident(opt);
                continue;
            }
            if (op == "quit") { emit({ {"ok", true}, {"event", "bye"} }); break; }
            if (op == "chat") {
                srv.chat(req.at("body"),req.value("vision",json::array()));
            } else if (op == "tokenize") {
                emit({ {"ok", true}, {"tokens", tok.tokenize(req.at("text").get<std::string>(), req.value("special", true))} });
            } else if (op == "detokenize") {
                std::string s;
                for (int32_t t : req.at("tokens").get<std::vector<int32_t>>()) s += tok.piece(t);
                emit({ {"ok", true}, {"text", s} });
            } else if (op == "prefill") {
                json r = srv.bring_to(req.at("tokens").get<std::vector<int32_t>>());
                r["ok"] = true;
                emit(r);
            } else if (op == "generate") {
                srv.generate(req);
            } else if (op == "observe") {
                // An instrumented unbiased teacher-forced prefix, separate from
                // the ordinary sampler API. All full-vocabulary rows are saved.
                if (srv.live.size() != size_t(eng.n_past()) || srv.last_logits.empty())
                    throw std::runtime_error("QFN observation checkpoint is not synchronized");
                const int rows = req.at("rows").get<int>();
                if (rows < 1 || rows > 4096) throw std::runtime_error("observe rows must be 1..4096");
                std::vector<int32_t> forced;
                if (req.contains("tokens")) forced = req.at("tokens").get<std::vector<int32_t>>();
                if (!forced.empty() && forced.size() != size_t(rows)) throw std::runtime_error("forced observation extent differs");
                FILE *file = nullptr;
                if (fopen_s(&file, req.at("path").get<std::string>().c_str(), "wbx") || !file)
                    throw std::runtime_error("observation output must be new");
                std::vector<int32_t> chosen;
                try {
                    for (int row = 0; row < rows; ++row) {
                        const auto &lg = srv.last_logits;
                        for (float x : lg) if (!std::isfinite(x)) throw std::runtime_error("nonfinite observation logits");
                        if (fwrite(lg.data(), sizeof(float), lg.size(), file) != lg.size()) throw std::runtime_error("observation write failed");
                        const int32_t next = forced.empty() ? int32_t(std::max_element(lg.begin(), lg.end()) - lg.begin()) : forced[row];
                        std::string error; qfn_decode_stats st;
                        if (!eng.feed({next}, srv.last_logits, st, error, 0.0f)) {
                            srv.live.clear();srv.last_logits.clear();throw std::runtime_error(error);
                        }
                        srv.live.push_back(next);chosen.push_back(next);srv.add_stats(st);
                    }
                } catch (...) { fclose(file);throw; }
                if (fclose(file)) throw std::runtime_error("observation close failed");
                emit({{"ok",true},{"event","observed"},{"tokens",chosen},{"rows",rows},{"vocab",eng.hparams().n_vocab},{"bias",0.0}});
            } else if (op == "export_state") {
                if (srv.live.size() != size_t(eng.n_past()) || srv.last_logits.empty())
                    throw std::runtime_error("QFN state/logit checkpoint is not synchronized");
                uint64_t bytes = 0; std::string error;
                if (!req.at("max_bytes").is_number_integer()) throw std::runtime_error("max_bytes must be an integer");
                const auto budget = req.at("max_bytes").get<int64_t>();
                if (budget <= 0 || !eng.export_state(req.at("path").get<std::string>(), srv.last_logits,
                                                   uint64_t(budget), bytes, error))
                    throw std::runtime_error(error.empty() ? "invalid snapshot budget" : error);
                emit({{"ok", true}, {"event", "state_exported"}, {"bytes", bytes},
                      {"mtp_alignment","hidden-q-token-q-plus-1"},
                      {"rows", eng.n_past()}, {"kv_bits", 16}, {"state_format", "qfn-native-c-v2"}});
            } else if (op == "stats") {
                json r = server::stats_json(srv.total);
                r["ok"] = true; r["n_past"] = eng.n_past();
                emit(r);
            } else {
                emit({ {"ok", false}, {"error", "unknown op " + op} });
            }
        } catch (const std::exception & e) {
            std::fprintf(stderr,"QFN_COMMAND_FAILURE op=%s error=%s\n",op.c_str(),e.what());
            emit({ {"ok", false}, {"error", e.what()} });
        }
        try{qfn_lock_host_resident(opt);}catch(const std::exception &e){std::fprintf(stderr,"QFN_HOST_PIN_FAILED %s\n",e.what());emit({{"ok",false},{"error",e.what()}});return 3;}
    }
    return 0;
}
