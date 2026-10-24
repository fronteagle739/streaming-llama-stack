#include "llama-dsv41-carry.h"
#include "llama-dsv41-locked-ram.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <malloc.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {
constexpr uint64_t alignment = 4096;
constexpr uint64_t record_bytes = 81920 + 16 + 512 * 4 + 2048 * 4;
constexpr uint64_t selected_offset = 81920 + 16;
constexpr uint64_t candidate_offset = selected_offset + 512 * 4;
constexpr uint64_t page_bytes = ((32 * record_bytes + alignment - 1) / alignment) * alignment;
static_assert(record_bytes==llama_dsv41_carry_store::ram_record_bytes&&
    page_bytes==llama_dsv41_carry_store::ram_page_bytes&&alignment==llama_dsv41_carry_store::ram_header_bytes,
    "Direct RAM geometry must remain the original physical carry layout");
std::atomic<uint64_t> next_ram_store_identity{1};
static_assert(sizeof(float) == 4 && sizeof(int32_t) == 4, "Native carry requires F32/I32");

void require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}
uint64_t add(uint64_t a, uint64_t b) {
    require(a <= std::numeric_limits<uint64_t>::max() - b, "Carry capacity overflow");
    return a + b;
}
bool selected_source(uint32_t layer) {
    return layer == 2 || layer == 8 || layer == 14 || layer == 20 ||
           layer == 24 || layer == 28 || layer == 32 || layer == 36;
}
bool finite_bits(const float & value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000u) != 0x7f800000u;
}
void put32(uint8_t * out, uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) { out[i] = uint8_t(value >> (i * 8)); }
}
void put64(uint8_t * out, uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) { out[i] = uint8_t(value >> (i * 8)); }
}
}

struct llama_dsv41_carry_store::implementation {
    enum phase { fresh, seeding, between_layers, running_layer, awaiting_head, sealed, poisoned };
    mutable std::mutex mutex;
    phase state = fresh;
    config options;
    budget capacity;
    std::filesystem::path path;
    std::vector<std::filesystem::path> parent_paths;
    std::shared_ptr<uint8_t[]> locked_page;
    uint8_t * page = nullptr;
    std::shared_ptr<uint8_t[]> resident;
    std::unique_lock<std::mutex> shared_bank_claim;
    uint64_t resident_bytes = 0;
    uint64_t buffer_offset = 0;
    bool direct_rows = false;
    uint64_t length = 0;
    uint32_t cursor = 0, layers = 0;
    uint32_t export_cursor = 0;
    uint32_t selected_width = 0, candidate_width = 0;
    uint32_t next_selected = 0, next_candidates = 0;
    uint32_t lease_start = 0, lease_count = 0;
    bool lease = false;
    bool ram_direct_lease = false;
    uint64_t ram_generation=0,ram_identity=next_ram_store_identity.fetch_add(1);
#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE;
    std::vector<HANDLE> parents;
    DWORD volume = 0, index_high = 0, index_low = 0;
#else
    int file = -1;
    std::vector<int> parents;
    dev_t device = 0;
    ino_t inode = 0;
    timespec modified {};
#endif

    ~implementation() {
#ifdef _WIN32
        if (file != INVALID_HANDLE_VALUE) { CloseHandle(file); }
        for (auto h : parents) { CloseHandle(h); }
#else
        if (file >= 0) { close(file); }
        for (auto fd : parents) { close(fd); }
#endif
        // Intentionally no unlink/remove/truncate, including failed fixtures.
    }

    bool open() const {
        if (options.resident_ram) { return bool(resident); }
#ifdef _WIN32
        return file != INVALID_HANDLE_VALUE;
#else
        return file >= 0;
#endif
    }
    bool hybrid() const {return options.ram_prefix_tokens!=0;}
    uint64_t physical_length() const {
        if(!hybrid())return length;
        return std::min(length,alignment)+(length>alignment+resident_bytes?length-alignment-resident_bytes:0);
    }
    void actual_budget() const {
        if (options.resident_ram) {
            require(capacity.resident_bytes == capacity.file_bytes &&
                    options.admitted_host_bytes >= capacity.bounded_host_bytes &&
                    length <= capacity.resident_bytes,
                    "Resident carry exceeds its explicit one-bank host admission");
            return;
        }
        const auto free = std::filesystem::space(path.parent_path()).available;
        require(length <= capacity.file_bytes, "Carry file exceeds admitted extent");
        const uint64_t need = add(add(capacity.disk_bytes - physical_length(), options.outstanding_bytes), options.disk_reserve_bytes);
        require(free >= need, "Actual carry volume lacks pending bytes plus explicit disk reserve");
        require(!hybrid()||(resident_bytes<=capacity.resident_bytes&&options.admitted_host_bytes>=capacity.bounded_host_bytes),
                "Hybrid carry exceeds explicit prefix/tile host admission");
    }
    void check_parents() const {
        require(!parents.empty() && parents.size() == parent_paths.size(), "Carry ancestor lease is incomplete");
        for (size_t i = 0; i < parents.size(); ++i) {
#ifdef _WIN32
            BY_HANDLE_FILE_INFORMATION held {}, visible {};
            require(GetFileInformationByHandle(parents[i], &held) != 0,
                    "Cannot inspect held carry ancestor");
            HANDLE named = CreateFileW(parent_paths[i].c_str(), FILE_READ_ATTRIBUTES,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
            require(named != INVALID_HANDLE_VALUE, "Cannot inspect named carry ancestor");
            const bool got = GetFileInformationByHandle(named, &visible) != 0;
            CloseHandle(named);
            require(got && (held.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) &&
                    (visible.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) &&
                    !((held.dwFileAttributes | visible.dwFileAttributes) & FILE_ATTRIBUTE_REPARSE_POINT) &&
                    held.dwVolumeSerialNumber == visible.dwVolumeSerialNumber &&
                    held.nFileIndexHigh == visible.nFileIndexHigh && held.nFileIndexLow == visible.nFileIndexLow,
                    "Named carry ancestor differs from its held physical directory");
#else
            struct stat held {}, visible {};
            require(fstat(parents[i], &held) == 0 && lstat(parent_paths[i].c_str(), &visible) == 0 &&
                    S_ISDIR(held.st_mode) && S_ISDIR(visible.st_mode) &&
                    held.st_dev == visible.st_dev && held.st_ino == visible.st_ino,
                    "Named carry ancestor differs from its opened physical directory");
#endif
        }
    }
    void check_identity() const {
        require(open(), "Carry descriptor is closed");
        if (options.resident_ram) {
            require(page && resident_bytes == capacity.resident_bytes &&
                    resident_bytes == capacity.file_bytes && length <= resident_bytes,
                    "Resident carry bank extent or lifetime changed");
            return;
        }
        check_parents();
#ifdef _WIN32
        BY_HANDLE_FILE_INFORMATION info {};
        require(GetFileInformationByHandle(file, &info) != 0, "Cannot inspect owned carry descriptor");
        const auto attrs = GetFileAttributesW(path.c_str());
        require(attrs != INVALID_FILE_ATTRIBUTES && !(attrs & (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DIRECTORY)),
                "Carry path changed to nonregular/reparse item");
        require(info.dwVolumeSerialNumber == volume && info.nFileIndexHigh == index_high &&
                info.nFileIndexLow == index_low && info.nNumberOfLinks == 1 &&
                !(info.dwFileAttributes & (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_SPARSE_FILE | FILE_ATTRIBUTE_COMPRESSED)),
                "Owned carry identity or physical allocation changed");
        require((uint64_t(info.nFileSizeHigh) << 32 | info.nFileSizeLow) == physical_length(),
                "Owned carry size changed");
#else
        struct stat fd {}, named {};
        require(fstat(file, &fd) == 0 && lstat(path.c_str(), &named) == 0,
                "Cannot inspect owned carry descriptor/path");
        require(S_ISREG(fd.st_mode) && S_ISREG(named.st_mode) && fd.st_nlink == 1 && named.st_nlink == 1 &&
                fd.st_dev == device && fd.st_ino == inode && named.st_dev == device && named.st_ino == inode &&
                uint64_t(fd.st_size) == physical_length() && fd.st_size == named.st_size,
                "Owned carry descriptor/path identity changed");
#if defined(__APPLE__)
        const auto now = fd.st_mtimespec;
#else
        const auto now = fd.st_mtim;
#endif
        require(now.tv_sec == modified.tv_sec && now.tv_nsec == modified.tv_nsec,
                "Carry bytes were modified outside the owner");
#endif
    }
    void remember_write(uint64_t new_length) {
        length = new_length;
#ifndef _WIN32
        if (!options.resident_ram) {
            struct stat s {};
            require(fstat(file, &s) == 0, "Cannot inspect written carry descriptor");
#if defined(__APPLE__)
            modified = s.st_mtimespec;
#else
            modified = s.st_mtim;
#endif
        }
#endif
        check_identity();
    }
    void io(bool write, uint64_t offset, size_t bytes) {
        check_identity();
        require(offset % alignment == 0 && bytes % alignment == 0 && bytes <= capacity.page_buffer_bytes,
                "Unaligned or unbounded carry I/O");
        require(offset <= capacity.file_bytes && bytes <= capacity.file_bytes - offset,
                "Carry I/O exceeds admitted capacity");
        require(write ? offset <= length : offset + bytes <= length,
                "Carry I/O would create a hole or read unwritten bytes");
        if (write) { actual_budget(); }
        if (options.resident_ram) {
            if (write) {
                std::memcpy(resident.get() + offset, page, bytes);
                remember_write(std::max(length, offset + bytes));
            } else {
                std::memcpy(page, resident.get() + offset, bytes);
            }
            return;
        }
        // The RAM prefix has no disk mirror. A crossing tile splits once;
        // overflow remains one contiguous aligned unbuffered operation.
        uint8_t *buffer=page;
        const auto original_end=offset+bytes;
        if(hybrid()&&offset>=alignment&&offset<alignment+resident_bytes){
            const auto ram_bytes=size_t(std::min<uint64_t>(bytes,alignment+resident_bytes-offset));
            if(write)std::memcpy(resident.get()+offset-alignment,buffer,ram_bytes);
            else std::memcpy(buffer,resident.get()+offset-alignment,ram_bytes);
            offset+=ram_bytes;buffer+=ram_bytes;bytes-=ram_bytes;
        }
        if(!bytes){if(write)remember_write(std::max(length,original_end));return;}
        const uint64_t physical_offset=hybrid()&&offset>=alignment+resident_bytes?offset-resident_bytes:offset;
#ifdef _WIN32
        LARGE_INTEGER where; where.QuadPart = static_cast<LONGLONG>(physical_offset);
        require(SetFilePointerEx(file, where, nullptr, FILE_BEGIN) != 0, "Carry seek failed");
        DWORD done = 0;
        const bool ok = write ? WriteFile(file, buffer, DWORD(bytes), &done, nullptr) != 0 :
                                ReadFile(file, buffer, DWORD(bytes), &done, nullptr) != 0;
        require(ok && done == bytes, "Carry aligned I/O failed or was short");
#else
        size_t done = 0;
        while (done != bytes) {
            const auto n = write ? pwrite(file, buffer + done, bytes - done, off_t(physical_offset + done)) :
                                   pread(file, buffer + done, bytes - done, off_t(physical_offset + done));
            if (n < 0 && errno == EINTR) { continue; }
            require(n > 0, "Carry positioned I/O failed or was short");
            done += size_t(n);
        }
#endif
        if (write) { remember_write(std::max(length, original_end)); }
    }
    void flush() {
        if (options.resident_ram) { check_identity(); return; }
#ifdef _WIN32
        require(FlushFileBuffers(file) != 0, "Carry durability flush failed");
#else
        require(fsync(file) == 0, "Carry durability flush failed");
#endif
    }
    void header() {
        std::memset(page, 0, alignment);
        std::memcpy(page, "DSV41-CARRY-1", 13);
        put32(page + 16, uint32_t(state)); put32(page + 20, options.tokens);
        put32(page + 24, layers); put32(page + 28, cursor);
        put32(page + 32, selected_width); put32(page + 36, candidate_width);
        put64(page + 40, record_bytes); put64(page + 48, page_bytes);
        put64(page + 56, capacity.file_bytes);
        io(true, 0, alignment);
    }
    void poison() noexcept {
        state = poisoned; lease = false;
        // The in-memory certificate is revoked immediately. Best effort only:
        // an identity/budget error must never cause writes through an unsafe path.
        if (open() && page && length >= alignment) {
            try { header(); flush(); } catch (...) {}
        }
    }
    template<class F> bool guarded(F fn, std::string & error) {
        std::lock_guard<std::mutex> guard(mutex);
        try {
            require(state != poisoned, "Carry sweep is permanently invalid");
            fn(); error.clear(); return true;
        } catch (const std::exception & e) {
            poison(); error = e.what(); return false;
        } catch (...) {
            poison(); error = "Unknown carry failure"; return false;
        }
    }
    void range(uint32_t start, uint32_t count,bool row_aware_lease=false) const {
        require(count > 0 && count <= options.max_tile_tokens && start < options.tokens && count <= options.tokens - start,
                "Carry tile is outside its explicit token bound");
        require((row_aware_lease&&options.resident_ram&&options.direct_ram_tiles)||
                ((options.resident_ram||hybrid()) && count > page_tokens && start%page_tokens==0) ||
                start / page_tokens == (start + count - 1) / page_tokens, "Carry tile crosses a physical page");
    }
    uint64_t page_offset(uint32_t start) const { return alignment + uint64_t(start / page_tokens) * page_bytes; }
    uint8_t * row(uint32_t token) {
        return (direct_rows ? resident.get() + page_offset(token) : page+page_offset(token)-buffer_offset) + uint64_t(token % page_tokens) * record_bytes;
    }

    void validate(const tile & value, uint32_t sw, uint32_t cw) const {
        range(value.start, value.count);
        require(value.selected_width == sw && value.candidate_width == cw &&
                value.residual.size() == size_t(hidden_floats) * value.count &&
                value.previous_pre.size() == size_t(pre_floats) * value.count &&
                value.selected.size() == size_t(sw) * value.count &&
                value.candidates.size() == size_t(cw) * value.count,
                "Carry tile shape or compact widths differ from the layer contract");
        require(std::all_of(value.residual.begin(), value.residual.end(), [](const float & x) { return finite_bits(x); }) &&
                std::all_of(value.previous_pre.begin(), value.previous_pre.end(), [](const float & x) { return finite_bits(x); }),
                "Carry contains non-finite F32 values");
        require(std::all_of(value.selected.begin(), value.selected.end(), [](int32_t x) { return x >= -1; }) &&
                std::all_of(value.candidates.begin(), value.candidates.end(), [](int32_t x) { return x >= 0; }),
                "Invalid compact carry IDs (candidate padding must not enter scatter)");
    }
    void unpack(uint32_t start, uint32_t count, tile & output) {
        tile result;
        result.start = start; result.count = count;
        result.selected_width = selected_width; result.candidate_width = candidate_width;
        result.residual.resize(size_t(hidden_floats) * count);
        result.previous_pre.resize(size_t(pre_floats) * count);
        result.selected.resize(size_t(selected_width) * count);
        result.candidates.resize(size_t(candidate_width) * count);
        for (uint32_t i = 0; i < count; ++i) {
            const auto r = row(start + i);
            std::memcpy(result.residual.data() + size_t(i) * hidden_floats, r, 81920);
            std::memcpy(result.previous_pre.data() + size_t(i) * pre_floats, r + 81920, 16);
            if (selected_width) { std::memcpy(result.selected.data() + size_t(i) * selected_width, r + selected_offset, selected_width * 4); }
            if (candidate_width) { std::memcpy(result.candidates.data() + size_t(i) * candidate_width, r + candidate_offset, candidate_width * 4); }
        }
        validate(result, selected_width, candidate_width);
        output = std::move(result);
    }
    void pack(const tile & value, bool initializing) {
        for (uint32_t i = 0; i < value.count; ++i) {
            auto r = row(value.start + i);
            if (!initializing && !selected_source(layers) && selected_width) {
                require(std::memcmp(r + selected_offset, value.selected.data() + size_t(i) * selected_width, selected_width * 4) == 0,
                        "Reader layer changed carried selected IDs");
            }
            if (!initializing && layers != 20 && candidate_width) {
                require(std::memcmp(r + candidate_offset, value.candidates.data() + size_t(i) * candidate_width, candidate_width * 4) == 0,
                        "Reader layer changed carried candidate IDs");
            }
            std::memcpy(r, value.residual.data() + size_t(i) * hidden_floats, 81920);
            std::memcpy(r + 81920, value.previous_pre.data() + size_t(i) * pre_floats, 16);
            std::memset(r + selected_offset, 0xff, (max_selected + max_candidates) * 4);
            if (value.selected_width) { std::memcpy(r + selected_offset, value.selected.data() + size_t(i) * value.selected_width, value.selected_width * 4); }
            if (value.candidate_width) { std::memcpy(r + candidate_offset, value.candidates.data() + size_t(i) * value.candidate_width, value.candidate_width * 4); }
        }
    }
};

llama_dsv41_carry_store::llama_dsv41_carry_store() : p(new implementation) {}
llama_dsv41_carry_store::~llama_dsv41_carry_store() = default;

bool llama_dsv41_carry_store::plan(uint32_t tokens, uint64_t outstanding, budget & result, std::string & error) {
    try {
        require(tokens > 0 && tokens <= max_tokens, "Carry context must be within 1..1048576 tokens");
        budget next;
        next.file_bytes = alignment + ((uint64_t(tokens) + page_tokens - 1) / page_tokens) * page_bytes;
        next.disk_bytes = next.file_bytes;
        next.page_buffer_bytes = page_bytes;
        next.bounded_host_bytes = page_bytes + 2 * page_tokens * record_bytes;
        next.required_free_bytes = add(add(next.file_bytes, outstanding), reserve_bytes);
        result = next; error.clear(); return true;
    } catch (const std::exception & e) { result = {}; error = e.what(); return false; }
}

bool llama_dsv41_carry_store::plan_resident_ram(uint32_t tokens, budget & result, std::string & error, uint32_t max_tile_tokens) {
    budget next;
    if (!plan(tokens, 0, next, error)) { result = {}; return false; }
    try {
        next.resident_bytes = next.file_bytes;
        next.disk_bytes = 0;
        next.bounded_host_bytes = add(next.bounded_host_bytes, next.resident_bytes);
        require(max_tile_tokens==32||max_tile_tokens==128||max_tile_tokens==512||max_tile_tokens==2048||max_tile_tokens==4096||max_tile_tokens==8192||max_tile_tokens==16384,
                "Unsupported explicit RAM compute tile");
        next.bounded_host_bytes = add(next.bounded_host_bytes, 2ull*(max_tile_tokens-page_tokens)*record_bytes);
        next.required_free_bytes = 0; // No disk space, file, mapping or flush in RAM mode.
        result = next; error.clear(); return true;
    } catch (const std::exception & e) { result = {}; error = e.what(); return false; }
}

bool llama_dsv41_carry_store::plan_resident_ram_direct(uint32_t tokens,budget &result,std::string &error,uint32_t tile){
    // Identical bank/page geometry. This mode enforces page-sized vector reads
    // below; wide tiles are borrowed slices only and allocate no tile vectors.
    budget next;if(!plan_resident_ram(tokens,next,error,tile)){result={};return false;}
    next.bounded_host_bytes-=2ull*(tile-page_tokens)*record_bytes;
    result=next;return true;
}

bool llama_dsv41_carry_store::plan_hybrid(uint32_t tokens,uint32_t prefix,uint32_t tile,
        uint64_t outstanding,uint64_t disk_reserve,budget &result,std::string &error){
    budget next;if(!plan_resident_ram(tokens,next,error,tile)){result={};return false;}
    try{
        require(prefix>0&&prefix<tokens&&prefix%page_tokens==0,"Hybrid RAM prefix must be whole pages and leave actual overflow");
        require(disk_reserve>=1024ull*1024*1024,"Hybrid disk reserve must be at least1GiB");
        next.resident_bytes=uint64_t(prefix/page_tokens)*page_bytes;
        next.disk_bytes=next.file_bytes-next.resident_bytes;
        next.page_buffer_bytes=uint64_t((tile+page_tokens-1)/page_tokens)*page_bytes;
        next.bounded_host_bytes=add(next.resident_bytes,add(next.page_buffer_bytes,2ull*tile*record_bytes));
        next.required_free_bytes=add(add(next.disk_bytes,outstanding),disk_reserve);
        result=next;error.clear();return true;
    }catch(const std::exception&e){result={};error=e.what();return false;}
}

std::shared_ptr<llama_dsv41_carry_store::ram_backing> llama_dsv41_carry_store::reserve_ram(uint32_t tokens,uint64_t admission,uint32_t tile){
    budget plan;std::string error;
    require(plan_resident_ram_direct(tokens,plan,error,tile)&&plan.bounded_host_bytes<=admission&&plan.resident_bytes<=SIZE_MAX,"Fixed carry RAM exceeds its explicit host admission");
    auto bank=std::shared_ptr<ram_backing>(new ram_backing);
    bank->bytes=plan.resident_bytes;bank->storage=llama_dsv41_locked_ram::allocate(bank->bytes,admission);return bank;
}

uint64_t llama_dsv41_carry_store::ram_backing::aligned_bytes()const{
#ifdef _WIN32
    SYSTEM_INFO system;GetSystemInfo(&system);const uint64_t page=system.dwPageSize;
#else
    const uint64_t page=uint64_t(sysconf(_SC_PAGESIZE));
#endif
    return (bytes+page-1)/page*page;
}
void llama_dsv41_carry_store::ram_backing::suspend_pages(){
    std::lock_guard<std::mutex> lock(owner);
    require(storage&&bytes&&!paged_out,"Carry RAM bank is absent or already suspended");
    llama_dsv41_locked_ram::suspend_pages(storage.get(),aligned_bytes());paged_out=true;
}
void llama_dsv41_carry_store::ram_backing::resume_pages(){
    std::lock_guard<std::mutex> lock(owner);
    require(storage&&bytes&&paged_out,"Carry RAM bank is not suspended");
    llama_dsv41_locked_ram::resume_pages(storage.get(),aligned_bytes());paged_out=false;
}

bool llama_dsv41_carry_store::retire_shared_ram(std::string &error){
    return p->guarded([&]{
        require(p->state==implementation::sealed&&!p->lease&&p->options.shared_ram&&p->shared_bank_claim.owns_lock(),"Only a completed sealed shared RAM owner can retire");
        p->state=implementation::poisoned;++p->ram_generation;p->resident.reset();p->resident_bytes=0;
        p->shared_bank_claim.unlock();p->options.shared_ram.reset();
    },error);
}

bool llama_dsv41_carry_store::create(const config & options, std::string & error) {
    return p->guarded([&] {
        require(p->state == implementation::fresh, "Carry store cannot be reopened or reused");
        require(!options.shared_ram||(options.resident_ram&&options.direct_ram_tiles&&!options.ram_prefix_tokens),"Shared RAM backing requires explicit direct single RAM bank");
        require(!options.resident_ram||!options.ram_prefix_tokens,"RAM and hybrid carry modes are mutually exclusive");
        require(!options.direct_ram_tiles||options.resident_ram,"Direct-only tile accounting requires the single resident RAM bank");
        require(!options.export_hidden_after_head||options.defer_hidden_readback,
            "Post-head hidden export requires explicit full-device deferred readback");
        require(!options.defer_hidden_readback||(options.resident_ram&&options.direct_ram_tiles),
            "Deferred hidden readback requires direct-only resident RAM accounting");
        require(options.resident_ram||options.ram_prefix_tokens||options.max_tile_tokens==page_tokens,"Larger file tiles require explicit hybrid carry");
        require(options.ram_prefix_tokens||options.disk_reserve_bytes==reserve_bytes,"Legacy carry retains its30GiB reserve");
        std::string why;
        require(options.resident_ram ? (options.direct_ram_tiles?plan_resident_ram_direct(options.tokens,p->capacity,why,options.max_tile_tokens):plan_resident_ram(options.tokens, p->capacity, why, options.max_tile_tokens)) :
                options.ram_prefix_tokens?plan_hybrid(options.tokens,options.ram_prefix_tokens,options.max_tile_tokens,options.outstanding_bytes,options.disk_reserve_bytes,p->capacity,why):
                plan(options.tokens, options.outstanding_bytes, p->capacity, why), "Invalid carry capacity plan");
        require((options.resident_ram || options.admitted_file_bytes >= p->capacity.disk_bytes) &&
                options.admitted_host_bytes >= p->capacity.bounded_host_bytes,
                "Carry file/page RAM were not explicitly admitted");
        p->options = options;
        const uint16_t little = 1;
        require(*reinterpret_cast<const uint8_t *>(&little) == 1, "Native carry requires a little-endian host");
        if (options.resident_ram) {
            require(p->capacity.resident_bytes <= SIZE_MAX,
                    "Resident carry bank exceeds addressable host allocation");
            p->actual_budget(); // Full bank + page/tile scratch charged BEFORE allocating.
            p->locked_page=llama_dsv41_locked_ram::allocate(page_bytes,options.admitted_host_bytes);
            p->page=p->locked_page.get();
            require(p->page != nullptr, "Resident carry aligned page allocation failed");
            if(options.shared_ram){
                require(options.shared_ram->storage&&options.shared_ram->bytes>=p->capacity.resident_bytes&&options.shared_ram->bytes<=options.admitted_host_bytes,
                    "Shared RAM bank differs from the admitted physical extent");
                p->shared_bank_claim=std::unique_lock<std::mutex>(options.shared_ram->owner,std::try_to_lock);
                require(p->shared_bank_claim.owns_lock(),"Shared RAM backing already has a live carry-store owner");
                p->resident=options.shared_ram->storage;
                std::memset(p->resident.get(),0,size_t(p->capacity.resident_bytes));
            }else p->resident=llama_dsv41_locked_ram::allocate(p->capacity.resident_bytes,options.admitted_host_bytes);
            p->resident_bytes = p->capacity.resident_bytes;
            p->state = implementation::seeding; p->header(); p->flush();
            return;
        }
        if(options.ram_prefix_tokens){
            p->resident=llama_dsv41_locked_ram::allocate(p->capacity.resident_bytes,options.admitted_host_bytes);
            p->resident_bytes=p->capacity.resident_bytes;
        }
        p->path = std::filesystem::u8path(options.path);
        require(p->path.is_absolute() && p->path == p->path.lexically_normal() && p->path.has_filename(),
                "Carry path must be absolute and normalized");
        require(std::filesystem::is_directory(p->path.parent_path()), "Carry parent must already exist");
        require(!std::filesystem::exists(std::filesystem::symlink_status(p->path)), "Carry path already exists");
        p->actual_budget();
        std::vector<std::filesystem::path> chain;
        for (auto d = p->path.parent_path();; d = d.parent_path()) {
            chain.push_back(d); if (d == d.root_path()) { break; }
        }
        std::reverse(chain.begin(), chain.end());
        for (const auto & d : chain) {
#ifdef _WIN32
            const auto attrs = GetFileAttributesW(d.c_str());
            require(attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY) && !(attrs & FILE_ATTRIBUTE_REPARSE_POINT),
                    "Carry ancestor must be a physical directory");
            HANDLE h = CreateFileW(d.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE,
                nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
            require(h != INVALID_HANDLE_VALUE, "Cannot pin physical carry ancestor against rename");
            p->parents.push_back(h);
            p->parent_paths.push_back(d);
            BY_HANDLE_FILE_INFORMATION opened {};
            require(GetFileInformationByHandle(h, &opened) != 0 &&
                    (opened.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) &&
                    !(opened.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT),
                    "Opened carry ancestor is a reparse/non-directory item");
#else
            const int fd = ::open(d.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
            require(fd >= 0, "Cannot open physical carry ancestor");
            p->parents.push_back(fd); p->parent_paths.push_back(d);
#endif
            // Windows denies deletion/rename through the held handle. POSIX
            // descriptors are not rename locks: compare their actual inode to
            // every named ancestor before creation and before each I/O instead.
            p->check_parents();
        }
        p->check_parents(); p->actual_budget();
#ifdef _WIN32
        require(p->path.filename().wstring().find(L':') == std::wstring::npos, "Carry alternate data streams are forbidden");
        require(GetDriveTypeW(p->path.root_path().c_str()) == DRIVE_FIXED, "Carry requires an actual fixed local volume");
        wchar_t device[32768] {};
        require(QueryDosDeviceW(p->path.root_name().c_str(), device, 32768) != 0 &&
                std::wstring(device).find(L"\\Device\\HarddiskVolume") == 0,
                "Carry drive must be a physical volume, not a substituted path");
        p->locked_page=llama_dsv41_locked_ram::allocate(p->capacity.page_buffer_bytes,options.admitted_host_bytes);
        p->page=p->locked_page.get();
        require(p->page != nullptr, "Carry aligned page allocation failed");
        p->check_parents(); p->actual_budget();
        p->file = CreateFileW(p->path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ,
            nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_NO_BUFFERING, nullptr);
        require(p->file != INVALID_HANDLE_VALUE, "Cannot create exclusive owned carry file");
        BY_HANDLE_FILE_INFORMATION info {};
        require(GetFileInformationByHandle(p->file, &info) != 0, "Cannot inspect new carry file");
        p->volume = info.dwVolumeSerialNumber; p->index_high = info.nFileIndexHigh; p->index_low = info.nFileIndexLow;
        DWORD sectors = 0, sector_bytes = 0, free_clusters = 0, clusters = 0;
        require(GetDiskFreeSpaceW(p->path.root_path().c_str(), &sectors, &sector_bytes, &free_clusters, &clusters) != 0 &&
                sector_bytes > 0 && alignment % sector_bytes == 0, "Unsupported carry volume sector alignment");
#else
        p->locked_page=llama_dsv41_locked_ram::allocate(p->capacity.page_buffer_bytes,options.admitted_host_bytes);
        p->page=p->locked_page.get();
        p->check_parents(); p->actual_budget();
        // Bind creation to the already-open physical parent. A namespace race
        // cannot redirect this CREATE_NEW into a symlink's unrelated directory.
        p->file = openat(p->parents.back(), p->path.filename().c_str(), O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        require(p->file >= 0, "Cannot create exclusive owned carry file");
        require(flock(p->file, LOCK_EX | LOCK_NB) == 0, "Cannot lease new carry file");
#if defined(__APPLE__)
        require(fcntl(p->file, F_NOCACHE, 1) == 0 && fcntl(p->file, F_RDAHEAD, 0) == 0,
                "Cannot disable carry file cache/read-ahead");
#endif
        struct stat info {};
        require(fstat(p->file, &info) == 0, "Cannot inspect new carry file");
        p->device = info.st_dev; p->inode = info.st_ino;
#if defined(__APPLE__)
        p->modified = info.st_mtimespec;
#else
        p->modified = info.st_mtim;
#endif
#endif
        p->check_identity(); p->actual_budget();
        p->state = implementation::seeding; p->header(); p->flush();
    }, error);
}

bool llama_dsv41_carry_store::seed(const tile & input, std::string & error) {
    return p->guarded([&] {
        require(p->state == implementation::seeding && !p->lease && input.start == p->cursor, "Noncontiguous, leased or repeated carry seed");
        p->validate(input, 0, 0);
        const auto offset = p->page_offset(input.start);
        p->buffer_offset=offset;
        if (offset == p->length) { std::memset(p->page, 0, page_bytes); }
        else { p->io(false, offset, page_bytes); }
        p->pack(input, true); p->io(true, offset, page_bytes); p->cursor += input.count;
    }, error);
}
bool llama_dsv41_carry_store::finish_seed(std::string & error) {
    return p->guarded([&] {
        require(p->state == implementation::seeding && !p->lease && p->cursor == p->options.tokens && p->length == p->capacity.file_bytes,
                "Cannot finish incomplete carry seed");
        p->state = implementation::between_layers; p->cursor = 0; p->header(); p->flush();
    }, error);
}
bool llama_dsv41_carry_store::begin_layer(uint32_t layer, uint32_t selected, uint32_t candidates, std::string & error) {
    return p->guarded([&] {
        require(p->state == implementation::between_layers && layer == p->layers && layer < 40 && !p->lease,
                "Carry layers must run exactly once in order 0..39");
        require(selected <= max_selected && candidates <= max_candidates, "Carry output widths exceed fixed maximum strides");
        require(selected_source(layer) || selected == p->selected_width, "Reader layer changed selected width");
        require(layer == 20 || candidates == p->candidate_width, "Candidate width can change only at source layer 20");
        p->next_selected = selected; p->next_candidates = candidates;
        p->cursor = 0; p->state = implementation::running_layer; p->header();
    }, error);
}
bool llama_dsv41_carry_store::read_tile(uint32_t start, uint32_t count, tile & input, std::string & error) {
    return p->guarded([&] {
        require(p->state == implementation::running_layer && !p->lease && start == p->cursor,
                "Carry read requires the next unleased input tile");
        require(!p->options.direct_ram_tiles||count<=page_tokens,"Direct RAM mode refuses vector reads larger than one page before allocation");
        require(!p->options.defer_hidden_readback,"Device-resident hidden generations cannot be read through stale host vectors");
        p->range(start, count);
        p->direct_rows=p->options.resident_ram&&count>page_tokens;
        p->buffer_offset=p->page_offset(start);
        const auto extent=uint64_t((start%page_tokens+count+page_tokens-1)/page_tokens)*page_bytes;
        if(!p->direct_rows)p->io(false, p->buffer_offset, size_t(extent));
        p->unpack(start, count, input);
        p->lease_start = start; p->lease_count = count; p->lease = true;
    }, error);
}
bool llama_dsv41_carry_store::write_tile(const tile & output, std::string & error) {
    return p->guarded([&] {
        require(p->state == implementation::running_layer && p->lease && !p->ram_direct_lease && output.start == p->lease_start && output.count == p->lease_count,
                "Carry write does not own the exact consumed input lease");
        p->validate(output, p->next_selected, p->next_candidates);
        p->pack(output, false);
        if(!p->direct_rows)p->io(true, p->page_offset(output.start), size_t((output.start%page_tokens+output.count+page_tokens-1)/page_tokens)*page_bytes);
        p->cursor += output.count; p->lease = false;p->direct_rows=false;
    }, error);
}

uint8_t *llama_dsv41_carry_store::ram_lease::row(uint32_t relative)const{
    require(base&&relative<rows,"RAM lease row is outside its borrowed range");
    const uint64_t token=uint64_t(first)+relative;
    const uint64_t at=ram_header_bytes+(token/page_tokens)*ram_page_bytes+(token%page_tokens)*ram_record_bytes;
    require(at<=bytes&&ram_record_bytes<=bytes-at,"RAM lease row exceeds the admitted physical bank");
    return base+at;
}

bool llama_dsv41_carry_store::ram_storage(void *&base,uint64_t &bytes,std::string &error){
    base=nullptr;bytes=0;
    return p->guarded([&]{
        require(p->options.resident_ram&&p->state!=implementation::fresh&&p->state!=implementation::poisoned&&!p->lease,
            "Direct RAM storage requires the admitted, unleased resident bank");
        p->check_identity();p->actual_budget();base=p->resident.get();bytes=p->resident_bytes;
    },error);
}

bool llama_dsv41_carry_store::acquire_ram_tile(uint32_t start,uint32_t count,ram_lease &out,std::string &error){
    out={};
    return p->guarded([&]{
        require(p->options.resident_ram&&p->state==implementation::running_layer&&!p->lease&&start==p->cursor,
            "Direct RAM read requires the next unleased resident tile");
        p->range(start,count,true);p->check_identity();p->actual_budget();
        require(p->ram_identity&&p->ram_generation<UINT64_MAX,"Direct RAM lease identity exhausted");
        ram_lease value;value.base=p->resident.get();value.bytes=p->resident_bytes;
        value.store_id=p->ram_identity;value.generation=++p->ram_generation;
        value.first=start;value.rows=count;value.sw=p->selected_width;value.cw=p->candidate_width;
        value.next_sw=p->next_selected;value.next_cw=p->next_candidates;
        (void)value.row(0);(void)value.row(count-1);
        p->lease_start=start;p->lease_count=count;p->lease=true;p->ram_direct_lease=true;out=value;
    },error);
}

bool llama_dsv41_carry_store::acquire_ram_seed(uint32_t start,uint32_t count,ram_lease &out,std::string &error){
    out={};return p->guarded([&]{
        require(p->options.resident_ram&&p->options.direct_ram_tiles&&p->state==implementation::seeding&&!p->lease&&start==p->cursor,
            "Direct RAM seed requires the next unleased initializing tile");
        p->range(start,count,true);p->check_identity();p->actual_budget();
        require(p->ram_identity&&p->ram_generation<UINT64_MAX,"Direct RAM seed identity exhausted");
        ram_lease value;value.base=p->resident.get();value.bytes=p->resident_bytes;
        value.store_id=p->ram_identity;value.generation=++p->ram_generation;
        value.first=start;value.rows=count;value.seeding=true;
        (void)value.row(0);(void)value.row(count-1);
        // The ordinary seed initializes unused compact-ID slots to -1. Keep
        // those exact bytes even though their initial logical widths are zero.
        for(uint32_t row=0;row<count;++row)std::memset(value.row(row)+selected_offset,0xff,(max_selected+max_candidates)*4);
        p->lease_start=start;p->lease_count=count;p->lease=true;p->ram_direct_lease=true;out=value;
    },error);
}

bool llama_dsv41_carry_store::complete_ram_seed(const ram_lease &value,const ram_completion &done,std::string &error){
    return p->guarded([&]{
        require(p->options.resident_ram&&p->options.direct_ram_tiles&&p->state==implementation::seeding&&p->lease&&p->ram_direct_lease,
            "Direct RAM seed completion has no initializing lease");
        auto exact=[&](const ram_lease &x){return x.seeding&&x.base==p->resident.get()&&x.bytes==p->resident_bytes&&
            x.store_id==p->ram_identity&&x.generation==p->ram_generation&&x.first==p->cursor&&x.first==p->lease_start&&
            x.rows==p->lease_count&&!x.sw&&!x.cw&&!x.next_sw&&!x.next_cw;};
        require(exact(value)&&exact(done.lease)&&done.hidden_mirrored&&done.selected_unchanged&&done.candidates_unchanged,
            "Direct RAM seed lacks exact completed validation/copy proof");
        p->check_identity();p->actual_budget();
        p->cursor+=value.rows;
        p->length=std::max(p->length,alignment+uint64_t((p->cursor+page_tokens-1)/page_tokens)*page_bytes);
        require(p->length<=p->capacity.file_bytes,"Direct RAM seed extent exceeds its bank");
        p->lease=false;p->ram_direct_lease=false;
    },error);
}

bool llama_dsv41_carry_store::lookahead_ram_tile(const ram_lease &parent,uint32_t count,ram_lookahead &out,std::string &error){
    out={};return p->guarded([&]{
        require(!p->options.defer_hidden_readback,"Deferred device hidden generations forbid stale RAM lookahead");
        require(p->options.resident_ram&&p->state==implementation::running_layer&&p->lease&&p->ram_direct_lease,
            "RAM lookahead requires a live parent direct lease");
        require(parent.base==p->resident.get()&&parent.bytes==p->resident_bytes&&parent.store_id==p->ram_identity&&
            parent.generation==p->ram_generation&&parent.generation<UINT64_MAX&&parent.first==p->cursor&&
            parent.first==p->lease_start&&parent.rows==p->lease_count&&parent.sw==p->selected_width&&parent.cw==p->candidate_width&&
            parent.next_sw==p->next_selected&&parent.next_cw==p->next_candidates,"RAM lookahead parent credit changed");
        const uint64_t first=uint64_t(parent.first)+parent.rows;
        require(first<p->options.tokens,"RAM lookahead cannot cross the layer boundary");
        p->range(uint32_t(first),count,true);p->check_identity();p->actual_budget();
        ram_lookahead result;result.lease=parent;result.lease.first=uint32_t(first);result.lease.rows=count;
        (void)result.row(0);(void)result.row(count-1);out=result;
    },error);
}

bool llama_dsv41_carry_store::complete_ram_tile(const ram_lease &value,const ram_completion &done,std::string &error){
    return p->guarded([&]{
        require(p->options.resident_ram&&p->state==implementation::running_layer&&p->lease&&p->ram_direct_lease,
            "Direct RAM completion has no live resident lease");
        auto exact=[&](const ram_lease &x){return !x.seeding&&x.base==p->resident.get()&&x.bytes==p->resident_bytes&&
            x.store_id==p->ram_identity&&x.generation==p->ram_generation&&x.first==p->lease_start&&x.rows==p->lease_count&&
            x.first==p->cursor&&x.sw==p->selected_width&&x.cw==p->candidate_width&&
            x.next_sw==p->next_selected&&x.next_cw==p->next_candidates;};
        require(exact(value)&&exact(done.lease),"Direct RAM completion differs from its exact bank/layer/tile lease");
        require(done.hidden_mirrored==(!p->options.defer_hidden_readback||(p->layers==39&&!p->options.export_hidden_after_head)),
            "Hidden readback completion differs from explicit resident/final-layer contract");
        require((selected_source(p->layers)||!p->selected_width||done.selected_unchanged)&&
            (p->layers==20||!p->candidate_width||done.candidates_unchanged),
            "Reader-layer compact IDs lack unchanged graph-source proof");
        p->check_identity();p->actual_budget();
        p->cursor+=value.rows;p->lease=false;p->ram_direct_lease=false;p->direct_rows=false;
    },error);
}
bool llama_dsv41_carry_store::acquire_ram_export(uint32_t start,uint32_t count,ram_lease &out,std::string &error){
    out={};return p->guarded([&]{
        require(p->options.export_hidden_after_head&&p->options.resident_ram&&p->options.direct_ram_tiles&&
            p->state==implementation::awaiting_head&&p->layers==40&&!p->lease&&start==p->export_cursor,
            "Final hidden export requires the next unleased completed-device rows");
        p->range(start,count,true);require(count<=p->options.max_tile_tokens,"Final hidden export exceeds admitted tile");
        require(p->ram_identity&&p->ram_generation<UINT64_MAX,"Final hidden export identity exhausted");
        p->check_identity();p->actual_budget();ram_lease x;x.base=p->resident.get();x.bytes=p->resident_bytes;
        x.store_id=p->ram_identity;x.generation=++p->ram_generation;x.first=start;x.rows=count;
        x.sw=x.next_sw=p->selected_width;x.cw=x.next_cw=p->candidate_width;
        (void)x.row(0);(void)x.row(count-1);p->lease_start=start;p->lease_count=count;
        p->lease=true;p->ram_direct_lease=true;out=x;
    },error);
}
bool llama_dsv41_carry_store::complete_ram_export(const ram_lease &value,const ram_completion &done,std::string &error){
    return p->guarded([&]{
        require(p->options.export_hidden_after_head&&p->state==implementation::awaiting_head&&p->layers==40&&p->lease&&p->ram_direct_lease,
            "Final hidden export completion has no live export lease");
        auto exact=[&](const ram_lease &x){return !x.seeding&&x.base==p->resident.get()&&x.bytes==p->resident_bytes&&
            x.store_id==p->ram_identity&&x.generation==p->ram_generation&&x.first==p->export_cursor&&
            x.first==p->lease_start&&x.rows==p->lease_count&&x.sw==p->selected_width&&x.next_sw==p->selected_width&&
            x.cw==p->candidate_width&&x.next_cw==p->candidate_width;};
        require(exact(value)&&exact(done.lease)&&done.hidden_mirrored&&done.selected_unchanged&&done.candidates_unchanged,
            "Final hidden export lacks its exact completed copy proof");
        p->check_identity();p->actual_budget();p->export_cursor+=value.rows;p->lease=false;p->ram_direct_lease=false;
    },error);
}
bool llama_dsv41_carry_store::commit_layer(std::string & error) {
    return p->guarded([&] {
        require(p->state == implementation::running_layer && !p->lease && p->cursor == p->options.tokens,
                "Cannot commit an incomplete/leased carry layer");
        p->flush(); ++p->layers; p->selected_width = p->next_selected; p->candidate_width = p->next_candidates;
        p->state = p->layers == 40 ? implementation::awaiting_head : implementation::between_layers;
        p->header();
    }, error);
}
bool llama_dsv41_carry_store::read_final_tile(uint32_t start, uint32_t count, tile & output, std::string & error) {
    return p->guarded([&] {
        require(p->state == implementation::awaiting_head && !p->lease && p->layers == 40,
                "Head carry requires all 40 complete layers");
        require(!p->options.export_hidden_after_head||p->export_cursor==p->options.tokens,
            "Final host hidden rows are stale until export readback completes");
        require(!p->options.direct_ram_tiles||count<=page_tokens,"Direct RAM mode refuses head vectors larger than one page before allocation");
        p->range(start, count);p->direct_rows=p->options.resident_ram&&count>page_tokens;p->buffer_offset=p->page_offset(start);
        if(!p->direct_rows)p->io(false,p->buffer_offset,size_t((start%page_tokens+count+page_tokens-1)/page_tokens)*page_bytes);
        p->unpack(start,count,output);p->direct_rows=false;
    }, error);
}
bool llama_dsv41_carry_store::certify_frontier(std::string & error) {
    return p->guarded([&] {
        require(p->state == implementation::awaiting_head && !p->lease && p->layers == 40,
                "Cannot publish a partial or failed native frontier");
        require(!p->options.export_hidden_after_head||p->export_cursor==p->options.tokens,
            "Cannot publish a device frontier before every hidden export row is copied");
        p->check_identity(); p->actual_budget(); p->flush();
        p->state = implementation::sealed; p->header(); p->flush();
    }, error);
}
bool llama_dsv41_carry_store::frontier_ready() const {
    std::lock_guard<std::mutex> guard(p->mutex); return p->state == implementation::sealed;
}
bool llama_dsv41_carry_store::failed() const {
    std::lock_guard<std::mutex> guard(p->mutex); return p->state == implementation::poisoned;
}
uint32_t llama_dsv41_carry_store::completed_layers() const {
    std::lock_guard<std::mutex> guard(p->mutex); return p->layers;
}
void llama_dsv41_carry_store::invalidate() {
    std::lock_guard<std::mutex> guard(p->mutex); p->poison();
}
