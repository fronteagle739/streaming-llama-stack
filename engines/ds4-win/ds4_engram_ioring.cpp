// Engram raw-row reads through one IoRing, mirroring streaming-llama's
// llama-dsv41-engram-prefetch.cpp (same API resolution, unbuffered overlapped
// handle verified against the original by FileId, registered file + slot
// arena, fixed depth, nonce-checked completions, drain-before-free cleanup).
//
// Why: ds4_win32_pread issues synchronous ReadFile on a non-overlapped CRT
// handle, which Windows serialises per file object, so the 16 Engram readers
// ran one 264-byte buffered read at a time (23.5 s at 16K against 2.5 s for the
// same rows in streaming-llama). IoRing declarations stay in this TU; the
// functions are resolved at runtime, so the binary gains no Win11 import.
#ifdef _WIN32
#include <sdkddkver.h>
#if !defined(NTDDI_VERSION) || NTDDI_VERSION < NTDDI_WIN10_CO
#undef NTDDI_VERSION
#define NTDDI_VERSION NTDDI_WIN10_CO
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <ioringapi.h>
#include <io.h>
#include <malloc.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {
constexpr uint32_t kDepth = 256, kPage = 8192, kRow = 264;

extern "C" void ds4_win32_request_graceful_abort(const char *why);

/* Owner rule: no force-kill while VRAM/DirectStorage may be live, no exceptions.
 * When submitted reads cannot be confirmed drained, their slot arena, ring, file and
 * event are deliberately leaked (never freed, so a late write lands in memory that is
 * never reused), the engine is marked STUCK through the graceful abort (requests fail,
 * new ones are refused) and the process stays alive. */
void mark_unconfirmed_io(bool &leak, const char *where) {
    if (!leak) {
        std::fprintf(stderr, "DS4_ENGRAM_IORING STUCK: unconfirmed IoRing reads at %s; buffers leaked (never freed), "
                             "engine failing requests, process NOT killed; owner action needed\n", where);
        std::fflush(stderr);
    }
    leak = true;
    ds4_win32_request_graceful_abort("Engram IoRing reads could not be confirmed drained");
}
void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }

struct api_table {
    decltype(&::CreateIoRing) CreateIoRing = nullptr;
    decltype(&::IsIoRingOpSupported) IsIoRingOpSupported = nullptr;
    decltype(&::SetIoRingCompletionEvent) SetIoRingCompletionEvent = nullptr;
    decltype(&::BuildIoRingRegisterFileHandles) BuildIoRingRegisterFileHandles = nullptr;
    decltype(&::BuildIoRingRegisterBuffers) BuildIoRingRegisterBuffers = nullptr;
    decltype(&::BuildIoRingReadFile) BuildIoRingReadFile = nullptr;
    decltype(&::SubmitIoRing) SubmitIoRing = nullptr;
    decltype(&::PopIoRingCompletion) PopIoRingCompletion = nullptr;
    decltype(&::CloseIoRing) CloseIoRing = nullptr;
    bool ok = false;
    api_table() {
        const auto m = GetModuleHandleW(L"KernelBase.dll");
        if (!m) return;
        CreateIoRing = reinterpret_cast<decltype(CreateIoRing)>(GetProcAddress(m, "CreateIoRing"));
        IsIoRingOpSupported = reinterpret_cast<decltype(IsIoRingOpSupported)>(GetProcAddress(m, "IsIoRingOpSupported"));
        SetIoRingCompletionEvent = reinterpret_cast<decltype(SetIoRingCompletionEvent)>(GetProcAddress(m, "SetIoRingCompletionEvent"));
        BuildIoRingRegisterFileHandles = reinterpret_cast<decltype(BuildIoRingRegisterFileHandles)>(GetProcAddress(m, "BuildIoRingRegisterFileHandles"));
        BuildIoRingRegisterBuffers = reinterpret_cast<decltype(BuildIoRingRegisterBuffers)>(GetProcAddress(m, "BuildIoRingRegisterBuffers"));
        BuildIoRingReadFile = reinterpret_cast<decltype(BuildIoRingReadFile)>(GetProcAddress(m, "BuildIoRingReadFile"));
        SubmitIoRing = reinterpret_cast<decltype(SubmitIoRing)>(GetProcAddress(m, "SubmitIoRing"));
        PopIoRingCompletion = reinterpret_cast<decltype(PopIoRingCompletion)>(GetProcAddress(m, "PopIoRingCompletion"));
        CloseIoRing = reinterpret_cast<decltype(CloseIoRing)>(GetProcAddress(m, "CloseIoRing"));
        ok = CreateIoRing && IsIoRingOpSupported && SetIoRingCompletionEvent && BuildIoRingRegisterFileHandles &&
             BuildIoRingRegisterBuffers && BuildIoRingReadFile && SubmitIoRing && PopIoRingCompletion && CloseIoRing;
    }
};
const api_table &api() { static const api_table table; return table; }

struct io_scope {
    HIORING ring = nullptr;
    HANDLE file = INVALID_HANDLE_VALUE, event = nullptr;
    uint8_t *pages = nullptr;
    uint32_t submitted = 0;
    bool leak = false;   /* unconfirmed IO: keep every resource alive forever */
    io_scope() = default;
    io_scope(const io_scope &) = delete;
    io_scope &operator=(const io_scope &) = delete;
    ~io_scope() {
        if (leak) return;
        if (ring) {
            if (submitted && file != INVALID_HANDLE_VALUE) CancelIoEx(file, nullptr);
            const auto deadline = GetTickCount64() + 5000;
            while (submitted) {
                IORING_CQE c{};
                const auto hr = api().PopIoRingCompletion(ring, &c);
                if (hr == S_OK) { --submitted; continue; }
                if (hr != S_FALSE || GetTickCount64() >= deadline) { mark_unconfirmed_io(leak, "drain"); return; }
                const auto status = WaitForSingleObject(event, 50);
                if (status != WAIT_OBJECT_0 && status != WAIT_TIMEOUT) { mark_unconfirmed_io(leak, "drain wait"); return; }
            }
            // Close abandons unsubmitted entries but does not cancel submitted
            // reads; every submitted completion was drained above.
            if (FAILED(api().CloseIoRing(ring))) { mark_unconfirmed_io(leak, "close"); return; }
        }
        if (event) CloseHandle(event);
        if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
        if (pages) _aligned_free(pages);
    }
};

void read_rows(int fd, uint64_t base, const uint32_t *rows, size_t n, uint8_t *out) {
    require(api().ok, "Required IoRing APIs unavailable; no serial fallback");
    const HANDLE original = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    require(original != INVALID_HANDLE_VALUE, "Engram table descriptor has no OS handle");
    std::wstring path(32768, L'\0');
    const DWORD len = GetFinalPathNameByHandleW(original, path.data(), DWORD(path.size()), FILE_NAME_NORMALIZED);
    require(len && len < path.size(), "Cannot resolve the Engram table path for its IoRing");
    path.resize(len);
    LARGE_INTEGER size{};
    require(GetFileSizeEx(original, &size) && size.QuadPart > 0, "Cannot size the Engram table file");
    const uint64_t file_bytes = uint64_t(size.QuadPart);
    io_scope io;
    io.file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                          FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    require(io.file != INVALID_HANDLE_VALUE, "Cannot open the Engram table file for its own IoRing");
    FILE_ID_INFO a{}, b{};
    require(GetFileInformationByHandleEx(original, FileIdInfo, &a, sizeof(a)) &&
            GetFileInformationByHandleEx(io.file, FileIdInfo, &b, sizeof(b)) &&
            a.VolumeSerialNumber == b.VolumeSerialNumber && !std::memcmp(a.FileId.Identifier, b.FileId.Identifier, 16),
            "Engram IoRing descriptor differs from the opened table file");
    IORING_CREATE_FLAGS flags{};
    require(SUCCEEDED(api().CreateIoRing(IORING_VERSION_1, flags, kDepth, 2 * kDepth, &io.ring)), "Required Engram IoRing unavailable; no serial fallback");
    require(api().IsIoRingOpSupported(io.ring, IORING_OP_READ) && api().IsIoRingOpSupported(io.ring, IORING_OP_REGISTER_FILES) &&
            api().IsIoRingOpSupported(io.ring, IORING_OP_REGISTER_BUFFERS), "Engram IoRing operations unavailable");
    io.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    require(io.event && SUCCEEDED(api().SetIoRingCompletionEvent(io.ring, io.event)), "Cannot register the Engram IoRing completion event");
    io.pages = static_cast<uint8_t *>(_aligned_malloc(size_t(kDepth) * kPage, 4096));
    require(io.pages, "Cannot allocate the Engram IoRing slots");
    IORING_BUFFER_INFO registered{io.pages, kDepth * kPage};
    require(SUCCEEDED(api().BuildIoRingRegisterFileHandles(io.ring, 1, &io.file, 1)) &&
            SUCCEEDED(api().BuildIoRingRegisterBuffers(io.ring, 1, &registered, 2)), "Cannot enqueue Engram IoRing registrations");
    UINT32 count = 0;
    if (api().SubmitIoRing(io.ring, 0, 0, &count) != S_OK) throw std::runtime_error("Engram IoRing registration submit failed");
    if (count != 2) { mark_unconfirmed_io(io.leak, "registration submit"); throw std::runtime_error("Engram IoRing registration submit count differs"); }
    io.submitted = count;
    bool file_ok = false, buffer_ok = false;
    const auto reg_deadline = GetTickCount64() + 5000;
    while (io.submitted) {
        IORING_CQE c{};
        const auto hr = api().PopIoRingCompletion(io.ring, &c);
        if (hr == S_FALSE) {
            require(GetTickCount64() < reg_deadline, "Engram IoRing registration timeout");
            const auto w = WaitForSingleObject(io.event, 50);
            require(w == WAIT_OBJECT_0 || w == WAIT_TIMEOUT, "Engram IoRing registration wait failed");
            continue;
        }
        require(hr == S_OK, "Cannot dequeue an Engram IoRing registration");
        --io.submitted;
        require(SUCCEEDED(c.ResultCode), "Engram IoRing resource registration failed");
        if (c.UserData == 1 && !file_ok) file_ok = true;
        else if (c.UserData == 2 && !buffer_ok) buffer_ok = true;
        else throw std::runtime_error("Unexpected Engram IoRing registration completion");
    }
    struct slot { uint32_t extent = 0, skip = 0; size_t row = 0; UINT_PTR id = 0; bool live = false; };
    std::array<slot, kDepth> slots;
    size_t next = 0, staged = 0;
    UINT_PTR sequence = 0;
    auto issue = [&](uint32_t i) {
        if (next == n) return false;
        auto &s = slots[i];
        require(!s.live, "Engram IoRing reused a live slot");
        s.row = next++;
        const uint64_t offset = base + uint64_t(rows[s.row]) * kRow;
        const uint64_t begin = offset & ~uint64_t(4095);
        s.skip = uint32_t(offset - begin);
        s.extent = (s.skip + kRow + 4095) & ~4095u;
        require(s.extent <= kPage && begin <= file_bytes && s.extent <= file_bytes - begin, "Engram aligned read exceeds the table file");
        require(sequence < (UINTPTR_MAX >> 9), "Engram IoRing nonce overflow");
        s.id = (++sequence << 9) | (i + 1);
        require(SUCCEEDED(api().BuildIoRingReadFile(io.ring, IoRingHandleRefFromIndex(0), IoRingBufferRefFromIndexAndOffset(0, i * kPage),
                                                    s.extent, begin, s.id, IOSQE_FLAGS_NONE)), "Cannot enqueue an Engram row read");
        s.live = true;
        ++staged;
        return true;
    };
    auto submit = [&] {
        if (!staged) return;
        UINT32 done = 0;
        if (api().SubmitIoRing(io.ring, 0, 0, &done) != S_OK) throw std::runtime_error("Engram IoRing submit failed; no serial fallback");
        if (done != staged) { mark_unconfirmed_io(io.leak, "read submit"); throw std::runtime_error("Engram IoRing submit count differs"); }
        io.submitted += done;
        staged = 0;
    };
    const auto began = GetTickCount64();
    for (uint32_t i = 0; i < kDepth; ++i) if (!issue(i)) break;
    submit();
    while (io.submitted) {
        require(GetTickCount64() - began < 120000, "Engram IoRing read exceeded 120 s");
        uint32_t completed = 0;
        for (; completed < 64; ++completed) {
            IORING_CQE c{};
            const auto hr = api().PopIoRingCompletion(io.ring, &c);
            if (hr == S_FALSE) break;
            require(hr == S_OK && io.submitted, "Cannot dequeue an Engram IoRing completion");
            --io.submitted;
            const auto id = uint32_t(c.UserData & 511u);
            require(id && id <= kDepth, "Engram completion names an unknown slot");
            auto &s = slots[id - 1];
            require(s.live && s.id == c.UserData, "Engram completion nonce/ownership differs");
            s.live = false;
            require(SUCCEEDED(c.ResultCode) && c.Information == s.extent, "Engram IoRing read failed or was short");
            std::memcpy(out + s.row * kRow, io.pages + uint64_t(id - 1) * kPage + s.skip, kRow);
            issue(id - 1);
        }
        submit();
        if (!completed && io.submitted) {
            const auto w = WaitForSingleObject(io.event, 100);
            require(w == WAIT_OBJECT_0 || w == WAIT_TIMEOUT, "Engram IoRing completion wait failed");
        }
    }
    require(next == n && !staged, "Incomplete Engram IoRing read");
}
}  // namespace

/* Read n raw 264-byte rows (row ids, table base offset) into out[n * 264], in
 * the given order. Returns false with a reason; never falls back. */
extern "C" bool ds4_engram_ioring_read_rows(int fd, uint64_t base, const uint32_t *rows, size_t n,
                                            uint8_t *out, char *why, size_t why_n) {
    try {
        if (n) read_rows(fd, base, rows, n, out);
        return true;
    } catch (const std::exception &e) {
        if (why && why_n) std::snprintf(why, why_n, "%s", e.what());
        return false;
    }
}
#endif
