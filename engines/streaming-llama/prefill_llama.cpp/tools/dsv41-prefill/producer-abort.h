#pragma once
// LLAMADECODE producer abort: every producer failure ends in ONE well-formed
// session error reply on the private session pipe plus a final stderr line
//   LLAMADECODE_PRODUCER_ABORT kind=<kind> code=0x<code> reason=<reason>
// instead of a silently closed pipe (the client saw "reply truncated").
// kinds: exception (std::exception on the session thread, including a worker
// failure marshalled to it), bad_alloc, unknown_exception, terminate (an
// exception escaped a thread: std::terminate), abort (abort() with no exception), seh (an unhandled structured
// exception: access violation, stack overflow, ...).
// The reply and the reason line are built without heap allocation into fixed
// buffers, so they still go out under bad_alloc or from the crash handlers.
#include <windows.h>
#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <string>

namespace producer_abort {

inline HANDLE wire = nullptr;                 // private session output once bound (else stderr only)
inline char prefix[512] = {};                 // {"status":"error","session":..,"model_sha256":..[,"owner_epoch":N]
inline std::atomic<uint64_t> sequence{0};     // the session operation being served (0 = startup / ready)
inline std::atomic<bool> reported{false};     // one report per process

// Bounded JSON string body: quotes, backslashes and control bytes escaped; bytes
// >= 0x80 become '?' so a reason is always valid UTF-8 JSON.
inline size_t escape(char *out, size_t cap, const char *in) {
    size_t n = 0;
    if (!cap) return 0;
    for (const unsigned char *p = reinterpret_cast<const unsigned char *>(in ? in : ""); *p && n + 7 < cap; ++p) {
        const unsigned char c = *p;
        if (c == '"' || c == '\\') { out[n++] = '\\'; out[n++] = char(c); }
        else if (c == '\n') { out[n++] = '\\'; out[n++] = 'n'; }
        else if (c == '\r') { out[n++] = '\\'; out[n++] = 'r'; }
        else if (c == '\t') { out[n++] = '\\'; out[n++] = 't'; }
        else if (c < 0x20) n += size_t(std::snprintf(out + n, cap - n, "\\u%04x", unsigned(c)));
        else out[n++] = c < 0x80 ? char(c) : '?';
    }
    out[n] = 0;
    return n;
}

// Bind the reply to the session pipe (after session_output.bind()).
inline void arm(HANDLE output, const std::string &session, const char *model_sha, uint64_t owner_epoch) {
    char s[160], m[160];
    escape(s, sizeof s, session.c_str());
    escape(m, sizeof m, model_sha);
    if (owner_epoch)
        std::snprintf(prefix, sizeof prefix, "{\"status\":\"error\",\"session\":\"%s\",\"model_sha256\":\"%s\",\"owner_epoch\":%llu", s, m,
                      (unsigned long long)owner_epoch);
    else
        std::snprintf(prefix, sizeof prefix, "{\"status\":\"error\",\"session\":\"%s\",\"model_sha256\":\"%s\"", s, m);
    wire = output;
}

// The reply line for (kind, code, reason); also used by the tests.
inline int reply_line(char *line, size_t cap, const char *kind, uint32_t code, const char *reason) {
    char text[2048];
    escape(text, sizeof text, reason && *reason ? reason : "(no reason)");
    return std::snprintf(line, cap, "%s,\"sequence\":%llu,\"error_kind\":\"%s\",\"error_code\":%lu,\"reason\":\"%s\"}\n", prefix,
                         (unsigned long long)sequence.load(), kind, (unsigned long)code, text);
}

// Write the stderr reason line and (once armed) the error reply. First report wins.
inline void report(const char *kind, const char *reason, uint32_t code = 0) {
    if (reported.exchange(true)) return;
    char text[2048];
    escape(text, sizeof text, reason && *reason ? reason : "(no reason)");
    std::fprintf(stderr, "LLAMADECODE_PRODUCER_ABORT kind=%s code=0x%08lX sequence=%llu reason=%s\n", kind, (unsigned long)code,
                 (unsigned long long)sequence.load(), text);
    std::fflush(stderr);
    if (!wire) return;
    char line[4096];
    const int n = reply_line(line, sizeof line, kind, code, reason);
    DWORD sent = 0;
    if (n > 0) WriteFile(wire, line, DWORD(std::min<size_t>(size_t(n), sizeof line - 1)), &sent, nullptr);
}

// Report the exception being handled (call inside a catch block).
inline void report_current(const char *context = nullptr, const char *kind_override = nullptr) {
    char reason[2048];
    const char *kind = "unknown_exception";
    try {
        throw;
    } catch (const std::bad_alloc &e) {
        kind = "bad_alloc";
        std::snprintf(reason, sizeof reason, "%s%sstd::bad_alloc (allocation failed): %s", context ? context : "", context ? ": " : "", e.what());
    } catch (const std::exception &e) {
        // A worker's or the sweep's bad_alloc may arrive converted to its message text.
        kind = std::strstr(e.what(), "std::bad_alloc") || std::strcmp(e.what(), "bad allocation") == 0 ? "bad_alloc" : "exception";
        std::snprintf(reason, sizeof reason, "%s%s%s", context ? context : "", context ? ": " : "", e.what());
    } catch (...) {
        std::snprintf(reason, sizeof reason, "%s%snon-standard C++ exception", context ? context : "", context ? ": " : "");
    }
    report(kind_override ? kind_override : kind, reason);
}

// Last-chance handlers: an exception escaping any thread (std::terminate) or an
// unhandled structured exception still leaves the reason line and the reply.
inline LONG WINAPI unhandled(EXCEPTION_POINTERS *info) {
    char reason[256];
    const DWORD code = info && info->ExceptionRecord ? info->ExceptionRecord->ExceptionCode : 0;
    const void *address = info && info->ExceptionRecord ? info->ExceptionRecord->ExceptionAddress : nullptr;
    std::snprintf(reason, sizeof reason, "unhandled structured exception 0x%08lX at %p (thread %lu)", (unsigned long)code, address,
                  (unsigned long)GetCurrentThreadId());
    report("seh", reason, code);
    return EXCEPTION_CONTINUE_SEARCH;   // the normal crash path (WER, exit code) continues
}
inline void terminate_handler() {
    char context[64];
    std::snprintf(context, sizeof context, "std::terminate on thread %lu", (unsigned long)GetCurrentThreadId());
    if (std::current_exception()) {
        try { std::rethrow_exception(std::current_exception()); } catch (...) { report_current(context, "terminate"); }
    } else report("terminate", context);
    std::abort();
}
// MSVC's std::set_terminate is per thread, so an exception escaping a worker
// thread reaches abort() through the default handler: SIGABRT (process-wide)
// reports it, with the escaping exception's text when it is still current.
inline void abort_signal(int) {
    char context[64];
    std::snprintf(context, sizeof context, "abort() on thread %lu", (unsigned long)GetCurrentThreadId());
    if (std::current_exception()) {
        try { std::rethrow_exception(std::current_exception()); } catch (...) { report_current(context, "terminate"); }
    } else report("abort", context);
}
inline void install() {
    SetUnhandledExceptionFilter(unhandled);
    std::set_terminate(terminate_handler);
    std::signal(SIGABRT, abort_signal);
}

} // namespace producer_abort
