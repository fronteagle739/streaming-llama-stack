// LLAMADECODE producer abort: every failure in a session operation ends in a
// well-formed session error reply carrying the reason, plus the stderr line.
// Faults are injected exactly as LLAMADECODE_TEST_FAULT does in the producer
// (llama_dsv41_test_fault::at_layer) and reported by the producer's own catch
// path (producer_abort::report_current). The last-chance handlers (an exception
// escaping a raw thread -> std::terminate; an access violation -> SEH filter)
// run in child processes of this test.
#define NOMINMAX
#include <windows.h>
#include "producer-abort.h"
#include "llama-dsv41-test-fault.h"
#include <nlohmann/json.hpp>
#include <cstdio>
#include <string>
#include <thread>

namespace fault = llama_dsv41_test_fault;
using nlohmann::json;
static int passed = 0, failed = 0;
static void expect(bool ok, const char *what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    ok ? ++passed : ++failed;
}
static const char *model = "2cc6bc254f22a0238020e61e931c0cac865218aada1eed9c0036d9f4031c6e5f";
static const std::string session = "6b646ad77b2ceeccb0b803a13443ac11";

static std::string drain(HANDLE read) {
    std::string out;
    for (;;) {
        DWORD available = 0;
        if (!PeekNamedPipe(read, nullptr, 0, nullptr, &available, nullptr) || !available) return out;
        std::string part(available, '\0');
        DWORD got = 0;
        if (!ReadFile(read, part.data(), available, &got, nullptr)) return out;
        out.append(part.data(), got);
    }
}
// One reply line: newline-terminated, exactly one line, valid JSON.
static bool one_line(const std::string &text, json &row) {
    if (text.empty() || text.back() != '\n' || text.find('\n') != text.size() - 1) return false;
    try { row = json::parse(text); } catch (...) { return false; }
    return true;
}
static bool identity(const json &row, uint64_t seq) {
    return row.value("status", "") == "error" && row.value("session", "") == session && row.value("model_sha256", "") == model &&
           row.value("owner_epoch", 0ull) == 7 && row.value("sequence", ~0ull) == seq;
}
// A session operation (sequence seq) whose sweep reaches layers 0..9 with `tokens` tokens.
static json run_op(const fault::spec &s, uint64_t seq, HANDLE read, size_t tokens = 4096) {
    producer_abort::reported = false;
    fault::fired = false;
    producer_abort::sequence.store(seq);
    try {
        for (uint32_t layer = 0; layer < 10; ++layer) fault::at_layer(layer, tokens, s);
        return json{{"no_fault", true}};
    } catch (...) {
        producer_abort::report_current();   // the producer's catch path
    }
    json row;
    if (!one_line(drain(read), row)) return json{{"malformed", true}};
    return row;
}

static int child(const char *mode, HANDLE wire) {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    producer_abort::arm(wire, session, model, 7);
    producer_abort::install();
    producer_abort::sequence.store(9);
    if (std::string(mode) == "terminate") {
        std::thread worker([] { throw std::runtime_error("escaped raw worker exception"); });
        worker.join();
    } else {
        volatile int *null = nullptr;
        *null = 1;
    }
    return 0;
}
static bool run_child(const char *exe, const char *mode, json &row, DWORD &code, std::string &err) {
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    HANDLE r = nullptr, w = nullptr, er = nullptr, ew = nullptr;
    if (!CreatePipe(&r, &w, &sa, 1 << 16) || !CreatePipe(&er, &ew, &sa, 1 << 16)) return false;
    SetHandleInformation(r, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(er, HANDLE_FLAG_INHERIT, 0);
    char command[1024];
    std::snprintf(command, sizeof command, "\"%s\" --child %s %llu", exe, mode, (unsigned long long)(uintptr_t)w);
    STARTUPINFOA si{};
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = ew;
    si.hStdError = ew;
    PROCESS_INFORMATION pi{};
    if (!CreateProcessA(nullptr, command, nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) return false;
    CloseHandle(w);
    CloseHandle(ew);
    WaitForSingleObject(pi.hProcess, 30000);
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    const bool ok = one_line(drain(r), row);
    err = drain(er);
    CloseHandle(r);
    CloseHandle(er);
    return ok;
}

int main(int argc, char **argv) {
    if (argc == 4 && std::string(argv[1]) == "--child") return child(argv[2], HANDLE(uintptr_t(std::strtoull(argv[3], nullptr, 10))));
    HANDLE r = nullptr, w = nullptr;
    if (!CreatePipe(&r, &w, nullptr, 1 << 16)) return 2;
    producer_abort::arm(w, session, model, 7);

    // Fault specification (LLAMADECODE_TEST_FAULT).
    expect(fault::parse(nullptr).k == fault::kind::none && fault::parse("").k == fault::kind::none, "unset: no fault");
    const auto m = fault::parse("main_throw@3"), k = fault::parse("worker_throw@5"), b = fault::parse("bad_alloc@0");
    expect(m.k == fault::kind::main_throw && m.layer == 3 && k.k == fault::kind::worker_throw && k.layer == 5 && b.k == fault::kind::bad_alloc && b.layer == 0,
           "parse main_throw@3, worker_throw@5, bad_alloc@0");
    int refused = 0;
    for (const char *bad : {"main_throw", "main_throw@", "main_throw@x", "boom@3", "worker_throw@3x", "bad_alloc@2000"}) {
        try { fault::parse(bad); } catch (const std::runtime_error &) { ++refused; }
    }
    expect(refused == 6, "malformed LLAMADECODE_TEST_FAULT values are refused");

    // Main thread: the sweep thread throws at the start of layer 3.
    auto row = run_op(m, 5, r);
    expect(identity(row, 5) && row.value("error_kind", "") == "exception" &&
               row.value("reason", "").find("main_throw at layer 3 on the sweep thread") != std::string::npos,
           "main_throw: one well-formed error reply naming the reason, sequence and identity");
    // Worker thread: thrown on a std::async worker, marshalled to the sweep thread.
    row = run_op(k, 6, r);
    expect(identity(row, 6) && row.value("error_kind", "") == "exception" &&
               row.value("reason", "").find("worker_throw at layer 5 on a worker thread") != std::string::npos,
           "worker_throw: worker exception marshalled to the session thread, error reply with the reason");
    // Allocation failure.
    row = run_op(b, 7, r);
    expect(identity(row, 7) && row.value("error_kind", "") == "bad_alloc" && row.value("reason", "").find("std::bad_alloc") != std::string::npos,
           "bad_alloc: error_kind bad_alloc");
    {   // a bad_alloc converted to its text by the sweep's own catch keeps its kind
        producer_abort::reported = false;
        try { throw std::runtime_error("std::bad_alloc: a host allocation failed during the V4.1 sweep run"); }
        catch (...) { producer_abort::report_current(); }
        json converted;
        expect(one_line(drain(r), converted) && converted.value("error_kind", "") == "bad_alloc", "sweep-converted bad_alloc text keeps error_kind bad_alloc");
    }
    // Fires once, only at its layer, only on a multi-token sweep.
    expect(run_op(m, 8, r, 1).contains("no_fault"), "one-token sweep (warmup/decode) does not fire");
    fault::fired = true;
    expect(run_op(fault::spec{}, 8, r).contains("no_fault"), "no fault armed: nothing fires");
    {
        producer_abort::reported = false;
        fault::fired = false;
        int fires = 0;
        for (int pass = 0; pass < 3; ++pass) try { fault::at_layer(3, 4096, m); } catch (...) { ++fires; }
        expect(fires == 1, "the fault fires once per process");
    }
    // One report per process; reasons are escaped into valid JSON.
    producer_abort::reported = false;
    producer_abort::report("exception", "quote \" backslash \\ newline \n tab \t byte \xe9 end");
    producer_abort::report("exception", "second report is suppressed");
    json escaped;
    expect(one_line(drain(r), escaped) && escaped.value("reason", "") == "quote \" backslash \\ newline \n tab \t byte ? end",
           "reason escaping round-trips; one report per process");

    // Last chance: an exception escaping a raw thread, and an access violation.
    char exe[MAX_PATH];
    GetModuleFileNameA(nullptr, exe, MAX_PATH);
    DWORD code = 0;
    std::string err;
    json crash;
    expect(run_child(exe, "terminate", crash, code, err) && identity(crash, 9) && crash.value("error_kind", "") == "terminate" &&
               crash.value("reason", "").find("escaped raw worker exception") != std::string::npos &&
               err.find("LLAMADECODE_PRODUCER_ABORT kind=terminate") != std::string::npos && code == 3,
           "raw thread exception -> std::terminate: error reply + stderr reason line, exit code 3");
    expect(run_child(exe, "seh", crash, code, err) && identity(crash, 9) && crash.value("error_kind", "") == "seh" &&
               crash.value("error_code", 0ull) == 0xC0000005ull && err.find("LLAMADECODE_PRODUCER_ABORT kind=seh code=0xC0000005") != std::string::npos &&
               code == 0xC0000005,
           "access violation -> SEH filter: error reply + stderr reason line, exit code 0xC0000005");
    std::printf("dsv41_producer_abort: %d PASS, %d FAIL\n", passed, failed);
    return failed ? 1 : 0;
}
