#include "ds4_win32.h"

#if defined(_WIN32)

#include <errno.h>
#include <ctype.h>
#include <limits.h>
#include <string.h>
#ifndef PSAPI_VERSION
#define PSAPI_VERSION 2
#endif
#include <psapi.h>

#undef open
#undef close
#undef fcntl
#undef pread
#undef dprintf
#undef ftruncate
#undef flock
#undef mkstemp
#undef clock_gettime
#undef nanosleep
#undef sleep
#undef usleep
#undef sysconf
#undef getpagesize
#undef mmap
#undef munmap
#undef mlock
#undef munlock
#undef msync
#undef madvise
#undef realpath
#undef pipe
#undef poll
#undef socket
#undef accept
#undef bind
#undef listen
#undef connect
#undef send
#undef recv
#undef shutdown
#undef setsockopt
#undef getsockopt
#undef inet_pton
#undef opendir
#undef readdir
#undef closedir
#undef fnmatch
#undef regcomp
#undef regexec
#undef regerror
#undef regfree
#undef ioctl
#undef fork
#undef waitpid
#undef kill
#undef setpgid
#undef execl

#define DS4_WIN32_SOCKET_BASE 0x40000000

#define DS4_HOST_RESERVE_BYTES (UINT64_C(1) << 30)
/* WDDM commitment is not physical residency. Match the owner's 1 GiB
 * available-RAM guard; retain OS commit headroom without a RAM+commit cap. */
static SRWLOCK g_host_memory_lock = SRWLOCK_INIT;
static LONG g_host_memory_ready;
static uint64_t g_host_memory_limit;
static HANDLE g_host_memory_job;
static HANDLE g_host_memory_port;

static uint64_t ds4_win32_host_commit_ceiling(const ds4_win32_host_memory_status *s) {
    return s->system_commit_limit_bytes > DS4_HOST_RESERVE_BYTES ?
        s->system_commit_limit_bytes - DS4_HOST_RESERVE_BYTES : 0;
}

bool ds4_win32_host_memory_read(ds4_win32_host_memory_status *out) {
    MEMORYSTATUSEX physical = {0};
    PERFORMANCE_INFORMATION performance = {0};
    PROCESS_MEMORY_COUNTERS_EX process = {0};
    if (!out) return false;
    memset(out, 0, sizeof(*out));
    physical.dwLength = sizeof(physical);
    performance.cb = sizeof(performance);
    process.cb = sizeof(process);
    if (!GlobalMemoryStatusEx(&physical) ||
        !GetPerformanceInfo(&performance, sizeof(performance)) ||
        !GetProcessMemoryInfo(GetCurrentProcess(),
            (PROCESS_MEMORY_COUNTERS *)&process, sizeof(process))) return false;
    out->physical_total_bytes = physical.ullTotalPhys;
    out->physical_available_bytes = physical.ullAvailPhys;
    out->system_commit_bytes = (uint64_t)performance.CommitTotal * performance.PageSize;
    out->system_commit_limit_bytes = (uint64_t)performance.CommitLimit * performance.PageSize;
    out->process_private_bytes = process.PrivateUsage;
    out->process_private_peak_bytes = process.PeakPagefileUsage;
    out->process_limit_bytes = g_host_memory_limit;
    out->reserve_bytes = DS4_HOST_RESERVE_BYTES;
    return true;
}

static void ds4_win32_host_memory_report(const char *event, const char *reason,
        uint64_t requested, const ds4_win32_host_memory_status *s) {
    /* A watchdog diagnostic must not wait on another thread's stdio lock. */
    char line[1024];
    const int n = snprintf(line, sizeof(line),
        "ds4: HOST_MEMORY_%s reason=%s requested_bytes=%llu "
        "private_bytes=%llu private_peak_bytes=%llu private_limit_bytes=%llu "
        "available_physical_bytes=%llu total_physical_bytes=%llu "
        "system_commit_bytes=%llu system_commit_limit_bytes=%llu system_commit_budget_bytes=%llu reserve_bytes=%llu\n",
        event, reason ? reason : "unspecified",
        (unsigned long long)requested,
        (unsigned long long)s->process_private_bytes,
        (unsigned long long)s->process_private_peak_bytes,
        (unsigned long long)s->process_limit_bytes,
        (unsigned long long)s->physical_available_bytes,
        (unsigned long long)s->physical_total_bytes,
        (unsigned long long)s->system_commit_bytes,
        (unsigned long long)s->system_commit_limit_bytes,
        (unsigned long long)ds4_win32_host_commit_ceiling(s),
        (unsigned long long)s->reserve_bytes);
    if (n > 0) {
        DWORD written;
        WriteFile(GetStdHandle(STD_ERROR_HANDLE), line,
            (DWORD)((size_t)n < sizeof(line) ? (size_t)n : sizeof(line) - 1u),
            &written, NULL);
    }
}

static const char *ds4_win32_host_memory_failure(uint64_t requested,
        const ds4_win32_host_memory_status *s, bool host_allocation) {
    if (!s) return "REFUSED_QUERY";
    const uint64_t ceiling = ds4_win32_host_commit_ceiling(s);
    const char *failure = NULL;
    if (s->process_private_bytes > s->process_limit_bytes ||
        requested > s->process_limit_bytes - s->process_private_bytes)
        failure = "REFUSED_PRIVATE_LIMIT";
    else if (s->physical_available_bytes < DS4_HOST_RESERVE_BYTES ||
        (host_allocation && requested > s->physical_available_bytes - DS4_HOST_RESERVE_BYTES))
        failure = "REFUSED_PHYSICAL_RESERVE";
    else if (s->system_commit_bytes > ceiling ||
        requested > ceiling - s->system_commit_bytes)
        failure = "REFUSED_SYSTEM_COMMIT_BUDGET";
    return failure;
}

static void ds4_win32_host_memory_job_events(void) {
    if (!g_host_memory_port) return;
    DWORD message;
    ULONG_PTR key;
    LPOVERLAPPED process;
    while (GetQueuedCompletionStatus(g_host_memory_port, &message, &key, &process, 0)) {
        if (message == JOB_OBJECT_MSG_PROCESS_MEMORY_LIMIT ||
            message == JOB_OBJECT_MSG_JOB_MEMORY_LIMIT) {
            ds4_win32_host_memory_status state;
            if (ds4_win32_host_memory_read(&state))
                ds4_win32_host_memory_report("JOB_LIMIT_NOTIFICATION",
                    message == JOB_OBJECT_MSG_PROCESS_MEMORY_LIMIT ?
                    "PROCESS_MEMORY_LIMIT" : "JOB_MEMORY_LIMIT", 0, &state);
        }
    }
    /* Windows does not guarantee every Job notification. Absence of a
     * message cannot prove an external allocator was not denied. */
}

bool ds4_win32_host_memory_fits_snapshot(const ds4_win32_host_memory_status *state,
        uint64_t requested_bytes, bool host_allocation) {
    return ds4_win32_host_memory_failure(requested_bytes, state, host_allocation) == NULL;
}

static bool ds4_win32_host_memory_fits(uint64_t requested,
        const char *reason, const ds4_win32_host_memory_status *s, bool host_allocation) {
    const char *failure = ds4_win32_host_memory_failure(requested, s, host_allocation);
    if (!failure) return true;
    ds4_win32_host_memory_report(failure, reason, requested, s);
    errno = ENOMEM;
    return false;
}

static DWORD WINAPI ds4_win32_host_memory_watchdog(void *unused) {
    (void)unused;
    for (;;) {
        Sleep(250);
        ds4_win32_host_memory_job_events();
        ds4_win32_host_memory_status state;
        if (!ds4_win32_host_memory_read(&state)) {
            memset(&state, 0, sizeof(state));
            ds4_win32_host_memory_report("WATCHDOG_QUERY_FAILED", "host watchdog", 0, &state);
            TerminateProcess(GetCurrentProcess(), 78);
            return 78;
        }
        if (!ds4_win32_host_memory_fits(0, "host watchdog", &state, false)) {
            ds4_win32_host_memory_report("WATCHDOG_EXIT", "host reserve breached", 0, &state);
            TerminateProcess(GetCurrentProcess(), 78);
            return 78;
        }
    }
}

bool ds4_win32_host_memory_begin(void) {
    bool ok = false;
    AcquireSRWLockExclusive(&g_host_memory_lock);
    if (g_host_memory_ready) {
        ok = g_host_memory_ready > 0;
        ReleaseSRWLockExclusive(&g_host_memory_lock);
        return ok;
    }
    g_host_memory_ready = -1;
    ds4_win32_host_memory_status state;
    if (!ds4_win32_host_memory_read(&state)) {
        fprintf(stderr, "ds4: HOST_MEMORY_REFUSED_QUERY: cannot measure Windows host memory (error=%lu)\n", GetLastError());
        goto done;
    }
    const uint64_t commit_ceiling = ds4_win32_host_commit_ceiling(&state);
    if (state.physical_total_bytes <= DS4_HOST_RESERVE_BYTES ||
        state.physical_available_bytes < DS4_HOST_RESERVE_BYTES ||
        state.system_commit_bytes >= commit_ceiling) {
        ds4_win32_host_memory_report("REFUSED_INITIAL_RESERVE", "startup", 0, &state);
        goto done;
    }
    /* Private commit can include VRAM backing. Bound it by actual remaining
     * OS commitment, not a second artificial physical-RAM limit. CPU/managed/
     * pinned requests separately admit against currently available RAM. */
    const uint64_t increment = commit_ceiling - state.system_commit_bytes;
    uint64_t cap = state.process_private_bytes > UINT64_MAX - increment ?
        UINT64_MAX : state.process_private_bytes + increment;
    const char *override = getenv("DS4_HOST_MEMORY_BUDGET_MB");
    if (override) {
        uint64_t mb = 0;
        const char *p = override;
        bool valid = *p != '\0';
        while (*p) {
            if (*p < '0' || *p > '9' || mb > (UINT64_MAX - (uint64_t)(*p - '0')) / 10u) {
                valid = false;
                break;
            }
            mb = mb * 10u + (uint64_t)(*p++ - '0');
        }
        if (!valid || !mb || mb > UINT64_MAX / UINT64_C(1048576) ||
            mb * UINT64_C(1048576) > cap) {
            fprintf(stderr, "ds4: HOST_MEMORY_REFUSED_BUDGET: DS4_HOST_MEMORY_BUDGET_MB must be a positive integer no larger than safe cap %llu bytes\n", (unsigned long long)cap);
            goto done;
        }
        cap = mb * UINT64_C(1048576);
    }
    state.process_limit_bytes = cap;
    if (cap <= state.process_private_bytes) {
        ds4_win32_host_memory_report("REFUSED_INITIAL_PRIVATE_LIMIT", "startup", 0, &state);
        goto done;
    }
    g_host_memory_job = CreateJobObjectW(NULL, NULL);
    g_host_memory_port = CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0, 1);
    JOBOBJECT_ASSOCIATE_COMPLETION_PORT association = {0};
    association.CompletionKey = g_host_memory_job;
    association.CompletionPort = g_host_memory_port;
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limit = {0};
    limit.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
    limit.ProcessMemoryLimit = (SIZE_T)cap;
    if (!g_host_memory_job || !g_host_memory_port ||
        !SetInformationJobObject(g_host_memory_job, JobObjectAssociateCompletionPortInformation,
            &association, sizeof(association)) ||
        !SetInformationJobObject(g_host_memory_job, JobObjectExtendedLimitInformation,
            &limit, sizeof(limit)) ||
        !AssignProcessToJobObject(g_host_memory_job, GetCurrentProcess())) {
        fprintf(stderr, "ds4: HOST_MEMORY_REFUSED_JOB: cannot enforce process private commit cap (error=%lu)\n", GetLastError());
        if (g_host_memory_job) CloseHandle(g_host_memory_job);
        if (g_host_memory_port) CloseHandle(g_host_memory_port);
        g_host_memory_job = NULL;
        g_host_memory_port = NULL;
        goto done;
    }
    /* Windows 8+ allows this dedicated Job nested inside Studio's owner Job.
     * Keep its handle for process lifetime, without KILL_ON_JOB_CLOSE. */
    g_host_memory_limit = cap;
    HANDLE watchdog = CreateThread(NULL, 64u * 1024u, ds4_win32_host_memory_watchdog,
        NULL, STACK_SIZE_PARAM_IS_A_RESERVATION, NULL);
    if (!watchdog) {
        fprintf(stderr, "ds4: HOST_MEMORY_REFUSED_WATCHDOG: cannot start host watchdog (error=%lu)\n", GetLastError());
        goto done;
    }
    CloseHandle(watchdog);
    g_host_memory_ready = 1;
    ds4_win32_host_memory_report("ADMITTED", "startup", 0, &state);
    ok = true;
done:
    ReleaseSRWLockExclusive(&g_host_memory_lock);
    return ok;
}

static bool ds4_win32_host_memory_check_kind(uint64_t requested_bytes,
        const char *reason, bool host_allocation) {
    if (!ds4_win32_host_memory_begin()) return false;
    ds4_win32_host_memory_status state;
    if (!ds4_win32_host_memory_read(&state)) {
        fprintf(stderr, "ds4: HOST_MEMORY_REFUSED_QUERY: cannot measure memory for %s (error=%lu)\n",
            reason ? reason : "allocation", GetLastError());
        return false;
    }
    return ds4_win32_host_memory_fits(requested_bytes, reason, &state, host_allocation);
}

bool ds4_win32_host_memory_check(uint64_t requested_bytes, const char *reason) {
    return ds4_win32_host_memory_check_kind(requested_bytes, reason, true);
}

bool ds4_win32_host_memory_check_commit(uint64_t requested_bytes, const char *reason) {
    return ds4_win32_host_memory_check_kind(requested_bytes, reason, false);
}

void ds4_win32_host_memory_snapshot(const char *reason) {
    ds4_win32_host_memory_job_events();
    ds4_win32_host_memory_status state;
    if (!ds4_win32_host_memory_read(&state)) {
        fprintf(stderr, "ds4: HOST_MEMORY_SNAPSHOT_QUERY_FAILED reason=%s error=%lu\n",
            reason ? reason : "unspecified", GetLastError());
        return;
    }
    ds4_win32_host_memory_report("SNAPSHOT", reason, 0, &state);
}

typedef struct {
    SOCKET s;
    bool nonblocking;
} ds4_win32_socket_slot;

static SRWLOCK g_socket_lock = SRWLOCK_INIT;
static ds4_win32_socket_slot *g_sockets;
static int g_sockets_cap;
static INIT_ONCE g_winsock_once = INIT_ONCE_STATIC_INIT;

void ds4_win32_enable_utf8_console(void) {
    if (GetConsoleWindow() == NULL) return;
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
}

static BOOL CALLBACK ds4_win32_winsock_init_once(PINIT_ONCE once, PVOID param, PVOID *ctx) {
    (void)once;
    (void)param;
    (void)ctx;
    WSADATA wsa;
    return WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
}

static bool ds4_win32_winsock_init(void) {
    return InitOnceExecuteOnce(&g_winsock_once,
                               ds4_win32_winsock_init_once,
                               NULL,
                               NULL) != 0;
}

static void ds4_win32_set_errno_from_wsa(void) {
    int e = WSAGetLastError();
    switch (e) {
    case WSAEWOULDBLOCK:
        errno = EAGAIN;
        break;
    case WSAEINTR:
        errno = EINTR;
        break;
    case WSAEADDRINUSE:
        errno = EADDRINUSE;
        break;
    case WSAECONNRESET:
    case WSAECONNABORTED:
        errno = ECONNRESET;
        break;
    case WSAETIMEDOUT:
        errno = ETIMEDOUT;
        break;
    case WSAEACCES:
        errno = EACCES;
        break;
    default:
        errno = EINVAL;
        break;
    }
}

static bool ds4_win32_socket_fd(int fd, SOCKET *out, ds4_win32_socket_slot **slot_out) {
    if (fd < DS4_WIN32_SOCKET_BASE) return false;
    int idx = fd - DS4_WIN32_SOCKET_BASE;
    bool ok = false;
    AcquireSRWLockShared(&g_socket_lock);
    if (idx >= 0 && idx < g_sockets_cap && g_sockets[idx].s != INVALID_SOCKET) {
        if (out) *out = g_sockets[idx].s;
        if (slot_out) *slot_out = &g_sockets[idx];
        ok = true;
    }
    ReleaseSRWLockShared(&g_socket_lock);
    return ok;
}

static int ds4_win32_socket_register(SOCKET s) {
    if (s == INVALID_SOCKET) {
        ds4_win32_set_errno_from_wsa();
        return -1;
    }

    AcquireSRWLockExclusive(&g_socket_lock);
    int idx = -1;
    for (int i = 0; i < g_sockets_cap; i++) {
        if (g_sockets[i].s == INVALID_SOCKET) {
            idx = i;
            break;
        }
    }
    if (idx < 0) {
        int old_cap = g_sockets_cap;
        int new_cap = old_cap ? old_cap * 2 : 64;
        ds4_win32_socket_slot *next =
            (ds4_win32_socket_slot *)realloc(g_sockets, (size_t)new_cap * sizeof(*next));
        if (!next) {
            ReleaseSRWLockExclusive(&g_socket_lock);
            closesocket(s);
            errno = ENOMEM;
            return -1;
        }
        g_sockets = next;
        for (int i = old_cap; i < new_cap; i++) {
            g_sockets[i].s = INVALID_SOCKET;
            g_sockets[i].nonblocking = false;
        }
        g_sockets_cap = new_cap;
        idx = old_cap;
    }
    g_sockets[idx].s = s;
    g_sockets[idx].nonblocking = false;
    ReleaseSRWLockExclusive(&g_socket_lock);
    return DS4_WIN32_SOCKET_BASE + idx;
}

typedef struct ds4_win32_mapping {
    void *addr;
    size_t length;
    HANDLE mapping;
    DWORD access;
    uint64_t offset;
    bool anonymous;
    struct ds4_win32_mapping *next;
} ds4_win32_mapping;

static SRWLOCK g_mapping_lock = SRWLOCK_INIT;
static ds4_win32_mapping *g_mappings;

static BOOL CALLBACK ds4_win32_init_freq(PINIT_ONCE once, PVOID param, PVOID *ctx) {
    (void)once;
    (void)ctx;
    QueryPerformanceFrequency((LARGE_INTEGER *)param);
    return TRUE;
}

static void ds4_win32_set_errno_from_last_error(void) {
    DWORD e = GetLastError();
    switch (e) {
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
        errno = ENOENT;
        break;
    case ERROR_ACCESS_DENIED:
    case ERROR_SHARING_VIOLATION:
    case ERROR_LOCK_VIOLATION:
        errno = EACCES;
        break;
    case ERROR_ALREADY_EXISTS:
    case ERROR_FILE_EXISTS:
        errno = EEXIST;
        break;
    case ERROR_NOT_ENOUGH_MEMORY:
    case ERROR_OUTOFMEMORY:
        errno = ENOMEM;
        break;
    case ERROR_INVALID_PARAMETER:
    default:
        errno = EINVAL;
        break;
    }
}

/* --- mutexes -------------------------------------------------------------
 *
 * pthread_mutex_t wraps a CRITICAL_SECTION (recursive) rather than an SRWLOCK
 * (not recursive); see the comment on the type in ds4_win32.h for the concrete
 * ds4-server re-entrancy path that requires this.
 *
 * A CRITICAL_SECTION cannot be statically initialized, so PTHREAD_MUTEX_INITIALIZER
 * only zero-fills and the object is created on first use.  Every explicit
 * pthread_mutex_init() still creates it eagerly, so the lazy path only ever runs
 * for file-scope mutexes (server_log_mu, the ds4.c async-load mutex, ...). */

/* An SRWLOCK spins before blocking; keep the same shape so replacing it with a
 * CRITICAL_SECTION does not add kernel transitions on short contention. */
#define DS4_WIN32_MUTEX_SPIN 4000

static void ds4_win32_mutex_ensure(pthread_mutex_t *mutex) {
    if (mutex->state == DS4_WIN32_MUTEX_READY) return;
    for (;;) {
        LONG prev = InterlockedCompareExchange(&mutex->state,
                                               DS4_WIN32_MUTEX_INITING,
                                               DS4_WIN32_MUTEX_UNINIT);
        if (prev == DS4_WIN32_MUTEX_READY) return;
        if (prev == DS4_WIN32_MUTEX_UNINIT) {
            InitializeCriticalSectionAndSpinCount(&mutex->cs,
                                                  DS4_WIN32_MUTEX_SPIN);
            InterlockedExchange(&mutex->state, DS4_WIN32_MUTEX_READY);
            return;
        }
        /* Another thread is inside InitializeCriticalSection right now. */
        SwitchToThread();
    }
}

int pthread_mutexattr_init(pthread_mutexattr_t *attr) {
    if (!attr) return EINVAL;
    attr->type = PTHREAD_MUTEX_DEFAULT;
    return 0;
}

int pthread_mutexattr_destroy(pthread_mutexattr_t *attr) {
    if (!attr) return EINVAL;
    attr->type = PTHREAD_MUTEX_DEFAULT;
    return 0;
}

/* Accepted and recorded, but not acted upon: the underlying CRITICAL_SECTION is
 * unconditionally recursive.  PTHREAD_MUTEX_RECURSIVE therefore behaves exactly
 * as requested; NORMAL/ERRORCHECK are silently upgraded to recursive, which can
 * only mask a would-be self-deadlock, never introduce one. */
int pthread_mutexattr_settype(pthread_mutexattr_t *attr, int type) {
    if (!attr) return EINVAL;
    if (type != PTHREAD_MUTEX_NORMAL && type != PTHREAD_MUTEX_RECURSIVE &&
        type != PTHREAD_MUTEX_ERRORCHECK) {
        return EINVAL;
    }
    attr->type = type;
    return 0;
}

int pthread_mutexattr_gettype(const pthread_mutexattr_t *attr, int *type) {
    if (!attr || !type) return EINVAL;
    *type = attr->type;
    return 0;
}

int pthread_mutex_init(pthread_mutex_t *mutex, const void *attr) {
    (void)attr;
    if (!mutex) return EINVAL;
    /* Do not inspect mutex->state here: callers may hand us uninitialized
     * storage.  Create unconditionally and publish the ready state. */
    InitializeCriticalSectionAndSpinCount(&mutex->cs, DS4_WIN32_MUTEX_SPIN);
    InterlockedExchange(&mutex->state, DS4_WIN32_MUTEX_READY);
    return 0;
}

int pthread_mutex_destroy(pthread_mutex_t *mutex) {
    if (!mutex) return EINVAL;
    if (InterlockedExchange(&mutex->state, DS4_WIN32_MUTEX_UNINIT) ==
        DS4_WIN32_MUTEX_READY) {
        DeleteCriticalSection(&mutex->cs);
    }
    return 0;
}

int pthread_mutex_lock(pthread_mutex_t *mutex) {
    if (!mutex) return EINVAL;
    ds4_win32_mutex_ensure(mutex);
    EnterCriticalSection(&mutex->cs);
    return 0;
}

int pthread_mutex_trylock(pthread_mutex_t *mutex) {
    if (!mutex) return EINVAL;
    ds4_win32_mutex_ensure(mutex);
    return TryEnterCriticalSection(&mutex->cs) ? 0 : EBUSY;
}

int pthread_mutex_unlock(pthread_mutex_t *mutex) {
    if (!mutex) return EINVAL;
    /* Deliberately no lazy init here: leaving a section this thread never
     * entered corrupts the CRITICAL_SECTION.  Unlocking a never-locked mutex is
     * a caller bug; report it instead of creating one to leave. */
    if (mutex->state != DS4_WIN32_MUTEX_READY) return EPERM;
    LeaveCriticalSection(&mutex->cs);
    return 0;
}

int pthread_cond_init(pthread_cond_t *cond, const void *attr) {
    (void)attr;
    InitializeConditionVariable(cond);
    return 0;
}

int pthread_cond_destroy(pthread_cond_t *cond) {
    (void)cond;
    return 0;
}

/* CRITICAL_SECTION-backed mutexes require SleepConditionVariableCS, not the
 * ...SRW variant.  Note this releases the section exactly once, mirroring
 * POSIX: waiting on a *recursively* held mutex would not release it.  No
 * condition variable in this tree is paired with a recursively-held mutex
 * (s->inference_mu, the only re-entered one, has no condvar). */
int pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex) {
    if (!cond || !mutex) return EINVAL;
    ds4_win32_mutex_ensure(mutex);
    return SleepConditionVariableCS(cond, &mutex->cs, INFINITE) ? 0 : EINVAL;
}

/* `abstime` is an absolute CLOCK_REALTIME deadline (POSIX); Windows wants a
 * relative timeout, so convert against the same clock ds4_win32_clock_gettime()
 * reports for CLOCK_REALTIME. */
int pthread_cond_timedwait(pthread_cond_t *cond, pthread_mutex_t *mutex,
                           const struct timespec *abstime) {
    if (!cond || !mutex) return EINVAL;
    ds4_win32_mutex_ensure(mutex);

    DWORD wait_ms = INFINITE;
    if (abstime) {
        struct timespec now;
        if (ds4_win32_clock_gettime(CLOCK_REALTIME, &now) != 0) return EINVAL;
        int64_t delta_ms =
            ((int64_t)abstime->tv_sec - (int64_t)now.tv_sec) * 1000 +
            ((int64_t)abstime->tv_nsec - (int64_t)now.tv_nsec) / 1000000;
        if (delta_ms <= 0) {
            wait_ms = 0;
        } else if (delta_ms >= (int64_t)(INFINITE - 1)) {
            wait_ms = INFINITE - 1;
        } else {
            wait_ms = (DWORD)delta_ms;
        }
    }

    if (SleepConditionVariableCS(cond, &mutex->cs, wait_ms)) return 0;
    return GetLastError() == ERROR_TIMEOUT ? ETIMEDOUT : EINVAL;
}

int pthread_cond_signal(pthread_cond_t *cond) {
    WakeConditionVariable(cond);
    return 0;
}

int pthread_cond_broadcast(pthread_cond_t *cond) {
    WakeAllConditionVariable(cond);
    return 0;
}

typedef struct {
    void *(*fn)(void *);
    void *arg;
} ds4_win32_thread_start;

static unsigned __stdcall ds4_win32_thread_main(void *arg) {
    ds4_win32_thread_start *start = (ds4_win32_thread_start *)arg;
    void *(*fn)(void *) = start->fn;
    void *fn_arg = start->arg;
    free(start);
    (void)fn(fn_arg);
    return 0;
}

int pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                   void *(*start_routine)(void *), void *arg) {
    (void)attr;
    if (!thread || !start_routine) return EINVAL;
    ds4_win32_thread_start *start =
        (ds4_win32_thread_start *)malloc(sizeof(*start));
    if (!start) return ENOMEM;
    start->fn = start_routine;
    start->arg = arg;
    uintptr_t h = _beginthreadex(NULL, 0, ds4_win32_thread_main, start, 0, NULL);
    if (h == 0) {
        free(start);
        return errno ? errno : EAGAIN;
    }
    *thread = (HANDLE)h;
    return 0;
}

int pthread_join(pthread_t thread, void **retval) {
    if (retval) *retval = NULL;
    if (!thread) return EINVAL;
    DWORD wr = WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return wr == WAIT_OBJECT_0 ? 0 : EINVAL;
}

int pthread_detach(pthread_t thread) {
    if (!thread) return EINVAL;
    CloseHandle(thread);
    return 0;
}

static BOOL CALLBACK ds4_win32_once_cb(PINIT_ONCE once, PVOID param, PVOID *ctx) {
    (void)once;
    (void)ctx;
    ((void (*)(void))param)();
    return TRUE;
}

int pthread_once(pthread_once_t *once_control, void (*init_routine)(void)) {
    if (!once_control || !init_routine) return EINVAL;
    return InitOnceExecuteOnce(once_control,
                               ds4_win32_once_cb,
                               (PVOID)init_routine,
                               NULL) ? 0 : EINVAL;
}

int sigemptyset(sigset_t *set) {
    if (set) *set = 0;
    return 0;
}

int sigaction(int signum, const struct sigaction *act, struct sigaction *oldact) {
    void (*old_handler)(int) = signal(signum, act ? act->sa_handler : SIG_DFL);
    if (old_handler == SIG_ERR) return -1;
    if (oldact) {
        oldact->sa_handler = old_handler;
        oldact->sa_mask = 0;
        oldact->sa_flags = 0;
    }
    return 0;
}

int ds4_win32_open(const char *path, int flags, ...) {
    int mode = 0;
    if (flags & _O_CREAT) {
        va_list ap;
        va_start(ap, flags);
        mode = va_arg(ap, int);
        va_end(ap);
    }
    if (path && strcmp(path, "/dev/null") == 0) path = "NUL";
    if (flags & O_NOFOLLOW) {
        if (!path || (flags & (_O_CREAT | _O_TRUNC | _O_APPEND))) {
            errno = EINVAL;
            return -1;
        }
        DWORD access = flags & _O_RDWR ? GENERIC_READ | GENERIC_WRITE :
                       flags & _O_WRONLY ? GENERIC_WRITE : GENERIC_READ;
        HANDLE file = CreateFileA(path, access,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
                OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, NULL);
        if (file == INVALID_HANDLE_VALUE) {
            ds4_win32_set_errno_from_last_error();
            return -1;
        }
        BY_HANDLE_FILE_INFORMATION info = {0};
        if (!GetFileInformationByHandle(file, &info) ||
            (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) {
            DWORD error = GetLastError();
            bool reparse = (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
            CloseHandle(file);
            if (reparse) errno = EACCES;
            else { SetLastError(error); ds4_win32_set_errno_from_last_error(); }
            return -1;
        }
        int fd = _open_osfhandle((intptr_t)file,
                                (flags & (_O_RDWR | _O_WRONLY | _O_NOINHERIT)) | _O_BINARY);
        if (fd < 0) CloseHandle(file);
        return fd;
    }
    return _open(path, flags | _O_BINARY, mode);
}

int ds4_win32_close(int fd) {
    if (fd >= DS4_WIN32_SOCKET_BASE) {
        int idx = fd - DS4_WIN32_SOCKET_BASE;
        SOCKET s = INVALID_SOCKET;
        AcquireSRWLockExclusive(&g_socket_lock);
        if (idx >= 0 && idx < g_sockets_cap) {
            s = g_sockets[idx].s;
            g_sockets[idx].s = INVALID_SOCKET;
            g_sockets[idx].nonblocking = false;
        }
        ReleaseSRWLockExclusive(&g_socket_lock);
        if (s == INVALID_SOCKET) {
            errno = EBADF;
            return -1;
        }
        if (closesocket(s) == 0) return 0;
        ds4_win32_set_errno_from_wsa();
        return -1;
    }
    return _close(fd);
}

int ds4_win32_fcntl(int fd, int cmd, ...) {
    SOCKET s;
    if (ds4_win32_socket_fd(fd, &s, NULL)) {
        if (cmd == F_GETFL) {
            bool nb = false;
            AcquireSRWLockShared(&g_socket_lock);
            int idx = fd - DS4_WIN32_SOCKET_BASE;
            if (idx >= 0 && idx < g_sockets_cap) nb = g_sockets[idx].nonblocking;
            ReleaseSRWLockShared(&g_socket_lock);
            return nb ? O_NONBLOCK : 0;
        }
        if (cmd == F_SETFL) {
            va_list ap;
            va_start(ap, cmd);
            int flags = va_arg(ap, int);
            va_end(ap);
            u_long mode = (flags & O_NONBLOCK) ? 1u : 0u;
            if (ioctlsocket(s, FIONBIO, &mode) != 0) {
                ds4_win32_set_errno_from_wsa();
                return -1;
            }
            AcquireSRWLockExclusive(&g_socket_lock);
            int idx = fd - DS4_WIN32_SOCKET_BASE;
            if (idx >= 0 && idx < g_sockets_cap)
                g_sockets[idx].nonblocking = mode != 0;
            ReleaseSRWLockExclusive(&g_socket_lock);
            return 0;
        }
    }
    if (cmd == F_GETFL) return 0;
    if (cmd == F_SETFL) return 0;
    return 0;
}

ssize_t ds4_win32_pread(int fd, void *buf, size_t count, int64_t offset) {
    if (offset < 0) {
        errno = EINVAL;
        return -1;
    }
    intptr_t osfh = _get_osfhandle(fd);
    if (osfh == -1) {
        errno = EBADF;
        return -1;
    }
    if (count > 0xffffffffu) count = 0xffffffffu;

    OVERLAPPED ov;
    memset(&ov, 0, sizeof(ov));
    ov.Offset = (DWORD)((uint64_t)offset & 0xffffffffu);
    ov.OffsetHigh = (DWORD)(((uint64_t)offset >> 32) & 0xffffffffu);

    DWORD got = 0;
    if (ReadFile((HANDLE)osfh, buf, (DWORD)count, &got, &ov)) {
        return (ssize_t)got;
    }
    if (GetLastError() == ERROR_HANDLE_EOF) return 0;
    ds4_win32_set_errno_from_last_error();
    return -1;
}

int ds4_win32_dprintf(int fd, const char *fmt, ...) {
    char stack[256];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(stack, sizeof(stack), fmt, ap);
    va_end(ap);
    if (n < 0) return -1;
    if ((size_t)n < sizeof(stack)) {
        return _write(fd, stack, (unsigned int)n) == n ? n : -1;
    }
    char *buf = (char *)malloc((size_t)n + 1u);
    if (!buf) {
        errno = ENOMEM;
        return -1;
    }
    va_start(ap, fmt);
    (void)vsnprintf(buf, (size_t)n + 1u, fmt, ap);
    va_end(ap);
    int wr = _write(fd, buf, (unsigned int)n);
    free(buf);
    return wr == n ? n : -1;
}

int ds4_win32_ftruncate(int fd, int64_t length) {
    return _chsize_s(fd, length) == 0 ? 0 : -1;
}

int ds4_win32_flock(int fd, int operation) {
    intptr_t osfh = _get_osfhandle(fd);
    if (osfh == -1) {
        errno = EBADF;
        return -1;
    }

    HANDLE h = (HANDLE)osfh;
    OVERLAPPED ov;
    memset(&ov, 0, sizeof(ov));
    if (operation & LOCK_UN) {
        if (UnlockFileEx(h, 0, MAXDWORD, MAXDWORD, &ov)) return 0;
        ds4_win32_set_errno_from_last_error();
        return -1;
    }

    DWORD flags = 0;
    if (operation & LOCK_EX) flags |= LOCKFILE_EXCLUSIVE_LOCK;
    if (operation & LOCK_NB) flags |= LOCKFILE_FAIL_IMMEDIATELY;
    if (LockFileEx(h, flags, 0, MAXDWORD, MAXDWORD, &ov)) return 0;
    ds4_win32_set_errno_from_last_error();
    return -1;
}

int ds4_win32_mkstemp(char *tmpl) {
    if (!tmpl) {
        errno = EINVAL;
        return -1;
    }

    char *base = strrchr(tmpl, '/');
    if (!base) base = strrchr(tmpl, '\\');
    base = base ? base + 1 : tmpl;

    char *slot = strstr(base, "XXXXXX");
    if (!slot) {
        errno = EINVAL;
        return -1;
    }
    const size_t prefix_len = (size_t)(slot - base);

    char path[MAX_PATH];
    DWORD pid = GetCurrentProcessId();
    for (unsigned i = 0; i < 1000; i++) {
        snprintf(path,
                 sizeof(path),
                 ".\\%.*s%06x",
                 (int)prefix_len,
                 base,
                 (unsigned)((pid + i) & 0xffffffu));
        int fd = _open(path,
                       _O_RDWR | _O_CREAT | _O_EXCL | _O_BINARY,
                       _S_IREAD | _S_IWRITE);
        if (fd >= 0) {
            strcpy(tmpl, path);
            return fd;
        }
        if (errno != EEXIST) return -1;
    }
    errno = EEXIST;
    return -1;
}

int ds4_win32_clock_gettime(int clock_id, struct timespec *ts) {
    if (!ts) {
        errno = EINVAL;
        return -1;
    }
    if (clock_id == CLOCK_MONOTONIC) {
        static LARGE_INTEGER freq;
        static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
        (void)InitOnceExecuteOnce(&once, ds4_win32_init_freq, &freq, NULL);
        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        ts->tv_sec = (time_t)(now.QuadPart / freq.QuadPart);
        ts->tv_nsec = (long)(((now.QuadPart % freq.QuadPart) * 1000000000LL) /
                             freq.QuadPart);
        return 0;
    }

    FILETIME ft;
    GetSystemTimePreciseAsFileTime(&ft);
    uint64_t t = ((uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    t -= UINT64_C(116444736000000000);
    ts->tv_sec = (time_t)(t / 10000000u);
    ts->tv_nsec = (long)((t % 10000000u) * 100u);
    return 0;
}

int ds4_win32_nanosleep(const struct timespec *req, struct timespec *rem) {
    if (!req) {
        errno = EINVAL;
        return -1;
    }
    if (rem) {
        rem->tv_sec = 0;
        rem->tv_nsec = 0;
    }
    uint64_t ms = (uint64_t)req->tv_sec * 1000u;
    ms += (uint64_t)(req->tv_nsec + 999999L) / 1000000u;
    if (ms > MAXDWORD) ms = MAXDWORD;
    Sleep((DWORD)ms);
    return 0;
}

unsigned int ds4_win32_sleep(unsigned int seconds) {
    Sleep(seconds * 1000u);
    return 0;
}

int ds4_win32_usleep(unsigned int usec) {
    Sleep((usec + 999u) / 1000u);
    return 0;
}

int ds4_win32_gettimeofday(struct timeval *tv, void *tz) {
    (void)tz;
    if (!tv) {
        errno = EINVAL;
        return -1;
    }
    FILETIME ft;
    GetSystemTimePreciseAsFileTime(&ft);
    uint64_t t = ((uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    t -= UINT64_C(116444736000000000);
    tv->tv_sec = (long)(t / 10000000u);
    tv->tv_usec = (long)((t % 10000000u) / 10u);
    return 0;
}

struct tm *ds4_win32_localtime_r(const time_t *timep, struct tm *result) {
    if (!timep || !result) return NULL;
    return localtime_s(result, timep) == 0 ? result : NULL;
}

long ds4_win32_sysconf(int name) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    switch (name) {
    case _SC_PAGESIZE:
        return (long)si.dwPageSize;
    case _SC_NPROCESSORS_ONLN:
        return (long)si.dwNumberOfProcessors;
    default:
        errno = EINVAL;
        return -1;
    }
}

int ds4_win32_getpagesize(void) {
    long page = ds4_win32_sysconf(_SC_PAGESIZE);
    return page > 0 ? (int)page : 4096;
}

static bool ds4_win32_mapping_add(void *addr, size_t length,
                                  HANDLE mapping, bool anonymous,
                                  DWORD access, uint64_t offset) {
    ds4_win32_mapping *m = (ds4_win32_mapping *)malloc(sizeof(*m));
    if (!m) return false;
    m->addr = addr;
    m->length = length;
    m->mapping = mapping;
    m->access = access;
    m->offset = offset;
    m->anonymous = anonymous;
    AcquireSRWLockExclusive(&g_mapping_lock);
    m->next = g_mappings;
    g_mappings = m;
    ReleaseSRWLockExclusive(&g_mapping_lock);
    return true;
}

void *ds4_win32_mmap(void *addr, size_t length, int prot, int flags,
                     int fd, int64_t offset) {
    (void)addr;
    if (length == 0) {
        errno = EINVAL;
        return MAP_FAILED;
    }

    if (flags & MAP_ANONYMOUS) {
        DWORD protect = (prot & PROT_WRITE) ? PAGE_READWRITE : PAGE_READONLY;
        void *p = VirtualAlloc(NULL, length, MEM_RESERVE | MEM_COMMIT, protect);
        if (!p) {
            ds4_win32_set_errno_from_last_error();
            return MAP_FAILED;
        }
        if (!ds4_win32_mapping_add(p, length, NULL, true, 0, 0)) {
            VirtualFree(p, 0, MEM_RELEASE);
            errno = ENOMEM;
            return MAP_FAILED;
        }
        return p;
    }

    intptr_t osfh = _get_osfhandle(fd);
    if (osfh == -1) {
        errno = EBADF;
        return MAP_FAILED;
    }

    DWORD protect = (prot & PROT_WRITE) ? PAGE_READWRITE : PAGE_READONLY;
    DWORD access = (prot & PROT_WRITE) ? FILE_MAP_WRITE : FILE_MAP_READ;
    HANDLE mapping = CreateFileMapping((HANDLE)osfh, NULL, protect, 0, 0, NULL);
    if (!mapping) {
        ds4_win32_set_errno_from_last_error();
        return MAP_FAILED;
    }

    DWORD off_hi = (DWORD)(((uint64_t)offset) >> 32);
    DWORD off_lo = (DWORD)(((uint64_t)offset) & 0xffffffffu);
    void *p = MapViewOfFile(mapping, access, off_hi, off_lo, length);
    if (!p) {
        ds4_win32_set_errno_from_last_error();
        CloseHandle(mapping);
        return MAP_FAILED;
    }

    if (!ds4_win32_mapping_add(p, length, mapping, false, access, (uint64_t)offset)) {
        UnmapViewOfFile(p);
        CloseHandle(mapping);
        errno = ENOMEM;
        return MAP_FAILED;
    }
    return p;
}

int ds4_win32_munmap(void *addr, size_t length) {
    if (!addr || length == 0) {
        errno = EINVAL;
        return -1;
    }

    AcquireSRWLockExclusive(&g_mapping_lock);
    ds4_win32_mapping **link = &g_mappings;
    while (*link) {
        const uintptr_t base = (uintptr_t)(*link)->addr;
        const uintptr_t requested = (uintptr_t)addr;
        if (requested >= base && requested - base < (*link)->length) break;
        link = &(*link)->next;
    }
    ds4_win32_mapping *m = *link;
    if (!m) {
        ReleaseSRWLockExclusive(&g_mapping_lock);
        errno = EINVAL;
        return -1;
    }

    const size_t prefix = (size_t)((uintptr_t)addr - (uintptr_t)m->addr);
    if (prefix) {
        /* Windows cannot unmap an interior part of a view. Engram and Qwen
         * remove only a trailing disk-only table, during model startup before
         * readers run. Remap the retained prefix at the SAME base so parsed
         * GGUF names and tensor pointers stay valid. Never accept relocation. */
        if (m->anonymous || length != m->length - prefix ||
            prefix % (size_t)ds4_win32_getpagesize() != 0) {
            ReleaseSRWLockExclusive(&g_mapping_lock);
            errno = EINVAL;
            return -1;
        }
        if (!UnmapViewOfFile(m->addr)) {
            DWORD error = GetLastError();
            ReleaseSRWLockExclusive(&g_mapping_lock);
            SetLastError(error);
            ds4_win32_set_errno_from_last_error();
            return -1;
        }
        void *replacement = MapViewOfFileEx(m->mapping, m->access,
                (DWORD)(m->offset >> 32), (DWORD)m->offset, prefix, m->addr);
        if (replacement == m->addr) {
            m->length = prefix;
            ReleaseSRWLockExclusive(&g_mapping_lock);
            return 0;
        }
        /* The caller must abort startup: its parsed pointers are invalid.
         * Remove the dead registration so cleanup cannot unmap a reused VA. */
        DWORD error = replacement ? ERROR_INVALID_ADDRESS : GetLastError();
        if (replacement) UnmapViewOfFile(replacement);
        *link = m->next;
        CloseHandle(m->mapping);
        free(m);
        ReleaseSRWLockExclusive(&g_mapping_lock);
        SetLastError(error);
        ds4_win32_set_errno_from_last_error();
        return -1;
    }

    BOOL ok = m->anonymous ? VirtualFree(m->addr, 0, MEM_RELEASE) :
                            UnmapViewOfFile(m->addr);
    if (!ok) {
        DWORD error = GetLastError();
        ReleaseSRWLockExclusive(&g_mapping_lock);
        SetLastError(error);
        ds4_win32_set_errno_from_last_error();
        return -1;
    }
    *link = m->next;
    if (m->mapping) CloseHandle(m->mapping);
    free(m);
    ReleaseSRWLockExclusive(&g_mapping_lock);
    return 0;
}

int ds4_win32_mlock(const void *addr, size_t length) {
    if (VirtualLock((LPVOID)addr, length)) return 0;
    ds4_win32_set_errno_from_last_error();
    return -1;
}

int ds4_win32_munlock(const void *addr, size_t length) {
    if (VirtualUnlock((LPVOID)addr, length)) return 0;
    ds4_win32_set_errno_from_last_error();
    return -1;
}

int ds4_win32_msync(void *addr, size_t length, int flags) {
    (void)flags;
    if (FlushViewOfFile(addr, length)) return 0;
    ds4_win32_set_errno_from_last_error();
    return -1;
}

/* Trim only an existing registered file view, holding off unmap/remap until
 * the hint completes. VirtualUnlock on unlocked pages removes their working-
 * set entries and returns ERROR_NOT_LOCKED; it does not discard mapped bytes.
 * https://learn.microsoft.com/windows/win32/api/memoryapi/nf-memoryapi-virtualunlock
 * This is deliberately separate from POSIX MADV_DONTNEED semantics. */
int ds4_win32_trim_mapped_pages(const void *addr, size_t length) {
    const uintptr_t requested = (uintptr_t)addr;
    if (!addr || !length || length > UINTPTR_MAX - requested) {
        errno = EINVAL;
        return -1;
    }
    const uintptr_t end = requested + length;
    AcquireSRWLockShared(&g_mapping_lock);
    ds4_win32_mapping *m = g_mappings;
    while (m) {
        const uintptr_t base = (uintptr_t)m->addr;
        if (requested >= base && requested - base < m->length &&
            length <= m->length - (size_t)(requested - base)) break;
        m = m->next;
    }
    if (!m || m->anonymous || !m->mapping) {
        ReleaseSRWLockShared(&g_mapping_lock);
        errno = EINVAL;
        return -1;
    }
    /* Refuse stale/private/uncommitted regions before applying any hint. */
    for (uintptr_t cursor = requested; cursor < end;) {
        MEMORY_BASIC_INFORMATION info;
        if (!VirtualQuery((const void *)cursor, &info, sizeof(info)) ||
            info.State != MEM_COMMIT || info.Type != MEM_MAPPED ||
            info.AllocationBase != m->addr || !info.RegionSize ||
            (uintptr_t)info.BaseAddress > cursor ||
            info.RegionSize > UINTPTR_MAX - (uintptr_t)info.BaseAddress) {
            ReleaseSRWLockShared(&g_mapping_lock);
            errno = EINVAL;
            return -1;
        }
        const uintptr_t next = (uintptr_t)info.BaseAddress + info.RegionSize;
        if (next <= cursor) {
            ReleaseSRWLockShared(&g_mapping_lock);
            errno = EINVAL;
            return -1;
        }
        cursor = next < end ? next : end;
    }
    const BOOL ok = VirtualUnlock((LPVOID)addr, length);
    const DWORD error = ok ? ERROR_SUCCESS : GetLastError();
    ReleaseSRWLockShared(&g_mapping_lock);
    if (ok || error == ERROR_NOT_LOCKED) return 0;
    SetLastError(error);
    ds4_win32_set_errno_from_last_error();
    return -1;
}

/* There is no safe per-view discard operation for file mappings on all
 * supported Windows versions. Report unsupported advice without discarding
 * mapped data; callers use MADV_DONTNEED as an optional residency hint. */
int ds4_win32_madvise(void *addr, size_t length, int advice) {
    if (!addr || !length || advice != MADV_DONTNEED) {
        errno = EINVAL;
        return -1;
    }
    errno = ENOSYS;
    return -1;
}

char *ds4_win32_realpath(const char *path, char *resolved) {
    if (!path) { errno = EINVAL; return NULL; }
    HANDLE file = CreateFileA(path, FILE_READ_ATTRIBUTES,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
            OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);
    if (file == INVALID_HANDLE_VALUE) {
        ds4_win32_set_errno_from_last_error();
        return NULL;
    }
    DWORD needed = GetFinalPathNameByHandleA(file, NULL, 0, FILE_NAME_NORMALIZED);
    if (!needed) {
        DWORD error = GetLastError();
        CloseHandle(file);
        SetLastError(error);
        ds4_win32_set_errno_from_last_error();
        return NULL;
    }
    if (resolved && needed >= PATH_MAX) {
        CloseHandle(file);
        errno = ENAMETOOLONG;
        return NULL;
    }
    char *absolute = resolved ? resolved : malloc((size_t)needed + 1u);
    if (!absolute) { CloseHandle(file); errno = ENOMEM; return NULL; }
    DWORD got = GetFinalPathNameByHandleA(file, absolute, needed + 1u, FILE_NAME_NORMALIZED);
    DWORD error = GetLastError();
    CloseHandle(file);
    if (!got || got > needed) {
        if (!resolved) free(absolute);
        SetLastError(error);
        ds4_win32_set_errno_from_last_error();
        return NULL;
    }
    return absolute;
}

int ds4_win32_pipe(int fds[2]) {
    if (!fds) {
        errno = EINVAL;
        return -1;
    }
    return _pipe(fds, 65536, _O_BINARY);
}

static bool ds4_win32_fd_read_ready(int fd) {
    intptr_t osfh = _get_osfhandle(fd);
    if (osfh == -1) return true;
    HANDLE h = (HANDLE)osfh;
    DWORD type = GetFileType(h);
    if (type == FILE_TYPE_PIPE) {
        DWORD avail = 0;
        if (PeekNamedPipe(h, NULL, 0, NULL, &avail, NULL)) return avail > 0;
        return true;
    }
    if (fd == STDIN_FILENO) {
        DWORD mode = 0;
        if (!GetConsoleMode(h, &mode)) return true;
        DWORD events = 0;
        if (GetNumberOfConsoleInputEvents(h, &events)) return events > 0;
    }
    return true;
}

int ds4_win32_poll(struct pollfd *fds, nfds_t nfds, int timeout_ms) {
    if (!fds && nfds) {
        errno = EINVAL;
        return -1;
    }
    double start = 0.0;
    struct timespec ts;
    if (timeout_ms > 0) {
        ds4_win32_clock_gettime(CLOCK_MONOTONIC, &ts);
        start = (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
    }

    for (;;) {
        int ready = 0;
        for (nfds_t i = 0; i < nfds; i++) {
            fds[i].revents = 0;
            SOCKET s;
            if (ds4_win32_socket_fd(fds[i].fd, &s, NULL)) {
                fd_set rfds;
                fd_set wfds;
                fd_set efds;
                FD_ZERO(&rfds);
                FD_ZERO(&wfds);
                FD_ZERO(&efds);
                if (fds[i].events & POLLIN) FD_SET(s, &rfds);
                if (fds[i].events & POLLOUT) FD_SET(s, &wfds);
                FD_SET(s, &efds);
                struct timeval tv = {0, 0};
                int rc = select(0, &rfds, &wfds, &efds, &tv);
                if (rc == SOCKET_ERROR) {
                    fds[i].revents |= POLLERR;
                    ready++;
                } else {
                    if (FD_ISSET(s, &rfds)) fds[i].revents |= POLLIN;
                    if (FD_ISSET(s, &wfds)) fds[i].revents |= POLLOUT;
                    if (FD_ISSET(s, &efds)) fds[i].revents |= POLLERR;
                    if (fds[i].revents) ready++;
                }
            } else {
                if ((fds[i].events & POLLIN) && ds4_win32_fd_read_ready(fds[i].fd)) {
                    fds[i].revents |= POLLIN;
                    ready++;
                } else if (fds[i].events & POLLOUT) {
                    fds[i].revents |= POLLOUT;
                    ready++;
                }
            }
        }
        if (ready || timeout_ms == 0) return ready;
        if (timeout_ms > 0) {
            ds4_win32_clock_gettime(CLOCK_MONOTONIC, &ts);
            double now = (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
            if ((now - start) * 1000.0 >= timeout_ms) return 0;
        }
        Sleep(timeout_ms < 0 ? 10 : 1);
    }
}

int ds4_win32_socket(int af, int type, int protocol) {
    if (!ds4_win32_winsock_init()) {
        errno = EINVAL;
        return -1;
    }
    SOCKET s = WSASocketA(af, type, protocol, NULL, 0, 0);
    return ds4_win32_socket_register(s);
}

int ds4_win32_accept(int fd, struct sockaddr *addr, socklen_t *addrlen) {
    SOCKET s;
    if (!ds4_win32_socket_fd(fd, &s, NULL)) {
        errno = EBADF;
        return -1;
    }
    int alen = addrlen ? *addrlen : 0;
    SOCKET a = WSAAccept(s, addr, addrlen ? &alen : NULL, NULL, 0);
    if (addrlen) *addrlen = alen;
    return ds4_win32_socket_register(a);
}

int ds4_win32_bind(int fd, const struct sockaddr *addr, socklen_t addrlen) {
    SOCKET s;
    if (!ds4_win32_socket_fd(fd, &s, NULL)) {
        errno = EBADF;
        return -1;
    }
    if (bind(s, addr, addrlen) == 0) return 0;
    ds4_win32_set_errno_from_wsa();
    return -1;
}

int ds4_win32_listen(int fd, int backlog) {
    SOCKET s;
    if (!ds4_win32_socket_fd(fd, &s, NULL)) {
        errno = EBADF;
        return -1;
    }
    if (listen(s, backlog) == 0) return 0;
    ds4_win32_set_errno_from_wsa();
    return -1;
}

int ds4_win32_connect(int fd, const struct sockaddr *addr, socklen_t addrlen) {
    SOCKET s;
    if (!ds4_win32_socket_fd(fd, &s, NULL)) {
        errno = EBADF;
        return -1;
    }
    if (connect(s, addr, addrlen) == 0) return 0;
    ds4_win32_set_errno_from_wsa();
    return -1;
}

ssize_t ds4_win32_send(int fd, const void *buf, size_t len, int flags) {
    SOCKET s;
    if (!ds4_win32_socket_fd(fd, &s, NULL)) {
        errno = EBADF;
        return -1;
    }
    if (len > INT_MAX) len = INT_MAX;
    int rc = send(s, (const char *)buf, (int)len, flags);
    if (rc >= 0) return (ssize_t)rc;
    ds4_win32_set_errno_from_wsa();
    return -1;
}

ssize_t ds4_win32_recv(int fd, void *buf, size_t len, int flags) {
    SOCKET s;
    if (!ds4_win32_socket_fd(fd, &s, NULL)) {
        errno = EBADF;
        return -1;
    }
    if (len > INT_MAX) len = INT_MAX;
    int rc = recv(s, (char *)buf, (int)len, flags);
    if (rc >= 0) return (ssize_t)rc;
    ds4_win32_set_errno_from_wsa();
    return -1;
}

int ds4_win32_shutdown(int fd, int how) {
    SOCKET s;
    if (!ds4_win32_socket_fd(fd, &s, NULL)) {
        errno = EBADF;
        return -1;
    }
    if (shutdown(s, how) == 0) return 0;
    ds4_win32_set_errno_from_wsa();
    return -1;
}

int ds4_win32_setsockopt(int fd, int level, int optname,
                         const void *optval, socklen_t optlen) {
    SOCKET s;
    if (!ds4_win32_socket_fd(fd, &s, NULL)) {
        errno = EBADF;
        return -1;
    }
    DWORD timeout_ms;
    if ((optname == SO_RCVTIMEO || optname == SO_SNDTIMEO) &&
        optval && optlen == sizeof(struct timeval)) {
        const struct timeval *tv = (const struct timeval *)optval;
        timeout_ms = (DWORD)(tv->tv_sec * 1000 + tv->tv_usec / 1000);
        optval = &timeout_ms;
        optlen = sizeof(timeout_ms);
    }
    if (setsockopt(s, level, optname, (const char *)optval, optlen) == 0) return 0;
    ds4_win32_set_errno_from_wsa();
    return -1;
}

int ds4_win32_getsockopt(int fd, int level, int optname,
                         void *optval, socklen_t *optlen) {
    SOCKET s;
    if (!ds4_win32_socket_fd(fd, &s, NULL)) {
        errno = EBADF;
        return -1;
    }
    int len = optlen ? *optlen : 0;
    if (getsockopt(s, level, optname, (char *)optval, &len) == 0) {
        if (optlen) *optlen = len;
        return 0;
    }
    ds4_win32_set_errno_from_wsa();
    return -1;
}

int ds4_win32_inet_pton(int af, const char *src, void *dst) {
    if (!ds4_win32_winsock_init()) return -1;
    return InetPtonA(af, src, dst);
}

struct ds4_win32_dir {
    HANDLE h;
    WIN32_FIND_DATAA data;
    struct dirent ent;
    bool first;
};

DIR *ds4_win32_opendir(const char *path) {
    if (!path) {
        errno = EINVAL;
        return NULL;
    }
    char pattern[MAX_PATH];
    snprintf(pattern, sizeof(pattern), "%s", path);
    size_t n = strlen(pattern);
    while (n && (pattern[n - 1] == '/' || pattern[n - 1] == '\\')) pattern[--n] = '\0';
    if (n + 3 >= sizeof(pattern)) {
        errno = ENAMETOOLONG;
        return NULL;
    }
    strcat(pattern, "\\*");
    DIR *d = (DIR *)calloc(1, sizeof(*d));
    if (!d) {
        errno = ENOMEM;
        return NULL;
    }
    d->h = FindFirstFileA(pattern, &d->data);
    if (d->h == INVALID_HANDLE_VALUE) {
        free(d);
        ds4_win32_set_errno_from_last_error();
        return NULL;
    }
    d->first = true;
    return d;
}

struct dirent *ds4_win32_readdir(DIR *dir) {
    if (!dir) {
        errno = EINVAL;
        return NULL;
    }
    if (dir->first) {
        dir->first = false;
    } else if (!FindNextFileA(dir->h, &dir->data)) {
        return NULL;
    }
    snprintf(dir->ent.d_name, sizeof(dir->ent.d_name), "%s", dir->data.cFileName);
    return &dir->ent;
}

int ds4_win32_closedir(DIR *dir) {
    if (!dir) {
        errno = EINVAL;
        return -1;
    }
    BOOL ok = FindClose(dir->h);
    free(dir);
    return ok ? 0 : -1;
}

static int ds4_win32_fnmatch_impl(const char *p, const char *s) {
    while (*p) {
        if (*p == '*') {
            while (*p == '*') p++;
            if (!*p) return 0;
            while (*s) {
                if (ds4_win32_fnmatch_impl(p, s) == 0) return 0;
                s++;
            }
            return FNM_NOMATCH;
        }
        if (*p == '?') {
            if (!*s) return FNM_NOMATCH;
            p++;
            s++;
            continue;
        }
        char pc = *p == '\\' ? '/' : *p;
        char sc = *s == '\\' ? '/' : *s;
        if (pc != sc) return FNM_NOMATCH;
        p++;
        s++;
    }
    return *s ? FNM_NOMATCH : 0;
}

int ds4_win32_fnmatch(const char *pattern, const char *string, int flags) {
    (void)flags;
    if (!pattern || !string) return FNM_NOMATCH;
    return ds4_win32_fnmatch_impl(pattern, string);
}

static bool ds4_win32_contains(const char *hay, const char *needle, bool icase) {
    size_t nn = strlen(needle);
    if (!nn) return true;
    for (const char *p = hay; *p; p++) {
        size_t i = 0;
        for (; i < nn && p[i]; i++) {
            unsigned char a = (unsigned char)p[i];
            unsigned char b = (unsigned char)needle[i];
            if (icase) {
                a = (unsigned char)tolower(a);
                b = (unsigned char)tolower(b);
            }
            if (a != b) break;
        }
        if (i == nn) return true;
    }
    return false;
}

int ds4_win32_regcomp(regex_t *preg, const char *regex, int cflags) {
    if (!preg || !regex) return REG_ESPACE;
    memset(preg, 0, sizeof(*preg));
    preg->pattern = _strdup(regex);
    preg->flags = cflags;
    return preg->pattern ? 0 : REG_ESPACE;
}

int ds4_win32_regexec(const regex_t *preg, const char *string,
                      size_t nmatch, regmatch_t pmatch[], int eflags) {
    (void)nmatch;
    (void)pmatch;
    (void)eflags;
    if (!preg || !preg->pattern || !string) return REG_NOMATCH;
    return ds4_win32_contains(string, preg->pattern, (preg->flags & REG_ICASE) != 0) ?
        0 : REG_NOMATCH;
}

size_t ds4_win32_regerror(int errcode, const regex_t *preg,
                          char *errbuf, size_t errbuf_size) {
    (void)preg;
    const char *msg = errcode == REG_NOMATCH ? "no match" :
                      errcode == REG_ESPACE ? "out of memory" :
                      "regex unavailable";
    size_t n = strlen(msg) + 1;
    if (errbuf && errbuf_size) snprintf(errbuf, errbuf_size, "%s", msg);
    return n;
}

void ds4_win32_regfree(regex_t *preg) {
    if (!preg) return;
    free(preg->pattern);
    preg->pattern = NULL;
}

int ds4_win32_ioctl(int fd, unsigned long request, void *arg) {
    if (request == TIOCGWINSZ && arg) {
        intptr_t osfh = _get_osfhandle(fd);
        if (osfh == -1) {
            errno = EBADF;
            return -1;
        }
        CONSOLE_SCREEN_BUFFER_INFO info;
        if (!GetConsoleScreenBufferInfo((HANDLE)osfh, &info)) {
            ds4_win32_set_errno_from_last_error();
            return -1;
        }
        struct winsize *ws = (struct winsize *)arg;
        ws->ws_col = (unsigned short)(info.srWindow.Right - info.srWindow.Left + 1);
        ws->ws_row = (unsigned short)(info.srWindow.Bottom - info.srWindow.Top + 1);
        ws->ws_xpixel = 0;
        ws->ws_ypixel = 0;
        return 0;
    }
    errno = ENOSYS;
    return -1;
}

pid_t ds4_win32_fork(void) {
    errno = ENOSYS;
    return -1;
}

pid_t ds4_win32_waitpid(pid_t pid, int *status, int options) {
    (void)pid;
    (void)options;
    if (status) *status = 127;
    errno = ENOSYS;
    return -1;
}

int ds4_win32_kill(pid_t pid, int sig) {
    (void)pid;
    (void)sig;
    errno = ENOSYS;
    return -1;
}

int ds4_win32_setpgid(pid_t pid, pid_t pgid) {
    (void)pid;
    (void)pgid;
    return 0;
}

int ds4_win32_execl(const char *path, const char *arg0, ...) {
    (void)path;
    (void)arg0;
    errno = ENOSYS;
    return -1;
}

#endif /* _WIN32 */
