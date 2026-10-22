#ifndef DS4_WIN32_H
#define DS4_WIN32_H

#if defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <direct.h>
#include <limits.h>
#include <process.h>
#include <share.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

/* Explicit Windows host admission. The process Job limit also covers private
 * allocations made inside libraries; it never uses kill-on-close. */
typedef struct ds4_win32_host_memory_status {
    uint64_t physical_total_bytes;
    uint64_t physical_available_bytes;
    uint64_t system_commit_bytes;
    uint64_t system_commit_limit_bytes;
    uint64_t process_private_bytes;
    uint64_t process_private_peak_bytes;
    uint64_t process_limit_bytes;
    uint64_t reserve_bytes;
} ds4_win32_host_memory_status;


#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
#endif

#ifndef ssize_t
typedef SSIZE_T ssize_t;
#endif
#ifndef SSIZE_MAX
#define SSIZE_MAX ((ssize_t)(SIZE_MAX >> 1))
#endif

/* CRT descriptors must not leak into child tools, matching POSIX O_CLOEXEC. */
#ifndef O_CLOEXEC
#define O_CLOEXEC _O_NOINHERIT
#endif

/* Private flag handled by ds4_win32_open before passing CRT flags. */
#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0x10000000
#endif

#ifndef STDIN_FILENO
#define STDIN_FILENO 0
#endif
#ifndef STDOUT_FILENO
#define STDOUT_FILENO 1
#endif
#ifndef STDERR_FILENO
#define STDERR_FILENO 2
#endif

#ifndef PATH_MAX
#define PATH_MAX MAX_PATH
#endif

#ifndef S_IRUSR
#define S_IRUSR _S_IREAD
#endif
#ifndef S_IWUSR
#define S_IWUSR _S_IWRITE
#endif

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif
#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME 0
#endif

#define _SC_PAGESIZE 1
#define _SC_NPROCESSORS_ONLN 2

#define PROT_READ  0x1
#define PROT_WRITE 0x2

#define MAP_PRIVATE   0x01
#define MAP_SHARED    0x02
#define MAP_ANONYMOUS 0x20
#define MAP_ANON MAP_ANONYMOUS
#define MAP_FAILED ((void *)-1)

#define MS_ASYNC 0x1
#define MS_SYNC 0x2
#define MS_INVALIDATE 0x4

#define LOCK_SH 1
#define LOCK_EX 2
#define LOCK_NB 4
#define LOCK_UN 8

#ifndef __thread
#define __thread __declspec(thread)
#endif

#ifndef F_SETFD
#define F_SETFD 2
#endif
#ifndef FD_CLOEXEC
#define FD_CLOEXEC 1
#endif

/* pthread_mutex_t is backed by a CRITICAL_SECTION, not an SRWLOCK, because the
 * server needs a *recursive* mutex.  ds4_server.c creates s.inference_mu with
 * PTHREAD_MUTEX_RECURSIVE and genuinely re-enters it on one thread:
 *
 *   generate_job_inner()            installs server_progress_cb on the session
 *     server_session_sync()
 *       server_prefill_enter()      -> pthread_mutex_lock(&s->inference_mu)   [1]
 *       ds4_session_sync()          -> ds4_session_note_prefill_progress()
 *         server_progress_cb()      -> kv_cache_maybe_store_continued()
 *           kv_cache_store_live_prefix_text()
 *                                   -> pthread_mutex_lock(&s->inference_mu)   [2]
 *
 * An SRWLOCK self-deadlocks at [2]; a CRITICAL_SECTION is recursive and does
 * not.  CRITICAL_SECTION has no static initializer, so PTHREAD_MUTEX_INITIALIZER
 * zero-fills the object and the first lock/wait initializes it lazily. */
#define DS4_WIN32_MUTEX_UNINIT   0
#define DS4_WIN32_MUTEX_INITING  1
#define DS4_WIN32_MUTEX_READY    2

typedef struct ds4_win32_mutex {
    CRITICAL_SECTION cs;
    volatile LONG    state; /* DS4_WIN32_MUTEX_* */
} pthread_mutex_t;

typedef struct ds4_win32_mutexattr {
    int type;
} pthread_mutexattr_t;

/* A CRITICAL_SECTION is always recursive, so RECURSIVE is honoured natively and
 * the other types are (harmlessly) upgraded to it. */
#define PTHREAD_MUTEX_NORMAL     0
#define PTHREAD_MUTEX_RECURSIVE  1
#define PTHREAD_MUTEX_ERRORCHECK 2
#define PTHREAD_MUTEX_DEFAULT    PTHREAD_MUTEX_NORMAL

typedef CONDITION_VARIABLE pthread_cond_t;
typedef HANDLE pthread_t;
typedef int pthread_attr_t;
typedef INIT_ONCE pthread_once_t;

typedef int pid_t;
typedef unsigned long nfds_t;
typedef int socklen_t;

#ifndef POLLIN
struct pollfd {
    int fd;
    short events;
    short revents;
};

#define POLLIN   0x0001
#define POLLOUT  0x0004
#define POLLERR  0x0008
#define POLLHUP  0x0010
#define POLLNVAL 0x0020
#endif

#ifndef F_GETFL
#define F_GETFL 3
#endif
#ifndef F_SETFL
#define F_SETFL 4
#endif
#ifndef O_NONBLOCK
#define O_NONBLOCK 0x4000
#endif

#ifndef F_OK
#define F_OK 0
#endif
#ifndef X_OK
#define X_OK 1
#endif

#ifndef S_ISDIR
#define S_ISDIR(m) (((m) & _S_IFDIR) != 0)
#endif
#ifndef S_ISREG
#define S_ISREG(m) (((m) & _S_IFREG) != 0)
#endif
#ifndef S_ISLNK
#define S_ISLNK(m) (0)
#endif

#ifndef SIGPIPE
#define SIGPIPE SIGTERM
#endif
#ifndef SIGKILL
#define SIGKILL SIGTERM
#endif
#ifndef EWOULDBLOCK
#define EWOULDBLOCK EAGAIN
#endif
#ifndef ENOSYS
#define ENOSYS EINVAL
#endif

#ifndef SHUT_WR
#define SHUT_WR SD_SEND
#endif

#define WNOHANG 1
#define WIFEXITED(status)   (1)
#define WEXITSTATUS(status) (status)
#define WIFSIGNALED(status) (0)
#define WTERMSIG(status)    (0)

#define REG_EXTENDED 1
#define REG_ICASE    2
#define REG_NOSUB    4
#define REG_NOMATCH  1
#define REG_ESPACE   12

typedef struct {
    char *pattern;
    int flags;
} regex_t;
typedef int regmatch_t;

#define FNM_NOMATCH 1

struct winsize {
    unsigned short ws_row;
    unsigned short ws_col;
    unsigned short ws_xpixel;
    unsigned short ws_ypixel;
};

#define TIOCGWINSZ 0x5413

typedef struct ds4_win32_dir DIR;
struct dirent {
    char d_name[MAX_PATH];
};

/* Zero-filled: the CRITICAL_SECTION is created on first use (see
 * ds4_win32_mutex_ensure() in ds4_win32.c). */
#define PTHREAD_MUTEX_INITIALIZER {0}
#define PTHREAD_COND_INITIALIZER CONDITION_VARIABLE_INIT
#define PTHREAD_ONCE_INIT INIT_ONCE_STATIC_INIT

#ifdef __cplusplus
extern "C" {
#endif

int pthread_mutexattr_init(pthread_mutexattr_t *attr);
int pthread_mutexattr_destroy(pthread_mutexattr_t *attr);
int pthread_mutexattr_settype(pthread_mutexattr_t *attr, int type);
int pthread_mutexattr_gettype(const pthread_mutexattr_t *attr, int *type);
int pthread_mutex_init(pthread_mutex_t *mutex, const void *attr);
int pthread_mutex_destroy(pthread_mutex_t *mutex);
int pthread_mutex_lock(pthread_mutex_t *mutex);
int pthread_mutex_trylock(pthread_mutex_t *mutex);
int pthread_mutex_unlock(pthread_mutex_t *mutex);
int pthread_cond_init(pthread_cond_t *cond, const void *attr);
int pthread_cond_destroy(pthread_cond_t *cond);
int pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex);
int pthread_cond_timedwait(pthread_cond_t *cond, pthread_mutex_t *mutex,
                           const struct timespec *abstime);
int pthread_cond_signal(pthread_cond_t *cond);
int pthread_cond_broadcast(pthread_cond_t *cond);
int pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                   void *(*start_routine)(void *), void *arg);
int pthread_join(pthread_t thread, void **retval);
int pthread_detach(pthread_t thread);
int pthread_once(pthread_once_t *once_control, void (*init_routine)(void));

typedef int sigset_t;
struct sigaction {
    void (*sa_handler)(int);
    sigset_t sa_mask;
    int sa_flags;
};

int sigemptyset(sigset_t *set);
int sigaction(int signum, const struct sigaction *act, struct sigaction *oldact);

int ds4_win32_open(const char *path, int flags, ...);
int ds4_win32_close(int fd);
int ds4_win32_fcntl(int fd, int cmd, ...);
ssize_t ds4_win32_pread(int fd, void *buf, size_t count, int64_t offset);
int ds4_win32_dprintf(int fd, const char *fmt, ...);
int ds4_win32_ftruncate(int fd, int64_t length);
int ds4_win32_flock(int fd, int operation);
int ds4_win32_mkstemp(char *tmpl);
int ds4_win32_clock_gettime(int clock_id, struct timespec *ts);
int ds4_win32_nanosleep(const struct timespec *req, struct timespec *rem);
unsigned int ds4_win32_sleep(unsigned int seconds);
int ds4_win32_usleep(unsigned int usec);
int ds4_win32_gettimeofday(struct timeval *tv, void *tz);
struct tm *ds4_win32_localtime_r(const time_t *timep, struct tm *result);
long ds4_win32_sysconf(int name);
int ds4_win32_getpagesize(void);
void *ds4_win32_mmap(void *addr, size_t length, int prot, int flags,
                     int fd, int64_t offset);
int ds4_win32_munmap(void *addr, size_t length);
int ds4_win32_mlock(const void *addr, size_t length);
int ds4_win32_munlock(const void *addr, size_t length);
int ds4_win32_msync(void *addr, size_t length, int flags);
/* Optional working-set hint for a registered file view; mapping/data survive. */
int ds4_win32_trim_mapped_pages(const void *addr, size_t length);
#ifndef MADV_DONTNEED
#define MADV_DONTNEED 4
#endif
int ds4_win32_madvise(void *addr, size_t length, int advice);
char *ds4_win32_realpath(const char *path, char *resolved);
void ds4_win32_enable_utf8_console(void);
bool ds4_win32_host_memory_begin(void);
bool ds4_win32_host_memory_check(uint64_t requested_bytes, const char *reason);
/* Device backing consumes private/system commit; its size is not physical
 * host residency. Both checks still require the current physical reserve. */
bool ds4_win32_host_memory_check_commit(uint64_t requested_bytes, const char *reason);
/* Pure validation of a measured/planned snapshot, without allocating. */
bool ds4_win32_host_memory_fits_snapshot(const ds4_win32_host_memory_status *state,
                                       uint64_t requested_bytes, bool host_allocation);
bool ds4_win32_host_memory_read(ds4_win32_host_memory_status *out);
void ds4_win32_host_memory_snapshot(const char *reason);
int ds4_win32_pipe(int fds[2]);
int ds4_win32_poll(struct pollfd *fds, nfds_t nfds, int timeout_ms);
int ds4_win32_socket(int af, int type, int protocol);
int ds4_win32_accept(int fd, struct sockaddr *addr, socklen_t *addrlen);
int ds4_win32_bind(int fd, const struct sockaddr *addr, socklen_t addrlen);
int ds4_win32_listen(int fd, int backlog);
int ds4_win32_connect(int fd, const struct sockaddr *addr, socklen_t addrlen);
ssize_t ds4_win32_send(int fd, const void *buf, size_t len, int flags);
ssize_t ds4_win32_recv(int fd, void *buf, size_t len, int flags);
int ds4_win32_shutdown(int fd, int how);
int ds4_win32_setsockopt(int fd, int level, int optname,
                         const void *optval, socklen_t optlen);
int ds4_win32_getsockopt(int fd, int level, int optname,
                         void *optval, socklen_t *optlen);
int ds4_win32_inet_pton(int af, const char *src, void *dst);
DIR *ds4_win32_opendir(const char *path);
struct dirent *ds4_win32_readdir(DIR *dir);
int ds4_win32_closedir(DIR *dir);
int ds4_win32_fnmatch(const char *pattern, const char *string, int flags);
int ds4_win32_regcomp(regex_t *preg, const char *regex, int cflags);
int ds4_win32_regexec(const regex_t *preg, const char *string,
                      size_t nmatch, regmatch_t pmatch[], int eflags);
size_t ds4_win32_regerror(int errcode, const regex_t *preg,
                          char *errbuf, size_t errbuf_size);
void ds4_win32_regfree(regex_t *preg);
int ds4_win32_ioctl(int fd, unsigned long request, void *arg);
pid_t ds4_win32_fork(void);
pid_t ds4_win32_waitpid(pid_t pid, int *status, int options);
int ds4_win32_kill(pid_t pid, int sig);
int ds4_win32_setpgid(pid_t pid, pid_t pgid);
int ds4_win32_execl(const char *path, const char *arg0, ...);

#ifdef __cplusplus
}
#endif

#define open ds4_win32_open
#define close ds4_win32_close
#define fcntl ds4_win32_fcntl
#define pread ds4_win32_pread
#define dprintf ds4_win32_dprintf
#define unlink _unlink
#define fileno _fileno
#define isatty _isatty
#define fdopen _fdopen
#define stat _stat64
#define fstat _fstat64
#define lstat _stat64
#define getpid _getpid
#define strdup _strdup
#define strcasecmp _stricmp
#define strncasecmp _strnicmp
#define ftello _ftelli64
#define fseeko _fseeki64
#define ftruncate ds4_win32_ftruncate
#define flock ds4_win32_flock
#define mkstemp ds4_win32_mkstemp
#define clock_gettime ds4_win32_clock_gettime
#define nanosleep ds4_win32_nanosleep
#define sleep ds4_win32_sleep
#define usleep ds4_win32_usleep
#define gettimeofday ds4_win32_gettimeofday
#define localtime_r ds4_win32_localtime_r
#define sysconf ds4_win32_sysconf
#define getpagesize ds4_win32_getpagesize
#define mmap ds4_win32_mmap
#define munmap ds4_win32_munmap
#define mlock ds4_win32_mlock
#define munlock ds4_win32_munlock
#define msync ds4_win32_msync
#define madvise ds4_win32_madvise
#define realpath ds4_win32_realpath
#define fchmod(fd, mode) (0)
#define fsync _commit
#define mkdir(path, mode) _mkdir(path)
#define rmdir _rmdir
#define access _access
#define chdir _chdir
#define getcwd _getcwd
#define pipe ds4_win32_pipe
#define poll ds4_win32_poll
#define socket ds4_win32_socket
#define accept ds4_win32_accept
#define bind ds4_win32_bind
#define listen ds4_win32_listen
#define connect ds4_win32_connect
#define send ds4_win32_send
#define recv ds4_win32_recv
#define shutdown ds4_win32_shutdown
#define setsockopt ds4_win32_setsockopt
#define getsockopt ds4_win32_getsockopt
#define inet_pton ds4_win32_inet_pton
#define opendir ds4_win32_opendir
#define readdir ds4_win32_readdir
#define closedir ds4_win32_closedir
#define fnmatch ds4_win32_fnmatch
#define regcomp ds4_win32_regcomp
#define regexec ds4_win32_regexec
#define regerror ds4_win32_regerror
#define regfree ds4_win32_regfree
#define ioctl ds4_win32_ioctl
#define fork ds4_win32_fork
#define waitpid ds4_win32_waitpid
#define kill ds4_win32_kill
#define setpgid ds4_win32_setpgid
#define execl ds4_win32_execl
#define dup _dup
#define dup2 _dup2

#endif /* _WIN32 */

#endif /* DS4_WIN32_H */
