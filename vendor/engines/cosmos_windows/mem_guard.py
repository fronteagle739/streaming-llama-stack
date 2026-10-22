# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/mem_guard.py
# Machine fallback paths cleared; no model data included.
import argparse
import ctypes
import subprocess
import sys
import time
_REDACT_VALUE_FLAGS = {'--prompt', '--negative', '--api-key', '--token', '--auth-token', '--bearer-token'}

def _display_cmd(argv):
    out = []
    redact_next = False
    for arg in argv:
        text = str(arg)
        if redact_next:
            out.append(f'<redacted:{len(text)} chars>')
            redact_next = False
            continue
        out.append(text.replace('\r', '\\r').replace('\n', '\\n')[:180])
        if text in _REDACT_VALUE_FLAGS:
            redact_next = True
    return ' '.join(out)

def avail_bytes():
    try:
        import psutil
        return psutil.virtual_memory().available
    except Exception:
        pass

    class MS(ctypes.Structure):
        _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
    ms = MS()
    ms.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    return int(ms.ullAvailPhys)

class MemoryGuardError(RuntimeError):
    pass

def free_gb() -> float:
    return avail_bytes() / 1000000000.0

def total_gb() -> float:
    try:

        class MS(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        ms = MS()
        ms.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return int(ms.ullTotalPhys) / 1000000000.0
    except Exception:
        return float('inf')

def proc_rss_gb() -> float:
    try:

        class PMC(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong), ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t), ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
            return int(c.WorkingSetSize) / 1000000000.0
    except Exception:
        pass
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1000000000.0
    except Exception:
        return 0.0

def preflight(min_free_gb: float=8.0, label: str='model load') -> None:
    import os as _os
    f = free_gb()
    if f < min_free_gb:
        msg = f'[mem_guard] preflight warning for {label}: {f:.1f} GB free RAM below {min_free_gb:.0f}GB sizing hint; attempting anyway. Runtime guard aborts near OOM.'
        if _os.environ.get('MEM_GUARD_STRICT_PREFLIGHT') == '1':
            raise MemoryGuardError(msg)
        print(msg, flush=True)
        return
    print(f'[mem_guard] preflight OK for {label}: {f:.1f} GB free / {total_gb():.0f} GB total.', flush=True)

def start_watchdog(max_process_gb: float=24.0, min_free_gb: float=1.0, poll_seconds: float=1.0, label: str='load'):
    import os as _os
    import threading
    if _os.environ.get('MEM_GUARD_DISABLE') == '1':
        print('[mem_guard] in-process watchdog DISABLED (MEM_GUARD_DISABLE=1)', flush=True)
        return None
    min_free_gb = float(_os.environ.get('MEM_GUARD_MIN_FREE_GB', _os.environ.get('MEM_GUARD_ABORT_FLOOR_GB', '1.0')))
    enforce_proc_cap = _os.environ.get('MEM_GUARD_ENFORCE_PROC_CAP') == '1' or 'MEM_GUARD_MAX_PROC_GB' in _os.environ
    max_process_gb = float(_os.environ.get('MEM_GUARD_MAX_PROC_GB', max_process_gb))

    def _loop():
        peak = 0.0
        while True:
            rss, f = (proc_rss_gb(), free_gb())
            peak = max(peak, rss)
            if enforce_proc_cap and rss > max_process_gb or f < min_free_gb:
                cap = f'{max_process_gb:.0f}' if enforce_proc_cap else 'disabled'
                sys.stderr.write(f'\n[mem_guard] ABORT {label}: RSS={rss:.1f}GB (cap {cap}), free={f:.1f}GB (floor {min_free_gb:.0f}). Killing NOW to protect the machine (peak RSS {peak:.1f}GB).\nMEM_GUARD_TRIP\n')
                sys.stderr.flush()
                try:
                    sys.stdout.flush()
                except Exception:
                    pass
                _os._exit(137)
            time.sleep(poll_seconds)
    t = threading.Thread(target=_loop, daemon=True, name='mem_guard')
    t.start()
    cap = f'{max_process_gb:.0f}GB' if enforce_proc_cap else 'disabled'
    print(f'[mem_guard] in-process watchdog armed: process RSS cap={cap}; kill if free<{min_free_gb:.0f}GB (start free={free_gb():.1f}GB).', flush=True)
    return t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-gb', type=float, default=1.0)
    ap.add_argument('--stdin-file', default='', help="feed this file to the child's stdin")
    ap.add_argument('argv', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    argv = args.argv[1:] if args.argv and args.argv[0] == '--' else args.argv
    start = avail_bytes()
    print(f'[guard] start: {start / 1000000000.0:.1f}GB free; floor={args.min_gb}GB; cmd={_display_cmd(argv)}', flush=True)
    stdin = open(args.stdin_file) if args.stdin_file else None
    proc = subprocess.Popen(argv, stdin=stdin)
    lo, t = (start, 0)
    while proc.poll() is None:
        a = avail_bytes()
        lo = min(lo, a)
        if t % 5 == 0:
            print(f'[guard] t={t}s free={a / 1000000000.0:.1f}GB (min {lo / 1000000000.0:.1f}GB)', flush=True)
        if a < args.min_gb * 1000000000.0:
            print(f'[guard] !!! ABORT: free RAM {a / 1000000000.0:.1f}GB < floor {args.min_gb}GB — killing child tree', flush=True)
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)], capture_output=True)
            return 2
        time.sleep(1)
        t += 1
    print(f'[guard] done rc={proc.returncode}; min free during run: {lo / 1000000000.0:.1f}GB (started {start / 1000000000.0:.1f}GB)', flush=True)
    return proc.returncode
if __name__ == '__main__':
    sys.exit(main())
