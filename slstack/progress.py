"""No-progress windows (owner rule: no fixed timeouts).

A wait lives while the thing it waits on shows progress; it fails only when nothing has
changed for a whole window, and then it says what stopped. Used by the VRAM release barriers,
the engine readiness waits and the guard's job window.
"""
import os
import time
import psutil

_UNSET = object()


class Window:
    def __init__(self, seconds, *, clock=None):
        seconds = float(seconds)
        if not 0 < seconds <= 86400: raise ValueError('No-progress window must be 0-86400 seconds')
        self.seconds, self.clock = seconds, clock or time.monotonic   # looked up now, so tests can patch it
        self.started = self.last = self.clock()
        self.signal = _UNSET; self.best = None

    def observe(self, signal):
        """Renew on ANY change of `signal` (log size, I/O counters, tree CPU...)."""
        if signal != self.signal: self.signal = signal; self.last = self.clock(); return True
        return False

    def rising(self, value):
        """Renew on a new MAXIMUM of `value` (free VRAM/RAM recovering)."""
        if self.best is None or value > self.best: self.best = value; self.last = self.clock(); return True
        return False

    def idle(self): return self.clock() - self.last
    def elapsed(self): return self.clock() - self.started
    def expired(self): return self.idle() >= self.seconds


def activity(pid, log=None):
    """Progress signal of a starting process tree: its log size, member count, read/write bytes
    and CPU time. Any change is progress; a hung or blocked tree leaves it unchanged."""
    size = 0
    if log:
        try: size = os.path.getsize(log)
        except OSError: pass
    io = cpu = members = 0
    try:
        root = psutil.Process(pid)
        for member in [root, *root.children(recursive=True)]:
            try:
                members += 1
                c = member.cpu_times(); cpu += c.user + c.system
                counters = member.io_counters(); io += counters.read_bytes + counters.write_bytes
            except (psutil.Error, AttributeError):
                pass
    except psutil.Error:
        pass
    return (size, members, io, round(cpu, 1))


def readiness_seconds(cfg):
    """No-progress window for an engine's readiness wait (cfg readiness_stall_seconds, default 120)."""
    return float(cfg.get('readiness_stall_seconds', 120))
