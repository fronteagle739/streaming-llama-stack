"""Certify process retirement AND recovered physical VRAM before the next owner."""
import time
from .progress import Window


def certified(proof):
    return isinstance(proof,dict) and all(proof.get(k) for k in ('owned_job_empty','owned_process_signaled','owned_pid_gone'))


def vram_window(cfg):
    """Owner rule (no fixed timeouts): the barrier waits while free VRAM keeps RISING (the
    driver is still releasing); vram_release_stall_seconds (default 30) is the no-progress window."""
    return Window(float(cfg.get('vram_release_stall_seconds',30)))


def release_barrier(runtime, native, cfg, baseline_free, cancelled, *, sleep=time.sleep, window=None):
    start=time.monotonic();proof=runtime.close()
    if not certified(proof):
        raise RuntimeError('GPU retirement certificate incomplete; ownership remains poisoned')
    window=window or vram_window(cfg)
    while True:
        if cancelled():raise RuntimeError('Handoff cancelled in the same owner epoch')
        free=native.gpu(0,cfg['gpu_uuid'])
        if free>=baseline_free-(128<<20):
            return dict(cleanup=proof,resident_admission=getattr(getattr(runtime,'process',None),'resume_admission',None),
                        gpu_free_bytes=free,baseline_free_bytes=baseline_free,
                        allowed_desktop_drift_bytes=128<<20,seconds=time.monotonic()-start)
        window.rising(free)
        if window.expired():
            raise RuntimeError('Owned process retired but GPU VRAM stopped recovering for %.1fs (free=%d, baseline=%d, waited %.1fs); next engine refused'
                               % (window.idle(),free,baseline_free,window.elapsed()))
        sleep(.1)

# DS4COMBO fast switch: a suspended keep-alive producer keeps only its CUDA
# context/modules and small non-arena allocations. That residual is measured
# against the pre-native baseline and must stay within an explicit admission.
def suspend_barrier(runtime, native, cfg, baseline_free, cancelled, *, sleep=time.sleep, window=None):
    start=time.monotonic()
    limit=int(cfg.get('producer_suspended_device_bytes',1<<30))
    window=window or vram_window(cfg)
    while True:
        if cancelled():raise RuntimeError('Handoff cancelled in the same owner epoch')
        free=native.gpu(0,cfg['gpu_uuid'])
        if free>=baseline_free-limit:
            return dict(suspended=True,gpu_free_bytes=free,baseline_free_bytes=baseline_free,
                        suspended_residual_bytes=max(0,baseline_free-free),admitted_residual_bytes=limit,seconds=time.monotonic()-start)
        window.rising(free)
        if window.expired():
            raise RuntimeError('Suspended producer kept %d bytes of VRAM (stopped falling for %.1fs); admitted residual is %d'
                               % (baseline_free-free,window.idle(),limit))
        sleep(.1)
