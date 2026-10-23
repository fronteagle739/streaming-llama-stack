"""Certify process retirement AND recovered physical VRAM before the next owner."""
import time


def certified(proof):
    return isinstance(proof,dict) and all(proof.get(k) for k in ('owned_job_empty','owned_process_signaled','owned_pid_gone'))


def release_barrier(runtime, native, cfg, baseline_free, cancelled):
    start=time.monotonic();proof=runtime.close()
    if not certified(proof):
        raise RuntimeError('GPU retirement certificate incomplete; ownership remains poisoned')
    while True:
        if cancelled():raise RuntimeError('Handoff cancelled in the same owner epoch')
        free=native.gpu(0,cfg['gpu_uuid'])
        if free>=baseline_free-(128<<20):
            return dict(cleanup=proof,resident_admission=getattr(getattr(runtime,'process',None),'resume_admission',None),
                        gpu_free_bytes=free,baseline_free_bytes=baseline_free,
                        allowed_desktop_drift_bytes=128<<20,seconds=time.monotonic()-start)
        if time.monotonic()-start>=30:raise RuntimeError('Owned process retired but GPU VRAM did not recover; next engine refused')
        time.sleep(.1)

# DS4COMBO fast switch: a suspended keep-alive producer keeps only its CUDA
# context/modules and small non-arena allocations. That residual is measured
# against the pre-native baseline and must stay within an explicit admission.
def suspend_barrier(runtime, native, cfg, baseline_free, cancelled):
    start=time.monotonic()
    limit=int(cfg.get('producer_suspended_device_bytes',1<<30))
    while True:
        if cancelled():raise RuntimeError('Handoff cancelled in the same owner epoch')
        free=native.gpu(0,cfg['gpu_uuid'])
        if free>=baseline_free-limit:
            return dict(suspended=True,gpu_free_bytes=free,baseline_free_bytes=baseline_free,
                        suspended_residual_bytes=max(0,baseline_free-free),admitted_residual_bytes=limit,seconds=time.monotonic()-start)
        if time.monotonic()-start>=30:
            raise RuntimeError('Suspended producer kept %d bytes of VRAM; admitted residual is %d' % (baseline_free-free,limit))
        time.sleep(.1)
