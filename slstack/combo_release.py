"""Certify process retirement AND recovered physical VRAM before the next owner."""
import time


def release_barrier(runtime, native, cfg, baseline_free, cancelled):
    start=time.monotonic();proof=runtime.close()
    if not all(proof.get(k) for k in ('owned_job_empty','owned_process_signaled','owned_pid_gone')):
        raise RuntimeError('GPU retirement certificate incomplete; ownership remains poisoned')
    while True:
        if cancelled():raise RuntimeError('Handoff cancelled in the same owner epoch')
        free=native.gpu(0,cfg['gpu_uuid'])
        if free>=baseline_free-(128<<20):
            return dict(cleanup=proof,gpu_free_bytes=free,baseline_free_bytes=baseline_free,
                        allowed_desktop_drift_bytes=128<<20,seconds=time.monotonic()-start)
        if time.monotonic()-start>=30:raise RuntimeError('Owned process retired but GPU VRAM did not recover; next engine refused')
        time.sleep(.1)
