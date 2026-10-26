"""Small, prompt-free progress snapshot shared by the media worker and WebUI."""
import json
import math
from pathlib import Path
import time
import uuid
from .generation import publish


class MediaProgress:
    def __init__(self, root, kind, total):
        self.path = Path(root) / 'jobs' / ('media-progress-' + uuid.uuid4().hex + '.json')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        publish(self.path, dict(id=self.path.stem, kind=kind, active=True, phase='preparing',
                               step=0, total=total, s_iter=None, started=time.time(), updated=time.time()))

    def snapshot(self):
        return read(self.path)

    def phase(self, phase, *, active=True):
        row = self.snapshot()
        row.update(phase=phase, active=active, updated=time.time())
        publish(self.path, row)


def read(path):
    try:
        with Path(path).open(encoding='utf-8') as source:
            return json.loads(source.read(8192))
    except (OSError, ValueError):
        return {}


def callback(path):
    """Studio's callback counts completed steps; preparation never enters s/it."""
    last_step, last_time = 0, None
    def update(step, total, preview=None, meta=None):
        nonlocal last_step, last_time
        if not path:return
        row = read(path)
        if not row:return
        now = time.monotonic()
        meta = meta if isinstance(meta, dict) else {}
        if type(step) is int and type(total) is int and total > 0 and 0 < step <= total:
            rate = meta.get('s_iter')
            if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
                rate = (now-last_time)/(step-last_step) if last_time is not None and step > last_step else row.get('s_iter')
            if step > last_step or step < last_step:
                last_step, last_time = step, now
            row.update(step=step, total=total, s_iter=rate,
                       phase='decoding' if step == total else 'denoising')
        elif meta.get('phase') in ('decode', 'mux', 'upscale'):
            row['phase'] = {'decode':'decoding', 'mux':'encoding', 'upscale':'upscaling'}[meta['phase']]
        row['updated'] = time.time()
        publish(Path(path), row)
    return update
