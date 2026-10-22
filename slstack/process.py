from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import psutil
from .config import ROOT, environment
from .guard import admit


class OwnedProcess:
    def __init__(self, cfg, command, name, *, cwd=ROOT, stdin=None, timeout=None, gpu=False, scope=None):
        admit(cfg['host_budget_gib'], cfg['min_available_gib'])
        root = Path(cfg['runtime_root']) / 'jobs'
        root.mkdir(parents=True, exist_ok=True)
        self.prefix = root / (name + '-' + uuid.uuid4().hex)
        self.stop_path = self.prefix.with_suffix('.stop')
        self.receipt = self.prefix.with_suffix('.receipt.json')
        self.log = self.prefix.with_suffix('.log')
        scope = scope or ('ds4-win' if name == 'ds4-server' else name if name in ('image', 'video') else 'studio')
        spec = {'command': command, 'cwd': str(cwd), 'env': environment(cfg, scope),
                'runtime_root': cfg['runtime_root'],
                'host_budget_gib': cfg['host_budget_gib'], 'min_available_gib': cfg['min_available_gib'],
                'job_commit_gib': cfg['job_commit_gib'], 'timeout_s': timeout or cfg['request_timeout_s'],
                'owner_pid': os.getpid(), 'owner_created': psutil.Process().create_time(),
                'stop': str(self.stop_path), 'receipt': str(self.receipt), 'log': str(self.log)}
        if stdin is not None:
            spec['stdin'] = stdin
        if gpu:
            spec['gpu'] = cfg
        self.spec = self.prefix.with_suffix('.spec.json')
        self.spec.write_text(json.dumps(spec))
        self.guard_log = self.prefix.with_suffix('.guard.log').open('ab')
        self.proc = subprocess.Popen([sys.executable, '-B', '-m', 'slstack.guard', str(self.spec)],
                                     cwd=ROOT, env=environment(cfg), stdout=self.guard_log, stderr=self.guard_log,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)

    def wait(self):
        code = self.proc.wait()
        self.guard_log.close()
        if code or not self.receipt.exists():
            detail = self.receipt.read_text() if self.receipt.exists() else self.prefix.with_suffix('.guard.log').read_text()
            tail = self.log.read_text(errors='replace')[-2000:] if self.log.exists() else ''
            raise RuntimeError(f'Guarded job failed: {detail}\n{tail}')
        return json.loads(self.receipt.read_text())

    def stop(self):
        self.stop_path.touch()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            # The Windows Job closes when its guardian dies and kills all child processes.
            self.proc.kill()
            self.proc.wait(timeout=10)
        self.guard_log.close()
