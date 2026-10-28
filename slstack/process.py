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


def commit_budget_gib(cfg, *, windows=None):
    """The WDDM private-commit allowance is Windows-only."""
    windows=os.name=='nt' if windows is None else windows
    return cfg['job_commit_gib'] if windows else cfg['host_budget_gib']


class OwnedProcess:
    def __init__(self, cfg, command, name, *, cwd=ROOT, stdin=None, timeout=None, gpu=False, scope=None, telemetry=None, unbounded=False):
        # A waiting guardian allocates no model. Full host admission is repeated
        # by the independent guardian immediately before it starts an engine.
        initial_budget = min(.25, cfg['host_budget_gib']) if gpu and cfg.get('gpu_wait_seconds', 0) else cfg['host_budget_gib']
        admit(initial_budget, cfg['min_available_gib'])
        root = Path(cfg['runtime_root']) / 'jobs'
        root.mkdir(parents=True, exist_ok=True)
        self.prefix = root / (name + '-' + uuid.uuid4().hex)
        self.stop_path = self.prefix.with_suffix('.stop')
        self.receipt = self.prefix.with_suffix('.receipt.json')
        self.log = self.prefix.with_suffix('.log')
        scope = scope or ('ds4-win' if name == 'ds4-server' else name if name in ('image', 'video') else 'studio')
        from .envvars import argv_overrides
        command = argv_overrides(command, cfg, scope)
        # Secret values never reach the spec file; the guard gets them through its own environment.
        from .envvars import split_secrets
        public_env, secrets = split_secrets(environment(cfg, scope))
        spec = {'command': command, 'cwd': str(cwd), 'env': public_env, 'secret_env': sorted(secrets),
                'runtime_root': cfg['runtime_root'],
                'host_budget_gib': cfg['host_budget_gib'], 'min_available_gib': cfg['min_available_gib'],
                'paging_telemetry': bool(cfg.get('paging_telemetry') or gpu or scope in ('ds4-win', 'ds4-mac', 'rtx-producer', 'mlx-serve', 'image', 'video'))
                                    if telemetry is None else telemetry,
                'job_commit_gib': commit_budget_gib(cfg), 'timeout_s': None if unbounded else (timeout or cfg['request_timeout_s']),   # DS4COMBO2: a server has no lifetime cap
                'runtime_min_free_gib': cfg.get('runtime_min_free_gib',0.5) if scope in ('ds4-win','ds4-mac','rtx-producer','mlx-serve') else 0,
                'owner_pid': os.getpid(), 'owner_created': psutil.Process().create_time(),
                'stop': str(self.stop_path), 'receipt': str(self.receipt), 'log': str(self.log)}
        if stdin is not None:
            spec['stdin'] = stdin
        if cfg.get('hold_watch'):
            spec['hold_watch'] = dict(cfg['hold_watch'])
        if gpu:
            from .envvars import redact_cfg
            spec['gpu'] = redact_cfg(cfg)
        self.spec = self.prefix.with_suffix('.spec.json')
        spec['_spec_path'] = str(self.spec)
        self.spec.write_text(json.dumps(spec))
        self.guard_log = self.prefix.with_suffix('.guard.log').open('ab')
        self.proc = subprocess.Popen([sys.executable, '-B', '-m', 'slstack.guard', str(self.spec)],
                                     cwd=ROOT, env={**environment(cfg), **secrets}, stdout=self.guard_log, stderr=self.guard_log,
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
            self.proc.wait(timeout=15 if os.name=='nt' else 90)
        except subprocess.TimeoutExpired:
            if os.name!='nt':
                error=RuntimeError('Graceful owned shutdown incomplete; no forced termination attempted')
                error.owned_cleanup_uncertain=True
                raise error
            # The Windows Job closes when its guardian dies and kills all child processes.
            self.proc.kill()
            self.proc.wait(timeout=10)
        self.guard_log.close()
