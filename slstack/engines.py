from __future__ import annotations
import json
from pathlib import Path
import sys
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from .config import path
from .process import OwnedProcess


def http_json(url, body=None, timeout=30, headers=None):
    data = json.dumps(body, allow_nan=False).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json',
                                                             'Accept': 'application/json', **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'Upstream HTTP {exc.code}: {exc.read(4096).decode(errors="replace")}') from exc


class Engines:
    def __init__(self, cfg):
        self.cfg = cfg
        from .models import Models
        self.models = Models(cfg)
        self.selected_model = self.models.default
        selected = self.models.resolve()
        self.cfg.update(model=selected['path'], model_name=selected['id'], lora=selected.get('lora', []))
        self.loading_model = None
        self.events = []
        self.llm = None
        self.remote = None
        self.lock = threading.RLock()
        from .journal import Journal
        self.journal = Journal(cfg['runtime_root'])
        self.attach_entry = None
        self.attach_guard_recorded = False
        self.journal_checked = False

    def status(self, message, model=None):
        row = {'time': time.time(), 'message': message, 'model': model or self.selected_model}
        self.events = [*self.events[-63:], row]
        print('Model status: ' + json.dumps(row), flush=True)

    def catalog(self):
        active = self.selected_model if self.llm and self.llm.proc.poll() is None and not self.loading_model else None
        if self.remote and self.remote.connected:
            active = self.selected_model
        return self.models.catalog(active, self.loading_model)

    def select(self, name=None):
        row = self.models.resolve(name)  # Refuse unknown input before ejecting anything.
        if row['id'] != self.selected_model:
            self.status('Unloading model to switch chats')
            self.close()
            self.selected_model = row['id']
            self.cfg.update(model=row['path'], model_name=row['id'], lora=row.get('lora', []))
            if row['id'] in self.models.resident:
                self.status('Resident pair requested; using normal unload/load until dual residency is available')
        return row

    def close(self):
        if self.remote:
            try:
                self.remote.close()
            finally:
                self.remote = None
        if self.llm:
            self.status('Unloading model')
            clean = self.llm.proc.poll() == 0
            if self.llm.proc.poll() is None:
                try:
                    result = http_json(self.cfg['ds4_url'] + '/v1/train/close', {}, timeout=10,
                                       headers={'Authorization': 'Bearer ' + self.cfg['_control_token']})
                    if result.get('closing'):
                        clean = self.llm.proc.wait(timeout=30) == 0
                except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired):
                    pass
            self.llm.stop()
            self.llm = None
            if self.attach_entry:
                if not self.attach_guard_recorded:
                    self.journal.update(self.attach_entry, 'failed', applied=False, error='Engine startup ownership was not established')
                elif clean:
                    self.journal.update(self.attach_entry, 'detached', applied=False)
                else:
                    reports = self.journal.recover()
                    print('LoRA owner stopped unexpectedly; recovered journal: ' + json.dumps(reports), flush=True)
                self.attach_entry = None
                self.attach_guard_recorded = False
            self.loading_model = None
            self.status('Model unloaded')

    def ready(self):
        try:
            return self._ready()
        except BaseException:
            if self.llm:
                self.close()
            elif self.attach_entry:
                self.journal.update(self.attach_entry, 'failed', applied=False,
                                    error='Engine constructor failed before creating an owned job')
                self.attach_entry = None
                self.attach_guard_recorded = False
            self.loading_model = None
            raise

    def _ready(self):
        if self.cfg.get('mac'):
            if self.remote is None:
                from .pairing import Pair
                self.remote = Pair(self.cfg, self.status)
            self.remote.start()
            return
        if self.llm and self.llm.proc.poll() is None:
            return
        self.close()
        if not self.journal_checked:
            self.status('Verifying model integrity and prior adapter ownership')
            reports = self.journal.recover()
            if reports:
                print('Recovered stale LoRA journal: ' + json.dumps(reports), flush=True)
            self.journal_checked = True
        if self.cfg['long_prompt']:
            raise RuntimeError('ULTRA handoff requires the Windows target-only adapter and parity gate')
        if not self.cfg['model'] or not path(self.cfg['model']).exists():
            raise ValueError('Configure an existing ds4 model path')
        address = urlsplit(self.cfg['ds4_url'])
        if address.hostname not in ('127.0.0.1', 'localhost') or address.scheme != 'http':
            raise ValueError('Owned ds4_url must be local loopback HTTP')
        with socket.socket() as probe:
            probe.bind((address.hostname, address.port or 80))
        self.loading_model = self.selected_model
        self.status('Loading model')
        command = [x.replace('{model}', str(path(self.cfg['model']))) for x in self.cfg['ds4_command']]
        if self.cfg.get('cache_affinity_spread_file'):
            command += ['--cache-affinity-spread-file', str(path(self.cfg['cache_affinity_spread_file']))]
        if self.cfg['lora']:
            from .lora import parse_lora
            import psutil
            if len(self.cfg['lora']) != 1:
                raise ValueError('Native DSV4.1 output fusion supports exactly one adapter')
            adapter, scale = parse_lora(self.cfg['lora'][0])
            adapter = Path(adapter)
            if adapter.is_dir():
                adapter = adapter / 'adapter_model.safetensors'
            if not adapter.is_file() or not -16 <= scale <= 16:
                raise ValueError('Provide one existing output-head Safetensors adapter and scale in [-16,16]')
            reports = self.journal.recover()
            if reports:
                print('Recovered stale LoRA journal: ' + json.dumps(reports), flush=True)
            row = self.journal.begin(path(self.cfg['model']), [(adapter, scale)], 'attaching')
            self.attach_entry = row['id']
            self.attach_guard_recorded = False
            command += ['--lora', str(adapter), '--lora-scale', str(scale), '--lora-base-sha256', row['base_sha256']]
        command[0] = str(path(command[0]))
        import secrets
        token = self.cfg.setdefault('_control_token', secrets.token_urlsafe(32))
        self.cfg.setdefault('env_overrides', {})['DS4_SERVER_AUTH_TOKEN'] = token
        self.llm = OwnedProcess(self.cfg, command, 'ds4-server', gpu=not self.cfg.get('_gpu_lease_managed', False),
                                cwd=Path(self.cfg['runtime_root']), timeout=self.cfg['lease_minutes'] * 60)
        if self.attach_entry:
            self.journal.update(self.attach_entry, 'attaching', pid=self.llm.proc.pid,
                                pid_created=psutil.Process(self.llm.proc.pid).create_time())
            self.attach_guard_recorded = True
        start = time.monotonic()
        while time.monotonic() - start < min(300, self.cfg['request_timeout_s']) + self.cfg.get('gpu_wait_seconds', 0):
            if self.llm.proc.poll() is not None:
                self.llm.wait()
                raise RuntimeError('ds4 server exited before readiness')
            try:
                http_json(self.cfg['ds4_url'] + '/v1/models', timeout=2)
                if self.attach_entry:
                    self.journal.update(self.attach_entry, 'attached', applied=True)
                self.loading_model = None
                self.status('Model ready')
                return
            except (OSError, RuntimeError):
                time.sleep(.2)
        self.close()
        raise TimeoutError('ds4 server readiness deadline exceeded')

    def quality(self, value=None):
        if self.cfg.get('mac'):
            return {'supported': False, 'phases_supported': False,
                    'reason': 'This Mac decoder has no quality/speed control'}
        # Control must remain available while another thread is decoding.
        # Cold start still shares the normal lifecycle lock with media/model swaps.
        if not self.llm or self.llm.proc.poll() is not None:
            with self.lock:
                self.ready()
        return http_json(self.cfg['ds4_url'] + '/v1/quality-speed',
                         None if value is None else {'value': value}, timeout=10,
                         headers={'Authorization': 'Bearer ' + self.cfg['_control_token']})

    def chat(self, body):
        self.select(body.get('model'))
        self.ready()
        if self.remote:
            return self.remote.chat(body)
        return http_json(self.cfg['ds4_url'] + '/v1/chat/completions', body, self.cfg['request_timeout_s'],
                         headers={'Authorization': 'Bearer ' + self.cfg['_control_token']})

    def chat_events(self, body):
        self.select(body.get('model'))
        self.ready()
        if not self.remote:
            raise ValueError('Native streaming proxy requires the paired Mac receiver')
        yield from self.remote.events(body)

    def media(self, kind, params):
        model = self.cfg[kind + '_model']
        if not model and not self.cfg['comfyui_models']:
            raise ValueError(f'No {kind} model selected. Set --{kind}-model and its architecture/support files.')
        self.close()  # Certify owned LLM process tree has exited before giving Studio the GPU.
        request = {'kind': kind, 'cfg': self.cfg, 'params': params}
        proc = OwnedProcess(self.cfg, [self.cfg['studio_python'], '-B', '-m', 'slstack.studio_worker'],
                            kind, stdin=json.dumps(request) + '\n', gpu=True)
        try:
            proc.wait()
            rows = [x[len('DS4STACK_RESULT '):] for x in proc.log.read_text(errors='replace').splitlines()
                    if x.startswith('DS4STACK_RESULT ')]
            if len(rows) != 1:
                raise RuntimeError(f'Studio did not return exactly one result; see {proc.log}')
            result = json.loads(rows[0])
            if not result.get('ok'):
                raise RuntimeError(result.get('error', 'Studio generation failed'))
            root = Path(self.cfg['runtime_root']).resolve() / 'outputs'
            artifacts = []
            for raw in result.get('files', []):
                source = Path(raw).resolve(strict=True)
                if not source.is_file() or not source.is_relative_to(root):
                    raise RuntimeError('Studio output escaped the isolated output directory')
                artifacts.append({'url': '/outputs/' + source.relative_to(root).as_posix(), 'path': str(source)})
            if not artifacts:
                raise RuntimeError('Studio returned no media files')
            return {'created': int(time.time()), 'data': artifacts, 'settings': params,
                    'mode': self.cfg[kind + '_mode'], 'engine': 'studio', 'receipt': str(proc.receipt),
                    'selected_model': result.get('selected_model'), 'selected_arch': result.get('selected_arch')}
        finally:
            proc.stop()
