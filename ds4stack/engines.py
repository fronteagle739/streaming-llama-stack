from __future__ import annotations
import json
from pathlib import Path
import sys
import socket
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from .config import path
from .process import OwnedProcess


def http_json(url, body=None, timeout=30):
    data = json.dumps(body, allow_nan=False).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json',
                                                             'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'Upstream HTTP {exc.code}: {exc.read(4096).decode(errors="replace")}') from exc


class Engines:
    def __init__(self, cfg):
        self.cfg = cfg
        self.llm = None
        self.lock = threading.RLock()

    def close(self):
        if self.llm:
            self.llm.stop()
            self.llm = None

    def ready(self):
        if self.llm and self.llm.proc.poll() is None:
            return
        self.close()
        if self.cfg['lora']:
            raise RuntimeError('Pinned ds4-win does not implement qualified LoRA attachment')
        if self.cfg['long_prompt']:
            raise RuntimeError('ULTRA handoff requires the Windows target-only adapter and parity gate')
        if not self.cfg['model'] or not path(self.cfg['model']).exists():
            raise ValueError('Configure an existing ds4 model path')
        address = urlsplit(self.cfg['ds4_url'])
        if address.hostname not in ('127.0.0.1', 'localhost') or address.scheme != 'http':
            raise ValueError('Owned ds4_url must be local loopback HTTP')
        with socket.socket() as probe:
            probe.bind((address.hostname, address.port or 80))
        command = [x.replace('{model}', str(path(self.cfg['model']))) for x in self.cfg['ds4_command']]
        command[0] = str(path(command[0]))
        self.llm = OwnedProcess(self.cfg, command, 'ds4-server', gpu=True,
                                cwd=Path(self.cfg['runtime_root']), timeout=self.cfg['lease_minutes'] * 60)
        start = time.monotonic()
        while time.monotonic() - start < min(300, self.cfg['request_timeout_s']):
            if self.llm.proc.poll() is not None:
                self.llm.wait()
                raise RuntimeError('ds4 server exited before readiness')
            try:
                http_json(self.cfg['ds4_url'] + '/v1/models', timeout=2)
                return
            except (OSError, RuntimeError):
                time.sleep(.2)
        self.close()
        raise TimeoutError('ds4 server readiness deadline exceeded')

    def chat(self, body):
        self.ready()
        return http_json(self.cfg['ds4_url'] + '/v1/chat/completions', body, self.cfg['request_timeout_s'])

    def media(self, kind, params):
        model = self.cfg[kind + '_model']
        if not model and not self.cfg['comfyui_models']:
            raise ValueError(f'No {kind} model selected. Set --{kind}-model and its architecture/support files.')
        self.close()  # Certify owned LLM process tree has exited before giving Studio the GPU.
        request = {'kind': kind, 'cfg': self.cfg, 'params': params}
        proc = OwnedProcess(self.cfg, [self.cfg['studio_python'], '-B', '-m', 'ds4stack.studio_worker'],
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
