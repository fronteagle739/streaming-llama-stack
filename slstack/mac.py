"""Owned Mac backend setup and control; native state transport stays in ds4."""
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
import urllib.request
from .config import ROOT, path
from .engines import http_json
from .journal import atomic_json, sha256
from .process import OwnedProcess


def budget(cfg, maximum=400):
    import psutil
    free = psutil.virtual_memory().available / (1 << 30)
    floor = max(2, cfg.get('min_available_gib', 2))
    requested = cfg.get('mac_host_budget_gib') or min(maximum, (free - floor) * .9)
    if requested <= 0 or requested + floor > free:
        raise MemoryError('Mac physical memory budget cannot be admitted')
    return {**cfg, 'host_budget_gib': requested, 'min_available_gib': floor}


def setup(cfg, backend='ds4', *, check=False, zig=None):
    if sys.platform != 'darwin' or platform.machine() != 'arm64':
        raise ValueError('Mac setup requires macOS on Apple Silicon')
    model = path(cfg['model'])
    if not model.exists():
        raise ValueError('Set an existing Mac model path')
    if shutil.disk_usage(cfg['runtime_root']).free < 512 * 1024**2:
        raise ValueError('Mac setup needs at least 512 MiB free scratch space')
    if backend == 'ds4':
        if not model.is_file():
            raise ValueError('ds4 requires a GGUF model file')
        wired = int(subprocess.check_output(['sysctl', '-n', 'iogpu.wired_limit_mb'], text=True).strip())
        if wired < 512000:
            raise ValueError('Native 1017 preset needs a larger wired limit. Run: sudo sysctl iogpu.wired_limit_mb=512000')
        binary = ROOT / 'engines/ds4-mac/ds4-server'
        if not binary.exists() and not check:
            if not shutil.which('clang') or not shutil.which('make'):
                raise ValueError('Install Xcode command line tools: xcode-select --install')
            build_cfg = {**cfg, 'host_budget_gib': 2, 'min_available_gib': 4}
            flags = '-O3 -ffast-math -g -mcpu=native -Wall -Wextra -std=c99 -DDS4_NATIVE_FRONTIER_ENABLE=1'
            OwnedProcess(build_cfg, ['make', '-j2', 'ds4-server', 'CFLAGS=' + flags], 'build-mac-ds4',
                         cwd=ROOT / 'engines/ds4-mac', timeout=900, scope='mac-ds4').wait()
    else:
        binary = ROOT / 'engines/mlx-serve/zig-out/bin/mlx-serve'
        if not binary.exists() and not check:
            zig = zig or shutil.which('zig')
            if not zig:
                raise ValueError('Install a portable Zig 0.17 nightly from https://ziglang.org/download/ and pass --zig <path>; no system installation is performed')
            build_cfg = {**cfg, 'host_budget_gib': 4, 'min_available_gib': 4}
            OwnedProcess(build_cfg, [zig, 'build', '-Doptimize=ReleaseFast', '-j2'], 'build-mlx-serve',
                         cwd=ROOT / 'engines/mlx-serve', timeout=1800, scope='mac-ds4').wait()
    manifest = {'backend': backend, 'ready': binary.is_file(), 'binary': str(binary),
                'binary_sha256': sha256(binary) if binary.is_file() else None}
    if backend == 'ds4':
        manifest['renderer_sha256'] = sha256(ROOT / 'engines/ds4-mac/ds4_server.c')
    if not check:
        atomic_json(Path(cfg['runtime_root']) / 'state' / (backend + '-build.json'), manifest)
    return manifest


def addresses():
    result = []
    try:
        text = subprocess.check_output(['networksetup', '-listallhardwareports'], text=True)
        interfaces = re.findall(r'Hardware Port: ([^\n]+)\nDevice: ([^\n]+)', text)
        interfaces.sort(key=lambda row: 0 if 'Thunderbolt' in row[0] else 1 if 'Ethernet' in row[0] else 2)
        for _, device in interfaces:
            answer = subprocess.run(['ipconfig', 'getifaddr', device.strip()], capture_output=True, text=True)
            address = answer.stdout.strip()
            try:
                if not ipaddress.ip_address(address).is_loopback and address not in result:
                    result.append(address)
            except ValueError:
                pass
    except (OSError, subprocess.SubprocessError):
        pass
    return result or ['127.0.0.1']


class Receiver:
    def __init__(self, cfg, backend, build):
        self.cfg, self.backend, self.build = cfg, backend, build
        self.lock = threading.RLock()
        self.engine = None
        self.engine_mode = None
        self.peer = None
        self.qualification = {}
        self.token_file = Path(cfg['runtime_root']) / 'state' / 'pair.token'
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.token_file.exists():
            with self.token_file.open('x', encoding='utf-8') as stream:
                stream.write(secrets.token_urlsafe(32) + '\n')
            self.token_file.chmod(0o600)
        self.token = self.token_file.read_text().strip()
        saved = self.token_file.with_name('peer.json')
        if saved.is_file():
            row = json.loads(saved.read_text())
            if row.get('receiver_sha256') == build['binary_sha256']:
                self.peer = row

    def close(self):
        if self.engine:
            self.engine.stop()
            self.engine = None
            self.engine_mode = None

    def pair(self, row, client_ip):
        allowed = {'producer_sha256', 'producer_source_revision', 'producer_ipv4', 'broker_port', 'tier'}
        if set(row) != allowed or not re.fullmatch('[0-9a-f]{64}', row['producer_sha256']) or not re.fullmatch('[0-9a-f]{40}', row['producer_source_revision']):
            raise ValueError('Invalid producer identity')
        if str(ipaddress.IPv4Address(row['producer_ipv4'])) != client_ip:
            raise ValueError('Producer address must match the authenticated peer connection')
        if type(row['broker_port']) is not int or not 1 <= row['broker_port'] <= 65535 or row['tier'] not in ('accurate', 'fast', 'ultra'):
            raise ValueError('Invalid broker port or unsupported no-requant tier')
        self.close()
        self.peer = {**row, 'receiver_sha256': self.build['binary_sha256'], 'status': 'candidate'}
        self.qualification = {}
        return self.peer

    def ready(self, native):
        if self.engine and self.engine.proc.poll() is None and self.engine_mode == native:
            return
        self.close()
        cfg = budget(self.cfg, 400 if self.backend == 'ds4' else 128)
        port = cfg.get('mac_engine_port', 8181)
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', port))
        env = dict(cfg.get('env_overrides', {}))
        binary = self.build['binary']
        if self.backend == 'mlx-serve':
            from .native_broker import vendor
            vendor()
            from studio.mlx_serve import mlx_serve_argv
            preset = dict(cfg.get('mlx_preset') or {})
            preset['model_path'] = str(path(cfg['model']))
            previous = os.environ.get('STUDIO_MLX_SERVE_BIN')
            os.environ['STUDIO_MLX_SERVE_BIN'] = binary
            try:
                command = mlx_serve_argv({'mode': 'local', 'rtx': preset}, port=port, bind_address='127.0.0.1')
            finally:
                if previous is None:
                    os.environ.pop('STUDIO_MLX_SERVE_BIN', None)
                else:
                    os.environ['STUDIO_MLX_SERVE_BIN'] = previous
            if not command:
                raise ValueError('Fresh mlx-serve binary not found')
        else:
            model = path(cfg['model'])
            mtp = path(cfg['mtp_model']) if cfg.get('mtp_model') else model.with_name('DeepSeek-V4.1-Flash-1017-DSpark-Q4.gguf')
            if not mtp.is_file():
                raise ValueError('Set the existing DSpark support GGUF with --mtp-model')
            from .native_broker import vendor
            vendor()
            from studio.mac_argv import ds4_server_argv
            preset = json.loads((ROOT / 'presets/dsv41-mac.json').read_text())
            preset.update(cfg.get('mac_native_preset') or {})
            command = ds4_server_argv(str(model), binary=binary, chdir=str(ROOT / 'engines/ds4-mac'),
                port=port, ctx=1000001, dspark=True, mtp_model=str(mtp),
                mtp_draft=preset.get('ds4_mtp_draft'), batched_session=1,
                kv_cache_min_tokens=preset.get('kv_cache_min_tokens'),
                kv_cache_cold_max_tokens=preset.get('kv_cache_cold_max_tokens'),
                kv_cache_continued_interval_tokens=preset.get('kv_cache_continued_interval_tokens'))
            env['DS4_NATIVE_SESSION'] = '1' if native else '0'
            if native:
                if not self.peer:
                    raise ValueError('Pair the RTX producer first')
                model_stat = model.stat()
                qualification = {'schema': 'dsv41-native-session-qualification-v2',
                    'model_sha256': '2cc6bc254f22a0238020e61e931c0cac865218aada1eed9c0036d9f4031c6e5f',
                    'mode': 3, 'wire_version': 2, 'tokenization': 'native-final-token-ids-le32-v1',
                    'producer_binary_sha256': self.peer['producer_sha256'],
                    'producer_source_revision': self.peer['producer_source_revision'],
                    'receiver_binary_sha256': self.build['binary_sha256'],
                    'renderer_sha256': self.build['renderer_sha256'],
                    'receiver_model_size': model_stat.st_size, 'receiver_model_device': model_stat.st_dev,
                    'receiver_model_inode': model_stat.st_ino, 'receiver_model_mtime_ns': model_stat.st_mtime_ns,
                    'qualification_state': self.peer['status']}
                artifact = self.token_file.with_name('native-qualification.json')
                atomic_json(artifact, qualification)
                dynamic = dict(DS4_NATIVE_SESSION_IPV4=self.peer['producer_ipv4'], DS4_NATIVE_SESSION_PORT=str(self.peer['broker_port']),
                    DS4_NATIVE_SESSION_TIER=self.peer['tier'], DS4_NATIVE_SESSION_PAIR_TOKEN_FILE=str(self.token_file),
                    DS4_NATIVE_SESSION_QUALIFICATION=str(artifact), DS4_NATIVE_SESSION_QUALIFICATION_SHA256=sha256(artifact),
                    DS4_NATIVE_SESSION_RENDERER_SHA256=self.build['renderer_sha256'],
                    DS4_NATIVE_SESSION_PRODUCER_SHA256=self.peer['producer_sha256'], DS4_NATIVE_SESSION_DEADLINE_SECONDS='600',
                    DS4_NATIVE_SESSION_HOST_BYTES=str(int(cfg['host_budget_gib'] * (1 << 30))),
                    DS4_NATIVE_SESSION_HOST_FLOOR_BYTES=str(int(cfg['min_available_gib'] * (1 << 30))))
                for name, value in dynamic.items():
                    env.setdefault(name, value)
        cfg['env_overrides'] = env
        self.engine = OwnedProcess(cfg, command, 'mac-' + self.backend, cwd=Path(cfg['runtime_root']),
                                   timeout=cfg['lease_minutes'] * 60, scope='mac-ds4')
        self.engine_mode = native
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if self.engine.proc.poll() is not None:
                self.engine.wait()
            try:
                http_json(f'http://127.0.0.1:{port}/v1/models', timeout=2)
                return
            except (OSError, RuntimeError):
                time.sleep(.25)
        self.close()
        raise TimeoutError('Mac backend readiness deadline exceeded')

    def chat(self, body, *, qualification=None):
        if qualification is None and self.backend == 'ds4' and (not self.peer or self.peer.get('status') != 'qualified'):
            raise ValueError('Complete the real native/pair parity handshake first')
        native = self.backend == 'ds4' and qualification != 'native'
        self.ready(native)
        body = dict(body, stream=False)
        body.setdefault('prompt_cache_key', 'slstack-' + secrets.token_hex(16))
        body.setdefault('include_previous_reasoning', True)
        response = http_json(f'http://127.0.0.1:{self.cfg.get("mac_engine_port", 8181)}/v1/chat/completions',
                             body, timeout=self.cfg['request_timeout_s'])
        if qualification:
            message = response['choices'][0]['message']
            logprobs = response['choices'][0].get('logprobs') or {}
            self.qualification[qualification] = {'content': message.get('content'),
                                                'reasoning_content': message.get('reasoning_content'),
                                                'greedy_tokens': [x.get('token') for x in logprobs.get('content', [])]}
            if not any(self.qualification[qualification].values()):
                raise ValueError('No observed greedy decisions; parity cannot be established')
        return response

    def accept(self):
        if self.backend == 'ds4':
            if set(self.qualification) != {'native', 'paired'} or self.qualification['native'] != self.qualification['paired']:
                raise ValueError('Native/paired greedy decisions differ; pair remains unqualified')
        if not self.peer:
            raise ValueError('No candidate peer')
        self.peer['status'] = 'qualified'
        self.peer['tiny_parity_sha256'] = hashlib.sha256(json.dumps(self.qualification, sort_keys=True).encode()).hexdigest()
        atomic_json(self.token_file.with_name('peer.json'), self.peer)
        return {k: v for k, v in self.peer.items() if k != 'producer_ipv4'}


def serve(cfg, backend, build):
    receiver = Receiver(cfg, backend, build)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, data):
            payload = json.dumps(data).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def authorized(self):
            return hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + receiver.token)

        def do_GET(self):
            if self.path == '/health':
                self.reply(200, {'ready': True, 'backend': backend, 'receiver_sha256': build['binary_sha256']})
            elif self.path == '/v1/models' and self.authorized():
                self.reply(200, {'object': 'list', 'data': [{'id': cfg['model_name'], 'object': 'model'}]})
            else:
                self.reply(401, {'error': 'Pairing required'})

        def do_POST(self):
            if not self.authorized():
                self.reply(401, {'error': 'Pairing required'})
                return
            try:
                count = int(self.headers.get('Content-Length', 0))
                if not 0 < count <= 16 * 1024**2:
                    raise ValueError('Invalid bounded request length')
                body = json.loads(self.rfile.read(count))
                with receiver.lock:
                    if self.path == '/pair':
                        result = receiver.pair(body, self.client_address[0])
                    elif self.path == '/pair/qualification':
                        if body['mode'] not in ('native', 'paired'):
                            raise ValueError('Unknown qualification mode')
                        fixed = {'messages': [{'role': 'user', 'content': 'Repeat exactly READY, with no explanation.'}],
                                 'max_tokens': 8, 'temperature': 0, 'logprobs': True, 'top_logprobs': 1,
                                 'prompt_cache_key': 'slstack-qualification-' + secrets.token_hex(16)}
                        result = receiver.chat(fixed, qualification=body['mode'])
                    elif self.path == '/pair/accept':
                        result = receiver.accept()
                    elif self.path == '/decode/setup':
                        receiver.ready(backend == 'ds4')
                        result = {'ready': True, 'backend': backend, 'receiver_sha256': build['binary_sha256']}
                    elif self.path == '/decode/eject':
                        receiver.close()
                        result = {'ejected': True}
                    elif self.path == '/v1/chat/completions':
                        result = receiver.chat(body)
                    else:
                        raise ValueError('Unknown receiver operation')
                self.reply(200, result)
            except (ValueError, KeyError, OSError, RuntimeError, TimeoutError) as exc:
                self.reply(409, {'error': str(exc)})

    port = cfg.get('mac_port', 8180)
    for address in addresses():
        print(f'MAC ENDPOINT: http://{address}:{port}', flush=True)
    print('RTX command: slstack rtx --mac <chosen-endpoint> --pair-token ' + receiver.token +
          ' --model <rtx-model.gguf> --comfyui-models <models>', flush=True)
    server = ThreadingHTTPServer((cfg.get('mac_bind', '0.0.0.0'), port), Handler)
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        server.server_close()
        receiver.close()
