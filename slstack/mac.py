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
from .progress import Window, activity, readiness_seconds


def budget(cfg, maximum=400):
    import psutil
    free = psutil.virtual_memory().available / (1 << 30)
    floor = max(4, cfg.get('min_available_gib', 4))
    reserve = max(8, floor)
    requested = cfg.get('mac_host_budget_gib') or min(maximum, (free - reserve) * .9)
    if requested <= 0 or requested + reserve > free:
        raise MemoryError('Mac physical memory budget cannot be admitted')
    return {**cfg, 'host_budget_gib': requested, 'min_available_gib': floor}


def replacement_budget(cfg, maximum, previous_budget, timeout=60):
    """Wait for certified retired buffers to drain, retaining the same cap."""
    if previous_budget is None:
        return budget(cfg, maximum)
    requested = {**cfg, 'mac_host_budget_gib': cfg.get('mac_host_budget_gib') or previous_budget}
    # Owner rule (no fixed timeouts): `timeout` is the no-progress window; it is renewed while
    # available memory keeps rising (the retired buffers are still draining).
    import psutil
    window = Window(timeout)
    announced = False
    while True:
        try:
            return budget(requested, maximum)
        except MemoryError:
            window.rising(psutil.virtual_memory().available)
            if window.expired():
                raise MemoryError('Mac physical memory stopped recovering for %.0fs after certified backend cleanup (%.0fs since retire)'
                                  % (window.idle(), window.elapsed())) from None
            if not announced:
                print('Waiting for retired Mac buffers to release physical memory', flush=True)
                announced = True
            time.sleep(.25)


def check_backend_port(port):
    # Match ds4-server's listener: TIME_WAIT from the certified retired engine
    # is reusable, while a live listener on the same address still refuses.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', port))


def source_digest(directory):
    files = sorted(p for p in directory.rglob('*') if p.is_file() and
                   (p.suffix in {'.c', '.h', '.m', '.metal', '.inc'} or p.name == 'Makefile'))
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(directory).as_posix().encode() + b'\0')
        digest.update(bytes.fromhex(sha256(file)))
    return digest.hexdigest()


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
        directory = ROOT / 'engines/ds4-mac'
        binary = directory / 'ds4-server'
        source_pin = source_digest(directory)
        build_record = Path(cfg['runtime_root']) / 'state' / 'ds4-build.json'
        previous = json.loads(build_record.read_text()) if build_record.is_file() else {}
        current = (binary.is_file() and previous.get('source_sha256') == source_pin
                   and previous.get('binary_sha256') == sha256(binary))
        if not current and not check:
            if not shutil.which('clang') or not shutil.which('make'):
                raise ValueError('Install Xcode command line tools: xcode-select --install')
            build_cfg = {**cfg, 'host_budget_gib': 2, 'min_available_gib': 4}
            flags = '-O3 -ffast-math -g -mcpu=native -Wall -Wextra -std=c99 -DDS4_NATIVE_FRONTIER_ENABLE=1'
            OwnedProcess(build_cfg, ['make', '-j2', 'ds4-server', 'CFLAGS=' + flags], 'build-mac-ds4',
                         cwd=ROOT / 'engines/ds4-mac', timeout=900, scope='ds4-mac').wait()
            if source_digest(directory) != source_pin:
                raise ValueError('Mac source changed during its build; binary is unqualified')
    else:
        binary = ROOT / 'engines/mlx-serve/zig-out/bin/mlx-serve'
        if not binary.exists() and not check:
            zig = zig or shutil.which('zig')
            if not zig:
                raise ValueError('Install a portable Zig 0.17 nightly from https://ziglang.org/download/ and pass --zig <path>; no system installation is performed')
            build_cfg = {**cfg, 'host_budget_gib': 4, 'min_available_gib': 4}
            OwnedProcess(build_cfg, [zig, 'build', '-Doptimize=ReleaseFast', '-j2'], 'build-mlx-serve',
                         cwd=ROOT / 'engines/mlx-serve', timeout=1800, scope='mlx-serve').wait()
    manifest = {'backend': backend, 'ready': binary.is_file(), 'binary': str(binary),
                'binary_sha256': sha256(binary) if binary.is_file() else None}
    if backend == 'ds4':
        manifest['renderer_sha256'] = sha256(ROOT / 'engines/ds4-mac/ds4_server.c')
        manifest['source_sha256'] = source_pin
        manifest['ready'] = manifest['ready'] and (current or not check)
    if not check:
        atomic_json(Path(cfg['runtime_root']) / 'state' / (backend + '-build.json'), manifest)
    return manifest


def addresses():
    import psutil
    # Internet Sharing and bridged Ethernet put the usable address on a bridge,
    # which networksetup's physical-port list does not necessarily include.
    states = psutil.net_if_stats()
    inactive = set()
    try:
        links = subprocess.check_output(['/sbin/ifconfig', '-a'], text=True, timeout=5)
        for block in re.split(r'(?m)(?=^[^\s:]+:)', links):
            if re.search(r'\bstatus:\s+inactive\b', block):
                inactive.add(block.partition(':')[0])
    except (OSError, subprocess.SubprocessError):
        pass
    candidates = []
    for device, rows in psutil.net_if_addrs().items():
        if device not in states or not states[device].isup or device in inactive:
            continue
        for row in rows:
            if row.family != socket.AF_INET:
                continue
            address = ipaddress.ip_address(row.address)
            if address.is_loopback or address.is_unspecified or address.is_link_local or address.is_multicast:
                continue
            candidates.append((0 if device.startswith('bridge') else 1, device, str(address)))
    return list(dict.fromkeys(row[2] for row in sorted(candidates))) or ['127.0.0.1']


class Receiver:
    def __init__(self, cfg, backend, build):
        self.cfg, self.backend, self.build = cfg, backend, build
        self.lock = threading.RLock()
        self.engine = None
        self.engine_mode = None
        self.engine_budget = None
        self.peer = None
        self.qualification = {}
        self.token_file = Path(cfg['runtime_root']) / 'state' / 'pair.token'
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        supplied = cfg.get('pair_token')
        if supplied is not None and (not isinstance(supplied,str) or not re.fullmatch(r'[A-Za-z0-9_-]{16,128}',supplied)):
            raise ValueError('Pairing token must contain 16–128 URL-safe letters, digits, _ or -')
        if supplied and self.token_file.exists() and self.token_file.read_text().strip() != supplied:
            raise ValueError('Stored token differs; use a separate runtime directory for another pairing')
        if not self.token_file.exists():
            with self.token_file.open('x', encoding='utf-8') as stream:
                stream.write((supplied or secrets.token_urlsafe(32)) + '\n')
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
            receipt = json.loads(self.engine.receipt.read_text()) if self.engine.receipt.is_file() else {}
            if not receipt.get('cleanup_confirmed'):
                error = RuntimeError('Mac backend cleanup unconfirmed; replacement refused')
                error.owned_cleanup_uncertain = True
                raise error
            self.engine = None
            self.engine_mode = None
            self.engine_budget = None

    def pair(self, row, client_ip):
        allowed = {'producer_sha256', 'producer_source_revision', 'producer_ipv4', 'broker_port', 'tier'}
        keep=row.get('yoco_keep',0);version=row.get('yoco_state_version',0)
        if type(keep) is not int or type(version) is not int or (keep,version) not in ((0,0),(140,1)):
            raise ValueError('Unsupported paired YOCO state provenance')
        if keep:allowed|={'yoco_keep','yoco_state_version'}
        if set(row) != allowed or not re.fullmatch('[0-9a-f]{64}', row['producer_sha256']) or not re.fullmatch('[0-9a-f]{40}', row['producer_source_revision']):
            raise ValueError('Invalid producer identity')
        if str(ipaddress.IPv4Address(row['producer_ipv4'])) != client_ip:
            raise ValueError('Producer address must match the authenticated peer connection')
        if type(row['broker_port']) is not int or not 1 <= row['broker_port'] <= 65535 or row['tier'] not in ('accurate', 'fast', 'ultra', 'hyper'):
            raise ValueError('Invalid broker port or native tier')
        self.close()
        self.peer = {**row, 'receiver_sha256': self.build['binary_sha256'], 'status': 'candidate'}
        self.qualification = {}
        return self.peer

    def ready(self, native):
        if self.engine and self.engine.proc.poll() is None and self.engine_mode == native:
            return
        previous_budget = self.engine_budget
        self.close()
        cfg = replacement_budget(self.cfg, 400 if self.backend == 'ds4' else 128, previous_budget)
        port = cfg.get('mac_engine_port', 8181)
        check_backend_port(port)
        env = dict(cfg.get('env_overrides', {}))
        binary = self.build['binary']
        if self.backend == 'mlx-serve':
            from .native_broker import vendor
            vendor()
            from studio.mlx_serve import mlx_serve_argv
            preset = dict(cfg.get('mlx_preset') or {})
            preset['model_path'] = str(path(cfg['model']))
            command = mlx_serve_argv({'mode': 'local', 'rtx': preset}, port=port,
                                     bind_address='127.0.0.1', binary=binary)
            if not command:
                raise ValueError('Fresh mlx-serve binary not found')
        else:
            model = path(cfg['model'])
            mtp = path(cfg['mtp_model']) if cfg.get('mtp_model') else model.with_name('DeepSeek-V4.1-Flash-1017-DSpark-Q4.gguf')
            if not mtp.is_file():
                raise ValueError('Set the existing DSpark support GGUF with --mtp-model')
            from .native_broker import vendor, model_receipt
            vendor()
            verified_model = json.loads(model_receipt(cfg).read_text())['receipts'][model.name]['sha256']
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
            env['DS4_NATIVE_SESSION_HOST_BYTES'] = str(int(cfg['host_budget_gib'] * (1 << 30)))
            env['DS4_NATIVE_SESSION_HOST_FLOOR_BYTES'] = str(int(cfg['min_available_gib'] * (1 << 30)))
            if native:
                if not self.peer:
                    raise ValueError('Pair the RTX producer first')
                model_stat = model.stat()
                qualification = {'schema': 'dsv41-native-session-qualification-v2',
                    'model_sha256': verified_model,
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
                env['DS4_NATIVE_SESSION_YOCO_KEEP']=str(self.peer.get('yoco_keep',0))
        cfg['env_overrides'] = env
        # DS4COMBO2 liveness: the Mac engine has no lifetime cap; the guard's memory limits stay.
        self.engine = OwnedProcess(cfg, command, 'mac-' + self.backend, cwd=Path(cfg['runtime_root']),
                                   unbounded=True, scope='ds4-mac' if self.backend == 'ds4' else 'mlx-serve')
        self.engine_mode = native
        self.engine_budget = cfg['host_budget_gib']
        # Owner rule (no fixed timeouts): readiness lives while the backend tree shows progress.
        window = Window(readiness_seconds(cfg))
        while not window.expired():
            if self.engine.proc.poll() is not None:
                self.engine.wait()
            try:
                http_json(f'http://127.0.0.1:{port}/v1/models', timeout=2)
                return
            except (OSError, RuntimeError):
                window.observe(activity(self.engine.proc.pid, self.engine.log))
                time.sleep(.25)
        self.close()
        raise TimeoutError('Mac backend readiness made no progress for %.0fs (log, I/O and CPU unchanged; %.0fs since launch)'
                           % (window.idle(), window.elapsed()))

    def chat(self, body, *, qualification=None):
        if qualification is None and self.backend == 'ds4' and (not self.peer or self.peer.get('status') != 'qualified'):
            raise ValueError('Complete the real native/pair parity handshake first')
        if self.backend == 'ds4':
            from .paired_generation import decode_request
            body = decode_request(body)
        native = self.backend == 'ds4' and qualification != 'native'
        self.ready(native)
        body = dict(body, stream=False)
        from .pairing import conversation_key
        body['prompt_cache_key'] = conversation_key(body)
        body.setdefault('include_previous_reasoning', True)
        response = http_json(f'http://127.0.0.1:{self.cfg.get("mac_engine_port", 8181)}/v1/chat/completions',
                             body, timeout=None)   # silent until done (stream=False); the owned engine bounds it
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

    def events(self, body):
        if self.backend != 'ds4' or not self.peer or self.peer.get('status') != 'qualified':
            raise ValueError('Complete native pairing before streaming')
        from .paired_generation import decode_request
        body = decode_request(body)
        self.ready(True)
        from .pairing import conversation_key
        from .streaming import events
        body = {**body,'stream':True,'prompt_cache_key':conversation_key(body)}
        body.setdefault('include_previous_reasoning',True)
        yield from events(f'http://127.0.0.1:{self.cfg.get("mac_engine_port",8181)}/v1/chat/completions',
                          body, timeout=self.cfg['request_timeout_s'])


def wait_for_receiver(cfg, job):
    """Relay only the owned receiver's user-facing startup lines to its terminal."""
    window = Window(readiness_seconds(cfg))   # no-progress window (owner rule), not a 60 s cap
    bind = cfg.get('mac_bind','0.0.0.0')
    host = '127.0.0.1' if bind == '0.0.0.0' else bind
    while not window.expired():
        if job.proc.poll() is not None:
            job.wait()
            raise RuntimeError('Receiver exited before startup')
        try:
            ready = http_json(f'http://{host}:{cfg.get("mac_port",8180)}/health',timeout=2)
            if ready.get('ready') and job.log.is_file():
                with job.log.open('rb') as stream:
                    lines = stream.read(131072).decode('utf-8',errors='replace').splitlines()
                visible = [line for line in lines if line.startswith(('MAC ENDPOINT:','RTX command:'))]
                if any(line.startswith('RTX command:') for line in visible):
                    for line in visible:
                        print(line,flush=True)
                    return
        except (OSError,RuntimeError):
            pass
        window.observe(activity(job.proc.pid, job.log))
        time.sleep(.25)
    raise TimeoutError('Receiver startup/address announcement made no progress for %.0fs (%.0fs since launch)'
                       % (window.idle(), window.elapsed()))


def serve(cfg, backend, build):
    receiver = Receiver(cfg, backend, build)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
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
                peer = receiver.peer if receiver.peer and receiver.peer.get('status') == 'qualified' else {}
                self.reply(200, {'ready': True, 'backend': backend, 'receiver_sha256': build['binary_sha256'],
                                 'paired_producer_sha256': peer.get('producer_sha256'),
                                 'paired_tier': peer.get('tier'), 'yoco_state_version':1,
                                 'paired_yoco_keep':peer.get('yoco_keep',0)})
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
                        if body.get('stream'):
                            from .streaming import send
                            send(self,receiver.events(body))
                            return
                        result = receiver.chat(body)
                    else:
                        raise ValueError('Unknown receiver operation')
                self.reply(200, result)
            except (ValueError, KeyError, OSError, RuntimeError, TimeoutError, MemoryError) as exc:
                self.reply(409, {'error': str(exc)})

    port = cfg.get('mac_port', 8180)
    server = ThreadingHTTPServer((cfg.get('mac_bind', '0.0.0.0'), port), Handler)
    server.daemon_threads = True
    announced = addresses() if cfg.get('mac_bind','0.0.0.0') == '0.0.0.0' else [cfg['mac_bind']]
    for address in announced:
        print(f'MAC ENDPOINT: http://{address}:{port}', flush=True)
    print(f'RTX command: ./slstack.ps1 rtx --mac http://{announced[0]}:{port} --pair-token ' + receiver.token +
          ' --model "<rtx-model.gguf>" --comfyui-models "<models>"', flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        server.server_close()
        receiver.close()
