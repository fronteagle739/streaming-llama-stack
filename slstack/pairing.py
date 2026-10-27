"""Pair the existing native receiver and Studio mode3 broker; no state codec here."""
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit

from .config import ROOT, path
from .engines import http_json
from .journal import atomic_json, sha256
from .process import OwnedProcess

_DEFAULT_TIMEOUT = object()


def endpoint(value):
    value = value if '://' in value else 'http://' + value
    parsed = urlsplit(value)
    if (parsed.scheme != 'http' or not parsed.hostname or parsed.username or parsed.password
            or parsed.path not in ('', '/') or parsed.query or parsed.fragment):
        raise ValueError('Use the receiver HTTP address without credentials, path or query')
    if ':' in parsed.hostname:
        raise ValueError('The existing native transport requires an IPv4 receiver')
    port = parsed.port or 8180
    return f'http://{parsed.hostname}:{port}'


def conversation_key(body):
    """Keep a key across append-only turns; explicit client keys take precedence."""
    if 'prompt_cache_key' in body:
        key = body['prompt_cache_key']
        if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_:-]{1,64}', key):
            raise ValueError('prompt_cache_key must contain 1–64 letters, digits, _, : or -')
        return key
    messages = body.get('messages') or []
    first = next((i for i, item in enumerate(messages) if item.get('role') == 'user'), None)
    if first is None:
        raise ValueError('A conversation needs an initial user message')
    seed = {'model': body.get('model'), 'messages': messages[:first + 1]}
    return 'slstack-' + hashlib.sha256(json.dumps(seed, sort_keys=True, ensure_ascii=False,
                                                separators=(',', ':')).encode()).hexdigest()[:48]


def producer_revision():
    """Use the actual CMake build identity, refusing source edits since that build."""
    info = ROOT / 'build/llama/common/build-info.cpp'
    match = re.search(r'LLAMA_COMMIT\s*=\s*"([0-9a-f]{7,40})"', info.read_text())
    if not match:
        raise ValueError('Producer build identity missing; run install --all')
    revision = subprocess.check_output(['git', 'rev-parse', '--verify', match[1] + '^{commit}'],
                                       cwd=ROOT, text=True, timeout=10).strip()
    result = subprocess.run(['git', 'diff', '--quiet', revision, '--', 'engines/streaming-llama',
                             ':(exclude)engines/streaming-llama/prefill_llama.cpp/tools/server/public'],
                            cwd=ROOT, timeout=10)
    if result.returncode:
        raise ValueError('Producer source changed since its build; rebuild before pairing')
    return revision


class Pair:
    def __init__(self, cfg, status=None):
        self.cfg, self.status = cfg, status or (lambda message: print(message, flush=True))
        self.url = endpoint(cfg['mac'])
        self.broker = None
        self.lease_context = None
        self.connected = False
        self.token_path = Path(cfg['runtime_root']) / 'state' / 'pair.token'
        self.record_path = self.token_path.with_name('rtx-peer.json')
        from .generation import Store
        self.generation = Store(cfg)

    def request(self, route, body=None, timeout=_DEFAULT_TIMEOUT):
        # These operations may stay silent on HTTP while the independently
        # guarded engine advances its load, hash, prefill or decode work.
        if timeout is _DEFAULT_TIMEOUT:
            timeout = None if route in ('/pair/qualification', '/decode/setup', '/v1/chat/completions') else self.cfg['request_timeout_s']
        return http_json(self.url + route, body, timeout,
                         headers={'Authorization': 'Bearer ' + self.token})

    def start(self):
        if self.connected:
            return
        if not self.cfg.get('model') or not path(self.cfg['model']).is_file():
            raise ValueError('Pairing requires the existing RTX DSV4.1 GGUF model path')
        supplied = self.cfg.get('pair_token')
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        if supplied:
            if not isinstance(supplied, str) or not re.fullmatch(r'[A-Za-z0-9_-]{16,128}', supplied):
                raise ValueError('Use the pairing token printed by the Mac receiver')
            if self.token_path.exists() and self.token_path.read_text().strip() != supplied:
                raise ValueError('Existing pairing token differs; use a separate runtime directory to pair another receiver')
            if not self.token_path.exists():
                with self.token_path.open('x', encoding='utf-8') as stream:
                    stream.write(supplied + '\n')
                self.token_path.chmod(0o600)
        if not self.token_path.is_file():
            raise ValueError('Start the Mac once and pass its printed --pair-token; later starts reuse it')
        self.token = self.token_path.read_text().strip()
        if not re.fullmatch(r'[A-Za-z0-9_-]{16,128}', self.token):
            raise ValueError('Stored pairing token is invalid')
        producer = path(self.cfg['llama_producer'])
        if not producer.is_file():
            raise ValueError('Build the streaming producer with install --all before pairing')
        pin = sha256(producer)
        revision = producer_revision()
        deadline = time.monotonic() + self.cfg.get('pair_wait_seconds', 7200)
        while True:
            try:
                health = self.request('/health', timeout=3)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Mac receiver did not start within the pairing wait budget')
                self.status('Waiting for Mac receiver; no GPU lease held')
                time.sleep(45)
        if health.get('backend') != 'ds4':
            raise ValueError('This paired mode requires the native ds4 receiver')
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            target = urlsplit(self.url)
            probe.connect((target.hostname, target.port))
            local_ip = probe.getsockname()[0]
        cfg = {**self.cfg, 'pair_token_file': str(self.token_path), 'producer_sha256': pin,
               'producer_source_revision': revision, 'broker_host': local_ip,
               'broker_port': self.cfg.get('broker_port', 8090)}
        with socket.socket() as probe:
            probe.bind((local_ip, cfg['broker_port']))
        if not cfg.get('producer_library_directories'):
            cache = (ROOT / 'build/llama/CMakeCache.txt').read_text()
            toolkit = re.search(r'^CUDAToolkit_BIN_DIR:PATH=(.+)$', cache, re.M)
            if not toolkit:
                raise ValueError('Built CUDA toolkit location missing; run install --all')
            binaries = Path(toolkit[1].strip())
            cfg['producer_library_directories'] = [str(p) for p in (binaries / 'x64', binaries) if p.is_dir()]
        effective = self.token_path.with_name('broker.local.json')
        try:
            # Reserve the complete native/paired handshake, including Mac model
            # startup. The owned broker/producer inherit this ancestor's lease.
            from .lease import gpu_lease, AdmissionRefused
            import psutil
            while True:
                fits = psutil.virtual_memory().available >= (cfg['host_budget_gib'] + cfg['min_available_gib']) * 2**30
                if fits:
                    reservation = gpu_lease(cfg)
                    try:
                        identity = reservation.__enter__()
                        self.lease_context = reservation
                        break
                    except AdmissionRefused:
                        pass
                if time.monotonic() >= deadline:
                    raise TimeoutError('Pairing GPU/RAM admission wait elapsed; no qualification started')
                self.status('Waiting for pairing GPU/RAM admission; no GPU lease held')
                time.sleep(45)
            cfg['_gpu_parent_lease'] = cfg.get('_gpu_parent_lease') or str(
                Path(cfg['runtime_root']) / 'state/leases' / (identity + '.json'))
            atomic_json(effective, cfg)
            self.broker = OwnedProcess(cfg, [sys.executable, '-B', '-m', 'slstack', '--config', str(effective),
                                            'broker-internal'], 'native-broker', timeout=cfg['lease_minutes']*60,
                                       scope='rtx-producer')
            ready_until = time.monotonic() + 30
            while True:
                if self.broker.proc.poll() is not None:
                    self.broker.wait()
                try:
                    ready = http_json(f'http://{local_ip}:{cfg["broker_port"]}/health', timeout=2)
                    if ready.get('producer_sha256') != pin:
                        raise ValueError('Broker producer pin differs')
                    break
                except OSError:
                    if time.monotonic() >= ready_until:
                        raise TimeoutError('Owned broker did not become ready')
                    time.sleep(.2)
            identity = {'producer_sha256': pin, 'producer_source_revision': revision,
                        'producer_ipv4': local_ip, 'broker_port': cfg['broker_port'],
                        'tier': cfg.get('native_tier', 'accurate')}
            keep=cfg.get('yoco_keep',0)
            if type(keep) is not int or keep not in (0,140):raise ValueError('Unsupported paired YOCO window')
            if keep:
                if health.get('yoco_state_version')!=1:raise ValueError('Mac receiver needs YOCO state version1 support')
                identity.update(yoco_keep=keep,yoco_state_version=1)
            saved = json.loads(self.record_path.read_text()) if self.record_path.is_file() else {}
            same = (saved.get('identity') == identity and saved.get('endpoint') == self.url
                    and saved.get('receiver_sha256') == health.get('receiver_sha256')
                    and health.get('paired_producer_sha256') == pin
                    and health.get('paired_tier') == identity['tier']
                    and health.get('paired_yoco_keep',0)==keep)
            if not same:
                self.status('Pairing streaming-llama prefill and Mac decoder')
                self.request('/pair', identity)
                self.request('/pair/qualification', {'mode': 'native'})
                self.request('/pair/qualification', {'mode': 'paired'})
                accepted = self.request('/pair/accept', {})
                atomic_json(self.record_path, {'identity': identity, 'endpoint': self.url,
                    'receiver_sha256': health['receiver_sha256'], 'qualification': accepted})
            self.request('/decode/setup', {})
            self.connected = True
            self.status('Paired Mac decoder ready')
        except BaseException:
            self.close()
            raise

    def decode_request(self, body):
        """Keep every client knob and attach the WebUI's separate toolbar state.

        The receiver must validate this profile; it must never interpret an
        unsupported knob as permission to substitute greedy decoding.
        """
        return {**body, 'prompt_cache_key': conversation_key(body),
                'slstack_generation': self.generation.get()}

    def chat(self, body):
        self.start()
        return self.request('/v1/chat/completions', self.decode_request(body))

    def events(self, body):
        self.start()
        from .streaming import events
        yield from events(self.url + '/v1/chat/completions',
                          {**self.decode_request(body),'stream':True},
                          timeout=None,  # the owned engine fails on stalled work
                          headers={'Authorization':'Bearer ' + self.token})

    def close(self):
        try:
            if self.broker:
                try:
                    self.request('/decode/eject', {}, timeout=60)
                finally:
                    self.broker.stop()
                    receipt = json.loads(self.broker.receipt.read_text()) if self.broker.receipt.exists() else {}
                    if not receipt.get('cleanup_confirmed'):
                        error = RuntimeError('Paired broker cleanup unconfirmed; GPU reservation retained')
                        error.owned_cleanup_uncertain = True
                        if self.lease_context:
                            self.lease_context.__exit__(type(error), error, None)
                            self.lease_context = None
                        raise error
                    self.broker = None
        finally:
            self.connected = False
            if self.lease_context and self.broker is None:
                self.lease_context.__exit__(None, None, None)
                self.lease_context = None
