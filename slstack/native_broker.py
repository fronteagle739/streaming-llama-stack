"""Launch glue around Studio's delivered mode3 broker and original-Q4 producer."""
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from .config import ROOT, environment, path
from .journal import atomic_json, sha256
from .lease import gpu_lease, HoldWatcher


def vendor():
    location = str(ROOT / 'vendor/studio')
    if location not in sys.path:
        sys.path.insert(0, location)


def model_receipt(cfg):
    """Reuse a verified original receipt only while all physical identities match."""
    vendor()
    from studio.dsv41_session_contract import MODEL_SHA256
    model = path(cfg['model'])
    current = model.stat()
    identity = {'size': current.st_size, 'dev': current.st_dev, 'ino': current.st_ino,
                'mtime_ns': current.st_mtime_ns}
    target = Path(cfg['runtime_root']) / 'state' / 'original-model.json'
    candidates = [target]
    if cfg.get('model_receipt'):
        candidates.append(path(cfg['model_receipt']))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        data = json.loads(candidate.read_text(encoding='utf-8-sig'))
        row = data.get('receipts', {}).get('DeepSeek-V4.1-Flash-1017-Q4-v41.gguf', {})
        if row.get('sha256') == MODEL_SHA256 and row.get('final_identity') == identity:
            if candidate != target:
                atomic_json(target, {'receipts': {'DeepSeek-V4.1-Flash-1017-Q4-v41.gguf': row},
                                     'validation': 'prior SHA receipt and unchanged physical identity',
                                     'source_receipt_sha256': sha256(candidate)})
            return target
    # A fresh install hashes the model once; weights are never copied.
    print('Verifying original model SHA-256 (read only; this can take several minutes)', flush=True)
    digest = sha256(model)
    after = model.stat()
    if digest != MODEL_SHA256 or (after.st_size, after.st_dev, after.st_ino, after.st_mtime_ns) != tuple(identity.values()):
        raise ValueError('Original 1017 model hash/identity differs; native handoff refused')
    atomic_json(target, {'receipts': {'DeepSeek-V4.1-Flash-1017-Q4-v41.gguf':
                                     {'sha256': digest, 'final_identity': identity}}, 'validation': 'full SHA-256'})
    return target


class OwnedProducer:
    def __init__(self, cfg, request, cancel):
        vendor()
        from studio.dsv41_session_contract import CONTEXT, require
        from studio.dsv41_session_process import Producer
        from .native_profiles import profile
        from studio.dsv41_windows_runner import _Native
        self.cfg, self.request, self.cancel = cfg, request, cancel
        self.stack = ExitStack()
        self.producer = None
        self.closed = False
        self.timer = None
        self.attempted = False
        self.failure_cleanup = None
        try:
            binary = path(cfg['llama_producer'])
            pin = sha256(binary)
            if pin != cfg['producer_sha256']:
                raise ValueError('Trusted producer binary pin changed')
            receipt = model_receipt(cfg)
            identity = self.stack.enter_context(gpu_lease(cfg))
            watcher = HoldWatcher(cfg.get('hold_file') or Path(cfg['runtime_root']) / 'state' / 'gpu.holds', identity)
            from .memory_counters import PagefilePressure
            self.pressure = PagefilePressure()
            next_pressure_check = 0

            def cancelled():
                nonlocal next_pressure_check
                if cancel.is_set() or watcher.reason():
                    return True
                if time.monotonic() >= next_pressure_check:
                    reason = self.pressure.check()
                    next_pressure_check = time.monotonic() + .5
                    if reason:
                        raise MemoryError(reason)
                return False

            native = _Native()
            directory = Path(cfg['runtime_root']) / 'native-sessions' / request['session']
            directory.mkdir(parents=True, exist_ok=False)
            self.directory = directory
            self.stack.enter_context(native.directory(directory))
            for file in (binary, path(cfg['model']), receipt):
                self.stack.enter_context(native.read_lease(file))
            tokens = directory / 'initial.tokens.le32'
            with tokens.open('xb') as output:
                output.write(request['tokens'])
            if hashlib.sha256(request['tokens']).hexdigest() != request['tokens_sha256']:
                raise ValueError('Initial native token identity differs')
            settings = profile(request['tier'], request.get('techniques'))
            args = dict(settings['arguments'])
            host_bytes = min(8 << 30, int(cfg['host_budget_gib'] * (1 << 30)))
            floor = int(cfg['min_available_gib'] * (1 << 30))
            if native.memory() < host_bytes + floor:
                raise MemoryError('Native producer physical host admission refused')
            if native.gpu(0, cfg['gpu_uuid']) < int(args['--job-device-bytes']) + int(args['--gpu-reserve-bytes']):
                raise MemoryError('Native producer physical GPU admission refused')
            args.update({'--model': str(path(cfg['model'])), '--copy-receipt': str(receipt),
                         '--tokens-le32': str(tokens), '--build-sha': pin, '--session-id': request['session'],
                         '--receipt': str(directory / 'producer.json'), '--job-host-bytes': str(host_bytes),
                         '--carry': str(directory / 'carry-not-created'), '--payload': str(directory / 'payload-not-created')})
            argv = [str(binary), *[str(v) for item in args.items() for v in item]]
            env = environment(cfg, 'rtx-producer')
            libraries = [str(path(p)) for p in cfg.get('producer_library_directories', [])]
            env['PATH'] = os.pathsep.join([str(binary.parent), *libraries, env.get('PATH', '')])
            self.timer = threading.Timer(cfg['lease_minutes'] * 60, cancel.set)
            self.timer.daemon = True
            self.timer.start()
            self.attempted = True
            self.producer = Producer(request['session'], argv, env, directory, host_bytes=host_bytes,
                                     physical_floor=floor, operation_seconds=min(600, cfg['request_timeout_s']),
                                     idle_seconds=cfg['lease_minutes'] * 60, native=native,
                                     cancelled=cancelled)
            cp = self.producer.ready['checkpoint']
            require(cp['context'] == CONTEXT and cp['position'] == len(request['tokens']) // 4
                    and cp['tokens_sha256'] == request['tokens_sha256'], 'Initial producer checkpoint differs')
            self.producer.ready['profile'] = settings
            atomic_json(directory / 'launch.json', {'producer_sha256': pin, 'settings': settings,
                        'source_revision': cfg['producer_source_revision'], 'model_copied': False,
                        'host_bytes': host_bytes, 'physical_floor': floor})
        except BaseException as exc:
            self.failure_cleanup = getattr(exc, 'owned_cleanup', None)
            self.close()
            raise

    def close(self):
        if self.closed:
            return self.receipt
        proof = self.producer.close() if self.producer is not None else self.failure_cleanup or {'creation_attempted': False}
        if self.attempted and not all(proof.get(k) for k in ('owned_job_empty', 'owned_process_signaled', 'owned_pid_gone')):
            raise RuntimeError('Native producer cleanup unconfirmed; ownership retained')
        if self.timer:
            self.timer.cancel()
        self.receipt = {'session': self.request['session'], 'cleanup': proof}
        if hasattr(self, 'pressure'):
            self.receipt['system_pagefile_pressure'] = self.pressure.receipt()
        if hasattr(self, 'directory'):
            if self.producer is not None:
                (self.directory / 'stderr.log').write_bytes(self.producer.diagnostic)
            atomic_json(self.directory / 'cleanup.json', self.receipt)
        self.stack.close()
        self.closed = True
        return self.receipt


def serve(cfg):
    vendor()
    import hmac
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, Header, HTTPException
    import uvicorn
    from studio.dsv41_session_broker import Broker, create_router
    token = Path(cfg['pair_token_file']).read_text(encoding='utf-8').strip()
    if len(token) < 16:
        raise ValueError('Explicit pairing token required')
    broker = Broker(lambda request, cancel: OwnedProducer(cfg, request, cancel))

    @asynccontextmanager
    async def lifespan(app):
        yield
        broker.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)

    def authorize(x_studio_token: str = Header(default='')):
        if not hmac.compare_digest(token, x_studio_token):
            raise HTTPException(401, 'Pairing required')

    app.include_router(create_router(broker, authorize))

    @app.get('/health')
    def health():
        return {'ready': True, 'protocol': 'existing-mode3', 'producer_sha256': cfg['producer_sha256']}

    uvicorn.run(app, host=cfg.get('broker_host', '0.0.0.0'), port=cfg.get('broker_port', 8090),
                access_log=False, log_level='warning')
