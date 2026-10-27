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


def hash_progress(report, interval=5.0):
    """A sha256() progress callback that reports percent, GB and rate at most every interval s."""
    started, last = time.monotonic(), [None]
    def update(done, total):
        now = time.monotonic()
        if done < total and last[0] is not None and now - last[0] < interval:
            return
        last[0] = now
        rate = done / 1e9 / max(now - started, 1e-6)
        report(f'Verifying original model SHA-256: {100 * done / max(total, 1):.1f}% '
               f'({done / 1e9:.1f} of {total / 1e9:.1f} GB, {rate:.2f} GB/s)')
    return update


def model_receipt(cfg, report=None):
    """Reuse a verified original receipt only while all physical identities match.

    The receipt is a correctness gate: the producer (dsv41-prefill --copy-receipt) refuses
    to start without a completed full-SHA receipt for the main weights. A missing receipt is
    therefore hashed here, blocking; report(message), when given, receives the progress."""
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
    message = 'Verifying original model SHA-256 (read only, once per model file; this can take several minutes)'
    if report is None:
        print(message, flush=True)
    else:
        report(message)
    digest = sha256(model, hash_progress(report) if report else None)
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
        self.cfg, self.request, self.cancel = cfg, {k:v for k,v in request.items() if k!='vision'}, cancel
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
            watcher = HoldWatcher(cfg.get('hold_file') or Path(cfg['runtime_root']) / 'state' / 'gpu.holds', identity, task=cfg.get('task', 'DS4STACK'))
            def cancelled():
                if cancel.is_set() or watcher.reason():
                    return True
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
            self.vision=[]
            self.vision_leases=set()
            def stage_vision(images,base):
                from .paired_vision import materialize
                descriptor=materialize(directory/'vision',images,previous=self.vision,base=base)
                rows=json.loads(descriptor.read_text())
                for row in rows:
                    if row['file'] not in self.vision_leases:
                        self.stack.enter_context(native.read_lease(Path(row['file'])))
                        self.vision_leases.add(row['file'])
                self.stack.enter_context(native.read_lease(descriptor))
                self.vision=[{k:v for k,v in row.items() if k!='data'} for row in images]
                return str(descriptor)
            if request.get('vision'):
                args['--vision-spans-file']=stage_vision(request['vision'],0)
            keep=cfg.get('yoco_keep',0)
            require(type(keep) is int and keep in (0,140), 'Unsupported paired YOCO window')
            require(request.get('yoco_keep',0)==keep and request.get('yoco_state_version',0)==(1 if keep else 0),
                    'Peer YOCO provenance differs from the paired preset')
            if keep:
                require(args['--capture-dspark']=='1' and (args['--fixed-resident-plan']=='1' or args['--pair-ffn-tiles']=='1'),
                        'YOCO transfer requires a fixed mode3 producer')
                args.update({'--yoco-keep':str(keep),'--yoco-remote-state-version':'1'})
            from .combo_ram import runtime_min_free_gib
            runtime_floor = runtime_min_free_gib(cfg)
            host_bytes = min(8 << 30, int(cfg['host_budget_gib'] * (1 << 30)))
            floor = int(cfg['min_available_gib'] * (1 << 30))
            desktop_reserve = max(floor, int(cfg.get('producer_desktop_reserve_gib',8) * (1 << 30)))
            if native.memory() < host_bytes + desktop_reserve:
                raise MemoryError('Native producer physical host admission refused')
            if native.gpu(0, cfg['gpu_uuid']) < int(args['--job-device-bytes']) + int(args['--gpu-reserve-bytes']):
                raise MemoryError('Native producer physical GPU admission refused')
            args.update({'--model': str(path(cfg['model'])), '--copy-receipt': str(receipt),
                         '--tokens-le32': str(tokens), '--build-sha': pin, '--session-id': request['session'],
                         '--receipt': str(directory / 'producer.json'), '--job-host-bytes': str(host_bytes),
                         '--runtime-min-free-gib': str(runtime_floor),
                         '--carry': str(directory / 'carry-not-created'), '--payload': str(directory / 'payload-not-created')})
            argv = [str(binary), *[str(v) for item in args.items() for v in item]]
            env = environment(cfg, 'rtx-producer')
            libraries = [str(path(p)) for p in cfg.get('producer_library_directories', [])]
            env['PATH'] = os.pathsep.join([str(binary.parent), *libraries, env.get('PATH', '')])
            from .combo_producer import stall_seconds   # local: combo_producer imports this module
            # DS4COMBO2 liveness: no fixed lease_minutes cap; the Producer's no-progress window
            # (stall_seconds) ends a stalled session, and idle_seconds still retires an idle one.
            self.timer = None
            # Leave 256 MiB within the aggregate for broker/metadata. The native
            # engine retains its full allocation/RSS cap; this is a hard minimum,
            # never a hard maximum that could force pages out.
            from .working_set import protect_working_set
            working_minimum = host_bytes - (256 << 20)
            def protect(job):
                return protect_working_set(job, working_minimum, floor, host_bytes,
                                           receipt=directory / 'working-set.json')
            self.attempted = True
            self.producer = Producer(request['session'], argv, env, directory, host_bytes=host_bytes,
                                     runtime_min_free_gib=runtime_floor,
                                     physical_floor=floor, operation_seconds=min(600, cfg['request_timeout_s']), stall_seconds=stall_seconds(cfg),
                                     idle_seconds=cfg['lease_minutes'] * 60, native=native,
                                     cancelled=cancelled, after_resume=protect)
            cp = self.producer.ready['checkpoint']
            self.producer.stage_vision=stage_vision
            require(self.producer.ready.get('yoco_keep',0)==keep and self.producer.ready.get('yoco_state_version',0)==(1 if keep else 0),
                    'Producer did not negotiate the requested YOCO state version')
            require(cp['context'] == CONTEXT and cp['position'] == len(request['tokens']) // 4
                    and cp['tokens_sha256'] == request['tokens_sha256'], 'Initial producer checkpoint differs')
            self.producer.ready['profile'] = settings
            atomic_json(directory / 'launch.json', {'producer_sha256': pin, 'settings': settings,
                        'source_revision': cfg['producer_source_revision'], 'model_copied': False,
                        'host_bytes': host_bytes, 'physical_floor': floor,
                        'working_set': self.producer.resume_admission})
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
        self.receipt['paging_monitor'] = 'Pinned shared guard on the owning broker process tree'
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
