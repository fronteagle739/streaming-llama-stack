"""Launch glue around Studio's delivered mode3 broker and original-Q4 producer."""
from contextlib import ExitStack
import hashlib
import json
import os
import re
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
    model = path(cfg['model'])
    from .gguf_header import read_header, validate_dsv41_types
    header=read_header(model)
    packed=header['metadata'].get('deepseek41.expert_layout')
    if packed is not None:
        env=cfg.get('env_overrides',{})
        if (cfg.get('native_fp4_prefill_only') is not True or
            not env.get('LLAMA_DSV41_FLASHINFER_NATIVE_FFN_DLL') or
            env.get('LLAMA_DSV41_STORED_MXFP4')!='1' or
            env.get('LLAMADECODE_DS4_DECODE')!='0' or env.get('LLAMADECODE_PREFILL_ROUTED')!='0'):
            raise ValueError('Kernel-native expert layout requires explicit prefill-only mode, its native bridge, full-stage reads and disabled ds4 decode')
    dense=header['metadata'].get('deepseek41.dense_layout')
    if dense is not None:
        env=cfg.get('env_overrides',{})
        if env.get('LLAMA_DSV41_DENSE_FP8')!='1' or not env.get('LLAMA_DSV41_DENSE_FP8_DLL') or env.get('LLAMA_DSV41_DENSE_BF16_STORE')!='1':
            raise ValueError('Native FP8 dense requires its explicit kernel and BF16 boundary profile')
    validate_dsv41_types(header,expert_layout=packed,dense_layout=dense)
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
        row = data.get('receipts', {}).get(model.name, {})
        if (re.fullmatch('[0-9a-f]{64}', str(row.get('sha256', ''))) and row.get('final_identity') == identity
                and (not cfg.get('model_sha256') or row['sha256']==cfg['model_sha256'])):
            if candidate != target:
                atomic_json(target, {'receipts': {model.name: row},
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
    if (cfg.get('model_sha256') and digest != cfg['model_sha256']) or (after.st_size, after.st_dev, after.st_ino, after.st_mtime_ns) != tuple(identity.values()):
        raise ValueError('Selected model hash/identity differs; native handoff refused')
    atomic_json(target, {'receipts': {model.name:
                                     {'sha256': digest, 'final_identity': identity}}, 'validation': 'full SHA-256'})
    return target


def model_digest(cfg, receipt=None):
    receipt = receipt or model_receipt(cfg)
    row = json.loads(Path(receipt).read_text(encoding='utf-8-sig'))['receipts'][path(cfg['model']).name]
    digest = row['sha256']
    if not re.fullmatch('[0-9a-f]{64}', digest):raise ValueError('Invalid selected model digest')
    return digest


class ProgressGuardedTransport:
    """HTTP transfers inherit the producer's progress guard, without a total cap.

    The pinned broker reads operation_seconds for asyncio.timeout(); the actual
    producer still retains its finite stall window for pipe, hashing and GPU work.
    Bound methods delegate to that original owner, including close and poisoning.
    """
    operation_seconds = None

    def __init__(self, producer):
        self.owner = producer

    def __getattr__(self, name):
        return getattr(self.owner, name)


class OwnedProducer:
    def __init__(self, cfg, request, cancel, *, on_start=None):
        vendor()
        from studio.dsv41_session_contract import CONTEXT, require
        from studio.dsv41_session_process import Producer
        from .native_profiles import profile
        from studio.dsv41_windows_runner import _Native
        self.cfg, self.request, self.cancel = cfg, {k:v for k,v in request.items() if k!='vision'}, cancel
        self.stack = ExitStack()
        self.producer = None
        self.loading_process = None
        self.prefill_plan = {}
        self.logical_tokens = len(request['tokens']) // 4
        self.progress_started = time.monotonic()
        self.progress_state = {}
        self.progress_lock = threading.Lock()
        self.closed = False
        self.timer = None
        self.attempted = False
        self.failure_cleanup = None
        self._progress_cleanup=on_start(self) if on_start else None
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
            # Use the exact single-engine fold/direct-open/cache configuration.
            # Only the DSpark capture and negotiated export provenance differ.
            from .combo_producer import prefill_arguments
            from .memory_counters import available_physical_bytes
            prefill_cfg = {**cfg, 'context': CONTEXT, 'producer_profile': request['tier']}
            # Match the single-engine 1M cache preset. Grow only the used rows;
            # BF16 preserves the existing QAT values and admits the full window.
            prefill_cfg.setdefault('ctx_cache_precision','bf16')
            prefill_cfg.setdefault('ctx_cache_growable',True)
            args, self.prefill_plan = prefill_arguments(prefill_cfg, self.logical_tokens,
                                                       available_physical_bytes(), export_state=True, techniques=request.get('techniques'))
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
            from .yoco import state_version
            version=state_version(keep)
            require(type(request.get('yoco_keep',0)) is int and type(request.get('yoco_state_version',0)) is int
                    and request.get('yoco_keep',0)==keep and request.get('yoco_state_version',0)==version,
                    'Peer YOCO provenance differs from the paired preset')
            if keep:
                require(args['--capture-dspark']=='1' and (args['--fixed-resident-plan']=='1' or args['--pair-ffn-tiles']=='1'),
                        'YOCO transfer requires a fixed mode3 producer')
                args.update({'--yoco-keep':str(keep),'--yoco-remote-state-version':str(version)})
            from .combo_ram import runtime_min_free_gib
            runtime_floor = runtime_min_free_gib(cfg)
            host_bytes = self.prefill_plan['host_bytes']
            floor = int(cfg['min_available_gib'] * (1 << 30))
            desktop_reserve = max(floor, int(cfg.get('producer_desktop_reserve_gib',.5) * (1 << 30)))
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
            if not env.get('LLAMADECODE_ABORT_FILE'):env['LLAMADECODE_ABORT_FILE']=str(directory/'guard.abort')
            libraries = [str(path(p)) for p in cfg.get('producer_library_directories', [])]
            env['PATH'] = os.pathsep.join([str(binary.parent), *libraries, env.get('PATH', '')])
            from .combo_producer import stall_seconds   # local: combo_producer imports this module
            # DS4COMBO2 liveness: no fixed lease_minutes cap; the Producer's no-progress window
            # (stall_seconds) ends a stalled session, and idle_seconds still retires an idle one.
            self.timer = None
            # Match single-engine residency protection; the outer controller
            # budget includes its separate measured margin.
            from .working_set import protect_working_set
            working_minimum = host_bytes
            def protect(job):
                return protect_working_set(job, working_minimum, floor, int(cfg['host_budget_gib']*(1<<30)),
                                           receipt=directory / 'working-set.json')
            self.attempted = True
            prefill_started=time.monotonic()
            owner=self
            class PublishingProducer(Producer):
                def _initialize(process,*a,**kw):
                    owner.loading_process=process
                    try:return super()._initialize(*a,**kw)
                    finally:owner.loading_process=None
            self.producer = PublishingProducer(request['session'], argv, env, directory, host_bytes=host_bytes,
                                     runtime_min_free_gib=runtime_floor,
                                     physical_floor=floor, operation_seconds=min(600, cfg['request_timeout_s']), stall_seconds=stall_seconds(cfg),
                                     idle_seconds=cfg['lease_minutes'] * 60, native=native,
                                     cancelled=cancelled, after_resume=protect)
            # Actual RTX operation clocks, separate from the Mac's transfer /
            # import / acknowledgment intervals. Initial time includes startup.
            self.prefill_timings=[dict(operation='startup_and_initial_prefill',
                tokens=len(request['tokens'])//4,seconds=time.monotonic()-prefill_started)]
            timing_file=directory/'prefill-timings.json'
            atomic_json(timing_file,self.prefill_timings)
            native_request=self.producer.request
            def timed_request(row):
                started=time.monotonic();result=native_request(row)
                operation=row.get('operation')
                self.prefill_timings.append(dict(operation=operation,
                    tokens=len(row.get('tokens',[])),seconds=time.monotonic()-started))
                atomic_json(timing_file,self.prefill_timings)
                return result
            self.producer.request=timed_request
            native_export=self.producer.begin_export
            def timed_export(row):
                started=time.monotonic();result=native_export(row)
                self.prefill_timings.append(dict(operation='export_prepare',tokens=0,
                    bytes=result.get('bytes'),seconds=time.monotonic()-started))
                atomic_json(timing_file,self.prefill_timings);return result
            self.producer.begin_export=timed_export
            cp = self.producer.ready['checkpoint']
            self.producer.stage_vision=stage_vision
            require(self.producer.ready.get('yoco_keep',0)==keep and self.producer.ready.get('yoco_state_version',0)==version,
                    'Producer did not negotiate the requested YOCO state version')
            require(cp['context'] == CONTEXT and cp['position'] == len(request['tokens']) // 4
                    and cp['tokens_sha256'] == request['tokens_sha256'], 'Initial producer checkpoint differs')
            self.producer.ready['profile'] = settings
            atomic_json(directory / 'launch.json', {'producer_sha256': pin, 'settings': settings,
                        'argv':argv,'prefill_plan':self.prefill_plan,
                        'backend':'streaming-llama single-engine prefill then mode3 export',
                        'source_revision': cfg['producer_source_revision'], 'model_copied': False,
                        'host_bytes': host_bytes, 'physical_floor': floor,
                        'working_set': self.producer.resume_admission})
            self.producer = ProgressGuardedTransport(self.producer)
        except BaseException as exc:
            self.failure_cleanup = getattr(exc, 'owned_cleanup', None)
            if hasattr(self, 'directory') and (diagnostic := getattr(exc, 'owned_diagnostic', None)):
                (self.directory / 'stderr.log').write_bytes(diagnostic)
            self.close()
            raise

    def prefill_progress(self):
        from types import SimpleNamespace
        from .combo_single import prefill_snapshot
        process=self.producer or self.loading_process
        active=SimpleNamespace(process=process,prefill_plan=self.prefill_plan)
        with self.progress_lock:
            return prefill_snapshot(SimpleNamespace(producer=active),
                dict(new_tokens=self.logical_tokens,prefill_started=self.progress_started),
                self.progress_state,time.monotonic())

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
        if self._progress_cleanup:self._progress_cleanup()
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
    def factory(request,cancel):
        def publish(owner):
            starting[request['session']]=owner
            return lambda:starting.pop(request['session'],None)
        try:
            return OwnedProducer(cfg,request,cancel,on_start=publish)
        except BaseException as error:
            import traceback
            traceback.print_exc()
            print('NATIVE_SESSION_START_FAILED '+json.dumps({'session':request.get('session'),
                  'error':str(error)}),flush=True)
            raise
    starting = {}
    class ProgressBroker(Broker):
        def status(self,sid):
            row=super().status(sid)
            owner=starting.get(sid)
            if owner and (progress:=owner.prefill_progress()):
                row.update(prefill_processed=progress['tokens_done'],prefill_total=progress['tokens_total'],
                           prefill_elapsed_ms=int(progress['elapsed_s']*1000))
            return row
    broker = ProgressBroker(factory)

    @asynccontextmanager
    async def lifespan(app):
        yield
        broker.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
    @app.exception_handler(HTTPException)
    async def rejected_request(request,error):
        from fastapi.responses import JSONResponse
        print('NATIVE_SESSION_HTTP_REFUSED '+json.dumps({'path':request.url.path,
              'status':error.status_code,'detail':error.detail}),flush=True)
        return JSONResponse({'detail':error.detail},status_code=error.status_code)

    def authorize(x_studio_token: str = Header(default='')):
        if not hmac.compare_digest(token, x_studio_token):
            raise HTTPException(401, 'Pairing required')

    app.include_router(create_router(broker, authorize))

    @app.get('/health')
    def health():
        return {'ready': True, 'protocol': 'existing-mode3', 'producer_sha256': cfg['producer_sha256']}

    uvicorn.run(app, host=cfg.get('broker_host', '0.0.0.0'), port=cfg.get('broker_port', 8090),
                access_log=False, log_level='warning')
