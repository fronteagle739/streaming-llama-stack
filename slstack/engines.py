from __future__ import annotations
import json
import os
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
from .progress import Window, activity, readiness_seconds


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
        self.media_progress = None
        self.llm = None
        self.remote = None
        self.lock = threading.RLock()
        from .journal import Journal
        self.journal = Journal(cfg['runtime_root'])
        self.attach_entry = None
        self.attach_guard_recorded = False
        self.journal_checked = False
        # Media swaps reuse the DS4COMBO native session export/restore boundary.
        import secrets
        self.combo_session = secrets.token_hex(16)
        self.combo_epoch = 0
        self.combo_sequence = 0
        self.combo_restore = None

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

    def single_engine(self):
        # Default single-RTX endpoint: combo_proxy serving_mode=single_engine (ULTRA producer
        # prefill + in-process ds4 decode). --serving native keeps the ds4-win server.
        return not self.cfg.get('mac') and self.cfg.get('serving', 'single_engine') == 'single_engine'

    def combo_enabled(self):
        # Single-RTX native engine: graceful shutdown and media swaps use the native session boundary.
        return not self.cfg.get('mac') and not self.cfg['lora'] and os.name == 'nt' and not self.single_engine()

    def single(self, operation):
        return http_json(self.cfg['ds4_url'] + '/v1/stack/' + operation, {}, timeout=self.cfg['request_timeout_s'],
                         headers={'Authorization': 'Bearer ' + self.cfg['_control_token']})

    def expert_list(self):
        return Path(self.cfg['runtime_root']) / ('expert-list-' + self.combo_session + '.txt')

    def combo(self, operation, **extra):
        row = dict(session=self.combo_session, owner_epoch=self.combo_epoch, sequence=self.combo_sequence, **extra)
        reply = http_json(self.cfg['ds4_url'] + '/decode/' + operation, row, timeout=self.cfg['request_timeout_s'],
                          headers={'Authorization': 'Bearer ' + self.cfg['_control_token']})
        if (reply.get('status'), reply.get('session'), reply.get('owner_epoch'), reply.get('next_sequence')) != \
                ('complete', self.combo_session, self.combo_epoch, self.combo_sequence + 1):
            raise RuntimeError('Native owner epoch or sequence differs')
        self.combo_sequence += 1
        return reply

    def combo_save(self):
        """Export the live native session into parent-owned locked RAM, then exit gracefully."""
        checkpoint = self.combo('checkpoint')['checkpoint']
        if not checkpoint:
            self.combo_shutdown()
            return None
        from .combo_ram import LockedRam, PAGE
        capacity = -(-int(checkpoint['bytes']) // PAGE) * PAGE
        ram = LockedRam(capacity, session=self.combo_session, host_budget_gib=self.cfg['host_budget_gib'],
                        min_available_gib=self.cfg['min_available_gib'])
        try:
            with ram.lease():
                reply = self.combo('export', mapping=ram.name, capacity=ram.capacity,
                                   checkpoint_sha256=checkpoint['payload_sha256'])
            if reply.get('written_bytes') != int(checkpoint['bytes']):
                raise RuntimeError('Native export wrote an unexpected extent')
            self.combo_shutdown()
        except BaseException:
            ram.close()
            raise
        return ram, checkpoint

    def combo_shutdown(self):
        """Graceful exit that must succeed (media swap): verified by exit code and the engine's log."""
        self.combo('shutdown')
        try:
            code = self.llm.proc.wait(timeout=float(self.cfg.get('native_graceful_shutdown_seconds', 60)))
        except subprocess.TimeoutExpired:
            raise RuntimeError('Native engine did not exit after its graceful shutdown request')
        log = self.llm.log.read_text(errors='replace') if self.llm.log.exists() else ''
        if code or 'DS4COMBO_NATIVE_GRACEFUL_SHUTDOWN_COMPLETE' not in log:
            raise RuntimeError(f'Native graceful shutdown incomplete (guardian exit {code}); see {self.llm.log}')
        self.close()

    def close(self):
        if self.remote:
            try:
                self.remote.close()
            finally:
                self.remote = None
        if self.llm:
            self.status('Unloading model')
            clean = self.llm.proc.poll() == 0
            if self.llm.proc.poll() is None and self.single_engine():
                # The producer exits gracefully (VRAM freed, GPU hold released), then the server ends.
                try:
                    self.single('shutdown')
                    clean = self.llm.proc.wait(timeout=float(self.cfg.get('producer_graceful_shutdown_seconds', 120))) == 0
                except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired):
                    print('Single-engine graceful shutdown failed; stopping the owned job', flush=True)
            if self.llm.proc.poll() is None and self.combo_enabled() and self.combo_sequence > 1:
                # Studio-style exit: the engine drains DirectStorage and frees GPU/host memory.
                try:
                    self.combo('shutdown')
                    clean = self.llm.proc.wait(timeout=float(self.cfg.get('native_graceful_shutdown_seconds', 60))) == 0
                except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired):
                    print('Native graceful shutdown failed; stopping the owned job', flush=True)
            if self.llm.proc.poll() is None and not self.single_engine():
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
        if self.cfg['long_prompt'] and not self.single_engine():
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
        if self.single_engine():
            return self._ready_single()
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
        overrides = self.cfg.setdefault('env_overrides', {})
        overrides['DS4_SERVER_AUTH_TOKEN'] = token
        # DS4COMBO2 liveness: the server has no lifetime cap; the guard's memory/GPU limits stay.
        for key in [k for k in overrides if k.startswith('DS4_COMBO_')]:
            del overrides[key]
        if self.combo_enabled():
            # Same owner session across a media swap; a new epoch per launch.
            self.combo_epoch += 1
            self.combo_sequence = 1
            overrides.update(DS4_COMBO_ENABLED='1', DS4_COMBO_OWNER_SESSION=self.combo_session,
                             DS4_COMBO_OWNER_EPOCH=str(self.combo_epoch), DS4_COMBO_SEQUENCE='1',
                             DS4_COMBO_EXPERT_SNAPSHOT_FILE=str(self.expert_list()))
            if self.combo_restore:
                ram, checkpoint = self.combo_restore
                overrides.update(DS4_COMBO_RESTORE_MAPPING=ram.name, DS4_COMBO_RESTORE_CAPACITY=str(ram.capacity),
                                 DS4_COMBO_RESTORE_SHA256=checkpoint['payload_sha256'])
                if self.expert_list().is_file():
                    overrides['DS4_COMBO_EXPERT_RESTORE_FILE'] = str(self.expert_list())
        self.llm = OwnedProcess(self.cfg, command, 'ds4-server', gpu=not self.cfg.get('_gpu_lease_managed', False),
                                cwd=Path(self.cfg['runtime_root']), unbounded=True)
        if self.attach_entry:
            self.journal.update(self.attach_entry, 'attaching', pid=self.llm.proc.pid,
                                pid_created=psutil.Process(self.llm.proc.pid).create_time())
            self.attach_guard_recorded = True
        # Owner rule (no fixed timeouts): readiness lives while the server tree shows progress (job
        # log, I/O, CPU, members); it fails on exit or a no-progress window. A guardian waiting for
        # the GPU hold is idle by design, so gpu_wait_seconds widens the window, not a total cap.
        window = Window(readiness_seconds(self.cfg) + self.cfg.get('gpu_wait_seconds', 0))
        while not window.expired():
            if self.llm.proc.poll() is not None:
                self.llm.wait()
                raise RuntimeError('ds4 server exited before readiness')
            try:
                http_json(self.cfg['ds4_url'] + '/v1/models', timeout=2,
                          headers={'Authorization': 'Bearer ' + token})
                if self.combo_enabled():
                    reply = self.combo('setup')
                    if self.combo_restore and (reply.get('checkpoint') or {}).get('payload_sha256') != \
                            self.combo_restore[1]['payload_sha256']:
                        raise RuntimeError('Native startup did not restore the exact saved chat state')
                if self.attach_entry:
                    self.journal.update(self.attach_entry, 'attached', applied=True)
                self.loading_model = None
                self.status('Model ready')
                return
            except (OSError, RuntimeError):
                window.observe(activity(self.llm.proc.pid, self.llm.log))
                time.sleep(.2)
        self.close()
        raise TimeoutError('ds4 server readiness made no progress for %.0fs (log, I/O and CPU unchanged; %.0fs since launch)'
                           % (window.idle(), window.elapsed()))

    def _ready_single(self):
        if self.cfg['lora']:
            raise ValueError('Single-engine serving has no LoRA path; use --serving native for an output-head adapter')
        from .single_runtime import stack_config
        from .journal import atomic_json
        import secrets
        single = stack_config(self.cfg)
        # Correctness gate, not a cache: the producer refuses to start without a completed
        # full-SHA receipt (dsv41-prefill --copy-receipt). install/check normally wrote it; a
        # missing one is hashed here, blocking, with progress in the model status.
        from .native_broker import model_receipt
        model_receipt(single, report=self.status)
        single['env_overrides'].pop('DS4_SERVER_AUTH_TOKEN', None)  # secrets reach children by environment only
        config = Path(self.cfg['runtime_root']) / 'single-engine.json'
        atomic_json(config, single)
        token = self.cfg.setdefault('_control_token', secrets.token_urlsafe(32))
        self.cfg.setdefault('env_overrides', {})['DS4_SERVER_AUTH_TOKEN'] = token
        # combo_proxy's own guardian owns the server and producer and checks each request's
        # GPU window; each producer takes the GPU hold. The outer job is the CPU-idle server's
        # tree budget only (ds4-win scope for the native token), so no 19-minute engine deadline.
        self.llm = OwnedProcess(self.cfg, [sys.executable, '-B', '-m', 'slstack.combo_proxy', '--config', str(config)],
                                'single-engine', scope='ds4-win', telemetry=False,
                                cwd=Path(__file__).resolve().parents[1], timeout=86400)
        window = Window(readiness_seconds(self.cfg))   # no-progress window (owner rule), as above
        while not window.expired():
            if self.llm.proc.poll() is not None:
                self.llm.wait()
                raise RuntimeError('Single-engine server exited before readiness')
            try:
                http_json(self.cfg['ds4_url'] + '/v1/models', timeout=2, headers={'Authorization': 'Bearer ' + token})
                self.loading_model = None
                self.status('Model ready (single engine; the producer loads with the first prompt)')
                return
            except (OSError, RuntimeError):
                window.observe(activity(self.llm.proc.pid, self.llm.log))
                time.sleep(.2)
        self.close()
        raise TimeoutError('Single-engine server readiness made no progress for %.0fs (log, I/O and CPU unchanged; %.0fs since launch)'
                           % (window.idle(), window.elapsed()))

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
        # A non-streaming request is silent for its whole prefill + decode, so a socket timeout
        # would be a fixed request-length cap. The owned, guarded server process bounds it instead.
        return http_json(self.cfg['ds4_url'] + '/v1/chat/completions', body, None,
                         headers={'Authorization': 'Bearer ' + self.cfg['_control_token']})

    def chat_events(self, body):
        self.select(body.get('model'))
        self.ready()
        if self.remote:
            yield from self.remote.events(body)
            return
        if not self.single_engine():
            raise ValueError('Streaming requires the paired Mac receiver or the single engine')
        # The single engine streams its own SSE: content deltas, 1 Hz prefill_progress chunks
        # and timings on the finish chunk. Its beats keep each read inside request_timeout_s.
        from .streaming import events
        yield from events(self.cfg['ds4_url'] + '/v1/chat/completions', {**body, 'stream': True},
                          timeout=self.cfg['request_timeout_s'],
                          headers={'Authorization': 'Bearer ' + self.cfg['_control_token']})

    def media(self, kind, params):
        from .media_progress import MediaProgress
        self.media_progress = MediaProgress(self.cfg['runtime_root'], kind, params.get('steps', 20))
        try:
            result = self._media_job(kind, params)
        except BaseException:
            self.media_progress.phase('failed', active=False)
            raise
        self.media_progress.phase('complete', active=False)
        return result

    def _media_job(self, kind, params):
        if hasattr(self, 'preferences'):self.preferences.reload()
        model = self.cfg[kind + '_model']
        if not model and not self.cfg['comfyui_models']:
            raise ValueError(f'No {kind} model selected. Set --{kind}-model and its architecture/support files.')
        # Studio ordering: save and release the LLM, run the media job, restore in finally.
        saved = None
        if self.llm and self.llm.proc.poll() is None and self.single_engine():
            return self._media_single(kind, params)
        if self.llm and self.llm.proc.poll() is None and self.combo_enabled():
            started = time.monotonic()
            self.status(f'Saving chat state and unloading model for {kind}')
            saved = self.combo_save()
            self.swap_timing = {'save_unload_s': time.monotonic() - started}
        else:
            self.close()  # Certify owned LLM process tree has exited before giving Studio the GPU.
        try:
            started = time.monotonic()
            result = self._media(kind, params)
            if saved:
                self.swap_timing['media_s'] = time.monotonic() - started
            return result
        finally:
            if saved:
                started = time.monotonic()
                self.status('Reloading model and restoring chat state')
                self.media_progress.phase('restoring')
                self.combo_restore = saved
                try:
                    self.ready()
                finally:
                    self.combo_restore = None
                    saved[0].close()
                self.swap_timing['reload_restore_s'] = time.monotonic() - started
                self.status('Chat state restored')
                print('Media swap timing: ' + json.dumps(self.swap_timing), flush=True)

    def _media_single(self, kind, params):
        """Save the native frontier, lend the GPU to Studio, then arm restore."""
        started = time.monotonic()
        self.status(f'Releasing the producer for {kind}')
        timing = {'release': self.single('release')}
        timing['release_s'] = time.monotonic() - started
        try:
            started = time.monotonic()
            result = self._media(kind, params)
            timing['media_s'] = time.monotonic() - started
            return result
        finally:
            started = time.monotonic()
            self.status('Restoring the conversation checkpoint')
            self.media_progress.phase('restoring')
            timing['restore'] = self.single('restore')
            timing['restore_s'] = time.monotonic() - started
            self.swap_timing = timing
            self.status('Conversation checkpoint ready' if timing['restore'].get('deferred') else 'Conversation restored')
            print('Media swap timing: ' + json.dumps(timing), flush=True)

    def _media(self, kind, params):
        from .envvars import redact_cfg
        params = dict(params);model_id = params.pop('model', None)
        cfg = dict(self.cfg)
        if hasattr(self, 'preferences'):
            selected = self.preferences.select(kind, model_id)
            if not selected:raise ValueError('No supported '+kind+' model found; choose a model in Stack settings')
            cfg[kind+'_model'] = selected['path'];cfg[kind+'_arch'] = selected['arch']
            cfg[kind+'_mode'] = self.preferences.media_mode(selected['path'],kind)
            discovered = self.preferences.catalog.get('support_files', {}).get(selected['arch'], {})
            cfg[kind+'_support_files'] = {**discovered, **cfg[kind+'_support_files']}
        elif model_id:raise ValueError('Model selection needs Stack settings')
        if params.get('loras'):
            params['lora_files']=self.preferences.resolve_loras(kind,model_id,params['loras'])
        from .media_profiles import python_for
        cfg['studio_python']=python_for(cfg,cfg.get(kind+'_arch'))
        if not Path(cfg['studio_python']).is_file():
            raise RuntimeError('The selected model worker is not installed; run slstack install --image')
        request = {'kind': kind, 'cfg': redact_cfg(cfg), 'params': params}
        progress = getattr(self, 'media_progress', None)
        if progress:request['progress_file'] = str(progress.path)
        proc = OwnedProcess(cfg, [cfg['studio_python'], '-X', 'faulthandler', '-B', '-m', 'slstack.studio_worker'],
                            kind, stdin=json.dumps(request) + '\n', gpu=True)
        try:
            failure = None
            try:proc.wait()
            except RuntimeError as exc:failure = exc
            output = proc.log.read_text(errors='replace') if proc.log.is_file() else ''
            rows = [x[len('DS4STACK_RESULT '):] for x in output.splitlines()
                    if x.startswith('DS4STACK_RESULT ')]
            if len(rows) != 1:
                if failure:raise failure
                raise RuntimeError(f'Studio did not return exactly one result; see {proc.log}')
            result = json.loads(rows[0])
            if not result.get('ok'):
                raise RuntimeError(result.get('error', 'Studio generation failed')) from failure
            if failure:raise failure
            root = Path(self.cfg['runtime_root']).resolve() / 'outputs'
            artifacts = []
            for raw in result.get('files', []):
                source = Path(raw).resolve(strict=True)
                if not source.is_file() or not source.is_relative_to(root):
                    raise RuntimeError('Studio output escaped the isolated output directory')
                artifacts.append({'url': '/outputs/' + source.relative_to(root).as_posix(), 'path': str(source)})
            if not artifacts:
                raise RuntimeError('Studio returned no media files')
            return {'created': int(time.time()), 'data': artifacts, 'settings': {k:v for k,v in params.items() if k!='lora_files' and not k.startswith('_')},
                    'mode': cfg[kind + '_mode'], 'engine': 'studio', 'receipt': str(proc.receipt),
                    'selected_model': result.get('selected_model'), 'selected_arch': result.get('selected_arch')}
        finally:
            proc.stop()
