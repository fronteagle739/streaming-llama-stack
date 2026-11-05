"""Qwen3.8 Flash Next on one RTX: an OpenAI-compatible front end for the qfn-engine streaming engine.

The engine (engines/streaming-llama/prefill_llama.cpp/tools/qfn-engine, `--serve`) owns everything model-specific:
it renders the conversation with the pack's own chat template, keeps the live sequence for prefix reuse, decodes with
the expert cache and parses reasoning and tool calls, streaming OpenAI chat-completion deltas as JSON lines. This
module adds HTTP, Bearer authentication, the shared GPU lease while the engine is resident, idle retirement and a
graceful shutdown. One request runs at a time (the engine holds one sequence).

Quality/speed: request field `quality_speed` = "max_quality" (the full model, router bias 0), "max_speed" (bias
0.68 toward experts already in VRAM) or a number (the bias itself); default from the config
key `qfn_quality_speed`.

Sampling fields (including typical_p, repeat_penalty/repeat_last_n, presence_penalty,
frequency_penalty, seed, stop, logit_bias and max_tokens/max_completion_tokens) are
forwarded unchanged to the engine. The pack defaults and validation live in the
shared native parser used by chat, generate and the CLI.
"""
from contextlib import ExitStack
import base64
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import secrets
import subprocess
import struct
import sys
import threading
import time

from .config import ROOT

# Max speed biases routing toward resident experts; numeric biases retain 0..8.
MAX_SPEED_BIAS = 0.68
EXE = 'bin/qfn-engine.exe'


def bias_of(value, settings=None, max_speed_bias=None):
    if isinstance(settings, (int,float)) and not isinstance(settings,bool):
        max_speed_bias,settings=settings,None
    if settings is not None and isinstance(value, str):
        from .model_settings import load, named_bias
        selected = named_bias(load(settings), value)
        if value == 'max_speed' and max_speed_bias is not None and selected != max_speed_bias:
            raise ValueError('Max speed bias differs between the explicit flag and model settings')
        return selected
    if value is None or value == 'max_quality':return 0.0
    if value == 'max_speed':
        if max_speed_bias is None:max_speed_bias = MAX_SPEED_BIAS
        if type(max_speed_bias) not in (int,float):raise ValueError('QFN Max speed bias must be numeric')
        return bias_of(max_speed_bias)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 8:return float(value)
    raise ValueError('quality_speed must be "max_quality", "max_speed" or a bias between 0 and 8')


class QfnEngine:
    """The resident engine process and the GPU lease it runs under."""
    def __init__(self, cfg, lease_factory=None, launch_factory=None):
        self.cfg = cfg;self.job = None;self.conn = None;self.rfile = None;self.leases = None;self.events = None
        self.lock = threading.Lock();self.last_used = time.monotonic();self.ready = None
        self.lease_factory = lease_factory or self._lease
        self.launch_factory = launch_factory or self.launch
        self.idle_seconds = float(cfg.get('qfn_idle_seconds', max(60, cfg.get('lease_minutes', 19) * 60 - 60)))
        from .vision import Images
        self.images=Images({**cfg,'model_family':'qfn'})

    def prepare_images(self,body):
        # Retire the complete language process before the DSV-style isolated
        # encoder, then reacquire the normal engine lease on its next request.
        with ExitStack() as ownership:
            held=[False]
            def before_encode():
                if not held[0]:
                    self.retire('lazy image encoder')
                    ownership.enter_context(self.lease_factory());held[0]=True
            body,blocks,_=self.images.prepare(body,before_encode)
        projected=[]
        for block in blocks:
            with Path(block['file']).open('rb') as f:
                header=f.read(48);magic,version,count,width=struct.unpack('<4I',header[:16]);data=f.read()
            if (magic,version,width)!=(0x49564c53,1,2560) or len(data)!=count*width*4:
                raise ValueError('QFN projected image extent differs')
            projected.append(dict(marker=block['marker'],count=count,grid_height=block['grid_height'],
                grid_width=block['grid_width'],fingerprint=header[16:].hex(),data=base64.b64encode(data).decode('ascii')))
        return dict(op='chat',body=body,vision=projected)

    def _lease(self):
        from .lease import gpu_lease
        return gpu_lease(self.cfg)

    def command(self):
        from .performance_contract import seal, validate
        if self.cfg.get('performance_manifest') is None:
            # Direct library callers use the same explicit, recorded defaults
            # as the CLI. Existing manifests are validated without resealing.
            resolved = dict(self.cfg)
            for key, value in dict(host_budget_gib=3, qfn_host_budget_gib=6,
                                   qfn_engram_scan_rings=1, qfn_tile=2048).items():
                resolved.setdefault(key, value)
            self.cfg = seal(resolved, 'qfn')
        validate(self.cfg, 'qfn')
        exe = Path(self.cfg.get('qfn_engine') or ROOT / EXE)
        runtime = Path(self.cfg['runtime_root']);runtime.mkdir(parents=True, exist_ok=True)
        vocab = self.cfg.get('qfn_vocab_gguf')
        if vocab is None:
            vocab = runtime / 'qfn-vocab.gguf'
        elif not isinstance(vocab, (str, os.PathLike)) or not str(vocab):
            raise ValueError('qfn_vocab_gguf must be a nonempty path')
        argv = [str(exe), '--serve', '--model', self.cfg['model'], '--vocab-gguf', str(vocab),
                '--ctx', str(int(self.cfg.get('context', 32768))), '--port', str(self.port()),
                '--protect-host-working-set',str(int(self.cfg.get('qfn_protect_host_working_set',1)))]
        argv+=['--lock-host-resident',str(int(self.cfg.get('qfn_lock_host_resident',1)))]
        for key, flag in (('qfn_tile', '--tile'), ('qfn_host_budget_gib', '--host-budget-gib'),
                          ('qfn_cache_slots', '--cache-slots'), ('qfn_experts_file', '--experts-file')):
            if self.cfg.get(key) is not None:argv += [flag, str(self.cfg[key])]
        indexer = self.cfg.get('qfn_qsa_indexer')
        if indexer is not None:
            if indexer not in ('auto', 'scalar', 'basalt-tc-refine-prefill', 'basalt-tc-refine', 'basalt-tc', 'basalt-tc-prefill', 'dsv-fp4'):
                raise ValueError('qfn_qsa_indexer must be auto, scalar, basalt-tc-refine-prefill, basalt-tc-refine, basalt-tc, basalt-tc-prefill or dsv-fp4')
            argv += ['--qsa-indexer', indexer]
        if 'qfn_qsa_attention' in self.cfg:
            policy = self.cfg['qfn_qsa_attention']
            if policy not in ('native', 'basalt'):
                raise ValueError('qfn_qsa_attention must be native or basalt')
            argv += ['--qsa-attention', policy]
        mode = self.cfg.get('qfn_nvfp4_mode')
        if mode is not None:
            if mode not in ('auto', 'w4a4', 'native-q8', 'mmq', 'calibrated-w4a4', 'dsv-native'):
                raise ValueError('qfn_nvfp4_mode must be auto, w4a4, native-q8, mmq, calibrated-w4a4 or dsv-native (prefill only)')
            argv += ['--nvfp4-mode', mode]
        decode_mode = self.cfg.get('qfn_nvfp4_decode_mode')
        if decode_mode is not None:
            if decode_mode not in ('native-q8', 'calibrated-w4a4', 'dsv-native'):
                raise ValueError('qfn_nvfp4_decode_mode must be native-q8, calibrated-w4a4 or dsv-native')
            argv += ['--nvfp4-decode-mode', decode_mode]
        return argv

    def port(self):
        return int(self.cfg.get('qfn_engine_port', 8095))

    def launch(self):
        """The engine runs under the stack's host-memory guard (tree-RSS budget, physical floor, Job commit cap,
        paging limits); it serves on 127.0.0.1 because the guard keeps the child's stdout in its job log."""
        from .process import OwnedProcess
        argv = self.command()
        cfg = dict(self.cfg, env_overrides={**self.cfg.get('env_overrides', {}),
            'LLAMADECODE_ENGRAM_SCAN_RINGS': str(self.cfg['qfn_engram_scan_rings'])})
        return OwnedProcess(cfg, argv, 'qfn-engine', cwd=Path(cfg['runtime_root']), unbounded=True, scope='qfn-engine', telemetry=True)

    def connect(self):
        """Progress-based: retry while the guarded engine is alive (it listens before it loads)."""
        import socket
        while True:
            if self.job.proc.poll() is not None:raise RuntimeError(f'qfn-engine guard exited {self.job.proc.returncode} before it listened')
            try:
                conn = socket.create_connection(('127.0.0.1', self.port()), timeout=2)
                conn.settimeout(None)
                return conn
            except OSError:
                time.sleep(.5)

    def start(self):
        if self.job is not None:return
        stack = ExitStack();stack.enter_context(self.lease_factory());self.leases = stack
        try:
            self.job = self.launch_factory()
            self.conn = self.connect()
            self.rfile = self.conn.makefile('rb')
            self.events = queue.Queue()
            threading.Thread(target=self._pump, args=(self.rfile, self.events), daemon=True).start()
            while True:
                try:row = self.events.get(timeout=5)
                except queue.Empty:
                    if self.job.proc.poll() is not None:raise RuntimeError('qfn-engine exited while loading')
                    continue
                if row is None:raise RuntimeError('qfn-engine closed its connection while loading')
                if row.get('event') == 'ready':self.ready = row;break
                if not row.get('ok', True):raise RuntimeError(row.get('error', 'qfn-engine refused to start'))
            print('QfnSingle engine ready: ' + json.dumps(row), flush=True)
        except BaseException:
            self.retire('start failed');raise

    @staticmethod
    def _pump(rfile, events):
        try:
            for line in rfile:
                line = line.strip()
                if not line:continue
                try:events.put(json.loads(line))
                except ValueError:events.put(dict(ok=False, error='unparseable engine line: ' + line[:200].decode('utf-8', 'replace')))
        except OSError as error:events.put(dict(ok=False,error='QFN native transport closed: '+str(error)))
        finally:events.put(None)

    def send(self, row):
        self.conn.sendall((json.dumps(row, separators=(',', ':')) + '\n').encode())

    def cancel(self):
        try:
            if self.conn is not None:self.send(dict(op='cancel'))
        except OSError:pass

    def alive(self):
        return self.job is not None and self.job.proc.poll() is None

    def run(self, row, terminal):
        """Send one op and yield its events until terminal(event) is true."""
        self.start();self.last_used = time.monotonic();self.send(row)
        while True:
            try:event = self.events.get(timeout=5)
            except queue.Empty:
                if not self.alive():raise RuntimeError('qfn-engine exited')
                continue
            if event is None:raise RuntimeError('qfn-engine closed its connection')
            yield event
            if not event.get('ok', True) or terminal(event):
                self.last_used = time.monotonic();return

    def retire(self, why):
        """Ask the engine to quit and wait for its guard to confirm the exit (VRAM freed), then release the GPU hold.
        A guard that does not end gets its progress-aware graceful stop (logged force fallback)."""
        job, conn, self.job, self.conn = self.job, self.conn, None, None
        certified=job is None
        try:
            if conn is not None:
                print('QfnSingle retiring engine: ' + why, flush=True)
                try:conn.sendall(b'{"op":"quit"}\n')
                except OSError:pass
            if job is not None:
                try:job.proc.wait(timeout=float(self.cfg.get('qfn_quit_grace_s', 120)))
                except subprocess.TimeoutExpired:job.stop()
                if job.proc.poll() is not None:
                    try:result=job.wait()
                    except Exception:
                        # A failed inference job can still have certified
                        # cleanup. Preserve the failure, but release its proven
                        # dead process so shutdown cannot strand the lease.
                        receipt=getattr(job,'receipt',None)
                        if receipt is None or not Path(receipt).is_file():raise
                        result=json.loads(Path(receipt).read_text())
                        if not result.get('release_allowed'):raise
                        print('QfnSingle retired failed job with certified cleanup: '+str(result.get('reason')),file=sys.stderr,flush=True)
                    if not result.get('release_allowed'):raise RuntimeError('QFN engine retirement was not certified')
                    certified=True
            if conn is not None:
                try:conn.close()
                except OSError:pass
        finally:
            if certified:
                leases, self.leases = self.leases, None
                if leases is not None:leases.close()
            else:self.job,self.conn=job,conn

    def idle_check(self):
        if self.job is None or not self.lock.acquire(blocking=False):return
        try:
            if time.monotonic() - self.last_used > self.idle_seconds:self.retire('idle')
        finally:self.lock.release()


def chunk(cid, model, created, delta, finish=None):
    return dict(id=cid, object='chat.completion.chunk', created=created, model=model,
                choices=[dict(index=0, delta=delta, finish_reason=finish)])


def handler(state):
    cfg, engine = state['cfg'], state['engine']
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def log_message(self, *args):pass
        def send(self, status, body):
            data = json.dumps(body, separators=(',', ':'), ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type', 'application/json');self.send_header('Content-Length', str(len(data)))
            self.end_headers();self.wfile.write(data)
        def authorized(self):
            token = cfg.get('api_key') or ''
            return not token or hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + token)
        def do_GET(self):
            if self.path in ('/health', '/healthz'):
                return self.send(200, dict(status='ok', resident=engine.job is not None, context=cfg.get('context')))
            if self.path == '/v1/models':
                return self.send(200, dict(object='list', data=[dict(id=state['model_id'], object='model', owned_by='slstack')]))
            return self.send(404, dict(error=dict(message='Unknown endpoint')))
        def do_POST(self):
            if not self.authorized():return self.send(401, dict(error=dict(message='Bearer authentication required')))
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= int(cfg.get('qfn_body_limit', 64 << 20)):raise ValueError('Request body exceeds its bound')
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):raise ValueError('The request body must be a JSON object')
            except (ValueError, OSError) as exc:return self.send(400, dict(error=dict(message=str(exc))))
            if self.path == '/v1/stack/shutdown':return self.shutdown()
            if self.path != '/v1/chat/completions':return self.send(404, dict(error=dict(message='Unknown endpoint')))
            try:body['bias'] = bias_of(body.pop('quality_speed', cfg.get('qfn_quality_speed')), cfg.get('model_settings'), cfg.get('qfn_max_speed_bias'))
            except ValueError as exc:return self.send(400, dict(error=dict(message=str(exc), type='invalid_request_error')))
            if not engine.lock.acquire(blocking=False):return self.send(409, dict(error=dict(message='Another request owns the engine')))
            try:self.chat(body)
            finally:engine.lock.release()
        def chat(self, body):
            streaming = body.get('stream') is True;cid = 'chatcmpl-' + secrets.token_hex(12);created = int(time.time())
            model = body.get('model') if isinstance(body.get('model'), str) else state['model_id']
            include_usage = isinstance(body.get('stream_options'), dict) and body['stream_options'].get('include_usage') is True
            alive = [True]
            def write(row):
                if not alive[0]:return
                try:self.wfile.write(b'data: ' + json.dumps(row, separators=(',', ':'), ensure_ascii=False).encode() + b'\n\n');self.wfile.flush()
                except OSError:alive[0] = False;engine.cancel()   # client gone: the engine stops at its next token
            headers = False;final = None
            try:
                if streaming:
                    self.send_response(200);self.send_header('Content-Type', 'text/event-stream');self.send_header('Cache-Control', 'no-cache')
                    self.send_header('Connection', 'close');self.end_headers();self.close_connection = True;headers = True
                    write(chunk(cid, model, created, dict(role='assistant')))
                request=engine.prepare_images(body)
                for event in engine.run(request, lambda e:e.get('done') is True):
                    if not event.get('ok', True):raise RuntimeError(event.get('error', 'engine error'))
                    if 'delta' in event and streaming:write(chunk(cid, model, created, event['delta']))
                    elif event.get('done'):final = event
                    elif event.get('event') == 'prefilled' and streaming:
                        write(dict(id=cid, object='chat.completion.chunk', created=created, model=model, choices=[],
                                   prefill=event['prefill']))
                finish = final['finish_reason']
                if finish == 'cancelled':finish = 'stop'
                if streaming:
                    done = chunk(cid, model, created, {}, finish);done['timings'] = final['timings'];write(done)
                    if include_usage:write(dict(id=cid, object='chat.completion.chunk', created=created, model=model, choices=[], usage=final['usage']))
                    if alive[0]:self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
                else:
                    message = final['message'];message.setdefault('content', '')
                    self.send(200, dict(id=cid, object='chat.completion', created=created, model=model,
                                        choices=[dict(index=0, message=message, finish_reason=finish)], usage=final['usage'],
                                        timings=final['timings'], qfn=final.get('stats')))
            except Exception as exc:
                error = dict(error=dict(message=str(exc), type='qfn_engine_refused'))
                if engine.job is not None and not engine.alive():engine.retire('engine exited')
                if headers:
                    try:self.wfile.write(b'data: ' + json.dumps(error).encode() + b'\n\ndata: [DONE]\n\n');self.wfile.flush()
                    except OSError:pass
                else:self.send(503, error)
        def shutdown(self):
            engine.cancel()
            with engine.lock:engine.retire('shutdown')
            state['stopping'].set()
            threading.Thread(target=state['server'].shutdown, daemon=True).start()
            try:self.send(200, dict(closing=True))
            except OSError:pass
    return Handler


def serve(cfg):
    """Serve until /v1/stack/shutdown. The engine starts on the first request and retires when idle."""
    from .model_settings import apply
    cfg = apply(cfg, 'qfn')
    cfg={**cfg,'model_family':'qfn'}
    for key,value in dict(vision_backend='CUDA0',vision_min_tokens=64,vision_max_tokens=1024,
                          vision_threads=8,vision_flash_attention=0,vision_host_budget_gib=3).items():cfg.setdefault(key,value)
    if not cfg.get('model'):raise ValueError('Set "model" to the Qwen3.8 Flash Next .basalt pack')
    engine = QfnEngine(cfg)
    state = dict(cfg=cfg, engine=engine, stopping=threading.Event(),
                 model_id=cfg.get('model_name') or Path(cfg['model']).stem)
    class Server(ThreadingHTTPServer):daemon_threads = True
    server = Server((cfg.get('host', '127.0.0.1'), int(cfg.get('port', 8088))), handler(state));state['server'] = server
    def idle():
        while not state['stopping'].wait(15):engine.idle_check()
    threading.Thread(target=idle, daemon=True).start()
    if cfg.get('qfn_preload'):engine.start()
    print(f'QfnSingle ready on {server.server_address[0]}:{server.server_address[1]} ({state["model_id"]})', flush=True)
    try:server.serve_forever(poll_interval=.1)
    finally:
        server.server_close()
        with engine.lock:engine.retire('server closed')
