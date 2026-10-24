"""Single-engine serving: the ULTRA producer prefills AND decodes; no native ds4-win.

One resident producer per conversation frontier. Each request is rendered and
tokenized in process (ds4_render, an exact port of ds4-win's renderer), matched
against the producer's live token ids with the native's own prefix rule, and
only the new suffix is appended with ComboProducer.prefill. A diverged prefix
retires the producer and starts a fresh one on the whole prompt. Decode is the
producer's in-process greedy decode with EOS (id 1) as the stop token; text is
streamed from its per-token LLAMADECODE_STEP lines and the finished text is
parsed with ds4-win's tool-call parser (ds4_tool_parse). /v1/quality-speed
drives ComboProducer.quality.

The GPU hold is taken when a producer launches and released when it retires
(idle timeout, divergence, shutdown); the guardian's request window is active
only while a request runs.
"""
from __future__ import annotations
from contextlib import ExitStack
import codecs
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import threading
import time
import psutil
from .journal import atomic_json
from .native_broker import vendor
from . import ds4_render
from .ds4_tool_parse import finish_message, find_any_tool_start, tool_call_deltas

EOS_ID = 1
MAX_DECODE_CALL = 4096
STEP = re.compile(rb'LLAMADECODE_STEP index=(\d+) token=(\d+) ')
TOOL_STARTS = ('\n\n' + ds4_render.DS41_TOOL_CALLS_START, ds4_render.DS41_TOOL_CALLS_START, '<｜DSML｜tool_calls>',
               '<DSML｜tool_calls>', '<tool_calls>', '<tool_call>')


def _partial_hold(text, markers):
    """Length of the longest suffix of text that is a proper prefix of a marker."""
    hold = 0
    for marker in markers:
        for n in range(min(len(marker) - 1, len(text)), hold, -1):
            if text.endswith(marker[:n]):hold = n;break
    return hold


class Splitter:
    """Live reasoning/content deltas whose concatenation is always a prefix of the
    final parse: reasoning stops at </think> (or a tool block), content holds back
    trailing whitespace and anything that may begin a tool block."""
    def __init__(self, think):
        self.text = '';self.sent = dict(reasoning='', content='');self.think = think;self.frozen = False

    def feed(self, piece):
        self.text += piece
        out = []
        if self.frozen:return out
        if self.think:
            end = self.text.find('</think>')
            region = self.text if end < 0 else self.text[:end]
            tool = find_any_tool_start(region)
            if tool is not None:region = region[:tool];self.frozen = end < 0 or tool < end
            ready = region if (end >= 0 and tool is None) else region[:len(region) - _partial_hold(region, ('</think>',) + TOOL_STARTS)]
            if len(ready) > len(self.sent['reasoning']):
                out.append(('reasoning_content', ready[len(self.sent['reasoning']):]));self.sent['reasoning'] = ready
            if end < 0 or self.frozen:return out
            body = self.text[end + 8:]
        else:body = self.text
        tool = find_any_tool_start(body)
        if tool is not None:body = body[:tool];self.frozen = True
        if not self.frozen:body = body[:len(body) - _partial_hold(body, TOOL_STARTS)]
        ready = body.rstrip(' \t\n\r\v\f')
        if len(ready) > len(self.sent['content']):
            out.append(('content', ready[len(self.sent['content']):]));self.sent['content'] = ready
        return out

    def remainder(self, field, final):
        sent = self.sent[field];final = final or ''
        if final.startswith(sent):return final[len(sent):]
        print('DS4SINGLE stream/final ' + field + ' mismatch: live ' + str(len(sent)) + ' chars not a prefix of the parse', flush=True)
        return ''


class SingleEngine:
    """The resident producer, its live token ids and the per-conversation state."""
    def __init__(self, cfg, *, tokenizer=None, producer_factory=None, lease_factory=None):
        self.cfg = deepcopy(cfg);self.tokenizer = tokenizer
        self.producer = None;self.live = [];self.leases = None;self.lease_id = None
        self.tool_memory = {};self.lock = threading.Lock();self.stopping = threading.Event()
        self.quality_value = None;self.applied_quality = None;self.last_used = time.monotonic()
        self.poison = None;self.decoding = False
        self.producer_factory = producer_factory or self._launch_producer
        self.lease_factory = lease_factory or self._lease
        self.idle_seconds = float(cfg.get('single_engine_idle_seconds', max(60, cfg['lease_minutes'] * 60 - 60)))
        self.stop_tokens = [int(t) for t in cfg.get('single_engine_stop_tokens', [EOS_ID])]
        if not self.stop_tokens or any(not 0 <= t < 129280 for t in self.stop_tokens):raise ValueError('Single-engine stop tokens must be vocabulary ids')

    # --- resources -------------------------------------------------------
    def _lease(self):
        from .lease import gpu_lease
        return gpu_lease(self.cfg)

    def _launch_producer(self, tokens):
        from .combo_producer import ComboProducer
        return ComboProducer(self.cfg, secrets.token_hex(16), secrets.randbits(63) or 1, tokens, self.stopping.is_set)

    def tokenize(self):
        if self.tokenizer is None:self.tokenizer = ds4_render.Tokenizer.from_gguf(self.cfg['model'])
        return self.tokenizer

    def retire(self, why):
        """Gracefully close the producer, then release the GPU hold."""
        producer, self.producer = self.producer, None
        self.live = [];self.applied_quality = None;self.tool_memory = {}
        receipt = None
        try:
            if producer is not None:
                print('DS4SINGLE retiring producer: ' + why, flush=True)
                receipt = producer.close()
        except BaseException as exc:
            self.poison = 'Unconfirmed GPU release: ' + repr(exc)
            raise
        finally:
            if self.leases is not None and self.poison is None:
                leases, self.leases, self.lease_id = self.leases, None, None
                leases.close()
        return receipt

    def _rotate_diagnostic(self):
        """Keep the producer's stderr below its RAM cap across a long-lived session."""
        process = getattr(self.producer, 'process', None)
        if process is None or not hasattr(process, 'diagnostic'):return
        with process.state_lock:
            data = bytes(process.diagnostic);process.diagnostic.clear()
        if data:
            with (Path(self.producer.directory) / 'stderr-rotated.log').open('ab') as out:out.write(data)

    # --- quality ---------------------------------------------------------
    def set_quality(self, value):
        vendor();from studio.quality_speed_contract import validate
        self.quality_value = validate(value)

    def quality_scalar(self, think):
        value = self.quality_value
        if isinstance(value, dict):return value['reasoning' if think else 'answer']
        return value

    def apply_quality(self, think):
        value = self.quality_scalar(think)
        if value is None or self.producer is None or value == self.applied_quality:return None
        reply = self.producer.quality(value);self.applied_quality = value
        return reply

    # --- one request -----------------------------------------------------
    def prepare(self, body, metrics):
        """Render, then make the producer's frontier equal the full prompt."""
        from .combo_session import render_body
        tokenizer = self.tokenize()
        rendered = render_body(body)
        text, canonical, request = ds4_render.render(rendered, tokenizer, tool_memory=self.tool_memory)
        if len(canonical) >= self.cfg['context'] - 1:raise ValueError('Prompt exceeds the admitted context')
        metrics.update(prompt_tokens=len(canonical))
        tokens, source = ds4_render.continue_live(self.live if self.producer else [], text, canonical, tokenizer)
        metrics['prefix_source'] = source
        if self.producer is not None and tokens is None:
            metrics['fresh_reason'] = 'rendered history diverges from the live frontier'
            self.retire('prefix diverged');tokens = canonical
        started = time.monotonic()
        if self.producer is None:
            if self.leases is None:
                stack = ExitStack();self.lease_id = stack.enter_context(self.lease_factory());self.leases = stack
            self.live = []
            try:self.producer = self.producer_factory(tokens)
            except BaseException as exc:
                self.producer = None
                if hasattr(exc, 'owned_cleanup'):
                    from .combo_release import certified
                    if not certified(exc.owned_cleanup):self.poison = 'Unconfirmed GPU release: ' + repr(exc);raise
                self.leases.close();self.leases = self.lease_id = None
                raise
            self.live = list(tokens);metrics['producer_launch_prefill_seconds'] = time.monotonic() - started
            metrics['reused_prefix_tokens'] = 0;metrics['new_tokens'] = len(tokens)
        else:
            append = tokens[len(self.live):];metrics['reused_prefix_tokens'] = len(self.live);metrics['new_tokens'] = len(append)
            chunk = int((getattr(self.producer.process, 'ready', None) or {}).get('maximum_prefill_tokens') or 49152)
            for offset in range(0, len(append), chunk):
                self.producer.prefill(append[offset:offset + chunk]);self.live.extend(append[offset:offset + chunk])
            metrics['append_prefill_seconds'] = time.monotonic() - started
        self._check_frontier()
        return text, request

    def _check_frontier(self):
        position = (getattr(self.producer, 'checkpoint', None) or {}).get('position')
        if position is not None and position != len(self.live):
            raise RuntimeError('Producer frontier %d differs from the live token count %d' % (position, len(self.live)))

    def decode(self, max_tokens, on_token):
        """Greedy decode until EOS or max_tokens; on_token(id) per evaluated step."""
        generated = [];stopped = False;steps = [];decode_ms = 0.0
        while len(generated) < max_tokens and not stopped:
            room = self.cfg['context'] - len(self.live) - 1
            budget = min(MAX_DECODE_CALL, max_tokens - len(generated), room)
            if budget < 1:break
            if generated:   # the token sampled at the previous call's limit is not evaluated yet
                on_token(generated[-1]);self.producer.prefill([generated[-1]]);self.live.append(generated[-1])
            process = self.producer.process;offset = [len(process.diagnostic)];seen = [len(generated)]
            done = threading.Event()
            def watch():
                while True:
                    finished = done.wait(.05)
                    with process.state_lock:data = bytes(process.diagnostic[offset[0]:])
                    cut = data.rfind(b'\n') + 1;offset[0] += cut
                    for match in STEP.finditer(data[:cut]):on_token(int(match[2]));seen[0] += 1
                    if finished:return
            watcher = threading.Thread(target=watch, daemon=True);watcher.start()
            try:reply = self.producer.decode(budget, stop_tokens=self.stop_tokens)
            finally:done.set();watcher.join()
            tokens = [int(t) for t in reply['tokens']];stopped = bool(reply.get('stopped'))
            evaluated = tokens[:-1] if tokens else []
            for token in tokens[seen[0] - len(generated):len(evaluated)]:on_token(token)
            self.live.extend(evaluated);generated.extend(tokens)
            steps.extend(reply.get('step_ms') or []);decode_ms += float(reply.get('decode_ms') or 0)
            self._check_frontier()
        if stopped and generated and generated[-1] in self.stop_tokens:text_tokens = generated[:-1]
        else:text_tokens = generated
        if not stopped and generated:on_token(generated[-1])
        return dict(tokens=generated, text_tokens=text_tokens, stopped=stopped, step_ms=steps, decode_ms=decode_ms)

    def run(self, body, emit, metrics, ready=None):
        """One chat request. emit(kind, value) receives ('delta', field, text) rows
        during decode and the final ('final', result) row; ready() runs once the
        producer holds the prompt (the request window then names its lease)."""
        text, request = self.prepare(body, metrics)
        if ready:ready()
        think = request['think_mode'] != ds4_render.THINK_NONE
        metrics['quality_reply'] = self.apply_quality(think)
        max_tokens = request['max_tokens'] if request['max_tokens'] else int(self.cfg.get('single_engine_default_max_tokens', 32768))
        max_tokens = min(max_tokens, self.cfg['context'] - len(self.live) - 1)
        splitter = Splitter(think);decoder = codecs.getincrementaldecoder('utf-8')('replace')
        tokenizer = self.tokenize();started = time.monotonic()
        def on_token(token):
            if token in self.stop_tokens:return
            if 'first_token_seconds' not in metrics:metrics['first_token_seconds'] = time.monotonic() - started
            piece = decoder.decode(tokenizer.token_bytes(token))
            for field, delta in splitter.feed(piece):emit('delta', field, delta)
        self.decoding = True
        try:result = self.decode(max_tokens, on_token) if max_tokens > 0 else dict(tokens=[], text_tokens=[], stopped=False, step_ms=[], decode_ms=0.0)
        finally:self.decoding = False
        generated_text = tokenizer.detokenize(result['text_tokens']).decode('utf-8', 'replace')
        finish = 'stop' if result['stopped'] else 'length'
        parsed = finish_message(generated_text, has_tools=request['has_tools'], think=think, finish=finish)
        if parsed['calls'] and parsed['raw_tool_text']:
            for call in parsed['calls']:self.tool_memory[call['id']] = parsed['raw_tool_text']
        metrics.update(completion_tokens=len(result['text_tokens']), decode_ms=result['decode_ms'],
                       decode_tok_s=len(result['step_ms']) * 1000.0 / result['decode_ms'] if result['decode_ms'] else None,
                       finish_reason=parsed['finish'], live_tokens=len(self.live))
        emit('final', dict(parsed=parsed, splitter=splitter, request=request, usage=dict(
            prompt_tokens=metrics['prompt_tokens'], completion_tokens=len(result['text_tokens']),
            total_tokens=metrics['prompt_tokens'] + len(result['text_tokens']),
            prompt_tokens_details=dict(cached_tokens=metrics.get('reused_prefix_tokens', 0), cache_write_tokens=metrics.get('new_tokens', 0)))))
        self._rotate_diagnostic();self.last_used = time.monotonic()

    def idle_check(self):
        if self.producer is None or not self.lock.acquire(blocking=False):return
        try:
            if self.producer is not None and time.monotonic() - self.last_used >= self.idle_seconds:self.retire('idle')
        finally:self.lock.release()

    def close(self):
        with self.lock:
            try:
                if self.producer is not None or self.leases is not None:self.retire('shutdown')
            finally:self.stopping.set()


class SingleProxy:
    """HTTP state for the single-engine mode (window file, token, models)."""
    def __init__(self, cfg, run_id, window_file, *, engine=None):
        from .combo_proxy import request_worker_config
        self.cfg = cfg;self.run_id = run_id;self.window_file = Path(window_file) if window_file else None
        self.token = os.environ.get('DS4_SERVER_AUTH_TOKEN', '')
        if len(self.token) < 32:raise ValueError('Normal Studio native Bearer token is required')
        self.engine = engine or SingleEngine(request_worker_config(cfg))
        self.model_id = 'deepseek-v4.1-flash'
        self.models = {'object': 'list', 'data': [dict(id=self.model_id, object='model', created=1767225600, owned_by='ds4.c',
            name=cfg.get('model_name', 'DeepSeek-V4.1-Flash'), context_length=cfg['context'],
            top_provider=dict(context_length=cfg['context'], max_completion_tokens=cfg['context'], is_moderated=False),
            supported_parameters=['tools', 'tool_choice', 'max_tokens', 'stream', 'reasoning_effort'])]}
        self.window(False)

    def window(self, active, lease_id=None):
        if self.window_file is None:return
        row = dict(active=active, run_id=self.run_id, owner_pid=os.getpid(), owner_created=psutil.Process().create_time(),
                   lease_id=lease_id, deadline=time.time() + self.cfg['lease_minutes'] * 60)
        started = time.monotonic()
        while True:
            try:atomic_json(self.window_file, row);return
            except PermissionError:
                if time.monotonic() - started >= 1:raise
                time.sleep(.01)

    def quality(self, args=None):
        vendor();from studio.quality_speed_contract import NAMED, setting_arguments
        engine = self.engine;deferred = False
        if args is not None:
            engine.set_quality(setting_arguments(args, engine.quality_value if engine.quality_value is not None else 0.0))
            # A running decode owns the session pipe: the value applies from the next decode.
            if engine.producer is not None and not engine.decoding and engine.lock.acquire(blocking=False):
                try:engine.apply_quality(False)
                finally:engine.lock.release()
            else:deferred = True
        value = engine.quality_value
        phases = deepcopy(value) if isinstance(value, dict) else None
        return dict(supported=True, engine='combo-producer', value=phases['answer'] if phases else (value if value is not None else 0.0),
                    phases=phases, per_phase=phases is not None, applied_value=engine.applied_quality,
                    requested_value=value, deferred=deferred, applied=not deferred and engine.applied_quality is not None,
                    decoder_running=engine.decoding, named_points=NAMED,
                    scope='resident producer; a running decode takes it from its next decode call')


def _chunk(cid, model, created, delta, finish=None):
    return dict(id=cid, object='chat.completion.chunk', created=created, model=model,
                choices=[dict(index=0, delta=delta, finish_reason=finish)])


def handler(proxy):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def log_message(self, *args):pass
        def send(self, status, body):
            data = json.dumps(body, separators=(',', ':'), allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type', 'application/json');self.send_header('Content-Length', str(len(data)));self.end_headers();self.wfile.write(data)
        def authorized(self):return hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + proxy.token)
        def read_body(self):
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 1 << 20:raise ValueError('Native request body must fit1MiB')
            self.connection.settimeout(30);data = self.rfile.read(length)
            if len(data) != length:raise ValueError('Truncated request')
            vendor();from studio.dsv41_session_contract import control
            return control(data)
        def do_GET(self):
            if self.path in ('/health', '/healthz'):
                return self.send(200, dict(status='ok', model_ready=True, context=proxy.cfg['context'], resident=proxy.engine.producer is not None))
            if self.path == '/v1/models':return self.send(200, proxy.models)
            if self.path == '/v1/quality-speed':
                if not self.authorized():return self.send(401, dict(error=dict(message='Native Bearer authentication required')))
                try:return self.send(200, proxy.quality())
                except (ValueError, RuntimeError, OSError) as exc:return self.send(409, dict(error=dict(message=str(exc))))
            return self.send(404, dict(error=dict(message='Unknown native endpoint')))
        def do_POST(self):
            if not self.authorized():return self.send(401, dict(error=dict(message='Native Bearer authentication required')))
            try:body = self.read_body()
            except (ValueError, OSError) as exc:return self.send(400, dict(error=dict(message=str(exc))))
            if self.path == '/v1/quality-speed':
                try:return self.send(200, proxy.quality(body))
                except (ValueError, RuntimeError, OSError) as exc:return self.send(409, dict(error=dict(message=str(exc))))
            if self.path != '/v1/chat/completions':return self.send(404, dict(error=dict(message='Unknown native endpoint')))
            engine = proxy.engine
            if not engine.lock.acquire(blocking=False):return self.send(409, dict(error=dict(message='Another request owns this decoder')))
            streaming = body.get('stream') is True;headers = False;metrics = dict(started=time.time());alive = [True]
            cid = 'chatcmpl-' + secrets.token_hex(12);created = int(time.time())
            model = body.get('model') if isinstance(body.get('model'), str) else proxy.model_id
            include_usage = isinstance(body.get('stream_options'), dict) and body['stream_options'].get('include_usage') is True
            def write(row):
                if not alive[0]:return
                try:self.wfile.write(b'data: ' + json.dumps(row, separators=(',', ':'), ensure_ascii=False).encode() + b'\n\n');self.wfile.flush()
                except OSError:alive[0] = False   # client gone: finish the decode so the frontier stays exact
            final = {}
            def emit(kind, *values):
                if kind == 'delta' and streaming:write(_chunk(cid, model, created, {values[0]: values[1]}))
                elif kind == 'final':final.update(values[0])
            try:
                if engine.poison:raise RuntimeError('Single-engine owner is poisoned: ' + engine.poison)
                if 'quality_speed' in body:engine.set_quality(body['quality_speed'])
                if streaming:
                    self.send_response(200);self.send_header('Content-Type', 'text/event-stream');self.send_header('Cache-Control', 'no-cache');self.send_header('Connection', 'close');self.end_headers()
                    self.close_connection = True;headers = True;self.wfile.write(b': accepted\n\n');self.wfile.flush()
                    write(_chunk(cid, model, created, dict(role='assistant')))
                try:
                    proxy.window(True, engine.lease_id)
                    engine.run(body, emit, metrics, ready=lambda: proxy.window(True, engine.lease_id))
                finally:proxy.window(False)
                parsed, splitter = final['parsed'], final['splitter']
                calls = tool_call_deltas(parsed['calls'])
                if streaming:
                    rest = splitter.remainder('reasoning', parsed['reasoning'])
                    if rest:write(_chunk(cid, model, created, dict(reasoning_content=rest)))
                    rest = splitter.remainder('content', parsed['content'])
                    if rest:write(_chunk(cid, model, created, dict(content=rest)))
                    if calls:write(_chunk(cid, model, created, dict(tool_calls=calls)))
                    write(_chunk(cid, model, created, {}, parsed['finish']))
                    if include_usage:write(dict(id=cid, object='chat.completion.chunk', created=created, model=model, choices=[], usage=final['usage']))
                    if alive[0]:self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
                else:
                    message = dict(role='assistant', content=parsed['content'] or '')
                    if parsed['reasoning']:message['reasoning_content'] = parsed['reasoning']
                    if calls:message['tool_calls'] = [{k: v for k, v in c.items() if k != 'index'} for c in calls]
                    self.send(200, dict(id=cid, object='chat.completion', created=created, model=model,
                                        choices=[dict(index=0, message=message, finish_reason=parsed['finish'])], usage=final['usage']))
            except ds4_render.RenderRefused as exc:
                metrics['failure'] = repr(exc);error = dict(error=dict(message=str(exc), type='invalid_request_error'))
                if headers:
                    try:self.wfile.write(b'data: ' + json.dumps(error, separators=(',', ':')).encode() + b'\n\ndata: [DONE]\n\n');self.wfile.flush()
                    except OSError:pass
                else:self.send(400, error)
            except Exception as exc:
                metrics['failure'] = repr(exc)
                if engine.poison and engine.poison.startswith('Unconfirmed GPU release'):
                    os._exit(72)   # outer kill-on-close Job retires descendants before orphan-lease recovery
                try:engine.retire('request failed: ' + repr(exc)[:200])
                except BaseException:os._exit(72)
                error = dict(error=dict(message=str(exc), type='single_engine_refused'))
                if headers:
                    try:self.wfile.write(b'data: ' + json.dumps(error, separators=(',', ':')).encode() + b'\n\ndata: [DONE]\n\n');self.wfile.flush()
                    except OSError:pass
                else:self.send(503, error)
            finally:
                try:atomic_json(Path(proxy.cfg['runtime_root']) / ('single-request-' + secrets.token_hex(8) + '.json'), metrics)
                except OSError:pass
                engine.lock.release()
    return Handler


def serve(cfg, run_id, window):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_uuid'];os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    proxy = SingleProxy(cfg, run_id, window)
    class Server(ThreadingHTTPServer):daemon_threads = True
    server = Server((cfg['host'], cfg['port']), handler(proxy))
    def idle():
        while not proxy.engine.stopping.wait(15):proxy.engine.idle_check()
    threading.Thread(target=idle, daemon=True).start()
    print('DS4SINGLE ready; ULTRA producer prefill + in-process ds4 decode, one engine', flush=True)
    try:server.serve_forever(poll_interval=.1)
    finally:
        server.server_close();proxy.engine.close()
