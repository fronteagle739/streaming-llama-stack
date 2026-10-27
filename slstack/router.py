from __future__ import annotations
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import gzip
import mimetypes
import re
from pathlib import Path
import threading
import time
from urllib.parse import unquote, urlsplit, parse_qs
import uuid
from .config import path, ROOT
from .engines import Engines
from .tools import Tools
from .tool_activity import activity, outcome, without_activity, discovery_results


def media_request(kind, body, cfg):
    """Map the common OpenAI image fields onto the startup-selected Studio tool."""
    body = dict(body)
    if 'model' in body:
        model = body.pop('model')
        selected = cfg[kind + '_model']
        allowed = {kind, cfg[kind + '_arch'], Path(selected).stem if selected else None}
        if model not in allowed:
            raise ValueError('model must name the startup-selected ' + kind + ' model (alias: ' + kind + ')')
    count = body.pop('n', 1)
    if type(count) is not int or count != 1:
        raise ValueError('This serialized media endpoint supports n=1')
    if 'size' in body:
        match = re.fullmatch(r'(\d+)x(\d+)', str(body.pop('size')))
        if not match:
            raise ValueError('size must be WIDTHxHEIGHT')
        for key, value in zip(('width', 'height'), map(int, match.groups())):
            if key in body and body[key] != value:
                raise ValueError('size conflicts with ' + key)
            body[key] = value
    response_format = body.pop('response_format', 'url')
    if response_format != 'url':
        raise ValueError('This endpoint returns response_format=url')
    return body


class Stack:
    def __init__(self, cfg, engines=None):
        self.cfg = cfg
        if engines is None and cfg.get('general'):
            from .general_engine import GeneralEngines
            engines=GeneralEngines(cfg)
        self.engines = engines or Engines(cfg)
        from .models import Models
        self.models = getattr(self.engines, 'models', None) or Models(cfg)
        self.tools = Tools(cfg, self.engines)
        self.preferences = self.tools.preferences
        from .generation import Store
        self.generation = Store(cfg)

    def prepare_chat(self, body, *, stream=False, references=None):
        if not isinstance(body.get('messages'), list) or not body['messages']:
            raise ValueError('messages must be a nonempty array')
        body = deepcopy(body)
        discoveries = {}
        for index, message in enumerate(body['messages']):
            if message.get('role')=='assistant' and isinstance(message.get('content'),str):
                for result in discovery_results(message['content']):
                    key=(result['tool'],json.dumps(result['arguments'],sort_keys=True))
                    anchor=discoveries[key][0] if key in discoveries else index
                    discoveries[key]=(anchor,result)
                message['content']=without_activity(message['content'])
        for anchor, result in discoveries.values():
            # A fixed original turn keeps subsequent prompt prefixes stable;
            # a deliberate refresh replaces the facts at that same anchor.
            body['messages'][anchor]['content'] += '\n\nKnown media discovery result (reuse until refreshed):\n' + json.dumps(result)
        if references is not None:references.annotate(body['messages'])
        # Client-owned tools remain client-owned. Stack tool names cannot be overridden.
        stack_names = set(self.tools.schemas)
        client_tools = body.get('tools', [])
        for tool in client_tools:
            if tool.get('function', {}).get('name') in stack_names:
                raise ValueError('Client tool overrides a reserved stack tool')
        body['stream'] = stream
        body['tools'] = client_tools + self.tools.openai()
        return body

    def tool_result(self, call, media, references=None):
        try:
            args = json.loads(call['function']['arguments'])
            if references is not None and (args.get('source_image') or args.get('reference_images')):
                outcome = self.tools.call(call['function']['name'], args, references=references)
            else:outcome = self.tools.call(call['function']['name'], args)
            if call['function']['name'].startswith('generate_'):
                media.append(outcome)
                outcome = {'data': [{'url': x['url']} for x in outcome.get('data', [])],
                           'settings': outcome.get('settings', {}), 'mode': outcome.get('mode'),
                           'display': 'The client displays this media inline. Generated image URLs may be used as source_image or in reference_images for a subsequent requested generation. Reply with a brief confirmation.'}
        except (ValueError, RuntimeError, OSError, MemoryError) as exc:
            outcome = {'error': str(exc)}
        content=json.dumps(outcome)
        if call['function']['name']=='generate_image' and self.cfg.get('vision_bridge_version')==1 and self.cfg.get('vision_encoder') and outcome.get('data'):
            content=[{'type':'text','text':content},*[{'type':'image_url','image_url':{'url':item['url']}} for item in outcome['data']]]
        return {'role': 'tool', 'tool_call_id': call['id'], 'content': content}

    def chat(self, body):
        from .media_references import References
        references=References(self.cfg,scoped=True)
        body = self.prepare_chat(body,references=references)
        messages = body['messages']
        stack_names = set(self.tools.schemas)
        media, activity_log = [], []
        while True:
            result = self.engines.chat(body)
            choices = result.get('choices', [])
            if len(choices) != 1:
                raise RuntimeError('Stack tools require exactly one upstream choice')
            message = choices[0]['message']
            calls = message.get('tool_calls') or []
            if not calls:
                content = ''.join(activity_log) + (message.get('content') or '')
                if message.get('tool_error'):
                    content += activity(dict(id='invalid-'+uuid.uuid4().hex,function=dict(name='Malformed tool call',arguments='{}')),
                                        'failed',message.pop('tool_error'))
                for item in media:
                    for artifact in item.get('data', []):
                        content += '\n\n' + f'![Generated media]({artifact["url"]})'
                message['content'] = content
                paired = bool(self.cfg.get('mac'))
                result['slstack'] = {'route': 'rtx-prefill-mac-decode' if paired else
                                      ('single-engine' if getattr(self.engines, 'single_engine', lambda: False)() else 'ds4-native'), 'tool_media': media,
                                      'buffered_stream': True,
                                      'ultra_handoff': paired and self.cfg.get('native_tier') == 'ultra'}
                return result
            names = [call['function']['name'] for call in calls]
            if not all(name in stack_names for name in names):
                if any(name in stack_names for name in names):
                    raise RuntimeError('Mixed client/stack tool calls are not supported in one upstream message')
                return result
            messages.append(message)
            for call in calls:
                result = self.tool_result(call, media,references)
                messages.append(result)
                reply = outcome(result)
                activity_log.append(activity(call,'failed' if 'error' in reply else 'complete',reply))

    def chat_events(self, body):
        from .media_references import References
        references=References(self.cfg,scoped=True)
        body = self.prepare_chat(body, stream=True,references=references)
        media, metadata = [], {}
        names = set(self.tools.schemas)
        identity = 'chatcmpl-' + uuid.uuid4().hex
        def chunk(delta, finish=None):
            row = {'id':identity, 'object':'chat.completion.chunk','created':int(time.time()),
                   'model':body.get('model') or self.models.default,
                   'choices':[{'index':0,'delta':delta,'finish_reason':finish}]}
            if finish:
                row.update(metadata)
                row['slstack'] = {'route':'rtx-prefill-mac-decode' if self.cfg.get('mac') else 'single-engine','tool_media':media,
                                   'buffered_stream':False,'ultra_handoff':self.cfg.get('native_tier')=='ultra'}
            return row
        def progress_chunk(progress):
            # The web UI draws its prompt-processing bar from llama-server's prompt_progress.
            row = {'id':identity, 'object':'chat.completion.chunk','created':int(time.time()),
                   'model':body.get('model') or self.models.default, 'choices':[], 'prefill_progress':progress}
            try:
                row['prompt_progress'] = {'total':int(progress['tokens_total']), 'cache':0,
                                          'processed':int(progress['tokens_done']),
                                          'time_ms':int(float(progress.get('elapsed_s') or 0) * 1000)}
            except (KeyError, TypeError, ValueError):
                pass
            return row
        while True:
            message = {'role':'assistant','content':''}
            calls, finish = {}, 'stop'
            for event in self.engines.chat_events(body):
                metadata.update({k:event[k] for k in ('usage','timings') if k in event})
                choices = event.get('choices',[])
                if not choices:
                    progress = event.get('prefill_progress')
                    if isinstance(progress, dict):
                        yield progress_chunk(progress)
                    if 'timings' in event:
                        row = chunk({});row['choices'] = [];row['timings'] = event['timings'];yield row
                    continue
                if len(choices) != 1:
                    raise RuntimeError('Stack tools require exactly one upstream choice')
                choice = choices[0]
                delta = dict(choice.get('delta') or {})
                diagnostic = delta.pop('tool_error', None)
                if diagnostic:
                    call=dict(id='invalid-'+uuid.uuid4().hex,function=dict(name='Malformed tool call',arguments='{}'))
                    yield chunk({'content':activity(call,'failed',diagnostic)})
                for part in delta.pop('tool_calls',[]) or []:
                    index = part.get('index',0)
                    call = calls.setdefault(index, {'id':'','type':'function','function':{'name':'','arguments':''}})
                    if part.get('id'):
                        call['id'] = part['id']
                    function = part.get('function') or {}
                    if function.get('name'):
                        call['function']['name'] += function['name']
                    call['function']['arguments'] += function.get('arguments') or ''
                for key in ('content','reasoning_content'):
                    if isinstance(delta.get(key),str):
                        message[key] = message.get(key,'') + delta[key]
                if delta:
                    row = chunk(delta)
                    if 'timings' in event:row['timings'] = event['timings']
                    yield row
                if choice.get('finish_reason'):
                    finish = choice['finish_reason']
            calls = [calls[index] for index in sorted(calls)]
            if not calls:
                yield chunk({},finish)
                return
            owned = [call['function']['name'] in names for call in calls]
            if not all(owned):
                if any(owned):
                    raise RuntimeError('Mixed client/stack tool calls are not supported in one upstream message')
                yield chunk({'tool_calls':[{'index':i,**call} for i,call in enumerate(calls)]})
                yield chunk({},finish)
                return
            message['tool_calls'] = calls
            body['messages'].append(message)
            for call in calls:
                yield chunk({'content':activity(call,'running')})
                before = len(media)
                result = self.tool_result(call,media,references)
                body['messages'].append(result)
                reply = outcome(result)
                yield chunk({'content':activity(call,'failed' if 'error' in reply else 'complete',reply)})
                markdown = ''.join('\n\n![Generated media](' + item['url'] + ')\n\n'
                                   for result in media[before:] for item in result.get('data',[]))
                if markdown:
                    # Flush artifacts before the next vision prefill or model reply.
                    yield chunk({'content':markdown})

    def mcp(self, body):
        identity, method = body.get('id'), body.get('method')
        if identity is None:
            return None
        try:
            if method == 'initialize':
                result = {'protocolVersion': '2024-11-05', 'capabilities': {'tools': {}},
                          'serverInfo': {'name': 'streaming-llama-stack', 'version': '0.1.0'}}
            elif method == 'tools/list':
                result = {'tools': self.tools.catalog()}
            elif method == 'tools/call':
                params = body.get('params') or {}
                if params['name'] in ('set_quality_speed', 'get_quality_speed'):
                    outcome = self.tools.call(params['name'], params.get('arguments', {}))
                else:
                    with self.engines.lock:
                        outcome = self.tools.call(params['name'], params.get('arguments', {}))
                result = {'content': [{'type': 'text', 'text': json.dumps(outcome)}], 'isError': False}
            elif method == 'ping':
                result = {}
            else:
                return {'jsonrpc': '2.0', 'id': identity, 'error': {'code': -32601, 'message': 'Method not found'}}
            return {'jsonrpc': '2.0', 'id': identity, 'result': result}
        except (ValueError, RuntimeError, OSError, MemoryError, KeyError) as exc:
            return {'jsonrpc': '2.0', 'id': identity, 'result': {'isError': True,
                    'content': [{'type': 'text', 'text': str(exc)}]}}


def handler(stack):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, fmt, *args):
            # Do not log prompts, model paths or private tool arguments.
            pass

        def send(self, code, data, mime='application/json', extra=None):
            payload = json.dumps(data, allow_nan=False).encode() if mime == 'application/json' else data
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            route = unquote(urlsplit(self.path).path)
            if route == '/health':
                ready = bool(stack.engines.llm and stack.engines.llm.proc.poll() is None)
                remote = getattr(stack.engines, 'remote', None)
                ready = ready or bool(remote and remote.connected)
                events = getattr(stack.engines, 'events', None) or []
                return self.send(200, {'status': 'router-ready', 'model_ready': ready,
                                       'model_status': events[-1]['message'] if events else None,
                                       'live_bias_configured': bool(stack.cfg['bias_mcp_url']), 'ultra': False})
            if route == '/v1/models':
                return self.send(200, stack.engines.catalog() if hasattr(stack.engines, 'catalog') else stack.models.catalog())
            if route == '/slstack/status':
                progress = getattr(stack.engines, 'media_progress', None)
                return self.send(200, {'events': getattr(stack.engines, 'events', []), 'dual_resident_active': False,
                                       'media_progress': progress.snapshot() if progress else None},
                                 extra={'Cache-Control':'no-store'})
            if route == '/slstack/generation':
                return self.send(200, dict(stack.generation.state(), supported=True,
                    applies='next request' if stack.cfg.get('mac') else 'next token boundary'))
            if route == '/slstack/settings':
                return self.send(200,stack.preferences.state())
            if route == '/slstack/browse':
                from .preferences import browse
                try:return self.send(200,browse(parse_qs(urlsplit(self.path).query).get('path',[''])[0]))
                except (OSError,ValueError) as exc:return self.send(400,{'error':{'message':str(exc)}})
            if route == '/props':
                command = stack.cfg['ds4_command']
                if stack.cfg.get('mac'):
                    n_ctx = 1000001
                elif getattr(stack.engines, 'single_engine', lambda: False)():
                    from .single_runtime import context
                    n_ctx = context(stack.cfg)
                else:
                    n_ctx = next((int(command[i + 1]) for i, x in enumerate(command[:-1]) if x in ('--ctx', '-c')), 4096)
                selected = parse_qs(urlsplit(self.path).query).get('model', [None])[0]
                if selected is not None:
                    try:
                        stack.models.resolve(selected)
                    except ValueError as exc:
                        return self.send(400, {'error': {'message': str(exc)}})
                return self.send(200, {'role': 'router' if len(stack.models.rows) > 1 and selected is None else 'model',
                                      'max_instances': 1, 'model_alias': selected or stack.models.default, 'model_path': '',
                                      'default_generation_settings': {'n_ctx': n_ctx, 'params': {}},
                                      'total_slots': 1, 'chat_template': '', 'modalities': {'vision': stack.cfg.get('vision_bridge_version')==1 and bool(stack.cfg.get('vision_encoder')), 'audio': False}})
            if route == '/mcp':
                return self.send(405, {'error': {'message': 'Stateless MCP endpoint; POST JSON-RPC'}})
            if route == '/slstack-media.js' and stack.cfg['webui']:
                return self.send(200, (ROOT / 'slstack' / 'media.js').read_bytes(), 'text/javascript; charset=utf-8',
                                 {'Cache-Control':'no-store'})
            if route == '/slstack-generation.js' and stack.cfg['webui']:
                return self.send(200, (ROOT / 'slstack' / 'generation.js').read_bytes(), 'text/javascript; charset=utf-8',
                                 {'Cache-Control': 'no-store'})
            if route == '/slstack-settings.js' and stack.cfg['webui']:
                return self.send(200,(ROOT/'slstack'/'settings.js').read_bytes(),'text/javascript; charset=utf-8',{'Cache-Control':'no-store'})
            if route.startswith('/outputs/'):
                root = Path(stack.cfg['runtime_root']).resolve() / 'outputs'
                target = (root / route.removeprefix('/outputs/')).resolve()
            elif stack.cfg['webui']:
                root = path(stack.cfg['webui_root'])
                target = (root / (route.lstrip('/') or 'index.html')).resolve()
            else:
                return self.send(404, {'error': {'message': 'Not found; enable --webui for chat'}})
            if not target.is_relative_to(root) or not target.is_file():
                if route == '/' and target.with_suffix('.html.gz').is_file():
                    target = target.with_suffix('.html.gz')
                else:
                    return self.send(404, {'error': {'message': 'Artifact not found'}})
            if route == '/':
                html = gzip.decompress(target.read_bytes()) if target.suffix == '.gz' else target.read_bytes()
                html = html.replace(b'</body>', b'<script defer src="/slstack-media.js"></script><script defer src="/slstack-generation.js"></script><script defer src="/slstack-settings.js"></script></body>')
                return self.send(200, html, 'text/html; charset=utf-8')
            mime = mimetypes.guess_type(str(target))[0] or 'application/octet-stream'
            if route.startswith('/outputs/') and target.suffix.lower() in ('.mp4','.webm'):
                return self.video_file(target,mime)
            return self.send(200, target.read_bytes(), mime)

        def video_file(self,target,mime):
            """Stream bounded chunks and honor browser seeks without buffering the clip."""
            import re
            size=target.stat().st_size;first,last=0,size-1
            requested=self.headers.get('Range')
            if requested:
                match=re.fullmatch(r'bytes=(\d*)-(\d*)',requested)
                valid=bool(match and any(match.groups()))
                if valid:
                    a,b=match.groups()
                    if a:first,last=int(a),min(int(b) if b else size-1,size-1)
                    else:first,last=max(0,size-int(b)),size-1
                    valid=0<=first<=last<size
                if not valid:
                    self.send_response(416);self.send_header('Content-Range',f'bytes */{size}')
                    self.send_header('Content-Length','0');self.end_headers();return
            self.send_response(206 if requested else 200)
            self.send_header('Content-Type',mime);self.send_header('Accept-Ranges','bytes')
            self.send_header('Content-Length',str(max(0,last-first+1)))
            if requested:self.send_header('Content-Range',f'bytes {first}-{last}/{size}')
            self.end_headers()
            try:
                with target.open('rb') as source:
                    source.seek(first);remaining=last-first+1
                    while remaining>0:
                        block=source.read(min(65536,remaining))
                        if not block:break
                        self.wfile.write(block);remaining-=len(block)
            except (BrokenPipeError,ConnectionResetError):pass

        def do_POST(self):
            try:
                length = int(self.headers.get('Content-Length', '0'))
                from .request_limits import chat_body_limit
                limit=chat_body_limit(stack.cfg) if urlsplit(self.path).path=='/v1/chat/completions' else 4<<20
                if not 0 < length <= limit:
                    return self.send(413, {'error': {'message': f'Request limit is {limit//(1<<20)} MiB'}})
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError('JSON body must be an object')
                route = urlsplit(self.path).path
                if route == '/slstack/settings':
                    return self.send(200,stack.preferences.update(body))
                if route == '/slstack/generation':
                    return self.send(200, dict(settings=stack.generation.update(body), supported=True,
                        scope='session', applies='next request' if stack.cfg.get('mac') else 'next token boundary'))
                if route == '/v1/stack/shutdown':
                    # Loopback-only router; the launcher asks for a graceful stop before any forced one.
                    threading.Thread(target=server_ref[0].shutdown, daemon=True).start()
                    return self.send(200, {'stopping': True})
                if route == '/mcp':
                    result = stack.mcp(body)
                    return self.send(200, result) if result is not None else self.send(202, b'', 'text/plain')
                if route in ('/models/load', '/models/unload'):
                    if set(body) != {'model'}:
                        raise ValueError('Model control accepts only model')
                    with stack.engines.lock:
                        row = stack.models.resolve(body['model'])
                        if route == '/models/load':
                            stack.engines.select(row['id'])
                            stack.engines.ready()
                        elif row['id'] == stack.engines.selected_model:
                            stack.engines.close()
                    return self.send(200, {'success': True})
                if route == '/v1/chat/completions':
                    with stack.engines.lock:
                        if body.get('stream') and (stack.cfg.get('mac') or getattr(stack.engines, 'single_engine', lambda: False)()):
                            from .streaming import send
                            return send(self,stack.chat_events(body))
                        result = stack.chat(body)
                    if not body.get('stream'):
                        return self.send(200, result)
                    # The old pinned webui expects SSE. Tool execution is buffered before this final response.
                    choice = result['choices'][0]
                    chunk = {'id': result.get('id', 'chatcmpl-' + uuid.uuid4().hex),
                             'object': 'chat.completion.chunk', 'created': result.get('created', int(time.time())),
                             'model': result.get('model', body.get('model') or stack.models.default), 'choices': [{'index': 0, 'delta': choice['message'], 'finish_reason': None}]}
                    end = {**chunk, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': choice.get('finish_reason', 'stop')}],
                           'usage': result.get('usage', {}), 'timings': result.get('timings', {})}
                    data = ''.join('data: ' + json.dumps(c) + '\n\n' for c in (chunk, end)) + 'data: [DONE]\n\n'
                    return self.send(200, data.encode(), 'text/event-stream')
                kinds = {'/v1/images/generations': 'image', '/v1/videos/generations': 'video'}
                if route in kinds:
                    kind = kinds[route]
                    with stack.engines.lock:
                        return self.send(200, stack.tools.call('generate_' + kind, media_request(kind, body, stack.cfg)))
                return self.send(404, {'error': {'message': 'Unknown endpoint'}})
            except (ValueError, KeyError, TypeError) as exc:
                return self.send(400, {'error': {'message': str(exc), 'type': 'invalid_request_error'}})
            except (RuntimeError, OSError, MemoryError) as exc:
                return self.send(503, {'error': {'message': str(exc), 'type': 'engine_unavailable'}})
    return Handler


server_ref = [None]


def serve(cfg):
    stack = Stack(cfg)
    server = ThreadingHTTPServer((cfg['host'], cfg['port']), handler(stack))
    server_ref[0] = server
    server.daemon_threads = True
    try:
        print(f'slstack router http://{cfg["host"]}:{cfg["port"]}; model loads on first request', flush=True)
        server.serve_forever()
    finally:
        server.server_close()
        stack.engines.close()
