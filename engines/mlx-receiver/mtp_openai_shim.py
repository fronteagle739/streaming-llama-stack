from __future__ import annotations
import argparse
import json
import socket
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional
from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

def _recv_lines(sock: socket.socket, idle_timeout_s: float) -> Iterator[Dict[str, Any]]:
    sock.settimeout(idle_timeout_s)
    buf = b''
    while True:
        while b'\n' not in buf:
            chunk = sock.recv(8192)
            if not chunk:
                return
            buf += chunk
        nl = buf.index(b'\n')
        line = buf[:nl].decode('utf-8', errors='replace')
        buf = buf[nl + 1:]
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        yield msg
        if msg.get('done'):
            return

def decode_via_receiver(*, host: str, port: int, prompt_token_ids: List[int], max_tokens: Optional[int], stop_words: Optional[List[str]]=None, sampling: Optional[Dict[str, Any]]=None, swi_reasoning: Optional[Dict[str, Any]]=None, connect_timeout_s: float=10.0, idle_timeout_s: float=600.0) -> Iterator[Dict[str, Any]]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(connect_timeout_s)
    sock.connect((host, port))
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass
    cfg: Dict[str, Any] = {'stop_words': list(stop_words or []), 'prompt_token_ids': [int(t) for t in prompt_token_ids]}
    if max_tokens is not None and int(max_tokens) >= 0:
        cfg['max_tokens'] = int(max_tokens)
    if sampling:
        cfg['sampling_profile'] = sampling
    if swi_reasoning:
        cfg['swi_reasoning'] = swi_reasoning
    try:
        sock.sendall((json.dumps(cfg, separators=(',', ':')) + '\n').encode('utf-8'))
        yield from _recv_lines(sock, idle_timeout_s)
    finally:
        try:
            sock.close()
        except OSError:
            pass

def _sampling_from_request(body: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for src, dst in (('temperature', 'temp'), ('top_p', 'top_p'), ('top_k', 'top_k'), ('min_p', 'min_p')):
        if body.get(src) is not None:
            out[dst] = body[src]
    return out

def _swi_reasoning_from_body(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:

    def _norm(raw: Any) -> Optional[Dict[str, Any]]:
        if isinstance(raw, dict):
            out = dict(raw)
            out.setdefault('enabled', True)
            return out if out.get('enabled') else None
        return None
    swi = _norm(body.get('swi_reasoning'))
    if swi is not None:
        return swi
    eb = body.get('extra_body')
    if isinstance(eb, dict):
        swi = _norm(eb.get('swi_reasoning'))
        if swi is not None:
            return swi
    dt = body.get('deep_thinker')
    if isinstance(dt, dict) and str(dt.get('mode', '')).lower() == 'swireasoning':
        params = dict(dt.get('swi_reasoning') or dt.get('params') or {})
        params['enabled'] = True
        return params
    return None

def _glm_reasoning_from_body(body: Dict[str, Any]) -> tuple:
    if isinstance(body.get('enable_thinking'), bool):
        return (body['enable_thinking'], str(body.get('reasoning_effort') or '').lower() or None)
    raw = body.get('reasoning_effort')
    if raw is None:
        r = body.get('reasoning')
        raw = r.get('effort') if isinstance(r, dict) else r
    if raw is None:
        eb = body.get('extra_body')
        if isinstance(eb, dict):
            r2 = eb.get('reasoning')
            raw = eb.get('reasoning_effort') or (r2.get('effort') if isinstance(r2, dict) else r2)
    s = str(raw if raw is not None else '').strip().lower()
    if s in ('off', 'none', 'no', '0', 'false', 'minimal', 'low'):
        return (False, None)
    if s in ('medium', 'med', 'mid', 'high'):
        return (True, 'high')
    return (True, 'max')

def make_reasoning_router(think: bool):
    MARK = '</think>'
    rs = {'reasoning': bool(think), 'buf': ''}

    def route(text: str):
        if not rs['reasoning']:
            return [('content', text)] if text else []
        rs['buf'] += text
        buf = rs['buf']
        idx = buf.find(MARK)
        if idx == -1:
            keep = len(MARK) - 1
            emit = buf[:-keep] if len(buf) > keep else ''
            rs['buf'] = buf[len(emit):]
            return [('reasoning', emit)] if emit else []
        before, after = (buf[:idx], buf[idx + len(MARK):])
        rs['buf'] = ''
        rs['reasoning'] = False
        out = []
        if before:
            out.append(('reasoning', before))
        if after:
            out.append(('content', after))
        return out

    def flush():
        buf = rs['buf']
        rs['buf'] = ''
        return [('content', buf)] if buf else []
    return (route, flush)

def build_app(receiver_host: str, receiver_port: int, model_id: str, tokenizer):
    app = FastAPI(title='GLM-5.2 MTP OpenAI shim')

    @app.get('/health')
    def health():
        return {'status': 'ok', 'receiver': f'{receiver_host}:{receiver_port}'}

    @app.get('/v1/models')
    def models():
        return {'object': 'list', 'data': [{'id': model_id, 'object': 'model', 'created': 0, 'owned_by': 'mtp-decode-receiver'}]}

    @app.post('/v1/chat/completions')
    def chat_completions(body: Dict[str, Any]=Body(...)):
        messages = body.get('messages') or []
        for _m in messages:
            if not isinstance(_m, dict):
                continue
            for _tc in _m.get('tool_calls') or []:
                _fn = _tc.get('function') if isinstance(_tc, dict) else None
                if isinstance(_fn, dict) and isinstance(_fn.get('arguments'), str):
                    try:
                        _fn['arguments'] = json.loads(_fn['arguments'])
                    except Exception:
                        pass
        max_tokens = body.get('max_tokens')
        stream = bool(body.get('stream'))
        stop = body.get('stop')
        stop_words = [stop] if isinstance(stop, str) else list(stop or [])
        sampling = _sampling_from_request(body)
        cid = 'chatcmpl-' + uuid.uuid4().hex[:24]
        created = int(time.time())
        _tools = body.get('tools') or None
        _think, _effort = _glm_reasoning_from_body(body)
        _tmpl_kw = {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': _think}
        if _tools is not None:
            _tmpl_kw['tools'] = _tools
        if _effort:
            _tmpl_kw['reasoning_effort'] = _effort
        try:
            text = tokenizer.apply_chat_template(messages, **_tmpl_kw)
        except Exception:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
        swi_reasoning = _swi_reasoning_from_body(body)

        def session() -> Iterator[Dict[str, Any]]:
            return decode_via_receiver(host=receiver_host, port=receiver_port, prompt_token_ids=prompt_ids, max_tokens=max_tokens, stop_words=stop_words, sampling=sampling, swi_reasoning=swi_reasoning)
        _route, _route_flush = make_reasoning_router(_think)
        if stream:

            def sse() -> Iterator[bytes]:
                first = {'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': model_id, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]}
                yield f'data: {json.dumps(first)}\n\n'.encode()
                finish = 'stop'
                for msg in session():
                    if msg.get('done'):
                        finish = 'stop'
                        break
                    if 'version' in msg:
                        continue
                    text = msg.get('text', '')
                    if not text:
                        continue
                    for _kind, _t in _route(text):
                        if not _t:
                            continue
                        _delta = {'reasoning_content': _t} if _kind == 'reasoning' else {'content': _t}
                        chunk = {'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': model_id, 'choices': [{'index': 0, 'delta': _delta, 'finish_reason': None}]}
                        yield f'data: {json.dumps(chunk)}\n\n'.encode()
                for _kind, _t in _route_flush():
                    if _t:
                        chunk = {'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': model_id, 'choices': [{'index': 0, 'delta': {'content': _t}, 'finish_reason': None}]}
                        yield f'data: {json.dumps(chunk)}\n\n'.encode()
                last = {'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': model_id, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}]}
                yield f'data: {json.dumps(last)}\n\n'.encode()
                yield b'data: [DONE]\n\n'
            return StreamingResponse(sse(), media_type='text/event-stream')
        text_parts: List[str] = []
        n_completion = 0
        for msg in session():
            if msg.get('done'):
                n_completion = int(msg.get('tokens_generated', n_completion) or n_completion)
                break
            if 'version' in msg:
                continue
            t = msg.get('text', '')
            if t:
                text_parts.append(t)
                n_completion += 1
        content = ''.join(text_parts)
        _msg = {'role': 'assistant', 'content': content}
        if _think:
            _i = content.find('</think>')
            if _i != -1:
                _msg['reasoning_content'] = content[:_i]
                _msg['content'] = content[_i + len('</think>'):]
        return JSONResponse({'id': cid, 'object': 'chat.completion', 'created': created, 'model': model_id, 'choices': [{'index': 0, 'message': _msg, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 0, 'completion_tokens': n_completion, 'total_tokens': n_completion}})
    return app

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receiver-host', default='127.0.0.1')
    p.add_argument('--receiver-port', type=int, default=9101)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8081)
    p.add_argument('--model-id', default='glm-5.2')
    p.add_argument('--model-dir', required=True, help='model dir with tokenizer + chat_template (for templating only; no weights)')
    args = p.parse_args()
    import uvicorn
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    app = build_app(args.receiver_host, args.receiver_port, args.model_id, tokenizer)
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
