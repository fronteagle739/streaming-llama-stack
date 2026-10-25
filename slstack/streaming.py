"""Bounded OpenAI SSE forwarding for the existing HTTP endpoints."""
import json
import urllib.error
import urllib.request


class UpstreamError(RuntimeError):
    """An engine error reply, kept as its plain message, type and HTTP status."""
    def __init__(self, error, status=None):
        error = error if isinstance(error, dict) else {'message': str(error)}
        self.message = str(error.get('message') or 'Engine error')
        self.type = str(error.get('type') or 'engine_error')
        self.status = status or (error.get('code') if isinstance(error.get('code'), int) else None)
        super().__init__(self.message)


def _error(exc, status):
    """llama-server's error object: {"code", "message", "type"}."""
    if isinstance(exc, UpstreamError):
        return {'code': exc.status or status, 'message': exc.message, 'type': exc.type}
    return {'code': status, 'message': str(exc), 'type': 'engine_error'}


def events(url, body, *, timeout, headers=None):
    request = urllib.request.Request(url, data=json.dumps(body, allow_nan=False).encode(),
        headers={'Content-Type':'application/json','Accept':'text/event-stream', **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            while True:
                line = response.readline((1 << 20) + 1)
                if not line:
                    raise RuntimeError('Upstream stream ended before [DONE]')
                if len(line) > 1 << 20:
                    raise RuntimeError('Upstream SSE line exceeds 1 MiB')
                if line.startswith(b'data:'):
                    value = line[5:].strip()
                    if value == b'[DONE]':
                        return
                    row = json.loads(value)
                    if not isinstance(row, dict):
                        raise ValueError('Upstream SSE data must be an object')
                    if row.get('error'):
                        raise UpstreamError(row['error'])
                    yield row
    except urllib.error.HTTPError as exc:
        text = exc.read(4096).decode(errors='replace')
        try:
            error = json.loads(text)['error']
        except (ValueError, KeyError, TypeError):
            error = {'message': f'Upstream HTTP {exc.code}: {text}'}
        raise UpstreamError(error, exc.code) from exc


def _role_only(row):
    choices = row.get('choices')
    return bool(choices) and all(set(c.get('delta') or {}) <= {'role'} and not c.get('finish_reason') for c in choices)


def send(handler, source):
    """Hold events until one carries more than the assistant role, then commit headers and
    flush each chunk. Like llama-server, an error before that point is a plain HTTP error reply
    (the web UI shows only non-200 bodies as errors), not a 200 stream."""
    source = iter(source)
    held = []
    try:
        for row in source:
            held.append(row)
            if not _role_only(row):
                break
    except Exception as exc:
        if hasattr(source, 'close'):
            source.close()
        error = _error(exc, 503)
        body = json.dumps({'error': error}).encode()
        handler.send_response(error['code'] if isinstance(error['code'], int) and error['code'] >= 400 else 503)
        handler.send_header('Content-Type', 'application/json')
        handler.send_header('Content-Length', str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
        handler.wfile.flush()
        return
    if not held:
        raise RuntimeError('Upstream returned an empty stream')
    handler.send_response(200)
    handler.send_header('Content-Type','text/event-stream')
    handler.send_header('Cache-Control','no-cache')
    handler.send_header('Transfer-Encoding','chunked')
    handler.send_header('X-Content-Type-Options','nosniff')
    handler.end_headers()
    def write(payload):
        handler.wfile.write(f'{len(payload):x}\r\n'.encode() + payload + b'\r\n')
        handler.wfile.flush()
    def data(row):
        write(('data: ' + json.dumps(row, allow_nan=False) + '\n\n').encode())
    last = held[-1]
    try:
        for row in held:
            data(row)
        for row in source:
            last = row
            data(row)
        write(b'data: [DONE]\n\n')
        handler.wfile.write(b'0\r\n\r\n')
        handler.wfile.flush()
    except OSError:
        handler.close_connection = True
    except Exception as exc:
        try:
            # The web UI drops a data: {"error": ...} row (it reads only choices[0].delta), so the
            # message also goes out as visible content; the error row keeps llama-server's format.
            error = _error(exc, 500)
            data({'id': last.get('id'), 'object': 'chat.completion.chunk', 'created': last.get('created'),
                  'model': last.get('model'),
                  'choices': [{'index': 0, 'delta': {'content': '\n\n**Error:** ' + error['message']}, 'finish_reason': None}]})
            data({'error': error})
            write(b'data: [DONE]\n\n')
            handler.wfile.write(b'0\r\n\r\n')
            handler.wfile.flush()
        except OSError:
            handler.close_connection = True
    finally:
        if hasattr(source,'close'):
            source.close()
