"""Bounded OpenAI SSE forwarding for the existing HTTP endpoints."""
import json
import urllib.error
import urllib.request


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
                        raise RuntimeError(str(row['error']))
                    yield row
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'Upstream HTTP {exc.code}: {exc.read(4096).decode(errors="replace")}') from exc


def send(handler, source):
    """Pull the first event before committing headers, then flush each chunk."""
    source = iter(source)
    try:
        first = next(source)
    except StopIteration:
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
    try:
        data(first)
        for row in source:
            data(row)
        write(b'data: [DONE]\n\n')
        handler.wfile.write(b'0\r\n\r\n')
        handler.wfile.flush()
    except OSError:
        handler.close_connection = True
    except Exception as exc:
        try:
            data({'error':{'message':str(exc),'type':'engine_error'}})
            write(b'data: [DONE]\n\n')
            handler.wfile.write(b'0\r\n\r\n')
            handler.wfile.flush()
        except OSError:
            handler.close_connection = True
    finally:
        if hasattr(source,'close'):
            source.close()
