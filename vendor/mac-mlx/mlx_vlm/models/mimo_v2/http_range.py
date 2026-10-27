"""System HTTP transport for macOS processes whose Python LAN route is denied.

Uses the same authenticated Studio URL and exact byte range. Credentials travel
over stdin, headers are checked before reading the body, and no file is staged.
"""
import contextlib
import email.parser
import json
import subprocess
from types import SimpleNamespace


@contextlib.contextmanager
def curl_range_response(request, size):
    config = '\n'.join('header = ' + json.dumps(f'{key}: {value}')
                       for key, value in request.header_items()) + '\n'
    proc = subprocess.Popen([
        '/usr/bin/curl', '--silent', '--show-error', '--include', '--no-buffer',
        '--noproxy', '*', '--max-time', '180', '--max-filesize', str(size),
        '--config', '-', '--url', request.full_url,
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        proc.stdin.write(config.encode())
        proc.stdin.close()
        status_line = proc.stdout.readline(8193)
        if len(status_line) > 8192 or not status_line.startswith(b'HTTP/'):
            raise OSError('system HTTP transport returned no valid status')
        status = int(status_line.split()[1])
        lines, total = [], 0
        while True:
            line = proc.stdout.readline(8193)
            total += len(line)
            if len(line) > 8192 or total > 65536:
                raise OSError('system HTTP transport returned oversized headers')
            if line in (b'\r\n', b'\n'):
                break
            if not line:
                raise OSError('system HTTP transport ended in response headers')
            lines.append(line)
        headers = email.parser.BytesParser().parsebytes(b''.join(lines))
        # The caller checks 206, Content-Range and checkpoint ETag before any
        # bytes are consumed. Its bounded read cannot turn into a shard download.
        yield SimpleNamespace(status=status, headers=headers, read=proc.stdout.read)
    finally:
        if not proc.stdin.closed:
            proc.stdin.close()
        proc.stdout.close()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=3)
        proc.stderr.close()
