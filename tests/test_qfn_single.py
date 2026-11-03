"""The QFN front end against a scripted stand-in engine on 127.0.0.1: streaming and non-streaming chat, auth, retire."""
from contextlib import nullcontext
import http.client
import json
import socket
import subprocess
import sys
import threading
import pytest

from slstack import qfn_single

FAKE = r'''
import json, socket, sys
srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", int(sys.argv[1]))); srv.listen(1)
conn, _ = srv.accept(); f = conn.makefile("rwb")
def out(row):
    f.write((json.dumps(row) + "\n").encode()); f.flush()
out({"ok": True, "event": "ready", "n_ctx": 4096, "n_vocab": 10, "load_seconds": 0.0})
for line in f:
    req = json.loads(line)
    with open(sys.argv[2], 'a', encoding='utf-8') as log:
        log.write(json.dumps(req) + '\n')
    if req["op"] == "quit":
        out({"ok": True, "event": "bye"}); break
    if req["op"] == "cancel":
        continue
    if req["op"] == "chat":
        body = req["body"]
        out({"ok": True, "event": "prefilled", "prompt_tokens": 5, "prefill": {"route": "layer-major", "reused": 0, "new_tokens": 5, "seconds": 0.1}})
        out({"ok": True, "delta": {"reasoning_content": "think"}})
        out({"ok": True, "delta": {"content": "Hel"}})
        out({"ok": True, "delta": {"content": "lo bias=%s" % body["bias"]}})
        out({"ok": True, "done": True, "finish_reason": "stop",
             "message": {"role": "assistant", "content": "Hello bias=%s" % body["bias"], "reasoning_content": "think"},
             "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
             "timings": {"predicted_n": 3}, "stats": {"tok_per_s": 10.0}})
'''


class FakeJob:
    def __init__(self, argv):
        self.proc = subprocess.Popen(argv)
    def wait(self):
        return self.proc.wait()
    def stop(self):
        self.proc.terminate();self.proc.wait()


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def start(tmp_path, overrides=None):
    fake = tmp_path / 'fake_engine.py';fake.write_text(FAKE)
    port = free_port()
    cfg = dict(model=str(tmp_path / 'x.basalt'), runtime_root=str(tmp_path), host='127.0.0.1', port=0, api_key='k', qfn_engine_port=port)
    cfg.update(overrides or {})
    engine = qfn_single.QfnEngine(cfg, lease_factory=nullcontext, launch_factory=lambda:FakeJob([
        sys.executable, '-B', '-u', str(fake), str(port), str(tmp_path / 'requests.jsonl')]))
    state = dict(cfg=cfg, engine=engine, stopping=threading.Event(), model_id='qfn-test')
    server = qfn_single.ThreadingHTTPServer(('127.0.0.1', 0), qfn_single.handler(state));state['server'] = server
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, engine


def post(server, body, token='k'):
    conn = http.client.HTTPConnection(*server.server_address, timeout=30)
    conn.request('POST', '/v1/chat/completions', json.dumps(body), {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
    return conn.getresponse()


def test_chat_non_streaming_and_streaming(tmp_path):
    server, engine = start(tmp_path)
    try:
        r = post(server, dict(messages=[dict(role='user', content='hi')], quality_speed='max_speed'))
        assert r.status == 200
        reply = json.loads(r.read())
        assert reply['choices'][0]['message']['content'] == 'Hello bias=0.35'
        assert reply['choices'][0]['message']['reasoning_content'] == 'think'
        assert reply['usage']['total_tokens'] == 8
        r = post(server, dict(messages=[dict(role='user', content='hi')], stream=True))
        assert r.status == 200
        rows = [line[6:] for line in r.read().decode().split('\n') if line.startswith('data: ')]
        assert rows[-1] == '[DONE]'
        chunks = [json.loads(x) for x in rows[:-1]]
        text = ''.join(c['choices'][0]['delta'].get('content', '') for c in chunks if c.get('choices'))
        assert text == 'Hello bias=0.0'
        assert chunks[-1]['choices'][0]['finish_reason'] == 'stop'
        assert post(server, dict(messages=[]), token='wrong').status == 401
        assert post(server, dict(messages=[], quality_speed='fastest')).status == 400
    finally:
        with engine.lock:engine.retire('test')
        server.shutdown()
    assert engine.job is None


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('fields', [
    dict(temperature=1.0, top_k=20, top_p=.95, min_p=.05, typical_p=.8),
    dict(do_sample=False, temperature=.7, top_k=20, top_p=.95, min_p=.05, typical_p=.8),
    dict(do_sample=True, temperature=.8, repeat_penalty=1.2, seed=0),
    dict(repeat_penalty=1.7, repeat_last_n=-1, presence_penalty=2.0, frequency_penalty=.7),
    dict(repetition_penalty=1.3, repeat_last_n=0),
    dict(seed=0, stop=['end', '\n\n'], max_tokens=7),
    dict(seed=987654321, stop='halt', max_completion_tokens=0),
    dict(logit_bias={'42': -100, 'hello world': 2.0}),
    dict(logit_bias=[[42, False], ['hello world', -3.0]]),
])
def test_sampler_fields_reach_engine_unchanged(tmp_path, streaming, fields):
    server, engine = start(tmp_path)
    try:
        body = dict(messages=[dict(role='user', content='hi')], stream=streaming, **fields)
        response = post(server, body)
        assert response.status == 200
        reply = response.read().decode()
        received = json.loads((tmp_path / 'requests.jsonl').read_text().splitlines()[0])
        assert received['op'] == 'chat'
        assert received['body'] == {**body, 'bias': 0.0}
        if streaming:
            assert 'data: [DONE]' in reply
            assert 'lo bias=0.0' in reply
        else:
            assert json.loads(reply)['choices'][0]['message']['content'] == 'Hello bias=0.0'
    finally:
        with engine.lock:engine.retire('test')
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize('mode,expected', [('max_quality', 0), ('balanced', .12), ('max_speed', .35), (8, 8)])
def test_machine_settings_reach_chat_without_changing_samplers(tmp_path, mode, expected):
    settings = json.loads((qfn_single.ROOT/'examples/single-machine/qfn.settings.json').read_text())
    settings['bias']['balanced'] = .12
    server, engine = start(tmp_path, {'model_settings': settings})
    try:
        fields = dict(temperature=.7, typical_p=.8, stop=['done'], seed=31, max_tokens=7)
        response = post(server, dict(messages=[dict(role='user', content='hi')], quality_speed=mode, **fields))
        assert response.status == 200
        response.read()
        received = json.loads((tmp_path/'requests.jsonl').read_text().splitlines()[0])['body']
        assert received['bias'] == expected
        assert all(received[k] == value for k, value in fields.items())
    finally:
        with engine.lock:engine.retire('test')
        server.shutdown()
        server.server_close()
