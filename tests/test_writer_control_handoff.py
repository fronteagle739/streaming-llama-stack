"""A completed control response must make the decoder available to its next call."""
import io
import threading
from types import SimpleNamespace

import pytest
from slstack.combo_single import handler
from slstack import combo_single, combo_session
from slstack import writer_directions


def control(route, monkeypatch, *, failure=None):
    lock=threading.Lock(); observations=[]
    registry=SimpleNamespace(register=lambda payload: {'registered':True},set_stack=lambda payload:{'applied':True})
    engine=SimpleNamespace(lock=lock,poison=None,directions=lambda:registry,tokenize=lambda:None)
    proxy=SimpleNamespace(engine=engine,request_active=False,window=lambda active:observations.append(('window',active,lock.locked())))
    instance=object.__new__(handler(proxy));instance.path=route
    payload=b'{}' if route.endswith('direction-prompt') else b'x'
    instance.headers={'Content-Length':str(len(payload))};instance.rfile=io.BytesIO(payload);instance.wfile=io.BytesIO()
    instance.connection=SimpleNamespace(settimeout=lambda timeout:None)
    def send(status,body):observations.append(('response',status,lock.locked(),body))
    instance.send=send
    instance.send_response=lambda status:send(status,{})
    instance.send_header=lambda *args:None;instance.end_headers=lambda:None
    monkeypatch.setattr(writer_directions,'token_offsets',lambda *args:{'tokens':[1]})
    monkeypatch.setattr(combo_session,'render_body',lambda body:body)
    monkeypatch.setattr(combo_single.ds4_render,'render',lambda *args:('',[1],{}))
    def capture(*args,**kwargs):
        if failure:raise failure
        return b'data',{'Content-Type':'application/octet-stream'}
    monkeypatch.setattr(writer_directions,'capture',capture)
    return instance,lock,observations


@pytest.mark.parametrize('route',['/v1/train/direction-register','/v1/train/direction-stack','/v1/train/direction-tokens','/v1/train/direction-prompt','/v1/train/directions','/v1/train/direction-evaluate'])
def test_success_response_releases_control_ownership_before_next_request(route,monkeypatch):
    instance,lock,events=control(route,monkeypatch)
    instance.writer_control()
    responses=[row for row in events if row[0]=='response']
    assert len(responses)==1 and responses[0][1]==200
    assert responses[0][2] is False, 'Client can submit its next sequential request as soon as the response is readable'
    assert not lock.locked()


@pytest.mark.parametrize('error,status',[(ValueError('invalid capture'),400),(RuntimeError('capture failed'),503)])
def test_failed_capture_also_retires_control_before_reply(error,status,monkeypatch):
    instance,lock,events=control('/v1/train/directions',monkeypatch,failure=error)
    instance.writer_control()
    response=next(row for row in events if row[0]=='response')
    assert response[1:3]==(status,False)
    assert not lock.locked()


def test_foreign_request_is_still_refused_without_releasing_its_lock(monkeypatch):
    instance,lock,events=control('/v1/train/directions',monkeypatch)
    lock.acquire()
    try:
        instance.writer_control()
        assert events==[('response',409,True,{'error':{'message':'Another request owns this decoder'}})]
        assert lock.locked()
    finally:lock.release()


def test_window_publication_error_never_strands_the_decoder_lock(monkeypatch):
    instance,lock,events=control('/v1/train/direction-register',monkeypatch)
    # The closure is reachable from the bound handler method's factory closure.
    proxy=instance.writer_control.__func__.__closure__[0].cell_contents
    proxy.window=lambda active:(_ for _ in ()).throw(OSError('window unavailable'))
    with pytest.raises(OSError,match='window unavailable'):instance.writer_control()
    assert not lock.locked()


def test_loopback_reply_waits_for_window_publication(monkeypatch):
    from http.client import HTTPConnection
    from http.server import ThreadingHTTPServer
    instance,lock,_=control('/v1/train/direction-register',monkeypatch)
    proxy=instance.writer_control.__func__.__closure__[0].cell_contents
    proxy.token='a'*32
    entered=threading.Event();release=threading.Event();replied=threading.Event();result=[]
    def window(active):
        if not active:
            entered.set()
            if not release.wait(3):raise TimeoutError('Test publication gate timed out')
    proxy.window=window
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(proxy))
    serving=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True);serving.start()
    def request():
        connection=HTTPConnection('127.0.0.1',server.server_port,timeout=4)
        try:
            connection.request('POST','/v1/train/direction-register',body=b'x',headers={'Authorization':'Bearer '+proxy.token})
            response=connection.getresponse();result.append((response.status,response.read()));replied.set()
        finally:connection.close()
    client=threading.Thread(target=request,daemon=True);client.start()
    try:
        assert entered.wait(2)
        assert not replied.wait(.1), 'HTTP response became readable before ownership publication finished'
        assert lock.locked()
        release.set();client.join(3)
        assert replied.is_set() and result[0][0]==200 and not lock.locked()
    finally:
        release.set();client.join(4);server.shutdown();serving.join(2);server.server_close()
