# Derived from Studio backend/studio/dsv41_session_broker.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1.
# Minimal source export; transport and owned process logic unchanged.
from __future__ import annotations
import asyncio, threading, time
from dataclasses import dataclass, field
from starlette.requests import Request as HTTPRequest
from .dsv41_session_contract import MAX_CONTROL, Refused, command, control, initial_request, require

@dataclass
class Record:
    request: dict
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    producer: object = None
    state: str = 'starting'
    error: str = ''
    metadata: dict = field(default_factory=dict)
    operations: dict = field(default_factory=dict)
    busy: bool = False
    payload: dict | None = None
    claimed: bool = False
    cleanup: dict = field(default_factory=dict)

class Broker:

    def __init__(self, factory):
        require(callable(factory), 'Trusted session factory required')
        self.factory = factory
        self.lock = threading.Lock()
        self.records = {}
        self.active = None

    def start(self, request):
        sid = request['session']
        with self.lock:
            require(self.active is None, 'Another native session owns the producer')
            require(sid not in self.records, 'Session nonce already used')
            if len(self.records) >= 64:
                key = next(iter(self.records))
                require(self.records[key].done.is_set(), 'Session status capacity reached')
                self.records.pop(key)
            record = Record(request)
            self.records[sid] = self.active = record
            try:
                threading.Thread(target=self._work, args=(record,), name='dsv41-session-owner', daemon=True).start()
            except BaseException:
                self.records.pop(sid)
                self.active = None
                raise
        return dict(session=sid, state='starting')

    def _work(self, record):
        owned = None
        try:
            owned = self.factory(record.request, record.cancel)
            with self.lock:
                require(not record.cancel.is_set(), 'Session startup cancelled')
                record.producer = owned.producer
                record.metadata = dict(owned.producer.ready)
                record.state = 'ready'
                record.request = {k: v for k, v in record.request.items() if k not in ('tokens','vision')}
            while not record.cancel.wait(0.1):
                p = record.producer
                require(p.failure is None, p.failure or 'Producer failed')
                if p.job is None or p.job.poll() is not None:
                    break
            with self.lock:
                if record.state != 'failed':
                    record.state = 'cancelled' if record.cancel.is_set() else 'closed'
        except BaseException as exc:
            with self.lock:
                record.state = 'failed'
                record.error = str(exc)
        finally:
            try:
                if owned is not None:
                    record.cleanup = owned.close()
                with self.lock:
                    record.producer = None
                    record.done.set()
                    if self.active is record:
                        self.active = None
            except BaseException as exc:
                with self.lock:
                    record.state = 'failed'
                    record.error = 'Owned cleanup unconfirmed: ' + str(exc)

    def _get(self, sid):
        require(sid in self.records, 'Unknown native session')
        return self.records[sid]

    def status(self, sid):
        with self.lock:
            r = self._get(sid)
            return {**r.metadata, 'session': sid, 'state': r.state, 'error': r.error, 'cleanup': r.cleanup, 'cleanup_complete': r.done.is_set()}

    def cancel(self, sid):
        with self.lock:
            r = self._get(sid)
            r.cancel.set()
            p = r.producer
        if p:
            p._poison(Refused('Authenticated session cancellation'))
        return dict(session=sid, state='cancelling' if not r.done.is_set() else r.state)

    def _claim(self, sid, row):
        with self.lock:
            r = self._get(sid)
            require(r.state == 'ready' and r.producer is not None and (not r.cancel.is_set()), 'Native session not ready')
            require(not r.busy and r.payload is None, 'Another native command or payload is active')
            command(row, sid, r.producer.sequence)
            r.busy = True
            r.operations[row['sequence']] = {'state': 'running', 'sequence': row['sequence']}
            while len(r.operations) > 32:
                r.operations.pop(next(iter(r.operations)))
            return (r, r.producer)

    def _complete(self, r, row, reply):
        with self.lock:
            r.operations[row['sequence']] = {'state': 'complete', **reply}
            r.busy = False
            if reply.get('status') == 'payload':
                r.payload = reply
                r.claimed = False
            elif 'checkpoint' in reply:
                r.metadata['checkpoint'] = reply['checkpoint']
            r.metadata['next_sequence'] = r.producer.sequence if r.producer else row['sequence'] + 1

    def _failed(self, r, row, exc):
        with self.lock:
            r.operations[row['sequence']] = {'state': 'failed', 'sequence': row['sequence'], 'error': str(exc)}
            r.state = 'failed'
            r.error = str(exc)
            r.cancel.set()
            r.busy = False

    def enqueue(self, sid, row, *, vision=None):
        require(row.get('operation') != 'apply_decode_delta', 'Decode delta requires its streamed request body')
        require('vision_spans_file' not in row,'Network paths are never trusted vision descriptors')
        require(vision is None or row.get('operation')=='prefill','Projected vision requires prefill')
        r, p = self._claim(sid, row)

        def work():
            try:
                actual=dict(row)
                if vision is not None:actual['vision_spans_file']=p.stage_vision(vision,row['base']['position'])
                reply = p.begin_export(actual) if row['operation'] in ['export_initial', 'export_delta'] else p.request(actual)
                self._complete(r, row, reply)
            except BaseException as exc:
                p._poison(exc)
                self._failed(r, row, exc)
        try:
            threading.Thread(target=work, name='dsv41-session-command', daemon=True).start()
        except BaseException as exc:
            self._failed(r, row, exc)
            raise
        return dict(session=sid, sequence=row['sequence'], state='running')

    def operation(self, sid, sequence):
        with self.lock:
            r = self._get(sid)
            require(sequence in r.operations, 'Unknown or expired native operation')
            return dict(r.operations[sequence])

    def claim_payload(self, sid):
        with self.lock:
            r = self._get(sid)
            require(r.state == 'ready' and r.producer is not None and (r.payload is not None) and (not r.claimed), 'Native payload unavailable')
            r.claimed = True
            return (r, r.producer, dict(r.payload))

    def payload_done(self, r, meta):
        with self.lock:
            r.metadata['checkpoint'] = meta['target']
            r.metadata['next_sequence'] = r.producer.sequence if r.producer else meta['sequence'] + 1
            r.payload = None
            r.claimed = False

    def close(self, timeout=45):
        require(0 < timeout <= 60, 'Bounded broker shutdown required')
        with self.lock:
            pending = [(k, v) for k, v in self.records.items() if not v.done.is_set()]
        deadline = time.monotonic() + timeout
        for sid, r in pending:
            r.cancel.set()
        for sid, r in pending:
            self.cancel(sid)
            require(r.done.wait(max(0, deadline - time.monotonic())), 'Native session cleanup unconfirmed at broker shutdown')

async def joined_call(fn, *args):
    task = asyncio.create_task(asyncio.to_thread(fn, *args))
    interrupted = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            interrupted = True
        except BaseException:
            break
    result = task.result()
    if interrupted:
        raise asyncio.CancelledError
    return result

async def owned_call(producer, fn, *args):
    task = asyncio.create_task(asyncio.to_thread(fn, *args))
    try:
        return await asyncio.shield(task)
    except BaseException:
        try:
            await joined_call(producer._poison, Refused('Native HTTP operation interrupted'))
        except asyncio.CancelledError:
            pass
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        if task.done() and (not task.cancelled()):
            task.exception()
        raise

def create_router(broker, authorize):
    from fastapi import APIRouter, Depends, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
    require(callable(authorize), 'Explicit native pairing authorization required')
    router = APIRouter(prefix='/dsv41-native-offload/sessions', dependencies=[Depends(authorize)])

    def headers(request):
        names = [k.lower() for k, _ in request.scope['headers']]
        require(len(names) == len(set(names)), 'Duplicate native HTTP header')
        return dict(request.headers)

    async def body(request, cap):
        data = bytearray()
        async for block in request.stream():
            require(len(data) + len(block) <= cap, 'Native request body exceeds limit')
            data.extend(block)
        return bytes(data)

    def rejected(exc):
        return HTTPException(409, str(exc))

    @router.post('/{sid}')
    async def start(sid: str, request: HTTPRequest):
        try:
            from slstack.paired_vision import MAX_BODY
            return JSONResponse(broker.start(initial_request(sid, headers(request), await body(request, MAX_BODY))), status_code=202)
        except Refused as exc:
            raise rejected(exc) from None

    @router.get('/{sid}')
    async def status(sid: str):
        try:
            return broker.status(sid)
        except Refused as exc:
            raise rejected(exc) from None

    @router.delete('/{sid}')
    async def cancel(sid: str):
        try:
            return await asyncio.to_thread(broker.cancel, sid)
        except Refused as exc:
            raise rejected(exc) from None

    @router.post('/{sid}/commands')
    async def enqueue(sid: str, request: HTTPRequest):
        try:
            headers(request)
            return JSONResponse(broker.enqueue(sid, control(await body(request, MAX_CONTROL))), status_code=202)
        except Refused as exc:
            raise rejected(exc) from None

    @router.get('/{sid}/commands/{sequence}')
    async def operation(sid: str, sequence: int):
        try:
            return broker.operation(sid, sequence)
        except Refused as exc:
            raise rejected(exc) from None

    @router.post('/{sid}/vision')
    async def vision_prefill(sid: str, request: HTTPRequest):
        try:
            import hashlib,struct
            from slstack.paired_vision import MEDIA,MAX_BODY,unpack
            h=headers(request);wire=h.get('x-dsv41-control','').encode('ascii')
            require(len(wire)<=16384,'Vision control header too large')
            row=control(wire)
            require(row.get('operation')=='prefill','Expected projected prefill')
            data=await body(request,MAX_BODY)
            require(h.get('content-type')==MEDIA and h.get('content-length')==str(len(data)) and h.get('x-dsv41-body-sha256')==hashlib.sha256(data).hexdigest(),'Projected body SHA/extent mismatch')
            tokens,images=unpack(data,base=row['base']['position'])
            row['tokens']=list(struct.unpack('<%dI'%(len(tokens)//4),tokens))
            return JSONResponse(broker.enqueue(sid,row,vision=images),status_code=202)
        except (ValueError,TypeError,KeyError,UnicodeError,struct.error) as exc:
            raise rejected(exc) from None

    class PayloadResponse(StreamingResponse):

        def __init__(self, r, p, meta):
            self.record, self.producer, self.meta = (r, p, meta)
            self.complete = False
            super().__init__(self.blocks(), media_type='application/octet-stream', headers={'Content-Length': str(meta['bytes']), 'Cache-Control': 'no-store'})

        async def blocks(self):
            try:
                while self.producer.remaining:
                    block = await owned_call(self.producer, self.producer.read_export)
                    if not self.producer.remaining:
                        await owned_call(self.producer, self.producer.end_export)
                        broker.payload_done(self.record, self.meta)
                        self.complete = True
                    yield block
            finally:
                if not self.complete:
                    self.record.cancel.set()
                    await joined_call(self.producer.abort_transfer)

        async def __call__(self, scope, receive, send):
            delivered = False
            try:
                async with asyncio.timeout(self.producer.operation_seconds):
                    await super().__call__(scope, receive, send)
                delivered = True
            finally:
                await self.body_iterator.aclose()
                if not delivered or not self.complete:
                    self.record.cancel.set()
                    await joined_call(self.producer.abort_transfer)

    @router.get('/{sid}/payload')
    async def payload(sid: str):
        try:
            r, p, meta = broker.claim_payload(sid)
            return PayloadResponse(r, p, meta)
        except Refused as exc:
            raise rejected(exc) from None

    @router.post('/{sid}/delta')
    async def delta(sid: str, request: HTTPRequest):
        r = p = row = None
        try:
            h = headers(request)
            wire = h.get('x-dsv41-control', '').encode('ascii')
            require(len(wire) <= 16384, 'Delta control header too large')
            row = control(wire)
            require(row.get('operation') == 'apply_decode_delta', 'Expected decode delta')
            require(h.get('content-type') == 'application/octet-stream' and h.get('content-length') == str(row.get('bytes')), 'Delta HTTP extent/media mismatch')
            r, p = broker._claim(sid, row)
            async with asyncio.timeout(p.operation_seconds):
                await owned_call(p, p.begin_decode, row)
                async for block in request.stream():
                    for off in range(0, len(block), 65536):
                        await owned_call(p, p.write_decode, block[off:off + 65536])
                reply = await owned_call(p, p.end_decode)
                broker._complete(r, row, reply)
                return reply
        except BaseException as exc:
            if p:
                try:
                    await joined_call(p.abort_transfer)
                except asyncio.CancelledError:
                    pass
            if r:
                broker._failed(r, row, exc)
            if isinstance(exc, (Refused, UnicodeError)):
                raise rejected(exc) from None
            raise
    return router
