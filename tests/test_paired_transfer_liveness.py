"""Exercise the pinned HTTP payload route with slow, progressing CPU bytes."""
import threading
import time
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from slstack.native_broker import ProgressGuardedTransport, vendor

vendor()
from studio.dsv41_session_broker import create_router


def test_slow_payload_outlives_the_old_total_transfer_deadline():
    class Export:
        operation_seconds = .02
        remaining = 5
        poisoned = False

        def read_export(self):
            time.sleep(.03)
            self.remaining -= 1
            return b'x'

        def end_export(self):
            pass

        def abort_transfer(self):
            self.poisoned = True

        def _poison(self, reason):
            self.poisoned = True

    owner = Export()
    transport = ProgressGuardedTransport(owner)
    record = SimpleNamespace(cancel=threading.Event())
    completed = []
    broker = SimpleNamespace(claim_payload=lambda sid: (record, transport, {'bytes': 5}),
                             payload_done=lambda *args: completed.append(True))
    app = FastAPI()
    app.include_router(create_router(broker, lambda: True))
    started = time.monotonic()
    with TestClient(app) as client:
        response = client.get('/dsv41-native-offload/sessions/test/payload')
    assert response.status_code == 200 and response.content == b'xxxxx'
    assert time.monotonic() - started > owner.operation_seconds
    assert completed and not owner.poisoned and not record.cancel.is_set()
