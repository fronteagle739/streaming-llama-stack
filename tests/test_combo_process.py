import os
import io
import json
from pathlib import Path
import unittest
from contextlib import contextmanager
import hashlib
import sys
from unittest.mock import patch
from slstack.native_broker import vendor
vendor()
from studio.dsv41_session_process import Producer
from studio.dsv41_session_contract import MODEL_SHA256


class FakeJob:
    pid=os.getpid()
    def __init__(self):self.running=True
    def resume(self):pass
    def stop(self):self.running=False
    def poll(self):return None if self.running else 0
    def active(self):return 1 if self.running else 0
    def close(self):self.running=False


class FakeNative:
    def suspended(self,*args):self.job=FakeJob();return self.job


class ProcessTest(unittest.TestCase):
    def test_native_receipt_write_failure_retires_started_owner_and_file_leases(self):
        from slstack.combo_native import Native
        events=[];created=[]
        class Files:
            @contextmanager
            def directory(self,path):
                events.append('directory pinned')
                try:yield
                finally:events.append('directory released')
            @contextmanager
            def read_lease(self,path):
                events.append('file pinned')
                try:yield
                finally:events.append('file released')
        class Process:
            def __init__(self,*args,startup,**kwargs):
                self.job=FakeJob();self.diagnostic=b'';created.append(self)
                self.ready=startup(self)
            def close(self):
                self.job.stop();events.append('owner retired')
                return dict(owned_job_empty=True,owned_process_signaled=True,owned_pid_gone=True)
        cfg=dict(runtime_root=os.environ['STUDIO_RUNTIME_ROOT'],ds4_command=[sys.executable],
            native_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),model=sys.executable,
            ds4_url='http://127.0.0.1:1',host_budget_gib=.5,min_available_gib=4,
            request_timeout_s=10,lease_minutes=1,context=262144)
        def reply(self,suffix,*args,**kwargs):
            if suffix=='/v1/models':return {}
            return dict(status='complete',session='1'*32,owner_epoch=42,next_sequence=2,checkpoint=None)
        with patch('studio.dsv41_session_process.Producer',Process),patch('slstack.combo_native.environment',return_value={}),patch('slstack.combo_native.model_receipt',return_value=Path(sys.executable)),patch.object(Native,'http',reply),patch('slstack.combo_native.atomic_json',side_effect=OSError('receipt disk full')):
            with self.assertRaisesRegex(OSError,'receipt disk full'):
                Native(cfg,'1'*32,42,1,lambda:False,runner=Files())
        self.assertFalse(created[0].job.running)
        self.assertEqual(events.count('file pinned'),events.count('file released'))
        self.assertLess(events.index('owner retired'),events.index('directory released'))

    def test_native_startup_uses_existing_budgeted_job_and_certified_close(self):
        native=FakeNative()
        with patch('studio.dsv41_session_process.psutil.pid_exists',return_value=False):
            process=Producer('1'*32,['unused'],dict(os.environ),Path('.'),host_bytes=512<<20,
                physical_floor=4<<30,native=native,native_mode=1,startup=lambda owned:dict(status='ready'))
            self.assertTrue(process.creation_attempted)
            proof=process.close()
        self.assertTrue(all(proof[k] for k in ('owned_job_empty','owned_process_signaled','owned_pid_gone')))
        self.assertFalse(native.job.running)

    def test_startup_failure_stops_the_same_owned_job(self):
        native=FakeNative()
        def fail(owned):raise ValueError('setup refused')
        with patch('studio.dsv41_session_process.psutil.pid_exists',return_value=False):
            with self.assertRaisesRegex(ValueError,'setup refused') as caught:
                Producer('1'*32,['unused'],dict(os.environ),Path('.'),host_bytes=512<<20,
                    physical_floor=4<<30,native=native,native_mode=1,startup=fail)
        self.assertFalse(native.job.running)
        self.assertTrue(caught.exception.owned_cleanup['owned_job_empty'])

    def test_epoch_is_retained_on_commands(self):
        native=FakeNative()
        with patch('studio.dsv41_session_process.psutil.pid_exists',return_value=False):
            process=Producer('1'*32,['unused'],dict(os.environ),Path('.'),host_bytes=512<<20,
                physical_floor=4<<30,native=native,native_mode=1,owner_epoch=42,
                startup=lambda owned:dict(status='ready',owner_epoch=42))
            output=io.BytesIO();process.input=output
            row=dict(session='1'*32,model_sha256=MODEL_SHA256,sequence=1,operation='checkpoint')
            process._enter(row)
            try:self.assertEqual(json.loads(output.getvalue())['owner_epoch'],42)
            finally:process._leave();process.close();output.close()
            self.assertNotIn('owner_epoch',row)

    def test_wrong_or_noninteger_epoch_refuses_readiness_and_retires_job(self):
        for epoch in (43,True,42.0,None):
            native=FakeNative()
            with self.subTest(epoch=epoch),patch('studio.dsv41_session_process.psutil.pid_exists',return_value=False):
                with self.assertRaises(ValueError) as caught:
                    Producer('1'*32,['unused'],dict(os.environ),Path('.'),host_bytes=512<<20,
                        physical_floor=4<<30,native=native,native_mode=1,owner_epoch=42,
                        startup=lambda owned:dict(status='ready',owner_epoch=epoch))
            self.assertFalse(native.job.running)
            self.assertTrue(caught.exception.owned_cleanup['owned_job_empty'])


if __name__=='__main__':unittest.main()
