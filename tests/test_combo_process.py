import os
import io
import json
from pathlib import Path
import unittest
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
