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
    def stop(self,**_):self.running=False
    def poll(self):return None if self.running else 0
    def active(self):return 1 if self.running else 0
    def close(self):self.running=False


class FakeNative:
    def suspended(self,*args):self.job=FakeJob();return self.job


class ProcessTest(unittest.TestCase):
    def test_broken_bootstrap_retains_diagnostic_and_exact_cleanup_owner(self):
        import uuid
        from slstack.combo_producer import startup_failure
        directory=Path(os.environ['STUDIO_RUNTIME_ROOT'])/('producer-startup-'+uuid.uuid4().hex)
        directory.mkdir()
        original=BrokenPipeError(32,'Broken pipe')
        original.owned_cleanup=dict(owned_job_empty=True,owned_process_signaled=True,owned_pid_gone=True)
        original.owned_runtime=object()
        original.owned_diagnostic=b'DS4PERF_COMBINED_ADMISSION planned_host=6000 admitted_host=5000\ncombined job exceeds total resource admission\n'
        reported=startup_failure(original,directory)
        self.assertIsInstance(reported,RuntimeError)
        self.assertIn('planned_host=6000',str(reported))
        self.assertIn('combined job exceeds',str(reported))
        self.assertIs(reported.owned_cleanup,original.owned_cleanup)
        self.assertIs(reported.owned_runtime,original.owned_runtime)
        self.assertEqual((directory/'stderr.log').read_bytes(),original.owned_diagnostic)

    def test_pre_readiness_admission_refusal_retires_job_without_startup(self):
        native=FakeNative();events=[]
        original=native.suspended
        def create(*args):
            job=original(*args)
            job.resume=lambda:events.append('resumed')
            return job
        def refuse(job):events.append('admission');raise MemoryError('CPU fixture resident admission refused')
        with patch.object(native,'suspended',side_effect=create),patch('studio.dsv41_session_process.psutil.pid_exists',return_value=False):
            with self.assertRaisesRegex(MemoryError,'resident admission refused') as caught:
                Producer('1'*32,['unused'],dict(os.environ),Path('.'),host_bytes=512<<20,
                    physical_floor=4<<30,native=native,native_mode=1,after_resume=refuse,
                    startup=lambda owned:events.append('startup'))
        self.assertEqual(events,['resumed','admission'])
        self.assertFalse(native.job.running)
        self.assertTrue(all(caught.exception.owned_cleanup[k] for k in ('owned_job_empty','owned_process_signaled','owned_pid_gone')))

    def test_failed_constructor_cleanup_preserves_original_error_and_owner(self):
        native=FakeNative()
        original=native.suspended
        def start(*args):
            job=original(*args)
            def cannot_stop(**_):raise OSError('CPU fixture stop failed')
            job.stop=cannot_stop
            return job
        def fail(owned):raise ValueError('original setup refusal')
        with patch.object(native,'suspended',side_effect=start),patch('studio.dsv41_session_process.psutil.pid_exists',return_value=False):
            with self.assertRaisesRegex(ValueError,'original setup refusal') as caught:
                Producer('1'*32,['unused'],dict(os.environ),Path('.'),host_bytes=512<<20,
                    physical_floor=4<<30,native=native,native_mode=1,startup=fail)
            failure=caught.exception
            self.assertFalse(failure.owned_cleanup['owned_job_empty'])
            self.assertIn('CPU fixture stop failed',failure.owned_cleanup_error)
            owner=failure.owned_process
            native.job.stop=lambda **_:setattr(native.job,'running',False)
            self.assertTrue(owner.close()['owned_job_empty'])

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
            self.assertEqual(process.runtime_min_free_gib,0.5)
            self.assertEqual(process.runtime_floor,512<<20)
            proof=process.close()
        self.assertTrue(all(proof[k] for k in ('owned_job_empty','owned_process_signaled','owned_pid_gone')))
        self.assertFalse(native.job.running)

    def test_runtime_floor_is_configurable_and_invalid_values_refuse_before_creation(self):
        for floor in (0,6.5):
            native=FakeNative()
            with self.subTest(floor=floor),patch('studio.dsv41_session_process.psutil.pid_exists',return_value=False):
                process=Producer('1'*32,['unused'],dict(os.environ),Path('.'),host_bytes=512<<20,
                    physical_floor=1<<30,runtime_min_free_gib=floor,native=native,native_mode=1,startup=lambda owned:dict(status='ready'))
                self.assertEqual(process.runtime_floor,int(floor*(1<<30)))
                self.assertEqual(process.close()['runtime_min_free_gib'],floor)
        for floor in (-1,float('nan'),float('inf'),True,'4',None,2**34):
            native=FakeNative()
            with self.subTest(floor=floor),self.assertRaisesRegex(ValueError,'runtime_min_free_gib') as caught:
                Producer('1'*32,['unused'],{},Path('.'),host_bytes=512<<20,physical_floor=1<<30,native=native,runtime_min_free_gib=floor)
            self.assertFalse(hasattr(native,'job'))
            self.assertFalse(caught.exception.owned_cleanup['creation_attempted'])

    def test_runtime_floor_reaches_producer_argv_and_session_watcher(self):
        from slstack.combo_producer import ComboProducer
        seen=[]
        class Files:
            @contextmanager
            def directory(self,path):yield
            @contextmanager
            def read_lease(self,path):yield
            def gpu(self,*args):return 1<<40
        class Process:
            def __init__(self,session,argv,env,directory,**kwargs):
                seen.append((argv,kwargs))
                self.ready=dict(checkpoint=dict(context=262144,position=1))
        cfg=dict(runtime_root=os.environ['STUDIO_RUNTIME_ROOT'],llama_producer=sys.executable,model=sys.executable,
            producer_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),producer_source_revision='r',
            context=262144,gpu_uuid='TEST',min_available_gib=1,host_budget_gib=8,request_timeout_s=10,lease_minutes=1)
        for floor in (None,0,6.5):
            chosen=cfg if floor is None else dict(cfg,runtime_min_free_gib=floor)
            with patch('studio.dsv41_session_process.Producer',Process),patch('slstack.combo_producer.environment',return_value={}),patch('slstack.combo_producer.model_receipt',return_value=Path(sys.executable)),patch('slstack.combo_producer.model_digest',return_value=MODEL_SHA256):
                p=ComboProducer(chosen,'1'*32,42,[1],lambda:False,runner=Files())
            p.leases.close()
            argv,kwargs=seen[-1];expected=0.5 if floor is None else floor
            self.assertEqual(float(argv[argv.index('--runtime-min-free-gib')+1]),expected)
            self.assertEqual(kwargs['runtime_min_free_gib'],expected)

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
