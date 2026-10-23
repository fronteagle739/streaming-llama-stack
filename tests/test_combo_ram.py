import ctypes
import hashlib
import os
import subprocess
import sys
import time
import json
from pathlib import Path
import uuid
import unittest
from slstack.combo_ram import LockedRam, kernel_api, protect_working_set,wait_for_host_capacity


class CapacityTest(unittest.TestCase):
    def test_waits_for_exact_physical_capacity_and_reports_elapsed(self):
        now=[0.];readings=iter((2,5))
        def sleep(seconds):now[0]+=seconds
        result=wait_for_host_capacity(3,2,lambda:False,available=lambda:next(readings),clock=lambda:now[0],sleep=sleep)
        self.assertEqual(result['required_bytes'],5);self.assertEqual(result['available_bytes'],5)
        self.assertAlmostEqual(result['seconds'],.1)

    def test_timeout_retains_required_and_available_without_launch(self):
        now=[0.]
        def sleep(seconds):now[0]+=seconds
        with self.assertRaisesRegex(MemoryError,'available=2, required=5') as caught:
            wait_for_host_capacity(3,2,lambda:False,timeout_s=.2,available=lambda:2,clock=lambda:now[0],sleep=sleep)
        self.assertAlmostEqual(caught.exception.admission_receipt['seconds'],.2)

    def test_cancel_does_not_wait_or_admit(self):
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            wait_for_host_capacity(3,2,lambda:True,available=lambda:5)


@unittest.skipUnless(sys.platform=='win32','Windows shared RAM contract')
class LockedRamTest(unittest.TestCase):
    def test_owned_child_has_resident_admission_before_token_bootstrap(self):
        from slstack.native_broker import vendor
        vendor()
        from studio.dsv41_windows_runner import _Native
        native=_Native();fds=[];job=None
        try:
            source,send=os.pipe();output,sink=os.pipe();errors,diagnostic=os.pipe()
            fds=[source,send,output,sink,errors,diagnostic]
            job=native.suspended([sys.executable,'-B','-c','import sys;sys.stdin.buffer.read(1);print("CPU_RESIDENT_PASS")'],
                dict(os.environ),os.environ['STUDIO_RUNTIME_ROOT'],source,sink,diagnostic)
            for fd in (source,sink,diagnostic):os.close(fd);fds.remove(fd)
            job.resume()
            receipt=Path(os.environ['STUDIO_RUNTIME_ROOT'])/('resident-cpu-'+uuid.uuid4().hex+'.json')
            admission=protect_working_set(job,128<<20,4<<30,512<<20,receipt=receipt)
            self.assertEqual(json.loads(receipt.read_text()),admission)
            self.assertEqual(admission['pid'],job.pid)
            self.assertEqual(admission['minimum'],128<<20)
            self.assertEqual(admission['flags']&5,1)
            self.assertIsNone(job.poll())
            os.write(send,b'G');deadline=time.monotonic()+10
            while job.poll() is None and time.monotonic()<deadline:time.sleep(.01)
            self.assertEqual(job.poll(),0)
            self.assertIn(b'CPU_RESIDENT_PASS',os.read(output,4096))
        finally:
            if job is not None:job.stop();job.close()
            for fd in fds:os.close(fd)

    def test_resident_admission_refuses_unbudgeted_controller_before_any_handle(self):
        class NoHandle:
            @property
            def pi(self):raise AssertionError('Refused aggregate must not touch a process')
        with self.assertRaisesRegex(MemoryError,'aggregate resident'):
            protect_working_set(NoHandle(),128<<20,4<<30,128<<20)

    def test_locked_shared_pages_and_bounded_transfer(self):
        with LockedRam(8<<20,session='1'*32,host_budget_gib=.5,min_available_gib=4) as ram:
            ram.write(0,b'checkpoint-owned-RAM')
            k=kernel_api();handle=k.OpenFileMappingW(6,False,ram.name)
            self.assertTrue(handle);address=k.MapViewOfFile(handle,6,0,0,ram.capacity)
            self.assertTrue(address)
            try:
                self.assertEqual(ctypes.string_at(address,20),b'checkpoint-owned-RAM')
                ctypes.memmove(address+20,b'!',1)
                self.assertEqual(ram.read(0,21),b'checkpoint-owned-RAM!')
            finally:k.UnmapViewOfFile(address);k.CloseHandle(handle)
            self.assertTrue(ram.locked)
            self.assertEqual(ram.sha256(0,21),hashlib.sha256(b'checkpoint-owned-RAM!').hexdigest())
            for offset,count in ((0,65537),(ram.capacity,1),(-1,4)):
                with self.assertRaises(ValueError):ram.read(offset,count)
            with ram.lease(),self.assertRaises(RuntimeError):ram.close()
        self.assertIsNone(ram.address)
        self.assertIsNone(ram.handle)
        with self.assertRaises(RuntimeError):ram.read(0,1)

    def test_budget_and_floor_refuse_before_allocation(self):
        with self.assertRaises(MemoryError):LockedRam(1<<20,session='1'*32,host_budget_gib=.0001,min_available_gib=4)
        with self.assertRaises(MemoryError):LockedRam(1<<20,session='1'*32,host_budget_gib=.5,min_available_gib=64)

    def test_cross_process_reads_the_same_section(self):
        with LockedRam(1<<20,session='2'*32,host_budget_gib=.5,min_available_gib=4) as ram:
            ram.write(123,b'native-to-ultra')
            program='from slstack.combo_ram import kernel_api;import ctypes,sys;k=kernel_api();h=k.OpenFileMappingW(4,False,sys.argv[1]);assert h;p=k.MapViewOfFile(h,4,0,0,1048576);assert p;assert ctypes.string_at(p+123,15)==b"native-to-ultra";assert k.UnmapViewOfFile(p);assert k.CloseHandle(h);print("SHARED_RAM_CHILD_PASS")'
            result=subprocess.run([sys.executable,'-B','-c',program,ram.name],capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('SHARED_RAM_CHILD_PASS',result.stdout)


if __name__=='__main__':unittest.main()
