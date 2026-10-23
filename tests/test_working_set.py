"""Real owned CPU child: admission, quota retention, and failed-hook cleanup."""
import ctypes
import json
import os
from pathlib import Path
import sys
import time
import unittest
import uuid
from slstack.working_set import protect_working_set, kernel_api
from slstack.native_broker import vendor
vendor()
from studio.dsv41_windows_runner import _Native
from studio.dsv41_session_process import Producer


@unittest.skipUnless(os.name == 'nt', 'Windows working set contract')
class WorkingSetTests(unittest.TestCase):
    # Reused CPU admission/pipe fixture from the protection provider.
    def test_owned_child_admission_and_legacy_allocator_preserve_hard_minimum(self):
        native=_Native();fds=[];job=None
        try:
            source,send=os.pipe();output,sink=os.pipe();errors,diagnostic=os.pipe()
            fds=[source,send,output,sink,errors,diagnostic]
            job=native.suspended([sys.executable,'-B','-c',
                'import sys;sys.stdin.buffer.read(1);print("CPU_RESIDENT_PASS")'],
                dict(os.environ),os.environ['STUDIO_RUNTIME_ROOT'],source,sink,diagnostic)
            for fd in (source,sink,diagnostic):os.close(fd);fds.remove(fd)
            job.resume()
            receipt=Path(os.environ['STUDIO_RUNTIME_ROOT'])/('resident-cpu-'+uuid.uuid4().hex+'.json')
            admission=protect_working_set(job,128<<20,4<<30,512<<20,receipt=receipt)
            self.assertEqual(json.loads(receipt.read_text()),admission)
            self.assertEqual(admission['pid'],job.pid)
            self.assertEqual(admission['minimum'],128<<20)
            self.assertEqual(admission['flags']&5,1)
            # The native locked allocator uses this older API with the larger
            # current minimum. Verify the OS preserves the hard-minimum flag.
            k=kernel_api()
            k.SetProcessWorkingSetSize.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t]
            k.SetProcessWorkingSetSize.restype=ctypes.c_int
            self.assertTrue(k.SetProcessWorkingSetSize(job.pi.process,admission['minimum'],admission['maximum']))
            low,high,flags=ctypes.c_size_t(),ctypes.c_size_t(),ctypes.c_ulong()
            self.assertTrue(k.GetProcessWorkingSetSizeEx(job.pi.process,ctypes.byref(low),ctypes.byref(high),ctypes.byref(flags)))
            self.assertEqual(low.value,128<<20)
            self.assertEqual(flags.value&5,1)
            os.write(send,b'G');deadline=time.monotonic()+10
            while job.poll() is None and time.monotonic()<deadline:time.sleep(.01)
            self.assertEqual(job.poll(),0)
            self.assertIn(b'CPU_RESIDENT_PASS',os.read(output,4096))
        finally:
            if job is not None:job.stop();job.close()
            for fd in fds:os.close(fd)

    def test_refuses_unbudgeted_controller_before_handle_access(self):
        class NoHandle:
            @property
            def pi(self):raise AssertionError('Refused aggregate must not touch a process')
        with self.assertRaisesRegex(MemoryError,'aggregate resident'):
            protect_working_set(NoHandle(),128<<20,4<<30,128<<20)

    def test_post_resume_failure_certifies_native_child_cleanup(self):
        def refusal(job):raise MemoryError('TEST resident admission refused')
        with self.assertRaisesRegex(MemoryError,'resident admission refused') as caught:
            Producer(uuid.uuid4().hex,[sys.executable,'-B','-c','import time;time.sleep(30)'],
                dict(os.environ),Path(os.environ['STUDIO_RUNTIME_ROOT']),host_bytes=256<<20,
                physical_floor=4<<30,operation_seconds=10,idle_seconds=10,after_resume=refusal)
        proof=caught.exception.owned_cleanup
        self.assertTrue(all(proof.get(k) for k in ('owned_job_empty','owned_process_signaled','owned_pid_gone')))


if __name__=='__main__':unittest.main()
