"""CPU fixtures for shutdown order, renewable grace and the final force fallback."""
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from slstack.shutdown import ProgressOwnedProcesses, RetirementProgress, wait_for_exit, wait_for_thread, request_interrupt, gpu_free_reader
from slstack.process import OwnedProcess
from slstack.native_broker import vendor

vendor()
from studio.dsv41_windows_runner import _Job, _Process


class RetirementTests(unittest.TestCase):
    def test_gpu_reader_is_lazy_and_throttled_and_uses_only_the_configured_device(self):
        identity='GPU-00000000-0000-0000-0000-000000000001';now=[0.]
        with patch('slstack.shutdown.subprocess.run',return_value=SimpleNamespace(stdout='100')) as run, \
             patch('slstack.shutdown.time.monotonic',side_effect=lambda:now[0]):
            read=gpu_free_reader(identity);run.assert_not_called()
            self.assertEqual(read(),100<<20);self.assertEqual(read(),100<<20)
            self.assertEqual(run.call_count,1);self.assertIn('--id='+identity,run.call_args.args[0])
            now[0]=2;run.return_value.stdout='101'
            self.assertEqual(read(),101<<20);self.assertEqual(run.call_count,2)

    def test_each_retirement_signal_extends_the_full_grace(self):
        for kind in ('log','rss','gpu','members'):
            with self.subTest(kind=kind):
                now=[0.];size=[0];free=[0]
                member=Mock();member.memory_info.return_value=SimpleNamespace(rss=1000)
                progress=RetirementProgress(60,clock=lambda:now[0],diagnostic=lambda:size[0],gpu_free=lambda:free[0])
                progress.observe([member,member])
                now[0]=50
                if kind=='log':size[0]+=1
                if kind=='rss':member.memory_info.return_value=SimpleNamespace(rss=999)
                if kind=='gpu':free[0]+=1
                members=[member] if kind=='members' else [member,member]
                progress.observe(members)
                now[0]=109;progress.observe(members)
                self.assertFalse(progress.window.expired())
                now[0]=110;progress.observe(members)
                self.assertTrue(progress.window.expired())

    def test_growing_rss_and_cpu_activity_do_not_renew_retirement(self):
        now=[0.];member=Mock()
        member.memory_info.return_value=SimpleNamespace(rss=1000)
        progress=RetirementProgress(60,clock=lambda:now[0])
        progress.observe([member])
        now[0]=60;member.memory_info.return_value=SimpleNamespace(rss=2000)
        member.cpu_times.return_value=SimpleNamespace(user=1000,system=1000)
        progress.observe([member])
        self.assertTrue(progress.window.expired())
        member.cpu_times.assert_not_called()

    def test_wait_survives_many_windows_while_diagnostics_advance(self):
        now=[0.]
        process=SimpleNamespace(poll=lambda:130 if now[0]>=300 else None)
        with patch('slstack.shutdown.process_members',return_value=[]):
            code=wait_for_exit(process,60,diagnostic=lambda:int(now[0]/40),clock=lambda:now[0],
                sleep=lambda seconds:now.__setitem__(0,now[0]+10))
        self.assertEqual(code,130);self.assertEqual(now[0],300)

    def test_wait_expires_a_whole_grace_after_last_progress(self):
        now=[0.]
        process=SimpleNamespace(poll=lambda:None)
        with patch('slstack.shutdown.process_members',return_value=[]),self.assertRaises(subprocess.TimeoutExpired):
            wait_for_exit(process,60,diagnostic=lambda:min(int(now[0]/40),3),clock=lambda:now[0],
                sleep=lambda seconds:now.__setitem__(0,now[0]+10))
        self.assertEqual(now[0],180)

    def test_launcher_requests_stop_before_waiting_for_its_guardian(self):
        with tempfile.TemporaryDirectory() as directory:
            owned=object.__new__(OwnedProcess)
            owned.stop_path=Path(directory)/'stop';owned.log=Path(directory)/'log'
            owned.proc=Mock();owned.guard_log=Mock()
            def wait(*args,**kwargs):self.assertTrue(owned.stop_path.exists());return 0
            with patch('slstack.process.wait_for_exit',side_effect=wait):owned.stop()
            owned.proc.kill.assert_not_called();owned.guard_log.close.assert_called_once()

    def test_retirement_worker_can_outlive_the_old_join_timeout(self):
        now=[0.];thread=Mock()
        thread.is_alive.side_effect=lambda:now[0]<300
        thread.join.side_effect=lambda seconds:now.__setitem__(0,now[0]+10)
        with patch('slstack.shutdown.time.monotonic',side_effect=lambda:now[0]):
            wait_for_thread(thread,None,60,diagnostic=lambda:int(now[0]/40))
        self.assertEqual(now[0],300)

    def test_posix_request_uses_the_python_cleanup_signal(self):
        import signal
        process=Mock()
        with patch('slstack.shutdown.os.name','posix'):request_interrupt(process)
        process.send_signal.assert_called_once_with(signal.SIGINT)
        process.terminate.assert_not_called();process.kill.assert_not_called()

    def test_children_get_a_request_if_their_controller_has_exited(self):
        parent=Mock(pid=1);child=Mock(pid=2)
        for member in (parent,child):member.memory_info.return_value=SimpleNamespace(rss=100)
        remaining=[parent,child];events=[]
        owned=object.__new__(ProgressOwnedProcesses)
        owned.spec={};owned.identities={};owned.retired_identities=[]
        owned.refresh=lambda:list(remaining);owned.live=owned.refresh
        def request(members,*args):
            events.append(members[0].pid);remaining.remove(members[0])
        owned.request=request
        with patch('slstack.shutdown.time.sleep'):
            proof=owned.stop()
        self.assertEqual(events,[1,2]);self.assertTrue(proof['confirmed'])
        parent.kill.assert_not_called();child.kill.assert_not_called()

    def test_owned_stop_waits_past_old_deadline_while_rss_falls(self):
        now=[0.];events=[];member=Mock(pid=4242)
        member.memory_info.side_effect=lambda:SimpleNamespace(rss=10000-int(min(now[0],200)))
        owned=object.__new__(ProgressOwnedProcesses)
        owned.spec={};owned.identities={};owned.retired_identities=[]
        owned.request=Mock(side_effect=lambda *args:events.append(('request',now[0])))
        owned.refresh=Mock(side_effect=lambda:[member] if now[0]<250 else [])
        owned.live=owned.refresh
        with patch('slstack.shutdown.time.monotonic',side_effect=lambda:now[0]), \
             patch('slstack.shutdown.time.sleep',side_effect=lambda seconds:now.__setitem__(0,now[0]+10)):
            proof=owned.stop()
        self.assertTrue(proof['confirmed']);self.assertFalse(proof['forced_kill'])
        self.assertEqual(events,[('request',0)]);member.kill.assert_not_called()
        self.assertEqual(now[0],250)


class NativeFallbackTests(unittest.TestCase):
    def job(self):
        job=object.__new__(_Job);job.pi=_Process();job.pi.pid=4242
        job.job=1;job.resumed=True;job.n=SimpleNamespace(k=Mock(),ok=lambda value:value)
        job.active=Mock(return_value=1);job.poll=Mock(return_value=130)
        return job

    def test_native_cancellation_returns_130_without_force(self):
        job=self.job();events=[]
        job.interrupt=Mock(side_effect=lambda:events.append('cooperative'))
        job._wait_empty=Mock(side_effect=lambda seconds:events.append(('wait',seconds)) or True)
        self.assertEqual(job.stop(30,120),'graceful')
        self.assertEqual(events,['cooperative',('wait',30)])
        job.n.k.TerminateJobObject.assert_not_called()

    def test_native_force_is_after_both_requests_and_both_grace_windows(self):
        job=self.job();events=[];results=iter((False,False,True))
        job.interrupt=Mock(side_effect=lambda:events.append('cooperative'))
        job._wait_empty=Mock(side_effect=lambda seconds:events.append(('wait',seconds)) or next(results))
        job.n.k.TerminateJobObject.side_effect=lambda *args:events.append(('force',args[1])) or True
        with self.assertLogs('studio.dsv41_windows_runner',level='CRITICAL'):
            result=job.stop(30,120)
        self.assertTrue(result.startswith('forced-kill:'))
        self.assertEqual(events,['cooperative',('wait',30),'cooperative',('wait',90),('force',137),('wait',30)])
        self.assertTrue(job.forced_kill)

    def test_native_wait_tracks_reclamation_beyond_its_original_window(self):
        job=self.job();now=[0.];member=Mock()
        member.memory_info.side_effect=lambda:SimpleNamespace(rss=10000-int(now[0]))
        job.active.side_effect=lambda:now[0]<300
        with patch('slstack.shutdown.process_members',return_value=[member]), \
             patch('studio.dsv41_windows_runner.time.monotonic',side_effect=lambda:now[0]), \
             patch('studio.dsv41_windows_runner.time.sleep',side_effect=lambda seconds:now.__setitem__(0,now[0]+10)):
            self.assertTrue(job._wait_empty(30))
        self.assertEqual(now[0],300)


if __name__=='__main__':unittest.main()
