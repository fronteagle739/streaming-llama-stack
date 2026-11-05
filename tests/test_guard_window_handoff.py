"""A stable request-window/ledger observation must preserve strict lease checks."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from slstack import guard
from slstack.combo_single import SingleEngine, SingleProxy


class LeasePublicationTest(unittest.TestCase):
    def test_retirement_publishes_no_producer_before_releasing_its_hold(self):
        engine=SingleEngine.__new__(SingleEngine)
        published=[];at_release=[]
        engine.producer=SimpleNamespace(close=lambda:dict(confirmed=True))
        engine.poison=None;engine.lease_id='a'*32
        engine.leases=SimpleNamespace(close=lambda:at_release.append(published[-1]if published else'not-published'))
        engine.lease_listener=lambda:published.append(engine.lease_id)
        engine.retire('test certified retirement')
        self.assertEqual(at_release,[None])

    def test_window_reads_current_lease_only_after_acquiring_publication_lock(self):
        proxy=SingleProxy.__new__(SingleProxy)
        proxy.window_file='synthetic-window';proxy.run_id='run';proxy.stall=30
        proxy.engine=SimpleNamespace(lease_id='a'*32)
        class Lock:
            def __enter__(self):proxy.engine.lease_id='b'*32
            def __exit__(self,*_):pass
        proxy.window_lock=Lock();published=[]
        with patch('slstack.combo_single.atomic_json',side_effect=lambda _,row:published.append(row)),patch('slstack.combo_single.psutil.Process',return_value=SimpleNamespace(create_time=lambda:100)):
            proxy.window(True)
        self.assertEqual(published[0]['lease_id'],'b'*32)


class StableWindowTest(unittest.TestCase):
    def window(self,lease='a'*32,**changes):
        return dict(active=True,run_id='run',owner_pid=123,owner_created=100,lease_id=lease,deadline=1000,**changes)

    def observe(self,windows,holds):
        source=iter(windows);ledger=iter(holds)
        reader=SimpleNamespace(read=lambda:next(source),transition_retries=0)
        spec=dict(gpu_run_id='run',gpu_hold_file='synthetic-ledger')
        child=SimpleNamespace(pid=123,poll=lambda:None)
        with patch('slstack.guard.psutil.Process',return_value=SimpleNamespace(create_time=lambda:100,ppid=lambda:999)),patch('pathlib.Path.read_text',side_effect=lambda **_:next(ledger)):
            result=guard.gpu_window_observation(reader,spec,child)
        return result,reader.transition_retries

    def test_released_old_lease_retries_against_the_new_complete_window(self):
        before=self.window();after=self.window('b'*32)
        (window,holds),retries=self.observe([before,after,after],['',f'ACTIVE: DIRECTIONS {"b"*32}; new lease'])
        self.assertEqual(window,after);self.assertEqual(len(holds),1);self.assertEqual(retries,1)

    def test_certified_retirement_transition_has_no_gpu_owner(self):
        before=self.window();after=self.window(None)
        (window,holds),retries=self.observe([before,after],[''])
        self.assertIsNone(window['lease_id']);self.assertIsNone(holds);self.assertEqual(retries,1)

    def test_stable_missing_hold_stays_missing_for_the_guard_to_reject(self):
        window=self.window()
        (observed,holds),retries=self.observe([window,window],[''])
        self.assertEqual(observed,window);self.assertEqual(holds,[]);self.assertEqual(retries,0)

    def test_stable_foreign_hold_is_not_filtered_out(self):
        window=self.window();foreign=f'ACTIVE: FOREIGN {"b"*32}; foreign GPU'
        (_,holds),_=self.observe([window,window],[foreign])
        self.assertEqual(holds,[foreign])

    def test_invalid_process_or_run_identity_fails_before_accepting_a_transition(self):
        for change in (dict(run_id='other'),dict(owner_pid=456),dict(owner_created=200)):
            bad=self.window();bad.update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):self.observe([bad],[])

    def test_continuously_changing_lease_is_bounded_and_fails(self):
        windows=[self.window(str(i)*32)for i in range(5)]
        with self.assertRaisesRegex(ValueError,'did not stabilize'):self.observe(windows,['']*4)

    def test_heartbeat_deadline_change_keeps_the_same_ownership_snapshot(self):
        before=self.window();after=dict(before,deadline=2000)
        (observed,holds),retries=self.observe([before,after],[f'ACTIVE: DIRECTIONS {"a"*32}; own lease'])
        self.assertEqual(observed['deadline'],2000);self.assertEqual(retries,0);self.assertEqual(len(holds),1)

    def test_cpu_reservation_never_substitutes_for_the_missing_own_gpu_hold(self):
        reservation='ACTIVE: KIT11FIX Windows native plugin build; host budget 3 GiB; physical RAM reserve 6 GiB; no GPU or Studio restart requested.'
        own=f'ACTIVE: DIRECTIONS {"a"*32}; own GPU'
        args=('a'*32,{},7,2,32*2**30,1*2**30)
        self.assertIsNone(guard.gpu_lease_observation([own,reservation],*args)[0])
        self.assertEqual(guard.gpu_lease_observation([reservation],*args)[0],'own ACTIVE hold is no longer present')


if __name__=='__main__':unittest.main()
