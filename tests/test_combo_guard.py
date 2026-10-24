"""Actual Windows CPU Jobs; no GPU lease, model, engine or production ledger."""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import uuid
import psutil
from slstack.guard import _supervise
from slstack.lease import active_holds, reservation_admission


@unittest.skipUnless(os.name=='nt','Windows Job qualification')
class GuardTest(unittest.TestCase):
    def test_kit11_mac_hold_also_blocks_and_cannot_retire_its_rtx_hold(self):
        rtx='ACTIVE: KIT11 same; RTX'
        mac='MAC ACTIVE: KIT11 same; Mac'
        self.assertEqual(active_holds(rtx+'\n'+mac),[rtx,mac])
        self.assertEqual(active_holds(rtx+'\n'+mac+'\nMAC RELEASED: KIT11 same; Mac'),[rtx])
        self.assertEqual(active_holds('MAC ACTIVE: OTHER ignored; Mac'),[])
        self.assertEqual(active_holds('MAC ACTIVE: KIT11FIX build; Mac'),['MAC ACTIVE: KIT11FIX build; Mac'])
    def test_exact_cpu_reservation_and_outstanding_budget(self):
        line='ACTIVE: KIT11FIX Windows native plugin build; host budget 3 GiB; physical RAM reserve 6 GiB; no GPU or Studio restart requested.'
        self.assertFalse(reservation_admission([line],9.25,2,available=int(20.25*2**30))[0])
        self.assertTrue(reservation_admission([line],9.25,2,available=int(20.25*2**30)-1)[0])
        self.assertFalse(reservation_admission([line],9.25,2,available=int(16.25*2**30),owned_rss=4*2**30)[0])
        self.assertTrue(reservation_admission([line.replace('3 GiB','4 GiB')],9.25,2,available=32*2**30)[0])
        self.assertTrue(reservation_admission([line+' GPU hold'],9.25,2,available=32*2**30)[0])

    def test_without_foreign_reservation_live_job_keeps_its_physical_floor(self):
        self.assertTrue(reservation_admission([],8,2,available=9*2**30)[0])
        self.assertFalse(reservation_admission([],8,2,available=9*2**30,owned_rss=1*2**30)[0])
        self.assertTrue(reservation_admission([],8,2,available=2*2**30-1,owned_rss=1*2**30)[0])

    def test_current_smaller_cpu_hold_preserves_full_owner_reservation(self):
        line=('ACTIVE: KIT11FIX cpu-qualification; expires 2026-10-23T03:44:17.2433082Z; '
              'Windows-only CPU build/test, host budget 2 GiB, reserve 6 GiB; '
              'yield RAM for bounded CPU window; no Studio restart')
        blocked,receipt=reservation_admission([line],7,2,available=18*2**30)
        self.assertFalse(blocked)
        self.assertEqual(receipt['foreign_reserved_bytes'],9*2**30)
        self.assertTrue(reservation_admission([line],7,2,available=18*2**30-1)[0])
        self.assertFalse(reservation_admission([line],7,2,available=14*2**30,owned_rss=4*2**30)[0])
        for changed in (line.replace('budget 2','budget 4'),line.replace('reserve 6','reserve 7'),
                        line+'; GPU hold',line.replace('CPU build/test','GPU build/test'),
                        line.replace('; no Studio restart','')):
            self.assertTrue(reservation_admission([changed],7,2,available=32*2**30)[0])

    def run_cpu(self,foreign=False,pages=0,dynamic=None,owned_pages=0):
        root=Path(os.environ['STUDIO_RUNTIME_ROOT']);nonce=uuid.uuid4().hex
        prefix=root/('guard-cpu-'+nonce)
        holds=prefix.with_suffix('.holds')
        holds.write_text('ACTIVE: DS4COMBO '+nonce+'; owned CPU fixture\n'+('ACTIVE: KIT11 foreign-fixture; CPU fixture only\n' if foreign else ''))
        spec=dict(command=[sys.executable,'-B','-c','import time; time.sleep(.3)'],cwd=str(root),env=dict(os.environ),
            host_budget_gib=.5,min_available_gib=4,job_commit_gib=.5,timeout_s=5,
            owner_pid=os.getpid(),owner_created=psutil.Process().create_time(),stop=str(prefix.with_suffix('.stop')),
            log=str(prefix.with_suffix('.log')),receipt=str(prefix.with_suffix('.json')),paging_telemetry=True,
            gpu_lease_identity=nonce,gpu=dict(hold_file=str(holds)))
        if dynamic is not None:
            spec.pop('gpu_lease_identity');spec.pop('gpu')
            window=prefix.with_suffix('.window.json')
            spec.update(gpu_window_file=str(window),gpu_run_id=nonce,gpu_hold_file=str(holds))
            spec['env']['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
            source=('import os,time,psutil; from slstack.journal import atomic_json; '
                    'atomic_json('+repr(str(window))+',dict(active='+str(dynamic!='idle')+',run_id='+repr(nonce)+
                    ',owner_pid=os.getpid(),owner_created=psutil.Process().create_time(),lease_id='+repr(nonce)+
                    ',deadline=time.time()'+('-1' if dynamic=='expired' else '+60')+')); time.sleep(.3)')
            spec['command']=[sys.executable,'-B','-c',source]
        calls=0
        def paging():
            nonlocal calls
            calls+=1
            return (100+(pages if calls>1 else 0))*4096
        def attributed(observer,process):
            observer.last_processes=[dict(pid=process.pid,paged_bytes=owned_pages)] if owned_pages else []
            return owned_pages
        with patch('slstack.guard.paging_bytes',side_effect=paging), patch('studio.job_paging.JobPaging.sample',autospec=True,side_effect=attributed):
            code=_supervise(spec)
        return code,json.loads(Path(spec['receipt']).read_text())

    def test_same_owned_hold_does_not_kill_the_cpu_job(self):
        code,proof=self.run_cpu();self.assertEqual(code,0);self.assertIsNone(proof['reason'])

    def test_dynamic_server_window_preempts_on_foreign_hold(self):
        code,proof=self.run_cpu(foreign=True,dynamic='active');self.assertEqual(code,1)
        self.assertIn('preempted by foreign ACTIVE hold',proof['reason'])

    def test_idle_server_does_not_claim_or_block_a_gpu_window(self):
        code,proof=self.run_cpu(foreign=True,dynamic='idle');self.assertEqual(code,0)

    def test_dynamic_server_window_has_independent_deadline(self):
        code,proof=self.run_cpu(dynamic='expired');self.assertEqual(code,1)
        self.assertIn('GPU request window made no progress before its renewed deadline',proof['reason'])

    def test_foreign_hold_stops_only_the_owned_cpu_job(self):
        code,proof=self.run_cpu(foreign=True);self.assertEqual(code,1)
        self.assertIn('preempted by foreign ACTIVE hold',proof['reason'])

    def test_small_global_paging_growth_warns_and_does_not_kill(self):
        code,proof=self.run_cpu(pages=1024);self.assertEqual(code,0)
        self.assertEqual(proof['global_paging_growth'],4<<20)
        self.assertTrue(proof['paging_warnings'])

    def test_global_growth_over256mib_stops_the_owned_cpu_job(self):
        code,proof=self.run_cpu(pages=65537);self.assertEqual(code,1)
        self.assertIn('global pagefile/swap grew',proof['reason'])

    def test_owned_4kib_warns_and_telemetry_is_closed(self):
        code,proof=self.run_cpu(owned_pages=4096)
        self.assertEqual(code,0);self.assertIsNone(proof['reason'])
        self.assertTrue(proof['release_allowed']);self.assertEqual(proof['owned_paged_bytes'],4096)
        self.assertEqual(proof['paged_processes'],[dict(pid=proof['child_pid'],paged_bytes=4096)])
        self.assertTrue(proof['paging_telemetry_closed'])
        self.assertEqual(proof['paging_policy']['peak_owned_paged_bytes'],4096)
        self.assertTrue(any('64 MiB' in warning for warning in proof['paging_warnings']))

    def test_owned_exact64mib_warns(self):
        code,proof=self.run_cpu(owned_pages=64<<20)
        self.assertEqual(code,0);self.assertEqual(proof['paging_policy']['owned_paging_limit_bytes'],64<<20)
        self.assertTrue(proof['paging_telemetry_closed'])

    def test_owned_over64mib_stops_with_ram_still_available(self):
        code,proof=self.run_cpu(owned_pages=(64<<20)+4096)
        self.assertEqual(code,1);self.assertIn('job-attributed',proof['reason'])
        self.assertTrue(proof['release_allowed']);self.assertTrue(proof['paging_telemetry_closed'])

    def test_telemetry_closes_when_launch_fails(self):
        with patch('slstack.guard.PagingWatch') as watcher, patch('slstack.guard._supervise_owned',side_effect=OSError('launch refused')):
            spec=dict(host_budget_gib=.5,min_available_gib=4,
                receipt=str(Path(os.environ['STUDIO_RUNTIME_ROOT'])/('no-launch-'+uuid.uuid4().hex+'.json')))
            with self.assertRaisesRegex(OSError,'launch refused'):_supervise(spec)
            watcher.return_value.close.assert_called_once_with()

    def test_telemetry_cleanup_failure_cannot_leave_a_success_receipt(self):
        target=Path(os.environ['STUDIO_RUNTIME_ROOT'])/('trace-close-'+uuid.uuid4().hex+'.json')
        target.write_text(json.dumps(dict(exit_code=0,reason=None,release_allowed=True)))
        with patch('slstack.guard.PagingWatch') as watcher, patch('slstack.guard._supervise_owned',return_value=0):
            watcher.return_value.close.side_effect=OSError('trace stop refused')
            with self.assertRaisesRegex(OSError,'trace stop refused'):
                _supervise(dict(host_budget_gib=.5,min_available_gib=4,receipt=str(target)))
        proof=json.loads(target.read_text())
        self.assertFalse(proof['paging_telemetry_closed'])
        self.assertIn('trace stop refused',proof['reason'])


if __name__=='__main__':unittest.main()
