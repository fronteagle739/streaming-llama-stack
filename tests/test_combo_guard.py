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
        with patch('slstack.guard.paging_bytes',side_effect=paging), patch('studio.job_paging.JobPaging.sample',return_value=owned_pages):
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
        self.assertIn('GPU request window exceeded19minutes',proof['reason'])

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

    def test_owned_paging_stops_with_ram_still_available(self):
        code,proof=self.run_cpu(owned_pages=4096)
        self.assertEqual(code,1);self.assertIn('job-attributed',proof['reason'])
        self.assertTrue(proof['release_allowed']);self.assertEqual(proof['owned_paged_bytes'],4096)


if __name__=='__main__':unittest.main()
