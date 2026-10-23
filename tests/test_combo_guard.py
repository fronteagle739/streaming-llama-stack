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
from slstack.lease import active_holds


@unittest.skipUnless(os.name=='nt','Windows Job qualification')
class GuardTest(unittest.TestCase):
    def test_kit11_mac_hold_also_blocks_and_cannot_retire_its_rtx_hold(self):
        rtx='ACTIVE: KIT11 same; RTX'
        mac='MAC ACTIVE: KIT11 same; Mac'
        self.assertEqual(active_holds(rtx+'\n'+mac),[rtx,mac])
        self.assertEqual(active_holds(rtx+'\n'+mac+'\nMAC RELEASED: KIT11 same; Mac'),[rtx])
        self.assertEqual(active_holds('MAC ACTIVE: OTHER ignored; Mac'),[])
        self.assertEqual(active_holds('MAC ACTIVE: KIT11FIX build; Mac'),['MAC ACTIVE: KIT11FIX build; Mac'])
    def run_cpu(self,foreign=False,pages=0,dynamic=None):
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
        def snapshot():
            nonlocal calls
            calls+=1
            return dict(paging_file_usage=dict(first=100+(pages if calls>1 else 0)))
        with patch('slstack.memory_counters.snapshot',side_effect=snapshot):code=_supervise(spec)
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
        self.assertEqual(proof['paging_warnings'][0]['global_pagefile_growth_bytes'],4<<20)

    def test_global_growth_over256mib_stops_the_owned_cpu_job(self):
        code,proof=self.run_cpu(pages=65537);self.assertEqual(code,1)
        self.assertIn('SYSTEM PAGING PRESSURE',proof['reason'])


if __name__=='__main__':unittest.main()
