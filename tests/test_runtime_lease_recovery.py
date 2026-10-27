import json
from pathlib import Path
import tempfile
import unittest
import psutil
from slstack.lease import recover_runtime_leases


class RuntimeLeaseRecoveryTests(unittest.TestCase):
    def test_custom_owner_recovers_both_runtimes_and_preserves_foreign_live_and_uncertain(self):
        root=Path(tempfile.mkdtemp(prefix='stack-lease-recovery-'))
        hold=root/'gpu.holds';hold.write_text('fixture ledger\n')
        current=psutil.Process()
        def record(location,identity,task='CUSTOM',live=False,uncertain=False):
            directory=location/'state/leases';directory.mkdir(parents=True,exist_ok=True)
            row=dict(id=identity,task=task,hold_file=str(hold),released=False,
                     guardian_pid=current.pid if live else 99999999,
                     guardian_created=current.create_time() if live else 0,
                     cleanup_uncertain=uncertain)
            path=directory/(identity+'.json');path.write_text(json.dumps(row));return path
        own=record(root,'own');nested=record(root/'single-engine','nested')
        protected=[record(root,'foreign',task='OTHER'),record(root,'live',live=True),
                   record(root/'single-engine','uncertain',uncertain=True)]
        before={p:p.read_bytes() for p in protected}
        self.assertEqual(set(recover_runtime_leases(root,'CUSTOM')),{'own','nested'})
        self.assertTrue(json.loads(own.read_text())['recovered'])
        self.assertTrue(json.loads(nested.read_text())['recovered'])
        self.assertEqual(before,{p:p.read_bytes() for p in protected})
        self.assertEqual(recover_runtime_leases(root,'CUSTOM'),[])


if __name__=='__main__':unittest.main()
