import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch
from types import SimpleNamespace

import psutil
from slstack.cooperative_stop import CooperativeOwnedProcesses,validate
from slstack.guard import _supervise,windows_job
from slstack.lease import recover_leases


class CooperativeStop(unittest.TestCase):
    def test_remote_shutdown_is_refused(self):
        for url in ('https://localhost/v1/stack/shutdown','http://example.invalid/v1/stack/shutdown',
                    'http://user:pass@localhost/v1/stack/shutdown','http://localhost/other'):
            with self.assertRaises(ValueError):validate(dict(url=url))

    def test_stalled_stop_forces_only_after_the_full_quiet_grace(self):
        with tempfile.TemporaryDirectory() as d:
            owned=object.__new__(CooperativeOwnedProcesses)
            owned.spec=dict(runtime_root=d,env={},log=str(Path(d)/'log'),
                shutdown_stall_seconds=60,
                cooperative_stop=dict(url='http://127.0.0.1:1/v1/stack/shutdown'))
            owned.identities={};owned.retired_identities=[]
            process=Mock(pid=4242);process.name.return_value='python.exe';process.create_time.return_value=1
            process.memory_info.return_value=SimpleNamespace(rss=100)
            now=[0.];events=[];alive=[True]
            owned.refresh=Mock(side_effect=lambda:[process] if alive[0] else [])
            owned.live=owned.refresh
            owned.request=Mock(side_effect=lambda *args:events.append(('cooperative',now[0])))
            def kill():events.append(('force',now[0]));alive[0]=False
            process.kill.side_effect=kill
            with patch('slstack.shutdown.time.monotonic',side_effect=lambda:now[0]), \
                 patch('slstack.shutdown.time.sleep',side_effect=lambda seconds:now.__setitem__(0,now[0]+10)), \
                 self.assertLogs('slstack.shutdown',level='CRITICAL') as messages:
                proof=owned.stop()
            self.assertTrue(proof['confirmed']);self.assertTrue(proof['forced_kill'])
            self.assertEqual(events,[('cooperative',0),('force',60)])
            self.assertIn('LAST FALLBACK',messages.output[0])
            self.assertIn('no progress for 60.0s',proof['force_reason'])

    def test_dead_cooperative_guardian_does_not_release_gpu_ownership(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);leases=root/'state/leases';leases.mkdir(parents=True)
            hold=root/'holds';hold.write_text('ACTIVE: TEST retained; exclusive configured GPU\n')
            record=leases/'retained.json'
            record.write_text(json.dumps(dict(task='TEST',id='retained',released=False,
                guardian_pid=99999999,guardian_created=0,hold_file=str(hold),cooperative_retirement=True)))
            self.assertEqual(recover_leases(root,'TEST'),[])
            self.assertNotIn('RELEASED',hold.read_text())
            self.assertFalse(json.loads(record.read_text())['released'])

    @unittest.skipUnless(os.name=='nt','Windows Job contract')
    def test_paging_failure_requests_http_stop_and_child_exits_itself(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'ultra-fixture').mkdir()
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            source='''import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
root=Path(%r)
class Handler(BaseHTTPRequestHandler):
 def do_POST(self):
  assert (root/'ultra-fixture/guard.abort').exists()
  self.send_response(200);self.end_headers();self.wfile.write(b'{}')
  threading.Thread(target=server.shutdown,daemon=True).start()
 def log_message(self,*args):pass
server=HTTPServer(('127.0.0.1',%d),Handler)
(root/'ready').touch()
server.serve_forever(poll_interval=.02);server.server_close()
print('COOPERATIVE_CHILD_EXIT',flush=True)
raise SystemExit(130)
'''%(str(root),port)
            spec=dict(command=[sys.executable,'-B','-c',source],cwd=d,env=dict(os.environ),runtime_root=d,
                host_budget_gib=.5,min_available_gib=1,job_commit_gib=.5,timeout_s=10,paging_telemetry=True,
                owner_pid=os.getpid(),owner_created=psutil.Process().create_time(),stop=str(root/'stop'),
                log=str(root/'log'),receipt=str(root/'receipt.json'),cooperative_stop=dict(url=f'http://127.0.0.1:{port}/v1/stack/shutdown'))
            def failure(*args,**kwargs):return 'synthetic global paging stop' if (root/'ready').exists() else None
            with patch('slstack.guard.PagingWatch.check',autospec=True,side_effect=failure),patch('slstack.guard.windows_job',wraps=windows_job) as job,patch('studio.job_guard.OwnedProcesses.stop',side_effect=AssertionError('must never force')):
                self.assertEqual(_supervise(spec),1)
            proof=json.loads((root/'receipt.json').read_text())
            self.assertEqual(proof['exit_code'],130);self.assertIn('paging stop',proof['reason'])
            self.assertTrue(proof['release_allowed']);self.assertFalse(proof['cleanup']['forced_kill'])
            self.assertFalse(job.call_args.kwargs['kill_on_close'])
            self.assertIn('COOPERATIVE_CHILD_EXIT',(root/'log').read_text())


if __name__=='__main__':unittest.main()
