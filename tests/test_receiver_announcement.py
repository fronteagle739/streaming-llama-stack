from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import uuid
import socket
from types import SimpleNamespace

from slstack.__main__ import main
from slstack.mac import addresses, serve, wait_for_receiver


class ReceiverAnnouncementTests(unittest.TestCase):
    def test_active_bridge_is_announced_before_other_interfaces(self):
        def ipv4(value):
            return SimpleNamespace(family=socket.AF_INET, address=value)
        rows = {'en1':[ipv4('198.51.100.2')], 'bridge1':[ipv4('192.0.2.1')],
                'en0':[ipv4('169.254.1.2')], 'bridge0':[ipv4('203.0.113.1')],
                'lo0':[ipv4('127.0.0.1')]}
        states = {name:SimpleNamespace(isup=name!='bridge0') for name in rows}
        with patch('psutil.net_if_addrs',return_value=rows), patch('psutil.net_if_stats',return_value=states):
            self.assertEqual(addresses(),['192.0.2.1','198.51.100.2'])

    def test_launcher_relays_only_public_startup_lines_after_health(self):
        file = Path(tempfile.gettempdir()) / ('receiver-startup-' + uuid.uuid4().hex + '.log')
        file.write_text('internal diagnostic\nMAC ENDPOINT: http://127.0.0.1:8180\n'
                        'RTX command: edit the model paths\n',encoding='utf-8')
        job = Mock(log=file)
        job.proc.poll.return_value = None
        output = io.StringIO()
        try:
            with patch('slstack.mac.http_json',return_value={'ready':True}), redirect_stdout(output):
                wait_for_receiver({},job)
            self.assertEqual(output.getvalue().splitlines(),[
                'MAC ENDPOINT: http://127.0.0.1:8180','RTX command: edit the model paths'])
        finally:
            file.unlink()

    def test_receiver_prints_a_concrete_endpoint_in_the_rtx_command(self):
        output = io.StringIO()
        with patch('slstack.mac.Receiver') as receiver, \
             patch('slstack.mac.addresses',return_value=['127.0.0.1']), \
             patch('slstack.mac.ThreadingHTTPServer') as server, redirect_stdout(output):
            receiver.return_value.token = 'TEST_VALUE_ONLY_1234'
            server.return_value.serve_forever.side_effect = KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                serve({},'ds4',{})
            handler = server.call_args.args[1]
            self.assertEqual(handler.protocol_version,'HTTP/1.1')
            receiver.return_value.close.assert_called_once()
        self.assertIn('MAC ENDPOINT: http://127.0.0.1:8180',output.getvalue())
        self.assertIn('rtx --mac http://127.0.0.1:8180 --pair-token',output.getvalue())
        self.assertNotIn('<chosen-endpoint>',output.getvalue())

    def test_startup_failure_stops_owned_receiver_and_honours_cli_budget(self):
        cfg = {'runtime_root':'unused'}
        args = ['slstack','mac','--host-budget-gib','99']
        with patch('sys.argv',args), patch('slstack.__main__.load',return_value=cfg), \
             patch('slstack.mac.setup',return_value={'ready':True}), \
             patch('slstack.mac.budget',side_effect=lambda value,maximum:dict(value)) as budget, \
             patch('slstack.journal.atomic_json'), patch('slstack.process.OwnedProcess') as job, \
             patch('slstack.mac.wait_for_receiver',side_effect=TimeoutError('startup failed')):
            job.return_value.proc.poll.return_value = None
            with self.assertRaises(TimeoutError):
                main()
            self.assertEqual(budget.call_args.args[0]['mac_host_budget_gib'],99)
            job.return_value.stop.assert_called_once()


if __name__ == '__main__':
    unittest.main()
