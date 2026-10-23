import io
import json
import threading
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

from slstack.mac import Receiver, replacement_budget, serve


class MacReplacementTests(unittest.TestCase):
    def test_retired_memory_must_recover_without_shrinking_the_cap(self):
        memory = [SimpleNamespace(available=n*2**30) for n in (80, 200, 450)]
        cfg = {'min_available_gib':32}
        with patch('psutil.virtual_memory',side_effect=memory), \
             patch('slstack.mac.time.sleep') as sleep, redirect_stdout(io.StringIO()):
            result = replacement_budget(cfg,400,384)
        self.assertEqual(result['host_budget_gib'],384)
        self.assertEqual(result['min_available_gib'],32)
        self.assertEqual(sleep.call_count,2)
        self.assertNotIn('mac_host_budget_gib',cfg)

    def test_replacement_wait_is_bounded(self):
        with patch('psutil.virtual_memory',return_value=SimpleNamespace(available=80*2**30)), \
             patch('slstack.mac.time.monotonic',side_effect=[10,71]), \
             patch('slstack.mac.time.sleep') as sleep:
            with self.assertRaisesRegex(MemoryError,'did not recover'):
                replacement_budget({'min_available_gib':32},400,384)
        sleep.assert_not_called()

    def test_uncertain_cleanup_preserves_ownership_and_blocks_replacement(self):
        receiver = Receiver.__new__(Receiver)
        receiver.cfg = {}
        receiver.backend = 'ds4'
        receiver.engine_mode = False
        receiver.engine_budget = 384
        engine = receiver.engine = Mock()
        engine.proc.poll.return_value = None
        engine.receipt.is_file.return_value = True
        engine.receipt.read_text.return_value = json.dumps({'cleanup_confirmed':False})
        with patch('slstack.mac.replacement_budget') as admission, \
             patch('slstack.mac.OwnedProcess') as launch:
            with self.assertRaisesRegex(RuntimeError,'cleanup unconfirmed') as failure:
                receiver.ready(True)
        self.assertTrue(failure.exception.owned_cleanup_uncertain)
        self.assertIs(receiver.engine,engine)
        admission.assert_not_called()
        launch.assert_not_called()

    def test_certified_cleanup_precedes_replacement_admission(self):
        receiver = Receiver.__new__(Receiver)
        receiver.cfg = {}
        receiver.backend = 'ds4'
        receiver.engine_mode = False
        receiver.engine_budget = 384
        engine = receiver.engine = Mock()
        engine.proc.poll.return_value = None
        engine.receipt.is_file.return_value = True
        engine.receipt.read_text.return_value = json.dumps({'cleanup_confirmed':True})
        def admission(cfg, maximum, previous):
            engine.stop.assert_called_once()
            self.assertIsNone(receiver.engine)
            self.assertEqual(previous,384)
            raise MemoryError('bounded refusal')
        with patch('slstack.mac.replacement_budget',side_effect=admission), \
             patch('slstack.mac.OwnedProcess') as launch:
            with self.assertRaisesRegex(MemoryError,'bounded refusal'):
                receiver.ready(True)
        launch.assert_not_called()

    def test_memory_refusal_returns_http_error_instead_of_closing_connection(self):
        with patch('slstack.mac.Receiver') as receiver, \
             patch('slstack.mac.ThreadingHTTPServer') as server, redirect_stdout(io.StringIO()):
            receiver.return_value.token = 'TEST_VALUE_ONLY_1234'
            receiver.return_value.lock = threading.RLock()
            receiver.return_value.chat.side_effect = MemoryError('physical admission refused')
            serve({'mac_bind':'127.0.0.1'},'ds4',{})
            handler_type = server.call_args.args[1]
        handler = handler_type.__new__(handler_type)
        handler.path = '/pair/qualification'
        payload = b'{"mode":"native"}'
        handler.headers = {'Content-Length':str(len(payload))}
        handler.rfile = io.BytesIO(payload)
        handler.authorized = lambda:True
        handler.reply = Mock()
        handler.do_POST()
        handler.reply.assert_called_once_with(409,{'error':'physical admission refused'})


if __name__ == '__main__':
    unittest.main()
