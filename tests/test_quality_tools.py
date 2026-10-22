import json
import threading
import unittest

from slstack.config import load
from slstack.router import Stack
from slstack.tools import Tools


class Backend:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = []
        self.state = {'supported': True, 'value': .35, 'per_phase': True,
                      'phases': {'reasoning': .9, 'tool_call': .35, 'answer': .35, 'code': .35}}

    def quality(self, value=None):
        self.calls.append(value)
        if value is not None:
            self.state = {**self.state, 'value': value if not isinstance(value, dict) else value['answer'],
                          'per_phase': isinstance(value, dict), 'phases': value if isinstance(value, dict) else None}
        return dict(self.state)


class QualityToolsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load(overrides={'gpu_uuid': 'GPU-test'})
        self.backend = Backend()
        self.tools = Tools(self.cfg, self.backend)

    def test_all_five_names_reach_the_backend(self):
        for name, expected in [('max_quality', 0), ('balanced', .35), ('fast', .65), ('faster', .9), ('max_speed', 1.36125)]:
            self.tools.call('set_quality_speed', {'value': name})
            self.assertEqual(self.backend.calls[-1], expected)

    def test_partial_four_phase_settings_preserve_other_phases(self):
        self.tools.call('set_quality_speed', {'reasoning': 'max_speed', 'code': 'max_quality'})
        self.assertEqual(self.backend.calls[-1], {'reasoning': 1.36125, 'tool_call': .35, 'answer': .35, 'code': 0})

    def test_legacy_three_phase_map_inherits_answer_for_code(self):
        self.tools.call('set_quality_speed', {'value': {'reasoning': 'fast', 'tool_call': 'max_quality', 'answer': 'balanced'}})
        self.assertEqual(self.backend.calls[-1]['code'], .35)

    def test_invalid_requests_never_touch_or_start_an_engine(self):
        for args in [{}, {'bogus': 1}, {'value': True}, {'value': float('nan')}, {'value': -1}, {'value': 1.36126}]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.tools.call('set_quality_speed', args)
        self.assertEqual(self.backend.calls, [])

    def test_description_is_the_minimal_provider_description(self):
        row = next(r for r in self.tools.catalog() if r['name'] == 'set_quality_speed')
        self.assertIn('Modes above Balanced are faster but may produce worse answers.', row['description'])
        self.assertLess(len(row['description'].split()), 60)
        self.assertNotIn('tok/s', row['description'])
        self.assertNotIn('KL', row['description'])
        self.assertEqual(set(row['inputSchema']['properties']), {'value', 'reasoning', 'tool_call', 'answer', 'code'})

    def test_mcp_control_does_not_wait_for_generation_lock(self):
        stack = Stack(self.cfg, self.backend)
        result = []
        self.backend.lock.acquire()
        thread = threading.Thread(target=lambda: result.append(stack.mcp({'id': 1, 'method': 'tools/call',
            'params': {'name': 'set_quality_speed', 'arguments': {'value': 'fast'}}})))
        try:
            thread.start()
            thread.join(1)
            finished_while_busy = not thread.is_alive()
        finally:
            self.backend.lock.release()
            thread.join(2)
        self.assertTrue(finished_while_busy)
        self.assertFalse(result[0]['result']['isError'])
        self.assertEqual(json.loads(result[0]['result']['content'][0]['text'])['value'], .65)


if __name__ == '__main__':
    unittest.main()
