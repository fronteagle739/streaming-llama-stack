import unittest
from unittest.mock import patch

from slstack.envvars import apply, rows


class TailPrefetchEnvironmentTests(unittest.TestCase):
    def test_default_and_explicit_native_candidate(self):
        names = {'LLAMADECODE_CONE_TAIL_PREFETCH', 'LLAMADECODE_GRAPH_MATERIALIZE_PLAN'}
        selected = [row for row in rows() if row['var'] in names]
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(row['scopes'] == ['rtx-llama'] for row in selected))
        with patch('slstack.envvars.rows', return_value=selected):
            env = apply({}, {}, 'rtx-producer')
            self.assertEqual(env['LLAMADECODE_CONE_TAIL_PREFETCH'], '0')
            self.assertEqual(env['LLAMADECODE_GRAPH_MATERIALIZE_PLAN'], '1')
            self.assertEqual(apply({}, {'env_overrides': {'LLAMADECODE_CONE_TAIL_PREFETCH': '1'}}, 'rtx-llama')['LLAMADECODE_CONE_TAIL_PREFETCH'], '1')
            self.assertNotIn('LLAMADECODE_CONE_TAIL_PREFETCH', apply({}, {}, 'mac-ds4'))
            with self.assertRaises(ValueError):
                apply({}, {'env_overrides': {'LLAMADECODE_CONE_TAIL_PREFETCH': '2'}}, 'rtx-llama')


if __name__ == '__main__':
    unittest.main()
