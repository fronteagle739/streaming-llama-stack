import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('singlepass', Path(__file__).resolve().parents[1]/'scripts/analyze_singlepass.py')
m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class Profile(unittest.TestCase):
    def test_cumulative_snapshots_across_sweeps(self):
        lines = []
        for count in (2, 5):
            lines.extend(['[dsv41-fetch] stage=0 bytes=1\n',
                'DS4PERF_NODE_PROFILE_HEADER\tcount\tgpu_ms\n',
                f'DS4PERF_NODE_PROFILE\t1\t{count}\tMUL_MAT_ID|node=gate\n',
                'SINGLEPASS_LAYER_NODE_SNAPSHOT layer=0\n'])
        result = m.analyze(lines)
        self.assertEqual([r['expert_gemm_gpu_ms'] for r in result['layers']], [2, 3])
        self.assertEqual(result['totals']['expert_gemm_gpu_ms'], 5)

    def test_nonadditive_host_clocks_and_publication(self):
        result = m.analyze(['[dsv41-fetch] stage=0 bytes=1\n',
            'LLAMADECODE_TILE_TIMING layer=2 tile=0 host_prepare_ms=5 host_bind_ms=3 host_register_ms=2 host_sync_ms=7 compute_ms=8\n',
            'DS4PERF_NODE_PROFILE_HEADER\tcount\tgpu_ms\n',
            'DS4PERF_NODE_PROFILE\t1\t4\tCPY|b=dsv41_compressed_l2\n',
            'SINGLEPASS_LAYER_NODE_SNAPSHOT layer=2\n'])
        t = result['totals']
        self.assertEqual(t['host_prepare_ms'], 5)
        self.assertEqual(t['compute_ms'], 8)
        self.assertEqual(t['cache_publication_gpu_ms'], 4)

    def test_backwards_profile_refused(self):
        with self.assertRaisesRegex(ValueError, 'backwards'):
            m.analyze(['DS4PERF_NODE_PROFILE_HEADER\n',
                'DS4PERF_NODE_PROFILE\t1\t4\tMUL_MAT_ID|x\n',
                'SINGLEPASS_LAYER_NODE_SNAPSHOT layer=0\n',
                'DS4PERF_NODE_PROFILE_HEADER\n',
                'DS4PERF_NODE_PROFILE\t1\t3\tMUL_MAT_ID|x\n',
                'SINGLEPASS_LAYER_NODE_SNAPSHOT layer=1\n'])

    def test_spill_byte_counts_use_completed_id_widths(self):
        result = m.analyze(['[dsv41-fetch] stage=0 bytes=1\n',
            'LLAMADECODE_PREFILL_SPILL_TILE layer=3 start=49152 count=100 h2d_ms=1 d2h_ms=2\n',
            'DS4PERF_RESIDENT_IDS_COMPLETE layer=3 start=49152 count=100 selected_width=512 candidate_width=2048\n'])
        self.assertEqual(result['totals']['spill_d2h_bytes'], 100*40976)
        self.assertEqual(result['totals']['spill_h2d_bytes'], 100*(40976+512*4))
        self.assertEqual(result['spill_id_witnesses_missing'], 0)

    def test_missing_cumulative_nodes_are_not_silently_counted_again(self):
        with self.assertRaisesRegex(ValueError, 'lost previously'):
            m.analyze(['DS4PERF_NODE_PROFILE_HEADER\n',
                'DS4PERF_NODE_PROFILE\t1\t4\tMUL_MAT_ID|x\n',
                'SINGLEPASS_LAYER_NODE_SNAPSHOT layer=0\n',
                'DS4PERF_NODE_PROFILE_HEADER\n',
                'SINGLEPASS_LAYER_NODE_SNAPSHOT layer=1\n'])


if __name__ == '__main__':
    unittest.main()
