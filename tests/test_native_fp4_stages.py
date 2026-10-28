import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location('fp4_stages', Path(__file__).parents[1] / 'examples/native-fp4/stage_report.py')
stages = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stages)


class StageAccounting(unittest.TestCase):
    def test_cumulative_tables_and_nested_host_clocks(self):
        with tempfile.TemporaryDirectory() as directory:
            measurement = Path(directory) / 'measurement.json'
            measurement.write_text(json.dumps(dict(status='PASS', profiled=True, decode=None,
                prefill_seconds=10, arm='proxy', label='fixture', prompt_tokens=8,
                plan={'sweeps': [8]}, producer_sha256='p', decode_library_sha256='d',
                model_sha256='m', input_sha256='i')))
            trace = Path(directory) / 'trace.log'
            expert = 'MUL_MAT_ID|a=blk.0.ffn_gate_exps.weight|dst=2304,6,8,1|ashape=5120,2304,384,1'
            trace.write_text('DS4PERF_NODE_PROFILE_HEADER\tcolumns\n'
                f'DS4PERF_NODE_PROFILE\t1\t1000\t{expert}\n'
                'DS4PERF_NODE_PROFILE_END\trows=1\n'
                'DS4PERF_NODE_PROFILE_HEADER\tcolumns\n'
                f'DS4PERF_NODE_PROFILE\t2\t2000\t{expert}\n'
                'DS4PERF_NODE_PROFILE\t1\t500\tCPY|node=dsv41_hc_flat_norm\n'
                'DS4PERF_NODE_PROFILE\t1\t250\tMUL_MAT|a=blk.0.attn_compressor_kv.weight\n'
                'DS4PERF_NODE_PROFILE_END\trows=3\n'
                'DS4PERF_LAYER_WALL_LEDGER layer=0 acquire_ms=1000\n'
                'LLAMADECODE_TILE_TIMING layer=0 tile=0 rows=8 host_prepare_ms=100 host_bind_ms=70 host_register_ms=60 host_upload_ms=200 host_launch_ms=900 host_sync_ms=100 gpu_ms=950 compute_ms=1000 host_readback_ms=30 host_cleanup_ms=10\n')
            result = stages.summarize(measurement, trace)
            self.assertEqual(result['stage_seconds']['expert_gemm'], 2)
            self.assertEqual(result['routed_expert_flops'], 2 * 2 * 5120 * 2304 * 6 * 8)
            self.assertEqual(result['stage_seconds']['other_wall'], 7)
            self.assertEqual(result['gpu_seconds']['other_gpu'], .75)
            self.assertEqual(result['other_gpu_by_named_origin_seconds'],
                {'named_hc': .5, 'named_compressor_and_pool': .25})
            self.assertAlmostEqual(sum(result['tile_phase_seconds'].values()), 1.34)
            self.assertEqual(result['tile_nested_seconds']['host_register'], .06)
            self.assertEqual(result['tile_count'], 1)
            prompt = stages.summarize(measurement, trace, prompt_timing={'prompt_prefill_seconds':8})
            self.assertEqual(prompt['stage_seconds']['other_wall'], 5)
            self.assertEqual(prompt['stage_seconds_scope'], 'prompt_only')
            self.assertEqual(prompt['stage_wall_seconds'], 8)


if __name__ == '__main__':
    unittest.main()
