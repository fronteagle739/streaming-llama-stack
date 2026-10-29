import importlib.util
from pathlib import Path
import unittest
path=Path(__file__).resolve().parents[1]/'examples/native-fp4/speed_proxy.py'
spec=importlib.util.spec_from_file_location('other22_speed_proxy',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class GraphWitnessTests(unittest.TestCase):
    def test_queued_completion_requires_every_tile_and_layer(self):
        trace=''.join(f'OTHER22_ASYNC_LAYER_CERTIFIED layer={i} \n' for i in range(40))
        trace+='OTHER22_ASYNC_TILE_CERTIFIED layer=0 start=0 count=4096\n'
        self.assertEqual(module.layer_witnesses(trace,1,queued=True)['asynchronous_certified_tiles'],1)
        with self.assertRaisesRegex(ValueError,'certification'):module.layer_witnesses(trace,2,queued=True)
        with self.assertRaisesRegex(ValueError,'certification'):module.layer_witnesses(trace.replace('layer=39 ','layer=38 '),1,queued=True)

    def test_fusion_and_pointer_table_requests_cannot_pass_as_eager(self):
        with self.assertRaisesRegex(ValueError,'norm launches'):module.layer_witnesses('',1,fusion=1)
        calls='OTHER22_LAYER_FUSION rows=4096 width=5120 HC_BF16_RMS_weight_BF16=1\n'*2
        with self.assertRaisesRegex(ValueError,'Pointer-table'):module.layer_witnesses(calls,1,fusion=2)
        trace=calls+'OTHER22_BINDING_GRAPH launches=1 captures=1 pointer_table=1 mutable_rows=1\nOTHER22_BINDING_GRAPH launches=2 captures=1 pointer_table=1 mutable_rows=1\n'
        self.assertEqual(module.layer_witnesses(trace,1,fusion=2)['binding_graph_captures'],1)
        # Head/export scopes deliberately prohibit capture. They may retain
        # direct fusion, while both layer norm launches per tile stay graphed.
        extra=trace+'OTHER22_LAYER_FUSION rows=1 width=5120 HC_BF16_RMS_weight_BF16=1\n'
        self.assertEqual(module.layer_witnesses(extra,1,fusion=2)['scoped_eager_fusions'],1)
        with self.assertRaisesRegex(ValueError,'Pointer-table'):module.layer_witnesses(trace.replace('launches=2 captures=1','launches=2 captures=2'),1,fusion=2)

    def test_missing_native_coverage_cannot_certify_layer_experiments(self):
        with self.assertRaisesRegex(ValueError,'positive'):module.layer_witnesses('',0,queued=True)

    def test_requested_graphs_cannot_pass_on_eager_execution(self):
        with self.assertRaisesRegex(ValueError,'insufficient'):module.graph_witnesses('native FFN completed, eager only')

    def test_embedding_launches_cannot_stand_in_for_layer_coverage(self):
        row='OTHER22_CUDA_GRAPH launch=1 captured=0 captures=1 replays=11 ledger_entries=0 bounded_executables=1\n'
        with self.assertRaisesRegex(ValueError,'insufficient'):module.graph_witnesses(row*12,563)

    def test_captures_and_replays_remain_distinct(self):
        rows='OTHER22_CUDA_GRAPH launch=1 captured=1 captures=1 replays=0 ledger_entries=2 bounded_executables=1\n'
        rows+='OTHER22_CUDA_GRAPH launch=1 captured=0 captures=1 replays=1 ledger_entries=2 bounded_executables=1\n'
        self.assertEqual(module.graph_witnesses(rows,2),dict(cuda_graph_launches=2,cuda_graph_captures=1,cuda_graph_replays=1))
