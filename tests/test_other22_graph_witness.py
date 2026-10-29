import importlib.util
from pathlib import Path
import unittest
path=Path(__file__).resolve().parents[1]/'examples/native-fp4/speed_proxy.py'
spec=importlib.util.spec_from_file_location('other22_speed_proxy',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class GraphWitnessTests(unittest.TestCase):
    def test_requested_graphs_cannot_pass_on_eager_execution(self):
        with self.assertRaisesRegex(ValueError,'insufficient'):module.graph_witnesses('native FFN completed, eager only')

    def test_embedding_launches_cannot_stand_in_for_layer_coverage(self):
        row='OTHER22_CUDA_GRAPH launch=1 captured=0 captures=1 replays=11 ledger_entries=0 bounded_executables=1\n'
        with self.assertRaisesRegex(ValueError,'insufficient'):module.graph_witnesses(row*12,563)

    def test_captures_and_replays_remain_distinct(self):
        rows='OTHER22_CUDA_GRAPH launch=1 captured=1 captures=1 replays=0 ledger_entries=2 bounded_executables=1\n'
        rows+='OTHER22_CUDA_GRAPH launch=1 captured=0 captures=1 replays=1 ledger_entries=2 bounded_executables=1\n'
        self.assertEqual(module.graph_witnesses(rows,2),dict(cuda_graph_launches=2,cuda_graph_captures=1,cuda_graph_replays=1))
