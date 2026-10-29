import importlib.util
from pathlib import Path
import sys
import unittest
scripts=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('coneall_nsight',scripts/'coneall_nsight.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class NsightIntervals(unittest.TestCase):
    def test_concurrent_kernels_and_partial_dispatch_boundaries(self):
        intervals=module.merged([(5,9),(0,3),(2,6),(10,12)])
        indexed=([a for a,b in intervals],[b for a,b in intervals])
        self.assertEqual(module.indexed_overlap(indexed,2,11),8)
        self.assertEqual(module.indexed_overlap(indexed,9,10),0)
        self.assertEqual(module.indexed_overlap(indexed,-10,-1),0)
        self.assertEqual(module.indexed_overlap(indexed,11,30),1)

    def test_clip_preserves_idle_gaps_and_does_not_count_adjacency(self):
        intervals=[(1,3),(5,7)]
        self.assertEqual(module.clipped(intervals,3,6),[(5,6)])
        self.assertEqual(module.overlap(intervals,3,5),0)
        self.assertEqual(module.overlap(intervals,2,6),2)
