import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('coneall_timeline',Path(__file__).resolve().parents[1]/'scripts/coneall_timeline.py')
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
timeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(timeline)


class TimelineTests(unittest.TestCase):
    def test_overlapping_resources_are_not_double_counted(self):
        text='''[weight-fetcher] [dsv41-fetch-timeline] stage=0 submit_epoch_ms=100.0 fence_epoch_ms=105.0
[dsv41-fetch] stage=0 bytes=10 fetch_ms=5.0
[weight-fetcher] [dsv41-fetch-timeline] stage=1 submit_epoch_ms=104.0 fence_epoch_ms=110.0
[dsv41-fetch] stage=1 bytes=20 fetch_ms=6.0
[dsv41-prefill-experts] stage=1 rows=1 bytes=5 fetch_ms=2.0 begin_epoch_ms=112.0 end_epoch_ms=114.0
[dsv41-compute-timeline] layer=0 start=0 begin_epoch_ms=110.0 end_epoch_ms=120.0
[dsv41-host-timeline] layer=0 start=0 rows=1 begin_epoch_ms=109.0 end_epoch_ms=121.0
[dsv41-perf] layer=0 wall_ms=12.0 acquire_ms=1.0 compute_ms=10.0 prepare_ms=1.0 upload_ms=0.0
'''
        result=timeline.summarize(text)
        self.assertEqual(result['passes'][0]['fetch_envelope_busy_ms'],12)
        self.assertEqual(result['passes'][0]['stream_utilization_percent'],60)
        self.assertEqual([r['duration_ms'] for r in result['idle_intervals']],[2,6])
        self.assertEqual(len(result['histogram']),40)
        self.assertEqual(result['histogram'][0]['fetch_bytes'],25)
        self.assertFalse(result['histogram'][39]['executed'])

    def test_reverse_clocks_refuse(self):
        with self.assertRaises(ValueError):timeline.merged([(2,1)])

    def test_hint_traffic_and_join_remain_separate_from_demand(self):
        text='''[dsv41-fetch-timeline] stage=0 submit_epoch_ms=100.0 fence_epoch_ms=105.0
[dsv41-fetch] stage=0 bytes=10 fetch_ms=5.0
[dsv41-prefill-hint] stage=1 bytes=20 fetch_ms=3.0 begin_epoch_ms=104.0 end_epoch_ms=107.0 exposed_join_ms=0.2
[dsv41-compute-timeline] layer=0 start=0 begin_epoch_ms=106.0 end_epoch_ms=110.0
[dsv41-perf] layer=0 wall_ms=4.0 acquire_ms=0.0 compute_ms=4.0
'''
        result=timeline.summarize(text)
        self.assertEqual(result['passes'][0]['fetch_envelope_busy_ms'],7)
        self.assertEqual(result['passes'][0]['routed_fetch_wait_ms'],0)
        self.assertEqual(result['histogram'][0]['fetch_bytes'],20)
        self.assertEqual(result['histogram'][0]['exposed_hint_join_ms'],.2)

    def test_unmatched_native_pass_refuses(self):
        with self.assertRaises(ValueError):timeline.summarize('[dsv41-fetch] stage=0 bytes=1 fetch_ms=1.0\n[dsv41-perf] layer=0 wall_ms=1.0')


if __name__=='__main__':unittest.main()
