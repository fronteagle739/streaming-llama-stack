import array
import importlib.util
from pathlib import Path
import struct
import unittest
from unittest.mock import mock_open,patch

spec=importlib.util.spec_from_file_location('attribution',Path(__file__).parents[1]/'scripts/perfexp_attribution.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class AttributionEvidenceTests(unittest.TestCase):
    def load(self,data):
        with patch.object(Path,'stat') as stat,patch.object(Path,'open',mock_open(read_data=data)):
            stat.return_value.st_size=len(data)
            return module.load(Path('trace.bin'))

    def record(self):
        return struct.pack('<5I6i',0x52313444,0,2125,3,6,0,1,2,3,4,383)

    def test_valid_ids(self):
        self.assertEqual(list(self.load(self.record())[(0,2125,3)]),[0,1,2,3,4,383])

    def test_corruption_refused(self):
        record=self.record()
        for data in (b'',record[:-1],record+b'x',record+record,b'BAD!'+record[4:],
                     struct.pack('<5I6i',0x52313444,0,2125,3,6,0,1,2,3,4,384)):
            with self.subTest(data=data[:20]),self.assertRaises(ValueError):self.load(data)

    def test_incomplete_coverage_refused(self):
        trace=self.load(self.record())
        with self.assertRaisesRegex(ValueError,'coverage'):module.compare(trace,trace)

    def test_first_divergence_and_exact_route_sets(self):
        trace={(layer,2125,point):array.array('i' if point==3 else 'f',range(6) if point==3 else [1.,2.])
               for layer in range(40) for point in (0,1,2,3,6,11)}
        trace[(0,2125,12)]=array.array('f',[1.,0.,0.,0.])
        other={k:array.array(v.typecode,v) for k,v in trace.items()}
        other[(0,2125,0)][0]=1.25
        other[(3,2125,3)]=array.array('i',[0,1,2,3,4,7])
        report=module.compare(trace,other)
        self.assertEqual(report['first_hidden_divergence']['point'],'attention_input')
        self.assertEqual(report['first_expert_set_divergence']['layer'],3)
        self.assertEqual(report['same_expert_sets'],39)

if __name__=='__main__':unittest.main()
