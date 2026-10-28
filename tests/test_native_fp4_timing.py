import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('fp4_timing',Path(__file__).parents[1]/'examples/native-fp4/timing_report.py')
timing=importlib.util.module_from_spec(spec);spec.loader.exec_module(timing)

class TimingBoundaries(unittest.TestCase):
    def fixture(self):
        row=dict(prompt_tokens=100,input_sha256='input',model_sha256='model',producer_sha256='producer',prefill_seconds=12)
        receipt=dict(status='complete',tokens=dict(count=100,sha256='input'),model=dict(sha256='model'),
            producer_binary_sha256='producer',caller_expected_build_sha256='producer',
            phase_ms=dict(embedding_40_layers_head=8500,append_owner_admission=400,chunk_token_preparation=100),
            stateful_chunks=dict(chunks=[dict(first=0,tokens=40,completed_context_tokens=40,prompt_processing_ms=3000),
                dict(first=40,tokens=60,completed_context_tokens=100,prompt_processing_ms=5000)]))
        return row,receipt

    def parse(self,row,receipt):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'receipt.json';path.write_text(json.dumps(receipt))
            return timing.from_receipt(path,row)

    def test_separates_prompt_clock_from_launch_to_ready(self):
        row,receipt=self.fixture();result=self.parse(row,receipt)
        self.assertEqual(result['prompt_prefill_seconds'],8.5)
        self.assertEqual(result['ttft_seconds'],12)
        self.assertEqual(result['setup_and_ready_overhead_seconds'],3.5)
        self.assertAlmostEqual(result['prompt_prefill_tok_s'],100/8.5)

    def test_rejects_mismatched_identity_missing_chunks_and_invalid_clocks(self):
        row,base=self.fixture()
        for key in ['input_sha256','model_sha256','producer_sha256']:
            candidate={**row,key:'wrong'}
            with self.subTest(key=key),self.assertRaises(ValueError):self.parse(candidate,base)
        for mutate in [lambda r:r.update(status='failed'),
            lambda r:r['stateful_chunks']['chunks'].pop(),
            lambda r:r['stateful_chunks']['chunks'][1].update(first=41),
            lambda r:r['phase_ms'].update(embedding_40_layers_head=9000),
            lambda r:r['phase_ms'].update(embedding_40_layers_head=float('nan'))]:
            receipt=copy.deepcopy(base);mutate(receipt)
            with self.assertRaises(ValueError):self.parse(row,receipt)
        with self.assertRaises(ValueError):self.parse({**row,'prefill_seconds':8},base)

    def test_recovers_after_fold_diagnostic_without_rewriting_original_evidence(self):
        row,receipt=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'producer.json';path.write_text(json.dumps(receipt))
            measurement=Path(directory)/'measurement.json';measurement.write_text(json.dumps(row))
            trace=Path(directory)/'trace.log'
            trace.write_text('LLAMADECODE_PREFILL_FOLD_TAIL tokens=100\nPending receipt (retained on failure): '+str(path)+'.pending-123-456\n')
            before=measurement.read_bytes(),trace.read_bytes()
            self.assertEqual(timing.recover(measurement,trace)['prompt_prefill_seconds'],8.5)
            self.assertEqual(before,(measurement.read_bytes(),trace.read_bytes()))

if __name__=='__main__':unittest.main()
