from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace
import unittest
from studio.streamed_lora_argv import streamed_lora_train_argv
from slstack.config import ROOT


class StreamedRecipeArgvTests(unittest.TestCase):
    def config(self, **changes):
        values = dict(quant_stream_arch='deepseek41', quant_stream_steps=100,
            quant_stream_lr=0.002, quant_stream_rank=8, quant_stream_alpha=16,
            quant_stream_lora_layers=12, quant_stream_batch=40,
            quant_stream_dataset_size=40, quant_stream_max_seq=64, quant_stream_seed=7,
            quant_stream_mix_neutral=0, quant_stream_dataset='TEST-corpus.jsonl',
            quant_stream_checkpoint_every=20, quant_stream_gpu_buffers=3,
            quant_stream_eval_tokens=8, quant_stream_rss_limit_gb=8,
            quant_stream_lr_phases='', quant_stream_prefix_cache=True,
            quant_stream_directstorage=True, quant_stream_suffix_ds=False,
            quant_stream_no_eval=True, quant_stream_selftest=False,
            quant_native_feature_url='http://127.0.0.1:19001',
            quant_native_feature_token_file='TEST-provider.token',
            quant_native_base_sha256='a' * 64)
        return SimpleNamespace(**(values | changes))

    def build(self, cfg):
        return streamed_lora_train_argv(cfg, python=sys.executable,
            trainer=ROOT / 'vendor/engines/cosmos_windows/gguf_qat/train_gguf_lora_streamed.py',
            gguf='TEST-not-loaded.gguf', save_dir='TEST-output')

    def test_studio_generated_argv_is_accepted_by_the_existing_trainer_parser(self):
        argv = self.build(self.config())
        result = subprocess.run([argv[0], '-X', 'utf8', '-B', *argv[1:], '--help'],
            cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=20,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('--no-prefix-cache', argv)

    def test_native_provider_must_be_loopback_and_bound_to_the_base(self):
        for change in [dict(quant_native_feature_url='https://example.org'),
                       dict(quant_native_base_sha256=''), dict(quant_native_feature_token_file='')]:
            with self.assertRaises(ValueError):
                self.build(self.config(**change))

    def test_native_recipe_requires_one_complete_prefix_and_separate_native_eval(self):
        for change in [dict(quant_stream_prefix_cache=False), dict(quant_stream_no_eval=False),
                       dict(quant_stream_suffix_ds=True), dict(quant_stream_batch=8),
                       dict(quant_stream_merge_after=True)]:
            with self.assertRaises(ValueError):
                self.build(self.config(**change))


if __name__ == '__main__':
    unittest.main()
