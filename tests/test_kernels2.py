from pathlib import Path
import json
import unittest
from slstack.config import load,environment


class ImportedKernelContractTests(unittest.TestCase):
    def config(self,overrides=None):
        return load(overrides={'gpu_uuid':'GPU-test','env_overrides':overrides or {}})

    def test_native_parity_is_default_and_off_is_preserved(self):
        self.assertEqual(environment(self.config(),'rtx-producer')['LLAMA_DSV41_FUSED_HC'],'2')
        self.assertEqual(environment(self.config({'LLAMA_DSV41_FUSED_HC':'0'}),'rtx-producer')['LLAMA_DSV41_FUSED_HC'],'0')

    def test_attention_new_variant_preserves_memory_policy(self):
        env=environment(self.config(),'rtx-producer')
        self.assertEqual(env['LLAMA_DSV41_TILELANG_ATTENTION'],'2')
        self.assertEqual(environment(self.config({'LLAMA_DSV41_TILELANG_ATTENTION':'0'}),'rtx-producer')['LLAMA_DSV41_TILELANG_ATTENTION'],'0')
        self.assertEqual(env['GGML_CUDA_DISABLE_GRAPHS'],'1')
        self.assertEqual(env.get('LLAMA_DSV41_UNPADDED_CONE','0'),'0')

    def test_unknown_kernel_variants_are_refused(self):
        for key in ('LLAMA_DSV41_FUSED_HC','LLAMA_DSV41_TILELANG_ATTENTION','LLAMA_DSV41_UNPADDED_CONE'):
            with self.subTest(key=key),self.assertRaises(ValueError):
                environment(self.config({key:'3'}),'rtx-producer')

    def test_upstream_imports_are_byte_pinned(self):
        import hashlib
        root=Path(__file__).resolve().parents[1]/'examples/kernels2/upstream'
        manifest=json.loads((root/'manifest.json').read_text())
        self.assertEqual(manifest['revision'],'1bae0e1088993b9b82de08f59de5be5a34e42170')
        for item in manifest['files']:
            self.assertEqual(hashlib.sha256((root/Path(item['path']).name).read_bytes()).hexdigest(),item['sha256'])

    def test_embedded_attention_is_the_qualified_image(self):
        import hashlib,re
        repo=Path(__file__).resolve().parents[1]
        manifest=json.loads((repo/'examples/kernels2/native-sparse-image.json').read_text())
        source=(repo/'engines/streaming-llama/prefill_llama.cpp/ggml/src/ggml-cuda/dsv41-tilelang-sm120.inc').read_text()
        binary=bytes(int(v,16) for v in re.findall(r'0x([0-9a-f]{2})',source))
        self.assertEqual(len(binary),manifest['cubin_bytes'])
        self.assertEqual(hashlib.sha256(binary).hexdigest(),manifest['cubin_sha256'])


if __name__=='__main__':unittest.main()
