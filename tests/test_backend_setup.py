"""Installer integrity checks run without loading a model or touching a GPU."""
import io
import json
from pathlib import Path
import tarfile
import unittest
from unittest.mock import patch
import uuid

from slstack.backend_cache import Cache,digest,snapshot
from slstack.backend_setup import check
from slstack.backends import installed_binary
from slstack.config import ROOT


class BackendSetup(unittest.TestCase):
    def fixture(self):
        root=ROOT/'runtime/fixtures'/('backend-setup-'+uuid.uuid4().hex)
        root.mkdir(parents=True)
        return root

    def cache(self,root,data):
        archive=root/'source.tar.gz'
        archive.write_bytes(data)
        cache=Cache({'runtime_root':str(root)},directory=root,offline=True)
        cache.items={'sample':dict(id='sample',format='tar',filename=archive.name,
                                  sha256=digest(archive),bytes=len(data))}
        return cache

    def archive(self,name='source/file.c',data=b'int value;'):
        out=io.BytesIO()
        with tarfile.open(fileobj=out,mode='w:gz') as archive:
            member=tarfile.TarInfo(name);member.size=len(data)
            archive.addfile(member,io.BytesIO(data))
        return out.getvalue()

    def test_offline_integrity_and_divergent_extraction_are_preserved(self):
        root=self.fixture();cache=self.cache(root,self.archive())
        source=cache.unpack('sample',root/'unpacked')
        self.assertEqual((source/'file.c').read_bytes(),b'int value;')
        self.assertEqual(cache.unpack('sample',root/'unpacked'),source)
        (source/'file.c').write_bytes(b'unique local changes')
        with self.assertRaisesRegex(ValueError,'local changes'):cache.unpack('sample',root/'unpacked')
        self.assertEqual((source/'file.c').read_bytes(),b'unique local changes')
        (root/'source.tar.gz').write_bytes(b'truncated')
        with self.assertRaisesRegex(ValueError,'pinned hash'):cache.get('sample')

    def test_archive_traversal_never_writes_outside_staging(self):
        root=self.fixture();cache=self.cache(root,self.archive('source/../../escaped'))
        with self.assertRaisesRegex(ValueError,'Unsafe'):cache.unpack('sample',root/'unpacked')
        self.assertFalse((root/'escaped').exists())

    def test_runtime_inside_checkout_keeps_untracked_snapshot_inputs(self):
        root=self.fixture();source=root/'inputs';source.mkdir()
        (source/'file.c').write_bytes(b'untracked build input')
        dest,_=snapshot(source,root,'sample')
        self.assertEqual((dest/'file.c').read_bytes(),b'untracked build input')
        (dest/'file.c').write_bytes(b'unique change')
        with self.assertRaisesRegex(ValueError,'diverged'):snapshot(source,root,'sample')

    def test_build_only_and_changed_binary_cannot_be_activated(self):
        root=self.fixture();binary=root/'server';binary.write_bytes(b'test executable')
        state=root/'state/backends';state.mkdir(parents=True)
        receipt=state/'llama-server.json'
        record=dict(binary=str(binary),binaries={str(binary):digest(binary)},verification_pending=True)
        receipt.write_text(json.dumps(record))
        cfg={'runtime_root':str(root)}
        self.assertTrue(check(cfg,'llama-server')['built'])
        self.assertFalse(check(cfg,'llama-server')['ready'])
        with self.assertRaisesRegex(ValueError,'unverified'):installed_binary(root,'llama-server','fallback')
        record['verification_pending']=False;receipt.write_text(json.dumps(record))
        self.assertEqual(installed_binary(root,'llama-server','fallback'),str(binary))
        binary.write_bytes(b'different executable')
        self.assertFalse(check(cfg,'llama-server')['ready'])
        with self.assertRaises(ValueError):installed_binary(root,'llama-server','fallback')

    def test_selected_gpu_is_not_replaced_by_studio_device_zero(self):
        from slstack.general_engine import launch_spec
        cfg=dict(runtime_root=str(self.fixture()),model='missing.gguf',host='127.0.0.1',gpu_uuid='GPU-selected',
                 general=dict(server='server',rtx=dict(arch='kimi',remote_layers=61),
                              mac=dict(host='127.0.0.1',backend='mlx-receiver')))
        _,env,_=launch_spec(cfg)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'GPU-selected')

    def test_peer_uses_decode_role_without_batched_session(self):
        from slstack.general_engine import launch_spec
        cfg=dict(runtime_root=str(self.fixture()),model='missing.gguf',host='127.0.0.1',gpu_uuid='GPU-selected',
                 general=dict(mode='peer',server='server',rtx={},
                              mac=dict(host='127.0.0.1',backend='ds4-peer',model_path='mac-model.gguf')))
        argv,env,_=launch_spec(cfg)
        self.assertEqual(argv[argv.index('--peer-role')+1],'decode')
        self.assertNotIn('--batched-session',argv)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'GPU-selected')
        self.assertEqual(env['DS4_CUDA_ATTN_COMP_CACHE_Q8'],'0')

    def test_swift_traits_do_not_leave_unpinned_network_dependencies(self):
        from slstack.backend_setup import swift_manifest
        original='''dependencies: [
 .package(url: "https://example.com/known.git", from: "1.0", traits: [.trait(name: "Xet", condition: .when(traits: ["Xet"]))]),
 .package(url: "https://github.com/huggingface/swift-xet.git", from: "0.2.0"),
], targets: [.product(name: "Xet", package: "swift-xet", condition: .when(traits: ["Xet"]))],'''
        result=swift_manifest(original,{'https://example.com/known':'known'})
        self.assertIn('.package(path: "../known")',result)
        self.assertNotIn('swift-xet',result)
        self.assertNotIn('url:',result)
        with self.assertRaisesRegex(ValueError,'Unpinned'):swift_manifest(original,{})

    def test_mlx_build_receipts_are_checked_without_importing_the_runtime(self):
        from slstack.mlx_setup import inspect_installation
        root=self.fixture();venv=root/'decode-venv';python=venv/'bin/python'
        python.parent.mkdir(parents=True);python.write_bytes(b'test interpreter')
        marker=venv/'slstack-source.json'
        marker.write_text(json.dumps(dict(source_sha256='source-pin',native_glm=True)))
        package=venv/'lib/python3.12/site-packages/mlx_vlm/custom_kernels/glm_moe_dsa'
        package.mkdir(parents=True);artifact=package/'_ext.so';artifact.write_bytes(b'test native artifact')
        manifest=package/'native-build.json';record=dict(artifacts={'_ext.so':digest(artifact)},smoke_verified=False)
        manifest.write_text(json.dumps(record))
        with patch('slstack.mlx_setup.source_digest',return_value='source-pin'):
            result=inspect_installation({'runtime_root':str(root)})
            self.assertTrue(result['built']);self.assertFalse(result['ready'])
            record['smoke_verified']=True;manifest.write_text(json.dumps(record))
            self.assertTrue(inspect_installation({'runtime_root':str(root)})['ready'])
            artifact.write_bytes(b'changed artifact')
            self.assertFalse(inspect_installation({'runtime_root':str(root)})['ready'])


if __name__=='__main__':unittest.main()
