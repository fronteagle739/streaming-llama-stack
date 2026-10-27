"""Check retained backend build inputs without loading a model or a GPU library."""
from pathlib import Path
import ast
import hashlib
import json
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from slstack.backends import BACKENDS


def audit(root=ROOT):
    components={
        'ds4':['engines/ds4-mac/ds4.c','engines/ds4-mac/ds4_metal.m','engines/ds4-mac/Makefile'],
        'llama':['engines/streaming-llama/prefill_llama.cpp/tools/decode-receiver/CMakeLists.txt',
                 'engines/streaming-llama/prefill_llama.cpp/tools/decode-receiver/main.cpp',
                 'engines/streaming-llama/prefill_llama.cpp/tools/decode-receiver/tail_tokens.cpp',
                 'engines/streaming-llama/prefill_llama.cpp/tools/decode-receiver/tail_tokens.h'],
        'llama-upstream':['engines/upstream-decode-receiver/CMakeLists.txt','engines/upstream-decode-receiver/wir-disabled.cpp',
                          'engines/upstream-decode-receiver/PROVENANCE.json'],
        'mlx':['engines/mlx-receiver/mlx_token_stream_server.py','engines/mlx-receiver/kv_handoff.py',
               'engines/mlx-receiver/kv_converter.py','engines/mlx-receiver/mtp_openai_shim.py',
               'engines/mlx-receiver/glm52_mtp.py','engines/mlx-receiver/glm52_mtp_drafter.py',
               'slstack/mlx_setup.py','locks/mac-mlx-py312.txt','vendor/mac-mlx/PROVENANCE.json'],
        'mference':['engines/mference/Package.swift','engines/mference/Package.resolved',
                    'engines/mference/LICENSE','engines/mference/LICENSE-APACHE',
                    'engines/mference/Sources/MferenceServer/Command/main.swift'],
        'mlx-serve':['engines/mlx-serve/build.zig','engines/mlx-serve/build.zig.zon',
                     'engines/mlx-serve/scripts/build-mlx.sh','engines/mlx-serve/scripts/fetch-llama.sh'],
        'omlx':['vendor/studio/studio/backend_argv.py'],
        'ds4-peer':['vendor/studio/studio/peer_launch.py','engines/ds4-peer/ds4_peer.c',
                    'engines/ds4-peer/ds4_peer.h','engines/ds4-peer/Makefile','engines/ds4-peer/ds4_metal.m'],
    }
    rows=[]
    for name,spec in BACKENDS.items():
        missing=[p for p in components[spec.setup] if not (root/p).is_file()]
        rows.append(dict(backend=name,transport=spec.transport,source_inputs_present=not missing,missing=missing,
                         qualification='Source audit only; does not establish native build, model compatibility or GPU output.'))
    native=root/'vendor/mac-mlx/mlx_vlm/custom_kernels/glm_moe_dsa/csrc'
    required=['CMakeLists.txt','bindings.cpp','dsa_indexer.cpp','dsa_indexer.h','exact_block_attention.cpp',
              'fused_moe.cpp','sparse_mla.cpp','sparse_mla.h','kernels/quantized_glm.h',
              'kernels/steel_attention_block_token.h','kernels/steel_dsa_indexer_score.h','kernels/steel_sparse_mla.h',
              'mlx/backend/metal/kernels/steel/attn/params.h']
    native_missing=[p for p in required if not (native/p).is_file()]
    manifest=json.loads((root/'vendor/mac-mlx/PROVENANCE.json').read_text())
    hashes=[]
    for row in manifest['files']:
        file=root/'vendor/mac-mlx'/row['path']
        if not file.is_file() or hashlib.sha256(file.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=row['export_sha256']:hashes.append(row['path'])
    peer_manifest=json.loads((root/'engines/ds4-peer/PROVENANCE.json').read_text())
    peer_hashes=[]
    for row in peer_manifest['files']:
        file=root/'engines/ds4-peer'/row['path']
        if not file.is_file() or hashlib.sha256(file.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=row['export_sha256']:peer_hashes.append(row['path'])
    # Resolve local receiver imports against the retained implementation modules.
    receiver=root/'engines/mlx-receiver';modules={p.stem for p in receiver.glob('*.py')}
    needed={'kv_handoff','kv_converter','glm52_mtp','glm52_mtp_drafter','kimi_vision','mlx_adapters',
            'mtp_adaptive_depth','prompt_cache_store','swi_reasoning'}
    missing_modules=sorted(needed-modules)
    for p in receiver.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    package=tomllib.loads((root/'vendor/mac-mlx/pyproject.toml').read_text())
    globs=package['tool']['setuptools']['package-data'].get('mlx_vlm.custom_kernels.glm_moe_dsa',[])
    packaged={p.relative_to(native.parent).as_posix() for pattern in globs for p in native.parent.glob(pattern) if p.is_file()}
    unpackaged=['csrc/'+p for p in required if 'csrc/'+p not in packaged]
    return dict(backends=rows,mlx_native_missing=native_missing,mlx_unpacked_build_inputs=unpackaged,
                mlx_source_files=len(manifest['files']),mlx_hash_mismatches=hashes,receiver_missing_modules=missing_modules,
                peer_hash_mismatches=peer_hashes,
                all_retained_components_complete=not(native_missing or unpackaged or hashes or peer_hashes or missing_modules or any(row['missing'] for row in rows)),
                external_requirements={'mlx-serve':'Initialize the pinned source and stage its MLX/MLX-C, libllama, Jinja and webp build dependencies.',
                                       'omlx':'An isolated pinned oMLX installation with compiled native kernels is required; an argv builder alone is not an installed engine.'})


if __name__=='__main__':
    result=audit();print(json.dumps(result,indent=2))
    raise SystemExit(not result['all_retained_components_complete'])
