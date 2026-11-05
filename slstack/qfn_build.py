"""Bind the QFN installation to the actual compiled source and linked inputs."""
import json,subprocess
from pathlib import Path
from .config import ROOT
from .journal import sha256
from .model_compatibility import digest


def source_inventory():
    base=ROOT/'engines/streaming-llama/prefill_llama.cpp'
    files=[p for p in (base/'tools/qfn-engine').rglob('*') if p.is_file() and p.suffix in {'.cpp','.h','.hpp','.cu','.cuh','.inc'}]
    files.extend(p for p in (base/'src').glob('*') if p.is_file() and (p.suffix in {'.h','.hpp','.inc'} or p.name=='llama-dsv41-engram-prefetch.cpp'))
    files.extend(p for p in (base/'include').glob('*.h'))
    files.extend(p for folder in ('ggml/include','ggml/src') for p in (base/folder).glob('*.h'))
    files.append(base/'ggml/src/ggml-cuda/ggml-cuda.cu')
    files.extend(p for p in (base/'ggml/src/ggml-cuda').rglob('*') if p.is_file() and p.suffix in {'.h','.cuh'})
    files.append(ROOT/'engines/ds4-mac/qfn-native-layout.h')
    files.extend(ROOT/'engines/streaming-llama/flashinfer-sm120'/name for name in
        ('qfn_native_nvfp4_bridge.cu','qfn_native_nvfp4_bridge.h','qfn_packed_vector.cuh'))
    return {p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(set(files))}


def verified_build(manifest,engine):
    row=json.loads(Path(manifest).read_text(encoding='utf-8'))
    current=source_inventory()
    if row.get('exit_code')!=0 or row.get('source_files')!=current or row.get('source_sha256')!=digest(current):
        raise ValueError('QFN source differs from its actual compiled input inventory; rebuild')
    if Path(row['output']).resolve()!=Path(engine).resolve() or row['sha256']!=sha256(Path(engine)):
        raise ValueError('QFN producer executable differs from its build manifest')
    for name,value in row['inputs'].items():
        if sha256(Path(name))!=value:raise ValueError('QFN linked input changed since build: '+name)
    return row
