"""Rebuild changed QFN objects into local archives; never modify the verified input build.

Run under the shared serial build lock and 4GiB RSS/6GiB commit/6GiB floor guardian.
The unchanged input objects are read-only. Every compiler output, response file, archive,
log and final executable is under the task root, with at most two compilers.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

STACK = Path(__file__).resolve().parents[1]
ROOT = STACK.parent
OLD = None
OUT = ROOT / 'build/nvfp4-native'
NINJA = shutil.which('ninja')
VCVARS = None
NATIVE_VCVARS = None


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(label, command, vcvars=None):
    vcvars = vcvars or VCVARS
    batch = OUT / (label + '.cmd')
    batch.write_text('@echo off\ncall "' + vcvars + '" >nul 2>&1\nif errorlevel 1 exit /b 90\n' +
                     command + '\nif errorlevel 1 exit /b %ERRORLEVEL%\n')
    with (OUT / (label + '.log')).open('wb') as log:
        child = subprocess.run(['cmd', '/d', '/c', str(batch)], cwd=OUT, stdout=log, stderr=subprocess.STDOUT)
    if child.returncode:
        print((OUT / (label + '.log')).read_text(errors='replace')[-10000:], flush=True)
        raise RuntimeError(f'{label} failed: {child.returncode}; see {OUT / (label + ".log")}')
    print(label + ' complete', flush=True)


def native_path(path):
    return str(path).replace('/', '\\')


def main():
    global ROOT,OLD,OUT,VCVARS,NATIVE_VCVARS,NATIVE_RECIPE
    if not NINJA:
        raise RuntimeError('The existing input build requires ninja on PATH')
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--input-build', type=Path, required=True)
    parser.add_argument('--native-recipe', type=Path, required=True)
    parser.add_argument('--vcvars', type=Path, required=True)
    parser.add_argument('--native-vcvars', type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--relink-only', action='store_true', help='Reuse the guarded, completed local CUDA/base object closure')
    selection.add_argument('--backend-core', action='store_true',
                           help='Recompile ggml-cuda.cu and qfn-qsa-attn.cu; hash/retain the other43 accepted objects')
    selection.add_argument('--qsa-integration', action='store_true',
                           help='Recompile ggml-cuda.cu, qfn-qsa-attn.cu and CPU ops.cpp; retain all other accepted objects')
    selection.add_argument('--calibrated-decode', action='store_true',
                           help='Recompile ggml-cuda.cu and CPU ops.cpp for the mode5 node; retain the other44 accepted objects')
    args = parser.parse_args()
    ROOT=args.workspace.resolve();OLD=args.input_build.resolve();OUT=ROOT/'build/nvfp4-native'
    VCVARS=str(args.vcvars.resolve());NATIVE_VCVARS=str(args.native_vcvars.resolve());NATIVE_RECIPE=args.native_recipe.resolve()
    include_cpu = args.qsa_integration or args.calibrated_decode or (OUT / 'ggml-cpu.lib').is_file()
    libraries = [r'ggml\src\ggml-base.lib', r'ggml\src\ggml-cuda\ggml-cuda.lib']
    if include_cpu: libraries.append(r'ggml\src\ggml-cpu.lib')
    selected_names = ({'ggml-cuda.cu.obj','ops.cpp.obj'} if args.calibrated_decode else
        {'ggml-cuda.cu.obj','qfn-qsa-attn.cu.obj','ops.cpp.obj'} if args.qsa_integration else
        {'ggml-cuda.cu.obj','qfn-qsa-attn.cu.obj'})
    OUT.mkdir(parents=True, exist_ok=True)
    for key in ('TEMP', 'TMP', 'TMPDIR', 'CUDA_CACHE_PATH'):
        os.environ[key] = str(ROOT / 'tmp')
    # Bind the actual native/header closure and adjacent DLL as well as the 46 backend TUs below.
    native_dir = STACK / 'engines/streaming-llama/prefill_llama.cpp/tools/qfn-engine'
    extra_sources = [p for p in native_dir.glob('qfn-*') if p.suffix in ('.cpp', '.h')]
    extra_sources += [STACK / 'engines/streaming-llama/prefill_llama.cpp' / relative for relative in (
        'src/llama-vocab.cpp', 'src/llama-vocab.h', 'src/llama-weight-fetcher.cpp', 'src/llama-weight-fetcher.h',
        'ggml/include/ggml-cuda.h', 'ggml/include/ggml-qfn-service.h', 'ggml/include/ggml-qfn-nvfp4.h',
        'ggml/src/ggml-cuda/qfn-nvfp4.cuh', 'ggml/src/ggml-cuda/qfn-index-fp4.cuh',
        'ggml/src/ggml-cuda/qfn-qsa-score.cuh', 'ggml/src/ggml-cuda/qfn-index-refine.cuh',
        'ggml/include/ggml-qfn-qsa-score.h', 'ggml/include/ggml-qfn-expert-reduce.h',
        'ggml/src/ggml-cuda/qfn-expert-reduce.cuh')]
    extra_sources += [STACK / 'engines/streaming-llama/flashinfer-sm120/qfn_packed_vector.cuh']
    extra_sources += [Path(__file__).resolve(), NATIVE_RECIPE]
    extra_hashes = {str(p): digest(p) for p in extra_sources}
    dll = STACK / 'bin/qfn-basalt-nvfp4.dll'
    dll_hash = digest(dll)
    commands = subprocess.check_output([NINJA, '-C', str(OLD), '-t', 'commands', 'qfn-engine'], text=True).splitlines()
    ninja = (OLD / 'build.ninja').read_text()
    old_stack = native_path(OLD.parent / 'stack')
    new_stack = native_path(STACK)
    # The standalone provider is linked directly, not through llama.lib. Rebuild
    # it from the private source; never silently retain the old provider object.
    provider_commands=[line for line in commands if ' -c ' in line and
        line.split(' -c ',1)[1].endswith('llama-weight-fetcher.cpp') and '-DWF_STATIC' in line
        and 'qfn-engine.dir' in line]
    if len(provider_commands)!=1: raise RuntimeError('Missing exact QFN static provider compile rule')
    provider_command=provider_commands[0]
    provider_old=re.search(r' /Fo(\S+)',provider_command)[1]
    provider_object=OUT/'llama-weight-fetcher.cpp.obj'
    provider_command=provider_command.replace(old_stack,new_stack)
    provider_command=re.sub(r' /Fo\S+',lambda _: ' /Fo'+native_path(provider_object),provider_command)
    provider_command=re.sub(r' /Fd\S+',lambda _: ' /Fd'+native_path(OUT/'qfn-provider.pdb'),provider_command)
    jobs, replacements = [], {}
    source_hashes = {}
    for command in commands:
        match = re.search(r' -c (\S+)', command)
        if not match:
            continue
        source = match[1]
        name = source.replace('\\', '/').rsplit('/', 1)[-1]
        if name not in ('ggml.c', 'ggml-cuda.cu', 'qfn-qsa-attn.cu', 'mmq.cu', 'mmq_id.cu') and not name.startswith('mmq-instance-') and not (include_cpu and name == 'ops.cpp'):
            continue
        old_output = re.search(r'(?: -o | /Fo)(\S+)', command)
        if not old_output:
            raise RuntimeError('Cannot identify compiler output: ' + name)
        output = OUT / (name + '.obj')
        replacements[old_output[1]] = output
        command = command.replace(old_stack, new_stack)
        command = re.sub(r' -o \S+', ' -o ' + native_path(output).replace('\\', r'\\'), command)
        command = re.sub(r' /Fo\S+', ' /Fo' + native_path(output).replace('\\', r'\\'), command)
        command = re.sub(r' /Fd\S+', ' /Fd' + native_path(OUT / 'native.pdb').replace('\\', r'\\'), command)
        command = re.sub(r' -MT \S+', ' -MT ' + native_path(output).replace('\\', r'\\'), command)
        command = re.sub(r' -MF \S+', ' -MF ' + native_path(OUT / (name + '.d')).replace('\\', r'\\'), command)
        source_path = Path(source.replace(old_stack,new_stack))
        source_hashes[str(source_path)] = digest(source_path)
        jobs.append((name.replace('.', '_'), command))
    if not {'ggml.c', 'ggml-cuda.cu', 'qfn-qsa-attn.cu', 'mmq.cu', 'mmq_id.cu'} <= {
            label.replace('_cu', '.cu').replace('_c', '.c') for label, _ in jobs}:
        raise RuntimeError('Missing a mandatory changed backend/base translation unit')

    archives = {}
    input_hashes = {}
    original_object_hashes = {}
    for library in libraries:
        rule = next(line for line in ninja.splitlines() if line.startswith('build ' + library + ': '))
        objects = rule.split(': ', 1)[1].split(' ||', 1)[0].split()[1:]
        if not objects or any(not item.endswith('.obj') for item in objects):
            raise RuntimeError('Unexpected archive object rule for ' + library)
        old_library = OLD / library
        input_hashes[str(old_library)] = digest(old_library)
        response = OUT / (old_library.stem + '.rsp')
        response.write_text('\n'.join('"' + native_path(replacements.get(obj, OLD / obj)) + '"' for obj in objects))
        for obj in objects:
            if obj not in replacements:
                original_object_hashes[str(OLD / obj)] = digest(OLD / obj)
        original = next(line for line in commands if 'lib.exe' in line and '/out:' + library in line)
        librarian = re.search(r'(\S+lib\.exe)', original)[1]
        target = OUT / old_library.name
        archives[library] = target
        archives[library + ':command'] = librarian + ' /nologo /machine:x64 /out:"' + native_path(target) + '" @"' + native_path(response) + '"'
    (OUT / 'commands.json').write_text(json.dumps(jobs, indent=2))
    accepted_hashes = {}
    if include_cpu and not any(obj.name=='ops.cpp.obj' for obj in replacements.values()):
        raise RuntimeError('CPU ops.cpp command missing from QSA closure')
    if args.backend_core or args.qsa_integration or args.calibrated_decode or args.relink_only:
        for obj in replacements.values():
            selected = not args.relink_only and obj.name in selected_names
            if selected: continue
            if not obj.is_file() or not obj.stat().st_size:
                raise RuntimeError('Missing accepted unchanged closure member: '+str(obj))
            accepted_hashes[str(obj)] = digest(obj)
        expected = len(jobs) if args.relink_only else len(jobs)-len(selected_names)
        if len(accepted_hashes)!=expected:
            raise RuntimeError('Incomplete hash-bound accepted unchanged closure')
    if args.relink_only:
        for obj in replacements.values():
            if not obj.is_file() or not obj.stat().st_size:
                raise RuntimeError('Missing accepted CUDA/base object: ' + str(obj))
        for library in libraries:
            if not archives[library].is_file():
                raise RuntimeError('Missing accepted local archive: ' + str(archives[library]))
        print(f'Reusing {len(jobs)} completed CUDA/base objects and local archives; native TUs/link only', flush=True)
    else:
        selected_labels={name[:-4].replace('.', '_') for name in selected_names}
        selected=[item for item in jobs if item[0] in selected_labels] if (args.backend_core or args.qsa_integration or args.calibrated_decode) else jobs
        if (args.backend_core or args.qsa_integration or args.calibrated_decode) and len(selected)!=len(selected_names):
            raise RuntimeError('Missing exact selected translation-unit rebuild commands')
        print(f'Rebuilding {len(selected)} CUDA/base objects; maximum two compilers; existing archives read-only', flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda item: run(*item), selected))
        for source, before in accepted_hashes.items():
            if digest(Path(source)) != before:
                raise RuntimeError('Accepted unchanged object mutated during selective rebuild: ' + source)
        for library in libraries:
            run(Path(library).stem + '-archive', archives[library + ':command'])

    # Existing integrated five-TU recipe is the authoritative SDK/library configuration.
    # Replace only the base/CUDA archive inputs with the local rebuilt closures.
    run('qfn-weight-fetcher',provider_command,NATIVE_VCVARS)
    native = (NATIVE_RECIPE).read_text()
    provider_original=native_path(OLD / provider_old)
    if provider_original not in native: raise RuntimeError('Provider direct object absent from native link')
    native=native.replace(provider_original,native_path(provider_object))
    for library in libraries:
        native = native.replace(native_path(OLD / library), native_path(archives[library]))
    batch = OUT / 'integrated-five-tu.cmd'
    batch.write_text(native)
    run('qfn-native', 'cmd /d /c "' + native_path(batch) + '"', NATIVE_VCVARS)
    for source, before in input_hashes.items():
        if digest(Path(source)) != before:
            raise RuntimeError('Read-only input archive changed: ' + source)
    for source, before in accepted_hashes.items():
        if digest(Path(source)) != before:
            raise RuntimeError('Accepted unchanged object mutated during native relink: ' + source)
    for source,before in original_object_hashes.items():
        if digest(Path(source))!=before: raise RuntimeError('Read-only input object changed: '+source)
    for source,before in source_hashes.items():
        if digest(Path(source))!=before: raise RuntimeError('Input translation unit changed during build: '+source)
    for source,before in extra_hashes.items():
        if digest(Path(source))!=before: raise RuntimeError('Native/header/recipe source changed during build: '+source)
    if digest(dll)!=dll_hash: raise RuntimeError('Adjacent calibrated DLL changed during native build')
    executable = STACK / 'bin/qfn-engine.exe'
    (OUT / 'result.json').write_text(json.dumps(dict(exe=str(executable), exe_sha256=digest(executable),
        rebuilt_objects=0 if args.relink_only else len(selected), accepted_objects=len(jobs),
        verified_unchanged_objects=accepted_hashes, readonly_input_objects=original_object_hashes, source_sha256=source_hashes,
        output_object_sha256={str(p):digest(p) for p in replacements.values()},
        native_header_recipe_sha256=extra_hashes, unchanged_dll_sha256=dll_hash,
        native_object_sha256={str(OUT/name):digest(OUT/name) for name in (
            'llama-vocab.cpp.obj','qfn-main.cpp.obj','qfn-serve.cpp.obj','qfn-engine.cpp.obj',
            'qfn-pack.cpp.obj','qfn-tokenizer.cpp.obj','qfn-op-test.cpp.obj','llama-weight-fetcher.cpp.obj')},
        local_archive_sha256={str(path):digest(path) for name,path in archives.items() if ":command" not in name},
        cuda_toolchain=VCVARS, native_toolchain=NATIVE_VCVARS,
        op_test_sha256=digest(STACK / 'bin/qfn-op-test.exe'), unchanged_input_archives=input_hashes,
        local_archives={name:str(path) for name, path in archives.items() if ':command' not in name}), indent=2))
    print('Integrated executable built: ' + str(executable), flush=True)


if __name__ == '__main__':
    main()
