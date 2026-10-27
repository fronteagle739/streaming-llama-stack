"""Build against the installed MLX binary, smoke-test, then optionally install.

Run by path using the serving Python. No changes to that environment's pip
packages; matching nanobind headers live in a unique build directory.
"""
import argparse
import hashlib
import importlib.util
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile


def abi_tag(path):
    tags = set(re.findall(rb'v\d+_system_libcpp_abi\d+', Path(path).read_bytes()))
    if len(tags) != 1:
        raise RuntimeError(f"Cannot unambiguously identify nanobind ABI in {path}: {tags}")
    return tags.pop().decode()


def installed_mlx():
    distribution=metadata.distribution('mlx')
    cores=[Path(distribution.locate_file(f)) for f in distribution.files or []
           if str(f).startswith('mlx/core.') and str(f).endswith('.so')]
    if len(cores)!=1:raise RuntimeError('Cannot identify the installed MLX core binary')
    cmake=cores[0].parent/'share/cmake/MLX'
    if not (cmake/'MLXConfig.cmake').is_file():raise RuntimeError('Installed MLX Metal build inputs are missing')
    return cores[0],cmake


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path)
    parser.add_argument('--install-dir', type=Path, help='explicit target package directory')
    parser.add_argument('--fetch-build-deps', action='store_true')
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--wheelhouse', type=Path)
    parser.add_argument('--skip-smoke', action='store_true',help='Compile only; mark GPU correctness verification pending')
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        raise RuntimeError('Build requires macOS and the Metal compiler toolchain')
    mlx_binary,mlx_cmake=installed_mlx()
    src = Path(__file__).resolve().parent
    build = (args.build_dir or Path(tempfile.mkdtemp(prefix='glm-native-'))).resolve()
    build.mkdir(parents=True, exist_ok=True)
    stage = build/'stage'; stage.mkdir(exist_ok=True)
    tag = abi_tag(mlx_binary)
    version = {'19':'2.12.0', '20':'2.13.0', '21':'2.15.0'}.get(tag.split('_')[0][1:])
    if version is None:
        raise RuntimeError(f'Unknown MLX nanobind ABI {tag}; select matching headers explicitly')
    if args.fetch_build_deps:
        deps = build/('deps-'+version)
        if not (deps/'nanobind/cmake').is_dir():
            offline=['--no-index','--find-links',str(args.wheelhouse.resolve())] if args.wheelhouse else []
            subprocess.run([sys.executable,'-m','pip','install',*offline,'--no-deps','--target',str(deps),
                            f'nanobind=={version}'],check=True)
        nb_root=deps/'nanobind/cmake'
    else:
        import importlib.metadata
        installed=importlib.metadata.version('nanobind')
        if installed != version:
            raise RuntimeError(f'MLX has {tag}; need nanobind {version}, found {installed}. '
                               'Use --fetch-build-deps for isolated headers.')
        nb_root=Path(subprocess.check_output([sys.executable,'-m','nanobind','--cmake_dir'],text=True).strip())
    subprocess.run(['xcrun','-sdk','macosx','metal','--version'],check=True)
    cmake=shutil.which('cmake') or str(Path(sys.executable).parent/'cmake')
    subprocess.run([cmake,'-S',str(src/'csrc'),'-B',str(build/'cmake'),
                    f'-DPython_EXECUTABLE={sys.executable}',f'-Dnanobind_ROOT={nb_root}',
                    f'-Dnanobind_DIR={nb_root}',
                    f'-DMLX_ROOT={mlx_cmake}',
                    '-DCMAKE_BUILD_TYPE=Release',f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={stage}'],check=True)
    subprocess.run([cmake,'--build',str(build/'cmake'),'-j',str(args.jobs)],check=True)
    ext=next(stage.glob('_ext*.so'))
    if abi_tag(ext) != tag:
        raise RuntimeError(f'Built ABI {abi_tag(ext)} differs from MLX {tag}')
    if not args.skip_smoke:subprocess.run([sys.executable,str(src/'smoke.py'),str(stage)],check=True)
    artifacts=[ext,stage/'libomlx_glm_kernel_ops.dylib',stage/'omlx_glm_kernels.metallib',
               src/'fast.py',src/'smoke.py']
    manifest=dict(python=sys.executable,mlx_binary=str(mlx_binary),mlx_abi=tag,nanobind=version,smoke_verified=not args.skip_smoke,
                  artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts})
    (stage/'native-build.json').write_text(json.dumps(manifest,indent=2))
    if args.install_dir:
        dest=args.install_dir.resolve()
        if not (dest/'fast.py').is_file() or not (dest/'csrc').is_dir():
            raise RuntimeError(f'Not a GLM native package: {dest}')
        backup=Path(tempfile.mkdtemp(prefix='previous-install-',dir=build))
        # Preserve prior artifacts. Never replace a loaded dylib in-place.
        for p in artifacts+[stage/'native-build.json']:
            target=dest/p.name
            if target.exists():
                shutil.copy2(target,backup/p.name)
            temporary=dest/(p.name+'.new')
            shutil.copy2(p,temporary)
            temporary.replace(target)
        if not args.skip_smoke:subprocess.run([sys.executable,str(src/'smoke.py'),str(dest)],check=True)
        manifest['installed_to']=str(dest)
        manifest['previous_install']=str(backup)
    print(json.dumps(manifest,indent=2))


if __name__=='__main__': main()
