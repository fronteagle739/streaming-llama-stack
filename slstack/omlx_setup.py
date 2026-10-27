"""Isolated oMLX profile with its own MLX ABI and native kernels."""
import json
from pathlib import Path
import re
import sys
from .config import ROOT

VERSIONS={'mlx-lm':'0.32.3.dev0+stack.0d4d3d04','mlx-embeddings':'0.1.0','mlx-vlm':'0.7.10','mlx-audio':'0.5.7'}


def portable_metadata(files):
    result=[]
    for name,data in files:
        if name=='pyproject.toml':
            text=data.decode()
            pattern=re.compile(r'([a-zA-Z0-9_-]+)(\[[^\]\s]+\])? @ git\+https://github\.com/[^"\s]+')
            text=pattern.sub(lambda m:m[1]+(m[2] or '')+'=='+VERSIONS[m[1]] if m[1] in VERSIONS else m[0],text)
            if '@ git+' in text:raise ValueError('oMLX introduced an unpinned Git dependency')
            data=text.encode()
        result.append((name,data))
    return result


def install(build,*,build_only=False):
    if sys.version_info[:2] not in ((3,12),(3,13)):raise ValueError('The pinned oMLX profile requires Python 3.12 or 3.13')
    python=build.runtime/'omlx-venv/bin/python'
    if not python.is_file():build.run([sys.executable,'-B','-m','venv',python.parent.parent],'omlx-venv')
    build.run([python,'-B','-m','pip','install','--disable-pip-version-check',*build.pip_options,
               '-r',ROOT/'locks/mac-omlx-py312.txt'],'omlx-dependencies')
    # Installed native kernels must use MLX 0.32.4 / nanobind 3.1.0. They never
    # share a site-packages directory with the customised MLX-VLM 0.32.2 profile.
    for name,version in VERSIONS.items():
        source,revision=build.stage(build.source('omlx-'+name),'omlx-'+name,transform=portable_metadata)
        wrapper='''import os,runpy,sys
version=sys.argv.pop(1)
if version:os.environ['SETUPTOOLS_SCM_PRETEND_VERSION']=version
os.environ['PATH']=os.path.dirname(sys.executable)+os.pathsep+os.environ.get('PATH','')
sys.argv=['pip','install',*sys.argv[1:]]
runpy.run_module('pip',run_name='__main__')
'''
        build.run([python,'-B','-c',wrapper,version if name=='mlx-lm' else '',
                   *build.pip_options,'--no-deps','--no-build-isolation',source],'omlx-'+name)
    source,revision=build.stage(build.source('omlx'),'omlx',transform=portable_metadata)
    wrapper='''import os,runpy,sys
jobs=sys.argv.pop(1)
os.environ.update(OMLX_WITH_CUSTOM_KERNEL='1',CMAKE_BUILD_PARALLEL_LEVEL=jobs,MAX_JOBS=jobs)
os.environ['PATH']=os.path.dirname(sys.executable)+os.pathsep+os.environ.get('PATH','')
sys.argv=['pip','install',*sys.argv[1:]]
runpy.run_module('pip',run_name='__main__')
'''
    build.run([python,'-B','-c',wrapper,str(build.jobs),*build.pip_options,'--no-deps','--no-build-isolation',source],'omlx-native-build')
    build.run([python,'-B','-m','pip','check'],'omlx-dependency-check')
    binaries=[python.parent/'omlx']
    site=python.parent.parent/('lib/python'+str(sys.version_info.major)+'.'+str(sys.version_info.minor)+'/site-packages/omlx/custom_kernels')
    binaries.extend(p for p in site.rglob('*') if p.suffix in {'.so','.dylib','.metallib'})
    if len(binaries)<4:raise RuntimeError('oMLX installed without its native kernel artifacts')
    if not build_only:
        probe='''import json
from omlx.custom_kernels import native_kernel_status
status=native_kernel_status()
bad={k:v.get('import_error') for k,v in status.items() if not v.get('available')}
print(json.dumps(dict(native_modules=len(status),failed=bad)))
raise SystemExit(bool(bad))
'''
        build.run([python,'-B','-c',probe],'omlx-native-check')
    return build.report('omlx',revision,binaries,binary=str(python.parent/'omlx'),python=str(python),mlx_abi='0.32.4 / nanobind 3.1.0')
