"""Reproducible backend builds in private runtime directories."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
from .backend_cache import Cache,digest,snapshot,tree_files
from .backends import BACKENDS,backend
from .config import ROOT
from .journal import atomic_json
from .process import OwnedProcess


class Build:
    def __init__(self,cfg,*,cache_dir=None,offline=False,wheelhouse=None,jobs=2):
        if jobs not in (1,2):raise ValueError('Backend builds support one or two compiler jobs')
        self.cfg=cfg;self.runtime=Path(cfg['runtime_root']);self.runtime.mkdir(parents=True,exist_ok=True)
        self.cache=Cache(cfg,directory=cache_dir,offline=offline);self.offline=offline;self.jobs=jobs
        self.wheelhouse=Path(wheelhouse).expanduser().resolve() if wheelhouse else None
        if self.wheelhouse and not self.wheelhouse.is_dir():raise ValueError('wheelhouse must be a directory')
        self.pip_options=['--no-index','--find-links',str(self.wheelhouse)] if self.wheelhouse else []
        if offline and not self.wheelhouse:raise ValueError('Offline builds require a wheelhouse for Python build tools')
        self.launch={**cfg,'host_budget_gib':cfg.get('backend_build_budget_gib',3),
                     'min_available_gib':max(4,cfg.get('min_available_gib',4))}
        self.python=self.runtime/'backend-build-venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        self.steps=[];self.retained_inputs={}

    def run(self,argv,name,*,cwd=ROOT):
        job=OwnedProcess(self.launch,[str(v) for v in argv],'backend-'+name,cwd=cwd,timeout=3600)
        self.steps.append(dict(name=name,log=str(job.log),receipt=str(job.receipt)))
        job.wait();return job.log.read_text(errors='replace')

    def tools(self):
        if not self.python.is_file():self.run([sys.executable,'-B','-m','venv',self.python.parent.parent],'build-venv')
        cmake=self.python.with_name('cmake.exe' if os.name=='nt' else 'cmake')
        if not cmake.is_file():
            self.run([self.python,'-B','-m','pip','install','--disable-pip-version-check',*self.pip_options,
                      'cmake==4.4.4','setuptools==84.2.1','wheel==0.48.0'],'build-tools')
        return str(cmake)

    def source(self,name):
        return self.cache.unpack(name,self.runtime/'downloads/unpacked')

    def stage(self,source,label,*,transform=None):
        source=Path(source)
        if any(source.is_relative_to(ROOT/name) for name in ('engines','vendor')):
            from .backend_cache import fingerprint
            self.retained_inputs[source.relative_to(ROOT).as_posix()]=fingerprint(tree_files(source))
        recipe=hashlib.sha256()
        for file in ('backend_setup.py','backend_cache.py','mlx_serve_setup.py','omlx_setup.py'):
            recipe.update((ROOT/'slstack'/file).read_bytes().replace(b'\r\n',b'\n'))
        receipt=json.dumps(dict(recipe_sha256=recipe.hexdigest(),platform=sys.platform,
                               machine=platform.machine(),python=list(sys.version_info[:2]))).encode()
        def stage_files(files):
            return [*(transform(files) if transform else files),('STACK_BUILD_RECIPE.json',receipt)]
        return snapshot(source,self.runtime,label,transform=stage_files)

    def cmake(self,source,build,options,targets=None,*,install=False):
        cmake=self.tools()
        self.run([cmake,'-S',source,'-B',build,'-DCMAKE_BUILD_TYPE=Release',*options],build.name+'-configure')
        command=[cmake,'--build',build,'--config','Release','--parallel',self.jobs]
        if targets:command+=['--target',*targets]
        self.run(command,build.name+'-compile')
        if install:self.run([cmake,'--install',build,'--config','Release'],build.name+'-install')

    def report(self,name,revision,binaries,**extra):
        missing=[str(p) for p in binaries if not Path(p).is_file()]
        if missing:raise RuntimeError('Backend build outputs are missing: '+', '.join(missing))
        result=dict(backend=name,built=True,ready=False,verification_pending=True,source_sha256=revision,
                    binaries={str(p):digest(p) for p in binaries},steps=self.steps,
                    retained_inputs=self.retained_inputs,**extra)
        atomic_json(self.runtime/'state/backends'/(name+'.json'),result)
        return result


def upstream_source(build):
    overlay=ROOT/'engines/upstream-decode-receiver'
    def transform(files):
        result=[]
        for name,data in files:
            if name=='tools/CMakeLists.txt':
                data+=b'\nadd_subdirectory("${SLSTACK_UPSTREAM_RECEIVER_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/stack-receiver")\n'
            result.append((name,data))
        return result
    source,revision=build.stage(build.source('llama-upstream'),'llama-upstream',transform=transform)
    return source,revision,['-DSLSTACK_UPSTREAM_RECEIVER_DIR='+str(overlay),
                            '-DCOSMOS_UPSTREAM_RECEIVER_SOURCE='+str(ROOT/'engines/streaming-llama/prefill_llama.cpp')]


def native_llama(build,name):
    if name=='llama-decode-upstream':source,revision,extra=upstream_source(build)
    else:
        source,revision=build.stage(ROOT/'engines/streaming-llama/prefill_llama.cpp','streaming-llama');extra=[]
    output=build.runtime/'build'/('llama-'+revision[:16])
    options=['-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_BUILD_TOOLS=ON',
             '-DLLAMA_BUILD_SERVER=ON','-DLLAMA_CURL=OFF','-DLLAMA_OPENSSL=OFF','-DBUILD_SHARED_LIBS=OFF',
             '-DLLAMA_BUILD_DECODE_RECEIVER=ON',*extra]
    if sys.platform=='darwin':options+=['-DGGML_METAL=ON','-DGGML_METAL_EMBED_LIBRARY=ON','-DGGML_ACCELERATE=ON']
    else:raise ValueError('Native Mac decoder setup requires macOS; RTX producer setup is a separate backend')
    target='llama-upstream-decode-receiver' if name=='llama-decode-upstream' else 'llama-decode-receiver'
    targets=['llama-server',target]
    build.cmake(source,output,options,targets)
    binary=output/'bin'/('llama-server' if name=='llama-server' else target)
    return build.report(name,revision,[binary],binary=str(binary))


def native_ds4(build,name='ds4'):
    directory='ds4-peer' if name=='ds4-peer' else 'ds4-mac'
    source,revision=build.stage(ROOT/'engines'/directory,name+'-mac')
    flags='-O3 -ffast-math -g -mcpu=native -Wall -Wextra -std=c99 -DDS4_NATIVE_FRONTIER_ENABLE=1'
    build.run(['make','-j'+str(build.jobs),'ds4-server','CFLAGS='+flags],'ds4-compile',cwd=source)
    binary=source/'ds4-server'
    return build.report(name,revision,[binary],binary=str(binary),renderer_sha256=digest(ROOT/'engines'/directory/'ds4_server.c'))


def swift_manifest(text,urls):
    """Use pinned local packages and Foundation networking for local models."""
    optional={'swift-xet','async-http-client'}
    pattern=re.compile(r'\.(package|product)\(')
    result=[];start=0
    for match in pattern.finditer(text):
        if match.start()<start:continue
        pos=match.end();depth=1;quoted=False;escaped=False
        while pos<len(text) and depth:
            char=text[pos]
            if quoted:
                if escaped:escaped=False
                elif char=='\\':escaped=True
                elif char=='"':quoted=False
            elif char=='"':quoted=True
            elif char=='(':depth+=1
            elif char==')':depth-=1
            pos+=1
        if depth:raise ValueError('Unbalanced Swift package declaration')
        call=text[match.start():pos]
        url=re.search(r'url:\s*"([^"]+)"',call)
        package=re.search(r'package:\s*"([^"]+)"',call)
        replacement=call
        if match[1]=='package' and url:
            address=url[1].removesuffix('.git')
            identity=urls.get(address)
            if address.rsplit('/',1)[-1] in optional:replacement=''
            elif identity:replacement='.package(path: "../'+identity+'")'
            else:raise ValueError('Unpinned Swift dependency: '+address)
        elif match[1]=='product' and package and package[1] in optional:replacement=''
        if not replacement:
            while pos<len(text) and text[pos] in ' \t':pos+=1
            if pos<len(text) and text[pos]==',':pos+=1
        result.extend((text[start:match.start()],replacement));start=pos
    result.append(text[start:]);return ''.join(result)


def swift_dependencies(build):
    pins=json.loads((ROOT/'engines/mference/Package.resolved').read_text())['pins']
    roots={r['identity']:build.source('swift-'+r['identity']) for r in pins}
    urls={r['location'].removesuffix('.git'):r['identity'] for r in pins}
    def rewrite(files):
        # Select the pinned Swift 6.1 manifest where upstream provides one.
        # Older manifests enable optional binary transports unconditionally.
        versioned={n:d for n,d in files if n=='Package@swift-6.1.swift'}
        result=[]
        for name,data in files:
            if name.startswith('Package@swift-') and name.endswith('.swift'):continue
            if name=='Package.swift':
                data=swift_manifest(versioned.get('Package@swift-6.1.swift',data).decode(),urls).encode()
            result.append((name,data))
        return result
    # All packages are siblings so path dependencies resolve without network or
    # synthetic Git histories. Their bytes come from exact upstream revisions.
    combined=[]
    for identity,source in roots.items():combined.extend((identity+'/'+n,d) for n,d in rewrite(tree_files(source,exclude={'stack-source.json'})))
    combined.extend(('mference/'+n,d) for n,d in rewrite(tree_files(ROOT/'engines/mference')) if n!='Package.resolved')
    source,revision=build.stage(ROOT/'engines/mference','mference-graph',transform=lambda _:combined)
    return source/'mference',revision


def mference(build):
    source,revision=swift_dependencies(build)
    output=build.runtime/'build'/('mference-'+revision[:16])
    build.run(['swift','build','--package-path',source,'--scratch-path',output,'-c','release',
               '--product','MferenceServer','-j',build.jobs],'mference-compile')
    build.run(['swift','build','--package-path',source,'--scratch-path',output,'-c','release',
               '--product','MferenceDecodeService','-j',build.jobs],'mference-receiver-compile')
    binary=output/'release/MferenceServer'
    receiver=output/'release/MferenceDecodeService'
    resources=[p for bundle in (output/'release').glob('*.bundle') for p in bundle.rglob('*') if p.is_file()]
    return build.report('mference',revision,[binary,receiver,*resources],binary=str(binary),decode_service=str(receiver))


def inputs_for(name):
    if name=='mlx-serve':return ['mlx-serve','mlx-serve-ds4','mlx-serve-mlx','mlx-serve-mlxc','mlx-serve-opencode','mlx-serve-llama','webp','metal-cpp','mlx-json','mlx-fmt','mlx-gguflib']
    if name=='omlx':return ['omlx','omlx-mlx-lm','omlx-mlx-embeddings','omlx-mlx-vlm','omlx-mlx-audio']
    if name=='mference':return ['swift-'+r['identity'] for r in json.loads((ROOT/'engines/mference/Package.resolved').read_text())['pins']]
    return ['llama-upstream'] if name=='llama-decode-upstream' else []


def check(cfg,name):
    path=Path(cfg['runtime_root'])/'state/backends'/(name+'.json')
    if not path.is_file():return dict(backend=name,built=False,ready=False,reason='Run setup-backend for this backend')
    result=json.loads(path.read_text())
    if name in ('mlx-receiver','mlx-lm','mlx-vlm','mtp-shim'):
        from .mlx_setup import setup as mlx_setup
        verified=mlx_setup(cfg,check=True)
        return {**result,**verified,'backend':name,'built':bool(verified.get('built',verified.get('ready')))}
    from .backend_cache import fingerprint
    artifacts=result.get('binaries',{})
    result['built']=bool(artifacts) and all(Path(p).is_file() and digest(p)==sha for p,sha in artifacts.items())
    result['built']=result['built'] and all(fingerprint(tree_files(ROOT/p))==sha for p,sha in result.get('retained_inputs',{}).items())
    result['ready']=result['built'] and not result.get('verification_pending',True)
    return result


def setup(cfg,name,*,check_only=False,fetch_only=False,build_only=False,cache_dir=None,offline=False,wheelhouse=None,jobs=2):
    name=backend(name).name
    if check_only:return check(cfg,name)
    if fetch_only:
        cache=Cache(cfg,directory=cache_dir,offline=offline)
        return dict(backend=name,downloaded=[str(cache.get(item)) for item in inputs_for(name)],built=False,ready=False)
    if sys.platform!='darwin' or platform.machine()!='arm64':raise ValueError('Decode backend installation requires Apple Silicon macOS')
    volume=Path(cfg['runtime_root']).resolve()
    while not volume.exists():volume=volume.parent
    if shutil.disk_usage(volume).free<16*1024**3:raise OSError('Backend setup requires 16 GiB free for inputs, build outputs and safety headroom')
    build=Build({**cfg,'backend_build_budget_gib':cfg.get('backend_build_budget_gib',8 if name=='mlx-serve' else 3)},
                cache_dir=cache_dir,offline=offline,wheelhouse=wheelhouse,jobs=jobs)
    if name in ('mlx-receiver','mlx-lm','mlx-vlm','mtp-shim'):
        from .mlx_setup import setup as mlx_setup
        result=mlx_setup(cfg,wheelhouse=wheelhouse,build_only=build_only)
        result.update(backend=name,built=bool(result.get('built',result.get('ready'))))
        atomic_json(build.runtime/'state/backends'/(name+'.json'),result)
        return result
    if name in ('llama-decode','llama-decode-upstream','llama-server'):result=native_llama(build,name)
    elif name in ('ds4','ds4-peer'):result=native_ds4(build,name)
    elif name=='mference':result=mference(build)
    elif name=='mlx-serve':
        from .mlx_serve_setup import install
        result=install(build)
    else:
        from .omlx_setup import install
        result=install(build,build_only=build_only)
    if not build_only:
        binary=result.get('binary')
        if binary:build.run([binary,'--help'],name+'-cli-check')
        result.update(ready=True,verification_pending=False)
        atomic_json(build.runtime/'state/backends'/(name+'.json'),result)
    return result
