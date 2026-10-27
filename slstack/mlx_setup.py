"""Install the retained MLX fork and its native kernels in an isolated runtime."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from .config import ROOT
from .journal import atomic_json
from .process import OwnedProcess

PACKAGE = ROOT / 'vendor/mac-mlx'
LOCK = ROOT / 'locks/mac-mlx-py312.txt'


def source_files():
    manifest=json.loads((PACKAGE/'PROVENANCE.json').read_text())
    names=[r['path'] for r in manifest['files']]
    names+=['pyproject.toml','requirements.txt','LICENSE','README.md','PROVENANCE.json']
    return [(name,(PACKAGE/name).read_bytes().replace(b'\r\n',b'\n')) for name in sorted(set(names))]


def source_digest():
    digest=hashlib.sha256(LOCK.read_bytes().replace(b'\r\n',b'\n'))
    for name,data in source_files():
        digest.update(name.encode()+b'\0'+hashlib.sha256(data).digest())
    return digest.hexdigest()


def inspect_installation(cfg):
    """Check saved build/smoke receipts without importing a GPU runtime."""
    venv=Path(cfg['runtime_root'])/'decode-venv';python=venv/'bin/python'
    marker=venv/'slstack-source.json'
    previous=json.loads(marker.read_text()) if marker.is_file() else {}
    current=python.is_file() and previous.get('source_sha256')==source_digest()
    manifests=list(venv.glob('lib/python*/site-packages/mlx_vlm/custom_kernels/glm_moe_dsa/native-build.json'))
    built=False;verified=False
    if len(manifests)==1:
        manifest=manifests[0];record=json.loads(manifest.read_text())
        artifacts=record.get('artifacts',{})
        def matches(name,sha):
            file=manifest.parent/name
            return file.is_file() and hashlib.sha256(file.read_bytes()).hexdigest()==sha
        built=bool(current and artifacts and all(matches(name,sha) for name,sha in artifacts.items()))
        verified=built and previous.get('native_glm') and record.get('smoke_verified')
    return dict(python=str(python),source_current=bool(current),built=bool(built),ready=bool(verified),
                verification_pending=not bool(verified))


def setup(cfg, *, check=False, native_glm=True, wheelhouse=None, build_only=False):
    if sys.platform!='darwin' or platform.machine()!='arm64':
        raise ValueError('MLX decode setup requires Apple Silicon macOS')
    if sys.version_info<(3,12):raise ValueError('Run setup with Python 3.12 or newer')
    runtime=Path(cfg['runtime_root']).resolve();runtime.mkdir(parents=True,exist_ok=True)
    offline=[]
    if wheelhouse:
        wheelhouse=Path(wheelhouse).expanduser().resolve()
        if not wheelhouse.is_dir():raise ValueError('wheelhouse must be an existing directory')
        offline=['--no-index','--find-links',str(wheelhouse)]
    python=runtime/'decode-venv/bin/python';revision=source_digest()
    marker=runtime/'decode-venv/slstack-source.json'
    previous=json.loads(marker.read_text()) if marker.is_file() else {}
    current=python.is_file() and previous.get('source_sha256')==revision
    already_native=bool(current and previous.get('native_glm'))
    if not check:
        if shutil.disk_usage(runtime).free<12*1024**3:
            raise OSError('MLX setup requires 12 GiB free for source, environment, build/cache and headroom')
        launch={**cfg,'host_budget_gib':2,'min_available_gib':max(4,cfg.get('min_available_gib',4))}
        def run(argv,name):
            job=OwnedProcess(launch,argv,name,cwd=ROOT,timeout=1800,scope='mac-mlx')
            job.wait()
            return job.log.read_text(errors='replace')
        if not python.is_file():run([sys.executable,'-B','-m','venv',str(python.parent.parent)],'setup-mlx-venv')
        if not current:
            # Build tools may write egg-info beside their input. Stage a small,
            # content-addressed snapshot so all such outputs remain in runtime.
            snapshot=runtime/'sources'/('mlx-vlm-'+revision[:16])
            for name,data in source_files():
                target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    if target.read_bytes()!=data:raise ValueError('Runtime source snapshot diverged: '+name)
                else:
                    with target.open('xb') as stream:stream.write(data)
            run([str(python),'-B','-m','pip','install','--disable-pip-version-check',*offline,'-r',str(LOCK),'setuptools>=61','wheel'],'setup-mlx-dependencies')
            run([str(python),'-B','-m','pip','install','--disable-pip-version-check',*offline,'--no-build-isolation','--no-deps',str(snapshot)],'setup-mlx-fork')
        if native_glm and (not current or not previous.get('native_glm')):
            run([str(python),'-B','-m','pip','install',*offline,'cmake>=3.27,<5'],'setup-mlx-build-tools')
            # Installed package data includes every C++/Metal/header/CMake input.
            # No compiled artifact is installed into the tracked source tree.
            builder='''import importlib.metadata as m,runpy,sys
p=m.distribution('mlx-vlm').locate_file('mlx_vlm/custom_kernels/glm_moe_dsa')
wheelhouse=sys.argv[2:]
sys.argv=[str(p/'build.py'),'--build-dir',sys.argv[1],'--jobs','2','--fetch-build-deps','--install-dir',str(p)]
if wheelhouse:sys.argv.extend(['--wheelhouse',wheelhouse[0]])
runpy.run_path(str(p/'build.py'),run_name='__main__')
'''
            if build_only:builder=builder.replace("runpy.run_path(","sys.argv.append('--skip-smoke')\nrunpy.run_path(")
            run([str(python),'-B','-c',builder,str(runtime/'build/mlx-glm-native'),*([str(wheelhouse)] if wheelhouse else [])],'setup-mlx-glm-native')
        run([str(python),'-B','-m','pip','check'],'setup-mlx-dependency-check')
        current=True
    result=dict(ready=False,python=str(python),source_current=bool(current),source_sha256=revision,native_glm=False)
    if not python.is_file():return result
    if build_only:
        if not check:atomic_json(marker,dict(source_sha256=revision,native_glm=already_native,built=True))
        result.update(built=bool(current),verification_pending=True)
        atomic_json(runtime/'state/mlx-backend-check.json',result)
        return result
    # The loader path and native availability are verified by the serving Python,
    # rather than an unrelated system environment or source-tree import.
    probe='''import importlib.metadata as m,json
from pathlib import Path
from mlx_vlm.models.glm_moe_dsa import glm_moe_dsa as g
from mlx_vlm.custom_kernels.glm_moe_dsa import fast
print('SLSTACK_MLX_PROBE='+json.dumps(dict(version=m.version('mlx-vlm'),legacy_indexer=g._use_legacy_indexer('0.30.7'),current_indexer=not g._use_legacy_indexer('0.32.0'),native=fast.is_native_available(),package=str(Path(g.__file__).resolve()))))
'''
    launch={**cfg,'host_budget_gib':2,'min_available_gib':max(4,cfg.get('min_available_gib',4))}
    job=OwnedProcess(launch,[str(python),'-B','-c',probe],'check-mlx-fork',cwd=runtime,timeout=120,scope='mac-mlx')
    job.wait()
    lines=[line.split('=',1)[1] for line in job.log.read_text(errors='replace').splitlines() if line.startswith('SLSTACK_MLX_PROBE=')]
    actual=json.loads(lines[-1]);package=Path(actual['package'])
    result['native_glm']=actual['native'];result['probe']=actual
    inside=package.is_relative_to(python.parent.parent)
    result['ready']=bool(current and inside and actual['legacy_indexer'] and actual['current_indexer'] and (not native_glm or (actual['native'] and (not check or already_native))))
    if not check and result['ready']:
        atomic_json(marker,dict(source_sha256=revision,native_glm=bool(actual['native']) or already_native))
    atomic_json(runtime/'state/mlx-backend-check.json',result)
    return result
