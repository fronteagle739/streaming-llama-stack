"""Run a retained decoder under the stack's process-tree memory guardian."""
from __future__ import annotations
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .backends import backend, commands
from .config import ROOT, environment
from .journal import atomic_json


def settings(cfg, model=None, selected=None):
    result=dict(cfg.get('decode') or (cfg.get('general') or {}).get('mac') or {})
    if model:result['model_path']=model
    if selected:result['backend']=selected
    if not result.get('model_path'):raise ValueError('Set decode.model_path in the local config or pass --model')
    result['model_path']=str(Path(result['model_path']).expanduser().resolve())
    return result


def scope_for(name):
    if name in ('mlx-receiver','mlx-lm','mlx-vlm','mtp-shim','omlx'):return 'mac-mlx'
    return {'mlx-serve':'mlx-serve','mference':'mac-mference','ds4':'ds4-mac','ds4-peer':'ds4-mac'}.get(name,'mac-llama')


def inspect(cfg, model=None, selected=None):
    local=settings(cfg,model,selected);spec=backend(local.get('backend','mlx-receiver'))
    missing=[]
    try:argv=commands(local,runtime=cfg['runtime_root'],write_settings=False)
    except (OSError,ValueError) as exc:argv=[];missing.append(str(exc))
    if not Path(local['model_path']).exists():missing.append('decode model_path does not exist')
    for command in argv:
        if not Path(command[0]).is_file() and not shutil.which(command[0]):missing.append('Missing backend executable: '+command[0])
    managed=(Path(cfg['runtime_root'])/'state/backends'/(spec.name+'.json')).is_file()
    managed=managed or (spec.setup=='mlx' and (Path(cfg['runtime_root'])/'decode-venv/slstack-source.json').is_file())
    if managed:
        from .backend_setup import check as installed
        if not installed(cfg,spec.name).get('ready'):missing.append('Backend installation is unverified or changed; run setup-backend --backend '+spec.name)
    return dict(backend=spec.name,transport=spec.transport,ready=not missing,missing=missing)


def run(cfg, *, model=None, selected=None, check=False):
    local=settings(cfg,model,selected);spec=backend(local.get('backend','mlx-receiver'))
    result=inspect(cfg,model,selected)
    if check:return result
    if not result['ready']:raise ValueError('; '.join(result['missing']))
    launch={**cfg,'decode':local}
    if sys.platform=='darwin':
        from .mac import budget
        launch=budget(launch,400)
    effective=Path(cfg['runtime_root'])/'decode-effective.local.json'
    atomic_json(effective,launch)
    from .process import OwnedProcess
    job=OwnedProcess(launch,[sys.executable,'-B','-m','slstack','--config',str(effective),'decode-internal'],
                     'decode-server',scope=scope_for(spec.name),unbounded=True)
    print('Decoder starting; log: '+str(job.log),flush=True)
    try:job.wait()
    except KeyboardInterrupt:pass
    finally:
        if job.proc.poll() is None:job.stop()
    return result


def serve(cfg):
    """All children share one outer guardian and one admitted host budget."""
    local=settings(cfg);spec=backend(local.get('backend','mlx-receiver'))
    env=environment(cfg,scope_for(spec.name));children=[]
    try:
        for command in commands(local,runtime=cfg['runtime_root']):
            children.append(subprocess.Popen(command,cwd=ROOT,env=env))
        while True:
            for child in children:
                code=child.poll()
                if code is not None:raise RuntimeError('Decoder child exited with code '+str(code))
            time.sleep(.25)
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
