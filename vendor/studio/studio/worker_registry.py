# Derived from Studio backend/studio/worker_registry.py at 5b39c3abf34523c69fb761f7dd174c710d3a7eb1.
# Minimal export; comments/docstrings removed; machine-specific fallback paths cleared.
from __future__ import annotations
import json
import logging
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid
import sys
import psutil
_owner = None

def stop_model_services(*, exclude_pids=()) -> None:
    if _owner is None:
        return
    port, owner = _owner
    stopped = []
    for path in registry_dir(port).glob('*.json'):
        try:
            row = json.loads(path.read_text(encoding='utf-8'))
            if row.get('owner_pid') != owner:
                continue
            if row.get('pid') in exclude_pids:
                continue
            proc = psutil.Process(row['pid'])
            if proc.create_time() != row['created'] or proc.exe() != row['executable'] or proc.environ().get('STUDIO_GPU_WORKER') != row['token']:
                continue
            cmd = ' '.join(proc.cmdline()).lower().replace('\\', '/')
            if not any((name in cmd for name in ('ninfer-serve', 'llama-server', 'ds4-server', 'comfyui/main.py', 'qwen-image-server'))):
                continue
            proc.terminate()
            stopped.append(proc)
        except psutil.NoSuchProcess:
            continue
        except (psutil.Error, OSError, ValueError, KeyError) as exc:
            raise RuntimeError('Could not verify or stop a managed model service') from exc
    _, alive = psutil.wait_procs(stopped, timeout=30)
    if alive:
        raise RuntimeError('Managed model service did not stop; refusing RTX streaming')

def activate(port: int) -> None:
    global _owner
    if os.name == 'nt' and 'pytest' not in ' '.join(psutil.Process().cmdline()).lower():
        _owner = (int(port), os.getpid())

def registry_dir(port: int) -> Path:
    return Path(__file__).resolve().parents[2] / '.studio-workers' / str(port)

def write_receipt(process, directory: Path, *, owner: int, token: str, preserve_on_restart: bool=False) -> bool:
    try:
        cmd = process.cmdline()
        if not cmd or 'pytest' in ' '.join(cmd).lower():
            return False
        if process.environ().get('STUDIO_GPU_WORKER') != token:
            return False
        record = {'pid': process.pid, 'created': process.create_time(), 'executable': process.exe(), 'owner_pid': owner, 'kind': 'gpu-worker', 'token': token, 'preserve_on_restart': preserve_on_restart}
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f'{process.pid}-{token}.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(record), encoding='utf-8')
        temporary.replace(target)
        return True
    except (psutil.Error, OSError):
        logging.getLogger(__name__).warning('Could not record Studio worker pid %s; sweep will skip it', process.pid)
        return False

def _spawn(*args, **kwargs):
    from .vr_session_guard import wait_for_gpu
    wait_for_gpu()
    audio = sys.modules.get('studio.audio')
    arbitrate = getattr(audio, 'spawn_gpu_worker', None)
    if arbitrate is None:
        return subprocess.Popen(*args, **kwargs)
    return arbitrate(lambda: subprocess.Popen(*args, **kwargs), args[0] if args else kwargs.get('args', []), kwargs.get('env') or os.environ)

def popen(*args, **kwargs):
    preserve_on_restart = bool(kwargs.pop('preserve_on_restart', False))
    if _owner is None or os.name != 'nt':
        return _spawn(*args, **kwargs)
    port, owner = _owner
    token = uuid.uuid4().hex
    source_env = kwargs.get('env')
    kwargs['env'] = {**(os.environ if source_env is None else source_env), 'STUDIO_GPU_WORKER': token}
    proc = _spawn(*args, **kwargs)
    directory = registry_dir(port)
    try:
        root = psutil.Process(proc.pid)
        root_recorded = write_receipt(root, directory, owner=owner, token=token, preserve_on_restart=preserve_on_restart)
    except psutil.Error:
        return proc

    def record_children():
        seen = set()
        recorded = root_recorded
        while root.is_running():
            try:
                if not recorded:
                    recorded = write_receipt(root, directory, owner=owner, token=token, preserve_on_restart=preserve_on_restart)
                for child in root.children(recursive=True):
                    identity = (child.pid, child.create_time())
                    if identity not in seen and write_receipt(child, directory, owner=owner, token=token, preserve_on_restart=preserve_on_restart):
                        seen.add(identity)
            except psutil.Error:
                pass
            time.sleep(0.25)
    threading.Thread(target=record_children, name='studio-worker-receipts', daemon=True).start()
    return proc

def run(*args, input=None, capture_output=False, timeout=None, check=False, **kwargs):
    if input is not None:
        if kwargs.get('stdin') is not None:
            raise ValueError('stdin and input arguments may not both be used')
        kwargs['stdin'] = subprocess.PIPE
    if capture_output:
        if kwargs.get('stdout') is not None or kwargs.get('stderr') is not None:
            raise ValueError('stdout and stderr arguments may not be used with capture_output')
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with popen(*args, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(input, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            exc.stdout, exc.stderr = process.communicate()
            raise
        except BaseException:
            process.kill()
            raise
        if check and process.returncode:
            raise subprocess.CalledProcessError(process.returncode, process.args, output=stdout, stderr=stderr)
        return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
