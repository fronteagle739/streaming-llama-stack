# Exported from Studio 167e764e6b19dfe4c30cc8ed6fb0938fa7663fd4; machine-specific fallback paths cleared.
from __future__ import annotations
import os, sys, signal, subprocess
from pathlib import Path
from typing import Optional
from . import worker_registry

def _cosmos_windows_roots() -> list[str]:
    roots: list[str] = []
    try:
        from .config import settings
        cwp = getattr(settings, 'cosmos_windows_path', None)
        if cwp and Path(str(cwp)).is_dir():
            roots.append(str(cwp))
    except Exception:
        pass
    vendor = Path(__file__).resolve().parent.parent / 'vendor'
    if vendor.is_dir() and str(vendor) not in roots:
        roots.append(str(vendor))
    backend_root = Path(__file__).resolve().parent.parent
    if backend_root.is_dir() and str(backend_root) not in roots:
        roots.append(str(backend_root))
    return roots

def _unbuffered_env() -> dict:
    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = '1'
    try:
        from .config import configure_model_storage_env, settings
        configure_model_storage_env(settings.models_root, env)
    except Exception:
        pass
    roots = _cosmos_windows_roots()
    if roots:
        existing = env.get('PYTHONPATH', '').strip()
        env['PYTHONPATH'] = os.pathsep.join(roots + ([existing] if existing else []))
    return env

def _popen_group_kwargs() -> dict:
    if os.name == 'nt':
        flags = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0) | getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        return {'creationflags': flags} if flags else {}
    return {'start_new_session': True}

def kill_process_tree(proc: subprocess.Popen, *, timeout: float=10.0) -> None:
    if proc.poll() is not None:
        return
    pid = proc.pid
    if os.name == 'nt':
        import psutil
        observed = []
        try:
            children = psutil.Process(pid).children(recursive=True)
        except psutil.Error:
            children = []
        for child in children:
            try:
                observed.append((child.pid, child.create_time()))
            except psutil.Error:
                pass
        try:
            stopped = subprocess.run(['taskkill', '/T', '/F', '/PID', str(pid)], capture_output=True, timeout=timeout, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except Exception:
            stopped = None
        if stopped is None or stopped.returncode != 0:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        owned = []
        for child_pid, created in observed:
            try:
                child = psutil.Process(child_pid)
                if child.create_time() != created:
                    continue
                owned.append(child)
                try:
                    child.terminate()
                except psutil.AccessDenied:
                    pass
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(owned, timeout=timeout)
        if alive:
            raise RuntimeError('owned worker descendants did not exit: ' + ', '.join((str(child.pid) for child in alive)))
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f'owned worker {pid} did not exit') from exc
        return
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            try:
                proc.wait(timeout=timeout)
                return
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                return
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass

def managed_popen(argv: list[str], env_extra: Optional[dict[str, str]]=None, *, gpu_worker: bool=True, isolate_python: bool=True, stdin_pipe: bool=False) -> subprocess.Popen:
    env = _unbuffered_env()
    if isolate_python:
        for key in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'PYTHONSTARTUP', 'PYTHONEXECUTABLE'):
            env.pop(key, None)
        env['PYTHONPATH'] = os.pathsep.join(_cosmos_windows_roots())
    if env_extra:
        env.update({str(k): str(v) for k, v in env_extra.items() if v is not None})
    launch = worker_registry.popen if gpu_worker else subprocess.Popen
    return launch(argv, stdin=subprocess.PIPE if stdin_pipe else None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True, env=env, **_popen_group_kwargs())
