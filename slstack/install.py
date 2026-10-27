"""Install isolated Python media runtimes and the pinned redistributable SDK."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import urllib.request
import zipfile
from .config import ROOT, path
from .journal import atomic_json
from .process import OwnedProcess

SDK_VERSION = '1.3.0'
SDK_SHA256 = '8ab6c1082537565388b79050d94fdd7f5f437cd0f1d909e7b19f786042b97177'
SDK_URL = f'https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.directstorage/{SDK_VERSION}/microsoft.direct3d.directstorage.{SDK_VERSION}.nupkg'


def python_path(root):
    return root / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def sdk(cfg, *, check=False):
    download = Path(cfg['runtime_root']) / 'downloads' / f'directstorage-{SDK_VERSION}.nupkg'
    targets = [ROOT / 'engines' / name / 'external/directstorage' for name in ('ds4-win', 'streaming-llama/prefill_llama.cpp')]
    if all((target / 'native/lib/x64/dstorage.lib').is_file() for target in targets):
        return {'sdk': SDK_VERSION, 'ready': True}
    if check:
        return {'sdk': SDK_VERSION, 'ready': False, 'action': 'slstack install --sdk'}
    download.parent.mkdir(parents=True, exist_ok=True)
    if not download.exists():
        with urllib.request.urlopen(SDK_URL, timeout=60) as response, download.open('xb') as output:
            total = 0
            while chunk := response.read(65536):
                total += len(chunk)
                if total > 64 * 1024**2:
                    raise ValueError('DirectStorage SDK download exceeds 64 MiB bound')
                output.write(chunk)
    digest = hashlib.file_digest(download.open('rb'), 'sha256').hexdigest()
    if digest != SDK_SHA256:
        raise ValueError('DirectStorage package hash differs from the reviewed pin')
    with zipfile.ZipFile(download) as archive:
        for entry in archive.infolist():
            if not entry.filename.startswith('native/') and entry.filename not in ('LICENSE.txt', 'LICENSE-CODE.txt', 'NOTICES.txt'):
                continue
            if entry.is_dir():
                continue
            for target in targets:
                destination = (target / entry.filename).resolve()
                if not destination.is_relative_to(target.resolve()):
                    raise ValueError('Unsafe SDK member path')
                destination.parent.mkdir(parents=True, exist_ok=True)
                data = archive.read(entry)
                if destination.exists() and destination.read_bytes() != data:
                    if destination.suffix == '.txt' and destination.read_bytes().replace(b'\r\n', b'\n') == data.replace(b'\r\n', b'\n'):
                        continue
                    raise ValueError('Existing SDK differs; preserving it for review: ' + str(destination))
                if not destination.exists():
                    destination.write_bytes(data)
    return {'sdk': SDK_VERSION, 'ready': True, 'package_sha256': digest}


def media_profile(cfg, profile, *, check=False):
    from .media_profiles import PROFILES
    spec=PROFILES[profile]
    runtime = Path(cfg['runtime_root']) / spec['directory']
    binary = python_path(runtime)
    lock = ROOT / 'locks' / spec['lock']
    lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest()
    marker = runtime / 'slstack-installed.json'
    install_cfg={**cfg,'host_budget_gib':min(3,cfg['host_budget_gib'])}
    ready = binary.is_file() and marker.is_file() and json.loads(marker.read_text())['lock_sha256'] == lock_sha
    if not check and not ready:
        destination=runtime
        while not destination.exists():destination=destination.parent
        if shutil.disk_usage(destination).free < 24 * 1024**3:
            raise RuntimeError('Media setup needs 24 GiB free for packages, download/build staging and headroom; choose a runtime on a larger volume')
        if not binary.exists():
            OwnedProcess(install_cfg, [sys.executable, '-B', '-m', 'venv', str(runtime)], 'install-venv', timeout=180).wait()
        OwnedProcess(install_cfg, [str(binary), '-B', '-m', 'pip', 'install', '--disable-pip-version-check', '-r', str(lock)],
                     'install-media', timeout=7200).wait()
        OwnedProcess(install_cfg, [str(binary), '-B', '-m', 'pip', 'check'], 'install-verify', timeout=120).wait()
        ready = True
    if binary.is_file():
        probe_cfg={**cfg,'host_budget_gib':min(3,cfg['host_budget_gib'])}
        probe=OwnedProcess(probe_cfg,[str(binary),'-B','-m','slstack.media_runtime',profile],'install-media-capabilities',timeout=180)
        try:
            failure=None
            try:probe.wait()
            except RuntimeError as exc:failure=exc
            rows=[json.loads(line.removeprefix('SLSTACK_MEDIA_RUNTIME ')) for line in probe.log.read_text(errors='replace').splitlines() if line.startswith('SLSTACK_MEDIA_RUNTIME ')]
            capabilities=rows[-1] if rows else dict(ready=False,missing=[dict(error='Media import verification did not finish; see '+str(probe.log))])
            ready=ready and capabilities['ready'] and failure is None
            if not ready and not check:raise RuntimeError('Media runtime verification failed: '+json.dumps(capabilities)) from failure
        finally:probe.stop()
    if not check and ready:
        atomic_json(marker, {'lock_sha256': lock_sha, 'python': str(binary), 'profile':profile,
                             'weights_downloaded': False,'capabilities_verified':True})
    return dict(profile=profile,ready=ready,python=str(binary),lock=spec['lock'],
                capabilities=capabilities if binary.is_file() else None)


def install(cfg, *, image=False, video=False, all_engines=False, check=False, sdk_only=False):
    if os.name != 'nt':
        raise ValueError('RTX media installer requires native Windows and Python 3.12; Mac backends use mac-setup')
    if sys.version_info[:2] != (3, 12):
        raise ValueError('Install Python 3.12, then run the launcher with that interpreter')
    result = {'sdk': sdk(cfg, check=check), 'engines': []}
    if sdk_only:return result
    if not (image or video or all_engines):all_engines=True
    from .media_assets import install as install_assets
    result['media_assets']=install_assets(cfg['runtime_root'],check=check)
    from .media_tools import install as install_tools
    result['video_tools']=install_tools(cfg['runtime_root'],check=check)
    profiles=['studio','qwen21'] if image or all_engines else ['studio']
    result['media_runtimes']=[media_profile(cfg,p,check=check) for p in profiles]
    ready=all(row['ready'] for row in result['media_runtimes']) and result['media_assets']['ready'] and result.get('video_tools',{'ready':True})['ready']
    runtime=Path(cfg['runtime_root'])
    binaries = [path(cfg['ds4_command'][0]), path(cfg['llama_producer']),
                path(cfg.get('weight_fetcher', 'build/llama/bin/slstack-weight-fetcher.dll')), ROOT / 'build/llama/bin/llama-server.exe',
                ROOT / 'build/llama/bin/llama-export-lora.exe', path(cfg['llama_producer']).parent / 'ds4-decode.dll']
    if not check and all_engines:
        # The documented install step prepares the native engines and UI too.
        # Builds never need model state or a GPU reservation.
        # Each guarded compiler step uses request_timeout_s; a cold CUDA build
        # exceeds the 900 s chat-request default, so give steps the build bound.
        build_cfg = {**cfg, 'host_budget_gib': min(3, cfg['host_budget_gib']),
                     'min_available_gib': max(4, cfg['min_available_gib']),
                     'request_timeout_s': 7200}
        build_config = Path(cfg['runtime_root']) / 'install-build.local.json'
        atomic_json(build_config, build_cfg)
        outer = {**build_cfg, 'host_budget_gib': build_cfg['host_budget_gib'] + .5}
        print('Building native engines and chat UI; progress is recorded in the runtime jobs directory.', flush=True)
        OwnedProcess(outer, ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                            str(ROOT / 'build.ps1'), '-Config', str(build_config), '-Target', 'all'],
                     'install-build', timeout=7200).wait()
    if all_engines:
        result['build'] = {'ready': all(p.is_file() for p in binaries)
                          and (path(cfg['webui_root']) / 'index.html').is_file()}
    for kind, enabled in (('image', image or all_engines), ('video', video or all_engines)):
        if enabled:
            result['engines'].append({'engine': kind, 'ready': ready, 'runtime': str(runtime),
                                      'note': 'Native weight-fetcher shared build is also required; flash-attn is an explicit optional prerequisite for H3 flash_attn2'})
    if not check:
        # The one-time full model SHA-256 runs here, not on the first prompt.
        from .check import verify_model_receipt
        receipt = verify_model_receipt(cfg)
        if receipt is not None:
            result['model_receipt'] = receipt
    return result
