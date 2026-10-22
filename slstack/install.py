"""Install isolated Python media runtimes and the pinned redistributable SDK."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
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


def install(cfg, *, image=False, video=False, all_engines=False, check=False, sdk_only=False):
    if os.name != 'nt':
        raise ValueError('RTX media installer requires native Windows and Python 3.12; Mac backends use mac-setup')
    if sys.version_info[:2] != (3, 12):
        raise ValueError('Install Python 3.12, then run the launcher with that interpreter')
    result = {'sdk': sdk(cfg, check=check), 'engines': []}
    if sdk_only:
        return result
    if not (image or video or all_engines):
        all_engines = True
    runtime = Path(cfg['runtime_root']) / 'media-venv'
    binary = python_path(runtime)
    lock = ROOT / 'locks/media-win-py312.txt'
    lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest()
    marker = runtime / 'slstack-installed.json'
    ready = binary.is_file() and marker.is_file() and json.loads(marker.read_text())['lock_sha256'] == lock_sha
    if not check and not ready:
        if not binary.exists():
            OwnedProcess(cfg, [sys.executable, '-B', '-m', 'venv', str(runtime)], 'install-venv', timeout=180).wait()
        OwnedProcess(cfg, [str(binary), '-B', '-m', 'pip', 'install', '--disable-pip-version-check', '-r', str(lock)],
                     'install-media', timeout=7200).wait()
        OwnedProcess(cfg, [str(binary), '-B', '-m', 'pip', 'check'], 'install-verify', timeout=120).wait()
        atomic_json(marker, {'lock_sha256': lock_sha, 'python': str(binary), 'kind': 'shared image/video venv',
                             'weights_downloaded': False})
        ready = True
    for kind, enabled in (('image', image or all_engines), ('video', video or all_engines)):
        if enabled:
            result['engines'].append({'engine': kind, 'ready': ready, 'runtime': str(runtime),
                                      'note': 'Native weight-fetcher shared build is also required; flash-attn is an explicit optional prerequisite for H3 flash_attn2'})
    return result
