"""Setup-owned FFmpeg tools for video delivery; never depend on an operator PATH."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request
import uuid
import zipfile

URL = 'https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-8.1.2-essentials_build.zip'
SHA256 = 'db580001caa24ac104c8cb856cd113a87b0a443f7bdf47d8c12b1d740584a2ec'
SIZE = 109728040


def directory(runtime):
    return Path(runtime)/'tools'/'ffmpeg-8.1.2'/'bin'


def install(runtime, *, check=False):
    if os.name != 'nt':
        found={name:shutil.which(name) for name in ('ffmpeg','ffprobe')}
        return dict(ready=all(found.values()),tools=found)
    root=directory(runtime).parent;marker=root/'installed.json'
    ready=False
    if marker.is_file():
        receipt=json.loads(marker.read_text())
        required={'bin/ffmpeg.exe','bin/ffprobe.exe'}
        ready=required<=set(receipt.get('files',{})) and receipt.get('archive_sha256')==SHA256 and all((root/name).is_file() and
            hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in receipt.get('files',{}).items())
        ready=ready and all((directory(runtime)/(name+'.exe')).is_file() for name in ('ffmpeg','ffprobe'))
    if not ready and not check:
        destination=Path(runtime)
        while not destination.exists():destination=destination.parent
        if shutil.disk_usage(destination).free<2*1024**3:
            raise RuntimeError('Video tool setup needs 2 GiB free for download, extraction and headroom')
        downloads=Path(runtime)/'downloads';downloads.mkdir(parents=True,exist_ok=True)
        archive=downloads/'ffmpeg-8.1.2-essentials_build.zip'
        if not archive.is_file():
            pending=archive.with_name(archive.name+'.'+uuid.uuid4().hex+'.pending')
            digest=hashlib.sha256()
            with pending.open('xb') as output:
                for start in range(0,SIZE,1024**2):
                    end=min(SIZE,start+1024**2)-1
                    for attempt in range(3):
                        request=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
                        with urllib.request.urlopen(request,timeout=60) as response:
                            block=response.read(1024**2+1)
                            actual=response.headers.get('Content-Range','')
                        if len(block)==end-start+1 and actual==f'bytes {start}-{end}/{SIZE}':break
                    else:raise ValueError('Incomplete FFmpeg download; retained partial file for inspection')
                    output.write(block);digest.update(block)
            if digest.hexdigest()!=SHA256:raise ValueError('FFmpeg archive differs from its pinned SHA-256')
            pending.rename(archive)
        if archive.stat().st_size!=SIZE or hashlib.sha256(archive.read_bytes()).hexdigest()!=SHA256:
            raise ValueError('Existing FFmpeg archive differs; preserved for review')
        files={}
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                relative=Path(*Path(member.filename).parts[1:])
                if relative.as_posix() not in ('bin/ffmpeg.exe','bin/ffprobe.exe','LICENSE','README.txt'):continue
                data=package.read(member);target=root/relative
                if target.exists() and target.read_bytes()!=data:raise ValueError('Preserving a changed video tool: '+str(target))
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():target.write_bytes(data)
                files[relative.as_posix()]=hashlib.sha256(data).hexdigest()
        if not all((directory(runtime)/(name+'.exe')).is_file() for name in ('ffmpeg','ffprobe')):
            raise ValueError('FFmpeg archive is missing required tools')
        from .generation import publish
        publish(marker,dict(archive_sha256=SHA256,source=URL,files=files))
        ready=True
    if ready:
        for name in ('ffmpeg','ffprobe'):
            subprocess.run([str(directory(runtime)/(name+'.exe')),'-version'],check=True,
                stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=30)
    return dict(ready=ready,directory=str(directory(runtime)),archive_sha256=SHA256)
