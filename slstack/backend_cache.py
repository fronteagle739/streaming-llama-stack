"""Pinned backend inputs and immutable runtime source snapshots."""
from __future__ import annotations
import hashlib
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request
import uuid
import zipfile
from .config import ROOT


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


class Cache:
    def __init__(self,cfg,*,directory=None,offline=False):
        self.root=Path(directory or cfg.get('backend_source_cache') or Path(cfg['runtime_root'])/'downloads/backends').expanduser().resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.offline=offline
        rows=json.loads((ROOT/'locks/backend-sources.json').read_text())['sources']
        self.items={r['id']:r for r in rows}

    def get(self,name):
        row=self.items[name];target=self.root/row['filename']
        if row['format']=='git':return self._git(row,target)
        if target.exists():
            if not target.is_file() or digest(target)!=row['sha256']:raise ValueError('Backend cache differs from its pinned hash: '+name)
            return target
        if self.offline:raise FileNotFoundError('Offline backend source is missing: '+row['filename'])
        if shutil.disk_usage(self.root).free<max(2*1024**3,row['bytes']*3):raise OSError('Insufficient backend download headroom')
        partial=self.root/(row['filename']+'.'+uuid.uuid4().hex+'.partial')
        request=urllib.request.Request(row['url'],headers={'User-Agent':'streaming-llama-stack-setup'})
        with urllib.request.urlopen(request,timeout=60) as response,partial.open('xb') as output:
            total=0
            while chunk:=response.read(1024**2):
                total+=len(chunk)
                if total>row['bytes']:raise ValueError('Backend download exceeds its pinned size: '+name)
                output.write(chunk)
        if partial.stat().st_size!=row['bytes'] or digest(partial)!=row['sha256']:
            raise ValueError('Backend input hash failed; the partial file is preserved for inspection: '+name)
        if target.exists():
            if digest(target)!=row['sha256']:raise ValueError('Concurrent backend cache collision: '+name)
        else:partial.replace(target)
        return target

    def _git(self,row,target):
        record=target.with_suffix(target.suffix+'.json')
        if target.is_file() and record.is_file():
            previous=json.loads(record.read_text())
            if previous.get('revision')==row['revision'] and digest(target)==previous.get('sha256'):return target
            raise ValueError('Pinned Git archive cache diverged: '+row['id'])
        if target.exists() or record.exists():raise ValueError('Incomplete Git archive is preserved: '+row['id'])
        if self.offline:raise FileNotFoundError('Offline source archive is missing: '+row['filename'])
        repo=self.root/(row['id']+'-'+row['revision'][:12]+'.git')
        env={**os.environ,'GIT_TERMINAL_PROMPT':'0'}
        def git(*args):
            try:return subprocess.check_output(['git','-c','gc.auto=0',*args],env=env,stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as exc:raise RuntimeError('Pinned source fetch failed for '+row['id']+': '+exc.stderr.decode(errors='replace')[-1000:]) from exc
        if not repo.exists():git('init','--bare',str(repo))
        git('-C',str(repo),'fetch','--quiet','--depth=1','--no-tags',row['repository'],row['revision'])
        actual=git('-C',str(repo),'rev-parse','FETCH_HEAD^{commit}').decode().strip()
        if actual!=row['revision']:raise ValueError('Backend source revision differs from its pin: '+row['id'])
        git('-C',str(repo),'fsck','--connectivity-only','--no-dangling')
        partial=target.with_name(target.name+'.'+uuid.uuid4().hex+'.partial')
        git('-C',str(repo),'archive','--format=tar.gz','--prefix=source/','-o',str(partial),actual)
        # Validate the gzip footer before publishing the complete archive.
        with gzip.open(partial,'rb') as stream:
            while stream.read(1024**2):pass
        partial.replace(target)
        record.write_text(json.dumps(dict(revision=actual,sha256=digest(target)),indent=2))
        return target

    def unpack(self,name,parent):
        row=self.items[name];pin=row.get('sha256') or row['revision'];destination=Path(parent)/(name+'-'+pin[:16])
        marker=destination/'stack-source.json'
        if marker.is_file():
            prior=json.loads(marker.read_text())
            if prior.get('source_pin')!=pin:raise ValueError('Source snapshot marker differs: '+name)
            actual=tree_files(destination,exclude={'stack-source.json'})
            if fingerprint(actual)!=prior['source_sha256']:raise ValueError('Source snapshot has local changes; preserving it: '+name)
            return destination
        if destination.exists():raise ValueError('Incomplete source snapshot is preserved; choose a separate runtime: '+str(destination))
        archive=self.get(name)
        Path(parent).mkdir(parents=True,exist_ok=True)
        work=destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.partial')
        work.mkdir()
        if row['format']=='zip':
            with zipfile.ZipFile(archive) as stream:
                members=[m for m in stream.infolist() if not m.filename.startswith('__MACOSX/')]
                roots={Path(m.filename).parts[0] for m in members if Path(m.filename).parts}
                if len(roots)!=1:raise ValueError('Expected one zip source root')
                size=sum(m.file_size for m in members)
                if size>4*1024**3 or shutil.disk_usage(parent).free<size+4*1024**3:raise OSError('Insufficient source extraction headroom')
                for member in members:
                    path=Path(member.filename)
                    if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe zip source member')
                    if ((member.external_attr>>16)&0o170000)==0o120000:raise ValueError('Zip source symlinks are not supported')
                    parts=path.parts[1:]
                    if not parts:continue
                    target=work.joinpath(*parts)
                    if member.is_dir():target.mkdir(parents=True,exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True,exist_ok=True)
                        with stream.open(member) as src,target.open('xb') as out:shutil.copyfileobj(src,out,1024**2)
        else:
          with tarfile.open(archive,'r:*') as stream:
            members=stream.getmembers();prefix=None;size=0
            for member in members:
                path=Path(member.name)
                if path.is_absolute() or '..' in path.parts or not path.parts:raise ValueError('Unsafe backend archive member')
                if prefix is None:prefix=path.parts[0]
                if path.parts[0]!=prefix:raise ValueError('Expected a single backend archive root')
                size+=member.size
            if size>4*1024**3 or shutil.disk_usage(parent).free<size+4*1024**3:
                raise OSError('Backend extraction exceeds the available source/build budget')
            for member in members:
                parts=Path(member.name).parts[1:]
                if not parts:continue
                member.name=Path(*parts).as_posix()
                stream.extract(member,work,filter='data')
        files=tree_files(work)
        (work/'stack-source.json').write_text(json.dumps(dict(source_pin=pin,archive_sha256=digest(archive),source_sha256=fingerprint(files)),indent=2))
        if work.resolve().parent!=Path(parent).resolve() or destination.resolve().parent!=Path(parent).resolve():raise ValueError('Source publication escaped its runtime directory')
        if destination.exists():raise ValueError('Concurrent source publication; both snapshots are preserved')
        work.replace(destination)
        return destination


def tree_files(root,*,exclude=()):
    result=[];root=Path(root)
    retained=any(root.is_relative_to(ROOT/name) for name in ('engines','vendor'))
    if retained and (ROOT/'.git').exists():
        names=subprocess.check_output(['git','-C',str(ROOT),'ls-files','--cached','-z','--',root.relative_to(ROOT).as_posix()]).decode().split('\0')
        for name in names:
            if not name:continue
            file=ROOT/name
            if not file.is_file():continue
            relative=file.relative_to(root).as_posix()
            if relative not in exclude:result.append((relative,file.read_bytes()))
        return sorted(result)
    for folder,children,names in os.walk(root):
        children[:]=sorted(n for n in children if n not in {'.git','__pycache__','.pytest_cache','.build','build','node_modules','zig-out','.zig-cache'})
        for name in sorted(names):
            file=Path(folder)/name;relative=file.relative_to(root).as_posix()
            if relative in exclude or file.suffix=='.pyc':continue
            if file.is_symlink() and not file.resolve().is_relative_to(root.resolve()):raise ValueError('External source symlink: '+relative)
            result.append((relative,file.read_bytes()))
    return result


def fingerprint(files):
    result=hashlib.sha256()
    for name,data in sorted(files):result.update(name.encode()+b'\0'+hashlib.sha256(data).digest())
    return result.hexdigest()


def snapshot(source,runtime,label,*,transform=None):
    """Never build in the retained source or overwrite a divergent build input."""
    files=tree_files(source,exclude={'stack-source.json'});revision=fingerprint(files)
    if transform:files=transform(files)
    revision=hashlib.sha256((revision+fingerprint(files)).encode()).hexdigest()
    destination=Path(runtime)/'sources'/(label+'-'+revision[:16])
    for name,data in files:
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if target.read_bytes()!=data:raise ValueError('Runtime build input diverged: '+name)
        else:
            with target.open('xb') as stream:stream.write(data)
    return destination,revision
