"""Pinned tokenizer/config assets installed explicitly, never fetched during a chat."""
import hashlib
import json
from pathlib import Path
import urllib.request
import uuid
from .config import ROOT


def specifications():
    return json.loads((ROOT/'locks/media-assets.json').read_text())


def directory(runtime, name):
    row=specifications()[name]
    return Path(runtime)/'cache'/'media-assets'/(name+'-'+row['revision'][:12])


def install(runtime, *, check=False):
    missing=[];verified=0
    for name, spec in specifications().items():
        root=directory(runtime,name)
        if not check:root.mkdir(parents=True,exist_ok=True)
        for row in spec['files']:
            target=root/row['name']
            if target.is_file():
                if hashlib.sha256(target.read_bytes()).hexdigest()!=row['sha256']:
                    raise ValueError('Preserving a changed media asset for review: '+str(target))
            elif check:missing.append(name+'/'+row['name']);continue
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                url=f'https://huggingface.co/{spec["repository"]}/resolve/{spec["revision"]}/{row["name"]}'
                temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.pending')
                size=0;digest=hashlib.sha256()
                with temporary.open('xb') as output:
                    # Bound individual transfers and verify complete ranges;
                    # a dropped connection must never create a "ready" cache.
                    for start in range(0,row['bytes'],1024**2):
                        end=min(start+1024**2,row['bytes'])-1
                        for attempt in range(3):
                            request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
                            with urllib.request.urlopen(request,timeout=60) as source:
                                block=source.read(1024**2+1)
                            if len(block)==end-start+1:break
                        else:raise ValueError('Incomplete media asset transfer: '+row['name'])
                        size+=len(block);digest.update(block);output.write(block)
                if size!=row['bytes'] or digest.hexdigest()!=row['sha256']:
                    raise ValueError('Media asset differs from its pinned SHA-256: '+row['name'])
                temporary.replace(target)
            verified+=1
    return dict(ready=not missing,verified_files=verified,missing=missing,model_weights_downloaded=False)
