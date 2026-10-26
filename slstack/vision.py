"""Original image embeddings in streaming chat context, with pixel-bound reuse."""
from __future__ import annotations
import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
from .journal import sha256
from .generation import publish


def has_images(body):
    return any(isinstance(m.get('content'),list) and any(isinstance(p,dict) and p.get('type') in ('image_url','input_image')
        for p in m['content']) for m in body.get('messages',[]))


def image_bytes(url,cfg):
    if not isinstance(url,str):raise ValueError('Image URL must be a string')
    if url.startswith('data:'):
        match=re.fullmatch(r'data:image/(png|jpeg|jpg|webp|gif|bmp);base64,([A-Za-z0-9+/=\r\n]+)',url)
        if not match:raise ValueError('Unsupported image data URL')
        raw=base64.b64decode(match[2],validate=True)
    elif url.startswith('/outputs/'):
        root=(Path(cfg['runtime_root']).parent/'outputs').resolve()
        # Stack preferences belong to the parent router's runtime root.
        if cfg.get('stack_preferences_file'):root=(Path(cfg['stack_preferences_file']).parent.parent/'outputs').resolve()
        path=(root/url.removeprefix('/outputs/')).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():raise ValueError('Image escaped the local output directory')
        if path.stat().st_size>20<<20:raise ValueError('Image exceeds20MiB')
        raw=path.read_bytes()
    else:raise ValueError('Attach an image as a data URL or use a local generated image')
    if not raw or len(raw)>20<<20:raise ValueError('Image must contain1..20MiB')
    return raw


class Images:
    def __init__(self,cfg):self.cfg=cfg;self.pins={}

    def pin(self,path):
        path=Path(path).resolve(strict=True);stat=path.stat();key=(str(path),stat.st_size,stat.st_mtime_ns)
        if key not in self.pins:self.pins[key]=sha256(path)
        return self.pins[key]

    def encoder(self):
        location=self.cfg.get('stack_preferences_file');value=self.cfg.get('vision_encoder')
        if location and Path(location).is_file():value=json.loads(Path(location).read_text(encoding='utf-8')).get('vision_encoder') or value
        if not value:raise ValueError('The vision-enabled launcher could not find the compatible DeepSeek V4.1 vision encoder beside the language model')
        return Path(value).resolve(strict=True)

    def encode(self,raw,before_encode):
        encoder=self.encoder();library=Path(self.cfg['llama_producer']).parent/'ds4-decode.dll'
        key=hashlib.sha256((hashlib.sha256(raw).hexdigest()+self.pin(encoder)+self.pin(library)).encode()).hexdigest()
        root=Path(self.cfg['runtime_root'])/'vision';root.mkdir(parents=True,exist_ok=True)
        output=root/(key+'.vision');manifest=root/(key+'.json')
        if manifest.is_file():
            row=json.loads(manifest.read_text(encoding='utf-8'))
            if row.get('key')!=key or self.pin(output)!=row['sha256']:raise ValueError('Cached image embedding identity differs')
            return row
        if output.exists():raise ValueError('An incomplete image encoding is retained; inspect it before retrying')
        if shutil.disk_usage(root).free<3<<30:raise MemoryError('Vision encoding needs3GiB free disk headroom')
        image=root/(key+'.input')
        if not image.exists():
            with image.open('xb') as f:f.write(raw)
        before_encode()
        from .config import environment
        libraries=[str(library.parent),*[str(p) for p in self.cfg.get('producer_library_directories',[])]]
        request=dict(library=str(library),libraries=libraries,model=self.cfg['model'],encoder=str(encoder),image=str(image),output=str(output))
        env=environment(self.cfg,'rtx-producer');env['PATH']=os.pathsep.join([*libraries,env.get('PATH','')])
        proc=subprocess.run([sys.executable,'-B','-m','slstack.vision_worker'],input=json.dumps(request)+'\n',
                            text=True,encoding='utf-8',errors='replace',capture_output=True,env=env,
                            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        (root/(key+'.log')).write_text(proc.stderr,encoding='utf-8')
        rows=[json.loads(line.removeprefix('SLSTACK_VISION ')) for line in proc.stdout.splitlines() if line.startswith('SLSTACK_VISION ')]
        if proc.returncode or len(rows)!=1 or rows[0].get('error'):
            raise RuntimeError((rows[0].get('error') if rows else 'Native vision worker failed')+'; see '+str(root/(key+'.log')))
        row=dict(rows[0],key=key,file=str(output),sha256=self.pin(output));publish(manifest,row);return row

    def prepare(self,body,before_encode):
        result=deepcopy(body);blocks=[]
        for message in result.get('messages',[]):
            if not isinstance(message.get('content'),list):continue
            content=[]
            for part in message['content']:
                kind=part.get('type') if isinstance(part,dict) else None
                if kind in ('image_url','input_image'):
                    if message.get('role') not in ('user','tool','function'):raise ValueError('Images belong in user or tool messages')
                    url=part.get('image_url');url=url.get('url') if isinstance(url,dict) else url
                    row=self.encode(image_bytes(url,self.cfg),before_encode)
                    marker='[slstack-image-'+row['key']+'-'+str(len(blocks))+']'
                    blocks.append(dict(row,marker=marker));content.append(marker)
                elif kind in ('text','input_text'):content.append(part.get('text',''))
                else:raise ValueError('This preset supports text and image content')
            message['content']=''.join(content)
        identity=hashlib.sha256('|'.join(row['key'] for row in blocks).encode()).hexdigest() if blocks else ''
        return result,blocks,identity


def render_tokens(text,tokenizer,blocks):
    tokens=[];spans=[];cursor=0
    for block in blocks:
        if text.count(block['marker'])!=1:raise ValueError('Image placeholder was altered by the chat template')
        at=text.index(block['marker'],cursor);tokens.extend(tokenizer.rendered(text[cursor:at]))
        spans.append(dict(position=len(tokens),count=block['count'],file=block['file'],sha256=block['sha256']))
        tokens.extend([129264]*block['count']);cursor=at+len(block['marker'])
    tokens.extend(tokenizer.rendered(text[cursor:]));return tokens,spans
