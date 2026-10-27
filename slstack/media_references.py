"""Pixel-bound chat image references for Studio's conditioning inputs."""
from __future__ import annotations
import base64
import hashlib
import io
import os
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit
import uuid

MAX_IMAGE = 20 << 20
IMAGE_ID = re.compile(r'image-[0-9a-f]{64}')
GENERATED = re.compile(r'!\[Generated media\]\(([^)\r\n]+\.(?:png|jpe?g|webp|gif|bmp))\)', re.I)


def image_extension(raw):
    if raw.startswith(b'\x89PNG\r\n\x1a\n'):return '.png'
    if raw.startswith(b'\xff\xd8\xff'):return '.jpg'
    if raw[:6] in (b'GIF87a',b'GIF89a'):return '.gif'
    if raw.startswith(b'RIFF') and raw[8:12]==b'WEBP':return '.webp'
    if raw.startswith(b'BM'):return '.bmp'
    raise ValueError('Reference must be a PNG, JPEG, WebP, GIF or BMP image')


class References:
    def __init__(self,cfg,*,scoped=False):
        self.cfg=cfg;self.runtime=Path(cfg['runtime_root']).resolve()
        self.root=(self.runtime/'input-images').resolve()
        if not self.root.is_relative_to(self.runtime):raise ValueError('Image storage escaped the runtime')
        self.scoped=scoped;self.paths={}

    def local_output(self,value):
        url=urlsplit(value)
        if url.scheme or url.netloc:
            if (url.scheme!='http' or url.username or url.password or
                url.hostname not in {self.cfg.get('host','127.0.0.1'),'localhost','127.0.0.1','::1'} or
                url.port!=int(self.cfg.get('port',8088))):raise ValueError('Use a chat image ID or a stack-local generated image URL')
        if url.query or url.fragment:raise ValueError('Reference URLs must not have query strings or fragments')
        route=unquote(url.path)
        if not route.startswith('/outputs/'):raise ValueError('Use a chat image ID or a stack-local generated image URL')
        root=(self.runtime/'outputs').resolve()
        if not root.is_relative_to(self.runtime):raise ValueError('Output folder escaped the runtime')
        target=(root/route.removeprefix('/outputs/')).resolve(strict=True)
        if not target.is_relative_to(root) or not target.is_file():raise ValueError('Reference escaped the local output folder')
        if target.stat().st_size>MAX_IMAGE:raise ValueError('Reference image exceeds 20 MiB')
        return target.read_bytes()

    def add(self,value):
        if value.startswith('data:'):
            if len(value)>MAX_IMAGE*4//3+1024:raise ValueError('Reference image exceeds 20 MiB')
            match=re.fullmatch(r'data:image/(?:png|jpeg|jpg|webp|gif|bmp);base64,([A-Za-z0-9+/=\r\n]+)',value)
            if not match:raise ValueError('Unsupported uploaded image encoding')
            raw=base64.b64decode(match[1],validate=True)
        else:raw=self.local_output(value)
        if not raw or len(raw)>MAX_IMAGE:raise ValueError('Reference image must contain 1 byte to 20 MiB')
        suffix=image_extension(raw)
        from PIL import Image,UnidentifiedImageError
        try:
            with Image.open(io.BytesIO(raw)) as image:
                if not 0<image.width*image.height<=32*1024**2:raise ValueError('Reference image exceeds 32 megapixels')
                image.verify()
        except (UnidentifiedImageError,OSError,SyntaxError,Image.DecompressionBombError) as exc:
            raise ValueError('Reference image cannot be decoded') from exc
        digest=hashlib.sha256(raw).hexdigest();identity='image-'+digest
        self.root.mkdir(parents=True,exist_ok=True);target=self.root/(digest+suffix)
        if not target.exists():
            if shutil.disk_usage(self.root).free<len(raw)+(512<<20):raise MemoryError('Image references need 512 MiB of disk headroom')
            pending=target.with_name(target.name+'.'+uuid.uuid4().hex+'.pending')
            with pending.open('xb') as output:output.write(raw);output.flush();os.fsync(output.fileno())
            os.replace(pending,target)
        target=target.resolve(strict=True)
        if not target.is_relative_to(self.root) or hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            raise ValueError('Saved reference image identity differs')
        self.paths[identity]=str(target)
        return identity

    def resolve(self,value):
        if not isinstance(value,str):raise ValueError('Image reference must be an image ID or URL')
        if IMAGE_ID.fullmatch(value):
            if self.scoped and value not in self.paths:raise ValueError('That image ID is not in this conversation')
            files=list(self.root.glob(value.removeprefix('image-')+'.*'))
            files=[p for p in files if p.suffix in ('.png','.jpg','.webp','.gif','.bmp')]
            if len(files)!=1:raise ValueError('Unknown or ambiguous chat image ID')
            target=files[0].resolve(strict=True)
            if not target.is_relative_to(self.root) or target.stat().st_size>MAX_IMAGE:
                raise ValueError('Invalid saved reference image')
            if hashlib.sha256(target.read_bytes()).hexdigest()!=value.removeprefix('image-'):
                raise ValueError('Saved reference image identity differs')
            return str(target)
        return self.resolve(self.add(value))

    def annotate(self,messages):
        """Keep each annotation beside its image, preserving stable prompt prefixes."""
        for message in messages:
            content=message.get('content')
            if isinstance(content,list):
                parts=[]
                for part in content:
                    parts.append(part)
                    if not isinstance(part,dict) or part.get('type') not in ('image_url','input_image'):continue
                    value=part.get('image_url');value=value.get('url') if isinstance(value,dict) else value
                    if not isinstance(value,str):raise ValueError('Image attachment needs an image URL')
                    identity=self.add(value)
                    parts.append(dict(type='text',text='\nChat image reference: '+identity+'. Use this ID as source_image or in reference_images when requested.\n'))
                message['content']=parts
            elif message.get('role')=='assistant' and isinstance(content,str):
                def describe(match):
                    identity=self.add(match[1])
                    return match[0]+'\nChat image reference: '+identity+'.'
                message['content']=GENERATED.sub(describe,content)


def conditioning(kind,args,arch,model):
    """Validate against Studio's model contract before releasing the LLM/GPU."""
    from .native_broker import vendor
    vendor()
    from studio.custom_models import _modalities_for_model
    image=args.get('_source_image_path');refs=args.get('_reference_image_paths',[])
    mode=args.get('modality')
    if not mode:
        mode=('i2i' if kind=='image' else 'ref2v' if refs else 'i2v') if (image or refs) else ('t2i' if kind=='image' else 't2v')
    allowed={'image':{'t2i','i2i'},'video':{'t2v','i2v','ref2v'}}[kind]
    if mode not in allowed:raise ValueError('Invalid '+kind+' conditioning mode')
    if mode in ('t2i','t2v') and (image or refs):raise ValueError('Choose an image-conditioned mode when supplying images')
    if mode=='i2v' and (not image or refs):raise ValueError('i2v requires source_image only; use ref2v for reference images')
    if mode in ('i2i','ref2v') and not (image or refs):raise ValueError(mode+' requires source_image or reference_images')
    native='r2v' if mode=='ref2v' else mode
    supported=_modalities_for_model(arch,model)
    # The Studio adapters also implement SDEdit for text-only Wan weights.
    if native=='i2i' and arch=='wan-2.2':supported=[*supported,'i2i']
    if native not in supported:raise ValueError(str(arch)+' does not support '+mode+'; choose a compatible model')
    multi={'qwen-image-2.1','boogu','boogu-edit','bernini'}
    if mode=='i2i' and arch not in multi:
        all_images=list(dict.fromkeys([*([image] if image else []),*refs]))
        if len(all_images)>1:raise ValueError(str(arch)+' supports one source image, not multiple references')
        if all_images:image=all_images[0];refs=[]
    elif mode=='i2i' and arch=='bernini':
        refs=list(dict.fromkeys([*([image] if image else []),*refs]));image=None
    if mode=='ref2v':
        if arch!='minimax-h3' and image:refs=list(dict.fromkeys([image,*refs]));image=None
        if not refs:refs=[image];image=None
    return dict(modality='ref2v' if mode=='ref2v' and arch=='minimax-h3' else native,
                image_path=image,reference_images=refs,denoise=args.get('denoise'))


def h3_reference_prompt(prompt,count):
    """Studio shot_render's ordered Picture/Subject bindings, without filenames."""
    header='Ordered MiniMax H3 visual references (keep this exact order):'
    if not count or prompt.startswith(header):return prompt
    labelled=[f'<Picture {index}> / <Subject {index}>: Supplied reference image {index}.' for index in range(1,count+1)]
    return '\n'.join([header,*labelled,'','Target video request:',prompt])
