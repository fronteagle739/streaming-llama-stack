"""Bounded-header model identification and cancellable background drive scans."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from .native_broker import vendor
from .lora_discovery import lora_info, is_adapter, header as tensor_header

FOLDER_CLASSES = {
    'AnimaPipeline':'anima-1.0', 'LTX2Pipeline':'ltx-2.3', 'LTXPipeline':'ltx-2.3',
    'WanPipeline':'wan-2.2', 'WanImageToVideoPipeline':'wan-2.2',
    'ZImagePipeline':'z-image', 'StableDiffusionXLPipeline':'sdxl',
    'QwenImagePipeline':'qwen-image-2.1', 'QwenImageEditPipeline':'qwen-image-2.1',
    'QwenImage21Pipeline':'qwen-image-2.1', 'QwenImage21Transformer2DModel':'qwen-image-2.1',
    'MOSSImagePipeline':'moss-image', 'MOSSImageTransformer':'moss-image',
    'Krea2Pipeline':'krea2-turbo', 'Krea2Transformer2DModel':'krea2-turbo',
    'MiniMaxH3Pipeline':'minimax-h3', 'MiniMaxH3DiTModel':'minimax-h3',
    'Flux2KleinPipeline':'flux2-klein', 'Flux2KleinKVPipeline':'flux2-klein', 'HunyuanImage3Pipeline':'hunyuan-3',
    'HunyuanImagePipeline':'hunyuan-3', 'Cosmos3Pipeline':'cosmos3-t2v',
}


def architectures():
    vendor();from studio.custom_models import ARCHITECTURES
    return {arch:[kind for kind,mode in [('image','t2i'),('video','t2v')] if mode in modes]
            for arch,modes in ARCHITECTURES.items() if 't2i' in modes or 't2v' in modes}


def json_config(path):
    if path.is_file() and path.stat().st_size <= 1 << 20:
        try:
            row=json.loads(path.read_text(encoding='utf-8'))
            return row if isinstance(row,dict) else {}
        except (ValueError,OSError):pass
    return {}


def support_folder(path, support):
    """Find local config bundles independently of standalone checkpoint discovery."""
    path=Path(path)
    conditioner=json_config(path/'text_conditioner'/'config.json')
    if conditioner.get('_class_name')=='AnimaTextConditioner':
        support.setdefault('anima-1.0',{}).setdefault('components_repo',str(path))
    index=json_config(path/'model_index.json')
    if index.get('_class_name')=='ZImagePipeline':
        values=support.setdefault('z-image',{});values.setdefault('config_dir',str(path))
        for component in ('text_encoder','vae'):
            folder=path/component
            if folder.is_dir() and any(folder.glob('*.safetensors')):
                values.setdefault(component,str(folder))
        for component in ('tokenizer','scheduler'):
            if (path/component).is_dir():values.setdefault(component,str(path/component))


def support_file(path, support):
    path=Path(path);name=path.name.casefold()
    if name in {'qwen_3_06b_base.safetensors','qwen3_0.6b.safetensors','qwen_3_0.6b.safetensors'}:
        support.setdefault('anima-1.0',{}).setdefault('text_encoder',str(path))
    if name in {'qwen_3_4b.safetensors','qwen3_4b.safetensors'}:
        support.setdefault('z-image',{}).setdefault('text_encoder',str(path))
    if name=='ae.safetensors':support.setdefault('z-image',{}).setdefault('vae',str(path))


def model_info(value, arch_hint=None):
    vendor();from studio.custom_models import checkpoint_metadata, _modalities_for_model, _safetensors_header
    path=Path(value).expanduser().resolve(strict=True);arch=None
    supported=architectures()
    if path.is_file() and path.suffix.lower()=='.safetensors':
        header=tensor_header(path);keys=[k for k in header if k!='__metadata__']
        if not keys or is_adapter(keys):return None
        stripped=[k.removeprefix('model.diffusion_model.').removeprefix('diffusion_model.') for k in keys]
        if any(k.startswith('txtfusion.') for k in stripped) and 'blocks.0.attn.wq.weight' in stripped:
            arch='krea2-native'
        elif any(k.startswith('double_blocks.') for k in stripped) and any(
            k.removeprefix('model.diffusion_model.').removeprefix('diffusion_model.')=='img_in.weight' and
            isinstance(v,dict) and len(v.get('shape',[]))==2 and v['shape'][1]==128 for k,v in header.items()):arch='flux2-klein'
        elif any(k.startswith('transformer_blocks.') and ('.img_mlp.' in k or '.img_mod.' in k) for k in stripped):arch='qwen-image-2.1'
        else:arch=checkpoint_metadata(str(path)).get('arch_hint')
        if arch=='ltx-2.3' and not any('adaln_single' in k or 'scale_shift_table' in k or 'audio_patchify_proj' in k for k in keys):arch=None
        metadata=header.get('__metadata__') or {}
        if not arch:
            declared=metadata.get('modelspec.architecture') or metadata.get('architecture') or metadata.get('model_type')
            if declared in supported:arch=declared
    elif path.is_dir():
        # A root with a transformer config is a valid Studio model even when
        # the exporter omitted diffusers' model_index.json (notably H3 packs).
        for candidate in (path/'model_index.json',path/'config.json',path/'transformer'/'config.json'):
            data=json_config(candidate)
            declared=data.get('_class_name') or data.get('model_type')
            arch=FOLDER_CLASSES.get(declared)
            if arch:break
        if arch=='z-image' and not any((path/'transformer').glob('*.safetensors')):
            return None  # Component/config bundle is support data, not a selectable model.
    else:return None
    if arch_hint:
        if arch_hint not in supported:raise ValueError('Unsupported model architecture')
        arch=arch_hint
    if arch not in supported:return None
    modalities=_modalities_for_model(arch,str(path))
    kinds=[kind for kind,mode in [('image','t2i'),('video','t2v')] if mode in modalities]
    if not kinds:return None
    return dict(id=hashlib.sha256(str(path).encode()).hexdigest()[:16],name=path.name,path=str(path),arch=arch,kinds=kinds)


SKIP={'$recycle.bin','system volume information','windows','node_modules','.git','.hg','.svn',
      '__pycache__','.venv','venv','site-packages','embeddings'}


def scan_models(root, explicit=(), *, overrides=None, cancelled=None, progress=None):
    rows={};loras={};support={};files_seen=directories=errors=0;last=0.;overrides=overrides or {}
    roots=[root] if isinstance(root,(str,Path)) else list(root or [])
    roots=list(dict.fromkeys(str(Path(p).expanduser().resolve()) for p in roots if p))
    def add(path):
        try:
            row=model_info(path,overrides.get(str(Path(path).resolve())))
            if row:rows[row['path']]=row
            return row
        except (OSError,ValueError,TypeError):return None
    def report(scanning):
        state=dict(models=sorted(rows.values(),key=lambda r:(r['name'].casefold(),r['path'])),support_files=deepcopy(support),
                   loras=sorted(loras.values(),key=lambda r:(r['arches'],r['name'].casefold(),r['path'])),roots=roots,
                   truncated=False,scanning=scanning,files_seen=files_seen,directories_seen=directories,unreadable_directories=errors)
        if progress:progress(state)
        return state
    for path in explicit:
        if path:add(path)
    report(bool(roots))
    for search_root in roots:
        base=Path(search_root).resolve()
        if not base.is_dir():errors+=1;continue
        def unreadable(_):
            nonlocal errors
            errors+=1
        for directory,dirs,files in os.walk(base,followlinks=False,onerror=unreadable):
            if cancelled and cancelled.is_set():return report(False)
            directories+=1
            dirs[:]=sorted(d for d in dirs if d.casefold() not in SKIP and not Path(directory,d).is_symlink() and
                           not (hasattr(Path(directory,d),'is_junction') and Path(directory,d).is_junction()))
            # Shards and component weights belong to the detected pipeline.
            if 'model_index.json' in files or 'config.json' in files or 'transformer' in dirs or 'text_conditioner' in dirs:
                support_folder(directory,support)
                pipeline=add(directory)
                if pipeline:
                    if pipeline['arch']=='qwen-image-2.1':support.setdefault('qwen-image-2.1',{})['components_repo']=pipeline['path']
                    if pipeline['arch']=='flux2-klein':support.setdefault('flux2-klein',{})['components_repo']=pipeline['path']
                    dirs[:]=[d for d in dirs if d not in {'transformer','text_encoder','text_encoder_2','text_encoder_3','vae','video_vae','audio_vae'}]
            for name in files:
                files_seen+=1
                if name.lower() in ('qwen3vl_4b_int8.safetensors','qwen_image_vae.safetensors'):
                    key='text_encoder' if name.lower().startswith('qwen3vl') else 'vae'
                    for arch in ('krea2-native','krea2-turbo','krea2-raw','krea2'):
                        support.setdefault(arch,{}).setdefault(key,str(Path(directory,name)))
                if name.lower().endswith('.safetensors'):
                    file=Path(directory)/name
                    support_file(file,support)
                    try:adapter=lora_info(file)
                    except (OSError,ValueError,TypeError):adapter=None
                    if adapter:loras[adapter['path']]=adapter
                    else:add(file)
                if cancelled and cancelled.is_set():return report(False)
            if time.monotonic()-last>.3:last=time.monotonic();report(True)
    return report(False)


class Scanner:
    def __init__(self,root,explicit=(),overrides=None):
        self.lock=threading.Lock();self.cancelled=threading.Event()
        self.current=dict(models=[],loras=[],support_files={},truncated=False,scanning=bool(root),files_seen=0,directories_seen=0,unreadable_directories=0)
        self.thread=threading.Thread(target=self.run,args=(root,explicit,overrides),daemon=True,name='model-discovery');self.thread.start()

    def run(self,root,explicit,overrides):
        try:scan_models(root,explicit,overrides=overrides,cancelled=self.cancelled,progress=self.publish)
        except (OSError,ValueError) as exc:
            state=self.state();state.update(scanning=False,error=str(exc));self.publish(state)

    def publish(self,state):
        with self.lock:self.current=state

    def state(self):
        with self.lock:return deepcopy(self.current)
