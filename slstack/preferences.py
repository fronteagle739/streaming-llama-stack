"""Portable stack settings, Studio model discovery and local filesystem picker."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import threading
from .generation import publish
from .native_broker import vendor
from .discovery import model_info, scan_models, Scanner, architectures

TOOL_NAMES = ('set_quality_speed', 'get_quality_speed', 'list_media_models', 'list_loras', 'generate_image', 'generate_video')
FIELDS = ('comfyui_models', 'image_model', 'video_model', 'image_arch', 'video_arch', 'vision_encoder',
          'enabled_tools', 'cache_media', 'cache_conversations', 'cache_budget_gib', 'model_arch_overrides', 'model_search_roots', 'media_load_modes')






def browse(raw=''):
    if not raw:
        roots=[Path(f'{letter}:/') for letter in 'CDEFGHIJKLMNOPQRSTUVWXYZ'] if os.name=='nt' else [Path('/'),Path.home()]
        return dict(path='',parent=None,entries=[dict(name=str(p),path=str(p),directory=True) for p in roots if p.is_dir()])
    path=Path(raw).expanduser().resolve(strict=True)
    if not path.is_dir():raise ValueError('Choose a directory')
    entries=[];truncated=False
    with os.scandir(path) as stream:
        for entry in stream:
            if len(entries)>=2000:truncated=True;break
            if entry.name.startswith('.'):continue
            try:
                directory=entry.is_dir()
                if not directory and Path(entry.name).suffix.lower() not in ('.safetensors','.gguf'):continue
                entries.append(dict(name=entry.name,path=str(Path(entry.path).resolve()),directory=directory))
            except OSError:continue
    entries.sort(key=lambda r:(not r['directory'],r['name'].casefold()))
    return dict(path=str(path),parent=str(path.parent) if path.parent!=path else '',entries=entries,truncated=truncated)


class Preferences:
    def __init__(self,cfg):
        self.cfg=cfg;self.path=Path(cfg.get('stack_preferences_file') or Path(cfg['runtime_root'])/'settings'/'stack.local.json')
        self.path.parent.mkdir(parents=True,exist_ok=True);self.lock=threading.RLock()
        self.base={k:deepcopy(cfg.get(k,'')) for k in FIELDS[:6]}
        self.base.update(enabled_tools={name:True for name in TOOL_NAMES},cache_media=True,cache_conversations=True,cache_budget_gib=16,model_arch_overrides={},model_search_roots=deepcopy(cfg.get('model_search_roots',[])),media_load_modes=deepcopy(cfg.get('media_load_modes',{})))
        self.scanner=None;self.stamp=None;self.scan_spec=None
        self.reload()

    def reload(self):
        with self.lock:
            stamp=self.path.stat().st_mtime_ns if self.path.exists() else 0
            if self.stamp==stamp:return
            row=deepcopy(self.base)
            if stamp:
                saved=json.loads(self.path.read_text(encoding='utf-8'));row.update(saved)
                row['enabled_tools']={**self.base['enabled_tools'],**saved.get('enabled_tools',{})}
            if not row['vision_encoder'] and self.cfg.get('vision_bridge_version')==1:row['vision_encoder']=self.base['vision_encoder']
            self.row=row;self.cfg.update(deepcopy(row));self.stamp=stamp
            spec=([row['comfyui_models'],*row['model_search_roots']],[row['image_model'],row['video_model']],row['model_arch_overrides'])
            if spec!=self.scan_spec:
                if self.scanner:self.scanner.cancelled.set()
                self.scan_spec=deepcopy(spec);self.scanner=Scanner(*spec)

    @property
    def catalog(self):return self.scanner.state()

    def update(self,changes):
        if not isinstance(changes,dict) or not set(changes)<=set(FIELDS):raise ValueError('Unknown stack setting')
        with self.lock:
            self.reload();row=deepcopy(self.row)
            for key,value in changes.items():
                if key=='enabled_tools':
                    if not isinstance(value,dict) or not set(value)<=set(TOOL_NAMES) or any(type(v) is not bool for v in value.values()):raise ValueError('Invalid tool toggle')
                    row[key].update(value)
                elif key=='model_search_roots':
                    if not isinstance(value,list) or len(value)>32 or any(not isinstance(p,str) or not p or len(p)>4096 for p in value):raise ValueError('Choose up to32 extra search folders')
                    paths=[Path(p).expanduser().resolve(strict=True) for p in value]
                    if any(not p.is_dir() for p in paths):raise ValueError('Every search root must be a directory or drive')
                    row[key]=list(dict.fromkeys(str(p) for p in paths))
                elif key=='model_arch_overrides':
                    if not isinstance(value,dict) or len(value)>4096 or any(not isinstance(k,str) or v not in architectures() for k,v in value.items()):raise ValueError('Invalid model architecture selection')
                    row[key].update(value)
                elif key=='media_load_modes':
                    if not isinstance(value,dict) or any(not isinstance(k,str) or not k or len(k)>4096 or v not in ('streaming','resident') for k,v in value.items()):
                        raise ValueError('Choose streamed or resident media loading')
                    merged={**row[key],**value}
                    if len(merged)>4096:raise ValueError('Too many model loading preferences')
                    row[key]=merged
                elif key.startswith('cache_'):
                    if key=='cache_budget_gib':
                        if type(value) is not int or not 1<=value<=128:raise ValueError('Cache budget must be 1..128 GiB')
                    elif type(value) is not bool:raise ValueError('Cache toggle must be a boolean')
                    row[key]=value
                else:
                    if not isinstance(value,str) or len(value)>4096:raise ValueError('Invalid model path or architecture')
                    row[key]=value
            if row['comfyui_models']:
                folder=Path(row['comfyui_models']).expanduser().resolve(strict=True)
                if folder.parent!=folder and (folder.name.lower()=='comfyui' or (folder/'main.py').is_file()) and (folder/'models').is_dir():folder=folder/'models'
                if not folder.is_dir():raise ValueError('ComfyUI folder must be a directory')
                row['comfyui_models']=str(folder)
            for kind in ('image','video'):
                if row[kind+'_model']:
                    selected=Path(row[kind+'_model']).expanduser().resolve(strict=True)
                    hint=changes.get(kind+'_arch') if kind+'_arch' in changes else row['model_arch_overrides'].get(str(selected))
                    info=model_info(selected,hint or None)
                    if not info or kind not in info['kinds']:raise ValueError('This model is not supported for '+kind+' generation')
                    row[kind+'_model']=info['path'];row[kind+'_arch']=info['arch']
                    if hint:row['model_arch_overrides'][info['path']]=info['arch']
            if row['vision_encoder']:
                encoder=Path(row['vision_encoder']).expanduser().resolve(strict=True)
                if not encoder.is_file() or encoder.suffix.lower()!='.gguf':raise ValueError('Choose a compatible vision encoder GGUF')
                row['vision_encoder']=str(encoder)
            if not changes:self.scan_spec=None  # The explicit Rescan action.
            publish(self.path,row);self.stamp=None;self.reload()
            return self.state()

    def state(self):
        self.reload()
        return dict(settings=deepcopy(self.row),catalog=self.catalog,tools=list(TOOL_NAMES),architectures=architectures(),media_load_modes=self.media_modes())

    def enabled(self,name):
        self.reload();return self.row['enabled_tools'].get(name,False)

    def media_mode(self,path,kind):
        self.reload()
        return self.row['media_load_modes'].get(str(path),self.cfg.get(kind+'_mode','streaming'))

    def media_modes(self):
        self.reload()
        return {model['id']:self.row['media_load_modes'].get(model['path'],self.cfg.get(model['kinds'][0]+'_mode','streaming')) for model in self.catalog['models']}

    def select(self,kind,ident=None):
        self.reload();rows=[r for r in self.catalog['models'] if kind in r['kinds']]
        if ident:
            match=next((r for r in rows if ident in (r['id'],r['path'],r['name'])),None)
            if not match:raise ValueError('Unknown or unsupported '+kind+' model; call list_media_models')
            return match
        selected=self.row[kind+'_model']
        return next((r for r in rows if r['path']==selected),None) or next(iter(rows),None)

    def list_loras(self,model=None,kind='image'):
        from .lora_discovery import public_loras
        self.reload();selected=self.select(kind,model) if model else None
        catalog=self.catalog;models=[m for m in catalog['models'] if kind in m['kinds']]
        families={m['arch'] for m in models}
        loras=[row for row in catalog.get('loras',[]) if not row['arches'] or families.intersection(row['arches'])]
        return public_loras({**catalog,'models':models,'loras':loras},selected)

    def resolve_loras(self,kind,model,specs):
        from .lora_discovery import compatibility
        selected=self.select(kind,model)
        if not selected:raise ValueError('Choose a supported model before attaching LoRAs')
        available=self.catalog.get('loras',[]);result=[];seen=set()
        for spec in specs:
            matches=[r for r in available if spec['name'] in (r['name'],r['id'])]
            if len(matches)!=1:raise ValueError('Unknown or ambiguous LoRA name; call list_loras and use its ID: '+spec['name'])
            row=matches[0]
            if row['id'] in seen:raise ValueError('Duplicate LoRA in this generation')
            ok,reason=compatibility(row,selected)
            if not ok:raise ValueError('LoRA '+row['name']+' is incompatible with '+selected['name']+': '+reason)
            seen.add(row['id']);result.append(dict(path=row['path'],weight=spec['strength'],name=row['name']))
        return result
