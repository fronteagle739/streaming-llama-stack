"""Header-only adapter discovery and per-model compatibility; no weights loaded."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from .native_broker import vendor


MARKERS=('lora_a','lora_b','lora_up','lora_down','.lora.','lokr_','hada_','dora_scale')


def is_adapter(keys):return any(any(marker in k.lower() for marker in MARKERS) for k in keys)


@lru_cache(maxsize=24)
def _header(location,size,modified):
    vendor();from studio.custom_models import _safetensors_header
    return _safetensors_header(location)


def header(path):
    path=Path(path);stat=path.stat()
    return _header(str(path.resolve()),stat.st_size,stat.st_mtime_ns)


def target(key):
    key=key.lower()
    for suffix in ('.lora_a.default.weight','.lora_b.default.weight','.lora_a.weight','.lora_b.weight',
                   '.lora_down.weight','.lora_up.weight','.lora_a','.lora_b'):
        if key.endswith(suffix):key=key[:-len(suffix)];break
    for prefix in ('base_model.model.','model.diffusion_model.','diffusion_model.','transformer.','diffusion_model_',
                   'lora_unet_','lora_transformer_','lycoris_','unet.','dit.','pipe.dit.'):
        if key.startswith(prefix):key=key[len(prefix):]
    return key.removesuffix('.weight').replace('.','_')


def lora_info(value):
    path=Path(value).resolve(strict=True)
    if not path.is_file() or path.suffix.lower()!='.safetensors':return None
    row=header(path);keys=[k for k in row if k!='__metadata__']
    if not is_adapter(keys):return None
    metadata=row.get('__metadata__') or {}
    declared=' '.join(str(metadata.get(k,'')) for k in ('modelspec.architecture','ss_base_model_version',
        'ss_sd_model_name','base_model','base_model_name_or_path','architecture','model_type','model')).lower()
    config=path.parent/'adapter_config.json'
    if config.is_file() and config.stat().st_size<1<<20:
        try:declared+=' '+str(json.loads(config.read_text(encoding='utf-8')).get('base_model_name_or_path','')).lower()
        except (OSError,ValueError):pass
    normalized='\n'.join(target(k) for k in keys)
    # Metadata and tensor topology determine families; filenames are labels.
    arches=[]
    if 'moss' in declared or re.search(r'layers_\d+_proj_ffn_(down|up_gate)',normalized):arches=['moss-image']
    elif 'minimax' in declared or 'h3' in declared or re.search(r'blocks_\d+_attn_(qkv_proj|out_proj)',normalized):arches=['minimax-h3']
    elif 'qwen' in declared and 'image' in declared or re.search(r'transformer_blocks_\d+_(img|txt)_(mlp|mod)',normalized):arches=['qwen-image-2.1']
    elif 'flux' in declared or 'double_blocks_' in normalized or 'single_blocks_' in normalized:arches=['flux2-klein']
    elif 'krea' in declared or 'txtfusion' in normalized:arches=['krea2','krea2-native','krea2-turbo','krea2-raw']
    elif 'z_image' in declared or 'z-image' in declared or 'zimage' in declared or 'all_x_embedder' in normalized or 'noise_refiner' in normalized:arches=['z-image']
    elif 'anima' in declared or 'adaln_modulation_cross_attn' in normalized or 'llm_adapter' in normalized:arches=['anima-1.0']
    elif 'ltx' in declared or 'audio_patchify_proj' in normalized:arches=['ltx-2.3']
    elif 'wan' in declared or 'blocks_' in normalized and ('self_attn_k' in normalized or 'cross_attn_k' in normalized):arches=['wan-2.2']
    elif 'sdxl' in declared or 'sd_xl' in declared or 'conditioner_embedders' in normalized or 'input_blocks_' in normalized:arches=['sdxl']
    return dict(id=hashlib.sha256(str(path).encode()).hexdigest()[:16],name=path.stem,path=str(path),arches=arches,
                compatibility='family detected' if arches else 'Unknown family; no compatible model established')


@lru_cache(maxsize=12)
def _model_shapes(files):
    result={}
    for location,size,modified in files:
        for key,row in _header(location,size,modified).items():
            if key!='__metadata__' and isinstance(row,dict) and key.endswith('.weight'):
                result[target(key)]=row.get('shape',[])
    return result


def model_shapes(model):
    path=Path(model['path'])
    if path.is_file():files=[path]
    else:
        component=path/'transformer' if (path/'transformer').is_dir() else path
        files=sorted(component.glob('*.safetensors'))
    if len(files)>256:return {}
    return _model_shapes(tuple((str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns) for p in files))


def compatibility(lora,model):
    if model['arch'] not in lora['arches']:return False,'Different model family'
    # Check every directly matching adapter matrix against the selected model.
    # Native converters handle aliases/fused targets and perform final validation.
    shapes=model_shapes(model);matched=0
    for key,row in header(lora['path']).items():
        if key=='__metadata__' or not isinstance(row,dict):continue
        lower=key.lower();shape=row.get('shape',[]);base=shapes.get(target(key))
        if not base or len(shape)<2 or len(base)<2:continue
        if 'lora_a' in lower or 'lora_down' in lower:
            if shape[1:]!=base[1:]:return False,'Input dimensions differ from '+model['name']
        elif 'lora_b' in lower or 'lora_up' in lower:
            if shape[0]!=base[0]:return False,'Output dimensions differ from '+model['name']
        else:continue
        matched+=1
    if not matched:return False,'No matching adapter targets established for '+model['name']
    return True,'Matching family and tensor dimensions; backend validates adapter format'


def public_loras(catalog,selected=None):
    models=[selected] if selected else catalog['models'];result=[]
    for row in catalog.get('loras',[]):
        matches=[];reasons=[]
        for model in models:
            if model['arch'] not in row['arches']:continue
            try:ok,reason=compatibility(row,model)
            except (OSError,ValueError,TypeError):ok,reason=False,'Model or adapter metadata unavailable'
            if ok:matches.append({k:model[k] for k in ('id','name','arch')})
            else:reasons.append(reason)
        if selected and not matches:continue
        result.append(dict(id=row['id'],name=row['name'],compatible_models=matches,
                           compatibility='Matching family and dimensions' if matches else next(iter(reasons),row['compatibility'])))
    return sorted(result,key=lambda r:(not bool(r['compatible_models']),r['name'].casefold(),r['id']))
