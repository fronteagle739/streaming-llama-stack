"""Verify installed media classes, rather than treating pip success as readiness."""
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
from .native_broker import vendor

REQUIRED = {
    'diffusers': ('AnimaAutoBlocks','AnimaTextConditioner','CosmosTransformer3DModel',
                  'Krea2Pipeline','Krea2Transformer2DModel','QwenImage21Pipeline',
                  'QwenImage21Transformer2DModel','AutoencoderKLQwenImage21',
                  'Flux2KleinPipeline','Flux2KleinKVPipeline','Flux2Transformer2DModel','ZImagePipeline'),
    'transformers': ('Qwen3VLForConditionalGeneration','Qwen3VLProcessor','Qwen3Model'),
    'studio.vram_residency': ('VramResidencyManager',),
    'studio.adapter': ('load_backend',),
    'studio.sim': ('_safe',),
}


def verify(profile='studio'):
    vendor()
    missing=[]
    required=dict(REQUIRED)
    if profile=='qwen21':
        required['diffusers']=('QwenImage21Pipeline','QwenImage21Transformer2DModel','AutoencoderKLQwenImage21')
        required['bitsandbytes']=('nn',)
    else:
        required['diffusers']=tuple(n for n in required['diffusers'] if 'QwenImage21' not in n and n!='AutoencoderKLQwenImage21')
        try:
            from studio.boogu_krea import _ensure_krea2_installed
            _ensure_krea2_installed()
        except ImportError as exc:missing.append(dict(module='studio.boogu_krea',error=str(exc)))
    for package, names in required.items():
        try:
            module=importlib.import_module(package)
            for name in names:getattr(module,name)
            if package in ('diffusers','transformers') and not Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
                raise RuntimeError('Package came from outside the isolated runtime')
        except (ImportError, AttributeError, RuntimeError) as exc:
            missing.append(dict(module=package,error=str(exc)))
    versions={}
    for name in ('torch','diffusers','transformers','peft','safetensors'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:missing.append(dict(module=name,error='Distribution is missing'))
    return dict(ready=not missing,profile=profile,missing=missing,versions=versions)


if __name__=='__main__':
    result=verify(sys.argv[1] if len(sys.argv)>1 else 'studio')
    print('SLSTACK_MEDIA_RUNTIME '+json.dumps(result),flush=True)
    raise SystemExit(0 if result['ready'] else 1)
