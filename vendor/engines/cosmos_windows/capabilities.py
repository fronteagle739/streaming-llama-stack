# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/capabilities.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
CONTROLNET_SUPPORT = {'zimage': {'kinds': ['canny', 'depth', 'pose', 'scribble', 'tile'], 'mechanism': 'ZImageControlNetPipeline (Fun-Control Union, fp8)'}, 'anima': {'kinds': ['depth', 'lineart', 'scribble', 'pose', 'inpainting'], 'mechanism': 'ControlNet-LLLite (kohya v2, standalone port)'}}
MULTI_LORA_SUPPORT = {'zimage': True, 'anima': True, 'ltx': True, 'wan': True, 'cosmos': True}

def _norm(model_arch: str) -> str:
    m = (model_arch or '').lower()
    for key in ('zimage', 'z-image', 'z_image'):
        if key in m:
            return 'zimage'
    for key in CONTROLNET_SUPPORT:
        if key in m:
            return key
    for key in ('ltx', 'wan', 'cosmos', 'anima', 'ideogram', 'hunyuan', 'flux'):
        if key in m:
            return key
    return m

def controlnet_supported(model_arch: str) -> bool:
    return _norm(model_arch) in CONTROLNET_SUPPORT

def controlnet_kinds(model_arch: str) -> list[str]:
    return list(CONTROLNET_SUPPORT.get(_norm(model_arch), {}).get('kinds', []))

def multi_lora_supported(model_arch: str) -> bool:
    return MULTI_LORA_SUPPORT.get(_norm(model_arch), False)

def capabilities(model_arch: str) -> dict:
    cn = controlnet_supported(model_arch)
    return {'controlnet': cn, 'controlnet_kinds': controlnet_kinds(model_arch), 'multi_lora': multi_lora_supported(model_arch)}
if __name__ == '__main__':
    import sys, json
    arg = sys.argv[1] if len(sys.argv) > 1 else 'zimage'
    print(json.dumps(capabilities(arg), indent=1))
