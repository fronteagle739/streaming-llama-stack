from __future__ import annotations
import json
import math
from pathlib import Path
CATALOG = json.loads(Path(__file__).with_name('sampler_capabilities.json').read_text(encoding='utf-8'))
FIELDS = CATALOG['fields']
PROFILES = CATALOG['profiles']
ALIASES = {'basalt-wsl': 'basalt', 'rtx-llama': 'llama', 'llama-server': 'llama', 'remote-llama': 'llama', 'mlx': 'mlx-vlm', 'gguf-mlx': 'mlx-receiver', 'kimi-prompt-direct': 'mlx-receiver', 'kimi-receiver': 'mlx-receiver', 'decode-receiver': 'mlx-receiver', 'kimi-decode': 'mlx-receiver', 'kimi-direct-shim': 'mlx-receiver', 'kimi-remote': 'mlx-receiver', 'kimi-openai-shim': 'mlx-receiver', 'kimi-shim': 'mlx-receiver', 'glm-mtp-chat': 'mtp-shim', 'ds4-windows': 'ds4', 'ds4-native': 'ds4', 'gguf-stream': 'llama-decode', 'llama-decode-upstream': 'llama-decode', 'upstream-llama-decode': 'llama-decode', 'llama-upstream-decode': 'llama-decode'}

def number(value, *, integer=False):
    if value is None or isinstance(value, bool) or (isinstance(value, str) and (not value.strip())):
        return None
    try:
        n = float(value)
        if not math.isfinite(n) or (integer and (not n.is_integer())):
            return None
        return int(n) if integer else n
    except (TypeError, ValueError, OverflowError):
        return None

def values(side: dict) -> dict:
    out = {}
    for key, field in FIELDS.items():
        value = side.get(key)
        if field['type'] == 'strings':
            if isinstance(value, list) and all((isinstance(v, str) for v in value)):
                out[key] = value
        elif (n := number(value, integer=field['type'] == 'integer')) is not None:
            out[key] = n
    return out

def launch_args(side: dict, profile: str='llama') -> list[str]:
    spec = PROFILES[profile]
    supplied = values(side)
    keys = spec['keys'] if spec.get('launch') else spec.get('launch_keys', [])
    out = []
    for key in keys:
        flag = FIELDS[key].get('flag')
        if key not in supplied or not flag:
            continue
        value = supplied[key]
        if key == 'dry_sequence_breakers':
            for breaker in value or ['none']:
                out.extend([flag, breaker])
        else:
            out.extend([flag, ';'.join(value) if key == 'samplers' else str(value)])
    return out
