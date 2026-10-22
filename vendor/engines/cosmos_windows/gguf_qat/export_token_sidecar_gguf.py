# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/export_token_sidecar_gguf.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
import numpy as np
_KEY = re.compile('^layers\\.(?P<layer>\\d+)\\.experts\\.(?P<expert>\\d+)\\.(?P<part>norm\\.weight|gate_up_proj\\.weight|down_proj\\.weight|gate_proj\\.weight|gate_proj\\.bias)$')

def export_token_sidecar(source: Path, output: Path, *, outtype: str='f16') -> Path:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError('safetensors is required to export a token sidecar') from exc
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / 'prefill_llama.cpp' / 'gguf-py'))
    import gguf
    dtype = np.float16 if outtype == 'f16' else np.float32
    tensors: dict[str, np.ndarray] = {}
    seen: dict[tuple[int, int], set[str]] = {}
    with safe_open(str(source), framework='pt', device='cpu') as handle:
        metadata = handle.metadata() or {}
        if metadata.get('schema') != 'studio.token_gated_sidecar.v1':
            raise ValueError('input is not a studio.token_gated_sidecar.v1 artifact')
        if metadata.get('architecture') != 'glm-dsa':
            raise ValueError('only glm-dsa token sidecars are supported')
        for key in handle.keys():
            match = _KEY.match(key)
            if match is None:
                raise ValueError(f'unexpected token-sidecar tensor: {key}')
            layer = int(match.group('layer'))
            expert = int(match.group('expert'))
            part = match.group('part')
            array = handle.get_tensor(key).float().numpy()
            prefix = f'blk.{layer}.token_sidecar.{expert}'
            if part == 'gate_up_proj.weight':
                if array.ndim != 2 or array.shape[0] % 2:
                    raise ValueError(f'{key} must have shape [2 * width, hidden]')
                gate, up = np.split(array, 2, axis=0)
                tensors[f'{prefix}.ffn_gate.weight'] = np.ascontiguousarray(gate, dtype=dtype)
                tensors[f'{prefix}.ffn_up.weight'] = np.ascontiguousarray(up, dtype=dtype)
                seen.setdefault((layer, expert), set()).update({'ffn_gate.weight', 'ffn_up.weight'})
            else:
                runtime_part = {'norm.weight': 'norm.weight', 'down_proj.weight': 'ffn_down.weight', 'gate_proj.weight': 'gate.weight', 'gate_proj.bias': 'gate.bias'}[part]
                tensors[f'{prefix}.{runtime_part}'] = np.ascontiguousarray(array, dtype=dtype)
                seen.setdefault((layer, expert), set()).add(runtime_part)
    required = {'norm.weight', 'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight', 'gate.weight', 'gate.bias'}
    for identity, parts in seen.items():
        missing = required - parts
        if missing:
            raise ValueError(f'sidecar layer/expert {identity} is missing {sorted(missing)}')
    if not tensors:
        raise ValueError('token sidecar contains no tensors')
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(output, 'glm-dsa')
    writer.add_type('adapter')
    writer.add_string('adapter.type', 'token_sidecar')
    writer.add_name(source.stem)
    writer.add_description('Full-rank additive GLM experts with independent learned per-token gates')
    writer.add_string('token_sidecar.schema', 'studio.token_gated_sidecar.gguf.v1')
    writer.add_string('token_sidecar.base_fingerprint', metadata.get('base_fingerprint', ''))
    writer.add_string('token_sidecar.routing', 'independent_per_token_sigmoid')
    writer.add_string('token_sidecar.base_router', 'unchanged_top8')
    writer.add_string('token_sidecar.activation', 'silu')
    writer.add_string('token_sidecar.placement', 'post_attention_ffn_residual')
    writer.add_string('token_sidecar.runtime_compatibility', 'rtx_llama,mac_llama')
    for name in sorted(tensors):
        writer.add_tensor(name, tensors[name])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return output

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--outtype', choices=('f16', 'f32'), default='f16')
    args = parser.parse_args()
    result = export_token_sidecar(args.source, args.output, outtype=args.outtype)
    print(result)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
