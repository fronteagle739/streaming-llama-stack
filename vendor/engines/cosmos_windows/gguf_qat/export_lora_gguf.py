# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/export_lora_gguf.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from cosmos_windows.gguf_qat.reader import _ensure_gguf_py
_ensure_gguf_py()
import gguf
import numpy as np
PROJ_TO_GGUF = {'q_proj': 'attn_q.weight', 'k_proj': 'attn_k.weight', 'v_proj': 'attn_v.weight', 'o_proj': 'attn_output.weight', 'q_a_proj': 'attn_q_a.weight', 'q_b_proj': 'attn_q_b.weight', 'kv_a_proj_with_mqa': 'attn_kv_a_mqa.weight', 'o_proj_mla': 'attn_output.weight'}

def export(adapter_dir, out_path, arch):
    from safetensors import safe_open
    cfg = json.load(open(os.path.join(adapter_dir, 'adapter_config.json')))
    rank = int(cfg['r'])
    alpha = float(cfg['lora_alpha'])
    A, B = ({}, {})
    sf = os.path.join(adapter_dir, 'adapter_model.safetensors')
    with safe_open(sf, 'pt') as f:
        for k in f.keys():
            t = f.get_tensor(k).to('cpu').float().numpy()
            if k.endswith('.lora_A.weight'):
                A[k[:-len('.lora_A.weight')]] = t
            elif k.endswith('.lora_B.weight'):
                B[k[:-len('.lora_B.weight')]] = t
    tmp_path = out_path + '.tmp'
    gguf_arch = 'glm-dsa' if arch == 'glm_dsa' else arch
    w = gguf.GGUFWriter(tmp_path, gguf_arch)
    w.add_type(gguf.GGUFType.ADAPTER)
    w.add_string('adapter.type', 'lora')
    w.add_float32('adapter.lora.alpha', float(alpha))
    n = 0
    for stem, a in A.items():
        if stem not in B:
            continue
        b = B[stem]
        parts = stem.split('.')
        proj = parts[-1]
        li = parts[parts.index('layers') + 1]
        if proj not in PROJ_TO_GGUF:
            print(f'  [skip] unmapped proj {proj!r}')
            continue
        base = f'blk.{li}.{PROJ_TO_GGUF[proj]}'
        assert a.shape[0] == rank and b.shape[1] == rank, (stem, a.shape, b.shape, rank)
        w.add_tensor(f'{base}.lora_a', np.ascontiguousarray(a, dtype=np.float32))
        w.add_tensor(f'{base}.lora_b', np.ascontiguousarray(b, dtype=np.float32))
        n += 1
    if n == 0:
        raise ValueError('adapter contains no supported LoRA projection pairs; refusing to write an empty GGUF')
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    os.replace(tmp_path, out_path)
    print(f'[export] wrote {out_path} ({os.path.getsize(out_path) / 1000000.0:.2f} MB): {n} projections, rank={rank} alpha={alpha} arch={arch}')

def export_mlx(adapter_dir, out_dir):
    from safetensors import safe_open
    from safetensors.numpy import save_file
    cfg = json.load(open(os.path.join(adapter_dir, 'adapter_config.json'), encoding='utf-8'))
    rank = int(cfg['r'])
    alpha = float(cfg['lora_alpha'])
    weights = {}
    target_keys = set()
    layers = set()
    with safe_open(os.path.join(adapter_dir, 'adapter_model.safetensors'), framework='np') as handle:
        keys = set(handle.keys())
        for key in sorted(keys):
            if not key.endswith('.lora_A.weight'):
                continue
            stem = key[:-len('.lora_A.weight')]
            bkey = stem + '.lora_B.weight'
            if bkey not in keys:
                raise ValueError(f'PEFT LoRA pair is missing {bkey}')
            parts = stem.split('.')
            if 'layers' not in parts:
                raise ValueError(f'cannot map PEFT LoRA target to MLX: {stem}')
            layer_pos = parts.index('layers')
            layer = int(parts[layer_pos + 1])
            relative = '.'.join(parts[layer_pos + 2:])
            target = f'layers.{layer}.{relative}'
            a = handle.get_tensor(key).astype(np.float32, copy=False)
            b = handle.get_tensor(bkey).astype(np.float32, copy=False)
            if a.shape[0] != rank or b.shape[1] != rank:
                raise ValueError(f'invalid LoRA rank for {stem}: A={a.shape} B={b.shape} rank={rank}')
            weights[target + '.lora_a'] = np.ascontiguousarray(a.T)
            weights[target + '.lora_b'] = np.ascontiguousarray(b.T)
            target_keys.add(relative)
            layers.add(layer)
    if not weights:
        raise ValueError('adapter contains no complete MLX LoRA pairs')
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    tensor_path = output / 'adapters.safetensors'
    tensor_tmp = output / 'adapters.safetensors.next'
    save_file(weights, str(tensor_tmp), metadata={'schema': 'studio.mlx_stackable_lora.v1', 'runtimes': 'mlx,mlx_vlm'})
    os.replace(tensor_tmp, tensor_path)
    manifest = {'schema': 'studio.mlx_stackable_lora.v1', 'fine_tune_type': 'lora', 'num_layers': max(layers) + 1, 'layers': sorted(layers), 'lora_parameters': {'rank': rank, 'scale': alpha / rank, 'dropout': float(cfg.get('lora_dropout', 0.0)), 'keys': sorted(target_keys)}, 'source_format': 'peft', 'runtime_compatibility': ['mlx', 'mlx_vlm']}
    config_path = output / 'adapter_config.json'
    config_tmp = output / 'adapter_config.json.next'
    with config_tmp.open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(config_tmp, config_path)
    print(f'[export] wrote MLX adapter {output}: {len(weights) // 2} projections, rank={rank} scale={alpha / rank:g}')
    return output
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', required=True, help='peft-style adapter dir from the streamed trainer')
    ap.add_argument('--out', required=True, help='output adapter GGUF path')
    ap.add_argument('--mlx-out', default='', help='optional MLX/MLX-VLM stackable adapter directory')
    ap.add_argument('--arch', default='step35', choices=['step35', 'deepseek2', 'glm_dsa'])
    args = ap.parse_args()
    export(args.adapter, args.out, args.arch)
    if args.mlx_out:
        export_mlx(args.adapter, args.mlx_out)
