# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/merge_back.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import os
import shutil
import sys
sys.path.insert(0, '')
from cosmos_windows.gguf_qat.rss_guard import start_rss_guard, peak_rss_gb, current_rss_gb
PROJ_TO_GGUF = {'q_proj': 'attn_q.weight', 'k_proj': 'attn_k.weight', 'v_proj': 'attn_v.weight', 'o_proj': 'attn_output.weight', 'q_a_proj': 'attn_q_a.weight', 'q_b_proj': 'attn_q_b.weight', 'kv_a_proj_with_mqa': 'attn_kv_a_mqa.weight'}

def load_lora_deltas(adapter_dir, device='cuda'):
    import json
    import torch
    from safetensors import safe_open
    cfg = json.load(open(os.path.join(adapter_dir, 'adapter_config.json')))
    r = int(cfg['r'])
    alpha = float(cfg['lora_alpha'])
    scaling = alpha / r
    A, B = ({}, {})
    path = os.path.join(adapter_dir, 'adapter_model.safetensors')
    with safe_open(path, 'pt') as f:
        for k in f.keys():
            if '.lora_A.' in k or '.lora_B.' in k:
                t = f.get_tensor(k).to(device).float()
                (A if '.lora_A.' in k else B)[k.replace('.lora_A.', '.').replace('.lora_B.', '.')] = (k, t)
    deltas = {}
    stems = {}
    with safe_open(path, 'pt') as f:
        for k in f.keys():
            if '.lora_A.weight' in k:
                stem = k.replace('.lora_A.weight', '')
                stems.setdefault(stem, {})['A'] = f.get_tensor(k).to(device).float()
            elif '.lora_B.weight' in k:
                stem = k.replace('.lora_B.weight', '')
                stems.setdefault(stem, {})['B'] = f.get_tensor(k).to(device).float()
    for stem, ab in stems.items():
        if 'A' not in ab or 'B' not in ab:
            continue
        parts = stem.split('.')
        proj = parts[-1]
        li = parts[parts.index('layers') + 1]
        if proj not in PROJ_TO_GGUF:
            continue
        delta = ab['B'] @ ab['A'] * scaling
        gname = f'blk.{li}.{PROJ_TO_GGUF[proj]}'
        deltas[gname] = delta
    return (deltas, scaling)

def merge(args):
    start_rss_guard(limit_gb=float(os.environ.get('GGUF_QAT_RSS_LIMIT_GB', '6.0')), label='merge_back')
    import numpy as np
    import torch
    from cosmos_windows.gguf_qat.reader import GGUFTensorReader
    from cosmos_windows.gguf_qat.dequant import dequantize, dequantize_q4_0
    from cosmos_windows.gguf_qat.sr_requant import sr_requantize_q4_0, merge_q4_0_with_residual, Q4_0_TYPE
    deltas, scaling = load_lora_deltas(args.adapter, device=args.device)
    print(f'[merge] LoRA scaling={scaling:.3f}; {len(deltas)} targeted Q4_0 tensors', flush=True)
    reader = GGUFTensorReader(args.gguf, device=args.device)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    print(f'[merge] streaming-copy base -> {args.out}', flush=True)
    with open(args.gguf, 'rb') as fin, open(args.out, 'wb') as fout:
        shutil.copyfileobj(fin, fout, length=8 * 1024 * 1024)
    changed_total = 0
    moved_l1 = 0.0
    with open(args.out, 'r+b') as fpatch:
        for gname, delta in deltas.items():
            info = reader.info(gname)
            assert info.quant == 'Q4_0', (gname, info.quant)
            raw_old = reader.raw_bytes(gname)
            out_f, in_f = (int(info.shape[1]), int(info.shape[0]))
            assert tuple(delta.shape) == (out_f, in_f), (gname, tuple(delta.shape), (out_f, in_f))
            delta_flat = delta.reshape(-1).to(torch.float32)
            if args.mode == 'one-shot':
                w_old = dequantize_q4_0(raw_old).reshape(-1)
                raw_new, _ = sr_requantize_q4_0(w_old + delta_flat, generator=gen, stochastic=args.stochastic, d_override=None if args.regrow_scale else _fixed_scale(raw_old))
                w_new = dequantize_q4_0(raw_new).reshape(-1)
                changed = int((raw_new.view(-1, Q4_0_TYPE) != raw_old.view(-1, Q4_0_TYPE)).any(1).sum())
            else:
                raw_new = raw_old.clone()
                res = torch.zeros_like(delta_flat)
                w0 = dequantize_q4_0(raw_old).reshape(-1)
                for _ in range(4):
                    raw_new, res, st = merge_q4_0_with_residual(raw_new, None, delta_flat / 4.0, residual=res, generator=gen, stochastic=args.stochastic, fixed_scale=not args.regrow_scale)
                w_new = dequantize_q4_0(raw_new).reshape(-1)
                changed = int((raw_new.view(-1, Q4_0_TYPE) != raw_old.view(-1, Q4_0_TYPE)).any(1).sum())
            moved_l1 += float((w_new - dequantize_q4_0(raw_old).reshape(-1)).abs().sum().item())
            changed_total += changed
            new_bytes = raw_new.detach().to('cpu').numpy().astype(np.uint8).tobytes()
            assert len(new_bytes) == info.n_bytes, (gname, len(new_bytes), info.n_bytes)
            fpatch.seek(info.data_offset)
            fpatch.write(new_bytes)
            del raw_old, raw_new, delta_flat
            torch.cuda.empty_cache()
    reader.close()
    print(f'[merge] mode={args.mode} stochastic={args.stochastic} regrow_scale={args.regrow_scale}: changed Q4_0 blocks={changed_total}, total |dw| L1={moved_l1:.2f}', flush=True)
    print(f'[merge] wrote {args.out} ({os.path.getsize(args.out) / 1000000.0:.1f} MB)', flush=True)
    print(f'[merge] peak RSS {peak_rss_gb():.3f} GB', flush=True)

def _fixed_scale(raw_old):
    import torch
    from cosmos_windows.gguf_qat.sr_requant import Q4_0_TYPE
    n = raw_old.numel() // Q4_0_TYPE
    return raw_old.view(n, Q4_0_TYPE)[:, :2].contiguous().view(torch.float16).to(torch.float32).reshape(n, 1)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gguf', default='')
    ap.add_argument('--adapter', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--mode', choices=['one-shot', 'delta4'], default='delta4')
    ap.add_argument('--stochastic', action='store_true', help='SR rounding (else round-to-nearest)')
    ap.add_argument('--regrow-scale', action='store_true', help='recompute per-block scale from merged target (else keep stored scale)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    merge(args)
