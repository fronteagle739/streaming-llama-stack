# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/magi2_attn_memprofile.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import json

def _mb(x):
    return round(x / 1000000.0, 1)

def profile_attention(tokens: int, heads: int, head_dim: int, cfg_batch: int=2, with_sink: bool=True) -> dict:
    import torch
    from cosmos_windows.weight_streaming.magi2_attention_adapter import preview_bidirectional_sdpa
    dev = 'cuda'
    dt = torch.bfloat16
    seg = tokens
    total = seg * cfg_batch
    cu = [0]
    for i in range(cfg_batch):
        cu.append(cu[-1] + seg)
    rep = {'tokens_per_segment': seg, 'cfg_batch': cfg_batch, 'packed_tokens': total, 'heads': heads, 'head_dim': head_dim, 'dtype': 'bf16'}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    q = torch.randn(total, heads, head_dim, device=dev, dtype=dt)
    k = torch.randn(total, heads, head_dim, device=dev, dtype=dt)
    v = torch.randn(total, heads, head_dim, device=dev, dtype=dt)
    torch.cuda.synchronize()
    qkv_bytes = torch.cuda.memory_allocated() - base
    rep['qkv_storage_mb'] = _mb(qkv_bytes)
    rep['qkv_storage_note'] = 'IRREDUCIBLE: a fused kernel still needs q,k,v resident'
    sink = torch.randn(heads, device=dev, dtype=torch.float32) if with_sink else None
    torch.cuda.reset_peak_memory_stats()
    pre = torch.cuda.memory_allocated()
    with torch.no_grad():
        out = preview_bidirectional_sdpa(q, k, v, cu, sink=None)
    torch.cuda.synchronize()
    rep['sdpa_only_peak_mb'] = _mb(torch.cuda.max_memory_allocated() - pre)
    rep['sdpa_only_output_mb'] = _mb(torch.cuda.memory_allocated() - pre)
    del out
    torch.cuda.empty_cache()
    if with_sink:
        torch.cuda.reset_peak_memory_stats()
        pre = torch.cuda.memory_allocated()
        with torch.no_grad():
            out = preview_bidirectional_sdpa(q, k, v, cu, sink=sink)
        torch.cuda.synchronize()
        rep['sdpa_with_sink_peak_mb'] = _mb(torch.cuda.max_memory_allocated() - pre)
        del out
        torch.cuda.empty_cache()
        rep['sink_lse_extra_mb'] = round(rep['sdpa_with_sink_peak_mb'] - rep['sdpa_only_peak_mb'], 1)
        rep['sink_lse_note'] = 'RECLAIMABLE by FA2: a varlen flash kernel returns the LSE natively, so the sink becomes an elementwise op on [H,L] instead of this chunked O(chunk*L) fp32 side pass'
    fused_floor = rep['qkv_storage_mb'] + rep.get('sdpa_only_output_mb', 0)
    rep['fused_kernel_floor_mb'] = round(fused_floor, 1)
    rep['reclaimable_mb'] = round(max(0.0, rep.get('sdpa_with_sink_peak_mb', rep['sdpa_only_peak_mb']) - fused_floor), 1)
    rep['reclaimable_note'] = 'peak attention transient above the q/k/v+output floor — the part a true tiled-in-SRAM kernel does not materialise'
    return rep

def _main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--tokens', type=int, default=57820, help='tokens per CFG segment (540p ~57820)')
    p.add_argument('--heads', type=int, default=24)
    p.add_argument('--head-dim', type=int, default=128)
    p.add_argument('--cfg-batch', type=int, default=2, help='2=batched CFG, 1=sequential CFG')
    args = p.parse_args(argv)
    rep = profile_attention(args.tokens, args.heads, args.head_dim, args.cfg_batch)
    print(json.dumps(rep, indent=2))
    return 0
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, '')
    sys.path.insert(0, os.environ.get('MAGI2_REPO_ROOT', ''))
    raise SystemExit(_main())
