# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/magi2_parity_gate.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

def _setup_reference(preview_dir: Path):
    from cosmos_windows.weight_streaming.magi2_attention_adapter import install_dependency_shims, install_magi2_attention_intercept
    from cosmos_windows.weight_streaming.magi2_moe_binding import install_moe_kernel_sm120_patch
    install_dependency_shims()
    import inference.model.magi2_preview
    import inference.model.magi2_refiner
    import inference.common.arch
    install_magi2_attention_intercept()
    install_moe_kernel_sm120_patch()

def moe_forward_parity(preview_dir: Path, layer_idx: int, seq_len: int, ring_budget_gb: float, device: str='cuda') -> dict:
    import torch
    from cosmos_windows.weight_streaming.magi2_layer_ring import FileLayerSource, plan_layer_ring
    from cosmos_windows.weight_streaming.magi2_moe_binding import CORE_MOE_PARAMS, bind_core_moe, core_moe_ckpt_name, core_moe_views_from_layer, make_core_moe, read_core_moe_views_direct
    from cosmos_windows.weight_streaming.safetensors_index import build_index
    res: dict = {'check': 'moe_forward_parity', 'layer_idx': layer_idx, 'seq_len': seq_len}
    index = build_index(preview_dir)
    plan = plan_layer_ring(index, vram_budget_bytes=int(ring_budget_gb * 1000000000.0), max_ring_depth=2)
    res['ring_depth'] = plan.ring_depth
    res['slot_gb'] = round(plan.largest_layer_bytes / 1000000000.0, 3)
    direct = read_core_moe_views_direct(preview_dir, layer_idx, device)
    res['direct_dtypes'] = {k: str(v.dtype) for k, v in direct.items()}
    res['direct_shapes'] = {k: tuple(v.shape) for k, v in direct.items()}
    from cosmos_windows.weight_streaming.magi2_layer_ring import Magi2LayerRingStreamer
    streamer = Magi2LayerRingStreamer(plan, device=device, source=FileLayerSource())
    try:
        streamer.new_pass()
        streamer.prefetch(layer_idx)
        layer_views = streamer.wait_layer(layer_idx)
        streamed = core_moe_views_from_layer(layer_views, layer_idx)
        byteq: Dict[str, bool] = {}
        maxabs: Dict[str, float] = {}
        for attr in CORE_MOE_PARAMS:
            d = direct[attr]
            s = streamed[attr].to(d.dtype)
            eq = bool(torch.equal(s, d))
            byteq[attr] = eq
            if not eq:
                maxabs[attr] = float((s.float() - d.float()).abs().max().item())
        res['byte_equal'] = byteq
        res['byte_mismatch_maxabs'] = maxabs
        res['all_byte_equal'] = all(byteq.values())
        torch.manual_seed(1234)
        x = torch.randn(seq_len, 3072, device=device, dtype=torch.bfloat16)
        moe_d = make_core_moe()
        bind_core_moe(moe_d, direct, device)
        with torch.no_grad():
            out_direct = moe_d._forward_impl(x)
            out_direct2 = moe_d._forward_impl(x)
        moe_s = make_core_moe()
        bind_core_moe(moe_s, {k: streamed[k] for k in CORE_MOE_PARAMS}, device)
        with torch.no_grad():
            out_streamed = moe_s._forward_impl(x)
        torch.cuda.synchronize()
        envelope = float((out_direct.float() - out_direct2.float()).abs().max().item())
        diff = (out_direct.float() - out_streamed.float()).abs()
        res['forward_out_shape'] = tuple(out_direct.shape)
        res['forward_max_abs_diff'] = float(diff.max().item())
        res['forward_mean_abs_diff'] = float(diff.mean().item())
        res['kernel_nondeterminism_envelope'] = envelope
        res['forward_within_envelope'] = bool(res['forward_max_abs_diff'] <= max(envelope * 2.0, 0.001))
        res['forward_bit_exact'] = bool(torch.equal(out_direct, out_streamed))
        res['forward_finite'] = bool(torch.isfinite(out_streamed).all())
        res['passed'] = res['all_byte_equal'] and res['forward_within_envelope'] and res['forward_finite']
    finally:
        streamer.close()
    return res

def dense_byte_parity(preview_dir: Path, layer_idx: int, ring_budget_gb: float, device: str='cuda') -> dict:
    import torch
    from safetensors import safe_open
    from cosmos_windows.weight_streaming.magi2_layer_ring import FileLayerSource, Magi2LayerRingStreamer, plan_layer_ring
    from cosmos_windows.weight_streaming.safetensors_index import build_index
    res: dict = {'check': 'dense_byte_parity', 'layer_idx': layer_idx}
    index = build_index(preview_dir)
    plan = plan_layer_ring(index, vram_budget_bytes=int(ring_budget_gb * 1000000000.0), max_ring_depth=2)
    layer_refs, _ = index.group_by_layer('')
    if layer_idx not in layer_refs:
        raise ValueError(f'layer {layer_idx} not in index')
    refs = layer_refs[layer_idx]
    res['tensor_count'] = len(refs)
    res['is_moe'] = any(('moe_mlp.W_gate' in r.name for r in refs))
    streamer = Magi2LayerRingStreamer(plan, device=device, source=FileLayerSource())
    try:
        streamer.new_pass()
        streamer.prefetch(layer_idx)
        views = streamer.wait_layer(layer_idx)
        eqs: Dict[str, bool] = {}
        for r in refs:
            with safe_open(r.file, framework='pt', device=device) as f:
                d = f.get_tensor(r.name)
            s = views[r.name].to(d.dtype)
            eqs[r.name] = bool(torch.equal(s, d.reshape(s.shape) if s.shape != d.shape else d))
        res['all_byte_equal'] = all(eqs.values())
        res['mismatches'] = [n for n, ok in eqs.items() if not ok]
        res['passed'] = res['all_byte_equal']
    finally:
        streamer.close()
    return res

def attention_adapter_parity(seq_len: int, num_heads: int=24, head_dim: int=128, device: str='cuda', dtype_str: str='float32') -> dict:
    import torch
    import torch.nn.functional as F
    from cosmos_windows.weight_streaming.magi2_attention_adapter import preview_bidirectional_sdpa
    res: dict = {'check': 'attention_adapter_parity', 'seq_len': seq_len, 'num_heads': num_heads, 'head_dim': head_dim, 'dtype': dtype_str}
    dtype = getattr(torch, dtype_str)
    torch.manual_seed(7)
    a = seq_len // 2
    cu = [0, a, seq_len]
    q = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    sink = torch.randn(num_heads, device=device, dtype=torch.float32) * 0.5
    gate = torch.randn(seq_len, num_heads, device=device, dtype=dtype)
    scale = head_dim ** (-0.5)
    out = preview_bidirectional_sdpa(q, k, v, cu, sink=sink, gate=gate, scale=scale)
    ref = torch.empty_like(q, dtype=torch.float32)
    for s0, s1 in [(0, a), (a, seq_len)]:
        qs = q[s0:s1].float()
        ks = k[s0:s1].float()
        vs = v[s0:s1].float()
        scores = torch.einsum('lhd,mhd->hlm', qs, ks) * scale
        sink_col = sink.view(num_heads, 1, 1).expand(num_heads, s1 - s0, 1)
        scores_aug = torch.cat([scores, sink_col], dim=-1)
        p = torch.softmax(scores_aug, dim=-1)[..., :s1 - s0]
        o = torch.einsum('hlm,mhd->lhd', p, vs)
        ref[s0:s1] = o
    ref = ref * torch.sigmoid(gate.float()).unsqueeze(-1)
    diff = (out.float() - ref).abs()
    res['max_abs_diff'] = float(diff.max().item())
    res['mean_abs_diff'] = float(diff.mean().item())
    denom = ref.abs().max().item() or 1.0
    res['max_rel_diff'] = float(diff.max().item() / denom)
    tol = 0.0003 if dtype_str == 'float32' else 0.03
    res['tol'] = tol
    res['passed'] = bool(res['max_abs_diff'] <= tol) and bool(torch.isfinite(out).all())
    return res

def run_gate(preview_dir: Path, *, moe_layer: int=2, dense_layer: int=0, seq_len: int=4096, attn_seq: int=2048, ring_budget_gb: float=14.0) -> dict:
    device = 'cuda'
    import torch
    if not torch.cuda.is_available():
        return {'error': 'CUDA unavailable'}
    torch.cuda.reset_peak_memory_stats()
    report: dict = {'device': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0)), 'preview_dir': str(preview_dir)}
    t0 = time.time()
    checks: List[dict] = []
    checks.append(moe_forward_parity(preview_dir, moe_layer, seq_len, ring_budget_gb, device))
    checks.append(dense_byte_parity(preview_dir, dense_layer, ring_budget_gb, device))
    checks.append(attention_adapter_parity(attn_seq, device=device, dtype_str='float32'))
    checks.append(attention_adapter_parity(attn_seq, device=device, dtype_str='bfloat16'))
    report['checks'] = checks
    report['peak_vram_gb'] = round(torch.cuda.max_memory_allocated() / 1000000000.0, 3)
    report['seconds'] = round(time.time() - t0, 1)
    report['passed'] = all((c.get('passed') for c in checks))
    return report

def _main(argv: Optional[List[str]]=None) -> int:
    p = argparse.ArgumentParser(description='MAGI-2 single-GPU correctness gate')
    p.add_argument('--preview-dir', type=Path, default=Path(''))
    p.add_argument('--moe-layer', type=int, default=2)
    p.add_argument('--dense-layer', type=int, default=0)
    p.add_argument('--seq-len', type=int, default=4096)
    p.add_argument('--attn-seq', type=int, default=2048)
    p.add_argument('--ring-budget-gb', type=float, default=14.0)
    p.add_argument('--json', action='store_true')
    args = p.parse_args(argv)
    _setup_reference(args.preview_dir)
    report = run_gate(args.preview_dir, moe_layer=args.moe_layer, dense_layer=args.dense_layer, seq_len=args.seq_len, attn_seq=args.attn_seq, ring_budget_gb=args.ring_budget_gb)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get('passed') else 1
if __name__ == '__main__':
    sys.path.insert(0, '')
    if os.environ.get('MAGI2_REPO_ROOT'):
        sys.path.insert(0, os.environ['MAGI2_REPO_ROOT'])
    else:
        sys.path.insert(0, '')
    raise SystemExit(_main())
