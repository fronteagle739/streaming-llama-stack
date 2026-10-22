# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/validate_bernini_gpu.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import os
import sys
DEFAULT_DIR = ''

def main() -> int:
    import torch
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
    from . import bernini_backend
    snap = os.environ.get('BERNINI_1P3B_DIR', DEFAULT_DIR)
    tf_dir = os.path.join(snap, 'transformer')
    if not os.path.isfile(os.path.join(tf_dir, 'config.json')):
        print(f'FAIL: Bernini transformer not found at {tf_dir}')
        return 1
    if not torch.cuda.is_available():
        print('FAIL: no CUDA device')
        return 1
    dev = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.get_device_properties(i).total_memory)
    device = f'cuda:{dev}'
    torch.cuda.set_device(dev)
    print(f'loading Bernini-R-1.3B transformer on {device} ({torch.cuda.get_device_name(dev)}, {torch.cuda.get_device_properties(dev).total_memory / 1000000000.0:.0f}GB)')
    model = WanTransformer3DModel.from_pretrained(tf_dir, torch_dtype=torch.bfloat16).to(device).eval()
    cfg = model.config
    print(f'loaded: {cfg.num_layers} layers, dim {cfg.attention_head_dim * cfg.num_attention_heads}, in_channels {cfg.in_channels}, patch {cfg.patch_size}')
    bernini_backend.patch_wan_in_context(model)
    B, C = (1, cfg.in_channels)
    T, H, W = (1, 32, 32)
    x = torch.randn(B, C, T, H, W, device=device, dtype=torch.bfloat16)
    ts = torch.tensor([500], device=device)
    text_dim = cfg.text_dim if hasattr(cfg, 'text_dim') else 4096
    enc = torch.randn(B, 16, text_dim, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        base = model(x, ts, enc, return_dict=False)[0]
        ctx = torch.randn(B, C, T, H, W, device=device, dtype=torch.bfloat16)
        out1 = model(x, ts, enc, return_dict=False, context_latents=[ctx])[0]
        refs = [torch.randn(B, C, T, H, W, device=device, dtype=torch.bfloat16) for _ in range(3)]
        out3 = model(x, ts, enc, return_dict=False, context_latents=refs)[0]
    ok = True
    if out1.shape != base.shape:
        print(f'FAIL: 1-context output {tuple(out1.shape)} != base {tuple(base.shape)}')
        ok = False
    if out3.shape != base.shape:
        print(f'FAIL: 3-context output {tuple(out3.shape)} != base {tuple(base.shape)}')
        ok = False
    if torch.allclose(out1.float(), base.float(), atol=0.001):
        print('FAIL: context_latents had no effect on the REAL transformer')
        ok = False
    if torch.allclose(out3.float(), out1.float(), atol=0.001):
        print('FAIL: extra references had no effect')
        ok = False
    peak = torch.cuda.max_memory_allocated(dev) / 1000000000.0
    if ok:
        print(f'BERNINI GPU VALIDATION PASS (real 1.3B transformer; target {tuple(base.shape)}; 1-ctx + 3-ctx multi-ref run + truncate + effective on CUDA; peak {peak:.2f}GB)')
        return 0
    return 1
if __name__ == '__main__':
    sys.exit(main())
