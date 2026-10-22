# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/sr_requant.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from typing import Optional, Tuple
import torch
Q4_0_BLOCK = 32
Q4_0_TYPE = 18

def _q4_0_scales(x_blocks: torch.Tensor) -> torch.Tensor:
    imax = x_blocks.abs().argmax(dim=1, keepdim=True)
    amax = torch.gather(x_blocks, 1, imax)
    d = amax / -8.0
    d = d.to(torch.float16).to(torch.float32)
    return d

def _pack_nibbles_q4_0(q: torch.Tensor) -> torch.Tensor:
    lo = q[:, :16]
    hi = q[:, 16:]
    return (lo | hi << 4).to(torch.uint8)

def _blocks_to_bytes_q4_0(d: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    n = d.shape[0]
    d16 = d.to(torch.float16).view(torch.uint8).reshape(n, 2)
    packed = _pack_nibbles_q4_0(q)
    return torch.cat([d16, packed], dim=1).reshape(-1)

def sr_requantize_q4_0(x: torch.Tensor, generator: Optional[torch.Generator]=None, stochastic: bool=True, d_override: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x.reshape(-1).to(torch.float32)
    n_blocks = x.numel() // Q4_0_BLOCK
    xb = x.view(n_blocks, Q4_0_BLOCK)
    d = d_override if d_override is not None else _q4_0_scales(xb)
    inv = torch.where(d == 0, torch.zeros_like(d), 1.0 / d)
    qf = xb * inv + 8.0
    if stochastic:
        floor = torch.floor(qf)
        frac = qf - floor
        rnd = torch.rand(qf.shape, generator=generator, device=qf.device, dtype=qf.dtype)
        q = floor + (rnd < frac).to(qf.dtype)
    else:
        q = torch.floor(qf + 0.5)
    q = q.clamp(0, 15).to(torch.uint8)
    raw = _blocks_to_bytes_q4_0(d, q)
    return (raw, q)

def dequantize_q4_0_from_q(d: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (d * (q.to(torch.int16) - 8).to(torch.float32)).reshape(-1)

def _scales_from_raw_q4_0(raw: torch.Tensor) -> torch.Tensor:
    n_blocks = raw.numel() // Q4_0_TYPE
    return raw.view(n_blocks, Q4_0_TYPE)[:, :2].contiguous().view(torch.float16).to(torch.float32).reshape(n_blocks, 1)

def merge_q4_0_with_residual(raw_old: torch.Tensor, shape, delta: torch.Tensor, residual: Optional[torch.Tensor]=None, generator: Optional[torch.Generator]=None, stochastic: bool=False, fixed_scale: bool=True) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    from cosmos_windows.gguf_qat.dequant import dequantize_q4_0
    w_old = dequantize_q4_0(raw_old).reshape(-1)
    delta = delta.reshape(-1).to(torch.float32)
    res_in = residual.reshape(-1).to(torch.float32) if residual is not None else torch.zeros_like(w_old)
    target = w_old + delta + res_in
    n_blocks = target.numel() // Q4_0_BLOCK
    if fixed_scale:
        d_new = _scales_from_raw_q4_0(raw_old)
        raw_new, q_new = sr_requantize_q4_0(target, generator=generator, stochastic=stochastic, d_override=d_new)
    else:
        d_new = _q4_0_scales(target.view(n_blocks, Q4_0_BLOCK))
        raw_new, q_new = sr_requantize_q4_0(target, generator=generator, stochastic=stochastic)
    w_new = dequantize_q4_0_from_q(d_new, q_new)
    residual_new = (target - w_new).to(torch.float16).to(torch.float32)
    changed = (raw_new != raw_old).reshape(n_blocks, Q4_0_TYPE).any(dim=1)
    stats = {'n_blocks': int(n_blocks), 'changed_blocks': int(changed.sum().item()), 'zero_delta_blocks': int((~changed).sum().item()), 'mean_abs_residual': float(residual_new.abs().mean().item()), 'weight_moved_l1': float((w_new - w_old).abs().sum().item())}
    return (raw_new, residual_new, stats)
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '')
    from cosmos_windows.gguf_qat.rss_guard import start_rss_guard, peak_rss_gb
    from cosmos_windows.gguf_qat.dequant import dequantize_q4_0
    start_rss_guard(limit_gb=6.0, label='sr_requant_selftest')
    dev = 'cuda'
    g = torch.Generator(device=dev).manual_seed(0)
    torch.manual_seed(1)
    x0 = torch.randn(64 * 32, device=dev) * 0.05
    raw0, _ = sr_requantize_q4_0(x0, stochastic=False)
    w0 = dequantize_q4_0(raw0)
    N = 4000
    acc = torch.zeros_like(w0)
    for _ in range(N):
        raw_i, _ = sr_requantize_q4_0(w0, generator=g, stochastic=True)
        acc += dequantize_q4_0(raw_i)
    mean = acc / N
    bias = mean - w0
    print(f'(A) SR mean-preservation over {N} draws: max|E[requant]-x|={bias.abs().max().item():.3e}  mean|bias|={bias.abs().mean().item():.3e}  (deterministic round-trip is exact by construction)')
    det_raw, _ = sr_requantize_q4_0(w0, stochastic=False)
    det_exact = torch.equal(det_raw, raw0)
    print(f'    deterministic requant(dequant(x)) byte-identical to input pack: {det_exact}')
    raw = raw0.clone()
    w_start = dequantize_q4_0(raw).clone()
    n_blocks = raw.numel() // Q4_0_TYPE
    d_blocks = raw.view(n_blocks, Q4_0_TYPE)[:, :2].contiguous().view(torch.float16).abs().to(torch.float32)
    lsb = d_blocks.mean().item()
    tiny = lsb * 0.12
    steps = 40
    delta_vec = torch.full_like(w_start, tiny)
    raw_no = raw0.clone()
    moved_no = []
    for s in range(steps):
        raw_no, _res, st = merge_q4_0_with_residual(raw_no, None, delta_vec, residual=None, stochastic=False)
        moved_no.append((dequantize_q4_0(raw_no) - w_start).abs().mean().item())
    raw_yes = raw0.clone()
    res = torch.zeros_like(w_start)
    moved_yes = []
    changed_cum = 0
    for s in range(steps):
        raw_yes, res, st = merge_q4_0_with_residual(raw_yes, None, delta_vec, residual=res, stochastic=False)
        changed_cum += st['changed_blocks']
        moved_yes.append((dequantize_q4_0(raw_yes) - w_start).abs().mean().item())
    print(f'(B) sub-LSB delta = {tiny:.3e} (LSB~{lsb:.3e}, {tiny / lsb * 100:.0f}% of LSB), {steps} steps:')
    print(f"    WITHOUT residual: mean|dw| after {steps} steps = {moved_no[-1]:.3e}  (blocks ever changed: {('no' if moved_no[-1] == 0 else 'YES')})")
    print(f'    WITH residual   : mean|dw| after {steps} steps = {moved_yes[-1]:.3e}  (cumulative changed-block events: {changed_cum})')
    print('    delta-accumulation table (step : mean|dw| without | with):')
    for s in range(0, steps, max(1, steps // 10)):
        print(f'      step {s:3d} : {moved_no[s]:.3e} | {moved_yes[s]:.3e}')
    print(f'      step {steps - 1:3d} : {moved_no[-1]:.3e} | {moved_yes[-1]:.3e}')
    ok_money = moved_no[-1] == 0.0 and moved_yes[-1] > 0.0
    ok_mean = bias.abs().max().item() < 0.0005
    print(f'peak RSS {peak_rss_gb():.3f} GB')
    print('SR/RESIDUAL PROOF:', 'PASS' if ok_money and ok_mean and det_exact else 'FAIL', f'(mean_preserve={ok_mean}, money_plot={ok_money}, det_roundtrip={det_exact})')
    sys.exit(0 if ok_money and ok_mean and det_exact else 1)
