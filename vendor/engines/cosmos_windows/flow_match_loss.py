# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/flow_match_loss.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import torch

def sample_sigma_t(bsz: int, device='cpu', distribution: str='logitnormal', loc: float=0.0, scale: float=1.0, generator: torch.Generator | None=None) -> torch.Tensor:
    if distribution == 'logitnormal':
        n = torch.randn(bsz, device=device, generator=generator) * scale + loc
        return torch.sigmoid(n)
    return torch.rand(bsz, device=device, generator=generator)

def _bcast(sigma_t: torch.Tensor, ndim: int) -> torch.Tensor:
    while sigma_t.ndim < ndim:
        sigma_t = sigma_t.unsqueeze(-1)
    return sigma_t

def rectified_flow_xt(clean: torch.Tensor, noise: torch.Tensor, sigma_t: torch.Tensor, cond_mask: torch.Tensor | None=None) -> torch.Tensor:
    s = _bcast(sigma_t, clean.ndim)
    xt = noise * s + clean * (1.0 - s)
    if cond_mask is not None:
        xt = clean * cond_mask + xt * (1.0 - cond_mask)
    return xt

def velocity_target(clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return noise - clean

def flow_match_loss(pred_velocity: torch.Tensor, clean: torch.Tensor, noise: torch.Tensor, cond_mask: torch.Tensor | None=None) -> torch.Tensor:
    target = velocity_target(clean, noise).float()
    pred = pred_velocity.float()
    if cond_mask is None:
        return torch.nn.functional.mse_loss(pred, target)
    keep = (1.0 - cond_mask).float()
    se = (pred - target) ** 2 * keep
    denom = keep.sum().clamp_min(1.0) * (pred.numel() / max(keep.numel(), 1))
    kept_elems = keep.expand_as(se).sum().clamp_min(1.0)
    return se.sum() / kept_elems

def _selftest():
    g = torch.Generator().manual_seed(0)
    B, F, C = (2, 5, 8)
    clean = torch.randn(B, F, C, generator=g)
    noise = torch.randn(B, F, C, generator=g)
    ok = True
    x0 = rectified_flow_xt(clean, noise, torch.zeros(B))
    x1 = rectified_flow_xt(clean, noise, torch.ones(B))
    ok &= torch.allclose(x0, clean, atol=1e-06)
    ok &= torch.allclose(x1, noise, atol=1e-06)
    print(f'  [interp] sigma0==clean:{torch.allclose(x0, clean, atol=1e-06)} sigma1==noise:{torch.allclose(x1, noise, atol=1e-06)}')
    ok &= torch.allclose(velocity_target(clean, noise), noise - clean)
    cond = torch.zeros(B, F, 1)
    cond[:, 0] = 1.0
    xt = rectified_flow_xt(clean, noise, torch.full((B,), 0.7), cond_mask=cond)
    anchored = torch.allclose(xt[:, 0], clean[:, 0], atol=1e-06)
    ok &= anchored
    print(f'  [mask] frame0 anchored to clean: {anchored}')
    target = velocity_target(clean, noise)
    ok &= flow_match_loss(target, clean, noise).item() < 1e-10
    bad = target.clone()
    bad[:, 0] += 1000.0
    masked = flow_match_loss(bad, clean, noise, cond_mask=cond).item()
    ok &= masked < 1e-06
    print(f'  [loss] pred==target ->0: {flow_match_loss(target, clean, noise).item():.2e} | err-on-conditioned-frame masked: {masked:.2e}')
    s = sample_sigma_t(10000, distribution='logitnormal', generator=torch.Generator().manual_seed(1))
    ok &= s.min() > 0 and s.max() < 1 and (0.3 < s.mean() < 0.7)
    print(f'  [sigma] logitnormal in (0,1), mean={s.mean():.3f} min={s.min():.3f} max={s.max():.3f}')
    print('flow_match_loss self-test:', 'PASS' if ok else 'FAIL')
    return ok
if __name__ == '__main__':
    import sys
    sys.exit(0 if _selftest() else 1)
