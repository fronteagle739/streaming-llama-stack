# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/sdedit.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations

def flow_start_index(num_steps: int, strength: float) -> int:
    n = int(num_steps)
    if n <= 0:
        return 0
    s = max(0.0, min(1.0, float(strength)))
    idx = int(round((1.0 - s) * n))
    return max(0, min(n - 1, idx))

def flow_shift_sigma(raw_sigma: float, shift: float) -> float:
    s = float(shift)
    r = float(raw_sigma)
    if s == 1.0:
        return r
    return s * r / (1.0 + (s - 1.0) * r)

def flow_noise_latent(x0, noise, sigma_start):
    sig = float(sigma_start)
    return (1.0 - sig) * x0 + sig * noise
