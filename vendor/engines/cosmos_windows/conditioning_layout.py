# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/conditioning_layout.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import inspect

def mask_indexes(values):
    if len(values) == 6:
        return (4, 5)
    if len(values) == 11:
        return (5, 6)
    raise ValueError(f'Unsupported Cosmos latent protocol: {len(values)} fields')

def image_latents(prepare, visual, mask, fps):
    if 'action_latents' in inspect.signature(prepare).parameters:
        return (visual, None, None, fps, None, mask, None, None, None, None, None)
    return (visual, None, fps, None, mask, None)
