# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/magi2_cfg_sequential.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from dataclasses import fields, replace
from typing import Any

def _slice_field(value: Any, i: int):
    import torch
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.dim() >= 1 and value.shape[0] == 2:
            return value[i:i + 1]
        return value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return type(value)([value[i]])
    return value

def slice_model_input(model_input, i: int):
    kwargs = {f.name: _slice_field(getattr(model_input, f.name), i) for f in fields(model_input)}
    return replace(model_input, **kwargs)

def install_sequential_cfg(sampler, emit=None):
    import torch
    if getattr(sampler, '_magi2_sequential_cfg', False):
        return sampler
    orig_forward = sampler.forward

    def _sequential_forward(model_input):
        x = getattr(model_input, 'x_t', None)
        if x is None or not hasattr(x, 'shape') or x.shape[0] != 2:
            return orig_forward(model_input)
        outs = []
        for i in (0, 1):
            out = orig_forward(slice_model_input(model_input, i))
            outs.append(out if isinstance(out, tuple) else (out,))
            torch.cuda.empty_cache()
        merged = tuple((torch.cat([o[k] for o in outs], dim=0) for k in range(len(outs[0]))))
        return merged if len(merged) > 1 else merged[0]
    sampler.forward = _sequential_forward
    sampler._magi2_sequential_cfg = True
    msg = 'sequential CFG active: cond/uncond run one after the other (identical math, ~half the activation peak, ~2x step time)'
    if emit is not None:
        emit(event='stage_log', stage='cfg_mode', message=msg, cfg_mode='sequential')
    else:
        print(f'[MAGI2-CFG] {msg}', flush=True)
    return sampler
