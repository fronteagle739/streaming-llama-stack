# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/optim_factory.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from typing import Iterable, List, Tuple
__all__ = ['build_optimizer', 'MultiPhaseLR']

def build_optimizer(trainable: Iterable, name: str='adamw', lr: float=0.0001):
    import torch
    params = list(trainable)
    key = (name or 'adamw').strip().lower()
    if key in ('adafactor', 'adafactor8', 'ada-factor'):
        try:
            from transformers.optimization import Adafactor
            opt = Adafactor(params, lr=lr, scale_parameter=False, relative_step=False, warmup_init=False)
            print(f'[optim] transformers.Adafactor (scale_parameter=False, relative_step=False) lr={lr}', flush=True)
            return opt
        except Exception as e:
            try:
                TorchAdafactor = torch.optim.Adafactor
            except AttributeError:
                raise RuntimeError(f'adafactor requested but neither transformers.optimization.Adafactor nor torch.optim.Adafactor is available ({type(e).__name__}: {e})') from e
            opt = TorchAdafactor(params, lr=lr)
            print(f'[optim] torch.optim.Adafactor (transformers fallback: {type(e).__name__}) lr={lr}', flush=True)
            return opt
    if key not in ('adamw', 'adam', ''):
        print(f"[optim] unknown optimizer '{name}' -> defaulting to AdamW", flush=True)
    return torch.optim.AdamW(params, lr=lr)

def _parse_phases(spec: str, default_lr: float) -> List[Tuple[int, float]]:
    spec = (spec or '').strip()
    if not spec:
        return [(0, float(default_lr))]
    phases: List[Tuple[int, float]] = []
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '@' in chunk:
            lr_s, step_s = chunk.split('@', 1)
            start = int(float(step_s.strip()))
        else:
            lr_s, start = (chunk, 0)
        phases.append((start, float(lr_s.strip())))
    if not phases:
        return [(0, float(default_lr))]
    phases.sort(key=lambda p: p[0])
    if phases[0][0] != 0:
        phases.insert(0, (0, float(default_lr)))
    return phases

class MultiPhaseLR:

    def __init__(self, optimizer, spec: str='', default_lr: float=0.0001):
        self.optimizer = optimizer
        self.phases = _parse_phases(spec, default_lr)
        self._active = -1
        self.spec = (spec or '').strip()

    @property
    def is_multiphase(self) -> bool:
        return len(self.phases) > 1

    def _phase_for(self, step_idx: int) -> int:
        idx = 0
        for i, (start, _lr) in enumerate(self.phases):
            if step_idx >= start:
                idx = i
            else:
                break
        return idx

    def current_lr(self) -> float:
        return self.phases[max(self._active, 0)][1]

    def step(self, step_idx: int) -> float:
        idx = self._phase_for(step_idx)
        lr = self.phases[idx][1]
        if idx != self._active:
            for g in self.optimizer.param_groups:
                g['lr'] = lr
            if self._active >= 0:
                prev = self.phases[self._active][1]
                print(f'[optim] LR phase transition @ step {step_idx}: {prev:g} -> {lr:g} (phase {idx + 1}/{len(self.phases)})', flush=True)
            else:
                print(f'[optim] LR phase {idx + 1}/{len(self.phases)} active @ step {step_idx}: lr={lr:g}' + (f'  (phases={self.phases})' if self.is_multiphase else ''), flush=True)
            self._active = idx
        return lr
if __name__ == '__main__':
    assert _parse_phases('0.00006@0,0.00003@14000', 0.0001) == [(0, 6e-05), (14000, 3e-05)]
    assert _parse_phases('', 0.0001) == [(0, 0.0001)]
    assert _parse_phases('3e-4@5000', 0.0001) == [(0, 0.0001), (5000, 0.0003)]
    assert _parse_phases('3e-4', 0.0001) == [(0, 0.0003)]
    print("parse '0.00006@0,0.00003@14000' ->", _parse_phases('0.00006@0,0.00003@14000', 0.0001))
    print("parse '' ->", _parse_phases('', 0.0001))
    print("parse '3e-4@5000' ->", _parse_phases('3e-4@5000', 0.0001))
    import torch
    import torch.nn as nn
    lin = nn.Linear(4, 4)
    opt_aw = build_optimizer(lin.parameters(), 'adamw', lr=6e-05)
    assert isinstance(opt_aw, torch.optim.AdamW), type(opt_aw)
    assert abs(opt_aw.param_groups[0]['lr'] - 6e-05) < 1e-12
    opt_af = build_optimizer(lin.parameters(), 'adafactor', lr=6e-05)
    assert 'Adafactor' in type(opt_af).__name__, type(opt_af)
    boundary = 2
    sched = MultiPhaseLR(opt_af, f'0.00006@0,0.00003@{boundary}', default_lr=6e-05)
    seen = []
    for step in range(3):
        lr = sched.step(step)
        loss = lin(torch.randn(2, 4)).pow(2).mean()
        opt_af.zero_grad(set_to_none=True)
        loss.backward()
        opt_af.step()
        seen.append(opt_af.param_groups[0]['lr'])
    print('lr per step:', seen)
    assert abs(seen[0] - 6e-05) < 1e-12 and abs(seen[1] - 6e-05) < 1e-12, seen
    assert abs(seen[2] - 3e-05) < 1e-12, seen
    assert seen[0] != seen[2], 'lr did not change across the phase boundary'
    print('optim_factory self-check PASSED')
