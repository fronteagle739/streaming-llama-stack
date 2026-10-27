from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class SwiConfig:
    enabled: bool = False
    alpha_0: float = 1.0
    beta_0: float = 0.7
    window_size: int = 512
    max_switch_count: Optional[int] = None
    termination_max_tokens: int = 32

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> 'SwiConfig':
        if not isinstance(d, dict):
            return cls()

        def _f(key: str, default: float, lo: float, hi: float) -> float:
            try:
                v = float(d.get(key, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        def _i(key: str, default: int, lo: int) -> int:
            try:
                v = int(d.get(key, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, v)
        msc_raw = d.get('max_switch_count', None)
        max_switch_count: Optional[int] = None
        if msc_raw is not None:
            try:
                msc = int(msc_raw)
                if msc >= 1:
                    max_switch_count = msc
            except (TypeError, ValueError):
                max_switch_count = None
        return cls(enabled=bool(d.get('enabled', False)), alpha_0=_f('alpha_0', 1.0, 0.0, 1.0), beta_0=_f('beta_0', 0.7, 0.0, 1.0), window_size=_i('window_size', 512, 1), max_switch_count=max_switch_count, termination_max_tokens=_i('termination_max_tokens', 32, 1))

    def to_public_dict(self) -> dict:
        return {'enabled': self.enabled, 'alpha_0': self.alpha_0, 'beta_0': self.beta_0, 'window_size': self.window_size, 'max_switch_count': self.max_switch_count, 'termination_max_tokens': self.termination_max_tokens}

def swi_alpha(alpha_0: float, step: int, n_predict: int) -> float:
    denom = float(max(1, n_predict))
    return alpha_0 + (1.0 - alpha_0) * (float(step) / denom)

def swi_beta(beta_0: float, step: int, n_predict: int) -> float:
    denom = float(max(1, n_predict))
    return beta_0 + (1.0 - beta_0) * (float(step) / denom)

def entropy_from_probs(probs: List[float], clamp: float=1e-12) -> float:
    import math
    h = 0.0
    for p in probs:
        pc = p if p > clamp else clamp
        h -= p * math.log(pc)
    return h

@dataclass
class SwiState:
    cfg: SwiConfig = field(default_factory=SwiConfig)
    endthink_ids: List[int] = field(default_factory=list)
    convergence_ids: List[int] = field(default_factory=list)
    termination_ids: List[int] = field(default_factory=list)
    mode: int = 0
    mode_stay_steps: int = 0
    ref_entropy: float = 0.0
    switch_count: int = 0
    locked_normal: bool = False
    first_step: bool = True
    last_to_soft: bool = False
    last_to_normal: bool = False
    inject_queue: List[int] = field(default_factory=list)
    answer_budget: int = -1
    soft_steps: int = 0
    normal_steps: int = 0

    def update_mode(self, cur_entropy: float) -> None:
        self.last_to_soft = False
        self.last_to_normal = False
        if self.first_step:
            self.ref_entropy = cur_entropy
            self.first_step = False
            return
        self.mode_stay_steps += 1
        allow_switch = self.mode_stay_steps >= self.cfg.window_size
        to_normal = self.mode == 0 and cur_entropy < self.ref_entropy
        to_soft = self.mode == 1 and cur_entropy > self.ref_entropy and allow_switch and (not self.locked_normal)
        if to_normal:
            self.mode = 1
            self.last_to_normal = True
            if self.cfg.max_switch_count is not None and self.cfg.max_switch_count > 0:
                self.switch_count += 1
        if to_soft:
            self.mode = 0
            self.last_to_soft = True
        if to_normal or to_soft:
            self.mode_stay_steps = 0
            self.ref_entropy = cur_entropy

    def should_use_soft(self) -> bool:
        return self.mode == 0 and (not self.locked_normal)

    def inject_triggers(self, step: int) -> None:
        msc = self.cfg.max_switch_count
        if not msc or msc <= 0 or step <= 0 or (not self.last_to_normal):
            return
        if self.switch_count > 2 * msc:
            self.inject_queue = list(self.termination_ids)
            self.answer_budget = int(self.cfg.termination_max_tokens)
        elif self.switch_count >= msc and self.answer_budget < 0:
            self.inject_queue = list(self.convergence_ids)

    def pop_forced(self) -> Optional[int]:
        if self.inject_queue:
            return self.inject_queue.pop(0)
        return None

    def note_emitted(self, token_id: int) -> None:
        if self.endthink_ids and token_id in self.endthink_ids:
            self.locked_normal = True

    def force_lock(self) -> None:
        self.locked_normal = True

    def budget_tick(self) -> bool:
        if self.answer_budget >= 0:
            self.answer_budget -= 1
            if self.answer_budget == 0:
                return True
        return False

    def count_step(self, latent: bool) -> None:
        if latent:
            self.soft_steps += 1
        else:
            self.normal_steps += 1

    def stats(self) -> dict:
        return {'soft_steps': self.soft_steps, 'normal_steps': self.normal_steps, 'switch_count': self.switch_count, 'final_mode': 'soft' if self.mode == 0 else 'normal', 'locked_normal': self.locked_normal}
