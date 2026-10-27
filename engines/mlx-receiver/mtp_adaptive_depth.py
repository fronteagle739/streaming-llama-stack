from __future__ import annotations
from dataclasses import dataclass

@dataclass
class AdaptiveDepthController:
    start_depth: int
    min_depth: int = 1
    max_depth: int = 8
    increase_after: int = 3
    decrease_below: float = 1.5
    target_only_below: float = 0.0
    target_only_after: int = 2
    reprobe_ladder: tuple = (16, 32, 64, 128)

    def __post_init__(self) -> None:
        self.min_depth = max(0, int(self.min_depth))
        self.max_depth = max(self.min_depth, int(self.max_depth))
        self._depth = self._clamp(int(self.start_depth))
        self._full_accept_streak = 0
        self._starved_streak = 0
        self.reprobe_ladder = tuple((max(1, int(x)) for x in self.reprobe_ladder or ())) or (16,)
        self._reprobe_index = 0
        self._target_only_steps = 0
        self.reprobes = 0

    def _clamp(self, d: int) -> int:
        return max(self.min_depth, min(self.max_depth, int(d)))

    @property
    def depth(self) -> int:
        return self._depth

    def next_depth(self) -> int:
        if self._depth == 0:
            self._target_only_steps += 1
            if self._target_only_steps >= self.reprobe_ladder[self._reprobe_index]:
                self._target_only_steps = 0
                self._reprobe_index = min(self._reprobe_index + 1, len(self.reprobe_ladder) - 1)
                self._depth = max(1, self.min_depth)
                self._starved_streak = 0
                self._full_accept_streak = 0
                self.reprobes += 1
        return self._depth

    def observe(self, accepted_count: int, draft_count: int) -> int:
        accepted = max(0, int(accepted_count))
        drafted = max(0, int(draft_count))
        if drafted <= 0:
            return self._depth
        if accepted >= drafted:
            self._full_accept_streak += 1
            self._starved_streak = 0
            self._reprobe_index = 0
            if self._full_accept_streak >= self.increase_after and self._depth < self.max_depth:
                self._depth = self._clamp(self._depth + 1)
                self._full_accept_streak = 0
        elif accepted < self.decrease_below:
            self._full_accept_streak = 0
            self._starved_streak += 1
            if self._depth > self.min_depth:
                self._depth = self._clamp(self._depth - 1)
            elif self.target_only_below > 0 and accepted < self.target_only_below and (self._starved_streak >= self.target_only_after):
                self._depth = 0
        else:
            self._full_accept_streak = 0
            self._starved_streak = 0
        return self._depth

def make_adaptive_depth_controller(policy: str, max_draft: int, *, min_depth: int=1) -> AdaptiveDepthController | None:
    normalized = str(policy or 'off').strip().lower().replace('_', '-')
    if normalized in ('', 'off', 'none', 'fixed', 'static'):
        return None
    max_draft = max(1, int(max_draft))
    if normalized in ('streak', 'adaptive', 'auto', 'on'):
        return AdaptiveDepthController(start_depth=max_draft, min_depth=min_depth, max_depth=max_draft)
    if normalized in ('streak-floor', 'floor', 'target-only-floor'):
        return AdaptiveDepthController(start_depth=max_draft, min_depth=min_depth, max_depth=max_draft, target_only_below=0.5, target_only_after=2)
    raise ValueError(f'unsupported MTP adaptive-depth policy {policy!r}; expected off, streak, or streak-floor')
