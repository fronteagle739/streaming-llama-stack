# SPDX-License-Identifier: MIT
"""Cost-feedback admission control for the DSpark round loop (v41 port, lever 3).

Ported from the shape of ds4-v41-m3ultra's controller (CHANGES-V41.md M17/M21/M22), which itself
ported GLM-5.3-Flash's DFlash2 controller:

* M17 -- a WINDOW of ``window_attempts`` drafted rounds is judged against the median cost of the
  last ``serial_samples`` serial tokens. A losing window backs the drafter off for
  ``backoff_ladder[i]`` serial tokens (4 / 8 / 16 / 32 by default, climbing while windows keep losing);
  a winning window resets the ladder. Serial rounds keep refreshing the median, so the judge
  follows the context length the serial step actually costs.
* M21 -- confidence admission: the proposal is the confident prefix (``p_min`` on the drafter's
  confidence head); a prefix shorter than ``min_draft`` DECLINES the cycle before the verify.
* M22 -- a decline is a chosen call: it takes a window slot at its drafter cost and commits nothing,
  so three declines close a losing window.

Pure Python (no MLX) so the policy is unit-tested on any box; the round loop feeds it wall-clock
milliseconds and committed-token counts and asks ``should_draft()`` before each round.
"""
from __future__ import annotations

import os
from collections import deque
from statistics import median
from typing import Deque, List, Tuple


class DSparkCostController:
    # Defaults are the measured best on OUR drafter (2026-09-15 runs 2-4, hybrid receiver, 5.8k prose /
    # code): admission OFF (p_min 0, min_draft 1 -- the v41 values 0.75 / 3 over-declined 40 of 79 code
    # attempts because our confidence head is calibrated differently), a 6-round window and a 4/8/16/32
    # ladder (3-round windows with 16+ lost 10-17 % on code to noise). Result: prose 20.4 / 19.9 tok/s
    # (serial 20.2 / 19.6), code 33.7 / 31.7 (fixed block 31.1 / 32.3).
    def __init__(self, *, serial_samples: int = 9, window_attempts: int = 6,
                 backoff_ladder: Tuple[int, ...] = (4, 8, 16, 32), p_min: float = 0.0,
                 min_draft: int = 1, losing_declines: int = 3) -> None:
        if serial_samples < 1 or window_attempts < 1 or not backoff_ladder or min_draft < 1:
            raise ValueError("DSparkCostController: samples, attempts, ladder and min_draft must be >= 1")
        if not 0.0 <= p_min <= 1.0:
            raise ValueError("DSparkCostController: p_min must be in [0, 1]")
        self.serial_samples = int(serial_samples)
        self.window_attempts = int(window_attempts)
        self.backoff_ladder = tuple(int(x) for x in backoff_ladder)
        self.p_min = float(p_min)
        self.min_draft = int(min_draft)
        self.losing_declines = int(losing_declines)
        self.serial_ms: Deque[float] = deque(maxlen=self.serial_samples)
        self.window: List[Tuple[str, int, float]] = []
        self.backoff_remaining = 0
        self.ladder_index = 0
        self.stats = {"serial_rounds": 0, "attempts": 0, "declines": 0, "windows_won": 0,
                      "windows_lost": 0, "backoff_tokens": 0, "drafted_committed": 0,
                      "serial_committed": 0, "drafted_ms": 0.0, "serial_ms": 0.0}

    @classmethod
    def from_env(cls, env=os.environ) -> "DSparkCostController":
        ladder = tuple(int(x) for x in str(env.get("DSV41_DSPARK_BACKOFF", "4,8,16,32")).split(",") if x.strip())
        return cls(serial_samples=int(env.get("DSV41_DSPARK_SERIAL_SAMPLES", "9")),
                   window_attempts=int(env.get("DSV41_DSPARK_WINDOW", "6")),
                   backoff_ladder=ladder,
                   p_min=float(env.get("DSV41_DSPARK_P_MIN", "0")),
                   min_draft=int(env.get("DSV41_DSPARK_MIN_DRAFT", "1")),
                   losing_declines=int(env.get("DSV41_DSPARK_LOSING_DECLINES", "3")))

    # ----- decisions
    def should_draft(self) -> bool:
        """Draft only with a full serial sample behind us and no backoff owed."""
        if self.backoff_remaining > 0:
            return False
        return len(self.serial_ms) >= self.serial_samples

    def admits(self, proposal_len: int) -> bool:
        """M21: a confident prefix shorter than ``min_draft`` declines the cycle."""
        return int(proposal_len) >= self.min_draft

    @property
    def serial_median_ms(self) -> float:
        return float(median(self.serial_ms)) if self.serial_ms else float("inf")

    # ----- observations
    def record_serial(self, ms: float, committed: int = 1) -> None:
        self.serial_ms.append(float(ms) / max(1, int(committed)))
        self.stats["serial_rounds"] += 1
        self.stats["serial_committed"] += int(committed)
        self.stats["serial_ms"] += float(ms)
        if self.backoff_remaining > 0:
            self.backoff_remaining = max(0, self.backoff_remaining - int(committed))

    def record_attempt(self, kind: str, committed: int, ms: float) -> None:
        """``kind`` is 'verify' (a drafted round that was verified) or 'decline' (drafter cost paid,
        nothing committed by it). Closes the window when it is full or has lost on declines."""
        if kind not in ("verify", "decline"):
            raise ValueError(f"unknown attempt kind {kind!r}")
        self.window.append((kind, int(committed), float(ms)))
        self.stats["attempts"] += 1
        self.stats["drafted_ms"] += float(ms)
        if kind == "decline":
            self.stats["declines"] += 1
        else:
            self.stats["drafted_committed"] += int(committed)
        declines = sum(1 for k, _, _ in self.window if k == "decline")
        if len(self.window) >= self.window_attempts or declines >= self.losing_declines:
            self._close_window()

    def _close_window(self) -> None:
        total_ms = sum(ms for _, _, ms in self.window)
        total_committed = sum(c for _, c, _ in self.window)
        cost = total_ms / total_committed if total_committed > 0 else float("inf")
        won = cost < self.serial_median_ms
        self.window = []
        if won:
            self.ladder_index = 0
            self.stats["windows_won"] += 1
            return
        self.backoff_remaining = self.backoff_ladder[self.ladder_index]
        self.stats["backoff_tokens"] += self.backoff_remaining
        self.ladder_index = min(self.ladder_index + 1, len(self.backoff_ladder) - 1)
        self.stats["windows_lost"] += 1

    def summary(self) -> dict:
        out = dict(self.stats)
        out.update(serial_median_ms=None if not self.serial_ms else round(self.serial_median_ms, 3),
                   backoff_remaining=self.backoff_remaining, ladder_index=self.ladder_index,
                   p_min=self.p_min, min_draft=self.min_draft)
        total_tokens = out["drafted_committed"] + out["serial_committed"]
        total_ms = out["drafted_ms"] + out["serial_ms"]
        out["ms_per_token"] = round(total_ms / total_tokens, 3) if total_tokens else None
        return out
