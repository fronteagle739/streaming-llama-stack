#!/usr/bin/env python3
"""Weight-free unit tests for the mlx-vlm SwiReasoning state machine.

Loads swi_state.py DIRECTLY by path (not through the mlx_vlm package __init__,
which imports mlx) so it runs anywhere python does.

Run:  python test_swi_state.py   (or pytest)
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "swi_state_standalone", os.path.join(_HERE, "swi_state.py")
)
swi_state = importlib.util.module_from_spec(_spec)
# Register before exec so dataclass forward-ref (Optional[int]) resolution works.
sys.modules[_spec.name] = swi_state
_spec.loader.exec_module(swi_state)

SwiConfig = swi_state.SwiConfig
SwiState = swi_state.SwiState
entropy_from_probs = swi_state.entropy_from_probs
swi_alpha = swi_state.swi_alpha
swi_beta = swi_state.swi_beta


def test_config_defaults_and_clamp():
    c = SwiConfig.from_dict(None)
    assert c.enabled is False and c.alpha_0 == 1.0 and c.window_size == 512
    c2 = SwiConfig.from_dict({"enabled": True, "alpha_0": 9, "window_size": -3,
                              "max_switch_count": 0})
    assert c2.alpha_0 == 1.0 and c2.window_size == 1 and c2.max_switch_count is None


def test_entropy_and_schedule():
    assert abs(entropy_from_probs([0.25] * 4) - math.log(4)) < 1e-9
    assert swi_alpha(1.0, 5, 100) == 1.0
    assert abs(swi_beta(0.7, 100, 100) - 1.0) < 1e-9


def test_step0_then_switches():
    s = SwiState(cfg=SwiConfig(enabled=True, window_size=1))
    s.update_mode(5.0)                 # step0: ref set, latent
    assert s.mode == 0 and s.should_use_soft()
    s.update_mode(4.0)                 # H<ref -> explicit
    assert s.mode == 1 and s.last_to_normal
    s.update_mode(6.0)                 # H>ref, window met -> back to soft
    assert s.mode == 0 and s.last_to_soft


def test_locking():
    s = SwiState(cfg=SwiConfig(enabled=True, window_size=1), endthink_ids=[42])
    s.update_mode(5.0); s.update_mode(4.0)
    s.note_emitted(42)
    assert s.locked_normal and not s.should_use_soft()
    s.update_mode(9.0); s.update_mode(9.0)
    assert s.mode == 1


def test_budget_two_stage():
    cfg = SwiConfig(enabled=True, window_size=1, max_switch_count=1,
                    termination_max_tokens=3)
    s = SwiState(cfg=cfg, convergence_ids=[10], termination_ids=[20, 21])
    s.update_mode(10.0)                              # step0
    s.update_mode(9.0); s.inject_triggers(1)         # sw=1 >= cap -> convergence
    assert s.switch_count == 1 and s.inject_queue == [10]
    assert s.pop_forced() == 10 and s.pop_forced() is None
    # push over 2*cap via soft/normal cycles
    st = 2
    for drop in (8.0, 7.0, 6.0):
        s.update_mode(12.0); s.inject_triggers(st); st += 1   # to_soft
        s.update_mode(drop); s.inject_triggers(st); st += 1   # to_normal
    assert s.switch_count > 2 and s.inject_queue == [20, 21] and s.answer_budget == 3
    assert s.budget_tick() is False and s.budget_tick() is False and s.budget_tick() is True


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} mlx-vlm swi_state tests passed")


if __name__ == "__main__":
    _run_all()
