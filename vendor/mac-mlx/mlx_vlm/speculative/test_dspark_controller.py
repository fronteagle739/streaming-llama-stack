"""The DSpark cost-feedback controller (v41 lever 3) is pure policy: no MLX, runs on any box.
    pytest mlx_vlm/speculative/test_dspark_controller.py
"""
import importlib.util
import os

import pytest

_PATH = os.path.join(os.path.dirname(__file__), "dspark_controller.py")
_SPEC = importlib.util.spec_from_file_location("dspark_controller", _PATH)
ctl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ctl)
C = ctl.DSparkCostController


def _serial(c, n, ms=50.0):
    for _ in range(n):
        c.record_serial(ms)


def test_needs_a_full_serial_sample_before_drafting():
    c = C(serial_samples=9)
    assert not c.should_draft()
    _serial(c, 8)
    assert not c.should_draft()
    _serial(c, 1)
    assert c.should_draft() and c.serial_median_ms == 50.0


def test_a_losing_window_backs_off_up_the_ladder_and_a_win_resets_it():
    c = C(serial_samples=3, window_attempts=3, backoff_ladder=(4, 8, 16))
    _serial(c, 3, 50.0)
    # three drafted rounds that each cost 120 ms for 2 tokens = 60 ms/token > 50 serial: lose
    for _ in range(3):
        c.record_attempt("verify", 2, 120.0)
    assert not c.should_draft() and c.backoff_remaining == 4 and c.stats["windows_lost"] == 1
    _serial(c, 4, 50.0)
    assert c.should_draft()
    for _ in range(3):
        c.record_attempt("verify", 2, 120.0)
    assert c.backoff_remaining == 8                       # the ladder climbs
    _serial(c, 8)
    for _ in range(3):
        c.record_attempt("verify", 4, 100.0)              # 25 ms/token < 50: win
    assert c.should_draft() and c.ladder_index == 0 and c.stats["windows_won"] == 1
    for _ in range(3):
        c.record_attempt("verify", 2, 120.0)
    assert c.backoff_remaining == 4                       # reset ladder starts from the bottom again


def test_the_ladder_saturates_at_its_top_rung():
    c = C(serial_samples=1, window_attempts=1, backoff_ladder=(2, 4))
    _serial(c, 1, 10.0)
    for expect in (2, 4, 4, 4):
        c.record_attempt("verify", 1, 100.0)
        assert c.backoff_remaining == expect
        _serial(c, expect, 10.0)


def test_three_declines_close_a_losing_window_priced_at_drafter_cost():
    c = C(serial_samples=1, window_attempts=5, losing_declines=3, backoff_ladder=(16,))
    _serial(c, 1, 50.0)
    c.record_attempt("decline", 0, 12.0)
    c.record_attempt("decline", 0, 12.0)
    assert c.should_draft()                               # two declines: still open
    c.record_attempt("decline", 0, 12.0)
    assert not c.should_draft() and c.backoff_remaining == 16
    assert c.stats["declines"] == 3 and c.stats["drafted_ms"] == 36.0


def test_declines_and_verifies_share_one_window():
    c = C(serial_samples=1, window_attempts=3, backoff_ladder=(16,))
    _serial(c, 1, 50.0)
    c.record_attempt("decline", 0, 12.0)                  # cost, no tokens
    c.record_attempt("verify", 5, 80.0)
    c.record_attempt("verify", 5, 80.0)                   # (12+80+80)/10 = 17.2 ms/token < 50: win
    assert c.should_draft() and c.stats["windows_won"] == 1


def test_admission_threshold_and_serial_refreshes_the_median_during_backoff():
    c = C(serial_samples=3, window_attempts=1, backoff_ladder=(3,), min_draft=3)
    assert not c.admits(2) and c.admits(3)
    _serial(c, 3, 50.0)
    c.record_attempt("verify", 1, 500.0)                  # lose
    _serial(c, 3, 90.0)                                   # context grew: serial is slower now
    assert c.should_draft() and c.serial_median_ms == 90.0


def test_defaults_are_the_measured_best_admission_off_window_six():
    c = C.from_env({})
    assert (c.p_min, c.min_draft, c.window_attempts, c.backoff_ladder) == (0.0, 1, 6, (4, 8, 16, 32))
    assert c.admits(1)


def test_from_env_reads_the_knobs_and_rejects_nonsense():
    c = C.from_env({"DSV41_DSPARK_BACKOFF": "8,16", "DSV41_DSPARK_P_MIN": "0.6",
                    "DSV41_DSPARK_MIN_DRAFT": "2", "DSV41_DSPARK_WINDOW": "4"})
    assert c.backoff_ladder == (8, 16) and c.p_min == 0.6 and c.min_draft == 2 and c.window_attempts == 4
    with pytest.raises(ValueError):
        C(p_min=1.5)
    with pytest.raises(ValueError):
        C(backoff_ladder=())
    with pytest.raises(ValueError):
        C().record_attempt("guess", 1, 1.0)


def test_summary_reports_the_blend():
    c = C(serial_samples=1, window_attempts=1, backoff_ladder=(1,))
    _serial(c, 1, 50.0)
    c.record_attempt("verify", 4, 100.0)
    s = c.summary()
    assert s["ms_per_token"] == 30.0 and s["serial_median_ms"] == 50.0 and s["windows_won"] == 1
