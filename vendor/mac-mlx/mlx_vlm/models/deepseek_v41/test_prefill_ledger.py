"""Synthetic ledger checks; no weights, model forward or GPU evaluation needed."""

from pathlib import Path
import runpy

import pytest


def _load(monkeypatch, enabled, cuts):
    # Read real env switches in an isolated namespace without reloading MLX modules
    # or changing the process-global counters used by other tests.
    for name, value in (("DSV41_PREFILL_LEDGER", enabled),
                        ("DSV41_PREFILL_LEDGER_CUTS", cuts)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    return runpy.run_path(str(Path(__file__).with_name("prefill_ledger.py")))


@pytest.mark.parametrize("enabled", [None, "0", "1"])
@pytest.mark.parametrize("cuts", [None, "0", "1", "2"])
def test_prefill_ledger_switches(monkeypatch, enabled, cuts):
    ledger = _load(monkeypatch, enabled, cuts)
    host = ledger["PREFILL_LEDGER"]
    layers = ledger["PREFILL_LAYER_MS"]
    before = host.copy()
    ledger["_prefill_ledger_chunk"](512, 1.25, 4.5)
    ledger["_prefill_ledger_chunk"](7, 0.75, 1.5)
    ledger["_prefill_ledger_add"]("hash_ms", 0.5)
    for layer, phase, ms in ((2, "attn_ms", 3.0), (2, "attn_ms", 2.0),
                             (2, "ffn_ms", 4.0), (4, "engram_ms", 1.0)):
        ledger["_prefill_layer_add"](layer, phase, ms)
    if enabled == "1":
        assert host == {"tokens": 519, "chunks": 2, "hash_ms": 2.5, "engram_fetch_ms": 6.0}
    else:
        assert host == before
    if enabled == "1" and cuts in ("1", "2"):
        assert layers == {(2, "attn_ms"): 5.0, (2, "ffn_ms"): 4.0, (4, "engram_ms"): 1.0}
    else:
        assert layers == {}
    # Decode and empty calls never count as prefill, even with both switches on.
    before = host.copy()
    ledger["_prefill_ledger_chunk"](1, 100.0, 200.0)
    ledger["_prefill_ledger_chunk"](0, 100.0, 200.0)
    assert host == before


@pytest.mark.parametrize("enabled", [None, "0", "1"])
@pytest.mark.parametrize("cuts, level", [
    (None, 0), ("0", 0), ("1", 1), ("2", 2), ("", 0), ("invalid", 0),
    ("-1", 0), ("3", 0),
])
def test_prefill_ledger_cuts_level(monkeypatch, enabled, cuts, level):
    ledger = _load(monkeypatch, enabled, cuts)
    expected = level if enabled == "1" else 0
    assert type(ledger["_PREFILL_LEDGER_CUTS_LEVEL"]) is int
    assert ledger["_PREFILL_LEDGER_CUTS_LEVEL"] == expected
    assert ledger["_PREFILL_LEDGER_CUTS"] is (expected > 0)
    ledger["_prefill_layer_add"](3, "attn_index_ms", 0.0)
    ledger["_prefill_layer_add"](3, "attn_proj_q_ms", 1.25)
    assert ledger["PREFILL_LAYER_MS"] == (
        {(3, "attn_index_ms"): 0.0, (3, "attn_proj_q_ms"): 1.25}
        if expected else {}
    )
