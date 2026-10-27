"""Opt-in prefill counters, sampled at import just like the decode ledger.

Host times are cumulative milliseconds; GPU cuts measure eval + synchronize,
including pending dependencies (the first phase also includes embeddings).
Attention includes the indexer, projections and hyper-connection work; FFN
includes its hyper-connections. Cuts serialize execution for attribution.
DSV41_PREFILL_LEDGER_CUTS=1 cuts at layer boundaries; 2 also cuts attention
sub-phases. Both require DSV41_PREFILL_LEDGER=1. At level 2, attn_ms measures
the remaining hyper-connection work, not the already evaluated sub-phases.
"""

import os

_PREFILL_LEDGER_ON = os.environ.get("DSV41_PREFILL_LEDGER", "0") == "1"
_PREFILL_LEDGER_CUTS_LEVEL: int = (
    {"1": 1, "2": 2}.get(os.environ.get("DSV41_PREFILL_LEDGER_CUTS", "0"), 0)
    if _PREFILL_LEDGER_ON else 0
)
_PREFILL_LEDGER_CUTS = _PREFILL_LEDGER_CUTS_LEVEL > 0
PREFILL_LEDGER = {"hash_ms": 0.0, "engram_fetch_ms": 0.0, "tokens": 0, "chunks": 0}
PREFILL_LAYER_MS: dict = {}


def _prefill_ledger_add(key, value):
    if _PREFILL_LEDGER_ON:
        PREFILL_LEDGER[key] = PREFILL_LEDGER.get(key, 0) + value


def _prefill_ledger_chunk(n, hash_ms, engram_fetch_ms):
    """Only multi-token forwards contribute, including a short final chunk."""
    if _PREFILL_LEDGER_ON and n > 1:
        _prefill_ledger_add("tokens", n)
        _prefill_ledger_add("chunks", 1)
        _prefill_ledger_add("hash_ms", hash_ms)
        _prefill_ledger_add("engram_fetch_ms", engram_fetch_ms)


def _prefill_layer_add(layer_id, phase, ms):
    if _PREFILL_LEDGER_CUTS:
        key = (layer_id, phase)
        PREFILL_LAYER_MS[key] = PREFILL_LAYER_MS.get(key, 0.0) + ms
