"""CPU-only deterministic routing fixtures shared by Mac and Windows tests."""
from functools import lru_cache
import heapq
import math
import random

import numpy as np

E, TOPK, TOKENS, SEED = 384, 6, 2048, 20260916


@lru_cache(maxsize=1)
def probe_a():
    # Exact routing() arm A from ffn_factorial_probe.py:56 (not NumPy RNG).
    rng = random.Random(SEED)
    logits = [-1.1 * math.log(e + 1) for e in range(E)]
    rows = []
    for _ in range(TOKENS):
        scores = [v - math.log(-math.log(max(rng.random(), 1e-300))) for v in logits]
        rows.append(heapq.nlargest(TOPK, range(E), key=scores.__getitem__))
    return np.asarray(rows, dtype=np.uint32)


def routes(case):
    if case == "probe_a":
        return probe_a().copy()
    if case == "balanced32":
        return (np.arange(TOKENS * TOPK, dtype=np.uint32) % E).reshape(TOKENS, TOPK)
    if case == "8x1536":
        return (np.arange(TOKENS * TOPK, dtype=np.uint32) % 8).reshape(TOKENS, TOPK)
    counts = np.zeros(E, dtype=np.int64)
    if case == "singleton_empty":
        counts[[0, 1, 191, 383]] = [1, 1536, 1, 1]
    elif case == "boundaries":
        counts[[0, 2, 5, 7, 190, 382, 383]] = [15, 16, 17, 31, 33, 1536, 1]
    elif case == "fused_boundaries":
        counts[[0, 2, 5, 7, 190, 191, 300, 381, 383]] = [15, 16, 17, 31, 32, 33, 47, 64, 1536]
    elif case.startswith("remainder_"):
        counts[[0, 191, 383]] = [1, 1536, 15 + int(case.split("_")[1])]
    elif case == "empty":
        pass
    else:
        raise ValueError(case)
    return np.repeat(np.arange(E, dtype=np.uint32), counts)


def metadata_numpy(sorted_ids, tile_rows=16):
    """The Metal lower-bound/two-buffer-scan formulas, with 512 CPU lanes."""
    ids = np.asarray(sorted_ids, dtype=np.uint32)
    offsets = np.searchsorted(ids, np.arange(E + 1), side="left").astype(np.uint32)
    offsets[E] = ids.size
    scan_a = np.zeros(512, dtype=np.uint32)
    scan_a[:E] = (np.diff(offsets) + tile_rows - 1) // tile_rows
    for step in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        scan_b = scan_a.copy()
        scan_b[step:] += scan_a[:-step]
        scan_a = scan_b.copy()
    prefix = np.concatenate((np.zeros(1, dtype=np.uint32), scan_a[:E]))
    return offsets, prefix
