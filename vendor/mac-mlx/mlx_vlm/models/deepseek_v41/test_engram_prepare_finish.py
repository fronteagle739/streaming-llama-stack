"""The row lookup split into a host half (prepare) and an MLX half (finish) is the same lookup
(v41 port lever 1, DSV41_STEP_OVERLAP): hits, misses, insertion and eviction all agree."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx_vlm.models.deepseek_v41.engram import EngramRowCache, dequant_rows_mx


class _Source:
    def __init__(self, n, head_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.num_embeddings = n
        self.weight = rng.integers(0, 256, size=(n, head_dim), dtype=np.uint8)
        self.scale = rng.integers(120, 134, size=(n, head_dim // 32), dtype=np.uint8)
        self.fetches = []

    def fetch(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        self.fetches.append(ids.copy())
        return self.weight[ids].copy(), self.scale[ids].copy()


def _reference(src, ids):
    raw, sc = src.fetch(ids)
    packed = np.ascontiguousarray(raw).view(np.uint32).reshape(-1, src.weight.shape[1] // 4)
    return dequant_rows_mx(mx.array(packed), mx.array(np.ascontiguousarray(sc)))


@pytest.mark.parametrize("capacity", [0, 6, 64])
def test_prepare_then_finish_equals_lookup(capacity):
    head_dim = 64
    a, b = _Source(100, head_dim), _Source(100, head_dim)
    ca = EngramRowCache(a, capacity, head_dim, "a")
    cb = EngramRowCache(b, capacity, head_dim, "b")
    rng = np.random.default_rng(1)
    for step in range(12):
        ids = rng.integers(0, 100, size=(8,), dtype=np.int64)
        if step % 3 == 0:
            ids[:4] = ids[0]                       # repeats within a lookup
        via_lookup = ca.lookup(ids)
        prepared = cb.prepare(ids)                  # host half (what the worker thread runs)
        assert all(not isinstance(v, mx.array) for v in prepared.values())
        via_split = cb.finish(prepared)
        # numpy has no bfloat16: compare through float32 (exact for bf16 values)
        f32 = lambda a: np.array(a.astype(mx.float32))
        assert np.array_equal(f32(via_lookup), f32(via_split)), step
        assert np.array_equal(f32(via_lookup), f32(_reference(_Source(100, head_dim), ids))), step
        assert via_lookup.shape == (8, head_dim)
    assert ca.stat_hits == cb.stat_hits and ca.stat_misses == cb.stat_misses
    assert len(a.fetches) == len(b.fetches)


def test_rows_prepare_is_none_for_ram_source(monkeypatch):
    from mlx_vlm.models.deepseek_v41 import engram as E
    obj = E.Engram.__new__(E.Engram)
    obj.source = "ram"
    assert E.Engram.rows_prepare(obj, np.zeros((1, 1, 3), np.int64)) is None
