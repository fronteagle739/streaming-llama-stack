"""Parity of our numpy Engram hash (mlx_vlm/models/deepseek_v41/engram.py) against DeepSeek's
reference inference/engram.py (torch), on the real V4.1 tokenizer.

Usage:  python test_engram_hash_parity.py <ref_dir_with_engram.py> <our_engram.py> <tokenizer_dir>
Compares: compressed token map, layout primes/offsets, multipliers, and the hash ids of random
sequences fed whole, split at arbitrary prefill/decode boundaries, and with a dead (image) span.
"""
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys
import types
from dataclasses import dataclass

import numpy as np
import torch


def load(path, name, stub_mlx=False):
    if stub_mlx and "mlx" not in sys.modules:
        # our module imports mlx at top level; the hash needs none of it on Windows
        mx = types.ModuleType("mlx.core")
        mx.array = object
        nn = types.ModuleType("mlx.nn")
        nn.Module = object
        nn.Linear = object
        pkg = types.ModuleType("mlx")
        pkg.core, pkg.nn = mx, nn
        for m, n in ((pkg, "mlx"), (mx, "mlx.core"), (nn, "mlx.nn")):
            m.__spec__ = importlib.machinery.ModuleSpec(n, None)
        sys.modules["mlx"], sys.modules["mlx.core"], sys.modules["mlx.nn"] = pkg, mx, nn
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class Args:
    engram_layer_ids: tuple = (1, 14)
    engram_num_embeddings: tuple = (384006168, 384016682)
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_id: int = 2
    engram_pad_token_id: int = 2
    engram_compressed_vocab_size: int = 99092
    max_batch_size: int = 2
    max_seq_len: int = 4096


def run_checks(ref_dir, ours_path, tok_dir):
    from transformers import AutoTokenizer  # before the mlx stub: transformers probes for mlx
    tok = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
    ref = load(f"{ref_dir}/engram.py", "ref_engram")
    ours = load(ours_path, "our_engram", stub_mlx=True)
    args = Args()

    # 1) compressed token map
    r_map, r_size = ref.build_compressed_token_map(tok)
    o_map, o_size = ours.build_compressed_token_map(f"{tok_dir}/tokenizer.json")
    assert r_size == o_size == args.engram_compressed_vocab_size, (r_size, o_size)
    assert len(r_map) == o_map.shape[0], (len(r_map), o_map.shape)
    diff = np.nonzero(np.array(r_map, dtype=np.int64) != o_map)[0]
    assert diff.size == 0, f"token map differs at {diff.size} ids, first {diff[:10]}"
    print(f"PASS token map: {len(r_map)} ids -> {r_size} compressed")

    # 2) layout
    r_lay = ref.EngramLayout.from_args(args)
    o_lay = ours.EngramLayout.from_config(args)
    assert np.array_equal(np.array(r_lay.primes, dtype=np.int64), o_lay.primes), "primes differ"
    r_mult = ref.compute_hash_multipliers(r_lay.layer_ids, r_lay.max_ngram_size, r_size).numpy()
    o_mult = ours.compute_hash_multipliers(o_lay.layer_ids, o_lay.max_ngram_size, o_size)
    assert np.array_equal(r_mult, o_mult), "multipliers differ"
    print(f"PASS layout: primes {o_lay.primes.shape}, offsets {o_lay.offsets[:, :3].tolist()} ..., multipliers {o_mult.shape}")

    # 3) hashes
    r_state = ref.NgramHashState(args, r_lay, tok)
    o_state = ours.NgramHashState(o_lay, o_map, args.engram_pad_id, o_size)
    rng = np.random.default_rng(7)
    vocab = len(tok)
    cases = 0
    for trial in range(6):
        L = int(rng.integers(5, 600))
        ids = rng.integers(0, vocab, size=(2, L), dtype=np.int64)
        mask = None
        if trial % 2 == 1:  # a dead span (image) in the middle of row 0
            mask = np.ones((2, L), dtype=bool)
            a, b = sorted(rng.integers(0, L, size=2))
            mask[0, a:b] = False
        # whole sequence at once
        r_all = r_state(torch.tensor(ids), 0, None if mask is None else torch.tensor(mask)).numpy()
        o_all = o_state.hashes(ids, 0, mask)
        assert r_all.shape == o_all.shape, (r_all.shape, o_all.shape)
        assert np.array_equal(r_all, o_all), f"trial {trial}: whole-sequence hashes differ"
        # split at random boundaries: prefill chunk then token-by-token decode
        r2 = ref.NgramHashState(args, r_lay, tok)
        o2 = ours.NgramHashState(o_lay, o_map, args.engram_pad_id, o_size)
        cut = int(rng.integers(1, L))
        r_parts = [r2(torch.tensor(ids[:, :cut]), 0, None if mask is None else torch.tensor(mask[:, :cut])).numpy()]
        o_parts = [o2.hashes(ids[:, :cut], 0, None if mask is None else mask[:, :cut])]
        for p in range(cut, L):
            m = None if mask is None else torch.tensor(mask[:, p : p + 1])
            r_parts.append(r2(torch.tensor(ids[:, p : p + 1]), p, m).numpy())
            o_parts.append(o2.hashes(ids[:, p : p + 1], p, None if mask is None else mask[:, p : p + 1]))
        r_split = np.concatenate(r_parts, axis=1)
        o_split = np.concatenate(o_parts, axis=1)
        assert np.array_equal(r_split, r_all), f"trial {trial}: reference split != whole (harness bug)"
        assert np.array_equal(o_split, o_all), f"trial {trial}: our split != whole"
        # trim and re-hash the tail (cache rollback)
        back = int(rng.integers(1, min(8, L)))
        o2.trim(back)
        o_re = o2.hashes(ids[:, L - back :], L - back, None if mask is None else mask[:, L - back :])
        assert np.array_equal(o_re, o_all[:, L - back :]), f"trial {trial}: re-hash after trim differs"
        cases += 1
        print(f"PASS hashes trial {trial}: L={L} cut={cut} mask={'yes' if mask is not None else 'no'} "
              f"range=[{o_all.min()}, {o_all.max()}] max_rows={max(args.engram_num_embeddings)}")
    assert o_all.max() < max(args.engram_num_embeddings)
    print(f"ALL PASS ({cases} sequences)")


def test_engram_hash_parity():
    import pytest

    # Reuse the oracle's env overrides and platform-aware local path defaults.
    oracle = load(Path(__file__).with_name("reference_cpu_dsv41.py"), "dsv41_hash_paths")
    ours = Path(os.environ.get("DSV41_OUR_ENGRAM", Path(__file__).with_name("engram.py"))).expanduser()
    required = [oracle.REF_DIR / "engram.py", ours,
                oracle.TOKENIZER_DIR / "tokenizer.json", oracle.TOKENIZER_DIR / "tokenizer_config.json"]
    if not all(path.is_file() for path in required):
        pytest.skip("Engram parity inputs absent; set DSV41_REF_DIR, DSV41_TOKENIZER_DIR "
                    "and DSV41_OUR_ENGRAM")
    names = ("mlx", "mlx.core", "mlx.nn", "ref_engram", "our_engram")
    previous = {name: sys.modules.get(name) for name in names}
    try:
        run_checks(oracle.REF_DIR, ours, oracle.TOKENIZER_DIR)
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def main():
    run_checks(*sys.argv[1:4])


if __name__ == "__main__":
    main()


def test_engram_row_cache_lazy_numpy():
    """Exercise arena growth, eviction, lookup and budgets without importing MLX."""
    import importlib.util
    import os
    from pathlib import Path
    import sys
    import types
    from unittest.mock import patch
    import numpy as np

    mx = types.ModuleType("mlx.core")
    for name in ("array", "zeros", "take", "uint32", "uint8", "float32"):
        setattr(mx, name, getattr(np, name))
    mx.bfloat16 = np.float32
    pkg = types.ModuleType("mlx")
    nn = types.ModuleType("mlx.nn")
    nn.Module = object
    pkg.core, pkg.nn = mx, nn
    module_name = "engram_lazy_numpy_test"
    spec = importlib.util.spec_from_file_location(module_name, Path(__file__).with_name("engram.py"))
    mod = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"mlx": pkg, "mlx.core": mx, "mlx.nn": nn, module_name: mod}):
        spec.loader.exec_module(mod)
    mx.dequantize = lambda packed, scales, **kw: mod.ShardRows.dequant_np(
        np.ascontiguousarray(packed).view(np.uint8), scales)

    class FakeShardRows:
        num_embeddings = 10000
        head_dim = 256

        def fetch(self, ids):
            ids = np.asarray(ids)
            raw = ((ids[:, None] + np.arange(256)[None, :]) % 120).astype(np.uint8)
            sc = np.broadcast_to((125 + ids % 5)[:, None], (ids.size, 8)).astype(np.uint8).copy()
            return raw, sc

    source = FakeShardRows()
    mod.EngramRowCache.ARENA_CHUNK_ROWS = 1024
    cache = mod.EngramRowCache(source, 2500, 256, "test")
    assert cache.allocated_rows == 0 and len(cache.cache_w) == len(cache.cache_s) == 0

    def check_lookup(cache, ids):
        expected = (mod.ShardRows.dequant_np(*source.fetch(ids)) if ids.size
                    else np.empty((0, 256), dtype=np.float32))
        np.testing.assert_array_equal(cache.lookup(ids), expected)

    for batch in range(3):
        ids = np.arange(batch * 1000, (batch + 1) * 1000)
        cache.insert(ids, *source.fetch(ids))
        assert len(cache.cache_w) == len(cache.cache_s) == batch + 1
        assert cache.allocated_rows == min((batch + 1) * 1024, 2500)
        check_lookup(cache, ids[::-1])
        occupied = cache.row_of[:cache.free_top]
        assert np.unique(occupied).size == cache.free_top
        np.testing.assert_array_equal(cache.slot_of[occupied], np.arange(cache.free_top))
    assert cache.free_top == 2500
    assert np.count_nonzero(cache.slot_of[:3000] < 0) == 500
    stats = cache.stats()
    assert stats["allocated_rows"] == stats["capacity_rows"] == 2500
    assert stats["allocated_gb"] == stats["capacity_gb"] == 0.00066
    evicted = np.flatnonzero(cache.slot_of[:3000] < 0)[:80]
    hits = cache.row_of[:80].copy()
    check_lookup(cache, np.concatenate([hits[::-1], evicted, hits]))
    # A large miss set forces the victim-sample fallback; preserve every live hit.
    check_lookup(cache, np.concatenate([cache.row_of[:500], np.arange(4000, 6000)]))
    # Only chunks appearing in the batch should be gathered (payload and scales).
    with patch.object(mx, "take", wraps=np.take) as take:
        check_lookup(cache, np.repeat(cache.row_of[0], 3))
        assert take.call_count == 3
    tiny = mod.EngramRowCache(source, 2, 256, "tiny")
    check_lookup(tiny, np.array([1, 2]))
    check_lookup(tiny, np.array([2, 3, 4, 5, 2]))
    assert tiny.slot_of[2] >= 0
    check_lookup(tiny, np.array([6, 7, 8, 9]))
    assert tiny.free_top == tiny.allocated_rows == 2
    empty = mod.EngramRowCache(source, 0, 256, "empty")
    check_lookup(empty, np.array([1, 2, 1]))
    check_lookup(empty, np.array([], dtype=np.int64))
    assert empty.allocated_rows == 0
    saved = {}
    hot_path = Path("unused-hot-set.npy")
    cache.hot_file = lambda: hot_path
    with patch.object(Path, "mkdir"), patch.object(np, "save", side_effect=lambda p, rows: saved.update(rows=rows.copy())):
        assert cache.save_hot() == 2500
    warmed = mod.EngramRowCache(source, 2500, 256, "warmed")
    warmed.hot_file = lambda: hot_path
    with patch.object(Path, "exists", return_value=True), patch.object(np, "load", side_effect=lambda p: saved["rows"]):
        assert warmed.warm(batch=1000) == 2500
        assert warmed.warm(batch=1000) == 0
    assert len(warmed.cache_w) == 3
    check_lookup(warmed, saved["rows"])
    print("PASS lazy arenas: chunks 0 -> 1 -> 2 -> 3; allocated rows 0 -> 1024 -> 2048 -> 2500")
    print("PASS eviction: 3000 inserts, 2500 residents, 500 evictions; mappings and hit/miss rows exact")
    print("PASS protected hits, large eviction batch, touched-chunk gathers, mixed/zero capacity, warm/save_hot")

    mx.device_info = lambda: {"max_recommended_working_set_size": 100_000_000_000}
    mx.get_active_memory = lambda: 20_000_000_000
    with patch("subprocess.check_output", return_value="0"), patch.dict(os.environ, {
        "LLAMA_MLX_ENGRAM_CACHE_FRACTION": "0.5", "LLAMA_MLX_ENGRAM_CACHE_GB": ""}):
        assert mod.auto_cache_budget_bytes() == 28_000_000_000
        with patch.dict(os.environ, {"LLAMA_MLX_ENGRAM_CACHE_GB": "10"}):
            assert mod.auto_cache_budget_bytes() == 10_000_000_000
        with patch.dict(os.environ, {"LLAMA_MLX_ENGRAM_CACHE_FRACTION": "0.25"}):
            assert mod.auto_cache_budget_bytes() == 14_000_000_000
        tables = []
        for layer_id in (1, 14):
            table = mod.Engram.__new__(mod.Engram)
            table.source, table.shard = "hybrid", None
            table.layer_id, table.table_count = layer_id, 2
            table.num_embeddings, table.head_dim = source.num_embeddings, 256
            with patch.object(mod, "ShardRows", return_value=source):
                decision = table.attach_storage("unused")
            assert decision["budget_bytes"] == 14_000_000_000
            assert decision["total_budget_bytes"] == 28_000_000_000
            assert decision["allocated_rows"] == 0
            tables.append(table)
        table = tables[0]
        table.shard = None
        with patch.object(mod, "ShardRows", return_value=source):
            assert table.attach_storage("unused", cache_budget_bytes=1000)["budget_bytes"] == 1000
        for invalid in ("-0.1", "1.1", "nan"):
            with patch.dict(os.environ, {"LLAMA_MLX_ENGRAM_CACHE_FRACTION": invalid}):
                try:
                    mod.auto_cache_budget_bytes()
                except ValueError:
                    pass
                else:
                    raise AssertionError("invalid fraction accepted")
    print("PASS auto budget: 28 GB total / 2 = 14 GB/table; GB cap, fraction and explicit per-table override")
