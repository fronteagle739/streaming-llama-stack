"""DeepSeek-V4.1 Engram: n-gram hash tables written into the residual stream at a few layers.

Port of ``inference/engram.py`` and the ``ParallelEngramEmbedding`` / ``Engram`` classes of
``inference/model.py`` shipped with deepseek-ai/DeepSeek-V4.1-Flash. The hash must reproduce the
reference bit for bit: every one of the 384M rows of each table is addressed by it, and a mismatch
turns 189 GiB of learned memory into noise.

Table storage (owner, 2026-09-10: "support for fetching ngrams from ram OR nvme"; "make a ngram
cache that fits as much into the mac unified memory and only fetch the last bits from nvme";
"weights, mtp and vision live in unified along with as much ngram as possible"; "a hotness tracking
lru cache for ngram like the ds4 one"):

* ``ram``    -- the whole table resident as MLX mxfp8 (uint32 payload + e8m0 scales, chunked so no
                single Metal buffer holds 98 GB). ~94 GiB per table.
* ``nvme``   -- every row read on demand from the safetensors shard through a memory map. A token
                touches 24 rows of 264 bytes per Engram layer: microseconds at decode, a few seconds
                of random 4 KiB reads for a 100k-token prefill. Nothing resident.
* ``hybrid`` -- (default) a row cache in unified memory sized to what is left after the weights
                (``auto``: the smaller of host-wide availability and the wired limit minus what MLX
                already holds, less a margin and a reserve for long-context KV/activations, times
                ``LLAMA_MLX_ENGRAM_CACHE_FRACTION``; or a hard ``LLAMA_MLX_ENGRAM_CACHE_GB`` cap),
                misses fetched from the memory map, eviction by sampled LRU with a hit counter, and
                the hot set saved (merged with the previous one) so the next launch pre-warms.

Selected by ``LLAMA_MLX_ENGRAM_SOURCE`` (``ram`` | ``nvme`` | ``hybrid``), else
``config.engram_source``. ``LLAMA_MLX_ENGRAM_REPORT=1`` logs cumulative counters
at prefill/decode transitions and memory after storage attachment (diagnostic only).
Source payload bytes count memmap rows copied, not physical SSD reads: the OS page
cache and page-size amplification must be measured separately.
"""

from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import time as _ledger_time
from .prefill_ledger import _PREFILL_LEDGER_ON

# Phase 0 of the v41 port (2026-09-15): host milliseconds spent fetching Engram rows per decode step.
_LEDGER_ON = os.environ.get("DSV41_STEP_LEDGER", "0") == "1"
LEDGER = {"engram_host_ms": 0.0, "engram_calls": 0}

DEAD = -1  # a position that takes no part in an n-gram (an image span)
ROW_BYTES = 256 + 8  # fp8 row + e8m0 scales
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

# Budget defaults (decimal GB, like every other GB knob here). The margin keeps the box breathing
# room; the reserve is for the KV/activations of a long context (a 262k-token conversation is
# ~3.4 GiB of compressed caches plus activations), so a huge n-gram cache cannot squeeze it out.
DEFAULT_CACHE_FRACTION = 0.9
DEFAULT_CACHE_MARGIN_GB = 16.0
DEFAULT_CACHE_RESERVE_GB = 8.0


# --------------------------------------------------------------------------- the hash


def is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin, exact below 3.3e24 (agrees with sympy.isprime)."""
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen: set) -> int:
    """The smallest prime above `start` that has not been handed out yet."""
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer_json: str | Path) -> Tuple[np.ndarray, int]:
    """Map every token id onto a smaller id space where tokens that normalize alike collapse.

    N-grams are hashed over these compressed ids, so " The", "the" and "THE" hash the same way.
    Returns (lookup[int64, vocab], compressed vocab size); the size feeds every hash multiplier.
    """
    from tokenizers import Regex, Tokenizer, normalizers

    backend = Tokenizer.from_file(str(tokenizer_json))
    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    vocab = backend.get_vocab_size(with_added_tokens=True)
    key_to_new: dict = {}
    lookup = np.zeros((vocab,), dtype=np.int64)
    for token_id in range(vocab):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(token_id)  # a partial UTF-8 byte token: keyed by its raw form
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def cache_dir() -> Path:
    return Path(os.environ.get("MLX_VLM_CACHE_DIR", Path.home() / ".cache" / "mlx_vlm"))


def load_compressed_token_map(tokenizer_json: str | Path, expected_size: int) -> Tuple[np.ndarray, int]:
    """The compressed map, cached under ~/.cache/mlx_vlm keyed by the tokenizer file's size+mtime
    (129,280 decodes take a few seconds; the table is deterministic)."""
    tokenizer_json = Path(tokenizer_json)
    st = tokenizer_json.stat()
    cache = cache_dir() / f"engram_token_map_{st.st_size}_{int(st.st_mtime)}_{expected_size}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["lookup"].astype(np.int64), int(z["size"])
    lookup, size = build_compressed_token_map(tokenizer_json)
    if size != expected_size:
        raise ValueError(
            f"engram compressed vocab is {size}, config says {expected_size}: the tokenizer does not "
            "match the checkpoint and every hash would land on the wrong row")
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, lookup=lookup, size=np.int64(size))
    except OSError:
        pass
    return lookup, size


def compute_hash_multipliers(layer_ids: Sequence[int], max_ngram_size: int, vocab_size: int) -> np.ndarray:
    """One odd multiplier per (layer, lookback), from a per-layer RNG so layers hash differently;
    bounded so that token_id * multiplier cannot overflow int64. [n_layers, max_ngram_size]."""
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        gen = np.random.default_rng(10007 * int(layer_id))
        values = gen.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(values * 2 + 1)
    return np.stack(rows).astype(np.int64)


@dataclass(frozen=True)
class EngramLayout:
    """Bucket layout of the tables: a position is hashed as max_ngram_size-1 n-grams (2-gram ..),
    each split over n_heads heads; every (n-gram size, head) pair owns a prime-sized bucket range
    in the layer's table, primes drawn in order and never reused, so the ranges are disjoint."""

    max_ngram_size: int
    layer_ids: Tuple[int, ...]
    num_embeddings: Tuple[int, ...]
    primes: np.ndarray  # [n_layers, max_ngram_size-1, n_heads] int64
    n_heads: int
    head_dim: int

    @property
    def n_cols(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads

    @property
    def offsets(self) -> np.ndarray:
        """[n_layers, n_cols]: where each (n-gram size, head) bucket range starts in the table."""
        flat = self.primes.reshape(len(self.layer_ids), -1)
        return np.concatenate([np.zeros((flat.shape[0], 1), np.int64), np.cumsum(flat, axis=1)[:, :-1]], axis=1)

    @classmethod
    def from_config(cls, cfg) -> Optional["EngramLayout"]:
        layer_ids = tuple(int(i) for i in (getattr(cfg, "engram_layer_ids", None) or ()))
        if not layer_ids:
            return None
        max_ngram, n_heads = int(cfg.engram_max_ngram_size), int(cfg.engram_n_heads)
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram - 1):
                sizes, current = [], int(cfg.engram_vocab_size) - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(sizes)
            primes.append(per_ngram)
        return cls(
            max_ngram_size=max_ngram,
            layer_ids=layer_ids,
            num_embeddings=tuple(int(n) for n in cfg.engram_num_embeddings),
            primes=np.array(primes, dtype=np.int64),
            n_heads=n_heads,
            head_dim=int(cfg.engram_head_dim),
        )


class NgramHashState:
    """Maps positions to the hash ids of the n-grams ending there, carrying the compressed ids of
    the whole sequence so a decode step can look back across the prefill/decode split. One per
    conversation: it lives in the cache list (EngramHashCache), not in the model."""

    def __init__(self, layout: EngramLayout, token_map: np.ndarray, pad_id: int, vocab_size: int):
        self.layout = layout
        self.token_map = token_map
        self.pad_id = int(token_map[pad_id])
        self.multipliers = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab_size)
        self.primes = layout.primes  # [n_layers, max_ngram-1, n_heads]
        self.offsets = layout.offsets  # [n_layers, n_cols]
        self.history: Optional[np.ndarray] = None  # [B, cap] compressed ids
        self.length = 0

    def _ensure(self, batch: int, needed: int) -> None:
        if self.history is None:
            self.history = np.full((batch, max(4096, needed)), self.pad_id, dtype=np.int64)
        elif self.history.shape[0] != batch or self.history.shape[1] < needed:
            cap = max(needed, 2 * self.history.shape[1])
            new = np.full((batch, cap), self.pad_id, dtype=np.int64)
            n = min(self.length, self.history.shape[1])
            b = min(batch, self.history.shape[0])
            new[:b, :n] = self.history[:b, :n]
            self.history = new

    def trim(self, n: int) -> None:
        self.length = max(0, self.length - int(n))

    def hashes(self, input_ids: np.ndarray, start_pos: int, token_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """input_ids [B, L] at positions start_pos..; token_mask [B, L] False for image tokens.
        Returns int64 [B, L, n_layers, n_cols] hash ids into the layers' tables."""
        ids = np.asarray(input_ids, dtype=np.int64)
        batch, seqlen = ids.shape
        compressed = self.token_map[ids]
        if token_mask is not None:
            compressed = np.where(np.asarray(token_mask, dtype=bool), compressed, DEAD)
        self._ensure(batch, start_pos + seqlen)
        self.history[:, start_pos : start_pos + seqlen] = compressed
        self.length = start_pos + seqlen

        positions = np.broadcast_to(np.arange(start_pos, start_pos + seqlen, dtype=np.int64), (batch, seqlen))
        blocked = np.zeros(positions.shape, dtype=bool)
        tokens = []
        for shift in range(self.layout.max_ngram_size):
            source = np.take_along_axis(self.history, np.maximum(positions - shift, 0), axis=1)
            blocked = blocked | (positions < shift) | (source == DEAD)
            tokens.append(np.where(blocked, self.pad_id, source))
        tokens = np.stack(tokens, axis=-1)  # [B, L, max_ngram]

        # XOR the multiplied ids together one lookback at a time: the running value after step i is
        # the hash of the (i+1)-gram; each lands in its own prime-sized bucket range
        products = tokens[:, :, None, :] * self.multipliers[None, None]  # [B, L, n_layers, max_ngram]
        rolling = products[..., 0]
        hashes = []
        for i in range(1, self.layout.max_ngram_size):
            rolling = np.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling[..., None] % self.primes[None, None, :, i - 1, :])
        return np.concatenate(hashes, axis=-1) + self.offsets[None, None]


class EngramHashCache:
    """The per-conversation n-gram state, carried as the last entry of the model's cache list so it
    is created, trimmed and discarded with the KV caches. Quacks like an mlx_lm cache."""

    def __init__(self, state: NgramHashState):
        self.hash_state = state
        self.keys = None
        self.values = None

    @property
    def offset(self) -> int:
        return self.hash_state.length

    def trim(self, n: int) -> int:
        n = min(int(n), self.hash_state.length)
        self.hash_state.trim(n)
        return n

    def is_trimmable(self) -> bool:
        return True

    @property
    def state(self):
        return []

    @state.setter
    def state(self, _v):
        pass

    @property
    def nbytes(self) -> int:
        h = self.hash_state.history
        return 0 if h is None else int(h.nbytes)


# --------------------------------------------------------------------------- table storage

_E4M3_LUT: Optional[np.ndarray] = None


def e4m3_lut() -> np.ndarray:
    """float32 value of every OCP float8_e4m3fn byte (0x7F / 0xFF are NaN in the spec)."""
    global _E4M3_LUT
    if _E4M3_LUT is None:
        lut = np.empty((256,), dtype=np.float32)
        for code in range(256):
            sign = -1.0 if code & 0x80 else 1.0
            exp = (code >> 3) & 0xF
            mant = code & 0x7
            if exp == 0:
                val = (mant / 8.0) * 2.0 ** (1 - 7)
            elif exp == 15 and mant == 7:
                val = float("nan")
            else:
                val = (1.0 + mant / 8.0) * 2.0 ** (exp - 7)
            lut[code] = sign * val
        _E4M3_LUT = lut
    return _E4M3_LUT


def dequant_rows_mx(packed: mx.array, scales: mx.array) -> mx.array:
    """packed uint32 [N, head_dim/4] + scales uint8 [N, head_dim/32] (mxfp8) -> bf16 [N, head_dim]."""
    return mx.dequantize(packed, scales, group_size=32, bits=8, mode="mxfp8", dtype=mx.bfloat16)


class EngramTableRAM(nn.Module):
    """The table resident as MLX mxfp8: `weight[i]` uint32 [rows_i, head_dim/4] and `scales[i]` uint8
    [rows_i, head_dim/32], chunked over rows. sanitize() produces these keys from the checkpoint."""

    def __init__(self, num_embeddings: int, head_dim: int, n_chunks: int):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.head_dim = int(head_dim)
        self.n_chunks = int(n_chunks)
        self.chunk_rows = -(-self.num_embeddings // self.n_chunks)
        rows = [min(self.chunk_rows, self.num_embeddings - i * self.chunk_rows) for i in range(self.n_chunks)]
        self.weight = [mx.zeros((r, self.head_dim // 4), dtype=mx.uint32) for r in rows]
        self.scales = [mx.zeros((r, self.head_dim // 32), dtype=mx.uint8) for r in rows]

    def __call__(self, ids: np.ndarray) -> mx.array:
        """Host hash ids [N] -> bf16 [N, head_dim], fetching each requested row once.

        Hashing already produces NumPy ids. Partition those before uploading indices;
        converting a device array here would introduce a synchronization. Each chunk
        gathers only its own rows, then one dequantization handles the packed result.
        """
        if not isinstance(ids, np.ndarray):
            raise TypeError("resident Engram expects host hash ids, not a device array")
        if ids.size == 0:
            return mx.zeros((0, self.head_dim), dtype=mx.bfloat16)
        valid = (ids >= 0) & (ids < self.num_embeddings)
        all_valid = valid.all()
        if not all_valid:
            ids = np.where(valid, ids, 0)
        chunks = ids // self.chunk_rows
        order = np.argsort(chunks, kind="stable")
        sorted_chunks = chunks[order]
        boundaries = np.flatnonzero(np.diff(sorted_chunks)) + 1
        weights, scales = [], []
        for positions in np.split(order, boundaries):
            chunk = int(chunks[positions[0]])
            local = mx.array((ids[positions] - chunk * self.chunk_rows).astype(np.int32))
            weights.append(mx.take(self.weight[chunk], local, axis=0))
            scales.append(mx.take(self.scales[chunk], local, axis=0))
        packed = weights[0] if len(weights) == 1 else mx.concatenate(weights, axis=0)
        scale = scales[0] if len(scales) == 1 else mx.concatenate(scales, axis=0)
        vals = dequant_rows_mx(packed, scale)
        if len(weights) != 1:
            inverse = np.empty(order.size, dtype=np.int32)
            inverse[order] = np.arange(order.size, dtype=np.int32)
            vals = mx.take(vals, mx.array(inverse), axis=0)
        return vals if all_valid else mx.where(mx.array(valid)[:, None], vals, mx.zeros_like(vals))


class ShardRows:
    """One fp8 table read straight from its safetensors shard through a memory map: `weight` uint8
    [rows, head_dim] e4m3 and `scale` uint8 [rows, head_dim/32] e8m0. Nothing resident."""

    def __init__(self, model_dir: str | Path, weight_key: str, scale_key: str, num_embeddings: int, head_dim: int):
        self.num_embeddings = int(num_embeddings)
        self.head_dim = int(head_dim)
        self.weight = self._memmap(Path(model_dir), weight_key, (self.num_embeddings, self.head_dim))
        self.scale = self._memmap(Path(model_dir), scale_key, (self.num_embeddings, self.head_dim // 32))

    @staticmethod
    def _memmap(model_dir: Path, key: str, shape: Tuple[int, int]) -> np.ndarray:
        index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        shard = model_dir / index["weight_map"][key]
        with open(shard, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hlen).decode("utf-8"))
        meta = header[key]
        start, end = meta["data_offsets"]
        if tuple(meta["shape"]) != tuple(shape) or end - start != shape[0] * shape[1]:
            raise ValueError(f"{key}: shard says {meta['shape']} {meta['dtype']}, expected {shape} of 1 byte")
        return np.memmap(shard, dtype=np.uint8, mode="r", offset=8 + hlen + start, shape=shape)

    def fetch(self, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ids int64 [N] -> (rows uint8 [N, head_dim], scales uint8 [N, head_dim/32]); one 4 KiB page
        read per row that is not in the page cache. Sorted ids read closer to sequentially."""
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        raw = np.asarray(self.weight[sorted_ids])
        sc = np.asarray(self.scale[sorted_ids])
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        return raw[inv], sc[inv]

    @staticmethod
    def dequant_np(raw: np.ndarray, sc: np.ndarray) -> np.ndarray:
        vals = e4m3_lut()[raw].reshape(raw.shape[0], -1, 32)
        vals = vals * np.exp2(sc.astype(np.int32) - 127).astype(np.float32)[..., None]
        return vals.reshape(raw.shape[0], -1)


class EngramRowCache:
    """A hotness-tracking row cache in unified memory in front of a ShardRows source.

    Slots hold packed mxfp8 rows in lazily allocated MLX chunks. Lookup dequantizes on the GPU
    with one gather per touched chunk for rows from both RAM and NVMe. Maps are numpy: `slot_of[row]` (int32, -1 when
    absent), `row_of[slot]`, `last_used[slot]` (a tick per lookup) and `hits[slot]` (saturating).
    Eviction is sampled LRU: 4x the needed victims are sampled at random and the least recently
    used win, hits breaking ties, so a row that keeps coming back outlives a one-off. The hot set
    (cached rows ordered by hits) is saved to ~/.cache/mlx_vlm and pre-fetched on the next launch.
    """

    SAMPLE = 4
    ARENA_CHUNK_ROWS = 1 << 20

    def __init__(self, source: ShardRows, capacity_rows: int, head_dim: int, name: str):
        self.source = source
        self.capacity = max(0, int(capacity_rows))
        self.head_dim = int(head_dim)
        self.name = name
        self.slot_of = np.full((source.num_embeddings,), -1, dtype=np.int32)
        self.row_of = np.full((self.capacity,), -1, dtype=np.int64)
        self.last_used = np.zeros((self.capacity,), dtype=np.uint32)
        self.hits = np.zeros((self.capacity,), dtype=np.uint16)
        self.free_top = 0  # slots [0, free_top) are in use
        self.tick = 1
        self.rng = np.random.default_rng(12345)
        self.stat_lookups = 0
        self.stat_hits = 0
        self.stat_misses = 0
        self.stat_requested_rows = 0
        self.stat_requested_hits = 0
        self.stat_source_bytes = 0
        self.stat_fetch_s = 0.0
        self.cache_w = []
        self.cache_s = []

    @property
    def allocated_rows(self) -> int:
        return sum(chunk.shape[0] for chunk in self.cache_w)

    def _ensure_arenas(self) -> None:
        while len(self.cache_w) * self.ARENA_CHUNK_ROWS < self.free_top:
            rows = min(self.ARENA_CHUNK_ROWS,
                       self.capacity - len(self.cache_w) * self.ARENA_CHUNK_ROWS)
            self.cache_w.append(mx.zeros((rows, self.head_dim // 4), dtype=mx.uint32))
            self.cache_s.append(mx.zeros((rows, self.head_dim // 32), dtype=mx.uint8))

    def _gather(self, slots: np.ndarray) -> mx.array:
        chunks = slots // self.ARENA_CHUNK_ROWS
        present = np.unique(chunks)
        out = mx.zeros((slots.size, self.head_dim), dtype=mx.bfloat16)
        for chunk in present:
            positions = np.flatnonzero(chunks == chunk)
            local = mx.array((slots[positions] % self.ARENA_CHUNK_ROWS).astype(np.int32))
            vals = dequant_rows_mx(mx.take(self.cache_w[chunk], local, axis=0),
                                  mx.take(self.cache_s[chunk], local, axis=0))
            if present.size == 1:
                return vals
            out[mx.array(positions.astype(np.int32))] = vals
        return out

    # ----- bookkeeping
    def _victims(self, n: int, protected: np.ndarray) -> np.ndarray:
        old_top = self.free_top
        take = min(self.capacity - old_top, n)
        slots = np.arange(old_top, old_top + take, dtype=np.int64)
        self.free_top += take
        n -= take
        if n <= 0:
            return slots
        # Sample occupied slots only; current lookup hits must survive insertion.
        cand = np.unique(self.rng.integers(0, old_top, size=min(old_top, self.SAMPLE * n)))
        cand = cand[~np.isin(cand, protected)]
        if cand.size < n:
            remaining = np.setdiff1d(np.arange(old_top), np.concatenate([protected, cand]))
            cand = np.concatenate([cand, remaining])
        order = np.lexsort((self.hits[cand], self.last_used[cand]))
        evict = cand[order[:n]]
        self.slot_of[self.row_of[evict]] = -1
        self.hits[evict] = 0
        return np.concatenate([slots, evict])

    def insert(self, ids: np.ndarray, raw: np.ndarray, sc: np.ndarray,
               protected: Optional[np.ndarray] = None) -> np.ndarray:
        """Place unique, absent fetched rows into slots; uncached rows return -1."""
        result = np.full(ids.shape, -1, dtype=np.int64)
        protected = np.empty(0, dtype=np.int64) if protected is None else protected
        n = min(ids.size, self.capacity - protected.size)
        if n <= 0:
            return result
        ids, raw, sc = ids[:n], raw[:n], sc[:n]
        slots = self._victims(n, protected)
        self._ensure_arenas()
        self.row_of[slots] = ids
        self.slot_of[ids] = slots.astype(np.int32)
        self.last_used[slots] = self.tick
        self.hits[slots] = 1
        packed = np.ascontiguousarray(raw).view(np.uint32).reshape(n, self.head_dim // 4)
        chunks = slots // self.ARENA_CHUNK_ROWS
        order = np.argsort(slots, kind="stable")
        boundaries = np.flatnonzero(np.diff(chunks[order])) + 1
        for selected in np.split(order, boundaries):
            chunk = int(chunks[selected[0]])
            local_np = (slots[selected] % self.ARENA_CHUNK_ROWS).astype(np.int32)
            weight, scale = mx.array(packed[selected]), mx.array(np.ascontiguousarray(sc[selected]))
            # Warmup/full-arena inserts can replace a buffer directly. Partial
            # and eviction writes still need one scatter per touched arena.
            if local_np.size == self.cache_w[chunk].shape[0] and np.array_equal(
                    local_np, np.arange(local_np.size)):
                self.cache_w[chunk], self.cache_s[chunk] = weight, scale
            else:
                local = mx.array(local_np)
                self.cache_w[chunk][local] = weight
                self.cache_s[chunk][local] = scale
        result[:n] = slots
        return result

    # ----- the lookup
    def lookup(self, ids: np.ndarray) -> mx.array:
        """ids int64 [N] -> bf16 [N, head_dim]; hits gathered from the cache, misses from NVMe."""
        return self.finish(self.prepare(ids))

    def prepare(self, ids: np.ndarray) -> dict:
        """The host half of a lookup: numpy bookkeeping and the memmap fetch of the misses. No MLX
        op is issued, so this may run on a worker thread (DSV41_STEP_OVERLAP) while the main thread
        builds the graph; one prepare per cache at a time, finished in order."""
        ids = np.asarray(ids, dtype=np.int64)
        self.tick += 1
        self.stat_lookups += 1
        self.stat_requested_rows += ids.size
        uniq, inverse = np.unique(ids, return_inverse=True)
        slots = self.slot_of[uniq].astype(np.int64) if self.capacity else np.full(uniq.shape, -1, np.int64)
        hit = slots >= 0
        if hit.any():
            hs = slots[hit]
            self.last_used[hs] = self.tick
            self.hits[hs] = np.minimum(self.hits[hs].astype(np.int32) + 1, 65535).astype(np.uint16)
        miss = ~hit
        # slot_of, hit and inverse are NumPy: these counters never read a
        # device scalar. Keep both unique-row and request-weighted hit rates.
        self.stat_hits += np.count_nonzero(hit)
        self.stat_misses += np.count_nonzero(miss)
        self.stat_requested_hits += np.count_nonzero(hit[inverse])
        raw = sc = None
        if miss.any():
            t0 = time.time()
            raw, sc = self.source.fetch(uniq[miss])
            self.stat_source_bytes += raw.nbytes + sc.nbytes
            self.stat_fetch_s += time.time() - t0
        return {"uniq": uniq, "inverse": inverse, "slots": slots, "hit": hit, "miss": miss,
                "raw": raw, "sc": sc}

    def finish(self, prepared: dict) -> mx.array:
        """The MLX half of a lookup: insert the fetched misses, gather, un-unique."""
        uniq, inverse, slots = prepared["uniq"], prepared["inverse"], prepared["slots"]
        hit, miss, raw, sc = prepared["hit"], prepared["miss"], prepared["raw"], prepared["sc"]
        if raw is not None:
            if self.capacity:
                new_slots = self.insert(uniq[miss], raw, sc, protected=slots[hit])
                if (new_slots < 0).any():
                    return self._mixed(uniq, inverse, slots, miss, new_slots, raw, sc)
                slots[miss] = new_slots
            else:
                vals = self._fetched_rows(raw, sc)
                return mx.take(vals, mx.array(inverse.astype(np.int32)), axis=0)
        rows = self._gather(slots)
        return mx.take(rows, mx.array(inverse.astype(np.int32)), axis=0)

    def _mixed(self, uniq, inverse, slots, miss, new_slots, raw, sc) -> mx.array:
        vals = mx.zeros((uniq.size, self.head_dim), dtype=mx.bfloat16)
        hit_idx = np.nonzero(~miss)[0]
        if hit_idx.size:
            vals[mx.array(hit_idx.astype(np.int32))] = self._gather(slots[hit_idx])
        vals[mx.array(np.flatnonzero(miss).astype(np.int32))] = self._fetched_rows(raw, sc)
        return mx.take(vals, mx.array(inverse.astype(np.int32)), axis=0)

    def _fetched_rows(self, raw, sc) -> mx.array:
        packed = np.ascontiguousarray(raw).view(np.uint32).reshape(-1, self.head_dim // 4)
        return dequant_rows_mx(mx.array(packed), mx.array(np.ascontiguousarray(sc)))

    # ----- hot set persistence
    def hot_file(self) -> Path:
        return cache_dir() / f"engram_hot_{self.name}_{self.source.num_embeddings}.npy"

    def set_hotness(self, rows, counts) -> int:
        """Seed the hit counters of cached rows from measured counts. A corpus pre-warm already
        knows how often each row was addressed, and save_hot orders by hits; without this every
        pre-warmed row would look equally warm and the hot set would be ordered by row id."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        counts = np.asarray(counts, dtype=np.int64).reshape(-1)
        if rows.size != counts.size:
            raise ValueError("set_hotness: rows and counts must be the same length")
        if rows.size == 0 or self.capacity == 0:
            return 0
        slots = self.slot_of[rows].astype(np.int64)
        live = slots >= 0
        self.hits[slots[live]] = np.clip(counts[live], 1, 65535).astype(np.uint16)
        return int(live.sum())

    def hot_rows(self, max_rows: Optional[int] = None) -> np.ndarray:
        """This session's resident rows, hottest first: hit count, then recency."""
        n = self.free_top
        if n == 0:
            return np.empty(0, dtype=np.int64)
        order = np.lexsort((self.last_used[:n], self.hits[:n]))[::-1]  # hottest first
        rows = self.row_of[:n][order]
        rows = rows[rows >= 0]
        return rows[:max_rows] if max_rows else rows

    def stored_hot(self) -> np.ndarray:
        """The hot set already on disk (hottest first); empty when absent or unreadable."""
        f = self.hot_file()
        try:
            if not f.exists():
                return np.empty(0, dtype=np.int64)
            rows = np.asarray(np.load(f), dtype=np.int64).reshape(-1)
        except (OSError, ValueError):
            return np.empty(0, dtype=np.int64)
        return rows[rows >= 0]

    @staticmethod
    def merge_hot(current: np.ndarray, stored: np.ndarray, limit: int) -> np.ndarray:
        """Hottest first, so successive runs accumulate instead of overwriting: rows hot in BOTH
        this session and the stored set lead (they survived two runs), then the rest of this
        session in hotness order, then the rest of the stored set in its order. Bounded by limit."""
        if stored.size == 0:
            merged = current
        elif current.size == 0:
            merged = stored
        else:
            in_stored = np.isin(current, stored)
            merged = np.concatenate([current[in_stored], current[~in_stored],
                                     stored[~np.isin(stored, current)]])
        _, first = np.unique(merged, return_index=True)  # keep the first (hottest) occurrence
        merged = merged[np.sort(first)]
        return merged[:limit] if limit else merged

    def save_hot(self, max_rows: Optional[int] = None) -> int:
        """Merge this session's hot rows into the stored hot set and write it back, bounded by the
        cache capacity. An empty cache leaves the file alone (nvme runs must not erase it)."""
        current = self.hot_rows()
        if current.size == 0:
            return 0
        limit = self.capacity if max_rows is None else min(self.capacity, int(max_rows))
        stored = self.stored_hot()
        rows = self.merge_hot(current, stored, limit)
        try:
            self.hot_file().parent.mkdir(parents=True, exist_ok=True)
            np.save(self.hot_file(), rows)
        except OSError:
            return 0
        print(f"[engram] save_hot {self.name}: rows={rows.size} (session {current.size}, "
              f"stored {stored.size}, cap {limit}) -> {self.hot_file()}", flush=True)
        return int(rows.size)

    def warm_rows(self, rows, max_rows: Optional[int] = None, batch: int = 1 << 20,
                  label: str = "warm") -> int:
        """Fetch `rows` into the cache in sorted, chunked batches (the shard reads close to
        sequentially). Rows already resident are skipped. Returns the number fetched."""
        started = time.time()
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        if self.capacity == 0 or rows.size == 0:
            return 0
        rows = rows[: min(rows.size, self.capacity, max_rows or rows.size)]
        rows = np.unique(rows)
        done = 0
        for i in range(0, rows.size, batch):
            chunk = rows[i : i + batch]
            chunk = chunk[self.slot_of[chunk] < 0]
            if chunk.size == 0:
                continue
            raw, sc = self.source.fetch(chunk)
            self.insert(chunk, raw, sc)
            done += int(chunk.size)
        elapsed = time.time() - started
        gb = done * (self.head_dim + self.head_dim // 32) / 1e9
        print(f"[engram] {label} {self.name}: rows={done}/{rows.size} {gb:.3f} GB "
              f"{elapsed:.2f}s ({done / max(elapsed, 1e-9):,.0f} rows/s) "
              f"resident={self.free_top}/{self.capacity}", flush=True)
        return done

    def warm(self, max_rows: Optional[int] = None, batch: int = 1 << 20) -> int:
        """Pre-fetch the saved hot set (sorted, so the shard is read close to sequentially)."""
        f = self.hot_file()
        if self.capacity == 0 or not f.exists():
            return 0
        return self.warm_rows(np.load(f), max_rows=max_rows, batch=batch, label="warm hot-set")

    def stats(self) -> dict:
        return {
            "name": self.name, "capacity_rows": self.capacity, "used_rows": int(self.free_top),
            "allocated_rows": self.allocated_rows,
            "capacity_gb": round(self.capacity * self.head_dim * 33 / 32 / 1e9, 6),
            "allocated_gb": round(self.allocated_rows * self.head_dim * 33 / 32 / 1e9, 6),
            "cache_gb": round(self.allocated_rows * self.head_dim * 33 / 32 / 1e9, 6),
            "lookups": self.stat_lookups,
            "row_hits": int(self.stat_hits), "row_misses": int(self.stat_misses),
            "requested_rows": int(self.stat_requested_rows),
            "requested_row_hits": int(self.stat_requested_hits),
            "requested_hit_rate": float(self.stat_requested_hits / max(1, self.stat_requested_rows)),
            "source_payload_bytes": int(self.stat_source_bytes),
            "source_bytes_per_requested_row": float(self.stat_source_bytes / max(1, self.stat_requested_rows)),
            "hit_rate": round(float(self.stat_hits / max(1, self.stat_hits + self.stat_misses)), 4),
            "nvme_fetch_s": round(self.stat_fetch_s, 3),
        }


def _sysctl_int(key: str) -> int:
    return int(subprocess.check_output(["sysctl", "-n", key], text=True).strip() or "0")


def process_memory_io() -> dict:
    """macOS rusage_info_v2 byte counters, sampled only by opt-in diagnostics.

    Disk I/O covers the whole process, including page-ins; it cannot attribute
    individual reads to Engram. Field layout is from sys/resource.h in the SDK.
    """
    import ctypes

    try:
        lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        # 16-byte UUID followed by 18 uint64 fields (RUSAGE_INFO_V2).
        usage = (ctypes.c_uint64 * 20)()
        lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.proc_pid_rusage.restype = ctypes.c_int
        if lib.proc_pid_rusage(os.getpid(), 2, ctypes.byref(usage)) != 0:
            return {"rusage_errno": ctypes.get_errno()}
        return {"process_wired_bytes": int(usage[7]),
                "process_resident_bytes": int(usage[8]),
                "process_footprint_bytes": int(usage[9]),
                "process_disk_read_bytes": int(usage[18])}
    except (OSError, AttributeError):
        return {}


def host_available_bytes() -> Tuple[int, dict, str]:
    """Host-wide available bytes -- free + inactive + speculative + purgeable, the same counters the
    smoke guard uses, so the cache is sized against what the whole box can hold and not just against
    the GPU's wired limit. Returns (bytes, page counters, source); 0 when nothing answers."""
    names = {"free": "Pages free", "inactive": "Pages inactive",
             "speculative": "Pages speculative", "purgeable": "Pages purgeable",
             "wired": "Pages wired down"}
    try:
        output = subprocess.check_output(["vm_stat"], text=True)
        page_size = int(re.search(r"page size of (\d+) bytes", output.splitlines()[0]).group(1))
        counts = {key.strip(): int(value.strip().rstrip("."))
                  for line in output.splitlines()[1:] if ":" in line
                  for key, value in [line.split(":", 1)]}
        memory = {name: counts[label] * page_size for name, label in names.items()}
        return sum(memory[n] for n in names if n != "wired"), memory, "vm_stat"
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, AttributeError):
        pass
    try:
        page_size = _sysctl_int("hw.pagesize")
        physical = _sysctl_int("hw.memsize")
        memory = {name: _sysctl_int(f"vm.page_{'wire' if name == 'wired' else name}_count") * page_size
                  for name in names}
        # Exclude host-wide wired, active and compressor use; cap by reclaimable pages.
        remaining = physical - memory["wired"] - page_size * (
            _sysctl_int("vm.page_active_count") + _sysctl_int("vm.compressor_page_count"))
        available = min(remaining, sum(memory[n] for n in names if n != "wired"))
        return max(0, available), memory, "sysctl"
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return 0, {}, "unknown"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    value = float(raw) if raw else float(default)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def cache_budget_decision(margin_gb: Optional[float] = None, reserve_gb: Optional[float] = None,
                          fraction: Optional[float] = None) -> dict:
    """Every term of the automatic total Engram row-cache budget, in bytes.

    headroom = min(host available, wired limit - MLX active) - margin - reserve, and the budget is
    `fraction` of that, capped by LLAMA_MLX_ENGRAM_CACHE_GB. The host-wide term is what stops a
    fraction near 1.0 from promising memory the box does not have; the wired term is what stops it
    from promising memory Metal will not map."""
    env = os.environ.get("LLAMA_MLX_ENGRAM_CACHE_GB", "").strip()
    cap = float(env) if env else None
    if cap is not None and (not np.isfinite(cap) or cap < 0):
        raise ValueError("LLAMA_MLX_ENGRAM_CACHE_GB must be finite and nonnegative")
    if fraction is None:
        fraction = float(os.environ.get("LLAMA_MLX_ENGRAM_CACHE_FRACTION",
                                        str(DEFAULT_CACHE_FRACTION)) or DEFAULT_CACHE_FRACTION)
    if not 0 <= fraction <= 1:
        raise ValueError("LLAMA_MLX_ENGRAM_CACHE_FRACTION must be in [0, 1]")
    if margin_gb is None:
        margin_gb = _env_float("LLAMA_MLX_ENGRAM_CACHE_MARGIN_GB", DEFAULT_CACHE_MARGIN_GB)
    if reserve_gb is None:
        reserve_gb = _env_float("LLAMA_MLX_ENGRAM_CACHE_RESERVE_GB", DEFAULT_CACHE_RESERVE_GB)
    limit = 0
    try:
        limit = _sysctl_int("iogpu.wired_limit_mb") * 1024 * 1024
    except Exception:  # noqa: BLE001
        pass
    if limit <= 0:
        try:
            limit = int(mx.device_info().get("max_recommended_working_set_size", 0))
        except Exception:  # noqa: BLE001
            limit = 0
    try:
        active = int(mx.get_active_memory())
    except Exception:  # noqa: BLE001
        active = 0
    host, counters, host_source = host_available_bytes()
    held = int((float(margin_gb) + float(reserve_gb)) * 1e9)
    wired_headroom = max(0, limit - active - held) if limit > 0 else None
    host_headroom = max(0, host - held) if host > 0 else None
    candidates = [h for h in (wired_headroom, host_headroom) if h is not None]
    headroom = min(candidates) if candidates else 0
    budget = int(headroom * fraction)
    if cap is not None:
        budget = min(budget, int(cap * 1e9))
    return {"fraction": float(fraction), "margin_gb": float(margin_gb),
            "reserve_gb": float(reserve_gb), "cap_gb": cap,
            "host_available_bytes": int(host), "host_source": host_source,
            "host_counters": counters, "wired_limit_bytes": int(limit),
            "mlx_active_bytes": int(active), "host_headroom_bytes": host_headroom,
            "wired_headroom_bytes": wired_headroom, "headroom_bytes": int(headroom),
            "bound_by": ("cap" if cap is not None and budget == int(cap * 1e9)
                         else "host" if host_headroom is not None and headroom == host_headroom
                         else "wired" if wired_headroom is not None else "unknown"),
            "budget_bytes": int(budget)}


def format_budget_decision(decision: dict) -> str:
    """One line carrying every term that produced the budget."""
    def gb(value):
        return "n/a" if value is None else f"{value / 1e9:.3f}"

    cap = decision["cap_gb"]
    return ("[engram] cache budget: "
            f"host_avail={gb(decision['host_available_bytes'])} GB ({decision['host_source']}) "
            f"wired_limit={gb(decision['wired_limit_bytes'])} GB "
            f"mlx_active={gb(decision['mlx_active_bytes'])} GB "
            f"margin={decision['margin_gb']:.1f} GB reserve={decision['reserve_gb']:.1f} GB "
            f"host_headroom={gb(decision['host_headroom_bytes'])} GB "
            f"wired_headroom={gb(decision['wired_headroom_bytes'])} GB "
            f"headroom={gb(decision['headroom_bytes'])} GB "
            f"x fraction={decision['fraction']:.3f} "
            f"cap={'auto' if cap is None else f'{cap:.3f} GB'} "
            f"bound_by={decision['bound_by']} "
            f"-> budget={gb(decision['budget_bytes'])} GB")


def cache_rows_for_budget(budget_bytes: int, num_embeddings: int) -> int:
    """How many rows a per-table byte budget buys. `slot_of` covers the WHOLE table whatever the
    capacity (4 bytes x 384M rows = 1.54 GB a table), so it comes off the top, not per row."""
    per_row = ROW_BYTES + 8 + 4 + 2 + 4  # slot payload + row_of + last_used + hits + slack
    fixed = int(num_embeddings) * 4
    return int(min(int(num_embeddings), max(0, int(budget_bytes) - fixed) // per_row))


def auto_cache_budget_bytes(margin_gb: Optional[float] = None,
                            reserve_gb: Optional[float] = None) -> int:
    """Total model cache budget, printing the decision line unless LLAMA_MLX_ENGRAM_CACHE_QUIET."""
    decision = cache_budget_decision(margin_gb, reserve_gb)
    if os.environ.get("LLAMA_MLX_ENGRAM_CACHE_QUIET", "").strip() not in ("1", "true", "yes"):
        print(format_budget_decision(decision), flush=True)
    return decision["budget_bytes"]


# --------------------------------------------------------------------------- the module


class Engram(nn.Module):
    """Writes an n-gram lookup into the residual stream, gated by how well it matches that stream.

    The hash ids fetch `n_cols` rows; `wkv` turns them into one key per hc copy plus a shared value.
    The gate is a normalized dot product of stream against key. x: [B, L, hc_mult, dim]."""

    def __init__(self, config, layer_id: int, layout: EngramLayout):
        super().__init__()
        self.layer_id = int(layer_id)
        self.table_count = len(layout.layer_ids)
        self.layer_hash_index = layout.layer_ids.index(self.layer_id)
        self.dim = int(config.hidden_size)
        self.hc_mult = int(config.hc_mult)
        self.eps = float(config.rms_norm_eps)
        self.clamp_value = 1e-6
        self.head_dim = layout.head_dim
        self.n_cols = layout.n_cols
        self.num_embeddings = layout.num_embeddings[self.layer_hash_index]
        self.source = (os.environ.get("LLAMA_MLX_ENGRAM_SOURCE") or getattr(config, "engram_source", "hybrid") or "hybrid").strip().lower()
        if self.source not in ("ram", "nvme", "hybrid"):
            raise ValueError(f"engram_source must be ram, nvme or hybrid, got {self.source!r}")
        n_chunks = int(getattr(config, "engram_ram_chunks", 8) or 8)
        self.embed = EngramTableRAM(self.num_embeddings, self.head_dim, n_chunks) if self.source == "ram" else None
        self.rows_cache: Optional[EngramRowCache] = None
        self.shard: Optional[ShardRows] = None
        self._lookup_calls = 0
        self._lookup_tokens = 0
        self._lookup_rows = 0
        self._report = os.environ.get("LLAMA_MLX_ENGRAM_REPORT", "") == "1"
        self._last_phase = None
        self.wkv = nn.Linear(self.n_cols * self.head_dim, self.dim * (self.hc_mult + 1), bias=False)
        self.q_weight = mx.ones((self.hc_mult, self.dim), dtype=mx.bfloat16)
        self.k_weight = mx.ones((self.hc_mult, self.dim), dtype=mx.bfloat16)

    @property
    def resident(self) -> bool:
        return self.source == "ram"

    def attach_storage(self, model_dir: str | Path, cache_budget_bytes: Optional[int] = None,
                       *, table_count: Optional[int] = None) -> dict:
        """Open the shard memory map (nvme/hybrid) and size the row cache (hybrid). Call once after
        the weights are loaded. Auto budgets are divided across layout tables; an explicit
        cache_budget_bytes is already per-table. Sequential warmups may reduce later auto budgets."""
        if self.source == "ram":
            if self._report:
                mx.eval(self.embed.parameters())
                self._report_snapshot("attached")
            return {"source": self.source}
        if self.shard is not None:
            return {"source": self.source}
        prefix = f"layers.{self.layer_id}.engram.embed"
        self.shard = ShardRows(model_dir, prefix + ".weight", prefix + ".scale", self.num_embeddings, self.head_dim)
        rows, budget, total, decision = 0, 0, None, None
        count = self.table_count if table_count is None else int(table_count)
        if count < 1:
            raise ValueError("table_count must be positive")
        if self.source == "hybrid":
            if cache_budget_bytes is None:
                decision = cache_budget_decision()
                if os.environ.get("LLAMA_MLX_ENGRAM_CACHE_QUIET", "").strip() not in ("1", "true", "yes"):
                    print(format_budget_decision(decision), flush=True)
                total = decision["budget_bytes"]
                budget = total // count
            else:
                budget = max(0, int(cache_budget_bytes))
            rows = cache_rows_for_budget(budget, self.num_embeddings)
        self.rows_cache = EngramRowCache(self.shard, rows, self.head_dim, name=f"L{self.layer_id}")
        return {**self.rows_cache.stats(), "source": self.source, "cache_rows": rows,
                "table_count": count, "budget_bytes": budget, "total_budget_bytes": total,
                "budget_mode": "auto" if cache_budget_bytes is None else "per_table",
                "budget_decision": decision, "maps_bytes": self.num_embeddings * 4,
                "cache_fraction": float(os.environ.get("LLAMA_MLX_ENGRAM_CACHE_FRACTION",
                                                       str(DEFAULT_CACHE_FRACTION))),
                "cache_margin_gb": _env_float("LLAMA_MLX_ENGRAM_CACHE_MARGIN_GB", DEFAULT_CACHE_MARGIN_GB),
                "cache_reserve_gb": _env_float("LLAMA_MLX_ENGRAM_CACHE_RESERVE_GB", DEFAULT_CACHE_RESERVE_GB),
                "cache_cap_gb": os.environ.get("LLAMA_MLX_ENGRAM_CACHE_GB")}

    def warm(self) -> int:
        count = self.rows_cache.warm() if self.rows_cache is not None else 0
        if self._report and self.rows_cache is not None:
            mx.eval(self.rows_cache.cache_w, self.rows_cache.cache_s)
            self._report_snapshot("attached")
        return count

    def save_hot(self) -> int:
        return self.rows_cache.save_hot() if self.rows_cache is not None else 0

    def stats(self) -> dict:
        stats = self.rows_cache.stats() if self.rows_cache is not None else {
            "requested_rows": self._lookup_rows, "requested_row_hits": self._lookup_rows,
            "requested_hit_rate": 1.0 if self._lookup_rows else 0.0,
            "source_payload_bytes": 0,
        }
        return {**stats, "source": self.source, "layer": self.layer_id,
                "lookup_calls": self._lookup_calls, "lookup_tokens": self._lookup_tokens,
                "logical_payload_bytes": self._lookup_rows * (self.head_dim + self.head_dim // 32),
                "source_payload_bytes_per_token": stats.get("source_payload_bytes", 0) / max(1, self._lookup_tokens)}

    def _report_snapshot(self, event: str) -> None:
        """Opt-in diagnostics at load/phase boundaries, never device scalar reads per step."""
        record = {"event": event, "run_id": os.environ.get("LLAMA_MLX_ENGRAM_RUN_ID"),
                  **self.stats(), **process_memory_io()}
        if event == "attached":
            _, counters, _ = host_available_bytes()
            record["host_wired_bytes"] = counters.get("wired")
            record["mlx_active_bytes"] = mx.get_active_memory()
            record["mlx_cache_bytes"] = mx.get_cache_memory()
        print("[engram-s3] " + json.dumps(record), flush=True)

    def rows_prepare(self, hash_ids: np.ndarray):
        """Host half of rows() for the hybrid/nvme sources (worker-thread safe); None for `ram`."""
        if self.source == "ram":
            return None
        if self.rows_cache is None:
            raise RuntimeError("engram storage not attached: call attach_storage(model_dir) after loading")
        if _LEDGER_ON:
            _t0 = _ledger_time.perf_counter()
            prepared = self.rows_cache.prepare(hash_ids.reshape(-1))
            LEDGER["engram_prefetch_ms"] = LEDGER.get("engram_prefetch_ms", 0.0) + (
                _ledger_time.perf_counter() - _t0) * 1000.0
            return prepared
        return self.rows_cache.prepare(hash_ids.reshape(-1))

    def rows(self, hash_ids: np.ndarray, prepared=None) -> mx.array:
        """hash_ids int64 [B, L, n_cols] -> bf16 [B, L, n_cols * head_dim]."""
        b, l, c = hash_ids.shape
        phase = "decode" if l == 1 else "prefill"
        if self._report and self._last_phase is not None and phase != self._last_phase:
            self._report_snapshot(self._last_phase + "_end")
        self._last_phase = phase
        self._lookup_calls += 1
        self._lookup_tokens += b * l
        self._lookup_rows += b * l * c
        flat = hash_ids.reshape(-1)
        if self.source == "ram":
            vals = self.embed(flat)
        else:
            if self.rows_cache is None:
                raise RuntimeError("engram storage not attached: call attach_storage(model_dir) after loading")
            vals = (self.rows_cache.finish(prepared) if prepared is not None
                    else self.rows_cache.lookup(flat))
        return vals.reshape(b, l, c * self.head_dim)

    def __call__(self, x: mx.array, hash_ids: np.ndarray, token_mask: Optional[np.ndarray] = None,
                 prepared=None) -> mx.array:
        if _LEDGER_ON or (_PREFILL_LEDGER_ON and x.shape[1] > 1):
            _t0 = _ledger_time.perf_counter()
            rows = self.rows(hash_ids, prepared)
            LEDGER["engram_host_ms"] += (_ledger_time.perf_counter() - _t0) * 1000.0
            LEDGER["engram_calls"] += 1
        else:
            rows = self.rows(hash_ids, prepared)
        kv = self.wkv(rows)
        key, value = mx.split(kv, [self.hc_mult * self.dim], axis=-1)
        key = key.astype(mx.float32).reshape(*key.shape[:-1], self.hc_mult, self.dim)
        weight = self.q_weight.astype(mx.float32) * self.k_weight.astype(mx.float32)  # only used as a product
        h = x.astype(mx.float32)
        # normalized per (token, hc copy) over dim, NOT jointly over the copies
        rstd = mx.rsqrt(mx.mean(mx.square(h), axis=-1) + self.eps) * mx.rsqrt(mx.mean(mx.square(key), axis=-1) + self.eps)
        dot = mx.sum(h * weight * key, axis=-1) * rstd * (self.dim ** -0.5)
        # signed sqrt before the sigmoid (copysign: a zero keeps the positive branch), as in training
        signed = mx.where(dot < 0, -1.0, 1.0) * mx.sqrt(mx.maximum(mx.abs(dot), self.clamp_value))
        gate = mx.sigmoid(signed)
        if token_mask is not None:
            gate = gate * mx.array(np.asarray(token_mask, dtype=np.float32))[..., None]
        return (h + gate[..., None] * value.astype(mx.float32)[..., None, :]).astype(x.dtype)
