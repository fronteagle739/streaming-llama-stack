# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/weight_streaming/magi2_expert_cache.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
Key = Tuple[int, int]

@dataclass
class CacheStats:
    hits: int = 0
    vram_hits: int = 0
    ram_hits: int = 0
    misses: int = 0
    evictions: int = 0
    ram_evictions: int = 0
    demotions: int = 0
    bytes_streamed: int = 0
    bytes_saved: int = 0
    prefetch_hits: int = 0
    prefetch_misses: int = 0

    def as_dict(self) -> Dict[str, int]:
        return dict(self.__dict__)

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        n = self.total_requests
        return self.hits / n if n else 0.0

@dataclass
class _Tier:
    capacity_bytes: int
    slot_bytes: int
    resident: 'OrderedDict[Key, int]' = field(default_factory=OrderedDict)
    used_bytes: int = 0

    @property
    def slots(self) -> int:
        return self.capacity_bytes // self.slot_bytes if self.slot_bytes else 0

    def __contains__(self, key: Key) -> bool:
        return key in self.resident

    def touch(self, key: Key) -> None:
        self.resident.move_to_end(key)

    def would_evict(self, incoming_bytes: int) -> bool:
        return self.used_bytes + incoming_bytes > self.capacity_bytes

    def pop_lru(self, pinned: Optional[set]=None) -> Optional[Tuple[Key, int]]:
        pinned = pinned or set()
        for key in list(self.resident.keys()):
            if key in pinned:
                continue
            nbytes = self.resident.pop(key)
            self.used_bytes -= nbytes
            return (key, nbytes)
        return None

    def insert(self, key: Key, nbytes: int) -> None:
        self.resident[key] = nbytes
        self.resident.move_to_end(key)
        self.used_bytes += nbytes

class Magi2MoEStreamCache:

    def __init__(self, *, vram_budget_bytes: int, host_budget_bytes: int=0, shard_index=None, expert_nbytes: Optional[int]=None, num_experts: int=256, prefetch: bool=False, prefetch_alpha: float=0.25, prefetch_top_p: int=0) -> None:
        if shard_index is not None and expert_nbytes is None:
            expert_nbytes = shard_index.expert_nbytes()
            num_experts = shard_index.num_experts
        if not expert_nbytes or expert_nbytes <= 0:
            raise ValueError('expert_nbytes must be > 0 (pass it or a shard_index)')
        self.shard_index = shard_index
        self.expert_nbytes = int(expert_nbytes)
        self.num_experts = num_experts
        if vram_budget_bytes < self.expert_nbytes:
            raise ValueError(f'vram_budget_bytes {vram_budget_bytes} < one expert {self.expert_nbytes}')
        self.vram = _Tier(capacity_bytes=vram_budget_bytes, slot_bytes=self.expert_nbytes)
        self.ram = _Tier(capacity_bytes=host_budget_bytes, slot_bytes=self.expert_nbytes) if host_budget_bytes >= self.expert_nbytes else None
        self.stats = CacheStats()
        self.active: Dict[int, List[int]] = {}
        self.prefetch = prefetch
        self.prefetch_alpha = prefetch_alpha
        self.prefetch_top_p = prefetch_top_p
        self._ema: Dict[int, Dict[int, float]] = defaultdict(dict)
        self._predicted: Dict[int, set] = {}

    @property
    def vram_slots(self) -> int:
        return self.vram.slots

    @property
    def ram_slots(self) -> int:
        return self.ram.slots if self.ram else 0

    def layer_fits_vram(self) -> bool:
        return self.vram_slots >= self.num_experts * (self.shard_index.num_heads if self.shard_index else 1)

    def set_active_experts(self, layer_idx: int, expert_ids: Sequence[int]) -> None:
        self.active[layer_idx] = sorted(set((int(e) for e in expert_ids)))

    def _admit(self, key: Key, *, pin: Optional[set]) -> None:
        while self.vram.would_evict(self.expert_nbytes):
            victim = self.vram.pop_lru(pinned=pin)
            if victim is None:
                break
            self.stats.evictions += 1
            if self.ram is not None:
                self._admit_ram(victim[0], victim[1])
                self.stats.demotions += 1
        self.vram.insert(key, self.expert_nbytes)

    def _admit_ram(self, key: Key, nbytes: int) -> None:
        assert self.ram is not None
        while self.ram.would_evict(nbytes):
            rv = self.ram.pop_lru()
            if rv is None:
                break
            self.stats.ram_evictions += 1
        self.ram.insert(key, nbytes)

    def request(self, layer_idx: int, flat_expert_id: int, *, pin: Optional[set]=None) -> bool:
        key = (layer_idx, flat_expert_id)
        if key in self.vram:
            self.vram.touch(key)
            self.stats.hits += 1
            self.stats.vram_hits += 1
            self.stats.bytes_saved += self.expert_nbytes
            return True
        if self.ram is not None and key in self.ram:
            nb = self.ram.resident.pop(key)
            self.ram.used_bytes -= nb
            self.stats.hits += 1
            self.stats.ram_hits += 1
            self.stats.bytes_saved += self.expert_nbytes
            self._admit(key, pin=pin)
            return True
        self.stats.misses += 1
        self.stats.bytes_streamed += self.expert_nbytes
        self._admit(key, pin=pin)
        return False

    def observe_routing(self, layer_idx: int, flat_expert_ids: Iterable[int]) -> None:
        ema = self._ema[layer_idx]
        for k in list(ema):
            ema[k] *= 1.0 - self.prefetch_alpha
        for e in flat_expert_ids:
            ema[e] = ema.get(e, 0.0) + self.prefetch_alpha

    def predict(self, layer_idx: int, top_p: Optional[int]=None) -> List[int]:
        top_p = self.prefetch_top_p if top_p is None else top_p
        ema = self._ema.get(layer_idx, {})
        if not ema or top_p <= 0:
            return []
        return [e for e, _ in sorted(ema.items(), key=lambda kv: -kv[1])[:top_p]]

    def dry_run_layer(self, layer_idx: int, active_flat_ids: Sequence[int], *, pin_active: bool=True, record_prefetch: bool=True) -> Dict[str, int]:
        ids = sorted(set((int(e) for e in active_flat_ids)))
        self.set_active_experts(layer_idx, ids)
        before = self.stats.as_dict()
        if self.prefetch and record_prefetch:
            predicted = self._predicted.get(layer_idx, set())
            for e in ids:
                if (layer_idx, e) in self.vram or e in predicted:
                    self.stats.prefetch_hits += 1
                else:
                    self.stats.prefetch_misses += 1
        pin: set = set() if not pin_active else set()
        for e in ids:
            key = (layer_idx, e)
            self.request(layer_idx, e, pin=pin if pin_active else None)
            if pin_active:
                pin.add(key)
        if self.prefetch:
            self.observe_routing(layer_idx, ids)
            self._predicted[layer_idx] = set(self.predict(layer_idx))
        after = self.stats.as_dict()
        return {k: after[k] - before[k] for k in after}

    def reset_stats(self) -> None:
        self.stats = CacheStats()

def simulate_denoise(cache: Magi2MoEStreamCache, routing: Sequence[Dict[int, Sequence[int]]], *, pin_active: bool=True) -> CacheStats:
    for step in routing:
        for layer_idx in sorted(step):
            cache.dry_run_layer(layer_idx, step[layer_idx], pin_active=pin_active)
    return cache.stats
