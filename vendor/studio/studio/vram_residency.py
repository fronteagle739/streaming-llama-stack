# Exported Studio VRAM component manager; portable standalone source.
from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

def _default_free_total_gb(device) -> tuple[float, float]:
    try:
        import torch
        free_b, total_b = torch.cuda.mem_get_info(device)
        return (free_b / 1000000000.0, total_b / 1000000000.0)
    except Exception:
        return (1000000000.0, 1000000000.0)

def _default_empty_cache() -> None:
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

@dataclass
class _Entry:
    key: str
    obj: object
    est_gb: float
    seq: int
    pinned: bool = False
    on_evict: Optional[Callable[[object], None]] = None

@dataclass
class VramResidencyManager:
    device: object = 'cuda:0'
    headroom_gb: float = 1.5
    free_total_gb: Callable[[object], tuple[float, float]] = _default_free_total_gb
    empty_cache: Callable[[], None] = _default_empty_cache
    _entries: dict = field(default_factory=dict)
    _seq: int = 0
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def free_gb(self) -> float:
        return self.free_total_gb(self.device)[0]

    def resident_keys(self) -> list:
        with self._lock:
            return list(self._entries.keys())

    def is_resident(self, key: str) -> bool:
        with self._lock:
            return key in self._entries

    def resident_est_gb(self) -> float:
        with self._lock:
            return sum((e.est_gb for e in self._entries.values()))

    def acquire(self, key: str, loader: Callable[[], object], *, est_gb: float=0.0, pin: bool=True, on_evict: Optional[Callable[[object], None]]=None) -> object:
        with self._lock:
            e = self._entries.get(key)
            if e is not None:
                self._seq += 1
                e.seq = self._seq
                if pin:
                    e.pinned = True
                return e.obj
            self.evict_to_fit(est_gb + self.headroom_gb, protect=())
            obj = loader()
            self._seq += 1
            self._entries[key] = _Entry(key=key, obj=obj, est_gb=float(est_gb), seq=self._seq, pinned=pin, on_evict=on_evict)
            return obj

    def release(self, key: str) -> None:
        with self._lock:
            e = self._entries.get(key)
            if e is not None:
                e.pinned = False
                self._seq += 1
                e.seq = self._seq

    def touch(self, key: str) -> None:
        with self._lock:
            e = self._entries.get(key)
            if e is not None:
                self._seq += 1
                e.seq = self._seq

    def evict(self, key: str) -> bool:
        with self._lock:
            e = self._entries.pop(key, None)
        if e is None:
            return False
        if e.on_evict is not None:
            try:
                e.on_evict(e.obj)
            except Exception:
                pass
        e.obj = None
        self.empty_cache()
        return True

    def evict_to_fit(self, need_gb: float, *, protect: tuple=()) -> float:
        if need_gb <= 0:
            return self.free_gb()
        protect = set(protect)
        while self.free_gb() < need_gb:
            with self._lock:
                candidates = [e for e in self._entries.values() if not e.pinned and e.key not in protect]
                if not candidates:
                    break
                victim = min(candidates, key=lambda e: e.seq)
                self._entries.pop(victim.key, None)
            if victim.on_evict is not None:
                try:
                    victim.on_evict(victim.obj)
                except Exception:
                    pass
            victim.obj = None
            self.empty_cache()
        return self.free_gb()

    def clear(self) -> None:
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for e in entries:
            if e.on_evict is not None:
                try:
                    e.on_evict(e.obj)
                except Exception:
                    pass
            e.obj = None
        self.empty_cache()
