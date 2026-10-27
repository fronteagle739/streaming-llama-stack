from __future__ import annotations
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
CHUNK_SIZE_TOKENS = 256
try:
    import xxhash as _xxhash

    def _xxh64(data: bytes, seed: int=0) -> int:
        return _xxhash.xxh64(data, seed=seed & 18446744073709551615).intdigest()
    _HAVE_XXHASH = True
except Exception:
    _HAVE_XXHASH = False
    _FNV_OFFSET = 14695981039346656037
    _FNV_PRIME = 1099511628211

    def _xxh64(data: bytes, seed: int=0) -> int:
        h = (_FNV_OFFSET ^ seed & 18446744073709551615) & 18446744073709551615
        for b in data:
            h = (h ^ b) * _FNV_PRIME & 18446744073709551615
        return h

def _tokens_to_bytes(token_ids: List[int]) -> bytes:
    import struct
    return struct.pack(f'<{len(token_ids)}I', *(int(t) & 4294967295 for t in token_ids))

def image_hash(raw_bytes: bytes) -> str:
    if not raw_bytes:
        return ''
    return f'{_xxh64(bytes(raw_bytes), seed=0):016x}'

def chunk_hashes(token_ids: List[int], seed: int) -> List[int]:
    out: List[int] = []
    h = seed & 18446744073709551615
    n_full = len(token_ids) // CHUNK_SIZE_TOKENS
    for c in range(n_full):
        lo = c * CHUNK_SIZE_TOKENS
        chunk = token_ids[lo:lo + CHUNK_SIZE_TOKENS]
        h = _xxh64(_tokens_to_bytes(chunk), seed=h)
        out.append(h)
    return out

def model_key_for(model_path: str, quant_config: Optional[Dict[str, Any]]=None, mlx_lm_version: str='') -> str:
    norm_path = os.path.basename(os.path.normpath(model_path)) if model_path else ''
    major = (mlx_lm_version or '').split('.')[0] if mlx_lm_version else ''
    qc = ''
    if quant_config:
        try:
            qc = json.dumps(quant_config, sort_keys=True, separators=(',', ':'))
        except Exception:
            qc = str(quant_config)
    payload = f'{norm_path}\x00{qc}\x00mlxlm{major}'.encode('utf-8')
    return f'{_xxh64(payload, seed=0):016x}'

def kv_quant_flags(cache: List[Any]) -> Dict[str, Any]:
    classes: List[str] = []
    bits: List[int] = []
    group_sizes: List[int] = []
    for c in cache:
        classes.append(type(c).__name__)
        b = getattr(c, 'bits', None)
        if b is not None:
            bits.append(int(b))
        gs = getattr(c, 'group_size', None)
        if gs is not None:
            group_sizes.append(int(gs))
    return {'n_layers': len(cache), 'classes': classes, 'bits': sorted(set(bits)), 'group_sizes': sorted(set(group_sizes))}

class _Entry:
    __slots__ = ('chain_hash', 'token_ids', 'n_tokens', 'chunk_hashes', 'kv_quant', 'bytes', 'last_used', 'disk_path', 'meta_path', 'image_hashes', 'image_positions')

    def __init__(self, chain_hash: str, token_ids: List[int], chunk_hashes_list: List[int], kv_quant: Dict[str, Any], nbytes: int, last_used: float, disk_path: Optional[str], meta_path: Optional[str], image_hashes: Optional[List[str]]=None, image_positions: Optional[List[List[int]]]=None):
        self.chain_hash = chain_hash
        self.token_ids = token_ids
        self.n_tokens = len(token_ids)
        self.chunk_hashes = chunk_hashes_list
        self.kv_quant = kv_quant
        self.bytes = nbytes
        self.last_used = last_used
        self.disk_path = disk_path
        self.meta_path = meta_path
        self.image_hashes: List[str] = list(image_hashes) if image_hashes else []
        self.image_positions: List[List[int]] = [list(p) for p in image_positions] if image_positions else []

class PromptCacheStore:

    def __init__(self, cache_dir: Optional[str], model_key: str, *, hot_size: int=8, max_gb: float=500.0, lcp_reuse_min: int=64, log=None):
        self.model_key = model_key
        self.hot_size = max(0, int(hot_size))
        self.max_bytes = int(max(0.0, float(max_gb)) * 1024 ** 3)
        self.lcp_reuse_min = int(lcp_reuse_min)
        self._log = log or (lambda m: (sys.stderr.write(m), sys.stderr.flush()))
        self._lock = threading.RLock()
        self._seed = int(model_key, 16) if model_key else 0
        self._hot: 'OrderedDict[str, List[Any]]' = OrderedDict()
        self._index: Dict[str, _Entry] = {}
        self.stats = {'hits': 0, 'misses': 0, 'tokens_skipped': 0, 'saves': 0, 'evictions': 0}
        self.enabled = bool(cache_dir)
        self.dir: Optional[Path] = None
        if self.enabled:
            self.dir = Path(os.path.expanduser(cache_dir)) / model_key
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
                self._load_index_from_disk()
            except Exception as e:
                self._log(f'[prompt-cache] init failed for {self.dir}: {e}; disabling\n')
                self.enabled = False
                self.dir = None

    def _disk_path(self, chain_hash: str) -> Path:
        assert self.dir is not None
        return self.dir / f'{chain_hash}.safetensors'

    def _meta_path(self, chain_hash: str) -> Path:
        assert self.dir is not None
        return self.dir / f'{chain_hash}.meta.json'

    def _load_index_from_disk(self) -> None:
        assert self.dir is not None
        total = 0
        for meta_file in self.dir.glob('*.meta.json'):
            try:
                meta = json.loads(meta_file.read_text())
            except Exception:
                continue
            chain_hash = meta_file.name[:-len('.meta.json')]
            blob = self._disk_path(chain_hash)
            if not blob.exists():
                try:
                    meta_file.unlink()
                except Exception:
                    pass
                continue
            tok = meta.get('token_ids') or []
            nbytes = int(meta.get('bytes', 0)) or blob.stat().st_size
            ent = _Entry(chain_hash=chain_hash, token_ids=list(tok), chunk_hashes_list=list(meta.get('chunk_hashes', [])), kv_quant=meta.get('kv_quant', {}), nbytes=nbytes, last_used=float(meta.get('last_used', meta_file.stat().st_mtime)), disk_path=str(blob), meta_path=str(meta_file), image_hashes=list(meta.get('image_hashes', []) or []), image_positions=list(meta.get('image_positions', []) or []))
            self._index[chain_hash] = ent
            total += nbytes
        if self._index:
            self._log(f'[prompt-cache] index loaded: {len(self._index)} entries, {total / 1024 ** 2:.1f} MB on disk for model_key={self.model_key}\n')

    def _atomic_write_bytes(self, path: Path, write_fn, suffix: str='.tmp') -> None:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=suffix)
        os.close(fd)
        try:
            written = write_fn(tmp)
            written = str(written) if written else tmp
            if written != tmp:
                try:
                    os.remove(tmp)
                except FileNotFoundError:
                    pass
            os.replace(written, str(path))
        except Exception:
            for p in (tmp, tmp + '.safetensors'):
                try:
                    os.remove(p)
                except Exception:
                    pass
            raise

    def _current_disk_bytes(self) -> int:
        return sum((e.bytes for e in self._index.values()))

    def _evict_disk_if_needed(self, protect: Optional[str]=None) -> None:
        if self.max_bytes <= 0:
            return
        while self._current_disk_bytes() > self.max_bytes:
            victims = sorted((e for k, e in self._index.items() if k != protect), key=lambda e: e.last_used)
            if not victims:
                break
            v = victims[0]
            self._drop_entry(v.chain_hash)
            self.stats['evictions'] += 1

    def _drop_entry(self, chain_hash: str) -> None:
        ent = self._index.pop(chain_hash, None)
        self._hot.pop(chain_hash, None)
        if ent is None:
            return
        for p in (ent.disk_path, ent.meta_path):
            if p:
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass
                except Exception as e:
                    self._log(f'[prompt-cache] evict unlink {p} failed: {e}\n')

    def _touch_hot(self, chain_hash: str, cache: List[Any]) -> None:
        if self.hot_size <= 0:
            return
        self._hot.pop(chain_hash, None)
        self._hot[chain_hash] = cache
        while len(self._hot) > self.hot_size:
            self._hot.popitem(last=False)

    def lookup(self, token_ids: List[int], image_hashes: Optional[List[str]]=None) -> Optional[Tuple[List[Any], int, _Entry]]:
        if not self.enabled:
            return None
        req_imgs: List[str] = list(image_hashes) if image_hashes else []
        with self._lock:
            cand_hashes = chunk_hashes(token_ids, self._seed)
            best: Optional[_Entry] = None
            best_depth = 0
            best_subchunk: Optional[_Entry] = None
            for ent in self._index.values():
                if ent.image_hashes != req_imgs:
                    continue
                ec = ent.chunk_hashes
                depth = 0
                lim = min(len(ec), len(cand_hashes))
                while depth < lim and ec[depth] == cand_hashes[depth]:
                    depth += 1
                if depth > 0:
                    if depth > best_depth or (depth == best_depth and best is not None and (ent.n_tokens > best.n_tokens)):
                        best = ent
                        best_depth = depth
                elif ent.n_tokens < CHUNK_SIZE_TOKENS:
                    lcp0 = _lcp(token_ids, ent.token_ids)
                    if lcp0 >= self.lcp_reuse_min and (best_subchunk is None or ent.n_tokens > best_subchunk.n_tokens):
                        best_subchunk = ent
            if best is None:
                best = best_subchunk
            if best is None:
                self.stats['misses'] += 1
                return None
            lcp = _lcp(token_ids, best.token_ids)
            if lcp < self.lcp_reuse_min:
                self.stats['misses'] += 1
                self._log(f'[prompt-cache] lookup: chain matched but LCP={lcp} < min={self.lcp_reuse_min} (entry n={best.n_tokens}); miss\n')
                return None
            if best.image_hashes and lcp < best.n_tokens:
                self.stats['misses'] += 1
                self._log(f'[prompt-cache] lookup: image entry but LCP={lcp} < n_tokens={best.n_tokens} (mid-prefix image reuse is phase-2); miss\n')
                return None
            if lcp >= len(token_ids):
                lcp = len(token_ids) - 1
                if lcp < self.lcp_reuse_min:
                    self.stats['misses'] += 1
                    return None
            needs_trim = lcp < best.n_tokens
            cache = self._restore_cache(best, force_disk=needs_trim)
            if cache is None:
                self._drop_entry(best.chain_hash)
                self.stats['misses'] += 1
                return None
            cache_target = lcp + _image_cell_total(best.image_positions)
            trimmed_to = self._trim_cache_to(cache, cache_target)
            if trimmed_to is None:
                self.stats['misses'] += 1
                return None
            lcp = trimmed_to - _image_cell_total(best.image_positions)
            best.last_used = time.time()
            if not needs_trim:
                self._touch_hot(best.chain_hash, cache)
            self.stats['hits'] += 1
            self.stats['tokens_skipped'] += lcp
            self._log(f'[prompt-cache] HIT: restored {best.n_tokens}-tok entry, reusing LCP={lcp} (skip {lcp} prefill tok), suffix={len(token_ids) - lcp} tok to prefill\n')
            return (cache, lcp, best)

    def _restore_cache(self, ent: _Entry, force_disk: bool=False) -> Optional[List[Any]]:
        if not force_disk:
            hot = self._hot.get(ent.chain_hash)
            if hot is not None:
                self._hot.move_to_end(ent.chain_hash)
                return hot
        if not ent.disk_path or not os.path.exists(ent.disk_path):
            hot = self._hot.get(ent.chain_hash)
            if hot is not None:
                self._hot.move_to_end(ent.chain_hash)
                import copy as _copy
                return [_copy.copy(c) for c in hot]
            return None
        try:
            from mlx_lm.models.cache import load_prompt_cache
            import mlx.core as _mx
            _orig_load = _mx.load

            def _patched_load(*a, **k):
                res = _orig_load(*a, **k)
                try:
                    if isinstance(res, tuple) and len(res) == 2 and hasattr(res[1], 'get'):
                        _ej = res[1].get('1.empty_arrays') or res[1].get('empty_arrays')
                        if _ej:
                            for _k, (_shape, _dt) in json.loads(_ej).items():
                                res[0][_k] = _mx.zeros(tuple(_shape), dtype=getattr(_mx, _dt, _mx.float16))
                except Exception:
                    pass
                return res
            _mx.load = _patched_load
            try:
                cache, meta = load_prompt_cache(ent.disk_path, return_metadata=True)
            finally:
                _mx.load = _orig_load
        except Exception as e:
            self._log(f'[prompt-cache] load {ent.disk_path} failed: {e}\n')
            return None
        disk_flags = {}
        try:
            disk_flags = json.loads(meta.get('kv_quant', '{}')) if meta else {}
        except Exception:
            disk_flags = ent.kv_quant
        live_flags = kv_quant_flags(cache)
        if disk_flags and (not _flags_compatible(disk_flags, live_flags)):
            self._log(f'[prompt-cache] REFUSE restore (kv-quant mismatch): disk={disk_flags} live={live_flags}\n')
            return None
        return cache

    def _trim_cache_to(self, cache: List[Any], target_offset: int) -> Optional[int]:
        try:
            from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache
        except Exception as e:
            self._log(f'[prompt-cache] trim import failed: {e}\n')
            return None
        cur = _cache_offset(cache)
        extra = cur - target_offset
        if extra < 0:
            self._log(f'[prompt-cache] trim: offset {cur} < target {target_offset}; bail\n')
            return None
        if extra == 0:
            return cur
        if not can_trim_prompt_cache(cache):
            self._log('[prompt-cache] cache not trimmable; cannot reuse as prefix\n')
            return None
        trim_prompt_cache(cache, extra)
        return _cache_offset(cache)

    def save(self, cache: List[Any], prompt_token_ids: List[int], generated_ids: Optional[List[int]]=None, image_hashes: Optional[List[str]]=None, image_positions: Optional[List[List[int]]]=None) -> bool:
        if not self.enabled or not prompt_token_ids:
            return False
        n_prompt = len(prompt_token_ids)
        img_hashes: List[str] = list(image_hashes) if image_hashes else []
        img_positions: List[List[int]] = [list(p) for p in image_positions] if image_positions else []
        with self._lock:
            chash_list = chunk_hashes(prompt_token_ids, self._seed)
            if chash_list:
                base = chash_list[-1]
            else:
                base = _xxh64(_tokens_to_bytes(prompt_token_ids), seed=self._seed)
            if img_hashes:
                key_seed = base
                for h in img_hashes:
                    key_seed = _xxh64(h.encode('ascii'), seed=key_seed)
                chain_hash = f'{key_seed:016x}'
            else:
                chain_hash = f'{base:016x}'
            cache_boundary = n_prompt + _image_cell_total(img_positions)
            saved_offset = self._save_prompt_pure(cache, cache_boundary, generated_ids)
            if saved_offset is None:
                return False
            existing = self._index.get(chain_hash)
            if existing is not None and existing.n_tokens >= n_prompt and (existing.image_hashes == img_hashes):
                existing.last_used = time.time()
                self._touch_hot(chain_hash, cache)
                return True
            kvq = kv_quant_flags(cache)
            blob = self._disk_path(chain_hash)
            meta_obj = {'chain_hash': chain_hash, 'model_key': self.model_key, 'token_ids': [int(t) for t in prompt_token_ids], 'n_tokens': n_prompt, 'chunk_hashes': chash_list, 'kv_quant': json.dumps(kvq, sort_keys=True), 'image_hashes': img_hashes, 'image_positions': img_positions, 'bytes': 0, 'last_used': time.time(), 'created': time.time()}
            try:
                from mlx_lm.models.cache import save_prompt_cache
            except Exception as e:
                self._log(f'[prompt-cache] save import failed: {e}\n')
                return False
            st_meta = {'n_tokens': str(n_prompt), 'kv_quant': meta_obj['kv_quant'], 'model_key': self.model_key}
            try:
                from mlx.utils import tree_flatten as _tree_flatten
                _flat = dict(_tree_flatten([c.state for c in cache]))
                _empties = {k: [list(v.shape), str(v.dtype).rsplit('.', 1)[-1]] for k, v in _flat.items() if hasattr(v, 'shape') and 0 in tuple(v.shape)}
            except Exception as _ee:
                self._log(f'[prompt-cache] empties precompute failed: {_ee}\n')
                _empties = {}
            if _empties:
                st_meta = {**st_meta, 'empty_arrays': json.dumps(_empties)}

            def _write_blob(tmp: str) -> str:
                import mlx.core as _mx
                _orig = _mx.save_safetensors

                def _patched(file, arrays, metadata=None):
                    fixed = {k: v for k, v in arrays.items() if not (hasattr(v, 'shape') and 0 in tuple(v.shape))}
                    return _orig(file, fixed, metadata if metadata is not None else {})
                _mx.save_safetensors = _patched
                try:
                    save_prompt_cache(tmp, cache, st_meta)
                finally:
                    _mx.save_safetensors = _orig
                return tmp if os.path.exists(tmp) else tmp + '.safetensors'
            try:
                self._atomic_write_bytes(blob, _write_blob, suffix='.safetensors')
            except Exception as e:
                self._log(f'[prompt-cache] disk persist failed; keeping RAM-only entry: {e}\n')
                try:
                    if blob.exists():
                        os.remove(blob)
                except Exception:
                    pass
                ent = _Entry(chain_hash=chain_hash, token_ids=meta_obj['token_ids'], chunk_hashes_list=chash_list, kv_quant=kvq, nbytes=0, last_used=meta_obj['last_used'], disk_path='', meta_path='', image_hashes=img_hashes, image_positions=img_positions)
                self._index[chain_hash] = ent
                self._touch_hot(chain_hash, cache)
                self.stats['saves'] += 1
                return True
            nbytes = blob.stat().st_size
            meta_obj['bytes'] = nbytes
            meta_file = self._meta_path(chain_hash)

            def _write_meta(tmp: str) -> str:
                Path(tmp).write_text(json.dumps(meta_obj, separators=(',', ':')))
                return tmp
            try:
                self._atomic_write_bytes(meta_file, _write_meta)
            except Exception as e:
                self._log(f'[prompt-cache] meta write failed: {e}\n')
                try:
                    os.remove(blob)
                except Exception:
                    pass
                return False
            ent = _Entry(chain_hash=chain_hash, token_ids=meta_obj['token_ids'], chunk_hashes_list=chash_list, kv_quant=kvq, nbytes=nbytes, last_used=meta_obj['last_used'], disk_path=str(blob), meta_path=str(meta_file), image_hashes=img_hashes, image_positions=img_positions)
            self._index[chain_hash] = ent
            self._touch_hot(chain_hash, cache)
            self.stats['saves'] += 1
            self._evict_disk_if_needed(protect=chain_hash)
            self._log(f'[prompt-cache] SAVED prefix: {n_prompt} tok' + (f' + {len(img_hashes)} img' if img_hashes else '') + f' -> {nbytes / 1024 ** 2:.2f} MB ({chain_hash})\n')
            return True

    def _save_prompt_pure(self, cache: List[Any], cache_boundary: int, generated_ids: Optional[List[int]]) -> Optional[int]:
        cur = _cache_offset(cache)
        if cur <= cache_boundary:
            return cur
        try:
            from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache
        except Exception:
            return None
        if not can_trim_prompt_cache(cache):
            self._log('[prompt-cache] save skipped: cache covers decoded tail and is not trimmable to the prompt boundary\n')
            return None
        trim_prompt_cache(cache, cur - cache_boundary)
        return _cache_offset(cache)

    def stats_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            image_entries = sum((1 for e in self._index.values() if e.image_hashes))
            return {**self.stats, 'enabled': self.enabled, 'model_key': self.model_key, 'hot_entries': len(self._hot), 'disk_entries': len(self._index), 'image_entries': image_entries, 'disk_bytes': self._current_disk_bytes() if self.enabled else 0}

    def clear(self) -> None:
        with self._lock:
            for ch in list(self._index.keys()):
                self._drop_entry(ch)
            self._hot.clear()

def _lcp(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

def _cache_offset(cache: List[Any]) -> int:
    for c in cache:
        off = getattr(c, 'offset', None)
        if off is not None:
            return int(off)
        st = getattr(c, 'state', None)
        stack, shapes = ([st], [])
        while stack:
            x = stack.pop()
            if isinstance(x, (list, tuple)):
                stack.extend(x)
            elif hasattr(x, 'shape'):
                shapes.append(tuple(x.shape))
        for sh in shapes:
            if len(sh) >= 2 and sh[-2] and (int(sh[-2]) > 0):
                return int(sh[-2])
    return 0

def _image_cell_total(image_positions: List[List[int]]) -> int:
    total = 0
    for p in image_positions or ():
        if len(p) >= 2:
            total += int(p[1])
    return total

def _flags_compatible(disk: Dict[str, Any], live: Dict[str, Any]) -> bool:
    if not disk:
        return True
    if int(disk.get('n_layers', -1)) != int(live.get('n_layers', -2)):
        return False
    if disk.get('classes') != live.get('classes'):
        return False
    if disk.get('bits') != live.get('bits'):
        return False
    if disk.get('group_sizes') != live.get('group_sizes'):
        return False
    return True
